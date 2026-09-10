"""Test the regime-filtered VWAP/EMA gold strategy of Bhatti (2026).

The paper (SSRN 6650958) specifies six entry conditions on XAU/USD 15-minute
candles, a 0.5-ATR stop, a 3R target and a close-conditioned 50 EMA trailing
stop, and reports +0.414R expectancy, a 45.3% win rate, a profit factor of 1.76
and an annualised Sharpe of 3.99 over 247 trades in 2024.

Its Section 6.1 also says those 247 trades were *"parameterised from the
strategy's structural logic"* - full wins with probability 0.30, partial wins
0.20, breakevens 0.08, losses 0.42. That is a Monte Carlo draw from an assumed
outcome distribution, so there is no measurement of gold to replicate. What is
fully specified, and what this script runs, is the rule set itself.

Eight passes, ordered so the claim fails as early as it is going to:

1. **Structure.** How often the six conditions fire, what 1R is worth, and what
   the round turn costs as a fraction of it.
2. **Condition attrition.** How many bars survive each of C1-C6, alone and
   cumulatively - because a six-condition AND can be carried by one condition
   and decorated by five, and no aggregate will say which.
3. **The rule exactly as written**, against the paper's own Tables 3 and 4.
4. **The exit surface.** Six stop widths x six targets x trail on/off x session
   flat on/off. The paper fixes one cell in advance, so where that cell sits on
   the surface is evidence - and the best cell has to be read against the number
   of cells searched.
5. **Do the entry conditions do any work?** C4 split into its two halves and
   removed entirely, the two directions separated, and a placebo that keeps the
   session, the count and the geometry while randomising *when* to enter.
6. **The VWAP anchor.** 13:30 UTC as the paper writes it, against 09:30 New York
   as "the New York session open" actually means - they differ for five months
   of the year.
7. **Hours.** The stand-in for the news filter this corpus cannot implement.
8. **Breadth.** The same rules on USTEC and two FX majors.

Run:  python scripts/backtests/backtest_vwap_ema.py
      python scripts/backtests/backtest_vwap_ema.py --placebo-draws 50
      python scripts/backtests/backtest_vwap_ema.py --splits test    # locked
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import deflated_threshold  # noqa: E402
from qlab.metrics import TRADING_DAYS  # noqa: E402
from qlab.stats import bootstrap_ci, newey_west  # noqa: E402
from qlab.strategies.vwap_ema import (  # noqa: E402
    STOP_MULTS,
    TARGET_RS,
    VWAPEmaConfig,
    condition_attrition,
    exit_reason_table,
    exit_surface,
    hour_breakdown,
    indicators,
    outcome_table,
    placebo_signals,
    ny_anchored,
    run,
    signals,
    with_config,
)

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "vwap_ema_trades"
PRIMARY = "XAUUSD"
OTHERS = ("USTEC", "EURUSD", "USDJPY")
RISK_PER_TRADE = 0.01     # the paper's kappa, used only to turn R into equity

#: The paper's own headline numbers, for a side-by-side that does not require
#: flipping back to the PDF. From its Tables 3 and 4.
PAPER = {
    "trades": 247,
    "win_rate_pct": 45.3,
    "expectancy_r": 0.414,
    "profit_factor": 1.76,
    "sharpe": 3.99,
    "max_dd_pct": 5.1,
    "return_pct": 102.2,
    "avg_win_r": 2.12,
    "avg_loss_r": -1.07,
    "outcomes": {
        "full win": (70, 28.3, 2.77),
        "partial win": (42, 17.0, 1.58),
        "breakeven": (24, 9.7, -0.07),
        "loss": (111, 44.9, -1.01),
    },
}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def summarise(trades: pl.DataFrame, column: str = "net_r", *, n_boot: int = 0) -> dict:
    """Per-trade mean with a Newey-West t, plus the numbers that qualify it.

    The t is HAC rather than plain: signals cluster inside a session and a trend
    that runs for a week puts several winners in a row, so consecutive trades
    are not independent draws.
    """
    if trades.is_empty():
        return {"n": 0}
    x = trades[column].drop_nulls().to_numpy()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"n": int(x.size)}
    nw = newey_west(x)
    wins, losses = x[x > 0], x[x <= 0]
    out = {
        "n": int(x.size),
        "mean_r": float(x.mean()),
        "sd_r": float(x.std(ddof=1)),
        "t_stat": nw.t_stat,
        "p_value": nw.p_value,
        "win_rate_pct": 100.0 * float((x > 0).mean()),
        "avg_win_r": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum())
        if losses.size and losses.sum() < 0 else float("inf"),
        "total_r": float(x.sum()),
    }
    if "mid_pnl" in trades.columns and "risk" in trades.columns:
        mid = (trades["mid_pnl"] / trades["risk"]).drop_nulls().to_numpy()
        out["mid_r"] = float(np.nanmean(mid))
        out["cost_r"] = float(trades["cost_r"].mean())
    if n_boot:
        boot = bootstrap_ci(x, n_boot=n_boot)
        out |= {"boot_lo": boot.lo, "boot_hi": boot.hi, "boot_p": boot.p_value}
    return out


def equity_curve(trades: pl.DataFrame, *, risk: float = RISK_PER_TRADE) -> dict:
    """The paper's Table 3 metrics, from R-multiples at a fixed fractional risk.

    Compounded, because risking 1% of *current* equity is what the paper's Eq. 7
    sizes to. Sharpe is computed on daily equity returns rather than per trade,
    which is the house convention and the reason this number will not match a
    per-trade annualisation like the paper's.
    """
    if trades.is_empty():
        return {}
    day = (
        trades.with_columns(date=pl.col("ts").dt.date())
        .group_by("date").agg(r=pl.col("net_r").sum()).sort("date")
    )
    growth = (1.0 + risk * day["r"]).cum_prod()
    total = float(growth[-1]) - 1.0
    # Days on which the strategy held nothing still belong in the denominator:
    # capital was committed to the strategy and idle.
    spine = pl.date_range(day["date"].min(), day["date"].max(), "1d", eager=True)
    spine = spine.filter(spine.dt.weekday() < 6)
    daily = (
        pl.DataFrame({"date": spine})
        .join(day.with_columns(ret=risk * pl.col("r")), on="date", how="left")
        .with_columns(pl.col("ret").fill_null(0.0))
    )
    ret = daily["ret"].to_numpy()
    curve = np.cumprod(1.0 + ret)
    dd = curve / np.maximum.accumulate(curve) - 1.0
    years = ret.size / TRADING_DAYS
    sd = float(ret.std(ddof=1)) if ret.size > 1 else 0.0
    return {
        "return_pct": 100.0 * total,
        "cagr_pct": 100.0 * ((1.0 + total) ** (1 / years) - 1.0) if years > 0 else 0.0,
        "sharpe": float(ret.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "max_dd_pct": 100.0 * float(dd.min()),
        "days": int(ret.size),
        "years": years,
        "risk_per_trade_pct": 100.0 * risk,
    }


def print_summary(rows: list[tuple[str, dict]], *, indent: str = "  ") -> None:
    width = max((len(name) for name, _ in rows), default=10)
    print(f"{indent}{'':<{width}}  {'n':>5}  {'mean R':>8}  {'t':>6}  "
          f"{'p':>6}  {'hit%':>6}  {'PF':>5}")
    for name, s in rows:
        if not s.get("n"):
            print(f"{indent}{name:<{width}}  {'-':>5}")
            continue
        pf = s.get("profit_factor", float("nan"))
        print(f"{indent}{name:<{width}}  {s['n']:>5}  {s['mean_r']:>+8.3f}  "
              f"{s['t_stat']:>+6.2f}  {s['p_value']:>6.3f}  "
              f"{s['win_rate_pct']:>6.1f}  {pf:>5.2f}")


# --------------------------------------------------------------------------
# The passes
# --------------------------------------------------------------------------

def pass_structure(bars: pl.DataFrame, trades: pl.DataFrame, costs: CostModel,
                   split: str) -> dict:
    print(f"\n{'=' * 74}\n1. STRUCTURE\n{'=' * 74}")
    sess = bars.filter(pl.col("in_session"))
    years = (bars["ts"].max() - bars["ts"].min()).total_seconds() / (365.25 * 86400)
    n_sig = int((bars["signal"] != 0).sum())
    price = float(sess["close"].median())
    rt_pips = costs.round_turn_pips(price=price)
    rt_usd = costs.round_turn_usd(price)
    risk_usd = float(trades["risk"].median()) * 100.0 if not trades.is_empty() else float("nan")

    out = {
        "session_bars": sess.height,
        "signals": n_sig,
        "signals_per_year": n_sig / years if years else float("nan"),
        "signal_rate_pct": 100.0 * n_sig / sess.height if sess.height else float("nan"),
        "median_atr": float(sess["atr"].median()),
        "median_risk_price": float(trades["risk"].median()) if not trades.is_empty() else float("nan"),
        "median_risk_usd_per_lot": risk_usd,
        "round_turn_pips": rt_pips,
        "round_turn_usd_per_lot": rt_usd,
        "cost_as_fraction_of_r": rt_usd / risk_usd if risk_usd else float("nan"),
        "median_price": price,
        "years": years,
    }
    print(f"  session bars ({split})      {out['session_bars']:,}"
          f"   over {years:.2f} years")
    print(f"  signals                    {n_sig:,}  "
          f"({out['signals_per_year']:.0f}/yr, {out['signal_rate_pct']:.2f}% of bars)"
          f"   -- the paper reports 247 in one year")
    print(f"  median ATR(14) on 15m      ${out['median_atr']:.2f}")
    print(f"  median 1R  (0.5 ATR + wick) ${out['median_risk_price']:.2f}"
          f"  = ${risk_usd:.0f} per standard lot")
    print(f"  round turn                 {rt_pips:.1f} pips = ${rt_usd:.0f} per lot")
    print(f"  cost as a fraction of 1R   {out['cost_as_fraction_of_r']:.3f} R"
          f"   -- the paper assumes 0.24 R")
    return out


def pass_attrition(bars: pl.DataFrame, cfg: VWAPEmaConfig) -> dict:
    print(f"\n{'=' * 74}\n2. CONDITION ATTRITION\n{'=' * 74}")
    table = condition_attrition(bars, cfg)
    print("  share of tradeable session bars passing each condition, and the")
    print("  running AND of C1..Ck:\n")
    for side in ("long", "short"):
        rows = table.filter(pl.col("side") == side)
        if rows.is_empty():
            continue
        print(f"  {side}:")
        for r in rows.iter_rows(named=True):
            print(f"    {r['condition']}  alone {r['alone_pct']:>6.2f}%   "
                  f"cumulative {r['cumulative_pct']:>6.3f}%  ({r['cumulative']:,} bars)")
    return {"table": table.to_dicts()}


def pass_as_written(trades: pl.DataFrame, args) -> dict:
    print(f"\n{'=' * 74}\n3. THE RULE EXACTLY AS WRITTEN\n{'=' * 74}")
    stats = summarise(trades, n_boot=args.bootstrap)
    eq = equity_curve(trades)
    reasons = exit_reason_table(trades)
    outcomes = outcome_table(trades)

    print("  0.5 ATR stop, 3R target, close-conditioned 50 EMA trail, 20 EMA")
    print("  final leg, flat at the 20:00 UTC session end.\n")
    print_summary([("as written", stats)])
    if "boot_lo" in stats:
        print(f"\n  bootstrap 95% CI on mean R   "
              f"[{stats['boot_lo']:+.3f}, {stats['boot_hi']:+.3f}]  "
              f"p = {stats['boot_p']:.3f}")
    print(f"  edge at the mid (cost-free)  {stats.get('mid_r', float('nan')):+.3f} R"
          f"    cost {stats.get('cost_r', float('nan')):.3f} R")

    print("\n  against the paper's Table 3:\n")
    print(f"    {'metric':<22} {'paper':>12}  {'measured':>12}")
    rows = [
        ("trades", PAPER["trades"], stats.get("n")),
        ("win rate %", PAPER["win_rate_pct"], stats.get("win_rate_pct")),
        ("expectancy R", PAPER["expectancy_r"], stats.get("mean_r")),
        ("avg win R", PAPER["avg_win_r"], stats.get("avg_win_r")),
        ("avg loss R", PAPER["avg_loss_r"], stats.get("avg_loss_r")),
        ("profit factor", PAPER["profit_factor"], stats.get("profit_factor")),
        ("Sharpe", PAPER["sharpe"], eq.get("sharpe")),
        ("max drawdown %", -PAPER["max_dd_pct"], eq.get("max_dd_pct")),
        ("total return %", PAPER["return_pct"], eq.get("return_pct")),
    ]
    for name, p, m in rows:
        m_s = "-" if m is None else f"{m:+.3f}" if abs(m) < 100 else f"{m:+.1f}"
        print(f"    {name:<22} {p:>12}  {m_s:>12}")

    print("\n  against the paper's Table 4 (how trades end):\n")
    print(f"    {'outcome':<14} {'paper n':>8} {'paper %':>8} {'paper R':>8}"
          f"  |{'meas n':>7} {'meas %':>7} {'meas R':>8}")
    got = {r["outcome"]: r for r in outcomes.iter_rows(named=True)}
    for name, (pn, pp, pr) in PAPER["outcomes"].items():
        g = got.get(name)
        if g:
            print(f"    {name:<14} {pn:>8} {pp:>8.1f} {pr:>+8.2f}  |"
                  f"{g['n']:>7} {g['pct']:>7.1f} {g['mean_r']:>+8.2f}")
        else:
            print(f"    {name:<14} {pn:>8} {pp:>8.1f} {pr:>+8.2f}  |{'-':>7}")

    print("\n  which clock ended the trade:\n")
    for r in reasons.iter_rows(named=True):
        print(f"    {r['exit_reason']:<8} {r['n']:>5}  ({r['pct']:>5.1f}%)  "
              f"mean {r['mean_r']:>+6.3f} R   held {r['mean_hold_min']:>6.0f} min")

    touch = trades["vwap_touch_r"].drop_nulls()
    touch = touch.filter(touch.is_finite())
    print(f"\n  Section 4.3's VWAP milestone fires on {touch.len():,} of "
          f"{trades.height:,} trades ({100.0 * touch.len() / trades.height:.0f}%),")
    print(f"  at a mean floating {float(touch.mean()):+.3f} R when it does - i.e. "
          "it is an adverse\n  event, not a milestone. See the module docstring.")

    return {"stats": stats, "equity": eq, "exit_reasons": reasons.to_dicts(),
            "outcomes": outcomes.to_dicts(),
            "vwap_touch": {"n": touch.len(), "mean_r": float(touch.mean())
                           if touch.len() else float("nan")}}


def pass_exit_surface(symbol: str, bars: pl.DataFrame, cfg: VWAPEmaConfig,
                      costs: CostModel, split: str, args) -> dict:
    print(f"\n{'=' * 74}\n4. THE EXIT SURFACE\n{'=' * 74}")
    print("  mean net R per trade. Rows are the stop as a multiple of ATR(14);")
    print("  columns are the target in R. The paper's cell is stop 0.5, target 3.\n")
    surfaces, cells = {}, []
    for use_trail in (True, False):
        for flat in (True, False):
            variant = f"trail={'on' if use_trail else 'off'}, flat={'on' if flat else 'off'}"
            c = with_config(cfg, use_trail=use_trail, close_at_session_end=flat)
            trades = run(symbol, c, bars=bars, split=split, costs=costs,
                         stop_mults=STOP_MULTS, targets=TARGET_RS)
            surf = exit_surface(trades)
            surfaces[variant] = surf.to_dicts()
            cells.extend(surf.to_dicts())
            print(f"  --- {variant} ---")
            head = "  " + " " * 7 + "".join(
                f"{('hold' if t is None else f'{t:g}R'):>9}" for t in TARGET_RS)
            print(head)
            for mult in STOP_MULTS:
                row = surf.filter(pl.col("stop_mult") == mult)
                vals = {r["target"]: r["mean"] for r in row.iter_rows(named=True)}
                line = f"  {mult:>5.2f}  "
                for t in TARGET_RS:
                    key = "hold" if t is None else f"t{t:g}".replace(".", "_")
                    v = vals.get(key)
                    line += f"{v:>+9.3f}" if v is not None else f"{'-':>9}"
                print(line)
            print()

    n_cells = len(cells)
    best = max(cells, key=lambda c: c["mean"])
    thresh = deflated_threshold(n_cells)
    print(f"  {n_cells} cells searched. Best: stop {best['stop_mult']:g}, "
          f"target {best['target']}, mean {best['mean']:+.3f} R, t = {best['t']:+.2f}.")
    print(f"  A t of {thresh:.2f} is what a single cell would need to survive "
          f"searching {n_cells};")
    print(f"  the best cell {'clears' if best['t'] > thresh else 'does not clear'} it.")
    return {"surfaces": surfaces, "n_cells": n_cells, "best": best,
            "deflated_t_threshold": thresh}


def pass_entry_work(symbol: str, bars: pl.DataFrame, cfg: VWAPEmaConfig,
                    costs: CostModel, split: str, args) -> dict:
    print(f"\n{'=' * 74}\n5. DO THE ENTRY CONDITIONS DO ANY WORK?\n{'=' * 74}")
    rows, out = [], {}

    for pattern in ("either", "pin", "engulf", "any"):
        c = with_config(cfg, entry_pattern=pattern)
        b = signals(bars, c)
        trades = run(symbol, c, bars=b, split=split, costs=costs)
        label = {"either": "C4 as written", "pin": "C4 = pin only",
                 "engulf": "C4 = engulf only", "any": "C4 removed"}[pattern]
        s = summarise(trades)
        rows.append((label, s))
        out[f"pattern:{pattern}"] = s

    base = run(symbol, cfg, bars=bars, split=split, costs=costs)
    for d, name in ((1, "longs only"), (-1, "shorts only")):
        s = summarise(base.filter(pl.col("direction") == d))
        rows.append((name, s))
        out[name] = s
    print_summary(rows)

    print(f"\n  placebo: {args.placebo_draws} draws that keep the session window, the")
    print("  trade count, the side mix and the whole exit geometry, and randomise")
    print("  only *when* the entry happens.\n")
    means = []
    for seed in range(args.placebo_draws):
        fake = placebo_signals(bars, seed=seed, cfg=cfg)
        trades = run(symbol, cfg, bars=fake, split=split, costs=costs)
        if not trades.is_empty():
            means.append(float(trades["net_r"].mean()))
    real = float(base["net_r"].mean())
    if means:
        arr = np.array(means)
        pct = 100.0 * float((arr < real).mean())
        print(f"    placebo mean R   {arr.mean():+.3f}  "
              f"(sd {arr.std(ddof=1):.3f}, range {arr.min():+.3f} to {arr.max():+.3f})")
        print(f"    the rule         {real:+.3f}  -> {pct:.0f}th percentile "
              f"of the placebo distribution")
        out["placebo"] = {"n_draws": len(means), "mean": float(arr.mean()),
                          "sd": float(arr.std(ddof=1)), "min": float(arr.min()),
                          "max": float(arr.max()), "real": real, "percentile": pct}
    return out


def pass_anchor(symbol: str, cfg: VWAPEmaConfig, costs: CostModel, split: str,
                allow_test: bool) -> dict:
    print(f"\n{'=' * 74}\n6. THE VWAP ANCHOR\n{'=' * 74}")
    print("  The paper writes 'the New York session open (13:30 UTC)'. Those are")
    print("  two different anchors for the five months the US is not on daylight")
    print("  saving time. Neither reading is privileged, so both are run.\n")
    rows, out = [], {}
    for anchor in ("utc", "ny"):
        # `ny_anchored` moves the session bounds onto the New York clock as
        # well as the anchor; switching the anchor alone would ask for
        # 13:30-20:00 New York, which is a different six hours of the day.
        c = cfg if anchor == "utc" else ny_anchored(cfg)
        b = signals(indicators(symbol, c, split=split, allow_test=allow_test), c)
        trades = run(symbol, c, bars=b, split=split, costs=costs)
        label = "13:30 UTC fixed" if anchor == "utc" else "09:30 New York"
        s = summarise(trades)
        rows.append((label, s))
        out[anchor] = s
    print_summary(rows)
    return out


def pass_hours(trades: pl.DataFrame) -> dict:
    print(f"\n{'=' * 74}\n7. HOURS - THE NEWS FILTER THIS CORPUS CANNOT IMPLEMENT\n{'=' * 74}")
    print("  The paper excludes entries within 15 minutes of NFP, CPI, FOMC and")
    print("  GDP. There is no macro calendar here, so this is the substitute: if")
    print("  excluding news would have mattered, the 13:00-14:00 and 18:00-19:00")
    print("  UTC buckets have to be the ones dragging.\n")
    table = hour_breakdown(trades)
    print(f"    {'hour UTC':>9}  {'n':>5}  {'mean R':>8}  {'t':>6}  {'hit%':>6}")
    for r in table.iter_rows(named=True):
        # A bucket holding one trade has no dispersion and therefore no t. That
        # is information, not an error, so it prints as a dash.
        t = f"{r['t']:>+6.2f}" if r["t"] is not None else f"{'-':>6}"
        print(f"    {r['hour']:>7}:00  {r['n']:>5}  {r['mean']:>+8.3f}  "
              f"{t}  {r['hit_pct']:>6.1f}")
    return {"table": table.to_dicts()}


def pass_breadth(cfg: VWAPEmaConfig, split: str, allow_test: bool,
                 primary: dict) -> dict:
    print(f"\n{'=' * 74}\n8. BREADTH\n{'=' * 74}")
    print("  The same six conditions on three instruments the paper never claims.")
    print("  Gold is the claim; the rest say whether the rules describe gold or")
    print("  describe a 15-minute chart.\n")
    rows = [(f"{PRIMARY} (the claim)", primary)]
    out = {PRIMARY: primary}
    for symbol in OTHERS:
        try:
            costs = CostModel.from_profiles(symbol, split=split)
            b = signals(indicators(symbol, cfg, split=split, allow_test=allow_test), cfg)
            trades = run(symbol, cfg, bars=b, split=split, costs=costs)
            s = summarise(trades)
            rows.append((symbol, s))
            out[symbol] = s
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {symbol}: skipped ({exc})")
    print_summary(rows)
    return out


# --------------------------------------------------------------------------

def run_split(split: str, args, *, allow_test: bool) -> dict:
    print(f"\n\n{'#' * 74}\n#  {PRIMARY}  --  {split.upper()} SPLIT\n{'#' * 74}")
    cfg = VWAPEmaConfig()
    costs = CostModel.from_profiles(PRIMARY, split=split)
    bars = signals(indicators(PRIMARY, cfg, split=split, allow_test=allow_test), cfg)
    trades = run(PRIMARY, cfg, bars=bars, split=split, costs=costs)
    if trades.is_empty():
        print("  no trades")
        return {"trades": 0}

    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    trades.write_parquet(TAPE_DIR / f"{PRIMARY}_{split}.parquet")

    result = {
        "structure": pass_structure(bars, trades, costs, split),
        "attrition": pass_attrition(bars, cfg),
        "as_written": pass_as_written(trades, args),
    }
    result["exit_surface"] = pass_exit_surface(PRIMARY, bars, cfg, costs, split, args)
    result["entry_work"] = pass_entry_work(PRIMARY, bars, cfg, costs, split, args)
    result["anchor"] = pass_anchor(PRIMARY, cfg, costs, split, allow_test)
    result["hours"] = pass_hours(trades)
    result["breadth"] = pass_breadth(cfg, split, allow_test,
                                     result["as_written"]["stats"])
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--placebo-draws", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("-o", "--out", default=str(REPORT_DIR / "vwap_ema.json"))
    args = p.parse_args(argv)

    if "test" in args.splits:
        print("!! the test split is locked; it is being read deliberately\n")

    results = {"primary": PRIMARY, "paper": PAPER,
               "config": VWAPEmaConfig().__dict__, "splits": {}}
    for split in args.splits:
        results["splits"][split] = run_split(
            split, args, allow_test=(split == "test"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
