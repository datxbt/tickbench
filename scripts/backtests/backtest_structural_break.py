#!/usr/bin/env python
"""Structural-break entries on gold: the controlled-variable harness (B.5-B.7).

One fixed core - swing-break entry, pending stop order, 3-dollar stop,
20-dollar target - and four axes moved one at a time:

  entry      Axis 1: the break against buy-and-hold, an MA(20,50) crossover,
             and a random-entry Monte Carlo with identical stops and targets
  exit       Axis 2: half-exit on (V10) against off (B5), and the candle filter
             on against off (V11_FixedR)
  sizing     Axis 3: V10 against V11 and its three sub-variants, at 1x and 2x
  modes      Axis 4: real ticks against interpolated ticks against minute closes
  walk       non-overlapping six-month windows
  sweep      one-factor-at-a-time parameter sensitivity
  validation the surviving cells on the validation split

Every headline comparison is a *paired* one on the common daily return grid,
because two exit rules on the same signal share most of their days and an
unpaired test would compare them through a variance those shared days have
already cancelled.

The test split is not read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.metrics import format_tearsheet, tearsheet  # noqa: E402
from qlab.stats import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    ljung_box,
    paired_bootstrap,
    sharpe,
    sharpe_with_se,
    two_proportion_z,
)
from qlab.strategies import structural_break as sb  # noqa: E402

SECTIONS = ("entry", "exit", "sizing", "modes", "walk", "sweep", "validation")
TRADING_DAYS = 252


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Daily returns on a common grid
# --------------------------------------------------------------------------

def _daily_pnl(out: sb.RunOutput) -> pl.DataFrame:
    if out.trades.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "pnl": pl.Float64})
    return (
        out.trades.with_columns(
            date=pl.from_epoch("exit_ts", time_unit="us").dt.date())
        .group_by("date").agg(pnl=pl.col("net_usd").sum()).sort("date")
    )


def aligned_returns(runs: dict[str, sb.RunOutput]) -> tuple[list, dict[str, np.ndarray]]:
    """Daily returns for several runs on one shared weekday spine.

    A shared spine is the whole point: a paired bootstrap needs the same dates
    on both sides, and per-run spines silently differ whenever one variant's
    first or last trade lands on a different day.
    """
    frames = {k: _daily_pnl(v) for k, v in runs.items()}
    dates = [f["date"] for f in frames.values() if not f.is_empty()]
    if not dates:
        return [], {k: np.array([]) for k in runs}
    lo = min(d.min() for d in dates)
    hi = max(d.max() for d in dates)
    spine = pl.DataFrame({
        "date": pl.date_range(lo, hi, "1d", eager=True)
    }).filter(pl.col("date").dt.weekday() < 6)

    out: dict[str, np.ndarray] = {}
    for key, frame in frames.items():
        joined = (spine.join(frame, on="date", how="left")
                  .with_columns(pl.col("pnl").fill_null(0.0)))
        equity = runs[key].starting_equity + joined["pnl"].cum_sum()
        prev = pl.concat([pl.Series([runs[key].starting_equity]), equity[:-1]])
        out[key] = (equity / prev - 1.0).to_numpy()
    return spine["date"].to_list(), out


def headline(out: sb.RunOutput, returns: np.ndarray) -> dict:
    """The B.7 reporting template for one run."""
    sheet = tearsheet(out.trades, starting_equity=out.starting_equity,
                      label=out.label)
    if sheet.get("trades", 0) == 0:
        return sheet
    trades = out.trades
    wins = trades.filter(pl.col("net_usd") > 0)
    losses = trades.filter(pl.col("net_usd") <= 0)
    fired = trades.filter(pl.col("half_fired") == 1) if "half_fired" in trades.columns \
        else trades.head(0)
    sheet["sharpe_daily"] = sharpe(returns, TRADING_DAYS)
    sheet["payoff"] = (abs(sheet["avg_win_usd"] / sheet["avg_loss_usd"])
                       if sheet["avg_loss_usd"] else float("nan"))
    sheet["half_rate_winners"] = (
        100.0 * fired.filter(pl.col("net_usd") > 0).height / max(wins.height, 1))
    sheet["half_rate_losers"] = (
        100.0 * fired.filter(pl.col("net_usd") <= 0).height / max(losses.height, 1))
    sheet["mean_r"] = float(trades["r_multiple"].mean())
    return sheet


def row(name: str, out: sb.RunOutput, returns: np.ndarray) -> str:
    s = headline(out, returns)
    if s.get("trades", 0) == 0:
        return f"  {name:<22} no trades"
    return (f"  {name:<22} {s['trades']:>5d} {s['return_pct']:>+9.1f} "
            f"{s['sharpe_daily']:>+7.2f} {s['sortino']:>+7.2f} "
            f"{s['calmar']:>+7.2f} {s['max_dd_pct']:>7.1f} "
            f"{s['win_rate_pct']:>7.1f} {s['payoff']:>7.2f} "
            f"{s['profit_factor']:>7.2f} {s['mean_r']:>+7.3f}")


HEAD = (f"  {'run':<22} {'n':>5} {'return%':>9} {'Sharpe':>7} {'Sortino':>7} "
        f"{'Calmar':>7} {'maxDD%':>7} {'win%':>7} {'payoff':>7} {'PF':>7} {'meanR':>7}")


def table(runs: dict[str, sb.RunOutput]) -> dict[str, np.ndarray]:
    _, rets = aligned_returns(runs)
    print(HEAD)
    print("  " + "-" * (len(HEAD) - 2))
    for name, out in runs.items():
        print(row(name, out, rets[name]))
    return rets


# --------------------------------------------------------------------------
# Axis 1 - entry validity
# --------------------------------------------------------------------------

def section_entry(ctx, split: str, paths: int, seed: int, mode: str,
                  exits: sb.ExitConfig | None = None) -> None:
    exits = exits or sb.ExitConfig()
    rule(f"Axis 1. Does the entry carry anything? (B.5)   exits: {exits.label}")
    print("  Every row below has the identical stop, target, sizing rule and")
    print("  exit overlay. Only the instant and the side of the entry differ, so")
    print("  a difference between rows is a statement about the entry alone.")
    print("  Worth running twice. With the overlay on, both the strategy and its")
    print("  null are dominated by the overlay's own arithmetic - the comparison")
    print("  is still fair, but the level is uninformative. With it off")
    print("  (--entry-exits b5) the entry is the only thing left, and that is the")
    print("  real test of whether the structural break carries anything.\n")

    runs = {
        "break (on_confirm)": sb.run(ctx=ctx, split=split, mode=mode,
                                     exits=exits, label="break on_confirm"),
        "MA(20,50) crossover": sb.run(ctx=ctx, split=split, mode=mode,
                                      exits=exits,
                                      entries=sb.ma_crossover_entries(ctx),
                                      label="MA crossover"),
    }
    retest_cfg = sb.BreakConfig(arm="retest")
    retest_ctx = sb.build_context(split=split, cfg=retest_cfg)
    runs["break (retest)"] = sb.run(ctx=retest_ctx, cfg=retest_cfg, split=split,
                                    mode=mode, exits=exits, label="break retest")
    rets = table(runs)

    bh = sb.buy_and_hold(ctx)
    bh_sharpe = sharpe_with_se(bh["returns"])
    print(f"\n  buy and hold gold, same window: total "
          f"{100 * (np.prod(1 + bh['returns']) - 1):+.1f}%, "
          f"Sharpe {bh_sharpe.mean:+.2f} (t {bh_sharpe.t_stat:+.2f})")
    print("  Quoted as a return series, not as trades: it has no round trips and")
    print("  inventing one would put it in a table it does not belong in.")

    print(f"\n  --- Random-entry Monte Carlo, {paths} paths, mode '{'interp'}' ---")
    print("  Same trade count, same clock, same stop and target; the timing and")
    print("  the side are random. This is the null distribution the entry has to")
    print("  beat, and it is run on the interpolated tape because a thousand")
    print("  tick-mode paths would take the better part of an hour.")
    tape = sb.synthetic_tape(ctx.bars_1m, "interp")
    real = sb.run(ctx=ctx, split=split, mode="interp", tape=tape, exits=exits,
                  label="break interp")
    n_trades = real.trades.height
    t0 = time.time()
    draws = []
    for i in range(paths):
        out = sb.run(ctx=ctx, split=split, mode="interp", tape=tape, exits=exits,
                     entries=sb.random_entries(ctx, n_trades, seed + i),
                     label=f"random {i}")
        if out.trades.is_empty():
            continue
        draws.append((float(out.trades["r_multiple"].mean()),
                      float(out.equity / out.starting_equity - 1.0)))
    draws = np.array(draws)
    elapsed = time.time() - t0
    real_r = float(real.trades["r_multiple"].mean())
    real_total = float(real.equity / real.starting_equity - 1.0)
    pct_r = float((draws[:, 0] >= real_r).mean())
    pct_ret = float((draws[:, 1] >= real_total).mean())
    print(f"\n  {paths} paths in {elapsed:.0f}s")
    print(f"  {'statistic':<16} {'strategy':>10} {'null mean':>10} "
          f"{'null sd':>9} {'null p05':>9} {'null p95':>9} {'pctile':>7}")
    print(f"  {'mean R/trade':<16} {real_r:>+10.4f} {draws[:, 0].mean():>+10.4f} "
          f"{draws[:, 0].std(ddof=1):>9.4f} {np.quantile(draws[:, 0], .05):>+9.4f} "
          f"{np.quantile(draws[:, 0], .95):>+9.4f} {1 - pct_r:>7.3f}")
    print(f"  {'total return':<16} {real_total:>+10.4f} {draws[:, 1].mean():>+10.4f} "
          f"{draws[:, 1].std(ddof=1):>9.4f} {np.quantile(draws[:, 1], .05):>+9.4f} "
          f"{np.quantile(draws[:, 1], .95):>+9.4f} {1 - pct_ret:>7.3f}")
    print(f"\n  One-sided p against the random-entry null: mean R {pct_r:.3f}, "
          f"total return {pct_ret:.3f}")

    print("\n  --- Family A multiplicity: three comparisons, corrected together ---")
    base = rets["break (on_confirm)"]
    tests = [
        ("vs MA crossover", paired_bootstrap(base, rets["MA(20,50) crossover"],
                                             lambda x: sharpe(x, TRADING_DAYS),
                                             n_boot=500, rng=np.random.default_rng(seed))),
        ("vs break retest", paired_bootstrap(base, rets["break (retest)"],
                                             lambda x: sharpe(x, TRADING_DAYS),
                                             n_boot=500, rng=np.random.default_rng(seed))),
    ]
    raw = [t[1].p_value for t in tests] + [pct_r]
    names = [t[0] for t in tests] + ["vs random entry"]
    bonf, bh_adj = bonferroni(raw), benjamini_hochberg(raw)
    print(f"  {'comparison':<20} {'dSharpe':>9} {'p':>8} {'Bonferroni':>11} {'BH':>8}")
    for i, name in enumerate(names):
        delta = f"{tests[i][1].point:>+9.2f}" if i < len(tests) else f"{'-':>9}"
        print(f"  {name:<20} {delta} {raw[i]:>8.4f} {bonf[i]:>11.4f} {bh_adj[i]:>8.4f}")


# --------------------------------------------------------------------------
# Axis 2 - the exit overlay
# --------------------------------------------------------------------------

def section_exit(ctx, split: str, mode: str, seed: int, n_boot: int) -> None:
    rule("Axis 2. What the half-exit overlay does (B.5)")
    variants = {
        "V10 half-exit on": sb.ExitConfig(),
        "B5 half-exit off": sb.ExitConfig(half_exit=False),
        "V11_FixedR no filter": sb.ExitConfig(require_counter_candle=False),
    }
    runs = {name: sb.run(ctx=ctx, split=split, mode=mode, exits=cfg, label=name)
            for name, cfg in variants.items()}
    rets = table(runs)

    print("\n  Half-exit fire rate, winners against losers - the disposition")
    print("  effect's own signature is a rule that fires on winners:\n")
    for name, out in runs.items():
        s = headline(out, rets[name])
        if s.get("trades", 0) == 0:
            continue
        print(f"  {name:<22} winners {s['half_rate_winners']:>5.1f}%   "
              f"losers {s['half_rate_losers']:>5.1f}%   "
              f"fired {out.meta['half_fired']:>4d}")

    print("\n  Trade counts differ between rows because a position that is not")
    print("  half-closed lives longer, and only one position is open at a time.")
    print("  That is why every test below is on the daily return grid rather")
    print("  than on the trade list: the days are common, the trades are not.\n")

    rng = np.random.default_rng(seed)
    base = rets["V10 half-exit on"]
    stats = [
        ("dSharpe", lambda x: sharpe(x, TRADING_DAYS)),
        ("dCalmar", _calmar),
        ("dMaxDD", _max_dd),
    ]
    family = []
    print(f"  {'comparison':<26} {'statistic':<9} {'point':>9} {'95% CI':>22} {'p':>8}")
    for other in ("B5 half-exit off", "V11_FixedR no filter"):
        for label, fn in stats:
            res = paired_bootstrap(base, rets[other], fn, n_boot=n_boot, rng=rng)
            family.append((f"{other} {label}", res))
            ci = f"[{res.lo:+.3f}, {res.hi:+.3f}]"
            print(f"  V10 vs {other:<19} {label:<9} {res.point:>+9.3f} "
                  f"{ci:>22} {res.p_value:>8.4f}")

    print("\n  Win rate, two-proportion z (V10 against each):")
    for other in ("B5 half-exit off", "V11_FixedR no filter"):
        a, b = runs["V10 half-exit on"].trades, runs[other].trades
        z = two_proportion_z(int((a["net_usd"] > 0).sum()), a.height,
                             int((b["net_usd"] > 0).sum()), b.height)
        print(f"    V10 vs {other:<22} {z}")

    print("\n  Family B multiplicity, corrected together:")
    raw = np.array([r.p_value for _, r in family])
    bonf, bh_adj = bonferroni(raw), benjamini_hochberg(raw)
    print(f"  {'test':<34} {'p':>8} {'Bonferroni':>11} {'BH':>8}")
    for (name, _), p, b, q in zip(family, raw, bonf, bh_adj):
        print(f"  {name:<34} {p:>8.4f} {b:>11.4f} {q:>8.4f}")

    print("\n  Residual serial dependence in the daily returns:")
    for name in runs:
        print(f"    {name:<22} {ljung_box(rets[name], 10)}")


def _calmar(returns: np.ndarray) -> float:
    growth = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    years = returns.size / TRADING_DAYS
    if years <= 0 or growth[-1] <= 0:
        return float("nan")
    cagr = growth[-1] ** (1 / years) - 1.0
    dd = _max_dd(returns)
    return cagr / abs(dd) if dd < 0 else float("nan")


def _max_dd(returns: np.ndarray) -> float:
    growth = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(growth)
    return float(np.min((growth - peak) / peak))


# --------------------------------------------------------------------------
# Axis 3 - sizing
# --------------------------------------------------------------------------

def section_sizing(ctx, split: str, mode: str, seed: int, n_boot: int) -> None:
    rule("Axis 3. Does regime-conditional sizing add anything? (B.5)")
    print("  The specification describes V11's two multipliers only as")
    print("  'calibrated lookups'. A calibration cannot be transcribed, so")
    print("  concrete forms were chosen and are stated in the module: the ATR")
    print("  term is 1/rho and the trend term is a three-state multiplier. A")
    print("  different calibration would give different numbers; what it could")
    print("  not change is the invariance below.\n")

    rules = ("V10", "V11", "V11_ATR", "V11_TREND", "V11_VOLTGT")
    runs: dict[str, sb.RunOutput] = {}
    for name in rules:
        for mult, tag in ((1.0, "1x"), (2.0, "2x")):
            key = f"{name} {tag}"
            runs[key] = sb.run(ctx=ctx, split=split, mode=mode,
                               sizing=sb.SizingConfig(rule=name),
                               risk_multiple=mult, label=key)
    rets = table(runs)

    print("\n  --- The invariance that has to hold ---")
    base = rets["V10 1x"]
    s1, s2 = sb.sharpe_invariance_check(base, 3.7)
    print(f"  Sharpe(r) = {s1:+.6f};  Sharpe(3.7 r) = {s2:+.6f};  "
          f"difference {abs(s1 - s2):.2e}")
    print("  A rule that only rescales every position cannot move a Sharpe")
    print("  ratio. So a sizing rule whose return doubles while its Sharpe does")
    print("  not has added leverage, and a rule that claims to add alpha has to")
    print("  show it in the Sharpe column, not the return column.\n")

    print(f"  {'rule':<14} {'1x Sharpe':>10} {'2x Sharpe':>10} {'2x/1x return':>13}")
    for name in rules:
        a, b = rets[f"{name} 1x"], rets[f"{name} 2x"]
        ra = float(np.prod(1 + a) - 1)
        rb = float(np.prod(1 + b) - 1)
        ratio = rb / ra if ra else float("nan")
        print(f"  {name:<14} {sharpe(a, TRADING_DAYS):>+10.3f} "
              f"{sharpe(b, TRADING_DAYS):>+10.3f} {ratio:>13.2f}")

    print("\n  --- Family C: each rule against V10, paired, at 1x ---")
    rng = np.random.default_rng(seed)
    family = []
    print(f"  {'comparison':<24} {'dSharpe':>9} {'95% CI':>22} {'p':>8}")
    for name in rules[1:]:
        res = paired_bootstrap(rets[f"{name} 1x"], base,
                               lambda x: sharpe(x, TRADING_DAYS),
                               n_boot=n_boot, rng=rng)
        family.append((name, res))
        ci = f"[{res.lo:+.3f}, {res.hi:+.3f}]"
        print(f"  {name + ' vs V10':<24} {res.point:>+9.3f} {ci:>22} "
              f"{res.p_value:>8.4f}")
    raw = np.array([r.p_value for _, r in family])
    bonf, bh_adj = bonferroni(raw), benjamini_hochberg(raw)
    print(f"\n  {'rule':<14} {'p':>8} {'Bonferroni':>11} {'BH':>8}")
    for (name, _), p, b, q in zip(family, raw, bonf, bh_adj):
        print(f"  {name:<14} {p:>8.4f} {b:>11.4f} {q:>8.4f}")


# --------------------------------------------------------------------------
# Axis 4 - backtest resolution
# --------------------------------------------------------------------------

def section_modes(ctx, split: str, seed: int, n_boot: int) -> None:
    rule("Axis 4. How much of the result is the backtest's resolution? (B.4)")
    print("  Three exit conditions - stop, target and the sub-bar overlay - can")
    print("  come due in the same minute, and OHLC cannot say which came first.")
    print("  Real ticks can. Overstatement is (interp - ticks)/|ticks|.\n")

    cells = {
        "V10 half-exit on": sb.ExitConfig(),
        "B5 half-exit off": sb.ExitConfig(half_exit=False),
        "V11_FixedR no filter": sb.ExitConfig(require_counter_candle=False),
    }
    print(f"  {'cell':<22} {'mode':<8} {'n':>5} {'return%':>9} {'Sharpe':>8} "
          f"{'meanR':>8} {'dR vs ticks':>12} {'overstate':>10}")
    print("  " + "-" * 90)
    per_cell: dict[str, dict[str, np.ndarray]] = {}
    for name, exits in cells.items():
        runs = {m: sb.run(ctx=ctx, split=split, mode=m, exits=exits,
                          label=f"{name} {m}") for m in sb.MODES}
        rets = aligned_returns(runs)[1]
        per_cell[name] = rets
        base_r = float(runs["ticks"].trades["r_multiple"].mean())
        for m in sb.MODES:
            out = runs[m]
            s = headline(out, rets[m])
            mean_r = float(out.trades["r_multiple"].mean())
            delta = mean_r - base_r
            if m == "ticks":
                delta_s, over_s = f"{'-':>12}", f"{'-':>10}"
            else:
                delta_s = f"{delta:>+12.4f}"
                # A ratio to a base near zero is not a measurement, so the
                # percentage is suppressed rather than printed as four digits
                # of noise. The absolute difference in R is always shown.
                over_s = (f"{delta / abs(base_r) * 100.0:>+9.1f}%"
                          if abs(base_r) > 0.02 else f"{'n/m':>10}")
            print(f"  {name if m == 'ticks' else '':<22} {m:<8} {s['trades']:>5d} "
                  f"{s['return_pct']:>+9.1f} {s['sharpe_daily']:>+8.2f} "
                  f"{mean_r:>+8.3f} {delta_s} {over_s}")
        print()
    print("  'n/m' where the tick-mode mean R is too close to zero for a ratio")
    print("  to mean anything - read the absolute column there instead.\n")

    print("  --- The difference-in-differences that actually matters ---")
    print("  The candle filter is the component whose value could most easily be")
    print("  an artifact of resolution, because it is the one that shares a")
    print("  minute with the stop and the target. So: does the V10-minus-FixedR")
    print("  gap itself change between the two tapes?\n")
    rng = np.random.default_rng(seed)
    for m in ("ticks", "interp"):
        a = per_cell["V10 half-exit on"][m]
        b = per_cell["V11_FixedR no filter"][m]
        res = paired_bootstrap(a, b, lambda x: sharpe(x, TRADING_DAYS),
                               n_boot=n_boot, rng=rng)
        print(f"    {m:<8} dSharpe(V10 - FixedR) = {res}")


# --------------------------------------------------------------------------
# Walk-forward and sensitivity
# --------------------------------------------------------------------------

def section_walk(ctx, split: str, mode: str) -> None:
    rule("Walk-forward: non-overlapping six-month windows (B.6)")
    runs = {"V10": sb.run(ctx=ctx, split=split, mode=mode, label="V10"),
            "B5": sb.run(ctx=ctx, split=split, mode=mode,
                         exits=sb.ExitConfig(half_exit=False), label="B5")}
    dates, rets = aligned_returns(runs)
    dates = np.array(dates)
    period = np.array([f"{d.year}H{1 if d.month <= 6 else 2}" for d in dates])
    windows = sorted(set(period.tolist()))
    print(f"  {'window':<10} {'n days':>7}" +
          "".join(f"{k + ' Sharpe':>14}{k + ' ret%':>12}" for k in runs))
    print("  " + "-" * (19 + 26 * len(runs)))
    for w in windows:
        mask = period == w
        if mask.sum() < 40:
            continue
        cells = ""
        for key in runs:
            r = rets[key][mask]
            cells += (f"{sharpe(r, TRADING_DAYS):>+14.2f}"
                      f"{100 * (np.prod(1 + r) - 1):>+12.1f}")
        print(f"  {w:<10} {int(mask.sum()):>7d}{cells}")
    print("\n  Stability across windows, V10 / B5:")
    for key in runs:
        per = [sharpe(rets[key][period == w], TRADING_DAYS) for w in windows
               if (period == w).sum() >= 40]
        per = np.array([p for p in per if np.isfinite(p)])
        print(f"    {key:<4} mean {per.mean():+.2f}  sd {per.std(ddof=1):+.2f}  "
              f"positive {int((per > 0).sum())}/{per.size}")


def section_sweep(split: str, mode: str) -> None:
    rule("One-factor-at-a-time parameter sensitivity (B.6)")
    print("  Each row moves one parameter around the specification's value and")
    print("  holds everything else fixed. The band is the realised range, which")
    print("  is the honest width of the claim - not the best cell in it.\n")
    grids = {
        "delta_SL ($)": ("delta_sl", [1.5, 2.0, 3.0, 4.5, 6.0]),
        "delta_TP ($)": ("delta_tp", [10.0, 15.0, 20.0, 30.0, 40.0]),
        "W (swing bars)": ("swing_bars", [5, 8, 10, 14, 20]),
    }
    print(f"  {'parameter':<16} {'value':>8} {'n':>6} {'return%':>9} "
          f"{'Sharpe':>8} {'meanR':>8}")
    for label, (field, values) in grids.items():
        sharpes = []
        for value in values:
            cfg = sb.BreakConfig(**{field: value})
            ctx = sb.build_context(split=split, cfg=cfg)
            out = sb.run(ctx=ctx, cfg=cfg, split=split, mode=mode,
                         label=f"{field}={value}")
            rets = aligned_returns({"x": out})[1]["x"]
            s = headline(out, rets)
            if s.get("trades", 0) == 0:
                print(f"  {label:<16} {value:>8} no trades")
                continue
            sharpes.append(s["sharpe_daily"])
            print(f"  {label if value == values[0] else '':<16} {value:>8} "
                  f"{s['trades']:>6d} {s['return_pct']:>+9.1f} "
                  f"{s['sharpe_daily']:>+8.2f} {s['mean_r']:>+8.3f}")
        if sharpes:
            print(f"  {'':<16} {'band':>8} Sharpe {min(sharpes):+.2f} .. "
                  f"{max(sharpes):+.2f}\n")

    print("  And the half-exit threshold, which is the overlay's own parameter:")
    print(f"\n  {'half-exit R':>12} {'n':>6} {'return%':>9} {'Sharpe':>8} {'meanR':>8}")
    ctx = sb.build_context(split=split)
    for r_trigger in (0.5, 0.75, 1.0, 1.5, 2.0):
        exits = sb.ExitConfig(half_exit_r=r_trigger)
        out = sb.run(ctx=ctx, split=split, mode=mode, exits=exits,
                     label=f"half at {r_trigger}R")
        rets = aligned_returns({"x": out})[1]["x"]
        s = headline(out, rets)
        print(f"  {r_trigger:>12.2f} {s['trades']:>6d} {s['return_pct']:>+9.1f} "
              f"{s['sharpe_daily']:>+8.2f} {s['mean_r']:>+8.3f}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="ticks", choices=sb.MODES,
                    help="fidelity for every section except 'modes'")
    ap.add_argument("--paths", type=int, default=200,
                    help="random-entry Monte Carlo paths; the specification "
                         "asks for 1000, which takes about an hour")
    ap.add_argument("--boot", type=int, default=1000,
                    help="stationary-bootstrap resamples")
    ap.add_argument("--entry-exits", default="v10",
                    choices=("v10", "b5", "fixedr"),
                    help="which exit overlay the entry axis holds fixed; "
                         "'b5' turns the half-exit off, which is the clean "
                         "test of the entry on its own")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--test", action="store_true",
                    help="unlock the held-out split (do not use without a survivor)")
    args = ap.parse_args()
    wanted = set(args.only or SECTIONS)

    if args.test and args.split != "test":
        print("--test only means anything with --split test", file=sys.stderr)
        return 2

    print(f"Structural break on XAUUSD, split={args.split}, mode={args.mode}")
    ctx = sb.build_context(split=args.split, allow_test=args.test)
    print(f"  {ctx.h4.height} H4 bars, {ctx.bars_1m.height:,} M1 bars, "
          f"{ctx.regime.height} daily regime rows")

    if "entry" in wanted:
        entry_exits = {
            "v10": sb.ExitConfig(),
            "b5": sb.ExitConfig(half_exit=False),
            "fixedr": sb.ExitConfig(require_counter_candle=False),
        }[args.entry_exits]
        section_entry(ctx, args.split, args.paths, args.seed, args.mode,
                      exits=entry_exits)
    if "exit" in wanted:
        section_exit(ctx, args.split, args.mode, args.seed, args.boot)
    if "sizing" in wanted:
        section_sizing(ctx, args.split, args.mode, args.seed, args.boot)
    if "modes" in wanted:
        section_modes(ctx, args.split, args.seed, args.boot)
    if "walk" in wanted:
        section_walk(ctx, args.split, args.mode)
    if "sweep" in wanted:
        section_sweep(args.split, args.mode)

    if "validation" in wanted and args.split == "dev":
        rule("The validation split")
        vctx = sb.build_context(split="validation")
        runs = {
            "V10 half-exit on": sb.run(ctx=vctx, split="validation",
                                       mode=args.mode, label="V10"),
            "B5 half-exit off": sb.run(ctx=vctx, split="validation",
                                       mode=args.mode,
                                       exits=sb.ExitConfig(half_exit=False),
                                       label="B5"),
        }
        rets = table(runs)
        print("\n  Full tearsheet, the better of the two:")
        best = max(runs, key=lambda k: sharpe(rets[k], TRADING_DAYS))
        print(format_tearsheet(headline(runs[best], rets[best])))
        print("\n  The test split is deliberately not read.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
