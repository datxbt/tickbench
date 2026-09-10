"""Test the 5-minute opening range breakout of Zarattini, Barbon and Aziz (2024).

The paper's rule: take the first 5 minutes of the US session as a range, let the
sign of that candle pick the side, enter on a stop order at the range edge, put
the protective stop 10% of the 14-day ATR away, and hold to the bell. Their base
version underperforms the index; the version they advertise adds a **Stocks in
Play** filter - only names whose opening-range volume is at least its own 14-day
average, and only the top 20 such names each day - and returns 1,600% over
2016-2023.

This corpus has no equity cross-section, so the script separates the two halves
of that claim and tests each on its own terms. It runs seven passes, ordered so
that the claim fails as early as it is going to:

1. **Structure.** What one unit of risk is, what the round turn costs as a
   fraction of it, and how often the order fills at all. A 10%-ATR stop is small
   enough that this is not a formality.
2. **The rule exactly as written.** Mean R with a t-statistic, the hit rate, the
   split between stop-outs and bells, and the year-by-year breakdown - because
   an intraday momentum strategy measured over 2020 is measured over a crash.
3. **The exit surface.** Six stop widths by seven exits. The paper fixes one
   cell of this grid in advance, so where that cell sits on the surface is
   evidence: at a peak reachable only by tuning, it is a fitted result; on a
   broad plateau, it is a real one.
4. **Does the opening candle's sign do any work?** The paper's directional rule
   against three nulls - take both sides, take the opposite side, and a placebo
   that keeps every piece of the geometry and shuffles only the sign. The
   placebo is run many times, so the paper's number can be read against a
   distribution rather than against one draw.
5. **Stocks in Play.** The relative-volume sort that is the paper's Figure 4,
   and the ``RelVol >= 1`` filter, on the tick-count proxy this feed allows.
6. **The opening range length.** 5, 15, 30 and 60 minutes - the paper's own
   Section 5 comparison.
7. **Breadth.** The same rule at the same New York open on gold and two FX
   majors. The US cash open is a real event for all four instruments, so this
   asks whether what is being measured is the opening range or the index.

Run:  python scripts/backtests/backtest_opening_range.py
      python scripts/backtests/backtest_opening_range.py --placebo-draws 50
      python scripts/backtests/backtest_opening_range.py --splits test    # locked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.stats import bootstrap_ci, newey_west  # noqa: E402
from qlab.strategies.opening_range import (  # noqa: E402
    OR_MINUTES,
    STOP_FRACTIONS,
    TARGET_MULTIPLES,
    ORBConfig,
    daily_context,
    placebo_signal,
    relvol_buckets,
    run,
    tag,
)

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "opening_range_trades"
PRIMARY = "USTEC"
OTHERS = ("XAUUSD", "EURUSD", "USDJPY")
EXITS: tuple[float | None, ...] = (None,) + TARGET_MULTIPLES
RISK_PER_TRADE = 0.01     # only used to turn R into an equity curve


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def summarise(trades: pl.DataFrame, column: str = "net_r") -> dict:
    """Per-trade mean with a Newey-West t, plus the numbers that qualify it.

    The t-statistic is HAC rather than plain, because one trade a day on one
    instrument is a time series and consecutive days are not independent draws -
    a trend that runs for a fortnight puts several winners in a row.
    """
    if trades.is_empty():
        return {"n": 0}
    x = trades[column].drop_nulls().to_numpy()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"n": int(x.size)}
    hac = newey_west(x)
    r = trades["r_multiple"].to_numpy()
    return {
        "n": int(x.size),
        "mean_r": float(x.mean()),
        "sd_r": float(x.std(ddof=1)),
        "t_hac": float(hac.t_stat),
        "t_plain": float(x.mean() / x.std(ddof=1) * np.sqrt(x.size)),
        "hit_pct": 100.0 * float((x > 0).mean()),
        "mean_bps": float(trades["net_bps"].mean()),
        "gross_r": float(trades["mid_r"].mean()),
        "cost_r": float(trades["cost_r"].mean()),
        "mean_win_r": float(x[x > 0].mean()) if (x > 0).any() else float("nan"),
        "mean_loss_r": float(x[x <= 0].mean()) if (x <= 0).any() else float("nan"),
        "worst_r": float(x.min()),
        "best_r": float(x.max()),
        "total_r": float(x.sum()),
        "max_dd_r": max_drawdown_r(x),
        "gross_r_total": float(r.sum()),
    }


def max_drawdown_r(x: np.ndarray) -> float:
    """Peak-to-trough of the cumulative R curve, in R."""
    curve = np.concatenate([[0.0], np.cumsum(x)])
    return float(np.max(np.maximum.accumulate(curve) - curve))


def compounded(trades: pl.DataFrame, *, risk: float = RISK_PER_TRADE) -> dict:
    """What the R curve does to an account risking a fixed fraction per trade.

    This is the paper's own sizing rule - a stop-out costs 1% of capital - and
    it is the only place in this script where the answer depends on leverage.
    """
    if trades.is_empty():
        return {}
    daily = (trades.group_by("nyd").agg(r=pl.col("net_r").sum())
             .sort("nyd")["r"].to_numpy())
    growth = np.cumprod(1.0 + risk * daily)
    peak = np.maximum.accumulate(growth)
    years = daily.size / 252.0
    return {
        "days": int(daily.size),
        "total_pct": 100.0 * (growth[-1] - 1.0),
        "cagr_pct": 100.0 * (growth[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0,
        "max_dd_pct": 100.0 * float(np.max(1.0 - growth / peak)),
        "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
        if daily.std(ddof=1) > 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def rule(title: str, width: int = 100) -> None:
    print(f"\n{title}")
    print("-" * width)


def fmt(value, spec=".3f", width=8) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return f"{value:{spec}}".rjust(width)


def print_summary(rows: list[tuple[str, dict]], width: int = 22) -> None:
    print(f"  {'variant':<{width}}  {'n':>5}  {'mean R':>8}  {'t (HAC)':>8}"
          f"  {'hit %':>7}  {'bps':>8}  {'cost R':>7}  {'total R':>8}"
          f"  {'maxDD R':>8}")
    for label, s in rows:
        if not s.get("n"):
            print(f"  {label:<{width}}  {'-':>5}")
            continue
        print(f"  {label:<{width}}  {s['n']:>5}  {fmt(s['mean_r'], '+.3f', 8)}"
              f"  {fmt(s['t_hac'], '+.2f', 8)}  {fmt(s['hit_pct'], '.1f', 7)}"
              f"  {fmt(s['mean_bps'], '+.2f', 8)}  {fmt(s['cost_r'], '.3f', 7)}"
              f"  {fmt(s['total_r'], '+.1f', 8)}  {fmt(s['max_dd_r'], '.1f', 8)}")


def print_surface(trades: pl.DataFrame, column: str, title: str) -> None:
    print(f"\n{title}")
    exits = [tag(t) for t in EXITS]
    header = "".join(f"{('EOD' if e == 'hold' else e.replace('t', '').replace('_', '.') + 'R'):>9}"
                     for e in exits)
    print(f"  {'stop':>6}{header}")
    for stop in sorted(trades["stop_frac"].unique().to_list()):
        cells = []
        for e in exits:
            cell = trades.filter((pl.col("stop_frac") == stop)
                                 & (pl.col("target") == e))
            cells.append(fmt(float(cell[column].mean()) if not cell.is_empty()
                             else None, "+.3f", 9))
        marker = "  <-- paper" if abs(stop - 0.10) < 1e-9 else ""
        print(f"  {stop:>6.2f}{''.join(cells)}{marker}")


# --------------------------------------------------------------------------
# One split
# --------------------------------------------------------------------------

def base_config(**kwargs) -> ORBConfig:
    return ORBConfig(**kwargs)


def run_split(split: str, args, *, allow_test: bool) -> dict:
    result: dict = {"split": split}
    costs = CostModel.from_profiles(PRIMARY, split=split)
    cfg = base_config()
    ctx = daily_context(PRIMARY, cfg, split=split, allow_test=allow_test)
    paper = run(PRIMARY, cfg, ctx=ctx, split=split, costs=costs)
    if paper.is_empty():
        print(f"no trades on {split}")
        return result

    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    paper.write_parquet(TAPE_DIR / f"opening_range_{PRIMARY}_{split}.parquet")

    rule(f"{split.upper()} 1/7 - structure: what the geometry costs before it "
         f"is asked to predict anything")
    fill_rate = 100.0 * paper.height / max(ctx.height, 1)
    print(f"  sessions                      {ctx.height:>8}")
    print(f"  order filled                  {paper.height:>8}   ({fill_rate:.1f}%)")
    print(f"  ATR(14), mean                 {ctx['atr'].mean():>8.1f} points")
    print(f"  1R = 10% ATR, mean            {paper['risk'].mean():>8.1f} points")
    print(f"  opening range width, mean     {ctx['or_width'].mean():>8.1f} points"
          f"   ({ctx['or_width'].mean() / ctx['atr'].mean():.2f} ATR)")
    print(f"  round turn, mean              {paper['cost_r'].mean():>8.3f} R")
    print(f"  minutes from 09:35 to fill    {paper['entry_min'].median():>8.1f} (median)")
    result["structure"] = {
        "sessions": ctx.height, "filled": paper.height, "fill_pct": fill_rate,
        "atr_mean": float(ctx["atr"].mean()), "risk_mean": float(paper["risk"].mean()),
        "or_width_mean": float(ctx["or_width"].mean()),
        "cost_r_mean": float(paper["cost_r"].mean()),
    }

    rule(f"{split.upper()} 2/7 - the rule exactly as the paper writes it")
    stats = summarise(paper)
    print_summary([("5m ORB, 10% ATR, EOD", stats)])
    boot = bootstrap_ci(paper["net_r"].to_numpy(), n_boot=args.bootstrap,
                        rng=np.random.default_rng(0))
    print(f"\n  mean R block-bootstrap 95%: [{boot.lo:+.3f}, {boot.hi:+.3f}]"
          f"   p {boot.p_value:.3f}")
    print(f"  win/loss: {stats['mean_win_r']:+.2f}R on {stats['hit_pct']:.1f}% "
          f"of trades against {stats['mean_loss_r']:+.2f}R on the rest"
          f"   (best {stats['best_r']:+.1f}R)")
    by_reason = (paper.group_by("exit_reason")
                 .agg(n=pl.len(), mean=pl.col("net_r").mean()).sort("exit_reason"))
    for row in by_reason.iter_rows(named=True):
        print(f"  exit by {row['exit_reason']:<6} {row['n']:>4} trades, "
              f"mean {row['mean']:+.3f} R")
    acct = compounded(paper)
    if acct:
        print(f"\n  risking {RISK_PER_TRADE:.0%} of equity per trade: "
              f"{acct['total_pct']:+.1f}% total, {acct['cagr_pct']:+.1f}% a year, "
              f"max drawdown {acct['max_dd_pct']:.1f}%, Sharpe {acct['sharpe']:+.2f}")
    print("\n  by year:")
    yearly = (paper.with_columns(y=pl.col("nyd").dt.year())
              .group_by("y").agg(n=pl.len(), mean=pl.col("net_r").mean(),
                                 sd=pl.col("net_r").std(),
                                 total=pl.col("net_r").sum())
              .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt())
              .sort("y"))
    for row in yearly.iter_rows(named=True):
        print(f"    {row['y']}  n {row['n']:>4}  mean {row['mean']:+.3f} R"
              f"  t {row['t']:+.2f}  total {row['total']:+.1f} R")
    result["paper_rule"] = stats | {
        "bootstrap_lo": boot.lo, "bootstrap_hi": boot.hi,
        "bootstrap_p": boot.p_value, "account": acct,
        "yearly": yearly.to_dicts(),
    }

    rule(f"{split.upper()} 3/7 - the exit surface: is 10% ATR to the bell a "
         f"peak or a plateau?")
    surface = run(PRIMARY, cfg, ctx=ctx, split=split, costs=costs,
                  stop_fractions=STOP_FRACTIONS, targets=EXITS)
    print_surface(surface, "mid_r", "  mean R, mid to mid (cost-free upper bound)")
    print_surface(surface, "net_r", "  mean R, net of spread, commission and slippage")
    print("\n  The paper fixes the 0.10/EOD cell in advance, so its position on "
          "this surface is\n  evidence rather than a choice: 42 cells were "
          "searched here and none of them was.")
    result["surface"] = (
        surface.group_by("stop_frac", "target")
        .agg(n=pl.len(), mean_net_r=pl.col("net_r").mean(),
             mean_mid_r=pl.col("mid_r").mean(),
             sd=pl.col("net_r").std())
        .with_columns(t=pl.col("mean_net_r") / pl.col("sd") * pl.col("n").sqrt())
        .sort("stop_frac", "target").to_dicts()
    )

    rule(f"{split.upper()} 4/7 - does the opening candle's sign do any work?")
    rows = [("paper: sign of candle", stats)]
    variants = {}
    for label, key in (("control: take both sides", "both"),
                       ("contra: opposite side", "contra")):
        alt = run(PRIMARY, base_config(direction_rule=key), ctx=ctx, split=split,
                  costs=costs)
        variants[key] = summarise(alt)
        rows.append((label, variants[key]))
    print_summary(rows)

    draws = []
    for seed in range(args.placebo_draws):
        shuffled = run(PRIMARY, cfg, ctx=placebo_signal(ctx, seed=seed),
                       split=split, costs=costs)
        if not shuffled.is_empty():
            draws.append(float(shuffled["net_r"].mean()))
    if draws:
        arr = np.array(draws)
        beaten = int((arr >= stats["mean_r"]).sum())
        print(f"\n  placebo (same geometry, shuffled sign), {arr.size} draws:"
              f"  mean {arr.mean():+.3f} R,"
              f"  5-95% [{np.percentile(arr, 5):+.3f}, {np.percentile(arr, 95):+.3f}]")
        print(f"  the real signal scores {stats['mean_r']:+.3f} R and is beaten "
              f"by {beaten} of {arr.size} placebos"
              f"  -> empirical p = {(beaten + 1) / (arr.size + 1):.3f}")
        result["placebo"] = {
            "draws": draws, "mean": float(arr.mean()),
            "p_empirical": (beaten + 1) / (arr.size + 1), "beaten": beaten,
        }
    result["direction"] = variants

    rule(f"{split.upper()} 5/7 - Stocks in Play: the relative-volume sort")
    print("  The paper's Figure 4. Their equity cross-section runs to 30x "
          "relative volume;\n  a single index does not, so the top buckets here "
          "are thin and say little.")
    buckets = relvol_buckets(paper)
    print(f"  {'bucket':>10}  {'n':>5}  {'mean R':>8}  {'t':>7}  {'hit %':>7}")
    for row in buckets.iter_rows(named=True):
        print(f"  {str(row['bucket']):>10}  {row['n']:>5}"
              f"  {fmt(row['mean'], '+.3f', 8)}  {fmt(row['t'], '+.2f', 7)}"
              f"  {fmt(row['hit_pct'], '.1f', 7)}")
    filtered = []
    for threshold in (1.0, 1.2, 1.5):
        alt = run(PRIMARY, base_config(relvol_min=threshold), ctx=ctx,
                  split=split, costs=costs)
        filtered.append((f"RelVol >= {threshold:g}", summarise(alt)))
    print()
    print_summary([("no filter (base)", stats)] + filtered)
    result["relvol"] = {"buckets": buckets.to_dicts(),
                        "filters": {k: v for k, v in filtered}}

    rule(f"{split.upper()} 6/7 - opening range length (the paper's Section 5)")
    lengths = []
    for minutes in OR_MINUTES:
        cfg_n = base_config(or_minutes=minutes)
        ctx_n = daily_context(PRIMARY, cfg_n, split=split, allow_test=allow_test)
        alt = run(PRIMARY, cfg_n, ctx=ctx_n, split=split, costs=costs)
        lengths.append((f"{minutes}-minute range", summarise(alt)))
    print_summary(lengths)
    result["or_length"] = {k: v for k, v in lengths}

    rule(f"{split.upper()} 7/7 - breadth: the same 09:30 New York rule elsewhere")
    print("  Not the paper's claim - it is about US equities - but the US cash "
          "open is a real\n  event for all four instruments, so this asks "
          "whether the range or the index is\n  doing the work.")
    breadth = [(f"{PRIMARY} (primary)", stats)]
    for symbol in OTHERS:
        try:
            other_costs = CostModel.from_profiles(symbol, split=split)
            ctx_o = daily_context(symbol, cfg, split=split, allow_test=allow_test)
            alt = run(symbol, cfg, ctx=ctx_o, split=split, costs=other_costs)
            breadth.append((symbol, summarise(alt)))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {symbol}: skipped ({exc})")
    print_summary(breadth)
    result["breadth"] = {k: v for k, v in breadth}
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--placebo-draws", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("-o", "--out", default=str(REPORT_DIR / "opening_range.json"))
    args = p.parse_args(argv)

    if "test" in args.splits:
        print("!! the test split is locked; it is being read deliberately\n")

    results = {"primary": PRIMARY, "config": ORBConfig().__dict__, "splits": {}}
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
