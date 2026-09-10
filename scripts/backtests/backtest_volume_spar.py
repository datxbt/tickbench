"""Test the SPAR intraday volume model of Tan, Zhang and Zhu (2026).

The paper's claim has two halves, and they do not stand or fall together:

* **Forecasting.** Removing intraday periodicity with a panel within-
  transformation, rather than fitting a pre-specified U-shaped curve, raises
  out-of-sample R-squared from 0.103 (OLS on identical features) to 0.293.
* **Execution.** Fed into the dynamic VWAP replication schedule of Bialkowski
  et al. (2008), that accuracy becomes money: mean tracking error 7.013 bp for
  SPAR3 against 7.275 bp for OLS3 and 8.804 bp for equal weight, which the
  paper monetises at roughly $14k and $99k a year on a $2.2m daily order.

This script tests both halves separately, on four CFDs the paper never saw, and
on two intraday grids. The 24-hour grid is the harder test and the more
interesting one: US equities have one session and a U-shaped curve, while these
instruments run around the clock with three activity humps, which is exactly
the case a pre-specified U-spline cannot fit and SPAR claims it does not need to.

Four passes:

1. **The periodic curve**, estimated without assuming a shape. If these
   instruments were U-shaped the paper's premise would not even be under test.
2. **Forecast accuracy.** SPAR against OLS on identical features, and both
   against the rolling historical mean that defines out-of-sample R-squared.
   Reported on logs, where the models are fitted, and on levels, where the
   paper's equation is written - they disagree, and the disagreement matters.
3. **VWAP tracking error**, and - this is the part the paper omits - a paired
   block bootstrap of the difference. The advertised SPAR-over-OLS edge is
   0.26 bp against a day-to-day dispersion above 5 bp, so whether it is a
   finding or a coincidence cannot be read off the means alone.
4. **The verdict per panel**, so a claim that holds for one instrument and not
   another says so.

Run:  python scripts/backtests/backtest_volume_spar.py
      python scripts/backtests/backtest_volume_spar.py --symbols USTEC --splits dev
      python scripts/backtests/backtest_volume_spar.py --splits test    # locked
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.bars import resample_bars  # noqa: E402
from qlab.loader import SPLITS, load_bars  # noqa: E402
from qlab.volume_spar import (  # noqa: E402
    build_panel,
    paired_te,
    periodic_curve,
    score_all,
)

OUT = paths.STRATEGY_REPORT_DIR
GRIDS = (("session", ("09:30", "16:00"), 0.98), ("24h", None, 0.80))
SYMBOLS = ("USTEC", "XAUUSD", "EURUSD", "USDJPY")


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def sparkline(curve: np.ndarray, width: int = 60) -> str:
    """The periodic curve as one line of text, so the shape is visible at a glance."""
    blocks = " .:-=+*#%@"
    step = max(1, len(curve) // width)
    thinned = curve[::step]
    lo, hi = thinned.min(), thinned.max()
    span = hi - lo or 1.0
    return "".join(blocks[min(9, int(9 * (v - lo) / span))] for v in thinned)


def panels_for(symbol: str, split: str, allow_test: bool):
    bars = load_bars(
        symbol, "1m", split=split, warmup=timedelta(days=200), allow_test=allow_test
    )
    five = resample_bars(bars, "5m")
    for grid, session, coverage in GRIDS:
        yield grid, build_panel(
            five, symbol=symbol, session=session, min_coverage=coverage
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    parser.add_argument("--splits", nargs="+", default=["dev", "validation"])
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    unknown = set(args.splits) - set(SPLITS)
    if unknown:
        parser.error(f"unknown split(s): {', '.join(sorted(unknown))}")
    allow_test = "test" in args.splits
    if allow_test:
        print("!! the test split is unlocked. One shot.\n")
    OUT.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        accuracy, tracking, pairs, curves = [], [], [], []

        rule(f"1. The periodic curve, no shape assumed  [{split}]")
        print("Low activity ' ' to high '@', left to right across the grid.\n")
        for symbol in args.symbols:
            for grid, panel in panels_for(symbol, split, allow_test):
                curve = periodic_curve(panel)
                curves.append(
                    pl.DataFrame(
                        {"symbol": symbol, "grid": grid, "slot": panel.slots, "F": curve}
                    )
                )
                print(f"  {symbol:<7}{grid:<9}|{sparkline(curve)}|  "
                      f"peak/trough {curve.max() / curve.min():.1f}x, "
                      f"{panel.shape[0]} days")

                acc, trk, schedules = score_all(panel)
                accuracy.append(acc)
                tracking.append(trk)
                for left, right in (
                    ("SPAR1", "OLS1"), ("SPAR3", "OLS3"),
                    ("SPAR4", "OLS4"), ("SPAR3", "EW"),
                ):
                    row = paired_te(schedules[left], schedules[right], n_boot=args.boot)
                    row.update(symbol=symbol, grid=grid)
                    pairs.append(row)

        acc = pl.concat(accuracy)
        trk = pl.concat(tracking)
        prs = pl.DataFrame(pairs)
        pl.concat(curves).write_parquet(OUT / f"spar_curves_{split}.parquet")
        acc.write_parquet(OUT / f"spar_accuracy_{split}.parquet")
        trk.write_parquet(OUT / f"spar_vwap_{split}.parquet")
        prs.write_parquet(OUT / f"spar_paired_{split}.parquet")

        rule(f"2. Forecast accuracy - out-of-sample R2 against the rolling mean  [{split}]")
        print("On logs, where the models are fitted:\n")
        print(
            acc.pivot(on="model", index=["symbol", "grid"], values="r2_log")
            .to_pandas().round(3).to_string(index=False)
        )
        wins = sum(
            1
            for spec in (1, 2, 3, 4)
            for (sym, grid), block in acc.group_by(["symbol", "grid"])
            if _beats(block, f"SPAR{spec}", f"OLS{spec}")
        )
        print(f"\n  SPAR beats OLS on identical features in {wins} of "
              f"{4 * acc.select('symbol', 'grid').unique().height} comparisons.")
        print("\nOn levels, where the paper's R2 equation is written:\n")
        print(
            acc.pivot(on="model", index=["symbol", "grid"], values="r2_level")
            .to_pandas().round(3).to_string(index=False)
        )
        print(
            "\n  A large negative here is the log-normal retransformation, not a\n"
            "  broken model: exp(E[log V]) is not E[V], and on a fat-tailed slot\n"
            "  the gap explodes. Anyone deploying this owes it a smearing\n"
            "  correction before reading a level forecast."
        )

        rule(f"3. VWAP replication - tracking error, and whether the gap is real  [{split}]")
        print(trk.to_pandas().round(3).to_string(index=False))
        print(
            "\nPaired block bootstrap of mean absolute tracking error on common\n"
            "days. A negative diff_bp means the left schedule tracked closer.\n"
        )
        print(
            prs.select("symbol", "grid", "left", "right", "days",
                       "diff_bp", "lo_bp", "hi_bp", "p")
            .to_pandas().round(3).to_string(index=False)
        )

        rule(f"4. Verdict  [{split}]")
        _verdict(prs, "SPAR3", "OLS3", "the within-transformation itself")
        _verdict(prs, "SPAR3", "EW", "forecasting at all, against equal weight")

    return 0


def _beats(block: pl.DataFrame, left: str, right: str) -> bool:
    lookup = dict(zip(block["model"], block["r2_log"]))
    return lookup.get(left, -9e9) > lookup.get(right, 9e9)


def _verdict(pairs: pl.DataFrame, left: str, right: str, what: str) -> None:
    subset = pairs.filter((pl.col("left") == left) & (pl.col("right") == right))
    better = subset.filter((pl.col("diff_bp") < 0) & (pl.col("p") < 0.05)).height
    worse = subset.filter((pl.col("diff_bp") > 0) & (pl.col("p") < 0.05)).height
    flat = subset.height - better - worse
    print(
        f"  {what}:\n"
        f"    {left} closer to VWAP than {right}, p<0.05 : {better} of {subset.height} panels\n"
        f"    {right} closer, p<0.05                     : {worse}\n"
        f"    no distinguishable difference              : {flat}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
