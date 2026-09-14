"""Build the daily and 4-hour bars the FX-ML replication reads.

The corpus stores 1-minute bars, one parquet per symbol-month.
:func:`qlab.bars.resample_bars` aggregates them exactly - the tests pin that
re-aggregating matches building from ticks - so this is a cache, not a model.

Bars land in ``data/processed/bars_whole/symbol=<SYM>/``, one file per
symbol-interval, *not* in the monthly ``bars/`` partition layout. Two reasons:
the loader prunes that layout on filenames alone, so a whole-history file there
cannot be placed and breaks every read of the partition; and this cache is
filtered - weekend bars, which on a 24/5 feed carry a handful of Sunday-open
ticks, are dropped by requiring a minimum tick count - whereas in the partition
layout a missing row means no ticks, not a judgement about thin ones.

Boundaries are UTC, inherited from ``group_by_dynamic``: days break at 00:00
UTC and 4-hour bars at 00/04/08/12/16/20 UTC. That is not the 17:00 New York FX
day used by :data:`qlab.strategies.risk_managed_long.FX_DAY` and by
:func:`qlab.strategies.sweep_orderblock.build_bars`; the replication this feeds
was run on the UTC grid and stays on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qlab import paths  # noqa: E402
from qlab.bars import resample_bars  # noqa: E402

SYMBOLS = ("EURUSD", "USDJPY", "XAUUSD", "USTEC")
INTERVALS = ("4h", "1d")
MIN_TICKS = {"4h": 200, "1d": 1000}


def load_1m(symbol: str) -> pl.DataFrame:
    root = paths.bar_partition_dir(symbol, "1m")
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no 1m bars for {symbol} under {root}")
    return pl.concat([pl.read_parquet(f) for f in files]).sort("ts_open")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    args = ap.parse_args()

    for symbol in args.symbols:
        minute = load_1m(symbol)
        print(f"{symbol}: {minute.height:,} 1m bars "
              f"{minute['ts'].min().date()} -> {minute['ts'].max().date()}")
        for interval in INTERVALS:
            bars = resample_bars(minute, every=interval)
            bars = bars.filter(pl.col("n_ticks") >= MIN_TICKS[interval])
            target = paths.whole_bar_path(symbol, interval)
            target.parent.mkdir(parents=True, exist_ok=True)
            bars.write_parquet(target)
            print(f"  {interval}: {bars.height:,} bars -> {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
