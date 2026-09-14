"""Build coarser monthly bars from the 1-minute partition.

``build_bars.py`` is the authoritative path: ticks in, bars out, one month at a
time. It is also the expensive one - 696 million ticks - and there is no reason
to pay it twice. :func:`qlab.bars.resample_bars` aggregates 1-minute bars into
any whole multiple of a minute and ``tests/test_bars.py`` pins the result as
identical to building that interval from ticks, so for anything coarser than a
minute this script is the same answer three orders of magnitude cheaper.

Output is the same monthly layout everything else reads:
``data/processed/bars/symbol=<SYM>/interval=<IV>/<SYM>_<IV>_<YYYY>_<MM>.parquet``,
which is what ``qlab.loader.load_bars`` prunes by filename.

Boundaries are UTC
------------------
Groups are anchored on the UTC clock: a ``1d`` bar covers [00:00, 24:00) UTC and
a ``4h`` bar breaks at 00/04/08/12/16/20 UTC. That is chosen for one reason -
the monthly layout depends on it. Every interval that divides a UTC day also
divides a month boundary, so no bar straddles two files and the monthly outputs
concatenate without a seam. Intervals that do not divide a day are refused here
rather than written with a silent seam.

This is deliberately **not** the 17:00 New York FX day used by
:data:`qlab.strategies.risk_managed_long.FX_DAY` and by
:func:`qlab.strategies.sweep_orderblock.build_bars`. On that boundary the day
opening 17:00 New York on the last of a month closes on the 1st of the next, so
its bar belongs to two files at once and the seam guarantee is gone. Strategies
that want the FX day build it themselves from 1-minute bars, where the straddle
is an in-memory grouping rather than a fact about the layout on disk.

Nothing is filtered. A thin bar is still a bar, and a missing row here means no
ticks arrived - never a judgement that too few did. Callers that want weekend
stubs gone drop them themselves; see ``prepare_fx_bars.py``, which does exactly
that for its own whole-history cache under ``bars_whole/``.

Usage
-----
    python scripts/pipeline/resample_bars.py -i 1d 4h            # all symbols
    python scripts/pipeline/resample_bars.py -i 4h -s XAUUSD --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qlab import paths  # noqa: E402
from qlab.bars import resample_bars  # noqa: E402
from qlab.loader import parse_month  # noqa: E402
from qlab.symbols import ALL_SYMBOLS, get_spec  # noqa: E402

SOURCE_INTERVAL = "1m"
DAY_MINUTES = 1440


def interval_minutes(interval: str) -> int:
    """Minutes in ``interval``, refusing anything the monthly layout cannot hold."""
    units = {"m": 1, "h": 60, "d": DAY_MINUTES}
    if len(interval) < 2 or interval[-1] not in units or not interval[:-1].isdigit():
        raise ValueError(f"interval {interval!r} is not <n>m / <n>h / <n>d")
    minutes = int(interval[:-1]) * units[interval[-1]]
    if minutes <= 1:
        raise ValueError(f"interval {interval!r} is not coarser than {SOURCE_INTERVAL}")
    if DAY_MINUTES % minutes:
        raise ValueError(
            f"interval {interval!r} ({minutes} min) does not divide a UTC day, so its "
            "bars would straddle month boundaries and the monthly files would no "
            "longer concatenate without a seam"
        )
    return minutes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-i", "--intervals", nargs="+", default=["4h", "1d"])
    parser.add_argument("--force", action="store_true", help="rebuild up-to-date months")
    args = parser.parse_args()

    for interval in args.intervals:
        interval_minutes(interval)

    paths.ensure_dirs()
    started = time.perf_counter()
    built = rows = 0

    for symbol in args.symbols:
        spec = get_spec(symbol)
        source_dir = paths.bar_partition_dir(spec.name, SOURCE_INTERVAL)
        months = sorted(source_dir.glob("*.parquet"))
        if not months:
            print(f"{spec.name}: no {SOURCE_INTERVAL} bars under {source_dir}, skipping", flush=True)
            continue
        newest = months[-1]

        for interval in args.intervals:
            for source in months:
                year, month = parse_month(source)
                out_path = paths.bar_parquet_path(spec.name, interval, year, month)
                # The newest month is always rebuilt: its final bar depends on
                # where the 1-minute data currently stops.
                stale = (
                    not out_path.exists()
                    or out_path.stat().st_mtime < source.stat().st_mtime
                    or source == newest
                )
                if not (args.force or stale):
                    continue

                bars = resample_bars(pl.read_parquet(source), every=interval)
                # Only the newest month is cut off mid-interval by "now" rather
                # than by the month ending; every other file's last bar is whole.
                if source == newest and bars.height:
                    bars = bars.head(bars.height - 1)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(".parquet.tmp")
                bars.write_parquet(
                    tmp_path, compression="zstd", compression_level=3, statistics=True
                )
                os.replace(tmp_path, out_path)
                built += 1
                rows += bars.height
                print(
                    f"{spec.name} {interval} {year}-{month:02d}  {bars.height:>5,} bars",
                    flush=True,
                )

    if not built:
        print("all bars up to date", flush=True)
        return 0
    print(
        f"\nbuilt {built} symbol-months, {rows:,} bars in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"output: {paths.BARS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
