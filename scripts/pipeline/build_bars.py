"""Build bars from converted ticks.

Ticks are the record; bars are the working surface. 696 million ticks is too
many to re-aggregate on every research iteration, and one-minute bars are three
orders of magnitude smaller, so they get materialised once and read many times.

One month of ticks in, one month of bars out. Bar intervals that divide a day
divide a month boundary too, so no bar straddles two files and the monthly
outputs concatenate without a seam.

Usage
-----
    python scripts/pipeline/build_bars.py                      # 1m bars, all symbols
    python scripts/pipeline/build_bars.py -i 1m 5m -s XAUUSD
    python scripts/pipeline/build_bars.py --force -w 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

from qlab import paths
from qlab.bars import time_bars
from qlab.loader import DEFAULT_POLICY, clean_ticks, cleaning_report
from qlab.symbols import ALL_SYMBOLS, get_spec


def _job(payload: tuple[str, str, str, bool]) -> dict:
    """Picklable worker: one symbol-month, one interval."""
    symbol, interval, tick_path_str, is_newest = payload
    tick_path = Path(tick_path_str)
    year, month = int(tick_path.stem[-7:-3]), int(tick_path.stem[-2:])
    out_path = paths.bar_parquet_path(symbol, interval, year, month)

    try:
        started = time.perf_counter()
        raw = pl.read_parquet(tick_path, columns=["ts", "bid", "ask"])
        report = cleaning_report(raw, DEFAULT_POLICY)
        ticks = clean_ticks(raw, DEFAULT_POLICY)

        # Only the newest month of a symbol is truncated by "now" rather than by
        # the month ending; every other file's final interval is genuinely complete.
        bars = time_bars(ticks, interval, drop_last=is_newest)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".parquet.tmp")
        bars.write_parquet(tmp_path, compression="zstd", compression_level=3, statistics=True)
        os.replace(tmp_path, out_path)

        return {
            "symbol": symbol,
            "interval": interval,
            "year": year,
            "month": month,
            "ticks_in": report["rows"],
            "dropped": report["rows"] - ticks.height,
            "shared_ts_quotes": report["distinct_quotes_at_shared_ts"],
            "crossed": report["crossed"],
            "bars": bars.height,
            "out_bytes": out_path.stat().st_size,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "interval": interval,
            "year": year,
            "month": month,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-i", "--intervals", nargs="+", default=["1m"])
    parser.add_argument("-w", "--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="rebuild up-to-date months")
    args = parser.parse_args()

    paths.ensure_dirs()
    payloads: list[tuple[str, str, str, bool]] = []
    for symbol in args.symbols:
        spec = get_spec(symbol)
        tick_files = sorted(paths.tick_partition_dir(spec.name).glob("*.parquet"))
        if not tick_files:
            print(f"{spec.name}: no converted ticks, skipping", flush=True)
            continue
        newest = tick_files[-1]
        for interval in args.intervals:
            for tick_path in tick_files:
                year, month = int(tick_path.stem[-7:-3]), int(tick_path.stem[-2:])
                out_path = paths.bar_parquet_path(spec.name, interval, year, month)
                stale = (
                    not out_path.exists()
                    or out_path.stat().st_mtime < tick_path.stat().st_mtime
                    # The newest month is rebuilt on every run: its final bar
                    # depends on where the data currently stops.
                    or tick_path == newest
                )
                if args.force or stale:
                    payloads.append(
                        (spec.name, interval, str(tick_path), tick_path == newest)
                    )

    if not payloads:
        print("all bars up to date", flush=True)
        return 0

    print(f"{len(payloads)} symbol-months to build", flush=True)
    started = time.perf_counter()
    records: list[dict] = []
    failures: list[dict] = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            label = (
                f"{record['symbol']} {record['interval']} "
                f"{record['year']}-{record['month']:02d}"
            )
            if record["status"] == "ok":
                print(
                    f"[{done:>3}/{len(payloads)}] {label}  "
                    f"{record['ticks_in']:>9,} ticks -> {record['bars']:>7,} bars  "
                    f"{record['dropped']:>7,} dropped  {record['elapsed_s']:>5.1f}s",
                    flush=True,
                )
            else:
                failures.append(record)
                print(f"[{done:>3}/{len(payloads)}] {label}  FAILED  {record['error']}", flush=True)

    ok = [r for r in records if r["status"] == "ok"]
    ticks_in = sum(r["ticks_in"] for r in ok)
    dropped = sum(r["dropped"] for r in ok)
    print(
        f"\nbuilt {len(ok)}/{len(payloads)} months, {sum(r['bars'] for r in ok):,} bars "
        f"from {ticks_in:,} ticks in {(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    # The cleaning total is reported, not buried: it is the one number that says
    # how much of the raw record the working surface is no longer showing you.
    shared = sum(r["shared_ts_quotes"] for r in ok)
    crossed = sum(r["crossed"] for r in ok)
    print(
        f"cleaning dropped {dropped:,} ticks ({100.0 * dropped / max(ticks_in, 1):.3f}%), "
        f"all of them repeats of the preceding tick; {crossed:,} crossed quotes",
        flush=True,
    )
    print(
        f"kept {shared:,} ticks ({100.0 * shared / max(ticks_in, 1):.3f}%) that share a "
        "millisecond with a different quote - real prices a unique-index policy would delete",
        flush=True,
    )
    print(f"output: {paths.BARS_DIR}", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILURES:", flush=True)
        for record in failures:
            print(f"  {record['symbol']} {record['year']}-{record['month']:02d}: {record['error']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
