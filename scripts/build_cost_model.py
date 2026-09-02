"""Measure the spread and latency profiles the cost model reads.

Two passes. The spread pass reads the built bars and is nearly free. The latency
pass reads ticks and does an as-of join per horizon, so it is the expensive one -
a few minutes for the whole corpus on 8 workers.

Both write one row per symbol-month-hour, not per symbol, so that the profiles
can be aggregated to a year, a split or a trailing window afterwards without
being rebuilt. Spread has compressed by an order of magnitude since 2020; a
single flat number for the corpus would be wrong at both ends of it.

Usage
-----
    python scripts/build_cost_model.py -w 8
    python scripts/build_cost_model.py -s XAUUSD --skip-latency
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from qlab import paths
from qlab.costprofile import HORIZONS_MS, latency_profile, spread_profile
from qlab.symbols import ALL_SYMBOLS, get_spec


def _spread_job(payload: tuple[str, str]) -> dict:
    symbol, bar_path_str = payload
    bar_path = Path(bar_path_str)
    year, month = int(bar_path.stem[-7:-3]), int(bar_path.stem[-2:])
    try:
        spec = get_spec(symbol)
        frame = spread_profile(pl.read_parquet(bar_path), spec)
        if frame.is_empty():
            return {"status": "empty", "symbol": symbol, "year": year, "month": month}
        frame = frame.with_columns(
            year=pl.lit(year, pl.Int32), month=pl.lit(month, pl.Int8)
        )
        return {
            "status": "ok",
            "symbol": symbol,
            "year": year,
            "month": month,
            "rows": frame.to_dicts(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "year": year,
            "month": month,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _latency_job(payload: tuple[str, str, tuple[int, ...]]) -> dict:
    symbol, tick_path_str, horizons = payload
    tick_path = Path(tick_path_str)
    year, month = int(tick_path.stem[-7:-3]), int(tick_path.stem[-2:])
    try:
        started = time.perf_counter()
        spec = get_spec(symbol)
        ticks = pl.read_parquet(tick_path, columns=["ts", "bid", "ask"])
        frame = latency_profile(ticks, spec, horizons_ms=horizons)
        if frame.is_empty():
            return {"status": "empty", "symbol": symbol, "year": year, "month": month}
        frame = frame.with_columns(
            year=pl.lit(year, pl.Int32), month=pl.lit(month, pl.Int8)
        )
        return {
            "status": "ok",
            "symbol": symbol,
            "year": year,
            "month": month,
            "ticks": ticks.height,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "rows": frame.to_dicts(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "year": year,
            "month": month,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run(label: str, job, payloads: list, workers: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record["status"] == "ok":
                rows.extend(record["rows"])
            elif record["status"] == "error":
                failures.append(record)
                print(
                    f"  {record['symbol']} {record['year']}-{record['month']:02d} "
                    f"FAILED {record['error']}",
                    flush=True,
                )
            if done % 40 == 0 or done == len(payloads):
                print(
                    f"  {label}: {done}/{len(payloads)} months, "
                    f"{(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
    return rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-w", "--workers", type=int, default=6)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    failures: list[dict] = []

    print("spread profile (from bars)", flush=True)
    spread_payloads = [
        (get_spec(s).name, str(p))
        for s in args.symbols
        for p in sorted(paths.bar_partition_dir(get_spec(s).name, "1m").glob("*.parquet"))
    ]
    if not spread_payloads:
        print("no bars found - run scripts/build_bars.py first", flush=True)
        return 1

    rows, failed = _run("spread", _spread_job, spread_payloads, args.workers)
    failures += failed
    spread = pl.DataFrame(rows).select(
        "symbol", "year", "month", "hour", "is_sunday",
        "n_bars", "n_ticks",
        "spread_mean_pips", "spread_p50_pips", "spread_p95_pips", "spread_max_pips",
    ).sort("symbol", "year", "month", "hour", "is_sunday")
    spread.write_parquet(paths.SPREAD_PROFILE_PATH)
    print(f"  wrote {spread.height:,} rows -> {paths.SPREAD_PROFILE_PATH}\n", flush=True)

    if not args.skip_latency:
        print(f"latency profile (from ticks, horizons {HORIZONS_MS} ms)", flush=True)
        latency_payloads = [
            (get_spec(s).name, str(p), HORIZONS_MS)
            for s in args.symbols
            for p in sorted(paths.tick_partition_dir(get_spec(s).name).glob("*.parquet"))
        ]
        rows, failed = _run("latency", _latency_job, latency_payloads, args.workers)
        failures += failed
        latency = pl.DataFrame(rows).select(
            "symbol", "year", "month", "hour", "horizon_ms",
            "n_anchors", "drift_mean_pips", "drift_p50_pips", "drift_p95_pips",
            "zero_share",
        ).sort("symbol", "year", "month", "horizon_ms", "hour")
        latency.write_parquet(paths.LATENCY_PROFILE_PATH)
        print(
            f"  wrote {latency.height:,} rows -> {paths.LATENCY_PROFILE_PATH}",
            flush=True,
        )

    if failures:
        print(f"\n{len(failures)} FAILURES", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
