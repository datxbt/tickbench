"""Tick-data quality assessment.

This module *measures* problems, it never repairs them. Its job is to tell you
which symbols and which date ranges are trustworthy enough to build a strategy
on, and to make any hidden defect in the vendor feed visible before it silently
becomes a backtest assumption.

Metric groups
-------------
integrity   duplicate timestamps, out-of-order ticks, exact duplicate rows
quotes      locked (bid == ask) and crossed (ask < bid) markets, spread quantiles
continuity  inter-tick gaps (weekend gaps separated out), missing weekdays
sanity      price range, extreme tick-to-tick jumps measured in basis points

Tick jumps are expressed in basis points of mid price rather than pips so the
number means the same thing on EURUSD, XAUUSD and USTEC.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from . import paths
from .symbols import ALL_SYMBOLS, SymbolSpec, get_spec

# A tick-to-tick mid move larger than this is flagged as a candidate bad print.
JUMP_THRESHOLD_BPS = 10.0
GAP_BUCKETS_S = (60, 300, 3600)


def _weekdays_between(start: date, end: date) -> set[date]:
    days: set[date] = set()
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.add(current)
        current += timedelta(days=1)
    return days


def month_metrics(parquet_path: Path, spec: SymbolSpec) -> dict:
    """Compute the full quality metric set for one symbol-month parquet file."""
    ticks = pl.read_parquet(parquet_path, columns=["ts", "bid", "ask"])
    rows = ticks.height

    prev_ts = pl.col("ts").shift(1)
    # A gap is a weekend gap if the interval it spans contains a Saturday.
    # (13 - weekday) % 7 keeps the numerator non-negative for weekday in 1..7.
    saturday_start = prev_ts.dt.truncate("1d") + pl.duration(
        days=(13 - prev_ts.dt.weekday()) % 7
    )

    # A gap that begins inside the instrument's daily maintenance window is
    # routine broker downtime, not a feed outage. Without this split the daily
    # break dominates the gap counts and buries real outages.
    if spec.daily_break_utc is None:
        in_break_window = pl.lit(False)
    else:
        start, end = spec.daily_break_utc
        in_break_window = (prev_ts.dt.hour() >= start) & (prev_ts.dt.hour() < end)

    enriched = ticks.with_columns(
        spread_pips=(pl.col("ask") - pl.col("bid")) / spec.pip,
        mid=(pl.col("bid") + pl.col("ask")) / 2,
        gap_s=(pl.col("ts") - prev_ts).dt.total_seconds(),
        crosses_weekend=saturday_start < pl.col("ts"),
        in_break_window=in_break_window,
    ).with_columns(
        # Return convention: normalise by the previous mid, not the current one.
        jump_bps=(pl.col("mid").diff().abs() / pl.col("mid").shift(1) * 10_000),
    )

    intraweek_gap = pl.when(~pl.col("crosses_weekend")).then(pl.col("gap_s"))
    outage_gap = pl.when(
        ~pl.col("crosses_weekend") & ~pl.col("in_break_window")
    ).then(pl.col("gap_s"))
    agg = enriched.select(
        locked=(pl.col("bid") == pl.col("ask")).sum(),
        crossed=(pl.col("ask") < pl.col("bid")).sum(),
        n_session_breaks=(pl.col("in_break_window") & (pl.col("gap_s") > 600)).sum(),
        max_outage_gap_s=outage_gap.max(),
        **{
            f"n_outage_gap_gt_{bucket}s": (outage_gap > bucket).sum()
            for bucket in GAP_BUCKETS_S
        },
        spread_mean_pips=pl.col("spread_pips").mean(),
        spread_p50_pips=pl.col("spread_pips").quantile(0.50),
        spread_p95_pips=pl.col("spread_pips").quantile(0.95),
        spread_p99_pips=pl.col("spread_pips").quantile(0.99),
        spread_max_pips=pl.col("spread_pips").max(),
        price_min=pl.col("bid").min(),
        price_max=pl.col("ask").max(),
        max_gap_s=pl.col("gap_s").max(),
        max_intraweek_gap_s=intraweek_gap.max(),
        max_jump_bps=pl.col("jump_bps").max(),
        n_jumps=(pl.col("jump_bps") > JUMP_THRESHOLD_BPS).sum(),
        **{
            f"n_intraweek_gap_gt_{bucket}s": (intraweek_gap > bucket).sum()
            for bucket in GAP_BUCKETS_S
        },
    ).row(0, named=True)

    ts_min, ts_max = ticks["ts"][0], ticks["ts"][-1]
    present_days = set(ticks.select(pl.col("ts").dt.date().unique())["ts"].to_list())
    expected_weekdays = _weekdays_between(ts_min.date(), ts_max.date())
    ticks_per_day = (
        ticks.group_by(pl.col("ts").dt.date()).len()["len"].median() if rows else 0
    )

    duplicate_ts = rows - ticks["ts"].n_unique()
    return {
        "symbol": spec.name,
        "year": ts_min.year,
        "month": ts_min.month,
        "rows": rows,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "days_present": len(present_days),
        "missing_weekdays": len(expected_weekdays - present_days),
        "median_ticks_per_day": float(ticks_per_day),
        "duplicate_ts": duplicate_ts,
        "duplicate_ts_pct": 100.0 * duplicate_ts / rows,
        "exact_duplicate_rows": rows - ticks.n_unique(),
        "out_of_order": int(not ticks["ts"].is_sorted()),
        "locked_pct": 100.0 * agg["locked"] / rows,
        "crossed_pct": 100.0 * agg["crossed"] / rows,
        **{
            key: agg[key]
            for key in (
                "spread_mean_pips",
                "spread_p50_pips",
                "spread_p95_pips",
                "spread_p99_pips",
                "spread_max_pips",
                "price_min",
                "price_max",
                "max_gap_s",
                "max_intraweek_gap_s",
                "max_outage_gap_s",
                "n_session_breaks",
                "max_jump_bps",
                "n_jumps",
                *(f"n_intraweek_gap_gt_{b}s" for b in GAP_BUCKETS_S),
                *(f"n_outage_gap_gt_{b}s" for b in GAP_BUCKETS_S),
            )
        },
    }


def hourly_profile(parquet_path: Path, symbol: str) -> pl.DataFrame:
    """Tick counts by UTC hour - reveals feed outages and session structure."""
    return (
        pl.read_parquet(parquet_path, columns=["ts"])
        .group_by(pl.col("ts").dt.hour().alias("hour"))
        .len()
        .with_columns(symbol=pl.lit(symbol))
        .select("symbol", "hour", "len")
    )


def _job(payload: tuple[str, str]) -> dict:
    """Picklable worker entry point."""
    symbol, parquet_path = payload
    path = Path(parquet_path)
    try:
        spec = get_spec(symbol)
        record = month_metrics(path, spec)
        record["status"] = "ok"
        profile = hourly_profile(path, symbol)
        record["_hourly"] = profile.to_dicts()
        return record
    except Exception as exc:
        return {
            "symbol": symbol,
            "file": path.name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(symbols: Iterable[str] | None = None, *, workers: int = 6) -> Iterator[dict]:
    """Assess every converted parquet file, yielding one record per symbol-month."""
    payloads: list[tuple[str, str]] = []
    for symbol in symbols or ALL_SYMBOLS:
        spec = get_spec(symbol)
        for path in sorted(paths.tick_partition_dir(spec.name).glob("*.parquet")):
            payloads.append((spec.name, str(path)))

    if not payloads:
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, p) for p in payloads]
        for future in as_completed(futures):
            yield future.result()
