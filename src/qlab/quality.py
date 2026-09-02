"""Tick-data quality assessment.

This module *measures* problems, it never repairs them. Its job is to tell you
which symbols and which date ranges are trustworthy enough to build a strategy
on, and to make any hidden defect in the vendor feed visible before it silently
becomes a backtest assumption.

Metric groups
-------------
integrity   duplicate timestamps, out-of-order ticks, exact duplicate rows
quotes      zero-spread (bid == ask) and crossed (ask < bid) ticks, spread quantiles
continuity  inter-tick gaps (weekend gaps separated out), missing weekdays
sanity      price range, extreme tick-to-tick jumps measured in basis points

A high zero-spread share is **not** by itself a defect. On an Exness Raw Spread
account the majors are quoted at a genuine 0.0 pip average spread and the broker
charges commission instead, so EURUSD sitting at 97% zero-spread agrees with the
published contract specification. What matters is whether the observed mean
spread matches that specification - see :func:`spec_agreement`. A crossed quote
(ask < bid) is always corrupt.

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
        zero_spread=(pl.col("bid") == pl.col("ask")).sum(),
        crossed=(pl.col("ask") < pl.col("bid")).sum(),
        n_session_breaks=(pl.col("in_break_window") & (pl.col("gap_s") > 600)).sum(),
        max_outage_gap_s=outage_gap.max(),
        **{
            f"n_outage_gap_gt_{bucket}s": (outage_gap > bucket).sum()
            for bucket in GAP_BUCKETS_S
        },
        spread_mean_pips=pl.col("spread_pips").mean(),
        # Sunday is the week's reopen, where spread widens by an order of
        # magnitude. The broker's published average is a weekday figure, and most
        # strategies will not trade the reopen, so keep a Mon-Fri mean alongside.
        spread_mean_mon_fri_pips=pl.when(pl.col("ts").dt.weekday() < 6)
        .then(pl.col("spread_pips"))
        .mean(),
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
        "zero_spread_pct": 100.0 * agg["zero_spread"] / rows,
        "crossed_pct": 100.0 * agg["crossed"] / rows,
        **{
            key: agg[key]
            for key in (
                "spread_mean_pips",
                "spread_mean_mon_fri_pips",
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


def spec_agreement(
    monthly: pl.DataFrame, symbol: str, trailing_months: int = 3
) -> dict:
    """Check the feed's mean spread against the broker's published average.

    This is the check that a raw zero-spread count cannot give you. The broker
    quotes an average, so the test is whether the tick-weighted mean matches that
    figure - not whether zero spreads occur.

    The window is a **trailing** one, not a fixed start year, because Exness
    publishes averages "based on the previous trading day". Comparing a
    point-in-time figure against a multi-year mean produces false alarms on any
    instrument whose spread has moved: XAUUSD averages 8.7 pips over the last
    three months, matching the published 9.0, but only 6.0 over 2024-2026,
    because gold's spread ran 5.8 -> 3.7 -> 9.0 across those years.

    Mon-Fri, because a single trading day's average cannot include the Sunday
    reopen, where spread widens by an order of magnitude.

    Agreement is judged economically rather than by exact rounding: residual
    spread only matters relative to the commission paid on the same trade. A
    feed quoting *tighter* than published is called out separately - it is a
    better fill than advertised, not a defect.
    """
    spec = get_spec(symbol)
    recent = (
        monthly.filter(pl.col("symbol") == symbol)
        .sort(["year", "month"])
        .tail(trailing_months)
    )
    if recent.is_empty():
        return {"symbol": symbol, "status": "no data"}

    weights = recent["rows"]
    observed = float(
        (recent["spread_mean_mon_fri_pips"] * weights).sum() / weights.sum()
    )

    result = {
        "symbol": symbol,
        "observed_mean_pips": observed,
        "spec_mean_pips": spec.spec_avg_spread_pips,
        "trailing_months": trailing_months,
    }
    if spec.spec_avg_spread_pips is None:
        result["status"] = "no published spec on file"
        return result

    deviation = observed - spec.spec_avg_spread_pips
    tolerance = 0.05  # half the published precision
    if spec.commission_per_lot_side_usd is not None:
        quote_rate = float(recent["price_max"].mean()) if spec.quote_ccy == "JPY" else 1.0
        tolerance = max(tolerance, 0.2 * spec.commission_pips(quote_rate))

    result["deviation_pips"] = deviation
    result["tolerance_pips"] = tolerance
    result["agrees"] = abs(deviation) <= tolerance
    if result["agrees"]:
        result["status"] = "agrees with spec"
    elif deviation < 0:
        result["status"] = "tighter than published"
    else:
        result["status"] = "wider than published - investigate"
    return result


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
