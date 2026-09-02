"""Bar construction from ticks.

The timestamp convention is the whole point of this module, so it is stated
once, here, and everything else follows from it:

    ``ts`` is the instant at which the bar's contents become known.

A one-minute bar covering ``[10:00:00, 10:01:00)`` is labelled ``10:01:00``.
That is right-edge labelling, and it is chosen over the more common left-edge
convention for one reason: it makes the naive thing safe. With right-edge
labels, joining a bar to anything else on ``ts``, or acting on row *i* at row
*i*'s timestamp, uses only information that existed at that moment. With
left-edge labels the same code silently trades on a candle that had not yet
closed - the most common lookahead bug there is, and one that flatters a
backtest rather than breaking it.

``ts_open`` carries the left edge for anyone who wants it, and
``first_tick_ts`` / ``last_tick_ts`` carry the actual observed extent, which is
how you tell a bar built from 400 ticks apart from one built from 2 in a thin
session.

Bars are OHLC on the **mid**, plus the closing bid and ask separately, because
you enter at the ask and exit at the bid and no single price can stand in for
both. Spread is kept in price units, matching the tick schema; divide by
``spec.pip`` where you want pips.

There is no volume. The Exness tick feed quotes bid and ask only, with no size,
so volume bars and dollar bars are not constructible from it. :func:`tick_bars`
is the available substitute: sampling every N quote updates gives bars that
stretch in quiet markets and compress in busy ones, which is most of what
volume bars are wanted for.
"""

from __future__ import annotations

import polars as pl

BAR_COLUMNS = [
    "ts",
    "ts_open",
    "first_tick_ts",
    "last_tick_ts",
    "open",
    "high",
    "low",
    "close",
    "mid_mean",
    "bid_close",
    "ask_close",
    "spread_mean",
    "spread_close",
    "n_ticks",
]

_AGGREGATIONS = dict(
    first_tick_ts=pl.col("ts").first(),
    last_tick_ts=pl.col("ts").last(),
    open=pl.col("mid").first(),
    high=pl.col("mid").max(),
    low=pl.col("mid").min(),
    close=pl.col("mid").last(),
    mid_mean=pl.col("mid").mean(),
    bid_close=pl.col("bid").last(),
    ask_close=pl.col("ask").last(),
    spread_mean=pl.col("spread").mean(),
    spread_close=pl.col("spread").last(),
    n_ticks=pl.len(),
)


def _with_derived(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        mid=(pl.col("bid") + pl.col("ask")) / 2,
        spread=pl.col("ask") - pl.col("bid"),
    )


def time_bars(
    ticks: pl.DataFrame | pl.LazyFrame,
    every: str = "1m",
    *,
    drop_last: bool = False,
) -> pl.DataFrame:
    """Aggregate ticks into fixed-interval bars labelled at the interval end.

    Windows are half-open ``[start, end)``: a tick at exactly 10:01:00.000 opens
    the next bar rather than closing this one.

    Only intervals that contain at least one tick produce a row. Weekends, the
    daily maintenance break and outages therefore appear as *missing* rows, not
    as fabricated flat bars - a flat bar is a price assertion, and there is no
    evidence for it. Downstream code that needs a regular grid should reindex
    explicitly, so that the filling is its own visible decision.

    ``drop_last=True`` discards the final bar, for the case where observation
    was cut off mid-interval rather than the market having closed. Within a
    complete calendar month it is unnecessary: every sub-daily interval divides
    the month boundary exactly, so no bar straddles two files and monthly bar
    files concatenate without seams.
    """
    lazy = _with_derived(ticks.lazy())
    bars = (
        lazy.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(**_AGGREGATIONS)
        .with_columns(ts_open=pl.col("ts").dt.offset_by(f"-{every}"))
        .select(BAR_COLUMNS)
        .collect()
    )
    if drop_last and bars.height:
        bars = bars.head(bars.height - 1)
    return bars


def tick_bars(
    ticks: pl.DataFrame | pl.LazyFrame, n: int = 1000, *, drop_last: bool = True
) -> pl.DataFrame:
    """Aggregate ticks into bars of exactly ``n`` quote updates.

    Sampling on activity rather than on the clock gives bars whose returns are
    closer to identically distributed than time bars are, because it spends more
    resolution where the information is. The cost is a timestamp axis that is no
    longer regular.

    ``ts`` is the timestamp of the bar's last tick - the instant the bar is
    complete, which keeps the convention identical to :func:`time_bars`. The
    trailing partial bar is dropped by default: it is not a bar yet, and letting
    it through means the newest row is built from fewer ticks than every other
    row, which shows up as a spurious volatility change at the sample edge.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    lazy = _with_derived(ticks.lazy()).sort("ts")
    bars = (
        lazy.with_columns(_bar=pl.int_range(pl.len(), dtype=pl.Int64) // n)
        .group_by("_bar", maintain_order=True)
        .agg(**_AGGREGATIONS)
        .with_columns(ts=pl.col("last_tick_ts"), ts_open=pl.col("first_tick_ts"))
        .select(BAR_COLUMNS)
        .collect()
    )
    if drop_last and bars.height and bars["n_ticks"][-1] < n:
        bars = bars.head(bars.height - 1)
    return bars


def resample_bars(bars: pl.DataFrame | pl.LazyFrame, every: str = "5m") -> pl.DataFrame:
    """Aggregate existing bars up to a coarser interval.

    Grouping is on ``ts_open``, not on ``ts``: with right-edge labels the 1-minute
    bars belonging to the 5-minute window ``[10:00, 10:05)`` are labelled 10:01
    through 10:05, so grouping on the label would slice the window one bar off.

    The result is identical to building the coarser bars from ticks directly,
    provided ``every`` is a whole multiple of the source interval - which is
    what :func:`~tests.test_bars` pins. Prefer this over re-reading ticks; it is
    three orders of magnitude cheaper.
    """
    lazy = bars.lazy().sort("ts_open")
    return (
        lazy.group_by_dynamic("ts_open", every=every, closed="left", label="right")
        .agg(
            first_tick_ts=pl.col("first_tick_ts").first(),
            last_tick_ts=pl.col("last_tick_ts").last(),
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            # Tick-weighted, so that re-aggregating matches building from ticks.
            # A plain mean of means would weight a 2-tick minute like a 500-tick one.
            mid_mean=(pl.col("mid_mean") * pl.col("n_ticks")).sum()
            / pl.col("n_ticks").sum(),
            bid_close=pl.col("bid_close").last(),
            ask_close=pl.col("ask_close").last(),
            spread_mean=(pl.col("spread_mean") * pl.col("n_ticks")).sum()
            / pl.col("n_ticks").sum(),
            spread_close=pl.col("spread_close").last(),
            n_ticks=pl.col("n_ticks").sum(),
        )
        .rename({"ts_open": "ts"})
        .with_columns(ts_open=pl.col("ts").dt.offset_by(f"-{every}"))
        .select(BAR_COLUMNS)
        .collect()
    )
