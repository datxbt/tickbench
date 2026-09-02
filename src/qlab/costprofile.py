"""Measuring the two cost components the contract specification does not give you.

The broker publishes a commission, so commission is arithmetic. Spread and
slippage have to be measured, and they are measured here, once, into two tables
that :mod:`qlab.costs` reads.

**Spread profile.** The published average is a single previous-trading-day
number, and Stage 0 showed the real thing varies by an order of magnitude across
the hours of the day and has compressed steadily since 2020. So the profile is
cut by symbol, month and UTC hour, with the Sunday reopen kept separate because
it is a different regime rather than a wide hour.

**Latency profile.** How far the mid moves between deciding and being filled.
This is the part no specification covers and the part a spread-plus-commission
cost model silently sets to zero. It is measured directly: sample an anchor
tick, look up the prevailing mid a fixed wall-clock horizon later, take the
absolute difference.

Two things about that measurement are worth stating, because they bound what it
can be used for:

- It is |move|, not signed move. Over a large sample the signed mean is
  approximately zero, since price is close to a martingale at this horizon. The
  cost to a strategy depends on how its fills correlate with the move -
  a breakout order chases and eats most of it, a passive one need not - which is
  why :class:`qlab.costs.SlippageModel` exposes that correlation as an explicit
  fraction rather than burying an assumption here.
- A horizon during which no tick arrives contributes a genuine zero: the price
  did not move, so a fill at that moment gets the same price. On EURUSD that is
  most of the sample at short horizons, which is why the mean matters more than
  the median.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from .symbols import SymbolSpec

# Retail market execution over a consumer connection is usually tens to a few
# hundred milliseconds. The range is measured rather than assumed so that the
# cost of being slow is visible, and so a VPS-colocated setup and a home
# connection can be priced separately.
HORIZONS_MS: tuple[int, ...] = (50, 100, 250, 500, 1000)

# Anchors per month. The quantity being estimated is a distribution, not a sum,
# so a sample is sufficient and full coverage would cost hours for no accuracy.
TARGET_ANCHORS = 150_000

# An anchor whose next tick is further away than this is sitting at a session
# edge - a weekend, a maintenance break - where nothing would be traded anyway.
SESSION_EDGE_S = 60.0


def spread_profile(bars: pl.DataFrame, spec: SymbolSpec) -> pl.DataFrame:
    """Spread by UTC hour for one symbol-month, in pips.

    Weighted by ``n_ticks`` for the mean, because a minute holding 400 quotes is
    400 quotes' worth of evidence about the spread and a minute holding 2 is not.
    The quantiles are deliberately *unweighted over bars*: they answer "what
    spread does a bar typically show", which is the question a strategy trading
    on bar closes is actually asking.
    """
    if bars.is_empty():
        return pl.DataFrame()

    return (
        bars.with_columns(
            spread_pips=pl.col("spread_mean") / spec.pip,
            hour=pl.col("ts_open").dt.hour(),
            is_sunday=pl.col("ts_open").dt.weekday() == 7,
        )
        .group_by("hour", "is_sunday")
        .agg(
            n_bars=pl.len(),
            n_ticks=pl.col("n_ticks").sum(),
            spread_mean_pips=(pl.col("spread_pips") * pl.col("n_ticks")).sum()
            / pl.col("n_ticks").sum(),
            spread_p50_pips=pl.col("spread_pips").quantile(0.50),
            spread_p95_pips=pl.col("spread_pips").quantile(0.95),
            spread_max_pips=pl.col("spread_pips").max(),
        )
        .with_columns(symbol=pl.lit(spec.name))
        .sort("hour", "is_sunday")
    )


def latency_profile(
    ticks: pl.DataFrame,
    spec: SymbolSpec,
    *,
    horizons_ms: tuple[int, ...] = HORIZONS_MS,
    target_anchors: int = TARGET_ANCHORS,
) -> pl.DataFrame:
    """Absolute mid drift over each latency horizon, by UTC hour, in pips."""
    if ticks.height < 2:
        return pl.DataFrame()

    enriched = ticks.select(
        "ts",
        mid=(pl.col("bid") + pl.col("ask")) / 2,
    ).with_columns(
        next_gap_s=(pl.col("ts").shift(-1) - pl.col("ts")).dt.total_milliseconds() / 1000.0
    )
    forward = enriched.select("ts", pl.col("mid").alias("mid_fwd"))

    stride = max(1, enriched.height // target_anchors)
    anchors = (
        enriched.gather_every(stride)
        .filter(pl.col("next_gap_s").fill_null(1e9) <= SESSION_EDGE_S)
        .select("ts", "mid", hour=pl.col("ts").dt.hour())
    )
    if anchors.is_empty():
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for horizon in horizons_ms:
        drift = (
            anchors.with_columns(ts_h=pl.col("ts") + timedelta(milliseconds=horizon))
            .sort("ts_h")
            # backward: the prevailing quote at t+horizon, which is the price a
            # fill arriving then would receive.
            .join_asof(forward, left_on="ts_h", right_on="ts", strategy="backward")
            .with_columns(
                drift_pips=((pl.col("mid_fwd") - pl.col("mid")).abs() / spec.pip)
            )
            .drop_nulls("drift_pips")
        )
        frames.append(
            drift.group_by("hour")
            .agg(
                n_anchors=pl.len(),
                drift_mean_pips=pl.col("drift_pips").mean(),
                drift_p50_pips=pl.col("drift_pips").quantile(0.50),
                drift_p95_pips=pl.col("drift_pips").quantile(0.95),
                # The share of horizons in which no quote arrived at all. High on
                # the majors, near zero on the index - it is the clearest single
                # measure of how fast an instrument's book actually moves.
                zero_share=(pl.col("drift_pips") == 0).mean(),
            )
            .with_columns(symbol=pl.lit(spec.name), horizon_ms=pl.lit(horizon, pl.Int32))
        )

    return pl.concat(frames).sort("horizon_ms", "hour")
