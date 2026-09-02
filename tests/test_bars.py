"""Bar construction, with the lookahead question asked directly.

Every test here is ultimately the same test: can a value in row *i* have come
from a tick that had not happened yet at row *i*'s timestamp? That bug does not
raise, does not look wrong in a plot, and makes a backtest better rather than
worse - which is why it gets this much of the suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qlab.bars import resample_bars, tick_bars, time_bars

UTC = timezone.utc


def _ticks(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime.fromisoformat(r[0]).replace(tzinfo=UTC) for r in rows],
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        },
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
    )


def _ramp(n: int, *, start: str = "2024-06-03T00:00:00", step_s: int = 10):
    """n ticks at a fixed cadence, price climbing by one pip per tick."""
    base = datetime.fromisoformat(start).replace(tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [base + timedelta(seconds=step_s * i) for i in range(n)],
            "bid": [1.0800 + 0.0001 * i for i in range(n)],
            "ask": [1.0801 + 0.0001 * i for i in range(n)],
        },
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
    )


def test_bars_are_labelled_at_the_interval_end():
    """The convention, pinned: ts is when the bar became knowable."""
    bars = time_bars(_ramp(120, step_s=10), "1m")

    assert bars["ts"][0] == datetime(2024, 6, 3, 0, 1, tzinfo=UTC)
    assert bars["ts_open"][0] == datetime(2024, 6, 3, 0, 0, tzinfo=UTC)
    # Every tick in a bar strictly precedes the bar's own timestamp. This is the
    # property that makes acting on row i at row i's ts free of lookahead.
    assert (bars["last_tick_ts"] < bars["ts"]).all()
    assert (bars["first_tick_ts"] >= bars["ts_open"]).all()


def test_the_window_is_half_open_at_the_boundary():
    """A tick at exactly 00:01:00 opens the next bar; it does not close this one."""
    ticks = _ticks(
        [
            ("2024-06-03T00:00:30", 1.0800, 1.0802),
            ("2024-06-03T00:01:00", 1.0900, 1.0902),  # exactly on the edge
        ]
    )
    bars = time_bars(ticks, "1m")

    assert bars.height == 2
    assert bars["close"][0] == pytest.approx(1.0801)
    assert bars["open"][1] == pytest.approx(1.0901)


def test_ohlc_comes_from_the_mid_and_the_close_quote_is_kept_separately():
    ticks = _ticks(
        [
            ("2024-06-03T00:00:05", 1.0800, 1.0802),  # mid 1.0801
            ("2024-06-03T00:00:15", 1.0810, 1.0814),  # mid 1.0812, high
            ("2024-06-03T00:00:25", 1.0794, 1.0796),  # mid 1.0795, low
            ("2024-06-03T00:00:35", 1.0805, 1.0809),  # mid 1.0807, close
        ]
    )
    bar = time_bars(ticks, "1m").row(0, named=True)

    assert bar["open"] == pytest.approx(1.0801)
    assert bar["high"] == pytest.approx(1.0812)
    assert bar["low"] == pytest.approx(1.0795)
    assert bar["close"] == pytest.approx(1.0807)
    # You buy at the ask and sell at the bid, so the close alone cannot price a
    # fill - both sides survive aggregation.
    assert bar["bid_close"] == pytest.approx(1.0805)
    assert bar["ask_close"] == pytest.approx(1.0809)
    assert bar["spread_close"] == pytest.approx(0.0004)
    assert bar["n_ticks"] == 4


def test_empty_intervals_are_missing_rather_than_invented():
    """A flat bar over a market closure is a price assertion with no evidence."""
    ticks = _ticks(
        [
            ("2024-06-03T00:00:30", 1.0800, 1.0802),
            ("2024-06-03T00:04:30", 1.0900, 1.0902),  # three empty minutes
        ]
    )
    bars = time_bars(ticks, "1m")

    assert bars.height == 2
    assert bars["ts"].to_list() == [
        datetime(2024, 6, 3, 0, 1, tzinfo=UTC),
        datetime(2024, 6, 3, 0, 5, tzinfo=UTC),
    ]


def test_drop_last_removes_the_truncated_interval():
    bars = time_bars(_ramp(90, step_s=10), "1m")
    truncated = time_bars(_ramp(90, step_s=10), "1m", drop_last=True)

    assert bars.height == truncated.height + 1
    assert truncated["ts"].to_list() == bars["ts"].to_list()[:-1]


def test_resampling_bars_matches_building_them_from_ticks():
    """Re-aggregation has to be exact, or the cheap path and the true path diverge."""
    ticks = _ramp(600, step_s=10)  # 100 minutes
    direct = time_bars(ticks, "5m")
    resampled = resample_bars(time_bars(ticks, "1m"), "5m")

    assert resampled.columns == direct.columns
    assert resampled.height == direct.height
    for column in direct.columns:
        if direct[column].dtype.is_numeric():
            # mid_mean and spread_mean are weighted sums of sums, so they agree
            # to floating point rather than bit-for-bit.
            assert (resampled[column] - direct[column]).abs().max() < 1e-12
        else:
            assert resampled[column].to_list() == direct[column].to_list()


def test_tick_bars_hold_exactly_n_ticks_and_drop_the_partial_tail():
    """The trailing stub is not a bar; keeping it fakes a volatility change."""
    bars = tick_bars(_ramp(250, step_s=1), n=100)

    assert bars.height == 2
    assert bars["n_ticks"].to_list() == [100, 100]
    # ts is the last tick of the bar - the instant it completed - so the
    # right-edge convention survives the change of sampling scheme.
    assert (bars["ts"] == bars["last_tick_ts"]).all()
    assert (bars["ts_open"] <= bars["ts"]).all()

    assert tick_bars(_ramp(250, step_s=1), n=100, drop_last=False).height == 3


def test_tick_bars_reject_a_meaningless_size():
    with pytest.raises(ValueError, match="at least 1"):
        tick_bars(_ramp(10), n=0)


def test_unsorted_input_does_not_corrupt_the_bars():
    """Order is a Stage 0 invariant, but a bar builder that quietly trusts it
    would turn a violated invariant into wrong prices rather than an error."""
    ticks = _ramp(120, step_s=10)
    shuffled = ticks.sample(fraction=1.0, shuffle=True, seed=7)

    assert time_bars(shuffled, "1m").equals(time_bars(ticks, "1m"))
