"""The event-study layer.

The assertions that matter are the ones about *cost* and *time*. Cost, because
the whole point of the module is that the mid column and the net column differ
by exactly one round turn and no more - if it ever charged the spread twice, or
forgot the commission, every conclusion drawn from it would move. Time, because
a forward window that quietly spans a weekend hands a strategy a gap it was
never exposed to, and that failure flatters rather than breaks a result.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qlab.costs import NO_SLIPPAGE, CostModel, SlippageModel
from qlab.eventstudy import (
    daily_pnl,
    deflated_threshold,
    forward_returns,
    placebo_events,
    summarize,
)
from qlab.symbols import get_spec

UTC = timezone.utc
START = datetime(2024, 6, 3, 8, 0, tzinfo=UTC)


def _model(slippage: SlippageModel | None = None) -> CostModel:
    """A flat XAUUSD model: 1.0 pip of spread, 2.0 pips of drift, everywhere."""
    hours = list(range(24))
    return CostModel(
        spec=get_spec("XAUUSD"),
        slippage=slippage if slippage is not None else SlippageModel(),
        spread_by_hour=pl.DataFrame(
            {
                "hour": hours,
                "is_sunday": [False] * 24,
                "n_ticks": [1000] * 24,
                "spread_mean_pips": [1.0] * 24,
                "spread_p95_pips": [2.0] * 24,
            }
        ),
        drift_by_hour=pl.DataFrame(
            {
                "hour": hours,
                "n_anchors": [1000] * 24,
                "drift_mean_pips": [2.0] * 24,
                "drift_p95_pips": [6.0] * 24,
            }
        ),
    )


def _bars(closes: list[float], *, minutes: int = 1, skip: int = 0) -> pl.DataFrame:
    """Bars at a fixed spread of 1.0 pip ($0.01) around each close.

    ``skip`` drops that many minutes into the middle of the series, which is how
    a maintenance break or a weekend looks to the loader.
    """
    stamps = []
    offset = 0
    for i in range(len(closes)):
        if i == len(closes) // 2:
            offset += skip
        stamps.append(START + timedelta(minutes=i * minutes + offset))
    return pl.DataFrame(
        {
            "ts": [s + timedelta(minutes=minutes) for s in stamps],
            "ts_open": stamps,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "bid_close": [c - 0.005 for c in closes],
            "ask_close": [c + 0.005 for c in closes],
            "spread_mean": [0.01] * len(closes),
            "n_ticks": [50] * len(closes),
        },
        schema_overrides={
            "ts": pl.Datetime("us", "UTC"),
            "ts_open": pl.Datetime("us", "UTC"),
        },
    )


def _events(bars: pl.DataFrame, index: int, direction: int) -> pl.DataFrame:
    return bars.select("ts").slice(index, 1).with_columns(
        direction=pl.lit(direction, pl.Int8)
    )


def test_the_mid_return_is_the_signed_move():
    bars = _bars([2000.0, 2000.0, 2002.0])
    out = forward_returns(bars, _events(bars, 0, 1), (2,), cost=_model())
    assert out.height == 1
    assert out["mid_bps"][0] == pytest.approx((2002 / 2000 - 1) * 10_000)


def test_a_short_is_the_same_move_with_the_sign_flipped():
    bars = _bars([2000.0, 2000.0, 2002.0])
    long = forward_returns(bars, _events(bars, 0, 1), (2,), cost=_model())
    short = forward_returns(bars, _events(bars, 0, -1), (2,), cost=_model())
    assert short["mid_bps"][0] == pytest.approx(-long["mid_bps"][0])
    # But both pay: reversing a signal cannot reverse its cost.
    assert short["net_bps"][0] < short["mid_bps"][0]
    assert long["net_bps"][0] < long["mid_bps"][0]


def test_mid_minus_net_is_exactly_one_round_turn():
    """The assertion the whole module exists to keep true.

    Spread arrives through the bid/ask fills, commission and slippage are added
    on top, and the total has to reconcile to :meth:`CostModel.round_turn_bps`
    at the same price. A double-charged spread would show up here and nowhere
    else.
    """
    price = 2000.0
    bars = _bars([price, price, price])
    cost = _model()
    out = forward_returns(bars, _events(bars, 0, 1), (2,), cost=cost)

    to_bps = cost.spec.pip / price * 10_000
    expected = cost.round_turn_bps(price=price)
    assert out["mid_bps"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["mid_bps"][0] - out["net_bps"][0] == pytest.approx(expected, rel=1e-6)
    # cost_bps holds commission and slippage only; the spread is already inside
    # the fills, and adding it here as well is the double charge being guarded
    # against.
    assert out["cost_bps"][0] == pytest.approx(
        expected - cost.spread_pips() * to_bps, rel=1e-6
    )


def test_turning_slippage_off_moves_net_to_the_fill_column():
    bars = _bars([2000.0, 2000.0, 2000.0])
    out = forward_returns(bars, _events(bars, 0, 1), (2,), cost=_model(NO_SLIPPAGE))
    assert out["net_bps"][0] == pytest.approx(out["fill_bps"][0])


def test_a_horizon_spanning_a_gap_is_dropped():
    """The trap: without this, a Friday event is scored on Sunday's reopen."""
    bars = _bars([2000.0] * 6, skip=3000)  # a two-day hole in the middle
    events = pl.concat(
        [_events(bars, i, 1) for i in range(4)], how="vertical_relaxed"
    )
    out = forward_returns(bars, events, (2,), cost=_model())
    # Six bars with the hole after the third. At a horizon of 2 only the first
    # event lands inside the first block and only the fourth inside the second;
    # the two whose windows straddle the hole are dropped rather than scored on
    # a move no position was open across.
    assert out.height == 2
    assert out["ts"].to_list() == [bars["ts"][0], bars["ts"][3]]


def test_horizons_are_measured_in_bars_not_in_rows():
    """Five-minute bars mean a horizon of 2 is ten minutes, and it is labelled so."""
    bars = _bars([2000.0] * 5, minutes=5)
    out = forward_returns(bars, _events(bars, 0, 1), (2,), cost=_model())
    assert out["minutes"][0] == 10


def test_daily_pnl_totals_the_day_rather_than_averaging_it():
    """Three events on one day is one observation worth three events of P&L.

    The distinction is the correction the study depends on: averaging within the
    day would understate a day that fired often, and treating each event as an
    observation would count a trending afternoon many times over.
    """
    bars = _bars([2000.0, 2001.0, 2002.0, 2003.0])
    events = pl.concat(
        [_events(bars, i, 1) for i in range(3)], how="vertical_relaxed"
    )
    out = forward_returns(bars, events, (1,), cost=_model())
    per_day, _, days = daily_pnl(out)
    assert days == 1
    assert per_day == pytest.approx(out["net_bps"].sum())


def test_summarize_reports_the_cost_the_mean_had_to_clear():
    bars = _bars([2000.0] * 4)
    events = pl.concat([_events(bars, i, 1) for i in range(2)], how="vertical_relaxed")
    out = forward_returns(bars, events, (1,), cost=_model())
    table = summarize(out)
    assert table["n"][0] == 2
    assert table["cost"][0] > 0


def test_the_placebo_keeps_the_time_of_day_and_loses_the_day():
    bars = _bars([2000.0] * (60 * 24 * 12))
    events = pl.concat(
        [_events(bars, i * 500, 1) for i in range(20)], how="vertical_relaxed"
    )
    fake = placebo_events(events, bars, seed=3)
    assert fake.height > 0
    assert set(fake["ts"].dt.time()).issubset(set(events["ts"].dt.time()))
    # And it must actually move: a placebo that lands on the event is no control.
    assert not set(fake["ts"]).intersection(set(events["ts"]))


def test_the_deflated_threshold_rises_with_the_number_of_trials():
    assert deflated_threshold(1) == pytest.approx(1.96, abs=0.01)
    assert deflated_threshold(200) > deflated_threshold(20) > deflated_threshold(1)


def test_no_events_is_an_empty_answer_not_a_crash():
    bars = _bars([2000.0] * 4)
    empty = bars.select("ts").clear().with_columns(direction=pl.lit(1, pl.Int8))
    assert forward_returns(bars, empty, (1,), cost=_model()).is_empty()
