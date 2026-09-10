"""Tests for the overnight drift.

The arithmetic here is simple and the *alignment* is not, so the tests are
almost all about alignment:

* A close-to-open leg reads yesterday's close and today's open. Off by one row
  and it becomes a strategy that trades on tomorrow's information.
* "Yesterday" means the previous **session**, not the previous calendar day, so
  a Monday's predecessor is a Friday and a holiday has no predecessor at all.
* The session close is the last quote *before* 16:00, not the first one after
  it - USTEC stops quoting at the cash close and the next print belongs to the
  evening session.
* A conditional strategy is flat, not absent, on the days it does not trade.
  Dropping those rows turns a strategy that trades half the time into one that
  only exists on its good days.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import qlab.strategies.overnight_drift as od
from qlab.strategies.overnight_drift import (
    CLOSE_MARK,
    OPEN_MARK,
    STRATEGIES,
    Conditional,
    Leg,
    Mark,
    closing_imbalance,
    hourly_returns,
    leg_return,
    minute_marks,
    price_panel,
    sort_by_signal,
    summarize,
)


# --------------------------------------------------------------------------
# Marks
# --------------------------------------------------------------------------

def test_mark_kind_is_validated():
    with pytest.raises(ValueError):
        Mark(600, "midpoint")


def test_mark_keys_distinguish_the_two_sides_of_a_minute():
    assert Mark(960, "open").key == "960"
    assert Mark(960, "close").key == "960c"


def test_the_session_close_is_a_close_mark():
    """Not cosmetic: USTEC has no bar starting at 16:00 New York, so an open
    mark there would silently drop most of the sample."""
    assert CLOSE_MARK.kind == "close"
    assert OPEN_MARK.kind == "open"
    assert STRATEGIES["CTC"].start is CLOSE_MARK
    assert STRATEGIES["OTC"].end is CLOSE_MARK


def test_the_drift_window_is_the_papers_hour():
    assert STRATEGIES["OD"].start.minute == 120
    assert STRATEGIES["OD"].end.minute == 180
    assert STRATEGIES["OD+"].start.minute == 90
    assert STRATEGIES["OD+"].end.minute == 210


# --------------------------------------------------------------------------
# A synthetic tape whose answer is known
# --------------------------------------------------------------------------

def bars_for(days, *, minutes=range(0, 24 * 60), price=None, spread=0.0,
             ticks=None):
    """One-minute bars over a set of New York dates.

    January, so New York is UTC-5 and a New York minute ``m`` is UTC ``m + 300``.
    ``price(day_index, minute)`` returns the bar's open; its close is the next
    minute's open, so a window return is exactly the difference the caller
    wrote.
    """
    price = price or (lambda d, m: 100.0)
    ticks = ticks or (lambda d, m: 10)
    rows = []
    for d, day in enumerate(days):
        for m in minutes:
            ts_open = datetime(day.year, day.month, day.day,
                               tzinfo=timezone.utc) + timedelta(minutes=m + 300)
            rows.append({
                "ts": ts_open + timedelta(minutes=1),
                "ts_open": ts_open,
                "open": price(d, m),
                "close": price(d, m + 1),
                "spread_mean": spread,
                "spread_close": spread,
                "n_ticks": ticks(d, m),
            })
    return pl.DataFrame(rows).with_columns(pl.col("n_ticks").cast(pl.UInt32))


DAYS = [datetime(2026, 1, 5) + timedelta(days=i) for i in range(4)]


def test_open_mark_takes_the_bar_that_begins_at_the_minute():
    bars = bars_for(DAYS, minutes=range(115, 125),
                    price=lambda d, m: 100.0 + m)
    marks = minute_marks("USTEC", [Mark(120)], bars=bars)
    assert marks.height == len(DAYS)
    assert marks["mid"].unique().to_list() == [220.0]


def test_close_mark_takes_the_last_bar_before_the_minute():
    """16:00 itself is not in the sample; 15:59 is, and its close is the day's
    last price."""
    bars = bars_for(DAYS, minutes=range(950, 960), price=lambda d, m: 100.0 + m)
    marks = minute_marks("USTEC", [CLOSE_MARK], bars=bars)
    assert marks.height == len(DAYS)
    # the 15:59 bar opens at 1059 and closes at 1060
    assert marks["mid"].unique().to_list() == [1060.0]


def test_a_mark_with_no_bar_nearby_produces_no_row():
    bars = bars_for(DAYS, minutes=range(0, 60))
    marks = minute_marks("USTEC", [Mark(120)], bars=bars)
    assert marks.is_empty()


def test_half_spread_is_half_the_spread():
    bars = bars_for(DAYS, minutes=range(115, 125), spread=0.4)
    marks = minute_marks("USTEC", [Mark(120)], bars=bars)
    assert marks["half_spread"].unique().to_list() == [0.2]


# --------------------------------------------------------------------------
# Leg returns
# --------------------------------------------------------------------------

def panel_for(bars, marks=None):
    marks = marks or [Mark(120), Mark(180), CLOSE_MARK, OPEN_MARK]
    return price_panel(minute_marks("USTEC", marks, bars=bars))


def test_intraday_leg_is_the_windows_own_return():
    bars = bars_for(DAYS, minutes=range(110, 190),
                    price=lambda d, m: 100.0 * (1.0 + 0.001 * (m - 110)))
    panel = panel_for(bars)
    got = panel.select(leg_return(panel, STRATEGIES["OD"]).alias("r"))["r"]
    # 100*(1+0.001*70) over 100*(1+0.001*10) - 1 = 5.94%
    assert got.drop_nulls().to_list() == pytest.approx([594.0594] * 4, rel=1e-4)


def test_overnight_leg_reads_yesterdays_close_and_todays_open():
    """Prices are flat within a day and step up between days, so a close-to-open
    leg earns the whole step and an open-to-close leg earns nothing."""
    bars = bars_for(DAYS, minutes=list(range(565, 575)) + list(range(950, 960)),
                    price=lambda d, m: 100.0 * 1.01 ** d)
    panel = panel_for(bars)
    cto = panel.select(leg_return(panel, STRATEGIES["CTO"]).alias("r"))["r"]
    otc = panel.select(leg_return(panel, STRATEGIES["OTC"]).alias("r"))["r"]
    assert cto[0] is None                       # no predecessor for day one
    assert cto.drop_nulls().to_list() == pytest.approx([100.0] * 3, rel=1e-6)
    assert otc.to_list() == pytest.approx([0.0] * 4)


def test_previous_session_means_the_previous_row_not_the_previous_date():
    """A gap in the calendar - a weekend or a holiday - must not become a
    two-day return attributed to one session."""
    sparse = [datetime(2026, 1, 5), datetime(2026, 1, 6), datetime(2026, 1, 12)]
    bars = bars_for(sparse, minutes=list(range(565, 575)) + list(range(950, 960)),
                    price=lambda d, m: 100.0 * 1.01 ** d)
    panel = panel_for(bars).sort("nyd")
    cto = panel.select(leg_return(panel, STRATEGIES["CTO"]).alias("r"))["r"]
    # Every step is one *session*, so both legs are the same size even though
    # the second spans six calendar days.
    assert cto.drop_nulls().to_list() == pytest.approx([100.0] * 2, rel=1e-6)


def test_net_leg_buys_the_ask_and_sells_the_bid():
    bars = bars_for(DAYS, minutes=range(110, 190), price=lambda d, m: 100.0,
                    spread=0.2)
    panel = panel_for(bars)
    gross = panel.select(leg_return(panel, STRATEGIES["OD"]).alias("r"))["r"]
    net = panel.select(leg_return(panel, STRATEGIES["OD"], net=True).alias("r"))["r"]
    assert gross.to_list() == pytest.approx([0.0] * 4)
    # buy 100.1, sell 99.9, on a flat market: -19.98 bps
    assert net.to_list() == pytest.approx([-19.98] * 4, rel=1e-3)


def test_a_leg_is_never_free_when_the_spread_is_not_zero():
    bars = bars_for(DAYS, minutes=range(110, 190), price=lambda d, m: 100.0,
                    spread=0.2)
    panel = panel_for(bars)
    for leg in (STRATEGIES["OD"], STRATEGIES["OD+"]):
        if f"mid_{leg.end.key}" not in panel.columns:
            continue
        net = panel.select(leg_return(panel, leg, net=True).alias("r"))["r"]
        assert (net.drop_nulls() < 0).all()


# --------------------------------------------------------------------------
# The conditioning variable
# --------------------------------------------------------------------------

def test_imbalance_is_bounded_and_signed_by_the_bars_own_return():
    """Every closing minute down: the proxy pins at -1."""
    bars = bars_for(DAYS, minutes=range(900, 960),
                    price=lambda d, m: 1000.0 - m)
    out = closing_imbalance("USTEC", bars=bars)
    assert out["rsv"].to_list() == pytest.approx([-1.0] * 4)
    assert (out["close_ret_bps"] < 0).all()


def test_imbalance_weights_by_activity_not_by_minutes():
    """One heavily quoted up-minute outweighs many quiet down-minutes."""
    bars = bars_for(
        [DAYS[0]], minutes=range(900, 960),
        # Every minute drifts down by a hundredth except the 15:30 bar, which
        # jumps five and carries a thousand times the quote traffic.
        price=lambda d, m: 100.0 + (5.0 if m > 930 else 0.0) - 0.01 * m,
        ticks=lambda d, m: 1000 if m == 930 else 1,
    )
    out = closing_imbalance("USTEC", bars=bars)
    # One up-minute carrying 1000 quotes against 59 down-minutes carrying one
    # each: (1000 - 59) / (1000 + 59).
    assert out["rsv"].item() == pytest.approx(941.0 / 1059.0)


def test_imbalance_drops_a_short_session():
    bars = bars_for(DAYS, minutes=range(950, 960))
    assert closing_imbalance("USTEC", bars=bars).is_empty()


# --------------------------------------------------------------------------
# Hourly returns
# --------------------------------------------------------------------------

def test_hourly_returns_drop_stub_hours():
    """An hour with twenty bars is not an hour. The daily maintenance break and
    holiday half-sessions arrive as exactly this."""
    bars = bars_for(DAYS, minutes=list(range(120, 180)) + list(range(240, 260)))
    out = hourly_returns("USTEC", bars=bars)
    assert set(out["hour"].to_list()) == {2}


def test_hourly_return_is_the_hours_log_return():
    bars = bars_for(DAYS, minutes=range(120, 181),
                    price=lambda d, m: 100.0 * np.exp(0.0001 * (m - 120)))
    out = hourly_returns("USTEC", bars=bars)
    assert out["ret_bps"].to_list() == pytest.approx([60.0] * 4, rel=1e-3)


# --------------------------------------------------------------------------
# Conditionals and reporting
# --------------------------------------------------------------------------

def test_conditional_mask_reads_the_previous_session():
    cond = Conditional("BtD", "OD+", "rsv", below=0.0)
    frame = pl.DataFrame({"prev_rsv": [-0.5, 0.5, None, -0.1]})
    got = frame.select(cond.mask().alias("m"))["m"].fill_null(False).to_list()
    assert got == [True, False, False, True]


def test_conditional_is_flat_not_absent_on_the_days_it_skips():
    """Averaging a conditional over only its active days is the classic way to
    turn a half-time strategy into a full-time one on paper."""
    returns = pl.DataFrame({
        "BtD": [2.0, 0.0, 0.0, 4.0],
        "BtD_active": [True, False, False, True],
    })
    stats = summarize(returns, ["BtD"]).row(0, named=True)
    assert stats["n_days"] == 4
    assert stats["days_active"] == 2
    assert stats["mean_bps"] == pytest.approx(1.5)


def test_summarize_annualises_over_252_days():
    returns = pl.DataFrame({"OD": [1.0] * 100})
    stats = summarize(returns, ["OD"]).row(0, named=True)
    assert stats["ann_pct"] == pytest.approx(252.0 / 1e4 * 100)
    assert not np.isfinite(stats["sharpe"])       # zero variance


def test_sort_is_monotone_in_the_signal_and_partitions_the_sample():
    rng = np.random.default_rng(0)
    n = 500
    signal = rng.normal(size=n)
    frame = pl.DataFrame({"prev_rsv": signal, "OD+": -signal})
    out = sort_by_signal(frame, column="prev_rsv", target="OD+")
    assert out["n"].sum() == n
    assert out["signal_mean"].is_sorted()
    # The target is minus the signal by construction, so the sort must invert.
    assert out["mean_bps"][0] > out["mean_bps"][-1]


def test_a_leg_can_be_declared_without_touching_the_defaults():
    custom = Leg("X", Mark(60), Mark(90))
    assert custom.marks == (Mark(60), Mark(90))
    assert "X" not in STRATEGIES
