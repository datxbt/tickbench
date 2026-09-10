"""Tests for the overnight basis measurement.

The arithmetic is checked on a synthetic tape where the answer is known by
construction, because the whole value of this module is that its sign is
trusted - a sign error would have turned a rejection into a deployment.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab import rollover
from qlab.loader import SplitLockedError


# --------------------------------------------------------------------------
# A synthetic tape, so the expected basis can be written down in advance
# --------------------------------------------------------------------------

def _tape(n_days=60, drift_bps=0.0, half_spread=0.00001, start=date(2021, 3, 1)):
    """A tape quoting one minute at the entry and one at the exit, per weekday.

    Price is flat except for a fixed drift applied between the entry minute and
    the exit minute, so the measured basis has to come back as that drift.
    """
    rows = []
    px = 1.10
    for i in range(n_days):
        day = start + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for minute, price in (
            (rollover.DEFAULT_ENTRY_UTC_MIN, px),
            (rollover.DEFAULT_EXIT_UTC_MIN, px * (1 + drift_bps / 1e4)),
        ):
            ts_open = datetime(day.year, day.month, day.day,
                               minute // 60, minute % 60, tzinfo=timezone.utc)
            rows.append({
                "ts": ts_open + timedelta(minutes=1),
                "ts_open": ts_open,
                "close": price,
                "bid_close": price - half_spread,
                "ask_close": price + half_spread,
            })
    return pl.DataFrame(rows)


def _patched(monkeypatch, frame):
    monkeypatch.setattr(rollover, "load_bars", lambda *a, **k: frame)


def test_basis_recovers_a_known_drift(monkeypatch):
    _patched(monkeypatch, _tape(drift_bps=3.0, half_spread=0.0))
    b = rollover.basis("EURUSD")
    assert b.mid_bps == pytest.approx(3.0, abs=1e-6)
    assert b.tradable_bps == pytest.approx(3.0, abs=1e-6)


def test_a_flat_tape_has_no_basis(monkeypatch):
    _patched(monkeypatch, _tape(drift_bps=0.0, half_spread=0.0))
    b = rollover.basis("EURUSD")
    assert b.mid_bps == pytest.approx(0.0, abs=1e-9)


def test_the_spread_is_paid_in_the_tradable_number_but_not_the_mid(monkeypatch):
    """Buying the ask and selling the bid costs the full spread, once."""
    half = 0.00005                      # 0.5 pip either side of a 1.10 mid
    _patched(monkeypatch, _tape(drift_bps=0.0, half_spread=half))
    b = rollover.basis("EURUSD")
    expected = -2 * half / 1.10 * 1e4
    assert b.mid_bps == pytest.approx(0.0, abs=1e-9)
    assert b.tradable_bps == pytest.approx(expected, rel=1e-3)
    assert b.artifact_bps == pytest.approx(-expected, rel=1e-3)


def test_commission_is_subtracted_once(monkeypatch):
    _patched(monkeypatch, _tape(drift_bps=5.0, half_spread=0.0))
    b = rollover.basis("EURUSD")
    # EURUSD: $2.50 a side on 100,000, so 0.5 pips round turn = 0.4545 bps at 1.10
    assert b.tradable_bps - b.net_bps == pytest.approx(0.5 * 1e-4 / 1.10 * 1e4, rel=1e-3)


def test_a_long_that_receives_carry_is_predicted_to_drift_down():
    """The prediction table is the thing the measurement is tested against, so
    a sign flip in it would silently invert every conclusion."""
    assert rollover.CARRY_SIGN_ON_LONG["USDJPY"] == +1     # receives
    assert rollover.CARRY_SIGN_ON_LONG["EURUSD"] == -1     # pays
    assert rollover.CARRY_SIGN_ON_LONG["XAUUSD"] == -1     # pays


def test_weekend_rows_are_excluded(monkeypatch):
    frame = _tape(drift_bps=1.0, half_spread=0.0)
    _patched(monkeypatch, frame)
    b = rollover.basis("EURUSD")
    weekdays = frame.filter(
        pl.col("ts_open").dt.weekday() <= 5
    ).select(pl.col("ts_open").dt.date().n_unique()).item()
    assert b.n == weekdays


def test_an_exit_before_the_entry_is_read_as_the_next_day(monkeypatch):
    """Exit at 00:30 UTC belongs to the day the position was opened, not the
    calendar day the clock rolled into."""
    entry, exit_ = 20 * 60 + 55, 30
    rows = []
    for i in range(40):
        day = date(2021, 3, 1) + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        rows.append({"ts": None, "ts_open": datetime(day.year, day.month, day.day,
                                                     entry // 60, entry % 60,
                                                     tzinfo=timezone.utc),
                     "close": 1.10, "bid_close": 1.10, "ask_close": 1.10})
        nxt = day + timedelta(days=1)
        rows.append({"ts": None, "ts_open": datetime(nxt.year, nxt.month, nxt.day,
                                                     exit_ // 60, exit_ % 60,
                                                     tzinfo=timezone.utc),
                     "close": 1.10 * 1.0002, "bid_close": 1.10 * 1.0002,
                     "ask_close": 1.10 * 1.0002})
    _patched(monkeypatch, pl.DataFrame(rows))
    b = rollover.basis("EURUSD", entry_min=entry, exit_min=exit_)
    assert b.n > 20
    assert b.mid_bps == pytest.approx(2.0, abs=1e-3)


# --------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------

def test_the_carry_prediction_holds_on_every_quoting_instrument():
    """The finding the EURUSD rejection rests on, asserted rather than described.

    If this ever fails the rejection has to be revisited - which is the point of
    having it here rather than only in a report.
    """
    table = rollover.basis_table()
    assert table.height == 6
    assert table["agrees"].all(), table


def test_usdjpy_and_eurusd_drift_in_opposite_directions():
    eur = rollover.basis("EURUSD", split="dev")
    jpy = rollover.basis("USDJPY", split="dev")
    assert eur.mid_bps > 0 and jpy.mid_bps < 0
    assert abs(eur.t_stat) > 3 and abs(jpy.t_stat) > 3


def test_the_wide_window_is_avoided():
    """Both fills land in an hour whose spread is normal, not in the roll."""
    b = rollover.basis("EURUSD", split="dev")
    assert b.spread_entry_bps < 0.2, "entry is inside the wide window"
    assert b.spread_exit_bps < 0.5, "exit is inside the wide window"


def test_test_split_stays_locked():
    with pytest.raises(SplitLockedError):
        rollover.basis("EURUSD", split="test")


# --------------------------------------------------------------------------
# Drift decomposition
# --------------------------------------------------------------------------

def _two_hour_tape(liquid_bps_per_min, roll_bps_per_min, days=40):
    """A tape that moves a known amount in liquid hours and a known amount in
    the roll hours, so the decomposition has an answer to be checked against."""
    rows = []
    px = 100.0
    for i in range(days):
        day = date(2021, 3, 1) + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for hour in range(24):
            rate = roll_bps_per_min if hour in rollover.ROLL_HOURS_UTC else liquid_bps_per_min
            for minute in range(60):
                px *= 1 + rate / 1e4
                ts_open = datetime(day.year, day.month, day.day, hour, minute,
                                   tzinfo=timezone.utc)
                rows.append({"ts": ts_open + timedelta(minutes=1),
                             "ts_open": ts_open, "close": px})
    return pl.DataFrame(rows)


def test_decomposition_separates_the_roll_from_the_rest(monkeypatch):
    frame = _two_hour_tape(liquid_bps_per_min=0.01, roll_bps_per_min=0.10)
    monkeypatch.setattr(rollover, "load_bars", lambda *a, **k: frame)
    d = rollover.drift_decomposition("XAUUSD")
    # 21 liquid hours a day at 0.01 bps a minute, 3 roll hours at 0.10
    ratio = d["roll_pct"] / d["liquid_pct"]
    assert ratio == pytest.approx((3 * 60 * 0.10) / (21 * 60 * 0.01), rel=0.02)


def test_the_three_buckets_sum_to_the_total(monkeypatch):
    frame = _two_hour_tape(liquid_bps_per_min=0.02, roll_bps_per_min=-0.05)
    monkeypatch.setattr(rollover, "load_bars", lambda *a, **k: frame)
    d = rollover.drift_decomposition("XAUUSD")
    assert d["roll_pct"] + d["gap_pct"] + d["liquid_pct"] == pytest.approx(
        d["total_pct"], abs=1e-6)


def test_a_gap_lands_in_the_gap_bucket(monkeypatch):
    """A jump across a missing minute belongs to neither quoted bucket."""
    rows = []
    for i, (hour, px) in enumerate([(9, 100.0), (10, 100.0), (12, 110.0), (13, 110.0)]):
        ts = datetime(2021, 3, 1, hour, 0, tzinfo=timezone.utc)
        rows.append({"ts": ts + timedelta(minutes=1), "ts_open": ts, "close": px})
    monkeypatch.setattr(rollover, "load_bars", lambda *a, **k: pl.DataFrame(rows))
    d = rollover.drift_decomposition("XAUUSD")
    assert d["liquid_pct"] == pytest.approx(0.0, abs=1e-9)
    assert d["roll_pct"] == pytest.approx(0.0, abs=1e-9)
    assert d["gap_pct"] == pytest.approx(d["total_pct"], abs=1e-9)


def test_gold_dev_drift_is_almost_all_financing():
    """The finding the XAUUSD rejection rests on, asserted rather than described.

    Gold's dev return is the roll plus the halt gap; what a strategy that goes
    home flat could actually reach is 0.42% a year. If this ever fails the
    rejection has to be revisited.
    """
    d = rollover.drift_decomposition("XAUUSD", split="dev")
    assert d["liquid_pct_per_year"] < 1.0
    assert d["financing_share"] > 0.85


def test_ustec_dev_drift_is_almost_all_liquid_hours():
    """The contrast that makes the gold number meaningful."""
    d = rollover.drift_decomposition("USTEC", split="dev")
    assert d["liquid_pct_per_year"] > 8.0
    assert d["financing_share"] < 0.15
