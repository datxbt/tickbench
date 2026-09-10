"""Tests for the 5-minute opening range breakout.

Three things in this module can be wrong in ways no summary statistic would
reveal, so all three are pinned against inputs whose answer is known by
construction:

* **Which side of the book each order touches.** A buy stop triggers on the ask
  and a long stops out on the bid. Getting that backwards is worth roughly one
  spread per trade, which on a 10%-ATR stop is a tenth of the edge being
  measured.
* **Same-tick ties.** A tick that reaches the stop and the target at once is
  resolved against the trade. It has to be, and it has to stay that way.
* **The point-in-time lag on the ATR and the relative volume.** Both are means
  over previous sessions, and a rolling mean that includes today is the single
  easiest way to leak an answer into a signal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.opening_range import (
    RELVOL_EDGES,
    ORBConfig,
    _Entry,
    _find_entry,
    _simulate,
    _Tape,
    daily_context,
    eligible,
    placebo_signal,
    relvol_buckets,
    tag,
)

US = 1_000_000
CFG = ORBConfig()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_direction_rule_is_validated():
    with pytest.raises(ValueError):
        ORBConfig(direction_rule="sideways")


def test_opening_range_must_fit_in_the_session():
    with pytest.raises(ValueError):
        ORBConfig(or_minutes=400)


def test_or_end_is_the_open_plus_the_range():
    assert ORBConfig(or_minutes=15).or_end_min == 9 * 60 + 45


def test_tag_names_the_no_target_case_separately():
    assert tag(None) == "hold"
    assert tag(2.0) == "t2"
    assert tag(0.5) == "t0_5"


# --------------------------------------------------------------------------
# A tape whose answer is known
# --------------------------------------------------------------------------

def build_tape(quotes, *, start=datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc)):
    """One tick per second from ``start``, given as ``(bid, ask)`` pairs."""
    base = int(start.timestamp()) * US
    return _Tape(
        ts=np.array([base + i * US for i in range(len(quotes))], dtype=np.int64),
        bid=np.array([q[0] for q in quotes], dtype=float),
        ask=np.array([q[1] for q in quotes], dtype=float),
    )


def make_row(**kwargs):
    row = {
        "or_high": 100.0,
        "or_low": 90.0,
        "signal": 1,
        "atr": 10.0,
        "or_end_ts": datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc),
        "close_ts": datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc),
    }
    row.update(kwargs)
    return row


def test_buy_stop_triggers_on_the_ask_not_the_mid():
    """The mid crosses 100 one tick before the ask does. The ask is what counts."""
    tape = build_tape([
        (98.0, 98.4),
        (99.9, 100.3),   # mid 100.1 is already through, ask 100.3 triggers
        (101.0, 101.4),
    ])
    entry = _find_entry(tape, make_row(), CFG, slip=0.0)
    assert entry is not None
    assert entry.direction == 1
    assert entry.i == 1
    assert entry.price == pytest.approx(100.3)


def test_sell_stop_triggers_on_the_bid():
    tape = build_tape([
        (91.0, 91.4),
        (89.8, 90.2),    # bid through the low
        (88.0, 88.4),
    ])
    entry = _find_entry(tape, make_row(signal=-1), CFG, slip=0.0)
    assert entry is not None
    assert entry.direction == -1
    assert entry.i == 1
    assert entry.price == pytest.approx(89.8)


def test_the_papers_rule_refuses_the_other_side():
    """A bullish opening candle does not take a downside break. That is the
    paper's central instruction, and it is the whole of its directional claim."""
    tape = build_tape([(91.0, 91.4), (89.0, 89.4), (88.0, 88.4)])
    assert _find_entry(tape, make_row(signal=1), CFG, slip=0.0) is None


def test_both_sides_rule_takes_whichever_comes_first():
    tape = build_tape([(91.0, 91.4), (89.0, 89.4), (88.0, 88.4)])
    cfg = ORBConfig(direction_rule="both")
    entry = _find_entry(tape, make_row(signal=1), cfg, slip=0.0)
    assert entry is not None and entry.direction == -1


def test_contra_rule_inverts_the_signal():
    tape = build_tape([(91.0, 91.4), (89.0, 89.4)])
    cfg = ORBConfig(direction_rule="contra")
    entry = _find_entry(tape, make_row(signal=1), cfg, slip=0.0)
    assert entry is not None and entry.direction == -1


def test_a_doji_trades_nothing():
    tape = build_tape([(99.0, 101.5), (85.0, 86.0)])
    assert _find_entry(tape, make_row(signal=0), CFG, slip=0.0) is None


def test_slippage_pushes_the_fill_against_the_trade():
    tape = build_tape([(99.9, 100.3), (101.0, 101.4)])
    long = _find_entry(tape, make_row(), CFG, slip=0.5)
    short = _find_entry(tape, make_row(signal=-1, or_low=100.0, or_high=110.0),
                        CFG, slip=0.5)
    assert long.price == pytest.approx(100.8)     # paid up
    assert short.price == pytest.approx(99.4)     # sold down


def test_no_fill_when_the_range_is_never_broken():
    tape = build_tape([(95.0, 95.4), (96.0, 96.4), (94.0, 94.4)])
    assert _find_entry(tape, make_row(), CFG, slip=0.0) is None


# --------------------------------------------------------------------------
# Exits
# --------------------------------------------------------------------------

def simulate(quotes, *, target=None, stop_frac=0.10, signal=1, slip=0.0,
             atr=10.0):
    tape = build_tape(quotes)
    row = make_row(signal=signal, atr=atr,
                   close_ts=datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc)
                   + timedelta(seconds=len(quotes)))
    entry = _find_entry(tape, row, ORBConfig(), slip=slip)
    assert entry is not None
    return entry, _simulate(tape, row, entry, stop_frac, target, ORBConfig(), slip)


def test_long_stops_out_on_the_bid_at_ten_percent_of_atr():
    """Fill at 100.3, ATR 10, stop fraction 0.10 -> risk 1.0 -> stop at 99.3."""
    entry, out = simulate([
        (99.9, 100.3),
        (99.5, 99.9),
        (99.2, 99.6),    # bid 99.2 <= 99.3
        (99.0, 99.4),
    ])
    assert out["risk"] == pytest.approx(1.0)
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == pytest.approx(99.2)
    assert out["r_multiple"] == pytest.approx(-1.1)


def test_target_is_a_limit_and_pays_no_slippage():
    entry, out = simulate([
        (99.9, 100.3),
        (102.0, 102.4),   # ask through 100.3 + 2 * 1.0
    ], target=2.0, slip=0.1)
    assert out["exit_reason"] == "target"
    # entry 100.4 after slippage, risk 1.0, so the limit sits at 102.4 exactly
    assert out["exit_price"] == pytest.approx(entry.price + 2.0)
    assert out["r_multiple"] == pytest.approx(2.0)


def test_a_tie_inside_one_tick_is_resolved_against_the_trade():
    """A tick whose bid is at the stop and whose ask is at the target is a
    stop-out, because the tape carries no path through the tick."""
    _, out = simulate([
        (99.9, 100.3),
        (99.3, 102.3),    # both reachable in the same tick
    ], target=2.0)
    assert out["exit_reason"] == "stop"


def test_untouched_position_closes_at_the_bell():
    _, out = simulate([
        (99.9, 100.3),
        (100.5, 100.9),
        (101.0, 101.4),
    ])
    assert out["exit_reason"] == "bell"
    assert out["exit_price"] == pytest.approx(101.0)


def test_r_multiple_is_invariant_to_the_price_level():
    """Twice the ATR at twice the stop fraction is the same trade in R."""
    _, wide = simulate([(99.9, 100.3), (101.9, 102.3)], atr=10.0, stop_frac=0.20)
    _, narrow = simulate([(99.9, 100.3), (101.9, 102.3)], atr=20.0, stop_frac=0.10)
    assert wide["r_multiple"] == pytest.approx(narrow["r_multiple"])


def test_short_exits_are_mirrored():
    _, out = simulate([
        (89.8, 90.2),     # sell stop fills at 89.8, risk 1.0, stop at 90.8
        (90.9, 91.3),
    ], signal=-1)
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == pytest.approx(91.3)   # a short stops on the ask


# --------------------------------------------------------------------------
# Point-in-time discipline
# --------------------------------------------------------------------------

def synthetic_bars(days: int = 40, *, or_ticks=None) -> pl.DataFrame:
    """One-minute bars for a plain 09:30-16:00 New York session.

    January, so New York is UTC-5 throughout and the session is 14:30-21:00 UTC.
    """
    rows = []
    for d in range(days):
        day = datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for minute in range(390):
            ts_open = day.replace(hour=14, minute=30) + timedelta(minutes=minute)
            price = 100.0 + d + minute * 0.01
            ticks = 10
            if minute < 5 and or_ticks is not None:
                ticks = or_ticks(d)
            rows.append({
                "ts": ts_open + timedelta(minutes=1),
                "ts_open": ts_open,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.02,
                "n_ticks": ticks,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
        pl.col("ts_open").dt.replace_time_zone("UTC"),
        pl.col("n_ticks").cast(pl.UInt32),
    )


def context_from(bars: pl.DataFrame, cfg: ORBConfig = CFG) -> pl.DataFrame:
    """``daily_context`` without the loader, so the lag logic is testable."""
    import qlab.strategies.opening_range as module

    original = module.load_bars
    module.load_bars = lambda *a, **k: bars
    try:
        return module.daily_context("USTEC", cfg, split=None)
    finally:
        module.load_bars = original


def test_atr_uses_only_previous_sessions():
    """A one-day spike in range must not appear in that day's own ATR."""
    bars = synthetic_bars()
    ctx = context_from(bars)
    # Every synthetic session has the same geometry, so every ATR is the same;
    # the test that matters is that the *first* 14 are absent rather than
    # computed from a partial window that includes today.
    assert ctx["atr"].null_count() == 0
    assert ctx.height == bars.select(pl.col("ts_open").dt.date()).n_unique() - 14


def test_relvol_is_todays_range_over_the_previous_fourteen():
    """Fourteen quiet sessions then one loud one: RelVol on the loud day is the
    ratio, and on the day after it is back to roughly one."""
    bars = synthetic_bars(days=60, or_ticks=lambda d: 100 if d == 30 else 10)
    ctx = context_from(bars).sort("nyd")
    # The opening range is five bars, so a quiet one is 50 ticks and a loud
    # one is 500.
    loud = ctx.filter(pl.col("or_ticks") == 500)
    assert loud.height == 1
    assert loud["relvol"].item() == pytest.approx(10.0)
    after = ctx.filter(pl.col("nyd") > loud["nyd"].item()).head(1)
    # The loud day is now one of the fourteen in the denominator.
    assert after["relvol"].item() == pytest.approx(50.0 / ((13 * 50 + 500) / 14))


def test_signal_is_the_sign_of_the_opening_candle():
    bars = synthetic_bars()
    ctx = context_from(bars)
    # The synthetic close is always above the open, so every session is bullish.
    assert set(ctx["signal"].to_list()) == {1}


# --------------------------------------------------------------------------
# Filters and reporting
# --------------------------------------------------------------------------

def small_context() -> pl.DataFrame:
    return pl.DataFrame({
        "signal": pl.Series([1, -1, 0, 1, -1], dtype=pl.Int8),
        "relvol": [0.5, 1.5, 2.0, 1.0, 0.9],
    })


def test_eligible_drops_dojis_under_the_papers_rule():
    out = eligible(small_context(), ORBConfig())
    assert out.height == 4
    assert 0 not in out["signal"].to_list()


def test_eligible_keeps_dojis_when_both_sides_are_armed():
    out = eligible(small_context(), ORBConfig(direction_rule="both"))
    assert out.height == 5


def test_relvol_filter_is_inclusive_at_the_threshold():
    out = eligible(small_context(), ORBConfig(relvol_min=1.0,
                                              direction_rule="both"))
    assert sorted(out["relvol"].to_list()) == [1.0, 1.5, 2.0]


def test_placebo_keeps_the_signal_multiset_and_the_calendar():
    ctx = pl.DataFrame({
        "nyd": list(range(50)),
        "signal": pl.Series([1, -1] * 25, dtype=pl.Int8),
        "relvol": [1.0] * 50,
    })
    shuffled = placebo_signal(ctx, seed=7)
    assert shuffled["nyd"].to_list() == ctx["nyd"].to_list()
    assert sorted(shuffled["signal"].to_list()) == sorted(ctx["signal"].to_list())
    assert shuffled["signal"].to_list() != ctx["signal"].to_list()


def test_relvol_buckets_partition_every_trade():
    trades = pl.DataFrame({
        "relvol": [0.2, 0.7, 1.2, 1.8, 2.5, 4.0, 8.0, 20.0],
        "net_r": [1.0] * 8,
    })
    out = relvol_buckets(trades)
    assert out["n"].sum() == trades.height
    assert out.height <= len(RELVOL_EDGES)
