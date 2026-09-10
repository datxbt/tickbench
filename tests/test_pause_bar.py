"""Tests for the pause-bar breakout.

Two things here are worth more than the edge being measured, so both are pinned
against hand-built inputs where the answer is known by construction.

**Which side of the book each level triggers on.** A buy stop is filled on the
ask and a stop-out on the bid. Getting either backwards moves every result by
roughly a spread, and on this setup a spread is a large fraction of 1R.

**Which of the stop and the target came first.** The pause bar is chosen for
being small, so the stop sits a few points from the entry and a single ordinary
bar routinely contains both levels. On bars that has to be resolved by
assumption; here it is resolved by the tape, and these tests pin that it really
is the tape doing it - including the case where the target is reached later on
the same trade and must still lose.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.pause_bar import (
    TARGET_MULTIPLES,
    US,
    PauseBarConfig,
    _resolve,
    _tag,
    _Tape,
    signals,
)
from qlab.symbols import get_spec

SPEC = get_spec("XAUUSD")            # pip 0.01, 100 oz, $3.50 a side
COMMISSION_PX = SPEC.commission_pips() * SPEC.pip   # 7.0 pips = 0.07 in price
NO_SLIP = {h: 0.0 for h in range(24)}
T0 = datetime(2023, 3, 6, 10, 0, tzinfo=timezone.utc)   # a Monday, mid-session


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def make_tape(quotes, *, start=T0, step_ms=100) -> _Tape:
    """A tape from (bid, ask) pairs, one every ``step_ms``."""
    base = int(start.timestamp() * US)
    return _Tape(
        ts=np.array([base + i * step_ms * 1000 for i in range(len(quotes))],
                    dtype=np.int64),
        bid=np.array([q[0] for q in quotes], dtype=np.float64),
        ask=np.array([q[1] for q in quotes], dtype=np.float64),
    )


def make_signal(*, direction=1, trigger=100.0, stop=99.0, arm=T0,
                interval="5m", live_minutes=5) -> dict:
    """A signal dict shaped exactly as ``signals`` emits it, in tape units."""
    return {
        "interval": interval,
        "arm_ts": int(arm.timestamp() * US),
        "expiry_ts": int((arm + timedelta(minutes=live_minutes)).timestamp() * US),
        "direction": direction,
        "trigger": trigger,
        "stop": stop,
        "risk": abs(trigger - stop),
        "level": 99.5, "range_b": 4.0, "range_p": 1.0, "pause_frac": 0.25,
        "atr_b": 2.0, "big_atr": 2.0,
        "high_p": trigger - 0.001 if direction == 1 else stop - 0.001,
        "low_p": stop + 0.001 if direction == 1 else trigger + 0.001,
        "close_p": (trigger + stop) / 2, "close_b": (trigger + stop) / 2,
        "high_b": max(trigger, stop), "low_b": min(trigger, stop),
        "spread_close_b": 0.02, "spread_close_p": 0.02,
    }


def resolve(tape, sig, cfg=None, *, bars=None):
    """``_resolve`` with the trail's bar grid defaulted to "no bar ever closes"."""
    cfg = cfg or PauseBarConfig()
    if bars is None:
        bars = (np.array([], dtype=np.int64), np.array([]), np.array([]))
    return _resolve(tape, sig, cfg, SPEC, NO_SLIP, spread_cap=1.0,
                    bar_ts=bars[0], bar_hi=bars[1], bar_lo=bars[2])


def make_bars(rows, *, start=T0, minutes=5) -> pl.DataFrame:
    """An OHLC frame with the columns ``signals`` reads, on a regular grid."""
    n = len(rows)
    ts_open = [start + timedelta(minutes=minutes * i) for i in range(n)]
    return pl.DataFrame({
        "ts_open": ts_open,
        "ts": [t + timedelta(minutes=minutes) for t in ts_open],
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "spread_close": [0.02] * n,
    }).with_columns(
        pl.col("ts_open").dt.replace_time_zone("UTC"),
        pl.col("ts").dt.replace_time_zone("UTC"),
    )


def flat_run(n, price=100.0, jitter=0.05):
    """Quiet bars: small ranges, so a later big bar clears the ATR test."""
    return [(price, price + jitter, price - jitter, price)] * n


# --------------------------------------------------------------------------
# The fill
# --------------------------------------------------------------------------

def test_buy_stop_triggers_on_the_ask_not_the_mid():
    """A mid that reaches the trigger is not a fill; the ask has to reach it.

    The two ticks below have the same mid drift, and only the second has an ask
    at the level. Triggering on the first would book a fill that the book never
    offered, at a price better than any that existed.
    """
    tape = make_tape([(99.90, 99.94), (99.98, 100.02), (100.10, 100.14)] + [(100.2, 100.24)] * 5)
    row = resolve(tape, make_signal(trigger=100.0))
    assert row["filled"]
    assert row["entry"] == pytest.approx(100.02)      # the ask that reached it
    assert row["entry_ts"] == int(tape.ts[1])


def test_sell_stop_triggers_on_the_bid():
    tape = make_tape([(100.10, 100.14), (98.98, 99.02), (98.5, 98.54)] + [(98.4, 98.44)] * 5)
    row = resolve(tape, make_signal(direction=-1, trigger=99.0, stop=100.0))
    assert row["filled"]
    assert row["entry"] == pytest.approx(98.98)


def test_order_expires_unfilled():
    """Price never reaches the trigger inside the arming window."""
    tape = make_tape([(99.0, 99.04)] * 60, step_ms=10_000)   # 10 minutes of tape
    row = resolve(tape, make_signal(trigger=100.0, live_minutes=5))
    assert row["filled"] is False
    assert "entry" not in row


def test_fill_after_expiry_is_not_taken():
    """The break happens, but one tick after the order should have been pulled."""
    quotes = [(99.0, 99.04)] * 31 + [(100.5, 100.54)] * 10
    tape = make_tape(quotes, step_ms=10_000)   # tick 31 is at +310 s > 300 s
    row = resolve(tape, make_signal(trigger=100.0, live_minutes=5))
    assert row["filled"] is False


def test_spread_guard_refuses_to_arm():
    tape = make_tape([(99.0, 101.0)] + [(100.5, 100.54)] * 5)
    assert resolve(tape, make_signal()) is None


def test_slippage_moves_the_entry_against_the_position():
    tape = make_tape([(99.9, 99.94), (100.10, 100.14)] + [(100.2, 100.24)] * 5)
    sig = make_signal(trigger=100.0)
    long_row = _resolve(tape, sig, PauseBarConfig(), SPEC, {h: 0.05 for h in range(24)},
                        1.0, np.array([], dtype=np.int64), np.array([]), np.array([]))
    assert long_row["entry"] == pytest.approx(100.14 + 0.05)


# --------------------------------------------------------------------------
# Stop, target, and which came first
# --------------------------------------------------------------------------

def test_target_fills_at_its_price_and_pays_no_slippage():
    """A fixed target is a limit: it fills at the level or not at all."""
    quotes = [(99.9, 99.94), (100.10, 100.14)] + [(102.20, 102.24)] * 5
    tape = make_tape(quotes)
    sig = make_signal(trigger=100.0, stop=99.0)          # risk 1.0
    row = _resolve(tape, sig, PauseBarConfig(), SPEC, {h: 0.05 for h in range(24)},
                   1.0, np.array([], dtype=np.int64), np.array([]), np.array([]))
    entry = 100.14 + 0.05
    target = entry + 2.0 * 1.0
    assert row["reason_t2"] == "tp"
    # Gross is exactly the target distance - no slippage on the way out - and
    # net is that less the round-turn commission, both divided by risk.
    assert row["r_t2_gross"] == pytest.approx((target - entry) / 1.0)
    assert row["r_t2"] == pytest.approx(2.0 - COMMISSION_PX / 1.0)


def test_stop_before_target_loses_even_though_the_target_is_reached_later():
    """The ordering test this whole engine exists for.

    The tape dips to the stop, then rallies well past 2R. A bar-level backtest
    seeing one bar with that high and that low has to guess. Here the answer is
    not a guess: the stop tick came first, so the trade is a loss and the later
    high is unreachable.
    """
    quotes = ([(99.9, 99.94), (100.10, 100.14)]      # fill at 100.14
              + [(98.90, 98.94)]                     # stop at 99.0 taken
              + [(105.0, 105.04)] * 5)               # far beyond every target
    tape = make_tape(quotes)
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0))
    for mult in TARGET_MULTIPLES:
        assert row[f"reason_{_tag(mult)}"] == "stop", mult
        assert row[f"r_{_tag(mult)}"] < 0
    assert row["stopped"] is True
    # The reachable excursion stops at the stop; the window excursion does not.
    assert row["mfe_pre_stop_r"] < 1.0
    assert row["mfe_r"] > 4.0


def test_target_before_stop_wins_for_the_multiples_it_cleared():
    """One tape, two verdicts: 2R is reached before the stop, 4R is not."""
    quotes = ([(99.9, 99.94), (100.10, 100.14)]      # fill at 100.14
              + [(103.0, 103.04)]                    # +2.86R, clears 2R not 4R
              + [(98.5, 98.54)] * 5)                 # then through the stop
    tape = make_tape(quotes)
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0))
    assert row["reason_t2"] == "tp" and row["r_t2"] > 0
    assert row["reason_t4"] == "stop" and row["r_t4"] < 0
    assert row["mfe_pre_stop_r"] == pytest.approx(103.0 - 100.14)


def test_stop_out_pays_slippage_and_the_time_exit_does_too():
    quotes = [(99.9, 99.94), (100.10, 100.14), (98.90, 98.94)] + [(99.0, 99.04)] * 5
    tape = make_tape(quotes)
    row = _resolve(tape, make_signal(trigger=100.0, stop=99.0), PauseBarConfig(),
                   SPEC, {h: 0.05 for h in range(24)}, 1.0,
                   np.array([], dtype=np.int64), np.array([]), np.array([]))
    entry = 100.14 + 0.05
    assert row["r_hold_mid"] == pytest.approx(((98.90 + 98.94) / 2 - (100.10 + 100.14) / 2))
    assert row["r_hold"] == pytest.approx((98.90 - 0.05 - entry) - COMMISSION_PX)


def test_unstopped_trade_exits_at_the_holding_horizon():
    """No stop and no target: the position runs its window out, at market."""
    quotes = [(99.9, 99.94), (100.10, 100.14)] + [(100.3, 100.34)] * 200
    tape = make_tape(quotes, step_ms=10_000)          # 2000 s of tape
    cfg = PauseBarConfig(max_hold_bars=2)             # 5m bars -> 600 s
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0), cfg)
    assert row["stopped"] is False
    assert row["reason_hold"] == "time"
    assert row["hold_t6_s"] == pytest.approx(600, abs=10)


# --------------------------------------------------------------------------
# The trail
# --------------------------------------------------------------------------

def test_trail_does_not_move_before_it_is_armed():
    """Below ``trail_arm_r`` the stop is still the pause bar's low.

    Price rises to +0.5R, then falls back through the original stop. If the
    trail had armed on that move the exit would be a small win instead of a
    full loss, so this pins that it did not.
    """
    quotes = ([(99.9, 99.94), (100.10, 100.14)]      # fill at 100.14
              + [(100.60, 100.64)] * 3               # +0.46R, short of 1R
              + [(98.90, 98.94)] * 5)                # back through 99.0
    tape = make_tape(quotes)
    bar_ts = np.array([tape.ts[3], tape.ts[5]], dtype=np.int64)
    bars = (bar_ts, np.array([100.64, 100.64]), np.array([100.10, 100.60]))
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0), bars=bars)
    assert row["reason_trail"] == "stop"
    assert row["r_trail"] < -0.9


def test_trail_locks_in_a_gain_once_armed():
    """Past 1R the stop follows the last completed bar's low."""
    quotes = ([(99.9, 99.94), (100.10, 100.14)]      # fill at 100.14
              + [(101.60, 101.64)] * 3               # +1.46R: armed
              + [(101.50, 101.54)] * 3               # a bar closes, its low 101.50
              + [(101.15, 101.19)] * 5)              # back down through that low
    tape = make_tape(quotes)
    bar_ts = np.array([tape.ts[5], tape.ts[8]], dtype=np.int64)
    bars = (bar_ts, np.array([101.64, 101.54]), np.array([100.10, 101.50]))
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0), bars=bars)
    assert row["reason_trail"] == "trail"
    assert row["r_trail"] == pytest.approx((101.15 - 100.14) - COMMISSION_PX)


def test_trail_does_not_guarantee_its_level():
    """A trailing stop is a market order: it fills where the market is.

    The same geometry as above, except price leaves the trail level behind in
    one move. The exit is the quote that was there, not the level that was
    passed - which is why the trail is reported as an exit rather than as a
    floor under the result.
    """
    quotes = ([(99.9, 99.94), (100.10, 100.14)]
              + [(101.60, 101.64)] * 3
              + [(101.50, 101.54)] * 3
              + [(99.50, 99.54)] * 5)                # straight through 101.50
    tape = make_tape(quotes)
    bar_ts = np.array([tape.ts[5], tape.ts[8]], dtype=np.int64)
    bars = (bar_ts, np.array([101.64, 101.54]), np.array([100.10, 101.50]))
    row = resolve(tape, make_signal(trigger=100.0, stop=99.0), bars=bars)
    assert row["reason_trail"] == "trail"
    assert row["r_trail"] == pytest.approx((99.50 - 100.14) - COMMISSION_PX)
    assert row["r_trail"] < 0        # a trailed trade can still be a loss


# --------------------------------------------------------------------------
# Signal detection
# --------------------------------------------------------------------------

def _setup_frame(*, pause=(104.0, 104.2, 104.0, 104.1), big=(100.0, 104.5, 99.9, 104.4)):
    """Quiet history, one big breakout bar, then the pause bar under test."""
    return make_bars(flat_run(45) + [big, pause] + flat_run(2, price=104.2))


def test_detects_the_textbook_setup():
    cfg = PauseBarConfig(lookback=20, atr_period=20, big_mult=1.5,
                         pause_mult=0.5, close_frac=0.6, buffer_points=1.0)
    got = signals(_setup_frame(), cfg, interval="5m", point=0.001)
    assert got.height == 1
    row = got.row(0, named=True)
    assert row["direction"] == 1
    assert row["trigger"] == pytest.approx(104.2 + 0.001)   # pause high + a tick
    assert row["stop"] == pytest.approx(104.0 - 0.001)      # pause low - a tick
    assert row["risk"] == pytest.approx(0.202)
    # The order arms at the pause bar's close and lives one bar.
    assert row["expiry_ts"] - row["arm_ts"] == timedelta(minutes=5)


@pytest.mark.parametrize(
    "kwargs, why",
    [
        (dict(pause=(104.0, 104.5, 100.5, 104.1)), "pause bar is not small"),
        (dict(big=(100.0, 104.5, 99.9, 100.1)), "breakout bar closes weak"),
        (dict(big=(100.0, 100.2, 99.9, 100.15)), "breakout bar is not big"),
    ],
)
def test_rejects_near_misses(kwargs, why):
    assert signals(_setup_frame(**kwargs), PauseBarConfig(),
                   interval="5m", point=0.001).is_empty(), why


def test_breakout_bar_must_clear_the_lookback_high():
    """A big strong bar inside the range is not a breakout.

    History here peaks at 106, above the bar's own 104.5 close, so the level
    test fails while every other condition still passes.
    """
    frame = make_bars(
        flat_run(30) + [(100.0, 106.0, 99.9, 100.0)] + flat_run(10)
        + [(100.0, 104.5, 99.9, 104.4), (104.0, 104.2, 104.0, 104.1)]
        + flat_run(2, price=104.2)
    )
    assert signals(frame, PauseBarConfig(), interval="5m", point=0.001).is_empty()
    # The control keeps it: dropping the level test is exactly what "any" means.
    assert signals(frame, PauseBarConfig(context="any"),
                   interval="5m", point=0.001).height >= 1


def test_pause_bar_must_be_the_next_bar_in_time():
    """A gap between the two bars kills the setup even though rows are adjacent.

    Bars are missing rather than flat over weekends and the maintenance break,
    so row adjacency is not clock adjacency - and a pause bar three days after
    its breakout is a different animal.
    """
    frame = _setup_frame()
    gapped = frame.with_columns(
        pl.when(pl.col("ts_open") >= frame["ts_open"][46])
        .then(pl.col("ts_open") + timedelta(days=3))
        .otherwise(pl.col("ts_open"))
        .alias("ts_open"),
        pl.when(pl.col("ts") > frame["ts"][45])
        .then(pl.col("ts") + timedelta(days=3))
        .otherwise(pl.col("ts"))
        .alias("ts"),
    )
    assert signals(gapped, PauseBarConfig(), interval="5m", point=0.001).is_empty()


def test_short_side_is_the_exact_mirror():
    frame = make_bars(
        flat_run(45)
        + [(100.0, 100.1, 95.5, 95.6), (95.9, 96.0, 95.8, 95.9)]
        + flat_run(2, price=95.8)
    )
    got = signals(frame, PauseBarConfig(), interval="5m", point=0.001)
    assert got.height == 1
    row = got.row(0, named=True)
    assert row["direction"] == -1
    assert row["trigger"] == pytest.approx(95.8 - 0.001)    # pause low - a tick
    assert row["stop"] == pytest.approx(96.0 + 0.001)       # pause high + a tick


def test_both_sides_off_drops_the_shorts():
    frame = make_bars(
        flat_run(45)
        + [(100.0, 100.1, 95.5, 95.6), (95.9, 96.0, 95.8, 95.9)]
        + flat_run(2, price=95.8)
    )
    assert signals(frame, PauseBarConfig(both_sides=False),
                   interval="5m", point=0.001).is_empty()


def test_contexts_partition_the_geometry():
    """``breakout`` and ``no_breakout`` are disjoint, and both sit inside ``any``.

    That is what makes the control a fair comparison rather than a different
    experiment: the entry mechanics are identical and only the context differs.
    """
    frame = make_bars(
        flat_run(30) + [(100.0, 106.0, 99.9, 100.0)] + flat_run(10)
        + [(100.0, 104.5, 99.9, 104.4), (104.0, 104.2, 104.0, 104.1)]
        + flat_run(2, price=104.2)
    )
    keys = ("arm_ts", "direction", "trigger", "stop")
    def arms(context):
        cfg = PauseBarConfig(context=context)
        return set(signals(frame, cfg, interval="5m", point=0.001).select(keys).rows())

    breakout, no_breakout, any_ctx = arms("breakout"), arms("no_breakout"), arms("any")
    assert not (breakout & no_breakout)
    assert (breakout | no_breakout) <= any_ctx


def test_unknown_context_is_rejected():
    with pytest.raises(ValueError, match="unknown context"):
        signals(_setup_frame(), PauseBarConfig(context="wishful"),
                interval="5m", point=0.001)
