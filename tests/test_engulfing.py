"""Tests for the engulfing-candle setup.

Four things here are worth more than the edge being measured, so each is pinned
against a hand-built input whose answer is known by construction.

**The pattern itself.** Body-only engulfment with inclusive bounds is a
four-way conjunction, and every near miss - equal open, equal close, same-colour
pair, body that clears one end but not the other - has to be rejected for the
right reason rather than accidentally.

**Which side of the book each level acts on.** A long enters at the ask and its
stop is triggered by the bid. Getting that backwards moves every result by a
spread, and since 1R here is one bar's range, on 1m bars a spread is a large
fraction of 1R.

**Which of the stop and the target came first.** The stop sits at the signal
candle's own extreme, so a single ordinary bar routinely spans the stop and any
target under about 2R. On bars that is resolved by assumption; here it must be
resolved by the tape, including the case where the target is reached later on
the same trade and must still lose.

**The reversal chain.** The exit of a trade is priced independently of the
account's state, and :func:`sequence` reconstructs the state afterwards. That
equivalence is the load-bearing claim of the whole module: if it is wrong, the
trade list is wrong in a way no aggregate would reveal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.engulfing import (
    HOLD,
    TARGET_MULTIPLES,
    US,
    EngulfingConfig,
    _resolve,
    _Tape,
    sequence,
    signals,
    tag,
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


def make_signal(*, direction=1, entry_ref=100.0, stop=99.0, arm=T0,
                horizon_minutes=None, interval="5m") -> dict:
    """A signal dict shaped exactly as ``signals`` emits it, in tape units."""
    arm_us = int(arm.timestamp() * US)
    return {
        "interval": interval,
        "arm_ts": arm_us,
        "horizon_ts": (None if horizon_minutes is None
                       else arm_us + horizon_minutes * 60 * US),
        "direction": direction,
        "entry_ref": entry_ref,
        "stop": stop,
        "risk": abs(entry_ref - stop),
        "body": 0.5, "body_prev": 0.3, "bar_range": abs(entry_ref - stop),
        "engulf_ratio": 1.7,
        "open_p": 99.5, "high_p": 100.0, "low_p": 99.0, "close_p": 100.0,
        "spread_close_p": 0.02,
        "fwd1_r": 0.0, "fwd5_r": 0.0, "fwd20_r": 0.0,
        "fwd1_bps": 0.0, "fwd5_bps": 0.0, "fwd20_bps": 0.0,
        "mfe20_r": 0.0, "mae20_r": 0.0,
    }


def resolve(tape, sig, cfg=None, *, slip=NO_SLIP, spread_cap=0.0):
    cfg = cfg or EngulfingConfig()
    return _resolve(tape, sig, cfg, SPEC, slip, spread_cap, cfg.hold_span_us())


def make_bars(rows, *, start=T0, minutes=5, gaps=None) -> pl.DataFrame:
    """OHLC bars on a regular grid, right-edge labelled like the real ones.

    ``rows`` are (open, high, low, close). ``gaps`` is a set of row indices to
    push a day forward, to test the contiguity rule.
    """
    step = timedelta(minutes=minutes)
    gaps = gaps or set()
    ts_open, offset = [], timedelta(0)
    for i in range(len(rows)):
        if i in gaps:
            offset += timedelta(days=1)
        ts_open.append(start + i * step + offset)
    return pl.DataFrame({
        "ts_open": ts_open,
        "ts": [t + step for t in ts_open],
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "spread_close": [0.02] * len(rows),
    }).with_columns(
        pl.col("ts_open").dt.replace_time_zone("UTC"),
        pl.col("ts").dt.replace_time_zone("UTC"),
    )


def flat(n, price=100.0, half_spread=0.01):
    return [(price - half_spread, price + half_spread)] * n


# Enough trailing bars that the forward-return windows are defined and the
# signal survives the `bars.height` guard.
PAD = [(100.0, 100.1, 99.9, 100.0)] * 25


# --------------------------------------------------------------------------
# The pattern
# --------------------------------------------------------------------------

def _sig_frame(rows, **kw):
    cfg = kw.pop("cfg", None) or EngulfingConfig()
    return signals(make_bars(rows, **kw), cfg, interval="5m")


BULL = [(100.0, 100.1, 99.0, 99.2),      # bearish: open 100.0, close 99.2
        (99.1, 100.5, 98.8, 100.3)]      # bullish body 99.1 -> 100.3, engulfs

BEAR = [(99.2, 100.1, 99.0, 100.0),      # bullish: open 99.2, close 100.0
        (100.3, 100.5, 98.8, 99.1)]      # bearish body 100.3 -> 99.1, engulfs


def test_detects_the_textbook_bullish_engulfing():
    out = _sig_frame([*PAD[:3], *BULL, *PAD])
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["direction"] == 1
    # Entry at the signal candle's close, stop at its low, 1R the gap between.
    assert row["entry_ref"] == pytest.approx(100.3)
    assert row["stop"] == pytest.approx(98.8)
    assert row["risk"] == pytest.approx(1.5)


def test_short_side_is_the_exact_mirror():
    out = _sig_frame([*PAD[:3], *BEAR, *PAD])
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["direction"] == -1
    assert row["entry_ref"] == pytest.approx(99.1)
    assert row["stop"] == pytest.approx(100.5)   # the candle's high
    assert row["risk"] == pytest.approx(1.4)


@pytest.mark.parametrize("second,why", [
    ((99.3, 100.5, 98.8, 100.3), "open above the prior close: not engulfed"),
    ((99.1, 100.5, 98.8, 99.9), "close below the prior open: not engulfed"),
    ((100.3, 100.5, 98.8, 99.1), "second candle is bearish, not bullish"),
])
def test_rejects_near_misses(second, why):
    out = _sig_frame([*PAD[:3], BULL[0], second, *PAD])
    assert out.height == 0, why


def test_first_candle_must_be_the_opposite_colour():
    # A bullish candle engulfing a bullish candle is not the pattern.
    rows = [(99.0, 99.4, 98.9, 99.2), (98.9, 100.5, 98.8, 100.3)]
    assert _sig_frame([*PAD[:3], *rows, *PAD]).height == 0


def test_inclusive_bounds_accept_the_exact_touch_and_strict_rejects_it():
    # open[0] == close[1] and close[0] == open[1]: engulfment by equality.
    rows = [(100.0, 100.1, 99.0, 99.2), (99.2, 100.5, 98.8, 100.0)]
    assert _sig_frame([*PAD[:3], *rows, *PAD]).height == 1
    strict = _sig_frame([*PAD[:3], *rows, *PAD],
                        cfg=EngulfingConfig(strict=True))
    assert strict.height == 0


def test_both_sides_off_drops_the_shorts():
    cfg = EngulfingConfig(both_sides=False)
    assert _sig_frame([*PAD[:3], *BEAR, *PAD], cfg=cfg).height == 0
    assert _sig_frame([*PAD[:3], *BULL, *PAD], cfg=cfg).height == 1


def test_candles_must_be_adjacent_on_the_clock():
    """A weekend between the two candles is a gap, not a pattern."""
    rows = [*PAD[:3], *BULL, *PAD]
    assert _sig_frame(rows).height == 1
    # Push the engulfing candle a day forward: same prices, no longer adjacent.
    assert _sig_frame(rows, gaps={4}).height == 0


def test_fade_flips_the_direction_and_keeps_the_geometry():
    plain = _sig_frame([*PAD[:3], *BULL, *PAD]).row(0, named=True)
    faded = _sig_frame([*PAD[:3], *BULL, *PAD],
                       cfg=EngulfingConfig(fade=True)).row(0, named=True)
    assert plain["direction"] == 1 and faded["direction"] == -1
    # The fade is a mirror of the *signal*, not of the levels: it still reads
    # the same candle, so the stop moves to the other extreme.
    assert faded["entry_ref"] == pytest.approx(plain["entry_ref"])
    assert faded["stop"] == pytest.approx(100.5)


# --------------------------------------------------------------------------
# The reversal horizon
# --------------------------------------------------------------------------

def test_horizon_points_at_the_next_opposite_signal():
    out = _sig_frame([*PAD[:3], *BULL, *PAD[:3], *BEAR, *PAD])
    assert out.height == 2
    first, second = out.row(0, named=True), out.row(1, named=True)
    assert first["direction"] == 1 and second["direction"] == -1
    # The long's forced exit is the short's entry instant.
    assert first["horizon_ts"] == second["arm_ts"]
    # The short has no opposite successor, so it has no reversal exit.
    assert second["horizon_ts"] is None


def test_reverse_off_leaves_no_horizon_at_all():
    out = _sig_frame([*PAD[:3], *BULL, *PAD[:3], *BEAR, *PAD],
                     cfg=EngulfingConfig(reverse=False))
    assert out.height == 2
    assert out["horizon_ts"].null_count() == 2


# --------------------------------------------------------------------------
# Fills: which side of the book
# --------------------------------------------------------------------------

def test_long_enters_at_the_ask_and_short_at_the_bid():
    tape = make_tape(flat(50, 100.0))          # bid 99.99, ask 100.01
    lng = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0))
    assert lng["entry"] == pytest.approx(100.01)
    sht = resolve(tape, make_signal(direction=-1, entry_ref=100.0, stop=101.0))
    assert sht["entry"] == pytest.approx(99.99)


def test_slippage_moves_the_entry_against_the_position():
    tape = make_tape(flat(50, 100.0))
    slip = {h: 0.05 for h in range(24)}
    lng = resolve(tape, make_signal(direction=1), slip=slip)
    assert lng["entry"] == pytest.approx(100.06)      # ask + slip
    sht = resolve(tape, make_signal(direction=-1, stop=101.0), slip=slip)
    assert sht["entry"] == pytest.approx(99.94)       # bid - slip


def test_long_stop_is_triggered_by_the_bid_not_the_mid():
    # Mid falls to 99.005, so the mid never reaches the 99.0 stop - but the bid
    # does. A mid-triggered engine would call this trade unresolved.
    tape = make_tape(flat(5, 100.0) + [(98.995, 99.015)] + flat(20, 100.0))
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0))
    assert row["stopped"] is True
    assert row[f"reason_{HOLD}"] == "stop"


def test_spread_guard_refuses_the_trade():
    tape = make_tape([(99.0, 101.0)] + flat(30, 100.0))   # 2.0 wide at entry
    assert resolve(tape, make_signal(), spread_cap=0.5) is None
    assert resolve(tape, make_signal(), spread_cap=5.0) is not None


def test_stop_already_breached_at_entry_is_counted_not_dropped():
    """A close-to-low distance under half a spread stops out on the entry tick.

    Real, and exactly the trade a cost-blind study would quietly discard.
    """
    # Long, entry ref 100.0, stop 99.99; bid is already 99.98 at entry.
    tape = make_tape([(99.98, 100.02)] * 30)
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.99))
    assert row["instant_stop"] is True
    assert row[f"reason_{HOLD}"] == "stop"


# --------------------------------------------------------------------------
# Exits: which barrier came first
# --------------------------------------------------------------------------

def test_target_fills_at_its_price_and_pays_no_slippage():
    """A fixed target is a limit: it fills at its level or not at all."""
    tape = make_tape(flat(5, 100.0) + flat(30, 103.0))
    slip = {h: 0.05 for h in range(24)}
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0),
                  slip=slip)
    # entry = ask + slip = 100.06, risk = 1.0 (planned, mid-based).
    # 1R target = 101.06, reached; exit is exactly the target, no slippage.
    assert row[f"reason_{tag(1.0)}"] == "tp"
    assert row[f"r_{tag(1.0)}"] == pytest.approx((1.0 - COMMISSION_PX) / 1.0)


def test_stop_before_target_loses_even_though_the_target_is_reached_later():
    """The whole reason this is resolved on ticks and not on bars."""
    tape = make_tape(
        flat(3, 100.0)          # entry
        + flat(3, 98.5)         # stop at 99.0 taken here
        + flat(30, 105.0)       # target reached afterwards - too late
    )
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0))
    for mult in TARGET_MULTIPLES:
        assert row[f"reason_{tag(mult)}"] == "stop", f"{mult}R should have lost"
        assert row[f"r_{tag(mult)}"] < 0


def test_target_before_stop_wins_only_for_the_multiples_it_cleared():
    # Runs to +2.5R, then reverses through the stop.
    tape = make_tape(flat(3, 100.0) + flat(3, 102.5) + flat(30, 98.0))
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0))
    for mult in TARGET_MULTIPLES:
        want = "tp" if mult <= 2.49 else "stop"
        assert row[f"reason_{tag(mult)}"] == want, f"{mult}R"


def test_stop_out_pays_slippage_but_the_target_does_not():
    tape_stop = make_tape(flat(3, 100.0) + flat(30, 98.0))
    slip = {h: 0.05 for h in range(24)}
    row = resolve(tape_stop, make_signal(direction=1, entry_ref=100.0, stop=99.0),
                  slip=slip)
    # entry 100.06 (ask+slip); stop fill 97.94 (bid-slip) - worse than the
    # 99.0 level, because the tape gapped through it.
    assert row[f"reason_{HOLD}"] == "stop"
    assert row[f"r_{HOLD}"] == pytest.approx((97.94 - 100.06 - COMMISSION_PX) / 1.0)


def test_unresolved_trade_exits_at_the_reversal_and_is_labelled_so():
    tape = make_tape(flat(200, 100.0), step_ms=1000)      # 200 s of flat tape
    sig = make_signal(direction=1, entry_ref=100.0, stop=99.0, horizon_minutes=2)
    row = resolve(tape, sig)
    assert row["reverse_horizon"] is True
    assert row[f"reason_{HOLD}"] == "reverse"
    assert row["stopped"] is False


def test_holding_cap_binds_when_there_is_no_reversal():
    tape = make_tape(flat(400, 100.0), step_ms=1000)
    cfg = EngulfingConfig(max_hold_hours=0.05)            # 3 minutes
    row = _resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0),
                   cfg, SPEC, NO_SLIP, 0.0, cfg.hold_span_us())
    assert row["reverse_horizon"] is False
    assert row[f"reason_{HOLD}"] == "time"


def test_cost_r_is_the_round_turn_over_the_risk():
    tape = make_tape(flat(30, 100.0))                    # spread 0.02
    slip = {h: 0.01 for h in range(24)}
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0),
                  slip=slip)
    assert row["cost_r"] == pytest.approx(
        (COMMISSION_PX + 0.02 + 2 * 0.01) / 1.0)
    # Halve the risk and the cost per unit of risk doubles - the structural
    # objection to running this setup on fast timeframes, in one assertion.
    tighter = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.5),
                      slip=slip)
    assert tighter["cost_r"] == pytest.approx(2 * row["cost_r"])


def test_fixed_lots_make_dollar_risk_track_the_stop_distance():
    tape = make_tape(flat(30, 100.0))
    wide = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0))
    tight = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.5))
    # XAUUSD at 0.01 lot: $1 per $1 of gold, so 1.0 of risk is $1.00.
    assert wide["risk_usd"] == pytest.approx(1.0)
    assert tight["risk_usd"] == pytest.approx(0.5)


def test_mid_column_strips_every_cost():
    tape = make_tape(flat(3, 100.0) + flat(30, 103.0))
    slip = {h: 0.05 for h in range(24)}
    row = resolve(tape, make_signal(direction=1, entry_ref=100.0, stop=99.0),
                  slip=slip)
    # Mid 100.0 -> 103.0 is +3.0 on a planned risk of 1.0, with no spread, no
    # commission and no slippage anywhere in it.
    assert row[f"r_{tag(1.0)}_mid"] == pytest.approx(3.0)
    assert row[f"r_{tag(1.0)}"] < row[f"r_{tag(1.0)}_mid"]


# --------------------------------------------------------------------------
# The account: reversal chaining
# --------------------------------------------------------------------------

def _tape_frame(rows) -> pl.DataFrame:
    """A minimal resolved-trade frame: (arm_ts_seconds, exit_ts_seconds)."""
    base = int(T0.timestamp() * US)
    return pl.DataFrame({
        "interval": ["5m"] * len(rows),
        "arm_ts": [base + r[0] * US for r in rows],
        f"exit_ts_{HOLD}": [base + r[1] * US for r in rows],
        "label": list(range(len(rows))),
    }).with_columns(pl.col("arm_ts").cast(pl.Datetime("us", "UTC")))


def test_sequence_skips_signals_that_fire_mid_trade():
    # Trade 0 runs 0 -> 100s. Signals at 30s and 60s land inside it and are
    # ignored; the one at 120s is taken.
    taken = sequence(_tape_frame([(0, 100), (30, 50), (60, 90), (120, 200)]),
                     HOLD)
    assert taken["label"].to_list() == [0, 3]


def test_sequence_takes_the_signal_that_lands_exactly_on_the_exit():
    """A reversal exit lands on the opposite signal's timestamp, by construction."""
    taken = sequence(_tape_frame([(0, 100), (100, 300), (400, 500)]), HOLD)
    assert taken["label"].to_list() == [0, 1, 2]


def test_sequence_always_advances_on_an_instant_stop():
    # A trade stopped out on its own entry tick must not re-take itself.
    taken = sequence(_tape_frame([(0, 0), (10, 10), (20, 20)]), HOLD)
    assert taken["label"].to_list() == [0, 1, 2]


def test_sequence_treats_each_timeframe_as_its_own_account():
    frame = _tape_frame([(0, 100), (30, 50)])
    other = frame.with_columns(interval=pl.lit("15m"))
    taken = sequence(pl.concat([frame, other]), HOLD)
    # The 5m account skips its second signal; the 15m account is independent
    # and skips its own - two trades, one per timeframe, not one overall.
    assert taken.height == 2
    assert set(taken["interval"].to_list()) == {"5m", "15m"}


def test_sequence_chain_matches_a_naive_stateful_walk():
    """The equivalence the module rests on, checked against a literal simulation."""
    rng = np.random.default_rng(7)
    arm = np.sort(rng.choice(np.arange(0, 4000), size=300, replace=False))
    out = arm + rng.integers(1, 200, size=arm.size)
    frame = _tape_frame(list(zip(arm.tolist(), out.tolist())))

    naive, busy_until = [], -1
    for i in range(arm.size):
        if arm[i] < busy_until:
            continue
        naive.append(i)
        busy_until = out[i]

    assert sequence(frame, HOLD)["label"].to_list() == naive


# --------------------------------------------------------------------------
# The forward-looking measurements
# --------------------------------------------------------------------------
# These columns deliberately look ahead: they are the unconditional behaviour
# of price after the signal, which is the read that precedes any exit choice.
# They are never inputs to a rule, so the lookahead is a measurement rather
# than a leak - but the arithmetic still has to be right, because the whole
# "does the pattern predict anything?" verdict rests on it.

def test_forward_return_is_signed_by_the_direction_taken():
    # A long signal, then a steady climb: forward returns must be positive.
    after = [(100.3 + i * 0.1, 100.4 + i * 0.1, 100.2 + i * 0.1, 100.3 + i * 0.1)
             for i in range(1, 25)]
    row = _sig_frame([*PAD[:3], *BULL, *after]).row(0, named=True)
    # Entry ref is the signal candle's close, 100.3. One bar later: 100.4.
    assert row["fwd1_r"] == pytest.approx((100.4 - 100.3) / 1.5)
    assert row["fwd5_r"] == pytest.approx((100.8 - 100.3) / 1.5)
    assert row["fwd1_bps"] == pytest.approx((100.4 - 100.3) / 100.3 * 1e4)
    assert row["fwd20_r"] > row["fwd5_r"] > row["fwd1_r"] > 0


def test_forward_return_of_a_short_is_positive_when_price_falls():
    after = [(99.1 - i * 0.1, 99.2 - i * 0.1, 99.0 - i * 0.1, 99.1 - i * 0.1)
             for i in range(1, 25)]
    row = _sig_frame([*PAD[:3], *BEAR, *after]).row(0, named=True)
    assert row["direction"] == -1
    assert row["fwd5_r"] > 0 and row["fwd5_bps"] > 0


def test_mfe_and_mae_span_the_twenty_bars_after_the_signal():
    # Flat, except for one spike up and one spike down inside the window.
    after = [(100.3, 100.4, 100.2, 100.3)] * 24
    after[2] = (100.3, 102.3, 100.2, 100.3)     # +2.0 above the entry ref
    after[7] = (100.3, 100.4, 99.3, 100.3)      # -1.0 below it
    row = _sig_frame([*PAD[:3], *BULL, *after]).row(0, named=True)
    assert row["mfe20_r"] == pytest.approx((102.3 - 100.3) / 1.5)
    assert row["mae20_r"] == pytest.approx((100.3 - 99.3) / 1.5)


def test_mfe_and_mae_ignore_the_signal_candle_itself():
    """The window starts at the *next* bar; the signal candle is already spent."""
    after = [(100.3, 100.35, 100.25, 100.3)] * 24
    row = _sig_frame([*PAD[:3], *BULL, *after]).row(0, named=True)
    # The signal candle's own high is 100.5 and its low 98.8. Neither may leak
    # into the forward window, whose extremes are 100.35 and 100.25.
    assert row["mfe20_r"] == pytest.approx((100.35 - 100.3) / 1.5)
    assert row["mae20_r"] == pytest.approx((100.3 - 100.25) / 1.5)


def test_control_draws_non_signal_bars_and_is_reproducible():
    rows = [*PAD[:3], *BULL, *PAD, *BEAR, *PAD]
    cfg = EngulfingConfig(control=True)
    a = _sig_frame(rows, cfg=cfg)
    b = _sig_frame(rows, cfg=cfg)
    assert a.height > 0
    assert a["arm_ts"].to_list() == b["arm_ts"].to_list()   # deterministic
    # Disjoint from the real signals, by construction.
    real = set(_sig_frame(rows)["arm_ts"].to_list())
    assert real and not (real & set(a["arm_ts"].to_list()))


def test_control_direction_comes_from_the_bar_body():
    rows = [*PAD[:3], *BULL, *PAD, *BEAR, *PAD]
    out = _sig_frame(rows, cfg=EngulfingConfig(control=True))
    for r in out.iter_rows(named=True):
        want = 1 if r["close_p"] > r["open_p"] else -1
        assert r["direction"] == want
        # And the stop is that bar's opposing extreme, as it is for a signal.
        assert r["stop"] == pytest.approx(r["low_p"] if want == 1 else r["high_p"])
