"""Tests for the regime-filtered VWAP/EMA gold strategy.

Four families of thing in this module can be wrong in ways that no summary
statistic would reveal, so all four are pinned against inputs whose answer is
known by construction:

* **The six entry conditions.** They are transcribed from a paper's equations,
  and a transcription error - a ``>`` for a ``>=``, a wick measured from the
  wrong end - produces a strategy that runs fine and tests something else.
* **Which side of the book each order touches.** A long pays the ask to get in
  and receives the bid on the way out, *including at its profit target*. Getting
  that backwards is worth a spread per trade against a 1R that is only a few
  dollars wide.
* **The three exit clocks and their precedence.** A stop, a bar-close trail and
  a target can all come due in the same bar, and the resolution has to be
  against the trade every time.
* **The close-only trail.** The paper's headline contribution is that intrabar
  wicks through the 50 EMA are *ignored*. A trail that fires on a wick is a
  different strategy with the same name.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.vwap_ema import (
    NY_CLOSE_MIN,
    NY_OPEN_MIN,
    US,
    VWAPEmaConfig,
    _Entry,
    _entry_at,
    _simulate,
    _Tape,
    _Window,
    _vwap_touch,
    condition_attrition,
    exit_surface,
    ny_anchored,
    outcome_table,
    placebo_signals,
    signals,
    tag,
)

CFG = VWAPEmaConfig()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_entry_pattern_is_validated():
    with pytest.raises(ValueError):
        VWAPEmaConfig(entry_pattern="hammer")


def test_side_is_validated():
    with pytest.raises(ValueError):
        VWAPEmaConfig(side="sideways")


def test_anchor_is_validated():
    with pytest.raises(ValueError):
        VWAPEmaConfig(anchor="london")


def test_a_zero_stop_multiple_is_refused_because_it_defines_1r():
    with pytest.raises(ValueError):
        VWAPEmaConfig(stop_atr_mult=0.0)


def test_sides_expands_the_direction_rule():
    assert VWAPEmaConfig(side="both").sides == (1, -1)
    assert VWAPEmaConfig(side="long").sides == (1,)
    assert VWAPEmaConfig(side="short").sides == (-1,)


def test_ny_anchoring_moves_the_window_with_the_clock():
    """Switching the anchor alone would ask for 13:30-20:00 New York."""
    cfg = ny_anchored(CFG)
    assert cfg.anchor == "ny"
    assert (cfg.session_start_min, cfg.session_end_min) == (NY_OPEN_MIN, NY_CLOSE_MIN)


def test_tag_names_the_no_target_case():
    assert tag(None) == "hold"
    assert tag(3.0) == "t3"
    assert tag(1.5) == "t1_5"


# --------------------------------------------------------------------------
# The six entry conditions
# --------------------------------------------------------------------------

def _bars(rows: list[dict]) -> pl.DataFrame:
    """A minimal indicator frame - only the columns :func:`signals` reads."""
    base = datetime(2024, 6, 3, 13, 30, tzinfo=timezone.utc)
    defaults = dict(
        open=100.0, high=101.0, low=99.0, close=100.0,
        ema_regime=90.0, ema_trail=99.5, ema_final=99.8, vwap=95.0,
        atr=2.0, volume=200.0, vol_ma=100.0, in_session=True,
    )
    out = []
    for i, row in enumerate(rows):
        r = defaults | row
        ts_open = base + timedelta(minutes=15 * i)
        r |= {
            "ts": ts_open + timedelta(minutes=15),
            "ts_open": ts_open,
            "utc_min": ts_open.hour * 60 + ts_open.minute,
            "ny_min": ts_open.hour * 60 + ts_open.minute,
            "session": ts_open.date(),
            "session_bar": i,
        }
        out.append(r)
    frame = pl.DataFrame(out)
    return frame.with_columns(
        prev_close=pl.col("close").shift(1),
        prev_open=pl.col("open").shift(1),
        prev_low=pl.col("low").shift(1),
        prev_high=pl.col("high").shift(1),
        body=(pl.col("close") - pl.col("open")).abs(),
        upper_wick=pl.col("high") - pl.max_horizontal("open", "close"),
        lower_wick=pl.min_horizontal("open", "close") - pl.col("low"),
    )


def _long_signal_bar(**overrides) -> dict:
    """A bar that satisfies every long condition, so a test can break one.

    Pin bar: body 0.2, lower wick 2.0 (>= 2x body), upper wick 0.1
    (<= 0.5x lower wick). Range 2.1 >= 0.8 x ATR 2.0. Volume 200 > 1.1 x 100.
    Close 100.0 is above EMA200 at 90, above VWAP at 95, and the low 97.8 is
    below the 50 EMA at 99.5, which is itself below the close.
    """
    return dict(open=99.8, close=100.0, high=100.1, low=97.8) | overrides


def test_a_bar_meeting_all_six_conditions_fires():
    bars = _bars([_long_signal_bar(), _long_signal_bar()])
    out = signals(bars, CFG)
    assert out["signal"][1] == 1


def test_c1_rejects_a_close_inside_the_ambiguity_band():
    """The paper skips trades within 0.1% of the 200 EMA, on both sides."""
    bars = _bars([_long_signal_bar(), _long_signal_bar(ema_regime=99.95)])
    assert signals(bars, CFG)["signal"][1] == 0


def test_c2_rejects_a_close_on_the_wrong_side_of_vwap():
    bars = _bars([_long_signal_bar(), _long_signal_bar(vwap=100.5)])
    assert signals(bars, CFG)["signal"][1] == 0


def test_c3_needs_the_50_ema_between_the_low_and_the_close():
    """Price must have *reached* the 50 EMA and closed back above it."""
    # The EMA sits below both lows: never touched, so no pullback happened.
    bars = _bars([_long_signal_bar(), _long_signal_bar(ema_trail=97.0)])
    assert signals(bars, CFG)["signal"][1] == 0
    # The EMA sits above the close: price did not close back above it.
    bars = _bars([_long_signal_bar(), _long_signal_bar(ema_trail=100.5)])
    assert signals(bars, CFG)["signal"][1] == 0


def test_c3_accepts_a_touch_on_the_previous_bar():
    """``min(L_t, L_{t-1})`` - the pullback may have happened one bar earlier."""
    prev = _long_signal_bar(low=97.0)
    # Today's low stops at 99.7, above the 99.5 EMA; the high is stretched so
    # that C6's above-average-range test still passes on its own terms.
    now = _long_signal_bar(low=99.7, high=101.5)
    assert signals(_bars([prev, now]), CFG)["signal"][1] == 1


def test_c4_pin_bar_needs_a_long_lower_wick_and_a_short_upper_one():
    bars = _bars([_long_signal_bar(),
                  _long_signal_bar(open=99.0, close=100.0, high=100.1, low=98.9)])
    # body 1.0, lower wick 0.1: nowhere near 2x body, and not an engulfing
    # either, since the previous open is 99.8 and this open 99.0 < prev close.
    out = signals(bars, CFG)
    assert not out["pin_long"][1]


def test_c4_engulfing_is_the_paper_s_two_inequalities():
    """``C_t > O_{t-1}`` and ``O_t < C_{t-1}``, not the prose gloss."""
    prev = dict(open=101.0, close=99.0, high=101.2, low=98.9)     # bearish
    now = dict(open=98.8, close=101.5, high=101.6, low=98.7)      # engulfs it
    out = signals(_bars([prev, now]), CFG)
    assert out["engulf_long"][1]
    assert out["signal"][1] == 1


def test_c5_rejects_below_average_volume():
    bars = _bars([_long_signal_bar(), _long_signal_bar(volume=105.0)])
    assert signals(bars, CFG)["signal"][1] == 0     # 105 < 1.1 x 100


def test_c6_rejects_a_below_average_range():
    """Range 2.1 against ATR 2.0 passes; against ATR 4.0 it does not."""
    bars = _bars([_long_signal_bar(), _long_signal_bar(atr=4.0)])
    assert signals(bars, CFG)["signal"][1] == 0


def test_short_conditions_are_the_exact_mirror():
    prev = dict(open=100.2, close=100.0, high=102.2, low=99.9)
    now = dict(open=100.2, close=100.0, high=102.2, low=99.9,
               ema_regime=110.0, vwap=105.0, ema_trail=100.5)
    out = signals(_bars([prev, now]), CFG)
    assert out["signal"][1] == -1


def test_out_of_session_bars_never_fire():
    bars = _bars([_long_signal_bar(), _long_signal_bar()]).with_columns(
        in_session=pl.lit(False)
    )
    assert (signals(bars, CFG)["signal"] == 0).all()


def test_side_restriction_suppresses_the_other_direction():
    bars = _bars([_long_signal_bar(), _long_signal_bar()])
    assert (signals(bars, VWAPEmaConfig(side="short"))["signal"] == 0).all()


def test_removing_c4_admits_bars_the_rejection_candle_rejected():
    """The control that says how much work the rejection candle does."""
    # Passes C1, C2, C3, C5 and C6 but is neither a pin (body 0.55 against a
    # 0.65 lower wick) nor an engulfing (it opens above the previous close).
    plain = _long_signal_bar(open=100.05, close=100.6, high=101.2, low=99.4)
    bars = _bars([_long_signal_bar(), plain])
    assert signals(bars, CFG)["signal"][1] == 0
    assert signals(bars, VWAPEmaConfig(entry_pattern="any"))["signal"][1] == 1


def test_attrition_is_monotone_and_ends_at_the_signal_count():
    bars = signals(_bars([_long_signal_bar() for _ in range(6)]), CFG)
    table = condition_attrition(bars, CFG).filter(pl.col("side") == "long")
    cumulative = table["cumulative"].to_list()
    assert cumulative == sorted(cumulative, reverse=True)
    assert cumulative[-1] == int((bars["signal"] == 1).sum())


# --------------------------------------------------------------------------
# Fills: which side of the book, and which clock
# --------------------------------------------------------------------------

def _tape(bids, asks, step_s: int = 1, start_us: int = 0) -> _Tape:
    n = len(bids)
    return _Tape(
        ts=np.arange(n, dtype=np.int64) * step_s * US + start_us,
        bid=np.asarray(bids, dtype=float),
        ask=np.asarray(asks, dtype=float),
    )


def _window(closes, trail, final=None, vwap=None, step_s: int = 1,
            start_us: int = 0) -> _Window:
    n = len(closes)
    return _Window(
        close=np.asarray(closes, dtype=float),
        ema_trail=np.asarray(trail, dtype=float),
        ema_final=np.asarray(final if final is not None else trail, dtype=float),
        vwap=np.asarray(vwap if vwap is not None else [0.0] * n, dtype=float),
        end_us=np.arange(1, n + 1, dtype=np.int64) * step_s * US + start_us,
        last_us=int(n * step_s * US + start_us),
    )


def test_a_long_enters_at_the_ask_and_a_short_at_the_bid():
    tape = _tape([100.0, 100.0], [100.2, 100.2])
    long = _entry_at(tape, 0, 1, slip=0.05)
    short = _entry_at(tape, 0, -1, slip=0.05)
    assert long.price == pytest.approx(100.25)     # ask + slippage
    assert short.price == pytest.approx(99.95)     # bid - slippage
    assert long.mid == pytest.approx(100.1)


def test_a_long_stops_out_on_the_bid_not_the_mid():
    """The bid reaches 98.9 while the ask is still 99.1 - the stop is hit."""
    tape = _tape([100.0, 98.9], [100.2, 99.1])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 99.0], [98.0, 98.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == pytest.approx(98.9)


def test_a_long_target_fills_on_the_bid_not_the_ask():
    """A sell limit at 103.2 needs the *bid* there; an ask of 103.3 is not a fill."""
    tape = _tape([100.0, 103.1, 103.2], [100.2, 103.3, 103.4])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 103.0, 103.2], [90.0, 90.0, 90.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=3.0, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "target"
    # Tick 1 had ask 103.3 >= 103.2 but bid 103.1: not a fill. Tick 2 is.
    assert out["exit_ts_us"] == 2 * US
    assert out["exit_price"] == pytest.approx(103.2)


def test_a_target_pays_no_slippage_because_it_is_a_limit():
    tape = _tape([100.0, 103.5], [100.2, 103.7])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 103.5], [90.0, 90.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=3.0, cfg=CFG, slip=0.5)
    assert out["exit_price"] == pytest.approx(103.2)     # exactly the limit


def test_a_market_exit_does_pay_slippage():
    tape = _tape([100.0, 98.9], [100.2, 99.1])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 99.0], [98.0, 98.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=None, cfg=CFG, slip=0.05)
    assert out["exit_price"] == pytest.approx(98.85)     # bid - slippage


def test_a_stop_and_a_target_in_one_tick_resolve_against_the_trade():
    """One tick carries a bid and an ask, not a path, so the loss is assumed."""
    tape = _tape([100.0, 97.0], [100.2, 104.0])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 100.0], [90.0, 90.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=3.0, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "stop"


def test_an_untouched_trade_is_flattened_at_the_end_of_the_window():
    tape = _tape([100.0, 100.1, 100.2], [100.2, 100.3, 100.4])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.1, 100.2], [90.0, 90.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=10.0, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "flat"


# --------------------------------------------------------------------------
# The close-only trail - the paper's headline mechanism
# --------------------------------------------------------------------------

def test_the_trail_ignores_an_intrabar_wick_through_the_ema():
    """A wick to 98.0 through a 99.0 EMA, closing at 100.0, is not an exit.

    This is the paper's whole mechanical contribution, so it is the one test
    that must never be allowed to pass by accident: the tape dips well below
    the EMA mid-bar and the position survives to the end of the window.
    """
    tape = _tape([100.0, 98.0, 100.0], [100.2, 98.2, 100.2])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 100.0], [99.0, 99.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "flat"


def test_the_trail_fires_on_a_close_beyond_the_ema():
    tape = _tape([100.0, 100.0, 100.0, 100.0], [100.2] * 4)
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    # Second forward bar closes at 98.5, under its 99.0 EMA.
    win = _window([100.0, 98.5, 100.0], [99.0, 99.0, 99.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "trail"
    assert out["exit_ts_us"] == 2 * US       # the second bar's right edge


def test_the_trail_can_be_switched_off_entirely():
    tape = _tape([100.0] * 4, [100.2] * 4)
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([100.0, 98.5, 100.0], [99.0, 99.0, 99.0])
    cfg = VWAPEmaConfig(use_trail=False)
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=cfg, slip=0.0)
    assert out["exit_reason"] == "flat"


def test_the_final_leg_switches_to_the_faster_ema_past_2_5r():
    """Eq. 6: past 2.5R the trail is the 20 EMA, which sits closer to price.

    The first forward bar puts the trade 3R up, arming the switch. The second
    closes at 100.5 - above the slow EMA at 99.0, below the fast one at 101.0.
    With the switch it exits; without it, it would not.
    """
    tape = _tape([100.0] * 4, [100.2] * 4)
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([103.2, 100.5, 100.5], trail=[99.0, 99.0, 99.0],
                  final=[101.0, 101.0, 101.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "trail"
    assert out["exit_ts_us"] == 2 * US


def test_the_final_leg_switch_is_a_ratchet_not_a_toggle():
    """Once 2.5R is passed the trail stays fast, even if profit falls back."""
    tape = _tape([100.0] * 5, [100.2] * 5)
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    # Bar 0 puts the trade 3R up and arms the switch; bar 1 falls back to 0.3R.
    # Bar 2 closes at 100.5 - above the slow EMA at 99.0, below the fast one at
    # 101.0 - so it can only exit there if the switch stayed armed through the
    # bar that fell back.
    win = _window([103.2, 100.5, 100.5], trail=[99.0, 99.0, 99.0],
                  final=[99.0, 99.0, 101.0])
    out = _simulate(tape, ent, win, risk=1.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "trail"
    # The same three bars without the opening spike never arm it, and survive.
    calm = _window([100.5, 100.5, 100.5], trail=[99.0, 99.0, 99.0],
                   final=[99.0, 99.0, 101.0])
    quiet = _simulate(tape, ent, calm, risk=1.0, target_r=None, cfg=CFG, slip=0.0)
    assert quiet["exit_reason"] == "flat"


def test_a_short_trails_on_a_close_above_the_ema():
    tape = _tape([100.0] * 4, [100.2] * 4)
    ent = _Entry(i=0, direction=-1, price=100.0, mid=100.1, ts_us=0)
    win = _window([100.0, 102.0, 100.0], [101.0, 101.0, 101.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "trail"


def test_a_stop_beats_a_trail_that_comes_due_later():
    tape = _tape([100.0, 94.0, 94.0], [100.2, 94.2, 94.2])
    ent = _Entry(i=0, direction=1, price=100.2, mid=100.1, ts_us=0)
    win = _window([94.0, 94.0], [99.0, 99.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["exit_reason"] == "stop"


# --------------------------------------------------------------------------
# R-multiples, MFE/MAE and the VWAP diagnostic
# --------------------------------------------------------------------------

def test_a_stopped_trade_loses_almost_exactly_one_r():
    tape = _tape([100.0, 95.0], [100.0, 95.0])
    ent = _Entry(i=0, direction=1, price=100.0, mid=100.0, ts_us=0)
    win = _window([100.0, 95.0], [80.0, 80.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["r_multiple"] == pytest.approx(-1.0)


def test_a_target_trade_makes_exactly_the_target_in_r():
    tape = _tape([100.0, 115.0], [100.0, 115.0])
    ent = _Entry(i=0, direction=1, price=100.0, mid=100.0, ts_us=0)
    win = _window([100.0, 115.0], [80.0, 80.0])
    out = _simulate(tape, ent, win, risk=5.0, target_r=3.0, cfg=CFG, slip=0.0)
    assert out["r_multiple"] == pytest.approx(3.0)


def test_mfe_and_mae_bracket_the_realised_r():
    tape = _tape([100.0, 108.0, 96.0, 102.0], [100.0, 108.0, 96.0, 102.0])
    ent = _Entry(i=0, direction=1, price=100.0, mid=100.0, ts_us=0)
    win = _window([108.0, 96.0, 102.0], [80.0] * 3)
    out = _simulate(tape, ent, win, risk=4.0, target_r=None, cfg=CFG, slip=0.0)
    assert out["mfe_r"] == pytest.approx(2.0)
    assert out["mae_r"] == pytest.approx(-1.0)
    assert out["mae_r"] <= out["r_multiple"] <= out["mfe_r"]


def test_the_vwap_touch_is_null_when_price_never_returns_to_vwap():
    ent = _Entry(i=0, direction=1, price=100.0, mid=100.0, ts_us=0)
    win = _window([101.0, 102.0], [99.0, 99.0], vwap=[95.0, 95.0])
    assert np.isnan(_vwap_touch(win, ent, risk=1.0))


def test_the_vwap_touch_reports_the_floating_r_at_the_first_touch():
    """For a long, entered above VWAP, a touch is an *adverse* move."""
    ent = _Entry(i=0, direction=1, price=100.0, mid=100.0, ts_us=0)
    win = _window([101.0, 94.0, 93.0], [99.0] * 3, vwap=[95.0] * 3)
    assert _vwap_touch(win, ent, risk=2.0) == pytest.approx(-3.0)


# --------------------------------------------------------------------------
# Reporting views
# --------------------------------------------------------------------------

def _trades(**cols) -> pl.DataFrame:
    n = len(next(iter(cols.values())))
    base = {
        "net_r": [0.0] * n, "exit_reason": ["stop"] * n,
        "stop_mult": [0.5] * n, "target": ["t3"] * n, "target_r": [3.0] * n,
        "hold_min": [10.0] * n,
    }
    return pl.DataFrame(base | cols)


def test_outcome_table_maps_a_target_exit_to_a_full_win():
    trades = _trades(net_r=[3.0, 0.8, 0.0, -1.0],
                     exit_reason=["target", "trail", "trail", "stop"])
    got = {r["outcome"]: r["n"] for r in outcome_table(trades).iter_rows(named=True)}
    assert got == {"full win": 1, "partial win": 1, "breakeven": 1, "loss": 1}


def test_outcome_table_breakeven_band_is_plus_or_minus_0_2r():
    trades = _trades(net_r=[0.19, 0.21, -0.19, -0.21],
                     exit_reason=["trail"] * 4)
    got = {r["outcome"]: r["n"] for r in outcome_table(trades).iter_rows(named=True)}
    assert got["breakeven"] == 2
    assert got["partial win"] == 1 and got["loss"] == 1


def test_exit_surface_has_one_row_per_swept_cell():
    trades = _trades(
        net_r=[1.0, -1.0, 2.0, 0.0],
        stop_mult=[0.5, 0.5, 1.0, 1.0],
        target=["t3", "t3", "t3", "t3"],
        target_r=[3.0] * 4,
    )
    surf = exit_surface(trades)
    assert surf.height == 2
    assert set(surf["n"].to_list()) == {2}


def test_the_placebo_keeps_the_trade_count_and_the_side_mix():
    bars = signals(_bars([_long_signal_bar() for _ in range(8)]), CFG)
    fake = placebo_signals(bars, seed=7, cfg=CFG)
    assert (fake["signal"] != 0).sum() == (bars["signal"] != 0).sum()
    assert sorted(fake["signal"].to_list()) == sorted(bars["signal"].to_list())


def test_the_placebo_only_places_signals_on_tradeable_bars():
    bars = signals(_bars([_long_signal_bar() for _ in range(8)]), CFG)
    fake = placebo_signals(bars, seed=3, cfg=CFG)
    assert not (fake.filter(~pl.col("tradeable"))["signal"] != 0).any()


def test_the_placebo_is_deterministic_in_its_seed():
    bars = signals(_bars([_long_signal_bar() for _ in range(8)]), CFG)
    a = placebo_signals(bars, seed=11, cfg=CFG)["signal"].to_list()
    b = placebo_signals(bars, seed=11, cfg=CFG)["signal"].to_list()
    assert a == b
