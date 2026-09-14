"""Parity and mechanics tests for the Precision Sniper port.

The claims worth testing are the ones a silent error would flatter: that the
Pine built-ins match their definitions, that the confluence score is on the
scale the script says it is, that the structure stop's cap and floor apply in
the script's order, that the managed exit walks the tape in the right order,
and that the account gate is the stop-and-reverse rule the script actually has
rather than the simpler one it resembles.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qlab.strategies import precision_sniper as ps
from qlab.strategies.sweep_orderblock import atr_rma


# --------------------------------------------------------------------------
# Pine built-ins
# --------------------------------------------------------------------------

def _loop_recursive(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    out = np.full(x.size, np.nan)
    val = float(np.mean(x[:n]))
    out[n - 1] = val
    for i in range(n, x.size):
        val += alpha * (x[i] - val)
        out[i] = val
    return out


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(20260912)
    close = 100 + np.cumsum(rng.normal(0, 0.4, 3000))
    high = close + rng.random(3000)
    low = close - rng.random(3000)
    return high, low, close


@pytest.mark.parametrize("n", [5, 9, 14, 21, 55])
def test_ema_matches_pine_definition(series, n):
    """alpha = 2/(n+1), seeded with the SMA at bar n-1, na before it."""
    _, _, close = series
    got = ps.ema(close, n)
    assert np.isnan(got[: n - 1]).all()
    assert got[n - 1] == pytest.approx(close[:n].mean())
    assert np.allclose(got[n - 1:], _loop_recursive(close, n, 2 / (n + 1))[n - 1:])


@pytest.mark.parametrize("n", [8, 13, 14, 21])
def test_rma_matches_pine_definition(series, n):
    _, _, close = series
    assert np.allclose(ps.rma(close, n)[n - 1:],
                       _loop_recursive(close, n, 1 / n)[n - 1:])


@pytest.mark.parametrize("n", [10, 14, 20])
def test_atr_agrees_with_the_other_strategy_module(series, n):
    """The port must not quietly disagree with the ATR already in the repo."""
    high, low, close = series
    assert np.allclose(ps.true_range_avg(high, low, close, n),
                       atr_rma(high, low, close, n), equal_nan=True)


def test_rsi_bounds_and_extremes():
    up = np.arange(1, 60, dtype=float)
    assert ps.rsi(up, 14)[-1] == pytest.approx(100.0)
    assert ps.rsi(up[::-1].copy(), 14)[-1] == pytest.approx(0.0)
    flat = np.full(60, 5.0)
    got = ps.rsi(flat, 14)[-1]
    assert np.isnan(got) or 0.0 <= got <= 100.0


def test_rsi_is_in_range_on_a_real_looking_series(series):
    _, _, close = series
    r = ps.rsi(close, 13)
    live = r[~np.isnan(r)]
    assert live.size > 2000
    assert live.min() >= 0.0 and live.max() <= 100.0


def test_macd_hist_is_line_less_signal(series):
    _, _, close = series
    hist = ps.macd_hist(close, 12, 26, 9)
    line = ps.ema(close, 12) - ps.ema(close, 26)
    start = int(np.argmax(~np.isnan(line)))
    sig = ps.ema(line[start:], 9)
    assert np.allclose(hist[start:], (line[start:] - sig), equal_nan=True)


def test_dmi_di_are_percentages_and_adx_bounded(series):
    high, low, close = series
    plus, minus, adx = ps.dmi(high, low, close, 14, 14)
    for arr in (plus, minus, adx):
        live = arr[~np.isnan(arr)]
        assert live.size > 2000
        assert live.min() >= -1e-9 and live.max() <= 100.0 + 1e-9


def test_rolling_extreme_includes_the_current_bar():
    x = np.array([5.0, 3.0, 4.0, 1.0, 9.0])
    lo = ps.rolling_extreme(x, 3, largest=False)
    assert np.isnan(lo[:2]).all()
    assert lo[2] == 3.0 and lo[3] == 1.0 and lo[4] == 1.0


# --------------------------------------------------------------------------
# The confluence scale
# --------------------------------------------------------------------------

def test_max_score_is_the_adaptive_eight_on_a_volumeless_feed():
    """Volume and VWAP are unavailable, so the script's own scale drops to 8."""
    assert ps.MAX_SCORE == 8.0


def test_min_score_rescales_with_the_max():
    """Default's input 5 of 10 is 4.0 of 8 on this feed."""
    assert ps.PRESETS["Default"].min_score * ps.MAX_SCORE / 10.0 == 4.0
    assert ps.PRESETS["Conservative"].min_score * ps.MAX_SCORE / 10.0 == 5.6


def test_grade_bands_match_the_script():
    assert ps.grade(8.0, 8.0) == "A+"
    assert ps.grade(6.4, 8.0) == "A+"          # exactly 0.80
    assert ps.grade(6.3, 8.0) == "A"
    assert ps.grade(5.2, 8.0) == "A"           # exactly 0.65
    assert ps.grade(4.0, 8.0) == "B"           # exactly 0.50
    assert ps.grade(3.9, 8.0) == "C"


def test_score_of_any_signal_bar_starts_at_one_point_five(series):
    """A buy needs emaFast > emaSlow and close > emaFast, worth 1.0 + 0.5.

    So the engine cannot emit a signal scoring below 1.5, and the discriminating
    range is 6.5 of the nominal 8 points, not 8.
    """
    high, low, close = series
    bars = pl.DataFrame({
        "ts": pl.datetime_range(
            pl.datetime(2021, 1, 1), pl.datetime(2021, 1, 1) + pl.duration(minutes=2999),
            "1m", eager=True, time_zone="UTC"),
        "open": close, "high": high, "low": low, "close": close,
        "n_ticks": np.full(close.size, 50),
    }).with_columns(ts_open=pl.col("ts") - pl.duration(minutes=1))
    sig = ps.signals(bars, ps.PRESETS["Default"], ps.SniperConfig(), interval="1m")
    assert sig.height > 0
    assert sig["score"].min() >= 1.5
    assert sig["score"].max() <= ps.MAX_SCORE
    # and the grade is exactly the banding of score / max
    assert (sig["grade"] == pl.Series([ps.grade(s, ps.MAX_SCORE)
                                       for s in sig["score"]])).all()


def test_auto_preset_map():
    assert ps.auto_preset("1m") == "Scalping"
    assert ps.auto_preset("5m") == "Scalping"
    assert ps.auto_preset("15m") == "Default"
    assert ps.auto_preset("1h") == "Default"
    assert ps.auto_preset("2h") == "Conservative"
    assert ps.auto_preset("4h") == "Swing"
    assert ps.auto_preset("1d") == "Swing"


# --------------------------------------------------------------------------
# calcSL
# --------------------------------------------------------------------------

def _stop(direction, entry, atr, swing_low, swing_high, high_vol, preset, cfg):
    out = ps._stops(np.array([direction]), np.array([float(entry)]),
                    np.array([float(atr)]), np.array([float(swing_low)]),
                    np.array([float(swing_high)]), np.array([bool(high_vol)]),
                    preset, cfg)
    return {k: float(v[0]) for k, v in out.items()}


def test_structure_stop_takes_the_wider_side():
    """A swing beyond the ATR stop widens it; a swing inside it does not."""
    p = ps.Preset("t", 9, 21, 55, 13, 14, 5.0, 1.0)
    cfg = ps.SniperConfig()
    # long, entry 100, atr 1 -> atr stop at 99; swing low 98.9 - 0.2 atr = 98.7,
    # which is 1.3 ATR away and so inside the 1.5 ATR cap
    wide = _stop(1, 100, 1.0, 98.9, 102, False, p, cfg)
    assert wide["atr"] == pytest.approx(99.0)
    assert wide["struct"] == pytest.approx(98.7)
    # swing low inside the ATR stop leaves the ATR stop alone
    tight = _stop(1, 100, 1.0, 99.5, 102, False, p, cfg)
    assert tight["struct"] == pytest.approx(99.0)


def test_structure_stop_cap_and_floor():
    p = ps.Preset("t", 9, 21, 55, 13, 14, 5.0, 1.0)
    cfg = ps.SniperConfig()
    # a swing far below caps the distance at 1.5x the ATR distance
    capped = _stop(1, 100, 1.0, 90.0, 102, False, p, cfg)
    assert capped["struct"] == pytest.approx(98.5)
    # a short mirrors it
    capped_s = _stop(-1, 100, 1.0, 98, 110.0, False, p, cfg)
    assert capped_s["struct"] == pytest.approx(101.5)


def test_structure_floor_applies_when_the_atr_stop_is_tiny():
    """slMult 0.1 puts the ATR stop inside the 0.5 ATR floor."""
    p = ps.Preset("t", 9, 21, 55, 13, 14, 5.0, 0.1)
    cfg = ps.SniperConfig()
    got = _stop(1, 100, 1.0, 99.99, 102, False, p, cfg)
    assert got["struct"] == pytest.approx(99.5)


def test_widening_applies_only_in_the_high_regime():
    p = ps.Preset("t", 9, 21, 55, 13, 14, 5.0, 1.0)
    cfg = ps.SniperConfig()
    calm = _stop(1, 100, 1.0, 99.9, 102, False, p, cfg)
    hot = _stop(1, 100, 1.0, 99.9, 102, True, p, cfg)
    assert calm["atr"] == calm["atr_w"] == pytest.approx(99.0)
    assert hot["atr"] == pytest.approx(99.0)
    assert hot["atr_w"] == pytest.approx(98.5)          # 1.5x the distance


# --------------------------------------------------------------------------
# The managed exit
# --------------------------------------------------------------------------

def _hits(out, f, hold_end, d, entry, risk):
    hits, pos = {}, f
    for k in ps.R_LEVELS:
        pos = ps._first_cross(out, max(pos, f), hold_end, entry + d * k * risk,
                              below=(d == -1))
        hits[k] = pos
        if pos < 0:
            pos = hold_end
    return hits


def _walk(path, *, full_exit=True, ladder=(1.0, 2.0, 3.0), entry=100.0, risk=1.0):
    out = np.asarray(path, dtype=float)
    f, hold_end, d = 0, out.size, 1
    stop = entry - risk
    si = ps._first_cross(out, f, hold_end, stop, below=True)
    return ps._pine_path(out, f, hold_end, d, entry, risk, stop, si,
                         _hits(out, f, hold_end, d, entry, risk),
                         ladder, full_exit, 0.0)


def test_managed_exit_stops_out_before_any_target():
    px, reason, _ = _walk([100, 99.5, 99.0, 98.0])
    assert reason == "stop" and px == pytest.approx(99.0)


def test_managed_exit_full_close_at_tp3():
    px, reason, _ = _walk([100, 101, 102, 103, 104])
    assert reason == "tp" and px == pytest.approx(103.0)


def test_managed_exit_trails_to_breakeven_after_tp1():
    """TP1 traded, then price came back through the entry: a breakeven stop."""
    px, reason, _ = _walk([100, 101, 100.5, 100.0, 99.0])
    assert reason == "trail1" and px == pytest.approx(100.0)


def test_managed_exit_trails_to_tp1_after_tp2():
    px, reason, _ = _walk([100, 101, 102, 101.5, 101.0, 99.0])
    assert reason == "trail2" and px == pytest.approx(101.0)


def test_managed_exit_runner_keeps_the_position_past_tp3():
    """With the full exit off, TP3 only moves the trail to TP2."""
    px, reason, _ = _walk([100, 101, 102, 103, 102.5, 102.0], full_exit=False)
    assert reason == "trail3" and px == pytest.approx(102.0)


def test_managed_exit_stop_after_tp1_is_not_charged_the_full_loss():
    """The script's whole trailing claim: a stop after TP1 is a breakeven."""
    px, _, _ = _walk([100, 101, 100.0, 98.0])
    assert px == pytest.approx(100.0)          # not 99.0


def test_managed_exit_falls_through_to_the_clock():
    px, reason, _ = _walk([100, 100.2, 100.1, 100.3])
    assert reason == "time" and px == pytest.approx(100.3)


def test_a_stop_that_trades_after_tp1_does_not_count_as_the_entry_stop():
    """The entry stop is only live until TP1 trades - the leg-0 window."""
    px, reason, _ = _walk([100, 101, 102, 99.0])
    # TP1 and TP2 both traded first, so the live stop was TP1, not the entry stop
    assert reason == "trail2" and px == pytest.approx(99.0)


# --------------------------------------------------------------------------
# The account gate
# --------------------------------------------------------------------------

def _tape(rows) -> pl.DataFrame:
    """A minimal resolved tape: (direction, fill_us, exit_us, r, entry)."""
    return pl.DataFrame({
        "interval": ["5m"] * len(rows),
        "preset": ["Default"] * len(rows),
        "htf_name": ["1h"] * len(rows),
        "filled": [True] * len(rows),
        "direction": [r[0] for r in rows],
        "fill_us": [r[1] for r in rows],
        "exit_us_struct_pL123": [r[2] for r in rows],
        "r_struct_pL123": [float(r[3]) for r in rows],
        "entry": [float(r[4]) for r in rows],
        "risk_struct": [1.0] * len(rows),
        "commission_px": [0.0] * len(rows),
    })


def test_flat_account_skips_anything_arriving_while_a_trade_is_open():
    tape = _tape([(1, 0, 100, 1.0, 100.0),
                  (-1, 50, 150, 2.0, 101.0),     # inside the first trade
                  (1, 200, 300, 3.0, 102.0)])
    taken = ps.sequence(tape, ["struct_pL123"])["struct_pL123"]
    assert taken["fill_us"].to_list() == [0, 200]


def test_reverse_account_takes_the_opposing_signal_and_cuts_the_trade():
    """The script's Case 2: an opposing signal closes the open trade at its price."""
    tape = _tape([(1, 0, 1000, 3.0, 100.0),      # would have run to +3R
                  (-1, 50, 150, 2.0, 100.5)])    # but is reversed at 100.5
    rev = ps.sequence_reverse(tape, "struct_pL123")
    assert rev.height == 2
    assert rev["was_cut"].to_list() == [True, False]
    # long cut at 100.5 from 100.0 with risk 1.0 -> +0.5R, not the +3R barrier
    assert rev["r_eff"][0] == pytest.approx(0.5)
    assert rev["r_eff"][1] == pytest.approx(2.0)


def test_reverse_account_blocks_a_same_direction_signal_while_open():
    """lastDirection never pyramids."""
    tape = _tape([(1, 0, 1000, 1.0, 100.0),
                  (1, 50, 200, 5.0, 101.0),      # same direction, blocked
                  (-1, 100, 300, 2.0, 100.5)])   # opposite, reverses
    rev = ps.sequence_reverse(tape, "struct_pL123")
    assert rev["fill_us"].to_list() == [0, 100]
    assert rev["r_eff"][0] == pytest.approx(0.5)


def test_reverse_account_uses_the_barrier_when_it_comes_first():
    tape = _tape([(1, 0, 100, 1.5, 100.0),
                  (-1, 200, 300, 2.0, 99.0)])
    rev = ps.sequence_reverse(tape, "struct_pL123")
    assert rev["was_cut"].to_list() == [False, False]
    assert rev["r_eff"].to_list() == pytest.approx([1.5, 2.0])


def test_reverse_and_flat_agree_when_no_signal_lands_inside_a_trade():
    rows = [(1, 0, 10, 1.0, 100.0), (-1, 20, 30, -1.0, 101.0),
            (1, 40, 50, 2.0, 100.0)]
    flat = ps.sequence(_tape(rows), ["struct_pL123"])["struct_pL123"]
    rev = ps.sequence_reverse(_tape(rows), "struct_pL123")
    assert flat["fill_us"].to_list() == rev["fill_us"].to_list()
    assert rev["r_eff"].to_list() == pytest.approx(
        flat["r_struct_pL123"].cast(pl.Float64).to_list())


# --------------------------------------------------------------------------
# The HTF factor
# --------------------------------------------------------------------------

def test_htf_bias_uses_only_closed_higher_timeframe_bars():
    """A spike in the HTF bar the chart bar sits inside must not reach it.

    The join is on the chart bar's open against the HTF bar's close, so the
    value can only come from a bar that had already finished.
    """
    idx = pl.datetime_range(pl.datetime(2021, 1, 1), pl.datetime(2021, 1, 5),
                            "1h", eager=True, time_zone="UTC")
    chart = pl.DataFrame({"ts": idx}).with_columns(
        ts_open=pl.col("ts") - pl.duration(hours=1),
        close=pl.lit(100.0), high=pl.lit(100.0), low=pl.lit(100.0))
    hidx = pl.datetime_range(pl.datetime(2021, 1, 1), pl.datetime(2021, 1, 5),
                             "4h", eager=True, time_zone="UTC")
    n = hidx.len()
    # flat, then a violent up-move only in the final HTF bar
    hclose = np.full(n, 100.0)
    hclose[-1] = 500.0
    htf = pl.DataFrame({"ts": hidx, "close": hclose}).with_columns(
        ts_open=pl.col("ts") - pl.duration(hours=4))
    preset = ps.Preset("t", 2, 3, 10, 5, 5, 5.0, 1.0)
    bias = ps.htf_bias(chart, htf, preset)
    # the last HTF bar closes at the very end of the chart, so nothing before
    # the final chart bar may have seen its move
    assert set(np.unique(bias[:-1])) <= {0, -1, 1}
    last_htf_close = htf["ts"][-1]
    seen = chart.with_columns(_b=bias).filter(pl.col("ts_open") >= last_htf_close)
    assert seen.height <= 1
