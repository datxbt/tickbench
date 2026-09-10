"""Structural-break entries: pivots, tapes, the overlay, and sizing.

Four things are worth asserting, and they map onto the four places this
strategy could be flattered by its own harness.

**The pivot cannot be known early.** A swing high is a local maximum, and a
local maximum is only local once the bars on its right exist. Every backtest of
every pivot-based system dies on this one, and it dies quietly: the equity curve
just gets better. So the test constructs a series with one unmistakable peak and
asserts the exact bar on which it becomes visible.

**The tape decides the fill, and the two tapes must disagree the right way.**
On a real tick tape a stop fills at the quote that crossed it; on a synthetic
monotone path there is no such quote and the fill belongs at the level, because
taking the bar's extreme instead charges the account for an excursion the model
invented. This is the difference the fidelity axis is measuring, so it has to be
correct in the code rather than merely different.

**The overlay does what it says.** Half the position, at +1R, only on a
counter-directional minute, with the stop then at breakeven - and never on a
losing minute that merely happens to be counter-directional.

**Sizing scales and nothing more.** The multipliers are bounded, the R-multiple
is invariant to them, and Sharpe is invariant to leverage. That last identity is
the one the whole sizing axis is read against.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.structural_break import (
    MODES,
    BreakConfig,
    ExitConfig,
    SizingConfig,
    Tape,
    _atr,
    confirmed_pivots,
    ma_crossover_entries,
    random_entries,
    sharpe_invariance_check,
    size_multiplier,
    synthetic_tape,
)

UTC = timezone.utc
START = datetime(2024, 6, 3, 0, 0, tzinfo=UTC)


def bars(opens, highs, lows, closes, spread: float = 0.02) -> pl.DataFrame:
    """A minimal M1 frame with everything ``synthetic_tape`` reads."""
    n = len(opens)
    ts_open = [START + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame({
        "ts": [t + timedelta(minutes=1) for t in ts_open],
        "ts_open": ts_open,
        "open": list(map(float, opens)),
        "high": list(map(float, highs)),
        "low": list(map(float, lows)),
        "close": list(map(float, closes)),
        "spread_close": [spread] * n,
        "bid_close": [c - spread / 2 for c in closes],
        "ask_close": [c + spread / 2 for c in closes],
    })


# --------------------------------------------------------------------------
# Pivots
# --------------------------------------------------------------------------

def test_a_pivot_is_invisible_until_w_bars_after_it():
    """The lookahead test. The peak sits at index 10; W is 3.

    It must be absent at bar 12 and present at bar 13, exactly. One bar early
    and the strategy is trading on a bar that has not happened.
    """
    close = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20,
                      9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    highs, _ = confirmed_pivots(close, w=3)
    assert not np.isfinite(highs[12])
    assert highs[13] == pytest.approx(20.0)
    assert highs[-1] == pytest.approx(20.0)   # and it persists


def test_a_swing_low_is_the_mirror_image():
    close = np.array([20.0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 1,
                      12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)
    _, lows = confirmed_pivots(close, w=3)
    assert not np.isfinite(lows[12])
    assert lows[13] == pytest.approx(1.0)


def test_a_monotone_series_has_no_confirmed_pivots():
    highs, lows = confirmed_pivots(np.arange(60, dtype=float), w=5)
    assert not np.isfinite(highs).any()
    assert not np.isfinite(lows).any()


def test_a_later_pivot_replaces_an_earlier_one():
    close = np.concatenate([
        np.array([1.0, 2, 3, 9, 3, 2, 1]),      # peak at 9
        np.array([2.0, 3, 4, 15, 4, 3, 2, 1]),  # then a higher one at 15
    ])
    highs, _ = confirmed_pivots(close, w=2)
    assert highs[6] == pytest.approx(9.0)
    assert highs[-1] == pytest.approx(15.0)


def test_atr_is_positive_and_warms_up():
    n = 60
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 2, n))
    high, low = close + 3.0, close - 3.0
    atr = _atr(high, low, close, 14)
    assert not np.isfinite(atr[:14]).any()
    assert np.all(atr[14:] > 0)


# --------------------------------------------------------------------------
# Tapes
# --------------------------------------------------------------------------

def test_interpolated_tape_lays_four_points_per_minute_in_candle_order():
    """A down minute is traversed high-first; an up minute low-first."""
    frame = bars(opens=[100, 100], highs=[105, 105], lows=[95, 95],
                 closes=[103, 97])          # first up, second down
    tape = synthetic_tape(frame, "interp")
    assert tape.interpolated and tape.mode == "interp"
    assert tape.ts.size == 8
    mid = (tape.bid + tape.ask) / 2
    assert list(np.round(mid[:4], 2)) == [100.0, 95.0, 105.0, 103.0]
    assert list(np.round(mid[4:], 2)) == [100.0, 105.0, 95.0, 97.0]


def test_interpolated_tape_timestamps_stay_inside_their_minute():
    frame = bars([100] * 3, [101] * 3, [99] * 3, [100.5] * 3)
    tape = synthetic_tape(frame, "interp")
    assert np.all(np.diff(tape.ts) >= 0)
    ends = np.repeat(tape.minute_end, 4)
    assert np.all(tape.ts < ends)


def test_close_only_tape_is_one_real_quote_per_minute():
    """Not interpolated: nothing is claimed about the path between closes."""
    frame = bars([100] * 5, [110] * 5, [90] * 5, [101, 102, 103, 104, 105])
    tape = synthetic_tape(frame, "close")
    assert not tape.interpolated
    assert tape.ts.size == 5
    assert np.allclose((tape.bid + tape.ask) / 2, [101, 102, 103, 104, 105])


def test_the_spread_is_carried_onto_the_synthetic_tape():
    frame = bars([100] * 2, [101] * 2, [99] * 2, [100.5] * 2, spread=0.40)
    tape = synthetic_tape(frame, "interp")
    assert np.allclose(tape.ask - tape.bid, 0.40)


def test_minute_direction_flags_match_the_candles():
    frame = bars([100, 100, 100], [101] * 3, [99] * 3, [101, 99, 100])
    tape = synthetic_tape(frame, "interp")
    assert tape.minute_up.tolist() == [True, False, False]


def test_index_lookups_are_half_open_and_ordered():
    frame = bars([100] * 4, [101] * 4, [99] * 4, [100.5] * 4)
    tape = synthetic_tape(frame, "close")
    first = int(tape.ts[0])
    assert tape.index_at(first) == 0
    assert tape.index_at(first + 1) == 1
    assert tape.index_at(int(tape.ts[-1]) + 10**9) == tape.ts.size


# --------------------------------------------------------------------------
# The exit walk
# --------------------------------------------------------------------------

def _run_one(frame: pl.DataFrame, *, direction: int, entry: float,
             exits: ExitConfig, cfg: BreakConfig | None = None,
             mode: str = "interp", lots: float = 1.0):
    """Drive ``_manage`` directly on one position, with no entry logic at all."""
    from qlab.engine import Fill, Trade
    from qlab.strategies.structural_break import _manage

    cfg = cfg or BreakConfig()
    tape = synthetic_tape(frame, mode)
    trade = Trade(
        symbol="XAUUSD", direction=direction,
        entry=Fill(int(tape.ts[0]), entry, lots, "entry", entry),
        stop_price=entry - direction * cfg.delta_sl,
        contract_size=100.0, commission_per_lot_side=0.0,
        setup={"half_fired": 0},
    )
    stats = {"half_fired": 0, "max_hold": 0}
    _manage(trade, tape, 0, int(tape.ts[-1]) + 1, cfg, exits, 0.0, stats)
    return trade, stats


def test_a_stop_on_an_interpolated_tape_fills_at_its_level():
    """Not at the bar's extreme, which the model invented.

    The minute's low is 1990 but the stop is at 1997; a monotone path from the
    open passes through 1997 on the way down, so that is the fill.
    """
    frame = bars(opens=[2000, 2000], highs=[2001, 2001],
                 lows=[2000, 1990], closes=[2000, 1991], spread=0.0)
    trade, _ = _run_one(frame, direction=1, entry=2000.0, exits=ExitConfig())
    assert trade.exit_reason == "stop"
    assert trade.exits[-1].price == pytest.approx(1997.0)


def test_a_target_fills_at_its_own_price():
    """A take profit is a limit: it fills at its price or not at all."""
    frame = bars(opens=[2000, 2000], highs=[2001, 2030],
                 lows=[1999, 2000], closes=[2000, 2029], spread=0.0)
    trade, _ = _run_one(frame, direction=1, entry=2000.0,
                        exits=ExitConfig(half_exit=False))
    assert trade.exit_reason == "tp"
    assert trade.exits[-1].price == pytest.approx(2020.0)


def test_the_stop_beats_the_target_when_the_path_reaches_it_first():
    """A down candle is traversed high-first, so a short's stop comes first."""
    frame = bars(opens=[2000, 2000], highs=[2001, 2004],
                 lows=[1999, 1975], closes=[2000, 1976], spread=0.0)
    trade, _ = _run_one(frame, direction=-1, entry=2000.0,
                        exits=ExitConfig(half_exit=False))
    assert trade.exit_reason == "stop"
    assert trade.exits[-1].price == pytest.approx(2003.0)


def test_the_half_exit_fires_at_one_r_on_a_counter_directional_minute():
    frame = bars(opens=[2000, 2004], highs=[2001, 2005],
                 lows=[2000, 2003], closes=[2000, 2003.5], spread=0.0)
    trade, stats = _run_one(frame, direction=1, entry=2000.0,
                            exits=ExitConfig())
    assert stats["half_fired"] == 1
    half = trade.exits[0]
    assert half.reason == "half"
    assert half.lots == pytest.approx(0.5)
    assert half.price == pytest.approx(2003.5)


def test_the_half_exit_does_not_fire_on_an_aligned_minute():
    """Same profit, same level - the candle closed *up*, so nothing happens."""
    frame = bars(opens=[2000, 2003], highs=[2001, 2005],
                 lows=[2000, 2003], closes=[2000, 2004.0], spread=0.0)
    _, stats = _run_one(frame, direction=1, entry=2000.0, exits=ExitConfig())
    assert stats["half_fired"] == 0


def test_dropping_the_candle_filter_fires_on_the_same_minute():
    """V11_FixedR: +1R alone is the trigger."""
    frame = bars(opens=[2000, 2003], highs=[2001, 2005],
                 lows=[2000, 2003], closes=[2000, 2004.0], spread=0.0)
    _, stats = _run_one(frame, direction=1, entry=2000.0,
                        exits=ExitConfig(require_counter_candle=False))
    assert stats["half_fired"] == 1


def test_the_half_exit_never_fires_below_its_threshold():
    frame = bars(opens=[2000, 2002], highs=[2001, 2002.5],
                 lows=[2000, 2001], closes=[2000, 2001.5], spread=0.0)
    _, stats = _run_one(frame, direction=1, entry=2000.0, exits=ExitConfig())
    assert stats["half_fired"] == 0


def test_switching_the_overlay_off_leaves_a_single_exit():
    frame = bars(opens=[2000, 2004, 2004], highs=[2001, 2005, 2005],
                 lows=[2000, 2003, 1996], closes=[2000, 2003.5, 1996.5],
                 spread=0.0)
    trade, stats = _run_one(frame, direction=1, entry=2000.0,
                            exits=ExitConfig(half_exit=False))
    assert stats["half_fired"] == 0
    assert len(trade.exits) == 1
    assert trade.exit_reason == "stop"


def test_the_stop_moves_to_breakeven_after_the_half():
    """And the remainder then exits at the entry price, not at -1R."""
    frame = bars(opens=[2000, 2004, 2003], highs=[2001, 2005, 2003.5],
                 lows=[2000, 2003, 1998], closes=[2000, 2003.5, 1998.5],
                 spread=0.0)
    trade, stats = _run_one(frame, direction=1, entry=2000.0,
                            exits=ExitConfig())
    assert stats["half_fired"] == 1
    assert trade.stop_price == pytest.approx(2000.0)
    assert trade.exits[-1].reason == "be"
    assert trade.exits[-1].price == pytest.approx(2000.0)
    assert trade.closed


def test_a_half_then_target_beats_a_bare_target_on_win_rate_and_loses_on_size():
    """The overlay's whole trade-off, in one comparison.

    Taking half at +1R caps a 6.67R winner at roughly 3.8R. That is the
    mechanism by which a rule that raises the win rate can lower the return.
    """
    frame = bars(opens=[2000, 2004, 2010], highs=[2001, 2005, 2025],
                 lows=[2000, 2003, 2009], closes=[2000, 2003.5, 2024],
                 spread=0.0)
    with_half, _ = _run_one(frame, direction=1, entry=2000.0,
                            exits=ExitConfig())
    without, _ = _run_one(frame, direction=1, entry=2000.0,
                          exits=ExitConfig(half_exit=False))
    assert with_half.gross_usd < without.gross_usd
    assert without.gross_usd == pytest.approx(20.0 * 100.0)


def test_an_unresolved_position_closes_at_the_deadline():
    frame = bars([2000] * 5, [2001] * 5, [1999] * 5, [2000] * 5, spread=0.0)
    trade, stats = _run_one(frame, direction=1, entry=2000.0,
                            exits=ExitConfig(half_exit=False))
    assert trade.exit_reason == "time"
    assert stats["max_hold"] == 1
    assert trade.closed


def test_excursions_are_recorded_in_the_position_s_own_favour():
    frame = bars(opens=[2000, 2000], highs=[2001, 2008],
                 lows=[1999, 1998], closes=[2000, 2007], spread=0.0)
    trade, _ = _run_one(frame, direction=1, entry=2000.0,
                        exits=ExitConfig(half_exit=False))
    assert trade.mfe_price > 0
    assert trade.mae_price < 0


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def test_v10_is_always_exactly_one():
    cfg = SizingConfig(rule="V10")
    for rho in (0.1, 1.0, 9.0):
        assert size_multiplier(cfg, direction=1, rho=rho, trend=1,
                               rvol=40.0) == 1.0


def test_the_atr_term_shrinks_when_short_horizon_volatility_runs_hot():
    cfg = SizingConfig(rule="V11_ATR")
    hot = size_multiplier(cfg, direction=1, rho=2.0, trend=0, rvol=15.0)
    calm = size_multiplier(cfg, direction=1, rho=0.5, trend=0, rvol=15.0)
    assert hot < 1.0 < calm


def test_the_trend_term_reads_the_trade_s_own_side():
    cfg = SizingConfig(rule="V11_TREND")
    assert size_multiplier(cfg, direction=1, rho=1.0, trend=1, rvol=15.0) \
        == pytest.approx(cfg.aligned_mult)
    assert size_multiplier(cfg, direction=1, rho=1.0, trend=-1, rvol=15.0) \
        == pytest.approx(cfg.opposed_mult)
    assert size_multiplier(cfg, direction=1, rho=1.0, trend=0, rvol=15.0) \
        == pytest.approx(cfg.flat_mult)


def test_vol_targeting_is_the_ratio_of_target_to_realised():
    cfg = SizingConfig(rule="V11_VOLTGT", target_vol_pct=15.0)
    assert size_multiplier(cfg, direction=1, rho=1.0, trend=0, rvol=30.0) \
        == pytest.approx(0.5)
    assert size_multiplier(cfg, direction=1, rho=1.0, trend=0, rvol=10.0) \
        == pytest.approx(1.5)


@pytest.mark.parametrize("rule", ["V11", "V11_ATR", "V11_TREND", "V11_VOLTGT"])
def test_every_multiplier_respects_its_clip(rule):
    cfg = SizingConfig(rule=rule)
    lo, hi = cfg.clip
    for rho, trend, rvol in [(1e-6, 1, 1e-6), (1e6, -1, 1e6), (1.0, 0, 15.0)]:
        assert lo <= size_multiplier(cfg, direction=1, rho=rho, trend=trend,
                                     rvol=rvol) <= hi


def test_a_missing_regime_falls_back_to_neutral_rather_than_to_nan():
    cfg = SizingConfig(rule="V11")
    got = size_multiplier(cfg, direction=1, rho=float("nan"), trend=0,
                          rvol=float("nan"))
    assert np.isfinite(got) and got > 0


def test_an_unknown_sizing_rule_is_refused():
    with pytest.raises(ValueError, match="unknown sizing rule"):
        SizingConfig(rule="V12")


def test_sharpe_is_invariant_to_pure_leverage():
    """The identity the whole sizing axis is read against."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0004, 0.01, 900)
    a, b = sharpe_invariance_check(r, 3.7)
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------
# Configuration and benchmarks
# --------------------------------------------------------------------------

def test_an_unknown_arm_mode_is_refused():
    with pytest.raises(ValueError, match="unknown arm mode"):
        BreakConfig(arm="whenever")


def test_a_non_positive_stop_is_refused():
    with pytest.raises(ValueError, match="positive"):
        BreakConfig(delta_sl=0.0)


def test_the_exit_config_names_itself_for_the_report():
    assert ExitConfig().label.startswith("V10")
    assert "no half exit" in ExitConfig(half_exit=False).label
    assert ExitConfig(require_counter_candle=False).label == "V11_FixedR"


def test_the_three_fidelity_modes_are_the_declared_ones():
    assert MODES == ("ticks", "interp", "close")


def _fake_ctx(closes):
    """Just enough context for the entry-generating helpers."""
    from qlab.strategies.structural_break import Context

    n = len(closes)
    h4 = pl.DataFrame({
        "ts": [START + timedelta(hours=4 * (i + 1)) for i in range(n)],
        "ts_open": [START + timedelta(hours=4 * i) for i in range(n)],
        "close": list(map(float, closes)),
    })
    return Context(h4=h4, bars_1m=pl.DataFrame(), regime=pl.DataFrame(),
                   cfg=BreakConfig(), sizing=SizingConfig())


def test_ma_crossover_fires_on_the_turn_and_carries_its_side():
    closes = list(np.concatenate([np.linspace(100, 60, 90),
                                  np.linspace(60, 140, 90)]))
    entries = ma_crossover_entries(_fake_ctx(closes), fast=5, slow=20)
    assert entries.shape[1] == 2
    assert set(np.unique(entries[:, 1]).tolist()) <= {-1, 1}
    assert (entries[:, 1] == 1).any() and (entries[:, 1] == -1).any()


def test_random_entries_are_drawn_from_the_sample_s_own_clock():
    ctx = _fake_ctx(list(np.arange(200, dtype=float)))
    stamps = ctx.h4["ts"].dt.epoch("us").to_numpy()
    entries = random_entries(ctx, 40, seed=3)
    assert entries.shape == (40, 2)
    assert set(entries[:, 0].tolist()) <= set(stamps.tolist())
    assert (np.diff(entries[:, 0]) > 0).all()          # sorted, no repeats


def test_random_entries_are_reproducible_and_seed_dependent():
    ctx = _fake_ctx(list(np.arange(200, dtype=float)))
    assert np.array_equal(random_entries(ctx, 30, 5), random_entries(ctx, 30, 5))
    assert not np.array_equal(random_entries(ctx, 30, 5),
                              random_entries(ctx, 30, 6))
