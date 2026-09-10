"""Tests for the Enkhbayar-Slepaczuk replication.

Three things in this module decide the result, so all three are pinned against
inputs whose answer is known by construction.

**Where the walk-forward boundaries fall.** Table 7 gives absolute bar counts,
and an off-by-one in either direction either leaks the validation block into
training or evaluates a window on bars the model already saw.

**Which bar's return a signal earns.** The whole study is a claim about
predicting ``t -> t+1``; if the backtest pays the signal ``t-1 -> t`` instead it
becomes a claim about the past and every Sharpe in the report is fiction.

**When turnover is charged.** Section 3.8 holds a position until the signal
changes, so an unchanged signal must be free. Charging it every bar would make a
2 bps cost into 500 bps a year at daily frequency and sink everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.strategies import fx_ml as fm


def _bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.003, "low": close * 0.997, "close": close},
        index=idx,
    )


# --------------------------------------------------------------------------
# Walk-forward geometry
# --------------------------------------------------------------------------


def test_window_sizes_match_table_7():
    assert fm.WalkForward.for_frequency("1d") == fm.WalkForward(600, 156, 126)
    assert fm.WalkForward.for_frequency("4h") == fm.WalkForward(3600, 936, 756)
    with pytest.raises(ValueError):
        fm.WalkForward.for_frequency("1h")


def test_windows_do_not_overlap_within_themselves_and_roll_by_test_length():
    wf = fm.WalkForward(train=10, validation=4, test=3)
    windows = wf.windows(30)
    assert len(windows) == (30 - 17) // 3 + 1
    for train, val, test in windows:
        assert train.stop == val.start          # no gap, no overlap
        assert val.stop == test.start
        assert train.stop - train.start == 10
        assert val.stop - val.start == 4
        assert test.stop - test.start == 3
    # Consecutive test blocks tile the timeline exactly once.
    starts = [t.start for _, _, t in windows]
    assert starts == list(range(14, 14 + 3 * len(windows), 3))


def test_a_corpus_shorter_than_one_window_yields_nothing():
    assert fm.WalkForward(10, 4, 3).windows(16) == []


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_paper_features_are_price_levels_and_stationary_ones_are_not():
    bars = _bars(400)
    paper = fm.build_features(bars, mode="paper")
    stat = fm.build_features(bars, mode="stationary")

    # A level feature tracks price; a rescaled one sits around 1.
    assert paper["ema50"].iloc[-1] == pytest.approx(bars["close"].iloc[-1], rel=0.15)
    assert stat["ema50"].dropna().between(0.8, 1.2).all()
    # Rescaling must not touch a bounded oscillator.
    pd.testing.assert_series_equal(paper["rsi14"], stat["rsi14"])
    assert set(paper.columns) == set(stat.columns)


def test_features_use_no_future_information():
    """Truncating the frame must not change any feature on the surviving rows."""
    bars = _bars(400)
    full = fm.build_features(bars, mode="paper")
    cut = fm.build_features(bars.iloc[:300], mode="paper")
    pd.testing.assert_frame_equal(full.iloc[:300], cut)


# --------------------------------------------------------------------------
# The signal transform (Eqs 49-51)
# --------------------------------------------------------------------------


def test_signal_gates():
    pred = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    lo, hi = -1.0, 1.0
    assert list(fm.to_signal(pred, mode="buy_sell", lo=lo, hi=hi)) == [-1, 0, 0, 0, 1]
    assert list(fm.to_signal(pred, mode="only_buy", lo=lo, hi=hi)) == [0, 0, 0, 1, 1]
    assert list(fm.to_signal(pred, mode="only_sell", lo=lo, hi=hi)) == [-1, -1, 0, 0, 0]


def test_threshold_mode_changes_how_often_the_gate_opens():
    """The paper's gate uses realised quartiles; a shrunk prediction rarely clears it."""
    res = fm.WalkForwardResult(
        index=pd.DatetimeIndex(pd.date_range("2020-01-01", periods=8, tz="UTC")),
        prediction=np.array([-0.4, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]) * 1e-3,
        forward_return=np.zeros(8),
        close=np.ones(8),
        train_q1=np.full(8, -0.01),   # realised quartiles, an order of magnitude wider
        train_q3=np.full(8, 0.01),
        window_id=np.zeros(8),
    )
    paper = fm.signal_from_result(res, signal_mode="buy_sell", threshold_mode="paper")
    pred = fm.signal_from_result(res, signal_mode="buy_sell", threshold_mode="prediction")
    assert (paper == 0).all()          # never trades
    assert (pred != 0).sum() == 4      # a quarter long, a quarter short


# --------------------------------------------------------------------------
# The backtest convention (Section 3.8)
# --------------------------------------------------------------------------


def test_signal_earns_the_forward_return_of_its_own_bar():
    sig = np.array([1.0, -1.0, 0.0])
    fwd = np.array([0.10, 0.20, 0.30])
    got = fm.backtest(sig, fwd, cost=0.0)
    assert list(got) == [0.10, -0.20, 0.0]


def test_holding_an_unchanged_signal_is_free_and_flipping_costs_twice():
    fwd = np.zeros(4)
    held = fm.backtest(np.array([1.0, 1.0, 1.0, 1.0]), fwd, cost=0.01, prev_position=1.0)
    assert list(held) == [0.0, 0.0, 0.0, 0.0]

    flip = fm.backtest(np.array([1.0, -1.0]), np.zeros(2), cost=0.01, prev_position=1.0)
    assert flip[0] == 0.0                     # already there
    assert flip[1] == pytest.approx(-0.02)    # +1 -> -1 is two units of turnover


def test_opening_from_flat_costs_one_unit():
    got = fm.backtest(np.array([1.0]), np.zeros(1), cost=0.01, prev_position=0.0)
    assert got[0] == pytest.approx(-0.01)


# --------------------------------------------------------------------------
# Trend following (Eqs 52-54)
# --------------------------------------------------------------------------


def test_ema_cross_directions_and_one_sided_modes():
    rising = pd.Series(np.linspace(1.0, 2.0, 300))
    both = fm.ema_cross_signal(rising, 10, 50, "buy_sell")
    assert both[-1] == 1.0
    assert fm.ema_cross_signal(rising, 10, 50, "only_sell")[-1] == 0.0
    falling = pd.Series(np.linspace(2.0, 1.0, 300))
    assert fm.ema_cross_signal(falling, 10, 50, "buy_sell")[-1] == -1.0
    assert fm.ema_cross_signal(falling, 10, 50, "only_buy")[-1] == 0.0


def test_ema_pairs_are_fast_slow_and_unique():
    assert len(fm.EMA_PAIRS) == 10      # Section 3.5: ten combinations
    assert all(f < s for f, s in fm.EMA_PAIRS)
    assert len(set(fm.EMA_PAIRS)) == 10


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_walkforward_evaluates_every_test_bar_exactly_once():
    bars = _bars(1000)
    wf = fm.WalkForward(train=300, validation=100, test=100)
    res = fm.run_walkforward(bars, fm.model_registry()["ridge"], wf)
    assert len(res.prediction) == len(res.index) == len(res.forward_return)
    assert res.index.is_monotonic_increasing
    assert not res.index.duplicated().any()
    assert len(np.unique(res.window_id)) == len(res.val_mae)


def test_walkforward_never_evaluates_a_bar_it_trained_on():
    bars = _bars(1000)
    wf = fm.WalkForward(train=300, validation=100, test=100)
    res = fm.run_walkforward(bars, fm.model_registry()["ridge"], wf)
    feats = fm.build_features(bars, mode="paper")
    usable = feats.replace([np.inf, -np.inf], np.nan).dropna().index
    first_evaluable = usable[wf.train + wf.validation]
    assert res.index[0] >= first_evaluable


def test_annualised_sharpe_scales_by_root_periods():
    r = np.array([0.01, -0.005, 0.02, 0.0, -0.01])
    daily = fm.annualised_sharpe(r, 252)
    four_hourly = fm.annualised_sharpe(r, 252 * 6)
    assert four_hourly == pytest.approx(daily * np.sqrt(6))
    assert np.isnan(fm.annualised_sharpe(np.zeros(10), 252))
