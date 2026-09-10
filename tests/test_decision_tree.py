"""Tests for the decision-tree intraday strategy.

The failure modes this module has to exclude are not the ones a Sharpe would
show, because a machine-learning backtest fails *silently* and upwards:

* **Lookahead.** The label is the only column allowed to see the future. If a
  feature does too, the accuracy goes up, the equity curve goes up, and nothing
  raises. Several tests here perturb a single future bar and assert that no
  feature at any earlier bar moves.
* **The lag chain.** ``signal_lag`` has to shift the *prediction*, and the
  position on a row has to be the one that earns that row's ``ret_fwd``. An
  off-by-one in either direction is worth more than the entire measured edge.
* **The indicator arithmetic.** RSI and ADX are Wilder-smoothed, and ADX in
  particular has a 0/0 at the first bar whose NaN, once it enters a recursive
  smoother, silently poisons every later value.
* **Cost accounting.** ``net = gross - cost``, and turnover has to charge a
  long/short flip twice.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest
from sklearn.tree import DecisionTreeClassifier

from qlab.strategies.decision_tree import (
    FEATURES,
    PAPER_PERIODS_PER_YEAR,
    TreeConfig,
    accuracy,
    benchmark,
    breakeven_cost_bps,
    clean,
    compute_features,
    daily,
    fit,
    paper_style_sharpe,
    positions,
    used_features,
)

CFG = TreeConfig()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _bars(closes, *, spread: float = 0.02, ticks: float = 100.0) -> pl.DataFrame:
    """A synthetic 1-minute frame with the columns the features read."""
    base = datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)
    closes = list(map(float, closes))
    n = len(closes)
    return pl.DataFrame({
        "ts": [base + timedelta(minutes=i + 1) for i in range(n)],
        "ts_open": [base + timedelta(minutes=i) for i in range(n)],
        "open": [closes[max(i - 1, 0)] for i in range(n)],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "spread_mean": [spread] * n,
        "n_ticks": [ticks] * n,
    })


def _rising(n: int = 80) -> pl.DataFrame:
    return _bars([100.0 + i for i in range(n)])


def _wiggly(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return _bars(100.0 + np.cumsum(rng.normal(0, 0.1, n)))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_mapping_is_validated():
    with pytest.raises(ValueError):
        TreeConfig(mapping="long_only")


def test_a_negative_signal_lag_is_refused_because_it_is_lookahead():
    with pytest.raises(ValueError):
        TreeConfig(signal_lag=-1)


def test_depth_must_be_at_least_one():
    with pytest.raises(ValueError):
        TreeConfig(max_depth=0)


def test_the_paper_annualisation_is_their_stated_252_by_375():
    assert PAPER_PERIODS_PER_YEAR == 94_500


# --------------------------------------------------------------------------
# The label, and the absence of lookahead anywhere else
# --------------------------------------------------------------------------

def test_the_label_is_the_sign_of_the_next_bar_s_return():
    f = compute_features(_bars([100.0, 101.0, 100.5, 100.5, 102.0]), CFG)
    assert f["label"].to_list()[:3] == [1, 0, 0]     # up, down, flat -> 0
    assert f["ret_fwd"][0] == pytest.approx(0.01)


def test_the_forward_return_is_null_on_the_last_bar_and_clean_drops_it():
    f = compute_features(_wiggly(300), CFG)
    assert f["ret_fwd"][-1] is None
    assert clean(f).height < f.height


def test_no_feature_sees_a_future_bar():
    """Perturb only the last close; every feature before it must be unchanged.

    This is the single test that stands between this module and the most
    flattering bug available to it.
    """
    closes = list(100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, 300)))
    a = compute_features(_bars(closes), CFG)
    b = compute_features(_bars(closes[:-1] + [closes[-1] + 25.0]), CFG)
    for name in FEATURES:
        left = a[name].to_numpy()[:-1]
        right = b[name].to_numpy()[:-1]
        assert np.allclose(left, right, equal_nan=True), f"{name} sees the future"


def test_only_the_forward_return_moves_when_the_future_changes():
    closes = list(100.0 + np.cumsum(np.random.default_rng(2).normal(0, 0.1, 200)))
    a = compute_features(_bars(closes), CFG)
    b = compute_features(_bars(closes[:-1] + [closes[-1] + 5.0]), CFG)
    assert a["ret_fwd"][-2] != b["ret_fwd"][-2]
    assert a["close"][-2] == b["close"][-2]


# --------------------------------------------------------------------------
# The nine features
# --------------------------------------------------------------------------

def test_all_nine_features_are_produced():
    f = compute_features(_wiggly(300), CFG)
    assert set(FEATURES) <= set(f.columns)


def test_rsi_is_100_when_every_move_is_a_gain():
    f = compute_features(_rising(60), CFG)
    assert f["rsi14"].to_list()[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_every_move_is_a_loss():
    f = compute_features(_bars([200.0 - i for i in range(60)]), CFG)
    assert f["rsi14"].to_list()[-1] == pytest.approx(0.0, abs=1e-9)


def test_adx_is_never_nan():
    """The 0/0 at the first bar, which a recursive smoother would propagate.

    This is a regression test: the first bar has no previous high, so both
    directional movements are zero while true range is positive, and taking
    ``100 * |0 - 0| / 0`` there turned every subsequent ADX into NaN.
    """
    f = compute_features(_wiggly(300), CFG)
    assert f["adx14"].is_finite().all()
    assert f["adx14"].null_count() == 0


def test_adx_is_high_in_a_clean_trend_and_low_in_noise():
    trend = compute_features(_rising(200), CFG)["adx14"][-1]
    noise = compute_features(_bars([100.0 + (i % 2) for i in range(200)]), CFG)["adx14"][-1]
    assert trend > 90.0
    assert noise < 20.0


def test_the_sma_ratio_is_below_one_in_an_uptrend():
    """A rising close outruns its own trailing mean, so SMA/close < 1."""
    f = compute_features(_rising(60), CFG)
    assert f["sma_ratio"][-1] < 1.0


def test_the_long_return_spans_fifteen_bars():
    f = compute_features(_bars([100.0] * 20 + [110.0]), CFG)
    assert f["ret15"][-1] == pytest.approx(0.10)


def test_vwap_ratio_is_one_when_price_and_volume_are_flat():
    """Typical price equals close on a flat series, so the ratio is exactly 1."""
    f = compute_features(_bars([100.0] * 40), CFG)
    assert f["vwap_ratio"][-1] == pytest.approx(1.0)


def test_clean_removes_rows_with_a_non_finite_feature():
    frame = compute_features(_wiggly(300), CFG)
    poisoned = frame.with_columns(
        vol14=pl.when(pl.int_range(pl.len()) == 100).then(float("nan"))
        .otherwise(pl.col("vol14"))
    )
    assert clean(poisoned).height == clean(frame).height - 1


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_the_fitted_tree_respects_the_configured_depth():
    train = clean(compute_features(_wiggly(2000), CFG))
    for depth in (2, 3, 5):
        model = fit(train, TreeConfig(max_depth=depth))
        assert model.get_depth() <= depth


def test_used_features_lists_only_features_the_tree_splits_on():
    train = clean(compute_features(_wiggly(2000), CFG))
    model = fit(train, TreeConfig(max_depth=2))
    used = used_features(model)
    assert set(used) <= set(FEATURES)
    assert 0 < len(used) <= 3          # a depth-2 tree has at most 3 splits


def test_the_fit_is_deterministic_in_its_random_state():
    train = clean(compute_features(_wiggly(2000), CFG))
    a = fit(train, CFG).predict(train.select(FEATURES).to_numpy())
    b = fit(train, CFG).predict(train.select(FEATURES).to_numpy())
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Predictions -> positions: the lag chain
# --------------------------------------------------------------------------

class _Constant(DecisionTreeClassifier):
    """A stand-in model that replays a fixed prediction sequence."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=np.int8)

    def predict(self, x):        # noqa: D102 - test double
        return self._preds[: len(x)]


def _frame(preds) -> pl.DataFrame:
    n = len(preds)
    return pl.DataFrame({
        "ts": [datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)
               + timedelta(minutes=i) for i in range(n)],
        "utc_min": [600 + i for i in range(n)],
        "close": [100.0] * n,
        "ret_fwd": [0.001] * n,
        "label": [1] * n,
        **{name: [0.0] * n for name in FEATURES},
    })


def test_signal_lag_one_shifts_the_prediction_forward_by_one_bar():
    preds = [1, 0, 0, 1]
    out = positions(_Constant(preds), _frame(preds), TreeConfig(signal_lag=1))
    # Bar 0 has no prior prediction, so it is flat; bar 1 acts on bar 0's.
    assert out["position"].to_list() == [0.0, 1.0, 0.0, 0.0]


def test_signal_lag_zero_acts_on_the_bar_the_model_predicted():
    preds = [1, 0, 0, 1]
    out = positions(_Constant(preds), _frame(preds), TreeConfig(signal_lag=0))
    assert out["position"].to_list() == [1.0, 0.0, 0.0, 1.0]


def test_long_short_maps_a_zero_to_a_short():
    preds = [1, 0, 1]
    out = positions(_Constant(preds), _frame(preds),
                    TreeConfig(signal_lag=0, mapping="long_short"))
    assert out["position"].to_list() == [1.0, -1.0, 1.0]


def test_a_long_short_flip_costs_two_units_of_turnover():
    preds = [1, 0, 1]
    out = positions(_Constant(preds), _frame(preds),
                    TreeConfig(signal_lag=0, mapping="long_short"))
    assert out["turnover"].to_list() == [1.0, 2.0, 2.0]


def test_holding_a_position_costs_no_turnover():
    preds = [1, 1, 1]
    out = positions(_Constant(preds), _frame(preds), TreeConfig(signal_lag=0))
    assert out["turnover"].to_list() == [1.0, 0.0, 0.0]


def test_a_session_window_forces_flat_outside_it():
    preds = [1] * 5
    frame = _frame(preds)          # utc_min runs 600, 601, 602, 603, 604
    cfg = TreeConfig(signal_lag=0, session_minutes=(601, 603))
    out = positions(_Constant(preds), frame, cfg)
    assert out["position"].to_list() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_the_prediction_column_is_kept_unlagged_for_accuracy():
    """Accuracy must measure the model against the bar it was trained on."""
    preds = [1, 0, 0, 1]
    out = positions(_Constant(preds), _frame(preds), TreeConfig(signal_lag=1))
    assert out["pred"].to_list() == preds


# --------------------------------------------------------------------------
# Returns, costs and aggregation
# --------------------------------------------------------------------------

def _priced(positions_, ret_fwd, cost_bps) -> pl.DataFrame:
    n = len(positions_)
    return pl.DataFrame({
        "ts": [datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)
               + timedelta(minutes=i) for i in range(n)],
        "position": list(map(float, positions_)),
        "ret_fwd": list(map(float, ret_fwd)),
        "turnover": [1.0] * n,
        "gross_bps": [1e4 * p * r for p, r in zip(positions_, ret_fwd)],
        "cost_bps": list(map(float, cost_bps)),
    }).with_columns(net_bps=pl.col("gross_bps") - pl.col("cost_bps"))


def test_daily_sums_minutes_into_calendar_days():
    priced = _priced([1, 1, 1], [0.001, 0.002, -0.001], [1.0, 1.0, 1.0])
    day = daily(priced)
    assert day.height == 1
    assert day["gross_bps"][0] == pytest.approx(20.0)
    assert day["net_bps"][0] == pytest.approx(17.0)


def test_daily_exposure_is_the_mean_absolute_position():
    priced = _priced([1, 0, 1, 0], [0.0] * 4, [0.0] * 4)
    assert daily(priced)["exposure"][0] == pytest.approx(0.5)


def test_net_is_gross_minus_cost_at_every_bar():
    priced = _priced([1, -1, 1], [0.001, 0.002, -0.001], [0.5, 0.7, 0.9])
    assert np.allclose(
        priced["net_bps"].to_numpy(),
        priced["gross_bps"].to_numpy() - priced["cost_bps"].to_numpy(),
    )


def test_breakeven_is_the_round_turn_cost_that_consumes_the_gross_edge():
    """Ten bps of gross over four units of turnover breaks even at 5 bps."""
    priced = pl.DataFrame({
        "gross_bps": [4.0, 3.0, 3.0], "turnover": [2.0, 1.0, 1.0],
    })
    assert breakeven_cost_bps(priced) == pytest.approx(5.0)


def test_breakeven_is_undefined_without_turnover():
    priced = pl.DataFrame({"gross_bps": [1.0], "turnover": [0.0]})
    assert np.isnan(breakeven_cost_bps(priced))


def test_charging_exactly_the_breakeven_cost_leaves_nothing():
    priced = pl.DataFrame({"gross_bps": [4.0, 3.0, 3.0], "turnover": [2.0, 1.0, 1.0]})
    be = breakeven_cost_bps(priced)
    net = priced["gross_bps"].sum() - (priced["turnover"] * be / 2.0).sum()
    assert net == pytest.approx(0.0)


def test_the_benchmark_holds_one_unit_and_pays_a_single_round_turn():
    frame = _frame([1, 1, 1, 1])
    bench = benchmark(frame)
    assert bench["position"].to_list() == [1.0] * 4
    assert bench["turnover"].sum() == pytest.approx(1.0)


def test_paper_style_sharpe_uses_their_annualisation():
    priced = pl.DataFrame({"net_bps": [10.0, -5.0, 8.0, 2.0, -1.0]})
    r = priced["net_bps"].to_numpy() / 1e4
    expected = r.mean() / r.std(ddof=1) * np.sqrt(PAPER_PERIODS_PER_YEAR)
    assert paper_style_sharpe(priced) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Accuracy against the base rate
# --------------------------------------------------------------------------

def test_accuracy_compares_against_the_majority_class_not_against_half():
    """A model that always predicts up scores the up-share, and no edge."""
    frame = pl.DataFrame({"label": [1, 1, 1, 0], "pred": [1, 1, 1, 1]})
    got = accuracy(frame)
    assert got["accuracy"] == pytest.approx(0.75)
    assert got["base_rate"] == pytest.approx(0.75)
    assert got["edge_vs_base"] == pytest.approx(0.0)


def test_accuracy_reports_a_real_edge_as_positive():
    frame = pl.DataFrame({"label": [1, 1, 0, 0], "pred": [1, 1, 0, 0]})
    got = accuracy(frame)
    assert got["accuracy"] == pytest.approx(1.0)
    assert got["edge_vs_base"] == pytest.approx(0.5)


def test_accuracy_is_empty_without_a_prediction_column():
    assert accuracy(pl.DataFrame({"label": [1, 0]}))["n"] == 0
