"""Tests for the OHLC-only microstructure estimators.

Corwin-Schultz is transcribed from equations, and a transcription error in it
would not raise - it would return plausible small numbers forever. So the
estimator is pinned three ways: against a tape whose spread is known by
construction, against its own published algebra, and against the degenerate
inputs (a flat bar, a single bar) that occur constantly in real data.

The one test here that is not about correctness is
:func:`test_a_constant_spread_is_not_recovered_bar_by_bar`. It pins a property
that matters for reading Eross et al. (2017): given a spread that never moves,
the per-bar estimates still scatter by more than half their own level. The
estimator is a level estimator, and a per-bar series of it is mostly noise.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from qlab.microstructure import (
    corwin_schultz,
    estimator_error,
    realised_variance,
    with_corwin_schultz,
)


def _tape(n: int, *, spread: float, sigma: float, seed: int = 0, price: float = 100.0):
    """A random walk observed through a fixed proportional spread.

    Each bar's high is the true high plus half a spread and its low the true low
    minus half a spread, which is the data-generating process Corwin and Schultz
    assume: highs are buyer-initiated and print at the ask, lows are seller-
    initiated and print at the bid.
    """
    rng = np.random.default_rng(seed)
    mid = price * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    # Two sub-periods per bar, so the bar has an interior range of its own.
    wiggle = np.abs(rng.normal(0.0, sigma, n)) * mid
    high = (mid + wiggle) * (1.0 + spread / 2.0)
    low = (mid - wiggle) * (1.0 - spread / 2.0)
    return high, low, mid


def test_recovers_a_known_spread_on_its_own_generating_process():
    """Given the process the estimator assumes, it should land near the truth."""
    high, low, _ = _tape(200_000, spread=0.0020, sigma=0.0004, seed=3)
    est = corwin_schultz(high, low)
    got = float(np.nanmean(est))
    assert got == pytest.approx(0.0020, rel=0.25)


def test_a_wider_spread_produces_a_wider_estimate():
    out = []
    for s in (0.0005, 0.0020, 0.0080):
        high, low, _ = _tape(50_000, spread=s, sigma=0.0004, seed=5)
        out.append(float(np.nanmean(corwin_schultz(high, low))))
    assert out[0] < out[1] < out[2]


def test_matches_the_published_algebra_on_one_pair():
    """Two bars, worked through equations (3)-(6) by hand."""
    h = np.array([101.0, 102.0])
    l = np.array([99.0, 100.0])
    den = 3.0 - 2.0 * math.sqrt(2.0)
    beta = math.log(101 / 99) ** 2 + math.log(102 / 100) ** 2
    gamma = math.log(102 / 99) ** 2
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / den - math.sqrt(gamma / den)
    expected = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    got = corwin_schultz(h, l, clamp_negative=False)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(expected, rel=1e-12)


def test_first_element_is_nan_because_a_pair_needs_two_bars():
    got = corwin_schultz([100.0, 101.0, 102.0], [99.0, 100.0, 101.0])
    assert np.isnan(got[0])
    assert np.isfinite(got[1:]).all()


def test_a_single_bar_yields_nothing_rather_than_raising():
    assert np.isnan(corwin_schultz([100.0], [99.0])).all()
    assert corwin_schultz([], []).size == 0


def test_flat_bars_give_a_zero_spread_not_a_nan():
    """A bar with high == low is common on an illiquid 5-minute tape."""
    got = corwin_schultz([100.0] * 5, [100.0] * 5)
    assert np.nan_to_num(got[1:]).max() == 0.0


def test_clamping_only_ever_raises_the_estimate():
    rng = np.random.default_rng(1)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 5000)))
    high, low = mid * 1.0005, mid * 0.9995
    clamped = corwin_schultz(high, low, clamp_negative=True)
    raw = corwin_schultz(high, low, clamp_negative=False)
    assert np.nanmean(clamped) >= np.nanmean(raw)
    assert (np.nan_to_num(clamped) >= 0).all()


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError, match="differ in shape"):
        corwin_schultz([1.0, 2.0], [1.0])


def test_over_restarts_the_pairing_at_each_group():
    """Pairing across a session break measures the gap, not a spread.

    Note which way the damage runs. A gap inflates gamma (the two-period range)
    without inflating beta (the sum of the single-period ranges), so alpha goes
    sharply *negative* and the pair produces a negative spread - which the clamp
    then reports as zero. Crossing a session boundary therefore biases the mean
    estimate **down**, which is the opposite of the intuition that a gap looks
    like a wide spread.
    """
    frame = pl.DataFrame(
        {
            "day": ["a", "a", "b", "b"],
            "high": [101.0, 101.0, 501.0, 501.0],
            "low": [99.0, 99.0, 499.0, 499.0],
        }
    )
    out = with_corwin_schultz(frame, over="day")
    # First bar of each group has no pair, so each group contributes one gap.
    assert int(np.isnan(out["cs_spread"].to_numpy()).sum()) == 2

    raw = with_corwin_schultz(frame, clamp_negative=False)["cs_spread"].to_numpy()
    assert raw[2] < -0.5, "the cross-session pair should read as a large negative"
    clamped = with_corwin_schultz(frame)["cs_spread"].to_numpy()
    assert clamped[2] == 0.0


def test_realised_variance_is_the_squared_log_return():
    close = np.array([100.0, 101.0, 99.0])
    got = realised_variance(close)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(math.log(101 / 100) ** 2)
    assert got[2] == pytest.approx(math.log(99 / 101) ** 2)


def test_estimator_error_reports_bias_and_correlation():
    frame = pl.DataFrame(
        {
            "cs_spread": [0.0002, 0.0004, 0.0006],
            "spread_close": [0.01, 0.02, 0.03],
            "close": [100.0, 100.0, 100.0],
        }
    )
    e = estimator_error(frame)
    assert e["mean_est_bps"] == pytest.approx(4.0)
    assert e["mean_true_bps"] == pytest.approx(2.0)
    assert e["bias_bps"] == pytest.approx(2.0)
    assert e["corr"] == pytest.approx(1.0)


def test_estimator_error_survives_an_empty_overlap():
    frame = pl.DataFrame(
        {"cs_spread": [None, None], "spread_close": [0.01, 0.02], "close": [1.0, 1.0]},
        schema={"cs_spread": pl.Float64, "spread_close": pl.Float64, "close": pl.Float64},
    )
    assert estimator_error(frame)["n"] == 0


def test_a_constant_spread_is_not_recovered_bar_by_bar():
    """The level is roughly right; the individual estimates are not.

    The tape below holds the spread *exactly* constant and varies only the
    range, so every bit of variation in the estimate is error. The estimator
    still produces a series whose dispersion exceeds its own mean - which is
    what makes it a level estimator rather than a time-series one, and is the
    reason a per-bar Corwin-Schultz series should not be fed into a lead-lag or
    Granger test without saying so.
    """
    rng = np.random.default_rng(9)
    n = 40_000
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    wiggle = np.abs(rng.normal(0, 0.0004, n) * rng.lognormal(0, 1.0, n)) * mid
    spread = 0.0010
    high = (mid + wiggle) * (1 + spread / 2)
    low = (mid - wiggle) * (1 - spread / 2)

    est = corwin_schultz(high, low)
    assert float(np.nanmean(est)) == pytest.approx(spread, rel=0.6)
    # Coefficient of variation of an estimate whose target never moves.
    assert np.nanstd(est) / np.nanmean(est) > 0.5
