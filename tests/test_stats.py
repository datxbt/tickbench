"""The inference layer.

Three kinds of assertion here, in descending order of how much they matter.

**Identities**, because they are the only checks that cannot be satisfied by a
plausible-looking wrong answer: chi-square survival against published critical
values, HAC standard errors collapsing to the textbook ones at zero lags, Lo's
Sharpe standard error collapsing to the iid formula on iid data, and OLS
recovering a coefficient it was handed.

**Directions**, because the whole reason the module exists is that the naive
statistic is wrong in a *known* direction: a positively autocorrelated series
must widen its HAC error relative to the iid one, or the module is not doing
the job it was written for.

**Structure**, for the bootstrap - a resampler is easy to write in a way that
looks random and quietly breaks the pairing or the block length.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    chi2_sf,
    ljung_box,
    newey_west,
    newey_west_lags,
    normal_sf,
    ols_hac,
    paired_bootstrap,
    sharpe,
    sharpe_se,
    sharpe_with_se,
    stationary_bootstrap_indices,
    two_proportion_z,
)


def ar1(n: int, rho: float, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """An AR(1) series, for the tests that need known serial dependence."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, scale, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# --------------------------------------------------------------------------
# Distribution functions
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stat, dof",
    [(3.8415, 1), (5.9915, 2), (7.8147, 3), (18.3070, 10), (31.4104, 20)],
)
def test_chi2_sf_matches_published_five_percent_points(stat, dof):
    assert chi2_sf(stat, dof) == pytest.approx(0.05, abs=1e-4)


def test_chi2_sf_is_monotone_and_bounded():
    values = [chi2_sf(x, 5) for x in (0.5, 2.0, 5.0, 11.07, 30.0)]
    assert values == sorted(values, reverse=True)
    assert chi2_sf(0.0, 5) == 1.0
    assert 0.0 <= chi2_sf(1e6, 5) < 1e-12


def test_normal_sf_at_landmarks():
    assert normal_sf(0.0) == pytest.approx(0.5)
    assert normal_sf(1.959964) == pytest.approx(0.025, abs=1e-6)


# --------------------------------------------------------------------------
# Newey-West
# --------------------------------------------------------------------------

def test_zero_lag_hac_is_the_textbook_standard_error():
    """At zero lags the sandwich collapses to the plain standard error.

    This is the anchor for everything else in the module: if it does not hold,
    the Bartlett weighting is compensating for a scaling bug.
    """
    x = ar1(400, 0.0, seed=1)
    got = newey_west(x, lags=0)
    plain = x.std(ddof=0) / math.sqrt(x.size)
    assert got.se == pytest.approx(plain, rel=1e-12)
    assert got.mean == pytest.approx(float(x.mean()))


def test_positive_autocorrelation_widens_the_interval():
    """The direction the whole module exists for."""
    x = ar1(2000, 0.6, seed=2)
    naive = newey_west(x, lags=0)
    hac = newey_west(x)
    assert hac.se > naive.se * 1.4
    assert abs(hac.t_stat) < abs(naive.t_stat)


def test_negative_autocorrelation_narrows_it():
    x = ar1(2000, -0.5, seed=3)
    assert newey_west(x).se < newey_west(x, lags=0).se


def test_automatic_lag_count_follows_newey_west_1994():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1000) == 6
    assert newey_west_lags(1) == 0
    assert newey_west_lags(10_000) == 11


def test_newey_west_survives_a_degenerate_series():
    assert math.isnan(newey_west([]).mean)
    constant = newey_west(np.zeros(50))
    assert constant.mean == 0.0 and math.isnan(constant.t_stat)


def test_non_finite_values_are_dropped_not_propagated():
    x = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
    assert newey_west(x, lags=0).n == 4


# --------------------------------------------------------------------------
# OLS with HAC errors
# --------------------------------------------------------------------------

def test_ols_recovers_the_coefficients_it_was_given():
    rng = np.random.default_rng(4)
    n = 4000
    X = rng.normal(size=(n, 2))
    y = 1.5 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)
    fit = ols_hac(y, X, names=["a", "b"])
    assert fit.names == ("const", "a", "b")
    assert fit.params[0] == pytest.approx(1.5, abs=0.1)
    assert fit.coef("a")[0] == pytest.approx(2.0, abs=0.1)
    assert fit.coef("b")[0] == pytest.approx(-0.5, abs=0.1)
    assert 0.7 < fit.r2 < 0.95


def test_ols_keep_mask_lines_residuals_back_up():
    """The mask is what lets a two-stage regression align its own stages.

    Without it the second stage has to guess which rows the first stage kept,
    and a guess of "the last n" is wrong the moment a gap is interior.
    """
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    X = np.arange(6, dtype=float)
    fit = ols_hac(y, X, names=["x"])
    assert fit.n == 5
    assert fit.keep.tolist() == [True, True, False, True, True, True]
    assert fit.resid.size == fit.n


def test_ols_refuses_an_underdetermined_design():
    with pytest.raises(ValueError, match="usable observations"):
        ols_hac(np.arange(2.0), np.arange(2.0).reshape(-1, 1), names=["x"])


def test_ols_rejects_a_name_count_mismatch():
    with pytest.raises(ValueError, match="names for"):
        ols_hac(np.arange(10.0), np.zeros((10, 2)), names=["only_one"])


# --------------------------------------------------------------------------
# Sharpe
# --------------------------------------------------------------------------

def test_sharpe_is_invariant_to_scale():
    """The identity the sizing study leans on: leverage cannot move a Sharpe."""
    x = ar1(500, 0.1, seed=5) + 0.05
    assert sharpe(x) == pytest.approx(sharpe(x * 7.3), rel=1e-12)


def test_sharpe_annualises_by_the_period_count():
    x = ar1(500, 0.0, seed=6) + 0.05
    assert sharpe(x, 252) == pytest.approx(
        sharpe(x, 1) * math.sqrt(252), rel=1e-12)


def test_lo_standard_error_matches_the_iid_formula_on_iid_data():
    """Lo (2002) reduces to sqrt((1 + SR^2/2)/n) when the returns are iid."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.001, 0.01, 5000)
    per_period = sharpe(x, 1)
    iid = math.sqrt((1.0 + 0.5 * per_period**2) / x.size) * math.sqrt(252)
    assert sharpe_se(x, 252) == pytest.approx(iid, rel=0.10)


def test_lo_standard_error_grows_with_autocorrelation():
    clean = ar1(3000, 0.0, seed=8) + 0.05
    sticky = ar1(3000, 0.6, seed=8) + 0.05
    assert sharpe_se(sticky, 252) > sharpe_se(clean, 252)


def test_sharpe_with_se_reports_a_consistent_triple():
    x = ar1(1000, 0.2, seed=9) + 0.02
    got = sharpe_with_se(x)
    assert got.mean == pytest.approx(sharpe(x))
    assert got.t_stat == pytest.approx(got.mean / got.se, rel=1e-12)


# --------------------------------------------------------------------------
# The stationary bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_indices_have_the_right_shape_and_range():
    idx = stationary_bootstrap_indices(200, block_len=10, n_boot=50,
                                       rng=np.random.default_rng(10))
    assert idx.shape == (50, 200)
    assert idx.min() >= 0 and idx.max() < 200


def test_short_blocks_break_more_often_than_long_ones():
    """The block length has to actually control the block length."""
    def break_rate(b: int) -> float:
        idx = stationary_bootstrap_indices(500, block_len=b, n_boot=40,
                                           rng=np.random.default_rng(11))
        contiguous = (idx[:, 1:] == (idx[:, :-1] + 1) % 500)
        return 1.0 - float(contiguous.mean())

    assert break_rate(2) > break_rate(20) * 3


def test_blocks_wrap_rather_than_truncate():
    """A wrap is what makes the resampled series stationary."""
    idx = stationary_bootstrap_indices(10, block_len=1000, n_boot=200,
                                       rng=np.random.default_rng(12))
    steps = (idx[:, 1:] - idx[:, :-1]) % 10
    assert (steps == 1).mean() > 0.95     # essentially one long block
    assert ((idx[:, :-1] == 9) & (idx[:, 1:] == 0)).any()


def test_bootstrap_interval_brackets_a_known_mean():
    rng = np.random.default_rng(13)
    x = rng.normal(1.0, 1.0, 800)
    got = bootstrap_ci(x, n_boot=400, rng=rng)
    assert got.lo < 1.0 < got.hi
    assert got.p_value < 0.01


def test_bootstrap_interval_contains_zero_for_pure_noise():
    rng = np.random.default_rng(14)
    got = bootstrap_ci(rng.normal(0.0, 1.0, 800), n_boot=400, rng=rng)
    assert got.lo < 0.0 < got.hi
    assert got.p_value > 0.05


def test_bootstrap_handles_a_statistic_that_is_not_the_mean():
    rng = np.random.default_rng(15)
    x = rng.normal(0.05, 1.0, 600)
    got = bootstrap_ci(x, lambda a: sharpe(a, 252), n_boot=300, rng=rng)
    assert got.lo < got.point < got.hi


def test_paired_bootstrap_uses_the_same_dates_on_both_sides():
    """Two series that differ by a constant have no sampling variation at all.

    Under independent resampling they would; under pairing the difference is
    exactly the constant in every draw. This is the assertion that would catch
    a pairing bug, and nothing else in the module would.
    """
    rng = np.random.default_rng(16)
    a = rng.normal(0.0, 1.0, 400)
    b = a - 0.25
    got = paired_bootstrap(a, b, n_boot=300, rng=rng)
    assert got.point == pytest.approx(0.25, rel=1e-9)
    assert got.lo == pytest.approx(0.25, abs=1e-9)
    assert got.hi == pytest.approx(0.25, abs=1e-9)


def test_paired_bootstrap_refuses_misaligned_series():
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap(np.zeros(10), np.zeros(11))


def test_paired_bootstrap_finds_a_real_difference():
    rng = np.random.default_rng(17)
    a = rng.normal(0.10, 1.0, 900)
    b = rng.normal(-0.10, 1.0, 900)
    got = paired_bootstrap(a, b, n_boot=500, rng=rng)
    assert got.point > 0
    assert got.p_value < 0.05


# --------------------------------------------------------------------------
# Proportions, serial dependence, multiplicity
# --------------------------------------------------------------------------

def test_two_proportion_z_on_identical_rates_is_zero():
    got = two_proportion_z(50, 100, 100, 200)
    assert got.diff == pytest.approx(0.0)
    assert got.z == pytest.approx(0.0)
    assert got.p_value == pytest.approx(1.0)


def test_two_proportion_z_scales_with_sample_size():
    small = two_proportion_z(60, 100, 40, 100)
    large = two_proportion_z(600, 1000, 400, 1000)
    assert small.diff == pytest.approx(large.diff)
    assert large.z > small.z * 2.5
    assert large.p_value < 1e-10


def test_ljung_box_finds_autocorrelation_and_clears_noise():
    assert ljung_box(ar1(1500, 0.5, seed=18), 10).p_value < 0.01
    assert ljung_box(ar1(1500, 0.0, seed=19), 10).p_value > 0.05


def test_robust_ljung_box_is_calmer_under_volatility_clustering():
    """A GARCH-like series is serially *uncorrelated* but not independent.

    The classical statistic rejects on the variance clustering alone, which is
    not the question. The robust correction is there to not do that.
    """
    rng = np.random.default_rng(20)
    n = 3000
    vol = np.ones(n)
    for i in range(1, n):
        vol[i] = math.sqrt(0.05 + 0.90 * vol[i - 1] ** 2 + 0.05 * rng.normal() ** 2)
    x = rng.normal(size=n) * vol
    assert ljung_box(x, 10, robust=True).stat < ljung_box(x, 10, robust=False).stat


def test_bonferroni_scales_by_the_family_size():
    assert bonferroni([0.01, 0.02]).tolist() == pytest.approx([0.02, 0.04])
    assert bonferroni([0.6, 0.7]).tolist() == pytest.approx([1.0, 1.0])


def test_benjamini_hochberg_is_monotone_and_in_input_order():
    raw = [0.20, 0.01, 0.03, 0.04]
    got = benjamini_hochberg(raw)
    assert got.size == 4
    assert got[1] <= got[2] <= got[3] <= got[0]
    # Never smaller than the raw p-value, never above one.
    assert np.all(got >= np.array(raw) - 1e-12)
    assert np.all(got <= 1.0)


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    raw = [0.001, 0.01, 0.02, 0.03, 0.04]
    assert np.all(benjamini_hochberg(raw) <= bonferroni(raw) + 1e-12)


def test_empty_families_are_handled():
    assert benjamini_hochberg([]).size == 0
    assert bonferroni([]).size == 0
