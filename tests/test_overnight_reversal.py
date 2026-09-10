"""The overnight-intraday reversal family.

The assertions worth having here are about **timing** and **sign**, because
those are the two ways a cross-sectional reversal backtest goes wrong without
looking wrong.

Timing: the signal is yesterday's, the return is today's, and the two are
shifted apart in exactly one place. A test that only checks the arithmetic of
the weights would pass happily while the strategy sorted on the very return it
was about to earn - which is the single most flattering bug available to a
reversal study, and it does not announce itself in any summary statistic.

Sign: the construction is contrarian by definition. The lowest signal must be
*bought*. Getting that backwards produces a momentum strategy that is reported
under a reversal strategy's name, and on a short sample it can even look good.

The cost asymmetry gets its own test because it is the finding: a variant that
is flat overnight pays a round turn a day, and one that holds through pays only
for the change in weight. A harness that charged both the same way would rank
them wrongly and never say so.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from qlab.strategies.overnight_reversal import (
    CORE_VARIANTS,
    RTH,
    VARIANTS,
    CrossSection,
    SessionWindow,
    abnormal_reversal_signal,
    build_weights,
    conditional_split,
    cross_section,
    overnight_dispersion,
    run_variant,
    trailing_vol,
    weekly_cross_section,
    with_returns,
)

SYMBOLS = ("A", "B", "C", "D")


def synthetic_cs(n_days: int = 300, seed: int = 0,
                 reversal: float = 0.0) -> CrossSection:
    """A panel with a *known* amount of overnight-to-intraday reversal.

    ``reversal`` is the fraction of each overnight move that the following
    intraday session gives back. At zero the panel is pure noise and every
    strategy in the module must find nothing; above zero CO-OC must find it and
    the other three must not.
    """
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    co = rng.normal(0.0, 0.004, (n_days, n))
    oc = rng.normal(0.0, 0.004, (n_days, n))
    # Today's intraday partly reverses yesterday's overnight.
    oc[1:] -= reversal * co[:-1]
    price = 100.0 * np.cumprod(1.0 + co + oc, axis=0)
    returns = {
        "r_co": co,
        "r_oc": oc,
        "r_cc": (1.0 + co) * (1.0 + oc) - 1.0,
        "r_oo": np.vstack([np.full((1, n), np.nan),
                           ((1.0 + oc[:-1]) * (1.0 + co[1:]) - 1.0)]),
    }
    dates = np.array([date.fromordinal(738000 + i) for i in range(n_days)])
    return CrossSection(dates, SYMBOLS, returns, price,
                        np.array([13] * n))


# --------------------------------------------------------------------------
# Session conventions
# --------------------------------------------------------------------------

def test_every_instrument_has_a_session_and_a_contract():
    for symbol, window in RTH.items():
        assert window.symbol == symbol
        assert window.open_min < window.close_min
        assert window.contract          # the session is transcribed, not invented


def test_a_backwards_window_is_rejected():
    with pytest.raises(ValueError, match="bad window"):
        SessionWindow("X", 900, 500, 100, "ZZ")


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def test_return_definitions_compose():
    """``(1+r_co)(1+r_oc)`` has to be ``1+r_cc``, or the decomposition is a fiction."""
    panel = pl.DataFrame({
        "nyd": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "symbol": ["A"] * 3,
        "o": [100.0, 103.0, 101.0],
        "c": [102.0, 101.0, 105.0],
        "open_hour_utc": [13] * 3,
    })
    got = with_returns(panel).sort("nyd")
    cc = got["r_cc"].to_numpy()[1:]
    co = got["r_co"].to_numpy()[1:]
    oc = got["r_oc"].to_numpy()[1:]
    assert np.allclose((1 + co) * (1 + oc) - 1, cc)


def test_shifts_never_cross_instruments():
    """A holiday in one market must not line another's price up against it."""
    panel = pl.DataFrame({
        "nyd": [date(2024, 1, 2), date(2024, 1, 3)] * 2,
        "symbol": ["A", "A", "B", "B"],
        "o": [100.0, 110.0, 50.0, 55.0],
        "c": [105.0, 115.0, 52.0, 57.0],
        "open_hour_utc": [13] * 4,
    })
    got = with_returns(panel).sort(["symbol", "nyd"])
    # The first row of each instrument has no predecessor of its own.
    assert got.filter(pl.col("symbol") == "A")["r_cc"][0] is None
    assert got.filter(pl.col("symbol") == "B")["r_cc"][0] is None
    assert got.filter(pl.col("symbol") == "B")["r_cc"][1] == pytest.approx(
        57.0 / 52.0 - 1)


def test_incomplete_days_are_dropped_not_filled():
    """A shut market must leave, not arrive as a zero return.

    A carried-forward price is a zero return, and demeaning turns a zero return
    into a *positive* contrarian weight - so filling would put money on an
    instrument that was not quoting.
    """
    rows = []
    for day in (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)):
        for symbol in ("A", "B"):
            if day == date(2024, 1, 3) and symbol == "B":
                continue                      # B is shut
            rows.append({"nyd": day, "symbol": symbol, "o": 100.0, "c": 101.0,
                         "open_hour_utc": 13})
    cs = cross_section(with_returns(pl.DataFrame(rows)), symbols=("A", "B"))
    assert date(2024, 1, 3) not in set(cs.dates.tolist())


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

@pytest.mark.parametrize("weighting", ["demean", "rank"])
def test_weights_are_zero_investment(weighting):
    signal = np.random.default_rng(1).normal(size=(50, 4))
    w = build_weights(signal, weighting=weighting)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("weighting", ["demean", "rank", "vol_scaled"])
def test_the_lowest_signal_is_bought(weighting):
    """Contrarian by construction. Reversing this would be a momentum study."""
    signal = np.array([[-0.03, -0.01, 0.01, 0.03]])
    vol = np.ones_like(signal) * 0.01
    w = build_weights(signal, weighting=weighting, vol=vol)
    assert w[0, 0] > 0 and w[0, -1] < 0
    assert np.all(np.diff(w[0]) < 0)


def test_rank_weighting_ignores_magnitude():
    """The point of rank weighting on a mixed cross-section."""
    a = build_weights(np.array([[-0.01, 0.0, 0.005, 0.02]]), weighting="rank")
    b = build_weights(np.array([[-9.90, 0.0, 0.001, 0.02]]), weighting="rank")
    assert np.allclose(a, b)


def test_demean_weighting_does_not():
    a = build_weights(np.array([[-0.01, 0.0, 0.005, 0.02]]), weighting="demean")
    b = build_weights(np.array([[-9.90, 0.0, 0.001, 0.02]]), weighting="demean")
    assert not np.allclose(a, b)


def test_vol_scaling_equalises_two_assets_that_differ_only_in_scale():
    signal = np.array([[0.02, 0.002, -0.02, -0.002]])
    vol = np.array([[0.01, 0.001, 0.01, 0.001]])
    plain = build_weights(signal, weighting="demean")
    scaled = build_weights(signal, weighting="vol_scaled", vol=vol)
    # Unscaled, the loud pair dominates; scaled, the pairs carry equal weight.
    assert abs(plain[0, 0]) > 5 * abs(plain[0, 1])
    assert abs(scaled[0, 0]) == pytest.approx(abs(scaled[0, 1]), rel=1e-9)


def test_vol_scaled_needs_a_volatility():
    with pytest.raises(ValueError, match="volatility"):
        build_weights(np.zeros((5, 4)), weighting="vol_scaled")


def test_unknown_weighting_is_refused():
    with pytest.raises(ValueError, match="unknown weighting"):
        build_weights(np.zeros((5, 4)), weighting="momentum")


def test_trailing_vol_is_strictly_backward_looking():
    """Row t must use rows [t-w, t) - never row t itself."""
    r = np.zeros((30, 1))
    r[20, 0] = 1.0                       # one enormous day
    vol = trailing_vol(r, window=5)
    assert not np.isfinite(vol[20, 0]) or vol[20, 0] == pytest.approx(0.0)
    assert vol[21, 0] > 0                # visible only from the next row on


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def test_the_signal_is_lagged_by_exactly_one_day():
    """The assertion the whole study rests on.

    A panel with no reversal at all still has a *contemporaneous* one: today's
    demeaned intraday return is trivially predicted by itself. If the shift were
    missing, OC-OC on a pure-noise panel would report an enormous negative mean,
    because it would be shorting the return it is about to receive.
    """
    cs = synthetic_cs(400, seed=2, reversal=0.0)
    run = run_variant(cs, "OC-OC", weighting="demean")
    hac = run.hac(net=False)
    assert abs(hac.t_stat) < 3.0
    assert run.dates.size == cs.n_days - 1
    assert run.dates[0] == cs.dates[1]


def test_a_planted_reversal_is_found_by_co_oc_and_not_by_the_others():
    """The positive control: the module must be able to see a real effect."""
    cs = synthetic_cs(1500, seed=3, reversal=0.6)
    co_oc = run_variant(cs, "CO-OC", weighting="demean").hac(net=False)
    oo_oo = run_variant(cs, "OO-OO", weighting="demean").hac(net=False)
    assert co_oc.mean > 0 and co_oc.t_stat > 5.0
    assert abs(oo_oo.t_stat) < abs(co_oc.t_stat)


def test_a_noise_panel_produces_nothing_anywhere():
    cs = synthetic_cs(1500, seed=4, reversal=0.0)
    for name in CORE_VARIANTS:
        assert abs(run_variant(cs, name, weighting="rank").hac(net=False).t_stat) < 3.0


def test_the_shuffle_placebo_destroys_a_planted_effect():
    cs = synthetic_cs(1500, seed=5, reversal=0.6)
    real = run_variant(cs, "CO-OC", weighting="demean").hac(net=False)
    placebo = run_variant(cs, "CO-OC", weighting="demean",
                          shuffle_seed=11).hac(net=False)
    assert real.mean > 0
    assert abs(placebo.mean) < 0.3 * real.mean


def test_the_placebo_leaves_turnover_and_cost_alone():
    """A placebo that changes the cost is not comparing what it claims to."""
    cs = synthetic_cs(400, seed=6)
    cost = np.full_like(cs.price_open, 1.0)
    real = run_variant(cs, "CO-OC", weighting="rank", cost_bps=cost)
    fake = run_variant(cs, "CO-OC", weighting="rank", cost_bps=cost,
                       shuffle_seed=1)
    assert np.allclose(np.nansum(real.turnover), np.nansum(fake.turnover))
    assert np.allclose(np.nansum(real.cost_bps), np.nansum(fake.cost_bps))


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def test_an_intraday_variant_pays_more_than_a_continuous_one():
    """The asymmetry that reorders the paper's own table.

    CO-OC is flat overnight, so it pays a full round turn every day; CC-CC
    holds through, so it pays only for the change in weight.
    """
    cs = synthetic_cs(500, seed=7)
    cost = np.full_like(cs.price_open, 2.0)
    intraday = run_variant(cs, "CO-OC", weighting="rank", cost_bps=cost)
    holding = run_variant(cs, "CC-CC", weighting="rank", cost_bps=cost)
    assert np.nanmean(intraday.cost_bps) > np.nanmean(holding.cost_bps)
    assert VARIANTS["CO-OC"].exposure == "intraday"
    assert VARIANTS["CC-CC"].exposure == "continuous"


def test_zero_cost_leaves_net_equal_to_gross():
    cs = synthetic_cs(300, seed=8)
    run = run_variant(cs, "CO-OC", weighting="rank", cost_bps=None,
                      winsorize=False)
    assert np.allclose(run.net_bps, run.gross_bps, equal_nan=True)


def test_cost_only_ever_subtracts():
    cs = synthetic_cs(300, seed=9)
    cost = np.full_like(cs.price_open, 3.0)
    run = run_variant(cs, "OC-OC", weighting="rank", cost_bps=cost,
                      winsorize=False)
    assert np.all(run.cost_bps >= -1e-12)
    assert np.all(run.net_bps <= run.gross_bps + 1e-12)


# --------------------------------------------------------------------------
# The rest of the surface
# --------------------------------------------------------------------------

def test_dispersion_is_positive_and_responds_to_a_spread_out_day():
    cs = synthetic_cs(300, seed=10)
    disp = overnight_dispersion(cs, scaled=False)
    assert np.nanmin(disp) >= 0.0
    loud = cs.returns["r_co"].copy()
    loud[100] = np.array([0.05, -0.05, 0.05, -0.05])
    louder = CrossSection(cs.dates, cs.symbols, {**cs.returns, "r_co": loud},
                          cs.price_open, cs.open_hour_utc)
    assert overnight_dispersion(louder, scaled=False)[100] > disp[100]


def test_conditional_split_lags_its_conditioner():
    """A split on today's value of anything is not a tradable rule."""
    cs = synthetic_cs(400, seed=11)
    run = run_variant(cs, "CO-OC", weighting="rank")
    # Conditioning on the strategy's own contemporaneous return would give a
    # gigantic difference; conditioning on its lag must not.
    padded = np.concatenate([[np.nan], run.net_bps])
    out = conditional_split(run, padded, label="self")
    assert out["high_n"] + out["low_n"] <= run.net_bps.size
    assert abs(out["diff_t"]) < 10.0


def test_weekly_aggregation_shrinks_the_sample_to_weeks():
    cs = synthetic_cs(300, seed=12)
    weekly = weekly_cross_section(cs)
    assert 0 < weekly.n_days <= cs.n_days / 4
    assert weekly.symbols == cs.symbols
    assert set(weekly.returns) >= {"r_co", "r_oc", "r_cc", "r_oo"}


def test_abnormal_reversal_signal_warms_up_before_it_reports():
    """AB_NR needs an interval and twelve of them; before that it is undefined."""
    cs = synthetic_cs(900, seed=13)
    signal = abnormal_reversal_signal(cs, interval=20)
    assert not np.isfinite(signal[:20 + 12 * 20 - 1]).any()
    assert np.isfinite(signal[-1]).all()
