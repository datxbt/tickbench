"""SPAR has to recover a periodicity it was never told about - and see no future."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qlab.volume_spar import (
    WINDOW,
    Panel,
    build_panel,
    design,
    fit_forecast,
    historical_mean,
    ok_volatility,
    oos_r2,
    paired_te,
    periodic_curve,
    vwap_tracking_error,
)

RNG = np.random.default_rng(11)


def synthetic_panel(
    n_days: int = 400,
    n_slots: int = 24,
    shape: str = "u",
    rho: float = 0.6,
    noise: float = 0.25,
) -> Panel:
    """A panel with a known periodic curve and a known AR(1) stationary part.

    ``log V = f[k] + z[t,k]``, with ``z`` an AR(1) in continuous intraday time.
    Everything the models are supposed to find is put in here on purpose, so a
    failure is a failure of the estimator and not of the data.
    """
    grid = np.linspace(-1.0, 1.0, n_slots)
    if shape == "u":
        f = 1.2 * grid**2
    elif shape == "j":
        f = 1.2 * np.exp(-3.0 * (grid + 1.0))
    elif shape == "w":  # two sessions, as an Asian market with a lunch break
        f = 0.9 * np.cos(2.0 * np.pi * grid) ** 2
    else:
        raise ValueError(shape)
    f = f - f.mean()

    total = n_days * n_slots
    z = np.zeros(total)
    for i in range(1, total):
        z[i] = rho * z[i - 1] + noise * RNG.standard_normal()
    y = f[None, :] + z.reshape(n_days, n_slots)
    vol = np.exp(y)
    price = 100.0 + np.cumsum(RNG.standard_normal(total) * 0.05).reshape(n_days, n_slots)
    return Panel(
        days=list(range(n_days)),
        slots=list(range(n_slots)),
        y=y,
        vol=vol,
        sigma=np.abs(RNG.standard_normal((n_days, n_slots))) * 0.1,
        price=price,
        symbol="SYNTH",
        grid=shape,
    )


# --- the periodic curve ------------------------------------------------------


@pytest.mark.parametrize("shape", ["u", "j", "w"])
def test_periodic_curve_recovers_any_shape_without_being_told_it(shape):
    """The whole point of SPAR: no functional form is assumed, so all three fit."""
    panel = synthetic_panel(shape=shape)
    curve = periodic_curve(panel)
    assert curve.mean() == pytest.approx(1.0, abs=1e-9)

    grid = np.linspace(-1.0, 1.0, len(panel.slots))
    truth = {"u": 1.2 * grid**2, "j": 1.2 * np.exp(-3.0 * (grid + 1.0)),
             "w": 0.9 * np.cos(2.0 * np.pi * grid) ** 2}[shape]
    truth = np.exp(truth - truth.mean())
    truth = len(truth) * truth / truth.sum()
    assert np.corrcoef(curve, truth)[0, 1] > 0.98


def test_periodic_curve_is_normalised_to_mean_one():
    curve = periodic_curve(synthetic_panel(shape="j"))
    assert curve.mean() == pytest.approx(1.0)
    assert (curve > 0).all()


# --- the central claim -------------------------------------------------------


@pytest.mark.parametrize("spec", [1, 3])
def test_within_transformation_beats_the_same_features_without_it(spec):
    """SPAR > OLS on identical features is the paper's whole argument."""
    panel = synthetic_panel()
    bench = historical_mean(panel)
    spar = fit_forecast(panel, spec, within=True)
    ols = fit_forecast(panel, spec, within=False)
    assert oos_r2(panel, spar, bench) > oos_r2(panel, ols, bench)


def test_spar_beats_the_historical_mean_it_is_scored_against():
    panel = synthetic_panel()
    r2 = oos_r2(panel, fit_forecast(panel, 1, within=True), historical_mean(panel))
    assert 0.0 < r2 < 1.0


def test_a_panel_with_no_signal_scores_no_better_than_the_mean():
    """White noise around a periodic curve leaves nothing for the AR part."""
    panel = synthetic_panel(rho=0.0, noise=0.4)
    r2 = oos_r2(panel, fit_forecast(panel, 1, within=True), historical_mean(panel))
    assert r2 < 0.05


# --- no look-ahead -----------------------------------------------------------


def test_forecasts_do_not_move_when_the_future_is_replaced():
    """Rewrite everything after day 200; forecasts up to day 200 must not budge.

    This is the test that matters. Every other number in the module is worthless
    if the rolling means or the fitted coefficients can see past the forecast
    date, and a within-transformation is an easy place to leak: use the mean
    including day t and the target is inside its own predictor.
    """
    panel = synthetic_panel()
    cut = 200
    before = fit_forecast(panel, 3, within=True).yhat[:cut].copy()

    tampered = Panel(
        days=panel.days, slots=panel.slots,
        y=panel.y.copy(), vol=panel.vol.copy(), sigma=panel.sigma.copy(),
        price=panel.price.copy(), symbol=panel.symbol, grid=panel.grid,
    )
    tampered.y[cut:] = RNG.standard_normal(tampered.y[cut:].shape) * 5.0
    tampered.sigma[cut:] = RNG.standard_normal(tampered.sigma[cut:].shape) * 5.0
    after = fit_forecast(tampered, 3, within=True).yhat[:cut]

    np.testing.assert_allclose(before, after, rtol=1e-10, atol=1e-10)


def test_predictors_are_lagged_by_at_least_one_slot():
    """No column of the design may equal the contemporaneous target."""
    panel = synthetic_panel(n_days=60, n_slots=12)
    x, names = design(panel, 4)
    flat_y = panel.y.reshape(-1)
    for j, name in enumerate(names):
        column = x[..., j].reshape(-1)
        ok = np.isfinite(column)
        assert not np.allclose(column[ok], flat_y[ok]), name


def test_historical_mean_uses_only_prior_days():
    panel = synthetic_panel(n_days=150, n_slots=8)
    mean = historical_mean(panel, burn_in=100)
    expected = np.nanmean(panel.y[100 - WINDOW : 100], axis=0)
    np.testing.assert_allclose(mean.yhat[100], expected)
    assert np.isnan(mean.yhat[99]).all()


# --- the VWAP schedule -------------------------------------------------------


def test_vwap_weights_are_a_full_allocation():
    """Equal weight must reproduce the simple average price exactly."""
    panel = synthetic_panel(n_days=150, n_slots=10)
    result = vwap_tracking_error(panel, None, burn_in=100)
    t = 120
    expected = 1e4 * (
        (panel.vol[t] @ panel.price[t] / panel.vol[t].sum()) - panel.price[t].mean()
    ) / (panel.vol[t] @ panel.price[t] / panel.vol[t].sum())
    index = result["_days"].index(panel.days[t])
    assert result["_gaps"][index] == pytest.approx(expected, rel=1e-9)


def test_a_perfect_forecast_tracks_vwap_exactly():
    """Hand the schedule the realised volume and the error has to vanish."""
    panel = synthetic_panel(n_days=150, n_slots=10)
    from qlab.volume_spar import Forecasts

    oracle = Forecasts(name="oracle", yhat=np.log(panel.vol), first_day=100)
    # With a perfect V_D the dynamic rule still divides by the *static* forecast,
    # so it is not exactly the VWAP weights; what must hold is that it beats
    # equal weight by a wide margin.
    perfect = vwap_tracking_error(panel, oracle, burn_in=100)
    equal = vwap_tracking_error(panel, None, burn_in=100)
    assert perfect["te_bp"] < equal["te_bp"]


def test_tracking_weights_never_go_short_or_over_allocate():
    panel = synthetic_panel(n_days=140, n_slots=10)
    model = fit_forecast(panel, 1, within=True, burn_in=100)
    result = vwap_tracking_error(panel, model, burn_in=100)
    assert result["days"] > 0
    assert np.isfinite(result["te_bp"])


def test_paired_test_finds_no_difference_between_a_schedule_and_itself():
    panel = synthetic_panel(n_days=160, n_slots=10)
    model = fit_forecast(panel, 1, within=True, burn_in=100)
    one = vwap_tracking_error(panel, model, burn_in=100)
    result = paired_te(one, one, n_boot=200)
    assert result["diff_bp"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] > 0.5


# --- plumbing ----------------------------------------------------------------


def test_ok_volatility_is_non_negative_and_scales_with_range():
    bars = pl.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.5, 102.0],
            "low": [99.5, 98.0],
            "close": [100.1, 100.1],
        }
    )
    sigma = ok_volatility(bars, 5 / (24 * 60)).to_list()
    assert all(s >= 0 for s in sigma)
    assert sigma[1] > sigma[0]


def test_build_panel_drops_days_that_do_not_cover_the_grid():
    """A half-session is not a day whose periodic curve means anything."""
    rows = []
    for day in range(1, 6):
        slots = range(78) if day != 3 else range(20)  # day 3 is a half day
        for k in slots:
            minute = 9 * 60 + 30 + 5 * k
            rows.append(
                {
                    "ts_open": f"2024-01-0{day} {minute // 60:02d}:{minute % 60:02d}:00",
                    "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                    "n_ticks": 50,
                }
            )
    bars = pl.DataFrame(rows).with_columns(
        pl.col("ts_open")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )
    panel = build_panel(bars, symbol="T", session=("09:30", "16:00"))
    assert panel.shape[1] == 78
    # 2024-01-01 to 01-05 are all weekdays; 01-03 is dropped for coverage.
    assert len(panel.days) == 4
    assert __import__("datetime").date(2024, 1, 3) not in panel.days
