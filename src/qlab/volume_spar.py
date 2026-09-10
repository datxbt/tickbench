"""Intraday volume forecasting by within-transformation, after Tan, Zhang and Zhu (2026).

The claim being tested
----------------------
*Forecasting Intraday Trading Volume with Periodicity* (SSRN 5757622) proposes
the **Separated Periodic AutoRegressive** model. Its argument in one line:
intraday volume has a strong time-of-day pattern, the usual way to handle it is
to *pre-specify* the shape - a U-curve fitted by splines, as in the Component
Multiplicative Error Model - and pre-specifying it is a mistake, because real
periodic curves are U-shaped, J-shaped, W-shaped or none of those depending on
the asset and the market.

SPAR removes the pattern instead of modelling it. Write log volume as::

    y[t,k] = h(X[t,k-1]) + f[k] + u[t,k]

where ``f[k]`` is a time-of-day fixed effect. Subtract the trailing ``W``-day
mean at the same time of day - the panel *within* transformation - and ``f[k]``
cancels without ever being named::

    y.[t,k] = X.[t,k-1] * beta + u.[t,k]

Estimate ``beta`` by OLS on the demeaned panel, then forecast by adding the
trailing mean back::

    yhat[t+1,k+1] = (X[t+1,k] - Xbar[t,k]) * beta + ybar[t,k+1]

The paper's headline numbers, on 30 US equities at 5-minute frequency,
2010-2020, 2,376 out-of-sample days: average out-of-sample R-squared of 0.293
for SPAR3 against 0.103 for the same features without the within-transformation
(OLS3), and 0.259 for the CMEM-Kalman benchmark. Applied to a dynamic VWAP
replication schedule, mean tracking error falls to 7.013 bp for SPAR3 from
7.275 bp (OLS3), 8.007 bp (CMEM-KF) and 8.804 bp (equal weight).

Features, exactly as specified
------------------------------
* ``X1`` intraday - the last six log-volume observations.
* ``X2`` cross-day - the same time-of-day slot on each of the previous 22 days.
* ``X3`` volatility - the last six values of the Optimal Candlestick range
  estimator of Li et al. (2024), ``0.811 * w - 0.369 * |r|`` on scaled log
  range and return.

Four specifications: ``1`` = X1, ``2`` = X1+X2, ``3`` = X1+X3, ``4`` = all
three. Each is fitted both with the within-transformation (SPAR) and without
it (OLS), which is the comparison the paper's central claim rests on.

What this corpus can and cannot test
------------------------------------
**Volume here is quote count, not share count.** These are OTC CFD feeds; there
is no consolidated tape and no size on a quote. ``n_ticks`` - the number of
quote revisions the broker published in the bar - is the stand-in, and it is
the same stand-in an FX or CFD execution desk actually works with. It is a
count of *events*, not of *shares*, so the level is not comparable to the
paper's. Everything the paper measures is scale-free, though: R-squared is a
variance ratio, and the VWAP weights are normalised to sum to one, so the units
cancel where it matters.

**The periodicity test is stronger here than in the paper.** US equities have
one session and a U-shape. These instruments run around the clock, so the
24-hour grid has three humps - Tokyo, London, New York - plus a daily broker
break. That is precisely the case SPAR claims to handle and a pre-specified
U-spline cannot, and it is out-of-sample for the paper in the strongest sense:
a different asset class, a different market structure, and years it never saw.
:func:`periodic_curve` returns the estimated shape so the report can show it.

**The CMEM-Kalman benchmark is not reimplemented.** It is an EM-estimated
state-space model and reproducing it faithfully is a project of its own. The
comparison that carries the paper's actual argument - within-transformed
against not, on identical features - is exact, and the historical-mean
benchmark that defines out-of-sample R-squared is exact. Where the paper's KF
column would sit, this module reports the rolling historical mean instead and
says so.

**Tracking error is measured against the realised VWAP of the same tape.** No
market impact is modelled: a replication schedule that trades tiny child orders
into a 24-hour CFD book does not move it, and the paper models none either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# The paper's defaults.
N_INTRADAY_LAGS = 6
N_CROSSDAY_LAGS = 22
N_VOL_LAGS = 6
WINDOW = 22  # W, the cross-day averaging window for the within-transformation

SPECS: dict[int, tuple[str, ...]] = {
    1: ("intraday",),
    2: ("intraday", "crossday"),
    3: ("intraday", "vol"),
    4: ("intraday", "crossday", "vol"),
}


# --- the panel ---------------------------------------------------------------


def ok_volatility(bars: pl.DataFrame, delta: float) -> pl.Series:
    """Optimal Candlestick spot volatility, Li et al. (2024) equation (9).

    ``0.811 * omega - 0.369 * |r|`` on the scaled log range and log return of
    the bar, with the weights chosen to minimise asymptotic variance. Negative
    values are possible when the close-open move nearly spans the whole range;
    they are clipped at zero, which is what a volatility proxy has to be.
    """
    root = np.sqrt(delta)
    log_hi = np.log(bars["high"].to_numpy())
    log_lo = np.log(bars["low"].to_numpy())
    log_op = np.log(bars["open"].to_numpy())
    log_cl = np.log(bars["close"].to_numpy())
    omega = (log_hi - log_lo) / root
    ret = (log_cl - log_op) / root
    return pl.Series("sigma", np.clip(0.811 * omega - 0.369 * np.abs(ret), 0.0, None))


@dataclass
class Panel:
    """One asset's intraday grid: ``T`` days by ``K`` slots, plus its features.

    ``y`` is log volume, shaped ``(T, K)``. ``sigma`` is the volatility proxy on
    the same grid. ``price`` is the slot's closing price, kept for VWAP. ``vol``
    is raw volume, kept because R-squared is defined on levels in the paper's
    VWAP section and on logs in its forecasting section.
    """

    days: list
    slots: list[int]
    y: np.ndarray
    vol: np.ndarray
    sigma: np.ndarray
    price: np.ndarray
    symbol: str
    grid: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.y.shape


def build_panel(
    bars: pl.DataFrame,
    *,
    symbol: str,
    every_min: int = 5,
    session: tuple[str, str] | None = ("09:30", "16:00"),
    tz: str = "America/New_York",
    min_coverage: float = 0.98,
) -> Panel:
    """Reshape 5-minute bars into a complete day-by-slot panel.

    ``session`` restricts to a wall-clock window in ``tz`` - use the US cash
    session to mirror the paper, or ``None`` for the full 24-hour grid, which is
    where the multi-hump periodicity these instruments actually have shows up.

    Days that do not carry at least ``min_coverage`` of the grid are dropped
    whole rather than interpolated. A half-session with four bars in it is not a
    day whose periodic curve means anything, and the paper drops half-days for
    the same reason.
    """
    frame = (
        bars.lazy()
        .with_columns(et=pl.col("ts_open").dt.convert_time_zone(tz))
        .with_columns(
            day=pl.col("et").dt.date(),
            slot=pl.col("et").dt.hour().cast(pl.Int32) * 60
            + pl.col("et").dt.minute().cast(pl.Int32),
        )
    )
    if session is not None:
        lo = int(session[0][:2]) * 60 + int(session[0][3:])
        hi = int(session[1][:2]) * 60 + int(session[1][3:])
        frame = frame.filter((pl.col("slot") >= lo) & (pl.col("slot") < hi))
    frame = frame.filter(pl.col("et").dt.weekday() <= 5).collect()
    if frame.is_empty():
        raise ValueError(f"{symbol}: no bars inside the requested session")

    slots = sorted(frame["slot"].unique().to_list())
    sigma_col = ok_volatility(frame, every_min / (24 * 60))
    frame = frame.with_columns(sigma=sigma_col)

    counts = frame.group_by("day").len()
    keep = set(
        counts.filter(pl.col("len") >= int(min_coverage * len(slots)))["day"].to_list()
    )
    frame = frame.filter(pl.col("day").is_in(list(keep)))
    days = sorted(frame["day"].unique().to_list())

    index = {(d, s): i for i, (d, s) in enumerate(zip(frame["day"], frame["slot"]))}
    shape = (len(days), len(slots))
    vol = np.full(shape, np.nan)
    sigma = np.full(shape, np.nan)
    price = np.full(shape, np.nan)
    raw_vol = frame["n_ticks"].to_numpy().astype(float)
    raw_sig = frame["sigma"].to_numpy()
    raw_px = frame["close"].to_numpy()
    for di, day in enumerate(days):
        for ki, slot in enumerate(slots):
            i = index.get((day, slot))
            if i is not None:
                vol[di, ki] = raw_vol[i]
                sigma[di, ki] = raw_sig[i]
                price[di, ki] = raw_px[i]

    # A slot with no quotes at all is a real zero, not a hole; log(0) is not.
    # Carrying the previous slot forward is what the paper does for missing
    # volatility, and one quote is the smallest volume that can be observed.
    vol = _forward_fill(np.where(vol < 1.0, 1.0, vol))
    sigma = _forward_fill(sigma)
    price = _forward_fill(price)
    return Panel(
        days=days,
        slots=slots,
        y=np.log(vol),
        vol=vol,
        sigma=sigma,
        price=price,
        symbol=symbol,
        grid="session" if session else "24h",
    )


def _forward_fill(matrix: np.ndarray) -> np.ndarray:
    """Fill NaNs along the flattened time order, then backfill any leading ones."""
    flat = matrix.reshape(-1).copy()
    last = np.nan
    for i, value in enumerate(flat):
        if np.isnan(value):
            flat[i] = last
        else:
            last = value
    if np.isnan(flat[0]):
        first = np.nanmin(flat) if np.isfinite(np.nanmin(flat)) else 0.0
        flat = np.nan_to_num(flat, nan=first)
    return flat.reshape(matrix.shape)


# --- features ----------------------------------------------------------------


def design(panel: Panel, spec: int) -> tuple[np.ndarray, list[str]]:
    """The ``(T, K, p)`` predictor tensor for one specification.

    Every column is lagged so that predicting slot ``k`` on day ``t`` uses only
    information stamped at ``k-1`` or earlier. The intraday and volatility lags
    wrap into the previous day at the open, exactly as the paper prescribes for
    ``k = 1``; the cross-day lags read the same slot on earlier days.
    """
    blocks, names = [], []
    n_days, n_slots = panel.shape
    flat_y = panel.y.reshape(-1)
    flat_s = panel.sigma.reshape(-1)

    if "intraday" in SPECS[spec]:
        for lag in range(1, N_INTRADAY_LAGS + 1):
            blocks.append(_shift_flat(flat_y, lag, panel.shape))
            names.append(f"y_lag{lag}")
    if "crossday" in SPECS[spec]:
        for lag in range(1, N_CROSSDAY_LAGS + 1):
            block = np.full(panel.shape, np.nan)
            block[lag:, :] = panel.y[:-lag, :]
            blocks.append(block)
            names.append(f"y_day{lag}")
    if "vol" in SPECS[spec]:
        for lag in range(1, N_VOL_LAGS + 1):
            blocks.append(_shift_flat(flat_s, lag, panel.shape))
            names.append(f"sig_lag{lag}")
    return np.stack(blocks, axis=-1), names


def _shift_flat(flat: np.ndarray, lag: int, shape: tuple[int, int]) -> np.ndarray:
    """Lag by ``lag`` slots in continuous intraday time, wrapping across days."""
    out = np.full(flat.shape, np.nan)
    out[lag:] = flat[:-lag]
    return out.reshape(shape)


# --- estimation --------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> np.ndarray:
    if intercept:
        x = np.hstack([np.ones((x.shape[0], 1)), x])
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() <= x.shape[1]:
        return np.zeros(x.shape[1])
    beta, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    return beta


@dataclass
class Forecasts:
    """One model's out-of-sample log-volume forecasts, aligned to the panel."""

    name: str
    yhat: np.ndarray  # (T, K), NaN before the first forecast day
    first_day: int


def fit_forecast(
    panel: Panel,
    spec: int,
    *,
    within: bool,
    window: int = WINDOW,
    burn_in: int = 90,
    refit_every: int = 22,
) -> Forecasts:
    """Rolling out-of-sample forecasts for one model.

    ``within=True`` is SPAR: both target and predictors are demeaned by their
    trailing ``window``-day mean at the same slot before the regression, and the
    trailing mean is added back to the forecast. ``within=False`` is the OLS
    counterpart on raw levels with an intercept - identical features, identical
    training window, no periodicity handling.

    Coefficients are re-estimated every ``refit_every`` days on the trailing
    ``window`` days and then held fixed, which is the paper's online scheme:
    beta is fixed at the end of a day and used throughout the next. Only data
    strictly earlier than the forecast slot enters either the demeaning or the
    fit - :func:`tests.test_volume_spar` pins that with a shuffled-future test.
    """
    x, _ = design(panel, spec)
    n_days, n_slots = panel.shape
    yhat = np.full(panel.shape, np.nan)
    beta = None

    for t in range(burn_in, n_days):
        if beta is None or (t - burn_in) % refit_every == 0:
            lo = max(0, t - window)
            train_y, train_x = panel.y[lo:t], x[lo:t]
            if within:
                # Demean within the training block itself: each slot against its
                # own mean over those days. This is the transformation in (6).
                ty = train_y - np.nanmean(train_y, axis=0, keepdims=True)
                tx = train_x - np.nanmean(train_x, axis=0, keepdims=True)
            else:
                ty, tx = train_y, train_x
            beta = _ols(
                tx.reshape(-1, tx.shape[-1]), ty.reshape(-1), intercept=not within
            )

        lo = max(0, t - window)
        if within:
            # Trailing means are computed on days strictly before t, so they are
            # known when the forecast is made. Remark 2 of the paper.
            ybar = np.nanmean(panel.y[lo:t], axis=0)
            xbar = np.nanmean(x[lo:t], axis=0)
            yhat[t] = ((x[t] - xbar) @ beta) + ybar
        else:
            row = np.hstack([np.ones((n_slots, 1)), x[t]])
            yhat[t] = row @ beta

    return Forecasts(
        name=f"{'SPAR' if within else 'OLS'}{spec}", yhat=yhat, first_day=burn_in
    )


def historical_mean(panel: Panel, *, window: int = WINDOW, burn_in: int = 90) -> Forecasts:
    """The benchmark that defines out-of-sample R-squared: the trailing slot mean."""
    yhat = np.full(panel.shape, np.nan)
    for t in range(burn_in, panel.shape[0]):
        yhat[t] = np.nanmean(panel.y[max(0, t - window) : t], axis=0)
    return Forecasts(name="HistMean", yhat=yhat, first_day=burn_in)


def periodic_curve(panel: Panel, *, window: int | None = None) -> np.ndarray:
    """The estimated time-of-day curve ``F[k]``, normalised to mean one.

    Equation (8) of the paper, in the degenerate case where beta is zero: the
    exponentiated slot mean of log volume, rescaled. This is SPAR's by-product -
    the shape it never had to assume - and it is what the report plots to show
    that these instruments are not U-shaped.
    """
    block = panel.y if window is None else panel.y[-window:]
    raw = np.nanmean(block, axis=0)
    scaled = np.exp(raw)
    return len(scaled) * scaled / scaled.sum()


# --- scoring -----------------------------------------------------------------


def oos_r2(panel: Panel, model: Forecasts, benchmark: Forecasts, *, on_log: bool = True) -> float:
    """``1 - SSE(model) / SSE(benchmark)`` over every forecast slot.

    ``on_log`` scores in log space, where the models are estimated. The paper's
    equation is written on levels; :func:`score_all` reports both, because a
    model can win on one and lose on the other when the log-normal correction is
    left out, and knowing which is which is the difference between a real
    improvement and a units artefact.
    """
    start = max(model.first_day, benchmark.first_day)
    actual = panel.y[start:] if on_log else panel.vol[start:]
    pred = model.yhat[start:] if on_log else np.exp(model.yhat[start:])
    base = benchmark.yhat[start:] if on_log else np.exp(benchmark.yhat[start:])
    ok = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(base)
    sse = np.sum((actual[ok] - pred[ok]) ** 2)
    sst = np.sum((actual[ok] - base[ok]) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


# --- the VWAP replication strategy -------------------------------------------


def vwap_tracking_error(
    panel: Panel, model: Forecasts | None, *, burn_in: int = 90, window: int = WINDOW
) -> dict:
    """Tracking error of the dynamic VWAP schedule of Bialkowski et al. (2008).

    The rule, equation (12): a static schedule ``V_S`` from the trailing
    22-day slot mean fixes the plan before the day starts; at each slot the
    intraday forecast ``V_D`` revises the weight, and the revision applies only
    to the volume left to trade::

        w[k] = V_D[k] / sum(V_S[k:]) * (1 - sum(w[:k])),   k < K
        w[K] = 1 - sum(w[:K])

    ``model=None`` is the equal-weight benchmark, which uses no forecast at all.
    The error is the signed gap between the day's realised VWAP and the price
    the schedule achieved, in basis points, averaged over days as the paper
    defines it. The mean of the *absolute* gap is reported alongside, because a
    schedule that is 8 bp wrong in both directions and a schedule that is right
    have the same signed mean and are not the same schedule.
    """
    n_days, n_slots = panel.shape
    gaps, on_days = [], []
    for t in range(burn_in, n_days):
        vol, price = panel.vol[t], panel.price[t]
        if not (np.isfinite(vol).all() and np.isfinite(price).all()) or vol.sum() <= 0:
            continue
        market = float(vol @ price / vol.sum())

        if model is None:
            weights = np.full(n_slots, 1.0 / n_slots)
        else:
            static = np.exp(np.nanmean(panel.y[max(0, t - window) : t], axis=0))
            dynamic = np.exp(model.yhat[t])
            if not np.isfinite(dynamic).all() or not np.isfinite(static).all():
                continue
            weights = np.zeros(n_slots)
            used = 0.0
            for k in range(n_slots - 1):
                remaining = static[k:].sum()
                share = (dynamic[k] / remaining) * (1.0 - used) if remaining > 0 else 0.0
                # A forecast can imply more than the whole remaining order.
                share = float(np.clip(share, 0.0, 1.0 - used))
                weights[k] = share
                used += share
            weights[-1] = max(0.0, 1.0 - used)

        achieved = float(weights @ price)
        gaps.append(1e4 * (market - achieved) / market)
        on_days.append(panel.days[t])

    arr = np.asarray(gaps)
    return {
        "model": "EW" if model is None else model.name,
        "days": int(arr.size),
        "te_bp": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
        "signed_bp": float(np.mean(arr)) if arr.size else float("nan"),
        "te_sd_bp": float(np.std(arr)) if arr.size else float("nan"),
        # Per-day errors, kept so two schedules can be compared on the days they
        # share. A 0.3 bp difference in the mean is meaningless without them:
        # the day-to-day standard deviation is twenty times that, so the only
        # way to read the gap is paired, on common dates.
        "_gaps": arr,
        "_days": on_days,
    }


def score_all(
    panel: Panel, *, specs=(1, 2, 3, 4), burn_in: int = 90, window: int = WINDOW
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every model on one panel: accuracy, VWAP tracking error, and the raw
    per-day error series each schedule produced."""
    bench = historical_mean(panel, window=window, burn_in=burn_in)
    models = {}
    for spec in specs:
        for within in (True, False):
            f = fit_forecast(panel, spec, within=within, window=window, burn_in=burn_in)
            models[f.name] = f

    accuracy = pl.DataFrame(
        [
            {
                "symbol": panel.symbol,
                "grid": panel.grid,
                "model": name,
                "r2_log": oos_r2(panel, f, bench, on_log=True),
                "r2_level": oos_r2(panel, f, bench, on_log=False),
            }
            for name, f in models.items()
        ]
    )
    schedules = {"EW": vwap_tracking_error(panel, None, burn_in=burn_in, window=window)}
    for name in ("SPAR1", "OLS1", "SPAR3", "OLS3", "SPAR4", "OLS4"):
        if name in models:
            schedules[name] = vwap_tracking_error(
                panel, models[name], burn_in=burn_in, window=window
            )

    tracking = pl.DataFrame(
        [{k: v for k, v in row.items() if not k.startswith("_")} for row in schedules.values()]
    ).with_columns(symbol=pl.lit(panel.symbol), grid=pl.lit(panel.grid))
    return accuracy, tracking, schedules


def paired_te(left: dict, right: dict, *, n_boot: int = 2000) -> dict:
    """Is ``left``'s schedule really closer to VWAP than ``right``'s?

    Compares absolute per-day tracking error on the days both schedules traded,
    with a stationary block bootstrap, because execution errors cluster: a
    volatile week is bad for every schedule at once. The paper reports a 0.26 bp
    mean advantage for SPAR3 over OLS3 with no interval around it; the day-level
    dispersion in both its sample and this one is over twenty times that, so
    whether such a gap is a finding or a coincidence is exactly what this
    answers.
    """
    from .stats import paired_bootstrap

    common = sorted(set(left["_days"]) & set(right["_days"]))
    li = {d: i for i, d in enumerate(left["_days"])}
    ri = {d: i for i, d in enumerate(right["_days"])}
    a = np.abs(left["_gaps"][[li[d] for d in common]])
    b = np.abs(right["_gaps"][[ri[d] for d in common]])
    boot = paired_bootstrap(a, b, n_boot=n_boot)
    return {
        "left": left["model"],
        "right": right["model"],
        "days": len(common),
        "diff_bp": boot.point,
        "lo_bp": boot.lo,
        "hi_bp": boot.hi,
        "p": boot.p_value,
    }
