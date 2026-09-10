"""Inference for backtest results: how sure are we that this number is not zero?

Every strategy module in this package ends in a mean return, a Sharpe or a
difference of two Sharpes. None of those is worth reading without an interval
around it, and the usual textbook interval is wrong here for two reasons that
compound:

**Returns are serially dependent.** A daily strategy return is autocorrelated
through volatility clustering, overlapping holds and regime persistence, so the
iid standard error understates the true one - sometimes by a factor of two.
Every t-statistic in this module is therefore HAC (Newey-West) by default, and
the lag truncation is stated rather than hidden.

**Sharpe is a ratio of two estimates.** Its sampling distribution is not the
mean's, and the correction is not a constant: it depends on the Sharpe itself
and on the return's autocorrelation. :func:`sharpe_se` implements Lo (2002)'s
GMM standard error rather than the iid approximation, because the difference is
largest exactly where it matters - a promising strategy on a short sample.

For anything that is not a mean - max drawdown, Calmar, a difference of two
Sharpes - there is no closed form worth trusting, so :func:`bootstrap_ci` and
:func:`paired_bootstrap` resample with the stationary block bootstrap of Politis
and Romano (1994), which preserves the serial dependence the iid bootstrap
destroys.

No scipy. The distribution functions needed - the normal tail and the
chi-square survival function - are short enough to write, and this package's
dependency list is deliberately small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HACResult",
    "RegressionResult",
    "BootstrapResult",
    "ProportionTest",
    "LjungBox",
    "newey_west",
    "newey_west_lags",
    "ols_hac",
    "sharpe",
    "sharpe_se",
    "sharpe_with_se",
    "stationary_bootstrap_indices",
    "bootstrap_ci",
    "paired_bootstrap",
    "two_proportion_z",
    "ljung_box",
    "bonferroni",
    "benjamini_hochberg",
    "normal_sf",
    "chi2_sf",
]


# --------------------------------------------------------------------------
# Distribution functions
# --------------------------------------------------------------------------

def normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _two_sided_normal_p(t: float) -> float:
    if not math.isfinite(t):
        return float("nan")
    return 2.0 * normal_sf(abs(t))


def _gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its series. For x < a+1."""
    term = 1.0 / a
    total = term
    n = 0
    while n < 1000:
        n += 1
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), continued fraction. x >= a+1."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(stat: float, dof: int) -> float:
    """P(chi-square with ``dof`` degrees of freedom > ``stat``)."""
    if dof <= 0:
        return float("nan")
    if stat <= 0:
        return 1.0
    a, x = dof / 2.0, stat / 2.0
    return 1.0 - _gamma_series(a, x) if x < a + 1.0 else _gamma_cf(a, x)


# --------------------------------------------------------------------------
# Newey-West
# --------------------------------------------------------------------------

def newey_west_lags(n: int) -> int:
    """Newey and West (1994)'s automatic truncation, ``floor(4 (n/100)^(2/9))``.

    A function rather than an inline expression because every result that quotes
    a t-statistic has to be able to say which lag count produced it.
    """
    if n <= 1:
        return 0
    return max(0, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class HACResult:
    """A mean, and how far it is from zero in its own standard errors."""

    mean: float
    se: float
    t_stat: float
    p_value: float
    n: int
    lags: int

    def __str__(self) -> str:
        return (f"mean {self.mean:+.6g}  se {self.se:.6g}  "
                f"t {self.t_stat:+.2f}  p {self.p_value:.4f}  "
                f"(n={self.n}, NW lags={self.lags})")


def _bartlett_omega(h: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West long-run covariance of the score ``h`` (n x k), unnormalised.

    ``Omega = Gamma_0 + sum_l w_l (Gamma_l + Gamma_l')`` with Bartlett weights
    ``w_l = 1 - l/(L+1)`` and ``Gamma_l = sum_t h_t h_{t-l}'``. Leaving it
    unnormalised keeps the sandwich formula free of stray factors of n.
    """
    n = h.shape[0]
    omega = h.T @ h
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = h[lag:].T @ h[:-lag]
        omega = omega + weight * (gamma + gamma.T)
    return omega


def newey_west(x, lags: int | None = None) -> HACResult:
    """HAC t-test of ``mean(x) == 0``.

    ``lags=None`` uses :func:`newey_west_lags`; ``lags=0`` is the plain
    heteroskedasticity-robust (White) standard error.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return HACResult(float(a.mean()) if n else float("nan"),
                         float("nan"), float("nan"), float("nan"), n, 0)
    lag = newey_west_lags(n) if lags is None else int(lags)
    mean = float(a.mean())
    u = (a - mean).reshape(-1, 1)
    omega = float(_bartlett_omega(u, lag)[0, 0])
    # (X'X)^-1 = 1/n for a constant regressor, so Var(mean) = Omega / n^2.
    var = omega / (n * n)
    if var <= 0:
        return HACResult(mean, 0.0, float("nan"), float("nan"), n, lag)
    se = math.sqrt(var)
    t = mean / se
    return HACResult(mean, se, t, _two_sided_normal_p(t), n, lag)


@dataclass(frozen=True)
class RegressionResult:
    """OLS coefficients with HAC standard errors."""

    names: tuple[str, ...]
    params: np.ndarray
    se: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r2: float
    adj_r2: float
    n: int
    lags: int
    resid: np.ndarray
    keep: np.ndarray
    """Boolean mask of the input rows that survived the finiteness filter, so a
    caller can line the residuals back up with the series they came from."""

    def coef(self, name: str) -> tuple[float, float, float]:
        """``(estimate, t, p)`` for one regressor, by name."""
        i = self.names.index(name)
        return float(self.params[i]), float(self.t_stats[i]), float(self.p_values[i])

    def table(self) -> str:
        width = max(len(n) for n in self.names)
        head = f"  {'term':<{width}}  {'coef':>12}  {'se':>10}  {'t':>7}  {'p':>7}"
        rows = [
            f"  {name:<{width}}  {b:>12.6g}  {s:>10.4g}  {t:>+7.2f}  {pv:>7.4f}"
            for name, b, s, t, pv in zip(
                self.names, self.params, self.se, self.t_stats, self.p_values)
        ]
        tail = (f"  R2 {self.r2:.4f}   adj {self.adj_r2:.4f}   "
                f"n {self.n}   NW lags {self.lags}")
        return "\n".join([head] + rows + [tail])


def ols_hac(
    y,
    X,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    add_const: bool = True,
    lags: int | None = None,
) -> RegressionResult:
    """OLS of ``y`` on ``X`` with Newey-West standard errors.

    Rows in which anything is non-finite are dropped jointly, so the design
    matrix and the left-hand side always describe the same observations - the
    alternative is a regression whose n differs by column, which silently
    changes what is being compared.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.size:
        raise ValueError(f"{y.size} observations against {X.shape[0]} rows of X")
    cols = list(names) if names is not None else [f"x{i + 1}" for i in range(X.shape[1])]
    if add_const:
        X = np.column_stack([np.ones(X.shape[0]), X])
        cols = ["const"] + cols
    if len(cols) != X.shape[1]:
        raise ValueError(f"{len(cols)} names for {X.shape[1]} regressors")

    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[keep], X[keep]
    n, k = X.shape
    if n <= k:
        raise ValueError(f"{n} usable observations for {k} regressors")

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    lag = newey_west_lags(n) if lags is None else int(lags)
    omega = _bartlett_omega(X * resid.reshape(-1, 1), lag)
    # The n/(n-k) small-sample correction every HAC implementation applies.
    cov = xtx_inv @ omega @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = np.array([_two_sided_normal_p(v) for v in t])

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if ss_tot > 0 else float("nan")
    return RegressionResult(tuple(cols), beta, se, t, p, r2, adj, n, lag,
                            resid, keep)


# --------------------------------------------------------------------------
# Sharpe ratio and its standard error
# --------------------------------------------------------------------------

def sharpe(x, periods_per_year: int = 252) -> float:
    """Annualised Sharpe of a periodic return series (excess returns assumed)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(a.mean() / sd * math.sqrt(periods_per_year))


def sharpe_se(x, periods_per_year: int = 252, lags: int | None = None) -> float:
    """Lo (2002) standard error of the annualised Sharpe ratio.

    Lo's point is that the iid formula ``sqrt((1 + SR^2/2)/n)`` is not
    conservative under serial dependence: a positively autocorrelated return
    series makes the naive standard error too small, so the strategy looks more
    significant than it is. This is his GMM version - the delta method applied
    to the HAC covariance of ``(r, r^2)`` - which collapses to the iid formula
    when the returns really are independent and normal.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        return float("nan")
    mu = float(a.mean())
    gamma = float((a * a).mean())
    var = gamma - mu * mu
    if var <= 0:
        return float("nan")
    sd = math.sqrt(var)
    sr = mu / sd

    grad = np.array([(1.0 + sr * sr) / sd, -sr / (2.0 * var)])
    h = np.column_stack([a - mu, a * a - gamma])
    lag = newey_west_lags(n) if lags is None else int(lags)
    # _bartlett_omega is an unnormalised sum; the GMM covariance is its mean.
    sigma = _bartlett_omega(h, lag) / n
    v = float(grad @ sigma @ grad)
    if v <= 0:
        return float("nan")
    return math.sqrt(v / n) * math.sqrt(periods_per_year)


def sharpe_with_se(x, periods_per_year: int = 252,
                   lags: int | None = None) -> HACResult:
    """Annualised Sharpe, its Lo standard error, and the implied t-statistic."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    sr = sharpe(a, periods_per_year)
    se = sharpe_se(a, periods_per_year, lags)
    t = sr / se if se and math.isfinite(se) and se > 0 else float("nan")
    lag = newey_west_lags(a.size) if lags is None else int(lags)
    return HACResult(sr, se, t, _two_sided_normal_p(t), a.size, lag)


# --------------------------------------------------------------------------
# Stationary block bootstrap
# --------------------------------------------------------------------------

def _default_block(n: int) -> int:
    """Mean block length. ``n^(1/3)`` is the standard rule of thumb."""
    return max(2, int(round(n ** (1.0 / 3.0))))


def stationary_bootstrap_indices(
    n: int,
    *,
    block_len: int | None = None,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``(n_boot, n)`` index matrix for the Politis-Romano (1994) bootstrap.

    Blocks have geometrically distributed lengths with mean ``block_len`` and
    wrap around the end of the sample, which is what makes the resampled series
    stationary - the fixed-block bootstrap is not, and its bias shows up exactly
    in the autocorrelation the block was there to preserve.
    """
    if n < 2:
        raise ValueError("need at least two observations to bootstrap")
    rng = np.random.default_rng() if rng is None else rng
    b = _default_block(n) if block_len is None else max(1, int(block_len))
    p = 1.0 / b

    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    # At each step either continue the current block or jump somewhere new.
    restart = rng.random((n_boot, n - 1)) < p
    fresh = rng.integers(0, n, size=(n_boot, n - 1))
    for j in range(1, n):
        cont = (idx[:, j - 1] + 1) % n
        idx[:, j] = np.where(restart[:, j - 1], fresh[:, j - 1], cont)
    return idx


@dataclass(frozen=True)
class BootstrapResult:
    """A statistic, a confidence interval, and a two-sided p-value."""

    point: float
    lo: float
    hi: float
    se: float
    p_value: float
    n_boot: int
    block_len: int
    alpha: float

    def __str__(self) -> str:
        return (f"{self.point:+.4g}  [{self.lo:+.4g}, {self.hi:+.4g}] "
                f"{100 * (1 - self.alpha):.0f}%  p {self.p_value:.4f}  "
                f"(B={self.n_boot}, block={self.block_len})")


def _boot_p_value(draws: np.ndarray) -> float:
    """Two-sided percentile p-value: how much of the distribution sits the wrong side of zero."""
    good = draws[np.isfinite(draws)]
    if good.size == 0:
        return float("nan")
    below = float((good <= 0).mean())
    above = float((good >= 0).mean())
    return min(1.0, 2.0 * min(below, above))


def bootstrap_ci(
    x,
    stat_fn=np.mean,
    *,
    n_boot: int = 1000,
    block_len: int | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Percentile interval for any statistic of one series.

    ``stat_fn`` receives a resampled 1-d array and returns a scalar, so Sharpe,
    Sortino, Calmar and max drawdown all go through the same code path as the
    mean. That is the point: the interval around a Calmar is the one nobody
    computes, and it is usually enormous.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"), 0, 0, alpha)
    b = _default_block(n) if block_len is None else max(1, int(block_len))
    idx = stationary_bootstrap_indices(n, block_len=b, n_boot=n_boot, rng=rng)
    draws = np.array([stat_fn(a[row]) for row in idx], dtype=float)
    good = draws[np.isfinite(draws)]
    if good.size == 0:
        return BootstrapResult(float(stat_fn(a)), float("nan"), float("nan"),
                               float("nan"), float("nan"), n_boot, b, alpha)
    lo, hi = np.quantile(good, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(float(stat_fn(a)), float(lo), float(hi),
                           float(good.std(ddof=1)), _boot_p_value(good),
                           n_boot, b, alpha)


def paired_bootstrap(
    a,
    b,
    stat_fn=np.mean,
    *,
    n_boot: int = 1000,
    block_len: int | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Interval for ``stat_fn(a) - stat_fn(b)`` on two contemporaneous series.

    Both series are indexed by the *same* resampled dates. That is what makes it
    a paired test: two exit rules on the same signal share most of their trades,
    so resampling them independently would compare them through a variance the
    common dates have already cancelled, and would find nothing significant
    however large the true difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired series must align: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = a.size
    if n < 3:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"), 0, 0, alpha)
    blk = _default_block(n) if block_len is None else max(1, int(block_len))
    idx = stationary_bootstrap_indices(n, block_len=blk, n_boot=n_boot, rng=rng)
    draws = np.array([stat_fn(a[row]) - stat_fn(b[row]) for row in idx], dtype=float)
    good = draws[np.isfinite(draws)]
    point = float(stat_fn(a) - stat_fn(b))
    if good.size == 0:
        return BootstrapResult(point, float("nan"), float("nan"),
                               float("nan"), float("nan"), n_boot, blk, alpha)
    lo, hi = np.quantile(good, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(point, float(lo), float(hi), float(good.std(ddof=1)),
                           _boot_p_value(good), n_boot, blk, alpha)


# --------------------------------------------------------------------------
# Proportions, serial dependence, multiplicity
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProportionTest:
    p1: float
    p2: float
    diff: float
    z: float
    p_value: float

    def __str__(self) -> str:
        return (f"{100 * self.p1:.1f}% vs {100 * self.p2:.1f}%  "
                f"diff {100 * self.diff:+.1f}pp  z {self.z:+.2f}  p {self.p_value:.4f}")


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> ProportionTest:
    """Pooled two-proportion z-test - for a win rate between two variants."""
    if n1 <= 0 or n2 <= 0:
        return ProportionTest(float("nan"), float("nan"), float("nan"),
                              float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2))
    if se <= 0:
        return ProportionTest(p1, p2, p1 - p2, float("nan"), float("nan"))
    z = (p1 - p2) / se
    return ProportionTest(p1, p2, p1 - p2, z, _two_sided_normal_p(z))


@dataclass(frozen=True)
class LjungBox:
    stat: float
    p_value: float
    lags: int
    n: int
    robust: bool

    def __str__(self) -> str:
        kind = "robust " if self.robust else ""
        return f"{kind}Q({self.lags}) = {self.stat:.2f}  p {self.p_value:.4f}  (n={self.n})"


def ljung_box(x, lags: int = 10, *, robust: bool = True) -> LjungBox:
    """Test for serial dependence in a return series.

    ``robust=True`` is the heteroskedasticity-corrected statistic of Lobato,
    Nankervis and Savin (2001) rather than the textbook Ljung-Box. The
    correction matters for exactly this application: the classical statistic
    assumes a constant conditional variance, and a return series that clusters
    its volatility violates that badly enough to reject independence on the
    volatility alone - which is not the question being asked.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    h = min(int(lags), max(1, n // 4))
    if n < 8 or h < 1:
        return LjungBox(float("nan"), float("nan"), h, n, robust)

    d = a - a.mean()
    c0 = float((d * d).mean())
    if c0 <= 0:
        return LjungBox(float("nan"), float("nan"), h, n, robust)

    stat = 0.0
    for k in range(1, h + 1):
        rho = float((d[k:] * d[:-k]).mean()) / c0
        if robust:
            tau = float((d[k:] ** 2 * d[:-k] ** 2).mean()) / (c0 * c0)
            if tau <= 0:
                continue
            stat += n * rho * rho / tau
        else:
            stat += n * (n + 2) * rho * rho / (n - k)
    return LjungBox(stat, chi2_sf(stat, h), h, n, robust)


def bonferroni(p_values) -> np.ndarray:
    """Family-wise adjusted p-values: ``min(1, m * p)``."""
    p = np.asarray(p_values, dtype=float)
    return np.minimum(1.0, p * p.size)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), in the input's own order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # Enforce monotonicity, walking down from the largest p.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(1.0, ranked)
    return out
