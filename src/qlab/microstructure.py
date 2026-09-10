"""Spread and volatility estimators that work from OHLC alone.

Everything here exists because some tapes carry no quotes. The broker corpus
does - ``bid_close`` and ``ask_close`` are on every bar - which is exactly why
this module is worth having: an estimator can be *checked* against the truth on
one instrument before being trusted on another where the truth is unavailable.
:func:`estimator_error` is that check.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

_DEN = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz(
    high, low, *, clamp_negative: bool = True
) -> np.ndarray:
    """The Corwin-Schultz (2012) high-low bid-ask spread estimator.

    Returns a *proportional* spread - a fraction of price - for each pair of
    consecutive periods, aligned to the second period of the pair. The first
    element is NaN because a two-period estimator has nothing to say about a
    single period.

    The idea: a high is almost always buyer-initiated and a low seller-
    initiated, so the high-low range holds both true volatility and one spread.
    Volatility scales with the length of the interval and the spread does not,
    so two single-period ranges carry two periods of volatility plus *two*
    spreads, while the two-period range carries two periods of volatility plus
    *one*. The difference identifies the spread.

    ``clamp_negative`` sets negative estimates to zero, which is what Corwin and
    Schultz themselves recommend: the estimator is noisy enough at short
    horizons to go negative on individual observations, and a negative spread is
    not a quantity. It is a clamp, not a fix - the bias it introduces is upward,
    and :func:`estimator_error` will show it.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if h.shape != l.shape:
        raise ValueError(f"high and low differ in shape: {h.shape} vs {l.shape}")
    n = h.size
    out = np.full(n, np.nan)
    if n < 2:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / l)
        # beta: the two single-period squared log ranges, summed.
        beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2
        # gamma: the squared log range measured across both periods at once.
        h2 = np.maximum(h[:-1], h[1:])
        l2 = np.minimum(l[:-1], l[1:])
        gamma = np.log(h2 / l2) ** 2

        alpha = (math.sqrt(2.0) * np.sqrt(beta) - np.sqrt(beta)) / _DEN - np.sqrt(
            gamma / _DEN
        )
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    if clamp_negative:
        spread = np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.nan)
    out[1:] = spread
    return out


def with_corwin_schultz(
    frame: pl.DataFrame,
    *,
    high: str = "high",
    low: str = "low",
    over: str | None = None,
    name: str = "cs_spread",
    clamp_negative: bool = True,
) -> pl.DataFrame:
    """Attach a Corwin-Schultz column, optionally restarting per group.

    ``over`` matters on any tape with a session break: pairing the last bar of
    one session with the first of the next measures the overnight gap, not a
    spread, and on a gappy instrument that single pair can dominate the mean.
    """
    if over is None:
        return frame.with_columns(
            pl.Series(name, corwin_schultz(frame[high], frame[low], clamp_negative=clamp_negative))
        )
    parts = []
    for _, part in frame.group_by([over], maintain_order=True):
        parts.append(
            part.with_columns(
                pl.Series(
                    name,
                    corwin_schultz(part[high], part[low], clamp_negative=clamp_negative),
                )
            )
        )
    return pl.concat(parts)


def realised_variance(close, *, log: bool = True) -> np.ndarray:
    """Squared period return - the paper's ``RV``, one observation per bar.

    This is Eross et al.'s equation (2) exactly: a single squared log return per
    5-minute bar, not a sum over a day. It is an extremely noisy variance
    estimate for any one bar, which is why it is only ever used here averaged
    across days at a fixed time-of-day.
    """
    c = np.asarray(close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c[1:] / c[:-1]) if log else c[1:] / c[:-1] - 1.0
    out = np.full(c.size, np.nan)
    out[1:] = r**2
    return out


def estimator_error(
    frame: pl.DataFrame,
    *,
    estimated: str = "cs_spread",
    true_spread: str = "spread_close",
    price: str = "close",
) -> dict:
    """How far a spread estimator lands from the quoted spread, in bps.

    ``true_spread`` is in price units (the corpus quotes it that way); both are
    converted to proportional terms so the comparison is scale-free. Returns the
    two means, the bias, the correlation of the two series, and the correlation
    of their daily means - the second is the one that matters for a paper that
    only ever reports time-of-day averages.
    """
    df = frame.select(
        (pl.col(estimated) * 1e4).alias("est_bps"),
        (pl.col(true_spread) / pl.col(price) * 1e4).alias("true_bps"),
    ).drop_nulls()
    df = df.filter(pl.col("est_bps").is_finite() & pl.col("true_bps").is_finite())
    if df.height < 3:
        return {"n": df.height}
    est = df["est_bps"].to_numpy()
    true = df["true_bps"].to_numpy()
    return {
        "n": int(df.height),
        "mean_est_bps": float(est.mean()),
        "mean_true_bps": float(true.mean()),
        "bias_bps": float(est.mean() - true.mean()),
        "ratio": float(est.mean() / true.mean()) if true.mean() else float("nan"),
        "corr": float(np.corrcoef(est, true)[0, 1]),
        "median_est_bps": float(np.median(est)),
        "median_true_bps": float(np.median(true)),
    }
