"""Bitcoin intraday dynamics, and the strategies they imply - after Eross,
McGroarty, Urquhart and Wolfe (2017).

The paper proposes no trading strategy
--------------------------------------
*The Intraday Dynamics of Bitcoin* (SSRN 3013699) is a descriptive
microstructure study of BTC-e 5-minute data from 1 November 2014 to 31 October
2016. It reports intraday shapes and dependence, and its closing sentence is
that the findings "will be of great interest to ... Bitcoin investors who could
use this information in their trading strategies". It never states a rule, an
entry, an exit or a backtest. Anything called "the paper's strategy" is
therefore a construction, and this module keeps the construction visible by
splitting the work in two.

**Part one replicates the descriptive claims** on a different exchange, a
different decade and a 3.3x longer sample. The findings under test:

=====  =====================================================================
F1     Volume is low until 07:00 GMT, rises to a late-morning peak, peaks
       again around 14:00 and decays - an "n-shape", attributed to European
       and North American trading hours.
F2     Realised variance is highest 07:00-18:00 GMT and declines after.
F3     The bid-ask spread is n-shaped and tracks RV closely, which the paper
       attributes to the absence of market makers (Roll 1984).
F4     Returns are highest between 08:00 and 16:00 GMT.
F5     Returns correlate negatively with volume and with RV, positively with
       the spread; every pair shows bilateral Granger causality at 7 lags.
=====  =====================================================================

**Part two tests the two strategies those claims imply.** Neither is the
paper's; both follow from it directly, and both are stated here before being
run rather than selected after.

``SessionStrategy``
    F4 says the return accrues in a window. Hold BTC long inside 08:00-16:00
    GMT and flat outside it. If F4 is a tradable statement, this beats holding
    around the clock per unit of risk. This is the *only* directional claim in
    the paper, so it is the honest reading of "use this in your strategy".

``ReversalStrategy``
    F5's negative cross-correlation between returns and *lagged* volume and RV
    is a prediction: a bar that moves hard on heavy volume gives some of it
    back. Trade against the last bar's return, but only when that bar was in
    the top ``q`` of the trailing volume (or RV) distribution - the conditioning
    is what makes it F5's claim rather than plain 5-minute reversal, so
    :func:`unconditional_reversal` is run alongside as the control.

What this corpus can and cannot test
------------------------------------
**The exchange is not BTC-e.** BTC-e was seized by US authorities in July 2017
and no longer exists. This uses Binance spot BTCUSDT, the venue with the
largest share of USD-denominated spot volume for most of the sample. That is
the point of the replication rather than a compromise in it: F1-F4 are claims
about *Bitcoin's clock*, and a clock that only existed on one defunct exchange
in 2015 is not what the paper argues for.

**Volume here is real.** BTC-e trade volume in the paper and Binance base-asset
volume here are both genuine traded size, so F1 and F5 are tested on the
quantity they were stated about - unlike everything else in this project, where
"volume" means quote updates.

**The spread is estimated, as the paper's was.** Binance klines carry no
quotes, so F3 uses the same Corwin-Schultz (2012) estimator the paper used.
:func:`qlab.microstructure.estimator_error` is run on USTEC first, where the
true quoted spread *is* known, to say what that estimator is worth before any
conclusion rests on it.

**Costs.** Binance spot at the standard taker tier is 10 bps a side, and the
sample's own Corwin-Schultz spread supplies the rest. A 5-minute reversal rule
turns over enormously, so cost is not a footnote here - it is the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from ..microstructure import corwin_schultz, realised_variance

BINANCE = Path("data/external/binance")

BARS_PER_DAY = 288  # 5-minute bars in 24 hours
MINUTES = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_btc(
    start: str | None = None,
    end: str | None = None,
    *,
    root: Path = BINANCE,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
) -> pl.DataFrame:
    """Load the Binance 5-minute klines, in GMT, with derived variables.

    Timestamps are already UTC, which is GMT for this purpose and is the clock
    the paper works in - it follows Ranaldo (2009) in stamping everything GMT so
    that daylight saving does not smear the session boundaries.
    """
    files = sorted(root.glob(f"{symbol}_{interval}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no {symbol} {interval} parquet under {root}; "
            "run scripts/pipeline/fetch_btc_binance.py first"
        )
    frame = pl.concat([pl.read_parquet(f) for f in files]).sort("ts_open").unique(
        subset=["ts_open"], keep="first", maintain_order=True
    )
    if start is not None:
        frame = frame.filter(pl.col("ts_open") >= pl.lit(start).str.to_datetime().dt.replace_time_zone("UTC"))
    if end is not None:
        frame = frame.filter(pl.col("ts_open") < pl.lit(end).str.to_datetime().dt.replace_time_zone("UTC"))
    return with_variables(frame)


def with_variables(frame: pl.DataFrame) -> pl.DataFrame:
    """The paper's four variables of interest, on its own definitions.

    ``ret`` is equation (1): 100 x the log return, in percent. ``rv`` is
    equation (2): the squared log return of that bar - a single squared return,
    not a daily sum. ``cs_spread`` is equations (3)-(6), the Corwin-Schultz
    proportional spread from consecutive high-low ranges.
    """
    close = frame["close"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.full(close.size, np.nan)
        logret[1:] = 100.0 * np.log(close[1:] / close[:-1])
    # NaN and null are different things in polars, and only null is skipped by
    # mean/corr. These estimators emit NaN for the first bar and for degenerate
    # inputs, so they are converted once, here - otherwise a single NaN turns an
    # entire hourly average into NaN and the failure looks like missing data.
    return frame.with_columns(
        pl.Series("ret", logret).fill_nan(None),
        pl.Series("rv", realised_variance(close)).fill_nan(None),
        pl.Series(
            "cs_spread", corwin_schultz(frame["high"], frame["low"])
        ).fill_nan(None),
        (
            pl.col("ts_open").dt.hour().cast(pl.Int32) * 60
            + pl.col("ts_open").dt.minute().cast(pl.Int32)
        ).alias("tod"),
        pl.col("ts_open").dt.date().alias("day"),
        pl.col("ts_open").dt.weekday().alias("weekday"),
    )


# ---------------------------------------------------------------------------
# Part one: the descriptive claims
# ---------------------------------------------------------------------------


def intraday_profile(frame: pl.DataFrame, *, by: str = "tod") -> pl.DataFrame:
    """Mean of each variable at each time-of-day - the paper's Figures 3-5."""
    return (
        frame.group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col("ret").mean().alias("ret"),
            pl.col("volume").mean().alias("volume"),
            pl.col("quote_volume").mean().alias("quote_volume"),
            pl.col("trades").mean().alias("trades"),
            pl.col("rv").mean().alias("rv"),
            (pl.col("cs_spread") * 1e4).mean().alias("cs_spread_bps"),
        )
        .sort(by)
    )


def hourly_profile(frame: pl.DataFrame) -> pl.DataFrame:
    return intraday_profile(
        frame.with_columns((pl.col("tod") // 60).alias("hour")), by="hour"
    )


def session_returns(frame: pl.DataFrame, *, lo: int = 8, hi: int = 16) -> pl.DataFrame:
    """Total return inside and outside the paper's 08:00-16:00 GMT window.

    One row per day carrying the two log-return sums, so the comparison is
    between two series of the same length on the same days and a paired test
    is available.
    """
    inside = (pl.col("tod") >= lo * 60) & (pl.col("tod") < hi * 60)
    return (
        frame.group_by("day")
        .agg(
            pl.col("ret").filter(inside).sum().alias("in_pct"),
            pl.col("ret").filter(~inside).sum().alias("out_pct"),
            pl.col("ret").sum().alias("all_pct"),
            pl.col("rv").filter(inside).sum().alias("rv_in"),
            pl.col("rv").filter(~inside).sum().alias("rv_out"),
            pl.len().alias("bars"),
        )
        .filter(pl.col("bars") >= 0.9 * BARS_PER_DAY)
        .sort("day")
    )


def cross_correlation(
    frame: pl.DataFrame, a: str, b: str, *, max_lag: int = 12
) -> pl.DataFrame:
    """corr(a_t, b_{t+j}) for j in [-max_lag, max_lag] - the paper's Figures 6-11.

    Positive ``j`` means ``b`` leads is *wrong*: with ``b`` shifted forward, a
    non-zero correlation at ``j > 0`` says ``a`` today moves with ``b``
    tomorrow, i.e. ``a`` leads. The column is named accordingly so the direction
    cannot be misread off the sign alone.
    """
    x = frame[a].to_numpy().astype(float)
    y = frame[b].to_numpy().astype(float)
    rows = []
    n = x.size
    for j in range(-max_lag, max_lag + 1):
        if j >= 0:
            xa, ya = x[: n - j], y[j:]
        else:
            xa, ya = x[-j:], y[: n + j]
        ok = np.isfinite(xa) & np.isfinite(ya)
        if ok.sum() < 30:
            continue
        r = float(np.corrcoef(xa[ok], ya[ok])[0, 1])
        rows.append(
            {
                "lag": j,
                "corr": r,
                "leads": a if j > 0 else (b if j < 0 else "contemporaneous"),
                "n": int(ok.sum()),
            }
        )
    return pl.DataFrame(rows)


def granger(frame: pl.DataFrame, cause: str, effect: str, *, lags: int = 7) -> dict:
    """Wald F-test that ``cause`` Granger-causes ``effect`` at ``lags`` lags.

    The paper's Table 3, at its own lag length of 7 (chosen, it says, because
    the market trades seven days a week). Restricted model: ``effect`` on its
    own lags. Unrestricted: plus the lags of ``cause``.
    """
    from ..stats import chi2_sf

    y = frame[effect].to_numpy().astype(float)
    x = frame[cause].to_numpy().astype(float)
    n = y.size
    rows = n - lags
    if rows < 10 * lags:
        return {"n": rows, "f_stat": float("nan"), "p_value": float("nan")}

    Y = y[lags:]
    cols_r = [np.ones(rows)] + [y[lags - k : n - k] for k in range(1, lags + 1)]
    cols_u = cols_r + [x[lags - k : n - k] for k in range(1, lags + 1)]
    Xr = np.column_stack(cols_r)
    Xu = np.column_stack(cols_u)

    ok = np.isfinite(Y) & np.all(np.isfinite(Xu), axis=1)
    Y, Xr, Xu = Y[ok], Xr[ok], Xu[ok]
    m = Y.size

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        e = Y - X @ beta
        return float(e @ e)

    rss_r, rss_u = rss(Xr), rss(Xu)
    df1, df2 = lags, m - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return {"n": m, "f_stat": float("nan"), "p_value": float("nan")}
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    # F(df1, df2) -> chi2(df1)/df1 as df2 grows; m is ~10^5 here, so the
    # chi-square tail is the F tail to well past the digits reported.
    return {
        "cause": cause,
        "effect": effect,
        "lags": lags,
        "n": m,
        "f_stat": f,
        "p_value": chi2_sf(f * df1, df1),
    }


# ---------------------------------------------------------------------------
# Part two: the strategies the claims imply
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSpec:
    """What a round turn costs on Binance spot.

    ``taker_bps`` is the published standard taker fee (0.10% a side). ``spread``
    adds the sample's own estimated half-spread on each side when
    ``pay_spread`` is set, which is what a market order actually crosses.
    """

    taker_bps: float = 10.0
    pay_spread: bool = True
    spread_multiplier: float = 1.0


@dataclass(frozen=True)
class SessionConfig:
    """F4 as a rule: long inside the window, flat outside."""

    lo_hour: int = 8
    hi_hour: int = 16
    cost: CostSpec = CostSpec()


def session_strategy(frame: pl.DataFrame, cfg: SessionConfig = SessionConfig()) -> pl.DataFrame:
    """Daily returns of holding long only inside the GMT window.

    Two round turns a day at most - in at the open of the window, out at its
    close - so cost is charged twice per day the position is taken, and the
    comparison against buy-and-hold is on the same days.
    """
    per_day = session_returns(frame, lo=cfg.lo_hour, hi=cfg.hi_hour)
    spread_bps = float(
        (frame["cs_spread"].drop_nulls().drop_nans() * 1e4).mean()
    ) if cfg.cost.pay_spread else 0.0
    per_turn = cfg.cost.taker_bps + 0.5 * cfg.cost.spread_multiplier * spread_bps
    cost_bps = 2.0 * per_turn

    return per_day.with_columns(
        (pl.col("in_pct") * 100.0).alias("gross_bps"),
        pl.lit(cost_bps).alias("cost_bps"),
    ).with_columns(
        (pl.col("gross_bps") - pl.col("cost_bps")).alias("net_bps"),
        (pl.col("all_pct") * 100.0).alias("bh_bps"),
        (pl.col("out_pct") * 100.0).alias("outside_bps"),
    )


@dataclass(frozen=True)
class ReversalConfig:
    """F5 as a rule: fade the last bar, when that bar was unusual."""

    signal: str = "volume"
    """``volume``, ``rv``, ``trades``, or ``none`` for the unconditional
    control."""

    quantile: float = 0.9
    """Trade only when the conditioning variable is above this quantile of its
    own trailing distribution."""

    window: int = BARS_PER_DAY
    """Bars in the trailing distribution. One day, so the threshold adapts to
    the level of activity without looking ahead."""

    hold: int = 1
    """Bars held. 1 is the paper's own horizon - its cross-correlations are
    computed at the 5-minute frequency."""

    cost: CostSpec = CostSpec()

    def __post_init__(self) -> None:
        if self.signal not in ("volume", "rv", "trades", "none"):
            raise ValueError(f"unknown signal {self.signal!r}")
        if not 0.0 <= self.quantile < 1.0:
            raise ValueError("quantile must be in [0, 1)")


def reversal_strategy(
    frame: pl.DataFrame, cfg: ReversalConfig = ReversalConfig()
) -> pl.DataFrame:
    """Per-bar returns of fading the previous bar's move.

    The position at bar ``t`` is ``-sign(ret_{t-1})``, taken only when bar
    ``t-1``'s conditioning variable exceeded its own trailing quantile. Both
    inputs are lagged, so nothing at ``t`` informs the trade held over ``t``.
    """
    df = frame.drop_nulls("ret")
    ret = df["ret"].to_numpy() / 100.0  # log return as a fraction

    if cfg.signal == "none":
        gate = np.ones(ret.size, dtype=bool)
    else:
        s = df[cfg.signal].to_numpy().astype(float)
        thresh = (
            pl.Series(s)
            .rolling_quantile(cfg.quantile, window_size=cfg.window, min_samples=cfg.window // 2)
            .shift(1)
            .to_numpy()
        )
        gate = np.isfinite(thresh) & (s > thresh)

    # Everything the trade depends on is observed at the close of bar t-1.
    prev_ret = np.concatenate([[np.nan], ret[:-1]])
    prev_gate = np.concatenate([[False], gate[:-1]])
    position = np.where(prev_gate & np.isfinite(prev_ret), -np.sign(prev_ret), 0.0)

    spread_bps = (
        float((df["cs_spread"].drop_nulls().drop_nans() * 1e4).mean())
        if cfg.cost.pay_spread
        else 0.0
    )
    per_turn = cfg.cost.taker_bps + 0.5 * cfg.cost.spread_multiplier * spread_bps
    turns = np.abs(np.diff(position, prepend=0.0))

    gross_bps = 1e4 * position * ret
    cost_bps = turns * per_turn
    return df.with_columns(
        pl.Series("position", position),
        pl.Series("gross_bps", gross_bps),
        pl.Series("cost_bps", cost_bps),
        pl.Series("net_bps", gross_bps - cost_bps),
    )


def unconditional_reversal(
    frame: pl.DataFrame, cfg: ReversalConfig = ReversalConfig()
) -> pl.DataFrame:
    """The control: the same rule with the conditioning switched off.

    If this earns what the conditioned version earns, F5's volume and RV terms
    add nothing and the result is plain 5-minute mean reversion.
    """
    return reversal_strategy(frame, replace(cfg, signal="none"))


def to_daily(bar_returns: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-bar bps to per-day bps, for a comparable tearsheet."""
    return (
        bar_returns.group_by("day")
        .agg(
            pl.col("gross_bps").sum(),
            pl.col("cost_bps").sum(),
            pl.col("net_bps").sum(),
            (pl.col("position").abs() > 0).sum().alias("bars_in"),
        )
        .sort("day")
    )
