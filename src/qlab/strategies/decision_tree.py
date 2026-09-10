"""Decision-tree intraday signals, as specified by Prajwal, Balivada, Nirmala
and Poornoday (2024).

The claim being tested
----------------------
*Decision Trees for Intuitive Intraday Trading Strategies* (SSRN 4838381) fits
one ``DecisionTreeClassifier`` per instrument on nine technical features of
1-minute bars, labels each bar by the **sign of the next bar's return**, and
trades the prediction. Depth 4, Gini impurity, train on 2022-2023, test on
2023-2024, signals shifted forward one bar for execution delay. On NIFTY50
constituents they report a test-set Sharpe of **5.81** against a buy-and-hold
benchmark's 2.36, with a 2.63% maximum drawdown against 7.02%.

The nine features, transcribed from their Section III.A:

=====  =========================================================
f1     1-period return of close
f2     15-period return of close
f3     RSI(14)
f4     ADX(14)
f5     SMA(14) / close
f6     rolling correlation of SMA(14) with close, window 14
f7     rolling volatility of the 1-period return, window 14
f8     rolling volatility of the 15-period return, window 14
f9     rolling VWAP(14) / close
=====  =========================================================

The one number that decides this
--------------------------------
Their Section IV.C states plainly: *"commissions and slippage were not
explicitly incorporated into the backtesting process."* A model that predicts
the next 1-minute bar changes its mind roughly every few minutes, so its
turnover is enormous, and the cost of that turnover is the entire question. This
module therefore reports **the same strategy twice** - once cost-free, which is
what the paper measured, and once through :class:`qlab.costs.CostModel`, which
is what an account would experience. :func:`breakeven_cost_bps` states the
threshold directly: the round-turn cost above which the gross edge is gone.

What differs from the paper, and why
------------------------------------
**The universe.** There is no NIFTY50 here. The paper's method is explicitly
per-name - *"decision trees create unique trading rules for each stock"* - so it
transfers to four instruments as four independent fits, and the paper's PSBBR /
PSBBS statistics (share of names beating their own benchmark) become a count out
of four. Four is not fifty and the report says so; what four *can* do is show
whether the result is instrument-specific or universal, which is the part of the
claim that matters for deployment.

**The split.** Their train/test boundary is a date they chose; this project's
splits were fixed before any strategy existed (:data:`qlab.loader.SPLITS`), so
the tree trains on ``dev`` and is measured on ``validation``. That is the same
discipline applied more strictly.

**Volume is quote updates.** f9's VWAP needs volume and the feed carries no
size, so ``n_ticks`` stands in - the same proxy the rest of this project uses,
and what MT5 reports as volume on these instruments.

**Annualisation.** The paper annualises 1-minute returns by 252 x 375 = 94,500
periods, which is arithmetically correct and makes a Sharpe uncomparable with
every other number in this project. Everything here aggregates net minute
returns into **daily** returns first and annualises by 252, which is what
:mod:`qlab.metrics` does for every other strategy. :func:`paper_style_sharpe`
reproduces their convention alongside, so the two can be read against each other.

**Position mapping.** The paper says 1 is a buy signal and 0 a sell signal, and
benchmarks against buy-and-hold, which leaves it ambiguous whether a 0 means
short or flat. Both are implemented; ``"long_flat"`` is the default because
their reported volatility (3.32% against the benchmark's 10.23%) is far below
what a permanently-invested long/short book would produce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier, export_text

from ..costs import CostModel
from ..loader import load_bars
from ..symbols import get_spec

#: The nine inputs of Section III.A, in the order the paper lists them.
FEATURES: tuple[str, ...] = (
    "ret1", "ret15", "rsi14", "adx14", "sma_ratio",
    "sma_corr", "vol14", "vol210", "vwap_ratio",
)

#: The depths the paper compares in its Figures 1-3, having settled on 4.
DEPTHS: tuple[int, ...] = (3, 4, 5, 6)

TRADING_DAYS = 252
#: The paper's own annualisation: 252 sessions x 375 one-minute intervals.
PAPER_PERIODS_PER_YEAR = 252 * 375


@dataclass(frozen=True)
class TreeConfig:
    """Everything that changes a fit or a fill.

    ``signal_lag``
        The paper's *"buying and selling signals were adjusted forward by one
        step to accommodate the trading delay"*. Note what that does: the model
        is trained to predict the return of bar ``t+1`` and the position is then
        held over ``t+2``. The prediction and the position are one bar apart, so
        the paper's own execution rule applies the edge to the wrong bar.
        ``signal_lag=0`` acts on the bar the model actually predicts, which
        needs a fill at the closing print and is optimistic. Both are reported.

    ``mapping``
        ``"long_flat"`` treats a 0 as flat, ``"long_short"`` as a short.

    ``session_minutes``
        A ``(start, end)`` window in UTC minutes to restrict trading to, with a
        forced flat outside it - the closest thing to the paper's "intraday"
        constraint on an instrument that trades around the clock. ``None``
        trades every bar.
    """

    interval: str = "1m"
    max_depth: int = 4
    criterion: str = "gini"
    min_samples_leaf: int = 1
    random_state: int = 0
    rsi_period: int = 14
    adx_period: int = 14
    sma_period: int = 14
    vol_period: int = 14
    long_return: int = 15
    vwap_period: int = 14
    signal_lag: int = 1
    mapping: str = "long_flat"
    session_minutes: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.mapping not in ("long_flat", "long_short"):
            raise ValueError(f"unknown mapping {self.mapping!r}")
        if self.signal_lag < 0:
            raise ValueError("signal_lag cannot be negative - that is lookahead")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _wilder(column: pl.Expr, period: int) -> pl.Expr:
    """Wilder's smoothing, which is an EWMA with alpha = 1/n.

    RSI and ADX are both defined with it, and substituting a simple mean shifts
    both indicators enough to change which side of a split a bar falls on.
    """
    return column.ewm_mean(alpha=1.0 / period, adjust=False)


def compute_features(
    bars: pl.DataFrame, cfg: TreeConfig = TreeConfig()
) -> pl.DataFrame:
    """The nine features and the next-bar label, on an already-loaded frame.

    Split out from :func:`features` so the indicator arithmetic can be pinned
    against inputs whose answer is known by hand, without a parquet corpus in
    the way. :func:`features` is this plus the loading and the warmup.

    Two things here decide whether the whole study is honest, so both are
    explicit. Every feature is computed from data at or before its own bar's
    right edge, and the **label is the only forward-looking column** -
    ``ret_fwd`` is the return from this bar's close to the next one's, which is
    what the position placed at this bar earns. Nothing else is shifted
    backwards, and the label is dropped before any feature matrix is built.
    """
    bars = bars.sort("ts")
    close, high, low = pl.col("close"), pl.col("high"), pl.col("low")
    volume = pl.col("n_ticks").cast(pl.Float64)
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)

    diff = close.diff()
    gain = _wilder(pl.max_horizontal(diff, pl.lit(0.0)), cfg.rsi_period)
    loss = _wilder(pl.max_horizontal(-diff, pl.lit(0.0)), cfg.rsi_period)

    up, down = high - prev_high, prev_low - low
    plus_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    tr = pl.max_horizontal(
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    )

    bars = bars.with_columns(
        ret1=close / prev_close - 1.0,
        ret15=close / close.shift(cfg.long_return) - 1.0,
        sma=close.rolling_mean(cfg.sma_period),
        # RSI has no defined value while every move is a gain; 100 is the limit
        # the ratio approaches, and is what every charting package shows.
        rsi14=pl.when(loss > 0).then(100.0 - 100.0 / (1.0 + gain / loss))
        .otherwise(100.0),
        _plus_dm=_wilder(plus_dm, cfg.adx_period),
        _minus_dm=_wilder(minus_dm, cfg.adx_period),
        _tr=_wilder(tr, cfg.adx_period),
        _typical=(high + low + close) / 3.0,
        _volume=volume,
        utc_min=pl.col("ts_open").dt.hour().cast(pl.Int32) * 60
        + pl.col("ts_open").dt.minute().cast(pl.Int32),
    )

    plus_di = 100.0 * pl.col("_plus_dm") / pl.col("_tr")
    minus_di = 100.0 * pl.col("_minus_dm") / pl.col("_tr")
    di_sum = plus_di + minus_di
    # The guard has to be on the *denominator of DX*, not on true range. The
    # first bar has no previous high, so both directional movements are zero
    # while true range is positive - DX is 0/0 there, and one NaN entering a
    # recursive smoother makes every subsequent value NaN for the rest of the
    # corpus. Wilder's own convention is that no directional movement is a DX
    # of zero.
    dx = pl.when(di_sum > 0).then(
        100.0 * (plus_di - minus_di).abs() / di_sum
    ).otherwise(0.0)

    bars = bars.with_columns(
        adx14=_wilder(dx, cfg.adx_period),
        sma_ratio=pl.col("sma") / close,
        sma_corr=pl.rolling_corr(pl.col("sma"), close, window_size=cfg.sma_period),
        vol14=pl.col("ret1").rolling_std(cfg.vol_period),
        # "the 210-period rolling volatility, which is essentially the 14-period
        # volatility of the 15-period rolling returns" - their own gloss, and the
        # only reading of it that produces a 210-period lookback.
        vol210=pl.col("ret15").rolling_std(cfg.vol_period),
        vwap_ratio=(pl.col("_typical") * pl.col("_volume"))
        .rolling_sum(cfg.vwap_period)
        / pl.col("_volume").rolling_sum(cfg.vwap_period) / close,
    ).with_columns(
        # The label, and the return the position placed on this bar earns. They
        # are the same quantity, which is the point: the classifier's target and
        # the backtest's payoff cannot disagree.
        ret_fwd=close.shift(-1) / close - 1.0,
    ).with_columns(
        label=(pl.col("ret_fwd") > 0).cast(pl.Int8),
    )

    return bars.drop("_plus_dm", "_minus_dm", "_tr", "_typical", "_volume")


def features(
    symbol: str,
    cfg: TreeConfig = TreeConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
) -> pl.DataFrame:
    """Load one instrument's bars and attach the nine features and the label.

    The warmup is read from before the split so the first evaluable bar already
    has a full 210-period lookback behind it, and the rows that exist only to
    warm the indicators are dropped before returning - they are not evaluable
    and must never reach a fit.
    """
    warmup = timedelta(days=10)
    bars = load_bars(
        symbol, cfg.interval, split=split, start=start, end=end, warmup=warmup,
        allow_test=allow_test,
        columns=["ts", "ts_open", "open", "high", "low", "close",
                 "spread_mean", "n_ticks"],
    ).sort("ts")
    warm_from = None
    if "is_warmup" in bars.columns:
        warm_from = bars.filter(~pl.col("is_warmup"))["ts_open"].min()
        bars = bars.drop("is_warmup")

    out = compute_features(bars, cfg)
    if warm_from is not None:
        out = out.filter(pl.col("ts_open") >= warm_from)
    return out


def clean(frame: pl.DataFrame, features_used: Sequence[str] = FEATURES) -> pl.DataFrame:
    """Rows where every feature and the label are finite.

    ``drop_nulls`` alone is not enough: ``ret15`` divides by a price and
    ``sma_corr`` by a standard deviation, and a flat 14-bar stretch in a thin
    session makes the latter ``NaN`` rather than null.
    """
    out = frame.drop_nulls(list(features_used) + ["label", "ret_fwd"])
    finite = pl.all_horizontal(
        [pl.col(c).is_finite() for c in list(features_used) + ["ret_fwd"]]
    )
    return out.filter(finite)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit(
    train: pl.DataFrame,
    cfg: TreeConfig = TreeConfig(),
    features_used: Sequence[str] = FEATURES,
) -> DecisionTreeClassifier:
    """Fit one tree on one instrument's training rows."""
    model = DecisionTreeClassifier(
        max_depth=cfg.max_depth,
        criterion=cfg.criterion,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
    )
    model.fit(train.select(features_used).to_numpy(), train["label"].to_numpy())
    return model


def describe_tree(
    model: DecisionTreeClassifier, features_used: Sequence[str] = FEATURES
) -> str:
    """The fitted rules as text - the paper's Figures 4 and 5.

    Interpretability is the paper's stated reason for choosing trees over
    anything else, so the rules belong in the report rather than only in a
    metric.
    """
    return export_text(model, feature_names=list(features_used), decimals=6)


def used_features(
    model: DecisionTreeClassifier, features_used: Sequence[str] = FEATURES
) -> list[str]:
    """The features the fitted tree actually splits on.

    The paper's claim that *"most models utilize approximately five to seven
    indicators"* out of nine, and that the set differs by instrument, is exactly
    this list - so it is computed rather than asserted.
    """
    return [
        name for name, imp in zip(features_used, model.feature_importances_)
        if imp > 0
    ]


# --------------------------------------------------------------------------
# From prediction to a return series
# --------------------------------------------------------------------------

def positions(
    model: DecisionTreeClassifier,
    frame: pl.DataFrame,
    cfg: TreeConfig = TreeConfig(),
    features_used: Sequence[str] = FEATURES,
) -> pl.DataFrame:
    """Predictions turned into a held position, bar by bar.

    ``position`` is what is held over the bar whose ``ret_fwd`` is on the same
    row. The lag is applied to the *prediction*, so ``signal_lag=1`` reproduces
    the paper's forward shift exactly, and the accuracy column keeps measuring
    the model against the bar it was trained on rather than the one it is traded
    on - which is how the gap between the two becomes visible.
    """
    if frame.is_empty():
        return frame
    x = frame.select(features_used).to_numpy()
    pred = model.predict(x).astype(np.int8)
    out = frame.with_columns(pred=pl.Series("pred", pred, dtype=pl.Int8))

    signal = pl.col("pred").shift(cfg.signal_lag)
    raw = (
        pl.when(signal == 1).then(1.0).otherwise(0.0)
        if cfg.mapping == "long_flat"
        else pl.when(signal == 1).then(1.0).otherwise(-1.0)
    )
    if cfg.session_minutes is not None:
        lo, hi = cfg.session_minutes
        raw = pl.when((pl.col("utc_min") >= lo) & (pl.col("utc_min") < hi)).then(
            raw
        ).otherwise(0.0)

    return out.with_columns(position=raw.fill_null(0.0)).with_columns(
        # Turnover is the notional traded to get from the last position to this
        # one. A long/short flip is 2.0, and it costs twice as much as an entry.
        turnover=(pl.col("position") - pl.col("position").shift(1).fill_null(0.0))
        .abs()
    )


def returns(
    frame: pl.DataFrame,
    symbol: str,
    *,
    costs: CostModel | None = None,
    split: str | None = None,
) -> pl.DataFrame:
    """Per-bar gross and net returns in bps, and the cost that separates them.

    Cost is charged on turnover at the measured half-spread plus half the
    commission plus one side of slippage - ``side_cost_pips`` from the cost
    model, which is the same arithmetic every other strategy in this project
    pays, so the comparison across strategies holds.
    """
    if frame.is_empty():
        return frame
    spec = get_spec(symbol)
    costs = costs or CostModel.from_profiles(symbol, split=split or "dev")
    priced = costs.with_costs(frame)
    return priced.with_columns(
        gross_bps=1e4 * pl.col("position") * pl.col("ret_fwd"),
        cost_bps=pl.col("turnover") * pl.col("side_cost_pips") * spec.pip
        / pl.col("close") * 1e4,
    ).with_columns(net_bps=pl.col("gross_bps") - pl.col("cost_bps"))


def daily(frame: pl.DataFrame) -> pl.DataFrame:
    """Minute returns summed into calendar days.

    Summing within a day and compounding across days is the standard treatment
    of an intraday book: the position is a notional weight that is not
    reinvested inside the session, and compounding minute by minute would
    attribute growth to a rebalance that never happened.
    """
    if frame.is_empty():
        return frame
    return (
        frame.with_columns(date=pl.col("ts").dt.date())
        .group_by("date")
        .agg(
            gross_bps=pl.col("gross_bps").sum(),
            net_bps=pl.col("net_bps").sum(),
            cost_bps=pl.col("cost_bps").sum(),
            turnover=pl.col("turnover").sum(),
            bars=pl.len(),
            exposure=pl.col("position").abs().mean(),
            bench_bps=1e4 * pl.col("ret_fwd").sum(),
        )
        .sort("date")
    )


def breakeven_cost_bps(frame: pl.DataFrame) -> float:
    """Round-turn cost, in bps of notional, at which the gross edge vanishes.

    Total gross return divided by total turnover, doubled to state it as a round
    turn. It is the single number that decides this strategy, because the paper
    charged nothing and this says what nothing was worth.
    """
    if frame.is_empty():
        return float("nan")
    turn = float(frame["turnover"].sum())
    if turn <= 0:
        return float("nan")
    return 2.0 * float(frame["gross_bps"].sum()) / turn


def paper_style_sharpe(frame: pl.DataFrame, *, column: str = "net_bps") -> float:
    """The Sharpe the paper would have printed: per-minute, x sqrt(94,500).

    Reported only so that their 5.81 has something on the same scale to be read
    against. It is not comparable with any other Sharpe in this project.
    """
    r = frame[column].drop_nulls().to_numpy() / 1e4
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) <= 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PAPER_PERIODS_PER_YEAR))


def benchmark(frame: pl.DataFrame) -> pl.DataFrame:
    """Buy-and-hold on the same bars - the paper's comparison.

    Held continuously, so it pays one round turn over the whole period rather
    than per bar, which is why its cost column is negligible and the strategy's
    is not. That contrast is the finding, not an artefact.
    """
    if frame.is_empty():
        return frame
    return frame.with_columns(
        position=pl.lit(1.0),
        turnover=pl.when(pl.int_range(pl.len()) == 0).then(1.0).otherwise(0.0),
    )


def accuracy(frame: pl.DataFrame) -> dict:
    """Directional accuracy, and the base rate it has to beat.

    A classifier on next-bar sign is measured against the majority class, not
    against 50%: on a drifting instrument the up-bar share is not a half, and a
    tree that learned only the drift would post an accuracy above chance while
    predicting one constant.
    """
    if frame.is_empty() or "pred" not in frame.columns:
        return {"n": 0}
    y = frame["label"].to_numpy()
    p = frame["pred"].to_numpy()
    base = max(float(y.mean()), 1.0 - float(y.mean()))
    return {
        "n": int(y.size),
        "accuracy": float((y == p).mean()),
        "base_rate": base,
        "up_share": float(y.mean()),
        "predicted_up_share": float(p.mean()),
        "edge_vs_base": float((y == p).mean()) - base,
    }
