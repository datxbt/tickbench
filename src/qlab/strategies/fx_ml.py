"""Enkhbayar and Slepaczuk (2024): ML trading signals on major FX pairs.

Subject: *Predictive Modeling of Foreign Exchange Trading Signals Using Machine
Learning Techniques* (SSRN 4862571). Eight regressors predict the next bar's
return from technical indicators; the prediction is thresholded into buy / hold /
sell; the position is held until the signal changes. Everything is refitted in a
rolling walk-forward. An EMA-crossover trend follower, itself walk-forward
optimised, is the traditional comparison, and buy-and-hold is the benchmark.

The paper's own verdict is mostly negative - its Table 10 reports annualised
Sharpe ratios that are *negative* for most models, against 0.22 for holding
EURUSD - so replication here is not asking "does this work" but "does it fail
here for the same reasons". Two design choices in the paper look like sufficient
causes, and both are made switchable so the replication can attribute the
failure rather than merely reproduce it:

**Non-stationary features.** Table 3 lists EMA10-200, MACD, Bollinger bands and
average price as features, and lists them as *levels*. A tree cannot extrapolate
past the range it was trained on, so a model fitted on 600 days of EURUSD at
1.05-1.12 and asked about 1.18 predicts from the boundary of its training box.
``feature_mode="paper"`` reproduces that; ``"stationary"`` divides every price-
scaled feature by close and leaves the oscillators alone.

**Thresholds from the target's quartiles.** Section 3.7 compares the *predicted*
return to the first and third quartile of the *training set*. A regressor fitted
with MAE loss on a near-unpredictable target shrinks hard toward the mean, so its
predictions live inside a band far narrower than the realised returns whose
quartiles are the gate. ``threshold_mode="paper"`` reproduces that;
``"prediction"`` takes the quartiles of the training-set predictions instead,
which is what actually delivers a balanced signal.

Position convention follows Section 3.8: a signal formed on the close of bar *t*
earns the return from *t* to *t+1*, and a position is held until a different
signal is given, so turnover is charged on ``|position change|``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Walk-forward geometry (paper Table 7)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForward:
    """Rolling train / validation / test windows, advancing by one test length.

    The paper's Table 7 fixes 68 / 18 / 14 percent: 600 / 156 / 126 bars daily,
    3600 / 936 / 756 on 4-hour data. Those are absolute counts, not fractions of
    whatever data is present, so a shorter corpus yields fewer windows rather
    than shorter ones - which is the honest failure mode.
    """

    train: int
    validation: int
    test: int

    @classmethod
    def for_frequency(cls, freq: str) -> "WalkForward":
        if freq == "1d":
            return cls(train=600, validation=156, test=126)
        if freq == "4h":
            return cls(train=3600, validation=936, test=756)
        raise ValueError(f"paper specifies windows for 1d and 4h only, got {freq!r}")

    @property
    def span(self) -> int:
        return self.train + self.validation + self.test

    def windows(self, n: int) -> list[tuple[slice, slice, slice]]:
        """Every ``(train, validation, test)`` slice triple that fits in ``n`` rows."""
        out = []
        start = 0
        while start + self.span <= n:
            a = start + self.train
            b = a + self.validation
            c = b + self.test
            out.append((slice(start, a), slice(a, b), slice(b, c)))
            start += self.test
        return out


# --------------------------------------------------------------------------
# Features (paper Table 3)
# --------------------------------------------------------------------------


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _true_range(high, low, close) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def _atr(high, low, close, n: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(alpha=1 / n, adjust=False).mean()


def _adx(high, low, close, n: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, n)
    plus_di = 100 * pd.Series(plus, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def _stochastic(high, low, close, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo, hi = low.rolling(n).min(), high.rolling(n).max()
    k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def _cci(high, low, close, n: int = 20, c: float = 0.015) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (tp - sma) / (c * mad.replace(0, np.nan))


def _williams_r(high, low, close, n: int = 14) -> pd.Series:
    hi, lo = high.rolling(n).max(), low.rolling(n).min()
    return -100 * (hi - close) / (hi - lo).replace(0, np.nan)


#: Features whose units are price, and which ``feature_mode="stationary"``
#: rescales by close. Everything else is already a bounded oscillator.
PRICE_SCALED = (
    "ema10", "ema20", "ema50", "ema100", "ema200",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "atr14",
    "avg_price", "range", "open", "high", "low", "close",
)


def build_features(bars: pd.DataFrame, *, mode: str = "paper") -> pd.DataFrame:
    """Table 3's feature set, from an OHLC frame indexed by timestamp.

    ``mode="paper"`` returns the indicators as levels, exactly as tabulated.
    ``mode="stationary"`` divides every price-scaled column by that bar's close
    (and the oscillators, already bounded, pass through untouched), which is the
    only change needed to let a tree see a price it was not trained on.
    """
    if mode not in ("paper", "stationary"):
        raise ValueError(f"mode must be paper/stationary, got {mode!r}")

    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    f = pd.DataFrame(index=bars.index)

    for n in (10, 20, 50, 100, 200):
        f[f"ema{n}"] = _ema(c, n)

    macd = _ema(c, 12) - _ema(c, 26)
    f["macd"] = macd
    f["macd_signal"] = _ema(macd, 9)
    f["macd_hist"] = macd - f["macd_signal"]

    f["adx14"] = _adx(h, l, c, 14)
    f["rsi14"] = _rsi(c, 14)
    k, d = _stochastic(h, l, c, 14, 3)
    f["stoch_k"], f["stoch_d"] = k, d
    f["cci20"] = _cci(h, l, c, 20, 0.015)
    f["williams_r14"] = _williams_r(h, l, c, 14)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    f["bb_mid"], f["bb_upper"], f["bb_lower"] = mid, mid + 2 * sd, mid - 2 * sd
    f["atr14"] = _atr(h, l, c, 14)

    # Statistical indicators, all at lag 1 per Table 3.
    f["momentum"] = c.diff(1)
    f["avg_price"] = (o + h + l + c) / 4
    f["range"] = h - l
    f["open"], f["high"], f["low"], f["close"] = o, h, l, c

    if mode == "stationary":
        for col in PRICE_SCALED:
            if col in f:
                f[col] = f[col] / c
        # A ratio of close to itself is the constant 1; replace it with the
        # thing the level was standing in for.
        f["close"] = c.pct_change()
        f["momentum"] = c.pct_change()

    return f


# --------------------------------------------------------------------------
# Models (paper Section 3.4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One estimator plus the grid the paper's Table 8 searches over."""

    name: str
    build: Callable[..., object]
    grid: Sequence[dict]
    needs_scaling: bool = False
    is_sequence: bool = False


def _sk(name: str):
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    return {
        "ridge": Ridge,
        "knn": KNeighborsRegressor,
        "rf": RandomForestRegressor,
        "gbdt": GradientBoostingRegressor,
        "ann": MLPRegressor,
    }[name]


def model_registry(seed: int = 0) -> dict[str, ModelSpec]:
    """The eight models of Table 2, with compact hyperparameter grids.

    The grids are smaller than the paper's RandomizedSearchCV ranges. That is a
    deliberate trade: the selection here is an explicit fit-on-train,
    score-on-validation pass rather than cross-validation inside the training
    window, which is what Section 3.6 actually describes, and a wider grid on
    600 rows buys variance rather than accuracy.
    """
    import xgboost as xgb

    return {
        "ridge": ModelSpec(
            "ridge", lambda **kw: _sk("ridge")(**kw),
            [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
            needs_scaling=True,
        ),
        "knn": ModelSpec(
            "knn", lambda **kw: _sk("knn")(**kw),
            [{"n_neighbors": k, "weights": w} for k in (5, 15, 50) for w in ("uniform", "distance")],
            needs_scaling=True,
        ),
        "rf": ModelSpec(
            "rf",
            lambda **kw: _sk("rf")(random_state=seed, n_jobs=-1, **kw),
            [{"n_estimators": 200, "max_depth": d, "min_samples_leaf": m}
             for d in (3, 6, None) for m in (1, 20)],
        ),
        "xgboost": ModelSpec(
            "xgboost",
            lambda **kw: xgb.XGBRegressor(
                random_state=seed, n_jobs=-1, objective="reg:absoluteerror",
                verbosity=0, **kw),
            [{"n_estimators": 200, "max_depth": d, "learning_rate": lr, "subsample": 0.8}
             for d in (2, 4, 6) for lr in (0.03, 0.1)],
        ),
        "gbdt": ModelSpec(
            "gbdt",
            lambda **kw: _sk("gbdt")(random_state=seed, loss="absolute_error", **kw),
            [{"n_estimators": 150, "max_depth": d, "learning_rate": lr}
             for d in (2, 3) for lr in (0.03, 0.1)],
        ),
        "ann": ModelSpec(
            "ann",
            lambda **kw: _sk("ann")(random_state=seed, max_iter=400, early_stopping=True, **kw),
            [{"hidden_layer_sizes": hl, "alpha": a}
             for hl in ((32,), (64, 32)) for a in (1e-4, 1e-2)],
            needs_scaling=True,
        ),
        "lstm": ModelSpec(
            "lstm", lambda **kw: _RecurrentRegressor(kind="lstm", seed=seed, **kw),
            [{"hidden": h, "lr": 1e-3, "epochs": 30} for h in (16, 32)],
            needs_scaling=True, is_sequence=True,
        ),
        "gru": ModelSpec(
            "gru", lambda **kw: _RecurrentRegressor(kind="gru", seed=seed, **kw),
            [{"hidden": h, "lr": 1e-3, "epochs": 30} for h in (16, 32)],
            needs_scaling=True, is_sequence=True,
        ),
    }


class _RecurrentRegressor:
    """A minimal LSTM / GRU regressor with the sklearn fit/predict surface.

    Sequence length is fixed at 10 bars: the paper gives no length, and 10 is
    both the shortest EMA in the feature set and short enough that a 600-row
    training window still holds 590 sequences.
    """

    def __init__(self, *, kind: str, hidden: int = 32, lr: float = 1e-3,
                 epochs: int = 30, seq_len: int = 10, seed: int = 0):
        self.kind, self.hidden, self.lr = kind, hidden, lr
        self.epochs, self.seq_len, self.seed = epochs, seq_len, seed
        self.net = None

    def _sequences(self, X: np.ndarray) -> np.ndarray:
        n, k = X.shape
        pad = np.repeat(X[:1], self.seq_len - 1, axis=0)
        padded = np.vstack([pad, X])
        return np.stack([padded[i : i + self.seq_len] for i in range(n)])

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        seq = torch.from_numpy(self._sequences(X))
        target = torch.from_numpy(y).unsqueeze(1)

        cell = nn.LSTM if self.kind == "lstm" else nn.GRU

        class Net(nn.Module):
            def __init__(self, k, hidden):
                super().__init__()
                self.rnn = cell(k, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        self.net = Net(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()  # MAE, per Section 3.1
        batch = 64
        for _ in range(self.epochs):
            perm = torch.randperm(len(seq))
            for i in range(0, len(seq), batch):
                idx = perm[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net(seq[idx]), target[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            return self.net(torch.from_numpy(self._sequences(X))).squeeze(1).numpy()


# --------------------------------------------------------------------------
# Signal transform (paper Section 3.7)
# --------------------------------------------------------------------------


def to_signal(pred: np.ndarray, *, mode: str, lo: float, hi: float) -> np.ndarray:
    """Eqs (49-51): predicted return -> ``{-1, 0, 1}``.

    ``lo`` / ``hi`` are the gates. Which quartiles those are depends on
    ``threshold_mode`` at the call site; only-buy and only-sell use zero, as the
    paper specifies, and ignore them.
    """
    if mode == "buy_sell":
        return np.where(pred >= hi, 1.0, np.where(pred <= lo, -1.0, 0.0))
    if mode == "only_buy":
        return np.where(pred > 0, 1.0, 0.0)
    if mode == "only_sell":
        return np.where(pred < 0, -1.0, 0.0)
    raise ValueError(f"unknown signal mode {mode!r}")


SIGNAL_MODES = ("buy_sell", "only_buy", "only_sell")


# --------------------------------------------------------------------------
# Trend following (paper Section 3.5)
# --------------------------------------------------------------------------

EMA_SPANS = (10, 20, 50, 100, 200)
EMA_PAIRS = tuple((f, s) for i, f in enumerate(EMA_SPANS) for s in EMA_SPANS[i + 1 :])


def ema_cross_signal(close: pd.Series, fast: int, slow: int, mode: str) -> np.ndarray:
    """Eqs (52-54). Computed on closes through ``t``, applied to ``t -> t+1``."""
    f, s = _ema(close, fast), _ema(close, slow)
    raw = np.where(f > s, 1.0, np.where(f < s, -1.0, 0.0))
    if mode == "only_buy":
        return np.maximum(raw, 0.0)
    if mode == "only_sell":
        return np.minimum(raw, 0.0)
    return raw


# --------------------------------------------------------------------------
# Backtest (paper Section 3.8)
# --------------------------------------------------------------------------


def backtest(
    signal: np.ndarray,
    forward_return: np.ndarray,
    *,
    cost: float = 0.0002,
    prev_position: float = 0.0,
) -> np.ndarray:
    """Net per-bar returns.

    ``signal[i]`` is formed on bar *i*'s close and earns ``forward_return[i]``,
    the return from *i* to *i+1*. Cost is charged on the change in position, so
    holding through an unchanged signal is free - which is Section 3.8's rule
    that a position runs until a different signal arrives.
    """
    signal = np.asarray(signal, dtype=float)
    fwd = np.asarray(forward_return, dtype=float)
    turnover = np.abs(np.diff(signal, prepend=prev_position))
    return signal * fwd - cost * turnover


def annualised_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


PERIODS_PER_YEAR = {"1d": 252.0, "4h": 252.0 * 6}


# --------------------------------------------------------------------------
# The walk-forward driver
# --------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions and diagnostics, concatenated across windows."""

    index: pd.DatetimeIndex
    prediction: np.ndarray
    forward_return: np.ndarray
    close: np.ndarray
    train_q1: np.ndarray
    train_q3: np.ndarray
    window_id: np.ndarray
    chosen: list[dict] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)


def _fit_select(spec: ModelSpec, Xtr, ytr, Xva, yva):
    """Fit every grid point on train, keep the one with the lowest validation MAE."""
    best, best_mae, best_params = None, np.inf, {}
    for params in spec.grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = spec.build(**params)
                model.fit(Xtr, ytr)
                mae = float(np.mean(np.abs(model.predict(Xva) - yva)))
        except Exception:
            continue
        if np.isfinite(mae) and mae < best_mae:
            best, best_mae, best_params = model, mae, params
    if best is None:
        raise RuntimeError(f"no grid point of {spec.name} fitted")
    return best, best_mae, best_params


def run_walkforward(
    bars: pd.DataFrame,
    spec: ModelSpec,
    wf: WalkForward,
    *,
    feature_mode: str = "paper",
    refit_on_validation: bool = False,
) -> WalkForwardResult:
    """Section 3.6, one model, one instrument, one frequency.

    ``bars`` must be indexed by timestamp with ``open/high/low/close`` columns and
    must already include whatever warmup the longest feature needs; rows whose
    features are not yet warm are dropped before the windows are cut, so no
    window ever contains a partially-formed EMA200.
    """
    from sklearn.preprocessing import StandardScaler

    feats = build_features(bars, mode=feature_mode)
    target = bars["close"].pct_change().shift(-1)  # Eq (1): t -> t+1

    frame = feats.copy()
    frame["_y"] = target
    frame["_close"] = bars["close"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    X = frame.drop(columns=["_y", "_close"]).to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    close = frame["_close"].to_numpy(dtype=float)
    index = frame.index

    windows = wf.windows(len(frame))
    if not windows:
        raise ValueError(
            f"{len(frame)} usable bars is fewer than one {wf.span}-bar window"
        )

    preds, fwd, idx, q1s, q3s, wid, cl = [], [], [], [], [], [], []
    result = WalkForwardResult(
        index=pd.DatetimeIndex([]), prediction=np.array([]), forward_return=np.array([]),
        close=np.array([]), train_q1=np.array([]), train_q3=np.array([]),
        window_id=np.array([]),
    )

    for w, (tr, va, te) in enumerate(windows):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Xte = X[te]

        if spec.needs_scaling:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        model, mae, params = _fit_select(spec, Xtr, ytr, Xva, yva)
        if refit_on_validation:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = spec.build(**params).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))

        p = np.asarray(model.predict(Xte), dtype=float)
        preds.append(p)
        fwd.append(y[te])
        cl.append(close[te])
        idx.append(index[te])
        q1s.append(np.full(len(p), np.quantile(ytr, 0.25)))
        q3s.append(np.full(len(p), np.quantile(ytr, 0.75)))
        wid.append(np.full(len(p), w))
        result.chosen.append(params)
        result.val_mae.append(mae)

    result.index = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    result.prediction = np.concatenate(preds)
    result.forward_return = np.concatenate(fwd)
    result.close = np.concatenate(cl)
    result.train_q1 = np.concatenate(q1s)
    result.train_q3 = np.concatenate(q3s)
    result.window_id = np.concatenate(wid)
    return result


def signal_from_result(
    res: WalkForwardResult, *, signal_mode: str, threshold_mode: str = "paper"
) -> np.ndarray:
    """Apply Section 3.7's gate, per window, to a walk-forward result."""
    if threshold_mode == "paper":
        lo, hi = res.train_q1, res.train_q3
    elif threshold_mode == "prediction":
        lo = np.empty_like(res.prediction)
        hi = np.empty_like(res.prediction)
        for w in np.unique(res.window_id):
            m = res.window_id == w
            lo[m] = np.quantile(res.prediction[m], 0.25)
            hi[m] = np.quantile(res.prediction[m], 0.75)
    else:
        raise ValueError(f"threshold_mode must be paper/prediction, got {threshold_mode!r}")
    return to_signal(res.prediction, mode=signal_mode, lo=lo, hi=hi)


def ema_cross_walkforward(
    bars: pd.DataFrame, wf: WalkForward, *, signal_mode: str,
    cost: float = 0.0002, periods_per_year: float = 252.0,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Table 6: pick the EMA pair with the best in-window Sharpe, apply it to test.

    The paper optimises on the training period; the validation block is left
    unused by the trend follower, so it is folded into the selection sample here
    rather than discarded - a choice that can only help the trend follower.
    """
    close = bars["close"]
    fwd = close.pct_change().shift(-1).to_numpy()
    usable = np.isfinite(fwd)
    close_v = close.to_numpy()

    signals = {p: ema_cross_signal(close, *p, signal_mode) for p in EMA_PAIRS}

    idx, sig, ret = [], [], []
    for tr, va, te in wf.windows(len(bars)):
        fit = slice(tr.start, va.stop)
        best, best_sr = None, -np.inf
        for pair, s in signals.items():
            r = backtest(s[fit], np.nan_to_num(fwd[fit]), cost=cost)
            sr = annualised_sharpe(r, periods_per_year)
            if np.isfinite(sr) and sr > best_sr:
                best, best_sr = pair, sr
        s_te = signals[best][te]
        idx.append(bars.index[te])
        sig.append(s_te)
        ret.append(backtest(s_te, np.nan_to_num(fwd[te]), cost=cost))

    joined = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    return joined, np.concatenate(sig), np.concatenate(ret)
