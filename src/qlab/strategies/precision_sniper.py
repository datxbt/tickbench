"""Precision Sniper (WillyAlgoTrader, Pine v6 v1.4.0) as a testable strategy.

The source is a TradingView *indicator*: it scores ten confluence factors,
fires on an EMA cross when the score clears a preset threshold, and draws a
stop and three targets. It never places an order, so every execution choice
below was made explicitly and swept rather than assumed.

The engine
----------
A signal is an ``ta.crossover(emaFast, emaSlow)`` (mirrored for shorts) that
also has price on the right side of both EMAs, RSI out of the opposite
extreme, and a confluence score at or above the preset's threshold. The score
sums ten factors; the script scales both the score and its A+/A/B grades by
the factors actually available, which matters here - see *Volume* below.

Presets bundle seven parameters at once (both EMA lengths, the trend EMA, RSI
and ATR lengths, the minimum score, and the stop multiple). ``Auto`` is not a
seventh setting, it is a timeframe-to-preset map, so the six named bundles are
the parameter axis and ``Auto`` is read off the result afterwards. ``Custom``
is the input defaults, which are identical to ``Default``.

Volume, and what this feed can say
----------------------------------
Two of the ten factors - the volume surge and the VWAP side - need volume.
This is a quote feed with no size, so ``symHasVol`` is false for every bar of
all 696M ticks, and the script's own adaptive scale applies: **max score 8.0,
not 10.0**, and the min-score threshold rescales with it (``Default`` 5 -> 4.0).
That is the script's documented behaviour on a volumeless symbol, so running
it here is a faithful evaluation of the engine as written - but it is an
evaluation of eight of its ten factors, and the write-up says so.

One further consequence is structural rather than incidental: of the eight
surviving factors, **1.5 points are unconditional on any signal bar**. A buy
needs ``ta.crossover(emaFast, emaSlow)``, which awards the 1.0 for
``emaFast > emaSlow``, and needs ``close > emaFast``, which awards the 0.5.
The score of a signal therefore ranges over [1.5, 8.0], not [0, 8.0], and the
B grade (>= 50% of max = 4.0) is only 2.5 discriminating points above the
floor. The volume factor is also added to *both* sides' scores, so on a feed
that had it, it would carry no directional information at all.

What the lastDirection gate actually is
--------------------------------------
The script gates signals on ``lastDirection``: set by a fill, cleared by
``slEvent`` and by a TP3 full exit, and overwritten by an opposing fill. Read
state by state, ``lastDirection != d`` is

* flat (stopped out, or closed at TP3) -> takes the signal;
* a position open, same direction -> blocks it, so the script never pyramids;
* a position open, **opposite** direction -> takes it, and the open trade is
  closed by the reversal. The script's own backtest tracker has this as its
  Case 2, valuing the closed trade at the opposing signal's price.

So the account is **stop-and-reverse with no pyramiding**, not
one-position-at-a-time: the two agree except that a reversal cuts a trade
short of its barrier. That difference is the script's, not an artifact, so it
is priced rather than assumed away - :func:`sequence_reverse` walks it, and
:func:`sequence` keeps the flat account for the single-target exits, which are
this study's constructions and have no reversal rule of their own. Both are
reported.

What the gate does **not** do is couple the entry stream to the exit rule.
Whether a signal is takeable depends only on the position state, and the
position state is a function of the fills and the exit rule applied afterwards.
Signals can therefore be generated once per (symbol, timeframe, preset),
resolved once against the tape, and sequenced afterwards per exit rule - which
is why a forty-four cell exit grid costs one tick walk.

Exits
-----
The script nominates one exit (three R-multiple targets with a stop that steps
to breakeven at TP1, to TP1 at TP2, to TP2 at TP3, and a full close at TP3),
so that rule is simulated on the tape exactly. It is not treated as the answer:
single targets from 0.5R to 4R and a stop-or-clock hold are priced on the same
fill, and the nominated rule is one point on that curve.

Stops are the other half. ``atr`` is the plain ATR stop; ``struct`` is the
script's structure stop - the **wider** of the ATR stop and the swing extreme
plus a 0.2 ATR buffer, capped at 1.5x the ATR distance and floored at 0.5 ATR.
The ``_w`` suffixes apply the High-volatility widening (1.5x) on bars whose
regime is High, which is the ``Widen SL`` filter mode; on every other bar they
are identical to their unsuffixed twin, and the report says how often that is.

The third volatility mode, ``Skip Signals``, is a filter on the signal's own
recorded regime and costs nothing to evaluate. So are the grade filter, the
C-grade switch, the minimum score and the HTF bias, all of which are recorded
per signal and applied to the resolved tape afterwards.

Deliberate departures from the Pine source
------------------------------------------
* **Intrabar order is taken from ticks, not assumed.** The script resolves a
  bar that touches both a target and the stop by marking the target first and
  checking the stop against the pre-update trail, which books a 0R breakeven
  where a tick-ordered path may book -1R. Here the tape decides.
* **A life cap.** The script holds until the stop or an opposing signal, with
  no clock. Positions here die after ``max_hold_bars`` bars of their own
  timeframe; the time-exit rate is reported for every cell.
* **Fills.** Entries are market orders at the standing quote at the bar's
  close - the last tick at or before it, because the feed prints only on
  change - and pay measured slippage. Targets are limits and fill at their
  price or not at all. Stops are market and pay slippage.
* **HTF bias.** ``request.security(tf, expr[1], lookahead_on)`` is the
  non-repainting idiom: during HTF bar *j* it returns the value from bar
  *j-1*. Implemented as an as-of join of the chart bar's open against the HTF
  bar's close time, which is the same quantity and cannot look ahead.
* **No ``ta.vwap`` fallback.** Where the script would score a VWAP factor it
  cannot compute, it drops the factor from the scale; this does the same
  rather than substituting a proxy.

Timeframes follow the house convention: to an hour the grid is UTC, and from
2h up bars are cut on the 17:00 New York day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import SPLITS, load_bars
from ..symbols import get_spec
from .engulfing_quadrant import _first_cross, _taken_indices
from .sweep_orderblock import (
    INTERVAL_MINUTES,
    US,
    _TickCache,
    atr_rma,
    build_bars,
    roll_instants,
)

ALL_INTERVALS: tuple[str, ...] = tuple(INTERVAL_MINUTES)


# --------------------------------------------------------------------------
# Presets - the script's seven-parameter bundles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    """One column of the script's ``switch resolvedPreset`` tables."""

    name: str
    ema_fast: int
    ema_slow: int
    ema_trend: int
    rsi_len: int
    atr_len: int
    min_score: float   # on the script's 10-point scale; rescaled by max_score
    sl_mult: float


PRESETS: dict[str, Preset] = {
    "Scalping":     Preset("Scalping",      5, 13, 34,  8, 10, 4.0, 0.8),
    "Aggressive":   Preset("Aggressive",    8, 18, 50, 11, 12, 3.0, 1.2),
    "Default":      Preset("Default",       9, 21, 55, 13, 14, 5.0, 1.5),
    "Conservative": Preset("Conservative", 12, 26, 89, 14, 14, 7.0, 2.0),
    "Swing":        Preset("Swing",        13, 34, 89, 21, 20, 6.0, 2.5),
    "Crypto 24/7":  Preset("Crypto 24/7",   9, 21, 55, 14, 20, 5.0, 2.0),
}
ALL_PRESETS: tuple[str, ...] = tuple(PRESETS)


def auto_preset(interval: str) -> str:
    """The script's ``Auto`` map: timeframe minutes to a preset name."""
    m = INTERVAL_MINUTES[interval]
    if m <= 5:
        return "Scalping"
    if m <= 60:
        return "Default"
    if m < 240:
        return "Conservative"
    return "Swing"


# The tooltip's recommendation - 1H for 5-15m, 4H for 1H, D for 4H - extended
# to the whole ladder in the same spirit (four to twelve chart bars per HTF
# bar) and capped at 1d, the coarsest bar the corpus has. ``none`` is the
# script's empty-string setting, which makes the HTF factor the chart itself.
HTF_FOR: dict[str, str] = {
    "1m": "15m", "2m": "30m", "3m": "30m", "5m": "1h", "10m": "1h",
    "15m": "1h", "20m": "4h", "30m": "4h", "1h": "4h", "2h": "1d",
    "4h": "1d", "1d": "1d",
}

GRADE_APLUS_R = 0.80
GRADE_A_R = 0.65
GRADE_B_R = 0.50


def grade(score: float, max_score: float) -> str:
    """The script's ``getGrade``: a band of the score's ratio to the max."""
    r = score / max_score if max_score > 0 else 0.0
    if r >= GRADE_APLUS_R:
        return "A+"
    if r >= GRADE_A_R:
        return "A"
    if r >= GRADE_B_R:
        return "B"
    return "C"


# --------------------------------------------------------------------------
# The exit grid
# --------------------------------------------------------------------------

STOP_KEYS: tuple[str, ...] = ("atr", "atr_w", "struct", "struct_w")

TARGET_R: dict[str, float] = {
    "t05": 0.5, "t1": 1.0, "t15": 1.5, "t2": 2.0, "t3": 3.0, "t4": 4.0,
}
HOLD = "hold"

# The script's own rule: a three-rung ladder with the stop stepping one rung
# behind it. ``p`` closes the whole position at TP3 (the v1.4.0 default);
# ``pr`` is the v1.2.x runner it replaced, where TP3 only moves the trail.
LADDERS: dict[str, tuple[float, float, float]] = {
    "L123": (1.0, 2.0, 3.0),    # the script's defaults
    "L112": (1.0, 1.5, 2.0),
    "L345": (1.5, 3.0, 4.5),
}
PINE_KEYS: tuple[str, ...] = (*(f"p{k}" for k in LADDERS), "prL123")
TARGET_KEYS: tuple[str, ...] = (*TARGET_R, HOLD, *PINE_KEYS)
EXIT_KEYS: tuple[str, ...] = tuple(
    f"{s}_{t}" for s in STOP_KEYS for t in TARGET_KEYS
)
# Every R level any exit needs a crossing time for.
R_LEVELS: tuple[float, ...] = tuple(sorted(
    set(TARGET_R.values()) | {r for lad in LADDERS.values() for r in lad}
))
# Exits that can run long enough for overnight swap to matter.
ROLL_KEYS: tuple[str, ...] = (HOLD, *PINE_KEYS)

FORWARD_BARS: tuple[int, ...] = (1, 5, 20)

# Exit reasons are stored as codes, not strings. Forty-four string columns per
# row is most of the tape's width, and the tape is the artifact every table in
# the report is re-derived from.
REASONS: tuple[str, ...] = ("stop", "tp", "time", "trail1", "trail2", "trail3")
REASON_CODE: dict[str, int] = {r: i for i, r in enumerate(REASONS)}


def fair_rate(target_r: float) -> float:
    """Strike rate a driftless walk gives a target ``k`` R away against a 1R stop.

    First passage between two fixed levels: ``1 / (1 + k)``. At zero cost it is
    also the break-even strike rate, so it is the null every exit is read
    against.
    """
    return 1.0 / (1.0 + target_r)


@dataclass(frozen=True)
class SniperConfig:
    """Everything the script fixes, and everything it leaves open."""

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_len: int = 14
    adx_smooth: int = 14
    adx_strong: float = 20.0
    rsi_ob: float = 75.0
    rsi_os: float = 25.0
    swing_lookback: int = 10
    vol_avg_len: int = 42
    high_vol_thresh: float = 1.3
    vol_widen: float = 1.5
    structure_buffer_atr: float = 0.2
    structure_cap: float = 1.5
    structure_floor: float = 0.5
    max_hold_bars: int = 60
    """A filled position's life cap, in bars of its own timeframe."""
    skip_break_hours: bool = True
    """Drop sub-hour signals armed inside the maintenance break - bars built
    from a handful of ticks. Hourly and slower bars close at the break anyway."""
    control: str = "none"
    """``none`` - the signals. ``matched`` - the mechanics-only null: a
    size-matched sample of ordinary bars of the same timeframe, entered at
    market on the bar's close with the same ATR and structure stops and the
    same exit grid, direction a hash coin. It separates "the confluence engine
    picks bars" from "a 1R stop and an R-multiple target does this to any bar"."""
    lots: float = 0.01


# --------------------------------------------------------------------------
# Pine's built-ins, reimplemented
# --------------------------------------------------------------------------

def _recursive_mean(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """``y[i] = y[i-1] + alpha * (x[i] - y[i-1])``, seeded with the SMA.

    Pine seeds both ``ta.ema`` and ``ta.rma`` with the SMA of the first ``n``
    values at bar ``n-1`` and leaves everything before that na. Written as an
    exponential filter over ``[seed, x[n:]]`` rather than a Python loop: the
    sweep runs this thirteen times per (timeframe, preset) over bar counts in
    the millions, and the loop is most of the run.
    """
    out = np.full(x.size, np.nan)
    if n <= 0 or x.size < n:
        return out
    head = np.asarray(x[:n], dtype=float)
    if not np.isfinite(head).all():
        return out
    tail = np.empty(x.size - n + 1)
    tail[0] = head.mean()
    tail[1:] = x[n:]
    out[n - 1:] = (pl.Series(tail)
                   .ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
                   .to_numpy())
    return out


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """``ta.rma``: Wilder's smoothing, seeded with the SMA of the first ``n``."""
    return _recursive_mean(x, n, 1.0 / n if n else 1.0)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """``ta.ema``: alpha = 2/(n+1), seeded with the SMA of the first ``n``."""
    return _recursive_mean(x, n, 2.0 / (n + 1.0) if n else 1.0)


def rsi(x: np.ndarray, n: int) -> np.ndarray:
    """``ta.rsi``: RMA of gains over RMA of losses."""
    d = np.diff(x, prepend=x[0])
    d[0] = 0.0
    up = rma(np.maximum(d, 0.0), n)
    down = rma(np.maximum(-d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + up / down)
    out = np.where(down == 0, 100.0, out)
    out = np.where(up == 0, 0.0, out)
    return np.where(np.isnan(up) | np.isnan(down), np.nan, out)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """``ta.tr``, with the first bar's range as its own true range."""
    prev = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    return tr


def true_range_avg(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   n: int) -> np.ndarray:
    """``ta.atr``: Wilder's RMA of the true range.

    Identical to :func:`qlab.strategies.sweep_orderblock.atr_rma`, which the
    tests assert, but built on the vectorised recursion.
    """
    return rma(true_range(high, low, close), n)


def macd_hist(x: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    """``ta.macd``'s third return: the histogram, macd line less its signal."""
    line = ema(x, fast) - ema(x, slow)
    ok = ~np.isnan(line)
    out = np.full(x.size, np.nan)
    if not ok.any():
        return out
    start = int(np.argmax(ok))
    sig = np.full(x.size, np.nan)
    sig[start:] = ema(line[start:], signal)
    return line - sig


def dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        di_len: int, adx_smooth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``ta.dmi``: +DI, -DI and ADX, all on Wilder's smoothing."""
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    up[0] = down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    trur = rma(true_range(high, low, close), di_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100.0 * rma(plus_dm, di_len) / trur
        minus = 100.0 * rma(minus_dm, di_len) / trur
        total = plus + minus
        dx = np.abs(plus - minus) / np.where(total == 0, 1.0, total)
    ok = ~np.isnan(dx)
    adx = np.full(close.size, np.nan)
    if ok.any():
        start = int(np.argmax(ok))
        adx[start:] = 100.0 * rma(dx[start:], adx_smooth)
    return plus, minus, adx


def rolling_extreme(x: np.ndarray, n: int, *, largest: bool) -> np.ndarray:
    """``ta.highest``/``ta.lowest`` over ``n`` bars including the current one."""
    out = np.full(x.size, np.nan)
    if x.size == 0:
        return out
    frame = pl.DataFrame({"x": x})
    expr = pl.col("x").rolling_max(n) if largest else pl.col("x").rolling_min(n)
    return frame.select(expr)["x"].to_numpy()


def _hash_u(ts_us: np.ndarray, mult: int, mod: int) -> np.ndarray:
    """A deterministic uniform draw from a timestamp - stable across subsets."""
    return ((ts_us // 1000) * mult % mod) / float(mod)


# --------------------------------------------------------------------------
# The confluence engine
# --------------------------------------------------------------------------

def htf_bias(bars: pl.DataFrame, htf: pl.DataFrame | None,
             preset: Preset) -> np.ndarray:
    """``htfBias`` per chart bar: +1, -1 or 0, from the last *closed* HTF bar.

    ``request.security(tf, ema[1], lookahead_on)`` returns, during HTF bar *j*,
    the value computed on bar *j-1*. An as-of join of the chart bar's **open**
    against the HTF bar's close time picks exactly that bar: the most recent
    HTF bar that had already closed when the chart bar began.
    """
    if htf is None:
        # htfInput = "" is the chart timeframe, and the expression is still
        # ema[1] - so the factor reads the *previous* bar's EMA pair.
        c = bars["close"].to_numpy()
        diff = ema(c, preset.ema_fast) - ema(c, preset.ema_slow)
        return np.sign(np.nan_to_num(np.r_[np.nan, diff[:-1]])).astype(np.int8)

    hc = htf["close"].to_numpy()
    ref = pl.DataFrame({
        "ts": htf["ts"],
        "_hf": ema(hc, preset.ema_fast),
        "_hs": ema(hc, preset.ema_slow),
    }).sort("ts")
    joined = (bars.select(ts_open="ts_open").sort("ts_open")
              .join_asof(ref, left_on="ts_open", right_on="ts", strategy="backward"))
    diff = (joined["_hf"] - joined["_hs"]).to_numpy()
    return np.sign(np.nan_to_num(diff)).astype(np.int8)


def indicators(bars: pl.DataFrame, preset: Preset,
               cfg: SniperConfig) -> dict[str, np.ndarray]:
    """Every series the script computes, on one timeframe and one preset.

    The HTF bias is deliberately not here: it is the only series that depends
    on the HTF setting, so keeping it out lets one (timeframe, preset) pay for
    the other twelve series once and vary only the bias.
    """
    c = bars["close"].to_numpy()
    h = bars["high"].to_numpy()
    low = bars["low"].to_numpy()

    e_fast = ema(c, preset.ema_fast)
    e_slow = ema(c, preset.ema_slow)
    e_trend = ema(c, preset.ema_trend)
    atr = true_range_avg(h, low, c, preset.atr_len)
    r = rsi(c, preset.rsi_len)
    hist = macd_hist(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    di_p, di_m, adx = dmi(h, low, c, cfg.adx_len, cfg.adx_smooth)

    # ``nz(ta.sma(atrVal, 42), atrVal)``: the script falls back to the ATR
    # itself while the average is unwarm, which makes the ratio exactly 1.
    atr_avg = pl.Series(atr).rolling_mean(cfg.vol_avg_len).to_numpy()
    atr_avg = np.where(np.isnan(atr_avg), atr, atr_avg)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(atr_avg > 0, atr / atr_avg, 1.0)

    return {
        "close": c, "high": h, "low": low,
        "ema_fast": e_fast, "ema_slow": e_slow, "ema_trend": e_trend,
        "atr": atr, "rsi": r, "hist": hist,
        "di_plus": di_p, "di_minus": di_m, "adx": adx,
        "vol_ratio": vol_ratio,
        "swing_low": rolling_extreme(low, cfg.swing_lookback + 1, largest=False),
        "swing_high": rolling_extreme(h, cfg.swing_lookback + 1, largest=True),
    }


# The eight factors this feed can compute. Volume and VWAP are dropped by the
# script's own adaptive scale, so the max is 8.0 rather than 10.0.
MAX_SCORE = 8.0


def scores(ind: dict[str, np.ndarray], bias: np.ndarray,
           cfg: SniperConfig) -> tuple[np.ndarray, np.ndarray]:
    """``bullScore`` and ``bearScore`` per bar, on the 8-point adaptive scale."""
    c, ef, es, et = ind["close"], ind["ema_fast"], ind["ema_slow"], ind["ema_trend"]
    r, hist, adx = ind["rsi"], ind["hist"], ind["adx"]
    dp, dm, htf = ind["di_plus"], ind["di_minus"], bias
    prev = np.r_[np.nan, hist[:-1]]
    prev[0] = hist[0]
    strong = adx > cfg.adx_strong

    bull = (
        (ef > es) * 1.0
        + (c > et) * 1.0
        + ((r > 50) & (r < cfg.rsi_ob)) * 1.0
        + (hist > 0) * 1.0
        + (hist > prev) * 1.0
        + (strong & (dp > dm)) * 1.0
        + (htf == 1) * 1.5
        + (c > ef) * 0.5
    )
    bear = (
        (ef < es) * 1.0
        + (c < et) * 1.0
        + ((r < 50) & (r > cfg.rsi_os)) * 1.0
        + (hist < 0) * 1.0
        + (hist < prev) * 1.0
        + (strong & (dm > dp)) * 1.0
        + (htf == -1) * 1.5
        + (c < ef) * 0.5
    )
    return bull, bear


SIGNAL_COLUMNS: tuple[str, ...] = (
    "interval", "preset", "htf_name", "direction", "arm_idx", "arm_us",
    "close", "atr", "score", "max_score", "grade", "score_r",
    "htf_bias", "rsi", "adx", "vol_ratio", "regime_high",
    "stop_atr", "stop_atr_w", "stop_struct", "stop_struct_w",
)


def _stops(direction: np.ndarray, entry: np.ndarray, atr: np.ndarray,
           swing_low: np.ndarray, swing_high: np.ndarray, high_vol: np.ndarray,
           preset: Preset, cfg: SniperConfig) -> dict[str, np.ndarray]:
    """``calcSL`` for all four stop keys at once.

    The structure stop takes the **wider** of the ATR stop and the swing
    extreme plus a buffer, then caps the distance at ``structure_cap`` times
    the ATR distance and floors it at ``structure_floor`` ATR - the script's
    order, which matters because the cap can undo the floor and does not.
    """
    out: dict[str, np.ndarray] = {}
    for widen in (False, True):
        mult = np.where(high_vol & widen, cfg.vol_widen, 1.0)
        atr_dist = atr * preset.sl_mult * mult
        atr_stop = entry - direction * atr_dist

        buf = atr * cfg.structure_buffer_atr
        struct_lvl = np.where(direction == 1, swing_low - buf, swing_high + buf)
        # wider = lower for a long, higher for a short
        stop = np.where(direction == 1,
                        np.minimum(atr_stop, struct_lvl),
                        np.maximum(atr_stop, struct_lvl))
        cap = atr_dist * cfg.structure_cap
        dist = np.abs(entry - stop)
        stop = np.where(dist > cap, entry - direction * cap, stop)
        floor = atr * cfg.structure_floor
        dist = np.abs(entry - stop)
        stop = np.where(dist < floor, entry - direction * floor, stop)

        suffix = "_w" if widen else ""
        out[f"atr{suffix}"] = atr_stop
        out[f"struct{suffix}"] = stop
    return out


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Float64 for c in SIGNAL_COLUMNS})


def signals(bars: pl.DataFrame, preset: Preset, cfg: SniperConfig, *,
            interval: str, htf: pl.DataFrame | None = None,
            htf_name: str = "none",
            ind: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Every bar the engine would fire on, with the context a filter needs.

    The ``lastDirection`` gate is **not** applied - it is identical to the
    account's one-position rule (see the module docstring), so it belongs in
    :func:`sequence`, where it can be applied once per exit rule for free.
    """
    n = bars.height
    warmup = max(preset.ema_trend, 50)
    if n <= warmup + 2:
        return _empty()

    ind = indicators(bars, preset, cfg) if ind is None else ind
    bias = htf_bias(bars, htf, preset)
    bull, bear = scores(ind, bias, cfg)
    c, ef, es = ind["close"], ind["ema_fast"], ind["ema_slow"]
    r = ind["rsi"]

    prev_f = np.r_[np.nan, ef[:-1]]
    prev_s = np.r_[np.nan, es[:-1]]
    cross_up = (prev_f <= prev_s) & (ef > es)
    cross_dn = (prev_f >= prev_s) & (ef < es)

    min_score = preset.min_score * MAX_SCORE / 10.0
    warm = np.arange(n) >= warmup
    warm &= ~np.isnan(ind["atr"]) & ~np.isnan(ind["adx"]) & ~np.isnan(r)

    buy = cross_up & (c > ef) & (c > es) & (r < cfg.rsi_ob) & (bull >= min_score) & warm
    sell = cross_dn & (c < ef) & (c < es) & (r > cfg.rsi_os) & (bear >= min_score) & warm

    ts = bars["ts"].cast(pl.Int64).to_numpy()
    if cfg.control == "matched":
        # Size-matched ordinary bars: the same count, drawn deterministically
        # from the same warm bars, with the direction a hash coin.
        n_sig = int(buy.sum() + sell.sum())
        pool = np.nonzero(warm & ~np.isnan(ind["ema_trend"]))[0]
        if n_sig == 0 or pool.size == 0:
            return _empty()
        rate = min(1.0, n_sig / pool.size)
        pick = pool[_hash_u(ts[pool], 2_654_435_761, 1_000_003) < rate]
        if pick.size == 0:
            return _empty()
        idx = pick
        direction = np.where(_hash_u(ts[pick], 40_503, 65_521) < 0.5, 1, -1).astype(np.int8)
    else:
        idx = np.nonzero(buy | sell)[0]
        if idx.size == 0:
            return _empty()
        direction = np.where(buy[idx], 1, -1).astype(np.int8)

    entry = c[idx]
    atr = ind["atr"][idx]
    high_vol = ind["vol_ratio"][idx] > cfg.high_vol_thresh
    score = np.where(direction == 1, bull[idx], bear[idx])
    stops = _stops(direction, entry, atr, ind["swing_low"][idx],
                   ind["swing_high"][idx], high_vol, preset, cfg)
    score_r = score / MAX_SCORE

    return pl.DataFrame({
        "interval": np.repeat(interval, idx.size),
        "preset": np.repeat(preset.name, idx.size),
        "htf_name": np.repeat(htf_name, idx.size),
        "direction": direction,
        "arm_idx": idx.astype(np.int64),
        "arm_us": ts[idx],
        "close": entry,
        "atr": atr,
        "score": score,
        "max_score": np.repeat(MAX_SCORE, idx.size),
        "grade": np.where(score_r >= GRADE_APLUS_R, "A+",
                 np.where(score_r >= GRADE_A_R, "A",
                 np.where(score_r >= GRADE_B_R, "B", "C"))),
        "score_r": score_r,
        "htf_bias": bias[idx],
        "rsi": r[idx],
        "adx": ind["adx"][idx],
        "vol_ratio": ind["vol_ratio"][idx],
        "regime_high": high_vol,
        **{f"stop_{k}": v for k, v in stops.items()},
    })


# --------------------------------------------------------------------------
# One signal, resolved against the tape
# --------------------------------------------------------------------------

def _pine_path(out: np.ndarray, f: int, hold_end: int, d: int, entry: float,
               risk: float, stop_lvl: float, stop_i: int, hits: dict[float, int],
               ladder: tuple[float, float, float], full_exit: bool,
               slip: float) -> tuple[float, str, int]:
    """The script's own managed exit, walked in tick order.

    The stop level steps: entry stop -> breakeven at TP1 -> TP1 at TP2 ->
    TP2 at TP3. Each step is only live from the instant its target trades, so
    the search for a breach restarts at that instant against the new level.
    Returns (exit price, reason, index).
    """
    r1, r2, r3 = ladder
    i1, i2, i3 = hits[r1], hits[r2], hits[r3]
    tp = [entry + d * r * risk for r in ladder]
    # (segment end, stop level, R booked if this segment's stop is hit)
    legs = [
        (i1 if i1 >= 0 else hold_end, stop_lvl, stop_i),
        (i2 if i2 >= 0 else hold_end, entry, -1),
        (i3 if i3 >= 0 else hold_end, tp[0], -1),
        (hold_end, tp[1], -1),
    ]
    pos = f
    for leg, (end, level, precomputed) in enumerate(legs):
        end = min(end, hold_end)
        if pos < end:
            # Leg 0's breach is the stop search already done for this stop key;
            # reuse it, but only where it lands inside this leg's window.
            j = precomputed if leg == 0 else _first_cross(
                out, pos, end, level, below=(d == 1))
            if leg == 0 and j >= end:
                j = -1
            if j >= 0:
                # A stop and a trail are both market orders: they fill at the
                # tick that breached, not at the level they were resting on.
                return float(out[j]) - d * slip, ("stop" if leg == 0
                                                  else f"trail{leg}"), j
        pos = end
        if leg == 2 and i3 >= 0 and full_exit:
            return tp[2], "tp", i3
        if pos >= hold_end:
            break
    return float(out[hold_end - 1]) - d * slip, "time", hold_end - 1


def _resolve(tape, sig: dict, bar_ts: np.ndarray, cfg: SniperConfig,
             slip: float, commission_px: float, usd_per_px: float,
             rolls: np.ndarray) -> dict:
    """Fill the signal at market, then price all forty-four exits on that fill.

    Tie-breaks go against the trade: a tick satisfying a stop and a target is a
    stop, because one quote cannot say which came first inside itself.
    """
    d = int(sig["direction"])
    row = dict(sig)
    row["filled"] = False
    work = tape.ask if d == 1 else tape.bid     # the side an entry trades on
    out = tape.bid if d == 1 else tape.ask      # the side an exit trades on
    n = tape.ts.size
    f = tape.standing(int(sig["arm_us"]))
    if f < 0 or f + 1 >= n:
        row["fill_reason"] = "no_tape"
        return row

    entry = float(work[f]) + d * slip
    fill_us = int(tape.ts[f])
    fb = int(sig["arm_idx"])
    last = bar_ts.size - 1
    hold_end = tape.after(int(bar_ts[min(fb + cfg.max_hold_bars, last)]))
    hold_end = min(max(hold_end, f + 1), n)

    stops = {k: float(sig[f"stop_{k}"]) for k in STOP_KEYS}
    risks = {k: d * (entry - v) for k, v in stops.items()}
    if min(risks.values()) <= 0:
        row["fill_reason"] = "bad_risk"
        return row

    mid_f = float((tape.bid[f] + tape.ask[f]) / 2)
    spread_f = float(tape.ask[f] - tape.bid[f])
    row.update(fill_reason="filled", filled=True, fill_us=fill_us, entry=entry,
               entry_mid=mid_f, spread_fill=spread_f, slip_px=slip,
               commission_px=commission_px, usd_per_px=usd_per_px,
               day=datetime.fromtimestamp(fill_us / US, tz=timezone.utc).date())

    # Barrier-free, cost-free: the signal itself, before any exit is chosen.
    for hbar in FORWARD_BARS:
        v = float("nan")
        if fb + hbar <= last:
            i = tape.standing(int(bar_ts[fb + hbar]))
            if i > f:
                v = d * (float((tape.bid[i] + tape.ask[i]) / 2) - mid_f) / mid_f * 1e4
        row[f"fwd{hbar}_bps"] = v

    seg = out[f:hold_end]
    best = float(seg.max() if d == 1 else seg.min())
    worst = float(seg.min() if d == 1 else seg.max())
    row["mfe_r"] = d * (best - entry) / risks["struct"]
    row["mae_r"] = d * (entry - worst) / risks["struct"]

    end_px = float(out[hold_end - 1]) - d * slip
    base_rolls = int(np.searchsorted(rolls, fill_us, side="right"))

    for sk, stop in stops.items():
        risk = risks[sk]
        row[f"risk_{sk}"] = risk
        row[f"risk_bps_{sk}"] = risk / entry * 1e4
        row[f"cost_r_{sk}"] = (commission_px + spread_f + 2.0 * slip) / risk

        si = _first_cross(out, f, hold_end, stop, below=(d == 1))
        stop_px = float(out[si]) - d * slip if si >= 0 else None

        # First crossing of every R level the grid needs, found once. The
        # levels ascend in R, so each search starts where the last one ended.
        hits: dict[float, int] = {}
        pos = f
        for k in R_LEVELS:
            pos = _first_cross(out, max(pos, f), hold_end, entry + d * k * risk,
                               below=(d == -1))
            hits[k] = pos
            if pos < 0:
                pos = hold_end

        results: list[tuple[str, float, str, int]] = []
        for tk, k in TARGET_R.items():
            ti = hits[k]
            if ti >= 0 and (si < 0 or ti < si):
                results.append((tk, entry + d * k * risk, "tp", ti))
            elif si >= 0:
                results.append((tk, stop_px, "stop", si))
            else:
                results.append((tk, end_px, "time", hold_end - 1))
        if si >= 0:
            results.append((HOLD, stop_px, "stop", si))
        else:
            results.append((HOLD, end_px, "time", hold_end - 1))

        for lname, ladder in LADDERS.items():
            px, reason, x = _pine_path(out, f, hold_end, d, entry, risk,
                                       stop, si, hits, ladder, True, slip)
            results.append((f"p{lname}", px, reason, x))
        px, reason, x = _pine_path(out, f, hold_end, d, entry, risk,
                                   stop, si, hits, LADDERS["L123"], False, slip)
        results.append(("prL123", px, reason, x))

        for tk, px, reason, x in results:
            net = d * (px - entry) - commission_px
            key = f"{sk}_{tk}"
            x_us = int(tape.ts[x])
            row[f"r_{key}"] = np.float32(net / risk)
            row[f"reason_{key}"] = np.int8(REASON_CODE[reason])
            row[f"exit_us_{key}"] = x_us
            if tk in ROLL_KEYS:
                row[f"rolls_{key}"] = int(
                    np.searchsorted(rolls, x_us, side="right") - base_rolls)
    return row


# --------------------------------------------------------------------------
# The account: one position at a time, which is also the lastDirection gate
# --------------------------------------------------------------------------

def sequence(trades: pl.DataFrame, exit_keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    """The trades a one-position-at-a-time account takes, for each exit rule.

    Each (interval, preset, htf_name) is its own account. This is the flat
    account: every trade runs to its own barrier and a signal arriving while
    one is open is skipped, whatever its direction. It is the right pairing for
    the single-target exits, which nominate no reversal;
    :func:`sequence_reverse` is the script's own rule.
    """
    if trades.is_empty():
        return {k: trades for k in exit_keys}
    keep: dict[str, list[pl.DataFrame]] = {k: [] for k in exit_keys}
    for _, g in trades.group_by(["interval", "preset", "htf_name"], maintain_order=True):
        g = g.filter(pl.col("filled")).sort("fill_us")
        if g.is_empty():
            continue
        for k in exit_keys:
            col = f"exit_us_{k}"
            if col not in g.columns:
                continue
            gk = g.filter(pl.col(col).is_not_null())
            if gk.is_empty():
                continue
            keep[k].append(gk[_taken_indices(gk["fill_us"].to_numpy(),
                                             gk[col].to_numpy())])
    return {k: (pl.concat(v).sort("fill_us") if v else trades.clear())
            for k, v in keep.items()}


def _reverse_walk(fill_us: np.ndarray, exit_us: np.ndarray, direction: np.ndarray,
                  entry: np.ndarray, risk: np.ndarray, commission: np.ndarray,
                  r_barrier: np.ndarray) -> tuple[list[int], list[float], list[bool]]:
    """One pass of the script's gate over one account's signals, in time order.

    Returns the taken rows, the R each realised, and whether the reversal cut
    it short. A cut trade is valued at the opposing signal's own fill price,
    which is the right price on both sides: that fill is a sell at the bid less
    slippage when the open trade is long, and a buy at the ask plus slippage
    when it is short - exactly the prices closing the open trade would pay.
    """
    taken: list[int] = []
    realised: list[float] = []
    cut: list[bool] = []
    n = fill_us.size
    live = -1
    for i in range(n):
        if live < 0:
            live = i
            continue
        if fill_us[i] >= exit_us[live]:
            # the barrier had already closed it; the account was flat
            taken.append(live); realised.append(float(r_barrier[live])); cut.append(False)
            live = i
        elif direction[i] != direction[live]:
            d = float(direction[live])
            pnl = d * (entry[i] - entry[live]) - commission[live]
            taken.append(live); realised.append(pnl / risk[live]); cut.append(True)
            live = i
        # same direction while a position is open: lastDirection blocks it
    if live >= 0:
        taken.append(live); realised.append(float(r_barrier[live])); cut.append(False)
    return taken, realised, cut


def sequence_reverse(trades: pl.DataFrame, exit_key: str) -> pl.DataFrame:
    """The trades the script's own account takes under one exit rule.

    Stop-and-reverse with no pyramiding. Adds ``r_eff`` - the R actually
    realised, which is the barrier's R unless an opposing signal cut the trade
    short - and ``was_cut``.
    """
    col, stop_key = f"r_{exit_key}", exit_key.rsplit("_", 1)[0]
    for sk in sorted(STOP_KEYS, key=len, reverse=True):
        if exit_key.startswith(sk + "_"):
            stop_key = sk
            break
    if trades.is_empty() or col not in trades.columns:
        return trades.clear().with_columns(r_eff=pl.lit(0.0), was_cut=pl.lit(False))
    out: list[pl.DataFrame] = []
    for _, g in trades.group_by(["interval", "preset", "htf_name"], maintain_order=True):
        g = g.filter(pl.col("filled") & pl.col(f"exit_us_{exit_key}").is_not_null())
        if g.is_empty():
            continue
        g = g.sort("fill_us")
        idx, realised, cut = _reverse_walk(
            g["fill_us"].to_numpy(), g[f"exit_us_{exit_key}"].to_numpy(),
            g["direction"].to_numpy(), g["entry"].to_numpy(),
            g[f"risk_{stop_key}"].to_numpy(), g["commission_px"].to_numpy(),
            g[col].cast(pl.Float64).to_numpy())
        out.append(g[idx].with_columns(
            r_eff=pl.Series(realised, dtype=pl.Float64),
            was_cut=pl.Series(cut, dtype=pl.Boolean)))
    if not out:
        return trades.clear().with_columns(r_eff=pl.lit(0.0), was_cut=pl.lit(False))
    return pl.concat(out).sort("fill_us")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(symbol: str, cfg: SniperConfig | None = None, *,
        presets: Sequence[str] = ALL_PRESETS,
        intervals: Sequence[str] = ALL_INTERVALS,
        htf_modes: Sequence[str] = ("auto", "none"),
        split: str = "dev", allow_test: bool = False,
        cost: CostModel | None = None, verbose: bool = False,
        sink: "Path | None" = None) -> pl.DataFrame:
    """Every preset on one symbol, every timeframe, one split.

    One row per signal. :func:`sequence` reduces it to what an account takes;
    every filter the script offers is a predicate on the columns returned here.

    ``sink`` writes each month straight to parquet and returns an empty frame,
    so a whole-symbol tape - 1.5M rows at ~1.5 kB each on the finest grid -
    never has to exist in memory at once. Four workers accumulating one each
    would be most of this machine's RAM.
    """
    cfg = cfg or SniperConfig()
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}
    sp = SPLITS[split]
    hard_end = (datetime(sp.end.year, sp.end.month, sp.end.day, tzinfo=timezone.utc)
                + timedelta(days=1))

    base = load_bars(symbol, "1m", split=split, allow_test=allow_test,
                     columns=["ts", "ts_open", "open", "high", "low", "close", "n_ticks"])
    if base.is_empty():
        return pl.DataFrame()

    built: dict[str, pl.DataFrame] = {iv: build_bars(base, iv) for iv in
                                      sorted({*intervals, *HTF_FOR.values()},
                                             key=lambda k: INTERVAL_MINUTES[k])}
    del base

    frames, bar_ts = [], {}
    for iv in intervals:
        bars = built[iv]
        bar_ts[iv] = bars["ts"].cast(pl.Int64).to_numpy()
        for pname in presets:
            preset = PRESETS[pname]
            ind = indicators(bars, preset, cfg)
            for mode in htf_modes:
                hname = HTF_FOR[iv] if mode == "auto" else "none"
                htf = built.get(hname) if mode == "auto" else None
                if htf is not None and INTERVAL_MINUTES[hname] <= INTERVAL_MINUTES[iv]:
                    htf, hname = None, "none"
                found = signals(bars, preset, cfg, interval=iv, htf=htf,
                                htf_name=hname, ind=ind)
                if (cfg.skip_break_hours and spec.daily_break_utc
                        and INTERVAL_MINUTES[iv] < 60 and not found.is_empty()):
                    lo, hi = spec.daily_break_utc
                    hour = (pl.col("arm_us") // (3600 * US)) % 24
                    found = found.filter(~hour.is_between(lo, hi - 1))
                if verbose:
                    print(f"  {symbol} {iv:>3} {pname:12} htf={hname:4}: "
                          f"{found.height:,}", flush=True)
                frames.append(found)
    for iv in list(built):
        if iv not in intervals:
            del built[iv]

    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    sigs = pl.concat(frames).with_columns(
        symbol=pl.lit(symbol)).sort("arm_us")

    rolls = roll_instants(sp.start - timedelta(days=3), sp.end + timedelta(days=3))
    cache = _TickCache(symbol, int(hard_end.timestamp() * US))
    lots_px = (spec.contract_size or 1.0) * cfg.lots
    span = cfg.max_hold_bars + max(FORWARD_BARS) + 1

    out: list[pl.DataFrame] = []
    parts: list[Path] = []
    if sink is not None:
        sink.parent.mkdir(parents=True, exist_ok=True)
    months = sorted({(d.year, d.month) for d in
                     sigs.select(pl.from_epoch(pl.col("arm_us") // US).dt.date()
                                 .alias("d"))["d"].to_list()})
    for y, m in months:
        m0 = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * US)
        m1 = int(datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * US)
        batch = sigs.filter((pl.col("arm_us") >= m0) & (pl.col("arm_us") < m1))
        if batch.is_empty():
            continue
        need = m1
        for iv, idx in batch.select("interval", "arm_idx").iter_rows():
            arr = bar_ts[iv]
            need = max(need, int(arr[min(idx + span, arr.size - 1)]))
        tape = cache.tape(m0 - 3 * 86400 * US, need + 3600 * US)
        if tape.ts.size == 0:
            continue
        rows = []
        for sig in batch.iter_rows(named=True):
            hour = (sig["arm_us"] // (3600 * US)) % 24
            rate = 1.0 if spec.quote_ccy == "USD" else sig["close"]
            rows.append(_resolve(tape, sig, bar_ts[sig["interval"]], cfg,
                                 slip_by_hour[hour],
                                 spec.commission_pips(rate) * spec.pip,
                                 lots_px / rate, rolls))
        frame = pl.DataFrame(rows, infer_schema_length=None)
        del tape, rows
        if sink is not None:
            part = sink.with_name(f"{sink.stem}.part{len(parts):03d}.parquet")
            frame.write_parquet(part)
            parts.append(part)
            del frame
        else:
            out.append(frame)
        if verbose:
            print(f"  {symbol} {y}-{m:02d}", flush=True)
    if sink is not None:
        if parts:
            (pl.scan_parquet(parts, allow_missing_columns=True)
             .sort("arm_us").sink_parquet(sink))
            for part in parts:
                part.unlink()
        return pl.DataFrame()
    if not out:
        return pl.DataFrame()
    return pl.concat(out, how="diagonal_relaxed").sort("arm_us")
