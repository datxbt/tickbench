"""Structural-break entries on gold, decomposed into entry, sizing and exit.

The strategy under test (Quan, Zhong and Jiang) is a swing-break system on H4
gold: find a confirmed pivot in the H4 closes, rest a stop order at it, take a
fixed 3-dollar stop and a fixed 20-dollar target, and overlay a partial exit
that takes half the position off at +1R when a one-minute candle turns against
it. Three things are claimed, and the whole point of the design is that they are
*separable* claims:

1. the structural-break **entry** carries alpha,
2. the regime-conditional **sizing** adds more than leverage does,
3. the half-exit **overlay** improves the risk-adjusted return.

So the module is built as three independent axes over one fixed core, and every
run states which cell of that grid it is. Holding the other two axes fixed while
one moves is the only way a component-level claim can be checked at all; a
harness that changes two things at once can only ever report that the pair
differs.

The fourth axis is the backtest itself
--------------------------------------
This strategy has three exit conditions - a stop, a target, and a sub-bar
overlay - that can all come due inside the same minute, and the order in which
they resolve is not observable from OHLC. That makes it exactly the kind of
system whose backtest resolution is a result rather than an implementation
detail, so resolution is the fourth axis:

``ticks``
    The real broker tick stream. Stops and targets fill on the quote that
    touched them. This is the headline mode and the only one whose numbers are
    reported without a caveat.
``interp``
    Four synthetic ticks per minute tracing open, high, low and close in the
    order implied by the candle's own direction - the "every tick" convention an
    MT5 tester uses when it has only bars. Cheap, and roughly what a bar-level
    backtest of this strategy would report.
``close``
    One price per minute. Deliberately the crudest tier, kept because it bounds
    the error from the other direction and because it is what a naive pandas
    backtest of the same rules would produce.

The gap between ``interp`` and ``ticks`` is not noise to be averaged away: it is
the measurement, and the report quotes it as an overstatement percentage.

The half-exit can only fire at a minute close
----------------------------------------------
Its trigger reads "the current M1 candle is counter-directional", and a candle
does not have a direction until it closes. Evaluating it at the close is
therefore not a simplification, it is the only non-anticipating reading - and it
is why the stop and the target, which are resting orders, get priority inside
the minute they share with it.

Sizing rules that only lever
----------------------------
:func:`sharpe_invariance_check` is wired into the harness rather than left to a
test file, because the single most common way a sizing study fools itself is to
report a rule that multiplies every position by a constant. Sharpe is invariant
to that, so any rule whose Sharpe moves must be changing the *pattern* of
exposure and not merely its scale - and any rule whose Sharpe does not move,
however much its return does, has added leverage and called it alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from ..bars import resample_bars
from ..costs import CostModel
from ..engine import Account, Fill, Trade, lots_for_risk
from ..loader import get_split, load_bars, load_ticks
from ..symbols import get_spec

US = 1_000_000
DAY_US = 86_400 * US


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakConfig:
    """The entry, and the two numbers that fix the trade's geometry."""

    swing_bars: int = 10
    """``W``: bars either side of a pivot. A swing high is an H4 close that no
    close in the surrounding 2W+1 bars exceeds, and it is not *confirmed* until
    W bars later, so nothing here can be acted on before it exists."""

    atr_period: int = 14
    atr_guard_mult: float = 5.0
    """Cancel a resting order whose level is further than this many ATR from the
    current price. It is a sanity guard, not a filter with a view."""

    delta_sl: float = 3.0
    """Stop distance in dollars per ounce. This is 1R and every risk number in
    the module is quoted in it."""

    delta_tp: float = 20.0
    """Target distance in dollars per ounce - 6.67R at the default stop."""

    arm: str = "on_confirm"
    """How the pending order relates to the break, which is the one genuinely
    ambiguous instruction in the specification.

    ``on_confirm``
        The stop order rests at the pivot from the moment the pivot is
        confirmed, and fills on the first quote that trades through it. By
        construction the confirming bar's close is at or below a swing high, so
        the order is always placeable - this is the reading that corresponds to
        an order actually sitting in the book.
    ``retest``
        The order is placed only *after* an H4 bar closes beyond the level, as
        the specification's step 3 literally says. Price is then already past
        the level, so a stop order at it needs a pullback first: the order is
        armed, becomes eligible once price trades back through the level, and
        fills on the re-break. Fewer trades, better entry prices.

    Both are run. Which one the specification meant changes the answer, so
    neither is presented as the answer.
    """

    retest_expiry_bars: int = 6
    """How many H4 bars a ``retest`` order waits for its pullback before it is
    cancelled. Six is one day."""

    interval: str = "4h"
    max_hold_days: float = 30.0
    """Bound on a position's life. Not part of the strategy - a static 3-dollar
    stop on gold is almost always hit within hours - but an unbounded hold makes
    the tick tape unchunkable, and the report says how many trades reach it."""

    def __post_init__(self) -> None:
        if self.arm not in ("on_confirm", "retest"):
            raise ValueError(f"unknown arm mode {self.arm!r}")
        if self.delta_sl <= 0 or self.delta_tp <= 0:
            raise ValueError("stop and target distances must be positive")


@dataclass(frozen=True)
class ExitConfig:
    """The overlay, and the two switches that take it apart."""

    half_exit: bool = True
    """``False`` is the ablation control the specification calls B5: identical
    entry, sizing, stop and target, with the partial exit never firing."""

    require_counter_candle: bool = True
    """``False`` is ``V11_FixedR``: fire at +1R regardless of the candle's shape.
    Isolating this is the whole point of the exit axis - it separates "taking
    something off at +1R helps" from "the candle filter helps"."""

    half_exit_r: float = 1.0
    half_fraction: float = 0.5
    breakeven_after_half: bool = True

    @property
    def label(self) -> str:
        if not self.half_exit:
            return "B5 (no half exit)"
        return "V10/V11" if self.require_counter_candle else "V11_FixedR"


@dataclass(frozen=True)
class SizingConfig:
    """How many lots, and on what evidence.

    The specification gives V11 as ``M = clip(m_ATR(rho) * m_Trend(s), 0.3, 3)``
    with both multipliers described only as "calibrated lookups". A calibration
    is not transcribable, so concrete functional forms had to be chosen, and
    they are stated here rather than tuned: ``m_ATR = 1/rho``, which shrinks the
    position when short-horizon volatility is running hot relative to its own
    baseline, and a three-state trend multiplier. Both are exposed as inputs so
    that a reader can see the choice and change it.
    """

    rule: str = "V10"
    """``V10`` | ``V11`` | ``V11_ATR`` | ``V11_TREND`` | ``V11_VOLTGT``."""

    risk_pct: float = 0.0036
    atr_fast: int = 5
    atr_slow: int = 20
    trend_fast: int = 20
    trend_slow: int = 50
    trend_band: float = 0.001
    aligned_mult: float = 1.25
    opposed_mult: float = 0.75
    flat_mult: float = 1.00
    target_vol_pct: float = 15.0
    vol_days: int = 20
    clip: tuple[float, float] = (0.30, 3.00)

    def __post_init__(self) -> None:
        known = {"V10", "V11", "V11_ATR", "V11_TREND", "V11_VOLTGT"}
        if self.rule not in known:
            raise ValueError(f"unknown sizing rule {self.rule!r}, not in {sorted(known)}")


MODES: tuple[str, ...] = ("ticks", "interp", "close")


# --------------------------------------------------------------------------
# Context: H4 bars, ATR, pivots, and the daily regime
# --------------------------------------------------------------------------

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int) -> np.ndarray:
    """Wilder's ATR. Index ``i`` uses bars up to and including ``i``."""
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.column_stack([
        high - low, np.abs(high - prev), np.abs(low - prev)]), axis=1)
    out = np.full(tr.size, np.nan)
    if tr.size <= period:
        return out
    out[period] = np.nanmean(tr[1:period + 1])
    for i in range(period + 1, tr.size):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def confirmed_pivots(close: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Most recent confirmed swing high and swing low, as of each bar's close.

    A pivot at ``t`` is confirmed at ``t + w``, so the value at index ``i`` uses
    only closes at or before ``i``. Returned arrays are therefore safe to read
    at the *open* of bar ``i + 1`` and nowhere earlier.
    """
    n = close.size
    highs = np.full(n, np.nan)
    lows = np.full(n, np.nan)
    last_hi = last_lo = np.nan
    for i in range(n):
        t = i - w
        if t >= w:
            window_left = close[t - w:t]
            window_right = close[t + 1:t + w + 1]
            if close[t] > window_left.max() and close[t] >= window_right.max():
                last_hi = close[t]
            if close[t] < window_left.min() and close[t] <= window_right.min():
                last_lo = close[t]
        highs[i] = last_hi
        lows[i] = last_lo
    return highs, lows


def _daily_regime(bars_1m: pl.DataFrame, cfg: SizingConfig) -> pl.DataFrame:
    """Daily close, trend state and realised volatility, all lagged one day.

    Everything here is shifted so that a value dated ``d`` was fully observable
    at the end of day ``d - 1``. The sizing rule reads it at the moment of a
    fill, which can be any time inside day ``d``, so an unlagged daily close
    would be tomorrow's information in today's position size.
    """
    daily = (
        bars_1m.group_by(pl.col("ts_open").dt.date().alias("d"))
        .agg(c=pl.col("close").last(), n=pl.len())
        .filter(pl.col("n") >= 200)
        .sort("d")
    )
    return daily.with_columns(
        ret=(pl.col("c") / pl.col("c").shift(1) - 1.0),
    ).with_columns(
        sma_fast=pl.col("c").rolling_mean(cfg.trend_fast),
        sma_slow=pl.col("c").rolling_mean(cfg.trend_slow),
        rvol=pl.col("ret").rolling_std(cfg.vol_days) * np.sqrt(252) * 100.0,
    ).with_columns(
        # Lag by one day: these describe the day that has already closed.
        sma_fast=pl.col("sma_fast").shift(1),
        sma_slow=pl.col("sma_slow").shift(1),
        rvol=pl.col("rvol").shift(1),
    ).with_columns(
        trend=pl.when(pl.col("sma_fast") > pl.col("sma_slow") * (1 + cfg.trend_band))
              .then(1)
              .when(pl.col("sma_fast") < pl.col("sma_slow") * (1 - cfg.trend_band))
              .then(-1)
              .otherwise(0)
              .cast(pl.Int8),
    ).select("d", "trend", "rvol")


@dataclass
class Context:
    """Everything the simulation needs that does not come from the tape."""

    h4: pl.DataFrame
    bars_1m: pl.DataFrame
    regime: pl.DataFrame
    cfg: BreakConfig
    sizing: SizingConfig

    @property
    def n_bars(self) -> int:
        return self.h4.height


def build_context(
    symbol: str = "XAUUSD",
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    cfg: BreakConfig | None = None,
    sizing: SizingConfig | None = None,
) -> Context:
    """H4 pivots and ATR, plus the daily regime, all point-in-time."""
    cfg = cfg or BreakConfig()
    sizing = sizing or SizingConfig()
    # Indicators need history before the split's first bar or the first weeks of
    # every run are silently wrong; warmup rows are flagged and dropped below.
    warmup = timedelta(days=max(90, cfg.swing_bars * 2 + sizing.trend_slow + 10))
    bars = load_bars(symbol, "1m", split=split, start=start, end=end,
                     warmup=warmup if split else None, allow_test=allow_test)
    h4 = resample_bars(bars, cfg.interval).sort("ts")

    close = h4["close"].to_numpy()
    high = h4["high"].to_numpy()
    low = h4["low"].to_numpy()
    hi_piv, lo_piv = confirmed_pivots(close, cfg.swing_bars)
    h4 = h4.with_columns(
        atr=pl.Series(_atr(high, low, close, cfg.atr_period)),
        atr_fast=pl.Series(_atr(high, low, close, sizing.atr_fast)),
        atr_slow=pl.Series(_atr(high, low, close, sizing.atr_slow)),
        swing_high=pl.Series(hi_piv),
        swing_low=pl.Series(lo_piv),
    )
    # A bar acts on the state known at its own open, which is the previous
    # bar's close. Shifting here means no downstream code has to remember to.
    h4 = h4.with_columns(
        [pl.col(c).shift(1).alias(c) for c in
         ("atr", "atr_fast", "atr_slow", "swing_high", "swing_low")]
    )
    if "is_warmup" in h4.columns:
        h4 = h4.filter(~pl.col("is_warmup"))
    regime = _daily_regime(bars, sizing)
    return Context(h4=h4, bars_1m=bars, regime=regime, cfg=cfg, sizing=sizing)


# --------------------------------------------------------------------------
# Tapes
# --------------------------------------------------------------------------

@dataclass
class Tape:
    """A price path as three parallel arrays, plus the minute grid over it."""

    ts: np.ndarray        # int64 microseconds, ascending
    bid: np.ndarray
    ask: np.ndarray
    minute_end: np.ndarray   # int64 us, right edge of each M1 bar
    minute_up: np.ndarray    # bool, M1 close > open
    minute_bid: np.ndarray   # bid at the minute's close
    minute_ask: np.ndarray
    mode: str = "ticks"
    interpolated: bool = False
    """Whether the points between quotes are a *path* or a gap.

    On the real tape they are a gap: nothing is known between two quotes, so a
    stop fills at the quote that crossed it and a jump past the level is a real
    jump the account really wears. On the synthetic tape they are a straight
    line by construction, so a stop fills at its own level - taking the bar's
    extreme as the fill price instead would charge the account for an excursion
    the model itself invented.
    """

    def index_at(self, when_us: int) -> int:
        return int(np.searchsorted(self.ts, when_us, side="left"))

    def minute_index_at(self, when_us: int) -> int:
        return int(np.searchsorted(self.minute_end, when_us, side="right"))


def _minute_grid(bars_1m: pl.DataFrame) -> tuple[np.ndarray, ...]:
    frame = bars_1m.sort("ts")
    return (
        frame["ts"].dt.epoch("us").to_numpy().astype(np.int64),
        (frame["close"] > frame["open"]).to_numpy(),
        frame["bid_close"].to_numpy(),
        frame["ask_close"].to_numpy(),
    )


def tick_tape(symbol: str, bars_1m: pl.DataFrame, start, end) -> Tape:
    """The real quote stream over ``[start, end)`` - fidelity mode ``ticks``.

    ``start`` and ``end`` are explicit datetimes, so the split guard in
    :func:`~qlab.loader.load_ticks` does not apply to them and the caller is
    responsible for not stepping outside its own split. :func:`run` clamps
    them; nothing else should call this directly.
    """
    ticks = load_ticks(symbol, start=start, end=end).sort("ts")
    window = bars_1m.filter(
        (pl.col("ts") >= pl.lit(start)) & (pl.col("ts") < pl.lit(end)))
    m_end, m_up, m_bid, m_ask = _minute_grid(window)
    return Tape(
        ts=ticks["ts"].dt.epoch("us").to_numpy().astype(np.int64),
        bid=ticks["bid"].to_numpy(), ask=ticks["ask"].to_numpy(),
        minute_end=m_end, minute_up=m_up, minute_bid=m_bid, minute_ask=m_ask,
        mode="ticks",
    )


def synthetic_tape(bars_1m: pl.DataFrame, mode: str = "interp") -> Tape:
    """A tape built from M1 bars alone - fidelity modes ``interp`` and ``close``.

    ``interp`` lays four quotes inside each minute. Their order follows the
    candle's own direction, which is the convention a bar-only tester uses: a
    minute that closed up is assumed to have gone down first. That assumption is
    exactly what mode ``ticks`` exists to check, and where the two disagree the
    disagreement is the finding.
    """
    frame = bars_1m.sort("ts")
    m_end, m_up, m_bid, m_ask = _minute_grid(frame)
    o = frame["open"].to_numpy()
    h = frame["high"].to_numpy()
    lo = frame["low"].to_numpy()
    c = frame["close"].to_numpy()
    half = frame["spread_close"].to_numpy() / 2.0
    t_open = frame["ts_open"].dt.epoch("us").to_numpy().astype(np.int64)

    if mode == "close":
        # Close-only: every point IS a real quote, and nothing is claimed about
        # the path between them, so this tape is not interpolated either.
        return Tape(ts=m_end, bid=c - half, ask=c + half,
                    minute_end=m_end, minute_up=m_up,
                    minute_bid=m_bid, minute_ask=m_ask, mode="close")

    # Open, then the far extreme, then the near one, then the close.
    first = np.where(m_up, lo, h)
    second = np.where(m_up, h, lo)
    mids = np.column_stack([o, first, second, c]).ravel()
    span = (m_end - t_open)
    offsets = np.column_stack([
        np.zeros_like(span), span // 3, (2 * span) // 3, span - 1]).ravel()
    ts = (np.repeat(t_open, 4) + offsets).astype(np.int64)
    spread = np.repeat(half, 4)
    return Tape(ts=ts, bid=mids - spread, ask=mids + spread,
                minute_end=m_end, minute_up=m_up,
                minute_bid=m_bid, minute_ask=m_ask, mode="interp",
                interpolated=True)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def size_multiplier(sizing: SizingConfig, *, direction: int, rho: float,
                    trend: int, rvol: float) -> float:
    """``M`` for one fill, under whichever rule is configured."""
    lo, hi = sizing.clip
    if sizing.rule == "V10":
        return 1.0

    m_atr = 1.0 / rho if (rho and np.isfinite(rho) and rho > 0) else 1.0
    if trend == 0 or not np.isfinite(trend):
        m_trend = sizing.flat_mult
    elif direction == trend:
        m_trend = sizing.aligned_mult
    else:
        m_trend = sizing.opposed_mult

    if sizing.rule == "V11_ATR":
        m = m_atr
    elif sizing.rule == "V11_TREND":
        m = m_trend
    elif sizing.rule == "V11_VOLTGT":
        m = (sizing.target_vol_pct / rvol) if (rvol and np.isfinite(rvol)
                                               and rvol > 0) else 1.0
    else:                                    # V11
        m = m_atr * m_trend
    return float(np.clip(m, lo, hi))


def sharpe_invariance_check(returns, k: float = 3.7) -> tuple[float, float]:
    """Sharpe of a return series, and of the same series scaled by ``k``.

    They must agree to numerical precision. Any sizing rule that "improves"
    Sharpe while only rescaling every position is contradicted by this identity,
    and the harness prints it next to the sizing table so that the comparison is
    impossible to read without it.
    """
    from ..stats import sharpe
    r = np.asarray(returns, dtype=float)
    return sharpe(r), sharpe(r * k)


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def _first_true(mask: np.ndarray) -> int:
    if mask.size == 0:
        return -1
    i = int(mask.argmax())
    return i if mask[i] else -1


@dataclass
class RunOutput:
    """A closed trade list plus the bookkeeping the diagnostics need."""

    trades: pl.DataFrame
    equity: float
    starting_equity: float
    mode: str
    label: str
    n_signals: int = 0
    n_guard_rejects: int = 0
    n_expired: int = 0
    n_max_hold: int = 0
    """Positions closed by the hold deadline or by the end of the tape, rather
    than by their own stop or target."""
    meta: dict = field(default_factory=dict)


def _simulate(
    ctx: Context,
    tape: Tape,
    *,
    exits: ExitConfig,
    sizing: SizingConfig,
    starting_equity: float,
    slip: float,
    spec,
    account: Account,
    entries: np.ndarray | None = None,
) -> tuple[list[Trade], dict]:
    """Walk the H4 bars, arm the stop orders, and manage whatever fills.

    One position at a time, so a bar whose span lies inside an open position is
    skipped entirely rather than arming an order that a real account could not
    have placed.

    ``entries`` overrides the entry rule with an explicit array of
    ``(ts_us, direction)`` rows - which is how the random-entry Monte Carlo and
    the moving-average benchmark reuse this exact exit and sizing machinery
    instead of reimplementing it.
    """
    cfg = ctx.cfg
    h4 = ctx.h4
    bar_open = h4["ts_open"].dt.epoch("us").to_numpy().astype(np.int64)
    bar_end = h4["ts"].dt.epoch("us").to_numpy().astype(np.int64)
    bar_close = h4["close"].to_numpy()
    swing_hi = h4["swing_high"].to_numpy()
    swing_lo = h4["swing_low"].to_numpy()
    atr = h4["atr"].to_numpy()
    atr_fast = h4["atr_fast"].to_numpy()
    atr_slow = h4["atr_slow"].to_numpy()

    regime_days = np.array([d.toordinal() for d in ctx.regime["d"].to_list()])
    regime_trend = ctx.regime["trend"].to_numpy()
    regime_rvol = ctx.regime["rvol"].to_numpy()

    contract = spec.contract_size or 100.0
    commission_side = spec.commission_per_lot_side_usd or 0.0
    max_hold_us = int(cfg.max_hold_days * DAY_US)

    trades: list[Trade] = []
    stats = {"signals": 0, "guard": 0, "expired": 0, "max_hold": 0,
             "half_fired": 0}
    busy_until = -1
    # retest state: level, direction, the bar it was armed on, whether price has
    # pulled back through it yet.
    pending: tuple[float, int, int, bool] | None = None

    explicit = None
    if entries is not None and entries.size:
        explicit = entries[np.argsort(entries[:, 0])]
    explicit_i = 0

    for k in range(len(bar_open)):
        lo_us, hi_us = int(bar_open[k]), int(bar_end[k])
        if hi_us <= busy_until:
            continue
        scan_from = max(lo_us, busy_until)
        a = tape.index_at(scan_from)
        b = tape.index_at(hi_us)
        if b <= a:
            continue

        direction = 0
        fill_i = -1
        level = float("nan")

        if explicit is not None:
            # Benchmark or placebo entries: take everything scheduled inside
            # this bar's span that the account is not already busy through.
            while (explicit_i < len(explicit)
                   and explicit[explicit_i, 0] < scan_from):
                explicit_i += 1
            if (explicit_i < len(explicit) and explicit[explicit_i, 0] < hi_us):
                direction = int(explicit[explicit_i, 1])
                fill_i = tape.index_at(int(explicit[explicit_i, 0]))
                fill_i = min(max(fill_i, a), b - 1)
                explicit_i += 1
                level = float(tape.ask[fill_i] if direction == 1
                              else tape.bid[fill_i])
        else:
            hi_level, lo_level = swing_hi[k], swing_lo[k]
            ref = bar_close[k - 1] if k else bar_close[k]
            guard = cfg.atr_guard_mult * atr[k] if np.isfinite(atr[k]) else np.inf

            if cfg.arm == "on_confirm":
                buy_ok = (np.isfinite(hi_level)
                          and abs(hi_level - ref) <= guard)
                sell_ok = (np.isfinite(lo_level)
                           and abs(lo_level - ref) <= guard)
                if (np.isfinite(hi_level) and not buy_ok) or (
                        np.isfinite(lo_level) and not sell_ok):
                    stats["guard"] += 1
                ask, bid = tape.ask[a:b], tape.bid[a:b]
                i_buy = _first_true(ask >= hi_level) if buy_ok else -1
                i_sell = _first_true(bid <= lo_level) if sell_ok else -1
                if i_buy >= 0 or i_sell >= 0:
                    stats["signals"] += 1
                    if i_sell < 0 or (0 <= i_buy < i_sell):
                        direction, fill_i, level = 1, a + i_buy, float(hi_level)
                    else:
                        direction, fill_i, level = -1, a + i_sell, float(lo_level)
            else:                                    # retest
                if pending is not None:
                    p_level, p_dir, p_bar, eligible = pending
                    if k - p_bar > cfg.retest_expiry_bars:
                        stats["expired"] += 1
                        pending = None
                    else:
                        ask, bid = tape.ask[a:b], tape.bid[a:b]
                        if not eligible:
                            # Wait for the pullback back through the level.
                            i_back = (_first_true(bid <= p_level) if p_dir == 1
                                      else _first_true(ask >= p_level))
                            if i_back >= 0:
                                eligible = True
                                ask, bid = ask[i_back:], bid[i_back:]
                                base = a + i_back
                            else:
                                base = a
                            pending = (p_level, p_dir, p_bar, eligible)
                        else:
                            base = a
                        if eligible:
                            i_go = (_first_true(ask >= p_level) if p_dir == 1
                                    else _first_true(bid <= p_level))
                            if i_go >= 0:
                                stats["signals"] += 1
                                direction, fill_i = p_dir, base + i_go
                                level = p_level
                                pending = None
                if pending is None and direction == 0 and k:
                    # A close beyond the confirmed pivot arms the next order.
                    prev = bar_close[k - 1]
                    guard_ok = np.isfinite(atr[k])
                    if (np.isfinite(swing_hi[k]) and prev > swing_hi[k]
                            and (not guard_ok
                                 or abs(swing_hi[k] - prev) <= guard)):
                        pending = (float(swing_hi[k]), 1, k, False)
                    elif (np.isfinite(swing_lo[k]) and prev < swing_lo[k]
                          and (not guard_ok
                               or abs(swing_lo[k] - prev) <= guard)):
                        pending = (float(swing_lo[k]), -1, k, False)

        if direction == 0 or fill_i < 0:
            continue

        entry_px = (float(tape.ask[fill_i]) + slip if direction == 1
                    else float(tape.bid[fill_i]) - slip)
        entry_mid = float((tape.bid[fill_i] + tape.ask[fill_i]) / 2.0)
        entry_ts = int(tape.ts[fill_i])
        stop = entry_px - direction * cfg.delta_sl
        target = entry_px + direction * cfg.delta_tp

        day = datetime.fromtimestamp(entry_ts / US, tz=timezone.utc).date()
        j = int(np.searchsorted(regime_days, day.toordinal(), side="right")) - 1
        trend = int(regime_trend[j]) if 0 <= j < regime_trend.size else 0
        rvol = float(regime_rvol[j]) if 0 <= j < regime_rvol.size else float("nan")
        rho = (atr_fast[k] / atr_slow[k]
               if np.isfinite(atr_fast[k]) and np.isfinite(atr_slow[k])
               and atr_slow[k] > 0 else 1.0)
        mult = size_multiplier(sizing, direction=direction, rho=float(rho),
                               trend=trend, rvol=rvol)

        account.roll_to(entry_ts // DAY_US)
        lots = lots_for_risk(account.equity, sizing.risk_pct, entry_px, stop,
                             contract) * mult
        lots = round(lots, 2)
        if lots < 0.01:
            continue

        trade = Trade(
            symbol=spec.name, direction=direction,
            entry=Fill(entry_ts, entry_px, lots, "entry", entry_mid),
            stop_price=stop, contract_size=contract,
            commission_per_lot_side=commission_side,
            setup={"level": level, "bar": k, "mult": mult, "rho": float(rho),
                   "trend": trend, "rvol": rvol, "atr": float(atr[k]),
                   "half_fired": 0},
        )
        _manage(trade, tape, fill_i, entry_ts + max_hold_us, cfg, exits,
                slip, stats)
        if not trade.exits:
            continue
        trades.append(trade)
        account.apply(trade.net_usd)
        busy_until = trade.exits[-1].ts

    return trades, stats


def _manage(trade: Trade, tape: Tape, fill_i: int, deadline_us: int,
            cfg: BreakConfig, exits: ExitConfig, slip: float,
            stats: dict) -> None:
    """Take one filled position to its exits.

    Minute by minute: inside a minute the stop and the target are resting orders
    and resolve on the tape; at the minute's close, and only there, the half-exit
    gets its chance. A position that has taken its half runs on with the stop at
    breakeven.
    """
    d = trade.direction
    stop = trade.stop_price
    target = trade.entry.price + d * cfg.delta_tp
    open_lots = trade.entry.lots
    trigger = cfg.delta_sl * exits.half_exit_r
    half_done = False

    end_i = tape.index_at(deadline_us)
    m = tape.minute_index_at(int(tape.ts[fill_i]))
    i = fill_i

    mfe = mae = 0.0
    while i < end_i and m < tape.minute_end.size:
        m_end = int(tape.minute_end[m])
        j = min(tape.index_at(m_end), end_i)
        if j <= i:
            m += 1
            continue

        bid = tape.bid[i:j]
        ask = tape.ask[i:j]
        favour = (bid - trade.entry.price) * d if d == 1 else (trade.entry.price - ask)
        if favour.size:
            mfe = max(mfe, float(favour.max()))
            mae = min(mae, float(favour.min()))

        if d == 1:
            i_stop, i_tp = _first_true(bid <= stop), _first_true(bid >= target)
        else:
            i_stop, i_tp = _first_true(ask >= stop), _first_true(ask <= target)

        if i_stop >= 0 or i_tp >= 0:
            if i_tp < 0 or (0 <= i_stop <= i_tp):
                x, reason = i + i_stop, "stop" if not half_done else "be"
                touch = float(tape.bid[x]) if d == 1 else float(tape.ask[x])
                cross = stop if tape.interpolated else touch
                px = (cross - slip) if d == 1 else (cross + slip)
            else:
                x, reason = i + i_tp, "tp"
                px = target
            trade.exits.append(Fill(int(tape.ts[x]), px, open_lots, reason,
                                    float((tape.bid[x] + tape.ask[x]) / 2)))
            trade.mfe_price, trade.mae_price = mfe, mae
            return

        # The minute closed with the position still open: the overlay's turn.
        if exits.half_exit and not half_done and m < tape.minute_end.size:
            m_bid = float(tape.minute_bid[m])
            m_ask = float(tape.minute_ask[m])
            profit = (m_bid - trade.entry.price) * d if d == 1 else (
                trade.entry.price - m_ask)
            counter = (not tape.minute_up[m]) if d == 1 else bool(tape.minute_up[m])
            if profit >= trigger and (counter or not exits.require_counter_candle):
                part = round(open_lots * exits.half_fraction, 2)
                if 0 < part < open_lots:
                    px = (m_bid - slip) if d == 1 else (m_ask + slip)
                    trade.exits.append(Fill(m_end, px, part, "half",
                                            (m_bid + m_ask) / 2))
                    open_lots = round(open_lots - part, 2)
                    half_done = True
                    stats["half_fired"] += 1
                    trade.setup["half_fired"] = 1
                    if exits.breakeven_after_half:
                        stop = trade.entry.price
                        trade.stop_price = stop
        i = j
        m += 1

    # Neither barrier came due before the hold deadline or the end of the tape -
    # and the end of the tape is a real case, not an edge case, because the tick
    # window is clamped to the split so that a dev trade is never resolved with
    # validation data. Both land here and both are counted together; the run
    # reports the total so that a large one is visible rather than absorbed.
    x = min(max(i, fill_i), tape.ts.size - 1)
    px = (float(tape.bid[x]) - slip if d == 1 else float(tape.ask[x]) + slip)
    trade.exits.append(Fill(int(tape.ts[x]), px, open_lots, "time",
                            float((tape.bid[x] + tape.ask[x]) / 2)))
    trade.mfe_price, trade.mae_price = mfe, mae
    stats["max_hold"] += 1


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def _chunks(ctx: Context, months: int = 12):
    """Yield ``(start, end)`` windows over the context's H4 span."""
    first = ctx.h4["ts_open"].min()
    last = ctx.h4["ts"].max()
    edges = []
    cursor = datetime(first.year, first.month, 1, tzinfo=timezone.utc)
    while cursor <= last:
        edges.append(cursor)
        year, month = divmod((cursor.year * 12 + cursor.month - 1) + months, 12)
        cursor = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    edges.append(cursor)
    return list(zip(edges[:-1], edges[1:]))


def run(
    symbol: str = "XAUUSD",
    *,
    split: str | None = "dev",
    mode: str = "ticks",
    cfg: BreakConfig | None = None,
    exits: ExitConfig | None = None,
    sizing: SizingConfig | None = None,
    ctx: Context | None = None,
    starting_equity: float = 100_000.0,
    cost_model: CostModel | None = None,
    allow_test: bool = False,
    entries: np.ndarray | None = None,
    label: str | None = None,
    risk_multiple: float = 1.0,
    tape: Tape | None = None,
) -> RunOutput:
    """One cell of the grid: an entry rule, a sizing rule, an exit rule, a mode.

    ``risk_multiple`` is the specification's 1x/2x axis and multiplies the risk
    budget without touching anything else, which is precisely the manipulation
    :func:`sharpe_invariance_check` says cannot move a Sharpe ratio.

    ``tape`` reuses an already-built synthetic tape. Building one costs more
    than walking it, so the thousand-path random-entry Monte Carlo would spend
    almost all of its time rebuilding an identical price path; passing it in
    once makes that study affordable. It is only accepted for the bar-derived
    modes - the tick tape is chunked and far too large to hold whole.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, not in {MODES}")
    if tape is not None and mode == "ticks":
        raise ValueError("a prebuilt tape cannot be used in tick mode")
    cfg = cfg or BreakConfig()
    exits = exits or ExitConfig()
    sizing = sizing or SizingConfig()
    if risk_multiple != 1.0:
        sizing = replace(sizing, risk_pct=sizing.risk_pct * risk_multiple)

    ctx = ctx or build_context(symbol, split=split, cfg=cfg, sizing=sizing,
                               allow_test=allow_test)
    spec = get_spec(symbol)
    model = cost_model or CostModel.from_profiles(symbol, split=split)
    slip = model.slippage_pips() * spec.pip

    account = Account(equity=starting_equity, risk_pct=sizing.risk_pct,
                      daily_stop_pct=1.0)
    all_trades: list[Trade] = []
    totals = {"signals": 0, "guard": 0, "expired": 0, "max_hold": 0,
              "half_fired": 0}

    # A position opened near a chunk boundary has to be resolved with data from
    # the other side of it, so each tick window carries a tail. The tail is
    # clamped to the split's own last day: resolving a dev trade with validation
    # ticks would be a small leak, and resolving a validation trade with test
    # ticks would spend the locked split by accident. Positions still open at
    # the clamp exit as "time", and the count is reported.
    hard_end = None
    if split is not None:
        hard_end = (datetime.combine(get_split(split, allow_test=allow_test).end,
                                     datetime.min.time(), tzinfo=timezone.utc)
                    + timedelta(days=1))

    windows = _chunks(ctx) if mode == "ticks" else [(None, None)]
    for start, end in windows:
        if mode == "ticks":
            tail = end + timedelta(days=cfg.max_hold_days + 2)
            if hard_end is not None:
                tail = min(tail, hard_end)
                if start >= hard_end:
                    continue
            sub = ctx.h4.filter((pl.col("ts_open") >= pl.lit(start))
                                & (pl.col("ts_open") < pl.lit(end)))
            if sub.is_empty():
                continue
            bars = ctx.bars_1m.filter((pl.col("ts") >= pl.lit(start))
                                      & (pl.col("ts") < pl.lit(tail)))
            tape = tick_tape(symbol, bars, start, tail)
            window_ctx = Context(h4=sub, bars_1m=bars, regime=ctx.regime,
                                 cfg=cfg, sizing=sizing)
        else:
            tape = tape if tape is not None else synthetic_tape(ctx.bars_1m, mode)
            window_ctx = ctx
        if tape.ts.size == 0:
            continue
        trades, stats = _simulate(
            window_ctx, tape, exits=exits, sizing=sizing,
            starting_equity=starting_equity, slip=slip, spec=spec,
            account=account, entries=entries)
        all_trades.extend(trades)
        for key in totals:
            totals[key] += stats.get(key, 0)

    frame = (pl.DataFrame([t.to_dict() for t in all_trades])
             if all_trades else _empty_trades())
    if not frame.is_empty():
        frame = frame.sort("entry_ts").with_columns(
            r_multiple=pl.col("net_usd")
            / (cfg.delta_sl * (spec.contract_size or 100.0) * pl.col("lots")),
        )
    name = label or (f"{cfg.arm}/{sizing.rule}/{exits.label}/{mode}")
    return RunOutput(trades=frame, equity=account.equity,
                     starting_equity=starting_equity, mode=mode, label=name,
                     n_signals=totals["signals"], n_guard_rejects=totals["guard"],
                     n_expired=totals["expired"], n_max_hold=totals["max_hold"],
                     meta={"half_fired": totals["half_fired"],
                           "risk_pct": sizing.risk_pct, "slip_price": slip})


def _empty_trades() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "symbol": pl.String, "direction": pl.Int64, "entry_ts": pl.Int64,
        "entry_price": pl.Float64, "lots": pl.Float64, "stop_price": pl.Float64,
        "exit_ts": pl.Int64, "exit_price": pl.Float64, "exit_reason": pl.String,
        "n_exits": pl.Int64, "gross_usd": pl.Float64, "mid_pnl_usd": pl.Float64,
        "execution_cost_usd": pl.Float64, "commission_usd": pl.Float64,
        "slippage_usd": pl.Float64, "net_usd": pl.Float64, "hold_s": pl.Float64,
        "mae_price": pl.Float64, "mfe_price": pl.Float64,
    })


# --------------------------------------------------------------------------
# Benchmarks and placebos (B.5, Axis 1)
# --------------------------------------------------------------------------

def buy_and_hold(ctx: Context, *, starting_equity: float = 100_000.0,
                 symbol: str = "XAUUSD") -> dict:
    """Long the instrument for the whole window, sized once at the start.

    Reported as a return series rather than as trades, because it has no round
    trips and inventing one would put it in a table it does not belong in.
    """
    daily = (
        ctx.bars_1m.group_by(pl.col("ts_open").dt.date().alias("d"))
        .agg(c=pl.col("close").last(), n=pl.len())
        .filter(pl.col("n") >= 200).sort("d")
    )
    ret = (daily["c"] / daily["c"].shift(1) - 1.0).drop_nulls().to_numpy()
    return {"dates": daily["d"].to_list()[1:], "returns": ret,
            "label": f"buy and hold {symbol}"}


def ma_crossover_entries(ctx: Context, fast: int = 20, slow: int = 50) -> np.ndarray:
    """``(ts_us, direction)`` rows where an H4 moving-average crossover fires.

    Deliberately naive, and deliberately fed through the *same* stop, target,
    sizing and exit machinery as the structural break. The comparison is only
    informative if the entry is the sole difference.
    """
    close = ctx.h4["close"].to_numpy()
    ts = ctx.h4["ts"].dt.epoch("us").to_numpy().astype(np.int64)
    f = pl.Series(close).rolling_mean(fast).to_numpy()
    s = pl.Series(close).rolling_mean(slow).to_numpy()
    side = np.where(f > s, 1, np.where(f < s, -1, 0))
    cross = np.zeros_like(side)
    cross[1:] = np.where(side[1:] != side[:-1], side[1:], 0)
    idx = np.flatnonzero(cross != 0)
    idx = idx[np.isfinite(f[idx]) & np.isfinite(s[idx])]
    # The crossover is known at the bar's close, so it trades from that instant.
    return np.column_stack([ts[idx], cross[idx]]).astype(np.int64)


def random_entries(ctx: Context, n_trades: int, seed: int) -> np.ndarray:
    """A random-entry path with the same trade count and the same clock.

    Times are drawn from the H4 bar closes actually present in the sample, so
    the placebo inherits the strategy's session coverage and its holidays; only
    the *timing* and the *side* are random, which is what isolates the entry
    signal from everything the calendar contributes.
    """
    rng = np.random.default_rng(seed)
    ts = ctx.h4["ts"].dt.epoch("us").to_numpy().astype(np.int64)
    if n_trades <= 0 or ts.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pick = rng.choice(ts.size, size=min(n_trades, ts.size), replace=False)
    pick.sort()
    side = rng.choice([-1, 1], size=pick.size)
    return np.column_stack([ts[pick], side]).astype(np.int64)
