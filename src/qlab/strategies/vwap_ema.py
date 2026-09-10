"""The regime-filtered VWAP/EMA intraday strategy for gold, as specified by
Bhatti (2026).

The claim being tested
----------------------
*A Regime-Filtered Intraday Trading Framework for Gold: Integrating VWAP
Microstructure and EMA-Based Dynamic Exit Mechanisms* (SSRN 6650958) states a
complete, closed rule set on XAU/USD 15-minute candles. Its own Section 3 is
emphatic that nothing is left discretionary, and it is right - every condition
below is transcribed from the paper's equations rather than from its prose.

**Entry (long; short is the exact mirror).** All six must hold on the signal
candle:

=====  ====================================================================
C1     ``close > EMA200`` - the binary regime classifier. Skipped when the
       close is within 0.1% of the EMA (the paper's ambiguity band).
C2     ``close > VWAP``, VWAP anchored to the New York open (13:30 UTC in the
       paper's own words) and reset at 20:00 UTC.
C3     ``min(low_t, low_{t-1}) <= EMA50_t <= close_t`` - price touched or
       pierced the 50 EMA intrabar and closed back above it.
C4     A rejection candle: **pin bar** (lower wick >= 2x body and upper wick
       <= 0.5x lower wick) **or engulfing** (``C_t > O_{t-1}`` and
       ``O_t < C_{t-1}``).
C5     ``V_t > 1.1 x SMA20(V)``.
C6     ``H_t - L_t >= 0.8 x ATR14_t``.
=====  ====================================================================

**Exit.** Initial stop at ``L_signal - 0.5 x ATR14`` (that distance is 1R).
Target 3R. Between the two, a trailing stop that fires *only* on a candle
**close** beyond the 50 EMA on the adverse side - intrabar wicks are ignored,
which is the paper's headline mechanical contribution. Once floating profit
passes 2.5R the trail switches from the 50 EMA to the 20 EMA.

What this corpus can and cannot test
------------------------------------
**The paper has no backtest.** This is the single most important thing to know
before reading any number produced by this module. Its Section 6.1 says the 247
trades are *"calibrated to XAU/USD 15-minute data characteristics"* and that
outcomes are *"parameterised from the strategy's structural logic"* - full wins
with probability 0.30, partial wins 0.20, breakevens 0.08, losses 0.42. That is
a Monte Carlo draw from an assumed outcome distribution, not a measurement of
gold. Its Sharpe of 3.99 and its 45.3% win rate are therefore assumptions
restated as results, and nothing here can replicate them because there is
nothing to replicate. What *can* be done - and is what this module does - is run
the paper's rules, which are fully specified, against the actual tape.

**Volume is quote updates, not contracts.** The Exness feed quotes bid and ask
with no size, so C5 and the VWAP weights use ``n_ticks``: how many quote
revisions the broker published in the bar. This is not a workaround grafted on
to a stock strategy - it is exactly what MT5 reports as "volume" on a spot gold
chart, so an implementation on the paper's own stated platform would use the
same quantity. It is still a proxy and is labelled as one everywhere.

**The news filter is not implemented.** The paper excludes entries within 15
minutes of NFP, CPI, FOMC and GDP releases. This corpus holds no macro calendar,
so that filter is *absent*, which makes the results here slightly pessimistic
relative to the paper's intent. :func:`hour_breakdown` is the substitute: it
shows what the 13:30 and 14:00 UTC buckets - where those releases land - do to
the aggregate.

**The VWAP milestone protocol is unreachable.** Section 4.3 manages the trade
as price *approaches* VWAP after entry, reducing 50% on a "compressed approach".
But C2 requires price to already be beyond VWAP at entry, so for a long, VWAP is
below the fill and touching it again is adverse rather than a milestone. The two
sections contradict each other. Rather than pick a reading, this module records
``vwap_touch_r`` - the floating R at the first post-entry VWAP touch, or null -
so the report can say how often the protocol would fire at all and what it would
have cost. It is not in the trade path.

Why this is resolved on ticks
-----------------------------
1R is half a 14-period ATR of a 15-minute gold candle - on this corpus a median
of about $1.70, against 15-minute bars whose own range routinely exceeds that.
A bar-level engine containing both the entry and the stop would have to *assume*
which came first, and on this geometry that assumption is worth more than the
edge. Entry, stop and target are therefore resolved against the tick tape; only
the trail is a bar-close event, because the paper defines it as one.

Costs
-----
Spread is paid, not modelled: fills cross a real bid and a real ask. A long
enters at the ask and leaves at the bid, including at the 3R target - a sell
limit fills when the *bid* reaches it, not when the ask does. Commission comes
from the published contract terms. Slippage is the one assumed term and is taken
from :class:`qlab.costs.CostModel`, applied to the market orders (entry, stop,
trail, session flat) and never to the target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np
import polars as pl

from ..bars import resample_bars
from ..costs import CostModel
from ..loader import load_bars, load_ticks
from ..symbols import get_spec

NY = "America/New_York"
US = 1_000_000  # microseconds in a second

#: Stop widths to sweep, as a multiple of ATR14. 0.5 is the paper's.
STOP_MULTS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

#: Profit targets to sweep, in R. 3.0 is the paper's; ``None`` removes the
#: target and leaves the trail as the only way out.
TARGET_RS: tuple[float | None, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, None)

HOLD = "hold"


def tag(target: float | None) -> str:
    return HOLD if target is None else f"t{target:g}".replace(".", "_")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VWAPEmaConfig:
    """Everything that changes a decision.

    ``anchor``
        ``"utc"`` anchors VWAP at a fixed 13:30 UTC, which is what the paper
        literally writes. ``"ny"`` anchors it at 09:30 America/New_York, which
        is what "the New York session open" actually means and which differs by
        an hour for the five winter months. The paper cannot have both; the
        default follows its stated number and the report tests the other.

    ``entry_pattern``
        ``"either"`` is C4 as written. ``"pin"`` and ``"engulf"`` isolate the two
        halves; ``"any"`` drops C4 entirely, which is the control that says how
        much work the rejection candle is doing.

    ``side``
        ``"both"`` is the paper. ``"long"`` / ``"short"`` isolate one direction,
        which matters on an instrument that rose through most of the sample.
    """

    interval: str = "15m"
    regime_ema: int = 200
    trail_ema: int = 50
    final_ema: int = 20
    atr_period: int = 14
    vol_ma: int = 20

    regime_band_pct: float = 0.1      # the 0.1% ambiguity zone around EMA200
    vol_mult: float = 1.1             # C5
    range_atr_frac: float = 0.8       # C6
    pin_body_mult: float = 2.0        # C4, pin bar
    pin_upper_frac: float = 0.5       # C4, pin bar
    entry_pattern: str = "either"     # "either" | "pin" | "engulf" | "any"
    side: str = "both"                # "both" | "long" | "short"

    stop_atr_mult: float = 0.5        # Eq. 4
    target_r: float | None = 3.0
    use_trail: bool = True
    final_leg_r: float = 2.5          # Eq. 6

    session_start_min: int = 13 * 60 + 30    # the paper's 13:30 UTC
    session_end_min: int = 20 * 60           # the paper's 20:00 UTC reset
    anchor: str = "utc"                      # "utc" | "ny"
    close_at_session_end: bool = True
    max_hold_bars: int = 26                  # one 6.5h session of 15m bars

    def __post_init__(self) -> None:
        if self.entry_pattern not in ("either", "pin", "engulf", "any"):
            raise ValueError(f"unknown entry_pattern {self.entry_pattern!r}")
        if self.side not in ("both", "long", "short"):
            raise ValueError(f"unknown side {self.side!r}")
        if self.anchor not in ("utc", "ny"):
            raise ValueError(f"unknown anchor {self.anchor!r}")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be positive - it defines 1R")
        if self.session_start_min >= self.session_end_min:
            raise ValueError("the VWAP session must have positive length")

    @property
    def sides(self) -> tuple[int, ...]:
        return {"both": (1, -1), "long": (1,), "short": (-1,)}[self.side]


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def _ema(column: str, span: int) -> pl.Expr:
    """The recursive EMA of Eq. 2, alpha = 2/(n+1), seeded on the first bar.

    ``adjust=False`` is the difference between a chart's EMA and a
    weighted-mean-of-everything; the paper writes the recursion, so this is it.
    """
    return pl.col(column).ewm_mean(span=span, adjust=False)


def indicators(
    symbol: str = "XAUUSD",
    cfg: VWAPEmaConfig = VWAPEmaConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
) -> pl.DataFrame:
    """15-minute bars with every indicator the rules read.

    The moving averages and the ATR run over the **whole** 15-minute tape, not
    just the session window, because that is what an EMA200 on a gold chart is:
    a 200-bar recursion over the bars the chart shows. VWAP is the exception -
    it is cumulative *within* a session by definition, and resets at the paper's
    20:00 UTC boundary.

    Everything here is known at the bar's own close, so nothing is lagged: with
    right-edge bar labels, reading row ``i`` at ``ts[i]`` uses only information
    that had printed by ``ts[i]``.
    """
    # 200 fifteen-minute bars is 50 hours of tape, but gold trades ~5 days a
    # week with a nightly break, so the warmup has to be wall-clock generous.
    warmup = timedelta(days=40)
    minute = load_bars(
        symbol, "1m", split=split, start=start, end=end, warmup=warmup,
        allow_test=allow_test,
        columns=["ts", "ts_open", "open", "high", "low", "close",
                 "bid_close", "ask_close", "spread_mean", "n_ticks",
                 "first_tick_ts", "last_tick_ts", "mid_mean", "spread_close"],
    )
    warm_from = None
    if "is_warmup" in minute.columns:
        warm_from = minute.filter(~pl.col("is_warmup"))["ts_open"].min()
        minute = minute.drop("is_warmup")

    bars = resample_bars(minute, cfg.interval).sort("ts_open")

    ny = pl.col("ts_open").dt.convert_time_zone(NY)
    bars = bars.with_columns(
        utc_min=pl.col("ts_open").dt.hour().cast(pl.Int32) * 60
        + pl.col("ts_open").dt.minute().cast(pl.Int32),
        ny_min=ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32),
        ny_date=ny.dt.date(),
        utc_date=pl.col("ts_open").dt.date(),
    )

    clock = pl.col("utc_min") if cfg.anchor == "utc" else pl.col("ny_min")
    stamp = pl.col("utc_date") if cfg.anchor == "utc" else pl.col("ny_date")
    in_session = (clock >= cfg.session_start_min) & (clock < cfg.session_end_min)

    bars = bars.with_columns(
        in_session=in_session,
        session=pl.when(in_session).then(stamp).otherwise(None),
        # The paper's Eq. 1 typical price. `volume` is the tick count - see the
        # module docstring; it is named `volume` here because that is the role
        # it plays in every formula that reads it.
        typical=(pl.col("high") + pl.col("low") + pl.col("close")) / 3.0,
        volume=pl.col("n_ticks").cast(pl.Float64),
    )

    # Both of these are session-relative and therefore meaningless outside one.
    # `over("session")` would otherwise lump every out-of-session bar in the
    # corpus into a single null group and run a four-year cumulative sum through
    # it, which is not wrong anywhere it is read but looks wrong everywhere.
    bars = bars.with_columns(
        vwap=pl.when(in_session).then(
            (pl.col("typical") * pl.col("volume")).cum_sum().over("session")
            / pl.col("volume").cum_sum().over("session")
        ),
        session_bar=pl.when(in_session).then(pl.int_range(pl.len()).over("session")),
        ema_regime=_ema("close", cfg.regime_ema),
        ema_trail=_ema("close", cfg.trail_ema),
        ema_final=_ema("close", cfg.final_ema),
        vol_ma=pl.col("volume").rolling_mean(cfg.vol_ma),
        prev_close=pl.col("close").shift(1),
        prev_open=pl.col("open").shift(1),
        prev_low=pl.col("low").shift(1),
        prev_high=pl.col("high").shift(1),
    )

    bars = bars.with_columns(
        tr=pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        )
    ).with_columns(
        # A plain mean of true range over 14 bars: "the 14-period Average True
        # Range", read literally, and what MT5's iATR reports.
        atr=pl.col("tr").rolling_mean(cfg.atr_period),
        body=(pl.col("close") - pl.col("open")).abs(),
        upper_wick=pl.col("high") - pl.max_horizontal("open", "close"),
        lower_wick=pl.min_horizontal("open", "close") - pl.col("low"),
    )

    if warm_from is not None:
        bars = bars.filter(pl.col("ts_open") >= warm_from)
    return bars


# --------------------------------------------------------------------------
# The six entry conditions
# --------------------------------------------------------------------------

def signals(bars: pl.DataFrame, cfg: VWAPEmaConfig = VWAPEmaConfig()) -> pl.DataFrame:
    """Evaluate C1-C6 on every bar and mark the ones that fire.

    Each condition is kept as its own boolean column rather than collapsed into
    one flag, because the only interesting question about a six-condition filter
    is which condition is doing the work - see :func:`condition_attrition`.
    """
    band = cfg.regime_band_pct / 100.0

    # C1. Regime, with the ambiguity band excluded on both sides.
    off_ema = (pl.col("close") - pl.col("ema_regime")).abs() / pl.col("ema_regime")
    c1_long = (pl.col("close") > pl.col("ema_regime")) & (off_ema > band)
    c1_short = (pl.col("close") < pl.col("ema_regime")) & (off_ema > band)

    # C2. The institutional side of VWAP.
    c2_long = pl.col("close") > pl.col("vwap")
    c2_short = pl.col("close") < pl.col("vwap")

    # C3. Touched or pierced the 50 EMA intrabar, closed back on the right side.
    c3_long = (
        pl.min_horizontal("low", "prev_low") <= pl.col("ema_trail")
    ) & (pl.col("ema_trail") <= pl.col("close"))
    c3_short = (
        pl.max_horizontal("high", "prev_high") >= pl.col("ema_trail")
    ) & (pl.col("ema_trail") >= pl.col("close"))

    # C4. Rejection candle, as two alternatives.
    pin_long = (
        pl.col("lower_wick") >= cfg.pin_body_mult * pl.col("body")
    ) & (pl.col("upper_wick") <= cfg.pin_upper_frac * pl.col("lower_wick"))
    pin_short = (
        pl.col("upper_wick") >= cfg.pin_body_mult * pl.col("body")
    ) & (pl.col("lower_wick") <= cfg.pin_upper_frac * pl.col("upper_wick"))
    # Transcribed from the paper's inequalities, not from its parenthetical
    # gloss: C_t > O_{t-1} and O_t < C_{t-1}. When the prior candle is bearish
    # these two imply the current one is bullish, so the gloss is a consequence
    # rather than an extra condition, and adding it would change the rule.
    engulf_long = (pl.col("close") > pl.col("prev_open")) & (
        pl.col("open") < pl.col("prev_close")
    )
    engulf_short = (pl.col("close") < pl.col("prev_open")) & (
        pl.col("open") > pl.col("prev_close")
    )
    if cfg.entry_pattern == "pin":
        c4_long, c4_short = pin_long, pin_short
    elif cfg.entry_pattern == "engulf":
        c4_long, c4_short = engulf_long, engulf_short
    elif cfg.entry_pattern == "any":
        c4_long = c4_short = pl.lit(True)
    else:
        c4_long, c4_short = pin_long | engulf_long, pin_short | engulf_short

    # C5. Above-average volume - the tick-count proxy.
    c5 = pl.col("volume") > cfg.vol_mult * pl.col("vol_ma")

    # C6. Above-average range.
    c6 = (pl.col("high") - pl.col("low")) >= cfg.range_atr_frac * pl.col("atr")

    out = bars.with_columns(
        c1_long=c1_long, c1_short=c1_short,
        c2_long=c2_long, c2_short=c2_short,
        c3_long=c3_long, c3_short=c3_short,
        c4_long=c4_long, c4_short=c4_short,
        c5=c5, c6=c6,
        pin_long=pin_long, pin_short=pin_short,
        engulf_long=engulf_long, engulf_short=engulf_short,
    )

    tradeable = (
        pl.col("in_session")
        & pl.col("atr").is_not_null()
        & (pl.col("atr") > 0)
        & pl.col("vol_ma").is_not_null()
        & pl.col("vwap").is_not_null()
        # The signal candle needs a bar after it to hold the position in, and
        # the last bar of the session has none.
        & (pl.col("utc_min") if cfg.anchor == "utc" else pl.col("ny_min")).lt(
            cfg.session_end_min - _interval_minutes(cfg.interval)
        )
    )
    long_ok = tradeable & pl.col("c1_long") & pl.col("c2_long") & pl.col(
        "c3_long") & pl.col("c4_long") & pl.col("c5") & pl.col("c6")
    short_ok = tradeable & pl.col("c1_short") & pl.col("c2_short") & pl.col(
        "c3_short") & pl.col("c4_short") & pl.col("c5") & pl.col("c6")
    if 1 not in cfg.sides:
        long_ok = pl.lit(False)
    if -1 not in cfg.sides:
        short_ok = pl.lit(False)

    return out.with_columns(
        tradeable=tradeable,
        # A bar cannot satisfy both: C1 alone puts the close on one side of the
        # 200 EMA. The `otherwise` is a guard, not a tie-break.
        signal=pl.when(long_ok).then(1).when(short_ok).then(-1).otherwise(0)
        .cast(pl.Int8),
    )


def _interval_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return 60 * int(interval[:-1])
    raise ValueError(f"cannot read an interval from {interval!r}")


def condition_attrition(
    bars: pl.DataFrame, cfg: VWAPEmaConfig = VWAPEmaConfig()
) -> pl.DataFrame:
    """How many tradeable bars survive each condition, alone and cumulatively.

    A six-condition AND can be carried by one condition and decorated by five,
    and the aggregate result will never say which. This does.
    """
    if bars.is_empty():
        return pl.DataFrame()
    base = bars.filter(pl.col("tradeable"))
    n = base.height
    rows = []
    for side, suffix in ((1, "long"), (-1, "short")):
        if side not in cfg.sides:
            continue
        names = [f"c{i}_{suffix}" for i in (1, 2, 3, 4)] + ["c5", "c6"]
        running = pl.Series([True] * n)
        for i, name in enumerate(names, start=1):
            col = base[name] if name in base.columns else pl.Series([True] * n)
            running = running & col
            rows.append({
                "side": suffix,
                "condition": f"C{i}",
                "alone": int(col.sum()),
                "alone_pct": 100.0 * float(col.mean()),
                "cumulative": int(running.sum()),
                "cumulative_pct": 100.0 * float(running.mean()),
            })
    return pl.DataFrame(rows).with_columns(tradeable_bars=pl.lit(n))


# --------------------------------------------------------------------------
# Tick-level simulation
# --------------------------------------------------------------------------

def _first_true(mask: np.ndarray) -> int:
    if mask.size == 0:
        return -1
    i = int(mask.argmax())
    return i if mask[i] else -1


@dataclass
class _Tape:
    ts: np.ndarray   # int64 microseconds, ascending
    bid: np.ndarray
    ask: np.ndarray

    def index_at(self, when_us: int) -> int:
        return int(np.searchsorted(self.ts, when_us, side="left"))


@dataclass
class _Entry:
    i: int
    direction: int
    price: float
    mid: float
    ts_us: int


@dataclass
class _Window:
    """The bars a trade may live through, and what the trail reads on each."""

    close: np.ndarray        # bar close, mid
    ema_trail: np.ndarray
    ema_final: np.ndarray
    vwap: np.ndarray
    end_us: np.ndarray       # each bar's right edge, in microseconds
    last_us: int             # when the position is flattened at the latest


def _entry_at(tape: _Tape, when_us: int, direction: int, slip: float) -> _Entry | None:
    """Fill a market order at the first quote at or after the signal close.

    The signal is knowable only once the candle has closed, so this is the
    earliest honest moment to be in the trade. A buy pays the ask.
    """
    i = tape.index_at(when_us)
    if i >= tape.ts.size:
        return None
    price = (tape.ask[i] + slip) if direction == 1 else (tape.bid[i] - slip)
    return _Entry(
        i=i, direction=direction, price=float(price),
        mid=float((tape.bid[i] + tape.ask[i]) / 2), ts_us=int(tape.ts[i]),
    )


def _simulate(
    tape: _Tape, ent: _Entry, win: _Window, risk: float,
    target_r: float | None, cfg: VWAPEmaConfig, slip: float,
) -> dict:
    """One (stop, target, trail) geometry on one already-resolved entry.

    Three clocks race and the earliest wins: the stop and the target run on
    ticks, the trail runs on bar closes, and the session flat runs on the clock.
    Ties inside a single tick are resolved against the trade.
    """
    d, entry = ent.direction, ent.price
    stop = entry - d * risk
    target = None if target_r is None else entry + d * target_r * risk

    a = ent.i
    b = max(tape.index_at(win.last_us), a + 1)
    bid, ask, ts = tape.bid[a:b], tape.ask[a:b], tape.ts[a:b]
    if ts.size == 0:
        return {"exit_reason": "no_tape"}

    # A long is closed by selling: both its stop and its target are reached
    # when the *bid* gets there. The ask is what it paid to get in and has no
    # say in how it gets out.
    if d == 1:
        i_stop = _first_true(bid <= stop)
        i_tgt = -1 if target is None else _first_true(bid >= target)
    else:
        i_stop = _first_true(ask >= stop)
        i_tgt = -1 if target is None else _first_true(ask <= target)

    # The trail: the first bar close beyond the active EMA on the adverse side.
    # The active EMA ratchets to the faster one once the *closed-bar* floating
    # profit has passed the paper's 2.5R, which is the only version of "floating
    # P&L" a close-conditioned rule can see.
    i_trail_bar = -1
    if cfg.use_trail and win.close.size:
        floating = d * (win.close - entry) / risk
        # Once passed, stay passed - Eq. 6 switches the trail, it does not
        # oscillate with every bar that dips back under 2.5R.
        tightened = np.maximum.accumulate(floating) >= cfg.final_leg_r
        level = np.where(tightened, win.ema_final, win.ema_trail)
        adverse = (win.close < level) if d == 1 else (win.close > level)
        i_trail_bar = _first_true(adverse)

    t_stop = int(ts[i_stop]) if i_stop >= 0 else None
    t_tgt = int(ts[i_tgt]) if i_tgt >= 0 else None
    t_trail = int(win.end_us[i_trail_bar]) if i_trail_bar >= 0 else None

    # Order of precedence at equal timestamps: stop, then trail, then target.
    # A stop and a target inside one tick is a path the tick does not record,
    # so it is resolved against the trade; a trail firing on the same close as
    # a target is the same argument one level up.
    best, reason = None, None
    for when, name in ((t_stop, "stop"), (t_trail, "trail"), (t_tgt, "target")):
        if when is not None and (best is None or when < best):
            best, reason = when, name

    if best is None or best > win.last_us:
        reason = "flat"
        exit_i = int(ts.size - 1)
    else:
        exit_i = int(np.searchsorted(ts, best, side="left"))
        exit_i = min(exit_i, ts.size - 1)

    if reason == "target":
        exit_price = float(target)          # a limit order: no slippage
    elif d == 1:
        exit_price = float(bid[exit_i] - slip)
    else:
        exit_price = float(ask[exit_i] + slip)

    mid_exit = float((bid[exit_i] + ask[exit_i]) / 2)
    pnl = d * (exit_price - entry)
    return {
        "exit_price": exit_price,
        "exit_ts_us": int(ts[exit_i]),
        "exit_reason": reason,
        "hold_min": (int(ts[exit_i]) - ent.ts_us) / (60 * US),
        "stop": float(stop),
        "target_price": float("nan") if target is None else float(target),
        "risk": float(risk),
        "target_r": float("nan") if target_r is None else float(target_r),
        "target": tag(target_r),
        "pnl": float(pnl),
        "r_multiple": float(pnl / risk) if risk > 0 else float("nan"),
        "mid_pnl": float(d * (mid_exit - ent.mid)),
        "mfe_r": float(
            (np.max(d * (bid[: exit_i + 1] - entry)) if d == 1
             else np.max(d * (ask[: exit_i + 1] - entry))) / risk
        ) if risk > 0 else float("nan"),
        "mae_r": float(
            (np.min(d * (bid[: exit_i + 1] - entry)) if d == 1
             else np.min(d * (ask[: exit_i + 1] - entry))) / risk
        ) if risk > 0 else float("nan"),
    }


def _vwap_touch(win: _Window, ent: _Entry, risk: float) -> float:
    """Floating R at the first post-entry bar that trades back through VWAP.

    Section 4.3's milestone protocol, measured rather than applied - see the
    module docstring for why it cannot be applied as written.
    """
    if not win.close.size or risk <= 0:
        return float("nan")
    d = ent.direction
    touched = (win.close <= win.vwap) if d == 1 else (win.close >= win.vwap)
    i = _first_true(touched)
    if i < 0:
        return float("nan")
    return float(d * (win.close[i] - ent.price) / risk)


def _months(frame: pl.DataFrame, column: str = "ts_open") -> list[tuple[int, int]]:
    return (
        frame.select(y=pl.col(column).dt.year(), m=pl.col(column).dt.month())
        .unique().sort("y", "m").rows()
    )


def run(
    symbol: str = "XAUUSD",
    cfg: VWAPEmaConfig = VWAPEmaConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    stop_mults: Sequence[float] | None = None,
    targets: Sequence[float | None] | None = None,
    costs: CostModel | None = None,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Backtest the rule set, month by month over the ticks.

    Returns one row per (signal, stop multiple, target). The entry is resolved
    **once** per signal and shared by every geometry in the sweep, so no cell of
    the exit surface can accidentally be handed a different fill from its
    neighbour. Note that the stop multiple *does* change 1R and therefore the
    trail's 2.5R switch, which is why it is swept here rather than rescaled
    afterwards.
    """
    spec = get_spec(symbol)
    stop_mults = (cfg.stop_atr_mult,) if stop_mults is None else tuple(stop_mults)
    targets = (cfg.target_r,) if targets is None else tuple(targets)

    if bars is None:
        bars = indicators(symbol, cfg, split=split, start=start, end=end,
                          allow_test=allow_test)
        bars = signals(bars, cfg)
    if "signal" not in bars.columns:
        bars = signals(bars, cfg)

    fired = bars.filter(pl.col("signal") != 0)
    if fired.is_empty():
        return pl.DataFrame()

    costs = costs or CostModel.from_profiles(symbol, split=split or "dev")
    slip = costs.slippage_pips(hour=None) * spec.pip

    # Arrays for the forward window, indexed the same way as `bars`.
    idx = {ts: i for i, ts in enumerate(bars["ts"].to_list())}
    close_a = bars["close"].to_numpy()
    trail_a = bars["ema_trail"].to_numpy()
    final_a = bars["ema_final"].to_numpy()
    vwap_a = bars["vwap"].to_numpy()
    end_us_a = bars["ts"].to_numpy().astype("datetime64[us]").astype(np.int64)
    clock_a = bars["utc_min" if cfg.anchor == "utc" else "ny_min"].to_numpy()
    sess_a = bars["session"].to_list()

    records: list[dict] = []
    for year, month in _months(fired):
        rows = fired.filter(
            (pl.col("ts_open").dt.year() == year)
            & (pl.col("ts_open").dt.month() == month)
        )
        if rows.is_empty():
            continue
        lo = rows["ts_open"].min().date() - timedelta(days=1)
        hi = rows["ts_open"].max().date() + timedelta(days=3)
        ticks = load_ticks(symbol, start=lo, end=hi, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(
            ts=ticks["ts"].to_numpy().astype("datetime64[us]").astype(np.int64),
            bid=ticks["bid"].to_numpy(),
            ask=ticks["ask"].to_numpy(),
        )

        for row in rows.iter_rows(named=True):
            i = idx[row["ts"]]
            direction = int(row["signal"])
            ent = _entry_at(tape, end_us_a[i], direction, slip)
            if ent is None:
                continue

            # Forward bars: same session while the session rule is on, capped by
            # max_hold_bars either way.
            j = i + 1
            k = min(i + 1 + cfg.max_hold_bars, bars.height)
            if cfg.close_at_session_end:
                while k > j and (sess_a[k - 1] != row["session"]):
                    k -= 1
            if k <= j:
                continue
            win = _Window(
                close=close_a[j:k], ema_trail=trail_a[j:k], ema_final=final_a[j:k],
                vwap=vwap_a[j:k], end_us=end_us_a[j:k], last_us=int(end_us_a[k - 1]),
            )

            base = {
                "symbol": symbol,
                "ts": row["ts"],
                "session": row["session"],
                "direction": direction,
                "hour": int(clock_a[i]) // 60,
                "session_bar": int(row["session_bar"]),
                "atr": float(row["atr"]),
                "bar_range": float(row["high"] - row["low"]),
                "rel_volume": float(row["volume"] / row["vol_ma"]),
                "dist_vwap_bps": 1e4 * (row["close"] - row["vwap"]) / row["close"],
                "dist_ema200_bps": 1e4 * (row["close"] - row["ema_regime"])
                / row["close"],
                "is_pin": bool(row["pin_long" if direction == 1 else "pin_short"]),
                "is_engulf": bool(
                    row["engulf_long" if direction == 1 else "engulf_short"]),
                "entry": ent.price,
                "entry_mid": ent.mid,
                "bars_available": int(k - j),
            }
            commission = costs.commission_pips(price=ent.mid) * spec.pip
            touch_cache: dict[float, float] = {}

            for mult in stop_mults:
                # Eq. 4: the stop sits below the *signal candle's* low, so 1R is
                # that whole distance from the fill, not the ATR term alone.
                anchor = row["low"] if direction == 1 else row["high"]
                stop_price = anchor - direction * mult * row["atr"]
                risk = direction * (ent.price - stop_price)
                if not (risk > 0):
                    continue
                if mult not in touch_cache:
                    touch_cache[mult] = _vwap_touch(win, ent, risk)
                for target_r in targets:
                    sim = _simulate(tape, ent, win, risk, target_r, cfg, slip)
                    if sim.get("exit_reason") == "no_tape":
                        continue
                    rec = base | sim
                    rec["stop_mult"] = float(mult)
                    rec["vwap_touch_r"] = touch_cache[mult]
                    rec["commission"] = commission
                    rec["net_pnl"] = rec["pnl"] - commission
                    rec["net_r"] = rec["net_pnl"] / risk
                    rec["cost_r"] = (rec["mid_pnl"] - rec["net_pnl"]) / risk
                    rec["net_bps"] = 1e4 * rec["net_pnl"] / ent.mid
                    rec["mid_bps"] = 1e4 * rec["mid_pnl"] / ent.mid
                    rec["cost_bps"] = rec["mid_bps"] - rec["net_bps"]
                    records.append(rec)
    return pl.DataFrame(records) if records else pl.DataFrame()


# --------------------------------------------------------------------------
# Views the report needs
# --------------------------------------------------------------------------

def exit_surface(
    trades: pl.DataFrame, *, column: str = "net_r"
) -> pl.DataFrame:
    """Mean outcome for every (stop multiple, target) cell of the sweep.

    The paper fixes one cell of this grid in advance - 0.5 ATR and 3R - so where
    that cell sits on the surface is evidence about the result. On a broad
    plateau it is a real geometry; at an isolated peak it is a fitted one.
    """
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.group_by("stop_mult", "target")
        .agg(
            n=pl.len(),
            mean=pl.col(column).mean(),
            sd=pl.col(column).std(),
            hit_pct=100.0 * (pl.col(column) > 0).mean(),
            target_r=pl.col("target_r").first(),
        )
        .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt())
        .sort("stop_mult", "target_r", nulls_last=True)
    )


def hour_breakdown(trades: pl.DataFrame, *, column: str = "net_r") -> pl.DataFrame:
    """Per-trade outcome by UTC hour of entry.

    The stand-in for the news filter this corpus cannot implement: NFP, CPI and
    FOMC land in the 13:30 and 18:00-19:00 UTC buckets, so if excluding news
    would have mattered, it has to show up as those hours dragging.
    """
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.group_by("hour")
        .agg(
            n=pl.len(),
            mean=pl.col(column).mean(),
            sd=pl.col(column).std(),
            hit_pct=100.0 * (pl.col(column) > 0).mean(),
        )
        .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt())
        .sort("hour")
    )


def outcome_table(trades: pl.DataFrame) -> pl.DataFrame:
    """The paper's Table 4: how trades end, how often, and at what average R.

    Its four rows are full win, partial win via the trail, breakeven and full
    loss. The mapping here is by exit reason and R, so the two tables are
    directly comparable.
    """
    if trades.is_empty():
        return pl.DataFrame()
    bucket = (
        pl.when(pl.col("exit_reason") == "target").then(pl.lit("full win"))
        .when(pl.col("net_r") > 0.2).then(pl.lit("partial win"))
        .when(pl.col("net_r") >= -0.2).then(pl.lit("breakeven"))
        .otherwise(pl.lit("loss"))
    )
    order = {"full win": 0, "partial win": 1, "breakeven": 2, "loss": 3}
    return (
        trades.with_columns(outcome=bucket)
        .group_by("outcome")
        .agg(n=pl.len(), mean_r=pl.col("net_r").mean())
        .with_columns(
            pct=100.0 * pl.col("n") / trades.height,
            _o=pl.col("outcome").replace_strict(order, return_dtype=pl.Int32),
        )
        .sort("_o").drop("_o")
    )


def exit_reason_table(trades: pl.DataFrame) -> pl.DataFrame:
    """Which of the three clocks ended each trade, and what it paid."""
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.group_by("exit_reason")
        .agg(
            n=pl.len(),
            mean_r=pl.col("net_r").mean(),
            mean_hold_min=pl.col("hold_min").mean(),
        )
        .with_columns(pct=100.0 * pl.col("n") / trades.height)
        .sort("n", descending=True)
    )


def placebo_signals(
    bars: pl.DataFrame, *, seed: int, cfg: VWAPEmaConfig = VWAPEmaConfig()
) -> pl.DataFrame:
    """The same number of signals, on tradeable bars, at random times.

    Keeps the instrument, the session window, the trade count and the sizing
    rule, and destroys only the six conditions. Anything the placebo also
    produces is produced by gold's drift and the exit geometry, not by the
    entry - which is the whole content of a six-condition filter's claim.
    """
    rng = np.random.default_rng(seed)
    real = bars["signal"].to_numpy()
    n_long = int((real == 1).sum())
    n_short = int((real == -1).sum())
    pool = np.flatnonzero(bars["tradeable"].to_numpy())
    take = min(n_long + n_short, pool.size)
    if take == 0:
        return bars.with_columns(signal=pl.zeros(bars.height, dtype=pl.Int8, eager=True))
    picked = rng.choice(pool, size=take, replace=False)
    fake = np.zeros(bars.height, dtype=np.int8)
    sides = np.array([1] * n_long + [-1] * n_short, dtype=np.int8)[:take]
    rng.shuffle(sides)
    fake[picked] = sides
    return bars.with_columns(signal=pl.Series("signal", fake, dtype=pl.Int8))


def with_config(cfg: VWAPEmaConfig, **changes) -> VWAPEmaConfig:
    """A copy of a config with fields replaced - for sweeps that vary one rule."""
    return replace(cfg, **changes)


#: 09:30 and 16:00 New York, in minutes - what "the New York session open"
#: means on the days it is not also 13:30 UTC.
NY_OPEN_MIN = 9 * 60 + 30
NY_CLOSE_MIN = 16 * 60


def ny_anchored(cfg: VWAPEmaConfig) -> VWAPEmaConfig:
    """The same rules with VWAP anchored to the actual New York open.

    Switching ``anchor`` alone is not enough and would silently produce a
    different study: the window bounds are read on whichever clock the anchor
    names, so leaving them at 13:30-20:00 while switching to New York time asks
    for 13:30-20:00 *New York*, which is the US afternoon and evening rather
    than its session. This moves the bounds with the clock.
    """
    return replace(
        cfg, anchor="ny",
        session_start_min=NY_OPEN_MIN, session_end_min=NY_CLOSE_MIN,
    )
