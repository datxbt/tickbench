"""The engulfing bar traded from its own quadrants: sell the retracement, target the low.

The setup, stated with no room left for interpretation - one bar, two
conditions, both measured against the bar before it:

* **Takes out the prior high**  ``high[0] > high[1]``
* **Closes below the prior open**  ``close[0] < open[1]``

Note what this is *not*. It is not the body-engulfing candle already rejected in
:mod:`qlab.strategies.engulfing`: condition 1 is about the **wick**, so the bar
must physically trade through the previous bar's high before turning, and
condition 2 references the previous **open** rather than its body's far edge.
The two conditions together describe a stop run - highs taken, then given back
past the point the prior bar started from - and roughly half the time the prior
bar is not even bullish. That difference is the reason this gets its own module
and its own tape walk rather than a flag on the old one.

The quadrants
-------------
The signal bar is divided into four equal quarters by the fibonacci ladder
``0, 0.25, 0.5, 0.75, 1``, anchored so that **0 is the target extreme and 1 is
the stop extreme** - for the bearish setup, 0 is the bar's low and 1 is its
high. Every price this strategy cares about is one of those levels:

* **Entry** - the ``0.25-0.5`` zone, worked as a resting order.
* **Target** - ``0``, the bar's low, as specified. Extensions below it are
  swept alongside because the idea nominates one target and one target only,
  and a single nominated exit is a hypothesis, not a measurement.
* **Stop** - unspecified by the idea, so it is swept: ``0.75``, ``1`` (the
  bar's own high, the house convention) and ``1.25`` (a quarter-range buffer
  above it).

What the geometry costs before any data is looked at
----------------------------------------------------
This is the first thing to state, because it constrains everything after it.
Entry between 0.25 and 0.5 with the target at 0 means the **reward is 0.25 to
0.5 of one bar range**. With the stop at the bar's high the risk is 0.5 to 0.75
of that range, so the trade is **0.33R to 1.0R** - it needs a 50% to 75% strike
rate to break even *before costs*. Only the 0.75 stop can produce a payoff
above 1, and only for entries in the upper half of the zone.

So this setup is the exact structural mirror of the body-engulfing study: that
one had a large stop and a fair-coin hit rate; this one has a small target and
therefore needs a high hit rate. Both are priced by the same arithmetic, and
``cost_r`` - the round turn as a fraction of what is risked - decides both.

The entry is a resting order, so signals go unfilled
-----------------------------------------------------
The old study's entry was a market order at the close and always filled. Here
the order sits in a zone the bar has already left, and three things can happen
to it before it fills. All three are counted, because omitting any one of them
is how a retracement entry gets flattered:

* **missed** - price reached the target without ever coming back to the zone.
  These are the trades the idea would have won, and they are exactly the ones a
  careless backtest silently books as wins by testing the target before the
  entry.
* **invalidated** - price traded back beyond the bar's stop extreme first.
* **expired** - neither happened within ``valid_bars``.

Where the close sits decides the order type, and both types occur
-----------------------------------------------------------------
These bars close low - the median close sits at 0.10 to 0.27 of the range - so
roughly 50-70% of signals close *below* the zone and are worked as a **sell
limit** on a bounce, which is the play as it is normally described. But 20-35%
close *inside* the zone (a market order at the close) and 10-18% close *above*
it, where "enter in the zone" can only mean a **sell stop** on the way down.
The three are priced differently - a limit does not pay slippage, a stop does -
and reported separately, because they are not the same trade.

Prices
------
Bars are OHLC on the mid, so every fibonacci level is a mid quantity - what a
trader reading a chart draws. Fills are not. A short is worked on the **bid**
(a sell limit fills when the bid reaches it; a sell stop triggers when the bid
falls to it) and covered on the **ask** (the target is a buy limit that fills
when the ask reaches it, the stop a buy stop triggered by the ask). Resting
orders fill at their price with no slippage; market legs pay it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from ..bars import resample_bars
from ..costs import CostModel
from ..loader import load_bars, load_ticks
from ..symbols import SymbolSpec, get_spec

US = 1_000_000  # microseconds in a second

# The same eleven-rung ladder the body-engulfing study swept, so the two
# results sit on the same axis. The ladder matters more here than usual: the
# whole question is whether any timeframe is slow enough that the round turn
# stops swamping a payoff of half a bar range, and that is a question about the
# gradient, not about its endpoints.
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "20m": 20, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
}
ALL_INTERVALS: tuple[str, ...] = tuple(INTERVAL_MINUTES)

# --------------------------------------------------------------------------
# The fibonacci ladder. 0 is the target extreme, 1 the stop extreme.
# --------------------------------------------------------------------------

# Entry conventions. "zone" is the idea read literally - the order works the
# whole 0.25-0.5 band and fills wherever price first touches it, which is the
# 0.25 edge on a bounce and the 0.5 edge on a break. The two fixed levels are
# what it is measured against: an edge that exists only at one of them is a
# fitted entry, not a zone.
ENTRY_MODES: tuple[str, ...] = ("zone", "q25", "q50")
ZONE_LO, ZONE_HI = 0.25, 0.5

# Stop levels, in the same units. The idea does not name one.
STOP_FIBS: dict[str, float] = {"s75": 0.75, "s100": 1.0, "s125": 1.25}

# Target levels. t0 is the bar's low - the idea as stated. The rest extend
# below it, in quarter-range steps, to answer "was the low the right place to
# stop?" rather than assume it.
TARGET_FIBS: dict[str, float] = {"t0": 0.0, "tm25": -0.25, "tm50": -0.5, "tm100": -1.0}

# The no-target variant: hold until the stop or the clock. Named like a target
# so the sweep can treat it as one more column.
HOLD = "hold"
TARGET_KEYS: tuple[str, ...] = (*TARGET_FIBS, HOLD)
EXIT_KEYS: tuple[str, ...] = tuple(
    f"{s}_{t}" for s in STOP_FIBS for t in TARGET_KEYS
)

# Bar counts at which the unconditional forward move is measured, before any
# barrier or cost is applied.
FORWARD_BARS: tuple[int, ...] = (1, 5, 20)

# Why a signal did not become a trade.
FILL_REASONS: tuple[str, ...] = ("filled", "missed", "invalidated", "expired", "no_tape")


# --------------------------------------------------------------------------
# The arithmetic of the geometry - which decides this study before any tick
# --------------------------------------------------------------------------

def payoff_ratio(entry_fib: float, stop_fib: float, target_fib: float) -> float:
    """Reward-to-risk of one (entry, stop, target) triple, in R."""
    risk = stop_fib - entry_fib
    return (entry_fib - target_fib) / risk if risk > 0 else float("nan")


def breakeven_rate(payoff: float, cost_r: float) -> float:
    """Strike rate a payoff of ``payoff`` needs to break even after ``cost_r``.

    Winning ``payoff`` and losing 1, plus the round turn on every trade,
    expectation is zero at ``p = (1 + cost) / (1 + payoff)``.
    """
    return (1.0 + cost_r) / (1.0 + payoff) if payoff > -1 else float("nan")


def fair_rate(entry_fib: float, stop_fib: float, target_fib: float) -> float:
    """Strike rate a driftless random walk gives this geometry.

    Unusually, this setup has an exact null. Every barrier is a fibonacci level
    of the same bar, so once the entry lands at ``entry_fib`` the trade is a
    first passage between two fixed levels, and for a driftless walk the
    probability of reaching the target before the stop is the distance ratio
    ``(stop - entry) / (stop - target)``. No simulation, no assumption about
    volatility, nothing fitted.

    **The identity that decides the study.** Substituting
    :func:`payoff_ratio` into :func:`breakeven_rate` at zero cost gives

        (stop - entry) / ((stop - entry) + (entry - target))
          = (stop - entry) / (stop - target)

    which is exactly this function. So for *every* (entry, stop, target) triple
    the break-even strike rate and the fair-coin strike rate **coincide when
    the round turn is zero** - the trade is a fair bet by construction, whatever
    levels are chosen. No entry level, no stop, no target and no timeframe can
    change that; costs can only push it below zero. It is why the exit grid is
    flat, and it is the reason this setup cannot be rescued by tuning.
    """
    span = stop_fib - target_fib
    return (stop_fib - entry_fib) / span if span > 0 else float("nan")


@dataclass(frozen=True)
class QuadrantConfig:
    """Every choice the idea leaves implicit, made explicit.

    Defaults are a literal reading: no trend filter, no session filter beyond
    the instrument's own maintenance break, no spread guard, and no requirement
    that the prior candle be bullish - the rules as given say nothing about it.
    Each is a field so that "does this rescue it?" is a measurement rather than
    a rewrite.
    """

    valid_bars: int = 12
    """How many bars the resting entry order stays live. Twelve is generous
    enough that expiry is rare and the fill statistics describe the setup
    rather than the deadline; the bar each fill actually landed on is recorded,
    so a shorter window can be imposed afterwards without a re-run."""

    max_hold_bars: int = 60
    """Cap on a position's life, in bars of its own timeframe. Bounds the tape
    each trade is walked over; its bind rate is reported, not assumed to be
    zero."""

    require_prev_bull: bool = False
    """Require ``close[1] > open[1]`` - the prior candle actually being an up
    candle for the signal bar to engulf. The rules as given do not say this and
    it is true only about half the time, so it is off by default and run as a
    variant."""

    mirror: bool = False
    """Trade the mirror setup instead: takes out the prior low, closes above
    the prior open, entry in the 0.25-0.5 zone measured down from the bar's
    high, target the high. The symmetry check - a real pattern pays on both
    sides, a drift capture does not."""

    fade: bool = False
    """Keep the bearish signal but take the *long* geometry on the same bars -
    buy the 0.25-0.5 zone measured from the high, target the high, stop the
    low. The placebo. A structure that rests a limit order inside a bar and
    targets the near extreme has a high strike rate by construction, whichever
    way it is pointed; if the fade earns what the setup earns, the pattern is
    decoration and the geometry is what was measured."""

    control: str = "none"
    """The null. ``matched`` draws a size-matched deterministic sample of
    ordinary bars that are *not* signals and applies the identical quadrant
    geometry to them. ``nosweep`` is the sharper one: bars that close below the
    prior open but did **not** take out the prior high - it drops condition 1
    alone, so it isolates what the stop run is worth."""

    max_spread_mult: float = 0.0
    """Skip a signal if the spread at the arm instant exceeds this multiple of
    the symbol's mean spread. ``0`` disables the guard."""

    skip_break_hours: bool = True
    """Skip signals whose candle closes inside the instrument's maintenance
    break. Data hygiene rather than alpha: a bar built from a handful of ticks
    during broker downtime is an artifact of the feed."""

    lots: float = 0.01
    """Fixed position size, so dollar risk per trade is *not* constant - it
    varies with the bar range. R and USD are both reported."""

    def bar_us(self, interval: str) -> int:
        return INTERVAL_MINUTES[interval] * 60 * US


# --------------------------------------------------------------------------
# Signals - pure bar arithmetic, no ticks, no costs
# --------------------------------------------------------------------------

SIGNAL_COLUMNS: tuple[str, ...] = (
    "interval", "arm_ts", "direction",
    "open_p", "high_p", "low_p", "close_p", "prev_open_p", "prev_high_p",
    "prev_low_p", "prev_close_p", "spread_close_p",
    "bar_range", "close_fib", "sweep_depth", "give_back",
    *(f"fwd{k}_bps" for k in FORWARD_BARS),
    "mfe20_rng", "mae20_rng",
)


def signals(bars: pl.DataFrame, cfg: QuadrantConfig, *,
            interval: str = "5m") -> pl.DataFrame:
    """Every signal bar in ``bars``, with the geometry its quadrants imply.

    One row per signal, carrying the instant the order goes to work - the
    signal bar's close, which under right-edge labelling is its ``ts``.

    **Contiguity.** Bars are missing rather than flat across weekends and the
    maintenance break, so adjacency by row is not adjacency by clock. A bar
    that "takes out the high" of the last bar before a two-day gap is a gap,
    not a stop run, and is dropped.

    The ``fwd*`` and ``mfe/mae`` columns look forward deliberately: they are
    the unconditional behaviour of price after the signal, with no order, no
    barrier and no cost - the thing that has to be non-zero before any entry
    geometry can matter. They are measurements, never inputs to a rule.
    """
    if bars.height < max(FORWARD_BARS) + 3:
        return _empty_signals()

    o, h, l, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")

    # Every cross-bar quantity is materialised before any filter touches the
    # frame. A live ``.shift(1)`` re-evaluated after a filter reads the previous
    # *surviving* row rather than the previous bar, which silently relabels
    # signals rather than dropping them.
    frame = bars.sort("ts_open").with_columns(
        **{f"_fwd{k}_px": c.shift(-k) for k in FORWARD_BARS},
        _po=o.shift(1), _ph=h.shift(1), _pl=l.shift(1), _pc=c.shift(1),
        _pts=pl.col("ts").shift(1),
        _mx20=h.reverse().rolling_max(20).reverse().shift(-1),
        _mn20=l.reverse().rolling_min(20).reverse().shift(-1),
        _range=h - l,
    )

    contiguous = pl.col("ts_open") == pl.col("_pts")
    if cfg.mirror:
        # Takes out the prior low, closes above the prior open.
        raw = (l < pl.col("_pl")) & (c > pl.col("_po"))
        direction = 1
        prev_ok = pl.col("_pc") < pl.col("_po")   # prior candle bearish
    else:
        raw = (h > pl.col("_ph")) & (c < pl.col("_po"))
        direction = -1
        prev_ok = pl.col("_pc") > pl.col("_po")   # prior candle bullish
    if cfg.require_prev_bull:
        raw = raw & prev_ok

    frame = frame.with_columns(_is_sig=contiguous & raw & (pl.col("_range") > 0))

    if cfg.control == "matched":
        frame = _size_matched(
            frame, contiguous & (pl.col("_range") > 0) & ~pl.col("_is_sig"))
    elif cfg.control == "nosweep":
        # Condition 2 alone: closes past the prior open, without ever having
        # taken out the prior extreme. Drops the sweep and nothing else.
        cond2 = (c > pl.col("_po")) if cfg.mirror else (c < pl.col("_po"))
        swept = (l < pl.col("_pl")) if cfg.mirror else (h > pl.col("_ph"))
        frame = _size_matched(
            frame, contiguous & cond2 & ~swept & (pl.col("_range") > 0))
    elif cfg.control == "none":
        frame = frame.filter(pl.col("_is_sig"))
    else:
        raise ValueError(f"unknown control {cfg.control!r}")

    if frame.is_empty():
        return _empty_signals()

    # The fade keeps the bars and flips the trade. Everything downstream reads
    # ``direction``, so the quadrant anchor flips with it.
    if cfg.fade:
        direction = -direction

    rng = pl.col("_range")
    if direction == -1:
        close_fib = (c - l) / rng          # 0 at the low, 1 at the high
        sweep = (h - pl.col("_ph")) / rng  # how far past the prior high it ran
        give = (pl.col("_po") - c) / rng   # how far past the prior open it closed
    else:
        close_fib = (h - c) / rng
        sweep = (pl.col("_pl") - l) / rng
        give = (c - pl.col("_po")) / rng

    frame = frame.with_columns(
        interval=pl.lit(interval),
        arm_ts=pl.col("ts"),  # right-edge label: the signal bar's close
        direction=pl.lit(direction, pl.Int32),
        open_p=o, high_p=h, low_p=l, close_p=c,
        prev_open_p=pl.col("_po"), prev_high_p=pl.col("_ph"),
        prev_low_p=pl.col("_pl"), prev_close_p=pl.col("_pc"),
        spread_close_p=pl.col("spread_close"),
        bar_range=rng,
        close_fib=close_fib,
        sweep_depth=sweep,
        give_back=give,
    )

    # Unconditional forward moves from the bar's close, signed by the direction
    # the setup takes, in basis points - the only unit that compares a 1m
    # EURUSD bar with a 4h gold one. And the excursions in *range* units, which
    # is where every barrier this study places lives.
    fwd = {
        f"fwd{k}_bps": (pl.lit(direction) * (pl.col(f"_fwd{k}_px") - c) / c * 1e4)
        for k in FORWARD_BARS
    }
    if direction == -1:
        mfe, mae = (c - pl.col("_mn20")) / rng, (pl.col("_mx20") - c) / rng
    else:
        mfe, mae = (pl.col("_mx20") - c) / rng, (c - pl.col("_mn20")) / rng
    frame = frame.with_columns(**fwd, mfe20_rng=mfe, mae20_rng=mae)

    return frame.select(SIGNAL_COLUMNS).sort("arm_ts")


def _size_matched(frame: pl.DataFrame, candidates: pl.Expr) -> pl.DataFrame:
    """A draw of ``candidates`` the same size as the signal set, keeping geometry.

    Both nulls go through here, and both are *size-matched to the signal* rather
    than taking every qualifying bar. Two reasons, and the second is the one
    that matters:

    * **It is the better comparison.** Every statistic these nulls are read on
      is a rate - strike rate, fill rate, mean R - so a larger control buys
      precision that is already far beyond what the question needs, while
      leaving the two samples different sizes for no reason.
    * **It is the difference between a 3 GB run and a 21 GB one.** The no-sweep
      condition matches roughly six times as many bars as the signal, and at
      eleven timeframes that is tens of millions of resolved rows.

    The draw is a hash of the bar's own timestamp rather than a random number
    generator, so the control is identical on every re-run and on every subset
    of the data - a control that moves between runs cannot settle an argument.
    """
    frame = frame.with_columns(
        _cand=candidates,
        _u=((pl.col("ts").cast(pl.Int64) // 1000) * 2654435761 % 1000003) / 1000003.0,
    )
    n_sig = int(frame["_is_sig"].sum())
    n_cand = int(frame["_cand"].sum())
    if n_sig == 0 or n_cand == 0:
        return frame.clear()
    rate = min(1.0, n_sig / n_cand)
    return frame.filter(pl.col("_cand") & (pl.col("_u") < rate))


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "interval": pl.String,
            "arm_ts": pl.Datetime("us", "UTC"),
            "direction": pl.Int32,
            **{c: pl.Float64 for c in SIGNAL_COLUMNS[3:]},
        }
    )


# --------------------------------------------------------------------------
# The tape
# --------------------------------------------------------------------------

@dataclass
class _Tape:
    ts: np.ndarray   # int64 microseconds, ascending
    bid: np.ndarray
    ask: np.ndarray

    def index_at(self, when_us: int) -> int:
        return int(np.searchsorted(self.ts, when_us, side="left"))


# First-passage search proceeds in geometrically growing chunks rather than over
# the whole window at once. Every level here is a fraction of one bar's range
# away and is usually touched within a bar or two, so testing the first few
# thousand ticks resolves the vast majority of them; materialising a ten-day
# boolean mask to find a crossing forty ticks in would dominate the whole run.
_CHUNK0 = 4096


def _first_cross(arr: np.ndarray, lo: int, hi: int, level: float,
                 below: bool) -> int:
    """First index in ``[lo, hi)`` where ``arr`` crosses ``level``, or -1.

    ``below`` selects ``arr <= level`` rather than ``arr >= level``.
    """
    if lo >= hi:
        return -1
    pos, size = lo, _CHUNK0
    while pos < hi:
        end = min(pos + size, hi)
        seg = arr[pos:end]
        mask = (seg <= level) if below else (seg >= level)
        i = int(mask.argmax())
        if mask[i]:
            return pos + i
        pos = end
        size *= 2
    return -1


# --------------------------------------------------------------------------
# One signal, resolved against the tape
# --------------------------------------------------------------------------

def _fib(sig: dict, f: float) -> float:
    """Price at fibonacci level ``f``: 0 is the target extreme, 1 the stop one."""
    if sig["direction"] == -1:
        lo, hi = sig["low_p"], sig["high_p"]
    else:
        lo, hi = sig["high_p"], sig["low_p"]
    return lo + f * (hi - lo)


def _resolve(tape: _Tape, sig: dict, cfg: QuadrantConfig, spec: SymbolSpec,
             slip_by_hour: dict[int, float], spread_cap: float,
             bar_us: int) -> list[dict]:
    """Work the resting order, then price every (stop, target) pair on one tape.

    Returns one row per entry mode - the same signal, three different readings
    of "enter in the 0.25-0.5 zone" - each carrying why it did or did not fill
    and, if it filled, the outcome of all fifteen exit combinations.

    Order of resolution matters and is deliberate. The entry is searched first,
    and the target and the invalidation level are searched *over the window
    before the fill*, so a signal whose price ran to the target without ever
    returning to the zone is recorded as ``missed`` rather than as a win. Doing
    it the other way round is the single most flattering mistake available to a
    retracement backtest.
    """
    direction = sig["direction"]
    arm_us = sig["arm_ts"]
    close_px = sig["close_p"]

    e = tape.index_at(arm_us)
    if e >= tape.ts.size:
        return []
    spread_arm = float(tape.ask[e] - tape.bid[e])
    if spread_cap > 0 and spread_arm > spread_cap:
        return []

    hour = datetime.fromtimestamp(arm_us / US, tz=timezone.utc).hour
    slip = slip_by_hour[hour]

    rate = 1.0 if spec.quote_ccy == "USD" else close_px
    commission_px = spec.commission_pips(rate) * spec.pip  # round turn, per lot
    usd_per_px = (spec.contract_size or 1.0) / rate * cfg.lots

    # A short is worked on the bid and covered on the ask; a long the reverse.
    work_arr = tape.bid if direction == -1 else tape.ask
    exit_arr = tape.ask if direction == -1 else tape.bid

    # The two deadlines: when the order dies unfilled, and when a filled
    # position is closed regardless of price.
    fill_deadline = tape.index_at(arm_us + cfg.valid_bars * bar_us)
    fill_deadline = min(max(fill_deadline, e + 1), tape.ts.size)

    stop_px_lvl = {k: _fib(sig, f) for k, f in STOP_FIBS.items()}
    target_px_lvl = {k: _fib(sig, f) for k, f in TARGET_FIBS.items()}
    invalid_lvl = stop_px_lvl["s100"]   # back beyond the bar's own extreme
    miss_lvl = target_px_lvl["t0"]      # the stated target, reached without us

    # Where the order can be cancelled: both are read on the exit side, which is
    # the price at which the position would have been closed had it existed.
    i_invalid = _first_cross(exit_arr, e, fill_deadline, invalid_lvl,
                             below=(direction == 1))
    i_miss = _first_cross(exit_arr, e, fill_deadline, miss_lvl,
                          below=(direction == -1))

    base = {
        **{k: sig[k] for k in SIGNAL_COLUMNS if k != "arm_ts"},
        "arm_ts": arm_us,
        "day": datetime.fromtimestamp(arm_us / US, tz=timezone.utc).date(),
        "hour": hour,
        "spread_arm": spread_arm,
        "slip_px": slip,
        "commission_px": commission_px,
        **{f"px_{k}": v for k, v in stop_px_lvl.items()},
        **{f"px_{k}": v for k, v in target_px_lvl.items()},
    }

    rows: list[dict] = []
    for mode in ENTRY_MODES:
        rows.append(_resolve_entry(
            tape, sig, cfg, base, mode, e, fill_deadline, i_invalid, i_miss,
            work_arr, exit_arr, stop_px_lvl, target_px_lvl, close_px,
            direction, slip, commission_px, usd_per_px, bar_us,
        ))
    return rows


def _resolve_entry(tape, sig, cfg, base, mode, e, fill_deadline, i_invalid,
                   i_miss, work_arr, exit_arr, stop_px_lvl, target_px_lvl,
                   close_px, direction, slip, commission_px, usd_per_px,
                   bar_us) -> dict:
    """One entry convention, from the resting order to every exit it implies."""
    if mode == "zone":
        # The literal reading: the order works the whole band, so it fills at
        # whichever edge price reaches first - which is the 0.25 edge on a
        # bounce, the 0.5 edge on a break down, and the close itself when the
        # bar has already closed inside the zone.
        lo_px, hi_px = _fib(sig, ZONE_LO), _fib(sig, ZONE_HI)
        near, far = (min(lo_px, hi_px), max(lo_px, hi_px))
        level = min(max(close_px, near), far)
    else:
        level = _fib(sig, ZONE_LO if mode == "q25" else ZONE_HI)

    row = dict(base)
    row["entry_mode"] = mode
    row["entry_level"] = level
    row["entry_fib"] = (
        abs(level - _fib(sig, 0.0)) / sig["bar_range"] if sig["bar_range"] else float("nan")
    )

    # A resting order on the favourable side of the market is a limit and fills
    # at its price; on the unfavourable side it is a stop and fills at market.
    is_limit = (level > close_px) if direction == -1 else (level < close_px)
    row["order_type"] = "limit" if is_limit else "stop"
    if is_limit:
        i_fill = _first_cross(work_arr, e, fill_deadline, level,
                              below=(direction == 1))
    else:
        i_fill = _first_cross(work_arr, e, fill_deadline, level,
                              below=(direction == -1))

    # Cancellation only counts if it happened strictly before the fill. A tick
    # that satisfies both is credited to the fill: the resting order was in the
    # book first and would have been taken on the way through.
    cancelled = (
        (i_invalid >= 0 and (i_fill < 0 or i_invalid < i_fill)),
        (i_miss >= 0 and (i_fill < 0 or i_miss < i_fill)),
    )
    if cancelled[1] and (not cancelled[0] or i_miss <= i_invalid):
        row["fill_reason"] = "missed"
    elif cancelled[0]:
        row["fill_reason"] = "invalidated"
    elif i_fill < 0:
        row["fill_reason"] = "expired"
    else:
        row["fill_reason"] = "filled"
    row["filled"] = row["fill_reason"] == "filled"
    if not row["filled"]:
        return row

    # A limit fills at its price; a stop is a market order and pays slippage,
    # adversely - a short sells lower than it meant to, a long buys higher.
    entry_px = level if is_limit else (
        float(work_arr[i_fill]) + (-slip if direction == -1 else slip)
    )
    row["fill_ts"] = int(tape.ts[i_fill])
    row["fill_bar"] = (int(tape.ts[i_fill]) - base["arm_ts"]) / bar_us
    row["immediate"] = bool(i_fill == e)
    row["entry"] = entry_px
    row["entry_mid"] = float((tape.bid[i_fill] + tape.ask[i_fill]) / 2)
    row["spread_fill"] = float(tape.ask[i_fill] - tape.bid[i_fill])
    row["slip_entry"] = 0.0 if is_limit else slip

    hold_end = tape.index_at(int(tape.ts[i_fill]) + cfg.max_hold_bars * bar_us)
    hold_end = min(max(hold_end, i_fill + 1), tape.ts.size)
    row["n_ticks_held"] = int(hold_end - i_fill)

    # First passage of every barrier, searched once each. The levels are
    # monotone in their own direction - a wider stop can only be reached later
    # than a tighter one - so each search starts where the previous one landed
    # rather than at the fill.
    i_stop: dict[str, int] = {}
    pos = i_fill
    for key in sorted(STOP_FIBS, key=lambda k: STOP_FIBS[k]):
        pos = _first_cross(exit_arr, max(pos, i_fill), hold_end,
                           stop_px_lvl[key], below=(direction == 1))
        i_stop[key] = pos
        if pos < 0:
            pos = hold_end   # no wider stop can be reached either
    i_tp: dict[str, int] = {}
    pos = i_fill
    for key in sorted(TARGET_FIBS, key=lambda k: -TARGET_FIBS[k]):
        pos = _first_cross(exit_arr, max(pos, i_fill), hold_end,
                           target_px_lvl[key], below=(direction == -1))
        i_tp[key] = pos
        if pos < 0:
            pos = hold_end

    # Excursions from the fill, in range units - barrier-free and cost-free, so
    # any exit at all can be judged against them afterwards.
    seg_fav = exit_arr[i_fill:hold_end]
    if seg_fav.size:
        best = float(seg_fav.min() if direction == -1 else seg_fav.max())
        worst = float(seg_fav.max() if direction == -1 else seg_fav.min())
        rng = sig["bar_range"]
        row["mfe_fill_rng"] = direction * (best - entry_px) / rng
        row["mae_fill_rng"] = direction * (entry_px - worst) / rng

    x_end = hold_end - 1
    end_px = float(exit_arr[x_end]) + (slip if direction == -1 else -slip)

    for skey in STOP_FIBS:
        risk = abs(entry_px - stop_px_lvl[skey])
        row[f"risk_{skey}"] = risk
        row[f"risk_usd_{skey}"] = risk * usd_per_px
        row[f"risk_bps_{skey}"] = risk / entry_px * 1e4 if entry_px else float("nan")
        # The round turn as a fraction of what is risked. The entry leg pays a
        # spread only when it is a market order; the exit leg's spread is
        # already inside the level crossing, so it is counted once here.
        row[f"cost_r_{skey}"] = (
            commission_px + row["spread_fill"] + row["slip_entry"] + slip
        ) / risk if risk > 0 else float("nan")
        row[f"commission_r_{skey}"] = commission_px / risk if risk > 0 else float("nan")

        si = i_stop[skey]
        stop_fill = (float(exit_arr[si]) + (slip if direction == -1 else -slip)
                     if si >= 0 else None)
        for tkey in TARGET_KEYS:
            # HOLD is the no-target variant: nothing to reach, so it always
            # ends at the stop or the clock.
            ti = -1 if tkey == HOLD else i_tp[tkey]
            # A tick that satisfies both barriers is credited to the stop. One
            # quote cannot tell which came first inside itself, and guessing in
            # the trade's favour is how a backtest invents an edge.
            won = ti >= 0 and (si < 0 or ti < si)
            if won:
                # A target is a limit: it fills at its price or not at all.
                exit_px, reason, x = target_px_lvl[tkey], "tp", ti
            elif si >= 0:
                exit_px, reason, x = stop_fill, "stop", si
            else:
                exit_px, reason, x = end_px, "time", x_end
            gross_px = direction * (exit_px - entry_px)
            key = f"{skey}_{tkey}"
            row[f"r_{key}"] = (gross_px - commission_px) / risk if risk > 0 else float("nan")
            row[f"usd_{key}"] = (gross_px - commission_px) * usd_per_px
            row[f"reason_{key}"] = reason
            row[f"exit_ts_{key}"] = int(tape.ts[x])
    return row


# --------------------------------------------------------------------------
# The account: one position at a time
# --------------------------------------------------------------------------

def _taken_indices(fill: np.ndarray, out: np.ndarray) -> list[int]:
    """Positions of the trades a one-position-at-a-time account actually takes."""
    idx: list[int] = []
    i, n = 0, fill.size
    while i < n:
        idx.append(i)
        # First signal filling at or after this trade's exit. ``i + 1``
        # guarantees progress when a trade exits on its own entry tick.
        j = int(np.searchsorted(fill, out[i], side="left"))
        i = max(j, i + 1)
    return idx


def sequence_many(trades: pl.DataFrame,
                  exit_keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    """:func:`sequence` for several exit rules, sharing one filter and sort.

    Every exit rule walks the *same* filled, time-ordered trades and differs
    only in the exit timestamp it reads, so filtering and sorting once and
    reusing it is worth an order of magnitude on a large tape - which is
    exactly the case where a study stops being re-summarisable.
    """
    if trades.is_empty():
        return {k: trades for k in exit_keys}
    keep: dict[str, list[pl.DataFrame]] = {k: [] for k in exit_keys}
    for _, group in trades.group_by(["interval", "entry_mode"], maintain_order=True):
        group = group.filter(pl.col("filled")).sort("fill_ts")
        if group.is_empty():
            continue
        fill = group["fill_ts"].to_numpy()
        for key in exit_keys:
            out = group[f"exit_ts_{key}"].to_numpy()
            keep[key].append(group[_taken_indices(fill, out)])
    return {
        k: (pl.concat(v).sort("fill_ts") if v else trades.clear())
        for k, v in keep.items()
    }


def sequence(trades: pl.DataFrame, exit_key: str) -> pl.DataFrame:
    """The subset of ``trades`` an account actually takes, for one exit rule.

    Every filled signal has been priced as though it were entered. Most were
    not: a signal arming while a position is already open is ignored, so only
    signals whose *fill* lands at or after the previous trade's exit start a
    trade. Each (interval, entry_mode) is a separate account, since a timeframe
    and an entry convention are separate strategies.
    """
    return sequence_many(trades, [exit_key])[exit_key]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(symbol: str, cfg: QuadrantConfig | None = None, *,
        intervals: Sequence[str] = ALL_INTERVALS,
        start=None, end=None, split: str | None = None,
        allow_test: bool = False, cost: CostModel | None = None,
        verbose: bool = False) -> pl.DataFrame:
    """Backtest the setup on one symbol across every timeframe in ``intervals``.

    The returned frame is one row per *(signal, entry mode)*, whether or not it
    filled. Use :func:`sequence` to reduce it to the trades a one-position-at-a-
    time account takes under a given exit rule.
    """
    cfg = cfg or QuadrantConfig()
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    spread_cap = (cost.spread_pips() * spec.pip * cfg.max_spread_mult
                  if cfg.max_spread_mult > 0 else 0.0)
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}

    base = load_bars(symbol, "1m", start=start, end=end, split=split,
                     allow_test=allow_test)
    if base.is_empty():
        return pl.DataFrame()

    sig_frames: list[pl.DataFrame] = []
    for interval in intervals:
        bars = base if interval == "1m" else resample_bars(base, interval)
        found = signals(bars, cfg, interval=interval)
        if cfg.skip_break_hours and spec.daily_break_utc is not None:
            lo, hi = spec.daily_break_utc
            found = found.filter(~pl.col("arm_ts").dt.hour().is_between(lo, hi - 1))
        if verbose:
            print(f"  {symbol} {interval}: {found.height:,} signals")
        sig_frames.append(found)

    sigs = pl.concat(sig_frames) if sig_frames else _empty_signals()
    if sigs.is_empty():
        return pl.DataFrame()
    sigs = sigs.sort("arm_ts")

    frames: list[pl.DataFrame] = []
    n_rows = 0
    months = (
        sigs.select(pl.col("arm_ts").dt.year().alias("y"),
                    pl.col("arm_ts").dt.month().alias("m"))
        .unique().sort(["y", "m"]).rows()
    )
    max_span_us = max(
        (cfg.valid_bars + cfg.max_hold_bars) * cfg.bar_us(iv) for iv in intervals
    )
    for y, m in months:
        m_start = datetime(y, m, 1, tzinfo=timezone.utc)
        m_end = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc)
        batch = sigs.filter((pl.col("arm_ts") >= m_start) & (pl.col("arm_ts") < m_end))
        if batch.is_empty():
            continue

        # allow_test is safe here: the window is bounded by signals that
        # themselves came from a split-guarded bar load.
        tail_end = m_end + timedelta(microseconds=max_span_us) + timedelta(hours=6)
        ticks = load_ticks(symbol, start=m_start, end=tail_end, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(ts=ticks["ts"].cast(pl.Int64).to_numpy(),
                     bid=ticks["bid"].to_numpy(), ask=ticks["ask"].to_numpy())
        del ticks

        month_rows: list[dict] = []
        for sig in batch.iter_rows(named=True):
            sig = dict(sig)
            sig["arm_ts"] = int(sig["arm_ts"].timestamp() * US)
            month_rows.extend(_resolve(tape, sig, cfg, spec, slip_by_hour,
                                       spread_cap, cfg.bar_us(sig["interval"])))
        # Fold each month down to a frame before starting the next one. A
        # resolved row is ~90 keys, and a Python dict of that shape costs
        # several kilobytes against a few hundred bytes as a column-store row -
        # so holding every month's dicts until the end is the difference
        # between a few GB and tens of them on the larger variants.
        if month_rows:
            frames.append(pl.DataFrame(month_rows, infer_schema_length=None))
            n_rows += len(month_rows)
        month_rows.clear()
        del tape
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {n_rows:,} rows")

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("arm_ts")
