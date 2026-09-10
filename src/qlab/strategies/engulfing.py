"""The engulfing candle, taken literally: enter at its close, stop at its extreme.

The setup, stated with no room left for interpretation - bodies only, wicks
ignored for the *signal* but not for the *stop*:

* **Long**  ``close[1] < open[1]`` and ``close[0] > open[0]``
  and ``open[0] <= close[1]`` and ``close[0] >= open[1]``
* **Short** the exact mirror.

Entry is a market order at the close of the signal candle. The stop is that
candle's own low (long) or high (short). The target is a fixed multiple of the
resulting risk, and which multiple is the question the sweep answers. If an
opposite signal fires while a position is open, the position is closed and
reversed.

Why this setup is worth resolving on ticks
------------------------------------------
The entry sits at the close of the bar and the stop sits at that same bar's
extreme, so **1R is one bar's range** - by construction, never more. On the
very next bar the range routinely spans both the stop and any target under
about 2R, and a bar-level engine has to guess which was touched first. That
guess is worth several times the edge on offer. Here the stop and every target
are first-passage events on the same tick tape, resolved in the order the tape
saw them.

The structural problem, named in advance
----------------------------------------
Because 1R is one bar's range, 1R *shrinks with the timeframe*, while the round
turn - spread plus commission plus two slippages - does not. Cost per unit of
risk is therefore roughly inversely proportional to bar range, and it is the
first thing that has to be looked at, not the last. Every row carries it as
``cost_r``. A setup whose 1m variant pays 40% of its risk to trade has to win
on gross terms by a margin that no candle pattern has ever been shown to have.

Exits are swept, not assumed
----------------------------
The idea fixes the entry and the risk and nominates 2R only provisionally, so
no single exit is privileged: :func:`run` resolves a whole family of fixed
targets from 0.5R to 6R against the same fills in one pass, plus the
no-target variant that rides until the stop or the reversal. An edge visible at
exactly one multiple is a fitted exit, not a working setup.

What the reversal does, and why it does not need a state machine to price
------------------------------------------------------------------------
"Reverse on the opposite signal" makes the trade sequence path-dependent: which
signals you are flat for depends on when the previous trade ended. But the
*exit* of any given trade does not. A trade opened at signal ``i`` ends at
whichever comes first of its stop, its target, and the first opposite signal
after ``i`` - and that last one is a property of the signal list alone, not of
the account. So every signal is resolved independently against the tape, and
:func:`sequence` afterwards walks the cheap state machine over the precomputed
exits to say which trades an account holding one position at a time actually
took. Same answer, without a tick walk inside a stateful loop.

Prices
------
Bars are OHLC on the mid (see :mod:`qlab.bars`), so the signal, the stop level
and 1R are all mid quantities - which is what a trader reading a chart sees.
Fills are not: a long enters at the ask and leaves at the bid, and a stop at
the mid low is triggered when the *bid* reaches it, which is how the broker
triggers it. That asymmetry is the reason a stop can already be breached at the
instant of entry when the bar's close-to-low distance is under half a spread.
Those trades are not dropped - they are counted, as ``instant_stop``, because
they are exactly the ones a cost-blind study would quietly discard.
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

INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "20m": 20, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
}

# The full m1 -> 4h ladder the study sweeps.
ALL_INTERVALS: tuple[str, ...] = tuple(INTERVAL_MINUTES)

# Target multiples resolved for every trade. 1R is the planned risk - the
# signal candle's close-to-extreme distance - so these are comparable across
# instruments, timeframes and volatility regimes. 2.0 is the idea as stated;
# the rest are what it is measured against.
TARGET_MULTIPLES: tuple[float, ...] = (
    0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
)

# Bar counts at which the unconditional forward return is measured, for the
# signal-quality read that precedes any exit choice.
FORWARD_BARS: tuple[int, ...] = (1, 5, 20)


def tag(mult: float) -> str:
    """Column-safe name for a target multiple: 1.25 -> ``t1_25``."""
    return "t" + f"{mult:g}".replace(".", "_")


# The no-target variant: hold until the stop or the reversal takes it. Named
# like a target so the exit sweep can treat it as one more column.
HOLD = "hold"
EXIT_KEYS: tuple[str, ...] = (*(tag(m) for m in TARGET_MULTIPLES), HOLD)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EngulfingConfig:
    """Every choice the idea leaves implicit, made explicit.

    The defaults are deliberately close to a literal reading of the idea:
    no trend filter, no size filter, no session filter beyond the instrument's
    own maintenance break, and no spread guard. Filters are available as
    fields so that "does a filter rescue it?" is a measurement rather than a
    rewrite, but none of them is on by default, because a filter chosen after
    seeing the result is a fitted parameter wearing a disguise.
    """

    reverse: bool = True
    """Close and reverse when an opposite signal fires. The idea as stated.
    ``False`` leaves the trade to its stop or target alone, which is the
    comparison that says how much the reversal rule is worth."""

    both_sides: bool = True
    """Trade the bearish engulfing as well as the bullish one."""

    strict: bool = False
    """Require strict inequality on the engulfment (``open[0] < close[1]``
    rather than ``<=``). The idea as given is inclusive, so this is off; it
    exists because on coarse timeframes an exactly-equal open is common enough
    (an unchanged open) that it is worth knowing whether it matters."""

    max_hold_hours: float = 504.0
    """Wall-clock cap on a position's life, in hours. Three weeks, chosen to
    sit above the observed maximum wait to an opposite signal on 4h bars
    (~440 h), so that with ``reverse=True`` the cap essentially never binds and
    the reversal is what ends the trade. It exists to bound the tape each trade
    is walked over, and its bind rate is reported rather than assumed to be
    zero."""

    min_risk_pips: float = 0.0
    """Skip signals whose 1R is below this. Zero takes them all, which is the
    honest default: the small-risk trades are precisely where the cost
    objection lives, and filtering them out before measuring would hide the
    thing most likely to kill the setup."""

    max_spread_mult: float = 0.0
    """Skip a signal if the spread at entry exceeds this multiple of the
    symbol's mean spread. ``0`` disables the guard, which is the literal
    reading of an idea that specifies no filters."""

    skip_break_hours: bool = True
    """Skip signals whose candle closes inside the instrument's daily
    maintenance break. This one *is* on by default, because it is data hygiene
    rather than alpha: bars built from a handful of ticks during a broker's
    downtime are artifacts, and an engulfing pattern found in one is a fact
    about the feed."""

    fade: bool = False
    """Take the mirror of every signal - sell the bullish engulfing, buy the
    bearish one - on the same candles, with the stop mirrored to that candle's
    other extreme so the trade has the same *shape*: entry at the close, stop
    at the opposing extreme, targets in multiples of the gap.

    The placebo the cost-free numbers need. A structure that enters at a bar's
    close, stops at that bar's extreme and targets a multiple of the distance
    between them can carry an expectation of its own that owes nothing to the
    pattern. If it does, the fade shows the *same* sign rather than the
    opposite one, and that is the difference between an edge and an artifact of
    the measurement."""

    control: bool = False
    """Replace the engulfing signals with a matched sample of ordinary bars.

    The null the setup has to beat. Every trade here keeps the geometry -
    market entry at a bar's close, stop at that bar's own extreme, direction
    from the bar's own body - and drops only the engulfment. It draws from
    directional bars that are *not* signals, deterministically, at whatever
    rate makes the sample the same size as the real one, so the two are
    disjoint and comparable.

    If the control earns what the setup earns, then the engulfment is
    decoration and what is being measured is the stop-at-the-bar's-extreme
    structure, which every candle in the sample has.
    """

    lots: float = 0.01
    """Fixed position size, as specified. Note what fixed lots means with a
    stop that is one bar's range: dollar risk per trade is *not* constant, and
    varies by more than an order of magnitude between a quiet 1m bar and a
    volatile 4h one. Both views are reported - R for the signal, USD for the
    account that actually trades it."""

    def hold_span_us(self) -> int:
        return int(self.max_hold_hours * 3600 * US)


# --------------------------------------------------------------------------
# Signals - pure bar arithmetic, no ticks, no costs
# --------------------------------------------------------------------------

SIGNAL_COLUMNS: tuple[str, ...] = (
    "interval", "arm_ts", "horizon_ts", "direction", "entry_ref", "stop",
    "risk", "body", "body_prev", "bar_range", "engulf_ratio",
    "open_p", "high_p", "low_p", "close_p", "spread_close_p",
    *(f"fwd{k}_r" for k in FORWARD_BARS),
    *(f"fwd{k}_bps" for k in FORWARD_BARS),
    "mfe20_r", "mae20_r",
)


def signals(bars: pl.DataFrame, cfg: EngulfingConfig, *,
            interval: str = "5m") -> pl.DataFrame:
    """Every engulfing candle in ``bars``, with its entry reference and stop.

    One row per signal, carrying the instant the trade is taken - the signal
    candle's close, which under right-edge labelling is its ``ts``.

    Two things here are not in the naive statement of the pattern and both
    matter:

    **Contiguity.** Bars are missing rather than flat across weekends and the
    maintenance break, so adjacency by row is not adjacency by clock. A candle
    that "engulfs" the last bar of Friday from the first bar of Sunday is a
    gap, not a pattern, and is dropped.

    **The reversal horizon.** ``horizon_ts`` is the close of the first
    *opposite* signal after this one, which is when a reversing account would
    be forced out regardless of price. It is computed here, from the signal
    list alone, because it does not depend on which trades were taken - see the
    module docstring.

    The ``fwd*`` and ``mfe/mae`` columns look forward deliberately. They are
    the unconditional behaviour of price after the signal, measured with no
    stop, no target and no costs - the thing that has to be non-zero before any
    exit geometry can matter. They are measurements, never inputs to a rule.
    """
    if bars.height < max(FORWARD_BARS) + 3:
        return _empty_signals()

    o, c = pl.col("open"), pl.col("close")

    # Every cross-bar quantity is materialised as a column *before* any filter
    # touches the frame. A live ``.shift(1)`` re-evaluated after a filter reads
    # the previous surviving row rather than the previous bar, which silently
    # relabels signals rather than dropping them.
    fwd = {f"_fwd{k}_px": c.shift(-k) for k in FORWARD_BARS}
    frame = bars.sort("ts_open").with_columns(
        **fwd,
        _prev_o=o.shift(1),
        _prev_c=c.shift(1),
        _prev_ts=pl.col("ts").shift(1),
        _mfe20=pl.col("high").reverse().rolling_max(20).reverse().shift(-1),
        _mae20=pl.col("low").reverse().rolling_min(20).reverse().shift(-1),
        _range=pl.col("high") - pl.col("low"),
    )

    prev_o, prev_c = pl.col("_prev_o"), pl.col("_prev_c")
    if cfg.strict:
        engulf_up = (o < prev_c) & (c > prev_o)
        engulf_dn = (o > prev_c) & (c < prev_o)
    else:
        engulf_up = (o <= prev_c) & (c >= prev_o)
        engulf_dn = (o >= prev_c) & (c <= prev_o)

    long_sig = (prev_c < prev_o) & (c > o) & engulf_up
    short_sig = (prev_c > prev_o) & (c < o) & engulf_dn
    if not cfg.both_sides:
        short_sig = pl.lit(False)

    # Right-edge labels: bar i spans [ts_open_i, ts_i), so the two bars are
    # adjacent on the clock exactly when the previous bar's close instant is
    # this bar's open instant.
    contiguous = pl.col("ts_open") == pl.col("_prev_ts")

    frame = frame.with_columns(
        _body_prev=(prev_c - prev_o).abs(),
        _is_sig=contiguous & (long_sig | short_sig),
        direction=pl.when(long_sig).then(pl.lit(1, pl.Int32))
        .otherwise(pl.lit(-1, pl.Int32)),
    )

    if cfg.control:
        frame = _control_sample(frame, contiguous)
    else:
        frame = frame.filter(pl.col("_is_sig"))

    if frame.is_empty():
        return _empty_signals()
    if cfg.fade:
        frame = frame.with_columns(direction=-pl.col("direction"))

    frame = frame.with_columns(
        interval=pl.lit(interval),
        arm_ts=pl.col("ts"),  # right-edge label: the signal candle's close
        entry_ref=c,
        # The stop is the signal candle's own extreme, wicks included.
        stop=pl.when(pl.col("direction") == 1)
        .then(pl.col("low"))
        .otherwise(pl.col("high")),
        body=(c - o).abs(),
        body_prev=pl.col("_body_prev"),
        bar_range=pl.col("_range"),
        open_p=o, high_p=pl.col("high"), low_p=pl.col("low"), close_p=c,
        spread_close_p=pl.col("spread_close"),
    ).with_columns(
        risk=(pl.col("entry_ref") - pl.col("stop")).abs(),
        engulf_ratio=pl.col("body")
        / pl.when(pl.col("body_prev") > 0).then(pl.col("body_prev")).otherwise(None),
    )

    # Unconditional forward moves, signed by the direction taken and expressed
    # both in R (comparable to any exit) and in basis points (comparable across
    # instruments regardless of how the stop happened to fall).
    fwd_cols = {}
    for k in FORWARD_BARS:
        move = pl.col("direction") * (pl.col(f"_fwd{k}_px") - pl.col("entry_ref"))
        fwd_cols[f"fwd{k}_r"] = move / pl.col("risk")
        fwd_cols[f"fwd{k}_bps"] = move / pl.col("entry_ref") * 1e4
    frame = frame.with_columns(
        **fwd_cols,
        mfe20_r=pl.when(pl.col("direction") == 1)
        .then(pl.col("_mfe20") - pl.col("entry_ref"))
        .otherwise(pl.col("entry_ref") - pl.col("_mae20")) / pl.col("risk"),
        mae20_r=pl.when(pl.col("direction") == 1)
        .then(pl.col("entry_ref") - pl.col("_mae20"))
        .otherwise(pl.col("_mfe20") - pl.col("entry_ref")) / pl.col("risk"),
    ).filter(pl.col("risk") > 0)

    if frame.is_empty():
        return _empty_signals()

    frame = _with_horizon(frame, cfg)
    return frame.select(SIGNAL_COLUMNS).sort("arm_ts")


def _control_sample(frame: pl.DataFrame, contiguous: pl.Expr) -> pl.DataFrame:
    """A size-matched draw of non-signal directional bars, keeping the geometry.

    The draw is a hash of the bar's own timestamp rather than a random number
    generator, so the control is identical on every re-run and on every subset
    of the data - a control that moves between runs cannot settle an argument.
    """
    o, c = pl.col("open"), pl.col("close")
    frame = frame.with_columns(
        _cand=contiguous & (c != o) & ~pl.col("_is_sig"),
        _u=((pl.col("ts").cast(pl.Int64) // 1000) * 2654435761 % 1000003)
        / 1000003.0,
    )
    n_sig = int(frame["_is_sig"].sum())
    n_cand = int(frame["_cand"].sum())
    if n_sig == 0 or n_cand == 0:
        return frame.clear()

    rate = min(1.0, n_sig / n_cand)
    return frame.filter(pl.col("_cand") & (pl.col("_u") < rate)).with_columns(
        # Direction comes from the bar's own body, exactly as it does for a
        # signal: a bullish bar is bought, a bearish bar is sold.
        direction=pl.when(c > o).then(pl.lit(1, pl.Int32))
        .otherwise(pl.lit(-1, pl.Int32))
    )


def _with_horizon(frame: pl.DataFrame, cfg: EngulfingConfig) -> pl.DataFrame:
    """Attach ``horizon_ts``: the close of the next opposite-direction signal.

    With ``reverse=False`` there is no such forced exit, and the horizon is the
    holding cap alone - represented here as a null, which ``_resolve`` reads as
    "cap only".
    """
    if not cfg.reverse:
        return frame.with_columns(
            horizon_ts=pl.lit(None, dtype=frame["arm_ts"].dtype)
        )

    frame = frame.sort("arm_ts")
    ts = frame["arm_ts"].cast(pl.Int64).to_numpy()
    d = frame["direction"].to_numpy()

    # Scanning backwards, remember the most recent signal of each sign seen so
    # far; walking from the end makes "most recent seen" mean "next in time".
    nxt = np.full(ts.size, -1, dtype=np.int64)
    last_long = last_short = -1
    for i in range(ts.size - 1, -1, -1):
        nxt[i] = last_short if d[i] == 1 else last_long
        if d[i] == 1:
            last_long = i
        else:
            last_short = i

    # A signal with no opposite successor - the last few of each direction in
    # the sample - has no reversal exit, and gets a null horizon, which
    # ``_resolve`` reads as "the holding cap alone".
    horizon = pl.Series(
        "horizon_ts",
        [None if n < 0 else int(ts[n]) for n in nxt],
        dtype=pl.Int64,
    ).cast(pl.Datetime("us", "UTC"))
    return frame.with_columns(horizon_ts=horizon)


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "interval": pl.String,
            "arm_ts": pl.Datetime("us", "UTC"),
            "horizon_ts": pl.Datetime("us", "UTC"),
            "direction": pl.Int32,
            **{c: pl.Float64 for c in SIGNAL_COLUMNS[4:]},
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


# First-passage search proceeds in geometrically growing chunks rather than
# over the whole holding window at once. The stop is one bar's range away and
# is usually taken within a few bars, so testing the first few thousand ticks
# resolves the vast majority of trades; materialising a three-week boolean mask
# to find a crossing 40 ticks in would dominate the entire run.
_CHUNK0 = 4096


def _first_cross(arr: np.ndarray, lo: int, hi: int, level: float,
                 below: bool) -> int:
    """First index in ``[lo, hi)`` where ``arr`` crosses ``level``, or -1.

    ``below`` selects ``arr <= level`` (a long's stop, on the bid) rather than
    ``arr >= level`` (a short's stop, on the ask).
    """
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

def _resolve(tape: _Tape, sig: dict, cfg: EngulfingConfig, spec: SymbolSpec,
             slip_by_hour: dict[int, float], spread_cap: float,
             hold_us: int) -> dict | None:
    """Fill at the candle's close, then price every exit against the same tape.

    Returns ``None`` only when the tape cannot support the trade at all - no
    quote at the entry instant, or a spread guard rejection. Unlike a resting
    stop order there is no such thing as an unfilled signal here: the entry is
    a market order, and it fills.
    """
    direction = sig["direction"]
    arm_us = sig["arm_ts"]

    e = tape.index_at(arm_us)
    if e >= tape.ts.size:
        return None
    spread_entry = float(tape.ask[e] - tape.bid[e])
    if spread_cap > 0 and spread_entry > spread_cap:
        return None

    hour = datetime.fromtimestamp(arm_us / US, tz=timezone.utc).hour
    slip = slip_by_hour[hour]
    stop = sig["stop"]
    risk = sig["risk"]  # planned, mid-based: what a position would be sized on

    # The horizon: the reversal if there is one, else the holding cap; and the
    # cap regardless, so a trade cannot outlive the tape loaded for it.
    cap_us = arm_us + hold_us
    horizon_us = cap_us if sig["horizon_ts"] is None else min(sig["horizon_ts"], cap_us)
    reverse_ends_it = sig["horizon_ts"] is not None and sig["horizon_ts"] <= cap_us
    hold_end = max(tape.index_at(horizon_us), e + 1)
    hold_end = min(hold_end, tape.ts.size)
    if hold_end <= e:
        return None

    entry = (tape.ask[e] + slip) if direction == 1 else (tape.bid[e] - slip)
    entry_mid = (tape.bid[e] + tape.ask[e]) / 2

    rate = 1.0 if spec.quote_ccy == "USD" else entry
    commission_px = spec.commission_pips(rate) * spec.pip  # round turn, per lot
    usd_per_px = (spec.contract_size or 1.0) / rate * cfg.lots

    base = {
        **{k: sig[k] for k in SIGNAL_COLUMNS
           if k not in ("arm_ts", "horizon_ts")},
        "arm_ts": arm_us,
        "day": datetime.fromtimestamp(arm_us / US, tz=timezone.utc).date(),
        "hour": hour,
        "entry_ts": int(tape.ts[e]),
        "entry": entry,
        "entry_mid": entry_mid,
        "spread_entry": spread_entry,
        "risk_actual": abs(entry - stop),
        "risk_usd": risk * usd_per_px,
        "slip_r": slip / risk,
        # The number this setup lives or dies by: the round turn as a fraction
        # of the risk being taken. Spread is observed at the entry instant,
        # commission is the contract term, slippage is charged on both market
        # legs - which is the worst case, and is what a stop-out actually pays.
        "cost_r": (commission_px + spread_entry + 2 * slip) / risk,
        "commission_r": commission_px / risk,
        "risk_bps": risk / entry_mid * 1e4,
        "reverse_horizon": reverse_ends_it,
    }

    # --- first passage of the stop --------------------------------------
    # A long's stop is triggered by the bid, a short's by the ask: the price
    # the broker fills you at has to reach the level, not the mid.
    exit_arr = tape.bid if direction == 1 else tape.ask
    i_stop = _first_cross(exit_arr, e, hold_end, stop, below=(direction == 1))
    base["instant_stop"] = bool(i_stop == e)
    n_before_stop = (hold_end - e) if i_stop < 0 else (i_stop - e)

    # A target beyond the entry and a stop behind it cannot both be reached by
    # one quote, so nothing after the stop tick is reachable and the running
    # favourable extreme only has to be built up to it.
    reachable = exit_arr[e:e + n_before_stop]
    if reachable.size:
        run_fav = (np.maximum.accumulate(reachable) if direction == 1
                   else np.minimum.accumulate(reachable))
    else:
        run_fav = reachable

    stop_px = (((tape.bid[i_stop] - slip) if direction == 1
                else (tape.ask[i_stop] + slip)) if i_stop >= 0 else None)
    x_end = hold_end - 1
    end_px = (tape.bid[x_end] - slip) if direction == 1 else (tape.ask[x_end] + slip)
    end_reason = "reverse" if reverse_ends_it else "time"

    base["stopped"] = i_stop >= 0
    base["n_ticks_held"] = int(hold_end - e)
    base["mfe_pre_stop_r"] = (
        direction * (float(run_fav[-1]) - entry) / risk if run_fav.size else 0.0
    )

    # --- the exit family, all from the one running extreme ---------------
    for mult in TARGET_MULTIPLES:
        target = entry + direction * mult * risk
        if direction == 1:
            i_tp = int(np.searchsorted(run_fav, target, side="left"))
        else:
            i_tp = int(np.searchsorted(-run_fav, -target, side="left"))
        won = i_tp < run_fav.size

        if won:
            # A fixed target is a limit: it fills at its price or not at all,
            # so no slippage is charged on it.
            exit_px, reason, x = target, "tp", e + i_tp
        elif i_stop >= 0:
            exit_px, reason, x = stop_px, "stop", i_stop
        else:
            exit_px, reason, x = end_px, end_reason, x_end
        _record(base, tag(mult), direction, entry, entry_mid, exit_px, reason,
                x, tape, risk, commission_px, usd_per_px)

    # The no-target variant: stop or reversal, nothing else. This is the exit
    # that keeps the most of whatever the signal itself predicts, and the one
    # the reversal rule was presumably written for.
    if i_stop >= 0:
        exit_px, reason, x = stop_px, "stop", i_stop
    else:
        exit_px, reason, x = end_px, end_reason, x_end
    _record(base, HOLD, direction, entry, entry_mid, exit_px, reason, x,
            tape, risk, commission_px, usd_per_px)

    return base


def _record(row: dict, key: str, direction: int, entry: float, entry_mid: float,
            exit_px: float, reason: str, x: int, tape: _Tape, risk: float,
            commission_px: float, usd_per_px: float) -> None:
    """Write one exit's outcome onto the trade row, in R, in USD and cost-free."""
    gross_px = direction * (exit_px - entry)
    exit_mid = (tape.bid[x] + tape.ask[x]) / 2
    row[f"r_{key}"] = (gross_px - commission_px) / risk
    # Mid to mid: the same geometry with the spread, the commission and the
    # slippage all stripped off. If this is not positive, nothing else can be.
    row[f"r_{key}_mid"] = direction * (exit_mid - entry_mid) / risk
    row[f"usd_{key}"] = (gross_px - commission_px) * usd_per_px
    row[f"reason_{key}"] = reason
    row[f"exit_ts_{key}"] = int(tape.ts[x])


# --------------------------------------------------------------------------
# The account: one position at a time, reversing on the opposite signal
# --------------------------------------------------------------------------

def sequence(trades: pl.DataFrame, exit_key: str) -> pl.DataFrame:
    """The subset of ``trades`` an account actually takes, for one exit rule.

    Every signal in ``trades`` has been priced as though it were entered. Most
    of them were not: a signal that fires while a position is already open in
    the same direction is ignored, so only signals arriving at or after the
    previous trade's exit start a trade.

    A reversal exit lands exactly on the opposite signal's timestamp, so the
    "next signal at or after the exit" is that opposite signal - the chain
    reverses with no special case. Each ``interval`` is a separate account,
    since a timeframe is a separate strategy.
    """
    if trades.is_empty():
        return trades

    keep: list[pl.DataFrame] = []
    for (interval,), group in trades.group_by(["interval"], maintain_order=True):
        group = group.sort("arm_ts")
        arm = group["arm_ts"].cast(pl.Int64).to_numpy()
        out = group[f"exit_ts_{exit_key}"].to_numpy()

        idx: list[int] = []
        i, n = 0, arm.size
        while i < n:
            idx.append(i)
            # First signal at or after this trade's exit. `i + 1` guarantees
            # progress when a trade is stopped out on its own entry tick.
            j = int(np.searchsorted(arm, out[i], side="left"))
            i = max(j, i + 1)
        keep.append(group[idx])

    return pl.concat(keep).sort("arm_ts") if keep else trades.clear()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(symbol: str, cfg: EngulfingConfig | None = None, *,
        intervals: Sequence[str] = ALL_INTERVALS,
        start=None, end=None, split: str | None = None,
        allow_test: bool = False, cost: CostModel | None = None,
        verbose: bool = False) -> pl.DataFrame:
    """Backtest the setup on one symbol across every timeframe in ``intervals``.

    Signals from all timeframes are pooled and resolved in a single pass over
    each month of tape, so adding a timeframe costs signal detection rather
    than another walk through the symbol's whole tick history.

    The returned frame is one row per *signal*, priced as if taken. Use
    :func:`sequence` to reduce it to the trades a one-position-at-a-time
    account takes under a given exit rule.
    """
    cfg = cfg or EngulfingConfig()
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    spread_cap = (cost.spread_pips() * spec.pip * cfg.max_spread_mult
                  if cfg.max_spread_mult > 0 else 0.0)
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}
    hold_us = cfg.hold_span_us()

    base = load_bars(symbol, "1m", start=start, end=end, split=split,
                     allow_test=allow_test)
    if base.is_empty():
        return pl.DataFrame()

    sig_frames: list[pl.DataFrame] = []
    for interval in intervals:
        bars = base if interval == "1m" else resample_bars(base, interval)
        found = signals(bars, cfg, interval=interval)
        if cfg.min_risk_pips > 0:
            found = found.filter(pl.col("risk") >= cfg.min_risk_pips * spec.pip)
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

    rows: list[dict] = []
    months = (
        sigs.select(pl.col("arm_ts").dt.year().alias("y"),
                    pl.col("arm_ts").dt.month().alias("m"))
        .unique().sort(["y", "m"]).rows()
    )
    for y, m in months:
        m_start = datetime(y, m, 1, tzinfo=timezone.utc)
        m_end = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc)
        batch = sigs.filter((pl.col("arm_ts") >= m_start) & (pl.col("arm_ts") < m_end))
        if batch.is_empty():
            continue

        # Load exactly as much tape past the month as this batch can need: the
        # furthest reversal horizon in it, capped by the holding cap. Most
        # months need only hours, and paying a three-week tail for all of them
        # would nearly double the read.
        need_us = int(batch.select(
            pl.min_horizontal(
                pl.col("horizon_ts").cast(pl.Int64).fill_null(
                    pl.col("arm_ts").cast(pl.Int64) + hold_us),
                pl.col("arm_ts").cast(pl.Int64) + hold_us,
            ).max()
        ).item())
        tail_end = max(m_end, datetime.fromtimestamp(need_us / US, tz=timezone.utc))
        tail_end += timedelta(hours=6)

        # allow_test is safe here: the window is bounded by signals that
        # themselves came from a split-guarded bar load.
        ticks = load_ticks(symbol, start=m_start, end=tail_end, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(ts=ticks["ts"].cast(pl.Int64).to_numpy(),
                     bid=ticks["bid"].to_numpy(), ask=ticks["ask"].to_numpy())
        del ticks

        for sig in batch.iter_rows(named=True):
            sig = dict(sig)
            sig["arm_ts"] = int(sig["arm_ts"].timestamp() * US)
            h = sig["horizon_ts"]
            sig["horizon_ts"] = None if h is None else int(h.timestamp() * US)
            rec = _resolve(tape, sig, cfg, spec, slip_by_hour, spread_cap, hold_us)
            if rec is not None:
                rows.append(rec)
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {len(rows):,} resolved")

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort("arm_ts")
