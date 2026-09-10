"""Breakout, pause bar, entry on the pause bar's break - the price-action classic.

The setup, stated the way a discretionary trader states it:

1. a big bar breaks out of resistance,
2. the next bar is a very small pause bar,
3. buy the break of the pause bar's high,
4. stop under the pause bar's low.

Every one of those four steps is a free parameter until it is written down, so
:class:`PauseBarConfig` writes them all down: how far back "resistance" looks,
how big "big" is relative to recent range, how small "small" is relative to the
breakout bar, and how long the order stays live. The short side is the exact
mirror, so nothing here privileges the long.

The claim attached to the setup is that it "works on all assets and
timeframes". That is unusually falsifiable, so :func:`run` takes a list of
intervals and the study runs all four instruments across four timeframes
spanning 60x.

Why ticks
---------
This setup cannot be honestly resolved on bars. The stop sits under a bar that
was selected *for being small*, so the entry and the stop are typically a few
points apart - and on the very next bar the range routinely spans both. A
bar-level engine has to guess which came first, and that guess is worth several
times the edge being measured. Here the entry, the stop and every target are
first-passage events on the same tick tape, in the order the tape saw them.

Exits are not part of the idea
------------------------------
Steps 1-4 fix the entry and the risk, and say nothing about the way out. So no
single exit is assumed: :func:`simulate` resolves a whole family of them
against the same fills in one pass - fixed targets from 0.5R to 6R, a
breakeven-then-trail, and a plain time stop - and reports the curve. An edge
that only appears at one target multiple is an exit that was fitted, not a
setup that works.

Costs
-----
Fills cross a real bid and a real ask from the tape, so spread is observed.
Commission is the published contract term. Slippage is the assumed term and
comes from :class:`qlab.costs.CostModel`, charged on market fills - the entry,
a stop-out, a trail-out, a time exit - and never on a fixed target, which is a
limit and fills at its price or not at all.

The cost that matters here is not cost per trade but **cost per unit of risk**,
because the pause bar deliberately makes the risk small. That ratio is carried
on every row as ``cost_r``; it is the first thing the report looks at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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

# The target multiples the exit sweep resolves for every filled trade. 1R is
# the planned risk - trigger to stop - so these are directly comparable across
# instruments, timeframes and volatility regimes.
TARGET_MULTIPLES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PauseBarConfig:
    """The four steps of the setup, with every implicit choice made explicit."""

    lookback: int = 20
    """Bars whose high defines "resistance". The breakout bar must close above
    the highest high of the ``lookback`` bars *before* it."""

    atr_period: int = 20
    """Bars of true range that "big" is measured against, ending before the
    breakout bar so the bar cannot inflate its own benchmark."""

    big_mult: float = 1.5
    """The breakout bar's range must be at least this many ATRs."""

    close_frac: float = 0.60
    """The breakout bar must close in the top ``close_frac`` of its own range -
    a big bar that closes back in the middle is a reversal, not a breakout."""

    pause_mult: float = 0.50
    """The pause bar's range must be at most this fraction of the breakout
    bar's range. This is the whole of "very small"."""

    require_inside: bool = False
    """Also demand a strict inside bar (high and low both contained by the
    breakout bar). Stricter than ``pause_mult`` alone and much rarer."""

    arm_bars: int = 1
    """How many bars the stop order stays live after the pause bar closes.
    ``1`` is the literal reading - enter on the break of the pause bar, meaning
    during the bar immediately after it."""

    max_hold_bars: int = 20
    """Bars after the fill at which an unresolved position is closed at market.
    Bounds the tape each trade has to be walked over, and is itself one of the
    exit geometries reported."""

    buffer_points: float = 1.0
    """Trigger and stop offsets, in the instrument's smallest price increment.
    A trader says "one tick above the high"; this is that tick."""

    min_risk_pips: float = 0.0
    """Skip signals whose planned risk is below this. Zero takes them all,
    which is the honest default - the tiny-risk trades are exactly where the
    cost objection lives, and filtering them out before measuring would hide
    it. Set it to test whether a floor rescues the setup."""

    both_sides: bool = True
    """Trade the mirrored breakdown as well as the breakout."""

    max_spread_mult: float = 3.0
    """Skip a signal if the spread at the arming instant exceeds this multiple
    of the symbol's own mean spread."""

    trail_arm_r: float = 1.0
    """Favourable excursion, in R, after which the trailing stop activates."""

    skip_break_hours: bool = True
    """Skip signals armed inside the instrument's daily maintenance break."""

    fade: bool = False
    """Take the mirror of every signal: sell the break of the pause bar's low
    on a bullish setup, and buy its high on a bearish one.

    A placebo, and the check that the cost-free numbers need. Reported R at mid
    prices comes from a stop-and-target structure whose barriers are crossed on
    the exit side of the book, and that structure can carry an expectation of
    its own that has nothing to do with the signal. If it does, fading the
    signal shows the *same* positive edge rather than its negative - which is
    the difference between an edge and an artifact of measurement.
    """

    context: str = "breakout"
    """What the bar before the pause bar has to be.

    ``"breakout"`` is the idea as stated: big, strong-closing, and closing
    beyond the ``lookback`` extreme. The other two exist to answer "compared
    with what?", because a setup is only worth its conditions if dropping them
    costs something:

    * ``"any"`` - drop the breakout requirement entirely and keep only the
      geometry: a small bar after a directional bar, entered on its break.
      This is the null the setup has to beat.
    * ``"no_breakout"`` - the same geometry, but the prior bar explicitly
      *failed* the level test. The complement of the setup rather than its
      superset, so the two samples are disjoint.
    """

    def entry_span_us(self, interval: str) -> int:
        return self.arm_bars * INTERVAL_MINUTES[interval] * 60 * US

    def hold_span_us(self, interval: str) -> int:
        return self.max_hold_bars * INTERVAL_MINUTES[interval] * 60 * US


# --------------------------------------------------------------------------
# Signals - pure bar arithmetic, no ticks, no costs
# --------------------------------------------------------------------------

def signals(bars: pl.DataFrame, cfg: PauseBarConfig, *,
            interval: str = "5m", point: float | None = None) -> pl.DataFrame:
    """Every (breakout bar, pause bar) pair in ``bars``, with its order levels.

    Returns one row per signal, carrying the instant the order arms - the pause
    bar's close, which under right-edge labelling is its ``ts`` - and the
    trigger and stop it arms with. Nothing here looks forward: every column is
    computed from bars at or before the pause bar.

    The breakout bar and the pause bar must be **contiguous in time**. Bars are
    missing rather than flat across weekends and the maintenance break, so
    adjacency by row is not adjacency by clock, and a "pause bar" three days
    after its breakout is not the setup.
    """
    if bars.height < cfg.lookback + cfg.atr_period + 3:
        return _empty_signals()

    step = timedelta(minutes=INTERVAL_MINUTES[interval])
    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )

    frame = bars.sort("ts_open").with_columns(
        bar_range=pl.col("high") - pl.col("low"),
        tr=true_range,
    ).with_columns(
        # Both benchmarks end on the bar *before* the breakout bar: a bar may
        # not help set the level it is required to exceed, nor the ATR it is
        # required to be large against.
        atr=pl.col("tr").rolling_mean(cfg.atr_period).shift(1),
        res=pl.col("high").rolling_max(cfg.lookback).shift(1),
        sup=pl.col("low").rolling_min(cfg.lookback).shift(1),
    )

    # Everything suffixed _b is the breakout bar, read from the pause bar's row.
    frame = frame.with_columns(
        ts_b=pl.col("ts").shift(1),
        open_b=pl.col("open").shift(1),
        high_b=pl.col("high").shift(1),
        low_b=pl.col("low").shift(1),
        close_b=pl.col("close").shift(1),
        range_b=pl.col("bar_range").shift(1),
        atr_b=pl.col("atr").shift(1),
        res_b=pl.col("res").shift(1),
        sup_b=pl.col("sup").shift(1),
        spread_close_b=pl.col("spread_close").shift(1),
    ).drop_nulls(["atr_b", "res_b", "sup_b"])

    contiguous = pl.col("ts_open") == pl.col("ts_b")
    big = (pl.col("range_b") >= cfg.big_mult * pl.col("atr_b")) & (
        pl.col("atr_b") > 0
    ) & (pl.col("range_b") > 0)
    small = pl.col("bar_range") <= cfg.pause_mult * pl.col("range_b")

    strong_up = (pl.col("close_b") - pl.col("low_b")) >= cfg.close_frac * pl.col("range_b")
    strong_dn = (pl.col("high_b") - pl.col("close_b")) >= cfg.close_frac * pl.col("range_b")
    inside = (
        (pl.col("high") <= pl.col("high_b")) & (pl.col("low") >= pl.col("low_b"))
        if cfg.require_inside
        else pl.lit(True)
    )

    broke_up = pl.col("close_b") > pl.col("res_b")
    broke_dn = pl.col("close_b") < pl.col("sup_b")
    if cfg.context == "breakout":
        long_sig = big & small & inside & strong_up & broke_up
        short_sig = big & small & inside & strong_dn & broke_dn
    elif cfg.context == "no_breakout":
        long_sig = big & small & inside & strong_up & ~broke_up
        short_sig = big & small & inside & strong_dn & ~broke_dn
    elif cfg.context == "any":
        # Geometry only: a small bar after a directional one. "Big" and the
        # level are both dropped, so this is what the setup is measured against.
        up = pl.col("close_b") > pl.col("open_b")
        long_sig = small & inside & up & (pl.col("range_b") > 0)
        short_sig = small & inside & ~up & (pl.col("range_b") > 0)
    else:
        raise ValueError(
            f"unknown context {cfg.context!r}; expected breakout, no_breakout or any"
        )
    if not cfg.both_sides:
        short_sig = pl.lit(False)

    frame = frame.filter(contiguous & (long_sig | short_sig)).with_columns(
        direction=pl.when(long_sig).then(1).otherwise(-1)
    )
    if cfg.fade:
        frame = frame.with_columns(direction=-pl.col("direction"))
    if frame.is_empty():
        return _empty_signals()

    # A bar can satisfy both sides only if it broke a high and a low at once,
    # which the contiguity and close-position filters make vanishingly rare;
    # when it happens the long branch above wins and the short is dropped.
    off = cfg.buffer_points * (point if point is not None else _point_of(bars))

    return (
        frame.with_columns(
            interval=pl.lit(interval),
            arm_ts=pl.col("ts"),  # right-edge label: the pause bar's close
            trigger=pl.when(pl.col("direction") == 1)
            .then(pl.col("high") + off)
            .otherwise(pl.col("low") - off),
            stop=pl.when(pl.col("direction") == 1)
            .then(pl.col("low") - off)
            .otherwise(pl.col("high") + off),
            range_p=pl.col("bar_range"),
            high_p=pl.col("high"),
            low_p=pl.col("low"),
            close_p=pl.col("close"),
            spread_close_p=pl.col("spread_close"),
        )
        .with_columns(
            risk=(pl.col("trigger") - pl.col("stop")).abs(),
            pause_frac=pl.col("range_p") / pl.col("range_b"),
            big_atr=pl.col("range_b") / pl.col("atr_b"),
            expiry_ts=pl.col("arm_ts") + pl.duration(
                minutes=cfg.arm_bars * INTERVAL_MINUTES[interval]
            ),
            level=pl.when(pl.col("direction") == 1)
            .then(pl.col("res_b"))
            .otherwise(pl.col("sup_b")),
        )
        .filter(pl.col("risk") > 0)
        .select(SIGNAL_COLUMNS)
        .sort("arm_ts")
    )


SIGNAL_COLUMNS: tuple[str, ...] = (
    "interval", "arm_ts", "expiry_ts", "direction", "trigger", "stop", "risk",
    "level", "range_b", "range_p", "pause_frac", "atr_b", "big_atr",
    "high_p", "low_p", "close_p", "close_b", "high_b", "low_b",
    "spread_close_b", "spread_close_p",
)


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "interval": pl.String,
            "arm_ts": pl.Datetime("us", "UTC"),
            "expiry_ts": pl.Datetime("us", "UTC"),
            "direction": pl.Int32,
            **{c: pl.Float64 for c in SIGNAL_COLUMNS[4:]},
        }
    )


def _point_of(bars: pl.DataFrame) -> float:
    """Fallback tick size, for calling ``signals`` without a symbol in hand.

    ``run`` always passes ``spec.point``; this only covers direct use on a
    hand-built frame, and it guesses from the price level because that is the
    one thing a bare OHLC frame carries.
    """
    sample = float(bars["close"].head(1).item()) if bars.height else 1.0
    return 10.0 ** -(5 if sample < 20 else (3 if sample < 5000 else 2))


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


def _first_true(mask: np.ndarray) -> int:
    """Index of the first True, or -1."""
    if mask.size == 0:
        return -1
    i = int(mask.argmax())
    return i if mask[i] else -1


# --------------------------------------------------------------------------
# One signal, resolved against the tape
# --------------------------------------------------------------------------

def _resolve(tape: _Tape, sig: dict, cfg: PauseBarConfig, spec: SymbolSpec,
             slip_by_hour: dict[int, float], spread_cap: float,
             bar_ts: np.ndarray, bar_hi: np.ndarray,
             bar_lo: np.ndarray) -> dict | None:
    """Arm the stop order, take the fill if it comes, then price every exit.

    Returns ``None`` when the order was never armed (spread guard, no tape).
    A signal that armed and expired unfilled comes back with ``filled=False``,
    because the fill rate is part of what the setup is being judged on.
    """
    direction = sig["direction"]
    arm_us = sig["arm_ts"]
    expiry_us = sig["expiry_ts"]

    a = tape.index_at(arm_us)
    order_end = tape.index_at(expiry_us)
    if order_end <= a or a >= tape.ts.size:
        return None
    if (tape.ask[a] - tape.bid[a]) > spread_cap:
        return None

    trigger, stop = sig["trigger"], sig["stop"]
    hour = datetime.fromtimestamp(arm_us / US, tz=timezone.utc).hour
    slip = slip_by_hour[hour]

    base = {
        **{k: sig[k] for k in SIGNAL_COLUMNS if k not in ("arm_ts", "expiry_ts")},
        "arm_ts": arm_us,
        "day": datetime.fromtimestamp(arm_us / US, tz=timezone.utc).date(),
        "hour": hour,
        "spread_arm": float(tape.ask[a] - tape.bid[a]),
    }

    # --- the fill -------------------------------------------------------
    # A buy stop triggers on the ask, a sell stop on the bid: it is the price
    # you would be filled at that has to reach the level, not the mid.
    if direction == 1:
        i_fill = _first_true(tape.ask[a:order_end] >= trigger)
    else:
        i_fill = _first_true(tape.bid[a:order_end] <= trigger)
    if i_fill < 0:
        return {**base, "filled": False}

    e = a + i_fill
    entry = (tape.ask[e] + slip) if direction == 1 else (tape.bid[e] - slip)
    entry_mid = (tape.bid[e] + tape.ask[e]) / 2

    # Risk as planned, which is what a position would have been sized on. The
    # realised distance differs because the fill lands past the trigger and
    # slippage pushes it further; both are reported rather than assumed away.
    risk = sig["risk"]
    risk_actual = abs(entry - stop)

    rate = 1.0 if spec.quote_ccy == "USD" else entry
    commission_px = spec.commission_pips(rate) * spec.pip  # round turn, per lot
    # Everything below divides by risk, and the USD-per-price-unit factor is
    # common to numerator and denominator, so R needs no currency conversion.

    hold_end = tape.index_at(int(tape.ts[e]) + cfg.hold_span_us(sig["interval"]))
    hold_end = max(hold_end, e + 1)
    exit_side = tape.bid[e:hold_end] if direction == 1 else tape.ask[e:hold_end]
    if exit_side.size == 0:
        return {**base, "filled": False}

    # --- first passage of the stop -------------------------------------
    hit_stop = (exit_side <= stop) if direction == 1 else (exit_side >= stop)
    i_stop = _first_true(hit_stop)
    n_before_stop = exit_side.size if i_stop < 0 else i_stop

    # A target above the entry and a stop below it cannot both be reached by a
    # single quote, so anything after the stop tick is unreachable and the
    # favourable running extreme only has to be built up to it.
    reachable = exit_side[:n_before_stop]
    if reachable.size:
        run_fav = (np.maximum.accumulate(reachable) if direction == 1
                   else np.minimum.accumulate(reachable))
    else:
        run_fav = reachable

    stop_px = ((tape.bid[e + i_stop] - slip) if direction == 1
               else (tape.ask[e + i_stop] + slip)) if i_stop >= 0 else None
    x_time = hold_end - 1
    time_px = (tape.bid[x_time] - slip) if direction == 1 else (tape.ask[x_time] + slip)

    row = {
        **base,
        "filled": True,
        "entry_ts": int(tape.ts[e]),
        "entry": entry,
        "entry_mid": entry_mid,
        "risk_actual": risk_actual,
        "slip_r": slip / risk,
        "cost_r": (commission_px + float(tape.ask[e] - tape.bid[e]) + 2 * slip) / risk,
        "commission_r": commission_px / risk,
        "risk_bps": risk / entry_mid * 1e4,
        "stopped": i_stop >= 0,
        "stop_ts": int(tape.ts[e + i_stop]) if i_stop >= 0 else None,
        "n_ticks_held": int(exit_side.size),
    }

    # --- excursions over the whole holding window -----------------------
    # Measured on the exit side, so they are excursions in money rather than in
    # mid: the favourable one is what a limit could actually have got.
    if direction == 1:
        row["mfe_r"] = float(exit_side.max() - entry) / risk
        row["mae_r"] = float(entry - exit_side.min()) / risk
    else:
        row["mfe_r"] = float(entry - exit_side.min()) / risk
        row["mae_r"] = float(exit_side.max() - entry) / risk

    # The excursion that was actually *reachable*: how far the trade ran before
    # the original stop retired it. Every fixed target above this one loses,
    # every target below it wins, so this single number carries the whole
    # target curve for this trade.
    if run_fav.size:
        best = float(run_fav[-1])
        row["mfe_pre_stop_r"] = direction * (best - entry) / risk
    else:
        row["mfe_pre_stop_r"] = 0.0

    # Unconditional forward return over the same horizon: no stop, no target,
    # mid to mid. This is the signal on its own, with the trade management and
    # every cost stripped off - the thing that has to be non-zero before any
    # exit geometry can matter.
    row["fwd_r"] = direction * (
        (tape.bid[x_time] + tape.ask[x_time]) / 2 - entry_mid) / risk
    row["fwd_bps"] = direction * (
        (tape.bid[x_time] + tape.ask[x_time]) / 2 / entry_mid - 1) * 1e4

    # --- fixed targets, all resolved from one running extreme -----------
    for mult in TARGET_MULTIPLES:
        target = entry + direction * mult * risk
        if direction == 1:
            i_tp = int(np.searchsorted(run_fav, target, side="left"))
        else:
            i_tp = int(np.searchsorted(-run_fav, -target, side="left"))
        won = i_tp < run_fav.size

        if won:
            exit_px, reason, x = target, "tp", e + i_tp
        elif i_stop >= 0:
            exit_px, reason, x = stop_px, "stop", e + i_stop
        else:
            exit_px, reason, x = time_px, "time", x_time

        gross_px = direction * (exit_px - entry)
        exit_mid = (tape.bid[x] + tape.ask[x]) / 2
        key = _tag(mult)
        row[f"r_{key}"] = (gross_px - commission_px) / risk
        row[f"r_{key}_gross"] = gross_px / risk
        row[f"r_{key}_mid"] = direction * (exit_mid - entry_mid) / risk
        row[f"reason_{key}"] = reason
        row[f"hold_{key}_s"] = (int(tape.ts[x]) - int(tape.ts[e])) / 1e6

    # --- breakeven-then-trail -------------------------------------------
    trail_px, trail_reason, trail_x = _trail_exit(
        tape, e, hold_end, direction, entry, stop, risk, slip, cfg,
        bar_ts, bar_hi, bar_lo)
    trail_mid = (tape.bid[trail_x] + tape.ask[trail_x]) / 2
    row["r_trail"] = (direction * (trail_px - entry) - commission_px) / risk
    row["r_trail_gross"] = direction * (trail_px - entry) / risk
    row["r_trail_mid"] = direction * (trail_mid - entry_mid) / risk
    row["reason_trail"] = trail_reason
    row["hold_trail_s"] = (int(tape.ts[trail_x]) - int(tape.ts[e])) / 1e6

    # --- the pure time stop ---------------------------------------------
    # Same fill, no target and no trail: the position simply runs its holding
    # window out unless the original stop takes it. This is the exit that keeps
    # the most of whatever the signal itself predicts.
    if i_stop >= 0:
        px, reason, x = stop_px, "stop", e + i_stop
    else:
        px, reason, x = time_px, "time", x_time
    row["r_hold"] = (direction * (px - entry) - commission_px) / risk
    row["r_hold_gross"] = direction * (px - entry) / risk
    row["r_hold_mid"] = direction * ((tape.bid[x] + tape.ask[x]) / 2 - entry_mid) / risk
    row["reason_hold"] = reason

    return row


def _tag(mult: float) -> str:
    """Column-safe name for a target multiple: 1.5 -> ``t1_5``."""
    return "t" + f"{mult:g}".replace(".", "_")


def _trail_exit(tape: _Tape, e: int, hold_end: int, direction: int,
                entry: float, stop: float, risk: float, slip: float,
                cfg: PauseBarConfig, bar_ts: np.ndarray, bar_hi: np.ndarray,
                bar_lo: np.ndarray) -> tuple[float, str, int]:
    # Returns (exit price, reason, tick index of the exit).
    """Stop under the previous completed bar, once ``trail_arm_r`` is reached.

    The trail only ever moves in favour, and only ever on a **completed** bar,
    so at any instant the stop in force was computable from bars that had
    already closed. Walking bar by bar rather than tick by tick is what keeps
    that true - a tick-updated trail would be reading the current bar's low
    before the bar had one.
    """
    arm_px = entry + direction * cfg.trail_arm_r * risk
    armed = False
    cur = stop
    j = int(np.searchsorted(bar_ts, tape.ts[e], side="right"))  # first bar to close after entry
    pos = e

    while pos < hold_end:
        bar_close_i = (min(int(np.searchsorted(tape.ts, bar_ts[j], side="left")), hold_end)
                       if j < bar_ts.size else hold_end)
        seg_end = max(bar_close_i, pos + 1)
        seg = (tape.bid[pos:seg_end] if direction == 1 else tape.ask[pos:seg_end])
        if seg.size:
            hit = (seg <= cur) if direction == 1 else (seg >= cur)
            i_hit = _first_true(hit)
            if i_hit >= 0:
                x = pos + i_hit
                px = ((tape.bid[x] - slip) if direction == 1
                      else (tape.ask[x] + slip))
                return px, ("trail" if armed else "stop"), x
            if not armed:
                reached = (seg.max() >= arm_px) if direction == 1 else (seg.min() <= arm_px)
                armed = bool(reached)

        pos = seg_end
        if j < bar_ts.size and armed:
            # The bar that just closed now sets the stop, but only outward.
            candidate = bar_lo[j] if direction == 1 else bar_hi[j]
            cur = max(cur, candidate) if direction == 1 else min(cur, candidate)
        j += 1
        if j > bar_ts.size:
            break

    x = hold_end - 1
    px = (tape.bid[x] - slip) if direction == 1 else (tape.ask[x] + slip)
    return px, "time", x


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(symbol: str, cfg: PauseBarConfig | None = None, *,
        intervals: Sequence[str] = ("1m", "5m", "15m", "1h"),
        start=None, end=None, split: str | None = None,
        allow_test: bool = False, cost: CostModel | None = None,
        verbose: bool = False) -> pl.DataFrame:
    """Backtest the setup on one symbol across several timeframes.

    Signals from every timeframe are collected first and resolved in a single
    pass over each month of tape, so adding a timeframe costs signal detection
    rather than another walk through 283 million gold ticks.
    """
    cfg = cfg or PauseBarConfig()
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    spread_cap = cost.spread_pips() * spec.pip * cfg.max_spread_mult
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}

    base = load_bars(symbol, "1m", start=start, end=end, split=split,
                     allow_test=allow_test)
    if base.is_empty():
        return pl.DataFrame()

    # Signals, and the bar grid each timeframe's trail walks on.
    sig_frames: list[pl.DataFrame] = []
    grids: dict[str, pl.DataFrame] = {}
    for interval in intervals:
        bars = base if interval == "1m" else resample_bars(base, interval)
        grids[interval] = bars.select("ts", "high", "low")
        found = signals(bars, cfg, interval=interval, point=spec.point)
        if cfg.min_risk_pips > 0:
            found = found.filter(pl.col("risk") >= cfg.min_risk_pips * spec.pip)
        if cfg.skip_break_hours and spec.daily_break_utc is not None:
            lo, hi = spec.daily_break_utc
            found = found.filter(
                ~pl.col("arm_ts").dt.hour().is_between(lo, hi - 1)
            )
        if verbose:
            print(f"  {symbol} {interval}: {found.height} signals")
        sig_frames.append(found)

    sigs = pl.concat(sig_frames) if sig_frames else _empty_signals()
    if sigs.is_empty():
        return pl.DataFrame()
    sigs = sigs.sort("arm_ts")

    # Longest holding window across the timeframes in play, so a trade armed on
    # the last day of a month still has its whole life on the loaded tape.
    tail = timedelta(microseconds=max(
        cfg.hold_span_us(i) + cfg.entry_span_us(i) for i in intervals
    )) + timedelta(hours=6)

    grid_np = {
        k: (v["ts"].cast(pl.Int64).to_numpy(), v["high"].to_numpy(),
            v["low"].to_numpy())
        for k, v in grids.items()
    }

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

        # allow_test is safe here: the window is already bounded by signals
        # that themselves came from a split-guarded bar load.
        ticks = load_ticks(symbol, start=m_start, end=m_end + tail, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(ts=ticks["ts"].cast(pl.Int64).to_numpy(),
                     bid=ticks["bid"].to_numpy(), ask=ticks["ask"].to_numpy())
        del ticks

        for sig in batch.iter_rows(named=True):
            sig = dict(sig)
            sig["arm_ts"] = int(sig["arm_ts"].timestamp() * US)
            sig["expiry_ts"] = int(sig["expiry_ts"].timestamp() * US)
            bts, bhi, blo = grid_np[sig["interval"]]
            rec = _resolve(tape, sig, cfg, spec, slip_by_hour, spread_cap,
                           bts, bhi, blo)
            if rec is not None:
                rows.append(rec)
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {len(rows)} resolved")

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort("arm_ts")
