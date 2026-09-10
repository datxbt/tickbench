"""Session opening-range breakout - a tick-level model of the MT5 expert.

This reproduces ``XAUUSD_SessionBreakout_2026.mq5`` closely enough to test the
claims in its header, and it is written against ticks rather than bars because
the strategy's entries, stops and targets are all stop orders: on a bar-level
engine a bar whose range spans both the entry and the stop has to be resolved
by assumption, and the assumption would be worth more than the edge.

What is reproduced
------------------
* The bracket is the high/low of the M1 bars over ``[hour:00, hour:00 + range)``
  on the **mid**. The expert reads bid-priced bars and shifts them up by half
  the spread, which is the same thing.
* A buy stop at the high triggers on the **ask**, a sell stop at the low on the
  **bid** - the expert's own comment says so, and it is how MT5 behaves.
* Manual OCO: the first fill retires the other side.
* The stop is the opposite edge of the bracket; the target is
  ``target_mult * width`` from the entry actually obtained (the expert
  re-anchors the take profit after the fill, and so does this).
* Unfilled orders expire ``order_expiry_hours`` after the bracket closes. A
  filled position is *not* closed then - it runs to its stop, its target, or
  the daily flatten.
* The daily flatten is anchored to the 16:58 New York halt and therefore moves
  with US DST, minus a lead of ``flat_lead_minutes``. Friday closes earlier.
* Range guards (``min_range_pct`` / ``max_range_pct``), the Sunday skip, the
  spread guard, and the "price already outside the bracket" skip.

What is not
-----------
* Position sizing. Every result is reported per unit risk (``r_multiple``),
  which is scale-free, so lots, volatility targeting and the daily loss halt
  change the scale of the equity curve but not its shape or its sign. The
  expert's header concedes its own figures were taken at a fixed 0.02 lots
  with targeting off.
* Requotes, rejections, margin and swap. The feed carries no size.

Costs
-----
Spread is not modelled: the ticks carry a real bid and a real ask, and fills
cross them. Commission comes from the published contract terms. Slippage is
the one assumed term and it comes from :class:`qlab.costs.CostModel`, applied
to market fills - the entry, a stop-out, and the flatten - but never to a take
profit, which is a limit and fills at its price or not at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import load_bars, load_ticks
from ..symbols import get_spec

US = 1_000_000  # microseconds in a second


# --------------------------------------------------------------------------
# Window sets, transcribed from the expert's LoadPreset()
# --------------------------------------------------------------------------

Window = tuple[int, int, float]  # (utc_hour, range_minutes, target_multiple)

PRESETS: dict[str, tuple[Window, ...]] = {
    "GEO_2026_NO_H13": ((0, 60, 3.0), (1, 60, 3.0), (2, 60, 3.0), (4, 60, 3.0),
                        (5, 60, 3.0), (6, 60, 3.0), (14, 60, 3.0)),
    "GEO_2026": ((0, 60, 3.0), (1, 60, 3.0), (2, 60, 3.0), (4, 60, 3.0),
                 (5, 60, 3.0), (6, 60, 3.0), (13, 60, 3.0), (14, 60, 3.0)),
    "TOP8_2026": ((0, 30, 3.0), (1, 60, 3.0), (2, 30, 3.0), (5, 30, 3.0),
                  (8, 60, 3.0), (14, 60, 3.0), (15, 60, 3.0), (18, 60, 3.0)),
    "ORIGINAL": ((0, 30, 1.0), (1, 60, 3.0), (2, 15, 3.0), (4, 30, 3.0),
                 (5, 60, 2.0), (6, 60, 3.0), (13, 30, 3.0), (14, 15, 2.0)),
}


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------

def _nth_sunday(year: int, month: int, nth: int) -> date:
    first = date(year, month, 1)
    offset = (6 - first.weekday()) % 7  # date.weekday(): Monday=0 .. Sunday=6
    return date(year, month, 1 + offset + 7 * (nth - 1))


def is_us_dst(day: date) -> bool:
    """US DST: second Sunday in March to first Sunday in November."""
    return _nth_sunday(day.year, 3, 2) <= day < _nth_sunday(day.year, 11, 1)


def flatten_second_utc(day: date, *, lead_minutes: int = 5,
                       friday_close_hour: int = 20) -> int:
    """UTC second-of-day at which everything is closed for the day.

    The Exness gold session halts at 16:58 New York, so in UTC the halt moves
    with US DST even though the server clock does not. The expert flattens a
    few minutes ahead of it; Friday closes earlier still.
    """
    if day.weekday() == 4 and friday_close_hour < 24:
        return friday_close_hour * 3600
    halt = (20 * 3600 + 58 * 60) if is_us_dst(day) else (21 * 3600 + 58 * 60)
    return halt - lead_minutes * 60


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakoutConfig:
    """Everything the expert exposes as an input that changes a decision."""

    windows: tuple[Window, ...]
    order_expiry_hours: float = 4.0
    flat_lead_minutes: int = 5
    friday_close_hour: int = 20
    min_range_pct: float = 0.05
    max_range_pct: float = 2.00
    skip_sunday: bool = True
    max_spread_mult: float = 3.4
    """Spread cap as a multiple of the symbol's own mean spread.

    The expert states this as ``InpMaxSpreadUSD = 0.30`` USD/oz, which is 3.4x
    gold's measured mean. Expressed as a multiple it carries across
    instruments; expressed in dollars it would never bind on FX.
    """
    stop_mult: float | None = None
    """Stop distance as a multiple of the bracket width, measured from the fill.

    ``None`` is the expert's own rule: the stop sits on the opposite edge of
    the bracket. That makes the realised risk slightly more than one width,
    because the fill lands beyond the edge that triggered it, and it fixes the
    risk-to-reward ratio at whatever the bracket happens to be. Setting this
    frees that ratio.
    """
    max_hold_hours: float | None = None
    """Close the position this long after the fill, if it is still open.

    ``None`` is the expert's own behaviour: a fill runs to its stop, its target
    or the daily flatten, whichever comes first. A finite value adds a time
    stop, which is the natural exit when the entry signal's information decays -
    and the attribution says this one's does, within the hour.
    """
    min_range_bars_frac: float = 0.5
    """Fraction of the bracket's minutes that must have quoted, or the day is
    skipped - the expert's "no M1 data for the bracket" branch."""

    @classmethod
    def preset(cls, name: str, **kwargs) -> "BreakoutConfig":
        return cls(windows=PRESETS[name], **kwargs)


# --------------------------------------------------------------------------
# Bracket construction
# --------------------------------------------------------------------------

def _brackets(bars: pl.DataFrame, hour: int, range_min: int,
              cfg: BreakoutConfig) -> pl.DataFrame:
    """High/low of the mid over [hour:00, hour:00+range) for each session day.

    Keys off ``ts_open``: bars are labelled at the interval end, so a bar
    covering [00:59, 01:00) carries ``ts`` 01:00 and would otherwise be filed
    under the next hour's bracket.
    """
    span = (
        bars.filter(
            (pl.col("ts_open").dt.hour() == hour)
            & (pl.col("ts_open").dt.minute() < range_min)
        )
        .group_by(pl.col("ts_open").dt.date().alias("day"))
        .agg(hi=pl.col("high").max(), lo=pl.col("low").min(), n_bars=pl.len())
    )
    return (
        span.filter(pl.col("n_bars") >= range_min * cfg.min_range_bars_frac)
        .with_columns(
            width=pl.col("hi") - pl.col("lo"),
            width_pct=(pl.col("hi") - pl.col("lo"))
            / ((pl.col("hi") + pl.col("lo")) / 2) * 100,
        )
        .filter(
            (pl.col("width") > 0)
            & (pl.col("width_pct") >= cfg.min_range_pct)
            & (pl.col("width_pct") <= cfg.max_range_pct)
        )
        .sort("day")
    )


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def _first_true(mask: np.ndarray) -> int:
    """Index of the first True, or -1."""
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
    """A fill. Identical for every exit geometry, so it is resolved once."""

    i: int              # tick index of the fill
    direction: int
    price: float        # fill price, slippage included
    mid: float
    flat_us: int        # the day's flatten instant


def _find_entry(tape: _Tape, day: date, hour: int, range_min: int,
                hi: float, lo: float, cfg: BreakoutConfig, slip: float,
                spread_cap: float) -> _Entry | None:
    """Arm the bracket and wait for the first side to trigger.

    Split out from the exit so a sweep over exit geometries walks the tape once
    and cannot accidentally let a different entry leak into a different cell.
    """
    midnight = int(datetime(day.year, day.month, day.day,
                            tzinfo=timezone.utc).timestamp()) * US
    arm_us = midnight + (hour * 3600 + range_min * 60) * US
    expiry_us = arm_us + int(cfg.order_expiry_hours * 3600) * US
    flat_us = midnight + flatten_second_utc(
        day, lead_minutes=cfg.flat_lead_minutes,
        friday_close_hour=cfg.friday_close_hour) * US
    if arm_us >= flat_us:
        return None

    a = tape.index_at(arm_us)
    order_end = tape.index_at(min(expiry_us, flat_us))
    if order_end <= a:
        return None

    # The expert's guards, evaluated on the first tick after the bracket closes.
    if (tape.ask[a] - tape.bid[a]) > spread_cap:
        return None
    if tape.ask[a] >= hi or tape.bid[a] <= lo:
        return None  # already broken out - the expert sits the day out

    ask = tape.ask[a:order_end]
    bid = tape.bid[a:order_end]
    i_buy = _first_true(ask >= hi)     # buy stop triggers on the ask
    i_sell = _first_true(bid <= lo)    # sell stop triggers on the bid
    if i_buy < 0 and i_sell < 0:
        return None                    # expired unfilled
    if i_sell < 0 or (0 <= i_buy < i_sell):
        direction, i_fill = 1, i_buy
    else:
        direction, i_fill = -1, i_sell

    e = a + i_fill
    price = (tape.ask[e] + slip) if direction == 1 else (tape.bid[e] - slip)
    return _Entry(i=e, direction=direction, price=price,
                  mid=(tape.bid[e] + tape.ask[e]) / 2, flat_us=flat_us)


def _simulate_window(tape: _Tape, day: date, hour: int, range_min: int,
                     target_mult: float, hi: float, lo: float, width: float,
                     cfg: BreakoutConfig, slip: float, spec,
                     spread_cap: float) -> dict | None:
    """One window on one day. Returns a trade record, or None if it never fired."""
    ent = _find_entry(tape, day, hour, range_min, hi, lo, cfg, slip, spread_cap)
    if ent is None:
        return None
    e, direction, entry, entry_mid = ent.i, ent.direction, ent.price, ent.mid

    # The stop is the opposite edge of the bracket unless a multiple of the
    # width is asked for explicitly; the target is always a multiple of it,
    # measured from the fill actually obtained.
    if cfg.stop_mult is None:
        stop = lo if direction == 1 else hi
    else:
        stop = entry - direction * cfg.stop_mult * width
    target = entry + direction * target_mult * width

    # Manage to stop, target or the flatten. A stop is a market order and pays
    # slippage; a target is a limit and does not.
    close_us = ent.flat_us
    if cfg.max_hold_hours is not None:
        close_us = min(close_us,
                       int(tape.ts[e]) + int(cfg.max_hold_hours * 3600) * US)
    pos_end = max(tape.index_at(close_us), e + 1)
    pbid = tape.bid[e:pos_end]
    pask = tape.ask[e:pos_end]
    if direction == 1:
        i_stop, i_tp = _first_true(pbid <= stop), _first_true(pbid >= target)
    else:
        i_stop, i_tp = _first_true(pask >= stop), _first_true(pask <= target)

    if i_stop < 0 and i_tp < 0:
        x, reason = pos_end - 1, "time" if cfg.max_hold_hours else "flat"
        exit_px = (tape.bid[x] - slip) if direction == 1 else (tape.ask[x] + slip)
    elif i_tp < 0 or (0 <= i_stop <= i_tp):
        x, reason = e + i_stop, "stop"
        exit_px = (tape.bid[x] - slip) if direction == 1 else (tape.ask[x] + slip)
    else:
        x, reason = e + i_tp, "tp"
        exit_px = target

    exit_mid = (tape.bid[x] + tape.ask[x]) / 2

    # Work in price units first, then convert once. For a symbol quoted in
    # something other than USD the pip's dollar value moves with the rate, so
    # both the commission and the conversion have to use the trade's own price
    # rather than a constant - on USDJPY that is a 30% swing across the corpus.
    rate = 1.0 if spec.quote_ccy == "USD" else entry
    usd_per_price_unit = (spec.contract_size or 1.0) / rate
    commission_px = spec.commission_pips(rate) * spec.pip   # round turn, per lot

    gross_px = direction * (exit_px - entry)
    net_px = gross_px - commission_px
    mid_px = direction * (exit_mid - entry_mid)

    gross = gross_px * usd_per_price_unit
    commission = commission_px * usd_per_price_unit
    net = net_px * usd_per_price_unit
    mid_pnl = mid_px * usd_per_price_unit
    # The distance actually risked, so two geometries with different stops
    # are compared per unit of risk rather than per unit of bracket.
    risk = abs(entry - stop) * usd_per_price_unit
    risk_width = width * usd_per_price_unit

    return {
        "day": day, "hour": hour, "range_min": range_min,
        "target_mult": target_mult, "direction": direction,
        "entry_ts": int(tape.ts[e]), "exit_ts": int(tape.ts[x]),
        "entry": entry, "exit": exit_px, "stop": stop, "target": target,
        "entry_mid": entry_mid, "exit_mid": exit_mid,
        "hi": hi, "lo": lo, "width": width,
        "width_pct": width / ((hi + lo) / 2) * 100,
        "reason": reason,
        "gross_usd": gross, "commission_usd": commission, "net_usd": net,
        "mid_pnl_usd": mid_pnl, "risk_usd": risk_width, "risk_actual_usd": risk,
        "r_multiple": net / risk_width, "r_gross": gross / risk_width,
        "r_mid": mid_pnl / risk_width, "r_risk": net / risk,
        "hold_s": (int(tape.ts[x]) - int(tape.ts[e])) / 1e6,
    }


def run(symbol: str, cfg: BreakoutConfig, *, start=None, end=None,
        split: str | None = None, allow_test: bool = False,
        cost: CostModel | None = None, verbose: bool = False) -> pl.DataFrame:
    """Backtest one window set on one symbol, month by month over the ticks."""
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    spread_cap = cost.spread_pips() * spec.pip * cfg.max_spread_mult

    bars = load_bars(symbol, "1m", start=start, end=end, split=split,
                     allow_test=allow_test,
                     columns=["ts", "ts_open", "high", "low"])
    if bars.is_empty():
        return pl.DataFrame()

    # Per-hour slippage in price units, so a market fill can be moved by it.
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}

    months = (
        bars.select(pl.col("ts_open").dt.year().alias("y"),
                    pl.col("ts_open").dt.month().alias("m"))
        .unique().sort(["y", "m"]).rows()
    )

    rows: list[dict] = []
    for y, m in months:
        m_start = datetime(y, m, 1, tzinfo=timezone.utc)
        m_end = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc)
        mbars = bars.filter((pl.col("ts_open") >= m_start)
                            & (pl.col("ts_open") < m_end))
        if mbars.is_empty():
            continue
        ticks = load_ticks(symbol, start=m_start, end=m_end, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(ts=ticks["ts"].cast(pl.Int64).to_numpy(),
                     bid=ticks["bid"].to_numpy(),
                     ask=ticks["ask"].to_numpy())
        del ticks

        for hour, range_min, target_mult in cfg.windows:
            slip = slip_by_hour[hour]
            for row in _brackets(mbars, hour, range_min, cfg).iter_rows(named=True):
                day = row["day"]
                if cfg.skip_sunday and day.weekday() == 6:
                    continue
                rec = _simulate_window(
                    tape, day, hour, range_min, target_mult,
                    row["hi"], row["lo"], row["width"], cfg, slip,
                    spec, spread_cap)
                if rec is not None:
                    rows.append(rec)
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {len(rows)} trades cumulative",
                  flush=True)
        del tape

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("entry_ts")


# --------------------------------------------------------------------------
# Exit-geometry sweep
# --------------------------------------------------------------------------

def sweep_exits(symbol: str, cfg: BreakoutConfig,
                stop_mults: Sequence[float], target_mults: Sequence[float],
                *, start=None, end=None, allow_test: bool = False,
                cost: CostModel | None = None, verbose: bool = False) -> dict:
    """Net R for every (stop, target) geometry, over one shared set of entries.

    The entry does not depend on the exit, so the tape is walked once and every
    cell is resolved against the same fills. That is not only faster - it means
    two cells can never differ because they happened to catch different trades,
    which is the usual way a geometry sweep flatters itself.

    Each cell is resolved by first passage to its own two price levels. The
    running extremes of the post-entry tape are monotone by construction, so
    "when did price first reach X" is a binary search on them rather than a
    scan, and the whole grid costs one search per level instead of one pass per
    cell.

    Risk is the distance to that cell's own stop, so ``r`` is per unit of risk
    actually taken and the cells are directly comparable. A stop is a market
    order and pays slippage; a target is a limit and fills at its price.

    Returns the grid axes, the per-trade day and direction, and an
    ``(n_trades, n_stops, n_targets)`` array of net R.
    """
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=None)
    spread_cap = cost.spread_pips() * spec.pip * cfg.max_spread_mult
    sm = np.asarray(stop_mults, dtype=float)
    tm = np.asarray(target_mults, dtype=float)

    bars = load_bars(symbol, "1m", start=start, end=end, allow_test=allow_test,
                     columns=["ts", "ts_open", "high", "low"])
    if bars.is_empty():
        raise ValueError(f"no bars for {symbol} in the requested window")
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}
    months = (bars.select(pl.col("ts_open").dt.year().alias("y"),
                          pl.col("ts_open").dt.month().alias("m"))
              .unique().sort(["y", "m"]).rows())

    days: list[date] = []
    dirs: list[int] = []
    widths: list[float] = []
    cells: list[np.ndarray] = []

    for y, m in months:
        m_start = datetime(y, m, 1, tzinfo=timezone.utc)
        m_end = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc)
        mbars = bars.filter((pl.col("ts_open") >= m_start)
                            & (pl.col("ts_open") < m_end))
        if mbars.is_empty():
            continue
        ticks = load_ticks(symbol, start=m_start, end=m_end, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(ts=ticks["ts"].cast(pl.Int64).to_numpy(),
                     bid=ticks["bid"].to_numpy(), ask=ticks["ask"].to_numpy())
        del ticks

        for hour, range_min, _ in cfg.windows:
            slip = slip_by_hour[hour]
            for row in _brackets(mbars, hour, range_min, cfg).iter_rows(named=True):
                day, hi, lo, width = row["day"], row["hi"], row["lo"], row["width"]
                if cfg.skip_sunday and day.weekday() == 6:
                    continue
                ent = _find_entry(tape, day, hour, range_min, hi, lo, cfg,
                                  slip, spread_cap)
                if ent is None:
                    continue

                e, d, entry = ent.i, ent.direction, ent.price
                close_us = ent.flat_us
                if cfg.max_hold_hours is not None:
                    close_us = min(close_us, int(tape.ts[e])
                                   + int(cfg.max_hold_hours * 3600) * US)
                pos_end = max(tape.index_at(close_us), e + 1)
                n = pos_end - e

                stop_px = entry - d * sm * width          # (S,)
                tgt_px = entry + d * tm * width           # (T,)
                if d == 1:
                    px = tape.bid[e:pos_end]              # a long exits on the bid
                    run_lo = np.minimum.accumulate(px)    # non-increasing
                    run_hi = np.maximum.accumulate(px)    # non-decreasing
                    i_stop = np.searchsorted(-run_lo, -stop_px, side="left")
                    i_tp = np.searchsorted(run_hi, tgt_px, side="left")
                else:
                    px = tape.ask[e:pos_end]              # a short exits on the ask
                    run_hi = np.maximum.accumulate(px)
                    run_lo = np.minimum.accumulate(px)
                    i_stop = np.searchsorted(run_hi, stop_px, side="left")
                    i_tp = np.searchsorted(-run_lo, -tgt_px, side="left")

                IS, IT = i_stop[:, None], i_tp[None, :]
                never = (IS >= n) & (IT >= n)
                stopped = IS <= IT                        # a tie goes to the stop
                x = np.clip(np.minimum(IS, IT), 0, n - 1)
                market = px[x] - d * slip                 # stop and time exits
                tgt_grid = np.broadcast_to(tgt_px, (sm.size, tm.size))
                exit_px = np.where(never, px[n - 1] - d * slip,
                                   np.where(stopped, market, tgt_grid))

                rate = 1.0 if spec.quote_ccy == "USD" else entry
                commission_px = spec.commission_pips(rate) * spec.pip
                net_px = d * (exit_px - entry) - commission_px
                cells.append(net_px / (sm[:, None] * width))
                days.append(day)
                dirs.append(d)
                widths.append(width)
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {len(days)} entries cumulative", flush=True)
        del tape

    return {
        "stop_mults": sm, "target_mults": tm,
        "day": np.array(days), "direction": np.array(dirs),
        "width": np.array(widths),
        "r": np.stack(cells) if cells else np.empty((0, sm.size, tm.size)),
    }
