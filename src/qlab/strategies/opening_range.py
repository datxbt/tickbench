"""The 5-minute opening range breakout, as specified by Zarattini, Barbon and
Aziz (2024).

The claim being tested
----------------------
*A Profitable Day Trading Strategy For The U.S. Equity Market* (SSRN 4729284)
states the rule in four lines:

1. Take the first ``n`` minutes of the regular session (their headline is
   ``n = 5``, 09:30-09:35 New York) as the **opening range**.
2. The sign of that first candle picks the side, and *only* that side. A bullish
   opening candle arms a buy stop at the range high; a bearish one arms a sell
   stop at the range low; a doji trades nothing. The paper is emphatic that the
   opposite break is not taken.
3. If the stop order fills, a protective stop goes **10% of the 14-day ATR**
   away from the fill.
4. There is no profit target. A position that survives is closed at the bell.

Their base version of this, run across ~7,000 US stocks from 2016 to 2023,
returned 29% total - worse than the index. The paper's contribution is the
filter that rescues it: restrict to **Stocks in Play**, defined by *relative
volume* in the opening range,

    RelVol = OR volume today / mean OR volume over the previous 14 days,

keep only ``RelVol >= 1``, and trade the top 20 names by RelVol each day. That
version returns 1,600% with a Sharpe of 2.81.

What this corpus can and cannot test
------------------------------------
This project holds four spot CFDs and no equity cross-section, so the two halves
of the paper are not equally testable, and the report says which is which before
any number.

**The geometry transfers; the cross-section does not.** Rules 1-4 are a
single-instrument rule and are implemented exactly. "Trade the top 20 names by
RelVol" needs a universe of names to rank, and there is none - with one
instrument per day the ranking is a tautology. What *is* implementable is the
part of the filter that carries the paper's mechanism: RelVol is a per-name,
per-day quantity, so ``RelVol >= 1`` is a real filter on a single instrument,
and the paper's Figure 4 - average PnL sorted into RelVol buckets - reproduces
as a sort. If the mechanism is "abnormal opening activity predicts an intraday
trend", the bucket sort is where it has to show up. See
:func:`relvol_buckets`.

**Volume is quote updates, not shares.** The feed carries no size. The stand-in
for opening-range volume is the **tick count** in the window, which is the
number of quote revisions the broker published. It correlates with activity for
the same reason volume does and it is not the same thing; every appearance of
RelVol here is that proxy. Because RelVol is a *ratio* to the same instrument's
own trailing mean, the units cancel, which is the one thing that makes the proxy
usable at all.

**The universe screens are not applicable.** Price > $5, 14-day volume > 1M
shares and ATR > $0.50 exist to throw out illiquid penny stocks. USTEC and gold
pass all three trivially and no screening is performed.

Why this is resolved on ticks
-----------------------------
The stop is 10% of a 14-day ATR. On USTEC that is roughly 25 index points
against 1-minute bars whose range in the first hour routinely exceeds it, so a
bar-level engine would have to *assume* whether the entry or the stop came first
inside the bar that contains both - and on this geometry that assumption is
worth more than the edge. The entry and every exit are therefore resolved
against the tick tape, with a buy stop triggering on the ask and a sell stop on
the bid, exactly as :mod:`qlab.strategies.session_breakout` does.

Costs
-----
Spread is paid, not modelled: fills cross a real bid and a real ask. Commission
comes from the published contract terms. Slippage is the one assumed term, taken
from :class:`qlab.costs.CostModel` and applied to market orders - the stop entry,
a stop-out, the closing bell - but never to a profit target, which is a limit.
The session runs 09:30-16:00 New York and never spans the 21:00 UTC financing
point, so there is no swap on any trade in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import load_bars, load_ticks
from ..symbols import get_spec

NY = "America/New_York"
US = 1_000_000  # microseconds in a second

#: How far the exit sweep runs, in multiples of the planned risk. ``None`` is
#: the paper's own rule - no target, close at the bell.
TARGET_MULTIPLES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
HOLD = "hold"

#: The stop, as a fraction of the 14-day ATR. 0.10 is the paper's.
STOP_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)

#: The opening range lengths the paper compares in its Section 5.
OR_MINUTES: tuple[int, ...] = (5, 15, 30, 60)


def tag(multiple: float | None) -> str:
    return HOLD if multiple is None else f"t{multiple:g}".replace(".", "_")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ORBConfig:
    """Everything that changes a decision.

    ``direction_rule``
        ``"paper"`` arms only the side the opening candle points at, which is
        the paper's rule. ``"both"`` arms both sides and takes whichever
        triggers first - the control that says whether the sign of the opening
        candle does any work. ``"contra"`` arms only the *opposite* side, which
        is the sharper version of the same question.

    ``relvol_min``
        The Stocks-in-Play filter, on the tick-count proxy. ``None`` disables
        it, which is the paper's Base Strategy.
    """

    or_minutes: int = 5
    stop_atr_frac: float = 0.10
    atr_days: int = 14
    relvol_days: int = 14
    relvol_min: float | None = None
    direction_rule: str = "paper"
    open_min: int = 9 * 60 + 30       # 09:30 New York
    close_min: int = 16 * 60          # 16:00 New York
    min_session_bars: int = 300
    open_tolerance_min: int = 5
    """How late the first quote of the session may be and still count as an
    open. A holiday half-session is dropped by ``min_session_bars`` instead."""

    def __post_init__(self) -> None:
        if self.direction_rule not in ("paper", "both", "contra"):
            raise ValueError(f"unknown direction_rule {self.direction_rule!r}")
        if self.or_minutes <= 0:
            raise ValueError("or_minutes must be positive")
        if self.open_min + self.or_minutes >= self.close_min:
            raise ValueError("the opening range does not fit inside the session")

    @property
    def or_end_min(self) -> int:
        return self.open_min + self.or_minutes


# --------------------------------------------------------------------------
# Daily context: the opening range, the ATR and the relative volume
# --------------------------------------------------------------------------

def _ny_minutes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        ny=pl.col("ts_open").dt.convert_time_zone(NY)
    ).with_columns(
        nyd=pl.col("ny").dt.date(),
        mins=pl.col("ny").dt.hour().cast(pl.Int32) * 60
        + pl.col("ny").dt.minute().cast(pl.Int32),
    )


def daily_context(
    symbol: str,
    cfg: ORBConfig = ORBConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
) -> pl.DataFrame:
    """One row per session: its opening range, its ATR and its relative volume.

    Everything here is a *decision input*, so everything here is lagged into
    place. ``atr`` is the mean true range over the ``atr_days`` sessions that
    ended **before** this one, and ``relvol`` divides today's opening-range tick
    count by the mean over the previous ``relvol_days`` sessions - neither can
    see a price that has not printed by 09:35.

    The warmup is loaded from before the split so that the first evaluable
    session already has a full ATR, and rows that are only there to warm the
    indicators are dropped before returning.
    """
    warmup = timedelta(days=int(2.5 * max(cfg.atr_days, cfg.relvol_days)) + 20)
    bars = load_bars(
        symbol, "1m", split=split, start=start, end=end, warmup=warmup,
        allow_test=allow_test,
        columns=["ts", "ts_open", "open", "high", "low", "close", "n_ticks"],
    )
    bars = _ny_minutes(bars).filter(
        (pl.col("mins") >= cfg.open_min) & (pl.col("mins") < cfg.close_min)
    ).sort("ny")

    session = (
        bars.group_by("nyd")
        .agg(
            first_min=pl.col("mins").first(),
            n_bars=pl.len(),
            rth_high=pl.col("high").max(),
            rth_low=pl.col("low").min(),
            rth_close=pl.col("close").last(),
            close_ts=pl.col("ts").last(),
            is_warmup=pl.col("is_warmup").any() if "is_warmup" in bars.columns
            else pl.lit(False),
        )
        .filter(
            (pl.col("n_bars") >= cfg.min_session_bars)
            & (pl.col("first_min") <= cfg.open_min + cfg.open_tolerance_min)
        )
        .sort("nyd")
    )

    opening = (
        bars.filter(pl.col("mins") < cfg.or_end_min)
        .group_by("nyd")
        .agg(
            or_open=pl.col("open").first(),
            or_high=pl.col("high").max(),
            or_low=pl.col("low").min(),
            or_close=pl.col("close").last(),
            or_ticks=pl.col("n_ticks").sum().cast(pl.Float64),
            or_bars=pl.len(),
            or_end_ts=pl.col("ts").last(),
        )
        .filter(pl.col("or_bars") >= max(1, cfg.or_minutes - 1))
    )

    ctx = session.join(opening, on="nyd", how="inner").sort("nyd")

    # True range against the *previous session's* close, then a trailing mean
    # that excludes today. Wilder's own smoothing would differ in the third
    # decimal; the paper says "average true range over the previous 14 days"
    # and this is that, read literally.
    ctx = ctx.with_columns(
        prev_close=pl.col("rth_close").shift(1),
    ).with_columns(
        tr=pl.max_horizontal(
            pl.col("rth_high") - pl.col("rth_low"),
            (pl.col("rth_high") - pl.col("prev_close")).abs(),
            (pl.col("rth_low") - pl.col("prev_close")).abs(),
        )
    ).with_columns(
        atr=pl.col("tr").shift(1).rolling_mean(cfg.atr_days),
        or_ticks_mean=pl.col("or_ticks").shift(1).rolling_mean(cfg.relvol_days),
    ).with_columns(
        relvol=pl.col("or_ticks") / pl.col("or_ticks_mean"),
        or_body=pl.col("or_close") - pl.col("or_open"),
        or_width=pl.col("or_high") - pl.col("or_low"),
    ).with_columns(
        signal=pl.when(pl.col("or_body") > 0).then(1)
        .when(pl.col("or_body") < 0).then(-1)
        .otherwise(0).cast(pl.Int8)
    )

    if "is_warmup" in ctx.columns:
        ctx = ctx.filter(~pl.col("is_warmup")).drop("is_warmup")
    return ctx.drop_nulls(["atr", "relvol"]).filter(pl.col("atr") > 0)


def eligible(ctx: pl.DataFrame, cfg: ORBConfig) -> pl.DataFrame:
    """The sessions the configuration would actually trade."""
    out = ctx
    if cfg.direction_rule != "both":
        out = out.filter(pl.col("signal") != 0)   # the paper's doji rule
    if cfg.relvol_min is not None:
        out = out.filter(pl.col("relvol") >= cfg.relvol_min)
    return out


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


def _find_entry(tape: _Tape, row: dict, cfg: ORBConfig, slip: float) -> _Entry | None:
    """Arm the stop order at the opening range edge and wait.

    A buy stop triggers on the **ask** and a sell stop on the **bid**, because
    that is the side of the book a market-if-touched order has to reach. The
    order is armed the instant the opening range closes and stands until the
    bell; there is no expiry in the paper.
    """
    arm_us = int(row["or_end_ts"].timestamp() * US)
    flat_us = int(row["close_ts"].timestamp() * US)
    a = tape.index_at(arm_us)
    end = tape.index_at(flat_us)
    if end <= a:
        return None

    hi, lo = row["or_high"], row["or_low"]
    signal = int(row["signal"])
    if cfg.direction_rule == "paper":
        sides = (signal,)
    elif cfg.direction_rule == "contra":
        sides = (-signal,)
    else:
        sides = (1, -1)
    sides = tuple(s for s in sides if s != 0)
    if not sides:
        return None

    ask, bid = tape.ask[a:end], tape.bid[a:end]
    i_buy = _first_true(ask >= hi) if 1 in sides else -1
    i_sell = _first_true(bid <= lo) if -1 in sides else -1
    if i_buy < 0 and i_sell < 0:
        return None
    if i_sell < 0 or (0 <= i_buy < i_sell):
        direction, i_fill = 1, i_buy
    else:
        direction, i_fill = -1, i_sell

    e = a + i_fill
    price = (tape.ask[e] + slip) if direction == 1 else (tape.bid[e] - slip)
    return _Entry(i=e, direction=direction, price=price,
                  mid=(tape.bid[e] + tape.ask[e]) / 2, ts_us=int(tape.ts[e]))


def _simulate(tape: _Tape, row: dict, ent: _Entry, stop_frac: float,
              target: float | None, cfg: ORBConfig, slip: float) -> dict:
    """One (stop, target) geometry on one already-resolved entry."""
    direction, entry = ent.direction, ent.price
    risk = stop_frac * row["atr"]
    stop = entry - direction * risk
    tgt = None if target is None else entry + direction * target * risk

    flat_us = int(row["close_ts"].timestamp() * US)
    end = max(tape.index_at(flat_us), ent.i + 1)
    pbid, pask = tape.bid[ent.i:end], tape.ask[ent.i:end]

    if direction == 1:
        i_stop = _first_true(pbid <= stop)          # a long stops out on the bid
        i_tgt = -1 if tgt is None else _first_true(pask >= tgt)
    else:
        i_stop = _first_true(pask >= stop)
        i_tgt = -1 if tgt is None else _first_true(pbid <= tgt)

    # A tie inside one tick is resolved against the trade. The tick carries a
    # bid and an ask, not a path, so the alternative would be a flattering
    # assumption made silently.
    if i_stop >= 0 and (i_tgt < 0 or i_stop <= i_tgt):
        exit_i, reason = i_stop, "stop"
        exit_price = (pbid[i_stop] - slip) if direction == 1 else (pask[i_stop] + slip)
    elif i_tgt >= 0:
        exit_i, reason = i_tgt, "target"
        exit_price = tgt                                   # a limit, no slippage
    else:
        exit_i, reason = len(pbid) - 1, "bell"
        exit_price = (pbid[exit_i] - slip) if direction == 1 else (pask[exit_i] + slip)

    mid_exit = (pbid[exit_i] + pask[exit_i]) / 2
    pnl = direction * (exit_price - entry)
    return {
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "hold_min": (int(tape.ts[ent.i + exit_i]) - ent.ts_us) / (60 * US),
        "risk": float(risk),
        "stop_frac": float(stop_frac),
        "target_r": float("nan") if target is None else float(target),
        "target": tag(target),
        "pnl": float(pnl),
        "r_multiple": float(pnl / risk) if risk > 0 else float("nan"),
        "mid_pnl": float(direction * (mid_exit - ent.mid)),
        "mid_r": float(direction * (mid_exit - ent.mid) / risk) if risk > 0
        else float("nan"),
    }


def _months(ctx: pl.DataFrame) -> list[tuple[int, int]]:
    return (
        ctx.select(y=pl.col("nyd").dt.year(), m=pl.col("nyd").dt.month())
        .unique().sort("y", "m").rows()
    )


def run(
    symbol: str,
    cfg: ORBConfig = ORBConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    stop_fractions: Sequence[float] | None = None,
    targets: Sequence[float | None] | None = None,
    costs: CostModel | None = None,
    ctx: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Backtest the opening range breakout, month by month over the ticks.

    Returns one row per (session, stop fraction, target). The entry is resolved
    **once** per session and shared by every geometry, so a sweep cannot
    accidentally let a different fill leak into a different cell.
    """
    spec = get_spec(symbol)
    stop_fractions = (cfg.stop_atr_frac,) if stop_fractions is None else tuple(stop_fractions)
    targets = (None,) if targets is None else tuple(targets)
    if ctx is None:
        ctx = daily_context(symbol, cfg, split=split, start=start, end=end,
                            allow_test=allow_test)
    ctx = eligible(ctx, cfg)
    if ctx.is_empty():
        return pl.DataFrame()

    costs = costs or CostModel.from_profiles(symbol, split=split or "dev")
    slip = costs.slippage_pips(hour=None) * spec.pip

    records: list[dict] = []
    for year, month in _months(ctx):
        day_rows = ctx.filter(
            (pl.col("nyd").dt.year() == year) & (pl.col("nyd").dt.month() == month)
        )
        if day_rows.is_empty():
            continue
        # The New York session of the last day of a month can spill into the
        # next UTC day, so the tick window is padded a day at each end.
        lo = day_rows["nyd"].min() - timedelta(days=1)
        hi = day_rows["nyd"].max() + timedelta(days=2)
        ticks = load_ticks(symbol, start=lo, end=hi, allow_test=True)
        if ticks.is_empty():
            continue
        tape = _Tape(
            ts=ticks["ts"].to_numpy().astype("datetime64[us]").astype(np.int64),
            bid=ticks["bid"].to_numpy(),
            ask=ticks["ask"].to_numpy(),
        )
        for row in day_rows.iter_rows(named=True):
            ent = _find_entry(tape, row, cfg, slip)
            if ent is None:
                continue
            base = {
                "symbol": symbol,
                "nyd": row["nyd"],
                "or_minutes": cfg.or_minutes,
                "direction": ent.direction,
                "signal": int(row["signal"]),
                "relvol": float(row["relvol"]),
                "atr": float(row["atr"]),
                "or_width": float(row["or_width"]),
                "entry": ent.price,
                "entry_mid": ent.mid,
                "entry_min": (ent.ts_us - int(row["or_end_ts"].timestamp() * US))
                / (60 * US),
            }
            commission = costs.commission_pips(price=ent.mid) * spec.pip
            for stop_frac in stop_fractions:
                for target in targets:
                    rec = base | _simulate(tape, row, ent, stop_frac, target, cfg, slip)
                    rec["commission"] = commission
                    rec["net_pnl"] = rec["pnl"] - commission
                    rec["net_r"] = (
                        rec["net_pnl"] / rec["risk"] if rec["risk"] > 0 else float("nan")
                    )
                    rec["cost_r"] = (
                        (rec["mid_pnl"] - rec["net_pnl"]) / rec["risk"]
                        if rec["risk"] > 0 else float("nan")
                    )
                    rec["net_bps"] = 1e4 * rec["net_pnl"] / ent.mid
                    rec["mid_bps"] = 1e4 * rec["mid_pnl"] / ent.mid
                    records.append(rec)
    return pl.DataFrame(records) if records else pl.DataFrame()


# --------------------------------------------------------------------------
# The paper's Figure 4
# --------------------------------------------------------------------------

#: The paper's own buckets, in relative-volume units.
RELVOL_EDGES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


def relvol_buckets(trades: pl.DataFrame, *, column: str = "net_r",
                   edges: Sequence[float] = RELVOL_EDGES) -> pl.DataFrame:
    """Average PnL per trade, sorted into relative-volume buckets.

    This is the paper's Figure 4 - the plot that carries their whole argument,
    since it is the reason for the Stocks-in-Play filter. They report -0.02R
    below 100% relative volume, +0.08R above it, and +0.38R above 3,000%.
    """
    if trades.is_empty():
        return pl.DataFrame()
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges, edges[1:])]
    labels.append(f"{edges[-1]:g}+")
    bucket = pl.col("relvol").cut(list(edges[1:]), labels=labels)
    return (
        trades.with_columns(bucket=bucket)
        .group_by("bucket")
        .agg(
            n=pl.len(),
            mean=pl.col(column).mean(),
            sd=pl.col(column).std(),
            hit_pct=100.0 * (pl.col(column) > 0).mean(),
            relvol_mean=pl.col("relvol").mean(),
        )
        .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt())
        .sort("relvol_mean")
    )


def placebo_signal(ctx: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    """The same sessions with the opening-candle sign shuffled.

    Keeps the calendar, the opening range geometry, the ATR and the relative
    volume, and destroys only the one thing the paper says is informative. Any
    result the placebo also produces is produced by the geometry, not by the
    signal.
    """
    rng = np.random.default_rng(seed)
    signal = ctx["signal"].to_numpy().copy()
    rng.shuffle(signal)
    return ctx.with_columns(signal=pl.Series("signal", signal, dtype=pl.Int8))
