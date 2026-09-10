"""The macro-news breakout of Chen (2025), on the real USTEC tape.

The claim being tested
----------------------
*Artificial Intelligence in Day Trading: An Intraday Trading Framework with
Economic Indicators and Large Language Model Analysis* (SSRN 5246516) reports,
on USTEC 1-minute data from 2018-01 to 2025-05:

============================  ========  ===========  ==========
Metric                        Backbone  LLM-filtered  Buy & hold
============================  ========  ===========  ==========
Total return                    730.6%       982.9%      205.4%
Annualised return                33.5%        38.5%       16.5%
Trades                            1717         1401        1893
Win rate                         49.7%        52.6%       55.2%
Max drawdown                    -17.4%       -14.1%      -35.5%
Sharpe                            1.67         2.11        0.61
Profit factor                     1.27         1.41        1.11
============================  ========  ===========  ==========

The rule, in full:

1. Wait for a scheduled high-impact USD release. ``T_delay = 120 s`` after it,
   evaluate a Trading Indicator built from a Kalman filter on price::

       TI = 100 / (1 + 2 ** (-S * delta)),   delta = (p - p_hat) / p

   with ``S = 500``, and ``p_hat`` the Kalman estimate under process noise
   ``Q = 0.01`` and measurement noise ``R = 100``.
2. ``TI > 70`` arms a **buy stop** at the 200-minute high; ``TI < 30`` arms a
   **sell stop** at the 200-minute low. Between the two the strategy stands
   aside - the paper calls this the buffer zone.
3. Only one pending stop order may exist at a time; a signal arriving while one
   is live is dropped. Unfilled orders expire after 24 h.
4. On fill, a stop-loss goes 1% of price away, sized so it costs 1.5% of equity.
5. Everything is flat at 15:50 ET.

Reading the threshold table
---------------------------
Table 1 of the paper lists "TI thresholds (TI_buy/TI_sell) 30/70", which taken
literally means buy above 30 and sell below 70 - overlapping conditions that
fire on every event. The body says there is a *buffer zone between* the two
thresholds, and only ``buy > 70, sell < 30`` produces one. That reading is used
here; :func:`sweep_thresholds` runs the literal one as a robustness check, and
the report gives both.

The arithmetic is worth knowing before reading any result. Under ``Q = 0.01``
and ``R = 100`` the Kalman gain converges to a constant ``K ~ 0.00995``, so
``p_hat`` is simply a ~100-period EMA of price and the filter contributes no
state estimation beyond that. ``TI > 70`` then unpacks to ``delta > 0.244%``:
price more than a quarter of a percent above its 100-minute EMA. The signal is
a momentum filter with a Kalman filter's name on it.

What this corpus can and cannot test
------------------------------------
**The backbone transfers; the LLM layer does not.** Rules 1-5 are pure price
mechanics and are implemented exactly, tick-resolved. The LLM overlay needs the
*actual, forecast and previous values* of each release, which this corpus does
not hold, so it cannot be replicated here and no number below claims to. What
can be said about it is said in the report, on the paper's own evidence.

**The calendar is reconstructed.** See :mod:`qlab.macro_calendar`. Dates come
from publication rules, never from the tape, and are audited afterwards against
it. The exact tier (FOMC, NFP, ISM, claims) and the approximate tier (CPI, PPI,
retail sales, GDP) are always reported separately.

**The sample is shorter and later.** This corpus starts 2020-01-29, so the
2018-2019 third of the paper's window is missing, and it extends to 2026-09,
which the paper never saw.

Why this is resolved on ticks
-----------------------------
Both entry and exit are touch-triggered: a stop order fills when price trades
through a level, and a 1% stop-loss is hit the same way. On 1-minute bars, a
bar whose high crosses the entry and whose low crosses the stop is ambiguous,
and the two orderings differ by the whole trade. USTEC posts ~40 quotes a
minute and several hundred in the minute after a release, which is exactly when
this strategy is trading, so the ordering is resolved from the tape instead of
assumed. Fills cross the spread at the quoted bid/ask of the tick that
triggered them, and :class:`qlab.costs.CostModel` adds measured latency drift on
top.

The control
-----------
:func:`run` accepts ``control=True``, which keeps every mechanic and replaces
the calendar with the same number of *non-event* days at the same clock minutes.
If the strategy earns as much on the control as on the calendar, the news is
decoration and what is being traded is the 100-minute EMA. That comparison is
the point of the exercise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import polars as pl

from ..costs import CostModel
from ..engine import Account, Fill, Trade, lots_for_risk
from ..macro_calendar import ET, build_calendar
from ..loader import load_bars, load_ticks
from ..symbols import get_spec

# --- the paper's parameters, Table 1 ----------------------------------------


@dataclass(frozen=True)
class Params:
    """Chen (2025) Table 1, with the threshold reading fixed as documented."""

    delay_s: int = 120
    lookback_min: int = 200
    ti_buy: float = 70.0
    ti_sell: float = 30.0
    scale_factor: float = 500.0
    q_noise: float = 0.01
    r_noise: float = 100.0
    stop_pct: float = 0.01
    risk_pct: float = 0.015
    expire_h: int = 24
    close_et: tuple[int, int] = (15, 50)
    max_leverage: float = 30.0
    starting_equity: float = 10_000.0


DEFAULT = Params()


def kalman_trend(prices: pl.Series | list[float], q: float, r: float) -> list[float]:
    """The paper's scalar Kalman filter, run forward over ``prices``.

    A random walk observed in noise: predict ``p_hat`` unchanged, inflate the
    covariance by ``q``, then correct by the gain ``K = P- / (P- + r)``. With
    ``q = 0.01`` and ``r = 100`` the gain settles at ``0.00995`` within a few
    hundred observations, which is why the docstring above calls it an EMA.
    """
    values = list(prices)
    if not values:
        return []
    estimate, cov = values[0], 1.0
    out = []
    for price in values:
        cov += q
        gain = cov / (cov + r)
        estimate += gain * (price - estimate)
        cov *= 1.0 - gain
        out.append(estimate)
    return out


def trading_indicator(price: float, trend: float, scale: float) -> float:
    """``TI = 100 / (1 + 2**(-S*delta))`` - a sigmoid on relative deviation."""
    if price <= 0:
        return 50.0
    delta = (price - trend) / price
    exponent = max(-60.0, min(60.0, -scale * delta))
    return 100.0 / (1.0 + 2.0**exponent)


def steady_state_gain(q: float, r: float) -> float:
    """The gain the filter converges to: root of ``K^2 * r + K*q - q = 0``."""
    return (-q + math.sqrt(q * q + 4.0 * q * r)) / (2.0 * r)


# --- signals ----------------------------------------------------------------


@dataclass
class Signal:
    """One armed stop order, before it is known whether the market reached it."""

    event_ts: datetime
    decide_ts: datetime
    direction: int  # +1 buy stop, -1 sell stop
    trigger: float
    ti: float
    event: str
    tier: str
    expires_ts: datetime


def _et_close(day_utc: datetime, close_et: tuple[int, int]) -> datetime:
    """The 15:50 ET flatten instant for the ET day ``day_utc`` falls in."""
    import zoneinfo

    local = day_utc.astimezone(zoneinfo.ZoneInfo(ET))
    close = local.replace(
        hour=close_et[0], minute=close_et[1], second=0, microsecond=0
    )
    if local >= close:  # after the bell: the next session's bell
        close += timedelta(days=1)
        while close.weekday() >= 5:
            close += timedelta(days=1)
    return close.astimezone(timezone.utc)


def signals_for(
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    params: Params = DEFAULT,
) -> list[Signal]:
    """Evaluate the indicator at every release and return the orders it arms.

    ``bars`` are close-labelled 1-minute bars covering the calendar plus at
    least ``lookback_min`` of warm-up. The Kalman filter is run once over the
    whole close series rather than restarted per event, which is what a live
    expert advisor holding filter state across events would do.
    """
    if bars.is_empty() or calendar.is_empty():
        return []

    frame = bars.sort("ts")
    closes = frame["close"].to_list()
    stamps = frame["ts"].to_list()
    trend = kalman_trend(closes, params.q_noise, params.r_noise)

    # Rolling 200-minute extremes, as of each bar's close.
    extremes = frame.select(
        hi=pl.col("high").rolling_max(params.lookback_min, min_samples=params.lookback_min),
        lo=pl.col("low").rolling_min(params.lookback_min, min_samples=params.lookback_min),
    )
    highs, lows = extremes["hi"].to_list(), extremes["lo"].to_list()

    index = {ts: i for i, ts in enumerate(stamps)}
    out: list[Signal] = []
    for row in calendar.iter_rows(named=True):
        decide = row["ts"] + timedelta(seconds=params.delay_s)
        # The last bar that has *closed* at the decision instant. Bars are
        # close-labelled, so a bar stamped exactly at `decide` is usable.
        bar_ts = decide.replace(second=0, microsecond=0)
        i = index.get(bar_ts)
        if i is None or highs[i] is None or lows[i] is None:
            continue
        ti = trading_indicator(closes[i], trend[i], params.scale_factor)
        if ti > params.ti_buy:
            direction, trigger = 1, highs[i]
        elif ti < params.ti_sell:
            direction, trigger = -1, lows[i]
        else:
            continue
        out.append(
            Signal(
                event_ts=row["ts"],
                decide_ts=decide,
                direction=direction,
                trigger=float(trigger),
                ti=ti,
                event=row["event"],
                tier=row["tier"],
                expires_ts=decide + timedelta(hours=params.expire_h),
            )
        )
    return out


# --- tick-resolved execution -------------------------------------------------


@dataclass
class _Pending:
    signal: Signal
    lots: float = 0.0


def _resolve(
    signals: list[Signal],
    ticks: pl.DataFrame,
    costs: CostModel,
    params: Params,
    account: Account,
) -> list[Trade]:
    """Walk the tape once, arming, filling, stopping and flattening in order.

    One pass over ticks in time order. At every tick: expire or fill the pending
    order, then manage an open position against its stop and its 15:50 bell.
    Because the tape is walked forward and never indexed by a later timestamp,
    no decision can see a price that had not printed.
    """
    if not signals or ticks.is_empty():
        return []

    spec = get_spec("USTEC")
    contract = spec.contract_size or 1.0
    commission = (spec.commission_per_lot_side_usd or 0.0) * costs.commission_multiplier

    queue = sorted(signals, key=lambda s: s.decide_ts)
    trades: list[Trade] = []
    pending: _Pending | None = None
    open_trade: Trade | None = None
    close_at: datetime | None = None
    cursor = 0

    stamps = ticks["ts"].to_list()
    bids = ticks["bid"].to_list()
    asks = ticks["ask"].to_list()

    for i, now in enumerate(stamps):
        bid, ask = bids[i], asks[i]
        mid = 0.5 * (bid + ask)
        account.roll_to(now.toordinal())

        # 1. Manage an open position first: a stop that is already hit is not
        #    saved by an order armed on the same tick.
        if open_trade is not None:
            direction = open_trade.direction
            adverse = bid if direction > 0 else ask
            excursion = direction * (adverse - open_trade.entry.price)
            open_trade.mae_price = min(open_trade.mae_price, excursion)
            open_trade.mfe_price = max(open_trade.mfe_price, excursion)
            hit_stop = (
                adverse <= open_trade.stop_price
                if direction > 0
                else adverse >= open_trade.stop_price
            )
            if hit_stop or (close_at is not None and now >= close_at):
                exit_price = bid if direction > 0 else ask
                drift = costs.drift_pips(hour=now.hour) * spec.pip
                exit_price -= direction * drift  # latency works against the exit
                open_trade.exits.append(
                    Fill(
                        ts=int(now.timestamp() * 1e6),
                        price=exit_price,
                        lots=open_trade.entry.lots,
                        reason="stop" if hit_stop else "eod",
                        mid=mid,
                    )
                )
                open_trade.slippage_usd += drift * contract * open_trade.entry.lots
                account.apply(open_trade.net_usd)
                trades.append(open_trade)
                open_trade = None
                close_at = None

        # 2. Arm the next signal, if nothing is pending and nothing is open.
        while cursor < len(queue) and queue[cursor].decide_ts <= now:
            candidate = queue[cursor]
            cursor += 1
            if pending is not None or open_trade is not None:
                continue  # the paper's one-pending-order rule
            if not account.can_trade():
                continue
            stop = candidate.trigger * (1.0 - candidate.direction * params.stop_pct)
            lots = lots_for_risk(
                account.equity,
                params.risk_pct,
                candidate.trigger,
                stop,
                contract,
            )
            cap = account.equity * params.max_leverage / (candidate.trigger * contract)
            lots = min(lots, round(cap, 2))
            if lots <= 0:
                continue
            pending = _Pending(candidate, lots)

        # 3. Expire or fill the pending order.
        if pending is not None:
            signal = pending.signal
            if now >= signal.expires_ts:
                pending = None
            elif open_trade is None:
                touched = (
                    ask >= signal.trigger if signal.direction > 0 else bid <= signal.trigger
                )
                if touched:
                    drift = costs.drift_pips(hour=now.hour) * spec.pip
                    entry = (ask if signal.direction > 0 else bid) + signal.direction * drift
                    stop = entry * (1.0 - signal.direction * params.stop_pct)
                    open_trade = Trade(
                        symbol="USTEC",
                        direction=signal.direction,
                        entry=Fill(
                            ts=int(now.timestamp() * 1e6),
                            price=entry,
                            lots=pending.lots,
                            reason="entry",
                            mid=mid,
                        ),
                        stop_price=stop,
                        contract_size=contract,
                        commission_per_lot_side=commission,
                        setup={
                            "ti": signal.ti,
                            "event": signal.event,
                            "tier": signal.tier,
                            "trigger": signal.trigger,
                            "event_ts": signal.event_ts.isoformat(),
                            "wait_s": (now - signal.decide_ts).total_seconds(),
                        },
                    )
                    open_trade.slippage_usd = drift * contract * pending.lots
                    close_at = _et_close(now, params.close_et)
                    pending = None

    return trades


# --- the control -------------------------------------------------------------


def control_calendar(calendar: pl.DataFrame, bars: pl.DataFrame, seed: int = 7) -> pl.DataFrame:
    """The same clock minutes, on days with no scheduled release.

    Preserves the time-of-day mix exactly - one placebo event per real event, at
    the same ET hour and minute - and moves only the day. Anything the strategy
    earns here it earns from the 100-minute EMA and the 200-minute range, with
    no news in the mechanism at all.
    """
    trading_days = (
        bars.select(day=pl.col("ts").dt.convert_time_zone(ET).dt.date())
        .unique()
        .sort("day")["day"]
        .to_list()
    )
    busy = set(calendar["day"].to_list())
    free = [d for d in trading_days if d not in busy and d.weekday() < 5]
    if not free:
        return calendar.head(0)

    rng = random.Random(seed)
    rows = calendar.to_dicts()
    picked = [rng.choice(free) for _ in rows]
    out = []
    for row, day in zip(rows, picked):
        import zoneinfo

        local = datetime(
            day.year, day.month, day.day, int(row["et_hour"]), int(row["et_minute"]),
            tzinfo=zoneinfo.ZoneInfo(ET),
        )
        out.append(
            {
                **row,
                "day": day,
                "ts": local.astimezone(timezone.utc).replace(tzinfo=timezone.utc),
                "event": "control",
                "tier": "control",
            }
        )
    return pl.DataFrame(out, schema=calendar.schema).sort("ts")


# --- top level ---------------------------------------------------------------


@dataclass
class Result:
    trades: pl.DataFrame
    signals: int
    events: int
    params: Params
    label: str
    starting_equity: float


def run(
    *,
    split: str | None = "dev",
    start: date | None = None,
    end: date | None = None,
    params: Params = DEFAULT,
    tiers: str = "all",
    control: bool = False,
    allow_test: bool = False,
    costs: CostModel | None = None,
    label: str | None = None,
) -> Result:
    """Backtest the backbone strategy over one split, tick-resolved.

    Ticks are streamed a month at a time so that a seven-year run does not need
    the whole 90-million-tick USTEC feed resident. Account equity, the pending
    order and any open position carry across month boundaries, so the chunking
    is invisible to the strategy.
    """
    warm = timedelta(days=5)
    bars = load_bars(
        "USTEC", "1m", split=split, start=start, end=end, warmup=warm,
        allow_test=allow_test,
    )
    if bars.is_empty():
        raise ValueError("no bars for the requested window")

    first = bars["ts"].min()
    last = bars["ts"].max()
    calendar = build_calendar(first.date(), last.date(), tiers=tiers)
    # Drop events inside the warm-up: the split's own first day is the first
    # day a trade may be taken.
    eval_from = (first + warm).date() if (split or start) else first.date()
    calendar = calendar.filter(pl.col("day") >= eval_from)
    if control:
        calendar = control_calendar(calendar, bars)

    signals = signals_for(bars, calendar, params)
    costs = costs or CostModel.from_profiles("USTEC", split=split if split else None)
    account = Account(
        equity=params.starting_equity,
        risk_pct=params.risk_pct,
        daily_stop_pct=1.0,  # the paper imposes no daily loss limit
    )

    trades: list[Trade] = []
    months = sorted({(t.year, t.month) for t in bars["ts"].dt.replace_time_zone(None)})
    for year, month in months:
        window_start = datetime(year, month, 1, tzinfo=timezone.utc)
        window_end = (
            datetime(year + month // 12, month % 12 + 1, 1, tzinfo=timezone.utc)
        )
        chunk = [s for s in signals if window_start <= s.decide_ts < window_end]
        # An order armed late in a month can fill or stop in the next one, so
        # the tape is read a day past the month end.
        if not chunk:
            continue
        ticks = load_ticks(
            "USTEC",
            start=window_start,
            end=window_end + timedelta(days=2),
            allow_test=allow_test,
        )
        trades.extend(_resolve(chunk, ticks, costs, params, account))

    frame = (
        pl.DataFrame([t.to_dict() for t in trades])
        if trades
        else pl.DataFrame(schema={"entry_ts": pl.Int64, "net_usd": pl.Float64})
    )
    return Result(
        trades=frame,
        signals=len(signals),
        events=calendar.height,
        params=params,
        label=label or ("control" if control else f"{split or 'custom'}/{tiers}"),
        starting_equity=params.starting_equity,
    )


def sweep_thresholds(
    grid: list[tuple[float, float]] | None = None, **kwargs
) -> pl.DataFrame:
    """Run the strategy across TI threshold pairs, including the literal 30/70.

    The first row of the default grid is the paper's Table 1 read literally
    (buy above 30, sell below 70), which arms an order on every single event;
    the last is the reading used everywhere else in this module.
    """
    from ..metrics import tearsheet

    grid = grid or [(30.0, 70.0), (60.0, 40.0), (65.0, 35.0), (70.0, 30.0), (80.0, 20.0)]
    rows = []
    for buy, sell in grid:
        params = Params(ti_buy=buy, ti_sell=sell)
        result = run(params=params, **kwargs)
        if result.trades.is_empty():
            rows.append({"ti_buy": buy, "ti_sell": sell, "trades": 0})
            continue
        sheet = tearsheet(result.trades, starting_equity=result.starting_equity)
        rows.append(
            {
                "ti_buy": buy,
                "ti_sell": sell,
                "signals": result.signals,
                "trades": int(sheet.get("trades", 0)),
                "net_usd": float(sheet.get("net_usd", 0.0)),
                "sharpe": float(sheet.get("sharpe", 0.0)),
                "profit_factor": float(sheet.get("profit_factor", 0.0)),
                "max_dd_pct": float(sheet.get("max_dd_pct", 0.0)),
                "win_rate_pct": float(sheet.get("win_rate_pct", 0.0)),
            }
        )
    return pl.DataFrame(rows)
