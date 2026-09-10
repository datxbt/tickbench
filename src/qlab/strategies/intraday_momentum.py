"""The Noise Area intraday momentum strategy, as specified by Zarattini, Aziz
and Barbon (2024/2025).

The claim being tested
----------------------
*Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)*
(SSRN 4824172, this version 22 September 2025) reports, on SPY 1-minute bars
from May 2007 to April 2024 and net of costs, a total return of 1,985%, a 19.6%
annualised return, 14.3% volatility, a Sharpe of 1.33, a 25% maximum drawdown
and a 43% daily hit ratio. The rules are stated completely enough to transcribe,
which is what this module does.

**The Noise Area.** For each time-of-day ``HH:MM`` in the session, the average
absolute move from that day's open, over the previous 14 sessions:

.. code::

    move[t-i, tau]  = | close[t-i, tau] / open[t-i, 09:30] - 1 |,   i = 1..14
    sigma[t, tau]   = mean(move[t-1..t-14, tau])
    upper[t, tau]   = max(open[t, 09:30], close[t-1, 16:00]) * (1 + sigma[t, tau])
    lower[t, tau]   = min(open[t, 09:30], close[t-1, 16:00]) * (1 - sigma[t, tau])

The ``max``/``min`` against the previous close is the paper's gap adjustment:
after a gap down, the upper boundary is pushed up by the size of the gap, so a
gap does not by itself count as an imbalance. Note the boundary is *time-of-day
dependent* - the move needed to signal an imbalance by 10:30 is roughly half the
move needed by 15:30 - which is what separates this from a fixed volatility
breakout of the Crabel/Kaufman kind.

**Decisions happen only on the half hour.** Entries, reversals and stops are
evaluated at ``HH:00`` and ``HH:30`` and nowhere else. The paper is explicit
about this for stops as well as entries, and it is the reason this module runs
on bars rather than on ticks: there is no intrabar ordering to assume, because
the rules never look inside a bar.

**Entry.** Flat, at a decision time: close above the upper boundary goes long,
close below the lower boundary goes short.

**Exit.** A trailing stop at ``max(upper, VWAP)`` for a long and
``min(lower, VWAP)`` for a short, VWAP anchored at the session open. Everything
is flat at the session close. The paper builds this in three steps and reports
all three, so :class:`NoiseAreaConfig` keeps them as ``stop="opposite"`` (the
base model, trailing at the far boundary), ``stop="band"`` and
``stop="band_vwap"`` (its final choice) rather than hard-coding the winner.

**Sizing.** ``shares = AUM * min(4, sigma_target / sigma_daily) / open``, where
``sigma_daily`` is the sample standard deviation of the previous 14 daily
returns and ``sigma_target`` is 2% *daily*. Leverage is capped at 4x. This is
the difference between the paper's 380% and its 1,985%: the underlying rule set
is the same, the second one is levered about twice on average.

What this corpus can and cannot test
------------------------------------
**The instrument is not SPY.** This corpus holds no US equity ETF. The primary
test is USTEC, the broker's Nasdaq-100 CFD, which is the closest available
instrument: a US equity index, trading the same 09:30-16:00 New York session,
with the same dealer-gamma and index-arbitrage plumbing underneath it. It is not
the same asset - the Nasdaq-100 is more volatile and more concentrated than the
S&P 500, so a strategy that survives here has cleared a *different* bar, not a
lower one. XAU/USD and EUR/USD are run as well, because a time-series momentum
claim that only works on one instrument is a claim about that instrument.

**The CFD has no opening auction.** SPY's 09:30 open is a single crossing
auction after a 17.5-hour halt. USTEC is a CFD on a future that has been trading
all night, so its 09:30 print is just another quote. The gap adjustment
therefore measures the actual overnight drift rather than an auction imbalance,
and ``sigma`` at early times-of-day is smaller here than the paper's. This makes
early entries easier to trigger, not harder, so :func:`entry_time_surface` sweeps
the first decision time instead of assuming the paper's.

**Volume is quote updates.** VWAP is weighted by ``n_ticks`` - how many quote
revisions the broker published in the bar - because the feed carries no size.
On a spot CFD this is exactly what MT5 would show as volume, so a practitioner
following the paper on this instrument would compute the same VWAP. It is still
a proxy and is labelled as one.

Costs
-----
Not the paper's $0.0035/share commission and $0.001/share slippage, which are
US equity terms and meaningless on a CFD. Spread is measured, per hour, from the
tick corpus; commission is the published Exness contract term; slippage is the
measured latency drift. All three arrive through :class:`qlab.costs.CostModel`,
so this strategy is costed on the same basis as everything else in the project.
Cost is charged per round turn in basis points of the notional traded, and the
notional is the levered one - a 4x day pays 4x the cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, time
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from ..costs import CostModel

NY = "America/New_York"

STOP_RULES = ("opposite", "band", "band_vwap")


@dataclass(frozen=True)
class NoiseAreaConfig:
    """Every knob the paper states, at the value the paper states it."""

    lookback_days: int = 14
    """Sessions averaged into sigma. The paper's value is 14."""

    session_open: time = time(9, 30)
    session_close: time = time(16, 0)

    decision_minutes: tuple[int, ...] = (0, 30)
    """Minutes past the hour at which the rules are allowed to act."""

    first_decision: time = time(10, 0)
    """Earliest decision time. Sigma at 09:30 is zero by construction and tiny
    just after, so the first few marks would fire on noise; the paper's own
    worked examples enter at 10:30. Swept by :func:`entry_time_surface`."""

    last_decision: time = time(15, 30)
    """Latest mark at which a *new* position may be opened. Exits and the
    session flat are unaffected."""

    stop: str = "band_vwap"
    """``opposite`` | ``band`` | ``band_vwap`` - the paper's three variants."""

    allow_reentry: bool = True
    """Whether a position may be re-opened after a stop on the same session."""

    vol_target: float | None = 0.02
    """Daily volatility target. ``None`` gives the paper's unlevered 100%
    notional variant."""

    max_leverage: float = 4.0

    vol_lookback: int = 14

    def __post_init__(self) -> None:
        if self.stop not in STOP_RULES:
            raise ValueError(f"stop must be one of {STOP_RULES}, got {self.stop!r}")
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be positive or None")
        if not self.decision_minutes:
            raise ValueError("decision_minutes cannot be empty")


def with_config(cfg: NoiseAreaConfig, **changes) -> NoiseAreaConfig:
    return replace(cfg, **changes)


# ---------------------------------------------------------------------------
# Session frame
# ---------------------------------------------------------------------------


def _minute_of_session(cfg: NoiseAreaConfig) -> tuple[int, int]:
    o = cfg.session_open.hour * 60 + cfg.session_open.minute
    c = cfg.session_close.hour * 60 + cfg.session_close.minute
    return o, c


def sessions(bars: pl.DataFrame, cfg: NoiseAreaConfig = NoiseAreaConfig()) -> pl.DataFrame:
    """Reduce 1-minute bars to the regular session, stamped in New York time.

    ``tod`` is minutes since midnight New York, which makes the time-of-day
    alignment across days a plain integer join and keeps daylight saving out of
    the arithmetic entirely.
    """
    open_m, close_m = _minute_of_session(cfg)
    frame = (
        bars.with_columns(pl.col("ts").dt.convert_time_zone(NY).alias("ny"))
        .with_columns(
            pl.col("ny").dt.date().alias("session"),
            (
                pl.col("ny").dt.hour().cast(pl.Int32) * 60
                + pl.col("ny").dt.minute().cast(pl.Int32)
            ).alias("tod"),
        )
        .filter(pl.col("tod").is_between(open_m, close_m, closed="both"))
        .sort("ts")
    )
    # A session missing its open or its close cannot be measured against the
    # paper's rules, and half-days are exactly where a naive fill would invent
    # a 16:00 print that never happened.
    complete = (
        frame.group_by("session")
        .agg(
            pl.col("tod").min().alias("first_tod"),
            pl.col("tod").max().alias("last_tod"),
            pl.len().alias("n_bars"),
        )
        .filter(
            (pl.col("first_tod") <= open_m + 1)
            & (pl.col("last_tod") >= close_m - 1)
            & (pl.col("n_bars") >= 0.8 * (close_m - open_m))
        )
        .select("session")
    )
    return frame.join(complete, on="session", how="semi")


def _vwap(frame: pl.DataFrame) -> pl.DataFrame:
    """Session-anchored VWAP on typical price, weighted by quote updates.

    Falls back to an unweighted running mean in the rare bar with no ticks
    recorded, which is what a zero-weight VWAP would otherwise turn into a null
    and then into a silently skipped stop check.
    """
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    weight = pl.col("n_ticks").cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
    return frame.with_columns(
        pl.when(weight.cum_sum().over("session") > 0)
        .then(
            (typical * weight).cum_sum().over("session")
            / weight.cum_sum().over("session")
        )
        .otherwise(
            typical.cum_sum().over("session")
            / pl.int_range(1, pl.len() + 1).over("session")
        )
        .alias("vwap")
    )


def noise_area(
    bars: pl.DataFrame, cfg: NoiseAreaConfig = NoiseAreaConfig()
) -> pl.DataFrame:
    """Attach ``sigma``, the two boundaries and VWAP to every session minute.

    The boundaries on day ``t`` use only sessions ``t-1 .. t-lookback``: the
    shift is applied per time-of-day *after* grouping, so no part of day ``t``
    can leak into its own band. The first ``lookback_days`` sessions have a null
    sigma and are dropped, which is why callers must pass a warmup.
    """
    frame = sessions(bars, cfg)
    if frame.is_empty():
        return frame

    opens = (
        frame.group_by("session")
        .agg(pl.col("open").first().alias("day_open"),
             pl.col("close").last().alias("day_close"))
        .sort("session")
    )
    opens = opens.with_columns(pl.col("day_close").shift(1).alias("prev_close"))
    frame = frame.join(opens, on="session", how="left")

    # Absolute move from the open, at each time-of-day.
    frame = frame.with_columns(
        (pl.col("close") / pl.col("day_open") - 1.0).abs().alias("move")
    ).sort("session", "tod")

    # Mean of the previous `lookback_days` sessions at the same time-of-day.
    # rolling_mean over a shifted column, partitioned by tod, is the whole of
    # the paper's step 2 - and the shift is what makes it point-in-time.
    frame = frame.with_columns(
        pl.col("move")
        .shift(1)
        .rolling_mean(cfg.lookback_days, min_samples=cfg.lookback_days)
        .over("tod", order_by="session")
        .alias("sigma")
    )

    anchor_hi = pl.max_horizontal("day_open", "prev_close")
    anchor_lo = pl.min_horizontal("day_open", "prev_close")
    frame = frame.with_columns(
        (anchor_hi * (1.0 + pl.col("sigma"))).alias("upper"),
        (anchor_lo * (1.0 - pl.col("sigma"))).alias("lower"),
    )
    return _vwap(frame.sort("ts")).drop_nulls("sigma")


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def daily_leverage(
    frame: pl.DataFrame, cfg: NoiseAreaConfig = NoiseAreaConfig()
) -> pl.DataFrame:
    """One row per session: the day's open, and the leverage the paper's
    volatility scaler asks for.

    ``sigma_daily`` is the sample standard deviation (``ddof=1``, as the paper
    writes it with a 1/13) of the previous 14 close-to-close session returns,
    shifted so day ``t`` is priced off days ``t-1..t-14``.
    """
    daily = (
        frame.group_by("session")
        .agg(pl.col("day_open").first(), pl.col("day_close").first())
        .sort("session")
    )
    daily = daily.with_columns(
        (pl.col("day_close") / pl.col("day_close").shift(1) - 1.0).alias("ret")
    )
    daily = daily.with_columns(
        pl.col("ret")
        .shift(1)
        .rolling_std(cfg.vol_lookback, min_samples=cfg.vol_lookback, ddof=1)
        .alias("sigma_daily")
    )
    if cfg.vol_target is None:
        lev = pl.lit(1.0)
    else:
        lev = (
            pl.when(pl.col("sigma_daily") > 0)
            .then(
                pl.min_horizontal(
                    pl.lit(cfg.max_leverage), cfg.vol_target / pl.col("sigma_daily")
                )
            )
            .otherwise(pl.lit(cfg.max_leverage))
        )
    return daily.with_columns(lev.alias("leverage"))


# ---------------------------------------------------------------------------
# The rule engine
# ---------------------------------------------------------------------------


@dataclass
class _Leg:
    side: int
    entry_tod: int
    entry_px: float
    exit_tod: int = -1
    exit_px: float = float("nan")
    reason: str = ""


def _simulate_session(
    tod: np.ndarray,
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    vwap: np.ndarray,
    decide: np.ndarray,
    cfg: NoiseAreaConfig,
    last_entry_tod: int,
) -> list[_Leg]:
    """Walk one session's decision marks and return the legs it produced.

    Ordering at a mark is the paper's: an open position is checked against its
    trailing stop first, and only a position that survives that check can hold.
    A stop and a fresh signal in the opposite direction therefore both fire at
    the same mark - which is the reversal the paper describes.
    """
    legs: list[_Leg] = []
    side = 0
    leg: _Leg | None = None

    idx = np.flatnonzero(decide)
    for i in idx:
        px = close[i]
        # --- manage an open position -------------------------------------
        if side != 0:
            assert leg is not None
            if cfg.stop == "opposite":
                stop = lower[i] if side > 0 else upper[i]
            elif cfg.stop == "band":
                stop = upper[i] if side > 0 else lower[i]
            else:
                stop = (
                    max(upper[i], vwap[i]) if side > 0 else min(lower[i], vwap[i])
                )
            hit = px < stop if side > 0 else px > stop
            if hit:
                leg.exit_tod, leg.exit_px, leg.reason = int(tod[i]), px, "stop"
                legs.append(leg)
                side, leg = 0, None

        # --- open a new one ----------------------------------------------
        if side == 0 and tod[i] <= last_entry_tod:
            if legs and not cfg.allow_reentry:
                continue
            if px > upper[i]:
                side = 1
            elif px < lower[i]:
                side = -1
            if side != 0:
                leg = _Leg(side=side, entry_tod=int(tod[i]), entry_px=px)

    if side != 0 and leg is not None:
        leg.exit_tod, leg.exit_px, leg.reason = int(tod[-1]), close[-1], "session_close"
        legs.append(leg)
    return legs


def run(
    bars: pl.DataFrame,
    cost: CostModel,
    cfg: NoiseAreaConfig = NoiseAreaConfig(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the strategy and return ``(daily, legs)``.

    ``daily`` is one row per session carrying ``gross_bps``, ``cost_bps`` and
    ``net_bps`` on equity - that is, already multiplied by the day's leverage,
    so the series is directly what :func:`qlab.metrics.tearsheet_from_returns`
    wants. ``legs`` is one row per entry-to-exit leg, in price terms.
    """
    frame = noise_area(bars, cfg)
    if frame.is_empty():
        return _empty_daily(), _empty_legs()

    lev = daily_leverage(frame, cfg).select("session", "leverage", "day_open")
    frame = frame.join(lev.drop("day_open"), on="session", how="left")

    open_m, close_m = _minute_of_session(cfg)
    first_m = cfg.first_decision.hour * 60 + cfg.first_decision.minute
    last_m = cfg.last_decision.hour * 60 + cfg.last_decision.minute

    daily_rows: list[dict] = []
    leg_rows: list[dict] = []

    for (sess,), part in frame.sort("ts").group_by(["session"], maintain_order=True):
        tod = part["tod"].to_numpy()
        close = part["close"].to_numpy()
        decide = (
            np.isin(tod % 60, np.asarray(cfg.decision_minutes))
            & (tod >= first_m)
            & (tod <= close_m)
        )
        if not decide.any():
            continue
        legs = _simulate_session(
            tod,
            close,
            part["upper"].to_numpy(),
            part["lower"].to_numpy(),
            part["vwap"].to_numpy(),
            decide,
            cfg,
            last_entry_tod=last_m,
        )
        leverage = float(part["leverage"][0]) if part["leverage"][0] is not None else 0.0
        hour_of = {int(t): int(t // 60) for t in tod}

        gross_bps = 0.0
        cost_bps = 0.0
        for leg in legs:
            r = leg.side * (leg.exit_px / leg.entry_px - 1.0)
            # Cost is charged on the notional actually traded, at the spread and
            # slippage measured for the hour the round turn straddles. Two
            # market orders, so one round turn per leg.
            rt = 0.5 * (
                cost.round_turn_bps(leg.entry_px, hour=hour_of[leg.entry_tod])
                + cost.round_turn_bps(leg.exit_px, hour=hour_of[leg.exit_tod])
            )
            gross_bps += 1e4 * r
            cost_bps += rt
            leg_rows.append(
                {
                    "session": sess,
                    "side": leg.side,
                    "entry_tod": leg.entry_tod,
                    "exit_tod": leg.exit_tod,
                    "entry_px": leg.entry_px,
                    "exit_px": leg.exit_px,
                    "reason": leg.reason,
                    "gross_bps": 1e4 * r,
                    "cost_bps": rt,
                    "leverage": leverage,
                }
            )
        daily_rows.append(
            {
                "session": sess,
                "n_legs": len(legs),
                "leverage": leverage,
                "gross_bps": leverage * gross_bps,
                "cost_bps": leverage * cost_bps,
                "net_bps": leverage * (gross_bps - cost_bps),
                "exposure_bps": leverage * sum(
                    abs(leg.exit_tod - leg.entry_tod) for leg in legs
                ),
            }
        )

    daily = pl.DataFrame(daily_rows, schema=_daily_schema()) if daily_rows else _empty_daily()
    legs_df = pl.DataFrame(leg_rows, schema=_legs_schema()) if leg_rows else _empty_legs()
    return daily.sort("session"), legs_df.sort("session", "entry_tod")


def _daily_schema() -> dict:
    return {
        "session": pl.Date,
        "n_legs": pl.Int64,
        "leverage": pl.Float64,
        "gross_bps": pl.Float64,
        "cost_bps": pl.Float64,
        "net_bps": pl.Float64,
        "exposure_bps": pl.Float64,
    }


def _legs_schema() -> dict:
    return {
        "session": pl.Date,
        "side": pl.Int64,
        "entry_tod": pl.Int64,
        "exit_tod": pl.Int64,
        "entry_px": pl.Float64,
        "exit_px": pl.Float64,
        "reason": pl.Utf8,
        "gross_bps": pl.Float64,
        "cost_bps": pl.Float64,
        "leverage": pl.Float64,
    }


def _empty_daily() -> pl.DataFrame:
    return pl.DataFrame(schema=_daily_schema())


def _empty_legs() -> pl.DataFrame:
    return pl.DataFrame(schema=_legs_schema())


# ---------------------------------------------------------------------------
# Benchmarks and diagnostics
# ---------------------------------------------------------------------------


def buy_and_hold(bars: pl.DataFrame, cfg: NoiseAreaConfig = NoiseAreaConfig()) -> pl.DataFrame:
    """Session close-to-close returns, in bps, on the same session calendar.

    The comparison the paper makes is against holding the index, so this is
    close-to-close and *includes* the overnight - the strategy's alternative is
    not an intraday-only passive position, it is owning the thing.
    """
    frame = noise_area(bars, cfg)
    if frame.is_empty():
        return pl.DataFrame(schema={"session": pl.Date, "bh_bps": pl.Float64})
    daily = (
        frame.group_by("session")
        .agg(pl.col("day_close").first())
        .sort("session")
        .with_columns(
            (1e4 * (pl.col("day_close") / pl.col("day_close").shift(1) - 1.0)).alias("bh_bps")
        )
        .select("session", "bh_bps")
        .drop_nulls()
    )
    return daily


def entry_time_surface(
    bars: pl.DataFrame,
    cost: CostModel,
    cfg: NoiseAreaConfig = NoiseAreaConfig(),
    first_decisions: Sequence[time] = (
        time(10, 0), time(10, 30), time(11, 0), time(12, 0), time(13, 0),
    ),
    stops: Sequence[str] = STOP_RULES,
) -> pl.DataFrame:
    """Sharpe and net return across the two knobs the paper actually chose.

    The first decision time is a free parameter the paper never justifies, and
    the stop rule is one it picks by inspecting the outcome. Both belong in a
    sweep, on dev, rather than in an assumption.
    """
    from ..metrics import tearsheet_from_returns

    rows = []
    for stop in stops:
        for first in first_decisions:
            trial = with_config(cfg, stop=stop, first_decision=first)
            daily, legs = run(bars, cost, trial)
            if daily.is_empty():
                continue
            sheet = tearsheet_from_returns(daily["net_bps"], label=f"{stop}@{first}")
            rows.append(
                {
                    "stop": stop,
                    "first_decision": first.strftime("%H:%M"),
                    "sessions": sheet["periods"],
                    "legs_per_day": legs.height / max(daily.height, 1),
                    "net_cagr_pct": sheet["cagr_pct"],
                    "sharpe": sheet["sharpe"],
                    "max_dd_pct": sheet["max_dd_pct"],
                    "hit_pct": sheet["hit_rate_pct"],
                    "cost_bps_per_day": float(daily["cost_bps"].mean()),
                }
            )
    return pl.DataFrame(rows)


def exit_reason_table(legs: pl.DataFrame) -> pl.DataFrame:
    if legs.is_empty():
        return pl.DataFrame(schema={"reason": pl.Utf8, "n": pl.Int64})
    return (
        legs.group_by("reason")
        .agg(
            pl.len().alias("n"),
            pl.col("gross_bps").mean().alias("mean_gross_bps"),
            (pl.col("gross_bps") > 0).mean().mul(100).alias("hit_pct"),
        )
        .sort("n", descending=True)
    )


def entry_hour_table(legs: pl.DataFrame) -> pl.DataFrame:
    if legs.is_empty():
        return pl.DataFrame(schema={"entry_tod": pl.Int64, "n": pl.Int64})
    return (
        legs.group_by("entry_tod")
        .agg(
            pl.len().alias("n"),
            pl.col("gross_bps").mean().alias("mean_gross_bps"),
            (pl.col("gross_bps") > 0).mean().mul(100).alias("hit_pct"),
        )
        .sort("entry_tod")
    )


def side_table(legs: pl.DataFrame) -> pl.DataFrame:
    if legs.is_empty():
        return pl.DataFrame(schema={"side": pl.Int64, "n": pl.Int64})
    return (
        legs.group_by("side")
        .agg(
            pl.len().alias("n"),
            pl.col("gross_bps").mean().alias("mean_gross_bps"),
            (pl.col("gross_bps") > 0).mean().mul(100).alias("hit_pct"),
        )
        .sort("side")
    )


def placebo(
    bars: pl.DataFrame,
    cost: CostModel,
    cfg: NoiseAreaConfig = NoiseAreaConfig(),
    *,
    seed: int = 0,
    draws: int = 200,
) -> pl.DataFrame:
    """The same trade calendar, with the *sign* of each leg randomised.

    This holds fixed everything about the strategy except its directional claim:
    same sessions, same entry and exit times, same leverage, same costs. If the
    edge is momentum rather than an artefact of when it happens to be in the
    market, the real Sharpe should sit in the tail of this distribution.
    """
    from ..metrics import tearsheet_from_returns

    daily, legs = run(bars, cost, cfg)
    if legs.is_empty():
        return pl.DataFrame(schema={"draw": pl.Int64, "sharpe": pl.Float64})
    rng = np.random.default_rng(seed)
    gross = legs["gross_bps"].to_numpy()
    costs = legs["cost_bps"].to_numpy()
    lev = legs["leverage"].to_numpy()
    codes = legs["session"].to_physical().to_numpy()
    uniq, inv = np.unique(codes, return_inverse=True)

    out = []
    for d in range(draws):
        flip = rng.choice(np.array([-1.0, 1.0]), size=gross.size)
        net = lev * (flip * gross - costs)
        agg = np.bincount(inv, weights=net, minlength=uniq.size)
        out.append({"draw": d, "sharpe": tearsheet_from_returns(pl.Series(agg))["sharpe"]})
    return pl.DataFrame(out)
