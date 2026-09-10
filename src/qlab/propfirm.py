"""Prop-firm evaluation: a first-passage problem, not a Sharpe problem.

A funded-account challenge asks a different question from every other report in
this project. Sharpe, CAGR and Calmar describe a strategy run forever. A
challenge is a **barrier problem**: reach +X% before touching either of two
floors, one of which resets every night.

Three things follow, and each of them changes what "good" means:

**Position size stops being a preference and becomes the whole decision.**
For a diffusion with drift ``mu`` and volatility ``sigma`` the probability of
touching ``+a`` before ``-b`` depends on ``theta = 2*mu/sigma**2``. Scaling a
strategy by ``k`` scales ``mu`` by ``k`` and ``sigma`` by ``k``, so ``theta``
scales by ``1/k``: **halving the size roughly doubles theta and pushes the pass
probability toward one.** The only thing size buys is speed. Since FTMO dropped
its calendar limit, speed is nearly free to give up - which makes the sizing
answer unusually clear-cut, and unusually far from what most people do.

**The daily floor is a floating-equity rule, so a close-to-close backtest cannot
see it.** A position that dips 6% intraday and closes flat has failed the
challenge and passed the backtest. This module therefore takes a per-day *worst
excursion* alongside the close-to-close return, and
:func:`daily_records_from_minutes` builds both from the tape.

**The total floor is static, not trailing.** FTMO measures the 10% against the
*initial* balance, so a run that gets to +6% has 16% of room rather than 10%.
That is a large asymmetry in the trader's favour and one a naive simulation
throws away by tracking drawdown from the peak.

Nothing here is an endorsement of prop accounts, and the numbers this produces
are only ever as good as the return series fed in. A strategy whose edge is a
backtest artifact fails a challenge exactly as fast as it fails live.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from .loader import load_bars


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    """One stage of a challenge."""

    name: str
    profit_target: float
    """Fraction of the initial balance that must be reached to advance."""
    min_trading_days: int = 4


@dataclass(frozen=True)
class PropRules:
    """A funded-account programme, as arithmetic.

    Defaults describe FTMO's two-step Challenge as published in 2025: 10% then
    5% targets, a 5% daily loss limit and a 10% overall loss limit, both against
    the initial balance, a four-day minimum and no calendar deadline. **Check
    them against the current terms before trusting a number** - these change,
    and the whole answer moves with them.
    """

    phases: tuple[Phase, ...] = (
        Phase("challenge", 0.10),
        Phase("verification", 0.05),
    )
    max_daily_loss: float = 0.05
    """Measured from the balance at the start of the trading day, on equity
    including open positions. Resets at the broker's midnight."""

    max_total_loss: float = 0.10
    """Measured from the **initial** balance and static - not a trailing
    drawdown. This is the single most favourable rule in the set."""

    max_days: int | None = None
    """Calendar limit, or None. FTMO removed theirs; other firms have not."""

    daily_stop: float | None = None
    """Flatten for the day once the floating loss reaches this fraction of the
    day's starting balance, and do not re-enter until tomorrow.

    This is the only tool that addresses the daily floor directly. Cushion
    sizing protects the *total* floor and does nothing for the daily one,
    because that floor resets each night no matter how much room the account
    has. A circuit breaker set inside the limit converts an uncontrolled breach
    into a controlled, known loss - which is what makes larger sizes survivable
    and is therefore the only route to a short timeline.

    Set it meaningfully inside the limit: the stop is a market order into a
    falling book, so ``daily_stop_slippage`` is added on top of it."""

    daily_stop_slippage: float = 0.002
    """How much worse than the stop the flatten actually fills, as a fraction of
    equity. A circuit breaker fires precisely when the market is moving, so
    assuming it fills at its level is the optimistic case, not the base one."""

    fee: float = 0.0
    """Challenge fee as a fraction of the account size, for the economics."""

    profit_split: float = 0.80


FTMO_TWO_STEP = PropRules()
"""FTMO 100k two-step. ``fee`` left at zero; set it for the economics."""


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

FAIL_DAILY = "daily_loss"
FAIL_TOTAL = "total_loss"
FAIL_TIME = "ran_out_of_days"
INCOMPLETE = "still_running"
PASSED = "passed"


@dataclass
class Attempt:
    """What happened to one run at one phase."""

    phase: str
    outcome: str
    days: int
    peak_equity: float
    trough_equity: float
    final_equity: float

    @property
    def passed(self) -> bool:
        return self.outcome == PASSED


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def flat_size(scale: float):
    """Trade the same multiple of the strategy every day."""
    def sizer(equity: float, floor: float) -> float:  # noqa: ARG001
        return scale
    sizer.label = f"flat {scale:.2f}x"
    return sizer


def cushion_size(scale: float, cap: float = 3.0):
    """Size in proportion to the distance still left to the total-loss floor.

    This is the rule the FTMO terms invite, and most people leave it on the
    table. The 10% total-loss limit is measured against the **initial** balance
    and never moves, so an account at +6% is not 10% from the floor, it is 16%
    from it. Sizing on the remaining cushion rather than on the balance means
    starting small, when there is least room, and growing only with room that
    has actually been earned.

    In continuous time it makes the floor unreachable, because the position goes
    to zero as the cushion does. Gaps break that guarantee - which is why the
    simulation still reports total-loss failures - but it removes most of them.
    """
    def sizer(equity: float, floor: float) -> float:
        cushion = (equity - floor) / (1.0 - floor)
        return float(np.clip(scale * cushion, 0.0, scale * cap))
    sizer.label = f"cushion {scale:.2f}x"
    return sizer


def run_phase(
    ret: np.ndarray,
    worst: np.ndarray,
    phase: Phase,
    rules: PropRules,
    start: int = 0,
    sizer=None,
    gap: np.ndarray | None = None,
) -> tuple[Attempt, int]:
    """Walk one phase from ``start``. Returns the attempt and the index reached.

    ``ret[i]`` is the day's close-to-close return at unit scale; ``worst[i]`` is
    the worst floating excursion during that day, also at unit scale, as a
    fraction of the equity the day began with. ``worst`` is what makes the daily
    rule real - the check happens against the trough, not the close.

    ``sizer(equity, floor) -> multiple`` decides the day's size from the equity
    it opens with. Defaults to unit size.

    ``gap[i]`` is the part of the day's move that happened while the market was
    shut - across the daily halt and over the weekend - and it is the part no
    circuit breaker can do anything about. Separating it matters only once the
    size is large: a breaker that is assumed to fire inside a gap turns the one
    risk leverage cannot manage into one that is managed for free, which is the
    most flattering error a challenge simulation can make. Passing ``None``
    treats the whole day as stoppable, which is the old, optimistic behaviour.
    """
    sizer = sizer or flat_size(1.0)
    floor = 1.0 - rules.max_total_loss
    equity = 1.0
    peak = trough = 1.0
    i = start
    n = ret.size
    traded = 0
    while i < n:
        if rules.max_days is not None and traded >= rules.max_days:
            return Attempt(phase.name, FAIL_TIME, traded, peak, trough, equity), i
        day_start = equity
        k = sizer(day_start, floor)
        day_ret, day_worst = k * ret[i], k * worst[i]
        if rules.daily_stop is not None and day_worst <= -rules.daily_stop:
            g = 0.0 if gap is None else k * gap[i]
            if g <= -rules.daily_stop:
                # The market reopened through the stop. The best that can be
                # done is to flatten at the reopen, wherever that is.
                day_ret = day_worst = g - rules.daily_stop_slippage
            else:
                # The breaker fires during trading: the day is closed out near
                # the stop and the position does not come back until tomorrow.
                day_ret = day_worst = -(rules.daily_stop + rules.daily_stop_slippage)
        low = day_start * (1.0 + min(day_worst, day_ret))
        trough = min(trough, low)
        # Both floors bite on the intraday trough, not on the close.
        if low <= day_start * (1.0 - rules.max_daily_loss) + 1e-12:
            return Attempt(phase.name, FAIL_DAILY, traded + 1, peak, low, low), i + 1
        if low <= floor + 1e-12:
            return Attempt(phase.name, FAIL_TOTAL, traded + 1, peak, low, low), i + 1
        equity = day_start * (1.0 + day_ret)
        peak = max(peak, equity)
        traded += 1
        i += 1
        # Targets are assessed on closed balance, so on the close rather than
        # on the day's high.
        if equity >= 1.0 + phase.profit_target and traded >= phase.min_trading_days:
            return Attempt(phase.name, PASSED, traded, peak, trough, equity), i
    return Attempt(phase.name, INCOMPLETE, traded, peak, trough, equity), i


def run_program(
    ret: np.ndarray,
    worst: np.ndarray,
    rules: PropRules = FTMO_TWO_STEP,
    start: int = 0,
    sizer=None,
    gap: np.ndarray | None = None,
) -> list[Attempt]:
    """Walk every phase in sequence from ``start``, stopping at the first failure.

    Each phase restarts the equity at 1.0, because each phase is a fresh account
    with its own floors - passing the Challenge does not carry a profit buffer
    into Verification.
    """
    out: list[Attempt] = []
    i = start
    for phase in rules.phases:
        attempt, i = run_phase(ret, worst, phase, rules, i, sizer=sizer, gap=gap)
        out.append(attempt)
        if not attempt.passed:
            break
    return out


# --------------------------------------------------------------------------
# Evaluating a strategy
# --------------------------------------------------------------------------

def by_start_date(
    ret: np.ndarray,
    worst: np.ndarray,
    rules: PropRules = FTMO_TWO_STEP,
    *,
    scale: float = 1.0,
) -> pl.DataFrame:
    """Run the programme from every possible start day in the history.

    This is the honest version of "would it have worked": a challenge begun on
    an unlucky Tuesday is a different experiment from one begun a month later,
    and a single backtest path hides that completely.
    """
    r, w = ret * scale, worst * scale
    rows = []
    for s in range(r.size):
        attempts = run_program(r, w, rules, start=s)
        rows.append({
            "start": s,
            "phases_passed": sum(a.passed for a in attempts),
            "passed_all": all(a.passed for a in attempts)
                          and len(attempts) == len(rules.phases),
            "outcome": attempts[-1].outcome,
            "days": sum(a.days for a in attempts),
            "worst_equity": min(a.trough_equity for a in attempts),
        })
    return pl.DataFrame(rows)


def block_bootstrap(
    ret: np.ndarray,
    worst: np.ndarray,
    rules: PropRules = FTMO_TWO_STEP,
    *,
    scale: float = 1.0,
    sizer=None,
    gap: np.ndarray | None = None,
    n_paths: int = 2000,
    block: int = 20,
    horizon: int = 3000,
    seed: int = 0,
) -> dict:
    """Pass statistics over resampled paths.

    Blocks rather than single days, because the thing that kills a challenge is
    a *run* of bad days, and independent resampling destroys exactly the
    volatility clustering that produces one. ``block=20`` is about a month.
    """
    rng = np.random.default_rng(seed)
    sizer = sizer or flat_size(scale)
    r, w = ret, worst
    n = r.size
    n_blocks = horizon // block + 1
    passed = np.zeros(n_paths, bool)
    days = np.zeros(n_paths)
    reasons: dict[str, int] = {}
    for p in range(n_paths):
        idx = np.concatenate([
            np.arange(s, min(s + block, n))
            for s in rng.integers(0, max(1, n - block), n_blocks)
        ])[:horizon]
        attempts = run_program(r[idx], w[idx], rules, sizer=sizer,
                               gap=None if gap is None else gap[idx])
        passed[p] = all(a.passed for a in attempts) and len(attempts) == len(rules.phases)
        days[p] = sum(a.days for a in attempts)
        reasons[attempts[-1].outcome] = reasons.get(attempts[-1].outcome, 0) + 1
    censored = reasons.get(INCOMPLETE, 0)
    resolved = n_paths - censored
    return {
        "label": getattr(sizer, "label", f"flat {scale:.2f}x"),
        "scale": scale,
        # Within the horizon simulated. Understates when many paths are censored.
        "pass_rate": float(passed.mean()),
        # The number that matters when the programme has no deadline: of the
        # paths that finished one way or the other, how many finished by passing.
        "pass_rate_resolved": float(passed.sum() / resolved) if resolved else float("nan"),
        "censored": censored / n_paths,
        "median_days_when_passed": float(np.median(days[passed])) if passed.any() else float("nan"),
        "p90_days_when_passed": float(np.quantile(days[passed], 0.9)) if passed.any() else float("nan"),
        "outcomes": reasons,
        "n_paths": n_paths,
    }


# --------------------------------------------------------------------------
# Building the input
# --------------------------------------------------------------------------

def daily_records_from_minutes(
    symbol: str,
    weights: pl.DataFrame,
    *,
    split: str,
    allow_test: bool = False,
    broker_utc_offset: int = 2,
    cost_bps: pl.Series | None = None,
    flat_over_closures: bool = False,
    closure_cost_bps: float = 1.05,
) -> pl.DataFrame:
    """Close-to-close return and worst intraday excursion, per broker day.

    ``weights`` carries one row per strategy session with columns ``nyd`` (the
    session date) and ``weight`` (notional as a multiple of equity), which is
    the shape :mod:`qlab.strategies.risk_managed_long` produces.

    The two returned columns are what a challenge is actually judged on:

    ``ret``
        close-to-close on the broker's day.
    ``worst``
        the lowest the account went *during* the day, as a fraction of the
        equity it opened with. This is the column that makes the 5% rule
        meaningful, and no close-to-close backtest contains it.

    The broker's day is taken as UTC+``broker_utc_offset``; FTMO runs CE(S)T, so
    2 in summer and 1 in winter. The difference moves a handful of minutes
    across a boundary and is not material here, but it is an approximation and
    is named as one.

    ``flat_over_closures`` closes the position before any shutdown longer than a
    normal overnight - weekends and holidays - and re-enters at the reopen, at a
    cost of ``closure_cost_bps`` per round turn. This is not a view on weekend
    returns, which are not measurable to any useful precision. It is a response
    to their shape: on USTEC the weekend leg carries two to three times the
    weeknight standard deviation and contains the worst reopen in each split,
    while its mean is negative in both and significant in neither. A challenge
    is judged on the worst moment rather than the average one, and a weekend
    reopen is the one move no stop can be placed in front of.
    """
    bars = load_bars(symbol, "1m", split=split, allow_test=allow_test,
                     columns=["ts", "ts_open", "open", "high", "low", "close"])
    bars = (
        bars.with_columns(
            broker=pl.col("ts_open") + pl.duration(hours=broker_utc_offset),
            ny=pl.col("ts_open").dt.convert_time_zone("America/New_York"))
        .with_columns(
            bday=pl.col("broker").dt.date(),
            nymin=pl.col("ny").dt.hour().cast(pl.Int32) * 60
                  + pl.col("ny").dt.minute().cast(pl.Int32),
            nyd=pl.col("ny").dt.date())
        .sort("ts_open")
    )
    # The weight in force during a minute is the one set at the most recent
    # session open, so it is carried forward from the decision that set it.
    w = weights.select(pl.col("nyd"), pl.col("weight"))
    bars = bars.join(w, on="nyd", how="left")
    bars = bars.with_columns(
        held=pl.when(pl.col("nymin") >= 9 * 60 + 30)
              .then(pl.col("weight"))
              .otherwise(pl.col("weight").shift(1))
    ).with_columns(pl.col("held").forward_fill().fill_null(0.0))

    day = (
        bars.group_by("bday")
        .agg(
            first=pl.col("open").first(),
            lo=pl.col("low").min(),
            hi=pl.col("high").max(),
            last=pl.col("close").last(),
            weight=pl.col("held").mean(),
            weight_max=pl.col("held").max(),
            weight_first=pl.col("held").first(),
            n=pl.len(),
        )
        .sort("bday")
        .filter(pl.col("n") >= 60)
    )
    day = day.with_columns(
        px_ret=(pl.col("last") / pl.col("first") - 1),
        px_low=(pl.col("lo") / pl.col("first") - 1),
        # The market is shut between one broker day's last quote and the next
        # day's first: the daily halt on weeknights, and the whole weekend. The
        # position is carried across it and the move is taken in full, with no
        # opportunity to act. Measuring the day from its own first quote drops
        # this entirely - immaterial at half size, decisive at four times it.
        px_gap=(pl.col("first") / pl.col("last").shift(1) - 1).fill_null(0.0),
    )
    # A long is hurt by the low; the worst excursion uses the largest weight the
    # day carried, which is the conservative reading of a rule that watches
    # floating equity continuously. The gap is carried at the weight held into
    # the close, which is the weight that was actually exposed to it.
    # A closure longer than one night: the weekend, or a holiday.
    day = day.with_columns(
        closed_days=(pl.col("bday") - pl.col("bday").shift(1)).dt.total_days())
    gap_weight = pl.col("weight_first")
    if flat_over_closures:
        gap_weight = pl.when(pl.col("closed_days") > 1).then(0.0).otherwise(gap_weight)
    day = day.with_columns(
        gap=gap_weight * pl.col("px_gap"),
    ).with_columns(
        ret=pl.col("gap") + pl.col("weight") * pl.col("px_ret"),
        worst=pl.col("gap") + pl.col("weight_max") * pl.col("px_low"),
    )
    if flat_over_closures:
        # The exit before the closure and the re-entry after it are one round
        # turn, charged to the day that reopens.
        day = day.with_columns(
            ret=pl.col("ret") - pl.when(pl.col("closed_days") > 1)
                                  .then(closure_cost_bps / 1e4 * pl.col("weight_first"))
                                  .otherwise(0.0))
    if cost_bps is not None:
        day = day.with_columns(ret=pl.col("ret") - cost_bps / 1e4)
    return day.select("bday", "ret", "worst", "gap", "weight", "weight_max")


def time_to_funded(
    ret: np.ndarray,
    worst: np.ndarray,
    rules: PropRules = FTMO_TWO_STEP,
    *,
    sizer=None,
    gap: np.ndarray | None = None,
    n_paths: int = 2000,
    block: int = 20,
    horizon: int = 3000,
    restart_days: int = 1,
    max_attempts: int = 40,
    seed: int = 0,
) -> dict:
    """Calendar days until the *first* funded account, retries included.

    :func:`block_bootstrap` reports ``median_days_when_passed``, which conditions
    on success and therefore flatters nothing and describes no one: it is the
    time taken by the runs that worked, and it says nothing about the runs that
    did not. Someone whose goal is a funded account does not stop at a failure.
    They buy another entry and start again, so what they experience is the sum
    over attempts until one sticks.

    That distinction inverts the ranking. A size with a 100% pass rate and a
    two-year run beats nothing; a size that resolves in three weeks at a 25%
    pass rate reaches a funded account sooner **and** is the more expensive
    route, because the fee is paid four times. Speed here is bought with money,
    not with edge, and this function prices it.

    A failed attempt consumes the days it lasted before failing - which is the
    other half of why aggressive sizing is fast: it fails quickly as well as
    passing quickly. ``restart_days`` covers buying the next entry.

    Paths that reach ``max_attempts`` or run off the end of the horizon are
    censored, not counted as failures, and the censored fraction is returned so
    a quantile computed from too few resolved paths can be spotted.
    """
    rng = np.random.default_rng(seed)
    sizer = sizer or flat_size(1.0)
    n = ret.size
    n_blocks = horizon // block + 1

    days = np.full(n_paths, np.nan)
    tries = np.full(n_paths, np.nan)
    for p in range(n_paths):
        idx = np.concatenate([
            np.arange(s, min(s + block, n))
            for s in rng.integers(0, max(1, n - block), n_blocks)
        ])[:horizon]
        r, w = ret[idx], worst[idx]
        g = None if gap is None else gap[idx]
        i = elapsed = attempts = 0
        while i < r.size and attempts < max_attempts:
            program = run_program(r, w, rules, start=i, sizer=sizer, gap=g)
            attempts += 1
            used = sum(a.days for a in program)
            i += used
            elapsed += used
            if all(a.passed for a in program) and len(program) == len(rules.phases):
                days[p], tries[p] = elapsed, attempts
                break
            if program[-1].outcome == INCOMPLETE:
                break          # ran off the end of the sample, not a failure
            i += restart_days
            elapsed += restart_days
    done = np.isfinite(days)
    q = (lambda a: float(np.quantile(days[done], a))) if done.any() else (lambda a: float("nan"))
    return {
        "label": getattr(sizer, "label", "flat 1.00x"),
        "censored": float(1.0 - done.mean()),
        "median_days": q(0.5),
        "p25_days": q(0.25),
        "p75_days": q(0.75),
        "p90_days": q(0.90),
        "median_months": q(0.5) / 21.0,
        "p90_months": q(0.90) / 21.0,
        "mean_attempts": float(np.nanmean(tries)) if done.any() else float("nan"),
        "median_attempts": float(np.nanmedian(tries)) if done.any() else float("nan"),
        "n_paths": n_paths,
    }
