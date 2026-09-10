"""Forward returns from an event table, at the mid and at the prices you get.

This sits between the bars and the engine, and deliberately stops short of the
engine. It answers "does this event move the next hour" - a distribution - and
not "what does this strategy make" - a backtest. Keeping those apart is the
point: a backtest bundles the signal with an entry rule, a stop, a target and a
size, and when the bundle loses there is no way to tell which part did it. The
LBMA-fix study in ``xauusd-rejected.md`` is the precedent, and this module is
that method made reusable.

**Three columns, always, in this order.** Every result here is reported at the
mid, then at real bid/ask fills plus commission, then net of measured slippage
as well. The gap between the first and the last is the whole question. Gold's
round turn is 0.69 bps against a 1-minute standard deviation of a few bps, so
an effect that looks real at the mid and vanishes at the fills is the *normal*
outcome, not a surprise - it is what happened to the best candidate this
project has found. Reporting only the mid would have called that one a winner.

**Entry is at the event bar's own closing quote.** Under the right-edge
labelling in :mod:`qlab.bars`, ``ts`` is the instant the bar's contents become
known, and ``bid_close``/``ask_close`` are the quotes standing at that instant.
So a long pays ``ask_close`` at ``ts`` and is out at ``bid_close`` h bars later,
and the spread is paid through the fills rather than added afterwards - the same
convention :mod:`qlab.engine` uses, for the same reason.

**A horizon that spans a gap is dropped, not stretched.** The forward bar has to
be exactly h intervals later. Otherwise a Friday-evening event silently gets
Sunday's reopen as its outcome, which is both the widest spread of the week and
a return no position was actually exposed to.
"""

from __future__ import annotations

import math

import polars as pl

from .costs import CostModel
from .levels import interval_seconds

OUTCOME_COLUMNS = ("mid_bps", "fill_bps", "net_bps")

_LABELS = {
    "mid_bps": "mid, no cost at all",
    "fill_bps": "bid/ask fills + commission",
    "net_bps": "+ measured slippage",
}


def forward_returns(
    bars: pl.DataFrame,
    events: pl.DataFrame,
    horizons: tuple[int, ...],
    *,
    cost: CostModel,
    keep: tuple[str, ...] = (),
) -> pl.DataFrame:
    """One row per (event, horizon), signed by the event's direction, in bps.

    ``keep`` names the event columns to carry through for conditioning. The
    returned frame always carries ``ts``, ``horizon``, ``bars_held``, the three
    outcome columns and the per-event cost in bps, so a summary can always state
    what the edge had to clear.
    """
    if events.is_empty():
        return _empty(keep)

    step = interval_seconds(bars)
    priced = cost.with_costs(bars.sort("ts_open"), spread_source="realized")

    # Commission is already a round-turn figure; slippage is per side and a
    # round trip takes two fills. Both converted to bps at the event's own price
    # so that a 2020 event is charged in 2020 money. The spread is deliberately
    # absent - it is paid through the bid/ask fills below, and adding it here
    # would charge it twice.
    to_bps = cost.spec.pip / pl.col("close") * 10_000
    priced = priced.with_columns(
        _comm_bps=pl.col("commission_pips") * to_bps,
        _slip_bps=2.0 * pl.col("slippage_pips") * to_bps,
    ).select(
        "ts", "ts_open", "close", "bid_close", "ask_close", "_comm_bps", "_slip_bps"
    )

    anchored = events.select(
        ["ts", "direction", *[c for c in keep if c in events.columns]]
    ).join(priced, on="ts", how="inner")

    frames: list[pl.DataFrame] = []
    for horizon in horizons:
        forward = priced.select(
            _ts_from=pl.col("ts").shift(horizon),
            ts_fwd=pl.col("ts"),
            close_fwd=pl.col("close"),
            bid_fwd=pl.col("bid_close"),
            ask_fwd=pl.col("ask_close"),
        ).drop_nulls("_ts_from")

        joined = anchored.join(
            forward, left_on="ts", right_on="_ts_from", how="inner"
        ).filter(
            # Exactly h intervals later, or the window spans a gap.
            (pl.col("ts_fwd") - pl.col("ts")).dt.total_seconds() == horizon * step
        )
        if joined.is_empty():
            continue

        direction = pl.col("direction").cast(pl.Float64)
        long = pl.col("direction") == 1
        frames.append(
            joined.with_columns(
                horizon=pl.lit(horizon, pl.Int32),
                minutes=pl.lit(int(round(horizon * step / 60)), pl.Int32),
                mid_bps=direction * (pl.col("close_fwd") / pl.col("close") - 1) * 10_000,
                # In at the ask and out at the bid for a long, the other way for
                # a short: the spread is inside the fill, not bolted on after.
                # Commission is not in the quote, so it is charged separately.
                _gross_bps=(
                    pl.when(long)
                    .then(pl.col("bid_fwd") / pl.col("ask_close") - 1)
                    .otherwise(pl.col("bid_close") / pl.col("ask_fwd") - 1)
                )
                * 10_000,
            )
            .with_columns(
                fill_bps=pl.col("_gross_bps") - pl.col("_comm_bps"),
                cost_bps=pl.col("_comm_bps") + pl.col("_slip_bps"),
            )
            .with_columns(net_bps=pl.col("fill_bps") - pl.col("_slip_bps"))
            .drop("_gross_bps", "_comm_bps", "_slip_bps")
        )

    if not frames:
        return _empty(keep)
    return pl.concat(frames, how="vertical_relaxed").sort("horizon", "ts")


def placebo_events(
    events: pl.DataFrame, bars: pl.DataFrame, *, seed: int = 0, max_days: int = 5
) -> pl.DataFrame:
    """The same events, moved to a different day at the same time of day.

    The control the study needs. Gold's forward return is not mean-zero across
    the clock - the roll hours carry most of the corpus's drift - so "was this
    better than nothing" has to mean "better than the same trade at the same
    hour on a day when the level was not there". Shifting whole days preserves
    the time-of-day, session and weekday mix exactly, and preserves the era, so
    the only thing removed is the level.

    Directions are preserved rather than randomised, so a placebo that comes out
    positive is telling you the *direction mix* is what carries the result and
    the level is decoration.
    """
    if events.is_empty():
        return events

    offsets = (
        pl.int_range(pl.len())
        .shuffle(seed=seed)
        .mod(2 * max_days)
        .sub(max_days)
        .cast(pl.Int64)
    )
    shifted = events.with_columns(
        ts=pl.col("ts").dt.offset_by(
            # Never zero: a zero offset is the event itself.
            pl.when(offsets < 0)
            .then(offsets)
            .otherwise(offsets + 1)
            .cast(pl.String)
            + "d"
        )
    )
    return shifted.join(bars.select("ts"), on="ts", how="semi").sort("ts")


def _empty(keep: tuple[str, ...]) -> pl.DataFrame:
    schema = {
        "ts": pl.Datetime("us", "UTC"),
        "direction": pl.Int8,
        **{name: pl.Float64 for name in keep},
        "horizon": pl.Int32,
        "minutes": pl.Int32,
        "mid_bps": pl.Float64,
        "fill_bps": pl.Float64,
        "net_bps": pl.Float64,
        "cost_bps": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize(
    outcomes: pl.DataFrame,
    by: list[str] | None = None,
    *,
    column: str = "net_bps",
) -> pl.DataFrame:
    """Mean, t-stat and hit rate per group, plus the cost the mean must clear.

    The t-stat is the plain one-sample statistic. It is not a p-value and it is
    not corrected for the number of cells in the table; the correction happens
    once, at the table level, in :func:`deflated_threshold`, because that is the
    only place the number of trials is actually known.
    """
    keys = by or []
    grouped = outcomes.group_by(keys) if keys else outcomes.group_by(pl.lit(1).alias("_all"))
    table = grouped.agg(
        n=pl.len(),
        mean=pl.col(column).mean(),
        sd=pl.col(column).std(),
        hit=(pl.col(column) > 0).mean(),
        cost=pl.col("cost_bps").mean(),
        mid=pl.col("mid_bps").mean(),
    ).with_columns(
        t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt(),
    )
    if keys:
        table = table.sort(keys)
    return table.drop("_all", strict=False)


def daily_pnl(outcomes: pl.DataFrame, *, column: str = "net_bps") -> tuple[float, float, int]:
    """Bps a day from taking every one of these events, its t-stat, and n days.

    The event-level t-stat in :func:`summarize` is not usable on its own here,
    for two reasons that push the same way. At a 2-hour horizon on 5-minute bars
    consecutive events share almost all of their forward window, so it counts
    one move many times. And events *cluster*: breaches arrive three to a day
    normally and thirteen on a trending day, so an event-weighted mean quietly
    overweights exactly the days the signal was going to work on.

    Summing within a day and testing across days fixes both, and it is the
    statistic ``section_windows`` in the XAUUSD study already uses. It also
    matches the economics - a day is what an account experiences - which is why
    the mean it returns is bps *per day*, not per event.
    """
    if outcomes.is_empty():
        return float("nan"), float("nan"), 0
    daily = (
        outcomes.with_columns(_day=pl.col("ts").dt.date())
        .group_by("_day")
        .agg(total=pl.col(column).sum())["total"]
        .to_numpy()
    )
    n = daily.size
    if n < 3:
        return float(daily.mean()) if n else float("nan"), float("nan"), n
    sd = daily.std(ddof=1)
    if sd == 0:
        return float(daily.mean()), float("nan"), n
    return float(daily.mean()), float(daily.mean() / sd * math.sqrt(n)), int(n)


def deflated_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """The |t| a single cell needs when ``n_trials`` cells were inspected.

    A Sidak correction on the two-sided normal quantile. Crude next to a Reality
    Check bootstrap, and it does the one job that matters here: it stops a table
    of a hundred cells from being read as a hundred independent chances at
    t = 2. Report it beside the table and the reader can see what was demanded.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    per_trial = 1.0 - (1.0 - alpha) ** (1.0 / n_trials)
    return abs(_norm_ppf(per_trial / 2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF - Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def format_three_columns(outcomes: pl.DataFrame, by: list[str], indent: str = "  ") -> str:
    """The mid / fills / net triple per group, which is the study's core table.

    The bracketed t beside each mean is the event-level one, and it is there for
    scale rather than for inference: it is uncorrected for the overlap between
    successive events and for their clustering into days. The last two columns
    are the ones to read - bps a day from taking every event in the group, and
    the t-stat of that across days.
    """
    lines: list[str] = []
    header = "".join(f"{k:<12}" for k in by)
    lines.append(
        f"{indent}{header}{'n':>7}"
        + "".join(f"{_LABELS[c]:>27}" for c in OUTCOME_COLUMNS)
        + f"{'bps/day':>10}{'t/day':>8}"
    )
    summaries = {c: summarize(outcomes, by, column=c) for c in OUTCOME_COLUMNS}
    for row in summaries["net_bps"].iter_rows(named=True):
        key = [row[k] for k in by]
        group = outcomes
        for k, v in zip(by, key):
            group = group.filter(pl.col(k) == v)

        cells = ""
        for column in OUTCOME_COLUMNS:
            match = summaries[column]
            for k, v in zip(by, key):
                match = match.filter(pl.col(k) == v)
            stats = match.row(0, named=True)
            cells += f"{stats['mean']:>+16.3f} (t{stats['t']:>+5.2f})"

        per_day, t_day, _ = daily_pnl(group)
        label = "".join(f"{str(k):<12}" for k in key)
        lines.append(
            f"{indent}{label}{row['n']:>7}{cells}{per_day:>+10.2f}{t_day:>+8.2f}"
        )
    return "\n".join(lines)
