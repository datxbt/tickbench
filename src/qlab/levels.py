"""Price levels, and the three ways a bar can interact with one.

This is the mechanism Osler documented on a real FX order book, and the one
hypothesis family this project has never tested. Take-profit orders cluster
*at* round numbers, which makes price revert there; stop-loss orders cluster
just *beyond* them, which makes price run once they are crossed. Two opposite
predictions out of one book, separated by which side of the level the bar
closed on - so the formalisation has to keep them apart, and that is what this
module does.

A bar meets a level in exactly one of three ways, and the classification is a
partition rather than a menu:

``touch``
    The bar reached the level - within ``touch_eps_atr`` of it - without
    trading through, and closed back on the side it came from. This is the
    take-profit cluster holding. Direction is *away* from the level.

``sweep``
    The bar traded through the level by no more than ``sweep_max_atr``, then
    closed back on the side it came from. The marginal new extreme that the
    price-action literature calls a liquidity sweep: stops beyond the level
    were filled and the move did not survive it. Direction is away from the
    level.

``breach``
    The bar *closed* beyond the level by more than ``breach_min_atr``, having
    opened the interval on the other side of it. The cascade case. Direction is
    *with* the breach.

    This is a single-bar crossing by construction, because the level a bar is
    judged against is the nearest one to the *previous* close: once price is
    through, the level is behind it and is tested as support instead. A slow
    grind across a level is therefore deliberately not a breach - the mechanism
    being tested is stops firing at once, and a grind is the null case. The
    cost is that ``breach`` selects larger bars than ``touch`` does, so it has
    to be read against the placebo rather than against the other kinds.

Everything else - a bar nowhere near a level, or one that pierced further than
a sweep's worth without closing beyond - is not an event. There is no fourth
category, and no bar produces two kinds against the same level.

**Freshness.** A level price has been sitting on for an hour is not the level
the mechanism describes; the orders resting at it are already filled. So every
event requires the level to have been untested for ``fresh_bars`` - no close
beyond it in that window - which also stops one approach from firing forty
times as price oscillates. First occurrence per level per kind per day on top
of that.

**No lookahead.** Levels are chosen from the *previous* bar's close, never the
current one, so which level a bar is judged against cannot depend on what the
bar did. Prior-day and prior-session extremes come from completed periods only.
Under the right-edge labelling in :mod:`qlab.bars` an event stamped ``ts`` is
known at ``ts``, and :mod:`qlab.eventstudy` enters at that bar's closing quote.

**The thresholds are not optimised.** They are round numbers chosen for being
economically sensible before anything was measured - a tenth of an ATR is "at"
the level, a quarter of an ATR through it is "marginal", an hour is "fresh".
Sweeping them belongs after an effect survives, as a stability check, not
before, as a search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .session import SESSIONS

# Round-number grids in price units, which for gold is dollars per ounce. $10,
# $50 and $100 are the three a gold trader would name, and they nest - a $100
# level is also a $50 and a $10 one. The nesting is deliberate: if the effect is
# real it should be strongest on the coarsest grid, and that ordering is itself
# a test the noise has no reason to pass.
ROUND_GRIDS: tuple[float, ...] = (10.0, 50.0, 100.0)

KINDS: tuple[str, ...] = ("touch", "sweep", "breach")

_SESSION_ORDINAL = {name: index for index, (name, _, _) in enumerate(SESSIONS)}


@dataclass(frozen=True)
class LevelConfig:
    """Every threshold in the study, in one object, with its default stated.

    Held together so that a report can print the whole configuration on one
    line, and so that a stability sweep varies one field of a known set rather
    than a scattered handful of literals.
    """

    atr_bars: int = 60
    """Lookback for the ATR every distance here is measured in.

    Distances are in ATR rather than dollars because gold ran $1,500 to $4,200
    over this corpus, and a fixed dollar tolerance would mean something
    different at each end of it.
    """

    touch_eps_atr: float = 0.10
    sweep_max_atr: float = 0.25
    breach_min_atr: float = 0.25
    fresh_bars: int = 60

    approach_bars: int = 12
    """Window for the approach-speed conditioner: how hard price ran into the
    level over the bars before it got there. Osler's cascades are stronger when
    the book is thin, and a fast approach is the observable proxy."""

    grids: tuple[float, ...] = ROUND_GRIDS

    horizons: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    """Forward horizons in bars. Deliberately a sweep, and deliberately fixed:
    a fixed horizon is the one outcome measure with no barrier in it, so it
    cannot be flattered by an exit rule chosen after seeing the result."""

    families: tuple[str, ...] = field(
        default_factory=lambda: ("round", "prior_day", "prior_session")
    )

    def describe(self) -> str:
        return (
            f"ATR({self.atr_bars})  touch<={self.touch_eps_atr:g}  "
            f"sweep<={self.sweep_max_atr:g}  breach>{self.breach_min_atr:g}  "
            f"fresh={self.fresh_bars}  "
            f"grids={'/'.join(format(g, 'g') for g in self.grids)}"
        )


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def interval_seconds(frame: pl.DataFrame) -> float:
    """Seconds per bar, read off the data rather than passed in and trusted."""
    deltas = (pl.col("ts_open") - pl.col("ts_open").shift(1)).dt.total_seconds()
    value = frame.select(deltas.drop_nulls().mode().first()).item()
    if value is None or value <= 0:
        raise ValueError("could not infer the bar interval from ts_open")
    return float(value)


def prepare(bars: pl.DataFrame, cfg: LevelConfig = LevelConfig()) -> pl.DataFrame:
    """Add the true range, the ATR, the previous close and a contiguity flag.

    ``contig`` marks bars whose predecessor is the immediately preceding
    interval. Everything downstream needs it: a "previous close" taken across a
    weekend or a maintenance break is not a previous close, and the true range
    computed against it is a gap rather than a range.
    """
    if bars.is_empty():
        return bars

    frame = bars.sort("ts_open")
    step = interval_seconds(frame)

    frame = frame.with_columns(
        prev_close=pl.col("close").shift(1),
        contig=(
            (pl.col("ts_open") - pl.col("ts_open").shift(1)).dt.total_seconds() == step
        ).fill_null(False),
    )
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("prev_close")).abs(),
        (pl.col("low") - pl.col("prev_close")).abs(),
    )
    return frame.with_columns(
        # Across a gap the only defensible range is the bar's own: the distance
        # to a close from before the weekend is not something a stop could have
        # been run through.
        tr=pl.when(pl.col("contig"))
        .then(true_range)
        .otherwise(pl.col("high") - pl.col("low")),
    ).with_columns(
        atr=pl.col("tr").rolling_mean(cfg.atr_bars, min_samples=cfg.atr_bars),
        approach=(
            pl.col("close").shift(1) - pl.col("close").shift(1 + cfg.approach_bars)
        ),
    )


# ---------------------------------------------------------------------------
# Level candidates
# ---------------------------------------------------------------------------


def _round_candidates(prepared: pl.DataFrame, grid: float) -> pl.DataFrame:
    """The nearest round level above and below the previous close.

    Two rows per bar per grid, one for each direction of approach. Keying off
    ``prev_close`` is what makes this lookahead-free: the level a bar is judged
    against is fixed before the bar opens.
    """
    base = prepared.with_columns(
        _up=(pl.col("prev_close") / grid).ceil() * grid,
        _dn=(pl.col("prev_close") / grid).floor() * grid,
    )
    frames = [
        base.with_columns(
            level=pl.col(column),
            side=pl.lit(side, pl.Int8),
            family=pl.lit(f"round_{grid:g}"),
        ).drop("_up", "_dn")
        for column, side in (("_up", 1), ("_dn", -1))
    ]
    # A price sitting exactly on a round number makes ceil and floor agree; drop
    # the degenerate row rather than counting one level twice.
    return pl.concat(frames).filter(
        ((pl.col("side") == 1) & (pl.col("level") > pl.col("prev_close")))
        | ((pl.col("side") == -1) & (pl.col("level") < pl.col("prev_close")))
    )


def _prior_period_candidates(prepared: pl.DataFrame, unit: str) -> pl.DataFrame:
    """High and low of the previous completed day, or the previous session.

    "Previous" means the previous row of the aggregate, not calendar minus one:
    a Monday's prior day is Friday, and Friday's extremes are what is on the
    chart.
    """
    if unit == "day":
        keys = [pl.col("ts_open").dt.date().alias("_period")]
        order = ["_period"]
    elif unit == "session":
        hour = pl.col("ts_open").dt.hour()
        ordinal = pl.lit(None, dtype=pl.Int32)
        for name, start, end in SESSIONS:
            ordinal = (
                pl.when((hour >= start) & (hour < end))
                .then(pl.lit(_SESSION_ORDINAL[name], pl.Int32))
                .otherwise(ordinal)
            )
        keys = [pl.col("ts_open").dt.date().alias("_period"), ordinal.alias("_ordinal")]
        order = ["_period", "_ordinal"]
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown period unit {unit!r}")

    tagged = prepared.with_columns(keys)
    aggregate = (
        tagged.group_by(order)
        .agg(_hi=pl.col("high").max(), _lo=pl.col("low").min())
        .sort(order)
        .with_columns(prior_hi=pl.col("_hi").shift(1), prior_lo=pl.col("_lo").shift(1))
        .drop("_hi", "_lo")
    )
    joined = tagged.join(aggregate, on=order, how="left").drop(order)

    prefix = "pd" if unit == "day" else "ps"
    frames = [
        joined.with_columns(
            level=pl.col(column),
            # The approach side is whichever side of the level price is on, so
            # a prior high is resistance below it and support once above it.
            side=pl.when(pl.col(column) > pl.col("prev_close"))
            .then(pl.lit(1, pl.Int8))
            .otherwise(pl.lit(-1, pl.Int8)),
            family=pl.lit(f"{prefix}{suffix}"),
        ).drop("prior_hi", "prior_lo")
        for column, suffix in (("prior_hi", "h"), ("prior_lo", "l"))
    ]
    return pl.concat(frames).drop_nulls("level")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(candidates: pl.DataFrame, cfg: LevelConfig) -> pl.DataFrame:
    """Reduce candidate (bar, level, side) rows to the events among them."""
    if candidates.is_empty():
        return candidates

    approaching_up = pl.col("side") == 1
    extreme = pl.when(approaching_up).then(pl.col("high")).otherwise(pl.col("low"))
    # How far through the level the bar traded; positive means it pierced.
    pierce = (
        pl.when(approaching_up)
        .then(extreme - pl.col("level"))
        .otherwise(pl.col("level") - extreme)
    )
    # How far past the level it closed; positive means beyond.
    beyond = (
        pl.when(approaching_up)
        .then(pl.col("close") - pl.col("level"))
        .otherwise(pl.col("level") - pl.col("close"))
    )
    atr = pl.col("atr")

    kind = (
        pl.when(beyond > cfg.breach_min_atr * atr)
        .then(pl.lit("breach"))
        .when((pierce > 0) & (pierce <= cfg.sweep_max_atr * atr) & (beyond <= 0))
        .then(pl.lit("sweep"))
        .when((pierce <= 0) & (pierce >= -cfg.touch_eps_atr * atr))
        .then(pl.lit("touch"))
        .otherwise(pl.lit(None, pl.String))
    )

    events = (
        candidates.with_columns(
            kind=kind,
            pierce_atr=pierce / atr,
            dist_atr=(pl.col("level") - pl.col("prev_close")).abs() / atr,
            approach_atr=pl.col("approach") / atr * pl.col("side"),
        )
        .drop_nulls("kind")
        .with_columns(
            # A touch or a sweep is a bet away from the level, a breach a bet
            # with it. `side` is +1 when the level sits above, so away is -1.
            direction=pl.when(pl.col("kind") == "breach")
            .then(pl.col("side"))
            .otherwise(-pl.col("side"))
            .cast(pl.Int8)
        )
        .filter(
            pl.when(pl.col("kind") == "breach")
            .then(pl.col("fresh_breach"))
            .otherwise(pl.col("fresh_level"))
            & pl.col("contig")
            & pl.col("atr").is_not_null()
        )
    )

    return (
        events.with_columns(_day=pl.col("ts_open").dt.date())
        .sort("ts")
        .unique(
            subset=["family", "level", "kind", "_day"],
            keep="first",
            maintain_order=True,
        )
        .drop("_day")
    )


def _with_freshness(prepared: pl.DataFrame, cfg: LevelConfig) -> pl.DataFrame:
    """Rolling extremes of the close, for the untested-level condition."""
    return prepared.with_columns(
        _close_max=pl.col("close")
        .rolling_max(cfg.fresh_bars, min_samples=cfg.fresh_bars)
        .shift(1),
        _close_min=pl.col("close")
        .rolling_min(cfg.fresh_bars, min_samples=cfg.fresh_bars)
        .shift(1),
    )


EVENT_COLUMNS = [
    "ts",
    "ts_open",
    "family",
    "kind",
    "level",
    "side",
    "direction",
    "close",
    "bid_close",
    "ask_close",
    "atr",
    "pierce_atr",
    "dist_atr",
    "approach_atr",
    "spread_mean",
    "n_ticks",
]


def level_events(bars: pl.DataFrame, cfg: LevelConfig = LevelConfig()) -> pl.DataFrame:
    """Every level interaction in ``bars``, one row each.

    The output is an *event table*, not a trade log: it says a thing happened
    and which way the mechanism points, and says nothing about position size,
    stop or exit. :mod:`qlab.eventstudy` turns it into forward returns.
    """
    prepared = _with_freshness(prepare(bars, cfg), cfg)

    builders = []
    if "round" in cfg.families:
        builders.extend(
            (lambda p, g=grid: _round_candidates(p, g)) for grid in cfg.grids
        )
    if "prior_day" in cfg.families:
        builders.append(lambda p: _prior_period_candidates(p, "day"))
    if "prior_session" in cfg.families:
        builders.append(lambda p: _prior_period_candidates(p, "session"))
    if not builders:
        raise ValueError("no level families selected")

    # One family at a time, classified and reduced to events before the next is
    # built. A whole corpus of gold minutes times ten candidate levels is tens
    # of millions of rows and only tens of thousands of them are events; holding
    # all the candidates at once buys nothing and costs gigabytes.
    found = [
        frame
        for frame in (
            _classify(_with_fresh_flags(builder(prepared), cfg), cfg)
            for builder in builders
        )
        if not frame.is_empty()
    ]
    if not found:
        # A quiet stretch with no level interaction is a legitimate answer, not
        # an error, and callers group and filter this frame - so it has to come
        # back with the right columns rather than as an empty concat.
        return _empty_events()
    combined = pl.concat(found, how="vertical_relaxed")
    return combined.select([c for c in EVENT_COLUMNS if c in combined.columns]).sort("ts")


def _empty_events() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "ts": pl.Datetime("us", "UTC"),
        "ts_open": pl.Datetime("us", "UTC"),
        "family": pl.String,
        "kind": pl.String,
        "side": pl.Int8,
        "direction": pl.Int8,
    }
    return pl.DataFrame(
        schema={
            name: schema.get(name, pl.Float64) for name in EVENT_COLUMNS
        }
    )


def _with_fresh_flags(candidates: pl.DataFrame, cfg: LevelConfig) -> pl.DataFrame:
    """The untested-level conditions, one per kind of event."""
    beyond = cfg.breach_min_atr * pl.col("atr")
    return candidates.with_columns(
        # An untested level: no close beyond it in the freshness window. This is
        # what separates the first approach from the fortieth.
        fresh_level=pl.when(pl.col("side") == 1)
        .then(pl.col("_close_max") < pl.col("level"))
        .otherwise(pl.col("_close_min") > pl.col("level")),
        # A breach uses its own threshold, and it has to. A level price grinds
        # across a cent at a time is stale by the plain test before any close
        # gets decisively beyond it, so the plain test would leave `breach`
        # measuring only the moves that clear a quarter of an ATR from a
        # standing start - a volatility filter smuggled in as a definition.
        fresh_breach=pl.when(pl.col("side") == 1)
        .then(pl.col("_close_max") < pl.col("level") + beyond)
        .otherwise(pl.col("_close_min") > pl.col("level") - beyond),
    )
