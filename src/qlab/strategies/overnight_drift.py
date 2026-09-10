"""The overnight drift, as documented by Boyarchenko, Larsen and Whelan (2022).

The claim being tested
----------------------
New York Fed Staff Report 917, *The Overnight Drift*. S&P 500 futures trade
almost around the clock, but the return does not accrue around the clock: the
hour between **02:00 and 03:00 New York** - the opening of the European cash
markets - carries an annualised 3.7% on its own, positive in 20 of 23 years,
and it is the only hour that survives a multiple-testing correction.

Their explanation is inventory risk in the sense of Grossman and Miller (1988).
Selling pressure into the US close leaves market makers long and holding risk
overnight; they unwind into the first wave of European liquidity and charge for
the wait. The prediction that follows is conditional and asymmetric: the drift
should be **largest after a US session that closed with negative order
imbalance**, and reversals after rallies should be much weaker than reversals
after selloffs.

The five strategies of their Table IX are transcribed in :data:`STRATEGIES`:

======  ========================================================
CTC     hold continuously, close to close - the passive benchmark
CTO     hold the close-to-open leg only
OTC     hold the open-to-close leg only
OD      long 02:00 - 03:00
OD+     long 01:30 - 03:30
BtD     OD+, but only after a close with negative order imbalance
======  ========================================================

Their result, 2004-2020, is that OD earns a Sharpe of 1.10 gross and -0.54 once
the bid-ask spread is paid, OD+ 1.30 and 0.26, and only the conditional BtD
survives execution: 1.78 gross, **1.10 net**, because it trades on half the days
and picks the half with the larger returns. The paper's own summary of the
unconditional version is that it is "not easily profitable" - the market makers
price their book so that the contrarian trade does not pay.

What this corpus can and cannot test
------------------------------------
**Different instrument, same session clock.** The paper trades ES. This corpus
holds USTEC, a Nasdaq-100 CFD, which quotes on the same 23-hour cycle and is
driven by the same US equity flow. The mechanism is about the US close and the
European open, so nothing in it is S&P-specific; the instrument is a
substitution, and the report treats a failure to replicate as ambiguous between
"the effect is gone" and "the effect is not in this instrument". The other three
instruments in the corpus - gold and two FX majors - are run as a falsification
set, because an inventory story about *US equity* order flow predicts nothing
for EURUSD at 02:00.

**Different era, and it is the era after publication.** The paper's sample ends
in December 2020. This corpus starts in January 2020. The overlap is eleven
months, so this is very nearly an out-of-sample test of a published result, run
over the period in which two ETFs (NightShares, launched June 2022, closed
2023) were set up explicitly to harvest it. A decayed effect is the single most
likely outcome and the report reads the numbers with that in mind.

**Order imbalance is a proxy, and the report says so every time.** RSV in the
paper is signed *trade* volume over gross volume, with the sign coming from the
aggressor. This feed carries quotes and no trades, so there is no aggressor and
no size. :func:`closing_imbalance` builds the nearest available stand-in - each
one-minute bar in the closing hour is signed by its own return and weighted by
its quote-update count - and, because a proxy that fails is uninformative about
the thing it proxies, the conditioning is also run on a quantity that needs no
proxy at all: the **closing-hour return** itself. The paper's mechanism implies
both should work, and the report reports both.

Costs
-----
Every strategy here holds for a fixed window and is flat outside it, so it pays
a **full round turn on every day it trades** - the asymmetry that decides the
paper's own table. Gross returns are mid to mid; net returns buy the ask and
sell the bid using the measured spread at that hour, then pay commission from
the published contract terms. None of the windows spans the 21:00 UTC financing
point, so no strategy in this module carries a swap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import load_bars
from ..symbols import get_spec

NY = "America/New_York"

#: The US cash session, in New York local minutes. The paper's futures session
#: closes at 16:15; the CFD tracks the cash index, so 16:00 is the boundary.
SESSION_OPEN_MIN = 9 * 60 + 30
SESSION_CLOSE_MIN = 16 * 60

#: The closing window the imbalance is measured over - the paper's 15:15-16:15
#: shifted onto the cash clock.
CLOSE_WINDOW: tuple[int, int] = (15 * 60, 16 * 60)


@dataclass(frozen=True)
class Mark:
    """A price at a wall-clock minute, and which side of it to take.

    ``kind="open"``
        The open of the first one-minute bar that begins at or after the minute.
        This is a price you can *arrive at* - it is how an entry is timed.
    ``kind="close"``
        The close of the last bar that ends at or before the minute. This is
        what a session close is, and it is not optional: USTEC stops quoting at
        the 16:00 New York cash close and does not print another bar until the
        evening reopen, so asking for the *open* at 16:00 discards 85% of the
        sample and silently replaces the rest with a price from the next
        session.
    """

    minute: int
    kind: str = "open"

    def __post_init__(self) -> None:
        if self.kind not in ("open", "close"):
            raise ValueError(f"unknown mark kind {self.kind!r}")

    @property
    def key(self) -> str:
        return f"{self.minute}c" if self.kind == "close" else str(self.minute)


CLOSE_MARK = Mark(SESSION_CLOSE_MIN, "close")
OPEN_MARK = Mark(SESSION_OPEN_MIN, "open")


@dataclass(frozen=True)
class Leg:
    """A holding period.

    ``prev_day`` marks a leg that opens on the *previous* session and carries
    over midnight, which is how CTO and CTC are expressed without a second
    calendar.
    """

    name: str
    start: Mark
    end: Mark
    prev_day: bool = False
    note: str = ""

    @property
    def marks(self) -> tuple[Mark, Mark]:
        return self.start, self.end


STRATEGIES: dict[str, Leg] = {
    "CTC": Leg("CTC", CLOSE_MARK, CLOSE_MARK, prev_day=True,
               note="passive: close to close"),
    "CTO": Leg("CTO", CLOSE_MARK, OPEN_MARK, prev_day=True,
               note="the overnight leg"),
    "OTC": Leg("OTC", OPEN_MARK, CLOSE_MARK,
               note="the daytime leg"),
    "OD": Leg("OD", Mark(2 * 60), Mark(3 * 60), note="the drift hour, 02:00-03:00"),
    "OD+": Leg("OD+", Mark(90), Mark(210), note="the wide window, 01:30-03:30"),
}

#: Every mark any strategy needs a price at.
DEFAULT_MARKS: tuple[Mark, ...] = tuple(sorted(
    {m for leg in STRATEGIES.values() for m in leg.marks},
    key=lambda m: (m.minute, m.kind),
))


# --------------------------------------------------------------------------
# Prices at a clock time
# --------------------------------------------------------------------------

def _ny(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        ny=pl.col("ts_open").dt.convert_time_zone(NY)
    ).with_columns(
        nyd=pl.col("ny").dt.date(),
        mins=pl.col("ny").dt.hour().cast(pl.Int32) * 60
        + pl.col("ny").dt.minute().cast(pl.Int32),
    )


def minute_marks(
    symbol: str,
    marks: Sequence[Mark] = DEFAULT_MARKS,
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    tolerance_min: int = 5,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The mid, and the half-spread, at each requested wall-clock mark.

    One row per (New York date, mark key). ``mid`` is the bar price the mark's
    ``kind`` selects; ``half_spread`` is half that bar's spread, so a buyer pays
    ``mid + half_spread`` and a seller receives ``mid - half_spread``.

    A mark with no bar inside ``tolerance_min`` produces **no row** rather than
    a filled one. Every window in this module is a few minutes wide, so a
    fabricated price is a fabricated return; the strategies drop the day
    instead, and the count of surviving days is reported alongside every result.
    """
    if bars is None:
        bars = load_bars(
            symbol, "1m", split=split, start=start, end=end,
            allow_test=allow_test,
            columns=["ts", "ts_open", "open", "close", "spread_mean",
                     "spread_close", "n_ticks"],
        )
    frame = _ny(bars).sort("mins")
    pieces = []
    for mark in marks:
        if mark.kind == "open":
            window = (pl.col("mins") >= mark.minute) & (
                pl.col("mins") < mark.minute + tolerance_min)
            price, spread, pick = pl.col("open"), pl.col("spread_mean"), "first"
        else:
            window = (pl.col("mins") < mark.minute) & (
                pl.col("mins") >= mark.minute - tolerance_min)
            price, spread, pick = pl.col("close"), pl.col("spread_close"), "last"
        agg = (
            frame.filter(window)
            .group_by("nyd")
            .agg(
                mid=getattr(price, pick)(),
                half_spread=getattr(spread, pick)() / 2.0,
                at_min=getattr(pl.col("mins"), pick)(),
                ts=getattr(pl.col("ts"), pick)(),
            )
            .with_columns(mark=pl.lit(mark.key))
        )
        pieces.append(agg)
    return pl.concat(pieces).sort("nyd", "mark")


def _pivot(marks: pl.DataFrame, column: str, prefix: str) -> pl.DataFrame:
    wide = marks.pivot(on="mark", index="nyd", values=column, aggregate_function="first")
    return wide.rename({c: f"{prefix}{c}" for c in wide.columns if c != "nyd"})


def price_panel(marks: pl.DataFrame) -> pl.DataFrame:
    """Marks reshaped to one row per session, with yesterday's marks attached."""
    mid = _pivot(marks, "mid", "mid_")
    half = _pivot(marks, "half_spread", "hs_")
    panel = mid.join(half, on="nyd", how="inner").sort("nyd")
    lagged = panel.select(
        pl.col("nyd"),
        *[pl.col(c).alias(f"prev_{c}") for c in panel.columns if c != "nyd"],
    ).with_columns(nyd_prev=pl.col("nyd"))
    lagged = lagged.with_columns(nyd=pl.col("nyd")).drop("nyd_prev")
    # Shift by one *row*, which is one trading session, not one calendar day:
    # a Monday's predecessor is the Friday, and a holiday has no row at all.
    lagged = lagged.with_columns(
        [pl.col(c).shift(1) for c in lagged.columns if c != "nyd"]
    )
    return panel.join(lagged, on="nyd", how="left")


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def leg_return(panel: pl.DataFrame, leg: Leg, *, net: bool = False) -> pl.Expr:
    """A long leg's return in basis points.

    Gross is mid to mid. Net buys the ask at the entry mark and sells the bid at
    the exit mark, which is the paper's own convention for its cost table -
    commission is added separately by :func:`strategy_returns`, because the
    paper's futures pay one and its table does not show it.
    """
    entry_prefix = "prev_" if leg.prev_day else ""
    p0 = pl.col(f"{entry_prefix}mid_{leg.start.key}")
    p1 = pl.col(f"mid_{leg.end.key}")
    if not net:
        return 1e4 * (p1 / p0 - 1.0)
    buy = p0 + pl.col(f"{entry_prefix}hs_{leg.start.key}")
    sell = p1 - pl.col(f"hs_{leg.end.key}")
    return 1e4 * (sell / buy - 1.0)


def hourly_returns(
    symbol: str,
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    min_bars: int = 30,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Log return of every clock hour of every session, in basis points.

    This is the paper's Figure 3 - the picture the whole result rests on. Hours
    with fewer than ``min_bars`` one-minute bars are dropped, which removes the
    daily maintenance break, the Friday close and holiday half-sessions rather
    than letting a 20-minute stub masquerade as an hour.
    """
    if bars is None:
        bars = load_bars(
            symbol, "1m", split=split, start=start, end=end,
            allow_test=allow_test, columns=["ts", "ts_open", "open", "close"],
        )
    frame = _ny(bars).with_columns(hour=(pl.col("mins") // 60).cast(pl.Int32))
    return (
        frame.sort("ny")
        .group_by("nyd", "hour")
        .agg(o=pl.col("open").first(), c=pl.col("close").last(), n_bars=pl.len())
        .filter(pl.col("n_bars") >= min_bars)
        .with_columns(ret_bps=1e4 * (pl.col("c") / pl.col("o")).log())
        .sort("nyd", "hour")
    )


# --------------------------------------------------------------------------
# The conditioning variable
# --------------------------------------------------------------------------

def closing_imbalance(
    symbol: str,
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    window: tuple[int, int] = CLOSE_WINDOW,
    min_bars: int = 30,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Two readings of what the US session did on its way out.

    ``rsv``
        Relative signed volume, as a **proxy**. The paper signs trades by
        aggressor and divides signed volume by gross volume. There are no trades
        in this feed, so each one-minute bar in the closing window is signed by
        its own return and weighted by its quote-update count::

            rsv = sum(sign(close - open) * n_ticks) / sum(n_ticks)

        It ranges over [-1, 1] like the paper's and it measures something
        adjacent to, not identical to, what the paper measures.

    ``close_ret_bps``
        The closing window's own return. No proxy is involved, and the paper's
        mechanism - selloffs into the close leave dealers long - implies it
        should condition the drift in the same direction as RSV does. Reporting
        both is the only way to tell a dead effect from a dead proxy.
    """
    if bars is None:
        bars = load_bars(
            symbol, "1m", split=split, start=start, end=end,
            allow_test=allow_test,
            columns=["ts", "ts_open", "open", "close", "n_ticks"],
        )
    lo, hi = window
    frame = _ny(bars).filter((pl.col("mins") >= lo) & (pl.col("mins") < hi))
    return (
        frame.sort("ny")
        .group_by("nyd")
        .agg(
            signed=(pl.col("close") - pl.col("open")).sign() * pl.col("n_ticks"),
            gross=pl.col("n_ticks"),
            o=pl.col("open").first(),
            c=pl.col("close").last(),
            n_bars=pl.len(),
        )
        .filter(pl.col("n_bars") >= min_bars)
        .with_columns(
            rsv=pl.col("signed").list.sum() / pl.col("gross").list.sum(),
            close_ret_bps=1e4 * (pl.col("c") / pl.col("o") - 1.0),
        )
        .drop("signed", "gross", "o", "c")
        .sort("nyd")
    )


# --------------------------------------------------------------------------
# The strategy table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Conditional:
    """A rule that decides, on each session, whether the leg is held.

    ``column`` is read from the **previous** session, which is what makes it a
    signal rather than a coincidence.
    """

    name: str
    leg: str
    column: str
    below: float | None = None
    above: float | None = None

    def mask(self) -> pl.Expr:
        value = pl.col(f"prev_{self.column}")
        expr = pl.lit(True)
        if self.below is not None:
            expr = expr & (value < self.below)
        if self.above is not None:
            expr = expr & (value > self.above)
        return expr


#: The paper's own conditional, plus its no-proxy twin and its mirror image.
CONDITIONALS: tuple[Conditional, ...] = (
    Conditional("BtD", "OD+", "rsv", below=0.0),
    Conditional("BtD(ret)", "OD+", "close_ret_bps", below=0.0),
    Conditional("Rally", "OD+", "rsv", above=0.0),
    Conditional("BtD(OD)", "OD", "rsv", below=0.0),
)


def strategy_returns(
    symbol: str,
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    costs: CostModel | None = None,
    conditionals: Sequence[Conditional] = CONDITIONALS,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per session, one column per strategy, in basis points.

    Three columns per strategy: ``<name>`` gross of everything, ``<name>_net``
    after crossing the spread, and ``<name>_all`` after commission as well. A
    conditional strategy is **zero**, not null, on a day it does not trade -
    that is the day it holds cash, and averaging it away would flatter it into a
    strategy that only exists on its good days.
    """
    spec = get_spec(symbol)
    if bars is None:
        bars = load_bars(
            symbol, "1m", split=split, start=start, end=end,
            allow_test=allow_test,
            columns=["ts", "ts_open", "open", "close", "spread_mean",
                     "spread_close", "n_ticks"],
        )
    panel = price_panel(minute_marks(symbol, bars=bars))
    imbalance = closing_imbalance(symbol, bars=bars)
    panel = panel.join(imbalance, on="nyd", how="left").sort("nyd")
    panel = panel.with_columns(
        [pl.col(c).shift(1).alias(f"prev_{c}") for c in ("rsv", "close_ret_bps")]
    )

    costs = costs or CostModel.from_profiles(symbol, split=split or "dev")
    level = float(panel[f"mid_{CLOSE_MARK.key}"].mean())
    round_turn_bps = 1e4 * costs.commission_pips(price=level) * spec.pip / level

    out = panel.select("nyd", "prev_rsv", "prev_close_ret_bps")
    for name, leg in STRATEGIES.items():
        gross = leg_return(panel, leg, net=False)
        net = leg_return(panel, leg, net=True)
        # Commission is a round turn on every day the leg is held - except CTC,
        # which is a buy-and-hold and pays it once at each end of the sample.
        comm_bps = 0.0 if name == "CTC" else round_turn_bps
        out = out.with_columns(
            panel.select(gross.alias(name)).to_series(),
            panel.select(net.alias(f"{name}_net")).to_series(),
            panel.select((net - comm_bps).alias(f"{name}_all")).to_series(),
        )

    for cond in conditionals:
        take = panel.select(cond.mask().alias("m")).to_series().fill_null(False)
        for suffix in ("", "_net", "_all"):
            source = f"{cond.leg}{suffix}"
            out = out.with_columns(
                pl.when(pl.Series("m", take))
                .then(pl.col(source))
                .otherwise(0.0)
                .alias(f"{cond.name}{suffix}")
            )
        out = out.with_columns(pl.Series(f"{cond.name}_active", take))
    return out


def summarize(
    returns: pl.DataFrame,
    columns: Sequence[str],
    *,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Annualised mean, volatility, Sharpe and t for each column.

    Sharpe is the excess-free ratio of the paper's Table IX in everything but
    the risk-free rate, which this project does not hold; over 2020-2026 that
    subtraction moves the overnight numbers by a few tenths and is stated as
    missing rather than guessed at.
    """
    rows = []
    for name in columns:
        series = returns[name].drop_nulls()
        series = series.filter(series.is_finite())
        n = series.len()
        if n < 2:
            continue
        mean, sd = float(series.mean()), float(series.std())
        active = returns.get_column(f"{name.split('_')[0]}_active", default=None)
        rows.append({
            "strategy": name,
            "n_days": n,
            "days_active": int(active.sum()) if active is not None else n,
            "mean_bps": mean,
            "ann_pct": mean / 1e4 * periods_per_year * 100.0,
            "ann_vol_pct": sd / 1e4 * np.sqrt(periods_per_year) * 100.0,
            "sharpe": mean / sd * np.sqrt(periods_per_year) if sd > 0 else float("nan"),
            "t_stat": mean / sd * np.sqrt(n) if sd > 0 else float("nan"),
            "skew": float(pl.Series(series).skew() or float("nan")),
            "hit_pct": 100.0 * float((series > 0).mean()),
        })
    return pl.DataFrame(rows)


def sort_by_signal(
    returns: pl.DataFrame,
    *,
    column: str = "prev_rsv",
    target: str = "OD+",
    quantiles: int = 5,
) -> pl.DataFrame:
    """Next-session drift, sorted on the previous close's imbalance.

    The paper's central conditional result: overnight reversals should be
    monotone in the closing imbalance and much stronger on the sell side than on
    the buy side.
    """
    frame = returns.drop_nulls([column, target])
    if frame.is_empty():
        return pl.DataFrame()
    frame = frame.with_columns(
        bucket=(pl.col(column).rank("ordinal") * quantiles
                / (pl.len() + 1)).floor().cast(pl.Int32)
    )
    return (
        frame.group_by("bucket")
        .agg(
            n=pl.len(),
            signal_mean=pl.col(column).mean(),
            mean_bps=pl.col(target).mean(),
            sd=pl.col(target).std(),
        )
        .with_columns(t=pl.col("mean_bps") / pl.col("sd") * pl.col("n").sqrt())
        .sort("bucket")
    )
