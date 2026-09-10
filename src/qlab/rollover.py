"""The price half of the overnight basis.

Stage 2 closed with a list of things the cost model does not include, and
overnight swap was on it: "the feed carries no size" covers market impact, and
a quote feed carries no financing at all. That is still true - the swap *charge*
cannot be recovered from this corpus.

What can be recovered is the other half. A position held across the broker's
daily rollover earns whatever the price does over that window, and that price
move is in the tape. This module measures it.

Why it matters
--------------
The two halves are not independent. Spot FX settles T+2, so at each rollover the
value date rolls forward and the quote adjusts by the tom-next forward points -
which *are* the interest differential. The broker then charges swap to offset
it. A client sees both, and only the difference is real.

The measurement here is decisive about which one is looking at you, because the
sign is predicted:

* a long that **pays** carry should see the price drift **up** across the roll
* a long that **receives** carry should see it drift **down**

Run :func:`basis_table` across instruments whose carry runs in opposite
directions and the prediction is either confirmed or it is not. On this corpus
it is confirmed on all three quoting instruments, which is what killed the
EURUSD rollover trade - see ``docs/findings/eurusd-rejected.md``.

What this is not
----------------
It is **not** a swap estimate, and the difference is the whole point. Knowing
the price leg tells you what a strategy earns for crossing the roll; it does not
tell you what it is charged. Any strategy that holds overnight still has an
unmeasured term, and this module narrows it rather than closing it.

Fills are at real bid and ask, and both endpoints sit **outside** the wide
window: the rollover hours carry a spread of 1.5-1.6 bps on the majors against a
weekday mean near 0.015, so a mid-to-mid measurement there would be reading a
quote-management artifact rather than a price. :func:`basis` reports both, and
the gap between them is how much of the apparent drift is not tradable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .loader import load_bars
from .symbols import get_spec

# The broker rolls at 21:00 UTC in northern summer and 22:00 in winter - it
# tracks 17:00 New York, the market-wide FX day boundary. Entering at 20:55 and
# leaving at 23:55 UTC therefore straddles the roll in either season while
# putting both fills in an hour whose spread is normal.
DEFAULT_ENTRY_UTC_MIN = 20 * 60 + 55
DEFAULT_EXIT_UTC_MIN = 23 * 60 + 55
WINDOW_MIN = 10
"""How far past the target minute to look for a fill, in minutes. A target that
happens to fall in a gap would otherwise drop the whole day."""


@dataclass(frozen=True)
class Basis:
    """One instrument's overnight price basis over one period."""

    symbol: str
    split: str | None
    n: int
    mid_bps: float
    """Mid-to-mid drift across the roll. Includes whatever the spread does."""
    tradable_bps: float
    """Buy the ask at the entry, sell the bid at the exit. What a long earns
    from price alone, before commission and before any swap."""
    net_bps: float
    """Tradable, less the round-turn commission. Still before swap."""
    t_stat: float
    hit_rate: float
    spread_entry_bps: float
    spread_exit_bps: float

    @property
    def artifact_bps(self) -> float:
        """How much of the mid drift is not available at tradable prices."""
        return self.mid_bps - self.tradable_bps

    def describe(self) -> str:
        return (
            f"{self.symbol:<7} {str(self.split or 'all'):<11} n={self.n:4d}  "
            f"mid {self.mid_bps:+6.3f}  tradable {self.tradable_bps:+6.3f}  "
            f"net {self.net_bps:+6.3f} bps/night  t={self.t_stat:+5.2f}  "
            f"hit={100 * self.hit_rate:4.1f}%"
        )


def _fills(
    symbol: str,
    split: str | None,
    minute: int,
    *,
    start=None,
    end=None,
    allow_test: bool = False,
) -> pl.DataFrame:
    """Bid, ask and mid at the first quoted minute at or after ``minute`` UTC."""
    bars = load_bars(
        symbol, "1m", split=split, start=start, end=end, allow_test=allow_test,
        columns=["ts", "ts_open", "close", "bid_close", "ask_close"],
    )
    bars = bars.with_columns(
        day=pl.col("ts_open").dt.date(),
        m=pl.col("ts_open").dt.hour().cast(pl.Int32) * 60
          + pl.col("ts_open").dt.minute().cast(pl.Int32),
        dow=pl.col("ts_open").dt.weekday(),
    )
    return (
        bars.filter((pl.col("m") >= minute) & (pl.col("m") < minute + WINDOW_MIN))
        .group_by("day")
        .agg(
            bid=pl.col("bid_close").first(),
            ask=pl.col("ask_close").first(),
            mid=pl.col("close").first(),
            at=pl.col("m").first(),
            dow=pl.col("dow").first(),
        )
        .sort("day")
    )


def basis(
    symbol: str,
    *,
    split: str | None = None,
    start=None,
    end=None,
    entry_min: int = DEFAULT_ENTRY_UTC_MIN,
    exit_min: int = DEFAULT_EXIT_UTC_MIN,
    allow_test: bool = False,
) -> Basis:
    """Measure the overnight price basis for one instrument.

    Long convention throughout: buy at the entry, sell at the exit. A positive
    number means a long earns from price by crossing the roll, which - if the
    carry story holds - is the compensation for the swap it is about to pay.
    """
    spec = get_spec(symbol)
    e = _fills(symbol, split, entry_min, start=start, end=end, allow_test=allow_test)
    x = _fills(symbol, split, exit_min, start=start, end=end, allow_test=allow_test)
    if exit_min < entry_min:
        # An exit past midnight belongs to the day the entry was taken.
        x = x.with_columns(day=pl.col("day") - pl.duration(days=1))

    j = (
        e.rename({"bid": "e_bid", "ask": "e_ask", "mid": "e_mid"})
        .join(
            x.rename({"bid": "x_bid", "ask": "x_ask", "mid": "x_mid"}).drop("dow", "at"),
            on="day", how="inner",
        )
        # Weekends have no roll of this kind; the Sunday reopen is its own animal
        # and is flagged, not folded in here.
        .filter(pl.col("dow") <= 5)
    )
    if j.is_empty():
        raise ValueError(f"{symbol}: no days with quotes at both {entry_min} and {exit_min} UTC")

    mid = (j["x_mid"].to_numpy() / j["e_mid"].to_numpy() - 1) * 1e4
    tradable = (j["x_bid"].to_numpy() / j["e_ask"].to_numpy() - 1) * 1e4
    good = np.isfinite(mid) & np.isfinite(tradable)
    mid, tradable = mid[good], tradable[good]

    price = float(np.median(j["e_mid"].to_numpy()))
    commission_bps = spec.commission_pips(1.0 if spec.quote_ccy == "USD" else price) \
        * spec.pip / price * 1e4
    net = tradable - commission_bps

    n = mid.size
    sd = float(tradable.std())
    return Basis(
        symbol=spec.name,
        split=split,
        n=n,
        mid_bps=float(mid.mean()),
        tradable_bps=float(tradable.mean()),
        net_bps=float(net.mean()),
        t_stat=float(tradable.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan"),
        hit_rate=float((net > 0).mean()),
        spread_entry_bps=float(
            ((j["e_ask"] - j["e_bid"]) / j["e_mid"] * 1e4).mean()
        ),
        spread_exit_bps=float(
            ((j["x_ask"] - j["x_bid"]) / j["x_mid"] * 1e4).mean()
        ),
    )


# Sign of the carry on a LONG position, from the policy rates over this corpus.
# ECB sat below the Fed throughout, so a long EURUSD pays; JPY sat at or below
# zero throughout, so a long USDJPY receives; gold pays financing and yields
# nothing. These are facts about the period, not estimates from the data - which
# is what makes them usable as a prediction to test the data against.
CARRY_SIGN_ON_LONG: dict[str, int] = {
    "EURUSD": -1,   # pays
    "USDJPY": +1,   # receives
    "XAUUSD": -1,   # pays
}


def basis_table(
    symbols=("EURUSD", "USDJPY", "XAUUSD"),
    splits=("dev", "validation"),
    **kwargs,
) -> pl.DataFrame:
    """The basis for several instruments, with the carry prediction beside it.

    ``carry_predicts`` is ``+1`` where carry compensation implies an upward
    price drift for a long (because the long pays), ``-1`` where it implies a
    downward one. ``agrees`` is whether the measured drift has that sign. Rows
    that agree across instruments whose carry runs in opposite directions are
    evidence the drift is the roll, not an edge.
    """
    rows = []
    for sym in symbols:
        for split in splits:
            b = basis(sym, split=split, **kwargs)
            predicted = -CARRY_SIGN_ON_LONG.get(sym, 0)
            rows.append({
                "symbol": b.symbol,
                "split": b.split,
                "n": b.n,
                "mid_bps": round(b.mid_bps, 3),
                "tradable_bps": round(b.tradable_bps, 3),
                "net_bps": round(b.net_bps, 3),
                "t": round(b.t_stat, 2),
                "hit_pct": round(100 * b.hit_rate, 1),
                "carry_predicts": predicted,
                "agrees": bool(np.sign(b.mid_bps) == predicted),
            })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# How much of an instrument's return is financing?
# --------------------------------------------------------------------------

ROLL_HOURS_UTC: tuple[int, ...] = (21, 22, 23)
"""The hours the daily roll lives in, in UTC. The broker rolls at 17:00 New
York, which is 21:00 or 22:00 UTC depending on daylight saving, and the price
finishes adjusting into the hour after. Gold and USTEC also halt inside this
window, so their share of the roll shows up partly as a gap rather than as
quoted minutes - :func:`drift_decomposition` reports the two separately."""


def drift_decomposition(
    symbol: str,
    *,
    split: str | None = None,
    allow_test: bool = False,
    roll_hours: tuple[int, ...] = ROLL_HOURS_UTC,
) -> dict:
    """Split an instrument's price return into the part around the roll and the rest.

    The question this answers is the one that decides whether a long is worth
    holding: **how much of what this thing returned is the financing adjustment,
    which a long earns in the price and hands straight back in swap?**

    Three buckets, which sum to the total price move:

    ``roll_pct``
        Contiguous minutes inside ``roll_hours``. Offset by swap.
    ``gap_pct``
        Everything no contiguous minute covers - the daily halt, weekends,
        outages. On an instrument that halts across the roll this is where most
        of the financing adjustment hides.
    ``liquid_pct``
        Everything else. **This is the only bucket a strategy that goes home
        flat every night can reach**, and on gold over dev it is 0.42% a year
        against a headline 5.7%.

    The decomposition is descriptive, not causal: a genuine trend that happens
    to move price during hour 22 lands in ``roll_pct`` too. What makes it
    informative is the comparison across instruments - see the report in
    ``docs/findings/xauusd-rejected.md``.
    """
    spec = get_spec(symbol)
    bars = load_bars(
        spec.name, "1m", split=split, allow_test=allow_test,
        columns=["ts", "ts_open", "close"],
    ).sort("ts_open")
    bars = bars.with_columns(
        hour=pl.col("ts_open").dt.hour().cast(pl.Int32),
        contiguous=((pl.col("ts_open") - pl.col("ts_open").shift(1))
                    .dt.total_minutes() == 1).fill_null(False),
    )
    bars = bars.with_columns(
        lr=pl.when(pl.col("contiguous"))
            .then((pl.col("close") / pl.col("close").shift(1)).log() * 1e4)
    )
    close = bars["close"].to_numpy().astype(float)
    lr = bars["lr"].to_numpy()
    hour = bars["hour"].to_numpy()
    in_roll = np.isin(hour, list(roll_hours)) & np.isfinite(lr)
    in_liquid = ~np.isin(hour, list(roll_hours)) & np.isfinite(lr)

    total_bps = float(np.log(close[-1] / close[0]) * 1e4)
    roll_bps = float(np.nansum(lr[in_roll]))
    liquid_bps = float(np.nansum(lr[in_liquid]))
    # Whatever no contiguous minute accounted for: halts, weekends, outages.
    gap_bps = total_bps - roll_bps - liquid_bps

    sessions = bars.select(pl.col("ts_open").dt.date().n_unique()).item()
    years = sessions / 252
    return {
        "symbol": spec.name,
        "split": split,
        "years": years,
        "total_pct": total_bps / 100,
        "roll_pct": roll_bps / 100,
        "gap_pct": gap_bps / 100,
        "liquid_pct": liquid_bps / 100,
        "liquid_pct_per_year": liquid_bps / 100 / years if years else float("nan"),
        "financing_share": (roll_bps + gap_bps) / total_bps
        if total_bps != 0 else float("nan"),
    }


def drift_table(
    symbols=("XAUUSD", "USTEC", "EURUSD", "USDJPY"),
    splits=("dev", "validation"),
    **kwargs,
) -> pl.DataFrame:
    """:func:`drift_decomposition` across several instruments, for comparison."""
    return pl.DataFrame([
        drift_decomposition(s, split=sp, **kwargs) for s in symbols for sp in splits
    ])
