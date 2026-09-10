"""Risk-managed long exposure to a US index CFD.

What this is, stated plainly
----------------------------
This strategy does not claim a directional forecast. Sixteen of them were tried
on USTEC first - intraday seasonality, session-boundary gaps, minute-level
autoregression, cross-asset lead-lag, gap continuation, turn-of-month,
day-of-week, dip-buying - and the ones that were not noise in the first place
were an order of magnitude smaller than the round turn. The report in
``docs/findings/ustec-risk-managed-long.md`` lists every one of them.

What survived is not a forecast of *return*, it is a forecast of *risk*.
Realised volatility predicts realised volatility with a correlation above 0.5
at a one-month horizon, which is the strongest and most reliable regularity in
the corpus. That is enough to build a strategy on, because on a leveraged CFD
account position size is not an afterthought - an unsized long is a margin call
waiting for a bad month, and the same long sized to a constant risk budget is a
business.

So the return here comes from the equity risk premium, and the work is in
harvesting it at a controlled and roughly constant volatility rather than at
whatever volatility the market happens to be running.

The rules
---------
One decision per US cash session, all of it from closed data:

1. **Trend gate.** ``close > SMA(close, ma_days)`` measured on cash-session
   closes. 200 is the conventional length and it is not tuned here.
2. **Volatility estimate.** Standard deviation of the last ``vol_days``
   close-to-close cash-session returns, annualised.
3. **Weight.** ``target_vol / realised_vol``, clipped into
   ``[0, max_weight]``, multiplied by the gate (or by ``gate_floor`` when the
   gate is off, so that "below the average" can mean *smaller* rather than
   *flat*).
4. **No-trade band.** The position only moves when the target differs from what
   is held by more than ``band``. This is what keeps turnover - and therefore
   cost - close to nothing.

Timing, and why it cannot look ahead
------------------------------------
Everything in the signal is known at the previous session's cash close; the
trade happens at the next cash open. The return a weight earns is measured
open-to-open, so a weight is never applied to a return that was already partly
observed when the weight was chosen. This is the one place a strategy like this
usually leaks, which is why the panel carries the open and the close as separate
columns rather than a single daily price.

Costs
-----
A weight change of ``|dw|`` is *one* side of a round turn, so it is charged half
of :meth:`CostModel.round_turn_bps` - half the spread, one commission, one
slippage - at the price it actually traded at.

Swap is the term this corpus cannot measure: the feed carries quotes, not
financing. It is therefore an explicit input, defaulting to zero, and the report
states the break-even rather than burying an assumption inside a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import load_bars
from ..symbols import get_spec

NY = "America/New_York"
CASH_OPEN_MIN = 9 * 60 + 30
CASH_CLOSE_MIN = 16 * 60
MIN_SESSION_BARS = 300
"""A real cash session quotes most of its 390 minutes. Holidays and half days
fall below this and are dropped rather than half-counted."""


# --------------------------------------------------------------------------
# Session conventions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSpec:
    """Where a trading day starts and ends, and where inside it we transact.

    Two things have to be said separately, and conflating them is a mistake an
    index-only implementation never notices:

    * **which close the signal reads** - the point at which a day's information
      is complete
    * **which price the trade gets** - which had better be a liquid moment

    On a US index those coincide: the cash session opens at 09:30 New York and
    that is both the start of the day and a fine time to trade. On FX they do
    not. The market-wide FX day ends at 17:00 New York, which is also the
    broker's rollover - the two hours whose spread runs 1.5 bps against a
    weekday mean near 0.015. Executing there would hand back most of the edge,
    so the FX convention reads the 17:00 close and trades the next morning.
    """

    name: str
    day_end_min: int | None
    """New York minute at which the trading day rolls over, or ``None`` when the
    day is simply the calendar date of the session window."""

    open_min: int
    """New York minute at which the position is taken, and at which the
    open-to-open return is measured."""

    close_min: int
    """New York minute at or before which the day's closing price is read."""

    min_bars: int
    """Quoted minutes a real session must have. Below this it is a holiday or a
    half day and is dropped rather than half-counted."""

    def __post_init__(self) -> None:
        if self.day_end_min is not None and self.day_end_min != self.close_min:
            # With a rollover the day's last quote IS the close, so the two have
            # to agree or the close being read is not the one being named.
            raise ValueError(
                f"{self.name}: day_end_min {self.day_end_min} and close_min "
                f"{self.close_min} must match for a rollover session"
            )
        if not 0 <= self.open_min < 24 * 60:
            raise ValueError(f"{self.name}: open_min out of range")


US_CASH = SessionSpec(
    name="us_cash", day_end_min=None,
    open_min=CASH_OPEN_MIN, close_min=CASH_CLOSE_MIN, min_bars=MIN_SESSION_BARS,
)
"""09:30 to 16:00 New York. Signal and trade both at the cash open."""

FX_DAY = SessionSpec(
    name="fx_day", day_end_min=17 * 60,
    open_min=8 * 60, close_min=17 * 60, min_bars=1000,
)
"""The FX day ends 17:00 New York; the trade happens at 08:00 the next morning,
in the London/New York overlap, rather than into the rollover."""


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def cash_session_panel(
    symbol: str = "USTEC",
    *,
    split: str | None = None,
    start=None,
    end=None,
    warmup: timedelta | None = None,
    allow_test: bool = False,
    session: SessionSpec = US_CASH,
) -> pl.DataFrame:
    """One row per session, in New York local time.

    New York local rather than UTC because the thing being tracked is a session,
    which does not move with the UTC clock even though its UTC hour does. Keying
    off UTC would file half the year in the wrong hour - and this broker has
    twice changed the *UTC* hours of USTEC's daily halt while never changing its
    New York hours.
    """
    bars = load_bars(
        symbol, "1m", split=split, start=start, end=end, warmup=warmup,
        allow_test=allow_test,
        columns=["ts", "ts_open", "open", "high", "low", "close",
                 "bid_close", "ask_close"],
    )
    bars = (
        bars.with_columns(ny=pl.col("ts_open").dt.convert_time_zone(NY))
        .with_columns(
            mins=pl.col("ny").dt.hour().cast(pl.Int32) * 60
                 + pl.col("ny").dt.minute().cast(pl.Int32),
        )
        .sort("ny")
    )
    if session.day_end_min is None:
        bars = bars.with_columns(nyd=pl.col("ny").dt.date())
        window = bars.filter(
            (pl.col("mins") >= session.open_min) & (pl.col("mins") < session.close_min)
        )
    else:
        # Everything after the rollover belongs to the next day's session.
        bars = bars.with_columns(
            nyd=pl.when(pl.col("mins") >= session.day_end_min)
                 .then(pl.col("ny").dt.date() + pl.duration(days=1))
                 .otherwise(pl.col("ny").dt.date()))
        window = bars

    agg = [
        pl.col("close").last().alias("c"),
        pl.len().alias("n_bars"),
    ]
    if warmup is not None:
        agg.append(pl.col("is_warmup").any().alias("is_warmup"))
    day = window.group_by("nyd").agg(agg).sort("nyd")
    day = day.filter(pl.col("n_bars") >= session.min_bars)

    # The execution point is looked up separately: on FX it is nowhere near the
    # close, and even on an index it must come from a bar that actually quoted.
    at_open = (
        bars.filter((pl.col("mins") >= session.open_min)
                    & (pl.col("mins") < session.open_min + 10))
        .group_by("nyd")
        .agg(
            o=pl.col("open").first(),
            ask_o=pl.col("ask_close").first(),
            bid_o=pl.col("bid_close").first(),
            open_ts=pl.col("ts").first(),
        )
        .sort("nyd")
    )
    panel = day.join(at_open, on="nyd", how="inner").sort("nyd")
    return panel.with_columns(
        dow=pl.col("nyd").dt.weekday(),
        open_hour_utc=pl.col("open_ts").dt.hour(),
        nights=(pl.col("nyd").shift(-1) - pl.col("nyd")).dt.total_days(),
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RMLConfig:
    """Every input that changes a decision. Six numbers, and none is a fit."""

    ma_days: int = 200
    """Trend filter length in cash sessions. 200 is the conventional value."""

    vol_days: int = 20
    """Window for the volatility estimate, in cash sessions."""

    target_vol_pct: float = 15.0
    """Annualised volatility the position is sized to."""

    max_weight: float = 2.0
    """Cap on notional as a multiple of equity. The cap binds in quiet markets,
    where the volatility estimate would otherwise ask for more leverage than an
    honest account should carry into the next surprise."""

    gate_floor: float = 0.0
    """Weight multiplier while price is below its moving average. ``0.0`` exits
    entirely; a positive value de-risks without leaving the market."""

    band: float = 0.10
    """No-trade band on the weight. The position only moves when the target has
    drifted this far from what is held, which is what keeps turnover - and so
    cost - down to a few round turns a year."""

    swap_bps_per_night: float = 0.0
    """Overnight financing on a long, in basis points of notional per night.
    Not measurable from a quote feed, so it is an input and never a silent
    default. See the break-even table in the report."""


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def run(
    symbol: str = "USTEC",
    cfg: RMLConfig | None = None,
    *,
    split: str | None = None,
    start=None,
    end=None,
    allow_test: bool = False,
    cost: CostModel | None = None,
    warmup_days: int = 400,
    session: SessionSpec = US_CASH,
) -> pl.DataFrame:
    """Backtest one configuration. Returns one row per evaluated session.

    ``warmup_days`` of history is loaded *before* the split so the moving
    average and the volatility estimate are warm at its first session, and
    those rows are flagged and then dropped rather than evaluated. Without it
    the first ``ma_days`` sessions of every split are silently signal-less, and
    on an 18-month validation split that is most of the split.
    """
    cfg = cfg or RMLConfig()
    get_spec(symbol)
    warmup = timedelta(days=warmup_days) if (split or start) else None
    panel = cash_session_panel(
        symbol, split=split, start=start, end=end,
        warmup=warmup, allow_test=allow_test, session=session,
    )
    if panel.height < cfg.ma_days + cfg.vol_days + 5:
        raise ValueError(
            f"{symbol}: {panel.height} sessions is not enough to warm a "
            f"{cfg.ma_days}-session average"
        )
    cost = cost or CostModel.from_profiles(symbol, split=split)

    c = panel["c"].to_numpy().astype(float)
    o = panel["o"].to_numpy().astype(float)
    n = c.size

    # --- signal, entirely from closes up to and including session i ---------
    ret_cc = np.full(n, np.nan)
    ret_cc[1:] = np.log(c[1:] / c[:-1])
    ma = pl.Series(c).rolling_mean(cfg.ma_days).to_numpy()
    vol_ann = (pl.Series(ret_cc).rolling_std(cfg.vol_days).to_numpy()
               * np.sqrt(252) * 100)

    gate = np.where(c > ma, 1.0, cfg.gate_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = cfg.target_vol_pct / vol_ann
    target = np.clip(raw, 0.0, cfg.max_weight) * gate
    # A weight decided from session i's close is held from session i+1's open.
    target = np.concatenate([[np.nan], target[:-1]])
    target[~np.isfinite(target)] = 0.0

    # --- the no-trade band, applied as a path -------------------------------
    held = np.zeros(n)
    current = 0.0
    for i in range(n):
        if abs(target[i] - current) > cfg.band:
            current = target[i]
        held[i] = current

    # --- returns: open to open, so a weight never sees its own return -------
    gross = np.full(n, np.nan)
    gross[:-1] = np.log(o[1:] / o[:-1])
    nights = panel["nights"].fill_null(1).to_numpy().astype(float)

    # --- costs --------------------------------------------------------------
    side_bps = np.array([
        0.5 * cost.round_turn_bps(price=float(px), hour=int(h))
        for px, h in zip(o, panel["open_hour_utc"].to_numpy())
    ])
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    cost_bps = side_bps * turnover
    swap_bps = cfg.swap_bps_per_night * held * nights

    out = panel.with_columns(
        ma=pl.Series(ma),
        vol_ann=pl.Series(vol_ann),
        target_w=pl.Series(target),
        weight=pl.Series(held),
        turnover=pl.Series(turnover),
        gross_bps=pl.Series(gross * 1e4),
        cost_bps=pl.Series(cost_bps),
        swap_bps=pl.Series(swap_bps),
        side_cost_bps=pl.Series(side_bps),
    )
    out = out.with_columns(
        net_bps=pl.col("weight") * pl.col("gross_bps")
        - pl.col("cost_bps") - pl.col("swap_bps"),
        bh_bps=pl.col("gross_bps"),
    )
    if "is_warmup" in out.columns:
        out = out.filter(~pl.col("is_warmup"))
    # The final session has no next open, so its return is undefined rather
    # than zero. NaN is not null in polars, so both have to be said.
    return out.filter(pl.col("gross_bps").is_not_null()
                      & pl.col("gross_bps").is_not_nan())
