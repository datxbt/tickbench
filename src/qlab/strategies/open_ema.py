"""The New York open EMA rule: one candle, one EMA, one day-long bet.

The claim being tested
----------------------
Stated by its creator in four lines, for NAS100 on a 5-minute chart:

1. Wait for the New York open, 09:30 ET.
2. Take the first 5-minute candle, 09:30 -> 09:35.
3. If it closes **above** the 12-period EMA, go long; **below**, go short.
4. Trail the stop behind the position and "stay in as long as momentum
   continues". Risk 1% per trade.

Reported: NAS100 5-minute data 2019-2026, 1,448 trades, +982% total return, 57%
win rate, 1.29 profit factor, 19.7% maximum drawdown. The creator's framing is
"probability, not prediction" - a statistical edge over many trades.

What the rule actually reduces to
---------------------------------
The comparison is between the candle's close and the EMA *at that candle*, and
with the usual recursion ``e_t = a * c_t + (1 - a) * e_{t-1}`` that comparison
is not the one it looks like:

    c_t > e_t
    c_t > a * c_t + (1 - a) * e_{t-1}
    (1 - a) * c_t > (1 - a) * e_{t-1}
    c_t > e_{t-1}                          for any a < 1

So "closes above its own EMA" is identical to "closes above the EMA as it stood
before the candle opened", and the 12-period span drops out of the *sign*
entirely - it survives only in how far away the level is, never in which side
the rule takes. The rule is a **momentum print**: did the first five minutes of
the cash session close above the level the previous hour of trading had settled
at? Reading it that way removes the only ambiguity in the specification -
whether the signal bar is included in its own average - because both readings
are the same signal. :func:`signal_equivalence` pins this.

Two ambiguities the specification does leave open, and how they are handled:

**Which 5-minute series feeds the EMA.** A NAS100 chart is continuous, so the
twelve bars behind 09:35 are the hour from 08:40 ET - overnight and pre-market
tape, not yesterday's session. That is the default (``ema_series="continuous"``).
``"rth"`` computes the EMA on regular-session bars only, where the twelve bars
behind 09:35 reach back into the previous afternoon; it is reported as a
robustness variant, not a second headline.

**The trailing stop.** The specification does not state a distance, and neither
does the video. Per this project's standing rule the exit is therefore *swept*
rather than chosen: the raw signal is measured first (forward mid return, MFE
and MAE), and then three exit families are laid over the same fills at seven
distances each, so no verdict rests on one arbitrary trail. See
:data:`EXIT_STYLES` and :data:`STOP_FRACTIONS`. Distances are fractions of the
14-day regular-session ATR, the same unit :mod:`qlab.strategies.opening_range`
uses, so the two opening-bell studies read on one scale.

Controls
--------
The rule is a day-long directional bet on an instrument that roughly tripled
over the sample, entered at a fixed clock minute with a momentum-shaped side
choice. Several things could produce the reported curve and only one of them is
the claim, so five direction rules are priced rather than one:

* ``"ema"``    - the rule.
* ``"contra"`` - the rule inverted. If the signal carries information, this has
  to lose what the rule makes.
* ``"coin"``   - the side from a deterministic hash of the session date. Same
  clock, same exits, no signal. This is the mechanics-only null: it says what
  the trailing-stop geometry is worth on its own.
* ``"long"`` / ``"short"`` - always one side. ``"long"`` is the drift control,
  and on this instrument over this sample it is the one that matters: if buying
  the open every day and trailing does as well, the EMA is decoration.

Every rule is priced on the **same fills**. :func:`run` resolves the 09:35
market entry once per session, for both sides, and then evaluates every exit
cell on it, so no comparison can be contaminated by a different fill.

Why this is resolved on ticks
-----------------------------
A trailing stop is a path. Its whole behaviour is "where did price go, in what
order" - a bar's high and low cannot say whether the extreme that moved the stop
printed before or after the tick that took it out, and on a 1-minute USTEC bar
in the first hour that ordering decides the trade. Entries and exits are
therefore resolved against the tick tape: the entry at the standing quote 250 ms
after the candle closes (the feed prints only on change, so the standing quote
is the last tick at or before that instant), a long's trailing stop on the bid
and a short's on the ask.

Costs
-----
Fills cross a real bid and a real ask, so the spread is paid rather than
modelled. Commission is the published contract term. Slippage is the one assumed
term, from :class:`qlab.costs.CostModel`, charged on the entry and on the
trailing stop-out - both market orders - and on the closing bell. The session
runs 09:30-16:00 New York and never spans the 21:00 UTC financing point, so no
trade in this module pays swap.
"""

from __future__ import annotations

import hashlib
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

#: Trail and stop distances, as a fraction of the 14-day regular-session ATR.
#: Deliberately the grid :mod:`qlab.strategies.opening_range` sweeps, so the
#: two opening-bell studies are read on one axis.
STOP_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.00)

#: ``trail``    - stop starts one distance behind the entry and ratchets one
#:                distance behind the best price the trade has seen. The
#:                creator's rule as described.
#: ``fixed``    - stop never moves; the bell closes what survives. The no-trail
#:                baseline, and the only way to say what trailing is worth.
#: ``be_trail`` - stop is fixed until the trade is one distance in favour, then
#:                trails. The "let it prove itself, then ride it" reading.
EXIT_STYLES: tuple[str, ...] = ("trail", "fixed", "be_trail")

#: Direction rules. See the module docstring; all five are priced on one set of
#: fills, so they are selected from a finished run rather than re-simulated.
DIRECTION_RULES: tuple[str, ...] = ("ema", "contra", "coin", "long", "short")

#: Forward mid-return horizons, in minutes after the entry. ``None`` is the bell.
FORWARD_MINUTES: tuple[int | None, ...] = (15, 30, 60, 120, None)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenEMAConfig:
    """Everything that changes a decision."""

    ema_len: int = 12
    bar_minutes: int = 5
    atr_days: int = 14
    ema_series: str = "continuous"    # continuous | rth
    open_min: int = 9 * 60 + 30       # 09:30 New York
    close_min: int = 16 * 60          # 16:00 New York
    min_session_bars: int = 300
    open_tolerance_min: int = 5
    """How late the first quote of the session may be and still count as an
    open. A holiday half-session is dropped by ``min_session_bars`` instead."""

    def __post_init__(self) -> None:
        if self.ema_series not in ("continuous", "rth"):
            raise ValueError(f"unknown ema_series {self.ema_series!r}")
        if self.ema_len < 2:
            raise ValueError("ema_len must be at least 2")
        if self.bar_minutes <= 0 or 60 % self.bar_minutes:
            raise ValueError("bar_minutes must divide an hour")
        if self.open_min + self.bar_minutes >= self.close_min:
            raise ValueError("the signal candle does not fit inside the session")

    @property
    def signal_end_min(self) -> int:
        """New York minute at which the signal candle closes - 09:35 by default."""
        return self.open_min + self.bar_minutes


def signal_equivalence(closes: Sequence[float], span: int = 12) -> bool:
    """Check that ``close > ema(close)`` and ``close > ema(close).shift(1)`` agree.

    The module docstring's algebra, as an assertion about real numbers: the sign
    of ``c_t - e_t`` cannot differ from the sign of ``c_t - e_{t-1}``, so
    including the signal candle in its own average is not a decision anyone has
    to make.
    """
    series = pl.Series("close", list(closes), dtype=pl.Float64)
    ema = series.ewm_mean(span=span, adjust=False)
    inclusive = (series > ema).to_numpy()[1:]
    exclusive = (series > ema.shift(1)).to_numpy()[1:]
    return bool(np.array_equal(inclusive, exclusive))


def coin_side(day: date) -> int:
    """A side from the session date alone: deterministic, balanced, no signal.

    A hash rather than a seeded RNG so that the control is identical across
    runs, machines and split boundaries - the convention
    :mod:`qlab.strategies.engulfing_quadrant` established for its matched null.
    """
    digest = hashlib.blake2b(day.isoformat().encode(), digest_size=8).digest()
    return 1 if int.from_bytes(digest, "big") & 1 else -1


# --------------------------------------------------------------------------
# Daily context: the signal candle, the EMA and the ATR
# --------------------------------------------------------------------------

def _ny_minutes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        ny=pl.col("ts_open").dt.convert_time_zone(NY)
    ).with_columns(
        nyd=pl.col("ny").dt.date(),
        mins=pl.col("ny").dt.hour().cast(pl.Int32) * 60
        + pl.col("ny").dt.minute().cast(pl.Int32),
    )


def _to_signal_bars(minute: pl.DataFrame, cfg: OpenEMAConfig) -> pl.DataFrame:
    """Aggregate 1-minute bars to the signal timeframe on the New York clock.

    Grouped on the New York wall clock rather than UTC so that the 09:30 bar is
    the 09:30 bar in both halves of the year. The offsets involved are whole
    hours, so on this instrument the two grids coincide; doing it on the New York
    clock means the module does not depend on that staying true.
    """
    every = f"{cfg.bar_minutes}m"
    has_warmup = "is_warmup" in minute.columns
    return (
        minute.lazy()
        .sort("ts_open")
        .with_columns(
            _ny=pl.col("ts_open").dt.convert_time_zone(NY).dt.replace_time_zone(None)
        )
        .group_by_dynamic("_ny", every=every, closed="left", label="left")
        .agg(
            ts_open=pl.col("ts_open").first(),
            ts=pl.col("ts").last(),
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            n_ticks=pl.col("n_ticks").sum(),
            n_minutes=pl.len(),
            is_warmup=(pl.col("is_warmup").any() if has_warmup else pl.lit(False)),
        )
        .collect()
    )


def daily_context(
    symbol: str,
    cfg: OpenEMAConfig = OpenEMAConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
) -> pl.DataFrame:
    """One row per session: its signal candle, the EMA behind it and its ATR.

    Everything here is a decision input, so everything here is lagged into
    place. The EMA at the signal candle uses only closes that had printed by
    09:35; ``atr`` is the mean true range over the ``atr_days`` sessions that
    ended **before** this one. The warmup is loaded from before the split so the
    first evaluable session has a full ATR and a settled EMA, and warmup rows are
    dropped before returning.
    """
    warmup = timedelta(days=int(2.5 * cfg.atr_days) + 30)
    minute = load_bars(
        symbol, "1m", split=split, start=start, end=end, warmup=warmup,
        allow_test=allow_test,
        columns=["ts", "ts_open", "open", "high", "low", "close", "n_ticks"],
    )

    # --- the signal: the 09:30 candle's close against the EMA behind it ------
    frame = minute
    if cfg.ema_series == "rth":
        frame = _ny_minutes(minute).filter(
            (pl.col("mins") >= cfg.open_min) & (pl.col("mins") < cfg.close_min)
        ).drop("ny", "nyd", "mins")
    bars = _to_signal_bars(frame, cfg)
    # adjust=False is the chart recursion. Platforms seed it with an SMA and this
    # seeds it with the first close; after the warmup the two agree to far less
    # than a tick, so the seed is not a modelling decision.
    bars = bars.with_columns(
        ema=pl.col("close").ewm_mean(
            span=cfg.ema_len, adjust=False, min_periods=cfg.ema_len
        )
    )
    signal = (
        _ny_minutes(bars)
        .filter(pl.col("mins") == cfg.open_min)
        .select(
            "nyd",
            sig_open=pl.col("open"),
            sig_high=pl.col("high"),
            sig_low=pl.col("low"),
            sig_close=pl.col("close"),
            sig_minutes=pl.col("n_minutes"),
            sig_end_ts=pl.col("ts"),
            ema=pl.col("ema"),
        )
        .filter(pl.col("sig_minutes") >= max(1, cfg.bar_minutes - 1))
        .with_columns(ema_gap=pl.col("sig_close") - pl.col("ema"))
        .with_columns(
            signal=pl.when(pl.col("ema_gap") > 0).then(1)
            .when(pl.col("ema_gap") < 0).then(-1)
            .otherwise(0).cast(pl.Int8)
        )
    )

    # --- the session, and the ATR that scales every stop --------------------
    rth = _ny_minutes(minute).filter(
        (pl.col("mins") >= cfg.open_min) & (pl.col("mins") < cfg.close_min)
    ).sort("ny")
    has_warmup = "is_warmup" in rth.columns
    session = (
        rth.group_by("nyd")
        .agg(
            first_min=pl.col("mins").first(),
            n_bars=pl.len(),
            rth_open=pl.col("open").first(),
            rth_high=pl.col("high").max(),
            rth_low=pl.col("low").min(),
            rth_close=pl.col("close").last(),
            close_ts=pl.col("ts").last(),
            is_warmup=(pl.col("is_warmup").any() if has_warmup else pl.lit(False)),
        )
        .filter(
            (pl.col("n_bars") >= cfg.min_session_bars)
            & (pl.col("first_min") <= cfg.open_min + cfg.open_tolerance_min)
        )
        .sort("nyd")
    )

    ctx = session.join(signal, on="nyd", how="inner").sort("nyd")
    # The bell is the clock, not the last bar that happened to print. On 590 of
    # the 960 dev sessions the 15:59 minute carries no quote, so taking the last
    # bar's right edge would flatten a minute early on some days and not others -
    # a difference in the exit rule driven by the feed rather than by the rule.
    ctx = ctx.with_columns(
        bell_ts=(
            pl.col("nyd").cast(pl.Datetime("us"))
            + pl.duration(minutes=cfg.close_min)
        ).dt.replace_time_zone(NY, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC"),
    ).with_columns(
        prev_close=pl.col("rth_close").shift(1),
    ).with_columns(
        tr=pl.max_horizontal(
            pl.col("rth_high") - pl.col("rth_low"),
            (pl.col("rth_high") - pl.col("prev_close")).abs(),
            (pl.col("rth_low") - pl.col("prev_close")).abs(),
        )
    ).with_columns(
        atr=pl.col("tr").shift(1).rolling_mean(cfg.atr_days),
    ).with_columns(
        coin=pl.col("nyd").map_elements(coin_side, return_dtype=pl.Int8),
    )

    if "is_warmup" in ctx.columns:
        ctx = ctx.filter(~pl.col("is_warmup")).drop("is_warmup")
    return ctx.drop_nulls(["atr", "ema", "bell_ts"]).filter(pl.col("atr") > 0)


def side_for(row: dict, rule: str) -> int:
    """The side a direction rule takes on one session. 0 means no trade."""
    if rule == "ema":
        return int(row["signal"])
    if rule == "contra":
        return -int(row["signal"])
    if rule == "coin":
        return int(row["coin"])
    if rule == "long":
        return 1
    if rule == "short":
        return -1
    raise ValueError(f"unknown direction rule {rule!r}")


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

    def after(self, when_us: int) -> int:
        """First tick at or after ``when_us``."""
        return int(np.searchsorted(self.ts, when_us, side="left"))

    def standing(self, when_us: int) -> int:
        """The quote standing at ``when_us``: the last tick at or before it, or -1."""
        return int(np.searchsorted(self.ts, when_us, side="right")) - 1


@dataclass
class _Fill:
    i: int
    direction: int
    price: float
    mid: float
    ts_us: int


def _stop_path(
    favour: np.ndarray, entry: float, direction: int, distance: float, style: str
) -> np.ndarray:
    """The stop level at every tick of the hold, for one exit style.

    ``favour`` is the running extreme of the price the trade would *exit* at -
    the bid for a long, the ask for a short - because a stop that trails a price
    the trade cannot actually leave at is trailing a number, not a level. Every
    style below is monotone in the trade's favour by construction, so none of
    them can loosen a stop that has already tightened.
    """
    if style == "fixed":
        return np.full(favour.shape, entry - direction * distance)
    if style == "trail":
        anchor = (np.maximum(favour, entry) if direction == 1
                  else np.minimum(favour, entry))
        return anchor - direction * distance
    if style == "be_trail":
        armed = (favour >= entry + distance if direction == 1
                 else favour <= entry - distance)
        return np.where(armed, favour - direction * distance,
                        entry - direction * distance)
    raise ValueError(f"unknown exit style {style!r}")


def _simulate(
    tape: _Tape, fill: _Fill, end: int, distance: float, style: str, slip: float
) -> dict:
    """One (style, distance) geometry on one already-resolved fill."""
    direction, entry = fill.direction, fill.price
    bid, ask = tape.bid[fill.i:end], tape.ask[fill.i:end]
    out = bid if direction == 1 else ask          # the side an exit trades on
    favour = (np.maximum.accumulate(out) if direction == 1
              else np.minimum.accumulate(out))
    stop = _stop_path(favour, entry, direction, distance, style)

    hit = (out <= stop) if direction == 1 else (out >= stop)
    i_stop = _first_true(hit)
    if i_stop >= 0:
        exit_i, reason = i_stop, "stop"
    else:
        exit_i, reason = out.size - 1, "bell"
    # Both exits are market orders and both pay slippage. A stop fills at the
    # quote that triggered it, not at the level it was resting at: a trailing
    # stop that gaps through is not filled at its price.
    exit_price = out[exit_i] - direction * slip

    mid_exit = (bid[exit_i] + ask[exit_i]) / 2
    pnl = direction * (exit_price - entry)
    return {
        "exit_style": style,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "hold_min": (int(tape.ts[fill.i + exit_i]) - fill.ts_us) / (60 * US),
        "pnl": float(pnl),
        "mid_pnl": float(direction * (mid_exit - fill.mid)),
    }


def _raw_stats(tape: _Tape, fill: _Fill, end: int, atr: float) -> dict:
    """The signal before any exit: forward mid returns, MFE and MAE.

    Measured on the **mid** so that the geometry of the exit and the cost of the
    round turn are held apart from the question of whether the side was right.
    Everything is in ATR units, which is the unit the stops are quoted in, so the
    raw signal and the exit surface are directly comparable.
    """
    mid = (tape.bid[fill.i:end] + tape.ask[fill.i:end]) / 2
    excursion = fill.direction * (mid - fill.mid)
    out: dict = {
        "mfe_atr": float(excursion.max() / atr),
        "mae_atr": float(excursion.min() / atr),
        "bell_min": (int(tape.ts[end - 1]) - fill.ts_us) / (60 * US),
    }
    for horizon in FORWARD_MINUTES:
        if horizon is None:
            key, j = "fwd_bell_atr", mid.size - 1
        else:
            key = f"fwd_{horizon}m_atr"
            j = min(tape.standing(fill.ts_us + horizon * 60 * US) - fill.i,
                    mid.size - 1)
        out[key] = float(excursion[j] / atr) if j >= 0 else float("nan")
    return out


def _months(ctx: pl.DataFrame) -> list[tuple[int, int]]:
    return (
        ctx.select(y=pl.col("nyd").dt.year(), m=pl.col("nyd").dt.month())
        .unique().sort("y", "m").rows()
    )


def run(
    symbol: str,
    cfg: OpenEMAConfig = OpenEMAConfig(),
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    allow_test: bool = False,
    stop_fractions: Sequence[float] | None = None,
    styles: Sequence[str] | None = None,
    costs: CostModel | None = None,
    ctx: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Backtest the rule month by month over the ticks.

    Returns one row per (session, side, exit style, stop distance). **Both**
    sides are priced for every session at the same fill instant, so a direction
    rule is a filter on the result rather than a second simulation - which is
    what makes the controls exactly comparable. :func:`select_rule` applies one
    rule to a finished frame.
    """
    spec = get_spec(symbol)
    fractions = tuple(STOP_FRACTIONS if stop_fractions is None else stop_fractions)
    styles = tuple(EXIT_STYLES if styles is None else styles)
    if ctx is None:
        ctx = daily_context(symbol, cfg, split=split, start=start, end=end,
                            allow_test=allow_test)
    if ctx.is_empty():
        return pl.DataFrame()

    costs = costs or CostModel.from_profiles(symbol, split=split or "dev")
    slip = costs.slippage_pips(hour=None) * spec.pip
    latency_us = int(costs.slippage.latency_ms) * 1000

    frames: list[pl.DataFrame] = []
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
        del ticks

        # One month of rows at a time, folded into a frame before the next month
        # is read: a ~40-key dict per cell costs several KB where a polars row
        # costs a few hundred bytes, and this sweep has tens of thousands.
        records: list[dict] = []
        for row in day_rows.iter_rows(named=True):
            arm_us = int(row["sig_end_ts"].timestamp() * US) + latency_us
            flat_us = int(row["bell_ts"].timestamp() * US)
            f = tape.standing(arm_us)
            # The hold is [entry, bell): every tick strictly before 16:00, and
            # the last of them is where an unexited position is flattened. A
            # session whose tape stops early flattens at its own last quote,
            # which is what a market order at the bell would have done.
            end = min(tape.after(flat_us), tape.ts.size)
            if f < 0 or end <= f + 1:
                continue

            atr = float(row["atr"])
            mid_f = float((tape.bid[f] + tape.ask[f]) / 2)
            commission = costs.commission_pips(price=mid_f) * spec.pip
            base = {
                "symbol": symbol,
                "nyd": row["nyd"],
                "signal": int(row["signal"]),
                "coin": int(row["coin"]),
                "atr": atr,
                "ema": float(row["ema"]),
                "ema_gap_atr": float(row["ema_gap"]) / atr,
                "sig_range_atr": float(row["sig_high"] - row["sig_low"]) / atr,
                "entry_mid": mid_f,
                "entry_spread": float(tape.ask[f] - tape.bid[f]),
            }
            for direction in (1, -1):
                price = ((tape.ask[f] + slip) if direction == 1
                         else (tape.bid[f] - slip))
                fill = _Fill(i=f, direction=direction, price=float(price),
                             mid=mid_f, ts_us=int(tape.ts[f]))
                side = (base | _raw_stats(tape, fill, end, atr)
                        | {"direction": direction, "entry": fill.price})
                for frac in fractions:
                    distance = frac * atr
                    if distance <= 0:
                        continue
                    for style in styles:
                        rec = side | _simulate(tape, fill, end, distance, style, slip)
                        rec["stop_frac"] = float(frac)
                        rec["risk"] = float(distance)
                        rec["commission"] = commission
                        rec["net_pnl"] = rec["pnl"] - commission
                        rec["r_multiple"] = rec["pnl"] / distance
                        rec["net_r"] = rec["net_pnl"] / distance
                        rec["mid_r"] = rec["mid_pnl"] / distance
                        rec["cost_r"] = (rec["mid_pnl"] - rec["net_pnl"]) / distance
                        rec["net_bps"] = 1e4 * rec["net_pnl"] / mid_f
                        rec["mid_bps"] = 1e4 * rec["mid_pnl"] / mid_f
                        records.append(rec)
        del tape
        if records:
            frames.append(pl.DataFrame(records))
    return pl.concat(frames) if frames else pl.DataFrame()


def select_rule(trades: pl.DataFrame, rule: str) -> pl.DataFrame:
    """The subset of a finished run that one direction rule would have traded.

    ``ema`` and ``contra`` drop the sessions where the candle closed exactly on
    the EMA, which is the doji case and is not a side.
    """
    if trades.is_empty():
        return trades
    if rule == "ema":
        keep = pl.col("direction") == pl.col("signal")
    elif rule == "contra":
        keep = pl.col("direction") == -pl.col("signal")
    elif rule == "coin":
        keep = pl.col("direction") == pl.col("coin")
    elif rule == "long":
        keep = pl.col("direction") == 1
    elif rule == "short":
        keep = pl.col("direction") == -1
    else:
        raise ValueError(f"unknown direction rule {rule!r}")
    if rule in ("ema", "contra"):
        keep = keep & (pl.col("signal") != 0)
    return trades.filter(keep).with_columns(rule=pl.lit(rule))


def equity_curve(
    cell: pl.DataFrame, *, risk_pct: float = 1.0,
    starting_equity: float = 100_000.0,
) -> pl.DataFrame:
    """Compound ``risk_pct`` of equity per trade, in date order.

    This is the arithmetic behind the creator's "+982%": with the stop as the
    unit of risk a trade returns ``risk_pct * net_r`` percent of equity, and the
    curve is the running product. It is reported separately from the per-trade
    edge because it is a *sizing* statement - the same edge at 2% risk squares
    the multiple and roughly doubles the drawdown - so a total-return headline
    says as much about the position sizing as about the signal.
    """
    if cell.is_empty():
        return pl.DataFrame()
    return (
        cell.sort("nyd")
        .with_columns(step=1.0 + (risk_pct / 100.0) * pl.col("net_r"))
        .select("nyd", "net_r", "step",
                equity=starting_equity * pl.col("step").cum_prod())
        # The high-water mark starts at the opening balance, not at the first
        # trade's result. Seeding it with the curve alone would report a losing
        # first trade as a zero drawdown, which is the one number in the
        # creator's headline that a bug like this would flatter.
        .with_columns(peak=pl.max_horizontal(
            pl.lit(starting_equity), pl.col("equity").cum_max()))
        .with_columns(drawdown=pl.col("equity") / pl.col("peak") - 1.0)
    )
