"""The cost model every backtest consumes.

One object, built once per symbol, so that two strategies compared against each
other are compared under the same assumptions. A cost figure invented inside a
backtest is the easiest place in this whole pipeline to be accidentally
optimistic, and the hardest to notice afterwards.

Three components, in descending order of how well they are known:

**Commission** is published, so it is arithmetic: twice the per-side charge,
converted into pips through the contract size. On a JPY-quoted symbol the
conversion moves with the rate, so it is computed per bar rather than fixed.

**Spread** is observed. Every bar carries the spread that was actually quoted
during it, so the default takes it from the bar and there is nothing to model.
:mod:`qlab.costprofile` builds an hour-by-hour profile as well, for stress tests
and for reasoning about when to trade rather than what a fill cost.

**Slippage** is the one that has to be assumed, and the assumption is made
visible rather than picked. The measured quantity is how far the mid moves over
a latency horizon; what that costs depends on how a strategy's fills correlate
with the move, which no measurement of the tape can tell you. That correlation
is :attr:`SlippageModel.adverse_fraction`, and it is the single most
consequential number in this module:

    ``0.0``   fills uncorrelated with the move - the drift is noise that averages
              out, and slippage is zero in expectation. Defensible for a passive
              or mean-reverting entry.
    ``0.5``   the default. Half the movement is adverse.
    ``1.0``   the strategy chases: it buys because the price is rising, so the
              move during the latency window is systematically against it. This
              is the right setting for breakout and momentum entries.

Setting it to zero is a claim about the strategy, not a saving, and a backtest
that needs it to be zero is telling you something.

Why slippage cannot be skipped
------------------------------
A spread-plus-commission model sets it to zero silently. Measured against this
corpus, that understates the round turn by roughly 10% on EURUSD, a quarter on
gold, and it can approach the entire published cost on USTEC in a fast month.
The error is largest exactly where a strategy trades most.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from datetime import date

import polars as pl

from . import paths
from .costprofile import HORIZONS_MS
from .loader import SPLITS
from .symbols import SymbolSpec, get_spec


@dataclass(frozen=True)
class SlippageModel:
    """How measured latency drift becomes a cost.

    ``latency_ms`` must be one of the horizons the profile was measured at.
    Interpolating between them would look more precise than the measurement is:
    drift does not follow a clean square root of time at these horizons, because
    microstructure noise dominates the diffusive component.
    """

    latency_ms: int = 250
    adverse_fraction: float = 0.5
    extra_pips: float = 0.0
    impact_pips_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if self.latency_ms not in HORIZONS_MS:
            raise ValueError(
                f"latency_ms must be a measured horizon {HORIZONS_MS}, "
                f"got {self.latency_ms}; rebuild the profile to add one"
            )
        if not 0.0 <= self.adverse_fraction <= 1.0:
            raise ValueError(
                f"adverse_fraction is a share of the move, so it lies in [0, 1]; "
                f"got {self.adverse_fraction}"
            )

    def per_side_pips(self, drift_pips: float, lots: float = 1.0) -> float:
        return (
            self.adverse_fraction * drift_pips
            + self.extra_pips
            + self.impact_pips_per_lot * lots
        )


NO_SLIPPAGE = SlippageModel(adverse_fraction=0.0)
CHASING = SlippageModel(adverse_fraction=1.0)


def _weighted_mean(column: str, weight: str) -> pl.Expr:
    return (pl.col(column) * pl.col(weight)).sum() / pl.col(weight).sum()


def _window_filter(
    frame: pl.DataFrame,
    *,
    since: date | None,
    until: date | None,
    trailing_months: int | None,
) -> pl.DataFrame:
    stamp = pl.date(pl.col("year"), pl.col("month"), 1)
    if since is not None:
        frame = frame.filter(stamp >= pl.lit(since.replace(day=1)))
    if until is not None:
        frame = frame.filter(stamp <= pl.lit(until.replace(day=1)))
    if trailing_months is not None:
        keep = (
            frame.select("year", "month")
            .unique()
            .sort("year", "month")
            .tail(trailing_months)
        )
        frame = frame.join(keep, on=["year", "month"], how="semi")
    return frame


@dataclass(frozen=True)
class CostModel:
    """Spread, commission and slippage for one symbol, in one object.

    Build it with :meth:`from_profiles`. Apply it to bars with :meth:`with_costs`
    for a vectorized backtest, or read the scalar methods for an event-driven
    one; both go through the same arithmetic, so the two engines cannot drift
    apart.
    """

    spec: SymbolSpec
    slippage: SlippageModel
    spread_by_hour: pl.DataFrame
    drift_by_hour: pl.DataFrame
    spread_multiplier: float = 1.0
    commission_multiplier: float = 1.0
    label: str = "measured"
    window: tuple[date, date] | None = None
    """The period the profiles were measured over.

    Carried so that applying a model to bars from a different era is caught
    rather than silently costing 2020 trades at 2026 spreads - which on gold and
    USTEC is close to a factor of two.
    """

    # -- construction -------------------------------------------------------

    @classmethod
    def from_profiles(
        cls,
        symbol: str,
        *,
        slippage: SlippageModel | None = None,
        since: date | None = None,
        until: date | None = None,
        trailing_months: int | None = None,
        split: str | None = None,
        spread_multiplier: float = 1.0,
        commission_multiplier: float = 1.0,
        label: str = "measured",
    ) -> "CostModel":
        """Load the measured profiles and reduce them to per-hour tables.

        The window matters. Spread has compressed by an order of magnitude since
        2020, so a model fitted over the whole corpus is wrong at both ends of
        it. Pass ``split="dev"`` to cost a dev-period backtest with dev-period
        spreads, or ``trailing_months`` to describe the account as it is now.
        """
        spec = get_spec(symbol)
        slippage = slippage or SlippageModel()

        if split is not None:
            if since is not None or until is not None:
                raise ValueError("pass either split= or since=/until=, not both")
            resolved = SPLITS[split]
            since, until = resolved.start, resolved.end

        spread = pl.read_parquet(paths.SPREAD_PROFILE_PATH).filter(
            pl.col("symbol") == spec.name
        )
        latency = pl.read_parquet(paths.LATENCY_PROFILE_PATH).filter(
            (pl.col("symbol") == spec.name)
            & (pl.col("horizon_ms") == slippage.latency_ms)
        )
        if spread.is_empty() or latency.is_empty():
            raise FileNotFoundError(
                f"no cost profile for {spec.name}; run scripts/pipeline/build_cost_model.py"
            )

        spread = _window_filter(
            spread, since=since, until=until, trailing_months=trailing_months
        )
        latency = _window_filter(
            latency, since=since, until=until, trailing_months=trailing_months
        )

        spread_by_hour = (
            spread.group_by("hour", "is_sunday")
            .agg(
                n_ticks=pl.col("n_ticks").sum(),
                spread_mean_pips=_weighted_mean("spread_mean_pips", "n_ticks"),
                spread_p95_pips=_weighted_mean("spread_p95_pips", "n_bars"),
            )
            .sort("hour", "is_sunday")
        )
        drift_by_hour = (
            latency.group_by("hour")
            .agg(
                n_anchors=pl.col("n_anchors").sum(),
                drift_mean_pips=_weighted_mean("drift_mean_pips", "n_anchors"),
                drift_p95_pips=_weighted_mean("drift_p95_pips", "n_anchors"),
            )
            .sort("hour")
        )
        span = spread.select(
            lo_y=pl.col("year").min(), hi_y=pl.col("year").max()
        ).row(0, named=True)
        first = spread.filter(pl.col("year") == span["lo_y"])["month"].min()
        last = spread.filter(pl.col("year") == span["hi_y"])["month"].max()
        window = (
            date(span["lo_y"], first, 1),
            date(span["hi_y"] + last // 12, last % 12 + 1, 1),
        )
        return cls(
            spec=spec,
            slippage=slippage,
            spread_by_hour=spread_by_hour,
            drift_by_hour=drift_by_hour,
            spread_multiplier=spread_multiplier,
            commission_multiplier=commission_multiplier,
            label=label,
            window=window,
        )

    def stressed(
        self,
        *,
        spread_multiplier: float = 1.0,
        latency_ms: int | None = None,
        adverse_fraction: float | None = None,
        extra_slippage_pips: float = 0.0,
        label: str = "stressed",
    ) -> "CostModel":
        """A harsher copy of this model, for Stage 6.

        Changing the latency needs the profile reloaded, since a different
        horizon is a different measurement rather than a scaling of this one.
        """
        if latency_ms is not None and latency_ms != self.slippage.latency_ms:
            raise ValueError(
                "changing latency requires rebuilding from the profile: use "
                "CostModel.from_profiles(..., slippage=SlippageModel(latency_ms=...))"
            )
        slippage = replace(
            self.slippage,
            adverse_fraction=(
                adverse_fraction
                if adverse_fraction is not None
                else self.slippage.adverse_fraction
            ),
            extra_pips=self.slippage.extra_pips + extra_slippage_pips,
        )
        return replace(
            self,
            slippage=slippage,
            spread_multiplier=self.spread_multiplier * spread_multiplier,
            label=label,
        )

    # -- scalar interface ---------------------------------------------------

    def commission_pips(self, price: float | None = None) -> float:
        """Round-turn commission in pips.

        ``price`` is only consulted for a symbol quoted in something other than
        USD, where a pip's dollar value moves with the rate.
        """
        rate = 1.0
        if self.spec.quote_ccy != "USD":
            if price is None:
                raise ValueError(
                    f"{self.spec.name} is quoted in {self.spec.quote_ccy}, so the "
                    "commission in pips depends on the rate - pass price="
                )
            rate = price
        return self.commission_multiplier * self.spec.commission_pips(rate)

    def spread_pips(self, hour: int | None = None, *, is_sunday: bool = False) -> float:
        """Profiled spread, over the whole window or for one UTC hour."""
        frame = self.spread_by_hour.filter(pl.col("is_sunday") == is_sunday)
        if hour is not None:
            frame = frame.filter(pl.col("hour") == hour)
        if frame.is_empty():
            return float("nan")
        value = frame.select(_weighted_mean("spread_mean_pips", "n_ticks")).item()
        return self.spread_multiplier * value

    def drift_pips(self, hour: int | None = None) -> float:
        """Measured absolute mid drift over the latency horizon."""
        frame = self.drift_by_hour
        if hour is not None:
            frame = frame.filter(pl.col("hour") == hour)
        if frame.is_empty():
            return float("nan")
        return frame.select(_weighted_mean("drift_mean_pips", "n_anchors")).item()

    def slippage_pips(self, hour: int | None = None, *, lots: float = 1.0) -> float:
        """Per-side slippage."""
        return self.slippage.per_side_pips(self.drift_pips(hour), lots)

    def round_turn_pips(
        self,
        hour: int | None = None,
        *,
        price: float | None = None,
        is_sunday: bool = False,
        lots: float = 1.0,
    ) -> float:
        """Full cost of entering and exiting one position, in pips.

        One round turn crosses the spread once - in at the ask, out at the bid -
        pays commission on both sides, and takes slippage on both fills.
        """
        return (
            self.spread_pips(hour, is_sunday=is_sunday)
            + self.commission_pips(price)
            + 2.0 * self.slippage_pips(hour, lots=lots)
        )

    def round_turn_bps(self, price: float, **kwargs) -> float:
        """The same cost as a share of price - the only cross-instrument unit.

        This is also the minimum edge: a strategy must find more than this per
        round turn before it has made anything at all.
        """
        return self.round_turn_pips(price=price, **kwargs) * self.spec.pip / price * 10_000

    def round_turn_usd(self, price: float, *, lots: float = 1.0, **kwargs) -> float:
        rate = price if self.spec.quote_ccy != "USD" else 1.0
        pips = self.round_turn_pips(price=price, lots=lots, **kwargs)
        return pips * self.spec.pip_value_usd(rate) * lots

    def breakdown(self, price: float, *, hour: int | None = None, lots: float = 1.0) -> dict:
        """Every component at once, in pips, USD and bps - for reports."""
        spread = self.spread_pips(hour)
        commission = self.commission_pips(price)
        slippage = 2.0 * self.slippage_pips(hour, lots=lots)
        total = spread + commission + slippage
        rate = price if self.spec.quote_ccy != "USD" else 1.0
        pip_usd = self.spec.pip_value_usd(rate)
        return {
            "symbol": self.spec.name,
            "label": self.label,
            "reference_price": price,
            "spread_pips": spread,
            "commission_pips": commission,
            "slippage_pips": slippage,
            "round_turn_pips": total,
            "round_turn_usd": total * pip_usd * lots,
            "round_turn_bps": total * self.spec.pip / price * 10_000,
            "spread_share": spread / total if total else float("nan"),
            "commission_share": commission / total if total else float("nan"),
            "slippage_share": slippage / total if total else float("nan"),
        }

    def hourly_table(self, price: float, *, lots: float = 1.0) -> pl.DataFrame:
        """Cost for each UTC hour, with closed hours marked rather than dropped.

        An hour inside an instrument's daily maintenance break has no quotes and
        so no cost - which is not the same as a cheap hour, and must not be
        allowed to look like one when something takes a minimum.
        """
        rows = []
        for hour in range(24):
            spread = self.spread_pips(hour)
            drift = self.drift_pips(hour)
            open_ = not (spread != spread or drift != drift)  # NaN check
            rows.append(
                {
                    "hour": hour,
                    "is_open": open_,
                    "spread_pips": spread,
                    "drift_pips": drift,
                    "slippage_pips": self.slippage.per_side_pips(drift, lots) if open_ else float("nan"),
                    "round_turn_pips": self.round_turn_pips(hour, price=price, lots=lots)
                    if open_
                    else float("nan"),
                }
            )
        return pl.DataFrame(rows).with_columns(
            round_turn_bps=pl.col("round_turn_pips") * self.spec.pip / price * 10_000
        )

    def annual_drag_bps(self, price: float, *, round_turns_per_day: float) -> float:
        """Cost as annualised basis points, at a given trading frequency.

        Turnover is what converts a small per-trade cost into a large one: at one
        round turn a day a half-basis-point cost is about 1.3% a year, and at ten
        it is 13%. This is the number to compare against a strategy's gross
        return before believing it.
        """
        return self.round_turn_bps(price) * round_turns_per_day * 252

    # -- vectorized interface -----------------------------------------------

    def with_costs(
        self, bars: pl.DataFrame, *, lots: float = 1.0, spread_source: str = "realized"
    ) -> pl.DataFrame:
        """Add per-bar cost columns to a bar frame.

        ``spread_source="realized"`` uses the spread each bar actually quoted,
        which is both the most accurate answer and free of any lookahead - the
        quote was there at the time. ``"profile"`` substitutes the hourly profile
        instead, which is what a stress test wants: it answers "what would this
        have cost in a typical hour like this one" rather than "what did it cost".

        Adds ``spread_pips``, ``slippage_pips`` (per side), ``commission_pips``
        (round turn), ``round_turn_pips``, ``side_cost_pips``, ``round_turn_bps``
        and ``round_turn_usd``. ``side_cost_pips`` is the composable one: a
        vectorized backtest multiplies it by ``|position change|``.
        """
        if spread_source not in ("realized", "profile"):
            raise ValueError(f"spread_source must be realized/profile, got {spread_source!r}")
        self._check_window(bars)

        # Keyed off the interval open, not the bar label: a bar covering
        # [20:59, 21:00) is labelled 21:00, and pricing it at the rollover hour
        # would charge it for a window none of its ticks were in.
        stamp = pl.col("ts_open") if "ts_open" in bars.columns else pl.col("ts")
        frame = bars.with_columns(_hour=stamp.dt.hour(), _is_sunday=stamp.dt.weekday() == 7)

        if spread_source == "realized":
            frame = frame.with_columns(
                spread_pips=pl.col("spread_mean") / self.spec.pip * self.spread_multiplier
            )
        else:
            frame = frame.join(
                self.spread_by_hour.select(
                    "hour", "is_sunday", pl.col("spread_mean_pips").alias("_profiled")
                ),
                left_on=["_hour", "_is_sunday"],
                right_on=["hour", "is_sunday"],
                how="left",
            ).with_columns(spread_pips=pl.col("_profiled") * self.spread_multiplier)

        frame = frame.join(
            self.drift_by_hour.select("hour", "drift_mean_pips"),
            left_on="_hour",
            right_on="hour",
            how="left",
        ).with_columns(
            slippage_pips=(
                self.slippage.adverse_fraction * pl.col("drift_mean_pips")
                + self.slippage.extra_pips
                + self.slippage.impact_pips_per_lot * lots
            )
        )

        rate = pl.col("close") if self.spec.quote_ccy != "USD" else pl.lit(1.0)
        pip_value = self.spec.contract_size * self.spec.pip / rate
        commission = (
            self.commission_multiplier
            * 2.0
            * self.spec.commission_per_lot_side_usd
            / pip_value
        )

        return (
            frame.with_columns(commission_pips=commission)
            .with_columns(
                round_turn_pips=pl.col("spread_pips")
                + pl.col("commission_pips")
                + 2.0 * pl.col("slippage_pips")
            )
            .with_columns(
                side_cost_pips=pl.col("round_turn_pips") / 2.0,
                round_turn_bps=pl.col("round_turn_pips") * self.spec.pip / pl.col("close") * 10_000,
                round_turn_usd=pl.col("round_turn_pips") * pip_value * lots,
            )
            .drop("_hour", "_is_sunday", "drift_mean_pips", strict=False)
            .drop("_profiled", strict=False)
        )

    def _check_window(self, bars: pl.DataFrame) -> None:
        """Refuse a model measured on a period the bars do not touch.

        Costing dev-split bars with a trailing-3-month model is the mistake this
        exists to catch: on gold and USTEC the two windows differ by nearly a
        factor of two in bps, and nothing about the output would look wrong.
        """
        if self.window is None or bars.is_empty():
            return
        stamp = "ts_open" if "ts_open" in bars.columns else "ts"
        lo, hi = bars[stamp].min().date(), bars[stamp].max().date()
        start, end = self.window
        if hi < start or lo >= end:
            raise ValueError(
                f"{self.spec.name} cost model was measured over {start} to {end} "
                f"but these bars run {lo} to {hi}; rebuild with "
                "CostModel.from_profiles(..., split=/since=/until=) for this period"
            )
        if lo < start or hi >= end:
            warnings.warn(
                f"{self.spec.name} bars ({lo} to {hi}) extend outside the cost "
                f"model's window ({start} to {end}); costs outside it are "
                "extrapolated from a different period",
                stacklevel=3,
            )

    def describe(self) -> str:
        """One line stating every assumption, for a report header or a log."""
        return (
            f"{self.spec.name} cost model [{self.label}]: "
            f"spread x{self.spread_multiplier:g} (measured), "
            f"commission x{self.commission_multiplier:g} "
            f"(${self.spec.commission_per_lot_side_usd}/lot/side), "
            f"slippage = {self.slippage.adverse_fraction:g} x drift over "
            f"{self.slippage.latency_ms} ms"
            + (
                f", measured {self.window[0]} to {self.window[1]}"
                if self.window
                else ""
            )
            + (f" + {self.slippage.extra_pips:g} pips" if self.slippage.extra_pips else "")
        )
