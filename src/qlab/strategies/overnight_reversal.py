"""The overnight-intraday reversal family, on a four-instrument cross-section.

The claim being tested
----------------------
Liu, Liu, Wang, Zhou and Zhu (2025) decompose the daily return into an
*overnight* leg (previous close to today's open) and an *intraday* leg (today's
open to today's close), and report that the traditional close-to-close reversal
strategy is dominated by a single component: sort on **yesterday's overnight
return**, hold **today's intraday return**, contrarian. They call it CO-OC and
find it profitable in equity index, rate, commodity and currency futures alike.

The mechanism they argue for is that the overnight session is thin, so a large
overnight move is disproportionately price pressure rather than information,
and the liquid daytime session takes it back. Their headline conditioning
variable is the *cross-sectional dispersion* of overnight returns - not the VIX,
not sentiment, not announcement days.

What this module can and cannot test
------------------------------------
This corpus is four spot CFDs - EURUSD, USDJPY, XAUUSD, USTEC - not a broad
futures panel, and that constrains the test in three ways that the report states
before any number is read:

**Breadth.** N = 4, against the paper's dozens per asset class. A cross-sectional
demeaned portfolio of four assets is a legitimate zero-investment portfolio, but
its diversification is nearly nil and its standard errors are correspondingly
wide. Nothing here can distinguish "the effect is absent" from "the effect is
present and this cross-section is too narrow to see it"; the report says which
of those the numbers support.

**No exchange close.** These instruments quote almost continuously, so there is
no literal overnight *gap* to trade - on EURUSD the close-to-open return across
a nominal boundary is a tick or two. The paper's futures have the same property
(the CME trades nearly 23 hours), and the resolution is the same one the
exchanges use: **the session is the regular trading hours of the matching
futures contract**, and everything outside it is "overnight". :data:`RTH` maps
each instrument to the pit session of the contract the paper would have traded -
NQ, GC, 6E, 6J - which is what makes "overnight" mean the thin hours rather
than a zero-length instant.

**No external data.** VIX, Baker-Wurgler sentiment, the Fama-French library, the
Asness value/momentum factors and an announcement calendar are all outside this
project, which holds quotes and nothing else. So the paper's risk-adjustment
regressions (their Table 4) and the global two-factor pricing test (Table 5)
are not implementable here and are not faked. The dispersion regressions, which
are the ones that carry the mechanism, need only the panel's own returns and are
implemented in full; :func:`realized_vol_proxy` supplies a VIX *proxy* built
from the index's own realised volatility, labelled as a proxy everywhere it is
used.

Scale, and why the weighting scheme is not a detail
---------------------------------------------------
The paper's construction ``w_i = -(1/N)(s_i - mean(s))`` is applied within an
asset class, where the assets have comparable volatility. This cross-section
mixes gold and USTEC with EURUSD, whose daily volatility is smaller by a factor
of five or more. Under the literal demeaned construction the portfolio is
therefore almost entirely a bet on whichever of gold and USTEC moved most
overnight, and EURUSD carries a weight indistinguishable from zero.

Three weightings are therefore implemented and all three are reported:

``demean``
    The paper's baseline, transcribed literally. Kept because it is what the
    paper says, not because it is the right thing for this universe.
``rank``
    The paper's own robustness variant (their Table A.1). Scale-free, so it is
    the honest primary specification for a mixed cross-section.
``vol_scaled``
    Demeaned after dividing each asset's signal by its own trailing volatility.
    Not in the paper; it is what the paper's within-class construction is
    implicitly doing, made explicit across classes.

Costs
-----
CO-OC and OC-OC are flat overnight, so they pay a **full round turn every day**
on the whole gross position. CC-CC and OO-OO hold continuously and pay only for
the change in weight - one side of a round turn per unit traded. That asymmetry
is not a modelling nicety: it is most of the answer, and it is why the paper's
comparison of the four variants looks different once execution is charged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl

from ..costs import CostModel
from ..loader import load_bars
from ..metrics import tearsheet_from_returns
from ..stats import newey_west, ols_hac
from ..symbols import get_spec

NY = "America/New_York"


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionWindow:
    """The regular trading hours of one instrument, in New York local minutes.

    New York local rather than UTC because a session does not move with the UTC
    clock even though its UTC hour does; keying off UTC would file half the year
    in the wrong window.
    """

    symbol: str
    open_min: int
    close_min: int
    min_bars: int
    contract: str
    """The futures contract whose pit session this is - the paper trades
    futures, so the session convention is transcribed from the contract rather
    than invented for the CFD."""

    def __post_init__(self) -> None:
        if not 0 <= self.open_min < self.close_min <= 24 * 60:
            raise ValueError(f"{self.symbol}: bad window "
                             f"{self.open_min}-{self.close_min}")


RTH: dict[str, SessionWindow] = {
    # Nasdaq-100: the cash session, which is also NQ's regular session.
    "USTEC": SessionWindow("USTEC", 9 * 60 + 30, 16 * 60, 300, "NQ"),
    # COMEX gold: 08:20-13:30 New York, the old floor session.
    "XAUUSD": SessionWindow("XAUUSD", 8 * 60 + 20, 13 * 60 + 30, 250, "GC"),
    # CME FX: 07:20-14:00 Chicago is 08:20-15:00 New York.
    "EURUSD": SessionWindow("EURUSD", 8 * 60 + 20, 15 * 60, 320, "6E"),
    "USDJPY": SessionWindow("USDJPY", 8 * 60 + 20, 15 * 60, 320, "6J"),
}

DEFAULT_UNIVERSE: tuple[str, ...] = ("EURUSD", "USDJPY", "XAUUSD", "USTEC")

OPEN_TOLERANCE_MIN = 15
"""How far past the nominal open the first quote may be and still count as the
open. A holiday half-session or a feed outage that misses the whole window is
dropped by ``min_bars`` instead; this only forgives a few quiet minutes."""


def session_panel(
    symbols: tuple[str, ...] | list[str] = DEFAULT_UNIVERSE,
    *,
    split: str | None = "dev",
    start=None,
    end=None,
    warmup: timedelta | None = None,
    allow_test: bool = False,
    windows: dict[str, SessionWindow] | None = None,
) -> pl.DataFrame:
    """One row per instrument per session: its open, its close, and the count.

    Long format rather than wide, because the number of instruments quoting on a
    given day varies with holidays and the cross-section has to be able to see
    that rather than silently carrying a stale price forward.
    """
    windows = RTH if windows is None else windows
    frames = []
    for symbol in symbols:
        window = windows[symbol]
        bars = load_bars(
            symbol, "1m", split=split, start=start, end=end, warmup=warmup,
            allow_test=allow_test,
            columns=["ts", "ts_open", "open", "close"],
        )
        bars = (
            bars.with_columns(ny=pl.col("ts_open").dt.convert_time_zone(NY))
            .with_columns(
                nyd=pl.col("ny").dt.date(),
                mins=pl.col("ny").dt.hour().cast(pl.Int32) * 60
                     + pl.col("ny").dt.minute().cast(pl.Int32),
            )
            .filter((pl.col("mins") >= window.open_min)
                    & (pl.col("mins") < window.close_min))
            .sort("ny")
        )
        agg = [
            pl.col("open").first().alias("o"),
            pl.col("close").last().alias("c"),
            pl.col("mins").first().alias("first_min"),
            pl.col("ts").first().alias("open_ts"),
            pl.len().alias("n_bars"),
        ]
        if warmup is not None:
            agg.append(pl.col("is_warmup").any().alias("is_warmup"))
        day = (
            bars.group_by("nyd").agg(agg)
            .filter(
                (pl.col("n_bars") >= window.min_bars)
                & (pl.col("first_min") <= window.open_min + OPEN_TOLERANCE_MIN)
            )
            .with_columns(
                symbol=pl.lit(symbol),
                open_hour_utc=pl.col("open_ts").dt.hour().cast(pl.Int32),
            )
            .sort("nyd")
        )
        frames.append(day)
    return pl.concat(frames, how="diagonal").sort(["nyd", "symbol"])


def with_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the four return definitions of A.1, per instrument.

    Shifts are taken *within* an instrument on its own sorted session sequence,
    so a holiday that closes one market and not another never lines a price up
    against the wrong previous close.
    """
    prev_c = pl.col("c").shift(1).over("symbol")
    prev_o = pl.col("o").shift(1).over("symbol")
    return panel.sort(["symbol", "nyd"]).with_columns(
        r_cc=pl.col("c") / prev_c - 1.0,
        r_co=pl.col("o") / prev_c - 1.0,
        r_oc=pl.col("c") / pl.col("o") - 1.0,
        r_oo=pl.col("o") / prev_o - 1.0,
        prev_gap_days=(pl.col("nyd") - pl.col("nyd").shift(1).over("symbol"))
                      .dt.total_days(),
    ).sort(["nyd", "symbol"])


RETURN_NAMES: tuple[str, ...] = ("r_cc", "r_co", "r_oc", "r_oo")


# --------------------------------------------------------------------------
# The cross-section
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossSection:
    """The panel as aligned matrices: dates down, instruments across."""

    dates: np.ndarray                    # (T,) of datetime.date
    symbols: tuple[str, ...]             # (N,)
    returns: dict[str, np.ndarray]       # each (T, N)
    price_open: np.ndarray               # (T, N) - for the cost model
    open_hour_utc: np.ndarray            # (N,) modal UTC hour of the session open

    @property
    def n_days(self) -> int:
        return self.dates.size

    @property
    def n_assets(self) -> int:
        return len(self.symbols)


def cross_section(
    panel: pl.DataFrame,
    *,
    symbols: tuple[str, ...] | list[str] | None = None,
    drop_warmup: bool = True,
) -> CrossSection:
    """Align the long panel into matrices, keeping only complete days.

    A day on which any instrument did not quote a full session is dropped
    outright rather than filled. The alternative - carrying a price forward -
    manufactures a zero return for the missing asset, which the demeaning then
    turns into a *positive* contrarian weight on an instrument that was shut.
    """
    frame = panel
    if drop_warmup and "is_warmup" in frame.columns:
        frame = frame.filter(~pl.col("is_warmup").fill_null(False))
    if symbols is None:
        symbols = tuple(sorted(frame["symbol"].unique().to_list()))
    symbols = tuple(symbols)

    frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
    needed = [c for c in RETURN_NAMES if c in frame.columns]
    frame = frame.drop_nulls(needed)

    complete = (
        frame.group_by("nyd").agg(pl.col("symbol").n_unique().alias("n"))
        .filter(pl.col("n") == len(symbols))
        .select("nyd")
    )
    frame = frame.join(complete, on="nyd", how="inner").sort(["nyd", "symbol"])
    dates = np.array(frame["nyd"].unique(maintain_order=True).to_list())

    def matrix(column: str) -> np.ndarray:
        wide = (
            frame.select("nyd", "symbol", column)
            .pivot(on="symbol", index="nyd", values=column)
            .sort("nyd")
        )
        return wide.select(list(symbols)).to_numpy()

    returns = {name: matrix(name) for name in needed}
    hours = np.array([
        int(frame.filter(pl.col("symbol") == s)["open_hour_utc"].mode()[0])
        for s in symbols
    ])
    return CrossSection(dates, symbols, returns, matrix("o"), hours)


# --------------------------------------------------------------------------
# Portfolio construction (A.2)
# --------------------------------------------------------------------------

def _rank_rows(signal: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks, 1 (lowest) to N, ties averaged."""
    order = signal.argsort(axis=1, kind="stable")
    ranks = np.empty_like(signal, dtype=float)
    rows = np.arange(signal.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, signal.shape[1] + 1, dtype=float)
    return ranks


def build_weights(
    signal: np.ndarray,
    *,
    weighting: str = "demean",
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """Zero-investment contrarian weights from a formation signal.

    Negative by construction: the lowest signal is bought, the highest sold.
    Every scheme returns weights that sum to zero row by row.
    """
    if weighting == "rank":
        ranks = _rank_rows(signal)
        centred = ranks - ranks.mean(axis=1, keepdims=True)
        return -centred / signal.shape[1]

    if weighting == "vol_scaled":
        if vol is None:
            raise ValueError("vol_scaled weighting needs a volatility matrix")
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(vol > 0, signal / vol, np.nan)
        # A missing volatility means no view, not a zero signal: neutralise the
        # asset by giving it the cross-sectional mean. A row with no usable
        # volatility at all - the warmup - neutralises entirely, to zero weight.
        with np.errstate(invalid="ignore"):
            counts = np.isfinite(scaled).sum(axis=1, keepdims=True)
            row_mean = np.where(
                counts > 0,
                np.nansum(np.where(np.isfinite(scaled), scaled, 0.0), axis=1,
                          keepdims=True) / np.maximum(counts, 1),
                0.0,
            )
        scaled = np.where(np.isfinite(scaled), scaled, row_mean)
        return -(scaled - scaled.mean(axis=1, keepdims=True)) / signal.shape[1]

    if weighting == "demean":
        return -(signal - signal.mean(axis=1, keepdims=True)) / signal.shape[1]

    raise ValueError(f"unknown weighting {weighting!r}")


def trailing_vol(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Standard deviation of each column over the previous ``window`` rows.

    Strictly previous: row *t* uses rows ``[t-window, t)``, so a weight formed
    from it never sees the return it is about to be applied to.
    """
    T, N = returns.shape
    out = np.full((T, N), np.nan)
    for t in range(window, T):
        block = returns[t - window:t]
        out[t] = np.nanstd(block, axis=0, ddof=1)
    return out


# --------------------------------------------------------------------------
# Variants (A.3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """One row of the paper's Table 1: what is sorted on, and what is held."""

    name: str
    signal: str
    holding: str
    exposure: str
    """``intraday`` - flat outside the session, so a full round turn every day.
    ``continuous`` - the position persists, so only the change in weight trades.
    """
    note: str = ""


VARIANTS: dict[str, Variant] = {
    "CC-CC": Variant("CC-CC", "r_cc", "r_cc", "continuous",
                     "the traditional short-horizon reversal"),
    "OO-OO": Variant("OO-OO", "r_oo", "r_oo", "continuous", ""),
    "OC-OC": Variant("OC-OC", "r_oc", "r_oc", "intraday", ""),
    "CO-OC": Variant("CO-OC", "r_co", "r_oc", "intraday",
                     "the paper's primary strategy"),
    # Appendix only in the paper, and for a good reason: both hold the overnight
    # window, which on a real book means executing into the thinnest hours of
    # the day. Implemented so the report can show what they are worth, never
    # recommended.
    "CO-CO": Variant("CO-CO", "r_co", "r_co", "overnight",
                     "appendix only - requires illiquid overnight execution"),
    "OC-CO": Variant("OC-CO", "r_oc", "r_co", "overnight",
                     "appendix only - requires illiquid overnight execution"),
}

CORE_VARIANTS: tuple[str, ...] = ("CC-CC", "OO-OO", "OC-OC", "CO-OC")


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------

def round_turn_bps_matrix(
    cs: CrossSection,
    *,
    split: str | None = "dev",
    models: dict[str, CostModel] | None = None,
    stress: dict | None = None,
) -> np.ndarray:
    """``(T, N)`` round-turn cost in basis points of price, at the session open.

    Priced at each day's own opening price rather than a sample average, because
    gold more than doubled across this corpus and a fixed pip cost is therefore
    a halving cost in bps.
    """
    out = np.empty_like(cs.price_open, dtype=float)
    for j, symbol in enumerate(cs.symbols):
        spec = get_spec(symbol)
        model = (models or {}).get(symbol)
        if model is None:
            model = CostModel.from_profiles(symbol, split=split)
        if stress:
            model = model.stressed(**stress)
        hour = int(cs.open_hour_utc[j])
        # Spread and slippage do not depend on price; commission does, but only
        # for a symbol quoted in something other than USD.
        base_pips = model.spread_pips(hour) + 2.0 * model.slippage_pips(hour)
        price = cs.price_open[:, j]
        if spec.quote_ccy == "USD":
            pips = base_pips + model.commission_pips(None)
            out[:, j] = pips * spec.pip / price * 10_000
        else:
            comm = np.array([model.commission_pips(float(p)) for p in price])
            out[:, j] = (base_pips + comm) * spec.pip / price * 10_000
    return out


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """Everything one variant produced, day by day, so nothing is re-simulated."""

    variant: str
    weighting: str
    dates: np.ndarray
    weights: np.ndarray          # (T, N), formed at t-1, applied to day t
    gross_bps: np.ndarray        # (T,) before cost
    net_bps: np.ndarray          # (T,) after cost
    cost_bps: np.ndarray         # (T,)
    long_leg_bps: np.ndarray     # (T,)
    short_leg_bps: np.ndarray    # (T,)
    turnover: np.ndarray         # (T,) dollars traded per dollar of capital
    symbols: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    def tearsheet(self, *, net: bool = True, label: str | None = None) -> dict:
        series = self.net_bps if net else self.gross_bps
        name = label or f"{self.variant} {self.weighting} {'net' if net else 'gross'}"
        return tearsheet_from_returns(series, label=name, turnover=self.turnover)

    def hac(self, *, net: bool = True):
        return newey_west(self.net_bps if net else self.gross_bps)


WINSOR_Q = (0.01, 0.99)


def _winsorize(x: np.ndarray, q=WINSOR_Q) -> np.ndarray:
    good = x[np.isfinite(x)]
    if good.size < 20:
        return x
    lo, hi = np.quantile(good, q)
    return np.clip(x, lo, hi)


def run_variant(
    cs: CrossSection,
    variant: str | Variant,
    *,
    weighting: str = "demean",
    cost_bps: np.ndarray | None = None,
    winsorize: bool = True,
    vol_window: int = 60,
    shuffle_seed: int | None = None,
) -> RunResult:
    """Form the portfolio on day t-1 and hold it through day t.

    ``shuffle_seed`` is the placebo: the formation signals are permuted *across
    instruments* within each day, which destroys the cross-sectional information
    while leaving every marginal distribution, the weighting, the turnover and
    the cost untouched. A placebo that still makes money is measuring the
    construction, not the signal.
    """
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    sig_all = cs.returns[v.signal]
    hold_all = cs.returns[v.holding]

    # Formation at t-1, holding at t. The shift is done once, here, so no
    # caller can accidentally line a signal up with its own holding return.
    signal = sig_all[:-1]
    hold = hold_all[1:]
    dates = cs.dates[1:]
    cost = None if cost_bps is None else cost_bps[1:]

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        signal = np.array([rng.permutation(row) for row in signal])

    vol = None
    if weighting == "vol_scaled":
        vol = trailing_vol(sig_all, vol_window)[:-1]

    w = build_weights(signal, weighting=weighting, vol=vol)

    gross_pnl = np.nansum(w * hold, axis=1)
    dollars = np.nansum(np.abs(w), axis=1)
    capital = dollars / 2.0                      # 50% margin on $1 of capital
    with np.errstate(divide="ignore", invalid="ignore"):
        gross_bps = np.where(capital > 0, gross_pnl / capital, np.nan) * 1e4

    # Turnover, and therefore cost, depends on whether the position persists.
    if v.exposure == "continuous":
        prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        traded = np.abs(w - prev)
        sides = 0.5      # one side of a round turn per unit traded
    else:
        traded = np.abs(w)
        sides = 1.0      # in at the open, out at the close, every day
    if cost is None:
        cost_pnl = np.zeros_like(gross_pnl)
    else:
        cost_pnl = np.nansum(traded * cost * sides, axis=1) / 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_series = np.where(capital > 0, cost_pnl / capital, np.nan) * 1e4
        turnover = np.where(capital > 0, traded.sum(axis=1) / capital, np.nan)
    net_bps = gross_bps - cost_series

    # A.6: what each side of the book contributed, per dollar committed to it.
    long_mask = w > 0
    short_mask = w < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        long_w = np.where(long_mask, w, 0.0).sum(axis=1)
        short_w = np.where(short_mask, np.abs(w), 0.0).sum(axis=1)
        long_leg = np.where(long_w > 0,
                            np.nansum(np.where(long_mask, w * hold, 0.0), axis=1)
                            / np.where(long_w > 0, long_w, np.nan), np.nan) * 1e4
        short_leg = np.where(short_w > 0,
                             np.nansum(np.where(short_mask, w * hold, 0.0), axis=1)
                             / np.where(short_w > 0, short_w, np.nan), np.nan) * 1e4

    if winsorize:
        gross_bps = _winsorize(gross_bps)
        net_bps = _winsorize(net_bps)

    return RunResult(
        variant=v.name, weighting=weighting, dates=dates, weights=w,
        gross_bps=gross_bps, net_bps=net_bps, cost_bps=cost_series,
        long_leg_bps=long_leg, short_leg_bps=short_leg, turnover=turnover,
        symbols=cs.symbols,
        meta={"exposure": v.exposure, "winsorized": winsorize,
              "placebo": shuffle_seed is not None},
    )


# --------------------------------------------------------------------------
# Weekly frequency (A.4)
# --------------------------------------------------------------------------

def weekly_cross_section(cs: CrossSection) -> CrossSection:
    """Aggregate to the paper's weekly convention.

    Intraday is Monday's open to Friday's close; overnight is the previous
    Friday's close to Monday's open. Weeks that do not contain both ends are
    dropped rather than approximated with whatever days did quote.
    """
    dates = cs.dates
    iso = np.array([(d.isocalendar()[0], d.isocalendar()[1]) for d in dates])
    keys, first = np.unique(iso, axis=0, return_index=True)
    order = np.argsort(first)
    keys, first = keys[order], first[order]
    last = np.array([np.max(np.flatnonzero((iso == k).all(axis=1))) for k in keys])

    # A week is usable only if its first session is a Monday-ish start and its
    # last is a Friday-ish end; requiring at least three sessions removes the
    # holiday stubs where "the week" is one day.
    counts = np.array([int(((iso == k).all(axis=1)).sum()) for k in keys])
    ok = counts >= 3
    first, last, keys = first[ok], last[ok], keys[ok]

    open_w = cs.price_open[first]
    # Reconstruct the weekly closes from the daily intraday returns.
    close_daily = cs.price_open * (1.0 + cs.returns["r_oc"])
    close_w = close_daily[last]

    prev_close = np.vstack([np.full((1, close_w.shape[1]), np.nan), close_w[:-1]])
    prev_open = np.vstack([np.full((1, open_w.shape[1]), np.nan), open_w[:-1]])
    returns = {
        "r_oc": close_w / open_w - 1.0,
        "r_co": open_w / prev_close - 1.0,
        "r_cc": close_w / prev_close - 1.0,
        "r_oo": open_w / prev_open - 1.0,
    }
    return CrossSection(dates[first], cs.symbols, returns, open_w, cs.open_hour_utc)


# --------------------------------------------------------------------------
# Mechanism: dispersion (A.9)
# --------------------------------------------------------------------------

def overnight_dispersion(cs: CrossSection, *, scaled: bool = True,
                         vol_window: int = 60) -> np.ndarray:
    """Cross-sectional dispersion of the overnight return, day by day.

    ``scaled=True`` divides each instrument's overnight return by its own
    trailing volatility first. On a single-asset-class panel that would be a
    refinement; on this one it is a necessity, because gold's overnight
    standard deviation is several times EURUSD's and an unscaled dispersion is
    therefore mostly a restatement of how far gold moved.
    """
    r = cs.returns["r_co"]
    if not scaled:
        return np.nanstd(r, axis=1, ddof=1)
    vol = trailing_vol(r, vol_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(vol > 0, r / vol, np.nan)
    return np.nanstd(z, axis=1, ddof=1)


def realized_vol_proxy(cs: CrossSection, symbol: str = "USTEC",
                       window: int = 20) -> np.ndarray:
    """A VIX stand-in: trailing realised volatility of the index leg, annualised.

    Explicitly a proxy. The VIX is an option-implied forward-looking measure and
    this is a backward-looking realised one; they correlate strongly but they
    are not the same variable, and every result that uses this says so.
    """
    j = cs.symbols.index(symbol)
    r = cs.returns["r_cc"][:, j]
    out = np.full(r.size, np.nan)
    for t in range(window, r.size):
        out[t] = np.nanstd(r[t - window:t], ddof=1) * np.sqrt(252) * 100
    return out


def dispersion_regression(
    run: RunResult,
    dispersion: np.ndarray,
    *,
    vix_proxy: np.ndarray | None = None,
    net: bool = True,
    lags: int | None = None,
):
    """A.9(a): does yesterday's dispersion predict today's CO-OC return?

    ``dispersion`` and ``vix_proxy`` are passed on the *daily* grid of the
    cross-section, and lagged here - once, in one place - so that the regressor
    is always information that existed before the return it is explaining.
    """
    y = run.net_bps if net else run.gross_bps
    disp = dispersion[1:][:-1]           # align to the run's dates, then lag
    lag_y = y[:-1]
    y = y[1:]
    columns = [disp, lag_y]
    names = ["dispersion[t-1]", "self[t-1]"]
    if vix_proxy is not None:
        columns.insert(1, vix_proxy[1:][:-1])
        names.insert(1, "vol_proxy[t-1]")
    return ols_hac(y, np.column_stack(columns), names=names, lags=lags)


def conditional_sharpe_regression(
    run: RunResult,
    dispersion: np.ndarray,
    *,
    net: bool = True,
    window: int = 63,
    lags: int | None = None,
):
    """A.9(b): does dispersion predict the *risk-adjusted* return?

    Two steps, as in the paper. First a conditional volatility is fitted from
    the first-stage residuals - the ``kappa`` power transform is theirs, and it
    is what makes the second stage a Sharpe rather than a scaled return. Then
    the scaled return is regressed on dispersion again. If only ``a1 > 0``
    survives the first stage and ``b1`` does not, dispersion is predicting
    volatility, not edge.
    """
    first = dispersion_regression(run, dispersion, net=net, lags=lags)
    resid = first.resid
    denom = float(np.abs(resid).mean())
    if denom <= 0:
        raise ValueError("degenerate residuals")
    kappa = float(np.sqrt((resid ** 2).mean())) / denom

    y = run.net_bps if net else run.gross_bps
    disp = dispersion[1:][:-1]
    lag_y = y[:-1]
    y = y[1:]
    # The first stage dropped whatever rows were not finite; ``keep`` says which,
    # so the residual series can be lined back up with its own regressors rather
    # than trimmed from one end and hoped for.
    keep = first.keep
    y, disp, lag_y = y[keep], disp[keep], lag_y[keep]

    second = ols_hac(np.abs(resid) ** kappa, np.column_stack([disp, lag_y]),
                     names=["dispersion[t-1]", "self[t-1]"], lags=lags)
    sigma = np.abs(np.asarray(second.params[0]
                              + second.params[1] * disp
                              + second.params[2] * lag_y)) ** (1.0 / kappa)
    # A rolling floor keeps a near-zero fitted sigma from manufacturing an
    # enormous scaled return out of one quiet day.
    floor = np.full_like(sigma, np.nan)
    for t in range(window, sigma.size):
        floor[t] = np.nanquantile(sigma[t - window:t], 0.10)
    sigma = np.where(np.isfinite(floor), np.maximum(sigma, floor), sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(sigma > 0, y / sigma, np.nan)

    third = ols_hac(scaled, np.column_stack([disp, lag_y]),
                    names=["dispersion[t-1]", "self[t-1]"], lags=lags)
    return {"kappa": kappa, "mean": first, "vol": second, "sharpe": third}


def conditional_split(run: RunResult, conditioner: np.ndarray, *,
                      net: bool = True, label: str = "") -> dict:
    """A.9(c): the strategy's mean return above and below the conditioner's median.

    The conditioner is lagged here, once, for the same reason the regressors
    are: a split on today's value of anything is not a tradable rule.
    """
    y = run.net_bps if net else run.gross_bps
    cond = conditioner[1:][:-1]
    y = y[1:]
    good = np.isfinite(y) & np.isfinite(cond)
    y, cond = y[good], cond[good]
    if y.size < 40:
        return {"label": label, "n": int(y.size)}
    median = float(np.median(cond))
    high, low = y[cond > median], y[cond <= median]
    h, l = newey_west(high), newey_west(low)
    # The two buckets are disjoint samples, so their HAC standard errors add in
    # quadrature. This is not a paired test and must not be written as one - the
    # days in the high bucket are not the days in the low one.
    se = float(np.sqrt(h.se ** 2 + l.se ** 2))
    t = (h.mean - l.mean) / se if se > 0 else float("nan")
    return {
        "label": label, "median": median,
        "high_mean": h.mean, "high_t": h.t_stat, "high_n": h.n,
        "low_mean": l.mean, "low_t": l.t_stat, "low_n": l.n,
        "diff": h.mean - l.mean, "diff_t": t,
    }


# --------------------------------------------------------------------------
# The investor-heterogeneity placebo (A.9(d))
# --------------------------------------------------------------------------

def abnormal_reversal_signal(cs: CrossSection, *, interval: int = 20,
                             negative: bool = True) -> np.ndarray:
    """``AB_NR`` (or ``AB_PR``): the tug-of-war signal, as a formation matrix.

    A "negative daytime reversal" day is one where the market gapped up and then
    sold off; ``NR`` is its frequency over the last ``interval`` days and
    ``AB_NR`` scales that by its own trailing twelve-interval average. The paper
    expects this to carry nothing in futures, and uses that to rule out the
    equity-specific explanation for CO-OC. It is implemented here for the same
    purpose - as a falsification check, not as a candidate.
    """
    co, oc = cs.returns["r_co"], cs.returns["r_oc"]
    event = ((co > 0) & (oc < 0)) if negative else ((co < 0) & (oc > 0))
    event = event.astype(float)

    T, N = event.shape
    nr = np.full((T, N), np.nan)
    for t in range(interval, T):
        nr[t] = event[t - interval:t].mean(axis=0)

    long_window = 12 * interval
    out = np.full((T, N), np.nan)
    for t in range(interval + long_window, T):
        base = np.nanmean(nr[t - long_window:t], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[t] = np.where(base > 0, nr[t] / base, np.nan)
    return out


def run_signal_matrix(
    cs: CrossSection,
    signal: np.ndarray,
    holding: str = "r_cc",
    *,
    weighting: str = "rank",
    rebalance: int = 20,
    cost_bps: np.ndarray | None = None,
    label: str = "custom",
) -> RunResult:
    """Run an arbitrary formation matrix at an arbitrary rebalance interval.

    Used for the AB_NR placebo, where the signal is not one of the four return
    definitions and the holding period is the rebalance interval rather than a
    day. Holding returns are compounded over the interval, so the reported
    series is one observation per rebalance.
    """
    r = cs.returns[holding]
    T = r.shape[0]
    starts = list(range(0, T - rebalance, rebalance))
    if not starts:
        raise ValueError("sample too short for that rebalance interval")

    sig = np.array([signal[s] for s in starts])
    hold = np.array([np.prod(1.0 + r[s + 1:s + 1 + rebalance], axis=0) - 1.0
                     for s in starts])
    dates = np.array([cs.dates[s] for s in starts])
    usable = np.all(np.isfinite(sig), axis=1) & np.all(np.isfinite(hold), axis=1)
    sig, hold, dates = sig[usable], hold[usable], dates[usable]
    if sig.shape[0] < 10:
        raise ValueError(f"only {sig.shape[0]} usable rebalances")

    w = build_weights(sig, weighting=weighting)
    gross_pnl = np.nansum(w * hold, axis=1)
    capital = np.nansum(np.abs(w), axis=1) / 2.0
    gross_bps = np.where(capital > 0, gross_pnl / capital, np.nan) * 1e4

    if cost_bps is None:
        cost_series = np.zeros_like(gross_bps)
    else:
        c = np.array([cost_bps[s] for s in starts])[usable]
        prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        cost_pnl = np.nansum(np.abs(w - prev) * c * 0.5, axis=1) / 1e4
        cost_series = np.where(capital > 0, cost_pnl / capital, np.nan) * 1e4

    zeros = np.zeros_like(gross_bps)
    return RunResult(
        variant=label, weighting=weighting, dates=dates, weights=w,
        gross_bps=gross_bps, net_bps=gross_bps - cost_series,
        cost_bps=cost_series, long_leg_bps=zeros.copy(),
        short_leg_bps=zeros.copy(),
        turnover=np.full_like(gross_bps, np.nan), symbols=cs.symbols,
        meta={"rebalance": rebalance, "holding": holding,
              "periods_per_year": 252 / rebalance},
    )
