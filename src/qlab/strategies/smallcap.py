"""Poudel (2025): small-cap strategy families, on a daily cross-section.

Subject: *Small-Cap Stock Trading Strategies for Retail Traders* (SSRN 5921742),
Chapter 5. Six families are specified; three of them are expressible on daily
bars and are implemented here:

* **A - volatility-scaled momentum with liquidity filters** (Section 5.1). Long
  when 20-day mean return clears a threshold, dollar volume clears $500K,
  annualised volatility is below a cap and the price sits in $2-$50. Size is
  ``R / (sigma * price)``. Out on an ATR profit target, a 2-ATR stop, or ten days.
* **D - breakout and retest** (Section 5.4). A three-phase state machine: close
  above a 20-day high on 1.2x volume, pull back into the prior day's range, then
  close back above the retest level on renewed volume.
* **F - regime-filtered composite** (Section 5.6). ``SPY > SMA(SPY, 50)`` gates
  the momentum and breakout families on; below it the paper switches to family C,
  which is intraday and therefore unavailable, so F is implemented as *flat when
  risk-off* and reported as such. That is the charitable reading: the alternative
  is to leave the risk-off half unhedged and inherit family A's losses there.

Families B, C and E need minute bars or a point-in-time earnings calendar and are
out of reach on a daily equity panel; see the report for what stands in.

Everything is written against a **long panel** - one row per ticker-day, columns
``date, ticker, open, high, low, close, volume``, prices already split- and
dividend-adjusted - so the same code runs on any universe.

The exits are parameters, not constants. Section 5.1.2 states three exit rules
that partly contradict each other (a time stop listed under "stop loss", and an
intraday 2% rule that daily bars cannot see), so :class:`FamilyA` exposes the
target, stop and horizon and the driver sweeps them rather than picking one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Panel plumbing
# --------------------------------------------------------------------------


def to_wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    """``date x ticker`` matrix for one column of a long panel."""
    return panel.pivot(index="date", columns="ticker", values=column).sort_index()


def adjust_ohlc(panel: pd.DataFrame) -> pd.DataFrame:
    """Scale raw OHLC by ``adj_close / close``.

    Yahoo adjusts only the close. Leaving the highs and lows unadjusted puts a
    2-for-1 split straight into the ATR and into every breakout level, which
    manufactures signals that never existed.
    """
    out = panel.copy()
    factor = out["adj_close"] / out["close"]
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    out["volume"] = out["volume"] / factor.replace(0, np.nan)
    return out.drop(columns=["adj_close"])


def _true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()]
    ).groupby(level=0).max()


def indicators(panel: pd.DataFrame, *, atr_window: int = 14, vol_window: int = 20,
               mom_window: int = 20, liq_window: int = 20) -> dict[str, pd.DataFrame]:
    """Every derived quantity the families need, as ``date x ticker`` matrices.

    Section 5.1.5: ATR is a 14-day SMA of true range; volatility is a 20-day
    rolling standard deviation of log returns, annualised by ``sqrt(252)``;
    momentum is the *mean* of the last ``L`` daily returns, not their sum.
    """
    o = to_wide(panel, "open")
    h = to_wide(panel, "high")
    l = to_wide(panel, "low")
    c = to_wide(panel, "close")
    v = to_wide(panel, "volume")

    logret = np.log(c / c.shift(1))
    simple = c.pct_change(fill_method=None)
    prev = c.shift(1)
    tr = pd.concat(
        [(h - l).stack(future_stack=True),
         (h - prev).abs().stack(future_stack=True),
         (l - prev).abs().stack(future_stack=True)],
        axis=1,
    ).max(axis=1).unstack()

    return {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "atr": tr.rolling(atr_window).mean(),
        "vol": logret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS),
        "mom": simple.rolling(mom_window).mean(),
        "dollar_volume": (v * c).rolling(liq_window).mean(),
        "avg_volume": v.rolling(liq_window).mean(),
        "ret": simple,
    }


# --------------------------------------------------------------------------
# Family A - volatility-scaled momentum (Section 5.1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyA:
    """Section 5.1's parameters, with the recommended column of Table 5.1.6."""

    lookback: int = 20
    mom_threshold: float = 0.005
    vol_cap: float = 1.5
    min_dollar_volume: float = 500_000
    price_min: float = 2.0
    price_max: float = 50.0
    target_atr: float = 1.0
    stop_atr: float = 2.0
    time_stop: int = 10

    def entries(self, ind: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Boolean ``date x ticker`` mask, true where the entry condition holds.

        Everything is evaluated on data through the close of the signal day, so
        the entry the simulator books is the *next* open. The paper's pseudocode
        enters at ``close[t] + spread_adjustment``, which is a fill at a price
        that only exists after the bar it is computed from; entering next open is
        the same rule made executable.
        """
        c = ind["close"]
        return (
            (ind["mom"] > self.mom_threshold)
            & (ind["dollar_volume"] > self.min_dollar_volume)
            & (ind["vol"] < self.vol_cap)
            & (c >= self.price_min)
            & (c <= self.price_max)
        ).fillna(False)


# --------------------------------------------------------------------------
# Family D - breakout and retest (Section 5.4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyD:
    resistance_window: int = 20
    breakout_volume: float = 1.2
    retest_fraction: float = 0.5
    retest_deadline: int = 5
    target_atr: float = 2.0
    stop_atr: float = 1.5
    time_stop: int = 10

    def entries(self, ind: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """The Section 5.4.4 state machine, vectorised over the cross-section.

        The paper's pseudocode only ever checks for a retest on the single day
        after the breakout, which makes the pattern almost unobservable; the
        deadline is a parameter here and defaults to five days, which is the
        charitable reading of "wait for next day".
        """
        c, h, l, v = ind["close"], ind["high"], ind["low"], ind["volume"]
        resistance = h.shift(1).rolling(self.resistance_window).max()
        support = l.shift(1).rolling(self.resistance_window).min()
        avg_v = ind["avg_volume"]

        broke = (c > resistance) & (v > self.breakout_volume * avg_v)
        prev_range = (h - l).shift(1)
        retest_level = resistance - self.retest_fraction * prev_range
        retested = (l <= retest_level) & (c > support)
        resumed = (c > retest_level) & (v > avg_v)

        out = pd.DataFrame(False, index=c.index, columns=c.columns)
        b = broke.to_numpy()
        rt = retested.to_numpy()
        rs = resumed.to_numpy()
        res = out.to_numpy()

        n_days, n_names = b.shape
        # 0 = idle, 1 = broke out, 2 = retested. Ages out after the deadline.
        phase = np.zeros(n_names, dtype=np.int8)
        age = np.zeros(n_names, dtype=np.int16)
        for t in range(n_days):
            live = phase > 0
            age[live] += 1
            expired = live & (age > self.retest_deadline)
            phase[expired] = 0
            age[expired] = 0

            fire = (phase == 2) & rs[t]
            res[t] = fire
            phase[fire] = 0
            age[fire] = 0

            step = (phase == 1) & rt[t]
            phase[step] = 2

            fresh = (phase == 0) & b[t]
            phase[fresh] = 1
            age[fresh] = 0

        return pd.DataFrame(res, index=c.index, columns=c.columns)


# --------------------------------------------------------------------------
# Family F - regime filter (Section 5.6)
# --------------------------------------------------------------------------


def regime_risk_on(spy_close: pd.Series, window: int = 50) -> pd.Series:
    """``sign(SPY - SMA(SPY, 50))`` > 0, lagged one day so it is knowable."""
    sma = spy_close.rolling(window).mean()
    return (spy_close > sma).shift(1).astype("boolean").fillna(False).astype(bool)


# --------------------------------------------------------------------------
# Costs (Section 7.1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityCosts:
    """The paper's own assumptions, Section 7.1, as a per-share charge.

    ``spread`` is crossed once per round turn - in at the ask, out at the bid -
    and ``slippage`` is charged on each side, which is what "+$0.01 on entry,
    -$0.01 on exit" says. ``min_bps`` is a floor as a share of price, because a
    fixed cent charge understates a $3 stock's spread and overstates a $45 one.
    """

    spread: float = 0.05
    slippage: float = 0.01
    commission_per_share: float = 0.0
    min_bps: float = 0.0

    def per_share_round_turn(self, price: float | np.ndarray) -> float | np.ndarray:
        fixed = self.spread + 2 * self.slippage + 2 * self.commission_per_share
        floor = np.asarray(price) * self.min_bps / 10_000
        return np.maximum(fixed, floor)


COST_SCENARIOS = {
    "paper": EquityCosts(),
    "paper_zero_spread": EquityCosts(spread=0.02, slippage=0.005),
    "relative_100bps": EquityCosts(spread=0.05, slippage=0.01, min_bps=100.0),
}


# --------------------------------------------------------------------------
# Portfolio simulator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Portfolio:
    """Chapter 6's constraints, reduced to the ones a daily panel can enforce.

    ``size_mode`` is the one place this departs from the text, and deliberately.

    ``"paper"`` is Section 5.1.3 exactly: ``shares = R / (sigma * price)``, so a
    position's *dollar* size is ``R / sigma`` and does not depend on the account
    at all. The paper's own worked example is a $250 position. Ten of those on
    any plausible retail account is single-digit gross exposure, and a strategy
    that is 90% cash cannot return 12% a year unless the deployed tenth returns
    over 100%. That arithmetic is a finding, so the mode stays available.

    ``"vol_target"`` keeps the idea - size inversely with volatility - and drops
    the accidental scale: each slot gets ``equity / max_positions``, tilted by
    ``target_vol / sigma`` and clipped, so a full book is fully invested. This is
    what a practitioner would actually deploy, and it is the version the verdict
    should rest on.
    """

    starting_equity: float = 25_000.0
    risk_per_trade: float = 200.0
    max_positions: int = 10
    max_position_pct: float = 0.10
    drawdown_scaling: bool = True
    size_mode: str = "paper"
    target_vol: float = 0.60
    tilt_clip: tuple[float, float] = (0.5, 2.0)
    max_gross: float = 1.0
    """Total notional as a multiple of equity. 1.0 is a cash account.

    The volatility tilt can otherwise ask for more than the account holds - a
    full book of ten slots each tilted 1.2x is 120% gross - and borrowing to
    reach it is a different strategy from the one being tested."""


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    daily_return: pd.Series
    exposure: pd.Series


def _drawdown_multiplier(dd: float) -> float:
    """Section 6.1's ladder: cut size as the drawdown deepens."""
    if dd < 0.10:
        return 1.0
    if dd < 0.15:
        return 0.75
    if dd < 0.20:
        return 0.5
    return 0.25


def simulate(
    entries: pd.DataFrame,
    ind: dict[str, pd.DataFrame],
    *,
    target_atr: float,
    stop_atr: float,
    time_stop: int,
    costs: EquityCosts,
    portfolio: Portfolio,
    eligible: pd.DataFrame | None = None,
    regime: pd.Series | None = None,
) -> Result:
    """Walk the panel day by day, holding at most ``max_positions`` names.

    Entry is the open of the day after the signal. Exits are checked against that
    day's high and low; when a bar spans both the target and the stop the **stop**
    is taken, because a daily bar cannot say which came first and assuming the
    good one is how a backtest flatters itself. A gap through a level fills at the
    open, not at the level.
    """
    dates = entries.index
    tickers = entries.columns

    o = ind["open"].reindex(index=dates, columns=tickers).to_numpy()
    h = ind["high"].reindex(index=dates, columns=tickers).to_numpy()
    l = ind["low"].reindex(index=dates, columns=tickers).to_numpy()
    c = ind["close"].reindex(index=dates, columns=tickers).to_numpy()
    atr = ind["atr"].reindex(index=dates, columns=tickers).to_numpy()
    vol = ind["vol"].reindex(index=dates, columns=tickers).to_numpy()
    sig = entries.to_numpy()
    elig = (
        eligible.reindex(index=dates, columns=tickers).fillna(False).to_numpy()
        if eligible is not None
        else np.ones_like(sig, dtype=bool)
    )
    on = (
        regime.reindex(dates).fillna(False).to_numpy()
        if regime is not None
        else np.ones(len(dates), dtype=bool)
    )

    equity = portfolio.starting_equity
    peak = equity
    open_pos: dict[int, dict] = {}
    trades: list[dict] = []
    equity_path = np.empty(len(dates))
    exposure_path = np.zeros(len(dates))

    for t in range(len(dates)):
        realised = 0.0

        # --- exits, on today's bar -----------------------------------------
        for j in list(open_pos):
            p = open_pos[j]
            if not np.isfinite(o[t, j]):
                continue
            exit_price, reason = None, None
            if o[t, j] <= p["stop"]:
                exit_price, reason = o[t, j], "gap_stop"
            elif l[t, j] <= p["stop"]:
                exit_price, reason = p["stop"], "stop"
            elif o[t, j] >= p["target"]:
                exit_price, reason = o[t, j], "gap_target"
            elif h[t, j] >= p["target"]:
                exit_price, reason = p["target"], "target"
            elif t - p["entry_i"] >= time_stop:
                exit_price, reason = c[t, j], "time_stop"
            if exit_price is None:
                continue
            cost = costs.per_share_round_turn(p["entry_price"]) * p["shares"]
            pnl = (exit_price - p["entry_price"]) * p["shares"] - cost
            realised += pnl
            trades.append(dict(
                ticker=tickers[j], entry_date=dates[p["entry_i"]], exit_date=dates[t],
                entry_price=p["entry_price"], exit_price=float(exit_price),
                shares=p["shares"], pnl=float(pnl), cost=float(cost), reason=reason,
                bars_held=t - p["entry_i"],
            ))
            del open_pos[j]

        equity += realised

        # --- entries, from yesterday's signal, filled at today's open ------
        if t > 0 and on[t]:
            room = portfolio.max_positions - len(open_pos)
            if room > 0:
                candidates = np.flatnonzero(
                    sig[t - 1] & elig[t - 1] & np.isfinite(o[t]) & np.isfinite(atr[t - 1])
                    & (atr[t - 1] > 0) & (vol[t - 1] > 0)
                )
                candidates = [j for j in candidates if j not in open_pos]
                # Tie-break by momentum, so the choice is stated rather than
                # falling out of column order.
                mom_row = ind["mom"].reindex(index=dates, columns=tickers).to_numpy()[t - 1]
                candidates.sort(key=lambda j: -mom_row[j] if np.isfinite(mom_row[j]) else 0.0)
                scale = _drawdown_multiplier((peak - equity) / peak) if portfolio.drawdown_scaling else 1.0
                gross = sum(
                    p["shares"] * c[t - 1, j2] for j2, p in open_pos.items()
                    if np.isfinite(c[t - 1, j2])
                )
                budget = max(0.0, portfolio.max_gross * equity - gross)
                for j in candidates[:room]:
                    price = float(o[t, j])
                    sigma = float(vol[t - 1, j])
                    if portfolio.size_mode == "paper":
                        notional = portfolio.risk_per_trade * scale / sigma
                    elif portfolio.size_mode == "vol_target":
                        lo_c, hi_c = portfolio.tilt_clip
                        tilt = min(max(portfolio.target_vol / sigma, lo_c), hi_c)
                        notional = equity / portfolio.max_positions * tilt * scale
                    else:
                        raise ValueError(f"unknown size_mode {portfolio.size_mode!r}")
                    notional = min(notional, budget)
                    shares = int(notional / price)
                    cap = int(equity * portfolio.max_position_pct / price)
                    shares = max(0, min(shares, cap))
                    if shares == 0:
                        continue
                    budget -= shares * price
                    a = float(atr[t - 1, j])
                    open_pos[j] = dict(
                        entry_i=t, entry_price=price, shares=shares,
                        target=price + target_atr * a, stop=price - stop_atr * a,
                    )

        # --- mark to market -------------------------------------------------
        unreal = sum(
            (c[t, j] - p["entry_price"]) * p["shares"]
            for j, p in open_pos.items() if np.isfinite(c[t, j])
        )
        marked = equity + unreal
        equity_path[t] = marked
        peak = max(peak, marked)
        exposure_path[t] = sum(
            p["shares"] * c[t, j] for j, p in open_pos.items() if np.isfinite(c[t, j])
        ) / marked if marked > 0 else 0.0

    eq = pd.Series(equity_path, index=dates, name="equity")
    return Result(
        trades=pd.DataFrame(trades),
        equity=eq,
        daily_return=eq.pct_change().fillna(0.0),
        exposure=pd.Series(exposure_path, index=dates, name="exposure"),
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def summarise(res: Result, label: str = "") -> dict:
    r = res.daily_return.to_numpy()
    eq = res.equity
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    years = len(r) / TRADING_DAYS
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    t = res.trades
    wins = t[t["pnl"] > 0] if len(t) else t
    losses = t[t["pnl"] <= 0] if len(t) else t
    gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = float(-losses["pnl"].sum()) if len(losses) else 0.0
    return dict(
        label=label,
        sharpe=sharpe,
        annual_return=cagr,
        total_return=total,
        max_drawdown=dd,
        calmar=float(cagr / abs(dd)) if dd < 0 and np.isfinite(cagr) else float("nan"),
        n_trades=int(len(t)),
        win_rate=float(len(wins) / len(t)) if len(t) else float("nan"),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("nan"),
        avg_bars_held=float(t["bars_held"].mean()) if len(t) else float("nan"),
        total_cost=float(t["cost"].sum()) if len(t) else 0.0,
        gross_pnl=float(t["pnl"].sum() + t["cost"].sum()) if len(t) else 0.0,
        avg_exposure=float(res.exposure.mean()),
        days=len(r),
    )


def sweep_exits(
    entries: pd.DataFrame,
    ind: dict[str, pd.DataFrame],
    *,
    costs: EquityCosts,
    portfolio: Portfolio,
    targets: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    stops: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    horizons: tuple[int, ...] = (5, 10, 20),
    **kwargs,
) -> pd.DataFrame:
    """The exit surface, not one point on it.

    Section 5.1.2 lists exits that partly contradict each other, so a verdict
    that rested on one of them would be a verdict about the choice.
    """
    rows = []
    for target in targets:
        for stop in stops:
            for horizon in horizons:
                res = simulate(entries, ind, target_atr=target, stop_atr=stop,
                               time_stop=horizon, costs=costs, portfolio=portfolio, **kwargs)
                rows.append({"target_atr": target, "stop_atr": stop, "time_stop": horizon,
                             **summarise(res)})
    return pd.DataFrame(rows).drop(columns=["label"])
