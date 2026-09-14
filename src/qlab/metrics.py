"""Performance measurement for a closed trade list.

One tearsheet function, used by every strategy, so that two strategies are
never compared on differently-computed Sharpes.

Two choices worth stating, because they change the headline number:

**Sharpe is computed on daily equity returns**, not per trade. A per-trade
Sharpe flatters any strategy that trades rarely, because it silently annualises
by trade count rather than by time. Daily returns of a strategy that is flat
most of the time are mostly zeros, and those zeros belong in the denominator -
capital was committed and idle.

**Costs reconcile exactly, and are not double counted.** A trade's price PnL is
computed from its actual fill prices, so the spread and the slippage are already
inside it. ``gross_usd`` therefore adds the slippage back to give the PnL before
slippage and commission, and ``net = gross - slippage - commission`` closes. The
one cost that cannot be separated out this way is the spread, which is paid by
entering at the ask and leaving at the bid; it stays inside ``gross_usd``.

**Cost drag is the share of the gross edge that costs consumed** - the question
that decides whether a strategy is worth running. It is only meaningful when the
gross edge is positive; when it is negative the strategy failed before costs and
the drag figure is suppressed.
"""

from __future__ import annotations

import math

import polars as pl

TRADING_DAYS = 252


def _drawdown(equity: pl.Series) -> tuple[float, int]:
    """Max drawdown as a fraction, and its length in samples."""
    peak = equity.cum_max()
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if dd.len() else 0.0

    longest = 0
    current = 0
    for value in dd.to_list():
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return max_dd, longest


def daily_equity(trades: pl.DataFrame, starting_equity: float) -> pl.DataFrame:
    """Equity at the end of each calendar day on which the strategy was live.

    Days with no trades are carried forward rather than dropped, so that idle
    capital is visible in the volatility rather than compressed out of it.
    """
    if trades.is_empty():
        return pl.DataFrame(
            {"date": [], "equity": []},
            schema={"date": pl.Date, "equity": pl.Float64},
        )

    by_day = (
        trades.with_columns(date=pl.from_epoch("exit_ts", time_unit="us").dt.date())
        .group_by("date")
        .agg(pnl=pl.col("net_usd").sum())
        .sort("date")
    )
    spine = pl.DataFrame(
        {
            "date": pl.date_range(
                by_day["date"].min(), by_day["date"].max(), "1d", eager=True
            )
        }
    ).filter(pl.col("date").dt.weekday() < 6)

    return (
        spine.join(by_day, on="date", how="left")
        .with_columns(pl.col("pnl").fill_null(0.0))
        .with_columns(equity=starting_equity + pl.col("pnl").cum_sum())
    )


def tearsheet(
    trades: pl.DataFrame,
    *,
    starting_equity: float = 100_000.0,
    label: str = "",
) -> dict:
    """Every headline number for one run, as a flat dict."""
    n = trades.height
    if n == 0:
        return {"label": label, "trades": 0}

    equity = daily_equity(trades, starting_equity)
    returns = (equity["equity"] / equity["equity"].shift(1) - 1).drop_nulls()
    days = equity.height
    years = days / TRADING_DAYS

    net = trades["net_usd"]
    wins = net.filter(net > 0)
    losses = net.filter(net <= 0)

    total_net = float(net.sum())
    total_commission = float(trades["commission_usd"].sum())
    total_slippage = float(trades["slippage_usd"].sum())
    # Fill prices already carry the slippage, so add it back to state the PnL
    # before it. net = gross - slippage - commission then closes exactly.
    total_gross = float(trades["gross_usd"].sum()) + total_slippage
    total_cost = total_commission + total_slippage
    # The cost-free counterfactual: the same trades filled at mid.
    total_mid = float(trades["mid_pnl_usd"].sum()) if "mid_pnl_usd" in trades.columns else float("nan")
    total_exec = float(trades["execution_cost_usd"].sum()) if "execution_cost_usd" in trades.columns else float("nan")

    final_equity = starting_equity + total_net
    cagr = (final_equity / starting_equity) ** (1 / years) - 1 if years > 0 else 0.0
    mean_r = float(returns.mean()) if returns.len() else 0.0
    std_r = float(returns.std()) if returns.len() > 1 else 0.0
    downside = returns.filter(returns < 0)
    std_down = float(downside.std()) if downside.len() > 1 else 0.0

    sharpe = mean_r / std_r * math.sqrt(TRADING_DAYS) if std_r > 0 else float("nan")
    sortino = (
        mean_r / std_down * math.sqrt(TRADING_DAYS) if std_down > 0 else float("nan")
    )
    max_dd, dd_len = _drawdown(equity["equity"])

    gross_win = float(wins.sum()) if wins.len() else 0.0
    gross_loss = abs(float(losses.sum())) if losses.len() else 0.0

    first_ts = trades["entry_ts"].min()
    last_ts = trades["exit_ts"].max()

    return {
        "label": label,
        "trades": n,
        "trades_per_year": n / years if years > 0 else float("nan"),
        "start": first_ts,
        "end": last_ts,
        "days": days,
        "years": years,
        "net_usd": total_net,
        "gross_usd": total_gross,
        "cost_usd": total_cost,
        "commission_usd": total_commission,
        "slippage_usd": total_slippage,
        "mid_pnl_usd": total_mid,
        "execution_cost_usd": total_exec,
        "mid_edge_per_trade": total_mid / n,
        "all_in_cost_per_trade": (total_mid - total_net) / n,
        "return_pct": 100.0 * total_net / starting_equity,
        "cagr_pct": 100.0 * cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": 100.0 * max_dd,
        "max_dd_days": dd_len,
        "calmar": (cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
        "win_rate_pct": 100.0 * wins.len() / n,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy_usd": total_net / n,
        "avg_win_usd": float(wins.mean()) if wins.len() else 0.0,
        "avg_loss_usd": float(losses.mean()) if losses.len() else 0.0,
        # The share of the gross edge that costs consumed. Above 100% means the
        # strategy found an edge and handed all of it to the broker; it is left
        # undefined when the gross edge is itself negative, because a ratio to a
        # negative denominator would read as a small drag on a total failure.
        "cost_drag_pct": 100.0 * (total_mid - total_net) / total_mid
        if total_mid > 0
        else float("nan"),
        "avg_hold_min": float(trades["hold_s"].mean()) / 60.0,
        "median_hold_min": float(trades["hold_s"].median()) / 60.0,
    }


def by_period(trades: pl.DataFrame, every: str = "1y") -> pl.DataFrame:
    """Net PnL and trade count per period - for spotting regime dependence."""
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.with_columns(
            period=pl.from_epoch("exit_ts", time_unit="us").dt.truncate(every)
        )
        .group_by("period")
        .agg(
            trades=pl.len(),
            net_usd=pl.col("net_usd").sum(),
            gross_usd=pl.col("gross_usd").sum(),
            win_rate=100.0 * (pl.col("net_usd") > 0).mean(),
        )
        .sort("period")
    )


def format_tearsheet(sheet: dict) -> str:
    """The tearsheet as aligned text, for a report or a terminal."""
    if sheet.get("trades", 0) == 0:
        return f"{sheet.get('label', '')}: no trades"

    rows = [
        ("trades", f"{sheet['trades']:,}  ({sheet['trades_per_year']:.0f}/yr)"),
        ("net return", f"{sheet['return_pct']:+.2f}%"),
        ("CAGR", f"{sheet['cagr_pct']:+.2f}%"),
        ("Sharpe (daily)", f"{sheet['sharpe']:.2f}"),
        ("Sortino", f"{sheet['sortino']:.2f}"),
        ("max drawdown", f"{sheet['max_dd_pct']:.2f}%"),
        ("Calmar", f"{sheet['calmar']:.2f}"),
        ("win rate", f"{sheet['win_rate_pct']:.1f}%"),
        ("profit factor", f"{sheet['profit_factor']:.2f}"),
        ("expectancy", f"${sheet['expectancy_usd']:+.2f}/trade"),
        ("edge at mid", f"${sheet['mid_pnl_usd']:+,.0f}  (${sheet['mid_edge_per_trade']:+.2f}/trade, no costs)"),
        ("  spread+slippage", f"${sheet['execution_cost_usd']:,.0f}"),
        ("  commission", f"${sheet['commission_usd']:,.0f}"),
        ("all-in cost", f"${sheet['all_in_cost_per_trade']:.2f}/trade"),
        (
            "cost drag",
            f"{sheet['cost_drag_pct']:.1f}% of gross"
            if sheet["cost_drag_pct"] == sheet["cost_drag_pct"]
            else "n/a - gross edge is negative",
        ),
        ("median hold", f"{sheet['median_hold_min']:.1f} min"),
    ]
    width = max(len(k) for k, _ in rows)
    head = f"--- {sheet['label']} ---" if sheet.get("label") else ""
    return "\n".join([head] + [f"  {k:<{width}}  {v}" for k, v in rows])


# --------------------------------------------------------------------------
# Position-based measurement
# --------------------------------------------------------------------------
#
# The tearsheet above starts from a list of round trips, which is the right
# shape for a strategy that opens and closes discrete trades. A strategy that
# instead holds a continuously varying weight has no round trips to list - its
# record is a series of periodic returns - so it needs its own entry point
# rather than a synthetic trade list built to satisfy this one.
#
# Both end in the same units and the same drawdown function, so a weight-based
# strategy and a trade-based one can still be put in the same table.


def tearsheet_from_returns(
    returns_bps,
    *,
    label: str = "",
    periods_per_year: int = TRADING_DAYS,
    turnover=None,
) -> dict:
    """Headline numbers for a series of periodic net returns, quoted in bps.

    Returns are compounded, not summed: a strategy that holds a *weight* is
    implicitly rebalanced to that weight on equity, so its equity curve is
    geometric and a sum would overstate it after a drawdown.
    """
    r = pl.Series(returns_bps).drop_nulls()
    r = r.filter(r.is_finite())
    n = r.len()
    if n == 0:
        return {"label": label, "periods": 0}

    # The curve is anchored at 1.0 before the first return, so a loss in the
    # very first period counts as drawdown rather than becoming the peak the
    # rest of the curve is measured against.
    growth = pl.concat([pl.Series([1.0]), (1.0 + r / 1e4).cum_prod()])
    years = n / periods_per_year
    total = float(growth[-1]) - 1.0
    cagr = (1.0 + total) ** (1 / years) - 1 if years > 0 else 0.0

    mean = float(r.mean())
    sd = float(r.std()) if n > 1 else 0.0
    downside = r.filter(r < 0)
    sd_down = float(downside.std()) if downside.len() > 1 else 0.0
    ann_vol = sd / 1e4 * math.sqrt(periods_per_year)
    max_dd, dd_len = _drawdown(growth)

    sheet = {
        "label": label,
        "periods": n,
        "years": years,
        "total_pct": 100.0 * total,
        "cagr_pct": 100.0 * cagr,
        "ann_vol_pct": 100.0 * ann_vol,
        "mean_bps": mean,
        "sharpe": mean / sd * math.sqrt(periods_per_year) if sd > 0 else float("nan"),
        "sortino": mean / sd_down * math.sqrt(periods_per_year)
        if sd_down > 0 else float("nan"),
        "max_dd_pct": 100.0 * max_dd,
        "max_dd_periods": dd_len,
        "calmar": (cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
        "hit_rate_pct": 100.0 * float((r > 0).mean()),
        "best_pct": float(r.max()) / 100.0,
        "worst_pct": float(r.min()) / 100.0,
        # A t-stat on the mean period return: the honest answer to "could this
        # be zero?", and on a daily strategy over a few years it is usually
        # humbling.
        "t_stat": mean / sd * math.sqrt(n) if sd > 0 else float("nan"),
    }
    if turnover is not None:
        t = pl.Series(turnover).drop_nulls()
        sheet["turnover_per_year"] = float(t.sum()) / years if years > 0 else float("nan")
    return sheet


def format_returns_tearsheet(sheet: dict) -> str:
    if sheet.get("periods", 0) == 0:
        return f"{sheet.get('label', '')}: no data"
    rows = [
        ("sessions", f"{sheet['periods']:,}  ({sheet['years']:.2f} yr)"),
        ("total return", f"{sheet['total_pct']:+.1f}%"),
        ("CAGR", f"{sheet['cagr_pct']:+.2f}%"),
        ("volatility", f"{sheet['ann_vol_pct']:.2f}%"),
        ("Sharpe", f"{sheet['sharpe']:.2f}"),
        ("Sortino", f"{sheet['sortino']:.2f}"),
        ("max drawdown", f"{sheet['max_dd_pct']:.1f}%  ({sheet['max_dd_periods']} sessions)"),
        ("Calmar", f"{sheet['calmar']:.2f}"),
        ("hit rate", f"{sheet['hit_rate_pct']:.1f}%"),
        ("mean session", f"{sheet['mean_bps']:+.2f} bps  (t = {sheet['t_stat']:+.2f})"),
        ("best / worst", f"{sheet['best_pct']:+.2f}% / {sheet['worst_pct']:+.2f}%"),
    ]
    if "turnover_per_year" in sheet:
        rows.append(("turnover", f"{sheet['turnover_per_year']:.1f}x notional/yr"))
    width = max(len(k) for k, _ in rows)
    head = f"--- {sheet['label']} ---" if sheet.get("label") else ""
    return "\n".join([head] + [f"  {k:<{width}}  {v}" for k, v in rows])
