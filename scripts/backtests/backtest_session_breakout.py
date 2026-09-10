"""Evaluate the session opening-range breakout across presets, periods and symbols.

Everything is reported per unit of risk taken (R), where 1R is the bracket
width - the distance to the stop. That is the only unit in which a $7 gold
bracket from 2023 and a $27 one from 2026 are the same bet, and the only one
in which gold, FX and an index can be put in the same table.

Run:  python scripts/backtests/backtest_session_breakout.py [-s SYMBOL ...] [-o OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.strategies.session_breakout import BreakoutConfig, PRESETS, run  # noqa: E402

# The strategy was re-tuned on 2026, which lies inside the locked test split.
# It is therefore already in-sample there, and the honest out-of-sample
# evidence for *this* parameter set is everything before it.
PERIODS: dict[str, tuple[date, date]] = {
    "dev 2020-2023": (date(2020, 1, 29), date(2023, 12, 31)),
    "val 2024-25H1": (date(2024, 1, 1), date(2025, 6, 30)),
    "test 25H2-26": (date(2025, 7, 1), date(2026, 9, 1)),
    "2026 (tuned)": (date(2026, 1, 1), date(2026, 9, 1)),
}


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(curve) - curve))


def summarise(trades: pl.DataFrame, label: str) -> dict:
    """Trade- and day-level statistics for one slice.

    The t-statistic is computed on **daily** totals, not on trades: seven
    windows on the same instrument on the same day are mostly the same bet,
    and a per-trade t-stat would count that correlation as independent
    evidence and overstate significance by roughly the square root of the
    number of windows.
    """
    n = len(trades)
    if n == 0:
        return {"label": label, "n_trades": 0}

    daily = (trades.group_by("day")
             .agg(r=pl.col("r_multiple").sum(),
                  r_mid=pl.col("r_mid").sum())
             .sort("day"))
    d = daily["r"].to_numpy()
    curve = np.cumsum(d)
    wins = trades.filter(pl.col("r_multiple") > 0)
    losses = trades.filter(pl.col("r_multiple") <= 0)
    gain = float(wins["r_multiple"].sum())
    loss = float(-losses["r_multiple"].sum())
    years = max((daily["day"].max() - daily["day"].min()).days / 365.25, 1e-9)

    t_stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else 0.0
    sharpe = float(d.mean() / d.std(ddof=1) * np.sqrt(252)) if len(d) > 2 else 0.0

    return {
        "label": label,
        "n_trades": n,
        "n_days": len(daily),
        "win_pct": round(100 * len(wins) / n, 1),
        "mean_r": round(float(trades["r_multiple"].mean()), 4),
        "mean_r_mid": round(float(trades["r_mid"].mean()), 4),
        "total_r": round(float(trades["r_multiple"].sum()), 1),
        "total_r_per_yr": round(float(trades["r_multiple"].sum()) / years, 1),
        "profit_factor": round(gain / loss, 3) if loss > 0 else float("inf"),
        "t_stat_daily": round(t_stat, 2),
        "sharpe_daily": round(sharpe, 2),
        "max_dd_r": round(_max_drawdown(curve), 1),
        "tp_pct": round(100 * len(trades.filter(pl.col("reason") == "tp")) / n, 1),
        "stop_pct": round(100 * len(trades.filter(pl.col("reason") == "stop")) / n, 1),
        "flat_pct": round(100 * len(trades.filter(pl.col("reason") == "flat")) / n, 1),
        "mean_width_pct": round(float(trades["width_pct"].mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--symbols", nargs="+",
                    default=["XAUUSD", "EURUSD", "USDJPY", "USTEC"])
    ap.add_argument("-p", "--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("-o", "--out", default="reports/strategies/session_breakout.json")
    args = ap.parse_args()

    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(200)

    results: list[dict] = []
    trade_store: dict[str, pl.DataFrame] = {}

    for symbol in args.symbols:
        # Cost is measured over the whole corpus here rather than per split,
        # because a single backtest spans all three; the per-split difference
        # on gold is about 0.03 bps a round turn against a 1R bracket of
        # ~30 bps, so it cannot change a sign.
        cost = CostModel.from_profiles(symbol)
        for preset in args.presets:
            cfg = BreakoutConfig.preset(preset)
            trades = run(symbol, cfg, start="2020-01-29", end="2026-09-02",
                         allow_test=True, cost=cost)
            if trades.is_empty():
                continue
            trade_store[f"{symbol}|{preset}"] = trades
            for pname, (lo, hi) in PERIODS.items():
                sl = trades.filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
                row = summarise(sl, pname)
                row |= {"symbol": symbol, "preset": preset, "period": pname}
                results.append(row)
            print(f"{symbol:8s} {preset:16s} {len(trades):6d} trades", flush=True)

    frame = pl.DataFrame(results)
    cols = ["symbol", "preset", "period", "n_trades", "n_days", "win_pct",
            "tp_pct", "stop_pct", "flat_pct", "mean_r", "mean_r_mid",
            "total_r", "total_r_per_yr", "profit_factor", "t_stat_daily",
            "sharpe_daily", "max_dd_r", "mean_width_pct"]
    frame = frame.select([c for c in cols if c in frame.columns])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=str))

    for symbol in args.symbols:
        sub = frame.filter(pl.col("symbol") == symbol)
        if sub.is_empty():
            continue
        print(f"\n===== {symbol} =====")
        print(sub.drop("symbol"))

    # Persist the trade tape so follow-up questions do not need a re-run.
    tape_dir = out.parent / "session_breakout_trades"
    tape_dir.mkdir(parents=True, exist_ok=True)
    for key, trades in trade_store.items():
        symbol, preset = key.split("|")
        trades.write_parquet(tape_dir / f"{symbol}_{preset}.parquet")
    print(f"\nwrote {out} and {len(trade_store)} trade tapes to {tape_dir}")


if __name__ == "__main__":
    main()
