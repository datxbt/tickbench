"""Replicate Poudel (2025) families A, D and F on a point-in-time S&P 600 panel.

Split discipline mirrors the paper and then goes one step further:

  ``is``    2018-01-01 - 2021-12-31   the paper's in-sample; the exit sweep lives here
  ``oos``   2022-01-01 - 2024-12-31   the paper's out-of-sample; the headline claim
  ``post``  2025-01-01 - 2026-08-31   after the paper was written; spent only on a survivor

The sweep runs on ``is`` and nothing else. Whatever it picks is carried to
``oos`` unchanged, and ``post`` stays unopened unless ``--spend-post`` is passed,
so a strategy that dies out of sample leaves the genuinely unseen period intact.

Outputs, all under ``reports/strategies/``:
  ``smallcap_sweep_is.parquet``    the whole exit surface on the in-sample period
  ``smallcap_runs.parquet``        one row per (family, cost scenario, split)
  ``smallcap_trades_<tag>.parquet`` trade tapes for the headline runs
  ``smallcap_equity.parquet``      daily equity curves next to the benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qlab import paths  # noqa: E402
from qlab.strategies import smallcap as sc  # noqa: E402

DATA = REPO / "data" / "external" / "smallcap"
OUT = paths.STRATEGY_REPORT_DIR

SPLITS = {
    "is": ("2018-01-01", "2021-12-31"),
    "oos": ("2022-01-01", "2024-12-31"),
    "post": ("2025-01-01", "2026-08-31"),
}
WARMUP_DAYS = 260  # long enough for a 200-day feature to be warm at every split start


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(DATA / "bars_daily.parquet")
    bars["date"] = pd.to_datetime(bars["date"])
    bars = sc.adjust_ohlc(bars).dropna(subset=["close"])
    bars = bars[bars["close"] > 0]
    bars = bars.drop_duplicates(subset=["date", "ticker"], keep="last")

    membership = pd.read_parquet(DATA / "universe_membership.parquet")
    membership.index = pd.to_datetime(membership.index)

    bench = pd.read_parquet(DATA / "benchmarks.parquet")
    bench["date"] = pd.to_datetime(bench["date"])
    bench = sc.adjust_ohlc(bench)
    return bars, membership, bench


def benchmark_stats(close: pd.Series, start: str, end: str) -> dict:
    s = close.loc[start:end].dropna()
    r = s.pct_change().dropna()
    sd = r.std(ddof=1)
    years = len(r) / sc.TRADING_DAYS
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    eq = s / s.iloc[0]
    return dict(
        sharpe=float(r.mean() / sd * np.sqrt(sc.TRADING_DAYS)) if sd > 0 else np.nan,
        annual_return=float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
        total_return=total,
        max_drawdown=float((eq / eq.cummax() - 1).min()),
        days=len(r),
    )


def equal_weight_universe(ind: dict, eligible: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Benchmark 1: hold the eligible universe, equally weighted, rebalanced daily."""
    ret = ind["ret"].loc[start:end]
    mask = eligible.reindex(index=ret.index, columns=ret.columns).fillna(False)
    masked = ret.where(mask)
    return masked.mean(axis=1).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spend-post", action="store_true",
                    help="also evaluate the post-publication period")
    ap.add_argument("--quick", action="store_true", help="tiny sweep, for wiring checks")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("loading panel")
    bars, membership, bench = load_panel()
    print(f"  {bars['ticker'].nunique()} tickers, {len(bars):,} rows")

    ind = sc.indicators(bars)
    dates = ind["close"].index
    tickers = ind["close"].columns

    eligible = membership.reindex(index=dates, columns=tickers).fillna(False).astype(bool)
    print(f"  eligible name-days: {eligible.to_numpy().sum():,}")

    spy = bench[bench["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    iwm = bench[bench["ticker"] == "IWM"].set_index("date")["close"].sort_index()
    regime = sc.regime_risk_on(spy.reindex(dates).ffill())
    print(f"  risk-on share of days: {regime.mean():.1%}")

    fam_a, fam_d = sc.FamilyA(), sc.FamilyD()
    entries = {"A": fam_a.entries(ind), "D": fam_d.entries(ind)}
    for k, e in entries.items():
        print(f"  family {k}: {int((e & eligible).to_numpy().sum()):,} signal name-days")

    SIZING = {
        "paper": sc.Portfolio(size_mode="paper"),
        "vol_target": sc.Portfolio(size_mode="vol_target"),
    }
    portfolio = SIZING["vol_target"]
    costs = sc.COST_SCENARIOS["paper"]

    def window(frame, split):
        lo, hi = SPLITS[split]
        return frame.loc[lo:hi]

    # ---- exit sweep, in-sample only ------------------------------------
    targets = (1.0, 2.0) if args.quick else (0.5, 1.0, 1.5, 2.0, 3.0)
    stops = (2.0,) if args.quick else (1.0, 1.5, 2.0, 3.0)
    horizons = (10,) if args.quick else (5, 10, 20)

    sweeps = []
    for fam in ("A", "D"):
        print(f"sweeping exits for family {fam} on the in-sample period "
              f"({len(targets)*len(stops)*len(horizons)} configs)")
        sw = sc.sweep_exits(
            window(entries[fam], "is"), ind, costs=costs, portfolio=portfolio,
            targets=targets, stops=stops, horizons=horizons,
            eligible=window(eligible, "is"),
        )
        sw.insert(0, "family", fam)
        sweeps.append(sw)
        best = sw.sort_values("sharpe", ascending=False).iloc[0]
        print(f"  best IS sharpe {best['sharpe']:+.2f} at target={best['target_atr']} "
              f"stop={best['stop_atr']} horizon={int(best['time_stop'])} "
              f"({int(best['n_trades'])} trades); median {sw['sharpe'].median():+.2f}")
    sweep = pd.concat(sweeps, ignore_index=True)
    sweep.to_parquet(OUT / "smallcap_sweep_is.parquet", index=False)

    chosen = {
        fam: sweep[sweep["family"] == fam].sort_values("sharpe", ascending=False).iloc[0]
        for fam in ("A", "D")
    }

    # ---- carry the chosen exits to every split -------------------------
    splits = ["is", "oos"] + (["post"] if args.spend_post else [])
    rows, curves, tapes = [], {}, {}

    for split in splits:
        lo, hi = SPLITS[split]
        for fam in ("A", "D", "F"):
            if fam == "F":
                ent, cfg, reg = window(entries["A"] | entries["D"], split), chosen["A"], regime
            else:
                ent, cfg, reg = window(entries[fam], split), chosen[fam], None
            exits = dict(target_atr=float(cfg["target_atr"]),
                         stop_atr=float(cfg["stop_atr"]),
                         time_stop=int(cfg["time_stop"]))
            for size_name, pf in SIZING.items():
                scenarios = dict(sc.COST_SCENARIOS)
                scenarios["none"] = sc.EquityCosts(spread=0.0, slippage=0.0)
                for cost_name, cost in scenarios.items():
                    res = sc.simulate(ent, ind, costs=cost, portfolio=pf,
                                      eligible=window(eligible, split), regime=reg, **exits)
                    rows.append(dict(split=split, family=fam, sizing=size_name,
                                     costs=cost_name, start=lo, end=hi,
                                     **exits, **sc.summarise(res)))
                    if cost_name == "paper" and size_name == "vol_target":
                        curves[f"{fam}|{split}"] = res.equity
                        tapes[f"{fam}|{split}"] = res.trades
            headline = rows[-len(scenarios)]  # vol_target is second, so step back
            net = [r for r in rows if r["split"] == split and r["family"] == fam
                   and r["sizing"] == "vol_target" and r["costs"] == "paper"][0]
            free = [r for r in rows if r["split"] == split and r["family"] == fam
                    and r["sizing"] == "vol_target" and r["costs"] == "none"][0]
            paper_sized = [r for r in rows if r["split"] == split and r["family"] == fam
                           and r["sizing"] == "paper" and r["costs"] == "paper"][0]
            print(f"{split:5s} {fam}: sharpe {net['sharpe']:+.2f} net / {free['sharpe']:+.2f} "
                  f"cost-free, ret {net['annual_return']:+.1%}, dd {net['max_drawdown']:.1%}, "
                  f"{net['n_trades']} trades, exposure {net['avg_exposure']:.0%} "
                  f"| paper sizing: {paper_sized['sharpe']:+.2f} @ "
                  f"{paper_sized['avg_exposure']:.1%} exposure")

        # benchmarks over the same window
        ew = equal_weight_universe(ind, eligible, lo, hi)
        ew_eq = (1 + ew).cumprod()
        curves[f"EW|{split}"] = ew_eq
        for name, series in (("SPY", spy), ("IWM", iwm)):
            rows.append(dict(split=split, family=f"bench_{name}", sizing="-", costs="-",
                             start=lo, end=hi, n_trades=0,
                             **benchmark_stats(series, lo, hi)))
        sd = ew.std(ddof=1)
        rows.append(dict(
            split=split, family="bench_EW_universe", sizing="-", costs="-",
            start=lo, end=hi, n_trades=0,
            sharpe=float(ew.mean() / sd * np.sqrt(sc.TRADING_DAYS)) if sd > 0 else np.nan,
            annual_return=float(ew_eq.iloc[-1] ** (sc.TRADING_DAYS / len(ew)) - 1),
            total_return=float(ew_eq.iloc[-1] - 1),
            max_drawdown=float((ew_eq / ew_eq.cummax() - 1).min()), days=len(ew),
        ))
        print(f"{split:5s} benchmarks: SPY {rows[-3]['sharpe']:+.2f} "
              f"IWM {rows[-2]['sharpe']:+.2f} EW-universe {rows[-1]['sharpe']:+.2f}")

    runs = pd.DataFrame(rows)
    runs.to_parquet(OUT / "smallcap_runs.parquet", index=False)
    pd.DataFrame({k: v for k, v in curves.items()}).to_parquet(OUT / "smallcap_equity.parquet")
    for k, tape in tapes.items():
        if len(tape):
            tape.to_parquet(OUT / f"smallcap_trades_{k.replace('|', '_')}.parquet", index=False)

    (OUT / "smallcap_meta.json").write_text(json.dumps({
        "splits": SPLITS, "spent_post": args.spend_post,
        "tickers": int(bars["ticker"].nunique()),
        "chosen_exits": {k: {c: float(v[c]) for c in ("target_atr", "stop_atr", "time_stop")}
                         for k, v in chosen.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
