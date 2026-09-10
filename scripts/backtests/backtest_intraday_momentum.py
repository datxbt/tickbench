"""Backtest the Zarattini/Aziz/Barbon Noise Area intraday momentum strategy.

Usage
-----
``python scripts/backtests/backtest_intraday_momentum.py dev``          - sweep on dev
``python scripts/backtests/backtest_intraday_momentum.py validation``   - selected config
``python scripts/backtests/backtest_intraday_momentum.py test --unlock``- once, at the end

The dev pass sweeps the two knobs the paper leaves free (the stop rule and the
first decision time), across every instrument in the corpus. The validation pass
runs the single configuration dev selected, and nothing else. The test pass
refuses to run without ``--unlock``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, time, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel
from qlab.loader import SPLITS, load_bars
from qlab.metrics import (
    format_returns_tearsheet,
    tearsheet_from_returns,
)
import numpy as np

from qlab.stats import bootstrap_ci, ols_hac, sharpe, sharpe_with_se
from qlab.strategies import intraday_momentum as im

SYMBOLS = ("USTEC", "XAUUSD", "EURUSD", "USDJPY")
OUT = Path("reports/strategies")

# 30 sessions of warmup covers the 14-day sigma and the 14-day vol estimate
# with room for holidays, so the first evaluable session of a split is the
# split's own first session and not a month into it.
WARMUP = timedelta(days=45)


def _cost(symbol: str, split: str) -> CostModel:
    # Cost the split with the split's own spreads. Spread on USTEC has roughly
    # halved since 2020, so a whole-corpus profile is wrong at both ends.
    return CostModel.from_profiles(symbol, split=split)


def _load(symbol: str, split: str, allow_test: bool) -> pl.DataFrame:
    return load_bars(
        symbol,
        "1m",
        split=split,
        warmup=WARMUP,
        allow_test=allow_test,
        columns=["ts", "open", "high", "low", "close", "n_ticks"],
    )


def _evaluable(daily: pl.DataFrame, split: str) -> pl.DataFrame:
    """Drop the warmup sessions, which exist only to warm sigma."""
    s = SPLITS[split]
    return daily.filter(pl.col("session") >= s.start)


def sweep(split: str, allow_test: bool = False) -> pl.DataFrame:
    rows = []
    for symbol in SYMBOLS:
        bars = _load(symbol, split, allow_test)
        cost = _cost(symbol, split)
        for stop in im.STOP_RULES:
            for first in (time(10, 0), time(10, 30), time(11, 0), time(12, 0)):
                for vt in (None, 0.02):
                    cfg = im.NoiseAreaConfig(
                        stop=stop, first_decision=first, vol_target=vt
                    )
                    daily, legs = im.run(bars, cost, cfg)
                    daily = _evaluable(daily, split)
                    if daily.height < 100:
                        continue
                    sheet = tearsheet_from_returns(daily["net_bps"])
                    gross = tearsheet_from_returns(daily["gross_bps"])
                    rows.append(
                        {
                            "symbol": symbol,
                            "stop": stop,
                            "first": first.strftime("%H:%M"),
                            "sizing": "vol-target" if vt else "flat",
                            "sessions": sheet["periods"],
                            "legs_day": round(legs.height / daily.height, 2),
                            "gross_sharpe": round(gross["sharpe"], 2),
                            "sharpe": round(sheet["sharpe"], 2),
                            "cagr_pct": round(sheet["cagr_pct"], 2),
                            "vol_pct": round(sheet["ann_vol_pct"], 1),
                            "mdd_pct": round(sheet["max_dd_pct"], 1),
                            "hit_pct": round(sheet["hit_rate_pct"], 1),
                            "cost_bps_day": round(float(daily["cost_bps"].mean()), 2),
                            "t_stat": round(sheet["t_stat"], 2),
                        }
                    )
    return pl.DataFrame(rows)


def detail(symbol: str, split: str, cfg: im.NoiseAreaConfig, allow_test: bool = False) -> dict:
    """Everything worth saying about one configuration on one instrument."""
    bars = _load(symbol, split, allow_test)
    cost = _cost(symbol, split)
    daily, legs = im.run(bars, cost, cfg)
    daily = _evaluable(daily, split)
    legs = legs.filter(pl.col("session") >= SPLITS[split].start)
    bh = _evaluable(im.buy_and_hold(bars, cfg).rename({"bh_bps": "net_bps"}), split)

    sheet = tearsheet_from_returns(daily["net_bps"], label=f"{symbol} {split}")
    bh_sheet = tearsheet_from_returns(bh["net_bps"], label=f"{symbol} buy&hold")

    net = daily["net_bps"].to_numpy() / 1e4
    hac = sharpe_with_se(net)
    rng = np.random.default_rng(11)
    boot = bootstrap_ci(net, stat_fn=lambda a: sharpe(a), n_boot=2000, rng=rng)

    # Alpha and beta against holding the thing, the paper's own regression.
    joined = daily.select("session", "net_bps").join(
        bh.rename({"net_bps": "bh_bps"}), on="session", how="inner"
    )
    reg = ols_hac(
        joined["net_bps"].to_numpy() / 1e4,
        joined["bh_bps"].to_numpy().reshape(-1, 1) / 1e4,
        names=["beta"],
    )

    placebo = im.placebo(bars, cost, cfg, draws=400)
    p_placebo = float((placebo["sharpe"].to_numpy() >= sheet["sharpe"]).mean())

    return {
        "symbol": symbol,
        "split": split,
        "sheet": sheet,
        "bh_sheet": bh_sheet,
        "sharpe_se": hac.se,
        "sharpe_t": hac.t_stat,
        "sharpe_ci": (boot.lo, boot.hi),
        "boot_p": boot.p_value,
        "alpha_ann_pct": 100.0 * reg.params[0] * 252,
        "alpha_t": reg.t_stats[0],
        "beta": reg.params[1],
        "beta_t": reg.t_stats[1],
        "placebo_p": p_placebo,
        "placebo_mean_sharpe": float(placebo["sharpe"].mean()),
        "legs_per_day": legs.height / max(daily.height, 1),
        "exit_reasons": im.exit_reason_table(legs),
        "entry_hours": im.entry_hour_table(legs),
        "sides": im.side_table(legs),
        "daily": daily,
        "bh": bh,
        "legs": legs,
    }


def _print_detail(d: dict) -> None:
    print(format_returns_tearsheet(d["sheet"]))
    print(format_returns_tearsheet(d["bh_sheet"]))
    print(
        f"  Sharpe {d['sheet']['sharpe']:.2f} +/- {d['sharpe_se']:.2f} (HAC), "
        f"bootstrap 95% CI [{d['sharpe_ci'][0]:.2f}, {d['sharpe_ci'][1]:.2f}], "
        f"p={d['boot_p']:.3f}"
    )
    print(
        f"  alpha {d['alpha_ann_pct']:+.2f}%/yr (t={d['alpha_t']:+.2f}), "
        f"beta {d['beta']:+.3f} (t={d['beta_t']:+.2f})"
    )
    print(
        f"  sign placebo: real Sharpe beats {100 * (1 - d['placebo_p']):.1f}% of "
        f"400 sign-randomised draws (mean {d['placebo_mean_sharpe']:+.2f})"
    )
    print(f"  legs/day {d['legs_per_day']:.2f}")
    print(d["exit_reasons"])
    print(d["sides"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "validation", "test"])
    ap.add_argument("--unlock", action="store_true", help="required for test")
    ap.add_argument("--stop", default="band_vwap")
    ap.add_argument("--first", default="10:00")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    allow_test = args.split == "test"
    if allow_test and not args.unlock:
        print("test split is locked; pass --unlock and mean it", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        table = sweep(args.split, allow_test)
        with pl.Config(tbl_rows=200, tbl_width_chars=200):
            print(table.sort("symbol", "sharpe", descending=[False, True]))
        table.write_csv(OUT / f"intraday_momentum_sweep_{args.split}.csv")
        return 0

    hh, mm = (int(x) for x in args.first.split(":"))
    cfg = im.NoiseAreaConfig(stop=args.stop, first_decision=time(hh, mm))
    for symbol in args.symbols.split(","):
        d = detail(symbol, args.split, cfg, allow_test)
        print(f"\n{'=' * 70}\n{symbol}  {args.split}  stop={args.stop} first={args.first}")
        _print_detail(d)
        d["daily"].write_parquet(OUT / f"im_daily_{symbol}_{args.split}.parquet")
        d["legs"].write_parquet(OUT / f"im_legs_{symbol}_{args.split}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
