"""The 15-minute opening range, pre-registered in ``opening-range-breakout.md``.

That report found the 15-minute range beat the paper's 5 minutes on both
splits (+0.223 R dev, +0.202 R validation) and named it "the one
pre-registerable follow-up": fix it, test it once on ``test``. This script does
exactly that and nothing else. The rule is the paper's in every other respect -
the sign of the opening range's candle picks the side, a stop order at the
range edge, a protective stop 10% of the 14-day ATR away, hold to 16:00 New
York - resolved on ticks by ``qlab.strategies.opening_range``.

Two modes, so the locked split cannot be read by accident:

    python scripts/backtests/backtest_orb15.py          # dev + validation, with controls
    python scripts/backtests/backtest_orb15.py --test   # the single test run

Default mode reproduces the cell and runs the controls the 5-minute study
showed matter: take both sides, and a shuffled-sign placebo that keeps the
geometry and removes the direction. The 5-minute rule was mostly geometry
(random signs earned +0.133 R on dev) and that geometry went to zero on
validation, so the placebo is read before anything else.

Test-mode protocol, fixed 2026-09-11 before the split was read:

* Rule: ``ORBConfig(or_minutes=15)`` with every other field at its default.
* PASS: test net mean R > 0 with one-sided Newey-West t > 1.28 (p < 0.10).
  At +0.20 R and a per-trade sd near 2.3 R, ~290 test sessions give an expected
  t near 1.5, so this is a low-power experiment and says so in advance.
* FAIL: test net mean R <= 0. Between the two: inconclusive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.stats import newey_west  # noqa: E402
from qlab.strategies.opening_range import ORBConfig, daily_context, placebo_signal, run  # noqa: E402

SYMBOL = "USTEC"
CFG = ORBConfig(or_minutes=15)
OUT = paths.STRATEGY_REPORT_DIR / "orb15.json"


def stats(trades: pl.DataFrame) -> dict:
    x = trades["net_r"].drop_nulls().to_numpy()
    x = x[np.isfinite(x)]
    return {"n": int(x.size), "mean_r": float(x.mean()), "sd_r": float(x.std(ddof=1)),
            "t_hac": float(newey_west(x).t_stat), "hit": float((x > 0).mean()),
            "mid_r": float(trades["mid_r"].mean())}


def show(label: str, s: dict) -> None:
    print(f"  {label:<34} n={s['n']:>4}  mean {s['mean_r']:+.3f} R  t(HAC) {s['t_hac']:+.2f}"
          f"  hit {100 * s['hit']:.1f}%  mid {s['mid_r']:+.3f} R")


def dev_validation(draws: int) -> dict:
    out: dict = {}
    pooled = []
    for split in ("dev", "validation"):
        costs = CostModel.from_profiles(SYMBOL, split=split)
        ctx = daily_context(SYMBOL, CFG, split=split)
        rule = run(SYMBOL, CFG, ctx=ctx, split=split, costs=costs)
        both = run(SYMBOL, ORBConfig(or_minutes=15, direction_rule="both"), ctx=ctx, split=split, costs=costs)
        print(f"\n== {split}")
        s = stats(rule)
        show("15m rule", s)
        show("control: both sides", stats(both))
        null = []
        for seed in range(draws):
            fake = run(SYMBOL, CFG, ctx=placebo_signal(ctx, seed=seed), split=split, costs=costs)
            if not fake.is_empty():
                null.append(float(fake["net_r"].mean()))
        arr = np.array(null)
        beaten = int((arr >= s["mean_r"]).sum())
        print(f"  placebo (shuffled sign) {arr.size} draws: mean {arr.mean():+.3f} R,"
              f" 5-95% [{np.percentile(arr, 5):+.3f}, {np.percentile(arr, 95):+.3f}],"
              f" beaten by {beaten} -> p {(beaten + 1) / (arr.size + 1):.3f}")
        out[split] = {"rule": s, "both": stats(both), "placebo_mean": float(arr.mean()),
                      "placebo_p": (beaten + 1) / (arr.size + 1)}
        pooled.append(rule)
    p = stats(pl.concat(pooled, how="vertical_relaxed"))
    print()
    show("POOLED dev + validation", p)
    out["pooled"] = p
    return out


def test_once() -> dict:
    print("!! reading the locked test split, once, under the protocol in the docstring\n")
    costs = CostModel.from_profiles(SYMBOL, split="test")
    ctx = daily_context(SYMBOL, CFG, split="test", allow_test=True)
    trades = run(SYMBOL, CFG, ctx=ctx, split="test", allow_test=True, costs=costs)
    s = stats(trades)
    show("15m rule on test", s)
    verdict = ("PASS" if s["mean_r"] > 0 and s["t_hac"] > 1.28
               else "FAIL" if s["mean_r"] <= 0 else "INCONCLUSIVE")
    print(f"\nVerdict under the pre-registered rule: {verdict}")
    yearly = (trades.with_columns(y=pl.col("nyd").dt.year()).group_by("y")
              .agg(n=pl.len(), mean=pl.col("net_r").mean()).sort("y"))
    for row in yearly.iter_rows(named=True):
        print(f"    {row['y']}  n {row['n']:>4}  mean {row['mean']:+.3f} R")
    return {"test": s, "verdict": verdict}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--test", action="store_true", help="run the single locked-split evaluation")
    p.add_argument("--placebo-draws", type=int, default=30)
    args = p.parse_args()
    result = test_once() if args.test else dev_validation(args.placebo_draws)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    target = OUT.with_name("orb15_test.json" if args.test else OUT.name)
    target.write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
