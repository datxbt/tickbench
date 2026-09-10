"""Test the decision-tree intraday strategy of Prajwal et al. (2024).

The paper (SSRN 4838381) fits one depth-4 Gini decision tree per name on nine
technical features of 1-minute bars, labels each bar by the sign of the next
bar's return, shifts the signal forward one bar for execution delay, and trades
it. On NIFTY50 constituents it reports a **test-set Sharpe of 5.81** against a
buy-and-hold benchmark's 2.36, with a 2.63% maximum drawdown against 7.02%.

Its Section IV.C also says: *"commissions and slippage were not explicitly
incorporated into the backtesting process."* A next-bar classifier on 1-minute
data turns its position over hundreds of times a day, so that sentence is the
whole study. This script therefore runs everything twice - once cost-free, which
is what the paper measured, and once through the measured cost model.

Six passes:

1. **Does the classifier predict anything?** Directional accuracy against the
   majority-class base rate, per instrument. A tree that has learned only the
   drift beats 50% while predicting a constant, so 50% is not the bar.
2. **The paper's own backtest, reproduced.** Cost-free, its annualisation, its
   forward shift - the closest thing to their Table 1 this corpus can produce.
3. **The cost that was not charged.** The breakeven round-turn cost implied by
   the gross edge, against what the round turn actually costs. This is the
   number the verdict rests on.
4. **Tree depth.** Their Figures 1-3 compare depths 3 to 6 and settle on 4.
5. **Execution and mapping.** The forward shift against acting on the predicted
   bar, and long/flat against long/short.
6. **Per-instrument spread**, which is their PSBBR / PSBBS statistic on four
   names instead of fifty.

Run:  python scripts/backtests/backtest_decision_tree.py
      python scripts/backtests/backtest_decision_tree.py --symbols XAUUSD USTEC
      python scripts/backtests/backtest_decision_tree.py --test-split          # locked
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
from qlab.metrics import (  # noqa: E402
    format_returns_tearsheet,
    tearsheet_from_returns,
)
from qlab.stats import bootstrap_ci, sharpe_with_se  # noqa: E402
from qlab.strategies.decision_tree import (  # noqa: E402
    DEPTHS,
    FEATURES,
    TreeConfig,
    accuracy,
    benchmark,
    breakeven_cost_bps,
    clean,
    daily,
    describe_tree,
    features,
    fit,
    paper_style_sharpe,
    positions,
    returns,
    used_features,
)

REPORT_DIR = paths.STRATEGY_REPORT_DIR
SYMBOLS = ("XAUUSD", "USTEC", "EURUSD", "USDJPY")

#: The paper's Table 1, test-set column.
PAPER = {
    "sharpe": 5.81, "benchmark_sharpe": 2.36,
    "total_pct": 28.62, "benchmark_total_pct": 34.23,
    "cagr_pct": 19.33, "max_dd_pct": -2.63, "benchmark_max_dd_pct": -7.02,
    "win_rate_pct": 48.15, "profit_factor": 1.11,
    "vol_pct": 3.32, "benchmark_vol_pct": 10.23,
    "psbbr_pct": 46.0, "psbbs_pct": 60.0,
}


def evaluate(day: pl.DataFrame, label: str, column: str = "net_bps") -> dict:
    """A returns tearsheet plus a Lo standard error on the Sharpe."""
    sheet = tearsheet_from_returns(day[column], label=label)
    r = day[column].to_numpy() / 1e4
    hac = sharpe_with_se(r)
    sheet |= {"sharpe_se": hac.se, "sharpe_t": hac.t_stat, "sharpe_p": hac.p_value}
    return sheet


def one_symbol(symbol: str, cfg: TreeConfig, args, *, allow_test: bool) -> dict:
    """Fit on dev, measure on the evaluation split, at every level of detail."""
    eval_split = "test" if allow_test else "validation"
    train = clean(features(symbol, cfg, split="dev"))
    held = clean(features(symbol, cfg, split=eval_split, allow_test=allow_test))
    if train.is_empty() or held.is_empty():
        return {"symbol": symbol, "rows": 0}

    costs = CostModel.from_profiles(symbol, split=eval_split)
    model = fit(train, cfg)
    held = positions(model, held, cfg)
    priced = returns(held, symbol, costs=costs, split=eval_split)
    day = daily(priced)
    bench = daily(returns(benchmark(held), symbol, costs=costs, split=eval_split))

    acc = accuracy(held)
    in_sample = accuracy(positions(model, train, cfg))
    out = {
        "symbol": symbol,
        "train_rows": train.height,
        "eval_rows": held.height,
        "eval_split": eval_split,
        "used_features": used_features(model),
        "accuracy": acc,
        "in_sample_accuracy": in_sample,
        "gross": evaluate(day, f"{symbol} gross (paper: no costs)", "gross_bps"),
        "net": evaluate(day, f"{symbol} net of measured costs", "net_bps"),
        "benchmark": evaluate(bench, f"{symbol} buy and hold", "net_bps"),
        "paper_style_sharpe_gross": paper_style_sharpe(priced, column="gross_bps"),
        "paper_style_sharpe_net": paper_style_sharpe(priced),
        "breakeven_round_turn_bps": breakeven_cost_bps(priced),
        "actual_round_turn_bps": 2.0 * float(priced["cost_bps"].sum())
        / float(priced["turnover"].sum()) if float(priced["turnover"].sum()) > 0
        else float("nan"),
        "turnover_per_day": float(day["turnover"].mean()),
        "exposure": float(day["exposure"].mean()),
        "tree": describe_tree(model),
    }
    boot = bootstrap_ci(day["gross_bps"].to_numpy(), n_boot=args.bootstrap)
    out["gross_daily_boot"] = {"mean": boot.point, "lo": boot.lo, "hi": boot.hi,
                               "p": boot.p_value}
    return out


# --------------------------------------------------------------------------
# The passes
# --------------------------------------------------------------------------

def pass_accuracy(results: list[dict]) -> None:
    print(f"\n{'=' * 78}\n1. DOES THE CLASSIFIER PREDICT ANYTHING?\n{'=' * 78}")
    print("  Directional accuracy on the held-out split against the majority-class")
    print("  base rate. Beating 50% is not the bar: on a drifting instrument the")
    print("  up-bar share is not a half, and a tree that learned only the drift")
    print("  would clear 50% while predicting one constant.\n")
    print(f"  {'symbol':<8} {'in-sample':>10} {'held-out':>10} {'base rate':>10} "
          f"{'edge':>9}  {'features used':>14}")
    for r in results:
        if not r.get("eval_rows"):
            continue
        a, i = r["accuracy"], r["in_sample_accuracy"]
        print(f"  {r['symbol']:<8} {100 * i['accuracy']:>9.3f}% "
              f"{100 * a['accuracy']:>9.3f}% {100 * a['base_rate']:>9.3f}% "
              f"{100 * a['edge_vs_base']:>+8.3f}%  {len(r['used_features']):>8} of 9")


def pass_paper_backtest(results: list[dict]) -> None:
    print(f"\n{'=' * 78}\n2. THE PAPER'S OWN BACKTEST, REPRODUCED (COST-FREE)\n{'=' * 78}")
    print("  Their execution rule, their cost assumption (none), their")
    print("  annualisation. Sharpe here is on daily returns and annualised by 252,")
    print("  which is the house convention; the 94,500-period column is theirs.\n")
    print(f"  {'symbol':<8} {'Sharpe':>8} {'(their ann.)':>13} {'total %':>9} "
          f"{'maxDD %':>9} {'vol %':>8}  |  {'B&H Sharpe':>10} {'B&H tot %':>10}")
    for r in results:
        if not r.get("eval_rows"):
            continue
        g, b = r["gross"], r["benchmark"]
        print(f"  {r['symbol']:<8} {g['sharpe']:>8.2f} "
              f"{r['paper_style_sharpe_gross']:>13.2f} {g['total_pct']:>9.1f} "
              f"{g['max_dd_pct']:>9.1f} {g['ann_vol_pct']:>8.2f}  |  "
              f"{b['sharpe']:>10.2f} {b['total_pct']:>10.1f}")
    print(f"\n  the paper's test-set row, for reference:")
    print(f"  {'NIFTY50':<8} {PAPER['sharpe']:>8.2f} {'':>13} "
          f"{PAPER['total_pct']:>9.1f} {PAPER['max_dd_pct']:>9.1f} "
          f"{PAPER['vol_pct']:>8.2f}  |  {PAPER['benchmark_sharpe']:>10.2f} "
          f"{PAPER['benchmark_total_pct']:>10.1f}")


def pass_costs(results: list[dict]) -> None:
    print(f"\n{'=' * 78}\n3. THE COST THAT WAS NOT CHARGED\n{'=' * 78}")
    print("  'Breakeven' is the round-turn cost at which the gross edge is exactly")
    print("  consumed: total gross return divided by total turnover. 'Actual' is")
    print("  what the round turn costs on this account, from the measured model.\n")
    print(f"  {'symbol':<8} {'turns/day':>10} {'breakeven bps':>14} "
          f"{'actual bps':>12} {'ratio':>8}  |  {'net Sharpe':>11} {'net total %':>12}")
    for r in results:
        if not r.get("eval_rows"):
            continue
        be, ac = r["breakeven_round_turn_bps"], r["actual_round_turn_bps"]
        ratio = ac / be if be and be > 0 else float("inf")
        print(f"  {r['symbol']:<8} {r['turnover_per_day']:>10.1f} {be:>14.4f} "
              f"{ac:>12.4f} {ratio:>8.1f}x  |  {r['net']['sharpe']:>11.2f} "
              f"{r['net']['total_pct']:>12.1f}")
    print("\n  'ratio' is how many times more expensive trading actually is than")
    print("  the strategy can afford. Anything above 1.0 means the paper's result")
    print("  exists only because the cost line was left out.")


def pass_depth(symbol: str, cfg: TreeConfig, args, *, allow_test: bool) -> dict:
    print(f"\n{'=' * 78}\n4. TREE DEPTH ({symbol})\n{'=' * 78}")
    print("  The paper's Figures 1-3 compare depths 3 to 6 and settle on 4. If the")
    print("  edge is real, depth should trade off bias against variance; if there")
    print("  is no edge, depth just moves the turnover around.\n")
    eval_split = "test" if allow_test else "validation"
    train = clean(features(symbol, cfg, split="dev"))
    held = clean(features(symbol, cfg, split=eval_split, allow_test=allow_test))
    costs = CostModel.from_profiles(symbol, split=eval_split)
    out = {}
    print(f"  {'depth':>6} {'held-out acc':>13} {'gross Sharpe':>13} "
          f"{'net Sharpe':>11} {'turns/day':>10} {'breakeven bps':>14}")
    for depth in DEPTHS:
        c = TreeConfig(**(cfg.__dict__ | {"max_depth": depth}))
        model = fit(train, c)
        pos = positions(model, held, c)
        priced = returns(pos, symbol, costs=costs, split=eval_split)
        day = daily(priced)
        g = tearsheet_from_returns(day["gross_bps"])
        n = tearsheet_from_returns(day["net_bps"])
        acc = accuracy(pos)
        row = {"accuracy": acc["accuracy"], "gross_sharpe": g["sharpe"],
               "net_sharpe": n["sharpe"], "turnover_per_day": float(day["turnover"].mean()),
               "breakeven_bps": breakeven_cost_bps(priced)}
        out[str(depth)] = row
        print(f"  {depth:>6} {100 * row['accuracy']:>12.3f}% "
              f"{row['gross_sharpe']:>13.2f} {row['net_sharpe']:>11.2f} "
              f"{row['turnover_per_day']:>10.1f} {row['breakeven_bps']:>14.4f}")
    return out


def pass_execution(symbol: str, cfg: TreeConfig, args, *, allow_test: bool) -> dict:
    print(f"\n{'=' * 78}\n5. EXECUTION AND MAPPING ({symbol})\n{'=' * 78}")
    print("  The paper shifts the signal forward one bar, which means the model")
    print("  predicts bar t+1 and the position is held over t+2 - the edge, if")
    print("  there were one, is applied to the wrong bar. lag=0 acts on the bar")
    print("  actually predicted and needs a fill at the closing print.\n")
    eval_split = "test" if allow_test else "validation"
    train = clean(features(symbol, cfg, split="dev"))
    held = clean(features(symbol, cfg, split=eval_split, allow_test=allow_test))
    costs = CostModel.from_profiles(symbol, split=eval_split)
    model = fit(train, cfg)
    out = {}
    print(f"  {'lag':>4} {'mapping':>11} {'gross Sharpe':>13} {'gross tot %':>12} "
          f"{'net Sharpe':>11} {'turns/day':>10}")
    for lag in (0, 1):
        for mapping in ("long_flat", "long_short"):
            c = TreeConfig(**(cfg.__dict__ | {"signal_lag": lag, "mapping": mapping}))
            priced = returns(positions(model, held, c), symbol, costs=costs,
                             split=eval_split)
            day = daily(priced)
            g = tearsheet_from_returns(day["gross_bps"])
            n = tearsheet_from_returns(day["net_bps"])
            key = f"lag{lag}:{mapping}"
            out[key] = {"gross_sharpe": g["sharpe"], "gross_total_pct": g["total_pct"],
                        "net_sharpe": n["sharpe"],
                        "turnover_per_day": float(day["turnover"].mean())}
            star = "  <- the paper" if lag == 1 and mapping == "long_flat" else ""
            print(f"  {lag:>4} {mapping:>11} {g['sharpe']:>13.2f} "
                  f"{g['total_pct']:>12.1f} {n['sharpe']:>11.2f} "
                  f"{float(day['turnover'].mean()):>10.1f}{star}")
    return out


def pass_spread(results: list[dict]) -> dict:
    print(f"\n{'=' * 78}\n6. PER-INSTRUMENT SPREAD (THEIR PSBBR / PSBBS)\n{'=' * 78}")
    print("  The share of names beating their own buy-and-hold benchmark, on")
    print("  return and on Sharpe. Four instruments is not fifty, and a share out")
    print("  of four moves 25 points at a time - so the count is printed too.\n")
    live = [r for r in results if r.get("eval_rows")]
    n = len(live)
    gross_ret = sum(r["gross"]["total_pct"] > r["benchmark"]["total_pct"] for r in live)
    gross_shp = sum(r["gross"]["sharpe"] > r["benchmark"]["sharpe"] for r in live)
    net_ret = sum(r["net"]["total_pct"] > r["benchmark"]["total_pct"] for r in live)
    net_shp = sum(r["net"]["sharpe"] > r["benchmark"]["sharpe"] for r in live)
    print(f"  {'':<22} {'count':>10} {'share':>8}   paper (50 names)")
    print(f"  {'PSBBR, cost-free':<22} {f'{gross_ret}/{n}':>10} "
          f"{100 * gross_ret / n:>7.0f}%   {PAPER['psbbr_pct']:.0f}%")
    print(f"  {'PSBBS, cost-free':<22} {f'{gross_shp}/{n}':>10} "
          f"{100 * gross_shp / n:>7.0f}%   {PAPER['psbbs_pct']:.0f}%")
    print(f"  {'PSBBR, net of costs':<22} {f'{net_ret}/{n}':>10} "
          f"{100 * net_ret / n:>7.0f}%   not reported")
    print(f"  {'PSBBS, net of costs':<22} {f'{net_shp}/{n}':>10} "
          f"{100 * net_shp / n:>7.0f}%   not reported")
    return {"n": n, "psbbr_gross": gross_ret, "psbbs_gross": gross_shp,
            "psbbr_net": net_ret, "psbbs_net": net_shp}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    p.add_argument("--depth-symbol", default="XAUUSD")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--test-split", action="store_true",
                   help="evaluate on the locked test split instead of validation")
    p.add_argument("-o", "--out", default=str(REPORT_DIR / "decision_tree.json"))
    args = p.parse_args(argv)

    if args.test_split:
        print("!! the test split is locked; it is being read deliberately\n")

    cfg = TreeConfig()
    print(f"{'#' * 78}\n#  DECISION-TREE INTRADAY STRATEGY  --  Prajwal et al. (2024)\n"
          f"{'#' * 78}")
    print(f"  depth {cfg.max_depth}, {cfg.criterion} impurity, {len(FEATURES)} features, "
          f"signal lag {cfg.signal_lag}, {cfg.mapping}")
    print(f"  trained on the dev split, measured on "
          f"{'test' if args.test_split else 'validation'}")

    results = []
    for symbol in args.symbols:
        try:
            results.append(one_symbol(symbol, cfg, args, allow_test=args.test_split))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {symbol}: skipped ({exc})")

    pass_accuracy(results)
    pass_paper_backtest(results)
    pass_costs(results)
    depth = pass_depth(args.depth_symbol, cfg, args, allow_test=args.test_split)
    execution = pass_execution(args.depth_symbol, cfg, args, allow_test=args.test_split)
    spread = pass_spread(results)

    print(f"\n{'=' * 78}\nFITTED TREE ({args.depth_symbol})\n{'=' * 78}")
    for r in results:
        if r["symbol"] == args.depth_symbol and r.get("tree"):
            print(r["tree"])
            break

    for r in results:
        if not r.get("eval_rows"):
            continue
        print(f"\n{format_returns_tearsheet(r['gross'])}")
        print(format_returns_tearsheet(r["net"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": cfg.__dict__, "paper": PAPER, "symbols": results,
         "depth": depth, "execution": execution, "spread": spread},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
