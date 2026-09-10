"""Evaluate the engulfing candle across four instruments, eleven timeframes and
a whole family of exits.

The idea under test: a candle whose body engulfs the previous candle's body
predicts continuation. Enter at its close, stop at its own extreme, target a
fixed multiple of that risk, and reverse when the opposite pattern prints.

That claim is testable in four separate places, and this script tests them
separately because they fail for different reasons:

1. **Does the pattern predict anything?** Unconditional forward return from the
   entry mid, signed by the signal's direction, in basis points - no stop, no
   target, no cost. Measured against a control that keeps the entry geometry
   and drops only the engulfment. If the two agree, the pattern is decoration.
2. **Does any exit make money before costs?** The full target curve from 0.5R
   to 6R plus the no-target variant, resolved on the same fills, mid to mid.
   An edge at exactly one multiple is a fitted exit.
3. **Does anything survive costs?** 1R here is one bar's range by construction,
   so 1R shrinks with the timeframe while the round turn does not. ``cost_r``
   is the round turn as a fraction of the risk taken, and on fast timeframes it
   is larger than any plausible edge.
4. **What does it do to an account?** The size is fixed at 0.01 lot, so dollar
   risk per trade is *not* constant and the equity curve is dominated by the
   widest-stop trades. R and USD are therefore both reported; they do not rank
   the timeframes the same way.

Everything in R is in multiples of the planned risk - the signal candle's close
to its own extreme - which is the only unit in which a 1.9 bps EURUSD 1m stop
and a 31 bps gold 4h stop are the same bet.

Run:  python scripts/backtests/backtest_engulfing.py [-s SYMBOL ...] [--control]
      python scripts/backtests/backtest_engulfing.py --no-reverse   [--placebo]
      python scripts/backtests/backtest_engulfing.py --splits test  # locked; needs a survivor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.strategies.engulfing import (  # noqa: E402
    ALL_INTERVALS,
    EXIT_KEYS,
    HOLD,
    TARGET_MULTIPLES,
    EngulfingConfig,
    run,
    sequence,
    tag,
)
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "engulfing_trades"

EXIT_LABELS: dict[str, str] = {
    **{tag(m): f"{m:g}R" for m in TARGET_MULTIPLES},
    HOLD: "hold",
}
START_EQUITY = 10_000.0   # only used to express the fixed-lot drawdown as a %


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def daily_t(trades: pl.DataFrame, column: str) -> tuple[float, float, int]:
    """(t-statistic, mean per day, n days) of a per-trade column, by day.

    Daily rather than per trade, deliberately. Four instruments and eleven
    timeframes fire on the same impulse - a 4h engulfing candle contains the 1h
    one that helped form it - and a per-trade t-statistic would count that
    overlap as independent evidence, overstating significance by roughly the
    square root of the number of correlated streams. Days are the coarsest unit
    that still leaves enough of them to measure.
    """
    if trades.is_empty() or column not in trades.columns:
        return float("nan"), float("nan"), 0
    daily = trades.group_by("day").agg(v=pl.col(column).sum()).sort("day")["v"]
    n = daily.len()
    if n < 2:
        return float("nan"), float("nan"), n
    sd, mean = float(daily.std()), float(daily.mean())
    if sd == 0:
        return float("nan"), mean, n
    return mean / (sd / np.sqrt(n)), mean, n


def max_drawdown(trades: pl.DataFrame, column: str) -> float:
    """Peak-to-trough of the cumulative curve, in that column's own units."""
    if trades.is_empty() or column not in trades.columns:
        return 0.0
    curve = trades.sort("arm_ts")[column].fill_null(0.0).cum_sum().to_numpy()
    if curve.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(curve) - curve))


def summarise(tape: pl.DataFrame) -> dict:
    """Cost structure, signal quality and every exit, for one symbol-timeframe.

    Signal-level statistics are computed over every signal. Exit-level
    statistics are computed over the trades an account *actually takes* under
    that exit, which is a different and smaller set for every exit rule,
    because when a trade ends decides which later signals it was flat for.
    """
    out: dict = {"signals": tape.height}
    if tape.is_empty():
        return out

    out["risk_bps_median"] = float(tape["risk_bps"].median())
    out["cost_r_median"] = float(tape["cost_r"].median())
    out["cost_r_mean"] = float(tape["cost_r"].mean())
    out["commission_r_median"] = float(tape["commission_r"].median())
    out["risk_usd_median"] = float(tape["risk_usd"].median())
    out["instant_stop_rate"] = float(tape["instant_stop"].mean())
    out["stop_rate"] = float(tape["stopped"].mean())
    out["mfe_pre_stop_r_median"] = float(tape["mfe_pre_stop_r"].median())
    out["mfe20_r_median"] = float(tape["mfe20_r"].median())
    out["mae20_r_median"] = float(tape["mae20_r"].median())

    # The pattern on its own: mid to mid, no stop, no target, no cost. In bps
    # as well as R, because R is a different size on every timeframe and only
    # bps compares the prediction itself across them.
    out["days"] = tape["day"].n_unique()
    for k in (1, 5, 20):
        t, _, _ = daily_t(tape, f"fwd{k}_bps")
        out[f"fwd{k}_bps_mean"] = float(tape[f"fwd{k}_bps"].mean())
        out[f"fwd{k}_bps_t"] = t
        out[f"fwd{k}_r_mean"] = float(tape[f"fwd{k}_r"].mean())

    exits: dict[str, dict] = {}
    for key in EXIT_KEYS:
        col = f"r_{key}"
        if col not in tape.columns:
            continue
        taken = sequence(tape, key)
        if taken.is_empty():
            continue
        t, _, _ = daily_t(taken, col)
        usd = float(taken[f"usd_{key}"].sum())
        exits[key] = {
            "trades": taken.height,
            "mean_r": float(taken[col].mean()),
            "total_r": float(taken[col].sum()),
            "t": t,
            "max_dd_r": max_drawdown(taken, col),
            "mean_r_mid": float(taken[f"{col}_mid"].mean()),
            "total_r_mid": float(taken[f"{col}_mid"].sum()),
            "win_rate": float((taken[f"reason_{key}"] == "tp").mean()),
            "stop_rate": float((taken[f"reason_{key}"] == "stop").mean()),
            "reverse_rate": float((taken[f"reason_{key}"] == "reverse").mean()),
            "time_rate": float((taken[f"reason_{key}"] == "time").mean()),
            "usd_total": usd,
            "usd_max_dd": max_drawdown(taken, f"usd_{key}"),
            "usd_per_trade": usd / taken.height,
        }
    out["exits"] = exits
    return out


def by_group(tape: pl.DataFrame, intervals) -> list[dict]:
    """``summarise`` for every (symbol, interval), in a stable order."""
    order = {iv: i for i, iv in enumerate(intervals)}
    rows = []
    for (symbol, interval), part in tape.group_by(["symbol", "interval"]):
        rows.append({"symbol": symbol, "interval": interval, **summarise(part)})
    rows.sort(key=lambda r: (r["symbol"], order.get(r["interval"], 99)))
    return rows


def best_exit(row: dict, field: str = "mean_r") -> tuple[str, dict]:
    """The exit with the highest ``field``, and its statistics."""
    exits = row.get("exits") or {}
    if not exits:
        return "", {}
    key = max(exits, key=lambda k: exits[k].get(field, -1e9))
    return key, exits[key]


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def fmt(value, spec=".3f", width=8) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return format(value, spec).rjust(width)


def rule(title: str, width: int = 100) -> None:
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def print_structure(rows: list[dict]) -> None:
    """What one unit of risk costs. This decides most of the study."""
    print(f"\n{'symbol':8} {'tf':>4} {'signals':>9} {'risk bps':>9} {'risk $':>8} "
          f"{'cost/R':>8} {'comm/R':>8} {'stop%':>7} {'inst%':>7} "
          f"{'MFE20':>7} {'MAE20':>7}")
    print("-" * 100)
    for r in rows:
        if not r.get("signals"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['signals']:9,} "
              f"{fmt(r['risk_bps_median'], '.2f', 9)} "
              f"{fmt(r['risk_usd_median'], '.2f', 8)} "
              f"{fmt(r['cost_r_median'], '.3f')} "
              f"{fmt(r['commission_r_median'], '.3f')} "
              f"{fmt(r['stop_rate'], '.2f', 7)} "
              f"{fmt(r['instant_stop_rate'], '.3f', 7)} "
              f"{fmt(r['mfe20_r_median'], '.2f', 7)} "
              f"{fmt(r['mae20_r_median'], '.2f', 7)}")


def print_signal_quality(rows: list[dict]) -> None:
    """Forward return with every cost and every exit rule stripped off."""
    print(f"\n{'symbol':8} {'tf':>4} {'n':>9} {'days':>6} "
          f"{'fwd1 bps':>10} {'t':>7} {'fwd5 bps':>10} {'t':>7} "
          f"{'fwd20 bps':>10} {'t':>7}")
    print("-" * 90)
    for r in rows:
        if not r.get("signals"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['signals']:9,} "
              f"{r.get('days', 0):6,} "
              f"{fmt(r['fwd1_bps_mean'], '.3f', 10)} {fmt(r['fwd1_bps_t'], '.2f', 7)} "
              f"{fmt(r['fwd5_bps_mean'], '.3f', 10)} {fmt(r['fwd5_bps_t'], '.2f', 7)} "
              f"{fmt(r['fwd20_bps_mean'], '.3f', 10)} {fmt(r['fwd20_bps_t'], '.2f', 7)}")


def print_exit_curve(rows: list[dict], field: str, title: str) -> None:
    """Mean R at every exit geometry: the answer to "which target?" all at once."""
    keys = list(EXIT_KEYS)
    head = "".join(EXIT_LABELS[k].rjust(8) for k in keys)
    print(f"\n{title}")
    print(f"{'symbol':8} {'tf':>4}" + head)
    print("-" * (13 + 8 * len(keys)))
    for r in rows:
        if not r.get("exits"):
            continue
        cells = "".join(fmt(r["exits"].get(k, {}).get(field), ".3f", 8) for k in keys)
        print(f"{r['symbol']:8} {r['interval']:>4}" + cells)


def print_best(rows: list[dict]) -> None:
    """The best exit per timeframe, in R and in the account's own money."""
    print(f"\n{'symbol':8} {'tf':>4} {'best':>6} {'trades':>8} {'mean R':>9} "
          f"{'t':>7} {'total R':>10} {'maxDD R':>9} {'USD':>10} "
          f"{'USD DD':>9} {'win%':>6}")
    print("-" * 100)
    for r in rows:
        key, e = best_exit(r)
        if not e:
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {EXIT_LABELS[key]:>6} "
              f"{e['trades']:8,} {fmt(e['mean_r'], '.4f', 9)} "
              f"{fmt(e['t'], '.2f', 7)} {fmt(e['total_r'], '.1f', 10)} "
              f"{fmt(e['max_dd_r'], '.1f', 9)} {fmt(e['usd_total'], '.2f', 10)} "
              f"{fmt(e['usd_max_dd'], '.2f', 9)} "
              f"{fmt(e['win_rate'] * 100, '.1f', 6)}")


def print_two_r(rows: list[dict]) -> None:
    """The idea exactly as stated: 2R, reversing, 0.01 lot."""
    print(f"\n{'symbol':8} {'tf':>4} {'trades':>8} {'mean R':>9} {'t':>7} "
          f"{'mid R':>9} {'cost/R':>8} {'total R':>10} {'USD':>10} {'USD DD':>9} "
          f"{'win%':>6}")
    print("-" * 100)
    for r in rows:
        e = (r.get("exits") or {}).get(tag(2.0))
        if not e:
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {e['trades']:8,} "
              f"{fmt(e['mean_r'], '.4f', 9)} {fmt(e['t'], '.2f', 7)} "
              f"{fmt(e['mean_r_mid'], '.4f', 9)} "
              f"{fmt(r['cost_r_median'], '.3f')} "
              f"{fmt(e['total_r'], '.1f', 10)} {fmt(e['usd_total'], '.2f', 10)} "
              f"{fmt(e['usd_max_dd'], '.2f', 9)} "
              f"{fmt(e['win_rate'] * 100, '.1f', 6)}")


def print_comparison(a: list[dict], b: list[dict], label_a: str, label_b: str,
                     field: str = "mean_r_mid", key: str = None) -> None:
    """Setup against its null, on the cost-free number, side by side."""
    key = key or tag(2.0)
    index_b = {(r["symbol"], r["interval"]): r for r in b}
    print(f"\n{'symbol':8} {'tf':>4} {label_a:>12} {label_b:>12} {'diff':>10} "
          f"{'fwd5 bps A':>12} {'fwd5 bps B':>12} {'diff':>10}")
    print("-" * 84)
    for r in a:
        other = index_b.get((r["symbol"], r["interval"]))
        if not other:
            continue
        va = (r.get("exits") or {}).get(key, {}).get(field)
        vb = (other.get("exits") or {}).get(key, {}).get(field)
        fa, fb = r.get("fwd5_bps_mean"), other.get("fwd5_bps_mean")
        diff = (va - vb) if (va is not None and vb is not None) else None
        fdiff = (fa - fb) if (fa is not None and fb is not None) else None
        print(f"{r['symbol']:8} {r['interval']:>4} {fmt(va, '.4f', 12)} "
              f"{fmt(vb, '.4f', 12)} {fmt(diff, '.4f', 10)} "
              f"{fmt(fa, '.3f', 12)} {fmt(fb, '.3f', 12)} {fmt(fdiff, '.3f', 10)}")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def collect(symbols, cfg, split, intervals, *, allow_test=False,
            verbose=False) -> pl.DataFrame:
    """Run every symbol and stack the tapes, tagging each with its symbol."""
    frames = []
    for symbol in symbols:
        t0 = time.time()
        # Cost is measured over the same window the bars come from. Spread has
        # compressed by roughly an order of magnitude since 2020, so pricing a
        # dev backtest at today's spreads would flatter it by more than this
        # setup's entire margin.
        cost = CostModel.from_profiles(symbol, split=split)
        tape = run(symbol, cfg, intervals=intervals, split=split,
                   allow_test=allow_test, cost=cost, verbose=verbose)
        if tape.is_empty():
            print(f"  {symbol}: no signals")
            continue
        frames.append(tape.with_columns(symbol=pl.lit(symbol)))
        print(f"  {symbol}: {tape.height:,} signals in {time.time() - t0:.0f}s")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def run_variant(args, cfg, split, name) -> tuple[pl.DataFrame, list[dict]]:
    tape = collect(args.symbols, cfg, split, args.intervals,
                   allow_test=(split == "test"), verbose=args.verbose)
    if tape.is_empty():
        return tape, []
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    tape.write_parquet(TAPE_DIR / f"engulfing_{name}_{split}.parquet")
    return tape, by_group(tape, args.intervals)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    p.add_argument("-i", "--intervals", nargs="+", default=list(ALL_INTERVALS))
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--control", action="store_true",
                   help="also run the geometry-only null")
    p.add_argument("--placebo", action="store_true",
                   help="also run the faded signal")
    p.add_argument("--no-reverse", action="store_true",
                   help="also run with the reversal rule switched off")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if "test" in args.splits:
        print("!! the test split is locked; it is being read deliberately\n")

    results: dict = {
        "config": {k: v for k, v in EngulfingConfig().__dict__.items()},
        "symbols": args.symbols,
        "intervals": args.intervals,
        "splits": {},
    }

    for split in args.splits:
        rule(f"{split.upper()} - the setup as stated: enter at the close, "
             f"stop at the extreme, reverse on the opposite signal")
        tape, rows = run_variant(args, EngulfingConfig(), split, "signal")
        if not rows:
            continue
        entry: dict = {"signal": rows}

        print("\n--- structure: what one unit of risk costs ---")
        print("   1R is the signal candle's close to its own extreme, so it "
              "shrinks with the timeframe\n   while the round turn does not. "
              "cost/R is that round turn per unit of risk.")
        print_structure(rows)

        print("\n--- signal quality: does the pattern predict anything? ---")
        print("   Mid to mid, signed by the direction taken. No stop, no "
              "target, no cost.\n   In bps, because R is a different size on "
              "every row of this table.")
        print_signal_quality(rows)

        print_exit_curve(rows, "mean_r_mid",
                         "--- exit curve, COST-FREE (mean R, mid to mid): "
                         "the best case for the idea ---")
        print_exit_curve(rows, "mean_r",
                         "--- exit curve, NET of spread, commission and "
                         "slippage (mean R) ---")

        print("\n--- the idea exactly as stated: 2R target, reversing, "
              "0.01 lot ---")
        print_two_r(rows)

        print("\n--- the best exit per timeframe, chosen in hindsight ---")
        print("   An upper bound, not a strategy: the target is picked after "
              "seeing the answer.")
        print_best(rows)

        if args.control:
            rule(f"{split.upper()} - CONTROL: the same geometry without the "
                 f"engulfment")
            _, crows = run_variant(args, EngulfingConfig(control=True), split,
                                   "control")
            if crows:
                entry["control"] = crows
                print("\n--- setup against its null, cost-free ---")
                print("   If the two columns agree, the engulfment is "
                      "decoration and what is\n   being measured is the "
                      "stop-at-the-bar's-extreme structure.")
                print_comparison(rows, crows, "signal 2R", "control 2R")

        if args.no_reverse:
            rule(f"{split.upper()} - NO REVERSAL: stop or target only")
            _, nrows = run_variant(args, EngulfingConfig(reverse=False), split,
                                   "noreverse")
            if nrows:
                entry["no_reverse"] = nrows
                print("\n--- what the reversal rule is worth, cost-free ---")
                print_comparison(rows, nrows, "reverse 2R", "no-rev 2R")

        if args.placebo:
            rule(f"{split.upper()} - PLACEBO: the faded signal")
            print("   Note the asymmetry this exposes: fading puts the stop at "
                  "the candle's\n   *near* extreme, so 1R collapses and cost/R "
                  "explodes. The R columns are\n   therefore not comparable "
                  "with the setup's; the bps columns are.")
            _, prows = run_variant(args, EngulfingConfig(fade=True), split,
                                   "placebo")
            if prows:
                entry["placebo"] = prows
                print_structure(prows)
                print_signal_quality(prows)

        results["splits"][split] = entry

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "engulfing.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
