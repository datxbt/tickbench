"""Evaluate the big-bar / pause-bar breakout across symbols, timeframes and exits.

The idea under test: a big bar breaks out of resistance, the next bar is a very
small pause bar, you buy the break of the pause bar with a stop under it - and
this is claimed to work on all assets and all timeframes.

That claim is testable in three separate places, and this script tests them
separately because they fail for different reasons:

1. **Does the signal predict anything?** Unconditional forward return from the
   entry mid over the holding window, against a control that keeps the entry
   geometry and drops the breakout context. If the two agree, the conditions
   are decoration.
2. **Does any exit make money before costs?** The full target curve from 0.5R
   to 6R, plus a trail and a plain time stop, resolved on the same fills.
3. **Does anything survive costs?** The pause bar is selected for being small,
   so 1R is small, so the round turn is a large fraction of 1R. This is a
   structural problem with the setup rather than a bad broker, and it is
   reported as ``cost_r`` - round-turn cost per unit of risk.

Everything is in **R**, multiples of the planned risk (trigger to stop). That
is the only unit in which a 1-pip EURUSD stop and a 467-pip gold stop are the
same bet, and it is what makes "all assets and all timeframes" a single table.

Run:  python scripts/backtests/backtest_pause_bar.py [-s SYMBOL ...] [--sweep] [--control]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.strategies.pause_bar import (  # noqa: E402
    TARGET_MULTIPLES,
    PauseBarConfig,
    _tag,
    run,
)
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "pause_bar_trades"
INTERVALS = ("1m", "5m", "15m", "1h")

# Every exit geometry the sweep resolves, as (column, label).
EXITS: tuple[tuple[str, str], ...] = (
    *((f"r_{_tag(m)}", f"{m:g}R target") for m in TARGET_MULTIPLES),
    ("r_trail", "trail"),
    ("r_hold", "time stop"),
)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def daily_t(trades: pl.DataFrame, column: str) -> tuple[float, float, int]:
    """(t-statistic, mean per day, n days) of a per-trade column, aggregated daily.

    Daily rather than per trade, deliberately. Four instruments and four
    timeframes fire on the same impulse, and a per-trade t-statistic would
    count that correlation as independent evidence - overstating significance
    by roughly the square root of the number of overlapping streams. Days are
    the coarsest unit that still leaves enough of them to measure.
    """
    if trades.is_empty() or column not in trades.columns:
        return float("nan"), float("nan"), 0
    daily = (
        trades.group_by("day").agg(v=pl.col(column).sum()).sort("day")["v"]
    )
    n = daily.len()
    if n < 2:
        return float("nan"), float("nan"), n
    sd = float(daily.std())
    mean = float(daily.mean())
    if sd == 0:
        return float("nan"), mean, n
    return mean / (sd / np.sqrt(n)), mean, n


def max_drawdown_r(trades: pl.DataFrame, column: str) -> float:
    """Peak-to-trough of the cumulative R curve, in R."""
    if trades.is_empty():
        return 0.0
    curve = trades.sort("arm_ts")[column].fill_null(0.0).cum_sum().to_numpy()
    if curve.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(curve) - curve))


def summarise(tape: pl.DataFrame, label: str) -> dict:
    """Fill rate, cost structure, signal quality and every exit, for one slice."""
    out: dict = {"label": label, "signals": tape.height}
    if tape.is_empty():
        return out

    filled = tape.filter(pl.col("filled"))
    out["filled"] = filled.height
    out["fill_rate"] = filled.height / tape.height
    if filled.is_empty():
        return out

    out["risk_bps_median"] = float(filled["risk_bps"].median())
    out["cost_r_median"] = float(filled["cost_r"].median())
    out["cost_r_mean"] = float(filled["cost_r"].mean())
    out["commission_r_median"] = float(filled["commission_r"].median())
    out["mfe_r_median"] = float(filled["mfe_r"].median())
    out["mae_r_median"] = float(filled["mae_r"].median())
    out["mfe_pre_stop_r_median"] = float(filled["mfe_pre_stop_r"].median())
    out["stop_rate"] = float(filled["stopped"].mean())

    # The signal on its own: no stop, no target, no cost, mid to mid.
    t, mean, n_days = daily_t(filled, "fwd_r")
    out["fwd_r_mean"] = float(filled["fwd_r"].mean())
    out["fwd_r_t"] = t
    out["fwd_bps_mean"] = float(filled["fwd_bps"].mean())
    out["days"] = n_days

    exits: dict[str, dict] = {}
    for col, name in EXITS:
        if col not in filled.columns:
            continue
        r = filled[col]
        t, _, _ = daily_t(filled, col)
        row = {
            "mean_r": float(r.mean()),
            "total_r": float(r.sum()),
            "t": t,
            "max_dd_r": max_drawdown_r(filled, col),
        }
        gross = f"{col}_gross"
        if gross in filled.columns:
            row["mean_r_gross"] = float(filled[gross].mean())
        mid = f"{col}_mid"
        if mid in filled.columns:
            row["mean_r_mid"] = float(filled[mid].mean())
        reason = col.replace("r_", "reason_", 1)
        if reason in filled.columns:
            row["win_rate"] = float((filled[reason] == "tp").mean())
        exits[name] = row
    out["exits"] = exits
    return out


def by_group(tape: pl.DataFrame, keys: list[str]) -> list[dict]:
    """``summarise`` for every combination of ``keys`` present in the tape."""
    rows = []
    for key, part in tape.group_by(keys, maintain_order=True):
        label = " ".join(str(k) for k in key)
        rows.append({**dict(zip(keys, key)), **summarise(part, label)})
    return rows


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def fmt(value, spec=".3f", width=8) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return format(value, spec).rjust(width)


def print_structure(rows: list[dict]) -> None:
    """The cost table - the one that decides most of this before any edge does."""
    print(f"\n{'symbol':8} {'tf':>4} {'signals':>8} {'fill':>6} {'risk bps':>9} "
          f"{'cost/R':>8} {'comm/R':>8} {'stop%':>7} {'MFE':>7} {'MAE':>7}")
    print("-" * 82)
    for r in rows:
        if not r.get("filled"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['signals']:8,} "
              f"{fmt(r['fill_rate'], '.2f', 6)} {fmt(r['risk_bps_median'], '.2f', 9)} "
              f"{fmt(r['cost_r_median'], '.3f')} {fmt(r['commission_r_median'], '.3f')} "
              f"{fmt(r['stop_rate'], '.2f', 7)} {fmt(r['mfe_r_median'], '.2f', 7)} "
              f"{fmt(r['mae_r_median'], '.2f', 7)}")


def print_exit_curve(rows: list[dict], *, field: str = "mean_r") -> None:
    """Mean R at every exit geometry: the answer to "which exit?" all at once."""
    names = [n for _, n in EXITS]
    head = "".join(n.replace(" target", "R").replace("time stop", "hold").rjust(9)
                   for n in names)
    print(f"\n{'symbol':8} {'tf':>4} {'n':>6}" + head)
    print("-" * (20 + 9 * len(names)))
    for r in rows:
        if not r.get("exits"):
            continue
        cells = "".join(fmt(r["exits"].get(n, {}).get(field), ".3f", 9) for n in names)
        print(f"{r['symbol']:8} {r['interval']:>4} {r['filled']:6,}" + cells)


def print_signal_quality(rows: list[dict]) -> None:
    """Forward return with every cost and every exit rule stripped off."""
    print(f"\n{'symbol':8} {'tf':>4} {'n':>7} {'days':>6} {'fwd R':>9} "
          f"{'fwd t':>8} {'fwd bps':>9} {'2R net':>9} {'2R t':>8} {'2R mid':>9}")
    print("-" * 88)
    for r in rows:
        if not r.get("exits"):
            continue
        two = r["exits"].get("2R target", {})
        print(f"{r['symbol']:8} {r['interval']:>4} {r['filled']:7,} {r['days']:6,} "
              f"{fmt(r['fwd_r_mean'], '.3f', 9)} {fmt(r['fwd_r_t'], '.2f', 8)} "
              f"{fmt(r['fwd_bps_mean'], '.2f', 9)} {fmt(two.get('mean_r'), '.3f', 9)} "
              f"{fmt(two.get('t'), '.2f', 8)} {fmt(two.get('mean_r_mid'), '.3f', 9)}")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def collect(symbols, cfg, split, intervals, *, allow_test=False,
            verbose=False) -> pl.DataFrame:
    """Run every symbol and stack the tapes, tagging each with its symbol."""
    frames = []
    for symbol in symbols:
        t0 = time.time()
        # Cost is measured over the same window the bars come from: dev-period
        # trading cost up to 2.1x what it costs now, and pricing a dev backtest
        # at today's spreads would flatter it by more than most edges are worth.
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


def main_run(args) -> dict:
    """The headline: the setup as stated, on dev and on validation."""
    cfg = PauseBarConfig()
    results: dict = {"config": cfg.__dict__, "splits": {}}

    for split in args.splits:
        print(f"\n{'=' * 82}\n{split.upper()} - the setup as stated\n{'=' * 82}")
        tape = collect(args.symbols, cfg, split, args.intervals,
                       allow_test=(split == "test"), verbose=args.verbose)
        if tape.is_empty():
            continue
        TAPE_DIR.mkdir(parents=True, exist_ok=True)
        tape.write_parquet(TAPE_DIR / f"pause_bar_{split}.parquet")

        rows = by_group(tape, ["symbol", "interval"])
        rows.sort(key=lambda r: (r["symbol"], INTERVALS.index(r["interval"])))
        print("\n--- structure: what one unit of risk costs ---")
        print_structure(rows)
        print("\n--- signal quality: forward return, no exit rule, no cost ---")
        print_signal_quality(rows)
        print("\n--- the exit curve: mean R, net of all costs ---")
        print_exit_curve(rows)
        print("\n--- the same curve before costs, filling at mid ---")
        print_exit_curve(rows, field="mean_r_mid")

        pooled = by_group(tape, ["interval"])
        pooled.sort(key=lambda r: INTERVALS.index(r["interval"]))
        for r in pooled:
            r["symbol"] = "ALL"
        print("\n--- pooled across symbols ---")
        print_exit_curve(pooled)

        results["splits"][split] = {
            "by_symbol_interval": rows,
            "pooled_by_interval": pooled,
            "all": summarise(tape, split),
        }
    return results


def control_run(args) -> dict:
    """Compared with what? The same entry geometry without the breakout context.

    ``context="any"`` keeps step 2 and step 3 - a small bar after a directional
    bar, entered on its break - and throws away step 1, the breakout. If the
    two samples have the same forward return, then steps 1 and 4 of the idea
    are doing no work and what is left is a stop order in a random place.
    """
    out: dict = {}
    intervals = tuple(i for i in args.intervals if i != "1m")
    for context in ("breakout", "no_breakout", "any"):
        cfg = PauseBarConfig(context=context)
        print(f"\n{'=' * 82}\nCONTROL: context={context}  (dev)\n{'=' * 82}")
        tape = collect(args.symbols, cfg, "dev", intervals, verbose=args.verbose)
        if tape.is_empty():
            continue
        rows = by_group(tape, ["symbol", "interval"])
        rows.sort(key=lambda r: (r["symbol"], INTERVALS.index(r["interval"])))
        print_signal_quality(rows)
        pooled = summarise(tape, context)
        print(f"\npooled {context}: n={pooled['filled']:,}  "
              f"fwd_r={pooled['fwd_r_mean']:+.4f} (t={pooled['fwd_r_t']:.2f})  "
              f"2R net={pooled['exits']['2R target']['mean_r']:+.4f}  "
              f"2R mid={pooled['exits']['2R target']['mean_r_mid']:+.4f}")
        out[context] = {"by_symbol_interval": rows, "pooled": pooled}
    return out


def placebo_run(args) -> dict:
    """Does fading the signal lose, or does it win too?

    Every cost-free number in this study comes from a stop-and-target structure
    whose barriers are crossed on the exit side of the book. That structure has
    an expectation of its own, independent of any signal, and it is added to
    both sides of the trade. The test is symmetry: a real edge reverses sign
    when the signal is faded, an artifact of the measurement does not.
    """
    out: dict = {}
    for fade in (False, True):
        cfg = PauseBarConfig(fade=fade)
        name = "fade" if fade else "signal"
        print(f"\n{'=' * 82}\nPLACEBO: {name}  (dev)\n{'=' * 82}")
        tape = collect(args.symbols, cfg, "dev", args.intervals, verbose=args.verbose)
        if tape.is_empty():
            continue
        rows = by_group(tape, ["interval"])
        rows.sort(key=lambda r: INTERVALS.index(r["interval"]))
        for r in rows:
            r["symbol"] = "ALL"
        print_signal_quality(rows)
        out[name] = {"by_interval": rows, "pooled": summarise(tape, name)}

    if len(out) == 2:
        print(f"\n{'tf':>4} {'signal 2R mid':>14} {'fade 2R mid':>13} {'sum':>8} "
              f"{'signal fwd':>11} {'fade fwd':>10} {'sum':>8}")
        print("-" * 72)
        for a, b in zip(out["signal"]["by_interval"], out["fade"]["by_interval"]):
            sa = a["exits"]["2R target"]["mean_r_mid"]
            sb = b["exits"]["2R target"]["mean_r_mid"]
            print(f"{a['interval']:>4} {sa:14.3f} {sb:13.3f} {sa + sb:8.3f} "
                  f"{a['fwd_r_mean']:11.3f} {b['fwd_r_mean']:10.3f} "
                  f"{a['fwd_r_mean'] + b['fwd_r_mean']:8.3f}")
        print("\nA real edge sums to ~0 across the two rows. A structural"
              "\nexpectation in the measurement sums to twice itself.")
    return out


SWEEPS: dict[str, tuple] = {
    "big_mult": (1.0, 1.25, 1.5, 2.0, 2.5),
    "pause_mult": (0.25, 0.35, 0.5, 0.7),
    "close_frac": (0.5, 0.6, 0.75, 0.9),
    "lookback": (10, 20, 40, 80),
    "arm_bars": (1, 2, 3),
    "max_hold_bars": (5, 10, 20, 40),
    "require_inside": (False, True),
    "min_risk_pips": (0.0,),  # replaced per symbol below if asked for
}


def sweep_run(args) -> dict:
    """Is the verdict a property of the idea, or of one parameter set?

    Each parameter is moved on its own, on dev, on the timeframes where the
    cost objection is weakest - so the sweep is looking for the edge where it
    has the best chance of existing, not where it is most convenient.
    """
    out: dict = {}
    intervals = ("15m", "1h")
    for name, values in SWEEPS.items():
        if name == "min_risk_pips":
            continue
        out[name] = []
        print(f"\n{'=' * 82}\nSWEEP {name}  (dev, {', '.join(intervals)})\n{'=' * 82}")
        for value in values:
            cfg = PauseBarConfig(**{name: value})
            tape = collect(args.symbols, cfg, "dev", intervals, verbose=False)
            if tape.is_empty():
                continue
            s = summarise(tape, f"{name}={value}")
            best = max(s["exits"].items(), key=lambda kv: kv[1]["mean_r"])
            print(f"  {name}={value!s:>6}  n={s['filled']:6,}  "
                  f"cost/R={s['cost_r_median']:.3f}  "
                  f"fwd_r={s['fwd_r_mean']:+.4f} (t={s['fwd_r_t']:5.2f})  "
                  f"2R={s['exits']['2R target']['mean_r']:+.4f}  "
                  f"best={best[0]} {best[1]['mean_r']:+.4f}")
            out[name].append({"value": value, **s})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-i", "--intervals", nargs="+", default=list(INTERVALS))
    parser.add_argument("--splits", nargs="+", default=["dev", "validation"])
    # Alternatives, not layers: each answers a different question and none is
    # meaningful stacked on another.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--control", action="store_true",
                      help="run the null comparison instead of the main study")
    mode.add_argument("--placebo", action="store_true",
                      help="run the faded-signal symmetry check")
    mode.add_argument("--sweep", action="store_true",
                      help="run the one-at-a-time parameter sensitivity")
    parser.add_argument("--test", action="store_true",
                        help="unlock the held-out split - only for a live candidate")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.test and "test" not in args.splits:
        args.splits = args.splits + ["test"]
    if "test" in args.splits and not args.test:
        parser.error("the test split is held out; pass --test to spend it")

    started = time.time()
    if args.control:
        results, name = control_run(args), "pause_bar_control"
    elif args.placebo:
        results, name = placebo_run(args), "pause_bar_placebo"
    elif args.sweep:
        results, name = sweep_run(args), "pause_bar_sweep"
    else:
        results, name = main_run(args), "pause_bar"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else REPORT_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
