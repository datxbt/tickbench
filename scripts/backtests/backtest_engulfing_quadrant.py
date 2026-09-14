"""Evaluate the quadrant engulfing bar: sell the 0.25-0.5 retracement, target the low.

The idea under test: a bar that takes out the prior bar's high and then closes
below the prior bar's open has shown its hand. Divide that bar into quarters,
work a resting sell order in the 0.25-0.5 quadrant, and target the bar's low.

That claim decomposes into five questions, and this script asks them
separately because they can fail independently:

1. **Does the resting order fill on the trades that work?** A retracement entry
   is not free. Price can reach the target without ever coming back to the
   zone, and those are precisely the trades the idea would have won. The
   fill/miss/invalidated split is table one, before any P&L is shown, because
   an entry convention that misses the winners and fills the losers is a
   losing strategy no matter what the pattern predicts.
2. **Does the geometry pay?** Entry at 0.25-0.5 with the target at 0 is a
   reward of a quarter to a half of one bar range. Against a stop at the bar's
   high that is 0.33R to 1.0R, so the strike rate has to clear 50-75% before a
   single cost is charged. The realised strike rate is measured against exactly
   that hurdle.
3. **Does anything survive costs?** ``cost_r`` is the round turn as a fraction
   of the risk taken. A resting limit entry does not cross the spread, which is
   a real saving over a market order - but the risk here is a *fraction* of a
   bar range rather than a whole one, which works the other way.
4. **Is the pattern doing the work, or the geometry?** Two nulls. ``matched``
   applies the identical quadrant machinery to ordinary bars. ``nosweep`` keeps
   condition 2 and drops condition 1 - bars that close below the prior open
   without having taken out the prior high - which isolates what the stop run
   itself is worth. And the fade takes the mirror trade on the same bars.
5. **What does it do to an account?** Fixed 0.01 lot, one position at a time,
   so dollar risk varies with the bar range and the equity curve is dominated
   by the widest bars. R and USD are both reported.

Everything in R is in multiples of the planned risk - the entry level to the
stop level - which is the only unit in which a 1m EURUSD quadrant and a 4h gold
one are the same bet.

Run:  python scripts/backtests/backtest_engulfing_quadrant.py [-s SYMBOL ...]
      python scripts/backtests/backtest_engulfing_quadrant.py --controls
      python scripts/backtests/backtest_engulfing_quadrant.py --variants
      python scripts/backtests/backtest_engulfing_quadrant.py --splits test  # locked
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
from qlab.strategies.engulfing_quadrant import (  # noqa: E402
    ALL_INTERVALS,
    ENTRY_MODES,
    EXIT_KEYS,
    HOLD,
    STOP_FIBS,
    TARGET_FIBS,
    TARGET_KEYS,
    QuadrantConfig,
    breakeven_rate,
    fair_rate,
    run,
    sequence_many,
    sequence,
)
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "engulfing_quadrant_trades"

# The idea exactly as stated: work the whole zone, stop at the bar's own high,
# target the bar's low. Everything else in the sweep is what it is measured
# against.
STATED_ENTRY = "zone"
STATED_STOP = "s100"
STATED_TARGET = "t0"
STATED = f"{STATED_STOP}_{STATED_TARGET}"

TARGET_LABELS = {**{k: f"{v:g}".replace("-", "−") for k, v in TARGET_FIBS.items()},
                 HOLD: "hold"}
START_EQUITY = 10_000.0   # only used to express the fixed-lot drawdown as a %


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def daily_t(trades: pl.DataFrame, column: str) -> tuple[float, float, int]:
    """(t-statistic, mean per day, n days) of a per-trade column, by day.

    Daily rather than per trade, deliberately. Four instruments and six
    timeframes fire on the same impulse - a 1h sweep contains the 15m one that
    formed it - and a per-trade t-statistic would count that overlap as
    independent evidence, overstating significance by roughly the square root
    of the number of correlated streams.
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


def max_drawdown(trades: pl.DataFrame, column: str, order: str) -> float:
    """Peak-to-trough of the cumulative curve, in that column's own units."""
    if trades.is_empty() or column not in trades.columns:
        return 0.0
    curve = trades.sort(order)[column].fill_null(0.0).cum_sum().to_numpy()
    if curve.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(curve) - curve))


# --------------------------------------------------------------------------
# Summarising one (symbol, interval, entry mode)
# --------------------------------------------------------------------------

def summarise(tape: pl.DataFrame) -> dict:
    """Fill economics, signal quality and every (stop, target) pair.

    Signal-level statistics are over every signal, filled or not. Exit-level
    statistics are over the trades an account *actually takes* under that exit,
    which is a different and smaller set for every rule, because when a trade
    ends decides which later signals it was flat for.
    """
    out: dict = {"signals": tape.height}
    if tape.is_empty():
        return out

    # --- the resting order: what happened to it -------------------------
    reasons = tape["fill_reason"].value_counts()
    counts = dict(zip(reasons["fill_reason"].to_list(), reasons["count"].to_list()))
    for key in ("filled", "missed", "invalidated", "expired"):
        out[f"{key}_rate"] = counts.get(key, 0) / tape.height
    out["days"] = tape["day"].n_unique()
    out["close_fib_median"] = float(tape["close_fib"].median())
    out["sweep_depth_median"] = float(tape["sweep_depth"].median())

    # The pattern on its own: from the bar's close, signed by the direction
    # taken, no order, no barrier, no cost. In bps, because a range is a
    # different size on every row of this table.
    for k in (1, 5, 20):
        t, _, _ = daily_t(tape, f"fwd{k}_bps")
        out[f"fwd{k}_bps_mean"] = float(tape[f"fwd{k}_bps"].mean())
        out[f"fwd{k}_bps_t"] = t
    out["mfe20_rng_median"] = float(tape["mfe20_rng"].median())
    out["mae20_rng_median"] = float(tape["mae20_rng"].median())

    filled = tape.filter(pl.col("filled"))
    out["fills"] = filled.height
    if filled.is_empty():
        return out

    out["limit_share"] = float((filled["order_type"] == "limit").mean())
    out["immediate_share"] = float(filled["immediate"].mean())
    out["fill_bar_median"] = float(filled["fill_bar"].median())
    out["entry_fib_mean"] = float(filled["entry_fib"].mean())
    out["mfe_fill_rng_median"] = float(filled["mfe_fill_rng"].median())
    out["mae_fill_rng_median"] = float(filled["mae_fill_rng"].median())

    for skey in STOP_FIBS:
        out[f"cost_r_{skey}"] = float(filled[f"cost_r_{skey}"].median())
        out[f"risk_bps_{skey}"] = float(filled[f"risk_bps_{skey}"].median())
        out[f"risk_usd_{skey}"] = float(filled[f"risk_usd_{skey}"].median())

    exits: dict[str, dict] = {}
    present = [k for k in EXIT_KEYS if f"r_{k}" in filled.columns]
    sequenced = sequence_many(filled, present)
    for key in present:
        col = f"r_{key}"
        skey, tkey = key.split("_", 1)
        taken = sequenced[key]
        if taken.is_empty():
            continue
        t, _, _ = daily_t(taken, col)
        usd = float(taken[f"usd_{key}"].sum())
        # Realised payoff on the winners, which is what the break-even rate has
        # to be computed against - it is not the nominal ratio, because the
        # entry level varies inside the zone.
        wins = taken.filter(pl.col(f"reason_{key}") == "tp")
        rr = float(wins[col].mean()) + float(taken[f"commission_r_{skey}"].mean()) \
            if wins.height else float("nan")
        cost_r = float(taken[f"cost_r_{skey}"].median())
        win_rate = float((taken[f"reason_{key}"] == "tp").mean())
        fair = (float("nan") if tkey == HOLD else
                fair_rate(float(taken["entry_fib"].mean()), STOP_FIBS[skey],
                          TARGET_FIBS[tkey]))
        exits[key] = {
            "trades": taken.height,
            "mean_r": float(taken[col].mean()),
            "total_r": float(taken[col].sum()),
            "t": t,
            "max_dd_r": max_drawdown(taken, col, "fill_ts"),
            "win_rate": win_rate,
            "stop_rate": float((taken[f"reason_{key}"] == "stop").mean()),
            "time_rate": float((taken[f"reason_{key}"] == "time").mean()),
            "payoff_r": rr,
            "cost_r": cost_r,
            "breakeven_rate": breakeven_rate(rr, cost_r),
            "win_gap": win_rate - breakeven_rate(rr, cost_r),
            # What the same barriers pay on a driftless walk. The pattern has
            # to beat this before the cost question is even worth asking.
            "fair_rate": fair,
            "edge_vs_fair": win_rate - fair,
            "usd_total": usd,
            "usd_max_dd": max_drawdown(taken, f"usd_{key}", "fill_ts"),
            "usd_per_trade": usd / taken.height,
            # Per *signal* rather than per fill: the missed and invalidated
            # signals earn nothing, and a strategy is judged on the
            # opportunities it was given, not the ones it took.
            "r_per_signal": float(taken[col].sum()) / tape.height,
        }
    out["exits"] = exits
    return out


def by_group(tape: pl.DataFrame, intervals, modes=ENTRY_MODES) -> list[dict]:
    """``summarise`` for every (symbol, interval, entry mode), in a stable order."""
    order = {iv: i for i, iv in enumerate(intervals)}
    morder = {m: i for i, m in enumerate(modes)}
    rows = []
    for (symbol, interval, mode), part in tape.group_by(
            ["symbol", "interval", "entry_mode"]):
        rows.append({"symbol": symbol, "interval": interval, "entry_mode": mode,
                     **summarise(part)})
    rows.sort(key=lambda r: (r["symbol"], order.get(r["interval"], 99),
                             morder.get(r["entry_mode"], 99)))
    return rows


def stated(rows: list[dict]) -> list[dict]:
    """Only the rows for the idea exactly as stated."""
    return [r for r in rows if r["entry_mode"] == STATED_ENTRY]


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def fmt(value, spec=".3f", width=8) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return format(value, spec).rjust(width)


def rule(title: str, width: int = 104) -> None:
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def print_fills(rows: list[dict]) -> None:
    """What happened to the resting order. This decides most of the study."""
    print(f"\n{'symbol':8} {'tf':>4} {'mode':>5} {'signals':>8} {'close':>7} "
          f"{'fill%':>7} {'miss%':>7} {'inval%':>7} {'exp%':>7} "
          f"{'limit%':>7} {'now%':>7} {'fillbar':>8}")
    print("-" * 104)
    for r in rows:
        if not r.get("signals"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['entry_mode']:>5} "
              f"{r['signals']:8,} {fmt(r['close_fib_median'], '.3f', 7)} "
              f"{fmt(r['filled_rate'], '.3f', 7)} {fmt(r['missed_rate'], '.3f', 7)} "
              f"{fmt(r['invalidated_rate'], '.3f', 7)} "
              f"{fmt(r['expired_rate'], '.3f', 7)} "
              f"{fmt(r.get('limit_share'), '.3f', 7)} "
              f"{fmt(r.get('immediate_share'), '.3f', 7)} "
              f"{fmt(r.get('fill_bar_median'), '.2f', 8)}")


def print_structure(rows: list[dict]) -> None:
    """What one unit of risk is, and what the round turn costs against it."""
    print(f"\n{'symbol':8} {'tf':>4} {'fills':>8} {'entryfib':>9} "
          + "".join(f"{'risk bps ' + s[1:]:>13}" for s in STOP_FIBS)
          + "".join(f"{'cost/R ' + s[1:]:>12}" for s in STOP_FIBS))
    print("-" * 104)
    for r in rows:
        if not r.get("fills"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['fills']:8,} "
              f"{fmt(r.get('entry_fib_mean'), '.3f', 9)} "
              + "".join(fmt(r.get(f'risk_bps_{s}'), '.2f', 13) for s in STOP_FIBS)
              + "".join(fmt(r.get(f'cost_r_{s}'), '.3f', 12) for s in STOP_FIBS))


def print_signal_quality(rows: list[dict]) -> None:
    """Forward move with every order, barrier and cost stripped off."""
    print(f"\n{'symbol':8} {'tf':>4} {'n':>8} {'days':>6} "
          f"{'fwd1 bps':>10} {'t':>7} {'fwd5 bps':>10} {'t':>7} "
          f"{'fwd20 bps':>10} {'t':>7} {'MFE20':>7} {'MAE20':>7}")
    print("-" * 104)
    for r in rows:
        if not r.get("signals"):
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {r['signals']:8,} "
              f"{r.get('days', 0):6,} "
              f"{fmt(r['fwd1_bps_mean'], '.3f', 10)} {fmt(r['fwd1_bps_t'], '.2f', 7)} "
              f"{fmt(r['fwd5_bps_mean'], '.3f', 10)} {fmt(r['fwd5_bps_t'], '.2f', 7)} "
              f"{fmt(r['fwd20_bps_mean'], '.3f', 10)} {fmt(r['fwd20_bps_t'], '.2f', 7)} "
              f"{fmt(r.get('mfe20_rng_median'), '.2f', 7)} "
              f"{fmt(r.get('mae20_rng_median'), '.2f', 7)}")


def print_hurdle(rows: list[dict], exit_key: str = STATED) -> None:
    """The strike rate against the strike rate the geometry demands.

    The single most informative table in the study: a payoff below 1 is not a
    flaw, it is the design, and the only question is whether the pattern clears
    the bar that payoff sets.
    """
    print(f"\n{'symbol':8} {'tf':>4} {'trades':>8} {'payoff':>8} {'cost/R':>8} "
          f"{'needs':>8} {'actual':>8} {'gap':>8} {'fair':>8} {'vs fair':>8} "
          f"{'mean R':>9} {'t':>7} {'USD':>10}")
    print("-" * 104)
    for r in rows:
        e = (r.get("exits") or {}).get(exit_key)
        if not e:
            continue
        print(f"{r['symbol']:8} {r['interval']:>4} {e['trades']:8,} "
              f"{fmt(e['payoff_r'], '.3f')} {fmt(e['cost_r'], '.3f')} "
              f"{fmt(e['breakeven_rate'], '.3f')} {fmt(e['win_rate'], '.3f')} "
              f"{fmt(e['win_gap'], '+.3f')} {fmt(e['fair_rate'], '.3f')} "
              f"{fmt(e['edge_vs_fair'], '+.3f')} {fmt(e['mean_r'], '.4f', 9)} "
              f"{fmt(e['t'], '.2f', 7)} {fmt(e['usd_total'], '.2f', 10)}")


def print_ladder(rows: list[dict], intervals, exit_key: str = STATED) -> None:
    """The timeframe sweep: one line per rung, pooled over the instruments.

    This is the table the whole study reduces to. Cost per unit of risk falls
    monotonically with the timeframe while the payoff barely moves, so the
    break-even strike rate falls too - and the question is whether the realised
    strike rate closes on it, or on the fair-coin value underneath it.
    """
    print(f"\n{'tf':>4} {'signals':>9} {'trades':>9} {'fill%':>7} {'miss%':>7} "
          f"{'payoff':>7} {'cost/R':>7} {'needs':>7} {'fair':>7} {'actual':>7} "
          f"{'vs BE':>7} {'vs fair':>8} {'mean R':>9} {'t':>7} {'USD':>11}")
    print("-" * 122)
    for tf in intervals:
        sub = [(r, r["exits"][exit_key]) for r in rows
               if r["interval"] == tf and exit_key in (r.get("exits") or {})]
        if not sub:
            continue
        n = sum(e["trades"] for _, e in sub)
        ns = sum(r["signals"] for r, _ in sub)
        # Trade-weighted for the exit statistics, signal-weighted for the fill
        # statistics: they have different denominators and mixing them would
        # quietly weight the fill rate by how often the order filled.
        w = lambda f: sum(f(e) * e["trades"] for _, e in sub) / n  # noqa: E731
        ws = lambda f: sum(f(r) * r["signals"] for r, _ in sub) / ns  # noqa: E731
        print(f"{tf:>4} {ns:9,} {n:9,} "
              f"{fmt(ws(lambda r: r['filled_rate']), '.3f', 7)} "
              f"{fmt(ws(lambda r: r['missed_rate']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['payoff_r']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['cost_r']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['breakeven_rate']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['fair_rate']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['win_rate']), '.3f', 7)} "
              f"{fmt(w(lambda e: e['win_gap']), '+.3f', 7)} "
              f"{fmt(w(lambda e: e['edge_vs_fair']), '+.3f', 8)} "
              f"{fmt(w(lambda e: e['mean_r']), '.4f', 9)} "
              f"{fmt(w(lambda e: e['t']), '.2f', 7)} "
              f"{fmt(sum(e['usd_total'] for _, e in sub), ',.0f', 11)}")


def print_grid(rows: list[dict], field: str, title: str) -> None:
    """Mean R at every (stop, target) pair: "which exit?" answered all at once."""
    print(f"\n{title}")
    head = "".join(
        f"{s[1:] + '/' + TARGET_LABELS[t]:>10}" for s in STOP_FIBS for t in TARGET_KEYS
    )
    print(f"{'symbol':8} {'tf':>4}" + head)
    print("-" * (13 + 10 * len(EXIT_KEYS)))
    for r in rows:
        if not r.get("exits"):
            continue
        cells = "".join(
            fmt(r["exits"].get(k, {}).get(field), ".4f", 10) for k in EXIT_KEYS
        )
        print(f"{r['symbol']:8} {r['interval']:>4}" + cells)


def print_modes(rows: list[dict], exit_key: str = STATED) -> None:
    """The three readings of "enter in the zone", side by side."""
    index = {(r["symbol"], r["interval"], r["entry_mode"]): r for r in rows}
    keys = sorted({(r["symbol"], r["interval"]) for r in rows})
    print(f"\n{'symbol':8} {'tf':>4}"
          + "".join(f"{m + ' fill%':>13}{m + ' meanR':>13}" for m in ENTRY_MODES))
    print("-" * 104)
    for symbol, interval in keys:
        line = f"{symbol:8} {interval:>4}"
        for mode in ENTRY_MODES:
            r = index.get((symbol, interval, mode))
            e = (r.get("exits") or {}).get(exit_key) if r else None
            line += fmt(r.get("filled_rate") if r else None, ".3f", 13)
            line += fmt(e.get("mean_r") if e else None, ".4f", 13)
        print(line)


def print_versus(a: list[dict], b: list[dict], label_a: str, label_b: str,
                 exit_key: str = STATED) -> None:
    """The setup against a null, on the numbers that separate them."""
    index_b = {(r["symbol"], r["interval"]): r for r in stated(b)}
    print(f"\n{'symbol':8} {'tf':>4} "
          f"{label_a + ' fill%':>14} {label_b + ' fill%':>14} "
          f"{label_a + ' win%':>14} {label_b + ' win%':>14} "
          f"{label_a + ' meanR':>14} {label_b + ' meanR':>14} {'diff':>9}")
    print("-" * 104)
    for r in stated(a):
        other = index_b.get((r["symbol"], r["interval"]))
        if not other:
            continue
        ea = (r.get("exits") or {}).get(exit_key, {})
        eb = (other.get("exits") or {}).get(exit_key, {})
        va, vb = ea.get("mean_r"), eb.get("mean_r")
        diff = (va - vb) if (va is not None and vb is not None) else None
        print(f"{r['symbol']:8} {r['interval']:>4} "
              f"{fmt(r.get('filled_rate'), '.3f', 14)} "
              f"{fmt(other.get('filled_rate'), '.3f', 14)} "
              f"{fmt(ea.get('win_rate'), '.3f', 14)} "
              f"{fmt(eb.get('win_rate'), '.3f', 14)} "
              f"{fmt(va, '.4f', 14)} {fmt(vb, '.4f', 14)} {fmt(diff, '+.4f', 9)}")


def print_pooled(rows: list[dict], label: str, exit_key: str = STATED) -> dict:
    """One line: the whole variant, pooled over instruments and timeframes."""
    cells = [(r.get("exits") or {}).get(exit_key) for r in rows]
    cells = [c for c in cells if c]
    if not cells:
        return {}
    n = sum(c["trades"] for c in cells)
    pooled = {
        "cells": len(cells),
        "trades": n,
        "mean_r": sum(c["mean_r"] * c["trades"] for c in cells) / n,
        "win_rate": sum(c["win_rate"] * c["trades"] for c in cells) / n,
        "breakeven_rate": sum(c["breakeven_rate"] * c["trades"] for c in cells) / n,
        "fair_rate": sum(c["fair_rate"] * c["trades"] for c in cells) / n,
        "usd_total": sum(c["usd_total"] for c in cells),
        "positive_cells": sum(1 for c in cells if c["mean_r"] > 0),
        "t_ge_2": sum(1 for c in cells if c["t"] >= 2),
        "t_le_m2": sum(1 for c in cells if c["t"] <= -2),
    }
    print(f"  {label:24} cells {pooled['cells']:3}  trades {n:9,}  "
          f"mean R {pooled['mean_r']:+.4f}  win {pooled['win_rate']:.3f} "
          f"(needs {pooled['breakeven_rate']:.3f}, fair {pooled['fair_rate']:.3f})  "
          f"USD {pooled['usd_total']:+10,.2f}  "
          f"positive {pooled['positive_cells']}/{pooled['cells']}  "
          f"t>=+2 {pooled['t_ge_2']}  t<=-2 {pooled['t_le_m2']}")
    return pooled


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
        print(f"  {symbol}: {tape.height:,} rows in {time.time() - t0:.0f}s")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def run_variant(args, cfg, split, name) -> tuple[pl.DataFrame, list[dict]]:
    path = TAPE_DIR / f"quadrant_{name}_{split}.parquet"
    # Re-summarising is seconds; re-walking the tape is tens of minutes. Every
    # table in this study is a pure function of the trade tape, so a new table
    # never needs a new backtest.
    if args.from_tapes and path.exists():
        tape = pl.read_parquet(path)
        print(f"  reusing {path.name}: {tape.height:,} rows")
        return tape, by_group(tape, args.intervals)
    tape = collect(args.symbols, cfg, split, args.intervals,
                   allow_test=(split == "test"), verbose=args.verbose)
    if tape.is_empty():
        return tape, []
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    tape.write_parquet(path)
    return tape, by_group(tape, args.intervals)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    p.add_argument("-i", "--intervals", nargs="+", default=list(ALL_INTERVALS))
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--controls", nargs="*", choices=["matched", "nosweep"],
                   default=None, metavar="NAME",
                   help="also run the nulls; bare flag runs both. The no-sweep "
                        "null draws every qualifying bar rather than a matched "
                        "sample, so it is ~6x the size and worth naming "
                        "explicitly when sweeping many timeframes")
    p.add_argument("--variants", nargs="*",
                   choices=["fade", "mirror", "prevbull"],
                   default=None, metavar="NAME",
                   help="also run the placebo and the symmetry checks; bare "
                        "flag runs all three")
    p.add_argument("--from-tapes", action="store_true",
                   help="re-summarise saved trade tapes instead of re-walking "
                        "the tick data; falls back to a real run per variant "
                        "whose tape is missing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if "test" in args.splits:
        print("!! the test split is locked; it is being read deliberately\n")

    results: dict = {
        "config": dict(QuadrantConfig().__dict__),
        "stated": {"entry": STATED_ENTRY, "stop": STATED_STOP,
                   "target": STATED_TARGET},
        "symbols": args.symbols,
        "intervals": args.intervals,
        "splits": {},
    }

    for split in args.splits:
        rule(f"{split.upper()} - the setup as stated: sweep the prior high, close "
             f"below the prior open,\nwork the 0.25-0.5 zone, stop at the bar's "
             f"high, target the bar's low")
        tape, rows = run_variant(args, QuadrantConfig(), split, "signal")
        if not rows:
            continue
        entry: dict = {"signal": rows}
        srows = stated(rows)

        print("\n--- 1. the resting order: does it fill on the trades that work? ---")
        print("   'miss' is price reaching the bar's low without ever returning to")
        print("   the zone - the trades the idea would have won. 'close' is where")
        print("   the signal bar closed on its own 0-1 ladder.")
        print_fills(srows)

        print("\n--- 2. structure: what one unit of risk is, and what it costs ---")
        print("   Risk is the entry level to the stop level, so it is a *fraction*")
        print("   of one bar range - 0.5 to 0.75 of it at the s100 stop.")
        print_structure(srows)

        print("\n--- 3. signal quality: does the pattern predict anything? ---")
        print("   From the bar's close, signed short. No order, no barrier, no cost.")
        print("   MFE/MAE are the 20-bar excursions in bar-range units.")
        print_signal_quality(srows)

        print("\n--- 3b. the timeframe ladder, pooled over the four instruments ---")
        print("   'needs' is the break-even strike rate the payoff and the cost")
        print("   imply; 'fair' is what a driftless walk pays on the same barriers.")
        print_ladder(srows, args.intervals)

        print("\n--- 4. the hurdle: the strike rate the geometry demands ---")
        print("   'payoff' is the realised winner in R, 'needs' is (1+cost)/(1+payoff),")
        print("   the strike rate that makes expectation zero. 'gap' is the verdict.")
        print_hurdle(srows)

        print_grid(srows, "mean_r",
                   "--- 5. the exit grid, NET (mean R). rows are symbol x timeframe, "
                   "columns stop/target ---")
        print_grid(srows, "win_rate",
                   "--- 5b. the same grid, strike rate ---")

        print("\n--- 6. the three readings of \"enter in the zone\" ---")
        print_modes(rows)

        print("\n--- pooled, at the stated geometry ---")
        entry["pooled"] = {"signal": print_pooled(srows, "signal")}

        if args.controls is not None:
            nulls = {"matched": QuadrantConfig(control="matched"),
                     "nosweep": QuadrantConfig(control="nosweep")}
            for name in (args.controls or list(nulls)):
                cfg = nulls[name]
                rule(f"{split.upper()} - null: {name}")
                _, crows = run_variant(args, cfg, split, name)
                if not crows:
                    continue
                entry[name] = crows
                entry["pooled"][name] = print_pooled(stated(crows), name)
                print(f"\n--- the setup against the {name} null ---")
                print_versus(rows, crows, "sig", name[:3])

        if args.variants is not None:
            kinds = {"fade": QuadrantConfig(fade=True),
                     "mirror": QuadrantConfig(mirror=True),
                     "prevbull": QuadrantConfig(require_prev_bull=True)}
            for name in (args.variants or list(kinds)):
                cfg = kinds[name]
                rule(f"{split.upper()} - variant: {name}")
                _, vrows = run_variant(args, cfg, split, name)
                if not vrows:
                    continue
                entry[name] = vrows
                entry["pooled"][name] = print_pooled(stated(vrows), name)
                print_hurdle(stated(vrows))

        results["splits"][split] = entry

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "engulfing_quadrant.json"
    # Merge rather than overwrite. A split is hours of tape walking, and
    # re-running one of them to add a variant should not silently destroy the
    # record of the others - which is exactly what a plain write does.
    if out.exists():
        try:
            prior = json.loads(out.read_text())
            merged = dict(prior.get("splits") or {})
            merged.update(results["splits"])
            results["splits"] = merged
        except json.JSONDecodeError:
            print(f"  (could not read prior {out.name}; writing fresh)")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}  [splits: {', '.join(results['splits'])}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
