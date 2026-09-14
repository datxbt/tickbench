"""Evaluate UAlgo's liquidity sweeps and order blocks, 1m to D1, four instruments.

Pre-registered in docs/findings/sweep-orderblock.md before any outcome was
computed. Every table is a pure function of the saved order tapes, so
``--from-tapes`` re-summarises without walking a tick.

Run:  python scripts/backtests/backtest_sweep_orderblock.py              # dev + validation, setups + null
      python scripts/backtests/backtest_sweep_orderblock.py -s USTEC -i 1h 4h 1d
      python scripts/backtests/backtest_sweep_orderblock.py --from-tapes
      python scripts/backtests/backtest_sweep_orderblock.py --diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.rollover import basis  # noqa: E402
from statistics import NormalDist  # noqa: E402

from qlab.strategies.sweep_orderblock import (  # noqa: E402
    ALL_EXIT_KEYS, ALL_INTERVALS, EXIT_KEYS, FAMILIES, HOLD, OB_TARGETS, TARGET_R,
    ToolkitConfig, fair_rate, run, sequence_many,
)
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "sweep_orderblock_trades"
PRIMARY = "s0_t2"
SIDAK_PRIMARY = 3.27    # 96 cells, one-sided 0.05
SIDAK_GRID = 3.82       # 768 cells
MIN_DAYS = 30           # a primary pass needs at least this many trading days


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def daily_t(frame: pl.DataFrame, col: str) -> tuple[float, int]:
    """Per-day t: summed within a day, tested across days - trades cluster."""
    if frame.is_empty():
        return float("nan"), 0
    v = frame.group_by("day").agg(pl.col(col).sum())[col].to_numpy()
    n = v.size
    if n < 3 or v.std(ddof=1) == 0:
        return float("nan"), n
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(n))), n


def summarise(tape: pl.DataFrame, swap_bps: float) -> dict:
    """Order economics, the raw signal, and all eight exits for one cell."""
    out: dict = {"orders": tape.height}
    if tape.is_empty():
        return out
    rc = tape["fill_reason"].value_counts()
    for reason, cnt in zip(rc["fill_reason"].to_list(), rc["count"].to_list()):
        out[f"{reason}_rate"] = cnt / tape.height
    filled = tape.filter(pl.col("filled"))
    out["fills"] = filled.height
    if filled.is_empty():
        return out
    out["market_share"] = float((filled["order_type"] == "market").mean())
    for k in (1, 5, 20):
        col = f"fwd{k}_bps"
        f = filled.filter(pl.col(col).is_not_nan())
        out[f"{col}_mean"] = float(f[col].mean()) if f.height else float("nan")
        out[f"{col}_t"] = daily_t(f, col)[0]
    out["mfe_r_median"] = float(filled["mfe_r"].median())
    out["mae_r_median"] = float(filled["mae_r"].median())
    for sk in ("s0", "s1"):
        out[f"cost_r_{sk}"] = float(filled[f"cost_r_{sk}"].median())
        out[f"risk_bps_{sk}"] = float(filled[f"risk_bps_{sk}"].median())

    if "tp_near" in tape.columns and tape["family"][0] == "ob":
        out["no_target_rate"] = float(tape["tp_near"].is_null().mean())
    exits = {}
    present = [k for k in ALL_EXIT_KEYS if f"r_{k}" in filled.columns]
    taken_by = sequence_many(filled, present)
    for key in present:
        taken = taken_by[key]
        if taken.is_empty():
            continue
        sk, tk = key.split("_", 1)
        col = f"r_{key}"
        # Swap at the measured overnight basis: price drift the carry puts in
        # the quote, handed back by the broker. Long pays +basis per roll.
        rcol = f"risk_{key}" if f"risk_{key}" in taken.columns else f"risk_{sk}"
        ccol = f"cost_r_{key}" if f"cost_r_{key}" in taken.columns else f"cost_r_{sk}"
        taken = taken.with_columns(
            (pl.col(col) - pl.col("direction") * pl.col(f"rolls_{key}")
             * swap_bps / 1e4 * pl.col("entry") / pl.col(rcol)).alias("_rs"))
        t, days = daily_t(taken, col)
        win = float((taken[f"reason_{key}"] == "tp").mean())
        tcol = f"tpr_{key}"
        if tcol in taken.columns and taken[tcol].null_count() < taken.height:
            # the target sits k R away with k varying per trade (the opposite
            # block, or any mirrored bracket): the coin is 1/(1+k) per trade
            tpr = float(taken[tcol].median())
            fair = float((1.0 / (1.0 + taken[tcol])).mean())
        elif tk in TARGET_R:
            fair, tpr = fair_rate(TARGET_R[tk]), TARGET_R[tk]
        else:
            fair, tpr = float("nan"), float("nan")
        exits[key] = {
            "tpr_median": tpr,
            "trades": taken.height, "days": days,
            "mean_r": float(taken[col].mean()), "t": t,
            "mean_r_swap": float(taken["_rs"].mean()), "t_swap": daily_t(taken, "_rs")[0],
            "win": win, "fair": fair, "edge": win - fair,
            "stop_rate": float((taken[f"reason_{key}"] == "stop").mean()),
            "time_rate": float((taken[f"reason_{key}"] == "time").mean()),
            "rolls_mean": float(taken[f"rolls_{key}"].mean()),
            "usd": float(taken[f"usd_{key}"].sum()),
            "cost_r": float(taken[ccol].median()),
        }
    out["exits"] = exits
    return out


def by_cell(tape: pl.DataFrame, intervals, swap: dict) -> list[dict]:
    order = {iv: i for i, iv in enumerate(intervals)}
    rows = []
    for (sym, iv, fam), part in tape.group_by(["symbol", "interval", "family"]):
        rows.append({"symbol": sym, "interval": iv, "family": fam,
                     **summarise(part, swap.get(sym, 0.0))})
    rows.sort(key=lambda r: (r["family"], r["symbol"], order.get(r["interval"], 99)))
    return rows


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def fmt(v, spec=".3f", w=8) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(w)
    return format(v, spec).rjust(w)


def print_cells(rows, key=PRIMARY, null: dict | None = None) -> None:
    print(f"{'fam':5} {'symbol':7} {'tf':>4} {'orders':>7} {'fill%':>6} {'fwd5':>7} {'t':>6} "
          f"{'fwd20':>7} {'t':>6} {'cost/R':>7} {'trades':>7} {'win':>6} {'fair':>6} "
          f"{'edge':>7} {'null':>7} {'meanR':>8} {'t':>6} {'swapR':>8}")
    for r in rows:
        e = (r.get("exits") or {}).get(key)
        if not e:
            continue
        ne = ((null or {}).get((r["family"], r["symbol"], r["interval"])) or {}).get(key, {})
        print(f"{r['family']:5} {r['symbol']:7} {r['interval']:>4} {r['orders']:7,} "
              f"{fmt(r.get('fills', 0) / r['orders'], '.2f', 6)} "
              f"{fmt(r.get('fwd5_bps_mean'), '+.2f', 7)} {fmt(r.get('fwd5_bps_t'), '.1f', 6)} "
              f"{fmt(r.get('fwd20_bps_mean'), '+.2f', 7)} {fmt(r.get('fwd20_bps_t'), '.1f', 6)} "
              f"{fmt(e['cost_r'], '.3f', 7)} {e['trades']:7,} {fmt(e['win'], '.3f', 6)} "
              f"{fmt(e['fair'], '.3f', 6)} {fmt(e['edge'], '+.3f', 7)} "
              f"{fmt(ne.get('edge'), '+.3f', 7)} {fmt(e['mean_r'], '+.4f', 8)} "
              f"{fmt(e['t'], '.2f', 6)} {fmt(e['mean_r_swap'], '+.4f', 8)}")


def print_ladder(rows, key=PRIMARY) -> None:
    """Pooled over instruments, per setup and timeframe, trade-weighted."""
    print(f"{'fam':5} {'tf':>4} {'trades':>8} {'cost/R':>7} {'win':>6} {'fair':>6} "
          f"{'edge':>7} {'meanR':>8} {'cells t>=2':>11} {'t<=-2':>6}")
    for fam in FAMILIES:
        for iv in ALL_INTERVALS:
            es = [r["exits"][key] for r in rows if r["family"] == fam and r["interval"] == iv
                  and key in (r.get("exits") or {})]
            if not es:
                continue
            n = sum(e["trades"] for e in es)
            w = lambda f: sum(e[f] * e["trades"] for e in es) / n  # noqa: E731
            print(f"{fam:5} {iv:>4} {n:8,} {fmt(w('cost_r'), '.3f', 7)} {fmt(w('win'), '.3f', 6)} "
                  f"{fmt(w('fair'), '.3f', 6)} {fmt(w('edge'), '+.3f', 7)} {fmt(w('mean_r'), '+.4f', 8)} "
                  f"{sum(1 for e in es if e['t'] >= 2):11} {sum(1 for e in es if e['t'] <= -2):6}")


def grid_counts(rows) -> dict:
    ts = [e["t"] for r in rows for e in (r.get("exits") or {}).values() if np.isfinite(e["t"])]
    pos = [e for r in rows for e in (r.get("exits") or {}).values() if e["mean_r"] > 0]
    return {"combos": len(ts), "positive": len(pos),
            "t_ge_2": sum(t >= 2 for t in ts), "t_le_m2": sum(t <= -2 for t in ts),
            "t_ge_sidak": sum(t >= SIDAK_GRID for t in ts)}


# --------------------------------------------------------------------------
# Diagnostics - pure functions of the saved tapes
# --------------------------------------------------------------------------

def _pool(rows, key) -> dict:
    es = [r["exits"][key] for r in rows if key in (r.get("exits") or {})]
    n = sum(e["trades"] for e in es)
    if not n:
        return {}
    w = lambda f: sum(e[f] * e["trades"] for e in es) / n  # noqa: E731
    return {"trades": n, "mean_r": w("mean_r"), "win": w("win"), "edge": w("edge"),
            "cost_r": w("cost_r"), "usd": sum(e["usd"] for e in es),
            "pos": sum(e["mean_r"] > 0 for e in es), "cells": len(es)}


def edge_on_cost(rows, key=PRIMARY) -> tuple[float, float]:
    """Trade-weighted fit of edge = a + b * cost/R across cells.

    Spread moves both barriers against the trade, so a cell's strike rate sits
    below the coin in proportion to its cost/R. The intercept is the edge a
    cell would have at zero cost - the only fair way to compare a setup with a
    null whose stops are a different size.
    """
    pts = [(e["cost_r"], e["edge"], e["trades"]) for r in rows
           for k, e in (r.get("exits") or {}).items() if k == key and np.isfinite(e["edge"])]
    if len(pts) < 3:
        return float("nan"), float("nan")
    x, y, w = (np.array(v, float) for v in zip(*pts))
    sw = np.sqrt(w)
    a, b = np.linalg.lstsq(np.c_[sw, sw * x], sw * y, rcond=None)[0]
    return float(a), float(b)


def diagnostics(args) -> dict:
    tapes = {(v, s): pl.read_parquet(TAPE_DIR / f"sob_{v}_{s}.parquet")
             for v in ("none", "matched") for s in ("dev", "validation")}
    out: dict = {}
    rows = {k: by_cell(t, args.intervals, {}) for k, t in tapes.items()}

    print("\n--- pooled dev + validation, per setup, every exit (edge = win - fair) ---")
    print(f"{'fam':5} {'variant':8} {'exit':8} {'trades':>8} {'meanR':>8} {'win':>6} "
          f"{'edge':>7} {'cost/R':>7} {'cells+':>7} {'USD':>11}")
    for fam in FAMILIES:
        for v in ("none", "matched"):
            for key in EXIT_KEYS:
                both = [r for s in ("dev", "validation") for r in rows[(v, s)] if r["family"] == fam]
                p = _pool(both, key)
                out[f"pooled/{fam}/{v}/{key}"] = p
                if p:
                    print(f"{fam:5} {v:8} {key:8} {p['trades']:8,} {p['mean_r']:+8.4f} "
                          f"{fmt(p['win'], '.3f', 6)} {fmt(p['edge'], '+.3f', 7)} "
                          f"{p['cost_r']:7.3f} {p['pos']:3}/{p['cells']:<3} {p['usd']:+11,.0f}")

    print(f"\n--- edge on cost/R across cells at {PRIMARY}: intercept = edge at zero cost ---")
    for fam in FAMILIES:
        for s in ("dev", "validation"):
            for v in ("none", "matched"):
                a, b = edge_on_cost([r for r in rows[(v, s)] if r["family"] == fam])
                out[f"edge_on_cost/{fam}/{s}/{v}"] = {"intercept": a, "slope": b}
                print(f"  {fam:5} {s:10} {v:8} intercept {a:+.4f}  slope {b:+.3f}")

    print("\n--- barrier-free forward return from the fill mid, bps, pooled over symbols ---")
    print(f"{'fam':5} {'split':10} {'variant':8} {'tf':>4} {'fills':>8} "
          f"{'fwd1':>7} {'t':>6} {'fwd5':>7} {'t':>6} {'fwd20':>7} {'t':>6}")
    for fam in FAMILIES:
        for s in ("dev", "validation"):
            for v in ("none", "matched"):
                f = tapes[(v, s)].filter(pl.col("filled") & (pl.col("family") == fam))
                for iv in ("1m", "5m", "15m", "1h"):
                    g = f.filter(pl.col("interval") == iv)
                    vals = []
                    for k in (1, 5, 20):
                        c = f"fwd{k}_bps"
                        gg = g.filter(pl.col(c).is_not_nan())
                        vals += [float(gg[c].mean()), daily_t(gg, c)[0]]
                    out[f"fwd/{fam}/{s}/{v}/{iv}"] = vals
                    print(f"{fam:5} {s:10} {v:8} {iv:>4} {g.height:8,} "
                          + " ".join(f"{fmt(vals[i], '+.3f', 7)} {fmt(vals[i + 1], '.1f', 6)}"
                                     for i in (0, 2, 4)))

    print(f"\n--- order blocks by break type, at {PRIMARY}, account-sequenced ---")
    for s in ("dev", "validation"):
        f = tapes[("none", s)].filter(pl.col("filled") & (pl.col("family") == "ob"))
        taken = sequence_many(f, [PRIMARY])[PRIMARY]
        for kind, g in taken.group_by("break_kind"):
            win = float((g[f"reason_{PRIMARY}"] == "tp").mean())
            out[f"kind/{s}/{kind[0]}"] = [g.height, float(g[f"r_{PRIMARY}"].mean()), win - 1 / 3]
            print(f"  {s:10} {kind[0]:6} trades {g.height:7,}  mean R {g[f'r_{PRIMARY}'].mean():+.4f}"
                  f"  edge {win - 1 / 3:+.4f}")

    print("\n--- order-book fate of block orders, pooled ---")
    for s in ("dev", "validation"):
        t = tapes[("none", s)].filter(pl.col("family") == "ob")
        vc = t["fill_reason"].value_counts(normalize=True).sort("fill_reason")
        print(f"  {s:10} " + "  ".join(f"{a} {b:.3f}" for a, b in vc.rows()))
    return out


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def _one(job):
    symbol, split, control, intervals, families, fade = job
    t0 = time.time()
    tape = run(symbol, ToolkitConfig(control=control, fade=fade), intervals=intervals,
               split=split, families=families)
    print(f"  {symbol} {split} {control}: {tape.height:,} rows in {time.time() - t0:.0f}s", flush=True)
    return tape.with_columns(symbol=pl.lit(symbol)) if not tape.is_empty() else tape


def collect(args, split, control) -> pl.DataFrame:
    path = TAPE_DIR / f"sob_{control}_{split}{args.tag}.parquet"
    if args.from_tapes and path.exists():
        return pl.read_parquet(path)
    jobs = [(s, split, control, args.intervals, args.families, args.fade) for s in args.symbols]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tapes = [t for t in pool.map(_one, jobs) if not t.is_empty()]
    tape = pl.concat(tapes, how="diagonal_relaxed") if tapes else pl.DataFrame()
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    if not tape.is_empty():
        tape.write_parquet(path)
    return tape


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    p.add_argument("-i", "--intervals", nargs="+", default=list(ALL_INTERVALS))
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--from-tapes", action="store_true")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("-f", "--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    p.add_argument("--fade", choices=["none", "flip", "mirror"], default="none",
                   help="Addendum B: take the opposite side of every order")
    p.add_argument("--primary", default=PRIMARY,
                   help="the pre-registered exit; s0_obn for Addendum A")
    p.add_argument("--diagnostics", action="store_true",
                   help="pooled splits, exit curve, edge-on-cost, forward returns, "
                        "break type - from the saved tapes, no tick walk")
    args = p.parse_args(argv)
    if args.diagnostics:
        diag = diagnostics(args)
        (REPORT_DIR / "sweep_orderblock_diagnostics.json").write_text(
            json.dumps(diag, indent=2, default=str))
        return 0
    if "test" in args.splits:
        raise SystemExit("the test split is locked; the registration spends it only on a validation survivor")
    # A run on a subset of the setups gets its own tapes and report, so it can
    # never overwrite the main study's.
    args.tag = "" if set(args.families) == set(FAMILIES) else "_" + "-".join(args.families)
    if args.fade != "none":
        args.tag += f"_fade-{args.fade}"
    cells = len(args.families) * len(args.symbols) * len(args.intervals)
    sidak = NormalDist().inv_cdf((0.95) ** (1 / cells))
    key = args.primary
    print(f"primary {key}: {cells} cells, one-sided Sidak t >= {sidak:.2f}")

    results: dict = {"config": dict(ToolkitConfig(fade=args.fade).__dict__), "primary": key,
                     "families": args.families, "sidak": sidak, "splits": {}}
    for split in args.splits:
        swap = {}
        for s in args.symbols:
            try:
                swap[s] = basis(s, split=split).mid_bps
            except Exception as exc:  # noqa: BLE001
                print(f"  basis {s}: {exc}")
        print(f"\n{'=' * 110}\n{split.upper()}  (overnight basis bps/night: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in swap.items()) + ")\n" + "=" * 110)
        tape = collect(args, split, "none")
        rows = by_cell(tape, args.intervals, swap)
        entry = {"swap_bps": swap, "setup": rows}
        null_idx = None
        if not args.no_null:
            ntape = collect(args, split, "matched")
            nrows = by_cell(ntape, args.intervals, swap)
            entry["null"] = nrows
            null_idx = {(r["family"], r["symbol"], r["interval"]): r.get("exits") for r in nrows}
        if "ob" in args.families:
            nt = [r.get("no_target_rate") for r in rows if r["family"] == "ob"]
            nt = [v for v in nt if v is not None]
            if nt:
                print(f"\nblock orders with no opposite block on the chart: "
                      f"{min(nt):.3f} - {max(nt):.3f} across cells")
        print(f"\n--- primary cells at {key} (edge = win - fair; null = the matched null's edge) ---")
        print_cells(rows, key=key, null=null_idx)
        ladder_keys = [key] + [k for k in ("s0_obf", "s1_obn") if key == "s0_obn"]
        if args.fade == "mirror":
            ladder_keys += ["s0_t1", "s0_t3", "s0_obn"]
        for lk in ladder_keys:
            print(f"\n--- the ladder at {lk}, pooled over instruments ---")
            print_ladder(rows, key=lk)
            if not args.no_null:
                print(f"--- the matched null's ladder at {lk} ---")
                print_ladder(entry["null"], key=lk)
        entry["grid"] = grid_counts(rows)
        print(f"\n--- full exit grid: {entry['grid']}")
        passes = []
        for r in rows:
            e = (r.get("exits") or {}).get(key)
            ne = ((null_idx or {}).get((r["family"], r["symbol"], r["interval"])) or {}).get(key, {})
            # A cell with a handful of near-identical trades has a near-zero
            # standard deviation and a meaningless t; the registration reads the
            # slow rungs for sign only, so a pass needs a real sample of days.
            ok = bool(e) and e["mean_r"] > 0 and e["t"] >= sidak and e["edge"] > 0
            ok = ok and e["days"] >= MIN_DAYS and e["edge"] > ne.get("edge", -np.inf)
            if ok:
                passes.append((r["family"], r["symbol"], r["interval"], e["mean_r"], e["t"]))
        entry["primary_passes"] = passes
        print()
        print(f"--- primary passes (t >= {sidak:.2f}, >= {MIN_DAYS} days, edge > 0, edge > null): {passes or 'none'}")
        results["splits"][split] = entry

    if {"dev", "validation"} <= set(results["splits"]):
        print("\n--- pooled dev + validation ---")
        print(f"{'fam':5} {'variant':8} {'exit':8} {'trades':>8} {'meanR':>8} {'win':>6} "
              f"{'edge':>7} {'cost/R':>7} {'cells+':>7} {'USD':>11}")
        for fam in args.families:
            for v in ("setup", "null"):
                for k in ALL_EXIT_KEYS:
                    both = [r for s in ("dev", "validation")
                            for r in results["splits"][s].get(v, []) if r["family"] == fam]
                    p = _pool(both, k)
                    if p:
                        print(f"{fam:5} {v:8} {k:8} {p['trades']:8,} {p['mean_r']:+8.4f} "
                              f"{fmt(p['win'], '.3f', 6)} {fmt(p['edge'], '+.3f', 7)} "
                              f"{p['cost_r']:7.3f} {p['pos']:3}/{p['cells']:<3} {p['usd']:+11,.0f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"sweep_orderblock{args.tag}.json"
    if out.exists():
        try:
            prior = json.loads(out.read_text())
            merged = dict(prior.get("splits") or {})
            merged.update(results["splits"])
            results["splits"] = merged
        except json.JSONDecodeError:
            pass
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
