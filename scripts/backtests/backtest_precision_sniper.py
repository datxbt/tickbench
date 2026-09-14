"""Evaluate the Precision Sniper confluence engine, 1m to D1, four instruments.

Pre-registered in docs/findings/precision-sniper.md before any outcome was
computed. Every table is a pure function of the saved order tapes, so
``--from-tapes`` re-summarises without walking a tick.

Run:  python scripts/backtests/backtest_precision_sniper.py --dry-run
      python scripts/backtests/backtest_precision_sniper.py                 # dev + validation
      python scripts/backtests/backtest_precision_sniper.py -s USTEC -i 1h 4h 1d
      python scripts/backtests/backtest_precision_sniper.py --from-tapes
      python scripts/backtests/backtest_precision_sniper.py --filters --from-tapes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.rollover import basis  # noqa: E402
from qlab.strategies.precision_sniper import (  # noqa: E402
    ALL_INTERVALS, ALL_PRESETS, EXIT_KEYS, GRADE_A_R, GRADE_APLUS_R, GRADE_B_R,
    HTF_FOR, MAX_SCORE, PINE_KEYS, PRESETS, REASON_CODE, ROLL_KEYS,
    STOP_KEYS, TARGET_KEYS, TARGET_R, SniperConfig, auto_preset, fair_rate,
    run, sequence, sequence_reverse,
)
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

REPORT_DIR = paths.STRATEGY_REPORT_DIR
TAPE_DIR = REPORT_DIR / "precision_sniper_trades"

# The script's own nominated exit: the structure stop it defaults to, managed
# by its own trailing ladder with the full close at TP3.
PRIMARY = "struct_pL123"
CELL_KEYS = ("symbol", "interval", "preset", "htf_name")


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


def summarise(tape: pl.DataFrame, swap_bps: float, *,
              exits: tuple[str, ...] = EXIT_KEYS) -> dict:
    """Order economics, the raw signal, and every exit for one cell."""
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

    out["long_share"] = float((filled["direction"] == 1).mean())
    out["score_median"] = float(filled["score"].median())
    out["score_min"] = float(filled["score"].min())
    for g in ("A+", "A", "B", "C"):
        out[f"grade_{g}"] = float((filled["grade"] == g).mean())
    out["regime_high"] = float(filled["regime_high"].mean())
    out["htf_agree"] = float((filled["htf_bias"] == filled["direction"]).mean())
    for k in (1, 5, 20):
        col = f"fwd{k}_bps"
        f = filled.filter(pl.col(col).is_not_nan())
        out[f"{col}_mean"] = float(f[col].mean()) if f.height else float("nan")
        out[f"{col}_t"] = daily_t(f, col)[0]
    out["mfe_r_median"] = float(filled["mfe_r"].median())
    out["mae_r_median"] = float(filled["mae_r"].median())
    for sk in STOP_KEYS:
        out[f"cost_r_{sk}"] = float(filled[f"cost_r_{sk}"].median())
        out[f"risk_bps_{sk}"] = float(filled[f"risk_bps_{sk}"].median())

    taken_by = sequence(filled, [k for k in exits if f"r_{k}" in filled.columns])
    graded: dict[str, dict] = {}
    for key, taken in taken_by.items():
        if taken.is_empty():
            continue
        sk, tk = _split_key(key)
        col = f"r_{key}"
        rolls = f"rolls_{key}"
        if rolls in taken.columns:
            taken = taken.with_columns(
                (pl.col(col) - pl.col("direction") * pl.col(rolls)
                 * swap_bps / 1e4 * pl.col("entry") / pl.col(f"risk_{sk}")).alias("_rs"))
        else:
            taken = taken.with_columns(pl.col(col).alias("_rs"))
        t, days = daily_t(taken, col)
        reason = taken[f"reason_{key}"]
        win = float((reason == REASON_CODE["tp"]).mean())
        fair = fair_rate(TARGET_R[tk]) if tk in TARGET_R else float("nan")
        graded[key] = {
            "trades": taken.height, "days": days,
            "mean_r": float(taken[col].mean()), "t": t,
            "mean_r_swap": float(taken["_rs"].mean()),
            "t_swap": daily_t(taken, "_rs")[0],
            "win": win, "fair": fair, "edge": win - fair,
            "stop_rate": float((reason == REASON_CODE["stop"]).mean()),
            "time_rate": float((reason == REASON_CODE["time"]).mean()),
            "trail_rate": float(reason.is_in(
                [REASON_CODE[r] for r in ("trail1", "trail2", "trail3")]).mean()),
            "usd": float((taken[col].cast(pl.Float64) * taken[f"risk_{sk}"]
                          * taken["usd_per_px"]).sum()),
            "cost_r": float(taken[f"cost_r_{sk}"].median()),
        }
        if tk in PINE_KEYS:
            # The script's own account: stop-and-reverse, no pyramiding.
            rev = sequence_reverse(filled, key)
            if not rev.is_empty():
                rt, _ = daily_t(rev, "r_eff")
                graded[key].update(
                    rev_trades=rev.height,
                    rev_mean_r=float(rev["r_eff"].mean()),
                    rev_t=rt,
                    rev_cut_rate=float(rev["was_cut"].mean()),
                    rev_usd=float((rev["r_eff"] * rev[f"risk_{sk}"]
                                   * rev["usd_per_px"]).sum()))
    out["exits"] = graded
    return out


def _split_key(key: str) -> tuple[str, str]:
    """``struct_w_pL123`` -> (``struct_w``, ``pL123``)."""
    for sk in sorted(STOP_KEYS, key=len, reverse=True):
        if key.startswith(sk + "_"):
            return sk, key[len(sk) + 1:]
    raise KeyError(key)


def by_cell(tape: pl.DataFrame, swap: dict, *,
            keys: tuple[str, ...] = CELL_KEYS,
            exits: tuple[str, ...] = EXIT_KEYS) -> list[dict]:
    order = {iv: i for i, iv in enumerate(ALL_INTERVALS)}
    rows = []
    for vals, part in tape.group_by(list(keys)):
        cell = dict(zip(keys, vals))
        rows.append({**cell, **summarise(part, swap.get(cell.get("symbol"), 0.0),
                                         exits=exits)})
    rows.sort(key=lambda r: (r.get("preset", ""), r.get("symbol", ""),
                             order.get(r.get("interval"), 99), r.get("htf_name", "")))
    return rows


def _needed_columns(exits: tuple[str, ...]) -> list[str]:
    """The narrowest projection :func:`summarise` can work from.

    A whole-symbol tape on the finest grid is a couple of gigabytes, and the
    filter sweep reads every shard once. Reading three columns per exit
    instead of two hundred is the difference between a re-summarisation that
    takes a minute and one that takes an hour.
    """
    cols = ["symbol", "interval", "preset", "htf_name", "direction", "day",
            "filled", "fill_reason", "fill_us", "entry", "score", "score_r",
            "grade", "regime_high", "htf_bias", "mfe_r", "mae_r", "usd_per_px",
            "commission_px", "spread_fill", "slip_px"]
    cols += [f"fwd{k}_bps" for k in (1, 5, 20)]
    for sk in STOP_KEYS:
        cols += [f"risk_{sk}", f"risk_bps_{sk}", f"cost_r_{sk}"]
    for k in exits:
        cols += [f"r_{k}", f"reason_{k}", f"exit_us_{k}"]
        if _split_key(k)[1] in ROLL_KEYS:
            cols.append(f"rolls_{k}")
    return cols


def read_shard(path: Path, exits: tuple[str, ...] = EXIT_KEYS) -> pl.DataFrame:
    have = set(pl.scan_parquet(path).collect_schema().names())
    want = [c for c in dict.fromkeys(_needed_columns(exits)) if c in have]
    return pl.scan_parquet(path).select(want).collect()


def by_cell_shards(paths, swap: dict, *,
                   exits: tuple[str, ...] = EXIT_KEYS) -> list[dict]:
    """:func:`by_cell` over per-symbol shards, one in memory at a time.

    A cell never spans two shards - the symbol is part of the cell key - so
    summarising shard by shard gives exactly the rows that summarising the
    concatenated tape would.
    """
    rows: list[dict] = []
    for path in paths:
        frame = read_shard(path, exits)
        rows += by_cell(frame, swap, exits=exits)
        del frame
    order = {iv: i for i, iv in enumerate(ALL_INTERVALS)}
    rows.sort(key=lambda r: (r.get("preset", ""), r.get("symbol", ""),
                             order.get(r.get("interval"), 99), r.get("htf_name", "")))
    return rows



# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def fmt(v, spec=".3f", w=8) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(w)
    return format(v, spec).rjust(w)


def print_cells(rows, key=PRIMARY, null: dict | None = None, limit: int = 0) -> None:
    print(f"{'preset':12} {'symbol':7} {'tf':>4} {'htf':>4} {'sig':>7} {'fwd5':>7} {'t':>6} "
          f"{'cost/R':>7} {'trades':>7} {'win':>6} {'fair':>6} {'edge':>7} {'null':>7} "
          f"{'meanR':>8} {'t':>6} {'stop%':>6} {'time%':>6}")
    shown = 0
    for r in rows:
        e = (r.get("exits") or {}).get(key)
        if not e:
            continue
        idx = tuple(r.get(k) for k in CELL_KEYS)
        ne = ((null or {}).get(idx) or {}).get(key, {})
        print(f"{r['preset']:12} {r['symbol']:7} {r['interval']:>4} {r['htf_name']:>4} "
              f"{r['orders']:7,} {fmt(r.get('fwd5_bps_mean'), '+.2f', 7)} "
              f"{fmt(r.get('fwd5_bps_t'), '.1f', 6)} {fmt(e['cost_r'], '.3f', 7)} "
              f"{e['trades']:7,} {fmt(e['win'], '.3f', 6)} {fmt(e['fair'], '.3f', 6)} "
              f"{fmt(e['edge'], '+.3f', 7)} {fmt(ne.get('edge'), '+.3f', 7)} "
              f"{fmt(e['mean_r'], '+.4f', 8)} {fmt(e['t'], '.2f', 6)} "
              f"{fmt(e['stop_rate'], '.3f', 6)} {fmt(e['time_rate'], '.3f', 6)}")
        shown += 1
        if limit and shown >= limit:
            print(f"  ... {sum(1 for x in rows if key in (x.get('exits') or {})) - shown} more")
            break


def _pool(rows, key) -> dict:
    es = [r["exits"][key] for r in rows if key in (r.get("exits") or {})]
    n = sum(e["trades"] for e in es)
    if not n:
        return {}
    w = lambda f: sum(e[f] * e["trades"] for e in es) / n  # noqa: E731
    return {"trades": n, "mean_r": w("mean_r"), "win": w("win"), "edge": w("edge"),
            "cost_r": w("cost_r"), "mean_r_swap": w("mean_r_swap"),
            "stop_rate": w("stop_rate"), "time_rate": w("time_rate"),
            "usd": sum(e["usd"] for e in es),
            "pos": sum(e["mean_r"] > 0 for e in es), "cells": len(es)}


def print_reverse(rows, null_rows=None) -> None:
    """The script's own account, pooled - its Pine ladders only."""
    print(f"{'exit':14} {'trades':>9} {'cut%':>6} {'meanR':>9} {'t':>7} "
          f"{'nullR':>9} {'flatR':>9} {'USD':>12}")
    for k in EXIT_KEYS:
        if _split_key(k)[1] not in PINE_KEYS:
            continue
        es = [r["exits"][k] for r in rows
              if k in (r.get("exits") or {}) and "rev_trades" in r["exits"][k]]
        if not es:
            continue
        n = sum(e["rev_trades"] for e in es)
        if not n:
            continue
        w = lambda f: sum(e[f] * e["rev_trades"] for e in es) / n  # noqa: E731
        ns = [r["exits"][k] for r in (null_rows or [])
              if k in (r.get("exits") or {}) and "rev_trades" in r["exits"][k]]
        nn = sum(e["rev_trades"] for e in ns)
        nr = (sum(e["rev_mean_r"] * e["rev_trades"] for e in ns) / nn) if nn else None
        flat = _pool(rows, k)
        ts = [e["rev_t"] for e in es if np.isfinite(e.get("rev_t", float("nan")))]
        print(f"{k:14} {n:9,} {w('rev_cut_rate'):6.3f} {w('rev_mean_r'):+9.4f} "
              f"{fmt(np.mean(ts) if ts else None, '.2f', 7)} {fmt(nr, '+.4f', 9)} "
              f"{fmt(flat.get('mean_r'), '+.4f', 9)} "
              f"{sum(e['rev_usd'] for e in es):+12,.0f}")


def print_pool(rows, keys, label: str, null_rows=None) -> None:
    print(f"{label:14} {'trades':>9} {'cost/R':>7} {'win':>6} {'fair':>6} {'edge':>7} "
          f"{'meanR':>8} {'nullR':>8} {'stop%':>6} {'time%':>6} {'cells+':>9} {'USD':>11}")
    for k in keys:
        p = _pool(rows, k)
        if not p:
            continue
        npool = _pool(null_rows, k) if null_rows else {}
        sk, tk = _split_key(k)
        fair = fair_rate(TARGET_R[tk]) if tk in TARGET_R else float("nan")
        print(f"{k:14} {p['trades']:9,} {p['cost_r']:7.3f} {fmt(p['win'], '.3f', 6)} "
              f"{fmt(fair, '.3f', 6)} {fmt(p['edge'], '+.3f', 7)} {p['mean_r']:+8.4f} "
              f"{fmt(npool.get('mean_r'), '+.4f', 8)} {p['stop_rate']:6.3f} "
              f"{p['time_rate']:6.3f} {p['pos']:4}/{p['cells']:<4} {p['usd']:+11,.0f}")


def print_ladder(rows, key=PRIMARY, by: str = "interval") -> None:
    """Pooled over everything but one axis, trade-weighted."""
    levels = {"interval": ALL_INTERVALS, "preset": ALL_PRESETS,
              "symbol": ALL_SYMBOLS, "htf_name": ("none", *sorted(set(HTF_FOR.values())))}
    print(f"{by:14} {'trades':>9} {'cost/R':>7} {'win':>6} {'edge':>7} {'meanR':>8} "
          f"{'t>=2':>6} {'t<=-2':>6} {'cells':>6}")
    for lv in levels[by]:
        sub = [r for r in rows if r.get(by) == lv]
        p = _pool(sub, key)
        if not p:
            continue
        ts = [r["exits"][key]["t"] for r in sub if key in (r.get("exits") or {})]
        ts = [t for t in ts if np.isfinite(t)]
        print(f"{str(lv):14} {p['trades']:9,} {p['cost_r']:7.3f} {fmt(p['win'], '.3f', 6)} "
              f"{fmt(p['edge'], '+.3f', 7)} {p['mean_r']:+8.4f} "
              f"{sum(t >= 2 for t in ts):6} {sum(t <= -2 for t in ts):6} {p['cells']:6}")


def grid_counts(rows, sidak: float) -> dict:
    ts = [e["t"] for r in rows for e in (r.get("exits") or {}).values()
          if np.isfinite(e["t"])]
    pos = [e for r in rows for e in (r.get("exits") or {}).values() if e["mean_r"] > 0]
    return {"combos": len(ts), "positive": len(pos),
            "t_ge_2": sum(t >= 2 for t in ts), "t_le_m2": sum(t <= -2 for t in ts),
            "t_ge_sidak": sum(t >= sidak for t in ts),
            "t_le_m_sidak": sum(t <= -sidak for t in ts)}


# --------------------------------------------------------------------------
# The script's own filters, applied to a resolved tape
# --------------------------------------------------------------------------

GRADE_FILTERS: dict[str, float] = {
    "All": 0.0, "A+ and A": GRADE_A_R, "A+ Only": GRADE_APLUS_R,
}
VOL_MODES: tuple[str, ...] = ("Off", "Skip Signals")
MIN_SCORES: tuple[float, ...] = (0.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def filter_tape(tape: pl.DataFrame, *, grade: str = "All", hide_c: bool = True,
                vol_mode: str = "Off", min_score: float = 0.0) -> pl.DataFrame:
    """The four switches the script exposes, as one predicate.

    ``min_score`` is on the script's 10-point scale and is rescaled the way the
    script rescales it, so 5.0 here means the same thing it means in the input.
    """
    out = tape
    floor = max(GRADE_FILTERS[grade], GRADE_B_R if hide_c else 0.0)
    if floor > 0:
        out = out.filter(pl.col("score_r") >= floor)
    if vol_mode == "Skip Signals":
        out = out.filter(~pl.col("regime_high"))
    if min_score > 0:
        out = out.filter(pl.col("score") >= min_score * MAX_SCORE / 10.0)
    return out


def filter_sweep(paths, swap: dict, key: str) -> list[dict]:
    """Every combination of the script's own switches, at one exit rule.

    Each shard is read once, narrowed to the columns this one exit needs; the
    seventy-two switch combinations are then filters over frames in memory.
    """
    frames = [read_shard(p, (key,)) for p in paths]
    rows = []
    for g in GRADE_FILTERS:
        for hide_c in (True, False):
            for vm in VOL_MODES:
                for ms in MIN_SCORES:
                    cells = []
                    for frame in frames:
                        sub = filter_tape(frame, grade=g, hide_c=hide_c,
                                          vol_mode=vm, min_score=ms)
                        if not sub.is_empty():
                            cells += by_cell(sub, swap, exits=(key,))
                    p = _pool(cells, key)
                    if not p or p["trades"] < 200:
                        continue
                    rows.append({"grade": g, "hide_c": hide_c, "vol": vm,
                                 "min_score": ms, **p})
    rows.sort(key=lambda r: -r["mean_r"])
    return rows


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def shard_path(symbol, split, control, tag) -> Path:
    return TAPE_DIR / f"ps_{control}_{split}_{symbol}{tag}.parquet"


def _one(job):
    symbol, split, control, intervals, presets, htf_modes, tag = job
    path = shard_path(symbol, split, control, tag)
    t0 = time.time()
    run(symbol, SniperConfig(control=control), intervals=intervals,
        presets=presets, htf_modes=htf_modes, split=split, sink=path)
    n = pl.scan_parquet(path).select(pl.len()).collect().item() if path.exists() else 0
    print(f"  {symbol} {split} {control}: {n:,} rows in "
          f"{time.time() - t0:.0f}s -> {path.name}", flush=True)
    return path if n else None


def collect(args, split, control) -> list[Path]:
    """One parquet shard per symbol. The parent never holds more than one."""
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [shard_path(s, split, control, args.tag) for s in args.symbols]
    if args.from_tapes and all(p.exists() for p in paths):
        return paths
    jobs = [(s, split, control, args.intervals, args.presets, args.htf_modes,
             args.tag) for s in args.symbols]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        return [p for p in pool.map(_one, jobs) if p is not None]


def dry_run(args) -> int:
    """Count signals without resolving a tick - how big the run is, before it runs."""
    from qlab.loader import load_bars
    from qlab.strategies.precision_sniper import (
        indicators, signals, INTERVAL_MINUTES,
    )
    from qlab.strategies.sweep_orderblock import build_bars

    total = 0
    per_axis: dict[tuple, int] = {}
    for split in args.splits:
        for sym in args.symbols:
            base = load_bars(sym, "1m", split=split,
                             columns=["ts", "ts_open", "open", "high", "low",
                                      "close", "n_ticks"])
            need = sorted({*args.intervals, *HTF_FOR.values()},
                          key=lambda k: INTERVAL_MINUTES[k])
            built = {iv: build_bars(base, iv) for iv in need}
            del base
            for iv in args.intervals:
                bars = built[iv]
                for pn in args.presets:
                    ind = indicators(bars, PRESETS[pn], SniperConfig())
                    for mode in args.htf_modes:
                        hn = HTF_FOR[iv] if mode == "auto" else "none"
                        htf = built.get(hn) if mode == "auto" else None
                        if htf is not None and INTERVAL_MINUTES[hn] <= INTERVAL_MINUTES[iv]:
                            htf, hn = None, "none"
                        n = signals(bars, PRESETS[pn], SniperConfig(), interval=iv,
                                    htf=htf, htf_name=hn, ind=ind).height
                        total += n
                        per_axis[(split, sym, iv, pn, hn)] = n
            print(f"  {split} {sym}: running total {total:,}", flush=True)
    print(f"\n--- signals by timeframe (all splits/symbols/presets/htf) ---")
    for iv in args.intervals:
        n = sum(v for k, v in per_axis.items() if k[2] == iv)
        print(f"  {iv:>4} {n:10,}")
    print(f"--- signals by preset ---")
    for pn in args.presets:
        n = sum(v for k, v in per_axis.items() if k[3] == pn)
        print(f"  {pn:14} {n:10,}")
    print(f"\nTOTAL signals to resolve: {total:,}")
    print(f"exit grid: {len(STOP_KEYS)} stops x {len(TARGET_KEYS)} targets "
          f"= {len(EXIT_KEYS)} outcomes per signal")
    return 0


# --------------------------------------------------------------------------
# Diagnostics - pure functions of the saved report, no tick walk
# --------------------------------------------------------------------------

def _w(es, field, weight="trades") -> float:
    n = sum(e[weight] for e in es)
    return sum(e[field] * e[weight] for e in es) / n if n else float("nan")


def mean_r_on_cost(rows, key) -> tuple[float, float, int]:
    """Trade-weighted fit of mean R = a + b * (cost/R) across cells.

    Both barriers move against the trade in proportion to what the round turn
    costs relative to the stop, so a cell's mean R falls roughly linearly in
    cost/R. The intercept is what the rule would return at zero cost - the
    only way to compare cells whose stops are wildly different sizes, and the
    only fair comparison against a null whose stops are the same size but
    whose bars were picked at random.
    """
    pts = [(e["cost_r"], e["mean_r"], e["trades"]) for r in rows
           for k, e in (r.get("exits") or {}).items()
           if k == key and np.isfinite(e["mean_r"]) and np.isfinite(e["cost_r"])]
    if len(pts) < 3:
        return float("nan"), float("nan"), len(pts)
    x, y, w = (np.array(v, float) for v in zip(*pts))
    sw = np.sqrt(w)
    a, b = np.linalg.lstsq(np.c_[sw, sw * x], sw * y, rcond=None)[0]
    return float(a), float(b), len(pts)


def diagnostics(args) -> dict:
    path = REPORT_DIR / f"precision_sniper{args.tag}.json"
    if not path.exists():
        raise SystemExit(f"no report at {path}; run the sweep first")
    saved = json.loads(path.read_text())
    key = args.primary
    out: dict = {"primary": key}

    print(f"\n--- {key}: the setup against its matched null, by timeframe ---")
    print(f"{'split':11} {'tf':>4} {'trades':>9} {'cost/R':>7} {'setupR':>9} "
          f"{'nullR':>9} {'diff':>9} {'setup win':>10} {'null win':>9}")
    for split, entry in saved["splits"].items():
        for iv in ALL_INTERVALS:
            es = [r["exits"][key] for r in entry.get("setup", [])
                  if r["interval"] == iv and key in (r.get("exits") or {})]
            ns = [r["exits"][key] for r in entry.get("null", [])
                  if r["interval"] == iv and key in (r.get("exits") or {})]
            if not es or not ns:
                continue
            s_r, n_r = _w(es, "mean_r"), _w(ns, "mean_r")
            out[f"{split}/{iv}"] = [s_r, n_r, s_r - n_r]
            print(f"{split:11} {iv:>4} {sum(e['trades'] for e in es):9,} "
                  f"{_w(es, 'cost_r'):7.3f} {s_r:+9.4f} {n_r:+9.4f} "
                  f"{s_r - n_r:+9.4f} {_w(es, 'win'):10.3f} {_w(ns, 'win'):9.3f}")

    print(f"\n--- mean R = a + b (cost/R), trade-weighted across cells ---")
    print(f"{'split':11} {'variant':9} {'exit':15} {'a (zero cost)':>14} "
          f"{'b':>9} {'cells':>7}")
    for split, entry in saved["splits"].items():
        for k in (key, key.split("_")[0] + "_t1", "struct_hold"):
            for variant in ("setup", "null"):
                rows = entry.get(variant, [])
                if not rows:
                    continue
                a, b, n = mean_r_on_cost(rows, k)
                if not np.isfinite(a):
                    continue
                out[f"fit/{split}/{variant}/{k}"] = [a, b, n]
                print(f"{split:11} {variant:9} {k:15} {a:+14.4f} {b:+9.3f} {n:7}")

    print(f"\n--- the raw signal: forward mid return, no barrier, no cost ---")
    print(f"{'split':11} {'variant':9} {'tf':>4} {'fwd1 bps':>9} {'t':>7} "
          f"{'fwd5 bps':>9} {'t':>7} {'fwd20 bps':>10} {'t':>7}")
    for split, entry in saved["splits"].items():
        for variant in ("setup", "null"):
            for iv in ALL_INTERVALS:
                rows = [r for r in entry.get(variant, [])
                        if r["interval"] == iv and r.get("fills")]
                if not rows:
                    continue
                cells = [{**r, "trades": r["fills"]} for r in rows]
                vals = []
                for h in (1, 5, 20):
                    m = [c for c in cells if np.isfinite(c.get(f"fwd{h}_bps_mean", np.nan))]
                    ts = [c[f"fwd{h}_bps_t"] for c in m
                          if np.isfinite(c.get(f"fwd{h}_bps_t", np.nan))]
                    vals += [_w(m, f"fwd{h}_bps_mean") if m else float("nan"),
                             float(np.mean(ts)) if ts else float("nan")]
                out[f"fwd/{split}/{variant}/{iv}"] = vals
                print(f"{split:11} {variant:9} {iv:>4} {vals[0]:+9.3f} {vals[1]:+7.2f} "
                      f"{vals[2]:+9.3f} {vals[3]:+7.2f} {vals[4]:+10.3f} {vals[5]:+7.2f}")

    print(f"\n--- pooled dev + validation, every exit, setup vs null ---")
    both = [r for s in saved["splits"].values() for r in s.get("setup", [])]
    bnull = [r for s in saved["splits"].values() for r in s.get("null", [])]
    print_pool(both, EXIT_KEYS, "exit", null_rows=bnull or None)
    out["pooled"] = {k: _pool(both, k) for k in EXIT_KEYS}
    out["pooled_null"] = {k: _pool(bnull, k) for k in EXIT_KEYS}
    return out

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    p.add_argument("-i", "--intervals", nargs="+", default=list(ALL_INTERVALS))
    p.add_argument("-p", "--presets", nargs="+", default=list(ALL_PRESETS))
    p.add_argument("--htf-modes", nargs="+", default=["auto", "none"])
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--from-tapes", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="count signals, resolve nothing")
    p.add_argument("--diagnostics", action="store_true",
                   help="setup vs null by timeframe, the zero-cost fit and the "
                        "raw forward returns - from the saved report, no tick walk")
    p.add_argument("--filters", action="store_true",
                   help="sweep the script's own grade/volatility/min-score switches")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("--primary", default=PRIMARY)
    args = p.parse_args(argv)

    if "test" in args.splits:
        raise SystemExit("the test split is locked; the registration spends it "
                         "only on a validation survivor")
    if args.dry_run:
        return dry_run(args)

    full = (set(args.symbols) == set(ALL_SYMBOLS)
            and set(args.intervals) == set(ALL_INTERVALS)
            and set(args.presets) == set(ALL_PRESETS))
    args.tag = "" if full else "_sub"

    if args.diagnostics:
        diag = diagnostics(args)
        (REPORT_DIR / f"precision_sniper_diagnostics{args.tag}.json").write_text(
            json.dumps(diag, indent=2, default=str))
        return 0

    cells = (len(args.symbols) * len(args.intervals) * len(args.presets)
             * len(args.htf_modes))
    grid = cells * len(EXIT_KEYS)
    sidak = NormalDist().inv_cdf(0.95 ** (1 / cells))
    sidak_grid = NormalDist().inv_cdf(0.95 ** (1 / grid))
    key = args.primary
    print(f"primary {key}: {cells:,} cells, one-sided Sidak t >= {sidak:.2f}")
    print(f"full grid:  {grid:,} cells, one-sided Sidak t >= {sidak_grid:.2f}")

    results: dict = {"config": dict(SniperConfig().__dict__), "primary": key,
                     "cells": cells, "grid": grid, "sidak": sidak,
                     "sidak_grid": sidak_grid, "splits": {}}
    for split in args.splits:
        swap = {}
        for s in args.symbols:
            try:
                swap[s] = basis(s, split=split).mid_bps
            except Exception as exc:  # noqa: BLE001
                print(f"  basis {s}: {exc}")
        print(f"\n{'=' * 120}\n{split.upper()}  (overnight basis bps/night: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in swap.items()) + ")\n" + "=" * 120)
        paths = collect(args, split, "none")
        if not paths:
            print("  no signals")
            continue
        rows = by_cell_shards(paths, swap)
        entry = {"swap_bps": swap, "setup": rows}
        null_idx, nrows = None, None
        if not args.no_null:
            npaths = collect(args, split, "matched")
            if npaths:
                nrows = by_cell_shards(npaths, swap)
                entry["null"] = nrows
                null_idx = {tuple(r.get(k) for k in CELL_KEYS): r.get("exits")
                            for r in nrows}

        print(f"\n--- the exit curve at every stop, flat account, pooled ---")
        print_pool(rows, EXIT_KEYS, "exit", null_rows=nrows)

        print(f"\n--- the script's own account (stop-and-reverse), pooled ---")
        print_reverse(rows, null_rows=nrows)

        for axis in ("interval", "preset", "symbol", "htf_name"):
            print(f"\n--- {key} by {axis}, pooled ---")
            print_ladder(rows, key=key, by=axis)

        print(f"\n--- the Auto preset's own cells at {key} ---")
        auto = [r for r in rows if r["preset"] == auto_preset(r["interval"])
                and r["htf_name"] != "none"]
        print_cells(auto, key=key, null=null_idx)

        entry["grid"] = grid_counts(rows, sidak_grid)
        print(f"\n--- full exit grid: {entry['grid']}")

        passes = []
        for r in rows:
            e = (r.get("exits") or {}).get(key)
            ne = ((null_idx or {}).get(tuple(r.get(k) for k in CELL_KEYS)) or {}).get(key, {})
            if e and e["mean_r"] > 0 and e["t"] >= sidak and \
                    e["mean_r"] > ne.get("mean_r", -np.inf):
                passes.append({**{k: r[k] for k in CELL_KEYS},
                               "mean_r": e["mean_r"], "t": e["t"],
                               "trades": e["trades"]})
        entry["primary_passes"] = passes
        print(f"\n--- primary passes (t >= {sidak:.2f}, meanR > 0, > null): "
              f"{len(passes)}")
        for x in passes[:20]:
            print(f"    {x}")

        # every cell of the whole grid that clears the grid-wide threshold
        wide = []
        for r in rows:
            for ek, e in (r.get("exits") or {}).items():
                if e["mean_r"] > 0 and np.isfinite(e["t"]) and e["t"] >= sidak_grid:
                    wide.append({**{k: r[k] for k in CELL_KEYS}, "exit": ek,
                                 "mean_r": e["mean_r"], "t": e["t"],
                                 "trades": e["trades"]})
        wide.sort(key=lambda x: -x["t"])
        entry["grid_passes"] = wide
        print(f"--- grid passes (t >= {sidak_grid:.2f}): {len(wide)}")
        for x in wide[:20]:
            print(f"    {x}")

        if args.filters:
            print(f"\n--- the script's own switches at {key}, pooled "
                  f"(best 25 of the sweep) ---")
            fs = filter_sweep(paths, swap, key)
            entry["filters"] = fs
            print(f"{'grade':10} {'hideC':>6} {'vol':>13} {'minScore':>8} "
                  f"{'trades':>10} {'win':>6} {'meanR':>8} {'USD':>12}")
            for r in fs[:25]:
                print(f"{r['grade']:10} {str(r['hide_c']):>6} {r['vol']:>13} "
                      f"{r['min_score']:>8.1f} {r['trades']:>10,} {r['win']:>6.3f} "
                      f"{r['mean_r']:>+8.4f} {r['usd']:>+12,.0f}")
        results["splits"][split] = entry

    if {"dev", "validation"} <= set(results["splits"]):
        print("\n" + "=" * 120 + "\nPOOLED dev + validation\n" + "=" * 120)
        both = [r for s in ("dev", "validation") for r in results["splits"][s]["setup"]]
        bnull = [r for s in ("dev", "validation")
                 for r in results["splits"][s].get("null", [])]
        print_pool(both, EXIT_KEYS, "exit", null_rows=bnull or None)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"precision_sniper{args.tag}.json"
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
