"""Optimise the risk-to-reward geometry of the session breakout.

The expert fixes the stop at the opposite edge of the bracket and the target at
3x its width, so the risk-to-reward ratio is whatever the bracket happens to
be. This frees both: the stop is placed a chosen multiple of the width from the
fill, the target another, and every combination is scored per unit of risk
actually taken.

Three questions, in the order that matters:

1. What does the surface look like on each period?
2. Does the optimum found on one period survive on the other? A surface with a
   sharp, unstable peak is a search over noise, whatever its best cell says.
3. Under honest walk-forward - choose on everything up to year Y, trade year
   Y+1 - does re-optimising beat leaving the geometry alone? This is the only
   question a live account cares about, and it is the one a grid search over a
   single period cannot answer.

Run:  python scripts/research/optimize_rr.py [-s SYMBOL] [--preset NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.strategies.session_breakout import (  # noqa: E402
    PRESETS,
    BreakoutConfig,
    sweep_exits,
)

STOPS = [0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TARGETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BASE = (1.0, 3.0)  # the expert's own geometry, near enough


def _daily(r: np.ndarray, inv: np.ndarray, n_days: int) -> np.ndarray:
    """Per-day totals for one cell - the unit the t-statistic is computed on."""
    return np.bincount(inv, weights=r, minlength=n_days)


def cell_stats(r: np.ndarray, inv: np.ndarray, n_days: int) -> tuple[float, float]:
    """Mean R per trade, and the daily t-statistic."""
    d = _daily(r, inv, n_days)
    if d.size < 3 or d.std(ddof=1) == 0:
        return float(r.mean()), 0.0
    return float(r.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def surface(res: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = res["r"][mask]
    days = res["day"][mask]
    _, inv = np.unique(days, return_inverse=True)
    n_days = inv.max() + 1 if inv.size else 0
    S, T = len(STOPS), len(TARGETS)
    mean = np.zeros((S, T))
    tstat = np.zeros((S, T))
    for i in range(S):
        for j in range(T):
            mean[i, j], tstat[i, j] = cell_stats(r[:, i, j], inv, n_days)
    return mean, tstat


def show(mat: np.ndarray, title: str, fmt: str = "%+.3f") -> None:
    print(f"\n{title}")
    head = "  stop\\tgt " + "".join(f"{t:>8g}" for t in TARGETS)
    print(head)
    best = np.unravel_index(np.argmax(mat), mat.shape)
    for i, s in enumerate(STOPS):
        cells = []
        for j in range(len(TARGETS)):
            txt = fmt % mat[i, j]
            if (i, j) == best:
                txt = f"[{txt}]"
            elif (s, TARGETS[j]) == BASE:
                txt = f"<{txt}>"
            cells.append(f"{txt:>8}")
        print(f"  {s:>8g} " + "".join(cells))
    print(f"  [best]  <expert's own geometry: stop {BASE[0]}x, target {BASE[1]}x>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--symbol", default="XAUUSD")
    ap.add_argument("--preset", default="GEO_2026_NO_H13")
    ap.add_argument("--min-width", type=float, default=0.05)
    args = ap.parse_args()

    windows = tuple((h, r, 3.0) for h, r, _ in PRESETS[args.preset])
    cfg = BreakoutConfig(windows=windows, min_range_pct=args.min_width)
    res = sweep_exits(args.symbol, cfg, STOPS, TARGETS,
                      start="2020-01-29", end="2026-09-02", allow_test=True,
                      cost=CostModel.from_profiles(args.symbol), verbose=False)

    years = np.array([d.year for d in res["day"]])
    dev, late = years < 2024, years >= 2024
    print(f"{args.symbol} / {args.preset} / min width {args.min_width}% "
          f"- {len(years)} entries, shared across all {len(STOPS)*len(TARGETS)} geometries")

    m_dev, t_dev = surface(res, dev)
    m_late, t_late = surface(res, late)
    show(m_dev, "=== mean R per unit risk - DEV 2020-2023 ===")
    show(m_late, "=== mean R per unit risk - 2024-2026 ===")
    show(t_dev, "=== daily t-statistic - DEV 2020-2023 ===", "%+.2f")
    show(t_late, "=== daily t-statistic - 2024-2026 ===", "%+.2f")

    # --- does the optimum survive the period it was not chosen on? ----------
    bi, bj = np.unravel_index(np.argmax(m_dev), m_dev.shape)
    li, lj = np.unravel_index(np.argmax(m_late), m_late.shape)
    base = (STOPS.index(BASE[0]), TARGETS.index(BASE[1]))
    print("\n=== does the optimum transfer? ===")
    print(f"  best on DEV   stop {STOPS[bi]:g} target {TARGETS[bj]:g} "
          f"(R:R {TARGETS[bj]/STOPS[bi]:.1f}) -> dev {m_dev[bi,bj]:+.4f}, "
          f"then late {m_late[bi,bj]:+.4f}")
    print(f"  best on LATE  stop {STOPS[li]:g} target {TARGETS[lj]:g} "
          f"(R:R {TARGETS[lj]/STOPS[li]:.1f}) -> late {m_late[li,lj]:+.4f}, "
          f"then dev  {m_dev[li,lj]:+.4f}")
    print(f"  expert's own  stop {BASE[0]:g} target {BASE[1]:g} "
          f"(R:R {BASE[1]/BASE[0]:.1f}) -> dev {m_dev[base]:+.4f}, "
          f"late {m_late[base]:+.4f}")
    rank = np.corrcoef(m_dev.ravel().argsort().argsort(),
                       m_late.ravel().argsort().argsort())[0, 1]
    print(f"  rank correlation of all {m_dev.size} cells across the two periods: {rank:+.2f}")

    # --- walk-forward: choose on the past, trade the next year --------------
    print("\n=== walk-forward: fit the geometry on everything before year Y, trade year Y ===")
    print("  year   chosen (stop,target)   R:R    optimised R    fixed 1x/3x R    n")
    opt_all, fix_all = [], []
    for y in range(2022, 2027):
        past, now = years < y, years == y
        if past.sum() < 500 or now.sum() < 100:
            continue
        m_past, _ = surface(res, past)
        i, j = np.unravel_index(np.argmax(m_past), m_past.shape)
        r_opt = res["r"][now][:, i, j]
        r_fix = res["r"][now][:, base[0], base[1]]
        opt_all.append(r_opt)
        fix_all.append(r_fix)
        print(f"  {y}   stop {STOPS[i]:<4g} target {TARGETS[j]:<4g}  "
              f"{TARGETS[j]/STOPS[i]:>4.1f}   {r_opt.mean():+9.4f}      "
              f"{r_fix.mean():+9.4f}   {now.sum():5d}")
    if opt_all:
        o, f = np.concatenate(opt_all), np.concatenate(fix_all)
        print(f"  {'pooled':<6} {'':<24}      {o.mean():+9.4f}      {f.mean():+9.4f}   {o.size:5d}")
        diff = o - f
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(diff.size))
        print(f"\n  re-optimising is worth {o.mean()-f.mean():+.4f} R per trade "
              f"against leaving it alone (paired t = {t:+.2f})")


if __name__ == "__main__":
    main()
