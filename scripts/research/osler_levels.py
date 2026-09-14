"""Osler's round-number mechanism on the three instruments gold's study left out.

Osler (2003, 2005) read a real FX dealer's order book: take-profit orders
cluster *at* round numbers, stop-loss orders just *beyond* them. Price should
revert at a round level that holds and run through one that breaks. The
project tested this only on gold (``level-interaction.md``), where reversion
came out backwards. Osler's evidence is FX, so the untested instruments are
exactly the ones the paper is about. The harness is ``qlab.levels`` and
``qlab.eventstudy`` unchanged; only the grids are new.

Pre-registered before the run (2026-09-11):

* Grids, nested fine -> coarse, round levels only:
  EURUSD 0.001 / 0.005 / 0.010 (10 pips, 50 pips, the big figure),
  USDJPY 0.10 / 0.50 / 1.00, USTEC 50 / 100 / 500 points. 5-minute bars, the
  gold study's default thresholds, nothing tuned.
* Direction is the mechanism's: away from the level for a touch or a sweep,
  with it for a breach.
* PRIMARY CELLS: coarsest grid x {touch, sweep, breach} x three instruments =
  nine cells. Statistic: net bps summed within each day, t across days, at
  30 minutes. PASS: t >= 2.77 (Bonferroni over nine) and the same sign on dev
  and validation.
* Ordering check: the effect should grow as the grid coarsens.
* Control: the 12-draw placebo that moves each event 1-5 days at the same time
  of day, compared at the mid.

Dev and validation only; nothing here can reach the test split.

    python scripts/research/osler_levels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.bars import resample_bars  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import daily_pnl, forward_returns, placebo_events, summarize  # noqa: E402
from qlab.levels import KINDS, LevelConfig, level_events  # noqa: E402
from qlab.loader import load_bars  # noqa: E402

GRIDS = {
    "EURUSD": (0.001, 0.005, 0.010),
    "USDJPY": (0.10, 0.50, 1.00),
    "USTEC": (50.0, 100.0, 500.0),
}
SPLITS = ("dev", "validation")
INTERVAL = "5m"
PRIMARY_H = 6  # bars: 30 minutes
REPORT_H = (1, 6, 24)
PLACEBO_SEEDS = tuple(range(12))
OUT = Path("reports/osler_levels")


def outcomes(symbol: str, cfg: LevelConfig, split: str, placebo_seed: int | None = None) -> pl.DataFrame:
    bars = resample_bars(load_bars(symbol, "1m", split=split), INTERVAL)
    events = level_events(bars, cfg)
    if placebo_seed is not None:
        events = placebo_events(events, bars, seed=placebo_seed)
    out = forward_returns(bars, events, cfg.horizons, cost=CostModel.from_profiles(symbol, split=split),
                          keep=("kind", "family"))
    return out.with_columns(split=pl.lit(split))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol, grids in GRIDS.items():
        cfg = LevelConfig(grids=grids, families=("round",))
        real = pl.concat([outcomes(symbol, cfg, s) for s in SPLITS])
        print(f"\n== {symbol}  grids {grids}   ({cfg.describe()})")
        print(f"  {'family':<12}{'kind':<8}{'min':>5}{'n':>7}{'mid':>8}{'net':>8}"
              f"{'bps/day':>9}{'t/day':>7}{'dev t':>7}{'val t':>7}")
        for grid in grids:
            fam = f"round_{grid:g}"
            for kind in KINDS:
                for h in REPORT_H:
                    cell = real.filter((pl.col("family") == fam) & (pl.col("kind") == kind)
                                       & (pl.col("horizon") == h))
                    if cell.height < 10:
                        continue
                    s = summarize(cell, column="net_bps")
                    mid = cell["mid_bps"].mean()
                    per_day, t_day, n_days = daily_pnl(cell)
                    ts = {sp: daily_pnl(cell.filter(pl.col("split") == sp))[1] for sp in SPLITS}
                    primary = grid == grids[-1] and h == PRIMARY_H
                    print(f"  {fam:<12}{kind:<8}{cell['minutes'][0]:>5}{s['n'][0]:>7,}{mid:>+8.2f}"
                          f"{s['mean'][0]:>+8.2f}{per_day:>+9.2f}{t_day:>+7.2f}"
                          f"{ts['dev']:>+7.2f}{ts['validation']:>+7.2f}" + ("  <- primary" if primary else ""))
                    rows.append({"symbol": symbol, "family": fam, "kind": kind, "minutes": int(cell["minutes"][0]),
                                 "n": int(s["n"][0]), "mid": float(mid), "net": float(s["mean"][0]),
                                 "per_day": per_day, "t_day": t_day, "dev_t": ts["dev"],
                                 "val_t": ts["validation"], "primary": primary})

        # Placebo on the coarsest grid only, at the mid: the level is the only
        # thing removed, so real minus placebo is what the level is worth.
        coarse = LevelConfig(grids=(grids[-1],), families=("round",))
        fake = [pl.concat([outcomes(symbol, coarse, s, seed) for s in SPLITS]) for seed in PLACEBO_SEEDS]
        fam = f"round_{grids[-1]:g}"
        print(f"  placebo, {fam}, 30 min, mid bps per event: real vs 12 draws [min, max]")
        for kind in KINDS:
            r = real.filter((pl.col("family") == fam) & (pl.col("kind") == kind) & (pl.col("horizon") == PRIMARY_H))
            draws = [f.filter((pl.col("kind") == kind) & (pl.col("horizon") == PRIMARY_H))["mid_bps"].mean()
                     for f in fake]
            draws = [d for d in draws if d is not None]
            if r.height and draws:
                print(f"    {kind:<8} real {r['mid_bps'].mean():+.3f}   placebo {sum(draws)/len(draws):+.3f}"
                      f" [{min(draws):+.3f}, {max(draws):+.3f}]")

    (OUT / "osler.json").write_text(json.dumps(rows, indent=1, default=str))
    prim = [r for r in rows if r["primary"]]
    print("\nPrimary cells (pass: t/day >= 2.77 and same sign in both splits):")
    for r in prim:
        ok = r["t_day"] >= 2.77 and r["dev_t"] > 0 and r["val_t"] > 0
        print(f"  {r['symbol']:<7}{r['kind']:<8} t/day {r['t_day']:+.2f}  dev {r['dev_t']:+.2f}  val {r['val_t']:+.2f}"
              f"  {'PASS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
