"""Evaluate the risk-managed long overlay on USTEC.

Default run compares the candidate set on dev and validation. The held-out test
split needs ``--test``, which is deliberately awkward to type by accident.

    python scripts/backtests/backtest_risk_managed_long.py
    python scripts/backtests/backtest_risk_managed_long.py --swap 1.5
    python scripts/backtests/backtest_risk_managed_long.py --test --json reports/...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.metrics import format_returns_tearsheet, tearsheet_from_returns  # noqa: E402
from qlab.strategies import risk_managed_long as rml  # noqa: E402

# The candidate set, fixed before any of it was evaluated. Each line is one
# structural claim, not one point on a parameter grid: whether to size by
# volatility at all, whether the trend gate should exit or merely de-risk, and
# whether the no-trade band pays for itself.
CANDIDATES: dict[str, rml.RMLConfig] = {
    "A vol-target only":       rml.RMLConfig(gate_floor=1.0, band=0.10),
    "B trend gate only":       rml.RMLConfig(target_vol_pct=0.0, gate_floor=0.0),
    "C gate + vol-target":     rml.RMLConfig(gate_floor=0.0, band=0.10),
    "D gate half + vol-target": rml.RMLConfig(gate_floor=0.5, band=0.10),
    "E C, no band":            rml.RMLConfig(gate_floor=0.0, band=0.0),
}


def _fixed_weight(frame: pl.DataFrame, weight: float, label: str,
                  swap: float) -> dict:
    """Benchmark: a constant notional, rebalanced never.

    It pays swap on every night it is held, exactly as the strategy does.
    Charging financing to the strategy and not to the thing it is measured
    against would make any rule that spends time flat look good for free.
    """
    nights = frame["nights"].fill_null(1).to_numpy()
    r = frame["gross_bps"].to_numpy() * weight - swap * weight * nights
    sheet = tearsheet_from_returns(r, label=label)
    sheet["avg_weight"] = weight
    sheet["turnover_per_year"] = 0.0
    return sheet


def evaluate(split: str, cfgs: dict, *, swap: float, allow_test: bool,
             stress: bool) -> tuple[list[dict], pl.DataFrame | None]:
    sheets: list[dict] = []
    kept: pl.DataFrame | None = None
    base = None
    for label, cfg in cfgs.items():
        cfg = replace(cfg, swap_bps_per_night=swap)
        if cfg.target_vol_pct == 0.0:
            # "trend gate only" means unit notional whenever the gate is on.
            cfg = replace(cfg, target_vol_pct=1e9, max_weight=1.0)
        cost = CostModel.from_profiles("USTEC", split=split)
        if stress:
            cost = cost.stressed(spread_multiplier=3.0, adverse_fraction=1.0)
        frame = rml.run("USTEC", cfg, split=split, allow_test=allow_test, cost=cost)
        sheet = tearsheet_from_returns(
            frame["net_bps"], label=label, turnover=frame["turnover"]
        )
        sheet["avg_weight"] = float(frame["weight"].mean())
        sheet["cost_bps_per_year"] = float(frame["cost_bps"].sum()) / sheet["years"]
        sheet["swap_bps_per_year"] = float(frame["swap_bps"].sum()) / sheet["years"]
        sheets.append(sheet)
        if base is None:
            base = frame
        if label.startswith("C "):
            kept = frame
    assert base is not None
    sheets.insert(0, _fixed_weight(base, 1.0, "0 buy and hold (1x)", swap))
    return sheets, kept


def table(sheets: list[dict]) -> str:
    head = (f"{'strategy':<26}{'n':>6}{'CAGR%':>8}{'vol%':>7}{'SR':>6}"
            f"{'maxDD%':>9}{'Calmar':>8}{'total%':>9}{'turn/yr':>9}{'avg w':>7}")
    lines = [head, "-" * len(head)]
    for s in sheets:
        lines.append(
            f"{s['label']:<26}{s['periods']:>6}{s['cagr_pct']:>8.2f}"
            f"{s['ann_vol_pct']:>7.2f}{s['sharpe']:>6.2f}{s['max_dd_pct']:>9.1f}"
            f"{s['calmar']:>8.2f}{s['total_pct']:>9.1f}"
            f"{s.get('turnover_per_year', float('nan')):>9.1f}"
            f"{s.get('avg_weight', float('nan')):>7.2f}"
        )
    return "\n".join(lines)


def swap_breakeven(frame: pl.DataFrame) -> str:
    """What overnight financing rate would take the whole return away."""
    years = frame.height / 252
    gross = float((frame["weight"] * frame["gross_bps"]).sum())
    costs = float(frame["cost_bps"].sum())
    nights = float((frame["weight"] * frame["nights"].fill_null(1)).sum())
    if nights <= 0:
        return "no overnight exposure"
    be = (gross - costs) / nights
    rows = ["  swap (bps/night)   annual drag on the strategy   CAGR left"]
    net_bps = frame["net_bps"].to_numpy()
    for s in (0.0, 0.5, 1.0, 1.5, 2.0, be):
        drag = s * nights / years / 100
        adj = net_bps - s * (frame["weight"] * frame["nights"].fill_null(1)).to_numpy()
        cagr = (np.prod(1 + adj / 1e4) ** (1 / years) - 1) * 100
        tag = "  <- break-even" if abs(s - be) < 1e-9 else ""
        rows.append(f"    {s:>6.2f}            {drag:>8.2f}%                "
                    f"{cagr:>7.2f}%{tag}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swap", type=float, default=0.0,
                    help="overnight financing, bps of notional per night")
    ap.add_argument("--stress", action="store_true",
                    help="3x spread and fully adverse slippage")
    ap.add_argument("--test", action="store_true",
                    help="also run the held-out test split (once)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    splits = ["dev", "validation"] + (["test"] if args.test else [])
    payload: dict = {"swap_bps_per_night": args.swap, "stress": args.stress,
                     "candidates": {k: asdict(v) for k, v in CANDIDATES.items()},
                     "splits": {}}

    for split in splits:
        sheets, kept = evaluate(split, CANDIDATES, swap=args.swap,
                                allow_test=(split == "test"), stress=args.stress)
        print(f"\n{'=' * 96}\n{split.upper()}"
              f"{'   [HELD OUT - one run]' if split == 'test' else ''}\n{'=' * 96}")
        print(table(sheets))
        if kept is not None:
            print("\n" + format_returns_tearsheet(
                {**tearsheet_from_returns(kept['net_bps'],
                                          label='C gate + vol-target (detail)',
                                          turnover=kept['turnover'])}))
            print("\n  swap sensitivity")
            print(swap_breakeven(kept))
            by_year = (kept.with_columns(y=pl.col("nyd").dt.year())
                       .group_by("y").agg(
                           sessions=pl.len(),
                           strat_pct=((1 + pl.col("net_bps") / 1e4).product() - 1) * 100,
                           bh_pct=((1 + pl.col("bh_bps") / 1e4).product() - 1) * 100,
                           avg_w=pl.col("weight").mean())
                       .sort("y"))
            print("\n  by calendar year")
            for row in by_year.iter_rows(named=True):
                print(f"    {row['y']}  n={row['sessions']:>4}  "
                      f"strategy {row['strat_pct']:+7.2f}%   "
                      f"buy&hold {row['bh_pct']:+7.2f}%   avg w {row['avg_w']:.2f}")
        payload["splits"][split] = sheets

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
