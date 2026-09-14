"""Discovery batch 2: USDJPY continuation (H05) and pooled transmission (H06).

Registered in docs/findings/discovery-program.md before this script was run.
Both hypotheses were selected by reading batch 1's dev tables, so dev only
describes them; each is tested by one fixed cell on validation.

    python scripts/research/discovery_batch2.py grid           # H05 dev grid + gate
    python scripts/research/discovery_batch2.py validate       # H05 + H06, validation, once

The diurnal factor and the transmission betas are always estimated on dev and
applied to validation, so validation contributes nothing but its outcomes.
The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from discovery_batch1 import (  # noqa: E402
    SIGMA_WINDOW, SYMBOLS, cell, first_crossings, move_z, pair_beta, placebo_band,
)

from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import forward_returns  # noqa: E402
from qlab.loader import load_bars  # noqa: E402

OUT = Path("reports/discovery/batch2")
WS, CS, HS = (60, 120, 240), (1.0, 2.0, 3.0), (60, 120, 240)
CENTRE = (120, 2.0, 120)


def diurnal_from_dev(symbol: str) -> pl.DataFrame:
    bars = _base(load_bars(symbol, "1m", split="dev"))
    overall = bars.select((pl.col("r1") ** 2).mean()).item()
    return bars.group_by("bucket").agg(d=((pl.col("r1") ** 2).mean() / overall).sqrt())


def _base(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.sort("ts").with_columns(
        lc=pl.col("close").log(),
        _gap=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes(),
        bucket=pl.col("ts_open").dt.hour().cast(pl.Int32) * 4
        + pl.col("ts_open").dt.minute().cast(pl.Int32) // 15,
    ).with_columns(
        r1=pl.when(pl.col("_gap") == 1).then(pl.col("lc") - pl.col("lc").shift(1)),
    ).drop("_gap")


def prepare(symbol: str, split: str, diurnal: pl.DataFrame) -> pl.DataFrame:
    return (
        _base(load_bars(symbol, "1m", split=split))
        .join(diurnal, on="bucket", how="left")
        .sort("ts")
        .with_columns(
            sig=(pl.col("r1") / pl.col("d")).rolling_std(
                window_size=SIGMA_WINDOW, min_samples=SIGMA_WINDOW // 2
            ),
            d2=pl.col("d") ** 2,
        )
    )


def continuation_events(bars: pl.DataFrame, window: int, c: float, hold: int) -> pl.DataFrame:
    """Hourly decisions, |z_W| >= c, trade the move's direction, non-overlapping holds."""
    bars = move_z(bars, window)
    z = bars[f"z{window}"].to_numpy()
    on_hour = (bars["ts"].dt.minute() == 0).to_numpy()
    flag = on_hour & np.isfinite(z) & (np.abs(z) >= c)
    idx = first_crossings(bars["ts"], flag, hold)
    return pl.DataFrame(
        {"ts": bars["ts"].gather(idx), "direction": np.sign(z[idx]).astype(np.int8)}
    )


def grid() -> None:
    rows = []
    for symbol in SYMBOLS:
        bars = prepare(symbol, "dev", diurnal_from_dev(symbol))
        cost = CostModel.from_profiles(symbol, split="dev")
        for w in WS:
            for c in CS:
                for h in HS:
                    ev = continuation_events(bars, w, c, h)
                    fr = forward_returns(bars, ev, (h,), cost=cost)
                    rows.append({"symbol": symbol, "W": w, "c": c, "h": h, **cell(fr)})
        print(symbol, flush=True)
    table = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "h05_dev_grid.csv")

    jpy = table.filter(pl.col("symbol") == "USDJPY")
    centre = jpy.filter((pl.col("W") == CENTRE[0]) & (pl.col("c") == CENTRE[1]) & (pl.col("h") == CENTRE[2])).row(0, named=True)
    positive = int((jpy["mid"] > 0).sum())
    gate = centre["mid"] > 0 and positive >= 18
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        for symbol in SYMBOLS:
            sub = table.filter(pl.col("symbol") == symbol)
            print(f"\n{symbol}: mid bps (t_day_mid) by W,c x h;  cells mid>0: {int((sub['mid'] > 0).sum())}/27")
            print(sub.select("W", "c", "h", "n", "mid", "net", "t_day_mid", "t_day_net"))
    print(f"\nH05 gate: centre mid {centre['mid']:+.2f} bps (t {centre['t_day_mid']:+.2f}), "
          f"USDJPY cells mid>0 {positive}/27 -> {'PASS' if gate else 'FAIL'}")
    (OUT / "h05_gate.json").write_text(json.dumps({"centre": centre, "positive": positive, "pass": gate}, indent=2, default=str))


def validate(seeds: int) -> None:
    gate = json.loads((OUT / "h05_gate.json").read_text())
    diurnals = {s: diurnal_from_dev(s) for s in SYMBOLS}
    val = {s: prepare(s, "validation", diurnals[s]) for s in SYMBOLS}
    costs = {s: CostModel.from_profiles(s, split="validation") for s in SYMBOLS}
    result: dict = {}

    if gate["pass"]:
        for symbol in SYMBOLS:
            ev = continuation_events(val[symbol], *CENTRE)
            fr = forward_returns(val[symbol], ev, (CENTRE[2],), cost=costs[symbol])
            row = cell(fr)
            if symbol == "USDJPY":
                row.update(placebo_band(ev, val[symbol], costs[symbol], CENTRE[2], seeds))
                row["by_year"] = {
                    str(y): cell(g)["mid"] for (y,), g in fr.group_by(pl.col("ts").dt.year())
                }
            result[f"H05 {symbol}"] = row
    else:
        result["H05"] = "gate failed on dev - validation not read for H05"

    # H06: betas from dev, events and outcomes from validation, pooled over pairs.
    dev = {s: move_z(prepare(s, "dev", diurnals[s]), 5) for s in SYMBOLS}
    frames = {s: move_z(b, 5) for s, b in val.items()}
    pooled = []
    for source in SYMBOLS:
        src = frames[source]
        z = src["z5"].to_numpy()
        idx = first_crossings(src["ts"], np.isfinite(z) & (np.abs(z) >= 4.0), 30)
        shocks = pl.DataFrame({"ts": src["ts"].gather(idx), "rs": src["r5"].gather(idx)})
        for target in SYMBOLS:
            if target == source:
                continue
            beta = pair_beta(dev[source], dev[target])
            ev = shocks.with_columns(
                direction=pl.Series(np.sign((beta * shocks["rs"]).to_numpy())).cast(pl.Int8)
            ).filter(pl.col("direction") != 0)
            fr = forward_returns(val[target], ev, (15,), cost=costs[target])
            result[f"H06 {source}->{target}"] = cell(fr)
            pooled.append(fr)
    result["H06 pooled"] = cell(pl.concat(pooled))

    (OUT / "validation.json").write_text(json.dumps(result, indent=2, default=str))
    for key, row in result.items():
        if isinstance(row, str):
            print(key, row)
            continue
        extra = f"  placebo [{row['pl_mid_lo']:+.2f}, {row['pl_mid_hi']:+.2f}]" if "pl_mid_lo" in row else ""
        print(f"{key:<24} n {row['n']:>5}  mid {row['mid']:+6.2f}  net {row['net']:+6.2f}  "
              f"t_day_mid {row['t_day_mid']:+5.2f}  t_day_net {row['t_day_net']:+5.2f}{extra}")
        if "by_year" in row:
            print("   by year (mid):", {k: round(v, 2) for k, v in sorted(row["by_year"].items())})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "validate"])
    parser.add_argument("--placebos", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "grid":
        grid()
    else:
        validate(args.placebos)


if __name__ == "__main__":
    main()
