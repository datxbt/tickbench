"""Discovery batch 6: H15, the minute-scale liquidity-provision portfolio.

H15 exactly as registered in docs/findings/discovery-program.md: select every
H14 grid cell whose cost-free edge replicated across both halves of dev and
clears the round turn of the side it trades, de-duplicate, trade them all as one
portfolio. ``dev`` reports the portfolio on dev and the expected validation t;
``validate`` counts validation event days from timestamps, and reads outcomes
only if the registered power rule is met.

    python scripts/research/discovery_batch6.py dev
    python scripts/research/discovery_batch6.py validate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import discovery_batch5 as d  # noqa: E402

from qlab.costs import CostModel, SlippageModel  # noqa: E402
from qlab.eventstudy import forward_returns  # noqa: E402
from qlab.loader import load_bars  # noqa: E402

B5 = d.OUT
OUT = Path("reports/discovery/batch6")
T_A, T_B, RATIO = 2.5, 2.0, 1.0


def select(cells: pl.DataFrame) -> pl.DataFrame:
    return cells.filter(
        pl.col("tA").is_finite() & pl.col("tB").is_finite()
        & (pl.col("tA").abs() >= T_A) & (pl.col("tB").abs() >= T_B)
        & (pl.col("tA").sign() == pl.col("tB").sign())
    ).with_columns(
        side=pl.col("tA").sign().cast(pl.Int64),
        mid_pooled=(pl.col("midA") * pl.col("nA") + pl.col("midB") * pl.col("nB")) / (pl.col("nA") + pl.col("nB")),
        score=pl.min_horizontal(pl.col("tA").abs(), pl.col("tB").abs()),
    )


def events(row: dict, base: d.Base, f: d.Feats):
    """A cell's events (conditions as in the diagnostic: relative to the base direction)
    and its per-event net_now in the traded direction."""
    usd = f.usd_z[base.idx] * base.direction
    tr = f.tr_z[base.idx] * base.direction
    m = dict(d.condition_masks(f.tw[base.idx], f.vr[base.idx], usd, tr))[row["cond"]]
    side = row["side"]
    cont = base.cont if side == 1 else not base.cont
    rt = (f.rt_chase if cont else f.rt_fade)[base.idx]
    mid = side * base.direction * f.fwd[base.idx, d.HORIZONS.index(row["h"])]
    return base.idx[m], (side * base.direction)[m].astype(np.int8), mid[m], rt[m], cont


def attach_costs(sel: pl.DataFrame, bases, feats) -> pl.DataFrame:
    rts = []
    for row in sel.iter_rows(named=True):
        b = bases[row["base"]]
        _, _, _, rt, _ = events(row, b, feats[b.symbol])
        rts.append(float(rt.mean()))
    return sel.with_columns(rt_side=pl.Series(rts)).with_columns(
        ratio=pl.col("mid_pooled").abs() / pl.col("rt_side"))


def dedupe(sel: pl.DataFrame, bases, feats) -> list[dict]:
    kept, kept_ts = [], {}
    for row in sel.sort("score", descending=True).iter_rows(named=True):
        b = bases[row["base"]]
        f = feats[b.symbol]
        idx, *_ = events(row, b, f)
        ts = np.sort(f.ts[idx])
        prior = kept_ts.get(b.symbol)
        if prior is not None and prior.size:
            pos = np.searchsorted(prior, ts)
            lo = np.abs(ts - prior[np.clip(pos - 1, 0, prior.size - 1)])
            hi = np.abs(prior[np.clip(pos, 0, prior.size - 1)] - ts)
            if (np.minimum(lo, hi) <= 5 * d.MINUTE_US).mean() >= 0.5:
                continue
        kept.append(row)
        kept_ts[b.symbol] = ts if prior is None else np.sort(np.concatenate([prior, ts]))
    return kept


def tape(kept, bases, feats) -> pl.DataFrame:
    frames = []
    for k, row in enumerate(kept):
        b = bases[row["base"]]
        f = feats[b.symbol]
        idx, _, mid, rt, _ = events(row, b, f)
        ok = np.isfinite(mid)
        frames.append(pl.DataFrame({"cell": k, "symbol": b.symbol, "tday": f.tdrank[idx][ok],
                                    "half": f.half[idx][ok], "mid": mid[ok], "net": (mid - rt)[ok]}))
    return pl.concat(frames)


def stats(t: pl.DataFrame, col: str = "net") -> dict:
    daily = t.group_by("tday").agg(pl.col(col).sum())[col].to_numpy()
    sd = daily.std(ddof=1)
    return {"trades": t.height, "days": int(daily.size), "per_trade": float(t[col].mean()),
            "bps_day": float(daily.mean()), "t_day": float(daily.mean() / sd * math.sqrt(daily.size)),
            "sharpe_ann": float(daily.mean() / sd * math.sqrt(252))}


def build_selected(split: str, state=None):
    feats, bases, diurnals, betas = d.build(split, state)
    lookup = {(b.symbol, b.family, b.param): i for i, b in enumerate(bases)}
    return feats, bases, lookup, diurnals, betas


def dev() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats, bases, lookup, diurnals, betas = build_selected("dev")
    report = {}
    for name, path in (("real", B5 / "diag_mid_cells.parquet"),):
        cells = pl.read_parquet(path)
        sel = select(cells)
        sel = sel.with_columns(base=pl.Series([lookup[(r["symbol"], r["family"], r["param"])]
                                               for r in sel.iter_rows(named=True)]))
        sel = attach_costs(sel, bases, feats).filter(pl.col("ratio") >= RATIO)
        kept = dedupe(sel, bases, feats)
        tp = tape(kept, bases, feats)
        report[name] = {"replicating_above_cost": sel.height, "kept": len(kept),
                        "A": stats(tp.filter(pl.col("half") == 0)),
                        "B": stats(tp.filter(pl.col("half") == 1)),
                        "all": stats(tp), "all_mid": stats(tp, "mid")}
        by_sym = tp.group_by("symbol").agg(n=pl.len(), mid=pl.col("mid").mean(), net=pl.col("net").mean())
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
            print(pl.DataFrame(kept).select("symbol", "family", "param", "cond", "h", "side",
                                            "nA", "nB", "midA", "tA", "midB", "tB", "rt_side", "ratio"))
            print(by_sym)
        pl.DataFrame(kept).write_parquet(OUT / "kept.parquet")

    # Placebo: the batch-5 diagnostic kept only placebo-1 cells passing the
    # replication rule inside its printout, so recompute them here.
    import discovery_batch5_diag as diag
    pc = diag.mid_cells(bases, feats, 1)
    psel = select(pc)
    psel = attach_costs(psel, bases, feats).filter(pl.col("ratio") >= RATIO)
    report["placebo1_replicating_above_cost"] = psel.height

    r = report["real"]
    weaker = min(("A", "B"), key=lambda h: r[h]["t_day"])
    report["weaker_half"] = weaker
    state = {"diurnal": {s: diurnals[s].to_dict(as_series=False) for s in d.SYMBOLS},
             "betas": betas, "kept": [{k: v for k, v in row.items()} for row in kept], "report": report}
    (OUT / "dev_state.json").write_text(json.dumps(state, indent=2, default=str))
    print(json.dumps(report, indent=2))


def validate() -> None:
    state = json.loads((OUT / "dev_state.json").read_text())
    feats, bases, lookup, _, _ = build_selected("validation", state)
    kept = state["kept"]
    days = set()
    for row in kept:
        b = bases[lookup[(row["symbol"], row["family"], row["param"])]]
        idx, *_ = events(row, b, feats[b.symbol])
        days.update(feats[b.symbol].tdrank[idx].tolist())
    r = state["report"]["real"][state["report"]["weaker_half"]]
    expected = r["t_day"] * math.sqrt(len(days) / r["days"])
    result = {"val_event_days": len(days), "weaker_half": state["report"]["weaker_half"],
              "weaker_t": r["t_day"], "weaker_days": r["days"], "expected_t": expected}
    print(json.dumps(result, indent=2))
    if expected < d.POWER_T:
        result["verdict"] = "shot not taken (underpowered)"
        (OUT / "validation.json").write_text(json.dumps(result, indent=2))
        return

    tapes = []
    for s in d.SYMBOLS:
        rows = [row for row in kept if row["symbol"] == s]
        if not rows:
            continue
        bars = load_bars(s, "1m", split="validation").sort("ts")
        costs = {af: CostModel.from_profiles(s, split="validation", slippage=SlippageModel(adverse_fraction=af))
                 for af in (0.5, 1.0)}
        for k, row in enumerate(rows):
            b = bases[lookup[(s, row["family"], row["param"])]]
            idx, direction, _, _, cont = events(row, b, feats[s])
            ev = pl.DataFrame({"ts": pl.from_epoch(pl.Series(feats[s].ts[idx]), time_unit="us")
                               .dt.replace_time_zone("UTC"), "direction": direction})
            fr = forward_returns(bars, ev, (row["h"],), cost=costs[1.0 if cont else 0.5])
            tapes.append(fr.select("ts", "mid_bps", "fill_bps", "net_bps").with_columns(
                symbol=pl.lit(s), cell=pl.lit(f"{s} {row['family']} {row['param']} {row['cond']} h{row['h']}")))
    tp = pl.concat(tapes)
    tp.write_parquet(OUT / "validation_tape.parquet")
    daily = tp.group_by(pl.col("ts").dt.date()).agg(pl.col("net_bps").sum())["net_bps"].to_numpy()
    sd = daily.std(ddof=1)
    t = float(daily.mean() / sd * math.sqrt(daily.size))
    result.update(trades=tp.height, days=int(daily.size), mid_per_trade=float(tp["mid_bps"].mean()),
                  net_per_trade=float(tp["net_bps"].mean()), bps_day=float(daily.mean()), t_day=t,
                  sharpe_ann=float(daily.mean() / sd * math.sqrt(252)),
                  verdict="PASS" if (t >= d.T_B and daily.mean() > 0) else "FAIL")
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
        print(tp.group_by("symbol").agg(n=pl.len(), mid=pl.col("mid_bps").mean(), net=pl.col("net_bps").mean()))
        print(tp.group_by("cell").agg(n=pl.len(), mid=pl.col("mid_bps").mean(),
                                      net=pl.col("net_bps").mean()).sort("cell"))
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["dev", "validate"])
    args = p.parse_args()
    dev() if args.mode == "dev" else validate()


if __name__ == "__main__":
    main()
