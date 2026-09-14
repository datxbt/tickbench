"""Discovery batch 4: H12, a cost-aware nonlinear forecast on four instruments.

Registered in docs/findings/discovery-program.md before this script was run.

    python scripts/research/discovery_batch4.py dev        # fit, dev-holdout, gate
    python scripts/research/discovery_batch4.py validate   # once, only if the gate passed

Everything that is chosen - the diurnal factor, winsorisation bounds, theta -
comes from dev. Validation supplies only features and outcomes. Test is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from discovery_batch1 import SYMBOLS, cell, first_crossings, move_z  # noqa: E402
from discovery_batch2 import diurnal_from_dev, prepare  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import daily_pnl, forward_returns  # noqa: E402

OUT = Path("reports/discovery/batch4")
HORIZON = 30
HOLDOUT_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
MULTS = (1.0, 1.5, 2.0)
GATE_T = 2.05
OWN_Z = (1, 5, 15, 60, 240)
CROSS_Z = (5, 15, 60)
HGB = dict(max_iter=200, learning_rate=0.03, max_leaf_nodes=31,
           min_samples_leaf=5000, l2_regularization=1.0, early_stopping=False,
           random_state=0)


# --- features -------------------------------------------------------------


def own_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol features at each 1m bar close, plus the forward target."""
    for w in sorted(set(OWN_Z) | set(CROSS_Z)):
        bars = move_z(bars, w)
    day = pl.col("ts_open").dt.date()
    b = bars.with_columns(
        _date=day,
        _rn=pl.col("r1") / pl.col("d"),
        _sbps=pl.col("spread_mean") / pl.col("close") * 1e4,
        _ticks=pl.col("n_ticks").cast(pl.Float64),
    ).with_columns(
        vol_ratio=pl.col("_rn").rolling_std(60, min_samples=30) / pl.col("sig"),
        sig_rel=pl.col("sig") / pl.col("sig").rolling_mean(28_800, min_samples=7_200),
        spread_rel=pl.col("_sbps").rolling_mean(5) / pl.col("_sbps").shift(5).rolling_mean(1440, min_samples=720),
        tick_rel=pl.col("_ticks").rolling_mean(15) / pl.col("_ticks").shift(15).rolling_mean(1440, min_samples=720),
        _dopen=pl.col("lc").first().over("_date"),
        _dn=pl.int_range(1, pl.len() + 1).over("_date"),
        _dhi=pl.col("high").cum_max().over("_date"),
        _dlo=pl.col("low").cum_min().over("_date"),
    ).with_columns(
        day_ret_z=(pl.col("lc") - pl.col("_dopen")) / (pl.col("sig") * pl.col("_dn").cast(pl.Float64).sqrt()),
        day_pos=pl.when(pl.col("_dhi") > pl.col("_dlo")).then(
            (pl.col("close") - pl.col("_dlo")) / (pl.col("_dhi") - pl.col("_dlo"))),
    )
    prev = (
        b.group_by("_date").agg(_phi=pl.col("high").max(), _plo=pl.col("low").min())
        .sort("_date").with_columns(pl.col("_phi", "_plo").shift(1))
    )
    b = b.join(prev, on="_date", how="left").with_columns(
        dist_phi=(pl.col("lc") - pl.col("_phi").log()) / (pl.col("sig") * math.sqrt(1440)),
        dist_plo=(pl.col("lc") - pl.col("_plo").log()) / (pl.col("sig") * math.sqrt(1440)),
        weekday=pl.col("ts_open").dt.weekday().cast(pl.Float64),
        bucket_f=pl.col("bucket").cast(pl.Float64),
        target=pl.when(
            (pl.col("ts").shift(-HORIZON) - pl.col("ts")).dt.total_minutes() == HORIZON
        ).then((pl.col("lc").shift(-HORIZON) - pl.col("lc")) * 1e4),
    )
    return b.sort("ts")


OWN_COLS = [f"z{w}" for w in OWN_Z] + [
    "vol_ratio", "sig_rel", "spread_rel", "tick_rel", "day_ret_z", "day_pos",
    "dist_phi", "dist_plo", "d",
]
TIME_COLS = ["bucket_f", "weekday"]


def feature_frames(split: str, diurnals: dict) -> dict[str, pl.DataFrame]:
    """Bars with own features for every symbol, then cross features joined in."""
    own = {}
    for s in SYMBOLS:
        own[s] = own_features(prepare(s, split, diurnals[s]))
        print(f"  features {split} {s}: {own[s].height:,} bars", flush=True)
    cross = {
        s: own[s].select("ts", *[pl.col(f"z{w}").alias(f"{s}_z{w}") for w in CROSS_Z])
        for s in SYMBOLS
    }
    out = {}
    for s in SYMBOLS:
        f = own[s]
        for o in SYMBOLS:
            if o != s:
                f = f.join(cross[o], on="ts", how="left")
        out[s] = f
    return out


def cols_for(symbol: str) -> list[str]:
    return OWN_COLS + TIME_COLS + [f"{o}_z{w}" for o in SYMBOLS if o != symbol for w in CROSS_Z]


def matrix(f: pl.DataFrame, cols: list[str]) -> np.ndarray:
    # Raw-spread FX often quotes zero spread, so ratio features can be +-inf.
    x = f.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e3, 1e3).astype(np.float32)


# --- model ----------------------------------------------------------------


def fit(f: pl.DataFrame, cols: list[str]) -> tuple[HistGradientBoostingRegressor, Ridge, dict]:
    rows = f.filter((pl.col("ts").dt.minute() % 3 == 0) & pl.col("target").is_not_null()
                    & pl.col("sig").is_not_null())
    X = matrix(rows, cols)
    y = rows["target"].to_numpy()
    lo, hi = np.quantile(y, [0.005, 0.995])
    y = np.clip(y, lo, hi)
    hgb = HistGradientBoostingRegressor(**HGB).fit(X, y)
    Xr = np.nan_to_num(X.astype(np.float64))
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0) + 1e-9
    ridge = Ridge(alpha=10.0).fit(np.clip((Xr - mu) / sd, -10, 10), y)
    return hgb, ridge, {"lo": float(lo), "hi": float(hi), "n": int(len(y)), "mu": mu, "sd": sd}


def predict(models, f: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    hgb, ridge, meta = models
    X = matrix(f, cols)
    Xr = np.clip((np.nan_to_num(X.astype(np.float64)) - meta["mu"]) / meta["sd"], -10, 10)
    return f.with_columns(
        pred=pl.Series(hgb.predict(X)),
        pred_lin=pl.Series(ridge.predict(Xr)),
    ).with_columns(
        pred=pl.when(pl.col("sig").is_not_null()).then(pl.col("pred")),
        pred_lin=pl.when(pl.col("sig").is_not_null()).then(pl.col("pred_lin")),
    )


def trades(f: pl.DataFrame, theta: float, cost: CostModel, column: str = "pred",
           randomise: bool = False) -> pl.DataFrame:
    p = f[column].to_numpy()
    flag = np.isfinite(p) & (np.abs(p) >= theta)
    idx = first_crossings(f["ts"], flag, HORIZON)
    if idx.size == 0:
        return pl.DataFrame()
    direction = np.sign(p[idx]).astype(np.int8)
    if randomise:
        rng = np.random.default_rng(12)
        direction = rng.choice(np.array([-1, 1], dtype=np.int8), size=idx.size)
    ev = pl.DataFrame({"ts": f["ts"].gather(idx), "direction": direction, "pred": p[idx]})
    return forward_returns(f, ev, (HORIZON,), cost=cost, keep=("pred",))


def dev_round_turn_bps(symbol: str, cost: CostModel, f: pl.DataFrame) -> float:
    return cost.round_turn_bps(float(f["close"].median()))


def decile_table(f: pl.DataFrame, column: str = "pred") -> pl.DataFrame:
    g = f.filter(pl.col(column).is_not_null() & pl.col("target").is_not_null()
                 & (pl.col("ts").dt.minute() % 30 == 0))
    return (
        g.with_columns(dec=pl.col(column).qcut(10, labels=[str(i) for i in range(10)]))
        .group_by("dec").agg(n=pl.len(), pred=pl.col(column).mean(), realised=pl.col("target").mean(),
                             se=pl.col("target").std() / pl.len().sqrt())
        .sort("dec")
    )


# --- driver ---------------------------------------------------------------


def dev() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diurnals = {s: diurnal_from_dev(s) for s in SYMBOLS}
    frames = feature_frames("dev", diurnals)
    cost = {s: CostModel.from_profiles(s, split="dev") for s in SYMBOLS}
    rt = {s: dev_round_turn_bps(s, cost[s], frames[s]) for s in SYMBOLS}
    print("dev round turn bps:", {s: round(v, 3) for s, v in rt.items()}, flush=True)

    # Stage 1: fit on dev-train, read dev-holdout.
    hold, rows, deciles = {}, [], []
    for s in SYMBOLS:
        cols = cols_for(s)
        train = frames[s].filter(pl.col("ts") < HOLDOUT_START)
        models = fit(train, cols)
        h = predict(models, frames[s].filter(pl.col("ts") >= HOLDOUT_START), cols)
        hold[s] = h
        print(f"  {s}: fitted on {models[2]['n']:,} rows; holdout pred sd "
              f"{h['pred'].std():.3f} bps, max |pred| {h['pred'].abs().max():.3f}", flush=True)
        for mdl in ("pred", "pred_lin"):
            d = decile_table(h, mdl)
            deciles.append(d.with_columns(symbol=pl.lit(s), model=pl.lit(mdl)))
            tt = h.filter(pl.col(mdl).is_not_null() & pl.col("target").is_not_null()
                          & (pl.col("ts").dt.minute() % 30 == 0))
            ic = float(np.corrcoef(tt[mdl].to_numpy(), tt["target"].to_numpy())[0, 1])
            rows.append({"stage": "holdout", "symbol": s, "model": mdl, "ic_30m_grid": ic})

    choice = {}
    for m in MULTS:
        pooled = []
        for s in SYMBOLS:
            tr = trades(hold[s], m * rt[s], cost[s])
            if tr.height:
                pooled.append(tr.with_columns(symbol=pl.lit(s)))
                rows.append({"stage": "holdout", "symbol": s, "model": "pred", "mult": m, **cell(tr)})
            lin = trades(hold[s], m * rt[s], cost[s], column="pred_lin")
            if lin.height:
                rows.append({"stage": "holdout", "symbol": s, "model": "pred_lin", "mult": m, **cell(lin)})
            rnd = trades(hold[s], m * rt[s], cost[s], randomise=True)
            if rnd.height:
                rows.append({"stage": "holdout", "symbol": s, "model": "random_dir", "mult": m, **cell(rnd)})
        if pooled:
            pdf = pl.concat(pooled, how="diagonal_relaxed")
            c = cell(pdf)
            rows.append({"stage": "holdout", "symbol": "POOLED", "model": "pred", "mult": m, **c})
            choice[m] = c
        else:
            choice[m] = {"n": 0, "t_day_net": float("nan")}

    finite = {m: c for m, c in choice.items() if c.get("n", 0) and math.isfinite(c.get("t_day_net", float("nan")))}
    best = max(finite, key=lambda m: finite[m]["t_day_net"]) if finite else None
    gate = bool(best is not None and finite[best]["t_day_net"] >= GATE_T and finite[best]["net"] > 0)
    print(f"\nholdout choice: mult {best}, pooled {finite.get(best)}; GATE {'PASS' if gate else 'FAIL'}",
          flush=True)

    # Stage 2: refit on all of dev (only needed if the gate passes, but cheap).
    final = {}
    if gate:
        for s in SYMBOLS:
            final[s] = fit(frames[s], cols_for(s))
        with open(OUT / "models.pkl", "wb") as fh:
            pickle.dump(final, fh)

    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "dev_holdout.csv")
    pl.concat(deciles).write_csv(OUT / "dev_holdout_deciles.csv")
    (OUT / "dev_state.json").write_text(json.dumps(
        {"mult": best, "gate": gate, "round_turn_dev_bps": rt, "choice": choice}, indent=2, default=str))

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(pl.concat(deciles).filter(pl.col("model") == "pred"))
        print(pl.concat(deciles).filter(pl.col("model") == "pred_lin"))
        t = pl.DataFrame(rows, infer_schema_length=None)
        keep = [c for c in ("stage", "symbol", "model", "mult", "ic_30m_grid", "n", "days", "mid", "net",
                            "cost", "t_day_mid", "t_day_net", "hit_mid") if c in t.columns]
        print(t.select(keep))


def validate() -> None:
    state = json.loads((OUT / "dev_state.json").read_text())
    if not state["gate"]:
        print("H12 did not pass its dev-holdout gate - validation not read")
        return
    raise SystemExit("gate passed: write the single validation shot after stating power in the log")


def main() -> None:
    global HORIZON, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dev", "validate"])
    parser.add_argument("--horizon", type=int, default=30, help="30 = H12, 120 = H13")
    args = parser.parse_args()
    HORIZON = args.horizon
    if HORIZON != 30:
        OUT = OUT.parent / f"batch4_h{HORIZON}"
    dev() if args.mode == "dev" else validate()


if __name__ == "__main__":
    main()
