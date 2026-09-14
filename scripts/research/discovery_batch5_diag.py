"""Batch 5 diagnostic (exploratory, read after the H14 verdict): is there any
conditional structure at the mid, whatever the cost?

H14 was screened on net_now and failed at stage A. This asks the prior question
the verdict cannot: do cells that are strong in half A (odd months) tend to be
strong in half B (even months) at the mid? Across ~40k cells that is a direct
test for replicating conditional structure, read against the same statistic on
date-shifted placebo outcomes. Everything here is descriptive; any hypothesis it
suggests is a child and is registered before it is tested.

    python scripts/research/discovery_batch5_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import discovery_batch5 as d  # noqa: E402

OUT = d.OUT


def mid_cells(bases, feats, seed):
    rows = []
    for bi, b in enumerate(bases):
        if b.idx.size < d.MIN_N:
            continue
        f = feats[b.symbol]
        oi = d.outcome_index(b, f, seed)
        M = np.full((b.idx.size, len(d.HORIZONS)), np.nan)
        ok = oi >= 0
        M[ok] = f.fwd[oi[ok]]
        M *= b.direction[:, None]
        rt = (f.rt_chase if b.cont else f.rt_fade)[b.idx]
        tw, vr, half, day = f.tw[b.idx], f.vr[b.idx], f.half[b.idx], f.tdrank[b.idx]
        usd = f.usd_z[b.idx] * b.direction
        tr = f.tr_z[b.idx] * b.direction
        for cname, cm in d.condition_masks(tw, vr, usd, tr):
            ma, mb = cm & (half == 0), cm & (half == 1)
            if ma.sum() < d.MIN_N:
                continue
            nA, dA, mA, tA = d.per_day_stats(day[ma], M[ma])
            nB, dB, mB, tB = d.per_day_stats(day[mb], M[mb])
            rtc = float(rt[cm].mean())
            for h, hz in enumerate(d.HORIZONS):
                if nA[h] < d.MIN_N or dA[h] < d.MIN_DAYS:
                    continue
                rows.append((bi, b.symbol, b.family, b.param, cname, hz, int(nA[h]), int(nB[h]),
                             float(mA[h]), float(tA[h]), float(mB[h]), float(tB[h]), rtc))
    return pl.DataFrame(rows, orient="row", schema=[
        "base", "symbol", "family", "param", "cond", "h", "nA", "nB",
        "midA", "tA", "midB", "tB", "rt"])


def replication(t: pl.DataFrame) -> dict:
    t = t.filter(pl.col("tA").is_finite() & pl.col("tB").is_finite())
    a, b = t["tA"].to_numpy(), t["tB"].to_numpy()
    strong = np.abs(a) >= 2
    return {
        "cells": t.height,
        "corr_tA_tB": float(np.corrcoef(a, b)[0, 1]),
        "strong_A": int(strong.sum()),
        "sign_agree_strong": float((np.sign(a[strong]) == np.sign(b[strong])).mean()) if strong.any() else np.nan,
        "both_abs_t_ge_2_same_sign": int(((np.abs(a) >= 2) & (np.abs(b) >= 2) & (np.sign(a) == np.sign(b))).sum()),
    }


def main() -> None:
    feats, bases, _, _ = d.build("dev", None)
    real = mid_cells(bases, feats, None)
    real.write_parquet(OUT / "diag_mid_cells.parquet")
    placebos = [mid_cells(bases, feats, s) for s in (1, 2)]

    print("\n== replication of cost-free cell t across halves (real vs placebo) ==")
    for name, t in [("real", real), ("placebo1", placebos[0]), ("placebo2", placebos[1])]:
        print(name, replication(t))

    print("\n== by family x horizon: corr(tA, tB), real | placebo1 ==")
    fams = []
    for (fam, h), g in sorted(real.group_by("family", "h"), key=lambda kv: kv[0]):
        p = placebos[0].filter((pl.col("family") == fam) & (pl.col("h") == h))
        r = replication(g)
        rp = replication(p) if p.height > 10 else {"corr_tA_tB": np.nan, "sign_agree_strong": np.nan}
        fams.append({"family": fam, "h": h, "cells": r["cells"], "corr": r["corr_tA_tB"],
                     "corr_pl": rp["corr_tA_tB"], "agree": r["sign_agree_strong"],
                     "agree_pl": rp["sign_agree_strong"]})
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(pl.DataFrame(fams))

    print("\n== by symbol x family (all horizons): corr(tA, tB), real | placebo1 ==")
    rows = []
    for (sym, fam), g in sorted(real.group_by("symbol", "family"), key=lambda kv: kv[0]):
        p = placebos[0].filter((pl.col("symbol") == sym) & (pl.col("family") == fam))
        rows.append({"symbol": sym, "family": fam, "cells": g.height,
                     "corr": replication(g)["corr_tA_tB"],
                     "corr_pl": replication(p)["corr_tA_tB"] if p.height > 10 else np.nan})
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(pl.DataFrame(rows))

    # Cells that replicate at the mid, and how large they are against the cost.
    rep = real.filter((pl.col("tA").abs() >= 2.5) & (pl.col("tB").abs() >= 2.0)
                      & (pl.col("tA").sign() == pl.col("tB").sign())).with_columns(
        mid_pooled=(pl.col("midA") * pl.col("nA") + pl.col("midB") * pl.col("nB")) / (pl.col("nA") + pl.col("nB")),
    ).with_columns(edge_over_cost=pl.col("mid_pooled").abs() / pl.col("rt"))
    rep.write_parquet(OUT / "diag_replicating.parquet")
    print(f"\n== cells with |tA|>=2.5 and |tB|>=2 same sign: {rep.height} "
          f"(placebo1 {placebos[0].filter((pl.col('tA').abs() >= 2.5) & (pl.col('tB').abs() >= 2.0) & (pl.col('tA').sign() == pl.col('tB').sign())).height}, "
          f"placebo2 {placebos[1].filter((pl.col('tA').abs() >= 2.5) & (pl.col('tB').abs() >= 2.0) & (pl.col('tA').sign() == pl.col('tB').sign())).height}) ==")
    with pl.Config(tbl_rows=50, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
        print(rep.sort("edge_over_cost", descending=True).head(40).select(
            "symbol", "family", "param", "cond", "h", "nA", "nB", "midA", "tA", "midB", "tB", "rt",
            "edge_over_cost"))
        print(rep.group_by("symbol", "family").agg(
            n=pl.len(), pos=(pl.col("tA") > 0).sum(), best_ratio=pl.col("edge_over_cost").max(),
            med_ratio=pl.col("edge_over_cost").median()).sort("n", descending=True))


if __name__ == "__main__":
    main()
