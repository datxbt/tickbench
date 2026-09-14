"""Turn the two replications' raw runs into the tables the reports quote.

Nothing here refits a model or resimulates a trade; it reads the parquets the
two backtest drivers wrote and adds the statistics that decide whether any of it
means anything - Lo standard errors on every Sharpe, stationary-block bootstrap
intervals, and a Benjamini-Hochberg correction across the whole grid, because a
study that scores 200 configurations and reports the best one has not found an
edge, it has found a maximum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402

from qlab import paths, stats  # noqa: E402

OUT = paths.STRATEGY_REPORT_DIR
PPY = {"1d": 252.0, "4h": 252.0 * 6}


def _forward_returns(symbol: str, freq: str) -> pd.Series:
    """Bar-to-bar return of the instrument, indexed the way a run's tape is.

    Needed because the only test that matters for a long-biased signal is a
    *paired* one against holding the thing. A Sharpe measured against zero says
    gold went up, which nobody disputes.
    """
    path = paths.whole_bar_path(symbol, freq)
    frame = pl.read_parquet(path).select("ts", "close").to_pandas().set_index("ts")
    return frame["close"].pct_change().shift(-1)


def _fmt(x, spec=".2f"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def _table(frame: pd.DataFrame, cols: list[str], fmts: dict[str, str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, rule]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append(_fmt(v, fmts[c]) if c in fmts and isinstance(v, (int, float, np.floating))
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==========================================================================
# Paper 2 - FX machine learning
# ==========================================================================


def analyse_fx(tag: str = "devval") -> dict:
    runs = pd.read_parquet(OUT / f"fx_ml_runs_{tag}.parquet")
    tapes = pd.read_parquet(OUT / f"fx_ml_returns_{tag}.parquet")

    # Significance, run by run, from the actual per-bar returns.
    recs = []
    for run, grp in tapes.groupby("run"):
        symbol, freq, family, model, fmode, tmode, smode = run.split("|")
        r = grp["ret"].to_numpy()
        sig = grp["signal"].to_numpy()
        res = stats.sharpe_with_se(r, periods_per_year=int(PPY[freq]))
        recs.append(dict(
            run=run, symbol=symbol, freq=freq, family=family, model=model,
            feature_mode=fmode, threshold_mode=tmode, signal_mode=smode,
            asr=res.mean, se=res.se, t=res.t_stat, p=res.p_value,
            in_market=float((sig != 0).mean()),
            turnover=float(np.abs(np.diff(sig, prepend=0.0)).sum()),
        ))
    sig_frame = pd.DataFrame(recs)
    merged = runs.merge(
        sig_frame.drop(columns=["asr"]),
        on=["symbol", "freq", "family", "model", "feature_mode",
            "threshold_mode", "signal_mode"], how="left")

    valid = merged[np.isfinite(merged["p"])].copy()
    valid["p_bh"] = stats.benjamini_hochberg(valid["p"].to_numpy())
    valid["beats_bh"] = valid["asr"] - valid["bh_asr"]

    faithful = valid[(valid["feature_mode"].isin(["paper", "-"]))
                     & (valid["threshold_mode"].isin(["paper", "-"]))]

    out = {
        "n_runs": int(len(valid)),
        "n_faithful": int(len(faithful)),
        "beat_bh_count": int((valid["beats_bh"] > 0).sum()),
        "beat_bh_share": float((valid["beats_bh"] > 0).mean()),
        "faithful_beat_bh_share": float((faithful["beats_bh"] > 0).mean()),
        "median_asr": float(valid["asr"].median()),
        "median_bh_asr": float(valid["bh_asr"].median()),
        "n_p_below_05": int((valid["p"] < 0.05).sum()),
        "n_survive_bh": int((valid["p_bh"] < 0.05).sum()),
        "expected_false_positives": float(0.05 * len(valid)),
        "median_shrinkage": float((valid["pred_sd"] / valid["target_sd"]).median()),
    }

    # Does the paper's own gate ever open?
    paper_gate = valid[(valid["threshold_mode"] == "paper")
                       & (valid["signal_mode"] == "buy_sell")
                       & (valid["family"] == "ML")]
    out["paper_gate_never_trades_share"] = float((paper_gate["n_trades"] <= 1).mean())
    out["paper_gate_median_in_market"] = float(paper_gate["in_market"].median())

    # Best run, and what its interval says.
    best = valid.sort_values("asr", ascending=False).iloc[0]
    best_ret = tapes[tapes["run"] == best["run"]]["ret"].to_numpy()
    ci = stats.bootstrap_ci(
        best_ret, lambda a: stats.sharpe(a, int(PPY[best["freq"]])), n_boot=2000)
    out["best"] = dict(
        run=best["run"], asr=float(best["asr"]), t=float(best["t"]),
        p=float(best["p"]), p_bh=float(best["p_bh"]),
        bh_asr=float(best["bh_asr"]),
        ci_lo=float(ci.lo), ci_hi=float(ci.hi),
    )

    # The test that matters: does the run beat *holding the instrument*?
    # A Sharpe against zero rewards a long-only signal for the asset's drift.
    paired = []
    rng = np.random.default_rng(7)
    fwd_cache: dict[tuple[str, str], pd.Series] = {}
    for run, grp in tapes.groupby("run"):
        symbol, freq, family, model, fmode, tmode, smode = run.split("|")
        key = (symbol, freq)
        if key not in fwd_cache:
            fwd_cache[key] = _forward_returns(symbol, freq)
        bh = fwd_cache[key].reindex(grp["ts"]).to_numpy()
        strat = grp["ret"].to_numpy()
        keep = np.isfinite(bh) & np.isfinite(strat)
        if keep.sum() < 50:
            continue
        ppy = int(PPY[freq])
        res = stats.paired_bootstrap(
            strat[keep], bh[keep], lambda a: stats.sharpe(a, ppy), n_boot=500, rng=rng)
        paired.append(dict(run=run, symbol=symbol, freq=freq, family=family, model=model,
                           feature_mode=fmode, threshold_mode=tmode, signal_mode=smode,
                           delta=res.point, delta_lo=res.lo, delta_hi=res.hi,
                           p_paired=res.p_value))
    pair_frame = pd.DataFrame(paired)
    # A single NaN would sort to the end and poison every adjusted value, so the
    # correction is applied to the finite p-values and the rest stay NaN.
    ok = np.isfinite(pair_frame["p_paired"].to_numpy())
    pair_frame["p_paired_bh"] = np.nan
    pair_frame.loc[ok, "p_paired_bh"] = stats.benjamini_hochberg(
        pair_frame.loc[ok, "p_paired"].to_numpy())
    pair_frame.to_parquet(OUT / f"fx_ml_paired_{tag}.parquet", index=False)
    out["paired"] = dict(
        n=int(len(pair_frame)),
        n_delta_positive=int((pair_frame["delta"] > 0).sum()),
        share_delta_positive=float((pair_frame["delta"] > 0).mean()),
        median_delta=float(pair_frame["delta"].median()),
        n_significant_raw=int((pair_frame["p_paired"] < 0.05).sum()),
        n_significant_positive_raw=int(((pair_frame["p_paired"] < 0.05)
                                        & (pair_frame["delta"] > 0)).sum()),
        n_significant_bh=int((pair_frame["p_paired_bh"] < 0.05).sum()),
        n_significant_positive_bh=int(((pair_frame["p_paired_bh"] < 0.05)
                                       & (pair_frame["delta"] > 0)).sum()),
        best=pair_frame.sort_values("delta", ascending=False).iloc[0].to_dict(),
    )
    faith_pair = pair_frame.merge(
        faithful[["run"]], on="run", how="inner") if "run" in faithful else pair_frame
    out["paired"]["faithful_share_positive"] = float((faith_pair["delta"] > 0).mean())
    out["paired"]["faithful_n_significant_positive_bh"] = int(
        ((faith_pair["p_paired_bh"] < 0.05) & (faith_pair["delta"] > 0)).sum())

    # Cost drag: how much of the gross edge survives 2 bps.
    valid["cost_drag"] = valid["asr_gross"] - valid["asr"]
    out["median_cost_drag"] = float(valid["cost_drag"].median())
    out["median_cost_drag_4h"] = float(valid[valid["freq"] == "4h"]["cost_drag"].median())
    out["median_cost_drag_1d"] = float(valid[valid["freq"] == "1d"]["cost_drag"].median())

    valid.to_parquet(OUT / f"fx_ml_analysis_{tag}.parquet", index=False)

    # Headline table: best model per symbol x frequency, faithful settings only.
    rows = []
    for (sym, freq), grp in faithful.groupby(["symbol", "freq"]):
        ml = grp[grp["family"] == "ML"]
        tf = grp[grp["family"] == "TF"]
        best_ml = ml.sort_values("asr", ascending=False).iloc[0] if len(ml) else None
        best_tf = tf.sort_values("asr", ascending=False).iloc[0] if len(tf) else None
        rows.append(dict(
            symbol=sym, freq=freq,
            bh=float(grp["bh_asr"].iloc[0]),
            best_ml=float(best_ml["asr"]) if best_ml is not None else np.nan,
            best_ml_name=f"{best_ml['model']}/{best_ml['signal_mode']}" if best_ml is not None else "-",
            median_ml=float(ml["asr"].median()) if len(ml) else np.nan,
            best_tf=float(best_tf["asr"]) if best_tf is not None else np.nan,
            n_ml=len(ml),
        ))
    out["headline"] = pd.DataFrame(rows).sort_values(["symbol", "freq"]).to_dict("records")
    return out


# ==========================================================================
# Paper 1 - small-cap equities
# ==========================================================================


def analyse_smallcap() -> dict:
    runs = pd.read_parquet(OUT / "smallcap_runs.parquet")
    meta = json.loads((OUT / "smallcap_meta.json").read_text())
    fetch = json.loads((REPO / "data/external/smallcap/fetch_report.json").read_text())
    sweep = pd.read_parquet(OUT / "smallcap_sweep_is.parquet")

    out = {
        "meta": meta,
        "universe": dict(
            requested=fetch["tickers_requested"], downloaded=fetch["tickers_downloaded"],
            missing=fetch["n_failed"],
            missing_share=fetch["n_failed"] / fetch["tickers_requested"],
        ),
        "sweep": {
            fam: dict(
                n=int(len(g)), best=float(g["sharpe"].max()),
                median=float(g["sharpe"].median()), worst=float(g["sharpe"].min()),
                share_positive=float((g["sharpe"] > 0).mean()),
            )
            for fam, g in sweep.groupby("family")
        },
    }

    # Per-trade economics: is the gross edge bigger than the round turn?
    # Both legs are read off the *same* run, because the cost-free twin follows a
    # different equity path and therefore a different number of differently
    # sized trades - dividing one run's P&L by another's trade count is not a
    # per-trade number at all.
    econ = {}
    for split in runs["split"].unique():
        for fam in ("A", "D", "F"):
            def pick(sizing, costs):
                m = runs[(runs["split"] == split) & (runs["family"] == fam)
                         & (runs["sizing"] == sizing) & (runs["costs"] == costs)]
                return None if m.empty else m.iloc[0]

            net, free, paper_sized = pick("vol_target", "paper"),                 pick("vol_target", "none"), pick("paper", "paper")
            if net is None or free is None:
                continue
            trades = max(int(net["n_trades"]), 1)
            econ[f"{fam}|{split}"] = dict(
                sharpe_net=float(net["sharpe"]), sharpe_free=float(free["sharpe"]),
                trades=int(net["n_trades"]),
                gross_pnl_per_trade=float(net["gross_pnl"] / trades),
                cost_per_trade=float(net["total_cost"] / trades),
                cost_share_of_gross=float(net["total_cost"] / net["gross_pnl"])
                if net["gross_pnl"] > 0 else float("inf"),
                net_pnl_per_trade=float((net["gross_pnl"] - net["total_cost"]) / trades),
                annual_return_net=float(net["annual_return"]),
                annual_return_free=float(free["annual_return"]),
                max_drawdown=float(net["max_drawdown"]),
                win_rate=float(net["win_rate"]), profit_factor=float(net["profit_factor"]),
                exposure=float(net["avg_exposure"]),
                paper_sizing_sharpe=float(paper_sized["sharpe"]) if paper_sized is not None else None,
                paper_sizing_exposure=float(paper_sized["avg_exposure"]) if paper_sized is not None else None,
                paper_sizing_return=float(paper_sized["annual_return"]) if paper_sized is not None else None,
            )
    out["economics"] = econ

    # Significance on the out-of-sample equity curve, family by family.
    curves = pd.read_parquet(OUT / "smallcap_equity.parquet")
    sigs = {}
    for col in curves.columns:
        eq = curves[col].dropna()
        if len(eq) < 30:
            continue
        r = eq.pct_change().dropna().to_numpy()
        res = stats.sharpe_with_se(r, periods_per_year=252)
        ci = stats.bootstrap_ci(r, lambda a: stats.sharpe(a, 252), n_boot=2000)
        sigs[col] = dict(sharpe=res.mean, se=res.se, t=res.t_stat, p=res.p_value,
                         ci_lo=float(ci.lo), ci_hi=float(ci.hi), n=int(len(r)))
    out["significance"] = sigs

    # Paired against the benchmarks a retail trader actually has: the small-cap
    # index, and the equal-weighted universe the strategy picks from.
    bench = pd.read_parquet(REPO / "data/external/smallcap/benchmarks.parquet")
    bench["date"] = pd.to_datetime(bench["date"])
    iwm = bench[bench["ticker"] == "IWM"].set_index("date")["adj_close"].sort_index()
    rng = np.random.default_rng(11)
    paired = {}
    for col in curves.columns:
        fam, split = col.split("|")
        if fam in ("EW",):
            continue
        eq = curves[col].dropna()
        strat = eq.pct_change().dropna()
        for bname, series in (("IWM", iwm.pct_change()),
                              ("EW_universe", curves[f"EW|{split}"].dropna().pct_change())):
            other = series.reindex(strat.index)
            keep = np.isfinite(strat.to_numpy()) & np.isfinite(other.to_numpy())
            if keep.sum() < 60:
                continue
            res = stats.paired_bootstrap(
                strat.to_numpy()[keep], other.to_numpy()[keep],
                lambda a: stats.sharpe(a, 252), n_boot=2000, rng=rng)
            paired[f"{col} vs {bname}"] = dict(
                delta=res.point, lo=res.lo, hi=res.hi, p=res.p_value)
    out["paired"] = paired
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=["fx", "smallcap"])
    ap.add_argument("--tag", default="devval")
    args = ap.parse_args()

    result = {}
    if "fx" in args.which:
        result["fx"] = analyse_fx(args.tag)
    if "smallcap" in args.which:
        result["smallcap"] = analyse_smallcap()

    path = OUT / "paper_replication_analysis.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float)[:8000])
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
