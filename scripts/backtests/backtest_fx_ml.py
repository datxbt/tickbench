"""Replicate Enkhbayar and Slepaczuk (2024) on this corpus.

Their sample is 2000-2023 on six pairs from an ICMarkets feed. This corpus is
2020-2026 on four Exness instruments, three of which (EURUSD, USDJPY, XAUUSD)
are FX and one (USTEC) is an index CFD carried along because the method claims
nothing FX-specific. The overlap with their sample is 2020-2023, so the bulk of
what runs here is out of sample *for the paper*.

Split discipline: the driver stops at the end of ``validation`` unless
``--allow-test`` is passed. The walk-forward would otherwise roll straight
through the locked test period on its own, which is exactly the accident
:mod:`qlab.loader`'s lock exists to prevent.

Four axes are crossed: instrument x frequency x model x signal mode, and each
run is scored under both threshold rules and both feature modes. Results land in
``reports/strategies/fx_ml_runs.parquet``; the per-bar net returns of every run
go to ``fx_ml_returns.parquet`` so the report can bootstrap them without refitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qlab import paths  # noqa: E402
from qlab.loader import SPLITS  # noqa: E402
from qlab.strategies import fx_ml as fm  # noqa: E402

OUT = paths.STRATEGY_REPORT_DIR
SYMBOLS = ("EURUSD", "USDJPY", "XAUUSD", "USTEC")
FREQS = ("1d", "4h")
COST = 0.0002  # paper Section 3.8: fixed 0.02 percent


def load_bars(symbol: str, freq: str, *, until: date) -> pd.DataFrame:
    path = paths.whole_bar_path(symbol, freq)
    frame = pl.read_parquet(path).filter(pl.col("ts").dt.date() <= until)
    out = frame.select("ts", "open", "high", "low", "close").to_pandas()
    return out.set_index("ts").sort_index()


def buy_hold(bars: pd.DataFrame, index: pd.DatetimeIndex, ppy: float) -> tuple[float, float]:
    """Benchmark: hold the instrument across exactly the evaluated bars."""
    r = bars["close"].pct_change().shift(-1).reindex(index).to_numpy()
    r = np.nan_to_num(r)
    return fm.annualised_sharpe(r, ppy), float(np.prod(1 + r) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    ap.add_argument("--freqs", nargs="*", default=list(FREQS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--feature-modes", nargs="*", default=["paper", "stationary"])
    ap.add_argument("--allow-test", action="store_true",
                    help="extend the walk-forward into the locked test period")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    until = SPLITS["test"].end if args.allow_test else SPLITS["validation"].end
    if args.allow_test:
        print("!! walk-forward extended into the LOCKED test split", file=sys.stderr)
    registry = fm.model_registry()
    models = args.models or list(registry)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, tapes = [], []
    t0 = time.time()

    for symbol in args.symbols:
        for freq in args.freqs:
            bars = load_bars(symbol, freq, until=until)
            wf = fm.WalkForward.for_frequency(freq)
            ppy = fm.PERIODS_PER_YEAR[freq]
            print(f"\n=== {symbol} {freq}: {len(bars):,} bars "
                  f"{bars.index[0].date()} -> {bars.index[-1].date()}", flush=True)

            # Trend following: one run per signal mode, no feature choice to make.
            for sig_mode in fm.SIGNAL_MODES:
                idx, sig, ret = fm.ema_cross_walkforward(
                    bars, wf, signal_mode=sig_mode, cost=COST, periods_per_year=ppy)
                bh_sr, bh_ret = buy_hold(bars, idx, ppy)
                rows.append(dict(
                    symbol=symbol, freq=freq, family="TF", model="ema_cross",
                    feature_mode="-", threshold_mode="-", signal_mode=sig_mode,
                    n_bars=len(ret), n_trades=int((np.diff(sig, prepend=0.0) != 0).sum()),
                    asr=fm.annualised_sharpe(ret, ppy),
                    total_return=float(np.prod(1 + ret) - 1),
                    asr_gross=fm.annualised_sharpe(
                        fm.backtest(sig, np.nan_to_num(
                            bars["close"].pct_change().shift(-1).reindex(idx).to_numpy()),
                            cost=0.0), ppy),
                    bh_asr=bh_sr, bh_return=bh_ret,
                    start=str(idx[0].date()), end=str(idx[-1].date()),
                ))
                tapes.append(pd.DataFrame(dict(
                    ts=idx, ret=ret, signal=sig,
                    run=f"{symbol}|{freq}|TF|ema_cross|-|-|{sig_mode}")))
                print(f"  TF  {sig_mode:9s} ASR {rows[-1]['asr']:+.2f} "
                      f"(B&H {bh_sr:+.2f})", flush=True)

            # Machine learning.
            for feature_mode in args.feature_modes:
                for name in models:
                    spec = registry[name]
                    t = time.time()
                    try:
                        res = fm.run_walkforward(bars, spec, wf, feature_mode=feature_mode)
                    except Exception as exc:
                        print(f"  ML  {name} [{feature_mode}] FAILED: {exc}", flush=True)
                        continue
                    bh_sr, bh_ret = buy_hold(bars, res.index, ppy)
                    for thr in ("paper", "prediction"):
                        for sig_mode in fm.SIGNAL_MODES:
                            sig = fm.signal_from_result(
                                res, signal_mode=sig_mode, threshold_mode=thr)
                            ret = fm.backtest(sig, res.forward_return, cost=COST)
                            gross = fm.backtest(sig, res.forward_return, cost=0.0)
                            rows.append(dict(
                                symbol=symbol, freq=freq, family="ML", model=name,
                                feature_mode=feature_mode, threshold_mode=thr,
                                signal_mode=sig_mode, n_bars=len(ret),
                                n_trades=int((np.diff(sig, prepend=0.0) != 0).sum()),
                                asr=fm.annualised_sharpe(ret, ppy),
                                total_return=float(np.prod(1 + ret) - 1),
                                asr_gross=fm.annualised_sharpe(gross, ppy),
                                bh_asr=bh_sr, bh_return=bh_ret,
                                start=str(res.index[0].date()), end=str(res.index[-1].date()),
                                val_mae=float(np.mean(res.val_mae)),
                                pred_sd=float(np.std(res.prediction)),
                                target_sd=float(np.std(res.forward_return)),
                                n_windows=len(res.val_mae),
                            ))
                            tapes.append(pd.DataFrame(dict(
                                ts=res.index, ret=ret, signal=sig,
                                run=f"{symbol}|{freq}|ML|{name}|{feature_mode}|{thr}|{sig_mode}")))
                    best = max(r["asr"] for r in rows[-6:])
                    print(f"  ML  {name:8s} [{feature_mode:10s}] best ASR {best:+.2f} "
                          f"(B&H {bh_sr:+.2f})  {time.time()-t:.0f}s", flush=True)

    suffix = args.tag or ("test" if args.allow_test else "devval")
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / f"fx_ml_runs_{suffix}.parquet", index=False)
    pd.concat(tapes, ignore_index=True).to_parquet(
        OUT / f"fx_ml_returns_{suffix}.parquet", index=False)
    (OUT / f"fx_ml_meta_{suffix}.json").write_text(json.dumps({
        "until": str(until), "cost": COST, "symbols": args.symbols,
        "freqs": args.freqs, "models": models, "feature_modes": args.feature_modes,
        "runs": len(frame), "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\n{len(frame)} runs in {time.time()-t0:.0f}s -> fx_ml_runs_{suffix}.parquet")


if __name__ == "__main__":
    main()
