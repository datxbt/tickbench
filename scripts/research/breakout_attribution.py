"""Does the breakout signal predict anything, or does it just ride the trend?

The backtest's own P&L cannot answer this. Its exit rule is direction-dependent
- a winner runs to 3R, a loser is cut at 1R - so any statistic measured from
entry to *its own exit* mixes the signal's information with the exit's shape.

So this measures a **fixed-horizon** forward return instead: from the moment of
entry, over a horizon chosen in advance, on the mid, with no stop and no target.
Normalised by the bracket width so it is in the same R units as everything else.

Three numbers settle it:

* ``uncond``  - mean forward return of simply being long over the same windows.
  This is the drift a strategy inherits for free in a trending market.
* ``long``/``short`` - mean forward return conditioned on which way the signal
  fired. A signal with information wants ``long`` above ``uncond`` and
  ``short`` below it.
* ``signal``  - ``direction * forward``, the edge as the strategy would take it,
  and ``edge_vs_drift`` - the same after the unconditional drift has been
  subtracted from each side. That last column is the part that a permanently
  long book would not already have given you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.loader import load_bars  # noqa: E402

TAPES = Path("reports/strategies/session_breakout_trades")


def attribute(symbol: str, preset: str, horizon_h: float) -> pl.DataFrame:
    trades = pl.read_parquet(TAPES / f"{symbol}_{preset}.parquet")
    bars = (load_bars(symbol, "1m", allow_test=True, columns=["ts", "close"])
            .select(pl.col("ts").cast(pl.Int64).alias("t"), pl.col("close"))
            .sort("t"))

    h_us = int(horizon_h * 3600 * 1_000_000)
    # p0 must be the mid at the instant of entry, NOT the last bar close before
    # it. A bar close is up to a minute stale and, because the entry happened by
    # breaking out, it sits on the wrong side of the entry by part of the very
    # move being measured - which would score the breakout itself as forward
    # return and make any signal look predictive.
    q = trades.select(
        pl.col("day"), pl.col("direction"), pl.col("width"),
        pl.col("entry_mid").alias("p0"),
        (pl.col("entry_ts") + h_us).alias("t1"),
    )
    q = (q.sort("t1")
          .join_asof(bars, left_on="t1", right_on="t", strategy="backward")
          .rename({"close": "p1"}).drop("t")
          .drop_nulls(["p0", "p1"]))

    q = q.with_columns(
        fwd_r=(pl.col("p1") - pl.col("p0")) / pl.col("width"),
        year=pl.col("day").dt.year(),
    ).with_columns(signal_r=pl.col("direction") * pl.col("fwd_r"))

    out = []
    for label, f in (("2020-2023", pl.col("year") < 2024),
                     ("2024-2026", pl.col("year") >= 2024),
                     ("2024", pl.col("year") == 2024),
                     ("2025", pl.col("year") == 2025),
                     ("2026", pl.col("year") == 2026)):
        s = q.filter(f)
        if s.is_empty():
            continue
        longs = s.filter(pl.col("direction") == 1)["fwd_r"].to_numpy()
        shorts = s.filter(pl.col("direction") == -1)["fwd_r"].to_numpy()
        uncond = float(s["fwd_r"].mean())
        sig = s["signal_r"].to_numpy()
        # Subtract the free drift from each side, then re-combine.
        edge = float(np.concatenate([longs - uncond, -(shorts - uncond)]).mean())
        se = float(sig.std(ddof=1) / np.sqrt(len(sig)))
        out.append({
            "period": label, "n": len(s),
            "uncond": round(uncond, 4),
            "long": round(float(longs.mean()), 4) if longs.size else None,
            "short": round(float(shorts.mean()), 4) if shorts.size else None,
            "signal": round(float(sig.mean()), 4),
            "signal_t": round(float(sig.mean()) / se, 2),
            "edge_vs_drift": round(edge, 4),
            "edge_t": round(edge / se, 2),
        })
    return pl.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--symbols", nargs="+",
                    default=["XAUUSD", "EURUSD", "USDJPY", "USTEC"])
    ap.add_argument("-p", "--preset", default="GEO_2026_NO_H13")
    ap.add_argument("-H", "--horizons", nargs="+", type=float, default=[1.0, 4.0])
    args = ap.parse_args()

    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(60)
    for h in args.horizons:
        for symbol in args.symbols:
            print(f"\n=== {symbol} / {args.preset} / forward {h:g}h, "
                  f"no stop, no target, in R ===")
            print(attribute(symbol, args.preset, h))


if __name__ == "__main__":
    main()
