"""Rebalancing flow into the US close: does the day's return carry into 15:30-16:00?

Leveraged and inverse index funds must trade in the direction of the day's move
near the close - a 3x fund buys 6x its AUM times the day's return - and dealers
short gamma hedge the same way (Cheng & Madhavan 2009; Baltussen, Da, Lammers &
Martens 2021). The Nasdaq-100 carries the largest such complex (TQQQ/SQQQ/QLD).
The prediction is momentum in the last half hour, scaled by the day's return.

USTEC's rejection log tested first-30-minutes -> last-30 (corr -0.047). That is
a different predictor; the mechanism's is the whole day so far, from the prior
close. Related, and not the same: the Zarattini intraday-momentum study, whose
test split is spent, used a noise-band breakout over the whole session.

Pre-registered before the run (2026-09-11):

* Signal: sign of the mid return from the prior session's 16:00 ET close to
  15:30 ET. Trade in that direction, 15:30 -> 16:00 ET, one-minute bar-close
  quotes, real fills, commission and measured slippage.
* PRIMARY CELL: USTEC, all sessions, net bps pooled dev + validation.
  Predicted > 0; passes at one-sided t > 1.645.
* Dose-response: the 15:30 -> 16:00 mid return regressed on the prior-close ->
  15:30 return should load positively.
* Secondary: |day return| > 1%; entry at 15:45 instead of 15:30.
* Falsification: the same rule on XAUUSD, EURUSD, USDJPY, which have no
  leveraged-fund complex rebalancing at the US equity close.

    python scripts/research/close_rebalance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from macro_event_screen import (  # noqa: E402
    OUT, SPLIT_NAMES, at, business_days, load, prev_business_day, quotes_at, report, trades,
)

from qlab.costs import CostModel  # noqa: E402
from qlab.stats import ols_hac  # noqa: E402
from qlab.symbols import ALL_SYMBOLS  # noqa: E402


def session_rows(bars: pl.DataFrame, entry_min: int) -> pl.DataFrame:
    days = [d for d in business_days() if prev_business_day(d) >= business_days()[0]]
    prior = quotes_at(bars, [at(prev_business_day(d), 16, 0) for d in days])["close"]
    sig = quotes_at(bars, [at(d, 15, 30) for d in days])["close"]
    entry = quotes_at(bars, [at(d, 15, entry_min) for d in days])["close"]
    close = quotes_at(bars, [at(d, 16, 0) for d in days])["close"]
    return pl.DataFrame({"day": days, "prior": prior, "sig": sig, "entry": entry, "close": close}).drop_nulls().with_columns(
        day_ret=(pl.col("sig") / pl.col("prior") - 1) * 1e4,
        last_ret=(pl.col("close") / pl.col("entry") - 1) * 1e4,
    ).filter(pl.col("day_ret") != 0)


def run(sym: str, bars: pl.DataFrame, models, rows: pl.DataFrame, entry_min: int, label: str) -> dict:
    days = rows["day"].to_list()
    dirs = [1 if r > 0 else -1 for r in rows["day_ret"].to_list()]
    ev = trades(sym, bars, models, [at(d, 15, entry_min) for d in days], [at(d, 16, 0) for d in days], dirs)
    return report(f"{sym} {label}", ev, None)


def main() -> None:
    out = []
    for sym in ALL_SYMBOLS:
        bars = load(sym)
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        print(f"\n== {sym}")
        rows = session_rows(bars, 30)
        out.append(run(sym, bars, models, rows, 30, "all days 15:30->16:00"))
        big = rows.filter(pl.col("day_ret").abs() > 100)
        out.append(run(sym, bars, models, big, 30, f"|day|>1% ({big.height}) 15:30"))
        rows45 = session_rows(bars, 45)
        out.append(run(sym, bars, models, rows45, 45, "all days 15:45->16:00"))
        res = ols_hac(rows["last_ret"].to_numpy(), rows["day_ret"].to_numpy(), names=["day_ret"])
        b, t, _ = res.coef("day_ret")
        print(f"  dose-response: last30 = a + b * day_ret   b {b:+.4f}  t(NW) {t:+.2f}  n {res.n}")
        out.append({"cell": f"{sym} dose-response", "beta": b, "t": t, "n": res.n})
    (OUT / "close_rebalance.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
