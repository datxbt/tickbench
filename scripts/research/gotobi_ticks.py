"""The gotobi unwind on ticks: how fast is the post-fix drop, and can you get in?

The bar-level robustness surface showed half of G2's edge sits in the minute
after 09:55 JST: entering at 09:56 instead of 09:55 costs ~1.06 bps at every
exit. The bar-level entry is the last quote before 09:55:00, so the question
the bars cannot answer is how fast that first-minute drop happens and whether
an order sent at the fix can get in ahead of it.

Two measurements, on raw USDJPY ticks (09:55 JST = 00:55 UTC; Japan keeps no
daylight saving, so the UTC instant never moves):

1. The average short-side mid path around 00:55:00 at second resolution, on
   gotobi days and on control days.
2. G2 repriced with tick-accurate fills: sell at the bid of the first tick at or
   after 00:55:00 + L, for explicit latencies L, buy back at the ask of the
   first tick at or after 02:00:00 UTC. Commission is charged; slippage is not,
   because the latency *is* the slippage here and it is stated explicitly.

    python scripts/research/gotobi_ticks.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from fx_fix_screen import gotobi_days, weekdays  # noqa: E402
from macro_event_screen import OUT, SPLIT_NAMES, _split_of  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.loader import load_ticks  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

SYM = "USDJPY"
OFFSETS = (-300, -60, -10, -1, 0, 1, 2, 5, 10, 30, 60, 120, 300, 600)
LATENCIES = (0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


def window_ticks() -> pl.DataFrame:
    frames = []
    for split in SPLIT_NAMES:
        lazy = load_ticks(SYM, split=split, lazy=True)
        # dt.hour() is Int8: widen before multiplying or 3600x silently wraps.
        tod = (pl.col("ts").dt.hour().cast(pl.Int32) * 3600
               + pl.col("ts").dt.minute().cast(pl.Int32) * 60
               + pl.col("ts").dt.second().cast(pl.Int32))
        frames.append(
            lazy.filter((tod >= 50 * 60) & (tod < 2 * 3600 + 5 * 60))
            .select("ts", "bid", "ask")
            .collect()
        )
    return pl.concat(frames).sort("ts").with_columns(
        day=pl.col("ts").dt.date(), mid=(pl.col("bid") + pl.col("ask")) / 2
    )


def first_at_or_after(day_ticks: pl.DataFrame, when: datetime) -> dict | None:
    idx = day_ticks["ts"].search_sorted(when, side="left")
    if idx >= day_ticks.height:
        return None
    row = day_ticks.row(idx, named=True)
    # A quote more than 30 s after the instant means the market was not there.
    if (row["ts"] - when).total_seconds() > 30:
        return None
    return row


def last_at_or_before(day_ticks: pl.DataFrame, when: datetime) -> dict | None:
    idx = day_ticks["ts"].search_sorted(when, side="right") - 1
    if idx < 0:
        return None
    return day_ticks.row(idx, named=True)


def main() -> None:
    ticks = window_ticks()
    print(f"{ticks.height:,} ticks in the 00:50-02:05 UTC window")
    days = weekdays()
    goto = gotobi_days(days)
    spec = get_spec(SYM)
    models = {s: CostModel.from_profiles(SYM, split=s) for s in SPLIT_NAMES}
    by_day = {d[0]: g for d, g in ticks.group_by("day", maintain_order=True)}

    paths = {"gotobi": [], "control": []}
    fills: dict[float, list[float]] = {L: [] for L in LATENCIES}
    for d in days:
        g = by_day.get(d)
        if g is None:
            continue
        t0 = datetime(d.year, d.month, d.day, 0, 55, tzinfo=timezone.utc)
        ref = last_at_or_before(g, t0)
        if ref is None or (t0 - ref["ts"]).total_seconds() > 30:
            continue
        row = []
        for off in OFFSETS:
            q = last_at_or_before(g, t0 + timedelta(seconds=off))
            row.append(np.nan if q is None else -(q["mid"] / ref["mid"] - 1) * 1e4)
        paths["gotobi" if d in goto else "control"].append(row)

        if d not in goto:
            continue
        exit_q = first_at_or_after(g, datetime(d.year, d.month, d.day, 2, 0, tzinfo=timezone.utc))
        if exit_q is None:
            continue
        model = models[_split_of(datetime(d.year, d.month, d.day))]
        for L in LATENCIES:
            e = first_at_or_after(g, t0 + timedelta(seconds=L))
            if e is None:
                fills[L].append(np.nan)
                continue
            gross = (e["bid"] / exit_q["ask"] - 1) * 1e4
            comm = model.commission_pips(e["mid"]) * spec.pip / e["mid"] * 1e4
            fills[L].append(gross - comm)

    print("\n== 1. mean short-side mid path from 09:55:00 JST, bps")
    print("   offset(s) " + "".join(f"{o:>7}" for o in OFFSETS))
    out = {"offsets": OFFSETS}
    for k, rows in paths.items():
        a = np.array(rows)
        mean = np.nanmean(a, axis=0)
        out[k] = mean.tolist()
        print(f"   {k:<9} " + "".join(f"{m:>+7.2f}" for m in mean) + f"   n={a.shape[0]}")
    diff = np.nanmean(np.array(paths["gotobi"]), 0) - np.nanmean(np.array(paths["control"]), 0)
    print(f"   {'diff':<9} " + "".join(f"{m:>+7.2f}" for m in diff))

    print("\n== 2. G2 on tick-accurate fills (sell bid at 09:55:00+L, buy ask at 11:00), net of commission")
    out["latency"] = []
    for L in LATENCIES:
        x = np.array(fills[L])
        x = x[np.isfinite(x)]
        t = x.mean() / x.std(ddof=1) * np.sqrt(x.size)
        print(f"   L = {L:>5.2f} s   n={x.size}  net {x.mean():+.2f} bps  t {t:+.2f}  hit {(x > 0).mean():.2f}")
        out["latency"].append({"L": L, "n": int(x.size), "mean": float(x.mean()), "t": float(t)})
    (OUT / "gotobi_ticks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
