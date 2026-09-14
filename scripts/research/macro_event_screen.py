"""Scheduled-event screen: four pre-registered hypotheses, four instruments.

Wave 1 of the untested-mechanism search. Every hypothesis below was written
down, with its window, its predicted sign and its control, before any number
was read. Each is a clock rule on a published calendar, so the control is the
same clock rule on days with no event - the mechanics-only null that separates
"the event does it" from "the clock does it".

H1  Pre-FOMC drift (Lucca & Moench 2015). Long 14:00 ET the day before a
    scheduled statement to 13:55 ET statement day. Predicted: USTEC up.
    Unscheduled meetings are excluded - nobody could have positioned for them.
H2  FOMC press-conference reversal. Fade the 14:00 -> 14:25 ET statement move,
    hold 14:25 -> 15:55 ET. Predicted: reversal (positive when faded).
H3  Post-release drift, CPI and NFP. Follow the 08:30 -> 08:35 ET move, hold
    60 / 120 / 240 minutes. Predicted: continuation.
H4  Month-end hedge rebalancing (Melvin & Prins 2015). Last business day,
    12:00 -> 16:00 London; sell USD when USTEC is up month-to-date.
    Exploratory: ~65 months.

Entry and exit are the 1-minute bar-close quotes standing at the stated clock
instant (right-edge labels, so the quote is known at that instant). Longs pay
the ask and leave at the bid. Commission and two sides of measured slippage are
charged from the split's own cost model. Results: mid, fills, net - pooled over
dev and validation, and per split. The test split is never read.

    python scripts/research/macro_event_screen.py
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from qlab.costs import CostModel
from qlab.loader import SPLITS, load_bars
from qlab.macro_calendar import FOMC_DATES, _HOLIDAYS, build_calendar
from qlab.symbols import ALL_SYMBOLS, get_spec

ET = ZoneInfo("America/New_York")
LDN = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")
SPLIT_NAMES = ("dev", "validation")
UNSCHEDULED = {date(2020, 3, 3), date(2020, 3, 15)}
OUT = Path("reports/macro_event_screen")


# --- prices at a clock instant ------------------------------------------------


def load(symbol: str) -> pl.DataFrame:
    frames = [
        load_bars(symbol, "1m", split=s, columns=["ts", "close", "bid_close", "ask_close"])
        for s in SPLIT_NAMES
    ]
    return pl.concat(frames).unique("ts").sort("ts")


def quotes_at(bars: pl.DataFrame, instants: list[datetime], tol_min: int = 3) -> pl.DataFrame:
    """The last bar-close quote at or before each instant, within a tolerance.

    A stale quote more than ``tol_min`` old is treated as missing: the market
    was shut, and a price from before the halt is not a price you could fill at.
    """
    q = pl.DataFrame({"at": instants}).with_columns(
        pl.col("at").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
    ).with_row_index("i")
    joined = q.sort("at").join_asof(
        bars, left_on="at", right_on="ts", strategy="backward",
        tolerance=timedelta(minutes=tol_min),
    )
    return joined.sort("i")


def trades(
    symbol: str,
    bars: pl.DataFrame,
    models: dict[str, CostModel],
    entries: list[datetime],
    exits: list[datetime],
    directions: list[int],
    tags: dict[str, list] | None = None,
) -> pl.DataFrame:
    """Price a batch of (entry, exit, direction) at real fills, in bps."""
    if not entries:
        return pl.DataFrame()
    spec = get_spec(symbol)
    a = quotes_at(bars, entries)
    b = quotes_at(bars, exits)
    df = pl.DataFrame(
        {
            "entry": entries,
            "direction": directions,
            "mid0": a["close"], "bid0": a["bid_close"], "ask0": a["ask_close"],
            "mid1": b["close"], "bid1": b["bid_close"], "ask1": b["ask_close"],
            **(tags or {}),
        }
    ).drop_nulls(["mid0", "mid1"])
    if df.is_empty():
        return df
    d = pl.col("direction").cast(pl.Float64)
    df = df.with_columns(
        mid_bps=d * (pl.col("mid1") / pl.col("mid0") - 1) * 1e4,
        gross_bps=pl.when(pl.col("direction") == 1)
        .then(pl.col("bid1") / pl.col("ask0") - 1)
        .otherwise(pl.col("bid0") / pl.col("ask1") - 1)
        * 1e4,
        split=pl.col("entry").map_elements(_split_of, return_dtype=pl.String),
        hour=pl.col("entry").dt.hour().cast(pl.Int32),
    )

    comm, slip = [], []
    for row in df.select("split", "mid0", "hour").iter_rows():
        model = models[row[0]]
        price = row[1]
        to_bps = spec.pip / price * 1e4
        comm.append(model.commission_pips(price) * to_bps)
        slip.append(2.0 * model.slippage_pips(row[2]) * to_bps)
    return df.with_columns(
        comm_bps=pl.Series(comm), slip_bps=pl.Series(slip)
    ).with_columns(
        fill_bps=pl.col("gross_bps") - pl.col("comm_bps"),
    ).with_columns(
        net_bps=pl.col("fill_bps") - pl.col("slip_bps"),
        cost_bps=pl.col("comm_bps") + pl.col("slip_bps"),
    )


def _split_of(ts: datetime) -> str:
    d = ts.date()
    for name in SPLIT_NAMES:
        if d in SPLITS[name]:
            return name
    return "other"


# --- clock helpers --------------------------------------------------------------


def at(day: date, hh: int, mm: int, tz: ZoneInfo = ET) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz).astimezone(UTC).replace(tzinfo=None)


def business_days() -> list[date]:
    out = []
    for name in SPLIT_NAMES:
        s = SPLITS[name]
        d = s.start
        while d <= s.end:
            if d.weekday() < 5 and d not in _HOLIDAYS:
                out.append(d)
            d += timedelta(days=1)
    return out


def prev_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5 or d in _HOLIDAYS:
        d -= timedelta(days=1)
    return d


def sign_move(bars: pl.DataFrame, starts: list[datetime], ends: list[datetime]) -> list[int | None]:
    a = quotes_at(bars, starts)["close"].to_list()
    b = quotes_at(bars, ends)["close"].to_list()
    out: list[int | None] = []
    for x, y in zip(a, b):
        if x is None or y is None or y == x:
            out.append(None)
        else:
            out.append(1 if y > x else -1)
    return out


# --- statistics -------------------------------------------------------------------


def stat(x: np.ndarray) -> dict:
    n = x.size
    if n < 3:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    sd = x.std(ddof=1)
    return {"n": int(n), "mean": float(x.mean()), "sd": float(sd),
            "t": float(x.mean() / sd * math.sqrt(n)) if sd > 0 else float("nan"),
            "hit": float((x > 0).mean())}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    return float((a.mean() - b.mean()) / math.sqrt(va + vb))


def report(label: str, ev: pl.DataFrame, ctl: pl.DataFrame | None) -> dict:
    row: dict = {"cell": label}
    for part in ("pooled", *SPLIT_NAMES):
        sub = ev if part == "pooled" else ev.filter(pl.col("split") == part)
        row[part] = {c: stat(sub[c].to_numpy()) for c in ("mid_bps", "net_bps")}
    if ctl is not None and not ctl.is_empty():
        row["control"] = stat(ctl["mid_bps"].to_numpy())
        row["event_minus_control_t"] = welch_t(ev["mid_bps"].to_numpy(), ctl["mid_bps"].to_numpy())
    p, d, v = row["pooled"]["net_bps"], row["dev"]["net_bps"], row["validation"]["net_bps"]
    c = row.get("control", {})
    print(
        f"  {label:<34} n={p['n']:>4}  net {p['mean']:+7.2f} bps t={p['t']:+5.2f}"
        f" | dev {d['mean']:+7.2f} (t {d['t']:+5.2f}) val {v['mean']:+7.2f} (t {v['t']:+5.2f})"
        f" | mid {row['pooled']['mid_bps']['mean']:+7.2f}"
        + (f" | ctl mid {c['mean']:+6.2f} diff t={row['event_minus_control_t']:+5.2f}" if c else "")
    )
    return row


# --- hypotheses ------------------------------------------------------------------------


def h1_pre_fomc(sym, bars, models) -> list[dict]:
    meetings = [d for d in FOMC_DATES if d not in UNSCHEDULED and _split_of(datetime(d.year, d.month, d.day)) != "other"]
    ev = trades(sym, bars, models,
                [at(prev_business_day(d), 14, 0) for d in meetings],
                [at(d, 13, 55) for d in meetings], [1] * len(meetings))
    fomc = set(FOMC_DATES)
    ctl_days = [d for d in business_days() if d not in fomc and prev_business_day(d) == d - timedelta(days=1)]
    ctl = trades(sym, bars, models,
                 [at(prev_business_day(d), 14, 0) for d in ctl_days],
                 [at(d, 13, 55) for d in ctl_days], [1] * len(ctl_days))
    return [report(f"{sym} H1 pre-FOMC long", ev, ctl)]


def _fade_rule(sym, bars, models, days, label) -> pl.DataFrame:
    signs = sign_move(bars, [at(d, 14, 0) for d in days], [at(d, 14, 25) for d in days])
    keep = [(d, -s) for d, s in zip(days, signs) if s is not None]
    return trades(sym, bars, models, [at(d, 14, 25) for d, _ in keep],
                  [at(d, 15, 55) for d, _ in keep], [s for _, s in keep])


def h2_fomc_fade(sym, bars, models) -> list[dict]:
    meetings = [d for d in FOMC_DATES if d not in UNSCHEDULED and d.weekday() < 5]
    ev = _fade_rule(sym, bars, models, meetings, "event")
    fomc = set(FOMC_DATES)
    ctl = _fade_rule(sym, bars, models, [d for d in business_days() if d not in fomc], "ctl")
    return [report(f"{sym} H2 FOMC 14:25 fade", ev, ctl)]


def h3_release_drift(sym, bars, models) -> list[dict]:
    cal = build_calendar(date(2020, 1, 1), date(2025, 6, 30))
    release_830 = set(cal.filter((pl.col("et_hour") == 8) & (pl.col("et_minute") == 30))["day"].to_list())
    out = []
    ctl_days = [d for d in business_days() if d not in release_830]
    for name, event in (("CPI", "CPI"), ("NFP", "Nonfarm Payrolls")):
        days = [d for d in cal.filter(pl.col("event") == event)["day"].to_list()
                if _split_of(datetime(d.year, d.month, d.day)) != "other"]
        for hold in (60, 120, 240):
            def run(ds):
                signs = sign_move(bars, [at(d, 8, 30) for d in ds], [at(d, 8, 35) for d in ds])
                keep = [(d, s) for d, s in zip(ds, signs) if s is not None]
                start = [at(d, 8, 35) for d, _ in keep]
                return trades(sym, bars, models, start,
                              [t + timedelta(minutes=hold) for t in start], [s for _, s in keep])
            out.append(report(f"{sym} H3 {name} follow {hold}m", run(days), run(ctl_days)))
    return out


def h4_month_end(sym, bars, models, ustec: pl.DataFrame) -> list[dict]:
    if sym == "USTEC":
        return []
    # USD direction of the instrument: +1 if a long is long USD.
    usd_long = {"EURUSD": -1, "USDJPY": 1, "XAUUSD": -1}[sym]
    days = business_days()
    by_month: dict[tuple[int, int], list[date]] = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(d)
    entries, exits, dirs = [], [], []
    for (_, _), ds in sorted(by_month.items()):
        if len(ds) < 15:
            continue
        last = ds[-1]
        first = ds[0]
        q = quotes_at(ustec, [at(first, 9, 35), at(last, 11, 0, LDN) if False else at(last, 12, 0, LDN)])["close"].to_list()
        if None in q:
            continue
        equity_up = q[1] > q[0]
        # Equity up -> hedgers sell USD into the fix.
        sell_usd = equity_up
        dirs.append(-usd_long if sell_usd else usd_long)
        entries.append(at(last, 12, 0, LDN))
        exits.append(at(last, 16, 0, LDN))
    ev = trades(sym, bars, models, entries, exits, dirs)
    return [report(f"{sym} H4 month-end fix", ev, None)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ustec = load("USTEC")
    for sym in ALL_SYMBOLS:
        bars = ustec if sym == "USTEC" else load(sym)
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        print(f"\n== {sym}  ({bars.height:,} bars)")
        rows += h1_pre_fomc(sym, bars, models)
        rows += h2_fomc_fade(sym, bars, models)
        rows += h3_release_drift(sym, bars, models)
        rows += h4_month_end(sym, bars, models, ustec)
    (OUT / "screen.json").write_text(json.dumps(rows, indent=1, default=str))
    n_cells = len(rows)
    from qlab.stats import normal_sf  # noqa: F401  (threshold printed below)
    thr = _bonf_t(n_cells)
    print(f"\n{n_cells} cells inspected; a single cell needs |t| >= {thr:.2f} (Bonferroni, 5% two-sided).")
    for r in rows:
        t = r["pooled"]["net_bps"]["t"]
        if abs(t) >= 2:
            print(f"  |t|>=2 : {r['cell']}  pooled net t={t:+.2f}")


def _bonf_t(n: int) -> float:
    from qlab.eventstudy import _norm_ppf
    return _norm_ppf(1 - 0.025 / n)


if __name__ == "__main__":
    main()
