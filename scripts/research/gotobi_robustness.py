"""Robustness battery for the gotobi unwind on USDJPY (wave-3 cell G2).

G2 was pre-registered in ``fx_fix_screen.py``: short USDJPY 09:55 -> 11:00 JST
on gotobi days. It came back +2.85 bps net, t = +4.38 pooled, positive in both
splits. Nothing here changes that specification - the cell that goes to the
test split, if any does, is G2 exactly as registered. This script only asks
whether G2 is fragile:

1. the entry x exit surface, so the result is not one lucky exit;
2. year by year;
3. by gotobi day type, and weekend-shifted versus not;
4. a weekday-matched permutation null - gotobi dates that fall on a weekend
   move to Friday, so the set is Friday-heavy and a Friday effect could fake it;
5. shifted-date placebos: the business day before and after each gotobi day;
6. cost stress: doubled spread, fully adverse slippage, 1000 ms latency;
7. with Japanese public holidays removed, when a holiday calendar is available.

    python scripts/research/gotobi_robustness.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from fx_fix_screen import gotobi_days, weekdays  # noqa: E402
from macro_event_screen import OUT, SPLIT_NAMES, at, load, report, trades  # noqa: E402

from qlab.costs import CostModel, SlippageModel  # noqa: E402

JST = ZoneInfo("Asia/Tokyo")
SYM = "USDJPY"

# Japanese public holidays and Tokyo bank holidays (Jan 2-3, Dec 31) that fall
# on weekdays, 2020 to mid-2025. Hand-transcribed from the Cabinet Office
# schedule including substitute holidays; used only by section 7.
JP_HOLIDAYS: set[date] | None = {
    date(y, m, d) for y, m, d in [
        (2020, 1, 1), (2020, 1, 2), (2020, 1, 3), (2020, 1, 13), (2020, 2, 11),
        (2020, 2, 24), (2020, 3, 20), (2020, 4, 29), (2020, 5, 4), (2020, 5, 5),
        (2020, 5, 6), (2020, 7, 23), (2020, 7, 24), (2020, 8, 10), (2020, 9, 21),
        (2020, 9, 22), (2020, 11, 3), (2020, 11, 23), (2020, 12, 31),
        (2021, 1, 1), (2021, 1, 11), (2021, 2, 11), (2021, 2, 23), (2021, 4, 29),
        (2021, 5, 3), (2021, 5, 4), (2021, 5, 5), (2021, 7, 22), (2021, 7, 23),
        (2021, 8, 9), (2021, 9, 20), (2021, 9, 23), (2021, 11, 3), (2021, 11, 23),
        (2021, 12, 31),
        (2022, 1, 3), (2022, 1, 10), (2022, 2, 11), (2022, 2, 23), (2022, 3, 21),
        (2022, 4, 29), (2022, 5, 3), (2022, 5, 4), (2022, 5, 5), (2022, 7, 18),
        (2022, 8, 11), (2022, 9, 19), (2022, 9, 23), (2022, 10, 10), (2022, 11, 3),
        (2022, 11, 23),
        (2023, 1, 2), (2023, 1, 3), (2023, 1, 9), (2023, 2, 23), (2023, 3, 21),
        (2023, 5, 3), (2023, 5, 4), (2023, 5, 5), (2023, 7, 17), (2023, 8, 11),
        (2023, 9, 18), (2023, 10, 9), (2023, 11, 3), (2023, 11, 23),
        (2024, 1, 1), (2024, 1, 2), (2024, 1, 3), (2024, 1, 8), (2024, 2, 12),
        (2024, 2, 23), (2024, 3, 20), (2024, 4, 29), (2024, 5, 3), (2024, 5, 6),
        (2024, 7, 15), (2024, 8, 12), (2024, 9, 16), (2024, 9, 23), (2024, 10, 14),
        (2024, 11, 4), (2024, 12, 31),
        (2025, 1, 1), (2025, 1, 2), (2025, 1, 3), (2025, 1, 13), (2025, 2, 11),
        (2025, 2, 24), (2025, 3, 20), (2025, 4, 29), (2025, 5, 5), (2025, 5, 6),
    ]
}


def short_window(bars, models, days, t0=(9, 55), t1=(11, 0)) -> pl.DataFrame:
    return trades(SYM, bars, models, [at(d, *t0, JST) for d in days],
                  [at(d, *t1, JST) for d in days], [-1] * len(days),
                  tags={"day": days})


def line(label: str, df: pl.DataFrame) -> dict:
    x = df["net_bps"].to_numpy()
    n = x.size
    t = x.mean() / x.std(ddof=1) * np.sqrt(n) if n > 2 else float("nan")
    print(f"  {label:<36} n={n:>4}  net {x.mean():+6.2f}  t {t:+5.2f}  hit {(x > 0).mean():.2f}")
    return {"label": label, "n": int(n), "mean": float(x.mean()), "t": float(t)}


def main() -> None:
    bars = load(SYM)
    models = {s: CostModel.from_profiles(SYM, split=s) for s in SPLIT_NAMES}
    days = weekdays()
    goto = sorted(gotobi_days(days))
    goto_set = set(goto)
    out: dict = {}

    print("== 1. entry x exit surface (JST), gotobi days, pooled net bps (t)")
    entries = [(9, 55), (9, 56), (9, 58), (10, 0), (10, 5)]
    exits = [(10, 15), (10, 30), (11, 0), (11, 30), (12, 0), (13, 0), (15, 0)]
    print("   entry \\ exit " + "".join(f"{h:02d}:{m:02d}".rjust(15) for h, m in exits))
    surface = []
    for e in entries:
        cells = []
        for x in exits:
            df = short_window(bars, models, goto, e, x)
            v = df["net_bps"].to_numpy()
            t = v.mean() / v.std(ddof=1) * np.sqrt(v.size)
            cells.append(f"{v.mean():+6.2f} ({t:+5.2f})")
            surface.append({"entry": e, "exit": x, "mean": float(v.mean()), "t": float(t)})
        print(f"   {e[0]:02d}:{e[1]:02d}         " + "".join(c.rjust(15) for c in cells))
    out["surface"] = surface

    g2 = short_window(bars, models, goto)
    allw = short_window(bars, models, days)

    print("\n== 2. year by year (G2)")
    g2 = g2.with_columns(year=pl.col("entry").dt.year())
    out["years"] = [line(str(y), g2.filter(pl.col("year") == y)) for y in sorted(g2["year"].unique())]

    print("\n== 3. day type")
    def dom_type(d: date) -> str:
        nxt = d + timedelta(days=1)
        # Which nominal gotobi date does this business day serve?
        for k in range(0, 3):
            c = d + timedelta(days=k)
            if k > 0 and c.weekday() < 5:
                break
            import calendar as _c
            last = _c.monthrange(c.year, c.month)[1]
            if c.day in (5, 10, 15, 20, 25) or c.day == last:
                tag = "month-end" if c.day == last else f"{c.day:02d}th"
                return tag + (" (shifted)" if k > 0 else "")
        return "?"
    g2 = g2.with_columns(dtype=pl.col("day").map_elements(dom_type, return_dtype=pl.String))
    out["types"] = [line(t, g2.filter(pl.col("dtype") == t)) for t in sorted(g2["dtype"].unique())]
    shifted = g2.filter(pl.col("dtype").str.contains("shifted"))
    line("ALL shifted-to-Friday", shifted)
    line("ALL on the nominal date", g2.filter(~pl.col("dtype").str.contains("shifted")))

    print("\n== 4. weekday-matched permutation null (20,000 draws)")
    allw = allw.with_columns(wd=pl.col("day").map_elements(lambda d: d.weekday(), return_dtype=pl.Int8))
    pool = allw.filter(~pl.col("day").is_in(goto))
    real = g2["net_bps"].mean()
    counts = (allw.filter(pl.col("day").is_in(goto)).group_by("wd").len())
    rng = np.random.default_rng(7)
    by_wd = {wd: pool.filter(pl.col("wd") == wd)["net_bps"].to_numpy() for wd in range(5)}
    draws = np.zeros(20_000)
    total = counts["len"].sum()
    for i in range(draws.size):
        s = 0.0
        for wd, k in counts.iter_rows():
            # With replacement: weekend-shifted gotobi dates make Fridays more
            # numerous among gotobi days than among the non-gotobi pool.
            s += rng.choice(by_wd[wd], size=k, replace=True).sum()
        draws[i] = s / total
    p = (draws >= real).mean()
    print(f"  real {real:+.3f} bps; null mean {draws.mean():+.3f}, sd {draws.std():.3f}; one-sided p = {p:.5f}")
    for wd in range(5):
        v = by_wd[wd]
        print(f"    non-gotobi weekday {wd}: n={v.size} net {v.mean():+.2f}")
    out["permutation"] = {"real": float(real), "null_mean": float(draws.mean()), "p": float(p)}

    print("\n== 5. shifted-date placebos")
    order = {d: i for i, d in enumerate(days)}
    for off in (-2, -1, 1, 2):
        pdays = sorted({days[order[d] + off] for d in goto if 0 <= order[d] + off < len(days)} - goto_set)
        line(f"business day {off:+d} (non-gotobi only)", short_window(bars, models, pdays))

    print("\n== 6. cost stress on G2")
    spread_extra = (
        (g2["ask0"] - g2["bid0"]) / 2 + (g2["ask1"] - g2["bid1"]) / 2
    ) / g2["mid0"] * 1e4
    x = g2["net_bps"].to_numpy()
    print(f"  base                     net {x.mean():+.2f}  cost {g2['cost_bps'].mean():.2f} bps, spread paid {spread_extra.mean() * 2:.2f}")
    print(f"  2x spread                net {(x - spread_extra.to_numpy()).mean():+.2f}")
    adv = {s: CostModel.from_profiles(SYM, split=s, slippage=SlippageModel(adverse_fraction=1.0)) for s in SPLIT_NAMES}
    lat = {s: CostModel.from_profiles(SYM, split=s, slippage=SlippageModel(adverse_fraction=1.0, latency_ms=1000)) for s in SPLIT_NAMES}
    line("adverse slippage 1.0", short_window(bars, adv, goto))
    v = short_window(bars, lat, goto)
    line("adverse 1.0 + 1000 ms", v)
    w = v["net_bps"].to_numpy() - spread_extra.to_numpy()[: v.height]
    print(f"  all three at once        net {w.mean():+.2f}  t {w.mean() / w.std(ddof=1) * np.sqrt(w.size):+.2f}")

    print("\n== 7. Japanese public holidays")
    if JP_HOLIDAYS is None:
        print("  no holiday calendar installed - skipped")
    else:
        hol = [d for d in goto if d in JP_HOLIDAYS]
        line("gotobi on a JP holiday", short_window(bars, models, hol))
        line("gotobi, Tokyo open", short_window(bars, models, [d for d in goto if d not in JP_HOLIDAYS]))

    (OUT / "gotobi_robustness.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
