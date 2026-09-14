"""Wave 2: confirm the two wave-1 patterns on events wave 1 never touched.

Wave 1 (``macro_event_screen.py``) found two consistent-but-uncorrected
patterns. Each is tested here on a disjoint set of days, with the primary cell
fixed in this docstring before the run.

A. USTEC fades its 08:30 ET macro reaction. On CPI and NFP days, following the
   08:30 -> 08:35 move lost in all six USTEC cells and both splits - the
   opposite of the pre-registered continuation. That is now an in-sample
   observation, so it is confirmed on *other* 08:30 releases:

   Confirmation set: business days carrying an 08:30 ET Initial Jobless Claims,
   PPI, Retail Sales or GDP release and no CPI or NFP release.
   PRIMARY CELL: USTEC, fade the sign of 08:30 -> 08:35, enter 08:35, exit
   10:35 ET, net bps at real fills. Predicted > 0; passes at one-sided t > 1.645.
   Dose-response: on release days the 08:35 -> 10:35 return should load
   negatively on the 08:30 -> 08:35 move; on non-release days it should not.

B. The FOMC statement fade. Positive on all four instruments in wave 1. FOMC
   minutes print at 14:00 ET three weeks after each scheduled meeting - the same
   clock minute, a different document, and no press conference.
   PRIMARY CELLS: each instrument, fade 14:00 -> 14:25, hold 14:25 -> 15:55,
   minutes days. Predicted > 0 if the mechanism is overreaction to a 14:00
   release; ~0 if it is the press conference.

Also printed, labelled in-sample: the exit-time and signal-window surface of
the fade on the CPI/NFP days wave 1 already saw.

    python scripts/research/macro_event_confirm.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from macro_event_screen import (  # noqa: E402
    OUT, SPLIT_NAMES, UNSCHEDULED, _split_of, at, business_days, load, quotes_at,
    report, sign_move, trades,
)

from qlab.costs import CostModel  # noqa: E402
from qlab.macro_calendar import FOMC_DATES, build_calendar  # noqa: E402
from qlab.stats import ols_hac  # noqa: E402
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

CONFIRM_EVENTS = {"Initial Jobless Claims", "PPI", "Retail Sales", "GDP"}
TIER1_EVENTS = {"CPI", "Nonfarm Payrolls"}


def in_sample(d: date) -> bool:
    return _split_of(datetime(d.year, d.month, d.day)) != "other"


def day_sets() -> dict[str, list[date]]:
    cal = build_calendar(date(2020, 1, 1), date(2025, 6, 30))
    at830 = cal.filter((pl.col("et_hour") == 8) & (pl.col("et_minute") == 30))
    by_day: dict[date, set[str]] = {}
    for d, ev in at830.select("day", "event").iter_rows():
        by_day.setdefault(d, set()).add(ev)
    tier1 = sorted(d for d, evs in by_day.items() if evs & TIER1_EVENTS and in_sample(d))
    confirm = sorted(d for d, evs in by_day.items()
                     if not (evs & TIER1_EVENTS) and evs & CONFIRM_EVENTS and in_sample(d))
    control = [d for d in business_days() if d not in by_day]
    return {"tier1": tier1, "confirm": confirm, "control": control}


def fade_830(sym, bars, models, days, *, k: int = 5, hold: int = 120,
             exit_clock: tuple[int, int] | None = None) -> pl.DataFrame:
    signs = sign_move(bars, [at(d, 8, 30) for d in days], [at(d, 8, 30 + k) for d in days])
    keep = [(d, -s) for d, s in zip(days, signs) if s is not None]
    start = [at(d, 8, 30 + k) for d, _ in keep]
    if exit_clock is None:
        end = [t + timedelta(minutes=hold) for t in start]
    else:
        end = [at(d, *exit_clock) for d, _ in keep]
    return trades(sym, bars, models, start, end, [s for _, s in keep])


def dose_response(bars, days, label) -> dict:
    t0 = quotes_at(bars, [at(d, 8, 30) for d in days])["close"].to_numpy()
    t1 = quotes_at(bars, [at(d, 8, 35) for d in days])["close"].to_numpy()
    t2 = quotes_at(bars, [at(d, 10, 35) for d in days])["close"].to_numpy()
    init = (t1 / t0 - 1) * 1e4
    fwd = (t2 / t1 - 1) * 1e4
    res = ols_hac(fwd, init, names=["init_bps"], lags=0)
    b, t, p = res.coef("init_bps")
    print(f"    {label:<10} fwd(08:35->10:35) on init(08:30->08:35): beta {b:+.3f}  t {t:+.2f}  n {res.n}"
          f"  |init| median {np.nanmedian(np.abs(init)):.1f} bps")
    return {"label": label, "beta": b, "t": t, "p": p, "n": res.n}


def minutes_days() -> list[date]:
    return [d + timedelta(days=21) for d in FOMC_DATES
            if d not in UNSCHEDULED and in_sample(d + timedelta(days=21))]


def fade_1400(sym, bars, models, days) -> pl.DataFrame:
    signs = sign_move(bars, [at(d, 14, 0) for d in days], [at(d, 14, 25) for d in days])
    keep = [(d, -s) for d, s in zip(days, signs) if s is not None]
    return trades(sym, bars, models, [at(d, 14, 25) for d, _ in keep],
                  [at(d, 15, 55) for d, _ in keep], [s for _, s in keep])


def main() -> None:
    sets = day_sets()
    print({k: len(v) for k, v in sets.items()})
    rows: list[dict] = []
    ustec = load("USTEC")

    print("\n=== A. 08:30 fade - CONFIRMATION set (primary cell: USTEC 120m)")
    per_symbol_bars = {}
    for sym in ALL_SYMBOLS:
        bars = ustec if sym == "USTEC" else load(sym)
        per_symbol_bars[sym] = bars
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        for hold in (60, 120, 240):
            ev = fade_830(sym, bars, models, sets["confirm"], hold=hold)
            ctl = fade_830(sym, bars, models, sets["control"], hold=hold)
            rows.append(report(f"{sym} A confirm fade {hold}m", ev, ctl))

    print("\n  dose-response, USTEC")
    dr = [dose_response(ustec, sets[k], k) for k in ("tier1", "confirm", "control")]

    print("\n=== A'. IN-SAMPLE surface on tier-1 (CPI/NFP) days, USTEC - not a test")
    models = {s: CostModel.from_profiles("USTEC", split=s) for s in SPLIT_NAMES}
    for k in (1, 2, 5, 10, 15):
        ev = fade_830("USTEC", ustec, models, sets["tier1"], k=k, hold=120)
        print(f"   k={k:>2}m", end="")
        report(f"USTEC tier1 fade k={k} 120m", ev, None)
    for clock in ((8, 45), (9, 0), (9, 30), (9, 45), (10, 0), (10, 35), (11, 30), (12, 35), (15, 55)):
        ev = fade_830("USTEC", ustec, models, sets["tier1"], exit_clock=clock)
        report(f"USTEC tier1 fade exit {clock[0]:02d}:{clock[1]:02d}", ev, None)

    print("\n=== B. FOMC-minutes fade (primary cells: each instrument)")
    mdays = minutes_days()
    print(f"  {len(mdays)} minutes days")
    for sym in ALL_SYMBOLS:
        bars = per_symbol_bars[sym]
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        rows.append(report(f"{sym} B minutes fade", fade_1400(sym, bars, models, mdays), None))

    (OUT / "confirm.json").write_text(json.dumps({"rows": rows, "dose": dr}, indent=1, default=str))


if __name__ == "__main__":
    main()
