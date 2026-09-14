"""Wave 3: order flow at the FX fixings. Primary cells fixed before the run.

Two scheduled benchmark prices at which non-discretionary customer flow is
executed, each with a published mechanism and neither tested in this project
outside month-end.

G. The Tokyo fix on gotobi days (Ito & Yamada 2017). Japanese importers settle
   on the 5th, 10th, 15th, 20th, 25th and last day of the month ("gotobi"),
   buying USD at the 09:55 JST Tokyo fix; banks pre-hedge into it.
   G1 PRIMARY: long USDJPY 09:00 -> 09:55 JST on gotobi days. Predicted > 0.
   G2 PRIMARY: short USDJPY 09:55 -> 11:00 JST on gotobi days (the unwind).
   Control: the same clock windows on non-gotobi weekdays.
   A gotobi date on a weekend moves to the preceding Friday. Japanese public
   holidays are not modelled - on those Tokyo is shut and the effect should be
   absent, which biases the estimate toward zero, not away from it.

F. The WM/R 16:00 London fix (Evans 2018). Fix orders are worked through the
   window around 16:00 and the price they push reverts afterwards.
   F1 PRIMARY: fade the sign of 15:30 -> 16:00 London, enter 16:03 (after the
   five-minute calculation window), exit 17:00 London; EURUSD, USDJPY, XAUUSD;
   every weekday. Predicted > 0.
   F2 PRIMARY: F1 on the last weekday of each month only (Melvin & Prins 2015).
   Control: the same fade one clock hour earlier on the same days -
   12:30 -> 13:00, enter 13:03, exit 14:00 London.

USTEC is carried through both as a falsification instrument: neither fix is
its benchmark, so neither rule should work on it.

    python scripts/research/fx_fix_screen.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from macro_event_screen import (  # noqa: E402
    OUT, SPLIT_NAMES, _bonf_t, at, load, report, sign_move, trades,
)

from qlab.costs import CostModel  # noqa: E402
from qlab.loader import SPLITS  # noqa: E402
from qlab.strategies.fix_flows import fx_weekdays, gotobi_days, month_ends  # noqa: E402,F401
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

JST = ZoneInfo("Asia/Tokyo")
LDN = ZoneInfo("Europe/London")


def weekdays() -> list[date]:
    """FX weekdays across dev and validation - never the locked split."""
    return [d for name in SPLIT_NAMES for d in fx_weekdays(SPLITS[name].start, SPLITS[name].end)]


def window(sym, bars, models, days, t0, t1, direction) -> pl.DataFrame:
    return trades(sym, bars, models, [at(d, *t0, JST) for d in days],
                  [at(d, *t1, JST) for d in days], [direction] * len(days))


def fade(sym, bars, models, days, sig0, sig1, entry, exit_, tz) -> pl.DataFrame:
    signs = sign_move(bars, [at(d, *sig0, tz) for d in days], [at(d, *sig1, tz) for d in days])
    keep = [(d, -s) for d, s in zip(days, signs) if s is not None]
    return trades(sym, bars, models, [at(d, *entry, tz) for d, _ in keep],
                  [at(d, *exit_, tz) for d, _ in keep], [s for _, s in keep])


def main() -> None:
    days = weekdays()
    goto = gotobi_days(days)
    ends = month_ends(days)
    g_days = sorted(goto)
    n_days = [d for d in days if d not in goto]
    e_days = sorted(ends)
    print(f"{len(days)} weekdays, {len(g_days)} gotobi, {len(e_days)} month-ends")

    rows: list[dict] = []
    for sym in ALL_SYMBOLS:
        bars = load(sym)
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        print(f"\n== {sym}")
        # USD-long direction for the Tokyo-fix legs: USDJPY long is USD long.
        usd = {"USDJPY": 1, "EURUSD": -1, "XAUUSD": -1, "USTEC": 1}[sym]
        rows.append(report(f"{sym} G1 into fix 09:00-09:55",
                           window(sym, bars, models, g_days, (9, 0), (9, 55), usd),
                           window(sym, bars, models, n_days, (9, 0), (9, 55), usd)))
        rows.append(report(f"{sym} G2 unwind 09:55-11:00",
                           window(sym, bars, models, g_days, (9, 55), (11, 0), -usd),
                           window(sym, bars, models, n_days, (9, 55), (11, 0), -usd)))
        rows.append(report(f"{sym} G1' into fix 08:00-09:55",
                           window(sym, bars, models, g_days, (8, 0), (9, 55), usd),
                           window(sym, bars, models, n_days, (8, 0), (9, 55), usd)))
        f_args = ((15, 30), (16, 0), (16, 3), (17, 0), LDN)
        c_args = ((12, 30), (13, 0), (13, 3), (14, 0), LDN)
        rows.append(report(f"{sym} F1 WM/R fade all days",
                           fade(sym, bars, models, days, *f_args),
                           fade(sym, bars, models, days, *c_args)))
        rows.append(report(f"{sym} F2 WM/R fade month-end",
                           fade(sym, bars, models, e_days, *f_args),
                           fade(sym, bars, models, e_days, *c_args)))

    (OUT / "fx_fix.json").write_text(json.dumps(rows, indent=1, default=str))
    n = len(rows)
    print(f"\n{n} cells; Bonferroni |t| >= {_bonf_t(n):.2f}. Primary cells pass at one-sided t > 1.645.")


if __name__ == "__main__":
    main()
