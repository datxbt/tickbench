"""Wave 4: fading the move into benchmark auctions this project never tested.

The Tokyo gotobi fix produced the one real in-sample regularity in the
scheduled-flows study before it vanished on test; the WM/R 16:00 fade lost on
every instrument. Two further benchmarks, each with its own counterparty and
each set daily:

* The Shanghai Gold Benchmark (SGE), auctions at 10:15 and 14:15 Beijing time.
  Chinese banks and importers execute there; China keeps no daylight saving, so
  the UTC instants never move (02:15 and 06:15).
* The ECB euro reference rate, set at 14:15 CET from a concertation of central
  banks; commercial contracts and fund valuations reference it.

Rule, the same shape as the WM/R fade: take the sign of the 30 minutes into the
benchmark, fade it from three minutes after the benchmark (the auction window),
exit an hour later. Control: the identical fade one clock hour earlier.

Pre-registered before the run (2026-09-11):
* PRIMARY CELLS: XAUUSD at SGE-AM, XAUUSD at SGE-PM, EURUSD at the ECB fix.
  Predicted > 0. PASS at one-sided t > 2.13 (Bonferroni over three) and the same
  sign in dev and validation.
* The other instruments are run through each window as falsification.
* Chinese public holidays (no SGE auction) are not modelled; that biases the
  gold cells toward zero, not away from it.

    python scripts/research/fix_benchmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from fx_fix_screen import fade, weekdays  # noqa: E402
from macro_event_screen import OUT, SPLIT_NAMES, load, report  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.symbols import ALL_SYMBOLS  # noqa: E402

BJ = ZoneInfo("Asia/Shanghai")
FFM = ZoneInfo("Europe/Berlin")

# name: (tz, signal start, benchmark, entry, exit, control shift in hours, primary instrument)
WINDOWS = {
    "SGE-AM": (BJ, (9, 45), (10, 15), (10, 18), (11, 18), "XAUUSD"),
    "SGE-PM": (BJ, (13, 45), (14, 15), (14, 18), (15, 18), "XAUUSD"),
    "ECB": (FFM, (13, 45), (14, 15), (14, 18), (15, 18), "EURUSD"),
}


def shift(hm: tuple[int, int], h: int) -> tuple[int, int]:
    return (hm[0] - h, hm[1])


def main() -> None:
    days = weekdays()
    rows = []
    for sym in ALL_SYMBOLS:
        bars = load(sym)
        models = {s: CostModel.from_profiles(sym, split=s) for s in SPLIT_NAMES}
        print(f"\n== {sym}")
        for name, (tz, s0, s1, e0, e1, primary) in WINDOWS.items():
            ev = fade(sym, bars, models, days, s0, s1, e0, e1, tz)
            ctl = fade(sym, bars, models, days, shift(s0, 1), shift(s1, 1), shift(e0, 1), shift(e1, 1), tz)
            row = report(f"{sym} {name} fade" + (" <- primary" if sym == primary else ""), ev, ctl)
            row["primary"] = sym == primary
            rows.append(row)
    (OUT / "fix_benchmarks.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\nPrimary cells (pass: one-sided t > 2.13 pooled, same sign in both splits):")
    for r in rows:
        if r["primary"]:
            p, d, v = (r[k]["net_bps"] for k in ("pooled", "dev", "validation"))
            ok = p["t"] > 2.13 and d["mean"] > 0 and v["mean"] > 0
            print(f"  {r['cell']:<34} t {p['t']:+.2f}  dev {d['mean']:+.2f}  val {v['mean']:+.2f}  {'PASS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
