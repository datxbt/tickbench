"""The gotobi unwind on the locked test split. Run once.

Protocol, fixed before the split was read (2026-09-11):

* Specification: G2 exactly as registered in ``fx_fix_screen.py`` - short
  USDJPY 09:55 -> 11:00 JST on gotobi days (5, 10, 15, 20, 25, month-end;
  a weekend date moves to the preceding Friday; Japanese holidays ignored).
  Priced two ways: the bar-level rule the dev/validation numbers use, and on
  ticks with an explicit 1-second latency on the entry (the deployable form).
* PASS: test net mean > 0 with one-sided t > 1.28 (p < 0.10) on the bar rule.
  At the dev+validation effect (+2.85 bps, sd ~12.8) the ~84 test events give an
  expected t near 2.0, so a 5% bar would fail a true effect about a third of the
  time; 10% is the stated compromise.
* FAIL: test net mean <= 0.
* Anything between is reported as inconclusive, not rounded either way.

Result, recorded in ``docs/findings/scheduled-flows.md``: FAIL, -0.51 bps at
t = -0.44 on 83 events. The split is spent for this family.

    python scripts/research/gotobi_test.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from gotobi_ticks import first_at_or_after  # noqa: E402
from macro_event_screen import OUT, at, trades  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.loader import get_split, load_bars, load_ticks  # noqa: E402
from qlab.strategies.fix_flows import fx_weekdays, gotobi_days  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

JST = ZoneInfo("Asia/Tokyo")
SYM = "USDJPY"


def gotobi_days_in_test() -> list[date]:
    s = get_split("test", allow_test=True)
    return sorted(gotobi_days(fx_weekdays(s.start, s.end)))


def summary(label: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    t = x.mean() / x.std(ddof=1) * np.sqrt(x.size)
    print(f"  {label:<28} n={x.size:>3}  net {x.mean():+.2f} bps  sd {x.std(ddof=1):.2f}  t {t:+.2f}  hit {(x > 0).mean():.2f}")
    return {"label": label, "n": int(x.size), "mean": float(x.mean()), "t": float(t)}


def main() -> None:
    days = gotobi_days_in_test()
    print(f"{len(days)} gotobi days in test ({days[0]} .. {days[-1]})")
    bars = load_bars(SYM, "1m", split="test", allow_test=True,
                     columns=["ts", "close", "bid_close", "ask_close"])
    # trades() files every date outside dev/validation under "other".
    model = CostModel.from_profiles(SYM, split="test")
    df = trades(SYM, bars, {"other": model}, [at(d, 9, 55, JST) for d in days],
                [at(d, 11, 0, JST) for d in days], [-1] * len(days))
    out = {"bar": summary("bar rule (as registered)", df["net_bps"].to_numpy())}

    spec = get_spec(SYM)
    lazy = load_ticks(SYM, split="test", allow_test=True, lazy=True)
    # dt.hour() is Int8: widen before multiplying or the product wraps.
    tod = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)
    ticks = lazy.filter((tod >= 50) & (tod < 125)).select("ts", "bid", "ask").collect().sort("ts")
    ticks = ticks.with_columns(day=pl.col("ts").dt.date())
    by_day = {k[0]: g for k, g in ticks.group_by("day", maintain_order=True)}
    net = []
    for d in days:
        g = by_day.get(d)
        if g is None:
            continue
        e = first_at_or_after(g, datetime(d.year, d.month, d.day, 0, 55, 1, tzinfo=timezone.utc))
        x = first_at_or_after(g, datetime(d.year, d.month, d.day, 2, 0, tzinfo=timezone.utc))
        if e is None or x is None:
            continue
        mid = (e["bid"] + e["ask"]) / 2
        comm = model.commission_pips(mid) * spec.pip / mid * 1e4
        net.append((e["bid"] / x["ask"] - 1) * 1e4 - comm)
    out["tick_1s"] = summary("ticks, 1 s entry latency", np.array(net))

    t = out["bar"]["t"]
    verdict = "PASS" if (out["bar"]["mean"] > 0 and t > 1.28) else ("FAIL" if out["bar"]["mean"] <= 0 else "INCONCLUSIVE")
    print(f"\nVerdict under the pre-registered rule: {verdict}")
    out["verdict"] = verdict
    (OUT / "gotobi_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
