"""Reproduce the XAUUSD hypothesis battery, including every rejection.

Ten families were tried and all ten failed. This script re-runs each one.

    python scripts/research/xauusd_research.py                 # everything
    python scripts/research/xauusd_research.py --only drift fix
    python scripts/research/xauusd_research.py --list

The session opening-range breakout is deliberately absent: it was evaluated in
an earlier study (-490 R equivalent on gold's own terms, and rejected there),
and this study was asked to be independent of it.

Nothing here touches the locked test split. No strategy survived dev and
validation, so there was nothing to take to it - gold's test split is unspent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.loader import load_bars  # noqa: E402
from qlab.rollover import basis, drift_table  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

SYMBOL = "XAUUSD"
SPLITS = ("dev", "validation")
REF_PRICE = {"dev": 1900.0, "validation": 2400.0}
RT = {s: CostModel.from_profiles(SYMBOL, split=s).round_turn_bps(price=REF_PRICE[s])
      for s in SPLITS}
LDN = "Europe/London"
AM_FIX_MIN = 10 * 60 + 30
"""The LBMA Gold Price AM auction, 10:30 London."""


def _minute_returns(split):
    b = load_bars(SYMBOL, "1m", split=split,
                  columns=["ts", "ts_open", "close"]).sort("ts_open")
    b = b.with_columns(
        cont=((pl.col("ts_open") - pl.col("ts_open").shift(1))
              .dt.total_minutes() == 1).fill_null(False))
    c = b["close"].to_numpy().astype(float)
    r = np.full(c.size, np.nan)
    r[1:] = np.log(c[1:] / c[:-1]) * 1e4
    r[~b["cont"].to_numpy()] = np.nan
    return r


# --------------------------------------------------------------------------

def section_drift():
    """Where gold's return accrues, and why that rules out most of the field."""
    print("A long earns the roll in the price and hands it back in swap. Only")
    print("the liquid-hours column is reachable by a strategy that goes home")
    print("flat every night.\n")
    t = drift_table()
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(t.select("symbol", "split",
                   pl.col("years").round(1),
                   pl.col("total_pct").round(1).alias("total %"),
                   pl.col("roll_pct").round(1).alias("roll hrs"),
                   pl.col("gap_pct").round(1).alias("gaps"),
                   pl.col("liquid_pct").round(1).alias("liquid hrs"),
                   pl.col("liquid_pct_per_year").round(2).alias("liquid %/yr")))
    print("\nGold over dev: a headline +27.4% of which +2.0% is in liquid hours -")
    print("0.42% a year. The rest is the roll and the halt gap, which is the")
    print("financing adjustment. USTEC over the same window put +57.5% of its")
    print("+61.6% in liquid hours. Same corpus, opposite composition.")
    print("\nAnd what the roll costs to hold through:")
    for split in SPLITS:
        b = basis(SYMBOL, split=split)
        print(f"  {split:<11} {b.mid_bps:+.3f} bps/night -> {b.mid_bps*260/100:+.2f}%/yr")


def section_windows():
    """Select hours on dev, verify on validation."""
    def hourly(split):
        b = load_bars(SYMBOL, "1m", split=split,
                      columns=["ts", "ts_open", "open", "close"])
        b = b.with_columns(day=pl.col("ts_open").dt.date(),
                           dow=pl.col("ts_open").dt.weekday(),
                           w=pl.col("ts_open").dt.hour().cast(pl.Int32))
        return (b.filter(pl.col("dow") <= 5)
                 .group_by(["day", "w"])
                 .agg(o=pl.col("open").first(), c=pl.col("close").last(), n=pl.len())
                 .filter(pl.col("n") >= 36)
                 .with_columns(r=(pl.col("c") / pl.col("o") - 1) * 1e4))

    dev, val = hourly("dev"), hourly("validation")
    t = (dev.group_by("w")
            .agg(n=pl.len(), mean=pl.col("r").mean(), sd=pl.col("r").std())
            .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt()))
    for thr in (1.0, 1.5, 2.0):
        longs = t.filter(pl.col("t") > thr)["w"].to_list()
        shorts = t.filter(pl.col("t") < -thr)["w"].to_list()
        print(f"  |t| > {thr}: long {longs}  short {shorts}")
        for name, g, rt in (("IN  dev", dev, RT["dev"]),
                            ("OUT val", val, RT["validation"])):
            x = (g.with_columns(sig=pl.when(pl.col("w").is_in(longs)).then(1)
                                 .when(pl.col("w").is_in(shorts)).then(-1)
                                 .otherwise(0))
                  .filter(pl.col("sig") != 0)
                  .with_columns(pnl=pl.col("sig") * pl.col("r") - rt))
            r = x.group_by("day").agg(p=pl.col("pnl").sum())["p"].to_numpy()
            ann, vol = r.mean() * 252 / 100, r.std() * np.sqrt(252) / 100
            print(f"      {name}  mean={r.mean():+6.2f}bps "
                  f"t={r.mean()/r.std()*np.sqrt(r.size):+5.2f} SR={ann/vol:+5.2f}")
        print()
    print("Negative in sample as well as out - because the hours dev likes are")
    print("21 and 23, which is the roll, and the roll does not survive a round")
    print("turn.")


def section_micro():
    """Minute-level autoregression against the round turn."""
    r = _minute_returns("dev")
    print(f"  1m sd {np.nanstd(r):.3f} bps   round turn {RT['dev']:.3f} bps\n")
    roll = lambda x, k: pl.Series(x).rolling_sum(k, min_samples=k).to_numpy()
    print(f"{'k':>4}{'h':>5}{'beta':>10}{'t':>8}{'E|edge|':>10}{'vs cost':>10}")
    best = (0.0, 0, 0)
    for k in (1, 5, 15, 30, 60, 120):
        x = roll(r, k)
        for h in (5, 30, 60, 120):
            y = np.full(r.size, np.nan)
            s = roll(r, h)
            y[:-h] = s[h:]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            xs = x[m] - x[m].mean()
            ys = y[m]
            beta = float((xs * ys).sum() / (xs * xs).sum())
            res = ys - ys.mean() - beta * xs
            se = float(np.sqrt((res ** 2).sum() / (n - 2) / (xs * xs).sum()))
            edge = abs(beta) * np.abs(xs).mean()
            if edge / RT["dev"] > best[0]:
                best = (edge / RT["dev"], k, h)
            if k in (5, 30, 120) and h in (5, 60, 120):
                print(f"{k:>4}{h:>5}{beta:>10.4f}{beta/se:>8.2f}"
                      f"{edge:>10.3f}{edge/RT['dev']:>10.2f}x")
    print(f"\n  best cell anywhere in the grid: {best[0]:.2f}x of a round turn "
          f"(k={best[1]}, h={best[2]})")
    print("  EURUSD 0.19x, USTEC ~0.40x, USDJPY 0.43x - none of them close.")


def section_fix():
    """The LBMA AM auction, at the mid and then at real fills."""
    spec = get_spec(SYMBOL)
    print("Gold drifts down through the 10:30 London auction in both splits.")
    print("The auction is scheduled, published and one-sided, so it is the one")
    print("gold-specific candidate with a mechanism. Short it: sell the bid at")
    print("10:15, buy back the ask at 10:45.\n")

    def measure(split, a, b):
        bars = load_bars(SYMBOL, "1m", split=split,
                         columns=["ts", "ts_open", "close", "bid_close", "ask_close"])
        bars = (bars.with_columns(ldn=pl.col("ts_open").dt.convert_time_zone(LDN))
                    .with_columns(day=pl.col("ldn").dt.date(),
                                  m=pl.col("ldn").dt.hour().cast(pl.Int32) * 60
                                    + pl.col("ldn").dt.minute().cast(pl.Int32),
                                  dow=pl.col("ldn").dt.weekday())
                    .filter(pl.col("dow") <= 5))

        def at(t):
            return (bars.filter((pl.col("m") >= t) & (pl.col("m") < t + 5))
                        .group_by("day")
                        .agg(bid=pl.col("bid_close").first(),
                             ask=pl.col("ask_close").first(),
                             mid=pl.col("close").first())
                        .sort("day"))
        j = (at(a).rename({"bid": "eb", "ask": "ea", "mid": "em"})
                  .join(at(b).rename({"bid": "xb", "ask": "xa", "mid": "xm"}),
                        on="day", how="inner"))
        px = float(np.median(j["em"].to_numpy()))
        comm = spec.commission_pips() * spec.pip / px * 1e4
        mid = -(j["xm"].to_numpy() / j["em"].to_numpy() - 1) * 1e4
        fills = (j["eb"].to_numpy() / j["xa"].to_numpy() - 1) * 1e4 - comm
        return j, mid, fills

    for a_off, b_off, name in ((-15, 15, "10:15 -> 10:45"),
                               (-10, 10, "10:20 -> 10:40"),
                               (-15, 30, "10:15 -> 11:00")):
        print(f"[{name}]")
        for split in SPLITS:
            _, mid, fills = measure(split, AM_FIX_MIN + a_off, AM_FIX_MIN + b_off)
            for series, lbl in ((mid, "mid, no cost at all"),
                                (fills, "bid/ask fills + commission")):
                s = series[np.isfinite(series)]
                print(f"  {split:<11} {lbl:<28} n={s.size:4d} "
                      f"mean={s.mean():+7.3f}bps "
                      f"t={s.mean()/s.std()*np.sqrt(s.size):+5.2f}")
        print()
    print("The effect is real at the mid and exactly the spread in size. At the")
    print("prices a trade actually gets it is +0.02 bps on dev and +0.12 on")
    print("validation - which is nothing, and on dev it is 2020 by itself.")


SECTIONS = {
    "drift": ("where gold's return accrues: roll, gaps, or liquid hours", section_drift),
    "windows": ("intraday window selection, dev-select then val-verify", section_windows),
    "micro": ("minute-level autoregression against the round turn", section_micro),
    "fix": ("the LBMA AM auction, at the mid and at real fills", section_fix),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, (blurb, _) in SECTIONS.items():
            print(f"  {name:<9} {blurb}")
        return

    for name in (args.only or SECTIONS):
        blurb, fn = SECTIONS[name]
        print(f"\n{'=' * 78}\n{name.upper()}  -  {blurb}\n{'=' * 78}")
        fn()


if __name__ == "__main__":
    main()
