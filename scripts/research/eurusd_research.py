"""Reproduce the EURUSD hypothesis battery, including every rejection.

Twelve families were tried and all twelve failed. This script is the record:
it re-runs each one and prints what it printed the first time, so the rejection
is checkable rather than asserted.

    python scripts/research/eurusd_research.py                # everything
    python scripts/research/eurusd_research.py --only cost windows
    python scripts/research/eurusd_research.py --list

Nothing here touches the locked test split. The one place test data was read in
this study is documented in docs/findings/eurusd-rejected.md.
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
from qlab.rollover import basis, basis_table  # noqa: E402

SYMBOL = "EURUSD"
SPLITS = ("dev", "validation")
# Round turn in bps at the median price, per split. Read once here rather than
# in each section so no section can quietly cost itself differently.
RT = {s: CostModel.from_profiles(SYMBOL, split=s).round_turn_bps(price=1.10)
      for s in SPLITS}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def minute_returns(symbol: str, split: str, allow_test: bool = False):
    """Contiguous minute log returns in bps; NaN wherever a minute is missing."""
    b = load_bars(symbol, "1m", split=split, allow_test=allow_test,
                  columns=["ts", "ts_open", "open", "close",
                           "bid_close", "ask_close"]).sort("ts_open")
    b = b.with_columns(
        cont=((pl.col("ts_open") - pl.col("ts_open").shift(1))
              .dt.total_minutes() == 1).fill_null(False))
    c = b["close"].to_numpy().astype(float)
    r = np.full(c.size, np.nan)
    r[1:] = np.log(c[1:] / c[:-1]) * 1e4
    r[~b["cont"].to_numpy()] = np.nan
    return b, c, r


def daily_panel(symbol: str, split: str):
    """One row per FX day, the day ending 17:00 New York."""
    b = load_bars(symbol, "1m", split=split,
                  columns=["ts", "ts_open", "open", "close"])
    b = (b.with_columns(ny=pl.col("ts_open").dt.convert_time_zone("America/New_York"))
          .with_columns(nymin=pl.col("ny").dt.hour().cast(pl.Int32) * 60
                              + pl.col("ny").dt.minute().cast(pl.Int32))
          .sort("ts_open"))
    b = b.with_columns(
        fxd=pl.when(pl.col("nymin") >= 17 * 60)
             .then(pl.col("ny").dt.date() + pl.duration(days=1))
             .otherwise(pl.col("ny").dt.date()))
    d = (b.group_by("fxd")
          .agg(c=pl.col("close").last(), n_bars=pl.len())
          .sort("fxd")
          .filter(pl.col("n_bars") >= 1000))
    return d.with_columns(dow=pl.col("fxd").dt.weekday(),
                          r=(pl.col("c") / pl.col("c").shift(1) - 1) * 1e4)


def line(x, label, cost=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - cost
    n = x.size
    if n < 25:
        return f"  {label:44s} n={n} (too few to say anything)"
    m, sd = x.mean(), x.std()
    t = m / sd * np.sqrt(n) if sd > 0 else float("nan")
    return (f"  {label:44s} n={n:5d} mean={m:+7.2f}bps t={t:+5.2f} "
            f"hit={100 * (x > 0).mean():4.1f}%")


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def section_cost():
    """What a round turn costs per unit of the move it has to beat."""
    print("A round turn in bps says which instrument is cheap. Divided by the")
    print("volatility a strategy has to beat, it says something different - and")
    print("it reverses the ranking.\n")
    print(f"{'symbol':<9}{'rt bps':>9}{'1m sd':>9}{'daily sd':>10}"
          f"{'rt/1m sd':>11}{'rt/daily sd':>13}")
    for sym in ("EURUSD", "USDJPY", "XAUUSD", "USTEC"):
        _, c, r = minute_returns(sym, "dev")
        sd1 = float(np.nanstd(r))
        rt = CostModel.from_profiles(sym, split="dev").round_turn_bps(
            price=float(np.nanmedian(c)))
        sdd = float(daily_panel(sym, "dev")["r"].std())
        print(f"{sym:<9}{rt:>9.3f}{sd1:>9.3f}{sdd:>10.1f}"
              f"{rt / sd1:>11.2f}{rt / sdd:>13.4f}")
    print("\nEURUSD is the cheapest instrument in bps and the second dearest per")
    print("unit of volatility. USTEC is the dearest in bps and much the cheapest")
    print("per unit of volatility - 1.6x cheaper than EURUSD either way you")
    print("normalise. Cost in bps is not the number that decides what to trade.")


def section_rollover():
    """The rollover basis, and the carry prediction it confirms."""
    print("A long held across the broker's daily roll earns a price drift. The")
    print("sign it should have, if that drift is compensation for swap rather")
    print("than an edge, is fixed by which way the carry runs:\n")
    print("  a long that PAYS carry     -> price should drift UP")
    print("  a long that RECEIVES carry -> price should drift DOWN\n")
    print("Over this corpus the ECB sat below the Fed throughout (long EURUSD")
    print("pays), JPY sat at or below zero throughout (long USDJPY receives),")
    print("and gold yields nothing while financing costs (long pays).\n")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print(basis_table())
    print()
    for sym in ("EURUSD", "USDJPY", "XAUUSD"):
        for split in SPLITS:
            b = basis(sym, split=split)
            print(" ", b.describe())
    print("\nSix of six agree. The instrument whose long receives carry drifts")
    print("down; both that pay drift up. That is not an edge, it is the roll -")
    print("and capturing it requires being charged the swap it is offsetting.")


def section_windows():
    """Select intraday windows on dev, verify on validation."""
    print("Rank the 24 UTC hours by their dev t-statistic, take everything past")
    print("a threshold, then run that fixed selection forward untouched.\n")

    def hourly(split):
        b = load_bars(SYMBOL, "1m", split=split,
                      columns=["ts", "ts_open", "open", "close"])
        b = b.with_columns(
            day=pl.col("ts_open").dt.date(),
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
            .sort("w")
            .with_columns(t=pl.col("mean") / pl.col("sd") * pl.col("n").sqrt()))
    for thr in (1.0, 1.5, 2.0):
        longs = t.filter(pl.col("t") > thr)["w"].to_list()
        shorts = t.filter(pl.col("t") < -thr)["w"].to_list()
        print(f"  selection at |t| > {thr}: long {longs}  short {shorts}")
        for name, g, rt in (("IN  dev", dev, RT["dev"]),
                            ("OUT val", val, RT["validation"])):
            x = (g.with_columns(sig=pl.when(pl.col("w").is_in(longs)).then(1)
                                 .when(pl.col("w").is_in(shorts)).then(-1)
                                 .otherwise(0))
                  .filter(pl.col("sig") != 0)
                  .with_columns(pnl=pl.col("sig") * pl.col("r") - rt))
            r = x.group_by("day").agg(p=pl.col("pnl").sum()).sort("day")["p"].to_numpy()
            m, sd = r.mean(), r.std()
            ann, vol = m * 252 / 100, sd * np.sqrt(252) / 100
            print(f"      {name}  days={r.size:5d} mean={m:+7.2f}bps "
                  f"t={m / sd * np.sqrt(r.size):+5.2f} ann={ann:+6.2f}% "
                  f"SR={ann / vol:+5.2f}")
        print()
    print("Every threshold: mildly positive in sample, sharply negative out.")
    print("Note which hours survive the dev filter at |t|>2 - 21 and 22 are the")
    print("roll, so the one stable part of the intraday shape is the thing the")
    print("previous section already disqualified.")


def section_micro():
    """Minute-level autoregression: real, and far too small."""
    print("Reversal is genuinely there in a million and a half minute bars. The")
    print("column that matters is the last one - the edge the fit implies, as a")
    print("multiple of what a round turn costs.\n")
    _, _, r = minute_returns(SYMBOL, "dev")
    print(f"  1m sd {np.nanstd(r):.3f} bps   round turn {RT['dev']:.3f} bps\n")
    print(f"{'k':>4}{'h':>5}{'n':>11}{'beta':>10}{'t':>8}"
          f"{'E|edge| bps':>13}{'vs cost':>10}")
    roll = lambda x, k: pl.Series(x).rolling_sum(k, min_samples=k).to_numpy()
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
            print(f"{k:>4}{h:>5}{n:>11,}{beta:>10.4f}{beta / se:>8.2f}"
                  f"{edge:>13.3f}{edge / RT['dev']:>10.2f}x")
    print("\nThe best cell reaches 0.19x of a round turn. Statistical")
    print("significance at t = -20 and economic significance are different")
    print("questions, and only one of them is answered here.")


def section_daily():
    """Momentum, reversal, day of week, volatility regime, month end."""
    print("Daily-frequency hypotheses, dev and validation side by side so a")
    print("sign that flips cannot be quietly dropped.\n")
    for split in SPLITS:
        rt = RT[split]
        d = daily_panel(SYMBOL, split).with_columns(
            prev=pl.col("r").shift(1),
            vol20=pl.col("r").rolling_std(20).shift(1),
        )
        for k in (5, 20, 60, 120, 250):
            d = d.with_columns(**{f"mom{k}": pl.col("r").rolling_sum(k).shift(1)})
        print(f"-- {split}  ({d.height} sessions, round turn {rt:.3f} bps)")
        for k in (5, 20, 60, 120, 250):
            g = d.drop_nulls([f"mom{k}", "r"])
            print(line(np.sign(g[f"mom{k}"].to_numpy()) * g["r"].to_numpy(),
                       f"follow {k}-day momentum", rt))
        g = d.drop_nulls(["prev", "r"])
        print(line(-np.sign(g["prev"].to_numpy()) * g["r"].to_numpy(),
                   "fade yesterday", rt))
        for dw in range(1, 6):
            print(line(d.filter(pl.col("dow") == dw)["r"].to_numpy(),
                       f"long on weekday {dw}", rt))
        g = d.drop_nulls(["vol20", "r"]).with_columns(
            q=pl.col("vol20").qcut(3, labels=["lo", "mid", "hi"]))
        for q in ("lo", "mid", "hi"):
            print(line(g.filter(pl.col("q") == q)["r"].to_numpy(),
                       f"long | volatility {q}", rt))
        eom = d.with_columns(
            is_eom=(pl.col("fxd").dt.month() != pl.col("fxd").dt.month().shift(-1)))
        print(line(eom.filter(pl.col("is_eom"))["r"].to_numpy(),
                   "long on the last day of the month", rt))
        print()
    print("Not one sign holds across the two periods, and nothing reaches t = 2")
    print("in either. Dev has about a thousand sessions at a 49 bps standard")
    print("deviation, so a strategy would need a Sharpe near 1.0 to register at")
    print("t = 2 at all - which is the real constraint here, not the search.")


SECTIONS = {
    "cost": ("cost per unit of volatility, all four symbols", section_cost),
    "rollover": ("the rollover basis and the carry prediction", section_rollover),
    "windows": ("intraday window selection, dev-select then val-verify", section_windows),
    "micro": ("minute-level autoregression against the round turn", section_micro),
    "daily": ("daily momentum, reversal, seasonality, volatility", section_daily),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), default=None)
    ap.add_argument("--list", action="store_true", help="name the sections and exit")
    args = ap.parse_args()

    if args.list:
        for name, (blurb, _) in SECTIONS.items():
            print(f"  {name:<10} {blurb}")
        return

    for name in (args.only or SECTIONS):
        blurb, fn = SECTIONS[name]
        print(f"\n{'=' * 78}\n{name.upper()}  -  {blurb}\n{'=' * 78}")
        fn()


if __name__ == "__main__":
    main()
