"""Test the overnight drift of Boyarchenko, Larsen and Whelan (NY Fed SR 917).

The paper's claim, in one line: US equity futures earn an annualised 3.7% in the
single hour between 02:00 and 03:00 New York, it is the only hour that survives
a multiple-testing correction, and the conditional version - hold 01:30-03:30
only after a US close with negative order imbalance - is the only version that
survives the bid-ask spread.

The script tests that in six passes, in the order in which the claim can die:

1. **The hour.** Every clock hour's mean return with a t-statistic, then
   Bonferroni and Benjamini-Hochberg corrections over the 23 hours, which is the
   test the paper itself insists on. If 02:00 is not special here, nothing after
   this matters.
2. **The strategy table.** The paper's Table IX - CTC, CTO, OTC, OD, OD+ and the
   conditional BtD - gross, after crossing the spread, and after commission too.
3. **The asymmetry.** BtD against its mirror image: hold the same window after a
   *positive* closing imbalance. The paper's whole mechanism is that selloffs
   reverse and rallies do not, so the two must differ. This is the sharpest
   test in the script because both legs have the same geometry and the same cost.
4. **The sort.** Next-session drift by quintile of the previous close's
   imbalance, on the proxy and on the closing-hour return, which needs no proxy.
5. **Predictability.** Each hour's return regressed on the previous close's
   imbalance with Newey-West errors - the paper's Table VII. The loading is
   supposed to appear at the European open and nowhere else.
6. **Falsification.** The same hour on gold and two FX majors. An inventory
   story about US equity order flow predicts nothing for EURUSD at 02:00, so a
   drift that shows up everywhere is a clock artefact, not this effect.

Run:  python scripts/backtests/backtest_overnight_drift.py
      python scripts/backtests/backtest_overnight_drift.py --splits test   # locked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.stats import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    normal_sf,
    ols_hac,
    paired_bootstrap,
)
from qlab.strategies.overnight_drift import (  # noqa: E402
    CONDITIONALS,
    STRATEGIES,
    closing_imbalance,
    hourly_returns,
    sort_by_signal,
    strategy_returns,
    summarize,
)

REPORT_DIR = paths.STRATEGY_REPORT_DIR
PRIMARY = "USTEC"
FALSIFICATION = ("XAUUSD", "EURUSD", "USDJPY")
DRIFT_HOUR = 2
BOOTSTRAP_DRAWS = 5000

SUFFIXES = ("", "_net", "_all")
SUFFIX_LABELS = {"": "gross", "_net": "+spread", "_all": "+commission"}

# The paper's own figures, for the column that says whether this replicates.
PAPER_SHARPE: dict[str, tuple[float, float]] = {   # (gross, post-cost)
    "CTC": (0.42, 0.42),
    "CTO": (0.34, -0.04),
    "OTC": (0.23, -0.06),
    "OD": (1.10, -0.54),
    "OD+": (1.30, 0.26),
    "BtD": (1.78, 1.10),
}


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def rule(title: str, width: int = 100) -> None:
    print(f"\n{title}")
    print("-" * width)


def fmt(value, spec=".2f", width=8) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return f"{value:{spec}}".rjust(width)


def print_hours(frame: pl.DataFrame) -> None:
    print(f"  {'hour':>4}  {'n':>5}  {'mean bps':>9}  {'ann %':>7}  {'t':>7}"
          f"  {'p':>7}  {'p Bonf':>8}  {'p BH':>7}")
    for row in frame.iter_rows(named=True):
        star = "  <--" if row["hour"] == DRIFT_HOUR else ""
        print(f"  {row['hour']:>4}  {row['n']:>5}  {fmt(row['mean_bps'], '.3f', 9)}"
              f"  {fmt(row['ann_pct'], '+.2f', 7)}  {fmt(row['t'], '+.2f', 7)}"
              f"  {fmt(row['p'], '.4f', 7)}  {fmt(row['p_bonf'], '.4f', 8)}"
              f"  {fmt(row['p_bh'], '.4f', 7)}{star}")


def print_strategies(frame: pl.DataFrame) -> None:
    print(f"  {'strategy':<10}  {'costs':<11}  {'days':>5}  {'active':>6}"
          f"  {'ann %':>7}  {'vol %':>7}  {'Sharpe':>7}  {'t':>6}"
          f"  {'paper':>7}")
    for name in list(STRATEGIES) + [c.name for c in CONDITIONALS]:
        for suffix in SUFFIXES:
            row = frame.filter(pl.col("strategy") == f"{name}{suffix}")
            if row.is_empty():
                continue
            r = row.row(0, named=True)
            paper = PAPER_SHARPE.get(name)
            reference = (
                "-" if paper is None
                else f"{paper[0]:+.2f}" if suffix == "" else
                (f"{paper[1]:+.2f}" if suffix == "_all" else "")
            )
            print(f"  {name:<10}  {SUFFIX_LABELS[suffix]:<11}  {r['n_days']:>5}"
                  f"  {r['days_active']:>6}  {fmt(r['ann_pct'], '+.2f', 7)}"
                  f"  {fmt(r['ann_vol_pct'], '.2f', 7)}"
                  f"  {fmt(r['sharpe'], '+.2f', 7)}  {fmt(r['t_stat'], '+.2f', 6)}"
                  f"  {reference:>7}")
        print()


def print_sort(frame: pl.DataFrame, label: str) -> None:
    print(f"  {label}")
    print(f"  {'quintile':>8}  {'n':>5}  {'signal':>9}  {'mean bps':>9}  {'t':>7}")
    for row in frame.iter_rows(named=True):
        print(f"  {row['bucket'] + 1:>8}  {row['n']:>5}"
              f"  {fmt(row['signal_mean'], '+.4f', 9)}"
              f"  {fmt(row['mean_bps'], '+.3f', 9)}  {fmt(row['t'], '+.2f', 7)}")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def hour_table(returns: pl.DataFrame) -> pl.DataFrame:
    """Mean return by clock hour, with corrections over the whole clock face.

    The correction is the point. Twenty-three hours is twenty-three hypotheses,
    and the paper is explicit that the drift hour is the one that survives
    them; an uncorrected t of 2 somewhere on the clock face is what a random
    clock face looks like.
    """
    stats = (
        returns.group_by("hour")
        .agg(n=pl.len(), mean_bps=pl.col("ret_bps").mean(),
             sd=pl.col("ret_bps").std())
        .filter(pl.col("n") >= 60)
        .with_columns(
            t=pl.col("mean_bps") / pl.col("sd") * pl.col("n").sqrt(),
            ann_pct=pl.col("mean_bps") / 1e4 * 252 * 100,
        )
        .sort("hour")
    )
    t = stats["t"].to_numpy()
    # Two-sided normal p; the samples here are 350-1000 sessions, where the
    # normal and the t distribution differ in the fourth decimal.
    p = np.array([2.0 * normal_sf(abs(v)) for v in t])
    return stats.with_columns(
        p=pl.Series("p", p),
        p_bonf=pl.Series("p_bonf", bonferroni(p)),
        p_bh=pl.Series("p_bh", benjamini_hochberg(p)),
    )


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def block_bootstrap_sharpe(series: pl.Series, *, draws: int = BOOTSTRAP_DRAWS,
                           seed: int = 0) -> tuple[float, float, float]:
    """A stationary-block bootstrap confidence interval for the Sharpe.

    Daily overnight returns are mildly autocorrelated and very fat-tailed, so
    the textbook Sharpe standard error understates the uncertainty. The block
    bootstrap keeps the dependence.
    """
    x = series.drop_nulls().to_numpy()
    x = x[np.isfinite(x)]
    if x.size < 30:
        return float("nan"), float("nan"), float("nan")
    result = bootstrap_ci(x, _sharpe, n_boot=draws,
                          rng=np.random.default_rng(seed))
    return result.point, result.lo, result.hi


def predictability(returns_by_hour: pl.DataFrame,
                   imbalance: pl.DataFrame) -> pl.DataFrame:
    """Each hour's return on the previous close's imbalance, Newey-West.

    The paper's Table VII. Because the imbalance is measured at the previous US
    close, a loading is a *prediction*, and it is supposed to appear when London
    and Frankfurt open and nowhere else.
    """
    # Attach yesterday's reading to today's session by shifting the signal
    # forward one *row* - one session, not one calendar day.
    lagged = imbalance.sort("nyd").select(
        nyd=pl.col("nyd"),
        prev_rsv=pl.col("rsv").shift(1),
        prev_close_ret=pl.col("close_ret_bps").shift(1),
    )
    joined = returns_by_hour.join(lagged, on="nyd", how="inner").drop_nulls(
        ["prev_rsv", "ret_bps"])
    rows = []
    for hour in sorted(joined["hour"].unique().to_list()):
        cell = joined.filter(pl.col("hour") == hour)
        if cell.height < 60:
            continue
        fit = ols_hac(cell["ret_bps"].to_numpy(),
                      cell["prev_rsv"].to_numpy().reshape(-1, 1),
                      names=["rsv"])
        beta, t, p = fit.coef("rsv")
        rows.append({"hour": hour, "n": fit.n, "beta": beta, "t": t, "p": p})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# One split
# --------------------------------------------------------------------------

def run_split(split: str, *, allow_test: bool) -> dict:
    result: dict = {"split": split}
    costs = CostModel.from_profiles(PRIMARY, split=split)

    rule(f"{split.upper()} 1/6 - every hour of the clock, "
         f"{PRIMARY}, corrected over 23 hypotheses")
    hours = hourly_returns(PRIMARY, split=split, allow_test=allow_test)
    table = hour_table(hours)
    print_hours(table)
    drift = table.filter(pl.col("hour") == DRIFT_HOUR)
    result["hours"] = table.to_dicts()
    if not drift.is_empty():
        d = drift.row(0, named=True)
        rank = int((table["t"] > d["t"]).sum()) + 1
        print(f"\n  the 02:00 hour ranks {rank} of {table.height} by t-statistic."
              f"  paper: 3.7% a year, the only hour to survive correction.")
        result["drift_rank"] = rank

    rule(f"{split.upper()} 2/6 - the paper's Table IX")
    returns = strategy_returns(PRIMARY, split=split, allow_test=allow_test,
                               costs=costs)
    names = [f"{n}{s}" for n in
             list(STRATEGIES) + [c.name for c in CONDITIONALS] for s in SUFFIXES]
    stats = summarize(returns, names)
    print_strategies(stats)
    result["strategies"] = stats.to_dicts()

    rule(f"{split.upper()} 3/6 - the asymmetry, which is the whole mechanism")
    print("  Same window, same cost, opposite condition. The paper says "
          "selloffs reverse\n  and rallies do not; if these two agree, the "
          "inventory story is not what is happening.")
    for name in ("OD+", "BtD", "BtD(ret)", "Rally"):
        column = f"{name}_all"
        point, lo, hi = block_bootstrap_sharpe(returns[column])
        active = returns.get_column(f"{name}_active", default=None)
        held = int(active.sum()) if active is not None else returns.height
        print(f"  {name:<10} held {held:>4}/{returns.height:<4} days"
              f"   Sharpe {fmt(point, '+.2f', 6)}"
              f"   95% block-bootstrap [{fmt(lo, '+.2f', 6)},{fmt(hi, '+.2f', 6)} ]")
        result.setdefault("bootstrap", {})[name] = {
            "sharpe": point, "lo": lo, "hi": hi, "days_held": held}

    # The two legs partition the same calendar, so the difference is paired and
    # the common dates cancel out of the variance.
    gap = paired_bootstrap(
        returns["BtD_all"].fill_null(0.0).to_numpy(),
        returns["Rally_all"].fill_null(0.0).to_numpy(),
        n_boot=BOOTSTRAP_DRAWS, rng=np.random.default_rng(1),
    )
    print(f"\n  BtD - Rally, mean bps a day: {gap.point:+.3f}"
          f"   95% [{gap.lo:+.3f}, {gap.hi:+.3f}]   p {gap.p_value:.3f}")
    result["asymmetry"] = {"diff_bps": gap.point, "lo": gap.lo, "hi": gap.hi,
                           "p_value": gap.p_value}

    rule(f"{split.upper()} 4/6 - sorted on the previous close")
    for column, label in (("prev_rsv", "relative signed volume (proxy)"),
                          ("prev_close_ret_bps", "closing-hour return (no proxy)")):
        sort = sort_by_signal(returns, column=column, target="OD+_all")
        if not sort.is_empty():
            print_sort(sort, f"OD+ net, by quintile of {label}")
            print()
            result.setdefault("sorts", {})[column] = sort.to_dicts()

    rule(f"{split.upper()} 5/6 - does the close predict the hour? (Table VII)")
    imbalance = closing_imbalance(PRIMARY, split=split, allow_test=allow_test)
    reg = predictability(hours, imbalance)
    print(f"  {'hour':>4}  {'n':>5}  {'beta':>10}  {'t':>7}  {'p':>7}")
    for row in reg.iter_rows(named=True):
        star = "  <--" if row["hour"] == DRIFT_HOUR else ""
        print(f"  {row['hour']:>4}  {row['n']:>5}  {fmt(row['beta'], '+.3f', 10)}"
              f"  {fmt(row['t'], '+.2f', 7)}  {fmt(row['p'], '.4f', 7)}{star}")
    print("\n  A negative beta is the paper's sign: a negative closing "
          "imbalance predicts a\n  positive return in the hour that follows.")
    result["predictability"] = reg.to_dicts()

    rule(f"{split.upper()} 6/6 - falsification: the same hour elsewhere")
    print("  An inventory story about US equity flow predicts nothing here. A "
          "drift that\n  shows up on every instrument is a clock artefact.")
    print(f"  {'symbol':<8}  {'n':>5}  {'mean bps':>9}  {'ann %':>7}  {'t':>7}")
    falsify = []
    for symbol in (PRIMARY,) + FALSIFICATION:
        other = hourly_returns(symbol, split=split, allow_test=allow_test)
        cell = other.filter(pl.col("hour") == DRIFT_HOUR)["ret_bps"]
        if cell.len() < 60:
            continue
        mean, sd, n = float(cell.mean()), float(cell.std()), cell.len()
        t = mean / sd * np.sqrt(n)
        row = {"symbol": symbol, "n": n, "mean_bps": mean,
               "ann_pct": mean / 1e4 * 252 * 100, "t": t}
        falsify.append(row)
        print(f"  {symbol:<8}  {n:>5}  {fmt(mean, '+.3f', 9)}"
              f"  {fmt(row['ann_pct'], '+.2f', 7)}  {fmt(t, '+.2f', 7)}")
    result["falsification"] = falsify
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", nargs="+", default=["dev", "validation"])
    p.add_argument("-o", "--out", default=str(REPORT_DIR / "overnight_drift.json"))
    args = p.parse_args(argv)

    if "test" in args.splits:
        print("!! the test split is locked; it is being read deliberately\n")

    results = {"primary": PRIMARY, "drift_hour": DRIFT_HOUR, "splits": {}}
    for split in args.splits:
        results["splits"][split] = run_split(
            split, allow_test=(split == "test"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
