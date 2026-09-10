"""Least time to a funded account, which is a different question from the last one.

`propfirm_eval.py` asked which sizing is safest, and answered: a small one, held
for years. This asks the question actually posed - how fast can a funded account
be reached - and the answer inverts. Three things drive it:

**Retries count.** A failed challenge is not the end of the project, it is an
entry fee and a few weeks. The statistic that matters is therefore the time
until the *first* success across however many attempts it takes, not the time
taken by the attempts that happened to work.

**A failure that arrives quickly is cheap in time.** Large sizes resolve fast in
both directions, which is why they win on calendar time while losing on pass
rate. Speed here is bought with fees, not with edge.

**The ceiling on size is not volatility, it is the gap.** A circuit breaker
handles volatility. Nothing handles the move between one session's last quote
and the next one's first, so the largest such move in the sample sets the
largest size that can be carried at all.

    python scripts/research/propfirm_speed.py                 # the sections below
    python scripts/research/propfirm_speed.py --section frontier
    python scripts/research/propfirm_speed.py --paths 3000    # tighter, slower

Nothing is fitted here. The strategy was selected in an earlier study and is
unchanged; what is chosen is a size and a breaker level, on dev and validation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import propfirm as pf  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.strategies import risk_managed_long as rml  # noqa: E402

SYMBOL = "USTEC"
PROP_CFG = replace(rml.RMLConfig(), max_weight=1.0)
"""The strategy as validated, with the 2.0 leverage cap brought down to 1.0 -
the one change the prop constraint forces, and it was made in the earlier
study for the excursion reason documented there."""

BREAKER = 0.03
SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
_cache: dict = {}


def records(split: str, *, flat_weekends: bool = False, allow_test: bool = False):
    """Close, worst moment and unstoppable part, per broker day."""
    key = (split, flat_weekends)
    if key not in _cache:
        frame = rml.run(SYMBOL, PROP_CFG, split=split, allow_test=allow_test,
                        cost=CostModel.from_profiles(SYMBOL, split=split))
        daily = pf.daily_records_from_minutes(
            SYMBOL, frame.select("nyd", "weight"), split=split,
            allow_test=allow_test, flat_over_closures=flat_weekends)
        _cache[key] = (daily["ret"].to_numpy(), daily["worst"].to_numpy(),
                       daily["gap"].to_numpy())
    return _cache[key]


def rules(breaker: float = BREAKER, slippage: float = 0.002) -> pf.PropRules:
    return replace(pf.FTMO_TWO_STEP, daily_stop=breaker,
                   daily_stop_slippage=slippage)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def section_ceiling(splits):
    """The size above which a single reopen ends the account outright."""
    print("A breaker is a promise to act, and a gap is the market refusing to")
    print("let you. Between one session's last quote and the next one's first")
    print("there is no fill at any price, so the worst gap in the sample - not")
    print("the worst day - is what caps the size.\n")
    print(f"{'split':>12}{'weekends':>10}{'worst gap':>11}{'1-in-1000':>11}"
          f"{'worst day':>11}{'size cap':>10}")
    print("-" * 66)
    for split in splits:
        for flat in (False, True):
            _, w, g = records(split, flat_weekends=flat)
            print(f"{split:>12}{'flat' if flat else 'held':>10}"
                  f"{g.min() * 100:>10.2f}%{np.quantile(g, 0.001) * 100:>10.2f}%"
                  f"{w.min() * 100:>10.2f}%"
                  f"{0.05 / abs(g.min()):>9.2f}x")
        print()


def section_frontier(splits, paths, horizon):
    """Time to funded against size, retries counted."""
    print("Median months is the honest headline: half the runs get there sooner,")
    print("half later, and every failed attempt spends both days and an entry fee.")
    print("P(pass) is per attempt, not per project.\n")
    print(f"{'split':>12}{'size':>7}{'med mo':>8}{'p75 mo':>8}{'p90 mo':>8}"
          f"{'attempts':>10}{'E[fees]':>9}{'P(pass)':>9}")
    print("-" * 71)
    for split in splits:
        r, w, g = records(split)
        for scale in SCALES:
            t = pf.time_to_funded(r, w, rules(), sizer=pf.flat_size(scale),
                                  gap=g, n_paths=paths, horizon=horizon, seed=17)
            b = pf.block_bootstrap(r, w, rules(), sizer=pf.flat_size(scale),
                                   gap=g, n_paths=paths, horizon=horizon, seed=17)
            print(f"{split:>12}{scale:>6.1f}x{t['median_months']:>8.1f}"
                  f"{t['p75_days'] / 21:>8.1f}{t['p90_months']:>8.1f}"
                  f"{t['mean_attempts']:>10.1f}"
                  f"{t['mean_attempts'] * 539:>9,.0f}{b['pass_rate_resolved']:>9.0%}")
        print()


def section_control(splits, paths, horizon):
    """How much of the speed is the strategy and how much is the leverage."""
    print("The same days, demeaned. A positive-drift index levered five times")
    print("reaches +10% quickly whether or not the strategy adds anything, so")
    print("the control is what separates an edge from a lottery ticket.\n")
    print(f"{'split':>12}{'size':>7}{'P real':>9}{'P zero':>9}{'lift':>8}"
          f"{'med mo real':>13}{'med mo zero':>13}")
    print("-" * 71)
    for split in splits:
        r, w, g = records(split)
        for scale in SCALES:
            out = []
            for series in (r, r - r.mean()):
                b = pf.block_bootstrap(series, w, rules(), sizer=pf.flat_size(scale),
                                       gap=g, n_paths=paths, horizon=horizon, seed=11)
                t = pf.time_to_funded(series, w, rules(), sizer=pf.flat_size(scale),
                                      gap=g, n_paths=paths, horizon=horizon, seed=11)
                out.append((b["pass_rate_resolved"], t["median_months"]))
            (pr, mr), (pz, mz) = out
            print(f"{split:>12}{scale:>6.1f}x{pr:>9.0%}{pz:>9.0%}{pr - pz:>+8.0%}"
                  f"{mr:>13.1f}{mz:>13.1f}")
        print()


def section_slippage(splits, paths, horizon):
    """The breaker fires on roughly a quarter of days at these sizes."""
    print("A stop is a market order into a book that is moving, which is the")
    print("whole reason it was sent. Assuming it fills at its level is the")
    print("optimistic case; here it fills at up to three times worse.\n")
    print(f"{'split':>12}{'size':>7}{'slippage':>10}{'P(pass)':>9}{'med mo':>8}"
          f"{'p90 mo':>8}{'E[fees]':>9}")
    print("-" * 63)
    for split in splits:
        r, w, g = records(split)
        for scale in (3.0, 4.0):
            for slip in (0.002, 0.004, 0.006):
                rl = rules(slippage=slip)
                b = pf.block_bootstrap(r, w, rl, sizer=pf.flat_size(scale), gap=g,
                                       n_paths=paths, horizon=horizon, seed=5)
                t = pf.time_to_funded(r, w, rl, sizer=pf.flat_size(scale), gap=g,
                                      n_paths=paths, horizon=horizon, seed=5)
                print(f"{split:>12}{scale:>6.1f}x{slip:>10.1%}"
                      f"{b['pass_rate_resolved']:>9.0%}{t['median_months']:>8.1f}"
                      f"{t['p90_months']:>8.1f}{t['mean_attempts'] * 539:>9,.0f}")
            print()


def section_funded(splits):
    """The size that clears a challenge is not the size that keeps one."""
    print("A funded account runs the same 5% daily and 10% total floors, and it")
    print("is the asset rather than the hurdle. What matters here is not speed")
    print("but whether any day in four and a half years would have ended it.\n")
    print(f"{'split':>12}{'weekends':>10}{'size':>7}{'worst day':>11}"
          f"{'breaches':>10}{'gross %/yr':>12}{'your 80%':>12}")
    print("-" * 74)
    for split in splits:
        for flat in (False, True):
            r, w, g = records(split, flat_weekends=flat)
            for scale in (0.5, 1.0, 1.5):
                # what the breaker leaves behind, gap overshoot included
                overshoot = scale * g <= -BREAKER
                fired = scale * w <= -BREAKER
                stopped = np.where(overshoot, scale * g - 0.002, -(BREAKER + 0.002))
                day = np.where(fired, stopped, scale * r)
                worst = np.where(fired, stopped, scale * w)
                print(f"{split:>12}{'flat' if flat else 'held':>10}{scale:>6.1f}x"
                      f"{worst.min() * 100:>10.2f}%{int((worst <= -0.05).sum()):>10d}"
                      f"{day.mean() * 252 * 100:>11.2f}%"
                      f"{day.mean() * 252 * 0.8 * 100_000:>12,.0f}")
        print()


SECTIONS = {
    "ceiling": ("THE SIZE CEILING  -  set by the gap, not by the volatility",
                lambda a, s: section_ceiling(s)),
    "frontier": ("TIME TO FUNDED  -  counting the attempts that failed",
                 lambda a, s: section_frontier(s, a.paths, a.horizon)),
    "control": ("ZERO-DRIFT CONTROL  -  edge or lottery ticket",
                lambda a, s: section_control(s, a.paths, a.horizon)),
    "slippage": ("SLIPPAGE STRESS  -  if the breaker fills worse than assumed",
                 lambda a, s: section_slippage(s, a.paths, a.horizon)),
    "funded": ("KEEPING IT  -  what a funded account can carry",
               lambda a, s: section_funded(s)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true",
                    help="also read the held-out split (confirmation only)")
    ap.add_argument("--paths", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=2500)
    ap.add_argument("--section", choices=sorted(SECTIONS), action="append",
                    help="run only these sections (repeatable)")
    args = ap.parse_args()

    splits = ["dev", "validation"] + (["test"] if args.test else [])
    for name in (args.section or list(SECTIONS)):
        title, fn = SECTIONS[name]
        print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
        fn(args, splits)


if __name__ == "__main__":
    main()
