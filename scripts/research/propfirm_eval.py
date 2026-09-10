"""Size the USTEC overlay for a two-step prop-firm challenge.

The objective here is not Sharpe. It is P(reach +10%, then +5%, without ever
touching a 5% daily floor or a static 10% total floor), and the decision
variable is position size.

    python scripts/research/propfirm_eval.py                  # dev + validation
    python scripts/research/propfirm_eval.py --test           # add the held-out split
    python scripts/research/propfirm_eval.py --fee 539 --account 100000

The strategy itself is unchanged and was selected in an earlier study; nothing
is fitted here. What is chosen is a sizing rule, on dev and validation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import propfirm as pf  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.metrics import tearsheet_from_returns  # noqa: E402
from qlab.strategies import risk_managed_long as rml  # noqa: E402

SYMBOL = "USTEC"
# The strategy as validated, with one change made for the prop constraint and
# for one reason: the 2.0 cap lets a quiet market ask for leverage that can put
# a single day through the 5% floor. See the excursion table below.
PROP_CFG = replace(rml.RMLConfig(), max_weight=1.0)

SCALES = (0.4, 0.5, 0.6, 0.75, 1.0)


def records(split: str, cfg: rml.RMLConfig, allow_test: bool = False):
    frame = rml.run(SYMBOL, cfg, split=split, allow_test=allow_test,
                    cost=CostModel.from_profiles(SYMBOL, split=split))
    daily = pf.daily_records_from_minutes(
        SYMBOL, frame.select("nyd", "weight"), split=split, allow_test=allow_test)
    return daily, frame


def paths_of(daily):
    """The three arrays the engine walks: close, worst moment, unstoppable part."""
    return (daily["ret"].to_numpy(), daily["worst"].to_numpy(),
            daily["gap"].to_numpy())


def section_excursions(splits):
    """Why the strategy's own leverage cap comes down for a challenge."""
    print("A challenge watches floating equity, so what matters is not the")
    print("worst CLOSE but the worst moment. At the shipped 2.0x cap a single")
    print("day already reaches the 5% floor unaided.\n")
    print(f"{'max_weight':>12}" + "".join(f"{s:>34}" for s in splits))
    for mw in (2.0, 1.5, 1.0, 0.75):
        cells = []
        for split in splits:
            d, _ = records(split, replace(rml.RMLConfig(), max_weight=mw),
                           allow_test=(split == "test"))
            w = d["worst"].to_numpy()
            cells.append(f"min {w.min()*100:6.2f}%  1st pct {np.quantile(w,0.01)*100:6.2f}%")
        print(f"{mw:>12.2f}" + "".join(f"{c:>34}" for c in cells))


def section_frontier(splits, n_paths, horizon):
    """Pass probability against time, flat sizing versus cushion sizing."""
    print("P(pass|resolved) is the number that matters when the programme has no")
    print("deadline: of the runs that finished one way or the other, how many")
    print("finished by passing. Paths still running at the horizon are censored,")
    print("not failed.\n")
    print(f"{'sizing':>18}{'split':>12}{'P(pass|res)':>13}{'median days':>13}"
          f"{'p90 days':>10}{'med yrs':>9}  failure modes")
    print("-" * 108)
    out = {}
    for maker, name in ((pf.flat_size, "flat"), (pf.cushion_size, "cushion")):
        for scale in SCALES:
            for split in splits:
                d, _ = records(split, PROP_CFG, allow_test=(split == "test"))
                r, w, g = paths_of(d)
                b = pf.block_bootstrap(
                    r, w, sizer=maker(scale), gap=g,
                    n_paths=n_paths, horizon=horizon, seed=13)
                out[(name, scale, split)] = b
                fails = {k: v for k, v in b["outcomes"].items()
                         if k not in (pf.PASSED, pf.INCOMPLETE)}
                fs = ", ".join(f"{k} {v/b['n_paths']:.0%}"
                               for k, v in sorted(fails.items(), key=lambda x: -x[1]))
                print(f"{name + ' ' + format(scale, '.2f') + 'x':>18}{split:>12}"
                      f"{b['pass_rate_resolved']:>13.1%}"
                      f"{b['median_days_when_passed']:>13.0f}"
                      f"{b['p90_days_when_passed']:>10.0f}"
                      f"{b['median_days_when_passed']/252:>9.1f}  {fs or 'none'}")
            print()
    return out


def section_economics(out, splits, fee, account):
    """What the frontier is worth against the fee."""
    print(f"Fee {fee:,.0f} on a {account:,.0f} account "
          f"({fee/account:.2%}), refunded on the first payout.\n")
    print(f"{'sizing':>18}{'split':>12}{'P(pass)':>9}{'E[fees]':>10}"
          f"{'med yrs':>9}{'funded/yr gross':>17}{'your 80% share':>16}")
    print("-" * 92)
    for scale in SCALES:
        for split in splits:
            b = out.get(("cushion", scale, split))
            if b is None:
                continue
            p = b["pass_rate_resolved"]
            d, frame = records(split, PROP_CFG, allow_test=(split == "test"))
            sheet = tearsheet_from_returns(frame["net_bps"])
            # Funded phase runs the same sizing; the average cushion multiple on
            # a funded account is close to 1.0 by construction at the start.
            gross = sheet["cagr_pct"] * scale
            print(f"{'cushion ' + format(scale, '.2f') + 'x':>18}{split:>12}"
                  f"{p:>9.1%}{fee/max(p, 1e-9):>10,.0f}"
                  f"{b['median_days_when_passed']/252:>9.1f}"
                  f"{gross:>16.1f}%{gross*0.8/100*account:>15,.0f}")
        print()
    print("The fee column is the expected total spend on entries before one")
    print("sticks. The right-hand columns assume the strategy keeps working,")
    print("which is the assumption doing all the work here.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true",
                    help="also read the held-out split (confirmation only; the "
                         "sizing is chosen on dev and validation)")
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--fee", type=float, default=539.0)
    ap.add_argument("--account", type=float, default=100_000.0)
    args = ap.parse_args()

    splits = ["dev", "validation"] + (["test"] if args.test else [])

    print(f"\n{'=' * 108}\nEXCURSIONS  -  why the leverage cap comes down\n{'=' * 108}")
    section_excursions(splits)

    print(f"\n{'=' * 108}\nFRONTIER  -  pass probability against time\n{'=' * 108}")
    out = section_frontier(splits, args.paths, args.horizon)

    print(f"\n{'=' * 108}\nECONOMICS\n{'=' * 108}")
    section_economics(out, splits, args.fee, args.account)


if __name__ == "__main__":
    main()
