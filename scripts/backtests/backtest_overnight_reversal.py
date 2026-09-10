#!/usr/bin/env python
"""The overnight-intraday reversal family (Liu, Liu, Wang, Zhou and Zhu, 2025).

Sections, each runnable on its own with ``--only``:

  variants     the four strategies of A.3 under all three weightings, gross and net
  legs         A.6, what the long and the short side each contributed
  weekly       A.4, the Monday-open to Friday-close frequency
  dispersion   A.9(a) and A.9(b), the mechanism regressions
  splits       A.9(c), conditioning on dispersion and on a realised-vol proxy
  placebo      the cross-sectionally shuffled signal, and A.9(d)'s AB_NR check
  subperiods   A.10.2, the sample halved
  validation   the same table on the validation split

The test split is not read. Nothing in this study produced a candidate worth
spending it on, and ``--test`` therefore exists only so that a future run which
does have one can be explicit about it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.metrics import format_returns_tearsheet  # noqa: E402
from qlab.stats import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    ljung_box,
    newey_west,
    sharpe_with_se,
)
from qlab.strategies import overnight_reversal as ovr  # noqa: E402

WEIGHTINGS = ("demean", "rank", "vol_scaled")
SECTIONS = ("variants", "legs", "weekly", "dispersion", "splits",
            "placebo", "subperiods", "validation")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load(split: str, *, allow_test: bool = False):
    """Panel, cross-section and cost matrix for one split."""
    panel = ovr.with_returns(
        ovr.session_panel(split=split, allow_test=allow_test))
    cs = ovr.cross_section(panel)
    cost = ovr.round_turn_bps_matrix(cs, split=split)
    return panel, cs, cost


# --------------------------------------------------------------------------

def section_variants(cs, cost, *, title: str = "1. The four variants (A.3, A.5)") -> dict:
    rule(title)
    print(f"  {cs.n_days} complete sessions, {cs.n_assets} instruments: "
          f"{', '.join(cs.symbols)}")
    print(f"  mean round-turn cost, bps of price: "
          + ", ".join(f"{s} {c:.2f}" for s, c in
                      zip(cs.symbols, cost.mean(axis=0))))
    print("\n  Mean daily return in bps of capital, with Newey-West t-statistics.")
    print("  'net' charges a full round turn every day for the two variants that")
    print("  are flat outside the session, and one side per unit of weight change")
    print("  for the two that hold continuously.\n")

    header = (f"  {'weighting':<11} {'variant':<7} {'gross':>8} {'t':>6} "
              f"{'net':>8} {'t':>6} {'Sharpe':>7} {'t(SR)':>6} {'turn':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    runs: dict[tuple[str, str], ovr.RunResult] = {}
    for weighting in WEIGHTINGS:
        for name in ovr.CORE_VARIANTS:
            run = ovr.run_variant(cs, name, weighting=weighting, cost_bps=cost)
            runs[(weighting, name)] = run
            g, n = run.hac(net=False), run.hac(net=True)
            sr = sharpe_with_se(run.net_bps)
            print(f"  {weighting:<11} {name:<7} {g.mean:>+8.2f} {g.t_stat:>+6.2f} "
                  f"{n.mean:>+8.2f} {n.t_stat:>+6.2f} {sr.mean:>+7.2f} "
                  f"{sr.t_stat:>+6.2f} {np.nanmean(run.turnover):>6.2f}")
        print()

    print("  The two variants the paper puts in its appendix and warns against,")
    print("  because both hold the overnight window and would have to be executed")
    print("  into the thinnest hours of the day:\n")
    for name in ("CO-CO", "OC-CO"):
        run = ovr.run_variant(cs, name, weighting="rank", cost_bps=cost)
        g, n = run.hac(net=False), run.hac(net=True)
        print(f"  {'rank':<11} {name:<7} {g.mean:>+8.2f} {g.t_stat:>+6.2f} "
              f"{n.mean:>+8.2f} {n.t_stat:>+6.2f}")

    print("\n  Correlation of CO-OC with the other three (rank weighting, net):")
    base = runs[("rank", "CO-OC")].net_bps
    for name in ("CC-CC", "OO-OO", "OC-OC"):
        other = runs[("rank", name)].net_bps
        good = np.isfinite(base) & np.isfinite(other)
        print(f"    CO-OC vs {name:<6} {np.corrcoef(base[good], other[good])[0, 1]:+.3f}")

    print("\n  Full tearsheet, the paper's primary strategy, rank weighting, net of cost:")
    print(format_returns_tearsheet(runs[("rank", "CO-OC")].tearsheet()))
    print("\n  And the traditional reversal it is supposed to dominate:")
    print(format_returns_tearsheet(runs[("rank", "CC-CC")].tearsheet()))

    print("\n  Residual serial dependence in the CO-OC net series:")
    print(f"    {ljung_box(runs[('rank', 'CO-OC')].net_bps, 10)}")

    # A.5's family of four is a family, so the multiplicity is real.
    print("\n  Multiple-testing correction over the family of four (rank, net):")
    names = list(ovr.CORE_VARIANTS)
    raw = np.array([runs[("rank", n)].hac().p_value for n in names])
    bonf, bh = bonferroni(raw), benjamini_hochberg(raw)
    print(f"    {'variant':<8} {'p':>8} {'Bonferroni':>11} {'BH':>8}")
    for name, p, b, q in zip(names, raw, bonf, bh):
        print(f"    {name:<8} {p:>8.4f} {b:>11.4f} {q:>8.4f}")
    return runs


def section_legs(runs) -> None:
    rule("2. Long and short legs (A.6)")
    print("  Mean return per dollar committed to each side, in bps. If the effect")
    print("  is price pressure being unwound, both legs should contribute.\n")
    print(f"  {'variant':<8} {'long':>8} {'t':>6} {'short':>8} {'t':>6}")
    print("  " + "-" * 40)
    for name in ovr.CORE_VARIANTS:
        run = runs[("rank", name)]
        long_h = newey_west(run.long_leg_bps)
        short_h = newey_west(run.short_leg_bps)
        print(f"  {name:<8} {long_h.mean:>+8.2f} {long_h.t_stat:>+6.2f} "
              f"{short_h.mean:>+8.2f} {short_h.t_stat:>+6.2f}")
    print("\n  The short leg is quoted as the return of the shorted basket, so a")
    print("  profitable short is a *negative* number here.")


def section_weekly(cs, split: str) -> None:
    rule("3. Weekly frequency (A.4)")
    weekly = ovr.weekly_cross_section(cs)
    cost = ovr.round_turn_bps_matrix(weekly, split=split)
    print(f"  {weekly.n_days} weeks. Intraday is Monday's open to Friday's close;")
    print("  overnight is the previous Friday's close to Monday's open. A week")
    print("  with fewer than three complete sessions is dropped, not patched.\n")
    print(f"  {'weighting':<11} {'variant':<7} {'gross':>8} {'t':>6} "
          f"{'net':>8} {'t':>6} {'Sharpe':>7}")
    print("  " + "-" * 58)
    for weighting in ("rank", "demean"):
        for name in ovr.CORE_VARIANTS:
            try:
                run = ovr.run_variant(weekly, name, weighting=weighting,
                                      cost_bps=cost)
            except (ValueError, KeyError) as exc:
                print(f"  {weighting:<11} {name:<7}  not computable: {exc}")
                continue
            g, n = run.hac(net=False), run.hac(net=True)
            sr = sharpe_with_se(run.net_bps, periods_per_year=52)
            print(f"  {weighting:<11} {name:<7} {g.mean:>+8.2f} {g.t_stat:>+6.2f} "
                  f"{n.mean:>+8.2f} {n.t_stat:>+6.2f} {sr.mean:>+7.2f}")
        print()


def section_dispersion(cs, runs) -> None:
    rule("4. The mechanism: overnight dispersion (A.9a, A.9b)")
    disp = ovr.overnight_dispersion(cs, scaled=True)
    disp_raw = ovr.overnight_dispersion(cs, scaled=False)
    vol = ovr.realized_vol_proxy(cs)
    print("  Dispersion is the cross-sectional standard deviation of the overnight")
    print("  return, computed on volatility-standardised returns so that it is not")
    print("  simply a restatement of whichever instrument is the most volatile.")
    print("  How far it succeeds at that, as correlation with each instrument's own")
    print("  absolute overnight move:\n")
    print(f"    {'instrument':<10} {'raw':>8} {'standardised':>14}")
    for j, symbol in enumerate(cs.symbols):
        move = np.abs(cs.returns["r_co"][:, j])
        print(f"    {symbol:<10} {_corr(disp_raw, move):>+8.3f} "
              f"{_corr(disp, move):>+14.3f}")
    print("\n  Both columns still lean on the two volatile legs. On a four-asset")
    print("  panel there is no construction that does not: with N = 4 a cross-")
    print("  sectional standard deviation has three degrees of freedom, and this")
    print("  is one of the places the narrow universe genuinely binds.\n")

    for name in ("CO-OC", "OC-OC"):
        run = runs[("rank", name)]
        print(f"  --- {name} on lagged dispersion (the paper predicts a1 > 0) ---")
        print(ovr.dispersion_regression(run, disp).table())
        print(f"  --- {name}, adding the realised-volatility proxy (a VIX stand-in) ---")
        print(ovr.dispersion_regression(run, disp, vix_proxy=vol).table())
        print()

    print("  --- A.9(b): the two-step conditional Sharpe, CO-OC ---")
    try:
        steps = ovr.conditional_sharpe_regression(runs[("rank", "CO-OC")], disp)
        print(f"  kappa = {steps['kappa']:.4f}")
        print("  step 1, conditional volatility:")
        print(steps["vol"].table())
        print("  step 2, the risk-adjusted return:")
        print(steps["sharpe"].table())
    except ValueError as exc:
        print(f"  not computable: {exc}")
    return disp, vol


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[good], b[good])[0, 1])


def section_splits(cs, runs, disp, vol) -> None:
    rule("5. Conditioning splits (A.9c)")
    print("  Mean net return in bps above and below each conditioner's median,")
    print("  with the conditioner lagged one day.\n")
    print(f"  {'variant':<8} {'conditioner':<16} {'high':>8} {'t':>6} "
          f"{'low':>8} {'t':>6} {'diff':>8} {'t':>6}")
    print("  " + "-" * 72)
    conditioners = [("dispersion", disp), ("vol proxy", vol)]
    for name in ("CO-OC", "OC-OC", "CC-CC"):
        run = runs[("rank", name)]
        for label, series in conditioners:
            out = ovr.conditional_split(run, series, label=label)
            if "high_mean" not in out:
                print(f"  {name:<8} {label:<16}  too few observations")
                continue
            print(f"  {name:<8} {label:<16} {out['high_mean']:>+8.2f} "
                  f"{out['high_t']:>+6.2f} {out['low_mean']:>+8.2f} "
                  f"{out['low_t']:>+6.2f} {out['diff']:>+8.2f} {out['diff_t']:>+6.2f}")
    print("\n  The volatility proxy is realised, not implied. It is not the VIX and")
    print("  no result here should be read as a statement about the VIX.")


def section_placebo(cs, cost, runs) -> None:
    rule("6. Placebos")
    print("  (a) The same construction on signals permuted across instruments")
    print("      within each day. Same marginals, same turnover, same cost, no")
    print("      cross-sectional information. Ten seeds.\n")
    print(f"  {'variant':<8} {'real net':>9} {'t':>6} {'placebo mean':>13} "
          f"{'sd':>7} {'seeds > real':>13}")
    print("  " + "-" * 62)
    for name in ovr.CORE_VARIANTS:
        real = runs[("rank", name)].hac()
        draws = np.array([
            ovr.run_variant(cs, name, weighting="rank", cost_bps=cost,
                            shuffle_seed=seed).hac().mean
            for seed in range(10)
        ])
        beat = int((draws >= real.mean).sum())
        print(f"  {name:<8} {real.mean:>+9.2f} {real.t_stat:>+6.2f} "
              f"{draws.mean():>+13.2f} {draws.std(ddof=1):>7.2f} {beat:>13d}")

    print("\n  (b) A.9(d), the investor-heterogeneity check. The paper expects this")
    print("      to carry nothing in futures, and uses that to rule out the")
    print("      equity-market tug-of-war story as the driver of CO-OC.\n")
    print(f"  {'signal':<10} {'interval':>8} {'mean':>9} {'t':>6} {'periods':>8}")
    print("  " + "-" * 46)
    for interval in (10, 20, 30):
        for negative in (True, False):
            label = "AB_NR" if negative else "AB_PR"
            signal = ovr.abnormal_reversal_signal(cs, interval=interval,
                                                  negative=negative)
            try:
                run = ovr.run_signal_matrix(cs, signal, rebalance=interval,
                                            cost_bps=cost, label=label)
            except ValueError as exc:
                print(f"  {label:<10} {interval:>8}  not computable: {exc}")
                continue
            h = run.hac()
            print(f"  {label:<10} {interval:>8} {h.mean:>+9.2f} "
                  f"{h.t_stat:>+6.2f} {h.n:>8d}")


def section_subperiods(cs, cost) -> None:
    rule("7. Subperiods (A.10.2)")
    half = cs.n_days // 2
    print(f"  Sample split at {cs.dates[half]}: "
          f"{cs.dates[0]}..{cs.dates[half - 1]} against "
          f"{cs.dates[half]}..{cs.dates[-1]}\n")
    print(f"  {'variant':<8} {'first net':>10} {'t':>6} {'second net':>11} {'t':>6}")
    print("  " + "-" * 46)
    for name in ovr.CORE_VARIANTS:
        run = ovr.run_variant(cs, name, weighting="rank", cost_bps=cost)
        first = newey_west(run.net_bps[:half])
        second = newey_west(run.net_bps[half:])
        print(f"  {name:<8} {first.mean:>+10.2f} {first.t_stat:>+6.2f} "
              f"{second.mean:>+11.2f} {second.t_stat:>+6.2f}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=None,
                    help="run only these sections")
    ap.add_argument("--split", default="dev", help="split for the main sections")
    ap.add_argument("--test", action="store_true",
                    help="unlock the held-out split (do not use without a survivor)")
    args = ap.parse_args()
    wanted = set(args.only or SECTIONS)

    if args.test and args.split != "test":
        print("--test only means anything with --split test", file=sys.stderr)
        return 2

    print(f"Overnight-intraday reversal, split={args.split}")
    _panel, cs, cost = load(args.split, allow_test=args.test)

    runs = None
    disp = vol = None
    if wanted & {"variants", "legs", "dispersion", "splits", "placebo"}:
        runs = section_variants(cs, cost)
    if "legs" in wanted:
        section_legs(runs)
    if "weekly" in wanted:
        section_weekly(cs, args.split)
    if wanted & {"dispersion", "splits"}:
        disp, vol = section_dispersion(cs, runs)
    if "splits" in wanted:
        section_splits(cs, runs, disp, vol)
    if "placebo" in wanted:
        section_placebo(cs, cost, runs)
    if "subperiods" in wanted:
        section_subperiods(cs, cost)

    if "validation" in wanted and args.split == "dev":
        _, vcs, vcost = load("validation")
        section_variants(vcs, vcost,
                         title="8. The validation split, same table")
        print("\n  The test split is deliberately not read. No variant survived dev")
        print("  with a t-statistic worth spending it on.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
