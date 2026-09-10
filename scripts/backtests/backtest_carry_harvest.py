"""Evaluate the USDJPY carry-harvest overlay.

Same machinery as the USTEC study - `risk_managed_long` with the FX session
convention - but a different diagnosis, and the candidate set reflects it. On
USTEC the sizing rule SHRINKS the position, because the index runs at 27%
volatility against a 15% target. On USDJPY at 9.6% it would LEVER, and that
made the drawdown worse than simply holding. So the cap comes down to 1.0 and
the target down to 8%, and the question becomes which gate protects the one
risk this instrument actually has: the carry unwind.

    python scripts/backtests/backtest_carry_harvest.py
    python scripts/backtests/backtest_carry_harvest.py --swap-credit 1.47
    python scripts/backtests/backtest_carry_harvest.py --test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.metrics import format_returns_tearsheet, tearsheet_from_returns  # noqa: E402
from qlab.rollover import basis  # noqa: E402
from qlab.strategies import risk_managed_long as rml  # noqa: E402

SYMBOL = "USDJPY"
RISK_SYMBOL = "USTEC"

# The candidate set, fixed before any of it was evaluated. Each is a different
# answer to "what protects a carry position", not a point on a grid.
BASE = rml.RMLConfig(ma_days=200, vol_days=20, target_vol_pct=8.0,
                     max_weight=1.0, gate_floor=0.0, band=0.10)
CANDIDATES: dict[str, dict] = {
    "J1 yen trend gate":        dict(cfg=replace(BASE, target_vol_pct=1e9), risk_gate=False),
    "J2 J1 + vol target":       dict(cfg=BASE, risk_gate=False),
    "J3 J2 + equity risk gate": dict(cfg=BASE, risk_gate=True),
    # gate_floor=1.0 is how the yen trend gate is switched OFF: the weight
    # multiplier becomes 1 whether price is above its average or not.
    "J4 equity risk gate only": dict(cfg=replace(BASE, target_vol_pct=1e9,
                                                 gate_floor=1.0), risk_gate=True),
    "J6 vol target, no gate":   dict(cfg=replace(BASE, gate_floor=1.0), risk_gate=False),
}

# The two occasions in this corpus when the carry trade actually unwound.
UNWINDS = (
    ("COVID, Mar 2020", "dev", date(2020, 2, 15), date(2020, 4, 15)),
    ("yen unwind, Aug 2024", "validation", date(2024, 7, 1), date(2024, 9, 15)),
)


def equity_risk_on(split: str, allow_test: bool) -> pl.DataFrame:
    """Is the equity market above its own 200-session average?

    Computed on USTEC's own calendar and only then joined. Computing it after a
    join to the FX calendar would run the window over a series holed by US
    holidays, and a rolling mean over a window containing a null is null - which
    silently switches the gate off on nearly every day.
    """
    from datetime import timedelta
    u = rml.cash_session_panel(RISK_SYMBOL, split=split, allow_test=allow_test,
                               warmup=timedelta(days=400), session=rml.US_CASH)
    u = u.with_columns(uma=pl.col("c").rolling_mean(200))
    return u.select("nyd", risk_on=pl.col("c") > pl.col("uma"))


def run_candidate(split: str, cfg: rml.RMLConfig, risk_gate: bool, *,
                  allow_test: bool, swap_credit: float) -> pl.DataFrame:
    """One candidate. ``swap_credit`` is bps a night RECEIVED on a long."""
    cost = CostModel.from_profiles(SYMBOL, split=split)
    frame = rml.run(SYMBOL, replace(cfg, swap_bps_per_night=-swap_credit),
                    split=split, allow_test=allow_test, cost=cost,
                    session=rml.FX_DAY)
    if risk_gate:
        gate = equity_risk_on(split, allow_test)
        frame = (frame.join(gate, on="nyd", how="left")
                      .with_columns(pl.col("risk_on").forward_fill().fill_null(False)))
        # Re-derive the weight, then the cost of the turnover it implies.
        w = frame["weight"].to_numpy() * frame["risk_on"].to_numpy()
        turn = np.abs(np.diff(np.concatenate([[0.0], w])))
        frame = frame.with_columns(
            weight=pl.Series(w), turnover=pl.Series(turn),
            cost_bps=pl.Series(frame["side_cost_bps"].to_numpy() * turn),
            swap_bps=pl.Series(-swap_credit * w * frame["nights"].fill_null(1).to_numpy()),
        ).with_columns(
            net_bps=pl.col("weight") * pl.col("gross_bps")
            - pl.col("cost_bps") - pl.col("swap_bps"))
    return frame


def sheet(frame: pl.DataFrame, label: str, buy_hold=False, swap_credit=0.0) -> dict:
    if buy_hold:
        nights = frame["nights"].fill_null(1).to_numpy()
        r = frame["gross_bps"].to_numpy() + swap_credit * nights
        s = tearsheet_from_returns(r, label=label)
        s["avg_w"] = 1.0
        s["turnover_per_year"] = 0.0
        return s
    s = tearsheet_from_returns(frame["net_bps"], label=label,
                               turnover=frame["turnover"])
    s["avg_w"] = float(frame["weight"].mean())
    return s


def table(sheets: list[dict]) -> str:
    head = (f"{'strategy':<27}{'n':>6}{'CAGR%':>8}{'vol%':>7}{'SR':>6}"
            f"{'maxDD%':>9}{'Calmar':>8}{'turn/yr':>9}{'avg w':>7}")
    out = [head, "-" * len(head)]
    for s in sheets:
        out.append(
            f"{s['label']:<27}{s['periods']:>6}{s['cagr_pct']:>8.2f}"
            f"{s['ann_vol_pct']:>7.2f}{s['sharpe']:>6.2f}{s['max_dd_pct']:>9.1f}"
            f"{s['calmar']:>8.2f}{s.get('turnover_per_year', 0.0):>9.1f}"
            f"{s.get('avg_w', float('nan')):>7.2f}")
    return "\n".join(out)


def unwind_report(swap_credit: float, allow_test: bool) -> str:
    lines = []
    for name, split, lo, hi in UNWINDS:
        lines.append(f"  {name}")
        for label, kw in CANDIDATES.items():
            f = run_candidate(split, allow_test=False, swap_credit=swap_credit, **kw)
            m = ((f["nyd"] >= lo) & (f["nyd"] <= hi)).to_numpy()
            r = f["net_bps"].to_numpy()[m]
            r = r[np.isfinite(r)]
            if r.size == 0:
                continue
            eq = np.cumprod(1 + r / 1e4)
            dd = float((eq / np.maximum.accumulate(eq) - 1).min()) * 100
            lines.append(f"    {label:<27} return {100*(eq[-1]-1):+6.2f}%   "
                         f"worst drawdown {dd:6.2f}%")
        f = run_candidate(split, allow_test=False, swap_credit=swap_credit,
                          **CANDIDATES["J2 J1 + vol target"])
        m = ((f["nyd"] >= lo) & (f["nyd"] <= hi)).to_numpy()
        b = f["gross_bps"].to_numpy()[m]
        b = b[np.isfinite(b)]
        eq = np.cumprod(1 + b / 1e4)
        dd = float((eq / np.maximum.accumulate(eq) - 1).min()) * 100
        lines.append(f"    {'J0 buy and hold':<27} return {100*(eq[-1]-1):+6.2f}%   "
                     f"worst drawdown {dd:6.2f}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swap-credit", type=float, default=0.0,
                    help="bps of notional per night RECEIVED on a long (0 = "
                         "conservative; the measured price basis is 1.47)")
    ap.add_argument("--test", action="store_true",
                    help="also run the held-out test split (once)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    splits = ["dev", "validation"] + (["test"] if args.test else [])
    payload = {"symbol": SYMBOL, "swap_credit_bps_per_night": args.swap_credit,
               "candidates": {k: asdict(v["cfg"]) | {"risk_gate": v["risk_gate"]}
                              for k, v in CANDIDATES.items()},
               "splits": {}}

    for split in splits:
        allow = split == "test"
        sheets = []
        kept = None
        for i, (label, kw) in enumerate(CANDIDATES.items()):
            f = run_candidate(split, allow_test=allow, swap_credit=args.swap_credit, **kw)
            if i == 0:
                sheets.append(sheet(f, "J0 buy and hold", buy_hold=True,
                                    swap_credit=args.swap_credit))
            sheets.append(sheet(f, label))
            if label.startswith("J2"):
                kept = f
        print(f"\n{'=' * 96}\n{split.upper()}"
              f"{'   [HELD OUT - one run]' if allow else ''}\n{'=' * 96}")
        print(table(sheets))
        if kept is not None:
            print("\n" + format_returns_tearsheet(
                tearsheet_from_returns(kept["net_bps"], label="J2 (detail)",
                                       turnover=kept["turnover"])))
            by_year = (kept.with_columns(y=pl.col("nyd").dt.year())
                       .group_by("y").agg(
                           n=pl.len(),
                           strat=((1 + pl.col("net_bps") / 1e4).product() - 1) * 100,
                           bh=((1 + pl.col("gross_bps") / 1e4).product() - 1) * 100,
                           w=pl.col("weight").mean()).sort("y"))
            print("\n  by calendar year")
            for r in by_year.iter_rows(named=True):
                print(f"    {r['y']}  n={r['n']:>4}  strategy {r['strat']:+7.2f}%   "
                      f"buy&hold {r['bh']:+7.2f}%   avg w {r['w']:.2f}")
        payload["splits"][split] = sheets

    print(f"\n{'=' * 96}\nThe two carry unwinds\n{'=' * 96}")
    print(unwind_report(args.swap_credit, allow_test=False))

    print(f"\n{'=' * 96}\nThe swap, which on USDJPY runs the other way\n{'=' * 96}")
    for split in ("dev", "validation"):
        b = basis(SYMBOL, split=split)
        print(f"  {split:<11} overnight price drift {b.mid_bps:+.3f} bps/night "
              f"-> {b.mid_bps * 260 / 100:+.2f}%/yr of price drag on a long,"
              f" offset by a swap CREDIT of up to the same size")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
