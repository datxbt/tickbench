"""Test the macro-news breakout of Chen (2025) on the USTEC tape.

The paper's rule: two minutes after a scheduled high-impact US release, a
Kalman-filtered Trading Indicator decides a side; a stop order goes at the
200-minute extreme on that side; a 1% stop-loss sized to 1.5% of equity
protects it; everything is flat at 15:50 ET. Backtested 2018-2025 it reports
730% total return, a 1.67 Sharpe and a -17.4% maximum drawdown, and an
LLM sentiment filter is said to lift that to 983% and 2.11.

This corpus has no economic calendar in it, so the calendar is reconstructed
from publication rules - see :mod:`qlab.macro_calendar` - and audited against
the tape before a single trade is taken. The script runs six passes, ordered so
the claim fails as early as it is going to:

1. **Is the calendar real?** Median range in each release minute against the
   same clock minute on every other day. A rule that does not land on releases
   makes everything downstream meaningless, so this comes first.
2. **What the indicator actually is.** The steady-state Kalman gain, and the
   price deviation each threshold implies. The paper does not state either.
3. **The rule exactly as written**, tick-resolved, on dev and validation:
   full calendar, and the exact-date tier alone.
4. **The control.** The same mechanics on non-event days at the same clock
   minutes. If this earns what the calendar earns, the news is decoration.
5. **The threshold reading.** Table 1 lists "30/70" and the body describes a
   buffer zone; the two readings are incompatible. Both are run, plus three
   more, so the result can be read off a surface rather than a point.
6. **Where the money went.** Direction, exit reason, event, year, and the
   edge measured at mid so a cost problem and a signal problem can be told
   apart.

Run:  python scripts/backtests/backtest_news_breakout.py
      python scripts/backtests/backtest_news_breakout.py --splits dev validation
      python scripts/backtests/backtest_news_breakout.py --sweep
      python scripts/backtests/backtest_news_breakout.py --splits test   # locked
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.loader import SPLITS, load_bars  # noqa: E402
from qlab.macro_calendar import build_calendar, validate  # noqa: E402
from qlab.metrics import format_tearsheet, tearsheet  # noqa: E402
from qlab.stats import bootstrap_ci  # noqa: E402
from qlab.strategies.news_breakout import (  # noqa: E402
    DEFAULT,
    Params,
    run,
    steady_state_gain,
    sweep_thresholds,
    trading_indicator,
)

OUT = paths.STRATEGY_REPORT_DIR


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def audit_calendar(allow_test: bool) -> pl.DataFrame:
    rule("1. Is the reconstructed calendar landing on real releases?")
    # The audit stops at the validation boundary unless the test split has been
    # deliberately unlocked. It informs no parameter, but a locked split is
    # locked for looking at as well as for trading.
    last = SPLITS["test"].end if allow_test else SPLITS["validation"].end
    bars = load_bars(
        "USTEC", "1m", start=date(2020, 2, 1), end=last,
        columns=["ts", "ts_open", "high", "low"], allow_test=allow_test,
    )
    calendar = build_calendar(bars["ts"].min().date(), bars["ts"].max().date())
    report = validate(calendar, bars)
    print(
        "Median 1-minute range in the release minute, against the same clock\n"
        "minute on every day the rule did not name. A ratio near 1 means the\n"
        "rule is missing the release more often than it hits it.\n"
    )
    print(report.to_pandas().round(2).to_string(index=False))
    return report


def describe_indicator() -> None:
    rule("2. What the Trading Indicator is, once the parameters are substituted")
    gain = steady_state_gain(DEFAULT.q_noise, DEFAULT.r_noise)
    print(
        f"Q = {DEFAULT.q_noise}, R = {DEFAULT.r_noise} converge to a constant Kalman\n"
        f"gain of {gain:.5f}. A constant-gain filter on a random walk is an EMA, so\n"
        f"p_hat is an EMA of period ~{1 / gain:.0f} minutes and the filter contributes\n"
        "no state estimation beyond it.\n"
    )
    print("  TI     implied deviation of price from that EMA")
    for threshold in (20, 30, 40, 50, 60, 70, 80):
        import math

        delta = -math.log2(100.0 / threshold - 1.0) / DEFAULT.scale_factor
        print(f"  {threshold:>3}    {100 * delta:+7.3f}%   ({15000 * delta:+7.1f} pts at 15,000)")
    check = trading_indicator(1.00244, 1.0, DEFAULT.scale_factor)
    print(f"\n  sanity: price 0.244% above trend gives TI = {check:.1f}")


def backtest(splits: list[str], allow_test: bool) -> dict:
    rule("3-4. The rule as written, and the control")
    results = {}
    for split in splits:
        for label, kwargs in (
            ("full calendar", dict(tiers="all")),
            ("exact dates only", dict(tiers="exact")),
            ("CONTROL (no news)", dict(tiers="all", control=True)),
        ):
            result = run(split=split, allow_test=allow_test, **kwargs)
            key = f"{split} / {label}"
            sheet = tearsheet(
                result.trades, starting_equity=result.starting_equity, label=key
            )
            results[key] = (result, sheet)
            print(
                f"\n[{key}]  {result.events} events -> {result.signals} signals "
                f"-> {sheet.get('trades', 0)} trades"
            )
            print(format_tearsheet(sheet))
            if not result.trades.is_empty():
                ci = bootstrap_ci(result.trades["net_usd"].to_numpy())
                print(
                    f"  mean PnL/trade   ${ci.point:,.2f}  "
                    f"[{ci.lo:,.2f}, {ci.hi:,.2f}]  p={ci.p_value:.3f}"
                )
                stem = f"newsbreakout_{split}_{label.split()[0].lower()}"
                result.trades.write_parquet(OUT / f"{stem}.parquet")
    return results


def threshold_surface(splits: list[str], allow_test: bool) -> None:
    rule("5. The threshold reading - Table 1 says 30/70, the body says otherwise")
    print(
        "Row 1 is Table 1 read literally (buy above 30, sell below 70), which\n"
        "arms an order on almost every event. Row 4 is the reading the body's\n"
        "'buffer zone' implies and the one used everywhere else.\n"
    )
    for split in splits:
        table = sweep_thresholds(split=split, tiers="all", allow_test=allow_test)
        print(f"\n[{split}]")
        print(table.to_pandas().round(3).to_string(index=False))
        table.write_parquet(OUT / f"newsbreakout_sweep_{split}.parquet")


def attribution(results: dict) -> None:
    rule("6. Where the money went")
    for key, (result, sheet) in results.items():
        trades = result.trades
        if trades.is_empty():
            continue
        print(f"\n[{key}]")
        print(
            "  edge at mid, before any cost:  "
            f"${sheet['mid_edge_per_trade']:,.2f}/trade   "
            f"all-in cost ${sheet['all_in_cost_per_trade']:,.2f}/trade"
        )
        for column in ("direction", "exit_reason", "event"):
            if column not in trades.columns:
                continue
            table = (
                trades.group_by(column)
                .agg(
                    n=pl.len(),
                    mid_per_trade=pl.col("mid_pnl_usd").mean().round(2),
                    net_total=pl.col("net_usd").sum().round(0),
                )
                .sort("n", descending=True)
            )
            print(f"\n  by {column}:")
            print("   " + table.to_pandas().to_string(index=False).replace("\n", "\n   "))
        yearly = (
            trades.with_columns(
                year=pl.from_epoch(pl.col("entry_ts"), time_unit="us").dt.year()
            )
            .group_by("year")
            .agg(n=pl.len(), net=pl.col("net_usd").sum().round(0))
            .sort("year")
        )
        print("\n  by year:")
        print("   " + yearly.to_pandas().to_string(index=False).replace("\n", "\n   "))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "validation"])
    parser.add_argument("--sweep", action="store_true", help="threshold surface only")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    unknown = set(args.splits) - set(SPLITS)
    if unknown:
        parser.error(f"unknown split(s): {', '.join(sorted(unknown))}")
    allow_test = "test" in args.splits
    if allow_test:
        print(
            "!! the test split is unlocked. It is one shot: a strategy rejected\n"
            "!! on dev has no business being measured here.\n"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    audit_calendar(allow_test)
    if args.audit_only:
        return 0
    describe_indicator()
    if args.sweep:
        threshold_surface(args.splits, allow_test)
        return 0
    results = backtest(args.splits, allow_test)
    threshold_surface(args.splits, allow_test)
    attribution(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
