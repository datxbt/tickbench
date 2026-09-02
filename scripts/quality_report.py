"""Assess converted tick data and write the Stage 0 data quality report.

Outputs (under reports/data_quality/):
    tick_quality_by_month.parquet / .csv   per symbol-month metrics
    tick_hourly_profile.csv                tick counts by UTC hour
    DATA_QUALITY.md                        human-readable summary

Usage
-----
    python scripts/quality_report.py
    python scripts/quality_report.py -s XAUUSD -w 4
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from qlab import paths, quality
from qlab.symbols import ALL_SYMBOLS, get_spec


def _fmt(value: float, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def build_markdown(monthly: pl.DataFrame, hourly: pl.DataFrame) -> str:
    lines: list[str] = [
        "# Stage 0 - Tick Data Quality Report",
        "",
        f"Generated from {monthly.height} symbol-months across "
        f"{monthly['symbol'].n_unique()} instruments.",
        "",
        "This report measures the raw vendor feed. Nothing here has been cleaned or",
        "repaired - the point is to decide what is trustworthy before any strategy",
        "work begins.",
        "",
        "## 1. Inventory",
        "",
        "| Symbol | Class | Months | Rows | First tick | Last tick | Median ticks/day |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]

    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sub = monthly.filter(pl.col("symbol") == symbol)
        spec = get_spec(symbol)
        lines.append(
            f"| {symbol} | {spec.asset_class} | {sub.height} | {sub['rows'].sum():,} | "
            f"{sub['ts_min'].min():%Y-%m-%d} | {sub['ts_max'].max():%Y-%m-%d} | "
            f"{sub['median_ticks_per_day'].median():,.0f} |"
        )

    lines += [
        "",
        "## 2. Quote integrity",
        "",
        "`zero spread` = ticks where bid == ask. On an Exness Raw Spread account this",
        "is expected rather than suspect: the majors are quoted at a published 0.0 pip",
        "average and the broker charges commission instead. `crossed` = ask < bid,",
        "which is always corrupt.",
        "",
        "| Symbol | Zero spread % | Crossed % | Dup ts % | Out-of-order files | Spread mean (pips) | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sub = monthly.filter(pl.col("symbol") == symbol)
        weights = sub["rows"]
        zero = (sub["zero_spread_pct"] * weights).sum() / weights.sum()
        crossed = (sub["crossed_pct"] * weights).sum() / weights.sum()
        dup = (sub["duplicate_ts_pct"] * weights).sum() / weights.sum()
        mean_spread = (sub["spread_mean_pips"] * weights).sum() / weights.sum()
        lines.append(
            f"| {symbol} | {zero:.2f} | {crossed:.4f} | {dup:.3f} | "
            f"{int(sub['out_of_order'].sum())} | {mean_spread:.4f} | "
            f"{_fmt(sub['spread_p95_pips'].median())} | {_fmt(sub['spread_max_pips'].max())} |"
        )

    # --- Agreement with the published contract specification -----------------
    lines += [
        "",
        "### Agreement with the broker's published spec",
        "",
        "The real test of the ask side. Exness publishes an average spread per symbol,",
        "so the check is whether the tick-weighted mean matches it - not whether zero",
        "spreads occur. Measured Mon-Fri from 2024 on, to match the single-weekday",
        "basis of the published figure.",
        "",
        "| Symbol | Observed mean (pips, Mon-Fri 2024+) | Published avg | Tolerance | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        check = quality.spec_agreement(monthly, symbol)
        spec_value = check.get("spec_mean_pips")
        tolerance = check.get("tolerance_pips")
        lines.append(
            f"| {symbol} | {check.get('observed_mean_pips', float('nan')):.4f} | "
            f"{'-' if spec_value is None else f'{spec_value:.1f}'} | "
            f"{'-' if tolerance is None else f'±{tolerance:.2f}'} | {check['status']} |"
        )

    # --- Round-turn cost decomposition ---------------------------------------
    lines += [
        "",
        "### Round-turn cost",
        "",
        "Commission is the missing half of the cost picture: on the majors it dwarfs",
        "the spread. Quoted per standard lot, round turn, Mon-Fri from 2024 on.",
        "",
        "| Symbol | Mean spread (pips) | Commission (pips) | Total (pips) | Commission share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        spec = get_spec(symbol)
        sub = monthly.filter((pl.col("symbol") == symbol) & (pl.col("year") >= 2024))
        weights = sub["rows"]
        spread = float((sub["spread_mean_mon_fri_pips"] * weights).sum() / weights.sum())
        if spec.commission_per_lot_side_usd is None:
            lines.append(
                f"| {symbol} | {spread:.4f} | not on file | - | - |"
            )
            continue
        # USDJPY's pip is worth a rate-dependent amount of USD; use the period's
        # own mean price rather than a hard-coded rate.
        rate = 1.0
        if spec.quote_ccy == "JPY":
            rate = float((sub["price_max"] + sub["price_min"]).mean() / 2)
        commission = spec.commission_pips(rate)
        total = spread + commission
        lines.append(
            f"| {symbol} | {spread:.4f} | {commission:.3f} | {total:.3f} | "
            f"{100 * commission / total:.0f}% |"
        )

    lines += [
        "",
        "## 3. Continuity",
        "",
        "Weekend gaps and the instrument's daily maintenance break are excluded, so",
        "the outage columns count only unexplained downtime. `Daily breaks` is the",
        "routine broker session break, shown for reference - roughly one per trading",
        "day is normal for XAUUSD and USTEC.",
        "",
        "| Symbol | Missing weekdays | Daily breaks | Outages >1min | >5min | >1h | Longest outage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sub = monthly.filter(pl.col("symbol") == symbol)
        longest = sub["max_outage_gap_s"].max() or 0
        lines.append(
            f"| {symbol} | {int(sub['missing_weekdays'].sum())} | "
            f"{int(sub['n_session_breaks'].sum()):,} | "
            f"{int(sub['n_outage_gap_gt_60s'].sum()):,} | "
            f"{int(sub['n_outage_gap_gt_300s'].sum()):,} | "
            f"{int(sub['n_outage_gap_gt_3600s'].sum()):,} | "
            f"{longest / 3600:.1f} h |"
        )

    lines += [
        "",
        "## 4. Price sanity",
        "",
        f"A jump is a tick-to-tick mid move greater than "
        f"{quality.JUMP_THRESHOLD_BPS:g} bps.",
        "",
        "| Symbol | Price range | Jumps | Largest jump (bps) |",
        "| --- | --- | ---: | ---: |",
    ]
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sub = monthly.filter(pl.col("symbol") == symbol)
        lines.append(
            f"| {symbol} | {sub['price_min'].min():,.3f} - {sub['price_max'].max():,.3f} | "
            f"{int(sub['n_jumps'].sum()):,} | {_fmt(sub['max_jump_bps'].max(), 1)} |"
        )

    # Verdict -----------------------------------------------------------------
    lines += ["", "## 5. Verdict", ""]
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sub = monthly.filter(pl.col("symbol") == symbol)
        issues: list[str] = []
        check = quality.spec_agreement(monthly, symbol)
        if check.get("agrees") is False:
            issues.append(
                f"observed mean spread {check['observed_mean_pips']:.3f} pips does not "
                f"match the published {check['spec_mean_pips']:.1f} - investigate before "
                f"trusting the ask side"
            )
        elif check.get("agrees") is None:
            issues.append(
                "no published contract spec on file - add it before cost modelling"
            )
        if sub["crossed_pct"].max() and sub["crossed_pct"].max() > 0:
            issues.append(
                f"crossed quotes present (max {sub['crossed_pct'].max():.4f}% in a month)"
            )
        if int(sub["missing_weekdays"].sum()) > 0:
            issues.append(f"{int(sub['missing_weekdays'].sum())} weekdays with no data")
        if int(sub["n_outage_gap_gt_3600s"].sum()) > 0:
            issues.append(
                f"{int(sub['n_outage_gap_gt_3600s'].sum())} unexplained outages over an hour"
            )
        verdict = "usable as-is" if not issues else "; ".join(issues)
        lines.append(f"- **{symbol}**: {verdict}")

    lines += [
        "",
        "## 6. UTC hour coverage",
        "",
        "Share of each symbol's ticks by UTC hour - the basis for session definitions",
        "in Stage 2 cost modelling.",
        "",
    ]
    pivot = (
        hourly.group_by(["symbol", "hour"])
        .agg(pl.col("len").sum())
        .with_columns(
            pct=100 * pl.col("len") / pl.col("len").sum().over("symbol")
        )
        .sort(["symbol", "hour"])
    )
    lines.append("| Symbol | " + " | ".join(f"{h:02d}" for h in range(24)) + " |")
    lines.append("| --- |" + " ---: |" * 24)
    for symbol in sorted(pivot["symbol"].unique().to_list()):
        row = pivot.filter(pl.col("symbol") == symbol)
        by_hour = dict(zip(row["hour"].to_list(), row["pct"].to_list()))
        cells = " | ".join(f"{by_hour.get(h, 0.0):.1f}" for h in range(24))
        lines.append(f"| {symbol} | {cells} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-w", "--workers", type=int, default=6)
    args = parser.parse_args()

    paths.ensure_dirs()
    started = time.perf_counter()
    records: list[dict] = []
    hourly_rows: list[dict] = []
    failures: list[dict] = []

    for record in quality.run(args.symbols, workers=args.workers):
        if record.get("status") != "ok":
            failures.append(record)
            print(f"FAILED {record.get('file')}: {record['error']}", flush=True)
            continue
        hourly_rows.extend(record.pop("_hourly"))
        record.pop("status")
        records.append(record)
        print(
            f"  {record['symbol']} {record['year']}-{record['month']:02d} "
            f"rows={record['rows']:>10,} zero_spread={record['zero_spread_pct']:6.2f}%",
            flush=True,
        )

    if not records:
        print("no converted parquet found - run scripts/convert_ticks.py first")
        return 1

    monthly = pl.DataFrame(records).sort(["symbol", "year", "month"])
    hourly = pl.DataFrame(hourly_rows)

    monthly.write_parquet(paths.QUALITY_DIR / "tick_quality_by_month.parquet")
    monthly.write_csv(paths.QUALITY_DIR / "tick_quality_by_month.csv")
    hourly.group_by(["symbol", "hour"]).agg(pl.col("len").sum()).sort(
        ["symbol", "hour"]
    ).write_csv(paths.QUALITY_DIR / "tick_hourly_profile.csv")

    report_path = paths.QUALITY_DIR / "DATA_QUALITY.md"
    report_path.write_text(build_markdown(monthly, hourly), encoding="utf-8")

    print(
        f"\n{monthly.height} symbol-months assessed in "
        f"{(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    print(f"report: {report_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
