"""Why one Strategy Tester report of the session breakout differs from another.

Each report is first parsed by ``scripts/research/tester_reconcile.py --phase
parse --out DIR`` into its own cache directory. Trades are matched on (day,
window) and each is expressed in R - net P&L over the dollars its stop put at
risk (lots x contract x bracket width) - so what a setting changed about
*which trades happen* separates from what it changed about *how big they were*.

The dollar difference is split along one chain, each step changing one thing:

  A total
  -> A on the trades B also took                  (selection: trades B skipped)
  -> B's outcomes on those trades, at A's size    (outcome: ticks, halt closes)
  -> B's own dollars on those trades              (sizing)
  + trades only B took                            = B total

Run:  python scripts/research/tester_compare.py A_DIR B_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qlab.symbols import get_spec  # noqa: E402
from tester_reconcile import _close_kind  # noqa: E402

CONTRACT = get_spec("XAUUSD").contract_size
KEYS = ["day", "hour", "range_min"]
HALT = "intraday (loss halt)"


def load(d: Path) -> tuple[pl.DataFrame, dict, pl.DataFrame]:
    settings = json.loads((d / "settings.json").read_text())
    bal = (pl.read_parquet(d / "tester_balance.parquet").sort("time")
           .rename({"time": "entry_time", "balance": "balance_before"}))
    tt = (pl.read_parquet(d / "tester_trades.parquet")
          .with_columns(risk_usd=pl.col("volume") * CONTRACT * pl.col("width"))
          .with_columns(
              r=pl.col("net") / pl.col("risk_usd"),
              year=pl.col("day").dt.year(),
              kind=pl.struct("day", "exit_time", "reason").map_elements(
                  lambda s: s["reason"] if s["reason"] != "ea"
                  else _close_kind(s["day"], s["exit_time"]), return_dtype=pl.Utf8))
          .sort("entry_time")
          # The balance just before the fill is what a percent-risk sizer saw.
          .join_asof(bal, on="entry_time", strategy="backward")
          .with_columns(risk_pct=pl.col("risk_usd") / pl.col("balance_before") * 100))
    return tt, settings, bal


def by_year(tt: pl.DataFrame) -> pl.DataFrame:
    return (tt.group_by("year")
            .agg(trades=pl.len(), net_usd=pl.col("net").sum(), sum_r=pl.col("r").sum(),
                 mean_r=pl.col("r").mean(), lots_p50=pl.col("volume").median(),
                 risk_usd_p50=pl.col("risk_usd").median(),
                 risk_pct_p50=pl.col("risk_pct").median(),
                 risk_pct_p95=pl.col("risk_pct").quantile(0.95),
                 halt_closes=(pl.col("kind") == HALT).sum())
            .sort("year").with_columns(pl.col(pl.Float64).round(3)))


def worst_drawdown(bal: pl.DataFrame) -> dict:
    b = bal.with_columns(peak=pl.col("balance_before").cum_max()).with_columns(
        dd_pct=(1 - pl.col("balance_before") / pl.col("peak")) * 100)
    trough = b.sort("dd_pct", descending=True).row(0, named=True)
    peak_row = (b.filter((pl.col("entry_time") <= trough["entry_time"])
                         & (pl.col("balance_before") == trough["peak"]))
                .sort("entry_time").row(-1, named=True))
    return {"peak_time": str(peak_row["entry_time"]), "peak": round(trough["peak"], 2),
            "trough_time": str(trough["entry_time"]),
            "trough": round(trough["balance_before"], 2),
            "dd_pct": round(trough["dd_pct"], 1)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir", type=Path)
    ap.add_argument("b_dir", type=Path)
    args = ap.parse_args()
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)

    A, sa, bal_a = load(args.a_dir)
    B, sb, bal_b = load(args.b_dir)
    out: dict = {}

    changed = {k: (sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb))
               if k.startswith("Inp") and sa.get(k) != sb.get(k)}
    print("settings that differ (A, B):", changed)
    out["settings_changed"] = changed

    for name, tt, bal in (("A", A, bal_a), ("B", B, bal_b)):
        print(f"\n=== report {name} by year ===")
        y = by_year(tt)
        print(y)
        out[f"{name}_by_year"] = y.to_dicts()
        out[f"{name}_worst_drawdown"] = worst_drawdown(bal)
        print("worst drawdown:", out[f"{name}_worst_drawdown"])

    # ---- match
    a = A.select(KEYS + [pl.col("direction").alias("a_dir"), pl.col("net").alias("a_net"),
                         pl.col("r").alias("a_r"), pl.col("risk_usd").alias("a_risk"),
                         pl.col("width_pct").alias("a_width_pct"),
                         pl.col("entry_time").alias("a_in")])
    b = B.select(KEYS + [pl.col("direction").alias("b_dir"), pl.col("net").alias("b_net"),
                         pl.col("r").alias("b_r"), pl.col("risk_usd").alias("b_risk"),
                         pl.col("kind").alias("b_kind")])
    j = a.join(b, on=KEYS, how="full", coalesce=True)
    both = j.filter(pl.col("a_dir").is_not_null() & pl.col("b_dir").is_not_null())
    b_only = j.filter(pl.col("a_dir").is_null())

    halts = B.filter(pl.col("kind") == HALT).group_by("day").agg(
        halt_time=pl.col("exit_time").min())
    floor = float(sb.get("InpMinRangePct", "0"))
    a_only = (j.filter(pl.col("b_dir").is_null())
              .join(halts, on="day", how="left")
              .with_columns(why=pl.when(pl.col("a_width_pct") < floor)
                            .then(pl.lit(f"bracket under {floor}% of price"))
                            .when(pl.col("halt_time").is_not_null()
                                  & (pl.col("a_in") >= pl.col("halt_time")))
                            .then(pl.lit("after B's daily loss halt"))
                            .otherwise(pl.lit("other: min-lot skip, position cap, ticks"))))
    skipped = (a_only.group_by("why")
               .agg(trades=pl.len(), a_net_usd=pl.col("a_net").sum(), a_sum_r=pl.col("a_r").sum(),
                    a_mean_r=pl.col("a_r").mean())
               .sort("trades", descending=True).with_columns(pl.col(pl.Float64).round(3)))
    print(f"\n=== trades A took and B did not ({a_only.height} of {A.height}) ===")
    print(skipped)
    out["a_only"] = skipped.to_dicts()

    same = both.filter(pl.col("a_dir") == pl.col("b_dir"))
    out["matched"] = {
        "matched": both.height, "b_only": b_only.height,
        "same_direction_pct": round(100 * same.height / max(both.height, 1), 2),
        "mean_r_a": round(float(same["a_r"].mean()), 4),
        "mean_r_b": round(float(same["b_r"].mean()), 4),
        "b_halt_closes_matched": int((same["b_kind"] == HALT).sum()),
    }
    print("\nmatched trades:", out["matched"])

    # ---- the chain
    a_total = float(A["net"].sum())
    b_total = float(B["net"].sum())
    step_sel = float(both["a_net"].sum())
    step_outcome = float((both["b_r"] * both["a_risk"]).sum())
    step_size = float(both["b_net"].sum())
    chain = {
        "A total": a_total,
        "selection: trades B did not take": step_sel - a_total,
        "outcome: B's result on the same trades, at A's size": step_outcome - step_sel,
        "sizing: B's lots instead of A's": step_size - step_outcome,
        "trades only B took": float(b_only["b_net"].sum()),
        "B total": b_total,
    }
    chain = {k: round(v, 2) for k, v in chain.items()}
    out["chain"] = chain
    print("\n=== A to B, one change at a time ($) ===")
    for k, v in chain.items():
        print(f"  {k:55s} {v:>12,.2f}")
    print(f"  (check: steps sum to {sum(v for k, v in chain.items() if k not in ('A total', 'B total')) + a_total:,.2f})")

    print("\nsum of R: A", round(float(A["r"].sum()), 1), " B", round(float(B["r"].sum()), 1),
          " | R on matched trades: A", round(float(both["a_r"].sum()), 1),
          " B", round(float(both["b_r"].sum()), 1))

    (args.b_dir / "compare.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {args.b_dir / 'compare.json'}")


if __name__ == "__main__":
    main()
