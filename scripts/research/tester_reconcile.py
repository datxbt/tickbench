"""Reconcile an MT5 Strategy Tester report of the session breakout with the tick model.

The tester report and ``scripts/backtests/backtest_session_breakout_top8.py``
describe the same expert, preset and sizing on the same instrument. This script
checks whether they tell the same story, and then asks what the report's own
settings would have shown over the rest of the corpus.

Phases, caching to ``reports/strategies/session_breakout_top8/tester/``:

  parse    pair the report's deals into trades. Every pairing is checked
           against the profit the report prints, so a mis-pair shows up as a
           residual rather than as a silently wrong trade.
  halt     reduce the model's tape to per-day daily-loss-halt tables (one pass
           over the ticks), so any deposit, limit and start date can be
           replayed without another pass.
  report   match tester trades to model trades, decompose the dollar gap,
           replay the tester's account settings on the model, and sweep the
           start date.

Run:  python scripts/research/tester_reconcile.py REPORT.xlsx [--phase parse halt report]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel  # noqa: E402
from qlab.strategies.session_breakout import (  # noqa: E402
    flatten_second_utc,
    halt_tables,
    is_us_dst,
)
from qlab.symbols import get_spec  # noqa: E402

SYMBOL = "XAUUSD"
LOTS = 0.02
BASE = Path("reports/strategies/session_breakout_top8")
OUT = BASE / "tester"
MODEL_END = date(2026, 8, 31)        # last day of the tick corpus
TESTER_START = date(2023, 1, 2)


def _era(day: date) -> str:
    if day < date(2024, 1, 1):
        return "dev"
    if day < date(2025, 7, 1):
        return "validation"
    return "test"


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------

def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)


def read_report(path: Path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    i_orders = next(i for i, r in enumerate(rows) if r and r[0] == "Orders")
    i_deals = next(i for i, r in enumerate(rows) if r and r[0] == "Deals")

    settings: dict[str, str] = {}
    for r in rows[:i_orders]:
        vals = [v for v in r if v is not None]
        for k, v in enumerate(vals):
            if isinstance(v, str) and v.startswith("Inp") and "=" in v:
                key, val = v.split("=", 1)
                settings[key] = val
            elif isinstance(v, str) and v.endswith(":") and k + 1 < len(vals):
                settings[v[:-1]] = str(vals[k + 1])

    # Column positions are read from each table's own header row: the Orders
    # table can carry blank merged cells, so a fixed index silently reads the
    # wrong field.
    oc = {name: k for k, name in enumerate(rows[i_orders + 1]) if name}
    orders = {}
    for r in rows[i_orders + 2:i_deals]:
        if r[0] is None or r[oc["Order"]] is None:
            continue
        orders[int(r[oc["Order"]])] = {
            "type": r[oc["Type"]], "price": r[oc["Price"]], "sl": r[oc["S / L"]],
            "tp": r[oc["T / P"]], "comment": r[oc["Comment"]]}

    deals = []
    for r in rows[i_deals + 2:]:
        if r[0] is None or r[4] not in ("in", "out"):
            continue
        deals.append({
            "time": _ts(r[0]), "deal": int(r[1]), "type": r[3], "dir": r[4],
            "volume": float(str(r[5]).split("/")[0]), "price": float(r[6]),
            "order": int(r[7]), "commission": float(r[8] or 0.0),
            "swap": float(r[9] or 0.0), "profit": float(r[10] or 0.0),
            "balance": float(r[11]), "comment": r[12],
        })
    return settings, orders, deals


def pair_deals(orders: dict, deals: list[dict], contract: float) -> tuple[pl.DataFrame, dict]:
    """Match each closing deal to the open position whose P&L it reproduces.

    The report does not carry position ids. But the printed profit of a close
    is ``direction x (exit - entry) x contract x volume`` for exactly one open
    position, so choosing the candidate that reproduces it is a check as well
    as a match. Ties (identical entries) go to the oldest position.
    """
    open_pos: list[dict] = []
    trades: list[dict] = []
    unpaired = 0
    for dl in sorted(deals, key=lambda x: (x["time"], x["deal"])):
        if dl["dir"] == "in":
            o = orders.get(dl["order"], {})
            open_pos.append({
                "window": dl["comment"], "direction": 1 if dl["type"] == "buy" else -1,
                "entry_time": dl["time"], "entry": dl["price"], "volume": dl["volume"],
                "comm_in": dl["commission"], "sl": o.get("sl"),
                "order_price": o.get("price"),
            })
            continue
        closes = 1 if dl["type"] == "sell" else -1          # a sell out closes a long
        cands = [p for p in open_pos
                 if p["direction"] == closes and abs(p["volume"] - dl["volume"]) < 1e-9]
        if not cands:
            unpaired += 1
            continue
        err = [abs(p["direction"] * (dl["price"] - p["entry"]) * contract * p["volume"]
                   - dl["profit"]) for p in cands]
        p = cands[int(np.argmin(err))]
        open_pos.remove(p)
        c = dl["comment"] or ""
        h, r, _ = (p["window"] or "h-1_r-1_t0").split("_")
        trades.append({
            "window": p["window"], "hour": int(h[1:]), "range_min": int(r[1:]),
            "direction": p["direction"], "day": p["entry_time"].date(),
            "entry_time": p["entry_time"], "exit_time": dl["time"],
            "entry": p["entry"], "exit": dl["price"], "volume": p["volume"],
            # The stop sits on the far edge, so the bracket is the pending
            # order's distance to its own stop.
            "width": (abs(p["order_price"] - p["sl"])
                      if p["order_price"] and p["sl"] else None),
            "width_pct": (abs(p["order_price"] - p["sl"])
                          / ((p["order_price"] + p["sl"]) / 2) * 100
                          if p["order_price"] and p["sl"] else None),
            "reason": "tp" if c.startswith("tp") else "sl" if c.startswith("sl") else "ea",
            "profit": dl["profit"], "commission": p["comm_in"] + dl["commission"],
            "swap": dl["swap"], "net": dl["profit"] + p["comm_in"] + dl["commission"] + dl["swap"],
            "pair_residual": float(min(err)),
        })
    frame = pl.DataFrame(trades).sort("entry_time")
    return frame, {"unpaired_closes": unpaired, "still_open": len(open_pos),
                   "max_residual": float(frame["pair_residual"].max()),
                   "residual_over_2c": int((frame["pair_residual"] > 0.02).sum())}


def phase_parse(report: Path) -> None:
    settings, orders, deals = read_report(report)
    trades, quality = pair_deals(orders, deals, get_spec(SYMBOL).contract_size)
    OUT.mkdir(parents=True, exist_ok=True)
    trades.write_parquet(OUT / "tester_trades.parquet")
    pl.DataFrame([{"time": d["time"], "balance": d["balance"]} for d in deals]
                 ).sort("time").write_parquet(OUT / "tester_balance.parquet")
    (OUT / "settings.json").write_text(json.dumps(settings | {"pairing": quality}, indent=1))
    print(f"parsed {trades.height} trades from {len(deals)} deals; pairing {quality}")


# --------------------------------------------------------------------------
# halt
# --------------------------------------------------------------------------

def phase_halt(verbose: bool) -> None:
    spec = get_spec(SYMBOL)
    slips = {seg: np.array([CostModel.from_profiles(SYMBOL, split=seg).slippage_pips(hour=h)
                            * spec.pip for h in range(24)])
             for seg in ("dev", "validation", "test")}
    base = pl.read_parquet(BASE / "tapes" / "base.parquet")
    tables = halt_tables(SYMBOL, base, lots=LOTS,
                         slip_by_hour=lambda d: slips[_era(d)], verbose=verbose)
    with open(OUT / "halt_tables.pkl", "wb") as fh:
        pickle.dump(tables, fh)
    print(f"wrote halt tables for {len(tables)} days")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _drawdown(balance: np.ndarray, deposit: float) -> tuple[float, float]:
    path = np.concatenate([[deposit], balance])
    peak = np.maximum.accumulate(path)
    dd = peak - path
    return float(dd.max()), float((dd / peak).max() * 100)


def replay(tables: dict, start: date, end: date, deposit: float, pct: float,
           checkpoint: date = date(2025, 12, 31)) -> dict:
    """The tester's account rules on the model's trades: fixed lots, and a
    daily loss limit of ``pct`` of each morning's balance."""
    bal, path, halted, ruined, at_checkpoint = deposit, [], 0, None, None
    for d in sorted(tables):
        if d < start or d > end:
            continue
        if at_checkpoint is None and d > checkpoint:
            at_checkpoint = bal
        pnl, fired = tables[d].pnl(bal * pct / 100.0 if pct > 0 else 0.0)
        bal += pnl
        halted += fired
        path.append(bal)
        if bal <= 0:
            ruined = d
            break
    dd_usd, dd_pct = _drawdown(np.array(path), deposit)
    return {"start": start, "deposit": deposit, "halt_pct": pct,
            "end_balance": round(bal, 2), "net": round(bal - deposit, 2),
            "net_before_2026": round((at_checkpoint if at_checkpoint is not None else bal)
                                     - deposit, 2),
            "min_balance": round(min([deposit] + path), 2),
            "max_dd_usd": round(dd_usd, 2), "max_dd_pct": round(dd_pct, 1),
            "halt_days": halted, "ruined_on": ruined}


def _close_kind(day: date, exit_time: datetime) -> str:
    sec = exit_time.hour * 3600 + exit_time.minute * 60 + exit_time.second
    flat = flatten_second_utc(day)
    if exit_time.date() > day:
        return "held past midnight"
    if sec < flat - 120:
        return "intraday (loss halt)"
    if sec <= flat + 300:
        return "at the flatten"
    return "after the halt (reopen)"


def phase_report() -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    rnd = lambda f: f.with_columns(pl.col(pl.Float64).round(2))  # noqa: E731
    settings = json.loads((OUT / "settings.json").read_text())
    tt = pl.read_parquet(OUT / "tester_trades.parquet")
    tbal = pl.read_parquet(OUT / "tester_balance.parquet")
    base = pl.read_parquet(BASE / "tapes" / "base.parquet")
    summary: dict = {"pairing": settings["pairing"]}

    print("settings:", {k: settings[k] for k in settings
                        if k.startswith("Inp") or k in ("Period", "History Quality",
                                                        "Initial Deposit")})
    print("pairing:", settings["pairing"])

    # ---- 1. the tester by year, and the days after the corpus ends
    deposit = float(settings.get("Initial Deposit", "1000"))
    model_win = base.filter(pl.col("day").is_between(TESTER_START, MODEL_END))
    ty = (tt.group_by(pl.col("day").dt.year().alias("year"))
          .agg(t_trades=pl.len(), t_net=pl.col("net").sum(),
               t_commission=pl.col("commission").sum(), t_swap=pl.col("swap").sum()))
    my = (model_win.group_by(pl.col("day").dt.year().alias("year"))
          .agg(m_trades=pl.len(), m_net=(pl.col("net_usd") * LOTS).sum(),
               m_commission=-(pl.col("commission_usd") * LOTS).sum()))
    print("\n=== by year: tester vs tick model (no halt), $ at 0.02 lots ===")
    by_year = rnd(ty.join(my, on="year", how="full", coalesce=True).sort("year"))
    print(by_year)
    after = tt.filter(pl.col("day") > MODEL_END)
    summary["by_year"] = by_year.to_dicts()
    summary["tester_after_corpus"] = {"trades": after.height,
                                      "net": round(float(after["net"].sum()), 2)}
    print("tester after the corpus ends (2026-09-01..):", summary["tester_after_corpus"])

    # ---- 2. how positions closed
    kinds = (tt.filter(pl.col("reason") == "ea")
             .with_columns(kind=pl.struct("day", "exit_time").map_elements(
                 lambda s: _close_kind(s["day"], s["exit_time"]), return_dtype=pl.Utf8))
             .group_by("kind").agg(trades=pl.len(), days=pl.col("day").n_unique(),
                                   net=pl.col("net").sum()).sort("trades", descending=True))
    print("\n=== tester closes not at stop or target ===")
    print(rnd(kinds))
    summary["close_kinds"] = kinds.to_dicts()
    holds = tt.with_columns(hold_h=(pl.col("exit_time") - pl.col("entry_time"))
                            .dt.total_seconds() / 3600)
    long_holds = holds.filter(pl.col("hold_h") > 12).sort("hold_h", descending=True)
    print(f"holds over 12h: {long_holds.height}, net {long_holds['net'].sum():.2f}; longest:")
    print(long_holds.select("window", "entry_time", "exit_time", "hold_h", "net").head(6))

    # ---- 3. trade-by-trade match over the shared window
    t = tt.filter(pl.col("day") <= MODEL_END).select(
        "day", "hour", "range_min", pl.col("direction").alias("t_dir"),
        pl.col("entry").alias("t_entry"), pl.col("reason").alias("t_reason"),
        pl.col("net").alias("t_net"), pl.col("commission").alias("t_comm"),
        pl.col("entry_time").dt.epoch("us").alias("t_ts"))
    m = model_win.select(
        "day", "hour", "range_min", pl.col("direction").alias("m_dir"),
        pl.col("entry").alias("m_entry"), pl.col("entry_mid").alias("m_mid"),
        # Name exits the way the report does, so "same reason" means it.
        pl.col("reason").replace({"flat": "ea", "stop": "sl"}).alias("m_reason"),
        (pl.col("net_usd") * LOTS).alias("m_net"),
        (-pl.col("commission_usd") * LOTS).alias("m_comm"),
        pl.col("entry_ts").alias("m_ts"))
    j = t.join(m, on=["day", "hour", "range_min"], how="full", coalesce=True)
    both = j.filter(pl.col("t_dir").is_not_null() & pl.col("m_dir").is_not_null())
    t_only = j.filter(pl.col("m_dir").is_null())
    m_only = j.filter(pl.col("t_dir").is_null())
    same = both.filter(pl.col("t_dir") == pl.col("m_dir"))
    same_reason = same.filter(pl.col("t_reason") == pl.col("m_reason"))

    match = {
        "tester_trades": t.height, "model_trades": m.height, "matched": both.height,
        "same_direction_pct": round(100 * same.height / max(both.height, 1), 2),
        "same_exit_reason_pct": round(100 * same_reason.height / max(same.height, 1), 2),
        "tester_only": t_only.height, "model_only": m_only.height,
        "entry_time_diff_s_median": float(((same["t_ts"] - same["m_ts"]).abs() / 1e6).median()),
        "entry_vs_model_fill_median": float(((same["t_entry"] - same["m_entry"])
                                             * same["t_dir"]).median()),
        "entry_vs_mid_median_tester": float(((same["t_entry"] - same["m_mid"])
                                             * same["t_dir"]).median()),
        "entry_vs_mid_median_model": float(((same["m_entry"] - same["m_mid"])
                                            * same["t_dir"]).median()),
    }
    summary["match"] = match
    print("\n=== trade-by-trade match, 2023-01-02..2026-08-31 ===")
    print(match)
    xt = (same.group_by("t_reason", "m_reason").agg(n=pl.len())
          .sort("n", descending=True))
    print("exit reason, tester x model (same direction):")
    print(xt)

    gap = {
        "tester_net": float(t["t_net"].sum()),
        "model_net": float(m["m_net"].sum()),
        "tester_only_net": float(t_only["t_net"].sum()),
        "model_only_net": float(m_only["m_net"].sum()),
        "same_trade_same_exit": float((same_reason["t_net"] - same_reason["m_net"]).sum()),
        "same_trade_other_exit": float(
            (same.filter(pl.col("t_reason") != pl.col("m_reason"))["t_net"]
             - same.filter(pl.col("t_reason") != pl.col("m_reason"))["m_net"]).sum()),
        "opposite_direction": float(
            (both.filter(pl.col("t_dir") != pl.col("m_dir"))["t_net"]
             - both.filter(pl.col("t_dir") != pl.col("m_dir"))["m_net"]).sum()),
        "of_which_commission": float((both["t_comm"] - both["m_comm"]).sum()),
    }
    gap = {k: round(v, 2) for k, v in gap.items()}
    summary["gap"] = gap
    print("\n=== where the dollar gap comes from (tester minus model) ===")
    for k, v in gap.items():
        print(f"  {k:24s} {v:>12,.2f}")

    # ---- 4. replay the tester's account on the model
    with open(OUT / "halt_tables.pkl", "rb") as fh:
        tables = pickle.load(fh)
    check = sum(h.pnl_full for d, h in tables.items())
    print(f"\nhalt tables: {len(tables)} days, no-halt total {check:,.2f} "
          f"vs tape {float((base['net_usd'] * LOTS).sum()):,.2f}")
    pct = float(settings.get("InpMaxDailyLossPct", "0"))
    tb = tbal.filter(pl.col("time").dt.date() <= MODEL_END)
    t_dd = _drawdown(tb["balance"].to_numpy()[1:], deposit)
    replays = [
        {"source": "tester report"} | {
            "end_balance": float(tb["balance"][-1]), "net": float(tb["balance"][-1]) - deposit,
            "max_dd_usd": round(t_dd[0], 2), "max_dd_pct": round(t_dd[1], 1),
            "halt_days": int(sum(r["days"] for r in summary["close_kinds"]
                                 if r["kind"] == "intraday (loss halt)"))},
        {"source": f"model, halt {pct}%"} | replay(tables, TESTER_START, MODEL_END, deposit, pct),
        {"source": "model, no halt"} | replay(tables, TESTER_START, MODEL_END, deposit, 0.0),
    ]
    print("\n=== the report's account settings, 2023-01-02..2026-08-31 ===")
    print(pl.DataFrame(replays, strict=False).select(
        "source", "end_balance", "net", "max_dd_usd", "max_dd_pct", "halt_days"))
    summary["replays"] = replays

    # ---- 5. the same report, started on other dates
    starts = [date(2020, 1, 29)] + [date(y, mo, 1) for y in range(2020, 2027)
                                    for mo in (1, 7)
                                    if date(2020, 2, 1) < date(y, mo, 1) <= date(2026, 1, 1)]
    starts.insert(starts.index(date(2023, 1, 1)) + 1, TESTER_START)
    sweep = [replay(tables, s, MODEL_END, deposit, pct) for s in starts]
    print(f"\n=== start-date sweep: ${deposit:,.0f}, 0.02 lots, halt {pct}%, to 2026-08-31 ===")
    print(pl.DataFrame(sweep).select("start", "end_balance", "net", "net_before_2026",
                                     "min_balance", "max_dd_usd", "max_dd_pct",
                                     "halt_days", "ruined_on"))
    summary["start_sweep"] = sweep

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {OUT / 'summary.json'}")


def phase_detail() -> None:
    """Why the tester and the model differ, trade by trade, and when gold halts.

    Two questions the headline gap cannot answer:

    * Which part of the gap is cost (commission, slippage) and which is exit
      handling (a close at the reopen instead of the flatten)?
    * On the days the tester closed at the reopen, did the market halt before
      the expert's scheduled flatten? If so the flatten never gets a tick, and
      that happens live as well as in the tester.
    """
    from qlab.loader import load_bars  # only this phase needs bars

    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    rnd = lambda f: f.with_columns(pl.col(pl.Float64).round(2))  # noqa: E731
    per_dollar = get_spec(SYMBOL).contract_size * LOTS
    tt = pl.read_parquet(OUT / "tester_trades.parquet").with_columns(
        kind=pl.struct("day", "exit_time", "reason").map_elements(
            lambda s: s["reason"] if s["reason"] != "ea"
            else _close_kind(s["day"], s["exit_time"]), return_dtype=pl.Utf8))
    base = pl.read_parquet(BASE / "tapes" / "base.parquet")
    detail: dict = {}

    # ---- the gap, decomposed
    t = tt.filter(pl.col("day") <= MODEL_END).select(
        "day", "hour", "range_min", pl.col("direction").alias("t_dir"),
        pl.col("entry").alias("t_entry"), pl.col("exit").alias("t_exit"),
        pl.col("kind").alias("t_kind"), pl.col("net").alias("t_net"),
        pl.col("commission").alias("t_comm"), pl.col("entry_time").alias("t_in"),
        pl.col("exit_time").alias("t_out"))
    m = base.filter(pl.col("day") >= TESTER_START).select(
        "day", "hour", "range_min", pl.col("direction").alias("m_dir"),
        pl.col("entry").alias("m_entry"), pl.col("exit").alias("m_exit"),
        pl.col("reason").replace({"flat": "ea"}).alias("m_reason"),
        (pl.col("net_usd") * LOTS).alias("m_net"),
        (-pl.col("commission_usd") * LOTS).alias("m_comm"),
        pl.from_epoch("entry_ts", time_unit="us").dt.replace_time_zone("UTC").alias("m_in"),
        pl.from_epoch("exit_ts", time_unit="us").dt.replace_time_zone("UTC").alias("m_out"))
    j = (t.join(m, on=["day", "hour", "range_min"], how="inner")
         .filter(pl.col("t_dir") == pl.col("m_dir"))
         .with_columns(
             entry_eff=-(pl.col("t_entry") - pl.col("m_entry")) * pl.col("t_dir") * per_dollar,
             exit_eff=(pl.col("t_exit") - pl.col("m_exit")) * pl.col("t_dir") * per_dollar,
             comm_eff=pl.col("t_comm") - pl.col("m_comm"),
             total=pl.col("t_net") - pl.col("m_net")))
    dec = rnd(j.group_by("t_kind", "m_reason")
              .agg(n=pl.len(), total=pl.col("total").sum(),
                   entry=pl.col("entry_eff").sum(), exit=pl.col("exit_eff").sum(),
                   commission=pl.col("comm_eff").sum())
              .sort("n", descending=True))
    print("=== tester minus model, same trade, by how each closed ($) ===")
    print(dec)
    detail["decomposition"] = dec.to_dicts()
    odd = (j.filter(pl.col("t_kind").is_in(["tp", "sl"]).not_()
                    | (pl.col("t_kind").replace({"sl": "stop"}) != pl.col("m_reason")))
           .filter(~((pl.col("t_kind").is_in(["at the flatten", "after the halt (reopen)",
                                              "held past midnight"]))
                     & (pl.col("m_reason") == "ea")))
           .sort(pl.col("total").abs(), descending=True))
    print("largest disagreements on exit:")
    print(rnd(odd.select("day", "hour", "range_min", "t_dir", "t_kind", "m_reason",
                         "t_out", "m_out", "t_exit", "m_exit", "total").head(12)))

    # ---- when the market actually halts, from the tick corpus
    bars = load_bars(SYMBOL, "1m", start=TESTER_START, end=MODEL_END,
                     allow_test=True, columns=["ts", "ts_open"])
    halts = (bars.sort("ts_open")
             .with_columns(day=pl.col("ts_open").dt.date(),
                           gap_min=(pl.col("ts_open").shift(-1) - pl.col("ts_open"))
                           .dt.total_minutes())
             .filter((pl.col("ts_open").dt.hour() >= 17) & (pl.col("gap_min") >= 30))
             .group_by("day").agg(last_bar=pl.col("ts_open").min()))
    kinds = (tt.filter(pl.col("kind").is_in(["at the flatten", "after the halt (reopen)",
                                              "held past midnight"]))
             .group_by("day").agg(kind=pl.col("kind").first())
             .join(halts, on="day", how="left")
             .with_columns(
                 flat_sec=pl.col("day").map_elements(flatten_second_utc,
                                                     return_dtype=pl.Int64),
                 # dt.hour() is Int8, and 21 * 3600 does not fit in one.
                 halt_sec=pl.col("last_bar").dt.hour().cast(pl.Int64) * 3600
                 + pl.col("last_bar").dt.minute().cast(pl.Int64) * 60 + 60,
                 dst=pl.col("day").map_elements(is_us_dst, return_dtype=pl.Boolean),
                 weekday=pl.col("day").dt.weekday())
             .with_columns(halt_before_flatten=pl.col("halt_sec") <= pl.col("flat_sec"),
                           halt_clock=pl.col("last_bar").dt.strftime("%H:%M")))
    ht = (kinds.group_by("kind", "dst")
          .agg(days=pl.len(), halt_before_flatten=pl.col("halt_before_flatten").sum(),
               typical_halt=pl.col("halt_clock").mode().first(),
               earliest=pl.col("halt_clock").min(), latest=pl.col("halt_clock").max())
          .sort("kind", "dst"))
    print("\n=== did the market halt before the scheduled flatten? (from the ticks) ===")
    print(ht)
    detail["halt_timing"] = ht.to_dicts()
    reopen_days = kinds.filter(pl.col("kind") == "after the halt (reopen)")
    print("halt clock on reopen-close days:")
    print(reopen_days.group_by("halt_clock", "dst").agg(days=pl.len())
          .sort("days", descending=True).head(10))
    print("by year:", reopen_days.group_by(pl.col("day").dt.year().alias("y"))
          .agg(pl.len()).sort("y").rows())
    held = tt.filter(pl.col("kind") == "held past midnight")
    print("held past midnight:", held.select("day", "window", "exit_time", "net").rows())

    # Exit effect of closing at the reopen rather than at the flatten.
    reopen = j.filter(pl.col("t_kind") == "after the halt (reopen)")
    detail["reopen_exit_effect"] = {
        "trades": reopen.height, "exit_effect_usd": round(float(reopen["exit_eff"].sum()), 2),
        "per_trade_sd_usd": round(float(reopen["exit_eff"].std()), 2),
        "worst_usd": round(float(reopen["exit_eff"].min()), 2),
        "best_usd": round(float(reopen["exit_eff"].max()), 2)}
    print("\nreopen closes vs the model's pre-halt close:", detail["reopen_exit_effect"])
    (OUT / "detail.json").write_text(json.dumps(detail, indent=1, default=str))


def main() -> None:
    global OUT
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--phase", nargs="+", default=["parse", "halt", "report"],
                    choices=["parse", "halt", "report", "detail"])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="cache directory, one per report")
    args = ap.parse_args()
    OUT = args.out
    if "parse" in args.phase:
        phase_parse(args.report)
    if "halt" in args.phase:
        phase_halt(args.verbose)
    if "report" in args.phase:
        phase_report()
    if "detail" in args.phase:
        phase_detail()


if __name__ == "__main__":
    main()
