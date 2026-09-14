"""Discovery batch 7: H16, the H15 portfolio executed on ticks.

H16 exactly as registered in docs/findings/discovery-program.md. The same 23
cells as H15; the assumed slippage term is replaced by tick fills at a stated
latency. ``dev`` measures on dev ticks and reports the power gate; ``validate``
counts validation event days first and reads fills only if the gate passed.

    python scripts/research/discovery_batch7.py dev
    python scripts/research/discovery_batch7.py validate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import discovery_batch5 as d  # noqa: E402
import discovery_batch6 as b6  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.loader import load_ticks  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

OUT = Path("reports/discovery/batch7")
LATENCIES_MS = (0, 250, 500, 1000)
PRIMARY_MS = 250
MAX_STALE_US = 60_000_000  # a standing quote older than this means the market was shut
VAL_EVENT_DAYS = 388


def cell_events(split: str, state: dict | None):
    """Every kept cell's events on ``split``: symbol, entry stamp, direction, horizon, tday, half."""
    feats, bases, lookup, diurnals, betas = b6.build_selected(split, state)
    kept = pl.read_parquet(b6.OUT / "kept.parquet").to_dicts()
    frames = []
    for k, row in enumerate(kept):
        b = bases[lookup[(row["symbol"], row["family"], row["param"])]]
        f = feats[b.symbol]
        idx, direction, mid, rt, cont = b6.events(row, b, f)
        frames.append(pl.DataFrame({
            "cell": k, "symbol": b.symbol, "T": f.ts[idx], "direction": direction.astype(np.int8),
            "h": row["h"], "tday": f.tdrank[idx], "half": f.half[idx], "hour": f.hour[idx],
            "bar_mid": mid, "bar_net_now": mid - rt,
        }))
    return pl.concat(frames), (diurnals, betas)


def month_range(ts_us: np.ndarray):
    lo = datetime.fromtimestamp(ts_us.min() / 1e6, tz=timezone.utc)
    hi = datetime.fromtimestamp(ts_us.max() / 1e6, tz=timezone.utc)
    y, m = lo.year, lo.month
    while (y, m) <= (hi.year, hi.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fill(ev: pl.DataFrame, symbol: str, split: str) -> pl.DataFrame:
    """Tick fills for one symbol's events at every latency, month by month."""
    spec = get_spec(symbol)
    out = []
    T = ev["T"].to_numpy()
    for y, m in month_range(T):
        start = datetime(y, m, 1, tzinfo=timezone.utc)
        nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        sel = (T >= start.timestamp() * 1e6) & (T < nxt.timestamp() * 1e6)
        if not sel.any():
            continue
        e = ev.filter(pl.Series(sel))
        end_day = date.fromtimestamp(min(nxt.timestamp() + 86400, 4102444800))
        ticks = load_ticks(symbol, start=start, end=end_day, columns=["ts", "bid", "ask"],
                           allow_test=(split == "test"))
        tt = ticks["ts"].dt.epoch("us").to_numpy()
        bid, ask = ticks["bid"].to_numpy(), ticks["ask"].to_numpy()
        del ticks
        Te = e["T"].to_numpy()
        hz = e["h"].to_numpy().astype(np.int64) * 60_000_000
        side = e["direction"].to_numpy().astype(np.float64)
        cols = {}
        for lat in LATENCIES_MS:
            # The quote standing at the instant the order arrives: the last tick
            # at or before it. The feed prints only on change, so a silent
            # second is a live quote, not a missing one.
            te, tx = Te + lat * 1000, Te + hz + lat * 1000
            ie = np.searchsorted(tt, te, "right") - 1
            ix = np.searchsorted(tt, tx, "right") - 1
            ok = (ie >= 0) & (ix >= 0)
            ie_c, ix_c = np.maximum(ie, 0), np.maximum(ix, 0)
            ok &= (te - tt[ie_c] <= MAX_STALE_US) & (tx - tt[ix_c] <= MAX_STALE_US)
            mid_e = (bid[ie_c] + ask[ie_c]) / 2
            mid_x = (bid[ix_c] + ask[ix_c]) / 2
            mid = side * (mid_x / mid_e - 1) * 1e4
            long = side > 0
            gross = np.where(long, bid[ix_c] / ask[ie_c] - 1, bid[ie_c] / ask[ix_c] - 1) * 1e4
            price = mid_e
            rate = price if spec.quote_ccy != "USD" else 1.0
            comm = spec.commission_pips(rate) * spec.pip / price * 1e4
            cols[f"mid_{lat}"] = np.where(ok, mid, np.nan)
            cols[f"fill_{lat}"] = np.where(ok, gross - comm, np.nan)
            cols[f"comm_{lat}"] = comm
            cols[f"px_{lat}"] = price
        out.append(e.with_columns(**{k: pl.Series(v) for k, v in cols.items()}))
    return pl.concat(out)


def net_now(t: pl.DataFrame, lat: int) -> pl.Series:
    """Tick mid move between the fills - validation-era mean spread (entry hour) - commission."""
    # Keyed by row, not (cell, T): an outside bar can sweep both sides and put
    # two events of one cell on the same stamp.
    out = np.full(t.height, np.nan)
    sym = t["symbol"].to_numpy()
    for s in d.SYMBOLS:
        rows = np.flatnonzero(sym == s)
        if rows.size == 0:
            continue
        spec = get_spec(s)
        cm = CostModel.from_profiles(s, split="validation")
        spread = np.array([cm.spread_pips(h) for h in range(24)])
        g = t[rows]
        sp_bps = spread[g["hour"].to_numpy()] * spec.pip / g[f"px_{lat}"].to_numpy() * 1e4
        out[rows] = g[f"mid_{lat}"].to_numpy() - sp_bps - g[f"comm_{lat}"].to_numpy()
    return pl.Series(out)


def day_stats(t: pl.DataFrame, col: str) -> dict:
    g = t.filter(pl.col(col).is_finite())
    daily = g.group_by("tday").agg(pl.col(col).sum())[col].to_numpy()
    sd = daily.std(ddof=1)
    return {"trades": g.height, "days": int(daily.size), "per_trade": float(g[col].mean()),
            "t_day": float(daily.mean() / sd * math.sqrt(daily.size)),
            "sharpe_ann": float(daily.mean() / sd * math.sqrt(252))}


def dev() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev, _ = cell_events("dev", None)
    tape = pl.concat([fill(ev.filter(pl.col("symbol") == s), s, "dev") for s in d.SYMBOLS
                      if ev.filter(pl.col("symbol") == s).height])
    for lat in LATENCIES_MS:
        tape = tape.with_columns(net_now(tape, lat).alias(f"netnow_{lat}"))
    tape.write_parquet(OUT / "dev_tape.parquet")

    rows = []
    for half in (0, 1, None):
        g = tape if half is None else tape.filter(pl.col("half") == half)
        label = {0: "A", 1: "B", None: "all"}[half]
        rows.append({"half": label, "measure": "bar model net_now (H15)", **day_stats(g, "bar_net_now")})
        for lat in LATENCIES_MS:
            rows.append({"half": label, "measure": f"tick net_now {lat}ms", **day_stats(g, f"netnow_{lat}")})
            rows.append({"half": label, "measure": f"tick dev-era fills {lat}ms", **day_stats(g, f"fill_{lat}")})
        rows.append({"half": label, "measure": "bar mid", **day_stats(g, "bar_mid")})
        rows.append({"half": label, "measure": "tick mid 0ms", **day_stats(g, "mid_0")})
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "dev_summary.csv")
    by_sym = tape.group_by("symbol").agg(
        n=pl.len(), bar_mid=pl.col("bar_mid").mean(), tick_mid_0=pl.col("mid_0").mean(),
        tick_mid_250=pl.col("mid_250").mean(), bar_net_now=pl.col("bar_net_now").mean(),
        tick_net_now_250=pl.col("netnow_250").mean())
    b = next(r for r in rows if r["half"] == "B" and r["measure"] == f"tick net_now {PRIMARY_MS}ms")
    expected = b["t_day"] * math.sqrt(VAL_EVENT_DAYS / b["days"])
    gate = {"half_B_t": b["t_day"], "half_B_days": b["days"], "expected_val_t": expected,
            "gate": expected >= d.POWER_T}
    (OUT / "dev_gate.json").write_text(json.dumps(gate, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(table)
        print(by_sym)
    print(json.dumps(gate, indent=2))


def validate() -> None:
    gate = json.loads((OUT / "dev_gate.json").read_text())
    if not gate["gate"]:
        print(f"dev gate failed (expected t {gate['expected_val_t']:.2f}); validation not read")
        return
    state = json.loads((b6.OUT / "dev_state.json").read_text())
    ev, _ = cell_events("validation", state)
    days = ev["tday"].n_unique()
    expected = gate["half_B_t"] * math.sqrt(days / gate["half_B_days"])
    print(f"validation event days {days}, expected t {expected:.2f}")
    if expected < d.POWER_T:
        (OUT / "validation.json").write_text(json.dumps({"expected_t": expected, "verdict": "not taken"}))
        return
    tape = pl.concat([fill(ev.filter(pl.col("symbol") == s), s, "validation") for s in d.SYMBOLS
                      if ev.filter(pl.col("symbol") == s).height])
    tape.write_parquet(OUT / "validation_tape.parquet")
    res = {"expected_t": expected, **day_stats(tape, f"fill_{PRIMARY_MS}"),
           "mid_per_trade": float(tape[f"mid_{PRIMARY_MS}"].mean())}
    res["stress_1000ms"] = day_stats(tape, "fill_1000")
    res["verdict"] = "PASS" if res["t_day"] >= d.T_B and res["per_trade"] > 0 else "FAIL"
    (OUT / "validation.json").write_text(json.dumps(res, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(tape.group_by("symbol").agg(n=pl.len(), mid=pl.col(f"mid_{PRIMARY_MS}").mean(),
                                          fill=pl.col(f"fill_{PRIMARY_MS}").mean()))
    print(json.dumps(res, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["dev", "validate"])
    args = p.parse_args()
    dev() if args.mode == "dev" else validate()


if __name__ == "__main__":
    main()
