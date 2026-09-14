"""Discovery batch 3: spread shocks (H08), activity-conditioned reversal (H09),
weekend reopen reversal (H10).

Registered in docs/findings/discovery-program.md before this script was run,
including the power rule that decides whether a validation shot is taken.

    python scripts/research/discovery_batch3.py dev         # all primaries, dev, gates
    python scripts/research/discovery_batch3.py validate    # only gated cells, once

Diurnal factors, tercile cut-offs and sigma scales come from dev and are applied
unchanged to validation. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from discovery_batch1 import (  # noqa: E402
    SYMBOLS, cell, daily_sums, first_crossings, move_z, placebo_band, welch,
)
from discovery_batch2 import diurnal_from_dev, prepare  # noqa: E402

from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import daily_pnl, deflated_threshold, forward_returns  # noqa: E402
from qlab.macro_calendar import build_calendar  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

OUT = Path("reports/discovery/batch3")
N_PRIMARY = 9
THRESHOLD = deflated_threshold(N_PRIMARY)
H08_HORIZONS = (15, 30, 60, 120)
H09_HORIZONS = (15, 30, 60)
H10_EXITS_H = (4, 8, 24)


def calendar() -> pl.DataFrame:
    return (
        build_calendar(date(2020, 1, 1), date(2025, 7, 1))
        .select(pl.col("ts").alias("rel_ts")).unique().sort("rel_ts")
    )


def tag_sched(events: pl.DataFrame, start_col: str, cal: pl.DataFrame) -> pl.DataFrame:
    """True if a scheduled release falls in [start - 10 min, ts + 1 min]."""
    probed = events.with_columns(_probe=pl.col("ts") + timedelta(minutes=1)).sort("_probe")
    joined = probed.join_asof(cal, left_on="_probe", right_on="rel_ts", strategy="backward")
    return joined.with_columns(
        sched=(pl.col("rel_ts") >= pl.col(start_col) - timedelta(minutes=10)).fill_null(False)
    ).drop("_probe", "rel_ts").sort("ts")


# --- H08 ------------------------------------------------------------------


def h08_events(bars: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    b = bars.with_columns(
        s_bps=pl.col("spread_mean") / pl.col("close") * 1e4,
        sc_bps=pl.col("spread_close") / pl.col("close") * 1e4,
        _gap=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes(),
    ).with_columns(
        base=pl.col("s_bps").shift(1).rolling_mean(1440, min_samples=720),
        _reopen=pl.when(pl.col("_gap") >= 30).then(pl.col("ts")).forward_fill(),
    ).with_columns(
        since_reopen=(pl.col("ts") - pl.col("_reopen")).dt.total_minutes(),
    )
    hour = b["ts_open"].dt.hour().to_numpy()
    weekday = b["ts_open"].dt.weekday().to_numpy()
    s = b["s_bps"].to_numpy()
    base = b["base"].to_numpy()
    since = b["since_reopen"].fill_null(10_000).to_numpy()
    flag = (
        np.isfinite(base) & (s >= np.maximum(5.0 * base, 0.5))
        & ~np.isin(hour, (21, 22)) & (weekday != 7) & (since >= 60)
    )
    shocks = first_crossings(b["ts"], flag, 60)

    sc = b["sc_bps"].to_numpy()
    lc = b["lc"].to_numpy()
    sig = b["sig"].to_numpy()
    d = b["d"].to_numpy()
    stamps = b["ts"].dt.epoch("us").to_numpy()
    minute = 60_000_000
    rows = []
    for i in shocks:
        if i < 5 or i + 15 >= lc.size:
            continue
        window = np.arange(i + 1, i + 16)
        ok = (stamps[window] - stamps[i]) == (window - i) * minute
        calm = np.flatnonzero(ok & (sc[window] <= 2.0 * base[i]))
        if calm.size == 0:
            continue
        j = window[calm[0]]
        if stamps[j] - stamps[i - 5] != (j - i + 5) * minute:
            continue
        disp = lc[j] - lc[i - 5]
        if disp == 0 or not np.isfinite(sig[i - 5]):
            continue
        rows.append({
            "ts": b["ts"][int(j)],
            "shock_ts": b["ts"][int(i)],
            "direction": -int(np.sign(disp)),
            "disp_bps": disp * 1e4,
            "disp_z": abs(disp) / (sig[i - 5] * d[i] * math.sqrt(j - i + 5)),
            "shock_bps": s[i],
        })
    if not rows:
        return pl.DataFrame()
    ev = pl.DataFrame(rows).with_columns(pl.col("direction").cast(pl.Int8))
    return tag_sched(ev, "shock_ts", cal)


# --- H09 ------------------------------------------------------------------


def h09_events(bars: pl.DataFrame) -> pl.DataFrame:
    b = move_z(bars, 15).with_columns(
        tick15=pl.when(
            (pl.col("ts") - pl.col("ts").shift(14)).dt.total_minutes() == 14
        ).then(pl.col("n_ticks").cast(pl.Float64).rolling_sum(15)),
    )
    grid = b.filter(pl.col("ts").dt.minute() % 15 == 0).sort("ts").with_columns(
        base=pl.col("tick15").shift(1).rolling_mean(20, min_samples=10).over("bucket"),
    ).with_columns(logA=(pl.col("tick15") / pl.col("base")).log())
    z = grid["z15"].to_numpy()
    la = grid["logA"].to_numpy()
    flag = np.isfinite(z) & np.isfinite(la) & (np.abs(z) >= 2.0)
    idx = first_crossings(grid["ts"], flag, 30)
    return pl.DataFrame({
        "ts": grid["ts"].gather(idx),
        "direction": (-np.sign(z[idx])).astype(np.int8),
        "logA": la[idx],
        "z": z[idx],
    })


# --- H10 ------------------------------------------------------------------


def h10_trades(symbol: str, bars: pl.DataFrame, cost: CostModel) -> pl.DataFrame:
    spec = get_spec(symbol)
    b = bars.select("ts", "close", "bid_close", "ask_close").sort("ts")
    gaps = b.with_columns(
        prev_close=pl.col("close").shift(1),
        gap_h=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_hours(),
    ).filter(pl.col("gap_h") >= 24)
    if gaps.is_empty():
        return pl.DataFrame()
    quote = b.rename({"ts": "q_ts"})

    def at(instants: pl.Series, tag: str) -> pl.DataFrame:
        frame = pl.DataFrame({"at": instants}).with_row_index("i").sort("at")
        got = frame.join_asof(
            quote, left_on="at", right_on="q_ts", strategy="backward",
            tolerance=timedelta(minutes=5),
        ).sort("i")
        return got.select(
            pl.col("close").alias(f"mid_{tag}"),
            pl.col("bid_close").alias(f"bid_{tag}"),
            pl.col("ask_close").alias(f"ask_{tag}"),
        )

    entry = gaps["ts"] + timedelta(minutes=60)
    frames = []
    for hours in H10_EXITS_H:
        e = at(entry, "0")
        x = at(entry + timedelta(hours=hours), "1")
        df = pl.concat([
            pl.DataFrame({"ts": entry, "fri_close": gaps["prev_close"]}), e, x,
        ], how="horizontal").drop_nulls()
        if df.is_empty():
            continue
        df = df.with_columns(
            gap_bps=(pl.col("mid_0") / pl.col("fri_close") - 1) * 1e4,
        ).filter(pl.col("gap_bps") != 0).with_columns(
            direction=(-pl.col("gap_bps").sign()).cast(pl.Int8),
        )
        dirf = pl.col("direction").cast(pl.Float64)
        df = df.with_columns(
            hours=pl.lit(hours),
            raw_bps=(pl.col("mid_1") / pl.col("mid_0") - 1) * 1e4,
            mid_bps=dirf * (pl.col("mid_1") / pl.col("mid_0") - 1) * 1e4,
            gross_bps=pl.when(pl.col("direction") == 1)
            .then(pl.col("bid_1") / pl.col("ask_0") - 1)
            .otherwise(pl.col("bid_0") / pl.col("ask_1") - 1) * 1e4,
        )
        hour = df["ts"].dt.hour().to_numpy()
        price = df["mid_0"].to_numpy()
        comm = np.array([cost.commission_pips(p) * spec.pip / p * 1e4 for p in price])
        slip = np.array([2.0 * cost.slippage_pips(int(h)) * spec.pip / p * 1e4 for h, p in zip(hour, price)])
        frames.append(df.with_columns(
            cost_bps=pl.Series(comm + slip),
            fill_bps=pl.col("gross_bps") - pl.Series(comm),
        ).with_columns(net_bps=pl.col("fill_bps") - pl.Series(slip), symbol=pl.lit(symbol)))
    return pl.concat(frames) if frames else pl.DataFrame()


def demeaned(fr: pl.DataFrame) -> pl.DataFrame:
    """Subtract each symbol's mean unsigned return, signed by the trade."""
    return fr.with_columns(
        mid_bps=pl.col("mid_bps")
        - pl.col("direction").cast(pl.Float64) * pl.col("raw_bps").mean().over("symbol", "hours"),
    )


# --- driver ---------------------------------------------------------------


def run_split(split: str, diurnals: dict, cutoffs: dict | None, seeds: int, only: set | None):
    """Every cell on one split. Returns (rows, primaries, cutoffs, day counts)."""
    cal = calendar()
    rows, primaries, days = [], [], {}
    cutoffs = dict(cutoffs or {})
    h10 = []
    for symbol in SYMBOLS:
        bars = prepare(symbol, split, diurnals[symbol])
        cost = CostModel.from_profiles(symbol, split=split)
        print(split, symbol, flush=True)

        if only is None or "H08" in only:
            ev = h08_events(bars, cal)
            days[f"H08 {symbol}"] = ev["ts"].dt.date().n_unique() if ev.height else 0
            fr = forward_returns(bars, ev, H08_HORIZONS, cost=cost,
                                 keep=("disp_z", "sched")) if ev.height else pl.DataFrame()
            for (m,), g in sorted(fr.group_by("minutes"), key=lambda kv: kv[0][0]) if fr.height else []:
                rows.append({"hyp": "H08", "symbol": symbol, "minutes": int(m), **cell(g)})
            if fr.height:
                g = fr.filter(pl.col("minutes") == 30)
                cut = np.nanquantile(ev["disp_z"].to_numpy(), [1 / 3, 2 / 3])
                for t, gg in enumerate((g.filter(pl.col("disp_z") < cut[0]),
                                        g.filter((pl.col("disp_z") >= cut[0]) & (pl.col("disp_z") < cut[1])),
                                        g.filter(pl.col("disp_z") >= cut[1]))):
                    rows.append({"hyp": "H08-dose", "symbol": symbol, "minutes": 30, "tercile": t, **cell(gg)})
                for flag_val in (True, False):
                    rows.append({"hyp": "H08-sched", "symbol": symbol, "minutes": 30, "sched": flag_val,
                                 **cell(g.filter(pl.col("sched") == flag_val))})
                p = {"cell": f"H08 {symbol} 30m", **cell(g)}
                if split == "dev":
                    p.update(placebo_band(ev, bars, cost, 30, seeds))
                primaries.append(p)

        if only is None or "H09" in only:
            ev = h09_events(bars)
            if split == "dev":
                cutoffs[symbol] = np.nanquantile(ev["logA"].to_numpy(), [1 / 3, 2 / 3]).tolist()
            lo, hi = cutoffs[symbol]
            ev = ev.with_columns(
                tercile=pl.when(pl.col("logA") < lo).then(0).when(pl.col("logA") < hi).then(1).otherwise(2)
            )
            days[f"H09 {symbol}"] = ev["ts"].dt.date().n_unique()
            fr = forward_returns(bars, ev, H09_HORIZONS, cost=cost, keep=("tercile", "logA"))
            for (m, t), g in sorted(fr.group_by("minutes", "tercile"), key=lambda kv: kv[0]):
                rows.append({"hyp": "H09", "symbol": symbol, "minutes": int(m), "tercile": int(t), **cell(g)})
            g = fr.filter(pl.col("minutes") == 30)
            high, low = g.filter(pl.col("tercile") == 2), g.filter(pl.col("tercile") == 0)
            dh, dl = daily_sums(high, "mid_bps"), daily_sums(low, "mid_bps")
            primaries.append({
                "cell": f"H09 {symbol} high-low 30m",
                "n": high.height + low.height,
                "mid_high": float(high["mid_bps"].mean()), "mid_low": float(low["mid_bps"].mean()),
                "net_high": float(high["net_bps"].mean()), "net_low": float(low["net_bps"].mean()),
                "mid": float(high["mid_bps"].mean() - low["mid_bps"].mean()),
                "t_day_mid": welch(dh, dl),
                "sd_day": float(np.sqrt(dh.var(ddof=1) / 1 + dl.var(ddof=1))),
                "days_high": int(dh.size), "days_low": int(dl.size),
            })

        if only is None or "H10" in only:
            tr = h10_trades(symbol, bars, cost)
            if tr.height:
                h10.append(tr)

    if h10:
        tr = pl.concat(h10)
        for label, frame in (("raw", tr), ("demeaned", demeaned(tr))):
            for (h,), g in sorted(frame.group_by("hours"), key=lambda kv: kv[0][0]):
                rows.append({"hyp": f"H10-{label}", "symbol": "pooled", "minutes": int(h) * 60, **cell(g)})
                for (s,), gg in sorted(g.group_by("symbol"), key=lambda kv: kv[0][0]):
                    rows.append({"hyp": f"H10-{label}", "symbol": s, "minutes": int(h) * 60, **cell(gg)})
        g = tr.filter(pl.col("hours") == 8)
        p = {"cell": "H10 pooled 8h", **cell(g)}
        dm = demeaned(tr).filter(pl.col("hours") == 8)
        p["mid_demeaned"], p["t_day_mid_demeaned"], _ = daily_pnl(dm, column="mid_bps")
        primaries.append(p)
        days["H10 pooled"] = g["ts"].dt.date().n_unique()
    return rows, primaries, cutoffs, days


def gate(row: dict) -> bool:
    t = row.get("t_day_mid", float("nan"))
    if not (math.isfinite(t) and abs(t) >= THRESHOLD):
        return False
    if "pl_mid_lo" in row and row["pl_mid_lo"] <= row["mid"] <= row["pl_mid_hi"]:
        return False
    if row["cell"].startswith("H10") and abs(row.get("t_day_mid_demeaned", 0.0)) < THRESHOLD:
        return False
    return True


def dev(seeds: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diurnals = {s: diurnal_from_dev(s) for s in SYMBOLS}
    rows, prim, cutoffs, days = run_split("dev", diurnals, None, seeds, None)
    for p in prim:
        p["gate"] = gate(p)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "dev_summary.csv")
    pl.DataFrame(prim, infer_schema_length=None).write_csv(OUT / "dev_primary.csv")
    (OUT / "dev_state.json").write_text(json.dumps(
        {"threshold": THRESHOLD, "cutoffs": cutoffs, "days": days, "primaries": prim},
        indent=2, default=str))
    show(rows, prim)


def show(rows, prim) -> None:
    table = pl.DataFrame(rows, infer_schema_length=None)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=240, float_precision=2):
        print(f"\nSidak |t| over {N_PRIMARY} primary cells: {THRESHOLD:.2f}")
        cols = [c for c in ("cell", "n", "days", "mid", "net", "cost", "t_day_mid", "t_day_net",
                            "pl_mid_lo", "pl_mid_hi", "mid_high", "mid_low", "net_high", "net_low",
                            "mid_demeaned", "t_day_mid_demeaned", "gate")
                if c in pl.DataFrame(prim, infer_schema_length=None).columns]
        print(pl.DataFrame(prim, infer_schema_length=None).select(cols))
        for hyp in ("H08", "H08-dose", "H08-sched", "H09", "H10-raw", "H10-demeaned"):
            sub = table.filter(pl.col("hyp") == hyp)
            keep = [c for c in ("symbol", "minutes", "tercile", "sched", "n", "mid", "net",
                                "t_day_mid", "t_day_net", "hit_mid") if c in sub.columns]
            print(f"\n{hyp}")
            print(sub.select(keep))


def validate() -> None:
    state = json.loads((OUT / "dev_state.json").read_text())
    passed = [p for p in state["primaries"] if p["gate"]]
    if not passed:
        print("no primary cell passed dev - validation not read")
        return
    raise SystemExit(
        "cells passed dev: " + ", ".join(p["cell"] for p in passed)
        + "\nApply the registered power rule to these before writing the validation run."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dev", "validate"])
    parser.add_argument("--placebos", type=int, default=20)
    args = parser.parse_args()
    dev(args.placebos) if args.mode == "dev" else validate()


if __name__ == "__main__":
    main()
