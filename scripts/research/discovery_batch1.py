"""Discovery batch 1: tail-conditioned structure, pre-registered.

H01-H04 exactly as registered in docs/findings/discovery-program.md, which was
written before this script was first run. Dev split only - validation and test
are never read here.

H01  large-move response: |z_15| >= k, forward return signed by the move
H02  H01 split by whether a scheduled US release sits inside the move
H03  cross-market shock: source |z_5| >= 4, target trades its beta shortfall
H04  compression -> expansion: first break of a bottom-decile 60-minute range

Primary cells (24) are judged at the mid on the per-day t against the Sidak
threshold for 24 trials, and against a date-shifted placebo band. Every other
number printed is exploratory.

    python scripts/research/discovery_batch1.py [--placebos 20]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qlab.costs import CostModel
from qlab.eventstudy import daily_pnl, deflated_threshold, forward_returns, placebo_events
from qlab.loader import load_bars
from qlab.macro_calendar import build_calendar
from qlab.stats import newey_west

SYMBOLS = ("EURUSD", "USDJPY", "XAUUSD", "USTEC")
SPLIT = "dev"
OUT = Path("reports/discovery/batch1")
SIGMA_WINDOW = 1440
N_PRIMARY = 24
THRESHOLD = deflated_threshold(N_PRIMARY)

H01_KS = (3.0, 4.0, 5.0)
H01_HORIZONS = (5, 15, 30, 60, 120, 240)
H03_HORIZONS = (1, 5, 15, 30, 60)
H04_HORIZONS = (15, 60, 240)


# --- preparation ----------------------------------------------------------


def prepare(symbol: str) -> pl.DataFrame:
    """1m dev bars with deseasonalised trailing sigma and the diurnal factor.

    ``d`` is the RMS 1-minute return of the bar's 15-minute UTC bucket relative
    to the whole sample - an in-sample scale, stated as such in the registry.
    ``sig`` is the trailing std of ``r1 / d`` and describes volatility per unit
    of diurnal scale, so the expected sd of a W-minute window is
    ``sig * sqrt(sum of d^2 over the window)``.
    """
    bars = load_bars(symbol, "1m", split=SPLIT).sort("ts")
    bars = bars.with_columns(
        lc=pl.col("close").log(),
        _gap=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes(),
        bucket=pl.col("ts_open").dt.hour().cast(pl.Int32) * 4
        + pl.col("ts_open").dt.minute().cast(pl.Int32) // 15,
    ).with_columns(
        r1=pl.when(pl.col("_gap") == 1).then(pl.col("lc") - pl.col("lc").shift(1)),
    )
    overall = bars.select((pl.col("r1") ** 2).mean()).item()
    diurnal = bars.group_by("bucket").agg(
        d=((pl.col("r1") ** 2).mean() / overall).sqrt()
    )
    return (
        bars.join(diurnal, on="bucket", how="left")
        .sort("ts")
        .with_columns(
            sig=(pl.col("r1") / pl.col("d")).rolling_std(
                window_size=SIGMA_WINDOW, min_samples=SIGMA_WINDOW // 2
            ),
            d2=pl.col("d") ** 2,
        )
        .drop("_gap")
    )


def move_z(bars: pl.DataFrame, window: int) -> pl.DataFrame:
    """Add ``r{W}`` and ``z{W}``: the W-minute log move and its z-score.

    Null unless the W bars are contiguous minutes, so no window spans a halt.
    Sigma is taken from before the window starts, so the move cannot inflate
    its own denominator.
    """
    contiguous = (pl.col("ts") - pl.col("ts").shift(window)).dt.total_minutes() == window
    r = pl.when(contiguous).then(pl.col("lc") - pl.col("lc").shift(window))
    expected = pl.col("sig").shift(window) * pl.col("d2").rolling_sum(window).sqrt()
    return bars.with_columns(r.alias(f"r{window}")).with_columns(
        (pl.col(f"r{window}") / expected).alias(f"z{window}")
    )


def first_crossings(ts: pl.Series, flag: np.ndarray, cooldown_min: int) -> np.ndarray:
    """Indices where ``flag`` is set and no kept event lies within the cooldown."""
    stamps = ts.dt.epoch("us").to_numpy()
    gap = cooldown_min * 60_000_000
    keep, last = [], -(10**18)
    for i in np.flatnonzero(flag):
        if stamps[i] - last >= gap:
            keep.append(i)
            last = stamps[i]
    return np.asarray(keep, dtype=np.int64)


def mfe_mae(bars: pl.DataFrame, idx: np.ndarray, direction: np.ndarray, horizon: int):
    """Mean max-favourable and max-adverse signed mid excursion, bps, contiguous paths only."""
    close = bars["close"].to_numpy()
    stamps = bars["ts"].dt.epoch("us").to_numpy()
    ok = idx + horizon < close.size
    idx, direction = idx[ok], direction[ok]
    ok = (stamps[idx + horizon] - stamps[idx]) == horizon * 60_000_000
    idx, direction = idx[ok], direction[ok]
    if idx.size == 0:
        return float("nan"), float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(close, horizon + 1)[idx][:, 1:]
    signed = direction[:, None] * (windows / close[idx, None] - 1.0) * 1e4
    return float(signed.max(axis=1).mean()), float(signed.min(axis=1).mean())


# --- statistics -----------------------------------------------------------


def cell(fr: pl.DataFrame) -> dict:
    if fr.is_empty():
        return {"n": 0}
    mid = fr["mid_bps"].to_numpy()
    net = fr["net_bps"].to_numpy()
    _, t_mid, days = daily_pnl(fr, column="mid_bps")
    _, t_net, _ = daily_pnl(fr, column="net_bps")
    nw = newey_west(net)
    sd = mid.std(ddof=1) if mid.size > 1 else float("nan")
    return {
        "n": int(mid.size),
        "days": days,
        "mid": float(mid.mean()),
        "fill": float(fr["fill_bps"].mean()),
        "net": float(net.mean()),
        "cost": float(fr["cost_bps"].mean()),
        "t_day_mid": t_mid,
        "t_day_net": t_net,
        "net_ci_lo": nw.mean - 1.96 * nw.se,
        "net_ci_hi": nw.mean + 1.96 * nw.se,
        "hit_mid": float((mid > 0).mean()),
        "effect_d": float(mid.mean() / sd) if sd else float("nan"),
    }


def by_horizon(fr: pl.DataFrame, tags: dict) -> list[dict]:
    rows = []
    for (minutes,), g in sorted(fr.group_by("minutes"), key=lambda kv: kv[0][0]):
        rows.append({**tags, "minutes": int(minutes), **cell(g)})
    return rows


def placebo_band(events, bars, cost, minutes, seeds) -> dict:
    mids, nets = [], []
    for seed in range(seeds):
        shifted = placebo_events(events.select("ts", "direction"), bars, seed=seed)
        fr = forward_returns(bars, shifted, (minutes,), cost=cost)
        if not fr.is_empty():
            mids.append(fr["mid_bps"].mean())
            nets.append(fr["net_bps"].mean())
    mids = np.asarray(mids)
    return {
        "pl_mid_mean": float(mids.mean()) if mids.size else float("nan"),
        "pl_mid_lo": float(mids.min()) if mids.size else float("nan"),
        "pl_mid_hi": float(mids.max()) if mids.size else float("nan"),
        "pl_n": int(mids.size),
    }


def classify(row: dict) -> str:
    """Primary-cell label under the registry's ladder (validation not read here)."""
    t = row.get("t_day_mid", float("nan"))
    if not math.isfinite(t) or abs(t) < THRESHOLD:
        return "OBSERVATION"
    outside = row["mid"] > row["pl_mid_hi"] or row["mid"] < row["pl_mid_lo"]
    if not outside:
        return "OBSERVATION (inside placebo)"
    return "EXPLOITABLE (dev)" if row.get("tradable_net", -1.0) > 0 else "PREDICTIVE"


def welch(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))


def daily_sums(fr: pl.DataFrame, column: str) -> np.ndarray:
    return (
        fr.group_by(pl.col("ts").dt.date())
        .agg(pl.col(column).sum())[column]
        .to_numpy()
    )


# --- hypotheses -----------------------------------------------------------


def tag_scheduled(events: pl.DataFrame, window: int) -> pl.DataFrame:
    cal = (
        build_calendar(date(2020, 1, 1), date(2024, 1, 1))
        .select(pl.col("ts").alias("rel_ts"))
        .unique()
        .sort("rel_ts")
    )
    probed = events.with_columns(_probe=pl.col("ts") + timedelta(minutes=1)).sort("_probe")
    joined = probed.join_asof(cal, left_on="_probe", right_on="rel_ts", strategy="backward")
    return joined.with_columns(
        sched=(pl.col("rel_ts") >= pl.col("ts") - timedelta(minutes=window + 2)).fill_null(False)
    ).drop("_probe", "rel_ts").sort("ts")


def h01_h02(symbol, bars, cost, seeds, summary, primaries, tapes):
    bars = move_z(bars, 15)
    z = bars["z15"].to_numpy()
    sig = bars["sig"].to_numpy()
    cut = np.nanquantile(sig, [1 / 3, 2 / 3])
    for k in H01_KS:
        flag = np.isfinite(z) & (np.abs(z) >= k)
        idx = first_crossings(bars["ts"], flag, 60)
        direction = np.sign(z[idx]).astype(np.int8)
        events = pl.DataFrame(
            {
                "ts": bars["ts"].gather(idx),
                "direction": direction,
                "z": z[idx],
                "regime": np.digitize(sig[idx], cut).astype(np.int8),
            }
        )
        events = tag_scheduled(events, 15)
        fr = forward_returns(bars, events, H01_HORIZONS, cost=cost, keep=("z", "regime", "sched"))
        fade = forward_returns(
            bars, events.with_columns(direction=-pl.col("direction")), H01_HORIZONS, cost=cost
        )
        tapes.append(fr.with_columns(symbol=pl.lit(symbol), hyp=pl.lit("H01"), k=pl.lit(k)))
        for row in by_horizon(fr, {"hyp": "H01", "symbol": symbol, "k": k}):
            f = fade.filter(pl.col("minutes") == row["minutes"])
            row["fade_net"] = float(f["net_bps"].mean()) if f.height else float("nan")
            summary.append(row)
        if k != 4.0:
            continue

        # Primary H01: k = 4, 60 minutes, two-sided at the mid.
        p = {**next(r for r in summary if r["hyp"] == "H01" and r["symbol"] == symbol
                    and r["k"] == 4.0 and r["minutes"] == 60)}
        p.update(placebo_band(events, bars, cost, 60, seeds))
        p["mfe60"], p["mae60"] = mfe_mae(bars, idx, direction.astype(float), 60)
        p["tradable_net"] = p["net"] if p["mid"] > 0 else p["fade_net"]
        p["label"] = classify(p)
        p["cell"] = f"H01 {symbol} k4 60m"
        primaries.append(p)

        # Exploratory: regime and year at the primary horizon.
        g = fr.filter(pl.col("minutes") == 60)
        for (reg,), gg in sorted(g.group_by("regime"), key=lambda kv: kv[0][0]):
            summary.append({"hyp": "H01-regime", "symbol": symbol, "k": k, "minutes": 60,
                            "regime": int(reg), **cell(gg)})
        for (yr,), gg in sorted(g.group_by(pl.col("ts").dt.year()), key=lambda kv: kv[0][0]):
            summary.append({"hyp": "H01-year", "symbol": symbol, "k": k, "minutes": 60,
                            "year": int(yr), **cell(gg)})

        # H02: the same events, scheduled against unscheduled.
        for flag_val, name in ((True, "scheduled"), (False, "unscheduled")):
            gg = g.filter(pl.col("sched") == flag_val)
            summary.append({"hyp": "H02", "symbol": symbol, "k": k, "minutes": 60,
                            "group": name, **cell(gg)})
        u = g.filter(~pl.col("sched"))
        s = g.filter(pl.col("sched"))
        diff = {
            "cell": f"H02 {symbol} unsched-sched 60m",
            "n_unsched": u.height,
            "n_sched": s.height,
            "mid_unsched": float(u["mid_bps"].mean()) if u.height else float("nan"),
            "mid_sched": float(s["mid_bps"].mean()) if s.height else float("nan"),
            "t_welch_daily_mid": welch(daily_sums(u, "mid_bps"), daily_sums(s, "mid_bps")),
        }
        diff["mid"] = diff["mid_unsched"] - diff["mid_sched"]
        t = diff["t_welch_daily_mid"]
        diff["label"] = (
            "PREDICTIVE" if math.isfinite(t) and t <= -THRESHOLD
            else "OBSERVATION"
        )
        primaries.append(diff)


def pair_beta(src: pl.DataFrame, tgt: pl.DataFrame) -> float:
    joined = (
        src.select("ts", pl.col("r5").alias("rs"))
        .join(tgt.select("ts", pl.col("r5").alias("rt")), on="ts")
        .filter((pl.col("ts").dt.minute() % 5 == 0))
        .drop_nulls()
    )
    rs, rt = joined["rs"].to_numpy(), joined["rt"].to_numpy()
    rs, rt = rs - rs.mean(), rt - rt.mean()
    return float((rs * rt).sum() / (rs * rs).sum())


def h03(prepared, costs, seeds, summary, primaries, tapes):
    frames = {s: move_z(b, 5) for s, b in prepared.items()}
    for source in SYMBOLS:
        src = frames[source]
        z = src["z5"].to_numpy()
        idx = first_crossings(src["ts"], np.isfinite(z) & (np.abs(z) >= 4.0), 30)
        shocks = pl.DataFrame({"ts": src["ts"].gather(idx), "rs": src["r5"].gather(idx)})
        for target in SYMBOLS:
            if target == source:
                continue
            tgt = frames[target]
            beta = pair_beta(src, tgt)
            ev = shocks.join(tgt.select("ts", pl.col("r5").alias("rt")), on="ts").drop_nulls()
            shortfall = beta * ev["rs"] - ev["rt"]
            ev = ev.with_columns(
                direction=pl.Series(np.sign(shortfall.to_numpy())).cast(pl.Int8),
                plain=pl.Series(np.sign((beta * ev["rs"]).to_numpy())).cast(pl.Int8),
                shortfall_bps=shortfall * 1e4,
            ).filter(pl.col("direction") != 0)
            tag = {"hyp": "H03", "symbol": f"{source}->{target}", "beta": beta}
            fr = forward_returns(tgt, ev, H03_HORIZONS, cost=costs[target], keep=("shortfall_bps",))
            tapes.append(fr.with_columns(symbol=pl.lit(f"{source}->{target}"), hyp=pl.lit("H03")))
            rows = by_horizon(fr, {**tag, "variant": "shortfall"})
            summary.extend(rows)
            plain = forward_returns(
                tgt, ev.with_columns(direction=pl.col("plain")), H03_HORIZONS, cost=costs[target]
            )
            summary.extend(by_horizon(plain, {**tag, "variant": "plain"}))

            p = {**next(r for r in rows if r["minutes"] == 15)}
            p.update(placebo_band(ev, tgt, costs[target], 15, seeds))
            p["tradable_net"] = p["net"]
            p["label"] = classify(p)
            p["cell"] = f"H03 {source}->{target} 15m"
            primaries.append(p)


def compression_events(bars: pl.DataFrame, lo_q: float, hi_q: float, q: np.ndarray):
    """Breakout events from ranges whose z-width lies in [q[lo], q[hi]]."""
    close = bars["close"].to_numpy()
    width = bars["cz"].to_numpy()
    hi_arr = bars["rhi"].to_numpy()
    lo_arr = bars["rlo"].to_numpy()
    stamps = bars["ts"].dt.epoch("us").to_numpy()
    flag = np.isfinite(width) & (width >= lo_q) & (width <= hi_q)
    starts = first_crossings(bars["ts"], flag, 60)
    ev_idx, ev_dir, fwd_range = [], [], []
    minute = 60_000_000
    for t0 in starts:
        end = t0 + 60
        if end >= close.size or stamps[end] - stamps[t0] != 60 * minute:
            continue
        path = close[t0 + 1 : end + 1]
        up = path > hi_arr[t0]
        dn = path < lo_arr[t0]
        hit = np.flatnonzero(up | dn)
        # Expansion observation: forward 60-minute range in the window's own z units.
        fr_width = (bars["high"][t0 + 1 : end + 1].max() - bars["low"][t0 + 1 : end + 1].min())
        fwd_range.append(fr_width / (hi_arr[t0] - lo_arr[t0]) if hi_arr[t0] > lo_arr[t0] else np.nan)
        if hit.size == 0:
            continue
        j = t0 + 1 + hit[0]
        ev_idx.append(j)
        ev_dir.append(1 if up[hit[0]] else -1)
    return np.asarray(ev_idx, dtype=np.int64), np.asarray(ev_dir, dtype=np.int8), np.asarray(fwd_range)


def h04(symbol, bars, cost, seeds, summary, primaries, tapes):
    contiguous = (pl.col("ts") - pl.col("ts").shift(59)).dt.total_minutes() == 59
    bars = bars.with_columns(
        rhi=pl.col("high").rolling_max(60),
        rlo=pl.col("low").rolling_min(60),
    ).with_columns(
        cz=pl.when(contiguous).then(
            ((pl.col("rhi") - pl.col("rlo")) / pl.col("close"))
            / (pl.col("sig").shift(60) * pl.col("d2").rolling_sum(60).sqrt())
        )
    )
    width = bars["cz"].to_numpy()
    q = np.nanquantile(width, [0.10, 0.40, 0.60])
    groups = {"compressed": (0.0, q[0]), "control": (q[1], q[2])}
    results = {}
    for name, (lo, hi) in groups.items():
        idx, direction, fwd = compression_events(bars, lo, hi, q)
        events = pl.DataFrame({"ts": bars["ts"].gather(idx), "direction": direction})
        fr = forward_returns(bars, events, H04_HORIZONS, cost=cost)
        tapes.append(fr.with_columns(symbol=pl.lit(symbol), hyp=pl.lit(f"H04-{name}")))
        rows = by_horizon(fr, {"hyp": "H04", "symbol": symbol, "group": name})
        for row in rows:
            row["fwd_range_ratio_median"] = float(np.nanmedian(fwd)) if fwd.size else float("nan")
        summary.extend(rows)
        results[name] = (events, fr, idx, direction)

    events, fr, idx, direction = results["compressed"]
    p = {**next(r for r in summary if r["hyp"] == "H04" and r["symbol"] == symbol
                and r.get("group") == "compressed" and r["minutes"] == 60)}
    ctrl = next(r for r in summary if r["hyp"] == "H04" and r["symbol"] == symbol
                and r.get("group") == "control" and r["minutes"] == 60)
    p["control_mid"] = ctrl["mid"]
    p.update(placebo_band(events, bars, cost, 60, seeds))
    p["mfe60"], p["mae60"] = mfe_mae(bars, idx, direction.astype(float), 60)
    p["tradable_net"] = p["net"]
    p["label"] = classify(p)
    if p["label"] != "OBSERVATION" and p["mid"] <= p["control_mid"]:
        p["label"] = "OBSERVATION (not above control)"
    p["cell"] = f"H04 {symbol} compressed 60m"
    primaries.append(p)


# --- driver ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placebos", type=int, default=20)
    parser.add_argument("--only", choices=["H01", "H03", "H04"], default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    prepared = {s: prepare(s) for s in SYMBOLS}
    costs = {s: CostModel.from_profiles(s, split=SPLIT) for s in SYMBOLS}
    summary: list[dict] = []
    primaries: list[dict] = []
    tapes: list[pl.DataFrame] = []

    if args.only in (None, "H01"):
        for s in SYMBOLS:
            print(f"H01/H02 {s}", flush=True)
            h01_h02(s, prepared[s], costs[s], args.placebos, summary, primaries, tapes)
    if args.only in (None, "H03"):
        print("H03", flush=True)
        h03(prepared, costs, args.placebos, summary, primaries, tapes)
    if args.only in (None, "H04"):
        for s in SYMBOLS:
            print(f"H04 {s}", flush=True)
            h04(s, prepared[s], costs[s], args.placebos, summary, primaries, tapes)

    table = pl.DataFrame(summary, infer_schema_length=None)
    prim = pl.DataFrame(primaries, infer_schema_length=None)
    table.write_csv(OUT / "summary.csv")
    prim.write_csv(OUT / "primary.csv")
    if tapes:
        pl.concat(tapes, how="diagonal_relaxed").write_parquet(OUT / "outcomes.parquet")
    (OUT / "meta.json").write_text(json.dumps(
        {"split": SPLIT, "n_primary": N_PRIMARY, "threshold": THRESHOLD,
         "placebos": args.placebos}, indent=2))

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
        print(f"\nSidak |t| threshold over {N_PRIMARY} primary cells: {THRESHOLD:.2f}\n")
        cols = [c for c in ("cell", "n", "days", "mid", "net", "cost", "t_day_mid", "t_day_net",
                            "pl_mid_lo", "pl_mid_hi", "control_mid", "mfe60", "mae60",
                            "mid_unsched", "mid_sched", "t_welch_daily_mid", "label")
                if c in prim.columns]
        print(prim.select(cols))


if __name__ == "__main__":
    main()
