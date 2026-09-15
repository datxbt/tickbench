"""TOP8_2026 on XAUUSD as it would actually be deployed.

Configuration under test, from ``XAUUSD_SessionBreakout_2026.mq5``:

* preset ``TOP8_2026`` - the 2026 hour re-selection
* ``InpUseVolTargeting = false`` - a flat 0.02 lots on every window
* ``InpMaxSpreadUSD = 0.30`` - the literal dollar cap, not a multiple of mean spread

The deploy criteria D1-D5 are registered in
``docs/findings/session-breakout-top8.md`` and were written before this script
produced a number. They are evaluated at the bottom of ``analyse``.

Phases, each caching to ``reports/strategies/session_breakout_top8/``:

  tapes    the preset, plus cost and spread-guard variants and the
           flipped-direction control, in one walk of the ticks per month
  grid     every (hour, bracket, target) cell with the expert's own stop, for
           the selection tests: random-set null and walk-forward re-selection
  analyse  every table, from the caches

Each era is costed with its own measured slippage profile (dev, validation,
test), because gold's latency drift in 2020-2023 is not today's.

Run:  python scripts/backtests/backtest_session_breakout_top8.py [--phase tapes grid analyse]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.costs import CostModel, SlippageModel  # noqa: E402
from qlab.stats import newey_west, paired_bootstrap, sharpe  # noqa: E402
from qlab.strategies.session_breakout import (  # noqa: E402
    PRESETS,
    BreakoutConfig,
    grid,
    run_many,
)

SYMBOL = "XAUUSD"
PRESET = "TOP8_2026"
LOTS = 0.02
SPREAD_CAP_USD = 0.30
OUT = Path("reports/strategies/session_breakout_top8")

# Month-aligned, end exclusive, so each era gets its own cost profile and no
# day is simulated twice.
SEGMENTS = (
    ("dev", date(2020, 1, 29), date(2024, 1, 1)),
    ("validation", date(2024, 1, 1), date(2025, 7, 1)),
    ("test", date(2025, 7, 1), date(2026, 9, 2)),
)
# Inclusive day ranges for reporting.
PERIODS = {
    "dev 2020-23": (date(2020, 1, 29), date(2023, 12, 31)),
    "val 2024-25H1": (date(2024, 1, 1), date(2025, 6, 30)),
    "dev+val pooled": (date(2020, 1, 29), date(2025, 6, 30)),
    "test 25H2-26": (date(2025, 7, 1), date(2026, 9, 1)),
    "2026 (fit window)": (date(2026, 1, 1), date(2026, 9, 1)),
}
GRID_HOURS = tuple(range(21))
GRID_RANGES = (15, 30, 60)
GRID_TARGETS = (1.0, 2.0, 3.0)
TOP8_CELLS = PRESETS[PRESET]
WF_LOOKBACKS = (6, 12)      # months of history the re-selection is scored on
WF_STEP = 3                 # months each selection is traded before re-picking
WF_FIRST, WF_LAST = date(2021, 1, 1), date(2026, 7, 1)


def base_config() -> BreakoutConfig:
    return BreakoutConfig.preset(PRESET, max_spread_usd=SPREAD_CAP_USD)


def variants(seg: str) -> dict[str, tuple[BreakoutConfig, CostModel]]:
    cost = CostModel.from_profiles(SYMBOL, split=seg)
    slow = CostModel.from_profiles(SYMBOL, split=seg,
                                   slippage=SlippageModel(latency_ms=1000))
    base = base_config()
    return {
        "base": (base, cost),
        "flip": (replace(base, flip_direction=True), cost),
        "cap_off": (replace(base, max_spread_usd=float("inf")), cost),
        "cap_0.20": (replace(base, max_spread_usd=0.20), cost),
        "cap_0.50": (replace(base, max_spread_usd=0.50), cost),
        "cap_3.4x_mean": (replace(base, max_spread_usd=None), cost),
        "adverse_0": (base, cost.stressed(adverse_fraction=0.0)),
        "adverse_1": (base, cost.stressed(adverse_fraction=1.0)),
        "latency_1000ms": (base, slow),
    }


def _in_segment(frame: pl.DataFrame, lo: date, hi: date) -> pl.DataFrame:
    return frame.filter((pl.col("day") >= lo) & (pl.col("day") < hi))


# --------------------------------------------------------------------------
# Phases that walk the ticks
# --------------------------------------------------------------------------

def phase_tapes(verbose: bool) -> None:
    stores: dict[str, list[pl.DataFrame]] = {}
    for seg, lo, hi in SEGMENTS:
        out = run_many(SYMBOL, variants(seg), start=lo, end=hi,
                       allow_test=True, verbose=verbose)
        for name, frame in out.items():
            if not frame.is_empty():
                stores.setdefault(name, []).append(
                    _in_segment(frame, lo, hi).with_columns(segment=pl.lit(seg)))
        print(f"tapes: {seg} done", flush=True)
    (OUT / "tapes").mkdir(parents=True, exist_ok=True)
    for name, frames in stores.items():
        pl.concat(frames, how="vertical_relaxed").sort("entry_ts").write_parquet(
            OUT / "tapes" / f"{name}.parquet")
    print(f"wrote {len(stores)} tapes to {OUT / 'tapes'}")


def phase_grid(verbose: bool) -> None:
    frames = []
    for seg, lo, hi in SEGMENTS:
        cost = CostModel.from_profiles(SYMBOL, split=seg)
        g = grid(SYMBOL, base_config(), GRID_HOURS, GRID_RANGES, GRID_TARGETS,
                 cost=cost, start=lo, end=hi, allow_test=True, verbose=verbose)
        frames.append(_in_segment(g, lo, hi).with_columns(segment=pl.lit(seg)))
        print(f"grid: {seg} done, {g.height} cell-trades", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(OUT / "grid.parquet")
    print(f"wrote {OUT / 'grid.parquet'}")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _slice(frame: pl.DataFrame, period: str) -> pl.DataFrame:
    lo, hi = PERIODS[period]
    return frame.filter(pl.col("day").is_between(lo, hi))


def _daily_usd(tape: pl.DataFrame, usd_col: str = "net_usd") -> pl.DataFrame:
    return (tape.group_by("day").agg(usd=(pl.col(usd_col) * LOTS).sum())
            .sort("day"))


def stats_row(tape: pl.DataFrame, r_col: str = "r_multiple",
              usd_col: str = "net_usd") -> dict:
    """Account-level statistics at fixed ``LOTS``.

    t-statistics are Newey-West on **daily** totals: eight windows on one
    instrument on one day are mostly one bet.
    """
    if tape.is_empty():
        return {"trades": 0}
    t = tape.with_columns(usd=pl.col(usd_col) * LOTS)
    daily = (t.group_by("day").agg(usd=pl.col("usd").sum(), r=pl.col(r_col).sum())
             .sort("day"))
    usd, r = daily["usd"].to_numpy(), daily["r"].to_numpy()
    eq = np.cumsum(usd)
    dd = np.maximum.accumulate(np.maximum(eq, 0.0)) - eq
    days = daily["day"].to_list()
    years = max((days[-1] - days[0]).days / 365.25, 1e-9)

    peak, last_high, underwater = -np.inf, days[0], 0
    for d, v in zip(days, eq):
        if v >= peak:
            peak, last_high = v, d
        underwater = max(underwater, (d - last_high).days)

    monthly = t.group_by(pl.col("day").dt.truncate("1mo")).agg(pl.col("usd").sum())
    gain = float(t.filter(pl.col("usd") > 0)["usd"].sum())
    loss = float(-t.filter(pl.col("usd") <= 0)["usd"].sum())
    return {
        "trades": t.height,
        "days": daily.height,
        "win_pct": 100 * float((t["usd"] > 0).cast(pl.Float64).mean()),
        "mean_r": float(t[r_col].mean()),
        "net_usd": float(usd.sum()),
        "usd_per_yr": float(usd.sum() / years),
        "t_usd": newey_west(usd).t_stat,
        "t_r": newey_west(r).t_stat,
        "sharpe_usd": sharpe(usd),
        "profit_factor": gain / loss if loss > 0 else float("inf"),
        "max_dd_usd": float(dd.max()),
        "longest_dd_days": underwater,
        "worst_day_usd": float(usd.min()),
        "worst_month_usd": float(monthly["usd"].min()),
        "months_up_pct": 100 * float((monthly["usd"] > 0).cast(pl.Float64).mean()),
    }


def _frame(rows: list[dict]) -> pl.DataFrame:
    out = pl.DataFrame(rows)
    return out.with_columns(pl.col(pl.Float64).round(3))


def period_table(tape: pl.DataFrame) -> pl.DataFrame:
    return _frame([{"period": p} | stats_row(_slice(tape, p)) for p in PERIODS])


def yearly_table(base: pl.DataFrame) -> pl.DataFrame:
    """Why fixed lots matter: dollar risk per trade follows the bracket width."""
    return (
        base.with_columns(year=pl.col("day").dt.year(),
                          risk=pl.col("risk_actual_usd") * LOTS)
        .group_by("year")
        .agg(
            trades=pl.len(),
            width_usd=pl.col("width").mean(),
            risk_p05=pl.col("risk").quantile(0.05),
            risk_p50=pl.col("risk").median(),
            risk_p95=pl.col("risk").quantile(0.95),
            cost_r=(pl.col("r_mid") - pl.col("r_multiple")).mean(),
            mid_r=pl.col("r_mid").mean(),
            net_r=pl.col("r_multiple").mean(),
            net_usd=(pl.col("net_usd") * LOTS).sum(),
            tp_pct=(pl.col("reason") == "tp").cast(pl.Float64).mean() * 100,
            stop_pct=(pl.col("reason") == "stop").cast(pl.Float64).mean() * 100,
        )
        .sort("year")
        .with_columns(pl.col(pl.Float64).round(3))
    )


def window_table(base: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for hour, rng, tgt in TOP8_CELLS:
        w = base.filter((pl.col("hour") == hour) & (pl.col("range_min") == rng))
        row = {"window": f"h{hour:02d}_r{rng}_t{tgt:g}"}
        for p in ("dev 2020-23", "val 2024-25H1", "2026 (fit window)"):
            s = _slice(w, p)
            d = s.group_by("day").agg(pl.col("r_multiple").sum())["r_multiple"].to_numpy()
            row[f"{p} mean_r"] = float(s["r_multiple"].mean()) if s.height else None
            row[f"{p} t"] = newey_west(d).t_stat if d.size > 2 else None
        rows.append(row)
    return _frame(rows)


def concurrency(base: pl.DataFrame) -> dict:
    """Peak simultaneous exposure - the portfolio is one instrument, eight times."""
    risk = pl.col("risk_actual_usd") * LOTS
    ev = pl.concat([
        base.select(ts=pl.col("entry_ts"), n=pl.lit(1, pl.Int64),
                    risk=risk, side=pl.col("direction").cast(pl.Int64)),
        base.select(ts=pl.col("exit_ts"), n=pl.lit(-1, pl.Int64),
                    risk=-risk, side=-pl.col("direction").cast(pl.Int64)),
    ]).sort(["ts", "n"])            # an exit at the same instant frees its slot first
    n = np.cumsum(ev["n"].to_numpy())
    side = np.cumsum(ev["side"].to_numpy())
    at_risk = np.cumsum(ev["risk"].to_numpy())
    return {"max_open": int(n.max()), "max_net_same_side": int(np.abs(side).max()),
            "max_open_risk_usd": round(float(at_risk.max()), 2),
            "p99_open_at_entry": float(np.quantile(n[ev["n"].to_numpy() == 1], 0.99))}


# --------------------------------------------------------------------------
# Selection tests on the grid
# --------------------------------------------------------------------------

def _add_months(d: date, k: int) -> date:
    m = d.month - 1 + k
    return date(d.year + m // 12, m % 12 + 1, 1)


def _cube(g: pl.DataFrame, windows: list[tuple[date, date]], col: str):
    """Per-window, per-cell sums and counts as (W, hours, ranges, targets) arrays."""
    shape = (len(windows), len(GRID_HOURS), len(GRID_RANGES), len(GRID_TARGETS))
    S, N = np.zeros(shape), np.zeros(shape)
    for wi, (lo, hi) in enumerate(windows):
        agg = (g.filter((pl.col("day") >= lo) & (pl.col("day") < hi))
               .group_by(["hour", "range_min", "target_mult"])
               .agg(s=pl.col(col).sum(), n=pl.len()))
        for h, rm, tg, s, n in agg.iter_rows():
            k = (wi, h, GRID_RANGES.index(rm), GRID_TARGETS.index(tg))
            S[k], N[k] = s, n
    return S, N


def _random_sets(S: np.ndarray, N: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Mean per trade of ``draws`` random 8-window portfolios, re-drawn per window.

    Eight distinct hours, each with a random bracket and target - the same
    space the preset was picked from.
    """
    rng = np.random.default_rng(seed)
    W, H, R, T = S.shape
    hours = np.argsort(rng.random((draws, W, H)), axis=2)[:, :, :8]
    ri = rng.integers(0, R, (draws, W, 8))
    ti = rng.integers(0, T, (draws, W, 8))
    w = np.arange(W)[None, :, None]
    return S[w, hours, ri, ti].sum(axis=(1, 2)) / np.maximum(
        N[w, hours, ri, ti].sum(axis=(1, 2)), 1)


def preset_null(g: pl.DataFrame) -> pl.DataFrame:
    """Where TOP8 ranks among random 8-window sets, period by period."""
    names = ["dev 2020-23", "val 2024-25H1", "dev+val pooled", "2026 (fit window)"]
    S, N = _cube(g, [(PERIODS[p][0], date.fromordinal(PERIODS[p][1].toordinal() + 1))
                     for p in names], "r")
    rows = []
    for wi, p in enumerate(names):
        k = [(GRID_RANGES.index(rm), GRID_TARGETS.index(tg), h) for h, rm, tg in TOP8_CELLS]
        top = sum(S[wi, h, ri, ti] for ri, ti, h in k) / max(
            sum(N[wi, h, ri, ti] for ri, ti, h in k), 1)
        null = _random_sets(S[wi:wi + 1], N[wi:wi + 1], 4000, seed=wi)
        rows.append({"period": p, "top8_mean_r": top,
                     "random_median_r": float(np.median(null)),
                     "random_p95_r": float(np.quantile(null, 0.95)),
                     "top8_percentile": float((null < top).mean() * 100)})
    return _frame(rows)


def walk_forward(g: pl.DataFrame, lookback_m: int) -> tuple[pl.DataFrame, dict]:
    """Re-select eight hours every quarter from trailing P&L, trade them next quarter.

    This is the *procedure* TOP8_2026 is one instance of. Score is total net R
    over the lookback; one cell per hour, best eight hours, no positivity
    requirement - the preset had none.
    """
    starts = []
    s = WF_FIRST
    while s <= WF_LAST:
        starts.append(s)
        s = _add_months(s, WF_STEP)

    keys = ["hour", "range_min", "target_mult"]
    oos, insample = [], []
    for s in starts:
        e = _add_months(s, WF_STEP)
        hist = g.filter((pl.col("day") >= _add_months(s, -lookback_m))
                        & (pl.col("day") < s))
        now = g.filter((pl.col("day") >= s) & (pl.col("day") < e))
        if now.is_empty():
            continue
        for src, sink in ((hist, oos), (now, insample)):
            pick = (src.group_by(keys).agg(score=pl.col("r").sum())
                    .sort(["score", "hour"], descending=[True, False])
                    .unique(subset="hour", keep="first", maintain_order=True)
                    .head(8).select(keys))
            sink.append(now.join(pick, on=keys, how="semi"))

    fwd = pl.concat(oos)
    ins = pl.concat(insample)
    windows = [(s, _add_months(s, WF_STEP)) for s in starts]
    S, N = _cube(g, windows, "r")
    null = _random_sets(S, N, 4000, seed=100 + lookback_m)
    fwd_mean = float(fwd["r"].mean())
    summary = {
        "lookback_m": lookback_m,
        "oos": stats_row(fwd, r_col="r", usd_col="usd"),
        "oos_pre2024": stats_row(fwd.filter(pl.col("day") < date(2024, 1, 1)),
                                 r_col="r", usd_col="usd"),
        "oos_2024on": stats_row(fwd.filter(pl.col("day") >= date(2024, 1, 1)),
                                r_col="r", usd_col="usd"),
        "insample_mean_r": float(ins["r"].mean()),
        "all_cells_mean_r": float(g.filter(pl.col("day") >= WF_FIRST)["r"].mean()),
        "random_selection_median_r": float(np.median(null)),
        "random_selection_p95_r": float(np.quantile(null, 0.95)),
        "oos_percentile_vs_random": float((null < fwd_mean).mean() * 100),
    }
    by_year = (fwd.group_by(pl.col("day").dt.year().alias("year"))
               .agg(trades=pl.len(), mean_r=pl.col("r").mean(),
                    net_usd=(pl.col("usd") * LOTS).sum())
               .sort("year").with_columns(pl.col(pl.Float64).round(3)))
    return by_year, summary


def rank_stability(g: pl.DataFrame) -> dict:
    """Do the cells that scored in 2026 H1 score before 2026?"""
    keys = ["hour", "range_min", "target_mult"]
    pre = (g.filter(pl.col("day") < date(2026, 1, 1)).group_by(keys)
           .agg(pre=pl.col("r").mean()))
    fit = (g.filter(pl.col("day").is_between(date(2026, 1, 1), date(2026, 6, 30)))
           .group_by(keys).agg(fit=pl.col("r").mean()))
    both = pre.join(fit, on=keys).with_columns(
        pre_rank=pl.col("pre").rank(descending=True),
        fit_rank=pl.col("fit").rank(descending=True))
    rho = float(np.corrcoef(both["pre_rank"], both["fit_rank"])[0, 1])
    top = both.join(pl.DataFrame(TOP8_CELLS, schema=keys, orient="row"), on=keys)
    return {"cells": both.height, "spearman_pre_vs_2026H1": round(rho, 3),
            "top8_rank_2026H1": sorted(int(x) for x in top["fit_rank"]),
            "top8_rank_pre2026": sorted(int(x) for x in top["pre_rank"])}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def phase_analyse() -> None:
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(30)
    tapes = {p.stem: pl.read_parquet(p) for p in sorted((OUT / "tapes").glob("*.parquet"))}
    base = tapes["base"]
    g = pl.read_parquet(OUT / "grid.parquet")
    report: dict = {}

    print(f"\n=== {SYMBOL} {PRESET}, {LOTS} lots fixed, spread cap ${SPREAD_CAP_USD} ===")
    t = period_table(base)
    print(t)
    report["periods"] = t.to_dicts()

    print("\n=== by year: dollar risk per trade at fixed lots, cost and edge in R ===")
    y = yearly_table(base)
    print(y)
    report["years"] = y.to_dicts()

    print("\n=== 2026 by month ($ at fixed lots) ===")
    m26 = (_slice(base, "2026 (fit window)")
           .group_by(pl.col("day").dt.month().alias("month"))
           .agg(trades=pl.len(), mean_r=pl.col("r_multiple").mean(),
                net_usd=(pl.col("net_usd") * LOTS).sum()).sort("month")
           .with_columns(pl.col(pl.Float64).round(3)))
    print(m26)
    report["months_2026"] = m26.to_dicts()

    print("\n=== per window, net mean R and daily NW t ===")
    w = window_table(base)
    print(w)
    report["windows"] = w.to_dicts()

    report["concurrency"] = concurrency(base)
    print("\nconcurrency:", report["concurrency"])

    print("\n=== variants: dev+val pooled and 2026 ===")
    rows = []
    for name, tape in tapes.items():
        pooled = stats_row(_slice(tape, "dev+val pooled"))
        recent = stats_row(_slice(tape, "2026 (fit window)"))
        rows.append({"variant": name, "trades": pooled["trades"],
                     "mean_r": pooled["mean_r"], "net_usd": pooled["net_usd"],
                     "t_usd": pooled["t_usd"], "t_r": pooled["t_r"],
                     "2026 trades": recent["trades"], "2026 mean_r": recent["mean_r"],
                     "2026 net_usd": recent["net_usd"], "2026 t_usd": recent["t_usd"]})
    v = _frame(rows)
    print(v)
    report["variants"] = v.to_dicts()

    # The grid must reproduce the preset exactly, or the selection tests are
    # about a different strategy.
    chk = (g.filter(pl.col("target_mult") == 3.0)
           .join(base.select("day", "hour", "range_min", "r_multiple"),
                 on=["day", "hour", "range_min"], how="inner"))
    report["grid_vs_tape_max_abs_r_diff"] = float((chk["r"] - chk["r_multiple"]).abs().max())
    report["grid_vs_tape_matched"] = chk.height
    print(f"\ngrid reproduces the tape: {chk.height} of {base.height} trades matched, "
          f"max |dR| {report['grid_vs_tape_max_abs_r_diff']:.2e}")

    print("\n=== TOP8 against random 8-window sets from the same space ===")
    nt = preset_null(g)
    print(nt)
    report["preset_null"] = nt.to_dicts()

    report["rank_stability"] = rank_stability(g)
    print("\nrank stability:", report["rank_stability"])

    report["walk_forward"] = []
    for lb in WF_LOOKBACKS:
        by_year, summ = walk_forward(g, lb)
        print(f"\n=== walk-forward re-selection, {lb}m lookback, {WF_STEP}m hold ===")
        print(by_year)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in summ.items() if not isinstance(v, dict)})
        for k in ("oos", "oos_pre2024", "oos_2024on"):
            print(f"  {k}: " + ", ".join(
                f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                for a, b in summ[k].items()))
        report["walk_forward"].append(summ | {"by_year": by_year.to_dicts()})

    # ------------------------------------------------------------ criteria
    pooled = lambda name: stats_row(_slice(tapes[name], "dev+val pooled"))  # noqa: E731
    dv = pooled("base")
    dev = stats_row(_slice(base, "dev 2020-23"))
    val = stats_row(_slice(base, "val 2024-25H1"))

    a = _daily_usd(_slice(base, "dev+val pooled"))
    b = _daily_usd(_slice(tapes["flip"], "dev+val pooled"))
    ab = a.join(b, on="day", how="full", coalesce=True, suffix="_flip").fill_null(0.0)
    flip = paired_bootstrap(ab["usd"].to_numpy(), ab["usd_flip"].to_numpy(),
                            n_boot=2000, rng=np.random.default_rng(7))

    wf = report["walk_forward"]
    criteria = {
        "D1 dev+val daily $ NW t >= 2": (dv["t_usd"] >= 2.0, round(dv["t_usd"], 2)),
        "D2 neither dev nor val t <= -2": (
            dev["t_usd"] > -2.0 and val["t_usd"] > -2.0,
            (round(dev["t_usd"], 2), round(val["t_usd"], 2))),
        "D3 beats flipped direction, p < 0.05": (
            flip.point > 0 and flip.p_value < 0.05, str(flip)),
        "D4 walk-forward OOS t >= 2 and >= 95th pct of random selection": (
            all(s["oos"]["t_usd"] >= 2.0 and s["oos_percentile_vs_random"] >= 95.0
                for s in wf),
            [(s["lookback_m"], round(s["oos"]["t_usd"], 2),
              round(s["oos_percentile_vs_random"], 1)) for s in wf]),
        "D5 dev+val net > 0 at adverse 1.0 and 1000 ms": (
            pooled("adverse_1")["net_usd"] > 0 and pooled("latency_1000ms")["net_usd"] > 0,
            (round(pooled("adverse_1")["net_usd"], 0),
             round(pooled("latency_1000ms")["net_usd"], 0))),
    }
    print("\n=== pre-registered deploy criteria ===")
    for k, (ok, val_) in criteria.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {val_}")
    deploy = all(ok for ok, _ in criteria.values())
    print(f"\nDEPLOYABLE: {deploy}")
    report["criteria"] = {k: {"pass": bool(ok), "value": str(v)} for k, (ok, v) in criteria.items()}
    report["deployable"] = deploy

    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {OUT / 'summary.json'}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", nargs="+", default=["tapes", "grid", "analyse"],
                    choices=["tapes", "grid", "analyse"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if "tapes" in args.phase:
        phase_tapes(args.verbose)
    if "grid" in args.phase:
        phase_grid(args.verbose)
    if "analyse" in args.phase:
        phase_analyse()


if __name__ == "__main__":
    main()
