"""Backtest the New York open EMA rule and its controls.

The rule under test, and the reported claim it is being read against, are stated
in :mod:`qlab.strategies.open_ema`. This script produces every number in
``docs/findings/open-ema.md``, in the order the report makes the argument:

0. the signal itself - how often each side, and how far from the EMA;
1. the raw signal, before any exit: forward mid return, MFE and MAE in ATR
   units, for the rule and for each control;
2. the exit surface - three trailing families at seven distances, net of cost;
3. the controls on the same fills, cell by cell;
4. the creator's headline numbers (win rate, profit factor, total return at 1%
   risk, maximum drawdown) recomputed on the best cell the rule can find;
5. validation, with the power to detect the claim stated before it is read;
6. pooled dev + validation, which is the verdict;
7. the ``ema_series="rth"`` variant, as a robustness check on the one
   specification choice the rule leaves genuinely open.

Usage
-----
    python scripts/backtests/backtest_open_ema.py                 # the study
    python scripts/backtests/backtest_open_ema.py --cached        # re-report only
    python scripts/backtests/backtest_open_ema.py --symbols USTEC XAUUSD
    python scripts/backtests/backtest_open_ema.py --no-variant

The locked ``test`` split is never read. This script has no flag that would.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qlab import paths  # noqa: E402
from qlab.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_ci,
    newey_west,
    paired_bootstrap,
)
from qlab.strategies.open_ema import (  # noqa: E402
    DIRECTION_RULES,
    EXIT_STYLES,
    STOP_FRACTIONS,
    OpenEMAConfig,
    daily_context,
    equity_curve,
    run,
    select_rule,
)

SPLITS = ("dev", "validation")
OUT = paths.STRATEGY_REPORT_DIR
RISK_PCT = 1.0

#: The creator's reported figures, for the side-by-side.
CLAIM = {
    "trades": 1448,
    "total_return_pct": 982.0,
    "win_rate": 0.57,
    "profit_factor": 1.29,
    "max_drawdown": -0.197,
}


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def cell_stats(cell: pl.DataFrame, column: str = "net_r") -> dict:
    """Per-trade edge of one (rule, style, distance) cell, with a HAC t.

    One trade per session, so the series is already daily and the Newey-West
    lag is about residual day-to-day dependence in the exit geometry rather than
    about overlapping holds - there are none, every trade dies at the bell.
    """
    if cell.is_empty():
        return {"n": 0}
    r = cell.sort("nyd")[column].to_numpy()
    r = r[np.isfinite(r)]
    if r.size < 3:
        return {"n": int(r.size)}
    hac = newey_west(r)
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = abs(float(losses.sum())) if losses.size else 0.0
    return {
        "n": int(r.size),
        "mean_r": hac.mean,
        "t": hac.t_stat,
        "p": hac.p_value,
        "win_rate": float((r > 0).mean()),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "payoff": (float(wins.mean()) / abs(float(losses.mean()))
                   if wins.size and losses.size else float("nan")),
        "median_r": float(np.median(r)),
        "sd_r": float(r.std(ddof=1)),
    }


def surface(trades: pl.DataFrame, rule: str, column: str = "net_r") -> pl.DataFrame:
    """Mean edge per trade over the whole exit grid, for one direction rule."""
    picked = select_rule(trades, rule)
    rows = []
    for style in EXIT_STYLES:
        for frac in STOP_FRACTIONS:
            cell = picked.filter(
                (pl.col("exit_style") == style)
                & (pl.col("stop_frac") - frac).abs().lt(1e-9)
            )
            rows.append({"rule": rule, "exit_style": style, "stop_frac": frac}
                        | cell_stats(cell, column)
                        | {"stop_rate": (float((cell["exit_reason"] == "stop").mean())
                                         if not cell.is_empty() else float("nan")),
                           "hold_min": (float(cell["hold_min"].mean())
                                        if not cell.is_empty() else float("nan"))})
    return pl.DataFrame(rows)


def pick_cell(trades: pl.DataFrame, rule: str, style: str, frac: float) -> pl.DataFrame:
    return select_rule(trades, rule).filter(
        (pl.col("exit_style") == style)
        & (pl.col("stop_frac") - frac).abs().lt(1e-9)
    ).sort("nyd")


def fmt_surface(frame: pl.DataFrame, title: str) -> str:
    """The grid, with Benjamini-Hochberg applied across the whole grid.

    Twenty-one cells are read at once, so the largest ``t`` in the table is a
    maximum over twenty-one draws and its nominal p-value is not the p-value of
    any decision anyone actually makes. ``q`` is that p corrected across the
    grid, and the report quotes ``q``.
    """
    live = frame.filter(pl.col("n") > 0)
    if live.is_empty():
        return f"\n{title}\n  (nothing resolved)"
    q = benjamini_hochberg(live["p"].to_numpy())
    lines = [f"\n{title}",
             "  style      frac     n   mean_r      t       q     win%    PF  payoff"
             "   stop%  hold_min"]
    for row, q_value in zip(live.iter_rows(named=True), q):
        lines.append(
            f"  {row['exit_style']:<9} {row['stop_frac']:>5.2f} {row['n']:>5} "
            f"{row['mean_r']:>+8.4f} {row['t']:>+6.2f} {q_value:>7.3f} "
            f"{100 * row['win_rate']:>7.1f} {row['profit_factor']:>6.2f} "
            f"{row['payoff']:>7.2f} {100 * row['stop_rate']:>6.1f} "
            f"{row['hold_min']:>8.0f}"
        )
    return "\n".join(lines)


def reachability(trades: pl.DataFrame, label: str) -> str:
    """Can any exit in this family produce the reported win rate and PF together?

    The reported pair - 57% wins at a profit factor of 1.29 - implies an average
    win of ``PF * (1 - w) / w`` average losses, which is 0.97: winners and losers
    about the same size. A trailing stop cannot make that shape. It makes the
    opposite one - many small losses against rare large winners - so its win rate
    falls as its payoff rises. Every cell in the sweep is checked against the
    pair rather than argued about.
    """
    want_w, want_pf = CLAIM["win_rate"], CLAIM["profit_factor"]
    rows = list(surface(trades, "ema").filter(pl.col("n") > 0).iter_rows(named=True))
    if not rows:
        return ""
    best_w = max(rows, key=lambda r: r["win_rate"])
    near = [r for r in rows
            if r["win_rate"] >= want_w - 0.02 and r["profit_factor"] >= want_pf - 0.05]
    implied = want_pf * (1 - want_w) / want_w
    return "\n".join([
        f"\n  {label}: the highest win rate anywhere in the 21-cell sweep is "
        f"{100 * best_w['win_rate']:.1f}% ({best_w['exit_style']} @ "
        f"{best_w['stop_frac']:g} ATR), and there it runs at profit factor "
        f"{best_w['profit_factor']:.2f}, payoff {best_w['payoff']:.2f}.",
        f"  The reported pair needs {100 * want_w:.0f}% wins at PF {want_pf:.2f}, "
        f"implying a payoff of {implied:.2f} - winners the same size as losers.",
        f"  Cells reaching both (within 2pp and 0.05 PF): {len(near)}.",
    ])


def by_year(cell: pl.DataFrame, label: str) -> str:
    """Where the edge sits in time. An edge present in one regime is not an edge."""
    if cell.is_empty():
        return ""
    lines = [f"\n  {label} by calendar year:",
             "    year     n   mean_r      t      win%    PF"]
    for year in sorted(cell["nyd"].dt.year().unique().to_list()):
        stats = cell_stats(cell.filter(pl.col("nyd").dt.year() == year))
        if not stats.get("n"):
            continue
        lines.append(
            f"    {year}  {stats['n']:>4} {stats['mean_r']:>+8.4f} "
            f"{stats['t']:>+6.2f} {100 * stats['win_rate']:>7.1f} "
            f"{stats['profit_factor']:>6.2f}"
        )
    return "\n".join(lines)


def gap_sort(trades: pl.DataFrame, style: str, frac: float, label: str) -> str:
    """Mean edge by how far the candle closed from the EMA, rule against controls.

    This is the mechanism test. If the distance to the EMA carries information
    then the rule should do better where that distance is larger, and the ordering
    is a much weaker thing to ask of the data than a significant mean. But a
    larger gap also means a more volatile open, and a tight stop with no target
    has fatter tails on a volatile day whichever way it is pointed - so the
    controls are sorted the same way. The claim only survives if the rule's
    ordering is steeper than the coin's, not merely present.
    """
    header = ["q1 nearest", "q2", "q3", "q4 furthest"]
    out = [f"\n  {label} - {style} @ {frac:g} ATR, mean net R by |close-EMA| quartile:",
           "    rule     " + "".join(f"{h:>13}" for h in header)]
    for rule in ("ema", "coin", "long"):
        cell = select_rule(trades, rule).filter(
            (pl.col("exit_style") == style)
            & (pl.col("stop_frac") - frac).abs().lt(1e-9)
        )
        if cell.is_empty():
            continue
        agg = (cell.with_columns(
                   bucket=pl.col("ema_gap_atr").abs().qcut(4, labels=header))
               .group_by("bucket").agg(m=pl.col("net_r").mean(), n=pl.len())
               .sort("bucket"))
        out.append(f"    {rule:<8} " + "".join(f"{m:>+13.4f}" for m in agg["m"]))
    out.append(f"    (n per bucket: {agg['n'][0]})")
    return '\n'.join(out)


def focus(frames: dict[str, pl.DataFrame], style: str, frac: float, why: str) -> None:
    """One exit cell, followed across dev, validation and the pool."""
    print(f"\n{'-' * 78}\n  CELL: {style} @ {frac:g} ATR  ({why})\n{'-' * 78}")
    pooled = pl.concat([f for f in frames.values() if not f.is_empty()])
    for name, frame in list(frames.items()) + [("pooled", pooled)]:
        if frame.is_empty():
            continue
        rows = [headline(pick_cell(frame, r, style, frac), r) for r in DIRECTION_RULES]
        print(f"\n  {name}:" + fmt_headlines(rows))
        ema = pick_cell(frame, "ema", style, frac)
        for other in ("coin", "long"):
            rival = pick_cell(frame, other, style, frac)
            common = ema.join(rival.select("nyd", rival_r=pl.col("net_r")),
                              on="nyd", how="inner").sort("nyd")
            if common.height < 3:
                continue
            diff = paired_bootstrap(common["net_r"].to_numpy(),
                                    common["rival_r"].to_numpy(),
                                    rng=np.random.default_rng(7))
            print(f"    ema - {other:<5} {diff.point:+.4f} R  "
                  f"95% CI [{diff.lo:+.4f}, {diff.hi:+.4f}]  p {diff.p_value:.3f}")
    cell = pick_cell(pooled, "ema", style, frac)
    boot = bootstrap_ci(cell["net_r"].to_numpy(), rng=np.random.default_rng(7))
    print(f"\n  pooled bootstrap: {boot.point:+.4f} R  95% CI "
          f"[{boot.lo:+.4f}, {boot.hi:+.4f}]  p {boot.p_value:.3f}")
    print(by_year(cell, "ema"))


def headline(cell: pl.DataFrame, label: str) -> dict:
    """The creator's four reported numbers, recomputed on one cell."""
    stats = cell_stats(cell)
    if not stats.get("n"):
        return {"label": label, "trades": 0}
    curve = equity_curve(cell, risk_pct=RISK_PCT)
    final = float(curve["equity"][-1])
    return {
        "label": label,
        "trades": stats["n"],
        "mean_r": stats["mean_r"],
        "t": stats["t"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "total_return_pct": 100.0 * (final / 100_000.0 - 1.0),
        "max_drawdown": float(curve["drawdown"].min()),
    }


def fmt_headlines(rows: list[dict]) -> str:
    lines = ["\n  source                       trades   mean_r      t     win%    PF"
             "      total%     maxDD%"]
    for row in rows:
        if not row.get("trades"):
            lines.append(f"  {row['label']:<28} {'-':>6}")
            continue
        lines.append(
            f"  {row['label']:<28} {row['trades']:>6} {row['mean_r']:>+8.4f} "
            f"{row['t']:>+6.2f} {100 * row['win_rate']:>7.1f} "
            f"{row['profit_factor']:>6.2f} {row['total_return_pct']:>+11.1f} "
            f"{100 * row['max_drawdown']:>+9.1f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------

def signal_diagnostic(symbol: str, cfg: OpenEMAConfig) -> None:
    print(f"\n{'=' * 78}\n0. The signal itself - {symbol}, ema_series={cfg.ema_series}\n{'=' * 78}")
    for split in SPLITS:
        ctx = daily_context(symbol, cfg, split=split)
        if ctx.is_empty():
            print(f"  {split}: no sessions")
            continue
        row = ctx.select(
            n=pl.len(),
            long=(pl.col("signal") == 1).mean(),
            doji=(pl.col("signal") == 0).sum(),
            gap=(pl.col("ema_gap").abs() / pl.col("atr")).median(),
            atr=pl.col("atr").mean(),
        ).row(0, named=True)
        print(
            f"  {split:<11} {row['n']:>4} sessions   long {100 * row['long']:.1f}%   "
            f"doji {row['doji']}   median |close-EMA| {row['gap']:.3f} ATR   "
            f"ATR {row['atr']:.0f} pts"
        )


def raw_signal(trades: pl.DataFrame, split: str) -> None:
    print(f"\n{'=' * 78}\n1. The raw signal on {split}, before any exit (mid, ATR units)\n{'=' * 78}")
    print("  rule        n    fwd15m    fwd30m     fwd1h     fwd2h      bell"
          "      t(bell)   MFE    MAE")
    for rule in DIRECTION_RULES:
        picked = select_rule(trades, rule).filter(
            (pl.col("exit_style") == EXIT_STYLES[0])
            & (pl.col("stop_frac") - STOP_FRACTIONS[0]).abs().lt(1e-9)
        ).sort("nyd")
        if picked.is_empty():
            continue
        bell = picked["fwd_bell_atr"].to_numpy()
        hac = newey_west(bell[np.isfinite(bell)])
        cols = [picked[c].mean() for c in
                ("fwd_15m_atr", "fwd_30m_atr", "fwd_60m_atr", "fwd_120m_atr",
                 "fwd_bell_atr")]
        print(
            f"  {rule:<8} {picked.height:>4} " + "".join(f"{v:>+9.4f} " for v in cols)
            + f"  {hac.t_stat:>+6.2f} "
            f"{picked['mfe_atr'].mean():>+6.3f} {picked['mae_atr'].mean():>+6.3f}"
        )


def exit_surface(trades: pl.DataFrame, split: str) -> pl.DataFrame:
    print(f"\n{'=' * 78}\n2. The exit surface on {split} - the rule, net of cost (R per trade)\n{'=' * 78}")
    grid = surface(trades, "ema")
    print(fmt_surface(grid, "  rule = ema"))
    print("\n  Same grid at the mid, to separate 'no edge' from 'cost ate it':")
    print(fmt_surface(surface(trades, "ema", "mid_r"), "  rule = ema (mid)"))
    return grid


def controls(trades: pl.DataFrame, split: str, best: tuple[str, float]) -> None:
    style, frac = best
    print(f"\n{'=' * 78}\n3. Controls on {split}, same fills, cell = {style} @ {frac:g} ATR\n{'=' * 78}")
    rows = []
    for rule in DIRECTION_RULES:
        cell = pick_cell(trades, rule, style, frac)
        rows.append(headline(cell, f"{rule}"))
    print(fmt_headlines(rows))

    ema = pick_cell(trades, "ema", style, frac)
    for other in ("coin", "long"):
        rival = pick_cell(trades, other, style, frac)
        common = ema.join(rival.select("nyd", rival_r=pl.col("net_r")), on="nyd",
                          how="inner").sort("nyd")
        if common.height < 3:
            continue
        diff = paired_bootstrap(common["net_r"].to_numpy(),
                                common["rival_r"].to_numpy(),
                                rng=np.random.default_rng(7))
        print(f"\n  ema - {other:<6} on {common.height} shared sessions: "
              f"{diff.point:+.4f} R  95% CI [{diff.lo:+.4f}, {diff.hi:+.4f}]  "
              f"bootstrap p {diff.p_value:.3f}")


def power_note(n: int, sd: float, target: float) -> None:
    se = sd / np.sqrt(n)
    print(f"\n  Power, stated before the split is read: {n} sessions, per-trade SD "
          f"{sd:.3f} R -> SE {se:.4f} R. The claimed expectancy of {target:+.3f} R "
          f"would land at t = {target / se:+.2f}; the smallest edge this split can "
          f"confirm at t = 2.0 is {2.0 * se:+.4f} R per trade.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["USTEC"])
    parser.add_argument("--cached", action="store_true",
                        help="re-report from saved parquet instead of re-simulating")
    parser.add_argument("--no-variant", action="store_true",
                        help="skip the ema_series=rth robustness run")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OpenEMAConfig()

    for symbol in args.symbols:
        signal_diagnostic(symbol, cfg)

        frames: dict[str, pl.DataFrame] = {}
        for split in SPLITS:
            path = OUT / f"open_ema_{symbol}_{split}.parquet"
            if args.cached and path.exists():
                frames[split] = pl.read_parquet(path)
            else:
                print(f"\nsimulating {symbol} {split} over the tick tape ...", flush=True)
                frames[split] = run(symbol, cfg, split=split)
                if not frames[split].is_empty():
                    frames[split].write_parquet(path)
            print(f"  {split}: {frames[split].height:,} cells")

        dev = frames["dev"]
        if dev.is_empty():
            print(f"{symbol}: nothing resolved on dev")
            continue

        raw_signal(dev, "dev")
        grid = exit_surface(dev, "dev")

        # The cell the rule does best in on dev. Selected on dev and then held
        # fixed for every later split, which is the only thing that makes the
        # validation read an out-of-sample one.
        live = grid.filter(pl.col("n") > 0)
        trail_dev = float(live.filter(pl.col("exit_style") == "trail")
                          .sort("mean_r", descending=True).row(0, named=True)["stop_frac"])
        best_row = live.sort("mean_r", descending=True).row(0, named=True)
        best = (best_row["exit_style"], best_row["stop_frac"])
        t_row = live.sort("t", descending=True).row(0, named=True)
        t_best = (t_row["exit_style"], t_row["stop_frac"])
        print(f"\n  Best dev cell by mean R: {best[0]} @ {best[1]:g} ATR, "
              f"{best_row['mean_r']:+.4f} R per trade (t {best_row['t']:+.2f}).")
        print(f"  Best dev cell by t:      {t_best[0]} @ {t_best[1]:g} ATR, "
              f"{t_row['mean_r']:+.4f} R per trade (t {t_row['t']:+.2f}).")
        # Both are carried forward. Which one a reader would have picked on dev
        # depends on whether they ranked by size or by significance, and that
        # choice alone moves the headline - which is a fact about the surface.
        print("  Both are carried forward unchanged; see section 6b.")
        boot = bootstrap_ci(pick_cell(dev, "ema", *best)["net_r"].to_numpy(),
                            rng=np.random.default_rng(7))
        print(f"  Stationary bootstrap on that cell: {boot.point:+.4f} R  "
              f"95% CI [{boot.lo:+.4f}, {boot.hi:+.4f}]  p {boot.p_value:.3f}  "
              f"(block {boot.block_len})")

        controls(dev, "dev", best)

        print(f"\n{'=' * 78}\n4. The creator's headline, recomputed - {symbol} dev\n{'=' * 78}")
        rows = [dict(CLAIM, label="reported (2019-2026)", mean_r=float("nan"),
                     t=float("nan"))]
        rows[0]["trades"] = CLAIM["trades"]
        print("\n  reported: %d trades, %+.0f%% total, %.0f%% wins, PF %.2f, maxDD %.1f%%"
              % (CLAIM["trades"], CLAIM["total_return_pct"], 100 * CLAIM["win_rate"],
                 CLAIM["profit_factor"], 100 * CLAIM["max_drawdown"]))
        print(fmt_headlines([headline(pick_cell(dev, "ema", *best), f"ema {best[0]} @{best[1]:g}")]
                            + [headline(pick_cell(dev, r, *best), f"  control: {r}")
                               for r in ("coin", "long")]))
        print(f"\n  (total% and maxDD% compound {RISK_PCT:g}% of equity per trade, "
              "which is the creator's sizing.)")
        print(reachability(dev, "dev"))

        # --- validation, with the power stated first ------------------------
        val = frames["validation"]
        if not val.is_empty():
            dev_cell = pick_cell(dev, "ema", *best)
            sd = float(dev_cell["net_r"].std())
            n_val = pick_cell(val, "ema", *best).height
            print(f"\n{'=' * 78}\n5. Validation - {symbol}\n{'=' * 78}")
            power_note(n_val, sd, 0.125)
            raw_signal(val, "validation")
            print(fmt_surface(surface(val, "ema"), "\n  rule = ema, validation"))
            controls(val, "validation", best)

            # --- pooled: the verdict -----------------------------------------
            print(f"\n{'=' * 78}\n6. Pooled dev + validation - {symbol}\n{'=' * 78}")
            pooled = pl.concat([dev, val])
            rows = [headline(pick_cell(pooled, r, *best), r) for r in DIRECTION_RULES]
            print(fmt_headlines(rows))
            pooled_cell = pick_cell(pooled, "ema", *best)
            boot = bootstrap_ci(pooled_cell["net_r"].to_numpy(),
                                rng=np.random.default_rng(7))
            print(f"\n  pooled ema: {boot.point:+.4f} R  95% CI "
                  f"[{boot.lo:+.4f}, {boot.hi:+.4f}]  bootstrap p {boot.p_value:.3f}")
            print(fmt_surface(surface(pooled, "ema"), "\n  pooled exit surface, rule = ema"))
            print(reachability(pooled, "pooled"))
            print(gap_sort(pooled, *t_best, "pooled"))
            print(gap_sort(pooled, "trail", trail_dev, "pooled"))

            print(f"\n{'=' * 78}\n6b. The two dev-selected cells, followed "
                  f"through - {symbol}\n{'=' * 78}")
            focus(frames, *best, "best dev mean R")
            if t_best != best:
                focus(frames, *t_best, "best dev t")

            print(f"\n{'=' * 78}\n6c. The rule's own stated exit, at its best "
                  f"dev distance - {symbol}\n{'=' * 78}")
            focus(frames, "trail", trail_dev,
                  "the creator's own exit, best trailing distance on dev")

        # --- the one open specification choice ------------------------------
        if not args.no_variant:
            print(f"\n{'=' * 78}\n7. Variant: ema_series='rth' - {symbol}\n{'=' * 78}")
            vcfg = OpenEMAConfig(ema_series="rth")
            signal_diagnostic(symbol, vcfg)
            for split in SPLITS:
                path = OUT / f"open_ema_rth_{symbol}_{split}.parquet"
                if args.cached and path.exists():
                    frame = pl.read_parquet(path)
                else:
                    frame = run(symbol, vcfg, split=split,
                                styles=[best[0]], stop_fractions=list(STOP_FRACTIONS))
                    if not frame.is_empty():
                        frame.write_parquet(path)
                if frame.is_empty():
                    continue
                print(fmt_surface(surface(frame, "ema"), f"\n  rth variant, {split}"))

    print(f"\nartifacts: {OUT}")
    print("the locked test split was not read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
