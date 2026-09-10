"""Replicate Eross et al. (2017) on modern data, then test what it implies.

Usage
-----
``python scripts/backtests/backtest_btc_intraday.py replicate``  - the descriptive claims
``python scripts/backtests/backtest_btc_intraday.py strategy``   - dev sweep, both rules
``python scripts/backtests/backtest_btc_intraday.py confirm``    - validation split
``python scripts/backtests/backtest_btc_intraday.py test --unlock``
``python scripts/backtests/backtest_btc_intraday.py estimator``  - Corwin-Schultz vs truth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab.loader import SPLITS, load_bars
from qlab.metrics import format_returns_tearsheet, tearsheet_from_returns
from qlab.microstructure import estimator_error, with_corwin_schultz
from qlab.stats import bootstrap_ci, newey_west, sharpe, two_proportion_z
from qlab.strategies import btc_intraday as btc

OUT = Path("reports/strategies")

# Bitcoin trades every day, so a year is 365 periods, not 252. Using 252 here
# would inflate every Sharpe on this page by 20%.
PPY = 365


def _window(split: str) -> tuple[str, str]:
    s = SPLITS[split]
    return str(s.start), str(s.end)


def _load(split: str | None, allow_test: bool = False):
    if split is None:  # dev + validation, for description only
        return btc.load_btc(str(SPLITS["dev"].start), str(SPLITS["validation"].end))
    if SPLITS[split].locked and not allow_test:
        raise SystemExit("test split is locked; pass --unlock and mean it")
    lo, hi = _window(split)
    return btc.load_btc(lo, hi)


def _sheet(series, label: str, **kw) -> dict:
    return tearsheet_from_returns(series, label=label, periods_per_year=PPY, **kw)


# ---------------------------------------------------------------------------


def replicate() -> None:
    frame = _load(None)
    print(
        f"BTCUSDT 5m: {frame.height:,} bars, "
        f"{frame['ts_open'].min()} .. {frame['ts_open'].max()}\n"
    )

    prof = btc.hourly_profile(frame)
    print("--- F1-F4: hourly profile, GMT ---")
    print("hour   ret(bp)   volume    trades      RV(1e6)   CS spread(bp)")
    for r in prof.iter_rows(named=True):
        print(
            f"{r['hour']:>4}  {1e2 * r['ret']:>8.3f}  {r['volume']:>8.1f}  "
            f"{r['trades']:>8.0f}  {1e6 * r['rv']:>10.3f}  {r['cs_spread_bps']:>10.2f}"
        )

    # F1/F2: is the 07:00-18:00 block genuinely more active?
    day_hours = prof.filter(pl.col("hour").is_between(7, 17))
    night_hours = prof.filter(~pl.col("hour").is_between(7, 17))
    print(
        f"\nF1 volume  07-18 GMT {day_hours['volume'].mean():.1f} vs rest "
        f"{night_hours['volume'].mean():.1f}  "
        f"(ratio {day_hours['volume'].mean() / night_hours['volume'].mean():.2f}x)"
    )
    print(
        f"F2 RV      07-18 GMT {1e6 * day_hours['rv'].mean():.3f} vs rest "
        f"{1e6 * night_hours['rv'].mean():.3f}  "
        f"(ratio {day_hours['rv'].mean() / night_hours['rv'].mean():.2f}x)"
    )
    print(
        f"F3 spread  07-18 GMT {day_hours['cs_spread_bps'].mean():.2f} vs rest "
        f"{night_hours['cs_spread_bps'].mean():.2f} bp"
    )
    rv = prof["rv"].to_numpy()
    sp = prof["cs_spread_bps"].to_numpy()
    vol = prof["volume"].to_numpy()
    print(f"F3 corr(RV, CS spread) across hours = {np.corrcoef(rv, sp)[0, 1]:+.3f}")
    print(f"   corr(volume, RV)   across hours = {np.corrcoef(vol, rv)[0, 1]:+.3f}")

    # F4: the return window, with a real test rather than an eyeball.
    sess = btc.session_returns(frame)
    inside = sess["in_pct"].to_numpy()
    outside = sess["out_pct"].to_numpy()
    print(f"\n--- F4: returns inside vs outside 08:00-16:00 GMT ({sess.height} days) ---")
    for name, x in (("inside", inside), ("outside", outside)):
        nw = newey_west(x)
        print(
            f"  {name:<8} mean {1e2 * nw.mean:+7.3f} bp/day  t {nw.t_stat:+5.2f}  "
            f"p {nw.p_value:.3f}   ann {1e2 * nw.mean * PPY / 100:+6.1f}%"
        )
    diff = newey_west(inside - outside)
    print(f"  {'diff':<8} mean {1e2 * diff.mean:+7.3f} bp/day  t {diff.t_stat:+5.2f}  p {diff.p_value:.3f}")

    # F5: contemporaneous correlations and the lead-lag structure.
    clean = frame.drop_nulls(["ret", "rv", "cs_spread"]).with_columns(
        pl.col("volume").log1p().alias("log_volume")
    )
    print("\n--- F5: contemporaneous correlations ---")
    pairs = [("ret", "volume"), ("ret", "rv"), ("ret", "cs_spread"),
             ("volume", "rv"), ("volume", "cs_spread"), ("rv", "cs_spread")]
    for a, b in pairs:
        x, y = clean[a].to_numpy(), clean[b].to_numpy()
        print(f"  corr({a:<10}, {b:<10}) = {np.corrcoef(x, y)[0, 1]:+.4f}")

    print("\n--- F5: cross-correlation of returns with lagged volume / RV ---")
    for other in ("volume", "rv"):
        cc = btc.cross_correlation(clean, "ret", other, max_lag=3)
        vals = {int(r["lag"]): r["corr"] for r in cc.iter_rows(named=True)}
        row = "  ret vs " + other.ljust(7) + "  " + "  ".join(
            f"j={j:+d}:{vals.get(j, float('nan')):+.4f}" for j in range(-3, 4)
        )
        print(row)
    print("  (j<0: the other variable leads returns; j>0: returns lead it)")

    print("\n--- F5: Granger causality, 7 lags ---")
    for a, b in [("volume", "ret"), ("ret", "volume"), ("rv", "ret"), ("ret", "rv"),
                 ("rv", "cs_spread"), ("cs_spread", "rv"), ("volume", "cs_spread")]:
        g = btc.granger(clean, a, b, lags=7)
        star = "***" if g["p_value"] < 0.01 else ("**" if g["p_value"] < 0.05 else "")
        print(f"  {a:<10} -> {b:<10}  F {g['f_stat']:>9.2f}  p {g['p_value']:.2e} {star}")

    prof.write_csv(OUT / "btc_hourly_profile.csv")
    btc.intraday_profile(frame).write_csv(OUT / "btc_tod_profile.csv")


def estimator() -> None:
    """What the Corwin-Schultz estimator is worth, where the truth is known."""
    print("--- Corwin-Schultz vs the quoted spread, 5-minute bars ---")
    print("(BTC klines carry no quotes, so this is checked on the broker corpus)")
    for sym in ("USTEC", "XAUUSD", "EURUSD"):
        bars = load_bars(sym, "1m", split="dev",
                         columns=["ts", "high", "low", "close", "spread_close"])
        b5 = (
            bars.group_by_dynamic("ts", every="5m")
            .agg(pl.col("high").max(), pl.col("low").min(),
                 pl.col("close").last(), pl.col("spread_close").mean())
            .with_columns(pl.col("ts").dt.date().alias("day"))
        )
        b5 = with_corwin_schultz(b5, over="day")
        e = estimator_error(b5)
        b5 = b5.with_columns(
            (pl.col("high") / pl.col("low")).log().alias("rng"),
            (pl.col("spread_close") / pl.col("close")).alias("true_prop"),
        ).filter(
            pl.col("cs_spread").is_finite() & pl.col("rng").is_finite()
            & pl.col("true_prop").is_finite()
        )
        r_est = np.corrcoef(b5["cs_spread"], b5["rng"])[0, 1]
        r_true = np.corrcoef(b5["true_prop"], b5["rng"])[0, 1]
        # And the aggregate the paper actually plots: means by time-of-day.
        agg = (
            b5.with_columns(
                (pl.col("ts").dt.hour().cast(pl.Int32) * 60
                 + pl.col("ts").dt.minute().cast(pl.Int32)).alias("tod")
            )
            .group_by("tod")
            .agg(pl.col("cs_spread").mean(), pl.col("true_prop").mean(), pl.len())
            .filter(pl.col("len") > 30)
        )
        r_agg = np.corrcoef(agg["cs_spread"], agg["true_prop"])[0, 1]
        print(
            f"\n  {sym}: est {e['mean_est_bps']:.2f} bp vs true {e['mean_true_bps']:.2f} bp "
            f"({e['ratio']:.2f}x), n={e['n']:,}"
        )
        print(f"    corr(est, true)  per bar          {e['corr']:+.3f}")
        print(f"    corr(est, true)  by time-of-day   {r_agg:+.3f}")
        print(f"    corr(est, high-low range)         {r_est:+.3f}")
        print(f"    corr(true, high-low range)        {r_true:+.3f}")


def _report(daily: pl.DataFrame, label: str, *, col: str = "net_bps") -> dict:
    sheet = _sheet(daily[col], label)
    print(format_returns_tearsheet(sheet))
    x = daily[col].to_numpy() / 1e4
    rng = np.random.default_rng(5)
    boot = bootstrap_ci(x, stat_fn=lambda a: sharpe(a, PPY), n_boot=1500, rng=rng)
    print(f"  bootstrap Sharpe 95% CI [{boot.lo:+.2f}, {boot.hi:+.2f}]  p={boot.p_value:.3f}")
    return sheet


def strategy(split: str, allow_test: bool = False) -> None:
    frame = _load(split, allow_test)
    print(f"BTCUSDT {split}: {frame.height:,} bars\n")

    # --- Strategy A: the session window ---------------------------------
    print("=" * 68)
    print("A. SessionStrategy - long 08:00-16:00 GMT, flat otherwise (F4)")
    print("=" * 68)
    sess = btc.session_strategy(frame)
    # Gross first. It separates "the pattern is not there" from "the pattern is
    # there and costs more than it pays" - two different conclusions that a net
    # number alone cannot tell apart.
    _report(sess, f"session 08-16 GROSS ({split})", col="gross_bps")
    _report(sess, f"session 08-16 net of fees ({split})")
    print(f"  cost charged: {sess['cost_bps'][0]:.1f} bp/day (2 round turns)")
    _report(sess, f"buy & hold, gross ({split})", col="bh_bps")
    _report(sess, f"outside-window only, gross ({split})", col="outside_bps")

    g = float(sess["gross_bps"].mean())
    print(
        f"\n  gross {g:+.2f} bp/day against {sess['cost_bps'][0]:.1f} bp/day "
        f"of cost: breakeven needs a round turn under {g / 2:.2f} bp, "
        f"i.e. a fee below {max(g / 2 - 3.0, 0):.2f} bp/side after the "
        f"estimated half-spread. Binance spot standard tier is 10 bp/side."
    )

    if split == "dev":
        print("\n  -- window sweep (dev only) --")
        rows = []
        for lo in (0, 6, 7, 8, 9, 12):
            for hi in (14, 16, 18, 20, 24):
                if hi - lo < 4:
                    continue
                s = btc.session_strategy(frame, btc.SessionConfig(lo_hour=lo, hi_hour=hi))
                sh = _sheet(s["net_bps"], "")
                gr = _sheet(s["gross_bps"], "")
                rows.append({"lo": lo, "hi": hi, "hours": hi - lo,
                             "gross_bp_day": round(float(s["gross_bps"].mean()), 2),
                             "gross_sharpe": round(gr["sharpe"], 2),
                             "net_sharpe": round(sh["sharpe"], 2)})
        with pl.Config(tbl_rows=40, tbl_width_chars=160):
            print(pl.DataFrame(rows).sort("gross_sharpe", descending=True).head(14))

    # --- Strategy B: the conditioned reversal ---------------------------
    print("\n" + "=" * 68)
    print("B. ReversalStrategy - fade the last bar when it was unusual (F5)")
    print("=" * 68)
    rows = []
    for signal in ("volume", "rv", "trades", "none"):
        for q in ((0.5, 0.8, 0.9, 0.95) if signal != "none" else (0.0,)):
            cfg = btc.ReversalConfig(signal=signal, quantile=q)
            bars = btc.reversal_strategy(frame, cfg)
            daily = btc.to_daily(bars)
            gross = _sheet(daily["gross_bps"], "")
            net = _sheet(daily["net_bps"], "")
            rows.append({
                "signal": signal,
                "q": q,
                "bars_in_pct": round(100 * float((bars["position"] != 0).mean()), 1),
                "gross_bp_bar": round(float(bars["gross_bps"].mean()), 4),
                "cost_bp_bar": round(float(bars["cost_bps"].mean()), 4),
                "gross_sharpe": round(gross["sharpe"], 2),
                "net_sharpe": round(net["sharpe"], 2),
                "net_cagr_pct": round(net["cagr_pct"], 1),
            })
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    # Is the gross edge even there before costs?
    best = btc.reversal_strategy(frame, btc.ReversalConfig(signal="volume", quantile=0.9))
    g = best.filter(pl.col("position") != 0)["gross_bps"].to_numpy()
    nw = newey_west(g)
    print(
        f"\n  gross edge per traded bar (volume q=0.9): {nw.mean:+.4f} bp  "
        f"t {nw.t_stat:+.2f}  p {nw.p_value:.3f}  n={nw.n:,}"
    )
    ctrl = btc.unconditional_reversal(frame)
    gc = ctrl.filter(pl.col("position") != 0)["gross_bps"].to_numpy()
    nwc = newey_west(gc)
    print(
        f"  gross edge per traded bar (unconditional): {nwc.mean:+.4f} bp  "
        f"t {nwc.t_stat:+.2f}  p {nwc.p_value:.3f}  n={nwc.n:,}"
    )
    print(f"  breakeven cost per round turn: {nw.mean / 2:.4f} bp "
          f"(Binance taker alone is 10 bp)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replicate", "strategy", "confirm", "test", "estimator"])
    ap.add_argument("--unlock", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode == "replicate":
        replicate()
    elif args.mode == "estimator":
        estimator()
    elif args.mode == "strategy":
        strategy("dev")
    elif args.mode == "confirm":
        strategy("validation")
    else:
        if not args.unlock:
            print("test split is locked; pass --unlock and mean it", file=sys.stderr)
            return 2
        strategy("test", allow_test=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
