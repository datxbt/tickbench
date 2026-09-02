"""Write reports/cost_model/COST_MODEL.md from the measured profiles.

Everything here is derived, so the report can be regenerated after any change to
the profiles or to the model's assumptions and will disagree with itself if one
of them has moved.

Usage
-----
    python scripts/cost_report.py
    python scripts/cost_report.py --adverse 1.0 --latency 500
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import polars as pl

from qlab import paths
from qlab.costprofile import HORIZONS_MS
from qlab.costs import CostModel, SlippageModel
from qlab.loader import SPLITS, load_bars
from qlab.symbols import ALL_SYMBOLS, get_spec

TRAILING = 3
SESSION_NAMES = {
    (0, 7): "Tokyo",
    (7, 12): "London",
    (12, 16): "London/NY overlap",
    (16, 21): "New York",
    (21, 24): "Sydney",
}


def _reference_prices(trailing_months: int) -> dict[str, float]:
    """Mean close over the same trailing months the profile is built from.

    Basis points need a price to be points of, and gold ran 2300 to 4400 across
    this corpus - so the price has to come from the window being costed, not
    from the corpus as a whole.
    """
    prices = {}
    for symbol in ALL_SYMBOLS:
        lazy = load_bars(symbol, "1m", columns=["ts", "close"], lazy=True)
        last = lazy.select(pl.col("ts").max()).collect().item()
        months_back = last.year * 12 + (last.month - 1) - (trailing_months - 1)
        start = date(months_back // 12, months_back % 12 + 1, 1)
        prices[symbol] = (
            lazy.filter(pl.col("ts") >= start)
            .select(pl.col("close").mean())
            .collect()
            .item()
        )
    return prices


def _fmt(value: float, digits: int = 3) -> str:
    return "-" if value != value else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adverse", type=float, default=0.5)
    parser.add_argument("--latency", type=int, default=250)
    args = parser.parse_args()

    paths.ensure_dirs()
    slippage = SlippageModel(latency_ms=args.latency, adverse_fraction=args.adverse)
    prices = _reference_prices(TRAILING)

    models = {
        s: CostModel.from_profiles(s, trailing_months=TRAILING, slippage=slippage)
        for s in ALL_SYMBOLS
    }
    order = sorted(ALL_SYMBOLS, key=lambda s: models[s].round_turn_bps(prices[s]))

    out: list[str] = []
    w = out.append

    w("# Cost model")
    w("")
    w(
        f"Generated from the measured spread and latency profiles over the trailing "
        f"{TRAILING} months. Slippage assumes **{args.adverse:g} x** the measured mid "
        f"drift over a **{args.latency} ms** fill latency; that fraction is an "
        f"assumption about the strategy, not a measurement, and every figure below "
        f"moves with it."
    )
    w("")

    # ---- 1. headline -----------------------------------------------------
    w("## What a round turn costs")
    w("")
    w("| symbol | spread | commission | slippage | total | | in bps | USD/lot |")
    w("| --- | ---: | ---: | ---: | ---: | :-- | ---: | ---: |")
    for symbol in order:
        model = models[symbol]
        parts = model.breakdown(prices[symbol])
        w(
            f"| {symbol} | {_fmt(parts['spread_pips'])} "
            f"| {_fmt(parts['commission_pips'])} "
            f"| {_fmt(parts['slippage_pips'])} "
            f"| **{_fmt(parts['round_turn_pips'])}** | pips "
            f"| {_fmt(parts['round_turn_bps'])} "
            f"| {parts['round_turn_usd']:.2f} |"
        )
    w("")
    w("Share of the total:")
    w("")
    w("| symbol | spread | commission | slippage |")
    w("| --- | ---: | ---: | ---: |")
    for symbol in order:
        parts = models[symbol].breakdown(prices[symbol])
        w(
            f"| {symbol} | {100 * parts['spread_share']:.0f}% "
            f"| {100 * parts['commission_share']:.0f}% "
            f"| {100 * parts['slippage_share']:.0f}% |"
        )
    w("")

    # ---- 2. the slippage band -------------------------------------------
    w("## How much of this is the slippage assumption")
    w("")
    w(
        "The adverse fraction is the one number here that cannot be measured from "
        "the tape, so the honest presentation is a band rather than a point. "
        "`0.0` is a fill uncorrelated with the move, `1.0` a strategy that chases."
    )
    w("")
    w("| symbol | 0.0 (none) | 0.5 (default) | 1.0 (chasing) | spread of the band |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for symbol in order:
        price = prices[symbol]
        values = []
        for fraction in (0.0, 0.5, 1.0):
            model = CostModel.from_profiles(
                symbol,
                trailing_months=TRAILING,
                slippage=SlippageModel(latency_ms=args.latency, adverse_fraction=fraction),
            )
            values.append(model.round_turn_bps(price))
        w(
            f"| {symbol} | {_fmt(values[0])} | {_fmt(values[1])} | {_fmt(values[2])} "
            f"| {values[2] / values[0]:.2f}x |"
        )
    w("")

    # ---- 3. by hour ------------------------------------------------------
    w("## When to trade")
    w("")
    w(
        "Cost is not spread evenly across the day. Hours with no quotes at all - "
        "the daily maintenance break - are marked closed rather than cheap."
    )
    w("")
    w("| symbol | cheapest hours | dearest hours | worst / best | closed |")
    w("| --- | --- | --- | ---: | --- |")
    for symbol in order:
        table = models[symbol].hourly_table(prices[symbol]).filter(pl.col("is_open"))
        closed = (
            models[symbol]
            .hourly_table(prices[symbol])
            .filter(~pl.col("is_open"))["hour"]
            .to_list()
        )
        ranked = table.sort("round_turn_pips")
        cheap = ", ".join(
            f"{r['hour']:02d}h {r['round_turn_pips']:.2f}" for r in ranked.head(3).to_dicts()
        )
        dear = ", ".join(
            f"{r['hour']:02d}h {r['round_turn_pips']:.2f}" for r in ranked.tail(3).to_dicts()
        )
        ratio = ranked["round_turn_pips"][-1] / ranked["round_turn_pips"][0]
        w(
            f"| {symbol} | {cheap} | {dear} | {ratio:.2f}x "
            f"| {', '.join(f'{h:02d}h' for h in closed) or 'none'} |"
        )
    w("")
    w("Spread at the Sunday reopen against the weekday mean:")
    w("")
    w("| symbol | weekday | Sunday | ratio |")
    w("| --- | ---: | ---: | ---: |")
    for symbol in order:
        model = models[symbol]
        weekday = model.spread_pips(is_sunday=False)
        sunday = model.spread_pips(is_sunday=True)
        w(
            f"| {symbol} | {_fmt(weekday, 4)} | {_fmt(sunday, 4)} "
            f"| {sunday / weekday:.1f}x |"
        )
    w("")

    # ---- 4. by split -----------------------------------------------------
    w("## Cost is not constant across the splits")
    w("")
    w(
        "Spreads have compressed since 2020, so a backtest run on the dev split "
        "with today's costs would be flattered. The cost model takes the same "
        "`split=` argument the loader does, and should always be given it."
    )
    w("")
    w("| symbol | dev | validation | test |")
    w("| --- | ---: | ---: | ---: |")
    for symbol in order:
        cells = []
        for name in ("dev", "validation", "test"):
            model = CostModel.from_profiles(symbol, split=name, slippage=slippage)
            bars = load_bars(
                symbol, "1m", split=name, allow_test=True, columns=["ts", "close"]
            )
            price = float(bars["close"].mean())
            cells.append(f"{model.round_turn_bps(price):.3f}")
        w(f"| {symbol} | {cells[0]} | {cells[1]} | {cells[2]} |")
    w("")
    w("(bps per round turn, priced at each split's own mean close.)")
    w("")

    # ---- 5. latency ------------------------------------------------------
    w("## What execution speed is worth")
    w("")
    w("| symbol | " + " | ".join(f"{h} ms" for h in HORIZONS_MS) + " | 1000 vs 50 |")
    w("| --- | " + " | ".join("---:" for _ in HORIZONS_MS) + " | ---: |")
    for symbol in order:
        values = []
        for horizon in HORIZONS_MS:
            model = CostModel.from_profiles(
                symbol,
                trailing_months=TRAILING,
                slippage=SlippageModel(
                    latency_ms=horizon, adverse_fraction=args.adverse
                ),
            )
            values.append(model.round_turn_bps(prices[symbol]))
        w(
            f"| {symbol} | "
            + " | ".join(_fmt(v) for v in values)
            + f" | {values[-1] / values[0]:.2f}x |"
        )
    w("")

    # ---- 6. turnover -----------------------------------------------------
    w("## The turnover budget")
    w("")
    w(
        "Annualised cost drag, in basis points, at a given number of round turns "
        "per trading day. This is the number a strategy's gross return has to "
        "clear before it has made anything."
    )
    w("")
    w("| symbol | 1/day | 5/day | 20/day |")
    w("| --- | ---: | ---: | ---: |")
    for symbol in order:
        model = models[symbol]
        w(
            f"| {symbol} | "
            + " | ".join(
                f"{model.annual_drag_bps(prices[symbol], round_turns_per_day=n):.0f}"
                for n in (1, 5, 20)
            )
            + " |"
        )
    w("")

    # ---- 7. limits -------------------------------------------------------
    w("## What this model does not cover")
    w("")
    w(
        "- **Market impact.** The feed carries no size, so there is no way to "
        "measure how a larger order moves the price. `impact_pips_per_lot` "
        "exists and defaults to zero; at retail size that is probably right, and "
        "at any size where it is not, this model is the wrong tool."
    )
    w(
        "- **Rejections and requotes.** Modelled as though every order fills. On "
        "market execution the practical equivalent is a worse fill, which the "
        "slippage term absorbs only in expectation."
    )
    w(
        "- **Swap and financing.** Positions held overnight pay or receive swap, "
        "which is published per symbol and is not in this model. Anything holding "
        "past 21:00 UTC needs it added."
    )
    w(
        "- **Two inferred contract sizes.** Gold's 100 oz lot and USTEC's single "
        "index point are inferred, not published. Every pip-denominated figure "
        "here scales linearly with them."
    )
    w("")

    path = paths.COST_REPORT_DIR / "COST_MODEL.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
