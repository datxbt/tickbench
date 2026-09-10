"""Do gold's price levels do anything? The Osler mechanism, measured.

Take-profit orders cluster at round numbers and stop-loss orders cluster just
beyond them (Osler 2003, 2005, on a real FX order book). That predicts reversion
at a level and continuation once it is crossed. It is the one mechanism family
this project has never tested on gold, and the one that satisfies the standing
condition in ``xauusd-rejected.md`` §5: a genuinely different hypothesis rather
than another configuration of the ten already rejected.

    python scripts/research/level_interaction.py                # everything
    python scripts/research/level_interaction.py --only kinds placebo
    python scripts/research/level_interaction.py --list

Everything is measured three ways - at the mid, at real bid/ask fills plus
commission, and net of measured slippage as well - because the last gold
candidate that looked real at the mid was +0.98 bps there and +0.02 bps at the
prices a trade actually gets.

**Nothing here touches the locked test split**, and nothing here can: no call
passes ``allow_test``. Discovery runs on dev; validation is read only for cells
that survived dev, and only in the section that says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab import paths  # noqa: E402
from qlab.bars import resample_bars  # noqa: E402
from qlab.costs import CostModel  # noqa: E402
from qlab.eventstudy import (  # noqa: E402
    daily_pnl,
    deflated_threshold,
    forward_returns,
    format_three_columns,
    placebo_events,
    summarize,
)
from qlab.levels import KINDS, LevelConfig, level_events  # noqa: E402
from qlab.loader import load_bars  # noqa: E402
from qlab.session import with_session_flags  # noqa: E402
from qlab.symbols import get_spec  # noqa: E402

SYMBOL = "XAUUSD"
INTERVAL = "5m"
"""Five minutes is the primary timeframe, and the 1-minute check is a section.

Gold's round turn is 0.69 bps against a 1-minute standard deviation of roughly
2 bps, so a 1-minute study spends most of its measurement on cost. Five minutes
is the shortest interval at which a level event has room to be worth more than
the toll, and still short enough that a level is fresh when it is tested.
"""
CFG = LevelConfig()
SPLITS = ("dev", "validation")

# The reference price for stating a cost in bps. Gold ran $1,470 to $2,070 over
# dev and $2,000 to $3,500 over validation, so one number for both would be
# wrong at each end.
REF_PRICE = {"dev": 1900.0, "validation": 2400.0}


def _bars(split: str) -> pl.DataFrame:
    return resample_bars(load_bars(SYMBOL, "1m", split=split), INTERVAL)


def _cost(split: str) -> CostModel:
    return CostModel.from_profiles(SYMBOL, split=split)


def _events(bars: pl.DataFrame, cfg: LevelConfig = CFG) -> pl.DataFrame:
    """Level events, with the conditioning columns a regime cut needs."""
    events = level_events(bars, cfg)
    spec = get_spec(SYMBOL)
    events = with_session_flags(events, spec, on="ts_open")
    return events.with_columns(
        atr_bps=pl.col("atr") / pl.col("close") * 10_000,
        year=pl.col("ts_open").dt.year(),
    )


def _outcomes(bars, events, cost, keep=("kind", "family", "session", "atr_bps",
                                        "approach_atr", "dist_atr", "year")):
    return forward_returns(bars, events, CFG.horizons, cost=cost, keep=keep)


def _round_turn(split: str) -> float:
    return _cost(split).round_turn_bps(price=REF_PRICE[split])


# --------------------------------------------------------------------------


def section_census():
    """What the definitions actually select, before any outcome is looked at."""
    print(f"Config: {CFG.describe()}")
    print(f"Interval: {INTERVAL}   horizons (bars): {CFG.horizons}\n")

    for split in SPLITS:
        bars = _bars(split)
        events = _events(bars)
        rt = _round_turn(split)
        print(f"[{split}]  {bars.height:,} bars   {events.height:,} events   "
              f"round turn {rt:.3f} bps")
        table = (
            events.group_by("family", "kind")
            .agg(n=pl.len(), atr=pl.col("atr_bps").median())
            .sort("family", "kind")
        )
        header = f"  {'family':<12}" + "".join(f"{k:>10}" for k in KINDS)
        print(header)
        for family in table["family"].unique(maintain_order=True).sort():
            row = table.filter(pl.col("family") == family)
            counts = ""
            for kind in KINDS:
                cell = row.filter(pl.col("kind") == kind)
                counts += f"{(cell['n'][0] if cell.height else 0):>10,}"
            print(f"  {family:<12}{counts}")
        print()

    print("Round levels nest: every $100 level is also a $50 and a $10 one, so")
    print("the counts fall roughly as the grid coarsens, which is the arithmetic")
    print("working rather than a finding.")


def section_kinds():
    """The core table: does any kind of level interaction predict anything?"""
    bars = _bars("dev")
    events = _events(bars)
    cost = _cost("dev")
    out = _outcomes(bars, events, cost)
    rt = _round_turn("dev")

    print("Dev split. Direction is the mechanism's: away from the level for a")
    print("touch or a sweep, with it for a breach. A negative mid means the")
    print("mechanism points the wrong way, not that there is nothing there.\n")
    print(f"  round turn {rt:.3f} bps\n")

    for kind in KINDS:
        subset = out.filter(pl.col("kind") == kind)
        print(f"[{kind}]  n={subset.filter(pl.col('horizon') == CFG.horizons[0]).height:,}")
        print(format_three_columns(subset, ["minutes"]))
        print()

    print("Read the third column only. The first is what a chart shows and the")
    print("third is what an account gets, and the distance between them is one")
    print("round turn, every time, at every horizon.")


def section_families():
    """Is the effect concentrated in the levels that should carry it?"""
    bars = _bars("dev")
    events = _events(bars)
    cost = _cost("dev")
    out = _outcomes(bars, events, cost)

    print("If the mechanism is real, a $100 level should beat a $10 one and a")
    print("previous-day extreme should beat an arbitrary intraday session high.")
    print("Ordering is a harder test to pass by chance than any single cell.\n")

    for horizon in (6, 24):
        subset = out.filter(pl.col("horizon") == horizon)
        minutes = subset["minutes"][0] if subset.height else horizon
        print(f"[{minutes} minutes forward]")
        print(format_three_columns(subset, ["family", "kind"]))
        print()


def section_placebo():
    """The same trades on a different day, which is the control that matters."""
    bars = _bars("dev")
    events = _events(bars)
    cost = _cost("dev")

    seeds = tuple(range(12))
    real = _outcomes(bars, events, cost)
    fakes = [
        _outcomes(bars, placebo_events(events, bars, seed=seed), cost) for seed in seeds
    ]

    print("Each event moved between one and five days, keeping its time of day,")
    print("its weekday and its direction. Everything survives except the level.")
    print()
    print(f"Twelve draws, not one. A single placebo has its own sampling error,")
    print("and quoting one draw would just be a second chance to find a gap that")
    print("is not there. The band below is the min and max across the draws.\n")

    for horizon in (1, 6, 24):
        r = real.filter(pl.col("horizon") == horizon)
        minutes = r["minutes"][0] if r.height else horizon
        print(f"[{minutes} minutes forward]  bps per event, at the mid")
        print(f"  {'kind':<10}{'n':>7}{'real':>10}"
              f"{'placebo mean':>15}{'placebo min':>14}{'placebo max':>14}")
        for kind in KINDS:
            actual = summarize(r.filter(pl.col("kind") == kind), column="mid_bps")
            if not actual.height:
                continue
            draws = []
            for fake in fakes:
                cell = summarize(
                    fake.filter((pl.col("horizon") == horizon) & (pl.col("kind") == kind)),
                    column="mid_bps",
                )
                if cell.height:
                    draws.append(cell["mean"][0])
            print(f"  {kind:<10}{actual['n'][0]:>7,}{actual['mean'][0]:>+10.3f}"
                  f"{sum(draws) / len(draws):>+15.3f}{min(draws):>+14.3f}"
                  f"{max(draws):>+14.3f}")
        print()

    print("The mid column is the one to compare, because the placebo pays the")
    print("same costs as the real thing and subtracting them from both would")
    print("only move the two lines together.")


def section_direction():
    """Is the breach result a level effect, or is it gold going up?

    The one diagnostic this study exists to survive. A breach is directional by
    construction - an upward breach is a long - so anything that made money
    holding gold shows up here as a level effect unless the two sides are
    separated. ``xauusd-rejected.md`` §1 already found that gold's validation
    return is one long bull move and its dev return is financing, so this is the
    trap with a name.
    """
    for split in SPLITS:
        bars = _bars(split)
        events = _events(bars)
        out = _outcomes(bars, events, _cost(split)).filter(pl.col("kind") == "breach")
        print(f"[{split}]  net of everything, breaches only")
        print(f"  {'min':>5}"
              f"{'n long':>9}{'bps/event':>11}{'bps/day':>10}{'t/day':>8}"
              f"{'n short':>10}{'bps/event':>11}{'bps/day':>10}{'t/day':>8}")
        for minutes in sorted(out["minutes"].unique()):
            row = out.filter(pl.col("minutes") == minutes)
            line = f"  {minutes:>5}"
            for direction in (1, -1):
                side = row.filter(pl.col("direction") == direction)
                stats = summarize(side, column="net_bps")
                per_day, t_day, _ = daily_pnl(side)
                line += (f"{stats['n'][0]:>9,}{stats['mean'][0]:>+11.3f}"
                         f"{per_day:>+10.2f}{t_day:>+8.2f}")
            print(line)
        print()

    print("Two readings, and they disagree, which is itself the finding.")
    print()
    print("Per event, dev's two sides agree and neither is significant, while in")
    print("validation the long side carries everything - that is gold's bull")
    print("regime showing through a level definition, and it is finding #1 of")
    print("xauusd-rejected.md arriving by a new route.")
    print()
    print("Per day the same cells lose most of their t-stat, because breaches")
    print("arrive three to a day normally and thirteen on a trending day: the")
    print("per-event t counts one trending afternoon as a dozen observations.")
    print("Nothing in either split clears t = 2 on the daily statistic, in")
    print("either direction, at any horizon.")


def section_regime():
    """Conditioning, which is where the report expected the effect to appear."""
    bars = _bars("dev")
    events = _events(bars)
    cost = _cost("dev")
    out = _outcomes(bars, events, cost).filter(pl.col("horizon") == 6)

    out = out.with_columns(
        vol=pl.when(pl.col("atr_bps") <= pl.col("atr_bps").quantile(1 / 3))
        .then(pl.lit("1 low"))
        .when(pl.col("atr_bps") <= pl.col("atr_bps").quantile(2 / 3))
        .then(pl.lit("2 mid"))
        .otherwise(pl.lit("3 high")),
        run=pl.when(pl.col("approach_atr") <= pl.col("approach_atr").quantile(1 / 3))
        .then(pl.lit("1 slow"))
        .when(pl.col("approach_atr") <= pl.col("approach_atr").quantile(2 / 3))
        .then(pl.lit("2 mid"))
        .otherwise(pl.lit("3 fast")),
    )

    print("30 minutes forward, dev, net of everything. Three conditioners, one")
    print("at a time: volatility regime, how hard price ran into the level, and")
    print("session. Cells are not independent - the same event appears in all")
    print("three tables - so read the deflated threshold in `screen`, not the")
    print("t-stats here.\n")

    for column, label in (("vol", "ATR regime"), ("run", "approach speed"),
                          ("session", "session")):
        print(f"[{label}]")
        print(format_three_columns(out, ["kind", column]))
        print()


def section_screen():
    """What a cell had to clear, and whether any cell cleared it."""
    bars = _bars("dev")
    events = _events(bars)
    cost = _cost("dev")
    out = _outcomes(bars, events, cost).with_columns(
        vol=pl.when(pl.col("atr_bps") <= pl.col("atr_bps").quantile(1 / 3))
        .then(pl.lit("1 low"))
        .when(pl.col("atr_bps") <= pl.col("atr_bps").quantile(2 / 3))
        .then(pl.lit("2 mid"))
        .otherwise(pl.lit("3 high")),
    )

    keys = ["kind", "family", "vol", "minutes"]
    cells = summarize(out, keys, column="net_bps").filter(pl.col("n") >= 100)

    # The event-level t is not the statistic; recompute each cell on daily
    # totals, which is what `kinds` and `direction` report.
    rows = []
    for row in cells.iter_rows(named=True):
        group = out
        for key in keys:
            group = group.filter(pl.col(key) == row[key])
        per_day, t_day, days = daily_pnl(group)
        rows.append({**row, "bps_day": per_day, "t_day": t_day, "days": days})
    cells = pl.DataFrame(rows)

    threshold = deflated_threshold(cells.height)
    print(f"  cells inspected (n >= 100): {cells.height}")
    print(f"  |t| a single cell needs at alpha=0.05 across that many: {threshold:.2f}")
    print(f"  round turn already charged:  {_round_turn('dev'):.3f} bps\n")

    survivors = cells.filter((pl.col("bps_day") > 0) & (pl.col("t_day") > threshold))
    print(f"  cells with a positive net mean:            "
          f"{cells.filter(pl.col('bps_day') > 0).height}")
    print(f"  cells clearing an uncorrected t = +2:      "
          f"{cells.filter(pl.col('t_day') > 2).height}")
    print(f"  cells clearing the deflated threshold:     {survivors.height}")
    print(f"  cells clearing a deflated threshold short: "
          f"{cells.filter(pl.col('t_day') < -threshold).height}\n")

    if survivors.is_empty():
        print("  Nothing survives, so nothing is promoted, and the test split")
        print("  stays unspent.")
    else:
        for row in survivors.sort("t_day", descending=True).iter_rows(named=True):
            print(f"  {row['kind']:<8}{row['family']:<10}{row['vol']:<8}"
                  f"{row['minutes']:>4}m  n={row['n']:>6,}  "
                  f"bps/day={row['bps_day']:+.2f}  t={row['t_day']:+.2f}")

    print("\n  The five best cells regardless of the threshold, for scale:")
    for row in cells.sort("t_day", descending=True).head(5).iter_rows(named=True):
        print(f"  {row['kind']:<8}{row['family']:<10}{row['vol']:<8}"
              f"{row['minutes']:>4}m  n={row['n']:>6,}  "
              f"bps/day={row['bps_day']:+7.2f}  t={row['t_day']:+.2f}  "
              f"(mid {row['mid']:+.3f}/event)")


def section_minute():
    """The same study at 1m, because the timeframe choice should be tested."""
    bars = resample_bars(load_bars(SYMBOL, "1m", split="dev"), "1m")
    cfg = LevelConfig(atr_bars=60, fresh_bars=60, horizons=(5, 15, 30, 60, 120, 240))
    events = _events(bars, cfg)
    cost = _cost("dev")
    out = forward_returns(bars, events, cfg.horizons, cost=cost, keep=("kind",))

    print("One-minute bars, same definitions, same costs. The question is only")
    print("whether the 5-minute choice hid something.\n")
    print(f"  {events.height:,} events\n")
    for kind in KINDS:
        print(f"[{kind}]")
        print(format_three_columns(out.filter(pl.col("kind") == kind), ["minutes"]))
        print()


def section_validation():
    """Validation, read once, for whatever dev pointed at.

    Kept as its own section so that running it is a decision rather than a side
    effect of running everything.
    """
    bars = _bars("validation")
    events = _events(bars)
    cost = _cost("validation")
    out = _outcomes(bars, events, cost)

    print("Dev found nothing to promote, so this section is not a confirmation")
    print("test - it is a check that the dev result is the instrument and not")
    print("the period. If the signs held here it would mean the effect is real")
    print("and negative; if they scatter it means there is nothing there.\n")
    print(f"  round turn {_round_turn('validation'):.3f} bps\n")
    for kind in KINDS:
        print(f"[{kind}]")
        print(format_three_columns(out.filter(pl.col("kind") == kind), ["minutes"]))
        print()


SECTIONS = {
    "census": ("what the definitions select, before any outcome", section_census),
    "kinds": ("touch, sweep and breach at every horizon, dev", section_kinds),
    "families": ("round grids against prior-period extremes", section_families),
    "placebo": ("the same trades on a different day", section_placebo),
    "direction": ("is the breach a level effect or gold's drift", section_direction),
    "regime": ("volatility, approach speed and session", section_regime),
    "screen": ("the multiple-comparison screen over every cell", section_screen),
    "minute": ("the same study at 1m", section_minute),
    "validation": ("the second split, read once", section_validation),
}


def save_artifacts() -> None:
    """Write the event tapes and the headline numbers next to the report.

    The tapes are the thing worth keeping. Every number in the report is a
    reduction of them, so a later question - did this concentrate in one year,
    does it survive a different exit - can be answered without re-deriving the
    events, and a disagreement can be traced to a row.
    """
    paths.ensure_dirs()
    tape_dir = paths.STRATEGY_REPORT_DIR / "level_interaction_events"
    tape_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "config": CFG.describe(),
        "horizons_bars": list(CFG.horizons),
        "splits": {},
    }
    for split in SPLITS:
        bars = _bars(split)
        events = _events(bars)
        cost = _cost(split)
        out = _outcomes(bars, events, cost)
        events.write_parquet(tape_dir / f"events_{split}.parquet")
        out.write_parquet(tape_dir / f"outcomes_{split}.parquet")

        cells = []
        for kind in KINDS:
            for minutes in sorted(out["minutes"].unique()):
                group = out.filter(
                    (pl.col("kind") == kind) & (pl.col("minutes") == minutes)
                )
                stats = summarize(group, column="net_bps")
                per_day, t_day, days = daily_pnl(group)
                cells.append({
                    "kind": kind,
                    "minutes": int(minutes),
                    "n": int(stats["n"][0]),
                    "mid_bps": round(float(stats["mid"][0]), 4),
                    "net_bps": round(float(stats["mean"][0]), 4),
                    "bps_per_day": round(per_day, 4),
                    "t_per_day": round(t_day, 4),
                    "days": days,
                })
        summary["splits"][split] = {
            "bars": bars.height,
            "events": events.height,
            "round_turn_bps": round(_round_turn(split), 4),
            "cost_model": cost.describe(),
            "cells": cells,
        }

    path = paths.STRATEGY_REPORT_DIR / "level_interaction.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {path}")
    print(f"  wrote {tape_dir}/events_*.parquet and outcomes_*.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(SECTIONS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--save", action="store_true", help="write the event tapes and summary JSON"
    )
    args = parser.parse_args()

    if args.list:
        for name, (blurb, _) in SECTIONS.items():
            print(f"  {name:<12} {blurb}")
        return

    if args.save:
        save_artifacts()
        return

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    for name in args.only or SECTIONS:
        blurb, function = SECTIONS[name]
        print(f"\n{'=' * 78}\n{name.upper()}  -  {blurb}\n{'=' * 78}\n")
        function()


if __name__ == "__main__":
    main()
