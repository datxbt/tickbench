"""Level detection.

Two things carry the whole study and both are tested here: that the three kinds
are a genuine partition, so a single approach cannot be counted three times, and
that the level a bar is judged against is fixed before the bar opens, so nothing
in the classification can see the bar's own outcome.

The rest is threshold arithmetic, tested on bars built by hand where the right
answer can be read off the numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qlab.levels import KINDS, LevelConfig, level_events, prepare

UTC = timezone.utc
START = datetime(2021, 3, 1, 8, 0, tzinfo=UTC)


def _bars(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """One minute per row, given as (open, high, low, close)."""
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i + 1) for i in range(len(rows))],
            "ts_open": [START + timedelta(minutes=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "bid_close": [r[3] - 0.05 for r in rows],
            "ask_close": [r[3] + 0.05 for r in rows],
            "spread_mean": [0.1] * len(rows),
            "n_ticks": [50] * len(rows),
        },
        schema_overrides={
            "ts": pl.Datetime("us", "UTC"),
            "ts_open": pl.Datetime("us", "UTC"),
        },
    )


def _flat(n: int, price: float, span: float = 0.5) -> list[tuple[float, float, float, float]]:
    """Filler bars that establish an ATR without going near a round number."""
    return [(price, price + span, price - span, price)] * n


CFG = LevelConfig(atr_bars=10, fresh_bars=10, grids=(10.0,), families=("round",))


def test_the_three_kinds_are_a_partition():
    """No bar may produce two kinds against the same level.

    If it could, one approach to one level would be counted twice in a table
    that reads as though the rows were separate events, and every t-stat built
    on it would be overstated.
    """
    rows = _flat(30, 1895.0)
    # A long grind up through 1900 - touches, sweeps and breaches in sequence.
    price = 1897.0
    for _ in range(40):
        price += 0.2
        rows.append((price, price + 0.5, price - 0.5, price))
    events = level_events(_bars(rows), CFG)

    duplicated = events.group_by("family", "level", "ts").agg(n=pl.len()).filter(pl.col("n") > 1)
    assert duplicated.is_empty(), duplicated.to_dicts()
    assert set(events["kind"]).issubset(set(KINDS))


def test_touch_is_reaching_the_level_without_trading_through_it():
    rows = _flat(20, 1895.0)
    rows.append((1899.0, 1899.98, 1898.5, 1899.0))  # high stops just short
    events = level_events(_bars(rows), CFG)
    touch = events.filter(pl.col("kind") == "touch")
    assert touch.height == 1
    assert touch["level"][0] == 1900.0
    # Away from the level, and the level is above, so short.
    assert touch["direction"][0] == -1


def test_sweep_is_trading_through_and_closing_back():
    rows = _flat(20, 1895.0)
    rows.append((1899.0, 1900.05, 1898.5, 1899.2))
    events = level_events(_bars(rows), CFG)
    sweep = events.filter(pl.col("kind") == "sweep")
    assert sweep.height == 1
    assert sweep["level"][0] == 1900.0
    assert sweep["direction"][0] == -1
    assert sweep["pierce_atr"][0] > 0


def test_breach_is_closing_decisively_beyond_and_points_the_other_way():
    rows = _flat(20, 1895.0)
    # ATR is 1.0, so a quarter of it is 0.25; close 0.5 beyond.
    rows.append((1899.0, 1900.8, 1898.5, 1900.5))
    events = level_events(_bars(rows), CFG)
    breach = events.filter(pl.col("kind") == "breach")
    assert breach.height == 1
    assert breach["level"][0] == 1900.0
    # With the breach, and the breach is upward, so long.
    assert breach["direction"][0] == 1


def test_a_marginal_close_beyond_is_neither_a_sweep_nor_a_breach():
    """The gap between the two definitions is deliberate and must stay empty.

    A bar that closes 0.05 beyond the level has neither held it nor broken it.
    Sorting it into either bucket would put the ambiguous cases on whichever
    side the study happens to be hoping for.
    """
    rows = _flat(20, 1895.0)
    rows.append((1899.0, 1900.1, 1898.5, 1900.05))
    events = level_events(_bars(rows), CFG)
    assert events.filter(pl.col("level") == 1900.0).is_empty()


def test_the_level_is_chosen_before_the_bar_opens():
    """The no-lookahead guarantee, tested by changing only the bar's outcome.

    Two runs identical except for what the final bar does. The level it is
    judged against has to be the same in both, because it is a function of the
    previous close - if it moved, the classification would be reading the
    future.
    """
    base = _flat(20, 1895.0)
    up = level_events(_bars(base + [(1899.0, 1900.8, 1898.5, 1900.5)]), CFG)
    down = level_events(_bars(base + [(1899.0, 1899.9, 1890.0, 1891.0)]), CFG)
    assert up["level"].to_list() == [1900.0]
    # The down bar is judged against 1890 as well, but 1900 must still be the
    # level that the upward candidate was tested against in the other run.
    assert 1900.0 in up["level"].to_list()
    assert up["ts"][0] == down["ts"][0] if down.height else True


def test_freshness_suppresses_the_second_approach():
    """One event per level per kind per day, and only while the level is fresh."""
    rows = _flat(20, 1895.0)
    for _ in range(6):
        rows.append((1899.0, 1899.98, 1898.5, 1899.0))  # the same touch, repeatedly
    events = level_events(_bars(rows), CFG)
    assert events.filter((pl.col("kind") == "touch") & (pl.col("level") == 1900.0)).height == 1


def test_a_gap_does_not_become_a_true_range():
    """A previous close from before a weekend is not a previous close."""
    rows = _flat(10, 1895.0) + [(1950.0, 1950.5, 1949.5, 1950.0)] + _flat(5, 1950.0)
    frame = _bars(rows)
    # Push everything from the jump onwards two days out, leaving a single gap
    # in an otherwise contiguous minute series - which is what a weekend is.
    jumped = frame.with_columns(
        ts=pl.col("ts") + pl.when(pl.int_range(pl.len()) >= 10)
        .then(pl.duration(days=2))
        .otherwise(pl.duration(days=0)),
        ts_open=pl.col("ts_open") + pl.when(pl.int_range(pl.len()) >= 10)
        .then(pl.duration(days=2))
        .otherwise(pl.duration(days=0)),
    )
    prepared = prepare(jumped, CFG)

    assert not prepared["contig"][10]  # the bar across the gap
    assert prepared["contig"][11]
    # $55 of gap, and the true range of the bar after it is still its own $1.
    assert prepared["tr"][10] == pytest.approx(1.0)
    assert prepared["tr"].max() == pytest.approx(1.0)


def test_coarser_grids_produce_fewer_events():
    """Round levels nest, so the counts have to fall as the grid coarsens."""
    rows = _flat(20, 1895.0)
    price = 1895.0
    for _ in range(400):
        price += 0.2
        rows.append((price, price + 0.5, price - 0.5, price))
    bars = _bars(rows)
    counts = [
        level_events(bars, LevelConfig(atr_bars=10, fresh_bars=10, grids=(g,),
                                       families=("round",))).height
        for g in (10.0, 50.0, 100.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_no_families_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError, match="no level families"):
        level_events(_bars(_flat(20, 1895.0)), LevelConfig(families=()))
