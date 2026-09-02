"""Cleaning policy and split discipline.

The cleaning tests are about *losslessness*: the audit's claim that dropping
duplicates costs nothing holds only for exact duplicates, so the report has to
keep the two cases apart. The split tests are about the guardrail actually
guarding.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from qlab.loader import (
    RAW_POLICY,
    SPLITS,
    UNIQUE_TS_POLICY,
    CleaningPolicy,
    SplitLockedError,
    clean_ticks,
    cleaning_report,
    get_split,
    load_ticks,
)

UTC = timezone.utc


def _ticks(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime.fromisoformat(r[0]).replace(tzinfo=UTC) for r in rows],
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        },
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
    )


def _write_month(tmp_path, symbol: str, year: int, month: int, frame: pl.DataFrame):
    directory = tmp_path / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / f"{symbol}_{year:04d}_{month:02d}.parquet")


def test_repeated_rows_are_dropped_losslessly():
    """A tick identical to its predecessor carries nothing, so it is free to drop."""
    ticks = _ticks(
        [
            ("2024-06-03T00:00:01", 1.0847, 1.0847),
            ("2024-06-03T00:00:02", 1.0848, 1.0849),
            ("2024-06-03T00:00:02", 1.0848, 1.0849),  # byte-identical repeat
        ]
    )
    report = cleaning_report(ticks)
    cleaned = clean_ticks(ticks)

    assert report["duplicate_ts_rows"] == 1
    assert report["repeated_rows"] == 1
    assert report["distinct_quotes_at_shared_ts"] == 0
    assert cleaned.height == 2
    assert cleaned["bid"].to_list() == [1.0847, 1.0848]


def test_different_quotes_in_one_millisecond_are_kept():
    """The correction that Stage 1 forced on Stage 0's reading of the duplicates.

    The feed stamps to the millisecond, so a fast market puts two real, sequential
    quotes on one timestamp. They are not repeats, and collapsing them deletes
    prices - preferentially in the fastest, most volatile moments, which is where
    a high or a low is least affordable to lose.
    """
    ticks = _ticks(
        [
            ("2024-06-03T00:00:01", 1.0847, 1.0847),
            ("2024-06-03T00:00:02", 1.0848, 1.0849),
            ("2024-06-03T00:00:02", 1.0850, 1.0851),  # same ms, book moved
        ]
    )
    report = cleaning_report(ticks)

    assert report["distinct_quotes_at_shared_ts"] == 1
    assert report["repeated_rows"] == 0
    assert clean_ticks(ticks)["bid"].to_list() == [1.0847, 1.0848, 1.0850]

    # Collapsing is available for consumers that need a unique index, and then
    # "last" is the state of the book leaving that millisecond.
    assert clean_ticks(ticks, UNIQUE_TS_POLICY)["bid"].to_list() == [1.0847, 1.0850]
    assert clean_ticks(ticks, CleaningPolicy(duplicate_ts="first"))["bid"].to_list() == [
        1.0847,
        1.0848,
    ]


def test_a_flickering_quote_keeps_its_return_leg():
    """Why repeats are defined as adjacent rather than globally distinct.

    A quote that moves away and comes back inside one millisecond is a sequence
    of three real prints; a global uniqueness test would drop the third and
    change what the book was when the millisecond ended.
    """
    ticks = _ticks(
        [
            ("2024-06-03T00:00:02", 1.0848, 1.0849),
            ("2024-06-03T00:00:02", 1.0850, 1.0851),
            ("2024-06-03T00:00:02", 1.0848, 1.0849),  # back to the first quote
        ]
    )

    assert cleaning_report(ticks)["repeated_rows"] == 0
    assert clean_ticks(ticks).height == 3
    assert clean_ticks(ticks, UNIQUE_TS_POLICY)["bid"].to_list() == [1.0848]


def test_crossed_and_nonpositive_quotes_are_dropped_and_counted():
    ticks = _ticks(
        [
            ("2024-06-03T00:00:01", 1.0847, 1.0848),
            ("2024-06-03T00:00:02", 1.0850, 1.0849),  # crossed: ask < bid
            ("2024-06-03T00:00:03", 0.0, 1.0849),  # unusable
            ("2024-06-03T00:00:04", 1.0851, 1.0852),
        ]
    )
    report = cleaning_report(ticks)

    assert report["crossed"] == 1
    assert report["nonpositive"] == 1
    assert report["rows_after_policy"] == 2
    assert clean_ticks(ticks).height == 2


def test_raw_policy_changes_nothing():
    """The escape hatch has to be a genuine no-op, or it cannot be used to audit."""
    ticks = _ticks(
        [
            ("2024-06-03T00:00:02", 1.0848, 1.0849),
            ("2024-06-03T00:00:02", 1.0848, 1.0849),
            ("2024-06-03T00:00:03", 1.0850, 1.0849),  # crossed
        ]
    )
    assert clean_ticks(ticks, RAW_POLICY).equals(ticks)


def test_policy_rejects_an_unknown_duplicate_rule():
    with pytest.raises(ValueError, match="last/first/keep"):
        CleaningPolicy(duplicate_ts="mean")


def test_the_test_split_is_locked():
    with pytest.raises(SplitLockedError, match="held out"):
        get_split("test")

    unlocked = get_split("test", allow_test=True)
    assert unlocked.start == date(2025, 7, 1)


def test_splits_are_contiguous_and_do_not_overlap():
    """A leaky boundary would put validation data in dev without anyone noticing."""
    ordered = sorted(SPLITS.values(), key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert later.start == earlier.end + timedelta(days=1)


def test_load_applies_the_date_window(tmp_path, monkeypatch):
    monkeypatch.setattr("qlab.paths.TICKS_DIR", tmp_path)
    _write_month(
        tmp_path,
        "EURUSD",
        2024,
        6,
        _ticks(
            [
                ("2024-06-01T12:00:00", 1.08, 1.08),
                ("2024-06-02T12:00:00", 1.09, 1.09),
                ("2024-06-03T12:00:00", 1.10, 1.10),
            ]
        ),
    )

    # A bare end date means "through that day", so the 2nd is included.
    window = load_ticks("EURUSD", start="2024-06-02", end="2024-06-02")
    assert window["bid"].to_list() == [1.09]


def test_months_outside_the_window_are_never_opened(tmp_path, monkeypatch):
    """Pruning on the filename is what keeps a narrow window cheap on 283M ticks."""
    monkeypatch.setattr("qlab.paths.TICKS_DIR", tmp_path)
    _write_month(
        tmp_path, "EURUSD", 2024, 6, _ticks([("2024-06-15T12:00:00", 1.08, 1.08)])
    )
    directory = tmp_path / "symbol=EURUSD"
    # A file polars could not read at all: touching it is a hard failure.
    (directory / "EURUSD_2024_07.parquet").write_bytes(b"not parquet")

    assert load_ticks("EURUSD", start="2024-06-01", end="2024-06-30").height == 1


def test_warmup_loads_history_and_flags_it(tmp_path, monkeypatch):
    """The point-in-time guarantee: warm indicators, but never evaluate the warmup."""
    monkeypatch.setattr("qlab.paths.TICKS_DIR", tmp_path)
    _write_month(
        tmp_path,
        "EURUSD",
        2023,
        12,
        _ticks([("2023-12-30T12:00:00", 1.05, 1.05), ("2023-12-31T12:00:00", 1.06, 1.06)]),
    )
    _write_month(
        tmp_path, "EURUSD", 2024, 1, _ticks([("2024-01-02T12:00:00", 1.07, 1.07)])
    )

    ticks = load_ticks("EURUSD", split="validation", warmup=timedelta(days=3))
    boundary = datetime(2024, 1, 1, tzinfo=UTC)

    assert ticks.height == 3
    assert ticks.filter(pl.col("is_warmup"))["ts"].max() < boundary
    assert ticks.filter(~pl.col("is_warmup"))["ts"].min() >= boundary


def test_split_and_explicit_dates_are_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr("qlab.paths.TICKS_DIR", tmp_path)
    _write_month(
        tmp_path, "EURUSD", 2024, 6, _ticks([("2024-06-15T12:00:00", 1.08, 1.08)])
    )
    with pytest.raises(ValueError, match="not both"):
        load_ticks("EURUSD", split="dev", start="2024-06-01")
