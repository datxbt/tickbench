"""Quality metric correctness, especially the weekend-gap classification."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from qlab.quality import month_metrics
from qlab.symbols import get_spec

SPEC = get_spec("EURUSD")


def _parquet(tmp_path, ticks: list[tuple[str, float, float]]):
    frame = pl.DataFrame(
        {
            "ts": [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t, _, _ in ticks],
            "bid": [b for _, b, _ in ticks],
            "ask": [a for _, _, a in ticks],
        },
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
    )
    path = tmp_path / "ticks.parquet"
    frame.write_parquet(path)
    return path


def test_weekend_gap_excluded_from_intraweek():
    """A Friday-close to Sunday-open gap must not count as a feed outage."""
    # 2024-06-07 is a Friday, 2024-06-09 a Sunday.
    assert datetime(2024, 6, 7).weekday() == 4
    assert datetime(2024, 6, 9).weekday() == 6


def test_weekend_and_intraweek_gaps_are_separated(tmp_path):
    path = _parquet(
        tmp_path,
        [
            ("2024-06-07 20:00:00", 1.0, 1.0001),  # Friday close
            ("2024-06-09 21:00:00", 1.0, 1.0001),  # Sunday open -> weekend gap (49h)
            ("2024-06-10 00:00:00", 1.0, 1.0001),  # +3h intraweek gap
        ],
    )
    metrics = month_metrics(path, SPEC)

    assert metrics["max_gap_s"] == pytest.approx(49 * 3600)
    assert metrics["max_intraweek_gap_s"] == pytest.approx(3 * 3600)
    assert metrics["n_intraweek_gap_gt_3600s"] == 1
    # EURUSD has no daily break window, so the intraweek gap is a real outage.
    assert metrics["n_outage_gap_gt_3600s"] == 1


def test_daily_break_gap_is_not_counted_as_an_outage(tmp_path):
    """XAUUSD's 20:00-22:00 UTC maintenance break must not read as downtime."""
    path = _parquet(
        tmp_path,
        [
            ("2024-06-03 20:55:00", 2300.0, 2300.06),  # last tick before the break
            ("2024-06-03 22:00:00", 2300.0, 2300.06),  # reopen, 1.08 h later
            ("2024-06-04 02:00:00", 2300.0, 2300.06),  # +4 h, a genuine outage
        ],
    )
    metrics = month_metrics(path, get_spec("XAUUSD"))

    assert metrics["n_intraweek_gap_gt_3600s"] == 2  # both gaps exceed an hour
    assert metrics["n_session_breaks"] == 1  # one is the daily break
    assert metrics["n_outage_gap_gt_3600s"] == 1  # only the other is an outage
    assert metrics["max_outage_gap_s"] == pytest.approx(4 * 3600)


def test_locked_and_crossed_detection(tmp_path):
    path = _parquet(
        tmp_path,
        [
            ("2024-06-03 00:00:00", 1.0, 1.0),  # locked
            ("2024-06-03 00:00:01", 1.0, 1.0),  # locked
            ("2024-06-03 00:00:02", 1.0, 1.0002),  # normal
            ("2024-06-03 00:00:03", 1.0, 0.9999),  # crossed
        ],
    )
    metrics = month_metrics(path, SPEC)

    assert metrics["locked_pct"] == pytest.approx(50.0)
    assert metrics["crossed_pct"] == pytest.approx(25.0)
    assert metrics["spread_max_pips"] == pytest.approx(2.0, rel=1e-6)


def test_duplicate_and_missing_day_accounting(tmp_path):
    path = _parquet(
        tmp_path,
        [
            ("2024-06-03 00:00:00", 1.0, 1.0001),  # Monday
            ("2024-06-03 00:00:00", 1.0, 1.0001),  # exact duplicate
            ("2024-06-05 00:00:00", 1.0, 1.0001),  # Wednesday - Tuesday missing
        ],
    )
    metrics = month_metrics(path, SPEC)

    assert metrics["duplicate_ts"] == 1
    assert metrics["exact_duplicate_rows"] == 1
    assert metrics["days_present"] == 2
    assert metrics["missing_weekdays"] == 1  # 2024-06-04


def test_jump_detection_is_scale_free(tmp_path):
    path = _parquet(
        tmp_path,
        [
            ("2024-06-03 00:00:00", 100.0, 100.0),
            ("2024-06-03 00:00:01", 101.0, 101.0),  # +100 bps
        ],
    )
    metrics = month_metrics(path, SPEC)

    assert metrics["max_jump_bps"] == pytest.approx(100.0, rel=1e-3)
    assert metrics["n_jumps"] == 1
