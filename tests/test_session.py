"""Session flagging - and specifically, which timestamp the flags key off."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from qlab.bars import time_bars
from qlab.session import ROLLOVER_HOUR_UTC, SESSIONS, session_of, with_session_flags
from qlab.symbols import get_spec

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


def test_every_hour_belongs_to_exactly_one_session():
    """Overlapping labels would double-count a session breakdown."""
    assert [session_of(hour) for hour in range(24)].count("tokyo") == 7
    covered = sum(end - start for _, start, end in SESSIONS)
    assert covered == 24


def test_bars_are_flagged_by_their_open_edge_not_their_label():
    """The trap that right-edge labelling sets.

    A bar covering [20:59, 21:00) is labelled 21:00. Keying the rollover flag off
    the label would file it under the rollover hour, when in fact not one of its
    ticks was in it - and the rollover is the hour whose costs we most need to
    attribute correctly.
    """
    bars = time_bars(
        _ticks(
            [
                ("2024-06-03T20:59:30", 1.08, 1.0802),
                ("2024-06-03T21:00:30", 1.08, 1.0830),  # the real rollover tick
            ]
        ),
        "1m",
    )
    flagged = with_session_flags(bars, get_spec("EURUSD"))

    assert flagged["ts"].to_list() == [
        datetime(2024, 6, 3, 21, 0, tzinfo=UTC),
        datetime(2024, 6, 3, 21, 1, tzinfo=UTC),
    ]
    assert flagged["is_rollover"].to_list() == [False, True]
    assert flagged["session"].to_list() == ["newyork", "sydney"]


def test_raw_ticks_fall_back_to_ts():
    ticks = _ticks([("2024-06-03T21:30:00", 1.08, 1.0830)])
    flagged = with_session_flags(ticks, get_spec("EURUSD"))

    assert flagged["is_rollover"].to_list() == [True]
    assert flagged["session"].to_list() == ["sydney"]


def test_the_sunday_reopen_is_flagged_rather_than_removed():
    """On the majors this window holds most of the sample's non-zero spread, so
    a strategy has to be able to see it in order to sit it out."""
    ticks = _ticks(
        [
            ("2024-06-02T22:05:00", 1.0800, 1.0810),  # Sunday reopen, wide
            ("2024-06-03T10:00:00", 1.0800, 1.0800),  # Monday, zero spread
        ]
    )
    flagged = with_session_flags(ticks, get_spec("EURUSD"))

    assert flagged["is_sunday"].to_list() == [True, False]
    assert flagged.height == 2  # nothing filtered
    assert flagged["ask"][0] - flagged["bid"][0] == pytest.approx(0.0010)


def test_the_break_window_follows_the_instrument():
    """Gold and USTEC halt daily; FX does not, so the flag must not be global."""
    ticks = _ticks([("2024-06-03T20:30:00", 2300.0, 2300.1)])

    assert with_session_flags(ticks, get_spec("XAUUSD"))["is_break_window"][0] is True
    assert with_session_flags(ticks, get_spec("EURUSD"))["is_break_window"][0] is False


def test_rollover_hour_matches_the_audited_spike():
    """21:00 UTC is where Stage 0 measured 1.5 and 3.3 pip means on the majors."""
    assert ROLLOVER_HOUR_UTC == 21
