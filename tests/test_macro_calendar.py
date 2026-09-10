"""The calendar has to be right about dates a trader could have known in advance."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from qlab.macro_calendar import (
    FOMC_DATES,
    build_calendar,
    build_events,
    nfp_release_day,
    validate,
)


# Release dates verified independently against the published BLS schedule.
# These are the check on the rule, not on the implementation of the rule.
KNOWN_NFP = {
    (2022, 5): date(2022, 6, 3),
    (2023, 8): date(2023, 9, 1),
    (2024, 12): date(2025, 1, 3),
    (2025, 1): date(2025, 2, 7),
    (2025, 9): date(2025, 10, 3),
}


@pytest.mark.parametrize("period,expected", sorted(KNOWN_NFP.items()))
def test_nfp_rule_matches_the_published_schedule(period, expected):
    assert nfp_release_day(*period) == expected


def test_nfp_lands_on_a_friday_every_month_for_seven_years():
    for year in range(2020, 2027):
        for month in range(1, 13):
            assert nfp_release_day(year, month).weekday() == 4


def test_nfp_is_the_first_or_second_friday_of_the_following_month():
    """The BLS rule is not "first Friday" - but it is always one of the first two."""
    for year in range(2020, 2027):
        for month in range(1, 13):
            day = nfp_release_day(year, month)
            assert day.day <= 14, (year, month, day)


def test_fomc_dates_are_business_days_and_eight_a_year_after_2020():
    from collections import Counter

    per_year = Counter(d.year for d in FOMC_DATES)
    for year in range(2021, 2027):
        assert per_year[year] == 8, (year, per_year[year])
    # One exception, and it is real: the emergency cut of 2020-03-15 was
    # announced on a Sunday evening. build_events drops it, because the cash
    # index was shut and the CFD could not be traded into it.
    weekend = [d for d in FOMC_DATES if d.weekday() >= 5]
    assert weekend == [date(2020, 3, 15)]
    assert date(2020, 3, 15) not in {
        e.day for e in build_events(date(2020, 3, 1), date(2020, 3, 31))
    }


def test_events_are_unique_sorted_and_inside_the_window():
    events = build_events(date(2023, 1, 1), date(2023, 12, 31))
    assert events == sorted(events, key=lambda e: (e.day, e.hour, e.minute, e.name))
    assert len(events) == len(set(events))
    assert all(date(2023, 1, 1) <= e.day <= date(2023, 12, 31) for e in events)


def test_no_event_lands_on_a_weekend_or_a_market_holiday():
    events = build_events(date(2020, 1, 1), date(2026, 9, 1))
    assert all(e.day.weekday() < 5 for e in events)
    assert date(2023, 7, 4) not in {e.day for e in events}
    assert date(2024, 12, 25) not in {e.day for e in events}


def test_tier_filter_partitions_the_calendar():
    window = (date(2022, 1, 1), date(2022, 12, 31))
    everything = build_calendar(*window, tiers="all")
    exact = build_calendar(*window, tiers="exact")
    approx = build_calendar(*window, tiers="approx")
    assert exact.height + approx.height == everything.height
    assert set(exact["tier"].unique()) == {"exact"}
    assert set(approx["tier"].unique()) == {"approx"}


def test_release_instants_respect_daylight_saving():
    """08:30 ET is 13:30 UTC in summer and 12:30 UTC in winter, not one or the other."""
    calendar = build_calendar(date(2024, 1, 1), date(2024, 12, 31), tiers="exact")
    nfp = calendar.filter(pl.col("event") == "Nonfarm Payrolls").with_columns(
        utc_hour=pl.col("ts").dt.hour()
    )
    hours = set(nfp["utc_hour"].to_list())
    assert hours == {12, 13}, hours


def test_calendar_is_empty_outside_the_covered_years():
    """The holiday table spans 2020-2026; outside it the rules must stay quiet.

    A date rule is timeless, but a *tradable* date rule is not: without a
    holiday table the generator would happily place a release on a day the
    exchange was closed.
    """
    assert build_calendar(date(1990, 1, 1), date(1990, 12, 31)).height == 0
    assert build_calendar(date(2027, 1, 1), date(2030, 12, 31)).height == 0
    assert build_calendar(date(2019, 1, 1), date(2020, 3, 31)).height > 0


def test_validate_scores_a_real_release_above_its_own_control():
    """The audit has to be able to tell a release minute from an ordinary one.

    Synthetic tape: one wide bar at 08:30 ET on every NFP day, narrow bars
    everywhere else. A working audit reports a ratio well above 1 for NFP.
    """
    calendar = build_calendar(date(2023, 1, 1), date(2023, 12, 31), tiers="exact")
    nfp_days = set(
        calendar.filter(pl.col("event") == "Nonfarm Payrolls")["day"].to_list()
    )
    rows = []
    for day in sorted({d for d in calendar["day"].to_list()}):
        for minute in (8 * 60 + 30, 9 * 60 + 30):
            width = 100.0 if (day in nfp_days and minute == 8 * 60 + 30) else 1.0
            rows.append(
                {
                    "ts_open": f"{day} {minute // 60:02d}:{minute % 60:02d}:00",
                    "high": 100.0 + width,
                    "low": 100.0,
                }
            )
    bars = pl.DataFrame(rows).with_columns(
        pl.col("ts_open")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )
    report = validate(calendar, bars)
    ratio = report.filter(pl.col("event") == "Nonfarm Payrolls")["ratio"][0]
    assert ratio > 10.0
