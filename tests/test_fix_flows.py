"""Tests for the gotobi calendar rule.

The rule is pure calendar arithmetic, but three details decide which days get
traded and each is easy to get wrong:

* A nominal date on a weekend moves to the *preceding* Friday, not the next
  Monday - settlement is brought forward, never deferred.
* The last day of the month is a gotobi day whatever its number, so February
  and 30-day months have to come out right.
* Only days the caller passed in can come back. The rule never invents a
  trading day, and a closure is not re-shifted - that is a separate choice.
"""

from __future__ import annotations

from datetime import date

from qlab.strategies.fix_flows import fx_weekdays, gotobi_days, month_ends


def test_weekend_dates_move_to_the_preceding_friday():
    # June 2024 starts on a Saturday: the 15th is a Saturday and the 30th a
    # Sunday, so both move back to Friday; the others are weekdays already.
    days = fx_weekdays(date(2024, 6, 1), date(2024, 6, 30))
    assert gotobi_days(days) == {
        date(2024, 6, 5), date(2024, 6, 10), date(2024, 6, 14),
        date(2024, 6, 20), date(2024, 6, 25), date(2024, 6, 28),
    }


def test_month_end_follows_the_calendar_not_a_fixed_number():
    # Leap February: the 29th is a Thursday and is the month-end gotobi day;
    # the 10th (Saturday) and 25th (Sunday) fall back to Fridays.
    days = fx_weekdays(date(2024, 2, 1), date(2024, 2, 29))
    got = gotobi_days(days)
    assert date(2024, 2, 29) in got
    assert {date(2024, 2, 9), date(2024, 2, 23)} <= got
    assert date(2024, 2, 10) not in got and date(2024, 2, 25) not in got


def test_never_returns_a_day_it_was_not_given():
    days = fx_weekdays(date(2024, 12, 1), date(2024, 12, 31))
    got = gotobi_days(days)
    assert got <= set(days)
    # Christmas is a gotobi date on a Wednesday but the market is shut; the
    # rule drops it rather than silently moving it to the 24th.
    assert date(2024, 12, 25) not in got
    assert date(2024, 12, 24) not in got


def test_fx_weekdays_excludes_weekends_and_the_two_closures():
    days = fx_weekdays(date(2024, 12, 23), date(2025, 1, 3))
    assert date(2024, 12, 25) not in days
    assert date(2025, 1, 1) not in days
    assert all(d.weekday() < 5 for d in days)
    assert len(days) == 8


def test_month_ends_are_the_last_trading_day():
    days = fx_weekdays(date(2024, 1, 1), date(2024, 3, 31))
    # 31 March 2024 is a Sunday, so the month's last weekday is Friday the 29th.
    assert month_ends(days) == {date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 29)}
