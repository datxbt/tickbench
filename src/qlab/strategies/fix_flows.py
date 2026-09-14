"""Calendar rules for flow-driven windows at the FX fixings.

The Tokyo fix is published at 09:55 JST. On gotobi days - the 5th, 10th, 15th,
20th, 25th and last day of the month, the dates Japanese firms settle on -
importers buy USD at that fix, and the hypothesis tested in
``docs/findings/scheduled-flows.md`` is that the price they push unwinds
afterwards. Rejected: it held on dev and validation and failed on test.

These are pure calendar functions, known in advance, with no price input.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

GOTOBI_DAYS_OF_MONTH: tuple[int, ...] = (5, 10, 15, 20, 25)


def fx_weekdays(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]`` less the two days the FX market is shut."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and (d.month, d.day) not in ((1, 1), (12, 25)):
            out.append(d)
        d += timedelta(days=1)
    return out


def gotobi_days(days: list[date]) -> set[date]:
    """The gotobi dates among ``days``.

    A nominal date that falls on a weekend moves to the preceding Friday, the
    convention settlement follows. Japanese public holidays are deliberately
    not applied - this is the rule as registered, and a holiday calendar is a
    separate robustness check rather than part of the definition.
    """
    have = set(days)
    out: set[date] = set()
    for y, m in sorted({(d.year, d.month) for d in days}):
        last = calendar.monthrange(y, m)[1]
        for dom in (*GOTOBI_DAYS_OF_MONTH, last):
            d = date(y, m, dom)
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            if d in have:
                out.add(d)
    return out


def month_ends(days: list[date]) -> set[date]:
    """The last of ``days`` in each calendar month."""
    by: dict[tuple[int, int], date] = {}
    for d in days:
        key = (d.year, d.month)
        by[key] = max(by.get(key, d), d)
    return set(by.values())
