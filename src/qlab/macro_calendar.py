"""A reconstructed calendar of scheduled high-impact US macro releases.

Why this module exists
----------------------
:mod:`qlab.strategies.news_breakout` tests a paper whose entire trigger is "a
high-impact scheduled USD event just happened". The paper took that calendar
from the MQL5 terminal. This corpus has no calendar in it - it is four price
feeds and nothing else - so the calendar has to be reconstructed, and the
honest thing is to reconstruct it from *release rules* rather than from the
tape.

That distinction matters more than it looks. A calendar derived from the tape
("the minutes where USTEC moved a lot") would be circular: the strategy trades
post-announcement volatility, so selecting announcements by their volatility
hands the backtest the answer. Every date below is therefore generated from a
publication rule that a trader knew months in advance, or from a schedule the
Federal Reserve published a year ahead. The tape is used only *afterwards*, to
audit how often the rule landed on a day that actually looks like a release -
see :func:`validate`.

Two tiers, deliberately separated
---------------------------------
**Tier A - exact.** The date is implied by a published rule with no judgement
in it, so a reconstruction cannot be wrong except through a holiday shift:

* ``FOMC`` - the Federal Reserve publishes its meeting calendar a year ahead.
  The dates in :data:`FOMC_DATES` are those schedules, plus the two unscheduled
  emergency cuts of March 2020. Statement at 14:00 ET.
* ``NFP`` - the BLS rule is mechanical: the Employment Situation is released on
  the *third Friday following the conclusion of the reference week*, where the
  reference week is the Sunday-Saturday week containing the 12th of the month.
  That is why the jobs report is usually, but not always, the first Friday.
  08:30 ET.
* ``ISM_MFG`` / ``ISM_SVC`` - first and third business day of the month, 10:00 ET.
* ``CLAIMS`` - initial jobless claims, every Thursday, 08:30 ET.

**Tier B - approximate.** The agency publishes on "roughly the Nth", and the
exact day moves with the calendar and with the agency's internal schedule. The
rule below gets the right week and usually the right day, but not always:

* ``CPI`` - BLS, the Tue/Wed/Thu nearest the 12th, 08:30 ET.
* ``PPI`` - BLS, the business day after CPI, 08:30 ET.
* ``RETAIL`` - Census, the weekday nearest the 15th, 08:30 ET.
* ``GDP`` - BEA, last Thursday of the month, 08:30 ET.

Any result that depends on Tier B and not on Tier A is a result about this
module's guesses, not about macro releases, which is why
:func:`qlab.strategies.news_breakout` reports the two tiers separately.

What is knowingly missing
-------------------------
Revisions and reschedules. The October-November 2025 federal shutdown delayed
or cancelled several BLS releases; the rules here emit them anyway. Releases
that fall on a US market holiday are dropped, since the exchange is shut, but a
release *shifted* by a holiday is emitted on its nominal day. Both show up in
:func:`validate` as rule-days with no volatility signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

ET = "America/New_York"

# Impact tier. "exact" dates come from a published rule or a published
# schedule; "approx" dates come from a rule of thumb about the release window.
EXACT = "exact"
APPROX = "approx"


# --- The Federal Reserve's own published meeting schedules -------------------
# Statement day only (the second day of each two-day meeting). The two March
# 2020 entries are the unscheduled inter-meeting cuts, which were announced
# outside the published calendar and moved the index more than any scheduled
# meeting in the corpus.
FOMC_DATES: tuple[date, ...] = tuple(
    date(y, m, d)
    for y, m, d in [
        (2020, 1, 29), (2020, 3, 3), (2020, 3, 15), (2020, 3, 18),
        (2020, 4, 29), (2020, 6, 10), (2020, 7, 29), (2020, 9, 16),
        (2020, 11, 5), (2020, 12, 16),
        (2021, 1, 27), (2021, 3, 17), (2021, 4, 28), (2021, 6, 16),
        (2021, 7, 28), (2021, 9, 22), (2021, 11, 3), (2021, 12, 15),
        (2022, 1, 26), (2022, 3, 16), (2022, 5, 4), (2022, 6, 15),
        (2022, 7, 27), (2022, 9, 21), (2022, 11, 2), (2022, 12, 14),
        (2023, 2, 1), (2023, 3, 22), (2023, 5, 3), (2023, 6, 14),
        (2023, 7, 26), (2023, 9, 20), (2023, 11, 1), (2023, 12, 13),
        (2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12),
        (2024, 7, 31), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
        (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
        (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
        (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
        (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 9),
    ]
)

# US market holidays 2020-2026, on which the cash index does not trade. Used
# only to drop releases that could not have been traded.
_HOLIDAYS: frozenset[date] = frozenset(
    date(y, m, d)
    for y, m, d in [
        (2020, 1, 1), (2020, 1, 20), (2020, 2, 17), (2020, 4, 10),
        (2020, 5, 25), (2020, 7, 3), (2020, 9, 7), (2020, 11, 26), (2020, 12, 25),
        (2021, 1, 1), (2021, 1, 18), (2021, 2, 15), (2021, 4, 2),
        (2021, 5, 31), (2021, 7, 5), (2021, 9, 6), (2021, 11, 25), (2021, 12, 24),
        (2022, 1, 17), (2022, 2, 21), (2022, 4, 15), (2022, 5, 30),
        (2022, 6, 20), (2022, 7, 4), (2022, 9, 5), (2022, 11, 24), (2022, 12, 26),
        (2023, 1, 2), (2023, 1, 16), (2023, 2, 20), (2023, 4, 7),
        (2023, 5, 29), (2023, 6, 19), (2023, 7, 4), (2023, 9, 4),
        (2023, 11, 23), (2023, 12, 25),
        (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 3, 29),
        (2024, 5, 27), (2024, 6, 19), (2024, 7, 4), (2024, 9, 2),
        (2024, 11, 28), (2024, 12, 25),
        (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17),
        (2025, 4, 18), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4),
        (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
        (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
        (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    ]
)


# The years the holiday table above covers, and therefore the only years for
# which :func:`build_events` will emit anything.
COVERED: tuple[date, date] = (date(2020, 1, 1), date(2026, 12, 31))


@dataclass(frozen=True)
class Event:
    """One scheduled release, at the minute it hits the wire."""

    day: date
    hour: int  # ET
    minute: int  # ET
    name: str
    category: str
    tier: str

    @property
    def ts_utc(self) -> datetime:
        """The release instant, in UTC, via the ET wall clock it is scheduled on."""
        import zoneinfo

        local = datetime(
            self.day.year, self.day.month, self.day.day, self.hour, self.minute,
            tzinfo=zoneinfo.ZoneInfo(ET),
        )
        return local.astimezone(zoneinfo.ZoneInfo("UTC"))


# --- date rules --------------------------------------------------------------


def _is_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _HOLIDAYS


def _nth_business_day(year: int, month: int, n: int) -> date | None:
    """The ``n``-th business day of a month, or None if the month is too short."""
    day, seen = date(year, month, 1), 0
    while day.month == month:
        if _is_business_day(day):
            seen += 1
            if seen == n:
                return day
        day += timedelta(days=1)
    return None


def _nearest_weekday(target: date) -> date:
    """``target`` if it is a business day, else the next one."""
    day = target
    for _ in range(7):
        if _is_business_day(day):
            return day
        day += timedelta(days=1)
    return target


def _nearest_midweek(target: date) -> date:
    """The Tuesday, Wednesday or Thursday closest to ``target``.

    BLS puts CPI in the middle of the week: across 2020-2026 the release never
    lands on a Monday or a Friday. Restricting to midweek is a fact about the
    publication schedule, not a fit to the tape, so it is allowed to inform the
    rule. It still only gets the day right about half the time - see the
    module docstring's Tier B caveat, and :func:`validate`.
    """
    best, best_gap = target, 99
    for offset in range(-4, 5):
        day = target + timedelta(days=offset)
        if day.weekday() in (1, 2, 3) and _is_business_day(day):
            if abs(offset) < best_gap:
                best, best_gap = day, abs(offset)
    return best


def _next_business_day(day: date) -> date:
    nxt = day + timedelta(days=1)
    for _ in range(7):
        if _is_business_day(nxt):
            return nxt
        nxt += timedelta(days=1)
    return nxt


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` (Mon=0) in a month."""
    if month == 12:
        day = date(year, 12, 31)
    else:
        day = date(year, month + 1, 1) - timedelta(days=1)
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day


def nfp_release_day(year: int, month: int) -> date:
    """The BLS Employment Situation release date for the ``month`` reference period.

    The rule, verbatim from the BLS schedule: the reference week is the
    Sunday-through-Saturday week containing the 12th, and the release is the
    third Friday after that week ends. It resolves to the first Friday of the
    following month most of the time and the second Friday otherwise, which is
    exactly the pattern the published schedule shows.
    """
    twelfth = date(year, month, 12)
    # Sunday-start week containing the 12th; Python's weekday() has Monday=0.
    sunday = twelfth - timedelta(days=(twelfth.weekday() + 1) % 7)
    saturday = sunday + timedelta(days=6)
    friday = saturday + timedelta(days=(4 - saturday.weekday()) % 7 or 7)
    return friday + timedelta(days=14)


def _month_range(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def build_events(start: date, end: date, *, tiers: str = "all") -> list[Event]:
    """Every scheduled release the rules place in ``[start, end]``.

    ``tiers`` is ``"exact"``, ``"approx"`` or ``"all"``. Releases landing on a
    US market holiday are dropped.
    """
    if tiers not in {"exact", "approx", "all"}:
        raise ValueError(f"tiers must be exact | approx | all, got {tiers!r}")

    # The holiday table below spans 2020-2026 only. Outside it a generated date
    # cannot be checked against a closure, so the rules emit nothing rather than
    # emitting releases on days the exchange may have been shut.
    start = max(start, COVERED[0])
    end = min(end, COVERED[1])
    if start > end:
        return []

    out: list[Event] = []

    def add(day: date, hour: int, minute: int, name: str, cat: str, tier: str) -> None:
        if not (start <= day <= end) or not _is_business_day(day):
            return
        if tiers != "all" and tier != tiers:
            return
        out.append(Event(day, hour, minute, name, cat, tier))

    for meeting in FOMC_DATES:
        add(meeting, 14, 0, "FOMC Statement", "rates", EXACT)

    for year, month in _month_range(
        date(start.year, start.month, 1) - timedelta(days=40), end
    ):
        add(nfp_release_day(year, month), 8, 30, "Nonfarm Payrolls", "employment", EXACT)

        first = _nth_business_day(year, month, 1)
        third = _nth_business_day(year, month, 3)
        if first:
            add(first, 10, 0, "ISM Manufacturing PMI", "activity", EXACT)
        if third:
            add(third, 10, 0, "ISM Services PMI", "activity", EXACT)

        cpi = _nearest_midweek(date(year, month, 12))
        add(cpi, 8, 30, "CPI", "inflation", APPROX)
        add(_next_business_day(cpi), 8, 30, "PPI", "inflation", APPROX)
        add(_nearest_weekday(date(year, month, 15)), 8, 30, "Retail Sales", "activity", APPROX)
        add(_last_weekday_of_month(year, month, 3), 8, 30, "GDP", "growth", APPROX)

    # Weekly initial claims: every Thursday.
    day = start - timedelta(days=7)
    while day <= end:
        if day.weekday() == 3:
            add(day, 8, 30, "Initial Jobless Claims", "employment", EXACT)
        day += timedelta(days=1)

    return sorted(set(out), key=lambda e: (e.day, e.hour, e.minute, e.name))


def build_calendar(start: date, end: date, *, tiers: str = "all") -> pl.DataFrame:
    """:func:`build_events` as a frame, one row per release, sorted by time.

    ``ts`` is the release instant in UTC. Simultaneous releases stay as separate
    rows; the strategy collapses them itself, because its "one pending order"
    rule is what decides how a cluster is handled.
    """
    events = build_events(start, end, tiers=tiers)
    if not events:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime("us", "UTC"), "day": pl.Date, "et_hour": pl.Int8,
                "et_minute": pl.Int8, "event": pl.Utf8, "category": pl.Utf8,
                "tier": pl.Utf8,
            }
        )
    return pl.DataFrame(
        {
            "ts": [e.ts_utc for e in events],
            "day": [e.day for e in events],
            "et_hour": [e.hour for e in events],
            "et_minute": [e.minute for e in events],
            "event": [e.name for e in events],
            "category": [e.category for e in events],
            "tier": [e.tier for e in events],
        }
    ).with_columns(
        pl.col("ts").dt.cast_time_unit("us"),
        pl.col("et_hour").cast(pl.Int8),
        pl.col("et_minute").cast(pl.Int8),
    )


# --- auditing the reconstruction against the tape ----------------------------


def validate(calendar: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """How much a rule-day's release minute moves, against its own control.

    For each event name, the release-minute range on rule-days is compared with
    the range in the *same ET minute* on every other day in the sample. A real
    scheduled release shows a ratio well above 1. A ratio near 1 means the rule
    is landing on the wrong day often enough that the event is mostly noise -
    which is a fact about this module, and belongs in the report next to any
    result that leans on it.

    ``bars`` must be 1-minute bars carrying ``ts_open``, ``high`` and ``low``.
    Bars in this corpus are close-labelled, so the minute a release lands in is
    the bar whose ``ts_open`` is the release minute, not the one whose ``ts`` is.
    """
    marks = (
        bars.lazy()
        .with_columns(et=pl.col("ts_open").dt.convert_time_zone(ET))
        .with_columns(
            rng=pl.col("high") - pl.col("low"),
            hm=pl.col("et").dt.hour().cast(pl.Int32) * 60
            + pl.col("et").dt.minute().cast(pl.Int32),
            day=pl.col("et").dt.date(),
        )
        .select("day", "hm", "rng")
        .collect()
    )
    cal = calendar.select(
        "day", "event", "tier", hm=pl.col("et_hour").cast(pl.Int32) * 60
        + pl.col("et_minute").cast(pl.Int32)
    ).unique()

    hit = marks.join(cal, on=["day", "hm"], how="inner")
    if hit.is_empty():
        return pl.DataFrame()

    # Control: the same clock minute, on days the rule did not name.
    control = (
        marks.join(cal.select("day", "hm").unique(), on=["day", "hm"], how="anti")
        .group_by("hm")
        .agg(control_rng=pl.col("rng").median())
    )
    return (
        hit.join(control, on="hm", how="left")
        .group_by("event", "tier")
        .agg(
            n=pl.len(),
            release_rng=pl.col("rng").median(),
            control_rng=pl.col("control_rng").median(),
        )
        .with_columns(ratio=pl.col("release_rng") / pl.col("control_rng"))
        .sort("ratio", descending=True)
    )
