"""Session, rollover and reopen flags.

Stage 0's sharpest finding was that FX spread is not spread out - it is
concentrated. Weekdays run 2-5% non-zero spread on the majors, but the Sunday
reopen hits 37% (EURUSD) and 53% (USDJPY), and the 21:00 UTC rollover averages
1.5 and 3.3 pips against a weekday mean near zero. Avoiding those two windows
matters more than everything else in the FX cost model put together.

The response is to **flag, not filter**. Those are real quotes at which real
orders fill; a strategy that cannot see them cannot decide to avoid them, and a
dataset with them deleted quietly promises a fill quality that does not exist.
So this module only adds boolean columns, and what to do about them stays a
strategy's decision.

Session boundaries are fixed UTC hours, which is an approximation: London and
New York shift by an hour relative to UTC twice a year, so around the DST
changeovers a bar can carry the neighbouring session's label. That is accurate
enough for regime breakdowns and cost attribution, and not accurate enough to
build a strategy that trades the first minute of a session - if you need that,
convert to the exchange's local time explicitly rather than trusting these.
"""

from __future__ import annotations

import polars as pl

from .symbols import SymbolSpec

# Fixed UTC hour ranges, non-overlapping so that every bar gets exactly one
# label. The "overlap" band is the London/New York overlap, which carries most
# of the day's volume.
SESSIONS: tuple[tuple[str, int, int], ...] = (
    ("tokyo", 0, 7),
    ("london", 7, 12),
    ("overlap", 12, 16),
    ("newyork", 16, 21),
    ("sydney", 21, 24),
)

# The daily contract rollover. Spread widens here on every instrument, and on
# the majors it is the single worst hour of the weekday.
ROLLOVER_HOUR_UTC = 21


def session_of(hour: int) -> str:
    for name, start, end in SESSIONS:
        if start <= hour < end:
            return name
    raise ValueError(f"hour out of range: {hour}")


def _time_column(frame: pl.DataFrame | pl.LazyFrame, on: str | None) -> str:
    """Pick the column that says *when the bar's interval began*.

    For bars this is ``ts_open``, not ``ts``: a one-minute bar covering
    [20:59, 21:00) is labelled 21:00 under the right-edge convention, so keying
    the session off the label would file it under the rollover hour when none of
    its ticks were in it. For raw ticks there is only ``ts``.
    """
    if on is not None:
        return on
    columns = frame.collect_schema().names()
    return "ts_open" if "ts_open" in columns else "ts"


def with_session_flags(
    frame: pl.DataFrame | pl.LazyFrame,
    spec: SymbolSpec,
    *,
    on: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Add ``session``, ``is_rollover``, ``is_sunday`` and ``is_break_window``.

    ``is_sunday``
        The weekly reopen. Not "the weekend" - these are tradable quotes, they
        are simply the widest of the week, and on the majors they account for
        most of the non-zero spread in the entire sample.

    ``is_break_window``
        Inside the instrument's daily maintenance window, for the two
        instruments that have one. Mostly this window is *empty* of bars rather
        than full of bad ones, so the flag is a label for the boundary bars on
        either side; the absence of rows is the stronger signal.
    """
    column = _time_column(frame, on)
    hour = pl.col(column).dt.hour()

    session = pl.lit(None, dtype=pl.String)
    for name, start, end in SESSIONS:
        session = pl.when((hour >= start) & (hour < end)).then(pl.lit(name)).otherwise(session)

    if spec.daily_break_utc is None:
        in_break = pl.lit(False)
    else:
        start, end = spec.daily_break_utc
        in_break = (hour >= start) & (hour < end)

    return frame.with_columns(
        session=session,
        is_rollover=hour == ROLLOVER_HOUR_UTC,
        is_sunday=pl.col(column).dt.weekday() == 7,
        is_break_window=in_break,
    )
