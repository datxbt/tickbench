"""Stage 1 data layer: the one supported way to get data out of storage.

Stage 0 deliberately converted the vendor CSVs *losslessly* - duplicates, zero
spreads and every other oddity survive into parquet untouched, because removing
them is a modelling decision and modelling decisions belong somewhere explicit.
This module is that somewhere.

Three responsibilities, in order of how easy they are to get wrong:

**Cleaning.** :class:`CleaningPolicy` states, as data rather than as buried
code, exactly what is dropped. Nothing is dropped silently:
:func:`cleaning_report` gives the count for every rule before you commit to it.
The default is close to nothing - Stage 0's audit reported duplicate timestamps
as exact duplicate rows, and building this layer showed that most of them are
not, so the default keeps them. See :class:`CleaningPolicy`.

**Split discipline.** :data:`SPLITS` fixes the dev / validation / test
boundaries *now*, before any hypothesis exists, because a split chosen after
seeing results is not a split. The test period is locked: reading it needs
``allow_test=True``, which is deliberately awkward to type by accident.

**Point-in-time discipline.** ``warmup`` loads history *before* a split so that
indicators are warm at the split's first bar, and flags those rows
``is_warmup`` so they can never be mistaken for evaluable ones. Without the
warmup the first N bars of every split are silently wrong; with the warmup and
no flag, the leakage just moves somewhere harder to see.

Reading is lazy where it matters. A whole symbol is 86-283 million ticks, so
``load_ticks(..., lazy=True)`` returns a :class:`polars.LazyFrame`, and the file
list is pruned by year and month before any of it is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import polars as pl

from . import paths
from .symbols import get_spec

TickFrame = pl.DataFrame | pl.LazyFrame
DateLike = date | datetime | str


# --------------------------------------------------------------------------
# Cleaning policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningPolicy:
    """What gets removed on the way from raw ticks to a usable frame.

    ``drop_repeated_rows``
        A tick byte-identical to the one before it carries no information, so
        dropping it is lossless. This is the only rule here that is free.

    ``duplicate_ts``
        Whether to collapse a timestamp that carries *several different* quotes.
        The default is ``"keep"``, and the reason is the feed's resolution: every
        timestamp in the corpus is a whole millisecond, none finer, so when the
        market moves fast enough for two quote updates to land inside one
        millisecond the feed stamps them identically. They are sequential real
        quotes, not repeats. Collapsing them therefore *destroys* prices, and it
        does so hardest exactly where prices matter most - 9.0% of USTEC's ticks
        in 2025-10 and 9.7% of gold's in the March 2020 crash, against 0.0% in
        the quiet months. That is a volatility-correlated deletion, which is the
        worst kind: it would quietly shave the highs and lows off the fastest
        bars in the sample.

        Use ``"last"`` when a consumer genuinely needs a unique timestamp index -
        an as-of join, say. It takes the final quote in the millisecond, which is
        the state of the book leaving it.

    ``drop_crossed``
        A quote with ``ask < bid`` is corrupt. The corpus contains none; this is
        a guard against a future feed, not a fix for the current one.

    ``drop_nonpositive``
        The same kind of guard. Zero or negative prices are unusable.

    Note what is *not* here. Wide spreads, the Sunday reopen and the 21:00
    rollover are not filtered, because they are real prices at which real orders
    fill. A strategy has to be able to see them in order to avoid them - see
    :mod:`qlab.session` for flagging them instead.
    """

    drop_repeated_rows: bool = True
    duplicate_ts: str = "keep"  # "keep" | "last" | "first"
    drop_crossed: bool = True
    drop_nonpositive: bool = True
    verify_sorted: bool = True

    def __post_init__(self) -> None:
        if self.duplicate_ts not in ("last", "first", "keep"):
            raise ValueError(
                f"duplicate_ts must be last/first/keep, got {self.duplicate_ts!r}"
            )


DEFAULT_POLICY = CleaningPolicy()
UNIQUE_TS_POLICY = CleaningPolicy(duplicate_ts="last")
RAW_POLICY = CleaningPolicy(
    drop_repeated_rows=False,
    duplicate_ts="keep",
    drop_crossed=False,
    drop_nonpositive=False,
    verify_sorted=False,
)

# A tick identical to its predecessor. Deliberately *adjacent* rather than
# globally distinct: a quote that flickers away and back within one millisecond
# is a sequence, and a global uniqueness test would silently drop its return leg.
_REPEATED_ROW = (
    (pl.col("ts") == pl.col("ts").shift(1))
    & (pl.col("bid") == pl.col("bid").shift(1))
    & (pl.col("ask") == pl.col("ask").shift(1))
).fill_null(False)


def clean_ticks(frame: TickFrame, policy: CleaningPolicy = DEFAULT_POLICY) -> TickFrame:
    """Apply a cleaning policy. Lazy in, lazy out."""
    lazy = frame.lazy()

    if policy.drop_nonpositive:
        lazy = lazy.filter((pl.col("bid") > 0) & (pl.col("ask") > 0))
    if policy.drop_crossed:
        lazy = lazy.filter(pl.col("ask") >= pl.col("bid"))
    if policy.drop_repeated_rows:
        lazy = lazy.filter(~_REPEATED_ROW)
    if policy.duplicate_ts != "keep":
        # maintain_order matters: without it, "last" is not deterministic.
        lazy = lazy.unique(subset=["ts"], keep=policy.duplicate_ts, maintain_order=True)

    return lazy if isinstance(frame, pl.LazyFrame) else lazy.collect()


def cleaning_report(
    frame: TickFrame, policy: CleaningPolicy = DEFAULT_POLICY
) -> dict:
    """Count what a policy *would* remove, and why, without removing it.

    The distinction that decides the policy is ``repeated_rows`` - identical to
    the previous tick, so free to drop - against ``distinct_quotes_at_shared_ts``,
    where a millisecond holds two genuinely different quotes. Only the second
    costs anything, which is why the default keeps them.
    """
    lazy = frame.lazy()
    stats = (
        lazy.select(
            rows=pl.len(),
            unique_ts=pl.col("ts").n_unique(),
            repeated=_REPEATED_ROW.sum(),
            crossed=(pl.col("ask") < pl.col("bid")).sum(),
            nonpositive=((pl.col("bid") <= 0) | (pl.col("ask") <= 0)).sum(),
            sorted=pl.col("ts").is_sorted(),
        )
        .collect()
        .row(0, named=True)
    )

    duplicate_ts_rows = stats["rows"] - stats["unique_ts"]
    removed = stats["crossed"] + stats["nonpositive"]
    if policy.drop_repeated_rows:
        removed += stats["repeated"]
    if policy.duplicate_ts != "keep":
        removed += duplicate_ts_rows - stats["repeated"]

    return {
        "rows": stats["rows"],
        "duplicate_ts_rows": duplicate_ts_rows,
        "repeated_rows": stats["repeated"],
        # Same millisecond, different quote: real prices that "last" would delete.
        "distinct_quotes_at_shared_ts": duplicate_ts_rows - stats["repeated"],
        "crossed": stats["crossed"],
        "nonpositive": stats["nonpositive"],
        "sorted": bool(stats["sorted"]),
        "rows_after_policy": stats["rows"] - removed,
    }


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A named, inclusive date range with a stated purpose."""

    name: str
    start: date
    end: date
    purpose: str
    locked: bool = False

    def __contains__(self, when: date) -> bool:
        return self.start <= when <= self.end


# Fixed 2026-09-02, before any strategy work, against a corpus running
# 2020-01-29 to 2026-09-01. Roughly 60 / 22 / 18 by calendar time. These
# boundaries are not to be moved to accommodate a result.
SPLITS: dict[str, Split] = {
    "dev": Split(
        name="dev",
        start=date(2020, 1, 29),
        end=date(2023, 12, 31),
        purpose="hypothesis generation, feature prototyping, parameter search",
    ),
    "validation": Split(
        name="validation",
        start=date(2024, 1, 1),
        end=date(2025, 6, 30),
        purpose="model selection between candidates that survived dev",
    ),
    "test": Split(
        name="test",
        start=date(2025, 7, 1),
        end=date(2026, 9, 1),
        purpose="one final honest estimate, run once, at the end",
        locked=True,
    ),
}


class SplitLockedError(RuntimeError):
    """Raised when the held-out test period is requested without an unlock."""


def get_split(name: str, *, allow_test: bool = False) -> Split:
    try:
        split = SPLITS[name]
    except KeyError:
        raise KeyError(
            f"unknown split {name!r}; known splits: {', '.join(SPLITS)}"
        ) from None
    if split.locked and not allow_test:
        raise SplitLockedError(
            f"the {split.name!r} split is held out: {split.purpose}. "
            "Pass allow_test=True only when you are running the final "
            "evaluation and will report whatever it gives you."
        )
    return split


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _to_utc(value: DateLike) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _month_files(
    directory: Path, lo: datetime | None, hi: datetime | None
) -> list[Path]:
    """Monthly files overlapping [lo, hi), pruned on the filename alone."""
    selected: list[Path] = []
    for path in sorted(directory.glob("*.parquet")):
        year, month = int(path.stem[-7:-3]), int(path.stem[-2:])
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        month_end = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
        if (lo is None or month_end > lo) and (hi is None or month_start < hi):
            selected.append(path)
    return selected


def _resolve_window(
    start: DateLike | None,
    end: DateLike | None,
    split: str | None,
    warmup: timedelta | None,
    allow_test: bool,
) -> tuple[datetime | None, datetime | None, datetime | None]:
    """Return (read_from, eval_from, read_to) as half-open UTC bounds.

    ``eval_from`` is the split boundary; ``read_from`` may precede it by the
    warmup period. Rows in between are loaded but flagged, never evaluated.
    """
    if split is not None:
        if start is not None or end is not None:
            raise ValueError("pass either split= or start=/end=, not both")
        resolved = get_split(split, allow_test=allow_test)
        eval_from = _to_utc(resolved.start)
        read_to = _to_utc(resolved.end) + timedelta(days=1)
    else:
        eval_from = _to_utc(start) if start is not None else None
        read_to = None
        if end is not None:
            read_to = _to_utc(end)
            # A bare date as `end` reads as "through that day", so make the
            # bound exclusive; an explicit datetime is taken at face value.
            if not isinstance(end, datetime) and not (
                isinstance(end, str) and "T" in end
            ):
                read_to += timedelta(days=1)

    read_from = eval_from
    if warmup is not None:
        if eval_from is None:
            raise ValueError("warmup= needs a split or a start date to warm up to")
        read_from = eval_from - warmup
    return read_from, eval_from, read_to


def _read(
    directory: Path,
    *,
    start: DateLike | None,
    end: DateLike | None,
    split: str | None,
    warmup: timedelta | None,
    allow_test: bool,
    columns: Sequence[str] | None,
) -> pl.LazyFrame:
    read_from, eval_from, read_to = _resolve_window(
        start, end, split, warmup, allow_test
    )
    files = _month_files(directory, read_from, read_to)
    if not files:
        raise FileNotFoundError(
            f"no parquet in {directory} covering the requested window"
        )

    # An explicit sorted file list rather than a glob: chronological order is an
    # invariant the rest of the layer relies on, so it should not depend on how
    # a glob happens to enumerate.
    lazy = pl.scan_parquet(files)
    if columns is not None:
        lazy = lazy.select(columns)
    if read_from is not None:
        lazy = lazy.filter(pl.col("ts") >= read_from)
    if read_to is not None:
        lazy = lazy.filter(pl.col("ts") < read_to)
    if warmup is not None:
        lazy = lazy.with_columns(is_warmup=pl.col("ts") < eval_from)
    return lazy


def load_ticks(
    symbol: str,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    split: str | None = None,
    warmup: timedelta | None = None,
    policy: CleaningPolicy = DEFAULT_POLICY,
    columns: Sequence[str] | None = None,
    lazy: bool = False,
    allow_test: bool = False,
) -> TickFrame:
    """Load cleaned ticks for one symbol.

    ``lazy=True`` is strongly preferred for a whole symbol: EURUSD is 86 million
    ticks and XAUUSD 283 million, so an eager whole-corpus load is measured in
    gigabytes. Filters push into the scan and whole months are skipped on their
    filename, so a narrow window costs close to nothing.

    Under the default policy ``ts`` is **not unique** - the feed stamps to the
    millisecond and a fast market puts several quotes inside one. Pass
    ``policy=UNIQUE_TS_POLICY`` if you need a unique index, knowing that it
    deletes real quotes to get one.
    """
    spec = get_spec(symbol)
    frame = clean_ticks(
        _read(
            paths.tick_partition_dir(spec.name),
            start=start,
            end=end,
            split=split,
            warmup=warmup,
            allow_test=allow_test,
            columns=columns,
        ),
        policy,
    )
    if lazy:
        return frame

    ticks = frame.collect()
    if policy.verify_sorted and not ticks["ts"].is_sorted():
        raise ValueError(
            f"{spec.name}: ticks are not ascending after load - conversion "
            "guarantees per-file order, so this means the file list is wrong"
        )
    return ticks


def load_bars(
    symbol: str,
    interval: str = "1m",
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    split: str | None = None,
    warmup: timedelta | None = None,
    columns: Sequence[str] | None = None,
    lazy: bool = False,
    allow_test: bool = False,
) -> TickFrame:
    """Load pre-built bars, which are cheap enough to load eagerly by default.

    There is no cleaning policy argument: bars were built from cleaned ticks, so
    re-applying a policy here would either be a no-op or a silent disagreement
    with what is on disk.
    """
    spec = get_spec(symbol)
    frame = _read(
        paths.bar_partition_dir(spec.name, interval),
        start=start,
        end=end,
        split=split,
        warmup=warmup,
        allow_test=allow_test,
        columns=columns,
    )
    return frame if lazy else frame.collect()


def available_months(symbol: str, interval: str | None = None) -> list[tuple[int, int]]:
    """(year, month) pairs on disk for a symbol's ticks, or for its bars."""
    directory = (
        paths.tick_partition_dir(symbol)
        if interval is None
        else paths.bar_partition_dir(symbol, interval)
    )
    return [
        (int(p.stem[-7:-3]), int(p.stem[-2:]))
        for p in sorted(directory.glob("*.parquet"))
    ]
