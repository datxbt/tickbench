"""Raw vendor CSV -> columnar Parquet conversion.

Design contract
---------------
Conversion is **lossless and non-opinionated**. It does exactly four things:

1. Drops the two constant columns (``Exness``, ``Symbol``) after *verifying*
   they are constant and match the expected feed symbol.
2. Parses the ISO-8601 timestamp string into a UTC-aware microsecond datetime.
3. Enforces ascending timestamp order.
4. Writes zstd-compressed Parquet.

It does **not** deduplicate, filter outliers, drop zero-spread ticks or repair
anything. Every such decision is a modelling choice that belongs downstream
(Stage 1), where it is explicit, documented and reversible. If conversion
silently "fixed" the data, every backtest built on it would inherit an
invisible assumption.

Structural violations raise :class:`ConversionError` rather than being papered
over, so a corrupt input file is loud instead of quietly wrong.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from . import paths
from .symbols import SymbolSpec, get_spec

TS_FORMAT = "%Y-%m-%d %H:%M:%S%.fZ"
RAW_COLUMNS = ["Symbol", "Timestamp", "Bid", "Ask"]
ROW_GROUP_SIZE = 1_000_000

_FILENAME_RE = re.compile(r"_(?P<year>\d{4})_(?P<month>\d{2})\.csv$", re.IGNORECASE)


class ConversionError(RuntimeError):
    """Raised when a source file violates an expected structural invariant."""


@dataclass(frozen=True)
class SourceFile:
    symbol: str
    year: int
    month: int
    csv_path: Path

    @property
    def parquet_path(self) -> Path:
        return paths.tick_parquet_path(self.symbol, self.year, self.month)


def discover(symbols: Iterable[str] | None = None) -> list[SourceFile]:
    """Enumerate every raw monthly CSV for the requested symbols."""
    from .symbols import ALL_SYMBOLS

    found: list[SourceFile] = []
    for symbol in symbols or ALL_SYMBOLS:
        spec = get_spec(symbol)
        raw_dir = paths.RAW_TICK_DIR / spec.raw_dirname
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw directory missing for {spec.name}: {raw_dir}")
        for csv_path in sorted(raw_dir.glob("*.csv")):
            match = _FILENAME_RE.search(csv_path.name)
            if match is None:
                raise ConversionError(
                    f"cannot parse year/month from filename: {csv_path.name}"
                )
            found.append(
                SourceFile(
                    symbol=spec.name,
                    year=int(match["year"]),
                    month=int(match["month"]),
                    csv_path=csv_path,
                )
            )
    return found


def is_stale(src: SourceFile) -> bool:
    """True if the parquet output is missing or older than its source CSV."""
    out = src.parquet_path
    if not out.exists():
        return True
    return out.stat().st_mtime < src.csv_path.stat().st_mtime


def convert_file(src: SourceFile, spec: SymbolSpec) -> dict:
    """Convert one monthly CSV to Parquet. Returns a manifest record."""
    started = time.perf_counter()

    frame = pl.read_csv(
        src.csv_path,
        columns=RAW_COLUMNS,
        schema_overrides={
            "Symbol": pl.String,
            "Timestamp": pl.String,
            "Bid": pl.Float64,
            "Ask": pl.Float64,
        },
    )
    if frame.height == 0:
        raise ConversionError(f"{src.csv_path.name}: file contains no rows")

    feed_symbols = frame["Symbol"].unique().to_list()
    if feed_symbols != [spec.feed_symbol]:
        raise ConversionError(
            f"{src.csv_path.name}: expected Symbol column to be constant "
            f"{spec.feed_symbol!r}, found {feed_symbols!r}"
        )

    ticks = frame.select(
        pl.col("Timestamp")
        .str.to_datetime(format=TS_FORMAT, time_unit="us", time_zone="UTC")
        .alias("ts"),
        pl.col("Bid").alias("bid"),
        pl.col("Ask").alias("ask"),
    )

    null_counts = dict(zip(ticks.columns, ticks.null_count().row(0)))
    if any(null_counts.values()):
        raise ConversionError(f"{src.csv_path.name}: null values present {null_counts}")

    was_sorted = bool(ticks["ts"].is_sorted())
    if not was_sorted:
        ticks = ticks.sort("ts", maintain_order=True)

    out_path = src.parquet_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file and rename, so an interrupted run can never leave a
    # truncated parquet that the resume logic would mistake for complete work.
    tmp_path = out_path.with_suffix(".parquet.tmp")
    ticks.write_parquet(
        tmp_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
    )
    os.replace(tmp_path, out_path)

    return {
        "symbol": src.symbol,
        "year": src.year,
        "month": src.month,
        "rows": ticks.height,
        "ts_min": ticks["ts"][0],
        "ts_max": ticks["ts"][-1],
        "duplicate_ts": ticks.height - ticks["ts"].n_unique(),
        "source_sorted": was_sorted,
        "src_bytes": src.csv_path.stat().st_size,
        "out_bytes": out_path.stat().st_size,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "source_file": src.csv_path.name,
    }


def _job(payload: tuple[str, int, int, str]) -> dict:
    """Picklable worker entry point (Windows uses spawn, so this must be top level)."""
    symbol, year, month, csv_path = payload
    src = SourceFile(symbol=symbol, year=year, month=month, csv_path=Path(csv_path))
    try:
        record = convert_file(src, get_spec(symbol))
        record["status"] = "ok"
        return record
    except Exception as exc:  # surfaced to the caller, never silently dropped
        return {
            "symbol": symbol,
            "year": year,
            "month": month,
            "source_file": Path(csv_path).name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(
    symbols: Iterable[str] | None = None,
    *,
    force: bool = False,
    workers: int = 6,
) -> Iterator[dict]:
    """Convert all (stale) source files, yielding one manifest record each.

    Results are yielded as they complete so callers can stream progress.
    """
    paths.ensure_dirs()
    sources = discover(symbols)
    todo = sources if force else [s for s in sources if is_stale(s)]

    for src in sources:
        src.parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if not todo:
        return

    payloads = [(s.symbol, s.year, s.month, str(s.csv_path)) for s in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, p) for p in payloads]
        for future in as_completed(futures):
            yield future.result()


MANIFEST_SCHEMA = {
    "symbol": pl.String,
    "year": pl.Int32,
    "month": pl.Int8,
    "rows": pl.Int64,
    "ts_min": pl.Datetime("us", "UTC"),
    "ts_max": pl.Datetime("us", "UTC"),
    "duplicate_ts": pl.Int64,
    "source_sorted": pl.Boolean,
    "src_bytes": pl.Int64,
    "out_bytes": pl.Int64,
    "elapsed_s": pl.Float64,
    "source_file": pl.String,
}


def write_manifest(records: list[dict]) -> Path:
    """Persist the inventory of converted files, merged with any prior run."""
    ok = [{k: r[k] for k in MANIFEST_SCHEMA} for r in records if r.get("status") == "ok"]
    fresh = pl.DataFrame(ok, schema=MANIFEST_SCHEMA)

    if paths.MANIFEST_PATH.exists():
        previous = pl.read_parquet(paths.MANIFEST_PATH)
        fresh = pl.concat([previous, fresh], how="vertical_relaxed").unique(
            subset=["symbol", "year", "month"], keep="last"
        )

    fresh = fresh.sort(["symbol", "year", "month"])
    fresh.write_parquet(paths.MANIFEST_PATH)
    return paths.MANIFEST_PATH
