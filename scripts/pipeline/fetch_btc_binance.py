"""Fetch BTCUSDT 5-minute klines from Binance's public monthly dumps.

Kept deliberately apart from ``data/processed`` - that tree is the broker tick
corpus, converted losslessly from one vendor under one cleaning policy, and
Binance spot klines are neither. They land in ``data/external/binance`` so no
later reader can mistake exchange klines for the broker feed.

The 5-minute frequency is not a choice of ours: Eross et al. (2017) aggregate
BTC-e tick data to 5 minutes, and a replication that changes the sampling
frequency is testing a different thing.
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{itv}/{sym}-{itv}-{ym}.zip"
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OUT = Path("data/external/binance")


def months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_month(sym: str, itv: str, ym: str) -> pl.DataFrame | None:
    url = BASE.format(sym=sym, itv=itv, ym=ym)
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        data = zf.read(zf.namelist()[0])
    # Some months ship a header row, some do not.
    has_header = data[:9] == b"open_time"
    df = pl.read_csv(
        io.BytesIO(data),
        has_header=has_header,
        new_columns=None if has_header else COLS,
        schema_overrides={c: pl.Float64 for c in
                          ("open", "high", "low", "close", "volume",
                           "quote_volume", "taker_buy_base", "taker_buy_quote")},
    )
    # open_time is epoch millis in older dumps and micros from 2025 onward.
    unit = "us" if df["open_time"].max() > 3e14 else "ms"
    return df.select(
        pl.from_epoch("open_time", time_unit=unit).dt.replace_time_zone("UTC").alias("ts_open"),
        pl.col("open", "high", "low", "close", "volume", "quote_volume"),
        pl.col("trades").cast(pl.Int64),
        pl.col("taker_buy_base"),
    )


def main() -> int:
    sym, itv = "BTCUSDT", "5m"
    OUT.mkdir(parents=True, exist_ok=True)
    got, missing = 0, []
    for ym in months((2020, 1), (2026, 8)):
        dest = OUT / f"{sym}_{itv}_{ym.replace('-', '_')}.parquet"
        if dest.exists():
            got += 1
            continue
        df = fetch_month(sym, itv, ym)
        if df is None:
            missing.append(ym)
            continue
        df.write_parquet(dest)
        got += 1
        print(f"{ym}: {df.height} bars", flush=True)
    print(f"done: {got} months on disk, missing={missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
