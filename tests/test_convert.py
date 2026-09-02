"""Conversion invariants: the pipeline must be lossless and must fail loudly."""

from __future__ import annotations

import polars as pl
import pytest

from qlab.convert import ConversionError, SourceFile, convert_file
from qlab.symbols import get_spec

HEADER = '"Exness","Symbol","Timestamp","Bid","Ask"\n'
SPEC = get_spec("EURUSD")


def _row(ts: str, bid: float, ask: float, symbol: str = "EURUSD_Raw_Spread") -> str:
    return f'"exness","{symbol}","{ts}",{bid},{ask}\n'


def _write_csv(tmp_path, body: str):
    csv_path = tmp_path / "Exness_EURUSD_Raw_Spread_2024_06.csv"
    csv_path.write_text(HEADER + body, encoding="utf-8")
    return csv_path


def _source(tmp_path, csv_path, monkeypatch) -> SourceFile:
    monkeypatch.setattr("qlab.paths.TICKS_DIR", tmp_path / "out")
    src = SourceFile("EURUSD", 2024, 6, csv_path)
    object.__setattr__(src, "_out", tmp_path / "out.parquet")
    return src


def test_roundtrip_preserves_every_tick(tmp_path, monkeypatch):
    body = (
        _row("2024-06-03 00:00:01.100Z", 1.08478, 1.08503)
        + _row("2024-06-03 00:00:02.250Z", 1.08479, 1.08479)
        + _row("2024-06-03 00:00:02.250Z", 1.08480, 1.08480)  # duplicate timestamp
    )
    csv_path = _write_csv(tmp_path, body)
    out = tmp_path / "out.parquet"
    monkeypatch.setattr(SourceFile, "parquet_path", property(lambda self: out))

    record = convert_file(SourceFile("EURUSD", 2024, 6, csv_path), SPEC)
    ticks = pl.read_parquet(out)

    # Duplicates are reported, never removed - cleaning is a Stage 1 decision.
    assert record["rows"] == 3
    assert record["duplicate_ts"] == 1
    assert ticks.height == 3
    assert ticks.columns == ["ts", "bid", "ask"]
    assert ticks["ts"].dtype == pl.Datetime("us", "UTC")
    assert ticks["bid"].to_list() == [1.08478, 1.08479, 1.08480]


def test_unsorted_input_is_sorted_and_flagged(tmp_path, monkeypatch):
    body = (
        _row("2024-06-03 00:00:09.000Z", 1.2, 1.2)
        + _row("2024-06-03 00:00:01.000Z", 1.1, 1.1)
    )
    csv_path = _write_csv(tmp_path, body)
    out = tmp_path / "out.parquet"
    monkeypatch.setattr(SourceFile, "parquet_path", property(lambda self: out))

    record = convert_file(SourceFile("EURUSD", 2024, 6, csv_path), SPEC)

    assert record["source_sorted"] is False
    assert pl.read_parquet(out)["ts"].is_sorted()


def test_wrong_symbol_raises(tmp_path, monkeypatch):
    csv_path = _write_csv(
        tmp_path, _row("2024-06-03 00:00:01.000Z", 1.1, 1.1, symbol="GBPUSD_Raw_Spread")
    )
    monkeypatch.setattr(
        SourceFile, "parquet_path", property(lambda self: tmp_path / "out.parquet")
    )
    with pytest.raises(ConversionError, match="Symbol column"):
        convert_file(SourceFile("EURUSD", 2024, 6, csv_path), SPEC)


def test_empty_file_raises(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path, "")
    monkeypatch.setattr(
        SourceFile, "parquet_path", property(lambda self: tmp_path / "out.parquet")
    )
    with pytest.raises(ConversionError, match="no rows"):
        convert_file(SourceFile("EURUSD", 2024, 6, csv_path), SPEC)
