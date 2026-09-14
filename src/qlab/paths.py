"""Canonical filesystem layout for the project.

Every module resolves paths through here so that nothing hard-codes a location.
The project root is the directory that contains ``Tick_Data``; it can be
overridden with the ``QLAB_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/qlab/paths.py -> src/qlab -> src -> <project root>
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(os.environ.get("QLAB_ROOT", _DEFAULT_ROOT)).resolve()

RAW_TICK_DIR = ROOT / "Tick_Data"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TICKS_DIR = PROCESSED_DIR / "ticks"
BARS_DIR = PROCESSED_DIR / "bars"
WHOLE_BARS_DIR = PROCESSED_DIR / "bars_whole"
COST_DIR = PROCESSED_DIR / "cost"
SPREAD_PROFILE_PATH = COST_DIR / "spread_profile.parquet"
LATENCY_PROFILE_PATH = COST_DIR / "latency_profile.parquet"
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"

REPORTS_DIR = ROOT / "reports"
QUALITY_DIR = REPORTS_DIR / "data_quality"
COST_REPORT_DIR = REPORTS_DIR / "cost_model"
STRATEGY_REPORT_DIR = REPORTS_DIR / "strategies"


def tick_partition_dir(symbol: str) -> Path:
    """Hive-style partition directory for one symbol's parquet ticks."""
    return TICKS_DIR / f"symbol={symbol}"


def tick_parquet_path(symbol: str, year: int, month: int) -> Path:
    return tick_partition_dir(symbol) / f"{symbol}_{year:04d}_{month:02d}.parquet"


def bar_partition_dir(symbol: str, interval: str) -> Path:
    """Hive-style partition directory for one symbol's bars at one interval."""
    return BARS_DIR / f"symbol={symbol}" / f"interval={interval}"


def bar_parquet_path(symbol: str, interval: str, year: int, month: int) -> Path:
    return bar_partition_dir(symbol, interval) / f"{symbol}_{interval}_{year:04d}_{month:02d}.parquet"


def whole_bar_partition_dir(symbol: str) -> Path:
    """Directory for one symbol's whole-history bar caches."""
    return WHOLE_BARS_DIR / f"symbol={symbol}"


def whole_bar_path(symbol: str, interval: str) -> Path:
    """One file covering a symbol's entire history at one interval.

    Deliberately outside :data:`BARS_DIR`. That tree is the monthly partition
    layout, and the loader prunes it on filenames alone - a file there that does
    not name its month cannot be placed, so a whole-history cache dropped into
    it breaks every read of that partition rather than just its own. These
    caches are also free to be *filtered* (see
    ``scripts/pipeline/prepare_fx_bars.py``, which drops thin weekend bars),
    which the partition layout is not: there, a missing row means no ticks.
    """
    return whole_bar_partition_dir(symbol) / f"{symbol}_{interval}.parquet"


def ensure_dirs() -> None:
    for path in (
        PROCESSED_DIR,
        TICKS_DIR,
        BARS_DIR,
        WHOLE_BARS_DIR,
        COST_DIR,
        REPORTS_DIR,
        QUALITY_DIR,
        COST_REPORT_DIR,
        STRATEGY_REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
