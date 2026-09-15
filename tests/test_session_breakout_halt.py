"""Tests for the daily loss halt reduction.

A halt fires on floating equity, so the numbers that matter are the intraday
marks and what closing at the market is worth at that instant. Both are pinned
on a hand-built tape where every mark can be written down.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.session_breakout import US, DayHalt, _day_halt, _Tape
from qlab.symbols import get_spec

DAY = date(2026, 1, 5)
SPEC = get_spec("XAUUSD")
LOTS = 0.02
NO_SLIP = np.zeros(24)


def _tape(prices, spread=0.10):
    base = int(datetime(DAY.year, DAY.month, DAY.day, 1,
                        tzinfo=timezone.utc).timestamp()) * US
    mid = np.asarray(prices, dtype=float)
    return _Tape(ts=base + np.arange(len(mid), dtype=np.int64) * US,
                 bid=mid - spread / 2, ask=mid + spread / 2)


# Long filled at the ask 2010.55 on tick 1, stopped on the bid 1998.95 at tick 5.
PRICES = [2005.0, 2010.5, 2008.0, 2004.0, 2001.0, 1999.0, 1999.0]
TAPE = _tape(PRICES)
NET_A = (1998.95 - 2010.55) * 100 - 7.0          # per lot, commission included


def _trades(rows):
    return pl.DataFrame(rows, schema={
        "entry_ts": pl.Int64, "exit_ts": pl.Int64, "direction": pl.Int64,
        "entry": pl.Float64, "net_usd": pl.Float64, "commission_usd": pl.Float64},
        orient="row")


def test_intraday_lows_are_marked_at_the_bid_with_half_commission():
    h = _day_halt(TAPE, _trades([(TAPE.ts[1], TAPE.ts[5], 1, 2010.55, NET_A, 7.0)]),
                  LOTS, SPEC, NO_SLIP)
    # Marks at 0.02 lots: $2 per $1 of gold, less the entry deal's $0.07.
    assert h.loss_at == pytest.approx([0.27, 5.27, 13.27, 19.27, 23.34])
    assert h.pnl_full == pytest.approx(NET_A * LOTS)


def test_halt_closes_at_the_first_low_past_the_limit():
    h = _day_halt(TAPE, _trades([(TAPE.ts[1], TAPE.ts[5], 1, 2010.55, NET_A, 7.0)]),
                  LOTS, SPEC, NO_SLIP)
    pnl, fired = h.pnl(10.0)
    assert fired
    assert pnl == pytest.approx((2003.95 - 2010.55) * 2 - 0.14)   # closed at tick 3
    assert h.pnl(100.0) == (pytest.approx(NET_A * LOTS), False)
    assert h.pnl(0.0) == (pytest.approx(NET_A * LOTS), False)     # 0 is "off"


def test_halt_close_pays_slippage():
    h = _day_halt(TAPE, _trades([(TAPE.ts[1], TAPE.ts[5], 1, 2010.55, NET_A, 7.0)]),
                  LOTS, SPEC, np.full(24, 0.5))
    pnl, _ = h.pnl(10.0)
    assert pnl == pytest.approx((2003.95 - 0.5 - 2010.55) * 2 - 0.14)


def test_entries_after_the_halt_never_happen():
    trades = _trades([
        (TAPE.ts[1], TAPE.ts[5], 1, 2010.55, NET_A, 7.0),
        (TAPE.ts[6], TAPE.ts[6], 1, 1999.05, 2500.0, 7.0),   # a +$50 winner, later
    ])
    h = _day_halt(TAPE, trades, LOTS, SPEC, NO_SLIP)
    assert h.pnl_full == pytest.approx(NET_A * LOTS + 50.0)
    pnl, fired = h.pnl(10.0)
    assert fired and pnl == pytest.approx((2003.95 - 2010.55) * 2 - 0.14)


def test_a_day_that_never_dips_has_nothing_to_halt():
    h = DayHalt(np.empty(0), np.empty(0), 12.5)
    assert h.pnl(1.0) == (12.5, False)
