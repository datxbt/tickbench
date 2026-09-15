"""Tests for what the TOP8_2026 deployment study added to the breakout model.

Three things: the literal dollar spread cap, the flipped-direction control, and
the grid resolver. The grid is what the selection tests run on, so it has to
agree with the scalar simulator to the cent - otherwise the walk-forward result
would describe a different strategy from the one backtested.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from qlab.strategies.session_breakout import (
    US,
    BreakoutConfig,
    _cells_for_entry,
    _find_entry,
    _simulate_window,
    _Tape,
    spread_cap_price,
)
from qlab.symbols import get_spec

DAY = date(2026, 1, 5)  # a Monday, US standard time
SPEC = get_spec("XAUUSD")
HI, LO = 2010.0, 2000.0
WIDTH = HI - LO


def _tape(prices, spread=0.10):
    base = int(datetime(DAY.year, DAY.month, DAY.day, 1,
                        tzinfo=timezone.utc).timestamp()) * US
    mid = np.asarray(prices, dtype=float)
    return _Tape(ts=base + np.arange(len(mid), dtype=np.int64) * US,
                 bid=mid - spread / 2, ask=mid + spread / 2)


def _cfg(**kw):
    return BreakoutConfig(windows=((0, 60, 3.0),), **kw)


def _sim(prices, *, cfg=None, spread=0.10, slip=0.0, target_mult=3.0, cap=1.0):
    return _simulate_window(_tape(prices, spread), DAY, 0, 60, target_mult,
                            HI, LO, WIDTH, cfg or _cfg(), slip, SPEC, spread_cap=cap)


# --------------------------------------------------------------------------
# The dollar cap
# --------------------------------------------------------------------------

def test_dollar_cap_is_taken_literally():
    assert spread_cap_price(_cfg(max_spread_usd=0.30), None, SPEC) == 0.30
    assert spread_cap_price(_cfg(max_spread_usd=float("inf")), None, SPEC) == float("inf")


def test_dollar_cap_binds_at_arming():
    prices = [2005.0, 2010.5, 2041.0]
    cap = spread_cap_price(_cfg(max_spread_usd=0.30), None, SPEC)
    assert _sim(prices, spread=0.25, cap=cap) is not None
    assert _sim(prices, spread=0.35, cap=cap) is None


# --------------------------------------------------------------------------
# The flipped-direction control
# --------------------------------------------------------------------------

def test_flip_takes_the_other_side_at_the_same_instant():
    prices = [2005.0, 2010.5, 2015.0, 2021.5, 2041.0]
    real = _sim(prices)
    flip = _sim(prices, cfg=_cfg(flip_direction=True))
    assert real["direction"] == 1 and real["reason"] == "tp"
    assert flip["direction"] == -1
    assert flip["entry_ts"] == real["entry_ts"]
    assert flip["entry"] == pytest.approx(2010.45)           # the bid, not the ask
    # Same stop distance as the real trade (ask 2010.55 down to LO), mirrored.
    assert flip["stop"] == pytest.approx(2010.45 + 10.55)
    assert flip["reason"] == "stop"
    assert flip["exit"] == pytest.approx(2021.55)


def test_flip_wins_where_the_breakout_loses():
    prices = [2005.0, 2010.5, 2005.0, 1999.0, 1979.0]
    assert _sim(prices)["reason"] == "stop"
    flip = _sim(prices, cfg=_cfg(flip_direction=True))
    assert flip["reason"] == "tp"
    assert flip["r_gross"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The grid resolver against the scalar simulator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prices", [
    [2005.0, 2010.5, 2041.0],                                # long to target
    [2005.0, 2010.5, 2015.0, 2021.5, 2041.0],                # long, gradual
    [2005.0, 2010.5, 2005.0, 1999.0, 1979.0],                # long stopped
    [2005.0, 1999.9, 1990.0, 1985.0, 1979.0, 2011.0],        # short, mixed by target
    [2005.0, 2010.5] + [2012.0] * 50,                        # held to the flatten
])
@pytest.mark.parametrize("slip", [0.0, 0.4])
def test_grid_resolver_matches_the_scalar_simulator(prices, slip):
    tape, cfg = _tape(prices), _cfg()
    targets = np.array([1.0, 2.0, 3.0])
    ent = _find_entry(tape, DAY, 0, 60, HI, LO, cfg, slip, 1.0)
    assert ent is not None
    cells = _cells_for_entry(tape, ent, HI, LO, WIDTH, targets, cfg, slip, SPEC)
    for k, t in enumerate(targets):
        real = _simulate_window(tape, DAY, 0, 60, t, HI, LO, WIDTH, cfg, slip, SPEC, 1.0)
        flip = _simulate_window(tape, DAY, 0, 60, t, HI, LO, WIDTH,
                                _cfg(flip_direction=True), slip, SPEC, 1.0)
        assert cells["r"][k] == pytest.approx(real["r_multiple"], abs=1e-12)
        assert cells["usd"][k] == pytest.approx(real["net_usd"], abs=1e-9)
        assert cells["r_flip"][k] == pytest.approx(flip["r_multiple"], abs=1e-12)
        assert cells["usd_flip"][k] == pytest.approx(flip["net_usd"], abs=1e-9)
