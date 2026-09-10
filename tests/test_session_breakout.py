"""Tests for the session opening-range breakout model.

The point of interest is the fill logic. A breakout entry, a stop and a target
are all resolved against the same tape, and getting any of the three trigger
sides wrong - ask versus bid, or slippage on a limit - moves the result by more
than the edge being measured. So those are pinned against a hand-built tape
where the right answer is known by construction.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.session_breakout import (
    PRESETS,
    US,
    BreakoutConfig,
    _brackets,
    _simulate_window,
    _Tape,
    flatten_second_utc,
    is_us_dst,
)
from qlab.symbols import get_spec

DAY = date(2026, 1, 5)  # a Monday, US standard time
SPEC = get_spec("XAUUSD")


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "day, dst",
    [
        (date(2026, 3, 7), False),   # day before the second Sunday in March
        (date(2026, 3, 8), True),    # the second Sunday itself
        (date(2026, 10, 31), True),
        (date(2026, 11, 1), False),  # the first Sunday in November
        (date(2024, 3, 10), True),   # 2024's second Sunday
        (date(2024, 3, 9), False),
    ],
)
def test_us_dst_boundaries(day, dst):
    assert is_us_dst(day) is dst


def test_flatten_tracks_the_new_york_halt():
    """Summer and winter differ by exactly the hour DST moves, not by the clock."""
    summer = flatten_second_utc(date(2026, 7, 6))   # Monday
    winter = flatten_second_utc(date(2026, 1, 5))   # Monday
    assert summer == 20 * 3600 + 53 * 60
    assert winter == 21 * 3600 + 53 * 60
    assert winter - summer == 3600


def test_friday_closes_early_and_overrides_the_halt():
    assert flatten_second_utc(date(2026, 1, 9)) == 20 * 3600  # a Friday
    assert flatten_second_utc(date(2026, 1, 8)) > 20 * 3600   # the Thursday


# --------------------------------------------------------------------------
# The presets, against the expert's LoadPreset()
# --------------------------------------------------------------------------

def test_presets_match_the_expert():
    assert PRESETS["GEO_2026"] == PRESETS["GEO_2026_NO_H13"][:6] + (
        (13, 60, 3.0), (14, 60, 3.0))
    # The whole claim of the 2026 tune is that geometry is uniform.
    assert {(r, t) for _, r, t in PRESETS["GEO_2026"]} == {(60, 3.0)}
    # ...and that the original was not.
    assert len({(r, t) for _, r, t in PRESETS["ORIGINAL"]}) > 1
    for name, windows in PRESETS.items():
        hours = [h for h, _, _ in windows]
        assert len(hours) == len(set(hours)), f"{name} arms an hour twice"
        assert len(windows) <= 8, f"{name} exceeds MAX_WINDOWS"


# --------------------------------------------------------------------------
# Bracket construction
# --------------------------------------------------------------------------

def _bars(rows):
    return pl.DataFrame(
        rows, schema={"ts_open": pl.Datetime("us", "UTC"),
                      "high": pl.Float64, "low": pl.Float64},
        orient="row")


def test_bracket_keys_off_ts_open_not_the_label():
    """A bar covering [00:59, 01:00) belongs to hour 0, not hour 1."""
    rows = [(datetime(2026, 1, 5, 0, m, tzinfo=timezone.utc), 100.0 + m, 99.0)
            for m in range(60)]
    rows.append((datetime(2026, 1, 5, 1, 0, tzinfo=timezone.utc), 999.0, 1.0))
    cfg = BreakoutConfig(windows=((0, 60, 3.0),), min_range_pct=0.0,
                         max_range_pct=100.0)
    out = _brackets(_bars(rows), 0, 60, cfg)
    assert out.height == 1
    assert out["hi"][0] == pytest.approx(159.0)   # not 999, which is hour 1's
    assert out["lo"][0] == pytest.approx(99.0)


def test_bracket_guards_drop_too_narrow_and_too_wide():
    def one(hi, lo):
        rows = [(datetime(2026, 1, 5, 0, m, tzinfo=timezone.utc), hi, lo)
                for m in range(60)]
        cfg = BreakoutConfig(windows=((0, 60, 3.0),),
                             min_range_pct=0.05, max_range_pct=2.0)
        return _brackets(_bars(rows), 0, 60, cfg).height

    assert one(2000.5, 2000.0) == 0    # 0.025% - under the floor
    assert one(2100.0, 2000.0) == 0    # 4.9%   - over the ceiling
    assert one(2010.0, 2000.0) == 1    # 0.50%  - inside


def test_bracket_requires_enough_quoted_minutes():
    rows = [(datetime(2026, 1, 5, 0, m, tzinfo=timezone.utc), 2010.0, 2000.0)
            for m in range(10)]           # only 10 of 60 minutes quoted
    cfg = BreakoutConfig(windows=((0, 60, 3.0),), min_range_pct=0.0)
    assert _brackets(_bars(rows), 0, 60, cfg).height == 0


# --------------------------------------------------------------------------
# Fills
# --------------------------------------------------------------------------

def _tape(prices, *, spread=0.10, start_hour=1):
    """A tape of mids at one-second spacing, starting at ``start_hour``:00."""
    base = int(datetime(DAY.year, DAY.month, DAY.day, start_hour,
                        tzinfo=timezone.utc).timestamp()) * US
    mid = np.asarray(prices, dtype=float)
    return _Tape(ts=base + np.arange(len(mid), dtype=np.int64) * US,
                 bid=mid - spread / 2, ask=mid + spread / 2)


HI, LO = 2010.0, 2000.0
WIDTH = HI - LO
CFG = BreakoutConfig(windows=((0, 60, 3.0),))


def _run(prices, *, slip=0.0, target_mult=3.0, spread=0.10):
    return _simulate_window(
        _tape(prices, spread=spread), DAY, 0, 60, target_mult,
        HI, LO, WIDTH, CFG, slip, SPEC, spread_cap=1.0)


def test_long_breakout_to_target_pays_exactly_the_target_gross():
    # Ask must reach HI to trigger; bid must reach entry + 3*width to take profit.
    prices = [2005.0] + [2010.5] + [2041.0] * 5
    r = _run(prices)
    assert r["direction"] == 1
    assert r["reason"] == "tp"
    # Entry is the ask, so the target sits 3 widths above the ask, not the mid.
    assert r["entry"] == pytest.approx(2010.55)
    assert r["target"] == pytest.approx(2010.55 + 3 * WIDTH)
    assert r["r_gross"] == pytest.approx(3.0)
    assert r["r_multiple"] < 3.0            # commission still comes off


def test_short_breakout_to_stop_loses_about_one_r():
    # Bid falls to LO to trigger the sell stop, then ask climbs back through HI.
    prices = [2005.0, 1999.9] + [2010.2] * 5
    r = _run(prices)
    assert r["direction"] == -1
    assert r["reason"] == "stop"
    assert r["r_multiple"] == pytest.approx(-1.0, abs=0.05)
    assert r["r_multiple"] < -1.0           # cost makes a stop worse than -1R


def test_a_stop_pays_slippage_but_a_target_does_not():
    """Slippage belongs on market orders only - a limit fills at its price."""
    # Entry slips to 2011.05, so the target moves up with it to 2041.05.
    tp = _run([2005.0, 2010.5] + [2042.0] * 5, slip=0.5)
    assert tp["reason"] == "tp"
    assert tp["exit"] == pytest.approx(tp["target"])     # untouched by slip

    st = _run([2005.0, 2010.5] + [1999.0] * 5, slip=0.5)
    assert st["reason"] == "stop"
    assert st["exit"] == pytest.approx(1999.0 - 0.05 - 0.5)  # bid, then slipped


def test_entry_slippage_moves_against_the_position_both_ways():
    long = _run([2005.0, 2010.5, 2041.0], slip=0.5)
    assert long["entry"] == pytest.approx(2010.55 + 0.5)
    short = _run([2005.0, 1999.9, 1960.0], slip=0.5)
    assert short["entry"] == pytest.approx(1999.85 - 0.5)


def test_no_touch_never_fires():
    assert _run([2005.0] * 20) is None


def test_already_outside_the_bracket_is_skipped():
    """The expert refuses to arm one side into a break that already happened."""
    assert _run([2015.0, 2020.0, 2030.0]) is None   # above HI at arming
    assert _run([1995.0, 1990.0, 1980.0]) is None   # below LO at arming


def test_wide_spread_at_arming_is_skipped():
    assert _run([2005.0, 2010.5, 2041.0], spread=3.0) is None


def test_unfilled_at_the_flatten_exits_at_the_market():
    """A position still open at the daily flatten is closed, not carried."""
    # Arms at 01:00 and drifts; the flatten is 21:53 UTC in January.
    r = _run([2005.0, 2010.5] + [2012.0] * 400)
    assert r is not None
    assert r["reason"] == "flat"
    assert r["exit"] < r["target"] and r["exit"] > r["stop"]


def test_risk_is_the_bracket_width_so_r_is_scale_free():
    r = _run([2005.0, 2010.5, 2041.0])
    assert r["risk_usd"] == pytest.approx(WIDTH * SPEC.contract_size)
    assert r["r_multiple"] == pytest.approx(r["net_usd"] / r["risk_usd"])


def test_target_multiple_is_honoured():
    for mult in (1.0, 2.0, 3.0):
        prices = [2005.0, 2010.5] + [2010.55 + mult * WIDTH + 1.0] * 3
        r = _run(prices, target_mult=mult)
        assert r["reason"] == "tp"
        assert r["r_gross"] == pytest.approx(mult)
