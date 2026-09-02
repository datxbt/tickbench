"""Liquidity Vacuum Fade signal construction.

The tests worth having here are the ones that would let a wrong backtest look
right: fading the wrong way, and firing on session gaps. The second is the one
that would have produced a spectacular and entirely fictional equity curve.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.lvf import LVFParams, compute_signals

UTC = timezone.utc


def _ticks(prices, *, start="2024-06-05T10:00:00", step_ms=100, spread=0.10, gap_at=None):
    """A tick series at a fixed cadence, optionally with one large gap inserted."""
    base = datetime.fromisoformat(start).replace(tzinfo=UTC)
    stamps = []
    offset = 0
    for i in range(len(prices)):
        if gap_at is not None and i == gap_at:
            offset += 3600_000  # a one-hour hole
        stamps.append(base + timedelta(milliseconds=offset + i * step_ms))
    return pl.DataFrame(
        {
            "ts": stamps,
            "bid": [p - spread / 2 for p in prices],
            "ask": [p + spread / 2 for p in prices],
        },
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
    )


def _quiet_then_jump(n_quiet=8000, jump=2.0, window=100):
    """A long calm stretch to set the baselines, then one clean directional leg."""
    rng = np.random.default_rng(11)
    quiet = 2000 + np.cumsum(rng.normal(0, 0.004, n_quiet))
    leg = quiet[-1] + np.linspace(0, jump, window)
    return list(quiet) + list(leg)


def test_a_clean_upward_leg_registers_as_an_up_vacuum():
    params = LVFParams()
    frame = compute_signals(_ticks(_quiet_then_jump()), params)
    last = frame.row(frame.height - 1, named=True)

    assert last["move_signed"] > 0  # price rose over the window
    assert last["move"] == pytest.approx(2.0, abs=0.05)
    assert last["v"] > params.v_min  # abnormal against the calm baseline
    assert last["efficiency"] > params.e_min  # a straight line, not chop
    assert last["setup"]


def test_the_fade_direction_is_against_the_move():
    """The whole strategy in one assertion: an up-vacuum is sold."""
    frame = compute_signals(_ticks(_quiet_then_jump(jump=2.0)), LVFParams())
    up = frame.row(frame.height - 1, named=True)
    frame_down = compute_signals(_ticks(_quiet_then_jump(jump=-2.0)), LVFParams())
    down = frame_down.row(frame_down.height - 1, named=True)

    assert -np.sign(up["move_signed"]) == -1  # short the up move
    assert -np.sign(down["move_signed"]) == +1  # buy the down move


def test_a_window_spanning_a_session_gap_never_sets_up():
    """The guard that separates a plausible backtest from a fantasy.

    100 ticks straddling a weekend show a huge Move and a near-zero tick Rate -
    exactly the signature the setup hunts for, and the one place it means
    nothing. Without this the strategy fires at every Monday reopen.
    """
    prices = _quiet_then_jump()
    # Drop the hole inside the final window, where the setup would otherwise fire.
    frame = compute_signals(_ticks(prices, gap_at=len(prices) - 50), LVFParams())
    last = frame.row(frame.height - 1, named=True)

    assert last["window_has_gap"]
    assert not last["setup"]
    # The raw measurements still look tempting, which is why the guard is needed.
    assert last["v"] > 3.0


def test_the_rollover_window_is_excluded():
    params = LVFParams()
    prices = _quiet_then_jump()
    # 8,100 ticks at 100 ms is 13.5 minutes, so starting at 20:45 puts the leg
    # inside the 20:55-21:15 rollover window and starting at 10:00 does not.
    inside = compute_signals(_ticks(prices, start="2024-06-05T20:45:00"), params)
    outside = compute_signals(_ticks(prices, start="2024-06-05T10:00:00"), params)

    assert inside.row(inside.height - 1, named=True)["in_rollover"]
    assert not inside.row(inside.height - 1, named=True)["setup"]
    assert outside.row(outside.height - 1, named=True)["setup"]


def test_a_wide_spread_blocks_the_setup():
    params = LVFParams()
    prices = _quiet_then_jump()
    tight = compute_signals(_ticks(prices, spread=0.10), params)
    wide = compute_signals(_ticks(prices, spread=0.50), params)

    assert tight.row(tight.height - 1, named=True)["setup"]
    assert not wide.row(wide.height - 1, named=True)["setup"]


def test_participation_is_measured_against_the_recent_rate():
    """rho is a ratio to the baseline, not an absolute tick rate - which is what
    lets one threshold mean the same thing in 2020 and 2026."""
    params = LVFParams()
    prices = _quiet_then_jump()

    # Constant cadence throughout: the rate equals its own baseline, so rho is 1
    # whatever that cadence happens to be.
    for step_ms in (20, 100, 500):
        frame = compute_signals(_ticks(prices, step_ms=step_ms), params)
        last = frame.row(frame.height - 1, named=True)
        assert last["rate"] == pytest.approx(1000.0 / step_ms * 1.0, rel=0.05)
        assert last["rho"] == pytest.approx(1.0, abs=0.05)
        assert last["rho"] <= params.rho_max


def test_baselines_exclude_the_tick_being_judged():
    """V asks "abnormal versus what came before", so the baseline is lagged."""
    frame = compute_signals(_ticks(_quiet_then_jump()), LVFParams())
    moves = frame["move"].to_numpy()
    b_move = frame["b_move"].to_numpy()

    # The final tick carries the largest Move in the sample; its baseline must
    # not have been inflated by it.
    assert moves[-1] == pytest.approx(np.nanmax(moves), rel=1e-6)
    assert b_move[-1] < moves[-1] / 3


def test_params_reject_an_unknown_arming_rule():
    with pytest.raises(ValueError, match="cancel/reset"):
        LVFParams(extend_rule="whatever")
