"""Tests for the prop-firm rules engine.

A rules engine is worth exactly as much as its edge cases. Every assertion here
is a rule that, if implemented one day or one basis point wrong, turns a failed
challenge into a passed one on paper.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qlab import propfirm as pf


def _const(n, ret, worst=None):
    """A run of identical days."""
    r = np.full(n, ret, dtype=float)
    w = np.full(n, ret if worst is None else worst, dtype=float)
    return r, w


# --------------------------------------------------------------------------
# The floors
# --------------------------------------------------------------------------

def test_the_daily_floor_bites_on_the_intraday_low_not_the_close():
    """A day that dips 6% and closes flat has failed, however good the close is.

    This is the single most common way a close-to-close backtest lies about a
    challenge, so it is the first thing asserted.
    """
    r, w = _const(10, 0.0, worst=-0.06)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome == pf.FAIL_DAILY
    assert a.days == 1


def test_a_dip_inside_the_daily_limit_survives():
    r, w = _const(10, 0.0, worst=-0.049)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome == pf.INCOMPLETE


def test_the_total_floor_is_static_not_trailing():
    """Up 8%, then down 9% from there, is -1.7% overall - and must NOT fail.

    Under a trailing 10% rule it would. FTMO measures against the initial
    balance, and getting that wrong understates every pass rate in the report.
    """
    r = np.array([0.08, -0.09] + [0.0] * 5)
    w = r.copy()
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome != pf.FAIL_TOTAL
    assert a.final_equity == pytest.approx(1.08 * 0.91, rel=1e-9)


def test_the_total_floor_does_bite_at_ten_percent_below_the_start():
    r = np.array([-0.04, -0.04, -0.04])
    a, _ = pf.run_phase(r, r, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome == pf.FAIL_TOTAL


def test_daily_and_total_can_both_be_live_and_daily_is_reported_first():
    """A single -12% day breaches both. Which one is reported matters for the
    diagnosis, so it is pinned rather than left to argument order."""
    r, w = _const(3, -0.12)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome == pf.FAIL_DAILY


# --------------------------------------------------------------------------
# The target
# --------------------------------------------------------------------------

def test_the_minimum_trading_days_gate_the_target():
    """Hitting +10% on day two does not pass a four-day minimum."""
    r = np.array([0.10, 0.0, 0.0, 0.0, 0.0])
    a, idx = pf.run_phase(r, r, pf.Phase("c", 0.10, min_trading_days=4),
                          pf.FTMO_TWO_STEP)
    assert a.outcome == pf.PASSED
    assert a.days == 4, "passed before the minimum was served"
    assert idx == 4


def test_the_target_is_read_on_the_close_not_the_high():
    """+10% intraday that closes at +9% is not a pass; the balance is what counts."""
    r = np.array([0.09] + [0.0] * 6)
    w = np.zeros(7)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP)
    assert a.outcome == pf.INCOMPLETE


def test_each_phase_starts_from_a_fresh_balance():
    """Passing the Challenge at +12% does not carry a 2% buffer into
    Verification - it is a new account with its own floors."""
    r = np.array([0.03] * 4 + [0.02] * 4)
    attempts = pf.run_program(r, r, pf.FTMO_TWO_STEP)
    assert [a.outcome for a in attempts] == [pf.PASSED, pf.PASSED]
    assert attempts[0].final_equity >= 1.10
    # verification needed +5% of its OWN starting balance, not of the carried one
    assert attempts[1].final_equity == pytest.approx(1.02 ** 4, rel=1e-9)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def test_flat_sizing_scales_the_return_linearly():
    r, w = _const(6, 0.02)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP,
                        sizer=pf.flat_size(0.5))
    assert a.final_equity == pytest.approx(1.01 ** 6, rel=1e-9)


def test_cushion_sizing_shrinks_as_the_floor_approaches():
    sizer = pf.cushion_size(1.0)
    floor = 0.90
    assert sizer(1.00, floor) == pytest.approx(1.0)     # full cushion
    assert sizer(0.95, floor) == pytest.approx(0.5)     # half spent
    assert sizer(0.905, floor) == pytest.approx(0.05, abs=1e-9)
    assert sizer(0.90, floor) == pytest.approx(0.0, abs=1e-12)


def test_cushion_sizing_grows_with_room_that_was_earned():
    """At +6% the account is 16% from a static floor, not 10%, and may size up."""
    sizer = pf.cushion_size(1.0)
    assert sizer(1.06, 0.90) == pytest.approx(1.6, rel=1e-9)


def test_cushion_sizing_respects_its_cap():
    sizer = pf.cushion_size(1.0, cap=1.2)
    assert sizer(2.00, 0.90) == pytest.approx(1.2)


def test_cushion_sizing_makes_the_total_floor_much_harder_to_reach():
    """The point of the rule, asserted on a losing streak that flat sizing
    cannot survive."""
    r, w = _const(200, -0.01)
    flat, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP,
                           sizer=pf.flat_size(1.0))
    cush, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), pf.FTMO_TWO_STEP,
                           sizer=pf.cushion_size(1.0))
    assert flat.outcome == pf.FAIL_TOTAL
    assert cush.outcome == pf.INCOMPLETE
    assert cush.final_equity > 0.90


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_censored_paths_are_not_counted_as_failures():
    """With no deadline, "still running at the horizon" is not a failure, and
    folding it into the pass rate understates small sizes badly.

    The fixture is tuned so roughly half the paths reach a modest target inside
    the horizon and the rest are simply unfinished; the floor is far enough away
    that nothing fails outright. The resolved rate should then be 100% while the
    horizon rate is about half.
    """
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.006, 500)
    rules = replace(pf.FTMO_TWO_STEP, phases=(pf.Phase("one", 0.03),))
    b = pf.block_bootstrap(r, r, rules=rules, sizer=pf.flat_size(1.0),
                           n_paths=400, horizon=60, seed=1)
    assert 0.1 < b["censored"] < 0.9, f"fixture censoring {b['censored']:.2f}"
    assert b["outcomes"].get(pf.FAIL_TOTAL, 0) == 0, "floor should be unreachable"
    assert b["pass_rate_resolved"] == pytest.approx(1.0)
    assert b["pass_rate"] < b["pass_rate_resolved"]


def test_a_deadline_turns_slowness_into_failure():
    r, w = _const(300, 0.0002)
    rules = replace(pf.FTMO_TWO_STEP, max_days=30)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules)
    assert a.outcome == pf.FAIL_TIME


def test_by_start_date_covers_every_start():
    r, w = _const(50, 0.001)
    t = pf.by_start_date(r, w, scale=1.0)
    assert t.height == 50
    assert set(t.columns) >= {"start", "passed_all", "outcome", "days"}


# --------------------------------------------------------------------------
# Time to a funded account
# --------------------------------------------------------------------------

def test_time_to_funded_counts_one_attempt_when_the_first_one_works():
    """A constant series makes the bootstrap a no-op, so the answer is exact:
    ln(1.10)/0.004 + ln(1.05)/0.004 days, and one fee."""
    r, w = _const(60, 0.004)
    t = pf.time_to_funded(r, w, n_paths=20, horizon=200)
    assert t["censored"] == 0.0
    assert t["mean_attempts"] == 1.0
    assert t["median_days"] == 37


def test_time_to_funded_censors_a_hopeless_size_rather_than_reporting_a_number():
    """Every attempt fails, so there is no time to funded - and reporting one
    anyway is how a sizing that never works acquires a plausible timeline."""
    r, w = _const(60, -0.01)
    t = pf.time_to_funded(r, w, n_paths=20, horizon=200, max_attempts=5)
    assert t["censored"] == 1.0
    assert np.isnan(t["median_days"])


def _mixed():
    """A series that passes sometimes and fails sometimes, so retries happen."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.002, 0.012, 400)
    return r, r.copy()


def test_retries_make_the_journey_longer_than_the_winning_attempt():
    """The whole point of the function. block_bootstrap conditions on success;
    this does not, and the gap between them is the cost of failing."""
    r, w = _mixed()
    t = pf.time_to_funded(r, w, sizer=pf.flat_size(1.0), n_paths=400,
                          horizon=1500, seed=2)
    b = pf.block_bootstrap(r, w, sizer=pf.flat_size(1.0), n_paths=400,
                           horizon=1500, seed=2)
    assert t["mean_attempts"] > 1.0, "fixture never fails; nothing is being tested"
    assert t["median_days"] > b["median_days_when_passed"]


def test_buying_the_next_entry_costs_days():
    r, w = _mixed()
    quick = pf.time_to_funded(r, w, n_paths=400, horizon=1500, restart_days=0, seed=2)
    slow = pf.time_to_funded(r, w, n_paths=400, horizon=1500, restart_days=20, seed=2)
    assert slow["median_days"] > quick["median_days"]


# --------------------------------------------------------------------------
# What a breaker cannot do
# --------------------------------------------------------------------------

def test_a_breaker_cannot_stop_a_gap():
    """The market reopens 6% against a 3% breaker. There is no fill at 3%.

    Getting this wrong is what makes leverage look safe: it converts the one
    risk a stop cannot manage into one that is managed for free.
    """
    r, w = _const(2, -0.06)
    g = np.full(2, -0.06)
    rules = replace(pf.FTMO_TWO_STEP, daily_stop=0.03, daily_stop_slippage=0.002)
    rescued, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules)
    honest, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules, gap=g)
    assert rescued.outcome == pf.INCOMPLETE, "fixture no longer exercises the breaker"
    assert honest.outcome == pf.FAIL_DAILY


def test_the_breaker_still_works_when_the_damage_happens_in_hours():
    """A small gap followed by a bad session is exactly what a breaker is for."""
    r, w = _const(2, -0.06)
    g = np.full(2, -0.005)
    rules = replace(pf.FTMO_TWO_STEP, daily_stop=0.03)
    a, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules, gap=g)
    assert a.outcome == pf.INCOMPLETE


def test_the_gap_is_scaled_by_the_size_like_everything_else():
    """At 1x the gap is survivable and at 3x it is not."""
    r, w = _const(2, -0.015)
    g = np.full(2, -0.015)
    rules = replace(pf.FTMO_TWO_STEP, daily_stop=0.03)
    small, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules,
                            sizer=pf.flat_size(1.0), gap=g)
    big, _ = pf.run_phase(r, w, pf.Phase("c", 0.10), rules,
                          sizer=pf.flat_size(4.0), gap=g)
    assert small.outcome == pf.INCOMPLETE
    assert big.outcome == pf.FAIL_DAILY
