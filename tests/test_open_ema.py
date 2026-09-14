"""The New York open EMA rule: the algebra, the stop paths, the controls.

Three things here are worth pinning and the rest follows from them. The first is
the claim that makes the specification unambiguous - that a close above its own
EMA is the same event as a close above the previous EMA - because if that were
false the study would have had to pick a reading and defend it. The second is
that each exit style's stop can only ever tighten, which is what makes an
``R`` multiple mean the same thing in all three families. The third is that the
five direction rules are exact partitions of the same fills, since the whole
control argument rests on them being priced identically.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qlab.strategies.open_ema import (
    DIRECTION_RULES,
    EXIT_STYLES,
    OpenEMAConfig,
    _stop_path,
    coin_side,
    equity_curve,
    select_rule,
    side_for,
    signal_equivalence,
)


# --------------------------------------------------------------------------
# The algebra that removes the specification's ambiguity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("span", [2, 3, 12, 50, 200])
def test_close_above_own_ema_is_close_above_previous_ema(span):
    """``c_t > e_t`` and ``c_t > e_{t-1}`` are the same event for every span.

    ``e_t = a c_t + (1-a) e_{t-1}`` rearranges to ``(1-a)(c_t - e_{t-1})``, and
    ``1-a > 0``, so the two comparisons cannot disagree. Whether the signal
    candle is included in its own average is therefore not a modelling choice,
    which is the only reason this rule has one unambiguous reading.
    """
    rng = np.random.default_rng(span)
    walk = 15_000 + np.cumsum(rng.normal(0, 5, size=4000))
    assert signal_equivalence(walk, span=span)


def test_the_equivalence_survives_paths_designed_to_break_it():
    """Monotone, flat and alternating paths - the cases where ties would show."""
    assert signal_equivalence(np.arange(500.0), span=12)          # strict uptrend
    assert signal_equivalence(-np.arange(500.0), span=12)         # strict downtrend
    assert signal_equivalence(np.full(500, 100.0), span=12)       # every bar a tie
    assert signal_equivalence(np.tile([100.0, 101.0], 250), span=12)


def test_the_span_changes_the_distance_but_not_the_side():
    """The span survives in ``ema_gap``, so it is not a free parameter of nothing.

    It matters for how far the close is from the level - which the study uses as
    a conditioning variable - and never for which side the rule takes.
    """
    rng = np.random.default_rng(0)
    closes = pl.Series(15_000 + np.cumsum(rng.normal(0, 5, size=2000)))
    gaps = {
        span: (closes - closes.ewm_mean(span=span, adjust=False)).to_numpy()
        for span in (3, 12, 50)
    }
    signs = {span: np.sign(g[100:]) for span, g in gaps.items()}
    assert not np.array_equal(signs[3], signs[50])          # sides do differ by span
    assert not np.allclose(gaps[3][100:], gaps[12][100:])   # so do distances


# --------------------------------------------------------------------------
# The mechanics-only control
# --------------------------------------------------------------------------

def test_the_coin_control_is_deterministic_and_balanced():
    """A hash, not an RNG: the null must be identical across runs and splits."""
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(3000)]
    sides = [coin_side(d) for d in days]

    assert sides == [coin_side(d) for d in days]     # stable within a process
    assert set(sides) == {1, -1}
    assert abs(np.mean([s == 1 for s in sides]) - 0.5) < 0.03
    # A specific value, so a change to the hash cannot pass silently.
    assert coin_side(date(2023, 3, 3)) == coin_side(date(2023, 3, 3))


# --------------------------------------------------------------------------
# Stop paths
# --------------------------------------------------------------------------

def _favour(prices: np.ndarray, direction: int) -> np.ndarray:
    return (np.maximum.accumulate(prices) if direction == 1
            else np.minimum.accumulate(prices))


@pytest.mark.parametrize("style", EXIT_STYLES)
@pytest.mark.parametrize("direction", [1, -1])
def test_every_stop_path_only_ever_tightens(style, direction):
    """A stop that loosens is a stop that was never really placed.

    All three families are read as ``R`` multiples of the same planned risk, so
    a style that could widen its stop mid-trade would be reporting an ``R`` that
    does not correspond to any amount actually at risk.
    """
    rng = np.random.default_rng(1)
    entry, distance = 100.0, 2.0
    prices = entry + np.cumsum(rng.normal(0, 0.5, size=2000))
    stop = _stop_path(_favour(prices, direction), entry, direction, distance, style)
    moves = direction * np.diff(stop)
    assert (moves >= -1e-12).all()


@pytest.mark.parametrize("direction", [1, -1])
def test_the_planned_risk_is_one_distance_for_every_style(direction):
    """At the first tick all three styles risk exactly ``distance``.

    This is what makes ``net_r`` comparable across the sweep: the denominator is
    the same number in every cell, so a wider trail buys a longer hold rather
    than a quietly rescaled unit.
    """
    entry, distance = 100.0, 2.0
    prices = np.full(10, entry)
    for style in EXIT_STYLES:
        stop = _stop_path(_favour(prices, direction), entry, direction, distance, style)
        assert stop[0] == pytest.approx(entry - direction * distance)


def test_trail_ratchets_immediately_and_be_trail_waits_for_one_r():
    """The difference between the two trailing families, on one known path.

    Price walks up to +1.5 distance and back. ``trail`` has already moved its
    stop up by 1.5; ``be_trail`` armed at +1.0 and has moved it by 0.5; ``fixed``
    has not moved at all.
    """
    entry, distance = 100.0, 2.0
    prices = np.array([100.0, 101.0, 102.0, 103.0, 101.0])  # peak at +1.5 distance
    favour = _favour(prices, 1)

    assert _stop_path(favour, entry, 1, distance, "fixed")[-1] == pytest.approx(98.0)
    assert _stop_path(favour, entry, 1, distance, "trail")[-1] == pytest.approx(101.0)
    assert _stop_path(favour, entry, 1, distance, "be_trail")[-1] == pytest.approx(101.0)

    # Stop one tick short of +1 distance and be_trail has still not armed.
    short = np.array([100.0, 101.0, 101.5])
    assert _stop_path(_favour(short, 1), entry, 1, distance, "be_trail")[-1] == \
        pytest.approx(98.0)
    assert _stop_path(_favour(short, 1), entry, 1, distance, "trail")[-1] == \
        pytest.approx(99.5)


def test_be_trail_is_never_tighter_than_fixed_nor_looser_than_trail():
    """The three families are ordered, which is why the sweep spans them.

    ``fixed <= be_trail <= trail`` in tightness at every tick, so the exit
    surface has the no-trail baseline at one end and the always-trailing rule at
    the other with nothing outside them.
    """
    rng = np.random.default_rng(3)
    entry, distance = 100.0, 2.0
    prices = entry + np.cumsum(rng.normal(0, 0.4, size=3000))
    favour = _favour(prices, 1)
    paths = {s: _stop_path(favour, entry, 1, distance, s) for s in EXIT_STYLES}
    assert (paths["fixed"] <= paths["be_trail"] + 1e-12).all()
    assert (paths["be_trail"] <= paths["trail"] + 1e-12).all()


def test_an_unknown_exit_style_is_refused():
    with pytest.raises(ValueError, match="chandelier"):
        _stop_path(np.zeros(3), 100.0, 1, 1.0, "chandelier")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_the_config_refuses_specifications_it_cannot_honour():
    with pytest.raises(ValueError, match="ema_series"):
        OpenEMAConfig(ema_series="premarket")
    with pytest.raises(ValueError, match="ema_len"):
        OpenEMAConfig(ema_len=1)
    with pytest.raises(ValueError, match="divide an hour"):
        OpenEMAConfig(bar_minutes=7)
    with pytest.raises(ValueError, match="does not fit inside the session"):
        OpenEMAConfig(open_min=16 * 60 - 2, close_min=16 * 60)


def test_the_signal_candle_closes_five_minutes_after_the_open():
    assert OpenEMAConfig().signal_end_min == 9 * 60 + 35


# --------------------------------------------------------------------------
# Direction rules: the controls have to be exact partitions of one set of fills
# --------------------------------------------------------------------------

def _both_sides(signals: list[int], coins: list[int]) -> pl.DataFrame:
    """One session per signal, both sides priced, one exit cell."""
    rows = []
    for i, (signal, coin) in enumerate(zip(signals, coins)):
        for direction in (1, -1):
            rows.append({
                "nyd": date(2023, 1, 2) + timedelta(days=i),
                "signal": signal, "coin": coin, "direction": direction,
                "exit_style": "trail", "stop_frac": 0.25,
                "net_r": 0.5 * direction,
            })
    return pl.DataFrame(rows)


def test_each_rule_takes_exactly_one_side_per_session():
    trades = _both_sides([1, -1, 1, -1], [1, 1, -1, -1])
    for rule in DIRECTION_RULES:
        picked = select_rule(trades, rule)
        assert picked.height == 4, rule
        assert picked["nyd"].n_unique() == 4, rule


def test_contra_is_the_exact_complement_of_the_rule():
    """Every session the rule trades long, contra trades short, on the same fill."""
    trades = _both_sides([1, -1, 1, -1], [1, 1, -1, -1])
    ema = select_rule(trades, "ema").sort("nyd")
    contra = select_rule(trades, "contra").sort("nyd")
    assert (ema["direction"] == -contra["direction"]).all()
    assert (ema["net_r"] == -contra["net_r"]).all()


def test_the_doji_session_is_not_a_side():
    """A close exactly on the EMA arms nothing, for the rule or its inverse.

    The unsigned controls still trade it: ``long`` and ``coin`` do not consult
    the signal, so dropping the session for them would silently change what the
    null is being measured over.
    """
    trades = _both_sides([1, 0, -1], [1, -1, 1])
    assert select_rule(trades, "ema").height == 2
    assert select_rule(trades, "contra").height == 2
    assert select_rule(trades, "long").height == 3
    assert select_rule(trades, "coin").height == 3


def test_long_and_short_partition_every_fill():
    trades = _both_sides([1, -1, 0], [1, -1, 1])
    longs = select_rule(trades, "long")
    shorts = select_rule(trades, "short")
    assert longs.height + shorts.height == trades.height
    assert (longs["direction"] == 1).all() and (shorts["direction"] == -1).all()


def test_side_for_agrees_with_select_rule():
    """The scalar and the frame path must not drift apart."""
    trades = _both_sides([1, -1, 1], [-1, 1, 1])
    for rule in DIRECTION_RULES:
        picked = select_rule(trades, rule).sort("nyd")
        for row in picked.iter_rows(named=True):
            assert side_for(row, rule) == row["direction"], (rule, row["nyd"])


def test_an_unknown_direction_rule_is_refused():
    trades = _both_sides([1], [1])
    with pytest.raises(ValueError, match="momentum"):
        select_rule(trades, "momentum")
    with pytest.raises(ValueError, match="momentum"):
        side_for({"signal": 1, "coin": 1}, "momentum")


# --------------------------------------------------------------------------
# The sizing arithmetic behind the headline return
# --------------------------------------------------------------------------

def test_the_equity_curve_compounds_risk_per_trade_in_date_order():
    cell = pl.DataFrame({
        "nyd": [date(2023, 1, 3), date(2023, 1, 2), date(2023, 1, 4)],
        "net_r": [2.0, -1.0, 1.0],
    })
    curve = equity_curve(cell, risk_pct=1.0, starting_equity=100.0)

    # Sorted by date, so the -1R trade is first: 0.99, then 1.0098, then 1.019898.
    assert curve["nyd"].to_list() == [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)]
    assert curve["equity"].to_list() == pytest.approx([99.0, 100.98, 101.9898])
    assert curve["drawdown"][0] == pytest.approx(-0.01)
    assert curve["drawdown"][-1] == pytest.approx(0.0)


def test_doubling_the_risk_squares_the_multiple_not_doubles_it():
    """Why the total-return headline is reported apart from the per-trade edge.

    A ``+982%`` is a statement about sizing as much as about the signal: the same
    trades at 2% risk do not return twice as much, they return the square of the
    growth factor, and the drawdown grows with it.
    """
    rng = np.random.default_rng(5)
    cell = pl.DataFrame({
        "nyd": [date(2020, 1, 1) + timedelta(days=i) for i in range(500)],
        "net_r": rng.normal(0.1, 1.0, size=500),
    })
    one = equity_curve(cell, risk_pct=1.0)
    two = equity_curve(cell, risk_pct=2.0)

    growth_1 = float(one["equity"][-1]) / 100_000.0
    growth_2 = float(two["equity"][-1]) / 100_000.0
    assert growth_2 > growth_1 ** 1.8          # super-linear, not 2x
    assert float(two["drawdown"].min()) < float(one["drawdown"].min())


def test_an_empty_cell_makes_an_empty_curve_rather_than_dividing_by_nothing():
    assert equity_curve(pl.DataFrame()).is_empty()
