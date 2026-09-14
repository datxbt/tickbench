"""What the quadrant engulfing setup has to get right to be worth believing.

Four things are pinned here, and they are the four places this particular
strategy could silently lie:

1. **The pattern's two conditions**, including the ones adjacent to them that
   must *not* fire - a bar that equals the prior high, or closes exactly at the
   prior open, is not a signal.
2. **The fibonacci anchor**: 0 is the target extreme and 1 the stop extreme, on
   both sides of the market.
3. **Which side of the book each level acts on**, and that a resting limit fills
   at its price while a stop pays slippage.
4. **The resolution order** - that a signal whose price reached the target
   before ever returning to the entry zone is recorded as ``missed`` and not as
   a win. This is the one that matters most: it is the flattering mistake a
   retracement backtest makes, and it is worth more than the whole edge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.engulfing_quadrant import (
    ENTRY_MODES,
    STOP_FIBS,
    TARGET_FIBS,
    QuadrantConfig,
    _fib,
    breakeven_rate,
    fair_rate,
    payoff_ratio,
    _first_cross,
    _resolve,
    _Tape,
    sequence,
    sequence_many,
    signals,
)
from qlab.symbols import get_spec

US = 1_000_000
T0 = datetime(2022, 3, 1, 10, 0, tzinfo=timezone.utc)


def make_bars(rows: list[tuple[float, float, float, float]], *,
              minutes: int = 1) -> pl.DataFrame:
    """Contiguous right-edge-labelled bars from (open, high, low, close) tuples."""
    step = timedelta(minutes=minutes)
    return pl.DataFrame({
        "ts": [T0 + step * (i + 1) for i in range(len(rows))],
        "ts_open": [T0 + step * i for i in range(len(rows))],
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "spread_close": [0.0] * len(rows),
    })


def pad(rows: list[tuple[float, float, float, float]]) -> list:
    """Signal detection needs 20 bars of lookahead for the excursion columns."""
    return rows + [(100.0, 100.5, 99.5, 100.0)] * 24


# --------------------------------------------------------------------------
# 1. the pattern
# --------------------------------------------------------------------------

# prior bar (o, h, l, c), signal bar (o, h, l, c), is it a signal?
PATTERN_CASES = [
    # takes out the prior high (11 > 10) and closes below the prior open (8 < 9)
    ((9.0, 10.0, 8.5, 9.8), (9.5, 11.0, 7.5, 8.0), True),
    # never took out the prior high - condition 1 fails
    ((9.0, 10.0, 8.5, 9.8), (9.5, 9.9, 7.5, 8.0), False),
    # took the high but closed above the prior open - condition 2 fails
    ((9.0, 10.0, 8.5, 9.8), (9.5, 11.0, 7.5, 9.5), False),
    # equalling the prior high is not taking it out: strict inequality
    ((9.0, 10.0, 8.5, 9.8), (9.5, 10.0, 7.5, 8.0), False),
    # closing exactly at the prior open is not closing below it
    ((9.0, 10.0, 8.5, 9.8), (9.5, 11.0, 7.5, 9.0), False),
    # the prior bar being bearish is not disqualifying - the rules never said so
    ((9.8, 10.0, 8.5, 9.0), (9.5, 11.0, 7.5, 8.0), True),
]


@pytest.mark.parametrize("prev,sig,expected", PATTERN_CASES)
def test_pattern_conditions(prev, sig, expected):
    found = signals(make_bars(pad([prev, sig])), QuadrantConfig(), interval="1m")
    hit = found.filter(pl.col("arm_ts") == T0 + timedelta(minutes=2))
    assert (hit.height == 1) is expected


def test_prev_bull_filter_is_off_by_default():
    """A bearish prior candle still signals unless the filter is switched on."""
    bars = make_bars(pad([(9.8, 10.0, 8.5, 9.0), (9.5, 11.0, 7.5, 8.0)]))
    assert signals(bars, QuadrantConfig()).height == 1
    assert signals(bars, QuadrantConfig(require_prev_bull=True)).height == 0


def test_gap_is_not_a_stop_run():
    """Bars that are adjacent by row but not by clock are dropped."""
    bars = make_bars(pad([(9.0, 10.0, 8.5, 9.8), (9.5, 11.0, 7.5, 8.0)]))
    # push the signal bar an hour later without changing its prices
    bars = bars.with_columns(
        ts_open=pl.when(pl.col("ts_open") == T0 + timedelta(minutes=1))
        .then(pl.col("ts_open") + timedelta(hours=1))
        .otherwise(pl.col("ts_open"))
    )
    assert signals(bars, QuadrantConfig()).height == 0


# A four-bar motif carrying exactly one signal and two no-sweep candidates.
#   bar 0: the prior bar
#   bar 1: SIGNAL - takes out bar 0's high (11.0 > 10.0), closes below its open
#   bar 2: closes below bar 1's open without taking bar 1's high - a candidate
#   bar 3: same again against bar 2 - a candidate
MOTIF = [
    (9.0, 10.0, 8.5, 9.8),
    (9.5, 11.0, 7.5, 8.0),
    (8.0, 8.60, 7.80, 8.50),
    (8.5, 8.55, 7.90, 7.95),
]


def motif_bars(repeats: int = 12) -> pl.DataFrame:
    return make_bars(pad(MOTIF * repeats))


def test_nosweep_control_keeps_condition_two_only():
    """The sharp null: closes below the prior open, never took the prior high."""
    bars = motif_bars()
    found = signals(bars, QuadrantConfig(control="nosweep"))
    assert found.height > 0
    # every drawn bar closed past the prior open and did NOT take its high
    assert (found["close_p"] < found["prev_open_p"]).all()
    assert (found["high_p"] <= found["prev_high_p"]).all()


@pytest.mark.parametrize("control", ["matched", "nosweep"])
def test_controls_are_size_matched_to_the_signal(control):
    """Neither null may outgrow the signal set.

    This is the invariant that keeps a null affordable. The no-sweep condition
    matches several times as many bars as the signal does, and drawing all of
    them buys precision nobody needs at a cost that scales with the timeframe
    ladder.

    The draw is a hash threshold rather than an exact-count sample, so the size
    lands binomially around the signal count rather than on it. What has to hold
    is the order of magnitude: the null is O(signals), not O(candidates).
    """
    bars = motif_bars(60)
    n_sig = signals(bars, QuadrantConfig()).height
    assert n_sig > 0
    n_null = signals(bars, QuadrantConfig(control=control)).height
    assert 0.5 * n_sig <= n_null <= 1.5 * n_sig


@pytest.mark.parametrize("control", ["matched", "nosweep"])
def test_controls_are_deterministic(control):
    """A control that moves between runs cannot settle an argument."""
    bars = motif_bars()
    cfg = QuadrantConfig(control=control)
    first = signals(bars, cfg)["arm_ts"].to_list()
    assert first == signals(bars, cfg)["arm_ts"].to_list()


def test_controls_never_draw_the_signal_bars_themselves():
    """The matched null must be disjoint from the signal set."""
    bars = motif_bars()
    sig_ts = set(signals(bars, QuadrantConfig())["arm_ts"].to_list())
    null_ts = set(signals(bars, QuadrantConfig(control="matched"))["arm_ts"].to_list())
    assert sig_ts and null_ts
    assert not (sig_ts & null_ts)


def test_mirror_is_the_exact_reflection():
    """The mirror fires on the reflected bars and takes the other side."""
    prev = (9.0, 10.0, 8.5, 9.8)
    sig = (9.5, 11.0, 7.5, 8.0)
    flip = lambda b: (20 - b[0], 20 - b[2], 20 - b[1], 20 - b[3])  # noqa: E731
    found = signals(make_bars(pad([flip(prev), flip(sig)])),
                    QuadrantConfig(mirror=True), interval="1m")
    hit = found.filter(pl.col("arm_ts") == T0 + timedelta(minutes=2))
    assert hit.height == 1
    assert int(hit["direction"][0]) == 1


# --------------------------------------------------------------------------
# 2. the fibonacci anchor
# --------------------------------------------------------------------------

def test_fib_anchors_zero_at_the_target_extreme():
    short = {"direction": -1, "low_p": 8.0, "high_p": 12.0}
    assert _fib(short, 0.0) == 8.0     # the target: the bar's low
    assert _fib(short, 1.0) == 12.0    # the stop: the bar's high
    assert _fib(short, 0.25) == 9.0
    assert _fib(short, 0.5) == 10.0
    # and below the low for the extension targets
    assert _fib(short, -0.25) == 7.0

    long = {"direction": 1, "low_p": 8.0, "high_p": 12.0}
    assert _fib(long, 0.0) == 12.0     # the target: the bar's high
    assert _fib(long, 1.0) == 8.0      # the stop: the bar's low
    assert _fib(long, 0.25) == 11.0


def test_quadrants_are_equal_quarters():
    sig = {"direction": -1, "low_p": 8.0, "high_p": 12.0}
    levels = [_fib(sig, f) for f in (0, 0.25, 0.5, 0.75, 1.0)]
    steps = np.diff(levels)
    assert np.allclose(steps, steps[0])


# --------------------------------------------------------------------------
# 2b. the arithmetic that decides the study
# --------------------------------------------------------------------------

def test_breakeven_equals_fair_coin_at_zero_cost():
    """The identity the whole rejection rests on.

    For any (entry, stop, target), the strike rate needed to break even with no
    costs is *exactly* the strike rate a driftless walk delivers. The trade is
    a fair bet by construction whatever levels are chosen, so no tuning of the
    entry, the stop, the target or the timeframe can produce an edge - costs can
    only move it below zero.
    """
    for entry in (0.25, 0.3125, 0.375, 0.5, 0.7):
        for stop in (0.75, 1.0, 1.25):
            for target in (0.0, -0.25, -0.5, -1.0):
                if not target < entry < stop:
                    continue
                rr = payoff_ratio(entry, stop, target)
                assert breakeven_rate(rr, 0.0) == pytest.approx(
                    fair_rate(entry, stop, target), abs=1e-12)


def test_any_cost_pushes_breakeven_above_the_fair_coin():
    """And with a real round turn the hurdle is strictly above what a coin pays."""
    for cost in (0.01, 0.1, 0.5):
        rr = payoff_ratio(0.334, 1.0, 0.0)
        assert breakeven_rate(rr, cost) > fair_rate(0.334, 1.0, 0.0)


def test_payoff_ratio_matches_the_stated_geometry():
    """Entry at 0.25 with the stop at the bar's high is 1:3; at 0.5 it is 1:1."""
    assert payoff_ratio(0.25, 1.0, 0.0) == pytest.approx(1 / 3)
    assert payoff_ratio(0.5, 1.0, 0.0) == pytest.approx(1.0)
    # the tight stop is the only one that pays better than even money
    assert payoff_ratio(0.5, 0.75, 0.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 3. first passage
# --------------------------------------------------------------------------

def test_first_cross_finds_the_first_and_respects_bounds():
    arr = np.array([5.0, 4.0, 3.0, 2.0, 3.0, 1.0])
    assert _first_cross(arr, 0, 6, 3.0, below=True) == 2
    assert _first_cross(arr, 3, 6, 1.5, below=True) == 5
    assert _first_cross(arr, 0, 3, 1.5, below=True) == -1
    assert _first_cross(arr, 0, 6, 4.5, below=False) == 0
    assert _first_cross(arr, 4, 4, 0.0, below=True) == -1


def test_first_cross_matches_a_naive_scan_on_random_data():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=20_000).cumsum()
    for _ in range(50):
        lo = int(rng.integers(0, 19_000))
        level = float(arr[lo] + rng.normal())
        for below in (True, False):
            mask = (arr[lo:] <= level) if below else (arr[lo:] >= level)
            want = lo + int(mask.argmax()) if mask.any() else -1
            assert _first_cross(arr, lo, arr.size, level, below) == want


# --------------------------------------------------------------------------
# 4. the tape walk: fills, misses and which side of the book
# --------------------------------------------------------------------------

def tape_from(mids: list[float], *, spread: float = 0.02,
              start: datetime = T0 + timedelta(minutes=2)) -> _Tape:
    """A synthetic tape of one quote per second around the given mids."""
    ts = np.array([int((start + timedelta(seconds=i)).timestamp() * US)
                   for i in range(len(mids))], dtype=np.int64)
    mid = np.array(mids, dtype=float)
    return _Tape(ts=ts, bid=mid - spread / 2, ask=mid + spread / 2)


def base_signal(**over) -> dict:
    """A short signal on a bar with low 8, high 12, closing at 9 (fib 0.25)."""
    sig = {
        "interval": "1m",
        "arm_ts": int((T0 + timedelta(minutes=2)).timestamp() * US),
        "direction": -1,
        "open_p": 11.5, "high_p": 12.0, "low_p": 8.0, "close_p": 9.0,
        "prev_open_p": 11.0, "prev_high_p": 11.8, "prev_low_p": 10.0,
        "prev_close_p": 11.5, "spread_close_p": 0.02,
        "bar_range": 4.0, "close_fib": 0.25, "sweep_depth": 0.05,
        "give_back": 0.5,
        "fwd1_bps": 0.0, "fwd5_bps": 0.0, "fwd20_bps": 0.0,
        "mfe20_rng": 0.0, "mae20_rng": 0.0,
    }
    sig.update(over)
    return sig


def resolve(sig: dict, mids: list[float], *, cfg=None, slip: float = 0.0,
            spread: float = 0.02) -> dict[str, dict]:
    cfg = cfg or QuadrantConfig()
    rows = _resolve(tape_from(mids, spread=spread), sig, cfg, get_spec("USTEC"),
                    {h: slip for h in range(24)}, 0.0, 60 * US)
    return {r["entry_mode"]: r for r in rows}


def test_missed_beats_the_target_and_is_not_a_win():
    """The whole point. Price runs to the low without ever revisiting the zone.

    The bar closed at 9.0, which is the 0.25 level, so the zone order sits at
    the close and fills immediately - but the q50 order at 10.0 never does, and
    price reaches the target at 8.0. If the target were tested before the
    entry, this would be booked as a winner.
    """
    out = resolve(base_signal(close_p=8.6), [8.6, 8.4, 8.2, 7.9, 7.8])
    assert out["q50"]["fill_reason"] == "missed"
    assert out["q50"]["filled"] is False
    # and nothing was recorded for it that could be mistaken for a result
    assert "r_s100_t0" not in out["q50"]


def test_invalidated_when_price_goes_back_over_the_bar_high():
    """The bar closed above the zone, so the order is a sell stop *below* the
    market - and price ran up through the bar's high before ever coming down to
    it. A limit resting *above* the close cannot be invalidated this way: it
    sits between the close and the high and fills on the way through.
    """
    out = resolve(base_signal(close_p=11.0), [11.0, 11.5, 12.2, 10.5, 9.0])
    assert out["q50"]["fill_reason"] == "invalidated"
    assert out["q50"]["filled"] is False


def test_expired_when_the_zone_is_never_reached_in_time():
    cfg = QuadrantConfig(valid_bars=1)   # one 1m bar = 60 quotes here
    out = resolve(base_signal(close_p=8.6), [8.6] * 40, cfg=cfg)
    assert out["q50"]["fill_reason"] == "expired"


def test_limit_entry_fills_at_its_price_and_pays_no_slippage():
    """A sell limit at 10.0 fills when the *bid* reaches it, at exactly 10.0."""
    out = resolve(base_signal(close_p=8.6), [8.6, 9.5, 10.05, 9.0, 8.0],
                  slip=0.5)
    row = out["q50"]
    assert row["fill_reason"] == "filled"
    assert row["order_type"] == "limit"
    assert row["entry"] == pytest.approx(10.0)
    assert row["slip_entry"] == 0.0


def test_limit_needs_the_bid_not_the_mid_to_reach_it():
    """Mid 10.0 with a 0.02 spread means a bid of 9.99: no fill."""
    filled = resolve(base_signal(close_p=8.6), [8.6, 9.5, 10.0, 9.0, 8.0])
    assert filled["q50"]["fill_reason"] != "filled"
    # nudge the mid up by half a spread and it does fill
    ok = resolve(base_signal(close_p=8.6), [8.6, 9.5, 10.01, 9.0, 8.0])
    assert ok["q50"]["fill_reason"] == "filled"


def test_stop_entry_pays_slippage_adversely():
    """Closing above the zone, the order is a sell stop and fills at market."""
    out = resolve(base_signal(close_p=11.0), [11.0, 10.5, 9.9, 9.0, 8.0],
                  slip=0.1)
    row = out["q50"]
    assert row["order_type"] == "stop"
    assert row["slip_entry"] == pytest.approx(0.1)
    # a short sells lower than it meant to
    assert row["entry"] < 10.0


def test_zone_order_fills_at_the_edge_price_reaches_first():
    """Below the zone it is the 0.25 edge; above it, the 0.5 edge."""
    below = resolve(base_signal(close_p=8.5), [8.5, 9.02, 9.5, 8.0])
    assert below["zone"]["entry"] == pytest.approx(9.0)
    above = resolve(base_signal(close_p=11.0), [11.0, 10.5, 9.9, 8.0])
    assert above["zone"]["entry_level"] == pytest.approx(10.0)
    # and inside the zone it is a market order at the close
    inside = resolve(base_signal(close_p=9.5), [9.5, 9.0, 8.0])
    assert inside["zone"]["immediate"] is True


def test_target_is_reached_on_the_ask_and_stop_triggered_on_the_ask():
    """A short covers at the ask, for both barriers. Mid 8.0 is not enough."""
    near = resolve(base_signal(close_p=9.5), [9.5, 8.0, 8.0, 8.0])
    assert near["zone"]["reason_s100_t0"] != "tp"      # ask is 8.01
    hit = resolve(base_signal(close_p=9.5), [9.5, 7.99, 7.99, 7.99])
    assert hit["zone"]["reason_s100_t0"] == "tp"


def test_the_earlier_barrier_wins():
    """Price hits the stop and only then the target: the stop is the outcome.

    Both barriers are read on the ask for a short, so a single quote can never
    satisfy the two at once and the tie-break is defensive rather than load
    bearing. What does happen constantly is this: a bar's range spans both
    levels, and only the tick order says which came first.
    """
    out = resolve(base_signal(close_p=9.5), [9.5, 12.1, 7.9])["zone"]
    assert out["reason_s100_t0"] == "stop"
    # the tighter stop was hit even earlier; the wider one not at all
    assert out["reason_s75_t0"] == "stop"
    assert out["reason_s125_t0"] == "tp"


def test_r_is_measured_against_the_stop_that_produced_it():
    """Each stop has its own risk, so the same win is a different R."""
    out = resolve(base_signal(close_p=9.5), [9.5, 7.9, 7.9])["zone"]
    entry = out["entry"]
    for skey, fib in STOP_FIBS.items():
        assert out[f"risk_{skey}"] == pytest.approx(abs(entry - (8.0 + fib * 4.0)))
    # tighter stop, larger R for the same money
    assert out["r_s75_t0"] > out["r_s100_t0"] > out["r_s125_t0"]


def test_deeper_targets_are_reached_later_or_not_at_all():
    out = resolve(base_signal(close_p=9.5), [9.5, 7.9, 7.4, 7.4])["zone"]
    order = ["t0", "tm25", "tm50", "tm100"]
    seen = [out[f"exit_ts_s100_{t}"] for t in order]
    assert seen == sorted(seen)
    # the 8.0 target is reached, the 4.0 one is not
    assert out["reason_s100_t0"] == "tp"
    assert out["reason_s100_tm100"] != "tp"


def test_hold_never_takes_a_target():
    out = resolve(base_signal(close_p=9.5), [9.5, 7.9, 7.9, 7.9])["zone"]
    assert out["reason_s100_t0"] == "tp"
    assert out["reason_s100_hold"] in ("stop", "time")


def test_every_entry_mode_is_priced_for_every_signal():
    out = resolve(base_signal(close_p=9.5), [9.5, 9.0, 8.0])
    assert set(out) == set(ENTRY_MODES)


# --------------------------------------------------------------------------
# 5. the account
# --------------------------------------------------------------------------

def test_sequence_takes_one_position_at_a_time():
    """Signals arriving while a position is open are not taken."""
    trades = pl.DataFrame({
        "interval": ["1m"] * 4,
        "entry_mode": ["zone"] * 4,
        "filled": [True] * 4,
        "fill_ts": [100, 150, 260, 400],
        "exit_ts_s100_t0": [250, 300, 350, 500],
    })
    taken = sequence(trades, "s100_t0")
    # 100 runs to 250, so 150 is skipped; 260 is free; 400 is free
    assert taken["fill_ts"].to_list() == [100, 260, 400]


def test_sequence_ignores_unfilled_signals():
    trades = pl.DataFrame({
        "interval": ["1m"] * 3,
        "entry_mode": ["zone"] * 3,
        "filled": [False, True, False],
        "fill_ts": [0, 150, 0],
        "exit_ts_s100_t0": [0, 300, 0],
    })
    assert sequence(trades, "s100_t0")["fill_ts"].to_list() == [150]


def reference_sequence(trades: pl.DataFrame, exit_key: str) -> list[int]:
    """A deliberately naive stateful walk, to check the vectorised one against.

    Holds one position, scans signals in fill order, and skips any that arrives
    while the previous one is still open. Obviously correct and obviously slow.
    """
    rows = (trades.filter(pl.col("filled")).sort("fill_ts")
            .select("fill_ts", f"exit_ts_{exit_key}").rows())
    taken, busy_until = [], None
    for fill_ts, exit_ts in rows:
        if busy_until is not None and fill_ts < busy_until:
            continue
        taken.append(fill_ts)
        busy_until = max(exit_ts, fill_ts + 1)
    return taken


def test_sequence_many_matches_a_naive_stateful_walk():
    """The batched implementation must agree with the obvious one, per key."""
    rng = np.random.default_rng(7)
    n = 400
    fill = np.sort(rng.integers(0, 100_000, size=n))
    frame = pl.DataFrame({
        "interval": ["5m"] * n,
        "entry_mode": ["zone"] * n,
        "filled": rng.random(n) > 0.3,
        "fill_ts": fill,
        "exit_ts_a": fill + rng.integers(0, 900, size=n),
        "exit_ts_b": fill + rng.integers(0, 4000, size=n),
    })
    out = sequence_many(frame, ["a", "b"])
    for key in ("a", "b"):
        assert out[key]["fill_ts"].to_list() == reference_sequence(frame, key)


def test_sequence_many_agrees_with_sequence_one_key_at_a_time():
    """Batching must not leak state between exit keys."""
    rng = np.random.default_rng(11)
    n = 200
    fill = np.sort(rng.integers(0, 50_000, size=n))
    frame = pl.DataFrame({
        "interval": rng.choice(["5m", "1h"], size=n).tolist(),
        "entry_mode": rng.choice(["zone", "q25"], size=n).tolist(),
        "filled": [True] * n,
        "fill_ts": fill,
        "exit_ts_a": fill + rng.integers(1, 700, size=n),
        "exit_ts_b": fill + rng.integers(1, 700, size=n),
    })
    batched = sequence_many(frame, ["a", "b"])
    for key in ("a", "b"):
        alone = sequence(frame, key)
        assert batched[key]["fill_ts"].to_list() == alone["fill_ts"].to_list()


def test_sequence_keeps_timeframes_and_modes_as_separate_accounts():
    trades = pl.DataFrame({
        "interval": ["1m", "5m"],
        "entry_mode": ["zone", "zone"],
        "filled": [True, True],
        "fill_ts": [100, 110],
        "exit_ts_s100_t0": [500, 500],
    })
    assert sequence(trades, "s100_t0").height == 2
