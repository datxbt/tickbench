"""What the sweep / order-block study has to get right to be worth believing.

1. **The indicator, replayed.** A sweep is a level *taken* and *closed back
   through* on the same bar; taken-and-held retires the level silently. Block
   zones are built from the bars after the last opposite swing, one ATR deep,
   and only the newest two per side are live.
2. **The day.** D1 and H4 are cut at 17:00 New York in both daylight regimes.
3. **The fills.** Market entries take the standing quote plus slippage; a
   resting limit needs the ask (long) to reach it; a tick satisfying stop and
   target together is a stop.
4. **The account** takes one position at a time.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.sweep_orderblock import (
    ToolkitConfig,
    _resolve,
    _Tape,
    atr_rma,
    block_lifetimes,
    build_bars,
    fair_rate,
    pivot_centres,
    roll_instants,
    sequence_many,
    structure,
    sweep_events,
)

US = 1_000_000


# --------------------------------------------------------------------------
# 1. the indicator
# --------------------------------------------------------------------------

def test_pivot_needs_strictly_higher_left_and_at_least_right():
    h = np.array([1, 2, 5, 2, 1, 1, 1], float)
    assert pivot_centres(h, 2, True).tolist() == [2]
    # a tie on the left kills it, a tie on the right does not
    assert pivot_centres(np.array([5, 2, 5, 2, 1], float), 2, True).tolist() == []
    assert pivot_centres(np.array([1, 2, 5, 5, 1], float), 2, True).tolist() == [2]


def sweep_case(close_on_sweep: float):
    #            0  1  2  3  4  5  6(sweep)  7
    h = np.array([1, 2, 5, 2, 1, 1, 6.0, 7.0])
    l = h - 0.5
    c = h - 0.2
    c[6] = close_on_sweep
    c[7] = 4.0
    return sweep_events(h, l, c, L=2)


def test_sweep_is_taken_and_closed_back_on_the_same_bar():
    ev = [e for e in sweep_case(4.0) if e[1] == -1]
    assert ev == [(6, -1, 5.0, 1, 4)]


def test_taken_and_held_retires_the_level_without_a_sweep():
    """Closing above the pivot on the taking bar is a breakout, and the level is
    gone - the next bar's poke above it is not a second chance."""
    assert [e for e in sweep_case(5.5) if e[1] == -1] == []


def test_only_seven_levels_live_per_side():
    """An eighth pivot pushes the oldest level out untested."""
    L = 1
    # eight isolated pivot highs of *falling* height, so none takes the one
    # before it, then one bar that takes every level still live
    h = [1.0]
    for k in range(8):
        h += [17.0 - k, 1.0]
    h += [100.0]
    h = np.array(h)
    l = np.full(h.size, 0.5)
    c = np.full(h.size, 0.8)
    ev = sweep_events(h, l, c, L=L, max_levels=7)
    last = [e for e in ev if e[0] == h.size - 1]
    assert last and last[0][3] == 7


def test_block_zone_is_built_from_the_bars_after_the_swing():
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 0.3, 4000).cumsum()
    h = c + rng.uniform(0.05, 0.4, c.size)
    l = c - rng.uniform(0.05, 0.4, c.size)
    atr = atr_rma(h, l, c, 14)
    st = structure(h, l, c, atr, 9)
    assert len(st.blocks) > 20
    last = None
    for b in st.blocks:
        t, s = b["create"], b["swing"]
        if b["direction"] == 1:
            assert c[t] > h[s]                       # a close above the swing high
            assert b["value"] == pytest.approx(l[s + 1:t + 1].min())
            assert b["top"] - b["bottom"] == pytest.approx(atr[t])
        else:
            assert c[t] < l[s]
            assert b["value"] == pytest.approx(h[s + 1:t + 1].max())
            assert b["top"] == pytest.approx(b["value"])
        # CHoCH is exactly "the previous break went the other way"
        want = "CHoCH" if last in (None, -b["direction"]) else "BoS"
        assert b["kind"] == want
        last = b["direction"]


def block(create, value, d=1):
    return dict(create=create, direction=d, value=value, atr=1.0,
                top=value + 1, bottom=value, swing=0, origin=0, kind="BoS")


def test_newest_two_are_live_and_the_third_is_hidden():
    bl = [block(1, 10.0), block(2, 11.0), block(3, 12.0)]
    block_lifetimes(bl, np.full(50, 20.0), show=2, valid_bars=30)
    assert (bl[0]["cancel"], bl[0]["cancel_reason"]) == (3, "hidden")
    assert bl[1]["cancel_reason"] == bl[2]["cancel_reason"] == "expired"


def test_a_close_through_the_far_edge_deletes_a_block():
    c = np.full(20, 20.0)
    c[5] = 9.5
    bl = [block(1, 10.0)]
    block_lifetimes(bl, c, show=2, valid_bars=30)
    assert (bl[0]["cancel"], bl[0]["cancel_reason"]) == (5, "invalidated")


def test_matched_controls_are_deterministic_and_size_matched():
    from qlab.strategies.sweep_orderblock import signals
    rng = np.random.default_rng(5)
    c = 100 + rng.normal(0, 0.3, 6000).cumsum()
    ts = [datetime(2022, 1, 3, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(c.size)]
    bars = pl.DataFrame({"ts": ts, "high": c + 0.2, "low": c - 0.2, "close": c})
    for fam in ("sweep", "ob"):
        real = signals(bars, fam, ToolkitConfig())
        a = signals(bars, fam, ToolkitConfig(control="matched"))
        b = signals(bars, fam, ToolkitConfig(control="matched"))
        assert a["arm_us"].to_list() == b["arm_us"].to_list()
        assert 0.5 * real.height <= a.height <= 1.5 * real.height
        assert not set(a["arm_us"].to_list()) & set(real["arm_us"].to_list()) or fam == "ob"


def test_fair_rate():
    assert fair_rate(1.0) == 0.5
    assert fair_rate(2.0) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# 2. the day
# --------------------------------------------------------------------------

def one_minute(start: datetime, n: int) -> pl.DataFrame:
    ts_open = [start + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame({
        "ts_open": ts_open, "ts": [t + timedelta(minutes=1) for t in ts_open],
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "n_ticks": 1,
    })


@pytest.mark.parametrize("day,roll_utc", [(date(2022, 7, 12), 21), (date(2022, 1, 12), 22)])
def test_d1_and_h4_are_cut_at_17_new_york(day, roll_utc):
    start = datetime(day.year, day.month, day.day, roll_utc - 1, tzinfo=timezone.utc)
    base = one_minute(start, 120)   # an hour either side of the roll
    d1 = build_bars(base, "1d")
    assert d1.height == 2
    assert d1["ts"][0].hour == roll_utc and d1["ts"][0].minute == 0
    h4 = build_bars(base, "4h")
    assert h4["ts"][0].hour == roll_utc


def test_roll_instants_skip_weekends():
    r = roll_instants(date(2022, 7, 8), date(2022, 7, 11))   # Fri .. Mon
    got = [datetime.fromtimestamp(x / US, tz=timezone.utc) for x in r]
    assert [g.weekday() for g in got] == [4, 0]
    assert all(g.hour == 21 for g in got)


# --------------------------------------------------------------------------
# 3. the fills
# --------------------------------------------------------------------------

T0 = int(datetime(2022, 3, 1, 10, tzinfo=timezone.utc).timestamp() * US)


def tape(mids, spread=0.02, start=T0 - 5 * US):
    ts = np.array([start + i * US for i in range(len(mids))], np.int64)
    m = np.array(mids, float)
    return _Tape(ts=ts, bid=m - spread / 2, ask=m + spread / 2)


def bar_ts():
    return np.array([T0 + i * 60 * US for i in range(200)], np.int64)


def sig(**over):
    s = dict(interval="1m", family="sweep", arm_idx=0, arm_us=T0, direction=-1,
             kind="market", entry_level=None, stop0=11.0, stop1=11.5, atr=2.0,
             bar_high=11.0, bar_low=9.0, close=10.0, cancel_us=None,
             cancel_reason=None)
    s.update(over)
    return s


def resolve(s, mids, slip=0.0):
    return _resolve(tape(mids), s, bar_ts(), ToolkitConfig(), slip, 0.0, 1.0,
                    np.empty(0, np.int64))


def test_market_entry_takes_the_standing_quote_and_pays_slippage():
    # quotes at T0-5s .. ; the one at T0 is index 5 and stands at the arm
    mids = [10.0] * 5 + [10.0, 9.0, 8.0, 7.0]
    row = resolve(sig(), mids, slip=0.1)
    assert row["order_type"] == "market"
    assert row["entry"] == pytest.approx(10.0 - 0.01 - 0.1)   # bid minus slippage


def test_stop_and_target_on_the_same_tick_is_a_stop():
    """Gap straight through both: the stop is credited, never the target."""
    row = resolve(sig(stop0=10.5), [10.0] * 6 + [12.0])
    assert row["reason_s0_t1"] == "stop"


def test_short_target_needs_the_ask():
    # entry at bid 9.99, s0 risk 1.01, 1R target 8.98 - reached on the ask only
    row = resolve(sig(), [10.0] * 6 + [8.99, 8.99])
    assert row["reason_s0_t1"] != "tp"
    row = resolve(sig(), [10.0] * 6 + [8.96, 8.96])
    assert row["reason_s0_t1"] == "tp"
    assert row["r_s0_t1"] == pytest.approx(1.0)


def test_limit_block_fills_only_when_the_ask_reaches_it():
    s = sig(family="ob", kind="limit", direction=1, entry_level=9.0,
            stop0=8.0, stop1=7.5, cancel_us=T0 + 100 * 60 * US, cancel_reason="hidden")
    miss = resolve(s, [10.0] * 6 + [9.005, 9.5])
    assert miss["filled"] is False and miss["fill_reason"] == "hidden"
    hit = resolve(s, [10.0] * 6 + [8.99, 9.5, 11.0])
    assert hit["filled"] and hit["order_type"] == "limit"
    assert hit["entry"] == 9.0


def test_block_already_in_its_zone_is_a_market_order():
    s = sig(family="ob", kind="limit", direction=1, entry_level=10.5,
            stop0=9.0, stop1=8.5, cancel_us=T0 + 100 * 60 * US, cancel_reason="hidden")
    row = resolve(s, [10.0] * 8)
    assert row["order_type"] == "market"


def test_order_cancelled_before_the_fill_is_not_filled():
    s = sig(family="ob", kind="limit", direction=1, entry_level=9.0,
            stop0=8.0, stop1=7.5, cancel_us=T0 + 2 * US, cancel_reason="invalidated")
    row = resolve(s, [10.0] * 6 + [9.5, 9.4, 8.5])
    assert row["filled"] is False and row["fill_reason"] == "invalidated"


def test_opposite_target_is_the_nearest_shown_block_beyond_the_entry():
    from qlab.strategies.sweep_orderblock import _opposite_target
    # two bearish blocks shown above a long entered at 10: bottoms 12 and 11
    bl = [block(1, 13.0, d=-1), block(2, 12.0, d=-1)]
    for b in bl:
        b["top"], b["bottom"] = b["value"], b["value"] - 1.0
    snap = block_lifetimes(bl, np.full(10, 10.5), snapshot=True, valid_bars=30)
    assert _opposite_target(snap, 5, 1, 10.0) == (11.0, 12.0)
    # nothing shown beyond an entry above both
    assert _opposite_target(snap, 5, 1, 12.5) == (None, None)
    # before either exists, nothing
    assert _opposite_target(snap, 0, 1, 10.0) == (None, None)


def test_snapshot_shows_only_the_newest_two():
    bl = [block(1, 10.0), block(2, 11.0), block(3, 12.0)]
    snap = block_lifetimes(bl, np.full(10, 20.0), snapshot=True, valid_bars=30)
    # bullish near edge is the top: value + 1
    assert sorted(snap["bull_near"][4].tolist()) == [12.0, 13.0]


def test_opposite_block_target_is_priced_as_a_limit():
    s = sig(family="ob", kind="limit", direction=1, entry_level=9.0, stop0=8.0,
            stop1=7.5, cancel_us=T0 + 100 * 60 * US, cancel_reason="hidden",
            tp_near=10.5, tp_far=11.0)
    row = resolve(s, [10.0] * 6 + [8.99, 9.8, 10.52, 10.6])
    assert row["reason_s0_obn"] == "tp"            # bid 10.51 reaches 10.5
    assert row["r_s0_obn"] == pytest.approx(1.5)   # 1.5 away on a 1.0 risk
    assert row["tpr_s0_obn"] == pytest.approx(1.5)
    assert row["reason_s0_obf"] != "tp"            # 11.0 never reached


def test_no_opposite_block_means_no_trade_under_that_exit():
    s = sig(family="ob", kind="limit", direction=1, entry_level=9.0, stop0=8.0,
            stop1=7.5, cancel_us=T0 + 100 * 60 * US, cancel_reason="hidden",
            tp_near=None, tp_far=None)
    row = resolve(s, [10.0] * 6 + [8.99, 9.8, 10.6])
    assert row["filled"] and "r_s0_obn" not in row and "r_s0_t2" in row


def test_rows_without_an_exit_neither_trade_nor_block_the_account():
    t = pl.DataFrame({
        "interval": ["1m"] * 3, "family": ["ob"] * 3, "filled": [True] * 3,
        "fill_us": [100, 150, 260], "exit_us_a": [None, 300, 350],
    })
    assert sequence_many(t, ["a"])["a"]["fill_us"].to_list() == [150]


# --------------------------------------------------------------------------
# 4. the account
# --------------------------------------------------------------------------

def test_one_position_at_a_time_per_interval_and_family():
    t = pl.DataFrame({
        "interval": ["1m"] * 3 + ["5m"], "family": ["sweep"] * 4,
        "filled": [True] * 4, "fill_us": [100, 150, 260, 120],
        "exit_us_a": [250, 300, 350, 500],
    })
    got = sequence_many(t, ["a"])["a"]
    assert sorted(got["fill_us"].to_list()) == [100, 120, 260]


# --------------------------------------------------------------------------
# 5. Addendum B - the opposite trade
# --------------------------------------------------------------------------

def test_short_stop_entry_triggers_on_the_bid_and_pays_slippage():
    s = sig(family="ob", kind="stop", direction=-1, entry_level=9.0,
            stop0=11.0, stop1=11.5, cancel_us=T0 + 100 * 60 * US, cancel_reason="hidden")
    miss = resolve(s, [10.0] * 6 + [9.02, 9.5])      # bid 9.01 never reaches 9.0
    assert miss["filled"] is False
    hit = resolve(s, [10.0] * 6 + [9.0, 8.5], slip=0.1)
    assert hit["order_type"] == "stop"
    assert hit["entry"] == pytest.approx(8.99 - 0.1)  # the triggering bid, less slippage


def test_mirror_swaps_the_original_stop_and_target():
    # The original: short a sweep at close 10.0, wick stop 11.0, 2R target 8.0.
    # The mirror is long: stop at 8.0, target at 11.0.
    s = sig(direction=1, stop0=11.0, stop1=11.5)
    cfg = ToolkitConfig(fade="mirror")
    none = np.empty(0, np.int64)
    row = _resolve(tape([10.0] * 6 + [10.5, 11.02]), s, bar_ts(), cfg, 0.0, 0.0, 1.0, none)
    entry = 10.01                                     # the ask
    assert row["entry"] == pytest.approx(entry)
    assert row["reason_s0_t2"] == "tp"                # bid 11.01 reaches 11.0
    assert row["risk_s0_t2"] == pytest.approx(entry - 8.0)
    assert row["r_s0_t2"] == pytest.approx((11.0 - entry) / (entry - 8.0))
    assert row["tpr_s0_t2"] == pytest.approx((11.0 - entry) / (entry - 8.0))
    assert "r_s0_hold" not in row                     # the mirror of hold has no stop
    # and it stops out exactly where the short would have won
    row = _resolve(tape([10.0] * 6 + [9.0, 7.9]), s, bar_ts(), cfg, 0.0, 0.0, 1.0, none)
    assert row["reason_s0_t2"] == "stop"


def test_fade_transforms_the_orders():
    from qlab.strategies.sweep_orderblock import SIGNAL_SCHEMA, _fade
    blank = {k: None for k in SIGNAL_SCHEMA}
    rows = [dict(blank, interval="1m", family="sweep", arm_idx=0, arm_us=0, direction=-1,
                 kind="market", close=10.0, stop0=11.0, stop1=11.5),
            dict(blank, interval="1m", family="ob", arm_idx=0, arm_us=0, direction=1,
                 kind="limit", entry_level=9.0, close=12.0, stop0=8.0, stop1=7.5,
                 tp_near=13.0, tp_far=14.0)]
    f = pl.DataFrame(rows, schema=SIGNAL_SCHEMA)
    flip = _fade(f, "flip")
    assert flip["direction"].to_list() == [1, -1]
    assert flip["kind"].to_list() == ["market", "stop"]
    assert flip["stop0"].to_list() == [9.0, 10.0]
    assert flip["stop1"].to_list() == [8.5, 10.5]
    assert flip["tp_near"].null_count() == 2
    mirror = _fade(f, "mirror")
    assert mirror["kind"].to_list() == ["market", "stop"]
    assert mirror["stop0"].to_list() == [11.0, 8.0]   # kept: the resolver swaps them
    assert mirror["tp_near"].to_list()[1] == 13.0
