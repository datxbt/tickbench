"""Tests for the Noise Area intraday momentum strategy.

Four things here can be wrong without any summary statistic looking wrong, so
all four are pinned against inputs whose answer is known by construction:

* **The point-in-time boundary.** Sigma on day ``t`` averages days ``t-1`` back
  to ``t-14`` at the *same time-of-day*. A missing shift, or a group-by on the
  wrong key, produces a band that has seen the day it is judging - and a
  strategy that looks superb and is worthless.
* **The gap adjustment.** The anchors are ``max(open, prev_close)`` for the
  upper band and ``min(...)`` for the lower. Swapping them turns a gap from
  something the model discounts into something it chases.
* **Decision timing.** Entries, reversals and stops fire at ``HH:00`` and
  ``HH:30`` and nowhere else. A rule that acts on any minute is a different,
  much more frequently traded strategy wearing the same name.
* **The three stop rules and their precedence.** At one mark a stop can fire
  and a fresh opposite signal can be true; the paper wants both to happen, in
  that order.

The time-of-day arithmetic gets its own test because it already had one silent
bug: polars ``dt.hour()`` is ``Int8``, so ``hour * 60 + minute`` overflows for
every hour past 02:00 and yields negative minutes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.strategies.intraday_momentum import (
    NY,
    NoiseAreaConfig,
    _Leg,
    _simulate_session,
    buy_and_hold,
    daily_leverage,
    entry_hour_table,
    exit_reason_table,
    noise_area,
    sessions,
    side_table,
    with_config,
)

CFG = NoiseAreaConfig()

# 13:30 UTC is 09:30 in New York on a summer date, which is the session open.
OPEN_UTC = 13 * 60 + 30


def _bars(
    days: int = 20,
    *,
    start: date = date(2023, 6, 1),
    minutes: int = 391,
    path=None,
    n_ticks: int = 10,
) -> pl.DataFrame:
    """A synthetic 1-minute tape: ``days`` weekday sessions, 09:30-16:00 NY.

    ``path(day_index, minute_index)`` returns the close for that minute; the
    default is a flat 100 so that any sigma a test sees is one the test put
    there.
    """
    path = path or (lambda d, m: 100.0)
    rows = []
    d = 0
    day = start
    while d < days:
        if day.weekday() < 5:
            for m in range(minutes):
                ts = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(
                    minutes=OPEN_UTC + m
                )
                px = float(path(d, m))
                rows.append(
                    {
                        "ts": ts,
                        "open": px,
                        "high": px,
                        "low": px,
                        "close": px,
                        "n_ticks": n_ticks,
                    }
                )
            d += 1
        day += timedelta(days=1)
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
        pl.col("n_ticks").cast(pl.UInt32),
    )


# ---------------------------------------------------------------------------
# Time-of-day arithmetic
# ---------------------------------------------------------------------------


def test_tod_does_not_overflow_int8():
    """hour * 60 + minute exceeds 127 from 02:00 onward; it must not wrap."""
    frame = sessions(_bars(days=3))
    assert frame["tod"].min() == 9 * 60 + 30
    assert frame["tod"].max() == 16 * 60
    assert frame["tod"].dtype in (pl.Int32, pl.Int64)


def test_sessions_keeps_only_the_regular_session():
    bars = _bars(days=2, minutes=391)
    # Bolt an overnight hour onto the front of the tape.
    extra = bars.head(60).with_columns(pl.col("ts") - pl.duration(hours=6))
    frame = sessions(pl.concat([extra, bars]).sort("ts"))
    assert frame["tod"].min() >= 9 * 60 + 30
    assert frame.group_by("session").len()["len"].to_list() == [391, 391]


def test_sessions_drops_a_half_day():
    """A session that stops at noon cannot be scored against a 16:00 close."""
    full = _bars(days=3)
    truncated = full.filter(
        ~(
            (pl.col("ts").dt.convert_time_zone(NY).dt.date() == date(2023, 6, 2))
            & (pl.col("ts").dt.convert_time_zone(NY).dt.hour() >= 12)
        )
    )
    kept = sessions(truncated)["session"].unique().to_list()
    assert date(2023, 6, 2) not in kept
    assert date(2023, 6, 1) in kept


# ---------------------------------------------------------------------------
# The Noise Area itself
# ---------------------------------------------------------------------------


def test_sigma_is_the_mean_absolute_move_at_that_time_of_day():
    """Day t's sigma at minute m is the mean |move| of the previous 14 days at
    minute m - computed here by hand from the same tape."""
    rng = np.random.default_rng(7)
    steps = rng.normal(0.0, 0.05, size=(30, 391))

    def path(d, m):
        return 100.0 + steps[d, : m + 1].sum()

    bars = _bars(days=30, path=path)
    frame = noise_area(bars, CFG)

    mark = 12 * 60  # noon
    # `moves` must be built from *every* session, including the 14 that
    # noise_area drops for want of a sigma - those are exactly the ones the
    # first evaluable sigma averages over.
    every = sessions(bars, CFG)
    opens = every.group_by("session").agg(pl.col("open").first().alias("o"))
    at_mark_all = (
        every.filter(pl.col("tod") == mark)
        .join(opens, on="session")
        .sort("session")
    )
    moves = (at_mark_all["close"] / at_mark_all["o"] - 1.0).abs().to_numpy()

    at_mark = frame.filter(pl.col("tod") == mark).sort("session")
    # Row 0 of the evaluable frame is session index 14, whose sigma is the mean
    # of moves[0:14].
    assert at_mark["session"][0] == at_mark_all["session"][14]
    for i in range(6):
        assert at_mark["sigma"][i] == pytest.approx(moves[i : i + 14].mean(), rel=1e-12)


def test_sigma_never_sees_its_own_day():
    """A single enormous day must not widen its own band.

    Day 20 moves 10% while every other day is flat. If sigma leaked, day 20's
    own band would be ~10% wide; point-in-time, it is zero wide and day 21's is
    the one that widens.
    """

    def path(d, m):
        return 100.0 * (1.10 if (d == 20 and m > 0) else 1.0)

    frame = noise_area(_bars(days=25, path=path), CFG)
    mark = frame.filter(pl.col("tod") == 12 * 60).sort("session")
    sig = dict(zip(mark["session"].to_list(), mark["sigma"].to_list()))
    days = sorted(sig)
    hot = days[6]  # session index 20 overall, minus the 14 dropped
    assert sig[hot] == pytest.approx(0.0, abs=1e-12)
    assert sig[days[7]] > 0.005


def test_gap_adjustment_widens_the_side_the_gap_came_from():
    """After a gap down, the upper band sits on the *previous close*, so the
    move needed to call abnormal buying includes the gap."""
    frame = noise_area(_bars(days=20, path=lambda d, m: 100.0 - (5.0 if d >= 15 else 0.0)), CFG)
    day = frame.filter(pl.col("session") == frame["session"].max())
    row = day.filter(pl.col("tod") == 12 * 60)
    day_open = row["day_open"][0]
    prev_close = row["prev_close"][0]
    sigma = row["sigma"][0]
    assert row["upper"][0] == pytest.approx(max(day_open, prev_close) * (1 + sigma))
    assert row["lower"][0] == pytest.approx(min(day_open, prev_close) * (1 - sigma))


def test_vwap_is_tick_weighted_and_session_anchored():
    """Two bars, the second carrying 9x the weight, must pull VWAP to it."""
    bars = _bars(days=16, minutes=391, path=lambda d, m: 100.0 + (10.0 if m >= 1 else 0.0))
    bars = bars.with_columns(
        pl.when(pl.col("ts").dt.minute() == 30)
        .then(pl.lit(1))
        .otherwise(pl.lit(9))
        .cast(pl.UInt32)
        .alias("n_ticks")
    )
    frame = noise_area(bars, CFG)
    day = frame.filter(pl.col("session") == frame["session"].max()).sort("tod")
    # First bar is 100 with weight 1, second is 110 with weight 9.
    assert day["vwap"][0] == pytest.approx(100.0)
    assert day["vwap"][1] == pytest.approx((100.0 * 1 + 110.0 * 9) / 10.0)
    # And it resets: the first bar of the next session is not carrying it.
    first_of_each = frame.group_by("session").agg(pl.col("vwap").first())
    assert set(np.round(first_of_each["vwap"].to_numpy(), 6)) == {100.0}


# ---------------------------------------------------------------------------
# The rule engine
# ---------------------------------------------------------------------------


def _walk(closes, upper, lower, vwap=None, cfg=CFG, decide=None, last=15 * 60 + 30):
    n = len(closes)
    tod = np.arange(10 * 60, 10 * 60 + 30 * n, 30, dtype=np.int64)
    d = np.ones(n, dtype=bool) if decide is None else np.asarray(decide)
    return _simulate_session(
        tod,
        np.asarray(closes, dtype=float),
        np.asarray(upper, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(vwap if vwap is not None else closes, dtype=float),
        d,
        cfg,
        last,
    )


def test_no_trade_inside_the_noise_area():
    legs = _walk([100, 101, 99, 100], [105] * 4, [95] * 4)
    assert legs == []


def test_break_above_upper_goes_long_and_closes_at_the_session_close():
    cfg = with_config(CFG, stop="opposite")
    legs = _walk([106, 107, 108], [105] * 3, [95] * 3, cfg=cfg)
    assert len(legs) == 1
    assert legs[0].side == 1
    assert legs[0].entry_px == 106
    assert legs[0].reason == "session_close"


def test_break_below_lower_goes_short():
    cfg = with_config(CFG, stop="opposite")
    legs = _walk([94, 93, 92], [105] * 3, [95] * 3, cfg=cfg)
    assert len(legs) == 1
    assert legs[0].side == -1


def test_opposite_band_stop_fires_only_at_the_far_boundary():
    """Base model: a long survives a fall to 100 and dies only below 95."""
    cfg = with_config(CFG, stop="opposite", allow_reentry=False)
    legs = _walk([106, 100, 94], [105] * 3, [95] * 3, cfg=cfg)
    assert len(legs) == 1
    assert legs[0].reason == "stop"
    assert legs[0].exit_px == 94


def test_current_band_stop_fires_at_the_near_boundary():
    """Same tape, ``stop='band'``: the long dies as soon as it re-enters the
    Noise Area, three marks earlier and for a much smaller loss."""
    cfg = with_config(CFG, stop="band", allow_reentry=False)
    legs = _walk([106, 100, 94], [105] * 3, [95] * 3, cfg=cfg)
    assert len(legs) == 1
    assert legs[0].exit_px == 100


def test_band_vwap_stop_takes_the_tighter_of_the_two():
    """For a long the stop is max(upper, VWAP), so a VWAP above the band binds
    first - which is the whole point of the paper's final refinement."""
    cfg = with_config(CFG, stop="band_vwap", allow_reentry=False)
    # Same tape both ways. With VWAP at 105 the long dies at mark 1 (104 < 105);
    # with the band alone at 100 it survives, because 104 is still above it.
    tape = ([106, 104, 103], [100] * 3, [95] * 3)
    tight = _walk(*tape, vwap=[90, 105, 105], cfg=cfg)
    loose = _walk(*tape, vwap=[90, 105, 105], cfg=with_config(cfg, stop="band"))
    assert tight[0].exit_px == 104 and tight[0].reason == "stop"
    assert loose[0].reason == "session_close"


def test_stop_and_reversal_happen_at_the_same_mark():
    """A long stopped out at a mark where price is already below the lower
    boundary must flip short there, not wait for the next mark."""
    cfg = with_config(CFG, stop="opposite")
    legs = _walk([106, 94, 93], [105] * 3, [95] * 3, cfg=cfg)
    assert [leg.side for leg in legs] == [1, -1]
    assert legs[0].exit_tod == legs[1].entry_tod
    assert legs[1].reason == "session_close"


def test_reentry_can_be_switched_off():
    cfg = with_config(CFG, stop="opposite", allow_reentry=False)
    legs = _walk([106, 94, 93], [105] * 3, [95] * 3, cfg=cfg)
    assert [leg.side for leg in legs] == [1]


def test_decisions_only_at_permitted_marks():
    """The same tape acted on at every mark and at every other mark gives
    different entries: the mask is load-bearing, not decorative."""
    cfg = with_config(CFG, stop="opposite")
    every = _walk([106, 100, 108], [105] * 3, [95] * 3, cfg=cfg)
    alternate = _walk(
        [106, 100, 108], [105] * 3, [95] * 3, cfg=cfg, decide=[False, True, True]
    )
    assert every[0].entry_px == 106
    assert alternate[0].entry_px == 108


def test_no_new_entry_after_the_last_decision_time():
    cfg = with_config(CFG, stop="opposite")
    legs = _walk([100, 100, 106], [105] * 3, [95] * 3, cfg=cfg, last=10 * 60)
    assert legs == []


def test_everything_is_flat_at_the_session_close():
    cfg = with_config(CFG, stop="opposite")
    legs = _walk([106, 107, 108], [105] * 3, [95] * 3, cfg=cfg)
    assert legs[-1].reason == "session_close"
    assert legs[-1].exit_tod == 10 * 60 + 60


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_leverage_is_the_vol_ratio_capped_at_four():
    steps = np.zeros((40, 391))
    steps[:, 0] = 0.0
    rng = np.random.default_rng(3)
    daily = rng.normal(0.0, 0.01, size=40)

    def path(d, m):
        return 100.0 * float(np.exp(np.cumsum(daily)[d]))

    frame = noise_area(_bars(days=40, path=path), CFG)
    lev = daily_leverage(frame, CFG).drop_nulls("sigma_daily")
    expected = np.minimum(4.0, 0.02 / lev["sigma_daily"].to_numpy())
    assert lev["leverage"].to_numpy() == pytest.approx(expected)
    assert lev["leverage"].max() <= 4.0


def test_vol_target_none_is_unlevered():
    frame = noise_area(_bars(days=25, path=lambda d, m: 100.0 + d), CFG)
    lev = daily_leverage(frame, with_config(CFG, vol_target=None))
    assert set(lev["leverage"].to_list()) == {1.0}


def test_daily_vol_never_uses_the_day_it_prices():
    """A single 20% session must not shrink its own leverage - only the ones
    that follow it."""
    jump = np.zeros(40)
    jump[25] = 0.20

    def path(d, m):
        return 100.0 * float(np.exp(np.cumsum(jump)[d]))

    frame = noise_area(_bars(days=40, path=path), CFG)
    lev = daily_leverage(frame, CFG).drop_nulls("sigma_daily").sort("session")
    sigmas = lev["sigma_daily"].to_numpy()
    # The jump day itself is priced off a run of flat days, so its sigma is 0
    # and its leverage is the cap; the day after is the one that de-levers.
    hot = int(np.argmax(sigmas))
    assert lev["leverage"][hot] < 4.0
    assert sigmas[hot - 1] < sigmas[hot]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def test_buy_and_hold_is_close_to_close_on_the_same_calendar():
    frame = _bars(days=25, path=lambda d, m: 100.0 * (1.01**d))
    bh = buy_and_hold(frame, CFG)
    assert bh["bh_bps"].to_numpy() == pytest.approx(100.0, abs=1e-6)
    assert bh["session"].to_list() == sorted(bh["session"].to_list())


def test_tables_are_empty_but_typed_when_there_are_no_legs():
    empty = pl.DataFrame(
        schema={
            "session": pl.Date, "side": pl.Int64, "entry_tod": pl.Int64,
            "exit_tod": pl.Int64, "entry_px": pl.Float64, "exit_px": pl.Float64,
            "reason": pl.Utf8, "gross_bps": pl.Float64, "cost_bps": pl.Float64,
            "leverage": pl.Float64,
        }
    )
    for fn in (exit_reason_table, entry_hour_table, side_table):
        assert fn(empty).is_empty()


def test_config_rejects_an_unknown_stop_rule():
    with pytest.raises(ValueError, match="stop must be one of"):
        NoiseAreaConfig(stop="trailing_atr")
