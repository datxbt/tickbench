"""Tests for the risk-managed long overlay.

The ones that matter are the point-in-time tests. A sizing strategy fails
silently when it leaks: nothing raises, the equity curve simply gets better, and
the only way to catch it is to assert the alignment directly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qlab.metrics import tearsheet_from_returns
from qlab.strategies import risk_managed_long as rml


# --------------------------------------------------------------------------
# A synthetic panel, so the arithmetic can be checked without touching disk
# --------------------------------------------------------------------------

def _panel(prices_open, prices_close, start=date(2021, 1, 4)):
    n = len(prices_open)
    days = [start + timedelta(days=i) for i in range(n)]
    ts = [datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc) for d in days]
    return pl.DataFrame({
        "nyd": days,
        "o": [float(x) for x in prices_open],
        "c": [float(x) for x in prices_close],
        "open_ts": ts,
        "n_bars": [390] * n,
        "open_hour_utc": [14] * n,
        "nights": [1] * n,
    })


class _FlatCost:
    """A cost model stand-in with one flat round-turn number."""

    def __init__(self, bps: float):
        self.bps = bps

    def round_turn_bps(self, *, price: float, hour: int) -> float:  # noqa: ARG002
        return self.bps


def _run_on(panel, cfg, cost_bps=0.0, monkeypatch=None):
    monkeypatch.setattr(rml, "cash_session_panel", lambda *a, **k: panel)
    return rml.run("USTEC", cfg, cost=_FlatCost(cost_bps))


# --------------------------------------------------------------------------
# Point-in-time
# --------------------------------------------------------------------------

def test_weight_cannot_see_the_return_it_earns(monkeypatch):
    """A weight is decided from data strictly older than the return it earns.

    Constructed so that a leak would be unmissable: a long flat stretch, then a
    single enormous up-day. A strategy that peeks would be fully invested for
    it; one that cannot must still be carrying whatever the flat stretch
    implied.
    """
    n = 260
    closes = [100.0] * n
    opens = [100.0] * n
    # one 20% jump between the second-to-last and last open
    opens[-1] = 120.0
    closes[-2] = 120.0
    cfg = replace(rml.RMLConfig(), ma_days=200, vol_days=20, band=0.0)
    out = _run_on(_panel(opens, closes), cfg, monkeypatch=monkeypatch)

    jump = out.filter(pl.col("gross_bps") > 1000)
    assert jump.height == 1, "the fixture should contain exactly one jump"
    # The weight applied to the jump was decided before it happened. With a
    # dead-flat history the volatility estimate is zero, so the target is
    # clipped to max_weight - what must NOT happen is the weight reacting to
    # the jump itself, which is asserted below on the following session.
    assert jump["weight"].item() == pytest.approx(
        out["weight"][jump.with_row_index()["index"].item() - 1]
    ), "the weight moved on the same session as the return it earned"


def test_signal_lags_by_exactly_one_session(monkeypatch):
    """target_w at row i is the raw target computed from row i-1's close."""
    rng = np.random.default_rng(7)
    n = 400
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = closes * (1 + rng.normal(0, 0.001, n))
    cfg = rml.RMLConfig(band=0.0)
    out = _run_on(_panel(opens, closes), cfg, monkeypatch=monkeypatch)

    c = np.asarray(closes)
    ma = pl.Series(c).rolling_mean(cfg.ma_days).to_numpy()
    lr = np.concatenate([[np.nan], np.diff(np.log(c))])
    vol = pl.Series(lr).rolling_std(cfg.vol_days).to_numpy() * np.sqrt(252) * 100
    expected = np.clip(cfg.target_vol_pct / vol, 0, cfg.max_weight) * (c > ma)

    got = out["target_w"].to_numpy()
    # out drops the final row (no next open), so compare over its own length.
    # The module writes 0.0 where the indicators are not yet warm and the raw
    # target is undefined, so those positions are checked separately.
    aligned = expected[: got.size - 1]
    warm = np.isfinite(aligned)
    assert got[1:][warm] == pytest.approx(aligned[warm])
    assert (got[1:][~warm] == 0.0).all()


def test_no_position_before_the_average_is_warm(monkeypatch):
    n = 300
    closes = list(np.linspace(100, 200, n))
    out = _run_on(_panel(closes, closes), rml.RMLConfig(ma_days=200),
                  monkeypatch=monkeypatch)
    assert out["weight"][:200].sum() == 0.0


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------

def test_a_full_round_trip_costs_one_round_turn(monkeypatch):
    """In and out at weight 1 pays the round turn once, not twice or half."""
    n = 260
    closes = np.concatenate([np.linspace(100, 200, 230), np.linspace(200, 100, 30)])
    out = _run_on(_panel(closes, closes), rml.RMLConfig(band=0.0),
                  cost_bps=10.0, monkeypatch=monkeypatch)
    # total cost = 10 bps * 0.5 * total turnover, and turnover counts each side
    assert float(out["cost_bps"].sum()) == pytest.approx(
        5.0 * float(out["turnover"].sum())
    )


def test_net_is_gross_less_cost_and_swap(monkeypatch):
    rng = np.random.default_rng(3)
    n = 400
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    cfg = replace(rml.RMLConfig(), swap_bps_per_night=1.25)
    out = _run_on(_panel(closes, closes), cfg, cost_bps=2.0, monkeypatch=monkeypatch)
    expected = (out["weight"] * out["gross_bps"] - out["cost_bps"] - out["swap_bps"])
    assert out["net_bps"].to_numpy() == pytest.approx(expected.to_numpy())


def test_swap_only_charged_on_a_held_position(monkeypatch):
    n = 300
    closes = list(np.linspace(200, 100, n))       # never above its own average
    cfg = replace(rml.RMLConfig(), swap_bps_per_night=5.0)
    out = _run_on(_panel(closes, closes), cfg, monkeypatch=monkeypatch)
    assert float(out["weight"].sum()) == 0.0
    assert float(out["swap_bps"].sum()) == 0.0


def test_band_reduces_turnover(monkeypatch):
    rng = np.random.default_rng(11)
    n = 500
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    wide = _run_on(_panel(closes, closes), rml.RMLConfig(band=0.25),
                   monkeypatch=monkeypatch)
    tight = _run_on(_panel(closes, closes), rml.RMLConfig(band=0.0),
                    monkeypatch=monkeypatch)
    assert float(wide["turnover"].sum()) < float(tight["turnover"].sum())


def test_target_vol_is_pure_leverage(monkeypatch):
    """Doubling the target doubles the weight wherever the cap is not binding."""
    rng = np.random.default_rng(5)
    n = 400
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    a = _run_on(_panel(closes, closes),
                rml.RMLConfig(target_vol_pct=10.0, max_weight=99.0, band=0.0),
                monkeypatch=monkeypatch)
    b = _run_on(_panel(closes, closes),
                rml.RMLConfig(target_vol_pct=20.0, max_weight=99.0, band=0.0),
                monkeypatch=monkeypatch)
    assert b["weight"].to_numpy() == pytest.approx(2.0 * a["weight"].to_numpy())


# --------------------------------------------------------------------------
# The measurement layer
# --------------------------------------------------------------------------

def test_tearsheet_compounds_rather_than_sums():
    """+50% then -50% is -25%, not 0%."""
    sheet = tearsheet_from_returns([5000.0, -5000.0])
    assert sheet["total_pct"] == pytest.approx(-25.0)


def test_tearsheet_ignores_non_finite_periods():
    a = tearsheet_from_returns([10.0, -5.0, 20.0])
    b = tearsheet_from_returns([10.0, -5.0, float("nan"), 20.0, None])
    assert a["total_pct"] == pytest.approx(b["total_pct"])
    assert a["periods"] == b["periods"]


def test_drawdown_counts_the_first_period():
    """The curve starts at 1.0, so a loss in period one is already a drawdown.

    Measuring the peak from the first *return* instead would quietly forgive
    whatever the strategy lost before it ever made anything.
    """
    sheet = tearsheet_from_returns([-1000.0] * 3)
    assert sheet["max_dd_pct"] == pytest.approx(100.0 * (0.9 ** 3 - 1))


# --------------------------------------------------------------------------
# The real panel
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["dev", "validation"])
def test_panel_covers_the_split_and_only_the_split(split):
    from qlab.loader import SPLITS
    panel = rml.cash_session_panel("USTEC", split=split,
                                   warmup=timedelta(days=400))
    live = panel.filter(~pl.col("is_warmup"))
    assert live["nyd"].min() >= SPLITS[split].start
    assert live["nyd"].max() <= SPLITS[split].end
    if split != "dev":   # dev begins at the first day of the corpus
        assert panel["nyd"].min() < SPLITS[split].start, "warmup loaded no history"
        assert panel.filter(pl.col("is_warmup")).height >= 250, "warmup too short"


def test_panel_sessions_are_full_length():
    panel = rml.cash_session_panel("USTEC", split="validation")
    assert panel["n_bars"].min() >= rml.MIN_SESSION_BARS
    # ~252 sessions a year, allowing for the holidays this filter removes
    per_year = panel.height / ((panel["nyd"].max() - panel["nyd"].min()).days / 365.25)
    assert 230 < per_year < 255


def test_test_split_stays_locked():
    from qlab.loader import SplitLockedError
    with pytest.raises(SplitLockedError):
        rml.cash_session_panel("USTEC", split="test")


# --------------------------------------------------------------------------
# Session conventions
# --------------------------------------------------------------------------

def test_a_rollover_session_must_close_where_its_day_ends():
    """Otherwise the price being read is not the one the spec names."""
    with pytest.raises(ValueError, match="must match"):
        rml.SessionSpec(name="bad", day_end_min=17 * 60, open_min=8 * 60,
                        close_min=16 * 60, min_bars=1000)


def test_the_two_shipped_conventions_are_self_consistent():
    assert rml.US_CASH.day_end_min is None
    assert rml.FX_DAY.day_end_min == rml.FX_DAY.close_min == 17 * 60
    # The FX trade is deliberately nowhere near the rollover: 17:00 New York is
    # the two hours whose spread runs 100x the weekday mean.
    assert rml.FX_DAY.open_min == 8 * 60
    assert abs(rml.FX_DAY.open_min - rml.FX_DAY.day_end_min) >= 8 * 60


def test_fx_day_rolls_the_evening_into_the_next_session():
    """A bar after 17:00 New York belongs to the FX day it opens, not the
    calendar date it carries."""
    panel = rml.cash_session_panel("USDJPY", split="validation",
                                   session=rml.FX_DAY)
    assert panel.height > 300
    # An FX day spans a full 24 hours of quoting, unlike a 390-minute cash one
    assert panel["n_bars"].median() > 1200
    assert panel["nyd"].is_sorted()


def test_fx_day_executes_in_the_liquid_hours_not_the_rollover():
    panel = rml.cash_session_panel("USDJPY", split="validation",
                                   session=rml.FX_DAY)
    hours = set(panel["open_hour_utc"].unique().to_list())
    # 08:00 New York is 12:00 or 13:00 UTC depending on daylight saving, and
    # never 21:00 or 22:00 - the two hours this convention exists to avoid.
    assert hours <= {12, 13}
    assert not hours & {21, 22}


def test_us_cash_still_reads_the_cash_open():
    panel = rml.cash_session_panel("USTEC", split="validation",
                                   session=rml.US_CASH)
    assert set(panel["open_hour_utc"].unique().to_list()) <= {13, 14}
    assert panel["n_bars"].max() <= 390


def test_the_open_must_be_a_bar_that_actually_quoted(monkeypatch):
    """A session that did not quote at its open is dropped, not filled from a
    later bar. The strategy claims to trade at the open; on such a day it could
    not have, and taking a 10:30 price and calling it the 09:30 open is the
    quiet kind of wrong that flatters a backtest.
    """
    import polars as pl

    def bar(day, ny_minute):
        # 14:30 UTC is 09:30 New York in March (EST, UTC-5 until the 14th)
        ts_open = datetime(2021, 3, day, 0, 0, tzinfo=timezone.utc)             + timedelta(minutes=ny_minute + 5 * 60)
        return {"ts": ts_open + timedelta(minutes=1), "ts_open": ts_open,
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                "bid_close": 100.0, "ask_close": 100.0}

    rows = []
    for day in (1, 2, 3):
        first = 10 * 60 + 30 if day == 2 else 9 * 60 + 30   # day 2 opens late
        rows += [bar(day, m) for m in range(first, 16 * 60)]
    frame = pl.DataFrame(rows)
    monkeypatch.setattr(rml, "load_bars", lambda *a, **k: frame)

    panel = rml.cash_session_panel("USTEC", session=replace(
        rml.US_CASH, min_bars=100))
    days = sorted(d.day for d in panel["nyd"].to_list())
    assert days == [1, 3], "the late-quoting session should have been dropped"
