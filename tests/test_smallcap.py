"""Tests for the Poudel small-cap replication.

The simulator is where this study can go quietly wrong, so the tests are aimed
at the four decisions that move every number in the report:

**Split adjustment.** Yahoo adjusts only the close. An unadjusted 2-for-1 split
puts a 50% bar into the ATR and under every breakout level, which manufactures
signals out of a corporate action.

**Which side of an ambiguous bar fills first.** A daily bar that contains both
the target and the stop cannot say which came first. Taking the target is how a
backtest flatters itself; these tests pin that the stop wins.

**When the entry happens.** The paper's pseudocode fills at ``close[t]`` on the
signal computed from ``close[t]``. The simulator fills at the next open, and that
has to stay true or the whole thing is trading on information it did not have.

**What a round turn costs.** Section 7.1's costs are per share, charged once per
trade; charging them per side, or forgetting the slippage leg, moves the verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.strategies import smallcap as sc


def _panel(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _flat_panel(n: int, ticker: str = "AAA", price: float = 10.0,
                volume: float = 1e6, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates, "ticker": ticker,
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "adj_close": price, "volume": volume,
    })


# --------------------------------------------------------------------------
# Adjustment
# --------------------------------------------------------------------------


def test_adjust_scales_every_leg_of_the_bar_not_just_the_close():
    panel = _panel([
        dict(date="2020-01-02", ticker="AAA", open=100.0, high=110.0, low=90.0,
             close=100.0, adj_close=50.0, volume=1_000.0),
    ])
    out = sc.adjust_ohlc(panel)
    row = out.iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (50.0, 55.0, 45.0, 50.0)
    assert row["volume"] == pytest.approx(2_000.0)   # shares double when price halves


def test_adjusted_split_leaves_no_phantom_return():
    """A clean 2-for-1 is a 0% day after adjustment, so it cannot fire a signal."""
    panel = _panel([
        dict(date="2020-01-02", ticker="AAA", open=100.0, high=100.0, low=100.0,
             close=100.0, adj_close=50.0, volume=1_000.0),
        dict(date="2020-01-03", ticker="AAA", open=50.0, high=50.0, low=50.0,
             close=50.0, adj_close=50.0, volume=2_000.0),
    ])
    ind = sc.indicators(sc.adjust_ohlc(panel))
    assert ind["ret"].iloc[1, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------


def test_atr_of_a_constant_range_bar_is_that_range():
    panel = _flat_panel(40)
    ind = sc.indicators(panel)
    # high - low = 0.2 every day, and the gap between bars is zero.
    assert ind["atr"].iloc[-1, 0] == pytest.approx(0.2)


def test_momentum_is_the_mean_not_the_sum_of_returns():
    dates = pd.bdate_range("2020-01-01", periods=40)
    close = 10 * (1.01 ** np.arange(40))          # exactly 1% a day
    panel = pd.DataFrame({
        "date": dates, "ticker": "AAA", "open": close, "high": close,
        "low": close, "close": close, "adj_close": close, "volume": 1e6,
    })
    ind = sc.indicators(panel)
    assert ind["mom"].iloc[-1, 0] == pytest.approx(0.01, rel=1e-6)


def test_dollar_volume_filter_is_a_dollar_amount():
    """A $1 stock trading 100k shares fails the $500K screen; a $10 one passes."""
    cheap = _flat_panel(40, ticker="AAA", price=1.0, volume=100_000)
    rich = _flat_panel(40, ticker="BBB", price=10.0, volume=100_000)
    ind = sc.indicators(pd.concat([cheap, rich], ignore_index=True))
    assert ind["dollar_volume"].iloc[-1]["AAA"] == pytest.approx(100_000)
    assert ind["dollar_volume"].iloc[-1]["BBB"] == pytest.approx(1_000_000)


# --------------------------------------------------------------------------
# Family A screens (Section 5.1.2)
# --------------------------------------------------------------------------


def test_family_a_rejects_a_price_outside_the_two_to_fifty_band():
    """Same 1% drift in every case, so only the price level can decide."""
    fam = sc.FamilyA()
    for price, expected in ((1.5, False), (10.0, True), (75.0, False)):
        dates = pd.bdate_range("2020-01-01", periods=60)
        drift = 1.01 ** np.arange(60)
        close = price * drift / drift[-1]      # ends exactly at `price`
        panel = pd.DataFrame({
            "date": dates, "ticker": "AAA", "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close, "adj_close": close, "volume": 5e6,
        })
        ind = sc.indicators(panel)
        got = bool(fam.entries(ind).iloc[-1, 0])
        assert got is expected, f"price {price} -> {got}"


def test_family_a_rejects_flat_momentum():
    ind = sc.indicators(_flat_panel(60))
    assert not sc.FamilyA().entries(ind).iloc[-1, 0]


# --------------------------------------------------------------------------
# Costs (Section 7.1)
# --------------------------------------------------------------------------


def test_round_turn_crosses_the_spread_once_and_slips_twice():
    costs = sc.EquityCosts(spread=0.05, slippage=0.01)
    assert costs.per_share_round_turn(10.0) == pytest.approx(0.07)


def test_relative_floor_binds_only_on_the_expensive_stock():
    costs = sc.EquityCosts(spread=0.05, slippage=0.01, min_bps=100.0)
    assert costs.per_share_round_turn(5.0) == pytest.approx(0.07)    # fixed wins
    assert costs.per_share_round_turn(50.0) == pytest.approx(0.50)   # 100 bps wins


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


def _one_name(bars: list[tuple[float, float, float, float]]) -> dict:
    """Build the indicator dict directly, so a test controls the exact bars."""
    dates = pd.bdate_range("2020-01-01", periods=len(bars))
    cols = ["open", "high", "low", "close"]
    frames = {
        c: pd.DataFrame({"AAA": [b[i] for b in bars]}, index=dates)
        for i, c in enumerate(cols)
    }
    frames["atr"] = pd.DataFrame({"AAA": 1.0}, index=dates)
    frames["vol"] = pd.DataFrame({"AAA": 0.5}, index=dates)
    frames["mom"] = pd.DataFrame({"AAA": 0.01}, index=dates)
    frames["volume"] = pd.DataFrame({"AAA": 1e6}, index=dates)
    return frames


def _entries(n: int, on: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    e = pd.DataFrame({"AAA": False}, index=dates)
    e.iloc[on] = True
    return e


FREE = sc.EquityCosts(spread=0.0, slippage=0.0)
PF = sc.Portfolio(starting_equity=100_000.0, max_positions=1,
                  max_position_pct=1.0, drawdown_scaling=False,
                  size_mode="vol_target", target_vol=0.5, tilt_clip=(1.0, 1.0))


def test_entry_is_the_next_open_not_the_signal_close():
    ind = _one_name([(10, 10, 10, 10), (12, 12, 12, 12), (12, 12, 12, 12)])
    res = sc.simulate(_entries(3, 0), ind, target_atr=100, stop_atr=100, time_stop=1,
                      costs=FREE, portfolio=PF)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_price"] == 12.0    # bar 1's open, not bar 0's close


def test_a_bar_holding_both_levels_is_resolved_as_a_stop():
    # Entry at 10 on bar 1; bar 2 spans 8 to 12, hitting a 1-ATR target and a
    # 2-ATR stop. A daily bar cannot order them, so the loss must be booked.
    ind = _one_name([(10, 10, 10, 10), (10, 10, 10, 10), (10, 12, 8, 9)])
    res = sc.simulate(_entries(3, 0), ind, target_atr=1.0, stop_atr=2.0, time_stop=99,
                      costs=FREE, portfolio=PF)
    trade = res.trades.iloc[0]
    assert trade["reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(8.0)
    assert trade["pnl"] < 0


def test_a_gap_through_the_stop_fills_at_the_open_not_at_the_level():
    ind = _one_name([(10, 10, 10, 10), (10, 10, 10, 10), (5, 6, 5, 6)])
    res = sc.simulate(_entries(3, 0), ind, target_atr=1.0, stop_atr=2.0, time_stop=99,
                      costs=FREE, portfolio=PF)
    trade = res.trades.iloc[0]
    assert trade["reason"] == "gap_stop"
    assert trade["exit_price"] == pytest.approx(5.0)    # not 8.0


def test_time_stop_closes_at_the_close_after_the_stated_number_of_bars():
    flat = [(10, 10, 10, 10)] * 8
    ind = _one_name(flat)
    res = sc.simulate(_entries(8, 0), ind, target_atr=99, stop_atr=99, time_stop=3,
                      costs=FREE, portfolio=PF)
    trade = res.trades.iloc[0]
    assert trade["reason"] == "time_stop"
    assert trade["bars_held"] == 3


def test_cost_is_charged_once_per_trade_and_shows_up_in_pnl():
    ind = _one_name([(10, 10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10)])
    costs = sc.EquityCosts(spread=0.05, slippage=0.01)
    res = sc.simulate(_entries(3, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                      costs=costs, portfolio=PF)
    trade = res.trades.iloc[0]
    assert trade["cost"] == pytest.approx(0.07 * trade["shares"])
    assert trade["pnl"] == pytest.approx(-trade["cost"])   # flat price, so cost is all of it


def test_max_positions_is_respected():
    dates = pd.bdate_range("2020-01-01", periods=5)
    names = ["AAA", "BBB", "CCC"]
    frames = {c: pd.DataFrame(10.0, index=dates, columns=names)
              for c in ("open", "high", "low", "close")}
    frames["atr"] = pd.DataFrame(1.0, index=dates, columns=names)
    frames["vol"] = pd.DataFrame(0.5, index=dates, columns=names)
    frames["mom"] = pd.DataFrame(0.01, index=dates, columns=names)
    frames["volume"] = pd.DataFrame(1e6, index=dates, columns=names)
    entries = pd.DataFrame(True, index=dates, columns=names)
    pf = sc.Portfolio(starting_equity=100_000.0, max_positions=2,
                      max_position_pct=1.0, drawdown_scaling=False,
                      size_mode="vol_target")
    res = sc.simulate(entries, frames, target_atr=99, stop_atr=99, time_stop=99,
                      costs=FREE, portfolio=pf)
    # Two slots, filled on the first eligible day and never released, and the
    # volatility tilt is not allowed to borrow past the account.
    assert res.trades.empty
    assert res.exposure.iloc[-1] <= 1.0


def test_regime_gate_blocks_every_entry_when_risk_off():
    ind = _one_name([(10, 10, 10, 10)] * 5)
    off = pd.Series(False, index=ind["close"].index)
    res = sc.simulate(_entries(5, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                      costs=FREE, portfolio=PF, regime=off)
    assert res.trades.empty
    assert res.equity.iloc[-1] == PF.starting_equity


def test_eligibility_mask_blocks_a_name_out_of_the_index():
    ind = _one_name([(10, 10, 10, 10)] * 5)
    out = pd.DataFrame({"AAA": False}, index=ind["close"].index)
    res = sc.simulate(_entries(5, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                      costs=FREE, portfolio=PF, eligible=out)
    assert res.trades.empty


# --------------------------------------------------------------------------
# Sizing (Section 5.1.3 vs the deployable variant)
# --------------------------------------------------------------------------


def test_paper_sizing_reproduces_the_worked_example():
    """Section 5.1.3: $50 price, 80% vol, $200 risk -> 5 shares."""
    ind = _one_name([(50, 50, 50, 50)] * 4)
    ind["vol"] = pd.DataFrame({"AAA": 0.8}, index=ind["close"].index)
    pf = sc.Portfolio(starting_equity=100_000.0, max_positions=1,
                      max_position_pct=1.0, drawdown_scaling=False,
                      size_mode="paper", risk_per_trade=200.0)
    res = sc.simulate(_entries(4, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                      costs=FREE, portfolio=pf)
    assert res.trades.iloc[0]["shares"] == 5


def test_paper_sizing_ignores_the_account_and_vol_target_does_not():
    ind = _one_name([(10, 10, 10, 10)] * 4)
    small = sc.Portfolio(starting_equity=10_000.0, max_positions=1, max_position_pct=1.0,
                         drawdown_scaling=False, size_mode="paper")
    big = sc.Portfolio(starting_equity=1_000_000.0, max_positions=1, max_position_pct=1.0,
                       drawdown_scaling=False, size_mode="paper")
    a = sc.simulate(_entries(4, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                    costs=FREE, portfolio=small).trades.iloc[0]["shares"]
    b = sc.simulate(_entries(4, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                    costs=FREE, portfolio=big).trades.iloc[0]["shares"]
    assert a == b                                   # the paper's formula has no equity term

    big_vt = sc.Portfolio(starting_equity=1_000_000.0, max_positions=1, max_position_pct=1.0,
                          drawdown_scaling=False, size_mode="vol_target")
    c = sc.simulate(_entries(4, 0), ind, target_atr=99, stop_atr=99, time_stop=1,
                    costs=FREE, portfolio=big_vt).trades.iloc[0]["shares"]
    assert c > 100 * b


def test_drawdown_ladder():
    assert sc._drawdown_multiplier(0.05) == 1.0
    assert sc._drawdown_multiplier(0.12) == 0.75
    assert sc._drawdown_multiplier(0.17) == 0.5
    assert sc._drawdown_multiplier(0.25) == 0.25


# --------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------


def test_regime_is_lagged_so_it_is_knowable_at_the_open():
    spy = pd.Series(np.arange(1.0, 101.0), index=pd.bdate_range("2020-01-01", periods=100))
    on = sc.regime_risk_on(spy, window=50)
    raw = spy > spy.rolling(50).mean()
    assert on.iloc[60] == raw.iloc[59]
    assert not on.iloc[0]
