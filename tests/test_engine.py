"""Trade accounting and the two account-level rules.

Arithmetic a strategy is never allowed to do for itself, because a strategy that
computes its own PnL will eventually compute it favourably.
"""

from __future__ import annotations

import polars as pl
import pytest

from qlab.engine import Account, Fill, Trade, lots_for_risk
from qlab.metrics import tearsheet

US = 1_000_000


def _short(entry_price=2000.0, entry_mid=2000.05, lots=2.0) -> Trade:
    """A 2-lot gold short, entered at the bid with the mid half a spread above."""
    return Trade(
        symbol="XAUUSD",
        direction=-1,
        entry=Fill(ts=0, price=entry_price, lots=lots, reason="entry", mid=entry_mid),
        stop_price=2001.0,
        contract_size=100.0,
        commission_per_lot_side=3.5,
    )


def test_a_short_profits_when_price_falls():
    trade = _short()
    trade.exits.append(Fill(60 * US, 1999.0, 2.0, "tp2", 1999.05))

    # 1.00 of price x 100 oz x 2 lots
    assert trade.gross_usd == pytest.approx(200.0)
    # $3.5/lot/side on 2 lots in and 2 lots out
    assert trade.commission_usd == pytest.approx(14.0)
    assert trade.net_usd == pytest.approx(186.0)


def test_partial_closes_pay_commission_only_on_what_closed():
    trade = _short(lots=2.0)
    trade.exits.append(Fill(30 * US, 1999.5, 1.0, "tp1", 1999.55))
    trade.exits.append(Fill(60 * US, 2000.0, 1.0, "breakeven", 2000.05))

    # 0.5 x 100 x 1 on the first half, nothing on the second
    assert trade.gross_usd == pytest.approx(50.0)
    # entry charged on 2 lots, exits on 1 + 1
    assert trade.commission_usd == pytest.approx(3.5 * 4)
    assert trade.net_usd == pytest.approx(36.0)
    assert trade.closed


def test_the_mid_counterfactual_separates_signal_from_execution():
    """The number that says whether a losing strategy was wrong or just expensive."""
    # Mid moves 1.00 in the trade's favour, but the fills give some of it back.
    trade = _short(entry_price=2000.0, entry_mid=2000.05, lots=1.0)
    trade.exits.append(Fill(60 * US, 1999.20, 1.0, "tp2", 1999.05))

    assert trade.mid_pnl_usd == pytest.approx(100.0)  # 1.00 x 100 x 1
    assert trade.net_usd == pytest.approx(80.0 - 7.0)
    # Everything that is not commission: spread crossed plus slippage taken.
    assert trade.execution_cost_usd == pytest.approx(20.0)
    assert (
        trade.mid_pnl_usd - trade.execution_cost_usd - trade.commission_usd
        == pytest.approx(trade.net_usd)
    )


def test_size_comes_from_the_stop_distance():
    """0.5% of $100k is $500; a $1.00 stop on 100 oz loses $100 a lot."""
    assert lots_for_risk(100_000, 0.005, 2000.0, 2001.0, 100.0) == pytest.approx(5.0)
    # Twice the stop distance, half the size.
    assert lots_for_risk(100_000, 0.005, 2000.0, 2002.0, 100.0) == pytest.approx(2.5)


def test_a_stop_too_wide_to_size_returns_no_position():
    """Better to skip than to silently trade below the broker's minimum lot."""
    assert lots_for_risk(1_000, 0.005, 2000.0, 2100.0, 100.0) == 0.0
    assert lots_for_risk(100_000, 0.005, 2000.0, 2000.0, 100.0) == 0.0


def test_the_daily_stop_blocks_new_entries_for_the_rest_of_the_day():
    account = Account(equity=100_000, daily_stop_pct=0.02)
    account.roll_to(day=1)

    account.apply(-1_500)
    assert account.can_trade()

    account.apply(-600)  # -2.1% on the day
    assert not account.can_trade()
    assert account.blocked_days == 1

    # A new day lifts it, measured against the reduced equity.
    account.roll_to(day=2)
    assert account.can_trade()
    assert account.day_start_equity == pytest.approx(97_900)


def test_the_daily_stop_does_not_close_an_open_position():
    """A broker does not flatten you at the limit either - it is an entry rule."""
    account = Account(equity=100_000, daily_stop_pct=0.02)
    account.roll_to(day=1)
    account.apply(-2_500)

    assert not account.can_trade()
    # A position already open still reports its own result afterwards.
    account.apply(+800)
    assert account.equity == pytest.approx(98_300)


def test_the_tearsheet_reconciles():
    """mid - execution - commission must equal net, or the report is fiction."""
    trades = pl.DataFrame(
        {
            "entry_ts": [0, 86_400 * US],
            "exit_ts": [60 * US, 86_400 * US + 60 * US],
            "net_usd": [100.0, -60.0],
            "gross_usd": [107.0, -53.0],
            "mid_pnl_usd": [130.0, -40.0],
            "execution_cost_usd": [23.0, 13.0],
            "commission_usd": [7.0, 7.0],
            "slippage_usd": [12.0, 12.0],
            "hold_s": [60.0, 60.0],
        }
    )
    sheet = tearsheet(trades, starting_equity=100_000, label="t")

    assert sheet["net_usd"] == pytest.approx(40.0)
    assert (
        sheet["mid_pnl_usd"] - sheet["execution_cost_usd"] - sheet["commission_usd"]
        == pytest.approx(sheet["net_usd"])
    )
    assert sheet["mid_edge_per_trade"] == pytest.approx(45.0)
    assert sheet["all_in_cost_per_trade"] == pytest.approx(25.0)


def test_cost_drag_is_suppressed_when_the_edge_is_negative():
    """A ratio to a negative denominator would read as a small drag on a rout."""
    trades = pl.DataFrame(
        {
            "entry_ts": [0],
            "exit_ts": [60 * US],
            "net_usd": [-100.0],
            "gross_usd": [-93.0],
            "mid_pnl_usd": [-50.0],
            "execution_cost_usd": [43.0],
            "commission_usd": [7.0],
            "slippage_usd": [12.0],
            "hold_s": [60.0],
        }
    )
    sheet = tearsheet(trades, starting_equity=100_000)
    assert sheet["cost_drag_pct"] != sheet["cost_drag_pct"]  # NaN
