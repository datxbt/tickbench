"""Broker cost terms from the Exness Raw Spread contract specification."""

from __future__ import annotations

import pytest

from qlab.symbols import ALL_SYMBOLS, get_spec


def test_eurusd_commission_in_pips():
    """$2.5/side on a 100k lot is $5 round turn; a EURUSD pip is $10, so 0.5 pips."""
    spec = get_spec("EURUSD")

    assert spec.pip_value_usd() == pytest.approx(10.0)
    assert spec.commission_pips() == pytest.approx(0.5)


def test_usdjpy_commission_depends_on_the_rate():
    """A JPY-quoted pip is worth a rate-dependent number of dollars."""
    spec = get_spec("USDJPY")

    # At 155, one pip on a 100k lot is 1000 JPY = $6.45, so $5 round turn is 0.775 pips.
    assert spec.pip_value_usd(155.0) == pytest.approx(6.4516, rel=1e-3)
    assert spec.commission_pips(155.0) == pytest.approx(0.775, rel=1e-3)

    # A stronger yen makes each pip worth more, so commission costs fewer pips.
    assert spec.commission_pips(100.0) < spec.commission_pips(155.0)


def test_commission_dominates_spread_on_the_majors():
    """The point of the Raw Spread account: near-zero spread, cost in commission.

    Observed EURUSD mean spread over 2024-2026 is 0.046 pips against 0.5 pips of
    commission, so a spread-only cost model would understate the true cost by
    roughly a factor of ten.
    """
    observed_mean_spread_pips = 0.0456
    commission = get_spec("EURUSD").commission_pips()

    assert commission / observed_mean_spread_pips > 10


def test_xauusd_commission_in_pips():
    """A 100 oz gold lot makes a 0.01 pip worth $1, so $7 round turn is 7 pips."""
    spec = get_spec("XAUUSD")

    assert spec.pip_value_usd() == pytest.approx(1.0)
    assert spec.commission_pips() == pytest.approx(7.0)


def test_ustec_commission_in_pips():
    """One index point per lot, so $0.626 round turn is 0.626 pips."""
    spec = get_spec("USTEC")

    assert spec.pip_value_usd() == pytest.approx(1.0)
    assert spec.commission_pips() == pytest.approx(0.626)


def test_commission_share_differs_sharply_by_asset_class():
    """The reason one blanket cost assumption will not do.

    On EURUSD commission is ~93% of the round turn; on gold, where the spread is
    genuinely 9 pips, it is under half. A strategy ported between them inherits a
    different cost structure, not just a different number.
    """
    eurusd = get_spec("EURUSD")
    xauusd = get_spec("XAUUSD")

    eurusd_share = eurusd.commission_pips() / (
        eurusd.commission_pips() + eurusd.spec_avg_spread_pips
    )
    xauusd_share = xauusd.commission_pips() / (
        xauusd.commission_pips() + xauusd.spec_avg_spread_pips
    )

    assert eurusd_share > 0.95
    assert xauusd_share < 0.5


def test_unknown_symbol_raises():
    with pytest.raises(KeyError, match="unknown symbol"):
        get_spec("GBPUSD")


def test_every_symbol_has_a_pip_and_digits():
    for symbol in ALL_SYMBOLS:
        spec = get_spec(symbol)
        assert spec.pip > 0
        assert spec.point == pytest.approx(10.0**-spec.digits)
