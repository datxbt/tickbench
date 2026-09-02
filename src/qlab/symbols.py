"""Instrument specifications.

``pip`` is the unit that spreads and returns are quoted in throughout the
project. It is deliberately the *conventional* trading unit, not the smallest
price increment:

===========  ========  =========  ==================================
symbol       digits    pip        smallest increment ("point")
===========  ========  =========  ==================================
EURUSD       5         0.0001     0.00001
USDJPY       3         0.01       0.001
XAUUSD       3         0.01       0.001
USTEC        2         1.0        0.01   (1 index point)
===========  ========  =========  ==================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    """Static description of one tradable instrument and its raw feed."""

    name: str
    asset_class: str
    raw_dirname: str
    feed_symbol: str
    digits: int
    pip: float
    daily_break_utc: tuple[int, int] | None = None
    """Half-open [start, end) UTC hour window in which the instrument's daily
    maintenance break begins, or None for instruments that trade 24/5.

    Continuity checks use this to tell a routine broker downtime apart from a
    genuine feed outage. Derived empirically: XAUUSD gaps cluster at 20:00-21:00
    UTC with a ~1.05 h median, USTEC at 19:00-21:00 with a ~2.0 h median, roughly
    one per trading day. FX shows no such cluster.
    """

    @property
    def point(self) -> float:
        """Smallest quotable price increment."""
        return 10.0**-self.digits

    def is_break_hour(self, hour: int) -> bool:
        if self.daily_break_utc is None:
            return False
        start, end = self.daily_break_utc
        return start <= hour < end


SYMBOLS: dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(
        name="EURUSD",
        asset_class="fx",
        raw_dirname="EURUSD_RawSpread_TickData",
        feed_symbol="EURUSD_Raw_Spread",
        digits=5,
        pip=1e-4,
    ),
    "USDJPY": SymbolSpec(
        name="USDJPY",
        asset_class="fx",
        raw_dirname="USDJPY_RawSpread_TickData",
        feed_symbol="USDJPY_Raw_Spread",
        digits=3,
        pip=1e-2,
    ),
    "XAUUSD": SymbolSpec(
        name="XAUUSD",
        asset_class="metal",
        raw_dirname="XAUUSD_RawSpread_TickData",
        feed_symbol="XAUUSD_Raw_Spread",
        digits=3,
        pip=1e-2,
        daily_break_utc=(20, 22),
    ),
    "USTEC": SymbolSpec(
        name="USTEC",
        asset_class="index",
        raw_dirname="USTEC_RawSpread_TickData",
        feed_symbol="USTEC_Raw_Spread",
        digits=2,
        pip=1.0,
        daily_break_utc=(19, 22),
    ),
}

ALL_SYMBOLS: tuple[str, ...] = tuple(SYMBOLS)


def get_spec(symbol: str) -> SymbolSpec:
    try:
        return SYMBOLS[symbol.upper()]
    except KeyError:
        raise KeyError(
            f"unknown symbol {symbol!r}; known symbols: {', '.join(ALL_SYMBOLS)}"
        ) from None
