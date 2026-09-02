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

    # --- Broker account terms (Exness Raw Spread) ----------------------------
    # From the published contract specification, not inferred from the feed.
    # On this account type the majors quote a genuine zero spread most of the
    # time and the broker takes revenue as commission instead, so a spread-only
    # cost model understates the true cost by roughly an order of magnitude.
    spec_avg_spread_pips: float | None = None
    commission_per_lot_side_usd: float | None = None
    contract_size: float | None = None
    quote_ccy: str = "USD"

    @property
    def point(self) -> float:
        """Smallest quotable price increment."""
        return 10.0**-self.digits

    def is_break_hour(self, hour: int) -> bool:
        if self.daily_break_utc is None:
            return False
        start, end = self.daily_break_utc
        return start <= hour < end

    def pip_value_usd(self, quote_per_usd: float = 1.0) -> float:
        """USD value of one pip on one standard lot.

        ``quote_per_usd`` is how many units of the quote currency one USD buys -
        1.0 for a USD-quoted symbol, and the prevailing rate for a JPY-quoted one
        (for USDJPY that is simply the symbol's own price).
        """
        if self.contract_size is None:
            raise ValueError(f"{self.name}: contract_size not specified")
        return self.contract_size * self.pip / quote_per_usd

    def commission_pips(self, quote_per_usd: float = 1.0) -> float:
        """Round-turn commission expressed in pips, so it adds to the spread.

        Round turn is two sides, so this is twice the per-side charge.
        """
        if self.commission_per_lot_side_usd is None:
            raise ValueError(f"{self.name}: commission not specified")
        return 2.0 * self.commission_per_lot_side_usd / self.pip_value_usd(quote_per_usd)


SYMBOLS: dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(
        name="EURUSD",
        asset_class="fx",
        raw_dirname="EURUSD_RawSpread_TickData",
        feed_symbol="EURUSD_Raw_Spread",
        digits=5,
        pip=1e-4,
        spec_avg_spread_pips=0.0,
        commission_per_lot_side_usd=2.5,
        contract_size=100_000,
        quote_ccy="USD",
    ),
    "USDJPY": SymbolSpec(
        name="USDJPY",
        asset_class="fx",
        raw_dirname="USDJPY_RawSpread_TickData",
        feed_symbol="USDJPY_Raw_Spread",
        digits=3,
        pip=1e-2,
        spec_avg_spread_pips=0.0,
        commission_per_lot_side_usd=2.5,
        contract_size=100_000,
        quote_ccy="JPY",
    ),
    "XAUUSD": SymbolSpec(
        name="XAUUSD",
        asset_class="metal",
        raw_dirname="XAUUSD_RawSpread_TickData",
        feed_symbol="XAUUSD_Raw_Spread",
        digits=3,
        pip=1e-2,
        daily_break_utc=(20, 22),
        # Metals sit on a separate Exness contract-specification page; commission
        # and contract size are left unset rather than guessed. Until they are
        # filled in, XAUUSD cost estimates cover spread only.
    ),
    "USTEC": SymbolSpec(
        name="USTEC",
        asset_class="index",
        raw_dirname="USTEC_RawSpread_TickData",
        feed_symbol="USTEC_Raw_Spread",
        digits=2,
        pip=1.0,
        daily_break_utc=(19, 22),
        # As above: indices have their own specification page.
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
