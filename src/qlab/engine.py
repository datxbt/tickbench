"""Trade accounting: what a filled trade is worth, and what an account does.

Deliberately small. This holds the arithmetic that is identical for every
strategy - fill prices, position size from risk, commission, the equity curve,
the daily loss limit - and nothing about when to trade. Strategies drive it;
they do not reimplement it, because a strategy that computes its own PnL will
eventually compute it favourably.

Sign conventions, stated once:

* ``direction`` is ``+1`` long, ``-1`` short.
* A long is opened at the **ask** and closed at the **bid**; a short is opened
  at the **bid** and closed at the **ask**. The spread is therefore paid through
  the fill prices themselves rather than added afterwards, which is the only way
  to get it right when the spread moves between entry and exit.
* Slippage is applied **against** the position on entry and on stop-outs, which
  are market orders. It is not applied to take-profit exits, which are limits
  and fill at their price or not at all.
* Commission is charged per lot per side, so a partial close pays commission
  only on the part closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_LOT = 0.01


@dataclass
class Fill:
    """One execution: the whole position on entry, or part of it on the way out."""

    ts: int  # microseconds since epoch, matching the tick clock
    price: float
    lots: float
    reason: str  # entry | tp1 | tp2 | stop | time | eod
    mid: float = 0.0
    """Mid at the moment of the fill.

    Kept so the mid-to-mid counterfactual can be computed exactly. Without it,
    separating "the signal had no edge" from "the edge existed and execution ate
    it" is guesswork, and those two failures call for opposite responses.
    """


@dataclass
class Trade:
    """One round trip, from entry through however many partial exits it takes."""

    symbol: str
    direction: int
    entry: Fill
    stop_price: float
    exits: list[Fill] = field(default_factory=list)
    contract_size: float = 100.0
    commission_per_lot_side: float = 3.5
    slippage_usd: float = 0.0
    mae_price: float = 0.0  # worst excursion against the position, in price
    mfe_price: float = 0.0  # best excursion in favour
    setup: dict = field(default_factory=dict)  # whatever the strategy wants to keep

    @property
    def closed(self) -> bool:
        return abs(self.open_lots) < 1e-9

    @property
    def open_lots(self) -> float:
        return self.entry.lots - sum(f.lots for f in self.exits)

    @property
    def gross_usd(self) -> float:
        """PnL from price alone, before any cost."""
        return sum(
            self.direction
            * (f.price - self.entry.price)
            * self.contract_size
            * f.lots
            for f in self.exits
        )

    @property
    def mid_pnl_usd(self) -> float:
        """PnL the same trade would have made filling at mid, free of all cost.

        This is the signal's edge with execution stripped out. The gap between
        it and :attr:`net_usd` is everything the broker and the book took.
        """
        return sum(
            self.direction * (f.mid - self.entry.mid) * self.contract_size * f.lots
            for f in self.exits
        )

    @property
    def execution_cost_usd(self) -> float:
        """Spread plus slippage: the whole gap, less the part that is commission."""
        return self.mid_pnl_usd - self.net_usd - self.commission_usd

    @property
    def commission_usd(self) -> float:
        closed_lots = sum(f.lots for f in self.exits)
        return self.commission_per_lot_side * (self.entry.lots + closed_lots)

    @property
    def net_usd(self) -> float:
        return self.gross_usd - self.commission_usd

    @property
    def hold_s(self) -> float:
        if not self.exits:
            return 0.0
        return (self.exits[-1].ts - self.entry.ts) / 1e6

    @property
    def exit_reason(self) -> str:
        return self.exits[-1].reason if self.exits else "open"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_ts": self.entry.ts,
            "entry_price": self.entry.price,
            "lots": self.entry.lots,
            "stop_price": self.stop_price,
            "exit_ts": self.exits[-1].ts if self.exits else None,
            "exit_price": self.exits[-1].price if self.exits else None,
            "exit_reason": self.exit_reason,
            "n_exits": len(self.exits),
            "gross_usd": self.gross_usd,
            "mid_pnl_usd": self.mid_pnl_usd,
            "execution_cost_usd": self.execution_cost_usd,
            "commission_usd": self.commission_usd,
            "slippage_usd": self.slippage_usd,
            "net_usd": self.net_usd,
            "hold_s": self.hold_s,
            "mae_price": self.mae_price,
            "mfe_price": self.mfe_price,
            **self.setup,
        }


def lots_for_risk(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    contract_size: float,
    *,
    min_lot: float = MIN_LOT,
    max_lot: float = 100.0,
) -> float:
    """Position size such that the stop costs ``risk_pct`` of equity.

    Sized on price risk only. Commission is charged on top rather than folded in,
    which is the convention every broker's own risk calculator uses - it means
    the realised loss on a stop is slightly worse than the nominal risk, and the
    report says by how much rather than hiding it in the size.
    """
    distance = abs(entry_price - stop_price)
    if distance <= 0:
        return 0.0
    loss_per_lot = distance * contract_size
    raw = equity * risk_pct / loss_per_lot
    if raw < min_lot:
        return 0.0
    return min(round(raw, 2), max_lot)


@dataclass
class Account:
    """Equity, and the two rules that can stop a strategy trading.

    The daily limit is measured against equity at the start of the trading day,
    and blocks *new* entries only - a position already open is managed to its
    own exit, because a broker does not close it for you either.
    """

    equity: float = 100_000.0
    risk_pct: float = 0.005
    daily_stop_pct: float = 0.02
    day: int | None = None  # days since epoch
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    blocked_days: int = 0
    _blocked_today: bool = False

    def roll_to(self, day: int) -> None:
        if self.day != day:
            self.day = day
            self.day_start_equity = self.equity
            self.day_pnl = 0.0
            self._blocked_today = False

    def can_trade(self) -> bool:
        return not self._blocked_today

    def apply(self, pnl: float) -> None:
        self.equity += pnl
        self.day_pnl += pnl
        if (
            not self._blocked_today
            and self.day_start_equity > 0
            and self.day_pnl <= -self.daily_stop_pct * self.day_start_equity
        ):
            self._blocked_today = True
            self.blocked_days += 1
