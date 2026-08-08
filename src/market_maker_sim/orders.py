"""Core order and event types shared by the matching engine and the simulator.

Prices are integer *ticks* everywhere inside the engine. Using integers makes
price comparison exact (no float-equality hazards in matching logic) and mirrors
how production engines work; conversion to dollars happens only at the
accounting/reporting boundary via ``tick_size``.

Quantities are integers (shares/contracts). Time is continuous simulation time
in seconds, carried on events for the audit trail but never used by the engine
for ordering: *time priority inside the book is determined by arrival sequence
numbers assigned by the engine*, so priority is well-defined even for events at
identical timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Accepted",
    "BookEvent",
    "Cancelled",
    "Fill",
    "RestingOrder",
    "Side",
]


class Side(Enum):
    """Order side. ``sign`` is +1 for BUY, -1 for SELL, so signed quantities
    and price comparisons can be written uniformly for both sides."""

    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        return self.value

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(slots=True, eq=False)
class RestingOrder:
    """A live order resting in the book.

    ``eq=False`` keeps identity comparison: two distinct orders are never equal
    even if all fields coincide, which is what queue removal relies on.
    ``qty`` is the *remaining* quantity and is mutated as fills occur.
    ``entry_seq`` is the engine-assigned arrival sequence number that defines
    time priority at a price level.
    """

    order_id: int
    owner: str
    side: Side
    price: int  # ticks
    qty: int  # remaining
    entry_seq: int


@dataclass(frozen=True, slots=True)
class Accepted:
    """A limit order (or its unfilled remainder) was added to the book."""

    time: float
    order_id: int
    owner: str
    side: Side
    price: int
    qty: int  # quantity actually rested (original minus any immediate fills)


@dataclass(frozen=True, slots=True)
class Fill:
    """One trade: an incoming (taker) order matched a resting (maker) order.

    The execution price is always the *maker's* limit price — the taker gets
    price improvement if it was willing to trade through. One incoming order
    can generate many fills as it walks the book.
    """

    time: float
    price: int  # ticks; the maker's resting price
    qty: int
    taker_order_id: int
    maker_order_id: int
    taker_owner: str
    maker_owner: str
    taker_side: Side  # maker side is the opposite by construction
    # Mid price (in ticks, may be half-integer hence float) prevailing just
    # before the incoming order began matching. Recorded so that spread-capture
    # accounting is measured against the pre-trade mid, not a mid the trade
    # itself has already moved.
    pre_trade_mid: float | None = None


@dataclass(frozen=True, slots=True)
class Cancelled:
    """An order was removed without (further) execution.

    ``reason`` is "user" for explicit cancels and "unfilled_market" for the
    remainder of a market order that exhausted available liquidity.
    """

    time: float
    order_id: int
    owner: str
    side: Side
    price: int | None  # None for market-order remainders (they never rest)
    qty: int  # quantity removed
    reason: str


BookEvent = Accepted | Fill | Cancelled
