"""A price-time-priority limit order book matching engine.

Semantics (standard continuous double auction, as on most equity exchanges):

* **Price priority**: an incoming order always trades against the best-priced
  opposite orders first (highest bid / lowest ask).
* **Time priority (FIFO)**: within a price level, orders execute in arrival
  order. Arrival order is defined by an engine-assigned sequence number, not
  wall-clock time, so it is total and unambiguous.
* **Execution price**: every fill prints at the *resting* order's limit price.
  An aggressive order that crosses several levels gets price improvement level
  by level ("walking the book").
* **Marketable limit orders** match immediately for whatever is available
  inside their limit, then rest the remainder at their limit price.
* **Market orders** match against everything available; any remainder is
  discarded (emitted as a ``Cancelled`` event with reason ``unfilled_market``)
  — market orders never rest.

Data structures
---------------
Each side is a ``dict[price_ticks, deque[RestingOrder]]`` (the FIFO queue per
level) plus a binary heap of level prices for O(log n) best-price access. The
heap is maintained *lazily*: prices whose level has emptied are popped on the
next peek. Cancellation is *eager* (the order is removed from its level's deque
immediately, O(level size)). Production engines usually prefer lazy tombstoning
with intrusive linked lists for O(1) cancels; eager removal is chosen here
because it keeps a much stronger invariant — every order present in a queue is
live — which makes the matching loop simpler to reason about and to test, and
level sizes in this simulation are small.
"""

from __future__ import annotations

import heapq
from collections import deque

from .orders import Accepted, BookEvent, Cancelled, Fill, RestingOrder, Side

__all__ = ["LimitOrderBook"]


class LimitOrderBook:
    def __init__(self) -> None:
        # price level -> FIFO queue of live resting orders
        self._levels: dict[Side, dict[int, deque[RestingOrder]]] = {
            Side.BUY: {},
            Side.SELL: {},
        }
        # Lazy heaps of level prices. Bid prices are stored negated so that
        # heapq's min-heap yields the highest bid first.
        self._price_heap: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, RestingOrder] = {}  # live orders by id, for cancel
        self._next_order_id = 1
        self._next_entry_seq = 1

    # ------------------------------------------------------------------ views

    def best_bid(self) -> int | None:
        return self._best_price(Side.BUY)

    def best_ask(self) -> int | None:
        return self._best_price(Side.SELL)

    def mid(self) -> float | None:
        """Mid price in ticks (may be half-integer). None if either side is empty."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: Side, n_levels: int | None = None) -> list[tuple[int, int]]:
        """Aggregate (price, total_qty) per level, best first."""
        levels = self._levels[side]
        prices = sorted(levels, reverse=(side is Side.BUY))
        if n_levels is not None:
            prices = prices[:n_levels]
        return [(p, sum(o.qty for o in levels[p])) for p in prices]

    def get_order(self, order_id: int) -> RestingOrder | None:
        return self._orders.get(order_id)

    def order_count(self) -> int:
        return len(self._orders)

    # ---------------------------------------------------------------- actions

    def submit_limit(
        self, owner: str, side: Side, price: int, qty: int, time: float
    ) -> tuple[int, list[BookEvent]]:
        """Submit a limit order. Returns (order_id, events).

        The order first matches against any crossing liquidity (it is the
        taker), then the remainder rests at ``price``.
        """
        self._validate(price=price, qty=qty)
        order_id = self._new_order_id()
        events: list[BookEvent] = []
        remaining = self._match(
            taker_id=order_id,
            taker_owner=owner,
            side=side,
            qty=qty,
            limit_price=price,
            time=time,
            events=events,
        )
        if remaining > 0:
            self._rest(order_id, owner, side, price, remaining)
            events.append(
                Accepted(time=time, order_id=order_id, owner=owner, side=side,
                         price=price, qty=remaining)
            )
        return order_id, events

    def submit_market(
        self, owner: str, side: Side, qty: int, time: float
    ) -> tuple[int, list[BookEvent]]:
        """Submit a market order. Any quantity that cannot be matched is
        discarded (reported as Cancelled/unfilled_market)."""
        self._validate(qty=qty)
        order_id = self._new_order_id()
        events: list[BookEvent] = []
        remaining = self._match(
            taker_id=order_id,
            taker_owner=owner,
            side=side,
            qty=qty,
            limit_price=None,
            time=time,
            events=events,
        )
        if remaining > 0:
            events.append(
                Cancelled(time=time, order_id=order_id, owner=owner, side=side,
                          price=None, qty=remaining, reason="unfilled_market")
            )
        return order_id, events

    def cancel(self, order_id: int, time: float) -> Cancelled | None:
        """Cancel a resting order. Returns the event, or None if the order is
        unknown or already fully filled/cancelled (a no-op, as on real venues
        where cancels race with fills)."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        level = self._levels[order.side][order.price]
        level.remove(order)  # identity-based; O(level size)
        if not level:
            del self._levels[order.side][order.price]
        return Cancelled(time=time, order_id=order_id, owner=order.owner,
                         side=order.side, price=order.price, qty=order.qty,
                         reason="user")

    # --------------------------------------------------------------- matching

    def _match(
        self,
        taker_id: int,
        taker_owner: str,
        side: Side,
        qty: int,
        limit_price: int | None,
        time: float,
        events: list[BookEvent],
    ) -> int:
        """Match an incoming order against the opposite side. Mutates the book,
        appends Fill events, and returns the unmatched remainder."""
        pre_trade_mid = self.mid()
        opposite = side.opposite
        remaining = qty
        while remaining > 0:
            best = self._best_price(opposite)
            if best is None:
                break
            # Crossing test, written sign-uniformly: a BUY crosses when its
            # limit >= best ask; a SELL crosses when its limit <= best bid.
            if limit_price is not None and side.sign * (limit_price - best) < 0:
                break
            queue = self._levels[opposite][best]
            while remaining > 0 and queue:
                maker = queue[0]
                traded = min(remaining, maker.qty)
                maker.qty -= traded
                remaining -= traded
                events.append(
                    Fill(time=time, price=maker.price, qty=traded,
                         taker_order_id=taker_id, maker_order_id=maker.order_id,
                         taker_owner=taker_owner, maker_owner=maker.owner,
                         taker_side=side, pre_trade_mid=pre_trade_mid)
                )
                if maker.qty == 0:
                    queue.popleft()
                    del self._orders[maker.order_id]
            if not queue:
                del self._levels[opposite][best]
        return remaining

    # ---------------------------------------------------------------- helpers

    def _rest(self, order_id: int, owner: str, side: Side, price: int, qty: int) -> None:
        order = RestingOrder(order_id=order_id, owner=owner, side=side,
                             price=price, qty=qty, entry_seq=self._next_entry_seq)
        self._next_entry_seq += 1
        levels = self._levels[side]
        if price not in levels:
            levels[price] = deque()
            key = -price if side is Side.BUY else price
            heapq.heappush(self._price_heap[side], key)
        levels[price].append(order)
        self._orders[order_id] = order

    def _best_price(self, side: Side) -> int | None:
        """Best live price on a side, cleaning stale heap entries lazily."""
        heap = self._price_heap[side]
        levels = self._levels[side]
        while heap:
            price = -heap[0] if side is Side.BUY else heap[0]
            if price in levels:
                return price
            heapq.heappop(heap)
        return None

    def _new_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    @staticmethod
    def _validate(qty: int, price: int | None = None) -> None:
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError(f"qty must be a positive integer, got {qty!r}")
        if price is not None and not isinstance(price, int):
            raise ValueError(f"price must be an integer tick count, got {price!r}")

    # ------------------------------------------------------------- invariants

    def check_invariants(self) -> None:
        """Assert internal consistency. Used by tests after every operation;
        never called on the hot path."""
        bid, ask = self.best_bid(), self.best_ask()
        assert bid is None or ask is None or bid < ask, (
            f"book is crossed: bid {bid} >= ask {ask}"
        )
        seen: set[int] = set()
        for side in Side:
            for price, queue in self._levels[side].items():
                assert queue, f"empty level {price} left in book"
                seqs = [o.entry_seq for o in queue]
                assert seqs == sorted(seqs), f"FIFO order violated at level {price}"
                for order in queue:
                    assert order.qty > 0, f"zero-qty order {order.order_id} in book"
                    assert order.price == price and order.side == side
                    assert self._orders.get(order.order_id) is order
                    seen.add(order.order_id)
        assert seen == set(self._orders), "order index out of sync with levels"
