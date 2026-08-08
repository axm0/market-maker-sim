"""Differential testing of the matching engine.

The strongest correctness argument for matching logic: run randomized order
streams through both the real engine and ``ReferenceBook`` — a deliberately
naive re-implementation of price-time priority (linear scans over a flat list,
no heaps, no deques) that is simple enough to be obviously correct — and
require *identical* fills and book state after every operation. Hypothesis
shrinks any divergence to a minimal counterexample. The engine's internal
invariants are also asserted after every step.
"""

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from market_maker_sim.book import LimitOrderBook
from market_maker_sim.orders import Fill, Side

PRICES = st.integers(min_value=95, max_value=105)
QTYS = st.integers(min_value=1, max_value=10)
SIDES = st.sampled_from([Side.BUY, Side.SELL])


@dataclass
class RefOrder:
    order_id: int
    side: Side
    price: int
    qty: int
    seq: int


class ReferenceBook:
    """Obviously-correct (and obviously slow) price-time priority matching."""

    def __init__(self):
        self.orders: list[RefOrder] = []
        self.next_id = 1
        self.next_seq = 1

    def _new_id(self):
        oid = self.next_id
        self.next_id += 1
        return oid

    def _best(self, side: Side):
        """The single order with best price, oldest first: max price for bids,
        min for asks; ties broken by lowest seq."""
        candidates = [o for o in self.orders if o.side == side]
        if not candidates:
            return None
        return min(candidates, key=lambda o: (-o.side.sign * o.price, o.seq))

    def _match(self, side: Side, qty: int, limit_price: int | None):
        """Returns (fills, remaining); fills are (price, qty, maker_id)."""
        out = []
        while qty > 0:
            maker = self._best(side.opposite)
            if maker is None:
                break
            if limit_price is not None and side.sign * (limit_price - maker.price) < 0:
                break
            traded = min(qty, maker.qty)
            out.append((maker.price, traded, maker.order_id))
            maker.qty -= traded
            qty -= traded
            if maker.qty == 0:
                self.orders.remove(maker)
        return out, qty

    def submit_limit(self, side: Side, price: int, qty: int):
        oid = self._new_id()
        fills, remaining = self._match(side, qty, price)
        if remaining > 0:
            self.orders.append(RefOrder(oid, side, price, remaining, self.next_seq))
            self.next_seq += 1
        return oid, fills

    def submit_market(self, side: Side, qty: int):
        oid = self._new_id()
        fills, _ = self._match(side, qty, None)
        return oid, fills

    def cancel(self, order_id: int) -> bool:
        for o in self.orders:
            if o.order_id == order_id:
                self.orders.remove(o)
                return True
        return False

    def state(self):
        """Full book state: per side, levels best-first, each level FIFO."""
        out = {}
        for side in Side:
            levels: dict[int, list[tuple[int, int]]] = {}
            for o in sorted((o for o in self.orders if o.side == side),
                            key=lambda o: o.seq):
                levels.setdefault(o.price, []).append((o.order_id, o.qty))
            out[side] = sorted(levels.items(), reverse=(side is Side.BUY))
        return out


def engine_state(book: LimitOrderBook):
    out = {}
    for side in Side:
        levels = {}
        for price, _ in book.depth(side):
            queue = book._levels[side][price]
            levels[price] = [(o.order_id, o.qty) for o in queue]
        out[side] = sorted(levels.items(), reverse=(side is Side.BUY))
    return out


operation = st.one_of(
    st.tuples(st.just("limit"), SIDES, PRICES, QTYS),
    st.tuples(st.just("market"), SIDES, QTYS),
    st.tuples(st.just("cancel"), st.integers(min_value=1, max_value=60)),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(operation, min_size=1, max_size=60))
def test_engine_matches_reference(ops):
    book = LimitOrderBook()
    ref = ReferenceBook()
    for op in ops:
        if op[0] == "limit":
            _, side, price, qty = op
            oid, events = book.submit_limit("x", side, price, qty, 0.0)
            ref_oid, ref_fills = ref.submit_limit(side, price, qty)
        elif op[0] == "market":
            _, side, qty = op
            oid, events = book.submit_market("x", side, qty, 0.0)
            ref_oid, ref_fills = ref.submit_market(side, qty)
        else:
            _, target = op
            cancelled = book.cancel(target, 0.0)
            ref_cancelled = ref.cancel(target)
            assert (cancelled is not None) == ref_cancelled
            book.check_invariants()
            assert engine_state(book) == ref.state()
            continue

        assert oid == ref_oid, "order-id assignment diverged"
        fills = [(e.price, e.qty, e.maker_order_id) for e in events
                 if isinstance(e, Fill)]
        assert fills == ref_fills, f"fills diverged after {op}"
        book.check_invariants()
        assert engine_state(book) == ref.state()

    # Terminal cross-check of the top of book.
    for side, best in ((Side.BUY, book.best_bid()), (Side.SELL, book.best_ask())):
        ref_best = ref._best(side)
        assert best == (ref_best.price if ref_best else None)


@settings(max_examples=200, deadline=None)
@given(st.lists(operation, min_size=1, max_size=60))
def test_book_never_crossed_and_conserves_quantity(ops):
    """Model-free invariants: the book is never crossed, and quantity is
    conserved across fills, cancels, and resting orders."""
    book = LimitOrderBook()
    submitted = filled_taker = filled_maker = cancelled = 0
    for op in ops:
        if op[0] == "limit":
            _, side, price, qty = op
            _, events = book.submit_limit("x", side, price, qty, 0.0)
            submitted += qty
        elif op[0] == "market":
            _, side, qty = op
            _, events = book.submit_market("x", side, qty, 0.0)
            submitted += qty
        else:
            ev = book.cancel(op[1], 0.0)
            events = [ev] if ev else []
        for e in events:
            if isinstance(e, Fill):
                filled_taker += e.qty
                filled_maker += e.qty
            elif e is not None and e.__class__.__name__ == "Cancelled":
                cancelled += e.qty
        book.check_invariants()
    resting = sum(q for side in Side for _, q in book.depth(side))
    assert submitted == filled_taker + filled_maker + cancelled + resting
