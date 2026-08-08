"""Scenario tests for the matching engine.

Every economically meaningful rule gets an explicit, hand-checkable scenario:
price priority, FIFO time priority, partial fills, walking the book, price
improvement, marketable limits, market-order remainders, and cancel semantics.
The differential/property tests in test_book_properties.py then cover the
combinatorial space; these tests are the human-readable specification.
"""

import pytest

from market_maker_sim.book import LimitOrderBook
from market_maker_sim.orders import Accepted, Cancelled, Fill, Side

B, S = Side.BUY, Side.SELL


def fills(events):
    return [e for e in events if isinstance(e, Fill)]


def make_book():
    return LimitOrderBook()


class TestResting:
    def test_limit_order_rests_and_is_accepted(self):
        book = make_book()
        oid, events = book.submit_limit("a", B, 100, 5, time=1.0)
        assert events == [Accepted(time=1.0, order_id=oid, owner="a", side=B,
                                   price=100, qty=5)]
        assert book.best_bid() == 100
        assert book.best_ask() is None
        assert book.get_order(oid).qty == 5
        book.check_invariants()

    def test_mid_and_spread(self):
        book = make_book()
        book.submit_limit("a", B, 99, 5, 0.0)
        book.submit_limit("a", S, 102, 5, 0.0)
        assert book.mid() == 100.5
        assert book.spread() == 3

    def test_depth_aggregates_by_level_best_first(self):
        book = make_book()
        book.submit_limit("a", B, 100, 5, 0.0)
        book.submit_limit("b", B, 100, 3, 0.0)
        book.submit_limit("c", B, 98, 7, 0.0)
        assert book.depth(B) == [(100, 8), (98, 7)]

    def test_non_crossing_limit_generates_no_fills(self):
        book = make_book()
        book.submit_limit("a", S, 101, 5, 0.0)
        _, events = book.submit_limit("b", B, 100, 5, 0.0)
        assert fills(events) == []
        assert book.best_bid() == 100 and book.best_ask() == 101


class TestPriority:
    def test_price_priority_better_priced_maker_fills_first(self):
        book = make_book()
        book.submit_limit("cheap", S, 101, 5, 0.0)
        book.submit_limit("expensive", S, 103, 5, 0.0)
        _, events = book.submit_market("t", B, 5, 1.0)
        (fill,) = fills(events)
        assert fill.maker_owner == "cheap"
        assert fill.price == 101

    def test_time_priority_fifo_within_level(self):
        book = make_book()
        first, _ = book.submit_limit("first", S, 101, 5, 0.0)
        second, _ = book.submit_limit("second", S, 101, 5, 0.5)
        _, events = book.submit_market("t", B, 7, 1.0)
        assert [(f.maker_order_id, f.qty) for f in fills(events)] == [
            (first, 5), (second, 2)]
        # The partially-filled second order keeps its place with the remainder.
        assert book.get_order(second).qty == 3

    def test_cancel_and_replace_loses_time_priority(self):
        book = make_book()
        first, _ = book.submit_limit("first", S, 101, 5, 0.0)
        book.submit_limit("second", S, 101, 5, 0.5)
        book.cancel(first, 0.6)
        book.submit_limit("first", S, 101, 5, 0.7)  # re-enter at same price
        _, events = book.submit_market("t", B, 5, 1.0)
        assert fills(events)[0].maker_owner == "second"


class TestMatching:
    def test_execution_at_maker_price_with_price_improvement(self):
        book = make_book()
        book.submit_limit("m", S, 101, 5, 0.0)
        # Taker willing to pay 105 still executes at 101.
        _, events = book.submit_limit("t", B, 105, 5, 1.0)
        (fill,) = fills(events)
        assert fill.price == 101
        assert not isinstance(events[-1], Accepted)  # fully filled: nothing rests

    def test_taker_walks_multiple_levels(self):
        book = make_book()
        book.submit_limit("m1", S, 101, 4, 0.0)
        book.submit_limit("m2", S, 102, 4, 0.0)
        book.submit_limit("m3", S, 103, 4, 0.0)
        _, events = book.submit_market("t", B, 10, 1.0)
        assert [(f.price, f.qty) for f in fills(events)] == [(101, 4), (102, 4), (103, 2)]
        assert book.best_ask() == 103
        assert book.get_order(fills(events)[-1].maker_order_id).qty == 2

    def test_marketable_limit_fills_within_limit_then_rests(self):
        book = make_book()
        book.submit_limit("m1", S, 101, 4, 0.0)
        book.submit_limit("m2", S, 103, 4, 0.0)
        oid, events = book.submit_limit("t", B, 102, 10, 1.0)
        # Fills the 101 level, cannot touch 103, rests 6 at its own limit 102.
        assert [(f.price, f.qty) for f in fills(events)] == [(101, 4)]
        assert events[-1] == Accepted(time=1.0, order_id=oid, owner="t", side=B,
                                      price=102, qty=6)
        assert book.best_bid() == 102 and book.best_ask() == 103
        book.check_invariants()

    def test_market_order_remainder_is_cancelled_not_rested(self):
        book = make_book()
        book.submit_limit("m", S, 101, 4, 0.0)
        oid, events = book.submit_market("t", B, 10, 1.0)
        assert [(f.price, f.qty) for f in fills(events)] == [(101, 4)]
        assert events[-1] == Cancelled(time=1.0, order_id=oid, owner="t", side=B,
                                       price=None, qty=6, reason="unfilled_market")
        assert book.best_ask() is None
        assert book.best_bid() is None

    def test_market_order_on_empty_book_is_fully_cancelled(self):
        book = make_book()
        oid, events = book.submit_market("t", S, 5, 1.0)
        assert events == [Cancelled(time=1.0, order_id=oid, owner="t", side=S,
                                    price=None, qty=5, reason="unfilled_market")]

    def test_sell_side_symmetry(self):
        book = make_book()
        book.submit_limit("m1", B, 100, 4, 0.0)
        book.submit_limit("m2", B, 99, 4, 0.0)
        _, events = book.submit_limit("t", S, 99, 6, 1.0)
        assert [(f.price, f.qty) for f in fills(events)] == [(100, 4), (99, 2)]

    def test_pre_trade_mid_is_recorded_before_impact(self):
        book = make_book()
        book.submit_limit("a", B, 99, 5, 0.0)
        book.submit_limit("a", S, 101, 5, 0.0)
        _, events = book.submit_market("t", B, 5, 1.0)
        (fill,) = fills(events)
        assert fill.pre_trade_mid == 100.0  # mid before the ask was consumed


class TestCancel:
    def test_cancel_removes_order(self):
        book = make_book()
        oid, _ = book.submit_limit("a", B, 100, 5, 0.0)
        event = book.cancel(oid, 1.0)
        assert event == Cancelled(time=1.0, order_id=oid, owner="a", side=B,
                                  price=100, qty=5, reason="user")
        assert book.best_bid() is None
        assert book.get_order(oid) is None
        book.check_invariants()

    def test_cancel_unknown_or_repeated_is_noop(self):
        book = make_book()
        oid, _ = book.submit_limit("a", B, 100, 5, 0.0)
        assert book.cancel(999, 1.0) is None
        book.cancel(oid, 1.0)
        assert book.cancel(oid, 2.0) is None

    def test_cancel_after_full_fill_is_noop(self):
        book = make_book()
        oid, _ = book.submit_limit("a", S, 101, 5, 0.0)
        book.submit_market("t", B, 5, 1.0)
        assert book.cancel(oid, 2.0) is None

    def test_cancel_partially_filled_order_returns_remainder(self):
        book = make_book()
        oid, _ = book.submit_limit("a", S, 101, 5, 0.0)
        book.submit_market("t", B, 2, 1.0)
        event = book.cancel(oid, 2.0)
        assert event.qty == 3

    def test_cancel_middle_of_queue_preserves_fifo(self):
        book = make_book()
        a, _ = book.submit_limit("a", S, 101, 1, 0.0)
        b, _ = book.submit_limit("b", S, 101, 1, 0.1)
        c, _ = book.submit_limit("c", S, 101, 1, 0.2)
        book.cancel(b, 0.5)
        _, events = book.submit_market("t", B, 2, 1.0)
        assert [f.maker_order_id for f in fills(events)] == [a, c]
        book.check_invariants()


class TestValidationAndInvariants:
    @pytest.mark.parametrize("qty", [0, -1, 2.5])
    def test_bad_qty_rejected(self, qty):
        book = make_book()
        with pytest.raises(ValueError):
            book.submit_limit("a", B, 100, qty, 0.0)
        with pytest.raises(ValueError):
            book.submit_market("a", B, qty, 0.0)

    def test_non_integer_price_rejected(self):
        book = make_book()
        with pytest.raises(ValueError):
            book.submit_limit("a", B, 100.5, 5, 0.0)

    def test_level_recreated_after_emptying(self):
        """Exercises the lazy heap: a level is emptied, then re-created."""
        book = make_book()
        book.submit_limit("a", S, 101, 5, 0.0)
        book.submit_market("t", B, 5, 1.0)
        assert book.best_ask() is None
        book.submit_limit("a", S, 101, 3, 2.0)
        assert book.best_ask() == 101
        book.check_invariants()

    def test_quantity_conservation(self):
        """Total submitted = filled (counted once per side) + resting + cancelled."""
        book = make_book()
        submitted = 0
        filled = 0
        cancelled = 0
        all_events = []
        ops = [
            ("limit", "a", S, 101, 10), ("limit", "b", S, 102, 8),
            ("limit", "c", B, 100, 6), ("market", "t", B, 12),
            ("limit", "d", B, 101, 9), ("market", "t", S, 30),
        ]
        for op in ops:
            if op[0] == "limit":
                _, events = book.submit_limit(op[1], op[2], op[3], op[4], 0.0)
                submitted += op[4]
            else:
                _, events = book.submit_market(op[1], op[2], op[3], 0.0)
                submitted += op[3]
            all_events.extend(events)
            book.check_invariants()
        for e in all_events:
            if isinstance(e, Fill):
                filled += 2 * e.qty  # consumes taker and maker quantity
            elif isinstance(e, Cancelled):
                cancelled += e.qty
        resting = sum(q for side in Side for _, q in book.depth(side))
        assert submitted == filled + cancelled + resting
