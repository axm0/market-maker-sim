"""Tests for the quoting strategies: formulas checked against hand-computed
values, and the economic properties (skew direction, spread dynamics, limits,
passivity) checked explicitly."""

import math

from market_maker_sim.book import LimitOrderBook
from market_maker_sim.orders import Side
from market_maker_sim.strategy import AvellanedaStoikov, SymmetricQuoter

TICK = 0.01


def make_as(gamma=0.1, kappa=30.0, sigma=0.02, horizon=600.0, **kw):
    return AvellanedaStoikov(gamma=gamma, kappa=kappa, sigma=sigma, horizon=horizon,
                             tick_size=TICK, size=5, max_inventory=50, **kw)


def two_sided_book(bid=9900, ask=10100):
    book = LimitOrderBook()
    book.submit_limit("bg", Side.BUY, bid, 100, 0.0)
    book.submit_limit("bg", Side.SELL, ask, 100, 0.0)
    return book


class TestFormulas:
    def test_reservation_price_hand_computed(self):
        s = make_as(gamma=0.1, sigma=0.02, horizon=600.0)
        # r = s - q * gamma * sigma^2 * tau = 100 - 10 * 0.1 * 0.0004 * 600 = 99.76
        assert s.reservation_price(100.0, 10, t=0.0) == 100.0 - 10 * 0.1 * 0.0004 * 600

    def test_reservation_price_equals_mid_when_flat(self):
        s = make_as()
        assert s.reservation_price(100.0, 0, t=0.0) == 100.0

    def test_optimal_spread_hand_computed(self):
        s = make_as(gamma=0.1, kappa=30.0, sigma=0.02, horizon=600.0)
        expected = 0.1 * 0.0004 * 600 + (2 / 0.1) * math.log(1 + 0.1 / 30.0)
        assert math.isclose(s.optimal_total_spread(0.0), expected)

    def test_spread_narrows_as_horizon_approaches(self):
        s = make_as()
        spreads = [s.optimal_total_spread(t) for t in (0.0, 300.0, 599.0)]
        assert spreads[0] > spreads[1] > spreads[2]
        # At t = T only the liquidity term (2/gamma) ln(1 + gamma/k) remains.
        floor = (2 / s.gamma) * math.log(1 + s.gamma / s.kappa)
        assert math.isclose(s.optimal_total_spread(600.0), floor)

    def test_reservation_price_pinned_to_mid_at_horizon(self):
        s = make_as()
        assert s.reservation_price(100.0, 25, t=600.0) == 100.0


class TestQuoteBehavior:
    def test_flat_inventory_quotes_symmetric_around_mid(self):
        s = make_as()
        q = s.quote(0.0, 10000.0, 0, two_sided_book())
        assert q.bid_price is not None and q.ask_price is not None
        assert (q.bid_price + q.ask_price) / 2 == 10000.0

    def test_long_inventory_shifts_both_quotes_down(self):
        s = make_as()
        book = two_sided_book()
        flat = s.quote(0.0, 10000.0, 0, book)
        long = s.quote(0.0, 10000.0, 20, book)
        short = s.quote(0.0, 10000.0, -20, book)
        assert long.bid_price < flat.bid_price and long.ask_price < flat.ask_price
        assert short.bid_price > flat.bid_price and short.ask_price > flat.ask_price

    def test_outward_tick_rounding_never_tightens_spread(self):
        s = make_as()
        q = s.quote(0.0, 10000.3, 3, two_sided_book())
        r = s.reservation_price(100.003, 3, 0.0)
        half = s.optimal_total_spread(0.0) / 2
        assert q.bid_price * TICK <= r - half
        assert q.ask_price * TICK >= r + half

    def test_higher_gamma_widens_spread(self):
        book = two_sided_book()
        narrow = make_as(gamma=0.05).quote(0.0, 10000.0, 0, book)
        wide = make_as(gamma=1.0).quote(0.0, 10000.0, 0, book)
        assert (wide.ask_price - wide.bid_price) > (narrow.ask_price - narrow.bid_price)


class TestConstraints:
    def test_position_limit_blocks_bid_when_long(self):
        s = make_as()
        q = s.quote(0.0, 10000.0, 50, two_sided_book())
        assert q.bid_price is None and q.ask_price is not None
        # One unit below the threshold that would breach the cap: bid returns.
        q = s.quote(0.0, 10000.0, 45, two_sided_book())
        assert q.bid_price is not None

    def test_position_limit_blocks_ask_when_short(self):
        s = make_as()
        q = s.quote(0.0, 10000.0, -50, two_sided_book())
        assert q.ask_price is None and q.bid_price is not None

    def test_quotes_clipped_passive_inside_tight_touch(self):
        s = make_as()
        book = two_sided_book(bid=9999, ask=10001)
        # Huge long inventory: raw model quotes would cross the best bid.
        q = s.quote(0.0, 10000.0, 50, book)
        assert q.ask_price >= 9999 + 1
        q2 = s.quote(0.0, 10000.0, -50, book)
        assert q2.bid_price <= 10001 - 1

    def test_bid_always_below_ask(self):
        s = make_as(gamma=2.0)
        book = two_sided_book(bid=9999, ask=10001)
        for inv in (-50, -20, 0, 20, 50):
            q = s.quote(0.0, 10000.0, inv, book)
            if q.bid_price is not None and q.ask_price is not None:
                assert q.bid_price < q.ask_price


class TestSymmetricBaseline:
    def test_fixed_half_spread_around_mid(self):
        s = SymmetricQuoter(half_spread_ticks=4, size=5, max_inventory=50)
        q = s.quote(0.0, 10000.0, 0, two_sided_book())
        assert q.bid_price == 9996 and q.ask_price == 10004

    def test_no_inventory_skew(self):
        s = SymmetricQuoter(half_spread_ticks=4, size=5, max_inventory=50)
        book = two_sided_book()
        assert s.quote(0.0, 10000.0, 30, book) == s.quote(0.0, 10000.0, -30, book)

    def test_position_limits_still_apply(self):
        s = SymmetricQuoter(half_spread_ticks=4, size=5, max_inventory=50)
        assert s.quote(0.0, 10000.0, 50, two_sided_book()).bid_price is None
