"""End-to-end tests of the backtest harness: determinism, accounting
consistency, passivity of the market maker, and position-limit enforcement."""

import numpy as np

from market_maker_sim.backtest import SimConfig, run_backtest
from market_maker_sim.orders import Side
from market_maker_sim.strategy import AvellanedaStoikov, SymmetricQuoter

MAX_INV = 30


def make_strategy():
    return AvellanedaStoikov(gamma=0.01, kappa=60.0, sigma=0.016, horizon=300.0,
                             tick_size=0.01, size=5, max_inventory=MAX_INV)


def run(seed=11, horizon=300.0, strategy=None):
    cfg = SimConfig(seed=seed, horizon=horizon)
    return run_backtest(cfg, strategy or make_strategy())


class TestDeterminism:
    def test_same_seed_same_run(self):
        r1, r2 = run(seed=4), run(seed=4)
        assert r1.fills == r2.fills
        assert r1.marks == r2.marks

    def test_different_seed_different_run(self):
        assert run(seed=4).fills != run(seed=5).fills


class TestAccounting:
    def test_marks_consistent_with_fill_replay(self):
        """Replaying the fill stream must reproduce the inventory and cash
        recorded at every mark — the two accounting paths cannot diverge."""
        result = run()
        fills = iter(result.fills)
        f = next(fills, None)
        inv, cash = 0, 0
        for m in result.marks:
            while f is not None and f.time <= m.time:
                inv += f.signed_qty
                cash -= f.signed_qty * f.price
                f = next(fills, None)
            assert m.inventory == inv
            assert m.cash_ticks == cash

    def test_flat_start_and_bounded_inventory(self):
        result = run()
        assert result.marks[0].inventory == 0
        inv = result.mark_inventory()
        assert np.max(np.abs(inv)) <= MAX_INV

    def test_no_fills_before_session_start(self):
        result = run()
        assert all(f.time >= 0.0 for f in result.fills)
        assert result.marks[0].time == 0.0

    def test_mm_is_always_maker_at_its_own_price(self):
        """Every MM fill must be at the MM's quoted price on the correct side
        of the pre-trade mid: buys strictly below, sells strictly above."""
        result = run()
        assert len(result.fills) > 10
        for f in result.fills:
            if f.side is Side.BUY:
                assert f.price < f.pre_trade_mid
            else:
                assert f.price > f.pre_trade_mid


class TestQuoteManagement:
    def test_quotes_present_at_marks(self):
        """The MM should be quoting two-sided most of the time (one side may
        be pulled at position limits)."""
        result = run()
        both = sum(1 for m in result.marks if m.bid is not None and m.ask is not None)
        assert both / len(result.marks) > 0.8

    def test_recorded_quotes_never_crossed(self):
        result = run()
        for m in result.marks:
            if m.bid is not None and m.ask is not None:
                assert m.bid < m.ask

    def test_baseline_strategy_runs(self):
        result = run(strategy=SymmetricQuoter(half_spread_ticks=4, size=5,
                                              max_inventory=MAX_INV))
        assert len(result.fills) > 10
        assert result.strategy_name == "symmetric"


class TestInventoryControl:
    def test_as_inventory_mean_reverts_vs_baseline(self):
        """The economic point of the reservation-price skew: across seeds, A-S
        carries systematically less inventory than the unskewed baseline in
        the same markets (common random numbers)."""
        as_inv, sym_inv = [], []
        for seed in range(6):
            cfg = SimConfig(seed=100 + seed, horizon=300.0)
            r_as = run_backtest(cfg, make_strategy())
            r_sym = run_backtest(cfg, SymmetricQuoter(half_spread_ticks=4, size=5,
                                                      max_inventory=MAX_INV))
            as_inv.append(np.mean(np.abs(r_as.mark_inventory())))
            sym_inv.append(np.mean(np.abs(r_sym.mark_inventory())))
        assert np.mean(as_inv) < np.mean(sym_inv)
