"""Tests for the evaluation layer. The key assertion is the exact PnL
decomposition identity on real backtests; hand-built examples pin down the
sign conventions of every metric."""

import numpy as np

from market_maker_sim.backtest import BacktestResult, Mark, MMFill, SimConfig, run_backtest
from market_maker_sim.metrics import decompose_pnl, episode_metrics, markouts
from market_maker_sim.orders import Side
from market_maker_sim.strategy import AvellanedaStoikov


def make_result(fills, marks, tick=0.01):
    return BacktestResult(config=SimConfig(tick_size=tick), strategy_name="test",
                          marks=marks, fills=fills, quoted_spreads=[])


def mark(t, mid, inv, cash):
    return Mark(time=t, mid=mid, efficient=mid, inventory=inv, cash_ticks=cash,
                bid=None, ask=None)


class TestDecompositionHandExamples:
    def test_pure_spread_capture_round_trip(self):
        """Buy 10 @ 99 (mid 100), sell 10 @ 101 (mid 100), mid never moves:
        all PnL is spread capture, inventory PnL is zero."""
        fills = [
            MMFill(time=1.0, side=Side.BUY, price=99, qty=10, pre_trade_mid=100.0,
                   taker_owner="noise"),
            MMFill(time=2.0, side=Side.SELL, price=101, qty=10, pre_trade_mid=100.0,
                   taker_owner="noise"),
        ]
        marks = [mark(0.0, 100.0, 0, 0), mark(3.0, 100.0, 0, 20)]
        d = decompose_pnl(make_result(fills, marks))
        assert np.isclose(d.spread_capture, 20 * 0.01)  # 10 ticks x 2 fills x qty... = 20 ticks
        assert np.isclose(d.inventory_pnl, 0.0)
        assert np.isclose(d.total, 0.20)

    def test_pure_inventory_pnl(self):
        """Buy 10 at the mid (zero edge), mid then rises 5 ticks: all PnL is
        inventory PnL."""
        fills = [MMFill(time=1.0, side=Side.BUY, price=100, qty=10,
                        pre_trade_mid=100.0, taker_owner="noise")]
        marks = [mark(0.0, 100.0, 0, 0), mark(2.0, 105.0, 10, -1000)]
        d = decompose_pnl(make_result(fills, marks))
        assert np.isclose(d.spread_capture, 0.0)
        assert np.isclose(d.inventory_pnl, 50 * 0.01)
        assert np.isclose(d.total, 0.50)

    def test_adverse_fill_negative_inventory_pnl(self):
        """Buy below mid (positive capture) but the mid then falls: capture
        stays positive, inventory PnL goes negative — the adverse-selection
        pattern the decomposition is built to expose."""
        fills = [MMFill(time=1.0, side=Side.BUY, price=98, qty=10,
                        pre_trade_mid=100.0, taker_owner="informed")]
        marks = [mark(0.0, 100.0, 0, 0), mark(2.0, 94.0, 10, -980)]
        d = decompose_pnl(make_result(fills, marks))
        assert np.isclose(d.spread_capture, 20 * 0.01)
        assert np.isclose(d.inventory_pnl, -60 * 0.01)
        assert np.isclose(d.total, -0.40)

    def test_decomposition_matches_cash_plus_inventory_accounting(self):
        """The decomposition total must equal cash + q*mid computed
        independently from the recorded marks."""
        fills = [
            MMFill(1.0, Side.BUY, 99, 5, 100.0, "noise"),
            MMFill(2.5, Side.SELL, 103, 3, 101.5, "noise"),
            MMFill(4.0, Side.BUY, 100, 7, 102.0, "informed"),
        ]
        cash = -99 * 5 + 103 * 3 - 100 * 7
        marks = [mark(0.0, 100.0, 0, 0), mark(2.0, 101.0, 5, -495),
                 mark(3.0, 101.5, 2, -186), mark(5.0, 99.0, 9, cash)]
        result = make_result(fills, marks)
        d = decompose_pnl(result)
        assert np.isclose(d.total, result.pnl_dollars()[-1])
        assert np.isclose(d.total, d.spread_capture + d.inventory_pnl)


class TestDecompositionOnRealBacktest:
    def test_identity_holds_exactly_end_to_end(self):
        cfg = SimConfig(seed=3, horizon=200.0)
        strat = AvellanedaStoikov(gamma=0.01, kappa=60.0, sigma=0.016, horizon=200.0,
                                  tick_size=0.01, size=5, max_inventory=30)
        result = run_backtest(cfg, strat)
        assert len(result.fills) > 10, "market maker never trades; sim is broken"
        d = decompose_pnl(result)
        assert np.isclose(d.total, result.pnl_dollars()[-1], atol=1e-9)
        assert np.isclose(d.total, d.spread_capture + d.inventory_pnl, atol=1e-12)
        # Paths agree at every mark, not just the endpoint.
        path_total = d.spread_capture_path + d.inventory_pnl_path
        assert np.allclose(path_total, result.pnl_dollars(), atol=1e-9)

    def test_every_passive_fill_has_positive_spread_capture(self):
        """A passive quote executes at its own price, strictly inside the
        pre-trade mid, so per-fill capture is always positive."""
        cfg = SimConfig(seed=3, horizon=200.0)
        strat = AvellanedaStoikov(gamma=0.01, kappa=60.0, sigma=0.016, horizon=200.0,
                                  tick_size=0.01, size=5, max_inventory=30)
        result = run_backtest(cfg, strat)
        for f in result.fills:
            assert f.signed_qty * (f.pre_trade_mid - f.price) > 0


class TestMarkouts:
    def test_markout_sign_convention(self):
        """We buy, mid falls: markout negative (we were adversely selected)."""
        fills = [MMFill(1.0, Side.BUY, 99, 10, 100.0, "informed")]
        marks = [mark(0.0, 100.0, 0, 0), mark(1.5, 100.0, 10, -990),
                 mark(5.5, 90.0, 10, -990), mark(40.0, 90.0, 10, -990)]
        mo = markouts(make_result(fills, marks), horizons=(5.0,))
        assert np.isclose(mo["informed"][5.0], (90 - 100) * 0.01)
        assert np.isnan(mo["noise"][5.0])  # no noise fills

    def test_fills_near_session_end_excluded(self):
        fills = [MMFill(38.0, Side.BUY, 99, 10, 100.0, "noise")]
        marks = [mark(0.0, 100.0, 0, 0), mark(40.0, 100.0, 10, -990)]
        mo = markouts(make_result(fills, marks), horizons=(5.0,))
        assert np.isnan(mo["all"][5.0])


class TestEpisodeMetrics:
    def test_metrics_computed_on_real_run(self):
        cfg = SimConfig(seed=5, horizon=200.0)
        strat = AvellanedaStoikov(gamma=0.01, kappa=60.0, sigma=0.016, horizon=200.0,
                                  tick_size=0.01, size=5, max_inventory=30)
        m = episode_metrics(run_backtest(cfg, strat), max_inventory=30)
        assert m.n_fills > 0 and m.volume >= m.n_fills
        assert m.max_abs_inventory <= 30
        assert np.isclose(m.final_pnl, m.spread_capture + m.inventory_pnl)
        assert m.pnl_volatility > 0
