"""Tests for the order-flow model: the efficient-price process, the structural
properties of the generated flow, and the statistical signature that informed
trading is actually informed (its trades predict future price moves)."""

import numpy as np
import pytest

from market_maker_sim.backtest import SimConfig, run_flow_only
from market_maker_sim.flow import EfficientPrice
from market_maker_sim.orders import Fill


class TestEfficientPrice:
    def test_time_cannot_go_backwards(self):
        p = EfficientPrice(10000.0, 2.0, np.random.default_rng(0))
        p.value_at(5.0)
        with pytest.raises(ValueError):
            p.value_at(4.0)

    def test_same_time_returns_same_value(self):
        p = EfficientPrice(10000.0, 2.0, np.random.default_rng(0))
        assert p.value_at(3.0) == p.value_at(3.0)

    def test_increment_scaling_is_sqrt_dt(self):
        """Brownian scaling: increments over dt have std sigma * sqrt(dt),
        independent of how the interval is chopped up."""
        rng = np.random.default_rng(42)
        sigma, dt, n = 2.0, 0.25, 20_000
        p = EfficientPrice(0.0, sigma, rng)
        values = [p.value_at(i * dt) for i in range(1, n + 1)]
        increments = np.diff(np.array(values))
        observed = increments.std(ddof=1)
        expected = sigma * np.sqrt(dt)
        assert abs(observed - expected) / expected < 0.05
        assert abs(increments.mean()) < 3 * expected / np.sqrt(n)  # driftless


def flow_run(seed=1, informed_rate=None, horizon=300.0):
    cfg = SimConfig(seed=seed, horizon=horizon, warmup=60.0)
    if informed_rate is not None:
        from dataclasses import replace

        cfg = replace(cfg, flow=replace(cfg.flow, informed_rate=informed_rate))
    return cfg, *run_flow_only(cfg)


class TestFlowStructure:
    def test_book_stays_two_sided(self):
        _, result, _ = flow_run()
        two_sided = [m for m in result.marks if m.mid is not None]
        assert len(two_sided) == len(result.marks)

    def test_mid_tracks_efficient_price(self):
        """Informed trading anchors the traded mid to the latent p*."""
        _, result, _ = flow_run()
        gap_ticks = np.abs(result.mark_mids()
                           - np.array([m.efficient for m in result.marks]))
        assert np.median(gap_ticks) < 10  # within 10 ticks median

    def test_without_informed_flow_mid_decouples_from_pstar(self):
        _, anchored, _ = flow_run(informed_rate=None)
        _, free, _ = flow_run(informed_rate=0.0)
        gap = lambda r: np.abs(  # noqa: E731
            r.mark_mids() - np.array([m.efficient for m in r.marks]))
        assert np.median(gap(free)) > np.median(gap(anchored))

    def test_book_relative_noise_limits_never_cross(self):
        """Noise limit orders are placed behind the opposite touch, so they
        must never fill on arrival. (Value-trader limits are allowed to cross
        — that is the anchoring mechanism — so they are excluded here.)"""
        _, _, events = flow_run(horizon=120.0)
        noise_rested = {e.order_id for e in events
                        if e.__class__.__name__ == "Accepted" and e.owner == "noise"}
        noise_takers = {e.taker_order_id for e in events
                        if isinstance(e, Fill) and e.taker_owner == "noise"}
        # A noise order id in both sets would be a limit order that crossed.
        assert noise_rested.isdisjoint(noise_takers)

    def test_value_orders_anchor_the_book(self):
        """When the book drifts from p*, marketable value-limit orders must
        exist (they are the correction channel)."""
        _, _, events = flow_run(horizon=300.0)
        value_taker_fills = [e for e in events
                             if isinstance(e, Fill) and e.taker_owner == "value"]
        assert len(value_taker_fills) > 0

    def test_deterministic_given_seed(self):
        _, r1, e1 = flow_run(seed=9, horizon=120.0)
        _, r2, e2 = flow_run(seed=9, horizon=120.0)
        assert e1 == e2
        assert [(m.time, m.mid) for m in r1.marks] == [(m.time, m.mid) for m in r2.marks]


class TestInformedFlowIsInformed:
    def test_informed_trades_predict_price_moves(self):
        """The defining property of adverse selection: conditional on an
        informed market order, the efficient price is on the far side of the
        trade, so subsequent mid moves are in the trade's direction. Noise
        trades have no such edge."""
        _, result, events = flow_run(horizon=600.0)
        times = result.mark_times()
        mids = result.mark_mids()
        session_start_offset = 60.0  # warmup: event times are absolute

        def mean_markout(owner: str) -> float:
            takers: dict[int, Fill] = {}
            for e in events:
                if isinstance(e, Fill) and e.taker_owner == owner:
                    takers.setdefault(e.taker_order_id, e)  # first fill per order
            drifts = []
            for f in takers.values():
                t = f.time - session_start_offset
                if t < 0 or t + 10.0 > times[-1]:
                    continue
                idx = int(np.searchsorted(times, t + 10.0, side="right")) - 1
                drifts.append(f.taker_side.sign * (mids[idx] - f.pre_trade_mid))
            return float(np.mean(drifts))

        informed_edge = mean_markout("informed")
        noise_edge = mean_markout("noise")
        assert informed_edge > 1.0  # ticks: clearly positive drift
        assert informed_edge > noise_edge + 0.5
