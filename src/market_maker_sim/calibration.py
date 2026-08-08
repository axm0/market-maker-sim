"""Empirical calibration of the Avellaneda-Stoikov inputs from simulated flow.

The A-S model takes two market parameters as given: the mid volatility
``sigma`` and the fill-intensity decay ``k`` in
``lambda(delta) = A exp(-k delta)`` — the arrival rate of trades that would
fill a quote resting ``delta`` behind the mid. In practice a desk estimates
both from data, and this module does the same from *simulated* data, so the
strategy never gets to peek at the generator's true parameters.

Fill-intensity estimation
-------------------------
Run the market with no market maker and watch every market order walk the
book. If an order's worst execution price reached depth ``d`` past the
pre-trade mid, then a hypothetical quote at any depth ``delta <= d`` on that
side would have been reached by it. So

    lambda(delta) ~= #{market orders with depth >= delta} / (2 * T)

(pooling buys and sells by symmetry, hence the 2). Regressing
``ln lambda(delta)`` on ``delta`` gives ``ln A`` (intercept) and ``-k``
(slope). The exponential form is an approximation to the simulated flow, not
an identity — the regression's fit quality tells you how good; the estimate is
biased slightly optimistic because the probe quote itself would add depth and
absorb flow, which is acknowledged and acceptable for setting a quoting
parameter.

Volatility estimation
---------------------
Realized volatility of the mid, sampled at a *coarse* interval (default 5s):
``sigma_hat = std(mid_{t+dt} - mid_t) / sqrt(dt)``. The sampling interval
matters enormously: at high frequency the mid's realized variance is dominated
by bid-ask bounce (a stationary noise term whose per-interval contribution
does not shrink with dt), inflating sigma_hat by 1.5-2.5x — the classic
volatility-signature-plot effect. Since the A-S inventory skew scales with
sigma^2, an inflated estimate makes the strategy shed inventory far too
aggressively. Sampling at 5s makes the diffusion term dominate the bounce
term, giving a nearly unbiased estimate of the fundamental volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .backtest import BacktestResult, SimConfig, run_flow_only
from .orders import BookEvent, Fill, Side

__all__ = ["FillIntensityFit", "calibrate_fill_intensity", "estimate_sigma"]


@dataclass(frozen=True)
class FillIntensityFit:
    """lambda(delta) = A * exp(-k * delta), delta in dollars."""

    A: float  # fills per second at zero depth
    k: float  # decay, 1/$
    depths_dollars: np.ndarray  # grid used for the fit
    intensities: np.ndarray  # empirical lambda(delta) on that grid
    r_squared: float

    def intensity(self, delta_dollars: float | np.ndarray) -> float | np.ndarray:
        out: float | np.ndarray = self.A * np.exp(-self.k * delta_dollars)
        return out


def _market_order_depths(events: list[BookEvent], tick_size: float) -> list[float]:
    """Depth (in dollars past the pre-trade mid) reached by each market order,
    from the raw event stream of a flow-only run. Consecutive fills sharing a
    taker_order_id belong to one order walking the book."""
    depths: dict[int, float] = {}
    for ev in events:
        if not isinstance(ev, Fill) or ev.pre_trade_mid is None:
            continue  # mid undefined when one book side is transiently empty
        if ev.taker_side is Side.BUY:
            depth_ticks = ev.price - ev.pre_trade_mid
        else:
            depth_ticks = ev.pre_trade_mid - ev.price
        depth = max(depth_ticks, 0.0) * tick_size
        key = ev.taker_order_id
        depths[key] = max(depths.get(key, 0.0), depth)
    return list(depths.values())


def calibrate_fill_intensity(
    config: SimConfig,
    n_episodes: int = 5,
    max_depth_ticks: int = 8,
    min_count: int = 20,
) -> FillIntensityFit:
    """Estimate A and k by running `n_episodes` flow-only simulations."""
    all_depths: list[float] = []
    total_time = 0.0
    for episode in range(n_episodes):
        cfg = replace(config, seed=config.seed + 1000 + episode)
        _, events = run_flow_only(cfg)
        all_depths.extend(_market_order_depths(events, cfg.tick_size))
        total_time += cfg.warmup + cfg.horizon

    depths = np.array(all_depths)
    grid_ticks = np.arange(1, max_depth_ticks + 1, dtype=float)
    grid = grid_ticks * config.tick_size
    counts = np.array([(depths >= d).sum() for d in grid])
    keep = counts >= min_count
    if keep.sum() < 3:
        raise ValueError(
            "not enough market-order depth samples to fit lambda(delta); "
            "run more/longer episodes"
        )
    grid, counts = grid[keep], counts[keep]
    lam = counts / (2.0 * total_time)

    slope, intercept = np.polyfit(grid, np.log(lam), 1)
    fitted = intercept + slope * grid
    ss_res = float(np.sum((np.log(lam) - fitted) ** 2))
    ss_tot = float(np.sum((np.log(lam) - np.log(lam).mean()) ** 2))
    return FillIntensityFit(
        A=float(np.exp(intercept)),
        k=float(-slope),
        depths_dollars=grid,
        intensities=lam,
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )


def estimate_sigma(result: BacktestResult, dt_target: float = 5.0) -> float:
    """Realized mid volatility in dollars per sqrt-second, from a flow-only
    run, sampled every ~dt_target seconds to suppress bid-ask-bounce bias."""
    mids = result.mark_mids() * result.config.tick_size
    times = result.mark_times()
    if len(mids) < 3:
        raise ValueError("not enough marks to estimate sigma")
    mark_dt = float(np.median(np.diff(times)))
    stride = max(1, round(dt_target / mark_dt))
    sampled = mids[::stride]
    if len(sampled) < 3:
        raise ValueError("not enough marks at the target sampling interval")
    return float(np.std(np.diff(sampled), ddof=1) / np.sqrt(stride * mark_dt))
