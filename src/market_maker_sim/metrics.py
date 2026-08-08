"""Risk-adjusted evaluation of a market-making backtest.

PnL decomposition (exact)
-------------------------
Let m_t be the mid, q_t the signed inventory, and consider the mark-to-market
PnL ``PnL_t = cash_t + q_t * m_t``. Walking through every fill and mark in time
order, each increment splits into two economically distinct terms:

* **Spread capture** (at a fill of signed quantity dq at price p, with
  pre-trade mid m): ``dq * (m - p)``. Buying below mid / selling above mid is
  the market maker's gross edge — what the spread pays for providing
  liquidity. Measured against the *pre-trade* mid so the trade's own impact
  on the touch does not contaminate the measurement.
* **Inventory PnL** (between consecutive checkpoints): ``q * dm`` — the
  revaluation of whatever position was carried while the mid moved. This is
  the inventory-risk term: pure exposure to price changes, zero-mean only if
  positions are uncorrelated with subsequent moves.

The identity ``total = spread_capture + inventory_pnl`` holds *exactly* (up to
float rounding of the mid), which the test suite asserts. A healthy market
maker earns its PnL in the first bucket; PnL arriving via the second means the
desk is speculating, not making markets.

Adverse selection (markouts)
----------------------------
For each fill, the ``tau``-markout is ``dq * (m_{t+tau} - m_t)`` per unit: how
the mid drifted after the trade, signed by the position received. Trading with
uninformed flow gives markouts near zero; informed counterparties buy just
before the price rises, so fills against them have systematically *negative*
markouts — that drift is the adverse-selection cost, and it is measured here
separately against informed and noise counterparties to show exactly who the
market maker loses to. Equivalently: effective spread (edge at trade time)
minus realized spread (edge remaining after tau) equals the adverse-selection
component.

Sharpe conventions
------------------
Within an episode, ``sharpe_step = mean(dPnL) / std(dPnL)`` over the mark grid,
scaled by sqrt(marks per episode) to a per-episode figure. Across seeds, the
*episode* Sharpe ``mean(final PnL) / std(final PnL)`` is the more honest
number (independent samples, no intra-episode autocorrelation caveats); both
are reported. No annualization is attempted — simulation seconds have no
calendar meaning, so an annualized figure would be theater.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from .backtest import BacktestResult, MMFill

__all__ = [
    "EpisodeMetrics",
    "PairedComparison",
    "PnLDecomposition",
    "StrategySummary",
    "decompose_pnl",
    "episode_metrics",
    "markouts",
    "paired_comparison",
    "summarize",
]

MARKOUT_HORIZONS = (1.0, 5.0, 30.0)


@dataclass(frozen=True)
class PnLDecomposition:
    """All values in dollars."""

    total: float
    spread_capture: float
    inventory_pnl: float
    # Cumulative paths on the mark grid (aligned with result.marks)
    spread_capture_path: np.ndarray = field(repr=False)
    inventory_pnl_path: np.ndarray = field(repr=False)


def decompose_pnl(result: BacktestResult) -> PnLDecomposition:
    """Exact decomposition of mark-to-market PnL into spread capture and
    inventory PnL, by processing fills and marks as one time-ordered stream."""
    tick = result.config.tick_size
    checkpoints: list[tuple[float, int, object]] = []  # (time, kind_rank, payload)
    for f in result.fills:
        checkpoints.append((f.time, 0, f))
    for i, m in enumerate(result.marks):
        checkpoints.append((m.time, 1, i))
    # Fills sort before the mark at the same instant: the mark's mid already
    # reflects the trade.
    checkpoints.sort(key=lambda c: (c[0], c[1]))

    spread_capture = 0.0
    inventory_pnl = 0.0
    q = 0
    prev_mid: float | None = None
    sc_path = np.zeros(len(result.marks))
    inv_path = np.zeros(len(result.marks))
    for _time, kind, payload in checkpoints:
        if kind == 0:  # fill
            f = cast(MMFill, payload)
            mid = f.pre_trade_mid
            if prev_mid is not None:
                inventory_pnl += q * (mid - prev_mid)
            spread_capture += f.signed_qty * (mid - f.price)
            q += f.signed_qty
            prev_mid = mid
        else:  # mark
            i = cast(int, payload)
            mid = result.marks[i].mid
            if prev_mid is not None:
                inventory_pnl += q * (mid - prev_mid)
            prev_mid = mid
            sc_path[i] = spread_capture
            inv_path[i] = inventory_pnl

    return PnLDecomposition(
        total=(spread_capture + inventory_pnl) * tick,
        spread_capture=spread_capture * tick,
        inventory_pnl=inventory_pnl * tick,
        spread_capture_path=sc_path * tick,
        inventory_pnl_path=inv_path * tick,
    )


def markouts(
    result: BacktestResult, horizons: tuple[float, ...] = MARKOUT_HORIZONS
) -> dict[str, dict[float, float]]:
    """Mean markout per share traded, in dollars, at each horizon, split by
    counterparty type ("all", "informed", "noise").

    markout(tau) = sign * (mid(t + tau) - mid(t)), averaged per share.
    Negative = the mid moved against the position received = adverse selection.
    Fills within tau of the session end are excluded (their markout window is
    truncated).
    """
    tick = result.config.tick_size
    times = result.mark_times()
    mids = result.mark_mids()
    end = times[-1] if len(times) else 0.0

    out: dict[str, dict[float, float]] = {}
    groups = {
        "all": result.fills,
        "informed": [f for f in result.fills if f.taker_owner == "informed"],
        "noise": [f for f in result.fills if f.taker_owner == "noise"],
        "value": [f for f in result.fills if f.taker_owner == "value"],
    }
    for name, fills in groups.items():
        out[name] = {}
        for tau in horizons:
            usable = [f for f in fills if f.time + tau <= end]
            if not usable:
                out[name][tau] = float("nan")
                continue
            total_qty = sum(f.qty for f in usable)
            total = 0.0
            for f in usable:
                # Step interpolation: last mark at or before t + tau.
                idx = int(np.searchsorted(times, f.time + tau, side="right")) - 1
                future_mid = mids[max(idx, 0)]
                total += f.signed_qty * (future_mid - f.pre_trade_mid)
            out[name][tau] = (total / total_qty) * tick
    return out


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics for a single episode (dollars unless noted)."""

    final_pnl: float
    spread_capture: float
    inventory_pnl: float
    sharpe_episode_scaled: float  # per-step Sharpe scaled by sqrt(n_steps)
    pnl_volatility: float  # std of per-mark PnL increments
    max_abs_inventory: int
    mean_abs_inventory: float
    n_fills: int
    volume: int  # shares traded
    mean_quoted_spread_ticks: float
    markout_per_share: dict[str, dict[float, float]]
    time_at_position_limit: float  # fraction of marks with |q| at/above 90% cap


def episode_metrics(result: BacktestResult, max_inventory: int | None = None) -> EpisodeMetrics:
    decomp = decompose_pnl(result)
    pnl = result.pnl_dollars()
    dpnl = np.diff(pnl)
    std = float(np.std(dpnl, ddof=1)) if len(dpnl) > 1 else 0.0
    sharpe = float(np.mean(dpnl)) / std * np.sqrt(len(dpnl)) if std > 0 else 0.0
    inv = result.mark_inventory()
    spreads = np.array([s for _, s in result.quoted_spreads], dtype=float)
    at_limit = 0.0
    if max_inventory:
        at_limit = float(np.mean(np.abs(inv) >= 0.9 * max_inventory))
    return EpisodeMetrics(
        final_pnl=float(pnl[-1]),
        spread_capture=decomp.spread_capture,
        inventory_pnl=decomp.inventory_pnl,
        sharpe_episode_scaled=sharpe,
        pnl_volatility=std,
        max_abs_inventory=int(np.max(np.abs(inv))) if len(inv) else 0,
        mean_abs_inventory=float(np.mean(np.abs(inv))) if len(inv) else 0.0,
        n_fills=len(result.fills),
        volume=sum(f.qty for f in result.fills),
        mean_quoted_spread_ticks=float(np.mean(spreads)) if len(spreads) else float("nan"),
        markout_per_share=markouts(result),
        time_at_position_limit=at_limit,
    )


@dataclass(frozen=True)
class StrategySummary:
    """Cross-seed summary for one strategy configuration."""

    strategy_name: str
    n_episodes: int
    mean_final_pnl: float
    std_final_pnl: float
    episode_sharpe: float  # mean(final) / std(final) across seeds
    mean_step_sharpe: float
    mean_spread_capture: float
    mean_inventory_pnl: float
    mean_max_abs_inventory: float
    mean_abs_inventory: float
    mean_quoted_spread_ticks: float
    mean_volume: float
    markout_5s_all: float  # mean 5s markout per share, dollars
    markout_5s_informed: float
    markout_5s_noise: float

    def row(self) -> dict[str, float | str | int]:
        return {
            "strategy": self.strategy_name,
            "episodes": self.n_episodes,
            "mean PnL ($)": round(self.mean_final_pnl, 2),
            "std PnL ($)": round(self.std_final_pnl, 2),
            "episode Sharpe": round(self.episode_sharpe, 2),
            "spread capture ($)": round(self.mean_spread_capture, 2),
            "inventory PnL ($)": round(self.mean_inventory_pnl, 2),
            "max |q|": round(self.mean_max_abs_inventory, 1),
            "mean |q|": round(self.mean_abs_inventory, 1),
            "quoted spread (ticks)": round(self.mean_quoted_spread_ticks, 2),
            "volume": round(self.mean_volume, 0),
            "markout 5s ($/sh)": round(self.markout_5s_all, 4),
        }


def summarize(
    results: list[BacktestResult], max_inventory: int | None = None
) -> StrategySummary:
    metrics = [episode_metrics(r, max_inventory) for r in results]
    finals = np.array([m.final_pnl for m in metrics])
    std_final = float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0

    def _mean_markout(group: str, tau: float) -> float:
        vals = [
            m.markout_per_share[group][tau]
            for m in metrics
            if not np.isnan(m.markout_per_share[group][tau])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return StrategySummary(
        strategy_name=results[0].strategy_name,
        n_episodes=len(results),
        mean_final_pnl=float(np.mean(finals)),
        std_final_pnl=std_final,
        episode_sharpe=float(np.mean(finals)) / std_final if std_final > 0 else 0.0,
        mean_step_sharpe=float(np.mean([m.sharpe_episode_scaled for m in metrics])),
        mean_spread_capture=float(np.mean([m.spread_capture for m in metrics])),
        mean_inventory_pnl=float(np.mean([m.inventory_pnl for m in metrics])),
        mean_max_abs_inventory=float(np.mean([m.max_abs_inventory for m in metrics])),
        mean_abs_inventory=float(np.mean([m.mean_abs_inventory for m in metrics])),
        mean_quoted_spread_ticks=float(
            np.nanmean([m.mean_quoted_spread_ticks for m in metrics])
        ),
        mean_volume=float(np.mean([m.volume for m in metrics])),
        markout_5s_all=_mean_markout("all", 5.0),
        markout_5s_informed=_mean_markout("informed", 5.0),
        markout_5s_noise=_mean_markout("noise", 5.0),
    )


@dataclass(frozen=True)
class PairedComparison:
    """Paired inference on per-episode final PnL between two strategies run on
    common random numbers (episode i of both saw the same market).

    Because the comparison is paired, the market-realization noise that
    dominates each strategy's own PnL variance cancels in the difference —
    this is the whole reason the harness uses common random numbers. Reported:
    the mean per-episode difference, its paired t-statistic, and a bootstrap
    95% CI on the mean difference (10,000 resamples of the difference vector;
    no normality assumption)."""

    n_pairs: int
    mean_diff: float  # mean(final_a - final_b), dollars
    t_stat: float  # mean_diff / (std_diff / sqrt(n))
    ci_low: float  # bootstrap 95% CI on mean_diff
    ci_high: float
    share_episodes_a_wins: float


def paired_comparison(
    results_a: list[BacktestResult],
    results_b: list[BacktestResult],
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Compare final PnL of strategy a vs b, paired by seed."""
    if len(results_a) != len(results_b):
        raise ValueError("paired comparison needs equal-length result lists")
    for ra, rb in zip(results_a, results_b, strict=True):
        if ra.config.seed != rb.config.seed:
            raise ValueError("results are not paired by seed")
    diffs = np.array(
        [ra.pnl_dollars()[-1] - rb.pnl_dollars()[-1]
         for ra, rb in zip(results_a, results_b, strict=True)]
    )
    n = len(diffs)
    se = float(np.std(diffs, ddof=1)) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return PairedComparison(
        n_pairs=n,
        mean_diff=float(diffs.mean()),
        t_stat=float(diffs.mean() / se) if se > 0 else 0.0,
        ci_low=float(lo),
        ci_high=float(hi),
        share_episodes_a_wins=float(np.mean(diffs > 0)),
    )
