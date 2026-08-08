"""Result figures.

Conventions used across all figures (so they read as one system):

* Colorblind-safe Okabe-Ito palette with a *fixed* assignment per entity —
  a series keeps its color in every figure (A-S is always blue, the baseline
  always orange, spread capture always green, ...). Line styles differ as a
  secondary encoding so no distinction rests on color alone.
* One y-axis per panel, recessive grid, no top/right spines, direct labels or
  legends on every multi-series panel.
* All money in dollars, time in session seconds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .backtest import BacktestResult
from .calibration import FillIntensityFit
from .metrics import StrategySummary, decompose_pnl
from .orders import Side

__all__ = [
    "plot_episode",
    "plot_fill_intensity",
    "plot_gamma_sweep",
    "plot_latency_sweep",
    "plot_pnl_distributions",
    "plot_sigma_sweep",
]

# Fixed entity -> color assignment (Okabe-Ito).
C = {
    "as": "#0072B2",  # Avellaneda-Stoikov
    "sym": "#E69F00",  # symmetric baseline
    "mid": "#333333",
    "efficient": "#E69F00",
    "total": "#333333",
    "spread_capture": "#009E73",
    "inventory_pnl": "#CC79A7",
    "bid": "#0072B2",
    "ask": "#D55E00",
    "grid": "#dddddd",
}


def _style(ax: Axes) -> None:
    ax.grid(True, color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_episode(result: BacktestResult, path: Path, title: str | None = None) -> Path:
    """Single-episode dashboard: prices & quotes, inventory, PnL decomposition."""
    tick = result.config.tick_size
    t = result.mark_times()
    mid = result.mark_mids() * tick
    eff = np.array([m.efficient for m in result.marks]) * tick
    inv = result.mark_inventory()
    bid = np.array([m.bid if m.bid is not None else np.nan for m in result.marks]) * tick
    ask = np.array([m.ask if m.ask is not None else np.nan for m in result.marks]) * tick
    decomp = decompose_pnl(result)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1.4]})
    fig.suptitle(title or f"{result.strategy_name}: one episode "
                 f"(seed {result.config.seed})", fontsize=12)

    ax = axes[0]
    ax.plot(t, eff, color=C["efficient"], lw=1.2, ls="--", label="efficient price $p^*$")
    ax.plot(t, mid, color=C["mid"], lw=1.4, label="mid")
    ax.plot(t, bid, color=C["bid"], lw=0.9, alpha=0.85, label="MM bid")
    ax.plot(t, ask, color=C["ask"], lw=0.9, alpha=0.85, label="MM ask")
    buys = [f for f in result.fills if f.side is Side.BUY]
    sells = [f for f in result.fills if f.side is Side.SELL]
    ax.scatter([f.time for f in buys], [f.price * tick for f in buys], marker="^",
               s=14, color=C["bid"], zorder=3, label="MM buys")
    ax.scatter([f.time for f in sells], [f.price * tick for f in sells], marker="v",
               s=14, color=C["ask"], zorder=3, label="MM sells")
    ax.set_ylabel("price ($)")
    ax.legend(loc="upper left", fontsize=8, ncols=3, frameon=False)
    _style(ax)

    ax = axes[1]
    ax.fill_between(t, inv, 0, step="post", color=C["as"], alpha=0.25)
    ax.plot(t, inv, color=C["as"], lw=1.2, drawstyle="steps-post")
    ax.axhline(0, color=C["mid"], lw=0.8)
    ax.set_ylabel("inventory (shares)")
    _style(ax)

    ax = axes[2]
    total = decomp.spread_capture_path + decomp.inventory_pnl_path
    ax.plot(t, total, color=C["total"], lw=1.6, label="total PnL")
    ax.plot(t, decomp.spread_capture_path, color=C["spread_capture"], lw=1.3,
            label="spread capture")
    ax.plot(t, decomp.inventory_pnl_path, color=C["inventory_pnl"], lw=1.3, ls="--",
            label="inventory PnL")
    ax.axhline(0, color=C["grid"], lw=0.8)
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("session time (s)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)

    return _save(fig, path)


def plot_pnl_distributions(
    results_by_strategy: dict[str, list[BacktestResult]], path: Path
) -> Path:
    """Final-PnL distribution per strategy across seeds (strip + summary), and
    the PnL paths behind them."""
    colors = {"avellaneda-stoikov": C["as"], "symmetric": C["sym"]}
    fig, (ax_paths, ax_dist) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]}
    )

    for name, results in results_by_strategy.items():
        color = colors.get(name, C["mid"])
        for r in results:
            ax_paths.plot(r.mark_times(), r.pnl_dollars(), color=color, lw=0.7, alpha=0.35)
        # direct label at the median endpoint
        finals = np.array([r.pnl_dollars()[-1] for r in results])
        ax_paths.annotate(name, xy=(results[0].mark_times()[-1], float(np.median(finals))),
                          color=color, fontsize=9, fontweight="bold",
                          xytext=(5, 0), textcoords="offset points")
    ax_paths.axhline(0, color=C["mid"], lw=0.8)
    ax_paths.set_xlabel("session time (s)")
    ax_paths.set_ylabel("mark-to-market PnL ($)")
    ax_paths.set_title("PnL paths across seeds", fontsize=10)
    _style(ax_paths)

    for i, (name, results) in enumerate(results_by_strategy.items()):
        color = colors.get(name, C["mid"])
        finals = np.array([r.pnl_dollars()[-1] for r in results])
        x = np.full(len(finals), i) + np.linspace(-0.12, 0.12, len(finals))
        ax_dist.scatter(x, finals, s=18, color=color, alpha=0.7, zorder=3)
        ax_dist.hlines(float(np.mean(finals)), i - 0.25, i + 0.25, color=color, lw=2.2)
    ax_dist.axhline(0, color=C["mid"], lw=0.8)
    ax_dist.set_xticks(range(len(results_by_strategy)),
                       list(results_by_strategy), fontsize=9)
    ax_dist.set_ylabel("final PnL ($)")
    ax_dist.set_title("final PnL by seed (bar = mean)", fontsize=10)
    _style(ax_dist)

    fig.tight_layout()
    return _save(fig, path)


def plot_gamma_sweep(
    gammas: list[float], summaries: list[StrategySummary], path: Path
) -> Path:
    """Risk-aversion sweep: PnL mean/std and inventory vs gamma."""
    mean_pnl = [s.mean_final_pnl for s in summaries]
    std_pnl = [s.std_final_pnl for s in summaries]
    mean_absq = [s.mean_abs_inventory for s in summaries]
    spread = [s.mean_quoted_spread_ticks for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        (axes[0], mean_pnl, std_pnl, "final PnL ($)", True),
        (axes[1], mean_absq, None, "mean |inventory| (shares)", False),
        (axes[2], spread, None, "mean quoted spread (ticks)", False),
    ]
    for ax, values, err, label, with_err in panels:
        if with_err and err is not None:
            ax.errorbar(gammas, values, yerr=err, color=C["as"], lw=1.4, marker="o",
                        ms=4, capsize=3, label="mean ± std across seeds")
            ax.legend(fontsize=8, frameon=False)
            ax.axhline(0, color=C["mid"], lw=0.8)
        else:
            ax.plot(gammas, values, color=C["as"], lw=1.4, marker="o", ms=4)
        ax.set_xscale("log")
        ax.set_xlabel(r"risk aversion $\gamma$ (1/\$)")
        ax.set_ylabel(label)
        _style(ax)
    fig.suptitle("Avellaneda-Stoikov sensitivity to risk aversion", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_sigma_sweep(
    sigmas: list[float],
    summaries_by_strategy: dict[str, list[StrategySummary]],
    path: Path,
) -> Path:
    """Volatility sweep: PnL, risk-adjusted performance, and the deepening of
    informed-counterparty markouts as sigma (hence informed edge) grows."""
    colors = {"avellaneda-stoikov": C["as"], "symmetric": C["sym"]}
    fig, (ax_pnl, ax_sh, ax_mo) = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for name, summaries in summaries_by_strategy.items():
        color = colors.get(name, C["mid"])
        ls = "-" if name == "avellaneda-stoikov" else "--"
        ax_pnl.errorbar(sigmas, [s.mean_final_pnl for s in summaries],
                        yerr=[s.std_final_pnl for s in summaries],
                        color=color, ls=ls, lw=1.4, marker="o", ms=4, capsize=3,
                        label=name)
        ax_sh.plot(sigmas, [s.episode_sharpe for s in summaries], color=color,
                   ls=ls, lw=1.4, marker="o", ms=4, label=name)
        ax_mo.plot(sigmas, [s.markout_5s_informed for s in summaries], color=color,
                   ls=ls, lw=1.4, marker="o", ms=4, label=f"{name} vs informed")
        ax_mo.plot(sigmas, [s.markout_5s_noise for s in summaries], color=color,
                   ls=":", lw=1.1, marker="s", ms=3, alpha=0.7,
                   label=f"{name} vs noise")
    for ax, ylabel in ((ax_pnl, "final PnL ($), mean ± std"),
                       (ax_sh, "episode Sharpe (across seeds)"),
                       (ax_mo, "5s markout ($/share)")):
        ax.axhline(0, color=C["mid"], lw=0.8)
        ax.set_xlabel(r"efficient-price volatility $\sigma$ ($/\sqrt{s}$)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.suptitle("Volatility as the adverse-selection axis: informed markouts deepen, "
                 "noise markouts stay flat", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_latency_sweep(
    intervals: list[float], summaries: list[StrategySummary], path: Path
) -> Path:
    """Quote-refresh latency sweep: what staleness costs, and through which
    channel (volume and repositioning, not per-fill markouts)."""
    fig, (ax_pnl, ax_vol, ax_mo) = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax_pnl.errorbar(intervals, [s.mean_final_pnl for s in summaries],
                    yerr=[s.std_final_pnl for s in summaries],
                    color=C["as"], lw=1.4, marker="o", ms=4, capsize=3,
                    label="mean ± std across seeds")
    ax_pnl.set_ylabel("final PnL ($)")
    ax_pnl.legend(fontsize=8, frameon=False)

    ax_vol.plot(intervals, [s.mean_volume for s in summaries], color=C["as"],
                lw=1.4, marker="o", ms=4)
    ax_vol.set_ylabel("volume traded (shares)")

    ax_mo.plot(intervals, [s.markout_5s_informed for s in summaries], color=C["as"],
               lw=1.4, marker="o", ms=4, label="vs informed")
    ax_mo.plot(intervals, [s.markout_5s_noise for s in summaries], color=C["as"],
               ls=":", lw=1.1, marker="s", ms=3, alpha=0.7, label="vs noise")
    ax_mo.set_ylabel("5s markout ($/share)")
    ax_mo.legend(fontsize=8, frameon=False)

    for ax in (ax_pnl, ax_vol, ax_mo):
        ax.axhline(0, color=C["mid"], lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("requote interval (s), log scale")
        _style(ax)
    fig.suptitle("The cost of quote latency: foregone volume and slow inventory "
                 "control, not deeper per-fill markouts", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_fill_intensity(fit: FillIntensityFit, path: Path) -> Path:
    """Empirical lambda(delta) and the fitted exponential, log scale."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fit.depths_dollars, fit.intensities, color=C["as"], s=24, zorder=3,
               label="empirical $\\lambda(\\delta)$")
    grid = np.linspace(fit.depths_dollars.min(), fit.depths_dollars.max(), 100)
    ax.plot(grid, fit.intensity(grid), color=C["ask"], lw=1.4, ls="--",
            label=f"fit: $A e^{{-k\\delta}}$  (A={fit.A:.2f}/s, k={fit.k:.0f}/\\$, "
                  f"$R^2$={fit.r_squared:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("quote depth $\\delta$ past mid (\\$)")
    ax.set_ylabel("fill intensity (1/s), log scale")
    ax.set_title("Calibration: fill intensity vs quote depth", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)
