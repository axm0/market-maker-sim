"""Reproducible experiments and figures.

Run ``mm-sim all`` (or ``python -m market_maker_sim.experiments all``) to
reproduce everything in docs/RESULTS.md: calibration, a sample episode per
strategy, the A-S vs baseline comparison across seeds, the risk-aversion
sweep, the volatility sweep, and the latency sweep. Figures land in docs/figures/, and the
numeric summaries in results/summary.json.

Experimental hygiene:

* Strategies are compared on **common random numbers**: episode i of every
  strategy uses the same seed, hence an identical market realization (the
  exogenous flow only differs through interaction with the MM's own quotes).
  This makes paired comparisons much sharper than independent draws.
* The strategy never sees generator truth: sigma is estimated from flow-only
  runs and k from the calibration fit — exactly what a live desk would have.
* Every sweep point that changes the market (informed rate) gets its own
  calibration, because the fill-intensity surface it quotes against changes.
"""

from __future__ import annotations

import argparse
import json
import time as _time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

from .backtest import BacktestResult, SimConfig, run_backtest, run_flow_only
from .calibration import FillIntensityFit, calibrate_fill_intensity, estimate_sigma
from .metrics import PairedComparison, StrategySummary, paired_comparison, summarize
from .strategy import AvellanedaStoikov, MarketMaker, SymmetricQuoter

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"

N_SEEDS = 24
BASE_SEED = 7
GAMMA_DEFAULT = 0.01
GAMMA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
SIGMA_GRID = [0.005, 0.01, 0.02, 0.03]
REQUOTE_GRID = [0.1, 0.25, 1.0, 2.5, 5.0, 10.0]
QUOTE_SIZE = 5
MAX_INVENTORY = 30
BASELINE_HALF_SPREAD_TICKS = 3


def default_config(seed: int = BASE_SEED) -> SimConfig:
    return SimConfig(seed=seed)


def calibrated_inputs(
    cfg: SimConfig, n_episodes: int = 5
) -> tuple[float, FillIntensityFit]:
    """Estimate (sigma_hat, fill-intensity fit) from flow-only runs of this
    market configuration."""
    fit = calibrate_fill_intensity(cfg, n_episodes=n_episodes)
    flow_result, _ = run_flow_only(replace(cfg, seed=cfg.seed + 999))
    sigma_hat = estimate_sigma(flow_result)
    return sigma_hat, fit


def make_avellaneda_stoikov(
    cfg: SimConfig, gamma: float, sigma_hat: float, k_hat: float
) -> AvellanedaStoikov:
    return AvellanedaStoikov(
        gamma=gamma,
        kappa=k_hat,
        sigma=sigma_hat,
        horizon=cfg.horizon,
        tick_size=cfg.tick_size,
        size=QUOTE_SIZE,
        max_inventory=MAX_INVENTORY,
    )


def make_baseline() -> SymmetricQuoter:
    return SymmetricQuoter(
        half_spread_ticks=BASELINE_HALF_SPREAD_TICKS,
        size=QUOTE_SIZE,
        max_inventory=MAX_INVENTORY,
    )


def run_seeds(
    cfg: SimConfig,
    strategy_factory: Callable[[], MarketMaker],
    n_seeds: int = N_SEEDS,
) -> list[BacktestResult]:
    """Run n_seeds independent episodes. strategy_factory is called per episode
    so no state can leak across runs; seed i is shared across strategies
    (common random numbers)."""
    return [
        run_backtest(replace(cfg, seed=BASE_SEED + i), strategy_factory())
        for i in range(n_seeds)
    ]


def print_table(rows: list[dict[str, float | str | int]]) -> str:
    keys = list(rows[0])
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
        *("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows),
    ]
    table = "\n".join(lines)
    print(table)
    return table


def _summary_json(s: StrategySummary) -> dict[str, object]:
    return dict(s.__dict__)


# ------------------------------------------------------------------ commands


def cmd_calibrate(save: bool = True) -> tuple[float, FillIntensityFit]:
    cfg = default_config()
    print("Calibrating fill intensity and volatility from flow-only runs...")
    sigma_hat, fit = calibrated_inputs(cfg)
    print(f"  sigma_hat = {sigma_hat:.4f} $/sqrt(s)   (generator truth: {cfg.sigma})")
    print(f"  A = {fit.A:.2f} fills/s, k = {fit.k:.1f} /$   (R^2 = {fit.r_squared:.3f})")
    if save:
        from .plotting import plot_fill_intensity

        path = plot_fill_intensity(fit, FIGURES / "fill_intensity.png")
        print(f"  figure: {path}")
    return sigma_hat, fit


def cmd_episode(sigma_hat: float, k_hat: float, seed: int = BASE_SEED) -> None:
    from .plotting import plot_episode

    cfg = default_config(seed)
    factories: list[tuple[Callable[[], MarketMaker], str]] = [
        (partial(make_avellaneda_stoikov, cfg, GAMMA_DEFAULT, sigma_hat, k_hat),
         "episode_avellaneda_stoikov.png"),
        (make_baseline, "episode_symmetric.png"),
    ]
    for factory, fname in factories:
        result = run_backtest(cfg, factory())
        path = plot_episode(result, FIGURES / fname)
        print(f"  {result.strategy_name}: final PnL "
              f"${result.pnl_dollars()[-1]:.2f}, {len(result.fills)} fills -> {path}")


def cmd_compare(
    sigma_hat: float, k_hat: float
) -> tuple[dict[str, StrategySummary], PairedComparison]:
    from .plotting import plot_pnl_distributions

    cfg = default_config()
    print(f"Comparing strategies over {N_SEEDS} seeds (common random numbers)...")
    results = {
        "avellaneda-stoikov": run_seeds(
            cfg, lambda: make_avellaneda_stoikov(cfg, GAMMA_DEFAULT, sigma_hat, k_hat)
        ),
        "symmetric": run_seeds(cfg, make_baseline),
    }
    summaries = {
        name: summarize(rs, max_inventory=MAX_INVENTORY) for name, rs in results.items()
    }
    print_table([s.row() for s in summaries.values()])
    for name, s in summaries.items():
        print(f"  {name}: 5s markout $/share — all {s.markout_5s_all:+.4f}, "
              f"vs informed {s.markout_5s_informed:+.4f}, vs noise {s.markout_5s_noise:+.4f}")
    paired = paired_comparison(results["avellaneda-stoikov"], results["symmetric"])
    print(f"  paired (A-S minus symmetric, {paired.n_pairs} common-random-number pairs): "
          f"mean diff ${paired.mean_diff:+.2f}, t = {paired.t_stat:.2f}, "
          f"bootstrap 95% CI [{paired.ci_low:+.2f}, {paired.ci_high:+.2f}], "
          f"A-S wins {paired.share_episodes_a_wins:.0%} of episodes")
    path = plot_pnl_distributions(results, FIGURES / "pnl_distributions.png")
    print(f"  figure: {path}")
    return summaries, paired


def cmd_sweep_gamma(sigma_hat: float, k_hat: float) -> list[StrategySummary]:
    from .plotting import plot_gamma_sweep

    cfg = default_config()
    print(f"Sweeping gamma over {GAMMA_GRID} ({N_SEEDS} seeds each)...")
    summaries = []
    for gamma in GAMMA_GRID:
        rs = run_seeds(cfg, partial(make_avellaneda_stoikov, cfg, gamma, sigma_hat, k_hat))
        s = summarize(rs, max_inventory=MAX_INVENTORY)
        summaries.append(s)
        print(f"  gamma={gamma:<5}: PnL {s.mean_final_pnl:8.2f} ± {s.std_final_pnl:6.2f}, "
              f"mean|q| {s.mean_abs_inventory:5.1f}, spread {s.mean_quoted_spread_ticks:.2f} ticks")
    path = plot_gamma_sweep(GAMMA_GRID, summaries, FIGURES / "gamma_sweep.png")
    print(f"  figure: {path}")
    return summaries


def cmd_sweep_sigma() -> dict[str, list[StrategySummary]]:
    """Volatility sweep — the adverse-selection axis. Raising sigma gives
    informed traders more edge per trade (the efficient price gets further
    through stale quotes before the book corrects), so markouts against
    informed flow deepen and inventory risk grows."""
    from .plotting import plot_sigma_sweep

    print(f"Sweeping sigma over {SIGMA_GRID} "
          f"({N_SEEDS} seeds each, re-calibrating per market)...")
    out: dict[str, list[StrategySummary]] = {"avellaneda-stoikov": [], "symmetric": []}
    for sigma in SIGMA_GRID:
        cfg = replace(default_config(), sigma=sigma)
        # Both the fill-intensity surface and realized volatility change with
        # the market regime, so the A-S inputs are re-estimated per point.
        sigma_hat, fit = calibrated_inputs(cfg, n_episodes=3)
        factories: list[tuple[str, Callable[[], MarketMaker]]] = [
            ("avellaneda-stoikov",
             partial(make_avellaneda_stoikov, cfg, GAMMA_DEFAULT, sigma_hat, fit.k)),
            ("symmetric", make_baseline),
        ]
        for name, factory in factories:
            rs = run_seeds(cfg, factory)
            s = summarize(rs, max_inventory=MAX_INVENTORY)
            out[name].append(s)
            print(f"  sigma={sigma:<6} {name:<20}: PnL {s.mean_final_pnl:8.2f} "
                  f"± {s.std_final_pnl:6.2f}, Sharpe {s.episode_sharpe:5.2f}, "
                  f"markout(5s) inf {s.markout_5s_informed:+.4f} $/sh")
    path = plot_sigma_sweep(SIGMA_GRID, out, FIGURES / "sigma_sweep.png")
    print(f"  figure: {path}")
    return out


def cmd_sweep_latency(sigma_hat: float, k_hat: float) -> list[StrategySummary]:
    """Quote-refresh latency sweep. The requote interval is the harness's
    latency knob: a slower loop reacts later to mid moves and inventory
    changes. Measures what quote staleness actually costs in this flow."""
    from .plotting import plot_latency_sweep

    cfg = default_config()
    print(f"Sweeping requote interval over {REQUOTE_GRID}s ({N_SEEDS} seeds each)...")
    summaries = []
    for interval in REQUOTE_GRID:
        cfg_i = replace(cfg, requote_interval=interval)
        rs = run_seeds(cfg_i,
                       lambda: make_avellaneda_stoikov(cfg, GAMMA_DEFAULT,
                                                       sigma_hat, k_hat))
        s = summarize(rs, max_inventory=MAX_INVENTORY)
        summaries.append(s)
        print(f"  dt={interval:<5}: PnL {s.mean_final_pnl:7.2f} ± {s.std_final_pnl:5.2f}, "
              f"volume {s.mean_volume:6.0f}, markout(5s) {s.markout_5s_all:+.4f} $/sh")
    path = plot_latency_sweep(REQUOTE_GRID, summaries, FIGURES / "latency_sweep.png")
    print(f"  figure: {path}")
    return summaries


def cmd_all() -> None:
    t0 = _time.time()
    sigma_hat, fit = cmd_calibrate()
    print()
    cmd_episode(sigma_hat, fit.k)
    print()
    compare, paired = cmd_compare(sigma_hat, fit.k)
    print()
    gamma = cmd_sweep_gamma(sigma_hat, fit.k)
    print()
    sigma_sweep = cmd_sweep_sigma()
    print()
    latency = cmd_sweep_latency(sigma_hat, fit.k)

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "calibration": {"sigma_hat": sigma_hat, "A": fit.A, "k": fit.k,
                        "r_squared": fit.r_squared},
        "compare": {k: _summary_json(v) for k, v in compare.items()},
        "paired": paired.__dict__,
        "gamma_sweep": {"grid": GAMMA_GRID,
                        "summaries": [_summary_json(s) for s in gamma]},
        "sigma_sweep": {"grid": SIGMA_GRID,
                        "summaries": {k: [_summary_json(s) for s in v]
                                      for k, v in sigma_sweep.items()}},
        "latency_sweep": {"grid": REQUOTE_GRID,
                          "summaries": [_summary_json(s) for s in latency]},
        "config": {"n_seeds": N_SEEDS, "gamma_default": GAMMA_DEFAULT,
                   "quote_size": QUOTE_SIZE, "max_inventory": MAX_INVENTORY,
                   "baseline_half_spread_ticks": BASELINE_HALF_SPREAD_TICKS},
    }
    path = RESULTS / "summary.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSummary data: {path}")
    print(f"Total runtime: {_time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="market-maker-sim experiments")
    parser.add_argument("command", choices=["all", "calibrate", "episode", "compare",
                                            "sweep-gamma", "sweep-sigma",
                                            "sweep-latency"])
    args = parser.parse_args()
    if args.command == "all":
        cmd_all()
    elif args.command == "calibrate":
        cmd_calibrate()
    else:
        sigma_hat, fit = cmd_calibrate(save=False)
        commands: dict[str, Callable[[], object]] = {
            "episode": lambda: cmd_episode(sigma_hat, fit.k),
            "compare": lambda: cmd_compare(sigma_hat, fit.k),
            "sweep-gamma": lambda: cmd_sweep_gamma(sigma_hat, fit.k),
            "sweep-sigma": cmd_sweep_sigma,
            "sweep-latency": lambda: cmd_sweep_latency(sigma_hat, fit.k),
        }
        commands[args.command]()


if __name__ == "__main__":
    main()
