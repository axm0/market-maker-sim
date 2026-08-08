"""Reproducible experiments and figures.

Run ``mm-sim all`` (or ``python -m market_maker_sim.experiments all``) to
reproduce everything in docs/RESULTS.md: calibration, a sample episode per
strategy, the A-S vs baseline comparison across seeds, the risk-aversion
sweep, and the informed-flow sweep. Figures land in docs/figures/, and the
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
from dataclasses import replace
from pathlib import Path

from .backtest import BacktestResult, SimConfig, run_backtest, run_flow_only
from .calibration import calibrate_fill_intensity, estimate_sigma
from .metrics import StrategySummary, summarize
from .strategy import AvellanedaStoikov, SymmetricQuoter

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"

N_SEEDS = 24
BASE_SEED = 7
GAMMA_DEFAULT = 0.01
GAMMA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
SIGMA_GRID = [0.005, 0.01, 0.02, 0.03]
QUOTE_SIZE = 5
MAX_INVENTORY = 30
BASELINE_HALF_SPREAD_TICKS = 3


def default_config(seed: int = BASE_SEED) -> SimConfig:
    return SimConfig(seed=seed)


def calibrated_inputs(cfg: SimConfig, n_episodes: int = 5):
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


def run_seeds(cfg: SimConfig, strategy_factory, n_seeds: int = N_SEEDS) -> list[BacktestResult]:
    """Run n_seeds independent episodes. strategy_factory is called per episode
    so no state can leak across runs; seed i is shared across strategies
    (common random numbers)."""
    return [
        run_backtest(replace(cfg, seed=BASE_SEED + i), strategy_factory())
        for i in range(n_seeds)
    ]


def print_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
        *("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows),
    ]
    table = "\n".join(lines)
    print(table)
    return table


def _summary_json(s: StrategySummary) -> dict:
    return {k: v for k, v in s.__dict__.items()}


# ------------------------------------------------------------------ commands


def cmd_calibrate(save: bool = True):
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


def cmd_episode(sigma_hat: float, k_hat: float, seed: int = BASE_SEED):
    from .plotting import plot_episode

    cfg = default_config(seed)
    for factory, fname in [
        (lambda: make_avellaneda_stoikov(cfg, GAMMA_DEFAULT, sigma_hat, k_hat),
         "episode_avellaneda_stoikov.png"),
        (make_baseline, "episode_symmetric.png"),
    ]:
        result = run_backtest(cfg, factory())
        path = plot_episode(result, FIGURES / fname)
        print(f"  {result.strategy_name}: final PnL "
              f"${result.pnl_dollars()[-1]:.2f}, {len(result.fills)} fills -> {path}")


def cmd_compare(sigma_hat: float, k_hat: float) -> dict[str, StrategySummary]:
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
    path = plot_pnl_distributions(results, FIGURES / "pnl_distributions.png")
    print(f"  figure: {path}")
    return summaries


def cmd_sweep_gamma(sigma_hat: float, k_hat: float) -> list[StrategySummary]:
    from .plotting import plot_gamma_sweep

    cfg = default_config()
    print(f"Sweeping gamma over {GAMMA_GRID} ({N_SEEDS} seeds each)...")
    summaries = []
    for gamma in GAMMA_GRID:
        rs = run_seeds(cfg, lambda g=gamma: make_avellaneda_stoikov(cfg, g, sigma_hat, k_hat))
        s = summarize(rs, max_inventory=MAX_INVENTORY)
        summaries.append(s)
        print(f"  gamma={gamma:<5}: PnL {s.mean_final_pnl:8.2f} ± {s.std_final_pnl:6.2f}, "
              f"mean|q| {s.mean_abs_inventory:5.1f}, spread {s.mean_quoted_spread_ticks:.2f} ticks")
    path = plot_gamma_sweep(GAMMA_GRID, summaries, FIGURES / "gamma_sweep.png")
    print(f"  figure: {path}")
    return summaries


def cmd_sweep_sigma() -> dict[str, list]:
    """Volatility sweep — the adverse-selection axis. Raising sigma gives
    informed traders more edge per trade (the efficient price gets further
    through stale quotes before the book corrects), so markouts against
    informed flow deepen and inventory risk grows."""
    from .plotting import plot_sigma_sweep

    print(f"Sweeping sigma over {SIGMA_GRID} "
          f"({N_SEEDS} seeds each, re-calibrating per market)...")
    out: dict[str, list] = {"avellaneda-stoikov": [], "symmetric": []}
    for sigma in SIGMA_GRID:
        cfg = replace(default_config(), sigma=sigma)
        # Both the fill-intensity surface and realized volatility change with
        # the market regime, so the A-S inputs are re-estimated per point.
        sigma_hat, fit = calibrated_inputs(cfg, n_episodes=3)
        for name, factory in [
            ("avellaneda-stoikov",
             lambda c=cfg, s=sigma_hat, k=fit.k:
                 make_avellaneda_stoikov(c, GAMMA_DEFAULT, s, k)),
            ("symmetric", make_baseline),
        ]:
            rs = run_seeds(cfg, factory)
            s = summarize(rs, max_inventory=MAX_INVENTORY)
            out[name].append(s)
            print(f"  sigma={sigma:<6} {name:<20}: PnL {s.mean_final_pnl:8.2f} "
                  f"± {s.std_final_pnl:6.2f}, Sharpe {s.episode_sharpe:5.2f}, "
                  f"markout(5s) inf {s.markout_5s_informed:+.4f} $/sh")
    path = plot_sigma_sweep(SIGMA_GRID, out, FIGURES / "sigma_sweep.png")
    print(f"  figure: {path}")
    return out


def cmd_all():
    t0 = _time.time()
    sigma_hat, fit = cmd_calibrate()
    print()
    cmd_episode(sigma_hat, fit.k)
    print()
    compare = cmd_compare(sigma_hat, fit.k)
    print()
    gamma = cmd_sweep_gamma(sigma_hat, fit.k)
    print()
    sigma_sweep = cmd_sweep_sigma()

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "calibration": {"sigma_hat": sigma_hat, "A": fit.A, "k": fit.k,
                        "r_squared": fit.r_squared},
        "compare": {k: _summary_json(v) for k, v in compare.items()},
        "gamma_sweep": {"grid": GAMMA_GRID,
                        "summaries": [_summary_json(s) for s in gamma]},
        "sigma_sweep": {"grid": SIGMA_GRID,
                        "summaries": {k: [_summary_json(s) for s in v]
                                      for k, v in sigma_sweep.items()}},
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
                                            "sweep-gamma", "sweep-sigma"])
    args = parser.parse_args()
    if args.command == "all":
        cmd_all()
    elif args.command == "calibrate":
        cmd_calibrate()
    else:
        sigma_hat, fit = cmd_calibrate(save=False)
        {
            "episode": lambda: cmd_episode(sigma_hat, fit.k),
            "compare": lambda: cmd_compare(sigma_hat, fit.k),
            "sweep-gamma": lambda: cmd_sweep_gamma(sigma_hat, fit.k),
            "sweep-sigma": cmd_sweep_sigma,
        }[args.command]()


if __name__ == "__main__":
    main()
