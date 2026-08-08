"""market-maker-sim: a limit-order-book market-making simulator.

A price-time-priority matching engine, a stochastic order-flow model with
informed and noise traders around a latent efficient price, the
Avellaneda-Stoikov quoting strategy, and a risk-adjusted backtest harness.
"""

from .backtest import BacktestResult, SimConfig, run_backtest, run_flow_only
from .book import LimitOrderBook
from .calibration import calibrate_fill_intensity, estimate_sigma
from .flow import EfficientPrice, FlowParams, OrderFlow
from .metrics import decompose_pnl, episode_metrics, markouts, summarize
from .orders import Accepted, Cancelled, Fill, Side
from .strategy import AvellanedaStoikov, Quote, SymmetricQuoter

__version__ = "0.1.0"

__all__ = [
    "Accepted",
    "AvellanedaStoikov",
    "BacktestResult",
    "Cancelled",
    "EfficientPrice",
    "Fill",
    "FlowParams",
    "LimitOrderBook",
    "OrderFlow",
    "Quote",
    "Side",
    "SimConfig",
    "SymmetricQuoter",
    "calibrate_fill_intensity",
    "decompose_pnl",
    "episode_metrics",
    "estimate_sigma",
    "markouts",
    "run_backtest",
    "run_flow_only",
    "summarize",
]
