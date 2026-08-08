"""Market-making strategies: Avellaneda-Stoikov and a naive symmetric baseline.

Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book",
solves the optimal quoting problem for a market maker with CARA utility
``-exp(-gamma * W_T)`` over terminal wealth, when the mid price follows an
arithmetic Brownian motion with volatility ``sigma`` and the probability of a
quote at distance ``delta`` from the mid being hit follows an exponential
intensity ``lambda(delta) = A exp(-k delta)``. Two closed-form quantities come
out (paper eqs. 29-30):

**Reservation price** — the price at which the agent is indifferent to a small
trade, given current inventory q::

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

The intuition: holding q units exposes the agent to variance
``q^2 sigma^2 (T-t)`` in terminal wealth. A long agent (q > 0) values the
asset *below* the mid — it shifts both quotes down, making its ask more
aggressive (more likely to sell, shedding inventory) and its bid less
aggressive (less likely to buy more). This *inventory skew* is the control
that mean-reverts inventory toward zero without ever crossing the spread.

**Optimal total spread** — quotes are placed symmetrically around r::

    delta_bid + delta_ask = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

The first term is inventory-risk compensation (grows with volatility and
remaining horizon); the second balances the margin-per-fill against the fill
intensity implied by k: when fills decay fast with depth (large k), the
optimal spread tightens.

Practical adaptations for a discrete-tick order book (each is a deliberate,
documented departure from the idealised continuous model):

* Prices are rounded *outward* to the tick grid (bid down, ask up), never
  inward, so the quoted spread is never tighter than the model's.
* Quotes are clipped to remain **passive** (at least one tick away from the
  opposite touch). The raw model can ask for a crossing quote when inventory
  is large; a quoting strategy should skew harder, not silently become a
  taker.
* **Hard position limits**: quoting on a side stops entirely when |q| would
  exceed ``max_inventory``. The paper's soft skew already mean-reverts
  inventory, but a hard limit bounds worst-case exposure — standard risk
  practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .book import LimitOrderBook

__all__ = ["AvellanedaStoikov", "MarketMaker", "Quote", "SymmetricQuoter"]


@dataclass(frozen=True, slots=True)
class Quote:
    """Desired two-sided quote in ticks. A None price means: do not quote that
    side (e.g. position limit reached)."""

    bid_price: int | None
    ask_price: int | None
    size: int


class MarketMaker(Protocol):
    name: str

    def quote(self, t: float, mid_ticks: float, inventory: int, book: LimitOrderBook) -> Quote:
        """Return the desired quote at time t given the current mid (in ticks,
        possibly the last known mid if one book side is empty) and current
        signed inventory."""
        ...


def _clip_passive(
    bid: int | None, ask: int | None, book: LimitOrderBook
) -> tuple[int | None, int | None]:
    """Force quotes to be passive: bid strictly below the best ask and ask
    strictly above the best bid. Keeps the strategy a pure liquidity provider.
    Preserves bid < ask (clipping only ever lowers the bid / raises the ask)."""
    best_bid, best_ask = book.best_bid(), book.best_ask()
    if bid is not None and best_ask is not None:
        bid = min(bid, best_ask - 1)
    if ask is not None and best_bid is not None:
        ask = max(ask, best_bid + 1)
    return bid, ask


def _apply_position_limit(
    bid: int | None, ask: int | None, inventory: int, max_inventory: int, size: int
) -> tuple[int | None, int | None]:
    """Stop quoting a side when a full fill there could push |inventory| past
    the hard limit."""
    if inventory + size > max_inventory:
        bid = None
    if inventory - size < -max_inventory:
        ask = None
    return bid, ask


@dataclass
class AvellanedaStoikov:
    """Finite-horizon Avellaneda-Stoikov quoting.

    Parameters are in *dollar* units, faithful to the paper; conversion to
    ticks happens only when emitting the quote.

    gamma : risk aversion (1/$). Higher gamma -> stronger skew, wider spread.
    kappa : fill-intensity decay k (1/$) of lambda(delta) = A exp(-k delta).
            Estimate it from the flow with calibration.py rather than guessing.
    sigma : mid-price volatility ($ per sqrt-second) — the same sigma that
            drives the simulated efficient price, or an estimate of it.
    horizon : trading session length T in seconds; t is measured from the
            session start (the simulator passes session-relative time).
    """

    gamma: float
    kappa: float
    sigma: float
    horizon: float
    tick_size: float
    size: int = 5
    max_inventory: int = 50
    name: str = "avellaneda-stoikov"

    def reservation_price(self, mid_dollars: float, inventory: int, t: float) -> float:
        tau = max(self.horizon - t, 0.0)
        return mid_dollars - inventory * self.gamma * self.sigma**2 * tau

    def optimal_total_spread(self, t: float) -> float:
        tau = max(self.horizon - t, 0.0)
        return self.gamma * self.sigma**2 * tau + (2.0 / self.gamma) * math.log(
            1.0 + self.gamma / self.kappa
        )

    def quote(self, t: float, mid_ticks: float, inventory: int, book: LimitOrderBook) -> Quote:
        mid = mid_ticks * self.tick_size
        r = self.reservation_price(mid, inventory, t)
        half_spread = self.optimal_total_spread(t) / 2.0
        # Outward rounding to the tick grid: never quote tighter than optimal.
        raw_bid = math.floor((r - half_spread) / self.tick_size)
        raw_ask = math.ceil((r + half_spread) / self.tick_size)
        if raw_ask <= raw_bid:  # degenerate only if the model spread is < 1 tick
            raw_ask = raw_bid + 1
        bid, ask = _clip_passive(raw_bid, raw_ask, book)
        bid, ask = _apply_position_limit(bid, ask, inventory, self.max_inventory, self.size)
        return Quote(bid_price=bid, ask_price=ask, size=self.size)


@dataclass
class SymmetricQuoter:
    """Naive baseline: a fixed half-spread around the mid, no inventory skew.

    Identical plumbing to AvellanedaStoikov (same size, same hard position
    limits, same passivity clipping), so any performance difference in the
    backtest is attributable to the quoting model itself, not to mechanics.
    Its failure mode is the point of the comparison: with no skew, inventory
    follows a random walk between the hard limits, so it carries much more
    inventory risk and keeps quoting into adverse flow.
    """

    half_spread_ticks: int
    size: int = 5
    max_inventory: int = 50
    name: str = "symmetric"

    def quote(self, t: float, mid_ticks: float, inventory: int, book: LimitOrderBook) -> Quote:
        raw_bid = math.floor(mid_ticks - self.half_spread_ticks)
        raw_ask = math.ceil(mid_ticks + self.half_spread_ticks)
        if raw_ask <= raw_bid:
            raw_ask = raw_bid + 1
        bid, ask = _clip_passive(raw_bid, raw_ask, book)
        bid, ask = _apply_position_limit(bid, ask, inventory, self.max_inventory, self.size)
        return Quote(bid_price=bid, ask_price=ask, size=self.size)
