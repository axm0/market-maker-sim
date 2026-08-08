"""Stochastic order flow: a latent efficient price plus two trader populations.

The market environment is built from three ingredients:

1. **Latent efficient price** ``p*(t)``: an arithmetic Brownian motion
   ``dp* = sigma dW`` (the same price model Avellaneda-Stoikov assume). It is
   *latent*: no agent trades at it directly, but informed traders observe it.

2. **Noise traders** (uninformed): arrive as a Poisson process and either send
   a market order in a uniformly random direction, or place a limit order at a
   geometrically distributed depth behind the opposite touch. Their limit
   orders have exponential lifetimes and are cancelled on expiry, which keeps
   the resting book stationary rather than growing without bound. Noise flow
   is symmetric and carries no information, so trading against it is
   profitable for a market maker on average.

   A fraction ``value_fraction`` of the limit-order flow comes from **value
   traders** who place their orders relative to ``p*`` instead of relative to
   the current touch. Without them, the resting book has no anchor: it drifts
   on its own noise, the mid's realized volatility becomes a large multiple of
   the fundamental volatility, and no quoting model referenced to the mid can
   survive. Value-trader liquidity keeps the book centred on fundamentals
   (its makers stand behind ``p*``, and when the book has drifted away their
   orders cross it and correct it), which is the economically sensible regime:
   real books track value because enough participants price off value.

3. **Informed traders**: arrive as a Poisson process, observe ``p*``, and send
   a market buy when ``p* > best_ask`` (sell when ``p* < best_bid``), i.e.
   exactly when a standing quote is mispriced relative to fundamentals. This
   is the mechanism of **adverse selection**: conditional on being lifted by
   an informed trader, the market maker's quote was on the wrong side of the
   true price, so the mid subsequently drifts against the position received.
   Informed trading is also what keeps the traded mid anchored to ``p*``:
   whenever the book drifts away from fundamentals, informed flow pushes it
   back.

The fill-rate consequence, which is what the Avellaneda-Stoikov model needs:
quotes placed deeper behind the touch are reached only by larger market orders
that walk through more of the book, so the fill intensity of a quote decays
with its depth ``delta`` — approximately like ``lambda(delta) = A e^(-k delta)``.
``calibration.py`` estimates A and k empirically from this flow.

All prices are integer ticks; the efficient price is float ticks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from .book import LimitOrderBook
from .orders import BookEvent, Side

__all__ = ["EfficientPrice", "FlowParams", "OrderFlow"]

# A scheduler accepts (absolute_time, action); the simulator's event loop will
# invoke action(time) at that simulation time. Injected by the simulator so the
# flow model can schedule its own limit-order expiries and next arrivals.
Scheduler = Callable[[float, Callable[[float], list[BookEvent]]], None]


@dataclass(frozen=True)
class FlowParams:
    """Arrival rates are per second of simulation time; sizes are in units
    (shares); depths are in ticks."""

    noise_market_rate: float = 2.0  # noise market orders per second
    noise_limit_rate: float = 5.0  # limit orders per second (noise + value mix)
    informed_rate: float = 2.0  # informed arrivals (trade only if mispriced)
    noise_size_mean: float = 5.0  # geometric mean size of noise orders
    informed_size_mean: float = 3.0  # geometric mean size of informed orders
    limit_offset_mean_ticks: float = 3.0  # geometric depth behind opposite touch
    limit_lifetime_mean: float = 20.0  # seconds until a resting limit order expires
    informed_edge_ticks: float = 0.0  # extra edge beyond the touch informed require
    value_fraction: float = 0.7  # share of limit flow placed relative to p*
    value_offset_mean_ticks: float = 2.0  # geometric depth behind p* for value orders
    seed_levels: int = 12  # ladder depth used to initialise the book
    seed_orders_per_level: int = 2


class EfficientPrice:
    """Lazily-sampled arithmetic Brownian motion in tick units.

    ``value_at(t)`` advances the path to time t (monotonically) and returns
    p*(t). Lazy sampling means the path is generated exactly at the event
    times the simulation asks for, with the correct sqrt(dt) scaling, instead
    of on a fixed grid.
    """

    def __init__(self, initial_ticks: float, sigma_ticks: float, rng: np.random.Generator):
        self._value = initial_ticks
        self._sigma = sigma_ticks
        self._time = 0.0
        self._rng = rng

    def value_at(self, t: float) -> float:
        if t < self._time:
            raise ValueError(f"time went backwards: {t} < {self._time}")
        dt = t - self._time
        if dt > 0:
            self._value += self._sigma * np.sqrt(dt) * self._rng.standard_normal()
            self._time = t
        return self._value


class OrderFlow:
    """Generates and executes the exogenous order flow against the book."""

    NOISE = "noise"
    INFORMED = "informed"
    VALUE = "value"

    def __init__(
        self,
        book: LimitOrderBook,
        efficient_price: EfficientPrice,
        params: FlowParams,
        rng: np.random.Generator,
        schedule: Scheduler,
    ):
        self.book = book
        self.efficient = efficient_price
        self.p = params
        self.rng = rng
        self.schedule = schedule
        # Fallback reference price for limit placement when a book side is
        # empty (rare after seeding, but must not crash).
        self._last_mid: float = efficient_price.value_at(0.0)

    # ----------------------------------------------------------------- set-up

    def start(self, time: float = 0.0) -> None:
        """Seed the book with a resting-liquidity ladder and schedule the first
        arrival of each Poisson stream."""
        mid0 = round(self._last_mid)
        for level in range(1, self.p.seed_levels + 1):
            for _ in range(self.p.seed_orders_per_level):
                for side, price in ((Side.BUY, mid0 - level), (Side.SELL, mid0 + level)):
                    qty = self._geometric(self.p.noise_size_mean)
                    order_id, _ = self.book.submit_limit(self.NOISE, side, price, qty, time)
                    self._schedule_expiry(order_id, time)
        self._schedule_arrival(time, self.p.noise_market_rate, self._noise_market)
        self._schedule_arrival(time, self.p.noise_limit_rate, self._noise_limit)
        self._schedule_arrival(time, self.p.informed_rate, self._informed)

    # ---------------------------------------------------------------- arrivals

    def _noise_market(self, t: float) -> list[BookEvent]:
        self._reschedule(t, self.p.noise_market_rate, self._noise_market)
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        qty = self._geometric(self.p.noise_size_mean)
        _, events = self.book.submit_market(self.NOISE, side, qty, t)
        self._update_last_mid()
        return events

    def _noise_limit(self, t: float) -> list[BookEvent]:
        self._reschedule(t, self.p.noise_limit_rate, self._noise_limit)
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        qty = self._geometric(self.p.noise_size_mean)
        if self.rng.random() < self.p.value_fraction:
            # Value trader: prices off the fundamental, not the book. If the
            # book has drifted away from p*, this order may cross and correct
            # it (the engine handles marketable limits) — that is the
            # anchoring mechanism, so no passivity clip is applied.
            owner = self.VALUE
            p_star = self.efficient.value_at(t)
            base = int(np.floor(p_star)) if side is Side.BUY else int(np.ceil(p_star))
            offset = self._geometric_offset(self.p.value_offset_mean_ticks)
            price = base - side.sign * offset
        else:
            owner = self.NOISE
            price = self._limit_price(side)
        order_id, events = self.book.submit_limit(owner, side, price, qty, t)
        self._schedule_expiry(order_id, t)
        self._update_last_mid()
        return events

    def _informed(self, t: float) -> list[BookEvent]:
        self._reschedule(t, self.p.informed_rate, self._informed)
        p_star = self.efficient.value_at(t)
        best_ask = self.book.best_ask()
        best_bid = self.book.best_bid()
        edge = self.p.informed_edge_ticks
        events: list[BookEvent] = []
        if best_ask is not None and p_star > best_ask + edge:
            qty = self._geometric(self.p.informed_size_mean)
            _, events = self.book.submit_market(self.INFORMED, Side.BUY, qty, t)
        elif best_bid is not None and p_star < best_bid - edge:
            qty = self._geometric(self.p.informed_size_mean)
            _, events = self.book.submit_market(self.INFORMED, Side.SELL, qty, t)
        self._update_last_mid()
        return events

    def _expire(self, order_id: int, t: float) -> list[BookEvent]:
        event = self.book.cancel(order_id, t)  # None if already filled: fine
        return [event] if event is not None else []

    # ----------------------------------------------------------------- helpers

    def _limit_price(self, side: Side) -> int:
        """Price a noise limit order: a geometric number of ticks behind the
        opposite touch (so it never crosses), falling back to the last known
        mid when the opposite side is empty. Placing relative to the *current*
        book makes resting liquidity follow the traded price."""
        offset = self._geometric_offset(self.p.limit_offset_mean_ticks)
        opposite_best = (
            self.book.best_ask() if side is Side.BUY else self.book.best_bid()
        )
        if opposite_best is not None:
            reference = opposite_best - side.sign  # one tick inside the touch
        else:
            reference = round(self._last_mid)
        return reference - side.sign * offset

    def _geometric(self, mean: float) -> int:
        return int(self.rng.geometric(1.0 / mean))

    def _geometric_offset(self, mean: float) -> int:
        """Geometric on {0, 1, 2, ...} with the given mean."""
        return int(self.rng.geometric(1.0 / (1.0 + mean))) - 1

    def _schedule_expiry(self, order_id: int, now: float) -> None:
        lifetime = self.rng.exponential(self.p.limit_lifetime_mean)
        self.schedule(now + lifetime, partial(self._expire, order_id))

    def _schedule_arrival(
        self, now: float, rate: float, action: Callable[[float], list[BookEvent]]
    ) -> None:
        if rate > 0:
            self.schedule(now + self.rng.exponential(1.0 / rate), action)

    def _reschedule(
        self, now: float, rate: float, action: Callable[[float], list[BookEvent]]
    ) -> None:
        self._schedule_arrival(now, rate, action)

    def _update_last_mid(self) -> None:
        mid = self.book.mid()
        if mid is not None:
            self._last_mid = mid
