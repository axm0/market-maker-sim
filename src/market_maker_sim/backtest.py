"""Event-driven backtest harness.

A single priority queue drives everything: flow arrivals, noise-limit-order
expiries, market-maker requotes, and mark-to-market sampling are all events
``(time, seq, action)``. The seq tiebreaker makes simultaneous events execute
in schedule order, so runs are exactly reproducible for a given seed.

Timeline: the flow model runs alone for ``warmup`` seconds so the book reaches
its stationary depth profile before the strategy starts quoting. The strategy
then trades for ``horizon`` seconds; strategy time (the ``t`` passed to
``quote``, which Avellaneda-Stoikov measures its ``T - t`` against) is relative
to the end of the warmup. Marks and fills are likewise recorded with
session-relative times.

Quote management is cancel-replace with *queue-priority preservation*: a live
order is left in place if the strategy still wants the same price, because
cancelling and re-adding would send it to the back of the FIFO queue at that
price. Time priority is a real asset for a passive market maker — a quote at
the front of the queue fills first, before the price level is exhausted — so
churning it away would systematically understate fill rates.

Accounting is exact: cash is kept in integer tick-units (price ticks x qty),
so realized cash flows carry no floating-point error; dollars appear only in
reporting.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .book import LimitOrderBook
from .flow import EfficientPrice, FlowParams, OrderFlow
from .orders import BookEvent, Fill, Side
from .strategy import MarketMaker

__all__ = ["BacktestResult", "MMFill", "Mark", "SimConfig", "run_backtest", "run_flow_only"]

MM_OWNER = "mm"


@dataclass(frozen=True)
class SimConfig:
    horizon: float = 600.0  # seconds of live trading
    warmup: float = 60.0  # flow-only book warm-up before trading starts
    tick_size: float = 0.01  # dollars per tick
    initial_price: float = 100.0  # dollars
    sigma: float = 0.01  # efficient-price volatility, dollars per sqrt-second
    requote_interval: float = 0.25  # seconds between strategy quote updates
    mark_interval: float = 0.5  # seconds between mark-to-market samples
    seed: int = 0
    flow: FlowParams = field(default_factory=FlowParams)

    @property
    def sigma_ticks(self) -> float:
        return self.sigma / self.tick_size

    @property
    def initial_price_ticks(self) -> float:
        return self.initial_price / self.tick_size


@dataclass(frozen=True, slots=True)
class MMFill:
    """One execution of the market maker's quote, from the MM's perspective."""

    time: float  # session-relative
    side: Side  # side of the MM's order (BUY = we bought)
    price: int  # ticks
    qty: int
    pre_trade_mid: float  # ticks; mid just before the aggressor arrived
    taker_owner: str  # "noise" or "informed" — who hit us

    @property
    def signed_qty(self) -> int:
        return self.side.sign * self.qty


@dataclass(frozen=True, slots=True)
class Mark:
    """A mark-to-market sample on the regular grid."""

    time: float  # session-relative
    mid: float  # ticks
    efficient: float  # ticks; latent p*, recorded for diagnostics only
    inventory: int
    cash_ticks: int  # cumulative realized cash in tick-units
    bid: int | None  # MM's live quote at the mark (ticks)
    ask: int | None


@dataclass
class BacktestResult:
    config: SimConfig
    strategy_name: str
    marks: list[Mark]
    fills: list[MMFill]
    quoted_spreads: list[tuple[float, int]]  # (time, ask - bid) when both quoted

    # -- convenience arrays (ticks unless noted) ---------------------------
    def mark_times(self) -> np.ndarray:
        return np.array([m.time for m in self.marks])

    def mark_mids(self) -> np.ndarray:
        return np.array([m.mid for m in self.marks])

    def mark_inventory(self) -> np.ndarray:
        return np.array([m.inventory for m in self.marks])

    def pnl_dollars(self) -> np.ndarray:
        """Mark-to-market PnL path in dollars: cash + inventory * mid."""
        cash = np.array([m.cash_ticks for m in self.marks], dtype=float)
        inv = np.array([m.inventory for m in self.marks], dtype=float)
        mid = self.mark_mids()
        return (cash + inv * mid) * self.config.tick_size


class _EventQueue:
    """Priority queue of (time, seq, action) with a stable tiebreaker."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Callable[[float], object]]] = []
        self._counter = itertools.count()

    def schedule(self, time: float, action: Callable[[float], object]) -> None:
        heapq.heappush(self._heap, (time, next(self._counter), action))

    def pop(self) -> tuple[float, Callable[[float], object]]:
        time, _, action = heapq.heappop(self._heap)
        return time, action

    def __bool__(self) -> bool:
        return bool(self._heap)


class _Simulator:
    """Wires book, flow, and (optionally) a strategy into one event loop."""

    def __init__(self, config: SimConfig, strategy: MarketMaker | None):
        self.cfg = config
        self.strategy = strategy
        self.queue = _EventQueue()
        self.book = LimitOrderBook()
        self.rng = np.random.default_rng(config.seed)
        self.efficient = EfficientPrice(config.initial_price_ticks, config.sigma_ticks, self.rng)
        self.flow = OrderFlow(self.book, self.efficient, config.flow, self.rng,
                              self.queue.schedule)
        # MM state
        self.inventory = 0
        self.cash_ticks = 0
        self.live_bid: tuple[int, int] | None = None  # (order_id, price)
        self.live_ask: tuple[int, int] | None = None
        self.last_mid: float = config.initial_price_ticks
        # Recording
        self.marks: list[Mark] = []
        self.fills: list[MMFill] = []
        self.quoted_spreads: list[tuple[float, int]] = []
        self.raw_flow_events: list[BookEvent] = []  # kept only in flow-only runs
        self._record_flow_events = strategy is None
        self.session_start = config.warmup
        self.end_time = config.warmup + config.horizon

    # ------------------------------------------------------------------ run

    def run(self) -> BacktestResult:
        self.flow.start(0.0)
        if self.strategy is not None:
            self.queue.schedule(self.session_start, self._requote)
        self.queue.schedule(self.session_start, self._mark)
        while self.queue:
            time, action = self.queue.pop()
            if time > self.end_time:
                break
            events = action(time)
            if isinstance(events, list):
                self._process_events(events)
        self._final_mark()
        return BacktestResult(
            config=self.cfg,
            strategy_name=self.strategy.name if self.strategy else "flow-only",
            marks=self.marks,
            fills=self.fills,
            quoted_spreads=self.quoted_spreads,
        )

    # ------------------------------------------------------------- MM events

    def _requote(self, t: float) -> list[BookEvent]:
        self.queue.schedule(t + self.cfg.requote_interval, self._requote)
        assert self.strategy is not None
        mid = self.book.mid()
        if mid is not None:
            self.last_mid = mid
        session_t = t - self.session_start
        quote = self.strategy.quote(session_t, self.last_mid, self.inventory, self.book)
        events: list[BookEvent] = []
        self.live_bid = self._reconcile(self.live_bid, Side.BUY, quote.bid_price,
                                        quote.size, t, events)
        self.live_ask = self._reconcile(self.live_ask, Side.SELL, quote.ask_price,
                                        quote.size, t, events)
        if quote.bid_price is not None and quote.ask_price is not None:
            self.quoted_spreads.append((session_t, quote.ask_price - quote.bid_price))
        return events

    def _reconcile(
        self,
        live: tuple[int, int] | None,
        side: Side,
        desired_price: int | None,
        size: int,
        t: float,
        events: list[BookEvent],
    ) -> tuple[int, int] | None:
        """Cancel-replace one side of the quote, keeping the live order (and
        its queue priority) when the desired price is unchanged."""
        if live is not None:
            order_id, price = live
            still_live = self.book.get_order(order_id) is not None
            if still_live and desired_price == price:
                return live  # unchanged: keep queue position
            if still_live:
                cancelled = self.book.cancel(order_id, t)
                if cancelled is not None:
                    events.append(cancelled)
        if desired_price is None:
            return None
        # A marketable quote would make us a taker; the strategy layer already
        # clips to passive prices, so this is a hard error, not a silent trade.
        order_id, new_events = self.book.submit_limit(MM_OWNER, side, desired_price, size, t)
        for ev in new_events:
            if isinstance(ev, Fill):
                raise RuntimeError(
                    f"market-maker quote crossed the book: {side} {desired_price}"
                )
        events.extend(new_events)
        return (order_id, desired_price)

    def _mark(self, t: float) -> None:
        self.queue.schedule(t + self.cfg.mark_interval, self._mark)
        self._record_mark(t)

    def _final_mark(self) -> None:
        if not self.marks or self.marks[-1].time < self.cfg.horizon:
            self._record_mark(self.end_time)

    def _record_mark(self, t: float) -> None:
        mid = self.book.mid()
        if mid is not None:
            self.last_mid = mid
        self.marks.append(
            Mark(
                time=t - self.session_start,
                mid=self.last_mid,
                efficient=self.efficient.value_at(t),
                inventory=self.inventory,
                cash_ticks=self.cash_ticks,
                bid=self.live_bid[1] if self.live_bid else None,
                ask=self.live_ask[1] if self.live_ask else None,
            )
        )

    # ------------------------------------------------------------ accounting

    def _process_events(self, events: list[BookEvent]) -> None:
        for ev in events:
            if self._record_flow_events:
                self.raw_flow_events.append(ev)
            if isinstance(ev, Fill) and ev.maker_owner == MM_OWNER:
                self._on_mm_fill(ev)

    def _on_mm_fill(self, fill: Fill) -> None:
        # The MM is always the maker (quotes are passive by construction).
        mm_side = fill.taker_side.opposite
        signed = mm_side.sign * fill.qty
        self.inventory += signed
        self.cash_ticks -= signed * fill.price
        # pre_trade_mid is None only if the *other* book side was empty when
        # the aggressor arrived; fall back to the last known mid.
        pre_mid = fill.pre_trade_mid if fill.pre_trade_mid is not None else self.last_mid
        self.fills.append(
            MMFill(
                time=fill.time - self.session_start,
                side=mm_side,
                price=fill.price,
                qty=fill.qty,
                pre_trade_mid=pre_mid,
                taker_owner=fill.taker_owner,
            )
        )


def run_backtest(config: SimConfig, strategy: MarketMaker) -> BacktestResult:
    """Run one episode of `strategy` in a freshly simulated market."""
    return _Simulator(config, strategy).run()


def run_flow_only(config: SimConfig) -> tuple[BacktestResult, list[BookEvent]]:
    """Run the market with no market maker; returns the result plus the raw
    event stream (used by calibration)."""
    sim = _Simulator(config, None)
    result = sim.run()
    return result, sim.raw_flow_events
