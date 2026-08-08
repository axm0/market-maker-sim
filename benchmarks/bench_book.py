"""Matching-engine throughput benchmark.

Generates a realistic mixed workload (limit orders around a drifting mid,
market orders, cancels of random live orders) and measures sustained
operations per second through the engine, at several resting-book sizes.

Run:  .venv/bin/python benchmarks/bench_book.py

This is a micro-benchmark of the engine alone (no simulation harness). The
engine favors auditability over raw speed — eager cancels, per-operation
event allocation — and the point of the numbers is to show those choices
still leave throughput far above what the simulation needs (~20 events/s of
simulated time), with headroom of several orders of magnitude.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_maker_sim.book import LimitOrderBook
from market_maker_sim.orders import Side


def preload_book(book: LimitOrderBook, n_resting: int, rng: np.random.Generator) -> int:
    """Fill the book with n_resting orders spread over ~n/50 price levels on
    each side of mid=10_000, deep enough that the workload never eats through
    them. Returns the number of levels per side."""
    levels = max(2, n_resting // 100)
    per_level = max(1, n_resting // (2 * levels))
    for i in range(levels):
        for _ in range(per_level):
            qty = int(rng.geometric(1.0 / 5.0))
            # Rest away from the touch (gap of 20 ticks) so the measured
            # workload trades in its own price band on top of a big book.
            book.submit_limit("bg", Side.BUY, 10_000 - 20 - i, qty, 0.0)
            book.submit_limit("bg", Side.SELL, 10_000 + 20 + i, qty, 0.0)
    return levels


def build_workload(n_ops: int, id_offset: int, rng: np.random.Generator) -> list[tuple]:
    """Pre-generate operations so generation cost is outside the timed loop.

    Mix: 50% limit orders at geometric depth around a slowly drifting mid,
    20% market orders, 30% cancels of a random earlier workload order (some
    already filled/cancelled, exercising the no-op path)."""
    ops: list[tuple] = []
    mid = 10_000
    next_id = id_offset + 1
    live_estimate: list[int] = []
    for _ in range(n_ops):
        mid = int(mid + rng.integers(-1, 2))
        mid = min(max(mid, 9_990), 10_010)  # stay inside the preloaded band
        u = rng.random()
        if u < 0.5 or not live_estimate:
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            offset = int(rng.geometric(1.0 / 4.0))
            price = mid - offset if side is Side.BUY else mid + offset
            qty = int(rng.geometric(1.0 / 5.0))
            ops.append(("limit", side, price, qty))
            live_estimate.append(next_id)
            next_id += 1
        elif u < 0.7:
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = int(rng.geometric(1.0 / 6.0))
            ops.append(("market", side, qty))
            next_id += 1
        else:
            k = int(rng.integers(0, len(live_estimate)))
            ops.append(("cancel", live_estimate.pop(k)))
    return ops


def run_workload(book: LimitOrderBook, ops: list[tuple]) -> tuple[float, int]:
    fills = 0
    t0 = time.perf_counter()
    for op in ops:
        if op[0] == "limit":
            _, events = book.submit_limit("b", op[1], op[2], op[3], 0.0)
        elif op[0] == "market":
            _, events = book.submit_market("b", op[1], op[2], 0.0)
        else:
            ev = book.cancel(op[1], 0.0)
            events = [ev] if ev else []
        fills += sum(1 for e in events if type(e).__name__ == "Fill")
    elapsed = time.perf_counter() - t0
    return elapsed, fills


def main() -> None:
    n_ops = 200_000
    print(f"workload: {n_ops:,} mixed ops (50% limit / 20% market / 30% cancel), "
          "on top of a preloaded resting book\n")
    print(f"{'preloaded orders':>17} {'ops/sec':>12} {'fills':>9} {'final book':>11}")
    for n_resting in (0, 1_000, 10_000, 100_000):
        rng = np.random.default_rng(0)
        book = LimitOrderBook()
        preload_book(book, n_resting, rng)
        preloaded = book.order_count()
        ops = build_workload(n_ops, id_offset=preloaded, rng=rng)
        elapsed, fills = run_workload(book, ops)
        print(f"{preloaded:>17,} {n_ops / elapsed:>12,.0f} "
              f"{fills:>9,} {book.order_count():>11,}")


if __name__ == "__main__":
    main()
