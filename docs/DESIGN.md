# Design notes: the microstructure reasoning behind every component

This document explains *why* each part of the simulator is built the way it is.
It is written to be read alongside the source; every claim here is implemented
and, where feasible, asserted by a test.

## 1. The problem being studied

A market maker continuously quotes a bid and an ask, hoping to earn the spread:
buy at the bid, sell at the ask, pocket the difference. Two risks eat that
income:

* **Inventory risk.** Fills arrive randomly, so the position drifts away from
  zero. While the market maker holds ±q, every move of the mid marks ±q × Δmid
  through its PnL. The spread income is small and steady; inventory PnL is
  zero-mean but large-variance — unmanaged, it dominates the PnL distribution.
* **Adverse selection.** Some counterparties trade *because they know the
  quote is wrong* (they observe value the quote hasn't incorporated yet).
  Conditional on trading with them, the price moves against the position the
  market maker just received. Unlike inventory risk this is not zero-mean: it
  is a systematic transfer from the quoter to the informed.

Everything in this repo exists to make those two effects measurable and
controllable: the matching engine makes fills mechanical and auditable, the
flow model produces both risks in tunable amounts, the Avellaneda-Stoikov
strategy is the canonical control for the first risk, and the metrics separate
the two effects in the realized PnL.

## 2. Matching engine (`book.py`)

**Semantics.** Standard continuous double auction with price-time (FIFO)
priority — the rules of essentially every modern limit-order market. Three
order types: limit (match what crosses, rest the remainder), market (match,
discard the remainder — market orders never rest), cancel. Every fill prints
at the *resting* order's price, so aggressors get price improvement level by
level as they walk the book.

**Integer ticks.** All engine prices are `int` tick counts. Matching logic
contains no floating point, so "does this cross?" is exact — no epsilon
comparisons, no accumulation error. Dollars exist only at the
reporting boundary (`tick_size`). This mirrors production practice and it
makes a whole class of bugs unrepresentable.

**Priority is sequence-based, not timestamp-based.** The engine assigns a
monotone arrival sequence number; FIFO order is defined by it. Timestamps are
carried on events for the audit trail but never used for ordering, so
simultaneous events cannot produce ambiguous priority.

**Data structures.** Per side: `dict[price -> deque]` for the FIFO queues plus
a lazy binary heap of level prices for O(log n) best-price access (stale heap
entries are popped on peek). Cancels are *eager* — the order is removed from
its deque immediately. Production engines usually prefer O(1) lazy tombstoning
with intrusive lists; eager removal was chosen deliberately because it keeps a
stronger invariant — *everything in a queue is live* — which makes the
matching loop simpler to reason about, and level sizes here are small. This is
a documented correctness-over-throughput trade-off, not an oversight — and a
measured one: `benchmarks/bench_book.py` shows ~580k ops/s on a mixed
workload, flat from an empty book to 100k resting orders, roughly four orders
of magnitude above what the simulation consumes.

**How correctness is established.** Three layers in the test suite:

1. *Scenario tests* (`test_book.py`): each economic rule — price priority,
   FIFO, partial fills, walking the book, marketable limits, market-order
   remainders, cancel-race no-ops — as a small hand-checkable case. These are
   the human-readable specification.
2. *Differential testing* (`test_book_properties.py`): a deliberately naive
   reference implementation (flat list, linear scans — slow but obviously
   correct) is driven with the same randomized order streams via Hypothesis,
   and the engine must produce **identical fills and identical book state
   after every operation**. Any divergence is shrunk to a minimal
   counterexample. This covers the combinatorial interaction space that
   hand-written cases cannot.
3. *Invariant checking*: after every operation in every test, the book
   asserts it is uncrossed, FIFO-ordered, free of empty levels and zero-qty
   orders, and that its order index matches its queues; separately, a
   conservation property (every submitted share is filled, cancelled, or
   resting) is checked under random streams.

## 3. Order-flow model (`flow.py`)

The environment must generate *both* risks endogenously, stay stationary, and
give the strategy a fill-rate structure it can be calibrated against. Three
agent populations around a latent price do this with minimal machinery:

**Latent efficient price.** `p*(t)` follows arithmetic Brownian motion with
volatility σ — exactly the price model Avellaneda-Stoikov assume, so the
strategy is evaluated in its own model class. It is sampled lazily at event
times with correct √dt scaling. Nobody trades "at" p*; it only drives
behavior.

**Noise traders** arrive Poisson and either send a market order in a random
direction or place a limit order a geometric number of ticks behind the
opposite touch. Their limits carry exponential lifetimes (cancelled on
expiry), which keeps the resting book stationary instead of growing without
bound. Noise flow is symmetric and information-free: it is the market maker's
revenue source.

**Informed traders** arrive Poisson, observe p*, and send a market order only
when a standing quote is strictly mispriced (`p* > best ask` → buy). Two
consequences: (a) they are pure adverse selection — *conditional on being hit
by one, your quote was on the wrong side of value*, and the mid subsequently
drifts against you as the book reprices; (b) they are the taker channel that
drags the traded price toward p*.

**Value traders** place a fraction of the limit-order flow relative to p\*
rather than relative to the touch, and their orders are allowed to cross the
book when it has drifted (the engine treats them as marketable limits). This
population is the result of an explicit design iteration: with only
book-relative liquidity the book has no anchor — it wanders on its own noise,
realized mid volatility becomes a large multiple of σ, and *no* mid-referenced
quoting model can survive, which is an artifact, not economics. Real books
track value because enough participants price off value; value traders supply
exactly that force. The flow tests assert the resulting anchoring (mid–p\* gap
of ~1 tick median at default parameters) and its absence when the anchoring
channels are switched off.

**Emergent fill intensity.** A quote resting δ behind the mid is only reached
by market orders big enough to walk through the book to that depth. Because
order sizes are geometric and depth accumulates level by level, the arrival
rate of such orders decays roughly exponentially in δ — which is precisely the
`λ(δ) = A e^{-kδ}` form the A-S model assumes. This is *emergent from the
flow*, not imposed; the calibration below measures it (empirical fit R² ≈
0.99).

**A statistical test that "informed" means informed.** The flow suite verifies
the defining property: mids drift in the direction of informed trades
(markouts of several ticks) but not of noise trades. Adverse selection in this
simulator is a measured phenomenon, not a label.

## 4. Calibration (`calibration.py`)

The strategy is never given generator truth. Both of its market inputs are
estimated from simulated data alone, the way a desk would estimate them from
market data:

* **k (fill-intensity decay):** run the market with no market maker, record
  for every market order the deepest price it reached past the pre-trade mid,
  and count, for each depth δ, the rate of orders reaching ≥ δ. Regress
  log-rate on δ: intercept gives A, slope gives −k. The estimate is slightly
  optimistic (a real resting quote would itself absorb flow and add depth) —
  acknowledged and acceptable for setting a quoting parameter.
* **σ (mid volatility):** realized volatility of the mid **sampled every ~5
  seconds**, not every mark. At high frequency the mid's variance is dominated
  by bid-ask bounce — a stationary noise whose per-interval contribution does
  not shrink with dt — inflating naive estimates by 1.5–2.5× (the volatility
  signature plot effect). Since the A-S skew scales with σ², using the naive
  estimate makes the strategy shed inventory far too aggressively; this
  showed up empirically in tuning as A-S underperforming at high σ, and
  coarse sampling fixed it. (Estimated σ̂ ≈ 0.0113 $/√s against a true 0.01.)

## 5. The Avellaneda-Stoikov strategy (`strategy.py`)

The model (Avellaneda & Stoikov 2008) solves optimal quoting for a market
maker with CARA utility `−exp(−γW_T)` when the mid follows arithmetic BM and
fills arrive with intensity `λ(δ) = A e^{−kδ}`. Two closed forms:

**Reservation price** — the inventory-adjusted private valuation:

    r = s − q · γσ²(T−t)

Holding q units adds `q²σ²(T−t)` of terminal-wealth variance. A long agent
therefore values the asset below the mid: both quotes shift down, the ask
becomes more aggressive (sell sooner) and the bid less aggressive (buy less).
**This is why inventory skew works**: it does not cross the spread to dump
risk — it *tilts the probabilities* of the next fill so inventory mean-reverts
toward zero while still earning the spread on both sides. The backtest
verifies the effect directly: A-S mean |inventory| ≈ 5.9 vs ≈ 12.5 for the
unskewed baseline in identical markets.

**Optimal spread**:

    δ_bid + δ_ask = γσ²(T−t) + (2/γ) ln(1 + γ/k)

The first term is compensation for inventory variance over the remaining
horizon (wider when vol is high or the session is young); the second trades
margin-per-fill against fill frequency through k. As t → T both the skew and
the risk term collapse — a position at the close has no time to hurt you.

**Discrete-market adaptations** (each a deliberate, documented departure):

* Quotes are rounded *outward* to the tick grid — never quote tighter than
  the model asks.
* Quotes are clipped to stay **passive** (≥1 tick away from the opposite
  touch). Under large inventory the raw formulas can request a crossing
  quote; a quoting strategy should not silently become a taker. The harness
  enforces this with a hard error if an MM order would ever match on arrival.
* **Hard position limits**: a side stops quoting when a full fill could push
  |q| past the cap. The soft skew already mean-reverts inventory; the hard
  limit bounds worst-case exposure. Standard desk practice.

**The baseline** (`SymmetricQuoter`) quotes a fixed half-spread around the mid
with *identical* size, limits, and clipping. It isolates exactly one variable
— the quoting model — so the comparison attributes differences to
reservation-price skew and spread optimization, not mechanics.

## 6. Backtest harness (`backtest.py`)

* **Single event queue.** Flow arrivals, order expiries, requotes, and marks
  are all `(time, seq, action)` entries in one priority queue; the seq
  tiebreaker makes simultaneous events execute in schedule order. Runs are
  exactly reproducible per seed (asserted by test), and strategies are
  compared with **common random numbers** — episode i of every strategy sees
  the same market realization. This is what makes *paired* inference valid:
  the reported comparison is the per-seed PnL difference with a paired
  t-statistic and a bootstrap CI, which cancels the market-realization noise
  that dominates each strategy's own variance.
* **Queue-priority-preserving quote management.** On requote, a live order is
  left untouched when the strategy still wants its price, because
  cancel-replace would send it to the back of the FIFO queue. Time priority is
  a real asset for a passive quoter; churning it away systematically
  understates fill quality.
* **Warm-up.** The flow runs alone until the book reaches its stationary depth
  profile; the strategy then trades a fixed horizon, with all recording in
  session-relative time.
* **Exact cash accounting.** Cash is integer tick-units (`price × qty` of
  ints); realized cash flows carry zero floating-point error. A test replays
  the fill stream independently and requires it to reproduce inventory and
  cash at every mark.

## 7. Evaluation (`metrics.py`)

**Exact PnL decomposition.** Walking fills and marks in time order, each
increment of `PnL = cash + q·mid` splits into

* *spread capture* at each fill: `signed_qty × (pre-trade mid − price)` — the
  edge earned for providing liquidity, measured against the pre-trade mid so
  the trade's own impact doesn't contaminate it; every passive fill's capture
  is strictly positive (asserted per-fill in tests);
* *inventory PnL* between checkpoints: `q × Δmid` — revaluation of the
  carried position.

The identity `total = spread capture + inventory PnL` holds exactly (machine
precision), verified end-to-end on real backtests at every mark. The
decomposition is the diagnostic: a healthy market maker earns in the first
bucket; the second is risk (and, when correlated with fills, adverse
selection).

**Markouts.** For each fill, `signed_qty × (mid_{t+τ} − mid_t)` per share at
τ ∈ {1, 5, 30}s, split by counterparty type. Negative markout = the price
moved against the position received = adverse selection cost. The split is the
point: against informed counterparties markouts are −2 to −5 ticks and deepen
with σ; against noise they sit near zero. Equivalently, effective spread minus
realized spread.

**Sharpe conventions.** Reported across independent seeds
(`mean(final PnL)/std(final PnL)`) and per-episode from the mark grid. No
annualization: simulation seconds have no calendar meaning, and dressing them
up as a yearly number would be theater.

## 8. Known limitations (and why they're acceptable here)

* **No queue competition from other market makers, and latency only via the
  requote interval.** The latency sweep in the results shows what that knob
  costs (~44% of PnL from 0.25s to 10s) and through which channel — foregone
  volume and slower inventory control, *not* deeper per-fill markouts,
  because informed flow here only attacks the touch. Modeling pick-off risk
  proper would need informed traders who snipe stale quotes deep in the book.
* **Calibration ignores the probe's own impact.** λ(δ) is measured without
  the MM's quote in the book; adding it would add depth and absorb flow.
  Bias is small at these sizes and conservative for the fit's use.
* **One asset, one venue, no fees/rebates.** Maker-taker economics would
  shift the optimal spread by the rebate; straightforward to add to the
  accounting.
* **Informed traders are impatient (market orders only).** Real informed flow
  also posts limits; the value-trader population partially covers this
  channel.
* **A-S assumptions vs. this market.** The model assumes continuous prices,
  no ticks, and state-independent fill intensity. The measured effects of
  those gaps (tick rounding, clipping, σ̂ bias) are discussed in the results
  write-up rather than hidden.
