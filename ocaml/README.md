# OCaml matching engine

A port of the simulator's limit-order-book matching engine
([`book.py`](../src/market_maker_sim/book.py)) to OCaml, written to learn the
language on the part of the system where correctness matters most.

```bash
make test    # compiles with the stock ocamlc; no opam switch or dune needed
```

```
best prices and mid                       ok
price priority                            ok
time priority is FIFO                     ok
walks the book with price improvement     ok
marketable limit rests remainder          ok
market order never rests                  ok
partial fill keeps priority               ok
cancel                                    ok
non-crossing limit just rests             ok
depth is aggregated, best first           ok
pre-trade mid recorded before trade       ok
rejects non-positive quantity             ok

38 checks, 0 failures
```

## What changed in the translation

The port keeps the Python engine's semantics exactly (verified by running the
same scenarios through both: a marketable buy for 15 against asks of 10 at 101
and 10 at 102 prints `(101, 10)` then `(102, 5)` in both, with a recorded
pre-trade mid of 100, and a market buy for 50 against 20 resting discards 30 as
`unfilled_market`). Two things are genuinely nicer in OCaml:

**A balanced-tree map replaces a heap plus a hash map.** The Python side keeps
`dict[price, deque]` for the levels *and* a binary heap of prices for
best-price access, with the heap cleaned lazily because entries go stale when a
level empties. `Map.Make(Int)` subsumes both: `max_binding` and `min_binding`
give the best bid and ask in `O(log n)` directly from the same structure that
stores the levels, so empty levels are just deleted and every price in the map
is live. The lazy-cleanup code and the invariant it needed disappear.

**The event type is a real sum type.** Python models `Accepted | Fill |
Cancelled` as a union alias and recovers the case with `isinstance`. In OCaml
these are constructors of one `event` variant, so pattern matching over them is
checked for exhaustiveness at compile time: adding a fourth event kind produces
a warning at every site that consumes events, rather than a bug found at
runtime.

Prices remain integer ticks throughout, so the matching path contains no
floating point and price comparison is exact.

## Files

| File | Contents |
|---|---|
| [`orders.ml`](orders.ml) | Order and event types: `side`, `resting_order`, and the `event` variant |
| [`book.ml`](book.ml) | The matching engine: price-time priority, walking the book, cancels, invariants |
| [`test_book.ml`](test_book.ml) | Tests covering priority, price improvement, resting, and cancellation |
