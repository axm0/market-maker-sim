# market-maker-sim

A limit-order-book market-making simulator: a price-time-priority matching engine, a
stochastic order-flow model with informed and noise traders, and an Avellaneda-Stoikov
market-making strategy, evaluated with risk-adjusted metrics.

## Motivation

Market making is the business of continuously quoting two-sided prices and earning the
spread while managing two opposing risks: **inventory risk** (holding a position when the
price moves against you) and **adverse selection** (being picked off by traders who know
more than you). This project builds a self-contained environment to study that trade-off
end to end: a realistic limit order book, a controllable flow of informed and uninformed
orders, and a market-making agent whose quotes respond to inventory and volatility.

Everything is self-contained and needs no proprietary market data: the book is a matching
engine, the order flow is simulated, and the strategy is a published model.

## Architecture

1. **Matching engine**: a limit order book with price-time (FIFO) priority, supporting
   limit, market, and cancel orders and emitting a full event stream of fills and book
   updates.
2. **Order-flow model**: a stochastic generator of order arrivals (Poisson) around a
   latent efficient price, mixing informed traders (who trade toward the true price, so
   they create adverse selection) and noise traders (who do not).
3. **Market-making strategy**: the Avellaneda-Stoikov optimal-quoting model. A reservation
   price shifted by inventory and risk aversion, an optimal spread set by volatility and
   order-arrival intensity, inventory-skewed quotes, and hard position limits.
4. **Backtest and evaluation**: runs the strategy through the simulated market and reports
   risk-adjusted performance, not just raw P&L.

## Metrics

* Realized and mark-to-market P&L
* Sharpe ratio and P&L volatility
* Inventory trajectory and maximum position held
* Spread capture vs. inventory (mark-to-market) PnL decomposition
* Adverse-selection cost

## Roadmap

* [ ] Price-time-priority matching engine
* [ ] Stochastic order-flow model (informed / noise mix)
* [ ] Avellaneda-Stoikov quoting strategy with inventory control
* [ ] Backtest harness and metrics
* [ ] Result plots (P&L decomposition, inventory, quote behavior)
* [ ] Optional: replay on real order-flow data (free crypto LOB feeds)

## References

* Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit order book.*
  Quantitative Finance, 8(3), 217-224.
