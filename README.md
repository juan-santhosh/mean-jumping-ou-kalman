# Mean-Reversion Trading Strategy

A mean-reversion trading strategy based on a jump-aware Ornstein–Uhlenbeck (OU) process using an adaptive Kalman filter. The model estimates the latent fair value of an asset in real time and generates entry signals when the market price deviates significantly below its estimated equilibrium.

## Method

The price movements of the asset are modelled as a discrete-time OU process and estimated using a Kalman filter with configurable bias to new data. Rather than assuming a fixed equilibrium, the long-run mean from the initial calibration is updated continuously using an exponentially weighted moving average, allowing the model to adapt to changing market conditions over time.

Additionally, large innovation z-scores are treated as structural price jumps. When detected, the filter state is reset to the new price level, allowing it to begin modelling around the new equilibrium price rather than on old dynamics.

## Features

* Adaptive Kalman filter state estimation
* Jump-aware Ornstein–Uhlenbeck model
* Continuous mean estimation
* Z-score signal generation
* Configurable entry, exit, and jump thresholds

## Backtesting Assumptions

Performance is evaluated on historical candle data using my custom-built backtesterlib Python library under the following assumptions:

* Signals execute on the following bar (no look-ahead bias)
* Fixed proportional transaction costs
* Long-only position (assuming cryptocurrencies cannot be shorted)
* No slippage, market impact, or liquidity constraints

This project was developed to explore stochastic processes, state-space estimation, and algorithmic trading using Python.