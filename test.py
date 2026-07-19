import pandas as pd
import matplotlib.pyplot as plt
from backtesterlib import Backtester, BuyAndHold

from data import load_data
from strategies import MeanReversion

plt.style.use('dark_background')

BACKTEST_MONTHS = 4
WINDOWS_PER_YEAR = 12 / BACKTEST_MONTHS
MINUTES_PER_YEAR = 60 * 24 * 365

df = load_data("data/SOLUSDT-1.csv", BACKTEST_MONTHS)
    
backtester = Backtester(
    df=df, close_column="close",
    bars_per_year=MINUTES_PER_YEAR, 
    windows_per_year=WINDOWS_PER_YEAR,
    fee_rate=0.00001, fee_in_usd=False
)

backtester.run(
    MeanReversion(
        calibration_window=20, entry_threshold=1.2, exit_threshold=0.2,
        model_trust=20.0, mu_ewma_alpha=0.1,
        jump_z_threshold=5.0,
    ),
    n_mote_carlo_paths=1000
)

backtester.run(BuyAndHold(amount=1.0), n_mote_carlo_paths=0, baseline=True)

backtester.plot_results(figsize=(12, 8), plot_correlations=False)
backtester.log_results()