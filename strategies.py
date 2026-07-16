from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesterlib import Strategy

from ou import JumpOUKalman

@dataclass(slots=True)
class MeanReversion(Strategy):
    """
    Mean reversion strategy using JumpOUKalman class.
    Implementation of backtesterlib's abstract Strategy class.

    Attributes:
        calibration_window (int): Number of data points to use for initial model calibration.
        entry_threshold (float): Z-score threshold to enter position.
        exit_threshold (float): Z-score threshold to exit held position.
        model_trust (float): Percentage trust in OU model rather than latest data.

        mu_ewma_alpha (float): Learning rate for mu adaption to latest price data.
        jump_z_threshold (float): Z-score threshold to flag price jump.
    """
    
    calibration_window: int
    entry_threshold: float
    exit_threshold: float
    model_trust: float

    mu_ewma_alpha: float
    jump_z_threshold: float

    @property
    def name(self) -> str:
        return (
            f"{self.calibration_window}-Bar MR"
            f" Z=[{self.exit_threshold}, {self.entry_threshold}]"
            f" {self.model_trust}% OU Trust"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prices = df["close"].to_numpy(dtype=float)

        # Calibrate model on initial window
        model = JumpOUKalman.calibrate(
            prices[:self.calibration_window], self.model_trust, 
            self.mu_ewma_alpha, self.jump_z_threshold
        )

        n = len(df)
        signal = np.zeros(n) # Prepare zero array for signal series.

        for t in range(self.calibration_window, n):
            z = model.update(prices[t])

            if model.just_jumped:
                continue

            # Signal[t] defaults to 0 so only 
            # non-zero target positions must be specified

            prev_signal = signal[t - 1]

            # If not holding position
            if prev_signal == 0.0: 
                if z < -self.entry_threshold:
                    signal[t] = 1.0 # Enter long

                # Assuming SOL cannot be in a short position

            elif z < -self.exit_threshold:
                signal[t] = prev_signal # Hold position

        return pd.Series(signal, index=range(n))