from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesterlib import Strategy

from ou import OUKalman

@dataclass(slots=True)
class MeanReversion(Strategy):
    window: int
    entry_threshold: float
    exit_threshold: float
    model_trust: float

    mu_alpha: float
    jump_z_threshold: float

    @property
    def name(self) -> str:
        return (
            f"{self.window}-Bar MR"
            f" Z=[{self.exit_threshold}, {self.entry_threshold}]"
            f" {self.model_trust}% OU Trust"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prices = df["close"].to_numpy(dtype=float)

        model = OUKalman.calibrate(
            prices[:self.window], self.model_trust, 
            self.mu_alpha, self.jump_z_threshold
        )

        n = len(df)
        signal = np.zeros(n)

        for t in range(self.window, n):
            z = model.update(prices[t])

            if model.just_jumped:
                continue

            prev_signal = signal[t - 1]

            if prev_signal == 0.0:
                if z < -self.entry_threshold:
                    signal[t] = 1.0

            elif z <= -self.exit_threshold:
                signal[t] = prev_signal

        return pd.Series(signal, index=range(n))