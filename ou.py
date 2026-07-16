from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class OUKalman:
    theta: float
    mu: float
    sigma: float

    Q: float
    P: float
    R: float
    x: float

    mu_alpha: float
    jump_z_threshold: float

    just_jumped: bool = False

    @property
    def stat_std(self) -> float:
        return self.sigma / np.sqrt(abs(2 * self.theta))

    @property
    def half_life(self) -> float:
        return np.log(2) / abs(self.theta)

    @classmethod
    def calibrate(
            cls, prices: np.ndarray, model_trust: float,
            mu_alpha: float, jump_z_threshold: float) -> OUKalman:
        
        p = max(0, min(100, model_trust)) / 100.0
        trust_scale = 1e8 if p >= 1.0 else 0.1 + (10.0 - 0.1) * p

        y = prices[np.isfinite(prices)]
        mu = float(np.mean(y))

        x_lag = y[:-1]
        x_current = y[1:]

        X = np.column_stack([np.ones_like(x_lag), x_lag])
        beta = np.linalg.lstsq(X, x_current, rcond=None)[0].astype(float)

        c, theta = beta
        theta = float(np.clip(theta, 0.01, 0.99))

        residuals = x_current - (c + theta * x_lag)
        sigma = np.sqrt(np.mean(residuals ** 2))

        if sigma <= 0 or not np.isfinite(sigma):
            sigma = max(np.std(y) * 0.01, 1e-9)

        var = sigma ** 2
        Q = var * max(1 - theta ** 2, 1e-6)
        P = R = var * max(trust_scale, 0.01)

        return cls(
            theta, mu, sigma, Q, P, R, mu,
            mu_alpha, jump_z_threshold,
        )

    def update(self, price: float) -> float:
        x_pred = self.theta * self.x + (1 - self.theta) * self.mu
        P_pred = self.theta ** 2 * self.P + self.Q

        S = P_pred + self.R
        innovation = price - x_pred
        z = innovation / np.sqrt(S)

        self.just_jumped = abs(z) > self.jump_z_threshold

        if self.just_jumped:
            self.x = self.mu = price
            self.P = self.R

            return 0.0

        K = P_pred / S
        self.x = x_pred + K * innovation
        self.P = (1 - K) * P_pred

        self.mu += self.mu_alpha * (self.x - self.mu)

        return z