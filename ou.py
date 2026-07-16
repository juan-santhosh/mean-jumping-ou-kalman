from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class JumpOUKalman:
    phi: float
    mu: float
    sigma: float

    Q: float
    P: float
    R: float
    x: float

    mu_ewma_alpha: float
    jump_z_threshold: float

    just_jumped: bool = False

    @classmethod
    def calibrate(
            cls, prices: np.ndarray, model_trust: float,
            mu_ewma_alpha: float, jump_z_threshold: float) -> JumpOUKalman:
        
        p = max(0, min(100, model_trust)) / 100.0
        trust_scale = 1e8 if p >= 1.0 else 0.1 + (10.0 - 0.1) * p

        y = prices[np.isfinite(prices)]

        if y.size < 3:
            raise ValueError(
                "At least three valid price observations are required."
            )
        
        mu = float(np.mean(y))

        x_prev = y[:-1]
        x_current = y[1:]

        X = np.column_stack([np.ones_like(x_prev), x_prev])
        c, phi = np.linalg.lstsq(X, x_current, rcond=None)[0].astype(float)

        phi = float(np.clip(phi, 0.01, 0.99))

        residuals = x_current - (c + phi * x_prev)
        sigma = np.sqrt(np.mean(residuals ** 2))

        if sigma <= 0.0 or not np.isfinite(sigma):
            sigma = max(float(np.std(y)) * 0.01, 1e-9)

        variance = sigma ** 2

        process_cov = variance * max(1.0 - phi**2, 1e-6)
        state_cov = variance * max(trust_scale, 0.01)
        observation_cov = state_cov

        return JumpOUKalman(
            phi=phi,
            mu=mu,
            sigma=sigma,
            Q=process_cov,
            P=state_cov,
            R=observation_cov,
            x=mu,
            mu_ewma_alpha=mu_ewma_alpha,
            jump_z_threshold=jump_z_threshold
        )

    def update(self, price: float) -> float:
        predicted_state = self.phi * self.x + (1.0 - self.phi) * self.mu
        predicted_covariance = self.phi**2 * self.P + self.Q

        total_variance = predicted_covariance + self.R
        innovation = price - predicted_state

        z_score = innovation / float(np.sqrt(total_variance))

        self.just_jumped = abs(z_score) > self.jump_z_threshold

        if self.just_jumped:
            self.x = self.mu = price
            self.P = self.R

            return 0.0

        kalman_gain = predicted_covariance / total_variance
        self.x = predicted_state + kalman_gain * innovation
        self.P = (1.0 - kalman_gain) * predicted_covariance

        self.mu += self.mu_ewma_alpha * (self.x - self.mu)

        return z_score