from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class JumpOUKalman:
    """
    Kalman filtered mean-jumping Ornstein-Uhlenbeck process.
    Parameters are chosen through the calibrate() class method.

    Attributes:
        phi (float): Mean reversion speed bounded by (0, 1.0).
        mu (float): OU model estimate of long-run mean price.
        sigma (float): Volatility of price movements.
        Q (float): Process noise covariance.
        P (float): Process state covariance.
        R (float): Observation noise covariance.
        x (float): Current latent state estimate.
        mu_ewma_alpha (float): Learning rate for adapting mu to x.
        jump_z_threshold (float): Innovation z-score required to classify an observation as a jump.
        just_jumped (bool): Default False. Stores whether the most recent price update was a jump.
    """

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
        
        """
        Estimate model parameters from historical prices.

        The discrete-time OU process is approximated as an AR(1) model
        and fit via ordinary least squares.

        Parameters:
            prices (np.ndarray): NumPy array of historical price observations.
            model_trust (float): Percentage confidence in OU model. Larger values reduce influence of new observations.
            mu_ewma_alpha (float): Learning rate for adapting mu to x.
            jump_z_threshold (float): Innovation z-score required to classify an observation as a jump.
        
        Returns:
            JumpOUKalman: Model with parameters calibrated to historical price data.
        """
        
        # Scale percentage confidence
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

        # Clip phi to bounds (0.0, 1.0)
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
        """
        Update Kalman filter with latest price and return z-score of price.

        Parameters:
            price (float): Latest observed market price.

        Returns:
            float: Innovation z-score. Returns zero on jump.
        """

        # Kalman prediction step
        predicted_state = self.phi * self.x + (1.0 - self.phi) * self.mu
        predicted_covariance = self.phi**2 * self.P + self.Q

        # Calculate innovation z-score
        innovation = price - predicted_state
        total_variance = predicted_covariance + self.R
        z_score = innovation / float(np.sqrt(total_variance))

        # Reset state if jumped
        self.just_jumped = abs(z_score) > self.jump_z_threshold

        if self.just_jumped:
            self.x = self.mu = price
            self.P = self.R

            return 0.0

        # Update kalman state
        kalman_gain = predicted_covariance / total_variance

        self.x = predicted_state + kalman_gain * innovation
        self.P = (1.0 - kalman_gain) * predicted_covariance

        # Adapt mu to x with an exponentially weighted moving average
        self.mu += self.mu_ewma_alpha * (self.x - self.mu)

        return z_score