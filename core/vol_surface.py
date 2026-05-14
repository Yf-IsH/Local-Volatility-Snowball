import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm


class VolSurface:
    """Smooth implied-volatility surface on strike and maturity grids."""

    def __init__(self, strikes, maturities, iv_matrix):
        self.K_grid = np.asarray(strikes, dtype=float)
        self.T_grid = np.asarray(maturities, dtype=float)
        self.iv_matrix = np.asarray(iv_matrix, dtype=float)

        if self.iv_matrix.shape != (len(self.K_grid), len(self.T_grid)):
            raise ValueError("iv_matrix must have shape (len(strikes), len(maturities))")
        if np.any(np.diff(self.K_grid) <= 0) or np.any(np.diff(self.T_grid) <= 0):
            raise ValueError("strikes and maturities must be strictly increasing")
        if np.any(self.iv_matrix <= 0):
            raise ValueError("all implied volatilities must be positive")

        kx = min(3, len(self.K_grid) - 1)
        ky = min(3, len(self.T_grid) - 1)
        self.spline = RectBivariateSpline(self.K_grid, self.T_grid, self.iv_matrix, kx=kx, ky=ky)

    @property
    def min_strike(self):
        return float(self.K_grid[0])

    @property
    def max_strike(self):
        return float(self.K_grid[-1])

    @property
    def min_maturity(self):
        return float(self.T_grid[0])

    @property
    def max_maturity(self):
        return float(self.T_grid[-1])

    def get_iv(self, K, T):
        """Return interpolated implied volatility, clamped to the calibrated grid."""
        K_eval = np.clip(float(K), self.min_strike, self.max_strike)
        T_eval = np.clip(float(T), self.min_maturity, self.max_maturity)
        return max(float(self.spline(K_eval, T_eval)[0, 0]), 1e-8)

    def black_scholes_call(self, S, K, T, r, q=0.0):
        """Black-Scholes call price using the interpolated implied volatility."""
        S = float(S)
        K = float(K)
        T = float(T)
        r = float(r)
        q = float(q)

        if T <= 1e-8:
            return max(S - K, 0.0)

        sigma = self.get_iv(K, T)
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def black_scholes_put(self, S, K, T, r, q=0.0):
        """Black-Scholes put price using call-put parity."""
        call = self.black_scholes_call(S, K, T, r, q)
        if T <= 1e-8:
            return max(K - S, 0.0)
        return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))
