import numpy as np


class LocalVolPricer:
    """Dupire local-volatility calculator from a smooth call price surface."""

    def __init__(self, vol_surface, spot, r=0.0, q=0.0):
        self.vol_surface = vol_surface
        self.S = float(spot)
        self.r = float(r)
        self.q = float(q)

    def _call(self, K, T):
        return self.vol_surface.black_scholes_call(self.S, K, T, self.r, self.q)

    def local_vol(self, K, T):
        """
        Dupire formula under dS/S = (r-q)dt + sigma_loc(S,t)dW:

        sigma_loc^2(K,T) =
            [C_T + q C + (r-q) K C_K] / [0.5 K^2 C_KK]

        where C(K,T) is the undiscounted input call price as a function of strike
        and maturity. Numerical derivatives are finite differences on the
        smoothed implied-volatility generated price surface.
        """
        K = float(K)
        T = float(T)

        if T <= 1e-5 or K <= 0:
            return self.vol_surface.get_iv(max(K, 1e-8), max(T, self.vol_surface.min_maturity))

        dT = max(1e-4, 1e-3 * T)
        dK = max(1e-2, 1e-4 * K)

        C_base = self._call(K, T)
        if T - dT >= self.vol_surface.min_maturity:
            C_dn_T = self._call(K, T - dT)
            C_up_T = self._call(K, T + dT)
            dC_dT = (C_up_T - C_dn_T) / (2.0 * dT)
        else:
            C_up_T = self._call(K, T + dT)
            dC_dT = (C_up_T - C_base) / dT

        C_up_K = self._call(K + dK, T)
        C_dn_K = self._call(max(K - dK, 1e-8), T)
        dC_dK = (C_up_K - C_dn_K) / (2.0 * dK)
        d2C_dK2 = (C_up_K - 2.0 * C_base + C_dn_K) / (dK**2)

        numerator = dC_dT + self.q * C_base + (self.r - self.q) * K * dC_dK
        denominator = 0.5 * K**2 * d2C_dK2

        if numerator <= 0.0 or denominator <= 1e-10:
            return self.vol_surface.get_iv(K, T)

        return float(np.sqrt(max(numerator / denominator, 1e-12)))
