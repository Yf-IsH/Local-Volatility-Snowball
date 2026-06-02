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

    def _dupire_terms(self, K, T):
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
        return numerator, denominator

    def local_vol(self, K, T):
        """
        Dupire formula under dS/S = (r-q)dt + sigma_loc(S,t)dW:

        sigma_loc^2(K,T) =
            [C_T + q C + (r-q) K C_K] / [0.5 K^2 C_KK]

        where C(K,T) is the time-0 discounted call price as a function of strike
        and maturity. Numerical derivatives are finite differences on the
        smoothed implied-volatility generated price surface.
        """
        K = float(K)
        T = float(T)

        if T <= 1e-5 or K <= 0:
            return self.vol_surface.get_iv(max(K, 1e-8), max(T, self.vol_surface.min_maturity))

        numerator, denominator = self._dupire_terms(K, T)

        if numerator <= 0.0 or denominator <= 1e-10:
            return self.vol_surface.get_iv(K, T)

        return float(np.sqrt(max(numerator / denominator, 1e-12)))

    def local_vol_diagnostics(self, strikes=None, maturities=None):
        """Return Dupire sign diagnostics on a strike/maturity grid."""
        if strikes is None:
            strikes = self.vol_surface.K_grid
        if maturities is None:
            maturities = self.vol_surface.T_grid

        total = 0
        positive_variance = 0
        positive_density = 0
        positive_numerator = 0
        fallback = 0
        min_local_variance = np.inf
        max_local_variance = -np.inf

        for K in np.asarray(strikes, dtype=float):
            for T in np.asarray(maturities, dtype=float):
                if T <= 1e-5 or K <= 0:
                    continue

                numerator, denominator = self._dupire_terms(K, T)
                total += 1
                positive_numerator += numerator > 0.0
                positive_density += denominator > 1e-10
                if numerator > 0.0 and denominator > 1e-10:
                    variance = numerator / denominator
                    positive_variance += variance > 0.0
                    min_local_variance = min(min_local_variance, variance)
                    max_local_variance = max(max_local_variance, variance)
                else:
                    fallback += 1

        if total == 0:
            return {
                "points": 0,
                "positive_variance_ratio": 0.0,
                "positive_density_ratio": 0.0,
                "positive_numerator_ratio": 0.0,
                "fallback_ratio": 0.0,
                "min_local_vol": 0.0,
                "max_local_vol": 0.0,
            }

        min_vol = np.sqrt(max(min_local_variance, 0.0)) if np.isfinite(min_local_variance) else 0.0
        max_vol = np.sqrt(max(max_local_variance, 0.0)) if np.isfinite(max_local_variance) else 0.0
        return {
            "points": total,
            "positive_variance_ratio": float(positive_variance / total),
            "positive_density_ratio": float(positive_density / total),
            "positive_numerator_ratio": float(positive_numerator / total),
            "fallback_ratio": float(fallback / total),
            "min_local_vol": float(min_vol),
            "max_local_vol": float(max_vol),
        }
