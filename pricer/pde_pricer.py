import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve


class PDEPricer:
    """Implicit finite-difference pricer for European options under local vol."""

    def __init__(self, local_vol_pricer, r=0.0, q=0.0, space_steps=220, time_steps=220):
        self.lv_pricer = local_vol_pricer
        self.r = float(r)
        self.q = float(q)
        self.space_steps = int(space_steps)
        self.time_steps = int(time_steps)

    def price_european(self, S0, K, T, is_call=True):
        S0 = float(S0)
        K = float(K)
        T = float(T)

        if T <= 1e-8:
            return max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)

        M = self.space_steps
        N = self.time_steps
        S_max = max(3.0 * S0, 2.5 * K)
        dS = S_max / M
        dt = T / N
        S_grid = np.linspace(0.0, S_max, M + 1)

        if is_call:
            V = np.maximum(S_grid - K, 0.0)
        else:
            V = np.maximum(K - S_grid, 0.0)

        for step in range(N):
            tau_old = step * dt
            tau_new = (step + 1) * dt

            lower = np.zeros(M - 1)
            diag = np.zeros(M - 1)
            upper = np.zeros(M - 1)
            rhs = V[1:M].copy()

            for i in range(1, M):
                S = S_grid[i]
                sigma = self.lv_pricer.local_vol(S, tau_new)
                drift = (self.r - self.q) * S
                diffusion = 0.5 * sigma**2 * S**2

                a = diffusion / dS**2 - drift / (2.0 * dS)
                b = -2.0 * diffusion / dS**2 - self.r
                c = diffusion / dS**2 + drift / (2.0 * dS)

                row = i - 1
                lower[row] = -dt * a
                diag[row] = 1.0 - dt * b
                upper[row] = -dt * c

            if is_call:
                low_boundary = 0.0
                high_boundary = S_max * np.exp(-self.q * tau_new) - K * np.exp(-self.r * tau_new)
            else:
                low_boundary = K * np.exp(-self.r * tau_new)
                high_boundary = 0.0

            rhs[0] -= lower[0] * low_boundary
            rhs[-1] -= upper[-1] * high_boundary

            A = sparse.diags(
                [lower[1:], diag, upper[:-1]],
                offsets=[-1, 0, 1],
                shape=(M - 1, M - 1),
                format="csc",
            )
            V_inner = spsolve(A, rhs)
            V[0] = low_boundary
            V[1:M] = V_inner
            V[M] = high_boundary

        interpolator = interp1d(S_grid, V, kind="cubic", fill_value="extrapolate")
        return float(interpolator(S0))
