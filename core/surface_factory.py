import numpy as np

from core.vol_surface import VolSurface


def build_demo_iv_surface(spot, base_vol, smile_strength=0.12, term_slope=-0.03):
    """
    Build a smooth demonstration IV surface from a historical-vol anchor.

    This is not a substitute for exchange/OTC option quotes. It gives the demo a
    stable surface shape when only index spot history is available.
    """
    spot = float(spot)
    base_vol = float(np.clip(base_vol, 0.06, 0.75))

    strikes = spot * np.array([0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5])
    maturities = np.array([1 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    iv = np.zeros((len(strikes), len(maturities)))
    for i, K in enumerate(strikes):
        moneyness = np.log(K / spot)
        skew = -0.08 * moneyness
        smile = smile_strength * moneyness**2
        for j, T in enumerate(maturities):
            term = term_slope * (1.0 - np.exp(-1.8 * T))
            iv[i, j] = np.clip(base_vol + skew + smile + term, 0.05, 0.9)

    return VolSurface(strikes, maturities, iv)
