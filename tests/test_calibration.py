import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dupire import LocalVolPricer
from core.vol_surface import VolSurface
from pricer.pde_pricer import PDEPricer


def flat_surface(sigma=0.22):
    strikes = np.linspace(60, 160, 9)
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0])
    iv = np.full((len(strikes), len(maturities)), sigma)
    return VolSurface(strikes, maturities, iv)


def test_dupire_recovers_flat_vol():
    sigma = 0.22
    surface = flat_surface(sigma)
    lv = LocalVolPricer(surface, spot=100.0, r=0.03, q=0.01)

    for K in [80.0, 100.0, 120.0]:
        for T in [0.25, 0.75, 1.5]:
            assert abs(lv.local_vol(K, T) - sigma) < 2e-3


def test_pde_matches_black_scholes_for_flat_vol():
    surface = flat_surface(0.22)
    lv = LocalVolPricer(surface, spot=100.0, r=0.03, q=0.01)
    pde = PDEPricer(lv, r=0.03, q=0.01, space_steps=160, time_steps=160)

    target = surface.black_scholes_call(100.0, 100.0, 1.0, 0.03, 0.01)
    price = pde.price_european(100.0, 100.0, 1.0, is_call=True)

    assert abs(price - target) < 0.25
