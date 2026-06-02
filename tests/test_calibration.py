import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dupire import LocalVolPricer
from core.vol_surface import VolSurface
from data.sse_option_iv import _fourth_wednesday, _parse_contract
from pricer.pde_pricer import PDEPricer
from pricer.snowball_mc import SnowballMCPricer, SnowballTerms


class ConstantLocalVol:
    def __init__(self, sigma):
        self.sigma = sigma

    def local_vol(self, K, T):
        return self.sigma


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


def test_dupire_diagnostics_are_clean_for_flat_vol():
    surface = flat_surface(0.22)
    lv = LocalVolPricer(surface, spot=100.0, r=0.03, q=0.01)

    diagnostics = lv.local_vol_diagnostics(strikes=[80.0, 100.0, 120.0], maturities=[0.25, 1.0])

    assert diagnostics["points"] == 6
    assert diagnostics["positive_variance_ratio"] == 1.0
    assert diagnostics["positive_density_ratio"] == 1.0
    assert diagnostics["positive_numerator_ratio"] == 1.0
    assert diagnostics["fallback_ratio"] == 0.0
    assert abs(diagnostics["min_local_vol"] - 0.22) < 2e-3
    assert abs(diagnostics["max_local_vol"] - 0.22) < 2e-3


def test_pde_matches_black_scholes_for_flat_vol():
    surface = flat_surface(0.22)
    lv = LocalVolPricer(surface, spot=100.0, r=0.03, q=0.01)
    pde = PDEPricer(lv, r=0.03, q=0.01, space_steps=160, time_steps=160)

    target = surface.black_scholes_call(100.0, 100.0, 1.0, 0.03, 0.01)
    price = pde.price_european(100.0, 100.0, 1.0, is_call=True)

    assert abs(price - target) < 0.25


def test_snowball_path_probabilities_are_exhaustive():
    pricer = SnowballMCPricer(ConstantLocalVol(0.25), seed=7)
    terms = SnowballTerms(
        maturity_years=1.0,
        knock_in_observation="daily",
        knock_out_observation="monthly",
        steps_per_year=252,
    )

    result = pricer.fair_coupon(100.0, terms, paths=1000)

    assert abs(result["probability_total"] - 1.0) < 1e-12
    assert abs(
        result["ki_probability"] - result["ki_no_ko_probability"] - result["ki_ko_probability"]
    ) < 1e-12
    assert abs(
        result["ko_probability"] - result["no_ki_ko_probability"] - result["ki_ko_probability"]
    ) < 1e-12


def test_lockout_reduces_early_knockout_probability():
    pricer = SnowballMCPricer(ConstantLocalVol(0.2), seed=11)
    base = SnowballTerms(knock_out_observation="monthly", lockout_months=0, steps_per_year=252)
    locked = SnowballTerms(knock_out_observation="monthly", lockout_months=6, steps_per_year=252)

    base_result = pricer.fair_coupon(100.0, base, paths=1000)
    locked_result = pricer.fair_coupon(100.0, locked, paths=1000)

    assert locked_result["ko_probability"] <= base_result["ko_probability"]


def test_sse_option_expiry_uses_fourth_wednesday():
    assert _fourth_wednesday(2026, 9).isoformat() == "2026-09-23"
    assert _fourth_wednesday(2026, 12).isoformat() == "2026-12-23"

    _, expiry, strike = _parse_contract("510050C2609M02700")

    assert expiry.isoformat() == "2026-09-23"
    assert strike == 2.7


def test_unclipped_fair_coupon_solves_par_condition_when_not_clipped():
    pricer = SnowballMCPricer(ConstantLocalVol(0.2), seed=19)
    terms = SnowballTerms(knock_out_observation="monthly", knock_in_observation="daily", steps_per_year=252)

    result = pricer.fair_coupon(100.0, terms, paths=1000)

    assert not result["fair_coupon_clipped"]
    par_value = result["principal_pv"] + result["unclipped_fair_coupon"] * result["coupon_annuity_pv"]
    assert abs(par_value - terms.notional) < 1e-8
