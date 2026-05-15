from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SnowballTerms:
    notional: float = 1_000_000.0
    maturity_years: float = 1.0
    coupon: float = 0.16
    knock_out: float = 1.03
    knock_in: float = 0.75
    observation_frequency: int = 12
    steps_per_year: int = 252
    lockout_months: int = 0
    knock_in_observation: str = "daily"
    knock_out_observation: str = "monthly"
    step_down_enabled: bool = False
    step_down: float = 0.0


def _observation_steps(mode: str, total_steps: int, steps_per_year: int):
    mode = (mode or "monthly").lower()
    if mode == "daily":
        return set(range(1, total_steps + 1))
    if mode == "monthly":
        every = max(1, int(round(steps_per_year / 12)))
    elif mode == "quarterly":
        every = max(1, int(round(steps_per_year / 4)))
    elif mode in {"maturity", "terminal"}:
        return {total_steps}
    else:
        raise ValueError(f"unsupported observation mode: {mode}")
    steps = set(range(every, total_steps + 1, every))
    steps.add(total_steps)
    return steps


class SnowballMCPricer:
    """Monte Carlo demo pricer for a basic autocallable snowball payoff."""

    def __init__(self, local_vol_pricer, r=0.0, q=0.0, seed=42):
        self.lv_pricer = local_vol_pricer
        self.r = float(r)
        self.q = float(q)
        self.seed = seed

    def _simulate_components(self, S0: float, terms: SnowballTerms, paths: int):
        rng = np.random.default_rng(self.seed)
        S0 = float(S0)
        total_steps = max(1, int(round(terms.maturity_years * terms.steps_per_year)))
        dt = terms.maturity_years / total_steps
        ko_steps = _observation_steps(terms.knock_out_observation, total_steps, terms.steps_per_year)
        ki_steps = _observation_steps(terms.knock_in_observation, total_steps, terms.steps_per_year)
        lockout_steps = int(round(max(0, terms.lockout_months) / 12.0 * terms.steps_per_year))
        ko_schedule = sorted(step for step in ko_steps if step > lockout_steps)
        ko_index = {step: idx for idx, step in enumerate(ko_schedule)}

        alive = np.ones(paths, dtype=bool)
        knocked_in = np.zeros(paths, dtype=bool)
        knocked_out = np.zeros(paths, dtype=bool)
        ki_before_ko = np.zeros(paths, dtype=bool)
        pay_time = np.full(paths, terms.maturity_years)
        principal_payoff = np.zeros(paths)
        coupon_accrual = np.zeros(paths)
        S = np.full(paths, S0, dtype=float)
        lv_grid = np.linspace(0.35 * S0, 2.2 * S0, 90)

        for step in range(1, total_steps + 1):
            tau = step * dt
            grid_sigmas = np.array([self.lv_pricer.local_vol(x, tau) for x in lv_grid])
            sigmas = np.interp(np.clip(S, lv_grid[0], lv_grid[-1]), lv_grid, grid_sigmas)
            z = rng.standard_normal(paths)
            S *= np.exp((self.r - self.q - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * z)

            active = alive
            if step in ki_steps:
                knocked_in[active] |= S[active] <= terms.knock_in * S0

            if step in ko_index:
                ko_level = terms.knock_out
                if terms.step_down_enabled:
                    ko_level = max(0.0, terms.knock_out - terms.step_down * ko_index[step])
                ko = active & (S >= ko_level * S0)
                elapsed = step * dt
                principal_payoff[ko] = terms.notional
                coupon_accrual[ko] = terms.notional * elapsed
                pay_time[ko] = elapsed
                knocked_out[ko] = True
                ki_before_ko[ko] = knocked_in[ko]
                alive[ko] = False

        remaining = alive
        remaining_indices = np.where(remaining)[0]
        remaining_loss = np.minimum(S[remaining] / S0 - 1.0, 0.0)
        remaining_knocked_in = knocked_in[remaining]

        ki_indices = remaining_indices[remaining_knocked_in]
        principal_payoff[ki_indices] = terms.notional * (1.0 + remaining_loss[remaining_knocked_in])

        no_ki_indices = remaining_indices[~remaining_knocked_in]
        principal_payoff[no_ki_indices] = terms.notional
        coupon_accrual[no_ki_indices] = terms.notional * terms.maturity_years

        discount = np.exp(-self.r * pay_time)
        return {
            "principal_pv": principal_payoff * discount,
            "coupon_annuity_pv": coupon_accrual * discount,
            "alive": alive,
            "knocked_in": knocked_in,
            "knocked_out": knocked_out,
            "ki_before_ko": ki_before_ko,
            "pay_time": pay_time,
            "terminal": S,
        }

    def _summary(self, components, pv):
        no_ki_no_ko = components["alive"] & ~components["knocked_in"]
        no_ki_ko = components["knocked_out"] & ~components["ki_before_ko"]
        ki_no_ko = components["alive"] & components["knocked_in"]
        ki_ko = components["knocked_out"] & components["ki_before_ko"]
        return {
            "price": float(np.mean(pv)),
            "std_error": float(np.std(pv, ddof=1) / np.sqrt(len(pv))),
            "ko_probability": float(np.mean(components["knocked_out"])),
            "ki_probability": float(np.mean(components["knocked_in"])),
            "bonus_probability": float(np.mean(no_ki_no_ko)),
            "no_ki_ko_probability": float(np.mean(no_ki_ko)),
            "ki_no_ko_probability": float(np.mean(ki_no_ko)),
            "ki_ko_probability": float(np.mean(ki_ko)),
            "probability_total": float(
                np.mean(no_ki_no_ko) + np.mean(no_ki_ko) + np.mean(ki_no_ko) + np.mean(ki_ko)
            ),
            "avg_ko_time": float(np.mean(components["pay_time"][~components["alive"]]))
            if np.any(~components["alive"])
            else 0.0,
            "terminal_mean": float(np.mean(components["terminal"])),
        }

    def price(self, S0: float, terms: SnowballTerms, paths: int = 8000) -> Dict[str, float]:
        components = self._simulate_components(S0, terms, paths)
        pv = components["principal_pv"] + terms.coupon * components["coupon_annuity_pv"]
        return self._summary(components, pv)

    def fair_coupon(self, S0: float, terms: SnowballTerms, paths: int = 8000) -> Dict[str, float]:
        """
        Solve coupon from the par condition:

        notional = E[discounted principal payoff] + coupon * E[discounted accrual].
        """
        components = self._simulate_components(S0, terms, paths)
        principal_pv = float(np.mean(components["principal_pv"]))
        coupon_annuity_pv = float(np.mean(components["coupon_annuity_pv"]))
        coupon = (terms.notional - principal_pv) / coupon_annuity_pv if coupon_annuity_pv > 1e-12 else 0.0
        coupon = float(np.clip(coupon, -1.0, 2.0))
        pv = components["principal_pv"] + coupon * components["coupon_annuity_pv"]
        summary = self._summary(components, pv)
        summary.update(
            {
                "fair_coupon": coupon,
                "principal_pv": principal_pv,
                "coupon_annuity_pv": coupon_annuity_pv,
            }
        )
        return summary
