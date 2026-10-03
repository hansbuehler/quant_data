r"""Section 4 / Task 4: the coherent latent DLV x extended-Bates product
construction, `thm:dlv-bates`, connected to the Section 12 reserve-CRC
Bates factor built in `bates_transform.py`.

X: the SAME finite-state nearest-neighbour absorbing chain already built
   and independently verified (exact zero generator residual, hence an
   exact martingale by construction -- Duffie-style birth-death rates
   `kappa/(gap)` give zero drift at every interior state) by
   `comparison_interface.CoherentChainBlackAdapter` -- reused here through
   its public `transition_matrix` method, not reimplemented. This is the
   "small finite-state martingale benchmark with exact transitions"
   required before any jump-extended or Wiese-Wissel construction is
   attempted.
Y: the reserve-CRC extended-Bates factor from `bates_transform.py`.

Independence of X and Y is a MODELLING ASSUMPTION of `thm:dlv-bates` (see
`AUDIT.md` Sec. 2) -- this module cannot verify independence from data (no
data is used; `DATA_STATUS.md` already recorded that market data are
unavailable, and this task must run without it). What it DOES verify are
the ALGEBRAIC CONSEQUENCES of the independence + individual-martingale
assumptions: the mixture pricing formula's mass/positivity and the
tower/forward-factorisation identity `E[S_T|F_t] = E[X_T|F_t] E[Y_T|F_t]
= S_t`. This is exactly the "verify positivity/tower/settlement... before
connecting to Wiese-Wissel" instruction -- no jump-extended finite-
coordinate drift identity is needed here because the martingale chain
already has zero drift by construction (see module docstring of
`comparison_interface.py`), so that fallback is not exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from quantdata.dysanos.bates_crc.bates_transform import (
    FourierPricingConfig,
    continuation_transform,
    fourier_call_price,
)
from quantdata.dysanos.bates_crc.comparison_interface import ChainGeneratorConfig, CoherentChainBlackAdapter
from quantdata.dysanos.bates_crc.heston_riccati import HestonParams
from quantdata.dysanos.bates_crc.reserve_crc import BcrcState, GaussianJumpParams, ReserveSchedule


@dataclass(frozen=True)
class DlvBatesProductModel:
    chain_config: ChainGeneratorConfig
    heston: HestonParams
    state: BcrcState
    schedule: ReserveSchedule
    envelope: GaussianJumpParams
    V_t: float
    fourier_config: FourierPricingConfig = FourierPricingConfig()
    n_reserve_quad: int = 200

    def x_chain(self) -> CoherentChainBlackAdapter:
        return CoherentChainBlackAdapter(self.chain_config)

    def background_state_weights(self, t: float, T: float) -> np.ndarray:
        """q_i(t,T) = P[X_T = x_i | X_t = x_{i0}], the chain's own row of
        its transition matrix -- positivity/mass are checked in the tests
        directly against this method, not against the priced output."""
        i0 = self.chain_config.initial_index
        tau = T - t
        if tau <= 0.0:
            q = np.zeros(len(self.chain_config.anchors))
            q[i0] = 1.0
            return q
        P = self.x_chain().transition_matrix(tau)
        return P[i0, :]

    def price(self, t: float, T: float, K: float) -> float:
        """C_t(T,K) = sum_i q_i(t,T) G_t(x_i;T,K), the manuscript's own
        finite-mixture pricing formula (eq:chain-price generalised to the
        Bates Y-factor): G_t(x;T,K) = E_Q[(x Y_T - K)^+ | G_t], computed by
        fourier_call_price with y_forward=x."""
        x = np.asarray(self.chain_config.anchors, dtype=float)
        i0 = self.chain_config.initial_index
        tau = T - t
        if tau <= 0.0:
            return float(max(x[i0] * 1.0 - K, 0.0))
        q = self.background_state_weights(t, T)
        total = 0.0
        for i, xi in enumerate(x):
            if q[i] <= 0.0:
                continue
            total += q[i] * fourier_call_price(
                K, t, T, self.V_t, self.heston, self.state, self.schedule, self.envelope,
                self.fourier_config, y_forward=float(xi), n_reserve_quad=self.n_reserve_quad,
            )
        return float(total)

    def forward_price(self, t: float, T: float) -> float:
        """E[S_T | F_t] via the mixture, using the FULL transform machinery
        (including reserve corrections) for E[Y_T|F_t] rather than
        hardcoding the martingale-normalisation value 1.0 -- a genuine,
        non-trivial regression check that the assembled stack reproduces
        it. Under X/Y independence and each factor's own martingale
        property this must equal S_t = x_{i0}*1 exactly; checked, not
        assumed, in the tests."""
        x = np.asarray(self.chain_config.anchors, dtype=float)
        i0 = self.chain_config.initial_index
        tau = T - t
        if tau <= 0.0:
            return float(x[i0])
        q = self.background_state_weights(t, T)
        e_x = float(q @ x)
        e_y = float(np.real(continuation_transform(
            -1j, t, T, self.V_t, self.heston, self.state, self.schedule, self.envelope,
        )))
        return e_x * e_y


class DlvBatesCrcAdapter:
    """Wraps DlvBatesProductModel to satisfy comparison_interface's
    PriceModelAdapter protocol, replacing the prior task's
    NotImplementedAdapter("coherent_dlv_x_extended_bates_crc", ...) stub
    now that this task has actually built the construction it pointed to.
    valuation_date is treated as t=0 in year-fraction units, matching
    MODEL_CONVENTIONS.md's day-count convention already used by
    Dy2FlexAdapter."""

    name = "coherent_dlv_x_extended_bates_crc"

    def __init__(self, model: "DlvBatesProductModel"):
        self.model = model

    def is_implemented(self) -> bool:
        return True

    def price(self, valuation_date: date, contracts: list) -> np.ndarray:
        prices = np.empty(len(contracts))
        for idx, c in enumerate(contracts):
            tau = (c.expiry - valuation_date).days / 365.25
            prices[idx] = self.model.price(0.0, tau, c.pure_strike)
        return prices
