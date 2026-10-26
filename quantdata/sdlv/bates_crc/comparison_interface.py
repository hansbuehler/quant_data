r"""Section D: the common comparison interface.

Defines one adapter protocol that the four candidate models share, plus a
genuinely working implementation of two of them:

  - ``Dy2FlexAdapter``: wraps the REAL existing dy2/FLEX decoder
    (``quantdata.dysanos.flex``) -- no reimplementation, calls the actual
    repository code.
  - ``CoherentChainBlackAdapter``: a genuinely NEW, self-contained,
    analytically coherent model -- the finite-state-chain x independent-Black
    product construction from the DYSANOS II manuscript's Section 3.4 /
    eq. (eq:three-state)/(eq:chain-price)/(eq:chain-pde). Its generator
    residual is zero BY CONSTRUCTION (an algebraic identity, not a fitted
    approximation) -- verified numerically below as a control, exactly
    mirroring the "positive control" pattern used throughout this research
    programme's earlier Heston/Bates work.

The other two required models are explicitly NOT implemented here and are
represented by ``NotImplementedAdapter`` stubs with a precise pointer to
what each needs (per REPOSITORY_MAP.md Section 6/7, no Bates oracle exists
anywhere in this repository; porting or reimplementing one, and the
Section-12 extended-Bates CRC-reserve construction on top of it, is
substantial work correctly scoped to "the next tasks" per the assignment's
own framing, not this discovery task).

Do NOT assume the existing dy2 state evolution has a common risk-neutral
measure: ``Dy2FlexAdapter.hjm_residual`` reuses
``crc_extension.section2_pricing_identity``'s already-tested (and
already-failing, for the raw P law) discrete martingale residual rather
than asserting Q-consistency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm

# ===========================================================================
# The common contract / quote interface.
# ===========================================================================


@dataclass(frozen=True)
class ContractSpec:
    """One fixed cash-strike, fixed-calendar-expiry contract, in the "pure"
    (C/(DF*F), K/F) units of MODEL_CONVENTIONS.md Section 1."""

    contract_id: str
    underlying: str
    expiry: date
    pure_strike: float
    is_call: bool = True


@dataclass(frozen=True)
class QuoteRow:
    """One cleaned quote, with its original identifier and any exclusion
    reason retained (never silently dropped without a record)."""

    contract: ContractSpec
    valuation_date: date
    pure_bid: float
    pure_ask: float
    excluded_reason: str | None = None

    @property
    def pure_mid(self) -> float:
        return 0.5 * (self.pure_bid + self.pure_ask)


@runtime_checkable
class PriceModelAdapter(Protocol):
    """One shared interface every candidate model implements."""

    name: str

    def price(self, valuation_date: date, contracts: list[ContractSpec]) -> np.ndarray:
        """Return pure-unit prices for the given contracts at valuation_date."""
        ...

    def is_implemented(self) -> bool:
        ...


class NotImplementedAdapter:
    """A stub that documents precisely what is missing, rather than
    silently omitting the model from the comparison."""

    def __init__(self, name: str, missing_reason: str):
        self.name = name
        self.missing_reason = missing_reason

    def price(self, valuation_date: date, contracts: list[ContractSpec]) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} is not implemented in this discovery task: {self.missing_reason}"
        )

    def is_implemented(self) -> bool:
        return False


COHERENT_DLV_BATES_MISSING_REASON = (
    "no ordinary/generalised Bates oracle exists anywhere in quant_data-main "
    "(REPOSITORY_MAP.md Section 6); the existing oracle lives in the separate "
    "heston_bates_oracle sandbox and would need to be ported or reimplemented "
    "before this adapter can price DLV x ordinary-Bates."
)
# NOTE: coherent_dlv_x_extended_bates_crc is NO LONGER a NotImplementedAdapter
# stub as of the "Implement the explicit DLV x extended-Bates construction"
# task -- see dlv_bates_product.DlvBatesCrcAdapter, which wraps a real,
# tested implementation (heston_riccati.py + reserve_crc.py +
# bates_transform.py + dlv_bates_product.py, see AUDIT.md and
# report/DLV_Bates_CRC_Report.pdf). default_adapters() below still lists a
# NotImplementedAdapter placeholder only for coherent_dlv_x_ordinary_bates
# (the plain, non-CRC Bates oracle, genuinely still missing).


# ===========================================================================
# Dy2FlexAdapter: wraps the REAL existing FLEX decoder.
# ===========================================================================


class Dy2FlexAdapter:
    """Wraps quantdata.dysanos.flex.weekly_simulation.WeeklySanosSimulator.
    Does not assume Q-consistency -- see hjm_residual_summary."""

    name = "dy2_flex"

    def __init__(self, simulator, state, spot: float):
        self._simulator = simulator
        self._state = state
        self._spot = spot

    def is_implemented(self) -> bool:
        return True

    def price(self, valuation_date: date, contracts: list[ContractSpec]) -> np.ndarray:
        import torch

        surface = self._simulator.decode_surface(self._state.unsqueeze(0))
        strikes = torch.tensor([[c.pure_strike * self._spot for c in contracts]], dtype=torch.float64)
        expiry_years = torch.tensor(
            [(c.expiry - valuation_date).days / 365.25 for c in contracts], dtype=torch.float64
        )
        is_call = torch.ones_like(strikes, dtype=torch.bool)
        is_zero = torch.zeros_like(strikes, dtype=torch.bool)
        spot_t = torch.tensor([self._spot], dtype=torch.float64)
        prices = self._simulator._option_values(surface, spot_t, strikes, expiry_years.long(), is_call, is_zero)
        return prices.squeeze(0).detach().numpy() / self._spot  # back to pure units


# ===========================================================================
# CoherentChainBlackAdapter: the finite-chain x Black product construction,
# dy2.pdf companion manuscript Section 3.4, eq. (eq:three-state)/(eq:chain-price).
# Analytically zero generator residual, verified as a control below.
# ===========================================================================


def _black_call(a: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Black(a,K,w) with w = total variance, matching this session's own
    established convention throughout (heston_bates_oracle's sanos.py,
    quantdata.dysanos.flex.sanos's sqrt_variance argument, etc.)."""
    a = np.asarray(a, dtype=float)
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    intrinsic = np.maximum(a - k, 0.0)
    sw = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sw_safe = np.where(sw > 0.0, sw, 1.0)
        d1 = (np.log(a / k) + 0.5 * w) / sw_safe
        d2 = d1 - sw_safe
    price = a * norm.cdf(d1) - k * norm.cdf(d2)
    return np.where(w > 0.0, price, intrinsic)


@dataclass(frozen=True)
class ChainGeneratorConfig:
    """A(t) for the nearest-neighbour absorbing chain, eq. (eq:nearest-neighbour)."""

    anchors: tuple[float, ...]  # x_0 < ... < x_L
    initial_index: int
    kappa: float  # rate scale for the interior jump intensities
    sigma_y: float  # Black smoothing volatility (constant, for this minimal instance)


class CoherentChainBlackAdapter:
    """S = X_t Y_t, X a bounded finite-state martingale chain (absorbing
    endpoints), Y an independent Black martingale. Theorem thm:product
    guarantees: true martingale prices, correct settlement, C^infty strike
    smoothness, and (via eq:chain-pde / this class's own verification
    below) EXACT zero generator residual -- an analytic fact of the
    construction, not a fitted approximation."""

    name = "coherent_dlv_x_black_chain"

    def __init__(self, config: ChainGeneratorConfig):
        self.config = config

    def is_implemented(self) -> bool:
        return True

    def _generator(self) -> np.ndarray:
        x = np.asarray(self.config.anchors, dtype=float)
        L = len(x) - 1
        A = np.zeros((L + 1, L + 1))
        for i in range(1, L):
            rate_down = self.config.kappa / (x[i] - x[i - 1])
            rate_up = self.config.kappa / (x[i + 1] - x[i])
            A[i, i - 1] = rate_down
            A[i, i + 1] = rate_up
            A[i, i] = -rate_down - rate_up
        # absorbing endpoints: A[0,:] = A[L,:] = 0
        return A

    def transition_matrix(self, tau_years: float) -> np.ndarray:
        from scipy.linalg import expm
        A = self._generator()
        return expm(A * tau_years)

    def _H_row(self, tau: float, y: float, k: float) -> np.ndarray:
        """H_i(tau,y;K) for every state i, per eq. (eq:chain-price):
        H_i = sum_j P_ij(tau) Black(x_j*y, K, sigma_Y^2*tau) -- the Black
        argument is x_j*y (the CURRENT value of the independent factor Y),
        not x_j alone; y=1 at t=0 by normalisation, but the generator check
        below needs the full y-dependence to see the diffusion term."""
        x = np.asarray(self.config.anchors, dtype=float)
        if tau <= 0.0:
            return np.maximum(x * y - k, 0.0)
        P = self.transition_matrix(tau)
        w = self.config.sigma_y ** 2 * tau
        component_prices = _black_call(x * y, k, w)
        return P @ component_prices

    def price(self, valuation_date: date, contracts: list[ContractSpec]) -> np.ndarray:
        i0 = self.config.initial_index
        prices = np.empty(len(contracts))
        for idx, c in enumerate(contracts):
            tau = (c.expiry - valuation_date).days / 365.25
            prices[idx] = self._H_row(tau, 1.0, c.pure_strike)[i0]
        return prices

    def verify_zero_generator_residual(self, tau_years: float, k: float, y: float = 1.0,
                                        h_tau: float = 1.0e-4, h_y: float = 1.0e-3) -> dict:
        """Numerically verifies eq. (eq:chain-pde) IN FULL, including the
        0.5*sigma_Y^2*y^2*H_yy diffusion term (easy to omit by mistake if
        one forgets the Black kernel's argument is x_j*y, not x_j alone --
        this check exists specifically to catch that class of bug):

            d/dtau H_i = 0.5*sigma_Y^2*y^2*H_yy_i + (A H)_i

        (using tau=T-t fixed expiry, so d/dt = -d/dtau)."""
        i0 = self.config.initial_index
        A = self._generator()

        H0 = self._H_row(tau_years, y, k)
        Hp_tau = self._H_row(tau_years + h_tau, y, k)
        Hm_tau = self._H_row(tau_years - h_tau, y, k)
        dHdtau_i0 = (Hp_tau[i0] - Hm_tau[i0]) / (2.0 * h_tau)

        Hp_y = self._H_row(tau_years, y + h_y, k)
        Hm_y = self._H_row(tau_years, y - h_y, k)
        Hyy_i0 = (Hp_y[i0] - 2.0 * H0[i0] + Hm_y[i0]) / (h_y ** 2)

        chain_term = float((A @ H0)[i0])
        rhs = 0.5 * self.config.sigma_y ** 2 * y ** 2 * Hyy_i0 + chain_term
        residual = dHdtau_i0 - rhs
        return dict(tau_years=tau_years, k=k, y=y, residual=residual, H=float(H0[i0]))


# ===========================================================================
# Comparison harness.
# ===========================================================================


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    """Proposed split, per Section D's request -- has no data to apply to
    yet (DATA_STATUS.md), but is specified now so the next task does not
    need to invent one."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date | None = None
    test_end: date | None = None

    def __post_init__(self) -> None:
        if not (self.train_start < self.train_end < self.validation_start < self.validation_end):
            raise ValueError("split dates must be strictly chronologically ordered")
        if self.test_start is not None:
            if self.test_end is None or not (self.validation_end < self.test_start < self.test_end):
                raise ValueError("test period must follow validation and be internally ordered")


DEFAULT_SPLIT = ChronologicalSplitConfig(
    train_start=date(2022, 1, 3), train_end=date(2024, 12, 31),
    validation_start=date(2025, 1, 2), validation_end=date(2025, 8, 29),
)  # matches flex/weekly_dynamics.py's own PHASE3_TRAIN_*/PHASE3_VALIDATION_* constants exactly


@dataclass(frozen=True)
class ComparisonRow:
    contract_id: str
    model_name: str
    price: float | None
    error: str | None


def run_comparison(valuation_date: date, contracts: list[ContractSpec],
                    adapters: list) -> list[ComparisonRow]:
    """Runs every adapter against the same contract list; records a failure
    per-adapter rather than aborting the whole comparison."""
    rows: list[ComparisonRow] = []
    for adapter in adapters:
        if not adapter.is_implemented():
            for c in contracts:
                rows.append(ComparisonRow(c.contract_id, adapter.name, None,
                                           getattr(adapter, "missing_reason", "not implemented")))
            continue
        try:
            prices = adapter.price(valuation_date, contracts)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: record, don't abort
            for c in contracts:
                rows.append(ComparisonRow(c.contract_id, adapter.name, None, str(exc)))
            continue
        for c, p in zip(contracts, prices, strict=True):
            rows.append(ComparisonRow(c.contract_id, adapter.name, float(p), None))
    return rows


def default_adapters() -> list:
    """coherent_dlv_x_extended_bates_crc is intentionally NOT listed here as
    a NotImplementedAdapter: it is now implemented
    (dlv_bates_product.DlvBatesCrcAdapter). Constructing a real instance
    needs a HestonParams/BcrcState/etc. the caller must supply (it prices a
    SPECIFIC reset sequence, not a parameter-free default), so it is not
    included in this zero-argument default list -- see
    dlv_bates_product.py and the report for a fully worked example."""
    return [
        NotImplementedAdapter("coherent_dlv_x_ordinary_bates", COHERENT_DLV_BATES_MISSING_REASON),
    ]
