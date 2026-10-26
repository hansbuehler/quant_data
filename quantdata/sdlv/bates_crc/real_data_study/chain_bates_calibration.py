r"""Calibration of the coherent DLV chain x ordinary Bates model to a real
(or synthetic) evaluation set. Mirrors chain_black_calibration.py's
structure; see that module's docstring for the shared design rationale
(fixed anchors, transition-rate-only calibration, spread-unit hinge
objective, multi-start with recorded diagnostics, post-fit martingale
verification).

Free parameters (10): chain kappa, V_t, the Bates background
(heston.kappa, heston.theta, heston.xi, heston.rho, jumps.lambda_J,
jumps.m_J, jumps.delta_J), and the anchor grid's geometric half-width
(see chain_black_calibration.py's docstring for why this is fitted rather
than frozen: a single fixed width could not span both short-dated ATM
curvature and far-OTM long-dated coverage on the 2005-11-21 SELECTION-date
diagnostics). `ChainGeneratorConfig.sigma_y` is unused by
price_calls_chain_bates (Bates supplies its own volatility structure) and
is set to 0.0 as a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from quantdata.sdlv.bates_crc.comparison_interface import (
    ChainGeneratorConfig,
    CoherentChainBlackAdapter,
)
from quantdata.sdlv.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.sdlv.bates_crc.real_data_study import config as cfg
from quantdata.sdlv.bates_crc.real_data_study.chain_pricer import price_calls_chain_bates
from quantdata.sdlv.bates_crc.real_data_study.day_loader import EvaluationSet
from quantdata.sdlv.bates_crc.verification import chain_composition_residual

MIN_HALF_SPREAD = 1e-4  # quantdata/sanos/arbfree.py ArbFreePricer.min_half_spread / MkArbFree.min_half_spread
V_T_BOUNDS = cfg.BATES_BOUNDS["theta"]  # V_t plausible range mirrors the long-run variance bound

# order of the 10-vector optimized by least_squares
_PARAM_NAMES = ("chain_kappa", "V_t", "heston_kappa", "heston_theta", "heston_xi", "heston_rho",
                 "lambda_J", "m_J", "delta_J", "anchor_half_width")


class ChainBatesCalibrationError(RuntimeError):
    """Raised when no multi-start attempt converges. Never silently
    returned as if it were a fit."""


def _fit_err(price: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    half_spread = np.maximum(0.5 * (ask - bid), MIN_HALF_SPREAD)
    return (np.maximum(0.0, price - ask) + np.maximum(0.0, bid - price)) / half_spread


def _bounds_vectors() -> tuple[np.ndarray, np.ndarray]:
    b = cfg.BATES_BOUNDS
    lo = np.array([cfg.CHAIN_KAPPA_BOUNDS[0], V_T_BOUNDS[0], b["kappa"][0], b["theta"][0], b["xi"][0], b["rho"][0],
                    b["lambda_J"][0], b["m_J"][0], b["delta_J"][0], cfg.ANCHOR_HALF_WIDTH_BOUNDS[0]])
    hi = np.array([cfg.CHAIN_KAPPA_BOUNDS[1], V_T_BOUNDS[1], b["kappa"][1], b["theta"][1], b["xi"][1], b["rho"][1],
                    b["lambda_J"][1], b["m_J"][1], b["delta_J"][1], cfg.ANCHOR_HALF_WIDTH_BOUNDS[1]])
    return lo, hi


def _unpack(params: np.ndarray) -> tuple[float, float, HestonParams, GaussianJumpParams, float]:
    chain_kappa, v_t, hk, ht, hx, hr, lj, mj, dj, half_width = params
    heston = HestonParams(kappa=float(hk), theta=float(ht), xi=float(hx), rho=float(hr))
    jumps = GaussianJumpParams(lambda_J=float(lj), m_J=float(mj), delta_J=float(dj))
    return float(chain_kappa), float(v_t), heston, jumps, float(half_width)


@dataclass(frozen=True)
class ChainBatesCalibrationResult:
    chain_config: ChainGeneratorConfig
    heston: HestonParams
    jumps: GaussianJumpParams
    V_t: float
    starts: list = field(default_factory=list)
    chosen_start_index: int = -1
    final_cost: float = float("inf")
    converged: bool = False
    chapman_kolmogorov_residual: float = float("nan")
    generator_residual: float = float("nan")

    def price(self, strikes: np.ndarray, expiries: np.ndarray, *, n_quad: int = cfg.FOURIER_N_QUAD,
              u_max: float = cfg.FOURIER_U_MAX) -> np.ndarray:
        return price_calls_chain_bates(self.chain_config, self.heston, self.jumps, self.V_t, strikes, expiries,
                                        n_quad=n_quad, u_max=u_max)


def _multi_start_initial_guesses(n_starts: int, seed: int) -> list[np.ndarray]:
    """One base guess at the BATES_BOUNDS midpoints (plus a plausible chain
    kappa / V_t), then (n_starts-1) further guesses jittered +/-25%
    multiplicatively around that base, seeded for reproducibility."""
    lo, hi = _bounds_vectors()
    base = 0.5 * (lo + hi)
    base[0] = 10.0  # chain_kappa midpoint of [0.5,150] is dominated by the upper bound; use a moderate value instead
    base[1] = 0.04  # V_t: a moderate variance level, not the bound midpoint
    base[9] = 0.15  # anchor_half_width: a moderate compromise per 2005-11-21 selection-date diagnostics
    rng = np.random.default_rng(seed)
    guesses = [base.copy()]
    for _ in range(max(0, n_starts - 1)):
        jitter = 1.0 + rng.uniform(-0.25, 0.25, size=base.shape)
        g = np.clip(base * jitter, lo, hi)
        # rho can flip sign under multiplicative jitter in an unhelpful way; keep it negative (equity skew prior)
        g[5] = -abs(g[5]) if base[5] < 0 else g[5]
        guesses.append(g)
    return guesses


def calibrate_chain_bates(evaluation_set: EvaluationSet, *, n_anchors: int = cfg.N_ANCHORS,
                           n_starts: int = cfg.N_MULTI_START, seed: int = 0,
                           n_quad: int = cfg.FOURIER_N_QUAD, u_max: float = cfg.FOURIER_U_MAX
                           ) -> ChainBatesCalibrationResult:
    contracts = evaluation_set.contracts
    if len(contracts) == 0:
        raise ChainBatesCalibrationError(
            f"cannot calibrate: evaluation set for {evaluation_set.ticker} {evaluation_set.d} has zero contracts"
        )

    strikes = np.array([c.strike for c in contracts], dtype=float)
    expiries = np.array([c.expiry_years for c in contracts], dtype=float)
    bids = np.array([c.bid for c in contracts], dtype=float)
    asks = np.array([c.ask for c in contracts], dtype=float)
    if np.any(asks <= bids):
        raise ChainBatesCalibrationError("evaluation set contains a contract with ask <= bid; cannot fit")

    lo, hi = _bounds_vectors()

    def residual(params: np.ndarray) -> np.ndarray:
        chain_kappa, v_t, heston, jumps, half_width = _unpack(params)
        anchors, initial_index = cfg.build_anchors(n_anchors, half_width)
        chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=chain_kappa,
                                             sigma_y=0.0)
        price = price_calls_chain_bates(chain_config, heston, jumps, v_t, strikes, expiries, n_quad=n_quad,
                                         u_max=u_max)
        return _fit_err(price, bids, asks)

    starts_log = []
    best = None
    for i, x0 in enumerate(_multi_start_initial_guesses(n_starts, seed)):
        try:
            result = least_squares(residual, x0, bounds=(lo, hi), method="trf")
        except Exception as exc:  # noqa: BLE001 -- record solver crashes, don't abort the sweep
            starts_log.append(dict(start_index=i, x0=x0.tolist(), success=False, status=None,
                                    message=f"exception: {exc}", nfev=None, cost=float("inf")))
            continue
        cost = float(0.5 * np.sum(result.fun ** 2))
        starts_log.append(dict(
            start_index=i, x0=x0.tolist(), x_final=result.x.tolist(), success=bool(result.success),
            status=int(result.status), message=str(result.message), nfev=int(result.nfev), cost=cost,
        ))
        if result.success and (best is None or cost < best[0]):
            best = (cost, i, result.x)

    if best is None:
        raise ChainBatesCalibrationError(
            f"all {n_starts} multi-start attempts failed to converge for {evaluation_set.ticker} "
            f"{evaluation_set.d}; see starts log: {starts_log}"
        )

    final_cost, chosen_index, x_final = best
    chain_kappa, v_t, heston, jumps, half_width = _unpack(x_final)
    anchors, initial_index = cfg.build_anchors(n_anchors, half_width)
    chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=chain_kappa, sigma_y=0.0)

    unique_tau = np.unique(expiries)
    tau1 = float(unique_tau[0])
    tau2 = float(unique_tau[len(unique_tau) // 2]) if len(unique_tau) > 1 else float(unique_tau[0]) + 1e-3
    ck_residual = chain_composition_residual(chain_config, tau1, tau2)

    adapter = CoherentChainBlackAdapter(chain_config)
    gen_report = adapter.verify_zero_generator_residual(tau_years=tau1, k=1.0, y=1.0)
    gen_residual = float(abs(gen_report["residual"]))

    if ck_residual >= 1e-6:
        raise RuntimeError(
            f"Chapman-Kolmogorov residual {ck_residual} >= 1e-6 for a calibrated chain -- this indicates a bug "
            "in the (parameter-independent) chain construction, not a data issue."
        )
    if gen_residual >= 1e-3:
        raise RuntimeError(
            f"generator residual {gen_residual} >= 1e-3 (finite-difference floor) for a calibrated chain -- "
            "this indicates a bug in the (parameter-independent) chain construction, not a data issue."
        )

    return ChainBatesCalibrationResult(
        chain_config=chain_config, heston=heston, jumps=jumps, V_t=v_t, starts=starts_log,
        chosen_start_index=chosen_index, final_cost=final_cost, converged=True,
        chapman_kolmogorov_residual=ck_residual, generator_residual=gen_residual,
    )
