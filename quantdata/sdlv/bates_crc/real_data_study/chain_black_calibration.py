r"""Calibration of the coherent DLV chain x Black model to a real (or
synthetic) evaluation set.

Free parameters: (kappa, sigma_Y, anchor_half_width) -- the chain's own
generator rate, the Black volatility of the independent factor Y, and the
GEOMETRIC HALF-WIDTH of the fixed anchor grid (config.build_anchors(n,
half_width), anchors = exp(linspace(-half_width,+half_width,n))). The
half-width was originally frozen at a single constant, but diagnostics on
the 2005-11-21 SELECTION date (never a held-out date) showed a fixed wide
grid overshoots realistic SHORT-dated ATM vol at every kappa in bounds
(even the loosest/slowest transition rate still spreads too much mass away
from the centre within a one-week horizon), while a narrow grid cannot
reach far-OTM strikes at LONGER dated expiries -- a single frozen width
cannot serve both. Letting the optimizer choose the width per day (bounded
by config.ANCHOR_HALF_WIDTH_BOUNDS) is the fix used here; anchors within a
single fit are still FIXED (not tampered with mid-optimization beyond
being a function of this one scalar), and the chain's martingale property
still holds exactly for ANY (kappa, half_width) pair, so this remains
"calibrate transition parameters directly," not a free-density fit needing
a separate completion step (comparison_interface.CoherentChainBlackAdapter's
birth-death rates kappa/(gap) give an EXACT zero-drift martingale for ANY
kappa and ANY anchor grid).

Objective: the same spread-unit hinge error already used throughout this
codebase's own SANOS fitting (quantdata/sanos/sanos_tools.py::_compute_market_fit,
and ArbFreePricer.min_half_spread / MkArbFree.min_half_spread, both = 1e-4):

    fit_err_i = (max(0, price_i - ask_i) + max(0, bid_i - price_i))
                / max(0.5*(ask_i - bid_i), 1e-4)

minimized in the least-squares sense (scipy.optimize.least_squares treats
the residual vector's sum of squares as the objective), method="trf",
bounded by config.CHAIN_KAPPA_BOUNDS / config.CHAIN_SIGMA_Y_BOUNDS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from quantdata.sdlv.bates_crc.comparison_interface import (
    ChainGeneratorConfig,
    CoherentChainBlackAdapter,
)
from quantdata.sdlv.bates_crc.real_data_study import config as cfg
from quantdata.sdlv.bates_crc.real_data_study.chain_pricer import price_calls_chain_black
from quantdata.sdlv.bates_crc.real_data_study.day_loader import EvaluationSet
from quantdata.sdlv.bates_crc.verification import chain_composition_residual

MIN_HALF_SPREAD = 1e-4  # quantdata/sanos/arbfree.py ArbFreePricer.min_half_spread / MkArbFree.min_half_spread


class ChainBlackCalibrationError(RuntimeError):
    """Raised when no multi-start attempt converges. Never silently
    returned as if it were a fit."""


def _fit_err(price: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    half_spread = np.maximum(0.5 * (ask - bid), MIN_HALF_SPREAD)
    return (np.maximum(0.0, price - ask) + np.maximum(0.0, bid - price)) / half_spread


@dataclass(frozen=True)
class ChainBlackCalibrationResult:
    chain_config: ChainGeneratorConfig
    sigma_Y: float
    starts: list = field(default_factory=list)
    chosen_start_index: int = -1
    final_cost: float = float("inf")
    converged: bool = False
    chapman_kolmogorov_residual: float = float("nan")
    generator_residual: float = float("nan")

    def price(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        return price_calls_chain_black(self.chain_config, self.sigma_Y, strikes, expiries)


def _multi_start_initial_guesses(n_starts: int, seed: int) -> list[np.ndarray]:
    """Log-spaced kappa guesses (rates plausibly span orders of magnitude)
    crossed with linearly-spaced sigma_Y and half_width guesses, then a
    small random jitter (seeded) so repeated starts are not degenerate to
    each other."""
    rng = np.random.default_rng(seed)
    k_lo, k_hi = cfg.CHAIN_KAPPA_BOUNDS
    s_lo, s_hi = cfg.CHAIN_SIGMA_Y_BOUNDS
    w_lo, w_hi = cfg.ANCHOR_HALF_WIDTH_BOUNDS
    kappas = np.geomspace(max(k_lo, 0.5), k_hi, n_starts)
    sigmas = np.linspace(s_lo + 0.05, min(s_hi, 0.6), n_starts)
    widths = np.linspace(w_lo + 0.02, min(w_hi, 0.30), n_starts)
    rng.shuffle(sigmas)
    rng.shuffle(widths)
    guesses = []
    for kap, sig, wid in zip(kappas, sigmas, widths):
        jitter_k = 1.0 + rng.uniform(-0.1, 0.1)
        jitter_s = 1.0 + rng.uniform(-0.1, 0.1)
        jitter_w = 1.0 + rng.uniform(-0.1, 0.1)
        k = float(np.clip(kap * jitter_k, k_lo, k_hi))
        s = float(np.clip(sig * jitter_s, s_lo, s_hi))
        w = float(np.clip(wid * jitter_w, w_lo, w_hi))
        guesses.append(np.array([k, s, w]))
    return guesses


def calibrate_chain_black(evaluation_set: EvaluationSet, *, n_anchors: int = cfg.N_ANCHORS,
                           n_starts: int = cfg.N_MULTI_START, seed: int = 0) -> ChainBlackCalibrationResult:
    contracts = evaluation_set.contracts
    if len(contracts) == 0:
        raise ChainBlackCalibrationError(
            f"cannot calibrate: evaluation set for {evaluation_set.ticker} {evaluation_set.d} has zero contracts"
        )

    strikes = np.array([c.strike for c in contracts], dtype=float)
    expiries = np.array([c.expiry_years for c in contracts], dtype=float)
    bids = np.array([c.bid for c in contracts], dtype=float)
    asks = np.array([c.ask for c in contracts], dtype=float)
    if np.any(asks <= bids):
        raise ChainBlackCalibrationError("evaluation set contains a contract with ask <= bid; cannot fit")

    def residual(params: np.ndarray) -> np.ndarray:
        kappa, sigma_Y, half_width = params
        anchors, initial_index = cfg.build_anchors(n_anchors, float(half_width))
        chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=float(kappa),
                                             sigma_y=float(sigma_Y))
        price = price_calls_chain_black(chain_config, float(sigma_Y), strikes, expiries)
        return _fit_err(price, bids, asks)

    bounds_lo = np.array([cfg.CHAIN_KAPPA_BOUNDS[0], cfg.CHAIN_SIGMA_Y_BOUNDS[0], cfg.ANCHOR_HALF_WIDTH_BOUNDS[0]])
    bounds_hi = np.array([cfg.CHAIN_KAPPA_BOUNDS[1], cfg.CHAIN_SIGMA_Y_BOUNDS[1], cfg.ANCHOR_HALF_WIDTH_BOUNDS[1]])

    starts_log = []
    best = None
    for i, x0 in enumerate(_multi_start_initial_guesses(n_starts, seed)):
        result = least_squares(residual, x0, bounds=(bounds_lo, bounds_hi), method="trf")
        cost = float(0.5 * np.sum(result.fun ** 2))
        starts_log.append(dict(
            start_index=i, x0=x0.tolist(), x_final=result.x.tolist(), success=bool(result.success),
            status=int(result.status), message=str(result.message), nfev=int(result.nfev), cost=cost,
        ))
        if result.success and (best is None or cost < best[0]):
            best = (cost, i, result.x)

    if best is None:
        raise ChainBlackCalibrationError(
            f"all {n_starts} multi-start attempts failed to converge for {evaluation_set.ticker} "
            f"{evaluation_set.d}; see starts log: {starts_log}"
        )

    final_cost, chosen_index, x_final = best
    kappa_final, sigma_final, half_width_final = float(x_final[0]), float(x_final[1]), float(x_final[2])
    anchors, initial_index = cfg.build_anchors(n_anchors, half_width_final)
    chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=kappa_final,
                                         sigma_y=sigma_final)

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

    return ChainBlackCalibrationResult(
        chain_config=chain_config, sigma_Y=sigma_final, starts=starts_log, chosen_start_index=chosen_index,
        final_cost=final_cost, converged=True, chapman_kolmogorov_residual=ck_residual,
        generator_residual=gen_residual,
    )
