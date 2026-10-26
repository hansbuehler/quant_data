r"""Section C: the reserve trade-off study. Built on top of an ordinary-
Bates calibration (chain_bates_calibration.ChainBatesCalibrationResult).

Three things this module answers, all on REAL calibrated single-day
parameters (not the purely synthetic Section-12 sequence from the prior
task):

1. `build_reserve_setup` + `price_with_reserve`: for one FIXED calibrated
   background (heston, jumps, chain), how much does turning on a reserve
   regime (of a given budget, at a hypothetical next reset date) change
   TODAY's prices, relative to the ordinary-Bates fit -- BEFORE any
   recalibration is attempted. Per AUDIT.md Sec. 8/10, this is nonzero for
   every expiry (not just those past the reset) because the reserve regime
   replaces the priced jump law with the dominating envelope everywhere,
   not only after a reset event.
2. `calibrate_chain_extended_bates`: refit background+chain parameters
   UNDER the extended-Bates+reserve regime (reset date and reserve budget
   held fixed), to see how much of the fit-quality cost from (1) is
   recovered by recalibration.
3. `admissible_theta_interval` + `reset_surface_change`: the admissible
   upward/downward theta movement at the hypothetical reset, and the
   resulting change in future variance / surface dynamics for a
   representative admissible choice -- reusing `reserve_crc.apply_reset`
   and `chain_pricer.price_calls_chain_extended_bates` directly.
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
from quantdata.sdlv.bates_crc.real_data_study.chain_bates_calibration import (
    ChainBatesCalibrationResult,
)
from quantdata.sdlv.bates_crc.real_data_study.chain_pricer import (
    price_calls_chain_bates,
    price_calls_chain_extended_bates,
)
from quantdata.sdlv.bates_crc.real_data_study.day_loader import EvaluationSet
from quantdata.sdlv.bates_crc.reserve_crc import (
    BcrcState,
    JumpBox,
    ReserveSchedule,
    admissible_delta_interval,
    apply_reset,
    dominating_envelope_params,
    initial_state,
    verify_envelope_dominates,
)
from quantdata.sdlv.bates_crc.verification import chain_composition_residual

MIN_HALF_SPREAD = 1.0e-4  # quantdata/sanos/arbfree.py ArbFreePricer.min_half_spread


@dataclass(frozen=True)
class ReserveSetup:
    state: BcrcState
    schedule: ReserveSchedule
    box: JumpBox
    envelope: GaussianJumpParams


def build_reserve_setup(calib: ChainBatesCalibrationResult, reserve_size: float, *,
                         reset_time: float = cfg.RESERVE_RESET_HORIZON_YEARS,
                         theta_half_width: float = cfg.THETA_BAND_HALF_WIDTH,
                         box_widen: float = cfg.JUMP_BOX_WIDEN_FACTOR) -> ReserveSetup | None:
    """None for reserve_size<=0.0 (the degenerate/ordinary-Bates case: no
    schedule can be constructed since ReserveSchedule requires strictly
    positive reserve sizes)."""
    if reserve_size <= 0.0:
        return None
    theta0 = calib.heston.theta
    jumps = calib.jumps
    box = JumpBox(
        Lambda=max(jumps.lambda_J * box_widen, 0.05),
        M=max(abs(jumps.m_J) * box_widen, 0.01),
        delta_min=jumps.delta_J / box_widen,
        delta_max=jumps.delta_J * box_widen,
        s_star=jumps.delta_J * box_widen * 1.3,
    )
    schedule = ReserveSchedule(reset_times=(reset_time,), reserve_sizes=(reserve_size,),
                                theta_min=theta0 - theta_half_width, theta_max=theta0 + theta_half_width)
    state = initial_state(theta_0=theta0, jumps_0=jumps, schedule=schedule)
    envelope = dominating_envelope_params(box)
    dominance = verify_envelope_dominates(box, jumps)
    assert dominance["dominated_on_grid"], "calibrated jump params unexpectedly outside their own widened box"
    return ReserveSetup(state=state, schedule=schedule, box=box, envelope=envelope)


def price_with_reserve(calib: ChainBatesCalibrationResult, setup: ReserveSetup, strikes: np.ndarray,
                        expiries: np.ndarray, *, n_quad: int = cfg.FOURIER_N_QUAD,
                        u_max: float = cfg.FOURIER_U_MAX, n_reserve_quad: int = 80) -> np.ndarray:
    return price_calls_chain_extended_bates(
        calib.chain_config, calib.heston, calib.jumps, setup.envelope, calib.V_t, strikes, expiries,
        setup.schedule.reset_times, setup.state.b_used, t=0.0, n_quad=n_quad, u_max=u_max,
        n_reserve_quad=n_reserve_quad,
    )


def reserve_price_shift(calib: ChainBatesCalibrationResult, reserve_size: float, strikes: np.ndarray,
                         expiries: np.ndarray) -> dict:
    """Section C: 'for one fixed latent specification, show how much the
    reserve itself changes prices before recalibration.'"""
    baseline = price_calls_chain_bates(calib.chain_config, calib.heston, calib.jumps, calib.V_t, strikes, expiries)
    setup = build_reserve_setup(calib, reserve_size)
    if setup is None:
        return dict(reserve_size=0.0, baseline_price=baseline, with_reserve_price=baseline,
                     shift=np.zeros_like(baseline), setup=None)
    with_reserve = price_with_reserve(calib, setup, strikes, expiries)
    return dict(reserve_size=reserve_size, baseline_price=baseline, with_reserve_price=with_reserve,
                 shift=with_reserve - baseline, setup=setup)


# ---------------------------------------------------------------------------
# Refit under the extended-Bates+reserve regime (recalibration cost recovery).
# ---------------------------------------------------------------------------


class ChainExtendedBatesCalibrationError(RuntimeError):
    pass


def _fit_err(price: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    half_spread = np.maximum(0.5 * (ask - bid), MIN_HALF_SPREAD)
    return (np.maximum(0.0, price - ask) + np.maximum(0.0, bid - price)) / half_spread


@dataclass(frozen=True)
class ChainExtendedBatesCalibrationResult:
    chain_config: ChainGeneratorConfig
    heston: HestonParams
    jumps: GaussianJumpParams
    V_t: float
    reserve_size: float
    reset_time: float
    setup: ReserveSetup
    starts: list = field(default_factory=list)
    chosen_start_index: int = -1
    final_cost: float = float("inf")
    converged: bool = False

    def price(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        return price_calls_chain_extended_bates(
            self.chain_config, self.heston, self.jumps, self.setup.envelope, self.V_t, strikes, expiries,
            self.setup.schedule.reset_times, self.setup.state.b_used, t=0.0,
        )


def calibrate_chain_extended_bates(evaluation_set: EvaluationSet, reserve_size: float, *,
                                    warm_start: ChainBatesCalibrationResult, reset_time: float = cfg.RESERVE_RESET_HORIZON_YEARS,
                                    n_starts: int = cfg.N_MULTI_START, seed: int = 0,
                                    n_quad: int = cfg.FOURIER_N_QUAD_OPTIMIZATION, u_max: float = cfg.FOURIER_U_MAX_OPTIMIZATION,
                                    n_reserve_quad: int = 40,
                                    ) -> ChainExtendedBatesCalibrationResult:
    """Refits (chain_kappa, V_t, heston params, jump params) under the
    extended-Bates+reserve regime at a FIXED reset_time/reserve_size, using
    `warm_start` (the ordinary-Bates fit) as one of the multi-start initial
    guesses (plus jittered variants), to see how much of the fit-quality
    cost of (1) turning on the reserve can be recovered by recalibrating.
    Note: the jump BOX/envelope is rebuilt from each trial's OWN jump
    params inside the objective (box_widen fixed), not fixed at the
    warm-start's box, so the optimizer can move the jump law and have the
    box/envelope follow consistently."""
    contracts = evaluation_set.contracts
    if len(contracts) == 0:
        raise ChainExtendedBatesCalibrationError("cannot calibrate: evaluation set has zero contracts")
    if reserve_size <= 0.0:
        raise ChainExtendedBatesCalibrationError("reserve_size must be > 0 for the extended-Bates regime")

    strikes = np.array([c.strike for c in contracts], dtype=float)
    expiries = np.array([c.expiry_years for c in contracts], dtype=float)
    bids = np.array([c.bid for c in contracts], dtype=float)
    asks = np.array([c.ask for c in contracts], dtype=float)

    n_anchors = len(warm_start.chain_config.anchors)
    warm_half_width = float(np.log(warm_start.chain_config.anchors[-1]))  # inverse of build_anchors' exp(linspace(-w,w,n))
    from quantdata.sdlv.bates_crc.real_data_study.chain_bates_calibration import (
        _bounds_vectors,
        _unpack,
    )
    lo, hi = _bounds_vectors()

    def residual(params: np.ndarray) -> np.ndarray:
        chain_kappa, v_t, heston, jumps, half_width = _unpack(params)
        anchors, initial_index = cfg.build_anchors(n_anchors, half_width)
        chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=chain_kappa,
                                             sigma_y=0.0)
        fake_calib = ChainBatesCalibrationResult(chain_config=chain_config, heston=heston, jumps=jumps, V_t=v_t)
        setup = build_reserve_setup(fake_calib, reserve_size, reset_time=reset_time)
        price = price_with_reserve(fake_calib, setup, strikes, expiries, n_quad=n_quad, u_max=u_max,
                                     n_reserve_quad=n_reserve_quad)
        return _fit_err(price, bids, asks)

    warm_x0 = np.array([warm_start.chain_config.kappa, warm_start.V_t, warm_start.heston.kappa,
                          warm_start.heston.theta, warm_start.heston.xi, warm_start.heston.rho,
                          warm_start.jumps.lambda_J, warm_start.jumps.m_J, warm_start.jumps.delta_J,
                          warm_half_width])
    warm_x0 = np.clip(warm_x0, lo, hi)
    rng = np.random.default_rng(seed)
    guesses = [warm_x0]
    for _ in range(max(0, n_starts - 1)):
        jitter = 1.0 + rng.uniform(-0.2, 0.2, size=warm_x0.shape)
        guesses.append(np.clip(warm_x0 * jitter, lo, hi))

    starts_log = []
    best = None
    for i, x0 in enumerate(guesses):
        try:
            result = least_squares(residual, x0, bounds=(lo, hi), method="trf")
        except Exception as exc:  # noqa: BLE001
            starts_log.append(dict(start_index=i, x0=x0.tolist(), success=False, message=f"exception: {exc}",
                                    cost=float("inf")))
            continue
        cost = float(0.5 * np.sum(result.fun ** 2))
        starts_log.append(dict(start_index=i, x0=x0.tolist(), x_final=result.x.tolist(),
                                success=bool(result.success), status=int(result.status),
                                message=str(result.message), nfev=int(result.nfev), cost=cost))
        if result.success and (best is None or cost < best[0]):
            best = (cost, i, result.x)

    if best is None:
        raise ChainExtendedBatesCalibrationError(
            f"all {n_starts} multi-start refit attempts failed to converge at reserve_size={reserve_size}"
        )

    final_cost, chosen_index, x_final = best
    chain_kappa, v_t, heston, jumps, half_width = _unpack(x_final)
    anchors, initial_index = cfg.build_anchors(n_anchors, half_width)
    chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=initial_index, kappa=chain_kappa, sigma_y=0.0)
    fake_calib = ChainBatesCalibrationResult(chain_config=chain_config, heston=heston, jumps=jumps, V_t=v_t)
    setup = build_reserve_setup(fake_calib, reserve_size, reset_time=reset_time)

    return ChainExtendedBatesCalibrationResult(
        chain_config=chain_config, heston=heston, jumps=jumps, V_t=v_t, reserve_size=reserve_size,
        reset_time=reset_time, setup=setup, starts=starts_log, chosen_start_index=chosen_index,
        final_cost=final_cost, converged=True,
    )


# ---------------------------------------------------------------------------
# Admissible theta movement and post-reset surface dynamics.
# ---------------------------------------------------------------------------


def admissible_theta_interval(calib: ChainBatesCalibrationResult, reserve_size: float, *,
                               reset_time: float = cfg.RESERVE_RESET_HORIZON_YEARS,
                               theta_half_width: float = cfg.THETA_BAND_HALF_WIDTH) -> tuple[float, float]:
    setup = build_reserve_setup(calib, reserve_size, reset_time=reset_time, theta_half_width=theta_half_width)
    if setup is None:
        raise ValueError("reserve_size must be > 0")
    return admissible_delta_interval(calib.heston.theta, reserve_size, setup.schedule)


def reset_surface_change(calib: ChainBatesCalibrationResult, reserve_size: float, delta_fraction: float,
                          strikes: np.ndarray, expiries: np.ndarray, *,
                          reset_time: float = cfg.RESERVE_RESET_HORIZON_YEARS) -> dict:
    """Applies a reset at `reset_time` choosing Delta = delta_fraction * the
    admissible interval (0=min/most-downward, 1=max/most-upward), reprices
    strikes/expiries with T>reset_time before/after, and returns the
    surface change -- Section C's 'changes in future variance and surface
    dynamics following admissible resets.'"""
    setup = build_reserve_setup(calib, reserve_size, reset_time=reset_time)
    if setup is None:
        raise ValueError("reserve_size must be > 0")
    lo, hi = admissible_delta_interval(calib.heston.theta, reserve_size, setup.schedule)
    delta = lo + delta_fraction * (hi - lo)
    proposed_jumps = calib.jumps  # no change in jump law at this reset -- isolates the theta/reserve effect
    new_state, diag = apply_reset(setup.state, proposed_delta_n=delta, proposed_jumps_n=proposed_jumps,
                                   box=setup.box, schedule=setup.schedule)

    mask = expiries > reset_time
    strikes_after = strikes[mask]
    expiries_after = expiries[mask]
    if strikes_after.size == 0:
        raise ValueError("no evaluation contracts with expiry beyond the reset time to reprice")

    before = price_calls_chain_extended_bates(
        calib.chain_config, calib.heston, calib.jumps, setup.envelope, calib.V_t, strikes_after, expiries_after,
        setup.schedule.reset_times, setup.state.b_used, t=0.0,
    )
    after = price_calls_chain_extended_bates(
        calib.chain_config, calib.heston, new_state.jump_params_path[-1], setup.envelope, calib.V_t, strikes_after,
        expiries_after, setup.schedule.reset_times, new_state.b_used, t=0.0,
    )
    return dict(
        delta_fraction=delta_fraction, delta_applied=delta, admissible_interval=(lo, hi),
        theta_before=calib.heston.theta, theta_after=calib.heston.theta + delta, diagnostics=diag,
        strikes=strikes_after, expiries=expiries_after, price_before=before, price_after=after,
        price_change=after - before,
    )
