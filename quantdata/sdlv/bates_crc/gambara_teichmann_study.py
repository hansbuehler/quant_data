"""No-lookahead 2025-08-29 study for generalized-Bates CRC Algorithm 1.

The target cross-section is calibrated to every quote in the same 2,662-quote
book used by the GROMM comparison.  Heston ``kappa`` and jump intensity are
fixed from backward-only data; the target variance, ``theta``, ``xi``, ``rho``
and five Gaussian jump mean/std pairs are fitted under Q.

This module prepares, but does not misrepresent, the dynamic audit.  A full
100,000-path CRC run additionally requires a validated learned inverse map for
Algorithm-1 step 6; direct per-state least squares is intentionally not used as
a computational shortcut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantdata.sdlv.bates_crc.gambara_teichmann_crc import (
    FiveBucketJumps,
    GeneralizedBatesState,
    paired_maturity_boundaries,
    price_calls_generalized_bates,
)
from quantdata.sdlv.bates_crc.heston_riccati import HestonParams
from quantdata.sdlv.bates_crc.reference_date_study import (
    DEFAULT_DATASET,
    TARGET_DATE,
    HistoricalDynamics,
    ProgressLogger,
    QuoteBook,
    _atm_variance_proxy,
    _sha256,
    _spread_residual,
    estimate_historical_dynamics,
    fit_metrics,
    load_dataset,
    quote_book,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class GeneralizedBatesFit:
    """Target-date generalized-Bates parameters, with fixed quantities explicit."""

    variance: float
    kappa_fixed_from_history: float
    theta: float
    xi: float
    rho: float
    intensity_fixed_from_history: float
    maturity_nodes: tuple[float, ...]
    boundaries: tuple[float, float, float, float]
    jump_means: tuple[float, float, float, float, float]
    jump_stds: tuple[float, float, float, float, float]
    quote_cost: float
    penalized_cost: float
    function_evaluations: int
    optimizer_converged: bool
    optimizer_message: str

    @property
    def state(self) -> GeneralizedBatesState:
        return GeneralizedBatesState(
            log_spot=0.0,
            variance=self.variance,
            heston=HestonParams(
                kappa=self.kappa_fixed_from_history,
                theta=self.theta,
                xi=self.xi,
                rho=self.rho,
            ),
            jumps=FiveBucketJumps(
                intensity=self.intensity_fixed_from_history,
                boundaries=self.boundaries,
                means=self.jump_means,
                stds=self.jump_stds,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ResetInnovationLaw:
    """Backward-only Gaussian law for relative changes of theta, xi, and rho.

    The innovations are ``(d log(theta), d log(xi), d atanh(rho))``.  The
    paper's five-percent cap is applied when the law is consumed by an online
    simulator, not while estimating its covariance.
    """

    cutoff_date_exclusive: str
    observation_count: int
    mean: tuple[float, float, float]
    covariance: tuple[tuple[float, float, float], ...]
    relative_change_cap: float = 0.05


def select_ten_maturity_nodes(expiries: FloatArray) -> FloatArray:
    """Select ten deterministic nodes spanning the observed target maturities."""

    unique = np.unique(np.asarray(expiries, dtype=np.float64))
    if unique.size < 10 or unique[0] <= 0.0:
        raise ValueError("at least ten positive target maturities are required")
    indices = np.rint(np.linspace(0, unique.size - 1, 10)).astype(int)
    nodes = unique[indices]
    if np.unique(nodes).size != 10:
        raise RuntimeError("maturity-node selection did not produce ten unique nodes")
    return nodes


def estimate_reset_innovation_law(
    dates: NDArray[np.str_],
    theta_xi_rho: FloatArray,
    *,
    target_date: str,
) -> ResetInnovationLaw:
    """Estimate the Algorithm-1 exogenous reset law without target-date leakage."""

    dates = np.asarray(dates).astype("U10")
    parameters = np.asarray(theta_xi_rho, dtype=np.float64)
    if dates.ndim != 1 or parameters.shape != (dates.size, 3):
        raise ValueError("theta_xi_rho must have shape (len(dates), 3)")
    if dates.size < 3:
        raise ValueError("at least three historical parameter observations are required")
    if np.any(dates >= target_date):
        raise ValueError("reset-law observations must be strictly before target_date")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("reset-law dates must be strictly increasing")
    theta, xi, rho = parameters.T
    if np.any(theta <= 0.0) or np.any(xi <= 0.0) or np.any(np.abs(rho) >= 1.0):
        raise ValueError("theta/xi must be positive and rho strictly inside (-1,1)")
    transformed = np.column_stack((np.log(theta), np.log(xi), np.arctanh(rho)))
    innovations = np.diff(transformed, axis=0)
    covariance = np.cov(innovations, rowvar=False, ddof=1)
    if covariance.shape != (3, 3) or np.any(~np.isfinite(covariance)):
        raise RuntimeError("reset-law covariance is not finite")
    return ResetInnovationLaw(
        cutoff_date_exclusive=target_date,
        observation_count=int(dates.size),
        mean=tuple(float(value) for value in np.mean(innovations, axis=0)),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )


def estimate_daily_reset_parameter_sequence(
    arrays: dict[str, np.ndarray],
    target_index: int,
    *,
    fixed_kappa: float,
    lookback_sessions: int = 63,
    minimum_history_sessions: int = 126,
) -> tuple[NDArray[np.str_], FloatArray]:
    """Construct an actual backward-only daily theta/xi/rho proxy sequence.

    For each pre-target date, ``theta`` is the trailing median 20-business-day
    ATM implied variance. ``xi`` is the trailing RMS of variance increments
    normalized by ``sqrt(V dt)``. ``rho`` is the trailing correlation between
    standardized spot returns and those variance innovations.  Every rolling
    window ends on the date being estimated, so later pre-target observations
    cannot leak backward and the target date is never read.

    These are empirical exogenous-dynamics proxies, not daily full-surface
    generalized-Bates calibrations.  That distinction is persisted in output.
    """

    if not math.isfinite(fixed_kappa) or fixed_kappa <= 0.0:
        raise ValueError("fixed_kappa must be finite and positive")
    if minimum_history_sessions < lookback_sessions + 2:
        raise ValueError("minimum_history_sessions must exceed lookback_sessions by at least two")
    if target_index <= minimum_history_sessions:
        raise ValueError("insufficient strict-pre-target history for daily reset parameters")
    dates = np.asarray(arrays["dates"][:target_index]).astype("U10")
    if np.any(dates >= TARGET_DATE) or np.any(dates[1:] <= dates[:-1]):
        raise RuntimeError("daily reset sequence must use increasing dates strictly before target")
    variance = np.asarray(
        [_atm_variance_proxy(quote_book(arrays, index)) for index in range(target_index)],
        dtype=np.float64,
    )
    spots = np.asarray(arrays["spot_ref"][:target_index], dtype=np.float64)
    returns = np.diff(np.log(spots))
    dt = 1.0 / 252.0
    parameter_rows: list[tuple[float, float, float]] = []
    selected_dates: list[str] = []
    for index in range(minimum_history_sessions, target_index):
        start = index - lookback_sessions
        variance_window = variance[start : index + 1]
        increments = np.diff(variance_window)
        current = variance_window[:-1]
        theta = float(np.clip(np.median(variance_window), 0.01, 0.50))
        innovations = increments - fixed_kappa * (theta - current) * dt
        xi = float(
            np.clip(
                np.sqrt(np.mean(innovations**2 / np.maximum(current * dt, 1.0e-12))),
                0.01,
                0.50,
            )
        )
        variance_scores = innovations / np.maximum(xi * np.sqrt(current * dt), 1.0e-12)
        return_window = returns[start:index]
        return_std = float(np.std(return_window, ddof=1))
        if return_std <= 0.0 or not math.isfinite(return_std):
            raise RuntimeError(f"invalid trailing return scale on {dates[index]}")
        return_scores = (return_window - np.mean(return_window)) / return_std
        rho = float(np.clip(np.corrcoef(return_scores, variance_scores)[0, 1], -0.99, 0.99))
        # Preserve the paper's post-update Feller/domain restriction.
        xi = min(xi, math.sqrt(max(2.0 * fixed_kappa * theta, 1.0e-12)) * (1.0 - 1.0e-10))
        parameter_rows.append((theta, xi, rho))
        selected_dates.append(str(dates[index]))
    parameters = np.asarray(parameter_rows, dtype=np.float64)
    if parameters.shape != (len(selected_dates), 3) or np.any(~np.isfinite(parameters)):
        raise RuntimeError("daily reset parameter sequence is incomplete or non-finite")
    return np.asarray(selected_dates, dtype="U10"), parameters


def apply_reset_innovation(
    heston: HestonParams,
    innovation: FloatArray,
    law: ResetInnovationLaw,
) -> HestonParams:
    """Apply a capped reset innovation while keeping kappa and Feller validity."""

    innovation = np.asarray(innovation, dtype=np.float64)
    if innovation.shape != (3,) or np.any(~np.isfinite(innovation)):
        raise ValueError("innovation must contain three finite values")
    cap = law.relative_change_cap
    theta = heston.theta * math.exp(float(np.clip(innovation[0], -cap, cap)))
    xi = heston.xi * math.exp(float(np.clip(innovation[1], -cap, cap)))
    rho = math.tanh(float(np.arctanh(np.clip(heston.rho, -0.999999, 0.999999)) + innovation[2]))
    theta = float(np.clip(theta, 0.01, 0.50))
    xi = float(np.clip(xi, 0.01, 0.50))
    # The paper enforces the Feller condition after each exogenous update.
    xi = min(xi, math.sqrt(max(2.0 * heston.kappa * theta, 1.0e-12)) * (1.0 - 1.0e-10))
    return HestonParams(kappa=heston.kappa, theta=theta, xi=xi, rho=rho)


def calibrate_target_book(
    book: QuoteBook,
    history: HistoricalDynamics,
    logger: ProgressLogger,
    *,
    n_quad: int = 240,
    u_max: float = 600.0,
    max_nfev: int = 1_000,
) -> GeneralizedBatesFit:
    """Fit the literal generalized-Bates background with kappa/lambda fixed."""

    nodes = select_ten_maturity_nodes(book.expiries)
    boundaries = paired_maturity_boundaries(nodes)
    lower = np.asarray((0.0025, 0.01, 0.05, -0.99, *([-0.50] * 5), *([0.005] * 5)))
    upper = np.asarray((0.50, 0.50, 0.50, 0.99, *([0.50] * 5), *([0.75] * 5)))
    initial = np.asarray(
        (
            history.theta,
            history.theta,
            min(history.xi, 0.50),
            history.rho,
            *([history.jump_mean] * 5),
            *([history.jump_std] * 5),
        ),
        dtype=np.float64,
    )
    initial = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
    evaluations = 0
    maximum_work_units = max_nfev * (initial.size + 1)
    logger.work("generalized_bates_target_fit", completed=0, total=maximum_work_units, force=True)

    # Historical P quantities are weak stabilizing priors, not hard Q values.
    prior_center = initial[:4]
    prior_scale = np.asarray((0.05, 0.05, 0.20, 0.30))
    prior_weight = math.sqrt(0.005 * len(book.strikes) / prior_center.size)

    def state_from_parameters(parameters: FloatArray) -> GeneralizedBatesState:
        return GeneralizedBatesState(
            log_spot=0.0,
            variance=float(parameters[0]),
            heston=HestonParams(
                kappa=history.kappa,
                theta=float(parameters[1]),
                xi=float(parameters[2]),
                rho=float(parameters[3]),
            ),
            jumps=FiveBucketJumps(
                intensity=history.jump_intensity,
                boundaries=boundaries,
                means=tuple(float(value) for value in parameters[4:9]),
                stds=tuple(float(value) for value in parameters[9:14]),
            ),
        )

    def residual(parameters: FloatArray) -> FloatArray:
        nonlocal evaluations
        evaluations += 1
        prices = price_calls_generalized_bates(
            state_from_parameters(parameters),
            book.strikes,
            book.expiries,
            n_quad=n_quad,
            u_max=u_max,
        )
        quote_residual = _spread_residual(prices, book)
        logger.work(
            "generalized_bates_target_fit",
            completed=evaluations,
            total=maximum_work_units,
            rms_excess=float(np.sqrt(np.mean(quote_residual**2))),
        )
        return np.concatenate((quote_residual, prior_weight * (parameters[:4] - prior_center) / prior_scale))

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
    )
    if np.any(~np.isfinite(result.x)) or not math.isfinite(float(result.cost)):
        raise RuntimeError(f"target generalized-Bates calibration failed: {result.message}")
    final_state = state_from_parameters(result.x)
    final_prices = price_calls_generalized_bates(
        final_state, book.strikes, book.expiries, n_quad=n_quad, u_max=u_max
    )
    quote_residual = _spread_residual(final_prices, book)
    logger.emit(
        "generalized_bates_calibration_complete",
        force=True,
        function_evaluations=int(result.nfev),
        quote_cost=float(0.5 * quote_residual @ quote_residual),
    )
    return GeneralizedBatesFit(
        variance=final_state.variance,
        kappa_fixed_from_history=history.kappa,
        theta=final_state.heston.theta,
        xi=final_state.heston.xi,
        rho=final_state.heston.rho,
        intensity_fixed_from_history=history.jump_intensity,
        maturity_nodes=tuple(float(value) for value in nodes),
        boundaries=boundaries,
        jump_means=final_state.jumps.means,
        jump_stds=final_state.jumps.stds,
        quote_cost=float(0.5 * quote_residual @ quote_residual),
        penalized_cost=float(result.cost),
        function_evaluations=int(result.nfev),
        optimizer_converged=bool(result.success),
        optimizer_message=str(result.message),
    )


def dynamic_audit_readiness() -> dict[str, object]:
    """State the exact boundary between implemented CRC and pending compute."""

    return {
        "model": "Gambara-Teichmann generalized Bates Algorithm 1",
        "static_target_fit": "implemented",
        "small_direct_inverse_smoke": "implemented-with-hard-tolerance-gate",
        "full_q_audit": "not-run",
        "required_before_full_q_audit": (
            "train and validate the paper's learned inverse map on synthetic generalized-Bates "
            "surfaces, including a held-out fixed-contract price-preservation gate"
        ),
        "requested_geometry": {
            "paths": 100_000,
            "antithetic_pairs": 50_000,
            "business_days": 20,
            "restricted_options": 44,
            "conditional_tests": ("squared", "entropy"),
        },
    }


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_target_study(dataset: Path, run_directory: Path) -> dict[str, object]:
    """Run and persist the static target fit; never claim a pending Q audit."""

    run_directory.mkdir(parents=True, exist_ok=False)
    logger = ProgressLogger(run_directory / "progress.jsonl")
    logger.emit("load", force=True, dataset=str(dataset), completed_work_units=0, total_work_units=1)
    arrays, target_index = load_dataset(dataset)
    book = quote_book(arrays, target_index)
    if len(book.strikes) != 2_662:
        raise RuntimeError(f"expected the comparison's 2,662 quotes, got {len(book.strikes)}")
    history = estimate_historical_dynamics(arrays, target_index)
    reset_dates, reset_parameters = estimate_daily_reset_parameter_sequence(
        arrays,
        target_index,
        fixed_kappa=history.kappa,
    )
    reset_law = estimate_reset_innovation_law(
        reset_dates,
        reset_parameters,
        target_date=TARGET_DATE,
    )
    logger.emit(
        "backward_history_complete",
        force=True,
        completed_work_units=1,
        total_work_units=1,
        estimated_remaining_seconds=0.0,
        cutoff_date_exclusive=history.cutoff_date_exclusive,
        observation_count=history.observation_count,
    )
    fit = calibrate_target_book(book, history, logger)
    prices = price_calls_generalized_bates(
        fit.state,
        book.strikes,
        book.expiries,
        n_quad=240,
        u_max=600.0,
    )
    metrics = fit_metrics(prices, book)
    payload: dict[str, object] = {
        "schema": "gambara-teichmann-generalized-bates-target-fit-v1",
        "status": "static-fit-complete-dynamic-crc-audit-not-run",
        "model_family": "Gambara-Teichmann generalized Bates Algorithm 1 background",
        "finite_dlv_chain_used": False,
        "target_date": TARGET_DATE,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "quote_count": len(book.strikes),
        "history_information_boundary": f"dates strictly before {TARGET_DATE}",
        "historical_dynamics": asdict(history),
        "reset_parameter_sequence": {
            "estimator": (
                "daily trailing-63-session proxies: theta=median 20bd ATM IV squared; "
                "xi=CIR-normalized variance-increment RMS; rho=spot/variance innovation correlation"
            ),
            "full_surface_daily_bates_calibration": False,
            "first_date": str(reset_dates[0]),
            "last_date": str(reset_dates[-1]),
            "observation_count": int(len(reset_dates)),
            "information_rule": "each row uses only observations on or before its own date",
        },
        "reset_innovation_law": asdict(reset_law),
        "target_fit": asdict(fit),
        "pricing_quadrature": {"nodes": 240, "u_max": 600.0},
        "initial_fit_metrics": metrics,
        "dynamic_audit": dynamic_audit_readiness(),
    }
    _atomic_json(payload, run_directory / "fit_report.json")
    np.savez_compressed(
        run_directory / "initial_fit_prices.npz",
        strikes=book.strikes,
        is_call=book.is_call,
        bids=book.bids,
        asks=book.asks,
        mids=book.mids,
        business_dtes=book.business_dtes,
        model_calls=prices,
        model_prices=np.where(book.is_call, prices, prices - 1.0 + book.strikes),
    )
    np.savez_compressed(
        run_directory / "backward_reset_parameter_sequence.npz",
        dates=reset_dates,
        theta=reset_parameters[:, 0],
        xi=reset_parameters[:, 1],
        rho=reset_parameters[:, 2],
    )
    logger.emit("complete", force=True, completed_work_units=1, total_work_units=1, estimated_remaining_seconds=0.0, **metrics)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-directory", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_target_study(arguments.dataset, arguments.run_directory)
    print(json.dumps(payload["initial_fit_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
