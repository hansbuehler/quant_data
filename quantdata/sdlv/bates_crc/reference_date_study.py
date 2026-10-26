"""No-lookahead in-repo DLV x Bates fit on the daily GROMM quote dataset.

Historical dynamics are estimated using observations strictly before the
reference date and enter the target-date Q calibration as explicit weak
priors.  The report keeps the backward estimates and fitted Q parameters
separate so the information boundary remains auditable.

This is the repository's finite-state DLV-chain product extension.  It is not
the literal generalized-Bates CRC algorithm of Gambara--Teichmann, whose
recalibration maps maturity-bucketed jump parameters without a DLV chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import ndtr

from quantdata.sdlv.bates_crc.comparison_interface import ChainGeneratorConfig
from quantdata.sdlv.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.sdlv.bates_crc.real_data_study import config as legacy_config
from quantdata.sdlv.bates_crc.real_data_study.chain_pricer import price_calls_chain_bates

FloatArray = NDArray[np.float64]

TARGET_DATE = "2025-08-29"
HISTORICAL_ATM_DTE = 20.0
MIN_HALF_SPREAD = 1.0e-6
DEFAULT_DATASET = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/daily-25-jax/"
    "spx-2022-2025-daily25-shared200-fixed5-market-v1.npz"
)
DEFAULT_RUN_ROOT = Path("/home/hans/quantdata_data/training/dysanos/bates-crc")


@dataclass(frozen=True)
class QuoteBook:
    """One normalized option book; prices retain their original call/put type."""

    date: str
    strikes: FloatArray
    is_call: NDArray[np.bool_]
    bids: FloatArray
    asks: FloatArray
    mids: FloatArray
    business_dtes: FloatArray
    spot_ref: float

    @property
    def expiries(self) -> FloatArray:
        return self.business_dtes / 252.0

    def call_equivalents(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        parity = np.where(self.is_call, 0.0, 1.0 - self.strikes)
        return self.bids + parity, self.asks + parity, self.mids + parity


@dataclass(frozen=True)
class HistoricalDynamics:
    """Backward-only diffusion and jump estimates used as fixed Q inputs."""

    cutoff_date_exclusive: str
    observation_count: int
    variance_proxy_dte: float
    kappa: float
    theta: float
    xi: float
    rho: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    jump_threshold_mad: float
    jump_count: int

    @property
    def heston(self) -> HestonParams:
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )

    @property
    def jumps(self) -> GaussianJumpParams:
        return GaussianJumpParams(
            lambda_J=self.jump_intensity,
            m_J=self.jump_mean,
            delta_J=self.jump_std,
        )


@dataclass(frozen=True)
class ReferenceFit:
    chain_kappa: float
    anchor_half_width: float
    target_variance: float
    heston_kappa: float
    heston_theta: float
    heston_xi: float
    heston_rho: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    quote_cost: float
    penalized_cost: float
    function_evaluations: int

    @property
    def chain_config(self) -> ChainGeneratorConfig:
        anchors, initial_index = legacy_config.build_anchors(
            legacy_config.N_ANCHORS,
            self.anchor_half_width,
        )
        return ChainGeneratorConfig(
            anchors=anchors,
            initial_index=initial_index,
            kappa=self.chain_kappa,
            sigma_y=0.0,
        )

    @property
    def heston(self) -> HestonParams:
        return HestonParams(
            kappa=self.heston_kappa,
            theta=self.heston_theta,
            xi=self.heston_xi,
            rho=self.heston_rho,
        )

    @property
    def jumps(self) -> GaussianJumpParams:
        return GaussianJumpParams(
            lambda_J=self.jump_intensity,
            m_J=self.jump_mean,
            delta_J=self.jump_std,
        )


class ProgressLogger:
    """Append small JSON records while keeping long fits observable."""

    def __init__(self, path: Path, *, minimum_seconds: float = 240.0):
        self.path = path
        self.minimum_seconds = minimum_seconds
        self.started = time.monotonic()
        self.last = -math.inf

    def emit(self, stage: str, *, force: bool = False, **values: Any) -> None:
        elapsed = time.monotonic() - self.started
        if not force and elapsed - self.last < self.minimum_seconds:
            return
        record = {"stage": stage, "elapsed_seconds": elapsed, **values}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.last = elapsed

    def work(
        self,
        stage: str,
        *,
        completed: int,
        total: int,
        force: bool = False,
        **values: Any,
    ) -> None:
        elapsed = time.monotonic() - self.started
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        remaining = (total - completed) / rate if rate > 0.0 else None
        self.emit(
            stage,
            force=force,
            completed_work_units=completed,
            total_work_units=total,
            estimated_remaining_seconds=remaining,
            **values,
        )


def _black_call(strike: FloatArray | float, tau: FloatArray | float, volatility: FloatArray | float):
    total_volatility = np.asarray(volatility) * np.sqrt(tau)
    d1 = -np.log(strike) / total_volatility + 0.5 * total_volatility
    return ndtr(d1) - np.asarray(strike) * ndtr(d1 - total_volatility)


def _implied_volatility(price: float, strike: float, tau: float) -> float:
    intrinsic = max(1.0 - strike, 0.0)
    if not intrinsic + 1.0e-10 < price < 1.0 - 1.0e-10:
        return math.nan
    lower, upper = 1.0e-5, 5.0
    for _ in range(70):
        middle = 0.5 * (lower + upper)
        if _black_call(strike, tau, middle) < price:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load the stable NPZ schema without importing another research project."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    dates = arrays["dates"].astype("U10")
    matches = np.flatnonzero(dates == TARGET_DATE)
    if matches.size != 1:
        raise RuntimeError(f"expected exactly one {TARGET_DATE} row in {path}")
    target_index = int(matches.item())
    if target_index == 0:
        raise RuntimeError("reference date has no backward history")
    if np.any(dates[:target_index] >= TARGET_DATE):
        raise RuntimeError("historical slice is not strictly before the reference date")
    return arrays, target_index


def quote_book(arrays: dict[str, np.ndarray], index: int) -> QuoteBook:
    count = int(arrays["quote_counts"][index])
    selected = slice(0, count)
    return QuoteBook(
        date=str(arrays["dates"][index]),
        strikes=np.asarray(arrays["strikes"][index, selected], dtype=np.float64),
        is_call=np.asarray(arrays["is_call"][index, selected], dtype=np.bool_),
        bids=np.asarray(arrays["bids"][index, selected], dtype=np.float64),
        asks=np.asarray(arrays["asks"][index, selected], dtype=np.float64),
        mids=np.asarray(arrays["mids"][index, selected], dtype=np.float64),
        business_dtes=np.asarray(
            arrays["business_dtes"][index, selected],
            dtype=np.float64,
        ),
        spot_ref=float(arrays["spot_ref"][index]),
    )


def _atm_variance_proxy(book: QuoteBook) -> float:
    _, _, call_mids = book.call_equivalents()
    unique_dtes = np.unique(book.business_dtes)
    selected_dte = float(unique_dtes[np.argmin(np.abs(unique_dtes - HISTORICAL_ATM_DTE))])
    expiry = np.isclose(book.business_dtes, selected_dte, rtol=0.0, atol=1.0e-6)
    candidates = np.flatnonzero(expiry)
    absolute_log_moneyness = np.abs(np.log(book.strikes[candidates]))
    nearest = np.min(absolute_log_moneyness)
    selected = candidates[absolute_log_moneyness <= nearest + 1.0e-7]
    strike = float(np.median(book.strikes[selected]))
    call_mid = float(np.median(call_mids[selected]))
    volatility = _implied_volatility(call_mid, strike, selected_dte / 252.0)
    if not math.isfinite(volatility):
        raise RuntimeError(f"could not invert historical ATM quote on {book.date}")
    return volatility**2


def estimate_historical_dynamics(
    arrays: dict[str, np.ndarray],
    target_index: int,
) -> HistoricalDynamics:
    """Estimate all dynamic parameters from dates before ``target_index``."""

    variance = np.asarray(
        [_atm_variance_proxy(quote_book(arrays, index)) for index in range(target_index)],
        dtype=np.float64,
    )
    if len(variance) < 100 or np.any(~np.isfinite(variance)):
        raise RuntimeError("insufficient finite backward variance history")
    dt = 1.0 / 252.0
    current = variance[:-1]
    change = np.diff(variance)
    design = np.column_stack((np.ones(current.size), current))
    intercept, slope = np.linalg.lstsq(design, change / dt, rcond=None)[0]
    kappa = float(np.clip(-slope, 0.1, 15.0))
    theta = float(np.clip(intercept / kappa, 0.005, 0.50))
    innovation = change - kappa * (theta - current) * dt
    xi = float(
        np.clip(
            np.sqrt(np.mean(innovation**2 / np.maximum(current * dt, 1.0e-12))),
            0.05,
            3.0,
        )
    )

    spots = np.asarray(arrays["spot_ref"][:target_index], dtype=np.float64)
    spot_returns = np.diff(np.log(spots))
    variance_scores = innovation / np.maximum(xi * np.sqrt(current * dt), 1.0e-12)
    spot_scores = (spot_returns - np.mean(spot_returns)) / np.std(spot_returns, ddof=1)
    rho = float(np.clip(np.corrcoef(spot_scores, variance_scores)[0, 1], -0.99, 0.99))

    median = float(np.median(spot_returns))
    robust_scale = float(1.4826 * np.median(np.abs(spot_returns - median)))
    threshold = 3.5 * robust_scale
    jump_values = spot_returns[np.abs(spot_returns - median) > threshold]
    if jump_values.size < 3:
        raise RuntimeError("fewer than three historical jumps under the fixed MAD rule")
    years = spot_returns.size / 252.0
    jump_intensity = float(np.clip(jump_values.size / years, 0.0, 5.0))
    jump_mean = float(np.clip(np.mean(jump_values), -0.5, 0.5))
    jump_std = float(np.clip(np.std(jump_values, ddof=1), 0.01, 1.0))
    return HistoricalDynamics(
        cutoff_date_exclusive=TARGET_DATE,
        observation_count=target_index,
        variance_proxy_dte=HISTORICAL_ATM_DTE,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        jump_threshold_mad=threshold,
        jump_count=int(jump_values.size),
    )


def _spread_residual(model_calls: FloatArray, book: QuoteBook) -> FloatArray:
    bid_calls, ask_calls, _ = book.call_equivalents()
    half_spread = np.maximum(0.5 * (ask_calls - bid_calls), MIN_HALF_SPREAD)
    return (
        np.maximum(model_calls - ask_calls, 0.0)
        + np.maximum(bid_calls - model_calls, 0.0)
    ) / half_spread


def _price_book(
    book: QuoteBook,
    chain: ChainGeneratorConfig,
    heston: HestonParams,
    jumps: GaussianJumpParams,
    variance: float,
    *,
    n_quad: int,
    u_max: float,
) -> FloatArray:
    return price_calls_chain_bates(
        chain,
        heston,
        jumps,
        variance,
        book.strikes,
        book.expiries,
        n_quad=n_quad,
        u_max=u_max,
    )


def calibrate_reference_fit(
    target_book: QuoteBook,
    dynamics: HistoricalDynamics,
    logger: ProgressLogger,
    *,
    n_quad: int = 80,
    u_max: float = 150.0,
) -> ReferenceFit:
    """Fit the target Q cross-section with backward estimates as weak priors."""

    evaluations = 0
    lower = np.asarray(
        (0.05, 0.005, 0.1, 0.005, 0.05, -0.99, 0.0, -0.5, 0.01, 0.03),
        dtype=np.float64,
    )
    upper = np.asarray(
        (300.0, 0.50, 15.0, 0.50, 3.0, 0.99, 5.0, 0.5, 1.0, 0.60),
        dtype=np.float64,
    )
    initial = np.asarray(
        (
            10.0,
            dynamics.theta,
            dynamics.kappa,
            dynamics.theta,
            dynamics.xi,
            dynamics.rho,
            dynamics.jump_intensity,
            dynamics.jump_mean,
            dynamics.jump_std,
            0.15,
        ),
        dtype=np.float64,
    )
    initial = np.clip(initial, lower + 1.0e-10, upper - 1.0e-10)
    prior_center = initial[2:9]
    prior_scale = np.asarray((5.0, 0.03, 0.50, 0.25, 2.0, 0.05, 0.05))
    # Seven weak pseudo-observations carry roughly one percent of the total
    # quote weight. They stabilize the ill-identified Bates parameters without
    # pretending historical P estimates are exact Q parameters.
    prior_multiplier = math.sqrt(0.01 * len(target_book.strikes) / len(prior_center))
    maximum_work_units = 150 * (len(initial) + 1)
    logger.work(
        "target_fit",
        completed=0,
        total=maximum_work_units,
        force=True,
        rms_excess=None,
    )

    def residual(parameters: FloatArray) -> FloatArray:
        nonlocal evaluations
        evaluations += 1
        (
            chain_kappa,
            variance,
            heston_kappa,
            heston_theta,
            heston_xi,
            heston_rho,
            jump_intensity,
            jump_mean,
            jump_std,
            half_width,
        ) = parameters
        anchors, initial_index = legacy_config.build_anchors(
            legacy_config.N_ANCHORS,
            float(half_width),
        )
        chain = ChainGeneratorConfig(
            anchors=anchors,
            initial_index=initial_index,
            kappa=float(chain_kappa),
            sigma_y=0.0,
        )
        heston = HestonParams(
            kappa=float(heston_kappa),
            theta=float(heston_theta),
            xi=float(heston_xi),
            rho=float(heston_rho),
        )
        jumps = GaussianJumpParams(
            lambda_J=float(jump_intensity),
            m_J=float(jump_mean),
            delta_J=float(jump_std),
        )
        prices = _price_book(
            target_book,
            chain,
            heston,
            jumps,
            float(variance),
            n_quad=n_quad,
            u_max=u_max,
        )
        quote_residual = _spread_residual(prices, target_book)
        prior_residual = prior_multiplier * (parameters[2:9] - prior_center) / prior_scale
        logger.work(
            "target_fit",
            completed=evaluations,
            total=maximum_work_units,
            rms_excess=float(np.sqrt(np.mean(quote_residual**2))),
        )
        return np.concatenate((quote_residual, prior_residual))

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=150,
        x_scale="jac",
    )
    if not result.success:
        raise RuntimeError(f"target-date calibration failed: {result.message}")
    (
        chain_kappa,
        target_variance,
        heston_kappa,
        heston_theta,
        heston_xi,
        heston_rho,
        jump_intensity,
        jump_mean,
        jump_std,
        half_width,
    ) = map(float, result.x)
    anchors, initial_index = legacy_config.build_anchors(
        legacy_config.N_ANCHORS,
        half_width,
    )
    chain = ChainGeneratorConfig(
        anchors=anchors,
        initial_index=initial_index,
        kappa=chain_kappa,
        sigma_y=0.0,
    )
    heston = HestonParams(
        kappa=heston_kappa,
        theta=heston_theta,
        xi=heston_xi,
        rho=heston_rho,
    )
    jumps = GaussianJumpParams(
        lambda_J=jump_intensity,
        m_J=jump_mean,
        delta_J=jump_std,
    )
    fitted_calls = _price_book(
        target_book,
        chain,
        heston,
        jumps,
        target_variance,
        n_quad=n_quad,
        u_max=u_max,
    )
    quote_residual = _spread_residual(fitted_calls, target_book)
    quote_cost = float(0.5 * quote_residual @ quote_residual)
    logger.emit(
        "calibration_complete",
        force=True,
        function_evaluations=int(result.nfev),
        quote_cost=quote_cost,
        penalized_cost=float(result.cost),
    )
    return ReferenceFit(
        chain_kappa=chain_kappa,
        anchor_half_width=half_width,
        target_variance=target_variance,
        heston_kappa=heston_kappa,
        heston_theta=heston_theta,
        heston_xi=heston_xi,
        heston_rho=heston_rho,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        quote_cost=quote_cost,
        penalized_cost=float(result.cost),
        function_evaluations=int(result.nfev),
    )


def fit_metrics(model_calls: FloatArray, book: QuoteBook) -> dict[str, float | int]:
    residual = _spread_residual(model_calls, book)
    positive = residual[residual > 0.0]
    model_prices = np.where(book.is_call, model_calls, model_calls - 1.0 + book.strikes)
    midpoint = 0.5 * (book.bids + book.asks)
    half_spread = np.maximum(0.5 * (book.asks - book.bids), MIN_HALF_SPREAD)
    return {
        "quote_count": int(residual.size),
        "percent_inside_bid_ask": float(100.0 * np.mean(residual == 0.0)),
        "percent_with_excess": float(100.0 * np.mean(residual > 0.0)),
        "mean_excess_half_spreads": float(np.mean(residual)),
        "conditional_mean_excess_half_spreads": float(np.mean(positive)),
        "conditional_median_excess_half_spreads": float(np.median(positive)),
        "midpoint_mae_half_spreads": float(
            np.mean(np.abs(model_prices - midpoint) / half_spread)
        ),
    }


def model_provenance() -> dict[str, Any]:
    """Return deterministic labels that prevent a paper-implementation claim."""

    return {
        "model_family": (
            "DLV x ordinary-Bates background calibration for the in-repo reserve CRC"
        ),
        "implementation_scope": "static initial fit of the no-reset background",
        "literal_gambara_teichmann_algorithm_1": False,
        "gambara_teichmann_distinction": (
            "Section 7.3 / Algorithm 1 uses generalized Bates directly with "
            "maturity-bucketed jump parameters and an inverse-map regrid; this "
            "repository extension additionally uses a finite-state DLV chain."
        ),
    }


def crc_dynamic_audit_blocker() -> dict[str, Any]:
    """Explain why the current reserve implementation cannot generate an audit."""

    return {
        "status": "blocked-not-run",
        "reason": (
            "The implemented reserve CRC defines admissible deterministic reset "
            "updates but no stochastic Q transition kernel for reset proposals."
        ),
        "missing_requirements": [
            "Q probabilities or intensities for theta and jump-parameter resets",
            "an equivalence/support rule for those reset probabilities",
            "conditional option values integrating over future reset outcomes",
            "a path generator using exactly that same conditional pricing law",
        ],
        "invalid_shortcut": (
            "A fixed-parameter DLV x Bates simulation is the no-reset special case, "
            "not a test of the reserve-CRC model, and is therefore not reported as CRC."
        ),
        "paper_implementation_distinction": (
            "Gambara--Teichmann Section 7.3 / Algorithm 1 uses generalized "
            "Bates directly, including maturity-bucketed jump parameters and "
            "an inverse-map regrid. The repository model adds an independent "
            "finite-state DLV chain and does not implement that literal algorithm."
        ),
        "requested_geometry": {
            "path_count": 100_000,
            "antithetic_pair_count": 50_000,
            "horizon_business_days": 20,
            "restricted_option_count": 44,
            "conditional_diagnostics": ["squared", "entropy"],
        },
    }


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(dataset: Path, run_directory: Path) -> dict[str, Any]:
    run_directory.mkdir(parents=True, exist_ok=False)
    logger = ProgressLogger(run_directory / "progress.jsonl")
    logger.emit("load", force=True, dataset=str(dataset))
    arrays, target_index = load_dataset(dataset)
    target_book = quote_book(arrays, target_index)
    dynamics = estimate_historical_dynamics(arrays, target_index)
    logger.emit("historical_estimate", force=True, **asdict(dynamics))
    fit = calibrate_reference_fit(target_book, dynamics, logger)
    final_calls = _price_book(
        target_book,
        fit.chain_config,
        fit.heston,
        fit.jumps,
        fit.target_variance,
        n_quad=200,
        u_max=250.0,
    )
    metrics = fit_metrics(final_calls, target_book)
    payload: dict[str, Any] = {
        "schema": "dlv-bates-reference-date-fit-v1",
        "status": "static-fit-complete-crc-dynamic-audit-blocked",
        **model_provenance(),
        "target_date": TARGET_DATE,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "historical_dynamics": asdict(dynamics),
        "reference_fit": asdict(fit),
        "chain_anchors": list(fit.chain_config.anchors),
        "initial_fit_metrics": metrics,
        "crc_dynamic_audit": crc_dynamic_audit_blocker(),
    }
    _atomic_json(payload, run_directory / "fit_report.json")
    np.savez_compressed(
        run_directory / "initial_fit_prices.npz",
        strikes=target_book.strikes,
        is_call=target_book.is_call,
        bids=target_book.bids,
        asks=target_book.asks,
        mids=target_book.mids,
        business_dtes=target_book.business_dtes,
        model_calls=final_calls,
        model_prices=np.where(
            target_book.is_call,
            final_calls,
            final_calls - 1.0 + target_book.strikes,
        ),
    )
    logger.emit("complete", force=True, **metrics)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-directory", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run(arguments.dataset, arguments.run_directory)
    print(json.dumps(payload["initial_fit_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
