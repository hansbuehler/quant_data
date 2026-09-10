"""Compare four-step local-SDE and finite-h LVsq simulations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lvsq_drift import (
    fixed_contract_lvsq_drift,
    fixed_contract_lvsq_euler_step,
    fixed_horizon_lvsq_exponential_step,
    regrid_raw_dlvsq_to_relative_spot,
    smooth_lvsq_log_loadings,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.torch import BaseEnvironment

Method = Literal["local_sde", "finite_h_exact"]


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FourStepLvsqComparisonConfig:
    """Logical simulation and processing specification."""

    num_paths: int = 10_000
    num_steps: int = 4
    calendar_business_days_per_step: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    exact_state_batch_size: int = 256
    sde_state_batch_size: int = 4_096
    surface_batch_size: int = 2_048
    moment_tolerance: float = 1.0e-14
    covariance_eigenvalue_fraction: float = 1.0e-8
    seed: int = 2_026_090_8

    def __post_init__(self) -> None:
        verify_inp(self.num_paths > 1, lambda: "num_paths must exceed one")
        verify_inp(self.num_steps > 0, lambda: "num_steps must be positive")
        verify_inp(
            self.calendar_business_days_per_step > 0,
            lambda: "calendar_business_days_per_step must be positive",
        )
        verify_inp(
            np.isfinite(self.dlvsq_annualized_log_volatility)
            and self.dlvsq_annualized_log_volatility > 0.0,
            lambda: "DLVsq log volatility must be finite and positive",
        )
        verify_inp(
            np.isfinite(self.bs_backbone_annualized_volatility)
            and self.bs_backbone_annualized_volatility > 0.0,
            lambda: "BS backbone volatility must be finite and positive",
        )
        verify_inp(
            self.num_option_strikes >= 3,
            lambda: "num_option_strikes must be at least three",
        )
        verify_inp(
            min(
                self.exact_state_batch_size,
                self.sde_state_batch_size,
                self.surface_batch_size,
            )
            > 0,
            lambda: "all batch sizes must be positive",
        )
        verify_inp(
            np.isfinite(self.moment_tolerance) and self.moment_tolerance > 0.0,
            lambda: "moment_tolerance must be finite and positive",
        )
        verify_inp(
            0.0 < self.covariance_eigenvalue_fraction < 1.0,
            lambda: "covariance_eigenvalue_fraction must lie in (0,1)",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-paths", type=int, default=10_000)
    parser.add_argument("--exact-state-batch-size", type=int, default=256)
    parser.add_argument("--sde-state-batch-size", type=int, default=4_096)
    parser.add_argument("--surface-batch-size", type=int, default=2_048)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_surface(decoder: Any, raw_states: torch.Tensor) -> Any:
    parameters = bound_native_raw(raw_states, geometry=decoder.geometry)
    return decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )


def _set_constant_bs_backbone(
    raw_states: torch.Tensor,
    *,
    decoder: Any,
    annualized_volatility: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set FLEX forward parameters to one constant-volatility BS backbone."""

    geometry = decoder.geometry
    model_expiries = decoder.model_expiries_y
    target_total_vols = annualized_volatility * torch.sqrt(model_expiries)
    target_increments = torch.diff(
        target_total_vols,
        prepend=torch.zeros(1, dtype=raw_states.dtype, device=raw_states.device),
    )
    forward_parameters = target_increments / torch.sqrt(decoder.model_dts_y)
    if not bool(
        torch.all(forward_parameters > geometry.min_forward_vol)
        and torch.all(forward_parameters < geometry.max_forward_vol)
    ):
        raise ValueError("constant BS backbone lies outside the FLEX forward bounds")
    probability = (
        (forward_parameters - geometry.min_forward_vol)
        / (geometry.max_forward_vol - geometry.min_forward_vol)
    )
    raw_forward = torch.logit(probability)
    blocks = raw_states.reshape(
        len(raw_states),
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    ).clone()
    blocks[..., geometry.num_inner_strikes] = raw_forward
    return blocks.reshape_as(raw_states), forward_parameters


def _surface_option_values(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    cumulative_spot: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    batch_size: int,
    fixed_cash: bool,
) -> torch.Tensor:
    """Price the fixed option set in either surface or current-spot units."""

    result = torch.empty(
        (
            len(raw_states),
            decoder.geometry.num_model_expiries,
            len(fixed_cash_strikes),
        ),
        dtype=torch.float64,
        device="cpu",
    )
    is_call = fixed_cash_strikes <= 1.0
    for start in range(0, len(raw_states), batch_size):
        stop = min(start + batch_size, len(raw_states))
        surface = _decode_surface(decoder, raw_states[start:stop])
        spot = cumulative_spot[start:stop]
        normalized_strikes = (
            fixed_cash_strikes[None] / spot[:, None]
            if fixed_cash
            else fixed_cash_strikes[None].expand(len(spot), -1)
        )
        component = surface.support_strikes[..., None]
        strikes = normalized_strikes[:, None, None]
        total_volatility = surface.backbone_total_vols[:, :, None, None]
        d1 = (
            torch.log(component / strikes) + 0.5 * total_volatility.square()
        ) / total_volatility
        d2 = d1 - total_volatility
        component_calls = (
            component * torch.special.ndtr(d1)
            - strikes * torch.special.ndtr(d2)
        )
        calls = torch.sum(surface.densities[..., None] * component_calls, dim=2)
        puts = calls - 1.0 + normalized_strikes[:, None]
        values = torch.where(is_call[None, None], calls, puts)
        if fixed_cash:
            values = spot[:, None, None] * values
        result[start:stop] = values.detach().cpu().double()
    return result


def _sample_front_spot_returns(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    gaussian_ranks: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    result = torch.empty_like(gaussian_ranks)
    for start in range(0, len(raw_states), batch_size):
        stop = min(start + batch_size, len(raw_states))
        surface = _decode_surface(decoder, raw_states[start:stop])
        result[start:stop] = front_week_sanos_inverse_cdf(
            surface,
            gaussian_ranks[start:stop],
        )
    return result


def _wald_statistic(
    observations: np.ndarray,
    *,
    eigenvalue_fraction: float,
) -> dict[str, float | int]:
    """Return a covariance-aware zero-mean test with small modes removed."""

    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2 or len(observations) < 2:
        raise ValueError("Wald observations must be a non-trivial matrix")
    covariance = np.cov(observations, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(np.max(eigenvalues))
    cutoff = maximum * eigenvalue_fraction
    retained = eigenvalues > cutoff
    mean = np.mean(observations, axis=0)
    projected_mean = eigenvectors[:, retained].T @ mean
    statistic = float(
        len(observations)
        * np.sum(projected_mean**2 / eigenvalues[retained])
    )
    degrees = int(np.sum(retained))
    return {
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": float(chi2.sf(statistic, degrees)),
        "eigenvalue_cutoff": cutoff,
        "maximum_eigenvalue": maximum,
        "minimum_retained_eigenvalue": float(np.min(eigenvalues[retained])),
        "discarded_eigenvalues": int(len(eigenvalues) - degrees),
    }


def _risk_neutrality_audit(
    changes: torch.Tensor,
    *,
    state_instruments: np.ndarray,
    eigenvalue_fraction: float,
) -> dict[str, object]:
    values = changes.reshape(len(changes), -1).numpy().astype(np.float64, copy=False)
    standard_deviation = np.std(values, axis=0, ddof=1)
    active = standard_deviation > max(1.0e-14, 1.0e-10 * float(np.max(standard_deviation)))
    active_values = values[:, active]
    active_standard_deviation = standard_deviation[active]
    mean = np.mean(active_values, axis=0)
    standard_error = active_standard_deviation / np.sqrt(len(values))
    z = mean / standard_error
    standardized_values = active_values / active_standard_deviation
    instrument_moments = (
        state_instruments[:, :, None] * standardized_values[:, None, :]
    ).reshape(len(values), -1)
    return {
        "num_paths": len(values),
        "num_coordinates": values.shape[1],
        "num_active_coordinates": int(np.sum(active)),
        "maximum_absolute_mean_bps_spot": float(10_000.0 * np.max(np.abs(mean))),
        "root_mean_square_mean_bps_spot": float(
            10_000.0 * np.sqrt(np.mean(mean**2))
        ),
        "maximum_absolute_coordinate_z": float(np.max(np.abs(z))),
        "root_mean_square_coordinate_z": float(np.sqrt(np.mean(z**2))),
        "num_absolute_coordinate_z_above_2": int(np.sum(np.abs(z) > 2.0)),
        "unconditional_omnibus": _wald_statistic(
            standardized_values,
            eigenvalue_fraction=eigenvalue_fraction,
        ),
        "state_instrument_omnibus": _wald_statistic(
            instrument_moments,
            eigenvalue_fraction=eigenvalue_fraction,
        ),
    }


class _CovarianceAccumulator:
    """Streaming covariance checks in the four-dimensional input factor space."""

    def __init__(self, loadings: torch.Tensor, *, step_y: float) -> None:
        self.loadings = loadings.reshape(-1, loadings.shape[-1])
        self.pseudoinverse_transpose = torch.linalg.pinv(self.loadings).T
        self.step_y = step_y
        self.count = 0
        factors = loadings.shape[-1]
        coordinates = self.loadings.shape[0]
        options = {"dtype": loadings.dtype, "device": loadings.device}
        self.pre_sum = torch.zeros(factors, **options)
        self.pre_outer = torch.zeros((factors, factors), **options)
        self.post_sum = torch.zeros(factors, **options)
        self.post_outer = torch.zeros((factors, factors), **options)
        self.coordinate_sum = torch.zeros(coordinates, **options)
        self.coordinate_square_sum = torch.zeros(coordinates, **options)
        self.post_outside_square = torch.zeros((), **options)
        self.post_total_square = torch.zeros((), **options)

    def update(
        self,
        *,
        raw_before: torch.Tensor,
        raw_before_regrid: torch.Tensor,
        raw_after_regrid: torch.Tensor,
        raw_drift: torch.Tensor,
        num_expiries: int,
        num_inner: int,
        block_width: int,
    ) -> None:
        before = raw_before.reshape(-1, num_expiries, block_width)[..., :num_inner]
        pre = raw_before_regrid.reshape(-1, num_expiries, block_width)[..., :num_inner]
        post = raw_after_regrid.reshape(-1, num_expiries, block_width)[..., :num_inner]
        drift = raw_drift.reshape(-1, num_expiries, block_width)[..., :num_inner]
        pre_residual = (pre - before - self.step_y * drift).reshape(len(before), -1)
        post_residual = (post - before - self.step_y * drift).reshape(len(before), -1)
        pre_factor = (
            pre_residual @ self.pseudoinverse_transpose / self.step_y**0.5
        )
        post_factor = (
            post_residual @ self.pseudoinverse_transpose / self.step_y**0.5
        )
        projected_post = (
            self.step_y**0.5 * post_factor @ self.loadings.T
        )
        outside = post_residual - projected_post
        self.count += len(before)
        self.pre_sum += torch.sum(pre_factor, dim=0)
        self.pre_outer += pre_factor.T @ pre_factor
        self.post_sum += torch.sum(post_factor, dim=0)
        self.post_outer += post_factor.T @ post_factor
        self.coordinate_sum += torch.sum(pre_residual, dim=0)
        self.coordinate_square_sum += torch.sum(pre_residual.square(), dim=0)
        self.post_outside_square += torch.sum(outside.square())
        self.post_total_square += torch.sum(post_residual.square())

    @staticmethod
    def _sample_covariance(
        total: torch.Tensor,
        outer: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        return (outer - torch.outer(total, total) / count) / (count - 1)

    def report(self) -> dict[str, object]:
        if self.count < 2:
            raise RuntimeError("covariance audit has too few observations")
        pre_covariance = self._sample_covariance(
            self.pre_sum, self.pre_outer, self.count
        )
        post_covariance = self._sample_covariance(
            self.post_sum, self.post_outer, self.count
        )
        identity = torch.eye(
            len(self.pre_sum),
            dtype=self.pre_sum.dtype,
            device=self.pre_sum.device,
        )
        coordinate_variance = (
            self.coordinate_square_sum
            - self.coordinate_sum.square() / self.count
        ) / (self.count - 1)
        input_coordinate_variance = self.step_y * torch.sum(
            self.loadings.square(), dim=1
        )
        active = input_coordinate_variance > 0.0
        relative_coordinate_error = (
            coordinate_variance[active] / input_coordinate_variance[active] - 1.0
        )
        return {
            "num_path_steps": self.count,
            "input_factor_covariance": identity.cpu().tolist(),
            "realized_factor_covariance_before_regrid": pre_covariance.cpu().tolist(),
            "realized_factor_covariance_after_regrid": post_covariance.cpu().tolist(),
            "maximum_absolute_factor_covariance_error_before_regrid": float(
                torch.max(torch.abs(pre_covariance - identity)).cpu()
            ),
            "frobenius_factor_covariance_error_before_regrid": float(
                torch.linalg.matrix_norm(pre_covariance - identity).cpu()
            ),
            "maximum_absolute_factor_covariance_error_after_regrid": float(
                torch.max(torch.abs(post_covariance - identity)).cpu()
            ),
            "maximum_absolute_coordinate_variance_relative_error": float(
                torch.max(torch.abs(relative_coordinate_error)).cpu()
            ),
            "root_mean_square_coordinate_variance_relative_error": float(
                torch.sqrt(torch.mean(relative_coordinate_error.square())).cpu()
            ),
            "post_regrid_variance_fraction_outside_input_factor_span": float(
                (self.post_outside_square / self.post_total_square).cpu()
            ),
        }


def _simulate(
    *,
    method: Method,
    regrid: bool,
    decoder: Any,
    initial_raw_states: torch.Tensor,
    loadings: torch.Tensor,
    innovations: torch.Tensor,
    branch_indices: torch.Tensor,
    spot_gaussian_ranks: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    initial_surface_options: torch.Tensor,
    state_instruments: np.ndarray,
    config: FourStepLvsqComparisonConfig,
) -> dict[str, object]:
    geometry = decoder.geometry
    step_y = config.calendar_business_days_per_step / DAYS_PER_YEAR
    state_batch_size = (
        config.exact_state_batch_size
        if method == "finite_h_exact"
        else config.sde_state_batch_size
    )
    current = initial_raw_states.clone()
    cumulative_spot = torch.ones(
        len(current), dtype=current.dtype, device=current.device
    )
    covariance = _CovarianceAccumulator(loadings, step_y=step_y)
    surface_audits: list[dict[str, object]] = []
    cash_audits: list[dict[str, object]] = []
    transition_seconds = 0.0
    spot_seconds = 0.0
    regrid_seconds = 0.0
    pricing_seconds = 0.0
    extrapolated_fractions: list[float] = []
    maximum_regrid_change = 0.0
    minimum_cap_margin = float("inf")
    newton_iterations: list[list[int]] = []
    started_total = perf_counter()

    for step_index in range(config.num_steps):
        _synchronize(current.device)
        started = perf_counter()
        gross_spot = _sample_front_spot_returns(
            decoder=decoder,
            raw_states=current,
            gaussian_ranks=spot_gaussian_ranks[step_index],
            batch_size=config.surface_batch_size,
        )
        _synchronize(current.device)
        spot_seconds += perf_counter() - started

        before = current
        before_regrid = torch.empty_like(before)
        drift = torch.empty_like(before)
        _synchronize(current.device)
        started = perf_counter()
        step_newton: list[int] = []
        for start in range(0, len(before), state_batch_size):
            stop = min(start + state_batch_size, len(before))
            selected_branches = branch_indices[step_index, start:stop]
            if method == "finite_h_exact":
                result = fixed_horizon_lvsq_exponential_step(
                    decoder,
                    before[start:stop],
                    raw_log_loadings=loadings,
                    calendar_time_increment_y=step_y,
                    moment_tolerance=config.moment_tolerance,
                )
                batch_arange = torch.arange(
                    stop - start, device=before.device
                )
                before_regrid[start:stop] = result.raw_states_by_branch[
                    batch_arange, selected_branches
                ]
                drift[start:stop] = result.effective_raw_drift
                step_newton.extend(result.newton_iterations_by_expiry)
                minimum_cap_margin = min(
                    minimum_cap_margin,
                    float(result.minimum_raw_cap_margin.cpu()),
                )
            else:
                restriction = fixed_contract_lvsq_drift(
                    decoder,
                    before[start:stop],
                    raw_log_loadings=loadings,
                )
                before_regrid[start:stop] = fixed_contract_lvsq_euler_step(
                    restriction,
                    before[start:stop],
                    calendar_time_increment_y=step_y,
                    standard_normals=innovations[selected_branches],
                )
                drift[start:stop] = restriction.raw_drift
                minimum_cap_margin = min(
                    minimum_cap_margin,
                    float(restriction.minimum_raw_cap_margin.cpu()),
                )
        _synchronize(current.device)
        transition_seconds += perf_counter() - started
        if step_newton:
            newton_iterations.append(step_newton)

        after_regrid = before_regrid
        if regrid:
            after_regrid = torch.empty_like(before_regrid)
            _synchronize(current.device)
            started = perf_counter()
            weighted_extrapolated = 0.0
            for start in range(0, len(before), config.surface_batch_size):
                stop = min(start + config.surface_batch_size, len(before))
                regridded = regrid_raw_dlvsq_to_relative_spot(
                    before_regrid[start:stop],
                    gross_spot[start:stop],
                    geometry=geometry,
                )
                after_regrid[start:stop] = regridded.raw_states
                weighted_extrapolated += (
                    (stop - start)
                    * float(regridded.extrapolated_coordinate_fraction.cpu())
                )
                maximum_regrid_change = max(
                    maximum_regrid_change,
                    float(regridded.maximum_absolute_raw_change.cpu()),
                )
            _synchronize(current.device)
            regrid_seconds += perf_counter() - started
            extrapolated_fractions.append(weighted_extrapolated / len(before))

        for start in range(0, len(before), config.surface_batch_size):
            stop = min(start + config.surface_batch_size, len(before))
            covariance.update(
                raw_before=before[start:stop],
                raw_before_regrid=before_regrid[start:stop],
                raw_after_regrid=after_regrid[start:stop],
                raw_drift=drift[start:stop],
                num_expiries=geometry.num_model_expiries,
                num_inner=geometry.num_inner_strikes,
                block_width=geometry.expiry_block_state_size,
            )
        current = after_regrid
        cumulative_spot = cumulative_spot * gross_spot
        current_blocks = current.reshape(
            len(current),
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )
        cap_margin = geometry.maximum_log_dlvsq - current_blocks[
            ..., : geometry.num_inner_strikes
        ]
        minimum_cap_margin = min(
            minimum_cap_margin, float(torch.min(cap_margin).cpu())
        )
        if minimum_cap_margin <= 0.0:
            raise RuntimeError("a simulated DLV reached the pricing cap")

        _synchronize(current.device)
        started = perf_counter()
        surface_options = _surface_option_values(
            decoder=decoder,
            raw_states=current,
            cumulative_spot=cumulative_spot,
            fixed_cash_strikes=fixed_cash_strikes,
            batch_size=config.surface_batch_size,
            fixed_cash=False,
        )
        cash_options = _surface_option_values(
            decoder=decoder,
            raw_states=current,
            cumulative_spot=cumulative_spot,
            fixed_cash_strikes=fixed_cash_strikes,
            batch_size=config.surface_batch_size,
            fixed_cash=True,
        )
        _synchronize(current.device)
        pricing_seconds += perf_counter() - started
        surface_audits.append(
            _risk_neutrality_audit(
                surface_options - initial_surface_options,
                state_instruments=state_instruments,
                eigenvalue_fraction=config.covariance_eigenvalue_fraction,
            )
        )
        cash_audits.append(
            _risk_neutrality_audit(
                cash_options - initial_surface_options,
                state_instruments=state_instruments,
                eigenvalue_fraction=config.covariance_eigenvalue_fraction,
            )
        )
        print(
            f"{method} regrid={regrid} step={step_index + 1}/{config.num_steps} "
            f"transition={transition_seconds:.2f}s "
            f"cash_max_z={cash_audits[-1]['maximum_absolute_coordinate_z']:.3f}",
            flush=True,
        )

    _synchronize(current.device)
    total_seconds = perf_counter() - started_total
    return {
        "method": method,
        "regrid": regrid,
        "speed": {
            "total_simulation_and_audit_seconds": total_seconds,
            "transition_seconds": transition_seconds,
            "spot_sampling_seconds": spot_seconds,
            "regrid_seconds": regrid_seconds,
            "option_pricing_seconds": pricing_seconds,
            "path_steps_per_transition_second": (
                config.num_paths * config.num_steps / transition_seconds
            ),
        },
        "risk_neutrality": {
            "surface_only_by_horizon": surface_audits,
            "fixed_cash_with_spot_by_horizon": cash_audits,
        },
        "realized_covariance": covariance.report(),
        "spot": {
            "terminal_mean": float(torch.mean(cumulative_spot).cpu()),
            "terminal_standard_deviation": float(
                torch.std(cumulative_spot, correction=1).cpu()
            ),
            "terminal_mean_z": float(
                (
                    (torch.mean(cumulative_spot) - 1.0)
                    / (torch.std(cumulative_spot, correction=1) / len(cumulative_spot) ** 0.5)
                ).cpu()
            ),
        },
        "regrid_diagnostics": {
            "extrapolated_coordinate_fraction_by_step": extrapolated_fractions,
            "maximum_absolute_raw_change": maximum_regrid_change,
        },
        "minimum_raw_cap_margin": minimum_cap_margin,
        "newton_iterations": {
            "maximum": max((max(value) for value in newton_iterations), default=0),
            "by_step_flattened_expiry_batches": newton_iterations,
        },
    }


def run(arguments: argparse.Namespace) -> None:
    """Run all four matched method/regridding combinations."""

    config = FourStepLvsqComparisonConfig(
        num_paths=arguments.num_paths,
        exact_state_batch_size=arguments.exact_state_batch_size,
        sde_state_batch_size=arguments.sde_state_batch_size,
        surface_batch_size=arguments.surface_batch_size,
    )
    env = BaseEnvironment(
        precision=arguments.precision,
        device=arguments.device,
        debug=False,
    )
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(
        geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig
    ):
        raise ValueError("four-step comparison requires the fixed relative grid")

    rng = np.random.default_rng(config.seed)
    initial_indices_numpy = rng.integers(
        0, len(latent_states), size=config.num_paths
    )
    with torch.no_grad():
        all_raw_states = model.raw_states_from_states(latent_states)
        initial_indices = torch.as_tensor(
            initial_indices_numpy, dtype=torch.long, device=env.device
        )
        initial_raw_states = all_raw_states[initial_indices]
        initial_raw_states, forward_parameters = _set_constant_bs_backbone(
            initial_raw_states,
            decoder=model.sanos_decoder,
            annualized_volatility=config.bs_backbone_annualized_volatility,
        )
        initial_surface = _decode_surface(model.sanos_decoder, initial_raw_states[:1])
        loadings = smooth_lvsq_log_loadings(
            initial_surface.support_strikes[0, 0],
            num_model_expiries=geometry.num_model_expiries,
            annualized_log_volatility=config.dlvsq_annualized_log_volatility,
        )
        innovations, probabilities = symmetric_rademacher_cubature(
            loadings.shape[-1], dtype=env.fdtype, device=env.device
        )
        del probabilities
        branch_indices = torch.as_tensor(
            rng.integers(
                0,
                len(innovations),
                size=(config.num_steps, config.num_paths),
            ),
            dtype=torch.long,
            device=env.device,
        )
        spot_gaussian_ranks = torch.as_tensor(
            rng.standard_normal((config.num_steps, config.num_paths)),
            dtype=env.fdtype,
            device=env.device,
        )
        fixed_cash_strikes = torch.exp(
            torch.linspace(
                -0.30,
                0.30,
                config.num_option_strikes,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        unit_spot = torch.ones(
            config.num_paths, dtype=env.fdtype, device=env.device
        )
        initial_surface_options = _surface_option_values(
            decoder=model.sanos_decoder,
            raw_states=initial_raw_states,
            cumulative_spot=unit_spot,
            fixed_cash_strikes=fixed_cash_strikes,
            batch_size=config.surface_batch_size,
            fixed_cash=False,
        )
        feature_loadings = loadings.reshape(-1, loadings.shape[-1])
        raw_dlvsq = initial_raw_states.reshape(
            config.num_paths,
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )[..., : geometry.num_inner_strikes].reshape(config.num_paths, -1)
        features = (raw_dlvsq @ feature_loadings).cpu().double().numpy()
        features = (features - np.mean(features, axis=0)) / np.std(
            features, axis=0, ddof=1
        )
        state_instruments = np.concatenate(
            (np.ones((config.num_paths, 1)), features), axis=1
        )

        simulations: list[dict[str, object]] = []
        for method in ("local_sde", "finite_h_exact"):
            for regrid in (False, True):
                simulations.append(
                    _simulate(
                        method=method,
                        regrid=regrid,
                        decoder=model.sanos_decoder,
                        initial_raw_states=initial_raw_states,
                        loadings=loadings,
                        innovations=innovations,
                        branch_indices=branch_indices,
                        spot_gaussian_ranks=spot_gaussian_ranks,
                        fixed_cash_strikes=fixed_cash_strikes,
                        initial_surface_options=initial_surface_options,
                        state_instruments=state_instruments,
                        config=config,
                    )
                )
                if initial_raw_states.device.type == "cuda":
                    torch.cuda.empty_cache()

    keyed = {
        f"{value['method']}_{'regrid' if value['regrid'] else 'no_regrid'}": value
        for value in simulations
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "four_step_heterogeneous_lvsq_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "initial_state_sampling": {
            "kind": "uniform_with_replacement",
            "num_source_states": len(latent_states),
            "num_paths": config.num_paths,
            "num_unique_source_states": int(len(np.unique(initial_indices_numpy))),
            "minimum_repetitions": int(
                np.min(np.bincount(initial_indices_numpy, minlength=len(latent_states)))
            ),
            "maximum_repetitions": int(
                np.max(np.bincount(initial_indices_numpy, minlength=len(latent_states)))
            ),
        },
        "backbone": {
            "annualized_bs_volatility": config.bs_backbone_annualized_volatility,
            "flex_forward_parameters": forward_parameters.cpu().tolist(),
            "total_volatilities": (
                config.bs_backbone_annualized_volatility
                * torch.sqrt(model.sanos_decoder.model_expiries_y)
            ).cpu().tolist(),
        },
        "innovation_law": {
            "kind": "four independent Rademacher factors",
            "num_branches": len(innovations),
            "matched_across_methods": True,
            "spot_gaussian_ranks_matched_across_methods": True,
        },
        "risk_neutrality_interpretation": {
            "surface_only": (
                "Same normalized strikes and expiry indices; this is the moment "
                "condition targeted directly by the fixed-contract LVsq step."
            ),
            "fixed_cash_with_spot": (
                "Accumulated spot times the current normalized option price at "
                "K/S; this tests whether adding spot and the coordinate regrid "
                "preserves fixed-cash option martingales."
            ),
            "state_instruments": (
                "Intercept plus four initial raw-DLV projections aligned with "
                "the four input loading directions."
            ),
            "covariance_handling": (
                "Sample covariance with eigenmodes below the configured fraction "
                "of the largest eigenvalue removed before the Wald test."
            ),
        },
        "regrid_interpretation": (
            "Shift each raw log excess-DLV curve in log relative strike after "
            "the realized spot move, with flat endpoint extrapolation. This is "
            "not an expiry roll or a density refit."
        ),
        "simulations": keyed,
        "speed_comparison": {
            "exact_over_sde_transition_time_no_regrid": (
                keyed["finite_h_exact_no_regrid"]["speed"]["transition_seconds"]
                / keyed["local_sde_no_regrid"]["speed"]["transition_seconds"]
            ),
            "exact_over_sde_total_time_no_regrid": (
                keyed["finite_h_exact_no_regrid"]["speed"]
                ["total_simulation_and_audit_seconds"]
                / keyed["local_sde_no_regrid"]["speed"]
                ["total_simulation_and_audit_seconds"]
            ),
            "exact_over_sde_transition_time_regrid": (
                keyed["finite_h_exact_regrid"]["speed"]["transition_seconds"]
                / keyed["local_sde_regrid"]["speed"]["transition_seconds"]
            ),
        },
    }
    write_json_atomic(report, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
