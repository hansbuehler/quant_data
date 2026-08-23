"""Sequential Monte Carlo fitting of weekly FLEX target-measure kernels."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.dysanos.flex.sanos import SanosSurface
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    NormalWeeklyMeasureKernel,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklySanosSimulator,
    front_week_inverse_cdf_components,
)

TargetKernel = NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class WeeklyMeasureTrainConfig:
    """Operational and statistical choices for target-kernel training."""

    epochs_per_line: int = 200
    origins_per_batch: int = 16
    paths_per_origin: int = 128
    audit_origins: int = 32
    audit_paths_per_origin: int = 256
    learning_rate: float = 5.0e-4
    p_regularization_weight: float = 1.0e-4
    gradient_clip: float = 10.0
    report_every: int = 10
    checkpoint_every: int = 25
    seed: int = 90210
    compile_objective: bool = True
    antithetic_paths: bool = True
    maximum_roll_lines: int | None = None
    log_moneyness: tuple[float, ...] = (
        -0.30,
        -0.24,
        -0.18,
        -0.12,
        -0.06,
        0.0,
        0.06,
        0.12,
        0.18,
        0.24,
        0.30,
    )
    minimum_relative_audit_improvement: float = 1.0e-3
    absolute_audit_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        verify_inp(
            self.epochs_per_line > 0
            and self.origins_per_batch > 0
            and self.paths_per_origin > 1
            and self.audit_origins > 0
            and self.audit_paths_per_origin > 1,
            lambda: "measure-training sample and epoch counts must be positive",
        )
        verify_inp(
            not self.antithetic_paths
            or (
                self.paths_per_origin % 2 == 0
                and self.audit_paths_per_origin % 2 == 0
            ),
            lambda: "antithetic path counts must be even",
        )
        verify_inp(
            self.maximum_roll_lines is None or self.maximum_roll_lines > 0,
            lambda: "'maximum_roll_lines' must be positive when provided",
        )
        verify_inp(
            self.learning_rate > 0.0
            and self.p_regularization_weight >= 0.0
            and self.gradient_clip > 0.0,
            lambda: "invalid target-measure optimizer controls",
        )
        verify_inp(
            self.report_every > 0 and self.checkpoint_every > 0,
            lambda: "report/checkpoint intervals must be positive",
        )
        verify_inp(
            len(self.log_moneyness) > 1
            and tuple(sorted(set(self.log_moneyness))) == self.log_moneyness,
            lambda: "target-measure log strikes must be unique and increasing",
        )
        verify_inp(
            0.0 <= self.minimum_relative_audit_improvement < 1.0
            and self.absolute_audit_tolerance >= 0.0,
            lambda: "invalid target-measure convergence gates",
        )


@dataclass(frozen=True)
class WeeklyMeasureTrainingData:
    """Precomputed current surfaces used as equally weighted Q origins."""

    reduced_states: torch.Tensor
    front_support: torch.Tensor
    front_density: torch.Tensor
    front_backbone_total_vol: torch.Tensor
    current_calls: torch.Tensor

    def __post_init__(self) -> None:
        rows = self.reduced_states.shape[0]
        if (
            self.reduced_states.ndim != 2
            or self.front_support.ndim != 2
            or self.front_density.shape != self.front_support.shape
            or self.front_support.shape[0] != rows
            or self.front_backbone_total_vol.shape != (rows,)
            or self.current_calls.ndim != 3
            or self.current_calls.shape[0] != rows
            or not all(
                value.device == self.reduced_states.device
                and value.dtype == self.reduced_states.dtype
                for value in (
                    self.front_support,
                    self.front_density,
                    self.front_backbone_total_vol,
                    self.current_calls,
                )
            )
        ):
            raise ValueError("invalid target-measure precomputed training data")


def precompute_measure_training_data(
    simulator: WeeklySanosSimulator,
    reduced_states: torch.Tensor,
    normalized_strikes: torch.Tensor,
    *,
    num_expiries: int | None = None,
) -> WeeklyMeasureTrainingData:
    """Decode each current surface once before stochastic Q optimization."""
    selected_expiries = (
        simulator.num_model_expiries if num_expiries is None else num_expiries
    )
    if not 1 <= selected_expiries <= simulator.num_model_expiries:
        raise ValueError(
            f"'num_expiries' must lie in [1,{simulator.num_model_expiries}]"
        )
    with torch.no_grad():
        surface = simulator.decode_surface(reduced_states)
        calls = torch.stack(
            tuple(
                simulator.normalized_call_prices(
                    reduced_states,
                    expiry_index=expiry_index,
                    normalized_strikes=normalized_strikes,
                )
                for expiry_index in range(selected_expiries)
            ),
            dim=1,
        )
    return WeeklyMeasureTrainingData(
        reduced_states=reduced_states.detach(),
        front_support=surface.support_strikes[:, 0].detach(),
        front_density=surface.densities[:, 0].detach(),
        front_backbone_total_vol=surface.backbone_total_vols[:, 0].detach(),
        current_calls=calls.detach(),
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _active_parameters(kernel: TargetKernel, line_index: int) -> list[torch.nn.Parameter]:
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    network = kernel.context_networks[line_index]
    parameters = list(network.parameters())
    local = (
        kernel.local_factor_raw[line_index]
        if isinstance(kernel, NormalWeeklyMeasureKernel)
        else kernel.strict_lower_raw[line_index]
    )
    parameters.append(local)
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _expanded_batch(
    data: WeeklyMeasureTrainingData,
    indices: torch.Tensor,
    *,
    paths_per_origin: int,
    generator: torch.Generator,
    antithetic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated = torch.repeat_interleave(indices, paths_per_origin)
    current = data.reduced_states[repeated]
    if antithetic:
        if paths_per_origin % 2 != 0:
            raise ValueError("antithetic path counts must be even")
        half = torch.randn(
            (len(indices), paths_per_origin // 2, current.shape[1] + 1),
            dtype=current.dtype,
            device=current.device,
            generator=generator,
        )
        normals = torch.cat((half, -half), dim=1).reshape(
            len(repeated), current.shape[1] + 1
        )
    else:
        normals = torch.randn(
            (len(repeated), current.shape[1] + 1),
            dtype=current.dtype,
            device=current.device,
            generator=generator,
        )
    gross = front_week_inverse_cdf_components(
        data.front_support[repeated],
        data.front_density[repeated],
        data.front_backbone_total_vol[repeated],
        normals[:, 0],
    )
    return current, gross, normals, repeated


def _make_residual_function(
    kernel: TargetKernel,
    simulator: WeeklySanosSimulator,
    normalized_strikes: torch.Tensor,
    *,
    current_expiry_index: int | None,
    compile_objective: bool,
) -> Any:
    if current_expiry_index is None:

        def residual_function(
            current: torch.Tensor,
            gross: torch.Tensor,
            normals: torch.Tensor,
            current_calls: torch.Tensor,
        ) -> torch.Tensor:
            draw = kernel.step(current, gross, normals)
            return simulator.hjm_roll_residuals(
                current,
                draw.reduced_state,
                gross,
                normalized_strikes,
                current_call_prices=current_calls,
            )[:, 1:]

    else:

        def residual_function(
            current: torch.Tensor,
            gross: torch.Tensor,
            normals: torch.Tensor,
            current_calls: torch.Tensor,
        ) -> torch.Tensor:
            draw = kernel.step(current, gross, normals)
            return simulator.hjm_roll_line_residuals(
                current,
                draw.reduced_state,
                gross,
                normalized_strikes,
                current_expiry_index=current_expiry_index,
                current_call_prices=current_calls,
            ).unsqueeze(1)

    return (
        torch.compile(residual_function, dynamic=False, fullgraph=False)
        if compile_objective
        else residual_function
    )


def _conditional_moment_statistics(
    residuals: torch.Tensor,
    *,
    origins: int,
    paths_per_origin: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    shaped = residuals.reshape(origins, paths_per_origin, *residuals.shape[1:])
    moments = torch.mean(shaped, dim=1)
    loss = torch.mean(moments.square())
    return loss, {
        "conditional_moment_rmse": float(torch.sqrt(loss).detach().cpu()),
        "conditional_moment_mae": float(torch.mean(torch.abs(moments)).detach().cpu()),
        "maximum_conditional_moment_abs": float(torch.max(torch.abs(moments)).detach().cpu()),
        "path_residual_rmse": float(torch.sqrt(torch.mean(residuals.square())).detach().cpu()),
    }


def _antithetic_mean_and_standard_error(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a path expectation using paired ``z,-z`` observations."""
    paths = values.shape[1]
    if paths < 4 or paths % 2 != 0:
        raise ValueError("antithetic diagnostics require an even path count >= 4")
    half = paths // 2
    pair_values = 0.5 * (values[:, :half] + values[:, half:])
    estimate = torch.mean(pair_values, dim=1)
    standard_error = torch.std(pair_values, dim=1, correction=1) / math.sqrt(half)
    return estimate, standard_error


def _moment_audit_summary(
    estimates: torch.Tensor,
    standard_errors: torch.Tensor,
    normalized_strikes: torch.Tensor,
) -> dict[str, object]:
    """Summarize conditional price errors in normalized spot units."""
    if estimates.shape != standard_errors.shape or estimates.ndim != 2:
        raise ValueError("moment audit estimates must have shape (origins,strikes)")
    scale = torch.clamp_min(
        standard_errors,
        32.0 * torch.finfo(estimates.dtype).eps,
    )
    standardized = torch.abs(estimates) / scale
    per_strike: list[dict[str, float]] = []
    for strike_index, strike in enumerate(normalized_strikes):
        line = estimates[:, strike_index]
        line_standardized = standardized[:, strike_index]
        per_strike.append(
            {
                "normalized_strike": float(strike.detach().cpu()),
                "signed_mean_bps_spot": float(torch.mean(line).detach().cpu())
                * 1.0e4,
                "rmse_bps_spot": float(torch.sqrt(torch.mean(line.square())).detach().cpu())
                * 1.0e4,
                "maximum_abs_bps_spot": float(torch.max(torch.abs(line)).detach().cpu())
                * 1.0e4,
                "percent_within_3_mc_se": float(
                    100.0 * torch.mean((line_standardized <= 3.0).to(estimates.dtype)).detach().cpu()
                ),
            }
        )
    return {
        "num_origins": estimates.shape[0],
        "num_strikes": estimates.shape[1],
        "conditional_moment_rmse": float(
            torch.sqrt(torch.mean(estimates.square())).detach().cpu()
        ),
        "conditional_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(estimates.square())).detach().cpu()
        )
        * 1.0e4,
        "conditional_moment_mae_bps_spot": float(
            torch.mean(torch.abs(estimates)).detach().cpu()
        )
        * 1.0e4,
        "maximum_conditional_moment_abs_bps_spot": float(
            torch.max(torch.abs(estimates)).detach().cpu()
        )
        * 1.0e4,
        "mean_mc_standard_error_bps_spot": float(
            torch.mean(standard_errors).detach().cpu()
        )
        * 1.0e4,
        "percent_within_1_mc_se": float(
            100.0 * torch.mean((standardized <= 1.0).to(estimates.dtype)).detach().cpu()
        ),
        "percent_within_2_mc_se": float(
            100.0 * torch.mean((standardized <= 2.0).to(estimates.dtype)).detach().cpu()
        ),
        "percent_within_3_mc_se": float(
            100.0 * torch.mean((standardized <= 3.0).to(estimates.dtype)).detach().cpu()
        ),
        "per_strike": per_strike,
    }


def audit_front_week_consistency(
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    *,
    paths_per_origin: int,
    origins_per_batch: int,
    seed: int,
) -> dict[str, object]:
    """Verify the implied-spot marginal against every current front call line."""
    if normalized_strikes.ndim != 1 or normalized_strikes.shape[0] != (
        data.current_calls.shape[-1]
    ):
        raise ValueError("front audit strike geometry differs from cached calls")
    if paths_per_origin < 4 or paths_per_origin % 2 != 0:
        raise ValueError("front audit requires an even path count >= 4")
    if origins_per_batch <= 0:
        raise ValueError("'origins_per_batch' must be positive")
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    residual_batches: list[torch.Tensor] = []
    standard_error_batches: list[torch.Tensor] = []
    gross_error_batches: list[torch.Tensor] = []
    gross_standard_error_batches: list[torch.Tensor] = []
    for start in range(0, len(data.reduced_states), origins_per_batch):
        stop = min(start + origins_per_batch, len(data.reduced_states))
        indices = torch.arange(start, stop, device=data.reduced_states.device)
        _, gross, _, _ = _expanded_batch(
            data,
            indices,
            paths_per_origin=paths_per_origin,
            generator=generator,
            antithetic=True,
        )
        shaped_gross = gross.reshape(len(indices), paths_per_origin)
        payoff = torch.relu(
            shaped_gross.unsqueeze(-1)
            - normalized_strikes.reshape(1, 1, -1)
        )
        estimate, standard_error = _antithetic_mean_and_standard_error(payoff)
        gross_estimate, gross_standard_error = _antithetic_mean_and_standard_error(
            shaped_gross.unsqueeze(-1)
        )
        residual_batches.append(
            (estimate - data.current_calls[indices, 0]).detach().cpu()
        )
        standard_error_batches.append(standard_error.detach().cpu())
        gross_error_batches.append((gross_estimate[:, 0] - 1.0).detach().cpu())
        gross_standard_error_batches.append(
            gross_standard_error[:, 0].detach().cpu()
        )
    residuals = torch.cat(residual_batches)
    standard_errors = torch.cat(standard_error_batches)
    gross_errors = torch.cat(gross_error_batches)
    gross_standard_errors = torch.cat(gross_standard_error_batches)
    density_mass_error = torch.sum(data.front_density, dim=-1) - 1.0
    density_mean_error = torch.sum(
        data.front_density * data.front_support, dim=-1
    ) - 1.0
    result = _moment_audit_summary(
        residuals,
        standard_errors,
        normalized_strikes.detach().cpu(),
    )
    gross_scale = torch.clamp_min(
        gross_standard_errors,
        32.0 * torch.finfo(gross_errors.dtype).eps,
    )
    result.update(
        {
            "paths_per_origin": paths_per_origin,
            "sampling": "fresh origin-wise antithetic Torch normals",
            "maximum_density_mass_error": float(
                torch.max(torch.abs(density_mass_error)).detach().cpu()
            ),
            "maximum_analytic_mean_gross_return_error": float(
                torch.max(torch.abs(density_mean_error)).detach().cpu()
            ),
            "sampled_mean_gross_return_rmse_bps": float(
                torch.sqrt(torch.mean(gross_errors.square()))
            )
            * 1.0e4,
            "sampled_mean_gross_return_maximum_abs_bps": float(
                torch.max(torch.abs(gross_errors))
            )
            * 1.0e4,
            "sampled_mean_gross_return_percent_within_3_mc_se": float(
                100.0
                * torch.mean(
                    (torch.abs(gross_errors) / gross_scale <= 3.0).to(
                        gross_errors.dtype
                    )
                )
            ),
        }
    )
    return result


def audit_weekly_roll_line(
    kernel: TargetKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    *,
    current_expiry_index: int,
    paths_per_origin: int,
    origins_per_batch: int,
    seed: int,
) -> dict[str, object]:
    """Evaluate one conditional roll line on every supplied historic origin."""
    if paths_per_origin < 4 or paths_per_origin % 2 != 0:
        raise ValueError("roll audit requires an even path count >= 4")
    residual_function = _make_residual_function(
        kernel,
        simulator,
        normalized_strikes,
        current_expiry_index=current_expiry_index,
        compile_objective=False,
    )
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    estimate_batches: list[torch.Tensor] = []
    standard_error_batches: list[torch.Tensor] = []
    path_square_total = 0.0
    path_count = 0
    with torch.no_grad():
        for start in range(0, len(data.reduced_states), origins_per_batch):
            stop = min(start + origins_per_batch, len(data.reduced_states))
            indices = torch.arange(start, stop, device=data.reduced_states.device)
            current, gross, normals, repeated = _expanded_batch(
                data,
                indices,
                paths_per_origin=paths_per_origin,
                generator=generator,
                antithetic=True,
            )
            residuals = residual_function(
                current,
                gross,
                normals,
                data.current_calls[repeated, current_expiry_index],
            )[:, 0].reshape(len(indices), paths_per_origin, -1)
            estimate, standard_error = _antithetic_mean_and_standard_error(
                residuals
            )
            estimate_batches.append(estimate.detach().cpu())
            standard_error_batches.append(standard_error.detach().cpu())
            path_square_total += float(torch.sum(residuals.square()).detach().cpu())
            path_count += residuals.numel()
    estimates = torch.cat(estimate_batches)
    standard_errors = torch.cat(standard_error_batches)
    result = _moment_audit_summary(
        estimates,
        standard_errors,
        normalized_strikes.detach().cpu(),
    )
    result.update(
        {
            "current_expiry_index": current_expiry_index,
            "rolled_expiry_index": current_expiry_index - 1,
            "paths_per_origin": paths_per_origin,
            "sampling": "fresh origin-wise antithetic Torch normals",
            "path_residual_rmse": math.sqrt(path_square_total / path_count),
        }
    )
    return result


def _audit_indices(rows: int, count: int, *, device: torch.device) -> torch.Tensor:
    count = min(rows, count)
    return torch.linspace(0, rows - 1, count, device=device).round().long()


def _audit_line(
    residual_function: Any,
    data: WeeklyMeasureTrainingData,
    *,
    current_expiry_index: int | None,
    config: WeeklyMeasureTrainConfig,
    line_index: int,
) -> dict[str, float]:
    origins = min(len(data.reduced_states), config.audit_origins)
    indices = _audit_indices(
        len(data.reduced_states), origins, device=data.reduced_states.device
    )
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(config.seed + 10_000 + line_index)
    current, gross, normals, repeated = _expanded_batch(
        data,
        indices,
        paths_per_origin=config.audit_paths_per_origin,
        generator=generator,
        antithetic=config.antithetic_paths,
    )
    calls = (
        data.current_calls[repeated]
        if current_expiry_index is None
        else data.current_calls[repeated, current_expiry_index]
    )
    with torch.no_grad():
        residuals = residual_function(current, gross, normals, calls)
        _, statistics = _conditional_moment_statistics(
            residuals,
            origins=origins,
            paths_per_origin=config.audit_paths_per_origin,
        )
    return statistics


def _save_checkpoint(
    path: Path,
    *,
    kernel: TargetKernel,
    config: WeeklyMeasureTrainConfig,
    state_sha256: str,
    line_index: int,
    completed_epochs: int,
    histories: list[dict[str, object]],
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator,
    phase: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-weekly-measure-kernel-v1",
            "phase": phase,
            "kernel_class": type(kernel).__name__,
            "kernel_config": asdict(kernel.config),
            "kernel_config_version": kernel.config.version.unique_id48,
            "train_config": asdict(config),
            "train_config_version": config.version.unique_id48,
            "state_sha256": state_sha256,
            "line_index": line_index,
            "completed_epochs": completed_epochs,
            "histories": histories,
            "model_state_dict": kernel.state_dict(),
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
            "generator_state": generator.get_state(),
        },
        temporary,
    )
    temporary.replace(path)


def _restore_checkpoint(
    path: Path,
    *,
    kernel: TargetKernel,
    config: WeeklyMeasureTrainConfig,
    state_sha256: str,
) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": "flex-weekly-measure-kernel-v1",
        "kernel_class": type(kernel).__name__,
        "kernel_config": asdict(kernel.config),
        "kernel_config_version": kernel.config.version.unique_id48,
        "train_config": asdict(config),
        "train_config_version": config.version.unique_id48,
        "state_sha256": state_sha256,
    }
    if not isinstance(payload, Mapping) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"target-measure checkpoint identity mismatch: {path}")
    kernel.load_state_dict(cast(Mapping[str, torch.Tensor], payload["model_state_dict"]))
    return payload


def fit_weekly_measure_kernel(
    kernel: TargetKernel,
    simulator: WeeklySanosSimulator,
    training_data: WeeklyMeasureTrainingData,
    validation_data: WeeklyMeasureTrainingData,
    *,
    checkpoint_path: Path,
    config: WeeklyMeasureTrainConfig,
) -> dict[str, object]:
    """Fit a direct joint kernel or sequential expiry-triangular kernel."""
    if kernel.dynamics is not simulator.dynamics:
        raise ValueError("kernel and simulator do not share the fitted P dynamics")
    if training_data.reduced_states.shape[1] != kernel.config.state_size:
        raise ValueError("training data and measure kernel dimensions differ")
    normalized_strikes = torch.exp(
        torch.tensor(
            config.log_moneyness,
            dtype=kernel.fdtype,
            device=kernel.device,
        )
    )
    state_sha256 = _tensor_sha256(training_data.reduced_states)
    restored = _restore_checkpoint(
        checkpoint_path,
        kernel=kernel,
        config=config,
        state_sha256=state_sha256,
    )
    histories = (
        list(cast(list[dict[str, object]], restored.get("histories", [])))
        if restored is not None
        else []
    )
    if restored is not None and restored.get("phase") == "complete":
        return {
            "status": "complete",
            "checkpoint_action": "loaded",
            "lines": histories,
        }
    if restored is not None and restored.get("phase") == "failed_convergence":
        raise RuntimeError(
            "Existing Q checkpoint records failed convergence; no silent "
            f"fallback or automatic reset is allowed: {checkpoint_path}"
        )
    start_line = int(restored.get("line_index", 0)) if restored is not None else 0
    start_epoch = int(restored.get("completed_epochs", 0)) if restored is not None else 0
    generator = torch.Generator(device=kernel.device)
    if restored is None:
        generator.manual_seed(config.seed)
    else:
        generator.set_state(cast(torch.Tensor, restored["generator_state"]))

    num_lines = (
        kernel.transformed_block_count
        if config.maximum_roll_lines is None
        else min(kernel.transformed_block_count, config.maximum_roll_lines)
    )
    for line_index in range(start_line, num_lines):
        current_expiry_index = line_index + 1 if kernel.config.hierarchical else None
        residual_function = _make_residual_function(
            kernel,
            simulator,
            normalized_strikes,
            current_expiry_index=current_expiry_index,
            compile_objective=config.compile_objective,
        )
        active = _active_parameters(kernel, line_index)
        optimizer = torch.optim.Adam(active, lr=config.learning_rate)
        line_start_epoch = start_epoch if line_index == start_line else 0
        if line_start_epoch > 0 and restored is not None:
            optimizer_state = restored.get("optimizer_state_dict")
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("in-progress Q checkpoint has no optimizer state")
            optimizer.load_state_dict(cast(dict[str, Any], optimizer_state))
        partial = (
            histories.pop()
            if restored is not None
            and restored.get("phase") == "training"
            and histories
            and histories[-1].get("status") == "training"
            and histories[-1].get("line_index") == line_index
            else None
        )
        initial_audit = (
            cast(dict[str, float], partial["initial_audit"])
            if partial is not None
            else _audit_line(
                residual_function,
                training_data,
                current_expiry_index=current_expiry_index,
                config=config,
                line_index=line_index,
            )
        )
        epoch_history = (
            list(cast(list[dict[str, float]], partial["epochs"]))
            if partial is not None
            else []
        )
        for epoch in range(line_start_epoch, config.epochs_per_line):
            indices = torch.randint(
                len(training_data.reduced_states),
                (config.origins_per_batch,),
                device=kernel.device,
                generator=generator,
            )
            current, gross, normals, repeated = _expanded_batch(
                training_data,
                indices,
                paths_per_origin=config.paths_per_origin,
                generator=generator,
                antithetic=config.antithetic_paths,
            )
            calls = (
                training_data.current_calls[repeated]
                if current_expiry_index is None
                else training_data.current_calls[repeated, current_expiry_index]
            )
            optimizer.zero_grad(set_to_none=True)
            residuals = residual_function(current, gross, normals, calls)
            moment_loss, statistics = _conditional_moment_statistics(
                residuals,
                origins=config.origins_per_batch,
                paths_per_origin=config.paths_per_origin,
            )
            regularization = (
                kernel.conditional_kl_loss(
                    current,
                    gross,
                    line_index=line_index,
                )
                if isinstance(kernel, NormalWeeklyMeasureKernel)
                and line_index == 0
                else kernel.regularization_loss(
                    training_data.reduced_states[indices]
                )
            )
            loss = moment_loss + config.p_regularization_weight * regularization
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                active, config.gradient_clip
            )
            optimizer.step()
            record = {
                "epoch": float(epoch + 1),
                "loss": float(loss.detach().cpu()),
                "moment_loss": float(moment_loss.detach().cpu()),
                "regularization": float(regularization.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                **statistics,
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise RuntimeError(
                    f"non-finite Q training statistic at line {line_index}, epoch {epoch + 1}"
                )
            epoch_history.append(record)
            if (epoch + 1) % config.report_every == 0 or epoch == line_start_epoch:
                print(
                    f"weekly Q {type(kernel).__name__} line={line_index + 1}/"
                    f"{num_lines} epoch={epoch + 1}/{config.epochs_per_line} "
                    f"moment_rmse={statistics['conditional_moment_rmse']:.7g}",
                    flush=True,
                )
            if (epoch + 1) % config.checkpoint_every == 0:
                partial_histories = histories + [
                    {
                        "line_index": line_index,
                        "status": "training",
                        "initial_audit": initial_audit,
                        "epochs": epoch_history,
                    }
                ]
                _save_checkpoint(
                    checkpoint_path,
                    kernel=kernel,
                    config=config,
                    state_sha256=state_sha256,
                    line_index=line_index,
                    completed_epochs=epoch + 1,
                    histories=partial_histories,
                    optimizer=optimizer,
                    generator=generator,
                    phase="training",
                )
        final_train_audit = _audit_line(
            residual_function,
            training_data,
            current_expiry_index=current_expiry_index,
            config=config,
            line_index=line_index,
        )
        final_validation_audit = _audit_line(
            residual_function,
            validation_data,
            current_expiry_index=current_expiry_index,
            config=config,
            line_index=line_index + 1_000,
        )
        initial_rmse = initial_audit["conditional_moment_rmse"]
        final_rmse = final_train_audit["conditional_moment_rmse"]
        relative_improvement = (initial_rmse - final_rmse) / max(
            initial_rmse, np.finfo(float).tiny
        )
        converged = final_rmse <= config.absolute_audit_tolerance or (
            relative_improvement >= config.minimum_relative_audit_improvement
        )
        line_report: dict[str, object] = {
            "line_index": line_index,
            "current_expiry_index": current_expiry_index,
            "status": "complete" if converged else "failed_convergence",
            "initial_audit": initial_audit,
            "final_train_audit": final_train_audit,
            "validation_2025_audit": final_validation_audit,
            "relative_train_audit_improvement": relative_improvement,
            "epochs": epoch_history,
        }
        histories.append(line_report)
        if not converged:
            _save_checkpoint(
                checkpoint_path,
                kernel=kernel,
                config=config,
                state_sha256=state_sha256,
                line_index=line_index,
                completed_epochs=config.epochs_per_line,
                histories=histories,
                optimizer=optimizer,
                generator=generator,
                phase="failed_convergence",
            )
            raise RuntimeError(
                f"Q line {line_index} failed measured convergence: "
                f"initial/final RMSE={initial_rmse:.7g}/{final_rmse:.7g}"
            )
        _save_checkpoint(
            checkpoint_path,
            kernel=kernel,
            config=config,
            state_sha256=state_sha256,
            line_index=line_index + 1,
            completed_epochs=0,
            histories=histories,
            optimizer=None,
            generator=generator,
            phase="training" if line_index + 1 < num_lines else "complete",
        )
        start_epoch = 0
        restored = None
    return {
        "status": "complete",
        "checkpoint_action": "fitted",
        "lines": histories,
    }


__all__ = [
    "WeeklyMeasureTrainConfig",
    "WeeklyMeasureTrainingData",
    "audit_front_week_consistency",
    "audit_weekly_roll_line",
    "fit_weekly_measure_kernel",
    "precompute_measure_training_data",
]
