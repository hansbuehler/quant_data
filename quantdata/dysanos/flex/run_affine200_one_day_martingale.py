"""Fit and audit statewise one-day martingale drifts for affine Direct-200.

The correction changes the conditional mean of all 200 Gaussian state
innovations.  It does not freeze forward-volatility or eta coordinates: the
entire affine decoder moves with the corrected latent state.  Spot continues
to be drawn from the current front-expiry SANOS law.  Equivalently, the
orthogonal spot innovation receives the compensating Gaussian mean shift that
keeps the front-spot score centered.

The default experiment fits 1,000 independently sampled initial states and
audits every fitted state on 10,000 fresh paths.  Both fitting and validation
are streamed in bounded state/path batches, and antithetic pairs remain the
independent units throughout the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2
from scipy.stats import t as student_t

from quantdata.dysanos.flex.front_spot_copula import (
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _nms_book,
    _nms_relative_strikes,
    _price_calls,
)
from quantdata.dysanos.flex.weekly_simulation import (
    front_week_inverse_cdf_components,
)

NMS_GRID = (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5)
EXPIRY_DAYS = (5, 10, 15, 20)
LINE_SEARCH_SCALES = (1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0)
CHECKPOINT_SCHEMA = "dysanos-affine200-one-day-martingale-v1"
_TANH_STANDARD_DEVIATION = 0.627928730349203
_TANH_SQUARE_MEAN = 0.3942944903979622
_TANH_SQUARE_STANDARD_DEVIATION = 0.31228790895077047


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-states", type=int, default=1_000)
    parser.add_argument(
        "--state-source",
        choices=("invariant", "history"),
        default="invariant",
        help="Sample fitted invariant P states or distinct fitted historical states.",
    )
    parser.add_argument("--fit-paths-per-state", type=int, default=10_000)
    parser.add_argument("--fit-path-batch-size", type=int, default=1_000)
    parser.add_argument("--validation-paths-per-state", type=int, default=10_000)
    parser.add_argument("--state-batch-size", type=int, default=8)
    parser.add_argument("--validation-path-batch-size", type=int, default=1_000)
    parser.add_argument(
        "--fit-backend",
        choices=("score-newton", "pathwise-adam"),
        default="score-newton",
    )
    parser.add_argument("--newton-iterations", type=int, default=2)
    parser.add_argument(
        "--newton-mode",
        choices=("minimum-final-kl", "incremental"),
        default="minimum-final-kl",
        help=(
            "Use minimum-final-KL SQP targets or conventional incremental "
            "truncated/damped Newton steps."
        ),
    )
    parser.add_argument(
        "--spot-basis-size",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "Number of bounded spot-score functions: 1, normalized tanh, and "
            "centered normalized tanh squared."
        ),
    )
    parser.add_argument("--relative-singular-value-cutoff", type=float, default=1.0e-6)
    parser.add_argument("--relative-damping", type=float, default=1.0e-3)
    parser.add_argument("--maximum-step-norm", type=float, default=10.0)
    parser.add_argument("--maximum-control-norm", type=float, default=10.0)
    parser.add_argument("--adam-epochs", type=int, default=50)
    parser.add_argument("--adam-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--adam-control-l2", type=float, default=0.0)
    parser.add_argument("--adam-absolute-sd-floor", type=float, default=1.0e-4)
    parser.add_argument("--adam-relative-sd-floor", type=float, default=1.0e-2)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help=(
            "Initialize pathwise-Adam controls from another completed run. "
            "Adam moments and zero-control P whitening are always rebuilt."
        ),
    )
    parser.add_argument("--correlation-eigenvalue-cutoff", type=float, default=0.01)
    parser.add_argument("--state-seed", type=int, default=2_026_100_9)
    parser.add_argument("--fit-seed", type=int, default=3_026_100_9)
    parser.add_argument("--validation-seed", type=int, default=4_026_100_9)
    parser.add_argument(
        "--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1)
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive_integers = (
        arguments.num_states,
        arguments.fit_paths_per_state,
        arguments.fit_path_batch_size,
        arguments.validation_paths_per_state,
        arguments.state_batch_size,
        arguments.validation_path_batch_size,
        arguments.newton_iterations,
        arguments.adam_epochs,
        arguments.spot_basis_size,
    )
    if min(positive_integers) <= 0:
        raise ValueError("state, path, batch, and Newton counts must be positive")
    even_counts = (
        arguments.fit_paths_per_state,
        arguments.fit_path_batch_size,
        arguments.validation_paths_per_state,
        arguments.validation_path_batch_size,
    )
    if any(value % 2 for value in even_counts):
        raise ValueError("fit, validation, and validation-batch path counts must be even")
    if arguments.validation_paths_per_state % arguments.validation_path_batch_size:
        raise ValueError("validation paths must be divisible by the path batch size")
    if arguments.fit_paths_per_state % arguments.fit_path_batch_size:
        raise ValueError("fit paths must be divisible by the fit path batch size")
    if arguments.fit_seed == arguments.validation_seed:
        raise ValueError("fit and validation random streams must use distinct seeds")
    if (
        arguments.warm_start_checkpoint is not None
        and arguments.fit_backend != "pathwise-adam"
    ):
        raise ValueError("warm starts are supported only by the pathwise-Adam backend")
    if not 0.0 <= arguments.relative_singular_value_cutoff < 1.0:
        raise ValueError("relative singular-value cutoff must lie in [0, 1)")
    if arguments.relative_damping < 0.0 or not math.isfinite(arguments.relative_damping):
        raise ValueError("relative damping must be finite and non-negative")
    if arguments.maximum_step_norm <= 0.0 or not math.isfinite(
        arguments.maximum_step_norm
    ):
        raise ValueError("maximum step norm must be finite and positive")
    if arguments.maximum_control_norm <= 0.0 or not math.isfinite(
        arguments.maximum_control_norm
    ):
        raise ValueError("maximum control norm must be finite and positive")
    if (
        not math.isfinite(arguments.adam_learning_rate)
        or arguments.adam_learning_rate <= 0.0
        or not 0.0 <= arguments.adam_beta1 < 1.0
        or not 0.0 <= arguments.adam_beta2 < 1.0
        or not math.isfinite(arguments.adam_epsilon)
        or arguments.adam_epsilon <= 0.0
        or not math.isfinite(arguments.adam_control_l2)
        or arguments.adam_control_l2 < 0.0
        or not math.isfinite(arguments.adam_absolute_sd_floor)
        or arguments.adam_absolute_sd_floor <= 0.0
        or not math.isfinite(arguments.adam_relative_sd_floor)
        or arguments.adam_relative_sd_floor < 0.0
    ):
        raise ValueError("pathwise-Adam controls are outside their valid ranges")
    if not 0.0 < arguments.correlation_eigenvalue_cutoff < 1.0:
        raise ValueError("correlation eigenvalue cutoff must lie in (0, 1)")
    common_roots = {Path("/tmp"), Path("/media/storage")}
    if arguments.output.resolve() in common_roots:
        raise ValueError("output must be a project-specific subdirectory")


def _resource_status(output: Path) -> dict[str, object]:
    memory: dict[str, int] = {}
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            key, value, *_ = line.split()
            memory[key.removesuffix(":")] = int(value)
    memory_used = 1.0 - memory["MemAvailable"] / memory["MemTotal"]
    temporary = os.statvfs("/tmp")
    temporary_used = 1.0 - temporary.f_bavail / temporary.f_blocks
    output_disk = os.statvfs(output)
    storage_free: float | None = None
    storage = Path("/media/storage")
    if storage.exists():
        storage_disk = os.statvfs(storage)
        storage_free = storage_disk.f_bavail / storage_disk.f_blocks
    result: dict[str, object] = {
        "memory_used_fraction": memory_used,
        "memory_warning": memory_used >= 0.8,
        "output_free_fraction": output_disk.f_bavail / output_disk.f_blocks,
        "storage_free_fraction": storage_free,
        "storage_warning": storage_free is not None and storage_free < 0.2,
        "tmp_used_fraction": temporary_used,
        "tmp_warning": temporary_used >= 0.8,
        "tmp_cleanup_action": (
            "no project temporary files: all generated artifacts are under output"
            if temporary_used >= 0.7
            else None
        ),
    }
    if memory_used >= 0.9:
        raise RuntimeError("RAM use reached the project stop threshold")
    if storage_free is not None and storage_free <= 0.1:
        raise RuntimeError("/media/storage reached the project stop threshold")
    if temporary_used >= 0.9:
        raise RuntimeError("temporary filesystem reached the project stop threshold")
    return result


class _ProgressLog:
    def __init__(self, output: Path) -> None:
        self.path = output / "progress.jsonl"
        self.started = time.perf_counter()

    def write(self, stage: str, **values: object) -> None:
        payload = {
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - self.started,
            **values,
        }
        line = json.dumps(payload, allow_nan=False)
        print(line, flush=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()


def _decode_surface(model: Any, states: torch.Tensor) -> tuple[Any, Any, torch.Tensor]:
    raw = model.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    return parameters, surface, raw


def _antithetic_pair_means(values: torch.Tensor) -> torch.Tensor:
    """Return first-half/second-half antithetic pair means."""
    if values.ndim < 2 or values.shape[1] <= 0 or values.shape[1] % 2:
        raise ValueError("antithetic values need a positive even path axis")
    half = values.shape[1] // 2
    return 0.5 * (values[:, :half] + values[:, half:])


def _batched_block_jacobian(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Differentiate independent state blocks using one reverse pass per moment.

    ``outputs[b]`` must depend only on ``inputs[b]``.  Summing an output
    coordinate across states therefore returns every diagonal Jacobian block
    without constructing the zero cross-state blocks.
    """
    if outputs.ndim != 2 or inputs.ndim != 2 or outputs.shape[0] != inputs.shape[0]:
        raise ValueError("outputs and inputs must be aligned state matrices")
    rows: list[torch.Tensor] = []
    for moment in range(outputs.shape[1]):
        gradient = torch.autograd.grad(
            outputs[:, moment].sum(),
            inputs,
            retain_graph=moment + 1 < outputs.shape[1],
        )[0]
        rows.append(gradient)
    return torch.stack(rows, dim=1)


@dataclass(frozen=True)
class _NewtonBatch:
    step: np.ndarray
    maximum_singular_value: np.ndarray
    minimum_singular_value: np.ndarray
    condition_number: np.ndarray
    effective_rank: np.ndarray
    unrestricted_step_norm: np.ndarray
    accepted_step_norm: np.ndarray
    trust_scale: np.ndarray
    linearized_residual_rms: np.ndarray


def _truncated_minimum_norm_steps(
    residual: np.ndarray,
    jacobian: np.ndarray,
    *,
    relative_cutoff: float,
    relative_damping: float,
    maximum_norm: float,
) -> _NewtonBatch:
    """Solve independent regularized minimum-norm Newton systems."""
    residual = np.asarray(residual, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if (
        residual.ndim != 2
        or jacobian.ndim != 3
        or jacobian.shape[:2] != residual.shape
    ):
        raise ValueError("residual and Jacobian must have shapes (B,M), (B,M,C)")
    batch = len(residual)
    steps = np.zeros((batch, jacobian.shape[2]), dtype=np.float64)
    maximum = np.zeros(batch, dtype=np.float64)
    minimum = np.zeros(batch, dtype=np.float64)
    condition = np.full(batch, np.inf, dtype=np.float64)
    rank = np.zeros(batch, dtype=np.int64)
    unrestricted_norm = np.zeros(batch, dtype=np.float64)
    accepted_norm = np.zeros(batch, dtype=np.float64)
    trust_scale = np.ones(batch, dtype=np.float64)
    linearized_rms = np.sqrt(np.mean(np.square(residual), axis=1))
    for state in range(batch):
        left, singular, right_t = np.linalg.svd(jacobian[state], full_matrices=False)
        if len(singular) == 0 or singular[0] <= np.finfo(np.float64).tiny:
            continue
        maximum[state] = singular[0]
        minimum[state] = singular[-1]
        condition[state] = singular[0] / max(singular[-1], np.finfo(np.float64).tiny)
        retained = singular > relative_cutoff * singular[0]
        rank[state] = int(np.sum(retained))
        if not np.any(retained):
            continue
        coordinates = left[:, retained].T @ residual[state]
        damping = relative_damping * singular[0]
        gains = singular[retained] / (singular[retained] ** 2 + damping**2)
        step = -(right_t[retained].T @ (coordinates * gains))
        unrestricted_norm[state] = np.linalg.norm(step)
        trust_scale[state] = min(
            1.0,
            maximum_norm / max(unrestricted_norm[state], np.finfo(np.float64).tiny),
        )
        step *= trust_scale[state]
        steps[state] = step
        accepted_norm[state] = np.linalg.norm(step)
        projected = residual[state] + jacobian[state] @ step
        linearized_rms[state] = np.sqrt(np.mean(np.square(projected)))
    return _NewtonBatch(
        step=steps,
        maximum_singular_value=maximum,
        minimum_singular_value=minimum,
        condition_number=condition,
        effective_rank=rank,
        unrestricted_step_norm=unrestricted_norm,
        accepted_step_norm=accepted_norm,
        trust_scale=trust_scale,
        linearized_residual_rms=linearized_rms,
    )


def _truncated_minimum_norm_targets(
    residual: np.ndarray,
    jacobian: np.ndarray,
    current: np.ndarray,
    *,
    relative_cutoff: float,
    relative_damping: float,
    maximum_step_norm: float,
    maximum_control_norm: float,
) -> _NewtonBatch:
    """Compute minimum-final-KL SQP targets, with a trust bound on movement."""
    residual = np.asarray(residual, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if (
        residual.ndim != 2
        or jacobian.ndim != 3
        or jacobian.shape[:2] != residual.shape
        or current.shape != (len(residual), jacobian.shape[2])
    ):
        raise ValueError("SQP arrays must have shapes (B,M), (B,M,C), and (B,C)")
    batch = len(residual)
    steps = np.zeros_like(current)
    maximum = np.zeros(batch, dtype=np.float64)
    minimum = np.zeros(batch, dtype=np.float64)
    condition = np.full(batch, np.inf, dtype=np.float64)
    rank = np.zeros(batch, dtype=np.int64)
    unrestricted_norm = np.zeros(batch, dtype=np.float64)
    accepted_norm = np.zeros(batch, dtype=np.float64)
    trust_scale = np.ones(batch, dtype=np.float64)
    linearized_rms = np.sqrt(np.mean(np.square(residual), axis=1))
    for state in range(batch):
        left, singular, right_t = np.linalg.svd(jacobian[state], full_matrices=False)
        if len(singular) == 0 or singular[0] <= np.finfo(np.float64).tiny:
            continue
        maximum[state] = singular[0]
        minimum[state] = singular[-1]
        condition[state] = singular[0] / max(singular[-1], np.finfo(np.float64).tiny)
        retained = singular > relative_cutoff * singular[0]
        rank[state] = int(np.sum(retained))
        if not np.any(retained):
            continue
        # J y = J c - r is the linearized zero-moment condition for the
        # candidate final control y.  Solving for minimum-norm y removes any
        # accumulated null-space drift instead of minimizing only the increment.
        right_hand_side = jacobian[state] @ current[state] - residual[state]
        projected = left[:, retained].T @ right_hand_side
        damping = relative_damping * singular[0]
        gains = singular[retained] / (singular[retained] ** 2 + damping**2)
        target = right_t[retained].T @ (projected * gains)
        target_norm = np.linalg.norm(target)
        if target_norm > maximum_control_norm:
            target *= maximum_control_norm / target_norm
        unrestricted_step = target - current[state]
        unrestricted_norm[state] = np.linalg.norm(unrestricted_step)
        trust_scale[state] = min(
            1.0,
            maximum_step_norm
            / max(unrestricted_norm[state], np.finfo(np.float64).tiny),
        )
        step = trust_scale[state] * unrestricted_step
        steps[state] = step
        accepted_norm[state] = np.linalg.norm(step)
        linearized = residual[state] + jacobian[state] @ step
        linearized_rms[state] = np.sqrt(np.mean(np.square(linearized)))
    return _NewtonBatch(
        step=steps,
        maximum_singular_value=maximum,
        minimum_singular_value=minimum,
        condition_number=condition,
        effective_rank=rank,
        unrestricted_step_norm=unrestricted_norm,
        accepted_step_norm=accepted_norm,
        trust_scale=trust_scale,
        linearized_residual_rms=linearized_rms,
    )


def _newton_update(
    residual: np.ndarray,
    jacobian: np.ndarray,
    current: np.ndarray,
    *,
    mode: str,
    relative_cutoff: float,
    relative_damping: float,
    maximum_step_norm: float,
    maximum_control_norm: float,
) -> _NewtonBatch:
    """Dispatch the configured Newton semantics and enforce the control ball."""
    if mode == "minimum-final-kl":
        return _truncated_minimum_norm_targets(
            residual,
            jacobian,
            current,
            relative_cutoff=relative_cutoff,
            relative_damping=relative_damping,
            maximum_step_norm=maximum_step_norm,
            maximum_control_norm=maximum_control_norm,
        )
    if mode != "incremental":
        raise ValueError(f"unknown Newton mode: {mode}")
    result = _truncated_minimum_norm_steps(
        residual,
        jacobian,
        relative_cutoff=relative_cutoff,
        relative_damping=relative_damping,
        maximum_norm=maximum_step_norm,
    )
    current = np.asarray(current, dtype=np.float64)
    candidate = current + result.step
    candidate_norm = np.linalg.norm(candidate, axis=1)
    projection_scale = np.minimum(
        1.0,
        maximum_control_norm
        / np.maximum(candidate_norm, np.finfo(np.float64).tiny),
    )
    projected_step = projection_scale[:, None] * candidate - current
    accepted_norm = np.linalg.norm(projected_step, axis=1)
    trust_scale = np.minimum(
        1.0,
        np.divide(
            accepted_norm,
            result.unrestricted_step_norm,
            out=np.ones_like(accepted_norm),
            where=result.unrestricted_step_norm > np.finfo(np.float64).tiny,
        ),
    )
    linearized = residual + np.einsum("bmc,bc->bm", jacobian, projected_step)
    return _NewtonBatch(
        step=projected_step,
        maximum_singular_value=result.maximum_singular_value,
        minimum_singular_value=result.minimum_singular_value,
        condition_number=result.condition_number,
        effective_rank=result.effective_rank,
        unrestricted_step_norm=result.unrestricted_step_norm,
        accepted_step_norm=accepted_norm,
        trust_scale=trust_scale,
        linearized_residual_rms=np.sqrt(np.mean(np.square(linearized), axis=1)),
    )


def _compensating_spot_shift(
    state_mean_shift: np.ndarray,
    *,
    copula_loading: np.ndarray,
    orthogonal_std: float,
) -> np.ndarray:
    """Mean-shift the orthogonal spot shock so the spot score remains centered."""
    shift = np.asarray(state_mean_shift, dtype=np.float64)
    loading = np.asarray(copula_loading, dtype=np.float64)
    if shift.shape[-1] != loading.shape[0] or orthogonal_std <= 0.0:
        raise ValueError("state shift and positive copula scale are required")
    return -(shift @ loading) / orthogonal_std


def _conditional_metric_inverse_sqrt(
    copula_loading: np.ndarray,
    *,
    orthogonal_std: float,
) -> np.ndarray:
    """Map Euclidean controls to shifts under the conditional Gaussian metric."""
    loading = np.asarray(copula_loading, dtype=np.float64)
    if loading.ndim != 1 or orthogonal_std <= 0.0:
        raise ValueError("a vector loading and positive orthogonal scale are required")
    norm = float(np.linalg.norm(loading))
    result = np.eye(len(loading), dtype=np.float64)
    if norm > np.finfo(np.float64).tiny:
        direction = loading / norm
        eigenvalue = 1.0 + norm**2 / orthogonal_std**2
        result += (eigenvalue**-0.5 - 1.0) * np.outer(direction, direction)
    return result


def _bounded_spot_basis(
    gaussian_score: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    """Return bounded functions orthonormal under a standard Gaussian score."""
    if size not in (1, 2, 3):
        raise ValueError("bounded spot basis size must be one, two, or three")
    squashed = torch.tanh(gaussian_score)
    values = [torch.ones_like(squashed)]
    if size >= 2:
        values.append(squashed / _TANH_STANDARD_DEVIATION)
    if size >= 3:
        values.append(
            (squashed.square() - _TANH_SQUARE_MEAN)
            / _TANH_SQUARE_STANDARD_DEVIATION
        )
    return torch.stack(values, dim=-1)


@dataclass(frozen=True)
class _OneDayDraw:
    base_next_states: torch.Tensor
    gross_spot: torch.Tensor
    gaussian_score: torch.Tensor
    state_normals: torch.Tensor
    independent_spot_normal: torch.Tensor


class _BatchedOneDayProblem:
    def __init__(
        self,
        *,
        model: Any,
        current_states: torch.Tensor,
        daily_transition: torch.Tensor,
        daily_factor: torch.Tensor,
        copula_loading: torch.Tensor,
        copula_orthogonal_std: float,
        copula_total_std: float,
        spot_basis_size: int,
    ) -> None:
        if current_states.ndim != 2 or current_states.shape[1] != model.config.state_size:
            raise ValueError("current states do not match the model state dimension")
        if daily_factor.shape != (model.config.state_size, model.config.state_size):
            raise ValueError("the daily Gaussian factor must span every latent direction")
        self.model = model
        self.current_states = current_states
        self.daily_factor = daily_factor
        self.copula_loading = copula_loading
        self.copula_orthogonal_std = copula_orthogonal_std
        self.copula_total_std = copula_total_std
        if spot_basis_size not in (1, 2, 3):
            raise ValueError("bounded spot basis size must be one, two, or three")
        self.spot_basis_size = spot_basis_size
        metric_inverse_sqrt = _conditional_metric_inverse_sqrt(
            copula_loading.detach().cpu().double().numpy(),
            orthogonal_std=copula_orthogonal_std,
        )
        self.conditional_metric_inverse_sqrt = torch.as_tensor(
            metric_inverse_sqrt,
            dtype=current_states.dtype,
            device=current_states.device,
        )
        self.expected_states = model.equilibrium_mean.detach().unsqueeze(0) + (
            current_states - model.equilibrium_mean.detach().unsqueeze(0)
        ) @ daily_transition

        book = _nms_book(EXPIRY_DAYS, normalized_moneyness=NMS_GRID)
        self.labels = book.labels
        self.initial_days = torch.as_tensor(
            book.initial_days, dtype=current_states.dtype, device=current_states.device
        )
        self.final_days = torch.as_tensor(
            book.final_days, dtype=current_states.dtype, device=current_states.device
        )
        self.nms = torch.as_tensor(
            book.normalized_moneyness,
            dtype=current_states.dtype,
            device=current_states.device,
        )
        self.is_put = torch.as_tensor(
            book.is_put, dtype=torch.bool, device=current_states.device
        )
        with torch.no_grad():
            _, current_surface, _ = _decode_surface(model, current_states)
            self.strikes = _nms_relative_strikes(
                model,
                current_surface,
                days=self.initial_days,
                normalized_moneyness=self.nms,
            )
            current_calls = _price_calls(
                model,
                current_surface,
                days=self.initial_days,
                strikes=self.strikes,
            )
            self.current_values = torch.where(
                self.is_put.unsqueeze(0),
                current_calls - 1.0 + self.strikes,
                current_calls,
            )
            if not bool(torch.all(torch.isfinite(self.current_values))):
                raise FloatingPointError("the current option book is non-finite")
            if not bool(torch.all(self.current_values > 0.0)):
                raise FloatingPointError("the current option book has a nonpositive price")
            self.front_support = current_surface.support_strikes[:, 0]
            self.front_density = current_surface.densities[:, 0]
            self.front_backbone = current_surface.backbone_total_vols[:, 0]

    @property
    def num_states(self) -> int:
        return len(self.current_states)

    @property
    def state_size(self) -> int:
        return self.current_states.shape[1]

    @property
    def num_option_moments(self) -> int:
        return self.current_values.shape[1]

    def draw(
        self,
        num_paths: int,
        *,
        seeds: np.ndarray,
        method: str,
        pair_offset: int = 0,
    ) -> _OneDayDraw:
        """Draw state and front-spot shocks with first/second-half antithetics."""
        if num_paths <= 0 or num_paths % 2:
            raise ValueError("path count must be positive and even")
        if np.asarray(seeds).shape != (self.num_states,):
            raise ValueError("one deterministic seed is required per initial state")
        if pair_offset < 0:
            raise ValueError("Sobol pair offset must be non-negative")
        half = num_paths // 2
        state_halves: list[torch.Tensor] = []
        spot_halves: list[torch.Tensor] = []
        for seed in np.asarray(seeds, dtype=np.int64):
            if method == "pseudorandom":
                generator = torch.Generator(
                    device=self.current_states.device
                ).manual_seed(int(seed))
                state_halves.append(
                    torch.randn(
                        (half, self.state_size),
                        dtype=self.current_states.dtype,
                        device=self.current_states.device,
                        generator=generator,
                    )
                )
                spot_halves.append(
                    torch.randn(
                        half,
                        dtype=self.current_states.dtype,
                        device=self.current_states.device,
                        generator=generator,
                    )
                )
            elif method == "scrambled-sobol":
                engine = torch.quasirandom.SobolEngine(
                    dimension=self.state_size + 1,
                    scramble=True,
                    seed=int(seed % (2**31 - 1)),
                )
                if pair_offset:
                    engine.fast_forward(pair_offset)
                uniform = engine.draw(half, dtype=torch.float64)
                epsilon = torch.finfo(uniform.dtype).eps
                normal = math.sqrt(2.0) * torch.erfinv(
                    2.0 * torch.clamp(uniform, epsilon, 1.0 - epsilon) - 1.0
                )
                normal = normal.to(
                    dtype=self.current_states.dtype,
                    device=self.current_states.device,
                )
                state_halves.append(normal[:, : self.state_size])
                spot_halves.append(normal[:, self.state_size])
            else:
                raise ValueError(f"unknown Gaussian draw method: {method}")
        state_half = torch.stack(state_halves)
        spot_half = torch.stack(spot_halves)
        state_normals = torch.cat((state_half, -state_half), dim=1)
        independent_spot = torch.cat((spot_half, -spot_half), dim=1)
        gaussian_score = (
            state_normals @ self.copula_loading
            + self.copula_orthogonal_std * independent_spot
        ) / self.copula_total_std
        support_width = self.front_support.shape[1]
        gross_spot = front_week_inverse_cdf_components(
            self.front_support[:, None]
            .expand(-1, num_paths, -1)
            .reshape(-1, support_width),
            self.front_density[:, None]
            .expand(-1, num_paths, -1)
            .reshape(-1, support_width),
            self.front_backbone[:, None].expand(-1, num_paths).reshape(-1),
            gaussian_score.reshape(-1),
        ).reshape(self.num_states, num_paths)
        base_next_states = self.expected_states[:, None] + (
            state_normals @ self.daily_factor.T
        )
        return _OneDayDraw(
            base_next_states=base_next_states,
            gross_spot=gross_spot,
            gaussian_score=gaussian_score,
            state_normals=state_normals,
            independent_spot_normal=independent_spot,
        )

    def returns(
        self,
        kl_control_coefficients: torch.Tensor,
        base_next_states: torch.Tensor,
        gross_spot: torch.Tensor,
        gaussian_score: torch.Tensor,
    ) -> torch.Tensor:
        """Return spot and 48 fixed-cash options under the shifted state law."""
        expected_shape = (
            self.num_states,
            self.spot_basis_size * self.state_size,
        )
        if kl_control_coefficients.shape != expected_shape:
            raise ValueError(
                f"conditional-KL controls must have shape {expected_shape}"
            )
        if base_next_states.shape[:2] != gross_spot.shape:
            raise ValueError("next states and spot draws are not aligned")
        controls = kl_control_coefficients.reshape(
            self.num_states, self.spot_basis_size, self.state_size
        )
        coefficients = controls @ self.conditional_metric_inverse_sqrt
        basis = _bounded_spot_basis(gaussian_score, size=self.spot_basis_size)
        whitened_shift = torch.einsum("bpq,bqd->bpd", basis, coefficients)
        state_shift = whitened_shift @ self.daily_factor.T
        next_states = base_next_states + state_shift
        return self._returns_from_next_states(next_states, gross_spot)

    def _returns_from_next_states(
        self,
        next_states: torch.Tensor,
        gross_spot: torch.Tensor,
    ) -> torch.Tensor:
        flat_states = next_states.reshape(-1, self.state_size)
        _, next_surface, _ = _decode_surface(self.model, flat_states)
        normalized_strikes = (
            self.strikes[:, None] / gross_spot[:, :, None]
        ).reshape(-1, self.num_option_moments)
        next_normalized_calls = _price_calls(
            self.model,
            next_surface,
            days=self.final_days,
            strikes=normalized_strikes,
        ).reshape(self.num_states, gross_spot.shape[1], self.num_option_moments)
        next_calls = gross_spot[:, :, None] * next_normalized_calls
        next_values = torch.where(
            self.is_put[None, None],
            next_calls - gross_spot[:, :, None] + self.strikes[:, None],
            next_calls,
        )
        option_returns = next_values / self.current_values[:, None] - 1.0
        result = torch.cat((gross_spot[:, :, None] - 1.0, option_returns), dim=2)
        if not bool(torch.all(torch.isfinite(result.detach()))):
            raise FloatingPointError("one-day return calculation produced non-finite values")
        return result

    def option_mean(
        self,
        kl_control_coefficients: torch.Tensor,
        base_next_states: torch.Tensor,
        gross_spot: torch.Tensor,
        gaussian_score: torch.Tensor,
    ) -> torch.Tensor:
        return self.returns(
            kl_control_coefficients,
            base_next_states,
            gross_spot,
            gaussian_score,
        )[:, :, 1:].double().mean(dim=1)

    def stability_values(
        self,
        kl_control_coefficients: torch.Tensor,
        base_next_states: torch.Tensor,
        gaussian_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-path latent and decoded boundary diagnostics."""
        controls = kl_control_coefficients.reshape(
            self.num_states, self.spot_basis_size, self.state_size
        )
        coefficients = controls @ self.conditional_metric_inverse_sqrt
        basis = _bounded_spot_basis(gaussian_score, size=self.spot_basis_size)
        whitened_shift = torch.einsum("bpq,bqd->bpd", basis, coefficients)
        next_states = base_next_states + whitened_shift @ self.daily_factor.T
        raw = self.model.raw_states_from_states(
            next_states.reshape(-1, self.state_size)
        )
        parameters = bound_native_raw(raw, geometry=self.model.config.geometry)
        geometry = self.model.config.geometry
        dlvsqs = parameters.dlvsqs.reshape(self.num_states, -1)
        forward = parameters.forward_vols.reshape(self.num_states, -1)
        eta = parameters.eta.reshape(self.num_states, -1)
        dlv_tolerance = 1.0e-6 * (geometry.max_dlvsq - geometry.min_dlvsq)
        forward_coordinate = (
            forward - geometry.min_forward_vol
        ) / (geometry.max_forward_vol - geometry.min_forward_vol)
        eta_coordinate = (eta - geometry.min_eta) / (
            geometry.max_eta - geometry.min_eta
        )
        return {
            "latent_state": next_states,
            "raw": raw.reshape(self.num_states, -1),
            "dlv_at_cap": dlvsqs >= geometry.max_dlvsq - dlv_tolerance,
            "forward_at_boundary": (forward_coordinate <= 1.0e-3)
            | (forward_coordinate >= 1.0 - 1.0e-3),
            "eta_at_boundary": (eta_coordinate <= 1.0e-3)
            | (eta_coordinate >= 1.0 - 1.0e-3),
        }

    def score_batch_sums(
        self,
        option_returns: torch.Tensor,
        draw: _OneDayDraw,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return return, score, and return-score cross-product sums."""
        if option_returns.shape[:2] != draw.gaussian_score.shape:
            raise ValueError("option returns and Gaussian innovations are not aligned")
        basis = _bounded_spot_basis(
            draw.gaussian_score, size=self.spot_basis_size
        )
        conditional_score = draw.state_normals - (
            self.copula_loading[None, None]
            * draw.independent_spot_normal[:, :, None]
            / self.copula_orthogonal_std
        )
        control_score = conditional_score @ self.conditional_metric_inverse_sqrt.T
        score_sum = torch.einsum("bpq,bpd->bqd", basis, control_score).reshape(
            self.num_states,
            self.spot_basis_size * self.state_size,
        )
        jacobian_sum = torch.einsum(
            "bpi,bpq,bpd->biqd",
            option_returns,
            basis,
            control_score,
        ).reshape(
            self.num_states,
            self.num_option_moments,
            self.spot_basis_size * self.state_size,
        )
        return (
            torch.sum(option_returns, dim=1).cpu().double().numpy(),
            score_sum.cpu().double().numpy(),
            jacobian_sum.cpu().double().numpy(),
        )


def _state_seeds(base_seed: int, state_indices: np.ndarray, draw_index: int) -> np.ndarray:
    # The strides are coprime to 2**63 and keep every state/draw stream distinct.
    modulus = 2**63 - 1
    values = (
        int(base_seed)
        + np.asarray(state_indices, dtype=np.int64) * 1_000_003
        + int(draw_index) * 97_409
    ) % modulus
    return values.astype(np.int64, copy=False)


def _sample_initial_states(
    *,
    model: Any,
    historical_states: torch.Tensor,
    num_states: int,
    source: str,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    if source == "history":
        if num_states > len(historical_states):
            raise ValueError(
                f"history has {len(historical_states)} states, fewer than {num_states}"
            )
        indices = rng.choice(len(historical_states), size=num_states, replace=False)
        states = historical_states[indices].double().numpy()
        diagnostics: dict[str, object] = {
            "source": source,
            "historical_indices": indices.tolist(),
        }
    elif source == "invariant":
        with torch.no_grad():
            transition = model.weekly_transition().cpu().double().numpy()
            innovation = model.innovation_cholesky().cpu().double().numpy()
            mean = model.equilibrium_mean.detach().cpu().double().numpy()
        covariance = solve_discrete_lyapunov(transition.T, innovation @ innovation.T)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = 64.0 * np.finfo(np.float64).eps * max(float(eigenvalues[-1]), 1.0)
        clipped = np.maximum(eigenvalues, floor)
        factor = eigenvectors @ np.diag(np.sqrt(clipped))
        states = mean + rng.standard_normal((num_states, len(mean))) @ factor.T
        indices = np.full(num_states, -1, dtype=np.int64)
        diagnostics = {
            "source": source,
            "stationary_minimum_eigenvalue": float(eigenvalues[0]),
            "stationary_maximum_eigenvalue": float(eigenvalues[-1]),
            "stationary_clipped_eigenvalues": int(np.sum(eigenvalues < floor)),
        }
    else:
        raise ValueError(f"unknown state source: {source}")
    if not np.all(np.isfinite(states)):
        raise FloatingPointError("initial-state sampling produced non-finite states")
    return (
        torch.as_tensor(states, dtype=model.fdtype, device="cpu"),
        indices,
        diagnostics,
    )


def _autodiff_mean_and_jacobian(
    problem: _BatchedOneDayProblem,
    shifts: torch.Tensor,
    base_next_states: torch.Tensor,
    gross_spot: torch.Tensor,
    gaussian_score: torch.Tensor,
    *,
    progress: _ProgressLog,
    global_state_first: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = shifts.detach().clone().requires_grad_(True)
    mean = problem.option_mean(
        active, base_next_states, gross_spot, gaussian_score
    )
    rows: list[torch.Tensor] = []
    started = time.perf_counter()
    for moment in range(mean.shape[1]):
        rows.append(
            torch.autograd.grad(
                mean[:, moment].sum(),
                active,
                retain_graph=moment + 1 < mean.shape[1],
            )[0].detach()
        )
        if (moment + 1) % 4 == 0 or moment + 1 == mean.shape[1]:
            elapsed = time.perf_counter() - started
            rate = (moment + 1) / max(elapsed, np.finfo(float).tiny)
            progress.write(
                "jacobian",
                global_state_first=global_state_first,
                state_batch_size=problem.num_states,
                iteration=iteration,
                completed_moments=moment + 1,
                total_moments=mean.shape[1],
                eta_seconds=(mean.shape[1] - moment - 1) / rate,
                gpu_allocated_gib=(
                    torch.cuda.memory_allocated(problem.current_states.device) / 2**30
                    if problem.current_states.device.type == "cuda"
                    else 0.0
                ),
            )
    jacobian = torch.stack(rows, dim=1)
    return mean.detach().cpu().numpy(), jacobian.cpu().numpy()


def _streamed_fit_moments(
    *,
    problem: _BatchedOneDayProblem,
    controls: torch.Tensor,
    state_indices: np.ndarray,
    fit_paths: int,
    path_batch_size: int,
    fit_seed: int,
    compute_jacobian: bool,
    progress: _ProgressLog,
    stage_detail: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Stream scrambled-Sobol means and, optionally, score Jacobians."""
    mean_sum = np.zeros(
        (problem.num_states, problem.num_option_moments), dtype=np.float64
    )
    jacobian_sum = (
        np.zeros(
            (
                problem.num_states,
                problem.num_option_moments,
                problem.spot_basis_size * problem.state_size,
            ),
            dtype=np.float64,
        )
        if compute_jacobian
        else None
    )
    score_sum = (
        np.zeros(
            (
                problem.num_states,
                problem.spot_basis_size * problem.state_size,
            ),
            dtype=np.float64,
        )
        if compute_jacobian
        else None
    )
    seeds = _state_seeds(fit_seed, state_indices, 0)
    pass_started = time.perf_counter()
    for first in range(0, fit_paths, path_batch_size):
        draw = problem.draw(
            path_batch_size,
            seeds=seeds,
            method="scrambled-sobol",
            pair_offset=first // 2,
        )
        with torch.inference_mode():
            option_returns = problem.returns(
                controls,
                draw.base_next_states,
                draw.gross_spot,
                draw.gaussian_score,
            )[:, :, 1:]
            if compute_jacobian:
                (
                    batch_mean_sum,
                    batch_score_sum,
                    batch_jacobian_sum,
                ) = problem.score_batch_sums(option_returns, draw)
                assert jacobian_sum is not None and score_sum is not None
                score_sum += batch_score_sum
                jacobian_sum += batch_jacobian_sum
            else:
                batch_mean_sum = (
                    torch.sum(option_returns, dim=1).cpu().double().numpy()
                )
        mean_sum += batch_mean_sum
        completed = first + path_batch_size
        pass_elapsed = time.perf_counter() - pass_started
        rate = completed / max(pass_elapsed, np.finfo(float).tiny)
        progress.write(
            "fit-paths",
            detail=stage_detail,
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            completed_paths_per_state=completed,
            total_paths_per_state=fit_paths,
            completed_path_state_products=completed * problem.num_states,
            total_path_state_products=fit_paths * problem.num_states,
            pass_elapsed_seconds=pass_elapsed,
            pass_eta_seconds=(fit_paths - completed) / rate,
        )
    if jacobian_sum is None:
        jacobian = None
    else:
        assert score_sum is not None
        # The conditional score has exact expectation zero.  The global
        # sample-covariance form removes a constant-payoff component and is
        # unbiased for IID draws; it also materially lowers scrambled-QMC error.
        centered_cross = jacobian_sum - np.einsum(
            "bi,bc->bic", mean_sum, score_sum
        ) / fit_paths
        jacobian = centered_cross / (fit_paths - 1)
    return mean_sum / fit_paths, jacobian


def _whitened_moment_loss(
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return one mean-squared whitened-moment loss per initial state."""
    if (
        mean.shape != scale.shape
        or mean.ndim != 2
        or not bool(torch.all(torch.isfinite(mean)))
        or not bool(torch.all(torch.isfinite(scale)))
        or bool(torch.any(scale <= 0.0))
    ):
        raise ValueError("moment means and positive whitening scales must align")
    return torch.mean(torch.square(mean / scale), dim=1)


def _pathwise_moment_surrogate(
    batch_return_sum: torch.Tensor,
    *,
    total_paths: int,
    detached_full_mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Scalar whose streamed gradient equals the full fixed-sample loss gradient."""
    if (
        total_paths <= 0
        or batch_return_sum.shape != detached_full_mean.shape
        or scale.shape != detached_full_mean.shape
    ):
        raise ValueError("pathwise moment surrogate inputs are inconsistent")
    num_moments = detached_full_mean.shape[1]
    influence = 2.0 * detached_full_mean / (num_moments * scale.square())
    # Sum, rather than average, over states so each independent state's Adam
    # gradient is invariant to the operational state batch size.
    return torch.sum(influence * batch_return_sum) / total_paths


def _project_controls_(controls: torch.Tensor, *, maximum_norm: float) -> torch.Tensor:
    """Project every state's whitened control into the closed Euclidean ball."""
    if controls.ndim != 2 or maximum_norm <= 0.0:
        raise ValueError("control projection requires a matrix and positive radius")
    with torch.no_grad():
        norms = torch.linalg.vector_norm(controls, dim=1)
        scale = torch.clamp(
            maximum_norm / torch.clamp_min(norms, torch.finfo(controls.dtype).tiny),
            max=1.0,
        )
        controls.mul_(scale[:, None])
        return torch.linalg.vector_norm(controls, dim=1)


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _streamed_base_mean_sd(
    *,
    problem: _BatchedOneDayProblem,
    controls: torch.Tensor,
    state_indices: np.ndarray,
    fit_paths: int,
    path_batch_size: int,
    fit_seed: int,
    absolute_floor: float,
    relative_floor: float,
    progress: _ProgressLog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate fixed base population SDs and apply a documented robust floor.

    The scale is the population standard deviation of individual option
    returns on the fixed fit paths.  Per state it is floored at the larger of
    ``absolute_floor`` and ``relative_floor`` times the median positive SD.
    """
    sums = np.zeros(
        (problem.num_states, problem.num_option_moments), dtype=np.float64
    )
    square_sums = np.zeros_like(sums)
    seeds = _state_seeds(fit_seed, state_indices, 0)
    pass_started = time.perf_counter()
    for first in range(0, fit_paths, path_batch_size):
        draw = problem.draw(
            path_batch_size,
            seeds=seeds,
            method="scrambled-sobol",
            pair_offset=first // 2,
        )
        with torch.inference_mode():
            returns = problem.returns(
                controls,
                draw.base_next_states,
                draw.gross_spot,
                draw.gaussian_score,
            )[:, :, 1:].double()
            sums += torch.sum(returns, dim=1).cpu().numpy()
            square_sums += torch.sum(returns.square(), dim=1).cpu().numpy()
        completed = first + path_batch_size
        elapsed = time.perf_counter() - pass_started
        rate = completed / max(elapsed, np.finfo(float).tiny)
        progress.write(
            "fit-paths",
            detail="pathwise Adam base whitening",
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            completed_paths_per_state=completed,
            total_paths_per_state=fit_paths,
            pass_elapsed_seconds=elapsed,
            pass_eta_seconds=(fit_paths - completed) / rate,
        )
    mean = sums / fit_paths
    population_variance = np.maximum(square_sums / fit_paths - np.square(mean), 0.0)
    population_sd = np.sqrt(population_variance)
    positive_reference = np.empty(problem.num_states, dtype=np.float64)
    for state, state_sd in enumerate(population_sd):
        positive = state_sd[state_sd > np.finfo(np.float64).tiny]
        positive_reference[state] = (
            float(np.median(positive)) if len(positive) else absolute_floor
        )
    floor = np.maximum(absolute_floor, relative_floor * positive_reference)
    scale = np.maximum(population_sd, floor[:, None])
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(population_sd))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise FloatingPointError("base-return whitening produced an invalid scale")
    return mean, population_sd, scale, floor


def _streamed_pathwise_gradient(
    *,
    problem: _BatchedOneDayProblem,
    controls: torch.Tensor,
    state_indices: np.ndarray,
    fit_paths: int,
    path_batch_size: int,
    fit_seed: int,
    detached_full_mean: torch.Tensor,
    scale: torch.Tensor,
    control_l2: float,
    optimizer: torch.optim.Optimizer,
    progress: _ProgressLog,
    epoch: int,
) -> torch.Tensor:
    """Accumulate the exact fixed-sample pathwise gradient in bounded memory."""
    optimizer.zero_grad(set_to_none=True)
    seeds = _state_seeds(fit_seed, state_indices, 0)
    pass_started = time.perf_counter()
    for first in range(0, fit_paths, path_batch_size):
        draw = problem.draw(
            path_batch_size,
            seeds=seeds,
            method="scrambled-sobol",
            pair_offset=first // 2,
        )
        option_returns = problem.returns(
            controls,
            draw.base_next_states,
            draw.gross_spot,
            draw.gaussian_score,
        )[:, :, 1:].double()
        surrogate = _pathwise_moment_surrogate(
            torch.sum(option_returns, dim=1),
            total_paths=fit_paths,
            detached_full_mean=detached_full_mean,
            scale=scale,
        )
        if not bool(torch.isfinite(surrogate.detach())):
            raise FloatingPointError("pathwise moment surrogate is non-finite")
        surrogate.backward()
        completed = first + path_batch_size
        elapsed = time.perf_counter() - pass_started
        rate = completed / max(elapsed, np.finfo(float).tiny)
        progress.write(
            "fit-paths",
            detail=f"pathwise Adam epoch {epoch} gradient",
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            completed_paths_per_state=completed,
            total_paths_per_state=fit_paths,
            pass_elapsed_seconds=elapsed,
            pass_eta_seconds=(fit_paths - completed) / rate,
        )
    if control_l2 > 0.0:
        # Controls are expressed in conditional-Gaussian KL coordinates, so
        # 0.5 * ||c||^2 is the per-state expected Gaussian KL penalty.
        regularizer = 0.5 * control_l2 * torch.sum(controls.square())
        regularizer.backward()
    gradient = controls.grad
    if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
        raise FloatingPointError("pathwise Adam produced a non-finite gradient")
    return torch.linalg.vector_norm(gradient.detach(), dim=1)


def _fit_state_batch_score_newton(
    *,
    problem: _BatchedOneDayProblem,
    state_indices: np.ndarray,
    fit_paths: int,
    fit_path_batch_size: int,
    fit_seed: int,
    newton_iterations: int,
    newton_mode: str,
    relative_cutoff: float,
    relative_damping: float,
    maximum_step_norm: float,
    maximum_control_norm: float,
    progress: _ProgressLog,
) -> dict[str, np.ndarray]:
    shifts = torch.zeros(
        (
            problem.num_states,
            problem.spot_basis_size * problem.state_size,
        ),
        dtype=problem.current_states.dtype,
        device=problem.current_states.device,
    )
    iteration_rms = np.full(
        (problem.num_states, newton_iterations + 1), np.nan, dtype=np.float64
    )
    effective_rank = np.zeros(
        (problem.num_states, newton_iterations), dtype=np.int64
    )
    accepted_scale = np.zeros(
        (problem.num_states, newton_iterations), dtype=np.float64
    )
    accepted_step_norm = np.zeros_like(accepted_scale)
    condition_number = np.full_like(accepted_scale, np.nan)
    base_mean: np.ndarray | None = None
    corrected_mean: np.ndarray | None = None
    for iteration in range(newton_iterations):
        residual, jacobian_optional = _streamed_fit_moments(
            problem=problem,
            controls=shifts,
            state_indices=state_indices,
            fit_paths=fit_paths,
            path_batch_size=fit_path_batch_size,
            fit_seed=fit_seed,
            compute_jacobian=True,
            progress=progress,
            stage_detail=f"Newton {iteration + 1} score Jacobian",
        )
        if jacobian_optional is None:
            raise RuntimeError("score-Jacobian pass returned no Jacobian")
        jacobian = jacobian_optional
        if base_mean is None:
            base_mean = residual.copy()
        iteration_rms[:, iteration] = np.sqrt(np.mean(np.square(residual), axis=1))
        solve = _newton_update(
            residual,
            jacobian,
            shifts.cpu().double().numpy(),
            mode=newton_mode,
            relative_cutoff=relative_cutoff,
            relative_damping=relative_damping,
            maximum_step_norm=maximum_step_norm,
            maximum_control_norm=maximum_control_norm,
        )
        step = torch.as_tensor(
            solve.step, dtype=shifts.dtype, device=shifts.device
        )
        current_rms = iteration_rms[:, iteration]
        accepted_rms = current_rms.copy()
        accepted_mean = residual.copy()
        accepted_scale_values = np.zeros(problem.num_states, dtype=np.float64)
        accepted_controls = shifts.clone()
        unresolved = np.ones(problem.num_states, dtype=np.bool_)
        for scale in LINE_SEARCH_SCALES:
            candidate_controls = shifts + scale * step
            candidate_mean, _ = _streamed_fit_moments(
                problem=problem,
                controls=candidate_controls,
                state_indices=state_indices,
                fit_paths=fit_paths,
                path_batch_size=fit_path_batch_size,
                fit_seed=fit_seed,
                compute_jacobian=False,
                progress=progress,
                stage_detail=f"Newton {iteration + 1} line scale {scale:.6g}",
            )
            candidate_rms = np.sqrt(np.mean(np.square(candidate_mean), axis=1))
            improved = unresolved & (candidate_rms < current_rms)
            if np.any(improved):
                mask = torch.as_tensor(
                    improved, dtype=torch.bool, device=shifts.device
                )
                accepted_controls[mask] = candidate_controls[mask]
                accepted_rms[improved] = candidate_rms[improved]
                accepted_mean[improved] = candidate_mean[improved]
                accepted_scale_values[improved] = scale
                unresolved[improved] = False
            if not np.any(unresolved):
                break
        shifts = accepted_controls.detach()
        corrected_mean = accepted_mean
        iteration_rms[:, iteration + 1] = accepted_rms
        effective_rank[:, iteration] = solve.effective_rank
        accepted_scale[:, iteration] = accepted_scale_values
        accepted_step_norm[:, iteration] = (
            solve.accepted_step_norm * accepted_scale_values
        )
        condition_number[:, iteration] = solve.condition_number
        progress.write(
            "newton-iteration-complete",
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            iteration=iteration + 1,
            newton_mode=newton_mode,
            residual_rms_median_before=float(
                np.median(iteration_rms[:, iteration])
            ),
            residual_rms_median_after=float(np.median(accepted_rms)),
            effective_rank_minimum=int(np.min(solve.effective_rank)),
            effective_rank_median=float(np.median(solve.effective_rank)),
            accepted_scale_minimum=float(np.min(accepted_scale_values)),
            stalled_states=int(np.sum(unresolved)),
        )
    if base_mean is None or corrected_mean is None:
        raise RuntimeError("Newton fitting performed no iterations")
    return {
        "shifts": shifts.cpu().numpy(),
        "base_mean": base_mean,
        "corrected_mean": corrected_mean,
        "iteration_rms": iteration_rms,
        "effective_rank": effective_rank,
        "accepted_scale": accepted_scale,
        "accepted_step_norm": accepted_step_norm,
        "condition_number": condition_number,
    }


def _fit_state_batch_pathwise_adam(
    *,
    problem: _BatchedOneDayProblem,
    state_indices: np.ndarray,
    fit_paths: int,
    fit_path_batch_size: int,
    fit_seed: int,
    adam_epochs: int,
    adam_learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    adam_control_l2: float,
    adam_absolute_sd_floor: float,
    adam_relative_sd_floor: float,
    maximum_control_norm: float,
    progress: _ProgressLog,
    resume_state: dict[str, object] | None,
    initial_controls: torch.Tensor | None,
    checkpoint_callback: Callable[[dict[str, object]], None],
) -> dict[str, np.ndarray]:
    """Fit fixed-sample moments with streamed exact pathwise gradients.

    Each epoch uses two passes over the same scrambled-Sobol population.  The
    first detaches the complete option-return mean; the second differentiates
    its exact mean-squared, diagonally whitened moment loss one path batch at a
    time.  Whitening is fixed once from the zero-control population so scale
    changes cannot become an optimization target.
    """
    if any(parameter.requires_grad for parameter in problem.model.parameters()):
        raise RuntimeError("pathwise Adam requires every decoder parameter to be frozen")
    control_size = problem.spot_basis_size * problem.state_size
    controls = torch.nn.Parameter(
        torch.zeros(
            (problem.num_states, control_size),
            dtype=problem.current_states.dtype,
            device=problem.current_states.device,
        )
    )
    optimizer = torch.optim.Adam(
        [controls],
        lr=adam_learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_parameters) != 1 or optimizer_parameters[0] is not controls:
        raise RuntimeError("pathwise Adam must optimize the control tensor only")

    history_shape = (problem.num_states, adam_epochs + 1)
    step_shape = (problem.num_states, adam_epochs)
    iteration_rms = np.full(history_shape, np.nan, dtype=np.float64)
    fit_objective = np.full(history_shape, np.nan, dtype=np.float64)
    accepted_scale = np.zeros(step_shape, dtype=np.float64)
    accepted_step_norm = np.zeros(step_shape, dtype=np.float64)
    effective_rank = np.zeros(step_shape, dtype=np.int64)
    condition_number = np.full(step_shape, np.nan, dtype=np.float64)
    completed_epochs = 0

    def resume_array(
        name: str,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype[Any] = np.dtype(np.float64),
    ) -> np.ndarray:
        assert resume_state is not None
        value = resume_state.get(name)
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        result = np.asarray(value, dtype=dtype)
        if result.shape != shape:
            raise ValueError(
                f"pathwise Adam resume field {name} has shape {result.shape}, "
                f"expected {shape}"
            )
        return result.copy()

    if resume_state is None:
        base_mean, population_sd, whitening_scale, whitening_floor = (
            _streamed_base_mean_sd(
                problem=problem,
                controls=controls,
                state_indices=state_indices,
                fit_paths=fit_paths,
                path_batch_size=fit_path_batch_size,
                fit_seed=fit_seed,
                absolute_floor=adam_absolute_sd_floor,
                relative_floor=adam_relative_sd_floor,
                progress=progress,
            )
        )
        if initial_controls is not None:
            if initial_controls.shape != controls.shape:
                raise ValueError("warm-start controls do not match the state batch")
            candidate = initial_controls.to(
                dtype=controls.dtype, device=controls.device
            )
            if not bool(torch.all(torch.isfinite(candidate))):
                raise FloatingPointError("warm-start controls are non-finite")
            candidate_norm = torch.linalg.vector_norm(candidate, dim=1)
            tolerance = 32.0 * torch.finfo(candidate.dtype).eps * maximum_control_norm
            if bool(torch.any(candidate_norm > maximum_control_norm + tolerance)):
                raise ValueError("warm-start controls exceed the configured control norm")
            with torch.no_grad():
                controls.copy_(candidate)
            if optimizer.state:
                raise RuntimeError("warm-start Adam moments were not reset")
    else:
        if resume_state.get("schema") != "dysanos-pathwise-adam-active-v1":
            raise ValueError("pathwise Adam active checkpoint schema is incompatible")
        saved_positions = resume_state.get("state_indices")
        if isinstance(saved_positions, torch.Tensor):
            saved_positions = saved_positions.cpu().numpy()
        if not np.array_equal(np.asarray(saved_positions), state_indices):
            raise ValueError("pathwise Adam active checkpoint is for another state batch")
        saved_controls = resume_state.get("controls")
        if not isinstance(saved_controls, torch.Tensor) or saved_controls.shape != (
            problem.num_states,
            control_size,
        ):
            raise ValueError("pathwise Adam active checkpoint has invalid controls")
        with torch.no_grad():
            controls.copy_(
                saved_controls.to(dtype=controls.dtype, device=controls.device)
            )
        if not bool(torch.all(torch.isfinite(controls.detach()))):
            raise FloatingPointError("pathwise Adam resumed non-finite controls")
        optimizer_state = resume_state.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("pathwise Adam active checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
        _optimizer_to_device(optimizer, controls.device)
        completed_epochs = int(resume_state.get("completed_epochs", -1))
        if not 0 <= completed_epochs <= adam_epochs:
            raise ValueError("pathwise Adam completed epoch count is invalid")
        option_shape = (problem.num_states, problem.num_option_moments)
        base_mean = resume_array("base_mean", option_shape)
        population_sd = resume_array("population_sd", option_shape)
        whitening_scale = resume_array("whitening_scale", option_shape)
        whitening_floor = resume_array("whitening_floor", (problem.num_states,))
        iteration_rms = resume_array("iteration_rms", history_shape)
        fit_objective = resume_array("fit_objective", history_shape)
        accepted_scale = resume_array("accepted_scale", step_shape)
        accepted_step_norm = resume_array("accepted_step_norm", step_shape)

    if not np.all(np.isfinite(whitening_scale)) or np.any(whitening_scale <= 0.0):
        raise FloatingPointError("pathwise Adam whitening scale is invalid")

    scale_tensor = torch.as_tensor(
        whitening_scale,
        dtype=torch.float64,
        device=controls.device,
    )
    fit_started = time.perf_counter()

    def active_payload() -> dict[str, object]:
        return {
            "schema": "dysanos-pathwise-adam-active-v1",
            "state_indices": torch.as_tensor(state_indices, dtype=torch.int64),
            "completed_epochs": completed_epochs,
            "controls": controls.detach().cpu(),
            "optimizer_state_dict": _tree_to_cpu(optimizer.state_dict()),
            "base_mean": torch.as_tensor(base_mean, dtype=torch.float64),
            "population_sd": torch.as_tensor(population_sd, dtype=torch.float64),
            "whitening_scale": torch.as_tensor(
                whitening_scale, dtype=torch.float64
            ),
            "whitening_floor": torch.as_tensor(
                whitening_floor, dtype=torch.float64
            ),
            "iteration_rms": torch.as_tensor(iteration_rms, dtype=torch.float64),
            "fit_objective": torch.as_tensor(fit_objective, dtype=torch.float64),
            "accepted_scale": torch.as_tensor(
                accepted_scale, dtype=torch.float64
            ),
            "accepted_step_norm": torch.as_tensor(
                accepted_step_norm, dtype=torch.float64
            ),
        }

    for epoch in range(completed_epochs, adam_epochs):
        if epoch == 0 and completed_epochs == 0 and initial_controls is None:
            mean = base_mean.copy()
        else:
            mean, _ = _streamed_fit_moments(
                problem=problem,
                controls=controls,
                state_indices=state_indices,
                fit_paths=fit_paths,
                path_batch_size=fit_path_batch_size,
                fit_seed=fit_seed,
                compute_jacobian=False,
                progress=progress,
                stage_detail=f"pathwise Adam epoch {epoch + 1} mean",
            )
        if not np.all(np.isfinite(mean)):
            raise FloatingPointError("pathwise Adam moment mean is non-finite")
        iteration_rms[:, epoch] = np.sqrt(np.mean(np.square(mean), axis=1))
        controls_before = controls.detach().clone()
        control_numpy = controls_before.cpu().double().numpy()
        fit_objective[:, epoch] = np.mean(
            np.square(mean / whitening_scale), axis=1
        ) + 0.5 * adam_control_l2 * np.sum(np.square(control_numpy), axis=1)
        gradient_norm = _streamed_pathwise_gradient(
            problem=problem,
            controls=controls,
            state_indices=state_indices,
            fit_paths=fit_paths,
            path_batch_size=fit_path_batch_size,
            fit_seed=fit_seed,
            detached_full_mean=torch.as_tensor(
                mean, dtype=torch.float64, device=controls.device
            ),
            scale=scale_tensor,
            control_l2=adam_control_l2,
            optimizer=optimizer,
            progress=progress,
            epoch=epoch + 1,
        )
        optimizer.step()
        control_norm = _project_controls_(
            controls, maximum_norm=maximum_control_norm
        )
        if not bool(torch.all(torch.isfinite(controls.detach()))):
            raise FloatingPointError("pathwise Adam produced non-finite controls")
        step_norm = torch.linalg.vector_norm(
            controls.detach() - controls_before, dim=1
        )
        accepted_scale[:, epoch] = adam_learning_rate
        accepted_step_norm[:, epoch] = step_norm.cpu().double().numpy()
        completed_epochs = epoch + 1
        checkpoint_callback(active_payload())

        elapsed = time.perf_counter() - fit_started
        completed_since_resume = epoch + 1 - (
            int(resume_state.get("completed_epochs", 0))
            if resume_state is not None
            else 0
        )
        epoch_rate = completed_since_resume / max(elapsed, np.finfo(float).tiny)
        progress.write(
            "adam-epoch-complete",
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            completed_epochs=completed_epochs,
            total_epochs=adam_epochs,
            epoch_elapsed_seconds=elapsed / completed_since_resume,
            epoch_eta_seconds=(adam_epochs - completed_epochs) / epoch_rate,
            objective_median=float(np.median(fit_objective[:, epoch])),
            residual_rms_median=float(np.median(iteration_rms[:, epoch])),
            gradient_norm_median=float(torch.median(gradient_norm).cpu()),
            gradient_norm_maximum=float(torch.max(gradient_norm).cpu()),
            control_norm_maximum=float(torch.max(control_norm).cpu()),
        )

    corrected_mean, _ = _streamed_fit_moments(
        problem=problem,
        controls=controls,
        state_indices=state_indices,
        fit_paths=fit_paths,
        path_batch_size=fit_path_batch_size,
        fit_seed=fit_seed,
        compute_jacobian=False,
        progress=progress,
        stage_detail="pathwise Adam final mean",
    )
    if not np.all(np.isfinite(corrected_mean)):
        raise FloatingPointError("pathwise Adam final moment mean is non-finite")
    iteration_rms[:, adam_epochs] = np.sqrt(
        np.mean(np.square(corrected_mean), axis=1)
    )
    control_numpy = controls.detach().cpu().double().numpy()
    fit_objective[:, adam_epochs] = np.mean(
        np.square(corrected_mean / whitening_scale), axis=1
    ) + 0.5 * adam_control_l2 * np.sum(np.square(control_numpy), axis=1)
    checkpoint_callback(active_payload())
    return {
        "shifts": control_numpy,
        "base_mean": base_mean,
        "corrected_mean": corrected_mean,
        "iteration_rms": iteration_rms,
        "effective_rank": effective_rank,
        "accepted_scale": accepted_scale,
        "accepted_step_norm": accepted_step_norm,
        "condition_number": condition_number,
        "whitening_population_sd": population_sd,
        "whitening_scale": whitening_scale,
        "whitening_floor": whitening_floor,
        "fit_objective": fit_objective,
    }


@dataclass(frozen=True)
class _AuditBatch:
    mean: np.ndarray
    standard_error: np.ndarray
    z: np.ndarray
    wald_statistic: np.ndarray
    wald_degrees_of_freedom: np.ndarray
    wald_p_value: np.ndarray
    minimum_retained_correlation_eigenvalue: np.ndarray


@dataclass(frozen=True)
class _StabilityBatch:
    maximum_absolute_latent_state: np.ndarray
    latent_state_rms: np.ndarray
    maximum_absolute_native_raw: np.ndarray
    dlv_cap_fraction: np.ndarray
    forward_boundary_fraction: np.ndarray
    eta_boundary_fraction: np.ndarray


def _audit_from_moments(
    *,
    count: int,
    sums: np.ndarray,
    cross_products: np.ndarray,
    correlation_eigenvalue_cutoff: float,
) -> _AuditBatch:
    """Build coordinate and covariance-aware tests from streamed pair moments."""
    if count < 2 or sums.ndim != 2 or cross_products.shape != (
        len(sums),
        sums.shape[1],
        sums.shape[1],
    ):
        raise ValueError("streamed audit moments have invalid dimensions")
    mean = sums / count
    covariance = (
        cross_products - np.einsum("bi,bj->bij", sums, sums) / count
    ) / (count - 1)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    variance = np.maximum(
        np.diagonal(covariance, axis1=1, axis2=2), 0.0
    )
    standard_deviation = np.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(count)
    z = np.zeros_like(mean)
    positive_standard_error = standard_error > 0.0
    z[positive_standard_error] = (
        mean[positive_standard_error] / standard_error[positive_standard_error]
    )
    unresolved_nonzero = (~positive_standard_error) & (mean != 0.0)
    z[unresolved_nonzero] = np.copysign(np.inf, mean[unresolved_nonzero])
    statistic = np.zeros(len(sums), dtype=np.float64)
    degrees = np.zeros(len(sums), dtype=np.int64)
    p_value = np.ones(len(sums), dtype=np.float64)
    minimum_retained = np.full(len(sums), np.nan, dtype=np.float64)
    for state in range(len(sums)):
        active = standard_deviation[state] > max(
            1.0e-14,
            1.0e-10 * float(np.max(standard_deviation[state])),
        )
        if not np.any(active):
            continue
        scale = standard_deviation[state, active]
        correlation = covariance[state][np.ix_(active, active)] / np.outer(scale, scale)
        correlation = 0.5 * (correlation + correlation.T)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        retained = eigenvalues > correlation_eigenvalue_cutoff
        if not np.any(retained):
            continue
        normalized_mean = mean[state, active] / scale
        projected = eigenvectors[:, retained].T @ normalized_mean
        statistic[state] = count * np.sum(
            np.square(projected) / eigenvalues[retained]
        )
        degrees[state] = int(np.sum(retained))
        p_value[state] = chi2.sf(statistic[state], degrees[state])
        minimum_retained[state] = np.min(eigenvalues[retained])
    return _AuditBatch(
        mean=mean,
        standard_error=standard_error,
        z=z,
        wald_statistic=statistic,
        wald_degrees_of_freedom=degrees,
        wald_p_value=p_value,
        minimum_retained_correlation_eigenvalue=minimum_retained,
    )


def _validate_state_batch(
    *,
    problem: _BatchedOneDayProblem,
    shifts: torch.Tensor,
    state_indices: np.ndarray,
    validation_paths: int,
    path_batch_size: int,
    validation_seed: int,
    correlation_eigenvalue_cutoff: float,
    progress: _ProgressLog,
) -> tuple[_AuditBatch, _AuditBatch, _StabilityBatch]:
    num_moments = 1 + problem.num_option_moments
    base_sum = np.zeros((problem.num_states, num_moments), dtype=np.float64)
    corrected_sum = np.zeros_like(base_sum)
    base_cross = np.zeros(
        (problem.num_states, num_moments, num_moments), dtype=np.float64
    )
    corrected_cross = np.zeros_like(base_cross)
    latent_square_sum = np.zeros(problem.num_states, dtype=np.float64)
    latent_count = np.zeros(problem.num_states, dtype=np.int64)
    maximum_absolute_latent = np.zeros(problem.num_states, dtype=np.float64)
    maximum_absolute_raw = np.zeros(problem.num_states, dtype=np.float64)
    dlv_cap_count = np.zeros(problem.num_states, dtype=np.int64)
    dlv_total = np.zeros(problem.num_states, dtype=np.int64)
    forward_boundary_count = np.zeros(problem.num_states, dtype=np.int64)
    forward_total = np.zeros(problem.num_states, dtype=np.int64)
    eta_boundary_count = np.zeros(problem.num_states, dtype=np.int64)
    eta_total = np.zeros(problem.num_states, dtype=np.int64)
    pair_count = 0
    zero = torch.zeros_like(shifts)
    validation_started = time.perf_counter()
    for first in range(0, validation_paths, path_batch_size):
        draw_index = first // path_batch_size + 1
        draw = problem.draw(
            path_batch_size,
            seeds=_state_seeds(validation_seed, state_indices, draw_index),
            method="pseudorandom",
        )
        with torch.no_grad():
            base_pairs = _antithetic_pair_means(
                problem.returns(
                    zero,
                    draw.base_next_states,
                    draw.gross_spot,
                    draw.gaussian_score,
                )
            ).cpu().double().numpy()
            corrected_pairs = _antithetic_pair_means(
                problem.returns(
                    shifts,
                    draw.base_next_states,
                    draw.gross_spot,
                    draw.gaussian_score,
                )
            ).cpu().double().numpy()
            stability = problem.stability_values(
                shifts, draw.base_next_states, draw.gaussian_score
            )
            latent = stability["latent_state"]
            raw = stability["raw"]
            dlv_at_cap = stability["dlv_at_cap"]
            forward_at_boundary = stability["forward_at_boundary"]
            eta_at_boundary = stability["eta_at_boundary"]
            latent_square_sum += torch.sum(
                latent.double().square(), dim=(1, 2)
            ).cpu().numpy()
            latent_count += latent.shape[1] * latent.shape[2]
            maximum_absolute_latent = np.maximum(
                maximum_absolute_latent,
                torch.amax(torch.abs(latent), dim=(1, 2)).cpu().double().numpy(),
            )
            maximum_absolute_raw = np.maximum(
                maximum_absolute_raw,
                torch.amax(torch.abs(raw), dim=1).cpu().double().numpy(),
            )
            dlv_cap_count += torch.sum(dlv_at_cap, dim=1).cpu().numpy()
            dlv_total += dlv_at_cap.shape[1]
            forward_boundary_count += torch.sum(
                forward_at_boundary, dim=1
            ).cpu().numpy()
            forward_total += forward_at_boundary.shape[1]
            eta_boundary_count += torch.sum(eta_at_boundary, dim=1).cpu().numpy()
            eta_total += eta_at_boundary.shape[1]
        base_sum += np.sum(base_pairs, axis=1)
        corrected_sum += np.sum(corrected_pairs, axis=1)
        base_cross += np.einsum("bpi,bpj->bij", base_pairs, base_pairs)
        corrected_cross += np.einsum(
            "bpi,bpj->bij", corrected_pairs, corrected_pairs
        )
        pair_count += path_batch_size // 2
        completed = first + path_batch_size
        validation_elapsed = time.perf_counter() - validation_started
        rate = completed / max(validation_elapsed, np.finfo(float).tiny)
        progress.write(
            "validation",
            global_state_first=int(state_indices[0]),
            state_batch_size=problem.num_states,
            completed_paths_per_state=completed,
            total_paths_per_state=validation_paths,
            completed_path_state_products=completed * problem.num_states,
            total_path_state_products=validation_paths * problem.num_states,
            pass_elapsed_seconds=validation_elapsed,
            pass_eta_seconds=(validation_paths - completed) / rate,
        )
    base = _audit_from_moments(
        count=pair_count,
        sums=base_sum,
        cross_products=base_cross,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )
    corrected = _audit_from_moments(
        count=pair_count,
        sums=corrected_sum,
        cross_products=corrected_cross,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )
    stability_summary = _StabilityBatch(
        maximum_absolute_latent_state=maximum_absolute_latent,
        latent_state_rms=np.sqrt(latent_square_sum / latent_count),
        maximum_absolute_native_raw=maximum_absolute_raw,
        dlv_cap_fraction=dlv_cap_count / dlv_total,
        forward_boundary_fraction=forward_boundary_count / forward_total,
        eta_boundary_fraction=eta_boundary_count / eta_total,
    )
    return base, corrected, stability_summary


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class _WarmStart:
    controls: torch.Tensor
    initial_states: torch.Tensor
    source_indices: torch.Tensor
    provenance: dict[str, object]


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_warm_start(arguments: argparse.Namespace) -> _WarmStart | None:
    path = arguments.warm_start_checkpoint
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("warm-start checkpoint schema is incompatible")
    source_identity = payload.get("identity")
    if not isinstance(source_identity, dict):
        raise ValueError("warm-start checkpoint has no run identity")
    if source_identity.get("fit_backend") != "pathwise-adam":
        raise ValueError("warm-start source was not fitted with pathwise Adam")
    expected_identity = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "target_history": str(arguments.target_history.resolve()),
        "num_states": arguments.num_states,
        "state_source": arguments.state_source,
        "spot_basis_size": arguments.spot_basis_size,
        "copula_fit_before": arguments.copula_fit_before.isoformat(),
    }
    for name, expected in expected_identity.items():
        if source_identity.get(name) != expected:
            raise ValueError(
                f"warm-start source has incompatible {name}: "
                f"{source_identity.get(name)!r} != {expected!r}"
            )
    fit_complete = payload.get("fit_complete")
    validation_complete = payload.get("validation_complete")
    if (
        not isinstance(fit_complete, torch.Tensor)
        or not isinstance(validation_complete, torch.Tensor)
        or fit_complete.shape != (arguments.num_states,)
        or validation_complete.shape != (arguments.num_states,)
        or not bool(torch.all(fit_complete))
        or not bool(torch.all(validation_complete))
        or payload.get("active_pathwise_adam") is not None
    ):
        raise ValueError("warm-start source is not a completed checkpoint")
    controls = payload.get("kl_control_coefficients")
    initial_states = payload.get("initial_states")
    source_indices = payload.get("source_indices")
    expected_control_shape = (
        arguments.num_states,
        arguments.spot_basis_size * 200,
    )
    if (
        not isinstance(controls, torch.Tensor)
        or controls.shape != expected_control_shape
        or not controls.is_floating_point()
        or not bool(torch.all(torch.isfinite(controls)))
    ):
        raise ValueError("warm-start controls have incompatible shape or values")
    if (
        not isinstance(initial_states, torch.Tensor)
        or initial_states.shape != (arguments.num_states, 200)
        or not bool(torch.all(torch.isfinite(initial_states)))
        or not isinstance(source_indices, torch.Tensor)
        or source_indices.shape != (arguments.num_states,)
    ):
        raise ValueError("warm-start source states have incompatible geometry")
    control_norm_tensor = torch.linalg.vector_norm(controls.double(), dim=1)
    control_tolerance = (
        32.0 * torch.finfo(controls.dtype).eps * arguments.maximum_control_norm
    )
    if bool(
        torch.any(
            control_norm_tensor > arguments.maximum_control_norm + control_tolerance
        )
    ):
        raise ValueError("warm-start controls exceed the configured control norm")

    report_path = path.parent / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "a completed warm-start source must retain its report.json for book validation"
        )
    source_report = json.loads(report_path.read_text())
    if not isinstance(source_report, dict):
        raise ValueError("warm-start report is malformed")
    if source_report.get("identity") != source_identity:
        raise ValueError("warm-start report does not match its checkpoint identity")
    expected_labels = list(
        _nms_book(EXPIRY_DAYS, normalized_moneyness=NMS_GRID).labels
    )
    if source_report.get("book_labels") != expected_labels:
        raise ValueError("warm-start source uses an incompatible option book")
    fit_history = payload.get("fit_iteration_rms")
    source_epochs = (
        fit_history.shape[1] - 1
        if isinstance(fit_history, torch.Tensor) and fit_history.ndim == 2
        else None
    )
    control_norm = control_norm_tensor.numpy()
    source_identity_json = json.dumps(
        source_identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    provenance: dict[str, object] = {
        "checkpoint": str(path),
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_backend": "pathwise-adam",
        "source_epochs": source_epochs,
        "source_identity_sha256": hashlib.sha256(
            source_identity_json.encode()
        ).hexdigest(),
        "source_controls_sha256": _tensor_sha256(controls),
        "source_initial_states_sha256": _tensor_sha256(initial_states),
        "source_control_norm_quantiles": {
            str(level): float(np.quantile(control_norm, level))
            for level in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "optimizer_state_reused": False,
        "whitening_reused": False,
    }
    return _WarmStart(
        controls=controls.detach().clone(),
        initial_states=initial_states.detach().clone(),
        source_indices=source_indices.detach().clone(),
        provenance=provenance,
    )


def _validate_warm_start_states(
    warm_start: _WarmStart,
    *,
    initial_states: torch.Tensor,
    source_indices: np.ndarray,
) -> None:
    if not torch.equal(warm_start.initial_states, initial_states.cpu()):
        raise ValueError("warm-start source has different initial states")
    expected_indices = torch.as_tensor(source_indices, dtype=torch.int64)
    if not torch.equal(warm_start.source_indices.to(torch.int64), expected_indices):
        raise ValueError("warm-start source has different state-source indices")


def _identity(
    arguments: argparse.Namespace,
    *,
    warm_start_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    if (
        arguments.warm_start_checkpoint is None
    ) != (warm_start_provenance is None):
        raise ValueError(
            "warm-start checkpoint and validated provenance must be supplied together"
        )
    result: dict[str, object] = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "target_history": str(arguments.target_history.resolve()),
        "num_states": arguments.num_states,
        "state_source": arguments.state_source,
        "fit_paths_per_state": arguments.fit_paths_per_state,
        "fit_path_batch_size": arguments.fit_path_batch_size,
        "validation_paths_per_state": arguments.validation_paths_per_state,
        "state_batch_size": arguments.state_batch_size,
        "validation_path_batch_size": arguments.validation_path_batch_size,
        "fit_backend": arguments.fit_backend,
        "newton_iterations": arguments.newton_iterations,
        "newton_mode": arguments.newton_mode,
        "adam_epochs": arguments.adam_epochs,
        "adam_learning_rate": arguments.adam_learning_rate,
        "adam_beta1": arguments.adam_beta1,
        "adam_beta2": arguments.adam_beta2,
        "adam_epsilon": arguments.adam_epsilon,
        "adam_control_l2": arguments.adam_control_l2,
        "adam_absolute_sd_floor": arguments.adam_absolute_sd_floor,
        "adam_relative_sd_floor": arguments.adam_relative_sd_floor,
        "spot_basis_size": arguments.spot_basis_size,
        "relative_singular_value_cutoff": arguments.relative_singular_value_cutoff,
        "relative_damping": arguments.relative_damping,
        "maximum_step_norm": arguments.maximum_step_norm,
        "maximum_control_norm": arguments.maximum_control_norm,
        "correlation_eigenvalue_cutoff": arguments.correlation_eigenvalue_cutoff,
        "state_seed": arguments.state_seed,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "copula_fit_before": arguments.copula_fit_before.isoformat(),
    }
    if warm_start_provenance is not None:
        result["warm_start"] = {
            name: warm_start_provenance[name]
            for name in (
                "checkpoint",
                "source_identity_sha256",
                "source_controls_sha256",
                "source_initial_states_sha256",
            )
        }
    return result


def _empty_checkpoint(
    *,
    identity: dict[str, object],
    initial_states: torch.Tensor,
    source_indices: np.ndarray,
    num_iterations: int,
    spot_basis_size: int,
) -> dict[str, object]:
    num_states, state_size = initial_states.shape
    num_moments = 1 + len(EXPIRY_DAYS) * len(NMS_GRID)
    nan_moments = torch.full((num_states, num_moments), torch.nan, dtype=torch.float64)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "initial_states": initial_states.cpu(),
        "source_indices": torch.as_tensor(source_indices, dtype=torch.int64),
        "fit_complete": torch.zeros(num_states, dtype=torch.bool),
        "validation_complete": torch.zeros(num_states, dtype=torch.bool),
        "kl_control_coefficients": torch.zeros(
            (num_states, spot_basis_size * state_size), dtype=initial_states.dtype
        ),
        "fit_base_option_mean": torch.full(
            (num_states, num_moments - 1), torch.nan, dtype=torch.float64
        ),
        "fit_corrected_option_mean": torch.full(
            (num_states, num_moments - 1), torch.nan, dtype=torch.float64
        ),
        "fit_iteration_rms": torch.full(
            (num_states, num_iterations + 1), torch.nan, dtype=torch.float64
        ),
        "fit_effective_rank": torch.zeros(
            (num_states, num_iterations), dtype=torch.int64
        ),
        "fit_accepted_scale": torch.zeros(
            (num_states, num_iterations), dtype=torch.float64
        ),
        "fit_accepted_step_norm": torch.zeros(
            (num_states, num_iterations), dtype=torch.float64
        ),
        "fit_condition_number": torch.full(
            (num_states, num_iterations), torch.nan, dtype=torch.float64
        ),
        "fit_objective": torch.full(
            (num_states, num_iterations + 1), torch.nan, dtype=torch.float64
        ),
        "fit_whitening_population_sd": torch.full(
            (num_states, num_moments - 1), torch.nan, dtype=torch.float64
        ),
        "fit_whitening_scale": torch.full(
            (num_states, num_moments - 1), torch.nan, dtype=torch.float64
        ),
        "fit_whitening_floor": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "active_pathwise_adam": None,
        "base_mean": nan_moments.clone(),
        "base_standard_error": nan_moments.clone(),
        "base_z": nan_moments.clone(),
        "corrected_mean": nan_moments.clone(),
        "corrected_standard_error": nan_moments.clone(),
        "corrected_z": nan_moments.clone(),
        "base_wald_statistic": torch.full((num_states,), torch.nan, dtype=torch.float64),
        "base_wald_degrees_of_freedom": torch.zeros(num_states, dtype=torch.int64),
        "base_wald_p_value": torch.full((num_states,), torch.nan, dtype=torch.float64),
        "corrected_wald_statistic": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "corrected_wald_degrees_of_freedom": torch.zeros(
            num_states, dtype=torch.int64
        ),
        "corrected_wald_p_value": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "maximum_absolute_latent_state": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "latent_state_rms": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "maximum_absolute_native_raw": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "dlv_cap_fraction": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "forward_boundary_fraction": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
        "eta_boundary_fraction": torch.full(
            (num_states,), torch.nan, dtype=torch.float64
        ),
    }


def _put_audit(
    checkpoint: dict[str, object],
    positions: np.ndarray,
    prefix: str,
    audit: _AuditBatch,
) -> None:
    index = torch.as_tensor(positions, dtype=torch.int64)
    for name, values in (
        ("mean", audit.mean),
        ("standard_error", audit.standard_error),
        ("z", audit.z),
        ("wald_statistic", audit.wald_statistic),
        ("wald_degrees_of_freedom", audit.wald_degrees_of_freedom),
        ("wald_p_value", audit.wald_p_value),
    ):
        target = checkpoint[f"{prefix}_{name}"]
        assert isinstance(target, torch.Tensor)
        target[index] = torch.as_tensor(values, dtype=target.dtype)


def _finite_values(
    tensor: torch.Tensor,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    values = tensor.cpu().double().numpy()
    invalid = np.isnan(values) if allow_infinite else ~np.isfinite(values)
    if np.any(invalid):
        raise RuntimeError("completed checkpoint contains non-finite diagnostics")
    return values


def _json_numeric_array(values: np.ndarray) -> list[object]:
    """Encode infinities explicitly while retaining exact numeric NPZ outputs."""
    array = np.asarray(values)
    result = array.astype(object)
    result[np.isposinf(array)] = "+infinity"
    result[np.isneginf(array)] = "-infinity"
    return result.tolist()


def _json_float(value: float) -> float | str:
    if math.isinf(value):
        return "+infinity" if value > 0.0 else "-infinity"
    return float(value)


def _aggregate_audit(checkpoint: dict[str, object], prefix: str) -> dict[str, object]:
    mean_tensor = checkpoint[f"{prefix}_mean"]
    se_tensor = checkpoint[f"{prefix}_standard_error"]
    z_tensor = checkpoint[f"{prefix}_z"]
    p_tensor = checkpoint[f"{prefix}_wald_p_value"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (mean_tensor, se_tensor, z_tensor, p_tensor)
    ):
        raise RuntimeError("audit checkpoint tensors are missing")
    mean = _finite_values(mean_tensor)
    standard_error = _finite_values(se_tensor)
    z = _finite_values(z_tensor, allow_infinite=True)
    p_value = _finite_values(p_tensor)
    clipped_p = np.clip(p_value, np.finfo(np.float64).tiny, 1.0)
    fisher = -2.0 * np.sum(np.log(clipped_p))
    state_mean = np.mean(mean, axis=0)
    if len(mean) > 1:
        state_se = np.std(mean, axis=0, ddof=1) / math.sqrt(len(mean))
        state_se_definition = "sampled-initial-state standard error"
    else:
        state_se = standard_error[0]
        state_se_definition = "single-state conditional Monte Carlo standard error"
    state_z = np.zeros_like(state_mean)
    active_state_se = state_se > 0.0
    state_z[active_state_se] = state_mean[active_state_se] / state_se[active_state_se]
    unresolved_state = (~active_state_se) & (state_mean != 0.0)
    state_z[unresolved_state] = np.copysign(
        np.inf, state_mean[unresolved_state]
    )
    num_pairs = int(checkpoint["identity"]["validation_paths_per_state"] // 2)
    num_coordinates = z.size
    coordinate_threshold = float(
        student_t.ppf(1.0 - 0.05 / (2.0 * num_coordinates), num_pairs - 1)
    )
    maximum_absolute_z = float(np.max(np.abs(z)))
    state_df = len(mean) - 1 if len(mean) > 1 else num_pairs - 1
    unconditional_threshold = float(
        student_t.ppf(1.0 - 0.05 / (2.0 * mean.shape[1]), state_df)
    )
    unconditional_maximum_z = float(np.max(np.abs(state_z)))
    state_maximum_z = np.max(np.abs(z), axis=1)
    return {
        "num_states": len(mean),
        "num_moments_per_state": mean.shape[1],
        "num_independent_antithetic_pairs_per_state": num_pairs,
        "maximum_absolute_mean": float(np.max(np.abs(mean))),
        "median_state_maximum_absolute_mean": float(
            np.median(np.max(np.abs(mean), axis=1))
        ),
        "primary_global_coordinate_screen": {
            "familywise_error_rate": 0.05,
            "method": "two-sided Bonferroni Student-t",
            "num_coordinates": num_coordinates,
            "degrees_of_freedom_per_coordinate": num_pairs - 1,
            "threshold": coordinate_threshold,
            "maximum_absolute_z": _json_float(maximum_absolute_z),
            "pass": bool(maximum_absolute_z <= coordinate_threshold),
        },
        "maximum_absolute_z": _json_float(maximum_absolute_z),
        "quantiles_state_maximum_absolute_z": {
            str(level): _json_float(
                float(np.quantile(state_maximum_z, level, method="higher"))
            )
            for level in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "count_absolute_z_above_1_96": int(np.sum(np.abs(z) > 1.96)),
        "count_absolute_z_above_2_576": int(np.sum(np.abs(z) > 2.576)),
        "marginal_null_expected_count_absolute_z_above_1_96": float(
            2.0 * student_t.sf(1.96, num_pairs - 1) * z.size
        ),
        "marginal_null_expected_count_absolute_z_above_2_576": float(
            2.0 * student_t.sf(2.576, num_pairs - 1) * z.size
        ),
        "secondary_covariance_aware_screen": {
            "role": (
                "secondary: correlation truncation can discard low-variance "
                "drift directions"
            ),
            "state_wald_p_value_quantiles": {
                str(level): float(np.quantile(p_value, level))
                for level in (0.01, 0.05, 0.5, 0.95, 0.99)
            },
            "count_state_wald_p_below_0_05": int(np.sum(p_value < 0.05)),
            "fisher_state_omnibus_statistic": float(fisher),
            "fisher_state_omnibus_degrees_of_freedom": 2 * len(p_value),
            "fisher_state_omnibus_p_value": float(
                chi2.sf(fisher, 2 * len(p_value))
            ),
            "state_wald_p_value": p_value.tolist(),
        },
        "unconditional_over_sampled_states": {
            "mean": state_mean.tolist(),
            "standard_error_across_states": state_se.tolist(),
            "standard_error_definition": state_se_definition,
            "z": _json_numeric_array(state_z),
            "maximum_absolute_z": _json_float(unconditional_maximum_z),
            "bonferroni_familywise_error_rate": 0.05,
            "bonferroni_degrees_of_freedom": state_df,
            "bonferroni_threshold": unconditional_threshold,
            "bonferroni_pass": bool(
                unconditional_maximum_z <= unconditional_threshold
            ),
        },
        "coordinate_mean": mean.tolist(),
        "coordinate_standard_error": standard_error.tolist(),
        "coordinate_z": _json_numeric_array(z),
    }


def main() -> None:
    arguments = _arguments()
    _validate_arguments(arguments)
    arguments.output.mkdir(parents=True, exist_ok=True)
    progress = _ProgressLog(arguments.output)
    progress.write("resources", **_resource_status(arguments.output))
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    env, model = load_presentation_model(arguments.checkpoint, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.state_size != 200:
        raise ValueError(f"the affine pilot requires 200 states, found {model.config.state_size}")
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    historical_states = history.get("states")
    spots = history.get("spot_refs")
    dates = history.get("dates")
    if (
        not isinstance(historical_states, torch.Tensor)
        or not isinstance(spots, torch.Tensor)
        or not isinstance(dates, list)
    ):
        raise ValueError("target history is missing states, spots, or dates")
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=historical_states.double().numpy(),
        spots=spots.double().numpy(),
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    checkpoint_path = arguments.output / "checkpoint.pt"
    warm_start = _load_warm_start(arguments)
    if (
        warm_start is not None
        and Path(str(warm_start.provenance["checkpoint"])) == checkpoint_path.resolve()
    ):
        raise ValueError("warm-start source and destination checkpoint must differ")
    identity = _identity(
        arguments,
        warm_start_provenance=(
            warm_start.provenance if warm_start is not None else None
        ),
    )
    state_source_diagnostics: dict[str, object]
    if arguments.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("martingale checkpoint schema is incompatible")
        if checkpoint.get("identity") != identity:
            raise ValueError("martingale checkpoint identity is incompatible")
        saved_source = checkpoint.get("state_source_diagnostics")
        state_source_diagnostics = (
            dict(saved_source)
            if isinstance(saved_source, dict)
            else {"source": arguments.state_source}
        )
        state_source_diagnostics["resumed"] = True
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"checkpoint already exists; use --resume or a new output: {checkpoint_path}"
            )
        initial_states, source_indices, state_source_diagnostics = _sample_initial_states(
            model=model,
            historical_states=historical_states,
            num_states=arguments.num_states,
            source=arguments.state_source,
            seed=arguments.state_seed,
        )
        if warm_start is not None:
            _validate_warm_start_states(
                warm_start,
                initial_states=initial_states,
                source_indices=source_indices,
            )
        checkpoint = _empty_checkpoint(
            identity=identity,
            initial_states=initial_states,
            source_indices=source_indices,
            num_iterations=(
                arguments.newton_iterations
                if arguments.fit_backend == "score-newton"
                else arguments.adam_epochs
            ),
            spot_basis_size=arguments.spot_basis_size,
        )
        checkpoint["state_source_diagnostics"] = state_source_diagnostics
        _atomic_torch_save(checkpoint, checkpoint_path)

    initial_states = checkpoint["initial_states"]
    fit_complete = checkpoint["fit_complete"]
    validation_complete = checkpoint["validation_complete"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (initial_states, fit_complete, validation_complete)
    ):
        raise RuntimeError("martingale checkpoint is missing progress tensors")
    if warm_start is not None:
        saved_source_indices = checkpoint.get("source_indices")
        if not isinstance(saved_source_indices, torch.Tensor):
            raise RuntimeError("martingale checkpoint is missing state-source indices")
        _validate_warm_start_states(
            warm_start,
            initial_states=initial_states,
            source_indices=saved_source_indices.cpu().numpy(),
        )
    started_work = time.perf_counter()
    for first in range(0, arguments.num_states, arguments.state_batch_size):
        last = min(first + arguments.state_batch_size, arguments.num_states)
        positions = np.arange(first, last, dtype=np.int64)
        if bool(torch.all(validation_complete[first:last])):
            continue
        batch_states = initial_states[first:last].to(dtype=env.fdtype, device=env.device)
        problem = _BatchedOneDayProblem(
            model=model,
            current_states=batch_states,
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=torch.as_tensor(
                copula.state_loading, dtype=env.fdtype, device=env.device
            ),
            copula_orthogonal_std=copula.orthogonal_std,
            copula_total_std=copula.total_std,
            spot_basis_size=arguments.spot_basis_size,
        )
        if not bool(torch.all(fit_complete[first:last])):
            if bool(torch.any(fit_complete[first:last])):
                raise RuntimeError("a state batch is only partially marked fitted")
            active_adam = checkpoint.get("active_pathwise_adam")
            if arguments.fit_backend == "score-newton":
                if active_adam is not None:
                    raise RuntimeError(
                        "a pathwise Adam active checkpoint cannot resume as score Newton"
                    )
                fitted = _fit_state_batch_score_newton(
                    problem=problem,
                    state_indices=positions,
                    fit_paths=arguments.fit_paths_per_state,
                    fit_path_batch_size=arguments.fit_path_batch_size,
                    fit_seed=arguments.fit_seed,
                    newton_iterations=arguments.newton_iterations,
                    newton_mode=arguments.newton_mode,
                    relative_cutoff=arguments.relative_singular_value_cutoff,
                    relative_damping=arguments.relative_damping,
                    maximum_step_norm=arguments.maximum_step_norm,
                    maximum_control_norm=arguments.maximum_control_norm,
                    progress=progress,
                )
            else:
                if active_adam is not None and not isinstance(active_adam, dict):
                    raise RuntimeError("pathwise Adam active checkpoint is malformed")

                def save_active_adam(payload: dict[str, object]) -> None:
                    checkpoint["active_pathwise_adam"] = payload
                    _atomic_torch_save(checkpoint, checkpoint_path)

                fitted = _fit_state_batch_pathwise_adam(
                    problem=problem,
                    state_indices=positions,
                    fit_paths=arguments.fit_paths_per_state,
                    fit_path_batch_size=arguments.fit_path_batch_size,
                    fit_seed=arguments.fit_seed,
                    adam_epochs=arguments.adam_epochs,
                    adam_learning_rate=arguments.adam_learning_rate,
                    adam_beta1=arguments.adam_beta1,
                    adam_beta2=arguments.adam_beta2,
                    adam_epsilon=arguments.adam_epsilon,
                    adam_control_l2=arguments.adam_control_l2,
                    adam_absolute_sd_floor=arguments.adam_absolute_sd_floor,
                    adam_relative_sd_floor=arguments.adam_relative_sd_floor,
                    maximum_control_norm=arguments.maximum_control_norm,
                    progress=progress,
                    resume_state=active_adam,
                    initial_controls=warm_start.controls[first:last]
                    if warm_start is not None
                    else None,
                    checkpoint_callback=save_active_adam,
                )
            index = torch.as_tensor(positions, dtype=torch.int64)
            for key, source in (
                ("kl_control_coefficients", "shifts"),
                ("fit_base_option_mean", "base_mean"),
                ("fit_corrected_option_mean", "corrected_mean"),
                ("fit_iteration_rms", "iteration_rms"),
                ("fit_effective_rank", "effective_rank"),
                ("fit_accepted_scale", "accepted_scale"),
                ("fit_accepted_step_norm", "accepted_step_norm"),
                ("fit_condition_number", "condition_number"),
            ):
                target = checkpoint[key]
                assert isinstance(target, torch.Tensor)
                target[index] = torch.as_tensor(fitted[source], dtype=target.dtype)
            if arguments.fit_backend == "pathwise-adam":
                for key, source in (
                    ("fit_objective", "fit_objective"),
                    ("fit_whitening_population_sd", "whitening_population_sd"),
                    ("fit_whitening_scale", "whitening_scale"),
                    ("fit_whitening_floor", "whitening_floor"),
                ):
                    target = checkpoint[key]
                    assert isinstance(target, torch.Tensor)
                    target[index] = torch.as_tensor(
                        fitted[source], dtype=target.dtype
                    )
            fit_complete[index] = True
            checkpoint["active_pathwise_adam"] = None
            _atomic_torch_save(checkpoint, checkpoint_path)
            progress.write(
                "fit-batch-complete",
                completed_states=int(torch.sum(fit_complete)),
                total_states=arguments.num_states,
                resources=_resource_status(arguments.output),
            )

        shift_tensor = checkpoint["kl_control_coefficients"]
        assert isinstance(shift_tensor, torch.Tensor)
        shifts = shift_tensor[first:last].to(dtype=env.fdtype, device=env.device)
        base_audit, corrected_audit, stability = _validate_state_batch(
            problem=problem,
            shifts=shifts,
            state_indices=positions,
            validation_paths=arguments.validation_paths_per_state,
            path_batch_size=arguments.validation_path_batch_size,
            validation_seed=arguments.validation_seed,
            correlation_eigenvalue_cutoff=arguments.correlation_eigenvalue_cutoff,
            progress=progress,
        )
        _put_audit(checkpoint, positions, "base", base_audit)
        _put_audit(checkpoint, positions, "corrected", corrected_audit)
        index = torch.as_tensor(positions, dtype=torch.int64)
        for name in (
            "maximum_absolute_latent_state",
            "latent_state_rms",
            "maximum_absolute_native_raw",
            "dlv_cap_fraction",
            "forward_boundary_fraction",
            "eta_boundary_fraction",
        ):
            target = checkpoint[name]
            assert isinstance(target, torch.Tensor)
            target[index] = torch.as_tensor(
                getattr(stability, name), dtype=target.dtype
            )
        validation_complete[index] = True
        _atomic_torch_save(checkpoint, checkpoint_path)
        completed = int(torch.sum(validation_complete))
        elapsed = time.perf_counter() - started_work
        rate = completed / max(elapsed, np.finfo(float).tiny)
        progress.write(
            "state-batch-complete",
            completed_states=completed,
            total_states=arguments.num_states,
            eta_seconds=(arguments.num_states - completed) / rate,
            resources=_resource_status(arguments.output),
        )

    shifts = checkpoint["kl_control_coefficients"]
    fit_iteration_rms = checkpoint["fit_iteration_rms"]
    fit_rank = checkpoint["fit_effective_rank"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (shifts, fit_iteration_rms, fit_rank)
    ):
        raise RuntimeError("completed checkpoint is missing fitted drift diagnostics")
    control_numpy = _finite_values(shifts).reshape(
        arguments.num_states, arguments.spot_basis_size, model.config.state_size
    )
    metric_inverse_sqrt = _conditional_metric_inverse_sqrt(
        copula.state_loading,
        orthogonal_std=copula.orthogonal_std,
    )
    shift_numpy = control_numpy @ metric_inverse_sqrt
    compensation = _compensating_spot_shift(
        shift_numpy,
        copula_loading=copula.state_loading,
        orthogonal_std=copula.orthogonal_std,
    )
    gaussian_kl = 0.5 * np.sum(np.square(control_numpy), axis=(1, 2))
    base_report = _aggregate_audit(checkpoint, "base")
    corrected_report = _aggregate_audit(checkpoint, "corrected")
    base_z_tensor = checkpoint["base_z"]
    corrected_z_tensor = checkpoint["corrected_z"]
    if not isinstance(base_z_tensor, torch.Tensor) or not isinstance(
        corrected_z_tensor, torch.Tensor
    ):
        raise RuntimeError("completed checkpoint is missing coordinate z scores")
    np.savez_compressed(
        arguments.output / "audit_arrays.npz",
        initial_states=_finite_values(initial_states),
        kl_control_coefficients=control_numpy,
        latent_shift_coefficients=shift_numpy,
        compensating_orthogonal_spot_shift=compensation,
        base_mean=np.asarray(base_report["coordinate_mean"]),
        base_standard_error=np.asarray(base_report["coordinate_standard_error"]),
        base_z=_finite_values(base_z_tensor, allow_infinite=True),
        corrected_mean=np.asarray(corrected_report["coordinate_mean"]),
        corrected_standard_error=np.asarray(
            corrected_report["coordinate_standard_error"]
        ),
        corrected_z=_finite_values(corrected_z_tensor, allow_infinite=True),
    )
    fit_rms = _finite_values(fit_iteration_rms)
    rank = fit_rank.cpu().numpy()
    fit_report: dict[str, object] = {
        "initial_residual_rms_quantiles": {
            str(level): float(np.quantile(fit_rms[:, 0], level))
            for level in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "final_residual_rms_quantiles": {
            str(level): float(np.quantile(fit_rms[:, -1], level))
            for level in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "latent_shift_coefficient_norm_quantiles": {
            str(level): float(
                np.quantile(
                    np.linalg.norm(
                        shift_numpy.reshape(arguments.num_states, -1), axis=1
                    ),
                    level,
                )
            )
            for level in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "conditional_gaussian_kl_quantiles": {
            str(level): float(np.quantile(gaussian_kl, level))
            for level in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "stability": {
            name: {
                str(level): float(np.quantile(_finite_values(checkpoint[name]), level))
                for level in (0.0, 0.5, 0.9, 0.99, 1.0)
            }
            for name in (
                "maximum_absolute_latent_state",
                "latent_state_rms",
                "maximum_absolute_native_raw",
                "dlv_cap_fraction",
                "forward_boundary_fraction",
                "eta_boundary_fraction",
            )
        },
    }
    if arguments.fit_backend == "score-newton":
        fit_report["effective_rank_quantiles_by_iteration"] = [
            {
                str(level): float(np.quantile(rank[:, iteration], level))
                for level in (0.0, 0.5, 0.9, 1.0)
            }
            for iteration in range(rank.shape[1])
        ]
    else:
        fit_objective_tensor = checkpoint["fit_objective"]
        population_sd_tensor = checkpoint["fit_whitening_population_sd"]
        whitening_scale_tensor = checkpoint["fit_whitening_scale"]
        whitening_floor_tensor = checkpoint["fit_whitening_floor"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                fit_objective_tensor,
                population_sd_tensor,
                whitening_scale_tensor,
                whitening_floor_tensor,
            )
        ):
            raise RuntimeError("pathwise Adam checkpoint lacks whitening diagnostics")
        fit_objective = _finite_values(fit_objective_tensor)
        fit_report.update(
            {
                "objective_definition": (
                    "mean squared option-return moments divided by squared fixed, "
                    "floored base-fit population SDs, plus the configured control "
                    "L2 term"
                ),
                "initial_objective_quantiles": {
                    str(level): float(np.quantile(fit_objective[:, 0], level))
                    for level in (0.0, 0.5, 0.9, 0.99, 1.0)
                },
                "final_objective_quantiles": {
                    str(level): float(np.quantile(fit_objective[:, -1], level))
                    for level in (0.0, 0.5, 0.9, 0.99, 1.0)
                },
                "whitening": {
                    "standard_deviation_convention": (
                        "fixed population SD of individual base-fit returns"
                    ),
                    "floor_definition": (
                        "max(absolute floor, relative floor times each state's "
                        "median positive base-return SD)"
                    ),
                    "population_sd_quantiles": {
                        str(level): float(
                            np.quantile(_finite_values(population_sd_tensor), level)
                        )
                        for level in (0.0, 0.5, 0.9, 0.99, 1.0)
                    },
                    "applied_scale_quantiles": {
                        str(level): float(
                            np.quantile(_finite_values(whitening_scale_tensor), level)
                        )
                        for level in (0.0, 0.5, 0.9, 0.99, 1.0)
                    },
                    "state_floor_quantiles": {
                        str(level): float(
                            np.quantile(_finite_values(whitening_floor_tensor), level)
                        )
                        for level in (0.0, 0.5, 0.9, 0.99, 1.0)
                    },
                },
            }
        )
    report: dict[str, object] = {
        "schema": "dysanos-affine200-one-day-martingale-report-v1",
        "definition": (
            "Statewise conditional Gaussian shifts of all 200 Direct-200 "
            "latent innovations, expanded in bounded functions of the "
            "front-spot Gaussian score. Forward volatility and eta move "
            "through the affine decoder; the triangular conditional transport "
            "preserves the exact front-expiry spot marginal."
        ),
        "identity": identity,
        "warm_start": warm_start.provenance if warm_start is not None else None,
        "solver": {
            "fit_backend": arguments.fit_backend,
            "newton_mode": arguments.newton_mode,
            "minimum_final_kl": (
                arguments.fit_backend == "score-newton"
                and arguments.newton_mode == "minimum-final-kl"
            ),
            "incremental_step": (
                arguments.fit_backend == "score-newton"
                and arguments.newton_mode == "incremental"
            ),
            "pathwise_adam": (
                {
                    "epochs": arguments.adam_epochs,
                    "learning_rate": arguments.adam_learning_rate,
                    "betas": [arguments.adam_beta1, arguments.adam_beta2],
                    "epsilon": arguments.adam_epsilon,
                    "control_l2": arguments.adam_control_l2,
                }
                if arguments.fit_backend == "pathwise-adam"
                else None
            ),
            "final_control_norm_cap": arguments.maximum_control_norm,
        },
        "state_source": state_source_diagnostics,
        "transition": transition_diagnostics,
        "copula": copula_diagnostics,
        "sampling": {
            "fit": (
                "independently scrambled Sobol sequence per state, with "
                "first-half/second-half antithetic pairs; used only for fitting"
            ),
            "validation": (
                "independent pseudorandom draws with first-half/second-half "
                "antithetic pairs; never used to fit or select the correction"
            ),
            "fit_gradient": (
                "globally payoff-centered conditional Gaussian score estimator"
                if arguments.fit_backend == "score-newton"
                else (
                    "two streamed passes over the fixed fit population: a detached "
                    "full mean followed by its exact pathwise autodiff gradient"
                )
            ),
        },
        "book_labels": list(
            _nms_book(EXPIRY_DAYS, normalized_moneyness=NMS_GRID).labels
        ),
        "fit": fit_report,
        "base_P_audit": base_report,
        "corrected_Q_audit": corrected_report,
        "resources_final": _resource_status(arguments.output),
    }
    _atomic_json(report, arguments.output / "report.json")
    progress.write(
        "complete",
        report=str(arguments.output / "report.json"),
        base_maximum_absolute_z=base_report["maximum_absolute_z"],
        corrected_maximum_absolute_z=corrected_report["maximum_absolute_z"],
        corrected_fisher_p=corrected_report["secondary_covariance_aware_screen"][
            "fisher_state_omnibus_p_value"
        ],
    )


if __name__ == "__main__":
    main()
