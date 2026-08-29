"""Fit one generic conditional front-state transport across six FLEX states."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw, dlvsq_cap_barrier
from quantdata.dysanos.flex.run_all_state_vector_q_existence import (
    CommonQDraw,
    normalize_front_components,
    sample_front_components_common,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    _atomic_torch_save,
    _load_inputs,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    GenericConditionalFrontLaw,
    GenericStateQLawConfig,
    make_generic_q_draw,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    _summary as generic_state_summary,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

OBJECTIVE_ID = "six-state-joint-generic-continuous-eleven-option-moments-v1"
_SHARED_STRIKES_TRAINING_ROOT = Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_OUTPUT_ROOT = _SHARED_STRIKES_TRAINING_ROOT / "q-existence" / "generic-joint-six-v1"
_DEFAULT_STATE_INDICES = (588, 721, 80, 286, 612, 613)


@dataclass(frozen=True, kw_only=True)
class JointGenericTransportConfig:
    """Geometry of one state-and-spot conditional triangular transport."""

    front_state_size: int
    base_noise_std: float = 0.01
    maximum_shift: float = 2.0
    maximum_log_scale: float = 2.5
    maximum_lower_loading: float = 2.0
    generic_amplitude_fraction: float = 0.9
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=4,
        num_width=384,
    )

    def __post_init__(self) -> None:
        if self.front_state_size <= 0 or self.base_noise_std <= 0.0:
            raise ValueError("joint generic transport dimensions must be positive")
        if (
            self.maximum_shift <= 0.0
            or self.maximum_log_scale <= 0.0
            or self.maximum_lower_loading <= 0.0
        ):
            raise ValueError("joint generic transport bounds must be positive")
        if not 0.0 < self.generic_amplitude_fraction < 1.0:
            raise ValueError("joint generic amplitude fraction must lie in (0, 1)")

    @property
    def num_strict_lower(self) -> int:
        return self.front_state_size * (self.front_state_size - 1) // 2

    @property
    def num_outputs(self) -> int:
        return 4 * self.front_state_size + self.num_strict_lower


class JointGenericFrontTransport(BaseModel):
    r"""One full-support front-state transport conditional on current state."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: JointGenericTransportConfig,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.context_network = FeedForward(
            env,
            config.front_state_size + 1,
            config.num_outputs,
            config.network,
        )
        lower_indices = torch.tril_indices(
            config.front_state_size,
            config.front_state_size,
            offset=-1,
            device=env.device,
        )
        self.register_buffer("lower_rows", lower_indices[0])
        self.register_buffer("lower_columns", lower_indices[1])
        final = self.context_network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        *,
        current_front_states: torch.Tensor,
        spot_rank: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Transform common Gaussian noise for every supplied current state."""

        width = self.config.front_state_size
        if (
            current_front_states.ndim != 2
            or current_front_states.shape[1] != width
            or spot_rank.ndim != 2
            or spot_rank.shape[0] != len(current_front_states)
            or epsilon.shape != (spot_rank.shape[1], width)
        ):
            raise ValueError("joint generic transport inputs are inconsistent")
        states = current_front_states[:, None].expand(-1, spot_rank.shape[1], -1)
        context = torch.cat((states, spot_rank.unsqueeze(-1)), dim=-1)
        output = self.context_network(context)
        shift_raw, scale_raw, amplitude_raw, gate_raw, lower_raw = torch.split(
            output,
            (
                width,
                width,
                width,
                width,
                self.config.num_strict_lower,
            ),
            dim=-1,
        )
        shift = self.config.maximum_shift * torch.tanh(shift_raw / self.config.maximum_shift)
        log_scale = self.config.maximum_log_scale * torch.tanh(
            scale_raw / self.config.maximum_log_scale
        )
        scale = torch.exp(log_scale)
        amplitude = self.config.generic_amplitude_fraction * scale * torch.tanh(amplitude_raw)
        gate = torch.sigmoid(gate_raw)
        lower_values = self.config.maximum_lower_loading * torch.tanh(lower_raw)
        lower = torch.zeros(
            output.shape[:-1] + (width, width),
            dtype=output.dtype,
            device=output.device,
        )
        lower[..., self.lower_rows, self.lower_columns] = lower_values
        noise = epsilon[None].expand(len(current_front_states), -1, -1)
        lower_effect = torch.einsum("...ij,...j->...i", lower, noise)
        proposal = scale * noise + amplitude * torch.tanh(noise) + lower_effect
        transformed_noise = (1.0 - gate) * noise + gate * proposal
        return states + shift + self.config.base_noise_std * transformed_noise


def _teacher_checkpoint(state_index: int) -> Path:
    if state_index == 588:
        return (
            _SHARED_STRIKES_TRAINING_ROOT
            / "q-existence"
            / "vector-all-history-exact-common-random-statewise-cv-v2"
            / "investigation"
            / "state-588-generic-continuous-v1"
            / "checkpoints"
            / "state-generic-q.pt"
        )
    return (
        _SHARED_STRIKES_TRAINING_ROOT
        / "q-existence"
        / "generic-independent-five-v1"
        / f"state-{state_index}"
        / "checkpoints"
        / "state-generic-q.pt"
    )


def _load_teachers(
    *,
    env: BaseEnvironment,
    state_indices: tuple[int, ...],
    current_front_states: torch.Tensor,
) -> tuple[list[GenericConditionalFrontLaw], list[dict[str, Any]]]:
    teachers: list[GenericConditionalFrontLaw] = []
    provenance: list[dict[str, Any]] = []
    front_state_size = current_front_states.shape[1]
    for local_index, state_index in enumerate(state_indices):
        path = _teacher_checkpoint(state_index)
        payload = torch.load(path, map_location=env.device, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-state-generic-q-training-v1"
        ):
            raise ValueError(f"state {state_index} generic teacher is invalid")
        teacher = GenericConditionalFrontLaw(
            env,
            current_front_state=current_front_states[local_index],
            config=GenericStateQLawConfig(front_state_size=front_state_size),
        )
        teacher.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        teacher.requires_grad_(False)
        teacher.eval()
        teachers.append(teacher)
        provenance.append(
            {
                "state_index": state_index,
                "checkpoint": str(path),
                "checkpoint_sha256": checkpoint_digest(path),
            }
        )
    return teachers, provenance


def _teacher_outputs(
    teachers: list[GenericConditionalFrontLaw],
    *,
    spot_rank: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            teacher(spot_rank=spot_rank[index], epsilon=epsilon)
            for index, teacher in enumerate(teachers)
        ]
    )


def _distill_teachers(
    *,
    law: JointGenericFrontTransport,
    teachers: list[GenericConditionalFrontLaw],
    current_front_states: torch.Tensor,
    steps: int,
    paths_per_state: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Fit the shared map to common-latent outputs of all six teachers."""

    device = current_front_states.device
    dtype = current_front_states.dtype
    width = current_front_states.shape[1]
    calibration_generator = torch.Generator(device=device).manual_seed(seed)
    calibration_rank = torch.randn(
        (len(teachers), 4 * paths_per_state),
        dtype=dtype,
        device=device,
        generator=calibration_generator,
    )
    calibration_epsilon = torch.randn(
        (4 * paths_per_state, width),
        dtype=dtype,
        device=device,
        generator=calibration_generator,
    )
    with torch.no_grad():
        calibration_target = _teacher_outputs(
            teachers,
            spot_rank=calibration_rank,
            epsilon=calibration_epsilon,
        )
        innovation_scale = torch.sqrt(
            torch.mean(
                (calibration_target - current_front_states[:, None]).square(),
                dim=(0, 1),
            )
        ).clamp_min(1.0e-3)

    fit_generator = torch.Generator(device=device).manual_seed(seed + 1)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 2)
    validation_rank = torch.randn(
        (len(teachers), paths_per_state),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_epsilon = torch.randn(
        (paths_per_state, width),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    with torch.no_grad():
        validation_target = _teacher_outputs(
            teachers,
            spot_rank=validation_rank,
            epsilon=validation_epsilon,
        )

    optimizer = torch.optim.Adam(law.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(steps):
        spot_rank = torch.randn(
            (len(teachers), paths_per_state),
            dtype=dtype,
            device=device,
            generator=fit_generator,
        )
        epsilon = torch.randn(
            (paths_per_state, width),
            dtype=dtype,
            device=device,
            generator=fit_generator,
        )
        with torch.no_grad():
            target = _teacher_outputs(
                teachers,
                spot_rank=spot_rank,
                epsilon=epsilon,
            )
        optimizer.zero_grad(set_to_none=True)
        prediction = law(
            current_front_states=current_front_states,
            spot_rank=spot_rank,
            epsilon=epsilon,
        )
        scaled_error = (prediction - target) / innovation_scale
        loss = torch.mean(scaled_error.square())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                validation_prediction = law(
                    current_front_states=current_front_states,
                    spot_rank=validation_rank,
                    epsilon=validation_epsilon,
                )
                validation_scaled = (validation_prediction - validation_target) / innovation_scale
                validation_rmse = torch.sqrt(torch.mean(validation_scaled.square()))
                validation_maximum = torch.max(torch.abs(validation_scaled))
            record = {
                "step": float(step + 1),
                "fit_normalized_rmse": float(torch.sqrt(loss.detach())),
                "validation_normalized_rmse": float(validation_rmse),
                "validation_maximum_abs_normalized_error": float(validation_maximum),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            print(
                f"distill step={step + 1} "
                f"fit_rmse={record['fit_normalized_rmse']:.5f} "
                f"validation_rmse={record['validation_normalized_rmse']:.5f} "
                "validation_max="
                f"{record['validation_maximum_abs_normalized_error']:.4f}",
                flush=True,
            )
    return {
        "steps": steps,
        "paths_per_state_per_step": paths_per_state,
        "learning_rate": learning_rate,
        "seed": seed,
        "innovation_scale": innovation_scale.detach().cpu().tolist(),
        "history": history,
    }


def joint_roll_returns(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draw: CommonQDraw,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Value all eleven rolls at every state under one common draw."""

    gross, spot_rank = sample_front_components_common(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draw=draw,
    )
    front_state_size = law.config.front_state_size
    next_front = law(
        current_front_states=current_states[:, :front_state_size],
        spot_rank=spot_rank,
        epsilon=draw.epsilon,
    )
    num_states, paths = gross.shape
    next_state = current_states[:, None].expand(-1, paths, -1).clone()
    next_state[..., :front_state_size] = next_front
    next_strikes = normalized_strikes[None, None] / gross[..., None]
    flat_state = next_state.reshape(num_states * paths, -1)
    raw = model.raw_states_from_states(flat_state)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    next_w1_calls = model.sanos_decoder.price_first_expiry_calls_from_parameters(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
        strikes=next_strikes.reshape(num_states * paths, -1),
    ).reshape(num_states, paths, -1)
    call_returns = gross[..., None] * next_w1_calls - current_w2_calls[:, None]
    returns = torch.where(
        option_is_call[None, None],
        call_returns,
        call_returns - (gross - 1.0)[..., None],
    )
    return returns, raw, gross


def _evaluate_joint(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
) -> dict[str, torch.Tensor]:
    raw_replication_means: list[torch.Tensor] = []
    spot_replication_means: list[torch.Tensor] = []
    path_sums: list[torch.Tensor] = []
    path_square_sums: list[torch.Tensor] = []
    spot_sums: list[torch.Tensor] = []
    spot_square_sums: list[torch.Tensor] = []
    product_sums: list[torch.Tensor] = []
    cap_values: list[torch.Tensor] = []
    num_states = len(current_states)
    for draw in draws:
        returns, raw_next, gross = joint_roll_returns(
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draw=draw,
        )
        spot_return = gross - 1.0
        raw_replication_means.append(torch.mean(returns, dim=1))
        spot_replication_means.append(torch.mean(spot_return, dim=1))
        values = returns.double()
        spots = spot_return.double()
        path_sums.append(torch.sum(values, dim=1))
        path_square_sums.append(torch.sum(values.square(), dim=1))
        spot_sums.append(torch.sum(spots, dim=1))
        spot_square_sums.append(torch.sum(spots.square(), dim=1))
        product_sums.append(torch.sum(spots[..., None] * values, dim=1))
        barrier = dlvsq_cap_barrier(raw_next, geometry=model.config.geometry).reshape(
            num_states, draw.num_paths, -1
        )
        cap_values.append(torch.mean(barrier.square(), dim=(1, 2)))

    num_paths = sum(draw.num_paths for draw in draws)
    path_sum = torch.sum(torch.stack(path_sums), dim=0)
    spot_sum = torch.sum(torch.stack(spot_sums), dim=0)
    denominator = (
        torch.sum(torch.stack(spot_square_sums), dim=0) - spot_sum.square() / num_paths
    ).clamp_min(torch.finfo(torch.float64).tiny)
    numerator = (
        torch.sum(torch.stack(product_sums), dim=0) - spot_sum[:, None] * path_sum / num_paths
    )
    beta = (numerator / denominator[:, None]).detach()
    raw_rep = torch.stack(raw_replication_means)
    spot_rep = torch.stack(spot_replication_means)
    adjusted_rep = raw_rep - spot_rep[..., None] * beta[None]
    return {
        "raw_replication_means": raw_rep,
        "spot_replication_means": spot_rep,
        "adjusted_replication_means": adjusted_rep,
        "beta": beta,
        "path_sums_by_replication": torch.stack(path_sums),
        "path_square_sums_by_replication": torch.stack(path_square_sums),
        "spot_sums_by_replication": torch.stack(spot_sums),
        "spot_square_sums_by_replication": torch.stack(spot_square_sums),
        "product_sums_by_replication": torch.stack(product_sums),
        "paths_per_replication": torch.as_tensor(
            [draw.num_paths for draw in draws],
            device=raw_rep.device,
        ),
        "path_sum": path_sum,
        "path_square_sum": torch.sum(torch.stack(path_square_sums), dim=0),
        "cap_mean_square": torch.mean(torch.stack(cap_values), dim=0),
        "num_paths": torch.as_tensor(num_paths, device=raw_rep.device),
    }


def _joint_summary(
    values: Mapping[str, torch.Tensor],
    *,
    state_indices: tuple[int, ...],
    dates: tuple[str, ...],
    z_target: float,
) -> dict[str, Any]:
    per_state: list[dict[str, Any]] = []
    raw_z_values: list[float] = []
    noise_z_values: list[float] = []
    for local_index, (state_index, date) in enumerate(zip(state_indices, dates, strict=True)):
        summary = generic_state_summary(
            {
                "raw_replication_means": values["raw_replication_means"][:, local_index],
                "spot_replication_means": values["spot_replication_means"][:, local_index],
                "adjusted_replication_means": values["adjusted_replication_means"][:, local_index],
                "beta": values["beta"][local_index],
                "path_sum": values["path_sum"][local_index],
                "path_square_sum": values["path_square_sum"][local_index],
                "cap_mean_square": values["cap_mean_square"][local_index],
                "num_paths": values["num_paths"],
            },
            z_target=z_target,
        )
        summary["state_index"] = state_index
        summary["date"] = date
        per_state.append(summary)
        raw_z_values.extend(abs(value) for value in summary["raw_mean_z"])
        noise_z_values.extend(abs(value) for value in summary["noise_scaled_adjusted_mean_z"])
    raw_z = torch.tensor(raw_z_values, dtype=torch.float64)
    noise_z = torch.tensor(noise_z_values, dtype=torch.float64)
    worst_count = max(1, math.ceil(0.10 * len(noise_z)))
    worst_flat_index = int(torch.argmax(noise_z))
    return {
        "num_states": len(per_state),
        "num_moments": len(noise_z),
        "z_target": z_target,
        "maximum_abs_raw_mean_z": float(torch.max(raw_z)),
        "worst_10_percent_mean_abs_raw_mean_z": float(
            torch.mean(torch.topk(raw_z, worst_count).values)
        ),
        "maximum_abs_noise_scaled_adjusted_mean_z": float(torch.max(noise_z)),
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
            torch.mean(torch.topk(noise_z, worst_count).values)
        ),
        "worst_state_index": state_indices[worst_flat_index // 11],
        "worst_instrument_index": worst_flat_index % 11,
        "passes_target": bool(torch.max(noise_z) <= z_target),
        "per_state": per_state,
    }


def _draws(
    *,
    env: BaseEnvironment,
    front_state_size: int,
    spot_contexts: int,
    surface_paths: int,
    replications: int,
    seed: int,
) -> list[CommonQDraw]:
    return [
        make_generic_q_draw(
            front_state_size=front_state_size,
            spot_contexts=spot_contexts,
            surface_paths=surface_paths,
            seed=seed + replication,
            dtype=env.fdtype,
            device=env.device,
        )
        for replication in range(replications)
    ]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument(
        "--state-indices",
        nargs="+",
        type=int,
        default=list(_DEFAULT_STATE_INDICES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports/joint-generic-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints/joint-generic-q.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distillation-steps", type=int, default=1_000)
    parser.add_argument("--distillation-paths-per-state", type=int, default=512)
    parser.add_argument("--distillation-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--distillation-seed", type=int, default=11_483_101)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--confirmation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--fit-seed", type=int, default=11_484_101)
    parser.add_argument("--validation-seed", type=int, default=11_485_101)
    parser.add_argument("--confirmation-seed", type=int, default=11_487_101)
    parser.add_argument("--audit-every", type=int, default=10)
    parser.add_argument("--plateau-epochs", type=int, default=200)
    parser.add_argument("--validation-z-target", type=float, default=1.5)
    parser.add_argument("--confirmation-z-target", type=float, default=2.0)
    parser.add_argument("--soft-maximum-temperature", type=float, default=1.0)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> tuple[int, ...]:
    state_indices = tuple(arguments.state_indices)
    counts = (
        arguments.distillation_steps,
        arguments.distillation_paths_per_state,
        arguments.epochs,
        arguments.fit_replications,
        arguments.validation_replications,
        arguments.confirmation_replications,
        arguments.spot_contexts,
        arguments.surface_paths,
        arguments.audit_every,
        arguments.plateau_epochs,
    )
    if arguments.device != "cuda:0" or any(value <= 0 for value in counts):
        raise ValueError("joint generic Q requires GPU0 and positive counts")
    if len(state_indices) != 6 or len(set(state_indices)) != len(state_indices):
        raise ValueError("joint generic Q requires six distinct state indices")
    if min(state_indices) < 0:
        raise ValueError("joint generic state indices must be non-negative")
    if arguments.surface_paths < 4 or arguments.surface_paths % 2:
        raise ValueError("joint generic surface paths must be at least four and even")
    if (
        min(
            arguments.fit_replications,
            arguments.validation_replications,
            arguments.confirmation_replications,
        )
        < 2
    ):
        raise ValueError("joint generic MC errors require two replications")
    if (
        len(
            {
                arguments.fit_seed,
                arguments.validation_seed,
                arguments.confirmation_seed,
            }
        )
        != 3
    ):
        raise ValueError("joint generic fit and audit streams must be disjoint")
    if (
        arguments.distillation_learning_rate <= 0.0
        or arguments.learning_rate <= 0.0
        or arguments.validation_z_target <= 0.0
        or arguments.confirmation_z_target <= 0.0
        or arguments.soft_maximum_temperature <= 0.0
    ):
        raise ValueError("joint generic rates, targets and temperature must be positive")
    return state_indices


def run(arguments: argparse.Namespace) -> None:
    state_indices = _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    all_dates = train_dates + validation_dates
    all_states = torch.cat((train_states, validation_states), dim=0)
    if max(state_indices) >= len(all_states):
        raise ValueError("joint generic state index is outside history")
    index_tensor = torch.tensor(state_indices, dtype=torch.long, device=env.device)
    current_states = all_states[index_tensor].detach()
    dates = tuple(all_dates[index].isoformat() for index in state_indices)
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = JointGenericTransportConfig(front_state_size=front_state_size)
    law = JointGenericFrontTransport(env, config=config)
    teachers, teacher_provenance = _load_teachers(
        env=env,
        state_indices=state_indices,
        current_front_states=current_states[:, :front_state_size],
    )

    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    with torch.no_grad():
        raw = model.raw_states_from_states(current_states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        current_surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_calls = model.sanos_decoder.price_calls_batch(
            current_surface,
            expiries_y=model.sanos_decoder.model_expiries_y[1].expand(len(current_states), 11),
            strikes=normalized_strikes[None].expand(len(current_states), -1),
        ).detach()
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )

    fit_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.fit_replications,
        seed=arguments.fit_seed,
    )
    validation_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.validation_replications,
        seed=arguments.validation_seed,
    )
    confirmation_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.confirmation_replications,
        seed=arguments.confirmation_seed,
    )
    simulation_config = {
        "state_indices": list(state_indices),
        "dates": list(dates),
        "fit_replications": arguments.fit_replications,
        "validation_replications": arguments.validation_replications,
        "confirmation_replications": arguments.confirmation_replications,
        "spot_contexts": arguments.spot_contexts,
        "surface_paths": arguments.surface_paths,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "actual_expected_gross_spot_return": 1.0,
        "common_random_numbers_across_states": True,
    }

    distillation: dict[str, Any]
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_loss = math.inf
    stale_epochs = 0
    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=2.5e-5,
    )
    checkpoint: dict[str, Any] | None = None
    if arguments.checkpoint.exists():
        payload = torch.load(arguments.checkpoint, map_location=env.device, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-six-state-joint-generic-q-training-v1"
            or payload.get("source_completion_sha256") != sources["completion_sha256"]
            or payload.get("law_config") != asdict(config)
            or payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("joint generic Q checkpoint identity is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state_dict"]))
        scheduler.load_state_dict(cast(dict[str, Any], payload["scheduler_state_dict"]))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = arguments.learning_rate
        distillation = dict(cast(Mapping[str, Any], payload["distillation"]))
        history = list(cast(list[dict[str, float]], payload["history"]))
        start_epoch = len(history)
        best_loss = float(payload["best_loss"])
        stale_epochs = int(payload["stale_epochs"])
        checkpoint = dict(payload)
        print(f"resuming six-state joint generic Q at epoch {start_epoch}", flush=True)
    else:
        distillation = _distill_teachers(
            law=law,
            teachers=teachers,
            current_front_states=current_states[:, :front_state_size],
            steps=arguments.distillation_steps,
            paths_per_state=arguments.distillation_paths_per_state,
            learning_rate=arguments.distillation_learning_rate,
            seed=arguments.distillation_seed,
        )
    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    validation_summary: dict[str, Any] | None = None
    stopped_for_validation = False
    stopped_for_plateau = False
    for epoch in range(start_epoch, arguments.epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fit_values = _evaluate_joint(
                model=model,
                law=law,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=fit_draws,
            )
            fit_summary = _joint_summary(
                fit_values,
                state_indices=state_indices,
                dates=dates,
                z_target=arguments.validation_z_target,
            )
            adjusted_rep = fit_values["adjusted_replication_means"].double()
            raw_rep = fit_values["raw_replication_means"].double()
            spot_rep = fit_values["spot_replication_means"].double()
            adjusted_mean = torch.mean(adjusted_rep, dim=0)
            raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(len(fit_draws))
            spot_se = torch.std(spot_rep, dim=0, correction=1) / math.sqrt(len(fit_draws))
            beta = fit_values["beta"].double()
            noise_scale = torch.maximum(torch.abs(beta) * spot_se[:, None], raw_se)
            scaled = adjusted_mean / noise_scale.clamp_min(1.0e-12)
            squared = scaled.square()
            temperature = arguments.soft_maximum_temperature
            loss = temperature * (
                torch.logsumexp(squared.reshape(-1) / temperature, dim=0)
                - math.log(squared.numel())
            )
            weights = torch.softmax(squared.reshape(-1) / temperature, dim=0).reshape_as(squared)
            mean_gradient = (2.0 * scaled * weights / noise_scale.clamp_min(1.0e-12)).to(
                dtype=env.fdtype
            )

        for draw in fit_draws:
            returns, _, gross = joint_roll_returns(
                model=model,
                law=law,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draw=draw,
            )
            spot_mean = torch.mean(gross - 1.0, dim=1)
            adjusted_draw_mean = (
                torch.mean(returns, dim=1)
                - fit_values["beta"].to(dtype=env.fdtype) * spot_mean[:, None]
            )
            surrogate = torch.sum(mean_gradient * adjusted_draw_mean) / len(fit_draws)
            surrogate.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        loss_value = float(loss)
        scheduler.step(loss_value)
        if loss_value < best_loss * (1.0 - 1.0e-5):
            best_loss = loss_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_value,
            "fit_maximum_abs_raw_mean_z": float(fit_summary["maximum_abs_raw_mean_z"]),
            "fit_worst_10_percent_mean_abs_raw_mean_z": float(
                fit_summary["worst_10_percent_mean_abs_raw_mean_z"]
            ),
            "fit_maximum_abs_noise_scaled_adjusted_mean_z": float(
                fit_summary["maximum_abs_noise_scaled_adjusted_mean_z"]
            ),
            "fit_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
                fit_summary["worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"]
            ),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise FloatingPointError("non-finite joint generic training record")
        history.append(record)
        checkpoint = {
            "schema": "flex-six-state-joint-generic-q-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "law_config": asdict(config),
            "simulation_config": simulation_config,
            "teacher_provenance": teacher_provenance,
            "distillation": distillation,
            "law_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
        }
        _atomic_torch_save(checkpoint, arguments.checkpoint)
        print(
            f"epoch={epoch + 1} loss={loss_value:.6g} "
            f"fit_raw_worst={record['fit_maximum_abs_raw_mean_z']:.3f} "
            "fit_raw_worst10="
            f"{record['fit_worst_10_percent_mean_abs_raw_mean_z']:.3f} "
            "fit_adjusted_worst="
            f"{record['fit_maximum_abs_noise_scaled_adjusted_mean_z']:.3f} "
            "fit_adjusted_worst10="
            f"{record['fit_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
            f"lr={record['learning_rate']:.6g}",
            flush=True,
        )
        if (epoch + 1) % arguments.audit_every == 0:
            with torch.no_grad():
                validation_summary = _joint_summary(
                    _evaluate_joint(
                        model=model,
                        law=law,
                        current_states=current_states,
                        current_w2_calls=current_w2_calls,
                        normalized_strikes=normalized_strikes,
                        option_is_call=option_is_call,
                        support=support,
                        density=density,
                        sqrt_variance=sqrt_variance,
                        draws=validation_draws,
                    ),
                    state_indices=state_indices,
                    dates=dates,
                    z_target=arguments.validation_z_target,
                )
            print(
                "validation "
                f"raw_worst={validation_summary['maximum_abs_raw_mean_z']:.3f} "
                "raw_worst10="
                f"{validation_summary['worst_10_percent_mean_abs_raw_mean_z']:.3f} "
                "adjusted_worst="
                f"{validation_summary['maximum_abs_noise_scaled_adjusted_mean_z']:.3f} "
                "adjusted_worst10="
                f"{validation_summary['worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
                f"worst_state={validation_summary['worst_state_index']} "
                f"target={arguments.validation_z_target:.3f}",
                flush=True,
            )
            if bool(validation_summary["passes_target"]):
                stopped_for_validation = True
                break
        if stale_epochs >= arguments.plateau_epochs:
            stopped_for_plateau = True
            break

    if checkpoint is None:
        raise RuntimeError("joint generic Q produced no checkpoint")
    if validation_summary is None or not stopped_for_validation:
        with torch.no_grad():
            validation_summary = _joint_summary(
                _evaluate_joint(
                    model=model,
                    law=law,
                    current_states=current_states,
                    current_w2_calls=current_w2_calls,
                    normalized_strikes=normalized_strikes,
                    option_is_call=option_is_call,
                    support=support,
                    density=density,
                    sqrt_variance=sqrt_variance,
                    draws=validation_draws,
                ),
                state_indices=state_indices,
                dates=dates,
                z_target=arguments.validation_z_target,
            )
    with torch.no_grad():
        confirmation_summary = _joint_summary(
            _evaluate_joint(
                model=model,
                law=law,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=confirmation_draws,
            ),
            state_indices=state_indices,
            dates=dates,
            z_target=arguments.confirmation_z_target,
        )
    semantic_success = bool(
        validation_summary["passes_target"] and confirmation_summary["passes_target"]
    )
    report = {
        "schema": "flex-six-state-joint-generic-q-report-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "objective": OBJECTIVE_ID,
        "state_indices": list(state_indices),
        "dates": list(dates),
        "law": {
            "form": "one triangular transport T_theta(h1_0,spot_rank,epsilon)",
            "full_support": True,
            "discrete_branch": False,
            "num_parameters": sum(parameter.numel() for parameter in law.parameters()),
        },
        "teacher_provenance": teacher_provenance,
        "distillation": distillation,
        "simulation_config": simulation_config,
        "training": {
            "epochs_completed": len(history),
            "best_loss": best_loss,
            "stopped_for_validation_target": stopped_for_validation,
            "stopped_for_plateau": stopped_for_plateau,
            "history": history,
        },
        "independent_validation": validation_summary,
        "independent_confirmation": confirmation_summary,
        "sources": sources,
    }
    write_json_atomic(report, arguments.output)
    final_checkpoint = {
        **checkpoint,
        "status": "complete" if semantic_success else "failed_terminal",
        "report": str(arguments.output),
        "report_sha256": checkpoint_digest(arguments.output),
    }
    _atomic_torch_save(final_checkpoint, arguments.checkpoint)
    completion = arguments.output.with_name(f"{arguments.output.stem}-complete.json")
    write_json_atomic(
        {
            "schema": "flex-six-state-joint-generic-q-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        },
        completion,
    )
    print(
        json.dumps(
            {
                "independent_validation": validation_summary,
                "independent_confirmation": confirmation_summary,
            },
            indent=2,
        ),
        flush=True,
    )
    if not semantic_success:
        raise RuntimeError("six-state joint generic Q failed an audit target")


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_six_state_joint_generic_q_existence",
        clean_sigterm=True,
    )
