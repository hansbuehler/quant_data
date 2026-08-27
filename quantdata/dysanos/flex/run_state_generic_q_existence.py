"""Fit a generic conditional front-state law at one historical FLEX state."""

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
    make_common_q_draw,
    normalize_front_components,
    sample_front_components_common,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    _atomic_torch_save,
    _load_inputs,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

OBJECTIVE_ID = "single-state-generic-continuous-eleven-option-moments-v1"
_SHARED_STRIKES_TRAINING_ROOT = (
    Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
)
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-all-history-exact-common-random-statewise-cv-v2"
    / "investigation"
    / "state-588-generic-continuous-v1"
)
_DEFAULT_TEACHER = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-all-history-exact-common-random-statewise-cv-v2"
    / "snapshots"
    / "all-state-vector-q-epoch351.pt"
)


@dataclass(frozen=True, kw_only=True)
class GenericStateQLawConfig:
    """Geometry of a spot-conditional, full-support front-state transport."""

    front_state_size: int
    initial_state_noise_std: float = 0.01
    minimum_state_noise_std: float = 1.0e-4
    maximum_network_shift: float = 2.0
    maximum_log_scale: float = 2.0
    maximum_lower_loading: float = 2.0
    generic_amplitude_fraction: float = 0.9
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=3,
        num_width=256,
    )

    def __post_init__(self) -> None:
        if self.front_state_size <= 0:
            raise ValueError("front state size must be positive")
        if not 0.0 < self.minimum_state_noise_std < self.initial_state_noise_std:
            raise ValueError("generic state noise bounds are invalid")
        if (
            self.maximum_network_shift <= 0.0
            or self.maximum_log_scale <= 0.0
            or self.maximum_lower_loading <= 0.0
        ):
            raise ValueError("generic transport bounds must be positive")
        if not 0.0 < self.generic_amplitude_fraction < 1.0:
            raise ValueError("generic amplitude fraction must lie in (0, 1)")


class GenericConditionalFrontLaw(BaseModel):
    r"""A continuous full-support law for ``h_t^1 | (h_0^1, S_t)``.

    Conditional on spot, each Gaussian coordinate is transformed by a
    triangular map whose diagonal derivative is strictly positive. This
    retains full support without imposing an artificial discrete branch.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_state: torch.Tensor,
        config: GenericStateQLawConfig,
    ) -> None:
        super().__init__(env)
        if current_front_state.shape != (config.front_state_size,):
            raise ValueError("generic law received an invalid current front state")
        self.config = config
        self.register_buffer(
            "current_front_state", current_front_state.detach().clone()
        )
        self.location_shift = torch.nn.Parameter(
            torch.zeros_like(current_front_state)
        )
        self.spot_loading = torch.nn.Parameter(
            torch.zeros_like(current_front_state)
        )
        initial_excess_std = (
            config.initial_state_noise_std - config.minimum_state_noise_std
        )
        self.raw_noise_std = torch.nn.Parameter(
            torch.as_tensor(
                math.log(math.expm1(initial_excess_std)),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.context_network = FeedForward(
            env,
            config.front_state_size + 1,
            4 * config.front_state_size,
            config.network,
        )
        self.strict_lower_raw = torch.nn.Parameter(
            torch.zeros(
                (config.front_state_size, config.front_state_size),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        final = (
            self.context_network.I0
            if len(self.context_network.In) == 0
            else self.context_network.In[-1]
        )
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    @property
    def noise_std(self) -> torch.Tensor:
        return self.config.minimum_state_noise_std + torch.nn.functional.softplus(
            self.raw_noise_std
        )

    def initialize_affine_teacher(
        self,
        *,
        location_shift: torch.Tensor,
        spot_loading: torch.Tensor,
        raw_noise_std: torch.Tensor,
    ) -> None:
        """Initialize the affine part from the preserved statewise law."""

        expected = (self.config.front_state_size,)
        if location_shift.shape != expected or spot_loading.shape != expected:
            raise ValueError("teacher front-state vectors have incompatible shapes")
        if raw_noise_std.numel() != 1:
            raise ValueError("teacher state noise must be scalar")
        with torch.no_grad():
            self.location_shift.copy_(location_shift)
            self.spot_loading.copy_(spot_loading)
            self.raw_noise_std.copy_(raw_noise_std.reshape(()))

    def forward(
        self,
        *,
        spot_rank: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Transform independent Gaussian noise conditional on the spot rank."""

        width = self.config.front_state_size
        if spot_rank.ndim != 1 or epsilon.shape != (len(spot_rank), width):
            raise ValueError("generic conditional-law inputs are inconsistent")
        current = self.current_front_state.expand(len(spot_rank), -1)
        context = torch.cat((current, spot_rank.unsqueeze(-1)), dim=-1)
        shift_raw, scale_raw, amplitude_raw, gate_raw = torch.split(
            self.context_network(context), width, dim=-1
        )
        shift = self.config.maximum_network_shift * torch.tanh(
            shift_raw / self.config.maximum_network_shift
        )
        log_scale = self.config.maximum_log_scale * torch.tanh(
            scale_raw / self.config.maximum_log_scale
        )
        scale = torch.exp(log_scale)
        amplitude = (
            self.config.generic_amplitude_fraction
            * scale
            * torch.tanh(amplitude_raw)
        )
        gate = torch.sigmoid(gate_raw)
        lower = self.config.maximum_lower_loading * torch.tanh(
            torch.tril(self.strict_lower_raw, diagonal=-1)
        )
        proposal = (
            scale * epsilon
            + amplitude * torch.tanh(epsilon)
            + epsilon @ lower.T
        )
        transformed_noise = (1.0 - gate) * epsilon + gate * proposal
        return (
            current
            + self.location_shift
            + spot_rank.unsqueeze(-1) * self.spot_loading
            + shift
            + self.noise_std * transformed_noise
        )


def make_generic_q_draw(
    *,
    front_state_size: int,
    spot_contexts: int,
    surface_paths: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CommonQDraw:
    """Construct a common spot draw with non-duplicated antithetic latent noise."""

    if surface_paths < 2 or surface_paths % 2:
        raise ValueError("generic surface paths must be positive and even")
    spot_draw = make_common_q_draw(
        front_state_size=front_state_size,
        spot_contexts=spot_contexts,
        surface_paths=4,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed + 10_000_000)
    half = torch.randn(
        (spot_contexts, surface_paths // 2, front_state_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    epsilon = torch.cat((half, -half), dim=1).reshape(
        spot_contexts * surface_paths, front_state_size
    )
    branch = torch.zeros(
        spot_contexts * surface_paths, dtype=dtype, device=device
    )
    return CommonQDraw(
        component_uniform=spot_draw.component_uniform,
        backbone_uniform=spot_draw.backbone_uniform,
        branch=branch,
        epsilon=epsilon,
        spot_contexts=spot_contexts,
        surface_paths=surface_paths,
    )


def generic_roll_returns(
    *,
    model: Any,
    law: GenericConditionalFrontLaw,
    current_state: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draw: CommonQDraw,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Value the eleven one-week rolls under one generic-law draw."""

    gross_batch, spot_rank_batch = sample_front_components_common(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draw=draw,
    )
    gross = gross_batch[0]
    next_front = law(spot_rank=spot_rank_batch[0], epsilon=draw.epsilon)
    next_state = current_state.expand(draw.num_paths, -1).clone()
    next_state[:, : law.config.front_state_size] = next_front
    next_strikes = normalized_strikes[None] / gross[:, None]
    raw = model.raw_states_from_states(next_state)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    next_w1_calls = model.sanos_decoder.price_first_expiry_calls_from_parameters(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
        strikes=next_strikes,
    )
    call_returns = gross[:, None] * next_w1_calls - current_w2_calls
    returns = torch.where(
        option_is_call[None],
        call_returns,
        call_returns - (gross - 1.0)[:, None],
    )
    return returns, raw, gross


def _evaluate(
    *,
    model: Any,
    law: GenericConditionalFrontLaw,
    current_state: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
) -> dict[str, torch.Tensor]:
    """Evaluate a fixed dataset and refit its pooled control slopes."""

    raw_replication_means: list[torch.Tensor] = []
    spot_replication_means: list[torch.Tensor] = []
    path_sums: list[torch.Tensor] = []
    path_square_sums: list[torch.Tensor] = []
    spot_sums: list[torch.Tensor] = []
    spot_square_sums: list[torch.Tensor] = []
    product_sums: list[torch.Tensor] = []
    cap_values: list[torch.Tensor] = []
    for draw in draws:
        returns, raw_next, gross = generic_roll_returns(
            model=model,
            law=law,
            current_state=current_state,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draw=draw,
        )
        spot_return = gross - 1.0
        raw_replication_means.append(torch.mean(returns, dim=0))
        spot_replication_means.append(torch.mean(spot_return))
        values = returns.double()
        spot_values = spot_return.double()
        path_sums.append(torch.sum(values, dim=0))
        path_square_sums.append(torch.sum(values.square(), dim=0))
        spot_sums.append(torch.sum(spot_values))
        spot_square_sums.append(torch.sum(spot_values.square()))
        product_sums.append(torch.sum(spot_values[:, None] * values, dim=0))
        barrier = dlvsq_cap_barrier(raw_next, geometry=model.config.geometry)
        cap_values.append(torch.mean(barrier.square()))

    num_paths = sum(draw.num_paths for draw in draws)
    path_sum = torch.sum(torch.stack(path_sums), dim=0)
    spot_sum = torch.sum(torch.stack(spot_sums))
    denominator = (
        torch.sum(torch.stack(spot_square_sums)) - spot_sum.square() / num_paths
    ).clamp_min(torch.finfo(torch.float64).tiny)
    numerator = (
        torch.sum(torch.stack(product_sums), dim=0)
        - spot_sum * path_sum / num_paths
    )
    beta = (numerator / denominator).detach()
    raw_rep = torch.stack(raw_replication_means)
    spot_rep = torch.stack(spot_replication_means)
    # The exact normalized front marginal has E[S_t/S_0 - 1] = 0.
    adjusted_rep = raw_rep - spot_rep[:, None] * beta[None]
    return {
        "raw_replication_means": raw_rep,
        "spot_replication_means": spot_rep,
        "adjusted_replication_means": adjusted_rep,
        "beta": beta,
        "path_sum": path_sum,
        "path_square_sum": torch.sum(torch.stack(path_square_sums), dim=0),
        "spot_sum": spot_sum,
        "spot_square_sum": torch.sum(torch.stack(spot_square_sums)),
        "cap_mean_square": torch.mean(torch.stack(cap_values)),
        "num_paths": torch.as_tensor(num_paths, device=raw_rep.device),
    }


def _summary(
    values: Mapping[str, torch.Tensor],
    *,
    z_target: float = 2.0,
) -> dict[str, Any]:
    """Summarize one independently sampled dataset in spot-normalized units."""

    if not math.isfinite(z_target) or z_target <= 0.0:
        raise ValueError("summary z target must be finite-positive")
    raw_rep = values["raw_replication_means"].double()
    spot_rep = values["spot_replication_means"].double()
    adjusted_rep = values["adjusted_replication_means"].double()
    beta = values["beta"].double()
    replications = len(raw_rep)
    raw_mean = torch.mean(raw_rep, dim=0)
    adjusted_mean = torch.mean(adjusted_rep, dim=0)
    raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(replications)
    adjusted_se = torch.std(adjusted_rep, dim=0, correction=1) / math.sqrt(
        replications
    )
    spot_mean = torch.mean(spot_rep)
    spot_se = torch.std(spot_rep, correction=1) / math.sqrt(replications)
    noise_scale = torch.maximum(torch.abs(beta) * spot_se, raw_se)
    raw_z = raw_mean / raw_se.clamp_min(1.0e-12)
    noise_z = adjusted_mean / noise_scale.clamp_min(1.0e-12)
    absolute_raw_z = torch.abs(raw_z)
    absolute_noise_z = torch.abs(noise_z)
    worst_count = max(1, math.ceil(0.10 * len(raw_mean)))
    worst_index = int(torch.argmax(absolute_noise_z))
    num_paths = int(values["num_paths"])
    path_mean = values["path_sum"].double() / num_paths
    path_variance = (
        values["path_square_sum"].double() - num_paths * path_mean.square()
    ) / (num_paths - 1)
    return {
        "raw_mean_bps_spot": (1.0e4 * raw_mean).tolist(),
        "adjusted_mean_bps_spot": (1.0e4 * adjusted_mean).tolist(),
        "raw_mean_mc_standard_error_bps_spot": (1.0e4 * raw_se).tolist(),
        "adjusted_mean_mc_standard_error_bps_spot": (
            1.0e4 * adjusted_se
        ).tolist(),
        "control_variate_noise_scale_bps_spot": (
            1.0e4 * noise_scale
        ).tolist(),
        "spot_control_beta": beta.tolist(),
        "spot_return_sample_mean_bps": float(1.0e4 * spot_mean),
        "spot_return_mc_standard_error_bps": float(1.0e4 * spot_se),
        "return_sample_std_bps_spot": (
            1.0e4 * torch.sqrt(torch.clamp_min(path_variance, 0.0))
        ).tolist(),
        "raw_mean_z": raw_z.tolist(),
        "noise_scaled_adjusted_mean_z": noise_z.tolist(),
        "maximum_abs_raw_mean_z": float(torch.max(absolute_raw_z)),
        "worst_10_percent_mean_abs_raw_mean_z": float(
            torch.mean(torch.topk(absolute_raw_z, worst_count).values)
        ),
        "maximum_abs_noise_scaled_adjusted_mean_z": float(
            torch.max(absolute_noise_z)
        ),
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
            torch.mean(torch.topk(absolute_noise_z, worst_count).values)
        ),
        "worst_instrument_index": worst_index,
        "z_target": z_target,
        "passes_target": bool(torch.max(absolute_noise_z) <= z_target),
        "mean_squared_dlvsq_cap_barrier": float(values["cap_mean_square"]),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument("--teacher-checkpoint", type=Path, default=_DEFAULT_TEACHER)
    parser.add_argument("--selected-state-index", type=int, default=588)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports" / "state-generic-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints" / "state-generic-q.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--confirmation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--fit-seed", type=int, default=10_484_101)
    parser.add_argument("--validation-seed", type=int, default=10_485_101)
    parser.add_argument("--confirmation-seed", type=int, default=10_487_101)
    parser.add_argument("--audit-every", type=int, default=10)
    parser.add_argument("--plateau-epochs", type=int, default=200)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-5)
    parser.add_argument("--validation-z-target", type=float, default=2.0)
    parser.add_argument("--confirmation-z-target", type=float, default=2.0)
    parser.add_argument("--soft-maximum-temperature", type=float, default=1.0)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    counts = (
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
        raise ValueError("generic Q arguments require GPU0 and positive counts")
    if (
        arguments.fit_replications < 2
        or arguments.validation_replications < 2
        or arguments.confirmation_replications < 2
    ):
        raise ValueError("MC standard errors require at least two replications")
    if arguments.surface_paths < 4 or arguments.surface_paths % 2:
        raise ValueError("surface paths must be at least four and even")
    if len(
        {
            arguments.fit_seed,
            arguments.validation_seed,
            arguments.confirmation_seed,
        }
    ) != 3:
        raise ValueError("fit, validation and confirmation streams must be disjoint")
    if arguments.selected_state_index < 0:
        raise ValueError("selected state index must be non-negative")
    if not 0.0 < arguments.minimum_relative_improvement < 1.0:
        raise ValueError("minimum relative improvement must lie in (0, 1)")
    if arguments.validation_z_target <= 0.0:
        raise ValueError("validation z target must be positive")
    if arguments.confirmation_z_target <= 0.0:
        raise ValueError("confirmation z target must be positive")
    if arguments.soft_maximum_temperature <= 0.0:
        raise ValueError("soft-maximum temperature must be positive")


def _draws(
    arguments: argparse.Namespace,
    *,
    seed: int,
    replications: int,
    front_state_size: int,
    env: BaseEnvironment,
) -> list[CommonQDraw]:
    return [
        make_generic_q_draw(
            front_state_size=front_state_size,
            spot_contexts=arguments.spot_contexts,
            surface_paths=arguments.surface_paths,
            seed=seed + replication,
            dtype=env.fdtype,
            device=env.device,
        )
        for replication in range(replications)
    ]


def run(arguments: argparse.Namespace) -> None:
    """Fit and independently validate the generic law at one state."""

    _validate_arguments(arguments)
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
    dates = train_dates + validation_dates
    current_states = torch.cat((train_states, validation_states), dim=0)
    selected = arguments.selected_state_index
    if selected >= len(current_states):
        raise ValueError("selected state index is outside the historical state set")
    current_state = current_states[selected].detach()
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = GenericStateQLawConfig(front_state_size=front_state_size)
    law = GenericConditionalFrontLaw(
        env,
        current_front_state=current_state[:front_state_size],
        config=config,
    )
    teacher_payload = torch.load(
        arguments.teacher_checkpoint,
        map_location=env.device,
        weights_only=False,
    )
    if not isinstance(teacher_payload, Mapping):
        raise ValueError("generic-law teacher checkpoint is invalid")
    teacher_state = cast(Mapping[str, torch.Tensor], teacher_payload["law_state_dict"])
    law.initialize_affine_teacher(
        location_shift=teacher_state["location_shift"][selected],
        spot_loading=teacher_state["spot_loading"][selected],
        raw_noise_std=teacher_state["raw_noise_std"][selected],
    )

    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    with torch.no_grad():
        raw = model.raw_states_from_states(current_state.unsqueeze(0))
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        current_surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_calls = model.sanos_decoder.price_calls_batch(
            current_surface,
            expiries_y=model.sanos_decoder.model_expiries_y[1].expand(1, 11),
            strikes=normalized_strikes[None],
        )[0].detach()
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )
    fit_draws = _draws(
        arguments,
        seed=arguments.fit_seed,
        replications=arguments.fit_replications,
        front_state_size=front_state_size,
        env=env,
    )
    validation_draws = _draws(
        arguments,
        seed=arguments.validation_seed,
        replications=arguments.validation_replications,
        front_state_size=front_state_size,
        env=env,
    )
    confirmation_draws = _draws(
        arguments,
        seed=arguments.confirmation_seed,
        replications=arguments.confirmation_replications,
        front_state_size=front_state_size,
        env=env,
    )

    simulation_config = {
        "selected_state_index": selected,
        "date": dates[selected].isoformat(),
        "fit_replications": arguments.fit_replications,
        "validation_replications": arguments.validation_replications,
        "confirmation_replications": arguments.confirmation_replications,
        "spot_contexts": arguments.spot_contexts,
        "surface_paths": arguments.surface_paths,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "actual_expected_gross_spot_return": 1.0,
        "control_beta_scope": "separate_pooled_fit_and_validation_samples",
        "stopping_noise_scale": (
            "max(raw_option_mean_mcse, abs(beta) * spot_return_mean_mcse)"
        ),
        "branch_variable": None,
        "branch_target": None,
    }
    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=5.0e-5,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_loss = math.inf
    stale_epochs = 0
    if arguments.checkpoint.exists():
        payload = torch.load(
            arguments.checkpoint, map_location=env.device, weights_only=False
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-state-generic-q-training-v1"
            or payload.get("source_completion_sha256")
            != sources["completion_sha256"]
            or payload.get("law_config") != asdict(config)
            or payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("generic Q checkpoint identity is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        optimizer.load_state_dict(
            cast(dict[str, Any], payload["optimizer_state_dict"])
        )
        scheduler.load_state_dict(
            cast(dict[str, Any], payload["scheduler_state_dict"])
        )
        # Learning rate is an operational training choice and may be reduced
        # safely when warm-starting an otherwise identical experiment.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = arguments.learning_rate
        history = list(cast(list[dict[str, float]], payload["history"]))
        start_epoch = len(history)
        best_loss = float(payload["best_loss"])
        stale_epochs = int(payload["stale_epochs"])
        print(f"resuming generic state Q at epoch {start_epoch}", flush=True)

    validation_summary: dict[str, Any] | None = None
    stopped_for_validation = False
    stopped_for_plateau = False
    for epoch in range(start_epoch, arguments.epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fit_values = _evaluate(
                model=model,
                law=law,
                current_state=current_state,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=fit_draws,
            )
            fit_summary = _summary(
                fit_values,
                z_target=arguments.validation_z_target,
            )
            adjusted_rep = fit_values["adjusted_replication_means"].double()
            adjusted_mean = torch.mean(adjusted_rep, dim=0)
            raw_rep = fit_values["raw_replication_means"].double()
            spot_rep = fit_values["spot_replication_means"].double()
            raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(
                len(fit_draws)
            )
            spot_se = torch.std(spot_rep, correction=1) / math.sqrt(len(fit_draws))
            beta = fit_values["beta"].double()
            noise_scale = torch.maximum(torch.abs(beta) * spot_se, raw_se)
            scaled = adjusted_mean / noise_scale.clamp_min(1.0e-12)
            squared = scaled.square()
            temperature = arguments.soft_maximum_temperature
            loss = temperature * (
                torch.logsumexp(squared / temperature, dim=0)
                - math.log(len(squared))
            )
            weights = torch.softmax(squared / temperature, dim=0)
            mean_gradient = (
                2.0
                * scaled
                * weights
                / noise_scale.clamp_min(1.0e-12)
            ).to(dtype=env.fdtype)

        for draw in fit_draws:
            returns, _, gross = generic_roll_returns(
                model=model,
                law=law,
                current_state=current_state,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draw=draw,
            )
            spot_mean = torch.mean(gross - 1.0)
            adjusted_draw_mean = (
                torch.mean(returns, dim=0)
                - fit_values["beta"].to(dtype=env.fdtype) * spot_mean
            )
            surrogate = torch.sum(mean_gradient * adjusted_draw_mean) / len(
                fit_draws
            )
            surrogate.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        loss_value = float(loss)
        scheduler.step(loss_value)
        if loss_value < best_loss * (1.0 - arguments.minimum_relative_improvement):
            best_loss = loss_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_value,
            "fit_maximum_abs_raw_mean_z": float(
                fit_summary["maximum_abs_raw_mean_z"]
            ),
            "fit_worst_10_percent_mean_abs_raw_mean_z": float(
                fit_summary["worst_10_percent_mean_abs_raw_mean_z"]
            ),
            "fit_maximum_abs_noise_scaled_adjusted_mean_z": float(
                fit_summary["maximum_abs_noise_scaled_adjusted_mean_z"]
            ),
            "fit_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
                fit_summary[
                    "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"
                ]
            ),
            "state_noise_std": float(law.noise_std.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise FloatingPointError("non-finite generic state-Q training record")
        history.append(record)
        checkpoint = {
            "schema": "flex-state-generic-q-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "teacher_checkpoint": str(arguments.teacher_checkpoint),
            "teacher_checkpoint_sha256": checkpoint_digest(
                arguments.teacher_checkpoint
            ),
            "law_config": asdict(config),
            "simulation_config": simulation_config,
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
            f"noise={record['state_noise_std']:.6f} "
            f"lr={record['learning_rate']:.6g}",
            flush=True,
        )

        if (epoch + 1) % arguments.audit_every == 0:
            with torch.no_grad():
                validation_summary = _summary(
                    _evaluate(
                        model=model,
                        law=law,
                        current_state=current_state,
                        current_w2_calls=current_w2_calls,
                        normalized_strikes=normalized_strikes,
                        option_is_call=option_is_call,
                        support=support,
                        density=density,
                        sqrt_variance=sqrt_variance,
                        draws=validation_draws,
                    ),
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
                f"target={arguments.validation_z_target:.3f}",
                flush=True,
            )
            if (
                validation_summary["maximum_abs_noise_scaled_adjusted_mean_z"]
                <= arguments.validation_z_target
            ):
                stopped_for_validation = True
                break
        if stale_epochs >= arguments.plateau_epochs:
            stopped_for_plateau = True
            break

    if validation_summary is None or not stopped_for_validation:
        with torch.no_grad():
            validation_summary = _summary(
                _evaluate(
                    model=model,
                    law=law,
                    current_state=current_state,
                    current_w2_calls=current_w2_calls,
                    normalized_strikes=normalized_strikes,
                    option_is_call=option_is_call,
                    support=support,
                    density=density,
                    sqrt_variance=sqrt_variance,
                    draws=validation_draws,
                ),
                z_target=arguments.validation_z_target,
            )
    with torch.no_grad():
        confirmation_summary = _summary(
            _evaluate(
                model=model,
                law=law,
                current_state=current_state,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=confirmation_draws,
            ),
            z_target=arguments.confirmation_z_target,
        )
    validation_success = bool(validation_summary["passes_target"])
    confirmation_success = bool(confirmation_summary["passes_target"])
    semantic_success = validation_success and confirmation_success
    report = {
        "schema": "flex-state-generic-q-report-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "objective": OBJECTIVE_ID,
        "state_index": selected,
        "date": dates[selected].isoformat(),
        "history_partition": (
            "model_fit" if selected < len(train_states) else "held_out_2025"
        ),
        "instruments": [
            {
                "index": index,
                "type": "call" if bool(option_is_call[index]) else "put",
                "log_strike_over_spot": float(torch.log(strike)),
                "strike_over_spot": float(strike),
            }
            for index, strike in enumerate(normalized_strikes)
        ],
        "law": {
            "form": "h1_t=generic_triangular_transport(h1_0,spot_rank,epsilon)",
            "latent": "epsilon~N(0,I), antithetic within spot context",
            "full_support": True,
            "discrete_branch": False,
            "state_noise_std": float(law.noise_std.detach()),
        },
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
            "schema": "flex-state-generic-q-completion-v1",
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
        raise RuntimeError(
            "generic state-Q law failed validation or confirmation target"
        )


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(main, program="run_state_generic_q_existence", clean_sigterm=True)
