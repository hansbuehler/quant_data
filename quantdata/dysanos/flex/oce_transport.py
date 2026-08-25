"""Conditional power-exp OCE targets and first-line latent transports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.dysanos.flex.weekly_dynamics import TorchWeeklyDynamics
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    WeeklyMeasureConfig,
    WeeklyMeasureDraw,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklySanosSimulator,
    front_week_inverse_cdf_components,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.5")
@dataclass(frozen=True, kw_only=True)
class ConditionalOCEConfig:
    """Architecture of the state-and-spot conditional OCE optimizer."""

    state_sizes_by_expiry: tuple[int, ...]
    num_instruments: int
    maximum_preconditioned_action_norm: float | None = None
    preconditioned_action_bound_mode: Literal[
        "smooth_radial", "hard_cap_penalty"
    ] = "smooth_radial"
    action_cap_excess_penalty_weight: float = 1.0
    include_current_log_spot: bool = False
    current_log_spot_mean: float = 0.0
    current_log_spot_std: float = 1.0
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=3,
        num_width=256,
    )

    def __post_init__(self) -> None:
        verify_inp(
            len(self.state_sizes_by_expiry) >= 2
            and all(value > 0 for value in self.state_sizes_by_expiry),
            lambda: f"invalid OCE state blocks {self.state_sizes_by_expiry}",
        )
        verify_inp(
            self.num_instruments > 0,
            lambda: "OCE requires at least one instrument",
        )
        verify_inp(
            self.maximum_preconditioned_action_norm is None
            or (
                math.isfinite(self.maximum_preconditioned_action_norm)
                and self.maximum_preconditioned_action_norm > 0.0
            ),
            lambda: (
                "maximum OCE preconditioned-action norm must be finite-positive "
                "when specified"
            ),
        )
        verify_inp(
            self.preconditioned_action_bound_mode
            in {"smooth_radial", "hard_cap_penalty"},
            lambda: (
                "invalid preconditioned action bound mode "
                f"{self.preconditioned_action_bound_mode!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.action_cap_excess_penalty_weight)
            and self.action_cap_excess_penalty_weight >= 0.0,
            lambda: "action cap-excess penalty weight must be finite-nonnegative",
        )
        verify_inp(
            math.isfinite(self.current_log_spot_mean)
            and math.isfinite(self.current_log_spot_std)
            and self.current_log_spot_std > 0.0,
            lambda: "current log-spot normalization must be finite with positive scale",
        )

    @property
    def state_context_width(self) -> int:
        """Current first- and second-expiry states visible to the W2 action."""

        return sum(self.state_sizes_by_expiry[:2])


class ConditionalOCEAction(BaseModel):
    """Full W2 action conditional on state and the realized implied spot."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ConditionalOCEConfig,
        return_rms: torch.Tensor,
    ):
        super().__init__(env)
        if return_rms.shape != (config.num_instruments,) or not bool(
            torch.all(torch.isfinite(return_rms) & (return_rms > 0.0))
        ):
            raise ValueError("OCE return RMS has invalid shape or values")
        self.config = config
        self.register_buffer(
            "return_rms",
            return_rms.to(dtype=env.fdtype, device=env.device).detach().clone(),
        )
        self.network = FeedForward(
            env,
            config.state_context_width + 1 + int(config.include_current_log_spot),
            config.num_instruments,
            config.network,
        )
        final = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        current_log_spot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the preconditioned call action."""

        if current_state.shape[:-1] != gross_spot_return.shape:
            raise ValueError("OCE state and spot batch shapes differ")
        values = [current_state[..., : self.config.state_context_width]]
        if self.config.include_current_log_spot:
            if current_log_spot is None or current_log_spot.shape != gross_spot_return.shape:
                raise ValueError("OCE current log spot has invalid shape")
            values.append(
                (
                    (current_log_spot - self.config.current_log_spot_mean)
                    / self.config.current_log_spot_std
                ).unsqueeze(-1)
            )
        values.append(torch.log(gross_spot_return).unsqueeze(-1))
        context = torch.cat(values, dim=-1)
        raw = self.network(context)
        return raw / self.return_rms


class SpotMarginalOCEAction(BaseModel):
    """Population OCE action by current state, shared across future spot ranks."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ConditionalOCEConfig,
        inverse_sqrt_second_moment: torch.Tensor,
    ):
        super().__init__(env)
        expected = (config.num_instruments, config.num_instruments)
        if inverse_sqrt_second_moment.shape != expected or not bool(
            torch.all(torch.isfinite(inverse_sqrt_second_moment))
        ):
            raise ValueError("OCE return whitener has invalid shape or values")
        self.config = config
        self.register_buffer(
            "inverse_sqrt_second_moment",
            inverse_sqrt_second_moment.to(dtype=env.fdtype, device=env.device).detach().clone(),
        )
        self.network = FeedForward(
            env,
            config.state_context_width + int(config.include_current_log_spot),
            config.num_instruments,
            config.network,
        )
        final = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        current_state: torch.Tensor,
        current_log_spot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return option actions and their whitened coordinates."""

        action, bounded, _, _ = self.forward_with_cap_penalty(
            current_state,
            current_log_spot,
        )
        return action, bounded

    def forward_with_cap_penalty(
        self,
        current_state: torch.Tensor,
        current_log_spot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the action, capped coordinates, penalty, and maximum excess."""

        context = current_state[..., : self.config.state_context_width]
        if self.config.include_current_log_spot:
            if current_log_spot is None or current_log_spot.shape != current_state.shape[:-1]:
                raise ValueError("OCE current log spot has invalid shape")
            normalized_log_spot = (
                current_log_spot - self.config.current_log_spot_mean
            ) / self.config.current_log_spot_std
            context = torch.cat((context, normalized_log_spot.unsqueeze(-1)), dim=-1)
        unbounded = self.network(context)
        maximum_norm = self.config.maximum_preconditioned_action_norm
        cap_penalty = torch.zeros((), dtype=unbounded.dtype, device=unbounded.device)
        maximum_excess = torch.zeros((), dtype=unbounded.dtype, device=unbounded.device)
        if maximum_norm is not None:
            norm = torch.linalg.vector_norm(unbounded, dim=-1, keepdim=True)
            if self.config.preconditioned_action_bound_mode == "smooth_radial":
                ratio = maximum_norm * torch.tanh(norm / maximum_norm) / norm.clamp_min(
                    torch.finfo(unbounded.dtype).tiny
                )
                scale = torch.where(norm > 0.0, ratio, torch.ones_like(ratio))
                bounded = unbounded * scale
            else:
                # Preserve the network output exactly below the cap. Above it,
                # use a mechanical radial cap while the explicit excess penalty
                # supplies the missing inward gradient; no saturating transform
                # is applied to admissible actions.
                scale = torch.clamp_max(
                    maximum_norm / norm.clamp_min(torch.finfo(unbounded.dtype).tiny),
                    1.0,
                )
                bounded = unbounded * scale
                excess = torch.clamp_min(norm[..., 0] - maximum_norm, 0.0)
                cap_penalty = (
                    self.config.action_cap_excess_penalty_weight
                    * torch.mean(excess.square())
                )
                maximum_excess = torch.max(excess)
        else:
            bounded = unbounded
        action = bounded @ self.inverse_sqrt_second_moment.T
        return action, bounded, cap_penalty, maximum_excess


class SpotConditionedFirstLineTransport(GenericWeeklyMeasureKernel):
    """Incremental W2 transport conditional on current absolute log spot."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        dynamics: TorchWeeklyDynamics | JointHierarchicalWeeklyDynamicsAdapter,
        config: WeeklyMeasureConfig,
        current_log_spot_mean: float,
        current_log_spot_std: float,
    ):
        if (
            not math.isfinite(current_log_spot_mean)
            or not math.isfinite(current_log_spot_std)
            or current_log_spot_std <= 0.0
        ):
            raise ValueError("transport log-spot normalization is invalid")
        super().__init__(env, dynamics=dynamics, config=config)
        width = config.state_sizes_by_expiry[0]
        self.context_networks[0] = FeedForward(
            env,
            self._context_width(0) + 1,
            4 * width,
            config.context_network,
        )
        final = (
            self.context_networks[0].I0
            if len(self.context_networks[0].In) == 0
            else self.context_networks[0].In[-1]
        )
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
        self.register_buffer(
            "current_log_spot_mean",
            torch.as_tensor(current_log_spot_mean, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "current_log_spot_std",
            torch.as_tensor(current_log_spot_std, dtype=env.fdtype, device=env.device),
        )

    def _first_line_context(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        current_log_spot: torch.Tensor | None,
    ) -> torch.Tensor:
        if current_log_spot is None or current_log_spot.shape != current_state.shape[:-1]:
            raise ValueError("transport current log spot has invalid shape")
        base = self._context(
            current_state,
            [],
            torch.log(gross_spot_return),
            0,
        )
        normalized = (
            current_log_spot - self.current_log_spot_mean
        ) / self.current_log_spot_std
        return torch.cat((base, normalized.unsqueeze(-1)), dim=-1)

    def step_first_line_with_log_abs_det(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
        current_log_spot: torch.Tensor | None = None,
    ) -> tuple[WeeklyMeasureDraw, torch.Tensor]:
        """Apply the spot-conditioned incremental map to the first state line."""

        self._validate_inputs(current_state, gross_spot_return, standard_normals)
        if not self.config.hierarchical:
            raise ValueError("first-line transport requires hierarchical geometry")
        block_slice = self._block_slice(0)
        width = block_slice.stop - block_slice.start
        shift_raw, scale_raw, amplitude_raw, gate_raw = torch.split(
            self.context_networks[0](
                self._first_line_context(
                    current_state,
                    gross_spot_return,
                    current_log_spot,
                )
            ),
            width,
            dim=-1,
        )
        shift = self.config.maximum_shift * torch.tanh(
            shift_raw / self.config.maximum_shift
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
        reference_surface = standard_normals[..., 1:]
        reference_block = reference_surface[..., block_slice]
        lower_effect = (
            reference_block
            @ torch.tril(self.strict_lower_raw[0], diagonal=-1).T
        )
        proposal = (
            shift
            + scale * reference_block
            + amplitude * torch.tanh(reference_block)
            + lower_effect
        )
        transformed_block = (1.0 - gate) * reference_block + gate * proposal
        coordinate_derivative = (1.0 - gate) + gate * (
            scale + amplitude * (1.0 - torch.tanh(reference_block).square())
        )
        self.verify(
            lambda: torch.all(coordinate_derivative > 0.0),
            "non-positive spot-conditioned first-line derivative",
        )
        log_abs_det = torch.sum(torch.log(coordinate_derivative), dim=-1)
        transformed_blocks = [transformed_block]
        for later_index in range(1, self.transformed_block_count):
            transformed_blocks.append(
                reference_surface[..., self._block_slice(later_index)]
            )
        return (
            self._assemble_draw(current_state, standard_normals, transformed_blocks),
            log_abs_det,
        )

    def minimum_first_line_coordinate_derivative(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        current_log_spot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the incremental map's analytic positive derivative bound."""

        width = self.config.state_sizes_by_expiry[0]
        _, scale_raw, amplitude_raw, gate_raw = torch.split(
            self.context_networks[0](
                self._first_line_context(
                    current_state,
                    gross_spot_return,
                    current_log_spot,
                )
            ),
            width,
            dim=-1,
        )
        scale = torch.exp(
            self.config.maximum_log_scale
            * torch.tanh(scale_raw / self.config.maximum_log_scale)
        )
        amplitude = (
            self.config.generic_amplitude_fraction
            * scale
            * torch.tanh(amplitude_raw)
        )
        gate = torch.sigmoid(gate_raw)
        return torch.min((1.0 - gate) + gate * (scale - torch.abs(amplitude)))


@dataclass(frozen=True)
class ConditionalPathBatch:
    """P paths grouped by a fixed current state and realized spot draw."""

    context_state: torch.Tensor
    context_gross_spot_return: torch.Tensor
    current_state: torch.Tensor
    gross_spot_return: torch.Tensor
    standard_normals: torch.Tensor
    repeated_origin: torch.Tensor
    spot_contexts_per_origin: int
    surface_paths_per_context: int

    @property
    def num_contexts(self) -> int:
        return len(self.context_state)

    @property
    def num_origins(self) -> int:
        """Number of current states represented in the conditional batch."""

        return self.num_contexts // self.spot_contexts_per_origin


def sample_conditional_paths(
    data: WeeklyMeasureTrainingData,
    origin_indices: torch.Tensor,
    *,
    spot_contexts_per_origin: int,
    surface_paths_per_context: int,
    generator: torch.Generator,
) -> ConditionalPathBatch:
    """Draw repeated surface innovations for every fixed implied-spot rank."""

    if (
        origin_indices.ndim != 1
        or len(origin_indices) < 1
        or spot_contexts_per_origin < 2
        or spot_contexts_per_origin % 2
        or surface_paths_per_context < 2
        or surface_paths_per_context % 2
    ):
        raise ValueError("conditional sampling requires positive even path counts")
    device = data.reduced_states.device
    dtype = data.reduced_states.dtype
    origins = len(origin_indices)
    spot_shift = torch.rand(
        (origins, 1),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    spot_grid = (
        torch.arange(spot_contexts_per_origin, dtype=dtype, device=device) + 0.5
    ) / spot_contexts_per_origin
    spot_uniform = torch.remainder(spot_grid[None] + spot_shift, 1.0)
    probability_floor = 4.0 * torch.finfo(dtype).eps
    spot_uniform = torch.clamp(spot_uniform, probability_floor, 1.0 - probability_floor)
    spot_normals = (
        math.sqrt(2.0) * torch.erfinv(2.0 * spot_uniform - 1.0)
    ).reshape(-1)
    context_origin = torch.repeat_interleave(origin_indices, spot_contexts_per_origin)
    context_state = data.reduced_states[context_origin]
    context_gross = front_week_inverse_cdf_components(
        data.front_support[context_origin],
        data.front_density[context_origin],
        data.front_backbone_total_vol[context_origin],
        spot_normals,
    )
    contexts = len(context_origin)
    state_size = data.reduced_states.shape[1]
    surface_half = torch.randn(
        (contexts, surface_paths_per_context // 2, state_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    surface_normals = torch.cat((surface_half, -surface_half), dim=1)
    full_normals = torch.cat(
        (
            spot_normals[:, None, None].expand(contexts, surface_paths_per_context, 1),
            surface_normals,
        ),
        dim=-1,
    ).reshape(contexts * surface_paths_per_context, state_size + 1)
    repeated_origin = torch.repeat_interleave(context_origin, surface_paths_per_context)
    return ConditionalPathBatch(
        context_state=context_state,
        context_gross_spot_return=context_gross,
        current_state=data.reduced_states[repeated_origin],
        gross_spot_return=torch.repeat_interleave(context_gross, surface_paths_per_context),
        standard_normals=full_normals,
        repeated_origin=repeated_origin,
        spot_contexts_per_origin=spot_contexts_per_origin,
        surface_paths_per_context=surface_paths_per_context,
    )


def powerexp_utility(value: torch.Tensor) -> torch.Tensor:
    """Evaluate the unit power-exponential utility in Torch."""

    positive = torch.clamp_min(value, 0.0)
    negative = torch.clamp_max(value, 0.0)
    return torch.where(
        value > 0.0,
        -torch.expm1(-positive),
        negative - 0.5 * negative.square(),
    )


def log_powerexp_marginal(value: torch.Tensor) -> torch.Tensor:
    """Return log marginal utility without evaluating an inactive bad branch."""

    nonpositive = torch.clamp_max(value, 0.0)
    return torch.where(value > 0.0, -value, torch.log1p(-nonpositive))


def solve_powerexp_translation(
    pnl: torch.Tensor,
    *,
    num_iterations: int = 64,
) -> torch.Tensor:
    r"""Solve ``mean(u'(y + pnl)) = 1`` for every conditional context.

    The scalar translation is an inner concave optimum. Solving it rather
    than predicting it makes conditional density normalization a numerical
    invariant. The detached solution gives the envelope-theorem gradient when
    the outer portfolio action is optimized.
    """

    if pnl.ndim != 2 or pnl.shape[1] < 2 or num_iterations < 1:
        raise ValueError("OCE translation requires (contexts,paths) and iterations")
    values = pnl.detach()
    lower = -torch.amax(values, dim=1) - 1.0
    upper = -torch.amin(values, dim=1) + 1.0
    for _ in range(num_iterations):
        middle = 0.5 * (lower + upper)
        log_marginal = log_powerexp_marginal(values + middle[:, None])
        residual = torch.mean(torch.exp(log_marginal), dim=1) - 1.0
        lower = torch.where(residual > 0.0, middle, lower)
        upper = torch.where(residual > 0.0, upper, middle)
    return 0.5 * (lower + upper)


def estimate_return_inverse_sqrt(
    returns: torch.Tensor,
    *,
    minimum_standard_deviation: float = 1.0e-4,
) -> torch.Tensor:
    """Estimate a symmetric inverse square root of the return second moment."""

    if returns.ndim != 3 or returns.shape[-1] < 1:
        raise ValueError("return whitening input must have conditional path geometry")
    if not math.isfinite(minimum_standard_deviation) or minimum_standard_deviation <= 0.0:
        raise ValueError("minimum return standard deviation must be finite-positive")
    flattened = returns.reshape(-1, returns.shape[-1])
    second_moment = flattened.T @ flattened / len(flattened)
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    floor = minimum_standard_deviation**2
    inverse_scale = torch.rsqrt(torch.clamp_min(eigenvalues, floor))
    result = (eigenvectors * inverse_scale[None]) @ eigenvectors.T
    if not bool(torch.all(torch.isfinite(result))):
        raise FloatingPointError("return inverse-square-root preconditioner is non-finite")
    return result


def solve_conditional_powerexp_oce(
    returns: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    *,
    num_iterations: int = 64,
    ridge: float = 1.0e-10,
    maximum_newton_step: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    r"""Solve the conditional power-exp OCE first-order conditions in batch.

    The 11 option returns are whitened by one invertible global transform, so
    the solve retains every restriction while avoiding near-null call-spread
    coordinates. Each origin has one option action shared across its spot
    draws, while a separate cash translation enforces unit density mass for
    every spot draw. Option moments are imposed only after integrating spot.
    """

    if (
        returns.ndim != 4
        or inverse_sqrt_second_moment.shape
        != (returns.shape[-1], returns.shape[-1])
        or num_iterations < 1
        or not math.isfinite(ridge)
        or ridge <= 0.0
        or not math.isfinite(maximum_newton_step)
        or maximum_newton_step <= 0.0
    ):
        raise ValueError("invalid conditional OCE Newton geometry or controls")
    origins, spot_contexts, paths, instruments = returns.shape
    # The call-return basis is nearly rank deficient. Solve the small dual in
    # float64 even though SANOS path valuation and transport training use the
    # configured float32 environment.
    values = returns.double()
    inverse_sqrt = inverse_sqrt_second_moment.double()
    whitened = values @ inverse_sqrt
    whitened_action = torch.zeros(
        (origins, instruments),
        dtype=values.dtype,
        device=returns.device,
    )
    identity = torch.eye(instruments, dtype=values.dtype, device=returns.device)
    for _ in range(num_iterations):
        pnl = torch.sum(whitened * whitened_action[:, None, None], dim=-1)
        translation = solve_powerexp_translation(
            pnl.reshape(origins * spot_contexts, paths)
        ).reshape(origins, spot_contexts)
        shifted = translation[..., None] + pnl
        log_marginal = log_powerexp_marginal(shifted)
        marginal = torch.exp(log_marginal)
        gradient = torch.mean(marginal[..., None] * whitened, dim=(1, 2))
        curvature = torch.where(shifted > 0.0, marginal, torch.ones_like(marginal))
        hessian_by_spot = torch.einsum(
            "osp,ospi,ospj->osij",
            curvature,
            whitened,
            whitened,
        ) / paths
        cash_cross = torch.mean(curvature[..., None] * whitened, dim=2)
        cash_curvature = torch.mean(curvature, dim=2)
        profiled_hessian = torch.mean(
            hessian_by_spot
            - cash_cross[..., :, None] * cash_cross[..., None, :]
            / cash_curvature[..., None, None].clamp_min(ridge),
            dim=1,
        )
        delta = torch.linalg.solve(
            profiled_hessian + ridge * identity[None],
            gradient[..., None],
        )[..., 0]
        delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        delta = delta * torch.clamp_max(maximum_newton_step / delta_norm.clamp_min(ridge), 1.0)
        current_objective = torch.mean(powerexp_utility(shifted), dim=(1, 2)) - torch.mean(
            translation, dim=1
        )
        accepted = torch.zeros(origins, dtype=torch.bool, device=returns.device)
        candidate_action = whitened_action
        scale = torch.ones(origins, dtype=values.dtype, device=returns.device)
        for _ in range(12):
            proposal = whitened_action + scale[:, None] * delta
            proposal_pnl = torch.sum(whitened * proposal[:, None, None], dim=-1)
            proposal_translation = solve_powerexp_translation(
                proposal_pnl.reshape(origins * spot_contexts, paths)
            ).reshape(origins, spot_contexts)
            proposal_shifted = proposal_translation[..., None] + proposal_pnl
            proposal_objective = torch.mean(
                powerexp_utility(proposal_shifted), dim=(1, 2)
            ) - torch.mean(proposal_translation, dim=1)
            improve = (~accepted) & (proposal_objective >= current_objective)
            candidate_action = torch.where(improve[:, None], proposal, candidate_action)
            accepted |= improve
            scale = torch.where(accepted, scale, 0.5 * scale)
        whitened_action = candidate_action

    action = whitened_action @ inverse_sqrt.T
    pnl = torch.sum(values * action[:, None, None], dim=-1)
    translation = solve_powerexp_translation(
        pnl.reshape(origins * spot_contexts, paths)
    ).reshape(origins, spot_contexts)
    shifted = translation[..., None] + pnl
    log_marginal = log_powerexp_marginal(shifted)
    marginal = torch.exp(log_marginal)
    mass_residual = torch.mean(marginal, dim=2) - 1.0
    action_residual = torch.mean(marginal[..., None] * values, dim=(1, 2))
    objective = torch.mean(powerexp_utility(shifted), dim=(1, 2)) - torch.mean(
        translation, dim=1
    )
    return action, translation, {
        "log_marginal": log_marginal,
        "mass_residual": mass_residual,
        "action_residual": action_residual,
        "objective": objective,
        "whitened_action": whitened_action,
    }


def _project_l1_ball(values: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    """Project batched vectors mechanically onto a raw-coordinate L1 ball."""

    if values.ndim != 2 or not math.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError("L1 projection requires batched vectors and a positive cap")
    absolute = torch.abs(values)
    already_inside = torch.sum(absolute, dim=1, keepdim=True) <= maximum_norm
    ordered, _ = torch.sort(absolute, dim=1, descending=True)
    indices = torch.arange(
        1,
        values.shape[1] + 1,
        dtype=values.dtype,
        device=values.device,
    )
    cumulative_excess = torch.cumsum(ordered, dim=1) - maximum_norm
    active = ordered - cumulative_excess / indices[None] > 0.0
    rho = torch.sum(active, dim=1).clamp_min(1) - 1
    threshold = cumulative_excess.gather(1, rho[:, None]) / (rho + 1).to(
        values.dtype
    )[:, None]
    projected = torch.sign(values) * torch.clamp_min(absolute - threshold, 0.0)
    return torch.where(already_inside, values, projected)


def solve_conditional_powerexp_oce_l1(
    returns: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    *,
    maximum_l1_action: float,
    l1_cost_per_unit: float,
    initial_action: torch.Tensor | None = None,
    num_iterations: int = 128,
    maximum_line_search_steps: int = 24,
    convergence_tolerance: float = 1.0e-9,
    initial_step_multiplier: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    r"""Solve a cost-penalized OCE inside a raw-option L1 notional ball.

    The L1 constraint is imposed by exact Euclidean projection in raw option
    coordinates. It is not represented by a saturating activation. Proximal
    soft-thresholding applies the transaction cost, exact projection applies
    the cap, and backtracking uses the profiled OCE objective.
    """

    if (
        returns.ndim != 4
        or inverse_sqrt_second_moment.shape
        != (returns.shape[-1], returns.shape[-1])
        or not math.isfinite(maximum_l1_action)
        or maximum_l1_action <= 0.0
        or not math.isfinite(l1_cost_per_unit)
        or l1_cost_per_unit < 0.0
        or num_iterations < 1
        or maximum_line_search_steps < 1
        or not math.isfinite(convergence_tolerance)
        or convergence_tolerance <= 0.0
        or not math.isfinite(initial_step_multiplier)
        or initial_step_multiplier <= 0.0
    ):
        raise ValueError("invalid L1 conditional OCE geometry or controls")
    origins, spot_contexts, paths, instruments = returns.shape
    values = returns.double()
    inverse_sqrt = inverse_sqrt_second_moment.double()
    if initial_action is None:
        raw_action = torch.zeros(
            (origins, instruments),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        if initial_action.shape != (origins, instruments) or not bool(
            torch.all(torch.isfinite(initial_action))
        ):
            raise ValueError("initial L1 OCE action has invalid shape or values")
        raw_action = initial_action.to(dtype=values.dtype, device=values.device)
    raw_action = _project_l1_ball(raw_action, maximum_l1_action)
    second_moment = torch.einsum("ospi,ospj->oij", values, values) / (
        spot_contexts * paths
    )
    base_step = torch.reciprocal(
        torch.linalg.eigvalsh(second_moment)[..., -1].clamp_min(
            torch.finfo(values.dtype).tiny
        )
    )
    accepted_iterations = torch.zeros(
        origins,
        dtype=torch.long,
        device=values.device,
    )
    iterations_run = torch.zeros_like(accepted_iterations)
    converged = torch.zeros(origins, dtype=torch.bool, device=values.device)
    last_relative_action_change = torch.full(
        (origins,),
        float("inf"),
        dtype=values.dtype,
        device=values.device,
    )
    last_objective_improvement = torch.full_like(
        last_relative_action_change,
        float("inf"),
    )
    for _ in range(num_iterations):
        pnl = torch.sum(values * raw_action[:, None, None], dim=-1)
        translation = solve_powerexp_translation(
            pnl.reshape(origins * spot_contexts, paths)
        ).reshape(origins, spot_contexts)
        shifted = translation[..., None] + pnl
        log_marginal = log_powerexp_marginal(shifted)
        marginal = torch.exp(log_marginal)
        gradient = torch.mean(marginal[..., None] * values, dim=(1, 2))
        smooth_objective = torch.mean(powerexp_utility(shifted), dim=(1, 2)) - torch.mean(
            translation,
            dim=1,
        )
        current_objective = smooth_objective - l1_cost_per_unit * torch.sum(
            torch.abs(raw_action),
            dim=1,
        )
        accepted = torch.zeros(origins, dtype=torch.bool, device=values.device)
        candidate_raw = raw_action
        candidate_objective = current_objective
        step = initial_step_multiplier * base_step
        for _ in range(maximum_line_search_steps):
            unpenalized = raw_action + step[:, None] * gradient
            proposal_raw = torch.sign(unpenalized) * torch.clamp_min(
                torch.abs(unpenalized) - step[:, None] * l1_cost_per_unit,
                0.0,
            )
            proposal_raw = _project_l1_ball(proposal_raw, maximum_l1_action)
            proposal_pnl = torch.sum(values * proposal_raw[:, None, None], dim=-1)
            proposal_translation = solve_powerexp_translation(
                proposal_pnl.reshape(origins * spot_contexts, paths)
            ).reshape(origins, spot_contexts)
            proposal_shifted = proposal_translation[..., None] + proposal_pnl
            proposal_smooth_objective = torch.mean(
                powerexp_utility(proposal_shifted),
                dim=(1, 2),
            ) - torch.mean(proposal_translation, dim=1)
            proposal_objective = proposal_smooth_objective - l1_cost_per_unit * torch.sum(
                torch.abs(proposal_raw),
                dim=1,
            )
            improve = (~accepted) & (proposal_objective >= current_objective)
            candidate_raw = torch.where(improve[:, None], proposal_raw, candidate_raw)
            candidate_objective = torch.where(
                improve,
                proposal_objective,
                candidate_objective,
            )
            accepted |= improve
            step = torch.where(accepted, step, 0.5 * step)
            if bool(torch.all(accepted)):
                break
        action_scale = 1.0 + torch.linalg.vector_norm(raw_action, dim=1)
        last_relative_action_change = (
            torch.linalg.vector_norm(candidate_raw - raw_action, dim=1) / action_scale
        )
        last_objective_improvement = candidate_objective - current_objective
        raw_action = candidate_raw
        accepted_iterations += accepted.long()
        iterations_run += 1
        converged = (
            last_relative_action_change <= convergence_tolerance
        ) & (last_objective_improvement <= convergence_tolerance)
        if bool(torch.all(converged)):
            break

    pnl = torch.sum(values * raw_action[:, None, None], dim=-1)
    translation = solve_powerexp_translation(
        pnl.reshape(origins * spot_contexts, paths)
    ).reshape(origins, spot_contexts)
    shifted = translation[..., None] + pnl
    log_marginal = log_powerexp_marginal(shifted)
    marginal = torch.exp(log_marginal)
    mass_residual = torch.mean(marginal, dim=2) - 1.0
    action_residual = torch.mean(marginal[..., None] * values, dim=(1, 2))
    smooth_objective = torch.mean(powerexp_utility(shifted), dim=(1, 2)) - torch.mean(
        translation,
        dim=1,
    )
    l1_norm = torch.sum(torch.abs(raw_action), dim=1)
    transaction_cost = l1_cost_per_unit * l1_norm
    whitened_action = torch.linalg.solve(inverse_sqrt, raw_action.T).T
    return raw_action, translation, {
        "log_marginal": log_marginal,
        "mass_residual": mass_residual,
        "action_residual": action_residual,
        "objective": smooth_objective,
        "cost_adjusted_objective": smooth_objective - transaction_cost,
        "transaction_cost": transaction_cost,
        "l1_norm": l1_norm,
        "whitened_action": whitened_action,
        "accepted_iterations": accepted_iterations,
        "iterations_run": iterations_run,
        "converged": converged,
        "last_relative_action_change": last_relative_action_change,
        "last_objective_improvement": last_objective_improvement,
    }


def conditional_oce_objective(
    returns: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the optimized conditional OCE and finite-batch residuals."""

    if returns.ndim != 3 or action.shape != returns.shape[::2]:
        raise ValueError(
            "OCE shapes must be returns=(contexts,paths,instruments), "
            "action=(contexts,instruments)"
        )
    pnl = torch.sum(returns * action[:, None], dim=-1)
    translation = solve_powerexp_translation(pnl)
    shifted = translation[:, None] + pnl
    utility = powerexp_utility(shifted)
    log_marginal = log_powerexp_marginal(shifted)
    marginal = torch.exp(log_marginal)
    objective = torch.mean(torch.mean(utility, dim=1) - translation)
    mass_residual = torch.mean(marginal, dim=1) - 1.0
    action_residual = torch.mean(marginal[..., None] * returns, dim=1)
    return objective, {
        "pnl": pnl,
        "translation": translation,
        "log_marginal": log_marginal,
        "mass_residual": mass_residual,
        "action_residual": action_residual,
    }


def spot_marginal_oce_objective(
    returns: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""Optimize cash per spot draw but impose option moments over spot.

    ``returns`` has axes ``(origin, spot, surface_path, instrument)`` and one
    action is shared by all spot draws of an origin. Thus ``E[D|spot]=1``
    preserves the front-week implied spot marginal, while ``E[D R]=0`` is the
    economically required unconditional option restriction.
    """

    if returns.ndim != 4 or action.shape != (returns.shape[0], returns.shape[-1]):
        raise ValueError("spot-marginal OCE has inconsistent return/action geometry")
    origins, spot_contexts, paths, _ = returns.shape
    pnl = torch.sum(returns * action[:, None, None], dim=-1)
    translation = solve_powerexp_translation(
        pnl.reshape(origins * spot_contexts, paths)
    ).reshape(origins, spot_contexts)
    shifted = translation[..., None] + pnl
    utility = powerexp_utility(shifted)
    log_marginal = log_powerexp_marginal(shifted)
    marginal = torch.exp(log_marginal)
    objective_by_origin = torch.mean(utility, dim=(1, 2)) - torch.mean(translation, dim=1)
    return torch.mean(objective_by_origin), {
        "translation": translation,
        "log_marginal": log_marginal,
        "mass_residual": torch.mean(marginal, dim=2) - 1.0,
        "action_residual": torch.mean(marginal[..., None] * returns, dim=(1, 2)),
        "objective_by_origin": objective_by_origin,
    }


def w2_roll_returns(
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    next_state: torch.Tensor,
    normalized_strikes: torch.Tensor,
) -> torch.Tensor:
    """Return fixed-cash-strike W2-to-W1 normalized call value changes."""

    return simulator.hjm_roll_line_residuals(
        batch.current_state,
        next_state,
        batch.gross_spot_return,
        normalized_strikes,
        current_expiry_index=1,
        current_call_prices=data.current_calls[batch.repeated_origin, 1],
    )


def apply_first_line_transport_sequence(
    kernels: Sequence[GenericWeeklyMeasureKernel],
    current_state: torch.Tensor,
    gross_spot_return: torch.Tensor,
    standard_normals: torch.Tensor,
    current_log_spot: torch.Tensor | None = None,
) -> tuple[WeeklyMeasureDraw, torch.Tensor]:
    r"""Draw from a chronological sequence of incremental W2 transports.

    If the accepted first transport is ``T1`` and the next incremental map is
    ``U2``, the updated law is sampled as ``T1(U2(Z))``.  Hence chronological
    kernels are applied in reverse order to the fixed base normal ``Z``.
    """

    if len(kernels) < 1:
        raise ValueError("a transport sequence requires at least one kernel")
    transformed_normals = standard_normals
    total_log_abs_det = torch.zeros(
        standard_normals.shape[:-1],
        dtype=standard_normals.dtype,
        device=standard_normals.device,
    )
    final_draw: WeeklyMeasureDraw | None = None
    for kernel in reversed(kernels):
        final_draw, log_abs_det = kernel.step_first_line_with_log_abs_det(
            current_state,
            gross_spot_return,
            transformed_normals,
            current_log_spot,
        )
        transformed_normals = final_draw.transformed_standard_normals
        total_log_abs_det = total_log_abs_det + log_abs_det
    if final_draw is None:
        raise RuntimeError("transport sequence produced no draw")
    return final_draw, total_log_abs_det


def sequential_population_oce_transport_reverse_kl_loss(
    *,
    incremental_kernel: GenericWeeklyMeasureKernel,
    current_kernels: Sequence[GenericWeeklyMeasureKernel],
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    oce_action: SpotMarginalOCEAction,
    current_log_spot: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""Distil one OCE correction relative to the currently sampled law.

    ``current_kernels`` sample ``Q_r`` directly.  The incremental map ``U`` is
    trained in the base-normal space so the candidate law is
    ``Q_{r+1}=T_r(U(Z))``.  The Jacobian of ``T_r`` cancels from the density
    ratio against ``Q_r``; only ``U`` and the new OCE marginal enter the reverse
    KL objective.
    """

    if current_log_spot.shape != (batch.num_origins,):
        raise ValueError("sequential transport current log spot has invalid shape")
    context_log_spot = torch.repeat_interleave(
        current_log_spot,
        batch.spot_contexts_per_origin,
    )
    path_log_spot = torch.repeat_interleave(
        context_log_spot,
        batch.surface_paths_per_context,
    )
    with torch.no_grad():
        current_draw, _ = apply_first_line_transport_sequence(
            current_kernels,
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
            path_log_spot,
        )
        current_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            current_draw.reduced_state,
            normalized_strikes,
        ).reshape(
            batch.num_origins,
            batch.spot_contexts_per_origin,
            batch.surface_paths_per_context,
            -1,
        )
        origin_state = batch.context_state[:: batch.spot_contexts_per_origin]
        action, _ = oce_action(origin_state, current_log_spot)
        _, target_details = spot_marginal_oce_objective(current_returns, action)
        translation = target_details["translation"]

    incremental_draw, incremental_log_abs_det = (
        incremental_kernel.step_first_line_with_log_abs_det(
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
            path_log_spot,
        )
    )
    candidate_draw, _ = apply_first_line_transport_sequence(
        current_kernels,
        batch.current_state,
        batch.gross_spot_return,
        incremental_draw.transformed_standard_normals,
        path_log_spot,
    )
    candidate_returns = w2_roll_returns(
        simulator,
        data,
        batch,
        candidate_draw.reduced_state,
        normalized_strikes,
    ).reshape(batch.num_contexts, batch.surface_paths_per_context, -1)
    context_action = torch.repeat_interleave(
        action,
        batch.spot_contexts_per_origin,
        dim=0,
    )
    shifted = translation.reshape(-1, 1) + torch.sum(
        candidate_returns * context_action[:, None], dim=-1
    )
    log_marginal = log_powerexp_marginal(shifted).reshape(-1)
    width = incremental_kernel.config.state_sizes_by_expiry[0]
    base = batch.standard_normals[:, 1 : 1 + width]
    transformed = incremental_draw.transformed_standard_normals[:, 1 : 1 + width]
    path_objective = (
        -0.5 * torch.sum(base.square(), dim=-1)
        - incremental_log_abs_det
        + 0.5 * torch.sum(transformed.square(), dim=-1)
        - log_marginal
    )
    return torch.mean(path_objective), {
        "incremental_log_abs_det": incremental_log_abs_det,
        "log_marginal": log_marginal,
        "candidate_returns": candidate_returns,
        "path_objective": path_objective,
    }


def transport_reverse_kl_loss(
    *,
    kernel: GenericWeeklyMeasureKernel,
    reference_kernel: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    oce_action: ConditionalOCEAction,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit a sampler to the OCE marginal-utility tilt by reverse KL."""

    # The OCE translation is defined under P. Re-solve it on the paired P
    # sample instead of normalizing the candidate transport's own draws.
    with torch.no_grad():
        reference_draw, reference_log_abs_det = (
            reference_kernel.step_first_line_with_log_abs_det(
                batch.current_state,
                batch.gross_spot_return,
                batch.standard_normals,
            )
        )
        if not bool(torch.all(reference_log_abs_det == 0.0)):
            raise RuntimeError("reference P transport is not the identity")
        reference_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            reference_draw.reduced_state,
            normalized_strikes,
        ).reshape(batch.num_contexts, batch.surface_paths_per_context, -1)
        action = oce_action(batch.context_state, batch.context_gross_spot_return)
        _, reference_details = conditional_oce_objective(reference_returns, action)
        translation = reference_details["translation"]

    draw, log_abs_det = kernel.step_first_line_with_log_abs_det(
        batch.current_state,
        batch.gross_spot_return,
        batch.standard_normals,
    )
    returns = w2_roll_returns(
        simulator,
        data,
        batch,
        draw.reduced_state,
        normalized_strikes,
    )
    paths = batch.surface_paths_per_context
    shaped_returns = returns.reshape(batch.num_contexts, paths, -1)
    shifted = translation[:, None] + torch.sum(shaped_returns * action[:, None], dim=-1)
    log_marginal = log_powerexp_marginal(shifted).reshape(-1)
    width = kernel.config.state_sizes_by_expiry[0]
    reference = batch.standard_normals[:, 1 : 1 + width]
    transformed = draw.transformed_standard_normals[:, 1 : 1 + width]
    log_reference_density = -0.5 * torch.sum(reference.square(), dim=-1)
    log_transformed_base_density = -0.5 * torch.sum(transformed.square(), dim=-1)
    path_objective = (
        log_reference_density - log_abs_det - log_transformed_base_density - log_marginal
    )
    loss = torch.mean(path_objective)
    return loss, {
        "log_abs_det": log_abs_det,
        "log_marginal": log_marginal,
        "returns": shaped_returns,
        "path_objective": path_objective,
    }


def nested_oce_transport_reverse_kl_loss(
    *,
    kernel: GenericWeeklyMeasureKernel,
    reference_kernel: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit a transport to an OCE density solved directly on its paired P sample."""

    with torch.no_grad():
        reference_draw, reference_log_abs_det = (
            reference_kernel.step_first_line_with_log_abs_det(
                batch.current_state,
                batch.gross_spot_return,
                batch.standard_normals,
            )
        )
        if not bool(torch.all(reference_log_abs_det == 0.0)):
            raise RuntimeError("reference P transport is not the identity")
        reference_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            reference_draw.reduced_state,
            normalized_strikes,
        ).reshape(batch.num_contexts, batch.surface_paths_per_context, -1)
        grouped_reference_returns = reference_returns.reshape(
            batch.num_origins,
            batch.spot_contexts_per_origin,
            batch.surface_paths_per_context,
            -1,
        )
        action, translation, target_details = solve_conditional_powerexp_oce(
            grouped_reference_returns,
            inverse_sqrt_second_moment,
        )

    draw, log_abs_det = kernel.step_first_line_with_log_abs_det(
        batch.current_state,
        batch.gross_spot_return,
        batch.standard_normals,
    )
    transported_returns = w2_roll_returns(
        simulator,
        data,
        batch,
        draw.reduced_state,
        normalized_strikes,
    ).reshape(batch.num_contexts, batch.surface_paths_per_context, -1)
    context_action = torch.repeat_interleave(
        action,
        batch.spot_contexts_per_origin,
        dim=0,
    )
    shifted = translation.reshape(-1, 1) + torch.sum(
        transported_returns * context_action[:, None], dim=-1
    )
    log_marginal = log_powerexp_marginal(shifted).reshape(-1)
    width = kernel.config.state_sizes_by_expiry[0]
    reference = batch.standard_normals[:, 1 : 1 + width]
    transformed = draw.transformed_standard_normals[:, 1 : 1 + width]
    log_reference_density = -0.5 * torch.sum(reference.square(), dim=-1)
    log_transformed_base_density = -0.5 * torch.sum(transformed.square(), dim=-1)
    path_objective = (
        log_reference_density - log_abs_det - log_transformed_base_density - log_marginal
    )
    return torch.mean(path_objective), {
        "log_abs_det": log_abs_det,
        "log_marginal": log_marginal,
        "returns": transported_returns,
        "path_objective": path_objective,
        "target_mass_residual": target_details["mass_residual"],
        "target_action_residual": target_details["action_residual"],
    }


def population_oce_transport_reverse_kl_loss(
    *,
    kernel: GenericWeeklyMeasureKernel,
    reference_kernel: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    oce_action: SpotMarginalOCEAction,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit a transport to a population OCE action with exact spotwise mass."""

    with torch.no_grad():
        reference_draw, reference_log_abs_det = (
            reference_kernel.step_first_line_with_log_abs_det(
                batch.current_state,
                batch.gross_spot_return,
                batch.standard_normals,
            )
        )
        if not bool(torch.all(reference_log_abs_det == 0.0)):
            raise RuntimeError("reference P transport is not the identity")
        reference_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            reference_draw.reduced_state,
            normalized_strikes,
        ).reshape(
            batch.num_origins,
            batch.spot_contexts_per_origin,
            batch.surface_paths_per_context,
            -1,
        )
        origin_state = batch.context_state[:: batch.spot_contexts_per_origin]
        action, _ = oce_action(origin_state)
        _, target_details = spot_marginal_oce_objective(reference_returns, action)
        translation = target_details["translation"]

    draw, log_abs_det = kernel.step_first_line_with_log_abs_det(
        batch.current_state,
        batch.gross_spot_return,
        batch.standard_normals,
    )
    transported_returns = w2_roll_returns(
        simulator,
        data,
        batch,
        draw.reduced_state,
        normalized_strikes,
    ).reshape(batch.num_contexts, batch.surface_paths_per_context, -1)
    context_action = torch.repeat_interleave(
        action,
        batch.spot_contexts_per_origin,
        dim=0,
    )
    shifted = translation.reshape(-1, 1) + torch.sum(
        transported_returns * context_action[:, None], dim=-1
    )
    log_marginal = log_powerexp_marginal(shifted).reshape(-1)
    width = kernel.config.state_sizes_by_expiry[0]
    reference = batch.standard_normals[:, 1 : 1 + width]
    transformed = draw.transformed_standard_normals[:, 1 : 1 + width]
    path_objective = (
        -0.5 * torch.sum(reference.square(), dim=-1)
        - log_abs_det
        + 0.5 * torch.sum(transformed.square(), dim=-1)
        - log_marginal
    )
    return torch.mean(path_objective), {
        "log_abs_det": log_abs_det,
        "log_marginal": log_marginal,
        "returns": transported_returns,
        "path_objective": path_objective,
    }


def summarize_oce_target(
    returns: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, Any]:
    """Summarize the normalized conditional target law on an audit batch."""

    objective, details = conditional_oce_objective(returns, action)
    log_marginal = details["log_marginal"]
    normalized_weight = torch.softmax(log_marginal, dim=1)
    target_moment = torch.sum(normalized_weight[..., None] * returns, dim=1)
    mass_residual = details["mass_residual"]
    effective_sample_size = 1.0 / torch.sum(normalized_weight.square(), dim=1)
    return {
        "oce": float(objective.detach().cpu()),
        "conditional_density_mass_rmse": float(
            torch.sqrt(torch.mean(mass_residual.square())).detach().cpu()
        ),
        "conditional_density_mass_maximum_abs": float(
            torch.max(torch.abs(mass_residual)).detach().cpu()
        ),
        "target_conditional_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(target_moment.square())).detach().cpu()
        )
        * 1.0e4,
        "target_conditional_moment_maximum_abs_bps_spot": float(
            torch.max(torch.abs(target_moment)).detach().cpu()
        )
        * 1.0e4,
        "minimum_marginal_effective_sample_size": float(
            torch.min(effective_sample_size).detach().cpu()
        ),
        "mean_action_rms_units": float(torch.mean(torch.abs(action) * 1.0).detach().cpu()),
        "maximum_abs_translation": float(
            torch.max(torch.abs(details["translation"])).detach().cpu()
        ),
    }


def estimate_return_rms(
    returns: torch.Tensor,
    *,
    minimum_rms: float = 1.0e-4,
) -> torch.Tensor:
    """Return a floored finite-positive preconditioner for every W2 call."""

    if returns.ndim != 3:
        raise ValueError("return RMS input must have conditional path geometry")
    if not math.isfinite(minimum_rms) or minimum_rms <= 0.0:
        raise ValueError("minimum return RMS must be finite-positive")
    rms = torch.sqrt(torch.mean(returns.square(), dim=(0, 1)))
    numerical_floor = 64.0 * torch.finfo(returns.dtype).eps
    if not bool(torch.all(torch.isfinite(rms) & (rms > 0.0))):
        raise ValueError(
            "W2 return RMS is degenerate: "
            f"numerical_floor={numerical_floor:.9g}, "
            f"values={rms.detach().cpu().tolist()}"
        )
    return torch.clamp_min(rms, minimum_rms)


def finite_training_record(**values: float) -> dict[str, float]:
    """Validate and return a compact scalar training record."""

    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"non-finite OCE transport record {values}")
    return values


__all__ = [
    "ConditionalOCEAction",
    "ConditionalOCEConfig",
    "SpotMarginalOCEAction",
    "SpotConditionedFirstLineTransport",
    "ConditionalPathBatch",
    "apply_first_line_transport_sequence",
    "conditional_oce_objective",
    "estimate_return_inverse_sqrt",
    "estimate_return_rms",
    "finite_training_record",
    "log_powerexp_marginal",
    "nested_oce_transport_reverse_kl_loss",
    "powerexp_utility",
    "population_oce_transport_reverse_kl_loss",
    "sample_conditional_paths",
    "sequential_population_oce_transport_reverse_kl_loss",
    "solve_conditional_powerexp_oce",
    "solve_conditional_powerexp_oce_l1",
    "solve_powerexp_translation",
    "spot_marginal_oce_objective",
    "summarize_oce_target",
    "transport_reverse_kl_loss",
    "w2_roll_returns",
]
