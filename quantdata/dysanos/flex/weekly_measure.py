"""Torch-native exact-week target-measure kernels for FLEX dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8, unique_label48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.dysanos.flex.model import JointHierarchicalDynamicsModel
from quantdata.dysanos.flex.weekly_dynamics import TorchWeeklyDynamics
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _validate_identity(label: str, group: str) -> None:
    group_parts = group.split("/")
    verify_inp(
        bool(label) and (label.startswith("*") or is_filename(label)),
        lambda: f"invalid target-kernel label {label!r}",
    )
    verify_inp(
        bool(group_parts)
        and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
        lambda: f"invalid target-kernel group {group!r}",
    )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class WeeklyMeasureConfig:
    """Geometry shared by normal and generic exact-week measure kernels."""

    label: str
    state_sizes_by_expiry: tuple[int, ...]
    hierarchical: bool
    group: str = "flex/weekly-measure"
    user_version: str = "0.1"
    context_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=256,
    )
    maximum_log_scale: float = 2.0
    maximum_shift: float = 5.0
    generic_amplitude_fraction: float = 0.9

    def __post_init__(self) -> None:
        _validate_identity(self.label, self.group)
        verify_inp(
            bool(self.state_sizes_by_expiry)
            and all(value > 0 for value in self.state_sizes_by_expiry),
            lambda: f"invalid measure state blocks {self.state_sizes_by_expiry}",
        )
        verify_inp(
            not self.hierarchical or len(self.state_sizes_by_expiry) > 1,
            lambda: "a hierarchical measure kernel needs at least two expiry blocks",
        )
        verify_inp(
            self.maximum_log_scale > 0.0 and self.maximum_shift > 0.0,
            lambda: "measure scale and shift bounds must be positive",
        )
        verify_inp(
            0.0 < self.generic_amplitude_fraction < 1.0,
            lambda: (
                "generic_amplitude_fraction must lie in (0,1), got "
                f"{self.generic_amplitude_fraction}"
            ),
        )

    @property
    def state_size(self) -> int:
        return sum(self.state_sizes_by_expiry)

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@dataclass(frozen=True)
class WeeklyMeasureDraw:
    """One exact-week target draw and its immutable P reference proposal."""

    reduced_state: torch.Tensor
    reference_reduced_state: torch.Tensor
    transformed_standard_normals: torch.Tensor
    reference_standard_normals: torch.Tensor


class JointHierarchicalWeeklyDynamicsAdapter(BaseModel):
    """Expose the accepted joint Hierarchical-200 P law to weekly Q kernels.

    The jointly trained model has no embedded spot equation, so the weekly P
    regression is supplied explicitly.  Its front-state coefficients and
    orthogonal residual define the joint Gaussian spot/surface covariance.
    The leading Cholesky coordinate is consequently the P-fitted standardized
    spot rank.  Under Q that correlated rank is mapped through the current
    front SANOS inverse CDF and retained unchanged.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        model: JointHierarchicalDynamicsModel,
        spot_front_beta: torch.Tensor,
        spot_orthogonal_std: float,
    ):
        super().__init__(env)
        if model.env is not env:
            raise ValueError("joint model and weekly adapter must share one environment")
        self.config = model.config
        self.reduced_sizes_by_expiry = model.config.state_sizes_by_expiry
        front_width = model.config.state_sizes_by_expiry[0]
        if spot_front_beta.shape != (front_width,) or not bool(
            torch.all(torch.isfinite(spot_front_beta))
        ):
            raise ValueError(
                f"spot beta must be finite with shape ({front_width},)"
            )
        if not math.isfinite(spot_orthogonal_std) or spot_orthogonal_std <= 0.0:
            raise ValueError("spot orthogonal standard deviation must be positive")
        transition = model.weekly_transition().detach().clone()
        surface_factor = model.innovation_cholesky().detach().clone()
        beta = spot_front_beta.to(dtype=env.fdtype, device=env.device).detach()
        spot_cross_surface = surface_factor[:, :front_width] @ beta
        spot_variance = torch.sum(beta.square()) + spot_orthogonal_std**2
        joint_covariance = torch.zeros(
            (model.config.state_size + 1, model.config.state_size + 1),
            dtype=env.fdtype,
            device=env.device,
        )
        joint_covariance[0, 0] = spot_variance
        joint_covariance[0, 1:] = spot_cross_surface
        joint_covariance[1:, 0] = spot_cross_surface
        joint_covariance[1:, 1:] = surface_factor @ surface_factor.T
        self.register_buffer("spot_front_beta", beta)
        self.register_buffer(
            "spot_orthogonal_std",
            torch.as_tensor(
                spot_orthogonal_std,
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        self.register_buffer("spot_innovation_std", torch.sqrt(spot_variance))
        self.register_buffer(
            "equilibrium_mean",
            model.equilibrium_mean.detach().clone(),
        )
        self.register_buffer("weekly_surface_transition", transition)
        self.register_buffer(
            "weekly_joint_covariance",
            joint_covariance,
        )

    @property
    def reduced_state_size(self) -> int:
        """The joint state is already the complete 200-dimensional h state."""
        return self.config.state_size

    def decode_full_state(self, reduced_state: torch.Tensor) -> torch.Tensor:
        """Return h unchanged; no additional PCA layer exists in joint mode."""
        if reduced_state.shape[-1] != self.reduced_state_size:
            raise ValueError(
                f"expected state width {self.reduced_state_size}, "
                f"found {reduced_state.shape}"
            )
        return reduced_state


class _WeeklyMeasureKernel(BaseModel):
    """Common Cholesky proposal and expiry-causal context construction."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        dynamics: TorchWeeklyDynamics | JointHierarchicalWeeklyDynamicsAdapter,
        config: WeeklyMeasureConfig,
    ):
        super().__init__(env)
        if dynamics.env is not env:
            raise ValueError("measure kernel and P dynamics must share one environment")
        if config.state_size != dynamics.reduced_state_size:
            raise ValueError(
                "measure blocks do not match the P dynamics dimension: "
                f"{config.state_size} versus {dynamics.reduced_state_size}"
            )
        if tuple(config.state_sizes_by_expiry) != tuple(
            dynamics.reduced_sizes_by_expiry
        ):
            raise ValueError(
                "measure blocks must equal the fitted per-expiry PCA blocks: "
                f"{config.state_sizes_by_expiry} versus "
                f"{dynamics.reduced_sizes_by_expiry}"
            )
        self.config = config
        self.dynamics = dynamics
        covariance = dynamics.weekly_joint_covariance
        eigenvalues = torch.linalg.eigvalsh(covariance)
        floor = torch.finfo(env.fdtype).eps * torch.clamp_min(
            torch.max(eigenvalues), 1.0
        ) * covariance.shape[0]
        regularized = covariance + torch.clamp_min(floor - torch.min(eigenvalues), 0.0) * (
            torch.eye(covariance.shape[0], dtype=env.fdtype, device=env.device)
        )
        self.register_buffer("reference_joint_cholesky", torch.linalg.cholesky(regularized))
        self._offsets = tuple(
            torch.cumsum(
                torch.tensor((0, *config.state_sizes_by_expiry)), dim=0
            ).tolist()
        )

    @property
    def transformed_block_count(self) -> int:
        return (
            len(self.config.state_sizes_by_expiry) - 1
            if self.config.hierarchical
            else 1
        )

    def _block_slice(self, index: int) -> slice:
        return slice(self._offsets[index], self._offsets[index + 1])

    def _context_width(self, block_index: int) -> int:
        if not self.config.hierarchical:
            return self.config.state_size + 1
        # Line r produces next expiry r from current expiry r+1. Its context
        # includes that complete current prefix, all already transformed lower
        # Gaussian blocks, and the realized implied log-spot return.
        current_width = self._offsets[min(block_index + 2, len(self._offsets) - 1)]
        previous_noise_width = self._offsets[block_index]
        return current_width + previous_noise_width + 1

    def _context(
        self,
        current_state: torch.Tensor,
        transformed_surface_normals: list[torch.Tensor],
        log_gross_spot_return: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if not self.config.hierarchical:
            return torch.cat((current_state, log_gross_spot_return.unsqueeze(-1)), dim=-1)
        current_stop = self._offsets[
            min(block_index + 2, len(self._offsets) - 1)
        ]
        parts = [current_state[..., :current_stop]]
        if transformed_surface_normals:
            parts.extend(transformed_surface_normals)
        parts.append(log_gross_spot_return.unsqueeze(-1))
        context = torch.cat(parts, dim=-1)
        assert context.shape[-1] == self._context_width(block_index), (
            context.shape,
            self._context_width(block_index),
        )
        return context

    def _validate_inputs(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
    ) -> None:
        if current_state.shape[:-1] != gross_spot_return.shape:
            raise ValueError("state and gross spot return batch shapes differ")
        if current_state.shape[-1] != self.config.state_size:
            raise ValueError("current state has the wrong target-kernel width")
        if standard_normals.shape != current_state.shape[:-1] + (
            self.config.state_size + 1,
        ):
            raise ValueError("standard normals have the wrong target-kernel geometry")
        if torch.any(gross_spot_return <= 0.0):
            raise ValueError("gross spot returns must be positive")

    def _assemble_draw(
        self,
        current_state: torch.Tensor,
        reference_standard_normals: torch.Tensor,
        transformed_surface_normals: list[torch.Tensor],
    ) -> WeeklyMeasureDraw:
        if self.config.hierarchical:
            boundary_index = len(self.config.state_sizes_by_expiry) - 1
            boundary_slice = self._block_slice(boundary_index)
            transformed_surface_normals.append(
                reference_standard_normals[..., 1:][..., boundary_slice]
            )
        transformed = torch.cat(
            (reference_standard_normals[..., :1], *transformed_surface_normals),
            dim=-1,
        )
        reference_innovation = reference_standard_normals @ (
            self.reference_joint_cholesky.T
        )
        transformed_innovation = transformed @ self.reference_joint_cholesky.T
        transition = self.dynamics.weekly_surface_transition
        reference_mean = self.dynamics.equilibrium_mean + (
            current_state - self.dynamics.equilibrium_mean
        ) @ transition
        reference_state = reference_mean + reference_innovation[..., 1:]
        transformed_state = reference_mean + transformed_innovation[..., 1:]
        self.verify(
            lambda: torch.all(torch.isfinite(transformed_state)),
            "non-finite target-measure state draw",
        )
        return WeeklyMeasureDraw(
            reduced_state=transformed_state,
            reference_reduced_state=reference_state,
            transformed_standard_normals=transformed,
            reference_standard_normals=reference_standard_normals,
        )


class NormalWeeklyMeasureKernel(_WeeklyMeasureKernel):
    """Conditional Gaussian weekly law with learnable mean and covariance."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        dynamics: TorchWeeklyDynamics | JointHierarchicalWeeklyDynamicsAdapter,
        config: WeeklyMeasureConfig,
    ):
        super().__init__(env, dynamics=dynamics, config=config)
        networks: list[FeedForward] = []
        local_factors: list[torch.nn.Parameter] = []
        for block_index in range(self.transformed_block_count):
            width = config.state_sizes_by_expiry[block_index]
            network = FeedForward(
                env,
                self._context_width(block_index),
                2 * width,
                config.context_network,
            )
            _zero_final_layer(network)
            networks.append(network)
            local_factors.append(
                torch.nn.Parameter(
                    torch.zeros((width, width), dtype=env.fdtype, device=env.device)
                )
            )
        self.context_networks = torch.nn.ModuleList(networks)
        self.local_factor_raw = torch.nn.ParameterList(local_factors)

    def step(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
    ) -> WeeklyMeasureDraw:
        """Draw from the exact-week normal target law without path weights."""
        self._validate_inputs(current_state, gross_spot_return, standard_normals)
        reference_surface = standard_normals[..., 1:]
        transformed_blocks: list[torch.Tensor] = []
        log_gross = torch.log(gross_spot_return)
        for block_index, (network, raw_factor) in enumerate(
            zip(self.context_networks, self.local_factor_raw, strict=True)
        ):
            block_slice = self._block_slice(block_index)
            width = block_slice.stop - block_slice.start
            context = self._context(
                current_state,
                transformed_blocks,
                log_gross,
                block_index,
            )
            mean_raw, log_scale_raw = torch.split(network(context), width, dim=-1)
            mean = self.config.maximum_shift * torch.tanh(
                mean_raw / self.config.maximum_shift
            )
            log_scale = self.config.maximum_log_scale * torch.tanh(
                log_scale_raw / self.config.maximum_log_scale
            )
            lower = torch.tril(raw_factor, diagonal=-1) + torch.diag_embed(
                torch.exp(torch.diagonal(raw_factor))
            )
            mixed = reference_surface[..., block_slice] @ lower.T
            transformed_blocks.append(mean + torch.exp(log_scale) * mixed)
        return self._assemble_draw(
            current_state,
            standard_normals,
            transformed_blocks,
        )

    def regularization_loss(self, current_state: torch.Tensor) -> torch.Tensor:
        """Penalize departures from the fitted P proposal at current states."""
        zero_gross = torch.zeros(current_state.shape[:-1], dtype=self.fdtype, device=self.device)
        prior_blocks: list[torch.Tensor] = []
        terms: list[torch.Tensor] = []
        for block_index, (network, raw_factor) in enumerate(
            zip(self.context_networks, self.local_factor_raw, strict=True)
        ):
            width = self.config.state_sizes_by_expiry[block_index]
            context = self._context(
                current_state,
                prior_blocks,
                zero_gross,
                block_index,
            )
            output = network(context)
            terms.append(torch.mean(output.square()))
            terms.append(torch.mean(raw_factor.square()))
            prior_blocks.append(
                torch.zeros(
                    current_state.shape[:-1] + (width,),
                    dtype=self.fdtype,
                    device=self.device,
                )
            )
        return torch.stack(terms).mean()

    def conditional_kl_loss(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        *,
        line_index: int,
    ) -> torch.Tensor:
        """Return conditional Gaussian ``KL(Q_line || P_line)`` per coordinate.

        This is evaluated in the ordered standard-normal coordinates before
        the frozen P Cholesky factor.  For the first roll line the context is
        exactly current ``h^1,h^2`` and realized implied spot.  Later lines are
        deliberately unsupported here because their contexts also contain
        random already-transformed blocks; the current Q1 experiment does not
        fit them.
        """
        if line_index != 0:
            raise ValueError("conditional Gaussian KL is currently defined for Q1 only")
        if current_state.shape[:-1] != gross_spot_return.shape:
            raise ValueError("state and gross spot return batch shapes differ")
        width = self.config.state_sizes_by_expiry[0]
        context = self._context(
            current_state,
            [],
            torch.log(gross_spot_return),
            0,
        )
        mean_raw, log_scale_raw = torch.split(
            self.context_networks[0](context), width, dim=-1
        )
        mean = self.config.maximum_shift * torch.tanh(
            mean_raw / self.config.maximum_shift
        )
        log_scale = self.config.maximum_log_scale * torch.tanh(
            log_scale_raw / self.config.maximum_log_scale
        )
        raw_factor = self.local_factor_raw[0]
        lower = torch.tril(raw_factor, diagonal=-1) + torch.diag_embed(
            torch.exp(torch.diagonal(raw_factor))
        )
        # The row-vector transform is z L' diag(exp(log_scale)); its covariance
        # trace is the squared Frobenius norm of diag(exp(log_scale)) L.
        scaled_lower = torch.exp(log_scale).unsqueeze(-1) * lower
        covariance_trace = torch.sum(scaled_lower.square(), dim=(-2, -1))
        log_determinant_factor = torch.sum(log_scale, dim=-1) + torch.sum(
            torch.diagonal(raw_factor)
        )
        kl = 0.5 * (
            covariance_trace
            + torch.sum(mean.square(), dim=-1)
            - width
            - 2.0 * log_determinant_factor
        )
        return torch.mean(kl) / width


class GenericWeeklyMeasureKernel(_WeeklyMeasureKernel):
    r"""Expiry-triangular nonlinear full-support transport of P Gaussian noise.

    For every coordinate, ``F(z)=shift+scale*z+amplitude*tanh(z)+L*z`` with
    strictly-lower ``L`` and ``scale>|amplitude|``. The requested blend
    ``q=(1-g)z+gF(z)`` therefore has a strictly positive triangular Jacobian,
    remains onto ``R^d``, and starts exactly at the identity.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        dynamics: TorchWeeklyDynamics | JointHierarchicalWeeklyDynamicsAdapter,
        config: WeeklyMeasureConfig,
    ):
        super().__init__(env, dynamics=dynamics, config=config)
        networks: list[FeedForward] = []
        strict_lower: list[torch.nn.Parameter] = []
        for block_index in range(self.transformed_block_count):
            width = config.state_sizes_by_expiry[block_index]
            network = FeedForward(
                env,
                self._context_width(block_index),
                4 * width,
                config.context_network,
            )
            _zero_final_layer(network)
            networks.append(network)
            strict_lower.append(
                torch.nn.Parameter(
                    torch.zeros((width, width), dtype=env.fdtype, device=env.device)
                )
            )
        self.context_networks = torch.nn.ModuleList(networks)
        self.strict_lower_raw = torch.nn.ParameterList(strict_lower)

    def step(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
    ) -> WeeklyMeasureDraw:
        """Apply the learned bijective nonlinear weekly transport."""
        self._validate_inputs(current_state, gross_spot_return, standard_normals)
        reference_surface = standard_normals[..., 1:]
        transformed_blocks: list[torch.Tensor] = []
        log_gross = torch.log(gross_spot_return)
        for block_index, (network, raw_lower) in enumerate(
            zip(self.context_networks, self.strict_lower_raw, strict=True)
        ):
            block_slice = self._block_slice(block_index)
            width = block_slice.stop - block_slice.start
            context = self._context(
                current_state,
                transformed_blocks,
                log_gross,
                block_index,
            )
            shift_raw, scale_raw, amplitude_raw, gate_raw = torch.split(
                network(context), width, dim=-1
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
            reference_block = reference_surface[..., block_slice]
            lower_effect = reference_block @ torch.tril(raw_lower, diagonal=-1).T
            proposal = (
                shift
                + scale * reference_block
                + amplitude * torch.tanh(reference_block)
                + lower_effect
            )
            transformed_blocks.append((1.0 - gate) * reference_block + gate * proposal)
        return self._assemble_draw(
            current_state,
            standard_normals,
            transformed_blocks,
        )

    def minimum_coordinate_derivative(
        self,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
    ) -> torch.Tensor:
        """Return the analytic lower bound on the triangular Jacobian diagonal."""
        self._validate_inputs(current_state, gross_spot_return, standard_normals)
        transformed_blocks: list[torch.Tensor] = []
        reference_surface = standard_normals[..., 1:]
        bounds: list[torch.Tensor] = []
        for block_index, (network, raw_lower) in enumerate(
            zip(self.context_networks, self.strict_lower_raw, strict=True)
        ):
            block_slice = self._block_slice(block_index)
            width = block_slice.stop - block_slice.start
            context = self._context(
                current_state,
                transformed_blocks,
                torch.log(gross_spot_return),
                block_index,
            )
            shift_raw, scale_raw, amplitude_raw, gate_raw = torch.split(
                network(context), width, dim=-1
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
            bounds.append((1.0 - gate) + gate * (scale - torch.abs(amplitude)))
            reference_block = reference_surface[..., block_slice]
            shift = self.config.maximum_shift * torch.tanh(
                shift_raw / self.config.maximum_shift
            )
            proposal = (
                shift
                + scale * reference_block
                + amplitude * torch.tanh(reference_block)
                + reference_block @ torch.tril(raw_lower, diagonal=-1).T
            )
            transformed_blocks.append(
                (1.0 - gate) * reference_block + gate * proposal
            )
        return torch.min(torch.cat(bounds, dim=-1))

    def regularization_loss(self, current_state: torch.Tensor) -> torch.Tensor:
        """Penalize the transport itself rather than dormant hidden weights."""
        zero_log_gross = torch.zeros(
            current_state.shape[:-1], dtype=self.fdtype, device=self.device
        )
        prior_blocks: list[torch.Tensor] = []
        terms: list[torch.Tensor] = []
        for block_index, (network, raw_lower) in enumerate(
            zip(self.context_networks, self.strict_lower_raw, strict=True)
        ):
            width = self.config.state_sizes_by_expiry[block_index]
            context = self._context(
                current_state,
                prior_blocks,
                zero_log_gross,
                block_index,
            )
            output = network(context)
            terms.append(torch.mean(output.square()))
            terms.append(torch.mean(raw_lower.square()))
            prior_blocks.append(
                torch.zeros(
                    current_state.shape[:-1] + (width,),
                    dtype=self.fdtype,
                    device=self.device,
                )
            )
        return torch.stack(terms).mean()


def _zero_final_layer(network: FeedForward) -> None:
    """Initialize both target kernels as the exact identity P proposal."""
    final = network.I0 if len(network.In) == 0 else network.In[-1]
    with torch.no_grad():
        final.weight.zero_()
        if final.bias is not None:
            final.bias.zero_()


__all__ = [
    "GenericWeeklyMeasureKernel",
    "JointHierarchicalWeeklyDynamicsAdapter",
    "NormalWeeklyMeasureKernel",
    "WeeklyMeasureConfig",
    "WeeklyMeasureDraw",
]
