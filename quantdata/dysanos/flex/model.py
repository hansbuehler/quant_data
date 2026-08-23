"""Native and compressed daily states for the fixed FLEX SANOS decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FixedEtaFlexGeometryConfig,
    FlexGeometryConfig,
    ForwardMuFlexGeometryConfig,
    JointDirectDynamicsConfig,
    JointHierarchicalDynamicsConfig,
    LatentDecoderConfig,
    NativeStateConfig,
    PcaDecoderConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward


@dataclass(frozen=True)
class NativeParameters:
    """Bounded native coordinates for one day or a batch of days."""

    dlvsqs: torch.Tensor
    forward_vols: torch.Tensor
    eta: torch.Tensor


def bound_native_raw(
    raw: torch.Tensor,
    *,
    geometry: FlexGeometryConfig,
) -> NativeParameters:
    """Map a raw native vector to the exact bounded FLEX coordinates."""
    T = geometry.num_model_expiries
    N = geometry.num_inner_strikes
    if raw.shape[-1] != geometry.native_state_size:
        raise ValueError(
            "Expected native raw state ending in "
            f"{geometry.native_state_size}, found {raw.shape}"
        )
    if isinstance(geometry, ForwardMuFlexGeometryConfig):
        expiry_blocks = raw.reshape(*raw.shape[:-1], T, N + 2)
        dlvsq_raw = expiry_blocks[..., :N]
        forward_probability = torch.sigmoid(expiry_blocks[..., N])
        eta_probability = torch.sigmoid(expiry_blocks[..., N + 1])
    else:
        probability = torch.sigmoid(raw)
        dlv_stop = T * N
        dlvsq_probability = probability[..., :dlv_stop].reshape(
            *raw.shape[:-1],
            T,
            N,
        )
        forward_probability = probability[..., dlv_stop : dlv_stop + T]
        eta_probability = probability[..., -1]
    if isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        # Clamp in the log coordinate before exponentiation.  This is exactly
        # min(min_dlvsq + exp(z), max_dlvsq), avoids overflow, and deliberately
        # leaves the price gradient flat beyond the cap; the separate smooth
        # cap barrier supplies the restoring gradient there.
        maximum_log_dlvsq = torch.as_tensor(
            geometry.maximum_log_dlvsq,
            dtype=raw.dtype,
            device=raw.device,
        )
        dlvsqs = geometry.min_dlvsq + torch.exp(
            torch.minimum(dlvsq_raw, maximum_log_dlvsq)
        )
    else:
        if isinstance(geometry, ForwardMuFlexGeometryConfig):
            dlvsq_probability = torch.sigmoid(dlvsq_raw)
        dlvsqs = geometry.min_dlvsq + (
            geometry.max_dlvsq - geometry.min_dlvsq
        ) * dlvsq_probability
    forward_vols = forward_probability
    forward_vols = geometry.min_forward_vol + (
        geometry.max_forward_vol - geometry.min_forward_vol
    ) * forward_vols
    eta = (
        torch.full_like(eta_probability, geometry.fixed_eta)
        if isinstance(geometry, FixedEtaFlexGeometryConfig)
        else geometry.min_eta
        + (geometry.max_eta - geometry.min_eta) * eta_probability
    )
    return NativeParameters(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )


def dlvsq_cap_barrier(
    raw: torch.Tensor,
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
) -> torch.Tensor:
    """Return the smooth one-sided cap residual for every native DLV node."""
    T = geometry.num_model_expiries
    N = geometry.num_inner_strikes
    if raw.shape[-1] != geometry.native_state_size:
        raise ValueError(
            "Expected native raw state ending in "
            f"{geometry.native_state_size}, found {raw.shape}"
        )
    dlvsq_raw = raw.reshape(*raw.shape[:-1], T, N + 2)[..., :N]
    scaled = (
        dlvsq_raw - geometry.maximum_log_dlvsq
    ) / geometry.dlv_cap_barrier_log_width
    return (
        geometry.dlv_cap_barrier_log_width
        * torch.nn.functional.softplus(scaled)
        / geometry.dlv_cap_barrier_log_std
    )


def dlvsq_cap_penalty(
    raw: torch.Tensor,
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
) -> torch.Tensor:
    """Return one AAE-equivalent half-squared cap penalty per surface."""
    barrier = dlvsq_cap_barrier(raw, geometry=geometry)
    return 0.5 * torch.sum(barrier.square(), dim=(-2, -1))


def raw_from_native_parameters(
    parameters: NativeParameters,
    *,
    geometry: FlexGeometryConfig,
    probability_floor: float = 1.0e-6,
) -> torch.Tensor:
    """Invert sigmoid bounds for market initialization and supervision targets."""
    if not 0.0 < probability_floor < 0.5:
        raise ValueError(
            "'probability_floor' must lie in (0,0.5), found "
            f"{probability_floor}"
        )
    T = geometry.num_model_expiries
    N = geometry.num_inner_strikes
    if parameters.dlvsqs.shape[-2:] != (T, N):
        raise ValueError(
            f"Expected DLV shape ending in {(T, N)}, found "
            f"{parameters.dlvsqs.shape}"
        )
    if parameters.forward_vols.shape != parameters.dlvsqs.shape[:-2] + (T,):
        raise ValueError("Forward-vol shape does not match the DLV batch shape")
    expected_eta_shape = (
        parameters.dlvsqs.shape[:-2] + (T,)
        if isinstance(geometry, ForwardMuFlexGeometryConfig)
        else parameters.dlvsqs.shape[:-2]
    )
    if parameters.eta.shape != expected_eta_shape:
        raise ValueError(
            "Backbone-multiplier shape does not match the geometry: "
            f"{parameters.eta.shape} versus {expected_eta_shape}"
        )

    dlv_probability = (
        (parameters.dlvsqs - geometry.min_dlvsq)
        / (geometry.max_dlvsq - geometry.min_dlvsq)
    )
    forward_probability = (
        (parameters.forward_vols - geometry.min_forward_vol)
        / (geometry.max_forward_vol - geometry.min_forward_vol)
    )
    eta_probability = (
        torch.full_like(
            parameters.eta,
            (geometry.fixed_eta - geometry.min_eta)
            / (geometry.max_eta - geometry.min_eta),
        )
        if isinstance(geometry, FixedEtaFlexGeometryConfig)
        else (parameters.eta - geometry.min_eta)
        / (geometry.max_eta - geometry.min_eta)
    )
    if isinstance(geometry, ForwardMuFlexGeometryConfig):
        bounded_probability = torch.cat(
            (
                forward_probability.unsqueeze(-1),
                eta_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        bounded_probability = torch.clamp(
            bounded_probability,
            min=probability_floor,
            max=1.0 - probability_floor,
        )
        bounded_raw = torch.logit(bounded_probability)
        if isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
            # The upper endpoint is finite in the log-DLV² coordinate.  Only
            # the strictly positive distance from the lower endpoint needs a
            # numerical floor for a finite inverse.
            dlv_distance = torch.clamp(
                parameters.dlvsqs - geometry.min_dlvsq,
                min=probability_floor
                * (geometry.max_dlvsq - geometry.min_dlvsq),
            )
            dlv_raw = torch.log(dlv_distance)
        else:
            dlv_probability = torch.clamp(
                dlv_probability,
                min=probability_floor,
                max=1.0 - probability_floor,
            )
            dlv_raw = torch.logit(dlv_probability)
        return torch.cat((dlv_raw, bounded_raw), dim=-1).flatten(start_dim=-2)
    else:
        probability = torch.cat(
            (
                dlv_probability.flatten(start_dim=-2),
                forward_probability,
                eta_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
    probability = torch.clamp(
        probability,
        min=probability_floor,
        max=1.0 - probability_floor,
    )
    return torch.logit(probability)


def forward_vols_from_total_vols(
    total_vols: torch.Tensor,
    *,
    geometry: FlexGeometryConfig,
) -> torch.Tensor:
    """Invert FLEX's cumulative square-root-time total-vol definition."""
    T = geometry.num_model_expiries
    if total_vols.shape[-1] != T:
        raise ValueError(
            f"Expected total-vol shape ending in {T}, found {total_vols.shape}"
        )
    model_expiries = torch.as_tensor(
        geometry.model_expiries_y,
        dtype=total_vols.dtype,
        device=total_vols.device,
    )
    dts = torch.diff(
        model_expiries,
        prepend=torch.zeros_like(model_expiries[:1]),
    )
    increments = torch.diff(
        total_vols,
        dim=-1,
        prepend=torch.zeros_like(total_vols[..., :1]),
    )
    return increments / torch.sqrt(dts)


class NativeStateSanosModel(BaseModel):
    """One independent bounded raw state per market day."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_days: int,
        config: NativeStateConfig,
    ):
        super().__init__(env)
        if num_days <= 0:
            raise ValueError(f"'num_days' must be positive, found {num_days}")
        self.config = config
        self.num_days = num_days
        self.sanos_decoder = FlexSanosDecoder(env, geometry=config.geometry)

        geometry = config.geometry
        initial_dlvsqs = torch.full(
            (num_days, geometry.num_model_expiries, geometry.num_inner_strikes),
            0.20**2,
            dtype=env.fdtype,
            device=env.device,
        )
        initial_forward_vols = torch.full(
            (num_days, geometry.num_model_expiries),
            0.20,
            dtype=env.fdtype,
            device=env.device,
        )
        initial_eta_shape = (
            (num_days, geometry.num_model_expiries)
            if isinstance(geometry, ForwardMuFlexGeometryConfig)
            else (num_days,)
        )
        initial_eta = torch.full(
            initial_eta_shape,
            0.49,
            dtype=env.fdtype,
            device=env.device,
        )
        initial_raw = raw_from_native_parameters(
            NativeParameters(
                dlvsqs=initial_dlvsqs,
                forward_vols=initial_forward_vols,
                eta=initial_eta,
            ),
            geometry=geometry,
        )
        self.daily_raw_embedding = torch.nn.Embedding(
            num_days,
            geometry.native_state_size,
            sparse=True,
            dtype=env.fdtype,
            device=env.device,
        )
        with torch.no_grad():
            self.daily_raw_embedding.weight.copy_(initial_raw)

    @property
    def daily_raw_states(self) -> torch.Tensor:
        """Expose the embedding table as the direct per-day raw state matrix."""
        return self.daily_raw_embedding.weight

    def parameters_for_days(self, day_indices: torch.Tensor) -> NativeParameters:
        """Return bounded parameters for the selected days."""
        return bound_native_raw(
            self.daily_raw_embedding(day_indices),
            geometry=self.config.geometry,
        )

    def decode_days(
        self,
        day_indices: torch.Tensor,
    ) -> tuple[NativeParameters, SanosSurface]:
        """Decode selected daily native states on the shared fixed geometry."""
        parameters = self.parameters_for_days(day_indices)
        surface = self.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        return parameters, surface

    def forward(
        self,
        *,
        day_indices: torch.Tensor,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
        is_call: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeParameters, SanosSurface]:
        """Price a padded batch of real call and put books."""
        parameters, surface = self.decode_days(day_indices)
        call_prices = self.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiries_y,
            strikes=strikes,
        )
        option_prices = torch.where(
            is_call,
            call_prices,
            call_prices - 1.0 + strikes,
        )
        return option_prices, parameters, surface


class LatentStateSanosModel(BaseModel):
    """Shared feed-forward map from a lower daily state to native raw state."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_days: int,
        config: (
            LatentDecoderConfig
            | PcaDecoderConfig
            | TriangularPcaDecoderConfig
        ),
    ):
        super().__init__(env)
        if num_days <= 0:
            raise ValueError(f"'num_days' must be positive, found {num_days}")
        self.config = config
        self.num_days = num_days
        self.raw_decoder = FeedForward(
            env,
            num_input=config.decoder_input_size,
            num_output=config.geometry.native_state_size,
            config=config.decoder_network,
        )
        if isinstance(config, TriangularPcaDecoderConfig):
            block_width = config.geometry.expiry_block_state_size
            weight_mask = torch.zeros(
                (config.geometry.native_state_size, config.state_size),
                dtype=env.fdtype,
                device=env.device,
            )
            state_stop = 0
            for expiry_index, state_width in enumerate(
                config.state_sizes_by_expiry
            ):
                state_stop += state_width
                output_start = expiry_index * block_width
                weight_mask[
                    output_start : output_start + block_width,
                    :state_stop,
                ] = 1.0
            self.register_buffer("decoder_weight_mask", weight_mask)
        self.sanos_decoder = FlexSanosDecoder(env, geometry=config.geometry)
        states = torch.empty(
            (num_days, config.state_size),
            dtype=env.fdtype,
            device=env.device,
        )
        torch.nn.init.normal_(states, mean=0.0, std=0.05)
        self.daily_state_embedding = torch.nn.Embedding(
            num_days,
            config.state_size,
            sparse=True,
            dtype=env.fdtype,
            device=env.device,
        )
        with torch.no_grad():
            self.daily_state_embedding.weight.copy_(states)

    @property
    def daily_states(self) -> torch.Tensor:
        """Expose the direct per-day lower-state parameter matrix."""
        return self.daily_state_embedding.weight

    def raw_states_for_days(self, day_indices: torch.Tensor) -> torch.Tensor:
        """Decode selected lower states into raw native vectors."""
        return self.raw_states_from_states(self.daily_state_embedding(day_indices))

    def raw_states_from_states(self, state: torch.Tensor) -> torch.Tensor:
        """Decode arbitrary lower states without consulting the day embedding.

        This is the path used by the weekly dynamics model: simulated states
        remain Torch tensors and retain gradients through the calibrated SANOS
        affine decoder.
        """
        if state.shape[-1] != self.config.state_size:
            raise ValueError(
                "Expected lower state ending in "
                f"{self.config.state_size}, found {tuple(state.shape)}"
            )
        decoder_input = torch.nn.functional.pad(
            state,
            (0, self.config.decoder_input_size - self.config.state_size),
        )
        if isinstance(self.config, TriangularPcaDecoderConfig):
            linear = self.raw_decoder.I0
            return torch.nn.functional.linear(
                decoder_input,
                linear.weight * self.decoder_weight_mask,
                linear.bias,
            )
        return self.raw_decoder(decoder_input)

    def decode_days(
        self,
        day_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeParameters, SanosSurface]:
        """Decode selected lower states through the Torch SANOS recursion."""
        raw = self.raw_states_for_days(day_indices)
        parameters = bound_native_raw(raw, geometry=self.config.geometry)
        surface = self.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        return raw, parameters, surface

    def forward(
        self,
        *,
        day_indices: torch.Tensor,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
        is_call: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, NativeParameters, SanosSurface]:
        """Price a padded batch through the lower-dimensional decoder."""
        raw, parameters, surface = self.decode_days(day_indices)
        call_prices = self.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiries_y,
            strikes=strikes,
        )
        option_prices = torch.where(
            is_call,
            call_prices,
            call_prices - 1.0 + strikes,
        )
        return option_prices, raw, parameters, surface


class JointDirectDynamicsModel(BaseModel):
    """Jointly learned affine SANOS factors and stable weekly Direct dynamics."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_days: int,
        config: JointDirectDynamicsConfig,
    ):
        super().__init__(env)
        if num_days <= 0:
            raise ValueError(f"'num_days' must be positive, found {num_days}")
        self.config = config
        self.num_days = num_days
        self.raw_decoder = FeedForward(
            env,
            num_input=config.state_size,
            num_output=config.geometry.native_state_size,
            config=config.decoder_network,
        )
        self.sanos_decoder = FlexSanosDecoder(env, geometry=config.geometry)
        self.daily_state_embedding = torch.nn.Embedding(
            num_days,
            config.state_size,
            sparse=False,
            dtype=env.fdtype,
            device=env.device,
        )
        torch.nn.init.normal_(self.daily_state_embedding.weight, mean=0.0, std=0.05)

        # K=LL'+S+k_min I has strictly positive numerical abscissa.  S is
        # skew-symmetric and therefore changes rotations/couplings without
        # weakening the stationary mean-reversion guarantee.
        self.damping_factor = torch.nn.Parameter(
            torch.zeros(
                (config.state_size, config.state_size),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.rotation_raw = torch.nn.Parameter(
            torch.zeros_like(self.damping_factor)
        )
        self.equilibrium_mean = torch.nn.Parameter(
            torch.zeros(
                config.state_size,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.innovation_cholesky_raw = torch.nn.Parameter(
            torch.zeros_like(self.damping_factor)
        )

    @property
    def daily_states(self) -> torch.Tensor:
        """Expose the free training-date factors used by both fitted losses."""
        return self.daily_state_embedding.weight

    def raw_states_from_states(self, state: torch.Tensor) -> torch.Tensor:
        """Decode arbitrary Direct factors into unconstrained SANOS coordinates."""
        if state.shape[-1] != self.config.state_size:
            raise ValueError(
                f"Expected state width {self.config.state_size}, found {state.shape}"
            )
        return self.raw_decoder(state)

    def raw_states_for_days(self, day_indices: torch.Tensor) -> torch.Tensor:
        """Decode selected training-date factors."""
        return self.raw_states_from_states(self.daily_state_embedding(day_indices))

    def mean_reversion_generator(self) -> torch.Tensor:
        """Return a real full non-diagonal generator with stable eigenvalues."""
        lower = torch.tril(self.damping_factor)
        damping = lower @ lower.transpose(-1, -2)
        identity = torch.eye(
            self.config.state_size,
            dtype=self.fdtype,
            device=self.device,
        )
        rotation = 0.5 * (
            self.rotation_raw - self.rotation_raw.transpose(-1, -2)
        )
        return (
            damping
            + self.config.minimum_mean_reversion_per_week * identity
            + rotation
        )

    def weekly_transition(self) -> torch.Tensor:
        """Return the exact one-week matrix exponential used in simulation."""
        return torch.matrix_exp(-self.mean_reversion_generator())

    def innovation_cholesky(self) -> torch.Tensor:
        """Return a nonsingular full weekly innovation factor."""
        raw = self.innovation_cholesky_raw
        diagonal = torch.diagonal(raw)
        positive_diagonal = self.config.innovation_std_floor + (
            torch.nn.functional.softplus(diagonal)
        )
        return torch.tril(raw, diagonal=-1) + torch.diag(positive_diagonal)

    def decode_days(
        self,
        day_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeParameters, SanosSurface]:
        """Decode selected factors through the Torch SANOS recursion."""
        raw = self.raw_states_for_days(day_indices)
        parameters = bound_native_raw(raw, geometry=self.config.geometry)
        surface = self.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        return raw, parameters, surface

    def forward(
        self,
        *,
        day_indices: torch.Tensor,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
        is_call: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, NativeParameters, SanosSurface]:
        """Price a padded option batch through the jointly learned factors."""
        raw, parameters, surface = self.decode_days(day_indices)
        call_prices = self.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiries_y,
            strikes=strikes,
        )
        option_prices = torch.where(
            is_call,
            call_prices,
            call_prices - 1.0 + strikes,
        )
        return option_prices, raw, parameters, surface


class JointHierarchicalDynamicsModel(BaseModel):
    """Joint expiry-causal affine decoder and stable weekly P dynamics."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_days: int,
        config: JointHierarchicalDynamicsConfig,
    ):
        super().__init__(env)
        if num_days <= 0:
            raise ValueError(f"'num_days' must be positive, found {num_days}")
        self.config = config
        self.num_days = num_days
        self.raw_decoder = FeedForward(
            env,
            num_input=config.state_size,
            num_output=config.geometry.native_state_size,
            config=config.decoder_network,
        )
        self.sanos_decoder = FlexSanosDecoder(env, geometry=config.geometry)
        self.daily_state_embedding = torch.nn.Embedding(
            num_days,
            config.state_size,
            sparse=False,
            dtype=env.fdtype,
            device=env.device,
        )
        torch.nn.init.normal_(self.daily_state_embedding.weight, mean=0.0, std=0.05)

        block_width = config.geometry.expiry_block_state_size
        decoder_mask = torch.zeros(
            (config.geometry.native_state_size, config.state_size),
            dtype=env.fdtype,
            device=env.device,
        )
        generator_diagonal_mask = torch.zeros(
            (config.state_size, config.state_size),
            dtype=env.fdtype,
            device=env.device,
        )
        generator_strict_upper_mask = torch.zeros_like(generator_diagonal_mask)
        state_stops = np.cumsum((0, *config.state_sizes_by_expiry))
        for expiry_index in range(config.geometry.num_model_expiries):
            state_start = int(state_stops[expiry_index])
            state_stop = int(state_stops[expiry_index + 1])
            output_start = expiry_index * block_width
            decoder_mask[
                output_start : output_start + block_width,
                :state_stop,
            ] = 1.0
            generator_diagonal_mask[
                state_start:state_stop,
                state_start:state_stop,
            ] = 1.0
            generator_strict_upper_mask[:state_start, state_start:state_stop] = 1.0
        self.register_buffer("decoder_weight_mask", decoder_mask)
        self.register_buffer("generator_diagonal_mask", generator_diagonal_mask)
        self.register_buffer(
            "generator_strict_upper_mask",
            generator_strict_upper_mask,
        )

        # Diagonal blocks determine all eigenvalues and are dissipative.  The
        # unrestricted strict-upper blocks couple earlier expiries into later
        # output states without breaking either stationarity or expiry causality.
        matrix = torch.zeros(
            (config.state_size, config.state_size),
            dtype=env.fdtype,
            device=env.device,
        )
        self.damping_factor = torch.nn.Parameter(matrix.clone())
        self.rotation_raw = torch.nn.Parameter(matrix.clone())
        self.causal_coupling_raw = torch.nn.Parameter(matrix.clone())
        self.equilibrium_mean = torch.nn.Parameter(
            torch.zeros(config.state_size, dtype=env.fdtype, device=env.device)
        )
        # A lower Cholesky factor gives the ordered hierarchical construction:
        # innovation block j can use shocks from blocks 1,...,j.
        self.innovation_cholesky_raw = torch.nn.Parameter(matrix.clone())

    @property
    def daily_states(self) -> torch.Tensor:
        """Expose the free training-date hierarchical factors."""
        return self.daily_state_embedding.weight

    @property
    def state_slices(self) -> tuple[slice, ...]:
        """Return ordered expiry-block slices in the compressed state."""
        stops = np.cumsum((0, *self.config.state_sizes_by_expiry))
        return tuple(
            slice(int(stops[index]), int(stops[index + 1]))
            for index in range(len(self.config.state_sizes_by_expiry))
        )

    def effective_decoder_weight(self) -> torch.Tensor:
        """Return the affine decoder weight with expiry causality enforced."""
        return self.raw_decoder.I0.weight * self.decoder_weight_mask

    def raw_states_from_states(self, state: torch.Tensor) -> torch.Tensor:
        """Decode factors so raw expiry j uses only factor blocks 1,...,j."""
        if state.shape[-1] != self.config.state_size:
            raise ValueError(
                f"Expected state width {self.config.state_size}, found {state.shape}"
            )
        return torch.nn.functional.linear(
            state,
            self.effective_decoder_weight(),
            self.raw_decoder.I0.bias,
        )

    def raw_states_for_days(self, day_indices: torch.Tensor) -> torch.Tensor:
        """Decode selected training-date factors."""
        return self.raw_states_from_states(self.daily_state_embedding(day_indices))

    def mean_reversion_generator(self) -> torch.Tensor:
        """Return a stable block-upper generator in row-vector convention."""
        blocks: list[torch.Tensor] = []
        for state_slice in self.state_slices:
            raw_factor = self.damping_factor[state_slice, state_slice]
            lower = torch.tril(raw_factor)
            damping = lower @ lower.transpose(-1, -2)
            raw_rotation = self.rotation_raw[state_slice, state_slice]
            rotation = 0.5 * (raw_rotation - raw_rotation.transpose(-1, -2))
            identity = torch.eye(
                state_slice.stop - state_slice.start,
                dtype=self.fdtype,
                device=self.device,
            )
            blocks.append(
                damping
                + self.config.minimum_mean_reversion_per_week * identity
                + rotation
            )
        generator = torch.block_diag(*blocks)
        coupling = self.causal_coupling_raw * self.generator_strict_upper_mask
        return generator + coupling

    def weekly_transition(self) -> torch.Tensor:
        """Return the exact stationary, expiry-causal weekly transition."""
        return torch.matrix_exp(-self.mean_reversion_generator())

    def innovation_cholesky(self) -> torch.Tensor:
        """Return a nonsingular ordered hierarchical innovation factor."""
        raw = self.innovation_cholesky_raw
        positive_diagonal = self.config.innovation_std_floor + (
            torch.nn.functional.softplus(torch.diagonal(raw))
        )
        return torch.tril(raw, diagonal=-1) + torch.diag(positive_diagonal)

    def decode_days(
        self,
        day_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeParameters, SanosSurface]:
        """Decode selected hierarchical factors through SANOS."""
        raw = self.raw_states_for_days(day_indices)
        parameters = bound_native_raw(raw, geometry=self.config.geometry)
        surface = self.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        return raw, parameters, surface

    def forward(
        self,
        *,
        day_indices: torch.Tensor,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
        is_call: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, NativeParameters, SanosSurface]:
        """Price a padded option batch through hierarchical factors."""
        raw, parameters, surface = self.decode_days(day_indices)
        call_prices = self.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiries_y,
            strikes=strikes,
        )
        option_prices = torch.where(
            is_call,
            call_prices,
            call_prices - 1.0 + strikes,
        )
        return option_prices, raw, parameters, surface
