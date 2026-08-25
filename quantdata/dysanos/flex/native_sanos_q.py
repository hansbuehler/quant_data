"""Torch-native P-free spot paths from the martingale kernel inside SANOS."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quantdata.dysanos.flex.config import FlexGeometryConfig
from quantdata.dysanos.flex.sanos import (
    FlexSanosDecoder,
    SanosSurface,
    uses_shared_strikes,
)
from quantdata.torch import BaseEnvironment, BaseModel

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class NativeSanosQRandoms:
    """Explicit base random tensors for reproducible native-Q path generation."""

    categorical_uniforms: torch.Tensor
    backbone_normals: torch.Tensor


@dataclass(frozen=True)
class NativeSanosQState:
    """Live filtration state after one native weekly Q step."""

    normalized_spot: torch.Tensor
    revealed_node: torch.Tensor
    expiry_index: int


@dataclass(frozen=True)
class NativeSanosQPathBatch:
    """Normalized native-Q spot paths and revealed SANOS node state."""

    normalized_spots: torch.Tensor
    revealed_nodes: torch.Tensor
    randoms: NativeSanosQRandoms


@dataclass(frozen=True)
class NativeSanosQCallCurves:
    """All remaining weekly calls conditional on spot and revealed node."""

    target_expiry_indices: torch.Tensor
    normalized_strikes: torch.Tensor
    normalized_call_prices: torch.Tensor
    normalized_support: torch.Tensor
    conditional_densities: torch.Tensor
    backbone_total_vols: torch.Tensor


class NativeSanosQRoller(BaseModel):
    """Generate and price the exact shared-strike SANOS martingale chain.

    The categorical draw which reveals a discrete SANOS node is not pathwise
    differentiable with respect to its probabilities.  Random uniforms are
    accepted explicitly so score-function, relaxed, or common-random-number
    research can be added deliberately.  Conditional on the sampled integer
    nodes, density propagation, option pricing, and a downstream strategy loss
    remain ordinary Torch graphs with gradients through SANOS parameters.

    No empirical P transition, entropy objective, or pre-generated path store
    enters this construction.

    The direct kernel retains its original five-business-day W1-to-W2 timing.
    The ``5/4.8`` DLVsq adjustment is needed only when re-encoding that kernel
    as a new front SANOS surface; direct row propagation does not use it.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        geometry: FlexGeometryConfig,
    ) -> None:
        super().__init__(env)
        if not uses_shared_strikes(geometry):
            raise ValueError("Native SANOS Q requires shared strikes across expiries")
        self.geometry = geometry
        self.sanos_decoder = FlexSanosDecoder(env, geometry=geometry)

    def enable_solver_compile(self) -> None:
        """Compile the shared differentiable tridiagonal density solve."""
        self.sanos_decoder.enable_solver_compile()

    def disable_solver_compile(self) -> None:
        """Use the eager tridiagonal density solve."""
        self.sanos_decoder.disable_solver_compile()

    def _validate_surface(self, surface: SanosSurface) -> tuple[int, int, int]:
        support = surface.support_strikes
        if support.ndim != 3:
            raise ValueError("SanosSurface must have batched support shape (B,T,N)")
        batch, expiries, support_size = support.shape
        if expiries != self.geometry.num_model_expiries:
            raise ValueError("SanosSurface expiry count differs from roller geometry")
        if surface.densities.shape != support.shape:
            raise ValueError("SanosSurface densities do not align with support")
        if surface.dlvsqs.shape != (batch, expiries, support_size - 2):
            raise ValueError("SanosSurface DLVs do not align with support")
        if surface.backbone_total_vols.shape != (batch, expiries):
            raise ValueError("SanosSurface backbone vols have the wrong shape")
        if not bool(
            torch.all(
                support
                == support[:, :1].expand_as(support)
            )
        ):
            raise ValueError("Native SANOS Q received a non-shared strike surface")
        if not bool(
            torch.all(
                torch.diff(surface.backbone_total_vols.square(), dim=-1)
                >= -32.0 * torch.finfo(support.dtype).eps
            )
        ):
            raise ValueError("SanosSurface backbone total variance must not decrease")
        return batch, expiries, support_size

    @staticmethod
    def _expand_expiry_tensor(
        value: torch.Tensor,
        sample_shape: tuple[int, ...],
    ) -> torch.Tensor:
        batch = value.shape[0]
        return value.reshape(
            batch,
            *((1,) * len(sample_shape)),
            value.shape[-1],
        ).expand(batch, *sample_shape, value.shape[-1])

    def transition_density(
        self,
        surface: SanosSurface,
        density: torch.Tensor,
        *,
        target_expiry_index: int,
    ) -> torch.Tensor:
        """Propagate a batched density into one later native model expiry."""
        batch, expiries, support_size = self._validate_surface(surface)
        if target_expiry_index <= 0 or target_expiry_index >= expiries:
            raise ValueError(
                "Target expiry must be an existing successor; no far-end tail is defined"
            )
        if density.ndim < 2 or density.shape[0] != batch or density.shape[-1] != support_size:
            raise ValueError("Density must have shape (B,...,support)")
        if (
            density.device != surface.support_strikes.device
            or density.dtype != surface.support_strikes.dtype
        ):
            raise ValueError("Density must preserve surface device and dtype")
        return self._transition_density_unchecked(
            surface,
            density,
            target_expiry_index=target_expiry_index,
        )

    def _transition_density_unchecked(
        self,
        surface: SanosSurface,
        density: torch.Tensor,
        *,
        target_expiry_index: int,
    ) -> torch.Tensor:
        sample_shape = tuple(density.shape[1:-1])
        support = self._expand_expiry_tensor(
            surface.support_strikes[:, target_expiry_index],
            sample_shape,
        )
        dlvsqs = self._expand_expiry_tensor(
            surface.dlvsqs[:, target_expiry_index],
            sample_shape,
        )
        return self.sanos_decoder.solve_density_step_batch(
            support=support,
            dlvsqs=dlvsqs,
            time_increment_y=self.sanos_decoder.model_dts_y[target_expiry_index],
            rhs_density=density,
        )

    def transition_rows(
        self,
        surface: SanosSurface,
        revealed_nodes: torch.Tensor,
        *,
        target_expiry_index: int,
    ) -> torch.Tensor:
        """Return only the required transition row for every sampled path."""
        batch, expiries, support_size = self._validate_surface(surface)
        if target_expiry_index <= 0 or target_expiry_index >= expiries:
            raise ValueError(
                "Target expiry must be an existing successor; no far-end tail is defined"
            )
        if revealed_nodes.ndim < 1 or revealed_nodes.shape[0] != batch:
            raise ValueError("Revealed nodes must have shape (B,...)")
        if revealed_nodes.device != surface.support_strikes.device:
            raise ValueError("Revealed nodes must use the surface device")
        if revealed_nodes.dtype not in _INTEGER_DTYPES:
            raise ValueError("Revealed nodes must have an integer dtype")
        if not bool(
            torch.all((revealed_nodes >= 0) & (revealed_nodes < support_size))
        ):
            raise ValueError("Revealed node lies outside the SANOS support")
        density = torch.nn.functional.one_hot(
            revealed_nodes.to(torch.int64),
            num_classes=support_size,
        ).to(dtype=surface.support_strikes.dtype)
        return self._transition_density_unchecked(
            surface,
            density,
            target_expiry_index=target_expiry_index,
        )

    @staticmethod
    def _sampling_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
        probabilities = torch.clamp_min(probabilities, 0.0)
        mass = torch.sum(probabilities, dim=-1, keepdim=True)
        return probabilities / torch.clamp_min(
            mass,
            torch.finfo(probabilities.dtype).tiny,
        )

    @classmethod
    def _sample_categorical(
        cls,
        probabilities: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> torch.Tensor:
        sampling_probabilities = cls._sampling_probabilities(probabilities)
        cdf = torch.cumsum(sampling_probabilities, dim=-1)
        cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), dim=-1)
        # This comparison intentionally produces a discrete node with no
        # pathwise derivative in the categorical probabilities.
        return torch.sum(uniforms.unsqueeze(-1) > cdf, dim=-1).to(torch.int64)

    def make_randoms(
        self,
        *,
        batch_size: int,
        num_paths: int,
        num_steps: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> NativeSanosQRandoms:
        """Generate explicit uniforms and normals on the roller device."""
        if batch_size <= 0 or num_paths <= 0 or num_steps <= 0:
            raise ValueError("Random tensor dimensions must be positive")
        if num_steps > self.geometry.num_model_expiries:
            raise ValueError("Requested path steps exceed the existing last expiry")
        if antithetic and num_paths % 2 != 0:
            raise ValueError("Antithetic path count must be even")
        paths_to_draw = num_paths // 2 if antithetic else num_paths
        shape = (batch_size, paths_to_draw, num_steps)
        uniforms = torch.rand(
            shape,
            dtype=self.fdtype,
            device=self.device,
            generator=generator,
        )
        normals = torch.randn(
            shape,
            dtype=self.fdtype,
            device=self.device,
            generator=generator,
        )
        if antithetic:
            maximum_uniform = torch.nextafter(
                torch.ones((), dtype=self.fdtype, device=self.device),
                torch.zeros((), dtype=self.fdtype, device=self.device),
            )
            uniforms = torch.cat(
                (uniforms, torch.clamp(1.0 - uniforms, max=maximum_uniform)),
                dim=1,
            )
            normals = torch.cat((normals, -normals), dim=1)
        return NativeSanosQRandoms(
            categorical_uniforms=uniforms,
            backbone_normals=normals,
        )

    def _validate_step_randoms(
        self,
        surface: SanosSurface,
        categorical_uniform: torch.Tensor,
        backbone_normal: torch.Tensor,
    ) -> tuple[int, int]:
        batch, _, support_size = self._validate_surface(surface)
        if (
            categorical_uniform.shape != backbone_normal.shape
            or categorical_uniform.ndim < 1
            or categorical_uniform.shape[0] != batch
        ):
            raise ValueError("Step randoms must share shape (B,...)")
        if (
            categorical_uniform.device != surface.support_strikes.device
            or backbone_normal.device != surface.support_strikes.device
            or categorical_uniform.dtype != surface.support_strikes.dtype
            or backbone_normal.dtype != surface.support_strikes.dtype
        ):
            raise ValueError("Step randoms must preserve surface device and dtype")
        if not bool(
            torch.all(
                torch.isfinite(categorical_uniform)
                & (categorical_uniform >= 0.0)
                & (categorical_uniform < 1.0)
                & torch.isfinite(backbone_normal)
            )
        ):
            raise ValueError("Step uniforms/normals are outside their valid domain")
        return batch, support_size

    def sample_front_state(
        self,
        surface: SanosSurface,
        *,
        categorical_uniform: torch.Tensor,
        backbone_normal: torch.Tensor,
    ) -> NativeSanosQState:
        """Sample front spot and reveal its discrete SANOS mixture node."""
        batch, support_size = self._validate_step_randoms(
            surface,
            categorical_uniform,
            backbone_normal,
        )
        return self._sample_front_state_unchecked(
            surface,
            categorical_uniform=categorical_uniform,
            backbone_normal=backbone_normal,
            batch=batch,
            support_size=support_size,
        )

    def _sample_front_state_unchecked(
        self,
        surface: SanosSurface,
        *,
        categorical_uniform: torch.Tensor,
        backbone_normal: torch.Tensor,
        batch: int,
        support_size: int,
    ) -> NativeSanosQState:
        sample_shape = tuple(categorical_uniform.shape[1:])
        probability = self._expand_expiry_tensor(surface.densities[:, 0], sample_shape)
        node = self._sample_categorical(probability, categorical_uniform)
        support = self._expand_expiry_tensor(
            surface.support_strikes[:, 0], sample_shape
        )
        assert support.shape == (*categorical_uniform.shape, support_size)
        latent_spot = torch.gather(support, -1, node.unsqueeze(-1)).squeeze(-1)
        front_vol = surface.backbone_total_vols[:, 0].reshape(
            batch, *((1,) * len(sample_shape))
        )
        spot = latent_spot * torch.exp(
            -0.5 * front_vol.square() + front_vol * backbone_normal
        )
        return NativeSanosQState(
            normalized_spot=spot,
            revealed_node=node,
            expiry_index=0,
        )

    def advance_state(
        self,
        surface: SanosSurface,
        state: NativeSanosQState,
        *,
        categorical_uniform: torch.Tensor,
        backbone_normal: torch.Tensor,
    ) -> NativeSanosQState:
        """Advance one existing native weekly expiry, retaining the new node."""
        batch, support_size = self._validate_step_randoms(
            surface,
            categorical_uniform,
            backbone_normal,
        )
        expiries = self.geometry.num_model_expiries
        if state.expiry_index < 0 or state.expiry_index >= expiries - 1:
            raise ValueError("Cannot advance beyond the existing last expiry")
        if (
            state.normalized_spot.shape != categorical_uniform.shape
            or state.revealed_node.shape != categorical_uniform.shape
            or state.normalized_spot.device != surface.support_strikes.device
            or state.revealed_node.device != surface.support_strikes.device
        ):
            raise ValueError("Native-Q state and step randoms have incompatible shapes")
        if state.normalized_spot.dtype != surface.support_strikes.dtype:
            raise ValueError("Native-Q spot must preserve surface dtype")
        if not bool(
            torch.all(torch.isfinite(state.normalized_spot) & (state.normalized_spot > 0.0))
        ):
            raise ValueError("Native-Q spot must be finite and positive")
        if state.revealed_node.dtype not in _INTEGER_DTYPES:
            raise ValueError("Revealed nodes must have an integer dtype")
        if not bool(
            torch.all((state.revealed_node >= 0) & (state.revealed_node < support_size))
        ):
            raise ValueError("Revealed node lies outside the SANOS support")
        return self._advance_state_unchecked(
            surface,
            state,
            categorical_uniform=categorical_uniform,
            backbone_normal=backbone_normal,
            batch=batch,
            support_size=support_size,
        )

    def _advance_state_unchecked(
        self,
        surface: SanosSurface,
        state: NativeSanosQState,
        *,
        categorical_uniform: torch.Tensor,
        backbone_normal: torch.Tensor,
        batch: int,
        support_size: int,
    ) -> NativeSanosQState:
        target_index = state.expiry_index + 1
        density = torch.nn.functional.one_hot(
            state.revealed_node.to(torch.int64),
            num_classes=support_size,
        ).to(dtype=surface.support_strikes.dtype)
        transition = self._transition_density_unchecked(
            surface,
            density,
            target_expiry_index=target_index,
        )
        next_node = self._sample_categorical(transition, categorical_uniform)
        sample_shape = tuple(categorical_uniform.shape[1:])
        previous_support = self._expand_expiry_tensor(
            surface.support_strikes[:, state.expiry_index], sample_shape
        )
        next_support = self._expand_expiry_tensor(
            surface.support_strikes[:, target_index], sample_shape
        )
        assert previous_support.shape == next_support.shape == (
            *categorical_uniform.shape,
            support_size,
        )
        previous_latent = torch.gather(
            previous_support, -1, state.revealed_node.unsqueeze(-1)
        ).squeeze(-1)
        next_latent = torch.gather(
            next_support, -1, next_node.unsqueeze(-1)
        ).squeeze(-1)
        variance_increment = (
            surface.backbone_total_vols[:, target_index].square()
            - surface.backbone_total_vols[:, state.expiry_index].square()
        ).reshape(batch, *((1,) * len(sample_shape)))
        increment_vol = torch.sqrt(torch.clamp_min(variance_increment, 0.0))
        spot = (
            state.normalized_spot
            * next_latent
            / previous_latent
            * torch.exp(
                -0.5 * increment_vol.square()
                + increment_vol * backbone_normal
            )
        )
        return NativeSanosQState(
            normalized_spot=spot,
            revealed_node=next_node,
            expiry_index=target_index,
        )

    def sample_paths(
        self,
        surface: SanosSurface,
        *,
        randoms: NativeSanosQRandoms,
    ) -> NativeSanosQPathBatch:
        """Generate live normalized spot paths without P or path pre-generation."""
        batch, expiries, _ = self._validate_surface(surface)
        uniforms = randoms.categorical_uniforms
        normals = randoms.backbone_normals
        if uniforms.shape != normals.shape or uniforms.ndim != 3:
            raise ValueError("Native-Q randoms must share shape (B,paths,steps)")
        if uniforms.shape[0] != batch:
            raise ValueError("Native-Q random batch differs from surface batch")
        if (
            uniforms.device != surface.support_strikes.device
            or normals.device != surface.support_strikes.device
            or uniforms.dtype != surface.support_strikes.dtype
            or normals.dtype != surface.support_strikes.dtype
        ):
            raise ValueError("Native-Q randoms must preserve surface device and dtype")
        _, num_steps = uniforms.shape[1:]
        if num_steps <= 0 or num_steps > expiries:
            raise ValueError(
                "Native-Q path must stop at the existing last expiry; no tail exists"
            )
        if not bool(
            torch.all(
                torch.isfinite(uniforms)
                & (uniforms >= 0.0)
                & (uniforms < 1.0)
                & torch.isfinite(normals)
            )
        ):
            raise ValueError("Native-Q uniforms/normals are outside their valid domain")
        state = self._sample_front_state_unchecked(
            surface,
            categorical_uniform=uniforms[..., 0],
            backbone_normal=normals[..., 0],
            batch=batch,
            support_size=surface.support_strikes.shape[-1],
        )
        spots = [state.normalized_spot]
        nodes = [state.revealed_node]
        for target_index in range(1, num_steps):
            state = self._advance_state_unchecked(
                surface,
                state,
                categorical_uniform=uniforms[..., target_index],
                backbone_normal=normals[..., target_index],
                batch=batch,
                support_size=surface.support_strikes.shape[-1],
            )
            assert state.expiry_index == target_index
            spots.append(state.normalized_spot)
            nodes.append(state.revealed_node)
        return NativeSanosQPathBatch(
            normalized_spots=torch.stack(spots, dim=-1),
            revealed_nodes=torch.stack(nodes, dim=-1),
            randoms=randoms,
        )

    @staticmethod
    def _black_calls(
        spots: torch.Tensor,
        strikes: torch.Tensor,
        total_vol: torch.Tensor,
    ) -> torch.Tensor:
        safe_vol = torch.clamp_min(total_vol, torch.finfo(spots.dtype).tiny)
        d1 = (
            torch.log(spots / strikes) + 0.5 * safe_vol.square()
        ) / safe_vol
        black = spots * torch.special.ndtr(d1) - strikes * torch.special.ndtr(
            d1 - safe_vol
        )
        return torch.where(total_vol > 0.0, black, torch.relu(spots - strikes))

    def price_remaining_calls(
        self,
        surface: SanosSurface,
        *,
        current_expiry_index: int,
        revealed_nodes: torch.Tensor,
        normalized_strikes: torch.Tensor,
    ) -> NativeSanosQCallCurves:
        """Price every remaining weekly call on its exact revealed-node grid."""
        batch, expiries, support_size = self._validate_surface(surface)
        if current_expiry_index < 0 or current_expiry_index >= expiries - 1:
            raise ValueError(
                "Current expiry has no existing successor; no far-end tail is defined"
            )
        if revealed_nodes.ndim < 1 or revealed_nodes.shape[0] != batch:
            raise ValueError("Revealed nodes must have shape (B,...)")
        if revealed_nodes.device != surface.support_strikes.device:
            raise ValueError("Revealed nodes must use the surface device")
        if revealed_nodes.dtype not in _INTEGER_DTYPES:
            raise ValueError("Revealed nodes must have an integer dtype")
        if not bool(
            torch.all((revealed_nodes >= 0) & (revealed_nodes < support_size))
        ):
            raise ValueError("Revealed node lies outside the SANOS support")
        sample_shape = tuple(revealed_nodes.shape[1:])
        if normalized_strikes.ndim == 1:
            strikes = normalized_strikes.reshape(
                *((1,) * revealed_nodes.ndim), normalized_strikes.shape[-1]
            ).expand(*revealed_nodes.shape, normalized_strikes.shape[-1])
        elif normalized_strikes.shape[:-1] == revealed_nodes.shape:
            strikes = normalized_strikes
        else:
            raise ValueError(
                "Normalized strikes must have shape (Q,) or (B,...,Q)"
            )
        if strikes.device != surface.support_strikes.device or strikes.dtype != (
            surface.support_strikes.dtype
        ):
            raise ValueError("Normalized strikes must preserve surface device and dtype")
        if not bool(torch.all(torch.isfinite(strikes) & (strikes > 0.0))):
            raise ValueError("Normalized strikes must be finite and positive")
        density = torch.nn.functional.one_hot(
            revealed_nodes.to(torch.int64),
            num_classes=support_size,
        ).to(dtype=surface.support_strikes.dtype)
        current_support = self._expand_expiry_tensor(
            surface.support_strikes[:, current_expiry_index],
            sample_shape,
        )
        current_node_support = torch.gather(
            current_support,
            -1,
            revealed_nodes.unsqueeze(-1),
        ).squeeze(-1)
        normalized_support = current_support / current_node_support.unsqueeze(-1)
        call_curves: list[torch.Tensor] = []
        densities: list[torch.Tensor] = []
        relative_vols: list[torch.Tensor] = []
        for target_index in range(current_expiry_index + 1, expiries):
            density = self._transition_density_unchecked(
                surface,
                density,
                target_expiry_index=target_index,
            )
            variance = (
                surface.backbone_total_vols[:, target_index].square()
                - surface.backbone_total_vols[:, current_expiry_index].square()
            )
            relative_vol = torch.sqrt(torch.clamp_min(variance, 0.0)).reshape(
                batch, *((1,) * len(sample_shape))
            ).expand(*revealed_nodes.shape)
            component_calls = self._black_calls(
                normalized_support.unsqueeze(-2),
                strikes.unsqueeze(-1),
                relative_vol.unsqueeze(-1).unsqueeze(-1),
            )
            call_curves.append(
                torch.sum(density.unsqueeze(-2) * component_calls, dim=-1)
            )
            densities.append(density)
            relative_vols.append(relative_vol)
        return NativeSanosQCallCurves(
            target_expiry_indices=torch.arange(
                current_expiry_index + 1,
                expiries,
                dtype=self.idtype,
                device=self.device,
            ),
            normalized_strikes=strikes,
            normalized_call_prices=torch.stack(call_curves, dim=-2),
            normalized_support=normalized_support,
            conditional_densities=torch.stack(densities, dim=-2),
            backbone_total_vols=torch.stack(relative_vols, dim=-1),
        )
