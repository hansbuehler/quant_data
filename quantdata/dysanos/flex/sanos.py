"""Torch-native SANOS on the fixed 14-by-200 FLEX geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.torch import BaseEnvironment, BaseModel


def uses_shared_strikes(geometry: FlexGeometryConfig) -> bool:
    """Return whether one daily floating support is shared across expiries."""
    return isinstance(
        geometry,
        SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    )


def make_inner_nms_nodes(
    *,
    num_inner_strikes: int,
    nms_range: tuple[float, float],
) -> np.ndarray:
    """Generate fixed NMS coordinates with exactly one ATM coordinate."""
    lower, upper = nms_range
    distance = float(upper - lower)
    num_low = min(
        max(2, int(float(num_inner_strikes * abs(lower)) // distance)),
        num_inner_strikes - 3,
    )
    num_high = num_inner_strikes - num_low - 1
    nodes = np.concatenate(
        (
            np.linspace(lower, 0.0, num_low, endpoint=False),
            np.linspace(0.0, upper, num_high + 1),
        )
    )
    assert nodes.shape == (num_inner_strikes,)
    assert np.sum(nodes == 0.0) == 1
    assert np.all(np.diff(nodes) > 0.0)
    return nodes


def make_shared_nms_node_sets(
    *,
    num_inner_strikes: int,
    nms_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return half front- and half final-expiry NMS nodes with one ATM total."""
    if num_inner_strikes % 2 != 0:
        raise ValueError(
            "Shared strikes require an even inner-strike count, found "
            f"{num_inner_strikes}"
        )
    half = num_inner_strikes // 2
    first = make_inner_nms_nodes(
        num_inner_strikes=half,
        nms_range=nms_range,
    )
    last = np.linspace(nms_range[0], nms_range[1], half)
    if np.any(np.isclose(last, 0.0, rtol=0.0, atol=1.0e-14)):
        edges = np.linspace(nms_range[0], nms_range[1], half + 1)
        last = 0.5 * (edges[:-1] + edges[1:])
    assert first.shape == last.shape == (half,)
    assert np.sum(first == 0.0) == 1
    assert not np.any(np.isclose(last, 0.0, rtol=0.0, atol=1.0e-14))
    return first, last


def _space_merged_log_strikes_numpy(
    candidates: np.ndarray,
    *,
    minimum_log_spacing_fraction: float,
) -> np.ndarray:
    """Sort merged log-strikes and impose a relative gap around fixed ATM."""
    ordered = np.sort(np.asarray(candidates, dtype=np.float64))
    atm = np.flatnonzero(np.isclose(ordered, 0.0, rtol=0.0, atol=1.0e-14))
    if len(atm) != 1:
        raise ValueError("Shared strike candidates must contain exactly one ATM")
    atm_index = int(atm[0])
    minimum_gap = (
        minimum_log_spacing_fraction
        * (ordered[-1] - ordered[0])
        / (len(ordered) - 1)
    )
    left_gaps = np.maximum(np.diff(ordered[: atm_index + 1]), minimum_gap)
    right_gaps = np.maximum(np.diff(ordered[atm_index:]), minimum_gap)
    left = -np.flip(np.cumsum(np.flip(left_gaps)))
    right = np.cumsum(right_gaps)
    output = np.concatenate((left, np.zeros(1), right))
    assert output.shape == ordered.shape
    assert np.sum(output == 0.0) == 1
    assert np.all(np.diff(output) >= minimum_gap * (1.0 - 1.0e-12))
    return output


def _space_merged_log_strikes_torch(
    candidates: torch.Tensor,
    *,
    atm_index: int,
    minimum_log_spacing_fraction: float,
) -> torch.Tensor:
    """Differentiable batched counterpart of the NumPy spacing construction."""
    ordered = torch.sort(candidates, dim=-1).values
    minimum_gap = (
        minimum_log_spacing_fraction
        * (ordered[..., -1] - ordered[..., 0])
        / (ordered.shape[-1] - 1)
    ).unsqueeze(-1)
    left_gaps = torch.clamp(
        torch.diff(ordered[..., : atm_index + 1], dim=-1),
        min=minimum_gap,
    )
    right_gaps = torch.clamp(
        torch.diff(ordered[..., atm_index:], dim=-1),
        min=minimum_gap,
    )
    left = -torch.flip(
        torch.cumsum(torch.flip(left_gaps, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    right = torch.cumsum(right_gaps, dim=-1)
    return torch.cat((left, torch.zeros_like(ordered[..., :1]), right), dim=-1)


def total_vols_from_forward_vols_numpy(
    forward_vols: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
) -> np.ndarray:
    """Apply FLEX's specified cumulative square-root-time total-vol formula."""
    forward_vols = np.asarray(forward_vols, dtype=np.float64)
    T = geometry.num_model_expiries
    if forward_vols.shape[-1] != T:
        raise ValueError(
            f"Expected forward-vol shape ending in {T}, found {forward_vols.shape}"
        )
    dts = np.diff(geometry.model_expiries_y, prepend=0.0)
    return np.cumsum(forward_vols * np.sqrt(dts), axis=-1)


def make_supports_from_total_vols(
    total_vols: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
) -> np.ndarray:
    """Construct the exact canonical-NMS supports from model total vols."""
    total_vols = np.asarray(total_vols, dtype=np.float64)
    T = geometry.num_model_expiries
    if total_vols.shape != (T,):
        raise ValueError(f"Expected total vols of shape {(T,)}, found {total_vols.shape}")
    if not np.all(np.isfinite(total_vols) & (total_vols > 0.0)):
        raise ValueError("Every model total volatility must be finite and positive")
    if not np.all(np.diff(total_vols) > 0.0):
        raise ValueError("FLEX model total volatilities must be strictly increasing")
    epsilon_vol = total_vols[0] / np.sqrt(geometry.model_expiries_y[0])
    nms_scale = np.sqrt(
        (
            epsilon_vol**2 * geometry.nms_epsilon_y
            + total_vols**2
        )
        / (1.0 + geometry.nms_epsilon_y)
    )
    if uses_shared_strikes(geometry):
        first_nms, last_nms = make_shared_nms_node_sets(
            num_inner_strikes=geometry.num_inner_strikes,
            nms_range=geometry.nms_range,
        )
        merged_log_strikes = _space_merged_log_strikes_numpy(
            np.concatenate(
                (
                    nms_scale[0] * first_nms,
                    nms_scale[-1] * last_nms,
                )
            ),
            minimum_log_spacing_fraction=(
                geometry.minimum_log_spacing_fraction
            ),
        )
        shared_inner = np.exp(merged_log_strikes)
        inner = np.broadcast_to(
            shared_inner[None, :],
            (T, geometry.num_inner_strikes),
        )
    else:
        inner_nms = make_inner_nms_nodes(
            num_inner_strikes=geometry.num_inner_strikes,
            nms_range=geometry.nms_range,
        )
        inner = np.exp(nms_scale[:, None] * inner_nms[None, :])
    return np.concatenate(
        (
            geometry.lower_boundary_scale * inner[:, :1],
            inner,
            geometry.upper_boundary_scale * inner[:, -1:],
        ),
        axis=1,
    )


@dataclass(frozen=True)
class SanosTimeInterpolation:
    """Dysanos density and variance interpolation coordinates."""

    lower_index: np.ndarray
    upper_index: np.ndarray
    upper_weight: np.ndarray
    before_first: np.ndarray


def sanos_time_interpolation_numpy(
    query_expiries_y: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
) -> SanosTimeInterpolation:
    """Return the exact left-extrapolation/interpolation rule for observations."""
    query = np.asarray(query_expiries_y, dtype=np.float64)
    nodes = geometry.model_expiries_y
    if not np.all(np.isfinite(query) & (query >= 0.0)):
        raise ValueError("Every queried expiry must be finite and non-negative")
    if np.any(query > nodes[-1] + 1.0e-12):
        raise ValueError(
            "FLEX does not fit market expiries beyond its final model expiry"
        )
    raw_upper = np.searchsorted(nodes, query, side="left")
    upper = np.minimum(raw_upper, len(nodes) - 1).astype(np.int64)
    before_first = raw_upper == 0
    lower = np.maximum(raw_upper - 1, 0).astype(np.int64)
    previous_time = np.where(before_first, 0.0, nodes[lower])
    interval = nodes[upper] - previous_time
    upper_weight = (query - previous_time) / interval
    upper_weight = np.clip(upper_weight, 0.0, 1.0)
    return SanosTimeInterpolation(
        lower_index=lower,
        upper_index=upper,
        upper_weight=upper_weight,
        before_first=before_first,
    )


def interpolated_total_variance_numpy(
    model_total_vols: np.ndarray,
    query_expiries_y: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
) -> np.ndarray:
    """Linearly interpolate squared model total volatility in expiry."""
    model_total_vols = np.asarray(model_total_vols, dtype=np.float64)
    if model_total_vols.shape != (geometry.num_model_expiries,):
        raise ValueError(
            "'model_total_vols' must have one value per model expiry"
        )
    interpolation = sanos_time_interpolation_numpy(
        query_expiries_y,
        geometry=geometry,
    )
    upper_variance = model_total_vols[interpolation.upper_index] ** 2
    lower_variance = model_total_vols[interpolation.lower_index] ** 2
    lower_variance = np.where(
        interpolation.before_first,
        0.0,
        lower_variance,
    )
    return (
        (1.0 - interpolation.upper_weight) * lower_variance
        + interpolation.upper_weight * upper_variance
    )


@dataclass(frozen=True)
class SanosSurface:
    """Decoded fixed-weekly density surface and its bounded parameters."""

    expiries_y: torch.Tensor
    support_strikes: torch.Tensor
    density0: torch.Tensor
    densities: torch.Tensor
    dlvsqs: torch.Tensor
    forward_vols: torch.Tensor
    total_vols: torch.Tensor
    eta: torch.Tensor
    backbone_total_vols: torch.Tensor


class FlexSanosDecoder(BaseModel):
    """Decode bounded native parameters on FLEX's fixed weekly geometry."""

    def __init__(self, env: BaseEnvironment, *, geometry: FlexGeometryConfig):
        super().__init__(env)
        self.geometry = geometry
        self.register_buffer(
            "inner_nms",
            torch.tensor(
                make_inner_nms_nodes(
                    num_inner_strikes=geometry.num_inner_strikes,
                    nms_range=geometry.nms_range,
                ),
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        if uses_shared_strikes(geometry):
            first_nms, last_nms = make_shared_nms_node_sets(
                num_inner_strikes=geometry.num_inner_strikes,
                nms_range=geometry.nms_range,
            )
            self.register_buffer(
                "shared_first_nms",
                torch.tensor(first_nms, dtype=env.fdtype, device=env.device),
            )
            self.register_buffer(
                "shared_last_nms",
                torch.tensor(last_nms, dtype=env.fdtype, device=env.device),
            )
            self.shared_atm_index = int(
                np.sum(first_nms < 0.0) + np.sum(last_nms < 0.0)
            )
        self.register_buffer(
            "model_expiries_y",
            torch.tensor(
                geometry.model_expiries_y,
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        self.register_buffer(
            "model_dts_y",
            torch.diff(
                self.model_expiries_y,
                prepend=torch.zeros_like(self.model_expiries_y[:1]),
            ),
        )
        self._sanos_step = self._solve_sanos_step

    def enable_solver_compile(self) -> None:
        """Compile the fixed-width differentiable tridiagonal solve."""
        self._sanos_step = torch.compile(self._solve_sanos_step, dynamic=True)

    def disable_solver_compile(self) -> None:
        """Restore eager arithmetic for exact, comparable fit reporting."""
        self._sanos_step = self._solve_sanos_step

    def _total_vols(self, forward_vols: torch.Tensor) -> torch.Tensor:
        """Apply the user-specified cumulative square-root-time formula."""
        return torch.cumsum(
            forward_vols * torch.sqrt(self.model_dts_y),
            dim=-1,
        )

    @staticmethod
    def _density_from_calls(
        support: torch.Tensor,
        calls: torch.Tensor,
    ) -> torch.Tensor:
        """Recover discrete martingale masses from convex calls on a support."""
        one = torch.ones_like(calls[..., :1])
        zero = torch.zeros_like(calls[..., :1])
        slopes = torch.diff(calls, dim=-1, prepend=one) / torch.diff(
            support,
            dim=-1,
            prepend=zero,
        )
        return torch.diff(slopes, dim=-1, append=zero)

    def _transfer_density(
        self,
        previous_support: torch.Tensor,
        previous_density: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Represent the preceding martingale law on the wider current support."""
        calls = torch.matmul(
            torch.relu(previous_support.unsqueeze(-2) - support.unsqueeze(-1)),
            previous_density.unsqueeze(-1),
        ).squeeze(-1)
        return self._density_from_calls(support, calls)

    @staticmethod
    def _solve_sanos_step(
        *,
        w_minus: torch.Tensor,
        w_plus: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the SANOS tridiagonal system without a dense matrix."""
        assert w_minus.shape == w_plus.shape
        assert rhs.shape == w_minus.shape[:-1] + (w_minus.shape[-1] + 2,)
        n = rhs.shape[-1]
        cp_terms = [-w_minus[..., 0]]
        y_terms = [rhs[..., 0]]
        for index in range(1, n - 1):
            a_i = -w_plus[..., index - 2] if index > 1 else 0.0
            b_i = 1.0 + w_minus[..., index - 1] + w_plus[..., index - 1]
            c_i = -w_minus[..., index] if index < n - 2 else 0.0
            denominator = b_i - a_i * cp_terms[index - 1]
            cp_terms.append(c_i / denominator)
            y_terms.append(
                (rhs[..., index] - a_i * y_terms[index - 1]) / denominator
            )
        a_last = -w_plus[..., -1]
        denominator = 1.0 - a_last * cp_terms[-1]
        y_terms.append((rhs[..., -1] - a_last * y_terms[-1]) / denominator)

        x_terms: list[torch.Tensor | None] = [None] * n
        x_terms[-1] = y_terms[-1]
        for index in range(n - 2, -1, -1):
            assert x_terms[index + 1] is not None
            x_terms[index] = (
                y_terms[index] - cp_terms[index] * x_terms[index + 1]
            )
        assert all(term is not None for term in x_terms)
        return torch.stack(x_terms, dim=-1)  # type: ignore[arg-type]

    def forward(
        self,
        *,
        dlvsqs: torch.Tensor,
        forward_vols: torch.Tensor,
        eta: torch.Tensor,
    ) -> SanosSurface:
        """Decode one day's bounded native state."""
        assert dlvsqs.ndim == 2
        T = self.geometry.num_model_expiries
        assert eta.shape in {(), (T,)}
        surface = self.forward_batch(
            dlvsqs=dlvsqs.unsqueeze(0),
            forward_vols=forward_vols.unsqueeze(0),
            eta=eta.unsqueeze(0),
        )
        return SanosSurface(
            expiries_y=surface.expiries_y[0],
            support_strikes=surface.support_strikes[0],
            density0=surface.density0[0],
            densities=surface.densities[0],
            dlvsqs=surface.dlvsqs[0],
            forward_vols=surface.forward_vols[0],
            total_vols=surface.total_vols[0],
            eta=surface.eta[0],
            backbone_total_vols=surface.backbone_total_vols[0],
        )

    def forward_batch(
        self,
        *,
        dlvsqs: torch.Tensor,
        forward_vols: torch.Tensor,
        eta: torch.Tensor,
    ) -> SanosSurface:
        """Decode a batch of bounded native states on the fixed weekly grid."""
        T = self.geometry.num_model_expiries
        N = self.geometry.num_inner_strikes
        assert dlvsqs.ndim == 3
        B = dlvsqs.shape[0]
        assert dlvsqs.shape == (B, T, N)
        assert forward_vols.shape == (B, T)
        assert eta.shape in {(B,), (B, T)}

        total_vols = self._total_vols(forward_vols)
        epsilon_vol = total_vols[:, :1] / torch.sqrt(self.model_expiries_y[:1])
        nms_scale = torch.sqrt(
            (
                epsilon_vol.square() * self.geometry.nms_epsilon_y
                + total_vols.square()
            )
            / (1.0 + self.geometry.nms_epsilon_y)
        )
        if uses_shared_strikes(self.geometry):
            merged_log_strikes = _space_merged_log_strikes_torch(
                torch.cat(
                    (
                        nms_scale[:, :1] * self.shared_first_nms,
                        nms_scale[:, -1:] * self.shared_last_nms,
                    ),
                    dim=-1,
                ),
                atm_index=self.shared_atm_index,
                minimum_log_spacing_fraction=(
                    self.geometry.minimum_log_spacing_fraction
                ),
            )
            shared_inner = torch.exp(merged_log_strikes)
            inner = shared_inner.unsqueeze(1).expand(-1, T, -1)
        else:
            inner = torch.exp(nms_scale.unsqueeze(-1) * self.inner_nms)
        support_strikes = torch.cat(
            (
                self.geometry.lower_boundary_scale * inner[..., :1],
                inner,
                self.geometry.upper_boundary_scale * inner[..., -1:],
            ),
            dim=-1,
        )

        calls0 = torch.relu(1.0 - support_strikes[:, 0])
        density0 = self._density_from_calls(support_strikes[:, 0], calls0)
        densities: list[torch.Tensor] = []
        previous_support = support_strikes[:, 0]
        previous_density = density0
        for expiry_index in range(T):
            support = support_strikes[:, expiry_index]
            density_before = (
                previous_density
                if expiry_index == 0
                else self._transfer_density(
                    previous_support,
                    previous_density,
                    support,
                )
            )
            box = (
                support[:, 1:-1].square()
                * self.model_dts_y[expiry_index]
                / (support[:, 2:] - support[:, :-2])
            )
            w_plus = (
                dlvsqs[:, expiry_index]
                * box
                / (support[:, 2:] - support[:, 1:-1])
            )
            w_minus = (
                dlvsqs[:, expiry_index]
                * box
                / (support[:, 1:-1] - support[:, :-2])
            )
            density = self._sanos_step(
                w_minus=w_minus,
                w_plus=w_plus,
                rhs=density_before,
            )
            densities.append(density)
            previous_support = support
            previous_density = density

        density_tensor = torch.stack(densities, dim=1)
        tolerance = max(1.0e-8, 256.0 * torch.finfo(density_tensor.dtype).eps)
        self.verify(
            lambda: torch.all(torch.isfinite(density_tensor)),
            "Decoded SANOS density contains a non-finite value",
        )
        self.verify(
            lambda: torch.min(density_tensor) >= -tolerance,
            lambda: f"Decoded SANOS density is negative: {density_tensor.min()}",
        )
        self.verify(
            lambda: torch.max(torch.abs(density_tensor.sum(dim=-1) - 1.0))
            <= tolerance,
            "Decoded SANOS density does not have unit mass",
        )
        self.verify(
            lambda: torch.max(
                torch.abs(
                    torch.sum(support_strikes * density_tensor, dim=-1) - 1.0
                )
            )
            <= tolerance,
            "Decoded SANOS density does not have unit mean",
        )
        backbone_forward_vols = (
            forward_vols * eta
            if eta.ndim == 2
            else forward_vols * eta.unsqueeze(-1)
        )
        return SanosSurface(
            expiries_y=self.model_expiries_y.unsqueeze(0).expand(B, -1),
            support_strikes=support_strikes,
            density0=density0,
            densities=density_tensor,
            dlvsqs=dlvsqs,
            forward_vols=forward_vols,
            total_vols=total_vols,
            eta=eta,
            backbone_total_vols=self._total_vols(backbone_forward_vols),
        )

    @staticmethod
    def _black_call(
        spot: torch.Tensor,
        strike: torch.Tensor,
        sqrt_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-carry Black call, reducing exactly to intrinsic at zero variance."""
        safe_sqrt_variance = torch.clamp(sqrt_variance, min=1.0e-10)
        d1 = (
            torch.log(spot / strike) + 0.5 * safe_sqrt_variance.square()
        ) / safe_sqrt_variance
        d2 = d1 - safe_sqrt_variance
        black = (
            spot * torch.special.ndtr(d1)
            - strike * torch.special.ndtr(d2)
        )
        return torch.where(
            sqrt_variance > 0.0,
            black,
            torch.relu(spot - strike),
        )

    def _torch_interpolation(
        self,
        query_expiries_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return lower/upper density indices, upper weight, and left mask."""
        T = self.geometry.num_model_expiries
        # Pricing frequently passes an expanded expiry column. Materialize it
        # once here instead of letting searchsorted make an implicit copy on
        # every call.
        query_expiries_y = query_expiries_y.contiguous()
        raw_upper = torch.searchsorted(self.model_expiries_y, query_expiries_y)
        upper = torch.clamp(raw_upper, max=T - 1)
        lower = torch.clamp(raw_upper - 1, min=0, max=T - 1)
        before_first = raw_upper == 0
        previous_time = torch.where(
            before_first,
            torch.zeros_like(query_expiries_y),
            self.model_expiries_y[lower],
        )
        interval = self.model_expiries_y[upper] - previous_time
        safe_interval = torch.clamp(
            interval,
            min=torch.finfo(query_expiries_y.dtype).tiny,
        )
        upper_weight = torch.clamp(
            (query_expiries_y - previous_time) / safe_interval,
            min=0.0,
            max=1.0,
        )
        return lower, upper, upper_weight, before_first

    def price_calls(
        self,
        surface: SanosSurface,
        *,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
    ) -> torch.Tensor:
        """Price one day's calls at all observed strikes and expiries."""
        assert expiries_y.shape == strikes.shape and strikes.ndim == 1
        batch_surface = SanosSurface(
            expiries_y=surface.expiries_y.unsqueeze(0),
            support_strikes=surface.support_strikes.unsqueeze(0),
            density0=surface.density0.unsqueeze(0),
            densities=surface.densities.unsqueeze(0),
            dlvsqs=surface.dlvsqs.unsqueeze(0),
            forward_vols=surface.forward_vols.unsqueeze(0),
            total_vols=surface.total_vols.unsqueeze(0),
            eta=surface.eta.unsqueeze(0),
            backbone_total_vols=surface.backbone_total_vols.unsqueeze(0),
        )
        return self.price_calls_batch(
            batch_surface,
            expiries_y=expiries_y.unsqueeze(0),
            strikes=strikes.unsqueeze(0),
        )[0]

    def price_calls_batch(
        self,
        surface: SanosSurface,
        *,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
    ) -> torch.Tensor:
        """Price padded batches with Dysanos interpolation and left extrapolation."""
        assert expiries_y.shape == strikes.shape and strikes.ndim == 2
        B, Q = strikes.shape
        assert surface.support_strikes.shape[:2] == (
            B,
            self.geometry.num_model_expiries,
        )
        lower, upper, alpha, before_first = self._torch_interpolation(expiries_y)
        batch_index = torch.arange(B, device=strikes.device).unsqueeze(1)

        upper_support = surface.support_strikes[batch_index, upper]
        upper_density = surface.densities[batch_index, upper]
        lower_support = surface.support_strikes[batch_index, lower]
        lower_density = surface.densities[batch_index, lower]
        first_support = surface.support_strikes[:, :1].expand(-1, Q, -1)
        initial_density = surface.density0[:, None, :].expand(-1, Q, -1)
        # Using the first learned density directly before T[0] would retain a
        # non-degenerate law as t -> 0 and therefore imply a jump at the origin.
        lower_support = torch.where(
            before_first.unsqueeze(-1),
            first_support,
            lower_support,
        )
        lower_density = torch.where(
            before_first.unsqueeze(-1),
            initial_density,
            lower_density,
        )

        upper_variance = surface.backbone_total_vols[
            batch_index,
            upper,
        ].square()
        lower_variance = surface.backbone_total_vols[
            batch_index,
            lower,
        ].square()
        lower_variance = torch.where(
            before_first,
            torch.zeros_like(lower_variance),
            lower_variance,
        )
        query_variance = (
            (1.0 - alpha) * lower_variance + alpha * upper_variance
        )
        positive_variance = query_variance > 0.0
        query_sqrt_variance = torch.where(
            positive_variance,
            torch.sqrt(
                torch.clamp(
                    query_variance,
                    min=torch.finfo(query_variance.dtype).tiny,
                )
            ),
            torch.zeros_like(query_variance),
        )

        upper_calls = torch.sum(
            self._black_call(
                upper_support,
                strikes.unsqueeze(-1),
                query_sqrt_variance.unsqueeze(-1),
            )
            * upper_density,
            dim=-1,
        )
        lower_calls = torch.sum(
            self._black_call(
                lower_support,
                strikes.unsqueeze(-1),
                query_sqrt_variance.unsqueeze(-1),
            )
            * lower_density,
            dim=-1,
        )
        return (1.0 - alpha) * lower_calls + alpha * upper_calls
