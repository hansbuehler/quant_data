"""Torch-native weekly SANOS path generation and option returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import NMS
from quantdata.dysanos.flex.config import (
    ForwardMuFlexGeometryConfig,
    JointHierarchicalDynamicsConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
    LatentStateSanosModel,
    bound_native_raw,
)
from quantdata.dysanos.flex.sanos import SanosSurface
from quantdata.dysanos.flex.spread_cost import (
    FittedSpreadCostModel,
    TorchSpreadCostModel,
)
from quantdata.dysanos.flex.weekly_dynamics import (
    STRONG_STATE_SIZES,
    TorchWeeklyDynamics,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    NormalWeeklyMeasureKernel,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForwardConfig


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class WeeklyOptionGridConfig:
    """Fixed inception-log-moneyness option grid and transaction cost."""

    log_strikes: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20)
    log_strikes_by_expiry: tuple[tuple[float, ...], ...] | None = None
    option_mode: str = "itm"
    proportional_cost: float = 0.0
    spot_proportional_cost: float = 0.0
    include_zero_strike_calls: bool = True

    def __post_init__(self) -> None:
        verify_inp(
            self.option_mode in {"itm", "otm"},
            lambda: f"option_mode must be 'itm' or 'otm', got {self.option_mode!r}",
        )
        verify_inp(
            math.isfinite(self.proportional_cost)
            and 0.0 <= self.proportional_cost < 1.0,
            lambda: f"invalid proportional option cost {self.proportional_cost}",
        )
        verify_inp(
            math.isfinite(self.spot_proportional_cost)
            and 0.0 <= self.spot_proportional_cost < 1.0,
            lambda: f"invalid proportional spot cost {self.spot_proportional_cost}",
        )
        verify_inp(
            isinstance(self.include_zero_strike_calls, bool),
            lambda: "include_zero_strike_calls must be bool",
        )
        strike_sets = (
            self.log_strikes_by_expiry
            if self.log_strikes_by_expiry is not None
            else (self.log_strikes,)
        )
        verify_inp(
            all(
                len(values) > 0
                and all(math.isfinite(value) for value in values)
                and all(right > left for left, right in zip(values[:-1], values[1:]))
                for values in strike_sets
            ),
            lambda: f"log-strike grids must be finite and strictly increasing: {strike_sets}",
        )


@dataclass(frozen=True)
class TradableOptionGrid:
    """One option side per strike and expiry, including cash-like calls."""

    expiry_index: torch.Tensor
    log_moneyness: torch.Tensor
    is_call: torch.Tensor
    is_zero_strike: torch.Tensor

    @property
    def num_options(self) -> int:
        return int(self.expiry_index.numel())


@dataclass(frozen=True)
class WeeklyPathBatch:
    """Retained part of a live-generated Torch path batch."""

    reduced_states: torch.Tensor
    log_spots: torch.Tensor

    @property
    def spots(self) -> torch.Tensor:
        return torch.exp(self.log_spots)


@dataclass(frozen=True)
class WeeklyOptionReturnBatch:
    """Period or to-expiry option returns for every entry time and path."""

    returns: torch.Tensor
    gross_returns: torch.Tensor
    entry_prices: torch.Tensor
    exit_values: torch.Tensor
    entry_costs: torch.Tensor
    exit_costs: torch.Tensor
    transaction_costs: torch.Tensor
    grid: TradableOptionGrid
    retained_paths: WeeklyPathBatch
    internal_simulation_steps: int


def make_tradable_option_grid(
    env: BaseEnvironment,
    *,
    num_expiries: int,
    config: WeeklyOptionGridConfig,
) -> TradableOptionGrid:
    """Create the same fixed log-moneyness specification at every entry time."""
    if config.log_strikes_by_expiry is not None:
        if len(config.log_strikes_by_expiry) != num_expiries:
            raise ValueError(
                "log_strikes_by_expiry must contain one grid per expiry; got "
                f"{len(config.log_strikes_by_expiry)} versus {num_expiries}"
            )
        strike_sets = config.log_strikes_by_expiry
    else:
        strike_sets = (config.log_strikes,) * num_expiries

    expiry_indices: list[int] = []
    log_moneyness: list[float] = []
    is_call: list[bool] = []
    is_zero: list[bool] = []
    for expiry_index, values in enumerate(strike_sets):
        if config.include_zero_strike_calls:
            expiry_indices.append(expiry_index)
            log_moneyness.append(float("-inf"))
            is_call.append(True)
            is_zero.append(True)
        for value in values:
            expiry_indices.append(expiry_index)
            log_moneyness.append(value)
            call = value <= 0.0 if config.option_mode == "itm" else value >= 0.0
            is_call.append(call)
            is_zero.append(False)
    return TradableOptionGrid(
        expiry_index=torch.tensor(expiry_indices, dtype=env.idtype, device=env.device),
        log_moneyness=torch.tensor(log_moneyness, dtype=env.fdtype, device=env.device),
        is_call=torch.tensor(is_call, dtype=torch.bool, device=env.device),
        is_zero_strike=torch.tensor(is_zero, dtype=torch.bool, device=env.device),
    )


def _front_mixture_log_cdf(
    log_value: torch.Tensor,
    *,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
) -> torch.Tensor:
    z = (
        log_value.unsqueeze(-1)
        - torch.log(support)
        + 0.5 * sqrt_variance.square().unsqueeze(-1)
    ) / sqrt_variance.unsqueeze(-1)
    return torch.sum(density * torch.special.ndtr(z), dim=-1)


def front_week_sanos_inverse_cdf(
    surface: SanosSurface,
    gaussian_coordinate: torch.Tensor,
    *,
    bisection_steps: int = 64,
) -> torch.Tensor:
    """Map a Gaussian copula coordinate through the current front SANOS law.

    The front distribution is the SANOS discrete martingale mixture convolved
    with its mean-one lognormal backbone. The bisection locates the quantile;
    a final implicit-function correction supplies the correct local gradient.
    """
    return front_week_inverse_cdf_components(
        surface.support_strikes[:, 0],
        surface.densities[:, 0],
        surface.backbone_total_vols[:, 0],
        gaussian_coordinate,
        bisection_steps=bisection_steps,
    )


def front_week_inverse_cdf_components(
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    gaussian_coordinate: torch.Tensor,
    *,
    bisection_steps: int = 64,
) -> torch.Tensor:
    """Invert cached front SANOS components for repeated Q-kernel training."""
    if support.ndim != 2 or density.shape != support.shape:
        raise ValueError("front support and density must be aligned batches")
    if gaussian_coordinate.shape != support.shape[:-1]:
        raise ValueError(
            "Gaussian coordinate must have one value per surface; got "
            f"{tuple(gaussian_coordinate.shape)} versus {tuple(support.shape[:-1])}"
        )
    if torch.any(sqrt_variance <= 0.0):
        raise ValueError("implied spot mode requires a positive front backbone variance")
    # The SANOS tridiagonal solve preserves mass, mean, and non-negativity up
    # to the decoder's floating-point tolerance.  An inverse CDF needs those
    # identities exactly: otherwise an upper-tail probability can exceed the
    # numerically available mass and drive the implicit derivative through a
    # near-zero density.  Remove only numerical negative mass, then normalize
    # both mass and mean.  This produces the closest simple martingale mixture
    # used for spot sampling without changing the calibrated pricing surface.
    sampling_density = torch.clamp_min(density, 0.0)
    sampling_mass = torch.sum(sampling_density, dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(sampling_mass)) or torch.any(sampling_mass <= 0.0):
        raise ValueError("front SANOS sampling density has no positive finite mass")
    sampling_density = sampling_density / sampling_mass
    sampling_mean = torch.sum(
        sampling_density * support,
        dim=-1,
        keepdim=True,
    )
    if torch.any(~torch.isfinite(sampling_mean)) or torch.any(sampling_mean <= 0.0):
        raise ValueError("front SANOS sampling density has no positive finite mean")
    sampling_support = support / sampling_mean
    eps = 32.0 * torch.finfo(gaussian_coordinate.dtype).eps
    probability = torch.clamp(torch.special.ndtr(gaussian_coordinate), eps, 1.0 - eps)
    lower = torch.log(sampling_support[:, 0]) - 10.0 * sqrt_variance
    upper = torch.log(sampling_support[:, -1]) + 10.0 * sqrt_variance
    with torch.no_grad():
        for _ in range(bisection_steps):
            middle = 0.5 * (lower + upper)
            cdf = _front_mixture_log_cdf(
                middle,
                support=sampling_support,
                density=sampling_density,
                sqrt_variance=sqrt_variance,
            )
            lower = torch.where(cdf < probability, middle, lower)
            upper = torch.where(cdf < probability, upper, middle)
        root = 0.5 * (lower + upper)
    z = (
        root.unsqueeze(-1)
        - torch.log(sampling_support)
        + 0.5 * sqrt_variance.square().unsqueeze(-1)
    ) / sqrt_variance.unsqueeze(-1)
    normal_pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    log_pdf = torch.sum(
        sampling_density * normal_pdf / sqrt_variance.unsqueeze(-1), dim=-1
    )
    cdf_at_root = _front_mixture_log_cdf(
        root,
        support=sampling_support,
        density=sampling_density,
        sqrt_variance=sqrt_variance,
    )
    implicit_correction = (probability - cdf_at_root) / torch.clamp(
        log_pdf, min=torch.finfo(log_pdf.dtype).tiny
    )
    # Bisection is the forward value.  The zero-valued straight-through term
    # supplies its implicit derivative without applying a numerically unstable
    # Newton update to the sampled spot itself.
    differentiable_root = (
        root + implicit_correction - implicit_correction.detach()
    )
    return torch.exp(differentiable_root)


def load_strong_calibrated_model(
    env: BaseEnvironment,
    checkpoint_path: Path,
) -> LatentStateSanosModel:
    """Reconstruct a converged Phase 2 or train-only Phase 3 Strong decoder."""
    checkpoint: Mapping[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    schema_version = checkpoint.get("schema_version")
    if checkpoint.get("phase") != "complete" or schema_version not in {1, 2}:
        raise ValueError(f"not a completed FLEX Strong checkpoint: {checkpoint_path}")
    if schema_version == 2:
        audits = checkpoint.get("training_audits")
        if not isinstance(audits, Mapping) or not all(
            isinstance(audits.get(name), Mapping)
            and audits[name].get("converged") is True
            for name in ("supervised", "market")
        ):
            raise ValueError("strong checkpoint has no successful convergence proof")
        identity = checkpoint.get("model_identity")
        model_config = (
            identity.get("model_config") if isinstance(identity, Mapping) else None
        )
        checkpoint_config_version = (
            identity.get("config_version") if isinstance(identity, Mapping) else None
        )
    else:
        convergence = checkpoint.get("convergence")
        if not isinstance(convergence, Mapping) or convergence.get("converged") is not True:
            raise ValueError("train-only Strong checkpoint has no convergence proof")
        fingerprints = checkpoint.get("train_market_fingerprints")
        native_digest = checkpoint.get("native_target_sha256")
        if (
            not isinstance(fingerprints, list)
            or len(fingerprints) == 0
            or not isinstance(native_digest, str)
            or len(native_digest) != 64
        ):
            raise ValueError("train-only Strong checkpoint provenance is incomplete")
        model_config = checkpoint.get("model_config")
        checkpoint_config_version = checkpoint.get("config_version")
    if not isinstance(model_config, Mapping):
        raise ValueError("strong checkpoint has no model configuration")
    allocation = tuple(model_config.get("state_sizes_by_expiry", ()))
    if allocation != STRONG_STATE_SIZES:
        raise ValueError(f"checkpoint is not the selected strong model: {allocation}")
    geometry_payload = model_config.get("geometry")
    network_payload = model_config.get("decoder_network")
    if not isinstance(geometry_payload, Mapping) or not isinstance(
        network_payload, Mapping
    ):
        raise ValueError("strong checkpoint geometry/network configuration is invalid")
    geometry = ForwardMuFlexGeometryConfig(**dict(geometry_payload))
    config = TriangularPcaDecoderConfig(
        label=str(model_config["label"]),
        group=str(model_config["group"]),
        user_version=str(model_config["user_version"]),
        state_sizes_by_expiry=allocation,
        state_size=int(model_config["state_size"]),
        geometry=geometry,
        decoder_network=FeedForwardConfig(**dict(network_payload)),
    )
    if checkpoint_config_version != config.version.unique_id48:
        raise ValueError(
            "strong checkpoint configuration code version is stale; cold rebuild required"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("strong checkpoint has no model state")
    daily_state = state_dict.get("daily_state_embedding.weight")
    if not isinstance(daily_state, torch.Tensor) or daily_state.ndim != 2:
        raise ValueError("strong checkpoint has no daily state embedding")
    if schema_version == 1 and len(checkpoint["train_market_fingerprints"]) != daily_state.shape[0]:
        raise ValueError("train-only Strong checkpoint date/state count mismatch")
    model = LatentStateSanosModel(
        env,
        num_days=daily_state.shape[0],
        config=config,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_shared_strikes_calibrated_model(
    env: BaseEnvironment,
    checkpoint_path: Path,
) -> LatentStateSanosModel:
    """Reconstruct a completed direct or hierarchical shared-strike decoder."""
    checkpoint: Mapping[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != 2 or checkpoint.get("phase") != "complete":
        raise ValueError(f"not a completed shared-strike decoder: {checkpoint_path}")
    audits = checkpoint.get("training_audits")
    if not isinstance(audits, Mapping) or not all(
        isinstance(audits.get(name), Mapping)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("shared-strike decoder has no convergence proof")
    identity = checkpoint.get("model_identity")
    model_config = identity.get("model_config") if isinstance(identity, Mapping) else None
    if not isinstance(model_config, Mapping):
        raise ValueError("shared-strike decoder has no model configuration")
    geometry_payload = model_config.get("geometry")
    network_payload = model_config.get("decoder_network")
    if not isinstance(geometry_payload, Mapping) or not isinstance(
        network_payload, Mapping
    ):
        raise ValueError("shared-strike decoder geometry/network is invalid")
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        **dict(geometry_payload)
    )
    common = {
        "label": str(model_config["label"]),
        "group": str(model_config["group"]),
        "user_version": str(model_config["user_version"]),
        "state_size": int(model_config["state_size"]),
        "geometry": geometry,
        "decoder_network": FeedForwardConfig(**dict(network_payload)),
    }
    raw_allocation = model_config.get("state_sizes_by_expiry")
    config: PcaDecoderConfig | TriangularPcaDecoderConfig
    if raw_allocation is None:
        config = PcaDecoderConfig(**common)
    else:
        config = TriangularPcaDecoderConfig(
            **common,
            state_sizes_by_expiry=tuple(int(value) for value in raw_allocation),
        )
    if not isinstance(identity, Mapping) or identity.get("config_version") != (
        config.version.unique_id48
    ):
        raise ValueError(
            "shared-strike decoder configuration code version is stale; "
            "cold rebuild required"
        )
    state_dict = checkpoint.get("model_state_dict")
    daily_state = (
        state_dict.get("daily_state_embedding.weight")
        if isinstance(state_dict, Mapping)
        else None
    )
    if not isinstance(daily_state, torch.Tensor) or daily_state.ndim != 2:
        raise ValueError("shared-strike decoder has no daily state embedding")
    model = LatentStateSanosModel(
        env,
        num_days=daily_state.shape[0],
        config=config,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class WeeklySanosSimulator(BaseModel):
    """Decode simulated compressed states and produce live option-return paths."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        dynamics: TorchWeeklyDynamics | JointHierarchicalWeeklyDynamicsAdapter,
        calibrated_model: LatentStateSanosModel | JointHierarchicalDynamicsModel,
    ):
        super().__init__(env)
        if dynamics.env is not env or calibrated_model.env is not env:
            raise ValueError("all weekly simulator child models must share one environment")
        if isinstance(calibrated_model.config, TriangularPcaDecoderConfig):
            decoder_blocks = calibrated_model.config.state_sizes_by_expiry
        elif isinstance(calibrated_model.config, PcaDecoderConfig):
            decoder_blocks = (calibrated_model.config.state_size,)
        elif isinstance(calibrated_model.config, JointHierarchicalDynamicsConfig):
            decoder_blocks = calibrated_model.config.state_sizes_by_expiry
        else:
            raise ValueError("weekly simulation requires an affine FLEX decoder")
        if decoder_blocks != dynamics.config.state_sizes_by_expiry:
            raise ValueError(
                "dynamics and calibrated decoder state allocations differ: "
                f"{dynamics.config.state_sizes_by_expiry} versus {decoder_blocks}"
            )
        self.dynamics = dynamics
        self.calibrated_model = calibrated_model

    @property
    def num_model_expiries(self) -> int:
        return self.calibrated_model.config.geometry.num_model_expiries

    def decode_surface(self, reduced_state: torch.Tensor) -> SanosSurface:
        leading = reduced_state.shape[:-1]
        flat = reduced_state.reshape(-1, reduced_state.shape[-1])
        full_state = self.dynamics.decode_full_state(flat)
        raw = self.calibrated_model.raw_states_from_states(full_state)
        parameters = bound_native_raw(
            raw, geometry=self.calibrated_model.config.geometry
        )
        surface = self.calibrated_model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        if len(leading) != 1:
            raise ValueError("decode_surface currently expects one batch axis")
        return surface

    def _simulate_internal(
        self,
        initial_reduced_state: torch.Tensor,
        initial_spot: torch.Tensor,
        *,
        num_paths: int,
        internal_steps: int,
        spot_mode: str,
        standard_normals: torch.Tensor | None,
        generator: torch.Generator | None,
        measure_kernel: (
            NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel | None
        ) = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spot_mode not in {"normal", "implied"}:
            raise ValueError(f"spot_mode must be 'normal' or 'implied', got {spot_mode!r}")
        if initial_reduced_state.shape == (self.dynamics.reduced_state_size,):
            state = initial_reduced_state.unsqueeze(0).expand(num_paths, -1)
        elif initial_reduced_state.shape == (num_paths, self.dynamics.reduced_state_size):
            state = initial_reduced_state
        else:
            raise ValueError("invalid initial reduced-state shape")
        if initial_spot.ndim == 0:
            spot = initial_spot.expand(num_paths)
        elif initial_spot.shape == (num_paths,):
            spot = initial_spot
        else:
            raise ValueError("initial_spot must be scalar or contain one value per path")
        if torch.any(spot <= 0.0):
            raise ValueError("initial_spot must be positive")
        if standard_normals is not None and standard_normals.shape != (
            num_paths,
            internal_steps,
            self.dynamics.reduced_state_size + 1,
        ):
            raise ValueError("standard_normals has the wrong path geometry")
        if measure_kernel is not None:
            if measure_kernel.dynamics is not self.dynamics:
                raise ValueError("measure kernel and simulator must share P dynamics")
            if spot_mode != "implied":
                raise ValueError(
                    "target-measure simulation requires implied front-week spot mode"
                )

        states = [state]
        log_spots = [torch.log(spot)]
        for step_index in range(internal_steps):
            current_state = states[-1]
            current_log_spot = log_spots[-1]
            step_normals = (
                torch.randn(
                    (num_paths, self.dynamics.reduced_state_size + 1),
                    dtype=self.fdtype,
                    device=self.device,
                    generator=generator,
                )
                if standard_normals is None
                else standard_normals[:, step_index]
            )
            if measure_kernel is not None:
                current_surface = self.decode_surface(current_state)
                gross_return = front_week_sanos_inverse_cdf(
                    current_surface, step_normals[..., 0]
                )
                measure_draw = measure_kernel.step(
                    current_state,
                    gross_return,
                    step_normals,
                )
                states.append(measure_draw.reduced_state)
                log_spots.append(current_log_spot + torch.log(gross_return))
                continue
            step = self.dynamics.step(
                current_state,
                current_log_spot,
                standard_normals=step_normals,
            )
            if spot_mode == "normal":
                next_log_spot = step.normal_log_spot
            else:
                # Decode before applying the shock: using the next state here
                # would leak contemporaneous future surface information.
                current_surface = self.decode_surface(current_state)
                gross_return = front_week_sanos_inverse_cdf(
                    current_surface, step.spot_standard_normal
                )
                next_log_spot = current_log_spot + torch.log(gross_return)
            states.append(step.reduced_state)
            log_spots.append(next_log_spot)
        return torch.stack(states, dim=1), torch.stack(log_spots, dim=1)

    def simulate_paths(
        self,
        initial_reduced_state: torch.Tensor,
        initial_spot: torch.Tensor,
        *,
        num_paths: int,
        num_steps: int,
        spot_mode: str = "normal",
        standard_normals: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        measure_kernel: (
            NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel | None
        ) = None,
    ) -> WeeklyPathBatch:
        states, log_spots = self._simulate_internal(
            initial_reduced_state,
            initial_spot,
            num_paths=num_paths,
            internal_steps=num_steps,
            spot_mode=spot_mode,
            standard_normals=standard_normals,
            generator=generator,
            measure_kernel=measure_kernel,
        )
        return WeeklyPathBatch(reduced_states=states, log_spots=log_spots)

    def _option_values(
        self,
        surface: SanosSurface,
        spot: torch.Tensor,
        strikes: torch.Tensor,
        expiry_index: torch.Tensor,
        is_call: torch.Tensor,
        is_zero: torch.Tensor,
    ) -> torch.Tensor:
        safe_strikes = torch.where(is_zero, spot.unsqueeze(-1), strikes)
        normalized_strikes = safe_strikes / spot.unsqueeze(-1)
        expiries = self.calibrated_model.sanos_decoder.model_expiries_y[
            expiry_index
        ].unsqueeze(0).expand_as(normalized_strikes)
        calls = spot.unsqueeze(-1) * self.calibrated_model.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiries,
            strikes=normalized_strikes,
        )
        # A zero-strike call is S exactly. Avoid passing K=0 into Black's log.
        calls = torch.where(is_zero, spot.unsqueeze(-1), calls)
        puts = calls - spot.unsqueeze(-1) + strikes
        return torch.where(is_call, calls, puts)

    def _spread_cost_rates(
        self,
        surface: SanosSurface,
        spot: torch.Tensor,
        option_prices: torch.Tensor,
        grid: TradableOptionGrid,
        model: FittedSpreadCostModel | TorchSpreadCostModel,
        *,
        spot_proportional_cost: float,
    ) -> torch.Tensor:
        """Evaluate the fitted linear spread attribution on simulated books.

        The SANOS strike-total-volatility curve supplies the differentiable ATM
        volatility proxy used for NMS and Black Greeks. This is an approximation
        to the smooth-reference implied volatility used to train the spread
        regression and is reported as such in Phase 3 diagnostics.
        """
        log_strike = torch.where(
            grid.is_zero_strike,
            torch.zeros_like(grid.log_moneyness),
            grid.log_moneyness,
        ).unsqueeze(0).expand_as(option_prices)
        expiry_index = grid.expiry_index.unsqueeze(0).expand_as(option_prices)
        sqrt_tte = torch.sqrt(
            self.calibrated_model.sanos_decoder.model_expiries_y[grid.expiry_index]
        ).unsqueeze(0).expand_as(option_prices)
        total_vol = torch.gather(surface.total_vols, dim=1, index=expiry_index)
        safe_total_vol = torch.clamp_min(total_vol, 1.0e-10)
        d1 = -log_strike / safe_total_vol + 0.5 * safe_total_vol
        call_delta = torch.special.ndtr(d1)
        delta = torch.where(grid.is_call.unsqueeze(0), call_delta, call_delta - 1.0)
        vega = (
            torch.exp(-0.5 * d1.square())
            / math.sqrt(2.0 * math.pi)
            * sqrt_tte
        )
        first_tte = self.calibrated_model.sanos_decoder.model_expiries_y[0]
        epsilon_vol = surface.total_vols[:, :1] / torch.sqrt(first_tte)
        nms_scale = torch.sqrt(
            (
                epsilon_vol.square() * float(NMS.EPS)
                + surface.total_vols.square()
            )
            / (1.0 + float(NMS.EPS))
        )
        selected_nms_scale = torch.gather(
            nms_scale,
            dim=1,
            index=expiry_index,
        )
        nms = log_strike / torch.clamp_min(selected_nms_scale, 1.0e-10)
        normalized_price = option_prices / spot.unsqueeze(-1)
        common = {
            "vega": vega,
            "sqrt_tte": sqrt_tte,
            "nms": nms,
            "log_strike": log_strike,
            "delta": delta,
        }
        if isinstance(model, TorchSpreadCostModel):
            cost = model.predict_proportional_cost_torch(
                normalized_price,
                spot=spot.unsqueeze(-1),
                **common,
            )
        else:
            cost = model.predict_proportional_cost_torch(
                normalized_price,
                **common,
            )
        return torch.where(
            grid.is_zero_strike.unsqueeze(0),
            torch.full_like(cost, spot_proportional_cost),
            cost,
        )

    def hjm_roll_residuals(
        self,
        current_reduced_state: torch.Tensor,
        next_reduced_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        normalized_strikes: torch.Tensor,
        *,
        current_call_prices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Return pathwise weekly call-price martingale residuals.

        For model expiry ``j >= 2`` this computes

        ``R * c_next[j-1](k/R) - c_current[j](k)``.

        For the front expiry it uses ``relu(R-k) - c_current[1](k)``.
        Conditional expectations of these residuals are the discrete weekly
        HJM/roll restrictions. The method deliberately returns pathwise values:
        a trainer can impose conditional moments without incorrectly forcing
        every stochastic payoff residual to zero.
        """
        if (
            current_reduced_state.ndim != 2
            or next_reduced_state.shape != current_reduced_state.shape
            or current_reduced_state.shape[1] != self.dynamics.reduced_state_size
        ):
            raise ValueError(
                "current and next state must be aligned (paths,reduced_state) tensors"
            )
        num_paths = current_reduced_state.shape[0]
        if gross_spot_return.shape != (num_paths,) or torch.any(
            gross_spot_return <= 0.0
        ):
            raise ValueError("gross_spot_return must be one positive value per path")
        if normalized_strikes.ndim == 1:
            strikes = normalized_strikes.unsqueeze(0).expand(num_paths, -1)
        elif normalized_strikes.ndim == 2 and normalized_strikes.shape[0] == num_paths:
            strikes = normalized_strikes
        else:
            raise ValueError(
                "normalized_strikes must have shape (strikes,) or (paths,strikes)"
            )
        if torch.any(~torch.isfinite(strikes)) or torch.any(strikes <= 0.0):
            raise ValueError("HJM normalized strikes must be finite and positive")

        num_strikes = strikes.shape[1]
        num_expiries = self.num_model_expiries
        if current_call_prices is None:
            current_surface = self.decode_surface(current_reduced_state)
            current_expiries = self.calibrated_model.sanos_decoder.model_expiries_y.view(
                1, num_expiries, 1
            ).expand(num_paths, num_expiries, num_strikes)
            current_strikes = strikes.unsqueeze(1).expand(
                num_paths, num_expiries, num_strikes
            )
            current_calls = self.calibrated_model.sanos_decoder.price_calls_batch(
                current_surface,
                expiries_y=current_expiries.reshape(num_paths, -1),
                strikes=current_strikes.reshape(num_paths, -1),
            ).reshape(num_paths, num_expiries, num_strikes)
        else:
            current_calls = current_call_prices
            if current_calls.shape != (num_paths, num_expiries, num_strikes):
                raise ValueError("precomputed current call surface has the wrong shape")

        gross = gross_spot_return.view(num_paths, 1)
        rolled_values = [torch.relu(gross - strikes).unsqueeze(1)]
        if num_expiries > 1:
            next_surface = self.decode_surface(next_reduced_state)
            next_expiries = self.calibrated_model.sanos_decoder.model_expiries_y[
                :-1
            ].view(1, num_expiries - 1, 1).expand(
                num_paths, num_expiries - 1, num_strikes
            )
            next_strikes = (strikes / gross).unsqueeze(1).expand(
                num_paths, num_expiries - 1, num_strikes
            )
            next_calls = self.calibrated_model.sanos_decoder.price_calls_batch(
                next_surface,
                expiries_y=next_expiries.reshape(num_paths, -1),
                strikes=next_strikes.reshape(num_paths, -1),
            ).reshape(num_paths, num_expiries - 1, num_strikes)
            rolled_values.append(gross.unsqueeze(1) * next_calls)
        rolled = torch.cat(rolled_values, dim=1)
        residuals = rolled - current_calls
        self.verify(
            lambda: torch.all(torch.isfinite(residuals)),
            "non-finite weekly HJM roll residual",
        )
        return residuals

    def normalized_call_prices(
        self,
        reduced_state: torch.Tensor,
        *,
        expiry_index: int,
        normalized_strikes: torch.Tensor,
    ) -> torch.Tensor:
        """Price one model-expiry call line in normalized spot units."""
        if not 0 <= expiry_index < self.num_model_expiries:
            raise ValueError(f"invalid model expiry index {expiry_index}")
        if reduced_state.ndim != 2:
            raise ValueError("reduced_state must have shape (batch,state)")
        batch = reduced_state.shape[0]
        if normalized_strikes.ndim == 1:
            strikes = normalized_strikes.unsqueeze(0).expand(batch, -1)
        elif normalized_strikes.ndim == 2 and normalized_strikes.shape[0] == batch:
            strikes = normalized_strikes
        else:
            raise ValueError(
                "normalized_strikes must have shape (strikes,) or (batch,strikes)"
            )
        if torch.any(~torch.isfinite(strikes)) or torch.any(strikes <= 0.0):
            raise ValueError("normalized call strikes must be finite and positive")
        surface = self.decode_surface(reduced_state)
        expiry = self.calibrated_model.sanos_decoder.model_expiries_y[
            expiry_index
        ].expand_as(strikes)
        return self.calibrated_model.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiry,
            strikes=strikes,
        )

    def hjm_roll_line_residuals(
        self,
        current_reduced_state: torch.Tensor,
        next_reduced_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        normalized_strikes: torch.Tensor,
        *,
        current_expiry_index: int,
        current_call_prices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Price only one fixed-cash-strike weekly roll restriction.

        ``current_expiry_index=j`` compares current line ``j`` with next line
        ``j-1``. The front line ``j=0`` rolls directly to payoff. Supplying
        precomputed ``current_call_prices`` avoids repeatedly decoding the
        fixed current state during target-kernel Monte Carlo training.
        """
        if current_reduced_state.shape != next_reduced_state.shape:
            raise ValueError("current and next reduced-state batches must align")
        batch = current_reduced_state.shape[0]
        if gross_spot_return.shape != (batch,) or torch.any(
            gross_spot_return <= 0.0
        ):
            raise ValueError("gross_spot_return must be one positive value per path")
        if normalized_strikes.ndim == 1:
            strikes = normalized_strikes.unsqueeze(0).expand(batch, -1)
        elif normalized_strikes.ndim == 2 and normalized_strikes.shape[0] == batch:
            strikes = normalized_strikes
        else:
            raise ValueError("normalized_strikes has the wrong roll-line shape")
        current_calls = (
            self.normalized_call_prices(
                current_reduced_state,
                expiry_index=current_expiry_index,
                normalized_strikes=strikes,
            )
            if current_call_prices is None
            else current_call_prices
        )
        if current_calls.shape != strikes.shape:
            raise ValueError("precomputed current calls do not match roll strikes")
        if current_expiry_index == 0:
            rolled = torch.relu(gross_spot_return.unsqueeze(-1) - strikes)
        else:
            gross = gross_spot_return.unsqueeze(-1)
            next_calls = self.normalized_call_prices(
                next_reduced_state,
                expiry_index=current_expiry_index - 1,
                normalized_strikes=strikes / gross,
            )
            rolled = gross * next_calls
        residuals = rolled - current_calls
        self.verify(
            lambda: torch.all(torch.isfinite(residuals)),
            "non-finite weekly HJM roll-line residual",
        )
        return residuals

    def simulate_option_returns(
        self,
        initial_reduced_state: torch.Tensor,
        initial_spot: torch.Tensor,
        *,
        num_paths: int,
        num_steps: int,
        grid_config: WeeklyOptionGridConfig = WeeklyOptionGridConfig(),
        return_mode: str = "period",
        spot_mode: str = "normal",
        proportional_costs: torch.Tensor | None = None,
        spread_cost_model: FittedSpreadCostModel | TorchSpreadCostModel | None = None,
        standard_normals: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        measure_kernel: (
            NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel | None
        ) = None,
    ) -> WeeklyOptionReturnBatch:
        """Generate one-week or hold-to-expiry returns without pre-generated paths.

        For entry time ``t`` and option ``q`` with expiry index ``j(q)``,

        ``R_period = (V[t+1,j-1] - V[t,j] - c(V[t,j]+V[t+1,j-1])) / V[t,j]``

        where ``V[t+1,-1]`` means expiry payoff. The to-expiry version replaces
        the exit value by payoff at ``t+j+1`` and charges entry cost only,
        because cash settlement has no option half-spread. With ``num_steps=M``
        and ``M`` expiries this requires exactly ``2M-1`` internal weekly
        increments.
        """
        if return_mode not in {"period", "expiry"}:
            raise ValueError("return_mode must be 'period' or 'expiry'")
        if num_paths <= 0 or num_steps <= 0:
            raise ValueError("num_paths and num_steps must be positive")
        if proportional_costs is not None and spread_cost_model is not None:
            raise ValueError(
                "Specify either proportional_costs or spread_cost_model, not both"
            )
        grid = make_tradable_option_grid(
            self.env,
            num_expiries=self.num_model_expiries,
            config=grid_config,
        )
        internal_steps = (
            num_steps
            if return_mode == "period"
            else num_steps + self.num_model_expiries - 1
        )
        states, log_spots = self._simulate_internal(
            initial_reduced_state,
            initial_spot,
            num_paths=num_paths,
            internal_steps=internal_steps,
            spot_mode=spot_mode,
            standard_normals=standard_normals,
            generator=generator,
            measure_kernel=measure_kernel,
        )
        spots = torch.exp(log_spots)
        finite_log_moneyness = torch.where(
            grid.is_zero_strike,
            torch.zeros_like(grid.log_moneyness),
            grid.log_moneyness,
        )
        entries: list[torch.Tensor] = []
        exits: list[torch.Tensor] = []
        fitted_cost_rates: list[torch.Tensor] = []
        for entry_time in range(num_steps):
            entry_spot = spots[:, entry_time]
            strikes = entry_spot.unsqueeze(-1) * torch.exp(finite_log_moneyness)
            strikes = torch.where(
                grid.is_zero_strike, torch.zeros_like(strikes), strikes
            )
            entry_surface = self.decode_surface(states[:, entry_time])
            entry_value = self._option_values(
                entry_surface,
                entry_spot,
                strikes,
                grid.expiry_index,
                grid.is_call,
                grid.is_zero_strike,
            )
            if spread_cost_model is not None:
                fitted_cost_rates.append(
                    self._spread_cost_rates(
                        entry_surface,
                        entry_spot,
                        entry_value,
                        grid,
                        spread_cost_model,
                        spot_proportional_cost=grid_config.spot_proportional_cost,
                    )
                )
            if return_mode == "expiry":
                expiry_times = entry_time + grid.expiry_index + 1
                exit_spot = spots[:, expiry_times]
                exit_value = torch.where(
                    grid.is_call,
                    torch.relu(exit_spot - strikes),
                    torch.relu(strikes - exit_spot),
                )
            else:
                next_spot = spots[:, entry_time + 1]
                expiring = grid.expiry_index == 0
                payoff = torch.where(
                    grid.is_call,
                    torch.relu(next_spot.unsqueeze(-1) - strikes),
                    torch.relu(strikes - next_spot.unsqueeze(-1)),
                )
                remaining_index = torch.clamp(grid.expiry_index - 1, min=0)
                next_surface = self.decode_surface(states[:, entry_time + 1])
                continuation = self._option_values(
                    next_surface,
                    next_spot,
                    strikes,
                    remaining_index,
                    grid.is_call,
                    grid.is_zero_strike,
                )
                exit_value = torch.where(expiring, payoff, continuation)
            entries.append(entry_value)
            exits.append(exit_value)
        entry_prices = torch.stack(entries, dim=1)
        exit_values = torch.stack(exits, dim=1)
        if spread_cost_model is not None:
            cost_rates = torch.stack(fitted_cost_rates, dim=1)
        elif proportional_costs is None:
            cost_rates = torch.full_like(entry_prices, grid_config.proportional_cost)
            cost_rates = torch.where(
                grid.is_zero_strike.view(1, 1, -1),
                torch.full_like(cost_rates, grid_config.spot_proportional_cost),
                cost_rates,
            )
        else:
            try:
                cost_rates = torch.broadcast_to(proportional_costs, entry_prices.shape)
            except RuntimeError as error:
                raise ValueError(
                    "proportional_costs must broadcast to path/time/option shape "
                    f"{tuple(entry_prices.shape)}, got {tuple(proportional_costs.shape)}"
                ) from error
            if torch.any(~torch.isfinite(cost_rates)) or torch.any(cost_rates < 0.0):
                raise ValueError("proportional_costs must be finite and non-negative")
        entry_costs = cost_rates * entry_prices
        exit_costs = (
            torch.zeros_like(exit_values)
            if return_mode == "expiry"
            else cost_rates * exit_values
        )
        transaction_costs = entry_costs + exit_costs
        denominator = torch.clamp(
            entry_prices, min=torch.finfo(entry_prices.dtype).tiny
        )
        gross_returns = (exit_values - entry_prices) / denominator
        returns = (exit_values - entry_prices - transaction_costs) / denominator
        self.verify(lambda: torch.all(torch.isfinite(returns)), "non-finite option return")
        return WeeklyOptionReturnBatch(
            returns=returns,
            gross_returns=gross_returns,
            entry_prices=entry_prices,
            exit_values=exit_values,
            entry_costs=entry_costs,
            exit_costs=exit_costs,
            transaction_costs=transaction_costs,
            grid=grid,
            retained_paths=WeeklyPathBatch(
                reduced_states=states[:, : num_steps + 1],
                log_spots=log_spots[:, : num_steps + 1],
            ),
            internal_simulation_steps=internal_steps,
        )


def hjm_roll_moment_loss(
    residuals: torch.Tensor,
    conditioning_instruments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Square unconditional or instrumented conditional HJM roll moments.

    ``conditioning_instruments`` has shape ``(paths,moments)`` and should be a
    function only of information at the start of the transition. A column of
    ones enforces the unconditional restriction. This loss is intended for a
    later target-measure fit; Phase 3 does not claim that its historical
    identity measure already satisfies risk-neutral HJM conditions.
    """
    if residuals.ndim != 3 or residuals.shape[0] == 0:
        raise ValueError("residuals must have shape (paths,expiries,strikes)")
    if conditioning_instruments is None:
        instruments = torch.ones(
            (residuals.shape[0], 1),
            dtype=residuals.dtype,
            device=residuals.device,
        )
    else:
        instruments = conditioning_instruments
        if (
            instruments.ndim != 2
            or instruments.shape[0] != residuals.shape[0]
            or instruments.device != residuals.device
            or instruments.dtype != residuals.dtype
        ):
            raise ValueError(
                "conditioning instruments must match residual path, dtype, and device"
            )
    moments = torch.einsum("bji,bl->lji", residuals, instruments) / residuals.shape[0]
    return torch.mean(moments.square())


__all__ = [
    "TradableOptionGrid",
    "WeeklyOptionGridConfig",
    "WeeklyOptionReturnBatch",
    "WeeklyPathBatch",
    "WeeklySanosSimulator",
    "front_week_sanos_inverse_cdf",
    "front_week_inverse_cdf_components",
    "hjm_roll_moment_loss",
    "load_strong_calibrated_model",
    "make_tradable_option_grid",
]
