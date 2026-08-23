"""Set-invariant market-to-Strong encoder for the Phase 3 prerequisite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.biz_calendar import NMS
from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.metrics import (
    excess_statistics,
    normalized_excess,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.model import (
    LatentStateSanosModel,
    bound_native_raw,
)
from quantdata.dysanos.flex.training import (
    TorchMarketBatch,
    equal_date_quote_objective,
    materialize_batch,
    materialize_day,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

STRONG_STATE_SIZES = (16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5)

_QUOTE_FEATURE_NAMES = (
    "expiry_fraction",
    "nms",
    "log_strike",
    "is_call",
    "bid",
    "ask",
    "fit",
    "log_half_spread",
    "log_volume",
    "log_open_interest",
)


@version("1.0", dependencies=[ForwardMuFlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class RbfQuoteEncoderConfig:
    """Logical architecture for a fixed-RBF set encoder."""

    label: str
    group: str = "flex/phase3-market-encoder"
    user_version: str = "1.0"
    geometry: ForwardMuFlexGeometryConfig = ForwardMuFlexGeometryConfig()
    state_sizes_by_expiry: tuple[int, ...] = STRONG_STATE_SIZES
    num_expiry_anchors: int = 14
    num_nms_anchors: int = 11
    expiry_bandwidth_business_days: float = 5.0
    nms_bandwidth: float = 0.50
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=512,
    )

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.label),
            lambda: f"'label' must be a valid filename, found {self.label!r}",
        )
        verify_inp(
            all(part not in {"", ".", ".."} and is_filename(part) for part in Path(self.group).parts),
            lambda: f"'group' must be a valid relative directory, found {self.group!r}",
        )
        verify_inp(
            self.state_sizes_by_expiry == STRONG_STATE_SIZES,
            lambda: "Phase 3 encoder must use the selected Strong allocation",
        )
        verify_inp(
            self.num_expiry_anchors > 1 and self.num_nms_anchors > 2,
            lambda: "RBF anchor counts are too small",
        )
        verify_inp(
            self.expiry_bandwidth_business_days > 0.0 and self.nms_bandwidth > 0.0,
            lambda: "RBF bandwidths must be positive",
        )

    @property
    def state_size(self) -> int:
        """Selected Strong state dimension."""
        return sum(self.state_sizes_by_expiry)

    @property
    def num_quote_features(self) -> int:
        """Number of normalized values accumulated at each RBF anchor."""
        return len(_QUOTE_FEATURE_NAMES)

    @property
    def summary_size(self) -> int:
        """Flattened RBF means plus one occupancy channel per anchor."""
        return (
            self.num_expiry_anchors
            * self.num_nms_anchors
            * (self.num_quote_features + 1)
        )

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        return named_unique_filename48_8(self.label, self)


@version("1.0", dependencies=[RbfQuoteEncoderConfig])
@dataclass(frozen=True, kw_only=True)
class HierarchicalRbfQuoteEncoderConfig(RbfQuoteEncoderConfig):
    """Expiry-causal RBF encoder with one conditional network per state block."""

    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=128,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(
            self.num_expiry_anchors == len(self.state_sizes_by_expiry),
            lambda: (
                "Hierarchical encoder requires one ordered RBF expiry anchor "
                "per Strong state block"
            ),
        )

    @property
    def summary_size_per_expiry(self) -> int:
        """Number of flattened normalized summary values at one expiry anchor."""
        return self.num_nms_anchors * (self.num_quote_features + 1)


@version("1.0", dependencies=[CappedUnboundedDlvFlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class NativeRbfQuoteEncoderConfig:
    """Flat market encoder into either full or rank-reduced native state."""

    label: str
    state_size: int
    group: str = "flex/phase3-native-market-encoder"
    user_version: str = "1.0"
    geometry: CappedUnboundedDlvFlexGeometryConfig = (
        CappedUnboundedDlvFlexGeometryConfig()
    )
    num_expiry_anchors: int = 14
    num_nms_anchors: int = 11
    expiry_bandwidth_business_days: float = 5.0
    nms_bandwidth: float = 0.50
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=512,
    )
    native_decoder: FeedForwardConfig = FeedForwardConfig.linear_config()

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.label),
            lambda: f"'label' must be a valid filename, found {self.label!r}",
        )
        verify_inp(
            all(
                part not in {"", ".", ".."} and is_filename(part)
                for part in Path(self.group).parts
            ),
            lambda: f"'group' must be a valid relative directory, found {self.group!r}",
        )
        verify_inp(
            0 < self.state_size <= self.geometry.native_state_size,
            lambda: (
                "Expected 0 < state_size <= native_state_size, found "
                f"{self.state_size} and {self.geometry.native_state_size}"
            ),
        )
        verify_inp(
            self.num_expiry_anchors > 1 and self.num_nms_anchors > 2,
            lambda: "RBF anchor counts are too small",
        )
        verify_inp(
            self.expiry_bandwidth_business_days > 0.0
            and self.nms_bandwidth > 0.0,
            lambda: "RBF bandwidths must be positive",
        )
        verify_inp(
            self.native_decoder.num_depth == 0,
            lambda: "The compressed native-state decoder must remain affine",
        )

    @property
    def num_quote_features(self) -> int:
        """Number of normalized quote channels accumulated per anchor."""
        return len(_QUOTE_FEATURE_NAMES)

    @property
    def summary_size(self) -> int:
        """Flattened RBF means plus one occupancy channel per anchor."""
        return (
            self.num_expiry_anchors
            * self.num_nms_anchors
            * (self.num_quote_features + 1)
        )

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class EncoderTrainConfig:
    """Operational controls for one encoder training phase."""

    num_epochs: int
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip: float = 10.0
    seed: int = 2718
    checkpoint_every: int = 10
    report_every: int = 10
    convergence_window: int = 25
    convergence_relative_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        verify_inp(self.num_epochs > 0, lambda: "'num_epochs' must be positive")
        verify_inp(self.batch_size > 0, lambda: "'batch_size' must be positive")
        verify_inp(self.learning_rate > 0.0, lambda: "'learning_rate' must be positive")
        verify_inp(self.weight_decay >= 0.0, lambda: "'weight_decay' cannot be negative")
        verify_inp(self.gradient_clip > 0.0, lambda: "'gradient_clip' must be positive")
        verify_inp(self.checkpoint_every > 0, lambda: "'checkpoint_every' must be positive")
        verify_inp(self.report_every > 0, lambda: "'report_every' must be positive")
        verify_inp(self.convergence_window > 0, lambda: "'convergence_window' must be positive")
        verify_inp(
            0.0 < self.convergence_relative_tolerance < 1.0,
            lambda: "'convergence_relative_tolerance' must lie in (0,1)",
        )


@dataclass(frozen=True)
class RbfFeatureStatistics:
    """Training-only normalizers for quote channels and RBF summaries."""

    quote_mean: np.ndarray
    quote_scale: np.ndarray
    summary_mean: np.ndarray
    summary_scale: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize the fitted statistics without pickle."""
        return {
            "quote_mean": self.quote_mean.tolist(),
            "quote_scale": self.quote_scale.tolist(),
            "summary_mean": self.summary_mean.tolist(),
            "summary_scale": self.summary_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RbfFeatureStatistics:
        """Restore checkpointed training-only normalizers."""
        return cls(
            quote_mean=np.asarray(payload["quote_mean"], dtype=np.float64),
            quote_scale=np.asarray(payload["quote_scale"], dtype=np.float64),
            summary_mean=np.asarray(payload["summary_mean"], dtype=np.float64),
            summary_scale=np.asarray(payload["summary_scale"], dtype=np.float64),
        )


@dataclass(frozen=True)
class EncoderEvaluation:
    """Frozen encoder diagnostics for one temporal split."""

    num_days: int
    num_quotes: int
    quote_weighted_objective: float
    date_weighted_objective: float
    all_statistics: dict[str, int | float]
    weekly_statistics: dict[str, int | float]
    surface_fit: dict[str, int | float]
    weekly_surface_fit: dict[str, int | float]
    state_mse: float | None
    state_mae: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics."""
        return asdict(self)


@dataclass(frozen=True)
class EncodedStrongHistory:
    """Frozen date-aligned encoder states and cash spots for dynamics fitting."""

    dates: tuple[date, ...]
    market_fingerprints: tuple[str, ...]
    states: np.ndarray
    spot_refs: np.ndarray
    encoder_checkpoint_sha256: str
    strong_decoder_sha256: str

    def __post_init__(self) -> None:
        num_days = len(self.dates)
        if self.states.shape != (num_days, 140):
            raise ValueError("Encoded Strong history must have shape (dates,140)")
        if self.spot_refs.shape != (num_days,):
            raise ValueError("Encoded Strong history must have one spot per date")
        if len(self.market_fingerprints) != num_days:
            raise ValueError("Encoded Strong history fingerprint count mismatch")
        if any(left >= right for left, right in zip(self.dates, self.dates[1:])):
            raise ValueError("Encoded Strong history dates must be strictly increasing")
        if any(value.year >= 2026 for value in self.dates):
            raise ValueError("Encoded Strong history must never contain 2026 data")
        if not np.all(np.isfinite(self.states)):
            raise ValueError("Encoded Strong history contains non-finite states")
        if not np.all(np.isfinite(self.spot_refs) & (self.spot_refs > 0.0)):
            raise ValueError("Encoded Strong history contains invalid spot references")


class RbfQuoteEncoder(BaseModel):
    """Map a fixed-size Gaussian-RBF set summary to the Strong state."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: RbfQuoteEncoderConfig,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ):
        super().__init__(env)
        if target_mean.shape != (config.state_size,) or target_scale.shape != (
            config.state_size,
        ):
            raise ValueError("Strong target normalizer has the wrong shape")
        if not torch.all(target_scale > 0.0):
            raise ValueError("Every Strong target scale must be positive")
        self.config = config
        self.register_buffer(
            "target_mean",
            target_mean.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "target_scale",
            target_scale.to(device=env.device, dtype=env.fdtype),
        )
        self.network = FeedForward(
            env,
            num_input=config.summary_size,
            num_output=config.state_size,
            config=config.network,
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Encode one or more already-normalized invariant summaries."""
        if summary.shape[-1] != self.config.summary_size:
            raise ValueError(
                f"Expected summary size {self.config.summary_size}, found {tuple(summary.shape)}"
            )
        return self.target_mean + self.target_scale * self.network(summary)


class HierarchicalRbfQuoteEncoder(RbfQuoteEncoder):
    """Predict Strong blocks front-to-back from causal summary/state prefixes.

    Block ``j`` sees RBF expiry-anchor rows ``0..j`` and the already-predicted
    normalized Strong blocks ``0..j-1``. It cannot use a later-expiry summary
    row. The Gaussian expiry kernels themselves remain smooth, so this is an
    expiry-anchor causality restriction rather than a hard market-quote cutoff.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: HierarchicalRbfQuoteEncoderConfig,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ):
        BaseModel.__init__(self, env)
        if target_mean.shape != (config.state_size,) or target_scale.shape != (
            config.state_size,
        ):
            raise ValueError("Strong target normalizer has the wrong shape")
        if not torch.all(target_scale > 0.0):
            raise ValueError("Every Strong target scale must be positive")
        self.config = config
        self.register_buffer(
            "target_mean",
            target_mean.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "target_scale",
            target_scale.to(device=env.device, dtype=env.fdtype),
        )
        previous_state_size = 0
        networks: list[FeedForward] = []
        for expiry_index, output_size in enumerate(config.state_sizes_by_expiry):
            summary_prefix_size = (
                expiry_index + 1
            ) * config.summary_size_per_expiry
            networks.append(
                FeedForward(
                    env,
                    num_input=summary_prefix_size + previous_state_size,
                    num_output=output_size,
                    config=config.network,
                )
            )
            previous_state_size += output_size
        self.networks = torch.nn.ModuleList(networks)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Encode summaries sequentially without later-expiry information."""
        if summary.shape[-1] != self.config.summary_size:
            raise ValueError(
                f"Expected summary size {self.config.summary_size}, found {tuple(summary.shape)}"
            )
        normalized_blocks: list[torch.Tensor] = []
        for expiry_index, network in enumerate(self.networks):
            summary_prefix_size = (
                expiry_index + 1
            ) * self.config.summary_size_per_expiry
            inputs = [summary[..., :summary_prefix_size]]
            if normalized_blocks:
                inputs.append(torch.cat(normalized_blocks, dim=-1))
            normalized_blocks.append(network(torch.cat(inputs, dim=-1)))
        normalized_state = torch.cat(normalized_blocks, dim=-1)
        return self.target_mean + self.target_scale * normalized_state


class NativeRbfQuoteEncoder(BaseModel):
    """Map the invariant market summary to capped-unbounded native SANOS raw state."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: NativeRbfQuoteEncoderConfig,
        native_mean: torch.Tensor,
        native_scale: torch.Tensor,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        pca_components: torch.Tensor | None = None,
    ):
        super().__init__(env)
        native_size = config.geometry.native_state_size
        if native_mean.shape != (native_size,) or native_scale.shape != (
            native_size,
        ):
            raise ValueError("Native target normalizer has the wrong shape")
        if state_mean.shape != (config.state_size,) or state_scale.shape != (
            config.state_size,
        ):
            raise ValueError("Encoder state normalizer has the wrong shape")
        if not torch.all(native_scale > 0.0) or not torch.all(state_scale > 0.0):
            raise ValueError("Every native and encoder-state scale must be positive")
        full_native_state = config.state_size == native_size
        if full_native_state and pca_components is not None:
            raise ValueError("Full-native encoder does not use PCA components")
        if not full_native_state and (
            pca_components is None
            or pca_components.shape != (config.state_size, native_size)
        ):
            raise ValueError("Compressed native encoder requires PCA components")
        self.config = config
        self.register_buffer(
            "native_mean",
            native_mean.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "native_scale",
            native_scale.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "state_mean",
            state_mean.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "state_scale",
            state_scale.to(device=env.device, dtype=env.fdtype),
        )
        self.state_network = FeedForward(
            env,
            num_input=config.summary_size,
            num_output=config.state_size,
            config=config.network,
        )
        self.native_decoder = (
            None
            if full_native_state
            else FeedForward(
                env,
                num_input=config.state_size,
                num_output=native_size,
                config=config.native_decoder,
            )
        )
        if self.native_decoder is not None:
            assert pca_components is not None
            with torch.no_grad():
                self.native_decoder.I0.weight.copy_(
                    pca_components.to(device=env.device, dtype=env.fdtype).T
                )
                cast(torch.Tensor, self.native_decoder.I0.bias).zero_()

    def encode_state(self, summary: torch.Tensor) -> torch.Tensor:
        """Return the explicit full or compressed surface state."""
        if summary.shape[-1] != self.config.summary_size:
            raise ValueError(
                f"Expected summary size {self.config.summary_size}, found "
                f"{tuple(summary.shape)}"
            )
        return self.state_mean + self.state_scale * self.state_network(summary)

    def native_raw_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Decode one explicit state into the 2,828 native raw coordinates."""
        if state.shape[-1] != self.config.state_size:
            raise ValueError(
                f"Expected state size {self.config.state_size}, found {tuple(state.shape)}"
            )
        standardized_native = (
            state
            if self.native_decoder is None
            else self.native_decoder(state)
        )
        return self.native_mean + self.native_scale * standardized_native

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Encode an invariant market summary directly to native raw state."""
        return self.native_raw_from_state(self.encode_state(summary))


def strong_decoder_config(
    *,
    label: str,
    geometry: ForwardMuFlexGeometryConfig,
) -> TriangularPcaDecoderConfig:
    """Build the exact selected Strong triangular affine configuration."""
    return TriangularPcaDecoderConfig(
        label=label,
        state_sizes_by_expiry=STRONG_STATE_SIZES,
        geometry=geometry,
    )


def _quote_features(
    day: FlexMarketDay,
    *,
    geometry: ForwardMuFlexGeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expiry/NMS coordinates and raw quote-value channels."""
    expiry_y = day.expiries_y[day.expiry_index]
    annual_vol = day.atm_annual_vols[day.expiry_index]
    eps_vol = day.model_atm_total_vols[0] / np.sqrt(geometry.model_expiries_y[0])
    nms = np.asarray(
        NMS.to_nms(
            day.strikes,
            expiry_y,
            vol=annual_vol,
            eps_vol=eps_vol,
        ),
        dtype=np.float64,
    )
    expiry_fraction = expiry_y / geometry.model_expiries_y[-1]
    half_spread = day.half_spreads
    values = np.column_stack(
        (
            expiry_fraction,
            nms,
            np.log(day.strikes),
            day.is_call.astype(np.float64),
            day.bids,
            day.asks,
            day.fit_prices,
            np.log(np.maximum(half_spread, 1.0e-12)),
            day.log_volumes,
            day.log_open_interests,
        )
    )
    if values.shape != (day.num_quotes, len(_QUOTE_FEATURE_NAMES)):
        raise RuntimeError("Internal quote feature shape mismatch")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite quote features on {day.trading_date}")
    return day.quote_business_dtes, nms, values


def _equal_day_quote_normalizer(
    days: tuple[FlexMarketDay, ...],
    *,
    geometry: ForwardMuFlexGeometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature moments with each market date carrying total weight one."""
    per_day_mean: list[np.ndarray] = []
    per_day_second: list[np.ndarray] = []
    for day in days:
        _, _, values = _quote_features(day, geometry=geometry)
        per_day_mean.append(np.mean(values, axis=0))
        per_day_second.append(np.mean(values**2, axis=0))
    mean = np.mean(np.stack(per_day_mean), axis=0)
    second = np.mean(np.stack(per_day_second), axis=0)
    scale = np.sqrt(np.maximum(second - mean**2, 1.0e-12))
    return mean, scale


def rbf_quote_summary(
    day: FlexMarketDay,
    *,
    config: RbfQuoteEncoderConfig | NativeRbfQuoteEncoderConfig,
    quote_mean: np.ndarray,
    quote_scale: np.ndarray,
) -> np.ndarray:
    """Aggregate an unordered quote set over fixed expiry/NMS anchors."""
    dte, nms, values = _quote_features(day, geometry=config.geometry)
    normalized = (values - quote_mean) / quote_scale
    expiry_anchors = np.linspace(
        config.geometry.model_business_dtes[0],
        config.geometry.model_business_dtes[-1],
        config.num_expiry_anchors,
    )
    nms_anchors = np.linspace(
        config.geometry.nms_range[0],
        config.geometry.nms_range[1],
        config.num_nms_anchors,
    )
    expiry_weight = np.exp(
        -0.5
        * (
            (dte[:, None] - expiry_anchors[None, :])
            / config.expiry_bandwidth_business_days
        )
        ** 2
    )
    nms_weight = np.exp(
        -0.5
        * ((nms[:, None] - nms_anchors[None, :]) / config.nms_bandwidth) ** 2
    )
    weights = (
        expiry_weight[:, :, None] * nms_weight[:, None, :]
    ).reshape(day.num_quotes, -1)
    denominator = np.sum(weights, axis=0)
    weighted_mean = (weights.T @ normalized) / np.maximum(denominator[:, None], 1.0e-12)
    occupancy = np.log1p(denominator / day.num_quotes)[:, None]
    summary = np.concatenate((weighted_mean, occupancy), axis=1).reshape(-1)
    if summary.shape != (config.summary_size,) or not np.all(np.isfinite(summary)):
        raise RuntimeError("Invalid fixed-RBF market summary")
    return summary


def fit_rbf_feature_statistics(
    training_days: tuple[FlexMarketDay, ...],
    *,
    config: RbfQuoteEncoderConfig | NativeRbfQuoteEncoderConfig,
) -> tuple[RbfFeatureStatistics, np.ndarray]:
    """Fit all normalizers on training dates only and return train summaries."""
    if len(training_days) == 0:
        raise ValueError("At least one training day is required")
    quote_mean, quote_scale = _equal_day_quote_normalizer(
        training_days,
        geometry=config.geometry,
    )
    raw = np.stack(
        [
            rbf_quote_summary(
                day,
                config=config,
                quote_mean=quote_mean,
                quote_scale=quote_scale,
            )
            for day in training_days
        ]
    )
    summary_mean = np.mean(raw, axis=0)
    summary_scale = np.std(raw, axis=0, ddof=0)
    summary_scale = np.maximum(summary_scale, 1.0e-6)
    statistics = RbfFeatureStatistics(
        quote_mean=quote_mean,
        quote_scale=quote_scale,
        summary_mean=summary_mean,
        summary_scale=summary_scale,
    )
    return statistics, (raw - summary_mean) / summary_scale


def transform_market_days(
    days: tuple[FlexMarketDay, ...],
    *,
    config: RbfQuoteEncoderConfig | NativeRbfQuoteEncoderConfig,
    statistics: RbfFeatureStatistics,
) -> np.ndarray:
    """Apply frozen training-only feature statistics to arbitrary dates."""
    raw = np.stack(
        [
            rbf_quote_summary(
                day,
                config=config,
                quote_mean=statistics.quote_mean,
                quote_scale=statistics.quote_scale,
            )
            for day in days
        ]
    )
    return (raw - statistics.summary_mean) / statistics.summary_scale


def prices_from_states(
    decoder: LatentStateSanosModel,
    states: torch.Tensor,
    batch: TorchMarketBatch,
) -> torch.Tensor:
    """Decode supplied Strong states and restore separate call/put books."""
    raw = decoder.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=decoder.config.geometry)
    surface = decoder.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    call_locations = decoder.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=batch.price_expiries_y,
        strikes=batch.price_strikes,
    )
    calls = torch.gather(call_locations, dim=1, index=batch.quote_price_index)
    return torch.where(batch.is_call, calls, calls - 1.0 + batch.strikes)


def weekly_quote_mask(
    day: FlexMarketDay,
    *,
    geometry: ForwardMuFlexGeometryConfig,
) -> np.ndarray:
    """Select each date's market expiry closest to every weekly model node."""
    if day.model_atm_total_vols.shape != (geometry.num_model_expiries,):
        raise ValueError("Market ATM nodes do not match the Strong geometry")
    target = geometry.model_business_dtes
    selected = np.unique(
        [int(np.argmin(np.abs(day.business_dtes - value))) for value in target]
    )
    return np.isin(day.expiry_index, selected)


def evaluate_encoder(
    encoder: RbfQuoteEncoder,
    summaries: torch.Tensor,
    decoder: LatentStateSanosModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    batch_size: int,
    state_targets: torch.Tensor | None = None,
) -> EncoderEvaluation:
    """Evaluate a frozen market encoder under quote- and date-weighting."""
    if summaries.shape != (len(market_days), encoder.config.summary_size):
        raise ValueError("Encoder summaries do not match the market-day universe")
    if state_targets is not None and state_targets.shape != (
        len(market_days),
        encoder.config.state_size,
    ):
        raise ValueError("Encoder state targets do not match the market-day universe")
    env = encoder.env
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    fits: list[np.ndarray] = []
    per_day_objectives: list[float] = []
    total_loss = 0.0
    total_quotes = 0
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(market_days), batch_size):
            indices = np.arange(start, min(start + batch_size, len(market_days)))
            batch = materialize_batch(torch_days, indices, env=env)
            states = encoder(summaries[batch.day_indices])
            prices = prices_from_states(decoder, states, batch)
            half_spread = 0.5 * (batch.asks - batch.bids)
            safe = torch.where(batch.mask, half_spread, torch.ones_like(half_spread))
            quote_loss = (
                torch.relu(prices - batch.asks)
                + torch.relu(batch.bids - prices)
                + objective_config.fit_weight * torch.abs(prices - batch.fit_prices)
            ) / safe
            valid = batch.mask.to(dtype=prices.dtype)
            day_values = torch.sum(quote_loss * valid, dim=1) / torch.sum(valid, dim=1)
            per_day_objectives.extend(day_values.cpu().tolist())
            total_loss += float(torch.sum(quote_loss * valid).cpu())
            total_quotes += int(torch.sum(batch.mask).cpu())
            for row, day_index in enumerate(indices):
                count = market_days[int(day_index)].num_quotes
                fits.append(prices[row, :count].cpu().numpy().astype(np.float64))

    fit = np.concatenate(fits)
    bid = np.concatenate([day.bids for day in market_days])
    ask = np.concatenate([day.asks for day in market_days])
    market_fit = np.concatenate([day.fit_prices for day in market_days])
    weekly_masks = tuple(
        weekly_quote_mask(day, geometry=encoder.config.geometry)
        for day in market_days
    )
    weekly_fit = np.concatenate(
        [value[mask] for value, mask in zip(fits, weekly_masks, strict=True)]
    )
    weekly_bid = np.concatenate(
        [day.bids[mask] for day, mask in zip(market_days, weekly_masks, strict=True)]
    )
    weekly_ask = np.concatenate(
        [day.asks[mask] for day, mask in zip(market_days, weekly_masks, strict=True)]
    )
    weekly_market_fit = np.concatenate(
        [
            day.fit_prices[mask]
            for day, mask in zip(market_days, weekly_masks, strict=True)
        ]
    )
    all_excess = normalized_excess(fit, bid, ask)
    weekly_excess = normalized_excess(weekly_fit, weekly_bid, weekly_ask)
    if state_targets is None:
        state_mse = None
        state_mae = None
    else:
        with torch.no_grad():
            prediction = encoder(summaries)
            error = prediction - state_targets
            state_mse = float(torch.mean(error.square()).cpu())
            state_mae = float(torch.mean(torch.abs(error)).cpu())
    return EncoderEvaluation(
        num_days=len(market_days),
        num_quotes=total_quotes,
        quote_weighted_objective=total_loss / total_quotes,
        date_weighted_objective=float(np.mean(per_day_objectives)),
        all_statistics=excess_statistics(all_excess).to_dict(),
        weekly_statistics=excess_statistics(weekly_excess).to_dict(),
        surface_fit=surface_fit_statistics(fit, market_fit, bid, ask).to_dict(),
        weekly_surface_fit=surface_fit_statistics(
            weekly_fit,
            weekly_market_fit,
            weekly_bid,
            weekly_ask,
        ).to_dict(),
        state_mse=state_mse,
        state_mae=state_mae,
    )


def checkpoint_digest(path: Path) -> str:
    """SHA-256 digest for immutable encoder provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    """Atomically write a stable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def encoder_checkpoint_identity(model: RbfQuoteEncoder) -> dict[str, object]:
    """Versioned logical identity for strict checkpoint loading."""
    return {
        "model_class": type(model).__qualname__,
        "config_version": model.config.version.unique_id48,
        "user_version": model.config.user_version,
        "model_config": asdict(model.config),
    }


def load_complete_encoder_checkpoint(
    model: RbfQuoteEncoder,
    path: Path,
    *,
    expected_train_fingerprints: tuple[str, ...],
    expected_decoder_sha256: str,
) -> dict[str, object]:
    """Load only a converged encoder bound to the training universe/decoder."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("phase") != "complete":
        raise ValueError(f"Encoder checkpoint is not complete: {path}")
    if payload.get("model_identity") != encoder_checkpoint_identity(model):
        raise ValueError(f"Encoder checkpoint model identity mismatch: {path}")
    if payload.get("train_market_fingerprints") != list(expected_train_fingerprints):
        raise ValueError(f"Encoder checkpoint training universe mismatch: {path}")
    if payload.get("decoder_sha256") != expected_decoder_sha256:
        raise ValueError(f"Encoder checkpoint Strong decoder mismatch: {path}")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or any(
        not isinstance(audits.get(phase), dict)
        or audits[phase].get("converged") is not True
        for phase in ("supervised", "market")
    ):
        raise ValueError(f"Encoder checkpoint has no convergence proof: {path}")
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]))
    return cast(dict[str, object], payload)


def load_encoded_strong_history(
    checkpoint_path: Path,
    report_path: Path,
) -> EncodedStrongHistory:
    """Load the frozen 2022-2025 encoder history with digest validation."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("phase") != "complete":
        raise ValueError("Encoded Strong history requires a complete encoder checkpoint")
    if report.get("status") != "complete":
        raise ValueError("Encoded Strong history report is not complete")
    checkpoint_sha256 = checkpoint_digest(checkpoint_path)
    if report.get("encoder_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Encoder checkpoint digest does not match its report")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or any(
        not isinstance(audits.get(name), dict)
        or audits[name].get("converged") is not True
        for name in ("supervised", "market")
    ):
        raise ValueError("Encoded Strong history lacks convergence proof")
    history = payload.get("encoded_history")
    if not isinstance(history, dict):
        raise ValueError("Complete encoder checkpoint contains no encoded history")
    dates = tuple(date.fromisoformat(str(value)) for value in history["dates"])
    fingerprints = tuple(str(value) for value in history["market_fingerprints"])
    states = cast(torch.Tensor, history["states"]).detach().cpu().numpy().copy()
    spot_refs = cast(torch.Tensor, history["spot_refs"]).detach().cpu().numpy().copy()
    if _array_sha256(states) != history.get("state_sha256"):
        raise ValueError("Encoded Strong state payload digest mismatch")
    if _array_sha256(spot_refs) != history.get("spot_ref_sha256"):
        raise ValueError("Encoded Strong spot payload digest mismatch")
    return EncodedStrongHistory(
        dates=dates,
        market_fingerprints=fingerprints,
        states=states,
        spot_refs=spot_refs,
        encoder_checkpoint_sha256=checkpoint_sha256,
        strong_decoder_sha256=str(payload["decoder_sha256"]),
    )


def _array_sha256(value: np.ndarray) -> str:
    """Hash one contiguous numeric history array."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
