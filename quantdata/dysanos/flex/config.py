"""Configuration for fixed-weekly FLEX capacity benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8, unique_label48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.torch.ffw import FeedForwardConfig


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class FlexGeometryConfig:
    """Exact strike, expiry, and parameter geometry shared by every stage."""

    num_inner_strikes: int = 200
    nms_range: tuple[float, float] = (-3.0, 2.0)
    num_model_expiries: int = 14
    first_model_business_dte: float = 4.8
    model_expiry_step_business_days: float = 5.0
    explicit_model_business_dtes: tuple[float, ...] | None = None

    min_dlvsq: float = 0.01**2
    max_dlvsq: float = 3.0**2
    min_forward_vol: float = 0.01
    max_forward_vol: float = 1.0
    min_eta: float = 0.01
    max_eta: float = 0.50
    lower_boundary_scale: float = 0.50
    upper_boundary_scale: float = 1.50

    def __post_init__(self) -> None:
        verify_inp(
            self.num_inner_strikes >= 5,
            lambda: (
                "'num_inner_strikes' must be at least five, found "
                f"{self.num_inner_strikes}"
            ),
        )
        verify_inp(
            len(self.nms_range) == 2
            and self.nms_range[0] < 0.0
            and self.nms_range[1] > 0.0,
            lambda: f"'nms_range' must straddle zero, found {self.nms_range}",
        )
        verify_inp(
            self.num_model_expiries >= 2,
            lambda: (
                "'num_model_expiries' must be at least two, found "
                f"{self.num_model_expiries}"
            ),
        )
        verify_inp(
            self.first_model_business_dte > 0.0,
            lambda: (
                "'first_model_business_dte' must be positive, found "
                f"{self.first_model_business_dte}"
            ),
        )
        verify_inp(
            self.model_expiry_step_business_days > 0.0,
            lambda: (
                "'model_expiry_step_business_days' must be positive, found "
                f"{self.model_expiry_step_business_days}"
            ),
        )
        if self.explicit_model_business_dtes is not None:
            nodes = np.asarray(
                self.explicit_model_business_dtes,
                dtype=np.float64,
            )
            verify_inp(
                nodes.shape == (self.num_model_expiries,),
                lambda: (
                    "'explicit_model_business_dtes' must contain exactly "
                    f"{self.num_model_expiries} nodes, found {nodes.shape}"
                ),
            )
            verify_inp(
                np.all(np.isfinite(nodes) & (nodes > 0.0))
                and np.all(np.diff(nodes) > 0.0),
                lambda: (
                    "'explicit_model_business_dtes' must be finite, positive, "
                    f"and strictly increasing, found {nodes}"
                ),
            )
        verify_inp(
            0.0 < self.min_dlvsq < self.max_dlvsq,
            lambda: (
                "Expected 0 < min_dlvsq < max_dlvsq, found "
                f"{self.min_dlvsq} and {self.max_dlvsq}"
            ),
        )
        verify_inp(
            0.0 < self.min_forward_vol < self.max_forward_vol,
            lambda: (
                "Expected 0 < min_forward_vol < max_forward_vol, found "
                f"{self.min_forward_vol} and {self.max_forward_vol}"
            ),
        )
        verify_inp(
            0.0 < self.min_eta < self.max_eta,
            lambda: (
                "Expected 0 < min_eta < max_eta, found "
                f"{self.min_eta} and {self.max_eta}"
            ),
        )
        verify_inp(
            0.0 < self.lower_boundary_scale < 1.0,
            lambda: (
                "'lower_boundary_scale' must lie in (0,1), found "
                f"{self.lower_boundary_scale}"
            ),
        )
        verify_inp(
            self.upper_boundary_scale > 1.0,
            lambda: (
                "'upper_boundary_scale' must exceed one, found "
                f"{self.upper_boundary_scale}"
            ),
        )

    @property
    def model_business_dtes(self) -> np.ndarray:
        """Return the configured model nodes in business-day units."""
        if self.explicit_model_business_dtes is not None:
            return np.asarray(
                self.explicit_model_business_dtes,
                dtype=np.float64,
            )
        return self.first_model_business_dte + (
            self.model_expiry_step_business_days
            * np.arange(self.num_model_expiries, dtype=np.float64)
        )

    @property
    def model_expiries_y(self) -> np.ndarray:
        """Return model expiries in the repository's business-year clock."""
        return self.model_business_dtes / DAYS_PER_YEAR

    @property
    def max_business_dte(self) -> float:
        """Exact final model horizon in business days."""
        return float(self.model_business_dtes[-1])

    @property
    def native_state_size(self) -> int:
        """Number of raw per-day DLV, forward-vol, and eta coordinates."""
        return (
            self.num_model_expiries * self.num_inner_strikes
            + self.num_model_expiries
            + 1
        )

    @property
    def nms_epsilon_y(self) -> float:
        """Canonical NMS epsilon; FLEX deliberately does not use NMS5."""
        return float(NMS.EPS)


@version("1.0", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class FixedEtaFlexGeometryConfig(FlexGeometryConfig):
    """FLEX geometry with one fixed backbone-volatility multiplier."""

    fixed_eta: float = 0.40

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(
            self.min_eta <= self.fixed_eta <= self.max_eta,
            lambda: (
                "'fixed_eta' must lie within the configured eta bounds, found "
                f"{self.fixed_eta} outside [{self.min_eta},{self.max_eta}]"
            ),
        )


@version("1.0", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class ForwardMuFlexGeometryConfig(FlexGeometryConfig):
    """Use one learned backbone multiplier in every forward-vol interval."""

    forward_mu_backbone: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(
            self.forward_mu_backbone,
            lambda: "'forward_mu_backbone' must remain true for this geometry",
        )

    @property
    def expiry_block_state_size(self) -> int:
        """DLVs, forward volatility, and forward mu stored for one expiry."""
        return self.num_inner_strikes + 2

    @property
    def native_state_size(self) -> int:
        """Number of raw coordinates in the expiry-major native state."""
        return self.num_model_expiries * self.expiry_block_state_size


@version("1.0", dependencies=[ForwardMuFlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class CappedUnboundedDlvFlexGeometryConfig(ForwardMuFlexGeometryConfig):
    """Use an upward-unbounded log-DLV² state with a pricing-only cap.

    A native DLV coordinate ``z`` represents ``min_dlvsq + exp(z)``.  SANOS
    pricing mechanically caps that physical value at ``max_dlvsq``.  Training
    adds the smooth one-sided log-coordinate barrier configured here, so a
    state can reach and cross the pricing cap without sigmoid saturation while
    still receiving a gradient back toward the economically supported range.
    """

    dlv_cap_barrier_log_width: float = 0.10
    dlv_cap_barrier_log_std: float = 0.35

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(
            np.isfinite(self.dlv_cap_barrier_log_width)
            and self.dlv_cap_barrier_log_width > 0.0,
            lambda: (
                "'dlv_cap_barrier_log_width' must be finite and positive, "
                f"found {self.dlv_cap_barrier_log_width}"
            ),
        )
        verify_inp(
            np.isfinite(self.dlv_cap_barrier_log_std)
            and self.dlv_cap_barrier_log_std > 0.0,
            lambda: (
                "'dlv_cap_barrier_log_std' must be finite and positive, "
                f"found {self.dlv_cap_barrier_log_std}"
            ),
        )

    @property
    def maximum_log_dlvsq(self) -> float:
        """Raw log-DLV² coordinate at which the pricing cap becomes active."""
        return float(np.log(self.max_dlvsq - self.min_dlvsq))


@version(
    "1.0",
    dependencies=[CappedUnboundedDlvFlexGeometryConfig],
)
@dataclass(frozen=True, kw_only=True)
class SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
    CappedUnboundedDlvFlexGeometryConfig
):
    """Use one daily floating strike support across every model expiry.

    Half of the inner strikes are canonical-NMS candidates at the front model
    expiry and half are candidates at the final model expiry. Their implied
    relative strikes are merged with an explicit minimum log-spacing safeguard
    and used at all model expiries for that day. The support still floats
    between days because both endpoint grids depend on that day's learned
    strike-total-volatility curve.
    """

    minimum_log_spacing_fraction: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(
            self.num_inner_strikes % 2 == 0,
            lambda: (
                "Shared-strike geometry requires an even number of inner "
                f"strikes, found {self.num_inner_strikes}"
            ),
        )
        verify_inp(
            0.0 < self.minimum_log_spacing_fraction <= 1.0,
            lambda: (
                "'minimum_log_spacing_fraction' must lie in (0,1], found "
                f"{self.minimum_log_spacing_fraction}"
            ),
        )


@version("1.0", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class FlexMarketConfig:
    """Upstream market controls that do not alter the fixed model geometry."""

    max_num_options: int = 100_000
    smoothness: float = 0.25

    def __post_init__(self) -> None:
        verify_inp(
            self.max_num_options >= 10_000,
            lambda: (
                "'max_num_options' must be deliberately non-binding, found "
                f"{self.max_num_options}"
            ),
        )
        verify_inp(
            self.smoothness > 0.01,
            lambda: f"'smoothness' must exceed 0.01, found {self.smoothness}",
        )


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FlexObjectiveConfig:
    """Common quote objective used by native state and compression."""

    fit_weight: float = 0.05

    def __post_init__(self) -> None:
        verify_inp(
            self.fit_weight >= 0.0,
            lambda: f"'fit_weight' must be non-negative, found {self.fit_weight}",
        )


@version("1.1", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class NativeStateConfig:
    """Logical identity of the uncompressed per-day raw-state model."""

    label: str
    group: str = "flex/native-state"
    user_version: str = "1.0"
    geometry: FlexGeometryConfig = FlexGeometryConfig()

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version("1.2", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class LatentDecoderConfig:
    """Controlled feed-forward decoder for a lower-dimensional daily state."""

    label: str
    state_size: int
    decoder_input_size: int = 1_000
    group: str = "flex/latent-decoder"
    user_version: str = "1.0"
    geometry: FlexGeometryConfig = FlexGeometryConfig()
    decoder_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=1_000,
    )

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)
        verify_inp(
            0 < self.state_size <= self.decoder_input_size,
            lambda: (
                "Expected 0 < state_size <= decoder_input_size, found "
                f"{self.state_size} and {self.decoder_input_size}"
            ),
        )

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version("1.0", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class PcaDecoderConfig:
    """Affine lower-state decoder initialized by exact PCA of native states."""

    label: str
    state_size: int
    group: str = "flex/pca-decoder"
    user_version: str = "1.0"
    geometry: FlexGeometryConfig = FlexGeometryConfig()
    decoder_network: FeedForwardConfig = FeedForwardConfig.linear_config()

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)
        verify_inp(
            0 < self.state_size <= self.geometry.native_state_size,
            lambda: (
                "Expected 0 < state_size <= native_state_size, found "
                f"{self.state_size} and {self.geometry.native_state_size}"
            ),
        )

    @property
    def decoder_input_size(self) -> int:
        """Use the state directly without padding or a hidden bottleneck."""
        return self.state_size

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version(
    "1.0",
    dependencies=[SharedStrikeCappedUnboundedDlvFlexGeometryConfig],
)
@dataclass(frozen=True, kw_only=True)
class JointDirectDynamicsConfig:
    """Joint affine Direct decoder and stationary weekly dynamics geometry.

    The mean-reversion generator is parameterized as ``K=LL' + S + k_min I``,
    where ``S`` is skew-symmetric.  Its symmetric part is therefore uniformly
    positive, so ``exp(-K)`` is a real stationary transition at every optimizer
    step rather than a property checked only after fitting a discrete AR(1).
    """

    label: str
    state_size: int = 200
    group: str = "flex/joint-direct-dynamics"
    user_version: str = "1.0"
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig = (
        SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    )
    decoder_network: FeedForwardConfig = FeedForwardConfig.linear_config()
    minimum_mean_reversion_per_week: float = 1.0e-3
    innovation_std_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)
        verify_inp(
            0 < self.state_size <= self.geometry.native_state_size,
            lambda: (
                "Expected 0 < state_size <= native_state_size, found "
                f"{self.state_size} and {self.geometry.native_state_size}"
            ),
        )
        verify_inp(
            self.decoder_network.num_depth == 0,
            lambda: "The joint Direct decoder must remain affine",
        )
        verify_inp(
            np.isfinite(self.minimum_mean_reversion_per_week)
            and self.minimum_mean_reversion_per_week > 0.0,
            lambda: (
                "'minimum_mean_reversion_per_week' must be finite and positive, "
                f"found {self.minimum_mean_reversion_per_week}"
            ),
        )
        verify_inp(
            np.isfinite(self.innovation_std_floor)
            and self.innovation_std_floor > 0.0,
            lambda: (
                "'innovation_std_floor' must be finite and positive, found "
                f"{self.innovation_std_floor}"
            ),
        )

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version(
    "1.0",
    dependencies=[SharedStrikeCappedUnboundedDlvFlexGeometryConfig],
)
@dataclass(frozen=True, kw_only=True)
class JointHierarchicalDynamicsConfig:
    """Joint expiry-causal decoder and stationary weekly P-law geometry.

    The row-vector transition is ``exp(-K)``.  ``K`` is block upper
    triangular in expiry order, so the next state block for expiry ``j`` can
    use only current states ``1,...,j``.  Each diagonal block has the stable
    Direct form ``LL' + S + k_min I``; strict-upper blocks remain unrestricted
    without changing the stable eigenvalues of the generator.
    """

    label: str
    state_sizes_by_expiry: tuple[int, ...] = (
        23,
        27,
        25,
        22,
        19,
        14,
        14,
        14,
        7,
        7,
        7,
        7,
        7,
        7,
    )
    state_size: int = 200
    group: str = "flex/joint-hierarchical-dynamics"
    user_version: str = "1.0"
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig = (
        SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    )
    decoder_network: FeedForwardConfig = FeedForwardConfig.linear_config()
    minimum_mean_reversion_per_week: float = 1.0e-3
    innovation_std_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)
        verify_inp(
            len(self.state_sizes_by_expiry) == self.geometry.num_model_expiries,
            lambda: (
                "Expected one hierarchical state block per model expiry, found "
                f"{len(self.state_sizes_by_expiry)} versus "
                f"{self.geometry.num_model_expiries}"
            ),
        )
        verify_inp(
            all(size > 0 for size in self.state_sizes_by_expiry),
            lambda: (
                "Every hierarchical state block must be positive, found "
                f"{self.state_sizes_by_expiry}"
            ),
        )
        verify_inp(
            sum(self.state_sizes_by_expiry) == self.state_size,
            lambda: (
                "Hierarchical state blocks must sum to state_size, found "
                f"{sum(self.state_sizes_by_expiry)} versus {self.state_size}"
            ),
        )
        verify_inp(
            self.state_size <= self.geometry.native_state_size,
            lambda: (
                "Hierarchical state_size cannot exceed native_state_size, found "
                f"{self.state_size} versus {self.geometry.native_state_size}"
            ),
        )
        verify_inp(
            self.decoder_network.num_depth == 0,
            lambda: "The joint hierarchical decoder must remain affine",
        )
        verify_inp(
            np.isfinite(self.minimum_mean_reversion_per_week)
            and self.minimum_mean_reversion_per_week > 0.0,
            lambda: (
                "'minimum_mean_reversion_per_week' must be finite and positive, "
                f"found {self.minimum_mean_reversion_per_week}"
            ),
        )
        verify_inp(
            np.isfinite(self.innovation_std_floor)
            and self.innovation_std_floor > 0.0,
            lambda: (
                "'innovation_std_floor' must be finite and positive, found "
                f"{self.innovation_std_floor}"
            ),
        )

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version("1.0", dependencies=[ForwardMuFlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class TriangularPcaDecoderConfig:
    """Affine expiry-causal decoder with a configurable state block per expiry."""

    label: str
    state_sizes_by_expiry: tuple[int, ...]
    state_size: int = 140
    group: str = "flex/triangular-pca-decoder"
    user_version: str = "1.0"
    geometry: ForwardMuFlexGeometryConfig = ForwardMuFlexGeometryConfig()
    decoder_network: FeedForwardConfig = FeedForwardConfig.linear_config()

    def __post_init__(self) -> None:
        _validate_model_identity(self.label, self.group)
        verify_inp(
            len(self.state_sizes_by_expiry) == self.geometry.num_model_expiries,
            lambda: (
                "Expected one triangular state size per model expiry, found "
                f"{len(self.state_sizes_by_expiry)} versus "
                f"{self.geometry.num_model_expiries}"
            ),
        )
        verify_inp(
            all(size > 0 for size in self.state_sizes_by_expiry),
            lambda: (
                "Every triangular state block must be positive, found "
                f"{self.state_sizes_by_expiry}"
            ),
        )
        verify_inp(
            sum(self.state_sizes_by_expiry) == self.state_size,
            lambda: (
                "Triangular state blocks must sum to state_size, found "
                f"{sum(self.state_sizes_by_expiry)} versus {self.state_size}"
            ),
        )
        verify_inp(
            self.state_size <= self.geometry.native_state_size,
            lambda: (
                "Triangular state_size cannot exceed native_state_size, found "
                f"{self.state_size} versus {self.geometry.native_state_size}"
            ),
        )
        verify_inp(
            self.decoder_network.num_depth == 0,
            lambda: "Triangular PCA decoding requires one affine layer",
        )

    @property
    def decoder_input_size(self) -> int:
        """Use the full concatenated expiry state without input padding."""
        return self.state_size

    @property
    def filename(self) -> str:
        """Stable checkpoint filename body."""
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@version("1.4")
@dataclass(frozen=True, kw_only=True)
class FlexTrainConfig:
    """Operational controls shared by the Torch calibration stages."""

    num_epochs: int = 300
    day_batch_size: int = 4
    learning_rate: float = 2.0e-2
    learning_rate_schedule: Literal["constant", "cosine"] = "cosine"
    learning_rate_warmup_epochs: int = 0
    minimum_learning_rate_fraction: float = 0.0
    weight_decay: float = 0.0
    gradient_clip: float = 10.0
    seed: int = 1729
    precision: int = 64
    device: str = "cpu"
    checkpoint_every: int = 5
    report_every: int = 25
    convergence_window: int = 25
    convergence_relative_tolerance: float = 1.0e-3
    compile_sanos_solver: bool = False
    just_load: bool = False
    warm_start: bool = False

    def __post_init__(self) -> None:
        verify_inp(
            self.num_epochs > 0,
            lambda: f"'num_epochs' must be positive, found {self.num_epochs}",
        )
        verify_inp(
            self.day_batch_size > 0,
            lambda: (
                "'day_batch_size' must be positive, found "
                f"{self.day_batch_size}"
            ),
        )
        verify_inp(
            self.learning_rate > 0.0,
            lambda: (
                "'learning_rate' must be positive, found "
                f"{self.learning_rate}"
            ),
        )
        verify_inp(
            self.learning_rate_schedule in {"constant", "cosine"},
            lambda: (
                "'learning_rate_schedule' must be 'constant' or 'cosine', found "
                f"{self.learning_rate_schedule!r}"
            ),
        )
        verify_inp(
            0 <= self.learning_rate_warmup_epochs < self.num_epochs,
            lambda: (
                "'learning_rate_warmup_epochs' must lie in [0,num_epochs), found "
                f"{self.learning_rate_warmup_epochs} for {self.num_epochs} epochs"
            ),
        )
        verify_inp(
            0.0 <= self.minimum_learning_rate_fraction <= 1.0,
            lambda: (
                "'minimum_learning_rate_fraction' must lie in [0,1], found "
                f"{self.minimum_learning_rate_fraction}"
            ),
        )
        verify_inp(
            self.precision in (32, 64),
            lambda: f"'precision' must be 32 or 64, found {self.precision}",
        )
        verify_inp(
            self.checkpoint_every > 0,
            lambda: (
                "'checkpoint_every' must be positive, found "
                f"{self.checkpoint_every}"
            ),
        )
        verify_inp(
            self.report_every > 0,
            lambda: (
                "'report_every' must be positive, found "
                f"{self.report_every}"
            ),
        )
        verify_inp(
            self.convergence_window > 0,
            lambda: (
                "'convergence_window' must be positive, found "
                f"{self.convergence_window}"
            ),
        )
        verify_inp(
            0.0 < self.convergence_relative_tolerance < 1.0,
            lambda: (
                "'convergence_relative_tolerance' must lie in (0,1), found "
                f"{self.convergence_relative_tolerance}"
            ),
        )
        verify_inp(
            isinstance(self.compile_sanos_solver, bool),
            lambda: (
                "'compile_sanos_solver' must be bool, found "
                f"{self.compile_sanos_solver!r}"
            ),
        )
        verify_inp(
            isinstance(self.just_load, bool),
            lambda: f"'just_load' must be bool, found {self.just_load!r}",
        )
        verify_inp(
            isinstance(self.warm_start, bool),
            lambda: f"'warm_start' must be bool, found {self.warm_start!r}",
        )
        verify_inp(
            not (self.just_load and self.warm_start),
            lambda: "'just_load' and 'warm_start' are mutually exclusive",
        )


def _validate_model_identity(label: str, group: str) -> None:
    """Validate reusable checkpoint labels and relative group directories."""
    group_path = Path(group)
    verify_inp(
        isinstance(label, str)
        and len(label) > 0
        and (label.startswith("*") or is_filename(label)),
        lambda: (
            "'label' must be a valid filename or start with '*', found "
            f"{label!r}"
        ),
    )
    verify_inp(
        all(
            part not in {"", ".", ".."} and is_filename(part)
            for part in group_path.parts
        ),
        lambda: f"'group' must be a valid relative directory, found {group!r}",
    )
