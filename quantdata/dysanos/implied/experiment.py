"""Versioned configurations for the PCA-IV-AR(1) benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class PCAFitConfig:
    """Logical inputs to the direct log-IV PCA representation."""

    grid_strikes: int = 100
    num_surface_state: int = 20
    nms_eps: float = NMS.EPS
    interpolation: str = "linear-total-variance-v1"

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.grid_strikes, int) and self.grid_strikes >= 2,
            lambda: (
                "'grid_strikes' must be an integer of at least two, got "
                f"{self.grid_strikes!r}"
            ),
        )
        verify_inp(
            isinstance(self.num_surface_state, int)
            and 0 < self.num_surface_state <= self.grid_strikes * 7,
            lambda: (
                "'num_surface_state' must be a positive feasible integer, got "
                f"{self.num_surface_state!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.nms_eps) and self.nms_eps > 0.0,
            lambda: f"'nms_eps' must be finite and positive, got {self.nms_eps}",
        )
        verify_inp(
            self.interpolation == "linear-total-variance-v1",
            lambda: f"unsupported interpolation convention {self.interpolation!r}",
        )

    def __str__(self) -> str:
        return (
            f"logiv-pca-h={self.num_surface_state},n={self.grid_strikes},"
            f"nms-eps={self.nms_eps:.6g},{self.interpolation}"
        )

    @property
    def filename(self) -> str:
        """Stable cache identity for a fitted PCA representation."""

        return named_unique_filename48_8("pca-iv-level-fit", self)


@version(
    "0.6",
    dependencies=[PCAFitConfig, GaussianSpotImportanceConfig],
)
@dataclass(frozen=True, kw_only=True)
class ImpliedPCAExperiment:
    """Logical inputs for one PCA-IV-AR(1) simulation and diagnostic run.

    The published comparison uses 100,000 or 1,000,000 paths, 20 level
    components, and either five or all surface innovation factors. Smaller
    positive path counts remain valid for deterministic smoke tests.
    """

    num_samples: int = 100_000
    grid_strikes: int = 100
    num_traded_nms: int = 20
    num_surface_state: int = 20
    num_alpha: int | None = 5
    use_matrix_exp: bool = True
    bandwidth: float = 1.0
    trading_frequency: str = "daily"
    return_horizon: str = "expiry"
    num_trading_times: int = 10
    # Deprecated aliases for earlier stored experiments.
    mode: str | None = None
    option_return_horizon: str | None = None
    num_steps: int | None = None
    dt: float = 1.0 / DAYS_PER_YEAR
    seed: int = 700_000
    block_size: int = 10_000
    importance_sampling: GaussianSpotImportanceConfig | None = (
        GaussianSpotImportanceConfig()
    )

    # Matched return diagnostics.
    neighbors: int = 1_024
    candidate_neighbors: int = 8_192
    query_paths_per_split: int = 1_024
    instrument_block_size: int = 20
    tail_ranks: tuple[int, ...] = (1, 2, 10, 20, 50)
    signal_thresholds: tuple[float, ...] = (2.0, 3.0)
    cost_proportion: float = 1.0e-4
    roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128)
    static_mesh_points: int = 201

    def __post_init__(self) -> None:
        if self.mode is not None:
            verify_inp(
                self.mode in {"one-step", "one-step-weekly", "path", "weekly"},
                lambda: f"unsupported PCA experiment mode {self.mode!r}",
            )
            object.__setattr__(
                self,
                "trading_frequency",
                "weekly" if self.mode in {"one-step-weekly", "weekly"} else "daily",
            )
        if self.option_return_horizon is not None:
            verify_inp(
                self.option_return_horizon in {"expiry", "one-day"},
                lambda: (
                    "'option_return_horizon' must be 'expiry' or 'one-day', got "
                    f"{self.option_return_horizon!r}"
                ),
            )
            object.__setattr__(
                self,
                "return_horizon",
                "incremental"
                if self.option_return_horizon == "one-day"
                else "expiry",
            )
        if self.mode is not None:
            if self.mode in {"one-step", "one-step-weekly"}:
                object.__setattr__(self, "num_trading_times", 1)
            elif self.num_steps is not None:
                legacy_times = (
                    self.num_steps
                    if self.return_horizon == "incremental"
                    else self.num_steps - 1
                )
                object.__setattr__(self, "num_trading_times", legacy_times)
        if self.mode is None:
            object.__setattr__(
                self,
                "mode",
                (
                    "one-step-weekly"
                    if self.num_trading_times == 1
                    and self.trading_frequency == "weekly"
                    else "one-step"
                    if self.num_trading_times == 1
                    else "weekly"
                    if self.trading_frequency == "weekly"
                    else "path"
                ),
            )
        if self.option_return_horizon is None:
            object.__setattr__(
                self,
                "option_return_horizon",
                "one-day" if self.return_horizon == "incremental" else "expiry",
            )
        positive_integers = (
            "num_samples",
            "grid_strikes",
            "num_traded_nms",
            "num_surface_state",
            "num_trading_times",
            "block_size",
            "neighbors",
            "candidate_neighbors",
            "query_paths_per_split",
            "instrument_block_size",
            "static_mesh_points",
        )
        for name in positive_integers:
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            self.num_traded_nms <= self.grid_strikes,
            lambda: (
                "'num_traded_nms' cannot exceed 'grid_strikes', got "
                f"{self.num_traded_nms}/{self.grid_strikes}"
            ),
        )
        verify_inp(
            self.num_samples >= 2,
            lambda: (
                "'num_samples' must provide nonempty reference and audit "
                f"sets, got {self.num_samples}"
            ),
        )
        verify_inp(
            self.num_alpha in (5, None),
            lambda: (
                "the primary benchmark requires 'num_alpha' to be 5 or None, "
                f"got {self.num_alpha!r}"
            ),
        )
        verify_inp(
            self.use_matrix_exp is True,
            "the PCA benchmark requires exact matrix-exponential transitions",
        )
        verify_inp(
            self.bandwidth == 1.0,
            lambda: (
                "the PCA benchmark uses Gaussian innovations only, so "
                f"'bandwidth' must equal 1, got {self.bandwidth}"
            ),
        )
        verify_inp(
            self.trading_frequency in {"daily", "weekly"},
            lambda: f"unsupported trading frequency {self.trading_frequency!r}",
        )
        verify_inp(
            self.return_horizon in {"expiry", "incremental"},
            lambda: (
                "'return_horizon' must be 'expiry' or 'incremental', got "
                f"{self.return_horizon!r}"
            ),
        )
        verify_inp(
            self.return_horizon == "expiry" or self.num_trading_times >= 2,
            "incremental returns require at least two trading timestamps",
        )
        verify_inp(
            math.isfinite(self.dt) and self.dt > 0.0,
            lambda: f"'dt' must be finite and positive, got {self.dt}",
        )
        verify_inp(
            isinstance(self.seed, int),
            lambda: f"'seed' must be an integer, got {self.seed!r}",
        )
        verify_inp(
            self.num_alpha is None or self.num_alpha <= self.num_surface_state,
            lambda: (
                "'num_alpha' cannot exceed 'num_surface_state', got "
                f"{self.num_alpha}/{self.num_surface_state}"
            ),
        )
        verify_inp(
            tuple(sorted(set(self.tail_ranks))) == self.tail_ranks
            and self.tail_ranks[0] > 0
            and self.tail_ranks[-1] <= self.neighbors,
            lambda: f"invalid tail ranks {self.tail_ranks}",
        )
        verify_inp(
            all(math.isfinite(value) and value > 0.0 for value in self.signal_thresholds),
            lambda: f"invalid signal thresholds {self.signal_thresholds}",
        )
        verify_inp(
            math.isfinite(self.cost_proportion) and self.cost_proportion >= 0.0,
            lambda: f"invalid proportional cost {self.cost_proportion}",
        )
        verify_inp(
            tuple(sorted(set(self.roundoff_multipliers)))
            == self.roundoff_multipliers
            and self.roundoff_multipliers[0] == 0
            and 32 in self.roundoff_multipliers,
            lambda: f"invalid roundoff multipliers {self.roundoff_multipliers}",
        )
        verify_inp(
            self.static_mesh_points >= 5 and self.static_mesh_points % 2 == 1,
            lambda: (
                "'static_mesh_points' must be an odd integer of at least five, "
                f"got {self.static_mesh_points}"
            ),
        )

    @property
    def fit_config(self) -> PCAFitConfig:
        """The result-affecting subset used by the fitted-PCA cache."""

        return PCAFitConfig(
            grid_strikes=self.grid_strikes,
            num_surface_state=self.num_surface_state,
        )

    @property
    def trading_times(self) -> int:
        return (
            self.num_trading_times
            if self.return_horizon == "expiry"
            else self.num_trading_times - 1
        )

    @property
    def trading_step_stride(self) -> int:
        return 5 if self.trading_frequency == "weekly" else 1

    @property
    def is_primary_path_count(self) -> bool:
        return self.num_samples in {100_000, 1_000_000}

    @property
    def reference_paths(self) -> int:
        """Paths used to fit diagnostic estimators (80% of the universe)."""

        return min(self.num_samples - 1, 4 * self.num_samples // 5)

    @property
    def audit_paths(self) -> int:
        """Held-out paths used only to audit fitted diagnostics."""

        return self.num_samples - self.reference_paths

    @property
    def filename(self) -> str:
        """Stable identity including every logical simulation input."""

        return named_unique_filename48_8(
            f"pca-iv-ar1-{self.num_samples}-{self.trading_frequency}-"
            f"{self.return_horizon}",
            self,
        )

    def __str__(self) -> str:
        alpha = "all" if self.num_alpha is None else str(self.num_alpha)
        importance = "direct-p" if self.importance_sampling is None else "spot-is"
        return (
            f"S={self.num_samples},frequency={self.trading_frequency},"
            f"return={self.return_horizon},"
            f"h={self.num_surface_state},alpha={alpha},n={self.grid_strikes},"
            f"nt={self.num_traded_nms},dt={self.dt:.6g},seed={self.seed},{importance}"
        )
