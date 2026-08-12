"""Versioned configurations for the Tangent Lévy implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sanos.arbfree import MkArbFree
from quantdata.widb.day import ExpiryFilter, RawDayFilter
from quantdata.widb.options_filter import OptionsFilter


def _default_tau_grid() -> tuple[float, ...]:
    # The market route still fits expiries through one business year.  The
    # fixed dynamics panel stops at nine months because the one-year contract
    # is not present on every SPX session after filtering.
    business_days = (0, 5, 10, 20, 40, 60, 90, 126, 189)
    return tuple(day / DAYS_PER_YEAR for day in business_days)


def _default_jump_grid() -> tuple[float, ...]:
    # The paper's discrete model uses [-0.75, 0.75]. A 101-point DETL grid keeps
    # generic-pricer discretization below the bid/ask scale on the SPX smoke
    # case without the path-memory cost of the discrete model's 301 points.
    return tuple(np.linspace(-0.75, 0.75, 101).tolist())


def _default_scoring_x_grid() -> tuple[float, ...]:
    return tuple(np.linspace(-0.35, 0.20, 57).tolist())


@version("0.1", dependencies=[RawDayFilter, ExpiryFilter, OptionsFilter, MkArbFree])
@dataclass(frozen=True, kw_only=True)
class MarketConfig:
    """Logical inputs to the canonical smoothed SPX market route."""

    smoothness: float = 0.25
    min_business_days_to_expiry: int = 5
    max_business_days_to_expiry: int = DAYS_PER_YEAR
    options_filter: OptionsFilter = field(
        default_factory=lambda: OptionsFilter(
            normalize=True,
            max_num_options=1_000,
            max_business_days_to_expiry=DAYS_PER_YEAR,
        )
    )
    raw_day_filter: RawDayFilter = field(default_factory=RawDayFilter)
    expiry_filter: ExpiryFilter = field(default_factory=ExpiryFilter)
    mk_arb_free: MkArbFree = field(default_factory=MkArbFree)

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.smoothness) and self.smoothness > 0.01,
            lambda: (
                "'smoothness' must be finite and strictly greater than 0.01, "
                f"got {self.smoothness!r}"
            ),
        )
        verify_inp(
            isinstance(self.min_business_days_to_expiry, int)
            and self.min_business_days_to_expiry > 0,
            lambda: (
                "'min_business_days_to_expiry' must be a positive integer, got "
                f"{self.min_business_days_to_expiry!r}"
            ),
        )
        verify_inp(
            isinstance(self.max_business_days_to_expiry, int)
            and self.max_business_days_to_expiry
            >= self.min_business_days_to_expiry,
            lambda: (
                "'max_business_days_to_expiry' must be at least the minimum, got "
                f"{self.max_business_days_to_expiry!r}"
            ),
        )
        verify_inp(
            self.options_filter.normalize,
            "'options_filter.normalize' must be True",
        )
        verify_inp(
            self.options_filter.max_business_days_to_expiry
            == self.max_business_days_to_expiry,
            lambda: (
                "the market and option-filter maximum expiries must agree, got "
                f"{self.max_business_days_to_expiry} and "
                f"{self.options_filter.max_business_days_to_expiry}"
            ),
        )

    @property
    def min_tau(self) -> float:
        """Minimum fitted time to maturity in business-year units."""

        return self.min_business_days_to_expiry / DAYS_PER_YEAR

    @property
    def max_tau(self) -> float:
        """Maximum fitted time to maturity in business-year units."""

        return self.max_business_days_to_expiry / DAYS_PER_YEAR

    def __str__(self) -> str:
        return (
            f"sm={self.smoothness:.3g},bdte="
            f"{self.min_business_days_to_expiry}-{self.max_business_days_to_expiry},"
            f"{self.options_filter}"
        )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class PricingConfig:
    """Numerical Fourier integration settings for normalized call prices."""

    integration_upper: float = 160.0
    integration_points: int = 1_281
    strike_chunk_size: int = 256

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.integration_upper) and self.integration_upper > 0.0,
            lambda: (
                "'integration_upper' must be finite and positive, got "
                f"{self.integration_upper!r}"
            ),
        )
        verify_inp(
            isinstance(self.integration_points, int)
            and self.integration_points >= 129
            and self.integration_points % 2 == 1,
            lambda: (
                "'integration_points' must be an odd integer of at least 129, got "
                f"{self.integration_points!r}"
            ),
        )
        verify_inp(
            isinstance(self.strike_chunk_size, int) and self.strike_chunk_size > 0,
            lambda: (
                "'strike_chunk_size' must be a positive integer, got "
                f"{self.strike_chunk_size!r}"
            ),
        )

    def __str__(self) -> str:
        return f"fourier-u={self.integration_upper:g},n={self.integration_points}"


@version("0.1", dependencies=[PricingConfig])
@dataclass(frozen=True, kw_only=True)
class StaticFitConfig:
    """Controls for daily double-exponential static calibration."""

    pricing: PricingConfig = field(default_factory=PricingConfig)
    num_starts: int = 6
    max_iterations: int = 300
    half_spread_floor: float = 1.0e-4
    intensity_bounds: tuple[float, float] = (1.0e-4, 500.0)
    positive_tail_bounds: tuple[float, float] = (1.001, 200.0)
    probability_floor: float = 1.0e-5
    forward_density_floor: float = 1.0e-6
    reconstruction_relative_tolerance: float = 0.25
    reconstruction_repricing_tolerance: float = 2.0
    optimizer_seed: int = 15_040_033

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.num_starts, int) and self.num_starts > 0,
            lambda: f"'num_starts' must be positive, got {self.num_starts!r}",
        )
        verify_inp(
            isinstance(self.max_iterations, int) and self.max_iterations > 0,
            lambda: (
                "'max_iterations' must be positive, got "
                f"{self.max_iterations!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.half_spread_floor) and self.half_spread_floor > 0.0,
            lambda: (
                "'half_spread_floor' must be finite and positive, got "
                f"{self.half_spread_floor!r}"
            ),
        )
        for name, bounds in (
            ("intensity_bounds", self.intensity_bounds),
            ("positive_tail_bounds", self.positive_tail_bounds),
        ):
            verify_inp(
                len(bounds) == 2
                and all(math.isfinite(value) for value in bounds)
                and 0.0 < bounds[0] < bounds[1],
                lambda name=name, bounds=bounds: f"invalid {name}: {bounds!r}",
            )
        verify_inp(
            self.positive_tail_bounds[0] > 1.0,
            lambda: (
                "the lower positive-tail bound must exceed one, got "
                f"{self.positive_tail_bounds[0]}"
            ),
        )
        verify_inp(
            0.0 < self.probability_floor < 0.5,
            lambda: (
                "'probability_floor' must lie in (0, 0.5), got "
                f"{self.probability_floor!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.forward_density_floor)
            and self.forward_density_floor >= 0.0,
            lambda: (
                "'forward_density_floor' must be finite and nonnegative, got "
                f"{self.forward_density_floor!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.reconstruction_relative_tolerance)
            and self.reconstruction_relative_tolerance >= 0.0,
            lambda: (
                "'reconstruction_relative_tolerance' must be finite and "
                f"nonnegative, got {self.reconstruction_relative_tolerance!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.reconstruction_repricing_tolerance)
            and self.reconstruction_repricing_tolerance >= 0.0,
            lambda: (
                "'reconstruction_repricing_tolerance' must be finite and "
                f"nonnegative, got {self.reconstruction_repricing_tolerance!r}"
            ),
        )
        verify_inp(
            isinstance(self.optimizer_seed, int),
            lambda: f"'optimizer_seed' must be an integer, got {self.optimizer_seed!r}",
        )

    def __str__(self) -> str:
        return (
            f"detl-starts={self.num_starts},iter={self.max_iterations},"
            f"h={self.half_spread_floor:g},{self.pricing}"
        )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class DynamicsConfig:
    """Fixed-grid covariance and HJM drift settings."""

    tau_grid: tuple[float, ...] = field(default_factory=_default_tau_grid)
    jump_x_grid: tuple[float, ...] = field(default_factory=_default_jump_grid)
    num_factors: int = 5
    minimum_observations: int = 60
    covariance_shrinkage: float = 0.05
    inactive_density_threshold: float = 1.0e-4
    boundary_standard_deviations: float = 5.0
    require_reconstruction_pass: bool = True
    dt: float = 1.0 / DAYS_PER_YEAR

    def __post_init__(self) -> None:
        tau = np.asarray(self.tau_grid, dtype=np.float64)
        jump_x = np.asarray(self.jump_x_grid, dtype=np.float64)
        verify_inp(
            tau.ndim == 1
            and tau.size >= 3
            and tau[0] == 0.0
            and np.all(np.diff(tau) > 0.0),
            lambda: (
                "'tau_grid' must start at zero and be strictly increasing, got "
                f"{self.tau_grid!r}"
            ),
        )
        verify_inp(
            jump_x.ndim == 1
            and jump_x.size >= 5
            and jump_x.size % 2 == 1
            and np.all(np.diff(jump_x) > 0.0)
            and np.allclose(jump_x, -jump_x[::-1]),
            lambda: (
                "'jump_x_grid' must be an odd, strictly increasing symmetric grid, got "
                f"{self.jump_x_grid!r}"
            ),
        )
        verify_inp(
            np.allclose(np.diff(jump_x), np.diff(jump_x)[0]),
            "'jump_x_grid' must be uniformly spaced for the HJM convolution",
        )
        verify_inp(
            isinstance(self.num_factors, int) and self.num_factors > 0,
            lambda: f"'num_factors' must be positive, got {self.num_factors!r}",
        )
        verify_inp(
            isinstance(self.minimum_observations, int)
            and self.minimum_observations >= 2,
            lambda: (
                "'minimum_observations' must be an integer of at least two, got "
                f"{self.minimum_observations!r}"
            ),
        )
        verify_inp(
            0.0 <= self.covariance_shrinkage < 1.0,
            lambda: (
                "'covariance_shrinkage' must lie in [0, 1), got "
                f"{self.covariance_shrinkage!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.inactive_density_threshold)
            and self.inactive_density_threshold >= 0.0,
            lambda: (
                "'inactive_density_threshold' must be finite and nonnegative, got "
                f"{self.inactive_density_threshold!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.boundary_standard_deviations)
            and self.boundary_standard_deviations > 0.0,
            lambda: (
                "'boundary_standard_deviations' must be finite and positive, got "
                f"{self.boundary_standard_deviations!r}"
            ),
        )
        verify_inp(
            isinstance(self.require_reconstruction_pass, bool),
            lambda: (
                "'require_reconstruction_pass' must be boolean, got "
                f"{self.require_reconstruction_pass!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.dt) and self.dt > 0.0,
            lambda: f"'dt' must be finite and positive, got {self.dt!r}",
        )

    def __str__(self) -> str:
        return (
            f"dyn-t={len(self.tau_grid)},x={len(self.jump_x_grid)},"
            f"f={self.num_factors},shr={self.covariance_shrinkage:g}"
        )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    """Risk-neutral path-generation settings."""

    num_paths: int = 5_000
    num_steps: int = 20
    seed: int = 15_040_033
    boundary_epsilon: float = 1.0e-6
    record_steps: tuple[int, ...] = (1, 5, 8, 20)

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.num_paths, int) and self.num_paths > 0,
            lambda: f"'num_paths' must be positive, got {self.num_paths!r}",
        )
        verify_inp(
            isinstance(self.num_steps, int) and self.num_steps > 0,
            lambda: f"'num_steps' must be positive, got {self.num_steps!r}",
        )
        verify_inp(
            isinstance(self.seed, int),
            lambda: f"'seed' must be an integer, got {self.seed!r}",
        )
        verify_inp(
            math.isfinite(self.boundary_epsilon) and self.boundary_epsilon > 0.0,
            lambda: (
                "'boundary_epsilon' must be finite and positive, got "
                f"{self.boundary_epsilon!r}"
            ),
        )
        verify_inp(
            len(self.record_steps) > 0
            and tuple(sorted(set(self.record_steps))) == self.record_steps
            and self.record_steps[0] > 0
            and self.record_steps[-1] <= self.num_steps,
            lambda: f"invalid 'record_steps': {self.record_steps!r}",
        )

    def __str__(self) -> str:
        return f"sim-p={self.num_paths},n={self.num_steps},seed={self.seed}"


@version(
    "0.1",
    dependencies=[MarketConfig, StaticFitConfig, DynamicsConfig, SimulationConfig],
)
@dataclass(frozen=True, kw_only=True)
class TangentLevyConfig:
    """Complete logical configuration for one Tangent Lévy experiment."""

    label: str
    group: str = "tangent-levy"
    user_version: str = "0.1"
    market: MarketConfig = field(default_factory=MarketConfig)
    static_fit: StaticFitConfig = field(default_factory=StaticFitConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scoring_x_grid: tuple[float, ...] = field(default_factory=_default_scoring_x_grid)

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and len(self.label.strip()) > 0,
            lambda: f"'label' must be a nonempty string, got {self.label!r}",
        )
        verify_inp(
            isinstance(self.group, str),
            lambda: f"'group' must be a string, got {self.group!r}",
        )
        scoring_x = np.asarray(self.scoring_x_grid, dtype=np.float64)
        verify_inp(
            scoring_x.ndim == 1
            and scoring_x.size >= 3
            and np.all(np.isfinite(scoring_x))
            and np.all(np.diff(scoring_x) > 0.0),
            lambda: (
                "'scoring_x_grid' must be finite and strictly increasing, got "
                f"{self.scoring_x_grid!r}"
            ),
        )
        verify_inp(
            self.dynamics.tau_grid[-1] <= self.market.max_tau,
            lambda: (
                "the dynamics maturity grid exceeds the market horizon: "
                f"{self.dynamics.tau_grid[-1]} > {self.market.max_tau}"
            ),
        )
        verify_inp(
            self.dynamics.inactive_density_threshold
            >= self.static_fit.forward_density_floor,
            lambda: (
                "the dynamics inactive-density threshold must be at least the "
                "reconstruction floor"
            ),
        )

    @property
    def filename(self) -> str:
        """Stable cache filename for this logical experiment."""

        return named_unique_filename48_8(self.label, self)

    def __str__(self) -> str:
        return f"{self.label},{self.market},{self.static_fit},{self.dynamics}"
