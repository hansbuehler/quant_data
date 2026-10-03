"""Density/slack linear-program benchmark for a fixed daily FLEX geometry."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cvxpy as cp
import highspy
import numpy as np
import torch
from cdxcore.verbose import Context
from cdxcore.version import version
from scipy import sparse
from scipy.special import ndtr

from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.metrics import (
    ExpiryExcessStatistics,
    SurfaceFitStatistics,
    expiry_excess_statistics,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.sanos import (
    FlexSanosDecoder,
    make_supports_from_total_vols,
    sanos_time_interpolation_numpy,
)
from quantdata.sanos.sanos_tools import SanosLPSolver
from quantdata.torch import BaseEnvironment


@version(
    "1.6",
    dependencies=[FlexGeometryConfig, FlexObjectiveConfig],
)
@dataclass(frozen=True, kw_only=True)
class FlexLpConfig:
    """Numerical controls for the fixed-geometry density LP."""

    backbone_mu: float | None = 0.5
    fixed_backbone_annual_vol: float | None = None
    solver_algorithm: Literal[
        "simplex",
        "ipm",
        "clarabel",
        "direct-simplex",
        "direct-ipm",
    ] = "simplex"
    solver_simplex_strategy: int = 1
    central_fit_target: Literal["smooth_reference", "book_mid"] = (
        "smooth_reference"
    )
    enforce_dlv_bounds: bool = True
    timeout_seconds: float = 600.0
    density_mass_floor: float = 1.0e-7

    def __post_init__(self) -> None:
        if (self.backbone_mu is None) == (
            self.fixed_backbone_annual_vol is None
        ):
            raise ValueError(
                "Exactly one of 'backbone_mu' and "
                "'fixed_backbone_annual_vol' must be supplied"
            )
        if self.backbone_mu is not None and not 0.0 <= self.backbone_mu <= 1.0:
            raise ValueError(
                "'backbone_mu' must lie in [0,1], found "
                f"{self.backbone_mu}"
            )
        if self.fixed_backbone_annual_vol is not None and not (
            np.isfinite(self.fixed_backbone_annual_vol)
            and self.fixed_backbone_annual_vol >= 0.0
        ):
            raise ValueError(
                "'fixed_backbone_annual_vol' must be finite and non-negative, "
                f"found {self.fixed_backbone_annual_vol}"
            )
        if self.timeout_seconds <= 0.0:
            raise ValueError(
                "'timeout_seconds' must be positive, found "
                f"{self.timeout_seconds}"
            )
        if self.solver_algorithm not in {
            "simplex",
            "ipm",
            "clarabel",
            "direct-simplex",
            "direct-ipm",
        }:
            raise ValueError(
                "Unsupported 'solver_algorithm', found "
                f"{self.solver_algorithm!r}"
            )
        if not 0 <= self.solver_simplex_strategy <= 4:
            raise ValueError(
                "'solver_simplex_strategy' must lie in [0,4], found "
                f"{self.solver_simplex_strategy}"
            )
        if self.central_fit_target not in {"smooth_reference", "book_mid"}:
            raise ValueError(
                "'central_fit_target' must be 'smooth_reference' or "
                f"'book_mid', found {self.central_fit_target!r}"
            )
        if not 0.0 < self.density_mass_floor < 1.0:
            raise ValueError(
                "'density_mass_floor' must lie in (0,1), found "
                f"{self.density_mass_floor}"
            )

    def backbone_total_vols(
        self,
        day: FlexMarketDay,
        *,
        geometry: FlexGeometryConfig,
    ) -> np.ndarray:
        """Return the configured Black-backbone total volatility curve."""
        if self.fixed_backbone_annual_vol is not None:
            return self.fixed_backbone_annual_vol * np.sqrt(
                geometry.model_expiries_y
            )
        assert self.backbone_mu is not None
        return self.backbone_mu * day.model_atm_total_vols


@dataclass(frozen=True)
class DlvBehaviorStatistics:
    """DLV regularity restricted to nodes carrying non-vanishing mass."""

    density_mass_floor: float
    num_nodes: int
    num_supported_nodes: int
    supported_percent: float
    minimum_bound_percent: float
    maximum_bound_percent: float
    spatial_log_jump_median: float
    spatial_log_jump_p90: float
    spatial_log_curvature_median: float
    spatial_log_curvature_p90: float
    expiry_log_change_median: float
    expiry_log_change_p90: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe representation."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class FlexLpFitResult:
    """One exact daily LP solution and its repricing diagnostics."""

    status: str
    fell_back: bool
    elapsed_seconds: float
    objective: float
    support_strikes: np.ndarray
    densities: np.ndarray
    dlvsqs: np.ndarray
    fits: np.ndarray
    statistics: ExpiryExcessStatistics
    surface_fit: SurfaceFitStatistics
    dlv_behavior: DlvBehaviorStatistics


def reprice_dlvsqs_with_torch(
    day: FlexMarketDay,
    dlvsqs: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
    config: FlexLpConfig = FlexLpConfig(),
) -> np.ndarray:
    """Reprice recovered LP DLVs through the independent Torch SANOS path."""
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    model_dts = np.diff(geometry.model_expiries_y, prepend=0.0)
    forward_vols = np.diff(
        day.model_atm_total_vols,
        prepend=0.0,
    ) / np.sqrt(model_dts)
    backbone_total_vols = config.backbone_total_vols(
        day,
        geometry=geometry,
    )
    backbone_forward_vols = np.diff(
        backbone_total_vols,
        prepend=0.0,
    ) / np.sqrt(model_dts)
    eta_values = backbone_forward_vols / forward_vols
    if isinstance(geometry, ForwardMuFlexGeometryConfig):
        eta = torch.as_tensor(eta_values[None, :], dtype=env.fdtype)
    else:
        if not np.allclose(eta_values, eta_values[0], rtol=1.0e-10):
            raise ValueError(
                "A non-forward-mu geometry cannot represent this fixed "
                "backbone total-volatility curve"
            )
        eta = torch.full((1,), float(eta_values[0]), dtype=env.fdtype)
    with torch.no_grad():
        surface = decoder.forward_batch(
            dlvsqs=torch.as_tensor(
                np.asarray(dlvsqs, dtype=np.float64)[None, ...],
                dtype=env.fdtype,
            ),
            forward_vols=torch.as_tensor(
                forward_vols[None, ...],
                dtype=env.fdtype,
            ),
            eta=eta,
        )
        call_prices = decoder.price_calls_batch(
            surface,
            expiries_y=torch.as_tensor(
                day.expiries_y[day.expiry_index][None, ...],
                dtype=env.fdtype,
            ),
            strikes=torch.as_tensor(
                day.strikes[None, ...],
                dtype=env.fdtype,
            ),
        )[0]
        prices = torch.where(
            torch.as_tensor(day.is_call),
            call_prices,
            call_prices - 1.0 + torch.as_tensor(
                day.strikes,
                dtype=env.fdtype,
            ),
        )
    return prices.cpu().numpy()


def _black_call_matrix(
    spots: np.ndarray,
    strikes: np.ndarray,
    total_vols: np.ndarray,
) -> np.ndarray:
    """Return zero-carry Black calls with rows indexed by queried strikes."""
    spots = np.asarray(spots, dtype=np.float64).reshape(1, -1)
    strikes = np.asarray(strikes, dtype=np.float64).reshape(-1, 1)
    total_vols = np.asarray(total_vols, dtype=np.float64).reshape(-1, 1)
    if not np.all(
        np.isfinite(spots)
        & (spots > 0.0)
        & np.isfinite(strikes)
        & (strikes > 0.0)
        & np.isfinite(total_vols)
        & (total_vols >= 0.0)
    ):
        raise ValueError("Black call inputs must be finite and non-negative")
    positive = total_vols > 0.0
    safe_total_vols = np.where(positive, total_vols, 1.0)
    d1 = (
        np.log(spots / strikes) + 0.5 * safe_total_vols**2
    ) / safe_total_vols
    d2 = d1 - safe_total_vols
    black = spots * ndtr(d1) - strikes * ndtr(d2)
    return np.where(positive, black, np.maximum(spots - strikes, 0.0))


def _density_from_intrinsic(support: np.ndarray) -> np.ndarray:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    density = np.diff(slopes, append=0.0)
    if np.min(density) < -1.0e-12:
        raise FloatingPointError("Intrinsic density is negative on the support")
    return np.maximum(density, 0.0)


def _pricing_matrix(
    day: FlexMarketDay,
    *,
    supports: np.ndarray,
    backbone_total_vols: np.ndarray,
    geometry: FlexGeometryConfig,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build the sparse affine map from all model densities to quote calls."""
    interpolation = sanos_time_interpolation_numpy(
        day.expiries_y[day.expiry_index],
        geometry=geometry,
    )
    upper_variance = backbone_total_vols[interpolation.upper_index] ** 2
    lower_variance = backbone_total_vols[interpolation.lower_index] ** 2
    lower_variance = np.where(
        interpolation.before_first,
        0.0,
        lower_variance,
    )
    query_total_vols = np.sqrt(
        (1.0 - interpolation.upper_weight) * lower_variance
        + interpolation.upper_weight * upper_variance
    )

    quote_count = day.num_quotes
    support_size = supports.shape[1]
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    constant = np.zeros(quote_count, dtype=np.float64)

    for upper_index in np.unique(interpolation.upper_index):
        mask = interpolation.upper_index == upper_index
        rows = np.flatnonzero(mask)
        kernels = _black_call_matrix(
            supports[upper_index],
            day.strikes[rows],
            query_total_vols[rows],
        )
        weights = interpolation.upper_weight[rows, None]
        row_parts.append(np.repeat(rows, support_size))
        column_parts.append(
            np.tile(
                upper_index * support_size
                + np.arange(support_size, dtype=np.int64),
                len(rows),
            )
        )
        value_parts.append((weights * kernels).reshape(-1))

    for lower_index in np.unique(interpolation.lower_index):
        mask = (
            (interpolation.lower_index == lower_index)
            & ~interpolation.before_first
        )
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            continue
        kernels = _black_call_matrix(
            supports[lower_index],
            day.strikes[rows],
            query_total_vols[rows],
        )
        weights = 1.0 - interpolation.upper_weight[rows, None]
        row_parts.append(np.repeat(rows, support_size))
        column_parts.append(
            np.tile(
                lower_index * support_size
                + np.arange(support_size, dtype=np.int64),
                len(rows),
            )
        )
        value_parts.append((weights * kernels).reshape(-1))

    before_rows = np.flatnonzero(interpolation.before_first)
    if len(before_rows) > 0:
        density0 = _density_from_intrinsic(supports[0])
        lower_kernels = _black_call_matrix(
            supports[0],
            day.strikes[before_rows],
            query_total_vols[before_rows],
        )
        constant[before_rows] = (
            (1.0 - interpolation.upper_weight[before_rows])
            * (lower_kernels @ density0)
        )

    matrix = sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(
            quote_count,
            geometry.num_model_expiries * support_size,
        ),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, constant


def _solve_density_lp_highs_direct(
    *,
    supports: np.ndarray,
    pricing_matrix: sparse.csr_matrix,
    pricing_constant: np.ndarray,
    call_bids: np.ndarray,
    call_asks: np.ndarray,
    call_fits: np.ndarray,
    half_spreads: np.ndarray,
    fit_weight: float,
    initial_densities: np.ndarray | None,
    algorithm: Literal["direct-simplex", "direct-ipm"],
    simplex_strategy: int,
    timeout_seconds: float,
) -> tuple[np.ndarray, str]:
    """Solve the density/slack LP through a scaled sparse HiGHS model."""
    supports = np.asarray(supports, dtype=np.float64)
    num_expiries, support_size = supports.shape
    num_inner = support_size - 2
    num_quotes = len(call_bids)
    density_count = num_expiries * support_size
    variable_count = density_count + 2 * num_quotes
    outside_slice = slice(density_count, density_count + num_quotes)
    central_slice = slice(density_count + num_quotes, variable_count)

    # Unit mass and centered first moment are equivalent to mass=mean=1,
    # while avoiding two nearly collinear rows on a narrow relative grid.
    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_values: list[float] = []
    for expiry_index, support in enumerate(supports):
        column_offset = expiry_index * support_size
        columns = column_offset + np.arange(support_size)
        equality_rows.extend(np.full(support_size, 2 * expiry_index))
        equality_columns.extend(columns)
        equality_values.extend(np.ones(support_size))
        equality_rows.extend(np.full(support_size, 2 * expiry_index + 1))
        equality_columns.extend(columns)
        equality_values.extend(support - 1.0)
    equality = sparse.coo_matrix(
        (equality_values, (equality_rows, equality_columns)),
        shape=(2 * num_expiries, density_count),
    ).tocsr()
    equality_lower = np.tile(np.asarray((1.0, 0.0)), num_expiries)
    equality_upper = equality_lower.copy()

    # Convex-order increments are the linear DLV positivity constraints.
    calendar_rows: list[np.ndarray] = []
    calendar_columns: list[np.ndarray] = []
    calendar_values: list[np.ndarray] = []
    calendar_lower = np.zeros(num_expiries * num_inner, dtype=np.float64)
    for expiry_index, support in enumerate(supports):
        row_offset = expiry_index * num_inner
        current = sparse.coo_matrix(
            np.maximum(
                support[None, :] - support[1:-1, None],
                0.0,
            )
        )
        calendar_rows.append(row_offset + current.row)
        calendar_columns.append(expiry_index * support_size + current.col)
        calendar_values.append(current.data)
        if expiry_index == 0:
            calendar_lower[:num_inner] = np.maximum(
                1.0 - support[1:-1],
                0.0,
            )
        else:
            previous = sparse.coo_matrix(
                np.maximum(
                    supports[expiry_index - 1][None, :]
                    - support[1:-1, None],
                    0.0,
                )
            )
            calendar_rows.append(row_offset + previous.row)
            calendar_columns.append(
                (expiry_index - 1) * support_size + previous.col
            )
            calendar_values.append(-previous.data)
    calendar = sparse.coo_matrix(
        (
            np.concatenate(calendar_values),
            (np.concatenate(calendar_rows), np.concatenate(calendar_columns)),
        ),
        shape=(num_expiries * num_inner, density_count),
    ).tocsr()
    calendar.eliminate_zeros()
    calendar_upper = np.full(len(calendar_lower), highspy.kHighsInf)

    # Quote slacks are measured in half-spreads. This is exactly the same
    # objective as price-unit slacks weighted by inverse half-spread.
    inverse_half_spreads = 1.0 / np.asarray(half_spreads, dtype=np.float64)
    normalized_pricing = pricing_matrix.multiply(
        inverse_half_spreads[:, None]
    ).tocsr()
    identity = sparse.eye(num_quotes, format="csr")
    quote_zero = sparse.csr_matrix((num_quotes, num_quotes))
    quote_rows = sparse.vstack(
        (
            sparse.hstack((normalized_pricing, -identity, quote_zero)),
            sparse.hstack((-normalized_pricing, -identity, quote_zero)),
            sparse.hstack((normalized_pricing, quote_zero, -identity)),
            sparse.hstack((-normalized_pricing, quote_zero, -identity)),
        ),
        format="csr",
    )
    normalized_constant = pricing_constant * inverse_half_spreads
    quote_upper = np.concatenate(
        (
            call_asks * inverse_half_spreads - normalized_constant,
            -call_bids * inverse_half_spreads + normalized_constant,
            call_fits * inverse_half_spreads - normalized_constant,
            -call_fits * inverse_half_spreads + normalized_constant,
        )
    )
    quote_lower = np.full(len(quote_upper), -highspy.kHighsInf)

    density_slack_zeros = sparse.csr_matrix(
        (equality.shape[0] + calendar.shape[0], 2 * num_quotes)
    )
    matrix = sparse.vstack(
        (
            sparse.hstack((equality, density_slack_zeros[: equality.shape[0]])),
            sparse.hstack((calendar, density_slack_zeros[equality.shape[0] :])),
            quote_rows,
        ),
        format="csr",
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_lower = np.concatenate(
        (equality_lower, calendar_lower, quote_lower)
    )
    row_upper = np.concatenate(
        (equality_upper, calendar_upper, quote_upper)
    )

    column_lower = np.zeros(variable_count, dtype=np.float64)
    column_upper = np.full(variable_count, highspy.kHighsInf)
    column_upper[:density_count] = 1.0
    costs = np.zeros(variable_count, dtype=np.float64)
    # Scaling the complete objective by the positive quote count leaves the
    # optimizer unchanged and keeps HiGHS away from its small-cost tolerance.
    costs[outside_slice] = 1.0
    costs[central_slice] = fit_weight

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("time_limit", timeout_seconds)
    solver.setOptionValue(
        "solver",
        "simplex" if algorithm == "direct-simplex" else "ipm",
    )
    solver.setOptionValue("presolve", "on")
    if algorithm == "direct-simplex":
        solver.setOptionValue("simplex_strategy", simplex_strategy)
    indices = np.arange(variable_count, dtype=np.int32)
    solver.addVars(variable_count, column_lower, column_upper)
    solver.changeColsCost(variable_count, indices, costs)
    solver.addRows(
        matrix.shape[0],
        row_lower,
        row_upper,
        matrix.nnz,
        np.asarray(matrix.indptr, dtype=np.int32),
        np.asarray(matrix.indices, dtype=np.int32),
        matrix.data,
    )

    if initial_densities is not None:
        initial = np.zeros(variable_count, dtype=np.float64)
        initial[:density_count] = initial_densities.reshape(-1)
        initial_calls = pricing_matrix @ initial[:density_count] + pricing_constant
        initial[outside_slice] = np.maximum.reduce(
            (
                initial_calls - call_asks,
                call_bids - initial_calls,
                np.zeros(num_quotes),
            )
        ) * inverse_half_spreads
        initial[central_slice] = (
            np.abs(initial_calls - call_fits) * inverse_half_spreads
        )
        solver.setSolution(variable_count, indices, initial)

    solver.run()
    model_status = solver.getModelStatus()
    status = solver.modelStatusToString(model_status).lower().replace(" ", "_")
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"Direct HiGHS density LP failed: {status}")
    solution = np.asarray(solver.getSolution().col_value, dtype=np.float64)
    densities = solution[:density_count].reshape(num_expiries, support_size)
    return densities, status


def _recover_dlvsqs(
    densities: np.ndarray,
    supports: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
    density_mass_floor: float,
    fallback_dlvsqs: np.ndarray,
    enforce_dlv_bounds: bool,
) -> np.ndarray:
    """Recover the DLV state from the LP densities on all defined nodes."""
    num_expiries = geometry.num_model_expiries
    num_strikes = geometry.num_inner_strikes
    result = np.asarray(fallback_dlvsqs, dtype=np.float64).copy()
    if result.shape != (num_expiries, num_strikes):
        raise ValueError(
            f"Expected fallback DLVs of shape {(num_expiries, num_strikes)}"
        )
    model_dts = np.diff(geometry.model_expiries_y, prepend=0.0)
    for expiry_index in range(num_expiries):
        support = supports[expiry_index]
        current_calls = (
            np.maximum(
                support[None, :] - support[1:-1, None],
                0.0,
            )
            @ densities[expiry_index]
        )
        if expiry_index == 0:
            previous_calls = np.maximum(1.0 - support[1:-1], 0.0)
        else:
            previous_calls = (
                np.maximum(
                    supports[expiry_index - 1][None, :]
                    - support[1:-1, None],
                    0.0,
                )
                @ densities[expiry_index - 1]
            )
        box = (
            support[1:-1] ** 2
            * model_dts[expiry_index]
            / (support[2:] - support[:-2])
        )
        denominator = box * densities[expiry_index, 1:-1]
        defined = denominator > density_mass_floor * box
        raw = np.divide(
            current_calls - previous_calls,
            denominator,
            out=result[expiry_index].copy(),
            where=defined,
        )
        bounded_raw = (
            np.clip(raw, geometry.min_dlvsq, geometry.max_dlvsq)
            if enforce_dlv_bounds
            else np.maximum(raw, 0.0)
        )
        result[expiry_index] = np.where(
            defined,
            bounded_raw,
            result[expiry_index],
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Recovered LP DLV state is non-finite")
    return result


def dlv_behavior_statistics(
    dlvsqs: np.ndarray,
    densities: np.ndarray,
    supports: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
    density_mass_floor: float,
) -> DlvBehaviorStatistics:
    """Measure DLV bounds and roughness where the discrete law has mass."""
    dlvsqs = np.asarray(dlvsqs, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    supports = np.asarray(supports, dtype=np.float64)
    expected_dlv_shape = (
        geometry.num_model_expiries,
        geometry.num_inner_strikes,
    )
    expected_density_shape = (
        geometry.num_model_expiries,
        geometry.num_inner_strikes + 2,
    )
    if dlvsqs.shape != expected_dlv_shape:
        raise ValueError(
            f"Expected DLV shape {expected_dlv_shape}, found {dlvsqs.shape}"
        )
    if (
        densities.shape != expected_density_shape
        or supports.shape != expected_density_shape
    ):
        raise ValueError("Density and support shapes do not match the geometry")

    active = densities[:, 1:-1] > density_mass_floor
    supported = int(np.sum(active))
    total = int(active.size)
    log_dlvsqs = np.log(
        np.maximum(dlvsqs, np.finfo(np.float64).tiny)
    )
    pair_mask = active[:, 1:] & active[:, :-1]
    jumps = np.abs(np.diff(log_dlvsqs, axis=1))[pair_mask]
    triple_mask = active[:, 2:] & active[:, 1:-1] & active[:, :-2]
    curvature = np.abs(
        log_dlvsqs[:, 2:]
        - 2.0 * log_dlvsqs[:, 1:-1]
        + log_dlvsqs[:, :-2]
    )[triple_mask]

    expiry_changes: list[np.ndarray] = []
    for expiry_index in range(1, geometry.num_model_expiries):
        previous_active = active[expiry_index - 1]
        current_active = active[expiry_index]
        if np.sum(previous_active) < 2 or np.sum(current_active) < 2:
            continue
        previous_x = np.log(
            supports[expiry_index - 1, 1:-1][previous_active]
        )
        current_x = np.log(
            supports[expiry_index, 1:-1][current_active]
        )
        lower = max(previous_x[0], current_x[0])
        upper = min(previous_x[-1], current_x[-1])
        if lower >= upper:
            continue
        comparison_x = np.linspace(lower, upper, 101)
        previous_dlv = np.interp(
            comparison_x,
            previous_x,
            log_dlvsqs[expiry_index - 1, previous_active],
        )
        current_dlv = np.interp(
            comparison_x,
            current_x,
            log_dlvsqs[expiry_index, current_active],
        )
        expiry_changes.append(np.abs(current_dlv - previous_dlv))

    def percentile(values: np.ndarray, quantile: float) -> float:
        return float(np.quantile(values, quantile)) if len(values) else 0.0

    active_dlvsqs = dlvsqs[active]
    bound_tolerance = 1.0e-8
    expiry_change_values = (
        np.concatenate(expiry_changes)
        if expiry_changes
        else np.empty(0, dtype=np.float64)
    )
    return DlvBehaviorStatistics(
        density_mass_floor=density_mass_floor,
        num_nodes=total,
        num_supported_nodes=supported,
        supported_percent=100.0 * supported / total,
        minimum_bound_percent=(
            100.0
            * float(
                np.mean(
                    active_dlvsqs <= geometry.min_dlvsq + bound_tolerance
                )
            )
            if supported
            else 0.0
        ),
        maximum_bound_percent=(
            100.0
            * float(
                np.mean(
                    active_dlvsqs >= geometry.max_dlvsq - bound_tolerance
                )
            )
            if supported
            else 0.0
        ),
        spatial_log_jump_median=percentile(jumps, 0.5),
        spatial_log_jump_p90=percentile(jumps, 0.9),
        spatial_log_curvature_median=percentile(curvature, 0.5),
        spatial_log_curvature_p90=percentile(curvature, 0.9),
        expiry_log_change_median=percentile(expiry_change_values, 0.5),
        expiry_log_change_p90=percentile(expiry_change_values, 0.9),
    )


def fit_flex_density_lp(
    day: FlexMarketDay,
    *,
    geometry: FlexGeometryConfig,
    objective_config: FlexObjectiveConfig = FlexObjectiveConfig(),
    config: FlexLpConfig = FlexLpConfig(),
    initial_densities: np.ndarray | None = None,
    verbose: Context = Context.quiet,
) -> FlexLpFitResult:
    """Fit one day's density state with SANOS DLV and martingale constraints."""
    started = perf_counter()
    num_expiries = geometry.num_model_expiries
    support_size = geometry.num_inner_strikes + 2
    supports = make_supports_from_total_vols(
        day.model_atm_total_vols,
        geometry=geometry,
    )
    backbone_total_vols = config.backbone_total_vols(
        day,
        geometry=geometry,
    )
    pricing_matrix, pricing_constant = _pricing_matrix(
        day,
        supports=supports,
        backbone_total_vols=backbone_total_vols,
        geometry=geometry,
    )

    densities = [cp.Variable(support_size, nonneg=True) for _ in range(num_expiries)]
    if initial_densities is not None:
        initial_densities = np.asarray(initial_densities, dtype=np.float64)
        expected_shape = (num_expiries, support_size)
        if initial_densities.shape != expected_shape:
            raise ValueError(
                "Expected initial densities of shape "
                f"{expected_shape}, found {initial_densities.shape}"
            )
        if not np.all(np.isfinite(initial_densities)):
            raise ValueError("Initial densities must be finite")
    constraints: list[cp.Constraint] = []
    model_dts = np.diff(geometry.model_expiries_y, prepend=0.0)
    for expiry_index, density in enumerate(densities):
        support = supports[expiry_index]
        constraints.extend(
            (
                cp.sum(density) == 1.0,
                density @ support == 1.0,
                density <= 1.0,
            )
        )
        current_calls = (
            np.maximum(
                support[None, :] - support[1:-1, None],
                0.0,
            )
            @ density
        )
        if expiry_index == 0:
            previous_calls = np.maximum(1.0 - support[1:-1], 0.0)
        else:
            previous_calls = (
                np.maximum(
                    supports[expiry_index - 1][None, :]
                    - support[1:-1, None],
                    0.0,
                )
                @ densities[expiry_index - 1]
            )
        call_increment = current_calls - previous_calls
        box = (
            support[1:-1] ** 2
            * model_dts[expiry_index]
            / (support[2:] - support[:-2])
        )
        density_box = cp.multiply(box, density[1:-1])
        if config.enforce_dlv_bounds:
            constraints.extend(
                (
                    call_increment >= geometry.min_dlvsq * density_box,
                    call_increment <= geometry.max_dlvsq * density_box,
                )
            )
        else:
            constraints.append(call_increment >= 0.0)

    flat_density = cp.hstack(densities)
    call_prices = pricing_matrix @ flat_density + pricing_constant
    parity_adjustment = np.where(day.is_call, 0.0, 1.0 - day.strikes)
    call_bids = day.bids + parity_adjustment
    call_asks = day.asks + parity_adjustment
    central_fit_prices = (
        0.5 * (day.bids + day.asks)
        if config.central_fit_target == "book_mid"
        else day.fit_prices
    )
    call_fits = central_fit_prices + parity_adjustment
    if config.solver_algorithm.startswith("direct-"):
        if config.enforce_dlv_bounds:
            raise ValueError(
                "Direct HiGHS currently supports DLV positivity without "
                "finite upper/lower DLV bounds"
            )
        density_values, status = _solve_density_lp_highs_direct(
            supports=supports,
            pricing_matrix=pricing_matrix,
            pricing_constant=pricing_constant,
            call_bids=call_bids,
            call_asks=call_asks,
            call_fits=call_fits,
            half_spreads=day.half_spreads,
            fit_weight=objective_config.fit_weight,
            initial_densities=initial_densities,
            algorithm=config.solver_algorithm,
            simplex_strategy=config.solver_simplex_strategy,
            timeout_seconds=config.timeout_seconds,
        )
        fell_back = False
    else:
        outside_slack = cp.Variable(day.num_quotes, nonneg=True)
        central_slack = cp.Variable(day.num_quotes, nonneg=True)
        constraints.extend(
            (
                outside_slack >= call_prices - call_asks,
                outside_slack >= call_bids - call_prices,
                central_slack >= call_prices - call_fits,
                central_slack >= call_fits - call_prices,
            )
        )
        quote_objective = (
            cp.sum(
                cp.multiply(
                    1.0 / day.half_spreads,
                    outside_slack + objective_config.fit_weight * central_slack,
                )
            )
            / day.num_quotes
        )
        success, status, fell_back = SanosLPSolver(
            timeout_seconds=config.timeout_seconds,
            algorithm=config.solver_algorithm,
            simplex_strategy=config.solver_simplex_strategy,
        )(
            objective=quote_objective,
            constraints=constraints,
            variables=[
                (
                    density,
                    None
                    if initial_densities is None
                    else initial_densities[expiry_index],
                )
                for expiry_index, density in enumerate(densities)
            ],
            message_context=f"FLEX {day.trading_date.isoformat()} density LP",
            verbose=verbose,
        )
        if not success:
            raise RuntimeError(
                f"FLEX density LP failed for {day.trading_date}: {status}"
            )
        density_values = np.stack(
            [np.asarray(density.value, dtype=np.float64) for density in densities]
        )
    if not np.all(np.isfinite(density_values)):
        raise FloatingPointError("FLEX density LP returned non-finite densities")
    density_values = np.maximum(density_values, 0.0)
    fallback_dlvsqs = np.broadcast_to(
        (
            day.model_atm_total_vols
            / np.sqrt(geometry.model_expiries_y)
        )[:, None]
        ** 2,
        (num_expiries, geometry.num_inner_strikes),
    )
    fallback_dlvsqs = np.clip(
        fallback_dlvsqs,
        geometry.min_dlvsq,
        geometry.max_dlvsq,
    )
    dlvsqs = _recover_dlvsqs(
        density_values,
        supports,
        geometry=geometry,
        density_mass_floor=config.density_mass_floor,
        fallback_dlvsqs=fallback_dlvsqs,
        enforce_dlv_bounds=config.enforce_dlv_bounds,
    )
    call_fit_values = (
        pricing_matrix @ density_values.reshape(-1) + pricing_constant
    )
    fits = np.where(
        day.is_call,
        call_fit_values,
        call_fit_values - 1.0 + day.strikes,
    )
    outside = (
        np.maximum(fits - day.asks, 0.0)
        + np.maximum(day.bids - fits, 0.0)
    ) / day.half_spreads
    central = np.abs(fits - day.fit_prices) / day.half_spreads
    objective = float(
        np.mean(outside + objective_config.fit_weight * central)
    )
    return FlexLpFitResult(
        status=status,
        fell_back=fell_back,
        elapsed_seconds=perf_counter() - started,
        objective=objective,
        support_strikes=supports,
        densities=density_values,
        dlvsqs=dlvsqs,
        fits=np.asarray(fits, dtype=np.float64),
        statistics=expiry_excess_statistics(
            fits,
            day.bids,
            day.asks,
            day.quote_business_dtes,
        ),
        surface_fit=surface_fit_statistics(
            fits,
            day.fit_prices,
            day.bids,
            day.asks,
        ),
        dlv_behavior=dlv_behavior_statistics(
            dlvsqs,
            density_values,
            supports,
            geometry=geometry,
            density_mass_floor=config.density_mass_floor,
        ),
    )
