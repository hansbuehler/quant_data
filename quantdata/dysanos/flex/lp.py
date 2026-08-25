"""Density/slack linear-program benchmark for a fixed daily FLEX geometry."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
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
    "1.0",
    dependencies=[FlexGeometryConfig, FlexObjectiveConfig],
)
@dataclass(frozen=True, kw_only=True)
class FlexLpConfig:
    """Numerical controls for the fixed-geometry density LP."""

    backbone_mu: float = 0.5
    timeout_seconds: float = 600.0
    density_mass_floor: float = 1.0e-7

    def __post_init__(self) -> None:
        if not 0.0 < self.backbone_mu <= 1.0:
            raise ValueError(
                "'backbone_mu' must lie in (0,1], found "
                f"{self.backbone_mu}"
            )
        if self.timeout_seconds <= 0.0:
            raise ValueError(
                "'timeout_seconds' must be positive, found "
                f"{self.timeout_seconds}"
            )
        if not 0.0 < self.density_mass_floor < 1.0:
            raise ValueError(
                "'density_mass_floor' must lie in (0,1), found "
                f"{self.density_mass_floor}"
            )


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
    eta = (
        torch.full(
            (1, geometry.num_model_expiries),
            config.backbone_mu,
            dtype=env.fdtype,
        )
        if isinstance(geometry, ForwardMuFlexGeometryConfig)
        else torch.full((1,), config.backbone_mu, dtype=env.fdtype)
    )
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
    return matrix, constant


def _recover_dlvsqs(
    densities: np.ndarray,
    supports: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
    density_mass_floor: float,
    fallback_dlvsqs: np.ndarray,
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
        result[expiry_index] = np.where(
            defined,
            np.clip(raw, geometry.min_dlvsq, geometry.max_dlvsq),
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
    backbone_total_vols = config.backbone_mu * day.model_atm_total_vols
    pricing_matrix, pricing_constant = _pricing_matrix(
        day,
        supports=supports,
        backbone_total_vols=backbone_total_vols,
        geometry=geometry,
    )

    densities = [cp.Variable(support_size, nonneg=True) for _ in range(num_expiries)]
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
        constraints.extend(
            (
                call_increment >= geometry.min_dlvsq * density_box,
                call_increment <= geometry.max_dlvsq * density_box,
            )
        )

    flat_density = cp.hstack(densities)
    call_prices = pricing_matrix @ flat_density + pricing_constant
    parity_adjustment = np.where(day.is_call, 0.0, 1.0 - day.strikes)
    call_bids = day.bids + parity_adjustment
    call_asks = day.asks + parity_adjustment
    call_fits = day.fit_prices + parity_adjustment
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
        timeout_seconds=config.timeout_seconds
    )(
        objective=quote_objective,
        constraints=constraints,
        variables=[],
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
