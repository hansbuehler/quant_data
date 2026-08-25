"""Two-day discrete-local-volatility diagnostic on a nested SANOS grid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from aae.dysanos.markets import DysanosMarketDay, MarketConfig, load_market_day
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sanos.tridiag import solve_sanos_tridiagonal


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Geometry and numerical controls for the initial DLV pilot."""

    model_expiry_days: tuple[int, ...] = (2, 5, 10, 15, 20, 30, 40, 50, 70, 120)
    nms_range: tuple[float, float] = (-3.0, 2.0)
    base_nms_strikes: int = 100
    short_anchor_days: int = 5
    merge_gap_fraction: float = 0.5
    full_reference_grid: bool = False
    front_week_vol_scaled_nms: bool = False
    smoothness: float = 0.25
    density_flux_floor: float = 1.0e-14
    density_negative_tolerance: float = 1.0e-8
    negative_increment_tolerance: float = 1.0e-6
    price_recovery_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        verify_inp(
            len(self.model_expiry_days) >= 2
            and self.model_expiry_days[0] > 0
            and all(
                left < right
                for left, right in zip(
                    self.model_expiry_days,
                    self.model_expiry_days[1:],
                    strict=False,
                )
            ),
            lambda: "model expiry days must be positive and strictly increasing",
        )
        verify_inp(
            self.short_anchor_days in self.model_expiry_days,
            lambda: "the short grid anchor must be one of the model expiries",
        )
        verify_inp(
            len(self.nms_range) == 2 and self.nms_range[0] < 0.0 < self.nms_range[1],
            lambda: f"the NMS interval must straddle zero, found {self.nms_range}",
        )
        verify_inp(
            self.base_nms_strikes >= 5,
            lambda: "at least five base NMS strikes are required",
        )
        verify_inp(
            0.0 < self.merge_gap_fraction <= 1.0,
            lambda: "merge_gap_fraction must lie in (0, 1]",
        )
        verify_inp(
            isinstance(self.full_reference_grid, bool),
            lambda: "full_reference_grid must be boolean",
        )
        verify_inp(
            isinstance(self.front_week_vol_scaled_nms, bool),
            lambda: "front_week_vol_scaled_nms must be boolean",
        )
        verify_inp(
            not (self.full_reference_grid and self.front_week_vol_scaled_nms),
            lambda: (
                "full_reference_grid and front_week_vol_scaled_nms are "
                "mutually exclusive"
            ),
        )
        verify_inp(self.density_flux_floor > 0.0, lambda: "density_flux_floor must be positive")
        verify_inp(
            self.density_negative_tolerance >= 0.0,
            lambda: "density_negative_tolerance must be non-negative",
        )
        verify_inp(
            self.negative_increment_tolerance >= 0.0,
            lambda: "negative_increment_tolerance must be non-negative",
        )
        verify_inp(
            self.price_recovery_tolerance > 0.0,
            lambda: "price_recovery_tolerance must be positive",
        )

    @property
    def model_expiries(self) -> np.ndarray:
        return np.asarray(self.model_expiry_days, dtype=np.float64) / DAYS_PER_YEAR


@dataclass(frozen=True)
class DayGridResult:
    """Complete arrays and diagnostics for one full-grid experiment day."""

    summary: dict[str, object]
    atm_vols: np.ndarray
    inner_grids: list[np.ndarray]
    model_grids: list[np.ndarray]
    call_prices: list[np.ndarray]
    forward_volatilities: list[np.ndarray]
    densities: list[np.ndarray]
    call_increments: list[np.ndarray]
    dlvs: list[np.ndarray]


def _base_nms_grid(config: ExperimentConfig) -> np.ndarray:
    lower, upper = config.nms_range
    distance = upper - lower
    num_lower = int(config.base_nms_strikes * abs(lower) // distance)
    num_lower = min(max(2, num_lower), config.base_nms_strikes - 3)
    num_upper = config.base_nms_strikes - num_lower - 1
    grid = np.concatenate(
        (
            np.linspace(lower, 0.0, num_lower, endpoint=False),
            np.linspace(0.0, upper, num_upper + 1),
        )
    )
    assert grid.shape == (config.base_nms_strikes,)
    assert np.count_nonzero(grid == 0.0) == 1
    return grid


def _atm_volatility(day: DysanosMarketDay, expiry: float) -> float:
    _, volatility = day.af_pricer.price(
        float(expiry),
        np.asarray([1.0], dtype=np.float64),
        and_vols=True,
        mode="surface",
    )
    result = float(np.asarray(volatility).item())
    verify(
        np.isfinite(result) and result > 0.0,
        lambda: f"invalid AF ATM volatility {result!r} at expiry {expiry}",
    )
    return result


def _nms_to_log_strikes(nms: np.ndarray, expiry: float, atm_vol: float) -> np.ndarray:
    strikes = NMS.from_nms(
        z=nms,
        t=float(expiry),
        vol=float(atm_vol),
        eps_vol=float(atm_vol),
    )
    return np.log(np.asarray(strikes, dtype=np.float64))


def _master_strikes(
    day: DysanosMarketDay,
    config: ExperimentConfig,
) -> tuple[np.ndarray, float, dict[str, object]]:
    nms = _base_nms_grid(config)
    short_expiry = config.short_anchor_days / DAYS_PER_YEAR
    long_expiry = config.model_expiries[-1]
    short_vol = _atm_volatility(day, short_expiry)
    if config.front_week_vol_scaled_nms:
        short_log = _nms_to_log_strikes(nms, short_expiry, short_vol)
        minimum_gap = float(np.min(np.diff(short_log)))
        return (
            np.exp(short_log),
            minimum_gap,
            {
                "grid_mode": "front_week_vol_scaled_nms",
                "short_anchor_atm_vol": short_vol,
                "minimum_log_strike_gap": minimum_gap,
            },
        )

    long_vol = _atm_volatility(day, long_expiry)
    short_log = _nms_to_log_strikes(nms, short_expiry, short_vol)
    long_log = _nms_to_log_strikes(nms, long_expiry, long_vol)
    native_gap = min(float(np.min(np.diff(short_log))), float(np.min(np.diff(long_log))))
    minimum_gap = config.merge_gap_fraction * native_gap

    # Preserve the dense short-expiry grid, then extend it with well-separated
    # points from the wider maximum-expiry grid.
    merged = list(short_log)
    for candidate in long_log:
        if np.min(np.abs(np.asarray(merged) - candidate)) >= minimum_gap:
            merged.append(float(candidate))
    merged = np.asarray(sorted(merged), dtype=np.float64)
    atm_index = int(np.argmin(np.abs(merged)))
    merged[atm_index] = 0.0
    verify(
        np.count_nonzero(merged == 0.0) == 1 and np.all(np.diff(merged) >= minimum_gap),
        lambda: "merged log-strike grid is not separated or has ambiguous ATM",
    )
    return (
        np.exp(merged),
        minimum_gap,
        {
            "short_anchor_atm_vol": short_vol,
            "long_anchor_atm_vol": long_vol,
            "minimum_log_strike_gap": minimum_gap,
        },
    )


def _expiry_grids(
    day: DysanosMarketDay,
    master_strikes: np.ndarray,
    config: ExperimentConfig,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    expiries = config.model_expiries
    atm_vols = np.asarray([_atm_volatility(day, expiry) for expiry in expiries])
    inner_grids: list[np.ndarray] = []
    model_grids: list[np.ndarray] = []
    mids: list[np.ndarray] = []
    forward_volatilities: list[np.ndarray] = []
    endpoints = np.asarray(config.nms_range, dtype=np.float64)
    front_week_index = config.model_expiry_days.index(config.short_anchor_days)
    front_week_vol = float(atm_vols[front_week_index])
    base_nms = _base_nms_grid(config)
    for expiry, atm_vol in zip(expiries, atm_vols, strict=True):
        if config.front_week_vol_scaled_nms:
            inner = np.exp(
                _nms_to_log_strikes(base_nms, float(expiry), front_week_vol)
            )
        elif config.full_reference_grid:
            inner = master_strikes
        else:
            bounds = np.exp(_nms_to_log_strikes(endpoints, float(expiry), float(atm_vol)))
            inner = master_strikes[
                (master_strikes >= bounds[0]) & (master_strikes <= bounds[1])
            ]
        verify(
            len(inner) >= 5 and np.count_nonzero(inner == 1.0) == 1,
            lambda: f"expiry {expiry} has an invalid inner strike grid",
        )
        model_grid = np.concatenate(([0.5 * inner[0]], inner, [1.5 * inner[-1]]))
        inner_grids.append(inner)
        model_grids.append(model_grid)
        prices, volatilities = day.af_pricer.price(
            float(expiry),
            inner,
            and_vols=True,
            mode="surface",
        )
        mids.append(np.asarray(prices, dtype=np.float64))
        forward_volatilities.append(np.asarray(volatilities, dtype=np.float64))

    for previous, current in zip(inner_grids, inner_grids[1:], strict=False):
        verify(
            current[0] <= previous[0] and current[-1] >= previous[-1],
            lambda: "the per-expiry master-grid slices are not nested",
        )
    return atm_vols, inner_grids, model_grids, mids, forward_volatilities


def _recover_dlvs(
    *,
    expiries: np.ndarray,
    model_grids: list[np.ndarray],
    mids: list[np.ndarray],
    density_flux_floor: float,
    density_negative_tolerance: float,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Apply the discrete Dupire/SANOS inverse directly to AF call prices."""

    densities: list[np.ndarray] = []
    fitted_calls: list[np.ndarray] = []
    call_increments: list[np.ndarray] = []
    dlvs: list[np.ndarray] = []
    previous_grid = np.asarray([1.0], dtype=np.float64)
    previous_density = np.asarray([1.0], dtype=np.float64)
    previous_expiry = 0.0
    for expiry, grid, target_calls in zip(expiries, model_grids, mids, strict=True):
        full_calls = np.concatenate(([1.0 - grid[0]], target_calls, [0.0]))
        slopes = np.diff(full_calls, prepend=1.0) / np.diff(grid, prepend=0.0)
        density = np.diff(slopes, append=0.0)
        verify(
            np.min(density) >= -density_negative_tolerance,
            lambda: f"AF call grid implies materially negative density {np.min(density)}",
        )
        inner = grid[1:-1]
        current_calls = np.maximum(grid[None, :] - inner[:, None], 0.0) @ density
        previous_calls = (
            np.maximum(previous_grid[None, :] - inner[:, None], 0.0)
            @ previous_density
        )
        increment = current_calls - previous_calls
        density_flux = (
            inner**2
            * float(expiry - previous_expiry)
            / (grid[2:] - grid[:-2])
            * density[1:-1]
        )
        valid = (density_flux > density_flux_floor) & (increment >= 0.0)
        dlvsq = np.divide(
            increment,
            density_flux,
            out=np.full_like(increment, np.nan),
            where=valid,
        )
        densities.append(density)
        fitted_calls.append(current_calls)
        call_increments.append(increment)
        dlvs.append(np.sqrt(dlvsq))
        previous_grid = grid
        previous_density = density
        previous_expiry = float(expiry)
    return densities, fitted_calls, call_increments, dlvs


def _recover_input_prices(
    *,
    expiries: np.ndarray,
    model_grids: list[np.ndarray],
    dlvs: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the SANOS implicit transitions forward on nested strike grids."""
    recovered_densities: list[np.ndarray] = []
    recovered_calls: list[np.ndarray] = []
    previous_grid = np.asarray([1.0], dtype=np.float64)
    previous_density = np.asarray([1.0], dtype=np.float64)
    previous_expiry = 0.0
    for expiry, grid, dlv in zip(expiries, model_grids, dlvs, strict=True):
        inner = grid[1:-1]
        previous_calls = (
            np.maximum(previous_grid[None, :] - grid[:, None], 0.0)
            @ previous_density
        )
        previous_calls[0] = 1.0 - grid[0]
        previous_calls[-1] = 0.0
        previous_slopes = np.diff(previous_calls, prepend=1.0) / np.diff(
            grid, prepend=0.0
        )
        right_hand_side = np.diff(previous_slopes, append=0.0)

        # A non-finite DLV occurs only where the density flux is numerically
        # zero (or a calendar increment is negative roundoff). It contributes
        # no resolvable transition, so use zero for this reconstruction test.
        finite = np.isfinite(dlv)
        dlvsq = np.zeros_like(dlv)
        dlvsq[finite] = np.square(dlv[finite])
        time_step = float(expiry - previous_expiry)
        span = grid[2:] - grid[:-2]
        upward_weight = (
            dlvsq * inner**2 * time_step / ((grid[2:] - inner) * span)
        )
        downward_weight = (
            dlvsq * inner**2 * time_step / ((inner - grid[:-2]) * span)
        )
        density = solve_sanos_tridiagonal(
            w_minus=downward_weight,
            w_plus=upward_weight,
            rhs=right_hand_side,
        )
        calls = np.maximum(grid[None, :] - inner[:, None], 0.0) @ density
        recovered_densities.append(density)
        recovered_calls.append(calls)
        previous_grid = grid
        previous_density = density
        previous_expiry = float(expiry)
    return recovered_densities, recovered_calls


def _price_recovery_summary(
    *,
    config: ExperimentConfig,
    model_grids: list[np.ndarray],
    input_calls: list[np.ndarray],
    input_densities: list[np.ndarray],
    dlvs: list[np.ndarray],
    recovered_calls: list[np.ndarray],
    recovered_densities: list[np.ndarray],
) -> dict[str, object]:
    """Summarize recovery of the AF input prices from the inferred DLVs."""

    rows: list[dict[str, object]] = []
    all_errors: list[np.ndarray] = []
    num_zeroed = 0
    for expiry_days, grid, target, input_density, dlv, recovered, density in zip(
        config.model_expiry_days,
        model_grids,
        input_calls,
        input_densities,
        dlvs,
        recovered_calls,
        recovered_densities,
        strict=True,
    ):
        error = np.abs(recovered - target)
        density_error = np.abs(density - input_density)
        zeroed = int(np.count_nonzero(~np.isfinite(dlv)))
        all_errors.append(error)
        num_zeroed += zeroed
        rows.append(
            {
                "expiry_days": expiry_days,
                "num_prices": len(target),
                "num_nonfinite_dlvs_set_to_zero": zeroed,
                "abs_error_median": float(np.median(error)),
                "abs_error_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "abs_error_max": float(np.max(error)),
                "abs_error_max_forward_bps": float(1.0e4 * np.max(error)),
                "density_abs_error_max": float(np.max(density_error)),
                "density_sum_error": float(abs(np.sum(density) - 1.0)),
                "density_mean_error": float(abs(density @ grid - 1.0)),
            }
        )
    combined = np.concatenate(all_errors)
    maximum = float(np.max(combined))
    return {
        "num_input_prices": int(combined.size),
        "num_nonfinite_dlvs_set_to_zero": num_zeroed,
        "tolerance": config.price_recovery_tolerance,
        "passed": maximum <= config.price_recovery_tolerance,
        "abs_error_median": float(np.median(combined)),
        "abs_error_rmse": float(np.sqrt(np.mean(np.square(combined)))),
        "abs_error_max": maximum,
        "abs_error_max_forward_bps": 1.0e4 * maximum,
        "expiries": rows,
    }


def _summary(
    *,
    day: DysanosMarketDay,
    config: ExperimentConfig,
    master_strikes: np.ndarray,
    minimum_gap: float,
    anchor_stats: dict[str, object],
    atm_vols: np.ndarray,
    inner_grids: list[np.ndarray],
    model_grids: list[np.ndarray],
    mids: list[np.ndarray],
    densities: list[np.ndarray],
    fitted_calls: list[np.ndarray],
    call_increments: list[np.ndarray],
    dlvs: list[np.ndarray],
) -> dict[str, object]:
    expiry_rows: list[dict[str, object]] = []
    finite_dlvs: list[np.ndarray] = []
    num_values = 0
    num_nonfinite = 0
    num_negative_increments = 0
    num_material_negative_increments = 0
    minimum_increment = np.inf
    for expiry_days, atm_vol, inner, model_grid, target, density, fitted, increment, dlv in zip(
        config.model_expiry_days,
        atm_vols,
        inner_grids,
        model_grids,
        mids,
        densities,
        fitted_calls,
        call_increments,
        dlvs,
        strict=True,
    ):
        finite = dlv[np.isfinite(dlv)]
        finite_strikes = inner[np.isfinite(dlv)]
        maximum_index = int(np.argmax(finite))
        negative_mask = increment < -config.negative_increment_tolerance
        finite_dlvs.append(finite)
        num_values += dlv.size
        num_nonfinite += int(np.count_nonzero(~np.isfinite(dlv)))
        num_negative_increments += int(np.count_nonzero(increment < 0.0))
        num_material_negative_increments += int(
            np.count_nonzero(increment < -config.negative_increment_tolerance)
        )
        minimum_increment = min(minimum_increment, float(np.min(increment)))
        error = np.abs(fitted - target)
        expiry_rows.append(
            {
                "expiry_days": expiry_days,
                "atm_vol": float(atm_vol),
                "num_inner_strikes": len(inner),
                "inner_strike_min": float(inner[0]),
                "inner_strike_max": float(inner[-1]),
                "outer_strike_min": float(model_grid[0]),
                "outer_strike_max": float(model_grid[-1]),
                "density_min": float(np.min(density)),
                "density_sum_error": float(abs(np.sum(density) - 1.0)),
                "density_mean_error": float(abs(density @ model_grid - 1.0)),
                "call_increment_min": float(np.min(increment)),
                "num_negative_call_increments": int(np.count_nonzero(increment < 0.0)),
                "num_material_negative_call_increments": int(
                    np.count_nonzero(increment < -config.negative_increment_tolerance)
                ),
                "dlv_min": float(np.min(finite)),
                "dlv_median": float(np.median(finite)),
                "dlv_p95": float(np.quantile(finite, 0.95)),
                "dlv_p99": float(np.quantile(finite, 0.99)),
                "dlv_max": float(np.max(finite)),
                "dlv_max_strike": float(finite_strikes[maximum_index]),
                "num_dlv_nonfinite": int(np.count_nonzero(~np.isfinite(dlv))),
                "nonfinite_dlv_strikes": inner[~np.isfinite(dlv)].tolist(),
                "material_negative_call_increments": [
                    {"strike": float(strike), "increment": float(value)}
                    for strike, value in zip(
                        inner[negative_mask],
                        increment[negative_mask],
                        strict=True,
                    )
                ],
                "num_dlv_above_2": int(np.count_nonzero(finite > 2.0)),
                "num_dlv_above_5": int(np.count_nonzero(finite > 5.0)),
                "density_repricing_abs_error_median": float(np.median(error)),
                "density_repricing_abs_error_max": float(np.max(error)),
            }
        )
    combined = np.concatenate(finite_dlvs)
    return {
        "date": day.market_date.isoformat(),
        "spot_ref": day.spot_ref,
        "source_expiries": len(day.expiries),
        "master_grid": {
            "strike_coordinate": "K_over_F",
            "num_strikes": len(master_strikes),
            "strike_min": float(master_strikes[0]),
            "strike_max": float(master_strikes[-1]),
            "minimum_log_strike_gap": minimum_gap,
            **anchor_stats,
        },
        "dlv": {
            "num_values": num_values,
            "num_finite": int(combined.size),
            "num_nonfinite": num_nonfinite,
            "num_negative_call_increments": num_negative_increments,
            "num_material_negative_call_increments": num_material_negative_increments,
            "minimum_call_increment": minimum_increment,
            "median": float(np.median(combined)),
            "p95": float(np.quantile(combined, 0.95)),
            "p99": float(np.quantile(combined, 0.99)),
            "max": float(np.max(combined)),
            "num_above_2": int(np.count_nonzero(combined > 2.0)),
            "num_above_5": int(np.count_nonzero(combined > 5.0)),
        },
        "expiries": expiry_rows,
    }


def compute_day_grid(
    ticker: str,
    market_date: date,
    *,
    config: ExperimentConfig = ExperimentConfig(),
    verbose: Context = Context.all,
) -> DayGridResult:
    """Compute the complete DLV grid and diagnostics for one market day."""

    day = load_market_day(
        ticker,
        market_date,
        config=MarketConfig(smoothness=config.smoothness),
        verbose=verbose(1),
    )
    verify(day is not None, lambda: f"no filtered market for {ticker} {market_date}")
    master, minimum_gap, anchor_stats = _master_strikes(day, config)
    atm_vols, inner_grids, model_grids, mids, forward_volatilities = _expiry_grids(
        day, master, config
    )
    densities, fitted_calls, call_increments, dlvs = _recover_dlvs(
        expiries=config.model_expiries,
        model_grids=model_grids,
        mids=mids,
        density_flux_floor=config.density_flux_floor,
        density_negative_tolerance=config.density_negative_tolerance,
    )
    result = _summary(
        day=day,
        config=config,
        master_strikes=master,
        minimum_gap=minimum_gap,
        anchor_stats=anchor_stats,
        atm_vols=atm_vols,
        inner_grids=inner_grids,
        model_grids=model_grids,
        mids=mids,
        densities=densities,
        fitted_calls=fitted_calls,
        call_increments=call_increments,
        dlvs=dlvs,
    )
    if config.full_reference_grid or config.front_week_vol_scaled_nms:
        recovered_densities, recovered_calls = _recover_input_prices(
            expiries=config.model_expiries,
            model_grids=model_grids,
            dlvs=dlvs,
        )
        result["input_price_recovery"] = _price_recovery_summary(
            config=config,
            model_grids=model_grids,
            input_calls=mids,
            input_densities=densities,
            dlvs=dlvs,
            recovered_calls=recovered_calls,
            recovered_densities=recovered_densities,
        )
    return DayGridResult(
        summary=result,
        atm_vols=atm_vols,
        inner_grids=inner_grids,
        model_grids=model_grids,
        call_prices=mids,
        forward_volatilities=forward_volatilities,
        densities=densities,
        call_increments=call_increments,
        dlvs=dlvs,
    )


def run_day(
    ticker: str,
    market_date: date,
    *,
    config: ExperimentConfig = ExperimentConfig(),
    verbose: Context = Context.all,
) -> tuple[dict[str, object], list[np.ndarray], list[np.ndarray]]:
    """Invert and summarize one day's nested-grid DLV surface."""

    result = compute_day_grid(
        ticker,
        market_date,
        config=config,
        verbose=verbose,
    )
    return result.summary, result.inner_grids, result.dlvs


def _plot(
    results: list[dict[str, object]],
    strike_grids: list[list[np.ndarray]],
    dlvs: list[list[np.ndarray]],
    output_file: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6), squeeze=False)
    for axis, result, grids, day_dlvs in zip(
        axes[0], results, strike_grids, dlvs, strict=True
    ):
        for expiry, strikes, values in zip(
            result["expiries"], grids, day_dlvs, strict=True
        ):
            axis.plot(strikes, values, label=f"{expiry['expiry_days']}d", linewidth=1.2)
        axis.set_title(str(result["date"]))
        axis.set_xlabel("forward-relative strike K/F")
        axis.set_ylabel("discrete local volatility")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.suptitle("Local volatility recovered from the AF SANOS surface")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_experiment(
    *,
    dates: tuple[date, ...] = (date(2024, 3, 5), date(2024, 3, 6)),
    ticker: str = "SPX",
    config: ExperimentConfig = ExperimentConfig(),
    output_directory: Path = Path("aae/ideas/localvolbysanos/results"),
    verbose: Context = Context.all,
) -> dict[str, object]:
    """Run the requested dates and write a compact JSON summary and plot."""

    verify_inp(len(dates) > 0, lambda: "at least one date is required")
    results: list[dict[str, object]] = []
    strike_grids: list[list[np.ndarray]] = []
    dlvs: list[list[np.ndarray]] = []
    for market_date in dates:
        result, grids, day_dlvs = run_day(
            ticker,
            market_date,
            config=config,
            verbose=verbose(1),
        )
        results.append(result)
        strike_grids.append(grids)
        dlvs.append(day_dlvs)
    report = {
        "ticker": ticker,
        "dates": [value.isoformat() for value in dates],
        "config": asdict(config),
        "days": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "two_day_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    _plot(results, strike_grids, dlvs, output_directory / "two_day_dlvs.png")
    return report


if __name__ == "__main__":
    completed = run_experiment()
    for day in completed["days"]:
        aggregate = day["dlv"]
        print(
            day["date"],
            f"DLV median={aggregate['median']:.6f}",
            f"p99={aggregate['p99']:.6f}",
            f"max={aggregate['max']:.6f}",
            f">2={aggregate['num_above_2']}",
            f"nonfinite={aggregate['num_nonfinite']}",
        )
