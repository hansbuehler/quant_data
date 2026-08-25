"""Deterministic one-week OCE pilot for actual listed SPX options."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
from scipy.linalg import expm
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp
from scipy.stats import norm, qmc

from quantdata.dysanos.localvolbysanos.markets import DysanosMarketDay, load_market_day
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS

from ..surface_dynamics import (
    MatrixExponentialAR1,
    PCAResult,
    SurfacePanel,
    compute_surface_pca,
    fit_matrix_exponential_ar1,
    load_surface_panel,
)

UniverseName = Literal["normalised", "actual", "ITM"]
UtilityName = Literal["powexp", "entropy", "cvar10", "cvar50"]

UNIVERSES: tuple[UniverseName, ...] = ("normalised", "actual", "ITM")
UTILITIES: tuple[UtilityName, ...] = ("powexp", "entropy")
CVAR_LOWER_TAIL_FRACTIONS: dict[UtilityName, float] = {
    "cvar10": 0.90,
    "cvar50": 0.50,
}
ALL_UTILITIES: tuple[UtilityName, ...] = (*UTILITIES, *CVAR_LOWER_TAIL_FRACTIONS)


@dataclass(frozen=True, kw_only=True)
class RealTradingConfig:
    """Numerical and portfolio controls for the first real-trading pilot."""

    num_factors: int = 5
    horizon_sessions: int = 5
    scenario_paths: int = 256
    utilities: tuple[UtilityName, ...] = UTILITIES
    risk_aversions: tuple[float, ...] = (1.0, 10.0, 100.0)
    risk_budget_dollars: float = 1_000_000.0
    max_gross_fraction: float = 1.0
    max_position_fraction: float = 0.10
    max_worst_loss_fraction: float = 0.25
    max_abs_net_delta_fraction: float = 1.0
    carry_spot_position: bool = False
    strikes_per_expiry_wing: int = 2
    maximum_expiries: int = 8
    minimum_normalized_bid: float = 1.0e-8
    maximum_relative_spread: float = 1.5
    volatility_floor: float = 0.02
    volatility_cap: float = 3.0
    optimizer_max_iterations: int = 500
    seed: int = 19_870_211

    def __post_init__(self) -> None:
        verify_inp(self.num_factors >= 1, lambda: "num_factors must be positive")
        verify_inp(
            self.horizon_sessions >= 1,
            lambda: "horizon_sessions must be positive",
        )
        verify_inp(
            self.scenario_paths >= 32
            and self.scenario_paths & (self.scenario_paths - 1) == 0,
            lambda: "scenario_paths must be a power of two and at least 32",
        )
        verify_inp(
            self.utilities
            and all(utility in ALL_UTILITIES for utility in self.utilities),
            lambda: f"utilities must be selected from {ALL_UTILITIES}",
        )
        verify_inp(
            self.risk_aversions
            and all(np.isfinite(value) and value > 0.0 for value in self.risk_aversions),
            lambda: "risk_aversions must be finite and positive",
        )
        verify_inp(
            np.isfinite(self.risk_budget_dollars) and self.risk_budget_dollars > 0.0,
            lambda: "risk_budget_dollars must be finite and positive",
        )
        verify_inp(
            0.0 < self.max_position_fraction <= self.max_gross_fraction,
            lambda: "position limit must lie in (0, max_gross_fraction]",
        )
        verify_inp(
            self.max_worst_loss_fraction > 0.0,
            lambda: "max_worst_loss_fraction must be positive",
        )
        verify_inp(
            np.isfinite(self.max_abs_net_delta_fraction)
            and self.max_abs_net_delta_fraction > 0.0,
            lambda: "max_abs_net_delta_fraction must be finite and positive",
        )
        verify_inp(
            self.strikes_per_expiry_wing >= 1,
            lambda: "strikes_per_expiry_wing must be positive",
        )
        verify_inp(
            self.maximum_expiries >= 1,
            lambda: "maximum_expiries must be positive",
        )
        verify_inp(
            0.0 < self.volatility_floor < self.volatility_cap,
            lambda: "volatility bounds must be positive and ordered",
        )


@dataclass(frozen=True)
class JointDynamics:
    """Full-sample PCA surface dynamics and joint spot innovation model."""

    pca: PCAResult
    ar: MatrixExponentialAR1
    spot_intercept: float
    spot_state_coefficients: np.ndarray
    spot_surface_shock_coefficients: np.ndarray
    joint_covariance: np.ndarray
    joint_cholesky: np.ndarray
    spot_r_squared: float


@dataclass(frozen=True)
class OptionPair:
    """One strike/expiry with executable call and put books."""

    expiry_index: int
    expiry_days: int
    expiry_date: date
    strike_over_forward: float
    effective_nms: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float


@dataclass(frozen=True)
class Instrument:
    """Tradable option and the side-specific dollar cost dividers."""

    pair_index: int
    expiry_days: int
    expiry_date: date
    strike_over_forward: float
    effective_nms: float
    is_call: bool
    bid: float
    ask: float
    long_divider: float
    short_divider: float


@dataclass(frozen=True)
class ScenarioPaths:
    """Joint factor endpoints and cumulative forward-return paths."""

    terminal_factor_state: np.ndarray
    cumulative_log_forward_returns: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    """One deterministic finite-scenario OCE solution."""

    weights: np.ndarray
    oce: float
    expected_return: float
    return_std: float
    return_min: float
    return_p01: float
    return_p05: float
    return_median: float
    return_p95: float
    gross_fraction: float
    conservative_loss_fraction: float
    net_forward_delta_fraction: float
    delta_adjustment_fraction: float
    num_positions: int
    success: bool
    message: str
    iterations: int


def _regularized_cholesky(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(eigenvalues)), 1.0e-12)
    regularized = (eigenvectors * np.maximum(eigenvalues, scale * 1.0e-10)) @ (
        eigenvectors.T
    )
    return regularized, np.linalg.cholesky(regularized)


def fit_joint_dynamics(
    panel: SurfacePanel,
    spot_simple_returns: np.ndarray,
    *,
    config: RealTradingConfig,
) -> JointDynamics:
    """Fit the full-sample diagnostic joint surface/forward-return model."""

    verify_inp(
        spot_simple_returns.shape == (len(panel.dates),),
        lambda: (
            "spot returns must align with panel dates; found "
            f"{spot_simple_returns.shape} and {len(panel.dates)}"
        ),
    )
    pca = compute_surface_pca(
        panel.states["log_forward_volatility"],
        num_factors=config.num_factors,
    )
    ar = fit_matrix_exponential_ar1(
        pca.scores,
        panel.dates,
        max_nfev=4_000,
    )
    standardized = pca.scores / ar.factor_scales[None, :]
    previous = standardized[:-1]
    current = standardized[1:]
    gap_years = np.asarray(
        [
            (right - left).days / 365.25
            for left, right in zip(panel.dates, panel.dates[1:], strict=False)
        ],
        dtype=np.float64,
    )
    predictions = np.empty_like(current)
    for gap in np.unique(gap_years):
        mask = gap_years == gap
        predictions[mask] = previous[mask] @ expm(ar.generator * gap).T
    surface_shocks = current - predictions

    verify(
        np.all(spot_simple_returns[1:] > -1.0),
        lambda: "spot simple returns contain a value at or below -100%",
    )
    spot_log_returns = np.log1p(spot_simple_returns[1:])
    design = np.column_stack(
        (
            np.ones(len(previous), dtype=np.float64),
            previous,
            surface_shocks,
        )
    )
    coefficients = np.linalg.lstsq(design, spot_log_returns, rcond=None)[0]
    fitted_spot = design @ coefficients
    spot_residual = spot_log_returns - fitted_spot
    total_variance = float(
        np.sum(np.square(spot_log_returns - np.mean(spot_log_returns)))
    )
    spot_r_squared = 1.0 - float(np.sum(np.square(spot_residual))) / total_variance
    joint_residuals = np.column_stack((surface_shocks, spot_residual))
    covariance = np.cov(joint_residuals, rowvar=False, ddof=1)
    covariance, cholesky = _regularized_cholesky(covariance)
    return JointDynamics(
        pca=pca,
        ar=ar,
        spot_intercept=float(coefficients[0]),
        spot_state_coefficients=np.asarray(
            coefficients[1 : 1 + config.num_factors], dtype=np.float64
        ),
        spot_surface_shock_coefficients=np.asarray(
            coefficients[1 + config.num_factors :], dtype=np.float64
        ),
        joint_covariance=covariance,
        joint_cholesky=cholesky,
        spot_r_squared=float(spot_r_squared),
    )


def deterministic_normal_nodes(config: RealTradingConfig) -> np.ndarray:
    """Return common scrambled-Sobol normal nodes for every date and universe."""

    dimension = config.horizon_sessions * (config.num_factors + 1)
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=config.seed)
    uniforms = sampler.random_base2(int(math.log2(config.scenario_paths)))
    epsilon = np.finfo(np.float64).eps
    return norm.ppf(np.clip(uniforms, epsilon, 1.0 - epsilon)).reshape(
        config.scenario_paths,
        config.horizon_sessions,
        config.num_factors + 1,
    )


def simulate_paths(
    current_state: np.ndarray,
    calendar_gaps: np.ndarray,
    dynamics: JointDynamics,
    normal_nodes: np.ndarray,
) -> ScenarioPaths:
    """Generate deterministic conditional five-session factor/forward paths."""

    num_paths, num_steps, dimension = normal_nodes.shape
    verify_inp(
        current_state.shape == (dimension - 1,),
        lambda: f"current state has incompatible shape {current_state.shape}",
    )
    verify_inp(
        calendar_gaps.shape == (num_steps,) and np.all(calendar_gaps > 0),
        lambda: f"invalid calendar gaps {calendar_gaps}",
    )
    joint_shocks = normal_nodes @ dynamics.joint_cholesky.T
    state = np.broadcast_to(current_state, (num_paths, len(current_state))).copy()
    cumulative_returns = np.empty((num_paths, num_steps), dtype=np.float64)
    running_return = np.zeros(num_paths, dtype=np.float64)
    for step, gap_days in enumerate(calendar_gaps):
        previous = state
        surface_shock = joint_shocks[:, step, :-1]
        state = previous @ expm(
            dynamics.ar.generator * (float(gap_days) / 365.25)
        ).T + surface_shock
        log_return = (
            dynamics.spot_intercept
            + previous @ dynamics.spot_state_coefficients
            + surface_shock @ dynamics.spot_surface_shock_coefficients
            + joint_shocks[:, step, -1]
        )
        running_return += log_return
        cumulative_returns[:, step] = running_return
    return ScenarioPaths(
        terminal_factor_state=state,
        cumulative_log_forward_returns=cumulative_returns,
    )


def _select_evenly(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count)
    return values[np.unique(np.rint(positions).astype(np.int64))]


def build_option_pairs(
    day: DysanosMarketDay,
    panel: SurfacePanel,
    current_log_volatility: np.ndarray,
    *,
    config: RealTradingConfig,
) -> list[OptionPair]:
    """Build matched call/put pairs inside the statistical surface support."""

    atm_expiry_index = int(np.argmin(np.abs(panel.expiry_days - 5)))
    atm_nms_index = int(np.argmin(np.abs(panel.effective_nms)))
    front_week_volatility = float(
        np.exp(current_log_volatility[atm_expiry_index, atm_nms_index])
    )
    expiry_days = np.rint(day.expiries * DAYS_PER_YEAR).astype(np.int64)
    in_support = np.flatnonzero(
        (expiry_days >= 1) & (expiry_days <= int(panel.expiry_days[-1]))
    )
    selected_expiries: list[int] = []
    target_indices = _select_evenly(
        np.arange(len(panel.expiry_days), dtype=np.int64),
        config.maximum_expiries,
    )
    for target in panel.expiry_days[target_indices]:
        if len(in_support) == 0:
            break
        candidate = int(in_support[int(np.argmin(np.abs(expiry_days[in_support] - target)))])
        if candidate not in selected_expiries:
            selected_expiries.append(candidate)

    pairs: list[OptionPair] = []
    lower_nms = float(panel.effective_nms[0])
    upper_nms = float(panel.effective_nms[-1])
    for expiry_index in selected_expiries:
        strikes = day.strikes[expiry_index]
        nms = np.asarray(
            NMS.to_nms(
                strikes,
                float(day.expiries[expiry_index]),
                vol=front_week_volatility,
                eps_vol=front_week_volatility,
            ),
            dtype=np.float64,
        )
        call_bid = day.call_bids[expiry_index]
        call_ask = day.call_asks[expiry_index]
        put_bid = day.put_bids[expiry_index]
        put_ask = day.put_asks[expiry_index]
        call_mid = 0.5 * (call_bid + call_ask)
        put_mid = 0.5 * (put_bid + put_ask)
        valid = (
            day.call_active[expiry_index]
            & day.put_active[expiry_index]
            & np.isfinite(strikes)
            & np.isfinite(call_bid)
            & np.isfinite(call_ask)
            & np.isfinite(put_bid)
            & np.isfinite(put_ask)
            & (call_bid > config.minimum_normalized_bid)
            & (put_bid > config.minimum_normalized_bid)
            & (call_ask > call_bid)
            & (put_ask > put_bid)
            & ((call_ask - call_bid) <= config.maximum_relative_spread * call_mid)
            & ((put_ask - put_bid) <= config.maximum_relative_spread * put_mid)
            & (nms >= lower_nms)
            & (nms <= upper_nms)
            & (strikes != 1.0)
        )
        selected: list[int] = []
        for wing in (strikes < 1.0, strikes > 1.0):
            indices = np.flatnonzero(valid & wing)
            indices = indices[np.argsort(nms[indices])]
            selected.extend(
                _select_evenly(indices, config.strikes_per_expiry_wing).tolist()
            )
        for strike_index in sorted(selected, key=lambda index: strikes[index]):
            pairs.append(
                OptionPair(
                    expiry_index=expiry_index,
                    expiry_days=int(expiry_days[expiry_index]),
                    expiry_date=day.expiry_dates[expiry_index],
                    strike_over_forward=float(strikes[strike_index]),
                    effective_nms=float(nms[strike_index]),
                    call_bid=float(call_bid[strike_index]),
                    call_ask=float(call_ask[strike_index]),
                    put_bid=float(put_bid[strike_index]),
                    put_ask=float(put_ask[strike_index]),
                )
            )
    return pairs


def build_instruments(
    pairs: list[OptionPair],
    universe: UniverseName,
) -> list[Instrument]:
    """Map paired books into one of the three requested trading universes."""

    verify_inp(universe in UNIVERSES, lambda: f"unknown universe {universe!r}")
    instruments: list[Instrument] = []
    for pair_index, pair in enumerate(pairs):
        call_is_itm = pair.strike_over_forward < 1.0
        if universe == "ITM":
            is_call = call_is_itm
        else:
            is_call = not call_is_itm
        bid = pair.call_bid if is_call else pair.put_bid
        ask = pair.call_ask if is_call else pair.put_ask
        if universe == "normalised":
            long_divider = pair.put_ask if is_call else pair.call_ask
            short_divider = pair.put_bid if is_call else pair.call_bid
        else:
            long_divider = ask
            short_divider = bid
        instruments.append(
            Instrument(
                pair_index=pair_index,
                expiry_days=pair.expiry_days,
                expiry_date=pair.expiry_date,
                strike_over_forward=pair.strike_over_forward,
                effective_nms=pair.effective_nms,
                is_call=is_call,
                bid=bid,
                ask=ask,
                long_divider=long_divider,
                short_divider=short_divider,
            )
        )
    return instruments


def _interpolate_surface_batch(
    surfaces: np.ndarray,
    expiry_days: np.ndarray,
    nms_grid: np.ndarray,
    target_expiry_days: float,
    target_nms: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate one point per scenario from a surface cube."""

    target_expiry = float(np.clip(target_expiry_days, expiry_days[0], expiry_days[-1]))
    upper_expiry = int(np.searchsorted(expiry_days, target_expiry, side="right"))
    upper_expiry = min(max(upper_expiry, 1), len(expiry_days) - 1)
    lower_expiry = upper_expiry - 1
    expiry_span = float(expiry_days[upper_expiry] - expiry_days[lower_expiry])
    expiry_weight = (target_expiry - float(expiry_days[lower_expiry])) / expiry_span

    clipped_nms = np.clip(target_nms, nms_grid[0], nms_grid[-1])
    upper_nms = np.searchsorted(nms_grid, clipped_nms, side="right")
    upper_nms = np.clip(upper_nms, 1, len(nms_grid) - 1)
    lower_nms = upper_nms - 1
    nms_weight = (clipped_nms - nms_grid[lower_nms]) / (
        nms_grid[upper_nms] - nms_grid[lower_nms]
    )
    scenario_index = np.arange(len(surfaces))
    lower_values = (
        (1.0 - nms_weight) * surfaces[scenario_index, lower_expiry, lower_nms]
        + nms_weight * surfaces[scenario_index, lower_expiry, upper_nms]
    )
    upper_values = (
        (1.0 - nms_weight) * surfaces[scenario_index, upper_expiry, lower_nms]
        + nms_weight * surfaces[scenario_index, upper_expiry, upper_nms]
    )
    return (1.0 - expiry_weight) * lower_values + expiry_weight * upper_values


def scenario_log_volatility_surfaces(
    current_log_volatility: np.ndarray,
    current_standardized_state: np.ndarray,
    paths: ScenarioPaths,
    dynamics: JointDynamics,
    *,
    config: RealTradingConfig,
) -> np.ndarray:
    """Decode future log-vol surfaces while retaining today's PCA residual."""

    loadings = dynamics.pca.loadings.reshape(config.num_factors, -1)
    current_reconstruction = dynamics.pca.mean_surface + (
        (current_standardized_state * dynamics.ar.factor_scales) @ loadings
    ).reshape(current_log_volatility.shape)
    current_residual = current_log_volatility - current_reconstruction
    future_scores = paths.terminal_factor_state * dynamics.ar.factor_scales[None, :]
    future = dynamics.pca.mean_surface[None, :, :] + (
        future_scores @ loadings
    ).reshape(len(future_scores), *current_log_volatility.shape)
    future += current_residual[None, :, :]
    return np.clip(
        future,
        math.log(config.volatility_floor),
        math.log(config.volatility_cap),
    )


def instrument_return_matrix(
    instruments: list[Instrument],
    paths: ScenarioPaths,
    future_log_volatility: np.ndarray,
    panel: SurfacePanel,
    *,
    config: RealTradingConfig,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Return scenario P&L per gross-cost dollar for long and short actions."""

    num_scenarios = len(paths.terminal_factor_state)
    horizon_return = paths.cumulative_log_forward_returns[:, -1]
    forward_ratio = np.exp(horizon_return)
    atm_expiry_index = int(np.argmin(np.abs(panel.expiry_days - 5)))
    atm_nms_index = int(np.argmin(np.abs(panel.effective_nms)))
    front_week_volatility = np.exp(
        future_log_volatility[:, atm_expiry_index, atm_nms_index]
    )
    columns: list[np.ndarray] = []
    actions: list[tuple[int, str]] = []
    for instrument_index, instrument in enumerate(instruments):
        strike = instrument.strike_over_forward
        if instrument.expiry_days <= config.horizon_sessions:
            settlement_step = min(max(instrument.expiry_days, 1), config.horizon_sessions)
            settlement_ratio = np.exp(
                paths.cumulative_log_forward_returns[:, settlement_step - 1]
            )
            if instrument.is_call:
                value = np.maximum(settlement_ratio - strike, 0.0)
            else:
                value = np.maximum(strike - settlement_ratio, 0.0)
            sell_value = value
            buy_value = value
        else:
            remaining_days = float(instrument.expiry_days - config.horizon_sessions)
            future_strike = strike / forward_ratio
            future_nms = np.asarray(
                NMS.to_nms(
                    future_strike,
                    remaining_days / DAYS_PER_YEAR,
                    vol=front_week_volatility,
                    eps_vol=front_week_volatility,
                ),
                dtype=np.float64,
            )
            log_volatility = _interpolate_surface_batch(
                future_log_volatility,
                panel.expiry_days.astype(np.float64),
                panel.effective_nms,
                remaining_days,
                future_nms,
            )
            normalized_value = np.asarray(
                bs.price(
                    k=future_strike,
                    vol=np.exp(log_volatility),
                    sqrtT=math.sqrt(remaining_days / DAYS_PER_YEAR),
                    is_call=instrument.is_call,
                ),
                dtype=np.float64,
            )
            value = forward_ratio * normalized_value
            sell_value = value
            buy_value = value

        long_return = (sell_value - instrument.ask) / instrument.long_divider
        short_return = (instrument.bid - buy_value) / instrument.short_divider
        verify(
            np.all(np.isfinite(long_return)) and np.all(np.isfinite(short_return)),
            lambda: f"non-finite scenario return for instrument {instrument_index}",
        )
        assert long_return.shape == short_return.shape == (num_scenarios,)
        columns.extend((long_return, short_return))
        actions.extend(((instrument_index, "long"), (instrument_index, "short")))
    if not columns:
        return np.empty((num_scenarios, 0), dtype=np.float64), actions
    return np.column_stack(columns), actions


def instrument_action_forward_deltas(
    instruments: list[Instrument],
    actions: list[tuple[int, str]],
) -> np.ndarray:
    """Return signed entry forward delta per dollar allocated to each action."""

    if not instruments:
        verify(
            not actions,
            lambda: "actions must be empty when there are no instruments",
        )
        return np.empty(0, dtype=np.float64)

    strikes = np.asarray(
        [instrument.strike_over_forward for instrument in instruments],
        dtype=np.float64,
    )
    mid_prices = np.asarray(
        [0.5 * (instrument.bid + instrument.ask) for instrument in instruments],
        dtype=np.float64,
    )
    is_call = np.asarray(
        [instrument.is_call for instrument in instruments],
        dtype=np.bool_,
    )
    sqrt_expiry = np.sqrt(
        np.asarray(
            [instrument.expiry_days for instrument in instruments],
            dtype=np.float64,
        )
        / DAYS_PER_YEAR
    )
    implied_volatility = np.asarray(
        bs.implied(
            k=strikes,
            prices=mid_prices,
            is_call=is_call,
            sqrtT=sqrt_expiry,
            on_exceed_bounds="quiet",
        ),
        dtype=np.float64,
    )
    pure_forward_delta = np.asarray(
        bs.delta(
            k=strikes,
            vol=implied_volatility,
            is_call=is_call,
            sqrtT=sqrt_expiry,
        ),
        dtype=np.float64,
    )
    verify(
        np.all(np.isfinite(implied_volatility))
        and np.all(np.isfinite(pure_forward_delta)),
        lambda: "one or more entry actions has no finite implied volatility or delta",
    )

    action_delta = np.empty(len(actions), dtype=np.float64)
    for action_index, (instrument_index, side) in enumerate(actions):
        instrument = instruments[instrument_index]
        side_sign = 1.0 if side == "long" else -1.0
        divider = (
            instrument.long_divider if side == "long" else instrument.short_divider
        )
        action_delta[action_index] = (
            side_sign * pure_forward_delta[instrument_index] / divider
        )
    verify(
        np.all(np.isfinite(action_delta)),
        lambda: "one or more entry action delta coefficients is non-finite",
    )
    return action_delta


def _entropy_value_gradient(
    weights: np.ndarray,
    returns: np.ndarray,
    risk_aversion: float,
    fixed_scenario_returns: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    portfolio = returns @ weights
    if fixed_scenario_returns is not None:
        portfolio = portfolio + fixed_scenario_returns
    tilted_log_weights = -risk_aversion * portfolio
    log_normalizer = logsumexp(tilted_log_weights)
    probabilities = np.exp(tilted_log_weights - log_normalizer)
    value = -(log_normalizer - math.log(len(portfolio))) / risk_aversion
    gradient = returns.T @ probabilities
    return float(value), gradient


def _powexp_utility_marginal(
    values: np.ndarray,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive = values >= 0.0
    utility = np.empty_like(values)
    marginal = np.empty_like(values)
    utility[positive] = -np.expm1(-risk_aversion * values[positive]) / risk_aversion
    marginal[positive] = np.exp(-risk_aversion * values[positive])
    negative_values = values[~positive]
    utility[~positive] = (
        negative_values - 0.5 * risk_aversion * np.square(negative_values)
    )
    marginal[~positive] = 1.0 - risk_aversion * negative_values
    return utility, marginal


def _powexp_oce_value(
    portfolio: np.ndarray,
    risk_aversion: float,
) -> float:
    """Return the scalar optimized certainty equivalent of a fixed portfolio."""

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        eta = float(value[0])
        utility, marginal = _powexp_utility_marginal(
            portfolio - eta,
            risk_aversion,
        )
        oce = eta + float(np.mean(utility))
        gradient = 1.0 - float(np.mean(marginal))
        return -oce, np.asarray([-gradient], dtype=np.float64)

    optimized = minimize(
        objective,
        np.asarray([float(np.mean(portfolio))], dtype=np.float64),
        method="SLSQP",
        jac=True,
        bounds=[(-5.0, 5.0)],
        options={"maxiter": 100, "ftol": 1.0e-12, "disp": False},
    )
    verify(
        optimized.success,
        lambda: f"fixed-portfolio power-exponential OCE failed: {optimized.message}",
    )
    return -float(optimized.fun)


def _empirical_lower_tail_mean(values: np.ndarray, fraction: float) -> float:
    """Average the requested lower fraction, interpolating its boundary mass."""

    verify_inp(
        values.ndim == 1 and len(values) > 0 and np.all(np.isfinite(values)),
        lambda: "lower-tail values must be a finite non-empty vector",
    )
    verify_inp(
        np.isfinite(fraction) and 0.0 < fraction <= 1.0,
        lambda: f"lower-tail fraction must lie in (0, 1], found {fraction}",
    )
    ordered = np.sort(values)
    tail_mass = fraction * len(ordered)
    whole = int(math.floor(tail_mass))
    fractional = tail_mass - whole
    total = float(np.sum(ordered[:whole]))
    if fractional > 0.0:
        total += fractional * float(ordered[whole])
    return total / tail_mass


def _optimize_lower_tail_mean(
    returns: np.ndarray,
    action_forward_deltas: np.ndarray,
    fixed_scenario_returns: np.ndarray,
    conservative_loss: np.ndarray,
    *,
    lower_tail_fraction: float,
    config: RealTradingConfig,
    initial_forward_delta_fraction: float,
) -> OptimizationResult:
    """Maximize finite-scenario lower-tail mean as an exact linear program."""

    num_scenarios, num_actions = returns.shape
    num_variables = num_actions + 1 + num_scenarios
    eta_index = num_actions
    slack_start = eta_index + 1

    objective = np.zeros(num_variables, dtype=np.float64)
    objective[eta_index] = -1.0
    objective[slack_start:] = 1.0 / (
        lower_tail_fraction * num_scenarios
    )

    constraint_rows: list[np.ndarray] = []
    constraint_bounds: list[float] = []
    for coefficients, bound in (
        (np.ones(num_actions), config.max_gross_fraction),
        (conservative_loss, config.max_worst_loss_fraction),
        (
            action_forward_deltas,
            config.max_abs_net_delta_fraction - initial_forward_delta_fraction,
        ),
        (
            -action_forward_deltas,
            config.max_abs_net_delta_fraction + initial_forward_delta_fraction,
        ),
    ):
        row = np.zeros(num_variables, dtype=np.float64)
        row[:num_actions] = coefficients
        constraint_rows.append(row)
        constraint_bounds.append(bound)

    # u_s >= eta - fixed_s - R_s w linearizes the lower-tail OCE.
    scenario_constraints = np.zeros(
        (num_scenarios, num_variables),
        dtype=np.float64,
    )
    scenario_constraints[:, :num_actions] = -returns
    scenario_constraints[:, eta_index] = 1.0
    scenario_constraints[:, slack_start:] = -np.eye(num_scenarios)
    constraints = np.vstack((*constraint_rows, scenario_constraints))
    bounds_vector = np.concatenate(
        (np.asarray(constraint_bounds), fixed_scenario_returns)
    )
    variable_bounds = (
        [(0.0, config.max_position_fraction)] * num_actions
        + [(None, None)]
        + [(0.0, None)] * num_scenarios
    )
    optimized = linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds_vector,
        bounds=variable_bounds,
        method="highs",
        options={"maxiter": config.optimizer_max_iterations},
    )
    verify(
        optimized.x is not None,
        lambda: f"lower-tail optimization returned no solution: {optimized.message}",
    )
    weights = np.asarray(optimized.x[:num_actions], dtype=np.float64)
    weights[np.abs(weights) < 1.0e-10] = 0.0
    portfolio = fixed_scenario_returns + returns @ weights
    delta_adjustment_fraction = float(action_forward_deltas @ weights)
    net_forward_delta_fraction = (
        initial_forward_delta_fraction + delta_adjustment_fraction
    )
    verify(
        abs(net_forward_delta_fraction)
        <= config.max_abs_net_delta_fraction + 1.0e-6,
        lambda: (
            f"optimized net delta {net_forward_delta_fraction} exceeds cap "
            f"{config.max_abs_net_delta_fraction}"
        ),
    )
    quantiles = np.quantile(portfolio, [0.01, 0.05, 0.5, 0.95])
    return OptimizationResult(
        weights=weights,
        oce=_empirical_lower_tail_mean(portfolio, lower_tail_fraction),
        expected_return=float(np.mean(portfolio)),
        return_std=float(np.std(portfolio, ddof=0)),
        return_min=float(np.min(portfolio)),
        return_p01=float(quantiles[0]),
        return_p05=float(quantiles[1]),
        return_median=float(quantiles[2]),
        return_p95=float(quantiles[3]),
        gross_fraction=float(np.sum(weights)),
        conservative_loss_fraction=float(conservative_loss @ weights),
        net_forward_delta_fraction=net_forward_delta_fraction,
        delta_adjustment_fraction=delta_adjustment_fraction,
        num_positions=int(np.count_nonzero(weights > 1.0e-7)),
        success=bool(optimized.success),
        message=str(optimized.message),
        iterations=int(optimized.nit),
    )


def optimize_oce(
    returns: np.ndarray,
    action_forward_deltas: np.ndarray,
    *,
    utility: UtilityName,
    risk_aversion: float,
    config: RealTradingConfig,
    fixed_scenario_returns: np.ndarray | None = None,
    initial_forward_delta_fraction: float = 0.0,
) -> OptimizationResult:
    """Solve OCE for new actions around a fixed carried spot endowment."""

    verify_inp(
        utility in ALL_UTILITIES,
        lambda: f"utility must be selected from {ALL_UTILITIES}, found {utility}",
    )
    verify_inp(
        returns.ndim == 2 and returns.shape[0] == config.scenario_paths,
        lambda: f"invalid return matrix shape {returns.shape}",
    )
    num_actions = returns.shape[1]
    verify_inp(
        action_forward_deltas.shape == (num_actions,)
        and np.all(np.isfinite(action_forward_deltas)),
        lambda: (
            "action_forward_deltas must be finite with shape "
            f"{(num_actions,)}, found {action_forward_deltas.shape}"
        ),
    )
    if fixed_scenario_returns is None:
        fixed_scenario_returns = np.zeros(config.scenario_paths, dtype=np.float64)
    verify_inp(
        fixed_scenario_returns.shape == (config.scenario_paths,)
        and np.all(np.isfinite(fixed_scenario_returns)),
        lambda: (
            "fixed_scenario_returns must be finite with shape "
            f"{(config.scenario_paths,)}, found {fixed_scenario_returns.shape}"
        ),
    )
    verify_inp(
        np.isfinite(initial_forward_delta_fraction)
        and abs(initial_forward_delta_fraction)
        <= config.max_abs_net_delta_fraction + 1.0e-12,
        lambda: (
            "initial_forward_delta_fraction must be finite and inside the delta "
            f"cap, found {initial_forward_delta_fraction}"
        ),
    )
    if num_actions == 0:
        quantiles = np.quantile(
            fixed_scenario_returns,
            [0.01, 0.05, 0.5, 0.95],
        )
        if utility == "entropy":
            oce = _entropy_value_gradient(
                np.empty(0, dtype=np.float64),
                returns,
                risk_aversion,
                fixed_scenario_returns,
            )[0]
        elif utility == "powexp":
            oce = _powexp_oce_value(fixed_scenario_returns, risk_aversion)
        else:
            oce = _empirical_lower_tail_mean(
                fixed_scenario_returns,
                CVAR_LOWER_TAIL_FRACTIONS[utility],
            )
        return OptimizationResult(
            weights=np.empty(0, dtype=np.float64),
            oce=oce,
            expected_return=float(np.mean(fixed_scenario_returns)),
            return_std=float(np.std(fixed_scenario_returns, ddof=0)),
            return_min=float(np.min(fixed_scenario_returns)),
            return_p01=float(quantiles[0]),
            return_p05=float(quantiles[1]),
            return_median=float(quantiles[2]),
            return_p95=float(quantiles[3]),
            gross_fraction=0.0,
            conservative_loss_fraction=0.0,
            net_forward_delta_fraction=initial_forward_delta_fraction,
            delta_adjustment_fraction=0.0,
            num_positions=0,
            success=True,
            message="empty universe",
            iterations=0,
        )

    conservative_loss = np.maximum(0.0, -np.min(returns, axis=0))
    if utility in CVAR_LOWER_TAIL_FRACTIONS:
        return _optimize_lower_tail_mean(
            returns,
            action_forward_deltas,
            fixed_scenario_returns,
            conservative_loss,
            lower_tail_fraction=CVAR_LOWER_TAIL_FRACTIONS[utility],
            config=config,
            initial_forward_delta_fraction=initial_forward_delta_fraction,
        )

    constraints = (
        {
            "type": "ineq",
            "fun": lambda value: config.max_gross_fraction - np.sum(value[:num_actions]),
            "jac": lambda value: np.concatenate(
                (-np.ones(num_actions, dtype=np.float64), np.zeros(len(value) - num_actions))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda value: (
                config.max_worst_loss_fraction
                - conservative_loss @ value[:num_actions]
            ),
            "jac": lambda value: np.concatenate(
                (-conservative_loss, np.zeros(len(value) - num_actions))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda value: (
                config.max_abs_net_delta_fraction
                - initial_forward_delta_fraction
                - action_forward_deltas @ value[:num_actions]
            ),
            "jac": lambda value: np.concatenate(
                (-action_forward_deltas, np.zeros(len(value) - num_actions))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda value: (
                config.max_abs_net_delta_fraction
                + initial_forward_delta_fraction
                + action_forward_deltas @ value[:num_actions]
            ),
            "jac": lambda value: np.concatenate(
                (action_forward_deltas, np.zeros(len(value) - num_actions))
            ),
        },
    )

    if utility == "entropy":

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            oce, gradient = _entropy_value_gradient(
                value,
                returns,
                risk_aversion,
                fixed_scenario_returns,
            )
            return -oce, -gradient

        initial = np.zeros(num_actions, dtype=np.float64)
        bounds = [(0.0, config.max_position_fraction)] * num_actions
    else:
        assert utility == "powexp", utility

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            weights = value[:num_actions]
            eta = float(value[-1])
            shifted = fixed_scenario_returns + returns @ weights - eta
            utility_values, marginal = _powexp_utility_marginal(
                shifted, risk_aversion
            )
            oce = eta + float(np.mean(utility_values))
            weight_gradient = np.mean(marginal[:, None] * returns, axis=0)
            eta_gradient = 1.0 - float(np.mean(marginal))
            return -oce, -np.concatenate((weight_gradient, [eta_gradient]))

        initial = np.zeros(num_actions + 1, dtype=np.float64)
        bounds = [(0.0, config.max_position_fraction)] * num_actions + [(-5.0, 5.0)]

    optimized = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": config.optimizer_max_iterations,
            "ftol": 1.0e-10,
            "disp": False,
        },
    )
    weights = np.asarray(optimized.x[:num_actions], dtype=np.float64)
    weights[np.abs(weights) < 1.0e-10] = 0.0
    portfolio = fixed_scenario_returns + returns @ weights
    delta_adjustment_fraction = float(action_forward_deltas @ weights)
    net_forward_delta_fraction = (
        initial_forward_delta_fraction + delta_adjustment_fraction
    )
    verify(
        abs(net_forward_delta_fraction)
        <= config.max_abs_net_delta_fraction + 1.0e-6,
        lambda: (
            f"optimized net delta {net_forward_delta_fraction} exceeds cap "
            f"{config.max_abs_net_delta_fraction}"
        ),
    )
    if utility == "entropy":
        oce = _entropy_value_gradient(
            weights,
            returns,
            risk_aversion,
            fixed_scenario_returns,
        )[0]
    else:
        assert utility == "powexp", utility
        eta = float(optimized.x[-1])
        utility_values, _ = _powexp_utility_marginal(
            portfolio - eta, risk_aversion
        )
        oce = eta + float(np.mean(utility_values))
    quantiles = np.quantile(portfolio, [0.01, 0.05, 0.5, 0.95])
    return OptimizationResult(
        weights=weights,
        oce=float(oce),
        expected_return=float(np.mean(portfolio)),
        return_std=float(np.std(portfolio, ddof=0)),
        return_min=float(np.min(portfolio)),
        return_p01=float(quantiles[0]),
        return_p05=float(quantiles[1]),
        return_median=float(quantiles[2]),
        return_p95=float(quantiles[3]),
        gross_fraction=float(np.sum(weights)),
        conservative_loss_fraction=float(conservative_loss @ weights),
        net_forward_delta_fraction=net_forward_delta_fraction,
        delta_adjustment_fraction=delta_adjustment_fraction,
        num_positions=int(np.count_nonzero(weights > 1.0e-7)),
        success=bool(optimized.success),
        message=str(optimized.message),
        iterations=int(optimized.nit),
    )


def _calendar_gaps(dates: list[date], date_index: int, horizon: int) -> np.ndarray:
    if date_index + horizon < len(dates):
        horizon_dates = dates[date_index : date_index + horizon + 1]
        return np.asarray(
            [
                (right - left).days
                for left, right in zip(horizon_dates, horizon_dates[1:], strict=False)
            ],
            dtype=np.float64,
        )
    return np.asarray([1.0] * (horizon - 1) + [3.0], dtype=np.float64)


def _load_markets(
    ticker: str,
    dates: list[date],
) -> tuple[list[DysanosMarketDay | None], np.ndarray]:
    markets: list[DysanosMarketDay | None] = []
    spot_returns = np.full(len(dates), np.nan, dtype=np.float64)
    for index, market_date in enumerate(dates):
        day = load_market_day(ticker, market_date, verbose=Context.quiet)
        markets.append(day)
        if day is not None:
            spot_returns[index] = day.spot_return
        if (index + 1) % 25 == 0 or index + 1 == len(dates):
            print(f"loaded markets {index + 1}/{len(dates)}", flush=True)
    verify(
        np.all(np.isfinite(spot_returns)),
        lambda: "one or more panel dates have no finite market spot return",
    )
    return markets, spot_returns


def run_real_trading_experiment(
    ticker: str,
    panel_file: Path,
    output_dir: Path,
    *,
    config: RealTradingConfig = RealTradingConfig(),
    start_date: date | None = None,
    end_date: date | None = None,
    max_dates: int | None = None,
) -> dict[str, object]:
    """Run and persist the three-universe 2025 OCE prototype."""

    panel = load_surface_panel(panel_file)
    markets, spot_returns = _load_markets(ticker, panel.dates)
    dynamics = fit_joint_dynamics(panel, spot_returns, config=config)
    nodes = deterministic_normal_nodes(config)
    standardized_scores = dynamics.pca.scores / dynamics.ar.factor_scales[None, :]

    selected_indices = [
        index
        for index, market_date in enumerate(panel.dates)
        if (start_date is None or market_date >= start_date)
        and (end_date is None or market_date <= end_date)
    ]
    if max_dates is not None:
        selected_indices = selected_indices[:max_dates]

    result_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    carried_spot_by_series: dict[tuple[str, str, float], float] = {}
    retained_pnl_by_series: dict[tuple[str, str, float], float] = {}
    previous_position_spot_ref: float | None = None
    for run_index, date_index in enumerate(selected_indices):
        market_date = panel.dates[date_index]
        day = markets[date_index]
        if day is None:
            failures.append({"date": str(market_date), "error": "market unavailable"})
            continue
        date_succeeded = False
        try:
            current_surface = panel.states["log_forward_volatility"][date_index]
            pairs = build_option_pairs(day, panel, current_surface, config=config)
            paths = simulate_paths(
                standardized_scores[date_index],
                _calendar_gaps(panel.dates, date_index, config.horizon_sessions),
                dynamics,
                nodes,
            )
            future_surfaces = scenario_log_volatility_surfaces(
                current_surface,
                standardized_scores[date_index],
                paths,
                dynamics,
                config=config,
            )
            for universe in UNIVERSES:
                instruments = build_instruments(pairs, universe)
                returns, actions = instrument_return_matrix(
                    instruments,
                    paths,
                    future_surfaces,
                    panel,
                    config=config,
                )
                action_forward_deltas = instrument_action_forward_deltas(
                    instruments,
                    actions,
                )
                scenario_spot_return = (
                    np.exp(paths.cumulative_log_forward_returns[:, -1]) - 1.0
                )
                universe_rows.append(
                    {
                        "date": market_date,
                        "date_index": date_index,
                        "universe": universe,
                        "num_pairs": len(pairs),
                        "num_actions": len(actions),
                    }
                )
                for utility in config.utilities:
                    utility_risk_aversions = (
                        (1.0,)
                        if utility in CVAR_LOWER_TAIL_FRACTIONS
                        else config.risk_aversions
                    )
                    for risk_aversion in utility_risk_aversions:
                        series_key = (universe, utility, risk_aversion)
                        previous_spot_fraction = (
                            carried_spot_by_series.get(series_key, 0.0)
                            if config.carry_spot_position
                            else 0.0
                        )
                        if (
                            config.carry_spot_position
                            and previous_position_spot_ref is not None
                        ):
                            realized_spot_ratio = (
                                day.spot_ref / previous_position_spot_ref
                            )
                        else:
                            realized_spot_ratio = 1.0
                        carried_before_cap = (
                            previous_spot_fraction * realized_spot_ratio
                        )
                        daily_spot_pnl_fraction = previous_spot_fraction * (
                            realized_spot_ratio - 1.0
                        )
                        retained_pnl_fraction = retained_pnl_by_series.get(
                            series_key,
                            0.0,
                        ) + daily_spot_pnl_fraction
                        if config.carry_spot_position:
                            carried_spot_fraction = float(
                                np.clip(
                                    carried_before_cap,
                                    -config.max_abs_net_delta_fraction,
                                    config.max_abs_net_delta_fraction,
                                )
                            )
                        else:
                            carried_spot_fraction = 0.0
                            carried_before_cap = 0.0
                            daily_spot_pnl_fraction = 0.0
                            retained_pnl_fraction = 0.0
                        cap_clip_cash_fraction = (
                            carried_before_cap - carried_spot_fraction
                        )
                        fixed_scenario_returns = (
                            carried_spot_fraction * scenario_spot_return
                        )
                        result = optimize_oce(
                            returns,
                            action_forward_deltas,
                            utility=utility,
                            risk_aversion=risk_aversion,
                            config=config,
                            fixed_scenario_returns=fixed_scenario_returns,
                            initial_forward_delta_fraction=(
                                carried_spot_fraction
                            ),
                        )
                        if config.carry_spot_position:
                            carried_spot_by_series[series_key] = (
                                result.net_forward_delta_fraction
                            )
                            retained_pnl_by_series[series_key] = (
                                retained_pnl_fraction
                            )
                        result_rows.append(
                            {
                                "date": market_date,
                                "date_index": date_index,
                                "universe": universe,
                                "utility": utility,
                                "risk_aversion": risk_aversion,
                                "cvar_lower_tail_fraction": (
                                    CVAR_LOWER_TAIL_FRACTIONS.get(utility)
                                ),
                                "oce": result.oce,
                                "expected_return": result.expected_return,
                                "return_std": result.return_std,
                                "return_min": result.return_min,
                                "return_p01": result.return_p01,
                                "return_p05": result.return_p05,
                                "return_median": result.return_median,
                                "return_p95": result.return_p95,
                                "gross_fraction": result.gross_fraction,
                                "gross_dollars": (
                                    result.gross_fraction * config.risk_budget_dollars
                                ),
                                "conservative_loss_fraction": (
                                    result.conservative_loss_fraction
                                ),
                                "net_forward_delta_fraction": (
                                    result.net_forward_delta_fraction
                                ),
                                "net_forward_delta_dollars": (
                                    result.net_forward_delta_fraction
                                    * config.risk_budget_dollars
                                ),
                                "delta_adjustment_fraction": (
                                    result.delta_adjustment_fraction
                                ),
                                "delta_adjustment_dollars": (
                                    result.delta_adjustment_fraction
                                    * config.risk_budget_dollars
                                ),
                                "previous_spot_fraction": previous_spot_fraction,
                                "carried_spot_fraction_before_cap": (
                                    carried_before_cap
                                ),
                                "carried_spot_fraction": carried_spot_fraction,
                                "cap_clip_cash_fraction": cap_clip_cash_fraction,
                                "daily_spot_pnl_fraction": (
                                    daily_spot_pnl_fraction
                                ),
                                "cumulative_retained_pnl_fraction": (
                                    retained_pnl_fraction
                                ),
                                "daily_spot_pnl_dollars": (
                                    daily_spot_pnl_fraction
                                    * config.risk_budget_dollars
                                ),
                                "cumulative_retained_pnl_dollars": (
                                    retained_pnl_fraction
                                    * config.risk_budget_dollars
                                ),
                                "spot_ref": day.spot_ref,
                                "num_positions": result.num_positions,
                                "success": result.success,
                                "message": result.message,
                                "iterations": result.iterations,
                            }
                        )
                        for action_index in np.flatnonzero(result.weights > 1.0e-7):
                            instrument_index, side = actions[int(action_index)]
                            instrument = instruments[instrument_index]
                            divider = (
                                instrument.long_divider
                                if side == "long"
                                else instrument.short_divider
                            )
                            position_rows.append(
                                {
                                    "date": market_date,
                                    "date_index": date_index,
                                    "universe": universe,
                                    "utility": utility,
                                    "risk_aversion": risk_aversion,
                                    "cvar_lower_tail_fraction": (
                                        CVAR_LOWER_TAIL_FRACTIONS.get(utility)
                                    ),
                                    "side": side,
                                    "option_type": (
                                        "call" if instrument.is_call else "put"
                                    ),
                                    "expiry_date": instrument.expiry_date,
                                    "expiry_days": instrument.expiry_days,
                                    "strike_over_forward": (
                                        instrument.strike_over_forward
                                    ),
                                    "effective_nms": instrument.effective_nms,
                                    "bid": instrument.bid,
                                    "ask": instrument.ask,
                                    "divider": divider,
                                    "allocation_fraction": result.weights[action_index],
                                    "allocation_dollars": (
                                        result.weights[action_index]
                                        * config.risk_budget_dollars
                                    ),
                                }
                            )
            date_succeeded = True
        except Exception as exception:
            failures.append(
                {
                    "date": str(market_date),
                    "error": f"{type(exception).__name__}: {exception}",
                }
            )
        if config.carry_spot_position and date_succeeded:
            previous_position_spot_ref = day.spot_ref
        if (run_index + 1) % 5 == 0 or run_index + 1 == len(selected_indices):
            print(
                f"optimized dates {run_index + 1}/{len(selected_indices)}; "
                f"failures={len(failures)}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pl.DataFrame(result_rows) if result_rows else pl.DataFrame()
    positions = pl.DataFrame(position_rows) if position_rows else pl.DataFrame()
    universes = pl.DataFrame(universe_rows) if universe_rows else pl.DataFrame()
    results.write_parquet(output_dir / "oce_results.parquet")
    positions.write_parquet(output_dir / "positions.parquet")
    universes.write_parquet(output_dir / "universes.parquet")

    summary_rows: list[dict[str, object]] = []
    if results.height:
        summary_rows = (
            results.group_by("universe", "utility", "risk_aversion")
            .agg(
                pl.len().alias("num_dates"),
                pl.col("success").mean().alias("optimizer_success_rate"),
                pl.col("oce").mean().alias("mean_oce"),
                pl.col("expected_return").mean().alias("mean_expected_return"),
                pl.col("return_p05").mean().alias("mean_return_p05"),
                pl.col("return_min").mean().alias("mean_return_min"),
                pl.col("gross_dollars").mean().alias("mean_gross_dollars"),
                pl.col("net_forward_delta_dollars")
                .abs()
                .mean()
                .alias("mean_abs_net_forward_delta_dollars"),
                pl.col("daily_spot_pnl_dollars")
                .mean()
                .alias("mean_daily_spot_pnl_dollars"),
                pl.col("daily_spot_pnl_dollars")
                .std(ddof=0)
                .alias("daily_spot_pnl_std_dollars"),
                pl.col("cumulative_retained_pnl_dollars")
                .sort_by("date")
                .last()
                .alias("final_retained_pnl_dollars"),
                pl.col("num_positions").mean().alias("mean_num_positions"),
            )
            .sort("universe", "utility", "risk_aversion")
            .to_dicts()
        )
        pl.DataFrame(summary_rows).write_parquet(output_dir / "summary.parquet")

    report = {
        "ticker": ticker,
        "panel_file": str(panel_file),
        "output_dir": str(output_dir),
        "methodology": "full-sample 2025 diagnostic; not a look-ahead-free backtest",
        "spot_position_state": (
            "carry previous target, mark with observed spot_ref, clip to the delta "
            "cap, and include it as a nontradable OCE endowment"
            if config.carry_spot_position
            else "independent daily optimizations with no carried spot position"
        ),
        "pricing_state": "five-factor log-forward-volatility PCA",
        "spot_surface_model": "matrix-exponential surface AR with joint Gaussian innovations",
        "forward_proxy": "SPX spot log return; zero funding/discount-ratio approximation",
        "exit_price": "scenario model mid; actual bid/ask used only at entry",
        "config": asdict(config),
        "num_panel_dates": len(panel.dates),
        "num_requested_dates": len(selected_indices),
        "num_completed_dates": len({row["date"] for row in result_rows}),
        "num_failures": len(failures),
        "failures": failures,
        "surface_ar_mean_reverting": dynamics.ar.is_mean_reverting,
        "surface_ar_r_squared": dynamics.ar.r_squared,
        "spot_r_squared": dynamics.spot_r_squared,
        "summary": summary_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "quantdata/dysanos/localvolbysanos/results/2025_full_reference_grid/"
            "balanced_panel.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantdata/dysanos/localvolbysanos/real_trading/results/2025"),
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--scenario-paths", type=int, default=256)
    parser.add_argument(
        "--utilities",
        nargs="+",
        choices=ALL_UTILITIES,
        default=list(UTILITIES),
    )
    parser.add_argument("--risk-budget-dollars", type=float, default=1_000_000.0)
    parser.add_argument("--max-abs-net-delta-fraction", type=float, default=1.0)
    parser.add_argument("--carry-spot-position", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = RealTradingConfig(
        scenario_paths=arguments.scenario_paths,
        utilities=tuple(arguments.utilities),
        risk_budget_dollars=arguments.risk_budget_dollars,
        max_abs_net_delta_fraction=arguments.max_abs_net_delta_fraction,
        carry_spot_position=arguments.carry_spot_position,
    )
    report = run_real_trading_experiment(
        arguments.ticker,
        arguments.panel,
        arguments.output_dir,
        config=config,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        max_dates=arguments.max_dates,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
