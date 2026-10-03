"""A forward-start stochastic-implied-volatility OCE experiment.

The stock has constant 20% diffusion volatility. Its Brownian shock is
correlated with the Brownian motion driving an Ornstein--Uhlenbeck implied-vol
state. The state evolves for ten business days, then an ATM forward-start
option is priced from the observed sigmoid-transformed state and held for the
next ten business days. Spot is tradable over the same second interval.

Trading policies are linear in their parameters. Consequently the exponential
OCE problem is convex and its fitted density has a transparent interpretation:
the gradient is exactly the vector of traded-return moments under that density.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class MarketConfig:
    """Parameters of the two-period market simulation."""

    kappa: float = 5.0
    xi: float = 1.0
    rho: float = -0.5
    equity_volatility: float = 0.2
    step_years: float = 10.0 / 255.0
    initial_v: float = 0.0
    normalized_moneyness: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    paths: int = 10_000
    seed: int = 20260930

    def __post_init__(self) -> None:
        verify_inp(self.kappa > 0.0, lambda: f"kappa must be positive, found {self.kappa}")
        verify_inp(self.xi > 0.0, lambda: f"xi must be positive, found {self.xi}")
        verify_inp(
            -1.0 < self.rho < 1.0,
            lambda: f"rho must lie strictly between -1 and 1, found {self.rho}",
        )
        verify_inp(
            self.equity_volatility > 0.0,
            lambda: "equity volatility must be positive",
        )
        verify_inp(self.step_years > 0.0, lambda: "step length must be positive")
        verify_inp(
            len(self.normalized_moneyness) > 0
            and all(math.isfinite(value) for value in self.normalized_moneyness),
            lambda: "normalized moneyness must be a nonempty finite tuple",
        )
        verify_inp(
            len(set(self.normalized_moneyness)) == len(self.normalized_moneyness),
            lambda: "normalized moneyness values must be unique",
        )
        verify_inp(self.paths > 1, lambda: "at least two paths are required")


@dataclass(frozen=True)
class MarketPaths:
    """Simulated state and one-period traded returns."""

    v: np.ndarray
    implied_volatility: np.ndarray
    stock_gross_return: np.ndarray
    stock_return: np.ndarray
    option_return: np.ndarray
    option_price: np.ndarray
    strikes: np.ndarray


@dataclass(frozen=True)
class OceResult:
    """Diagnostics from a full-population exponential-OCE solve."""

    coefficients: np.ndarray
    density: np.ndarray
    initial_oce: float
    oce: float
    closure_evaluations: int
    seconds: float
    gradient_l2_norm: float
    gradient_rms: float
    gradient_maximum_absolute: float
    effective_sample_size: float
    maximum_log_density_ratio: float
    radial_oce_derivative: float
    hessian_condition_number: float
    converged: bool
    preconditioned: bool
    optimizer: str
    updates: int
    trace: tuple[dict[str, float | int], ...] = ()

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible scalar diagnostics."""

        result: dict[str, Any] = {
            "initial_oce": self.initial_oce,
            "oce": self.oce,
            "closure_evaluations": self.closure_evaluations,
            "seconds": self.seconds,
            "gradient_l2_norm": self.gradient_l2_norm,
            "gradient_rms": self.gradient_rms,
            "gradient_maximum_absolute": self.gradient_maximum_absolute,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_size_fraction": self.effective_sample_size / len(self.density),
            "maximum_log_density_ratio": self.maximum_log_density_ratio,
            "radial_oce_derivative": self.radial_oce_derivative,
            "hessian_condition_number": self.hessian_condition_number,
            "coefficient_l2_norm": float(np.linalg.norm(self.coefficients)),
            "coefficient_maximum_absolute": float(np.max(np.abs(self.coefficients))),
            "converged": self.converged,
            "preconditioned": self.preconditioned,
            "optimizer": self.optimizer,
            "updates": self.updates,
        }
        if self.trace:
            result["trace_points"] = len(self.trace)
        return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Evaluate a stable NumPy sigmoid."""

    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def black_scholes_call(
    *,
    strike: np.ndarray,
    volatility: np.ndarray,
    time_to_expiry: float,
) -> np.ndarray:
    """Return zero-rate Black--Scholes call prices for unit spot."""

    strike = np.asarray(strike, dtype=np.float64)
    volatility = np.asarray(volatility, dtype=np.float64)
    strike, volatility = np.broadcast_arrays(strike, volatility)
    if np.any(~np.isfinite(strike) | (strike <= 0.0)):
        raise ValueError("Black--Scholes strike must be finite-positive")
    if np.any(~np.isfinite(volatility) | (volatility <= 0.0)):
        raise ValueError("Black--Scholes volatility must be finite-positive")
    if not math.isfinite(time_to_expiry) or time_to_expiry <= 0.0:
        raise ValueError("time to expiry must be finite-positive")
    total_volatility = volatility * math.sqrt(time_to_expiry)
    d1 = (-np.log(strike) + 0.5 * np.square(total_volatility)) / total_volatility
    d2 = d1 - total_volatility

    def normal_cdf(value: np.ndarray) -> np.ndarray:
        flat = np.fromiter(
            (0.5 * math.erfc(-float(item) / math.sqrt(2.0)) for item in value.flat),
            dtype=np.float64,
            count=value.size,
        )
        return flat.reshape(value.shape)

    return normal_cdf(d1) - strike * normal_cdf(d2)


def atm_black_scholes_call(*, volatility: np.ndarray, time_to_expiry: float) -> np.ndarray:
    """Return the zero-rate ATM Black--Scholes call price for unit spot."""

    return black_scholes_call(
        strike=np.ones_like(volatility, dtype=np.float64),
        volatility=volatility,
        time_to_expiry=time_to_expiry,
    )


def simulate_market(config: MarketConfig) -> MarketPaths:
    """Simulate two exact OU steps and the consistently correlated stock shocks."""

    generator = np.random.default_rng(config.seed)
    normal = generator.standard_normal((2, config.paths, 3), dtype=np.float64)

    decay = math.exp(-config.kappa * config.step_years)
    ou_variance = (1.0 - decay * decay) / (2.0 * config.kappa)
    ou_stock_covariance = (1.0 - decay) / config.kappa
    shared_loading = ou_stock_covariance / math.sqrt(config.step_years)
    residual_variance = max(0.0, ou_variance - shared_loading * shared_loading)

    v = np.empty((config.paths, 3), dtype=np.float64)
    stock_gross_return = np.empty((config.paths, 2), dtype=np.float64)
    v[:, 0] = config.initial_v
    for step in range(2):
        stock_wiener = math.sqrt(config.step_years) * normal[step, :, 0]
        independent_wiener = math.sqrt(config.step_years) * normal[step, :, 1]
        ou_innovation = (
            shared_loading * normal[step, :, 0]
            + math.sqrt(residual_variance) * normal[step, :, 2]
        )
        v[:, step + 1] = decay * v[:, step] + config.xi * ou_innovation
        equity_wiener = (
            config.rho * stock_wiener
            + math.sqrt(1.0 - config.rho * config.rho) * independent_wiener
        )
        stock_gross_return[:, step] = np.exp(
            -0.5 * config.equity_volatility**2 * config.step_years
            + config.equity_volatility * equity_wiener
        )

    implied_volatility = _sigmoid(v)
    normalized_moneyness = np.asarray(config.normalized_moneyness, dtype=np.float64)
    strikes = np.exp(
        normalized_moneyness * config.equity_volatility * math.sqrt(config.step_years)
    )
    option_price = black_scholes_call(
        strike=strikes[None, None, :],
        volatility=implied_volatility[:, :2, None],
        time_to_expiry=config.step_years,
    )
    payoff = np.maximum(stock_gross_return[:, :, None] - strikes[None, None, :], 0.0)
    option_return = payoff / option_price - 1.0
    stock_return = stock_gross_return - 1.0

    arrays = (
        v,
        implied_volatility,
        stock_gross_return,
        stock_return,
        option_return,
        option_price,
        strikes,
    )
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise RuntimeError("market simulation produced a non-finite value")
    return MarketPaths(
        v=v,
        implied_volatility=implied_volatility,
        stock_gross_return=stock_gross_return,
        stock_return=stock_return,
        option_return=option_return,
        option_price=option_price,
        strikes=strikes,
    )


def build_trading_design(paths: MarketPaths) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build P&L returns for day-10 spot and option positions linear in ``v``.

    At day 10, both positions are affine functions of the then-observable OU
    state. Past returns are deliberately excluded: expressed in units of
    current spot, the continuation problem is Markov in ``v``.
    """

    path_count = len(paths.v)
    if paths.v.shape != (path_count, 3):
        raise ValueError(f"expected v shape {(path_count, 3)}, found {paths.v.shape}")
    if paths.stock_return.shape != (path_count, 2):
        raise ValueError("stock returns must have shape (paths, 2)")
    expected_option_shape = (path_count, 2, len(paths.strikes))
    if paths.option_return.shape != expected_option_shape:
        raise ValueError(
            f"option returns must have shape {expected_option_shape}, "
            f"found {paths.option_return.shape}"
        )

    state_scale = math.sqrt(np.mean(np.square(paths.v[:, 1])))
    if not math.isfinite(state_scale) or state_scale <= 0.0:
        raise ValueError("day-10 OU state must have positive finite RMS")
    normalized_v = paths.v[:, 1] / state_scale
    information = np.column_stack((np.ones(path_count, dtype=np.float64), normalized_v))
    columns: list[np.ndarray] = []
    names: list[str] = []
    information_names = ("constant", "v_day10")
    instruments = [("stock", paths.stock_return[:, 1])]
    instruments.extend(
        (f"call_k{strike:.6f}", paths.option_return[:, 1, strike_index])
        for strike_index, strike in enumerate(paths.strikes)
    )
    for instrument_name, instrument_return in instruments:
        for feature_index, feature_name in enumerate(information_names):
            columns.append(information[:, feature_index] * instrument_return)
            names.append(f"day10_{instrument_name}_x_{feature_name}")
    design = np.column_stack(columns)
    if np.any(~np.isfinite(design)):
        raise RuntimeError("trading design contains non-finite values")
    return design, tuple(names)


def solve_exponential_oce(
    design: np.ndarray,
    *,
    base_density: np.ndarray | None = None,
    initial_coefficients: np.ndarray | None = None,
    precondition: bool = True,
    maximum_iterations: int = 500,
    gradient_tolerance: float = 1.0e-9,
) -> OceResult:
    """Solve the empirical exponential-OCE problem with full-batch L-BFGS.

    ``design @ coefficients`` is terminal P&L. The minimized loss is
    ``log E_base[exp(-PnL)]`` and the returned OCE is its negative. Column RMS
    scaling changes only optimizer coordinates, not the strategy span.
    """

    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] < 2 or design.shape[1] == 0:
        raise ValueError("design must be a nonempty two-dimensional population")
    if np.any(~np.isfinite(design)):
        raise ValueError("design must be finite")
    if maximum_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("gradient tolerance must be finite-positive")

    paths = design.shape[0]
    if base_density is None:
        base_probability = np.full(paths, 1.0 / paths, dtype=np.float64)
    else:
        base_probability = np.asarray(base_density, dtype=np.float64)
        if base_probability.shape != (paths,):
            raise ValueError("base density must have one entry per path")
        if np.any(~np.isfinite(base_probability) | (base_probability <= 0.0)):
            raise ValueError("base density must be finite-positive")
        base_probability = base_probability / np.sum(base_probability, dtype=np.float64)

    scale = np.sqrt(np.mean(np.square(design), axis=0)) if precondition else np.ones(design.shape[1])
    if np.any(~np.isfinite(scale) | (scale <= 0.0)):
        raise ValueError("every trading-design column must have positive finite RMS")
    optimized_design = torch.as_tensor(design / scale, dtype=torch.float64)
    log_base_probability = torch.log(torch.as_tensor(base_probability, dtype=torch.float64))
    if initial_coefficients is None:
        initial = np.zeros(design.shape[1], dtype=np.float64)
    else:
        initial = np.asarray(initial_coefficients, dtype=np.float64)
        if initial.shape != (design.shape[1],) or np.any(~np.isfinite(initial)):
            raise ValueError("initial coefficients must be a finite design-width vector")
    optimized_coefficients = torch.tensor(
        initial * scale, dtype=torch.float64, requires_grad=True
    )
    with torch.no_grad():
        initial_pnl = optimized_design @ optimized_coefficients
        initial_log_normalizer = torch.logsumexp(
            log_base_probability - initial_pnl, dim=0
        )
    optimizer = torch.optim.LBFGS(
        [optimized_coefficients],
        lr=1.0,
        max_iter=maximum_iterations,
        tolerance_grad=gradient_tolerance,
        tolerance_change=1.0e-14,
        history_size=min(100, design.shape[1] * 5),
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        pnl = optimized_design @ optimized_coefficients
        loss = torch.logsumexp(log_base_probability - pnl, dim=0)
        loss.backward()
        return loss

    started = time.monotonic()
    optimizer.step(closure)
    seconds = time.monotonic() - started
    updates = int(optimizer.state[optimized_coefficients].get("n_iter", 0))

    with torch.no_grad():
        pnl = optimized_design @ optimized_coefficients
        log_probability = log_base_probability - pnl
        log_normalizer = torch.logsumexp(log_probability, dim=0)
        probability = torch.exp(log_probability - log_normalizer)
        gradient = -(optimized_design.T @ probability)
        coefficients = optimized_coefficients / torch.as_tensor(scale, dtype=torch.float64)
        centered_design = optimized_design - probability @ optimized_design
        hessian = centered_design.T @ (probability[:, None] * centered_design)
        eigenvalues = torch.linalg.eigvalsh(hessian)

    density = probability.numpy()
    gradient_rms = float(torch.sqrt(torch.mean(torch.square(gradient))))
    gradient_l2_norm = float(torch.linalg.vector_norm(gradient))
    gradient_maximum = float(torch.max(torch.abs(gradient)))
    positive_eigenvalues = eigenvalues[eigenvalues > torch.max(eigenvalues) * 1.0e-14]
    condition_number = float(torch.max(eigenvalues) / torch.min(positive_eigenvalues))
    return OceResult(
        coefficients=coefficients.numpy(),
        density=density,
        initial_oce=-float(initial_log_normalizer),
        oce=-float(log_normalizer),
        closure_evaluations=closure_evaluations,
        seconds=seconds,
        gradient_l2_norm=gradient_l2_norm,
        gradient_rms=gradient_rms,
        gradient_maximum_absolute=gradient_maximum,
        effective_sample_size=float(1.0 / np.sum(np.square(density), dtype=np.float64)),
        maximum_log_density_ratio=float(np.max(np.abs(np.log(density / base_probability)))),
        radial_oce_derivative=float(probability @ pnl),
        hessian_condition_number=condition_number,
        converged=gradient_l2_norm <= gradient_tolerance,
        preconditioned=precondition,
        optimizer="lbfgs",
        updates=updates,
    )


def solve_exponential_oce_adam(
    design: np.ndarray,
    *,
    base_density: np.ndarray | None = None,
    initial_coefficients: np.ndarray | None = None,
    precondition: bool = True,
    learning_rate: float = 0.05,
    maximum_updates: int = 10_000,
    gradient_tolerance: float = 1.0e-10,
    trace_interval: int = 0,
    reference_oce: float | None = None,
) -> OceResult:
    """Solve the same full-population OCE problem using fixed-rate Adam."""

    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] < 2 or design.shape[1] == 0:
        raise ValueError("design must be a nonempty two-dimensional population")
    if np.any(~np.isfinite(design)):
        raise ValueError("design must be finite")
    if maximum_updates <= 0:
        raise ValueError("maximum updates must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Adam learning rate must be finite-positive")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("gradient tolerance must be finite-positive")
    if trace_interval < 0:
        raise ValueError("trace interval cannot be negative")
    if reference_oce is not None and not math.isfinite(reference_oce):
        raise ValueError("reference OCE must be finite")

    paths = design.shape[0]
    if base_density is None:
        base_probability = np.full(paths, 1.0 / paths, dtype=np.float64)
    else:
        base_probability = np.asarray(base_density, dtype=np.float64)
        if base_probability.shape != (paths,):
            raise ValueError("base density must have one entry per path")
        if np.any(~np.isfinite(base_probability) | (base_probability <= 0.0)):
            raise ValueError("base density must be finite-positive")
        base_probability = base_probability / np.sum(base_probability, dtype=np.float64)

    scale = np.sqrt(np.mean(np.square(design), axis=0)) if precondition else np.ones(design.shape[1])
    if np.any(~np.isfinite(scale) | (scale <= 0.0)):
        raise ValueError("every trading-design column must have positive finite RMS")
    if initial_coefficients is None:
        initial = np.zeros(design.shape[1], dtype=np.float64)
    else:
        initial = np.asarray(initial_coefficients, dtype=np.float64)
        if initial.shape != (design.shape[1],) or np.any(~np.isfinite(initial)):
            raise ValueError("initial coefficients must be a finite design-width vector")

    optimized_design = torch.as_tensor(design / scale, dtype=torch.float64)
    raw_design = torch.as_tensor(design, dtype=torch.float64)
    log_base_probability = torch.log(torch.as_tensor(base_probability, dtype=torch.float64))
    optimized_coefficients = torch.tensor(
        initial * scale, dtype=torch.float64, requires_grad=True
    )
    with torch.no_grad():
        initial_pnl = optimized_design @ optimized_coefficients
        initial_log_normalizer = torch.logsumexp(
            log_base_probability - initial_pnl, dim=0
        )

    optimizer = torch.optim.Adam([optimized_coefficients], lr=learning_rate)
    started = time.monotonic()
    updates = 0
    converged = False
    trace: list[dict[str, float | int]] = []
    previous_parameters: torch.Tensor | None = None
    two_updates_ago: torch.Tensor | None = None
    previous_gradient: torch.Tensor | None = None

    def record_trace(
        update: int,
        *,
        step_norm: float,
        cycle_distance: float,
        consecutive_gradient_cosine: float,
    ) -> None:
        """Record stationarity, curvature, weight, and Adam-state diagnostics."""

        with torch.no_grad():
            trace_pnl = optimized_design @ optimized_coefficients
            trace_log_probability = log_base_probability - trace_pnl
            trace_log_normalizer = torch.logsumexp(trace_log_probability, dim=0)
            trace_probability = torch.exp(trace_log_probability - trace_log_normalizer)
            scaled_gradient = -(optimized_design.T @ trace_probability)
            raw_gradient = -(raw_design.T @ trace_probability)
            centered = optimized_design - trace_probability @ optimized_design
            hessian = centered.T @ (trace_probability[:, None] * centered)
            eigenvalues = torch.linalg.eigvalsh(hessian)
            positive = eigenvalues[eigenvalues > torch.max(eigenvalues) * 1.0e-14]
            condition_number = torch.max(eigenvalues) / torch.min(positive)
            off_diagonal = hessian - torch.diag(torch.diag(hessian))
            newton_decrement_squared = scaled_gradient @ (
                torch.linalg.pinv(hessian, rtol=1.0e-12) @ scaled_gradient
            )
            state = optimizer.state.get(optimized_coefficients, {})
            if update > 0 and "exp_avg_sq" in state:
                beta2 = optimizer.param_groups[0]["betas"][1]
                epsilon = optimizer.param_groups[0]["eps"]
                corrected_second_moment = state["exp_avg_sq"] / (1.0 - beta2**update)
                denominator = torch.sqrt(corrected_second_moment) + epsilon
                effective_rate = learning_rate / denominator
                inverse_sqrt_denominator = torch.rsqrt(denominator)
                preconditioned_hessian = (
                    inverse_sqrt_denominator[:, None]
                    * hessian
                    * inverse_sqrt_denominator[None, :]
                )
                preconditioned_maximum = torch.max(
                    torch.linalg.eigvalsh(preconditioned_hessian)
                )
                minimum_effective_rate = float(torch.min(effective_rate))
                maximum_effective_rate = float(torch.max(effective_rate))
                stability_product = learning_rate * float(preconditioned_maximum)
            else:
                minimum_effective_rate = math.nan
                maximum_effective_rate = math.nan
                stability_product = math.nan
            sorted_probability = torch.sort(trace_probability, descending=True).values
            top_one_percent = torch.sum(sorted_probability[: max(1, paths // 100)])
            kl_divergence = torch.sum(
                trace_probability * (torch.log(trace_probability) - log_base_probability)
            )
            current_oce = -float(trace_log_normalizer)
            trace.append(
                {
                    "update": update,
                    "oce": current_oce,
                    "oce_gap_to_reference": (
                        math.nan if reference_oce is None else reference_oce - current_oce
                    ),
                    "raw_gradient_l2_norm": float(torch.linalg.vector_norm(raw_gradient)),
                    "scaled_gradient_l2_norm": float(
                        torch.linalg.vector_norm(scaled_gradient)
                    ),
                    "gradient_maximum_absolute": float(torch.max(torch.abs(scaled_gradient))),
                    "parameter_l2_norm": float(torch.linalg.vector_norm(optimized_coefficients)),
                    "step_l2_norm": step_norm,
                    "two_cycle_distance": cycle_distance,
                    "consecutive_gradient_cosine": consecutive_gradient_cosine,
                    "radial_oce_derivative": float(trace_probability @ trace_pnl),
                    "newton_decrement_squared": float(newton_decrement_squared),
                    "hessian_condition_number": float(condition_number),
                    "hessian_off_diagonal_fraction": float(
                        torch.linalg.matrix_norm(off_diagonal)
                        / torch.linalg.matrix_norm(hessian)
                    ),
                    "minimum_effective_learning_rate": minimum_effective_rate,
                    "maximum_effective_learning_rate": maximum_effective_rate,
                    "adam_preconditioned_stability_product": stability_product,
                    "effective_sample_size": float(
                        1.0 / torch.sum(torch.square(trace_probability))
                    ),
                    "kl_q_to_base": float(kl_divergence),
                    "maximum_probability": float(torch.max(trace_probability)),
                    "top_one_percent_probability": float(top_one_percent),
                }
            )

    for update in range(1, maximum_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        pnl = optimized_design @ optimized_coefficients
        loss = torch.logsumexp(log_base_probability - pnl, dim=0)
        loss.backward()
        current_gradient = optimized_coefficients.grad.detach().clone()
        gradient_l2_norm = float(torch.linalg.vector_norm(current_gradient))
        current_parameters = optimized_coefficients.detach().clone()
        step_norm = (
            math.nan
            if previous_parameters is None
            else float(torch.linalg.vector_norm(current_parameters - previous_parameters))
        )
        cycle_distance = (
            math.nan
            if two_updates_ago is None
            else float(torch.linalg.vector_norm(current_parameters - two_updates_ago))
        )
        gradient_cosine = (
            math.nan
            if previous_gradient is None
            else float(
                torch.dot(current_gradient, previous_gradient)
                / (
                    torch.linalg.vector_norm(current_gradient)
                    * torch.linalg.vector_norm(previous_gradient)
                )
            )
        )
        if trace_interval and (update == 1 or (update - 1) % trace_interval == 0):
            record_trace(
                update - 1,
                step_norm=step_norm,
                cycle_distance=cycle_distance,
                consecutive_gradient_cosine=gradient_cosine,
            )
        if gradient_l2_norm <= gradient_tolerance:
            converged = True
            break
        optimizer.step()
        updates = update
        two_updates_ago = previous_parameters
        previous_parameters = current_parameters
        previous_gradient = current_gradient
    seconds = time.monotonic() - started

    if trace_interval and (not trace or trace[-1]["update"] != updates):
        final_step_norm = (
            math.nan
            if previous_parameters is None
            else float(
                torch.linalg.vector_norm(
                    optimized_coefficients.detach() - previous_parameters
                )
            )
        )
        final_cycle_distance = (
            math.nan
            if two_updates_ago is None
            else float(
                torch.linalg.vector_norm(
                    optimized_coefficients.detach() - two_updates_ago
                )
            )
        )
        record_trace(
            updates,
            step_norm=final_step_norm,
            cycle_distance=final_cycle_distance,
            consecutive_gradient_cosine=math.nan,
        )

    with torch.no_grad():
        pnl = optimized_design @ optimized_coefficients
        log_probability = log_base_probability - pnl
        log_normalizer = torch.logsumexp(log_probability, dim=0)
        probability = torch.exp(log_probability - log_normalizer)
        gradient = -(optimized_design.T @ probability)
        coefficients = optimized_coefficients / torch.as_tensor(scale, dtype=torch.float64)
        centered_design = optimized_design - probability @ optimized_design
        hessian = centered_design.T @ (probability[:, None] * centered_design)
        eigenvalues = torch.linalg.eigvalsh(hessian)

    density = probability.numpy()
    gradient_rms = float(torch.sqrt(torch.mean(torch.square(gradient))))
    gradient_l2_norm = float(torch.linalg.vector_norm(gradient))
    gradient_maximum = float(torch.max(torch.abs(gradient)))
    positive_eigenvalues = eigenvalues[eigenvalues > torch.max(eigenvalues) * 1.0e-14]
    condition_number = float(torch.max(eigenvalues) / torch.min(positive_eigenvalues))
    return OceResult(
        coefficients=coefficients.numpy(),
        density=density,
        initial_oce=-float(initial_log_normalizer),
        oce=-float(log_normalizer),
        closure_evaluations=updates + 1,
        seconds=seconds,
        gradient_l2_norm=gradient_l2_norm,
        gradient_rms=gradient_rms,
        gradient_maximum_absolute=gradient_maximum,
        effective_sample_size=float(1.0 / np.sum(np.square(density), dtype=np.float64)),
        maximum_log_density_ratio=float(np.max(np.abs(np.log(density / base_probability)))),
        radial_oce_derivative=float(probability @ pnl),
        hessian_condition_number=condition_number,
        converged=converged or gradient_l2_norm <= gradient_tolerance,
        preconditioned=precondition,
        optimizer="adam",
        updates=updates,
        trace=tuple(trace),
    )


def run_experiment(
    config: MarketConfig,
    *,
    adam_trace_interval: int = 0,
) -> dict[str, Any]:
    """Simulate the market and compare raw with preconditioned OCE solves."""

    paths = simulate_market(config)
    design, names = build_trading_design(paths)
    raw = solve_exponential_oce(design, precondition=False)
    preconditioned = solve_exponential_oce(design, precondition=True)
    certificate = solve_exponential_oce(
        design,
        base_density=preconditioned.density,
        initial_coefficients=preconditioned.coefficients,
        precondition=True,
    )
    adam = solve_exponential_oce_adam(
        design,
        precondition=True,
        trace_interval=adam_trace_interval,
        reference_oce=preconditioned.oce,
    )
    adam_certificate = solve_exponential_oce_adam(
        design,
        base_density=preconditioned.density,
        initial_coefficients=preconditioned.coefficients,
        precondition=True,
        trace_interval=adam_trace_interval,
        reference_oce=0.0,
    )
    result = {
        "config": asdict(config),
        "design_columns": names,
        "physical_moments": {
            "day10_state_standard_deviation": float(np.std(paths.v[:, 1])),
            "traded_stock_return": float(np.mean(paths.stock_return[:, 1])),
            "normalized_moneyness": list(config.normalized_moneyness),
            "strikes": paths.strikes.tolist(),
            "traded_option_return": np.mean(paths.option_return[:, 1], axis=0).tolist(),
            "positive_option_payoff_paths": np.sum(
                paths.option_return[:, 1] > -1.0, axis=0
            ).tolist(),
            "day10_implied_volatility": float(np.mean(paths.implied_volatility[:, 1])),
        },
        "raw": raw.summary(),
        "preconditioned": preconditioned.summary(),
        "warm_started_q_resolve": certificate.summary(),
        "adam": adam.summary(),
        "adam_warm_started_q_resolve": adam_certificate.summary(),
        "maximum_density_difference_raw_vs_preconditioned": float(
            np.max(np.abs(raw.density - preconditioned.density))
        ),
    }
    if adam_trace_interval:
        result["adam_trace"] = list(adam.trace)
        result["adam_q_trace"] = list(adam_certificate.trace)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=-0.5)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--adam-trace-interval", type=int, default=0)
    parser.add_argument("--trace-output", type=Path, default=None)
    return parser


def main() -> None:
    """Run the command-line experiment and print its diagnostics as JSON."""

    arguments = _parser().parse_args()
    if arguments.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(arguments.torch_threads)
    result = run_experiment(
        MarketConfig(
            kappa=arguments.kappa,
            xi=arguments.xi,
            rho=arguments.rho,
            paths=arguments.paths,
            seed=arguments.seed,
        ),
        adam_trace_interval=arguments.adam_trace_interval,
    )
    if arguments.trace_output is not None:
        arguments.trace_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.trace_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        for key in ("adam_trace", "adam_q_trace"):
            if key in result:
                result[key] = {
                    "points": len(result[key]),
                    "written_to": str(arguments.trace_output),
                }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
