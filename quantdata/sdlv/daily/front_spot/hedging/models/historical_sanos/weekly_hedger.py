"""Open-loop multistep historical SANOS policy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyIndifferencePrices,
    WeeklyRiskObjective,
    WeeklyScenarioBatch,
    weekly_certainty_equivalent,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyOpenLoopFit:
    """One signed-claim open-loop schedule and diagnostics."""

    spot_positions: FloatArray
    option_positions: FloatArray
    cash_translation: float | None
    training_loss: float
    validation_loss: float
    certainty_equivalent: float
    iterations: int
    converged: bool
    maximum_absolute_gradient: float
    cap_hits: int


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyHedgeFit:
    """Pricing policies and the cost-consistent pure hedge schedule."""

    spot_positions: FloatArray
    option_positions: FloatArray
    long_policy: HistoricalWeeklyOpenLoopFit
    short_policy: HistoricalWeeklyOpenLoopFit
    zero_policy: HistoricalWeeklyOpenLoopFit
    pure_policy: HistoricalWeeklyOpenLoopFit
    prices: WeeklyIndifferencePrices
    converged: bool
    cap_hits: int


def fit_historical_weekly_hedge(
    scenario_set: HistoricalWeeklyScenarioSet,
    *,
    claim_quantity: float,
    initial_spot_position: float,
    initial_option_positions: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> HistoricalWeeklyHedgeFit:
    """Fit pricing policies and a cost-consistent pure hedge schedule."""
    return fit_weekly_open_loop_hedge(
        scenario_set.scenarios,
        claim_quantity=claim_quantity,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=config,
    )


def fit_weekly_open_loop_hedge(
    scenarios: WeeklyScenarioBatch,
    *,
    claim_quantity: float,
    initial_spot_position: float,
    initial_option_positions: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> HistoricalWeeklyHedgeFit:
    """Fit pricing policies and a drift-neutral pure hedge on weekly paths."""
    if not math.isfinite(claim_quantity) or claim_quantity <= 0.0:
        raise ValueError("weekly claim quantity must be finite-positive")
    price_geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(scenarios.contract_count),
        config=config,
    )
    long_policy = price_geometry.fit_policy(
        signed_claim_quantity=claim_quantity,
    )
    short_policy = price_geometry.fit_policy(
        signed_claim_quantity=-claim_quantity,
    )
    zero_policy = price_geometry.fit_policy(signed_claim_quantity=0.0)
    execution_geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=config,
        center_instrument_gains=True,
    )
    pure_policy = execution_geometry.fit_policy(
        signed_claim_quantity=claim_quantity,
    )
    prices = WeeklyIndifferencePrices(
        buyer=long_policy.certainty_equivalent - zero_policy.certainty_equivalent,
        seller=zero_policy.certainty_equivalent - short_policy.certainty_equivalent,
    )
    spot_positions = pure_policy.spot_positions
    option_positions = pure_policy.option_positions
    cap_hits = int(
        np.count_nonzero(
            np.abs(
                np.concatenate((spot_positions[:, None], option_positions), axis=1)
            )
            >= 0.999 * config.position_cap
        )
    )
    return HistoricalWeeklyHedgeFit(
        spot_positions=spot_positions,
        option_positions=option_positions,
        long_policy=long_policy,
        short_policy=short_policy,
        zero_policy=zero_policy,
        pure_policy=pure_policy,
        prices=prices,
        converged=(
            long_policy.converged
            and short_policy.converged
            and zero_policy.converged
            and pure_policy.converged
            and cap_hits == 0
        ),
        cap_hits=(
            cap_hits
            + long_policy.cap_hits
            + short_policy.cap_hits
            + zero_policy.cap_hits
            + pure_policy.cap_hits
        ),
    )


class _WeeklyOpenLoopGeometry:
    def __init__(
        self,
        scenarios: WeeklyScenarioBatch | HistoricalWeeklyScenarioSet,
        *,
        initial_spot_position: float,
        initial_option_positions: NDArray[np.floating],
        config: HistoricalSanosConfig,
        center_instrument_gains: bool = False,
    ) -> None:
        self.scenarios = (
            scenarios.scenarios
            if isinstance(scenarios, HistoricalWeeklyScenarioSet)
            else scenarios
        )
        self.initial_spot = float(initial_spot_position)
        self.initial_options = np.asarray(initial_option_positions, dtype=np.float64)
        self.config = config
        self.center_instrument_gains = center_instrument_gains
        if not math.isfinite(self.initial_spot):
            raise ValueError("historical weekly initial spot position must be finite")
        if self.initial_options.shape != (self.scenarios.contract_count,) or not np.all(
            np.isfinite(self.initial_options)
        ):
            raise ValueError("historical weekly initial option positions do not align")
        if np.any(
            np.abs(self.initial_options[~self.scenarios.option_alive[0]]) > 1.0e-14
        ):
            raise ValueError("expired historical weekly options cannot retain positions")
        self.option_parameter_pairs = tuple(
            (step, contract)
            for step in range(self.scenarios.step_count)
            for contract in np.flatnonzero(self.scenarios.option_alive[step])
        )
        self.parameter_count = self.scenarios.step_count + len(
            self.option_parameter_pairs
        )
        self._option_parameter_index = {
            pair: self.scenarios.step_count + index
            for index, pair in enumerate(self.option_parameter_pairs)
        }
        self._spot_gain_means = np.mean(
            self.scenarios.spot_step_gains,
            axis=0,
        )
        self._option_gain_means = np.mean(
            self.scenarios.option_step_gains,
            axis=0,
        )

    def fit_policy(self, *, signed_claim_quantity: float) -> HistoricalWeeklyOpenLoopFit:
        observation_count = self.scenarios.scenario_count
        validation_count = max(
            1,
            int(round(observation_count * self.config.validation_fraction)),
        )
        split = observation_count - validation_count
        if split < 2:
            raise ValueError("historical weekly training split has fewer than two rows")
        include_translation = (
            self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
        )
        initial = np.zeros(self.parameter_count + int(include_translation))
        bounds = [
            (-self.config.position_cap, self.config.position_cap)
        ] * self.parameter_count
        if include_translation:
            bounds.append((None, None))
        selection = self._minimize(
            initial,
            np.arange(split),
            signed_claim_quantity=signed_claim_quantity,
            include_translation=include_translation,
            bounds=bounds,
        )
        validation_loss = self._objective_value(
            selection.x,
            np.arange(split, observation_count),
            signed_claim_quantity=signed_claim_quantity,
            include_translation=include_translation,
        )
        final = self._minimize(
            np.asarray(selection.x, dtype=np.float64),
            np.arange(observation_count),
            signed_claim_quantity=signed_claim_quantity,
            include_translation=include_translation,
            bounds=bounds,
        )
        training_loss = self._objective_value(
            final.x,
            np.arange(observation_count),
            signed_claim_quantity=signed_claim_quantity,
            include_translation=include_translation,
        )
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            optimality = float(getattr(final, "optimality", math.inf))
        else:
            _, gradient = self._loss_and_gradient(
                final.x,
                np.arange(observation_count),
                signed_claim_quantity=signed_claim_quantity,
                include_translation=include_translation,
            )
            optimality = float(np.max(np.abs(gradient)))
        spot, options = self.unpack(final.x[: self.parameter_count])
        certainty_equivalent = self._policy_certainty_equivalent(
            final.x,
            signed_claim_quantity=signed_claim_quantity,
            include_translation=include_translation,
        )
        positions = np.concatenate((spot[:, None], options), axis=1)
        cap_hits = int(
            np.count_nonzero(
                np.abs(positions) >= 0.999 * self.config.position_cap
            )
        )
        return HistoricalWeeklyOpenLoopFit(
            spot_positions=spot,
            option_positions=options,
            cash_translation=(
                float(final.x[-1]) if include_translation else None
            ),
            training_loss=float(training_loss),
            validation_loss=float(validation_loss),
            certainty_equivalent=certainty_equivalent,
            iterations=int(selection.nit) + int(final.nit),
            converged=bool(selection.success and final.success and cap_hits == 0),
            maximum_absolute_gradient=optimality,
            cap_hits=cap_hits,
        )

    def _policy_certainty_equivalent(
        self,
        parameters: NDArray[np.floating],
        *,
        signed_claim_quantity: float,
        include_translation: bool,
    ) -> float:
        indices = np.arange(self.scenarios.scenario_count, dtype=np.int64)
        values = np.asarray(parameters, dtype=np.float64)
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            claim, instrument_gains = self._linear_wealth_geometry(
                indices,
                signed_claim_quantity=signed_claim_quantity,
            )
            turnover_matrix, turnover_offset, turnover_weights = (
                self._turnover_geometry(indices)
            )
            positions = values[: self.parameter_count]
            deterministic_cost = float(
                turnover_weights
                @ np.abs(turnover_matrix @ positions - turnover_offset)
            )
            wealth = claim + instrument_gains @ positions - deterministic_cost
        else:
            spot, options = self.unpack(values[: self.parameter_count])
            wealth, _ = self._smooth_wealth_and_jacobian(
                spot,
                options,
                indices=indices,
                signed_claim_quantity=signed_claim_quantity,
            )
            if include_translation and len(values) != self.parameter_count + 1:
                raise ValueError("weekly translated policy has the wrong shape")
        return _weekly_certainty_equivalent(wealth, config=self.config)

    def _minimize(
        self,
        initial: FloatArray,
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        include_translation: bool,
        bounds: list[tuple[float | None, float | None]],
    ) -> OptimizeResult:
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            return self._minimize_convex_cost_lasso(
                initial,
                np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
            )
        return minimize(
            lambda parameters: self._loss_and_gradient(
                parameters,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                include_translation=include_translation,
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.config.maximum_iterations,
                "gtol": self.config.gradient_tolerance,
                "ftol": 1.0e-14,
            },
        )

    def _minimize_convex_cost_lasso(
        self,
        initial: FloatArray,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
    ) -> OptimizeResult:
        claim, instrument_gains = self._linear_wealth_geometry(
            indices,
            signed_claim_quantity=signed_claim_quantity,
        )
        turnover_matrix, turnover_offset, turnover_weights = (
            self._turnover_geometry(indices)
        )
        positions = cp.Variable(self.parameter_count)
        wealth_before_cost = claim + instrument_gains @ positions
        centered = wealth_before_cost - cp.sum(wealth_before_cost) / len(indices)
        cost = turnover_weights @ cp.abs(
            turnover_matrix @ positions - turnover_offset
        )
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            risk_penalty = (
                0.5
                * self.config.risk_parameter
                * cp.sum_squares(centered)
                / len(indices)
            )
        else:
            risk_penalty = (
                self.config.risk_parameter
                * cp.norm(centered, 2)
                / math.sqrt(len(indices))
            )
        ridge = 0.5 * self.config.quadratic_ridge * cp.sum_squares(positions)
        problem = cp.Problem(
            cp.Minimize(
                -cp.sum(wealth_before_cost) / len(indices)
                + risk_penalty
                + cost
                + ridge
            ),
            [
                positions >= -self.config.position_cap,
                positions <= self.config.position_cap,
            ],
        )
        positions.value = np.asarray(
            initial[: self.parameter_count],
            dtype=np.float64,
        )
        try:
            problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                max_iter=self.config.maximum_iterations,
                tol_gap_abs=self.config.gradient_tolerance,
                tol_gap_rel=self.config.gradient_tolerance,
                tol_feas=self.config.gradient_tolerance,
            )
        except cp.error.SolverError as error:
            return OptimizeResult(
                x=np.asarray(initial[: self.parameter_count], dtype=np.float64),
                nit=0,
                success=False,
                optimality=math.inf,
                message=f"CLARABEL failed: {error}",
            )
        fitted = (
            np.asarray(initial[: self.parameter_count], dtype=np.float64)
            if positions.value is None
            else np.asarray(positions.value, dtype=np.float64)
        )
        fitted = np.clip(
            fitted,
            -self.config.position_cap,
            self.config.position_cap,
        )
        success = problem.status == cp.OPTIMAL
        return OptimizeResult(
            x=fitted,
            nit=int(problem.solver_stats.num_iters or 0),
            success=success,
            optimality=0.0 if success else math.inf,
            message=problem.status,
            fun=float(problem.value) if problem.value is not None else math.inf,
        )

    def _objective_value(
        self,
        parameters: NDArray[np.floating],
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        include_translation: bool,
    ) -> float:
        if self.config.objective not in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            return self._loss_and_gradient(
                parameters,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                include_translation=include_translation,
            )[0]
        values = np.asarray(parameters, dtype=np.float64)
        claim, instrument_gains = self._linear_wealth_geometry(
            np.asarray(indices, dtype=np.int64),
            signed_claim_quantity=signed_claim_quantity,
        )
        turnover_matrix, turnover_offset, turnover_weights = (
            self._turnover_geometry(np.asarray(indices, dtype=np.int64))
        )
        positions = values[: self.parameter_count]
        wealth = claim + instrument_gains @ positions
        deterministic_cost = float(
            turnover_weights
            @ np.abs(turnover_matrix @ positions - turnover_offset)
        )
        centered = wealth - np.mean(wealth)
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            risk_penalty = (
                0.5
                * self.config.risk_parameter
                * float(np.mean(centered**2))
            )
        else:
            risk_penalty = self.config.risk_parameter * float(
                np.sqrt(np.mean(centered**2))
            )
        ridge = 0.5 * self.config.quadratic_ridge * float(positions @ positions)
        return -float(np.mean(wealth)) + risk_penalty + deterministic_cost + ridge

    def _linear_wealth_geometry(
        self,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
    ) -> tuple[FloatArray, FloatArray]:
        scenarios = self.scenarios
        gains = np.zeros((len(indices), self.parameter_count), dtype=np.float64)
        for step in range(scenarios.step_count):
            gains[:, step] = scenarios.spot_step_gains[indices, step]
            if self.center_instrument_gains:
                gains[:, step] -= self._spot_gain_means[step]
        for pair, parameter_index in self._option_parameter_index.items():
            step, contract = pair
            gains[:, parameter_index] = scenarios.option_step_gains[
                indices,
                step,
                contract,
            ]
            if self.center_instrument_gains:
                gains[:, parameter_index] -= self._option_gain_means[step, contract]
        claim = (
            signed_claim_quantity * scenarios.terminal_claim_values_pv[indices]
        )
        return np.asarray(claim, dtype=np.float64), gains

    def _turnover_geometry(
        self,
        indices: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        scenarios = self.scenarios
        rows: list[FloatArray] = []
        offsets: list[float] = []
        weights: list[float] = []

        for step in range(scenarios.step_count):
            row = np.zeros(self.parameter_count, dtype=np.float64)
            row[step] = 1.0
            if step == 0:
                offset = self.initial_spot
            else:
                row[step - 1] = -1.0
                offset = 0.0
            rows.append(row)
            offsets.append(offset)
            weights.append(
                float(np.mean(scenarios.spot_values_pv[indices, step])) * 1.0e-4
            )

            for contract in np.flatnonzero(scenarios.option_alive[step]):
                parameter_index = self._option_parameter_index[(step, int(contract))]
                row = np.zeros(self.parameter_count, dtype=np.float64)
                row[parameter_index] = 1.0
                if step == 0:
                    offset = float(self.initial_options[contract])
                else:
                    previous_index = self._option_parameter_index[
                        (step - 1, int(contract))
                    ]
                    row[previous_index] = -1.0
                    offset = 0.0
                rows.append(row)
                offsets.append(offset)
                weights.append(
                    float(
                        np.mean(
                            scenarios.option_half_spreads_pv[
                                indices,
                                step,
                                contract,
                            ]
                        )
                    )
                )

        terminal_spot = np.zeros(self.parameter_count, dtype=np.float64)
        terminal_spot[scenarios.step_count - 1] = 1.0
        rows.append(terminal_spot)
        offsets.append(0.0)
        weights.append(
            float(np.mean(scenarios.spot_values_pv[indices, -1])) * 1.0e-4
        )
        for contract in np.flatnonzero(scenarios.option_alive[-1]):
            row = np.zeros(self.parameter_count, dtype=np.float64)
            row[
                self._option_parameter_index[
                    (scenarios.step_count - 1, int(contract))
                ]
            ] = 1.0
            rows.append(row)
            offsets.append(0.0)
            weights.append(
                float(
                    np.mean(
                        scenarios.option_half_spreads_pv[
                            indices,
                            -1,
                            contract,
                        ]
                    )
                )
            )
        return (
            np.stack(rows),
            np.asarray(offsets, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )

    def _loss_and_gradient(
        self,
        parameters: NDArray[np.floating],
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        include_translation: bool,
    ) -> tuple[float, FloatArray]:
        values = np.asarray(parameters, dtype=np.float64)
        spot, options = self.unpack(values[: self.parameter_count])
        wealth, jacobian = self._smooth_wealth_and_jacobian(
            spot,
            options,
            indices=np.asarray(indices, dtype=np.int64),
            signed_claim_quantity=signed_claim_quantity,
        )
        sample_count = len(wealth)
        risk = self.config.risk_parameter
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            centered = wealth - np.mean(wealth)
            loss = -np.mean(wealth) + 0.5 * risk * np.mean(centered**2)
            path_gradient = (-1.0 + risk * centered) / sample_count
        elif self.config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
            centered = wealth - np.mean(wealth)
            standard_deviation = math.sqrt(
                float(np.mean(centered**2)) + np.finfo(np.float64).eps
            )
            loss = -np.mean(wealth) + risk * standard_deviation
            path_gradient = (
                -1.0 + risk * centered / standard_deviation
            ) / sample_count
        elif self.config.objective is HistoricalSanosObjective.ENTROPY:
            exponent = -risk * wealth
            loss = (
                logsumexp(exponent) - math.log(sample_count)
            ) / risk
            path_gradient = -np.exp(exponent - logsumexp(exponent))
        elif self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
            if not include_translation or len(values) != self.parameter_count + 1:
                raise ValueError("weekly power-exponential policy requires translation")
            translation = float(values[-1])
            residual = wealth - translation
            utility = np.where(
                residual >= 0.0,
                -np.expm1(-risk * np.maximum(residual, 0.0)) / risk,
                residual - 0.5 * risk * residual**2,
            )
            marginal = np.where(
                residual >= 0.0,
                np.exp(-risk * np.maximum(residual, 0.0)),
                1.0 - risk * residual,
            )
            loss = -(translation + float(np.mean(utility)))
            path_gradient = -marginal / sample_count
        else:
            raise ValueError(
                f"unsupported historical weekly objective {self.config.objective!r}"
            )
        gradient = jacobian.T @ path_gradient
        if include_translation:
            translation_gradient = -1.0 - float(np.sum(path_gradient))
            gradient = np.concatenate((gradient, [translation_gradient]))
        if not math.isfinite(float(loss)) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("historical weekly objective is non-finite")
        return float(loss), np.asarray(gradient, dtype=np.float64)

    def _smooth_wealth_and_jacobian(
        self,
        spot: FloatArray,
        options: FloatArray,
        *,
        indices: NDArray[np.int64],
        signed_claim_quantity: float,
    ) -> tuple[FloatArray, FloatArray]:
        scenarios = self.scenarios
        count = len(indices)
        wealth = (
            signed_claim_quantity * scenarios.terminal_claim_values_pv[indices]
        ).copy()
        jacobian = np.zeros((count, self.parameter_count), dtype=np.float64)
        epsilon = self.config.cost_smoothing
        previous_spot = self.initial_spot
        for step in range(scenarios.step_count):
            spot_index = step
            gains = scenarios.spot_step_gains[indices, step]
            if self.center_instrument_gains:
                gains = gains - self._spot_gain_means[step]
            wealth += spot[step] * gains
            jacobian[:, spot_index] += gains
            weight = scenarios.spot_values_pv[indices, step] * 1.0e-4
            difference = spot[step] - previous_spot
            root = math.sqrt(difference**2 + epsilon**2)
            wealth -= weight * (root - epsilon)
            jacobian[:, spot_index] -= weight * difference / root
            if step > 0:
                jacobian[:, step - 1] += weight * difference / root
            previous_spot = spot[step]

            for contract in np.flatnonzero(scenarios.option_alive[step]):
                pair = (step, int(contract))
                parameter_index = self._option_parameter_index[pair]
                gains = scenarios.option_step_gains[indices, step, contract]
                if self.center_instrument_gains:
                    gains = gains - self._option_gain_means[step, contract]
                wealth += options[step, contract] * gains
                jacobian[:, parameter_index] += gains
                previous = (
                    self.initial_options[contract]
                    if step == 0
                    else options[step - 1, contract]
                )
                difference = options[step, contract] - previous
                root = math.sqrt(difference**2 + epsilon**2)
                weight = scenarios.option_half_spreads_pv[
                    indices,
                    step,
                    contract,
                ]
                wealth -= weight * (root - epsilon)
                jacobian[:, parameter_index] -= weight * difference / root
                if step > 0:
                    previous_index = self._option_parameter_index[(step - 1, int(contract))]
                    jacobian[:, previous_index] += weight * difference / root

        terminal_spot_weight = scenarios.spot_values_pv[indices, -1] * 1.0e-4
        terminal_spot_root = math.sqrt(spot[-1] ** 2 + epsilon**2)
        wealth -= terminal_spot_weight * (terminal_spot_root - epsilon)
        jacobian[:, scenarios.step_count - 1] -= (
            terminal_spot_weight * spot[-1] / terminal_spot_root
        )
        for contract in np.flatnonzero(scenarios.option_alive[-1]):
            parameter_index = self._option_parameter_index[
                (scenarios.step_count - 1, int(contract))
            ]
            terminal = options[-1, contract]
            root = math.sqrt(terminal**2 + epsilon**2)
            weight = scenarios.option_half_spreads_pv[indices, -1, contract]
            wealth -= weight * (root - epsilon)
            jacobian[:, parameter_index] -= weight * terminal / root
        return wealth, jacobian

    def unpack(self, parameters: NDArray[np.floating]) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.parameter_count,):
            raise ValueError("historical weekly parameter vector has wrong shape")
        spot = values[: self.scenarios.step_count].copy()
        options = np.zeros(
            (self.scenarios.step_count, self.scenarios.contract_count),
            dtype=np.float64,
        )
        for index, pair in enumerate(self.option_parameter_pairs):
            options[pair] = values[self.scenarios.step_count + index]
        return spot, options


def _weekly_objective(objective: HistoricalSanosObjective) -> WeeklyRiskObjective:
    return WeeklyRiskObjective(objective.value)


def _weekly_certainty_equivalent(
    wealth: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> float:
    return weekly_certainty_equivalent(
        wealth,
        objective=_weekly_objective(config.objective),
        risk_parameter=config.risk_parameter,
    )
