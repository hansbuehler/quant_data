"""Orthogonal affine-plus-neural policies for the small OU/OCE market."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.oce.simple_ou_oce import (
    MarketConfig,
    MarketPaths,
    build_trading_design,
    simulate_market,
    solve_exponential_oce,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.0", dependencies=[FeedForwardConfig])
@dataclass(frozen=True, kw_only=True)
class OrthogonalPolicyConfig:
    """Geometry and identity of the nonlinear trading policy."""

    label: str
    group: str = "oce/orthogonal-policy"
    user_version: str = "0.1"
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=16,
    )

    def __post_init__(self) -> None:
        group_parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid policy label {self.label!r}",
        )
        verify_inp(
            bool(group_parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
            lambda: f"invalid policy group {self.group!r}",
        )

    @property
    def filename(self) -> str:
        """Return the stable filename body for this policy specification."""

        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


@dataclass(frozen=True)
class NonlinearOceResult:
    """Full-population diagnostics for a nonlinear policy solve."""

    model: OrthogonalResidualPolicy
    action: np.ndarray
    pnl: np.ndarray
    density: np.ndarray
    projection_coefficients: np.ndarray
    initial_oce: float
    oce: float
    initial_gradient_l2_norm: float
    initial_gradient_maximum_absolute: float
    closure_evaluations: int
    updates: int
    seconds: float
    gradient_l2_norm: float
    gradient_maximum_absolute: float
    affine_gradient_l2_norm: float
    residual_gradient_l2_norm: float
    residual_weighted_affine_moment: float
    action_residual_rms: float
    effective_sample_size: float
    minimum_log_density: float
    zero_density_paths: int
    converged: bool

    def summary(self) -> dict[str, float | int | bool]:
        """Return JSON-compatible scalar diagnostics."""

        return {
            "initial_oce": self.initial_oce,
            "oce": self.oce,
            "oce_improvement": self.oce - self.initial_oce,
            "initial_gradient_l2_norm": self.initial_gradient_l2_norm,
            "initial_gradient_maximum_absolute": self.initial_gradient_maximum_absolute,
            "closure_evaluations": self.closure_evaluations,
            "updates": self.updates,
            "seconds": self.seconds,
            "gradient_l2_norm": self.gradient_l2_norm,
            "gradient_maximum_absolute": self.gradient_maximum_absolute,
            "affine_gradient_l2_norm": self.affine_gradient_l2_norm,
            "residual_gradient_l2_norm": self.residual_gradient_l2_norm,
            "residual_weighted_affine_moment": self.residual_weighted_affine_moment,
            "action_residual_rms": self.action_residual_rms,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_size_fraction": self.effective_sample_size / len(self.density),
            "minimum_log_density": self.minimum_log_density,
            "zero_density_paths": self.zero_density_paths,
            "converged": self.converged,
        }


class OrthogonalResidualPolicy(BaseModel):
    """An affine policy plus a differentiably orthogonalized neural residual."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: OrthogonalPolicyConfig,
        instrument_count: int,
    ) -> None:
        super().__init__(env)
        verify_inp(instrument_count > 0, lambda: "instrument count must be positive")
        self.config = config
        self.instrument_count = instrument_count
        self.affine_coefficients = torch.nn.Parameter(
            torch.zeros((2, instrument_count), dtype=env.fdtype, device=env.device)
        )
        self.residual_network = FeedForward(
            env,
            num_input=1,
            num_output=instrument_count,
            config=config.network,
        )

        # A zero final layer makes the initial policy exactly affine while retaining
        # nonzero gradients for the final residual features.
        final = (
            self.residual_network.I0
            if len(self.residual_network.In) == 0
            else self.residual_network.In[-1]
        )
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        normalized_state: torch.Tensor,
        base_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actions and the residual after weighted affine projection."""

        if normalized_state.ndim != 1:
            raise ValueError("normalized state must be a path vector")
        if base_probability.shape != normalized_state.shape:
            raise ValueError("base probability must have one value per state")
        if not bool(
            torch.all(torch.isfinite(base_probability) & (base_probability > 0.0))
        ):
            raise ValueError("base probability must be finite-positive")
        if not bool(torch.isclose(torch.sum(base_probability), base_probability.new_tensor(1.0))):
            raise ValueError("base probability must sum to one")

        affine_basis = torch.stack(
            (torch.ones_like(normalized_state), normalized_state), dim=1
        )
        raw_residual = self.residual_network(normalized_state[:, None])
        weighted_basis = base_probability[:, None] * affine_basis
        gram = affine_basis.T @ weighted_basis
        cross_moment = weighted_basis.T @ raw_residual
        projection_coefficients = torch.linalg.solve(gram, cross_moment)
        residual = raw_residual - affine_basis @ projection_coefficients
        action = affine_basis @ self.affine_coefficients + residual

        self.verify(lambda: torch.isfinite(action), lambda: "policy action is not finite")
        return action, residual

    def action_with_projection(
        self,
        normalized_state: torch.Tensor,
        projection_coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions using projection coefficients fitted on another population."""

        expected_shape = (2, self.instrument_count)
        if projection_coefficients.shape != expected_shape:
            raise ValueError(
                f"projection coefficients have shape {projection_coefficients.shape}, "
                f"expected {expected_shape}"
            )
        affine_basis = torch.stack(
            (torch.ones_like(normalized_state), normalized_state), dim=1
        )
        raw_residual = self.residual_network(normalized_state[:, None])
        residual = raw_residual - affine_basis @ projection_coefficients
        return affine_basis @ self.affine_coefficients + residual, residual


def build_policy_population(paths: MarketPaths) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized day-10 state and day-10-to-20 instrument returns."""

    path_count = len(paths.v)
    state = np.asarray(paths.v[:, 1], dtype=np.float64)
    state_rms = math.sqrt(float(np.mean(np.square(state))))
    if not math.isfinite(state_rms) or state_rms <= 0.0:
        raise ValueError("day-10 state must have positive finite RMS")
    instrument_return = np.column_stack(
        (paths.stock_return[:, 1], paths.option_return[:, 1, :])
    )
    if instrument_return.shape[0] != path_count or np.any(~np.isfinite(instrument_return)):
        raise ValueError("instrument returns have invalid shape or values")
    return state / state_rms, instrument_return


def solve_nonlinear_oce(
    normalized_state: np.ndarray,
    instrument_return: np.ndarray,
    *,
    config: OrthogonalPolicyConfig,
    base_density: np.ndarray | None = None,
    projection_density: np.ndarray | None = None,
    initial_affine_coefficients: np.ndarray | None = None,
    initial_model: OrthogonalResidualPolicy | None = None,
    incremental_from_initial: bool = False,
    maximum_iterations: int = 500,
    gradient_tolerance: float = 1.0e-9,
    seed: int = 17,
) -> NonlinearOceResult:
    """Fit the full-batch exponential OCE with a weighted-orthogonal residual."""

    normalized_state = np.asarray(normalized_state, dtype=np.float64)
    instrument_return = np.asarray(instrument_return, dtype=np.float64)
    if normalized_state.ndim != 1 or len(normalized_state) < 2:
        raise ValueError("normalized state must contain at least two paths")
    if instrument_return.ndim != 2 or instrument_return.shape[0] != len(normalized_state):
        raise ValueError("instrument returns must have one row per state")
    if np.any(~np.isfinite(normalized_state)) or np.any(~np.isfinite(instrument_return)):
        raise ValueError("policy population must be finite")
    verify_inp(maximum_iterations > 0, lambda: "maximum iterations must be positive")
    verify_inp(
        math.isfinite(gradient_tolerance) and gradient_tolerance > 0.0,
        lambda: "gradient tolerance must be finite-positive",
    )

    path_count, instrument_count = instrument_return.shape
    if base_density is None:
        base_probability = np.full(path_count, 1.0 / path_count, dtype=np.float64)
    else:
        base_probability = np.asarray(base_density, dtype=np.float64)
        if base_probability.shape != (path_count,):
            raise ValueError("base density must have one value per path")
        if np.any(~np.isfinite(base_probability) | (base_probability <= 0.0)):
            raise ValueError("base density must be finite-positive")
        base_probability = base_probability / np.sum(base_probability, dtype=np.float64)
    if projection_density is None:
        projection_probability = base_probability
    else:
        projection_probability = np.asarray(projection_density, dtype=np.float64)
        if projection_probability.shape != (path_count,):
            raise ValueError("projection density must have one value per path")
        if np.any(~np.isfinite(projection_probability) | (projection_probability <= 0.0)):
            raise ValueError("projection density must be finite-positive")
        projection_probability = projection_probability / np.sum(
            projection_probability, dtype=np.float64
        )
    if incremental_from_initial and initial_model is None:
        raise ValueError("an incremental solve requires an initial model")

    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    torch.manual_seed(seed)
    if initial_model is None:
        model = OrthogonalResidualPolicy(
            env,
            config=config,
            instrument_count=instrument_count,
        )
    else:
        if initial_model.instrument_count != instrument_count:
            raise ValueError("initial model has the wrong instrument count")
        model = OrthogonalResidualPolicy(
            env,
            config=config,
            instrument_count=instrument_count,
        )
        model.load_state_dict(initial_model.state_dict())
    if initial_affine_coefficients is not None:
        initial_affine_coefficients = np.asarray(initial_affine_coefficients, dtype=np.float64)
        if initial_affine_coefficients.shape != (2, instrument_count):
            raise ValueError("initial affine coefficients have the wrong shape")
        with torch.no_grad():
            model.affine_coefficients.copy_(
                torch.as_tensor(initial_affine_coefficients, dtype=env.fdtype)
            )

    state = torch.as_tensor(normalized_state, dtype=env.fdtype)
    returns = torch.as_tensor(instrument_return, dtype=env.fdtype)
    probability = torch.as_tensor(base_probability, dtype=env.fdtype)
    projection_probability_tensor = torch.as_tensor(
        projection_probability, dtype=env.fdtype
    )
    log_probability = torch.log(probability)

    if incremental_from_initial:
        with torch.no_grad():
            reference_action, reference_residual = model(
                state, projection_probability_tensor
            )
            reference_action = reference_action.detach().clone()
            reference_residual = reference_residual.detach().clone()
    else:
        reference_action = torch.zeros_like(returns)
        reference_residual = torch.zeros_like(returns)

    def loss_and_outputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, residual = model(state, projection_probability_tensor)
        action = action - reference_action
        residual = residual - reference_residual
        pnl = torch.sum(action * returns, dim=1)
        loss = torch.logsumexp(log_probability - pnl, dim=0)
        return loss, action, residual, pnl

    model.zero_grad(set_to_none=True)
    initial_loss, _, _, _ = loss_and_outputs()
    initial_loss.backward()
    initial_gradients = [
        parameter.grad.detach().reshape(-1) for parameter in model.parameters()
    ]
    initial_flat_gradient = torch.cat(initial_gradients)
    initial_gradient_l2_norm = float(
        torch.linalg.vector_norm(initial_flat_gradient)
    )
    initial_gradient_maximum = float(torch.max(torch.abs(initial_flat_gradient)))
    initial_oce = -float(initial_loss.detach())
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=maximum_iterations,
        tolerance_grad=gradient_tolerance,
        tolerance_change=1.0e-14,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = loss_and_outputs()
        loss.backward()
        return loss

    started = time.monotonic()
    optimizer.step(closure)
    seconds = time.monotonic() - started
    first_parameter = next(model.parameters())
    updates = int(optimizer.state[first_parameter].get("n_iter", 0))

    optimizer.zero_grad(set_to_none=True)
    final_loss, action, residual, pnl = loss_and_outputs()
    final_loss.backward()
    gradients = [parameter.grad.detach().reshape(-1) for parameter in model.parameters()]
    flat_gradient = torch.cat(gradients)
    affine_gradient = model.affine_coefficients.grad.detach().reshape(-1)
    residual_gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.residual_network.parameters()
    ]
    residual_gradient = torch.cat(residual_gradients)

    with torch.no_grad():
        tilted_log_probability = log_probability - pnl
        log_normalizer = torch.logsumexp(tilted_log_probability, dim=0)
        normalized_log_probability = tilted_log_probability - log_normalizer
        tilted_probability = torch.exp(normalized_log_probability)
        affine_basis = torch.stack((torch.ones_like(state), state), dim=1)
        orthogonality = affine_basis.T @ (
            projection_probability_tensor[:, None] * residual
        )
        raw_residual = model.residual_network(state[:, None])
        projection_coefficients = torch.linalg.solve(
            affine_basis.T
            @ (projection_probability_tensor[:, None] * affine_basis),
            (projection_probability_tensor[:, None] * affine_basis).T
            @ raw_residual,
        )
        density = tilted_probability.detach().cpu().numpy()

    return NonlinearOceResult(
        model=model,
        action=action.detach().cpu().numpy(),
        pnl=pnl.detach().cpu().numpy(),
        density=density,
        projection_coefficients=projection_coefficients.detach().cpu().numpy(),
        initial_oce=initial_oce,
        oce=-float(log_normalizer.detach()),
        initial_gradient_l2_norm=initial_gradient_l2_norm,
        initial_gradient_maximum_absolute=initial_gradient_maximum,
        closure_evaluations=closure_evaluations,
        updates=updates,
        seconds=seconds,
        gradient_l2_norm=float(torch.linalg.vector_norm(flat_gradient)),
        gradient_maximum_absolute=float(torch.max(torch.abs(flat_gradient))),
        affine_gradient_l2_norm=float(torch.linalg.vector_norm(affine_gradient)),
        residual_gradient_l2_norm=float(torch.linalg.vector_norm(residual_gradient)),
        residual_weighted_affine_moment=float(torch.max(torch.abs(orthogonality))),
        action_residual_rms=float(
            torch.sqrt(torch.mean(torch.square(residual.detach())))
        ),
        effective_sample_size=float(1.0 / np.sum(np.square(density), dtype=np.float64)),
        minimum_log_density=float(torch.min(normalized_log_probability.detach())),
        zero_density_paths=int(np.sum(density == 0.0)),
        converged=float(torch.linalg.vector_norm(flat_gradient.detach()))
        <= gradient_tolerance,
    )


def evaluate_nonlinear_policy(
    result: NonlinearOceResult,
    normalized_state: np.ndarray,
    instrument_return: np.ndarray,
) -> dict[str, float]:
    """Evaluate a fitted policy without refitting its population projection."""

    normalized_state = np.asarray(normalized_state, dtype=np.float64)
    instrument_return = np.asarray(instrument_return, dtype=np.float64)
    if instrument_return.shape != (len(normalized_state), result.action.shape[1]):
        raise ValueError("evaluation returns have the wrong shape")
    state = torch.as_tensor(normalized_state, dtype=result.model.fdtype)
    returns = torch.as_tensor(instrument_return, dtype=result.model.fdtype)
    projection = torch.as_tensor(
        result.projection_coefficients, dtype=result.model.fdtype
    )
    with torch.no_grad():
        action, residual = result.model.action_with_projection(state, projection)
        pnl = torch.sum(action * returns, dim=1)
        oce = -torch.logsumexp(-pnl, dim=0) + math.log(len(pnl))
    return {
        "oce": float(oce),
        "mean_pnl": float(torch.mean(pnl)),
        "pnl_standard_deviation": float(torch.std(pnl, correction=0)),
        "minimum_pnl": float(torch.min(pnl)),
        "maximum_pnl": float(torch.max(pnl)),
        "action_rms": float(torch.sqrt(torch.mean(torch.square(action)))),
        "residual_action_rms": float(torch.sqrt(torch.mean(torch.square(residual)))),
    }


def run_nonlinear_experiment(
    market_config: MarketConfig,
    *,
    policy_config: OrthogonalPolicyConfig,
    maximum_iterations: int = 500,
    q_maximum_iterations: int = 0,
) -> dict[str, Any]:
    """Compare affine and affine-plus-orthogonal-residual OCE policies."""

    paths = simulate_market(market_config)
    affine_design, _ = build_trading_design(paths)
    state, instrument_return = build_policy_population(paths)
    affine = solve_exponential_oce(affine_design, precondition=True)
    affine_coefficients = affine.coefficients.reshape((-1, 2)).T
    nonlinear = solve_nonlinear_oce(
        state,
        instrument_return,
        config=policy_config,
        initial_affine_coefficients=affine_coefficients,
        maximum_iterations=maximum_iterations,
    )
    validation_paths = simulate_market(
        MarketConfig(
            **{
                **asdict(market_config),
                "seed": market_config.seed + 1,
            }
        )
    )
    training_state_rms = math.sqrt(float(np.mean(np.square(paths.v[:, 1]))))
    validation_state = validation_paths.v[:, 1] / training_state_rms
    validation_return = np.column_stack(
        (
            validation_paths.stock_return[:, 1],
            validation_paths.option_return[:, 1, :],
        )
    )
    validation = evaluate_nonlinear_policy(
        nonlinear,
        validation_state,
        validation_return,
    )
    validation_basis = np.column_stack(
        (np.ones_like(validation_state), validation_state)
    )
    validation_affine_action = validation_basis @ affine_coefficients
    validation_affine_pnl = np.sum(
        validation_affine_action * validation_return, axis=1
    )
    validation_affine_pnl_tensor = torch.as_tensor(
        validation_affine_pnl, dtype=torch.float64
    )
    affine_validation = {
        "oce": float(
            -torch.logsumexp(-validation_affine_pnl_tensor, dim=0)
            + math.log(len(validation_affine_pnl_tensor))
        ),
        "mean_pnl": float(np.mean(validation_affine_pnl)),
        "pnl_standard_deviation": float(np.std(validation_affine_pnl)),
        "minimum_pnl": float(np.min(validation_affine_pnl)),
        "maximum_pnl": float(np.max(validation_affine_pnl)),
        "action_rms": float(np.sqrt(np.mean(np.square(validation_affine_action)))),
    }

    # This is the exact finite-sample OCE measure generated by the nonlinear
    # optimum. Its expectation of every policy Jacobian direction is the final
    # P-gradient above, which is the relevant local risk-neutrality certificate.
    q_affine_design = np.column_stack(
        tuple(
            feature * instrument_return[:, instrument]
            for instrument in range(instrument_return.shape[1])
            for feature in (np.ones_like(state), state)
        )
    )
    if np.all(nonlinear.density > 0.0):
        q_affine_certificate: dict[str, Any] = solve_exponential_oce(
            q_affine_design,
            base_density=nonlinear.density,
            precondition=True,
        ).summary()
    else:
        q_affine_certificate = {
            "available": False,
            "reason": (
                "The fitted log-density exceeded the float64 dynamic range. "
                "Clipping would change the OCE measure, so the Q re-solve was skipped."
            ),
        }
    result = {
        "market_config": asdict(market_config),
        "policy_config": {
            "label": policy_config.label,
            "group": policy_config.group,
            "user_version": policy_config.user_version,
            "network": asdict(policy_config.network),
        },
        "affine": affine.summary(),
        "affine_validation": affine_validation,
        "nonlinear": nonlinear.summary(),
        "nonlinear_validation": validation,
        "nonlinear_minus_affine_oce": nonlinear.oce - affine.oce,
        "q_affine_resolve": q_affine_certificate,
        "interpretation": (
            "The nonlinear P-gradient is the Q expectation of the fitted policy's "
            "Jacobian, up to sign. The separate Q affine re-solve checks the stable "
            "affine subspace; a fresh neural network would generally have a different "
            "tangent space and is not the same certificate."
        ),
    }
    if q_maximum_iterations > 0:
        nonlinear_q = solve_nonlinear_oce(
            state,
            instrument_return,
            config=policy_config,
            base_density=nonlinear.density,
            projection_density=np.ones(len(state), dtype=np.float64),
            initial_model=nonlinear.model,
            incremental_from_initial=True,
            maximum_iterations=q_maximum_iterations,
        )
        result["q_full_nonlinear_increment_resolve"] = nonlinear_q.summary()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--maximum-iterations", type=int, default=500)
    parser.add_argument("--q-maximum-iterations", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main() -> None:
    """Run the nonlinear residual comparison and print JSON diagnostics."""

    arguments = _parser().parse_args()
    torch.set_num_threads(arguments.torch_threads)
    market_config = MarketConfig(paths=arguments.paths, seed=arguments.seed)
    policy_config = OrthogonalPolicyConfig(
        label="simple-ou",
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=arguments.depth,
            num_width=arguments.width,
        ),
    )
    result = run_nonlinear_experiment(
        market_config,
        policy_config=policy_config,
        maximum_iterations=arguments.maximum_iterations,
        q_maximum_iterations=arguments.q_maximum_iterations,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
