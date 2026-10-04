"""Batched optimizer comparison for the orthogonal nonlinear OCE policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.stats import chi2
from torch.func import functional_call, jacfwd

from quantdata.oce.nonlinear_residual_oce import (
    OrthogonalPolicyConfig,
    OrthogonalResidualPolicy,
    build_policy_population,
)
from quantdata.oce.simple_ou_oce import (
    MarketConfig,
    build_trading_design,
    simulate_market,
    solve_exponential_oce,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class BatchedTrainConfig:
    """Training controls shared by every optimizer comparison."""

    batch_size: int = 1_024
    epochs: int = 100
    patience: int = 15
    minimum_learning_rate: float = 1.0e-6
    validation_gradient_check_interval: int = 5
    validation_gradient_alpha: float = 0.05
    validation_gradient_consecutive_checks: int = 2
    seed: int = 71

    def __post_init__(self) -> None:
        verify_inp(self.batch_size > 1, lambda: "batch size must exceed one")
        verify_inp(self.epochs > 0, lambda: "epochs must be positive")
        verify_inp(self.patience > 0, lambda: "patience must be positive")
        verify_inp(
            self.validation_gradient_check_interval > 0,
            lambda: "validation gradient check interval must be positive",
        )
        verify_inp(
            0.0 < self.validation_gradient_alpha < 1.0,
            lambda: "validation gradient alpha must lie in (0,1)",
        )
        verify_inp(
            self.validation_gradient_consecutive_checks > 0,
            lambda: "validation consecutive checks must be positive",
        )
        verify_inp(
            self.minimum_learning_rate > 0.0,
            lambda: "minimum learning rate must be positive",
        )


@dataclass(frozen=True)
class BatchedOptimizerResult:
    """Best-validation checkpoint and untouched-test diagnostics."""

    optimizer: str
    stop_epoch: int
    stopped_for_validation_stationarity: bool
    updates: int
    seconds: float
    training_oce: float
    validation_oce: float
    test_oce: float
    final_training_oce: float
    final_validation_oce: float
    training_gradient_l2_norm: float
    validation_gradient_l2_norm: float
    validation_gradient_wald_statistic: float
    validation_gradient_degrees_of_freedom: int
    validation_gradient_p_value: float
    validation_effective_sample_size: float
    residual_rms: float
    final_learning_rate: float

    def summary(self) -> dict[str, float | int | str]:
        """Return JSON-compatible comparison metrics."""

        return asdict(self)


class Yogi(torch.optim.Optimizer):
    """Yogi with bias correction and the original signed second-moment update."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1.0e-8,
    ) -> None:
        verify_inp(lr > 0.0, lambda: "Yogi learning rate must be positive")
        super().__init__(
            parameters,
            defaults={"lr": lr, "betas": betas, "epsilon": epsilon},
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply one Yogi update."""

        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                first = state.setdefault("first_moment", torch.zeros_like(parameter))
                second = state.setdefault("second_moment", torch.zeros_like(parameter))
                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                gradient_square = gradient.square()
                second.addcmul_(
                    torch.sign(gradient_square - second),
                    gradient_square,
                    value=1.0 - beta2,
                )
                first_hat = first / (1.0 - beta1**step)
                second_hat = second / (1.0 - beta2**step)
                parameter.addcdiv_(
                    first_hat,
                    torch.sqrt(torch.clamp_min(second_hat, 0.0)).add(group["epsilon"]),
                    value=-group["lr"],
                )
        return loss


def _symmetric_inverse_power(
    matrix: torch.Tensor,
    *,
    exponent: float,
    damping: float,
) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    powered = torch.clamp_min(eigenvalues, 0.0).add(damping).pow(exponent)
    return (eigenvectors * powered[None]) @ eigenvectors.T


class Shampoo(torch.optim.Optimizer):
    """Small-matrix Shampoo with diagonal AdaGrad norm grafting."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        beta2: float = 0.9,
        damping: float = 1.0e-4,
        maximum_parameter_step: float = 0.05,
    ) -> None:
        verify_inp(lr > 0.0, lambda: "Shampoo learning rate must be positive")
        super().__init__(
            parameters,
            defaults={
                "lr": lr,
                "beta2": beta2,
                "damping": damping,
                "maximum_parameter_step": maximum_parameter_step,
            },
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply one matrix-preconditioned stochastic update."""

        loss = closure() if closure is not None else None
        for group in self.param_groups:
            directions: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                diagonal = state.setdefault("diagonal", torch.zeros_like(parameter))
                diagonal.mul_(group["beta2"]).addcmul_(
                    gradient, gradient, value=1.0 - group["beta2"]
                )
                graft = gradient / torch.sqrt(diagonal + group["damping"])
                if gradient.ndim == 2:
                    rows, columns = gradient.shape
                    left = state.setdefault(
                        "left",
                        torch.zeros((rows, rows), dtype=gradient.dtype),
                    )
                    right = state.setdefault(
                        "right",
                        torch.zeros((columns, columns), dtype=gradient.dtype),
                    )
                    left.mul_(group["beta2"]).addmm_(
                        gradient,
                        gradient.T,
                        alpha=(1.0 - group["beta2"]) / columns,
                    )
                    right.mul_(group["beta2"]).addmm_(
                        gradient.T,
                        gradient,
                        alpha=(1.0 - group["beta2"]) / rows,
                    )
                    direction = (
                        _symmetric_inverse_power(
                            left, exponent=-0.25, damping=group["damping"]
                        )
                        @ gradient
                        @ _symmetric_inverse_power(
                            right, exponent=-0.25, damping=group["damping"]
                        )
                    )
                    direction_norm = torch.linalg.vector_norm(direction)
                    if float(direction_norm) > 0.0:
                        direction *= torch.linalg.vector_norm(graft) / direction_norm
                else:
                    direction = graft
                directions.append((parameter, direction))
            total_norm = math.sqrt(
                sum(float(torch.sum(direction.square())) for _, direction in directions)
            )
            scale = min(
                1.0,
                group["maximum_parameter_step"]
                / max(group["lr"] * total_norm, torch.finfo(torch.float64).tiny),
            )
            for parameter, direction in directions:
                parameter.add_(direction, alpha=-group["lr"] * scale)
        return loss


class _FixedProjectionWrapper(torch.nn.Module):
    def __init__(
        self,
        policy: OrthogonalResidualPolicy,
        projection: torch.Tensor,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer("projection", projection.detach().clone())

    def forward(self, state: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        action, _ = self.policy.action_with_projection(state, self.projection)
        return torch.sum(action * returns, dim=1)


def _projection_coefficients(
    model: OrthogonalResidualPolicy,
    state: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        basis = torch.stack((torch.ones_like(state), state), dim=1)
        raw = model.residual_network(state[:, None])
        return torch.linalg.solve(basis.T @ basis, basis.T @ raw)


def _oce_from_pnl(pnl: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(-pnl, dim=0) + math.log(len(pnl))


def _evaluate(
    model: OrthogonalResidualPolicy,
    projection: torch.Tensor,
    state: torch.Tensor,
    returns: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        action, residual = model.action_with_projection(state, projection)
        pnl = torch.sum(action * returns, dim=1)
        return float(_oce_from_pnl(pnl)), float(torch.sqrt(torch.mean(residual.square())))


def _gauss_newton_step(
    model: OrthogonalResidualPolicy,
    projection: torch.Tensor,
    state: torch.Tensor,
    returns: torch.Tensor,
    *,
    damping: float,
    maximum_parameter_step: float,
) -> float:
    wrapper = _FixedProjectionWrapper(model, projection)
    parameters = dict(wrapper.named_parameters())

    def path_pnl(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(wrapper, values, (state, returns))

    with torch.no_grad():
        pnl = path_pnl(parameters)
        probability = torch.softmax(-pnl, dim=0)
    jacobian_tree = jacfwd(path_pnl)(parameters)
    jacobian = torch.cat(
        [value.reshape(len(state), -1) for value in jacobian_tree.values()], dim=1
    ).detach()
    centered = jacobian - probability @ jacobian
    gradient = -(jacobian.T @ probability)
    curvature = centered.T @ (probability[:, None] * centered)
    identity = torch.eye(curvature.shape[0], dtype=curvature.dtype)
    original = {name: value.detach().clone() for name, value in parameters.items()}
    current_loss = -_oce_from_pnl(pnl)
    for _ in range(5):
        direction = torch.linalg.solve(curvature + damping * identity, gradient)
        direction_norm = float(torch.linalg.vector_norm(direction.detach()))
        direction *= min(
            1.0, maximum_parameter_step / max(direction_norm, 1.0e-30)
        )
        offset = 0
        with torch.no_grad():
            for name, parameter in parameters.items():
                width = parameter.numel()
                parameter.copy_(original[name])
                parameter.add_(
                    direction[offset : offset + width].reshape(parameter.shape),
                    alpha=-1.0,
                )
                offset += width
            candidate_loss = -_oce_from_pnl(path_pnl(parameters))
        if float(candidate_loss) <= float(current_loss):
            return max(damping / 1.5, 1.0e-8)
        damping *= 10.0
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(original[name])
    return damping


def _gradient_stationarity(
    model: OrthogonalResidualPolicy,
    projection: torch.Tensor,
    state: torch.Tensor,
    returns: torch.Tensor,
) -> dict[str, float | int]:
    """Return a global Wald test for a zero self-normalized OCE gradient."""

    wrapper = _FixedProjectionWrapper(model, projection)
    parameters = dict(wrapper.named_parameters())

    def path_pnl(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(wrapper, values, (state, returns))

    with torch.no_grad():
        pnl = path_pnl(parameters)
        probability = torch.softmax(-pnl, dim=0)
    jacobian_tree = jacfwd(path_pnl)(parameters)
    jacobian = torch.cat(
        [value.reshape(len(state), -1) for value in jacobian_tree.values()], dim=1
    ).detach()
    score = -jacobian
    gradient = probability @ score
    centered_score = score - gradient
    # For the ratio E[w s]/E[w], n*q_i*(s_i-g) is the empirical influence
    # function. This is its sample covariance divided by n.
    covariance = (
        len(state)
        / (len(state) - 1.0)
        * centered_score.T
        @ (probability.square()[:, None] * centered_score)
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    threshold = max(float(torch.max(eigenvalues)), 0.0) * 1.0e-10
    keep = eigenvalues > threshold
    degrees_of_freedom = int(torch.sum(keep))
    if degrees_of_freedom == 0:
        wald_statistic = 0.0 if float(torch.linalg.vector_norm(gradient)) == 0.0 else math.inf
    else:
        coordinates = eigenvectors[:, keep].T @ gradient
        wald_statistic = float(torch.sum(coordinates.square() / eigenvalues[keep]))
    p_value = float(chi2.sf(wald_statistic, degrees_of_freedom))
    return {
        "gradient_l2_norm": float(torch.linalg.vector_norm(gradient)),
        "wald_statistic": wald_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "effective_sample_size": float(1.0 / torch.sum(probability.square())),
    }


def _make_optimizer(
    name: str,
    model: OrthogonalResidualPolicy,
) -> torch.optim.Optimizer | None:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=1.0e-3)
    if name == "amsgrad":
        return torch.optim.Adam(model.parameters(), lr=1.0e-3, amsgrad=True)
    if name == "yogi":
        return Yogi(model.parameters(), lr=1.0e-3)
    if name == "shampoo":
        return Shampoo(model.parameters(), lr=1.0e-2)
    if name == "gauss_newton":
        return None
    raise ValueError(f"unknown optimizer {name!r}")


def train_batched_policy(
    optimizer_name: str,
    *,
    policy_config: OrthogonalPolicyConfig,
    train_config: BatchedTrainConfig,
    training_state: np.ndarray,
    training_return: np.ndarray,
    validation_state: np.ndarray,
    validation_return: np.ndarray,
    test_state: np.ndarray,
    test_return: np.ndarray,
    initial_affine_coefficients: np.ndarray,
) -> BatchedOptimizerResult:
    """Fit one optimizer, select on validation, then evaluate test once."""

    torch.manual_seed(train_config.seed)
    env = BaseEnvironment(precision=64, device="cpu")
    model = OrthogonalResidualPolicy(
        env,
        config=policy_config,
        instrument_count=training_return.shape[1],
    )
    with torch.no_grad():
        model.affine_coefficients.copy_(
            torch.as_tensor(initial_affine_coefficients, dtype=torch.float64)
        )
    train_state = torch.as_tensor(training_state, dtype=torch.float64)
    train_return = torch.as_tensor(training_return, dtype=torch.float64)
    valid_state = torch.as_tensor(validation_state, dtype=torch.float64)
    valid_return = torch.as_tensor(validation_return, dtype=torch.float64)
    heldout_state = torch.as_tensor(test_state, dtype=torch.float64)
    heldout_return = torch.as_tensor(test_return, dtype=torch.float64)
    optimizer = _make_optimizer(optimizer_name, model)
    generator = torch.Generator().manual_seed(train_config.seed + 1)
    projection = _projection_coefficients(model, train_state)
    best_training, _ = _evaluate(model, projection, train_state, train_return)
    epochs_without_training_improvement = 0
    updates = 0
    damping = 1.0e-2
    stop_epoch = train_config.epochs
    stopped_for_stationarity = False
    consecutive_stationary_checks = 0
    started = time.monotonic()

    for epoch in range(1, train_config.epochs + 1):
        projection = _projection_coefficients(model, train_state)
        permutation = torch.randperm(len(train_state), generator=generator)
        for start in range(0, len(train_state), train_config.batch_size):
            indices = permutation[start : start + train_config.batch_size]
            batch_state = train_state[indices]
            batch_return = train_return[indices]
            if optimizer is None:
                damping = _gauss_newton_step(
                    model,
                    projection,
                    batch_state,
                    batch_return,
                    damping=damping,
                    maximum_parameter_step=0.05,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                action, _ = model.action_with_projection(batch_state, projection)
                loss = -_oce_from_pnl(torch.sum(action * batch_return, dim=1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            updates += 1

        projection = _projection_coefficients(model, train_state)
        training_oce, _ = _evaluate(model, projection, train_state, train_return)
        if training_oce > best_training + 1.0e-10:
            best_training = training_oce
            epochs_without_training_improvement = 0
        else:
            epochs_without_training_improvement += 1
        if epochs_without_training_improvement >= train_config.patience:
            if optimizer is None:
                damping *= 10.0
            else:
                for group in optimizer.param_groups:
                    group["lr"] = max(
                        train_config.minimum_learning_rate, group["lr"] * 0.5
                    )
            epochs_without_training_improvement = 0
        if epoch % train_config.validation_gradient_check_interval == 0:
            statistics = _gradient_stationarity(
                model, projection, valid_state, valid_return
            )
            if statistics["p_value"] > train_config.validation_gradient_alpha:
                consecutive_stationary_checks += 1
            else:
                consecutive_stationary_checks = 0
            if (
                consecutive_stationary_checks
                >= train_config.validation_gradient_consecutive_checks
            ):
                stop_epoch = epoch
                stopped_for_stationarity = True
                break

    seconds = time.monotonic() - started
    final_projection = _projection_coefficients(model, train_state)
    final_training_oce, _ = _evaluate(
        model, final_projection, train_state, train_return
    )
    final_validation_oce, _ = _evaluate(
        model, final_projection, valid_state, valid_return
    )
    training_oce, residual_rms = _evaluate(
        model, final_projection, train_state, train_return
    )
    validation_oce, _ = _evaluate(
        model, final_projection, valid_state, valid_return
    )
    test_oce, _ = _evaluate(model, final_projection, heldout_state, heldout_return)
    validation_statistics = _gradient_stationarity(
        model, final_projection, valid_state, valid_return
    )
    model.zero_grad(set_to_none=True)
    action, _ = model.action_with_projection(train_state, final_projection)
    (-_oce_from_pnl(torch.sum(action * train_return, dim=1))).backward()
    gradient = torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in model.parameters()]
    )
    final_learning_rate = (
        damping if optimizer is None else float(optimizer.param_groups[0]["lr"])
    )
    return BatchedOptimizerResult(
        optimizer=optimizer_name,
        stop_epoch=stop_epoch,
        stopped_for_validation_stationarity=stopped_for_stationarity,
        updates=updates,
        seconds=seconds,
        training_oce=training_oce,
        validation_oce=validation_oce,
        test_oce=test_oce,
        final_training_oce=final_training_oce,
        final_validation_oce=final_validation_oce,
        training_gradient_l2_norm=float(torch.linalg.vector_norm(gradient)),
        validation_gradient_l2_norm=float(
            validation_statistics["gradient_l2_norm"]
        ),
        validation_gradient_wald_statistic=float(
            validation_statistics["wald_statistic"]
        ),
        validation_gradient_degrees_of_freedom=int(
            validation_statistics["degrees_of_freedom"]
        ),
        validation_gradient_p_value=float(validation_statistics["p_value"]),
        validation_effective_sample_size=float(
            validation_statistics["effective_sample_size"]
        ),
        residual_rms=residual_rms,
        final_learning_rate=final_learning_rate,
    )


def run_batched_comparison(
    *,
    market_config: MarketConfig,
    policy_config: OrthogonalPolicyConfig,
    train_config: BatchedTrainConfig,
    optimizers: tuple[str, ...],
) -> dict[str, Any]:
    """Run train/validation/test comparison with independent path populations."""

    training_paths = simulate_market(market_config)
    validation_paths = simulate_market(
        MarketConfig(**{**asdict(market_config), "seed": market_config.seed + 1})
    )
    test_paths = simulate_market(
        MarketConfig(**{**asdict(market_config), "seed": market_config.seed + 2})
    )
    training_state, training_return = build_policy_population(training_paths)
    state_rms = math.sqrt(float(np.mean(np.square(training_paths.v[:, 1]))))

    def population(paths: Any) -> tuple[np.ndarray, np.ndarray]:
        state = paths.v[:, 1] / state_rms
        returns = np.column_stack(
            (paths.stock_return[:, 1], paths.option_return[:, 1, :])
        )
        return state, returns

    validation_state, validation_return = population(validation_paths)
    test_state, test_return = population(test_paths)
    affine_design, _ = build_trading_design(training_paths)
    affine = solve_exponential_oce(affine_design, precondition=True)
    affine_coefficients = affine.coefficients.reshape((-1, 2)).T
    validation_basis = np.column_stack((np.ones_like(validation_state), validation_state))
    test_basis = np.column_stack((np.ones_like(test_state), test_state))
    affine_validation_pnl = np.sum(
        (validation_basis @ affine_coefficients) * validation_return, axis=1
    )
    affine_test_pnl = np.sum((test_basis @ affine_coefficients) * test_return, axis=1)
    results = [
        train_batched_policy(
            name,
            policy_config=policy_config,
            train_config=train_config,
            training_state=training_state,
            training_return=training_return,
            validation_state=validation_state,
            validation_return=validation_return,
            test_state=test_state,
            test_return=test_return,
            initial_affine_coefficients=affine_coefficients,
        )
        for name in optimizers
    ]
    return {
        "market_config": asdict(market_config),
        "train_config": asdict(train_config),
        "affine_training_oce": affine.oce,
        "affine_validation_oce": float(
            _oce_from_pnl(torch.as_tensor(affine_validation_pnl, dtype=torch.float64))
        ),
        "affine_test_oce": float(
            _oce_from_pnl(torch.as_tensor(affine_test_pnl, dtype=torch.float64))
        ),
        "results": [result.summary() for result in results],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["adam", "amsgrad", "yogi", "shampoo", "gauss_newton"],
    )
    return parser


def main() -> None:
    """Run the comparison and print JSON diagnostics."""

    arguments = _parser().parse_args()
    torch.set_num_threads(arguments.torch_threads)
    result = run_batched_comparison(
        market_config=MarketConfig(paths=arguments.paths, seed=20260930),
        policy_config=OrthogonalPolicyConfig(
            label="batched-simple-ou",
            network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
        ),
        train_config=BatchedTrainConfig(
            batch_size=arguments.batch_size,
            epochs=arguments.epochs,
            patience=arguments.patience,
        ),
        optimizers=tuple(arguments.optimizers),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
