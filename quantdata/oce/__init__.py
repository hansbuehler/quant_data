"""Small, self-contained experiments with optimized certainty equivalents."""

from .batched_optimizer_oce import (
    BatchedOptimizerResult,
    BatchedTrainConfig,
    run_batched_comparison,
    train_batched_policy,
)
from .nonlinear_residual_oce import (
    NonlinearOceResult,
    OrthogonalPolicyConfig,
    OrthogonalResidualPolicy,
    build_policy_population,
    evaluate_nonlinear_policy,
    run_nonlinear_experiment,
    solve_nonlinear_oce,
)
from .simple_ou_oce import (
    MarketConfig,
    MarketPaths,
    OceResult,
    build_trading_design,
    run_experiment,
    simulate_market,
    solve_exponential_oce,
    solve_exponential_oce_adam,
)

__all__ = [
    "BatchedOptimizerResult",
    "BatchedTrainConfig",
    "MarketConfig",
    "MarketPaths",
    "NonlinearOceResult",
    "OceResult",
    "OrthogonalPolicyConfig",
    "OrthogonalResidualPolicy",
    "build_policy_population",
    "build_trading_design",
    "evaluate_nonlinear_policy",
    "run_nonlinear_experiment",
    "run_batched_comparison",
    "run_experiment",
    "simulate_market",
    "solve_exponential_oce_adam",
    "solve_exponential_oce",
    "solve_nonlinear_oce",
    "train_batched_policy",
]
