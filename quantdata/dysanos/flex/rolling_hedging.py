"""One-session statistical hedges and lagged-open-interest proxy books."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from quantdata.biz_calendar import get_business_days


@dataclass(frozen=True)
class HedgeEstimate:
    """One fitted spot hedge for a signed portfolio change."""

    method: str
    spot_position: float
    objective: float
    cash_translation: float | None
    first_order_condition_cash: float | None
    first_order_condition_hedge: float
    optimizer_success: bool
    spot_bound_active: bool

    def to_dict(self) -> dict[str, float | str | bool | None]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def one_session_pairs(dates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent full-business-session pairs from an ISO date history."""
    parsed = [date.fromisoformat(value) for value in dates]
    if len(parsed) < 2 or parsed != sorted(set(parsed)):
        raise ValueError("dates must be unique, increasing, and contain two observations")
    calendar = get_business_days(parsed[0], parsed[-1], exclude_half_days=True)
    position = {value: index for index, value in enumerate(calendar)}
    origin: list[int] = []
    endpoint: list[int] = []
    for index in range(len(parsed) - 1):
        left = position.get(parsed[index])
        right = position.get(parsed[index + 1])
        if left is not None and right is not None and right - left == 1:
            origin.append(index)
            endpoint.append(index + 1)
    return np.asarray(origin, dtype=np.int64), np.asarray(endpoint, dtype=np.int64)


def mean_variance_hedge(
    portfolio_change: np.ndarray,
    spot_change: np.ndarray,
) -> HedgeEstimate:
    """Minimize the sample variance of ``portfolio_change + delta*spot_change``."""
    z, x = _aligned_vectors(portfolio_change, spot_change)
    centered_z = z - np.mean(z)
    centered_x = x - np.mean(x)
    denominator = float(centered_x @ centered_x)
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("spot changes have zero sample variance")
    delta = -float(centered_x @ centered_z) / denominator
    residual = z + delta * x
    variance = float(np.var(residual))
    foc = float(np.mean((residual - np.mean(residual)) * centered_x))
    return HedgeEstimate(
        method="mean-variance",
        spot_position=delta,
        objective=-0.5 * variance,
        cash_translation=None,
        first_order_condition_cash=None,
        first_order_condition_hedge=foc,
        optimizer_success=True,
        spot_bound_active=False,
    )


def oce_hedge(
    portfolio_change: np.ndarray,
    spot_change: np.ndarray,
    *,
    utility: str,
    risk_aversion: float,
    initial_spot_position: float | None = None,
) -> HedgeEstimate:
    r"""Maximize ``y + E[u(Z + delta X - y)]`` over ``(y, delta)``."""
    z, x = _aligned_vectors(portfolio_change, spot_change)
    if utility not in {"entropy", "power-exp"}:
        raise ValueError(f"unsupported utility {utility!r}")
    if not math.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be finite and positive")
    if initial_spot_position is None:
        initial_spot_position = mean_variance_hedge(z, x).spot_position

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray]:
        cash, delta = map(float, theta)
        residual = z + delta * x - cash
        marginal, utility_values = _utility_and_marginal(
            residual,
            utility=utility,
            risk_aversion=risk_aversion,
        )
        value = cash + float(np.mean(utility_values))
        gradient = np.asarray((1.0 - float(np.mean(marginal)), float(np.mean(marginal * x))))
        return -value, -gradient

    result = minimize(
        evaluate,
        np.asarray((float(np.mean(z)), initial_spot_position)),
        method="L-BFGS-B",
        jac=True,
        bounds=((-5.0, 5.0), (-20.0, 20.0)),
        options={"maxiter": 2_000, "ftol": 1.0e-15, "gtol": 1.0e-11, "maxls": 50},
    )
    cash, delta = map(float, result.x)
    residual = z + delta * x - cash
    marginal, _ = _utility_and_marginal(
        residual,
        utility=utility,
        risk_aversion=risk_aversion,
    )
    return HedgeEstimate(
        method=f"{utility}-oce",
        spot_position=delta,
        objective=-float(result.fun),
        cash_translation=cash,
        first_order_condition_cash=1.0 - float(np.mean(marginal)),
        first_order_condition_hedge=float(np.mean(marginal * x)),
        optimizer_success=bool(result.success),
        spot_bound_active=abs(delta) >= 19.999,
    )


def summarize_realized_pnl(values: np.ndarray) -> dict[str, float | int]:
    """Summarize a chronological vector of realized one-session P&Ls."""
    pnl = np.asarray(values, dtype=np.float64)
    if pnl.ndim != 1 or len(pnl) == 0 or np.any(~np.isfinite(pnl)):
        raise ValueError("realized P&L must be a non-empty finite vector")
    cutoff = float(np.quantile(pnl, 0.05))
    return {
        "observations": len(pnl),
        "mean": float(np.mean(pnl)),
        "standard_deviation": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        "root_mean_square": float(np.sqrt(np.mean(np.square(pnl)))),
        "mean_absolute": float(np.mean(np.abs(pnl))),
        "q01": float(np.quantile(pnl, 0.01)),
        "q05": cutoff,
        "median": float(np.median(pnl)),
        "q95": float(np.quantile(pnl, 0.95)),
        "q99": float(np.quantile(pnl, 0.99)),
        "lower_five_percent_mean": float(np.mean(pnl[pnl <= cutoff])),
    }


def summarize_hedge_estimates(
    estimates: Sequence[HedgeEstimate],
) -> dict[str, float | int]:
    """Summarize optimizer status and first-order conditions across dates."""
    if not estimates:
        raise ValueError("at least one hedge estimate is required")
    spot_positions = np.asarray(
        [estimate.spot_position for estimate in estimates], dtype=np.float64
    )
    hedge_conditions = np.asarray(
        [estimate.first_order_condition_hedge for estimate in estimates],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(spot_positions)) or np.any(~np.isfinite(hedge_conditions)):
        raise ValueError("hedge estimates must contain finite positions and conditions")
    cash_conditions = np.asarray(
        [
            estimate.first_order_condition_cash
            for estimate in estimates
            if estimate.first_order_condition_cash is not None
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(cash_conditions)):
        raise ValueError("cash first-order conditions must be finite")
    result: dict[str, float | int] = {
        "estimates": len(estimates),
        "optimizer_failures": int(
            sum(not estimate.optimizer_success for estimate in estimates)
        ),
        "spot_bound_hits": int(
            sum(estimate.spot_bound_active for estimate in estimates)
        ),
        "minimum_spot_position": float(np.min(spot_positions)),
        "median_spot_position": float(np.median(spot_positions)),
        "maximum_spot_position": float(np.max(spot_positions)),
        "maximum_absolute_hedge_first_order_condition": float(
            np.max(np.abs(hedge_conditions))
        ),
    }
    if len(cash_conditions):
        result["maximum_absolute_cash_first_order_condition"] = float(
            np.max(np.abs(cash_conditions))
        )
    return result


def paired_block_bootstrap_improvement(
    unhedged: np.ndarray,
    hedged: np.ndarray,
    *,
    block_length: int = 5,
    replications: int = 9_999,
    seed: int = 74_921,
) -> dict[str, object]:
    """Test reductions in absolute and squared loss with circular blocks."""
    baseline, candidate = _aligned_vectors(unhedged, hedged)
    if block_length <= 0 or replications <= 0:
        raise ValueError("block_length and replications must be positive")
    block_length = min(block_length, len(baseline))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0, len(baseline), size=(replications, math.ceil(len(baseline) / block_length))
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % len(baseline)
    indices = indices.reshape(replications, -1)[:, : len(baseline)]
    output: dict[str, object] = {
        "block_length": block_length,
        "replications": replications,
    }
    for name, baseline_loss, candidate_loss in (
        ("absolute", np.abs(baseline), np.abs(candidate)),
        ("squared", np.square(baseline), np.square(candidate)),
    ):
        difference = baseline_loss - candidate_loss
        draws = np.mean(difference[indices], axis=1)
        observed = float(np.mean(difference))
        null_draws = np.mean((difference - observed)[indices], axis=1)
        output[name] = {
            "mean_loss_reduction": observed,
            "bootstrap_95_percent_interval": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "one_sided_p_no_improvement": float(
                (1 + np.sum(null_draws >= observed)) / (replications + 1)
            ),
        }
    return output


def _aligned_vectors(
    portfolio_change: np.ndarray,
    spot_change: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(portfolio_change, dtype=np.float64)
    x = np.asarray(spot_change, dtype=np.float64)
    if z.ndim != 1 or x.shape != z.shape or len(z) < 3:
        raise ValueError("portfolio and spot changes must be aligned vectors")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(x)):
        raise ValueError("portfolio and spot changes must be finite")
    return z, x


def _utility_and_marginal(
    residual: np.ndarray,
    *,
    utility: str,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    if utility == "entropy":
        scaled = np.clip(-risk_aversion * residual, -700.0, 700.0)
        marginal = np.exp(scaled)
        utility_values = -marginal / risk_aversion
        return marginal, utility_values
    positive = residual >= 0.0
    marginal = np.empty_like(residual)
    utility_values = np.empty_like(residual)
    marginal[positive] = np.exp(-risk_aversion * residual[positive])
    marginal[~positive] = 1.0 - risk_aversion * residual[~positive]
    utility_values[positive] = -np.expm1(-risk_aversion * residual[positive]) / risk_aversion
    utility_values[~positive] = residual[~positive] - 0.5 * risk_aversion * np.square(
        residual[~positive]
    )
    return marginal, utility_values


__all__ = [
    "HedgeEstimate",
    "mean_variance_hedge",
    "oce_hedge",
    "one_session_pairs",
    "paired_block_bootstrap_improvement",
    "summarize_hedge_estimates",
    "summarize_realized_pnl",
]
