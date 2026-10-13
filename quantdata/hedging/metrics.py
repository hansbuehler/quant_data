"""Model-neutral records emitted by hedging benchmark smoke tests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HedgeSmokeResult:
    """Compact common result for one deterministic model smoke test."""

    model: str
    measure: str
    origin_date: str
    terminal_date: str
    claim: str
    market_price: float
    model_price: float
    payoff: float
    unhedged_pnl: float
    hedged_pnl: float
    transaction_cost: float
    actions: dict[str, float]
    pricing_difference: float
    pure_hedge_error: float
    objective_value: float
    converged: bool
    cap_hits: int
    claim_quantity: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema: str = "hedging-smoke-v2"

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.measure.strip() or not self.claim.strip():
            raise ValueError("model, measure, and claim must be non-empty")
        numeric = (
            self.market_price,
            self.model_price,
            self.payoff,
            self.unhedged_pnl,
            self.hedged_pnl,
            self.transaction_cost,
            self.pricing_difference,
            self.pure_hedge_error,
            self.objective_value,
            self.claim_quantity,
            *self.actions.values(),
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("smoke-result numerical fields must be finite")
        if self.market_price < 0.0 or self.model_price < 0.0 or self.payoff < 0.0:
            raise ValueError("prices and payoff must be non-negative")
        if self.transaction_cost < 0.0 or self.cap_hits < 0:
            raise ValueError("cost and cap hits must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    def write_json(self, path: Path) -> None:
        """Write the compact result with deterministic formatting."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def assert_smoke_fixture(result: HedgeSmokeResult, fixture_path: Path) -> None:
    """Check selected deterministic outputs against a committed tolerance fixture."""
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    expected = payload.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("smoke fixture must contain an expected object")
    absolute_tolerance = float(payload.get("absolute_tolerance", 0.0))
    relative_tolerance = float(payload.get("relative_tolerance", 0.0))
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("smoke fixture tolerances must be non-negative")
    _assert_fixture_value(
        actual=result.to_dict(),
        expected=expected,
        path="result",
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def mean_variance_risk(losses: ArrayLike, *, risk_parameter: float) -> float:
    """Return mean loss plus one-half risk aversion times population variance."""
    values = _finite_losses(losses)
    risk = _positive_risk_parameter(risk_parameter)
    return float(np.mean(values) + 0.5 * risk * np.var(values))


def mean_volatility_risk(losses: ArrayLike, *, risk_parameter: float) -> float:
    """Return mean loss plus risk aversion times population volatility."""
    values = _finite_losses(losses)
    risk = _positive_risk_parameter(risk_parameter)
    return float(np.mean(values) + risk * np.std(values))


def entropy_risk(losses: ArrayLike, *, risk_parameter: float) -> float:
    """Return the stable empirical entropic risk of the supplied losses."""
    values = _finite_losses(losses)
    risk = _positive_risk_parameter(risk_parameter)
    return float((logsumexp(risk * values) - math.log(len(values))) / risk)


def power_exponential_risk(losses: ArrayLike, *, risk_parameter: float) -> float:
    """Return minus the empirical power-exponential OCE of terminal wealth."""
    values = _finite_losses(losses)
    risk = _positive_risk_parameter(risk_parameter)
    wealth = -values
    width = 50.0 / risk
    lower = float(np.min(wealth) - width)
    upper = float(np.max(wealth) + width)

    def first_order_condition(translation: float) -> float:
        residual = wealth - translation
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk * np.maximum(residual, 0.0)),
            1.0 - risk * residual,
        )
        return 1.0 - float(np.mean(marginal))

    translation = float(brentq(first_order_condition, lower, upper))
    residual = wealth - translation
    utility = np.where(
        residual >= 0.0,
        -np.expm1(-risk * np.maximum(residual, 0.0)) / risk,
        residual - 0.5 * risk * residual**2,
    )
    return -float(translation + np.mean(utility))


def maximum_drawdown_from_losses(losses: ArrayLike) -> float:
    """Return maximum cumulative-P&L drawdown for chronological losses.

    The account starts at zero, period P&L is minus loss, and the statistic is
    expressed in the same units as one loss observation.
    """
    values = _finite_losses(losses)
    cumulative_pnl = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(-values, dtype=np.float64))
    )
    running_peak = np.maximum.accumulate(cumulative_pnl)
    return float(np.max(running_peak - cumulative_pnl))


def _finite_losses(losses: ArrayLike) -> FloatArray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite vector")
    return values


def _positive_risk_parameter(risk_parameter: float) -> float:
    value = float(risk_parameter)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("risk_parameter must be finite-positive")
    return value


def _assert_fixture_value(
    *,
    actual: Any,
    expected: Any,
    path: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected an object, found {type(actual)}")
        missing = set(expected) - set(actual)
        if missing:
            raise AssertionError(f"{path}: missing fixture keys {sorted(missing)}")
        for name, value in expected.items():
            _assert_fixture_value(
                actual=actual[name],
                expected=value,
                path=f"{path}.{name}",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            float(actual),
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(f"{path}: expected {expected}, found {actual}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, found {actual!r}")
