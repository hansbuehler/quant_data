"""Claim-minus-zero GROMM policy fitting and architecture selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
    GrommTrainingResult,
    fit_gromm_policy,
)
from quantdata.torch import BaseEnvironment, torch

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, kw_only=True)
class GrommHedgeFit:
    """Signed pricing policies and one executable net-turnover hedge."""

    claim: GrommTrainingResult
    short: GrommTrainingResult
    zero: GrommTrainingResult
    pure: GrommTrainingResult
    objective: GrommObjective
    risk_parameter: float

    @property
    def buyer_indifference_price(self) -> float:
        """Return the long-claim certainty-equivalent increment."""
        return self.zero.validation_loss - self.claim.validation_loss

    @property
    def seller_indifference_price(self) -> float:
        """Return the short-claim certainty-equivalent decrement."""
        return self.short.validation_loss - self.zero.validation_loss


@dataclass(frozen=True, kw_only=True)
class GrommArchitectureFit:
    """One candidate's validation result and complexity."""

    config: GrommPolicyConfig
    result: GrommTrainingResult
    parameter_count: int


@dataclass(frozen=True, kw_only=True)
class GrommArchitectureSelection:
    """Precommitted cheapest-within-one-standard-error selection."""

    selected: GrommPolicyConfig
    candidates: tuple[GrommArchitectureFit, ...]
    best_validation_loss: float
    best_validation_standard_error: float


def fit_gromm_claim_minus_zero(
    data: GrommTrainingData,
    *,
    pure_data: GrommTrainingData,
    policy_config: GrommPolicyConfig,
    objective: GrommObjective,
    risk_parameter: float,
    checkpoint_directory: Path,
    train_config: GrommTrainConfig = GrommTrainConfig(),
    environment: BaseEnvironment | None = None,
    claim_initial_positions: FloatArray | None = None,
    short_initial_positions: FloatArray | None = None,
    zero_initial_positions: FloatArray | None = None,
    pure_initial_positions: FloatArray | None = None,
) -> GrommHedgeFit:
    """Fit signed pricing policies and a direct cost-consistent pure hedge."""
    _validate_pure_data(data, pure_data)
    claim_config = replace(
        policy_config,
        label=_role_label(policy_config.label, "claim"),
    )
    zero_config = replace(
        policy_config,
        label=_role_label(policy_config.label, "zero"),
    )
    short_config = replace(
        policy_config,
        label=_role_label(policy_config.label, "short"),
    )
    pure_config = replace(
        policy_config,
        label=_role_label(policy_config.label, "pure"),
    )
    claim_data = _with_initial_positions(data, claim_initial_positions)
    short_data = replace(
        _with_initial_positions(data, short_initial_positions),
        claim_returns=-data.claim_returns,
    )
    zero_data = replace(
        _with_initial_positions(data, zero_initial_positions),
        claim_returns=np.zeros_like(data.claim_returns),
    )
    claim = fit_gromm_policy(
        claim_data,
        policy_config=claim_config,
        objective=objective,
        risk_parameter=risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        environment=environment,
    )
    short = fit_gromm_policy(
        short_data,
        policy_config=short_config,
        objective=objective,
        risk_parameter=risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        environment=environment,
    )
    zero = fit_gromm_policy(
        zero_data,
        policy_config=zero_config,
        objective=objective,
        risk_parameter=risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        environment=environment,
    )
    pure = fit_gromm_policy(
        _with_initial_positions(
            pure_data,
            pure_initial_positions,
        ),
        policy_config=pure_config,
        objective=objective,
        risk_parameter=risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        environment=environment,
    )
    return GrommHedgeFit(
        claim=claim,
        short=short,
        zero=zero,
        pure=pure,
        objective=objective,
        risk_parameter=risk_parameter,
    )


def gromm_pure_hedge_actions(
    fit: GrommHedgeFit,
    *,
    features: FloatArray,
    active_mask: BoolArray,
) -> tuple[FloatArray, int]:
    """Evaluate the directly trained executable net-position policy."""
    policy = fit.pure.policy
    feature_tensor = torch.as_tensor(
        features,
        dtype=policy.fdtype,
        device=policy.device,
    )
    mask_tensor = torch.as_tensor(
        active_mask,
        dtype=torch.bool,
        device=policy.device,
    )
    policy.eval()
    with torch.no_grad():
        raw = np.asarray(policy(feature_tensor, mask_tensor).cpu(), dtype=np.float64)
    capped = np.clip(
        raw,
        -policy.config.position_cap,
        policy.config.position_cap,
    )
    return capped, int(np.count_nonzero(capped != raw))


def gromm_signed_policy_actions(
    fit: GrommHedgeFit,
    *,
    features: FloatArray,
    active_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate independently trained long, short, and zero policy positions."""
    policies = (fit.claim.policy, fit.short.policy, fit.zero.policy)
    reference = policies[0]
    if any(
        policy.config.position_cap != reference.config.position_cap
        or policy.device != reference.device
        or policy.fdtype != reference.fdtype
        for policy in policies[1:]
    ):
        raise ValueError("signed GROMM policies have incompatible environments")
    feature_tensor = torch.as_tensor(
        features,
        dtype=reference.fdtype,
        device=reference.device,
    )
    mask_tensor = torch.as_tensor(
        active_mask,
        dtype=torch.bool,
        device=reference.device,
    )
    for policy in policies:
        policy.eval()
    with torch.no_grad():
        values = tuple(
            np.asarray(policy(feature_tensor, mask_tensor).cpu(), dtype=np.float64)
            for policy in policies
        )
    return values


def _with_initial_positions(
    data: GrommTrainingData,
    positions: FloatArray | None,
) -> GrommTrainingData:
    if positions is None:
        return data
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (data.instrument_count,) or not np.all(np.isfinite(values)):
        raise ValueError("signed GROMM initial positions do not align")
    return replace(data, initial_positions=values)


def _validate_pure_data(
    pricing_data: GrommTrainingData,
    pure_data: GrommTrainingData,
) -> None:
    """Require Q-shadow gains on exactly the pricing scenario geometry."""
    if (
        pure_data.features.shape != pricing_data.features.shape
        or pure_data.instrument_returns.shape
        != pricing_data.instrument_returns.shape
        or pure_data.claim_returns.shape != pricing_data.claim_returns.shape
        or pure_data.active_mask.shape != pricing_data.active_mask.shape
        or pure_data.linear_transaction_costs.shape
        != pricing_data.linear_transaction_costs.shape
    ):
        raise ValueError("pure GROMM training data do not match pricing geometry")
    for name in (
        "features",
        "claim_returns",
        "active_mask",
        "linear_transaction_costs",
        "settles_next",
        "terminal_transaction_costs",
    ):
        pricing_value = getattr(pricing_data, name)
        pure_value = getattr(pure_data, name)
        if pricing_value is None or pure_value is None:
            if pricing_value is not pure_value:
                raise ValueError(f"pure GROMM {name} does not match pricing data")
        elif not np.array_equal(pricing_value, pure_value):
            raise ValueError(f"pure GROMM {name} does not match pricing data")


def select_gromm_architecture(
    data: GrommTrainingData,
    *,
    candidates: tuple[GrommPolicyConfig, ...],
    objective: GrommObjective,
    risk_parameter: float,
    checkpoint_directory: Path,
    train_config: GrommTrainConfig = GrommTrainConfig(),
    environment: BaseEnvironment | None = None,
) -> GrommArchitectureSelection:
    """Choose the smallest candidate within one SE of the best validation loss."""
    if len(candidates) != 4:
        raise ValueError("GROMM architecture selection requires four candidates")
    fits: list[GrommArchitectureFit] = []
    for candidate in candidates:
        result = fit_gromm_policy(
            data,
            policy_config=replace(
                candidate,
                label=_role_label(candidate.label, "architecture"),
            ),
            objective=objective,
            risk_parameter=risk_parameter,
            checkpoint_directory=checkpoint_directory,
            train_config=train_config,
            environment=environment,
        )
        fits.append(
            GrommArchitectureFit(
                config=candidate,
                result=result,
                parameter_count=sum(
                    parameter.numel() for parameter in result.policy.parameters()
                ),
            )
        )
    losses = np.asarray([fit.result.validation_loss for fit in fits])
    best_index = int(np.argmin(losses))
    best_loss = float(losses[best_index])
    best_standard_error = fits[best_index].result.validation_standard_error
    eligible = tuple(
        fit
        for fit in fits
        if fit.result.validation_loss <= best_loss + best_standard_error
    )
    selected = min(
        eligible,
        key=lambda fit: (fit.parameter_count, fit.result.validation_loss),
    )
    return GrommArchitectureSelection(
        selected=selected.config,
        candidates=tuple(fits),
        best_validation_loss=best_loss,
        best_validation_standard_error=best_standard_error,
    )


def _role_label(label: str, role: str) -> str:
    base = label[1:] if label.startswith("*") else label
    return f"{base}-{role}"
