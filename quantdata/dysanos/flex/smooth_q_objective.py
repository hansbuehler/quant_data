"""Smooth price-normalized objectives and blockwise validation for FLEX Q fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class SmoothPriceObjectiveConfig:
    """Configuration of the smooth option-return and robust-cut objective."""

    temperature: float = 1.0e-2
    robust_block_weight: float = 0.25
    block_weight_decay: float = 0.5
    minimum_option_price: float = 1.0e-6

    def __post_init__(self) -> None:
        verify_inp(
            self.temperature > 0.0,
            lambda: f"temperature must be positive, found {self.temperature}",
        )
        verify_inp(
            self.robust_block_weight >= 0.0,
            lambda: (
                "robust_block_weight must be non-negative, found "
                f"{self.robust_block_weight}"
            ),
        )
        verify_inp(
            0.0 < self.block_weight_decay <= 1.0,
            lambda: (
                "block_weight_decay must lie in (0, 1], found "
                f"{self.block_weight_decay}"
            ),
        )
        verify_inp(
            self.minimum_option_price > 0.0,
            lambda: (
                "minimum_option_price must be positive, found "
                f"{self.minimum_option_price}"
            ),
        )


def _normalized_logsumexp(
    squared_errors: torch.Tensor,
    *,
    temperature: float,
    dim: int,
) -> torch.Tensor:
    count = squared_errors.shape[dim]
    if count <= 0:
        raise ValueError("smooth maximum received an empty reduction axis")
    return temperature * (
        torch.logsumexp(squared_errors / temperature, dim=dim) - math.log(count)
    )


def _scalar_statistics(values: torch.Tensor) -> dict[str, Any]:
    finite = values.detach().double().reshape(-1).cpu()
    if len(finite) < 2 or not bool(torch.all(torch.isfinite(finite))):
        raise ValueError("block statistics require at least two finite values")
    return {
        "values": finite.tolist(),
        "mean": float(torch.mean(finite)),
        "median": float(torch.median(finite)),
        "mc_standard_error": float(
            torch.std(finite, correction=1) / math.sqrt(len(finite))
        ),
        "q10": float(torch.quantile(finite, 0.10)),
        "q25": float(torch.quantile(finite, 0.25)),
        "q75": float(torch.quantile(finite, 0.75)),
        "q90": float(torch.quantile(finite, 0.90)),
    }


def smooth_price_objective(
    adjusted_block_means: torch.Tensor,
    option_prices: torch.Tensor,
    *,
    config: SmoothPriceObjectiveConfig,
    derivative: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Evaluate the full-sample LSE loss plus a geometric worst-cut penalty.

    ``adjusted_block_means`` has axes ``(blocks, states, instruments)``.  Each
    moment is divided by its current option premium before squaring.  The
    normalized log-sum-exp is a smooth maximum over all state-instrument
    constraints.  Ten block losses are ranked only after that smooth global
    reduction, so the sole piecewise-smooth operation selects among ten cuts,
    not among thousands of individual constraints.
    """

    if (
        adjusted_block_means.ndim != 3
        or option_prices.shape != adjusted_block_means.shape[1:]
        or adjusted_block_means.shape[0] < 2
        or adjusted_block_means.shape[1] < 1
        or adjusted_block_means.shape[2] < 1
    ):
        raise ValueError("smooth price objective received inconsistent tensor shapes")
    if not bool(torch.all(torch.isfinite(adjusted_block_means))):
        raise FloatingPointError("smooth price objective received non-finite means")
    if not bool(torch.all(torch.isfinite(option_prices) & (option_prices > 0.0))):
        raise FloatingPointError("option prices must be finite and strictly positive")

    floored_prices = option_prices.clamp_min(config.minimum_option_price)
    floor_count = int(torch.sum(option_prices < config.minimum_option_price))
    blocks = adjusted_block_means.shape[0]
    constraints = adjusted_block_means.shape[1] * adjusted_block_means.shape[2]

    with torch.enable_grad():
        means = adjusted_block_means.detach().double()
        if derivative:
            means = means.requires_grad_(True)
        normalized = means / floored_prices.detach().double()[None]
        full_normalized = torch.mean(normalized, dim=0)
        full_squared = full_normalized.square().reshape(-1)
        full_loss = _normalized_logsumexp(
            full_squared,
            temperature=config.temperature,
            dim=0,
        )
        block_squared = normalized.square().reshape(blocks, constraints)
        block_losses = _normalized_logsumexp(
            block_squared,
            temperature=config.temperature,
            dim=1,
        )
        ordered_block_losses, block_order = torch.sort(
            block_losses,
            descending=True,
        )
        rank_weights = torch.pow(
            torch.as_tensor(
                config.block_weight_decay,
                dtype=means.dtype,
                device=means.device,
            ),
            torch.arange(blocks, dtype=means.dtype, device=means.device),
        )
        rank_weights = rank_weights / torch.sum(rank_weights)
        robust_block_loss = torch.sum(rank_weights * ordered_block_losses)
        total_loss = full_loss + config.robust_block_weight * robust_block_loss
        mean_derivative = (
            torch.autograd.grad(total_loss, means)[0].detach()
            if derivative
            else None
        )

    full_weights = torch.softmax(full_squared.detach() / config.temperature, dim=0)
    block_weights = torch.softmax(block_squared.detach() / config.temperature, dim=1)
    full_ess = 1.0 / torch.sum(full_weights.square())
    block_ess = 1.0 / torch.sum(block_weights.square(), dim=1)
    absolute_full = torch.abs(full_normalized.detach()).reshape(-1)
    worst_decile_count = max(1, math.ceil(0.10 * constraints))
    metrics: dict[str, Any] = {
        "loss": float(total_loss.detach()),
        "full_loss": float(full_loss.detach()),
        "robust_block_loss": float(robust_block_loss.detach()),
        "block_loss": _scalar_statistics(block_losses),
        "block_order_worst_first": block_order.detach().cpu().tolist(),
        "block_rank_weights": rank_weights.detach().cpu().tolist(),
        "maximum_abs_price_normalized_mean": float(torch.max(absolute_full)),
        "worst_10_percent_mean_abs_price_normalized_mean": float(
            torch.mean(torch.topk(absolute_full, worst_decile_count).values)
        ),
        "constraint_weight_effective_sample_size": float(full_ess),
        "constraint_weight_maximum": float(torch.max(full_weights)),
        "block_constraint_weight_effective_sample_size_minimum": float(
            torch.min(block_ess)
        ),
        "block_constraint_weight_effective_sample_size_mean": float(
            torch.mean(block_ess)
        ),
        "num_constraints": constraints,
        "num_blocks": blocks,
        "minimum_observed_option_price": float(torch.min(option_prices)),
        "option_price_floor_count": floor_count,
    }
    return total_loss.detach(), mean_derivative, metrics


def studentized_block_sign_flip_test(
    adjusted_block_means: torch.Tensor,
    *,
    tail_fraction: float = 0.10,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Enumerate the exact block sign-flip null for joint moment diagnostics.

    One sign is applied to the complete vector of state-option means in each
    block.  This retains all cross-constraint dependence.  Studentization uses
    the sample standard deviation across blocks and therefore has a marginal
    Student-t reference under independent Gaussian blocks.
    """

    if adjusted_block_means.ndim != 3:
        raise ValueError("sign-flip test expects block/state/instrument means")
    blocks = adjusted_block_means.shape[0]
    if not 2 <= blocks <= 20 or not 0.0 < tail_fraction <= 1.0 or threshold <= 0.0:
        raise ValueError("sign-flip test configuration is invalid")
    values = adjusted_block_means.detach().double().reshape(blocks, -1)
    if not bool(torch.all(torch.isfinite(values))):
        raise FloatingPointError("sign-flip test received non-finite block means")
    constraints = values.shape[1]
    tail_count = max(1, math.ceil(tail_fraction * constraints))
    patterns = 1 << blocks
    pattern_index = torch.arange(patterns, device=values.device, dtype=torch.int64)
    bits = torch.arange(blocks, device=values.device, dtype=torch.int64)
    signs = 2.0 * ((pattern_index[:, None] >> bits[None]) & 1).double() - 1.0
    signed_mean = signs @ values / blocks
    square_sum = torch.sum(values.square(), dim=0)
    signed_variance = (
        square_sum[None] - blocks * signed_mean.square()
    ) / (blocks - 1)
    signed_se = torch.sqrt(
        torch.clamp_min(signed_variance, 0.0) / blocks
    ).clamp_min(1.0e-15)
    absolute_t_null = torch.abs(signed_mean / signed_se)
    null_maximum = torch.max(absolute_t_null, dim=1).values
    null_tail_mean = torch.mean(
        torch.topk(absolute_t_null, tail_count, dim=1).values,
        dim=1,
    )
    null_count = torch.sum(absolute_t_null > threshold, dim=1).double()

    observed_mean = torch.mean(values, dim=0)
    observed_se = (
        torch.std(values, dim=0, correction=1) / math.sqrt(blocks)
    ).clamp_min(1.0e-15)
    observed_absolute_t = torch.abs(observed_mean / observed_se)
    observed_maximum = torch.max(observed_absolute_t)
    observed_tail_mean = torch.mean(
        torch.topk(observed_absolute_t, tail_count).values
    )
    observed_count = torch.sum(observed_absolute_t > threshold).double()

    # The first and last patterns are the all-negative and all-positive
    # transforms.  Their absolute statistics equal the observed statistic
    # algebraically; pinning them removes roundoff from two different reduction
    # paths and preserves the exact test's minimum attainable p-value.
    null_maximum[[0, -1]] = observed_maximum
    null_tail_mean[[0, -1]] = observed_tail_mean
    null_count[[0, -1]] = observed_count

    def p_value(null: torch.Tensor, observed: torch.Tensor) -> float:
        return float(torch.mean((null >= observed).double()))

    return {
        "num_blocks": blocks,
        "degrees_of_freedom": blocks - 1,
        "num_constraints": constraints,
        "num_sign_patterns": patterns,
        "same_block_sign_across_all_constraints": True,
        "beta_treated_as_fixed_within_resamples": True,
        "maximum_abs_t": float(observed_maximum),
        "worst_tail_fraction": tail_fraction,
        "worst_tail_mean_abs_t": float(observed_tail_mean),
        "count_abs_t_above_threshold": int(observed_count),
        "threshold": threshold,
        "global_maximum_p_value": p_value(null_maximum, observed_maximum),
        "global_tail_mean_p_value": p_value(null_tail_mean, observed_tail_mean),
        "global_count_p_value": p_value(null_count, observed_count),
        "simultaneous_95_percent_abs_t_critical_value": float(
            torch.quantile(null_maximum, 0.95)
        ),
        "null_maximum_abs_t_median": float(torch.median(null_maximum)),
        "null_worst_tail_mean_abs_t_median": float(torch.median(null_tail_mean)),
        "marginal_abs_t_quantiles": {
            "median": float(torch.quantile(observed_absolute_t, 0.50)),
            "q90": float(torch.quantile(observed_absolute_t, 0.90)),
            "q99": float(torch.quantile(observed_absolute_t, 0.99)),
        },
    }


def validation_confirmation_consistency(
    validation_block_means: torch.Tensor,
    confirmation_block_means: torch.Tensor,
) -> dict[str, float]:
    """Studentize validation-confirmation mean differences constraintwise."""

    if (
        validation_block_means.ndim != 3
        or confirmation_block_means.shape != validation_block_means.shape
        or validation_block_means.shape[0] < 2
    ):
        raise ValueError("validation and confirmation blocks are inconsistent")
    validation = validation_block_means.detach().double()
    confirmation = confirmation_block_means.detach().double()
    blocks = len(validation)
    difference = torch.mean(validation, dim=0) - torch.mean(confirmation, dim=0)
    difference_se = torch.sqrt(
        torch.var(validation, dim=0, correction=1) / blocks
        + torch.var(confirmation, dim=0, correction=1) / blocks
    ).clamp_min(1.0e-15)
    absolute_t = torch.abs(difference / difference_se).reshape(-1)
    worst_count = max(1, math.ceil(0.10 * len(absolute_t)))
    return {
        "maximum_abs_difference_t": float(torch.max(absolute_t)),
        "worst_10_percent_mean_abs_difference_t": float(
            torch.mean(torch.topk(absolute_t, worst_count).values)
        ),
        "median_abs_difference_t": float(torch.median(absolute_t)),
    }
