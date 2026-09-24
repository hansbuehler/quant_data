"""Exact exponential-utility density represented by traded return directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TheoreticalExponentialAction:
    """Minimum-norm traded action which spans a prescribed factor drift."""

    action: torch.Tensor
    replicated_factor_shift: torch.Tensor
    factor_replication_residual: torch.Tensor
    numerical_rank: torch.Tensor


@dataclass(frozen=True)
class ViableExponentialAction:
    """A regularized compatible factor shift and its exact traded action."""

    theoretical_action: TheoreticalExponentialAction
    requested_factor_shift: torch.Tensor
    viable_factor_shift: torch.Tensor
    factor_retention_fraction: torch.Tensor
    unscaled_action_l1: torch.Tensor
    total_scale: torch.Tensor


def theoretical_exponential_action(
    return_loadings: torch.Tensor,
    factor_shift: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-10,
) -> TheoreticalExponentialAction:
    r"""Construct the exact Gaussian exponential-utility action.

    ``return_loadings`` is :math:`B` in

    .. math:: dR = B\lambda\,dt + B\,dW^P.

    The returned minimum-norm action satisfies :math:`B^\top a=\lambda`
    whenever the requested factor shift is spanned by traded returns. Hence
    :math:`a=(BB^\top)^+B\lambda=\Sigma_R^+\mu_R`; no sample moments or
    trainable parameters enter the calculation.
    """

    if return_loadings.ndim != 3:
        raise ValueError("return_loadings must have shape (paths, instruments, factors)")
    if factor_shift.shape != (return_loadings.shape[0], return_loadings.shape[2]):
        raise ValueError("factor_shift must align with path and factor axes")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie strictly between zero and one")
    if not bool(torch.all(torch.isfinite(return_loadings))) or not bool(
        torch.all(torch.isfinite(factor_shift))
    ):
        raise ValueError("loadings and factor shifts must be finite")

    # Work on B directly. Forming B'B squares the condition number and would
    # silently discard precisely the weak traded directions this diagnostic is
    # intended to retain.
    left, singular_values, right_transpose = torch.linalg.svd(
        return_loadings,
        full_matrices=False,
    )
    largest = singular_values[:, :1]
    retained = singular_values > relative_tolerance * largest
    right_projection = torch.einsum("bfg,bg->bf", right_transpose, factor_shift)
    scaled_projection = torch.where(
        retained,
        right_projection / singular_values.clamp_min(torch.finfo(singular_values.dtype).tiny),
        0.0,
    )
    action = torch.einsum("bif,bf->bi", left, scaled_projection)
    replicated = torch.einsum("bif,bi->bf", return_loadings, action)
    return TheoreticalExponentialAction(
        action=action,
        replicated_factor_shift=replicated,
        factor_replication_residual=replicated - factor_shift,
        numerical_rank=torch.sum(retained, dim=1),
    )


def viable_exponential_action(
    return_loadings: torch.Tensor,
    requested_factor_shift: torch.Tensor,
    *,
    traded_ridge_ratio: float,
    maximum_factor_shift_norm: float,
    maximum_action_l1: float,
    relative_tolerance: float = 1.0e-10,
) -> ViableExponentialAction:
    r"""Shrink a compatible drift until its density and action are viable.

    If ``B=U diag(s) V'``, the traded ridge first maps the requested shift to

    ``lambda_r = V diag(s^2 / (s^2 + gamma)) V' lambda``,

    where ``gamma=traded_ridge_ratio*max(s)^2`` path by path. Its exact traded
    action is ``U diag(s / (s^2 + gamma)) V' lambda``. Radial scaling then
    enforces the factor-norm and action-L1 limits while preserving
    ``B' a = lambda_viable`` exactly.
    """

    if return_loadings.ndim != 3:
        raise ValueError("return_loadings must have shape (paths, instruments, factors)")
    if requested_factor_shift.shape != (
        return_loadings.shape[0],
        return_loadings.shape[2],
    ):
        raise ValueError("requested_factor_shift must align with path and factor axes")
    if not 0.0 <= traded_ridge_ratio < 1.0:
        raise ValueError("traded_ridge_ratio must lie in [0, 1)")
    if maximum_factor_shift_norm <= 0.0 or maximum_action_l1 <= 0.0:
        raise ValueError("factor and action limits must be positive")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie strictly between zero and one")

    left, singular_values, right_transpose = torch.linalg.svd(
        return_loadings,
        full_matrices=False,
    )
    largest = singular_values[:, :1]
    retained = singular_values > relative_tolerance * largest
    right_projection = torch.einsum(
        "bfg,bg->bf",
        right_transpose,
        requested_factor_shift,
    )
    ridge = traded_ridge_ratio * largest.square()
    denominator = singular_values.square() + ridge
    mode_shrinkage = torch.where(
        retained,
        singular_values.square() / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
        0.0,
    )
    viable_factor_shift = torch.einsum(
        "bfg,bf->bg",
        right_transpose,
        mode_shrinkage * right_projection,
    )
    action = torch.einsum(
        "bif,bf->bi",
        left,
        torch.where(
            retained,
            singular_values * right_projection
            / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
            0.0,
        ),
    )
    unscaled_action_l1 = torch.sum(torch.abs(action), dim=1)
    factor_norm = torch.linalg.vector_norm(viable_factor_shift, dim=1)
    factor_scale = torch.clamp_max(
        maximum_factor_shift_norm / factor_norm.clamp_min(torch.finfo(factor_norm.dtype).tiny),
        1.0,
    )
    action_after_factor_scale = unscaled_action_l1 * factor_scale
    action_scale = torch.clamp_max(
        maximum_action_l1
        / action_after_factor_scale.clamp_min(torch.finfo(action_after_factor_scale.dtype).tiny),
        1.0,
    )
    total_scale = factor_scale * action_scale
    viable_factor_shift = viable_factor_shift * total_scale[:, None]
    action = action * total_scale[:, None]
    replicated = torch.einsum("bif,bi->bf", return_loadings, action)
    requested_norm = torch.linalg.vector_norm(requested_factor_shift, dim=1)
    retention = torch.linalg.vector_norm(viable_factor_shift, dim=1) / requested_norm.clamp_min(
        torch.finfo(requested_norm.dtype).tiny
    )
    return ViableExponentialAction(
        theoretical_action=TheoreticalExponentialAction(
            action=action,
            replicated_factor_shift=replicated,
            factor_replication_residual=replicated - viable_factor_shift,
            numerical_rank=torch.sum(retained, dim=1),
        ),
        requested_factor_shift=requested_factor_shift,
        viable_factor_shift=viable_factor_shift,
        factor_retention_fraction=torch.where(requested_norm > 0.0, retention, 1.0),
        unscaled_action_l1=unscaled_action_l1,
        total_scale=total_scale,
    )


def exponential_log_density_increment(
    action: torch.Tensor,
    realized_returns: torch.Tensor,
    return_drift_difference: torch.Tensor,
    *,
    time_increment: float,
) -> torch.Tensor:
    r"""Evaluate the normalized Gaussian marginal-utility density increment.

    For :math:`dR=\mu_Rdt+B\,dW^P` and
    :math:`a=\Sigma_R^+\mu_R`, this is

    .. math::
       d\log Z=-a^\top dR+\tfrac12 a^\top\mu_Rdt.

    It is therefore the normalized exponential marginal utility
    :math:`u'(a^\top dR)` expressed solely through traded returns.
    """

    if action.shape != realized_returns.shape or action.shape != return_drift_difference.shape:
        raise ValueError("action, returns, and return drift must have the same shape")
    if time_increment <= 0.0:
        raise ValueError("time_increment must be positive")
    if not bool(
        torch.all(
            torch.isfinite(action)
            & torch.isfinite(realized_returns)
            & torch.isfinite(return_drift_difference)
        )
    ):
        raise ValueError("density inputs must be finite")
    return -torch.sum(action * realized_returns, dim=1) + 0.5 * time_increment * torch.sum(
        action * return_drift_difference,
        dim=1,
    )


__all__ = [
    "TheoreticalExponentialAction",
    "ViableExponentialAction",
    "exponential_log_density_increment",
    "theoretical_exponential_action",
    "viable_exponential_action",
]
