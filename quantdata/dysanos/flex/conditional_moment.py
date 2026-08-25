"""Direct conditional option-moment objectives for sampleable weekly laws."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from quantdata.dysanos.flex.oce_transport import (
    ConditionalPathBatch,
    SpotConditionedFirstLineTransport,
    apply_first_line_transport_sequence,
    w2_roll_returns,
)
from quantdata.dysanos.flex.weekly_measure import GenericWeeklyMeasureKernel
from quantdata.dysanos.flex.weekly_measure_training import WeeklyMeasureTrainingData
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator


def squared_conditional_moment_loss(
    returns: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return mean squared whitened moments, raw moments, and whitened moments.

    ``returns`` has axes ``(origin, spot, surface_path, instrument)``.  The
    expectation is conditional on the current origin but integrates both the
    implied next-spot marginal and the conditional surface innovation.  Zero
    moment for every instrument implies zero conditional gain for every
    predictable linear portfolio of those instruments.
    """

    if (
        returns.ndim != 4
        or inverse_sqrt_second_moment.shape
        != (returns.shape[-1], returns.shape[-1])
    ):
        raise ValueError("conditional-moment loss has inconsistent geometry")
    moments = torch.mean(returns, dim=(1, 2))
    whitened_moments = moments @ inverse_sqrt_second_moment
    return (
        torch.mean(whitened_moments.square()),
        moments,
        whitened_moments,
    )


def replicated_conditional_moment_statistics(
    replicate_moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate conditional moments and Monte Carlo errors across replications.

    ``replicate_moments`` has axes ``(origin, replication, instrument)``. Each
    replication must use an independent randomized spot grid and independent
    antithetic surface normals. The returned standard error is the sample
    standard deviation of those independent replication estimates divided by
    the square root of the replication count.
    """

    if (
        replicate_moments.ndim != 3
        or replicate_moments.shape[0] < 1
        or replicate_moments.shape[1] < 2
        or replicate_moments.shape[2] < 1
        or not replicate_moments.is_floating_point()
        or not bool(torch.all(torch.isfinite(replicate_moments)))
    ):
        raise ValueError(
            "replicated moments must be a finite floating tensor with geometry "
            "(origin,replication,instrument) and at least two replications"
        )
    replications = replicate_moments.shape[1]
    estimate = torch.mean(replicate_moments, dim=1)
    standard_error = torch.std(
        replicate_moments,
        dim=1,
        correction=1,
    ) / math.sqrt(replications)
    return estimate, standard_error


def standard_normal_transport_kl(
    base: torch.Tensor,
    transformed: torch.Tensor,
    log_abs_det: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``KL(U#N || N)`` and its sampled path contributions.

    The conditional invertible map is ``U``.  For ``z ~ N(0,I)`` and
    ``u=U(z)``, change of variables gives

    ``log d(U#N)/dN(u) = 0.5*(||u||^2-||z||^2) - log|det J_U(z)|``.
    """

    if (
        base.shape != transformed.shape
        or base.ndim < 2
        or log_abs_det.shape != base.shape[:-1]
    ):
        raise ValueError("standard-normal transport KL has inconsistent geometry")
    path_kl = (
        0.5 * torch.sum(transformed.square() - base.square(), dim=-1)
        - log_abs_det
    )
    return torch.mean(path_kl), path_kl


def direct_w2_conditional_moment_transport_loss(
    *,
    incremental_kernel: SpotConditionedFirstLineTransport,
    current_kernels: Sequence[GenericWeeklyMeasureKernel],
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    current_log_spot: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train an incremental sampleable law on W2 option moments directly."""

    if len(current_kernels) < 1:
        raise ValueError("direct W2 moment training requires a source law")
    if current_log_spot.shape != (batch.num_origins,):
        raise ValueError("direct W2 moment current log spot has invalid shape")
    if not math.isfinite(kl_weight) or kl_weight < 0.0:
        raise ValueError("direct W2 moment KL weight must be finite-nonnegative")
    context_log_spot = torch.repeat_interleave(
        current_log_spot,
        batch.spot_contexts_per_origin,
    )
    path_log_spot = torch.repeat_interleave(
        context_log_spot,
        batch.surface_paths_per_context,
    )
    incremental_draw, incremental_log_abs_det = (
        incremental_kernel.step_first_line_with_log_abs_det(
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
            path_log_spot,
        )
    )
    candidate_draw, _ = apply_first_line_transport_sequence(
        current_kernels,
        batch.current_state,
        batch.gross_spot_return,
        incremental_draw.transformed_standard_normals,
        path_log_spot,
    )
    candidate_returns = w2_roll_returns(
        simulator,
        data,
        batch,
        candidate_draw.reduced_state,
        normalized_strikes,
    ).reshape(
        batch.num_origins,
        batch.spot_contexts_per_origin,
        batch.surface_paths_per_context,
        -1,
    )
    moment_loss, moments, whitened_moments = squared_conditional_moment_loss(
        candidate_returns,
        inverse_sqrt_second_moment,
    )
    width = incremental_kernel.config.state_sizes_by_expiry[0]
    base = batch.standard_normals[:, 1 : 1 + width]
    transformed = incremental_draw.transformed_standard_normals[:, 1 : 1 + width]
    incremental_kl, path_kl = standard_normal_transport_kl(
        base,
        transformed,
        incremental_log_abs_det,
    )
    return moment_loss + kl_weight * incremental_kl, {
        "candidate_returns": candidate_returns,
        "conditional_moments": moments,
        "whitened_conditional_moments": whitened_moments,
        "moment_loss": moment_loss,
        "incremental_kl": incremental_kl,
        "incremental_path_kl": path_kl,
        "incremental_log_abs_det": incremental_log_abs_det,
    }


__all__ = [
    "direct_w2_conditional_moment_transport_loss",
    "replicated_conditional_moment_statistics",
    "squared_conditional_moment_loss",
    "standard_normal_transport_kl",
]
