"""Conditional dispersion checks for the VolGAN surface generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import VolganTrainingData
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.hedger import (
    sample_volgan_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.training import (
    VolganTrainingResult,
)


@dataclass(frozen=True, kw_only=True)
class VolganConditionalAudit:
    """Out-of-sample residual scale versus within-state generated scale."""

    validation_rows: int
    generated_states: int
    paths_per_state: int
    level_train_explained_fraction: float | None
    increment_train_explained_fraction: float | None
    level_validation_explained_fraction: float | None
    increment_validation_explained_fraction: float | None
    historical_spot_residual_std_daily: float
    generated_spot_conditional_std_daily: float
    historical_iv_residual_rms: float
    generated_iv_conditional_rms: float
    historical_spot_first_iv_pc_correlation: float | None
    generated_spot_first_iv_pc_correlation: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def audit_volgan_conditional_dispersion(
    data: VolganTrainingData,
    fit: VolganTrainingResult,
    *,
    validation_fraction: float,
    maximum_states: int = 20,
    paths_per_state: int = 256,
    seed: int = 20261024,
) -> VolganConditionalAudit:
    """Compare generator shocks with validation residuals of a train-only ridge predictor.

    The predictor is fitted on the chronological training block in the same
    latent condition/target coordinates as the GAN. Historical residuals are
    out of sample; generated dispersion is measured within each fixed state.
    """
    if maximum_states < 1 or paths_per_state < 2:
        raise ValueError("VolGAN conditional audit requires states and at least two paths")
    generator = fit.generator
    train_count = len(data.conditions) - max(1, int(round(len(data.conditions) * validation_fraction)))
    if not 2 <= train_count < len(data.conditions):
        raise ValueError("VolGAN conditional audit has an invalid chronological split")
    level_mean = (
        None if generator.config.pca_rank is None else generator.level_mean.cpu().numpy()
    )
    level_basis = (
        None if generator.config.pca_rank is None else generator.level_components.cpu().numpy()
    )
    increment_mean = (
        None if generator.config.pca_rank is None else generator.increment_mean.cpu().numpy()
    )
    increment_basis = (
        None if generator.config.pca_rank is None else generator.increment_components.cpu().numpy()
    )
    conditions = data.conditions.copy()
    targets = data.targets.copy()
    if level_basis is not None:
        conditions = np.concatenate(
            (conditions[:, :3], (conditions[:, 3:] - level_mean) @ level_basis.T), axis=1
        )
        targets = np.concatenate(
            (targets[:, :1], (targets[:, 1:] - increment_mean) @ increment_basis.T), axis=1
        )
    condition_mean = generator.condition_mean.cpu().numpy()
    condition_scale = generator.condition_scale.cpu().numpy()
    normalized_conditions = (conditions - condition_mean) / condition_scale
    train_design = np.column_stack((np.ones(train_count), normalized_conditions[:train_count]))
    penalty = np.eye(train_design.shape[1]) * train_count
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ targets[:train_count],
    )
    validation_design = np.column_stack(
        (np.ones(len(data.conditions) - train_count), normalized_conditions[train_count:])
    )
    residual = targets[train_count:] - validation_design @ coefficients
    historical_spot = residual[:, 0] / np.sqrt(252.0)
    if increment_basis is None:
        historical_iv = residual[:, 1:]
    else:
        historical_iv = residual[:, 1:] @ increment_basis

    selected = np.linspace(
        train_count, len(data.conditions) - 1,
        num=min(maximum_states, len(data.conditions) - train_count),
        dtype=int,
    )
    spot_shocks: list[np.ndarray] = []
    iv_shocks: list[np.ndarray] = []
    generated_first_scores: list[np.ndarray] = []
    for index in selected:
        generated = sample_volgan_transitions(
            fit,
            condition=data.conditions[index],
            path_count=paths_per_state,
            seed=seed + int(index),
        )
        spot = generated.forward_log_returns
        increment = generated.next_log_implied_volatilities.reshape(paths_per_state, -1) - data.conditions[index, 3:]
        spot_shocks.append(spot - np.mean(spot))
        iv_shocks.append(increment - np.mean(increment, axis=0))
        if increment_basis is not None:
            generated_first_scores.append(iv_shocks[-1] @ increment_basis[0])
    generated_spot = np.stack(spot_shocks)
    generated_iv = np.stack(iv_shocks)
    generated_first = (
        None if increment_basis is None else np.stack(generated_first_scores).ravel()
    )
    historical_first = None if increment_basis is None else residual[:, 1]
    return VolganConditionalAudit(
        validation_rows=len(residual),
        generated_states=len(selected),
        paths_per_state=paths_per_state,
        level_train_explained_fraction=fit.level_pca_explained_fraction,
        increment_train_explained_fraction=fit.increment_pca_explained_fraction,
        level_validation_explained_fraction=_validation_explained(
            data.conditions[train_count:, 3:], level_mean, level_basis
        ),
        increment_validation_explained_fraction=_validation_explained(
            data.targets[train_count:, 1:], increment_mean, increment_basis
        ),
        historical_spot_residual_std_daily=float(np.std(historical_spot, ddof=1)),
        generated_spot_conditional_std_daily=float(
            np.mean(np.std(generated_spot, axis=1, ddof=1))
        ),
        historical_iv_residual_rms=float(
            np.sqrt(np.mean(np.var(historical_iv, axis=0, ddof=1)))
        ),
        generated_iv_conditional_rms=float(
            np.mean(np.sqrt(np.mean(np.var(generated_iv, axis=1, ddof=1), axis=1)))
        ),
        historical_spot_first_iv_pc_correlation=_correlation(historical_spot, historical_first),
        generated_spot_first_iv_pc_correlation=_correlation(generated_spot.ravel(), generated_first),
    )


def _validation_explained(
    values: np.ndarray, mean: np.ndarray | None, components: np.ndarray | None
) -> float | None:
    if components is None or mean is None:
        return None
    centered = values - mean
    reconstructed = (centered @ components.T) @ components
    energy = float(np.sum(centered**2))
    return 1.0 - float(np.sum((centered - reconstructed) ** 2) / energy) if energy > 0 else 1.0


def _correlation(first: np.ndarray, second: np.ndarray | None) -> float | None:
    if second is None or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])
