"""Leakage-safe daily sequence data for conditional DYSANOS innovations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular

from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.factorial_statistics import AUDIT_BUSINESS_DTES
from quantdata.dysanos.v11.native_atm_history import load_atm_history


@dataclass(frozen=True)
class ConditionalSequenceData:
    """Frozen features/targets with a chronological validation split."""

    sequences: np.ndarray
    current_spot_scores: np.ndarray
    observed_white: np.ndarray
    white_mean_basis: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    state_mean: np.ndarray
    state_basis: np.ndarray
    initial_context: np.ndarray
    initial_state: np.ndarray
    transition_indices: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    dates: tuple[str, ...]


def _principal_directions(values: np.ndarray, rank: int) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    if rank > len(right):
        raise ValueError("requested factor rank exceeds sample rank")
    return right[:rank].T.copy()


def prepare_conditional_sequence_data(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    dynamics_parameters: Path,
    native_atm_history: Path,
    historical_dataset: Path,
    sequence_length: int = 20,
    mean_rank: int = 8,
    validation_transitions: int = 126,
    device: str = "cpu",
) -> ConditionalSequenceData:
    """Build one-step features using only information available by each origin."""
    model, _ = load_capped_joint_model(joint_checkpoint, device=device)
    payload = torch.load(target_history, map_location="cpu", weights_only=False)
    dates = tuple(payload["dates"])
    states = payload["states"].double().numpy()
    observed_dates, business_dtes, atm_iv, simple_spot = load_atm_history(
        history_path=native_atm_history, dataset_path=historical_dataset
    )
    if (
        states.shape != tuple(model.daily_states.shape)
        or list(observed_dates[: len(dates)]) != list(dates)
        or sequence_length < 2
        or validation_transitions < 30
        or len(states) - 1 <= validation_transitions + sequence_length
    ):
        raise ValueError("daily states, native history or sequence split invalid")
    tenor_indices = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(business_dtes, tenor, atol=1.0e-9))
        if len(matches) != 1:
            raise ValueError(f"native ATM history misses {tenor}D")
        tenor_indices.append(int(matches[0]))
    current_spot = _historical_spot_scores(
        model,
        states[:-1],
        1.0 + simple_spot[: len(states) - 1],
        batch_size=128,
    )
    with np.load(dynamics_parameters, allow_pickle=False) as stream:
        parameters = {name: stream[name] for name in stream.files}
    mean = parameters["equilibrium_mean"]
    transition = parameters["transition"]
    factor = parameters["innovation_cholesky"]
    loading = parameters["spot_loading"]
    score_mean = float(parameters["training_spot_score_mean"])
    baseline_mean = (
        mean + (states[:-1] - mean) @ transition
        + (current_spot - score_mean)[:, None] * loading[None, :]
    )
    observed_white = solve_triangular(
        factor, (states[1:] - baseline_mean).T, lower=True
    ).T
    transition_count = len(observed_white)
    training_cut = transition_count - validation_transitions
    if np.any(~np.isfinite(observed_white)) or np.any(~np.isfinite(current_spot)):
        raise ValueError("conditional sequence residuals are not finite")
    state_mean = np.mean(states[: training_cut + 1], axis=0)
    state_basis = _principal_directions(states[: training_cut + 1], mean_rank)
    white_basis = _principal_directions(observed_white[:training_cut], mean_rank)
    state_scores = (states[:-1] - state_mean) @ state_basis
    log_iv = np.log(atm_iv[: len(states), tenor_indices])
    iv_change = np.diff(log_iv, axis=0)
    prior_spot = np.concatenate(([0.0], current_spot[:-1]))
    prior_log_radius = np.concatenate(
        ([0.0], np.log(np.maximum(np.mean(observed_white[:-1] ** 2, axis=1), 1.0e-8)))
    )
    prior_iv_change = np.vstack((np.zeros((1, len(tenor_indices))), iv_change[:-1]))
    features = np.column_stack(
        (state_scores, prior_spot, prior_log_radius, prior_iv_change)
    )
    transition_indices = np.arange(sequence_length - 1, transition_count)
    sequences = np.stack(
        [features[index - sequence_length + 1 : index + 1] for index in transition_indices]
    )
    train_mask = transition_indices < training_cut
    validation_mask = ~train_mask
    if np.sum(train_mask) < 100 or np.sum(validation_mask) < 30:
        raise ValueError("conditional sequence split is too small")
    feature_mean = np.mean(sequences[train_mask], axis=(0, 1))
    feature_std = np.maximum(np.std(sequences[train_mask], axis=(0, 1)), 1.0e-4)
    sequences = (sequences - feature_mean) / feature_std
    final_origin_feature = np.concatenate(
        (
            (states[-1] - state_mean) @ state_basis,
            np.asarray([current_spot[-1]]),
            np.asarray([np.log(max(np.mean(observed_white[-1] ** 2), 1.0e-8))]),
            iv_change[-1],
        )
    )
    initial_context = np.vstack(
        (features[-(sequence_length - 1):], final_origin_feature)
    )
    initial_context = (initial_context - feature_mean) / feature_std
    return ConditionalSequenceData(
        sequences=sequences.astype(np.float32),
        current_spot_scores=current_spot[transition_indices].astype(np.float32),
        observed_white=observed_white[transition_indices].astype(np.float32),
        white_mean_basis=white_basis.astype(np.float32),
        feature_mean=feature_mean,
        feature_std=feature_std,
        state_mean=state_mean,
        state_basis=state_basis,
        initial_context=initial_context.astype(np.float32),
        initial_state=states[-1].astype(np.float32),
        transition_indices=transition_indices,
        train_mask=train_mask,
        validation_mask=validation_mask,
        dates=dates,
    )


__all__ = ["ConditionalSequenceData", "prepare_conditional_sequence_data"]
