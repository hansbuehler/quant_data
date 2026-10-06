"""Regularized exponential OCE policies for front-spot paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FrontSpotOcePolicyConfig:
    """Identity and geometry of an uncapped affine-plus-deep OCE policy."""

    label: str
    group: str = "sdlv/daily/front-spot/oce"
    user_version: str = "0.1"
    feature_count: int
    instrument_count: int
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=32,
    )

    def __post_init__(self) -> None:
        group_parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid policy label {self.label!r}",
        )
        verify_inp(
            bool(group_parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
            lambda: f"invalid policy group {self.group!r}",
        )
        verify_inp(self.feature_count > 0, lambda: "feature count must be positive")
        verify_inp(
            self.instrument_count > 0,
            lambda: "instrument count must be positive",
        )

    @property
    def filename(self) -> str:
        """Return the stable filename body for this policy."""

        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


class FrontSpotOcePolicy(BaseModel):
    """Uncapped affine-plus-deep policy evaluated at every trading day."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: FrontSpotOcePolicyConfig,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.feature_count,)
        if feature_mean.shape != expected or feature_scale.shape != expected:
            raise ValueError(
                "policy feature normalization must have shape "
                f"{expected}, found {feature_mean.shape} and {feature_scale.shape}"
            )
        if not bool(torch.all(torch.isfinite(feature_mean))):
            raise ValueError("policy feature mean must be finite")
        if not bool(torch.all(torch.isfinite(feature_scale) & (feature_scale > 0.0))):
            raise ValueError("policy feature scale must be finite-positive")
        self.config = config
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.affine = torch.nn.Linear(
            config.feature_count,
            config.instrument_count,
            bias=True,
            dtype=env.fdtype,
            device=env.device,
        )
        self.deep = FeedForward(
            env,
            num_input=config.feature_count,
            num_output=config.instrument_count,
            config=config.network,
        )
        final = self.deep.I0 if len(self.deep.In) == 0 else self.deep.In[-1]
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias.zero_()
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return uncapped signed actions, zeroing unavailable instruments."""

        if features.ndim != 3 or features.shape[-1] != self.config.feature_count:
            raise ValueError(
                "features must have shape (paths,steps,feature_count), found "
                f"{tuple(features.shape)}"
            )
        expected_mask = (features.shape[1], self.config.instrument_count)
        if active_mask.shape != expected_mask:
            raise ValueError(
                f"active mask must have shape {expected_mask}, found {active_mask.shape}"
            )
        normalized = (features - self.feature_mean) / self.feature_scale
        action = self.affine(normalized) + self.deep(normalized)
        action = torch.where(active_mask[None], action, torch.zeros_like(action))
        self.verify(lambda: torch.isfinite(action), lambda: "policy action is not finite")
        return action


def regularized_transition_gain(
    actions: torch.Tensor,
    returns: torch.Tensor,
    *,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Return ``a(X)'R-epsilon*a(X)'a(X)/2`` at each path-time point."""

    if actions.shape != returns.shape or actions.ndim != 3:
        raise ValueError("actions and returns must be matching path-step-instrument tensors")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite-positive, found {epsilon}")
    return torch.sum(
        actions * returns - 0.5 * epsilon * actions.square(),
        dim=-1,
    )


def log_exponential_loss(transition_gain: torch.Tensor) -> torch.Tensor:
    """Return the pooled empirical ``log E[exp(-transition_gain)]``."""

    if transition_gain.numel() == 0:
        raise ValueError("transition gain must be nonempty")
    flat = transition_gain.double().reshape(-1)
    return torch.logsumexp(-flat, dim=0) - math.log(len(flat))


def negative_exponential_statistics(
    transition_gain: NDArray[np.float64],
) -> dict[str, float | dict[str, float]]:
    """Summarize pooled ``-E exp(-gain)`` and its observation-level error."""

    values = np.asarray(transition_gain, dtype=np.float64).reshape(-1)
    if len(values) < 2 or np.any(~np.isfinite(values)):
        raise ValueError("transition gain must contain at least two finite entries")
    log_payoff = -values
    maximum = float(np.max(log_payoff))
    scaled = np.exp(log_payoff - maximum)
    mean_scaled = float(np.mean(scaled, dtype=np.float64))
    scaled_sum = float(np.sum(scaled, dtype=np.float64))
    tilted_effective_sample_size = scaled_sum**2 / float(
        np.sum(scaled**2, dtype=np.float64)
    )
    sorted_scaled = np.sort(scaled)[::-1]
    log_expectation = maximum + math.log(mean_scaled)
    expectation = (
        math.exp(log_expectation)
        if log_expectation <= math.log(np.finfo(np.float64).max)
        else math.inf
    )
    scaled_standard_deviation = float(np.std(scaled, ddof=1))
    if scaled_standard_deviation == 0.0:
        standard_error = 0.0
    else:
        log_standard_error = (
            maximum
            + math.log(scaled_standard_deviation)
            - 0.5 * math.log(len(values))
        )
        standard_error = (
            math.exp(log_standard_error)
            if log_standard_error <= math.log(np.finfo(np.float64).max)
            else math.inf
        )
    return {
        "negative_expectation": -expectation,
        "expectation": expectation,
        "standard_error": standard_error,
        "log_expectation": log_expectation,
        "maximum_log_payoff": maximum,
        "minimum_transition_gain": float(np.min(values)),
        "maximum_transition_gain": float(np.max(values)),
        "tilted_effective_sample_size": tilted_effective_sample_size,
        "largest_tilted_weight_share": float(sorted_scaled[0] / scaled_sum),
        "top_10_tilted_weight_share": float(
            np.sum(sorted_scaled[:10], dtype=np.float64) / scaled_sum
        ),
        "top_100_tilted_weight_share": float(
            np.sum(sorted_scaled[:100], dtype=np.float64) / scaled_sum
        ),
        "log_payoff_quantiles": {
            "50%": float(np.quantile(log_payoff, 0.5)),
            "90%": float(np.quantile(log_payoff, 0.9)),
            "99%": float(np.quantile(log_payoff, 0.99)),
            "99.9%": float(np.quantile(log_payoff, 0.999)),
            "99.99%": float(np.quantile(log_payoff, 0.9999)),
        },
        "transition_gain_quantiles": {
            "0.01%": float(np.quantile(values, 0.0001)),
            "0.1%": float(np.quantile(values, 0.001)),
            "1%": float(np.quantile(values, 0.01)),
            "50%": float(np.quantile(values, 0.5)),
            "99%": float(np.quantile(values, 0.99)),
            "99.9%": float(np.quantile(values, 0.999)),
            "99.99%": float(np.quantile(values, 0.9999)),
        },
        "mean_regularized_transition_gain": float(np.mean(values, dtype=np.float64)),
        "standard_deviation_regularized_transition_gain": float(
            np.std(values, ddof=0)
        ),
    }


def paired_conditional_drift_statistics(
    actions: NDArray[np.float64],
    returns: NDArray[np.float64],
) -> tuple[
    dict[str, float | int | list[float] | dict[str, float]],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return the held-out path and antithetic-pair averages of raw ``a'R``.

    Consecutive paths are the two members of one antithetic pair.  Standard
    errors therefore use pair averages, not pooled path-time observations.
    """

    action = np.asarray(actions, dtype=np.float64)
    instrument_return = np.asarray(returns, dtype=np.float64)
    if action.shape != instrument_return.shape or action.ndim != 3:
        raise ValueError("actions and returns must be matching path-step-instrument tensors")
    if len(action) < 4 or len(action) % 2:
        raise ValueError("paired drift inference requires an even number of at least four paths")
    if np.any(~np.isfinite(action)) or np.any(~np.isfinite(instrument_return)):
        raise ValueError("actions and returns must be finite")
    raw_transition_gain = np.sum(action * instrument_return, axis=-1)
    path_average = np.mean(raw_transition_gain, axis=1, dtype=np.float64)
    summary, pair_average = paired_path_average_statistics(path_average)
    return summary, path_average, pair_average


def paired_path_average_statistics(
    path_average: NDArray[np.float64],
) -> tuple[
    dict[str, float | int | list[float] | dict[str, float]],
    NDArray[np.float64],
]:
    """Compute antithetic-pair inference from consecutive path averages."""

    path_average = np.asarray(path_average, dtype=np.float64)
    if path_average.ndim != 1 or len(path_average) < 4 or len(path_average) % 2:
        raise ValueError("path averages must be an even vector of length at least four")
    if np.any(~np.isfinite(path_average)):
        raise ValueError("path averages must be finite")
    pair_average = np.mean(path_average.reshape(-1, 2), axis=1, dtype=np.float64)
    pair_count = len(pair_average)
    estimate = float(np.mean(pair_average, dtype=np.float64))
    standard_error = float(np.std(pair_average, ddof=1) / math.sqrt(pair_count))
    z_score = estimate / standard_error if standard_error > 0.0 else math.copysign(
        math.inf, estimate
    ) if estimate != 0.0 else 0.0
    return {
        "path_count": len(path_average),
        "antithetic_pair_count": pair_count,
        "mean_daily_conditional_drift": estimate,
        "pair_based_standard_error": standard_error,
        "z_score": z_score,
        "confidence_interval_95": [
            estimate - 1.959963984540054 * standard_error,
            estimate + 1.959963984540054 * standard_error,
        ],
        "path_average_standard_deviation": float(np.std(path_average, ddof=0)),
        "pair_average_standard_deviation": float(np.std(pair_average, ddof=0)),
        "minimum_pair_score": float(np.min(pair_average)),
        "maximum_pair_score": float(np.max(pair_average)),
        "pair_score_quantiles": {
            "0.01%": float(np.quantile(pair_average, 0.0001)),
            "0.1%": float(np.quantile(pair_average, 0.001)),
            "1%": float(np.quantile(pair_average, 0.01)),
            "50%": float(np.quantile(pair_average, 0.5)),
            "99%": float(np.quantile(pair_average, 0.99)),
            "99.9%": float(np.quantile(pair_average, 0.999)),
            "99.99%": float(np.quantile(pair_average, 0.9999)),
        },
    }, pair_average


def action_moments(
    actions: NDArray[np.float64],
) -> tuple[dict[str, float], NDArray[np.float64]]:
    """Return action-energy scalars and pointwise ``E[a_t a_t']``.

    The pointwise expectation uses a uniformly selected path-time pair.  Expired
    instruments must already be zero.  Its trace therefore equals pointwise
    ``E[a_t'a_t]``; multiplying by the number of steps gives pathwise
    ``E[sum_t a_t'a_t]``.
    """

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 3 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("actions must be a finite path-step-instrument tensor")
    path_energy = np.sum(values**2, axis=(1, 2), dtype=np.float64)
    pointwise_outer = np.einsum(
        "pti,ptj->ij",
        values,
        values,
        dtype=np.float64,
        optimize=True,
    ) / (values.shape[0] * values.shape[1])
    pointwise_energy = float(np.trace(pointwise_outer))
    direct_pointwise = float(np.mean(np.sum(values**2, axis=-1), dtype=np.float64))
    if not np.isclose(pointwise_energy, direct_pointwise, rtol=5.0e-12, atol=1.0e-14):
        raise RuntimeError("pointwise action second moment failed its trace identity")
    return (
        {
            "expected_path_action_energy": float(np.mean(path_energy)),
            "standard_error_path_action_energy": (
                float(np.std(path_energy, ddof=1)) / math.sqrt(len(path_energy))
                if len(path_energy) > 1
                else 0.0
            ),
            "expected_pointwise_action_energy": pointwise_energy,
            "maximum_absolute_action": float(np.max(np.abs(values))),
            "rms_action": float(np.sqrt(np.mean(values**2, dtype=np.float64))),
        },
        pointwise_outer,
    )
