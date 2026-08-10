"""Conditional extreme-quantile and lower-CVaR estimation.

For a signed instrument return ``Y`` and lower-tail probability ``alpha``, the
model estimates

``Q_alpha(Y | z) = inf {q: P[Y <= q | z] >= alpha}``.

Both ``Y=dC`` (long) and ``Y=-dC`` (short) are fitted explicitly.  The
associated lower-CVaR functional is represented by the Rockafellar--Uryasev
score

``q - relu(q - Y) / alpha``.

At the population optimum its conditional expectation is the lower expected
shortfall.  A finite empirical distribution has no resolution below one
observation.  Requested levels below ``1 / training_paths`` are consequently
reported but tied to the effective empirical level ``1 / training_paths``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

LONG_DIRECTION = 0
SHORT_DIRECTION = 1
TAIL_DIRECTIONS = ("long_dC", "short_minus_dC")


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class ConditionalTailConfig:
    """Logical geometry and tail levels for the conditional quantile model.

    ``tail_path_counts`` specifies tail masses in units of independent paths.
    For example, with 50,000 training paths, ``10`` means a lower-tail
    probability of ``10 / 50_000``.  Values below one path are retained as
    requested metadata but use one path as their effective empirical level.
    """

    label: str
    state_features: int
    instruments: int
    training_paths: int
    group: str = ""
    user_version: str = "1.0"
    seed: int = 83_771
    tail_path_counts: tuple[float, ...] = (0.1, 1.0, 2.0, 10.0)
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=200,
        activation="LeakyReLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        group = PurePath(self.group)
        verify_inp(
            not group.is_absolute()
            and ".." not in group.parts
            and all(part not in {"", "."} and is_filename(part) for part in group.parts),
            lambda: f"'group' must be a relative directory structure, got {self.group!r}",
        )
        verify_inp(
            isinstance(self.user_version, str) and self.user_version != "",
            "'user_version' must be a nonempty string",
        )
        verify_inp(isinstance(self.seed, int), lambda: f"'seed' must be an integer, got {self.seed!r}")
        for name in ("state_features", "instruments", "training_paths"):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: f"'{name}' must be a positive integer, got {value!r}",
            )
        verify_inp(len(self.tail_path_counts) > 0, "at least one tail level is required")
        previous = -math.inf
        for value in self.tail_path_counts:
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda value=value: f"tail path counts must be finite and positive, got {value}",
            )
            verify_inp(
                value > previous,
                lambda: f"tail path counts must be strictly increasing, got {self.tail_path_counts}",
            )
            verify_inp(
                value <= self.training_paths,
                lambda value=value: (
                    f"tail path count {value} exceeds {self.training_paths} training paths"
                ),
            )
            previous = value

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)

    @property
    def requested_tail_probabilities(self) -> tuple[float, ...]:
        return tuple(value / self.training_paths for value in self.tail_path_counts)

    @property
    def effective_tail_path_counts(self) -> tuple[float, ...]:
        return tuple(max(1.0, value) for value in self.tail_path_counts)

    @property
    def effective_tail_probabilities(self) -> tuple[float, ...]:
        return tuple(value / self.training_paths for value in self.effective_tail_path_counts)

    @property
    def unique_effective_tail_probabilities(self) -> tuple[float, ...]:
        return tuple(dict.fromkeys(self.effective_tail_probabilities))

    @property
    def requested_to_unique_level(self) -> tuple[int, ...]:
        unique = self.unique_effective_tail_probabilities
        return tuple(unique.index(value) for value in self.effective_tail_probabilities)

    def level_metadata(self) -> list[dict[str, Any]]:
        """Return requested and effective probability conventions."""

        return [
            {
                "requested_tail_paths": requested_paths,
                "effective_tail_paths": effective_paths,
                "requested_tail_probability": requested_probability,
                "effective_tail_probability": effective_probability,
                "confidence_level": 1.0 - requested_probability,
                "unique_level_index": unique_index,
                "empirically_distinct": requested_paths >= 1.0,
            }
            for (
                requested_paths,
                effective_paths,
                requested_probability,
                effective_probability,
                unique_index,
            ) in zip(
                self.tail_path_counts,
                self.effective_tail_path_counts,
                self.requested_tail_probabilities,
                self.effective_tail_probabilities,
                self.requested_to_unique_level,
                strict=True,
            )
        ]


def signed_returns(returns: Tensor) -> Tensor:
    """Return explicit long ``dC`` and short ``-dC`` outcomes."""

    verify_inp(
        isinstance(returns, Tensor) and returns.dtype.is_floating_point,
        lambda: f"'returns' must be a floating Torch tensor, got {type(returns)}",
    )
    verify_inp(returns.ndim >= 1, lambda: f"'returns' must have an instrument axis, got {returns.shape}")
    if not torch.isfinite(returns).all().item():
        raise ValueError("'returns' contains non-finite values")
    return torch.stack((returns, -returns), dim=-2)


def lower_quantile_pinball(
    *,
    quantiles: Tensor,
    returns: Tensor,
    tail_probabilities: Tensor,
    normalize_levels: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return normalized pinball loss and its direction/level/instrument means.

    Parameters
    ----------
    quantiles
        Shape ``(..., 2, L, J)`` with long/short as the direction axis.
    returns
        Shape ``(..., J)``.  Long and short targets are constructed internally.
    tail_probabilities
        One-dimensional tensor of the ``L`` effective lower-tail levels.
    normalize_levels
        Divide each level by its tail probability.  This does not change any
        population minimizer, but prevents the least-extreme level from
        mechanically dominating all rarer levels in a joint model.
    """

    verify_inp(
        tail_probabilities.ndim == 1 and tail_probabilities.numel() > 0,
        lambda: f"tail probabilities must be one-dimensional and nonempty, got {tail_probabilities.shape}",
    )
    if not torch.isfinite(quantiles).all().item():
        raise ValueError("quantiles contain non-finite values")
    if not torch.isfinite(tail_probabilities).all().item() or not torch.all(
        (tail_probabilities > 0.0) & (tail_probabilities <= 1.0)
    ).item():
        raise ValueError("tail probabilities must be finite and in (0, 1]")
    targets = signed_returns(returns).unsqueeze(-2)
    expected = returns.shape[:-1] + (2, tail_probabilities.numel(), returns.shape[-1])
    verify_inp(
        quantiles.shape == expected,
        lambda: f"quantile shape {quantiles.shape} does not match expected {expected}",
    )
    return _lower_quantile_pinball_unchecked(
        quantiles=quantiles,
        signed_targets=targets,
        tail_probabilities=tail_probabilities,
        normalize_levels=normalize_levels,
    )


def _lower_quantile_pinball_unchecked(
    *,
    quantiles: Tensor,
    signed_targets: Tensor,
    tail_probabilities: Tensor,
    normalize_levels: bool,
) -> tuple[Tensor, Tensor]:
    """Performance path after dtype, shape, probability, and finiteness checks."""

    alpha = tail_probabilities.reshape((1,) * (quantiles.ndim - 2) + (-1, 1))
    residual = signed_targets - quantiles
    loss = alpha * residual.clamp_min(0.0) + (1.0 - alpha) * (-residual).clamp_min(0.0)
    if normalize_levels:
        loss = loss / alpha
    observation_dims = tuple(range(quantiles.ndim - 3))
    mean_by_output = loss.mean(dim=observation_dims) if observation_dims else loss
    return loss.mean(), mean_by_output


def lower_cvar_score(
    *,
    quantiles: Tensor,
    returns: Tensor,
    tail_probabilities: Tensor,
) -> Tensor:
    """Return the lower-CVaR variational score per observation and output.

    Conditional expectation of this score is maximized by a lower quantile;
    at that optimum it equals lower expected shortfall.
    """

    targets = signed_returns(returns).unsqueeze(-2)
    expected = returns.shape[:-1] + (2, tail_probabilities.numel(), returns.shape[-1])
    verify_inp(
        quantiles.shape == expected,
        lambda: f"quantile shape {quantiles.shape} does not match expected {expected}",
    )
    if not torch.isfinite(tail_probabilities).all().item() or not torch.all(
        (tail_probabilities > 0.0) & (tail_probabilities <= 1.0)
    ).item():
        raise ValueError("tail probabilities must be finite and in (0, 1]")
    alpha = tail_probabilities.reshape((1,) * (quantiles.ndim - 2) + (-1, 1))
    score = quantiles - (quantiles - targets).clamp_min(0.0) / alpha
    if not torch.isfinite(score).all().item():
        raise RuntimeError("lower-CVaR score is non-finite")
    return score


def empirical_lower_cvar(
    values: Tensor,
    *,
    tail_probability: float,
    dim: int = 0,
) -> Tensor:
    """Return exact lower expected shortfall of an empirical distribution.

    Fractional probability mass at the boundary order statistic is included.
    Therefore every level at or below ``1 / observations`` is exactly the
    sample minimum; a fractional observation cannot reveal a lower value.
    """

    verify_inp(values.ndim > 0, "'values' must have at least one dimension")
    verify_inp(
        values.dtype.is_floating_point,
        lambda: f"'values' must have floating dtype, got {values.dtype}",
    )
    verify_inp(
        math.isfinite(tail_probability) and 0.0 < tail_probability <= 1.0,
        lambda: f"'tail_probability' must be in (0, 1], got {tail_probability}",
    )
    if not torch.isfinite(values).all().item():
        raise ValueError("'values' contains non-finite values")
    dim = dim % values.ndim
    observations = values.shape[dim]
    verify_inp(observations > 0, "the empirical tail axis is empty")
    sorted_values = values.sort(dim=dim).values
    tail_mass = tail_probability * observations
    whole = min(int(math.floor(tail_mass)), observations)
    fractional = tail_mass - whole
    if whole == 0:
        return sorted_values.select(dim, 0)
    index = torch.arange(whole, device=values.device)
    total = sorted_values.index_select(dim, index).sum(dim=dim)
    if fractional > 0.0 and whole < observations:
        total = total + fractional * sorted_values.select(dim, whole)
    return total / tail_mass


@version("1.0", dependencies=[ConditionalTailConfig])
class ConditionalTailQuantileModel(BaseModel):
    """State-only monotone lower quantiles for long and short returns."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ConditionalTailConfig,
        state_mean: Tensor,
        state_scale: Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.state_features,)
        verify_inp(
            state_mean.shape == state_scale.shape == expected,
            lambda: (
                f"state normalization must have shape {expected}, found "
                f"{state_mean.shape} and {state_scale.shape}"
            ),
        )
        if not torch.isfinite(state_mean).all().item():
            raise ValueError("state mean contains non-finite values")
        if not torch.isfinite(state_scale).all().item() or not torch.all(state_scale > 0.0).item():
            raise ValueError("state scale must be finite and positive")
        self.config = config
        unique_levels = len(config.unique_effective_tail_probabilities)
        self.network = FeedForward(
            env,
            num_input=config.state_features,
            num_output=(2, unique_levels, config.instruments),
            config=config.network_config,
        )
        self.register_buffer("state_mean", state_mean.detach().clone().to(env.device, env.fdtype))
        self.register_buffer("state_scale", state_scale.detach().clone().to(env.device, env.fdtype))
        self.register_buffer(
            "unique_tail_probabilities",
            torch.as_tensor(
                config.unique_effective_tail_probabilities,
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        self.register_buffer(
            "requested_level_indices",
            torch.as_tensor(
                config.requested_to_unique_level,
                dtype=torch.int64,
                device=env.device,
            ),
        )

    def unique_quantiles(self, z: Tensor) -> Tensor:
        """Return monotone quantiles at empirically distinct tail levels."""

        verify_inp(
            z.shape[-1] == self.config.state_features,
            lambda: f"state shape {z.shape} has the wrong final dimension",
        )
        normalized = (z - self.state_mean) / self.state_scale
        raw = self.network(normalized)
        quantiles = raw.sort(dim=-2).values
        self.verify(lambda: torch.isfinite(quantiles), "conditional quantiles are non-finite")
        return quantiles

    def forward(self, z: Tensor) -> Tensor:
        """Return requested levels, tying levels below one empirical path."""

        unique = self.unique_quantiles(z)
        return unique.index_select(-2, self.requested_level_indices)

    def loss(self, z: Tensor, returns: Tensor) -> tuple[Tensor, Tensor]:
        """Return joint normalized pinball loss at unique effective levels."""

        return lower_quantile_pinball(
            quantiles=self.unique_quantiles(z),
            returns=returns,
            tail_probabilities=self.unique_tail_probabilities,
        )

    def training_loss(self, z: Tensor, returns: Tensor) -> tuple[Tensor, Tensor]:
        """Compiled training kernel after data preparation established invariants."""

        targets = torch.stack((returns, -returns), dim=-2).unsqueeze(-2)
        return _lower_quantile_pinball_unchecked(
            quantiles=self.unique_quantiles(z),
            signed_targets=targets,
            tail_probabilities=self.unique_tail_probabilities,
            normalize_levels=True,
        )

    def cvar_scores(self, z: Tensor, returns: Tensor) -> Tensor:
        """Return requested lower-CVaR scores using effective finite-sample levels."""

        effective = torch.as_tensor(
            self.config.effective_tail_probabilities,
            dtype=z.dtype,
            device=z.device,
        )
        return lower_cvar_score(
            quantiles=self(z),
            returns=returns,
            tail_probabilities=effective,
        )


@torch.no_grad()
def moment_signal_statistics(
    drift: Tensor,
    second_moment: Tensor,
    *,
    g: float = 3.0,
) -> dict[str, Any]:
    """Measure sampled states satisfying ``abs(mu) > g * sigma``.

    The variance is the population conditional variance
    ``max(E[Y**2|z] - E[Y|z]**2, 0)``.  Zero predicted variance is reported
    separately because an infinite signal ratio can indicate inconsistent
    moment-network outputs rather than an economic opportunity.
    """

    verify_inp(
        drift.shape == second_moment.shape and drift.ndim >= 2,
        lambda: f"moment shapes must agree and include an instrument axis, got {drift.shape} and {second_moment.shape}",
    )
    verify_inp(
        math.isfinite(g) and g > 0.0,
        lambda: f"'g' must be finite and positive, got {g}",
    )
    if not torch.isfinite(drift).all().item() or not torch.isfinite(second_moment).all().item():
        raise ValueError("conditional moments contain non-finite values")
    raw_variance = second_moment - drift.square()
    variance = raw_variance.clamp_min(0.0)
    sigma = variance.sqrt()
    hits = drift.abs() > g * sigma
    positive_hits = drift > g * sigma
    negative_hits = drift < -g * sigma
    zero_sigma = sigma == 0.0
    positive_sigma_hits = hits & ~zero_sigma
    ratio = torch.where(
        zero_sigma,
        torch.where(drift == 0.0, torch.zeros_like(drift), torch.full_like(drift, torch.inf)),
        drift.abs() / sigma,
    )
    observation_dims = tuple(range(drift.ndim - 1))
    states_any = hits.any(dim=-1)
    comparisons = hits.numel()
    finite_ratio = ratio[torch.isfinite(ratio)]
    per_instrument_hits = hits.sum(dim=observation_dims)
    observations_per_instrument = drift[..., 0].numel()
    return {
        "g": float(g),
        "comparisons": comparisons,
        "states": states_any.numel(),
        "instruments": drift.shape[-1],
        "hit_comparisons": int(hits.sum().item()),
        "hit_comparison_fraction": float(hits.to(torch.float64).mean().item()),
        "states_with_any_hit": int(states_any.sum().item()),
        "states_with_any_hit_fraction": float(states_any.to(torch.float64).mean().item()),
        "instruments_with_any_hit": int((per_instrument_hits > 0).sum().item()),
        "positive_hit_comparisons": int(positive_hits.sum().item()),
        "negative_hit_comparisons": int(negative_hits.sum().item()),
        "positive_sigma_hit_comparisons": int(positive_sigma_hits.sum().item()),
        "positive_sigma_hit_comparison_fraction": float(
            positive_sigma_hits.to(torch.float64).mean().item()
        ),
        "raw_negative_variance_comparisons": int((raw_variance < 0.0).sum().item()),
        "zero_sigma_comparisons": int(zero_sigma.sum().item()),
        "zero_sigma_nonzero_mean_comparisons": int((zero_sigma & (drift != 0.0)).sum().item()),
        "max_finite_abs_mean_over_sigma": (
            None if finite_ratio.numel() == 0 else float(finite_ratio.max().item())
        ),
        "per_instrument_hit_count": per_instrument_hits.cpu().tolist(),
        "per_instrument_hit_fraction": (
            per_instrument_hits.to(torch.float64) / float(observations_per_instrument)
        ).cpu().tolist(),
        "per_instrument_positive_sigma_hit_count": positive_sigma_hits.sum(
            dim=observation_dims
        ).cpu().tolist(),
    }


__all__ = [
    "ConditionalTailConfig",
    "ConditionalTailQuantileModel",
    "LONG_DIRECTION",
    "SHORT_DIRECTION",
    "TAIL_DIRECTIONS",
    "empirical_lower_cvar",
    "lower_cvar_score",
    "lower_quantile_pinball",
    "moment_signal_statistics",
    "signed_returns",
]
