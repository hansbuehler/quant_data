"""Orthogonal-state direction and notional tracing for conditional arbitrage.

Arb tracing separates the conditional portfolio into a bounded direction
function and an external scalar notional.  A direction is learned at a
moderate probe notional, after which its notional is traced explicitly.  A
finite utility optimum identifies a risky trade; a direction whose notional
can keep increasing is passed to the empirical no-loss audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from scipy.special import logsumexp
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

from .required_runs import clopper_pearson_one_sided

ArbTracingUtility = Literal["powerexp", "entropy"]


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class ArbTracingConfig:
    """Logical geometry and objective of one arb-tracing direction."""

    label: str
    source_cache_uid: str
    state_features: int
    risk_aversion: float
    utility: ArbTracingUtility
    trading_times: int = 10
    expiries: int = 7
    instruments_per_expiry: int = 21
    probe_notional: float = 2.0
    raw_norm_penalty: float = 1.0e-5
    raw_norm_soft_limit: float = 8.0
    covariance_relative_floor: float = 1.0e-10
    group: str = ""
    user_version: str = "0.1"
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=96,
        activation="SiLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        verify_inp(
            isinstance(self.group, str) and not self.group.startswith("/"),
            lambda: f"'group' must be relative, got {self.group!r}",
        )
        verify_inp(
            isinstance(self.source_cache_uid, str) and bool(self.source_cache_uid),
            "'source_cache_uid' must be a non-empty string",
        )
        verify_inp(
            self.utility in {"powerexp", "entropy"},
            lambda: f"unknown utility {self.utility!r}",
        )
        for name in (
            "state_features",
            "trading_times",
            "expiries",
            "instruments_per_expiry",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        for name in (
            "risk_aversion",
            "probe_notional",
            "raw_norm_soft_limit",
            "covariance_relative_floor",
        ):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value!r}"
                ),
            )
        verify_inp(
            math.isfinite(self.raw_norm_penalty) and self.raw_norm_penalty >= 0.0,
            lambda: f"invalid raw-norm penalty {self.raw_norm_penalty!r}",
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class ArbTracingTrainConfig:
    """Operational settings which permit checkpoint warm starts."""

    batch_size: int = 4_096
    valuation_batch_size: int = 16_384
    epochs: int = 20_000
    minimum_epochs: int = 4_000
    value_update: int = 250
    checkpoint_update: int = 2_000
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    compile: bool = True
    convergence_points: int = 12
    convergence_relative_tolerance: float = 1.0e-4
    seed: int = 91_301

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "epochs",
            "minimum_epochs",
            "value_update",
            "checkpoint_update",
            "convergence_points",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            self.minimum_epochs <= self.epochs,
            "minimum epochs cannot exceed maximum epochs",
        )
        for name in ("lr", "convergence_relative_tolerance"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value!r}"
                ),
            )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: f"invalid weight decay {self.weight_decay!r}",
        )


def fit_physical_state_whitener(
    state: Tensor,
    path_weight: Tensor,
    indices: Tensor,
    *,
    relative_eigenvalue_floor: float = 1.0e-10,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Fit a time-specific physical-measure orthogonal state basis.

    The cached market is sampled under an importance proposal.  Consequently
    the physical state covariance is estimated with ``p/q`` weights, using
    training paths only.  A caller may replace these tensors with analytic
    Gaussian moments when they are persisted with a future universe cache.
    """

    verify_inp(state.ndim == 3, lambda: f"state must have shape (S,T,F), got {state.shape}")
    verify_inp(
        path_weight.shape == (state.shape[0],),
        lambda: f"path weights have invalid shape {path_weight.shape}",
    )
    verify_inp(
        indices.ndim == 1 and indices.numel() >= 2,
        lambda: f"whitening indices have invalid shape {indices.shape}",
    )
    verify_inp(
        math.isfinite(relative_eigenvalue_floor) and relative_eigenvalue_floor > 0.0,
        lambda: f"invalid eigenvalue floor {relative_eigenvalue_floor!r}",
    )

    sample = state[indices].double()
    weight = path_weight[indices].double()
    if not torch.isfinite(sample).all().item():
        raise FloatingPointError("state whitening sample contains NaN or infinity")
    if not torch.isfinite(weight).all().item() or not torch.all(weight > 0.0).item():
        raise FloatingPointError("state whitening weights are not finite-positive")
    normalized_weight = weight / weight.sum()
    mean = torch.einsum("s,stf->tf", normalized_weight, sample)
    centered = sample - mean[None, :, :]
    covariance = torch.einsum(
        "s,stf,stg->tfg", normalized_weight, centered, centered
    )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalue, eigenvector = torch.linalg.eigh(covariance)
    maximum = eigenvalue[:, -1].clamp_min(torch.finfo(eigenvalue.dtype).tiny)
    floor = relative_eigenvalue_floor * maximum
    clipped = torch.maximum(eigenvalue, floor[:, None])
    # Rows are observations.  Multiplying by Q Lambda^{-1/2} produces factors.
    whitener = eigenvector * torch.rsqrt(clipped)[:, None, :]
    factor = torch.einsum("stf,tfg->stg", centered, whitener)
    factor_mean = torch.einsum("s,stf->tf", normalized_weight, factor)
    factor_centered = factor - factor_mean[None, :, :]
    factor_covariance = torch.einsum(
        "s,stf,stg->tfg", normalized_weight, factor_centered, factor_centered
    )
    identity = torch.eye(
        state.shape[-1], dtype=factor_covariance.dtype, device=state.device
    )
    residual = factor_covariance - identity[None, :, :]
    diagnostics: dict[str, object] = {
        "definition": (
            "xi_t=(x_t-m_t) Q_t Lambda_t^{-1/2}; m_t and covariance are "
            "estimated on training paths with physical p/q weights"
        ),
        "relative_eigenvalue_floor": relative_eigenvalue_floor,
        "training_path_count": int(indices.numel()),
        "training_weight_effective_sample_size": float(
            weight.sum().square().div(weight.square().sum()).cpu()
        ),
        "minimum_covariance_eigenvalue_by_time": eigenvalue[:, 0].cpu().tolist(),
        "maximum_covariance_eigenvalue_by_time": eigenvalue[:, -1].cpu().tolist(),
        "clipped_eigenvalue_count_by_time": (eigenvalue < floor[:, None])
        .sum(dim=1)
        .cpu()
        .tolist(),
        "maximum_absolute_weighted_factor_mean": float(factor_mean.abs().max().cpu()),
        "maximum_absolute_weighted_factor_covariance_error": float(
            residual.abs().max().cpu()
        ),
    }
    result_mean = mean.to(dtype=state.dtype)
    result_whitener = whitener.to(dtype=state.dtype)
    if not torch.isfinite(result_mean).all().item() or not torch.isfinite(
        result_whitener
    ).all().item():
        raise FloatingPointError("state whitener contains NaN or infinity")
    return result_mean, result_whitener, diagnostics


@version("1.0")
class OrthogonalArbDirection(BaseModel):
    """Independent per-time direction networks in whitened state coordinates."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ArbTracingConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        super().__init__(env)
        expected_mean = (config.trading_times, config.state_features)
        expected_whitener = (
            config.trading_times,
            config.state_features,
            config.state_features,
        )
        verify_inp(
            state_mean.shape == expected_mean,
            lambda: f"state mean must have shape {expected_mean}, got {state_mean.shape}",
        )
        verify_inp(
            state_whitener.shape == expected_whitener,
            lambda: (
                f"state whitener must have shape {expected_whitener}, "
                f"got {state_whitener.shape}"
            ),
        )
        for name, value in (("state mean", state_mean), ("state whitener", state_whitener)):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains NaN or infinity")
        self.config = config
        instruments = config.expiries * config.instruments_per_expiry
        self.networks = torch.nn.ModuleList(
            FeedForward(
                env,
                num_input=config.state_features,
                num_output=instruments,
                config=config.network_config,
            )
            for _ in range(config.trading_times)
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_whitener", state_whitener.detach().clone())
        self.zero_output_layers()

    @torch.no_grad()
    def zero_output_layers(self) -> None:
        """Start at no trade while preserving the local output derivative."""

        for network in self.networks:
            output_layer = network.In[-1] if network.In else network.I0
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        expected = (
            state.ndim == 3
            and state.shape[1] == self.config.trading_times
            and state.shape[2] == self.config.state_features
        )
        assert expected, (
            "state must have shape (S,trading_times,state_features), found "
            f"{tuple(state.shape)}"
        )
        centered = state - self.state_mean[None, :, :]
        factor = torch.einsum("btf,tfg->btg", centered, self.state_whitener)
        raw = torch.stack(
            [network(factor[:, time]) for time, network in enumerate(self.networks)],
            dim=1,
        ).reshape(
            state.shape[0],
            self.config.trading_times,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        raw_l1 = raw.abs().sum(dim=(-1, -2))
        # The map has no exterior flat region.  Scale is deliberately excluded
        # from the network and is searched as a separate scalar per time.
        direction = raw / (1.0 + raw_l1[..., None, None])
        self.verify(lambda: torch.isfinite(factor), "orthogonal state factor is non-finite")
        self.verify(lambda: torch.isfinite(raw), "raw arb direction is non-finite")
        self.verify(lambda: torch.isfinite(direction), "arb direction is non-finite")
        return direction, raw_l1


def tracing_pnl(action: Tensor, returns: Tensor, costs: Tensor) -> Tensor:
    """Return exact net P&L for each path and trading time."""

    assert action.shape == returns.shape == costs.shape
    pnl = (action * returns - action.abs() * costs).sum(dim=(-1, -2))
    if not torch.isfinite(pnl).all().item():
        raise FloatingPointError("arb-tracing P&L contains NaN or infinity")
    return pnl


def tracing_pnl_right_derivative(
    base_action: Tensor,
    direction: Tensor,
    notional: float,
    returns: Tensor,
    costs: Tensor,
) -> Tensor:
    """Return the right derivative of exact net P&L along one direction."""

    assert base_action.shape == direction.shape == returns.shape == costs.shape
    verify_inp(
        math.isfinite(notional) and notional >= 0.0,
        lambda: f"invalid tracing notional {notional!r}",
    )
    action = base_action + notional * direction
    zero = action == 0.0
    cost_derivative = torch.where(
        zero,
        direction.abs(),
        torch.sign(action) * direction,
    )
    result = (direction * returns - cost_derivative * costs).sum(dim=(-1, -2))
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("arb-tracing P&L derivative contains NaN or infinity")
    return result


def tracing_numerical_band(
    action: Tensor,
    returns: Tensor,
    costs: Tensor,
    instrument_band: Tensor,
    *,
    roundoff_multiplier: float = 32.0,
) -> Tensor:
    """Propagate instrument arithmetic bands through a traced portfolio."""

    assert action.shape == returns.shape == costs.shape == instrument_band.shape
    absolute_action = action.abs()
    eps32 = torch.finfo(torch.float32).eps
    result = (absolute_action * instrument_band).sum(dim=(-1, -2)) + (
        roundoff_multiplier
        * eps32
        * (absolute_action * (returns.abs() + costs)).sum(dim=(-1, -2))
    )
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("arb-tracing numerical band contains NaN or infinity")
    return result


def tracing_utility_objective(
    pnl: Tensor,
    path_weight: Tensor,
    *,
    utility: ArbTracingUtility,
    risk_aversion: float,
) -> Tensor:
    """Return the physical utility objective separately by trading time."""

    assert pnl.ndim == 2 and path_weight.shape == pnl.shape[:1]
    verify_inp(utility in {"powerexp", "entropy"}, lambda: f"unknown utility {utility}")
    verify_inp(
        math.isfinite(risk_aversion) and risk_aversion > 0.0,
        lambda: f"invalid risk aversion {risk_aversion!r}",
    )
    if not torch.isfinite(pnl).all().item():
        raise FloatingPointError("utility P&L contains NaN or infinity")
    if not torch.isfinite(path_weight).all().item() or not torch.all(path_weight > 0.0).item():
        raise FloatingPointError("utility weights are not finite-positive")
    if utility == "entropy":
        log_weight = torch.log(path_weight)
        log_normalizer = torch.logsumexp(log_weight, dim=0)
        log_moment = torch.logsumexp(
            log_weight[:, None] - risk_aversion * pnl, dim=0
        ) - log_normalizer
        value = -log_moment / risk_aversion
    else:
        positive = torch.clamp_min(pnl, 0.0)
        negative = torch.clamp_max(pnl, 0.0)
        value_by_path = torch.where(
            pnl > 0.0,
            -torch.expm1(-risk_aversion * positive) / risk_aversion,
            negative - 0.5 * risk_aversion * negative.square(),
        )
        value = (path_weight[:, None] * value_by_path).sum(dim=0) / path_weight.sum()
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{utility} tracing objective contains NaN or infinity")
    return value


def tracing_raw_norm_penalty(
    raw_l1: Tensor,
    *,
    soft_limit: float,
) -> Tensor:
    """Keep the direction map away from its asymptotically flat radial tail."""

    verify_inp(soft_limit > 0.0 and math.isfinite(soft_limit), "invalid soft limit")
    return torch.relu(raw_l1 - soft_limit).square().mean(dim=0)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    """Return a linearly interpolated quantile of a finite weighted sample."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    verify_inp(
        values.ndim == weights.ndim == 1 and values.shape == weights.shape,
        "weighted quantile inputs must be equal-length vectors",
    )
    verify_inp(0.0 <= probability <= 1.0, lambda: f"invalid probability {probability}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise FloatingPointError("weighted quantile input contains NaN or infinity")
    verify_inp(np.all(weights >= 0.0) and np.any(weights > 0.0), "invalid weights")
    order = np.argsort(values, kind="stable")
    sorted_value = values[order]
    sorted_weight = weights[order]
    midpoint = np.cumsum(sorted_weight) - 0.5 * sorted_weight
    midpoint /= sorted_weight.sum()
    return float(np.interp(probability, midpoint, sorted_value))


def _weighted_mean_error(values: np.ndarray, normalized_weight: np.ndarray) -> tuple[float, float]:
    mean = float(np.dot(normalized_weight, values))
    error = float(
        np.sqrt(np.sum(np.square(normalized_weight * (values - mean))))
    )
    return mean, error


def traced_scale_statistics(
    pnl: np.ndarray,
    band: np.ndarray,
    slope: np.ndarray,
    path_weight: np.ndarray,
    component: np.ndarray,
    *,
    utility: ArbTracingUtility,
    risk_aversion: float,
    loss_probability_epsilon: float,
    confidence_z: float = 3.0,
    loss_mc_multiplier: float = 1.0,
) -> list[dict[str, float | int | bool]]:
    """Measure utility, marginal scale value, losses, and finite-risk quality."""

    pnl = np.asarray(pnl, dtype=np.float64)
    band = np.asarray(band, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    path_weight = np.asarray(path_weight, dtype=np.float64)
    component = np.asarray(component)
    verify_inp(
        pnl.shape == band.shape == slope.shape and pnl.ndim == 2,
        "scale-statistic arrays must share shape (paths,trading_times)",
    )
    verify_inp(
        path_weight.shape == component.shape == pnl.shape[:1],
        "weight/component vectors have invalid shape",
    )
    verify_inp(
        utility in {"powerexp", "entropy"}, lambda: f"unknown utility {utility!r}"
    )
    verify_inp(
        math.isfinite(loss_probability_epsilon) and loss_probability_epsilon >= 0.0,
        lambda: f"invalid loss probability epsilon {loss_probability_epsilon!r}",
    )
    verify_inp(
        math.isfinite(loss_mc_multiplier) and loss_mc_multiplier > 0.0,
        lambda: f"invalid loss MC multiplier {loss_mc_multiplier!r}",
    )
    if not all(np.all(np.isfinite(value)) for value in (pnl, band, slope, path_weight)):
        raise FloatingPointError("scale statistic input contains NaN or infinity")
    verify_inp(np.all(path_weight > 0.0), "path weights must be positive")
    normalized_weight = path_weight / path_weight.sum()
    direct = component == 0
    reports: list[dict[str, float | int | bool]] = []
    for time in range(pnl.shape[1]):
        value = pnl[:, time]
        tolerance = band[:, time]
        derivative_path = slope[:, time]
        loss = value < -tolerance
        gain = value > tolerance
        probability, probability_error = _weighted_mean_error(
            loss.astype(np.float64), normalized_weight
        )
        gain_probability, gain_probability_error = _weighted_mean_error(
            gain.astype(np.float64), normalized_weight
        )
        mean, mean_error = _weighted_mean_error(value, normalized_weight)
        if utility == "entropy":
            log_weight = np.log(normalized_weight)
            log_tilt = log_weight - risk_aversion * value
            log_tilt -= logsumexp(log_tilt)
            tilted_weight = np.exp(log_tilt)
            objective = -(
                logsumexp(log_weight - risk_aversion * value)
                - logsumexp(log_weight)
            ) / risk_aversion
            marginal_normalizer = math.exp(
                min(700.0, logsumexp(log_weight - risk_aversion * value))
            )
        else:
            positive = np.maximum(value, 0.0)
            negative = np.minimum(value, 0.0)
            utility_value = np.where(
                value > 0.0,
                -np.expm1(-risk_aversion * positive) / risk_aversion,
                negative - 0.5 * risk_aversion * np.square(negative),
            )
            marginal = np.where(
                value > 0.0,
                np.exp(-risk_aversion * positive),
                1.0 - risk_aversion * negative,
            )
            objective = float(np.dot(normalized_weight, utility_value))
            marginal_weight = normalized_weight * marginal
            marginal_normalizer = float(marginal_weight.sum())
            tilted_weight = marginal_weight / marginal_normalizer
        normalized_derivative, normalized_derivative_error = _weighted_mean_error(
            derivative_path, tilted_weight
        )
        raw_derivative = marginal_normalizer * normalized_derivative
        objective_derivative = (
            normalized_derivative if utility == "entropy" else raw_derivative
        )
        direct_loss_count = int(np.sum(loss & direct))
        direct_paths = int(np.sum(direct))
        _, direct_loss_upper = (
            clopper_pearson_one_sided(direct_loss_count, direct_paths)
            if direct_paths
            else (0.0, 1.0)
        )
        direct_loss_probability = (
            direct_loss_count / direct_paths if direct_paths else math.nan
        )
        direct_loss_error = (
            math.sqrt(
                direct_loss_probability * (1.0 - direct_loss_probability) / direct_paths
            )
            if direct_paths
            else math.nan
        )
        weighted_near_nonnegative = bool(
            probability
            <= loss_mc_multiplier * probability_error + loss_probability_epsilon
        )
        direct_near_nonnegative = bool(
            direct_paths > 0
            and direct_loss_probability
            <= loss_mc_multiplier * direct_loss_error + loss_probability_epsilon
        )
        reports.append(
            {
                "trading_time_index": time,
                "objective": float(objective),
                "P_mean_pnl": mean,
                "P_mean_pnl_standard_error": mean_error,
                "P_mean_pnl_lower_confidence": mean - confidence_z * mean_error,
                "P_loss_probability": probability,
                "P_loss_probability_standard_error": probability_error,
                "P_gain_probability": gain_probability,
                "P_gain_probability_standard_error": gain_probability_error,
                "P_pnl_01_quantile": weighted_quantile(value, path_weight, 0.01),
                "P_pnl_05_quantile": weighted_quantile(value, path_weight, 0.05),
                "P_pnl_median": weighted_quantile(value, path_weight, 0.50),
                "proposal_minimum_pnl_minus_negative_band": float(
                    np.min(value + tolerance)
                ),
                "direct_P_paths": direct_paths,
                "direct_P_loss_count": direct_loss_count,
                "direct_P_gain_count": int(np.sum(gain & direct)),
                "direct_P_loss_probability": direct_loss_probability,
                "direct_P_loss_probability_standard_error": direct_loss_error,
                "direct_P_loss_probability_upper_95": direct_loss_upper,
                "normalized_marginal_derivative": normalized_derivative,
                "normalized_marginal_derivative_standard_error": (
                    normalized_derivative_error
                ),
                "raw_marginal_derivative": raw_derivative,
                "objective_derivative": objective_derivative,
                "marginal_density_effective_sample_size": float(
                    1.0 / np.square(tilted_weight).sum()
                ),
                "maximum_marginal_density_weight_share": float(
                    np.max(tilted_weight)
                ),
                "weighted_near_nonnegative": weighted_near_nonnegative,
                "direct_P_near_nonnegative": direct_near_nonnegative,
                "near_nonnegative_within_mc_noise": bool(
                    weighted_near_nonnegative and direct_near_nonnegative
                ),
                "statistically_positive_mean": bool(
                    mean - confidence_z * mean_error > 0.0
                ),
            }
        )
    return reports


__all__ = [
    "ArbTracingConfig",
    "ArbTracingTrainConfig",
    "ArbTracingUtility",
    "OrthogonalArbDirection",
    "fit_physical_state_whitener",
    "traced_scale_statistics",
    "tracing_numerical_band",
    "tracing_pnl",
    "tracing_pnl_right_derivative",
    "tracing_raw_norm_penalty",
    "tracing_utility_objective",
    "weighted_quantile",
]
