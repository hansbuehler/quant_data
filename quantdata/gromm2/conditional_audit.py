"""Local entropy and squared action diagnostics, without changing Q weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from scipy.stats import norm

from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.rolling_audit import paired_samples
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

DIAGNOSTICS = ('entropy', 'squared')


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class ActionNetworkConfig(FeedForwardConfig):
    """Pointwise deep component, in addition to an unrestricted affine action."""

    activation: str = 'SiLU'
    num_depth: int = 2
    num_width: int = 128


@version('0.2', dependencies=[ActionNetworkConfig])
@dataclass(frozen=True, kw_only=True)
class LocalActionConfig:
    """Logical action class and fixed quadratic penalty in original return units."""

    label: str
    feature_count: int
    instrument_count: int
    action_scale: float
    group: str = 'gromm2/local-action-audits'
    user_version: str = '0.1'
    network: FeedForwardConfig = ActionNetworkConfig()
    squared_lambda: float = 1.

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and (self.label.startswith('*') or is_filename(self.label)), lambda: 'invalid action label')
        verify_inp(bool(self.group) and all(is_filename(part) and part not in {'.', '..', ''} for part in self.group.split('/')),
                   lambda: 'invalid action group')
        verify_inp(self.feature_count > 0 and self.instrument_count > 0, lambda: 'action dimensions must be positive')
        verify_inp(math.isfinite(self.action_scale) and self.action_scale > 0., lambda: 'action_scale must be finite and positive')
        verify_inp(math.isfinite(self.squared_lambda) and self.squared_lambda >= 0., lambda: 'invalid squared lambda')

    @property
    def filename(self) -> str:
        return label_unique_filename48_8(self.label[1:]) if self.label.startswith('*') else named_unique_filename48_8(self.label, self)


class AffineDeepAction(BaseModel):
    """Return an uncapped unit action; actual holdings equal action_scale times it.

    Inputs have training-fold normalization. An observed eligibility mask zeros
    instruments after settlement. There is no transaction cost or action cap.
    """

    def __init__(self, env: BaseEnvironment, *, config: LocalActionConfig) -> None:
        super().__init__(env=env)
        self.config = config
        self.affine_only = False
        self.affine = FeedForward(env, config.feature_count, config.instrument_count,
                                  config=FeedForwardConfig.linear_config())
        self.deep = FeedForward(env, config.feature_count, config.instrument_count, config=config.network)
        with torch.no_grad():
            self.affine.I0.weight.zero_()
            self.affine.I0.bias.zero_()
            deep_head = self.deep.In[-1] if len(self.deep.In) else self.deep.I0
            deep_head.weight.zero_()
            deep_head.bias.zero_()

    def forward(self, features: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        action = self.affine(features)
        if not self.affine_only:
            action = action + self.deep(features)
        return action * eligible


class IndependentLocalActions(BaseModel):
    """Separate weights for the two diagnostics, not a cumulative hedging policy."""

    def __init__(self, env: BaseEnvironment, *, config: LocalActionConfig) -> None:
        super().__init__(env=env)
        self.config = config
        self.policies = torch.nn.ModuleDict({name: AffineDeepAction(env, config=config) for name in DIAGNOSTICS})


def local_objective(unit_action: torch.Tensor, scaled_returns: torch.Tensor, *,
                    diagnostic: str, action_scale: float, squared_lambda: float = 1.) -> torch.Tensor:
    """Minimize the specified local objective, up to positive deterministic scaling.

    Set a=s*u and R=s*r. The squared loss divided by s^2 is
    E[-u'r+|u|^2/2+lambda*s^2*(u'r)^2/2]. For entropy, minimizing
    E[expm1(s^2*(-u'r+|u|^2/2))]/s^2 is equivalent to minimizing the
    population log-exponential loss. expm1 avoids cancellation near zero;
    neither returns nor the Q measure are centered or fitted away.
    """
    gain = torch.sum(unit_action * scaled_returns, dim=-1)
    norm_squared = torch.sum(unit_action.square(), dim=-1)
    scale_squared = action_scale**2
    if diagnostic == 'squared':
        return (-gain + .5 * norm_squared + .5 * squared_lambda * scale_squared * gain.square()).mean()
    if diagnostic == 'entropy':
        return (torch.expm1(scale_squared * (-gain + .5 * norm_squared)) / scale_squared).mean()
    raise ValueError(f'unknown diagnostic {diagnostic!r}')


class LocalActionStep(IndependentPortfolioStep):
    """Reuse atomic captured GPU updates, but use local transition losses only."""

    def _compute(self, features: torch.Tensor, scaled_returns: torch.Tensor, eligible: torch.Tensor,
                 active_names: tuple[str, ...]) -> torch.Tensor:
        records = []
        inputs_finite = torch.isfinite(features).all() & torch.isfinite(scaled_returns).all() & torch.isfinite(eligible).all()
        for name in active_names:
            policy = self.model.policies[name]
            self.optimizers[name].zero_grad(set_to_none=not self.use_cuda_graphs)
            action = self.forwards[name](features, eligible)
            loss = local_objective(action, scaled_returns, diagnostic=name,
                action_scale=self.model.config.action_scale, squared_lambda=self.model.config.squared_lambda)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip, error_if_nonfinite=False)
            records.extend(((torch.isfinite(loss) & inputs_finite).to(torch.float64), torch.isfinite(gradient_norm).to(torch.float64),
                            loss.detach().to(torch.float64), gradient_norm.detach().to(torch.float64)))
        return torch.stack(records)


def fold_path_ranges(path_count: int) -> tuple[slice, slice]:
    """Split complete paths without separating either member of an antithetic pair."""
    verify_inp(path_count >= 8 and path_count % 2 == 0, lambda: 'two folds require at least four complete antithetic pairs')
    midpoint = 2 * (path_count // 4)
    return slice(0, midpoint), slice(midpoint, path_count)


def frozen_score_statistics(path_scores: np.ndarray) -> dict[str, float | int | str | None]:
    """Pair-cluster errors for a frozen action evaluated on one untouched fold."""
    scores = np.asarray(path_scores, dtype=np.float64)
    if scores.ndim == 2:
        scores = scores.mean(axis=-1)
    if scores.ndim != 1 or np.any(~np.isfinite(scores)):
        raise ValueError('frozen scores must be finite per-path scores or per-path daily scores')
    pairs = paired_samples(scores)
    if len(pairs) < 2:
        raise ValueError('at least two independent holdout pairs are required')
    mean = float(pairs.mean())
    standard_error = float(pairs.std(ddof=1) / np.sqrt(len(pairs)))
    statistic = mean / standard_error if standard_error > 0. else None
    status = 'ok' if standard_error > 0. else ('degenerate_zero_score' if mean == 0. else 'degenerate_nonzero_score')
    return {'paths': len(scores), 'independent_pairs': len(pairs), 'mean': mean, 'standard_error': standard_error,
            'action_conditional_statistic': statistic, 'test_status': status,
            'ci99_lower': mean - norm.ppf(.995) * standard_error, 'ci99_upper': mean + norm.ppf(.995) * standard_error}
