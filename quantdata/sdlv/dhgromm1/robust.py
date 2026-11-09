"""Bottom-population-tail OCE, using a shared threshold rather than batch sorting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    SIGNS,
    FloatingConfig,
    FloatingTriple,
    FloatingTripleConfig,
)
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.training import _economic_utility, optimize_translation
from quantdata.torch import BaseEnvironment


@version("0.1", dependencies=[FloatingTripleConfig])
@dataclass(frozen=True, kw_only=True)
class RobustTripleConfig(FloatingTripleConfig):
    group: str = "sdlv/dhgromm1/robust"
    policy: FloatingConfig = FloatingConfig(label="policy",trade_l1=0.,trade_l2=0.)
    tail_fraction: float = .9
    tail_definition: str = "lowest_individual_path_utilities_full_population"

    def __post_init__(self) -> None:
        verify_inp(np.isfinite(self.tail_fraction) and 0 < self.tail_fraction <= 1,
                   lambda: "tail fraction must be finite and in (0,1]")
        verify_inp(self.policy.trade_l1 == self.policy.trade_l2 == 0,
                   lambda: "robust experiment requires zero costs AND no L1/L2 regularization")
        verify_inp(self.policy.risk_aversion == 100.,lambda: "this experiment retains u_100")
        verify_inp(self.tail_definition == "lowest_individual_path_utilities_full_population",
                   lambda: "do not sort minibatches or antithetic-pair averages")


def lower_tail_value(values: torch.Tensor, threshold: torch.Tensor, fraction: float) -> torch.Tensor:
    """Variational lower-tail mean, additive over arbitrary minibatches.

    sup_eta [eta - E[(eta-V)+]/q] = mean of the bottom q of the population.
    A single learned eta per policy is shared across every path/minibatch.
    No order statistic or tail-membership mask is fitted inside this function.
    """
    if fraction == 1:
        return values.mean()
    difference = threshold-values
    hinge = difference.relu()
    # Any slope in [0,1] is valid at the hinge kink. q makes an all-constant
    # zero hedge stationary instead of spuriously moving both cash and eta.
    # This changes NO forward values and uses no minibatch tail ranking.
    hinge = hinge+(difference-difference.detach())*(difference==0).to(values.dtype)*fraction
    return threshold-hinge.mean()/fraction


def exact_tail_size(paths: int, fraction: float) -> int:
    """Require an exact number of retained individual paths, not rounded batches."""
    size = int(round(paths*fraction))
    if paths < 2 or size < 1 or not np.isclose(size, paths*fraction, atol=1e-9, rtol=0):
        raise ValueError("tail fraction must retain an integer number of population paths")
    return size


class RobustTriple(FloatingTriple):
    """Unchanged causal a0/a10 policies, plus three training-only tail thresholds."""

    def __init__(self, env: BaseEnvironment, *, config: RobustTripleConfig, **kwargs) -> None:
        super().__init__(env, config=config, **kwargs)
        for label in LABELS:
            # This scalar never enters action(), recurrence, prices or the book.
            getattr(self, label).register_parameter("tail_threshold",
                torch.nn.Parameter(torch.zeros((), device=env.device, dtype=env.fdtype)))

    def losses(self, results: tuple[HedgeOutput, ...], payoff: torch.Tensor) -> torch.Tensor:
        losses = []
        for label, sign, result in zip(LABELS, SIGNS, results, strict=True):
            policy = getattr(self,label)
            net = result.gain+sign*(payoff-self.payoff_mean)
            value = self._utility(net-policy.translation, torch.zeros_like(net))
            losses.append(-policy.translation-lower_tail_value(value,policy.tail_threshold,self.config.tail_fraction))
        return torch.stack(losses)


def migrate_selected_weights(model: RobustTriple, saved: dict) -> None:
    """Explicitly add ONLY tail thresholds; retain every selected neural weight."""
    missing = model.load_state_dict(saved["state_dict"], strict=False)
    expected = {f"{label}.tail_threshold" for label in LABELS}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise ValueError("source is not the unchanged a0/a10 policy architecture")
    for label in LABELS:
        missing = getattr(model,label).load_state_dict(saved["best_states"][label],strict=False)
        if missing.missing_keys != ["tail_threshold"] or missing.unexpected_keys:
            raise ValueError("selected policy geometry changed during robust warm start")


def profile_tail(net: np.ndarray, *, fraction: float, risk_aversion: float = 100.) -> dict:
    """Full-population tail OCE and winsorized influence for a frozen policy.

    Monotonic utility means sorting net wealth gives the same order at every
    cash translation. Quantile uncertainty uses the winsorized influence, not
    the (incorrect) variance of only the selected observations. Tied cutoffs
    get equal fractional subgradient weights; complete ties have zero MC SE.
    """
    net = np.asarray(net,dtype=np.float64)
    if net.ndim != 1 or not np.isfinite(net).all():
        raise ValueError("expected finite one-dimensional path wealth")
    count = exact_tail_size(len(net),fraction)
    ordered = np.sort(net)
    cash = optimize_translation(ordered[:count],risk_aversion)
    utility = _economic_utility(net,cash,risk_aversion)-cash
    if count == len(net):
        cutoff, weights = float(ordered[-1]),np.ones_like(net)
    else:
        cutoff = float(ordered[count-1]+(ordered[count]-ordered[count-1])/2)
        below, tied = net < cutoff, net == cutoff
        weights = below.astype(np.float64)
        if tied.any():
            weights[tied] = (count-int(below.sum()))/int(tied.sum())
    threshold = float(_economic_utility(np.array([cutoff]),cash,risk_aversion)[0]-cash)
    value = float(_economic_utility(ordered[:count],cash,risk_aversion).mean())
    variates = (cash+utility if fraction==1 else cash+threshold-np.maximum(threshold-utility,0)/fraction)
    np.testing.assert_allclose(variates.mean(),value,atol=1e-12,rtol=1e-10)
    z = net-cash
    derivative = np.exp(-risk_aversion*np.maximum(z,0))+risk_aversion*np.maximum(-z,0)
    return dict(cash=cash,threshold=threshold,oce=value,variates=variates,
                derivative_weight=weights*derivative/fraction,cutoff_wealth=cutoff,retained_paths=count)


def exact_population_scores(model: RobustTriple, gains: torch.Tensor, payoff: torch.Tensor) -> dict:
    """Score the exact global bottom qN and the learned-threshold lower bound.

    Only scoring sorts, once across ALL paths. Training continues to use the
    additive variational loss, which is not a minibatch-specific tail estimate.
    """
    count = exact_tail_size(len(payoff),model.config.tail_fraction)
    scores = {}
    for i,(label,sign) in enumerate(zip(LABELS,SIGNS,strict=True)):
        policy = getattr(model,label)
        net = gains[i]+sign*(payoff-model.payoff_mean)
        utility = model._utility(net-policy.translation,torch.zeros_like(net))
        exact = policy.translation+utility.sort().values[:count].double().mean()
        lower = policy.translation+lower_tail_value(utility,policy.tail_threshold,model.config.tail_fraction)
        scores[label] = dict(robust_oce=float(exact),variational_objective=float(lower),
            full_mean_at_robust_cash=float(policy.translation+utility.double().mean()),
            threshold_gap=float(exact-lower),threshold=float(policy.tail_threshold),cash=float(policy.translation),
            fraction_below_threshold=float((utility<policy.tail_threshold).double().mean()))
    return scores
