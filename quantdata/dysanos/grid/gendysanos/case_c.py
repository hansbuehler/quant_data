"""Case C models and analytic gradients for conditional-arbitrage tracing.

Case C keeps the power-exponential utility parameter fixed at one and scales
the economic P&L by an externally supplied notional ``n``.  Long and short
notionals are separate nonnegative variables.  Optimization uses the true
utility gradient divided by expected marginal utility, which preserves every
stationary point while removing the common exponentially small scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.2")
@dataclass(frozen=True, kw_only=True)
class CaseCConfig:
    """Logical configuration of a deterministic/conditional Case C pair."""

    label: str
    source_cache_uid: str
    state_features: int
    notionals: tuple[float, ...]
    deterministic_time: int = 0
    conditional_time: int = 6
    expiries: int = 7
    instruments_per_expiry: int = 21
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
        for name in (
            "state_features",
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
        verify_inp(
            isinstance(self.deterministic_time, int)
            and isinstance(self.conditional_time, int)
            and 0 <= self.deterministic_time < self.conditional_time,
            "Case C trading-time indices must be ordered nonnegative integers",
        )
        verify_inp(
            bool(self.notionals)
            and all(math.isfinite(value) and value > 0.0 for value in self.notionals)
            and all(
                left < right
                for left, right in zip(self.notionals, self.notionals[1:])
            ),
            lambda: (
                "notionals must be finite-positive and strictly increasing, "
                f"got {self.notionals!r}"
            ),
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CaseCTrainConfig:
    """Operational settings for the notional continuation."""

    batch_size: int = 4_096
    valuation_batch_size: int = 16_384
    epochs_per_notional: int = 20_000
    minimum_epochs_per_notional: int = 2_000
    value_update: int = 250
    checkpoint_update: int = 2_000
    convergence_points: int = 8
    action_change_tolerance: float = 2.0e-5
    lr: float = 3.0e-4
    compile: bool = True
    seed: int = 182_071

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "epochs_per_notional",
            "minimum_epochs_per_notional",
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
            self.minimum_epochs_per_notional <= self.epochs_per_notional,
            "minimum epochs cannot exceed maximum epochs",
        )
        for name in ("action_change_tolerance", "lr"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value!r}"
                ),
            )


def projected_long_short(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return nonnegative long/short actions with total gross norm at most one.

    The forward value is the radial projection of the positive part.  At zero,
    a straight-through positive-part derivative keeps both long and short
    directions learnable.  Outside the ball, the exact radial-projection
    derivative removes only the infeasible radial direction and preserves all
    tangential direction-changing gradients.  No exterior penalty is used.
    """

    verify_inp(
        raw.ndim >= 4 and raw.shape[-3] == 2,
        lambda: f"long/short raw action has invalid shape {raw.shape}",
    )
    feasible_positive = torch.clamp_min(raw, 0.0)
    positive = raw + (feasible_positive - raw).detach()
    raw_gross = positive.sum(dim=(-3, -2, -1))
    scale = torch.reciprocal(raw_gross.clamp_min(1.0))
    action = positive * scale[..., None, None, None]
    gross = action.detach().sum(dim=(-3, -2, -1))
    boundary = raw_gross >= 1.0
    return action, gross, boundary


def simplex_long_short(
    logits: Tensor,
    *,
    expiries: int,
    instruments_per_expiry: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map portfolio logits and one no-trade logit to the unit gross simplex.

    This is the mirror-descent geometry natural for separate nonnegative long
    and short notionals.  The final coordinate is uninvested gross capacity,
    so an interior optimum is represented without clipping or a penalty.
    """

    components = 2 * expiries * instruments_per_expiry
    verify_inp(
        logits.ndim >= 2 and logits.shape[-1] == components + 1,
        lambda: (
            f"simplex logits must end in {components + 1}, got {logits.shape}"
        ),
    )
    probability = torch.softmax(logits, dim=-1)
    flat_action = probability[..., :components]
    action = flat_action.reshape(
        *logits.shape[:-1], 2, expiries, instruments_per_expiry
    )
    gross = flat_action.sum(dim=-1)
    boundary = probability[..., -1] <= 1.0e-6
    return action, gross, boundary


def case_c_pnl(action: Tensor, returns: Tensor, costs: Tensor) -> Tensor:
    """Return per-unit net P&L for separate long and short notionals."""

    expected = (*returns.shape[:-2], 2, *returns.shape[-2:])
    assert action.shape == expected, f"expected action {expected}, got {action.shape}"
    assert returns.shape == costs.shape
    long = action[..., 0, :, :]
    short = action[..., 1, :, :]
    result = (
        (long - short) * returns - (long + short) * costs
    ).sum(dim=(-1, -2))
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("Case C P&L contains NaN or infinity")
    return result


def case_c_numerical_band(
    action: Tensor,
    returns: Tensor,
    costs: Tensor,
    instrument_band: Tensor,
) -> Tensor:
    """Propagate observation-specific float32 errors through one portfolio."""

    assert returns.shape == costs.shape == instrument_band.shape
    expected = (*returns.shape[:-2], 2, *returns.shape[-2:])
    assert action.shape == expected, f"expected action {expected}, got {action.shape}"
    gross = action[..., 0, :, :] + action[..., 1, :, :]
    accumulation = 32.0 * torch.finfo(torch.float32).eps * (
        gross * (returns.abs() + costs)
    ).sum(dim=(-1, -2))
    result = (gross * instrument_band).sum(dim=(-1, -2)) + accumulation
    if not torch.isfinite(result).all().item() or not torch.all(result >= 0.0).item():
        raise FloatingPointError("Case C numerical band is not finite-nonnegative")
    return result


def powerexp_one(scaled_pnl: Tensor) -> tuple[Tensor, Tensor]:
    """Return fixed-lambda-one power-exp utility and marginal utility."""

    positive = torch.clamp_min(scaled_pnl, 0.0)
    negative = torch.clamp_max(scaled_pnl, 0.0)
    utility = torch.where(
        scaled_pnl > 0.0,
        -torch.expm1(-positive),
        negative - 0.5 * negative.square(),
    )
    marginal = torch.where(
        scaled_pnl > 0.0,
        torch.exp(-positive),
        1.0 - negative,
    )
    if not torch.isfinite(utility).all().item():
        raise FloatingPointError("Case C utility contains NaN or infinity")
    # The exponential branch may legitimately underflow to exactly zero in
    # float32 at large notionals.  Optimization does not use this raw value:
    # ``normalized_marginal_weights`` works directly in log space.
    if not torch.isfinite(marginal).all().item() or not torch.all(marginal >= 0.0).item():
        raise FloatingPointError("Case C marginal utility is not finite-nonnegative")
    return utility, marginal


def normalized_marginal_weights(
    pnl: Tensor, path_weight: Tensor, *, notional: float | Tensor
) -> Tensor:
    """Return normalized physical marginal-utility weights in log space."""

    assert pnl.ndim == 2 and path_weight.shape == pnl.shape[:1]
    if not isinstance(notional, Tensor):
        verify_inp(
            math.isfinite(notional) and notional > 0.0,
            lambda: f"invalid Case C notional {notional!r}",
        )
    scale = torch.as_tensor(notional, dtype=pnl.dtype, device=pnl.device)
    scaled = scale * pnl
    # Clamp the inactive branch as well: ``torch.where`` evaluates both
    # branches, and log1p(-scaled) is otherwise invalid for positive scaled
    # P&L above one even though that value would not be selected.
    nonpositive_scaled = torch.clamp_max(scaled, 0.0)
    log_marginal = torch.where(
        scaled > 0.0,
        -scaled,
        torch.log1p(-nonpositive_scaled),
    )
    log_density = torch.log(path_weight)[:, None] + log_marginal
    result = torch.softmax(log_density, dim=0)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("Case C normalized marginal weights are non-finite")
    return result


def case_c_objective_and_surrogate(
    pnl: Tensor, path_weight: Tensor, *, notional: float | Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return economic utility and its normalized analytic-gradient surrogate.

    The surrogate has no economic interpretation as a level.  Its derivative
    is exactly the true power-exp derivative divided by expected marginal
    utility, with the irrelevant positive notional factor omitted.
    """

    scale = torch.as_tensor(notional, dtype=pnl.dtype, device=pnl.device)
    utility, _ = powerexp_one(scale * pnl)
    normalized_weight = path_weight / path_weight.sum()
    objective = (normalized_weight[:, None] * utility).sum(dim=0)
    marginal_weight = normalized_marginal_weights(
        pnl, path_weight, notional=scale
    )
    surrogate = (marginal_weight.detach() * pnl).sum(dim=0)
    return objective, surrogate, marginal_weight


@version("1.0")
class CaseCAction(BaseModel):
    """A deterministic ``t0`` action and a conditional ``t6`` action network."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: CaseCConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        super().__init__(env)
        verify_inp(
            state_mean.shape == (config.state_features,),
            lambda: f"invalid state mean shape {state_mean.shape}",
        )
        verify_inp(
            state_whitener.shape == (config.state_features, config.state_features),
            lambda: f"invalid state whitener shape {state_whitener.shape}",
        )
        self.config = config
        components = 2 * config.expiries * config.instruments_per_expiry
        output = components + 1
        self.deterministic_logits = torch.nn.Parameter(
            torch.zeros(
                output,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.conditional = FeedForward(
            env,
            num_input=config.state_features,
            num_output=output,
            config=config.network_config,
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_whitener", state_whitener.detach().clone())
        self.initialize_output_layer()

    @torch.no_grad()
    def initialize_output_layer(self) -> None:
        """Start at half gross with uniform long/short direction weights."""

        output_layer = self.conditional.In[-1] if self.conditional.In else self.conditional.I0
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()
            output_layer.bias[-1] = math.log(output_layer.bias.numel() - 1)
        self.deterministic_logits.zero_()
        self.deterministic_logits[-1] = math.log(
            self.deterministic_logits.numel() - 1
        )

    def raw_action(self, conditional_state: Tensor) -> Tensor:
        """Return raw long/short variables for deterministic and conditional times."""

        assert conditional_state.ndim == 2
        assert conditional_state.shape[1] == self.config.state_features
        paths = conditional_state.shape[0]
        factor = (conditional_state - self.state_mean[None]) @ self.state_whitener
        conditional = self.conditional(factor)
        deterministic = self.deterministic_logits[None].expand(paths, -1)
        result = torch.stack((deterministic, conditional), dim=1)
        self.verify(lambda: torch.isfinite(factor), "Case C state factor is non-finite")
        self.verify(lambda: torch.isfinite(result), "Case C raw action is non-finite")
        return result

    def forward(self, conditional_state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return feasible action, gross norm, boundary flag, and raw action."""

        raw = self.raw_action(conditional_state)
        action, gross, boundary = simplex_long_short(
            raw,
            expiries=self.config.expiries,
            instruments_per_expiry=self.config.instruments_per_expiry,
        )
        self.verify(lambda: torch.isfinite(action), "Case C action is non-finite")
        return action, gross, boundary, raw

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Apply any direct parameter constraints after an optimizer step."""

        # The feed-forward action is constrained by its forward map.  Kernel
        # actions override this hook because their portfolio templates are
        # optimized directly in feasible coordinates.


__all__ = [
    "CaseCAction",
    "CaseCConfig",
    "CaseCTrainConfig",
    "case_c_numerical_band",
    "case_c_objective_and_surrogate",
    "case_c_pnl",
    "normalized_marginal_weights",
    "powerexp_one",
    "projected_long_short",
    "simplex_long_short",
]
