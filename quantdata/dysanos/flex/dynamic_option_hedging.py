"""Differentiable dynamic option selection for FLEX deep hedging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.network_oce import l1_capped_action
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.3")
@dataclass(frozen=True, kw_only=True)
class DynamicOptionPolicyConfig:
    """Identity and geometry of an affine-plus-deep option selector."""

    label: str
    group: str = "flex/dynamic-option-deep-hedging"
    user_version: str = "0.1"
    state_size: int = 200
    hidden_size: int = 64
    option_slots: int = 5
    maximum_action_l1: float = 16.0
    minimum_nms: float = -1.5
    maximum_nms: float = 0.5
    minimum_option_price: float = 1.0e-6
    decision_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU", num_depth=2, num_width=128
    )
    memory_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU", num_depth=1, num_width=128
    )

    def __post_init__(self) -> None:
        parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid policy label {self.label!r}",
        )
        verify_inp(
            bool(parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in parts),
            lambda: f"invalid policy group {self.group!r}",
        )
        verify_inp(self.state_size > 0, lambda: "state size must be positive")
        verify_inp(self.hidden_size > 0, lambda: "hidden size must be positive")
        verify_inp(self.option_slots > 0, lambda: "option slots must be positive")
        verify_inp(
            math.isfinite(self.maximum_action_l1) and self.maximum_action_l1 > 0.0,
            lambda: "action L1 cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.minimum_nms)
            and math.isfinite(self.maximum_nms)
            and self.minimum_nms < self.maximum_nms,
            lambda: "NMS bounds must be finite and increasing",
        )
        verify_inp(
            math.isfinite(self.minimum_option_price)
            and self.minimum_option_price > 0.0,
            lambda: "minimum option price must be finite-positive",
        )

    @property
    def filename(self) -> str:
        """Return the stable filename body for this policy."""

        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


@dataclass(frozen=True)
class PolicyDecision:
    """One day's bounded holdings and differentiable contract coordinates."""

    quantities: torch.Tensor
    raw_quantities: torch.Tensor
    nms: torch.Tensor
    maturity_fraction: torch.Tensor
    put_weight: torch.Tensor
    hidden: torch.Tensor


class DynamicOptionPolicy(BaseModel):
    """History-aware affine-plus-deep policy with five option slots."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: DynamicOptionPolicyConfig,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.state_size,)
        if state_mean.shape != expected or state_scale.shape != expected:
            raise ValueError("state normalization has the wrong shape")
        if not bool(torch.all(torch.isfinite(state_mean))):
            raise ValueError("state mean is non-finite")
        if not bool(torch.all(torch.isfinite(state_scale) & (state_scale > 0.0))):
            raise ValueError("state scale must be finite-positive")
        self.config = config
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        feature_size = config.state_size + 2 + config.hidden_size
        output_size = 1 + 4 * config.option_slots
        self.affine = torch.nn.Linear(
            feature_size,
            output_size,
            bias=True,
            dtype=env.fdtype,
            device=env.device,
        )
        self.deep = FeedForward(
            env,
            num_input=feature_size,
            num_output=output_size,
            config=config.decision_network,
        )
        self.memory = FeedForward(
            env,
            num_input=feature_size + output_size,
            num_output=config.hidden_size,
            config=config.memory_network,
        )
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias.zero_()
            selector_bias = self.affine.bias[1 + config.option_slots :].reshape(
                config.option_slots, 3
            )
            slot_unit = (
                torch.arange(
                    config.option_slots,
                    dtype=env.fdtype,
                    device=env.device,
                )
                + 0.5
            ) / config.option_slots
            selector_bias[:, 0].copy_(torch.logit(slot_unit))
            selector_bias[:, 1].copy_(torch.logit(torch.flip(slot_unit, dims=(0,))))
            selector_bias[:, 2].copy_(
                torch.where(
                    torch.arange(config.option_slots, device=env.device) % 2 == 0,
                    selector_bias.new_full((), -1.0),
                    selector_bias.new_full((), 1.0),
                )
            )
            final = self.deep.I0 if len(self.deep.In) == 0 else self.deep.In[-1]
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def initial_hidden(self, paths: int) -> torch.Tensor:
        """Return a zero recurrent state for a new path population."""

        return torch.zeros(
            (paths, self.config.hidden_size), dtype=self.fdtype, device=self.device
        )

    def forward(
        self,
        states: torch.Tensor,
        log_spot: torch.Tensor,
        time_fraction: float,
        hidden: torch.Tensor,
    ) -> PolicyDecision:
        """Select bounded quantities and continuous option coordinates."""

        paths = len(states)
        if states.shape != (paths, self.config.state_size):
            raise ValueError("policy states have the wrong shape")
        if log_spot.shape != (paths,) or hidden.shape != (
            paths,
            self.config.hidden_size,
        ):
            raise ValueError("policy spot or hidden state has the wrong shape")
        time = states.new_full((paths, 1), float(time_fraction))
        features = torch.cat(
            (
                (states - self.state_mean) / self.state_scale,
                log_spot[:, None],
                time,
                hidden,
            ),
            dim=1,
        )
        raw = self.affine(features) + self.deep(features)
        slots = self.config.option_slots
        raw_quantities = torch.cat((raw[:, :1], raw[:, 1 : 1 + slots]), dim=1)
        quantities = l1_capped_action(raw_quantities, self.config.maximum_action_l1)
        selector = raw[:, 1 + slots :].reshape(paths, slots, 3)
        nms_unit = torch.sigmoid(selector[..., 0])
        nms = self.config.minimum_nms + (
            self.config.maximum_nms - self.config.minimum_nms
        ) * nms_unit
        maturity_fraction = torch.sigmoid(selector[..., 1])

        # Contracts are actual calls or puts in the forward pass.  The sigmoid
        # supplies the straight-through gradient needed to learn this discrete choice.
        soft_put = torch.sigmoid(selector[..., 2])
        hard_put = (soft_put >= 0.5).to(dtype=soft_put.dtype)
        put_weight = hard_put + soft_put - soft_put.detach()
        next_hidden = torch.tanh(self.memory(torch.cat((features, raw), dim=1)))
        self.verify(lambda: torch.isfinite(raw), lambda: "policy produced non-finite heads")
        return PolicyDecision(
            quantities=quantities,
            raw_quantities=raw_quantities,
            nms=nms,
            maturity_fraction=maturity_fraction,
            put_weight=put_weight,
            hidden=next_hidden,
        )


def pathwise_nms_strikes(
    model: Any,
    surface: Any,
    *,
    days: torch.Tensor,
    normalized_moneyness: torch.Tensor,
) -> torch.Tensor:
    """Map pathwise day maturities and NMS coordinates to relative strikes."""

    if days.shape != normalized_moneyness.shape or days.ndim != 2:
        raise ValueError("days and NMS must be matching path-by-slot matrices")
    nodes = model.sanos_decoder.model_expiries_y
    query = days / DAYS_PER_YEAR
    if bool(torch.any(query < 0.0)) or bool(torch.any(query > nodes[-1] + 1.0e-6)):
        raise ValueError("option expiry lies outside the decoded surface")
    raw_upper = torch.searchsorted(nodes, query, right=False)
    upper = torch.clamp(raw_upper, max=len(nodes) - 1)
    lower = torch.clamp(raw_upper - 1, min=0)
    before_first = raw_upper == 0
    previous_time = torch.where(before_first, torch.zeros_like(query), nodes[lower])
    alpha = torch.clamp(
        (query - previous_time) / (nodes[upper] - previous_time), 0.0, 1.0
    )
    batch = torch.arange(len(days), device=days.device)[:, None]
    upper_variance = surface.total_vols[batch, upper].square()
    lower_variance = surface.total_vols[batch, lower].square()
    lower_variance = torch.where(before_first, torch.zeros_like(lower_variance), lower_variance)
    total_variance = (1.0 - alpha) * lower_variance + alpha * upper_variance
    epsilon_vol = surface.total_vols[:, :1] / torch.sqrt(nodes[:1])
    nms_scale = torch.sqrt(
        (epsilon_vol.square() * float(NMS.EPS) + total_variance)
        / (1.0 + float(NMS.EPS))
    )
    strikes = torch.exp(normalized_moneyness * nms_scale)
    if not bool(torch.all(torch.isfinite(strikes) & (strikes > 0.0))):
        raise FloatingPointError("NMS conversion produced invalid strikes")
    return strikes


def _price_puts_batch(
    model: Any,
    surface: Any,
    *,
    expiries_y: torch.Tensor,
    strikes: torch.Tensor,
) -> torch.Tensor:
    """Price puts directly, avoiding cancellation in call-put parity."""

    if expiries_y.shape != strikes.shape or strikes.ndim != 2:
        raise ValueError("put expiries and strikes must be matching matrices")
    decoder = model.sanos_decoder
    nodes = decoder.model_expiries_y
    raw_upper = torch.searchsorted(nodes, expiries_y.contiguous())
    upper = torch.clamp(raw_upper, max=len(nodes) - 1)
    lower = torch.clamp(raw_upper - 1, min=0, max=len(nodes) - 1)
    before_first = raw_upper == 0
    previous_time = torch.where(before_first, torch.zeros_like(expiries_y), nodes[lower])
    alpha = torch.clamp(
        (expiries_y - previous_time)
        / torch.clamp_min(nodes[upper] - previous_time, torch.finfo(expiries_y.dtype).tiny),
        0.0,
        1.0,
    )
    batch = torch.arange(len(strikes), device=strikes.device)[:, None]
    upper_support = surface.support_strikes[batch, upper]
    upper_density = surface.densities[batch, upper]
    lower_support = surface.support_strikes[batch, lower]
    lower_density = surface.densities[batch, lower]
    lower_support = torch.where(
        before_first.unsqueeze(-1),
        surface.support_strikes[:, :1].expand(-1, strikes.shape[1], -1),
        lower_support,
    )
    lower_density = torch.where(
        before_first.unsqueeze(-1),
        surface.density0[:, None, :].expand(-1, strikes.shape[1], -1),
        lower_density,
    )
    upper_variance = surface.backbone_total_vols[batch, upper].square()
    lower_variance = surface.backbone_total_vols[batch, lower].square()
    lower_variance = torch.where(before_first, torch.zeros_like(lower_variance), lower_variance)
    sqrt_variance = torch.sqrt(
        torch.clamp_min(
            (1.0 - alpha) * lower_variance + alpha * upper_variance,
            0.0,
        )
    )

    def black_put(support: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        variance = sqrt_variance.unsqueeze(-1)
        safe_variance = torch.clamp_min(variance, 1.0e-10)
        strike = strikes.unsqueeze(-1)
        d1 = (torch.log(support / strike) + 0.5 * safe_variance.square()) / safe_variance
        d2 = d1 - safe_variance
        put = strike * torch.special.ndtr(-d2) - support * torch.special.ndtr(-d1)
        put = torch.where(variance > 0.0, put, torch.relu(strike - support))
        return torch.sum(put * density, dim=-1)

    upper_puts = black_put(upper_support, upper_density)
    lower_puts = black_put(lower_support, lower_density)
    return (1.0 - alpha) * lower_puts + alpha * upper_puts


def interpolated_option_returns(
    model: Any,
    surface: Any,
    *,
    issue_day: int,
    remaining_days: int,
    path_spots: torch.Tensor,
    nms: torch.Tensor,
    maturity_fraction: torch.Tensor,
    put_weight: torch.Tensor,
    price_floor: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Price and settle one differentiable calendar-interpolated option per slot.

    The synthetic contract uses the same interpolation weights for its issue
    premium and realized payoff.  Therefore no maturity rounding or hidden
    pricing/settlement mismatch is introduced.
    """

    if remaining_days <= 0:
        raise ValueError("remaining days must be positive")
    if nms.shape != maturity_fraction.shape or nms.shape != put_weight.shape:
        raise ValueError("selector coordinates must have matching shapes")
    paths, slots = nms.shape
    if path_spots.shape[0] != paths or path_spots.shape[1] <= issue_day + remaining_days:
        raise ValueError("spot paths do not cover option settlement")
    if remaining_days == 1:
        continuous_days = torch.ones_like(maturity_fraction)
    else:
        continuous_days = 1.0 + (remaining_days - 1.0) * maturity_fraction
    lower_days = torch.floor(continuous_days).detach()
    upper_days = torch.clamp(lower_days + 1.0, max=float(remaining_days))
    alpha = continuous_days - lower_days
    alpha = torch.where(upper_days == lower_days, torch.zeros_like(alpha), alpha)

    lower_strike = pathwise_nms_strikes(
        model, surface, days=lower_days, normalized_moneyness=nms
    )
    upper_strike = pathwise_nms_strikes(
        model, surface, days=upper_days, normalized_moneyness=nms
    )
    lower_call = model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=lower_days / DAYS_PER_YEAR,
        strikes=lower_strike,
    )
    upper_call = model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=upper_days / DAYS_PER_YEAR,
        strikes=upper_strike,
    )
    lower_put = _price_puts_batch(
        model,
        surface,
        expiries_y=lower_days / DAYS_PER_YEAR,
        strikes=lower_strike,
    )
    upper_put = _price_puts_batch(
        model,
        surface,
        expiries_y=upper_days / DAYS_PER_YEAR,
        strikes=upper_strike,
    )
    lower_price = lower_call + put_weight * (lower_put - lower_call)
    upper_price = upper_call + put_weight * (upper_put - upper_call)

    lower_index = lower_days.to(dtype=torch.long) + issue_day
    upper_index = upper_days.to(dtype=torch.long) + issue_day
    batch = torch.arange(paths, device=nms.device)[:, None]
    issue_spot = path_spots[:, issue_day : issue_day + 1]
    lower_gross = path_spots[batch, lower_index] / issue_spot
    upper_gross = path_spots[batch, upper_index] / issue_spot
    lower_call_payoff = torch.clamp_min(lower_gross - lower_strike, 0.0)
    upper_call_payoff = torch.clamp_min(upper_gross - upper_strike, 0.0)
    lower_put_payoff = torch.clamp_min(lower_strike - lower_gross, 0.0)
    upper_put_payoff = torch.clamp_min(upper_strike - upper_gross, 0.0)
    lower_payoff = lower_call_payoff + put_weight * (
        lower_put_payoff - lower_call_payoff
    )
    upper_payoff = upper_call_payoff + put_weight * (
        upper_put_payoff - upper_call_payoff
    )
    price = (1.0 - alpha) * lower_price + alpha * upper_price
    payoff = (1.0 - alpha) * lower_payoff + alpha * upper_payoff
    if bool(torch.any(~torch.isfinite(price))) or bool(torch.any(price <= 0.0)):
        minimum = float(torch.amin(price.detach()))
        raise FloatingPointError(
            f"selected option has a non-finite or nonpositive premium; minimum={minimum:.9g}"
        )
    eligible = price >= price_floor
    safe_price = torch.where(eligible, price, torch.ones_like(price))
    option_return = torch.where(eligible, payoff / safe_price - 1.0, 0.0)
    if not bool(torch.all(torch.isfinite(option_return))):
        raise FloatingPointError("selected option return is non-finite")
    return option_return, {
        "continuous_days": continuous_days,
        "lower_days": lower_days,
        "upper_days": upper_days,
        "alpha": alpha,
        "lower_strike": lower_strike,
        "upper_strike": upper_strike,
        "price": price,
        "payoff": payoff,
        "eligible": eligible,
    }


def five_day_straddle_claim(path_spots: torch.Tensor) -> torch.Tensor:
    """Return the sum of five normalized ATM five-day straddle payoffs."""

    if path_spots.ndim != 2 or path_spots.shape[1] != 26:
        raise ValueError("claim requires spot paths at days zero through 25")
    starts = torch.arange(0, 25, 5, device=path_spots.device)
    gross = path_spots[:, starts + 5] / path_spots[:, starts]
    return torch.sum(torch.abs(gross - 1.0), dim=1)


def power_exponential_utility(value: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """Evaluate the FLEX power-exponential utility."""

    lam = float(risk_aversion)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("risk aversion must be finite-positive")
    # Evaluate the exponential only on a nonnegative argument.  ``torch.where``
    # evaluates both branches, so applying exp directly to a large negative
    # value would overflow even though the quadratic branch is selected.
    positive_value = torch.clamp_min(value, 0.0)
    return torch.where(
        value >= 0.0,
        -torch.expm1(-lam * positive_value) / lam,
        value - 0.5 * lam * value.square(),
    )


def oce_value(
    wealth: torch.Tensor,
    *,
    utility: str,
    risk_aversion: float,
    cash_translation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an empirical OCE value under entropy or power-exponential utility."""

    if wealth.ndim != 1 or not bool(torch.all(torch.isfinite(wealth))):
        raise ValueError("wealth must be a finite path vector")
    lam = float(risk_aversion)
    if utility == "entropy":
        return -(torch.logsumexp(-lam * wealth, dim=0) - math.log(len(wealth))) / lam
    if utility != "power-exp" or cash_translation is None:
        raise ValueError("power-exp requires a cash translation")
    residual = wealth - cash_translation
    return cash_translation + torch.mean(power_exponential_utility(residual, lam))


def marginal_utility_density(
    wealth: torch.Tensor,
    *,
    utility: str,
    risk_aversion: float,
    cash_translation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the normalized empirical OCE marginal-utility density."""

    lam = float(risk_aversion)
    if utility == "entropy":
        log_density = -lam * wealth
        return torch.exp(log_density - torch.logsumexp(log_density, dim=0) + math.log(len(wealth)))
    if utility != "power-exp" or cash_translation is None:
        raise ValueError("power-exp requires a cash translation")
    residual = wealth - cash_translation
    positive_residual = torch.clamp_min(residual, 0.0)
    derivative = torch.where(
        residual >= 0.0,
        torch.exp(-lam * positive_residual),
        1.0 - lam * residual,
    )
    if not bool(torch.all(torch.isfinite(derivative) & (derivative > 0.0))):
        raise FloatingPointError("power-exponential marginal utility is invalid")
    return derivative / torch.mean(derivative)
