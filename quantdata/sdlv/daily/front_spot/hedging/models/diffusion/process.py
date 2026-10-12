"""Cosine DDPM schedule and static option-arbitrage penalties."""

from __future__ import annotations

import math

from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
)
from quantdata.torch import torch


def cosine_beta_schedule(
    steps: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    offset: float = 0.008,
) -> torch.Tensor:
    """Return the standard clipped cosine beta schedule."""
    grid = torch.linspace(0, steps, steps + 1, dtype=dtype, device=device) / steps
    cumulative = torch.cos((grid + offset) / (1.0 + offset) * math.pi / 2.0).square()
    cumulative = cumulative / cumulative[0]
    return torch.clamp(1.0 - cumulative[1:] / cumulative[:-1], 1.0e-5, 0.999)


def static_arbitrage_penalty(
    log_implied_volatilities: torch.Tensor,
    *,
    config: DiffusionModelConfig,
) -> torch.Tensor:
    """Penalize calendar, call-spread, and butterfly violations on the fixed grid."""
    if log_implied_volatilities.shape[-2:] != (
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    ):
        raise ValueError("arbitrage-penalty surface has the wrong shape")
    expiries = log_implied_volatilities.new_tensor(config.grid.expiries_y)
    strikes = log_implied_volatilities.new_tensor(config.grid.strikes_over_forward)
    volatility = torch.exp(log_implied_volatilities)
    standard_deviation = volatility * torch.sqrt(expiries)[None, :, None]
    d1 = -torch.log(strikes)[None, None, :] / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    normal = torch.distributions.Normal(
        log_implied_volatilities.new_tensor(0.0),
        log_implied_volatilities.new_tensor(1.0),
    )
    calls = normal.cdf(d1) - strikes[None, None, :] * normal.cdf(d2)
    calendar = torch.relu(calls[:, :-1] - calls[:, 1:]).square().mean()
    strike_slopes = torch.diff(calls, dim=-1) / torch.diff(strikes)[None, None]
    call_spread = (
        torch.relu(strike_slopes).square()
        + torch.relu(-1.0 - strike_slopes).square()
    ).mean()
    butterfly = torch.relu(-torch.diff(strike_slopes, dim=-1)).square().mean()
    return calendar + call_spread + butterfly
