"""Cosine DDPM schedule and static option-arbitrage penalties."""

from __future__ import annotations

import math

from quantdata.biz_calendar import NMS
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
        config.grid.strike_count,
    ):
        raise ValueError("arbitrage-penalty surface has the wrong shape")
    expiries = log_implied_volatilities.new_tensor(config.grid.expiries_y)
    volatility = torch.exp(log_implied_volatilities)
    if config.grid.nms_nodes is None:
        strikes = log_implied_volatilities.new_tensor(
            config.grid.strikes_over_forward
        )[None, None, :]
        scale = None
    else:
        nodes = log_implied_volatilities.new_tensor(config.grid.nms_nodes)
        atm_index = config.grid.nms_nodes.index(0.0)
        scale = torch.sqrt(
            (NMS.SCALE**2 * NMS.EPS + volatility[..., atm_index].square() * expiries[None, :])
            / (NMS.EPS + 1.0)
        )
        strikes = torch.exp(scale[..., None] * nodes[None, None, :])
    standard_deviation = volatility * torch.sqrt(expiries)[None, :, None]
    d1 = -torch.log(strikes) / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    normal = torch.distributions.Normal(
        log_implied_volatilities.new_tensor(0.0),
        log_implied_volatilities.new_tensor(1.0),
    )
    calls = normal.cdf(d1) - strikes * normal.cdf(d2)
    if scale is None:
        calendar = torch.relu(calls[:, :-1] - calls[:, 1:]).square().mean()
    else:
        short_strikes = strikes[:, :-1]
        query_nms = torch.log(short_strikes) / scale[:, 1:, None]
        upper = torch.searchsorted(nodes, query_nms.contiguous()).clamp(1, len(nodes) - 1)
        lower = upper - 1
        long_vols = volatility[:, 1:]
        lower_vol = torch.gather(long_vols, -1, lower)
        upper_vol = torch.gather(long_vols, -1, upper)
        weight = ((query_nms - nodes[lower]) / (nodes[upper] - nodes[lower])).clamp(0.0, 1.0)
        long_vol = lower_vol + weight * (upper_vol - lower_vol)
        long_std = long_vol * torch.sqrt(expiries[1:])[None, :, None]
        long_d1 = -torch.log(short_strikes) / long_std + 0.5 * long_std
        long_call = normal.cdf(long_d1) - short_strikes * normal.cdf(long_d1 - long_std)
        calendar = torch.relu(calls[:, :-1] - long_call).square().mean()
    strike_slopes = torch.diff(calls, dim=-1) / torch.diff(strikes, dim=-1)
    call_spread = (
        torch.relu(strike_slopes).square()
        + torch.relu(-1.0 - strike_slopes).square()
    ).mean()
    butterfly = torch.relu(-torch.diff(strike_slopes, dim=-1)).square().mean()
    return calendar + call_spread + butterfly
