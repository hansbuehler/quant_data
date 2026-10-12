"""Conditional one-day sampling from fitted sequential diffusion models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.process import (
    cosine_beta_schedule,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.training import (
    DiffusionTrainingResult,
)
from quantdata.torch import torch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class DiffusionGeneratedTransitions:
    """One-day normalized return and next-surface scenarios."""

    forward_log_returns: FloatArray
    next_log_implied_volatilities: FloatArray
    return_channel_max_dispersion: float

    def __post_init__(self) -> None:
        if self.forward_log_returns.ndim != 1:
            raise ValueError("diffusion generated returns must be a vector")
        if self.next_log_implied_volatilities.ndim != 3:
            raise ValueError("diffusion generated surfaces must be path-expiry-strike")
        if len(self.forward_log_returns) != len(self.next_log_implied_volatilities):
            raise ValueError("diffusion generated paths do not align")
        if not np.all(np.isfinite(self.forward_log_returns)) or not np.all(
            np.isfinite(self.next_log_implied_volatilities)
        ):
            raise ValueError("diffusion generated transitions must be finite")
        if (
            not np.isfinite(self.return_channel_max_dispersion)
            or self.return_channel_max_dispersion < 0.0
        ):
            raise ValueError("return-channel dispersion must be finite and non-negative")


def sample_diffusion_transitions(
    fit: DiffusionTrainingResult,
    *,
    observed_sequence: NDArray[np.floating],
    path_count: int,
    seed: int,
) -> DiffusionGeneratedTransitions:
    """Generate day 22 conditional on the observed first 21 public states."""
    model = fit.ema_model
    config = model.config
    observed = np.asarray(observed_sequence, dtype=np.float64)
    expected = (
        config.sequence_length - 1,
        config.channel_count,
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    )
    if observed.shape != expected:
        raise ValueError(f"observed diffusion sequence must have shape {expected}")
    if path_count < 1:
        raise ValueError("diffusion path_count must be positive")
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    economic = _sample_diffusion_next_states(
        fit,
        observed_sequences=np.broadcast_to(
            observed,
            (path_count, *observed.shape),
        ).copy(),
        generator=generator,
    )
    return_channel = economic[:, 1]
    return DiffusionGeneratedTransitions(
        forward_log_returns=np.mean(return_channel, axis=(-2, -1)),
        next_log_implied_volatilities=economic[:, 0],
        return_channel_max_dispersion=float(
            np.max(np.ptp(return_channel, axis=(-2, -1)))
        ),
    )


def _sample_diffusion_next_states(
    fit: DiffusionTrainingResult,
    *,
    observed_sequences: NDArray[np.floating],
    generator: torch.Generator,
) -> FloatArray:
    """Sample one next state for each path-specific observed context."""
    model = fit.ema_model
    config = model.config
    observed = np.asarray(observed_sequences, dtype=np.float64)
    expected_tail = (
        config.sequence_length - 1,
        config.channel_count,
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    )
    if observed.ndim != 5 or observed.shape[1:] != expected_tail:
        raise ValueError(
            "batched diffusion contexts must be path-time-channel-expiry-strike"
        )
    path_count = len(observed)
    if path_count < 1 or not np.all(np.isfinite(observed)):
        raise ValueError("batched diffusion contexts must be non-empty and finite")
    with torch.no_grad():
        observed_tensor = torch.as_tensor(
            observed,
            dtype=model.fdtype,
            device=model.device,
        )
        observed_normalized = model.normalize(
            torch.cat(
                (
                    observed_tensor,
                    torch.zeros_like(observed_tensor[:, :1]),
                ),
                dim=1,
            )
        )[:, :-1]
        context_noise = torch.randn(
            observed_normalized.shape,
            generator=generator,
            dtype=model.fdtype,
            device=model.device,
        )
        values = torch.randn(
            (
                path_count,
                config.sequence_length,
                config.channel_count,
                len(config.grid.expiries_y),
                len(config.grid.strikes_over_forward),
            ),
            generator=generator,
            dtype=model.fdtype,
            device=model.device,
        )
        betas = cosine_beta_schedule(
            config.diffusion_steps,
            dtype=model.fdtype,
            device=model.device,
        )
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        for step in range(config.diffusion_steps - 1, -1, -1):
            alpha_bar = cumulative[step]
            values[:, :-1] = (
                torch.sqrt(alpha_bar) * observed_normalized
                + torch.sqrt(1.0 - alpha_bar) * context_noise
            )
            steps = torch.full(
                (path_count,),
                step,
                dtype=torch.long,
                device=model.device,
            )
            predicted_clean = model(values, steps)
            previous_alpha_bar = (
                cumulative[step - 1]
                if step > 0
                else values.new_tensor(1.0)
            )
            denominator = 1.0 - alpha_bar
            mean = (
                betas[step] * torch.sqrt(previous_alpha_bar) / denominator
                * predicted_clean
                + (1.0 - previous_alpha_bar)
                * torch.sqrt(alphas[step])
                / denominator
                * values
            )
            if step > 0:
                variance = betas[step] * (1.0 - previous_alpha_bar) / denominator
                values = mean + torch.sqrt(variance) * torch.randn(
                    values.shape,
                    generator=generator,
                    dtype=model.fdtype,
                    device=model.device,
                )
            else:
                values = mean
        values[:, :-1] = observed_normalized
        return np.asarray(
            model.denormalize(values)[:, -1].cpu().numpy(),
            dtype=np.float64,
        )
