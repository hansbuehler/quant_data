"""The JAX diffusion adapter preserves the fitted PyTorch inference equations."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from quantdata.hedging.model_audit.diffusion_jax import (
    make_denoiser,
    make_sampler,
    parameters_from_torch,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.process import cosine_beta_schedule
from quantdata.torch import BaseEnvironment


@pytest.mark.parametrize("variant", [DiffusionVariant.AD_SEQ_VOL, DiffusionVariant.AD_SEQ_VOL_FT])
def test_jax_denoiser_and_diffusion_loop_match_torch(variant):
    torch.manual_seed(12)
    config = DiffusionModelConfig(label="test-jax", variant=variant, model_dimension=16,
        transformer_layers=1, attention_heads=4, diffusion_steps=3)
    model = SequentialVolatilityDenoiser(BaseEnvironment(precision=32, device="cpu"), config=config,
        channel_mean=torch.tensor([.1, -.2]), channel_scale=torch.tensor([.7, 1.3]))
    model.eval()
    if model.output_adapter is not None:
        with torch.no_grad():
            model.output_adapter.up.weight.fill_(.03)
    weights = parameters_from_torch(model)
    denoiser = make_denoiser(layers=config.transformer_layers, heads=config.attention_heads,
        dimension=config.model_dimension, adapter=variant is DiffusionVariant.AD_SEQ_VOL_FT)
    sampler = make_sampler(denoiser, diffusion_steps=config.diffusion_steps)
    shape = (2, config.sequence_length, config.channel_count, len(config.grid.expiries_y), config.grid.strike_count)
    generator = np.random.default_rng(8)
    initial = generator.standard_normal(shape, dtype=np.float32)
    observed = generator.standard_normal((shape[0], shape[1] - 1, *shape[2:]), dtype=np.float32)
    context_noise = generator.standard_normal(observed.shape, dtype=np.float32)
    step_noise = generator.standard_normal((config.diffusion_steps, *shape), dtype=np.float32)
    steps = torch.full((shape[0],), 1, dtype=torch.long)
    betas = cosine_beta_schedule(config.diffusion_steps, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        expected = model(torch.from_numpy(initial), steps).numpy()
        actual = np.asarray(denoiser(weights, jnp.asarray(initial), jnp.asarray(steps.numpy())))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        normalized = (torch.from_numpy(observed) - model.channel_mean.reshape(1, 1, -1, 1, 1)) / model.channel_scale.reshape(1, 1, -1, 1, 1)
        values = torch.from_numpy(initial.copy())
        alphas = 1. - betas
        cumulative = torch.cumprod(alphas, 0)
        for step in range(config.diffusion_steps - 1, -1, -1):
            current = cumulative[step]
            previous = cumulative[step - 1] if step else torch.tensor(1.)
            values[:, :-1] = current.sqrt() * normalized + (1. - current).sqrt() * torch.from_numpy(context_noise)
            predicted = model(values, torch.full((shape[0],), step, dtype=torch.long))
            denominator = 1. - current
            conditional = betas[step] * previous.sqrt() / denominator * predicted
            conditional += (1. - previous) * alphas[step].sqrt() / denominator * values
            values = conditional + (betas[step] * (1. - previous) / denominator).sqrt() * torch.from_numpy(step_noise[step])
        expected_sample = model.denormalize(values)[:, -1].numpy()
    actual_sample = np.asarray(sampler(weights, jnp.asarray(observed), jnp.asarray(initial),
        jnp.asarray(context_noise), jnp.asarray(step_noise), jnp.asarray(betas.numpy())))
    np.testing.assert_allclose(actual_sample, expected_sample, atol=2e-5, rtol=2e-4)
