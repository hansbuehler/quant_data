"""Experimental JAX inference for the saved PyTorch diffusion denoiser."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def parameters_from_torch(model) -> dict[str, jax.Array]:
    """Copy saved evaluation weights without changing the fitted model."""
    if model.training:
        raise ValueError("The JAX inference adapter requires an evaluation-mode model")
    return {name: jnp.asarray(np.asarray(value.detach().cpu())) for name, value in model.state_dict().items()}


def make_denoiser(*, layers: int, heads: int, dimension: int, adapter: bool):
    """Compile the eval-mode transformer with dynamic weights and batch inputs."""
    if dimension % heads:
        raise ValueError("Transformer dimension must be divisible by its attention heads")

    def denoise(weights, values, steps):
        def linear(data, prefix):
            return data @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

        def norm(data, prefix):
            mean = jnp.mean(data, axis=-1, keepdims=True)
            variance = jnp.mean(jnp.square(data - mean), axis=-1, keepdims=True)
            return (data - mean) * jax.lax.rsqrt(variance + 1e-5) * weights[prefix + ".weight"] + weights[prefix + ".bias"]

        batch, sequence = values.shape[:2]
        encoded = linear(values.reshape(batch, sequence, -1), "input_projection") + weights["position_embedding"][None]
        frequencies = jnp.exp(-math.log(10_000.0) * jnp.arange(dimension // 2, dtype=values.dtype) / max(dimension // 2 - 1, 1))
        angles = steps.astype(values.dtype)[:, None] * frequencies[None]
        embedding = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
        if dimension % 2:
            embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
        encoded = encoded + linear(jax.nn.silu(linear(embedding, "time_network.0")), "time_network.2")[:, None]
        causal = jnp.arange(sequence)[:, None] >= jnp.arange(sequence)[None, :]
        for index in range(layers):
            prefix = f"transformer.layers.{index}"
            normalized = norm(encoded, prefix + ".norm1")
            projection = normalized @ weights[prefix + ".self_attn.in_proj_weight"].T + weights[prefix + ".self_attn.in_proj_bias"]
            query, key, value = jnp.split(projection, 3, axis=-1)
            query, key, value = [entry.reshape(batch, sequence, heads, dimension // heads).transpose(0, 2, 1, 3) for entry in (query, key, value)]
            logits = (query @ jnp.swapaxes(key, -1, -2)) / math.sqrt(dimension // heads)
            attention = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
            joined = (attention @ value).transpose(0, 2, 1, 3).reshape(batch, sequence, dimension)
            encoded = encoded + linear(joined, prefix + ".self_attn.out_proj")
            hidden = jax.nn.gelu(linear(norm(encoded, prefix + ".norm2"), prefix + ".linear1"), approximate=False)
            encoded = encoded + linear(hidden, prefix + ".linear2")
        prediction = linear(encoded, "output_projection")
        if adapter:
            prediction = prediction + (encoded @ weights["output_adapter.down.weight"].T) @ weights["output_adapter.up.weight"].T
        return prediction.reshape(values.shape)

    return jax.jit(denoise)


def make_sampler(denoiser, *, diffusion_steps: int):
    """Compile the DDPM loop with explicit common random numbers for validation."""
    if diffusion_steps < 1:
        raise ValueError("The diffusion schedule must have at least one step")

    def sample(weights, observed, initial_noise, context_noise, step_noise, betas):
        mean = weights["channel_mean"].reshape((1, 1, -1, 1, 1))
        scale = weights["channel_scale"].reshape((1, 1, -1, 1, 1))
        normalized = (observed - mean) / scale
        alphas = 1. - betas
        cumulative = jnp.cumprod(alphas)

        def body(position, values):
            step = diffusion_steps - position - 1
            alpha_bar = cumulative[step]
            previous = jnp.where(step > 0, cumulative[jnp.maximum(step - 1, 0)], 1.)
            values = values.at[:, :-1].set(jnp.sqrt(alpha_bar) * normalized + jnp.sqrt(1. - alpha_bar) * context_noise)
            predicted = denoiser(weights, values, jnp.full((values.shape[0],), step, dtype=jnp.int32))
            denominator = 1. - alpha_bar
            conditional_mean = betas[step] * jnp.sqrt(previous) / denominator * predicted
            conditional_mean += (1. - previous) * jnp.sqrt(alphas[step]) / denominator * values
            variance = betas[step] * (1. - previous) / denominator
            return conditional_mean + jnp.sqrt(variance) * step_noise[step]

        final = jax.lax.fori_loop(0, diffusion_steps, body, initial_noise)
        return (final * scale + mean)[:, -1]

    return jax.jit(sample)


def make_random_sampler(denoiser, *, diffusion_steps: int):
    """Compile the complete DDPM transition with independent JAX normal draws."""
    if diffusion_steps < 1:
        raise ValueError("The diffusion schedule must have at least one step")

    def sample(weights, observed, key, betas):
        mean = weights["channel_mean"].reshape((1, 1, -1, 1, 1))
        scale = weights["channel_scale"].reshape((1, 1, -1, 1, 1))
        normalized = (observed - mean) / scale
        key, context_key, initial_key = jax.random.split(key, 3)
        context_noise = jax.random.normal(context_key, observed.shape, dtype=observed.dtype)
        shape = (observed.shape[0], observed.shape[1] + 1, *observed.shape[2:])
        initial = jax.random.normal(initial_key, shape, dtype=observed.dtype)
        alphas = 1. - betas
        cumulative = jnp.cumprod(alphas)

        def body(position, carry):
            values, current_key = carry
            step = diffusion_steps - position - 1
            alpha_bar = cumulative[step]
            previous = jnp.where(step > 0, cumulative[jnp.maximum(step - 1, 0)], 1.)
            values = values.at[:, :-1].set(jnp.sqrt(alpha_bar) * normalized + jnp.sqrt(1. - alpha_bar) * context_noise)
            predicted = denoiser(weights, values, jnp.full((shape[0],), step, dtype=jnp.int32))
            denominator = 1. - alpha_bar
            conditional_mean = betas[step] * jnp.sqrt(previous) / denominator * predicted
            conditional_mean += (1. - previous) * jnp.sqrt(alphas[step]) / denominator * values
            variance = betas[step] * (1. - previous) / denominator
            current_key, noise_key = jax.random.split(current_key)
            noise = jax.random.normal(noise_key, shape, dtype=observed.dtype)
            return conditional_mean + jnp.sqrt(variance) * noise, current_key

        final, key = jax.lax.fori_loop(0, diffusion_steps, body, (initial, key))
        return (final * scale + mean)[:, -1], key

    return jax.jit(sample)
