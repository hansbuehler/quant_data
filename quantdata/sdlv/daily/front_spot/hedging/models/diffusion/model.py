"""Causal transformer denoiser for adapted sequential volatility diffusion."""

from __future__ import annotations

import math

from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionVariant,
)
from quantdata.torch import BaseEnvironment, BaseModel, tnn, torch


class LowRankOutputAdapter(BaseModel):
    """Zero-initialized low-rank residual used by arbitrage fine-tuning."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        input_dimension: int,
        output_dimension: int,
        rank: int,
    ) -> None:
        super().__init__(env)
        self.down = tnn.Linear(
            input_dimension,
            rank,
            bias=False,
            dtype=env.fdtype,
            device=env.device,
        )
        self.up = tnn.Linear(
            rank,
            output_dimension,
            bias=False,
            dtype=env.fdtype,
            device=env.device,
        )
        with torch.no_grad():
            self.up.weight.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


class SequentialVolatilityDenoiser(BaseModel):
    """Causal time-conditioned transformer predicting clean sequences."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: DiffusionModelConfig,
        channel_mean: torch.Tensor,
        channel_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        if channel_mean.shape != (config.channel_count,) or channel_scale.shape != (
            config.channel_count,
        ):
            raise ValueError("diffusion channel normalization has the wrong shape")
        self.config = config
        self.register_buffer("channel_mean", channel_mean.detach().clone())
        self.register_buffer("channel_scale", channel_scale.detach().clone())
        self.input_projection = tnn.Linear(
            config.day_dimension,
            config.model_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.position_embedding = tnn.Parameter(
            torch.zeros(
                config.sequence_length,
                config.model_dimension,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.time_network = tnn.Sequential(
            tnn.Linear(
                config.model_dimension,
                config.model_dimension,
                dtype=env.fdtype,
                device=env.device,
            ),
            tnn.SiLU(),
            tnn.Linear(
                config.model_dimension,
                config.model_dimension,
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        layer = tnn.TransformerEncoderLayer(
            d_model=config.model_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_multiplier * config.model_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=env.fdtype,
            device=env.device,
        )
        self.transformer = tnn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = tnn.Linear(
            config.model_dimension,
            config.day_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.output_adapter = (
            LowRankOutputAdapter(
                env,
                input_dimension=config.model_dimension,
                output_dimension=config.day_dimension,
                rank=config.lora_rank,
            )
            if config.variant is DiffusionVariant.AD_SEQ_VOL_FT
            else None
        )

    def normalize(self, sequences: torch.Tensor) -> torch.Tensor:
        shape = (1, 1, -1, 1, 1)
        return (sequences - self.channel_mean.reshape(shape)) / self.channel_scale.reshape(shape)

    def denormalize(self, sequences: torch.Tensor) -> torch.Tensor:
        shape = (1, 1, -1, 1, 1)
        return self.channel_mean.reshape(shape) + self.channel_scale.reshape(shape) * sequences

    def forward(
        self,
        noisy_sequences: torch.Tensor,
        diffusion_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean normalized sequences from noisy normalized inputs."""
        expected_tail = (
            self.config.sequence_length,
            self.config.channel_count,
            len(self.config.grid.expiries_y),
            self.config.grid.strike_count,
        )
        if noisy_sequences.shape[1:] != expected_tail:
            raise ValueError(
                f"diffusion input tail must be {expected_tail}, found {noisy_sequences.shape[1:]}"
            )
        if diffusion_steps.shape != (len(noisy_sequences),):
            raise ValueError("diffusion step vector does not align with the batch")
        flattened = noisy_sequences.flatten(start_dim=2)
        hidden = self.input_projection(flattened) + self.position_embedding[None]
        hidden = hidden + self.time_network(
            _sinusoidal_embedding(
                diffusion_steps,
                dimension=self.config.model_dimension,
                dtype=self.fdtype,
            )
        )[:, None]
        causal_mask = torch.triu(
            torch.ones(
                self.config.sequence_length,
                self.config.sequence_length,
                dtype=torch.bool,
                device=self.device,
            ),
            diagonal=1,
        )
        encoded = self.transformer(hidden, mask=causal_mask)
        prediction = self.output_projection(encoded)
        if self.output_adapter is not None:
            prediction = prediction + self.output_adapter(encoded)
        return prediction.reshape_as(noisy_sequences)


def _sinusoidal_embedding(
    steps: torch.Tensor,
    *,
    dimension: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=steps.device, dtype=dtype)
        / max(half - 1, 1)
    )
    angles = steps.to(dtype=dtype)[:, None] * frequencies[None]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding
