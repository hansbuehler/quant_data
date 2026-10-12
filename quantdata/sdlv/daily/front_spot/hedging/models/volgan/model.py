"""Project-native VolGAN generator and discriminator."""

from __future__ import annotations

from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
)
from quantdata.torch import BaseEnvironment, BaseModel, tnn, torch


class VolganGenerator(BaseModel):
    """Published two-hidden-layer conditional generator."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: VolganModelConfig,
        condition_mean: torch.Tensor,
        condition_scale: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.register_buffer("condition_mean", condition_mean.detach().clone())
        self.register_buffer("condition_scale", condition_scale.detach().clone())
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_scale", target_scale.detach().clone())
        self.linear1 = tnn.Linear(
            config.condition_dimension + config.noise_dimension,
            config.hidden_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.linear2 = tnn.Linear(
            config.hidden_dimension,
            2 * config.hidden_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.linear3 = tnn.Linear(
            2 * config.hidden_dimension,
            config.output_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.activation = tnn.Softplus()

    def forward(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Generate standardized next-return and log-IV increments."""
        if noise.shape[:-1] != conditions.shape[:-1]:
            raise ValueError("VolGAN noise and condition batch axes do not align")
        if noise.shape[-1] != self.config.noise_dimension:
            raise ValueError("VolGAN noise has the wrong dimension")
        if conditions.shape[-1] != self.config.condition_dimension:
            raise ValueError("VolGAN conditions have the wrong dimension")
        normalized = (conditions - self.condition_mean) / self.condition_scale
        hidden = self.activation(self.linear1(torch.cat((noise, normalized), dim=-1)))
        hidden = self.activation(self.linear2(hidden))
        return self.linear3(hidden)

    def decode(self, standardized_targets: torch.Tensor) -> torch.Tensor:
        """Map generated standardized targets back to economic units."""
        return self.target_mean + self.target_scale * standardized_targets


class VolganDiscriminator(BaseModel):
    """Published one-hidden-layer conditional discriminator returning logits."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: VolganModelConfig,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.linear1 = tnn.Linear(
            config.condition_dimension + config.output_dimension,
            config.hidden_dimension,
            dtype=env.fdtype,
            device=env.device,
        )
        self.linear2 = tnn.Linear(
            config.hidden_dimension,
            1,
            dtype=env.fdtype,
            device=env.device,
        )
        self.activation = tnn.Softplus()

    def forward(
        self,
        standardized_conditions: torch.Tensor,
        standardized_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return real/fake logits."""
        if standardized_conditions.shape[:-1] != standardized_targets.shape[:-1]:
            raise ValueError("VolGAN discriminator batch axes do not align")
        joined = torch.cat((standardized_conditions, standardized_targets), dim=-1)
        return self.linear2(self.activation(self.linear1(joined))).squeeze(-1)
