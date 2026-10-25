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
        level_mean: torch.Tensor | None = None,
        level_components: torch.Tensor | None = None,
        increment_mean: torch.Tensor | None = None,
        increment_components: torch.Tensor | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.register_buffer("condition_mean", condition_mean.detach().clone())
        self.register_buffer("condition_scale", condition_scale.detach().clone())
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_scale", target_scale.detach().clone())
        if config.pca_rank is not None:
            if any(
                value is None
                for value in (level_mean, level_components, increment_mean, increment_components)
            ):
                raise ValueError("PCA VolGAN requires both fitted surface bases")
            self.register_buffer("level_mean", level_mean.detach().clone())
            self.register_buffer("level_components", level_components.detach().clone())
            self.register_buffer("increment_mean", increment_mean.detach().clone())
            self.register_buffer("increment_components", increment_components.detach().clone())
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

    def encode_full_conditions(self, full_conditions: torch.Tensor) -> torch.Tensor:
        """Project a full live log-IV surface onto frozen train-only coordinates."""
        if full_conditions.shape[-1] != self.config.full_condition_dimension:
            raise ValueError("VolGAN full condition has the wrong dimension")
        if self.config.pca_rank is None:
            return full_conditions
        scores = (full_conditions[..., 3:] - self.level_mean) @ self.level_components.T
        return torch.cat((full_conditions[..., :3], scores), dim=-1)

    def decode_full_targets(self, standardized_targets: torch.Tensor) -> torch.Tensor:
        """Recover a full log-IV increment for pricing and smoothness penalties."""
        decoded = self.decode(standardized_targets)
        if self.config.pca_rank is None:
            return decoded
        increments = self.increment_mean + decoded[..., 1:] @ self.increment_components
        return torch.cat((decoded[..., :1], increments), dim=-1)


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
