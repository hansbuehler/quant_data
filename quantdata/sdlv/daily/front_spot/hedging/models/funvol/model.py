"""Joint GRU neural SDE for functional IV scores and normalized log spot."""

from __future__ import annotations

from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import FunvolConfig
from quantdata.torch import BaseEnvironment, BaseModel, tnn, torch


class _GruHead(BaseModel):
    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: FunvolConfig,
        output_dimension: int,
    ) -> None:
        super().__init__(env)
        self.gru = tnn.GRU(
            input_size=config.state_dimension,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dtype=env.fdtype,
            device=env.device,
        )
        self.output = tnn.Linear(
            config.gru_layers * config.hidden_size,
            output_dimension,
            dtype=env.fdtype,
            device=env.device,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, final_hidden = self.gru(history)
        return self.output(final_hidden.transpose(0, 1).flatten(start_dim=1))


class FunvolNeuralSde(BaseModel):
    """Separate GRU drift and full-covariance diffusion heads, as in FuNVol."""

    def __init__(self, env: BaseEnvironment, *, config: FunvolConfig) -> None:
        super().__init__(env)
        self.config = config
        dimension = config.state_dimension
        self.drift = _GruHead(env, config=config, output_dimension=dimension)
        self.diffusion = _GruHead(
            env,
            config=config,
            output_dimension=dimension * (dimension + 1) // 2,
        )
        self.register_buffer(
            "lower_row",
            torch.tril_indices(dimension, dimension, device=env.device)[0],
        )
        self.register_buffer(
            "lower_column",
            torch.tril_indices(dimension, dimension, device=env.device)[1],
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return annualized drift and lower covariance factor of the state SDE."""
        if history.ndim != 3 or history.shape[1:] != (
            self.config.lag_count,
            self.config.state_dimension,
        ):
            raise ValueError("FuNVol history has incompatible lag/state dimensions")
        drift = self.drift(history)
        lower_values = self.diffusion(history)
        lower = torch.zeros(
            history.shape[0],
            self.config.state_dimension,
            self.config.state_dimension,
            dtype=history.dtype,
            device=history.device,
        )
        lower[:, self.lower_row, self.lower_column] = lower_values
        diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
        positive_diagonal = torch.nn.functional.softplus(diagonal) + 1.0e-3
        lower = lower - torch.diag_embed(diagonal) + torch.diag_embed(positive_diagonal)
        return drift, lower
