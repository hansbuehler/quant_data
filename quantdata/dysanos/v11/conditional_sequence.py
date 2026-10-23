"""Small conditional-innovation models on a frozen daily DYSANOS joint law.

These models alter the conditional mean and scalar variance of the *normal*
surface innovation.  They do not replace the front SANOS spot marginal and do
not themselves create a heavy-tailed conditional innovation law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.torch import BaseEnvironment, BaseModel


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class ConditionalSequenceConfig:
    """Geometry of one conditional normal-innovation network."""

    label: str
    architecture: Literal["lstm", "selective_ssm"]
    group: str = "dysanos/v11/conditional-sequence"
    user_version: str = "0.1"
    input_size: int = 16
    hidden_size: int = 32
    mean_rank: int = 8
    state_size: int = 200
    sequence_length: int = 20

    def __post_init__(self) -> None:
        verify_inp(bool(self.label), lambda: "conditional model label must be nonempty")
        verify_inp(
            self.architecture in {"lstm", "selective_ssm"},
            lambda: f"unsupported conditional architecture {self.architecture!r}",
        )
        verify_inp(
            0 < self.mean_rank <= self.state_size
            and min(self.input_size, self.hidden_size, self.sequence_length) > 0,
            lambda: "conditional model dimensions must be positive and aligned",
        )


class SelectiveDiagonalSSM(BaseModel):
    """Compact input-dependent diagonal state-space recurrence."""

    def __init__(
        self, env: BaseEnvironment, *, input_size: int, hidden_size: int
    ) -> None:
        super().__init__(env)
        self.decay = torch.nn.Linear(input_size, hidden_size)
        self.input_map = torch.nn.Linear(input_size, hidden_size)
        self.output_gate = torch.nn.Linear(input_size, hidden_size)
        self.to(dtype=env.fdtype, device=env.device)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return last selective state; time is the middle tensor axis."""
        state = torch.zeros(
            sequence.shape[0], self.decay.out_features,
            dtype=sequence.dtype, device=sequence.device,
        )
        for observation in sequence.unbind(dim=1):
            retention = torch.exp(-torch.nn.functional.softplus(self.decay(observation)))
            innovation = torch.tanh(self.input_map(observation))
            state = retention * state + (1.0 - retention) * innovation
        return torch.sigmoid(self.output_gate(sequence[:, -1])) * state


class ConditionalInnovationModel(BaseModel):
    """Predict rank-limited innovation drift and one scalar log scale."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ConditionalSequenceConfig,
        white_mean_basis: torch.Tensor,
    ) -> None:
        super().__init__(env)
        if white_mean_basis.shape != (config.state_size, config.mean_rank):
            raise ValueError("white innovation basis does not match model config")
        self.config = config
        self.register_buffer(
            "white_mean_basis",
            white_mean_basis.to(dtype=env.fdtype, device=env.device),
        )
        if config.architecture == "lstm":
            self.sequence_model: torch.nn.Module = torch.nn.LSTM(
                config.input_size, config.hidden_size, batch_first=True
            )
        else:
            self.sequence_model = SelectiveDiagonalSSM(
                env, input_size=config.input_size, hidden_size=config.hidden_size
            )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size + 1, config.hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_size, config.mean_rank + 1),
        )
        self.to(dtype=env.fdtype, device=env.device)

    def forward(
        self, sequence: torch.Tensor, spot_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean in white coordinates and scalar log innovation scale."""
        if sequence.shape[1:] != (
            self.config.sequence_length, self.config.input_size
        ) or spot_score.shape != (len(sequence),):
            raise ValueError("conditional sequence and spot score shapes differ")
        if self.config.architecture == "lstm":
            _, (hidden, _) = self.sequence_model(sequence)
            summary = hidden[-1]
        else:
            summary = self.sequence_model(sequence)
        output = self.readout(torch.cat((summary, spot_score[:, None]), dim=1))
        white_mean = output[:, : self.config.mean_rank] @ self.white_mean_basis.T
        log_scale = 2.0 * torch.tanh(output[:, -1] / 2.0)
        return white_mean, log_scale


def conditional_normal_nll(
    observed_white: torch.Tensor,
    predicted_white_mean: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    mean_penalty: float = 0.001,
    scale_penalty: float = 0.001,
) -> torch.Tensor:
    """Average per-coordinate conditional normal NLL without its constant."""
    if (
        observed_white.shape != predicted_white_mean.shape
        or observed_white.ndim != 2
        or log_scale.shape != (len(observed_white),)
        or mean_penalty < 0.0
        or scale_penalty < 0.0
    ):
        raise ValueError("conditional normal NLL inputs are inconsistent")
    dimension = observed_white.shape[1]
    residual = observed_white - predicted_white_mean
    mean_square = torch.sum(residual.square(), dim=1) / dimension
    data_loss = 0.5 * (
        torch.exp(-2.0 * log_scale) * mean_square + 2.0 * log_scale
    )
    regularizer = (
        mean_penalty * torch.mean(predicted_white_mean.square())
        + scale_penalty * torch.mean(log_scale.square())
    )
    return torch.mean(data_loss) + regularizer


def baseline_normal_nll(observed_white: torch.Tensor) -> float:
    """Per-coordinate frozen-AR normal NLL on the same observations."""
    if observed_white.ndim != 2 or observed_white.shape[1] <= 0:
        raise ValueError("white innovation matrix has invalid shape")
    return float(0.5 * torch.mean(observed_white.square()).detach().cpu())


__all__ = [
    "ConditionalInnovationModel",
    "ConditionalSequenceConfig",
    "SelectiveDiagonalSSM",
    "baseline_normal_nll",
    "conditional_normal_nll",
]
