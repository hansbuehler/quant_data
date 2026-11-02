"""Small recurrent dynamics calibrated through decoded variance indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from cdxcore.err import verify_inp
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.dysanos.flex.config import FlexGeometryConfig
from quantdata.dysanos.v11.conditional_sequence import SelectiveDiagonalSSM
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1", dependencies=[FlexGeometryConfig])
@dataclass(frozen=True, kw_only=True)
class VixSequenceConfig:
    label: str
    architecture: Literal["lstm", "selective_ssm"]
    input_size: int
    factor_rank: int
    state_size: int = 200
    hidden_size: int = 32
    sequence_length: int = 20
    group: str = "dysanos/v11/vix-sequence"
    user_version: str = "0.1"
    readout: FeedForwardConfig = FeedForwardConfig(activation="Identity", num_depth=0)
    increment_weight: float = .1
    correlation_weight: float = .05
    control_penalty: float = .001

    def __post_init__(self) -> None:
        verify_inp(is_filename(self.label), lambda: "Invalid sequence label")
        verify_inp(self.architecture in {"lstm", "selective_ssm"}, lambda: "Unknown recurrent architecture")
        verify_inp(0 < self.factor_rank <= self.state_size and min(self.input_size, self.hidden_size, self.sequence_length) > 0,
                   lambda: "Invalid sequence geometry")
        verify_inp(min(self.increment_weight, self.correlation_weight, self.control_penalty) >= 0,
                   lambda: "Objective weights must be nonnegative")


class VixSequenceModel(BaseModel):
    """Retain all 200 Gaussian noise directions; alter price-sensitive factors.

    A factor-specific scale and low-rank mean augment the frozen AR transition.
    The sequence observes past variance-index levels and spot scores. The new
    front-normal spot score also enters the readout, without changing its law.
    """

    def __init__(self, env: BaseEnvironment, *, config: VixSequenceConfig, white_basis: torch.Tensor):
        super().__init__(env)
        if white_basis.shape != (config.state_size, config.factor_rank):
            raise ValueError("White factor basis has the wrong geometry")
        self.config = config
        self.register_buffer("white_basis", white_basis.to(device=env.device, dtype=env.fdtype))
        if config.architecture == "lstm":
            self.recurrent = torch.nn.LSTM(config.input_size, config.hidden_size, batch_first=True,
                                           device=env.device, dtype=env.fdtype)
        else:
            self.recurrent = SelectiveDiagonalSSM(env, input_size=config.input_size, hidden_size=config.hidden_size)
        self.readout = FeedForward(env, num_input=config.hidden_size + 1,
                                   num_output=2 * config.factor_rank + 1, config=config.readout)
        # Both architectures begin at the unchanged daily Gaussian AR law.
        with torch.no_grad():
            self.readout.I0.weight.zero_()
            self.readout.I0.bias.zero_()

    def forward(self, context: torch.Tensor, spot_score: torch.Tensor, epsilon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.shape[1:] != (self.config.sequence_length, self.config.input_size):
            raise ValueError("Sequence context shape mismatch")
        if self.config.architecture == "lstm":
            _, (hidden, _) = self.recurrent(context)
            summary = hidden[-1]
        else:
            summary = self.recurrent(context)
        raw = self.readout(torch.cat((summary, spot_score[:, None]), dim=-1))
        rank = self.config.factor_rank
        means = 2 * torch.tanh(raw[:, :rank] / 2)
        scales = torch.exp(1.5 * torch.tanh(raw[:, rank:2 * rank] / 1.5))
        background_scale = torch.exp(1.5 * torch.tanh(raw[:, -1:] / 1.5))
        factors = epsilon @ self.white_basis
        white = background_scale * epsilon + (
            factors * (scales - background_scale) + means
        ) @ self.white_basis.T
        return white, torch.mean(raw.square())


def quantile_distribution_loss(
    samples: torch.Tensor, *, probabilities: torch.Tensor,
    target_quantiles: torch.Tensor, target_scale: torch.Tensor,
) -> torch.Tensor:
    """Equal-tenor, standardized squared Wasserstein/quantile discrepancy."""
    quantiles = torch.quantile(samples, probabilities, dim=0)
    return torch.mean(((quantiles - target_quantiles) / target_scale).square())
