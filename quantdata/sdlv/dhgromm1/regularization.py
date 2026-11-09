"""Action regularization outside utility, never a trading cash flow or cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.dhgromm1.hedge import HedgeOutput


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class ActionRegularizerConfig:
    label: str = "floating-action-L1-L2"
    group: str = "sdlv/dhgromm1/regularized"
    user_version: str = "0.1"
    l1: float = 1e-4
    l2: float = 1e-8
    treatment: str = "outside_utility"
    reduction: str = "mean_paths_sum_dates_instruments"

    def __post_init__(self) -> None:
        verify_inp(np.isfinite(self.l1) and self.l1 >= 0 and np.isfinite(self.l2) and self.l2 >= 0,
                   lambda: "action regularization coefficients must be finite/nonnegative")
        verify_inp(self.treatment == "outside_utility"
                   and self.reduction == "mean_paths_sum_dates_instruments",
                   lambda: "regularizers are not costs and must retain the stated reduction")


def action_regularizer(trades: torch.Tensor, config: ActionRegularizerConfig) -> torch.Tensor:
    """One penalty per path, in floating quantities (not premium returns).

    Sum over the 20 decision dates and 21 instruments, then average over paths
    in the objective. In particular, do not divide the coefficients by 420.
    """
    return (config.l1 * trades.abs() + config.l2 * trades.square()).sum((-1, -2))


def regularized_losses(raw_losses: torch.Tensor, results: tuple[HedgeOutput, ...],
                       config: ActionRegularizerConfig) -> torch.Tensor:
    """Minimize -OCE + E[penalty]; the wealth passed to u is unchanged."""
    return raw_losses + torch.stack([action_regularizer(out.trades, config).mean() for out in results])


def action_metrics(raw_losses: torch.Tensor, results: tuple[HedgeOutput, ...],
                   config: ActionRegularizerConfig) -> torch.Tensor:
    """Per-policy raw OCE, penalty, L1 size, squared L2 size and max |trade|."""
    rows = []
    for raw, out in zip(raw_losses, results, strict=True):
        rows.append(torch.stack((-raw, action_regularizer(out.trades, config).mean(),
            out.trades.abs().sum((-1, -2)).mean(), out.trades.square().sum((-1, -2)).mean(),
            out.trades.abs().max())))
    return torch.stack(rows)
