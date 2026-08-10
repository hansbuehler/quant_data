"""Kernel portfolios for fixed-notional conditional-arbitrage searches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class KernelCaseCConfig:
    """Logical configuration of a fixed-basis kernel portfolio."""

    label: str
    source_cache_uid: str
    state_features: int
    kernels: int
    notionals: tuple[float, ...]
    deterministic_time: int = 0
    conditional_time: int = 6
    expiries: int = 7
    instruments_per_expiry: int = 21
    group: str = ""
    user_version: str = "0.1"

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        verify_inp(
            isinstance(self.group, str) and not self.group.startswith("/"),
            lambda: f"'group' must be relative, got {self.group!r}",
        )
        verify_inp(
            isinstance(self.source_cache_uid, str) and bool(self.source_cache_uid),
            "'source_cache_uid' must be non-empty",
        )
        for name in (
            "state_features",
            "kernels",
            "expiries",
            "instruments_per_expiry",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            isinstance(self.deterministic_time, int)
            and isinstance(self.conditional_time, int)
            and 0 <= self.deterministic_time < self.conditional_time,
            "kernel Case C trading times must be ordered and nonnegative",
        )
        verify_inp(
            bool(self.notionals)
            and all(math.isfinite(value) and value > 0.0 for value in self.notionals),
            lambda: f"notionals must be finite-positive, got {self.notionals!r}",
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


def normalized_gaussian_kernel_weights(
    factor: Tensor,
    centers: Tensor,
    inverse_bandwidth: Tensor,
) -> Tensor:
    """Return normalized Gaussian weights for fixed orthogonal-state centers."""

    verify_inp(
        factor.ndim == 2 and centers.ndim == 2,
        lambda: f"factor/centers must be matrices, got {factor.shape}/{centers.shape}",
    )
    verify_inp(
        factor.shape[1] == centers.shape[1],
        lambda: f"factor/center dimensions differ: {factor.shape}/{centers.shape}",
    )
    verify_inp(
        inverse_bandwidth.shape == centers.shape[:1],
        lambda: (
            "inverse bandwidth must have one value per center, got "
            f"{inverse_bandwidth.shape}/{centers.shape}"
        ),
    )
    difference = factor[:, None, :] - centers[None, :, :]
    distance_square = difference.square().sum(dim=-1)
    log_kernel = -0.5 * distance_square * inverse_bandwidth.square()[None, :]
    return torch.softmax(log_kernel, dim=-1)


@version("1.0")
class KernelCaseCAction(BaseModel):
    """A deterministic action and a convex mixture of local action templates."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: KernelCaseCConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
        centers: Tensor,
        inverse_bandwidth: Tensor,
    ) -> None:
        super().__init__(env)
        verify_inp(
            state_mean.shape == (config.state_features,),
            lambda: f"invalid state mean shape {state_mean.shape}",
        )
        verify_inp(
            state_whitener.shape
            == (config.state_features, config.state_features),
            lambda: f"invalid state whitener shape {state_whitener.shape}",
        )
        verify_inp(
            centers.shape == (config.kernels, config.state_features),
            lambda: f"invalid kernel-center shape {centers.shape}",
        )
        verify_inp(
            inverse_bandwidth.shape == (config.kernels,),
            lambda: f"invalid inverse-bandwidth shape {inverse_bandwidth.shape}",
        )
        for name, value in (
            ("state mean", state_mean),
            ("state whitener", state_whitener),
            ("kernel centers", centers),
            ("inverse bandwidth", inverse_bandwidth),
        ):
            verify_inp(
                bool(torch.isfinite(value).all().item()),
                lambda name=name: f"{name} contains NaN or infinity",
            )
        verify_inp(
            bool(torch.all(inverse_bandwidth > 0.0).item()),
            "kernel inverse bandwidths must be positive",
        )

        self.config = config
        shape = (2, config.expiries, config.instruments_per_expiry)
        self.deterministic_action = torch.nn.Parameter(
            torch.zeros(shape, dtype=env.fdtype, device=env.device)
        )
        self.template_action = torch.nn.Parameter(
            torch.zeros(
                config.kernels,
                *shape,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_whitener", state_whitener.detach().clone())
        self.register_buffer("centers", centers.detach().clone())
        self.register_buffer(
            "inverse_bandwidth", inverse_bandwidth.detach().clone()
        )

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Project each directly optimized portfolio onto its gross L1 ball."""

        for parameter in (self.deterministic_action, self.template_action):
            parameter.clamp_(min=0.0)
            gross = parameter.sum(dim=(-3, -2, -1), keepdim=True)
            parameter.div_(gross.clamp_min(1.0))

    def forward(
        self, conditional_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return feasible long/short actions at deterministic and conditional times."""

        assert conditional_state.ndim == 2
        assert conditional_state.shape[1] == self.config.state_features
        paths = conditional_state.shape[0]
        factor = (conditional_state - self.state_mean[None]) @ self.state_whitener
        kernel_weight = normalized_gaussian_kernel_weights(
            factor, self.centers, self.inverse_bandwidth
        )
        conditional = torch.einsum(
            "sk,kqei->sqei", kernel_weight, self.template_action
        )
        deterministic = self.deterministic_action[None].expand(paths, -1, -1, -1)
        action = torch.stack((deterministic, conditional), dim=1)
        gross = action.sum(dim=(-3, -2, -1))
        boundary = gross >= 1.0 - 1.0e-6
        raw = action.reshape(paths, 2, -1)
        self.verify(lambda: torch.isfinite(factor), "kernel factors are non-finite")
        self.verify(
            lambda: torch.isfinite(kernel_weight), "kernel weights are non-finite"
        )
        self.verify(lambda: torch.isfinite(action), "kernel actions are non-finite")
        return action, gross, boundary, raw


__all__ = [
    "KernelCaseCAction",
    "KernelCaseCConfig",
    "normalized_gaussian_kernel_weights",
]
