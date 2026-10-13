"""History-aware affine-plus-deep policy for GROMM hedging."""

from __future__ import annotations

from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.torch import BaseEnvironment, BaseModel, tnn, torch
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class GrommPolicyArchitecture:
    """Shape-independent architecture selected on the 2022 initialization set."""

    hidden_size: int
    recurrent_layers: int
    deep_activation: str
    deep_depth: int
    deep_width: int

    def __post_init__(self) -> None:
        verify_inp(self.hidden_size > 0, lambda: "hidden_size must be positive")
        verify_inp(
            self.recurrent_layers > 0,
            lambda: "recurrent_layers must be positive",
        )
        verify_inp(
            bool(self.deep_activation.strip()),
            lambda: "deep_activation must be non-empty",
        )
        verify_inp(self.deep_depth > 0, lambda: "deep_depth must be positive")
        verify_inp(self.deep_width > 0, lambda: "deep_width must be positive")

    def policy_config(
        self,
        *,
        label: str,
        feature_count: int,
        instrument_count: int,
        position_cap: float,
    ) -> GrommPolicyConfig:
        """Bind the selected architecture to one dated policy geometry."""
        return GrommPolicyConfig(
            label=label,
            feature_count=feature_count,
            instrument_count=instrument_count,
            hidden_size=self.hidden_size,
            recurrent_layers=self.recurrent_layers,
            decision_network=FeedForwardConfig(
                activation=self.deep_activation,
                num_depth=self.deep_depth,
                num_width=self.deep_width,
            ),
            position_cap=position_cap,
        )


@version("0.3")
@dataclass(frozen=True, kw_only=True)
class GrommPolicyConfig:
    """Identity and geometry of one GROMM hedge policy."""

    label: str
    group: str = "hedging/gromm/policy"
    user_version: str = "0.1"
    feature_count: int
    instrument_count: int
    hidden_size: int = 32
    recurrent_layers: int = 1
    decision_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=64,
    )
    position_cap: float = 1_000.0

    def __post_init__(self) -> None:
        group_parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid policy label {self.label!r}",
        )
        verify_inp(
            bool(group_parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
            lambda: f"invalid policy group {self.group!r}",
        )
        verify_inp(self.feature_count > 0, lambda: "feature_count must be positive")
        verify_inp(
            self.instrument_count > 0,
            lambda: "instrument_count must be positive",
        )
        verify_inp(self.hidden_size > 0, lambda: "hidden_size must be positive")
        verify_inp(
            self.recurrent_layers > 0,
            lambda: "recurrent_layers must be positive",
        )
        verify_inp(
            self.position_cap > 0.0,
            lambda: "position_cap must be positive",
        )

    @property
    def filename(self) -> str:
        """Return a stable filename body for checkpoints and reports."""
        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


def gromm_architecture_candidates(
    *,
    label_prefix: str,
    feature_count: int,
    instrument_count: int,
    position_cap: float,
) -> tuple[GrommPolicyConfig, ...]:
    """Return the four precommitted 2022 architecture candidates."""
    geometries = (
        (16, 1, 32),
        (32, 2, 64),
        (64, 2, 128),
        (64, 3, 128),
    )
    return tuple(
        GrommPolicyConfig(
            label=f"{label_prefix}-h{hidden}-d{depth}-w{width}",
            feature_count=feature_count,
            instrument_count=instrument_count,
            hidden_size=hidden,
            decision_network=FeedForwardConfig(
                activation="SiLU",
                num_depth=depth,
                num_width=width,
            ),
            position_cap=position_cap,
        )
        for hidden, depth, width in geometries
    )


class GrommPolicy(BaseModel):
    """LSTM state encoder followed by affine-plus-deep hedge actions."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: GrommPolicyConfig,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.feature_count,)
        if feature_mean.shape != expected or feature_scale.shape != expected:
            raise ValueError("GROMM feature normalization has the wrong shape")
        if not bool(torch.all(torch.isfinite(feature_mean))):
            raise ValueError("GROMM feature mean must be finite")
        if not bool(torch.all(torch.isfinite(feature_scale) & (feature_scale > 0.0))):
            raise ValueError("GROMM feature scale must be finite-positive")
        self.config = config
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.recurrent = tnn.LSTM(
            input_size=config.feature_count,
            hidden_size=config.hidden_size,
            num_layers=config.recurrent_layers,
            batch_first=True,
            dtype=env.fdtype,
            device=env.device,
        )
        combined_size = config.feature_count + config.hidden_size
        self.affine = tnn.Linear(
            combined_size,
            config.instrument_count,
            bias=True,
            dtype=env.fdtype,
            device=env.device,
        )
        self.deep = FeedForward(
            env,
            num_input=combined_size,
            num_output=config.instrument_count,
            config=config.decision_network,
        )
        final = self.deep.I0 if len(self.deep.In) == 0 else self.deep.In[-1]
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias.zero_()
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return capped actions for each path and decision time."""
        if features.ndim != 3 or features.shape[-1] != self.config.feature_count:
            raise ValueError(
                "features must have shape (paths,steps,feature_count), found "
                f"{tuple(features.shape)}"
            )
        static_shape = (features.shape[1], self.config.instrument_count)
        pathwise_shape = (
            features.shape[0],
            features.shape[1],
            self.config.instrument_count,
        )
        if active_mask.shape not in {static_shape, pathwise_shape}:
            raise ValueError(
                "active_mask must have shape "
                f"{static_shape} or {pathwise_shape}, found {tuple(active_mask.shape)}"
            )
        normalized = (features - self.feature_mean) / self.feature_scale
        memory, _ = self.recurrent(normalized)
        combined = torch.cat((normalized, memory), dim=-1)
        raw_action = self.affine(combined) + self.deep(combined)
        cap = raw_action.new_tensor(self.config.position_cap)
        action = cap * torch.tanh(raw_action / cap)
        mask = active_mask[None] if active_mask.ndim == 2 else active_mask
        action = torch.where(mask, action, torch.zeros_like(action))
        self.verify(
            lambda: torch.isfinite(action),
            lambda: "GROMM policy produced non-finite actions",
        )
        return action
