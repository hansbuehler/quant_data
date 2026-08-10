"""Path-level marginal-utility measure change for simulated Dysanos universes.

Each model step optimizes utility of the complete path P&L and produces one
terminal density increment per simulated path.  The same path density weights
every time step in martingale and transaction-cost diagnostics.

All Monte Carlo variances use the population convention.  Paths are the
independent sampling units, so costs and reported Monte Carlo errors scale with
``sqrt(S_split)`` rather than ``sqrt(S_split * TAU)``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePath
from typing import Any

from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8

try:
    from cdxcore.uniquehash import label_unique_filename48_8
except ImportError:  # Compatibility with cdxcore releases before this explicit alias.
    from cdxcore.uniquehash import unique_label48_8

    def label_unique_filename48_8(label: str) -> str:
        return unique_label48_8(label, as_file_name=True)
import numpy as np
import torch
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock
from scipy.optimize import OptimizeResult, minimize
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel, DeviceLock
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility

_CHECKPOINT_SCHEMA = 5
_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
_EXCESS_COST_MULTIPLES = (1.0, 2.0, 3.0, 4.0)


def _finite_nonnegative(value: float | None, name: str, *, allow_none: bool = False) -> None:
    if value is None:
        verify_inp(allow_none, lambda: f"'{name}' cannot be None")
        return
    verify_inp(
        math.isfinite(value) and value >= 0.0,
        lambda: f"'{name}' must be finite and nonnegative, got {value}",
    )


def _valid_group(group: str) -> bool:
    path = PurePath(group)
    return not path.is_absolute() and ".." not in path.parts and all(
        part not in {"", "."} and is_filename(part) for part in path.parts
    )


@version("3.8", dependencies=[torch_utility])
@dataclass(frozen=True, kw_only=True)
class DysanosConfig:
    """Logical OCE model and measure-change configuration."""

    label: str
    group: str = ""
    user_version: str = "0.1"
    utility: str = "powexp"
    lmbda: float = 1.0
    oce_variant: str = "simplified"
    action_type: str = "network"
    deterministic_precondition: bool = False
    deterministic_preprocess: bool = False
    state_mode: str = "full"
    zero_action_initialization: bool = True
    trade_spot: bool = True
    delta_hedged: bool = True
    cash_returns: bool = False
    learn_cost: bool = True
    cost_cond: float | None = None
    cost_uncond: float = 1.0
    cost_prop: float = 0.0001
    seed: int = 1_231_231
    pf_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=100,
        activation="LeakyReLU",
        use_bias=True,
    )
    moment_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=100,
        activation="LeakyReLU",
        use_bias=True,
    )
    action_scale: float = 1.0
    drift_scale: float = 1.0
    sqr_scale: float = 1.0
    cash_floor: float = 1.0e-8
    density_floor: float = 1.0e-12
    denominator_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        verify_inp(isinstance(self.label, str) and self.label != "", "'label' must be set")
        verify_inp(
            (self.label.startswith("*") and len(self.label) > 1)
            or (not self.label.startswith("*") and is_filename(self.label)),
            lambda: f"'label' must be a filename or start with '*', got {self.label!r}",
        )
        verify_inp(
            self.group == "" or _valid_group(self.group),
            lambda: f"'group' must be a relative directory structure, got {self.group!r}",
        )
        verify_inp(
            self.utility in torch_utility.utilities,
            lambda: f"'utility' must be one of {torch_utility.utilities}, got {self.utility!r}",
        )
        verify_inp(
            self.oce_variant in {"simplified", "global_y"},
            lambda: (
                "'oce_variant' must be 'simplified' or 'global_y', got "
                f"{self.oce_variant!r}"
            ),
        )
        verify_inp(
            self.oce_variant == "simplified" or self.utility != "cvar",
            "the mechanical CVaR objective does not support a global cash offset",
        )
        verify_inp(
            self.action_type in {"network", "deterministic"},
            lambda: (
                "'action_type' must be 'network' or 'deterministic', got "
                f"{self.action_type!r}"
            ),
        )
        verify_inp(
            isinstance(self.deterministic_precondition, bool),
            lambda: (
                "'deterministic_precondition' must be a bool, got "
                f"{type(self.deterministic_precondition)}"
            ),
        )
        verify_inp(
            isinstance(self.deterministic_preprocess, bool),
            lambda: (
                "'deterministic_preprocess' must be a bool, got "
                f"{type(self.deterministic_preprocess)}"
            ),
        )
        verify_inp(
            not self.deterministic_preprocess
            or (
                self.action_type == "deterministic"
                and self.oce_variant == "simplified"
                and self.utility in {"entropy", "powexp"}
            ),
            (
                "deterministic per-instrument preprocessing requires a "
                "simplified deterministic entropy or power-exponential model"
            ),
        )
        verify_inp(
            self.state_mode in {"full", "raw"},
            lambda: (
                "'state_mode' must be 'full' or 'raw', got "
                f"{self.state_mode!r}"
            ),
        )
        verify_inp(
            isinstance(self.zero_action_initialization, bool),
            lambda: (
                "'zero_action_initialization' must be a bool, got "
                f"{type(self.zero_action_initialization)}"
            ),
        )
        verify_inp(
            isinstance(self.trade_spot, bool),
            lambda: f"'trade_spot' must be a bool, got {type(self.trade_spot)}",
        )
        verify_inp(
            math.isfinite(self.lmbda) and self.lmbda > 0.0,
            lambda: f"'lmbda' must be finite and strictly positive, got {self.lmbda}",
        )
        verify_inp(isinstance(self.seed, int), lambda: f"'seed' must be an int, got {self.seed!r}")

        if self.cost_cond is None:
            object.__setattr__(self, "cost_cond", 1.0 if self.learn_cost else 0.0)
        _finite_nonnegative(self.cost_cond, "cost_cond")
        _finite_nonnegative(self.cost_uncond, "cost_uncond")
        _finite_nonnegative(self.cost_prop, "cost_prop")
        for name in ("action_scale", "drift_scale", "sqr_scale"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )
        for name in ("cash_floor", "density_floor", "denominator_floor"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)

    def __str__(self) -> str:
        return (
            f"{self.label},u={self.utility}@{self.lmbda:g},dh={int(self.delta_hedged)},"
            f"oce={self.oce_variant},cash={int(self.cash_returns)},state={self.state_mode},cost={self.cost_cond:g}/"
            f"{self.cost_uncond:g}/{self.cost_prop:g}"
        )


@version("2.1")
@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Operational training configuration; ``epochs`` belongs to :func:`train`."""

    trn_batch_size: int | None = 1_024
    val_batch_size: int | None = 4_096
    validation_fraction: float = 0.10
    optimizer: str = "AdamW"
    lr: float = 1.0e-3
    final_lr: float = 1.0e-3
    moment_lr: float = 1.0e-3
    moment_final_lr: float = 1.0e-5
    action_initialization_epochs: int = 0
    action_initialization_lr: float = 1.0e-3
    action_initialization_report: str | None = None
    action_initialization_max_cap_fraction: float = 0.25
    weight_decay: float = 1.0e-4
    clip_grad_norm: float | None = None
    compile: bool = False
    value_seconds: float = 10.0
    value_update: int = 100
    checkpoint_seconds: float = 60.0
    checkpoint_update: int = 500
    heavy_output_update: int = 5_000
    early_stopping: bool = False
    minimum_epochs: int = 0
    minimum_moment_epochs: int = 0
    convergence_points: int = 5
    convergence_relative_tolerance: float = 5.0e-3
    require_cost_band: bool = False
    cost_band_tolerance: float = 0.0
    scipy_gradient_tolerance: float = 1.0e-10
    scipy_function_tolerance: float = 0.0
    scipy_max_line_search_steps: int = 50
    checkpoint_root: str = "~/.cache/rndysanos/oce"

    def __post_init__(self) -> None:
        for name in ("trn_batch_size", "val_batch_size"):
            value = getattr(self, name)
            verify_inp(
                value is None or value > 0,
                lambda name=name, value=value: f"'{name}' must be None or positive, got {value}",
            )
        verify_inp(
            0.0 <= self.validation_fraction < 1.0,
            lambda: "'validation_fraction' must be in [0, 1), "
            f"got {self.validation_fraction}",
        )
        verify_inp(
            self.optimizer in {"Adam", "AdamW", "LBFGS", "SciPy"},
            lambda: (
                "'optimizer' must be 'Adam', 'AdamW', 'LBFGS', or 'SciPy', got "
                f"{self.optimizer!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.lr) and self.lr > 0.0,
            lambda: f"'lr' must be finite and positive, got {self.lr}",
        )
        verify_inp(
            self.final_lr == self.lr,
            lambda: (
                "the main expected-utility learning rate is constant; "
                f"'final_lr' must equal 'lr', got {self.final_lr} and {self.lr}"
            ),
        )
        for initial, final, prefix in (
            (self.moment_lr, self.moment_final_lr, "moment_"),
        ):
            verify_inp(
                math.isfinite(initial) and math.isfinite(final) and initial >= final > 0.0,
                lambda initial=initial, final=final, prefix=prefix: (
                    f"'{prefix}lr' and '{prefix}final_lr' must satisfy "
                    f"initial >= final > 0, got {initial} and {final}"
                ),
            )
        verify_inp(
            self.action_initialization_epochs >= 0,
            lambda: (
                "'action_initialization_epochs' must be nonnegative, got "
                f"{self.action_initialization_epochs}"
            ),
        )
        verify_inp(
            math.isfinite(self.action_initialization_lr)
            and self.action_initialization_lr > 0.0,
            lambda: (
                "'action_initialization_lr' must be finite and positive, got "
                f"{self.action_initialization_lr}"
            ),
        )
        verify_inp(
            self.action_initialization_report is None
            or (
                isinstance(self.action_initialization_report, str)
                and self.action_initialization_report != ""
            ),
            "'action_initialization_report' must be None or a non-empty path",
        )
        verify_inp(
            0.0 < self.action_initialization_max_cap_fraction < 1.0,
            lambda: (
                "'action_initialization_max_cap_fraction' must lie in (0,1), "
                f"got {self.action_initialization_max_cap_fraction}"
            ),
        )
        verify_inp(
            self.action_initialization_report is None
            or self.action_initialization_epochs > 0,
            "a no-arbitrage report warm start requires positive initialization epochs",
        )
        _finite_nonnegative(self.weight_decay, "weight_decay")
        verify_inp(
            self.optimizer not in {"LBFGS", "SciPy"}
            or self.weight_decay == 0.0,
            "deterministic quasi-Newton optimizers do not support optimizer weight decay",
        )
        _finite_nonnegative(self.clip_grad_norm, "clip_grad_norm", allow_none=True)
        verify_inp(
            self.clip_grad_norm is None,
            "gradient clipping is disabled; 'clip_grad_norm' must be None",
        )
        for name in ("value_seconds", "checkpoint_seconds"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )
        for name in (
            "value_update",
            "checkpoint_update",
            "heavy_output_update",
            "convergence_points",
        ):
            value = getattr(self, name)
            verify_inp(value > 0, lambda name=name, value=value: f"'{name}' must be positive, got {value}")
        for name in ("minimum_epochs", "minimum_moment_epochs"):
            value = getattr(self, name)
            verify_inp(
                value >= 0,
                lambda name=name, value=value: f"'{name}' must be nonnegative, got {value}",
            )
        _finite_nonnegative(
            self.convergence_relative_tolerance,
            "convergence_relative_tolerance",
        )
        _finite_nonnegative(self.cost_band_tolerance, "cost_band_tolerance")
        _finite_nonnegative(
            self.scipy_gradient_tolerance,
            "scipy_gradient_tolerance",
        )
        _finite_nonnegative(
            self.scipy_function_tolerance,
            "scipy_function_tolerance",
        )
        verify_inp(
            self.scipy_max_line_search_steps > 0,
            lambda: (
                "'scipy_max_line_search_steps' must be positive, got "
                f"{self.scipy_max_line_search_steps}"
            ),
        )
        verify_inp(
            isinstance(self.checkpoint_root, str) and self.checkpoint_root != "",
            "'checkpoint_root' must be a non-empty string",
        )

    def __str__(self) -> str:
        return (
            f"batch={self.trn_batch_size},val_batch={self.val_batch_size},"
            f"optimizer={self.optimizer},lr={self.lr:g} constant,moment_lr="
            f"{self.moment_lr:g}->{self.moment_final_lr:g},action_init="
            f"{self.action_initialization_epochs},action_init_report="
            f"{self.action_initialization_report is not None},compile={self.compile}"
        )


@dataclass(frozen=True)
class UniverseGeometry:
    samples: int
    steps: int
    expiries: int
    strikes: int
    state_variables: int
    include_contract_features: bool = True

    @property
    def instruments(self) -> int:
        return 1 + self.expiries * self.strikes

    @property
    def state_features(self) -> int:
        if not self.include_contract_features:
            return self.state_variables
        return self.state_variables + self.expiries * self.strikes + self.expiries


def _as_floating_tensor(value: Any, *, name: str, env: BaseEnvironment) -> Tensor:
    if isinstance(value, Tensor):
        verify_inp(value.dtype.is_floating_point, lambda: f"'{name}' must have floating dtype, got {value.dtype}")
        tensor = value.detach().to(device=env.device, dtype=env.fdtype)
    else:
        array = np.asarray(value)
        verify_inp(
            np.issubdtype(array.dtype, np.floating),
            lambda: f"'{name}' must have floating dtype, got {array.dtype}",
        )
        tensor = torch.as_tensor(array, dtype=env.fdtype, device=env.device)
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"'{name}' contains non-finite values")
    return tensor


def validate_universe(universe: Mapping[str, Any]) -> UniverseGeometry:
    """Validate all required fields, shapes, floating dtypes, and finiteness."""

    required = {"C", "dC", "dS", "delta", "expiries", "log_strikes", "spot", "xstate"}
    missing = required.difference(universe)
    verify_inp(not missing, lambda: f"universe is missing fields {sorted(missing)}")

    values: dict[str, Any] = {}
    for name in required:
        value = universe[name]
        if isinstance(value, Tensor):
            dtype_is_float = value.dtype.is_floating_point
            finite = torch.isfinite(value).all().item()
        else:
            value = np.asarray(value)
            dtype_is_float = np.issubdtype(value.dtype, np.floating)
            finite = np.isfinite(value).all()
        verify_inp(dtype_is_float, lambda name=name, value=value: f"'{name}' must be floating, got {value.dtype}")
        verify_inp(finite, lambda name=name: f"'{name}' contains non-finite values")
        values[name] = value

    dC = values["dC"]
    verify_inp(dC.ndim == 4, lambda: f"'dC' must have shape (S, TAU, T, N), got {dC.shape}")
    S, TAU, T, N = dC.shape
    verify_inp(min(S, TAU, T, N) > 0, lambda: f"all universe dimensions must be positive, got {dC.shape}")
    expected = (S, TAU, T, N)
    for name in ("C", "delta", "log_strikes"):
        verify_inp(
            values[name].shape == expected,
            lambda name=name: f"'{name}' must have shape {expected}, got {values[name].shape}",
        )
    for name in ("dS", "spot"):
        verify_inp(
            values[name].shape == (S, TAU),
            lambda name=name: f"'{name}' must have shape {(S, TAU)}, got {values[name].shape}",
        )
    verify_inp(
        values["expiries"].shape in {(1, 1, T), (T,)},
        lambda: f"'expiries' must have shape {(1, 1, T)} or {(T,)}, got {values['expiries'].shape}",
    )
    xstate = values["xstate"]
    verify_inp(
        xstate.ndim == 3 and xstate.shape[:2] in {(S, TAU), (S, TAU + 1)},
        lambda: f"'xstate' must have shape (S, TAU[, +1], XS), got {xstate.shape}",
    )
    verify_inp(xstate.shape[-1] > 0, lambda: f"'xstate' has no state variables: {xstate.shape}")
    return UniverseGeometry(S, TAU, T, N, xstate.shape[-1])


def instrument_returns_and_values(
    *,
    dS: Tensor,
    option_dS: Tensor | None = None,
    dC: Tensor,
    spot: Tensor,
    C: Tensor,
    delta: Tensor,
    delta_hedged: bool,
    cash_returns: bool,
    cash_floor: float,
) -> tuple[Tensor, Tensor]:
    """Construct ``dH`` and ``H`` for all four return/value modes.

    Raw spot and option values must be positive. For a delta-hedged option,
    ``dH = dC - delta*option_dS`` is the net P&L while the traded gross
    notional is ``H = |C| + |delta*S|``. ``option_dS`` may contain one spot
    change per option expiry; legacy one-step universes fall back to ``dS``.
    Option and hedge notionals must not cancel when computing proportional
    transaction costs or cash-normalized returns.
    """

    for name, value in (
        ("dS", dS),
        ("dC", dC),
        ("spot", spot),
        ("C", C),
        ("delta", delta),
    ):
        verify_inp(
            isinstance(value, Tensor) and value.dtype.is_floating_point,
            lambda name=name, value=value: f"'{name}' must be a floating Torch tensor, got {type(value)} / {getattr(value, 'dtype', None)}",
        )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"'{name}' contains non-finite values")

    verify_inp(dS.ndim == 2, lambda: f"'dS' must have shape (S, TAU), got {dS.shape}")
    verify_inp(dC.ndim == 4, lambda: f"'dC' must have shape (S, TAU, T, N), got {dC.shape}")
    verify_inp(
        C.shape == dC.shape == delta.shape,
        lambda: f"'C', 'dC', and 'delta' shapes differ: {C.shape}, {dC.shape}, {delta.shape}",
    )
    verify_inp(
        spot.shape == dS.shape,
        lambda: f"'spot' and 'dS' shapes differ: {spot.shape}, {dS.shape}",
    )
    verify_inp(
        math.isfinite(cash_floor) and cash_floor > 0.0,
        lambda: f"'cash_floor' must be finite and positive, got {cash_floor}",
    )
    S, TAU, T, N = dC.shape
    if option_dS is None:
        option_dS = dS[..., None]
    verify_inp(
        option_dS.shape == (S, TAU, T),
        lambda: (
            f"'option_dS' must have shape {(S, TAU, T)}, got "
            f"{option_dS.shape}"
        ),
    )
    if not torch.isfinite(option_dS).all().item():
        raise ValueError("'option_dS' contains non-finite values")
    if not torch.all(spot > 0.0).item() or not torch.all(C > 0.0).item():
        minimum = min(spot.amin().item(), C.amin().item())
        raise ValueError(
            "all unhedged instrument values must be positive; "
            f"found minimum {minimum:g}"
        )
    option_returns = (
        dC - delta * option_dS[..., None] if delta_hedged else dC
    )
    option_values = (
        C.abs() + (delta * spot[..., None, None]).abs()
        if delta_hedged
        else C
    )
    dH = torch.cat((dS[..., None], option_returns.reshape(S, TAU, T * N)), dim=-1)
    H = torch.cat((spot[..., None], option_values.reshape(S, TAU, T * N)), dim=-1)
    if not torch.isfinite(dH).all().item() or not torch.isfinite(H).all().item():
        raise ValueError("instrument returns or values contain non-finite values")
    if cash_returns:
        if torch.any(H == 0.0).item():
            count = torch.count_nonzero(H == 0.0).item()
            raise ValueError(
                f"cash returns require nonzero signed instrument values; found {count} zeros"
            )
        denominator = H.clamp_min(cash_floor)
        dH = dH / denominator
        H = torch.ones_like(H)
    if not torch.isfinite(dH).all().item():
        raise ValueError("cash-normalized instrument returns contain non-finite values")
    assert dH.shape == H.shape == (S, TAU, 1 + T * N), (
        f"unexpected instrument shapes {dH.shape} and {H.shape}"
    )
    return dH, H


def prepare_universe(
    universe: Mapping[str, Any],
    *,
    config: DysanosConfig,
    env: BaseEnvironment,
) -> tuple[dict[str, Tensor], UniverseGeometry]:
    """Validate and copy one :func:`prep_experiment` universe to Torch."""

    geometry = validate_universe(universe)
    tensors = {
        name: _as_floating_tensor(value, name=name, env=env)
        for name, value in universe.items()
        if name
        in {
            "C",
            "dC",
            "dS",
            "expiry_dS",
            "option_dS",
            "delta",
            "expiries",
            "log_strikes",
            "log_path_weight",
            "spot",
            "xstate",
        }
    }
    S, TAU, T, N = (
        geometry.samples,
        geometry.steps,
        geometry.expiries,
        geometry.strikes,
    )
    xstate = tensors["xstate"][:, :TAU]
    if config.state_mode == "raw":
        geometry = replace(geometry, include_contract_features=False)
        z = xstate
    else:
        expiries = tensors["expiries"].reshape(1, 1, T).expand(S, TAU, T)
        log_strikes = tensors["log_strikes"].reshape(S, TAU, T * N)
        z = torch.cat((xstate, log_strikes, expiries), dim=-1)
    dH, H = instrument_returns_and_values(
        dS=tensors["dS"],
        option_dS=tensors.get("option_dS", tensors.get("expiry_dS")),
        dC=tensors["dC"],
        spot=tensors["spot"],
        C=tensors["C"],
        delta=tensors["delta"],
        delta_hedged=config.delta_hedged,
        cash_returns=config.cash_returns,
        cash_floor=config.cash_floor,
    )
    assert z.shape == (S, TAU, geometry.state_features), (
        f"state shape {z.shape} != {(S, TAU, geometry.state_features)}"
    )
    if "log_path_weight" in tensors:
        log_path_weight = tensors["log_path_weight"]
        verify_inp(
            log_path_weight.shape == (S,),
            lambda: (
                f"'log_path_weight' must have shape {(S,)}, got "
                f"{log_path_weight.shape}"
            ),
        )
        sampling_weight = torch.exp(log_path_weight)
        if (
            not torch.isfinite(sampling_weight).all().item()
            or not torch.any(sampling_weight > 0.0).item()
        ):
            raise ValueError("importance-sampling weights are numerically invalid")
    else:
        sampling_weight = torch.ones(S, dtype=env.fdtype, device=env.device)
    data = {
        "z": z,
        "dH": dH,
        "H": H,
        "sampling_weight": sampling_weight,
    }
    if "excluded_directions" in universe:
        excluded = universe["excluded_directions"]
        if isinstance(excluded, Tensor):
            excluded_tensor = excluded.detach().to(device=env.device, dtype=torch.bool)
        else:
            excluded_array = np.asarray(excluded)
            verify_inp(
                excluded_array.dtype == np.bool_,
                lambda: (
                    "'excluded_directions' must have bool dtype, got "
                    f"{excluded_array.dtype}"
                ),
            )
            excluded_tensor = torch.as_tensor(
                np.ascontiguousarray(excluded_array),
                dtype=torch.bool,
                device=env.device,
            )
        expected_mask = (S, TAU, 2 * geometry.instruments)
        verify_inp(
            excluded_tensor.shape == expected_mask,
            lambda: (
                f"'excluded_directions' must have shape {expected_mask}, got "
                f"{excluded_tensor.shape}"
            ),
        )
        data["excluded_directions"] = excluded_tensor
    return data, geometry


def split_training_validation(
    data: Mapping[str, Tensor], validation_fraction: float
) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
    """Split paths once; no validation observation enters training calibration."""

    S = data["z"].shape[0]
    verify_inp(S > 0, "data must contain at least one path")
    num_validation = int(math.floor(S * validation_fraction))
    if validation_fraction > 0.0 and S > 1:
        num_validation = max(1, num_validation)
    num_validation = min(num_validation, S - 1)
    num_training = S - num_validation
    training = {name: value[:num_training] for name, value in data.items()}
    validation = (
        {name: value[num_training:] for name, value in data.items()}
        if num_validation > 0
        else None
    )

    # Each split represents P through a self-normalized importance sample.
    # Normalizing once here keeps uniform-Q minibatch gradients unbiased for
    # that fixed empirical objective without introducing batch normalization.
    for split in (training, validation):
        if split is None:
            continue
        weight = split["sampling_weight"]
        mean_weight = weight.mean(dtype=torch.float64)
        if not torch.isfinite(mean_weight).item() or mean_weight <= 0.0:
            raise ValueError("split importance weights must have positive finite mean")
        split["sampling_weight"] = weight / mean_weight.to(weight.dtype)
    return training, validation


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


@version("1.1")
class ScaledTanhNetwork(BaseModel):
    """A feed-forward network, ``tanh``, and learned positive output scale."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_input: int,
        num_output: int,
        config: FeedForwardConfig,
        initial_scale: float,
        scale_per_output: bool = False,
        nonnegative: bool = False,
    ) -> None:
        super().__init__(env)
        self.network = FeedForward(
            env,
            num_input=num_input,
            num_output=num_output,
            config=config,
        )
        scale_shape = (num_output,) if scale_per_output else ()
        self.raw_scale = torch.nn.Parameter(
            torch.full(
                scale_shape,
                _inverse_softplus(initial_scale),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.scale_per_output = scale_per_output
        self.nonnegative = nonnegative

    def zero_output_layer(self) -> None:
        """Initialize the policy at the economically neutral zero action."""

        output_layer = (
            self.network.In[-1] if len(self.network.In) > 0 else self.network.I0
        )
        with torch.no_grad():
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    @property
    def scale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_scale)

    def forward(self, z: Tensor) -> Tensor:
        bounded = torch.tanh(self.network(z))
        if self.nonnegative:
            bounded = 1.0 + bounded
        output = self.scale * bounded
        self.verify(lambda: torch.isfinite(output), lambda: f"non-finite {type(self).__name__} output")
        return output


@version("1.2")
class DeterministicTimeAction(BaseModel):
    """One unrestricted deterministic action vector for every path time.

    ``base_value`` holds an optional per-instrument optimum ``b`` in optimizer
    coordinates. ``value`` is the subsequently optimized correction ``a``.
    Their sum is the physical strategy after applying ``preconditioner``.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        steps: int,
        instruments: int,
    ) -> None:
        super().__init__(env)
        self.value = torch.nn.Parameter(
            torch.zeros(
                (steps, instruments),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.register_buffer(
            "preconditioner",
            torch.ones(
                (steps, instruments),
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        self.register_buffer(
            "base_value",
            torch.zeros(
                (steps, instruments),
                dtype=env.fdtype,
                device=env.device,
            ),
        )

    @torch.no_grad()
    def set_preconditioner(self, rms: Tensor, *, floor: float) -> None:
        """Use reciprocal return RMS as an optimizer-only coordinate scale."""

        if rms.shape != self.value.shape:
            raise ValueError(
                f"return RMS shape {rms.shape} != action shape {self.value.shape}"
            )
        if not torch.isfinite(rms).all().item() or not torch.all(rms >= 0.0).item():
            raise ValueError("return RMS must be finite and nonnegative")
        self.preconditioner.copy_(rms.clamp_min(floor).reciprocal())

    @torch.no_grad()
    def set_preprocessed_action(self, physical_action: Tensor) -> None:
        """Set the fixed per-instrument action ``b`` and reset correction ``a``."""

        if physical_action.shape != self.value.shape:
            raise ValueError(
                f"preprocessed action shape {physical_action.shape} != "
                f"{self.value.shape}"
            )
        if not torch.isfinite(physical_action).all().item():
            raise ValueError("preprocessed action must be finite")
        self.base_value.copy_(physical_action / self.preconditioner)
        self.value.zero_()

    @property
    def physical_base_action(self) -> Tensor:
        return self.base_value * self.preconditioner

    @property
    def physical_correction_action(self) -> Tensor:
        return self.value * self.preconditioner

    def forward(self, z: Tensor) -> Tensor:
        expected = (self.value.shape[0], z.shape[-1])
        self.verify(
            lambda: torch.tensor(
                z.shape[-2:] == expected,
                dtype=torch.bool,
                device=z.device,
            ),
            lambda: f"state shape {z.shape[-2:]} != expected {expected}",
        )
        physical_value = (self.base_value + self.value) * self.preconditioner
        output = physical_value.unsqueeze(0).expand(z.shape[0], -1, -1)
        self.verify(
            lambda: torch.isfinite(output),
            "deterministic action is non-finite",
        )
        return output


@version("2.0")
class UnitDensity(BaseModel):
    """First-stage terminal path density, identically one."""

    def forward(
        self,
        z: Tensor,
        dH: Tensor | None = None,
        H: Tensor | None = None,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        del dH, H, excluded_directions
        return torch.ones(z.shape[0], dtype=z.dtype, device=z.device)

    def path_density(
        self,
        z: Tensor,
        dH: Tensor | None = None,
        H: Tensor | None = None,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        del dH, H, excluded_directions
        return torch.ones(z.shape[0], dtype=z.dtype, device=z.device)

    def log_path_density(
        self,
        z: Tensor,
        dH: Tensor | None = None,
        H: Tensor | None = None,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        del dH, H, excluded_directions
        return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)


def conditional_variance(drift: Tensor, sqr: Tensor) -> Tensor:
    """Return ``max(SQR - DRIFT**2, 0)`` elementwise."""

    verify_inp(drift.shape == sqr.shape, lambda: f"moment shapes differ: {drift.shape} and {sqr.shape}")
    if not torch.isfinite(drift).all().item() or not torch.isfinite(sqr).all().item():
        raise ValueError("conditional moments contain non-finite values")
    return (sqr - drift.square()).clamp_min(0.0)


def cost_components(
    *,
    variance: Tensor,
    H: Tensor,
    bar_c: Tensor,
    n_cost: int,
    cost_cond: float,
    cost_prop: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return conditional, proportional, global, and effective costs."""

    verify_inp(n_cost > 0, lambda: f"'n_cost' must be positive, got {n_cost}")
    _finite_nonnegative(cost_cond, "cost_cond")
    _finite_nonnegative(cost_prop, "cost_prop")
    verify_inp(variance.shape == H.shape, lambda: f"variance/H shapes differ: {variance.shape}, {H.shape}")
    verify_inp(bar_c.shape == (H.shape[-1],), lambda: f"'bar_c' shape {bar_c.shape} is invalid")
    if not torch.isfinite(variance).all().item() or not torch.all(variance >= 0.0).item():
        raise ValueError("conditional variance must be finite and nonnegative")
    if not torch.isfinite(H).all().item():
        raise ValueError("instrument values must be finite")
    if not torch.isfinite(bar_c).all().item() or not torch.all(bar_c >= 0.0).item():
        raise ValueError("global costs must be finite and nonnegative")
    c_cond = cost_cond * torch.sqrt(variance / float(n_cost))
    c_prop = cost_prop * H.abs()
    global_cost = bar_c.reshape(1, 1, -1).expand_as(H)
    effective = torch.maximum(torch.maximum(c_cond, global_cost), c_prop)
    return c_cond, c_prop, global_cost, effective


def apply_direction_mask(action: Tensor, excluded_directions: Tensor | None) -> Tensor:
    """Zero long/short action components excluded by ``arb(z; omega)``."""

    if excluded_directions is None:
        return action
    expected = action.shape[:-1] + (2 * action.shape[-1],)
    verify_inp(
        excluded_directions.shape == expected,
        lambda: f"excluded-direction shape {excluded_directions.shape} != {expected}",
    )
    verify_inp(
        excluded_directions.dtype == torch.bool,
        lambda: (
            "excluded directions must have bool dtype, got "
            f"{excluded_directions.dtype}"
        ),
    )
    instruments = action.shape[-1]
    positive = action.clamp_min(0.0).masked_fill(
        excluded_directions[..., :instruments], 0.0
    )
    negative = (-action).clamp_min(0.0).masked_fill(
        excluded_directions[..., instruments:], 0.0
    )
    return positive - negative


def _call_observation_density(
    density: torch.nn.Module,
    z: Tensor,
    dH: Tensor,
    H: Tensor,
) -> Tensor:
    if isinstance(density, (UnitDensity, RealizedDensity)):
        output = density(z, dH, H)
    else:
        output = density(z)
    if output.shape == (z.shape[0],):
        output = output[:, None].expand(z.shape[:-1])
    if output.shape == z.shape[:-1] + (1,):
        output = output[..., 0]
    assert output.shape == z.shape[:-1], (
        f"previous density shape {output.shape} != {z.shape[:-1]}"
    )
    return output


def _call_path_density(
    density: torch.nn.Module,
    z: Tensor,
    dH: Tensor,
    H: Tensor,
    floor: float,
    excluded_directions: Tensor | None = None,
) -> Tensor:
    return torch.exp(
        _call_log_path_density(
            density,
            z,
            dH,
            H,
            floor,
            excluded_directions,
        )
    )


def _call_log_path_density(
    density: torch.nn.Module,
    z: Tensor,
    dH: Tensor,
    H: Tensor,
    floor: float,
    excluded_directions: Tensor | None = None,
) -> Tensor:
    log_path_method = getattr(density, "log_path_density", None)
    if log_path_method is not None:
        if isinstance(density, (UnitDensity, RealizedDensity)):
            output = log_path_method(z, dH, H, excluded_directions)
        else:
            output = log_path_method(z, dH, H)
        if output.shape == (z.shape[0], 1):
            output = output[:, 0]
        assert output.shape == (z.shape[0],), (
            f"previous log-path-density shape {output.shape} != {(z.shape[0],)}"
        )
        return output
    path_method = getattr(density, "path_density", None)
    if path_method is not None:
        if isinstance(density, (UnitDensity, RealizedDensity)):
            output = path_method(z, dH, H, excluded_directions)
        else:
            output = path_method(z, dH, H)
        output = torch.log(output.clamp_min(floor))
    else:
        observation = _call_observation_density(density, z, dH, H)
        output = torch.log(observation.clamp_min(floor)).sum(dim=1)
    if output.shape == (z.shape[0], 1):
        output = output[:, 0]
    assert output.shape == (z.shape[0],), (
        f"previous path-density shape {output.shape} != {(z.shape[0],)}"
    )
    return output


@version(
    "3.2",
    dependencies=[
        DysanosConfig,
        DeterministicTimeAction,
        ScaledTanhNetwork,
        UnitDensity,
        torch_utility,
    ],
)
class RealizedDensity(BaseModel):
    """Frozen terminal density produced by one complete-path OCE step."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        action: ScaledTanhNetwork | DeterministicTimeAction,
        drift: ScaledTanhNetwork | None,
        sqr: ScaledTanhNetwork | None,
        previous_density: torch.nn.Module,
        config: DysanosConfig,
        bar_c: Tensor,
        n_cost: int,
        path_normalizer: Tensor,
        cash_offset: Tensor,
        tail_cutoff: Tensor,
        tail_boundary_fraction: Tensor,
    ) -> None:
        super().__init__(env)
        self.action = action
        self.drift = drift
        self.sqr = sqr
        self.previous_density = previous_density
        self.config = config
        self.n_cost = int(n_cost)
        _, self.utility_derivative, _ = torch_utility(
            config.utility,
            config.lmbda,
        )
        self.register_buffer("bar_c", bar_c.detach().clone())
        self.register_buffer("path_normalizer", path_normalizer.detach().clone())
        self.register_buffer("cash_offset", cash_offset.detach().clone())
        self.register_buffer("tail_cutoff", tail_cutoff.detach().clone())
        self.register_buffer(
            "tail_boundary_fraction",
            tail_boundary_fraction.detach().clone(),
        )
        self.requires_grad_(False)
        self.eval()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the local utility closure from pickle/deepcopy payloads."""

        state = super().__getstate__().copy()
        state.pop("utility_derivative", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the utility closure outside any compiled forward pass."""

        super().__setstate__(state)
        _, self.utility_derivative, _ = torch_utility(
            self.config.utility,
            self.config.lmbda,
        )

    def moments(self, z: Tensor, dH: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.drift is None:
            zeros = torch.zeros_like(dH)
            return zeros, zeros, zeros
        assert self.sqr is not None
        drift = self.drift(z)
        sqr = self.sqr(z)
        variance = (sqr - drift.square()).clamp_min(0.0)
        return drift, sqr, variance

    def costs(self, z: Tensor, dH: Tensor, H: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if float(self.config.cost_cond) > 0.0:
            _, _, variance = self.moments(z, dH)
            c_cond = float(self.config.cost_cond) * torch.sqrt(
                variance / float(self.n_cost)
            )
        else:
            c_cond = torch.zeros((), dtype=z.dtype, device=z.device).expand_as(H)
        c_prop = self.config.cost_prop * H.abs()
        bar_c = self.bar_c.reshape(1, 1, -1).expand_as(H)
        effective = torch.maximum(torch.maximum(c_cond, bar_c), c_prop)
        self.verify(
            lambda: torch.isfinite(effective) & (effective >= 0.0),
            "realized-density cost is invalid",
        )
        return c_cond, c_prop, bar_c, effective

    def path_increment(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        action = apply_direction_mask(self.action(z), excluded_directions)
        _, _, _, cost = self.costs(z, dH, H)
        step_pnl = (action * dH - action.abs() * cost).sum(dim=-1)
        path_pnl = step_pnl.sum(dim=1)
        if self.config.utility == "cvar":
            increment = torch.where(
                path_pnl < self.tail_cutoff,
                torch.ones_like(path_pnl),
                torch.where(
                    path_pnl == self.tail_cutoff,
                    self.tail_boundary_fraction.expand_as(path_pnl),
                    torch.zeros_like(path_pnl),
                ),
            )
        else:
            increment = self.utility_derivative(path_pnl, self.cash_offset)
        self.verify(
            lambda: torch.isfinite(increment) & (increment >= 0.0),
            "path-level OCE density increment is invalid",
        )
        return increment

    def forward(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        return self.path_density(z, dH, H, excluded_directions)

    def path_density(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        density = torch.exp(
            self.log_path_density(z, dH, H, excluded_directions)
        )
        self.verify(
            lambda: torch.isfinite(density) & (density >= 0.0),
            "composed path density is invalid",
        )
        return density

    def log_path_density(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        previous_log_path = _call_log_path_density(
            self.previous_density,
            z,
            dH,
            H,
            self.config.density_floor,
            excluded_directions,
        )
        increment = self.path_increment(z, dH, H, excluded_directions)
        log_path = previous_log_path + torch.log(
            increment.clamp_min(self.config.density_floor)
        )
        log_path = log_path - torch.log(
            self.path_normalizer.clamp_min(self.config.density_floor)
        )
        self.verify(
            lambda: torch.isfinite(log_path),
            "composed log-path density is invalid",
        )
        return log_path


@version(
    "3.1",
    dependencies=[
        DysanosConfig,
        DeterministicTimeAction,
        ScaledTanhNetwork,
        RealizedDensity,
        torch_utility,
    ],
)
class DysanosModel(BaseModel):
    """Complete-path expected-utility model with an optional moment phase."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        geometry: UniverseGeometry,
        config: DysanosConfig,
        previous_density: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(env)
        self.geometry = geometry
        self.config = config
        if config.action_type == "network":
            self.action = ScaledTanhNetwork(
                env,
                num_input=geometry.state_features,
                num_output=geometry.instruments,
                config=config.pf_config,
                initial_scale=config.action_scale,
                scale_per_output=True,
            )
            if config.zero_action_initialization:
                self.action.zero_output_layer()
        else:
            self.action = DeterministicTimeAction(
                env,
                steps=geometry.steps,
                instruments=geometry.instruments,
            )
        if float(config.cost_cond) > 0.0:
            self.drift = ScaledTanhNetwork(
                env,
                num_input=geometry.state_features,
                num_output=geometry.instruments,
                config=config.moment_config,
                initial_scale=config.drift_scale,
            )
            self.sqr = ScaledTanhNetwork(
                env,
                num_input=geometry.state_features,
                num_output=geometry.instruments,
                config=config.moment_config,
                initial_scale=config.sqr_scale,
                nonnegative=True,
            )
        else:
            self.drift = None
            self.sqr = None
        self.previous_density = previous_density or UnitDensity(env)
        self.previous_density.requires_grad_(False)
        self.previous_density.eval()
        self.utility, self.utility_derivative, self.utility_label = torch_utility(
            config.utility, config.lmbda
        )
        if config.oce_variant == "global_y":
            self.cash_offset = torch.nn.Parameter(
                torch.zeros((), dtype=env.fdtype, device=env.device)
            )
        else:
            self.register_buffer(
                "cash_offset",
                torch.zeros((), dtype=env.fdtype, device=env.device),
            )
        self.register_buffer(
            "bar_variance",
            torch.zeros(geometry.instruments, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "bar_c",
            torch.zeros(geometry.instruments, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "n_cost",
            torch.zeros((), dtype=torch.int64, device=env.device),
        )
        self.register_buffer(
            "path_normalizer",
            torch.ones((), dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "tail_cutoff",
            torch.zeros((), dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "tail_boundary_fraction",
            torch.ones((), dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "tail_calibrated",
            torch.zeros((), dtype=torch.bool, device=env.device),
        )
        self._moments_frozen = self.drift is None

    def _previous_path(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> Tensor:
        previous = _call_path_density(
            self.previous_density,
            z,
            dH,
            H,
            self.config.density_floor,
            excluded_directions,
        )
        self.verify(
            lambda: torch.isfinite(previous) & (previous >= 0.0),
            "previous terminal path density is invalid",
        )
        return previous

    def moment_outputs(self, z: Tensor, dH: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.drift is None:
            zeros = torch.zeros_like(dH)
            return zeros, zeros, zeros
        assert self.sqr is not None
        drift = self.drift(z)
        sqr = self.sqr(z)
        variance = (sqr - drift.square()).clamp_min(0.0)
        self.verify(lambda: torch.isfinite(drift), "conditional DRIFT is non-finite")
        self.verify(lambda: torch.isfinite(sqr), "conditional SQR is non-finite")
        self.verify(
            lambda: torch.isfinite(variance) & (variance >= 0.0),
            "conditional variance is invalid",
        )
        return drift, sqr, variance

    def moment_loss(self, data: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if self.drift is None:
            raise RuntimeError("conditional-moment phase is disabled when cost_cond is zero")
        z, dH, H = data["z"], data["dH"], data["H"]
        with torch.no_grad():
            previous = self._previous_path(
                z,
                dH,
                H,
                data.get("excluded_directions"),
            )
            target = (previous[:, None, None] * dH).detach()
            target_sqr = target.square().detach()
        drift, sqr, _ = self.moment_outputs(z, dH)
        sampling_weight = data.get("sampling_weight")
        if sampling_weight is None:
            sampling_weight = torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
        weight = sampling_weight[:, None, None]
        denominator = sampling_weight.sum().clamp_min(
            self.config.density_floor
        ) * dH.shape[1] * dH.shape[2]
        drift_loss = (weight * (drift - target).square()).sum() / denominator
        sqr_loss = (weight * (sqr - target_sqr).square()).sum() / denominator
        loss = drift_loss + sqr_loss
        self.verify(lambda: torch.isfinite(loss), "conditional-moment loss is non-finite")
        return loss, drift_loss, sqr_loss

    def freeze_moments(self) -> None:
        """Freeze conditional moments before the main OCE optimizer can step."""

        for module in (self.drift, self.sqr):
            if module is not None:
                for parameter in module.parameters():
                    parameter.grad = None
                module.requires_grad_(False)
                module.eval()
        self._moments_frozen = True

    def main_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.action.parameters()
        if isinstance(self.cash_offset, torch.nn.Parameter):
            yield self.cash_offset

    def moment_parameters(self) -> Iterator[torch.nn.Parameter]:
        if self.drift is not None:
            yield from self.drift.parameters()
            assert self.sqr is not None
            yield from self.sqr.parameters()

    def costs(
        self,
        z: Tensor,
        dH: Tensor,
        H: Tensor,
        *,
        variance: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if float(self.config.cost_cond) > 0.0 and not self._moments_frozen:
            raise RuntimeError("conditional moments must be frozen before computing the OCE cost")
        if float(self.config.cost_cond) > 0.0:
            if variance is None:
                with torch.no_grad():
                    _, _, variance = self.moment_outputs(z, dH)
            variance = variance.detach()
            c_cond = float(self.config.cost_cond) * torch.sqrt(
                variance / self.n_cost.to(z.dtype)
            )
        else:
            c_cond = torch.zeros((), dtype=z.dtype, device=z.device).expand_as(H)
        c_prop = self.config.cost_prop * H.abs()
        global_cost = self.bar_c.reshape(1, 1, -1).expand_as(H)
        effective = torch.maximum(torch.maximum(c_cond, global_cost), c_prop)
        self.verify(
            lambda: torch.isfinite(effective) & (effective >= 0.0),
            "effective cost is invalid",
        )
        return c_cond, c_prop, global_cost, effective

    def _trading_outputs(self, data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        z, dH, H = data["z"], data["dH"], data["H"]
        sampling_weight = data.get("sampling_weight")
        if sampling_weight is None:
            sampling_weight = torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
        assert sampling_weight.shape == (z.shape[0],), (
            f"sampling-weight shape {sampling_weight.shape} != {(z.shape[0],)}"
        )
        excluded_directions = data.get("excluded_directions")
        action = apply_direction_mask(self.action(z), excluded_directions)
        if not self.config.trade_spot:
            action = torch.cat((torch.zeros_like(action[..., :1]), action[..., 1:]), dim=-1)
        drift, sqr, variance = self.moment_outputs(z, dH)
        if self._moments_frozen:
            drift, sqr, variance = drift.detach(), sqr.detach(), variance.detach()
        c_cond, c_prop, global_cost, cost = self.costs(
            z, dH, H, variance=variance
        )
        pnl = (action * dH - action.abs() * cost).sum(dim=-1)
        path_pnl = pnl.sum(dim=1)
        previous = self._previous_path(z, dH, H, excluded_directions)
        self.verify(lambda: torch.isfinite(action), "action is non-finite")
        return {
            "action": action,
            "drift": drift,
            "sqr": sqr,
            "variance": variance,
            "c_cond": c_cond,
            "c_prop": c_prop,
            "bar_c": global_cost,
            "cost": cost,
            "pnl": pnl,
            "path_pnl": path_pnl,
            "previous_density": previous,
            "sampling_weight": sampling_weight,
            "dH": dH,
            "H": H,
        }

    def pathwise(self, data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        output = self._trading_outputs(data)
        path_pnl = output["path_pnl"]
        previous = output["previous_density"]
        sampling_weight = output["sampling_weight"]
        if self.config.utility == "cvar":
            if not self.tail_calibrated.item():
                raise RuntimeError("CVaR tail cutoff has not been calibrated")
            increment = torch.where(
                path_pnl < self.tail_cutoff,
                torch.ones_like(path_pnl),
                torch.where(
                    path_pnl == self.tail_cutoff,
                    self.tail_boundary_fraction.expand_as(path_pnl),
                    torch.zeros_like(path_pnl),
                ),
            )
            utility = (
                sampling_weight
                * previous
                * increment
                * path_pnl
                / self.path_normalizer.clamp_min(self.config.density_floor)
            )
        else:
            utility = (
                sampling_weight
                * previous
                * self.utility(path_pnl, self.cash_offset)
            )
            increment = self.utility_derivative(path_pnl, self.cash_offset)
        density = previous * increment / self.path_normalizer.clamp_min(
            self.config.density_floor
        )
        self.verify(lambda: torch.isfinite(utility), "weighted utility is non-finite")
        self.verify(
            lambda: torch.isfinite(increment) & (increment >= 0.0),
            "path-level density increment is invalid",
        )
        self.verify(
            lambda: torch.isfinite(density) & (density >= 0.0),
            "new terminal path density is invalid",
        )
        output.update(
            utility=utility,
            density_increment=increment,
            density=density,
        )
        return output

    def forward(self, data: Mapping[str, Tensor]) -> Tensor:
        """Return negative previous-density-weighted complete-path utility."""

        output = self._trading_outputs(data)
        path_pnl = output["path_pnl"]
        previous = output["previous_density"]
        sampling_weight = output["sampling_weight"]
        if self.config.utility == "cvar":
            fraction = 1.0 / (1.0 + self.config.lmbda)
            measure_weight = sampling_weight * previous
            selected, _, _ = lower_tail_weights(
                path_pnl, measure_weight, fraction
            )
            selected_mass = (measure_weight * selected).sum().clamp_min(
                self.config.density_floor
            )
            objective = (
                measure_weight * selected * path_pnl
            ).sum() / selected_mass
        else:
            objective = (
                sampling_weight
                * previous
                * self.utility(path_pnl, self.cash_offset)
            ).mean()
        loss = -objective
        self.verify(lambda: torch.isfinite(loss), "main utility loss is non-finite")
        return loss

    @torch.no_grad()
    def calibrate_unconditional(
        self, data: Mapping[str, Tensor], *, batch_size: int | None = None
    ) -> None:
        """Calibrate population variance and global cost on full training data."""

        S, TAU, TNX = data["dH"].shape
        observations = S * TAU
        verify_inp(S > 0, "the full training experiment must contain paths")
        total = torch.zeros(TNX, dtype=torch.float64, device=self.device)
        total_sqr = torch.zeros_like(total)
        density_total = torch.zeros((), dtype=torch.float64, device=self.device)
        density_total_sqr = torch.zeros_like(density_total)
        sampling_mass = torch.zeros_like(density_total)
        condition_sqr = (
            torch.zeros(
                (TAU, TNX), dtype=torch.float64, device=self.device
            )
            if (
                self.config.deterministic_precondition
                and isinstance(self.action, DeterministicTimeAction)
            )
            else None
        )
        for batch in _iter_path_batches(data, batch_size):
            previous = self._previous_path(
                batch["z"],
                batch["dH"],
                batch["H"],
                batch.get("excluded_directions"),
            )
            if not torch.isfinite(previous).all().item() or not torch.all(previous >= 0.0).item():
                raise ValueError("previous density must be finite and nonnegative")
            weighted = previous[:, None, None] * batch["dH"]
            sampling_weight = batch["sampling_weight"]
            sampling = sampling_weight[:, None, None]
            total += (sampling * weighted).sum(dim=(0, 1), dtype=torch.float64)
            total_sqr += (sampling * weighted.square()).sum(
                dim=(0, 1), dtype=torch.float64
            )
            density_contribution = sampling_weight * previous
            density_total += density_contribution.sum(dtype=torch.float64)
            density_total_sqr += density_contribution.square().sum(
                dtype=torch.float64
            )
            sampling_mass += sampling_weight.sum(dtype=torch.float64)
            if condition_sqr is not None:
                condition_sqr += (
                    density_contribution[:, None, None]
                    * batch["dH"].square()
                ).sum(dim=0, dtype=torch.float64)
        previous_mean = density_total / float(S)
        previous_variance = (
            density_total_sqr / float(S) - previous_mean.square()
        ).clamp_min(0.0)
        previous_error = torch.sqrt(previous_variance / float(S))
        normalization_tolerance = torch.maximum(
            5.0 * previous_error,
            torch.full_like(previous_error, 1.0e-4),
        )
        if (torch.abs(previous_mean - 1.0) > normalization_tolerance).item():
            raise ValueError(
                "previous density is not normalized within five population-MC errors "
                "on the full training split; found mean "
                f"{previous_mean.item():.8g}, error {previous_error.item():.8g}"
            )
        weighted_observations = sampling_mass * TAU
        mean = total / weighted_observations
        variance = (
            total_sqr / weighted_observations - mean.square()
        ).clamp_min(0.0)
        variance = variance.to(self.fdtype)
        bar_c = self.config.cost_uncond * torch.sqrt(variance / float(S))
        if not torch.isfinite(variance).all().item() or not torch.all(variance >= 0.0).item():
            raise RuntimeError("unconditional sampling variance is invalid")
        if not torch.isfinite(bar_c).all().item() or not torch.all(bar_c >= 0.0).item():
            raise RuntimeError("global unconditional cost is invalid")
        self.bar_variance.copy_(variance)
        self.bar_c.copy_(bar_c)
        self.n_cost.fill_(S)
        if condition_sqr is not None:
            assert isinstance(self.action, DeterministicTimeAction)
            condition_rms = torch.sqrt(
                condition_sqr / density_total.clamp_min(
                    self.config.denominator_floor
                )
            ).to(self.fdtype)
            self.action.set_preconditioner(
                condition_rms,
                floor=self.config.denominator_floor,
            )

    @torch.no_grad()
    def calibrate_density(
        self, data: Mapping[str, Tensor], *, batch_size: int | None = None
    ) -> None:
        """Normalize the terminal density using the full training split only."""

        if self.config.utility == "cvar":
            pnl_chunks: list[Tensor] = []
            previous_chunks: list[Tensor] = []
            for batch in _iter_path_batches(data, batch_size):
                output = self._trading_outputs(batch)
                pnl_chunks.append(output["path_pnl"].detach().cpu())
                previous_chunks.append(
                    (
                        output["sampling_weight"]
                        * output["previous_density"]
                    ).detach().cpu()
                )
            fraction = 1.0 / (1.0 + self.config.lmbda)
            _, cutoff, boundary = lower_tail_weights(
                torch.cat(pnl_chunks),
                torch.cat(previous_chunks),
                fraction,
            )
            self.tail_cutoff.copy_(cutoff.to(self.device, self.fdtype))
            self.tail_boundary_fraction.copy_(
                boundary.to(self.device, self.fdtype)
            )
            self.tail_calibrated.fill_(True)

        path_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        sampling_mass = torch.zeros_like(path_sum)
        for batch in _iter_path_batches(data, batch_size):
            out = self.pathwise(batch)
            raw_path = out["previous_density"] * out["density_increment"]
            if not torch.isfinite(raw_path).all().item() or not torch.all(
                raw_path >= 0.0
            ).item():
                raise RuntimeError("raw terminal path density is invalid")
            sampling_weight = out["sampling_weight"]
            path_sum += (sampling_weight * raw_path).sum(dtype=torch.float64)
            sampling_mass += sampling_weight.sum(dtype=torch.float64)
        path_normalizer = path_sum / sampling_mass
        if path_normalizer <= 0.0 or not torch.isfinite(path_normalizer).item():
            raise RuntimeError("path-density normalizer must be strictly positive")
        self.path_normalizer.copy_(path_normalizer.to(self.fdtype))

    def density(self) -> RealizedDensity:
        """Return a frozen self-contained callable for this trained stage."""

        if self.n_cost.item() <= 0:
            raise RuntimeError("model costs have not been calibrated")
        density = RealizedDensity(
            self.env,
            action=copy.deepcopy(self.action),
            drift=copy.deepcopy(self.drift),
            sqr=copy.deepcopy(self.sqr),
            previous_density=copy.deepcopy(self.previous_density),
            config=self.config,
            bar_c=self.bar_c,
            n_cost=self.n_cost.item(),
            path_normalizer=self.path_normalizer,
            cash_offset=self.cash_offset,
            tail_cutoff=self.tail_cutoff,
            tail_boundary_fraction=self.tail_boundary_fraction,
        )
        return density


def oce_objective(
    *,
    action: Tensor,
    dH: Tensor,
    cost: Tensor,
    previous_density: Tensor,
    utility: str = "powexp",
    lmbda: float = 1.0,
    cash_offset: Tensor | None = None,
) -> Tensor:
    """Return the previous-density-weighted complete-path utility objective."""

    verify_inp(
        action.shape == dH.shape == cost.shape,
        lambda: f"action, dH, and cost shapes differ: {action.shape}, {dH.shape}, {cost.shape}",
    )
    verify_inp(
        previous_density.shape == (action.shape[0],),
        lambda: (
            f"previous-density shape {previous_density.shape} != "
            f"{(action.shape[0],)}"
        ),
    )
    for name, value in (
        ("action", action),
        ("dH", dH),
        ("cost", cost),
        ("previous_density", previous_density),
    ):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"'{name}' contains non-finite values")
    if not torch.all(cost >= 0.0).item():
        raise ValueError("'cost' must be nonnegative")
    if not torch.all(previous_density >= 0.0).item():
        raise ValueError("'previous_density' must be nonnegative")
    u, _, _ = torch_utility(utility, lmbda)
    path_pnl = (action * dH - action.abs() * cost).sum(dim=(-1, -2))
    y = path_pnl.new_zeros(()) if cash_offset is None else cash_offset
    verify_inp(
        y.ndim == 0 and y.dtype.is_floating_point,
        lambda: f"'cash_offset' must be a floating scalar, got {y.shape}/{y.dtype}",
    )
    if not torch.isfinite(y).item():
        raise ValueError("'cash_offset' must be finite")
    return (previous_density * u(path_pnl, y)).mean()


def _iter_path_batches(
    data: Mapping[str, Tensor], batch_size: int | None
) -> Iterator[dict[str, Tensor]]:
    S = data["z"].shape[0]
    size = S if batch_size is None else min(batch_size, S)
    for start in range(0, S, size):
        stop = min(start + size, S)
        yield {name: value[start:stop] for name, value in data.items()}


def _sample_paths(data: Mapping[str, Tensor], batch_size: int | None) -> dict[str, Tensor]:
    S = data["z"].shape[0]
    if batch_size is None or batch_size >= S:
        return dict(data)
    indices = torch.randint(S, (batch_size,), device=data["z"].device)
    return {name: value[indices] for name, value in data.items()}


def lower_tail_weights(
    values: Tensor,
    weights: Tensor,
    fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select exactly ``fraction`` of weight from the lowest sample values.

    The returned multipliers are one below the empirical cutoff, zero above
    it, and a common fractional multiplier for observations tied at the
    cutoff.  Selection is intentionally detached: it is the almost-everywhere
    gradient of the mechanically trimmed empirical expectation.
    """

    verify_inp(values.ndim == weights.ndim == 1, "values and weights must be vectors")
    verify_inp(values.shape == weights.shape, "values and weights must have equal shape")
    verify_inp(
        math.isfinite(fraction) and 0.0 < fraction <= 1.0,
        lambda: f"'fraction' must be in (0, 1], got {fraction}",
    )
    if not torch.isfinite(values).all().item():
        raise ValueError("tail-selection values contain non-finite entries")
    if not torch.isfinite(weights).all().item() or not torch.all(weights >= 0.0).item():
        raise ValueError("tail-selection weights must be finite and nonnegative")
    detached_values = values.detach()
    detached_weights = weights.detach()
    total = detached_weights.sum()
    if not (total > 0.0).item():
        raise ValueError("tail-selection weights must have positive total mass")
    order = torch.argsort(detached_values, stable=True)
    cumulative = torch.cumsum(detached_weights[order], dim=0)
    target = fraction * total
    cutoff_index = torch.searchsorted(cumulative, target, right=False).clamp_max(
        values.numel() - 1
    )
    cutoff = detached_values[order[cutoff_index]]
    below = detached_values < cutoff
    equal = detached_values == cutoff
    below_mass = detached_weights[below].sum()
    equal_mass = detached_weights[equal].sum()
    boundary = ((target - below_mass) / equal_mass.clamp_min(torch.finfo(weights.dtype).tiny)).clamp(
        0.0, 1.0
    )
    multipliers = torch.where(
        below,
        torch.ones_like(detached_weights),
        torch.where(equal, boundary.expand_as(detached_weights), torch.zeros_like(detached_weights)),
    )
    return multipliers, cutoff, boundary


def entropy_diagonal_mean_variance_action(
    *,
    drift: Tensor,
    variance: Tensor,
    cost: Tensor,
    lmbda: float,
    cap: Tensor,
    variance_floor: float,
) -> Tensor:
    """Return the diagonal-Gaussian entropy/Kelly strategy approximation.

    For conditionally Gaussian, cross-instrument independent returns, the
    entropy-OCE certainty equivalent for one instrument is
    ``a*mu - abs(a)*cost - lmbda*a**2*variance/2``.  Its optimizer is the
    cost-soft-thresholded ratio of conditional mean to conditional variance.
    """

    verify_inp(
        drift.shape == variance.shape == cost.shape,
        lambda: (
            f"drift/variance/cost shapes differ: {drift.shape}, "
            f"{variance.shape}, {cost.shape}"
        ),
    )
    verify_inp(
        cap.shape == (drift.shape[-1],),
        lambda: f"cap shape {cap.shape} != {(drift.shape[-1],)}",
    )
    verify_inp(
        math.isfinite(lmbda) and lmbda > 0.0,
        lambda: f"'lmbda' must be finite and positive, got {lmbda}",
    )
    verify_inp(
        math.isfinite(variance_floor) and variance_floor > 0.0,
        lambda: f"'variance_floor' must be finite and positive, got {variance_floor}",
    )
    edge = (drift.abs() - cost).clamp_min(0.0)
    target = torch.sign(drift) * edge / (
        lmbda * variance.clamp_min(variance_floor)
    )
    interior_cap = 0.99 * cap.reshape(1, 1, -1)
    return torch.maximum(torch.minimum(target, interior_cap), -interior_cap)


def initialize_entropy_action(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    epochs: int,
    batch_size: int | None,
    lr: float,
) -> list[float]:
    """Fit the action network to the conditional diagonal entropy solution."""

    if epochs <= 0:
        return []
    if model.config.utility != "entropy":
        raise ValueError("mean/variance action initialization requires entropy utility")
    if not isinstance(model.action, ScaledTanhNetwork):
        raise ValueError(
            "mean/variance action initialization requires the state-network action"
        )
    if model.drift is None or model.sqr is None or not model._moments_frozen:
        raise RuntimeError(
            "mean/variance action initialization requires frozen conditional moments"
        )
    optimizer = torch.optim.AdamW(
        model.action.network.parameters(), lr=lr, weight_decay=0.0
    )
    losses: list[float] = []
    model.action.train()
    for epoch in range(epochs):
        batch = _sample_paths(data, batch_size)
        with torch.no_grad():
            drift, _, variance = model.moment_outputs(batch["z"], batch["dH"])
            _, _, _, cost = model.costs(
                batch["z"], batch["dH"], batch["H"], variance=variance
            )
            target = entropy_diagonal_mean_variance_action(
                drift=drift,
                variance=variance,
                cost=cost,
                lmbda=model.config.lmbda,
                cap=model.action.scale.detach(),
                variance_floor=model.config.denominator_floor,
            )
        optimizer.zero_grad(set_to_none=True)
        loss = (model.action(batch["z"]) - target).square().mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"non-finite entropy action-initialization loss at epoch {epoch + 1}"
            )
        loss.backward()
        _, nonfinite = _gradient_diagnostics(
            list(model.action.network.parameters())
        )
        if nonfinite:
            raise FloatingPointError(
                f"found {nonfinite} non-finite entropy initialization gradients "
                f"at epoch {epoch + 1}"
            )
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.action.eval()
    return losses


def _no_arb_local_basket_targets(
    model: DysanosModel,
    report_path: str | Path,
    *,
    maximum_cap_fraction: float,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Load reference-only local-basket states and map their option actions.

    The benchmark expiry basket has one spot leg per expiry, whereas the OCE
    policy has one explicit spot output and delta-hedges every option itself.
    The warm start therefore discards the benchmark spot legs, copies the
    option quantities, and leaves the OCE spot output at zero.  Audit outcomes
    are deliberately ignored: they are held-out evidence, not training data.
    """

    if not isinstance(model.action, ScaledTanhNetwork):
        raise ValueError("a no-arbitrage basket warm start requires a network action")
    if model.config.state_mode != "raw":
        raise ValueError("a no-arbitrage basket warm start requires state_mode='raw'")
    path = Path(report_path).expanduser()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read no-arbitrage report {path}: {exc}") from exc
    configuration = report.get("configuration", {})
    if configuration.get("return_horizon") != "expiry":
        raise ValueError("the no-arbitrage warm-start report must use expiry returns")
    if configuration.get("hedge") != "raw":
        raise ValueError(
            "the no-arbitrage warm-start report must contain raw baskets so its "
            "option quantities can seed delta-hedged OCE actions"
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rejected = surviving = 0
    option_count = model.geometry.expiries * model.geometry.strikes
    for time_report in report.get("conditional", {}).get("times", []):
        normalization = time_report.get("state_normalization", {})
        mean = np.asarray(normalization.get("reference_only_mean", []), dtype=np.float64)
        scale = np.asarray(
            normalization.get("reference_only_population_scale", []),
            dtype=np.float64,
        )
        if mean.shape != (model.geometry.state_variables,) or scale.shape != mean.shape:
            raise ValueError(
                "no-arbitrage state normalization has shape "
                f"{mean.shape}/{scale.shape}, expected "
                f"{(model.geometry.state_variables,)}"
            )
        candidates = time_report.get("local_conditional_baskets", {})
        for status in ("surviving", "rejected"):
            items = candidates.get(status, [])
            if status == "surviving":
                surviving += len(items)
            else:
                rejected += len(items)
            for item in items:
                center = np.asarray(item["state_region"]["center"], dtype=np.float64)
                source_action = np.asarray(
                    item["reference_solution"]["action"], dtype=np.float64
                )
                source_spot_count = source_action.size - option_count
                if center.shape != mean.shape or source_spot_count <= 0:
                    raise ValueError(
                        "no-arbitrage local-basket target has incompatible state/action "
                        f"geometry {center.shape}/{source_action.shape}"
                    )
                mapped = np.zeros(model.geometry.instruments, dtype=np.float64)
                mapped[1:] = source_action[source_spot_count:]
                states.append(mean + scale * center)
                actions.append(mapped)
    if not states:
        raise ValueError(f"no local-basket targets found in {path}")

    state = torch.as_tensor(
        np.stack(states), dtype=model.fdtype, device=model.device
    )
    target = torch.as_tensor(
        np.stack(actions), dtype=model.fdtype, device=model.device
    )
    cap = model.action.scale.detach().reshape(1, -1)
    target = torch.maximum(
        torch.minimum(target, maximum_cap_fraction * cap),
        -maximum_cap_fraction * cap,
    )
    cap_fraction = (target.abs() / cap).amax().item()
    metadata = {
        "source_report": str(path),
        "target_count": len(states),
        "source_surviving_candidates": surviving,
        "source_rejected_candidates": rejected,
        "source_audit_outcomes_used": False,
        "source_spot_mapping": (
            "discard all expiry-specific spot legs; initialize the explicit OCE "
            "spot action at zero; copy raw option quantities into delta-hedged options"
        ),
        "maximum_target_cap_fraction": float(cap_fraction),
        "minimum_target_tanh_derivative": float(1.0 - cap_fraction * cap_fraction),
    }
    return state, target, metadata


def initialize_no_arb_action(
    model: DysanosModel,
    report_path: str | Path,
    *,
    epochs: int,
    batch_size: int | None,
    lr: float,
    maximum_cap_fraction: float,
) -> dict[str, Any]:
    """Warm-start an OCE policy from reference-fitted local basket directions."""

    if epochs <= 0:
        raise ValueError("no-arbitrage action initialization requires positive epochs")
    state, target, metadata = _no_arb_local_basket_targets(
        model,
        report_path,
        maximum_cap_fraction=maximum_cap_fraction,
    )
    parameters = list(model.action.network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    rows = state.shape[0]
    active_batch_size = rows if batch_size is None else min(rows, batch_size)

    def relative_loss(prediction: Tensor, expected: Tensor) -> Tensor:
        denominator = expected.square().sum(dim=-1).clamp_min(
            model.config.denominator_floor
        )
        return ((prediction - expected).square().sum(dim=-1) / denominator).mean()

    model.action.train()
    with torch.no_grad():
        first_loss = float(relative_loss(model.action(state), target).item())
    for epoch in range(epochs):
        if active_batch_size == rows:
            batch_state, batch_target = state, target
        else:
            index = torch.randint(rows, (active_batch_size,), device=model.device)
            batch_state, batch_target = state[index], target[index]
        optimizer.zero_grad(set_to_none=True)
        loss = relative_loss(model.action(batch_state), batch_target)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"non-finite no-arbitrage action-initialization loss at epoch {epoch + 1}"
            )
        loss.backward()
        _, nonfinite = _gradient_diagnostics(parameters)
        if nonfinite:
            raise FloatingPointError(
                f"found {nonfinite} non-finite no-arbitrage initialization "
                f"gradients at epoch {epoch + 1}"
            )
        optimizer.step()
    model.action.eval()
    with torch.no_grad():
        fitted = model.action(state)
        final_loss = float(relative_loss(fitted, target).item())
        fitted_fraction = fitted.abs() / model.action.scale.detach().reshape(1, -1)
        maximum_fitted_fraction = float(fitted_fraction.amax().item())
        cosine = torch.nn.functional.cosine_similarity(fitted, target, dim=-1)
    return {
        **metadata,
        "epochs": epochs,
        "learning_rate": lr,
        "first_relative_mse": first_loss,
        "final_relative_mse": final_loss,
        "mean_direction_cosine": float(cosine.mean().item()),
        "minimum_direction_cosine": float(cosine.amin().item()),
        "maximum_fitted_cap_fraction": maximum_fitted_fraction,
        "minimum_fitted_tanh_derivative": float(
            1.0 - maximum_fitted_fraction * maximum_fitted_fraction
        ),
    }


def _safe_std_from_sums(total: Tensor, total_sqr: Tensor, n: int) -> Tensor:
    mean = total / float(n)
    return (total_sqr / float(n) - mean.square()).clamp_min(0.0).sqrt()


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return left-continuous weighted empirical quantiles."""

    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("weighted quantiles require matching one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("weighted quantile inputs must be finite")
    if np.any(weights < 0.0) or not float(weights.sum()) > 0.0:
        raise ValueError("weighted quantile weights must be nonnegative with positive mass")
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    targets = probabilities * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return sorted_values[np.minimum(indices, len(sorted_values) - 1)]


def cap_objective_derivative(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> np.ndarray:
    """Return the full-split OCE-objective derivative by action cap.

    The trainable parameter is the unconstrained ``raw_scale``.  Dividing its
    derivative by the softplus Jacobian reports the derivative with respect to
    the positive cap itself.  Batches are weighted by path count so the result
    equals the derivative of the full-split mean objective.
    """

    if not isinstance(model.action, ScaledTanhNetwork):
        raise ValueError("the deterministic action has no cap derivative")
    model.eval()
    raw_scale = model.action.raw_scale
    if raw_scale.shape != (model.geometry.instruments,):
        raise RuntimeError(
            "the action cap must have one entry per instrument; found "
            f"{raw_scale.shape}"
        )
    total = torch.zeros_like(raw_scale, dtype=torch.float64)
    paths = data["z"].shape[0]
    for batch in _iter_path_batches(data, batch_size):
        objective = model(batch)
        derivative = torch.autograd.grad(objective, raw_scale)[0]
        weight = batch["z"].shape[0] / float(paths)
        total += derivative.detach().to(torch.float64) * weight
    softplus_jacobian = torch.sigmoid(raw_scale.detach()).to(torch.float64)
    derivative_by_cap = total / softplus_jacobian.clamp_min(model.config.denominator_floor)
    output = derivative_by_cap.cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("non-finite action-cap derivative during valuation")
    return output


def objective_gradient_diagnostics(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> dict[str, float | int]:
    """Return the exact full-split OCE gradient at the current strategy.

    Path batches are weighted by their fraction of the split, so the result is
    the gradient of the full path-level mean rather than a mean of batch norms.
    A nonzero cap derivative can be a boundary/KKT effect while the complete
    action-parameter gradient directly measures stationarity of the fitted
    strategy.
    """

    model.eval()
    action_parameters = list(model.action.parameters())
    parameters = list(model.main_parameters())
    totals = [
        torch.zeros_like(parameter, dtype=torch.float64, device=model.device)
        for parameter in parameters
    ]
    paths = data["z"].shape[0]
    for batch in _iter_path_batches(data, batch_size):
        loss = model(batch)
        gradients = torch.autograd.grad(loss, parameters)
        weight = batch["z"].shape[0] / float(paths)
        for total, gradient in zip(totals, gradients, strict=True):
            total.add_(gradient.detach().to(torch.float64), alpha=weight)

    action_count = sum(parameter.numel() for parameter in action_parameters)
    action_totals = totals[: len(action_parameters)]
    cash_totals = totals[len(action_parameters) :]

    def summary(values: list[Tensor], count: int, prefix: str) -> dict[str, float | int]:
        square = sum(float(value.square().sum().item()) for value in values)
        absolute = sum(float(value.abs().sum().item()) for value in values)
        maximum = max((float(value.abs().max().item()) for value in values), default=0.0)
        nonfinite = sum(int(torch.count_nonzero(~torch.isfinite(value)).item()) for value in values)
        return {
            f"{prefix}_elements": count,
            f"{prefix}_nonfinite": nonfinite,
            f"{prefix}_l2": math.sqrt(square),
            f"{prefix}_rms": math.sqrt(square / max(count, 1)),
            f"{prefix}_mean_abs": absolute / max(count, 1),
            f"{prefix}_linf": maximum,
        }

    result = {
        **summary(action_totals, action_count, "action_gradient"),
    }
    if isinstance(model.action, DeterministicTimeAction):
        physical_action_totals = [
            action_totals[0]
            / model.action.preconditioner.to(torch.float64).clamp_min(
                model.config.denominator_floor
            )
        ]
        result.update(
            summary(
                physical_action_totals,
                action_count,
                "physical_action_gradient",
            )
        )
    if cash_totals:
        result.update(summary(cash_totals, 1, "cash_offset_gradient"))
        result["cash_offset"] = float(model.cash_offset.detach().item())
    combined_square = sum(float(value.square().sum().item()) for value in totals)
    result["combined_gradient_l2"] = math.sqrt(combined_square)
    if any(not math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError("non-finite full-split objective-gradient diagnostic")
    return result


def _series_has_converged(
    epochs: Sequence[int],
    values: Sequence[Mapping[str, Any]],
    key: str,
    *,
    points: int,
    relative_tolerance: float,
) -> bool:
    """Test the projected one-valuation change of the latest objective values."""

    count = min(len(epochs), len(values))
    if count < points:
        return False
    x = np.asarray(epochs[-points:], dtype=np.float64)
    y = np.asarray([value[key] for value in values[-points:]], dtype=np.float64)
    if not np.isfinite(y).all():
        return False
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    interval = float(np.median(np.diff(x)))
    scale = max(abs(float(y[-1])), float(np.mean(np.abs(y))), 1.0e-12)
    return abs(slope * interval) / scale <= relative_tolerance


def _moment_training_ready(tracker: TrainingTracker, config: TrainConfig) -> bool:
    if not tracker.moment_epochs or tracker.moment_epochs[-1] < config.minimum_moment_epochs:
        return False
    arguments = {
        "points": config.convergence_points,
        "relative_tolerance": config.convergence_relative_tolerance,
    }
    return _series_has_converged(
        tracker.moment_valuation_epochs,
        tracker.moment_training_values,
        "drift_loss_mean",
        **arguments,
    ) and _series_has_converged(
        tracker.moment_valuation_epochs,
        tracker.moment_training_values,
        "sqr_loss_mean",
        **arguments,
    )


def _main_training_ready(tracker: TrainingTracker, config: TrainConfig) -> bool:
    if not tracker.epochs or tracker.epochs[-1] < config.minimum_epochs:
        return False
    if not _series_has_converged(
        tracker.valuation_epochs,
        tracker.training_values,
        "loss",
        points=config.convergence_points,
        relative_tolerance=config.convergence_relative_tolerance,
    ):
        return False
    if not config.require_cost_band:
        return True
    if not tracker.training_values:
        return False
    latest = tracker.training_values[-1]
    return_on_cost = latest.get("return_on_cost_by_time", latest["return_on_cost"])
    return bool(
        np.all(np.abs(return_on_cost) <= 1.0 + config.cost_band_tolerance)
    )


@torch.no_grad()
def _validation_density_rescaling(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> tuple[Tensor, Tensor]:
    """Return validation-sample rescalings for current and prior densities.

    A trained strategy transfers to validation, but the denominator in
    ``D = raw_density / E[raw_density]`` does not.  It is an expectation under
    the population represented by the split and must therefore be estimated
    again on validation.  The prior density is rescaled separately so the
    reference diagnostics use a probability density on the same split too.
    """

    paths = data["z"].shape[0]
    density_sum = torch.zeros((), dtype=torch.float64, device=model.device)
    reference_sum = torch.zeros_like(density_sum)
    sampling_sum = torch.zeros_like(density_sum)
    for batch in _iter_path_batches(data, batch_size):
        output = model.pathwise(batch)
        sampling_weight = output["sampling_weight"]
        density_sum += (
            sampling_weight * output["density"]
        ).sum(dtype=torch.float64)
        reference_sum += (
            sampling_weight * output["previous_density"]
        ).sum(dtype=torch.float64)
        sampling_sum += sampling_weight.sum(dtype=torch.float64)
    if (
        not torch.isfinite(density_sum).item()
        or not torch.isfinite(reference_sum).item()
        or density_sum <= 0.0
        or reference_sum <= 0.0
    ):
        raise RuntimeError("validation density normalizers must be finite and positive")
    return sampling_sum / density_sum, sampling_sum / reference_sum


@torch.no_grad()
def valuate(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
    split: str,
) -> dict[str, Any]:
    """Run a full split valuation in path batches.

    Standard deviations are population standard deviations.  Paths are the
    independent sampling units, so ``err_dH`` divides by ``sqrt(S_split)``.
    """

    verify_inp(split in {"training", "validation"}, lambda: f"unknown split {split!r}")
    S, TAU, TNX = data["dH"].shape
    n = S * TAU
    device = model.device
    if split == "validation":
        density_rescaling, reference_density_rescaling = (
            _validation_density_rescaling(
                model,
                data,
                batch_size=batch_size,
            )
        )
    else:
        density_rescaling = torch.ones((), dtype=torch.float64, device=device)
        reference_density_rescaling = torch.ones_like(density_rescaling)
    zero_i = lambda: torch.zeros(TNX, dtype=torch.float64, device=device)
    zero_ti = lambda: torch.zeros((TAU, TNX), dtype=torch.float64, device=device)
    total_utility = torch.zeros((), dtype=torch.float64, device=device)
    sum_density = torch.zeros((), dtype=torch.float64, device=device)
    sqr_density = torch.zeros_like(sum_density)
    pre_normalization_density_sum = torch.zeros_like(sum_density)
    pre_normalization_density_sqr = torch.zeros_like(sum_density)
    density_min = torch.full((), math.inf, dtype=torch.float64, device=device)
    density_max = torch.full((), -math.inf, dtype=torch.float64, device=device)
    positive_increment_count = torch.zeros((), dtype=torch.float64, device=device)
    previous_active_mass = torch.zeros((), dtype=torch.float64, device=device)

    reference_density_sum = torch.zeros((), dtype=torch.float64, device=device)
    reference_density_sqr = torch.zeros_like(reference_density_sum)
    pre_normalization_reference_sum = torch.zeros_like(reference_density_sum)
    pre_normalization_reference_sqr = torch.zeros_like(reference_density_sum)
    reference_sum_dH, reference_sqr_dH = zero_i(), zero_i()
    reference_sum_dH_by_time, reference_sqr_dH_by_time = zero_ti(), zero_ti()
    reference_sum_density_cost = zero_i()
    reference_sum_density_cost_by_time = zero_ti()

    sum_dH, sqr_dH = zero_i(), zero_i()
    sum_dH_by_time, sqr_dH_by_time = zero_ti(), zero_ti()
    sum_density_cost = zero_i()
    sum_density_cost_by_time = zero_ti()
    sum_action, sqr_action = zero_i(), zero_i()
    max_action = torch.zeros(TNX, dtype=torch.float64, device=device)
    saturation_count = zero_i()
    bounded_action = isinstance(model.action, ScaledTanhNetwork)
    sum_c_cond, sum_c_prop, sum_cost = zero_i(), zero_i(), zero_i()
    bind_cond, bind_uncond, bind_prop = zero_i(), zero_i(), zero_i()
    outside_count, abs_violation_sum = zero_i(), zero_i()
    multiple_shape = (len(_EXCESS_COST_MULTIPLES), TNX)
    signed_excess_sum = torch.zeros(multiple_shape, dtype=torch.float64, device=device)
    signed_excess_sqr = torch.zeros_like(signed_excess_sum)
    unsigned_excess_sum = torch.zeros_like(signed_excess_sum)
    unsigned_excess_sqr = torch.zeros_like(signed_excess_sum)
    reference_signed_excess_sum = torch.zeros_like(signed_excess_sum)
    reference_signed_excess_sqr = torch.zeros_like(signed_excess_sum)
    reference_unsigned_excess_sum = torch.zeros_like(signed_excess_sum)
    reference_unsigned_excess_sqr = torch.zeros_like(signed_excess_sum)
    ratio_zero_denominator = zero_i()
    sum_drift, sum_sqr, sum_variance = zero_i(), zero_i(), zero_i()
    drift_error_sqr, sqr_error_sqr = zero_i(), zero_i()
    residual_sqr_sum = zero_i()
    nonfinite_density_count = 0
    sampling_weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    sampling_weight_sqr = torch.zeros_like(sampling_weight_sum)
    sampling_weight_min = torch.full_like(sampling_weight_sum, math.inf)
    sampling_weight_max = torch.zeros_like(sampling_weight_sum)

    density_chunks: list[np.ndarray] = []
    reference_density_chunks: list[np.ndarray] = []
    ratio_chunks: list[np.ndarray] = []
    excess_chunks: list[np.ndarray] = []
    path_density_chunks: list[np.ndarray] = []
    path_pnl_chunks: list[np.ndarray] = []
    path_pnl_weight_chunks: list[np.ndarray] = []

    for batch in _iter_path_batches(data, batch_size):
        out = model.pathwise(batch)
        sampling_weight = out["sampling_weight"]
        sampling_weight_sum += sampling_weight.sum(dtype=torch.float64)
        sampling_weight_sqr += sampling_weight.square().sum(dtype=torch.float64)
        sampling_weight_min = torch.minimum(
            sampling_weight_min, sampling_weight.amin().to(torch.float64)
        )
        sampling_weight_max = torch.maximum(
            sampling_weight_max, sampling_weight.amax().to(torch.float64)
        )
        sampling_i = sampling_weight[:, None, None]
        pre_normalization_density = sampling_weight * out["density"]
        density = pre_normalization_density * density_rescaling.to(
            pre_normalization_density.dtype
        )
        nonfinite = torch.count_nonzero(~torch.isfinite(density)).item()
        nonfinite_density_count += int(nonfinite)
        if nonfinite or not torch.all(density >= 0.0).item():
            raise RuntimeError("valuation produced a non-finite or negative density")
        density_i = density[:, None, None]
        increment = out["density_increment"]
        pre_normalization_reference = (
            sampling_weight * out["previous_density"]
        )
        reference_density = pre_normalization_reference * (
            reference_density_rescaling.to(pre_normalization_reference.dtype)
        )
        if not torch.isfinite(reference_density).all().item() or not torch.all(
            reference_density >= 0.0
        ).item():
            raise RuntimeError("valuation produced a non-finite reference density")
        reference_density_i = reference_density[:, None, None]
        active_increment = increment > 0.0
        positive_increment_count += (
            sampling_weight * active_increment
        ).sum(dtype=torch.float64)
        previous_active_mass += (
            reference_density * active_increment
        ).sum(dtype=torch.float64)
        weighted_dH = density_i * out["dH"]
        weighted_cost = density_i * out["cost"]
        reference_weighted_dH = reference_density_i * out["dH"]
        action = out["action"]
        abs_action = action.abs()

        batch_utility = out["utility"].sum(dtype=torch.float64)
        batch_utility_cpu = float(batch_utility.detach().cpu().item())
        if not math.isfinite(batch_utility_cpu):
            raise FloatingPointError(
                f"non-finite {split} OCE objective detected on the CPU during valuation"
            )
        total_utility += batch_utility
        sum_density += density.sum(dtype=torch.float64)
        sqr_density += density.square().sum(dtype=torch.float64)
        pre_normalization_density_sum += pre_normalization_density.sum(
            dtype=torch.float64
        )
        pre_normalization_density_sqr += pre_normalization_density.square().sum(
            dtype=torch.float64
        )
        reference_density_sum += reference_density.sum(dtype=torch.float64)
        reference_density_sqr += reference_density.square().sum(dtype=torch.float64)
        pre_normalization_reference_sum += pre_normalization_reference.sum(
            dtype=torch.float64
        )
        pre_normalization_reference_sqr += pre_normalization_reference.square().sum(
            dtype=torch.float64
        )
        density_min = torch.minimum(density_min, density.amin().to(torch.float64))
        density_max = torch.maximum(density_max, density.amax().to(torch.float64))
        # The pooled mean is the mean of each complete path's time-average.
        # Its MC error therefore uses the cross-path dispersion of that
        # time-average, preserving all dependence between dates on a path.
        path_mean_dH = weighted_dH.mean(dim=1)
        reference_path_mean_dH = reference_weighted_dH.mean(dim=1)
        sum_dH += path_mean_dH.sum(dim=0, dtype=torch.float64)
        sqr_dH += path_mean_dH.square().sum(dim=0, dtype=torch.float64)
        reference_sum_dH += reference_path_mean_dH.sum(dim=0, dtype=torch.float64)
        reference_sqr_dH += reference_path_mean_dH.square().sum(
            dim=0, dtype=torch.float64
        )
        sum_dH_by_time += weighted_dH.sum(dim=0, dtype=torch.float64)
        sqr_dH_by_time += weighted_dH.square().sum(dim=0, dtype=torch.float64)
        reference_sum_dH_by_time += reference_weighted_dH.sum(
            dim=0, dtype=torch.float64
        )
        reference_sqr_dH_by_time += reference_weighted_dH.square().sum(
            dim=0, dtype=torch.float64
        )
        sum_density_cost += weighted_cost.sum(dim=(0, 1), dtype=torch.float64)
        sum_density_cost_by_time += weighted_cost.sum(dim=0, dtype=torch.float64)
        reference_sum_density_cost += (
            reference_density_i * out["cost"]
        ).sum(dim=(0, 1), dtype=torch.float64)
        reference_sum_density_cost_by_time += (
            reference_density_i * out["cost"]
        ).sum(dim=0, dtype=torch.float64)
        sum_action += (sampling_i * action).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sqr_action += (sampling_i * action.square()).sum(
            dim=(0, 1), dtype=torch.float64
        )
        max_action = torch.maximum(max_action, abs_action.amax(dim=(0, 1)).to(torch.float64))
        if bounded_action:
            action_scale = model.action.scale.detach().to(torch.float64)
            saturation_count += (
                sampling_i * (abs_action >= 0.99 * action_scale)
            ).sum(dim=(0, 1), dtype=torch.float64)
        sum_c_cond += (sampling_i * out["c_cond"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sum_c_prop += (sampling_i * out["c_prop"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sum_cost += (sampling_i * out["cost"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        bind_cond += (
            sampling_i
            * (
                (out["c_cond"] >= out["bar_c"])
                & (out["c_cond"] >= out["c_prop"])
            )
        ).sum(dim=(0, 1), dtype=torch.float64)
        bind_uncond += (
            sampling_i
            * (
                (out["bar_c"] >= out["c_cond"])
                & (out["bar_c"] >= out["c_prop"])
            )
        ).sum(dim=(0, 1), dtype=torch.float64)
        bind_prop += (
            sampling_i
            * (
                (out["c_prop"] >= out["c_cond"])
                & (out["c_prop"] >= out["bar_c"])
            )
        ).sum(dim=(0, 1), dtype=torch.float64)

        absolute_return = out["dH"].abs()
        outside = absolute_return > out["cost"]
        violation = (absolute_return - out["cost"]).clamp_min(0.0)
        multiples = torch.as_tensor(
            _EXCESS_COST_MULTIPLES,
            dtype=out["cost"].dtype,
            device=out["cost"].device,
        ).reshape(-1, 1, 1, 1)
        multiple_violation = (
            absolute_return.unsqueeze(0) - multiples * out["cost"].unsqueeze(0)
        ).clamp_min(0.0)
        signed_multiple = (
            density_i.unsqueeze(0)
            * torch.sign(out["dH"]).unsqueeze(0)
            * multiple_violation
        )
        unsigned_multiple = density_i.unsqueeze(0) * multiple_violation
        reference_signed_multiple = (
            reference_density_i.unsqueeze(0)
            * torch.sign(out["dH"]).unsqueeze(0)
            * multiple_violation
        )
        reference_unsigned_multiple = (
            reference_density_i.unsqueeze(0) * multiple_violation
        )
        for total, values in (
            (signed_excess_sum, signed_multiple),
            (unsigned_excess_sum, unsigned_multiple),
            (reference_signed_excess_sum, reference_signed_multiple),
            (reference_unsigned_excess_sum, reference_unsigned_multiple),
        ):
            total += values.sum(dim=(1, 2), dtype=torch.float64)
        for total, values in (
            (signed_excess_sqr, signed_multiple),
            (unsigned_excess_sqr, unsigned_multiple),
            (reference_signed_excess_sqr, reference_signed_multiple),
            (reference_unsigned_excess_sqr, reference_unsigned_multiple),
        ):
            total += values.square().sum(dim=(1, 2), dtype=torch.float64)
        signed_excess = signed_multiple[0]
        outside_count += (sampling_i * outside).sum(
            dim=(0, 1), dtype=torch.float64
        )
        abs_violation_sum += (density_i * violation).sum(dim=(0, 1), dtype=torch.float64)

        valid_ratio = (density_i > 0.0) & (out["cost"] > model.config.denominator_floor)
        ratio_zero_denominator += (~valid_ratio).sum(dim=(0, 1), dtype=torch.float64)
        ratio = torch.where(valid_ratio, out["dH"] / out["cost"], torch.nan)

        previous_target = out["previous_density"][:, None, None] * out["dH"]
        target_sqr = previous_target.square()
        sum_drift += (sampling_i * out["drift"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sum_sqr += (sampling_i * out["sqr"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sum_variance += (sampling_i * out["variance"]).sum(
            dim=(0, 1), dtype=torch.float64
        )
        drift_error_sqr += (
            sampling_i * (out["drift"] - previous_target).square()
        ).sum(
            dim=(0, 1), dtype=torch.float64
        )
        sqr_error_sqr += (
            sampling_i * (out["sqr"] - target_sqr).square()
        ).sum(
            dim=(0, 1), dtype=torch.float64
        )
        residual_sqr_sum += (
            sampling_i * (previous_target - out["drift"]).square()
        ).sum(
            dim=(0, 1), dtype=torch.float64
        )

        density_chunks.append(density.reshape(-1).cpu().numpy())
        reference_density_chunks.append(reference_density.reshape(-1).cpu().numpy())
        ratio_chunks.append(ratio.reshape(-1, TNX).cpu().numpy())
        excess_chunks.append(signed_excess.reshape(-1, TNX).cpu().numpy())
        path_density_chunks.append(density.cpu().numpy())
        path_pnl_chunks.append(out["path_pnl"].cpu().numpy())
        path_pnl_weight_chunks.append(sampling_weight.cpu().numpy())

    density_values = np.concatenate(density_chunks)
    reference_density_values = np.concatenate(reference_density_chunks)
    ratio_values = np.concatenate(ratio_chunks, axis=0)
    excess_values = np.concatenate(excess_chunks, axis=0)
    path_density_values = np.concatenate(path_density_chunks)
    path_pnl_values = np.concatenate(path_pnl_chunks)
    path_pnl_weights = np.concatenate(path_pnl_weight_chunks).astype(
        np.float64, copy=False
    )
    path_pnl_weight_mass = float(path_pnl_weights.sum())
    path_pnl_mean = float(
        np.dot(path_pnl_weights, path_pnl_values) / path_pnl_weight_mass
    )
    path_pnl_variance = float(
        np.dot(path_pnl_weights, np.square(path_pnl_values - path_pnl_mean))
        / path_pnl_weight_mass
    )
    path_pnl_quantile_probabilities = np.asarray(
        [0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999],
        dtype=np.float64,
    )
    path_pnl_thresholds = np.asarray(
        [-1.0e-4, -1.0e-6, -1.0e-8, -1.0e-12, 0.0], dtype=np.float64
    )
    mean_dH = sum_dH / float(S)
    std_dH = _safe_std_from_sums(sum_dH, sqr_dH, S)
    err_dH = std_dH / math.sqrt(float(S))
    mean_dH_by_time = sum_dH_by_time / float(S)
    std_dH_by_time = _safe_std_from_sums(
        sum_dH_by_time, sqr_dH_by_time, S
    )
    err_dH_by_time = std_dH_by_time / math.sqrt(float(S))
    mean_density_cost = sum_density_cost / float(n)
    mean_density_cost_by_time = sum_density_cost_by_time / float(S)
    protected_cost = mean_density_cost.clamp_min(model.config.denominator_floor)
    return_on_cost = mean_dH / protected_cost
    return_on_cost_by_time = mean_dH_by_time / mean_density_cost_by_time.clamp_min(
        model.config.denominator_floor
    )
    reference_mean_dH = reference_sum_dH / float(S)
    reference_std_dH = _safe_std_from_sums(
        reference_sum_dH, reference_sqr_dH, S
    )
    reference_err_dH = reference_std_dH / math.sqrt(float(S))
    reference_mean_dH_by_time = reference_sum_dH_by_time / float(S)
    reference_std_dH_by_time = _safe_std_from_sums(
        reference_sum_dH_by_time, reference_sqr_dH_by_time, S
    )
    reference_err_dH_by_time = reference_std_dH_by_time / math.sqrt(float(S))
    reference_return_on_cost = reference_mean_dH / reference_sum_density_cost.div(
        float(n)
    ).clamp_min(model.config.denominator_floor)
    reference_return_on_cost_by_time = (
        reference_mean_dH_by_time
        / reference_sum_density_cost_by_time.div(float(S)).clamp_min(
            model.config.denominator_floor
        )
    )

    # Ratios with zero cost (or zero density) are excluded and their frequency
    # is reported explicitly instead of manufacturing infinite diagnostics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio_std = np.nanstd(ratio_values, axis=0, ddof=0)
        ratio_quantiles = np.nanquantile(ratio_values, _QUANTILES, axis=0)
    excess_std = np.std(excess_values, axis=0, ddof=0)
    excess_quantiles = np.quantile(excess_values, _QUANTILES, axis=0)
    action_mean = sum_action / float(n)
    action_std = _safe_std_from_sums(sum_action, sqr_action, n)
    density_mean = sum_density / float(S)
    density_std = _safe_std_from_sums(sum_density, sqr_density, S)
    pre_normalization_density_mean = pre_normalization_density_sum / float(S)
    pre_normalization_density_std = _safe_std_from_sums(
        pre_normalization_density_sum,
        pre_normalization_density_sqr,
        S,
    )
    density_err = density_std / math.sqrt(float(S))
    density_z = torch.where(
        density_err > 0.0,
        (density_mean - 1.0) / density_err,
        torch.zeros_like(density_err),
    )
    reference_density_mean = reference_density_sum / float(S)
    reference_density_std = _safe_std_from_sums(
        reference_density_sum, reference_density_sqr, S
    )
    reference_density_err = reference_density_std / math.sqrt(float(S))
    pre_normalization_reference_mean = pre_normalization_reference_sum / float(S)
    pre_normalization_reference_std = _safe_std_from_sums(
        pre_normalization_reference_sum,
        pre_normalization_reference_sqr,
        S,
    )
    reference_density_z = torch.where(
        reference_density_err > 0.0,
        (reference_density_mean - 1.0) / reference_density_err,
        torch.zeros_like(reference_density_err),
    )
    predicted_variance = sum_variance / float(n)
    realized_residual_variance = residual_sqr_sum / float(n)
    variance_calibration_ratio = realized_residual_variance / predicted_variance.clamp_min(
        model.config.denominator_floor
    )
    density_ess = (sum_density.square() / sqr_density.clamp_min(model.config.density_floor)).item()
    reference_density_ess = (
        reference_density_sum.square()
        / reference_density_sqr.clamp_min(model.config.density_floor)
    ).item()
    path_sum = float(path_density_values.sum(dtype=np.float64))
    path_sqr = float(np.square(path_density_values, dtype=np.float64).sum())
    path_ess = path_sum * path_sum / max(path_sqr, model.config.density_floor)

    def cpu(value: Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    result: dict[str, Any] = {
        "split": split,
        "loss": float((-total_utility / float(S)).item()),
        "utility": float((total_utility / float(S)).item()),
        "mean_dH": cpu(mean_dH),
        "std_dH": cpu(std_dH),
        "err_dH": cpu(err_dH),
        "mean_dH_over_err": cpu(
            torch.where(err_dH > 0.0, mean_dH / err_dH, torch.zeros_like(mean_dH))
        ),
        "mean_dH_by_time": cpu(mean_dH_by_time),
        "std_dH_by_time": cpu(std_dH_by_time),
        "err_dH_by_time": cpu(err_dH_by_time),
        "mean_dH_over_err_by_time": cpu(
            torch.where(
                err_dH_by_time > 0.0,
                mean_dH_by_time / err_dH_by_time,
                torch.zeros_like(mean_dH_by_time),
            )
        ),
        "std_convention": "population",
        "mc_error_convention": (
            "S_split independent complete paths; pooled errors use each path's "
            "time-average and time-resolved errors use S_split paths"
        ),
        "n_error_observations": S,
        "return_on_cost": cpu(return_on_cost),
        "return_on_cost_by_time": cpu(return_on_cost_by_time),
        "reference_mean_dH": cpu(reference_mean_dH),
        "reference_std_dH": cpu(reference_std_dH),
        "reference_err_dH": cpu(reference_err_dH),
        "reference_mean_dH_over_err": cpu(
            torch.where(
                reference_err_dH > 0.0,
                reference_mean_dH / reference_err_dH,
                torch.zeros_like(reference_mean_dH),
            )
        ),
        "reference_mean_dH_by_time": cpu(reference_mean_dH_by_time),
        "reference_std_dH_by_time": cpu(reference_std_dH_by_time),
        "reference_err_dH_by_time": cpu(reference_err_dH_by_time),
        "reference_mean_dH_over_err_by_time": cpu(
            torch.where(
                reference_err_dH_by_time > 0.0,
                reference_mean_dH_by_time / reference_err_dH_by_time,
                torch.zeros_like(reference_mean_dH_by_time),
            )
        ),
        "reference_return_on_cost": cpu(reference_return_on_cost),
        "reference_return_on_cost_by_time": cpu(
            reference_return_on_cost_by_time
        ),
        "return_on_cost_denominator": cpu(mean_density_cost),
        "sample_ratio_std": ratio_std,
        "sample_ratio_quantiles": ratio_quantiles,
        "sample_ratio_zero_denominator_fraction": cpu(ratio_zero_denominator / float(n)),
        "quantile_levels": np.asarray(_QUANTILES),
        "excess_mean": np.mean(excess_values, axis=0),
        "excess_std": excess_std,
        "excess_quantiles": excess_quantiles,
        "excess_cost_multiples": np.asarray(_EXCESS_COST_MULTIPLES),
        "signed_excess_mean_by_multiple": cpu(signed_excess_sum / float(n)),
        "signed_excess_std_by_multiple": cpu(
            _safe_std_from_sums(signed_excess_sum, signed_excess_sqr, n)
        ),
        "unsigned_excess_mean_by_multiple": cpu(unsigned_excess_sum / float(n)),
        "unsigned_excess_std_by_multiple": cpu(
            _safe_std_from_sums(unsigned_excess_sum, unsigned_excess_sqr, n)
        ),
        "unsigned_excess_mean_across_instruments": cpu(
            (unsigned_excess_sum / float(n)).mean(dim=1)
        ),
        "reference_signed_excess_mean_by_multiple": cpu(
            reference_signed_excess_sum / float(n)
        ),
        "reference_signed_excess_std_by_multiple": cpu(
            _safe_std_from_sums(
                reference_signed_excess_sum, reference_signed_excess_sqr, n
            )
        ),
        "reference_unsigned_excess_mean_by_multiple": cpu(
            reference_unsigned_excess_sum / float(n)
        ),
        "reference_unsigned_excess_std_by_multiple": cpu(
            _safe_std_from_sums(
                reference_unsigned_excess_sum, reference_unsigned_excess_sqr, n
            )
        ),
        "reference_unsigned_excess_mean_across_instruments": cpu(
            (reference_unsigned_excess_sum / float(n)).mean(dim=1)
        ),
        "mean_absolute_violation": cpu(abs_violation_sum / float(n)),
        "outside_cost_fraction": cpu(outside_count / float(n)),
        "density_mean": float(density_mean.item()),
        "density_std": float(density_std.item()),
        "density_err": float(density_err.item()),
        "density_mean_minus_one_over_err": float(density_z.item()),
        "density_min": float(density_min.item()),
        "density_max": float(density_max.item()),
        "density_quantiles": np.quantile(density_values, _QUANTILES),
        "density_ess": density_ess,
        "density_normalization_error": float(abs(sum_density.item() / float(S) - 1.0)),
        "density_pre_normalization_mean": float(
            pre_normalization_density_mean.item()
        ),
        "density_pre_normalization_std": float(
            pre_normalization_density_std.item()
        ),
        "density_split_rescaling_factor": float(density_rescaling.item()),
        "density_nonfinite_count": nonfinite_density_count,
        "density_measure": (
            "combined dP_new/dQ = (p/q) * dP_new/dP; all martingale "
            "means and MC errors are computed under the proposal Q"
        ),
        "sampling_weight_mean": float(
            (sampling_weight_sum / float(S)).item()
        ),
        "sampling_weight_min": float(sampling_weight_min.item()),
        "sampling_weight_max": float(sampling_weight_max.item()),
        "sampling_weight_ess": float(
            (
                sampling_weight_sum.square()
                / sampling_weight_sqr.clamp_min(model.config.density_floor)
            ).item()
        ),
        "density_increment_positive_fraction": float(
            (positive_increment_count / float(S)).item()
        ),
        "density_increment_zero_fraction": float(
            (1.0 - positive_increment_count / float(S)).item()
        ),
        "previous_measure_active_mass": float(
            (previous_active_mass / reference_density_sum.clamp_min(
                model.config.density_floor
            )).item()
        ),
        "reference_density_mean": float(reference_density_mean.item()),
        "reference_density_std": float(reference_density_std.item()),
        "reference_density_err": float(reference_density_err.item()),
        "reference_density_mean_minus_one_over_err": float(
            reference_density_z.item()
        ),
        "reference_density_quantiles": np.quantile(
            reference_density_values, _QUANTILES
        ),
        "reference_density_ess": reference_density_ess,
        "reference_density_pre_normalization_mean": float(
            pre_normalization_reference_mean.item()
        ),
        "reference_density_pre_normalization_std": float(
            pre_normalization_reference_std.item()
        ),
        "reference_density_split_rescaling_factor": float(
            reference_density_rescaling.item()
        ),
        "path_density_mean": float(np.mean(path_density_values)),
        "path_density_min": float(np.min(path_density_values)),
        "path_density_max": float(np.max(path_density_values)),
        "path_density_quantiles": np.quantile(path_density_values, _QUANTILES),
        "path_density_ess": path_ess,
        "path_density_normalization_error": float(abs(np.mean(path_density_values) - 1.0)),
        "path_pnl_mean_under_P": path_pnl_mean,
        "path_pnl_std_under_P": math.sqrt(max(path_pnl_variance, 0.0)),
        "path_pnl_minimum_sampled": float(np.min(path_pnl_values)),
        "path_pnl_maximum_sampled": float(np.max(path_pnl_values)),
        "path_pnl_quantile_probabilities": path_pnl_quantile_probabilities,
        "path_pnl_weighted_quantiles_under_P": _weighted_quantiles(
            path_pnl_values,
            path_pnl_weights,
            path_pnl_quantile_probabilities,
        ),
        "path_pnl_downside_thresholds": path_pnl_thresholds,
        "path_pnl_probability_below_threshold_under_P": np.asarray(
            [
                np.dot(path_pnl_weights, path_pnl_values < threshold)
                / path_pnl_weight_mass
                for threshold in path_pnl_thresholds
            ],
            dtype=np.float64,
        ),
        "action_mean": cpu(action_mean),
        "action_std": cpu(action_std),
        "action_max": cpu(max_action),
        "DRIFT": cpu(sum_drift / float(n)),
        "SQR": cpu(sum_sqr / float(n)),
        "v": cpu(predicted_variance),
        "drift_loss": cpu(drift_error_sqr / float(n)),
        "sqr_loss": cpu(sqr_error_sqr / float(n)),
        "variance_realized": cpu(realized_residual_variance),
        "variance_calibration_ratio": cpu(variance_calibration_ratio),
        "c_cond": cpu(sum_c_cond / float(n)),
        "c_prop": cpu(sum_c_prop / float(n)),
        "bar_c": cpu(model.bar_c),
        "c": cpu(sum_cost / float(n)),
        "binding_cond_fraction": cpu(bind_cond / float(n)),
        "binding_uncond_fraction": cpu(bind_uncond / float(n)),
        "binding_prop_fraction": cpu(bind_prop / float(n)),
        "bar_variance": cpu(model.bar_variance),
        "n_cost": int(model.n_cost.item()),
        "oce_variant": model.config.oce_variant,
        "cash_offset": float(model.cash_offset.detach().item()),
        "marginal_utility_training_normalizer": float(
            model.path_normalizer.detach().item()
        ),
        "cash_offset_first_order_residual": float(
            (
                pre_normalization_density_mean
                * model.path_normalizer.to(torch.float64)
                - pre_normalization_reference_mean
            ).item()
        ),
    }
    if bounded_action:
        result["action_cap"] = cpu(model.action.scale)
        result["action_cap_saturation_rate"] = cpu(
            saturation_count / float(n)
        )
    elif isinstance(model.action, DeterministicTimeAction):
        result["action_preconditioner"] = cpu(
            model.action.preconditioner
        )
        result["deterministic_preprocessed_action"] = cpu(
            model.action.physical_base_action
        )
        result["deterministic_correction_action"] = cpu(
            model.action.physical_correction_action
        )
        result["deterministic_total_action"] = cpu(
            model.action.physical_base_action
            + model.action.physical_correction_action
        )
    if model.config.utility == "cvar":
        result["tail_fraction"] = 1.0 / (1.0 + model.config.lmbda)
        result["tail_cutoff"] = float(model.tail_cutoff.item())
        result["tail_boundary_fraction"] = float(
            model.tail_boundary_fraction.item()
        )
    for name, value in result.items():
        if isinstance(value, (float, np.ndarray)) and not np.isfinite(value).all():
            # All-NaN ratio columns can only occur when every denominator is
            # excluded.  Preserve the explicit exclusion frequency and expose
            # finite zero placeholders to keep trackers/checkpoints robust.
            if name in {"sample_ratio_std", "sample_ratio_quantiles"}:
                result[name] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise RuntimeError(f"valuation output {name!r} is non-finite")
    return result


def valuate_with_cap_derivative(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
    split: str,
    include_objective_gradient: bool = False,
) -> dict[str, Any]:
    """Value a split and attach cap and optional strategy derivatives.

    The complete parameter-gradient diagnostic is intentionally reserved for
    the converged/final epoch because it requires an additional backward pass
    over every path in the split.
    """

    output = valuate(model, data, batch_size=batch_size, split=split)
    if isinstance(model.action, ScaledTanhNetwork):
        output["action_cap_objective_derivative"] = cap_objective_derivative(
            model,
            data,
            batch_size=batch_size,
        )
    if include_objective_gradient:
        output["objective_gradient"] = objective_gradient_diagnostics(
            model,
            data,
            batch_size=batch_size,
        )
    return output


@torch.no_grad()
def valuate_moments(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
    split: str,
) -> dict[str, Any]:
    """Value the conditional first/second moments without using OCE costs."""

    if model.drift is None:
        raise RuntimeError("conditional-moment model is not configured")
    S, TAU, TNX = data["dH"].shape
    n = S * TAU
    zero = lambda: torch.zeros(TNX, dtype=torch.float64, device=model.device)
    drift_sum, sqr_sum, variance_sum = zero(), zero(), zero()
    drift_loss, sqr_loss, residual_sqr = zero(), zero(), zero()
    for batch in _iter_path_batches(data, batch_size):
        previous = model._previous_path(
            batch["z"],
            batch["dH"],
            batch["H"],
            batch.get("excluded_directions"),
        )
        target = previous[:, None, None] * batch["dH"]
        target_sqr = target.square()
        drift, sqr, variance = model.moment_outputs(batch["z"], batch["dH"])
        drift_sum += drift.sum(dim=(0, 1), dtype=torch.float64)
        sqr_sum += sqr.sum(dim=(0, 1), dtype=torch.float64)
        variance_sum += variance.sum(dim=(0, 1), dtype=torch.float64)
        drift_loss += (drift - target).square().sum(dim=(0, 1), dtype=torch.float64)
        sqr_loss += (sqr - target_sqr).square().sum(dim=(0, 1), dtype=torch.float64)
        residual_sqr += (target - drift).square().sum(dim=(0, 1), dtype=torch.float64)
    variance_mean = variance_sum / float(n)
    residual_mean = residual_sqr / float(n)
    calibration = residual_mean / variance_mean.clamp_min(model.config.denominator_floor)

    def cpu(value: Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    result = {
        "split": split,
        "DRIFT": cpu(drift_sum / float(n)),
        "SQR": cpu(sqr_sum / float(n)),
        "v": cpu(variance_mean),
        "drift_loss": cpu(drift_loss / float(n)),
        "sqr_loss": cpu(sqr_loss / float(n)),
        "drift_loss_mean": float((drift_loss.sum() / float(n * TNX)).item()),
        "sqr_loss_mean": float((sqr_loss.sum() / float(n * TNX)).item()),
        "variance_realized": cpu(residual_mean),
        "variance_calibration_ratio": cpu(calibration),
        "observations": n,
    }
    for name, value in result.items():
        if isinstance(value, (float, np.ndarray)) and not np.isfinite(value).all():
            raise RuntimeError(f"moment valuation output {name!r} is non-finite")
    return result


@dataclass
class TrainingTracker:
    """Serializable training, valuation, gradient, and restart history."""

    moment_epochs: list[int] = field(default_factory=list)
    moment_batch_loss: list[float] = field(default_factory=list)
    moment_drift_batch_loss: list[float] = field(default_factory=list)
    moment_sqr_batch_loss: list[float] = field(default_factory=list)
    moment_lr: list[float] = field(default_factory=list)
    moment_elapsed_seconds: list[float] = field(default_factory=list)
    moment_valuation_epochs: list[int] = field(default_factory=list)
    moment_training_values: list[dict[str, Any]] = field(default_factory=list)
    moment_validation_values: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    batch_loss: list[float] = field(default_factory=list)
    batch_utility: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    nonfinite_gradient_count: list[int] = field(default_factory=list)
    batch_elapsed_seconds: list[float] = field(default_factory=list)
    valuation_epochs: list[int] = field(default_factory=list)
    training_values: list[dict[str, Any]] = field(default_factory=list)
    validation_values: list[dict[str, Any]] = field(default_factory=list)
    restart_epochs: list[int] = field(default_factory=list)
    elapsed_seconds: list[float] = field(default_factory=list)
    moment_converged_epoch: int | None = None
    main_converged_epoch: int | None = None
    action_initialization_completed: bool = False
    action_initialization_first_loss: float | None = None
    action_initialization_final_loss: float | None = None
    action_initialization: dict[str, Any] | None = None
    deterministic_preprocessing: dict[str, Any] | None = None
    deterministic_optimizer: dict[str, Any] | None = None


@dataclass
class TrainingResult:
    """Completed model, frozen density, split outputs, and persisted history."""

    model: DysanosModel
    density: RealizedDensity
    training: dict[str, Any]
    validation: dict[str, Any] | None
    tracker: TrainingTracker
    checkpoint_path: Path
    restarted: bool
    universe_index: int

    def __iter__(self) -> Iterator[Any]:
        """Allow the familiar ``model, training, validation = result`` form."""

        yield self.model
        yield self.training
        yield self.validation


def _rng_state_dict() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if state is None:
        return
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [
            saved.cpu() for saved in state["torch_cuda"][: torch.cuda.device_count()]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _scipy_parameter_minimize(
    parameter: torch.nn.Parameter,
    value_and_gradient: Callable[[], tuple[float, np.ndarray]],
    *,
    maxiter: int,
    gradient_tolerance: float,
    function_tolerance: float,
    max_line_search_steps: int,
) -> tuple[OptimizeResult, list[dict[str, Any]], float, np.ndarray]:
    """Minimize a GPU Torch objective with SciPy L-BFGS-B orchestration."""

    verify_inp(maxiter > 0, lambda: f"'maxiter' must be positive, got {maxiter}")
    shape = parameter.shape
    history: list[dict[str, Any]] = []
    cache_x: np.ndarray | None = None
    cache_value: float | None = None
    cache_gradient: np.ndarray | None = None

    def evaluate(flat_value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cache_x, cache_value, cache_gradient
        flat_value = np.asarray(flat_value, dtype=np.float64)
        if not np.isfinite(flat_value).all():
            raise FloatingPointError("SciPy proposed a non-finite action")
        with torch.no_grad():
            parameter.copy_(
                torch.as_tensor(
                    flat_value.reshape(shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
        value, gradient = value_and_gradient()
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError(
                "SciPy objective or gradient became non-finite during valuation"
            )
        cache_x = flat_value.copy()
        cache_value = float(value)
        cache_gradient = gradient.copy()
        return cache_value, cache_gradient

    def callback(flat_value: np.ndarray) -> None:
        nonlocal cache_x, cache_value, cache_gradient
        candidate = np.asarray(flat_value, dtype=np.float64)
        if cache_x is None or not np.array_equal(candidate, cache_x):
            evaluate(candidate)
        assert cache_value is not None and cache_gradient is not None
        history.append(
            {
                "loss": cache_value,
                "gradient_l2": float(np.linalg.norm(cache_gradient)),
                "gradient_linf": float(np.max(np.abs(cache_gradient))),
            }
        )

    initial = parameter.detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
    result = minimize(
        evaluate,
        initial,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": maxiter,
            "maxls": max_line_search_steps,
            "maxcor": 50,
            "gtol": gradient_tolerance,
            "ftol": function_tolerance,
        },
    )
    final_value, final_gradient = evaluate(np.asarray(result.x, dtype=np.float64))
    if not history or history[-1]["loss"] != final_value:
        history.append(
            {
                "loss": final_value,
                "gradient_l2": float(np.linalg.norm(final_gradient)),
                "gradient_linf": float(np.max(np.abs(final_gradient))),
            }
        )
    return result, history, final_value, final_gradient


def _scipy_summary(
    result: OptimizeResult,
    *,
    loss: float,
    gradient: np.ndarray,
    gradient_tolerance: float,
) -> dict[str, Any]:
    gradient = np.asarray(gradient, dtype=np.float64)
    return {
        "method": "scipy.optimize.minimize/L-BFGS-B",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "loss": float(loss),
        "gradient_elements": int(gradient.size),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_rms": float(np.sqrt(np.mean(np.square(gradient)))),
        "gradient_mean_abs": float(np.mean(np.abs(gradient))),
        "gradient_linf": float(np.max(np.abs(gradient))),
        "gradient_tolerance": float(gradient_tolerance),
        "gradient_tolerance_met": bool(
            np.max(np.abs(gradient)) <= gradient_tolerance
        ),
    }


def scipy_deterministic_preprocess(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
    maxiter: int,
    gradient_tolerance: float,
    function_tolerance: float,
    max_line_search_steps: int,
) -> dict[str, Any]:
    r"""Fit every ``b[t, omega]`` through ``sum E[u(b*dH)]``."""

    if not isinstance(model.action, DeterministicTimeAction):
        raise ValueError("per-instrument preprocessing requires deterministic actions")
    if model.config.utility not in {"entropy", "powexp"}:
        raise ValueError("per-instrument preprocessing requires a smooth utility")
    paths = data["dH"].shape[0]
    physical_action = torch.nn.Parameter(
        model.action.physical_base_action.detach().clone()
    )
    zero = physical_action.new_zeros(())

    def value_and_gradient() -> tuple[float, np.ndarray]:
        physical_action.grad = None
        total_loss = torch.zeros((), dtype=torch.float64, device=model.device)
        for batch in _iter_path_batches(data, batch_size):
            previous = model._previous_path(
                batch["z"],
                batch["dH"],
                batch["H"],
                batch.get("excluded_directions"),
            )
            action = physical_action.unsqueeze(0).expand_as(batch["dH"])
            action = apply_direction_mask(
                action,
                batch.get("excluded_directions"),
            )
            utility = model.utility(action * batch["dH"], zero)
            measure_weight = batch["sampling_weight"] * previous
            batch_loss = -(
                measure_weight[:, None, None] * utility
            ).sum() / float(paths)
            batch_loss.backward()
            total_loss += batch_loss.detach().to(torch.float64)
        if physical_action.grad is None:
            raise RuntimeError("per-instrument objective produced no action gradient")
        return (
            float(total_loss.item()),
            physical_action.grad.detach().cpu().numpy(),
        )

    result, _, loss, gradient = _scipy_parameter_minimize(
        physical_action,
        value_and_gradient,
        maxiter=maxiter,
        gradient_tolerance=gradient_tolerance,
        function_tolerance=function_tolerance,
        max_line_search_steps=max_line_search_steps,
    )
    model.action.set_preprocessed_action(physical_action.detach())
    summary = _scipy_summary(
        result,
        loss=loss,
        gradient=gradient,
        gradient_tolerance=gradient_tolerance,
    )
    summary["objective"] = "-sum_{t,omega} E[u(b[t,omega]*dH[t,omega])]"
    return summary


def scipy_deterministic_optimize(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
    maxiter: int,
    gradient_tolerance: float,
    function_tolerance: float,
    max_line_search_steps: int,
    objective_step: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r"""Fit correction ``a`` in ``E[u(sum((a+b)*dH))]`` with SciPy."""

    if not isinstance(model.action, DeterministicTimeAction):
        raise ValueError("SciPy deterministic optimization requires deterministic actions")
    if len(list(model.main_parameters())) != 1:
        raise ValueError("SciPy deterministic optimization supports no cash offset")
    step = model if objective_step is None else objective_step
    paths = data["dH"].shape[0]

    def value_and_gradient() -> tuple[float, np.ndarray]:
        model.action.value.grad = None
        total_loss = torch.zeros((), dtype=torch.float64, device=model.device)
        for batch in _iter_path_batches(data, batch_size):
            batch_loss = step(batch)
            if not torch.isfinite(batch_loss).item():
                raise FloatingPointError(
                    "deterministic objective became non-finite during valuation"
                )
            weight = batch["dH"].shape[0] / float(paths)
            (batch_loss * weight).backward()
            total_loss += batch_loss.detach().to(torch.float64) * weight
        gradient = model.action.value.grad
        if gradient is None:
            raise RuntimeError("deterministic objective produced no action gradient")
        return float(total_loss.item()), gradient.detach().cpu().numpy()

    result, history, loss, gradient = _scipy_parameter_minimize(
        model.action.value,
        value_and_gradient,
        maxiter=maxiter,
        gradient_tolerance=gradient_tolerance,
        function_tolerance=function_tolerance,
        max_line_search_steps=max_line_search_steps,
    )
    summary = _scipy_summary(
        result,
        loss=loss,
        gradient=gradient,
        gradient_tolerance=gradient_tolerance,
    )
    summary["objective"] = "-E[u(sum_{t,omega}((a+b)*dH))]"
    return summary, history


def _optimizer(
    parameters: Iterator[torch.nn.Parameter],
    *,
    config: TrainConfig,
    lr: float,
    optimizer_name: str | None = None,
) -> torch.optim.Optimizer:
    name = config.optimizer if optimizer_name is None else optimizer_name
    if name == "SciPy":
        # SciPy owns the deterministic update. This optimizer is only a
        # serializable state carrier for the shared checkpoint machinery.
        return torch.optim.SGD(parameters, lr=lr)
    if name == "LBFGS":
        return torch.optim.LBFGS(
            parameters,
            lr=lr,
            max_iter=1,
            max_eval=25,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
    optimizer_class = getattr(torch.optim, name)
    return optimizer_class(parameters, lr=lr, weight_decay=config.weight_decay)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    initial_lr: float,
    final_lr: float,
    epochs: int,
) -> torch.optim.lr_scheduler.ExponentialLR:
    gamma = 1.0 if epochs <= 1 else (final_lr / initial_lr) ** (1.0 / float(epochs - 1))
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def _gradient_diagnostics(parameters: Sequence[torch.nn.Parameter]) -> tuple[float, int]:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        nonfinite += int(torch.count_nonzero(~torch.isfinite(gradient)).item())
        finite_gradient = torch.nan_to_num(gradient.to(torch.float64))
        total += finite_gradient.square().sum()
    return float(torch.sqrt(total).item()), nonfinite


def entropy_log_moment_loss(
    model: DysanosModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> Tensor:
    r"""Return the stable full-split entropy objective.

    Minimizing ``log E[D_prev exp(-lambda * path_pnl)] / lambda`` has exactly
    the same action minimizer as maximizing expected exponential utility.
    Unlike the raw exponential moment, its scale remains finite during an
    L-BFGS line search.  The log-sum-exp is accumulated across all path batches
    before differentiation, so this is the exact full-training objective.
    """

    if model.config.utility != "entropy":
        raise ValueError("stable log-moment loss requires entropy utility")
    paths = data["z"].shape[0]
    verify_inp(paths > 0, "entropy log-moment data must contain paths")
    total_log_sum: Tensor | None = None
    for batch in _iter_path_batches(data, batch_size):
        output = model._trading_outputs(batch)
        previous = output["previous_density"]
        log_previous = torch.where(
            previous > 0.0,
            torch.log(previous),
            torch.full_like(previous, -torch.inf),
        )
        log_integrand = (
            log_previous - model.config.lmbda * output["path_pnl"]
        )
        batch_log_sum = torch.logsumexp(log_integrand, dim=0)
        total_log_sum = (
            batch_log_sum
            if total_log_sum is None
            else torch.logaddexp(total_log_sum, batch_log_sum)
        )
    assert total_log_sum is not None
    loss = (total_log_sum - math.log(float(paths))) / model.config.lmbda
    model.verify(lambda: torch.isfinite(loss), "entropy log-moment loss is non-finite")
    return loss


def _checkpoint_file(
    config: DysanosConfig,
    train_config: TrainConfig,
    universe_index: int,
) -> Path:
    root = Path(train_config.checkpoint_root).expanduser()
    if config.group:
        root = root.joinpath(*PurePath(config.group).parts)
    return root / f"{config.filename}-u{universe_index:03d}.pt"


def _checkpoint_compatible(state: Mapping[str, Any], config: DysanosConfig) -> str | None:
    checks = (
        ("schema", _CHECKPOINT_SCHEMA),
        ("model_version", DysanosModel.version.unique_id48),
        ("config_version", DysanosConfig.version.unique_id48),
        ("user_version", config.user_version),
        (
            "objective_semantics",
            f"complete-path-prev-weighted-utility-{config.oce_variant}-v2",
        ),
        (
            "density_semantics",
            "normalized-terminal-path-marginal-utility-v2",
        ),
    )
    for name, expected in checks:
        actual = state.get(name)
        if actual != expected:
            return f"checkpoint {name} {actual!r} does not match required {expected!r}"
    return None


def _atomic_save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_state(
    *,
    model: DysanosModel,
    config: DysanosConfig,
    train_config: TrainConfig,
    tracker: TrainingTracker,
    phase: str,
    main_epoch: int,
    moment_epoch: int,
    main_optimizer: torch.optim.Optimizer,
    main_scheduler: torch.optim.lr_scheduler.LRScheduler,
    moment_optimizer: torch.optim.Optimizer | None,
    moment_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, Any]:
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "model_version": DysanosModel.version.unique_id48,
        "config_version": DysanosConfig.version.unique_id48,
        "user_version": config.user_version,
        "objective_semantics": (
            f"complete-path-prev-weighted-utility-{config.oce_variant}-v2"
        ),
        "density_semantics": "normalized-terminal-path-marginal-utility-v2",
        "config": asdict(config),
        "train_config": asdict(train_config),
        "geometry": asdict(model.geometry),
        "phase": phase,
        "main_epoch": main_epoch,
        "moment_epoch": moment_epoch,
        "tracker": asdict(tracker),
        "model_state": model.state_dict(),
        "previous_density": copy.deepcopy(model.previous_density),
        "main_optimizer": main_optimizer.state_dict(),
        "main_scheduler": main_scheduler.state_dict(),
        "moment_optimizer": None if moment_optimizer is None else moment_optimizer.state_dict(),
        "moment_scheduler": None if moment_scheduler is None else moment_scheduler.state_dict(),
        "rng_state": _rng_state_dict(),
    }


def _train_one(
    universe: Mapping[str, Any],
    *,
    config: DysanosConfig,
    train_config: TrainConfig,
    epochs: int,
    moment_epochs: int,
    previous_density: torch.nn.Module | None,
    restart: bool,
    just_load: bool,
    env: BaseEnvironment,
    universe_index: int,
) -> TrainingResult:
    if (
        train_config.optimizer in {"LBFGS", "SciPy"}
        and config.action_type != "deterministic"
    ):
        raise ValueError(
            "quasi-Newton optimizers are reserved for the convex "
            "deterministic-time action problem"
        )
    if train_config.optimizer == "SciPy" and (
        config.oce_variant != "simplified"
        or config.utility not in {"entropy", "powexp"}
        or any(
            value != 0.0
            for value in (
                float(config.cost_cond),
                config.cost_uncond,
                config.cost_prop,
            )
        )
    ):
        raise ValueError(
            "SciPy deterministic optimization currently requires a smooth, "
            "cost-free, simplified entropy or power-exponential objective"
        )
    data, geometry = prepare_universe(universe, config=config, env=env)
    training, validation = split_training_validation(data, train_config.validation_fraction)
    checkpoint_path = _checkpoint_file(config, train_config, universe_index)
    checkpoint: dict[str, Any] | None = None
    restarted = False

    if checkpoint_path.exists() and not restart:
        candidate = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        incompatibility = _checkpoint_compatible(candidate, config)
        if incompatibility is None and candidate.get("geometry") != asdict(geometry):
            incompatibility = (
                f"checkpoint geometry {candidate.get('geometry')} does not match {asdict(geometry)}"
            )
        if incompatibility is None and candidate.get("config") != asdict(config):
            incompatibility = "checkpoint logical model configuration differs"
        if incompatibility is not None:
            if just_load:
                raise RuntimeError(f"cannot just-load {checkpoint_path}: {incompatibility}")
            warnings.warn(f"{incompatibility}; cold-starting {config.label!r}")
        else:
            checkpoint = candidate
            restarted = True
    elif just_load:
        raise RuntimeError(f"checkpoint {checkpoint_path} does not exist")

    if checkpoint is None:
        np.random.seed(config.seed + universe_index)
        torch.manual_seed(config.seed + universe_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + universe_index)
    elif previous_density is None:
        previous_density = checkpoint.get("previous_density")

    if previous_density is not None:
        previous_density = previous_density.to(env.device)
    model = DysanosModel(
        env,
        geometry=geometry,
        config=config,
        previous_density=previous_density,
    )
    main_parameters = list(model.main_parameters())
    moment_parameters = list(model.moment_parameters())
    main_optimizer = _optimizer(iter(main_parameters), config=train_config, lr=train_config.lr)
    main_scheduler = _scheduler(
        main_optimizer,
        initial_lr=train_config.lr,
        final_lr=train_config.final_lr,
        epochs=epochs,
    )
    moment_optimizer = (
        _optimizer(
            iter(moment_parameters),
            config=train_config,
            lr=train_config.moment_lr,
            optimizer_name=(
                "AdamW"
                if train_config.optimizer in {"LBFGS", "SciPy"}
                else None
            ),
        )
        if moment_parameters
        else None
    )
    moment_scheduler = (
        _scheduler(
            moment_optimizer,
            initial_lr=train_config.moment_lr,
            final_lr=train_config.moment_final_lr,
            epochs=moment_epochs,
        )
        if moment_optimizer is not None
        else None
    )
    tracker = TrainingTracker()
    phase = "moments" if moment_parameters else "main"
    main_epoch = 0
    moment_epoch = 0

    if checkpoint is not None:
        # Matching versions with an incompatible state dict indicate damage or
        # an unversioned architecture change and must not be silently ignored.
        model.load_state_dict(checkpoint["model_state"])
        main_optimizer.load_state_dict(checkpoint["main_optimizer"])
        main_scheduler.load_state_dict(checkpoint["main_scheduler"])
        if moment_optimizer is not None:
            moment_optimizer.load_state_dict(checkpoint["moment_optimizer"])
            assert moment_scheduler is not None
            moment_scheduler.load_state_dict(checkpoint["moment_scheduler"])
        tracker = TrainingTracker(**checkpoint["tracker"])
        phase = checkpoint["phase"]
        main_epoch = int(checkpoint["main_epoch"])
        moment_epoch = int(checkpoint["moment_epoch"])
        tracker.restart_epochs.append(main_epoch)
        _restore_rng_state(checkpoint.get("rng_state"))

    # This is recalculated from every full-training observation even on a warm
    # start.  It can never be influenced by the validation split or minibatches.
    model.calibrate_unconditional(training, batch_size=train_config.val_batch_size)
    if phase == "main":
        model.freeze_moments()
    if just_load:
        if phase != "main":
            raise RuntimeError(f"checkpoint {checkpoint_path} contains an unfinished moment phase")
        model.eval()
        model.calibrate_density(training, batch_size=train_config.val_batch_size)
        training_output = valuate_with_cap_derivative(
            model,
            training,
            batch_size=train_config.val_batch_size,
            split="training",
            include_objective_gradient=True,
        )
        validation_output = (
            valuate_with_cap_derivative(
                model,
                validation,
                batch_size=train_config.val_batch_size,
                split="validation",
                include_objective_gradient=True,
            )
            if validation is not None
            else None
        )
        return TrainingResult(
            model=model,
            density=model.density(),
            training=training_output,
            validation=validation_output,
            tracker=tracker,
            checkpoint_path=checkpoint_path,
            restarted=True,
            universe_index=universe_index,
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_value = started
    last_checkpoint = started
    lock = FileLock(f"{checkpoint_path}.lock")

    def save(current_phase: str) -> None:
        nonlocal last_checkpoint
        _atomic_save_checkpoint(
            checkpoint_path,
            _checkpoint_state(
                model=model,
                config=config,
                train_config=train_config,
                tracker=tracker,
                phase=current_phase,
                main_epoch=main_epoch,
                moment_epoch=moment_epoch,
                main_optimizer=main_optimizer,
                main_scheduler=main_scheduler,
                moment_optimizer=moment_optimizer,
                moment_scheduler=moment_scheduler,
            ),
        )
        last_checkpoint = time.monotonic()

    def moment_value_and_save() -> None:
        nonlocal last_value
        model.eval()
        training_value = valuate_moments(
            model,
            training,
            batch_size=train_config.val_batch_size,
            split="training",
        )
        validation_value = (
            valuate_moments(
                model,
                validation,
                batch_size=train_config.val_batch_size,
                split="validation",
            )
            if validation is not None
            else None
        )
        if not tracker.moment_valuation_epochs or tracker.moment_valuation_epochs[-1] != moment_epoch:
            tracker.moment_valuation_epochs.append(moment_epoch)
            tracker.moment_training_values.append(training_value)
            if validation_value is not None:
                tracker.moment_validation_values.append(validation_value)
        else:
            tracker.moment_training_values[-1] = training_value
            if validation_value is not None:
                tracker.moment_validation_values[-1] = validation_value
        tracker.elapsed_seconds.append(time.monotonic() - started)
        save("moments")
        last_value = time.monotonic()
        training_loss = tracker.moment_training_values[-1]["drift_loss_mean"]
        training_sqr_loss = tracker.moment_training_values[-1]["sqr_loss_mean"]
        validation_text = ""
        if tracker.moment_validation_values:
            last = tracker.moment_validation_values[-1]
            validation_text = (
                f", validation drift/sqr {last['drift_loss_mean']:.4g}/"
                f"{last['sqr_loss_mean']:.4g}"
            )
        print(
            f"Moment epoch {moment_epoch}/{moment_epochs}: full-training drift/sqr "
            f"{training_loss:.4g}/{training_sqr_loss:.4g}{validation_text}",
            flush=True,
        )

    final_training: dict[str, Any] | None = None
    final_validation: dict[str, Any] | None = None

    def main_value_and_save(*, current_phase: str = "main") -> None:
        nonlocal last_value, final_training, final_validation
        model.eval()
        model.calibrate_density(training, batch_size=train_config.val_batch_size)
        final_training = valuate_with_cap_derivative(
            model,
            training,
            batch_size=train_config.val_batch_size,
            split="training",
        )
        final_validation = (
            valuate_with_cap_derivative(
                model,
                validation,
                batch_size=train_config.val_batch_size,
                split="validation",
            )
            if validation is not None
            else None
        )
        if not tracker.valuation_epochs or tracker.valuation_epochs[-1] != main_epoch:
            tracker.valuation_epochs.append(main_epoch)
            tracker.training_values.append(final_training)
            if final_validation is not None:
                tracker.validation_values.append(final_validation)
            tracker.elapsed_seconds.append(time.monotonic() - started)
        else:
            tracker.training_values[-1] = final_training
            if final_validation is not None:
                tracker.validation_values[-1] = final_validation
        save(current_phase)
        last_value = time.monotonic()
        validation_text = (
            "" if final_validation is None else f", validation {final_validation['loss']:.6g}"
        )
        current_lr = tracker.lr[-1] if tracker.lr else main_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {main_epoch}/{epochs}: batch {tracker.batch_loss[-1]:.6g}, "
            f"full-training {final_training['loss']:.6g}{validation_text}, lr {current_lr:.4g}",
            flush=True,
        )

    with lock:
        try:
            if moment_parameters and phase == "moments" and moment_epoch < moment_epochs:
                assert moment_optimizer is not None and moment_scheduler is not None
                moment_step: Callable[[Mapping[str, Tensor]], tuple[Tensor, Tensor, Tensor]] = (
                    model.moment_loss
                )
                if train_config.compile and not model.debug:
                    moment_step = torch.compile(moment_step, dynamic=True)
                first_new_moment = True
                for epoch_index in range(moment_epoch, moment_epochs):
                    model.drift.train()
                    assert model.sqr is not None
                    model.sqr.train()
                    moment_optimizer.zero_grad(set_to_none=True)
                    batch = _sample_paths(training, train_config.trn_batch_size)
                    loss, drift_loss, sqr_loss = moment_step(batch)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"non-finite moment loss at epoch {epoch_index + 1}")
                    loss.backward()
                    _, nonfinite = _gradient_diagnostics(moment_parameters)
                    if nonfinite:
                        raise RuntimeError(
                            f"found {nonfinite} non-finite moment-gradient elements at epoch {epoch_index + 1}"
                        )
                    moment_optimizer.step()
                    current_lr = moment_optimizer.param_groups[0]["lr"]
                    moment_epoch = epoch_index + 1
                    if moment_epoch < moment_epochs:
                        moment_scheduler.step()
                    tracker.moment_epochs.append(moment_epoch)
                    tracker.moment_batch_loss.append(float(loss.detach().item()))
                    tracker.moment_drift_batch_loss.append(float(drift_loss.detach().item()))
                    tracker.moment_sqr_batch_loss.append(float(sqr_loss.detach().item()))
                    tracker.moment_lr.append(float(current_lr))
                    tracker.moment_elapsed_seconds.append(time.monotonic() - started)
                    now = time.monotonic()
                    should_value = (
                        first_new_moment
                        or moment_epoch % train_config.value_update == 0
                        or now - last_value >= train_config.value_seconds
                    )
                    if should_value:
                        moment_value_and_save()
                        first_new_moment = False
                        if train_config.early_stopping and _moment_training_ready(
                            tracker, train_config
                        ):
                            tracker.moment_converged_epoch = moment_epoch
                            save("moments")
                            break
                    elif (
                        moment_epoch % train_config.checkpoint_update == 0
                        or now - last_checkpoint >= train_config.checkpoint_seconds
                    ):
                        save("moments")
                if (
                    not tracker.moment_valuation_epochs
                    or tracker.moment_valuation_epochs[-1] != moment_epoch
                ):
                    moment_value_and_save()

            if moment_parameters:
                model.freeze_moments()
            if (
                train_config.action_initialization_epochs > 0
                and not tracker.action_initialization_completed
            ):
                if train_config.action_initialization_report is not None:
                    initialization = initialize_no_arb_action(
                        model,
                        train_config.action_initialization_report,
                        epochs=train_config.action_initialization_epochs,
                        batch_size=train_config.trn_batch_size,
                        lr=train_config.action_initialization_lr,
                        maximum_cap_fraction=(
                            train_config.action_initialization_max_cap_fraction
                        ),
                    )
                    initialization_losses = [
                        initialization["first_relative_mse"],
                        initialization["final_relative_mse"],
                    ]
                    tracker.action_initialization = initialization
                    print(
                        "No-arbitrage basket action initialization: relative MSE "
                        f"{initialization_losses[0]:.6g} -> "
                        f"{initialization_losses[-1]:.6g}; mean direction cosine "
                        f"{initialization['mean_direction_cosine']:.6g}; minimum "
                        "fitted tanh derivative "
                        f"{initialization['minimum_fitted_tanh_derivative']:.6g}",
                        flush=True,
                    )
                else:
                    if config.utility != "entropy":
                        raise ValueError(
                            "action mean/variance initialization is only supported "
                            "for entropy utility"
                        )
                    initialization_losses = initialize_entropy_action(
                        model,
                        training,
                        epochs=train_config.action_initialization_epochs,
                        batch_size=train_config.trn_batch_size,
                        lr=train_config.action_initialization_lr,
                    )
                tracker.action_initialization_completed = True
                tracker.action_initialization_first_loss = initialization_losses[0]
                tracker.action_initialization_final_loss = initialization_losses[-1]
                if train_config.action_initialization_report is None:
                    print(
                        "Entropy diagonal mean/variance action initialization: "
                        f"MSE {initialization_losses[0]:.6g} -> "
                        f"{initialization_losses[-1]:.6g} over "
                        f"{len(initialization_losses)} epochs",
                        flush=True,
                    )
            phase = "main"
            save("main")

            use_lbfgs = train_config.optimizer == "LBFGS"
            use_stable_entropy_lbfgs = use_lbfgs and config.utility == "entropy"
            main_step: Callable[[Mapping[str, Tensor]], Tensor] = model
            if (
                train_config.compile
                and not model.debug
                and config.utility != "cvar"
                and not use_stable_entropy_lbfgs
            ):
                main_step = torch.compile(main_step, dynamic=True)
            use_scipy = train_config.optimizer == "SciPy"
            if use_scipy and main_epoch < epochs:
                model.action.train()
                model.previous_density.eval()
                if config.deterministic_preprocess and (
                    tracker.deterministic_preprocessing is None
                ):
                    tracker.deterministic_preprocessing = (
                        scipy_deterministic_preprocess(
                            model,
                            training,
                            batch_size=train_config.val_batch_size,
                            maxiter=epochs,
                            gradient_tolerance=(
                                train_config.scipy_gradient_tolerance
                            ),
                            function_tolerance=(
                                train_config.scipy_function_tolerance
                            ),
                            max_line_search_steps=(
                                train_config.scipy_max_line_search_steps
                            ),
                        )
                    )
                    print(
                        "SciPy per-instrument preprocessing: "
                        f"loss {tracker.deterministic_preprocessing['loss']:.6g}, "
                        "gradient Linf "
                        f"{tracker.deterministic_preprocessing['gradient_linf']:.4g}",
                        flush=True,
                    )
                optimizer_summary, scipy_history = scipy_deterministic_optimize(
                    model,
                    training,
                    batch_size=train_config.val_batch_size,
                    maxiter=epochs - main_epoch,
                    gradient_tolerance=train_config.scipy_gradient_tolerance,
                    function_tolerance=train_config.scipy_function_tolerance,
                    max_line_search_steps=(
                        train_config.scipy_max_line_search_steps
                    ),
                    objective_step=main_step,
                )
                tracker.deterministic_optimizer = optimizer_summary
                history_start = len(tracker.epochs)
                for offset, item in enumerate(scipy_history, start=1):
                    tracker.epochs.append(history_start + offset)
                    tracker.batch_loss.append(float(item["loss"]))
                    tracker.batch_utility.append(-float(item["loss"]))
                    tracker.gradient_norm.append(float(item["gradient_l2"]))
                    tracker.nonfinite_gradient_count.append(0)
                    tracker.batch_elapsed_seconds.append(
                        time.monotonic() - started
                    )
                main_epoch = epochs
                if bool(optimizer_summary["gradient_tolerance_met"]):
                    tracker.main_converged_epoch = main_epoch
                if not tracker.batch_loss:
                    tracker.batch_loss.append(float(optimizer_summary["loss"]))
                    tracker.batch_utility.append(-float(optimizer_summary["loss"]))
                    tracker.gradient_norm.append(
                        float(optimizer_summary["gradient_l2"])
                    )
                    tracker.nonfinite_gradient_count.append(0)
                    tracker.batch_elapsed_seconds.append(
                        time.monotonic() - started
                    )
                print(
                    "SciPy deterministic optimization: "
                    f"{optimizer_summary['message']}; loss "
                    f"{optimizer_summary['loss']:.6g}, gradient Linf "
                    f"{optimizer_summary['gradient_linf']:.4g}",
                    flush=True,
                )
            first_new_main = True
            for epoch_index in range(main_epoch, epochs):
                model.action.train()
                model.previous_density.eval()
                if model.drift is not None:
                    model.drift.eval()
                    assert model.sqr is not None
                    model.sqr.eval()
                if use_lbfgs:
                    closure_output: dict[str, float | int] = {}

                    def closure() -> Tensor:
                        main_optimizer.zero_grad(set_to_none=True)
                        if use_stable_entropy_lbfgs:
                            full_loss = entropy_log_moment_loss(
                                model,
                                training,
                                batch_size=train_config.val_batch_size,
                            )
                            if not torch.isfinite(full_loss).item():
                                raise RuntimeError(
                                    "non-finite full-training entropy log-moment in "
                                    f"LBFGS closure at epoch {epoch_index + 1}"
                                )
                            full_loss.backward()
                        else:
                            full_loss = torch.zeros(
                                (), dtype=model.fdtype, device=model.device
                            )
                            paths = training["z"].shape[0]
                            for full_batch in _iter_path_batches(
                                training, train_config.val_batch_size
                            ):
                                batch_loss = main_step(full_batch)
                                if not torch.isfinite(batch_loss).item():
                                    raise RuntimeError(
                                        "non-finite full-training utility loss in "
                                        f"LBFGS closure at epoch {epoch_index + 1}"
                                    )
                                weight = full_batch["z"].shape[0] / float(paths)
                                (batch_loss * weight).backward()
                                full_loss += batch_loss.detach() * weight
                        gradient_norm, nonfinite = _gradient_diagnostics(
                            main_parameters
                        )
                        if nonfinite:
                            raise RuntimeError(
                                f"found {nonfinite} non-finite LBFGS-gradient "
                                f"elements at epoch {epoch_index + 1}"
                            )
                        closure_output.update(
                            loss=float(full_loss.item()),
                            gradient_norm=gradient_norm,
                            nonfinite=nonfinite,
                        )
                        return full_loss

                    main_optimizer.step(closure)
                    loss_value = float(closure_output["loss"])
                    gradient_norm = float(closure_output["gradient_norm"])
                    nonfinite = int(closure_output["nonfinite"])
                else:
                    main_optimizer.zero_grad(set_to_none=True)
                    batch = _sample_paths(training, train_config.trn_batch_size)
                    loss = main_step(batch)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(
                            "non-finite expected-utility loss at epoch "
                            f"{epoch_index + 1}"
                        )
                    loss.backward()
                    gradient_norm, nonfinite = _gradient_diagnostics(
                        main_parameters
                    )
                    if nonfinite:
                        raise RuntimeError(
                            f"found {nonfinite} non-finite main-gradient "
                            f"elements at epoch {epoch_index + 1}"
                        )
                    main_optimizer.step()
                    loss_value = float(loss.detach().item())
                current_lr = main_optimizer.param_groups[0]["lr"]
                main_epoch = epoch_index + 1
                if main_epoch < epochs:
                    main_scheduler.step()
                tracker.epochs.append(main_epoch)
                tracker.batch_loss.append(loss_value)
                tracker.batch_utility.append(-loss_value)
                tracker.lr.append(float(current_lr))
                tracker.gradient_norm.append(gradient_norm)
                tracker.nonfinite_gradient_count.append(nonfinite)
                tracker.batch_elapsed_seconds.append(time.monotonic() - started)
                now = time.monotonic()
                should_value = (
                    first_new_main
                    or main_epoch % train_config.value_update == 0
                    or now - last_value >= train_config.value_seconds
                )
                if should_value:
                    main_value_and_save()
                    first_new_main = False
                    if train_config.early_stopping and _main_training_ready(
                        tracker, train_config
                    ):
                        tracker.main_converged_epoch = main_epoch
                        save("main")
                        break
                elif (
                    main_epoch % train_config.checkpoint_update == 0
                    or now - last_checkpoint >= train_config.checkpoint_seconds
                ):
                    save("main")
        except KeyboardInterrupt:
            print(
                "Training interrupted; valuating the active phase and writing an atomic checkpoint.",
                flush=True,
            )
            if phase == "moments":
                moment_value_and_save()
            else:
                main_value_and_save()
            raise
        except FloatingPointError:
            print(
                "A non-finite CPU valuation was caught; writing the active checkpoint.",
                flush=True,
            )
            save(phase)
            raise

        if phase == "main" and (
            final_training is None
            or not tracker.valuation_epochs
            or tracker.valuation_epochs[-1] != main_epoch
        ):
            main_value_and_save()

    assert final_training is not None
    # Stationarity is measured only at the final fitted epoch.  Computing it
    # at intermediate valuation epochs would add two full-split backward
    # passes to every monitoring update on million-path experiments.
    final_training["objective_gradient"] = objective_gradient_diagnostics(
        model,
        training,
        batch_size=train_config.val_batch_size,
    )
    if final_validation is not None and validation is not None:
        final_validation["objective_gradient"] = objective_gradient_diagnostics(
            model,
            validation,
            batch_size=train_config.val_batch_size,
        )
    return TrainingResult(
        model=model,
        density=model.density(),
        training=final_training,
        validation=final_validation,
        tracker=tracker,
        checkpoint_path=checkpoint_path,
        restarted=restarted,
        universe_index=universe_index,
    )


def train(
    universe_or_experiment: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Any,
    *,
    config: DysanosConfig,
    epochs: int,
    train_config: TrainConfig = TrainConfig(),
    moment_epochs: int | None = None,
    previous_density: torch.nn.Module | Sequence[torch.nn.Module | None] | None = None,
    restart: bool = False,
    just_load: bool = False,
    env: BaseEnvironment | None = None,
    universe_index: int = 0,
    plot_sampler: bool = False,
) -> TrainingResult | list[TrainingResult]:
    """Train one model per independently simulated universe.

    Passing an :class:`RNDysanosExperiment` invokes the cached
    :func:`rndysanos.torch_prep.prep_experiment` data pipeline.  When ``env``
    is omitted, one GPU is acquired with :class:`DeviceLock` for the duration
    of the complete training call, with a CPU fallback when CUDA is absent.
    A caller supplying a CUDA environment is responsible for holding its own
    device lock.
    """

    verify_inp(epochs > 0, lambda: f"'epochs' must be positive, got {epochs}")
    if moment_epochs is None:
        moment_epochs = epochs
    verify_inp(moment_epochs > 0, lambda: f"'moment_epochs' must be positive, got {moment_epochs}")
    verify_inp(not (restart and just_load), "'restart' and 'just_load' cannot both be true")

    if isinstance(universe_or_experiment, Mapping):
        universes = [universe_or_experiment]
        return_single = True
    elif isinstance(universe_or_experiment, Sequence):
        universes = list(universe_or_experiment)
        return_single = False
    else:
        from .torch_prep import prep_experiment

        universes = prep_experiment(universe_or_experiment, plot_sampler=plot_sampler)
        return_single = False
    verify_inp(len(universes) > 0, "at least one universe is required")

    if isinstance(previous_density, Sequence) and not isinstance(previous_density, torch.nn.Module):
        previous_densities = list(previous_density)
        verify_inp(
            len(previous_densities) == len(universes),
            lambda: "number of previous densities must match number of universes",
        )
    else:
        previous_densities = [previous_density] * len(universes)

    def run(active_env: BaseEnvironment) -> list[TrainingResult]:
        results = []
        for offset, (universe, prior) in enumerate(zip(universes, previous_densities, strict=True)):
            results.append(
                _train_one(
                    universe,
                    config=config,
                    train_config=train_config,
                    epochs=epochs,
                    moment_epochs=moment_epochs,
                    previous_density=None if prior is None else copy.deepcopy(prior),
                    restart=restart,
                    just_load=just_load,
                    env=active_env,
                    universe_index=universe_index + offset,
                )
            )
        return results

    if env is None and torch.cuda.is_available():
        with DeviceLock() as device_lock:
            results = run(BaseEnvironment(device=device_lock.device, debug=False))
    elif env is None:
        warnings.warn("CUDA is unavailable; running the OCE measure change on CPU")
        results = run(BaseEnvironment(device="cpu", debug=False))
    else:
        results = run(env)
    return results[0] if return_single else results


__all__ = [
    "DeterministicTimeAction",
    "DysanosConfig",
    "DysanosModel",
    "RealizedDensity",
    "TrainConfig",
    "TrainingResult",
    "TrainingTracker",
    "UnitDensity",
    "UniverseGeometry",
    "cap_objective_derivative",
    "conditional_variance",
    "cost_components",
    "entropy_log_moment_loss",
    "initialize_no_arb_action",
    "instrument_returns_and_values",
    "lower_tail_weights",
    "oce_objective",
    "prepare_universe",
    "scipy_deterministic_optimize",
    "scipy_deterministic_preprocess",
    "split_training_validation",
    "train",
    "validate_universe",
    "valuate",
    "valuate_with_cap_derivative",
    "valuate_moments",
]
