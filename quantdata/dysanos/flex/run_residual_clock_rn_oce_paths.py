"""Generate the dy21 floating-option OCE population under the functional RN law.

The runner starts all paths from one authenticated historical Direct-200 state,
but thereafter evolves the uncompressed residual-clock DLV surface.  Twenty-five
daily books are issued and the simulation continues for another nineteen days
so that the final 20D options settle.  Adjacent rows are one antithetic pair.

The path arrays are memory mapped and the active path batch can be checkpointed
at configured transition boundaries.  The production default checkpoints each
completed batch; a resumed run may replay at most that small deterministic batch
and never changes the global path ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex import residual_clock_dlv, rn_oce_features
from quantdata.dysanos.flex.market_encoder import checkpoint_digest
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.presentation_hedging import load_presentation_model

SCHEMA = "dysanos-residual-clock-rn-oce-paths-v1"
DEFAULT_CHECKPOINT = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/"
    "front-term-eta/dte1-5-weekly-no-am-all-history-v1/"
    "joint-checkpoints/joint-direct-200.pt"
)
DEFAULT_HISTORY = DEFAULT_CHECKPOINT.with_name("joint-direct-200-targets.pt")

EXPIRY_DAYS = (5, 10, 15, 20)
NORMALIZED_MONEYNESS = (
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.1,
    0.0,
    0.1,
    0.3,
    0.5,
)
NUM_INSTRUMENTS = 1 + len(EXPIRY_DAYS) * len(NORMALIZED_MONEYNESS)
OUTPUT_FILENAMES = {
    "policy_states": "policy-states.npy",
    "option_prices": "policy-option-prices.npy",
    "relative_strikes": "policy-relative-strikes.npy",
    "returns": "returns.npy",
    "feature_valid": "policy-feature-valid.npy",
    "return_valid": "return-valid.npy",
}


@dataclass(frozen=True, kw_only=True)
class Config:
    """Logical simulation and operational batching controls."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    history: Path = DEFAULT_HISTORY
    output: Path
    device: str = "cuda:0"
    state_index: int = 445
    num_paths: int = 100_000
    issue_days: int = 25
    path_batch_size: int = 512
    loading_scale: float = 0.05
    maximum_loading_halvings: int = 8
    path_seed: int = 90_261_009
    checkpoint_interval_days: int = 44
    policy_state_storage_dtype: str = "float32"
    resume: bool = False
    plan_only: bool = False

    @property
    def total_transitions(self) -> int:
        """Return the transitions needed to settle the last issued option."""

        return self.issue_days + max(EXPIRY_DAYS) - 1


@dataclass(frozen=True)
class WorkSpec:
    """One contiguous interval of adjacent antithetic path pairs."""

    pair_offset: int
    num_pairs: int

    @property
    def path_first(self) -> int:
        return 2 * self.pair_offset

    @property
    def num_paths(self) -> int:
        return 2 * self.num_pairs

    @property
    def path_stop(self) -> int:
        return self.path_first + self.num_paths

    @property
    def path_slice(self) -> slice:
        return slice(self.path_first, self.path_stop)

    @property
    def label(self) -> str:
        return f"pairs-{self.pair_offset}-n-{self.num_pairs}"


@dataclass(frozen=True)
class OutputShapes:
    """Shapes of the fixed memory-mapped OCE population."""

    num_paths: int
    issue_days: int
    policy_state_size: int
    num_expiries: int = len(EXPIRY_DAYS)
    num_strikes: int = len(NORMALIZED_MONEYNESS)

    @property
    def policy_states(self) -> tuple[int, int, int]:
        return (self.num_paths, self.issue_days, self.policy_state_size)

    @property
    def option_prices(self) -> tuple[int, int, int, int]:
        return (
            self.num_paths,
            self.issue_days,
            self.num_expiries,
            self.num_strikes,
        )

    @property
    def relative_strikes(self) -> tuple[int, int, int, int]:
        return self.option_prices

    @property
    def returns(self) -> tuple[int, int, int]:
        return (self.num_paths, self.issue_days, NUM_INSTRUMENTS)

    @property
    def feature_valid(self) -> tuple[int, int]:
        return (self.num_paths, self.issue_days)

    @property
    def return_valid(self) -> tuple[int, int, int]:
        return self.returns


@dataclass
class OutputMaps:
    """Open NumPy arrays and their stable on-disk paths."""

    policy_states: np.memmap
    option_prices: np.memmap
    relative_strikes: np.memmap
    returns: np.memmap
    feature_valid: np.memmap
    return_valid: np.memmap
    paths: dict[str, Path]

    def flush(self) -> None:
        for value in (
            self.policy_states,
            self.option_prices,
            self.relative_strikes,
            self.returns,
            self.feature_valid,
            self.return_valid,
        ):
            value.flush()


def make_work_plan(num_paths: int, path_batch_size: int) -> tuple[WorkSpec, ...]:
    """Split an even population into contiguous, unsplit antithetic pairs."""

    if num_paths <= 0 or num_paths % 2:
        raise ValueError("num_paths must be positive and even")
    if path_batch_size <= 0 or path_batch_size % 2:
        raise ValueError("path_batch_size must be positive and even")
    total_pairs = num_paths // 2
    pairs_per_batch = path_batch_size // 2
    plan = tuple(
        WorkSpec(
            pair_offset=first,
            num_pairs=min(pairs_per_batch, total_pairs - first),
        )
        for first in range(0, total_pairs, pairs_per_batch)
    )
    if sum(item.num_paths for item in plan) != num_paths:
        raise AssertionError("work plan does not cover the requested paths")
    if any(
        current.path_stop != following.path_first
        for current, following in zip(plan, plan[1:], strict=False)
    ):
        raise AssertionError("work plan is not globally contiguous")
    return plan


def maturity_entry_day(
    *, transition_index: int, horizon_days: int, issue_days: int
) -> int | None:
    """Map a zero-based transition to the issue date settling on that step."""

    if transition_index < 0 or horizon_days <= 0 or issue_days <= 0:
        raise ValueError("transition, horizon, and issue-day controls are invalid")
    entry_day = transition_index + 1 - horizon_days
    return entry_day if 0 <= entry_day < issue_days else None


def _arguments() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=445)
    parser.add_argument("--num-paths", type=int, default=100_000)
    parser.add_argument("--issue-days", type=int, default=25)
    parser.add_argument("--path-batch-size", type=int, default=512)
    parser.add_argument("--loading-scale", type=float, default=0.05)
    parser.add_argument("--maximum-loading-halvings", type=int, default=8)
    parser.add_argument("--path-seed", type=int, default=90_261_009)
    parser.add_argument("--checkpoint-interval-days", type=int, default=44)
    parser.add_argument(
        "--policy-state-storage-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    values = parser.parse_args()
    return Config(
        checkpoint=values.checkpoint.resolve(),
        history=values.history.resolve(),
        output=values.output.resolve(),
        device=values.device,
        state_index=values.state_index,
        num_paths=values.num_paths,
        issue_days=values.issue_days,
        path_batch_size=values.path_batch_size,
        loading_scale=values.loading_scale,
        maximum_loading_halvings=values.maximum_loading_halvings,
        path_seed=values.path_seed,
        checkpoint_interval_days=values.checkpoint_interval_days,
        policy_state_storage_dtype=values.policy_state_storage_dtype,
        resume=values.resume,
        plan_only=values.plan_only,
    )


def _validate_config(config: Config) -> None:
    if not config.checkpoint.is_file() or not config.history.is_file():
        raise FileNotFoundError("model checkpoint and authenticated history must exist")
    if config.output in (Path("/"), Path("/tmp"), Path("/media/storage")):
        raise ValueError("output must be a named project-specific directory")
    if config.state_index < 0:
        raise ValueError("state_index must be nonnegative")
    make_work_plan(config.num_paths, config.path_batch_size)
    if config.issue_days <= 0 or config.checkpoint_interval_days <= 0:
        raise ValueError("issue_days and checkpoint interval must be positive")
    if not math.isfinite(config.loading_scale) or config.loading_scale < 0.0:
        raise ValueError("loading_scale must be finite and nonnegative")
    if config.maximum_loading_halvings < 0:
        raise ValueError("maximum_loading_halvings must be nonnegative")
    if config.policy_state_storage_dtype not in {"float32", "float64"}:
        raise ValueError("unsupported policy-state storage dtype")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class ProgressLog:
    """Append flushed machine-readable progress rows."""

    def __init__(self, output: Path, *, append: bool) -> None:
        self.path = output / "progress.jsonl"
        self.started = time.perf_counter()
        if self.path.exists() and not append:
            raise FileExistsError(self.path)

    def write(self, stage: str, **values: object) -> None:
        row = {
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - self.started,
            **values,
        }
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()


def _resource_status(output: Path) -> dict[str, object]:
    memory: dict[str, int] = {}
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            key, value, *_ = line.split()
            memory[key.removesuffix(":")] = int(value)
    memory_used = 1.0 - memory["MemAvailable"] / memory["MemTotal"]
    temporary = os.statvfs("/tmp")
    temporary_used = 1.0 - temporary.f_bavail / temporary.f_blocks
    removed: list[str] = []
    if temporary_used >= 0.70:
        now = time.time()
        for path in output.glob("*.tmp"):
            if path.is_file() and now - path.stat().st_mtime > 3_600:
                path.unlink()
                removed.append(path.name)
        temporary = os.statvfs("/tmp")
        temporary_used = 1.0 - temporary.f_bavail / temporary.f_blocks
    output_disk = os.statvfs(output)
    storage_free: float | None = None
    if Path("/media/storage").exists():
        storage = os.statvfs("/media/storage")
        storage_free = storage.f_bavail / storage.f_blocks
    return {
        "memory_used_fraction": memory_used,
        "memory_warning": memory_used >= 0.80,
        "memory_stop": memory_used >= 0.90,
        "temporary_used_fraction": temporary_used,
        "temporary_warning": temporary_used >= 0.80,
        "temporary_stop": temporary_used >= 0.90,
        "temporary_files_removed": removed,
        "output_free_fraction": output_disk.f_bavail / output_disk.f_blocks,
        "storage_free_fraction": storage_free,
        "storage_warning": storage_free is not None and storage_free < 0.20,
        "storage_stop": storage_free is not None and storage_free <= 0.10,
    }


def _stop_for_resources(resources: Mapping[str, object]) -> bool:
    return bool(
        resources["memory_stop"]
        or resources["temporary_stop"]
        or resources["storage_stop"]
    )


def _npy_shape_dtype(path: Path) -> tuple[tuple[int, ...], np.dtype[Any]]:
    value = np.load(path, mmap_mode="r")
    return value.shape, value.dtype


def open_output_maps(
    output: Path,
    *,
    shapes: OutputShapes,
    state_dtype: str,
    resume: bool,
) -> OutputMaps:
    """Create or reopen all path arrays, validating resume compatibility."""

    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / filename for name, filename in OUTPUT_FILENAMES.items()}
    definitions: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "policy_states": (shapes.policy_states, np.dtype(state_dtype)),
        "option_prices": (shapes.option_prices, np.dtype(np.float32)),
        "relative_strikes": (shapes.relative_strikes, np.dtype(np.float32)),
        "returns": (shapes.returns, np.dtype(np.float32)),
        "feature_valid": (shapes.feature_valid, np.dtype(np.uint8)),
        "return_valid": (shapes.return_valid, np.dtype(np.uint8)),
    }
    maps: dict[str, np.memmap] = {}
    for name, (shape, dtype) in definitions.items():
        path = paths[name]
        if resume:
            if not path.is_file():
                raise FileNotFoundError(f"resume output is missing {path}")
            found_shape, found_dtype = _npy_shape_dtype(path)
            if found_shape != shape or found_dtype != dtype:
                raise ValueError(
                    f"resume array {path} has {(found_shape, found_dtype)}, "
                    f"expected {(shape, dtype)}"
                )
            maps[name] = np.lib.format.open_memmap(path, mode="r+")
        else:
            if path.exists():
                raise FileExistsError(path)
            maps[name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
    if not resume:
        maps["returns"].fill(np.nan)
        maps["feature_valid"].fill(0)
        maps["return_valid"].fill(0)
        for name in ("returns", "feature_valid", "return_valid"):
            maps[name].flush()
    return OutputMaps(paths=paths, **maps)


def _required_output_bytes(shapes: OutputShapes, state_dtype: str) -> int:
    return int(
        np.prod(shapes.policy_states, dtype=np.int64) * np.dtype(state_dtype).itemsize
        + np.prod(shapes.option_prices, dtype=np.int64) * np.dtype(np.float32).itemsize
        + np.prod(shapes.relative_strikes, dtype=np.int64)
        * np.dtype(np.float32).itemsize
        + np.prod(shapes.returns, dtype=np.int64) * np.dtype(np.float32).itemsize
        + np.prod(shapes.feature_valid, dtype=np.int64) * np.dtype(np.uint8).itemsize
        + np.prod(shapes.return_valid, dtype=np.int64) * np.dtype(np.uint8).itemsize
    )


def _load_initial_surface(
    config: Config,
) -> tuple[Any, residual_clock_dlv.ResidualClockDlvSurface, dict[str, object]]:
    _, model = load_presentation_model(
        config.checkpoint,
        device=config.device,
        precision=64,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    history = torch.load(config.history, map_location="cpu", weights_only=False)
    if history.get("joint_checkpoint_sha256") != checkpoint_digest(config.checkpoint):
        raise ValueError("history does not authenticate the requested model checkpoint")
    states = history.get("states")
    dates = history.get("dates")
    spots = history.get("spot_refs")
    if not isinstance(states, torch.Tensor) or not isinstance(dates, list):
        raise ValueError("target history is missing states or dates")
    if not isinstance(spots, torch.Tensor):
        raise ValueError("target history is missing spot references")
    if not 0 <= config.state_index < len(states):
        raise ValueError("state_index is outside the authenticated history")
    device = next(model.parameters()).device
    state = states[config.state_index : config.state_index + 1].to(
        dtype=model.fdtype,
        device=device,
    )
    with torch.no_grad():
        raw = model.raw_states_from_states(state)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        decoded = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        exact_clock = torch.as_tensor(
            model.config.geometry.model_expiries_y,
            dtype=model.fdtype,
            device=device,
        )
        decoded_surface = residual_clock_dlv.residual_clock_surface_from_decoded(
            model.sanos_decoder,
            decoded,
            residual_expiries_y=exact_clock,
        )
        ingress = residual_clock_dlv.canonicalize_residual_clock_dlv_ingress(
            model.sanos_decoder,
            decoded_surface,
        )
        densification = _densify_front_twenty(
            model.sanos_decoder,
            ingress.surface,
        )
        surface = densification.surface
        residual_clock_dlv.validate_residual_clock_dlv_surface(
            model.sanos_decoder,
            surface,
        )
    metadata = {
        "checkpoint_sha256": checkpoint_digest(config.checkpoint),
        "history_sha256": checkpoint_digest(config.history),
        "state_index": config.state_index,
        "state_date": str(dates[config.state_index]),
        "spot_reference": float(spots[config.state_index]),
        "model_state_size": int(model.config.state_size),
        "native_state_size": int(model.config.geometry.native_state_size),
        "initial_surface_rows": int(len(surface.residual_expiries_y)),
        "surface_ingress": {
            "maximum_input_density_perturbation": float(
                ingress.maximum_input_density_perturbation
            ),
            "maximum_input_density0_perturbation": float(
                ingress.maximum_input_density0_perturbation
            ),
            "maximum_input_density_tower_perturbation": float(
                ingress.maximum_input_density_tower_perturbation
            ),
            "maximum_recursion_residual": float(
                ingress.maximum_recursion_residual
            ),
            "minimum_density": float(ingress.minimum_density),
            "minimum_dlvsq_bound_margin": float(
                ingress.minimum_dlvsq_bound_margin
            ),
        },
        "initial_front_twenty_densification": _densification_diagnostics(
            densification
        ),
    }
    return model, surface, metadata


def _densify_front_twenty(
    decoder: Any,
    surface: residual_clock_dlv.ResidualClockDlvSurface,
) -> residual_clock_dlv.ResidualClockDlvDensification:
    """Insert only the residual 1D--20D nodes needed at an issue date."""

    target = torch.arange(
        1,
        rn_oce_features.RN_OCE_DAILY_SLABS + 1,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    ) / DAYS_PER_YEAR
    distance = torch.abs(
        target[:, None] - surface.residual_expiries_y[None]
    )
    nearest_distance, nearest_position = torch.min(distance, dim=1)
    clock_tolerance_y = torch.as_tensor(
        5.0e-10 / DAYS_PER_YEAR,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    requested = torch.where(
        nearest_distance <= clock_tolerance_y,
        surface.residual_expiries_y[nearest_position],
        target,
    )
    return residual_clock_dlv.densify_residual_clock_dlv_surface(
        decoder,
        surface,
        requested_residual_expiries_y=requested,
    )


def _densification_diagnostics(
    value: residual_clock_dlv.ResidualClockDlvDensification,
) -> dict[str, float]:
    return {
        "old_density_change": float(value.maximum_old_density_change),
        "old_dlvsq_change": float(value.maximum_old_dlvsq_change),
        "old_backbone_variance_change": float(
            value.maximum_old_backbone_variance_change
        ),
        "old_option_change": float(value.maximum_old_option_change),
        "inserted_variance_flux_residual": float(
            value.maximum_inserted_variance_flux_residual
        ),
        "split_decoder_residual": float(value.maximum_split_decoder_residual),
    }


def _extract_features(
    model: Any,
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    current_spot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    if current_spot.shape != (len(surface.densities),) or not bool(
        torch.all(torch.isfinite(current_spot) & (current_spot > 0.0))
    ):
        raise ValueError("current_spot must be finite, positive and path aligned")
    normalized_moneyness = torch.as_tensor(
        NORMALIZED_MONEYNESS,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    value = rn_oce_features.make_rn_oce_floating_book_snapshot(
        surface,
        minimum_dlvsq=model.config.geometry.min_dlvsq,
        normalized_moneyness=normalized_moneyness,
    )
    if tuple(value.expiry_days) != EXPIRY_DAYS:
        raise ValueError("RN OCE helper expiry ordering differs from this run")
    expected_is_put = _is_put_vector(surface.densities.device).reshape(
        len(EXPIRY_DAYS),
        len(NORMALIZED_MONEYNESS),
    )
    if not torch.equal(value.is_put, expected_is_put):
        raise ValueError("RN OCE helper option ordering differs from this run")
    state = value.policy_state
    prices = value.normalized_option_entry_prices
    strikes = value.relative_strikes
    batch = len(current_spot)
    expected = (batch, len(EXPIRY_DAYS), len(NORMALIZED_MONEYNESS))
    if state.ndim != 2 or len(state) != batch:
        raise ValueError("functional policy state must have shape (paths, features)")
    if prices.shape != expected or strikes.shape != expected:
        raise ValueError(
            "OCE price and strike tensors must have shape "
            f"{expected}, found {tuple(prices.shape)} and {tuple(strikes.shape)}"
        )
    if not bool(
        torch.all(torch.isfinite(state))
        and torch.all(torch.isfinite(prices) & (prices > 0.0))
        and torch.all(torch.isfinite(strikes) & (strikes > 0.0))
    ):
        raise FloatingPointError("OCE feature extraction produced invalid values")
    metadata = value.policy_state_metadata.as_dict()
    return state, prices, strikes, metadata


def _expand_surface(
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    copies: int,
) -> residual_clock_dlv.ResidualClockDlvSurface:
    return residual_clock_dlv.expand_residual_clock_dlv_surface(surface, copies)


def _surface_payload(
    surface: residual_clock_dlv.ResidualClockDlvSurface,
) -> dict[str, object]:
    return {
        "residual_expiries_y": surface.residual_expiries_y.detach().cpu(),
        "support_strikes": surface.support_strikes.detach().cpu(),
        "density0": surface.density0.detach().cpu(),
        "densities": surface.densities.detach().cpu(),
        "dlvsqs": surface.dlvsqs.detach().cpu(),
        "backbone_total_vols": surface.backbone_total_vols.detach().cpu(),
        "temporary_frozen_front": surface.temporary_frozen_front,
    }


def _restore_surface(
    payload: Mapping[str, object], device: torch.device
) -> residual_clock_dlv.ResidualClockDlvSurface:
    tensor_names = (
        "residual_expiries_y",
        "support_strikes",
        "density0",
        "densities",
        "dlvsqs",
        "backbone_total_vols",
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        value = payload[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"surface checkpoint field {name} is not a tensor")
        tensors[name] = value.to(device=device)
    return residual_clock_dlv.ResidualClockDlvSurface(
        **tensors,
        temporary_frozen_front=bool(payload["temporary_frozen_front"]),
    )


def _slice_surface(
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    index: slice,
) -> residual_clock_dlv.ResidualClockDlvSurface:
    return residual_clock_dlv.ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes[index],
        density0=surface.density0[index],
        densities=surface.densities[index],
        dlvsqs=surface.dlvsqs[index],
        backbone_total_vols=surface.backbone_total_vols[index],
        temporary_frozen_front=surface.temporary_frozen_front,
    )


def _merge_characteristics(
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    values: list[residual_clock_dlv.FunctionalDlvCharacteristics],
) -> residual_clock_dlv.FunctionalDlvCharacteristics:
    reference = values[0]
    concatenate = (
        "raw_log_loadings",
        "psi",
        "relative_psi",
        "beta",
        "alpha",
        "mu_z",
    )
    maximum = (
        "maximum_volatility_equation_residual",
        "maximum_drift_equation_residual",
        "maximum_density_diffusion_mass_residual",
        "maximum_density_diffusion_forward_residual",
        "maximum_absolute_relative_density_diffusion",
    )
    return residual_clock_dlv.FunctionalDlvCharacteristics(
        surface=surface,
        expiry_time_increments_y=reference.expiry_time_increments_y,
        **{
            name: torch.cat([getattr(value, name) for value in values], dim=0)
            for name in concatenate
        },
        **{
            name: torch.stack([getattr(value, name) for value in values]).max()
            for name in maximum
        },
        minimum_density=torch.stack([value.minimum_density for value in values]).min(),
    )


def _merge_kernels(
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    values: list[residual_clock_dlv.FunctionalDlvKernel],
) -> residual_clock_dlv.FunctionalDlvKernel:
    reference = values[0]
    if any(
        not torch.equal(value.branch_probabilities, reference.branch_probabilities)
        or not torch.equal(
            value.standardized_innovations,
            reference.standardized_innovations,
        )
        for value in values[1:]
    ):
        raise ValueError("functional cubature changed across path-local retries")
    concatenate = (
        "densities_by_branch",
        "dlvsqs_by_branch",
        "effective_mu_z",
        "effective_log_excess_dlvsq_diffusion",
        "centered_log_increment_variance",
    )
    maximum = (
        "maximum_density_tower_residual",
        "maximum_variance_flux_tower_residual",
        "maximum_variance_flux_recursion_residual",
        "maximum_decoder_residual",
        "maximum_mass_residual",
        "maximum_mean_residual",
        "maximum_effective_minus_local_mu_z",
        "maximum_effective_diffusion_minus_loading",
        "maximum_dlvsq",
    )
    return residual_clock_dlv.FunctionalDlvKernel(
        initial_surface=surface,
        characteristics=_merge_characteristics(
            surface,
            [value.characteristics for value in values],
        ),
        calendar_time_increment_y=reference.calendar_time_increment_y,
        branch_probabilities=reference.branch_probabilities,
        standardized_innovations=reference.standardized_innovations,
        **{
            name: torch.cat([getattr(value, name) for value in values], dim=0)
            for name in concatenate
        },
        **{
            name: torch.stack([getattr(value, name) for value in values]).max()
            for name in maximum
        },
        fallback_coordinate_fraction=torch.stack(
            [value.fallback_coordinate_fraction for value in values]
        ).mean(),
        minimum_density=torch.stack([value.minimum_density for value in values]).min(),
        minimum_dlvsq=torch.stack([value.minimum_dlvsq for value in values]).min(),
    )


def _functional_kernel(
    decoder: Any,
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    *,
    loading_scale: float,
    maximum_halvings: int,
) -> tuple[residual_clock_dlv.FunctionalDlvKernel, torch.Tensor, list[str]]:
    """Build the requested law, reducing loading only on inadmissible paths."""

    step_y = 1.0 / DAYS_PER_YEAR
    failures: list[str] = []

    def build(
        value: residual_clock_dlv.ResidualClockDlvSurface, scale: float
    ) -> residual_clock_dlv.FunctionalDlvKernel:
        loadings = residual_clock_dlv.smooth_residual_clock_log_loadings(
            value,
            annualized_log_volatility=scale,
        )
        return residual_clock_dlv.build_functional_dlv_cubature(
            decoder,
            value,
            raw_log_loadings=loadings,
            calendar_time_increment_y=step_y,
        )

    try:
        value = build(surface, loading_scale)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, residual_clock_dlv.FunctionalDlvInadmissibleError) as error:
        failures.append(f"batched {loading_scale:.8g}: {error}")
    else:
        scales = torch.full(
            (len(surface.support_strikes),),
            loading_scale,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
        return value, scales, failures

    selected: list[residual_clock_dlv.FunctionalDlvKernel] = []
    scales: list[float] = []
    for path in range(len(surface.support_strikes)):
        path_surface = _slice_surface(surface, slice(path, path + 1))
        path_errors: list[str] = []
        candidates = [
            loading_scale / (2**halving)
            for halving in range(maximum_halvings + 1)
        ]
        if not candidates or candidates[-1] != 0.0:
            candidates.append(0.0)
        for scale in candidates:
            try:
                selected.append(build(path_surface, scale))
            except torch.cuda.OutOfMemoryError:
                raise
            except (
                RuntimeError,
                residual_clock_dlv.FunctionalDlvInadmissibleError,
            ) as error:
                path_errors.append(f"{scale:.8g}: {error}")
                continue
            scales.append(scale)
            break
        else:
            raise RuntimeError(
                f"functional RN kernel is inadmissible for path {path}: "
                + " | ".join(path_errors)
            )
        failures.extend(f"path {path} {message}" for message in path_errors)
    return (
        _merge_kernels(surface, selected),
        torch.as_tensor(
            scales,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        ),
        failures,
    )


def _randoms(
    spec: WorkSpec,
    *,
    total_paths: int,
    transition_index: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce batch-invariant adjacent antithetic innovations."""

    total_pairs = total_paths // 2
    generator = torch.Generator(device=device).manual_seed(
        seed + (transition_index + 1) * 10_007
    )
    branch_all = torch.rand(
        total_pairs,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    node_all = torch.rand(
        total_pairs,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    normal_all = torch.randn(
        total_pairs,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    interval = slice(spec.pair_offset, spec.pair_offset + spec.num_pairs)
    maximum = torch.nextafter(
        torch.ones((), dtype=torch.float64, device=device),
        torch.zeros((), dtype=torch.float64, device=device),
    )

    def pair_uniform(value: torch.Tensor) -> torch.Tensor:
        selected = value[interval]
        return torch.stack(
            (selected, torch.clamp(1.0 - selected, max=maximum)),
            dim=-1,
        ).reshape(-1)

    normal = normal_all[interval]
    return (
        pair_uniform(branch_all),
        pair_uniform(node_all),
        torch.stack((normal, -normal), dim=-1).reshape(-1),
    )


def _is_put_vector(device: torch.device) -> torch.Tensor:
    by_strike = torch.as_tensor(
        [score > 0.0 for score in NORMALIZED_MONEYNESS],
        dtype=torch.bool,
        device=device,
    )
    return by_strike.repeat(len(EXPIRY_DAYS))


def _option_returns(
    *,
    gross_horizon: torch.Tensor,
    issue_prices: torch.Tensor,
    relative_strikes: torch.Tensor,
    is_put: torch.Tensor,
) -> torch.Tensor:
    """Return unit-dollar terminal option returns in expiry-major ordering."""

    if issue_prices.shape != relative_strikes.shape:
        raise ValueError("issue prices and relative strikes must align")
    if issue_prices.ndim != 2 or is_put.shape != issue_prices.shape[1:]:
        raise ValueError("terminal option inputs have incompatible shapes")
    intrinsic = torch.where(
        is_put[None],
        torch.clamp_min(relative_strikes - gross_horizon[:, None], 0.0),
        torch.clamp_min(gross_horizon[:, None] - relative_strikes, 0.0),
    )
    result = intrinsic / issue_prices - 1.0
    if not bool(torch.all(torch.isfinite(result))):
        raise FloatingPointError("terminal option return is non-finite")
    return result


def _empty_extrema() -> dict[str, float]:
    return {
        "densification_old_density_change": 0.0,
        "densification_old_dlvsq_change": 0.0,
        "densification_old_backbone_variance_change": 0.0,
        "densification_old_option_change": 0.0,
        "densification_variance_flux_residual": 0.0,
        "densification_split_decoder_residual": 0.0,
        "density_tower_residual": 0.0,
        "variance_flux_tower_residual": 0.0,
        "variance_flux_recursion_residual": 0.0,
        "branch_decoder_residual": 0.0,
        "front_split_decoder_residual": 0.0,
        "front_split_old_density_change": 0.0,
        "front_split_old_option_change": 0.0,
        "conditional_mass_residual": 0.0,
        "conditional_mean_residual": 0.0,
        "minimum_dlvsq": float("inf"),
        "maximum_dlvsq": 0.0,
        "fallback_coordinate_fraction": 0.0,
    }


def _merge_extrema(target: dict[str, float], values: Mapping[str, float]) -> None:
    for name, value in values.items():
        if name.startswith("minimum_"):
            target[name] = min(target[name], value)
        else:
            target[name] = max(target[name], value)


def _serialize_config(config: Config) -> dict[str, object]:
    value = asdict(config)
    value.pop("resume")
    value.pop("plan_only")
    value["checkpoint"] = str(config.checkpoint)
    value["history"] = str(config.history)
    value["output"] = str(config.output)
    value["total_transitions"] = config.total_transitions
    value["expiry_days"] = list(EXPIRY_DAYS)
    value["normalized_moneyness"] = list(NORMALIZED_MONEYNESS)
    return value


def _identity(config: Config, plan: tuple[WorkSpec, ...]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "config": _serialize_config(config),
        "work_plan": [asdict(item) for item in plan],
        "source_hashes": {
            "checkpoint": _sha256(config.checkpoint),
            "history": _sha256(config.history),
            "residual_clock_dlv": _sha256(Path(residual_clock_dlv.__file__).resolve()),
            "rn_oce_features": _sha256(
                Path(rn_oce_features.__file__).resolve()
            ),
            "runner": _sha256(Path(__file__).resolve()),
        },
    }


def _active_payload(
    *,
    spec: WorkSpec,
    transition_completed: int,
    surface: residual_clock_dlv.ResidualClockDlvSurface,
    current_spot: torch.Tensor,
    entry_spots: list[torch.Tensor],
    entry_prices: list[torch.Tensor],
    entry_strikes: list[torch.Tensor],
    extrema: Mapping[str, float],
) -> dict[str, object]:
    return {
        "label": spec.label,
        "transition_completed": transition_completed,
        "surface": _surface_payload(surface),
        "current_spot": current_spot.detach().cpu(),
        "entry_spots": [value.detach().cpu() for value in entry_spots],
        "entry_prices": [value.detach().cpu() for value in entry_prices],
        "entry_strikes": [value.detach().cpu() for value in entry_strikes],
        "extrema": dict(extrema),
    }


def _checkpoint_payload(
    *,
    identity: Mapping[str, object],
    work_index: int,
    completed: list[dict[str, object]],
    extrema: Mapping[str, float],
    active: Mapping[str, object] | None,
    status: str,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "identity": dict(identity),
        "work_index": work_index,
        "completed": completed,
        "extrema": dict(extrema),
        "active": active,
        "status": status,
    }


def _write_issue_features(
    outputs: OutputMaps,
    *,
    spec: WorkSpec,
    day: int,
    policy_state: torch.Tensor,
    prices: torch.Tensor,
    strikes: torch.Tensor,
) -> None:
    index = spec.path_slice
    outputs.policy_states[index, day] = policy_state.detach().cpu().numpy()
    outputs.option_prices[index, day] = prices.detach().cpu().numpy()
    outputs.relative_strikes[index, day] = strikes.detach().cpu().numpy()
    outputs.feature_valid[index, day] = 1


def _write_spot_return(
    outputs: OutputMaps,
    *,
    spec: WorkSpec,
    day: int,
    value: torch.Tensor,
) -> None:
    outputs.returns[spec.path_slice, day, 0] = value.detach().cpu().numpy()
    outputs.return_valid[spec.path_slice, day, 0] = 1


def _write_option_return(
    outputs: OutputMaps,
    *,
    spec: WorkSpec,
    entry_day: int,
    expiry_position: int,
    value: torch.Tensor,
) -> None:
    width = len(NORMALIZED_MONEYNESS)
    first = 1 + expiry_position * width
    last = first + width
    outputs.returns[spec.path_slice, entry_day, first:last] = (
        value.detach().cpu().numpy()
    )
    outputs.return_valid[spec.path_slice, entry_day, first:last] = 1


def _validate_complete_outputs(outputs: OutputMaps, shapes: OutputShapes) -> None:
    path_batch = 1_024
    for first in range(0, shapes.num_paths, path_batch):
        last = min(first + path_batch, shapes.num_paths)
        if not np.all(outputs.feature_valid[first:last] == 1):
            raise RuntimeError("completed run has missing policy features")
        if not np.all(outputs.return_valid[first:last] == 1):
            raise RuntimeError("completed run has missing returns")
        for value, name in (
            (outputs.policy_states[first:last], "policy states"),
            (outputs.option_prices[first:last], "option prices"),
            (outputs.relative_strikes[first:last], "relative strikes"),
            (outputs.returns[first:last], "returns"),
        ):
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"completed run has non-finite {name}")


def run(config: Config) -> None:
    """Generate the checkpointed residual-clock RN OCE population."""

    _validate_config(config)
    plan = make_work_plan(config.num_paths, config.path_batch_size)
    config.output.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema": SCHEMA + "-plan",
        "config": _serialize_config(config),
        "num_work_batches": len(plan),
        "work_plan": [
            {
                **asdict(item),
                "path_first": item.path_first,
                "path_stop": item.path_stop,
                "num_paths": item.num_paths,
                "label": item.label,
            }
            for item in plan
        ],
    }
    if config.plan_only:
        _atomic_json(plan_payload, config.output / "plan.json")
        print(json.dumps(plan_payload, indent=2), flush=True)
        return

    checkpoint_path = config.output / "checkpoint.pt"
    report_path = config.output / "report.json"
    metadata_path = config.output / "population.json"
    lock_path = config.output / ".run.lock"
    with FileLock(str(lock_path), timeout=0):
        progress = ProgressLog(config.output, append=config.resume)
        resources = _resource_status(config.output)
        progress.write("resources", **resources)
        if _stop_for_resources(resources):
            raise RuntimeError("host resource stop threshold reached before model load")

        model, source_surface, provenance = _load_initial_surface(config)
        device = next(model.parameters()).device
        preview_spot = torch.ones(1, dtype=model.fdtype, device=device)
        preview_state, preview_prices, preview_strikes, feature_metadata = (
            _extract_features(model, source_surface, preview_spot)
        )
        del preview_prices, preview_strikes
        shapes = OutputShapes(
            num_paths=config.num_paths,
            issue_days=config.issue_days,
            policy_state_size=preview_state.shape[1],
        )
        required_bytes = _required_output_bytes(
            shapes,
            config.policy_state_storage_dtype,
        )
        disk = os.statvfs(config.output)
        available_bytes = disk.f_bavail * disk.f_frsize
        if not config.resume and required_bytes > 0.90 * available_bytes:
            raise RuntimeError(
                f"RN OCE arrays require {required_bytes / 2**30:.2f} GiB but only "
                f"{available_bytes / 2**30:.2f} GiB are available"
            )
        outputs = open_output_maps(
            config.output,
            shapes=shapes,
            state_dtype=config.policy_state_storage_dtype,
            resume=config.resume,
        )
        identity = _identity(config, plan)
        population_metadata = {
            "schema": SCHEMA + "-population",
            "identity": identity,
            "provenance": provenance,
            "feature_metadata": feature_metadata,
            "shapes": asdict(shapes),
            "array_shapes": {
                name: list(getattr(shapes, name))
                for name in OUTPUT_FILENAMES
            },
            "array_dtypes": {
                "policy_states": config.policy_state_storage_dtype,
                "option_prices": "float32",
                "relative_strikes": "float32",
                "returns": "float32",
                "feature_valid": "uint8",
                "return_valid": "uint8",
            },
            "files": {name: str(path) for name, path in outputs.paths.items()},
            "estimated_bytes": required_bytes,
            "path_order": (
                "adjacent rows (2k,2k+1) are one antithetic pair; batches are "
                "contiguous global pair intervals"
            ),
            "return_order": [
                "spot",
                *[
                    f"{expiry}D/NMS-{score:g}/{'put' if score > 0.0 else 'call'}"
                    for expiry in EXPIRY_DAYS
                    for score in NORMALIZED_MONEYNESS
                ],
            ],
        }
        if config.resume:
            saved_metadata = json.loads(metadata_path.read_text())
            if saved_metadata != population_metadata:
                raise ValueError("resume population metadata does not match this run")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved.get("schema") != SCHEMA or saved.get("identity") != identity:
                raise ValueError("resume checkpoint identity does not match this run")
            work_index = int(saved["work_index"])
            completed = list(saved["completed"])
            extrema = dict(saved["extrema"])
            active = saved.get("active")
        else:
            existing = [path for path in (checkpoint_path, report_path) if path.exists()]
            if existing:
                raise FileExistsError(f"output already contains run files: {existing}")
            _atomic_json(population_metadata, metadata_path)
            work_index = 0
            completed: list[dict[str, object]] = []
            extrema = _empty_extrema()
            active = None

        total_path_transitions = config.num_paths * config.total_transitions
        completed_path_transitions = sum(
            int(value["num_paths"]) * config.total_transitions for value in completed
        )
        session_base = completed_path_transitions
        if active is not None and work_index < len(plan):
            session_base += plan[work_index].num_paths * int(
                active["transition_completed"]
            )
        started = time.perf_counter()
        is_put = _is_put_vector(device)
        step_y = 1.0 / DAYS_PER_YEAR

        while work_index < len(plan):
            spec = plan[work_index]
            if active is None:
                surface = _expand_surface(source_surface, spec.num_paths)
                current_spot = torch.ones(
                    spec.num_paths,
                    dtype=model.fdtype,
                    device=device,
                )
                transition_completed = 0
                entry_spots: list[torch.Tensor] = []
                entry_prices: list[torch.Tensor] = []
                entry_strikes: list[torch.Tensor] = []
                work_extrema = _empty_extrema()
            else:
                if active.get("label") != spec.label:
                    raise ValueError("active checkpoint work label is incompatible")
                surface = _restore_surface(active["surface"], device)
                current_spot = active["current_spot"].to(device=device)
                transition_completed = int(active["transition_completed"])
                entry_spots = [value.to(device=device) for value in active["entry_spots"]]
                entry_prices = [value.to(device=device) for value in active["entry_prices"]]
                entry_strikes = [value.to(device=device) for value in active["entry_strikes"]]
                work_extrema = dict(active["extrema"])

            for transition_index in range(
                transition_completed,
                config.total_transitions,
            ):
                densification_values = {
                    "densification_old_density_change": 0.0,
                    "densification_old_dlvsq_change": 0.0,
                    "densification_old_backbone_variance_change": 0.0,
                    "densification_old_option_change": 0.0,
                    "densification_variance_flux_residual": 0.0,
                    "densification_split_decoder_residual": 0.0,
                }
                if transition_index < config.issue_days:
                    densification = _densify_front_twenty(
                        model.sanos_decoder,
                        surface,
                    )
                    surface = densification.surface
                    diagnostics = _densification_diagnostics(densification)
                    densification_values = {
                        "densification_old_density_change": diagnostics[
                            "old_density_change"
                        ],
                        "densification_old_dlvsq_change": diagnostics[
                            "old_dlvsq_change"
                        ],
                        "densification_old_backbone_variance_change": (
                            diagnostics["old_backbone_variance_change"]
                        ),
                        "densification_old_option_change": diagnostics[
                            "old_option_change"
                        ],
                        "densification_variance_flux_residual": diagnostics[
                            "inserted_variance_flux_residual"
                        ],
                        "densification_split_decoder_residual": diagnostics[
                            "split_decoder_residual"
                        ],
                    }
                insertion = residual_clock_dlv.insert_residual_front_by_density_interpolation(
                    model.sanos_decoder,
                    surface,
                    elapsed_y=step_y,
                )
                surface = insertion.surface
                if transition_index < config.issue_days:
                    policy_state, prices, strikes, _ = _extract_features(
                        model,
                        surface,
                        current_spot,
                    )
                    _write_issue_features(
                        outputs,
                        spec=spec,
                        day=transition_index,
                        policy_state=policy_state,
                        prices=prices,
                        strikes=strikes,
                    )
                    entry_spots.append(current_spot.detach().clone())
                    entry_prices.append(prices.detach().flatten(start_dim=1).clone())
                    entry_strikes.append(strikes.detach().flatten(start_dim=1).clone())

                kernel, accepted_scales, failures = _functional_kernel(
                    model.sanos_decoder,
                    surface,
                    loading_scale=config.loading_scale,
                    maximum_halvings=config.maximum_loading_halvings,
                )
                branch_u, node_u, normal = _randoms(
                    spec,
                    total_paths=config.num_paths,
                    transition_index=transition_index,
                    seed=config.path_seed,
                    device=device,
                )
                rolled = residual_clock_dlv.roll_residual_clock_dlv(
                    model.sanos_decoder,
                    kernel,
                    surface_branch_uniforms=branch_u,
                    front_node_uniforms=node_u,
                    front_backbone_normals=normal,
                    elapsed_y=step_y,
                )
                next_spot = current_spot * rolled.gross_spot_return
                if transition_index < config.issue_days:
                    _write_spot_return(
                        outputs,
                        spec=spec,
                        day=transition_index,
                        value=rolled.gross_spot_return - 1.0,
                    )
                for expiry_position, horizon in enumerate(EXPIRY_DAYS):
                    entry_day = maturity_entry_day(
                        transition_index=transition_index,
                        horizon_days=horizon,
                        issue_days=config.issue_days,
                    )
                    if entry_day is None:
                        continue
                    first = expiry_position * len(NORMALIZED_MONEYNESS)
                    last = first + len(NORMALIZED_MONEYNESS)
                    gross_horizon = next_spot / entry_spots[entry_day]
                    option_return = _option_returns(
                        gross_horizon=gross_horizon,
                        issue_prices=entry_prices[entry_day][:, first:last],
                        relative_strikes=entry_strikes[entry_day][:, first:last],
                        is_put=is_put[first:last],
                    )
                    _write_option_return(
                        outputs,
                        spec=spec,
                        entry_day=entry_day,
                        expiry_position=expiry_position,
                        value=option_return,
                    )

                current_spot = next_spot
                surface = rolled.surface
                day_values = {
                    **densification_values,
                    "density_tower_residual": float(
                        kernel.maximum_density_tower_residual
                    ),
                    "variance_flux_tower_residual": float(
                        kernel.maximum_variance_flux_tower_residual
                    ),
                    "variance_flux_recursion_residual": float(
                        kernel.maximum_variance_flux_recursion_residual
                    ),
                    "branch_decoder_residual": float(kernel.maximum_decoder_residual),
                    "front_split_decoder_residual": float(
                        insertion.maximum_split_decoder_residual
                    ),
                    "front_split_old_density_change": float(
                        insertion.maximum_old_density_change
                    ),
                    "front_split_old_option_change": float(
                        insertion.maximum_old_option_change
                    ),
                    "conditional_mass_residual": float(
                        rolled.maximum_conditional_mass_residual
                    ),
                    "conditional_mean_residual": float(
                        rolled.maximum_conditional_mean_residual
                    ),
                    "minimum_dlvsq": float(kernel.minimum_dlvsq),
                    "maximum_dlvsq": float(kernel.maximum_dlvsq),
                    "fallback_coordinate_fraction": float(
                        kernel.fallback_coordinate_fraction
                    ),
                }
                _merge_extrema(work_extrema, day_values)
                _merge_extrema(extrema, day_values)
                transition_completed = transition_index + 1
                checkpoint_due = (
                    transition_completed % config.checkpoint_interval_days == 0
                    and transition_completed < config.total_transitions
                )
                if checkpoint_due:
                    active = _active_payload(
                        spec=spec,
                        transition_completed=transition_completed,
                        surface=surface,
                        current_spot=current_spot,
                        entry_spots=entry_spots,
                        entry_prices=entry_prices,
                        entry_strikes=entry_strikes,
                        extrema=work_extrema,
                    )
                    outputs.flush()
                    _atomic_torch(
                        _checkpoint_payload(
                            identity=identity,
                            work_index=work_index,
                            completed=completed,
                            extrema=extrema,
                            active=active,
                            status="running",
                        ),
                        checkpoint_path,
                    )
                logical_completed = (
                    completed_path_transitions
                    + spec.num_paths * transition_completed
                )
                elapsed = time.perf_counter() - started
                session_completed = logical_completed - session_base
                eta = (
                    elapsed
                    * (total_path_transitions - logical_completed)
                    / session_completed
                    if session_completed > 0
                    else None
                )
                resources = _resource_status(config.output)
                progress.write(
                    "transition",
                    work_index=work_index,
                    total_work_batches=len(plan),
                    work_label=spec.label,
                    completed_transitions=transition_completed,
                    total_transitions=config.total_transitions,
                    completed_path_transitions=logical_completed,
                    total_path_transitions=total_path_transitions,
                    estimated_remaining_seconds=eta,
                    accepted_loading_scale_minimum=float(torch.min(accepted_scales)),
                    accepted_loading_scale_maximum=float(torch.max(accepted_scales)),
                    num_paths_with_loading_halving=int(
                        torch.sum(accepted_scales < config.loading_scale)
                    ),
                    loading_failure_count=len(failures),
                    minimum_spot=float(torch.min(current_spot)),
                    maximum_spot=float(torch.max(current_spot)),
                    **day_values,
                    **resources,
                )
                if _stop_for_resources(resources):
                    if not checkpoint_due:
                        active = _active_payload(
                            spec=spec,
                            transition_completed=transition_completed,
                            surface=surface,
                            current_spot=current_spot,
                            entry_spots=entry_spots,
                            entry_prices=entry_prices,
                            entry_strikes=entry_strikes,
                            extrema=work_extrema,
                        )
                    outputs.flush()
                    _atomic_torch(
                        _checkpoint_payload(
                            identity=identity,
                            work_index=work_index,
                            completed=completed,
                            extrema=extrema,
                            active=active,
                            status="stopped_resource",
                        ),
                        checkpoint_path,
                    )
                    progress.write("resource_stop", **resources)
                    raise RuntimeError(
                        "host resource stop threshold reached after checkpoint"
                    )

            if len(entry_spots) != config.issue_days:
                raise AssertionError("completed batch has the wrong number of issue dates")
            outputs.flush()
            completed.append(
                {
                    "label": spec.label,
                    "path_first": spec.path_first,
                    "path_stop": spec.path_stop,
                    "num_paths": spec.num_paths,
                    "minimum_terminal_spot": float(torch.min(current_spot)),
                    "maximum_terminal_spot": float(torch.max(current_spot)),
                    "extrema": work_extrema,
                }
            )
            completed_path_transitions += spec.num_paths * config.total_transitions
            work_index += 1
            active = None
            _atomic_torch(
                _checkpoint_payload(
                    identity=identity,
                    work_index=work_index,
                    completed=completed,
                    extrema=extrema,
                    active=None,
                    status="running" if work_index < len(plan) else "complete",
                ),
                checkpoint_path,
            )
            progress.write(
                "work_complete",
                completed_work_batches=work_index,
                total_work_batches=len(plan),
                completed_paths=spec.path_stop,
                total_paths=config.num_paths,
            )

        _validate_complete_outputs(outputs, shapes)
        report = {
            "schema": SCHEMA,
            "status": "complete",
            "identity": identity,
            "provenance": provenance,
            "population": population_metadata,
            "completed": completed,
            "extrema": extrema,
            "resources_final": _resource_status(config.output),
        }
        _atomic_json(report, report_path)
        progress.write("complete", report=str(report_path), extrema=extrema)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
