"""Joint JAX fit of GROMM2 DLV states to synthetic public-SANOS books."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.special import ndtr

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.market import load_public_sanos_days, public_sanos_prices

FloatArray = NDArray[np.float64]

_CAP_BARRIER_LOG_WIDTH = 0.10
_CAP_BARRIER_LOG_STD = 0.35


@dataclass(frozen=True, kw_only=True)
class CallKernelSpecification:
    """Static or date-varying call kernels used by the DLV decoder."""

    kernels: FloatArray
    identity: str
    metadata: str
    lower_indices: NDArray[np.int32] | None = None
    interpolation_weights: FloatArray | None = None

    @property
    def is_date_varying(self) -> bool:
        return self.lower_indices is not None


@dataclass(frozen=True, kw_only=True)
class DlvCalibrationConfig:
    """Controls matching the retained DYSANOS affine JAX fit."""

    rank: int = 200
    minimum_dlvsq: float = 0.01**2
    maximum_dlvsq: float = 3.0**2
    backbone_annual_volatility: float = 0.05
    fit_weight: float = 0.05
    cap_penalty_weight: float = 1.0
    epochs: int = 1_000
    learning_rate: float = 0.01
    minimum_learning_rate_fraction: float = 0.1
    checkpoint_every: int = 10
    report_every: int = 5
    plateau_review_epochs: int = 0
    seed: int = 20261012

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.epochs <= 0:
            raise ValueError("rank and epochs must be positive")
        if not 0.0 < self.minimum_dlvsq < self.maximum_dlvsq:
            raise ValueError("DLV-squared bounds are invalid")
        if self.backbone_annual_volatility < 0.0:
            raise ValueError("backbone volatility must be non-negative")
        if not 0.0 <= self.fit_weight <= 1.0:
            raise ValueError("fit_weight must lie in [0,1]")
        if self.cap_penalty_weight < 0.0:
            raise ValueError("cap_penalty_weight must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate_fraction <= 1.0:
            raise ValueError("minimum learning-rate fraction must lie in [0,1]")
        if self.checkpoint_every <= 0 or self.report_every <= 0:
            raise ValueError("checkpoint and report intervals must be positive")
        if self.plateau_review_epochs < 0:
            raise ValueError("plateau_review_epochs must be non-negative")


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_hdf5_path(path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(temporary), path


def _black_call_kernels(
    supports: FloatArray,
    strikes: FloatArray,
    expiries_y: FloatArray,
    *,
    annual_volatility: float,
) -> FloatArray:
    if annual_volatility == 0.0:
        return np.maximum(supports[:, None, :] - strikes[:, :, None], 0.0)
    total_volatility = annual_volatility * np.sqrt(expiries_y)
    log_ratio = np.log(supports[:, None, :] / strikes[:, :, None])
    d1 = log_ratio / total_volatility[:, None, None]
    d1 += 0.5 * total_volatility[:, None, None]
    d2 = d1 - total_volatility[:, None, None]
    return supports[:, None, :] * ndtr(d1) - strikes[:, :, None] * ndtr(d2)


def _load_call_kernel_specification(
    path: Path,
    *,
    dates: NDArray[np.str_],
    expected_shape: tuple[int, int, int],
) -> CallKernelSpecification:
    """Load a static v1 or interpolated date-varying v2 kernel bundle."""
    with np.load(path, allow_pickle=False) as values:
        schema = str(values["schema"].item())
        identity = str(values["identity"].item())
        metadata = str(values["metadata"].item())
        kernels = np.asarray(values["call_kernels"], dtype=np.float32)
        if schema == "gromm2-call-kernels-v1":
            if kernels.shape != expected_shape:
                raise ValueError(
                    f"call kernels have shape {kernels.shape}, expected {expected_shape}"
                )
            lower_indices = None
            interpolation_weights = None
        elif schema == "gromm2-call-kernels-v2":
            if kernels.ndim != 4 or kernels.shape[1:] != expected_shape:
                raise ValueError(
                    "call-kernel grid has shape "
                    f"{kernels.shape}, expected (G,{expected_shape[0]},"
                    f"{expected_shape[1]},{expected_shape[2]})"
                )
            variance_grid = np.asarray(values["variance_grid"], dtype=np.float64)
            date_variances = np.asarray(values["date_variances"], dtype=np.float64)
            kernel_dates = np.asarray(values["dates"]).astype("U10")
            if (
                variance_grid.shape != (len(kernels),)
                or len(variance_grid) < 2
                or np.any(~np.isfinite(variance_grid))
                or np.any(variance_grid <= 0.0)
                or np.any(np.diff(variance_grid) <= 0.0)
            ):
                raise ValueError("invalid call-kernel variance grid")
            if len(kernel_dates) < len(dates) or not np.array_equal(
                kernel_dates[: len(dates)], dates
            ):
                raise ValueError("call-kernel dates do not match calibration dates")
            date_variances = date_variances[: len(dates)]
            if (
                date_variances.shape != (len(dates),)
                or np.any(~np.isfinite(date_variances))
                or np.any(date_variances < variance_grid[0])
                or np.any(date_variances > variance_grid[-1])
            ):
                raise ValueError("date variances fall outside the call-kernel grid")
            lower_indices = np.searchsorted(
                variance_grid, date_variances, side="right"
            ) - 1
            lower_indices = np.clip(lower_indices, 0, len(variance_grid) - 2)
            lower_variance = variance_grid[lower_indices]
            upper_variance = variance_grid[lower_indices + 1]
            interpolation_weights = (
                (date_variances - lower_variance)
                / (upper_variance - lower_variance)
            )
            lower_indices = lower_indices.astype(np.int32)
        else:
            raise ValueError(f"unexpected call-kernel schema in {path}")
    if np.any(~np.isfinite(kernels)):
        raise ValueError("call kernels contain non-finite values")
    return CallKernelSpecification(
        kernels=kernels,
        identity=identity,
        metadata=metadata,
        lower_indices=lower_indices,
        interpolation_weights=interpolation_weights,
    )


def _price_call_kernels(
    kernels: jax.Array,
    densities: jax.Array,
    *,
    lower_indices: jax.Array | None,
    interpolation_weights: jax.Array | None,
) -> jax.Array:
    """Price densities with static kernels or streamed variance interpolation."""
    kernels = jnp.asarray(kernels)
    densities = jnp.asarray(densities)
    if lower_indices is None:
        return jnp.einsum("jkn,djn->djk", kernels, densities)
    assert interpolation_weights is not None
    lower_indices = jnp.asarray(lower_indices)
    interpolation_weights = jnp.asarray(interpolation_weights)

    return _price_interpolated_call_kernels(
        kernels,
        densities,
        lower_indices,
        interpolation_weights,
    )


def _price_interpolated_call_kernels_impl(
    kernels: jax.Array,
    densities: jax.Array,
    lower_indices: jax.Array,
    interpolation_weights: jax.Array,
) -> jax.Array:
    """Price with date-varying kernels without materializing date kernels."""

    def price_date(arguments: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        density, lower_index, weight = arguments
        lower_calls = jnp.einsum("jkn,jn->jk", kernels[lower_index], density)
        upper_calls = jnp.einsum("jkn,jn->jk", kernels[lower_index + 1], density)
        return lower_calls + weight * (upper_calls - lower_calls)

    return jax.lax.map(
        price_date,
        (densities, lower_indices, interpolation_weights),
    )


def _price_interpolated_call_kernels_forward(
    kernels: jax.Array,
    densities: jax.Array,
    lower_indices: jax.Array,
    interpolation_weights: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    calls = _price_interpolated_call_kernels_impl(
        kernels,
        densities,
        lower_indices,
        interpolation_weights,
    )
    return calls, (kernels, lower_indices, interpolation_weights)


def _price_interpolated_call_kernels_backward(
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    call_cotangents: jax.Array,
) -> tuple[None, jax.Array, None, None]:
    kernels, lower_indices, interpolation_weights = residuals

    def density_gradient(
        arguments: tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        lower_index, weight, call_cotangent = arguments
        kernel = kernels[lower_index] + weight * (
            kernels[lower_index + 1] - kernels[lower_index]
        )
        return jnp.einsum("jk,jkn->jn", call_cotangent, kernel)

    density_cotangents = jax.lax.map(
        density_gradient,
        (lower_indices, interpolation_weights, call_cotangents),
    )
    return None, density_cotangents, None, None


_price_interpolated_call_kernels = jax.custom_vjp(
    _price_interpolated_call_kernels_impl
)
_price_interpolated_call_kernels.defvjp(
    _price_interpolated_call_kernels_forward,
    _price_interpolated_call_kernels_backward,
)


def _unique_curve(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(x)
    sorted_x = x[order]
    sorted_y = y[order]
    unique, inverse = np.unique(sorted_x, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_y)
    counts = np.bincount(inverse)
    return unique, sums / counts


def _spread_surface(
    *,
    expiry_row: dict[str, object],
    instrument_row: dict[str, object],
    target_business_dtes: FloatArray,
    target_strikes: FloatArray,
    fit_nms_range: tuple[float, float],
) -> tuple[FloatArray, np.ndarray, FloatArray]:
    observed_dtes = np.asarray(expiry_row["business_dtes"], dtype=np.float64)
    forward_over_spot = np.asarray(
        expiry_row["forward_over_spot"], dtype=np.float64
    )
    discount_factors = np.asarray(
        expiry_row["discount_factors"], dtype=np.float64
    )
    atm_volatilities = np.asarray(
        expiry_row["book_selection_annualized_atm_vols"], dtype=np.float64
    )
    expiry_indices = np.asarray(
        instrument_row["expiry_indices"], dtype=np.int64
    )
    strikes = np.asarray(
        instrument_row["strikes_over_forward"], dtype=np.float64
    )
    nms = np.asarray(
        instrument_row["standardized_log_moneyness"], dtype=np.float64
    )
    is_call = np.asarray(instrument_row["is_call"], dtype=bool)
    retained = np.asarray(
        instrument_row["in_paper_retained_population"], dtype=bool
    )
    half_spreads = np.asarray(
        instrument_row["half_spreads_over_spot"], dtype=np.float64
    )
    half_spreads /= (
        discount_factors[expiry_indices] * forward_over_spot[expiry_indices]
    )
    retained &= (is_call & (strikes >= 1.0)) | (~is_call & (strikes < 1.0))
    curves: list[tuple[FloatArray, FloatArray] | None] = []
    for expiry_index in range(len(observed_dtes)):
        selected = retained & (expiry_indices == expiry_index)
        if np.count_nonzero(selected) < 2:
            curves.append(None)
            continue
        curves.append(_unique_curve(nms[selected], half_spreads[selected]))
    valid_expiries = np.asarray(
        [index for index, curve in enumerate(curves) if curve is not None],
        dtype=np.int64,
    )
    if len(valid_expiries) < 2:
        raise RuntimeError("synthetic book requires two observed spread expiries")
    target_atm_volatilities = np.interp(
        target_business_dtes,
        observed_dtes,
        atm_volatilities,
    )
    target_total_volatility = target_atm_volatilities * np.sqrt(
        target_business_dtes / 252.0
    )
    target_nms = np.log(target_strikes) / target_total_volatility[:, None]
    result = np.empty_like(target_strikes)
    valid_dtes = observed_dtes[valid_expiries]
    for target_index, target_dte in enumerate(target_business_dtes):
        values_by_expiry = []
        for expiry_index in valid_expiries:
            curve = curves[int(expiry_index)]
            assert curve is not None
            values_by_expiry.append(
                np.interp(target_nms[target_index], curve[0], curve[1])
            )
        values = np.stack(values_by_expiry)
        lower = int(np.searchsorted(valid_dtes, target_dte, side="right") - 1)
        lower = int(np.clip(lower, 0, len(valid_dtes) - 1))
        upper = min(lower + 1, len(valid_dtes) - 1)
        if upper == lower:
            result[target_index] = values[lower]
        else:
            alpha = (target_dte - valid_dtes[lower]) / (
                valid_dtes[upper] - valid_dtes[lower]
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            result[target_index] = (
                (1.0 - alpha) * values[lower] + alpha * values[upper]
            )
    positive_floor = max(
        float(np.quantile(half_spreads[retained], 0.01)), 1.0e-8
    )
    result = np.maximum(result, positive_floor)
    mask = (target_nms >= fit_nms_range[0]) & (target_nms <= fit_nms_range[1])
    return result, mask, target_nms


def prepare_synthetic_books(
    *,
    source_history: Path,
    dataset: Path,
    output: Path,
) -> None:
    """Create SANOS-mid synthetic books with published half-spread bands."""
    with h5py.File(source_history, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_dtes = np.asarray(
            stream["model_business_dtes"], dtype=np.float64
        )
        projected_variances = np.asarray(
            stream["projected_atm_total_variances"], dtype=np.float64
        )
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        geometry_config = json.loads(stream.attrs["geometry_config"])
        source_terminal_dte = np.asarray(
            stream["source_terminal_business_dte"], dtype=np.float64
        )
        terminal_extrapolated = np.asarray(
            stream["terminal_extrapolated"], dtype=bool
        )
    expiry_rows = {
        row["date"].isoformat(): row
        for row in pl.read_parquet(dataset / "expiries.parquet").iter_rows(
            named=True
        )
    }
    instrument_rows = {
        row["date"].isoformat(): row
        for row in pl.read_parquet(dataset / "instruments.parquet").iter_rows(
            named=True
        )
    }
    public_days = {
        day.trading_date.isoformat(): day for day in load_public_sanos_days(dataset)
    }
    missing = sorted(
        set(dates) - (set(expiry_rows) & set(instrument_rows) & set(public_days))
    )
    if missing:
        raise ValueError(f"public dataset is missing {len(missing)} dates")
    target_strikes = supports[:, 1:-1]
    mids = np.empty((len(dates),) + target_strikes.shape, dtype=np.float32)
    half_spreads = np.empty_like(mids)
    masks = np.empty_like(mids, dtype=bool)
    target_nms = np.empty_like(mids)
    started = perf_counter()
    for index, trading_date in enumerate(dates):
        calls = public_sanos_prices(
            public_days[trading_date],
            expiries_y=np.broadcast_to(expiries_y[:, None], target_strikes.shape),
            strikes=target_strikes,
        )
        mids[index] = np.where(
            target_strikes >= 1.0,
            calls,
            calls - 1.0 + target_strikes,
        )
        spreads, active, daily_nms = _spread_surface(
            expiry_row=expiry_rows[trading_date],
            instrument_row=instrument_rows[trading_date],
            target_business_dtes=business_dtes,
            target_strikes=target_strikes,
            fit_nms_range=tuple(geometry_config["fit_nms_range"]),
        )
        half_spreads[index] = spreads
        masks[index] = active
        target_nms[index] = daily_nms
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(dates):
            elapsed = perf_counter() - started
            rate = (index + 1) / max(elapsed, 1.0e-9)
            eta = (len(dates) - index - 1) / max(rate, 1.0e-9)
            print(
                f"synthetic-books completed={index + 1}/{len(dates)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    raw_initial = np.log(
        np.maximum(
            projected_variances / expiries_y[None, :] - 0.01**2,
            1.0e-8,
        )
    )[..., None]
    raw_initial = np.broadcast_to(raw_initial, mids.shape).astype(np.float32)
    temporary, destination = _atomic_hdf5_path(output)
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = "gromm2-synthetic-sanos-books-v1"
            stream.attrs["source_history"] = str(source_history.resolve())
            stream.attrs["source_dataset"] = str(dataset.resolve())
            stream.attrs["geometry_config"] = json.dumps(
                geometry_config, sort_keys=True
            )
            stream.create_dataset("dates", data=np.asarray(dates, dtype="S10"))
            stream.create_dataset("model_expiries_y", data=expiries_y)
            stream.create_dataset("model_business_dtes", data=business_dtes)
            stream.create_dataset("supports", data=supports)
            stream.create_dataset("transports", data=transports)
            stream.create_dataset("nms_nodes", data=nms_nodes)
            stream.create_dataset("target_strikes", data=target_strikes)
            stream.create_dataset(
                "source_terminal_business_dte", data=source_terminal_dte
            )
            stream.create_dataset(
                "terminal_extrapolated", data=terminal_extrapolated
            )
            for name, values in (
                ("mids", mids),
                ("half_spreads", half_spreads),
                ("mask", masks),
                ("target_nms", target_nms),
                ("raw_initial", raw_initial),
            ):
                stream.create_dataset(
                    name,
                    data=values,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            stream.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"wrote {destination}", flush=True)


def _initial_parameters(
    raw_initial: FloatArray,
    *,
    rank: int,
    seed: int,
) -> dict[str, FloatArray]:
    generator = np.random.default_rng(seed)
    mean = np.mean(raw_initial, axis=0, dtype=np.float64).astype(np.float32)
    codes = generator.normal(
        scale=1.0e-3, size=(len(raw_initial), rank)
    ).astype(np.float32)
    loadings = generator.normal(
        scale=1.0e-3, size=(rank,) + mean.shape
    ).astype(np.float32)
    return {"mean": mean, "codes": codes, "loadings": loadings}


def _load_warm_start_parameters(
    path: Path,
    *,
    expected_shapes: dict[str, tuple[int, ...]],
    source_dates: np.ndarray | None = None,
    target_dates: np.ndarray | None = None,
) -> tuple[dict[str, FloatArray], str]:
    with np.load(path, allow_pickle=False) as values:
        if str(values["schema"].item()) != "gromm2-affine-jax-checkpoint-v1":
            raise ValueError(f"unexpected warm-start checkpoint schema in {path}")
        parameters = {
            name: np.asarray(values[f"parameter_{name}"], dtype=np.float32)
            for name in ("mean", "codes", "loadings")
        }
        source_identity = str(values["identity"].item())
        completed_epochs = int(values["completed_epochs"])
    for name, expected_shape in expected_shapes.items():
        if name == "codes" and source_dates is not None:
            continue
        actual_shape = parameters[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"warm-start {name} shape {actual_shape} does not match "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(parameters[name])):
            raise ValueError(f"warm-start {name} contains non-finite values")
    if (source_dates is None) != (target_dates is None):
        raise ValueError("warm-start source and target dates must be supplied together")
    if source_dates is not None and target_dates is not None:
        source = np.asarray(source_dates).astype("U10")
        target = np.asarray(target_dates).astype("U10")
        if (
            source.shape != (parameters["codes"].shape[0],)
            or target.shape != (expected_shapes["codes"][0],)
            or len(source) < 2
            or len(target) < 2
            or not np.all(source[:-1] < source[1:])
            or not np.all(target[:-1] < target[1:])
            or len(set(source)) != len(source)
            or len(set(target)) != len(target)
            or source[-1] != target[-2]
            or target[-1] <= source[-1]
        ):
            raise ValueError("warm-start rolling date windows do not align")
        shared_indices = np.searchsorted(source, target[:-1])
        if (
            np.any(shared_indices >= len(source))
            or not np.array_equal(source[shared_indices], target[:-1])
            or not np.array_equal(
                shared_indices,
                np.arange(shared_indices[0], len(source))
            )
        ):
            raise ValueError("warm-start rolling date windows do not align")
        previous_codes = parameters["codes"]
        parameters["codes"] = np.concatenate(
            (previous_codes[shared_indices], previous_codes[-1:]), axis=0
        )
    if parameters["codes"].shape != expected_shapes["codes"]:
        raise ValueError(
            f"warm-start codes shape {parameters['codes'].shape} does not match "
            f"expected {expected_shapes['codes']}"
        )
    source_digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
    identity = f"sha256:{source_digest}:completed-epochs={completed_epochs}"
    if source_dates is not None and target_dates is not None:
        identity += f":rolled={source[-1]}->{target[-1]}"
    return parameters, identity


def _raw_states(parameters: dict[str, jax.Array]) -> jax.Array:
    return parameters["mean"][None] + jnp.einsum(
        "dr,rjn->djn", parameters["codes"], parameters["loadings"]
    )


def _checkpoint_identity(
    config: DlvCalibrationConfig,
    *,
    data_identity: str = "black-default",
) -> str:
    identity = {
        "rank": config.rank,
        "minimum_dlvsq": config.minimum_dlvsq,
        "maximum_dlvsq": config.maximum_dlvsq,
        "backbone_annual_volatility": config.backbone_annual_volatility,
        "fit_weight": config.fit_weight,
        "cap_penalty_weight": config.cap_penalty_weight,
    }
    if data_identity != "black-default":
        identity["data_identity"] = data_identity
    return json.dumps(identity, sort_keys=True)


def _save_checkpoint(
    path: Path,
    *,
    parameters: dict[str, jax.Array],
    first: dict[str, jax.Array],
    second: dict[str, jax.Array],
    step: jax.Array,
    completed_epochs: int,
    config: DlvCalibrationConfig,
    data_identity: str = "black-default",
) -> None:
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("gromm2-affine-jax-checkpoint-v1"),
        "config": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "identity": np.asarray(
            _checkpoint_identity(config, data_identity=data_identity)
        ),
        "completed_epochs": np.asarray(completed_epochs, dtype=np.int32),
        "step": np.asarray(step),
    }
    for prefix, tree in (
        ("parameter", parameters),
        ("adam_first", first),
        ("adam_second", second),
    ):
        for name, value in tree.items():
            payload[f"{prefix}_{name}"] = np.asarray(value)
    _atomic_npz(path, payload)


def _load_checkpoint(
    path: Path,
    *,
    config: DlvCalibrationConfig,
    data_identity: str = "black-default",
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    int,
]:
    with np.load(path, allow_pickle=False) as values:
        if str(values["schema"].item()) != "gromm2-affine-jax-checkpoint-v1":
            raise ValueError(f"unexpected checkpoint schema in {path}")
        if str(values["identity"].item()) != _checkpoint_identity(
            config,
            data_identity=data_identity,
        ):
            raise ValueError("checkpoint model identity does not match")
        parameters = {
            name: np.asarray(values[f"parameter_{name}"])
            for name in ("mean", "codes", "loadings")
        }
        first = {
            name: np.asarray(values[f"adam_first_{name}"])
            for name in parameters
        }
        second = {
            name: np.asarray(values[f"adam_second_{name}"])
            for name in parameters
        }
        return (
            parameters,
            first,
            second,
            np.asarray(values["step"]),
            int(values["completed_epochs"]),
        )


def fit_affine_history(
    *,
    prepared_books: Path,
    output: Path,
    checkpoint: Path,
    config: DlvCalibrationConfig,
    call_kernels: Path | None = None,
    end_date_exclusive: str | None = None,
    warm_start_checkpoint: Path | None = None,
    warm_start_prepared_books: Path | None = None,
    evaluate_only: bool = False,
) -> None:
    """Fit the affine DLV history with the DYSANOS bid--ask objective."""
    if jax.default_backend() not in {"cuda", "gpu"}:
        raise RuntimeError("the full affine GROMM2 fit requires a managed JAX GPU")
    with h5py.File(prepared_books, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-synthetic-sanos-books-v1":
            raise ValueError(f"unexpected synthetic-book schema in {prepared_books}")
        dates = np.asarray(stream["dates"])
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float32)
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float32)
        supports = np.asarray(stream["supports"], dtype=np.float32)
        transports = np.asarray(stream["transports"], dtype=np.float32)
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float32)
        strikes = np.asarray(stream["target_strikes"], dtype=np.float32)
        mids = np.asarray(stream["mids"], dtype=np.float32)
        half_spreads = np.asarray(stream["half_spreads"], dtype=np.float32)
        mask = np.asarray(stream["mask"], dtype=bool)
        raw_initial = np.asarray(stream["raw_initial"], dtype=np.float32)
        terminal_dte = np.asarray(stream["source_terminal_business_dte"])
        terminal_extrapolated = np.asarray(stream["terminal_extrapolated"])
        geometry_config = str(stream.attrs["geometry_config"])
        source_history = str(stream.attrs["source_history"])
        source_dataset = str(stream.attrs["source_dataset"])
    decoded_dates = dates.astype("U10")
    if end_date_exclusive is not None:
        retained = decoded_dates < end_date_exclusive
        retained_count = int(np.count_nonzero(retained))
        expected = np.arange(len(dates)) < retained_count
        if retained_count == 0 or retained_count == len(dates):
            raise ValueError(
                "end_date_exclusive must retain a non-empty strict prefix"
            )
        if not np.array_equal(retained, expected):
            raise ValueError("end_date_exclusive must define a chronological prefix")
        dates = dates[retained]
        decoded_dates = decoded_dates[retained]
        mids = mids[retained]
        half_spreads = half_spreads[retained]
        mask = mask[retained]
        raw_initial = raw_initial[retained]
        terminal_dte = terminal_dte[retained]
        terminal_extrapolated = terminal_extrapolated[retained]
    if config.rank >= len(dates):
        raise ValueError("affine rank must be smaller than the number of dates")
    intervals = np.diff(expiries_y, prepend=np.float32(0.0))
    if call_kernels is None:
        kernels = _black_call_kernels(
            supports.astype(np.float64),
            strikes.astype(np.float64),
            expiries_y.astype(np.float64),
            annual_volatility=config.backbone_annual_volatility,
        ).astype(np.float32)
        kernel_identity = (
            f"black-annual-volatility={config.backbone_annual_volatility:.17g}"
        )
        kernel_metadata = json.dumps(
            {
                "model": "Black",
                "annual_volatility": config.backbone_annual_volatility,
            },
            sort_keys=True,
        )
        lower_indices = None
        interpolation_weights = None
    else:
        expected_shape = (len(expiries_y), strikes.shape[1], supports.shape[1])
        specification = _load_call_kernel_specification(
            call_kernels,
            dates=decoded_dates,
            expected_shape=expected_shape,
        )
        kernels = specification.kernels
        kernel_identity = specification.identity
        kernel_metadata = specification.metadata
        lower_indices = specification.lower_indices
        interpolation_weights = specification.interpolation_weights
    if warm_start_prepared_books is not None and warm_start_checkpoint is None:
        raise ValueError("warm-start books require a warm-start checkpoint")
    warm_start_parameters = None
    warm_start_identity = "none"
    if warm_start_checkpoint is not None:
        source_dates = None
        if warm_start_prepared_books is not None:
            with h5py.File(warm_start_prepared_books, "r") as source_stream:
                if source_stream.attrs.get("schema") != "gromm2-synthetic-sanos-books-v1":
                    raise ValueError("warm-start prepared-book schema is incompatible")
                source_dates = np.asarray(source_stream["dates"])
        warm_start_parameters, warm_start_identity = _load_warm_start_parameters(
            warm_start_checkpoint,
            expected_shapes={
                "mean": raw_initial.shape[1:],
                "codes": (len(raw_initial), config.rank),
                "loadings": (config.rank, *raw_initial.shape[1:]),
            },
            source_dates=source_dates,
            target_dates=decoded_dates if source_dates is not None else None,
        )
    data_identity = json.dumps(
        {
            "kernel": kernel_identity,
            "first_date": str(decoded_dates[0]),
            "last_date": str(decoded_dates[-1]),
            "date_count": len(decoded_dates),
            "warm_start": warm_start_identity,
        },
        sort_keys=True,
    )
    is_call = strikes >= 1.0
    arrays = {
        "supports": jax.device_put(jnp.asarray(supports)),
        "transports": jax.device_put(jnp.asarray(transports)),
        "intervals": jax.device_put(jnp.asarray(intervals)),
        "kernels": jax.device_put(jnp.asarray(kernels)),
        "strikes": jax.device_put(jnp.asarray(strikes)),
        "is_call": jax.device_put(jnp.asarray(is_call)),
        "mids": jax.device_put(jnp.asarray(mids)),
        "half_spreads": jax.device_put(jnp.asarray(half_spreads)),
        "mask": jax.device_put(jnp.asarray(mask)),
    }
    if lower_indices is not None:
        arrays["kernel_lower_indices"] = jax.device_put(jnp.asarray(lower_indices))
        arrays["kernel_interpolation_weights"] = jax.device_put(
            jnp.asarray(interpolation_weights, dtype=jnp.float32)
        )
    maximum_raw = math.log(config.maximum_dlvsq - config.minimum_dlvsq)

    def decode(parameters: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        raw = _raw_states(parameters)
        dlvsq = config.minimum_dlvsq + jnp.exp(jnp.minimum(raw, maximum_raw))
        densities = decode_density_tower(
            arrays["supports"], arrays["transports"], dlvsq, arrays["intervals"]
        )
        calls = _price_call_kernels(
            arrays["kernels"],
            densities,
            lower_indices=arrays.get("kernel_lower_indices"),
            interpolation_weights=arrays.get("kernel_interpolation_weights"),
        )
        prices = jnp.where(
            arrays["is_call"][None],
            calls,
            calls - 1.0 + arrays["strikes"][None],
        )
        return raw, prices

    def loss(
        parameters: dict[str, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        raw, prices = decode(parameters)
        absolute_error = jnp.abs(prices - arrays["mids"])
        scaled_error = absolute_error / arrays["half_spreads"]
        valid = arrays["mask"].astype(prices.dtype)
        quote_loss = jnp.sum(
            (jax.nn.relu(scaled_error - 1.0) + config.fit_weight * scaled_error)
            * valid
        ) / jnp.sum(valid)
        scaled_cap = (raw - maximum_raw) / _CAP_BARRIER_LOG_WIDTH
        barrier = (
            _CAP_BARRIER_LOG_WIDTH
            * jax.nn.softplus(scaled_cap)
            / _CAP_BARRIER_LOG_STD
        )
        cap_penalty = 0.5 * jnp.mean(jnp.sum(barrier**2, axis=(1, 2)))
        objective = quote_loss + config.cap_penalty_weight * cap_penalty
        return objective, (quote_loss, cap_penalty)

    value_and_gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))
    decode_only = jax.jit(decode)

    @jax.jit
    def adam_update(
        parameters: dict[str, jax.Array],
        first: dict[str, jax.Array],
        second: dict[str, jax.Array],
        step: jax.Array,
        gradients: dict[str, jax.Array],
        learning_rate: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        dict[str, jax.Array],
        dict[str, jax.Array],
        jax.Array,
        jax.Array,
    ]:
        next_step = step + 1
        next_first = jax.tree.map(
            lambda value, gradient: 0.9 * value + 0.1 * gradient,
            first,
            gradients,
        )
        next_second = jax.tree.map(
            lambda value, gradient: 0.999 * value + 0.001 * gradient**2,
            second,
            gradients,
        )
        corrected_first = jax.tree.map(
            lambda value: value / (1.0 - 0.9**next_step), next_first
        )
        corrected_second = jax.tree.map(
            lambda value: value / (1.0 - 0.999**next_step), next_second
        )
        next_parameters = jax.tree.map(
            lambda value, moment, square: value
            - learning_rate * moment / (jnp.sqrt(square) + 1.0e-8),
            parameters,
            corrected_first,
            corrected_second,
        )
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(value**2) for value in jax.tree.leaves(gradients))
        )
        return next_parameters, next_first, next_second, next_step, gradient_norm

    if checkpoint.exists():
        parameters_cpu, first_cpu, second_cpu, step_cpu, completed_epochs = (
            _load_checkpoint(
                checkpoint,
                config=config,
                data_identity=data_identity,
            )
        )
    else:
        parameters_cpu = (
            warm_start_parameters
            if warm_start_parameters is not None
            else _initial_parameters(raw_initial, rank=config.rank, seed=config.seed)
        )
        first_cpu = jax.tree.map(np.zeros_like, parameters_cpu)
        second_cpu = jax.tree.map(np.zeros_like, parameters_cpu)
        step_cpu = np.asarray(0, dtype=np.int32)
        completed_epochs = 0
    if evaluate_only and completed_epochs == 0:
        raise ValueError("evaluate_only requires an existing compatible checkpoint")
    parameters = jax.device_put(jax.tree.map(jnp.asarray, parameters_cpu))
    first = jax.device_put(jax.tree.map(jnp.asarray, first_cpu))
    second = jax.device_put(jax.tree.map(jnp.asarray, second_cpu))
    step = jax.device_put(jnp.asarray(step_cpu))
    started = perf_counter()
    best_objective = math.inf
    best_objective_epoch = completed_epochs
    final_epoch = completed_epochs
    stopped_for_plateau_review = False
    for epoch in range(completed_epochs + 1, config.epochs + 1):
        if evaluate_only:
            break
        fraction = (epoch - 1) / max(config.epochs - 1, 1)
        learning_rate = config.learning_rate * (
            config.minimum_learning_rate_fraction
            + (1.0 - config.minimum_learning_rate_fraction)
            * 0.5
            * (1.0 + math.cos(math.pi * fraction))
        )
        (objective, auxiliary), gradients = value_and_gradient(parameters)
        quote_loss, cap_penalty = auxiliary
        objective_value = float(objective)
        if objective_value < best_objective:
            best_objective = objective_value
            best_objective_epoch = epoch
        parameters, first, second, step, gradient_norm = adam_update(
            parameters,
            first,
            second,
            step,
            gradients,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            _save_checkpoint(
                checkpoint,
                parameters=parameters,
                first=first,
                second=second,
                step=step,
                completed_epochs=epoch,
                config=config,
                data_identity=data_identity,
            )
        if epoch == 1 or epoch % config.report_every == 0 or epoch == config.epochs:
            elapsed = perf_counter() - started
            rate = (epoch - completed_epochs) / max(elapsed, 1.0e-9)
            eta = (config.epochs - epoch) / max(rate, 1.0e-9)
            print(
                f"affine rank={config.rank} epoch={epoch}/{config.epochs} "
                f"objective={float(objective):.7g} quote={float(quote_loss):.7g} "
                f"cap={float(cap_penalty):.3g} grad={float(gradient_norm):.3g} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        final_epoch = epoch
        if (
            config.plateau_review_epochs > 0
            and epoch - best_objective_epoch >= config.plateau_review_epochs
        ):
            _save_checkpoint(
                checkpoint,
                parameters=parameters,
                first=first,
                second=second,
                step=step,
                completed_epochs=epoch,
                config=config,
                data_identity=data_identity,
            )
            print(
                f"affine rank={config.rank} plateau review after "
                f"{config.plateau_review_epochs} epochs without a new "
                f"objective minimum; best={best_objective:.7g} "
                f"at epoch={best_objective_epoch}; checkpoint saved",
                flush=True,
            )
            stopped_for_plateau_review = True
            break
    raw, prices = decode_only(parameters)
    raw_np = np.asarray(raw)
    dlvsq = config.minimum_dlvsq + np.exp(np.minimum(raw_np, maximum_raw))
    prices_np = np.asarray(prices)
    absolute_error = np.abs(prices_np - mids)
    scaled_error = absolute_error / half_spreads
    excess = np.maximum(scaled_error - 1.0, 0.0)[mask]
    temporary, destination = _atomic_hdf5_path(output)
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = "gromm2-affine-dlv-history-v1"
            stream.attrs["calibration_config"] = json.dumps(
                asdict(config), sort_keys=True
            )
            stream.attrs["completed_epochs"] = final_epoch
            stream.attrs["stopped_for_plateau_review"] = stopped_for_plateau_review
            stream.attrs["geometry_config"] = geometry_config
            stream.attrs["source_history"] = source_history
            stream.attrs["source_dataset"] = source_dataset
            stream.attrs["call_kernel_identity"] = kernel_identity
            stream.attrs["call_kernel_metadata"] = kernel_metadata
            stream.attrs["warm_start_identity"] = warm_start_identity
            if end_date_exclusive is not None:
                stream.attrs["end_date_exclusive"] = end_date_exclusive
            stream.attrs["percent_inside_spread"] = float(
                100.0 * np.mean(excess == 0.0)
            )
            stream.attrs["mean_excess_half_spreads"] = float(np.mean(excess))
            stream.attrs["midpoint_mae_half_spreads"] = float(
                np.mean(scaled_error[mask])
            )
            stream.create_dataset("dates", data=dates)
            stream.create_dataset("model_expiries_y", data=expiries_y)
            stream.create_dataset("model_business_dtes", data=business_dtes)
            stream.create_dataset("supports", data=supports)
            stream.create_dataset("transports", data=transports)
            stream.create_dataset("nms_nodes", data=nms_nodes)
            stream.create_dataset(
                "source_terminal_business_dte", data=terminal_dte
            )
            stream.create_dataset(
                "terminal_extrapolated", data=terminal_extrapolated
            )
            stream.create_dataset(
                "dlvsq",
                data=dlvsq.astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            stream.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"wrote {destination}; inside={100.0 * np.mean(excess == 0.0):.2f}% "
        f"mean_excess={np.mean(excess):.5f} midpoint_mae={np.mean(scaled_error[mask]):.5f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-history", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--prepared-books", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--rank", type=int, default=200)
    fit.add_argument("--epochs", type=int, default=1_000)
    fit.add_argument("--learning-rate", type=float, default=0.01)
    fit.add_argument("--backbone-annual-volatility", type=float, default=0.05)
    fit.add_argument("--checkpoint-every", type=int, default=10)
    fit.add_argument("--plateau-review-epochs", type=int, default=0)
    fit.add_argument("--evaluate-only", action="store_true")
    fit.add_argument("--call-kernels", type=Path)
    fit.add_argument("--end-date-exclusive")
    fit.add_argument("--warm-start-checkpoint", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare_synthetic_books(
            source_history=arguments.source_history,
            dataset=arguments.dataset,
            output=arguments.output,
        )
    else:
        fit_affine_history(
            prepared_books=arguments.prepared_books,
            output=arguments.output,
            checkpoint=arguments.checkpoint,
            config=DlvCalibrationConfig(
                rank=arguments.rank,
                epochs=arguments.epochs,
                learning_rate=arguments.learning_rate,
                backbone_annual_volatility=arguments.backbone_annual_volatility,
                checkpoint_every=arguments.checkpoint_every,
                plateau_review_epochs=arguments.plateau_review_epochs,
            ),
            call_kernels=arguments.call_kernels,
            end_date_exclusive=arguments.end_date_exclusive,
            warm_start_checkpoint=arguments.warm_start_checkpoint,
            evaluate_only=arguments.evaluate_only,
        )


if __name__ == "__main__":
    main()
