"""Project ragged-grid physical drift onto the attainable GROMM2 factor span."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from time import perf_counter

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import apply_sanos_matrix, solve_sanos_step
from quantdata.gromm2.geometry import barycentric_transport

FloatArray = NDArray[np.float64]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_flat(
    values: FloatArray,
    offsets: NDArray[np.int64],
) -> tuple[FloatArray, ...]:
    return tuple(
        values[..., offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    )


def _weighted_quantile(
    values: FloatArray,
    weights: FloatArray,
    probability: float,
) -> float:
    selected = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    ordered = np.argsort(values[selected])
    ordered_values = values[selected][ordered]
    ordered_weights = weights[selected][ordered]
    cumulative = np.cumsum(ordered_weights)
    index = np.searchsorted(cumulative, probability * cumulative[-1])
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _ragged_q_drift_function(
    *,
    supports: tuple[FloatArray, ...],
    transports: tuple[FloatArray, ...],
    interval_years: FloatArray,
    annualized_loadings: FloatArray,
    future_offsets: NDArray[np.int64],
    front_years: float,
    density_loading_floor: float,
):
    factor_count = annualized_loadings.shape[1]
    supports_jax = tuple(jnp.asarray(value) for value in supports)
    transports_jax = tuple(jnp.asarray(value) for value in transports)
    loading_blocks = tuple(
        jnp.asarray(
            annualized_loadings[
                future_offsets[index] : future_offsets[index + 1]
            ]
        )
        for index in range(len(future_offsets) - 1)
    )
    intervals_jax = tuple(jnp.asarray(value) for value in interval_years)

    def calculate(
        front_density: jax.Array,
        future_dlvsq: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dlvsq_blocks = _split_flat(future_dlvsq, future_offsets)
        previous_density = front_density
        previous_loading = jnp.zeros(
            front_density.shape + (factor_count,), dtype=front_density.dtype
        )
        previous_loading = previous_loading.at[..., 0].set(
            front_density_volatility(
                front_density,
                remaining_time=jnp.asarray(front_years),
            )
        )
        drift_blocks: list[jax.Array] = []
        effective_blocks: list[jax.Array] = []
        weight_blocks: list[jax.Array] = []
        expiry_count = len(supports_jax) - 1
        for index in range(expiry_count):
            support = supports_jax[index + 1]
            transport = transports_jax[index]
            local_variance = dlvsq_blocks[index]
            interval = intervals_jax[index]
            transported_density = jnp.einsum(
                "ij,...j->...i", transport, previous_density
            )
            transported_loading = jnp.einsum(
                "ij,...jf->...if", transport, previous_loading
            )
            current_density = solve_sanos_step(
                support,
                transported_density,
                local_variance * interval,
            )
            interior = jnp.maximum(current_density[..., 1:-1], 0.0)
            scale = jnp.maximum(
                jnp.max(interior, axis=-1, keepdims=True),
                jnp.finfo(interior.dtype).tiny,
            )
            localization = interior / (
                interior + density_loading_floor * scale
            )
            current_xi = loading_blocks[index] * localization[..., None]
            boundary_zeros = jnp.zeros(
                current_xi.shape[:-2] + (1, factor_count),
                dtype=current_xi.dtype,
            )
            extended_xi = jnp.concatenate(
                (boundary_zeros, current_xi, boundary_zeros), axis=-2
            )
            xi_density = extended_xi * current_density[..., :, None]
            matrix_value = apply_sanos_matrix(
                support,
                xi_density,
                local_variance * interval,
            )
            current_loading = solve_sanos_step(
                support,
                transported_loading + xi_density - matrix_value,
                local_variance * interval,
            )
            safe_density = jnp.maximum(
                current_density[..., 1:-1, None],
                jnp.finfo(current_density.dtype).tiny,
            )
            relative_density_loading = (
                current_loading[..., 1:-1, :] / safe_density
            )
            relative_level_drift = -jnp.sum(
                current_xi * relative_density_loading, axis=-1
            )
            log_drift = relative_level_drift - 0.5 * jnp.sum(
                current_xi**2, axis=-1
            )
            density_weight = interior / jnp.maximum(
                jnp.sum(interior, axis=-1, keepdims=True),
                jnp.finfo(interior.dtype).tiny,
            )
            drift_blocks.append(log_drift)
            effective_blocks.append(current_xi)
            weight_blocks.append(density_weight / expiry_count)
            previous_density = current_density
            previous_loading = current_loading
        return (
            jnp.concatenate(drift_blocks, axis=-1),
            jnp.concatenate(effective_blocks, axis=-2),
            jnp.concatenate(weight_blocks, axis=-1),
        )

    return jax.jit(calculate)


def estimate_ragged_risk_prices(
    *,
    history_path: Path,
    joint_diffusion_path: Path,
    output: Path,
    level_rank: int = 8,
    ridge: float = 1.0e-3,
    batch_size: int = 32,
    dt: float = 1.0 / 252.0,
    density_loading_floor: float = 1.0e-4,
) -> dict[str, object]:
    """Estimate exact-Q-compatible physical factor risk prices."""
    with h5py.File(history_path, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-causal-multiscale-history-v1":
            raise ValueError("history is not a causal multiscale GROMM2 panel")
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        support_offsets = np.asarray(stream["support_offsets"], dtype=np.int64)
        supports = _split_flat(
            np.asarray(stream["supports"], dtype=np.float64), support_offsets
        )
        dlvsq_offsets = np.asarray(stream["dlvsq_offsets"], dtype=np.int64)
        dlvsq = np.asarray(stream["dlvsq"], dtype=np.float64)
        densities = np.asarray(stream["densities"], dtype=np.float64)
    with np.load(joint_diffusion_path, allow_pickle=False) as values:
        if str(values["schema"].item()) != "gromm2-causal-joint-diffusion-v1":
            raise ValueError("joint diffusion has an unexpected schema")
        origin_indices = np.asarray(values["origin_indices"], dtype=np.int64)
        factor_scores = np.asarray(values["factor_scores"], dtype=np.float64)
        raw_spot_scores = np.asarray(values["raw_spot_scores"], dtype=np.float64)
        ar_intercept = np.asarray(values["ar_intercept"], dtype=np.float64)
        ar_slope = np.asarray(values["ar_slope"], dtype=np.float64)
        annualized_loadings = np.asarray(
            values["annualized_log_dlvsq_loadings"], dtype=np.float64
        )
        factor_variances = np.asarray(values["factor_variances"], dtype=np.float64)
        core_mask = np.asarray(values["core_mask"], dtype=bool)
    if batch_size <= 0 or level_rank <= 0:
        raise ValueError("batch_size and level_rank must be positive")
    if factor_scores.shape[0] != len(origin_indices):
        raise ValueError("joint factor scores and origin indices are misaligned")
    first_future = int(dlvsq_offsets[1])
    future_offsets = dlvsq_offsets[1:] - first_future
    origins = np.log(dlvsq[origin_indices, first_future:])
    if annualized_loadings.shape[:1] != origins.shape[1:]:
        raise ValueError("ragged loadings do not match future DLV coordinates")
    empirical_p_drift = (
        ar_intercept + ar_slope * origins - origins
    ) / dt
    future_dlvsq = dlvsq[origin_indices, first_future:]
    front_densities = densities[
        origin_indices, support_offsets[0] : support_offsets[1]
    ]
    transports = tuple(
        barycentric_transport(supports[index], supports[index + 1])
        for index in range(len(supports) - 1)
    )
    intervals = np.diff(business_dtes) / 252.0
    q_drift_function = _ragged_q_drift_function(
        supports=supports,
        transports=transports,
        interval_years=intervals,
        annualized_loadings=annualized_loadings,
        future_offsets=future_offsets,
        front_years=float(business_dtes[0] / 252.0),
        density_loading_floor=density_loading_floor,
    )
    q_drift_blocks: list[FloatArray] = []
    effective_blocks: list[FloatArray] = []
    weight_blocks: list[FloatArray] = []
    started = perf_counter()
    for start in range(0, len(origins), batch_size):
        stop = min(start + batch_size, len(origins))
        q_drift, effective, weights = q_drift_function(
            jnp.asarray(front_densities[start:stop]),
            jnp.asarray(future_dlvsq[start:stop]),
        )
        q_drift_blocks.append(np.asarray(q_drift))
        effective_blocks.append(np.asarray(effective))
        weight_blocks.append(np.asarray(weights))
        completed = stop
        if start == 0 or completed == len(origins) or completed % (4 * batch_size) == 0:
            elapsed = perf_counter() - started
            rate = completed / max(elapsed, 1.0e-9)
            eta = (len(origins) - completed) / max(rate, 1.0e-9)
            print(
                f"ragged-q-drift completed={completed}/{len(origins)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    q_drift = np.concatenate(q_drift_blocks)
    effective_loadings = np.concatenate(effective_blocks)
    coordinate_weights = np.concatenate(weight_blocks)
    target = empirical_p_drift - q_drift
    factor_count = factor_scores.shape[1]
    risk_prices = np.empty((len(origins), factor_count), dtype=np.float64)
    spot_market_price = float(np.mean(raw_spot_scores) / math.sqrt(dt))
    risk_prices[:, 0] = spot_market_price
    fitted = np.empty_like(target)
    for sample in range(len(origins)):
        loadings = effective_loadings[sample]
        weights = coordinate_weights[sample]
        residual_target = target[sample] - loadings[:, 0] * spot_market_price
        remaining = loadings[:, 1:]
        weighted = remaining * weights[:, None]
        gram = remaining.T @ weighted
        penalty = ridge * max(float(np.trace(gram)) / len(gram), 1.0)
        risk_prices[sample, 1:] = np.linalg.solve(
            gram + penalty * np.eye(factor_count - 1),
            remaining.T @ (weights * residual_target),
        )
        fitted[sample] = loadings @ risk_prices[sample]
    target_energy = float(np.sum(coordinate_weights * target**2))
    residual_energy = float(
        np.sum(coordinate_weights * (target - fitted) ** 2)
    )
    projection_fraction = (
        1.0 - residual_energy / target_energy
        if target_energy > np.finfo(np.float64).tiny
        else 1.0
    )

    level_mean = np.mean(origins, axis=0)
    centered_levels = origins - level_mean
    _, singular, right = np.linalg.svd(centered_levels, full_matrices=False)
    retained_level_rank = min(level_rank, len(singular))
    level_basis = right[:retained_level_rank].T
    level_scores = centered_levels @ level_basis
    level_scale = np.std(level_scores, axis=0, ddof=1)
    level_scale = np.maximum(level_scale, 1.0e-8)
    standardized_level_scores = level_scores / level_scale

    reference_weights = coordinate_weights[-1] * core_mask
    reference_weights /= np.sum(reference_weights)
    log_dlvsq_volatility = np.sqrt(np.sum(annualized_loadings**2, axis=1))
    spot_correlation = np.divide(
        annualized_loadings[:, 0],
        log_dlvsq_volatility,
        out=np.zeros_like(log_dlvsq_volatility),
        where=log_dlvsq_volatility > 0.0,
    )
    arrays = {
        "schema": np.asarray("gromm2-ragged-compatible-dynamics-v1"),
        "dates": dates[origin_indices].astype("S10"),
        "pair_origin_indices": origin_indices,
        "factor_scores": factor_scores,
        "factor_variances": factor_variances,
        "physical_market_prices_of_risk": risk_prices,
        "spot_market_price_of_risk": np.asarray(spot_market_price),
        "physical_projection_fraction": np.asarray(projection_fraction),
        "annualized_log_dlvsq_loadings": annualized_loadings,
        "level_mean": level_mean,
        "level_basis": level_basis,
        "level_scale": level_scale,
        "level_scores": standardized_level_scores,
        "future_dlvsq_offsets": future_offsets,
        "core_mask": core_mask,
        "reference_density_weights": reference_weights,
    }
    _atomic_save_npz(output, arrays)
    volatility_summary = {
        f"density_weighted_p{probability:02d}": _weighted_quantile(
            log_dlvsq_volatility,
            reference_weights,
            probability / 100.0,
        )
        for probability in (10, 50, 90, 99)
    }
    correlation_summary = {
        f"density_weighted_p{probability:02d}": _weighted_quantile(
            spot_correlation,
            reference_weights,
            probability / 100.0,
        )
        for probability in (10, 50, 90)
    }
    correlation_summary["density_weighted_mean"] = float(
        reference_weights @ spot_correlation
    )
    report: dict[str, object] = {
        "schema": "gromm2-ragged-compatible-dynamics-v1",
        "history_path": str(history_path.resolve()),
        "history_sha256": _file_sha256(history_path),
        "joint_diffusion_path": str(joint_diffusion_path.resolve()),
        "joint_diffusion_sha256": _file_sha256(joint_diffusion_path),
        "pair_count": len(origins),
        "factor_count": factor_count,
        "level_rank": retained_level_rank,
        "strictly_pre_origin": True,
        "spot_market_price_of_risk": spot_market_price,
        "physical_projection_fraction": projection_fraction,
        "annualized_log_dlvsq_volatility": volatility_summary,
        "spot_correlation": correlation_summary,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--joint-diffusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    report = estimate_ragged_risk_prices(
        history_path=arguments.history,
        joint_diffusion_path=arguments.joint_diffusion,
        output=arguments.output,
        level_rank=arguments.level_rank,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
