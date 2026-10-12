"""Estimated constant and selective GROMM2 diffusion fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

Array = jax.Array
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class EstimatedDynamics:
    """Serialized joint spot/log-DLV diffusion and selective state model."""

    log_dlvsq_loadings: FloatArray
    physical_ar_intercept: FloatArray
    physical_ar_slope: FloatArray
    level_mean: FloatArray
    level_basis: FloatArray
    level_scale: FloatArray
    physical_factor_coefficients: FloatArray
    ssm_input_mean: FloatArray
    ssm_input_scale: FloatArray
    ssm_terminal_hidden: FloatArray
    ssm_parameters: dict[str, FloatArray]

    @property
    def factor_count(self) -> int:
        return self.log_dlvsq_loadings.shape[-1]


def load_estimated_dynamics(path: Path) -> EstimatedDynamics:
    """Load one estimated GROMM2 dynamics artifact."""
    with np.load(path, allow_pickle=False) as values:
        parameters = {
            name.removeprefix("ssm_"): np.asarray(values[name], dtype=np.float64)
            for name in values.files
            if name.startswith("ssm_")
            and name not in {"ssm_input_mean", "ssm_input_scale", "ssm_terminal_hidden"}
        }
        return EstimatedDynamics(
            log_dlvsq_loadings=np.asarray(
                values["log_dlvsq_loadings"], dtype=np.float64
            ),
            physical_ar_intercept=np.asarray(
                values["physical_ar_intercept"], dtype=np.float64
            ),
            physical_ar_slope=np.asarray(
                values["physical_ar_slope"], dtype=np.float64
            ),
            level_mean=np.asarray(values["level_mean"], dtype=np.float64),
            level_basis=np.asarray(values["level_basis"], dtype=np.float64),
            level_scale=np.asarray(values["level_scale"], dtype=np.float64),
            physical_factor_coefficients=np.asarray(
                values["physical_factor_coefficients"], dtype=np.float64
            ),
            ssm_input_mean=np.asarray(values["ssm_input_mean"], dtype=np.float64),
            ssm_input_scale=np.asarray(values["ssm_input_scale"], dtype=np.float64),
            ssm_terminal_hidden=np.asarray(
                values["ssm_terminal_hidden"], dtype=np.float64
            ),
            ssm_parameters=parameters,
        )


def interpolate_loading_expiries(
    loadings: FloatArray,
    reference_business_dtes: FloatArray,
    remaining_business_dtes: FloatArray,
) -> FloatArray:
    """Interpolate fixed-DTE loading fields onto active fixed expiries."""
    source = np.asarray(loadings, dtype=np.float64)
    reference = np.asarray(reference_business_dtes, dtype=np.float64)
    target = np.asarray(remaining_business_dtes, dtype=np.float64)
    if source.ndim != 3 or source.shape[0] != len(reference):
        raise ValueError("loading fields and reference expiries do not align")
    flattened = source.reshape(len(reference), -1)
    interpolated = np.stack(
        [
            np.interp(target, reference, flattened[:, coordinate])
            for coordinate in range(flattened.shape[1])
        ],
        axis=1,
    )
    return interpolated.reshape((len(target),) + source.shape[1:])


def floating_level_surface(
    active_dlvsq: Array,
    active_business_dtes: Array,
    reference_business_dtes: Array,
) -> Array:
    """Interpolate an aging fixed-expiry state back to the reference DTE grid."""
    log_levels = jnp.log(active_dlvsq)
    strike_major = jnp.moveaxis(log_levels, -2, -1)

    def interpolate_curve(curve: Array) -> Array:
        return jnp.interp(
            reference_business_dtes,
            active_business_dtes,
            curve,
            left=curve[0],
            right=curve[-1],
        )

    interpolated = jax.vmap(
        jax.vmap(interpolate_curve, in_axes=0, out_axes=0),
        in_axes=0,
        out_axes=0,
    )(strike_major)
    return jnp.moveaxis(interpolated, -1, -2)


def selective_factor_scales(
    *,
    dynamics: EstimatedDynamics,
    hidden: Array,
    floating_log_dlvsq: Array,
    previous_factor_score: Array,
    log_scale_bound: float = np.log(3.0),
) -> tuple[Array, Array]:
    """Advance the selective state and return bounded factor amplitudes."""
    flat_levels = floating_log_dlvsq.reshape(
        floating_log_dlvsq.shape[:-2] + (-1,)
    )
    level_scores = (
        flat_levels - jnp.asarray(dynamics.level_mean).reshape(-1)
    ) @ jnp.asarray(dynamics.level_basis)
    level_scores /= jnp.asarray(dynamics.level_scale)
    inputs = jnp.concatenate((level_scores, previous_factor_score), axis=-1)
    inputs = (inputs - jnp.asarray(dynamics.ssm_input_mean)) / jnp.asarray(
        dynamics.ssm_input_scale
    )
    parameters = {
        name: jnp.asarray(value) for name, value in dynamics.ssm_parameters.items()
    }
    decay = jax.nn.sigmoid(
        inputs @ parameters["decay_weight"] + parameters["decay_bias"]
    )
    candidate = jnp.tanh(
        inputs @ parameters["candidate_weight"] + parameters["candidate_bias"]
    )
    next_hidden = decay * hidden + (1.0 - decay) * candidate
    log_scales = jnp.clip(
        next_hidden @ parameters["output_weight"] + parameters["output_bias"],
        -log_scale_bound,
        log_scale_bound,
    )
    return next_hidden, jnp.exp(log_scales)
