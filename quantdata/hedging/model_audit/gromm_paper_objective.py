"""Published GROMM quote objective on the requested GROMM2 geometry."""

from __future__ import annotations

import numpy as np

PAPER_OBJECTIVE = {
    "schema": "gromm-paper-eq20-cap-v1",
    "targets": "observed bid and ask; observed midpoint only for the centering term",
    "quote_loss": "linear normalized excess + 0.05 absolute normalized midpoint distance",
    "midpoint_weight": .05,
    "cap_weight": 1.,
    "cap_log_width": .10,
    "cap_log_std": .35,
    "retained_quotes": "equal quote weight inside GROMM2 report band; equal total weight per fitted view",
    "strike_curvature_weight": 0.,
    "expiry_smoothing_weight": 0.,
    "half_spread_floor": None,
    "cap_coordinates": "active fitted DLV controls, excluding the unquoted end-view dummy surface",
}


def observed_view(view, geometry, day: dict):
    """Keep exact geometry/kernels while replacing only the calibration target and weights."""
    from dataclasses import replace

    from openqdata.models.gromm import calibrate
    from openqdata.models.gromm.config import DAY

    grids = calibrate.view_grids(geometry, view.kind, view.ratio)
    times = DAY * np.arange(geometry.config.model_days + 1)
    last = geometry.config.model_days - (view.kind == "end")
    tte, strikes = day["tte"], day["k"]
    later = np.clip(np.searchsorted(times, tte - calibrate.TOL), 1, geometry.config.model_days)
    keep = (tte >= DAY - calibrate.TOL) & (tte <= last * DAY + calibrate.TOL)
    keep &= (strikes > grids[later - 1, 0]) & (strikes < grids[later - 1, -1])
    if not np.array_equal(tte[keep], view.tte) or not np.array_equal(strikes[keep], view.k):
        raise ValueError("Observed quotes do not exactly match the prepared calibration view")
    half = .5 * (day["ask"][keep] - day["bid"][keep])
    if not np.isfinite(half).all() or np.any(half <= 0.) or not np.array_equal(half, view.hs):
        raise ValueError("Observed calibration half-spreads are invalid or changed")
    midpoint = .5 * (day["ask"][keep] + day["bid"][keep])
    return replace(view, sanos=midpoint, weight=np.asarray(view.report, dtype=np.float64))


def paper_view_loss(x, geo, blocks, active, smooth_t, q0, k0, k1, j0, j1, lam, midpoint, hs, weight):
    """Equation (20), plus the paper's restoring soft barrier around the raw DLV cap."""
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm import calibrate

    prices = calibrate._view_prices(geo, blocks, active, q0, x, k0, k1, j0, j1, lam)
    divisor = jnp.where(weight > 0., hs, 1.)
    distance = (prices - midpoint) / divisor
    quote = jnp.sum(weight * (jax.nn.relu(jnp.abs(distance) - 1.) + .05 * jnp.abs(distance))) / jnp.maximum(weight.sum(), 1.)
    maximum = jnp.log(calibrate.SIGMA_CAP - calibrate.SIGMA_FLOOR)
    barrier = .10 / .35 * jax.nn.softplus((x - maximum) / .10)
    live_rows = jnp.concatenate([jnp.ones(1), smooth_t])
    cap = .5 * jnp.sum(jnp.where(active[None] & (live_rows[:, None] > 0.), barrier**2, 0.))
    return quote + cap


def install_paper_objective() -> None:
    """Install the explicit versioned loss without modifying the external source repository."""
    import jax
    from openqdata.models.gromm import calibrate

    calibrate._view_loss = paper_view_loss
    calibrate._loss_v = jax.vmap(paper_view_loss, in_axes=calibrate._IN_AXES)
    calibrate._grad = jax.value_and_grad(calibrate._total, has_aux=True)
    calibrate._adam_step.clear_cache()
