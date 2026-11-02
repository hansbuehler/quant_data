"""Retained fixed-contract paths for unconditional and local-action Q audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy.stats import norm

from quantdata.gromm2.dlv import SparseTransport, prepare_sparse_transports
from quantdata.gromm2.forward_hedging import load_model
from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.rolling_audit import (
    check_resource_headroom,
    moment_statistics,
    paired_samples,
)
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.runtime import configure_backend
from quantdata.gromm2.saved_model_audit import AUDIT_NMS
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step

PATH_HEALTH_FLAGS = {'nonfinite': 1, 'nonpositive_future_dlvsq': 2, 'negative_density': 4,
    'mass_or_forward_error': 8, 'nonpositive_spot': 16, 'float32_storage_overflow': 32}


@version("0.5")
@dataclass(frozen=True, kw_only=True)
class PathAuditConfig:
    """Original-Q, daily trading on a fixed reduced contract universe."""

    paths: int = 100_000
    batch_size: int = 256
    days: int = 125
    substeps: int = 8
    seed: int = 20261031
    density_loading_floor: float = 1.e-4
    drift_taming: bool = False
    retain_nonfinite: bool = False
    collect_dlv_variation: bool = False
    preserve_log_dlvsq: bool = True
    linear_solver: str = 'pcr_unsymmetric'
    standard_error_floor: float = 1.e-8
    nms: tuple[float, ...] = AUDIT_NMS

    def __post_init__(self) -> None:
        verify_inp(self.paths >= 4 and self.paths % 2 == 0, lambda: "paths must contain independent antithetic pairs")
        verify_inp(self.batch_size >= 2 and self.batch_size % 2 == 0, lambda: "batch_size must preserve antithetic pairs")
        verify_inp(1 <= self.days <= 125 and self.substeps in {1, 8}, lambda: "invalid horizon or substeps")
        verify_inp(0. <= self.density_loading_floor < 1. and self.standard_error_floor > 0., lambda: "invalid numerical floors")
        verify_inp(isinstance(self.drift_taming, bool) and isinstance(self.retain_nonfinite, bool), lambda: 'scheme switches must be boolean')
        verify_inp(isinstance(self.preserve_log_dlvsq, bool) and isinstance(self.collect_dlv_variation, bool),
                   lambda: 'state and measurement switches must be boolean')
        verify_inp(self.linear_solver in ('thomas', 'pcr', 'thomas_unsymmetric', 'pcr_unsymmetric'),
                   lambda: 'unknown SANOS linear solver')
        verify_inp(len(self.nms) > 0 and np.all(np.isfinite(self.nms)), lambda: "NMS strikes must be finite")


def audit_contracts(*, expiries_y: np.ndarray, reference_variances: np.ndarray,
                    config: PathAuditConfig) -> dict[str, np.ndarray]:
    """Freeze inception strikes and one OTM payoff side for every five-day expiry."""
    business_days = np.rint(np.asarray(expiries_y) * 252.).astype(np.int64)
    audit_days = np.arange(5, config.days + 1, 5, dtype=np.int64)
    if not len(audit_days):
        audit_days = np.asarray([config.days], dtype=np.int64)
    indices = np.searchsorted(business_days, audit_days)
    if np.any(indices == len(business_days)) or np.any(business_days[indices] != audit_days):
        raise ValueError("every audit expiry must be an existing model expiry")
    variances = np.asarray(reference_variances)
    if variances.shape != business_days.shape or np.any(~np.isfinite(variances)) or np.any(variances <= 0.):
        raise ValueError("reference ATM total variances must match the model clock")
    strikes = np.exp(np.sqrt(variances[indices])[:, None] * np.asarray(config.nms)[None])
    return {"expiry_days": audit_days, "model_indices": indices, "strikes": strikes, "is_call": strikes >= 1.}


def normal_block(*, config: PathAuditConfig, start: int, count: int, factors: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible path blocks with adjacent antithetic partners kept together."""
    generator = np.random.default_rng(np.random.SeedSequence([config.seed, start]))
    factor_half = generator.standard_normal((config.days, config.substeps, count // 2, factors))
    background_half = generator.standard_normal((config.days, count // 2))
    factor = np.stack((factor_half, -factor_half), axis=3).reshape(config.days, config.substeps, count, factors)
    background = np.stack((background_half, -background_half), axis=2).reshape(config.days, count)
    return factor, background


def path_function(*, arrays: dict[str, object], background_volatility: float,
                  contracts: dict[str, np.ndarray], config: PathAuditConfig):
    """Compile the whole path, skipping expired solves without changing live dynamics.

    States are observed before their next-day innovations. Gains are formed in
    float64 before optional float32 storage. Settled option values are retained
    as cash and never generate another option return.
    """
    supports = np.asarray(arrays["supports"], dtype=np.float64)
    dlvsq = np.asarray(arrays["dlvsq"], dtype=np.float64)
    loadings = np.asarray(arrays["loadings"], dtype=np.float64)
    dtes = np.rint(252. * np.asarray(arrays["expiries_y"])).astype(np.int64)
    if dtes[0] != 1 or dtes[-1] < config.days or loadings.shape[:2] != dlvsq[1:].shape:
        raise ValueError("saved model does not cover the requested path horizon")
    factors = loadings.shape[-1]
    maturities, nodes = supports.shape
    support_initial = jnp.asarray(supports)
    transport_initial = prepare_sparse_transports(np.asarray(arrays["transports"]))
    identity_transport = prepare_sparse_transports(np.eye(nodes, dtype=np.float64))
    interval_initial = jnp.asarray(np.diff(np.asarray(arrays["expiries_y"])))
    reference_days = np.asarray(arrays.get("loading_reference_days", dtes[1:]), dtype=np.float64)
    lower_indices, upper_indices, fractions = [], [], []
    for day in range(config.days):
        front = np.searchsorted(dtes, day, side="right")
        targets = np.pad(dtes[front + 1:] - day, (0, front), mode="edge") if front < maturities - 1 else np.full(maturities - 1, reference_days[0])
        upper = np.clip(np.searchsorted(reference_days, targets, side="right"), 1, len(reference_days) - 1)
        lower = upper - 1
        fraction = np.clip((targets - reference_days[lower]) / (reference_days[upper] - reference_days[lower]), 0., 1.)
        lower_indices.append(lower)
        upper_indices.append(upper)
        fractions.append(fraction)
    lower_indices, upper_indices, fractions = map(jnp.asarray, (np.stack(lower_indices), np.stack(upper_indices), np.stack(fractions)))
    base_field = jnp.asarray(loadings)
    projection, _ = np.linalg.qr(loadings.reshape(-1, factors))
    projection = jnp.asarray(np.concatenate((np.zeros((1, dlvsq.shape[1], factors)), projection.reshape(loadings.shape))))
    reference_log = jnp.log(jnp.asarray(dlvsq))
    ar1 = arrays.get("ar1_parameters")
    ar1 = None if ar1 is None else {name: jnp.asarray(value) for name, value in ar1.items()}
    option_days = jnp.asarray(contracts["expiry_days"])
    option_indices = jnp.asarray(contracts["model_indices"])
    option_strikes = jnp.asarray(contracts["strikes"])
    option_calls = jnp.asarray(contracts["is_call"])

    def prices(support, tower, background, spot, day, front):
        relative_indices = jnp.clip(option_indices - front, 0, maturities - 1)
        calls = price_active_calls(support[relative_indices], tower[:, relative_indices], background,
            jnp.maximum(option_days - day, 0) / 252., option_strikes,
            background_annual_volatility=background_volatility)
        options = jnp.where(option_calls, calls, calls - spot[:, None, None] + option_strikes)
        return jnp.concatenate((spot[:, None], options.reshape(len(spot), -1)), axis=-1)

    @jax.jit
    def simulate(normals, background_normals):
        paths = normals.shape[2]
        state = SimulationState(jnp.broadcast_to(jnp.asarray(arrays["densities"])[0], (paths, nodes)),
            jnp.broadcast_to(jnp.asarray(dlvsq), (paths,) + dlvsq.shape), jnp.ones(paths))
        tower = jnp.broadcast_to(jnp.asarray(arrays["densities"]), (paths,) + supports.shape)
        initial_prices = prices(support_initial, tower, state.background_level, jnp.ones(paths), jnp.asarray(0), jnp.asarray(0))
        initial_ar1 = jnp.zeros((paths, factors)) if ar1 is None else jnp.broadcast_to(ar1["terminal_log_squared"], (paths, factors))
        carry = (state, support_initial, transport_initial, interval_initial, jnp.asarray(0), initial_prices,
                 jnp.zeros((paths, factors)), initial_ar1)
        if config.preserve_log_dlvsq:
            carry += (jnp.log(state.dlvsq),)
        if config.collect_dlv_variation:
            carry += (jnp.zeros((paths, 5, maturities, nodes - 2), dtype=state.dlvsq.dtype),)

        def day_step(carry, inputs):
            state, support, transport, intervals, front, previous, cumulative, ar1_state = carry[:8]
            day, factor_normal, background_normal = inputs
            active_count = maturities - front
            if ar1 is None:
                factor_scales = jnp.ones((paths, factors))
            else:
                predicted = ar1["variance_intercept"] + ar1["variance_slope"] * ar1_state
                factor_scales = ar1["fixed_factor_scales"] * ar1["scale_calibration"] * jnp.exp(
                    jnp.clip(.5 * predicted, -ar1["log_scale_bound"], ar1["log_scale_bound"]))
            absolute_indices = jnp.clip(front + jnp.arange(maturities), 0, maturities - 1)
            live = jnp.arange(maturities) < active_count
            current_logs = carry[8] if config.preserve_log_dlvsq else jnp.log(state.dlvsq)
            centered = jnp.where(live[None, :, None], current_logs - reference_log[absolute_indices], 0.)
            scores = jnp.einsum("bjn,jnf->bf", centered, projection[absolute_indices])
            relative_prices = previous[:, 1:] / previous[:, :1]
            eligible = jnp.broadcast_to((option_days > day)[None], (paths, len(option_days)))
            features = jnp.concatenate((state.front_density, scores, jnp.where(live, jnp.mean(centered, axis=-1), 0.), cumulative,
                ar1_state, factor_scales, jnp.log(state.background_level)[:, None], jnp.log(previous[:, :1]),
                jnp.full((paths, 1), day / config.days), relative_prices, eligible), axis=-1)
            alpha = fractions[day, :, None, None]
            field = (1. - alpha) * base_field[lower_indices[day]] + alpha * base_field[upper_indices[day]]
            remaining = jnp.asarray(dtes)[front] - day
            result = one_day_risk_neutral_step(support, transport, intervals, state, field, factor_normal, background_normal,
                remaining_front_days=remaining, substeps=config.substeps, background_annual_volatility=background_volatility,
                density_loading_floor=config.density_loading_floor, linear_solver=config.linear_solver, factor_scales=factor_scales,
                active_maturity_count=active_count, drift_taming=config.drift_taming,
                collect_dlv_variation=config.collect_dlv_variation,
                log_dlvsq=current_logs if config.preserve_log_dlvsq else None)
            raw_prices = prices(support, result.density_tower, result.state.background_level, result.spot, day + 1, front)
            active_options = jnp.repeat(option_days > day, option_strikes.shape[-1])
            next_prices = jnp.concatenate((raw_prices[:, :1], jnp.where(active_options, raw_prices[:, 1:], previous[:, 1:])), axis=-1)
            gains = (next_prices - previous) / previous[:, :1]
            should_roll = (remaining == 1) & (active_count > 1)

            def roll():
                next_state = SimulationState(result.state.front_density,
                    jnp.concatenate((result.state.dlvsq[:, 1:], result.state.dlvsq[:, -1:]), axis=1), result.state.background_level)
                next_support = jnp.concatenate((support[1:], support[-1:]))
                next_transport = SparseTransport(jnp.concatenate((transport.indices[1:], identity_transport.indices[None])),
                    jnp.concatenate((transport.weights[1:], identity_transport.weights[None])))
                return next_state, next_support, next_transport, jnp.concatenate((intervals[1:], intervals[-1:])), front + 1

            next_state, next_support, next_transport, next_intervals, next_front = jax.lax.cond(should_roll, roll,
                lambda: (result.state, support, transport, intervals, front))
            next_ar1 = ar1_state if ar1 is None else jnp.log(
                (factor_scales * result.daily_factor_score / ar1["fixed_factor_scales"])**2 + ar1["squared_score_floor"])
            live_tower = jnp.arange(maturities) < active_count
            mass_error = jnp.max(jnp.where(live_tower, jnp.abs(result.density_tower.sum(axis=-1) - 1.), 0.))
            mean_error = jnp.max(jnp.where(live_tower, jnp.abs(jnp.sum(result.density_tower * support, axis=-1) - result.latent_spot[:, None]), 0.))
            diagnostic = jnp.stack((result.drift_taming_fraction, mass_error, mean_error, jnp.min(result.density_tower),
                                    jnp.min(result.state.dlvsq), jnp.max(result.state.dlvsq)))
            live_future = (jnp.arange(maturities) > 0) & live_tower
            if config.preserve_log_dlvsq:
                underflow_count = jnp.sum(live_future[None, :, None] & (result.state.dlvsq == 0.)
                                         & jnp.isfinite(result.log_dlvsq))
                diagnostic = jnp.concatenate((diagnostic, underflow_count[None]))
            if config.retain_nonfinite:
                future_finite = jnp.all(jnp.where(live_future[None, :, None], jnp.isfinite(result.state.dlvsq), True), axis=(1, 2))
                finite = jnp.all(jnp.isfinite(features), axis=1) & jnp.all(jnp.isfinite(gains), axis=1)
                finite &= jnp.all(jnp.isfinite(next_prices), axis=1) & future_finite
                if config.preserve_log_dlvsq:
                    positive = jnp.all(jnp.where(live_future[None, :, None],
                        jnp.isfinite(result.log_dlvsq) & (result.state.dlvsq >= 0.), True), axis=(1, 2))
                else:
                    positive = jnp.all(jnp.where(live_future[None, :, None], result.state.dlvsq > 0., True), axis=(1, 2))
                density_valid = jnp.all(jnp.where(live_tower[None, :, None], result.density_tower >= -1.e-12, True), axis=(1, 2))
                masses = jnp.sum(result.density_tower, axis=-1)
                forwards = jnp.sum(result.density_tower * support, axis=-1)
                invariant = jnp.all(jnp.where(live_tower[None],
                    (jnp.abs(masses - 1.) <= 1.e-10) & (jnp.abs(forwards - result.latent_spot[:, None])
                        <= 1.e-10 * (1. + jnp.abs(result.latent_spot[:, None]))), True), axis=1)
                storage_limit = jnp.finfo(jnp.float32).max
                storage = jnp.all(jnp.abs(features) <= storage_limit, axis=1)
                storage &= jnp.all(jnp.abs(gains) <= storage_limit, axis=1) & jnp.all(jnp.abs(next_prices) <= storage_limit, axis=1)
                health = ((~finite).astype(jnp.uint8) | ((~positive).astype(jnp.uint8) << 1)
                    | ((~density_valid).astype(jnp.uint8) << 2) | ((~invariant).astype(jnp.uint8) << 3)
                    | ((result.spot <= 0.).astype(jnp.uint8) << 4) | ((~storage).astype(jnp.uint8) << 5))
            next_carry = (next_state, next_support, next_transport, next_intervals, next_front, next_prices,
                          cumulative + result.daily_factor_score / jnp.sqrt(252.), next_ar1)
            if config.preserve_log_dlvsq:
                next_logs = jax.lax.cond(should_roll,
                    lambda: jnp.concatenate((result.log_dlvsq[:, 1:], result.log_dlvsq[:, -1:]), axis=1),
                    lambda: result.log_dlvsq)
                next_carry += (next_logs,)
            if config.collect_dlv_variation:
                totals = carry[8 + int(config.preserve_log_dlvsq)]
                live_future = jnp.arange(maturities - 1) + 1 < active_count
                absolute_future = jnp.minimum(front + 1 + jnp.arange(maturities - 1), maturities - 1)
                addition = jnp.where(live_future[None, None, :, None], result.dlv_variation, 0.)
                totals = totals.at[:, :4, absolute_future, :].add(addition[:, :4])
                totals = totals.at[:, 4, absolute_future, :].max(addition[:, 4])
                next_carry += (totals,)
            values = (features.astype(jnp.float32), gains, next_prices[:, 0], diagnostic)
            return next_carry, (*values, health) if config.retain_nonfinite else values

        final, values = jax.lax.scan(day_step, carry,
            (jnp.arange(config.days), normals, background_normals))
        features, gains, spots, diagnostics = values[:4]
        output = (jnp.swapaxes(features, 0, 1), jnp.swapaxes(gains, 0, 1),
                jnp.concatenate((jnp.ones((paths, 1)), spots.T), axis=1), final[5] - initial_prices, diagnostics)
        if config.retain_nonfinite:
            output += (jnp.swapaxes(values[4], 0, 1),)
        if config.collect_dlv_variation:
            totals = final[8 + int(config.preserve_log_dlvsq)]
            pairs = .5 * (totals[::2] + totals[1::2])
            difference = pairs[:, 0] - pairs[:, 1]
            summary = jnp.concatenate((jnp.sum(pairs[:, :4], axis=0),
                jnp.sum(difference**2, axis=0)[None], jnp.max(totals[:, 4], axis=0)[None]), axis=0)
            output += (summary,)
        return output

    return simulate


def summarize_statistics(statistics: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, object]:
    """Report raw z values and an explicitly non-Gaussian economic floor score."""
    z = np.abs(statistics["z"][selected])
    floor_score = np.abs(statistics["economic_floor_score"][selected])
    familywise = float(norm.ppf(1. - .01 / (2. * len(z))))
    percentile = float(np.quantile(z, .99, method='higher'))
    maximum = float(np.max(z))
    return {"tested_coordinates": len(z), "99_percentile_abs_z": percentile if np.isfinite(percentile) else 'infinity',
            "maximum_abs_z": maximum if np.isfinite(maximum) else 'infinity',
            "nonzero_mean_zero_sample_variance_coordinates": int(np.sum(~np.isfinite(z))),
            "pointwise_99_threshold": float(norm.ppf(.995)),
            "pointwise_99_breaches": int(np.sum(z > norm.ppf(.995))), "familywise_99_threshold": familywise,
            "familywise_99_breaches": int(np.sum(z > familywise)),
            "maximum_absolute_mean_gain_over_spot": float(np.max(np.abs(statistics["mean_gain_over_spot"][selected]))),
            "maximum_absolute_economic_floor_score": float(np.max(floor_score)),
            "99_percentile_absolute_economic_floor_score": float(np.quantile(floor_score, .99))}


def generate_audit_paths(*, arrays: dict[str, object], metadata: dict[str, object], reference_variances: np.ndarray,
                         output_directory: Path, config: PathAuditConfig) -> dict[str, object]:
    """Checkpoint complete path batches; retain all predictable states and gains."""
    check_resource_headroom()
    output_directory.mkdir(parents=True, exist_ok=True)
    contracts = audit_contracts(expiries_y=arrays["expiries_y"], reference_variances=reference_variances, config=config)
    identity = {"schema": "gromm2-local-action-paths-v1", "config": asdict(config),
                "code_version": PathAuditConfig.version.unique_id48, "model": metadata,
                "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('path_audit.py', 'simulation.py', 'dlv.py', 'bridge.py', 'pricing.py', 'tridiagonal.py', 'geometry.py')},
                "reference_variances_sha256": hashlib.sha256(np.asarray(reference_variances).tobytes()).hexdigest()}
    identity_json = json.dumps(identity, sort_keys=True)
    simulate = path_function(arrays=arrays, background_volatility=float(metadata["background_volatility"]), contracts=contracts, config=config)
    stop = threading.Event()
    done = threading.Event()
    started = time.monotonic()
    progress = {"stage": "initializing", "completed_paths": 0, "total_paths": config.paths}

    def heartbeat():
        while not done.wait(60.):
            print(json.dumps({"event": "path_audit_progress", **progress, "elapsed_seconds": time.monotonic() - started}), flush=True)
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    threading.Thread(target=heartbeat, daemon=True).start()
    path = output_directory / 'paths.h5'
    try:
        with FileLock(str(path) + '.lock', timeout=0), h5py.File(path, 'a') as stream:
            if 'identity' in stream.attrs and stream.attrs['identity'] != identity_json:
                raise ValueError("retained audit paths have a different model or sampling identity")
            if 'identity' not in stream.attrs:
                stream.attrs['identity'] = identity_json
                stream.attrs['completed_paths'] = 0
                for name, value in contracts.items():
                    stream.create_dataset(name, data=value)
            completed = int(stream.attrs['completed_paths'])
            if completed % 2 or (completed % config.batch_size and completed != config.paths):
                raise RuntimeError("path checkpoint is not a committed complete batch")
            for start in range(completed, config.paths, config.batch_size):
                if stop.is_set():
                    raise InterruptedError("path audit stopped at the last committed antithetic batch")
                count = min(config.batch_size, config.paths - start)
                progress.update(stage='compiled_path_simulation', completed_paths=start)
                normals, background = normal_block(config=config, start=start, count=count, factors=np.asarray(arrays['loadings']).shape[-1])
                output = simulate(jnp.asarray(normals), jnp.asarray(background))
                values = jax.device_get(output)
                features, gains, spots, terminal, diagnostics = values[:5]
                health = values[5] if config.retain_nonfinite else np.zeros((count, config.days), dtype=np.uint8)
                variation = values[6 if config.retain_nonfinite else 5] if config.collect_dlv_variation else None
                valid = np.isfinite(features).all(axis=(1, 2)) & np.isfinite(gains).all(axis=(1, 2))
                valid &= np.isfinite(spots).all(axis=1) & (spots > 0.).all(axis=1) & np.isfinite(terminal).all(axis=1)
                valid &= ~np.any(health, axis=1)
                for stored in (gains, spots, terminal):
                    valid &= np.all(np.abs(stored).reshape(count, -1) <= np.finfo(np.float32).max, axis=1)
                diagnostic_valid = bool(np.isfinite(diagnostics).all() and np.min(diagnostics[:, 3]) >= -1.e-12
                    and np.max(diagnostics[:, 1]) <= 1.e-10)
                if not config.retain_nonfinite and (not np.all(valid) or not diagnostic_valid):
                    raise FloatingPointError(f"non-finite retained path batch at {start}")
                progress['stage'] = 'saving_path_batch'
                daily_pairs, terminal_pairs = paired_samples(gains), paired_samples(terminal)
                if 'features' not in stream:
                    stream.create_dataset('path_validity', shape=(config.paths,), dtype='bool')
                    stream.create_dataset('path_health', shape=(config.paths, config.days), dtype='u1')
                    stream.create_dataset('first_invalid_day', shape=(config.paths,), dtype='i2')
                    for name, values in [('features', features), ('returns', gains), ('spot', spots), ('terminal_gains', terminal)]:
                        trailing = values.shape[1:]
                        chunks = (min(128, config.paths),) + trailing
                        stream.create_dataset(name, shape=(config.paths,) + trailing, dtype='f4', chunks=chunks, compression='lzf')
                    batches = (config.paths + config.batch_size - 1) // config.batch_size
                    for name, shape in [('daily_pair_moments', gains.shape[1:]), ('terminal_pair_moments', terminal.shape[1:])]:
                        stream.create_dataset(name, shape=(batches, 2) + shape, dtype='f8')
                    stream.create_dataset('batch_diagnostics', shape=(batches, config.days, diagnostics.shape[-1]), dtype='f8')
                    stream.create_dataset('batch_diagnostic_validity', shape=(batches,), dtype='bool')
                    if config.collect_dlv_variation:
                        stream.create_dataset('dlv_variation_moments', shape=(batches,) + variation.shape, dtype='f8')
                        stream['dlv_variation_moments'].attrs['fields'] = json.dumps([
                            'pair_sum_integrated_noise_squared', 'pair_sum_input_quadratic_variation',
                            'pair_sum_daily_log_change', 'pair_sum_daily_log_change_squared',
                            'pair_sum_variance_difference_squared', 'maximum_log_update_identity_error'])
                        stream.create_dataset('dlv_model_expiry_days', data=np.rint(np.asarray(arrays['expiries_y']) * 252.).astype(int))
                for name, values in [('features', features), ('returns', gains), ('spot', spots), ('terminal_gains', terminal)]:
                    stream[name][start:start + count] = values
                chunk = start // config.batch_size
                stream['path_validity'][start:start + count] = valid
                stream['path_health'][start:start + count] = health
                stream['first_invalid_day'][start:start + count] = np.where(np.any(health, axis=1),
                    np.argmax(health != 0, axis=1), -1)
                stream['batch_diagnostic_validity'][chunk] = diagnostic_valid
                for name, pairs in [('daily_pair_moments', daily_pairs), ('terminal_pair_moments', terminal_pairs)]:
                    pair_mean = pairs.mean(axis=0)
                    stream[name][chunk] = np.stack((pair_mean, ((pairs - pair_mean)**2).sum(axis=0)))
                stream['batch_diagnostics'][chunk] = diagnostics
                if config.collect_dlv_variation:
                    stream['dlv_variation_moments'][chunk] = variation
                stream.flush()
                stream.attrs.modify('completed_paths', start + count)
                stream.flush()
                elapsed = time.monotonic() - started
                rate = (start + count - completed) / max(elapsed, 1.e-9)
                progress.update(completed_paths=start + count, stage='batch_committed',
                                remaining_seconds=(config.paths - start - count) / max(rate, 1.e-9))
                print(json.dumps({'event': 'path_batch_committed', **progress, 'elapsed_seconds': elapsed,
                                  'invalid_paths_in_batch': int(np.sum(~valid)),
                                  'max_mass_error': float(np.max(diagnostics[:, 1])) if diagnostic_valid else None}), flush=True)
            result = {'identity': identity, 'paths': config.paths, 'independent_pairs': config.paths // 2,
                      'measure_preconditioning': False, 'return_definition': 'fixed-contract gain divided by pre-trade spot',
                      'conditional_tests': 'pending dedicated cross-fitted local entropy and squared actions'}
            invalid_count = int(np.count_nonzero(~stream['path_validity'][:]))
            invalid_batches = int(np.count_nonzero(~stream['batch_diagnostic_validity'][:]))
            result['stability'] = {'invalid_retained_paths': invalid_count,
                'batches_with_invalid_density_or_diagnostics': invalid_batches,
                'paths_dropped': 0, 'all_paths_retained': True,
                'health_flag_bits': PATH_HEALTH_FLAGS,
                'paths_by_failure_flag': {name: int(np.count_nonzero(np.any(stream['path_health'][:] & bit, axis=1)))
                    for name, bit in PATH_HEALTH_FLAGS.items()},
                'mass_absolute_tolerance': 1.e-10, 'forward_tolerance': '1e-10 * (1 + abs(latent_spot))'}
            if invalid_count or invalid_batches:
                result['status'] = 'numerically_invalid_population; no martingale inference'
                result['conditional_tests'] = 'not defined on this complete population; invalid paths were not discarded'
                write_json(output_directory / 'unconditional_report.json', result)
                return result
            result['status'] = 'numerically_valid_population; statistical tests required'
            saved = {}
            for label, dataset_name in [('daily', 'daily_pair_moments'), ('option_to_expiry', 'terminal_pair_moments')]:
                moments = stream[dataset_name]
                mean, squares = np.zeros(moments.shape[2:]), np.zeros(moments.shape[2:])
                total = 0
                for chunk in range(len(moments)):
                    pairs = min(config.batch_size, config.paths - chunk * config.batch_size) // 2
                    pair_mean, pair_squares = moments[chunk]
                    difference = pair_mean - mean
                    squares += pair_squares + difference**2 * total * pairs / (total + pairs)
                    mean += difference * pairs / (total + pairs)
                    total += pairs
                statistics = moment_statistics(total, mean, squares, epsilon=config.standard_error_floor)
                for name, values in statistics.items():
                    saved[f'{label}_{name}'] = values
                selected = np.ones_like(mean, dtype=bool)
                if label == 'daily':
                    selected[:, 1:] = np.repeat(contracts['expiry_days'][None] > np.arange(config.days)[:, None], len(config.nms), axis=-1)
                result[label] = summarize_statistics(statistics, selected)
            np.savez(output_directory / 'unconditional_statistics.npz', **saved)
            write_json(output_directory / 'unconditional_report.json', result)
            return result
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--history', type=Path)
    models.add_argument('--calibration-root', type=Path)
    parser.add_argument('--estimate', type=Path)
    parser.add_argument('--factor-law', type=Path)
    parser.add_argument('--reference-inputs', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--origin', default='2025-08-29')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--substeps', type=int, choices=(1, 8), required=True)
    parser.add_argument('--collect-dlv-variation', action='store_true')
    parser.add_argument('--level-dlv-state', action='store_true')
    parser.add_argument('--paths', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--days', type=int, default=125)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--density-loading-floor', type=float, default=1.e-4)
    parser.add_argument('--legacy-drift-taming', action='store_true')
    parser.add_argument('--retain-nonfinite', action='store_true')
    arguments = parser.parse_args()
    configure_backend(arguments.backend)
    if arguments.history is not None:
        if any(value is None for value in (arguments.estimate, arguments.factor_law, arguments.reference_inputs)):
            parser.error('legacy saved models require --estimate, --factor-law and --reference-inputs')
        arrays, metadata = load_legacy_ar1_model(history=arguments.history, estimate=arguments.estimate,
            factor_law=arguments.factor_law, origin=arguments.origin, backend=arguments.backend)
    else:
        arrays, metadata = load_model(arguments.calibration_root, arguments.origin)
        report = json.loads((arguments.calibration_root / 'rolling' / arguments.origin / 'report.json').read_text())
        if report['history_end'] >= arguments.origin or np.asarray(arrays['loadings']).shape[-1] != 16:
            raise ValueError('the rolling model must have a causal historical window and exactly 16 Q factors')
        metadata.update(model_label=report['model_label'], historical_calibration=report['historical_calibration'],
            history_start=report['history_start'], history_end=report['history_end'],
            diffusion_law='saved constant joint innovation loading field; physical AR(1) drift is not used under Q')
        if arguments.reference_inputs is None:
            arguments.reference_inputs = arguments.calibration_root / 'own_dates' / arguments.origin / 'inputs.npz'
    with np.load(arguments.reference_inputs, allow_pickle=False) as saved:
        reference_variances = np.asarray(saved['projected_total_variances'])
        reference_expiries = np.asarray(saved['model_expiries_y'])
    np.testing.assert_array_equal(np.rint(reference_expiries * 252.), np.rint(np.asarray(arrays['expiries_y']) * 252.))
    config = PathAuditConfig(paths=arguments.paths, batch_size=arguments.batch_size,
        days=arguments.days, substeps=arguments.substeps, seed=arguments.seed,
        collect_dlv_variation=arguments.collect_dlv_variation,
        preserve_log_dlvsq=not arguments.level_dlv_state,
        density_loading_floor=arguments.density_loading_floor,
        drift_taming=arguments.legacy_drift_taming, retain_nonfinite=arguments.retain_nonfinite)
    print(json.dumps({'event': 'path_audit_preflight', 'backend': jax.default_backend(), 'model': metadata,
                      'config': asdict(config)}), flush=True)
    generate_audit_paths(arrays=arrays, metadata=metadata, reference_variances=reference_variances,
                         output_directory=arguments.output_directory, config=config)


if __name__ == '__main__':
    main()
