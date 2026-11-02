"""Fit density-attenuated joint spot/DLV covariance without deattenuating returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class AttenuationFitConfig:
    """The fitted covariance is diag(a_t) B B' diag(a_t), jointly with spot."""

    density_floor: float = 1.e-4
    annualization: float = 252.

    def __post_init__(self) -> None:
        verify_inp(np.isfinite(self.density_floor) and self.density_floor >= 0.,
                   lambda: 'density_floor must be finite and nonnegative')
        verify_inp(np.isfinite(self.annualization) and self.annualization > 0.,
                   lambda: 'annualization must be finite and positive')


@dataclass(frozen=True, kw_only=True)
class AttenuationTrainConfig:
    """Full-batch Adam controls; checkpoints preserve moments and plateau clocks."""

    learning_rate: float = 1.e-2
    minimum_learning_rate: float = 1.e-6
    minimum_epochs: int = 200
    maximum_epochs: int = 5000
    plateau_epochs: int = 50
    floor_epochs: int = 100
    improvement_tolerance: float = 1.e-8
    gradient_tolerance: float = 1.e-8
    checkpoint_epochs: int = 25
    compile: bool = True

    def __post_init__(self) -> None:
        verify_inp(0. < self.minimum_learning_rate <= self.learning_rate,
                   lambda: 'learning rates must be positive and ordered')
        verify_inp(self.maximum_epochs >= self.minimum_epochs > 0,
                   lambda: 'epoch limits must be positive and ordered')
        verify_inp(min(self.plateau_epochs, self.floor_epochs, self.checkpoint_epochs) > 0,
                   lambda: 'plateau, floor and checkpoint intervals must be positive')
        verify_inp(self.improvement_tolerance >= 0. and self.gradient_tolerance >= 0.,
                   lambda: 'convergence tolerances must be nonnegative')


def joint_second_moment_score(parameters: jax.Array, rms_scaled_attenuation: jax.Array,
                              weighted_observations: jax.Array, spot_scores: jax.Array) -> jax.Array:
    """Weighted proper covariance score, excluding its forecast-independent term.

    Parameters use unit-variance node units. Node weights are 1/sqrt(N), and
    spot has unit weight. No realized innovation is divided by attenuation.
    The first joint loading row is fixed to e_1, identifying the spot factor.
    """
    nodes = parameters.shape[0]
    weighted_loadings = parameters / jnp.sqrt(float(nodes))
    gram = jnp.einsum('ti,if,ig->tfg', rms_scaled_attenuation**2,
                      weighted_loadings, weighted_loadings)
    gram = gram.at[:, 0, 0].add(1.)
    projected = (weighted_observations * rms_scaled_attenuation) @ weighted_loadings
    projected = projected.at[:, 0].add(spot_scores)
    return jnp.mean(jnp.sum(gram**2, axis=(-2, -1)) - 2. * jnp.sum(projected**2, axis=-1))


def prepare_estimation_arrays(*, innovations: np.ndarray, spot_scores: np.ndarray,
                              start_densities: np.ndarray, initial_loadings: np.ndarray,
                              config: AttenuationFitConfig) -> dict[str, np.ndarray]:
    """Center residuals and determine every scale from the supplied training set."""
    innovations = np.asarray(innovations, dtype=np.float64)
    mass = np.asarray(start_densities, dtype=np.float64)
    initial = np.asarray(initial_loadings, dtype=np.float64)
    scores = np.asarray(spot_scores, dtype=np.float64)
    if (innovations.ndim != 3 or mass.shape != innovations.shape
            or initial.shape[:-1] != innovations.shape[1:] or initial.shape[-1] < 1
            or scores.shape != (len(innovations),) or len(scores) < 3):
        raise ValueError('joint innovations, start densities, spot and initial loading geometry differ')
    if not all(np.isfinite(value).all() for value in (innovations, mass, initial, scores)) or np.any(mass < 0.):
        raise ValueError('estimation inputs must be finite and densities nonnegative')
    denominator = mass + config.density_floor * np.max(mass, axis=-1, keepdims=True)
    if config.density_floor == 0.:
        attenuation = np.ones_like(mass)
    else:
        if np.any(denominator <= 0.):
            raise ValueError('an entire interior maturity density is zero; attenuation is unidentified')
        attenuation = mass / denominator
    nodes = int(np.prod(innovations.shape[1:]))
    annual_residuals = innovations.reshape(len(scores), nodes) * np.sqrt(config.annualization)
    residual_mean = np.mean(annual_residuals, axis=0)
    centered = annual_residuals - residual_mean
    scales = np.sqrt(np.mean(centered**2, axis=0))
    score_mean, score_sd = float(scores.mean()), float(scores.std(ddof=0))
    if np.any(scales <= 0.) or score_sd <= 0.:
        raise ValueError('zero historical variance makes a row unidentified; no silent prior is substituted')
    flat = attenuation.reshape(len(scores), nodes)
    largest = np.max(flat, axis=0)
    if np.any(largest <= 0.):
        raise ValueError('a node is never active in training; its raw loading cannot be estimated')
    rms = largest * np.sqrt(np.mean((flat / largest)**2, axis=0))
    return {'rms_scaled_attenuation': flat / rms,
            'weighted_observations': centered / scales / np.sqrt(float(nodes)),
            'spot_scores': (scores - score_mean) / score_sd,
            'initial_parameters': initial.reshape(nodes, -1) / scales[:, None],
            'annual_residual_mean': residual_mean, 'annual_residual_sd': scales,
            'attenuation_rms': rms, 'attenuation': flat,
            'spot_mean': np.asarray(score_mean), 'spot_sd': np.asarray(score_sd),
            'node_shape': np.asarray(innovations.shape[1:], dtype=np.int64)}


def _atomic_checkpoint(path: Path, *, parameters: jax.Array, best: jax.Array,
                       optimizer_state: object, control: dict) -> None:
    leaves = jax.tree_util.tree_leaves(optimizer_state)
    temporary = path.with_suffix('.temporary.npz')
    np.savez(temporary, parameters=np.asarray(parameters), best=np.asarray(best),
             control_json=np.asarray(json.dumps(control)),
             **{f'optimizer_{index}': np.asarray(value) for index, value in enumerate(leaves)})
    os.replace(temporary, path)


def _fit_converged(control: dict, train: AttenuationTrainConfig) -> bool:
    return (control['epoch'] >= train.minimum_epochs and (
        control.get('gradient_norm', float('inf')) <= train.gradient_tolerance or (
            control['learning_rate'] <= train.minimum_learning_rate
            and control['floor_work_epochs'] >= train.floor_epochs
            and control['stale_epochs'] >= train.floor_epochs)))


def fit_joint_loadings(*, data: dict[str, np.ndarray], output: Path,
                       train: AttenuationTrainConfig, identity: str,
                       stop_requested: Callable[[], bool] = lambda: False,
                       resource_check: Callable[[], None] = lambda: None) -> dict:
    """Fit on CPU or an allocated backend; never report a budget exit as convergence."""
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'checkpoint.npz'
    score = lambda parameters: joint_second_moment_score(parameters,
        jnp.asarray(data['rms_scaled_attenuation']), jnp.asarray(data['weighted_observations']),
        jnp.asarray(data['spot_scores']))
    optimizer = optax.scale_by_adam()

    def update(parameters, state, rate):
        loss, gradient = jax.value_and_grad(score)(parameters)
        updates, next_state = optimizer.update(gradient, state, parameters)
        candidate = optax.apply_updates(parameters, jax.tree_util.tree_map(lambda value: -rate * value, updates))
        return candidate, next_state, loss, jnp.linalg.norm(gradient)

    update = jax.jit(update) if train.compile else update
    evaluate = jax.jit(jax.value_and_grad(score)) if train.compile else jax.value_and_grad(score)
    started = time.monotonic()
    with FileLock(str(target) + '.lock', timeout=0):
        parameters = jnp.asarray(data['initial_parameters'])
        state = optimizer.init(parameters)
        best = parameters
        initial_loss = float(score(parameters))
        control = {'identity': identity, 'epoch': 0, 'best_epoch': 0, 'best_loss': initial_loss,
                   'initial_loss': initial_loss, 'anchor_loss': initial_loss, 'stale_epochs': 0,
                   'floor_work_epochs': 0, 'learning_rate': train.learning_rate,
                   'train': asdict(train), 'status': 'running', 'elapsed_seconds': 0.}
        if target.is_file():
            with np.load(target, allow_pickle=False) as saved:
                control = json.loads(str(saved['control_json']))
                if control['identity'] != identity:
                    raise ValueError('checkpoint belongs to a different estimation problem')
                parameters, best = jnp.asarray(saved['parameters']), jnp.asarray(saved['best'])
                template, structure = jax.tree_util.tree_flatten(state)
                state = jax.tree_util.tree_unflatten(structure,
                    [jnp.asarray(saved[f'optimizer_{index}'], dtype=value.dtype) for index, value in enumerate(template)])
            previous_train = control['train']
            if previous_train['learning_rate'] != train.learning_rate:
                control.update(learning_rate=train.learning_rate, stale_epochs=0, floor_work_epochs=0)
            if previous_train['minimum_learning_rate'] != train.minimum_learning_rate:
                control.update(learning_rate=max(control['learning_rate'], train.minimum_learning_rate),
                               floor_work_epochs=0)
            if previous_train['improvement_tolerance'] != train.improvement_tolerance:
                control.update(anchor_loss=control.get('loss', control['best_loss']), stale_epochs=0)
            control['train'] = asdict(train)
            if control['status'] == 'converged' and _fit_converged(control, train):
                _atomic_checkpoint(target, parameters=parameters, best=best, optimizer_state=state, control=control)
                return control
            control['status'] = 'running'
        previous_elapsed = control['elapsed_seconds']
        last_resource, last_progress = started, started
        with (output / 'training.jsonl').open('a', buffering=1) as log:
            for epoch in range(control['epoch'] + 1, train.maximum_epochs + 1):
                rate = control['learning_rate']
                candidate, next_state, old_loss, gradient_norm = update(parameters, state, rate)
                candidate_loss, candidate_gradient = evaluate(candidate)
                loss, norm = float(candidate_loss), float(jnp.linalg.norm(candidate_gradient))
                if not np.isfinite(loss) or not np.isfinite(norm):
                    control['status'] = 'nonfinite_optimizer_update'
                    break
                parameters, state = candidate, next_state
                if loss < control['best_loss']:
                    best = parameters
                    control.update(best_loss=loss, best_epoch=epoch)
                tolerance = train.improvement_tolerance * max(1., abs(control['anchor_loss']))
                if loss < control['anchor_loss'] - tolerance:
                    control.update(anchor_loss=loss, stale_epochs=0)
                else:
                    control['stale_epochs'] += 1
                at_floor = rate <= train.minimum_learning_rate
                control['floor_work_epochs'] = control['floor_work_epochs'] + 1 if at_floor else 0
                if not at_floor and control['stale_epochs'] >= train.plateau_epochs:
                    control.update(learning_rate=max(train.minimum_learning_rate, .5 * rate), stale_epochs=0)
                control.update(epoch=epoch, loss=loss, gradient_norm=norm,
                               elapsed_seconds=previous_elapsed + time.monotonic() - started,
                               status='running')
                if _fit_converged(control, train):
                    control['status'] = 'converged'
                event = {'event': 'attenuation_fit_epoch',
                         **{key: value for key, value in control.items() if key != 'identity'}}
                log.write(json.dumps(event) + '\n')
                now = time.monotonic()
                if epoch == 1 or epoch % 25 == 0 or now - last_progress >= 60.:
                    print(json.dumps(event), flush=True)
                    last_progress = now
                if now - last_resource >= 60.:
                    resource_check()
                    last_resource = now
                if stop_requested():
                    control['status'] = 'stopped'
                if epoch % train.checkpoint_epochs == 0 or control['status'] != 'running':
                    _atomic_checkpoint(target, parameters=parameters, best=best, optimizer_state=state, control=control)
                if control['status'] != 'running':
                    break
            if control['status'] == 'running':
                control['status'] = 'budget_exhausted_not_converged'
            _atomic_checkpoint(target, parameters=parameters, best=best, optimizer_state=state, control=control)
        scales = data['annual_residual_sd'] / data['attenuation_rms']
        raw = np.asarray(best) * scales[:, None]
        if not np.isfinite(raw).all():
            raise FloatingPointError('estimated raw loading is not representable; no loading clipping is applied')
        loadings = raw.reshape(tuple(data['node_shape']) + (raw.shape[-1],))
        np.savez(output / 'estimated-loadings.npz', q_log_dlvsq_loadings=loadings,
                 **{name: data[name] for name in ('attenuation_rms', 'annual_residual_mean', 'annual_residual_sd',
                                                 'spot_mean', 'spot_sd')})
        write_json(output / 'fit-report.json', control)
        return control


def _load_training_data(calibration_root: Path, origin: str) -> tuple[dict, dict]:
    rolling = calibration_root / 'rolling' / origin
    source = rolling / 'ar1.npz'
    with np.load(source, allow_pickle=False) as saved:
        values = {name: np.asarray(saved[name]) for name in saved.files}
    dates = list(map(str, values['history_dates']))
    indices = {day: index for index, day in enumerate(dates)}
    densities, starting_dates, source_hashes = [], [], {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    for endpoint in map(str, values['pair_end_dates']):
        index = indices[endpoint]
        if index <= 0:
            raise ValueError('a historical pair has no preceding fitted state')
        start = dates[index - 1]
        path = rolling / 'history' / start / 'fit.npz'
        with np.load(path, allow_pickle=False) as state:
            densities.append(np.asarray(state['densities'], dtype=np.float64)[1:, 1:-1])
        starting_dates.append(start)
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'innovations': values['innovations'], 'spot_scores': values['spot_gaussian_scores'],
            'start_densities': np.asarray(densities), 'initial_loadings': values['q_log_dlvsq_loadings']}, {
                'source_sha256': source_hashes, 'start_dates': starting_dates,
                'endpoint_dates': list(map(str, values['pair_end_dates'])),
                'fixed_contract_clipped_query_fraction': float(values['fixed_contract_clipped_query_fraction'])}


def run_estimation(*, calibration_root: Path, origin: str, output: Path,
                   config: AttenuationFitConfig, train: AttenuationTrainConfig) -> dict:
    """Use existing fixed-contract AR residuals; never refit SANOS or historical DLVs."""
    check_resource_headroom()
    output.mkdir(parents=True, exist_ok=True)
    inputs, provenance = _load_training_data(calibration_root, origin)
    data = prepare_estimation_arrays(**inputs, config=config)
    identity_data = {'config': asdict(config), 'origin': origin, 'source_sha256': provenance['source_sha256'],
                     'version': AttenuationFitConfig.version.unique_id48,
                     'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity = json.dumps(identity_data, sort_keys=True)
    write_json(output / 'provenance.json', {**identity_data, **provenance,
        'estimator': 'proper weighted centered joint second-moment score; fixed spot loading e_1',
        'node_weight': '1 / (training population annual residual SD * sqrt(node count))',
        'no_observed_return_deattenuation': True, 'density_timing': 'start of historical pair',
        'interpretation': 'constant B in xi_t=D(q_t)B; not the original constant-xi PCA estimate',
        'historical_representation_refitted': False, 'training_pairs': len(inputs['innovations']),
        'historical_attenuation_quantiles': np.quantile(data['attenuation'], [0., .01, .5, .99, 1.]).tolist()})
    stopped = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopped.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        report = fit_joint_loadings(data=data, output=output / 'fit', train=train, identity=identity,
                                   stop_requested=lambda: stopped[0], resource_check=check_resource_headroom)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report.update(origin=origin, training_pairs=len(inputs['innovations']),
                  fitted_loading_shape=list(inputs['initial_loadings'].shape))
    with np.load(output / 'fit' / 'estimated-loadings.npz', allow_pickle=False) as saved:
        loadings = np.asarray(saved['q_log_dlvsq_loadings'])
    raw_norm = np.linalg.norm(loadings, axis=-1)
    effective_norm = raw_norm.reshape(1, -1) * data['attenuation']
    report.update(raw_loading_norm_quantiles=np.quantile(raw_norm, [0., .5, .9, .99, 1.]).tolist(),
                  effective_loading_norm_quantiles=np.quantile(effective_norm, [0., .5, .9, .99, 1.]).tolist())
    if report['status'] == 'converged':
        target = output / 'calibration'
        own = target / 'own_dates' / origin
        rolling = target / 'rolling' / origin
        own.mkdir(parents=True, exist_ok=True)
        rolling.mkdir(parents=True, exist_ok=True)
        for name in ('inputs.npz', 'fit.npz', 'report.json'):
            shutil.copy2(calibration_root / 'own_dates' / origin / name, own / name)
        with np.load(calibration_root / 'rolling' / origin / 'ar1.npz', allow_pickle=False) as saved:
            values = {name: np.asarray(saved[name]) for name in saved.files}
        values.update(q_log_dlvsq_loadings=loadings, attenuation_density_floor=np.asarray(config.density_floor),
                      attenuation_fit_identity=np.asarray(identity))
        np.savez(rolling / 'ar1.npz', **values)
        rolling_report = json.loads((calibration_root / 'rolling' / origin / 'report.json').read_text())
        rolling_report['attenuation_consistent_estimation'] = report
        rolling_report['diffusion_estimator'] = 'xi_t=D(q_t)B estimated jointly with Gaussianized spot'
        write_json(rolling / 'report.json', rolling_report)
        report['calibration_root'] = str(target)
    write_json(output / 'report.json', report)
    print(json.dumps({'event': 'attenuation_estimation_finished', **report}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--origin', default='2025-08-29')
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--density-floor', type=float, default=1.e-4)
    arguments = parser.parse_args()
    run_estimation(calibration_root=arguments.calibration_root, origin=arguments.origin,
                   output=arguments.output_directory, config=AttenuationFitConfig(density_floor=arguments.density_floor),
                   train=AttenuationTrainConfig())


if __name__ == '__main__':
    main()
