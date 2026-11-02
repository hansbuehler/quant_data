"""Fresh, streaming martingale screens for immutable terminal-trained actions."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
from filelock import FileLock
from scipy.special import logsumexp
from scipy.stats import norm

from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    frozen_score_statistics,
)
from quantdata.gromm2.forward_hedging import load_model
from quantdata.gromm2.path_audit import (
    PathAuditConfig,
    audit_contracts,
    normal_block,
    path_function,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom, paired_samples
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.runtime import configure_backend
from quantdata.gromm2.terminal_audit import validated_evaluation_plan
from quantdata.torch import BaseEnvironment

ENDPOINTS = tuple(f'fold-{fold}/{name}' for fold in range(2) for name in DIAGNOSTICS) + tuple(f'ensemble/{name}' for name in DIAGNOSTICS)
SCORE_FIELDS = ('mean_daily_gain', 'mean_daily_squared_action', 'log_mean_daily_exponential_loss', 'mean_daily_squared_utility')


@dataclass(frozen=True, kw_only=True)
class FrozenTerminalAction:
    model: AffineDeepAction
    mean: torch.Tensor
    scale: torch.Tensor
    action_scale: float
    squared_lambda: float


def load_terminal_actions(manifest: Path, *, device: str) -> tuple[dict, dict[str, FrozenTerminalAction]]:
    """Load only hashed terminal weights and their bound training normalization."""
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()):
        raise ValueError('terminal evaluation requires an explicit CPU or managed CUDA device')
    frozen = json.loads(manifest.read_text())
    if frozen['schema'] != 'gromm2-terminal-frozen-actions-v2':
        raise ValueError('unknown frozen terminal-action schema')
    plan_path = Path(frozen['evaluation_plan'])
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != frozen['evaluation_plan_sha256']:
        raise ValueError('immutable confirmation plan changed')
    plan = validated_evaluation_plan(plan_path)
    if (frozen['semantic_sha256'] != plan['semantic_sha256']
        or frozen['protocol_sha256'] != plan['protocol_sha256']
        or Path(frozen['protocol']).resolve() != Path(plan['protocol']).resolve()
        or frozen['evaluation_settings'] != plan['variants'][frozen['evaluation_variant']]):
        raise ValueError('frozen confirmation declaration changed')
    if [fold['fold'] for fold in frozen['folds']] != [0, 1]:
        raise ValueError('both ordered fitted folds are required')
    actions = {}
    for fold in frozen['folds']:
        for name in DIAGNOSTICS:
            selected = fold['policies'][name]
            values = {}
            for field in ('weights', 'normalization'):
                content = Path(selected[field]).read_bytes()
                if hashlib.sha256(content).hexdigest() != selected[field + '_sha256']:
                    raise ValueError('immutable frozen action artifact differs from its manifest')
                values[field] = content
            saved = torch.load(io.BytesIO(values['weights']), map_location='cpu', weights_only=False)
            fields = dict(saved['model_config'])
            fields['network'] = ActionNetworkConfig(**fields['network'])
            config = LocalActionConfig(**fields)
            if (saved['code_version'] != LocalActionConfig.version.unique_id48
                or saved['user_version'] != config.user_version or saved['diagnostic'] != name
                or saved['policy_state_source'] != 'terminal trained model, not best_state'):
                raise ValueError('terminal action checkpoint version or interpretation changed')
            floating = [value.dtype for value in saved['state_dict'].values() if value.is_floating_point()]
            if not floating or len(set(floating)) != 1 or floating[0] not in (torch.float32, torch.float64):
                raise ValueError('unsupported or mixed frozen parameter precision')
            env = BaseEnvironment(device=device, precision=32 if floating[0] == torch.float32 else 64)
            normalization_dtype = saved.get('input_normalization_dtype', str(floating[0]).removeprefix('torch.'))
            if normalization_dtype not in ('float32', 'float64'):
                raise ValueError('unsupported frozen input normalization precision')
            normalization_dtype = getattr(torch, normalization_dtype)
            model = AffineDeepAction(env, config=config)
            model.load_state_dict(saved['state_dict'], strict=True)
            model.affine_only = False
            model.eval()
            with np.load(io.BytesIO(values['normalization']), allow_pickle=False) as normalization:
                if float(normalization['action_scale']) != config.action_scale:
                    raise ValueError('frozen original-unit action scale changed')
                mean = np.asarray(normalization['mean'])
                scale = np.asarray(normalization['scale'])
                if mean.shape != (config.feature_count,) or scale.shape != mean.shape or not np.isfinite([mean, scale]).all() or np.any(scale <= 0.):
                    raise ValueError('invalid frozen feature normalization')
                actions[f'fold-{fold["fold"]}/{name}'] = FrozenTerminalAction(model=model,
                    mean=torch.as_tensor(mean, dtype=normalization_dtype, device=env.device),
                    scale=torch.as_tensor(scale, dtype=normalization_dtype, device=env.device),
                    action_scale=config.action_scale, squared_lambda=config.squared_lambda)
    if len({action.squared_lambda for action in actions.values()}) != 1:
        raise ValueError('ensemble objectives require the same declared squared lambda')
    return frozen, actions


def score_terminal_batch(*, features: np.ndarray, gains: np.ndarray, eligible: np.ndarray,
                         actions: dict[str, FrozenTerminalAction]) -> np.ndarray:
    """Score unpenalized gains; normalize losses only in their separate fields.

    The float32 feature/gain view matches the original retained audit arrays.
    Original-unit actions follow the fitted policy precision; dot products,
    squared norms and log-domain utility aggregation use float64. Ensemble
    holdings are averaged before nonlinear losses or squared norms are formed.
    """
    features, gains = np.asarray(features, dtype=np.float32), np.asarray(gains, dtype=np.float32)
    if features.ndim != 3 or gains.ndim != 3 or features.shape[:2] != gains.shape[:2] or eligible.shape != gains.shape[1:]:
        raise ValueError('state, daily gain and eligibility shapes disagree')
    if not np.isfinite(features).all() or not np.isfinite(gains).all() or set(actions) != set(ENDPOINTS[:4]):
        raise ValueError('nonfinite audit data or incomplete frozen actions')
    first = next(iter(actions.values()))
    device = first.mean.device
    mask = torch.as_tensor(eligible, dtype=torch.bool, device=device)
    observed = torch.as_tensor(features, dtype=first.mean.dtype, device=device)
    returns = torch.as_tensor(gains, dtype=torch.float64, device=device)
    holdings, records = {}, []
    with torch.no_grad():
        for name, action in actions.items():
            if action.mean.device != device:
                raise ValueError('frozen actions must share the evaluation device')
            states = (observed.to(action.mean.dtype) - action.mean) / action.scale
            holdings[name] = (action.action_scale * action.model(states.to(action.model.fdtype), mask)).to(torch.float64)
        for name in DIAGNOSTICS:
            holdings[f'ensemble/{name}'] = .5 * (holdings[f'fold-0/{name}'] + holdings[f'fold-1/{name}'])
        squared_lambda = first.squared_lambda
        for name in ENDPOINTS:
            action = holdings[name]
            gain = torch.sum(action * returns, dim=-1)
            norm_squared = torch.sum(action.square(), dim=-1)
            exponent = -gain + .5 * norm_squared
            records.append(torch.stack((gain.mean(dim=-1), norm_squared.mean(dim=-1),
                torch.logsumexp(exponent, dim=-1) - math.log(gains.shape[1]),
                (gain - .5 * norm_squared - .5 * squared_lambda * gain.square()).mean(dim=-1)), dim=-1))
        return torch.stack(records, dim=1).cpu().numpy()


def summarize_terminal_scores(scores: np.ndarray, *, family_size: int = 12) -> dict:
    """Pair-clustered two-sided screens; no validity filter or economic floor."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1:] != (len(ENDPOINTS), len(SCORE_FIELDS)) or scores.shape[0] < 4 or scores.shape[0] % 2:
        raise ValueError('expected complete antithetic path scores for all endpoints')
    if not np.isfinite(scores).all() or family_size < len(ENDPOINTS):
        raise ValueError('nonfinite complete scores or invalid multiplicity family')
    result = {}
    for index, name in enumerate(ENDPOINTS):
        values = scores[:, index]
        statistics = frozen_score_statistics(values[:, 0])
        statistic = statistics['action_conditional_statistic']
        probability = float(2. * norm.sf(abs(statistic))) if statistic is not None else None
        pairs = paired_samples(values[:, 0])
        deviation_squared = (pairs - pairs.mean())**2
        total = float(deviation_squared.sum())
        result[name] = {**statistics, 'two_sided_pvalue': probability,
            'familywise_pvalue': min(1., family_size * probability) if probability is not None else None,
            'familywise_tests': family_size, 'familywise_99_threshold': float(norm.ppf(1. - .01 / (2. * family_size))),
            'ci95_lower': statistics['mean'] - norm.ppf(.975) * statistics['standard_error'],
            'ci95_upper': statistics['mean'] + norm.ppf(.975) * statistics['standard_error'],
            'mean_squared_action': float(values[:, 1].mean()),
            'log_exponential_loss_relative_to_zero': float(logsumexp(values[:, 2]) - math.log(len(values))),
            'mean_squared_utility_relative_to_zero': float(values[:, 3].mean()),
            'maximum_pair_share_of_sample_variance': float(deviation_squared.max() / total) if total > 0. else None,
            'variance_concentration_effective_pairs': float(total**2 / np.sum(deviation_squared**2)) if total > 0. else None}
    return result


def _open_terminal_scores(path: Path, *, identity: str, paths: int, config: PathAuditConfig) -> h5py.File:
    """Publish a complete initial schema atomically, then validate each resume."""
    batches = (paths + config.batch_size - 1) // config.batch_size
    schema = {'scores': ((paths, len(ENDPOINTS), len(SCORE_FIELDS)), 'f8'),
        'path_validity': ((paths,), 'bool'), 'policy_validity': ((paths,), 'bool'),
        'path_health': ((paths, config.days), 'u1'),
        'batch_diagnostic_validity': ((batches,), 'bool'),
        'batch_diagnostics': ((batches, config.days, 7 if config.preserve_log_dlvsq else 6), 'f8')}
    if not path.exists():
        temporary = path.with_suffix('.initializing')
        with h5py.File(temporary, 'w') as saved:
            for name, (shape, dtype) in schema.items():
                saved.create_dataset(name, shape=shape, dtype=dtype)
            saved.attrs.update(identity=identity, completed_paths=0, schema_complete=True)
            saved.flush()
        temporary.replace(path)
    saved = h5py.File(path, 'r+')
    try:
        if saved.attrs.get('identity') != identity:
            raise ValueError('fresh evaluation identity changed; preserve the existing run')
        if not saved.attrs.get('schema_complete', False) or 'completed_paths' not in saved.attrs:
            raise ValueError('published fresh evaluation schema is incomplete; preserve it for investigation')
        for name, (shape, dtype) in schema.items():
            if name not in saved or saved[name].shape != shape or saved[name].dtype != np.dtype(dtype):
                raise ValueError('published fresh evaluation dataset schema changed')
        if not 0 <= int(saved.attrs['completed_paths']) <= paths:
            raise ValueError('fresh evaluation completed count is outside the population')
        return saved
    except BaseException:
        saved.close()
        raise


def evaluate_fresh_terminal(*, manifest: Path, output: Path, seed: int, paths: int,
                            backend: str) -> dict:
    """Generate fresh unchanged-law paths and checkpoint compact scores by batch."""
    check_resource_headroom()
    configure_backend(backend)
    frozen, actions = load_terminal_actions(manifest, device='cuda' if backend == 'gpu' else 'cpu')
    source = frozen['source_population_identity']
    if seed == source['config']['seed']:
        raise ValueError('confirmation requires a separately seeded population')
    settings = frozen['evaluation_settings']
    if (seed != settings['seed'] or paths != settings['paths']
        or source['config']['substeps'] != settings['substeps']
        or source['config']['batch_size'] != settings['batch_size']):
        raise ValueError('confirmation seed, count or geometry differs from its frozen declaration')
    fields = dict(source['config'])
    fields['nms'] = tuple(fields['nms'])
    config = replace(PathAuditConfig(**fields), paths=paths, seed=seed)
    if source['code_version'] != PathAuditConfig.version.unique_id48:
        raise ValueError('simulation version changed after original population generation')
    for filename, digest in source['implementation_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('simulation implementation changed after the frozen source population')
    calibration = Path(source['model']['calibration_root'])
    arrays, metadata = load_model(calibration, source['model']['origin'])
    if any(source['model'][key] != value for key, value in metadata.items()):
        raise ValueError('fresh simulation model inputs differ from fitted-action inputs')
    with np.load(calibration / 'own_dates' / metadata['origin'] / 'inputs.npz', allow_pickle=False) as inputs:
        reference = np.asarray(inputs['projected_total_variances'])
    if hashlib.sha256(reference.tobytes()).hexdigest() != source['reference_variances_sha256']:
        raise ValueError('inception contract-defining ATM variances changed')
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=reference, config=config)
    expiries = np.repeat(contracts['expiry_days'], contracts['strikes'].shape[-1])
    eligible = np.column_stack((np.ones(config.days, dtype=bool), expiries[None] > np.arange(config.days)[:, None]))
    simulate = path_function(arrays=arrays, background_volatility=metadata['background_volatility'], contracts=contracts, config=config)
    identity = {'schema': 'gromm2-fresh-terminal-scores-v1', 'source_population': source,
        'config': asdict(config), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'backend': backend,
        'semantic_sha256': frozen['semantic_sha256'],
        'evaluation_plan_sha256': frozen['evaluation_plan_sha256'],
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'endpoint_names': ENDPOINTS, 'score_fields': SCORE_FIELDS,
        'evaluation_array_view': 'float32 features and gains, matching original retained population',
        'no_path_censoring_or_measure_reweighting': True, 'familywise_endpoint_count': 12}
    identity_json = json.dumps(identity, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    stop, done = threading.Event(), threading.Event()
    progress = {'stage': 'initializing', 'completed_paths': 0, 'total_paths': paths}
    started = time.monotonic()

    def heartbeat() -> None:
        while not done.wait(60.):
            print(json.dumps({'event': 'fresh_terminal_progress', **progress, 'elapsed_seconds': time.monotonic()-started}), flush=True)
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(output / 'evaluation.lock'), timeout=0), _open_terminal_scores(
            output / 'path_scores.h5', identity=identity_json, paths=paths, config=config) as saved:
            completed = int(saved.attrs['completed_paths'])
            if completed % 2 or (completed != paths and completed % config.batch_size):
                raise ValueError('fresh checkpoint splits an antithetic batch')
            initial = completed
            for start in range(completed, paths, config.batch_size):
                if stop.is_set():
                    raise InterruptedError('fresh evaluation stopped at the last committed complete batch')
                count = min(config.batch_size, paths - start)
                progress.update(stage='simulating_unchanged_Q_paths', completed_paths=start)
                normals, background = normal_block(config=config, start=start, count=count, factors=np.asarray(arrays['loadings']).shape[-1])
                values = jax.device_get(simulate(jnp.asarray(normals), jnp.asarray(background)))
                features, gains, spots, terminal, diagnostics = values[:5]
                health = values[5] if config.retain_nonfinite else np.zeros((count, config.days), dtype=np.uint8)
                valid = np.isfinite(features).all(axis=(1, 2)) & np.isfinite(gains).all(axis=(1, 2))
                valid &= np.isfinite(spots).all(axis=1) & (spots > 0.).all(axis=1) & np.isfinite(terminal).all(axis=1)
                valid &= ~np.any(health, axis=1)
                for array in (features, gains, spots, terminal):
                    valid &= np.all(np.abs(array).reshape(count, -1) <= np.finfo(np.float32).max, axis=1)
                diagnostic_valid = bool(np.isfinite(diagnostics).all() and np.min(diagnostics[:, 3]) >= -1.e-12 and np.max(diagnostics[:, 1]) <= 1.e-10)
                records = np.full((count, len(ENDPOINTS), len(SCORE_FIELDS)), np.nan)
                policy_valid = np.zeros(count, dtype=bool)
                if np.all(valid) and diagnostic_valid:
                    progress['stage'] = 'evaluating_immutable_frozen_actions'
                    records = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=actions)
                    policy_valid = np.isfinite(records).all(axis=(1, 2))
                saved['scores'][start:start+count] = records
                saved['path_validity'][start:start+count] = valid
                saved['policy_validity'][start:start+count] = policy_valid
                saved['path_health'][start:start+count] = health
                saved['batch_diagnostic_validity'][start // config.batch_size] = diagnostic_valid
                saved['batch_diagnostics'][start // config.batch_size] = diagnostics
                saved.flush()
                saved.attrs.modify('completed_paths', start + count)
                saved.flush()
                elapsed = time.monotonic() - started
                progress.update(stage='batch_committed', completed_paths=start+count,
                    remaining_seconds=(paths-start-count)*elapsed/max(start+count-initial, 1))
                print(json.dumps({'event': 'fresh_terminal_batch_committed', **progress, 'elapsed_seconds': elapsed,
                    'invalid_paths_in_batch': int(np.sum(~valid)), 'invalid_policy_scores_in_batch': int(np.sum(~policy_valid))}), flush=True)
                check_resource_headroom()
                del values, features, gains, spots, terminal, diagnostics, records
            invalid = int(np.sum(~saved['path_validity'][:]))
            invalid_policies = int(np.sum(~saved['policy_validity'][:]))
            invalid_batches = int(np.sum(~saved['batch_diagnostic_validity'][:]))
            report = {'status': 'complete' if not (invalid or invalid_policies or invalid_batches) else 'unavailable: invalid complete population or policy scores',
                'identity': identity, 'completed_paths': paths, 'invalid_paths': invalid,
                'invalid_policy_scores': invalid_policies, 'invalid_batches': invalid_batches, 'paths_dropped': 0,
                'zero_baseline_selection_is_still_uninformative': True, 'optimizer_stationarity_certified': False}
            if report['status'] == 'complete':
                report['diagnostics'] = summarize_terminal_scores(saved['scores'][:])
            write_json(output / 'fresh_terminal_report.json', report)
            print(json.dumps({'event': 'fresh_terminal_complete', 'status': report['status'], 'paths': paths,
                'diagnostics': report.get('diagnostics'), 'elapsed_seconds': time.monotonic()-started}), flush=True)
            return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--paths', type=int, default=100000)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    arguments = parser.parse_args()
    evaluate_fresh_terminal(manifest=arguments.manifest, output=arguments.output_directory,
        seed=arguments.seed, paths=arguments.paths, backend=arguments.backend)


if __name__ == '__main__':
    main()
