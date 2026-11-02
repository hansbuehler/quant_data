"""Fresh confirmation of explicitly frozen validation-selected full actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock

from quantdata.gromm2.forward_hedging import load_model
from quantdata.gromm2.path_audit import (
    PathAuditConfig,
    audit_contracts,
    normal_block,
    path_function,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.runtime import configure_backend
from quantdata.gromm2.selected_audit_artifacts import load_selected_actions
from quantdata.gromm2.terminal_inference import (
    ENDPOINTS,
    SCORE_FIELDS,
    _open_terminal_scores,
    score_terminal_batch,
    summarize_terminal_scores,
)


def _selected_population(frozen: dict, *, seed: int, paths: int) -> tuple:
    """Reconstruct the exact declared simulator, contracts and independently seeded law."""
    source, settings = frozen['source_population_identity'], frozen['evaluation_settings']
    if (seed == source['config']['seed'] or seed != settings['seed'] or paths != settings['paths']
        or source['config']['substeps'] != settings['substeps']
        or source['config']['batch_size'] != settings['batch_size']):
        raise ValueError('selected confirmation seed, count or geometry differs from its declaration')
    fields = dict(source['config'])
    fields['nms'] = tuple(fields['nms'])
    config = replace(PathAuditConfig(**fields), paths=paths, seed=seed)
    if source['code_version'] != PathAuditConfig.version.unique_id48:
        raise ValueError('simulator version changed after development population generation')
    for filename, digest in source['implementation_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('simulator implementation changed after development population generation')
    calibration = Path(source['model']['calibration_root'])
    arrays, metadata = load_model(calibration, source['model']['origin'])
    if any(source['model'][key] != value for key, value in metadata.items()):
        raise ValueError('fresh selected simulation inputs differ from development inputs')
    with np.load(calibration/'own_dates'/metadata['origin']/'inputs.npz', allow_pickle=False) as values:
        reference = np.asarray(values['projected_total_variances'])
    if hashlib.sha256(reference.tobytes()).hexdigest() != source['reference_variances_sha256']:
        raise ValueError('inception ATM variances defining the audit contracts changed')
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=reference, config=config)
    expiries = np.repeat(contracts['expiry_days'], contracts['strikes'].shape[-1])
    eligible = np.column_stack((np.ones(config.days, dtype=bool), expiries[None] > np.arange(config.days)[:, None]))
    simulate = path_function(arrays=arrays, background_volatility=metadata['background_volatility'],
        contracts=contracts, config=config)
    return config, arrays, eligible, simulate


def evaluate_fresh_selected(*, manifest: Path, output: Path, seed: int, paths: int, backend: str) -> dict:
    """Reuse original-unit scoring and pair SEs with a distinct artifact contract.

    No development returns or fitted path weights enter confirmation. Every
    generated path belongs to the result; any numerical invalidity makes the
    whole diagnostic unavailable rather than filtering it into a passing sample.
    """
    check_resource_headroom()
    configure_backend(backend)
    frozen, actions = load_selected_actions(manifest, device='cuda' if backend == 'gpu' else 'cpu')
    config, arrays, eligible, simulate = _selected_population(frozen, seed=seed, paths=paths)
    identity = {'schema': 'gromm2-fresh-selected-scores-v1',
        'source_population': frozen['source_population_identity'], 'config': asdict(config),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'backend': backend,
        'semantic_sha256': frozen['semantic_sha256'], 'implementation_sha256': frozen['implementation_sha256'],
        'evaluation_plan_sha256': frozen['evaluation_plan_sha256'], 'endpoint_names': ENDPOINTS,
        'score_fields': SCORE_FIELDS, 'familywise_endpoint_count': 12,
        'evaluation_array_view': 'float32 features and gains, matching original retained population',
        'no_path_censoring_or_measure_reweighting': True, 'independent_fresh_confirmation_only': True}
    identity_json = json.dumps(identity, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    stop, done = threading.Event(), threading.Event()
    started = time.monotonic()
    progress = {'stage': 'initializing', 'completed_paths': 0, 'total_paths': paths, 'remaining_seconds': None}

    def heartbeat() -> None:
        while not done.wait(60.):
            print(json.dumps({'event': 'fresh_selected_heartbeat', **progress,
                'elapsed_seconds': time.monotonic()-started}), flush=True)
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(output/'evaluation.lock'), timeout=0), _open_terminal_scores(
            output/'path_scores.h5', identity=identity_json, paths=paths, config=config) as saved:
            initial = int(saved.attrs['completed_paths'])
            if initial % 2 or (initial != paths and initial % config.batch_size):
                raise ValueError('selected confirmation checkpoint splits an antithetic batch')
            for start in range(initial, paths, config.batch_size):
                if stop.is_set():
                    raise InterruptedError('selected confirmation stopped at the last committed complete batch')
                count = min(config.batch_size, paths-start)
                progress.update(stage='simulating_unchanged_Q_paths', completed_paths=start)
                normals, background = normal_block(config=config, start=start, count=count,
                    factors=np.asarray(arrays['loadings']).shape[-1])
                values = jax.device_get(simulate(jnp.asarray(normals), jnp.asarray(background)))
                features, gains, spots, terminal, diagnostics = values[:5]
                health = values[5] if config.retain_nonfinite else np.zeros((count, config.days), dtype=np.uint8)
                valid = np.isfinite(features).all(axis=(1, 2)) & np.isfinite(gains).all(axis=(1, 2))
                valid &= np.isfinite(spots).all(axis=1) & (spots > 0.).all(axis=1) & np.isfinite(terminal).all(axis=1)
                valid &= ~np.any(health, axis=1)
                for array in (features, gains, spots, terminal):
                    valid &= np.all(np.abs(array).reshape(count, -1) <= np.finfo(np.float32).max, axis=1)
                diagnostic_valid = bool(np.isfinite(diagnostics).all()
                    and np.min(diagnostics[:, 3]) >= -1.e-12 and np.max(diagnostics[:, 1]) <= 1.e-10)
                records = np.full((count, len(ENDPOINTS), len(SCORE_FIELDS)), np.nan)
                policy_valid = np.zeros(count, dtype=bool)
                if np.all(valid) and diagnostic_valid:
                    progress['stage'] = 'scoring_immutable_selected_actions'
                    records = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=actions)
                    policy_valid = np.isfinite(records).all(axis=(1, 2))
                saved['scores'][start:start+count] = records
                saved['path_validity'][start:start+count] = valid
                saved['policy_validity'][start:start+count] = policy_valid
                saved['path_health'][start:start+count] = health
                saved['batch_diagnostic_validity'][start//config.batch_size] = diagnostic_valid
                saved['batch_diagnostics'][start//config.batch_size] = diagnostics
                saved.flush()
                saved.attrs.modify('completed_paths', start+count)
                saved.flush()
                elapsed = time.monotonic()-started
                progress.update(stage='batch_committed', completed_paths=start+count,
                    remaining_seconds=(paths-start-count)*elapsed/max(start+count-initial, 1))
                print(json.dumps({'event': 'fresh_selected_batch_committed', **progress,
                    'elapsed_seconds': elapsed, 'invalid_paths_in_batch': int(np.sum(~valid)),
                    'invalid_policy_scores_in_batch': int(np.sum(~policy_valid))}), flush=True)
                check_resource_headroom()
                del values, features, gains, spots, terminal, diagnostics, records
            invalid = int(np.sum(~saved['path_validity'][:]))
            invalid_policy = int(np.sum(~saved['policy_validity'][:]))
            invalid_batches = int(np.sum(~saved['batch_diagnostic_validity'][:]))
            report = {'status': 'complete' if not (invalid or invalid_policy or invalid_batches)
                else 'unavailable: invalid complete population or policy scores', 'identity': identity,
                'completed_paths': paths, 'invalid_paths': invalid, 'invalid_policy_scores': invalid_policy,
                'invalid_batches': invalid_batches, 'paths_dropped': 0, 'optimizer_stationarity_certified': False,
                'selection_is_not_a_martingale_proof': True, 'zero_score_is_uninformative': True}
            if report['status'] == 'complete':
                report['diagnostics'] = summarize_terminal_scores(saved['scores'][:])
            write_json(output/'fresh_selected_report.json', report)
            print(json.dumps({'event': 'fresh_selected_complete', 'status': report['status'],
                'paths': paths, 'diagnostics': report.get('diagnostics'),
                'elapsed_seconds': time.monotonic()-started}), flush=True)
            return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--paths', type=int, default=100000)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    arguments = parser.parse_args()
    evaluate_fresh_selected(manifest=arguments.manifest, output=arguments.output_directory,
        seed=arguments.seed, paths=arguments.paths, backend=arguments.backend)


if __name__ == '__main__':
    main()
