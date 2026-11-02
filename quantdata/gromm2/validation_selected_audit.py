"""Separate full-class producer for an explicitly declared validation protocol."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import AffineDeepAction, LocalActionConfig, fold_path_ranges
from quantdata.gromm2.full_gradient_audit import _adopt_affine_initializer, _atomic_torch_save
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _normalization,
    _path_identity,
    _resident_fold,
    internal_training_mask,
)
from quantdata.gromm2.metric_audit_optimization import MetricSolveConfig
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.terminal_audit import terminal_semantic_hashes
from quantdata.gromm2.validation_action_optimization import (
    ValidationSelectionConfig,
    fit_validation_selected_action,
)
from quantdata.torch import BaseEnvironment


def selection_source_hashes() -> dict[str, str]:
    """Bind existing numerical semantics and the separate selection implementation."""
    hashes = terminal_semantic_hashes()
    for name in ('validation_selected_audit', 'validation_action_optimization', 'metric_audit_optimization',
                 'selected_audit_artifacts', 'selected_audit_inference'):
        path = Path(__file__).with_name(name+'.py').resolve()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def fit_validation_selected_fold(*, path: Path, output: Path, fold: int, declaration: Path,
                                 protocol: LocalAuditProtocol = LocalAuditProtocol(),
                                 affine_config: AffineSolveConfig = AffineSolveConfig(),
                                 optimizer_config: MetricSolveConfig = MetricSolveConfig(),
                                 selection_config: ValidationSelectionConfig = ValidationSelectionConfig(),
                                 affine_initializer_root: Path | None = None, device: str = 'cuda') -> dict:
    """Complete both independent objectives; numerical convergence is descriptive.

    The caller must agree and publish the selection declaration before using
    this alternative protocol. Confirmation is never read by the producer.
    Both fitted halves will require independent fresh confirmation rather than
    a claim of untouched original-population cross-fitting after development.
    """
    if fold not in (0, 1) or device not in ('cpu', 'cuda'):
        raise ValueError('an ordered fold and explicit compute device are required')
    if device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('validation-selected training requires a managed CUDA allocation')
    controls = {'protocol': asdict(protocol), 'affine': asdict(affine_config),
        'optimizer': asdict(optimizer_config), 'selection': asdict(selection_config)}
    content = declaration.read_bytes()
    declared = json.loads(content)
    implementation = selection_source_hashes()
    if (declared['schema'] != 'gromm2-validation-selection-declaration-v1'
        or declared['controls'] != controls or declared['implementation_sha256'] != implementation
        or declared['independent_fresh_confirmation_only'] is not True):
        raise ValueError('selection declaration controls, source or independence contract changed')
    check_resource_headroom()
    source_identity, geometry = _path_identity(path)
    if hashlib.sha256(source_identity.encode()).hexdigest() != declared['source_population_identity_sha256']:
        raise ValueError('selection declaration binds a different development population')
    outer = fold_path_ranges(geometry['paths'])[fold]
    mask = internal_training_mask(outer.stop-outer.start, protocol=protocol, fold=fold)
    normalization_identity = json.dumps({'path_identity': source_identity, 'protocol': asdict(protocol),
        'fold': fold, 'protocol_version': LocalAuditProtocol.version.unique_id48}, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    stop, done = threading.Event(), threading.Event()
    started = checked = time.monotonic()
    progress = {'stage': 'initializing_validation_selected_fold', 'fold': fold,
        'completed_work_units': 0, 'total_work_units': 2}

    def observe(entry: dict) -> None:
        nonlocal checked
        if stop.is_set():
            raise InterruptedError('selection fit stopped; retain committed native and selection checkpoints')
        progress.update(entry)
        now = time.monotonic()
        if now-checked >= 60. or entry.get('stage') == 'inner_validation_selection':
            print(json.dumps({'event': 'validation_selected_fold_progress', **progress,
                'elapsed_seconds': now-started, 'remaining_seconds': None}), flush=True)
            check_resource_headroom()
            checked = now

    def heartbeat() -> None:
        while not done.wait(60.):
            print(json.dumps({'event': 'validation_selected_fold_heartbeat', **progress,
                'elapsed_seconds': time.monotonic()-started, 'remaining_seconds': None}), flush=True)
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(output/'producer.lock'), timeout=0):
            if affine_initializer_root is not None and not (output/'normalization.npz').exists():
                saved = torch.load(affine_initializer_root/'squared-affine-initializer.pt', map_location='cpu', weights_only=False)
                copied = Path(saved['plan']['normalization']).read_bytes()
                if hashlib.sha256(copied).hexdigest() != saved['plan']['normalization_sha256']:
                    raise ValueError('external affine normalization changed')
                temporary = output/'normalization.tmp'
                temporary.write_bytes(copied)
                temporary.replace(output/'normalization.npz')
            if (output/'normalization.npz').exists():
                with np.load(output/'normalization.npz', allow_pickle=False) as saved:
                    previous = str(saved['identity'].item())
                if json.loads(previous) != json.loads(normalization_identity):
                    raise ValueError('selection normalization population, fold or protocol changed')
                normalization_identity = previous
            normalization = _normalization(path, outer=outer, mask=mask, protocol=protocol,
                output=output/'normalization.npz', identity=normalization_identity, upload_paths=128)
            if not np.array_equal(normalization['inner_training_mask'], mask):
                raise ValueError('saved selection normalization path-pair partition changed')
            identity = {'source_population_identity': json.loads(source_identity),
                'source_population_identity_sha256': hashlib.sha256(source_identity.encode()).hexdigest(),
                'normalization_sha256': hashlib.sha256((output/'normalization.npz').read_bytes()).hexdigest(),
                'fold': fold, 'protocol': asdict(protocol), 'implementation_sha256': implementation,
                'declaration': str(declaration.resolve()), 'declaration_sha256': hashlib.sha256(content).hexdigest(),
                'input_precision': 'original resident float32 normalization, float64 arithmetic'}
            data_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            report = {'identity': identity, 'controls': controls, 'data_identity': data_identity,
                'outer_training_paths': [outer.start, outer.stop], 'outer_holdout_read': False,
                'inner_training_paths': int(mask.sum()), 'inner_validation_paths': int((~mask).sum()),
                'inference_performed': False, 'diagnostics': {}}
            report_path = output/'training-report.json'
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report['identity'] != identity or report['controls'] != controls:
                    raise ValueError('validation-selected producer restart identity changed')
            report['status'] = 'in_progress; no frozen actions or inference'
            write_json(report_path, report)
            states, returns, eligible = _resident_fold(path, outer=outer, normalization=normalization,
                env=BaseEnvironment(device=device, precision=32),
                train_config=LocalAuditTrainConfig(device=device, upload_paths=512, host_cache=False),
                progress=progress, stop_requested=stop.is_set)
            rows = torch.arange(len(states), device=states.device).reshape(-1, geometry['days'])
            inner = torch.as_tensor(mask, dtype=torch.bool, device=states.device)
            arguments = {'states': states, 'returns': returns, 'eligible': eligible,
                'action_scale': float(normalization['action_scale']), 'squared_lambda': protocol.squared_lambda,
                'config': affine_config, 'progress': observe}
            training = AffineAuditProblem(rows=rows[inner].flatten(), **arguments)
            validation = AffineAuditProblem(rows=rows[~inner].flatten(), **arguments)
            training.prepare()
            for index, name in enumerate(('squared', 'entropy')):
                observe({'stage': 'initializing_full_validation_selected_policy',
                    'diagnostic': name, 'completed_work_units': index})
                config = LocalActionConfig(label=f'full-gradient-fold-{fold}', feature_count=geometry['features'],
                    instrument_count=geometry['instruments'], action_scale=training.action_scale,
                    network=protocol.network, squared_lambda=protocol.squared_lambda)
                torch.manual_seed(protocol.split_seed+fold+1000*index)
                policy = AffineDeepAction(BaseEnvironment(device=device, precision=64), config=config)
                if (output/name/'resume.pt').exists():
                    if name not in report['diagnostics']:
                        raise ValueError('accepted selection resume lacks its committed initialization provenance')
                    initializer = report['diagnostics'][name]['initializer']
                elif affine_initializer_root is not None:
                    affine_report, provenance = _adopt_affine_initializer(root=affine_initializer_root,
                        diagnostic=name, policy=policy, problem=training, identity=identity)
                    initializer = {'affine_report': affine_report, 'provenance': provenance}
                else:
                    affine_path = output/f'{name}-affine.pt'
                    initial = None
                    if affine_path.exists():
                        saved = torch.load(affine_path, map_location=states.device, weights_only=False)
                        if saved['identity'] != identity:
                            raise ValueError('selection affine checkpoint identity changed')
                        initial = saved['weights']

                    def save_affine(weights: torch.Tensor, state: dict) -> None:
                        _atomic_torch_save(affine_path, {'identity': identity, 'weights': weights.cpu(), 'state': state})

                    weights, affine_report = training.solve(diagnostic=name, initial=initial, checkpoint=save_affine)
                    if not affine_report['converged']:
                        raise RuntimeError('selection affine initializer did not meet its numerical criteria')
                    with torch.no_grad():
                        policy.affine.I0.weight.copy_(weights[:-1].T)
                        policy.affine.I0.bias.copy_(weights[-1])
                    initializer = {'affine_report': affine_report, 'provenance': None}
                initializer_sha256 = hashlib.sha256(json.dumps(initializer, sort_keys=True).encode()).hexdigest()
                policy_data_identity = hashlib.sha256(json.dumps({'fold_data_identity': data_identity,
                    'diagnostic': name, 'initializer_sha256': initializer_sha256}, sort_keys=True).encode()).hexdigest()
                report['diagnostics'][name] = {'initializer': initializer, 'status': 'fitting',
                    'initializer_sha256': initializer_sha256, 'policy_data_identity': policy_data_identity}
                write_json(report_path, report)
                result = fit_validation_selected_action(policy=policy, training=training, validation=validation,
                    diagnostic=name, data_identity=policy_data_identity, output=output/name,
                    optimizer_config=optimizer_config, selection_config=selection_config, progress=observe)
                report['diagnostics'][name].update(status='selection_complete', result=result)
                write_json(report_path, report)
            report['status'] = 'both independent full-class selections complete; fresh confirmation pending'
            write_json(report_path, report)
            return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
