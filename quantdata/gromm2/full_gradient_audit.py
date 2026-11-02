"""Fit the original affine-plus-deep diagnostics with deterministic gradients."""

from __future__ import annotations

import hashlib
import io
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
from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    AffineDeepAction,
    LocalActionConfig,
    fold_path_ranges,
)
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _normalization,
    _path_identity,
    _resident_fold,
    internal_training_mask,
)
from quantdata.gromm2.neural_audit_optimization import (
    NeuralSolveConfig,
    fit_full_gradient_action,
    full_action_pass,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.terminal_audit import validated_evaluation_plan
from quantdata.torch import BaseEnvironment


def _atomic_torch_save(path: Path, values: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(values, temporary)
    temporary.replace(path)


def _adopt_affine_initializer(*, root: Path, diagnostic: str, policy: AffineDeepAction,
                              problem: AffineAuditProblem, identity: dict) -> tuple[dict, dict]:
    paths = {'plan': root/'plan.json', 'report': root/f'{diagnostic}-report.json',
        'checkpoint': root/f'{diagnostic}-checkpoint.pt', 'weights': root/f'{diagnostic}-affine-initializer.pt'}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    plan, report = json.loads(contents['plan']), json.loads(contents['report'])
    saved = torch.load(io.BytesIO(contents['weights']), map_location='cpu', weights_only=False)
    checkpoint = torch.load(io.BytesIO(contents['checkpoint']), map_location='cpu', weights_only=False)
    config = policy.config
    if (saved['plan'] != plan or checkpoint['plan'] != plan or not report['converged']
        or not checkpoint['state']['gradient_converged'] or checkpoint['state']['diagnostic'] != diagnostic
        or checkpoint['state']['history'] != report['history']
        or report['source_population_identity_sha256'] != identity['source_population_identity_sha256']
        or plan['normalization_sha256'] != identity['normalization_sha256']
        or saved['code_version'] != LocalActionConfig.version.unique_id48
        or saved['model_config']['action_scale'] != config.action_scale
        or saved['model_config']['feature_count'] != config.feature_count
        or saved['model_config']['instrument_count'] != config.instrument_count
        or saved['model_config']['squared_lambda'] != config.squared_lambda
        or saved['model_config']['network'] != asdict(config.network) or saved['precision'] != 64
        or report.get('outer_holdout_read') is not False):
        raise ValueError('external affine initializer is not a matching converged training-only fit')
    for filename, digest in plan['implementation_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('external affine initializer implementation changed')
    weights = torch.cat((saved['state_dict']['affine.I0.weight'].T,
        saved['state_dict']['affine.I0.bias'][None]), dim=0)
    if not torch.equal(weights, checkpoint['weights']):
        raise ValueError('external affine export differs from its accepted checkpoint')
    policy.load_state_dict(saved['state_dict'], strict=True)
    head = policy.deep.In[-1] if len(policy.deep.In) else policy.deep.I0
    if bool(head.weight.ne(0.).any()) or bool(head.bias.ne(0.).any()):
        raise ValueError('external affine initializer contains an unvalidated nonlinear action')
    loss, gradient = problem.evaluate(weights.to(problem.states.device), diagnostic=diagnostic)
    last = report['history'][-1]
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    dual_norm = float(torch.sqrt(problem.residual_energy(gradient)))
    if (not np.isclose(float(loss), last['loss'], rtol=1.e-10, atol=1.e-12)
        or gradient_norm > last['gradient_threshold']*1.01+1.e-12
        or dual_norm > last['preconditioned_gradient_threshold']*1.01+1.e-12):
        raise ValueError('external affine convergence does not hold on the actual training observations')
    provenance = {'artifacts_sha256': {str(paths[name]): hashlib.sha256(content).hexdigest()
        for name, content in contents.items()}, 'plan': plan, 'verified_training_loss': float(loss),
        'verified_gradient_norm': gradient_norm, 'verified_preconditioned_gradient_norm': dual_norm,
        'checkpoint_export_equal': True, 'input_normalization_dtype': 'float32'}
    return report, provenance


def fit_full_gradient_fold(*, path: Path, output: Path, fold: int,
                           protocol: LocalAuditProtocol = LocalAuditProtocol(),
                           affine_config: AffineSolveConfig = AffineSolveConfig(),
                           neural_config: NeuralSolveConfig = NeuralSolveConfig(),
                           affine_initializer_root: Path | None = None,
                           device: str = 'cuda') -> dict:
    """Fit one complete-path fold, without opening its opposite fold's arrays.

    Inner validation is descriptive only: deterministic optimization and its
    stopping rules use full inner-training objectives/gradients. Terminal
    trained actions are retained; zero is an objective reference, not a
    fallback checkpoint. No inference is performed by this function.
    """
    if fold not in (0, 1) or device not in ('cpu', 'cuda'):
        raise ValueError('an ordered fold and explicit compute device are required')
    if device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('full-gradient GPU fitting requires a managed allocation')
    check_resource_headroom()
    source_identity, geometry = _path_identity(path)
    outer = fold_path_ranges(geometry['paths'])[fold]
    normalization_identity = json.dumps({'path_identity': source_identity, 'protocol': asdict(protocol),
        'fold': fold, 'protocol_version': LocalAuditProtocol.version.unique_id48}, sort_keys=True)
    mask = internal_training_mask(outer.stop-outer.start, protocol=protocol, fold=fold)
    output.mkdir(parents=True, exist_ok=True)
    stop, done = threading.Event(), threading.Event()
    started = checked = time.monotonic()
    progress = {'stage': 'initializing_full_gradient_fold', 'fold': fold,
        'completed_work_units': 0, 'total_work_units': len(DIAGNOSTICS)}

    def observe(entry: dict) -> None:
        nonlocal checked
        if stop.is_set():
            raise InterruptedError('full-gradient fit stopped; retain its last accepted checkpoint')
        progress.update(entry)
        now = time.monotonic()
        if now-checked >= 30. or entry.get('stage') in ('affine_newton_iteration', 'neural_full_gradient_iteration'):
            print(json.dumps({'event': 'full_gradient_fold_progress', **progress,
                'elapsed_seconds': now-started, 'remaining_seconds': None}), flush=True)
            check_resource_headroom()
            checked = now

    def heartbeat() -> None:
        while not done.wait(60.):
            print(json.dumps({'event': 'full_gradient_fold_heartbeat', **progress,
                'elapsed_seconds': time.monotonic()-started, 'remaining_seconds': None}), flush=True)
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(output/'fit.lock'), timeout=0):
            if affine_initializer_root is not None and not (output/'normalization.npz').exists():
                source_initializer = torch.load(affine_initializer_root/'squared-affine-initializer.pt', map_location='cpu', weights_only=False)
                normalization_source = Path(source_initializer['plan']['normalization'])
                content = normalization_source.read_bytes()
                if hashlib.sha256(content).hexdigest() != source_initializer['plan']['normalization_sha256']:
                    raise ValueError('external affine normalization changed')
                temporary = output/'normalization.tmp'
                temporary.write_bytes(content)
                temporary.replace(output/'normalization.npz')
            if (output/'normalization.npz').exists():
                with np.load(output/'normalization.npz', allow_pickle=False) as saved:
                    stored_identity = str(saved['identity'].item())
                if json.loads(stored_identity) != json.loads(normalization_identity):
                    raise ValueError('saved training-only normalization has a different identity')
                normalization_identity = stored_identity
            normalization = _normalization(path, outer=outer, mask=mask, protocol=protocol,
                output=output/'normalization.npz', identity=normalization_identity, upload_paths=128)
            source_hashes = {str(filename): hashlib.sha256(filename.read_bytes()).hexdigest() for filename in (
                Path(__file__), Path(__file__).with_name('affine_audit_optimization.py'),
                Path(__file__).with_name('neural_audit_optimization.py'), Path(__file__).with_name('conditional_audit.py'),
                Path(__file__).with_name('local_audit_training.py'), Path(__file__).with_name('audit_host_cache.py'),
                Path(__file__).parents[1]/'torch'/'__init__.py',
                Path(__file__).parents[1]/'torch'/'ffw.py')}
            identity = {'source_population_identity_sha256': hashlib.sha256(source_identity.encode()).hexdigest(),
                'normalization_sha256': hashlib.sha256((output/'normalization.npz').read_bytes()).hexdigest(),
                'fold': fold, 'protocol': asdict(protocol), 'implementation_sha256': source_hashes,
                'input_precision': 'original resident float32 normalization, float64 arithmetic',
                'source_population_identity': json.loads(source_identity)}
            identity_json = json.dumps(identity, sort_keys=True)
            data_identity = hashlib.sha256(identity_json.encode()).hexdigest()
            states, returns, eligible = _resident_fold(path, outer=outer, normalization=normalization,
                env=BaseEnvironment(device=device, precision=32),
                train_config=LocalAuditTrainConfig(device=device, upload_paths=512, host_cache=False),
                progress=progress, stop_requested=stop.is_set)
            inner = torch.as_tensor(mask, dtype=torch.bool, device=states.device)
            rows = torch.arange(len(states), device=states.device).reshape(-1, geometry['days'])
            train_rows, validation_rows = rows[inner].flatten(), rows[~inner].flatten()
            arguments = {'states': states, 'returns': returns, 'eligible': eligible,
                'action_scale': float(normalization['action_scale']), 'squared_lambda': protocol.squared_lambda,
                'config': affine_config, 'progress': observe}
            training = AffineAuditProblem(rows=train_rows, **arguments)
            validation = AffineAuditProblem(rows=validation_rows, **arguments)
            training.prepare()
            report = {'identity': identity, 'data_identity': data_identity, 'outer_training_paths': [outer.start, outer.stop],
                'outer_holdout_read': False, 'inner_training_paths': int(mask.sum()),
                'inner_validation_paths': int((~mask).sum()), 'affine_config': asdict(affine_config),
                'neural_config': asdict(neural_config), 'diagnostics': {}, 'inference_performed': False}
            affine_initial_weights = None
            for index, name in enumerate(('squared', 'entropy')):
                observe({'stage': 'initializing_independent_policy', 'diagnostic': name, 'completed_work_units': index})
                config = LocalActionConfig(label=f'full-gradient-fold-{fold}', feature_count=geometry['features'],
                    instrument_count=geometry['instruments'], action_scale=training.action_scale,
                    network=protocol.network, squared_lambda=protocol.squared_lambda)
                torch.manual_seed(protocol.split_seed+fold+1000*index)
                policy = AffineDeepAction(BaseEnvironment(device=device, precision=64), config=config)
                affine_report = None
                initializer_provenance = None
                if affine_initializer_root is not None:
                    affine_report, initializer_provenance = _adopt_affine_initializer(root=affine_initializer_root,
                        diagnostic=name, policy=policy, problem=training, identity=identity)
                    write_json(output/f'{name}-initializer-adoption.json', initializer_provenance)
                else:
                    affine_path = output/f'{name}-affine.pt'
                    initial = affine_initial_weights
                    if affine_path.exists():
                        saved = torch.load(affine_path, map_location=states.device, weights_only=False)
                        if saved['identity'] != identity:
                            raise ValueError('saved affine initializer belongs to another full-gradient fit')
                        initial = saved['weights']

                    def save_affine(weights: torch.Tensor, state: dict) -> None:
                        _atomic_torch_save(affine_path, {'identity': identity, 'weights': weights.cpu(), 'state': state})

                    weights, affine_report = training.solve(diagnostic=name, initial=initial, checkpoint=save_affine)
                    write_json(output/f'{name}-affine-report.json', affine_report)
                    if not affine_report['converged']:
                        raise RuntimeError('full affine initialization has not met its numerical criteria')
                    with torch.no_grad():
                        policy.affine.I0.weight.copy_(weights[:-1].T)
                        policy.affine.I0.bias.copy_(weights[-1])
                affine_initial_weights = torch.cat((policy.affine.I0.weight.detach().T,
                    policy.affine.I0.bias.detach()[None]), dim=0).clone()
                checkpoint_path = output/f'{name}-neural.pt'
                optimizer, resume = None, None
                if checkpoint_path.exists():
                    saved = torch.load(checkpoint_path, map_location=states.device, weights_only=False)
                    if (saved['identity'] != identity or saved['code_version'] != LocalActionConfig.version.unique_id48
                        or saved['user_version'] != config.user_version):
                        raise ValueError('saved neural policy belongs to another full-gradient fit')
                    policy.load_state_dict(saved['model'], strict=True)
                    optimizer = torch.optim.LBFGS(policy.parameters(), lr=neural_config.initial_learning_rate,
                        max_iter=1, max_eval=neural_config.maximum_line_search_evaluations,
                        history_size=neural_config.history_size, tolerance_grad=0., tolerance_change=0., line_search_fn='strong_wolfe')
                    optimizer.load_state_dict(saved['optimizer'])
                    group = optimizer.param_groups[0]
                    if group['history_size'] != neural_config.history_size:
                        raise ValueError('resuming LBFGS requires its original curvature history size')
                    group.update(lr=neural_config.initial_learning_rate, max_eval=neural_config.maximum_line_search_evaluations,
                        history_size=neural_config.history_size)
                    resume = saved['state']
                    torch.set_rng_state(saved['torch_rng'].cpu())
                    if device == 'cuda' and saved['cuda_rng'] is not None:
                        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())

                def save_neural(model: AffineDeepAction, current_optimizer: torch.optim.LBFGS, state: dict) -> None:
                    _atomic_torch_save(checkpoint_path, {'identity': identity, 'model': model.state_dict(),
                        'optimizer': current_optimizer.state_dict(), 'state': state,
                        'code_version': LocalActionConfig.version.unique_id48, 'user_version': config.user_version,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None})

                neural_report = fit_full_gradient_action(policy=policy, problem=training, diagnostic=name,
                    config=neural_config, optimizer=optimizer, resume_state=resume,
                    data_identity=data_identity, progress=observe, checkpoint=save_neural)
                validation_loss = float(full_action_pass(policy, validation, diagnostic=name, backward=False))
                diagnostics = {'affine': affine_report, 'neural': neural_report,
                    'initializer_provenance': initializer_provenance,
                    'inner_validation_loss': validation_loss, 'inner_validation_used_for_optimization': False}
                report['diagnostics'][name] = diagnostics
                write_json(output/'training-report.json', report)
                if not neural_report['converged']:
                    raise RuntimeError('full affine-plus-deep fit has not met its full-gradient stopping rule')
                if neural_report['history'][-1]['training_loss'] >= 0.:
                    raise RuntimeError('a zero-or-worse empirical fit cannot supply the requested trained-action diagnostic')
                _atomic_torch_save(output/f'{name}-weights.pt', {'model_config': asdict(config),
                    'code_version': LocalActionConfig.version.unique_id48, 'user_version': config.user_version,
                    'diagnostic': name, 'state_dict': policy.state_dict(), 'identity': identity,
                    'input_normalization_dtype': 'float32',
                    'policy_state_source': 'terminal trained model, not best_state'})
                diagnostics['weights_sha256'] = hashlib.sha256((output/f'{name}-weights.pt').read_bytes()).hexdigest()
            report['status'] = 'both independent full-gradient policies fitted; inference pending'
            write_json(output/'training-report.json', report)
            return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def freeze_full_gradient_actions(*, folds: tuple[Path, Path], output: Path,
                                 evaluation_plan: Path, variant: str) -> dict:
    """Publish trained terminal actions only after both complete-fold fits pass."""
    confirmation = validated_evaluation_plan(evaluation_plan)
    settings = confirmation['variants'][variant]
    controls = confirmation['full_gradient_controls']
    reports = [json.loads((directory/'training-report.json').read_text()) for directory in folds]
    if [report['identity']['fold'] for report in reports] != [0, 1]:
        raise ValueError('two ordered, distinct complete training folds are required')
    source = reports[0]['identity']['source_population_identity']
    if any(report['identity']['source_population_identity'] != source for report in reports):
        raise ValueError('fitted folds belong to different source populations')
    if (source['config']['paths'] != settings['paths'] or source['config']['substeps'] != settings['substeps']
        or source['config']['batch_size'] != settings['batch_size']
        or source['config']['seed'] == settings['seed']):
        raise ValueError('confirmation geometry or independent seed differs from its declaration')
    provenance = {str(directory/'training-report.json'):
        hashlib.sha256((directory/'training-report.json').read_bytes()).hexdigest() for directory in folds}
    result = {'schema': 'gromm2-terminal-frozen-actions-v2', 'producer': 'full-gradient-affine-plus-deep-v1',
        'protocol': confirmation['protocol'], 'protocol_sha256': confirmation['protocol_sha256'],
        'semantic_sha256': confirmation['semantic_sha256'], 'evaluation_plan': str(evaluation_plan.resolve()),
        'evaluation_plan_sha256': hashlib.sha256(evaluation_plan.read_bytes()).hexdigest(),
        'evaluation_variant': variant, 'evaluation_settings': settings,
        'source_population_identity': source, 'training_reports_sha256': provenance,
        'selection': 'terminal empirically fitted actions; no zero checkpoint fallback or evaluation selection',
        'exact_global_optimum_certified': False, 'outer_or_fresh_evaluation_arrays_read': False, 'folds': []}
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'terminal-policies.json'
    with FileLock(str(output/'freeze.lock'), timeout=0):
        if destination.exists():
            previous = json.loads(destination.read_text())
            if (previous['training_reports_sha256'] != provenance
                or previous['evaluation_plan_sha256'] != result['evaluation_plan_sha256']):
                raise ValueError('immutable full-gradient freeze inputs changed')
            for fold in previous['folds']:
                for selected in fold['policies'].values():
                    for field in ('weights', 'normalization'):
                        if hashlib.sha256(Path(selected[field]).read_bytes()).hexdigest() != selected[field+'_sha256']:
                            raise ValueError('an immutable full-gradient action artifact changed')
            return previous
        expected_ranges = fold_path_ranges(source['config']['paths'])
        for number, (directory, report) in enumerate(zip(folds, reports, strict=True)):
            expected = expected_ranges[number]
            if (report.get('status') != 'both independent full-gradient policies fitted; inference pending'
                or report['outer_training_paths'] != [expected.start, expected.stop]
                or report['inference_performed'] or report['outer_holdout_read']):
                raise ValueError('a full-gradient training fold is incomplete or its inference separation changed')
            if (report['affine_config'] != controls['affine'] or report['neural_config'] != controls['neural']
                or report['identity']['protocol'] != controls['protocol']):
                raise ValueError('fitted full-gradient controls differ from their confirmation declaration')
            for filename, digest in report['identity']['implementation_sha256'].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                    raise ValueError('full-gradient policy implementation changed since fitting')
            normalization = (directory/'normalization.npz').read_bytes()
            if hashlib.sha256(normalization).hexdigest() != report['identity']['normalization_sha256']:
                raise ValueError('full-gradient training normalization changed')
            with np.load(io.BytesIO(normalization), allow_pickle=False) as saved:
                mask = saved['inner_training_mask']
                if (len(mask) != expected.stop-expected.start or not np.array_equal(mask[::2], mask[1::2])
                    or int(mask.sum()) != report['inner_training_paths']):
                    raise ValueError('training split no longer preserves declared complete path pairs')
            policies = {}
            for name in DIAGNOSTICS:
                fit = report['diagnostics'][name]
                if (not fit['affine']['converged'] or not fit['neural']['converged']
                    or fit['neural']['history'][-1]['training_loss'] >= 0.
                    or fit['inner_validation_used_for_optimization']):
                    raise ValueError('cannot freeze a failed, zero or evaluation-selected fit')
                content = (directory/f'{name}-weights.pt').read_bytes()
                if hashlib.sha256(content).hexdigest() != fit['weights_sha256']:
                    raise ValueError('terminal action weights changed after their completed training report')
                saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
                if (saved['identity'] != report['identity'] or saved['diagnostic'] != name
                    or saved['code_version'] != LocalActionConfig.version.unique_id48
                    or saved.get('input_normalization_dtype') != 'float32'
                    or saved['policy_state_source'] != 'terminal trained model, not best_state'
                    or any(not torch.isfinite(value).all() for value in saved['state_dict'].values())):
                    raise ValueError('trained terminal action differs from its completed training report')
                weights_path = output/f'fold-{number}-{name}.pt'
                normalization_path = output/f'fold-{number}-{name}-normalization.npz'
                weights_path.write_bytes(content)
                normalization_path.write_bytes(normalization)
                policies[name] = {'weights': str(weights_path.resolve()),
                    'weights_sha256': hashlib.sha256(content).hexdigest(),
                    'normalization': str(normalization_path.resolve()),
                    'normalization_sha256': hashlib.sha256(normalization).hexdigest(),
                    'source_report': str(directory/'training-report.json'),
                    'terminal_training_loss': fit['neural']['history'][-1]['training_loss'],
                    'terminal_inner_validation_loss': fit['inner_validation_loss'],
                    'numerical_local_stationarity_met': True, 'exact_global_optimum_certified': False}
            result['folds'].append({'fold': number, 'policies': policies})
        write_json(destination, result)
    return result
