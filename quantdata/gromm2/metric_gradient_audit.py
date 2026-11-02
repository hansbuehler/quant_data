"""Full-class conditional diagnostics with a training-only SPD optimizer metric."""

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
from quantdata.gromm2.full_gradient_audit import (
    _adopt_affine_initializer,
    _atomic_torch_save,
    freeze_full_gradient_actions,
)
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _normalization,
    _path_identity,
    _resident_fold,
    internal_training_mask,
)
from quantdata.gromm2.metric_audit_optimization import (
    AffineBlockMetric,
    MetricLBFGS,
    MetricSolveConfig,
    OptimizerMigration,
    fit_metric_action,
)
from quantdata.gromm2.neural_audit_optimization import full_action_pass, full_gradient_norm
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.terminal_audit import validated_evaluation_plan
from quantdata.torch import BaseEnvironment


def validate_migration_resolution(*, resolution: dict, plan: dict) -> None:
    """Check actual guard failure and accepted-state provenance before allocation."""
    previous = Path(plan['preceding_plan']).parent
    report_path = previous/'eight_step/fold-0/training-report.json'
    checkpoint = previous/'eight_step/fold-0/squared-neural.pt'
    report_content, checkpoint_content = report_path.read_bytes(), checkpoint.read_bytes()
    plan_content = Path(plan['preceding_plan']).read_bytes()
    if (resolution['preceding_plan_sha256'] != plan['preceding_plan_sha256']
        or hashlib.sha256(plan_content).hexdigest() != plan['preceding_plan_sha256']
        or resolution['preceding_report'] != str(report_path)
        or resolution['preceding_report_sha256'] != hashlib.sha256(report_content).hexdigest()
        or set(resolution['specifications']) != {'squared'}):
        raise ValueError('resolved migration source plan, report or diagnostics changed')
    specification = resolution['specifications']['squared']
    if (specification['checkpoint'] != str(checkpoint) or specification['diagnostic'] != 'squared'
        or specification['checkpoint_sha256'] != hashlib.sha256(checkpoint_content).hexdigest()):
        raise ValueError('resolved migration accepted checkpoint changed')
    report, source_plan = json.loads(report_content), json.loads(plan_content)
    fitted = report['diagnostics']['squared']['neural']
    saved = torch.load(io.BytesIO(checkpoint_content), map_location='cpu', weights_only=False)
    state = saved['state']
    if (fitted['converged'] or fitted['stop_reason'] != 'work_limit_not_converged'
        or fitted['history'][-1]['iteration'] != source_plan['controls']['neural']['maximum_iterations']
        or state['history'] != fitted['history'] or saved['identity'] != report['identity']
        or state['closure_evaluations'] != fitted['history'][-1]['closure_evaluations']
        or state['empirical_local_stationarity']
        or report['identity']['source_population_identity_sha256'] != plan['source_population_identity_sha256']):
        raise ValueError('resolved migration is not the matching original actual work-guard failure')


def _adopt_neural_checkpoint(*, specification: dict, policy: AffineDeepAction,
                             problem: AffineAuditProblem, identity: dict) -> tuple[OptimizerMigration, dict]:
    content = Path(specification['checkpoint']).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != specification['checkpoint_sha256']:
        raise ValueError('the declared source accepted checkpoint changed')
    saved = torch.load(io.BytesIO(content), map_location=problem.states.device, weights_only=False)
    source = saved['identity']
    for field in ('source_population_identity_sha256', 'normalization_sha256', 'fold', 'protocol', 'input_precision'):
        if source[field] != identity[field]:
            raise ValueError(f'optimizer migration changes {field}')
    state = saved['state']
    if (saved['code_version'] != LocalActionConfig.version.unique_id48
        or saved['user_version'] != policy.config.user_version or state['identity']['model_config'] != asdict(policy.config)
        or state['identity']['data_identity'] != hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        or state['identity']['diagnostic'] != specification['diagnostic']):
        raise ValueError('source optimizer checkpoint model or training identity differs')
    for filename, expected in source['implementation_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('source optimizer implementation changed before migration')
    policy.load_state_dict(saved['model'], strict=True)
    policy.requires_grad_(True)
    policy.affine_only = False
    policy.zero_grad(set_to_none=True)
    actual = float(full_action_pass(policy, problem, diagnostic=specification['diagnostic'], backward=True))
    norm = float(full_gradient_norm(policy))
    previous = state['history'][-1]
    if (not np.isclose(actual, previous['training_loss'], rtol=1.e-10, atol=1.e-11)
        or not np.isclose(norm, previous['full_gradient_norm'], rtol=1.e-9, atol=1.e-10)):
        raise ValueError('source checkpoint is not its declared actual accepted objective/gradient')
    torch.set_rng_state(saved['torch_rng'].cpu())
    if policy.affine.I0.weight.device.type == 'cuda' and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    migration = OptimizerMigration(accepted_checkpoint_sha256=digest, source_optimizer='torch.optim.LBFGS',
        initial_gradient_norm=state['initial_gradient_norm'], absolute_gradient_threshold=state['gradient_threshold'],
        prior_accepted_updates=previous['iteration'], prior_full_objective_passes=state['closure_evaluations'])
    return migration, {'specification': specification, 'verified_loss': actual, 'verified_gradient_norm': norm,
        'migration': asdict(migration), 'source_identity': source, 'source_curvature_history_discarded': True}


def fit_metric_gradient_fold(*, path: Path, output: Path, fold: int,
                             protocol: LocalAuditProtocol = LocalAuditProtocol(),
                             affine_config: AffineSolveConfig = AffineSolveConfig(),
                             metric_config: MetricSolveConfig = MetricSolveConfig(),
                             affine_initializer_root: Path | None = None,
                             migration_specs: dict[str, dict] | None = None,
                             device: str = 'cuda') -> dict:
    """Continue or initialize independent full policies without evaluation selection.

    The SPD metric is computed from inner-training states only. Every
    accepted update uses the unchanged full empirical entropy or squared
    objective. Native restarts preserve accepted weights and curvature;
    migration explicitly preserves the prior convergence threshold and work.
    """
    if fold not in (0, 1) or device not in ('cpu', 'cuda'):
        raise ValueError('an ordered fold and explicit compute device are required')
    if any(name not in DIAGNOSTICS or specification.get('diagnostic') != name
        for name, specification in (migration_specs or {}).items()):
        raise ValueError('migration specifications must preserve each named diagnostic')
    if device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('metric fitting requires a managed CUDA allocation')
    check_resource_headroom()
    source_identity, geometry = _path_identity(path)
    outer = fold_path_ranges(geometry['paths'])[fold]
    normalization_identity = json.dumps({'path_identity': source_identity, 'protocol': asdict(protocol),
        'fold': fold, 'protocol_version': LocalAuditProtocol.version.unique_id48}, sort_keys=True)
    mask = internal_training_mask(outer.stop-outer.start, protocol=protocol, fold=fold)
    output.mkdir(parents=True, exist_ok=True)
    stop, done = threading.Event(), threading.Event()
    started = checked = time.monotonic()
    progress = {'stage': 'initializing_metric_fold', 'fold': fold, 'completed_work_units': 0,
        'total_work_units': len(DIAGNOSTICS)}

    def observe(entry: dict) -> None:
        nonlocal checked
        if stop.is_set():
            raise InterruptedError('metric fit stopped; retain its last accepted checkpoint')
        progress.update(entry)
        now = time.monotonic()
        if now-checked >= 30. or entry.get('stage') in ('affine_newton_iteration', 'metric_full_gradient_iteration'):
            print(json.dumps({'event': 'metric_fold_progress', **progress,
                'elapsed_seconds': now-started, 'remaining_seconds': None}), flush=True)
            check_resource_headroom()
            checked = now

    def heartbeat() -> None:
        while not done.wait(60.):
            print(json.dumps({'event': 'metric_fold_heartbeat', **progress,
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
                initializer = torch.load(affine_initializer_root/'squared-affine-initializer.pt', map_location='cpu', weights_only=False)
                content = Path(initializer['plan']['normalization']).read_bytes()
                if hashlib.sha256(content).hexdigest() != initializer['plan']['normalization_sha256']:
                    raise ValueError('source affine normalization changed')
                temporary = output/'normalization.tmp'
                temporary.write_bytes(content)
                temporary.replace(output/'normalization.npz')
            if (output/'normalization.npz').exists():
                with np.load(output/'normalization.npz', allow_pickle=False) as saved:
                    stored_identity = str(saved['identity'].item())
                if json.loads(stored_identity) != json.loads(normalization_identity):
                    raise ValueError('metric normalization source changed')
                normalization_identity = stored_identity
            normalization = _normalization(path, outer=outer, mask=mask, protocol=protocol,
                output=output/'normalization.npz', identity=normalization_identity, upload_paths=128)
            module_paths = [Path(__file__).with_name(name+'.py') for name in (
                'metric_gradient_audit', 'metric_audit_optimization', 'full_gradient_audit',
                'affine_audit_optimization', 'neural_audit_optimization', 'conditional_audit',
                'local_audit_training', 'audit_host_cache')]
            module_paths.extend((Path(__file__).parents[1]/'torch'/'__init__.py', Path(__file__).parents[1]/'torch'/'ffw.py'))
            identity = {'source_population_identity_sha256': hashlib.sha256(source_identity.encode()).hexdigest(),
                'normalization_sha256': hashlib.sha256((output/'normalization.npz').read_bytes()).hexdigest(),
                'fold': fold, 'protocol': asdict(protocol),
                'implementation_sha256': {str(filename): hashlib.sha256(filename.read_bytes()).hexdigest() for filename in module_paths},
                'input_precision': 'original resident float32 normalization, float64 arithmetic',
                'source_population_identity': json.loads(source_identity), 'optimizer': 'fixed-SPD-full-class-LBFGS-v1',
                'migration_specs': migration_specs or {}}
            data_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            states, returns, eligible = _resident_fold(path, outer=outer, normalization=normalization,
                env=BaseEnvironment(device=device, precision=32),
                train_config=LocalAuditTrainConfig(device=device, upload_paths=512, host_cache=False),
                progress=progress, stop_requested=stop.is_set)
            inner = torch.as_tensor(mask, dtype=torch.bool, device=states.device)
            rows = torch.arange(len(states), device=states.device).reshape(-1, geometry['days'])
            arguments = {'states': states, 'returns': returns, 'eligible': eligible,
                'action_scale': float(normalization['action_scale']), 'squared_lambda': protocol.squared_lambda,
                'config': affine_config, 'progress': observe}
            training = AffineAuditProblem(rows=rows[inner].flatten(), **arguments)
            validation = AffineAuditProblem(rows=rows[~inner].flatten(), **arguments)
            training.prepare()
            report = {'identity': identity, 'data_identity': data_identity, 'outer_training_paths': [outer.start, outer.stop],
                'outer_holdout_read': False, 'inner_training_paths': int(mask.sum()),
                'inner_validation_paths': int((~mask).sum()), 'affine_config': asdict(affine_config),
                'neural_config': asdict(metric_config.neural), 'metric_config': asdict(metric_config),
                'diagnostics': {}, 'inference_performed': False}
            affine_initial_weights = None
            for index, name in enumerate(('squared', 'entropy')):
                observe({'stage': 'initializing_independent_metric_policy', 'diagnostic': name, 'completed_work_units': index})
                config = LocalActionConfig(label=f'full-gradient-fold-{fold}', feature_count=geometry['features'],
                    instrument_count=geometry['instruments'], action_scale=training.action_scale,
                    network=protocol.network, squared_lambda=protocol.squared_lambda)
                torch.manual_seed(protocol.split_seed+fold+1000*index)
                policy = AffineDeepAction(BaseEnvironment(device=device, precision=64), config=config)
                checkpoint_path = output/f'{name}-metric.pt'
                optimizer, resume, migration, provenance = None, None, None, None
                if checkpoint_path.exists():
                    saved = torch.load(checkpoint_path, map_location=states.device, weights_only=False)
                    if (saved['identity'] != identity or saved['code_version'] != LocalActionConfig.version.unique_id48
                        or saved['user_version'] != config.user_version):
                        raise ValueError('metric checkpoint training or model identity changed')
                    policy.load_state_dict(saved['model'], strict=True)
                    optimizer = MetricLBFGS(AffineBlockMetric(policy, training, deep_scale=metric_config.deep_metric_scale),
                        history_size=metric_config.neural.history_size, curvature_floor=metric_config.curvature_relative_floor)
                    optimizer.load_state_dict(saved['optimizer'])
                    resume, provenance = saved['state'], saved['initializer_provenance']
                    affine_report = saved['affine_report']
                    affine_initial_weights = saved['affine_initial_weights']
                    if resume['identity']['optimizer_migration'] is not None:
                        migration = OptimizerMigration(**resume['identity']['optimizer_migration'])
                    torch.set_rng_state(saved['torch_rng'].cpu())
                    if device == 'cuda' and saved['cuda_rng'] is not None:
                        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
                else:
                    if affine_initializer_root is not None:
                        affine_report, provenance = _adopt_affine_initializer(root=affine_initializer_root,
                            diagnostic=name, policy=policy, problem=training, identity=identity)
                    else:
                        affine_path = output/f'{name}-affine.pt'
                        initial = affine_initial_weights
                        if affine_path.exists():
                            saved = torch.load(affine_path, map_location=states.device, weights_only=False)
                            if saved['identity'] != identity:
                                raise ValueError('metric affine initializer identity changed')
                            initial = saved['weights']

                        def save_affine(weights: torch.Tensor, state: dict) -> None:
                            _atomic_torch_save(affine_path, {'identity': identity, 'weights': weights.cpu(), 'state': state})

                        weights, affine_report = training.solve(diagnostic=name, initial=initial, checkpoint=save_affine)
                        write_json(output/f'{name}-affine-report.json', affine_report)
                        if not affine_report['converged']:
                            raise RuntimeError('metric affine initializer has not converged')
                        with torch.no_grad():
                            policy.affine.I0.weight.copy_(weights[:-1].T)
                            policy.affine.I0.bias.copy_(weights[-1])
                    affine_initial_weights = torch.cat((policy.affine.I0.weight.detach().T,
                        policy.affine.I0.bias.detach()[None]), dim=0).clone()
                    if name in (migration_specs or {}):
                        migration, prior = _adopt_neural_checkpoint(specification=migration_specs[name],
                            policy=policy, problem=training, identity=identity)
                        provenance = {'affine': provenance, 'neural_migration': prior}
                    write_json(output/f'{name}-initializer-adoption.json', provenance)

                def save_neural(model: AffineDeepAction, current_optimizer: MetricLBFGS, state: dict) -> None:
                    _atomic_torch_save(checkpoint_path, {'identity': identity, 'model': model.state_dict(),
                        'optimizer': current_optimizer.state_dict(), 'state': state, 'initializer_provenance': provenance,
                        'affine_report': affine_report, 'affine_initial_weights': affine_initial_weights,
                        'code_version': LocalActionConfig.version.unique_id48,
                        'user_version': config.user_version, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None})

                neural_report = fit_metric_action(policy=policy, problem=training, diagnostic=name,
                    config=metric_config, optimizer=optimizer, resume_state=resume, migration=migration,
                    data_identity=data_identity, progress=observe, checkpoint=save_neural)
                diagnostics = {'affine': affine_report, 'neural': neural_report, 'initializer_provenance': provenance,
                    'inner_validation_loss': float(full_action_pass(policy, validation, diagnostic=name, backward=False)),
                    'inner_validation_used_for_optimization': False}
                report['diagnostics'][name] = diagnostics
                write_json(output/'training-report.json', report)
                if not neural_report['converged']:
                    raise RuntimeError('full metric policy has not met its raw and affine-metric gradient criteria')
                if neural_report['history'][-1]['training_loss'] >= 0.:
                    raise RuntimeError('zero-or-worse metric policy is not an informative trained diagnostic')
                _atomic_torch_save(output/f'{name}-weights.pt', {'model_config': asdict(config),
                    'code_version': LocalActionConfig.version.unique_id48, 'user_version': config.user_version,
                    'diagnostic': name, 'state_dict': policy.state_dict(), 'identity': identity,
                    'input_normalization_dtype': 'float32', 'policy_state_source': 'terminal trained model, not best_state'})
                diagnostics['weights_sha256'] = hashlib.sha256((output/f'{name}-weights.pt').read_bytes()).hexdigest()
            report['status'] = 'both independent full-gradient policies fitted; inference pending'
            write_json(output/'training-report.json', report)
            return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def freeze_metric_actions(*, folds: tuple[Path, Path], output: Path, evaluation_plan: Path, variant: str) -> dict:
    """Add metric-specific gates before using the shared immutable-action format."""
    confirmation = validated_evaluation_plan(evaluation_plan)
    for directory in folds:
        report = json.loads((directory/'training-report.json').read_text())
        if (report['metric_config'] != confirmation['metric_controls']
            or report['identity']['optimizer'] != 'fixed-SPD-full-class-LBFGS-v1'):
            raise ValueError('metric training controls differ from the declared confirmation protocol')
        for name in DIAGNOSTICS:
            fit = report['diagnostics'][name]['neural']
            last = fit['history'][-1]
            if (not fit['converged'] or last['full_gradient_norm'] > last['gradient_threshold']
                or last['affine_metric_gradient_norm'] > last['affine_metric_gradient_threshold']):
                raise ValueError('metric policy has not met both actual convergence criteria')
    return freeze_full_gradient_actions(folds=folds, output=output, evaluation_plan=evaluation_plan, variant=variant)
