"""Freeze trained nonzero companion policies without reading evaluation paths."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.gromm2.conditional_audit import DIAGNOSTICS, LocalActionConfig
from quantdata.gromm2.rolling_calibration import write_json


def terminal_semantic_hashes() -> dict[str, str]:
    """Pin model-forward, simulator, loader and scoring source dependencies."""
    directory = Path(__file__).resolve().parent
    modules = ('terminal_audit', 'terminal_inference', 'conditional_audit',
        'conditional_refinement', 'local_audit_training', 'market_training_runtime',
        'forward_hedging', 'path_audit', 'rolling_audit', 'rolling_calibration',
        'runtime', 'bridge', 'dlv', 'simulation', 'tridiagonal', 'pricing', 'dynamics', 'geometry')
    modules += ('affine_audit_optimization', 'neural_audit_optimization', 'full_gradient_audit', 'audit_host_cache')
    paths = [directory / f'{module}.py' for module in modules]
    paths.extend((directory.parent / 'torch' / '__init__.py', directory.parent / 'torch' / 'ffw.py',
                  directory.parent / 'jax' / 'tridiagonal.py'))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validated_evaluation_plan(path: Path) -> dict:
    """Reject undeclared confirmation settings or changed evaluation semantics."""
    plan = json.loads(path.read_text())
    if plan['schema'] != 'gromm2-terminal-confirmation-plan-v1' or plan['familywise_endpoint_count'] != 12:
        raise ValueError('unknown terminal confirmation plan')
    if hashlib.sha256(Path(plan['protocol']).read_bytes()).hexdigest() != plan['protocol_sha256']:
        raise ValueError('declared confirmation protocol changed')
    if terminal_semantic_hashes() != plan['semantic_sha256']:
        raise ValueError('declared confirmation semantic dependencies changed')
    if set(plan['variants']) != {'one_step', 'eight_step'}:
        raise ValueError('confirmation plan must declare both discretizations')
    for name, substeps in (('one_step', 1), ('eight_step', 8)):
        variant = plan['variants'][name]
        if (variant['substeps'] != substeps or type(variant['seed']) is not int
            or type(variant['paths']) is not int or variant['paths'] < 4 or variant['paths'] % 2
            or type(variant['batch_size']) is not int or variant['batch_size'] < 2 or variant['batch_size'] % 2):
            raise ValueError('invalid declared confirmation population')
    if plan['variants']['one_step']['seed'] == plan['variants']['eight_step']['seed']:
        raise ValueError('confirmation populations require distinct seeds')
    return plan


def terminal_trial_selection(reports: list[dict]) -> dict[str, int]:
    """Select terminal losses on internal validation, never an outer score.

    The explicit zero checkpoint remains a loss reference, not the action in
    this separately declared companion screen. A nonzero terminal training
    loss is sufficient evidence that its action is not identically zero on
    training observations. Plateau stopping is not a stationary-optimum claim.
    """
    if not reports:
        raise ValueError('terminal selection needs every declared trial')
    for report in reports:
        controls, history = report['train_config'], report['history']
        if report['stop_reason'] != 'internal_validation_plateau' or not history:
            raise ValueError('all declared terminal trials must finish their stopping rule')
        minimum = controls['affine_epochs'] + controls['minimum_full_epochs']
        patience = controls['convergence_patience']
        if report['epoch'] < minimum or history[-1]['epoch'] != report['epoch'] or len(history) < patience:
            raise ValueError('terminal trial did not complete its minimum training work')
        for name in DIAGNOSTICS:
            active = [entry for entry in history if name in entry['learning_rates_used']]
            if len(active) < patience or sum(entry['full_network'] for entry in active) < controls['minimum_full_epochs']:
                raise ValueError('terminal trial did not finish each independent policy phase')
            for entry in active[-patience:]:
                rates = entry['learning_rates_used'][name]
                if not entry['full_network'] or not rates or any(
                    rate > controls['minimum_learning_rate'] * (1. + 1.e-12) for rate in rates
                ):
                    raise ValueError('terminal trial lacks actual learning-rate-floor convergence work')
    selected = {}
    for name in DIAGNOSTICS:
        eligible = [index for index, report in enumerate(reports)
            if math.isfinite(report['history'][-1]['training'][name])
            and report['history'][-1]['training'][name] != 0.
            and math.isfinite(report['history'][-1]['internal_validation'][name])]
        if not eligible:
            raise ValueError(f'{name}: no informative nonzero trained terminal candidate')
        selected[name] = min(eligible, key=lambda index: (reports[index]['history'][-1]['internal_validation'][name], index))
    return selected


def freeze_terminal_actions(*, trials: Path, output: Path, protocol: Path, evaluation_plan: Path,
                            expected_trials: int = 4) -> dict:
    """Freeze both fitted-fold actions using only completed training artifacts.

    No population arrays are opened. Fresh evaluation must be generated after
    this manifest is published. Each exported state is the terminal trained
    state, not the source checkpoint's potentially zero ``best_state``.
    """
    plan = json.loads((trials / 'frozen-policies.json').read_text())
    confirmation = validated_evaluation_plan(evaluation_plan)
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if confirmation['protocol_sha256'] != protocol_hash or Path(confirmation['protocol']).resolve() != protocol.resolve():
        raise ValueError('freezing protocol differs from declared confirmation protocol')
    variant = 'eight_step' if plan['source_identity']['config']['substeps'] == 8 else 'one_step'
    settings = confirmation['variants'][variant]
    source_config = plan['source_identity']['config']
    if (source_config['substeps'] != settings['substeps'] or source_config['batch_size'] != settings['batch_size']
        or source_config['seed'] == settings['seed']):
        raise ValueError('declared confirmation differs from the source population geometry')
    if not isinstance(expected_trials, int) or isinstance(expected_trials, bool) or expected_trials < 1:
        raise ValueError('expected_trials must be a positive integer')
    if len(plan['candidate_controls']) != expected_trials:
        raise ValueError('declared candidate controls differ from expected trials')
    if plan['status'] != 'policies_frozen; inference not yet performed' or [entry['fold'] for entry in plan['folds']] != [0, 1]:
        raise ValueError('both outer training folds must be complete before freezing')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'terminal-policies.json'
    confirmation_hash = hashlib.sha256(evaluation_plan.read_bytes()).hexdigest()
    source_plan_hash = hashlib.sha256((trials / 'frozen-policies.json').read_bytes()).hexdigest()
    with FileLock(str(output / 'freeze.lock')):
        if destination.exists():
            previous = json.loads(destination.read_text())
            if (previous['protocol_sha256'] != protocol_hash or previous['training_manifest_sha256'] != source_plan_hash
                or previous['evaluation_plan_sha256'] != confirmation_hash
                or previous['semantic_sha256'] != confirmation['semantic_sha256']):
                raise RuntimeError('terminal freeze identity changed; preserve the existing manifest')
            for fold in previous['folds']:
                for policy in fold['policies'].values():
                    for field in ('weights', 'normalization'):
                        if hashlib.sha256(Path(policy[field]).read_bytes()).hexdigest() != policy[field + '_sha256']:
                            raise RuntimeError('an immutable frozen policy artifact changed')
            return previous
        result = {'schema': 'gromm2-terminal-frozen-actions-v2', 'protocol': str(protocol.resolve()),
            'protocol_sha256': protocol_hash, 'training_manifest_sha256': source_plan_hash,
            'evaluation_plan': str(evaluation_plan.resolve()), 'evaluation_plan_sha256': confirmation_hash,
            'evaluation_variant': variant, 'evaluation_settings': settings,
            'semantic_sha256': confirmation['semantic_sha256'],
            'source_population_identity': plan['source_identity'], 'expected_trials_per_fold': expected_trials,
            'selection': 'terminal internal-validation loss, deterministic trial-index tie break',
            'zero_baseline_endpoint_replaced': False, 'outer_or_fresh_evaluation_arrays_read': False,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'folds': []}
        for fold in plan['folds']:
            if len(fold['reports']) != expected_trials:
                raise ValueError('missing a declared terminal trial')
            if len({Path(path).resolve() for path in fold['reports']}) != expected_trials:
                raise ValueError('terminal trial checkpoint paths must be distinct')
            snapshots, reports = [], []
            for index, filename in enumerate(fold['reports']):
                path = Path(filename)
                content = path.read_bytes()
                saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
                report = saved['report']
                if saved['code_version'] != LocalActionConfig.version.unique_id48:
                    raise ValueError('terminal checkpoint model version differs from the implementation')
                if (report['fold'] != fold['fold'] or set(saved['state']['stopped']) != set(DIAGNOSTICS)
                    or not all(saved['state']['stopped'].values()) or saved['state']['next_batch'] != 0):
                    raise ValueError('terminal checkpoint is incomplete or belongs to the wrong fold')
                if report['path_identity_sha256'] != plan['source_identity_sha256']:
                    raise ValueError('terminal checkpoint belongs to another population')
                encoded = json.loads(saved['identity'])
                if (encoded['fold'] != fold['fold'] or encoded['protocol'] != plan['protocol']
                    or encoded['protocol_version'] != plan['protocol_version']
                    or hashlib.sha256(encoded['path_identity'].encode()).hexdigest() != plan['source_identity_sha256']):
                    raise ValueError('terminal checkpoint protocol or population identity changed')
                if report['train_config'] != plan['candidate_controls'][index]:
                    raise ValueError('terminal trial does not match its declared candidate controls')
                for name in DIAGNOSTICS:
                    active_epoch = max(entry['epoch'] for entry in report['history'] if name in entry['learning_rates_used'])
                    floor = saved['state']['floor_since'][name]
                    if floor is None or active_epoch - max(floor, saved['state']['last_improvement'][name],
                        report['train_config']['affine_epochs']) < report['train_config']['convergence_patience']:
                        raise ValueError('terminal checkpoint lacks its declared no-improvement convergence work')
                for filename, digest in report['implementation_sha256'].items():
                    if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
                        raise ValueError('terminal model implementation changed since fitting')
                snapshots.append((path, content, saved))
                reports.append(report)
            chosen = terminal_trial_selection(reports)
            policies = {}
            for name, index in chosen.items():
                path, content, saved = snapshots[index]
                report = reports[index]
                prefix = f'policies.{name}.'
                state = {key.removeprefix(prefix): value for key, value in saved['model'].items() if key.startswith(prefix)}
                if not state or any(not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError('nonfinite or missing terminal policy weights')
                normalization = path.parent.parent / 'normalization.npz'
                with np.load(normalization, allow_pickle=False) as values:
                    if str(values['identity'].item()) != saved['identity']:
                        raise ValueError('terminal normalization identity differs from its fitted checkpoint')
                    action_scale = float(values['action_scale'])
                    if not math.isfinite(action_scale) or action_scale != report['model_config']['action_scale']:
                        raise ValueError('terminal policy normalization scale changed')
                    if (values['mean'].shape != (report['model_config']['feature_count'],)
                        or values['scale'].shape != values['mean'].shape
                        or not np.isfinite(values['mean']).all() or not np.isfinite(values['scale']).all()
                        or np.any(values['scale'] <= 0.) or not float(values['action_scale']) > 0.):
                        raise ValueError('invalid terminal policy normalization')
                weights_path = output / f'fold-{fold["fold"]}-{name}.pt'
                temporary = weights_path.with_suffix('.tmp')
                torch.save({'model_config': report['model_config'], 'code_version': saved['code_version'],
                    'user_version': saved['user_version'], 'diagnostic': name, 'state_dict': state,
                    'input_normalization_dtype': f"float{report['train_config']['precision']}",
                    'policy_state_source': 'terminal trained model, not best_state'}, temporary)
                temporary.replace(weights_path)
                normalization_path = output / f'fold-{fold["fold"]}-{name}-normalization.npz'
                normalization_path.write_bytes(normalization.read_bytes())
                policies[name] = {'trial_index': index, 'source_checkpoint': str(path),
                    'source_checkpoint_sha256': hashlib.sha256(content).hexdigest(),
                    'weights': str(weights_path.resolve()), 'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                    'normalization': str(normalization_path.resolve()),
                    'normalization_sha256': hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
                    'terminal_epoch': report['epoch'], 'terminal_training_loss': report['history'][-1]['training'][name],
                    'terminal_last_active_epoch': max(entry['epoch'] for entry in report['history'] if name in entry['learning_rates_used']),
                    'terminal_internal_validation_loss': report['history'][-1]['internal_validation'][name],
                    'original_best_epoch': report['best_epochs'][name],
                    'original_zero_checkpoint_selected': report['best_epochs'][name] == 0,
                    'optimizer_stationarity_certified': False,
                    'nonzero_evidence': 'finite nonzero exact terminal inner-training objective',
                    'train_config': report['train_config'], 'actual_epoch_controls': report['history'],
                    'effective_optimizer_controls': report['effective_optimizer_controls'],
                    'stopping_evidence': {key: saved['state'][key] for key in
                                          ('stopped', 'floor_since', 'last_improvement', 'next_batch')}}
            result['folds'].append({'fold': fold['fold'], 'policies': policies,
                'candidate_checkpoints': [{'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                    'terminal_epoch': saved['report']['epoch'], 'train_config': saved['report']['train_config']}
                    for path, content, saved in snapshots],
                'candidate_terminal_losses': {name: [report['history'][-1]['internal_validation'][name]
                                                   for report in reports] for name in DIAGNOSTICS}})
        write_json(destination, result)
        return result
