"""Immutable, accurately labelled artifacts for validation-selected actions."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    fold_path_ranges,
)
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.terminal_audit import validated_evaluation_plan
from quantdata.gromm2.terminal_inference import FrozenTerminalAction
from quantdata.gromm2.validation_action_optimization import SELECTION_SOURCE
from quantdata.gromm2.validation_selected_audit import selection_source_hashes
from quantdata.torch import BaseEnvironment


def selected_confirmation_plan(path: Path) -> dict:
    """Check simulator/scoring declarations plus the separate selection contract."""
    plan = validated_evaluation_plan(path)
    content = Path(plan['selection_declaration']).read_bytes()
    if hashlib.sha256(content).hexdigest() != plan['selection_declaration_sha256']:
        raise ValueError('declared validation-selection protocol changed')
    declaration = json.loads(content)
    if (declaration['schema'] != 'gromm2-validation-selection-declaration-v1'
        or declaration['implementation_sha256'] != selection_source_hashes()
        or declaration['independent_fresh_confirmation_only'] is not True):
        raise ValueError('validation-selected implementation or independence contract changed')
    return plan


def _check_normalization_identity(content: bytes, *, fold: int, source: dict, declaration: dict) -> None:
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        identity = json.loads(str(saved['identity'].item()))
    if (identity['fold'] != fold or json.loads(identity['path_identity']) != source
        or identity['protocol'] != declaration['controls']['protocol']
        or hashlib.sha256(identity['path_identity'].encode()).hexdigest() != declaration['source_population_identity_sha256']):
        raise ValueError('normalization fold, development population or training protocol differs from its declaration')


def _checked_candidate(directory: Path, *, report: dict, name: str, declaration: dict) -> tuple[bytes, dict]:
    fitted = report['diagnostics'][name]
    if fitted['status'] != 'selection_complete' or not fitted['initializer']['affine_report']['converged']:
        raise ValueError('cannot freeze an incomplete full-class selection')
    result = fitted['result']
    identity = result['identity']
    initializer_sha256 = hashlib.sha256(json.dumps(fitted['initializer'], sort_keys=True).encode()).hexdigest()
    bound = hashlib.sha256(json.dumps({'fold_data_identity': report['data_identity'],
        'diagnostic': name, 'initializer_sha256': initializer_sha256}, sort_keys=True).encode()).hexdigest()
    if (result['confirmation_read'] is not False or result['policy_state_source'] != SELECTION_SOURCE
        or result['inner_validation_used_for_selection'] is not True
        or fitted['initializer_sha256'] != initializer_sha256
        or fitted['policy_data_identity'] != bound or identity['data_identity'] != bound
        or identity['diagnostic'] != name or identity['optimizer_config'] != declaration['controls']['optimizer']
        or identity['selection_config'] != declaration['controls']['selection']):
        raise ValueError('selected policy identity, initializer or controls changed')
    native = torch.load(directory/name/'resume.pt', map_location='cpu', weights_only=False)
    selection = result['selection']
    if native['identity'] != identity or native['selection'] != selection:
        raise ValueError('selection report differs from the committed native ledger')
    history = native['native_state']['history']
    terminal = history[-1]
    numerical = result['numerical']
    if (numerical['history'] != history or numerical['full_objective_evaluations'] != native['native_state']['full_objective_evaluations']
        or numerical['converged'] != native['native_state']['empirical_local_stationarity']):
        raise ValueError('terminal numerical status differs from its native accepted checkpoint')
    completed = (selection['stop_reason'] == 'declared_inner_validation_patience'
        or numerical['converged']
        or (numerical['stop_reason'] == 'work_limit_not_converged'
            and terminal['iteration'] == declaration['controls']['optimizer']['neural']['maximum_iterations']))
    if not completed or numerical['stop_reason'] == 'line_search_not_converged':
        raise ValueError('selection did not finish its declared stopping rule')
    controls = declaration['controls']['selection']
    eligible = [entry for entry in selection['history'] if entry['iteration'] >= controls['minimum_updates']
        and math.isfinite(entry['value']) and math.isfinite(entry['mean_squared_unit_action'])
        and entry['mean_squared_unit_action'] > 0.]
    if not eligible:
        raise ValueError('no genuinely trained nonzero selected action')
    best = min(eligible, key=lambda entry: (entry['value'], entry['iteration']))
    content = Path(result['selected_weights']).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if (digest != result['selected_weights_sha256'] or digest != selection['best_sha256']
        or Path(result['selected_weights']).resolve() != (directory/name/selection['best_filename']).resolve()
        or best['iteration'] != selection['best_iteration'] or best['value'] != selection['best_value']):
        raise ValueError('selected candidate differs from the declared best validation checkpoint')
    saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    if (saved['identity'] != identity or saved['selected'] != best or saved['diagnostic'] != name
        or saved['policy_state_source'] != SELECTION_SOURCE
        or saved['code_version'] != LocalActionConfig.version.unique_id48
        or saved['input_normalization_dtype'] != 'float32'
        or saved['user_version'] != saved['model_config']['user_version']
        or any(not bool(torch.isfinite(value).all()) for value in saved['state_dict'].values())):
        raise ValueError('selected weights interpretation, version or numerical values changed')
    return content, {'selected': best, 'policy_data_identity': bound, 'initializer_sha256': initializer_sha256,
        'selected_numerical_stationarity_met': best['empirical_local_stationarity'],
        'terminal_numerical_stationarity_met': numerical['converged'],
        'terminal_stop_reason': numerical['stop_reason'], 'terminal_iteration': terminal['iteration'],
        'terminal_full_objective_evaluations': numerical['full_objective_evaluations'],
        'global_optimum_certified': False}


def freeze_selected_actions(*, folds: tuple[Path, Path], output: Path,
                            evaluation_plan: Path, variant: str) -> dict:
    """Freeze all four genuine policies without reading any path/return arrays."""
    plan = selected_confirmation_plan(evaluation_plan)
    declaration = json.loads(Path(plan['selection_declaration']).read_text())
    settings = plan['variants'][variant]
    reports = [json.loads((directory/'training-report.json').read_text()) for directory in folds]
    if [report['identity']['fold'] for report in reports] != [0, 1]:
        raise ValueError('two ordered fitted development halves are required')
    source = reports[0]['identity']['source_population_identity']
    if (any(report['identity']['source_population_identity'] != source for report in reports)
        or source['config']['paths'] != settings['paths'] or source['config']['substeps'] != settings['substeps']
        or source['config']['batch_size'] != settings['batch_size'] or source['config']['seed'] == settings['seed']):
        raise ValueError('selected confirmation population geometry or independent seed changed')
    provenance = {str(directory/'training-report.json'): hashlib.sha256((directory/'training-report.json').read_bytes()).hexdigest()
        for directory in folds}
    manifest = {'schema': 'gromm2-validation-selected-actions-v1', 'protocol': plan['protocol'],
        'protocol_sha256': plan['protocol_sha256'], 'semantic_sha256': plan['semantic_sha256'],
        'implementation_sha256': selection_source_hashes(), 'evaluation_plan': str(evaluation_plan.resolve()),
        'evaluation_plan_sha256': hashlib.sha256(evaluation_plan.read_bytes()).hexdigest(),
        'evaluation_variant': variant, 'evaluation_settings': settings,
        'source_population_identity': source, 'training_reports_sha256': provenance,
        'source_population_identity_sha256': declaration['source_population_identity_sha256'],
        'independent_fresh_confirmation_only': True, 'outer_or_fresh_evaluation_arrays_read': False,
        'policy_state_source': SELECTION_SOURCE, 'global_optimum_certified': False, 'folds': []}
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'selected-policies.json'
    with FileLock(str(output/'freeze.lock'), timeout=0):
        if destination.exists():
            previous = json.loads(destination.read_text())
            if (previous['training_reports_sha256'] != provenance
                or previous['evaluation_plan_sha256'] != manifest['evaluation_plan_sha256']):
                raise ValueError('immutable selection freeze inputs changed')
            load_selected_actions(destination, device='cpu')
            return previous
        expected_ranges = fold_path_ranges(source['config']['paths'])
        for number, (directory, report) in enumerate(zip(folds, reports, strict=True)):
            identity = report['identity']
            expected = expected_ranges[number]
            if (report['status'] != 'both independent full-class selections complete; fresh confirmation pending'
                or report['outer_training_paths'] != [expected.start, expected.stop]
                or report['outer_holdout_read'] is not False or report['inference_performed'] is not False
                or report['controls'] != declaration['controls']
                or identity['declaration_sha256'] != plan['selection_declaration_sha256']
                or Path(identity['declaration']).resolve() != Path(plan['selection_declaration']).resolve()
                or identity['source_population_identity_sha256'] != declaration['source_population_identity_sha256']
                or identity['implementation_sha256'] != manifest['implementation_sha256']):
                raise ValueError('selection fold completion, declaration or source contract changed')
            if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != report['data_identity']:
                raise ValueError('selection report data identity changed')
            normalization = (directory/'normalization.npz').read_bytes()
            if hashlib.sha256(normalization).hexdigest() != identity['normalization_sha256']:
                raise ValueError('selection normalization changed')
            _check_normalization_identity(normalization, fold=number, source=source, declaration=declaration)
            with np.load(io.BytesIO(normalization), allow_pickle=False) as saved:
                mask = saved['inner_training_mask']
                if (mask.dtype != np.bool_ or len(mask) != expected.stop-expected.start
                    or not np.array_equal(mask[::2], mask[1::2]) or int(mask.sum()) != report['inner_training_paths']
                    or int((~mask).sum()) != report['inner_validation_paths']):
                    raise ValueError('selection partition no longer preserves complete path pairs')
            policies = {}
            for name in DIAGNOSTICS:
                content, evidence = _checked_candidate(directory, report=report, name=name, declaration=declaration)
                weights_path, normalization_path = output/f'fold-{number}-{name}.pt', output/f'fold-{number}-{name}.npz'
                weights_path.write_bytes(content)
                normalization_path.write_bytes(normalization)
                policies[name] = {**evidence, 'weights': str(weights_path.resolve()),
                    'weights_sha256': hashlib.sha256(content).hexdigest(), 'normalization': str(normalization_path.resolve()),
                    'normalization_sha256': hashlib.sha256(normalization).hexdigest()}
            manifest['folds'].append({'fold': number, 'training_identity': identity,
                'data_identity': report['data_identity'], 'policies': policies})
        write_json(destination, manifest)
    return manifest


def load_selected_actions(manifest: Path, *, device: str) -> tuple[dict, dict[str, FrozenTerminalAction]]:
    """Load only explicitly selected weights, preserving F32-normalized inputs."""
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()):
        raise ValueError('selected inference requires explicit CPU or managed CUDA device')
    frozen = json.loads(manifest.read_text())
    if frozen['schema'] != 'gromm2-validation-selected-actions-v1' or frozen['policy_state_source'] != SELECTION_SOURCE:
        raise ValueError('unknown selected-action schema or interpretation')
    plan_path = Path(frozen['evaluation_plan'])
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != frozen['evaluation_plan_sha256']:
        raise ValueError('immutable selected confirmation plan changed')
    plan = selected_confirmation_plan(plan_path)
    declaration = json.loads(Path(plan['selection_declaration']).read_text())
    if (frozen['semantic_sha256'] != plan['semantic_sha256'] or frozen['implementation_sha256'] != selection_source_hashes()
        or frozen['protocol_sha256'] != plan['protocol_sha256']
        or Path(frozen['protocol']).resolve() != Path(plan['protocol']).resolve()
        or frozen['evaluation_settings'] != plan['variants'][frozen['evaluation_variant']]
        or frozen['independent_fresh_confirmation_only'] is not True
        or frozen['source_population_identity_sha256'] != declaration['source_population_identity_sha256']
        or frozen['outer_or_fresh_evaluation_arrays_read'] is not False
        or [fold['fold'] for fold in frozen['folds']] != [0, 1]):
        raise ValueError('selected declaration, implementation or complete-fold contract changed')
    actions = {}
    for fold in frozen['folds']:
        identity = fold['training_identity']
        if (identity['fold'] != fold['fold']
            or identity['source_population_identity'] != frozen['source_population_identity']
            or identity['source_population_identity_sha256'] != frozen['source_population_identity_sha256']
            or identity['protocol'] != declaration['controls']['protocol']
            or identity['implementation_sha256'] != frozen['implementation_sha256']
            or identity['declaration_sha256'] != plan['selection_declaration_sha256']
            or Path(identity['declaration']).resolve() != Path(plan['selection_declaration']).resolve()
            or hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != fold['data_identity']):
            raise ValueError('selected fold identity differs from its fitted development half')
        if set(fold['policies']) != set(DIAGNOSTICS):
            raise ValueError('both diagnostics are required for each selected fold')
        for name in DIAGNOSTICS:
            selected = fold['policies'][name]
            content = {}
            for field in ('weights', 'normalization'):
                content[field] = Path(selected[field]).read_bytes()
                if hashlib.sha256(content[field]).hexdigest() != selected[field+'_sha256']:
                    raise ValueError('immutable selected action artifact changed')
            if selected['normalization_sha256'] != identity['normalization_sha256']:
                raise ValueError('selected action normalization differs from its fitted half')
            _check_normalization_identity(content['normalization'], fold=fold['fold'],
                source=frozen['source_population_identity'], declaration=declaration)
            bound = hashlib.sha256(json.dumps({'fold_data_identity': fold['data_identity'],
                'diagnostic': name, 'initializer_sha256': selected['initializer_sha256']}, sort_keys=True).encode()).hexdigest()
            if selected['policy_data_identity'] != bound:
                raise ValueError('selected policy belongs to a different fitted fold or objective')
            saved = torch.load(io.BytesIO(content['weights']), map_location='cpu', weights_only=False)
            fields = dict(saved['model_config'])
            fields['network'] = ActionNetworkConfig(**fields['network'])
            config = LocalActionConfig(**fields)
            if (saved['code_version'] != LocalActionConfig.version.unique_id48 or saved['user_version'] != config.user_version
                or saved['policy_state_source'] != SELECTION_SOURCE or saved['diagnostic'] != name
                or saved['identity']['data_identity'] != selected['policy_data_identity']
                or saved['selected'] != selected['selected'] or saved['input_normalization_dtype'] != 'float32'
                or any(value.dtype != torch.float64 or not bool(torch.isfinite(value).all())
                    for value in saved['state_dict'].values() if value.is_floating_point())):
                raise ValueError('selected model version, precision, selection record or interpretation changed')
            env = BaseEnvironment(device=device, precision=64)
            model = AffineDeepAction(env, config=config)
            model.load_state_dict(saved['state_dict'], strict=True)
            model.affine_only = False
            model.requires_grad_(False)
            model.eval()
            with np.load(io.BytesIO(content['normalization']), allow_pickle=False) as values:
                mean, scale = np.asarray(values['mean']), np.asarray(values['scale'])
                if (float(values['action_scale']) != config.action_scale or mean.shape != (config.feature_count,)
                    or scale.shape != mean.shape or not np.isfinite([mean, scale]).all() or np.any(scale <= 0.)):
                    raise ValueError('invalid selected-action normalization or original-unit scale')
                actions[f'fold-{fold["fold"]}/{name}'] = FrozenTerminalAction(model=model,
                    mean=torch.as_tensor(mean, dtype=torch.float32, device=env.device),
                    scale=torch.as_tensor(scale, dtype=torch.float32, device=env.device),
                    action_scale=config.action_scale, squared_lambda=config.squared_lambda)
    if len({action.squared_lambda for action in actions.values()}) != 1:
        raise ValueError('selected ensemble objectives require one declared squared lambda')
    return frozen, actions
