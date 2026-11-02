"""Export four strictly verified native actions without changing scoring semantics."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem
from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    fold_path_ranges,
)
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256
from quantdata.gromm2.ggn_audit_finalization import _checked_bytes, _native_point_sha256
from quantdata.gromm2.local_audit_training import LocalAuditProtocol, internal_training_mask
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.strict_action_verification import PHASES, verify_strict_action
from quantdata.gromm2.terminal_audit import validated_evaluation_plan
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class StrictActionSource:
    """One native optimizer endpoint and its future immutable receipt path."""

    fold: int
    diagnostic: str
    run_plan: Path
    run_plan_sha256: str
    accepted_checkpoint: Path
    accepted_checkpoint_sha256: str
    receipt: Path

    def __post_init__(self) -> None:
        verify_inp(type(self.fold) is int and self.fold in (0, 1) and self.diagnostic in DIAGNOSTICS,
            lambda: 'a source must identify one of the four original fold/objective policies')
        for value in (self.run_plan_sha256, self.accepted_checkpoint_sha256):
            verify_inp(isinstance(value, str) and len(value) == 64
                and all(character in '0123456789abcdef' for character in value), lambda: 'invalid strict source SHA256')

    def declaration(self) -> dict:
        return {**asdict(self), 'run_plan': str(self.run_plan.resolve()),
            'accepted_checkpoint': str(self.accepted_checkpoint.resolve()), 'receipt': str(self.receipt.resolve())}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _strict_producers() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    modules = ('strict_action_bundle', 'strict_action_verification', 'strict_population_verification', 'ggn_audit_finalization',
        'ggn_audit_continuation', 'ggn_audit_optimization', 'gradient_reliable_continuation',
        'gradient_reliable_ggn', 'preconditioned_reliable_continuation', 'lbfgs_audit_preconditioner',
        'full_hessian_audit', 'metric_audit_optimization', 'affine_audit_optimization', 'neural_audit_optimization')
    return {str(directory/f'{name}.py'): _sha256((directory/f'{name}.py').read_bytes()) for name in modules}


def _merge_pins(destination: dict, additions: dict) -> None:
    for filename, digest in additions.items():
        resolved = str(Path(filename).resolve())
        if resolved in destination and destination[resolved] != digest:
            raise ValueError('strict policies require incompatible source dependencies')
        destination[resolved] = digest


def _four_sources(entries: list[dict]) -> None:
    expected = {(fold, name) for fold in range(2) for name in DIAGNOSTICS}
    if (len(entries) != 4 or {(entry['fold'], entry['diagnostic']) for entry in entries} != expected
        or any(type(entry['fold']) is not int for entry in entries)
        or len({entry['accepted_checkpoint'] for entry in entries}) != 4
        or len({entry['receipt'] for entry in entries}) != 4):
        raise ValueError('strict export requires all four distinct original policy/receipt endpoints')


def _native_source(entry: dict) -> tuple[dict, dict, dict]:
    plan = json.loads(_checked_bytes(Path(entry['run_plan']), entry['run_plan_sha256']))
    native = torch.load(io.BytesIO(_checked_bytes(Path(entry['accepted_checkpoint']),
        entry['accepted_checkpoint_sha256'])), map_location='cpu', weights_only=False)
    phase = native.get('optimizer_phase_schema')
    if phase not in PHASES or plan['schema'] != phase or not plan['original_thresholds_unchanged']:
        raise ValueError('strict export requires a matching explicitly declared native phase')
    active = native[PHASES[phase]]
    if active['identity']['input_precision'] != {
        'states': 'torch.float32', 'returns': 'torch.float32', 'arithmetic': 'torch.float64'}:
        raise ValueError('strict export requires the original float32 input/float64 arithmetic contract')
    if (native['identity']['fold'] != entry['fold'] or active['identity']['diagnostic'] != entry['diagnostic']
        or not active['converged'] or not active['history']
        or active['stop_reason'] != 'original_raw_and_affine_metric_gradient_tolerances'
        or active['validation_used_for_steps_or_stopping'] or active.get('rejected_work') is not None):
        raise ValueError('all four original policies must meet their own strict gates before export')
    reference = plan['reference']
    norms = active['current_norms']
    if (active['identity']['reference'] != reference or not math.isfinite(active['current_training_loss'])
        or active['current_training_loss'] >= 0. or not all(math.isfinite(value) for value in norms.values())
        or norms['full_gradient_norm'] > reference['raw_threshold']
        or norms['affine_metric_gradient_norm'] > reference['affine_metric_threshold']):
        raise ValueError('a native source does not meet its original finite negative-loss/gradient conditions')
    return plan, native, active


def declare_strict_action_export(*, sources: list[StrictActionSource], evaluation_plan: Path,
                                declaration: Path) -> dict:
    """Bind all four converged endpoints and producer code before verification/export.

    This declaration opens no path populations. It does not replace the
    original immutable confirmation plan, certify a full gradient, or publish
    actions. Four independently recomputed verification receipts remain needed.
    """
    expected = {(fold, name) for fold in range(2) for name in DIAGNOSTICS}
    if len(sources) != 4 or {(item.fold, item.diagnostic) for item in sources} != expected:
        raise ValueError('strict export needs exactly four distinct fold/objective endpoints')
    entries = sorted((item.declaration() for item in sources), key=lambda item: (item['fold'], DIAGNOSTICS.index(item['diagnostic'])))
    _four_sources(entries)
    confirmation = validated_evaluation_plan(evaluation_plan)
    strict_pins = _strict_producers()
    pins, population = {}, None
    for entry in entries:
        plan, native, _ = _native_source(entry)
        actual = native['identity']['source_population_identity']
        if population is not None and population != actual:
            raise ValueError('strict actions must belong to the same original source population')
        population = actual
        _merge_pins(pins, plan['implementation_sha256'])
        _merge_pins(pins, {entry['run_plan']: entry['run_plan_sha256'],
            entry['accepted_checkpoint']: entry['accepted_checkpoint_sha256'],
            plan['normalization']: native['identity']['normalization_sha256']})
    variant = 'eight_step' if population['config']['substeps'] == 8 else 'one_step'
    settings = confirmation['variants'][variant]
    if (population['config']['substeps'] != settings['substeps']
        or population['config']['paths'] != settings['paths']
        or population['config']['batch_size'] != settings['batch_size']
        or population['config']['seed'] == settings['seed']):
        raise ValueError('strict source geometry or independent seed differs from confirmation')
    _merge_pins(pins, strict_pins)
    _merge_pins(pins, confirmation['semantic_sha256'])
    _merge_pins(pins, {str(evaluation_plan.resolve()): _sha256(evaluation_plan.read_bytes())})
    for filename, digest in pins.items():
        _checked_bytes(Path(filename), digest)
    result = {'schema': 'gromm2-strict-action-export-declaration-v1', 'sources': entries,
        'evaluation_plan': str(evaluation_plan.resolve()), 'evaluation_plan_sha256': _sha256(evaluation_plan.read_bytes()),
        'evaluation_variant': variant, 'source_population_identity': population,
        'strict_implementation_sha256': strict_pins, 'implementation_sha256': pins,
        'original_thresholds_unchanged': True, 'verification_or_inference_performed': False}
    declaration.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(declaration)+'.lock', timeout=0):
        if declaration.exists():
            raise ValueError('preserve the existing immutable strict export declaration')
        write_json(declaration, result)
    return result


def _declaration(path: Path, digest: str) -> tuple[dict, dict]:
    declared = json.loads(_checked_bytes(path, digest))
    _four_sources(declared['sources'])
    if (declared['schema'] != 'gromm2-strict-action-export-declaration-v1'
        or declared['strict_implementation_sha256'] != _strict_producers()
        or not declared['original_thresholds_unchanged']):
        raise ValueError('strict export producer code or original gate interpretation changed')
    for filename, expected in declared['implementation_sha256'].items():
        _checked_bytes(Path(filename), expected)
    confirmation = validated_evaluation_plan(Path(declared['evaluation_plan']))
    if _sha256(Path(declared['evaluation_plan']).read_bytes()) != declared['evaluation_plan_sha256']:
        raise ValueError('original strict confirmation plan changed')
    return declared, confirmation


def verify_declared_strict_action(*, declaration: Path, declaration_sha256: str,
                                 fold: int, diagnostic: str, policy: AffineDeepAction,
                                 problem: AffineAuditProblem, optimizer: MetricLBFGS) -> dict:
    """Write one source-bound, recomputed receipt without taking optimizer steps."""
    declared, _ = _declaration(declaration, declaration_sha256)
    selected = [entry for entry in declared['sources'] if (entry['fold'], entry['diagnostic']) == (fold, diagnostic)]
    if len(selected) != 1:
        raise ValueError('the requested strict policy is not uniquely declared')
    entry = selected[0]
    destination = Path(entry['receipt'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination)+'.lock', timeout=0):
        if destination.exists():
            raise ValueError('preserve the existing immutable strict verification receipt')
        receipt = verify_strict_action(policy=policy, problem=problem, optimizer=optimizer,
            run_plan=Path(entry['run_plan']), run_plan_sha256=entry['run_plan_sha256'],
            accepted_checkpoint=Path(entry['accepted_checkpoint']),
            accepted_checkpoint_sha256=entry['accepted_checkpoint_sha256'])
        if (receipt['fold'], receipt['diagnostic']) != (fold, diagnostic):
            raise ValueError('verified receipt belongs to a different original policy')
        result = {**receipt, 'export_declaration': str(declaration.resolve()),
            'export_declaration_sha256': declaration_sha256,
            'strict_implementation_sha256': declared['strict_implementation_sha256']}
        _declaration(declaration, declaration_sha256)
        write_json(destination, result)
    return result


def _original_normalization(*, plan: dict, native: dict, active: dict, source: dict) -> bytes:
    """Check original byte-level normalization, deterministic fold and row mask."""
    normalization = _checked_bytes(Path(plan['normalization']), native['identity']['normalization_sha256'])
    config_fields = native['state']['identity']['model_config']
    fold = native['identity']['fold']
    with np.load(io.BytesIO(normalization), allow_pickle=False) as saved:
        identity = json.loads(str(saved['identity'].item()))
        interval = fold_path_ranges(source['config']['paths'])[fold]
        protocol_fields = native['identity']['protocol']
        protocol = LocalAuditProtocol(**{**protocol_fields, 'network': ActionNetworkConfig(**protocol_fields['network'])})
        expected_mask = internal_training_mask(interval.stop-interval.start, protocol=protocol, fold=fold)
        days = source['config']['days']
        expected_rows = (np.flatnonzero(expected_mask)[:, None]*days+np.arange(days, dtype=np.int64)).reshape(-1)
        mask = saved['inner_training_mask']
        if (identity['fold'] != fold or identity['protocol'] != native['identity']['protocol']
            or identity['protocol_version'] != LocalAuditProtocol.version.unique_id48
            or json.loads(identity['path_identity']) != source
            or _sha256(identity['path_identity'].encode()) != native['identity']['source_population_identity_sha256']
            or mask.dtype != np.bool_ or mask.shape != (interval.stop-interval.start,)
            or not np.array_equal(mask, expected_mask)
            or len(expected_rows) != active['identity']['training_rows']
            or _sha256(expected_rows.tobytes()) != active['identity']['training_rows_sha256']
            or saved['mean'].shape != (config_fields['feature_count'],)
            or saved['scale'].shape != saved['mean'].shape or not np.isfinite(saved['mean']).all()
            or not np.isfinite(saved['scale']).all() or np.any(saved['scale'] <= 0.)
            or float(saved['action_scale']) != config_fields['action_scale']):
            raise ValueError('original strict normalization, complete-pair fold or population binding changed')
        if 'training_observations' in saved:
            observations = saved['training_observations']
            if observations.shape != () or observations.dtype.kind not in 'iu' or int(observations) != len(expected_rows):
                raise ValueError('strict normalization training observation count differs from intended rows')
    return normalization


def _verified_policy(entry: dict, declared: dict, declaration: Path,
                     declaration_sha256: str) -> tuple[dict, bytes, dict, dict]:
    plan, native, active = _native_source(entry)
    receipt_content = Path(entry['receipt']).read_bytes()
    receipt = json.loads(receipt_content)
    reference = plan['reference']
    if (receipt['schema'] != 'gromm2-strict-active-phase-verification-v1'
        or receipt['export_declaration'] != str(declaration.resolve())
        or receipt['export_declaration_sha256'] != declaration_sha256
        or receipt['strict_implementation_sha256'] != declared['strict_implementation_sha256']
        or receipt['active_optimizer_phase'] != native['optimizer_phase_schema']
        or receipt['run_plan'] != entry['run_plan'] or receipt['run_plan_sha256'] != entry['run_plan_sha256']
        or receipt['accepted_checkpoint'] != entry['accepted_checkpoint']
        or receipt['accepted_checkpoint_sha256'] != entry['accepted_checkpoint_sha256']
        or (receipt['fold'], receipt['diagnostic']) != (entry['fold'], entry['diagnostic'])
        or receipt['original_reference'] != reference or not receipt['original_thresholds_met']
        or not receipt['accepted_history_preserved'] or receipt['optimization_performed']
        or receipt['validation_or_holdout_used'] or receipt['verification_full_objective_passes'] != 1
        or receipt['additional_curvature_passes'] != 0
        or receipt['continuation_state_sha256'] != _sha256(json.dumps(active, sort_keys=True, allow_nan=False).encode())
        or receipt['point_sha256'] != active['point_sha256']
        or receipt['optimizer_state_sha256'] != active['optimizer_state_sha256']
        or receipt['cumulative_full_objective_passes'] != active['full_objective_passes']+1
        or receipt['cumulative_full_curvature_passes'] != active['full_curvature_passes']):
        raise ValueError('strict receipt does not bind the exact verified native phase and one-pass computation')
    if (not math.isfinite(receipt['training_loss']) or receipt['training_loss'] >= 0.
        or not math.isclose(receipt['training_loss'], active['current_training_loss'], rel_tol=1.e-10, abs_tol=1.e-11)
        or receipt['full_gradient_norm'] > reference['raw_threshold']
        or receipt['affine_metric_gradient_norm'] > reference['affine_metric_threshold']
        or any(not math.isfinite(receipt[name]) or not math.isclose(receipt[name], value, rel_tol=1.e-9, abs_tol=1.e-10)
            for name, value in active['current_norms'].items())):
        raise ValueError('strict recomputed gradient receipt failed its unchanged original gates')
    normalization = _original_normalization(plan=plan, native=native, active=active,
        source=declared['source_population_identity'])
    if (receipt['normalization'] != plan['normalization']
        or receipt['normalization_sha256'] != _sha256(normalization)):
        raise ValueError('strict receipt normalization changed')
    config_fields = native['state']['identity']['model_config']
    config = LocalActionConfig(**{**config_fields, 'network': ActionNetworkConfig(**config_fields['network'])})
    if (any(not bool(torch.isfinite(value).all()) for value in native['model'].values())
        or any(value.dtype != torch.float64 for value in native['model'].values() if value.is_floating_point())):
        raise ValueError('strict native floating weights must remain finite float64')
    with torch.random.fork_rng(devices=[]):
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=config)
        if _native_point_sha256(native, policy) != receipt['point_sha256']:
            raise ValueError('strict source model weights differ from its verified native point')
    weights = {'model_config': config_fields, 'code_version': native['code_version'],
        'user_version': native['user_version'], 'diagnostic': entry['diagnostic'], 'state_dict': native['model'],
        'input_normalization_dtype': 'float32', 'policy_state_source': 'terminal trained model, not best_state',
        'native_optimizer_phase': native['optimizer_phase_schema'],
        'strict_verification_receipt_sha256': _sha256(receipt_content)}
    selected = {'source_checkpoint': entry['accepted_checkpoint'],
        'source_checkpoint_sha256': entry['accepted_checkpoint_sha256'],
        'strict_verification_receipt': entry['receipt'], 'strict_verification_receipt_sha256': _sha256(receipt_content),
        'terminal_training_loss': receipt['training_loss'], 'point_sha256': receipt['point_sha256'],
        'optimizer_stationarity_certified': True, 'exact_global_optimum_certified': False,
        'strict_original_reference': reference, 'active_optimizer_phase': native['optimizer_phase_schema']}
    return weights, normalization, selected, {entry['receipt']: _sha256(receipt_content)}


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    if path.exists() or temporary.exists():
        raise ValueError('preserve completed or orphaned strict export artifacts')
    temporary.write_bytes(content)
    temporary.replace(path)


def freeze_strict_action_bundle(*, declaration: Path, declaration_sha256: str, output: Path) -> dict:
    """Publish original weights/normalizers first and the four-policy manifest last."""
    declared, confirmation = _declaration(declaration, declaration_sha256)
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'terminal-policies.json'
    with FileLock(str(output/'freeze.lock'), timeout=0):
        if destination.exists():
            return preflight_strict_action_bundle(manifest=destination,
                declaration=declaration, declaration_sha256=declaration_sha256)
        if any(output.glob('fold-*')) or destination.with_suffix('.json.tmp').exists():
            raise ValueError('preserve orphaned strict action artifacts before any export retry')
        snapshots, receipt_pins, output_pins, geometry = [], {}, {}, None
        for entry in declared['sources']:
            weights, normalization, selected, pins = _verified_policy(entry, declared, declaration, declaration_sha256)
            current = {name: weights['model_config'][name] for name in ('feature_count', 'instrument_count', 'squared_lambda')}
            if geometry is not None and geometry != current:
                raise ValueError('the four strict policies have incompatible trading geometry or objective')
            geometry = current
            snapshots.append((entry, weights, normalization, selected))
            _merge_pins(receipt_pins, pins)
        result = {'schema': 'gromm2-terminal-frozen-actions-v2', 'producer': 'strict-active-phase-affine-plus-deep-v1',
            'protocol': confirmation['protocol'], 'protocol_sha256': confirmation['protocol_sha256'],
            'semantic_sha256': confirmation['semantic_sha256'], 'evaluation_plan': declared['evaluation_plan'],
            'evaluation_plan_sha256': declared['evaluation_plan_sha256'], 'evaluation_variant': declared['evaluation_variant'],
            'evaluation_settings': confirmation['variants'][declared['evaluation_variant']],
            'source_population_identity': declared['source_population_identity'],
            'strict_export_declaration': str(declaration.resolve()), 'strict_export_declaration_sha256': declaration_sha256,
            'strict_implementation_sha256': declared['strict_implementation_sha256'],
            'selection': 'terminal original-gradient-converged actions; no validation selection or zero fallback',
            'optimizer_stationarity_certified': True, 'exact_global_optimum_certified': False,
            'outer_or_fresh_evaluation_arrays_read': False,
            'generic_fresh_report_stationarity_field': 'unchanged generic scorer makes no optimizer certificate; use these strict receipts',
            'folds': [{'fold': fold, 'policies': {}} for fold in range(2)]}
        for entry, weights, normalization, selected in snapshots:
            prefix = f'fold-{entry["fold"]}-{entry["diagnostic"]}'
            weights_path, normalization_path = output/f'{prefix}.pt', output/f'{prefix}-normalization.npz'
            stream = io.BytesIO()
            torch.save(weights, stream)
            content = stream.getvalue()
            _atomic_bytes(weights_path, content)
            _atomic_bytes(normalization_path, normalization)
            selected.update(weights=str(weights_path.resolve()), weights_sha256=_sha256(content),
                normalization=str(normalization_path.resolve()), normalization_sha256=_sha256(normalization))
            output_pins[weights_path] = selected['weights_sha256']
            output_pins[normalization_path] = selected['normalization_sha256']
            result['folds'][entry['fold']]['policies'][entry['diagnostic']] = selected
        _declaration(declaration, declaration_sha256)
        for filename, digest in receipt_pins.items():
            _checked_bytes(Path(filename), digest)
        for filename, digest in output_pins.items():
            _checked_bytes(filename, digest)
        write_json(destination, result)
    return result


def preflight_strict_action_bundle(*, manifest: Path, declaration: Path,
                                  declaration_sha256: str) -> dict:
    """Validate strict provenance before the unchanged generic scorer sees paths."""
    declared, confirmation = _declaration(declaration, declaration_sha256)
    manifest_content = manifest.read_bytes()
    frozen = json.loads(manifest_content)
    if (frozen['producer'] != 'strict-active-phase-affine-plus-deep-v1'
        or frozen['strict_export_declaration'] != str(declaration.resolve())
        or frozen['strict_export_declaration_sha256'] != declaration_sha256
        or frozen['strict_implementation_sha256'] != declared['strict_implementation_sha256']
        or not frozen['optimizer_stationarity_certified']
        or frozen['source_population_identity'] != declared['source_population_identity']
        or frozen['evaluation_plan'] != declared['evaluation_plan']
        or frozen['evaluation_plan_sha256'] != declared['evaluation_plan_sha256']
        or frozen['evaluation_variant'] != declared['evaluation_variant']
        or frozen['evaluation_settings'] != confirmation['variants'][declared['evaluation_variant']]
        or frozen['semantic_sha256'] != confirmation['semantic_sha256']
        or frozen['protocol'] != confirmation['protocol'] or frozen['protocol_sha256'] != confirmation['protocol_sha256']):
        raise ValueError('frozen bundle does not bind the declared strict producer and native population')
    from quantdata.gromm2.terminal_inference import load_terminal_actions

    with torch.random.fork_rng(devices=[]):
        loaded, actions = load_terminal_actions(manifest, device='cpu')
    if loaded != frozen:
        raise ValueError('strict manifest changed while loading its actions')
    receipt_pins = {}
    for entry in declared['sources']:
        weights, normalization, expected, pins = _verified_policy(entry, declared, declaration, declaration_sha256)
        _merge_pins(receipt_pins, pins)
        selected = frozen['folds'][entry['fold']]['policies'][entry['diagnostic']]
        exported = torch.load(io.BytesIO(_checked_bytes(Path(selected['weights']), selected['weights_sha256'])),
            map_location='cpu', weights_only=False)
        if (any(selected.get(name) != value for name, value in expected.items())
            or Path(selected['normalization']).read_bytes() != normalization
            or action_point_sha256(actions[f'fold-{entry["fold"]}/{entry["diagnostic"]}'].model) != expected['point_sha256']
            or actions[f'fold-{entry["fold"]}/{entry["diagnostic"]}'].mean.dtype != torch.float32):
            raise ValueError('frozen original weights, normalization precision or strict receipt changed')
        if any(exported[name] != value for name, value in weights.items() if name != 'state_dict'):
            raise ValueError('strict exported objective, config, precision or native-phase metadata changed')
    _declaration(declaration, declaration_sha256)
    for filename, digest in receipt_pins.items():
        _checked_bytes(Path(filename), digest)
    if manifest.read_bytes() != manifest_content:
        raise ValueError('strict manifest changed during its preflight')
    return frozen
