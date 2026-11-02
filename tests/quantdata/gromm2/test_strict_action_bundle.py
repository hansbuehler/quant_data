from __future__ import annotations

import hashlib
import io
import json
import signal
from copy import deepcopy
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2 import (
    local_audit_training,
    strict_action_bundle,
    strict_population_verification,
)
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_continuation import GGNStoppingReference, action_point_sha256
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig, _flatten
from quantdata.gromm2.local_audit_training import LocalAuditProtocol, internal_training_mask
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.gromm2.preconditioned_reliable_continuation import (
    PreconditionedReliableConfig,
    continue_preconditioned_reliable_action,
)
from quantdata.gromm2.strict_action_bundle import (
    StrictActionSource,
    declare_strict_action_export,
    freeze_strict_action_bundle,
    preflight_strict_action_bundle,
    verify_declared_strict_action,
)
from quantdata.gromm2.terminal_audit import terminal_semantic_hashes
from quantdata.gromm2.terminal_inference import load_terminal_actions, score_terminal_batch
from quantdata.torch import BaseEnvironment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return digest(path)


def build_prepared(tmp_path, *, normalization_change=None, input_dtype=torch.float32, training_rows_change=None,
                   retained_population=False):
    protocol = LocalAuditProtocol(network=ActionNetworkConfig(num_depth=1, num_width=2))
    population = {'config': {'paths': 16, 'days': 2, 'substeps': 8, 'batch_size': 2, 'seed': 123}}
    population_json = json.dumps(population, sort_keys=True)
    population_path = tmp_path/'paths.h5'
    mean = np.array([.123456789, -.333333333333])
    scale = np.array([.987654321, 1.234567891])
    if retained_population:
        raw_states = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float32).repeat(8, 1).reshape(16, 2, 2)
        raw_returns = torch.tensor([[.02, .04]], dtype=torch.float32).repeat(32, 1).reshape(16, 2, 2)
        with h5py.File(population_path, 'w') as stream:
            stream.attrs.update(identity=population_json, completed_paths=16)
            stream.create_dataset('features', data=raw_states.numpy())
            stream.create_dataset('returns', data=raw_returns.numpy())
            stream.create_dataset('expiry_days', data=[2])
            stream.create_dataset('strikes', data=[[1.]])
    description = tmp_path/'protocol.txt'
    description.write_text('four strict policies, then independent confirmation')
    evaluation = tmp_path/'evaluation.json'
    save_json(evaluation, {'schema': 'gromm2-terminal-confirmation-plan-v1', 'familywise_endpoint_count': 12,
        'protocol': str(description), 'protocol_sha256': digest(description), 'semantic_sha256': terminal_semantic_hashes(),
        'variants': {'one_step': {'paths': 16, 'substeps': 1, 'batch_size': 2, 'seed': 456},
            'eight_step': {'paths': 16, 'substeps': 8, 'batch_size': 2, 'seed': 789}}})
    entries, arguments = [], {}
    for fold in range(2):
        for name in DIAGNOSTICS:
            directory = tmp_path/f'fold-{fold}-{name}'
            directory.mkdir()
            mask = internal_training_mask(8, protocol=protocol, fold=fold)
            rows = torch.arange(16).reshape(8, 2)[torch.as_tensor(mask)].flatten()
            if training_rows_change is not None:
                rows = training_rows_change(rows)
            states = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=input_dtype).repeat(4, 1)
            returns = torch.tensor([[.1, .2]], dtype=input_dtype).repeat(16, 1)
            if retained_population:
                states = (raw_states[fold*8:(fold+1)*8].reshape(16, 2)-torch.as_tensor(mean, dtype=torch.float32))/torch.as_tensor(scale, dtype=torch.float32)
                returns = raw_returns[fold*8:(fold+1)*8].reshape(16, 2)/.2
            else:
                states[~torch.repeat_interleave(torch.as_tensor(mask), 2)] = float('nan')
            problem = AffineAuditProblem(states=states, returns=returns, eligible=torch.ones((2, 2), dtype=torch.bool),
                rows=rows, action_scale=.2, config=AffineSolveConfig(batch_size=3))
            config = LocalActionConfig(label=f'strict-export-{fold}-{name}', feature_count=2, instrument_count=2,
                action_scale=.2, network=protocol.network)
            policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=config)
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.zero_()
            loss = float(full_action_pass(policy, problem, diagnostic=name, backward=True))
            optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.2), history_size=5, curvature_floor=1.e-12)
            norms = optimizer.metric.diagnostics(_flatten({key: value.grad for key, value in policy.named_parameters()}))
            baseline = {'iteration': 0, 'training_loss': loss, **norms, 'gradient_threshold': 1.e-7,
                'affine_metric_gradient_threshold': 1.e-7, 'full_objective_evaluations': 7, 'full_curvature_passes': 3}
            normalization = directory/'normalization.npz'
            normalization_values = dict(mean=mean, scale=scale, action_scale=np.array(.2), inner_training_mask=mask,
                training_observations=np.array(len(rows)), identity=np.array(json.dumps({'fold': fold,
                    'path_identity': population_json, 'protocol': asdict(protocol),
                    'protocol_version': LocalAuditProtocol.version.unique_id48})))
            if normalization_change is not None:
                normalization_change(normalization_values)
            np.savez(normalization, **normalization_values)
            identity = {'fold': fold, 'normalization_sha256': digest(normalization), 'protocol': asdict(protocol),
                'source_population_identity': population,
                'source_population_identity_sha256': hashlib.sha256(population_json.encode()).hexdigest()}
            parent = {'model': deepcopy(policy.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
                'identity': identity, 'code_version': config.version.unique_id48, 'user_version': config.user_version,
                'optimizer_phase_schema': 'gromm2-gradient-reliable-GGN-v1',
                'ggn_continuation_state': {'full_curvature_passes': 3},
                'reliable_continuation_state': {'full_curvature_passes': 17},
                'torch_rng': torch.get_rng_state().clone(), 'cuda_rng': None,
                'state': {'identity': {'model_config': asdict(config), 'data_identity': f'original-fold-{fold}', 'diagnostic': name},
                    'history': [baseline], 'initial_gradient_norm': norms['full_gradient_norm'],
                    'full_objective_evaluations': 7, 'full_curvature_evaluations': 17, 'empirical_local_stationarity': False}}
            source = directory/'source.pt'
            torch.save(parent, source)
            reference = GGNStoppingReference(source_checkpoint_sha256=digest(source),
                initial_gradient_norm=norms['full_gradient_norm'], raw_threshold=1.e-7, affine_metric_threshold=1.e-7,
                source_accepted_updates=0, prior_full_objective_passes=7, prior_full_curvature_passes=17)
            controls = PreconditionedReliableConfig(step=GGNStepConfig(damping=.001,
                cg_relative_tolerance=1.e-6, maximum_cg_iterations=30), maximum_accepted_steps=16)
            point = {key: baseline[key] for key in ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}
            state = continue_preconditioned_reliable_action(policy=policy, problem=problem, optimizer=optimizer,
                diagnostic=name, data_identity=f'original-fold-{fold}', reference=reference,
                source_point={**point, 'point_sha256': action_point_sha256(policy)}, config=controls)
            assert state['converged'] and state['history']
            saved = deepcopy(parent)
            saved.update(model=deepcopy(policy.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                optimizer_phase_schema='gromm2-history-preconditioned-reliable-GGN-v1', preconditioned_reliable_state=state)
            saved['state']['history'].extend(deepcopy(state['history']))
            saved['state']['history'][-1]['full_objective_evaluations'] = state['full_objective_passes']
            saved['state'].update(full_objective_evaluations=state['full_objective_passes'],
                full_curvature_evaluations=state['full_curvature_passes'], empirical_local_stationarity=True)
            native = directory/'native.pt'
            torch.save(saved, native)
            plan = directory/'plan.json'
            plan_sha = save_json(plan, {'schema': saved['optimizer_phase_schema'], 'original_thresholds_unchanged': True,
                'source_native': str(source), 'source_native_sha256': digest(source), 'normalization': str(normalization),
                'population': str(population_path),
                'implementation_sha256': {str(source): digest(source)}, 'reference': asdict(reference),
                'config': asdict(controls), 'source_point': point})
            entry = StrictActionSource(fold=fold, diagnostic=name, run_plan=plan, run_plan_sha256=plan_sha,
                accepted_checkpoint=native, accepted_checkpoint_sha256=digest(native), receipt=directory/'receipt.json')
            entries.append(entry)
            arguments[fold, name] = {'policy': policy, 'problem': problem, 'optimizer': optimizer}
    declaration = tmp_path/'export-declaration.json'
    declare_strict_action_export(sources=entries, evaluation_plan=evaluation, declaration=declaration)
    return dict(entries=entries, arguments=arguments, declaration=declaration, declaration_sha256=digest(declaration),
        evaluation=evaluation, output=tmp_path/'frozen')


def test_original_population_producer_verifies_all_four_without_cross_fold_array_reads(tmp_path, monkeypatch):
    prepared = build_prepared(tmp_path, retained_population=True)
    monkeypatch.setattr(strict_population_verification, 'check_resource_headroom', lambda: None)
    original_read = local_audit_training.read_audit_block
    active_fold, reads = 0, []

    def guarded_read(stream, name, rows, **keywords):
        assert active_fold*8 <= rows.start < rows.stop <= (active_fold+1)*8
        reads.append((active_fold, name, rows.start, rows.stop))
        return original_read(stream, name, rows, **keywords)

    monkeypatch.setattr(local_audit_training, 'read_audit_block', guarded_read)
    rng = torch.get_rng_state().clone()
    for entry in prepared['entries']:
        active_fold = entry.fold
        native_sha = digest(entry.accepted_checkpoint)
        receipt = strict_population_verification.verify_strict_population_action(
            declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
            fold=entry.fold, diagnostic=entry.diagnostic, device='cpu', upload_paths=3)
        assert receipt['verification_full_objective_passes'] == 1
        assert receipt['additional_curvature_passes'] == 0 and not receipt['optimization_performed']
        assert receipt['original_thresholds_met']
        assert digest(entry.accepted_checkpoint) == native_sha
        assert torch.equal(torch.get_rng_state(), rng)
    assert len(reads) == 24
    manifest = freeze(prepared)
    assert preflight_strict_action_bundle(manifest=prepared['output']/'terminal-policies.json',
        declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256']) == manifest
    monkeypatch.setattr(h5py, 'File', lambda *args, **keywords: pytest.fail('existing receipt must fail before reading population'))
    with pytest.raises(ValueError, match='preserve the existing immutable strict verification receipt'):
        strict_population_verification.verify_strict_population_action(
            declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
            fold=0, diagnostic='squared', device='cpu')


@pytest.mark.parametrize('change_identity', [False, True])
def test_population_producer_refuses_changed_population_without_publishing_receipt(tmp_path, monkeypatch, change_identity):
    prepared = build_prepared(tmp_path, retained_population=True)
    monkeypatch.setattr(strict_population_verification, 'check_resource_headroom', lambda: None)
    with h5py.File(tmp_path/'paths.h5', 'r+') as stream:
        if change_identity:
            stream.attrs['identity'] = json.dumps({'different': True})
        else:
            stream['returns'][:8] += .01
    with pytest.raises(ValueError, match='original training population identity changed|restore the actual original accepted point'):
        strict_population_verification.verify_strict_population_action(
            declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
            fold=0, diagnostic='squared', device='cpu', upload_paths=3)
    assert not any(entry.receipt.exists() for entry in prepared['entries'])


def test_population_producer_requires_managed_cuda_before_any_io_or_initialization(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('unallocated GPU initialization'))
    with pytest.raises(ValueError, match='managed GPU allocation'):
        strict_population_verification.verify_strict_population_action(declaration=tmp_path/'absent.json',
            declaration_sha256='0'*64, fold=0, diagnostic='squared', device='cuda')


def test_population_verification_ignores_validation_and_opposite_fold_returns(tmp_path, monkeypatch):
    prepared = build_prepared(tmp_path, retained_population=True)
    monkeypatch.setattr(strict_population_verification, 'check_resource_headroom', lambda: None)
    mask = internal_training_mask(8, protocol=LocalAuditProtocol(
        network=ActionNetworkConfig(num_depth=1, num_width=2)), fold=0)
    with h5py.File(tmp_path/'paths.h5', 'r+') as stream:
        stream['features'][8:] = np.nan
        stream['returns'][8:] = np.nan
        validation = np.flatnonzero(~mask)
        stream['features'][validation] = np.full((len(validation), 2, 2), 1.e8, dtype=np.float32)
        stream['returns'][validation] = np.full((len(validation), 2, 2), -1.e8, dtype=np.float32)
    receipt = strict_population_verification.verify_strict_population_action(
        declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
        fold=0, diagnostic='squared', device='cpu', upload_paths=3)
    assert receipt['original_thresholds_met'] and receipt['verification_full_objective_passes'] == 1
    assert not receipt['validation_or_holdout_used']


def test_population_verification_signal_exits_without_receipt_and_restores_handler(tmp_path, monkeypatch):
    prepared = build_prepared(tmp_path, retained_population=True)
    monkeypatch.setattr(strict_population_verification, 'check_resource_headroom', lambda: None)
    original_read = local_audit_training.read_audit_block
    original_handler = signal.getsignal(signal.SIGTERM)

    def interrupted_read(*arguments, **keywords):
        result = original_read(*arguments, **keywords)
        signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(local_audit_training, 'read_audit_block', interrupted_read)
    with pytest.raises(InterruptedError, match='stopped during fold upload'):
        strict_population_verification.verify_strict_population_action(
            declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
            fold=0, diagnostic='squared', device='cpu', upload_paths=1)
    assert signal.getsignal(signal.SIGTERM) is original_handler
    assert not any(entry.receipt.exists() for entry in prepared['entries'])


@pytest.fixture
def prepared(tmp_path):
    return build_prepared(tmp_path)


def receipts(prepared):
    for entry in prepared['entries']:
        verify_declared_strict_action(declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'],
            fold=entry.fold, diagnostic=entry.diagnostic, **prepared['arguments'][entry.fold, entry.diagnostic])


def freeze(prepared):
    return freeze_strict_action_bundle(declaration=prepared['declaration'],
        declaration_sha256=prepared['declaration_sha256'], output=prepared['output'])


def test_four_phase_preserving_exports_load_with_unchanged_precision_and_scoring(prepared, monkeypatch):
    source_bytes = {entry.accepted_checkpoint: entry.accepted_checkpoint.read_bytes() for entry in prepared['entries']}

    def forbidden(*unused, **keywords):
        raise AssertionError('strict verification/export must not read any path populations')

    monkeypatch.setattr(h5py, 'File', forbidden)
    receipts(prepared)
    rng = torch.get_rng_state().clone()
    manifest = freeze(prepared)
    assert torch.equal(torch.get_rng_state(), rng)
    assert manifest['optimizer_stationarity_certified'] and not manifest['exact_global_optimum_certified']
    assert [entry['fold'] for entry in manifest['folds']] == [0, 1]
    assert all(entry.accepted_checkpoint.read_bytes() == source_bytes[entry.accepted_checkpoint] for entry in prepared['entries'])
    path = prepared['output']/'terminal-policies.json'
    assert preflight_strict_action_bundle(manifest=path, declaration=prepared['declaration'],
        declaration_sha256=prepared['declaration_sha256']) == manifest
    _, loaded = load_terminal_actions(path, device='cpu')
    features = np.array([[[.23456789, -.87654321], [.22222222, 1.23456789]]], dtype=np.float32)
    gains = np.array([[[.1, -.2], [.3, -.1]]], dtype=np.float32)
    eligible = np.ones((2, 2), dtype=bool)
    score = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=loaded)
    for index, entry in enumerate(prepared['entries']):
        selected = manifest['folds'][entry.fold]['policies'][entry.diagnostic]
        original_plan = json.loads(entry.run_plan.read_text())
        assert selected['active_optimizer_phase'] == 'gromm2-history-preconditioned-reliable-GGN-v1'
        assert selected['strict_original_reference'] == original_plan['reference']
        normalization = selected['normalization']
        assert open(normalization, 'rb').read() == open(original_plan['normalization'], 'rb').read()
        action = loaded[f'fold-{entry.fold}/{entry.diagnostic}']
        assert action.mean.dtype == torch.float32 and action.model.fdtype == torch.float64
        raw = torch.as_tensor(features, dtype=torch.float32)
        normalized = ((raw-action.mean)/action.scale).to(torch.float64)
        native = prepared['arguments'][entry.fold, entry.diagnostic]['policy']
        with torch.no_grad():
            expected = .2*native(normalized, torch.as_tensor(eligible))
        np.testing.assert_allclose(score[0, index, 0], float((expected*torch.as_tensor(gains).double()).sum(-1).mean()),
            rtol=1.e-12, atol=1.e-14)
    assert freeze(prepared) == manifest


def test_incomplete_source_set_and_duplicate_receipt_cannot_be_declared(prepared):
    with pytest.raises(ValueError, match='exactly four'):
        declare_strict_action_export(sources=prepared['entries'][:3], evaluation_plan=prepared['evaluation'],
            declaration=prepared['output']/'invalid.json')
    copied = list(prepared['entries'])
    copied[1] = replace(copied[1], receipt=copied[0].receipt)
    with pytest.raises(ValueError, match='four distinct'):
        declare_strict_action_export(sources=copied, evaluation_plan=prepared['evaluation'],
            declaration=prepared['output']/'invalid.json')


def test_one_missing_or_bad_receipt_cannot_publish_any_actions(prepared):
    receipts(prepared)
    selected = prepared['entries'][-1]
    values = json.loads(selected.receipt.read_text())
    values['full_gradient_norm'] = 10.
    save_json(selected.receipt, values)
    with pytest.raises(ValueError, match='unchanged original gates'):
        freeze(prepared)
    assert not list(prepared['output'].glob('fold-*'))
    assert not (prepared['output']/'terminal-policies.json').exists()


def test_normalization_or_receipt_mutation_after_freeze_rejects_without_repair(prepared):
    receipts(prepared)
    result = freeze(prepared)
    path = prepared['output']/'terminal-policies.json'
    original = path.read_bytes()
    selected = result['folds'][0]['policies']['entropy']
    with open(selected['normalization'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='immutable frozen action'):
        preflight_strict_action_bundle(manifest=path, declaration=prepared['declaration'],
            declaration_sha256=prepared['declaration_sha256'])
    assert path.read_bytes() == original


def test_orphaned_artifact_is_not_overwritten(prepared):
    receipts(prepared)
    prepared['output'].mkdir()
    orphan = prepared['output']/'fold-0-entropy.pt.tmp'
    orphan.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='orphaned'):
        freeze(prepared)
    assert orphan.read_bytes() == b'preserve'
    assert not (prepared['output']/'terminal-policies.json').exists()


def test_changed_receipt_during_publication_prevents_manifest_and_retry(prepared, monkeypatch):
    receipts(prepared)
    original = strict_action_bundle._atomic_bytes
    changed = False

    def interfere(path, content):
        nonlocal changed
        original(path, content)
        if not changed:
            changed = True
            with prepared['entries'][0].receipt.open('ab') as stream:
                stream.write(b' ')

    monkeypatch.setattr(strict_action_bundle, '_atomic_bytes', interfere)
    with pytest.raises(ValueError, match='finalization source changed'):
        freeze(prepared)
    assert not (prepared['output']/'terminal-policies.json').exists()
    with pytest.raises(ValueError, match='orphaned'):
        freeze(prepared)


def test_frozen_objective_config_cannot_change_even_with_rehashed_manifest(prepared):
    receipts(prepared)
    result = freeze(prepared)
    selected = result['folds'][0]['policies']['entropy']
    weights = torch.load(selected['weights'], map_location='cpu', weights_only=False)
    weights['model_config']['label'] = 'changed-provenance-label'
    stream = io.BytesIO()
    torch.save(weights, stream)
    content = stream.getvalue()
    with open(selected['weights'], 'wb') as output:
        output.write(content)
    selected['weights_sha256'] = hashlib.sha256(content).hexdigest()
    save_json(prepared['output']/'terminal-policies.json', result)
    with pytest.raises(ValueError, match='objective, config'):
        preflight_strict_action_bundle(manifest=prepared['output']/'terminal-policies.json',
            declaration=prepared['declaration'], declaration_sha256=prepared['declaration_sha256'])


def test_same_count_complete_pair_mask_must_match_original_protocol(tmp_path):
    def changed_mask(values):
        mask = values['inner_training_mask'].copy()
        training = int(np.flatnonzero(mask[::2])[0])
        validation = int(np.flatnonzero(~mask[::2])[0])
        mask[2*training:2*training+2] = False
        mask[2*validation:2*validation+2] = True
        values['inner_training_mask'] = mask

    prepared = build_prepared(tmp_path, normalization_change=changed_mask)
    receipts(prepared)
    with pytest.raises(ValueError, match='original strict normalization'):
        freeze(prepared)
    assert not list(prepared['output'].glob('fold-*'))
    assert not (prepared['output']/'terminal-policies.json').exists()


def test_verified_rows_must_equal_exact_original_mask_rows(tmp_path):
    def duplicate_row(rows):
        changed = rows.clone()
        changed[0] = changed[1]
        return changed

    prepared = build_prepared(tmp_path, training_rows_change=duplicate_row)
    receipts(prepared)
    with pytest.raises(ValueError, match='original strict normalization'):
        freeze(prepared)
    assert not list(prepared['output'].glob('fold-*'))


@pytest.mark.parametrize('metadata', [np.array(1), np.array(8.5), np.array([8])])
def test_normalization_training_observations_must_match_intended_rows(tmp_path, metadata):
    prepared = build_prepared(tmp_path,
        normalization_change=lambda values: values.update(training_observations=metadata))
    receipts(prepared)
    with pytest.raises(ValueError, match='training observation count'):
        freeze(prepared)
    assert not list(prepared['output'].glob('fold-*'))


def test_native_float64_inputs_cannot_be_relabelled_float32(tmp_path):
    with pytest.raises(ValueError, match='original float32 input/float64 arithmetic'):
        build_prepared(tmp_path, input_dtype=torch.float64)
    assert not (tmp_path/'export-declaration.json').exists()


@pytest.mark.parametrize('suffix', ['.pt', '.npz'])
def test_corrupt_written_artifact_blocks_manifest_and_preserves_orphans(prepared, monkeypatch, suffix):
    receipts(prepared)
    original = strict_action_bundle._atomic_bytes
    corrupted = []

    def interfere(path, content):
        original(path, content)
        if path.suffix == suffix and not corrupted:
            with path.open('ab') as stream:
                stream.write(b'corrupted-after-atomic-write')
            corrupted.append(path)

    monkeypatch.setattr(strict_action_bundle, '_atomic_bytes', interfere)
    with pytest.raises(ValueError, match='finalization source changed'):
        freeze(prepared)
    assert corrupted and corrupted[0].read_bytes().endswith(b'corrupted-after-atomic-write')
    assert not (prepared['output']/'terminal-policies.json').exists()
    with pytest.raises(ValueError, match='orphaned'):
        freeze(prepared)
