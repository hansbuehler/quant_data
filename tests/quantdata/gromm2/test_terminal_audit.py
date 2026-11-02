from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.gromm2.conditional_audit import LocalActionConfig
from quantdata.gromm2.terminal_audit import (
    freeze_terminal_actions,
    terminal_semantic_hashes,
    terminal_trial_selection,
)


def report(*, fold: int = 0, entropy: float = .3, squared: float = .4) -> dict:
    return {'fold': fold, 'epoch': 3, 'stop_reason': 'internal_validation_plateau',
        'best_epochs': {'entropy': 0, 'squared': 0},
        'train_config': {'affine_epochs': 1, 'minimum_full_epochs': 2,
            'precision': 32,
            'convergence_patience': 1, 'minimum_learning_rate': 1.e-6},
        'model_config': {'feature_count': 3, 'action_scale': .01},
        'effective_optimizer_controls': {'entropy': {'learning_rates': [1.e-6]}, 'squared': {'learning_rates': [1.e-6]}},
        'path_identity_sha256': 'fixture-source', 'implementation_sha256': {},
        'history': [{'epoch': epoch, 'full_network': True,
            'training': {'entropy': .1, 'squared': .2},
            'internal_validation': {'entropy': entropy, 'squared': squared},
            'learning_rates_used': {'entropy': [1.e-6], 'squared': [1.e-6]}} for epoch in (2, 3)]}


def test_terminal_selection_is_nonzero_even_when_zero_has_better_loss() -> None:
    reports = [report(entropy=.2, squared=.5), report(entropy=.4, squared=.1)]
    assert terminal_trial_selection(reports) == {'entropy': 0, 'squared': 1}
    for item in reports:
        item['outer_holdout_score'] = -1.e100
    assert terminal_trial_selection(reports) == {'entropy': 0, 'squared': 1}
    assert terminal_trial_selection([report(), report()]) == {'entropy': 0, 'squared': 0}


@pytest.mark.parametrize('field', ['zero', 'nonfinite'])
def test_all_degenerate_candidates_are_uninformative(field) -> None:
    item = report()
    item['history'][-1]['training']['entropy'] = 0. if field == 'zero' else float('nan')
    with pytest.raises(ValueError, match='no informative nonzero'):
        terminal_trial_selection([item])


@pytest.mark.parametrize('field', ['reason', 'minimum_work', 'floor_work'])
def test_incomplete_trials_cannot_be_selected(field) -> None:
    item = report()
    if field == 'reason':
        item['stop_reason'] = 'maximum_epochs_not_converged'
    elif field == 'minimum_work':
        item['train_config']['minimum_full_epochs'] = 100
    else:
        item['history'][-1]['learning_rates_used']['entropy'] = [2.e-6]
    with pytest.raises(ValueError):
        terminal_trial_selection([item])


def make_trials(tmp_path: Path) -> tuple[Path, Path]:
    trials = tmp_path / 'trials'
    source = {'test': True, 'config': {'seed': 71, 'substeps': 1, 'batch_size': 4}}
    source_identity = json.dumps(source, sort_keys=True)
    source_hash = hashlib.sha256(source_identity.encode()).hexdigest()
    candidate_controls = []
    folds = []
    for fold in range(2):
        reports = []
        for trial in range(2):
            checkpoint = trials / f'fold-{fold}' / f'trial-{trial}' / 'model' / 'checkpoint.pt'
            checkpoint.parent.mkdir(parents=True)
            normalization = checkpoint.parent.parent / 'normalization.npz'
            item = report(fold=fold, entropy=.1 + trial, squared=1. - .5 * trial)
            item['train_config']['batch_size'] = 8 + trial
            item['path_identity_sha256'] = source_hash
            if fold == 0:
                candidate_controls.append(item['train_config'])
            identity = json.dumps({'path_identity': source_identity, 'fold': fold, 'protocol': {}, 'protocol_version': 'fixture'})
            np.savez(normalization, mean=np.zeros(3), scale=np.ones(3), action_scale=.01, identity=identity)
            item['checkpoint'] = str(checkpoint)
            torch.save({'code_version': LocalActionConfig.version.unique_id48, 'user_version': '0.1',
                'identity': identity,
                'report': item, 'state': {'stopped': {'entropy': True, 'squared': True}, 'next_batch': 0,
                    'floor_since': {'entropy': 2, 'squared': 2}, 'last_improvement': {'entropy': 0, 'squared': 0},
                    'best_state': {'entropy': {'bias': torch.zeros(2)}, 'squared': {'bias': torch.zeros(2)}}},
                'model': {'policies.entropy.bias': torch.tensor([1., 2.]),
                          'policies.squared.bias': torch.tensor([3., 4.])}}, checkpoint)
            reports.append(str(checkpoint))
        folds.append({'fold': fold, 'reports': reports})
    (trials / 'frozen-policies.json').write_text(json.dumps({'source_identity': source,
        'source_identity_sha256': source_hash, 'folds': folds, 'protocol': {}, 'protocol_version': 'fixture',
        'candidate_controls': candidate_controls,
        'status': 'policies_frozen; inference not yet performed'}))
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Predeclared fresh evaluation, no source arrays accessed.\n')
    protocol.with_suffix('.json').write_text(json.dumps({'schema': 'gromm2-terminal-confirmation-plan-v1',
        'protocol': str(protocol), 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'semantic_sha256': terminal_semantic_hashes(), 'familywise_endpoint_count': 12,
        'variants': {'one_step': {'substeps': 1, 'seed': 72, 'paths': 8, 'batch_size': 4},
                     'eight_step': {'substeps': 8, 'seed': 73, 'paths': 8, 'batch_size': 4}}}))
    return trials, protocol


def test_freezing_exports_terminal_weights_not_zero_best_state(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    source = trials / 'fold-0/trial-0/model/checkpoint.pt'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen = freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
        evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)
    assert frozen['outer_or_fresh_evaluation_arrays_read'] is False
    for fold in frozen['folds']:
        for name, expected in [('entropy', [1., 2.]), ('squared', [3., 4.])]:
            chosen = fold['policies'][name]
            saved = torch.load(chosen['weights'], weights_only=False)
            torch.testing.assert_close(saved['state_dict']['bias'], torch.tensor(expected))
            assert chosen['original_zero_checkpoint_selected'] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
        evaluation_plan=protocol.with_suffix('.json'), expected_trials=2) == frozen


def test_published_freeze_is_immutable(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    frozen = freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
        evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)
    Path(frozen['folds'][0]['policies']['entropy']['weights']).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='artifact changed'):
        freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
            evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)


def test_checkpoint_with_wrong_population_cannot_be_frozen(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    path = trials / 'fold-0/trial-0/model/checkpoint.pt'
    saved = torch.load(path, weights_only=False)
    saved['report'] = copy.deepcopy(saved['report'])
    saved['report']['path_identity_sha256'] = 'different'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='another population'):
        freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
            evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)


def test_independent_policy_stopping_does_not_require_later_inactive_rates() -> None:
    item = report()
    later = copy.deepcopy(item['history'][-1])
    later['epoch'] = 4
    del later['learning_rates_used']['entropy']
    item['history'].append(later)
    item['epoch'] = 4
    assert terminal_trial_selection([item]) == {'entropy': 0, 'squared': 0}


def test_normalization_of_another_fold_is_not_accepted(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    path = trials / 'fold-0/trial-0/normalization.npz'
    np.savez(path, mean=np.ones(3), scale=np.ones(3), action_scale=.01, identity='different-fold')
    with pytest.raises(ValueError, match='normalization identity differs'):
        freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
            evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)


def test_duplicate_checkpoint_cannot_substitute_for_missing_trial(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    path = trials / 'frozen-policies.json'
    plan = json.loads(path.read_text())
    plan['folds'][0]['reports'][1] = plan['folds'][0]['reports'][0]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='must be distinct'):
        freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
            evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)


def test_swapped_candidate_order_is_not_accepted(tmp_path) -> None:
    trials, protocol = make_trials(tmp_path)
    path = trials / 'frozen-policies.json'
    plan = json.loads(path.read_text())
    plan['folds'][0]['reports'].reverse()
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='declared candidate controls'):
        freeze_terminal_actions(trials=trials, output=tmp_path / 'frozen', protocol=protocol,
            evaluation_plan=protocol.with_suffix('.json'), expected_trials=2)
