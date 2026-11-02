from __future__ import annotations

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from quantdata.gromm2.conditional_refinement import fit_trials, select_trials, trial_settings
from quantdata.gromm2.local_audit_training import LocalAuditProtocol, LocalAuditTrainConfig


def test_declared_grid_controls_batch_and_learning_rate() -> None:
    base = LocalAuditTrainConfig(device='cpu', minimum_learning_rate=1.e-8)
    settings = trial_settings(base=base, batch_sizes=(1024, 4096), learning_rates=(1.e-6, 1.e-5))
    assert [(entry.batch_size, entry.learning_rate) for entry in settings] == [
        (1024, 1.e-6), (1024, 1.e-5), (4096, 1.e-6), (4096, 1.e-5)]
    with pytest.raises(ValueError, match='unique'):
        trial_settings(base=base, batch_sizes=(1024, 1024), learning_rates=(1.e-6,))


def test_selection_uses_inner_losses_not_an_outer_score(tmp_path) -> None:
    reports = []
    for index, losses in enumerate([(-.1, -.3), (-.2, -.1)]):
        checkpoint = tmp_path / str(index) / 'checkpoint.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(str(index).encode())
        reports.append({'checkpoint': str(checkpoint), 'stop_reason': 'internal_validation_plateau',
            'best_epochs': {'entropy': 1, 'squared': 2},
            'best_internal_losses': dict(zip(('entropy', 'squared'), losses, strict=True)),
            'outer_holdout_score': 1.e12 if index == 0 else -1.e12})
    selected = select_trials(reports)
    assert selected['entropy']['checkpoint'] == reports[1]['checkpoint']
    assert selected['squared']['checkpoint'] == reports[0]['checkpoint']
    reports[0]['stop_reason'] = 'maximum_epochs_not_converged'
    with pytest.raises(RuntimeError, match='all declared trials'):
        select_trials(reports)


def test_trial_runner_freezes_policies_without_holdout_evaluation(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    with h5py.File(path, 'w') as stream:
        stream.attrs['identity'] = json.dumps({'config': {'seed': 1}, 'model': 'unchanged'})
        stream.attrs['completed_paths'] = 16
        stream.create_dataset('features', data=np.zeros((16, 2, 3), dtype='f4'))
        stream.create_dataset('returns', data=np.zeros((16, 2, 2), dtype='f4'))
    calls = []

    def fit(*, path, output, fold, protocol, train_config, cache):
        calls.append((fold, train_config.learning_rate))
        output.mkdir(parents=True)
        checkpoint = output / 'checkpoint.pt'
        checkpoint.write_bytes(b'fixture')
        return {'checkpoint': str(checkpoint), 'stop_reason': 'internal_validation_plateau',
            'best_epochs': {'entropy': 1, 'squared': 1},
            'best_internal_losses': {'entropy': -train_config.learning_rate, 'squared': -train_config.learning_rate}}

    monkeypatch.setattr('quantdata.gromm2.conditional_refinement.check_resource_headroom', lambda: None)
    monkeypatch.setattr('quantdata.gromm2.conditional_refinement.train_fold', fit)
    base = LocalAuditTrainConfig(device='cpu', minimum_learning_rate=1.e-8)
    candidates = [replace(base, learning_rate=rate) for rate in (1.e-6, 1.e-5)]
    report = fit_trials(path=path, output=tmp_path / 'output', candidates=candidates,
        protocol=LocalAuditProtocol(), fresh_evaluation_seed=2)
    assert calls == [(0, 1.e-6), (0, 1.e-5), (1, 1.e-6), (1, 1.e-5)]
    assert report['outer_holdout_evaluated'] is False
    assert report['source_identity']['model'] == 'unchanged'
    assert (tmp_path / 'output' / 'frozen-policies.json').is_file()
    with pytest.raises(ValueError, match='independent simulation seed'):
        fit_trials(path=path, output=tmp_path / 'other', candidates=candidates,
            protocol=LocalAuditProtocol(), fresh_evaluation_seed=1)
