from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
import torch

from quantdata.gromm2 import validation_action_optimization
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.metric_audit_optimization import MetricSolveConfig, fit_metric_action
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig, full_action_pass
from quantdata.gromm2.validation_action_optimization import (
    SELECTION_SOURCE,
    ValidationSelectionConfig,
    fit_validation_selected_action,
    validation_action_value,
)
from quantdata.torch import BaseEnvironment


def fixture():
    generator = torch.Generator().manual_seed(61)
    states = torch.randn((80, 3), generator=generator, dtype=torch.float64)
    returns = .2*states[:, :2]+.1*states[:, :2].square()
    returns[48:64] = -returns[48:64]
    eligible = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
    arguments = dict(states=states, returns=returns, eligible=eligible,
        action_scale=.2, config=AffineSolveConfig(batch_size=13))
    training = AffineAuditProblem(rows=torch.arange(48), **arguments)
    validation = AffineAuditProblem(rows=torch.arange(48, 64), **arguments)
    torch.manual_seed(71)
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='test', feature_count=3, instrument_count=2, action_scale=.2,
            network=ActionNetworkConfig(num_depth=1, num_width=6)))
    return training, validation, policy


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_validation_matches_original_objective_without_mutating_training_state(diagnostic):
    training, validation, policy = fixture()
    with torch.no_grad():
        policy.affine.I0.bias.fill_(.3)
    policy.zero_grad(set_to_none=True)
    full_action_pass(policy, training, diagnostic=diagnostic, backward=True)
    parameters = deepcopy(policy.state_dict())
    gradients = [parameter.grad.clone() for parameter in policy.parameters()]
    random_state = torch.get_rng_state().clone()
    observed = validation_action_value(policy, validation, diagnostic=diagnostic)
    expected = float(full_action_pass(policy, validation, diagnostic=diagnostic, backward=False))
    value = expected if diagnostic == 'squared' else torch.log1p(torch.tensor(expected*.2**2)).item()
    assert observed['value'] == pytest.approx(value, abs=1.e-8)
    assert torch.equal(random_state, torch.get_rng_state())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0., atol=0.)
    for parameter, gradient in zip(policy.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0., atol=0.)


def test_entropy_validation_uses_exact_log_objective_when_plain_exponential_overflows():
    _, validation, policy = fixture()
    with torch.no_grad():
        policy.affine.I0.bias.fill_(1000.)
    assert not torch.isfinite(full_action_pass(policy, validation, diagnostic='entropy', backward=False))
    observed = validation_action_value(policy, validation, diagnostic='entropy')
    assert observed['value'] > 10000.
    assert observed['observations_dropped'] == 0


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_selection_changes_no_training_updates_when_patience_does_not_expire(tmp_path, diagnostic):
    training, validation, policy = fixture()
    original = deepcopy(policy)
    controls = MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=6,
        gradient_relative_tolerance=1.e-12, gradient_absolute_tolerance=1.e-15))
    baseline = fit_metric_action(policy=original, problem=training, diagnostic=diagnostic,
        data_identity='fixture', config=controls)
    report = fit_validation_selected_action(policy=policy, training=training, validation=validation,
        diagnostic=diagnostic, data_identity='fixture', output=tmp_path/'selected', optimizer_config=controls,
        selection_config=ValidationSelectionConfig(interval=1, patience=20))
    terminal = torch.load(tmp_path/'selected/resume.pt', weights_only=False)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(terminal['model'][name], value, rtol=0., atol=0.)
    assert terminal['native_state']['history'][-1]['training_loss'] == baseline['history'][-1]['training_loss']
    assert report['numerical']['stop_reason'] == baseline['stop_reason']
    assert report['numerical']['converged'] is False
    assert report['confirmation_read'] is False
    assert report['selection']['best_iteration'] >= 1
    selected = torch.load(report['selected_weights'], weights_only=False)
    assert selected['policy_state_source'] == SELECTION_SOURCE
    assert selected['selected']['iteration'] == report['selection']['best_iteration']
    assert selected['selected']['mean_squared_unit_action'] > 0.
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_declared_patience_stops_and_restores_best_not_terminal_without_zero_fallback(tmp_path):
    training, validation, policy = fixture()
    report = fit_validation_selected_action(policy=policy, training=training, validation=validation,
        diagnostic='squared', data_identity='fixture', output=tmp_path,
        optimizer_config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=20)),
        selection_config=ValidationSelectionConfig(interval=1, patience=2,
            improvement_absolute=1.e10, improvement_relative=0.))
    assert report['selection']['stop_reason'] == 'declared_inner_validation_patience'
    assert report['numerical']['history'][-1]['iteration'] == 3
    assert report['numerical']['converged'] is False
    assert report['policy_state_source'] == SELECTION_SOURCE
    assert len(report['selection']['history']) == 3
    selected = torch.load(report['selected_weights'], weights_only=False)
    for name, actual in policy.state_dict().items():
        torch.testing.assert_close(actual, selected['state_dict'][name], rtol=0., atol=0.)


def test_accepted_checkpoint_restart_deduplicates_validation_and_preserves_native_training(tmp_path):
    training, validation, policy = fixture()
    baseline = deepcopy(policy)
    controls = dict(diagnostic='squared', data_identity='fixture', training=training, validation=validation,
        optimizer_config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=8)),
        selection_config=ValidationSelectionConfig(interval=1, patience=2,
            improvement_absolute=1.e10, improvement_relative=0.))
    expected = fit_validation_selected_action(policy=baseline, output=tmp_path/'full', **controls)

    def interrupt(entry):
        if entry['stage'] == 'metric_full_gradient_iteration' and entry['iteration'] == 2:
            raise InterruptedError('accepted checkpoint saved')

    with pytest.raises(InterruptedError):
        fit_validation_selected_action(policy=policy, output=tmp_path/'resumed', progress=interrupt, **controls)
    actual = fit_validation_selected_action(policy=policy, output=tmp_path/'resumed', **controls)
    assert actual['selection']['history'] == expected['selection']['history']
    assert actual['selection']['stale_checks'] == expected['selection']['stale_checks']
    assert actual['numerical']['history'][-1]['iteration'] == expected['numerical']['history'][-1]['iteration']
    for parameter, target in zip(policy.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(parameter, target, rtol=0., atol=0.)


def test_nonfinite_validation_or_overlap_fails_without_certifying_a_fit(tmp_path):
    training, validation, policy = fixture()
    validation.rows[0] = training.rows[0]
    with pytest.raises(ValueError, match='overlap'):
        fit_validation_selected_action(policy=policy, training=training, validation=validation,
            diagnostic='squared', data_identity='fixture', output=tmp_path)
    validation.rows[0] = 48
    validation.returns[48] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        fit_validation_selected_action(policy=policy, training=training, validation=validation,
            diagnostic='squared', data_identity='fixture', output=tmp_path)
    assert not (tmp_path/'resume.pt').exists()


def test_changed_selected_artifact_or_controls_rejected_on_restart(tmp_path):
    training, validation, policy = fixture()
    controls = dict(policy=policy, training=training, validation=validation, diagnostic='squared',
        data_identity='fixture', output=tmp_path,
        optimizer_config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=3)))
    report = fit_validation_selected_action(**controls)
    selected = tmp_path/report['selection']['best_filename']
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == report['selected_weights_sha256']
    selected.write_bytes(b'changed')
    with pytest.raises(ValueError, match='accepted resume ledger'):
        fit_validation_selected_action(**controls)
    with pytest.raises(ValueError, match='identity or declared controls'):
        fit_validation_selected_action(**controls,
            selection_config=ValidationSelectionConfig(interval=2))


def test_interrupt_after_selection_publication_resumes_committed_candidate(tmp_path):
    training, validation, policy = fixture()
    controls = dict(policy=policy, training=training, validation=validation,
        diagnostic='squared', data_identity='fixture', output=tmp_path,
        optimizer_config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=4)),
        selection_config=ValidationSelectionConfig(interval=1, patience=20))

    def interrupt(entry):
        if entry['stage'] == 'inner_validation_selection':
            raise InterruptedError('selection event is committed first')

    with pytest.raises(InterruptedError):
        fit_validation_selected_action(**controls, progress=interrupt)
    saved = torch.load(tmp_path/'resume.pt', weights_only=False)
    assert saved['selection']['best_iteration'] == 1
    report = fit_validation_selected_action(**controls)
    assert [entry['iteration'] for entry in report['selection']['history']] == [1, 2, 3, 4]


def test_candidate_published_before_failed_native_commit_remains_unselected(tmp_path, monkeypatch):
    training, validation, policy = fixture()
    controls = dict(policy=policy, training=training, validation=validation,
        diagnostic='squared', data_identity='fixture', output=tmp_path,
        optimizer_config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=3)),
        selection_config=ValidationSelectionConfig(interval=1, patience=20))
    original = validation_action_optimization._atomic_torch_save

    def failed_commit(path, values):
        if path.name == 'resume.pt' and values['selection']['best_iteration'] == 1:
            raise OSError('simulated failure after candidate, before selection commit')
        original(path, values)

    monkeypatch.setattr(validation_action_optimization, '_atomic_torch_save', failed_commit)
    with pytest.raises(OSError):
        fit_validation_selected_action(**controls)
    orphan = next(tmp_path.glob('selected-000001-*.pt'))
    saved = torch.load(tmp_path/'resume.pt', weights_only=False)
    assert saved['selection']['best_sha256'] is None
    monkeypatch.setattr(validation_action_optimization, '_atomic_torch_save', original)
    report = fit_validation_selected_action(**controls)
    assert report['selected_weights'] != str(orphan.resolve())
    assert orphan.exists()


def test_no_accepted_nonzero_candidate_is_not_a_martingale_pass(tmp_path, monkeypatch):
    training, validation, policy = fixture()

    def immediate_return(**kwargs):
        return {'stop_reason': 'full_and_affine_metric_gradient_tolerances', 'converged': True}

    monkeypatch.setattr(validation_action_optimization, 'fit_metric_action', immediate_return)
    with pytest.raises(RuntimeError, match='no zero fallback'):
        fit_validation_selected_action(policy=policy, training=training, validation=validation,
            diagnostic='squared', data_identity='fixture', output=tmp_path)
