"""Descent checks preserve the full objective, parameters, gradients and restart."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
import torch

from quantdata.hedge2 import terminal_train
from quantdata.hedge2.terminal_descent import parameter_vector, probe_terminal_descent
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    audit_fitted_policy,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_policy import sample_data


def test_bidirectional_probe_detects_descent_without_changing_parameters_or_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
    parameter.grad = torch.tensor([4., -2.], dtype=torch.float64)
    original, gradient = parameter.detach().clone(), parameter.grad.clone()
    result = probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.square().sum(),
        directions=[-parameter.grad], relative_distances=(1e-4, .01), tolerance=1e-6)
    assert not result.no_detected_descent
    assert result.evaluations == 5
    torch.testing.assert_close(parameter, original, rtol=0., atol=0.)
    torch.testing.assert_close(parameter.grad, gradient, rtol=0., atol=0.)


def test_probe_at_quadratic_minimum_finds_no_descent() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
    result = probe_terminal_descent(parameters=[parameter],
        evaluate_loss=lambda: (parameter - torch.tensor([2., -1.])).square().sum(),
        directions=[torch.ones_like(parameter)], relative_distances=(1e-6, .01), tolerance=1e-6)
    assert result.no_detected_descent
    assert result.maximum_improvement == 0.


@pytest.mark.parametrize("raises", [True, False])
def test_probe_restores_parameters_after_exception_or_nonfinite_objective(raises) -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    original = parameter.detach().clone()
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls > 1:
            if raises:
                raise RuntimeError("Deliberate evaluation failure")
            return torch.tensor(float("nan"))
        return parameter.square().sum()

    with pytest.raises(RuntimeError):
        probe_terminal_descent(parameters=[parameter], evaluate_loss=loss,
            directions=[torch.ones_like(parameter)], relative_distances=(.01,), tolerance=1e-6)
    torch.testing.assert_close(parameter, original, rtol=0., atol=0.)


def test_float32_probes_that_round_away_are_not_numerical_evidence() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
    result = probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.square().sum(),
        directions=[torch.ones_like(parameter)], relative_distances=(1e-30,), tolerance=1e-6)
    assert result.evaluations == 1
    assert result.rounded_away == 2


@pytest.mark.parametrize("distances", [(), (0.,), (float("nan"),), (1e-3, 1e-3)])
def test_probe_rejects_invalid_distances(distances) -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):
        probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.sum(),
            directions=[torch.ones_like(parameter)], relative_distances=distances, tolerance=1e-6)


def arguments():
    paths, book = sample_data()
    return dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="lbfgs-fit", feature_count=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")


def test_lbfgs_resume_preserves_accepted_weights_curvature_and_history(tmp_path) -> None:
    inputs = arguments()
    training = TerminalTrainConfig(optimizer="lbfgs", maximum_updates=12, minimum_updates=1,
        check_every=2, stable_checks=100, lbfgs_learning_rate=.1)
    whole = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "whole")
    partial = train_terminal_policy(**inputs, train_config=replace(training, maximum_updates=6),
        output_directory=tmp_path / "split")
    assert partial.status == "training_limit"
    resumed = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "split")
    assert resumed.updates == whole.updates == 12
    assert resumed.history == whole.history
    torch.testing.assert_close(parameter_vector(resumed.policy.parameters()),
        parameter_vector(whole.policy.parameters()), rtol=0., atol=0.)
    retained = torch.load(resumed.checkpoint_path, weights_only=True)
    assert retained["optimizer_type"] == "lbfgs"
    assert retained["acceptance_protocol"] == "lbfgs-training-descent-v1"
    audit = audit_fitted_policy(policy=resumed.policy, paths=inputs["paths"], book=inputs["book"],
        initial_inventory=inputs["initial_inventory"], objective="mean_variance", risk_aversion=10., target_side=-1)
    assert audit["maximum_replay_error"] < 1e-12


def test_zero_parameter_step_cannot_manufacture_convergence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.optim.LBFGS, "step", lambda optimizer, closure: closure())
    fit = train_terminal_policy(**arguments(),
        train_config=TerminalTrainConfig(optimizer="lbfgs", maximum_updates=10, minimum_updates=1,
            stable_checks=1, check_every=1), output_directory=tmp_path)
    assert fit.status == "line_search_review_required"
    assert fit.updates == 1
    assert fit.history[-1]["parameter_step_l2"] == 0.
    assert not fit.history[-1]["accepted_changed_step"]


def test_true_zero_gradient_is_retained_for_review_not_fabricated_stable_history(tmp_path) -> None:
    inputs = arguments()
    inputs["paths"] = replace(inputs["paths"], claim_gain=np.zeros_like(inputs["paths"].claim_gain))
    fit = train_terminal_policy(**inputs,
        train_config=TerminalTrainConfig(optimizer="lbfgs", maximum_updates=10, minimum_updates=1,
            stable_checks=1, check_every=1), output_directory=tmp_path)
    assert fit.status == "stationary_review_required"
    assert fit.updates == 1


def test_optimizer_cannot_change_in_place(tmp_path) -> None:
    inputs = arguments()
    training = TerminalTrainConfig(maximum_updates=1, minimum_updates=1)
    fit = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path)
    before = fit.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="explicit migration"):
        train_terminal_policy(**inputs, train_config=replace(training, optimizer="lbfgs"), output_directory=tmp_path)
    assert fit.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("changes", [{"descent_probe_distances": (1e-5, .01)},
    {"relative_loss_tolerance": 1e-5}, {"lbfgs_history_size": 10}, {"check_every": 1}])
def test_changed_acceptance_or_lbfgs_settings_do_not_reuse_saved_counters(tmp_path, changes) -> None:
    inputs = arguments()
    training = TerminalTrainConfig(optimizer="lbfgs", maximum_updates=2, minimum_updates=1,
        check_every=2, stable_checks=100)
    fit = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path)
    before = fit.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="numerical protocol"):
        train_terminal_policy(**inputs, train_config=replace(training, **changes), output_directory=tmp_path)
    assert fit.checkpoint_path.read_bytes() == before


def test_review_required_status_survives_budget_extension_without_explicit_retry(tmp_path, monkeypatch) -> None:
    calls = 0

    def stalled_step(optimizer, closure):
        nonlocal calls
        calls += 1
        return closure()

    monkeypatch.setattr(torch.optim.LBFGS, "step", stalled_step)
    inputs = arguments()
    training = TerminalTrainConfig(optimizer="lbfgs", maximum_updates=2, minimum_updates=1,
        stable_checks=1, check_every=1)
    first = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path)
    second = train_terminal_policy(**inputs, train_config=replace(training, maximum_updates=4), output_directory=tmp_path)
    assert first.status == second.status == "line_search_review_required"
    assert first.updates == second.updates == 1
    assert calls == 1
    third = train_terminal_policy(**inputs, train_config=replace(training, maximum_updates=4, retry_review=True),
        output_directory=tmp_path)
    assert third.status == "line_search_review_required"
    assert third.updates == 2
    assert calls == 2


def test_unvalidated_float32_lbfgs_cannot_claim_float64_acceptance(tmp_path) -> None:
    inputs = arguments()
    inputs["env"] = BaseEnvironment(device="cpu", precision=32)
    with pytest.raises(ValueError, match="float64"):
        train_terminal_policy(**inputs, train_config=TerminalTrainConfig(optimizer="lbfgs"),
            output_directory=tmp_path)


def test_roundoff_dominated_nonzero_step_is_not_a_verified_line_search(tmp_path, monkeypatch) -> None:
    def tiny_step(optimizer, closure):
        loss = closure()
        with torch.no_grad():
            for parameter in optimizer.param_groups[0]["params"]:
                if parameter.grad is not None:
                    parameter.add_(-torch.finfo(parameter.dtype).eps * parameter.grad.sign())
        return loss

    monkeypatch.setattr(torch.optim.LBFGS, "step", tiny_step)
    fit = train_terminal_policy(**arguments(),
        train_config=TerminalTrainConfig(optimizer="lbfgs", maximum_updates=10, minimum_updates=1,
            stable_checks=1, check_every=1), output_directory=tmp_path)
    assert fit.status == "line_search_review_required"
    assert fit.history[-1]["parameter_step_l2"] > 0.
    assert not fit.history[-1]["line_search_numerically_resolved"]


def test_stable_candidate_and_saved_best_weights_are_the_actual_probed_iterate(tmp_path, monkeypatch) -> None:
    recorded = []
    original = terminal_train.probe_terminal_descent

    def capture(**kwargs):
        recorded.append(parameter_vector(kwargs["parameters"]).clone())
        return original(**kwargs)

    monkeypatch.setattr(terminal_train, "probe_terminal_descent", capture)
    fit = train_terminal_policy(**arguments(),
        train_config=TerminalTrainConfig(optimizer="lbfgs", maximum_updates=10, minimum_updates=1,
            stable_checks=1, check_every=1, relative_loss_tolerance=.1, dollar_action_tolerance=10.),
        output_directory=tmp_path)
    assert fit.status == "training_stable"
    assert recorded
    torch.testing.assert_close(parameter_vector(fit.policy.parameters()), recorded[-1], rtol=0., atol=0.)
    assert fit.history[-1]["descent_probe"]["evaluations"] > 1


def test_bidirectional_probes_cover_a_nonsmooth_turnover_minimum() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    result = probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.abs().sum(),
        directions=[torch.ones_like(parameter)], relative_distances=(1e-6, .01), tolerance=1e-6)
    assert result.no_detected_descent
    assert result.evaluations == 5
    assert float(parameter.detach()) == 0.


def test_transformed_probes_use_original_parameter_distances_and_restore_exact_optimizer_state() -> None:
    matrix = torch.tensor([[3., .4], [.1, .25]], dtype=torch.float64)
    inverse = torch.linalg.inv(matrix)
    raw = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
    optimized = torch.nn.Parameter(matrix @ raw.detach())
    optimized.grad = torch.tensor([7., 8.], dtype=torch.float64)
    saved = optimized.detach().clone()
    direction = torch.tensor([1., 2.], dtype=torch.float64)
    reference = probe_terminal_descent(parameters=[raw], evaluate_loss=lambda: raw.square().sum(),
        directions=[direction], relative_distances=(1e-4, .01), tolerance=1e-6)
    actual = probe_terminal_descent(parameters=[optimized], evaluate_loss=lambda: (inverse @ optimized).square().sum(),
        directions=[direction], relative_distances=(1e-4, .01), tolerance=1e-6,
        to_probe_coordinates=lambda vector: inverse @ vector,
        from_probe_coordinates=lambda vector: matrix @ vector)
    assert actual.baseline_loss == pytest.approx(reference.baseline_loss, abs=1e-13)
    assert actual.best_loss == pytest.approx(reference.best_loss, abs=1e-13)
    assert actual.evaluations == reference.evaluations
    torch.testing.assert_close(optimized, saved, rtol=0., atol=0.)
    torch.testing.assert_close(optimized.grad, torch.tensor([7., 8.], dtype=torch.float64), rtol=0., atol=0.)


def test_mapped_probe_exception_restores_optimizer_parameters_and_rejects_half_mapping() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    original = parameter.detach().clone()
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("Deliberate mapped failure")
        return parameter.square().sum()

    with pytest.raises(RuntimeError, match="mapped failure"):
        probe_terminal_descent(parameters=[parameter], evaluate_loss=loss, directions=[torch.ones_like(parameter)],
            relative_distances=(.01,), tolerance=1e-6, to_probe_coordinates=lambda vector: vector / 3.,
            from_probe_coordinates=lambda vector: 3. * vector)
    torch.testing.assert_close(parameter, original, rtol=0., atol=0.)
    with pytest.raises(ValueError, match="probes need"):
        probe_terminal_descent(parameters=[parameter], evaluate_loss=loss, directions=[torch.ones_like(parameter)],
            relative_distances=(.01,), tolerance=1e-6, to_probe_coordinates=lambda vector: vector / 3.)


def test_roundtrip_error_cannot_turn_a_rounded_away_raw_probe_into_evidence() -> None:
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    result = probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.square().sum(),
        directions=[torch.ones_like(parameter)], relative_distances=(1e-30,), tolerance=1e-6,
        to_probe_coordinates=lambda vector: vector,
        from_probe_coordinates=lambda vector: torch.nextafter(vector, torch.full_like(vector, float("inf"))))
    assert result.evaluations == 1
    assert result.rounded_away == 2
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0., atol=0.)


def test_probe_rejects_coordinate_roundtrip_error_larger_than_the_intended_perturbation() -> None:
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="roundoff obscures"):
        probe_terminal_descent(parameters=[parameter], evaluate_loss=lambda: parameter.square().sum(),
            directions=[torch.ones_like(parameter)], relative_distances=(1e-6,), tolerance=1e-6,
            to_probe_coordinates=lambda vector: vector, from_probe_coordinates=lambda vector: vector + .01)
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0., atol=0.)
