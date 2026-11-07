"""Training coordinates preserve the strategy class, accounting and exact restart."""

import os
from copy import deepcopy
from dataclasses import fields, replace
from datetime import date

import numpy as np
import pytest
import torch

from quantdata.hedge2 import terminal_train
from quantdata.hedge2.terminal_coordinates import (
    ActorInputCoordinates,
    TerminalCoordinateMap,
    fit_actor_coordinates,
)
from quantdata.hedge2.terminal_descent import gradient_vector, parameter_vector
from quantdata.hedge2.terminal_features import TerminalFeatureMapping
from quantdata.hedge2.terminal_torch import TensorTerminalPaths
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    audit_fitted_policy,
    load_terminal_policy,
    make_terminal_evaluator,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_descent import arguments
from tests.quantdata.hedge2.test_terminal_history_compaction import branching_inputs


def evaluator(paths, book, policy, inventory, *, compact=False, objective="mean_variance"):
    return make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
        book=book.tensor(policy.env), initial_inventory=inventory, objective=objective, risk_aversion=10.,
        target_side=-1, history_ids=paths.history_ids if compact else None)


def fitted_coordinates(paths, book, policy, inventory, *, mode="whitened", compact=False):
    return fit_actor_coordinates(policy=policy, evaluate=evaluator(paths, book, policy, inventory, compact=compact),
        weights=torch.tensor(paths.weights, dtype=policy.fdtype), steps=paths.gains.shape[1],
        instruments=paths.gains.shape[2], fitting_data_sha256="a" * 64, mode=mode,
        history_ids=paths.history_ids if compact else None)


@pytest.mark.parametrize("mode", ["diagonal", "whitened"])
@pytest.mark.parametrize("fee_preconditioning", [False, True])
def test_streamed_training_moments_match_explicit_path_time_asset_weights(mode, fee_preconditioning) -> None:
    paths, book, policy, inventory = branching_inputs(fee_preconditioning=fee_preconditioning)
    original = deepcopy(policy.state_dict())
    rows = []
    handle = policy.affine.I0.register_forward_pre_hook(lambda module, inputs: rows.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad():
            evaluator(paths, book, policy, inventory)()
    finally:
        handle.remove()
    values = torch.stack(rows, dim=1)
    weights = torch.tensor(paths.weights, dtype=policy.fdtype)
    mean = (values.mean(dim=(1, 2)) * weights[:, None]).sum(dim=0)
    centered = (values - mean).reshape(-1, values.shape[-1])
    probabilities = weights.repeat_interleave(values.shape[1] * values.shape[2]) / (values.shape[1] * values.shape[2])
    covariance = centered.T @ (centered * probabilities[:, None])
    full = fitted_coordinates(paths, book, policy, inventory, mode=mode)
    compact = fitted_coordinates(paths, book, policy, inventory, mode=mode, compact=True)
    torch.testing.assert_close(full.mean, mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(compact.mean, full.mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(compact.matrix, full.matrix, rtol=1e-9, atol=1e-11)
    if mode == "diagonal":
        scales = torch.sqrt(torch.clamp_min(covariance.diag(), 0.))
        expected = torch.diag(torch.where(scales > 1e-8, scales, 1.))
    else:
        eigenvalues, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
        floor = 1e-6 * max(float(eigenvalues.max()), 1.)
        expected = (vectors * torch.sqrt(torch.clamp_min(eigenvalues, floor))[None, :]) @ vectors.T
    torch.testing.assert_close(full.matrix, expected, rtol=1e-9, atol=1e-11)
    assert full.fitting_summary["probability_mass"] == pytest.approx(1.)
    assert not policy.affine.I0._forward_pre_hooks
    for name, value in original.items():
        torch.testing.assert_close(policy.state_dict()[name], value, rtol=0., atol=0.)


def test_compact_moments_preserve_original_order_for_shuffled_all_unique_histories() -> None:
    paths, book, policy, inventory = branching_inputs()
    histories = paths.history_ids.copy()
    histories[:, -1] = [8, 4, 7, 2, 9, 0, 6, 3]
    paths = replace(paths, history_ids=histories)
    full = fitted_coordinates(paths, book, policy, inventory)
    compact = fitted_coordinates(paths, book, policy, inventory, compact=True)
    torch.testing.assert_close(compact.mean, full.mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(compact.matrix, full.matrix, rtol=1e-9, atol=1e-11)


def test_coordinates_use_actual_model_tensor_device_for_indexed_cpu_environment() -> None:
    from quantdata.hedge2.terminal_policy import TerminalPolicy

    paths, book, original, inventory = branching_inputs()
    policy = TerminalPolicy(env=BaseEnvironment(device="cpu:0", precision=64), config=original.config)
    policy.load_state_dict(original.state_dict())
    coordinates = fitted_coordinates(paths, book, policy, inventory)
    expected = fitted_coordinates(paths, book, original, inventory)
    torch.testing.assert_close(coordinates.mean, expected.mean, rtol=0., atol=0.)
    torch.testing.assert_close(coordinates.matrix, expected.matrix, rtol=0., atol=0.)
    restored = ActorInputCoordinates.from_record(coordinates.record(), policy=policy,
        expected_data_sha256="a" * 64, expected_mode="whitened")
    mapping = TerminalCoordinateMap(policy=policy, coordinates=restored)
    mapping.activate()
    mapping.remove()


@pytest.mark.skipif(os.environ.get("HEDGE2_GPU_TEST") != "1", reason="Requires an explicitly gpuctl-allocated GPU")
def test_coordinates_accept_unindexed_cuda_environment_and_concrete_model_device() -> None:
    from quantdata.hedge2.terminal_policy import TerminalPolicy

    paths, book, original, inventory = branching_inputs()
    policy = TerminalPolicy(env=BaseEnvironment(device="cuda", precision=64), config=original.config)
    policy.load_state_dict(original.state_dict())
    inventory = inventory.to(policy.affine.I0.weight.device)
    evaluate = evaluator(paths, book, policy, inventory, compact=True)
    coordinates = fit_actor_coordinates(policy=policy, evaluate=evaluate,
        weights=torch.tensor(paths.weights, dtype=policy.fdtype, device=policy.affine.I0.weight.device),
        steps=paths.gains.shape[1], instruments=paths.gains.shape[2], fitting_data_sha256="a" * 64,
        mode="whitened", history_ids=paths.history_ids)
    mapping = TerminalCoordinateMap(policy=policy, coordinates=coordinates)
    mapping.activate()
    result, _ = evaluate()
    assert torch.isfinite(result.wealth).all()
    mapping.remove()


@pytest.mark.parametrize("mode", ["diagonal", "whitened"])
@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
@pytest.mark.parametrize("fee_preconditioning", [False, True])
@pytest.mark.parametrize("use_deep", [False, True])
def test_coordinates_preserve_all_path_values_raw_gradients_and_inference(mode, objective, fee_preconditioning, use_deep) -> None:
    paths, book, original, inventory = branching_inputs(fee_preconditioning=fee_preconditioning, use_deep=use_deep)
    policy = deepcopy(original)
    raw_evaluate = evaluator(paths, book, original, inventory, objective=objective)
    raw_result, raw_risk = raw_evaluate()
    raw_risk.value.backward()
    raw_gradient = gradient_vector(original.parameters())
    coordinates = fitted_coordinates(paths, book, policy, inventory, mode=mode, compact=True)
    mapping = TerminalCoordinateMap(policy=policy, coordinates=coordinates)
    raw_vector = parameter_vector(policy.parameters())
    optimized = mapping.vector(raw_vector, to_physical=False)
    torch.testing.assert_close(mapping.vector(optimized), raw_vector, rtol=1e-10, atol=1e-11)
    mapping.activate()
    try:
        mapped_result, mapped_risk = evaluator(paths, book, policy, inventory, compact=True, objective=objective)()
        mapped_risk.value.backward()
        optimized_gradient = gradient_vector(policy.parameters())
        torch.testing.assert_close(mapping.vector(optimized_gradient, gradient=True), raw_gradient, rtol=1e-9, atol=1e-10)
        for field in fields(raw_result):
            torch.testing.assert_close(getattr(mapped_result, field.name), getattr(raw_result, field.name), rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(mapped_risk.value, raw_risk.value, rtol=1e-12, atol=1e-12)
        direction = torch.cos(torch.arange(len(raw_vector), dtype=raw_vector.dtype))
        torch.testing.assert_close(optimized_gradient @ mapping.vector(direction, to_physical=False),
            mapping.vector(optimized_gradient, gradient=True) @ direction, rtol=1e-10, atol=1e-11)
        inference = deepcopy(original)
        inference.load_state_dict(mapping.state_dict(policy.state_dict(), to_physical=True))
        independent, risk = evaluator(paths, book, inference, inventory, objective=objective)()
        torch.testing.assert_close(independent.wealth, mapped_result.wealth, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(risk.value, mapped_risk.value, rtol=1e-12, atol=1e-12)
        assert not inference.affine.I0._forward_pre_hooks
        with pytest.raises(RuntimeError, match="already active"):
            mapping.activate()
    finally:
        mapping.remove()
    assert not policy.affine.I0._forward_pre_hooks


@pytest.mark.parametrize("mode", ["diagonal", "whitened"])
@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
def test_coordinate_training_restores_exact_curvature_history_and_raw_inference(tmp_path, monkeypatch, mode, objective) -> None:
    inputs = arguments()
    inputs["objective"] = objective
    training = TerminalTrainConfig(optimizer="lbfgs", actor_coordinates=mode, compact_histories=True,
        maximum_updates=12, minimum_updates=1, check_every=2, stable_checks=100, lbfgs_learning_rate=.1)
    whole = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "whole")
    partial = train_terminal_policy(**inputs, train_config=replace(training, maximum_updates=6), output_directory=tmp_path / "split")
    assert partial.updates == 6

    def unexpected_refit(**kwargs):
        raise AssertionError("Same-date restart must restore its original input coordinates")

    monkeypatch.setattr(terminal_train, "fit_actor_coordinates", unexpected_refit)
    resumed = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "split")
    assert resumed.updates == whole.updates == 12
    assert resumed.history == whole.history
    torch.testing.assert_close(parameter_vector(resumed.policy.parameters()), parameter_vector(whole.policy.parameters()), rtol=0., atol=0.)
    records = [torch.load(fit.checkpoint_path, weights_only=True) for fit in (whole, resumed)]
    for name in ("optimization_model", "model", "best_model"):
        for key, value in records[0][name].items():
            torch.testing.assert_close(records[1][name][key], value, rtol=0., atol=0.)
    for name, value in records[0]["optimizer"]["state"][0].items():
        other = records[1]["optimizer"]["state"][0][name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(other, value, rtol=0., atol=0.)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            for left, right in zip(value, other, strict=True):
                torch.testing.assert_close(left, right, rtol=0., atol=0.)
        else:
            assert other == value
    assert records[0]["actor_coordinate_binding"] == records[1]["actor_coordinate_binding"]
    loaded = load_terminal_policy(resumed.checkpoint_path, env=inputs["env"])
    torch.testing.assert_close(parameter_vector(loaded.parameters()), parameter_vector(resumed.policy.parameters()), rtol=0., atol=0.)
    audit = audit_fitted_policy(policy=loaded, paths=inputs["paths"], book=inputs["book"],
        initial_inventory=inputs["initial_inventory"], objective=objective, risk_aversion=10., target_side=-1)
    assert audit["maximum_replay_error"] < 1e-12
    assert not loaded.affine.I0._forward_pre_hooks
    assert not resumed.policy.affine.I0._forward_pre_hooks


@pytest.mark.parametrize("damaged", ["actor_coordinates", "optimization_model", "optimization_layout", "model", "best_model"])
def test_incomplete_or_changed_coordinate_binding_is_rejected_without_overwriting(tmp_path, damaged) -> None:
    inputs = arguments()
    training = TerminalTrainConfig(optimizer="lbfgs", actor_coordinates="whitened", maximum_updates=2,
        minimum_updates=1, check_every=1, stable_checks=100)
    fit = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path)
    record = torch.load(fit.checkpoint_path, weights_only=True)
    if damaged in ("model", "best_model"):
        record[damaged]["affine.I0.bias"].add_(1.)
    else:
        del record[damaged]
    torch.save(record, fit.checkpoint_path)
    original = fit.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="checkpoint"):
        load_terminal_policy(fit.checkpoint_path, env=inputs["env"])
    assert fit.checkpoint_path.read_bytes() == original


def test_new_rolling_date_rebases_physical_weights_then_fits_new_coordinates(tmp_path, monkeypatch) -> None:
    inputs = arguments()
    training = TerminalTrainConfig(optimizer="lbfgs", actor_coordinates="whitened", maximum_updates=2,
        minimum_updates=1, check_every=1, stable_checks=100)
    prior = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "prior")
    old_weights = deepcopy(prior.policy.state_dict())
    initial = parameter_vector(prior.policy.parameters())
    captured = []
    original = terminal_train.fit_actor_coordinates

    def capture(**kwargs):
        captured.append(parameter_vector(kwargs["policy"].parameters()))
        assert not kwargs["policy"].affine.I0._forward_pre_hooks
        return original(**kwargs)

    monkeypatch.setattr(terminal_train, "fit_actor_coordinates", capture)
    inputs["as_of_date"] = date(2024, 4, 22)
    current = train_terminal_policy(**inputs, train_config=training, output_directory=tmp_path / "current",
        previous_policy=prior.policy, feature_mapping=TerminalFeatureMapping(matrix=np.eye(1), offset=np.zeros(1)))
    assert current.initialisation_date == prior.policy.as_of_date
    torch.testing.assert_close(captured[0], initial, rtol=1e-13, atol=1e-13)
    for name, value in old_weights.items():
        torch.testing.assert_close(prior.policy.state_dict()[name], value, rtol=0., atol=0.)


def test_coordinate_records_reject_incompatible_modes_data_rank_and_provenance() -> None:
    paths, book, policy, inventory = branching_inputs()
    coordinates = fitted_coordinates(paths, book, policy, inventory)
    record = coordinates.record()
    restored = ActorInputCoordinates.from_record(record, policy=policy, expected_data_sha256="a" * 64, expected_mode="whitened")
    torch.testing.assert_close(restored.matrix, coordinates.matrix, rtol=0., atol=0.)
    for changes in ({"expected_data_sha256": "b" * 64}, {"expected_mode": "diagonal"}):
        kwargs = {"expected_data_sha256": "a" * 64, "expected_mode": "whitened", **changes}
        with pytest.raises(ValueError, match="checkpoint"):
            ActorInputCoordinates.from_record(record, policy=policy, **kwargs)
    with pytest.raises(ValueError, match="invertible"):
        replace(coordinates, matrix=torch.zeros_like(coordinates.matrix))
    with pytest.raises(ValueError, match="provenance"):
        replace(coordinates, fitting_summary={**coordinates.fitting_summary, "probability_mass": float("nan")})


def test_coordinate_mode_is_opt_in_and_requires_float64_lbfgs() -> None:
    assert TerminalTrainConfig().actor_coordinates == "none"
    for mode in ("unsupported", "diagonal", "whitened"):
        with pytest.raises(ValueError, match="coordinates"):
            TerminalTrainConfig(actor_coordinates=mode)
