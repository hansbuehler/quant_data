from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cdxcore.subdir import SubDir

import quantdata.torch as quantdata_torch
from quantdata.dysanos.grid.rndysanos import (
    DeterministicTimeAction,
    DysanosConfig,
    DysanosModel,
    TrainConfig,
    apply_direction_mask,
    cap_objective_derivative,
    conditional_variance,
    cost_components,
    entropy_diagonal_mean_variance_action,
    entropy_log_moment_loss,
    instrument_returns_and_values,
    lower_tail_weights,
    oce_objective,
    prepare_universe,
    split_training_validation,
    train,
    valuate,
    valuate_with_cap_derivative,
)
from quantdata.dysanos.grid.rndysanos.run_experiment import (
    _baseline_split,
    _instrument_outputs,
    _parser,
    attach_arb_direction_mask,
    convergence_summary,
    martingale_summary,
    pair_training_validation_universes,
    stage_universes,
)
from quantdata.dysanos.grid.rndysanos.torch_drift_change import (
    ScaledTanhNetwork,
    _restore_rng_state,
)
from quantdata.torch import BaseEnvironment, BaseModel, DeviceLock
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility


def _universe(
    *,
    samples: int = 8,
    steps: int = 2,
    validation_shift: float = 0.0,
    zero_returns: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    expiries, strikes, state_variables = 1, 2, 3
    option_shape = (samples, steps, expiries, strikes)
    dC = (
        np.zeros(option_shape, dtype=np.float32)
        if zero_returns
        else rng.normal(0.0, 0.2, option_shape).astype(np.float32)
    )
    dS = (
        np.zeros((samples, steps), dtype=np.float32)
        if zero_returns
        else rng.normal(0.0, 0.1, (samples, steps)).astype(np.float32)
    )
    if validation_shift:
        dC[-2:] += validation_shift
        dS[-2:] += validation_shift
    return {
        "C": np.full(option_shape, 8.0, dtype=np.float32),
        "dC": dC,
        "dS": dS,
        "delta": np.full(option_shape, 0.2, dtype=np.float32),
        "expiries": np.array([[[0.5]]], dtype=np.float32),
        "log_strikes": rng.normal(2.3, 0.1, option_shape).astype(np.float32),
        "spot": np.full((samples, steps), 10.0, dtype=np.float32),
        "xstate": rng.normal(size=(samples, steps, state_variables)).astype(np.float32),
    }


def _independent_binary_universe() -> dict[str, np.ndarray]:
    """Three independent returns, each taking values -1 and 2 equally."""

    combinations = np.asarray(
        [
            [spot, first, second]
            for spot in (-1.0, 2.0)
            for first in (-1.0, 2.0)
            for second in (-1.0, 2.0)
        ],
        dtype=np.float32,
    )
    samples = combinations.shape[0]
    option_shape = (samples, 1, 1, 2)
    return {
        "C": np.full(option_shape, 8.0, dtype=np.float32),
        "dC": combinations[:, None, None, 1:].copy(),
        "dS": combinations[:, None, 0].copy(),
        "delta": np.zeros(option_shape, dtype=np.float32),
        "expiries": np.array([[[0.5]]], dtype=np.float32),
        "log_strikes": np.zeros(option_shape, dtype=np.float32),
        "spot": np.full((samples, 1), 10.0, dtype=np.float32),
        "xstate": np.zeros((samples, 1, 1), dtype=np.float32),
    }


def _ff_config() -> FeedForwardConfig:
    return FeedForwardConfig(
        num_depth=1,
        num_width=6,
        activation="LeakyReLU",
        use_bias=True,
    )


def _model_config(label: str, *, cost_cond: float = 0.0, user_version: str = "0.1") -> DysanosConfig:
    network = _ff_config()
    return DysanosConfig(
        label=label,
        user_version=user_version,
        cost_cond=cost_cond,
        pf_config=network,
        moment_config=network,
    )


def _train_config(
    checkpoint_root: Path,
    *,
    validation_fraction: float = 0.25,
) -> TrainConfig:
    return TrainConfig(
        trn_batch_size=3,
        val_batch_size=2,
        validation_fraction=validation_fraction,
        value_seconds=1_000.0,
        value_update=100,
        checkpoint_seconds=1_000.0,
        checkpoint_update=100,
        checkpoint_root=str(checkpoint_root),
    )


def _cpu_env(*, debug: bool = True) -> BaseEnvironment:
    return BaseEnvironment(device="cpu", debug=debug)


def test_torch_environment_feedforward_and_scaled_network_contracts() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    network = ScaledTanhNetwork(
        env,
        num_input=3,
        num_output=2,
        config=_ff_config(),
        initial_scale=1.5,
    )
    assert isinstance(network, BaseModel)
    assert isinstance(network.network, FeedForward)
    assert network.env is env
    assert network.fdtype == torch.float64
    assert network.idtype == torch.int64
    assert network.device == torch.device("cpu")
    assert network.scale.item() == pytest.approx(1.5)
    assert all(parameter.dtype == env.fdtype for parameter in network.parameters())

    inputs = torch.tensor(
        [[[0.1, -0.2, 0.3], [0.3, 0.2, -0.1]]],
        dtype=env.fdtype,
        requires_grad=True,
    )
    output = network(inputs)
    assert output.shape == (1, 2, 2)
    assert torch.isfinite(output).all()
    assert torch.all(output.abs() <= network.scale)
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )
    with pytest.raises(RuntimeError, match="debug verification"):
        network.verify(lambda: torch.tensor(False), "debug verification")


def test_action_network_has_one_independent_cap_per_instrument() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    network = ScaledTanhNetwork(
        env,
        num_input=3,
        num_output=2,
        config=_ff_config(),
        initial_scale=1.5,
        scale_per_output=True,
    )
    assert network.scale.shape == (2,)
    assert torch.allclose(network.scale, torch.full((2,), 1.5, dtype=env.fdtype))
    with torch.no_grad():
        network.raw_scale[1].add_(1.0)
    assert network.scale[1] > network.scale[0]

    output = network(torch.ones((2, 3), dtype=env.fdtype))
    output.sum().backward()
    assert network.raw_scale.grad is not None
    assert network.raw_scale.grad.shape == (2,)
    assert torch.isfinite(network.raw_scale.grad).all()


def test_deterministic_action_is_shared_across_paths_but_not_times() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    action = DeterministicTimeAction(env, steps=2, instruments=3)
    with torch.no_grad():
        action.value.copy_(
            torch.arange(6, dtype=env.fdtype).reshape(2, 3)
        )
    output = action(torch.zeros((4, 2, 5), dtype=env.fdtype))
    assert output.shape == (4, 2, 3)
    assert torch.equal(output[0], action.value)
    assert torch.equal(output[0], output[-1])


def test_deterministic_action_preconditioner_preserves_physical_coordinates() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    action = DeterministicTimeAction(env, steps=2, instruments=2)
    rms = torch.tensor([[0.5, 2.0], [0.25, 4.0]], dtype=env.fdtype)
    action.set_preconditioner(rms, floor=1.0e-12)
    with torch.no_grad():
        action.value.fill_(2.0)

    output = action(torch.zeros((3, 2, 1), dtype=env.fdtype))
    expected = 2.0 / rms
    assert torch.allclose(action.preconditioner, rms.reciprocal())
    assert torch.allclose(output[0], expected)
    assert torch.equal(output[0], output[-1])

    base = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=env.fdtype)
    action.set_preprocessed_action(base)
    with torch.no_grad():
        action.value.fill_(0.5)
    output = action(torch.zeros((3, 2, 1), dtype=env.fdtype))
    assert torch.allclose(action.physical_base_action, base)
    assert torch.allclose(action.physical_correction_action, 0.5 / rms)
    assert torch.allclose(output[0], base + 0.5 / rms)


def test_action_network_starts_at_zero_without_blocking_output_gradients() -> None:
    env = _cpu_env(debug=True)
    config = _model_config("zero-action-initialization")
    data, geometry = prepare_universe(_universe(), config=config, env=env)
    model = DysanosModel(env, geometry=geometry, config=config)

    action = model.action(data["z"])
    assert torch.equal(action, torch.zeros_like(action))
    action.sum().backward()
    output_layer = model.action.network.In[-1]
    assert output_layer.bias is not None
    assert torch.count_nonzero(output_layer.bias.grad).item() == geometry.instruments


def test_gradient_clipping_is_disabled() -> None:
    with pytest.raises(ValueError, match="gradient clipping is disabled"):
        TrainConfig(clip_grad_norm=1.0)


def test_main_learning_rate_must_be_constant() -> None:
    with pytest.raises(ValueError, match="learning rate is constant"):
        TrainConfig(lr=1.0e-3, final_lr=1.0e-4)


def test_device_lock_allocates_and_releases_one_logical_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quantdata_torch,
        "_lock_dir",
        SubDir(str(tmp_path / "locks"), ext=".lck"),
    )
    monkeypatch.setattr(quantdata_torch.torch.cuda, "device_count", lambda: 1)
    with DeviceLock() as first:
        assert first.device == torch.device("cuda:0")
    with DeviceLock() as second:
        assert second.device == torch.device("cuda:0")


def test_same_universe_staging_and_convergence_summary() -> None:
    universe = _universe()
    stages = stage_universes([universe], stages=3, same_universe=True)
    assert len(stages) == 3
    assert all(stage is universe for stage in stages)
    independent = [_universe(), _universe(), _universe()]
    selected = stage_universes(independent, stages=3, same_universe=False)
    assert all(left is right for left, right in zip(selected, independent, strict=True))
    with pytest.raises(ValueError, match="expected 3"):
        stage_universes([universe], stages=3, same_universe=False)

    tracker = SimpleNamespace(
        valuation_epochs=[1, 100, 200, 300, 400, 500],
        training_values=[
            {"loss": value} for value in (1.0, 0.5, 0.499, 0.4989, 0.49885, 0.49884)
        ],
        moment_valuation_epochs=[],
        moment_training_values=[],
        lr=[1.0e-3, 1.0e-5],
        moment_lr=[],
    )
    summary = convergence_summary(tracker)
    assert summary["clearly_converged"]
    assert summary["main"]["points"] == 6
    assert summary["final_lr"] == pytest.approx(1.0e-5)


def test_diagnostic_continuation_override_is_explicit() -> None:
    default = _parser().parse_args(
        ["--paths", "1000", "--epochs", "100", "--report", "report.json"]
    )
    overridden = _parser().parse_args(
        [
            "--paths",
            "1000",
            "--epochs",
            "100",
            "--report",
            "report.json",
            "--continue-after-failed-gate",
        ]
    )
    assert not default.continue_after_failed_gate
    assert overridden.continue_after_failed_gate


def test_independent_training_validation_universe_pairing() -> None:
    training = _universe()
    validation = _universe()
    validation["dS"] = validation["dS"] + 1.0
    validation["dC"] = validation["dC"] + 2.0
    paired = pair_training_validation_universes([training, validation])
    assert len(paired) == 1
    assert paired[0]["dS"].shape[0] == 2 * training["dS"].shape[0]
    assert np.array_equal(
        paired[0]["dS"][: training["dS"].shape[0]], training["dS"]
    )
    assert np.array_equal(
        paired[0]["dS"][training["dS"].shape[0] :], validation["dS"]
    )
    assert np.array_equal(paired[0]["expiries"], training["expiries"])

    consumed_sources = [training, validation]
    consumed = pair_training_validation_universes(
        consumed_sources,
        consume=True,
    )
    assert consumed_sources == []
    assert np.array_equal(consumed[0]["dS"], paired[0]["dS"])

    with pytest.raises(ValueError, match="exact duplicates"):
        pair_training_validation_universes([training, training])
    with pytest.raises(ValueError, match="even positive"):
        pair_training_validation_universes([training])


def test_all_return_value_modes_and_delta_hedging() -> None:
    dS = torch.tensor([[2.0]], dtype=torch.float64)
    dC = torch.tensor([[[[5.0]]]], dtype=torch.float64)
    spot = torch.tensor([[10.0]], dtype=torch.float64)
    C = torch.tensor([[[[8.0]]]], dtype=torch.float64)
    delta = torch.tensor([[[[0.5]]]], dtype=torch.float64)
    expected = {
        (False, False): ([2.0, 5.0], [10.0, 8.0]),
        (True, False): ([2.0, 4.0], [10.0, 13.0]),
        (False, True): ([0.2, 0.625], [1.0, 1.0]),
        (True, True): ([0.2, 4.0 / 13.0], [1.0, 1.0]),
    }

    for (delta_hedged, cash_returns), (expected_dH, expected_H) in expected.items():
        dH, H = instrument_returns_and_values(
            dS=dS,
            dC=dC,
            spot=spot,
            C=C,
            delta=delta,
            delta_hedged=delta_hedged,
            cash_returns=cash_returns,
            cash_floor=1.0e-8,
        )
        assert torch.allclose(dH.flatten(), torch.tensor(expected_dH, dtype=dH.dtype))
        assert torch.allclose(H.flatten(), torch.tensor(expected_H, dtype=H.dtype))

    expiry_matched_dH, _ = instrument_returns_and_values(
        dS=dS,
        option_dS=torch.tensor([[[6.0]]], dtype=torch.float64),
        dC=dC,
        spot=spot,
        C=C,
        delta=delta,
        delta_hedged=True,
        cash_returns=False,
        cash_floor=1.0e-8,
    )
    assert torch.allclose(
        expiry_matched_dH.flatten(), torch.tensor([2.0, 2.0], dtype=torch.float64)
    )


def test_raw_state_mode_excludes_derived_contract_features() -> None:
    universe = _universe(samples=4)
    env = _cpu_env()
    full, full_geometry = prepare_universe(
        universe,
        config=replace(_model_config("full-state"), state_mode="full"),
        env=env,
    )
    raw, raw_geometry = prepare_universe(
        universe,
        config=replace(_model_config("raw-state"), state_mode="raw"),
        env=env,
    )

    assert full["z"].shape == (4, 2, 6)
    assert raw["z"].shape == (4, 2, 3)
    assert raw_geometry.state_features == raw_geometry.state_variables == 3
    assert full_geometry.state_features == 6
    assert torch.equal(raw["z"], torch.as_tensor(universe["xstate"]))


def test_prepare_universe_prefers_explicit_option_horizon_spot_change() -> None:
    universe = _universe(samples=4, steps=1)
    universe["expiry_dS"] = np.full((4, 1, 1), 6.0, dtype=np.float32)
    universe["option_dS"] = np.full((4, 1, 1), 2.0, dtype=np.float32)
    config = replace(
        _model_config("explicit-option-horizon"),
        state_mode="raw",
        delta_hedged=True,
        cash_returns=False,
    )

    data, _ = prepare_universe(universe, config=config, env=_cpu_env())

    expected_options = torch.as_tensor(universe["dC"].reshape(4, 1, -1))
    expected_options -= 0.2 * 2.0
    assert torch.allclose(data["dH"][..., 1:], expected_options)


def test_cash_normalization_floor_and_invalid_denominators() -> None:
    dS = torch.tensor([[2.0e-9]], dtype=torch.float64)
    dC = torch.tensor([[[[3.0e-9]]]], dtype=torch.float64)
    spot = torch.tensor([[1.0e-10]], dtype=torch.float64)
    C = torch.tensor([[[[2.0e-10]]]], dtype=torch.float64)
    delta = torch.zeros_like(C)
    dH, H = instrument_returns_and_values(
        dS=dS,
        dC=dC,
        spot=spot,
        C=C,
        delta=delta,
        delta_hedged=False,
        cash_returns=True,
        cash_floor=1.0e-8,
    )
    assert torch.allclose(dH, torch.tensor([[[0.2, 0.3]]], dtype=dH.dtype))
    assert torch.equal(H, torch.ones_like(H))

    # Cash normalization uses the gross option-plus-hedge traded notional.
    gross_dH, gross_H = instrument_returns_and_values(
        dS=torch.tensor([[2.0]], dtype=torch.float64),
        dC=torch.tensor([[[[3.0]]]], dtype=torch.float64),
        spot=torch.tensor([[10.0]], dtype=torch.float64),
        C=torch.tensor([[[[1.0]]]], dtype=torch.float64),
        delta=torch.tensor([[[[0.5]]]], dtype=torch.float64),
        delta_hedged=True,
        cash_returns=True,
        cash_floor=1.0e-8,
    )
    assert torch.allclose(gross_dH, torch.tensor([[[0.2, 1.0 / 3.0]]], dtype=gross_dH.dtype))
    assert torch.equal(gross_H, torch.ones_like(gross_H))

    with pytest.raises(ValueError, match="must be positive"):
        instrument_returns_and_values(
            dS=dS,
            dC=dC,
            spot=torch.zeros_like(spot),
            C=C,
            delta=delta,
            delta_hedged=False,
            cash_returns=True,
            cash_floor=1.0e-8,
        )
    with pytest.raises(ValueError, match="must be positive"):
        instrument_returns_and_values(
            dS=dS,
            dC=dC,
            spot=spot,
            C=torch.zeros_like(C),
            delta=delta,
            delta_hedged=False,
            cash_returns=False,
            cash_floor=1.0e-8,
        )


def test_conditional_variance_cost_maximum_and_fixed_training_scaling() -> None:
    drift = torch.tensor([2.0, 1.0])
    sqr = torch.tensor([3.0, 5.0])
    assert torch.equal(conditional_variance(drift, sqr), torch.tensor([0.0, 4.0]))

    variance = torch.tensor([[[16.0, 1.0, 1.0]]])
    H = torch.tensor([[[1.0, 1.0, 10.0]]])
    bar_c = torch.tensor([1.0, 2.0, 0.2])
    c_cond, c_prop, global_cost, effective = cost_components(
        variance=variance,
        H=H,
        bar_c=bar_c,
        n_cost=4,
        cost_cond=1.0,
        cost_prop=0.1,
    )
    assert torch.allclose(c_cond, torch.tensor([[[2.0, 0.5, 0.5]]]))
    assert torch.allclose(c_prop, torch.tensor([[[0.1, 0.1, 1.0]]]))
    assert torch.allclose(global_cost, torch.tensor([[[1.0, 2.0, 0.2]]]))
    assert torch.allclose(effective, torch.tensor([[[2.0, 2.0, 1.0]]]))

    # Expanding a minibatch changes neither n_cost nor any per-observation cost.
    expanded = variance.expand(5, 1, -1)
    expanded_H = H.expand_as(expanded)
    expanded_cond, _, expanded_bar, expanded_effective = cost_components(
        variance=expanded,
        H=expanded_H,
        bar_c=bar_c,
        n_cost=4,
        cost_cond=1.0,
        cost_prop=0.1,
    )
    assert torch.equal(expanded_cond[0], c_cond[0])
    assert torch.equal(expanded_bar[0], global_cost[0])
    assert torch.equal(expanded_effective[0], effective[0])


def test_entropy_diagonal_mean_variance_action_soft_threshold_and_cap() -> None:
    target = entropy_diagonal_mean_variance_action(
        drift=torch.tensor([[[2.0, -3.0, 0.5]]]),
        variance=torch.tensor([[[4.0, 2.0, 1.0]]]),
        cost=torch.tensor([[[1.0, 0.5, 1.0]]]),
        lmbda=1.0,
        cap=torch.tensor([0.1, 1.0, 0.1]),
        variance_floor=1.0e-8,
    )
    assert torch.allclose(target, torch.tensor([[[0.099, -0.99, 0.0]]]))


def test_oce_objective_forward_value_and_gradients() -> None:
    action = torch.tensor([[[0.4, -0.2]]], dtype=torch.float64, requires_grad=True)
    dH = torch.tensor([[[0.3, -0.5]]], dtype=torch.float64)
    cost = torch.tensor([[[0.01, 0.02]]], dtype=torch.float64)
    previous = torch.tensor([1.25], dtype=torch.float64)
    objective = oce_objective(
        action=action,
        dH=dH,
        cost=cost,
        previous_density=previous,
        utility="powexp",
        lmbda=1.3,
    )
    utility, _, _ = torch_utility("powexp", 1.3)
    pnl = (action * dH - action.abs() * cost).sum(dim=(1, 2))
    expected = (previous * utility(pnl, pnl.new_zeros(()))).mean()
    assert torch.allclose(objective, expected)

    objective.backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert torch.count_nonzero(action.grad).item() > 0


def test_global_y_oce_objective_value_and_cash_offset_gradient() -> None:
    action = torch.tensor(
        [[[0.4]], [[-0.2]]], dtype=torch.float64, requires_grad=True
    )
    dH = torch.tensor([[[0.3]], [[-0.5]]], dtype=torch.float64)
    cost = torch.zeros_like(dH)
    previous = torch.ones(2, dtype=torch.float64)
    cash_offset = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    objective = oce_objective(
        action=action,
        dH=dH,
        cost=cost,
        previous_density=previous,
        utility="powexp",
        lmbda=1.3,
        cash_offset=cash_offset,
    )
    utility, derivative, _ = torch_utility("powexp", 1.3)
    pnl = (action * dH).sum(dim=(1, 2))
    expected = utility(pnl, cash_offset).mean()

    assert torch.allclose(objective, expected)
    objective.backward()
    expected_y_gradient = (
        derivative(pnl.detach(), cash_offset.detach()) - 1.0
    ).mean()
    assert cash_offset.grad is not None
    assert cash_offset.grad.item() == pytest.approx(
        expected_y_gradient.item(), abs=1.0e-12
    )


def test_global_y_rejects_mechanical_cvar() -> None:
    with pytest.raises(ValueError, match="does not support a global cash offset"):
        replace(
            _model_config("invalid-global-y-cvar"),
            utility="cvar",
            oce_variant="global_y",
        )


def test_importance_weights_are_normalized_per_split_and_weight_objective() -> None:
    universe = _universe(samples=4, steps=1)
    universe["log_path_weight"] = np.log(
        np.array([1.0, 3.0, 2.0, 6.0], dtype=np.float32)
    )
    config = replace(
        _model_config("importance-weighted-objective"),
        action_type="deterministic",
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    data, geometry = prepare_universe(universe, config=config, env=_cpu_env())
    training, validation = split_training_validation(data, 0.5)

    assert validation is not None
    assert training["sampling_weight"].mean().item() == pytest.approx(1.0)
    assert validation["sampling_weight"].mean().item() == pytest.approx(1.0)
    assert training["sampling_weight"].tolist() == pytest.approx([0.5, 1.5])
    assert validation["sampling_weight"].tolist() == pytest.approx([0.5, 1.5])

    model = DysanosModel(_cpu_env(), geometry=geometry, config=config)
    model.calibrate_unconditional(training)
    with torch.no_grad():
        model.action.value.fill_(0.2)
    output = model._trading_outputs(training)
    utility, _, _ = torch_utility("powexp", config.lmbda)
    expected = (
        training["sampling_weight"]
        * utility(output["path_pnl"], model.cash_offset)
    ).mean()
    assert -model(training).item() == pytest.approx(expected.item())


def test_no_offset_marginal_utility_density_is_a_martingale() -> None:
    dH = torch.tensor([[[-1.0]], [[2.0]]], dtype=torch.float64)
    action = torch.full(
        dH.shape,
        math.log(2.0) / 3.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    objective = oce_objective(
        action=action,
        dH=dH,
        cost=torch.zeros_like(dH),
        previous_density=torch.ones(2, dtype=torch.float64),
        utility="entropy",
        lmbda=1.0,
    )
    objective.backward()
    assert action.grad is not None
    assert action.grad.sum().item() == pytest.approx(0.0, abs=1.0e-12)

    _, derivative, _ = torch_utility("entropy", 1.0)
    path_pnl = (action.detach() * dH).sum(dim=(1, 2))
    raw_density = derivative(path_pnl, path_pnl.new_zeros(()))
    density = raw_density / raw_density.mean()
    assert density.mean().item() == pytest.approx(1.0)
    assert (density * dH[:, 0, 0]).mean().item() == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_cvar_without_offset_has_trivial_zero_action_optimum() -> None:
    lmbda = 1.0 / 9.0
    utility, _, _ = torch_utility("cvar", lmbda)
    pnl = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    zero = pnl.new_zeros(())
    assert utility(pnl, zero).mean().item() < 0.0
    action = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    objective = utility(action * pnl, zero).mean()
    objective.backward()
    assert objective.item() == pytest.approx(0.0)
    assert action.grad is not None and action.grad.item() == pytest.approx(0.0)


def test_mechanical_lower_tail_selects_exact_weight_with_ties() -> None:
    values = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    weights = torch.ones_like(values)
    selected, cutoff, boundary = lower_tail_weights(values, weights, 0.625)
    assert cutoff.item() == pytest.approx(1.0)
    assert boundary.item() == pytest.approx(0.75)
    assert torch.equal(
        selected,
        torch.tensor([1.0, 0.75, 0.75, 0.0], dtype=torch.float64),
    )
    assert (selected * weights).sum().item() == pytest.approx(2.5)


def test_cvar_density_calibration_uses_worst_ninety_percent() -> None:
    env = _cpu_env(debug=False)
    config = replace(
        _model_config("cvar-mechanical-tail"),
        utility="cvar",
        lmbda=1.0 / 9.0,
        action_type="deterministic",
    )
    data, geometry = prepare_universe(_universe(samples=20), config=config, env=env)
    model = DysanosModel(env, geometry=geometry, config=config)
    with torch.no_grad():
        model.action.value[0, 0] = 1.0
    model.calibrate_unconditional(data, batch_size=5)
    model.calibrate_density(data, batch_size=5)
    output = model.pathwise(data)
    assert output["density_increment"].sum().item() == pytest.approx(18.0)
    assert output["density"].mean().item() == pytest.approx(1.0)
    assert model.tail_cutoff.item() == pytest.approx(
        torch.sort(output["path_pnl"])[0][17].item()
    )


def test_deterministic_powerexp_lbfgs_optimizer_runs(tmp_path: Path) -> None:
    config = replace(
        _model_config("deterministic-lbfgs"),
        utility="powexp",
        action_type="deterministic",
        deterministic_precondition=True,
    )
    train_config = replace(
        _train_config(tmp_path),
        optimizer="LBFGS",
        lr=0.5,
        final_lr=0.5,
        weight_decay=0.0,
    )
    result = train(
        _universe(samples=20),
        config=config,
        train_config=train_config,
        epochs=2,
        restart=True,
        env=_cpu_env(debug=False),
    )
    assert isinstance(result.model.action, DeterministicTimeAction)
    assert result.tracker.lr == pytest.approx([0.5, 0.5])
    assert result.training["density_nonfinite_count"] == 0
    assert result.training["objective_gradient"]["action_gradient_nonfinite"] == 0
    assert result.training["objective_gradient"][
        "physical_action_gradient_nonfinite"
    ] == 0
    assert np.isfinite(result.training["action_preconditioner"]).all()
    assert np.all(result.training["action_preconditioner"] > 0.0)


def test_scipy_deterministic_preprocessing_and_joint_gradient_converge(
    tmp_path: Path,
) -> None:
    config = replace(
        _model_config("deterministic-scipy-preprocess"),
        utility="entropy",
        action_type="deterministic",
        deterministic_preprocess=True,
        delta_hedged=False,
        cost_cond=0.0,
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    train_config = replace(
        _train_config(tmp_path, validation_fraction=0.0),
        optimizer="SciPy",
        lr=1.0,
        final_lr=1.0,
        weight_decay=0.0,
        compile=False,
        scipy_gradient_tolerance=1.0e-11,
        scipy_function_tolerance=1.0e-15,
        scipy_max_line_search_steps=50,
    )
    result = train(
        _independent_binary_universe(),
        config=config,
        train_config=train_config,
        epochs=100,
        restart=True,
        env=BaseEnvironment(precision=64, device="cpu", debug=False),
    )

    expected = math.log(2.0) / 3.0
    base = result.training["deterministic_preprocessed_action"]
    correction = result.training["deterministic_correction_action"]
    total = result.training["deterministic_total_action"]
    assert np.allclose(base, expected, atol=1.0e-8)
    assert np.allclose(correction, 0.0, atol=1.0e-8)
    assert np.allclose(total, expected, atol=1.0e-8)
    assert result.tracker.deterministic_preprocessing is not None
    assert result.tracker.deterministic_preprocessing["gradient_linf"] < 1.0e-9
    assert result.tracker.deterministic_optimizer is not None
    assert result.tracker.deterministic_optimizer["gradient_linf"] < 1.0e-9
    gradient = result.training["objective_gradient"]
    assert gradient["physical_action_gradient_linf"] < 1.0e-9
    assert np.max(np.abs(result.training["mean_dH_by_time"])) < 1.0e-9


def test_entropy_log_moment_is_finite_for_extreme_deterministic_action() -> None:
    env = _cpu_env(debug=True)
    config = replace(
        _model_config("stable-entropy-log-moment"),
        utility="entropy",
        action_type="deterministic",
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    data, geometry = prepare_universe(_universe(samples=20), config=config, env=env)
    model = DysanosModel(env, geometry=geometry, config=config)
    model.calibrate_unconditional(data, batch_size=3)
    with torch.no_grad():
        model.action.value.fill_(1.0e6)

    loss = entropy_log_moment_loss(model, data, batch_size=3)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.action.value.grad is not None
    assert torch.isfinite(model.action.value.grad).all()


def test_direction_mask_removes_only_selected_long_short_gradients() -> None:
    action = torch.tensor(
        [[[2.0, -3.0], [2.0, -3.0]]], requires_grad=True
    )
    excluded = torch.tensor(
        [[
            [True, False, False, True],
            [False, False, False, False],
        ]]
    )
    masked = apply_direction_mask(action, excluded)
    assert torch.equal(masked, torch.tensor([[[0.0, 0.0], [2.0, -3.0]]]))
    masked.sum().backward()
    assert action.grad is not None
    assert torch.equal(action.grad, torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))


def test_saved_arb_mask_maps_option_directions_and_leaves_spot_active(
    tmp_path: Path,
) -> None:
    universe = _universe(samples=4, steps=2)
    split_masks: dict[str, Path] = {}
    for split in ("training", "validation"):
        options = np.zeros((2, 2, 2, 2), dtype=bool)
        if split == "training":
            options[0, 0, 0, 0] = True
        else:
            options[0, 0, 1, 1] = True
        packed = np.packbits(
            options.reshape(4, 2, 2), axis=-1, bitorder="little"
        )
        split_masks[split] = tmp_path / f"{split}.npy"
        np.save(split_masks[split], packed)

    report = {
        "status": "complete",
        "paths": 4,
        "training_paths": 2,
        "validation_paths": 2,
        "universe_seed_offset": 7,
        "geometry": {"steps": 2},
        "n_cost": 4,
        "mask_files": {name: str(path) for name, path in split_masks.items()},
        "training": {
            "candidate_count_by_direction": [1, 0],
            "candidate_definition": "test",
        },
        "validation": {
            "candidate_count_by_direction": [0, 1],
            "candidate_definition": "test",
        },
    }
    report_path = tmp_path / "mask.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    masked, metadata = attach_arb_direction_mask(
        universe,
        report_path=report_path,
        validation_fraction=0.5,
        universe_seed_offset=7,
    )
    excluded = masked["excluded_directions"]
    assert excluded.shape == (4, 2, 6)
    assert not excluded[..., 0].any() and not excluded[..., 3].any()
    assert excluded[0, 0, 1]
    assert excluded[2, 0, 5]
    assert np.count_nonzero(excluded) == 2
    assert metadata["candidate_count"] == 2
    assert not metadata["empty_mask_optimized"]


def test_two_phase_training_freezes_moments_and_tracks_both_splits(tmp_path: Path) -> None:
    result = train(
        _universe(),
        config=_model_config("two-phase", cost_cond=1.0),
        train_config=_train_config(tmp_path),
        epochs=2,
        moment_epochs=2,
        env=_cpu_env(),
    )
    assert result.model.drift is not None and result.model.sqr is not None
    moment_parameters = [
        *result.model.drift.parameters(),
        *result.model.sqr.parameters(),
    ]
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in moment_parameters)
    assert result.tracker.moment_epochs == [1, 2]
    assert result.tracker.moment_valuation_epochs[-1] == 2
    assert result.tracker.valuation_epochs[-1] == 2
    assert result.tracker.moment_training_values
    assert result.tracker.moment_validation_values
    for output in (result.training, result.validation):
        assert output is not None
        for name in ("DRIFT", "SQR", "v", "drift_loss", "sqr_loss"):
            assert output[name].shape == (3,)
            assert np.isfinite(output[name]).all()
        assert np.all(output["v"] >= 0.0)
        reported = _instrument_outputs(output)
        for name in (
            "excess_mean",
            "excess_std",
            "excess_quantiles",
            "outside_cost_fraction",
            "mean_absolute_violation",
            "quantile_levels",
        ):
            assert name in reported


def test_entropy_action_initialization_runs_after_moment_phase(tmp_path: Path) -> None:
    config = replace(_model_config("entropy-action-init", cost_cond=1.0), utility="entropy")
    train_config = replace(
        _train_config(tmp_path),
        action_initialization_epochs=2,
        action_initialization_lr=1.0e-3,
    )
    result = train(
        _universe(),
        config=config,
        train_config=train_config,
        epochs=1,
        moment_epochs=1,
        env=_cpu_env(),
    )
    tracker = result.tracker
    assert tracker.action_initialization_completed
    assert tracker.action_initialization_first_loss is not None
    assert tracker.action_initialization_final_loss is not None
    assert math.isfinite(tracker.action_initialization_first_loss)
    assert math.isfinite(tracker.action_initialization_final_loss)


def test_density_normalization_nonnegativity_and_path_composition(tmp_path: Path) -> None:
    universe = _universe()
    train_config = _train_config(tmp_path)
    env = _cpu_env()
    first = train(
        universe,
        config=_model_config("density-stage-1"),
        train_config=train_config,
        epochs=1,
        env=env,
    )
    second = train(
        universe,
        config=_model_config("density-stage-2"),
        train_config=train_config,
        epochs=1,
        previous_density=first.density,
        env=env,
    )
    assert first.training["density_mean"] == pytest.approx(1.0, abs=2.0e-6)
    assert first.training["path_density_mean"] == pytest.approx(1.0, abs=2.0e-6)
    assert first.training["density_mean_minus_one_over_err"] == pytest.approx(
        0.0, abs=1.0e-4
    )
    assert second.training["density_mean"] == pytest.approx(1.0, abs=2.0e-6)
    assert second.training["path_density_mean"] == pytest.approx(1.0, abs=2.0e-6)

    all_data, _ = prepare_universe(universe, config=second.model.config, env=env)
    training, _ = split_training_validation(all_data, train_config.validation_fraction)
    z, dH, H = training["z"], training["dH"], training["H"]
    increment = second.density.path_increment(z, dH, H)
    expected_path = (
        first.density(z, dH, H)
        * increment
        / second.density.path_normalizer
    )
    expected_log_path = (
        first.density.log_path_density(z, dH, H)
        + torch.log(increment.clamp_min(second.model.config.density_floor))
        - torch.log(second.density.path_normalizer)
    )
    observation = second.density(z, dH, H)
    log_path = second.density.log_path_density(z, dH, H)
    path = second.density.path_density(z, dH, H)
    assert observation.shape == (z.shape[0],)
    assert torch.allclose(observation, expected_path)
    assert torch.allclose(log_path, expected_log_path)
    assert torch.allclose(path, torch.exp(log_path))
    assert torch.isfinite(observation).all() and torch.all(observation >= 0.0)
    assert torch.isfinite(path).all() and torch.all(path >= 0.0)
    assert observation.mean().item() == pytest.approx(1.0, abs=2.0e-6)
    assert path.mean().item() == pytest.approx(1.0, abs=2.0e-6)


def test_validation_reestimates_density_normalizer(tmp_path: Path) -> None:
    env = _cpu_env(debug=False)
    config = replace(
        _model_config("validation-density-normalizer"),
        utility="powexp",
        action_type="deterministic",
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    data, geometry = prepare_universe(
        _universe(samples=8, validation_shift=2.0),
        config=config,
        env=env,
    )
    training, validation = split_training_validation(data, 0.25)
    assert validation is not None
    model = DysanosModel(env, geometry=geometry, config=config)
    with torch.no_grad():
        model.action.value.zero_()
        model.action.value[:, 0] = 0.5
    model.calibrate_unconditional(training, batch_size=2)
    model.calibrate_density(training, batch_size=2)

    raw_density = model.pathwise(validation)["density"]
    expected_density = raw_density / raw_density.mean()
    output = valuate(model, validation, batch_size=1, split="validation")
    expected_mean_dH = (
        expected_density[:, None, None] * validation["dH"]
    ).mean(dim=1).mean(dim=0)

    assert output["density_pre_normalization_mean"] == pytest.approx(
        raw_density.mean().item()
    )
    assert output["density_split_rescaling_factor"] == pytest.approx(
        raw_density.mean().reciprocal().item()
    )
    assert output["density_pre_normalization_mean"] != pytest.approx(1.0)
    assert output["density_mean"] == pytest.approx(1.0, abs=1.0e-6)
    assert output["density_normalization_error"] == pytest.approx(0.0, abs=1.0e-6)
    assert np.allclose(
        output["mean_dH"], expected_mean_dH.detach().cpu().numpy()
    )


def test_excess_multiples_baseline_density_and_cap_derivative() -> None:
    env = _cpu_env(debug=False)
    config = _model_config("excess-multiples")
    data, geometry = prepare_universe(_universe(samples=4), config=config, env=env)
    model = DysanosModel(env, geometry=geometry, config=config)
    with torch.no_grad():
        for parameter in model.action.network.parameters():
            parameter.zero_()
    model.calibrate_unconditional(data, batch_size=2)
    model.calibrate_density(data, batch_size=2)

    output = valuate_with_cap_derivative(
        model,
        data,
        batch_size=2,
        split="training",
        include_objective_gradient=True,
    )
    multiples = np.arange(1.0, 5.0).reshape(4, 1, 1, 1)
    returns = data["dH"].cpu().numpy()
    cost = np.broadcast_to(output["c"], returns.shape)
    violation = np.maximum(np.abs(returns)[None] - multiples * cost[None], 0.0)
    expected_unsigned = violation.mean(axis=(1, 2))
    expected_signed = (np.sign(returns)[None] * violation).mean(axis=(1, 2))

    assert np.allclose(output["excess_cost_multiples"], np.arange(1.0, 5.0))
    assert np.allclose(output["unsigned_excess_mean_by_multiple"], expected_unsigned)
    assert np.allclose(output["signed_excess_mean_by_multiple"], expected_signed)
    assert np.allclose(
        output["reference_unsigned_excess_mean_by_multiple"], expected_unsigned
    )
    assert output["density_mean"] == pytest.approx(1.0)
    assert output["density_mean_minus_one_over_err"] == pytest.approx(0.0)
    derivative = cap_objective_derivative(model, data, batch_size=2)
    assert derivative.shape == (geometry.instruments,)
    assert np.isfinite(derivative).all()
    assert np.allclose(output["action_cap_objective_derivative"], derivative)
    gradient = output["objective_gradient"]
    assert gradient["action_gradient_nonfinite"] == 0
    assert math.isfinite(gradient["combined_gradient_l2"])
    assert output["path_pnl_mean_under_P"] == pytest.approx(0.0)
    assert output["path_pnl_std_under_P"] == pytest.approx(0.0)
    assert np.allclose(output["path_pnl_weighted_quantiles_under_P"], 0.0)
    assert np.allclose(
        output["path_pnl_probability_below_threshold_under_P"], 0.0
    )
    returns_by_time = data["dH"].cpu().numpy()
    assert np.allclose(output["mean_dH_by_time"], returns_by_time.mean(axis=0))
    assert np.allclose(output["std_dH_by_time"], returns_by_time.std(axis=0))
    assert np.allclose(
        output["err_dH_by_time"],
        returns_by_time.std(axis=0) / math.sqrt(returns_by_time.shape[0]),
    )
    path_means = returns_by_time.mean(axis=1)
    assert np.allclose(output["mean_dH"], path_means.mean(axis=0))
    assert np.allclose(output["std_dH"], path_means.std(axis=0))

    baseline = _baseline_split(output)
    assert baseline["martingale"]["density_mean"] == pytest.approx(1.0)
    assert len(
        baseline["instruments"]["unsigned_excess_mean_by_multiple"]
    ) == 4


def test_cpu_valuation_catches_infinite_objective() -> None:
    env = _cpu_env(debug=False)
    config = _model_config("cpu-infinite-objective")
    data, geometry = prepare_universe(_universe(samples=4), config=config, env=env)
    model = DysanosModel(env, geometry=geometry, config=config)
    model.calibrate_unconditional(data, batch_size=2)
    model.calibrate_density(data, batch_size=2)
    model.utility = lambda pnl, offset: torch.full_like(pnl + offset, math.inf)

    with pytest.raises(FloatingPointError, match="CPU during valuation"):
        valuate(model, data, batch_size=2, split="training")


def test_early_stopping_requires_flat_loss_and_cost_band(tmp_path: Path) -> None:
    config = replace(
        _model_config("early-stop"),
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    train_config = TrainConfig(
        trn_batch_size=3,
        val_batch_size=2,
        validation_fraction=0.0,
        value_update=1,
        value_seconds=1_000.0,
        checkpoint_update=100,
        checkpoint_seconds=1_000.0,
        checkpoint_root=str(tmp_path),
        early_stopping=True,
        minimum_epochs=2,
        convergence_points=2,
        require_cost_band=True,
    )
    result = train(
        _universe(zero_returns=True),
        config=config,
        train_config=train_config,
        epochs=10,
        env=_cpu_env(debug=False),
    )
    assert result.tracker.main_converged_epoch is not None
    assert result.tracker.main_converged_epoch < 10
    assert np.all(np.abs(result.training["return_on_cost"]) <= 1.0)


def test_nested_realized_density_compiles_as_one_full_graph(tmp_path: Path) -> None:
    """A later stage must compile through all earlier density increments."""

    universe = _universe()
    env = _cpu_env(debug=False)
    train_config = _train_config(tmp_path)
    first = train(
        universe,
        config=_model_config("compiled-density-stage-1"),
        train_config=train_config,
        epochs=1,
        env=env,
    )
    second = train(
        universe,
        config=_model_config("compiled-density-stage-2"),
        train_config=train_config,
        epochs=1,
        previous_density=first.density,
        env=env,
    )
    all_data, _ = prepare_universe(universe, config=second.model.config, env=env)
    z, dH, H = all_data["z"], all_data["dH"], all_data["H"]
    eager = second.density(z, dH, H)
    compiled = torch.compile(second.density, backend="eager", fullgraph=True)
    actual = compiled(z, dH, H)
    assert torch.allclose(actual, eager)

    density_file = tmp_path / "nested-density.pt"
    torch.save(second.density, density_file)
    restored = torch.load(density_file, map_location="cpu", weights_only=False)
    assert torch.allclose(restored(z, dH, H), eager)


def test_training_validation_isolation_for_unconditional_calibration() -> None:
    config = _model_config("split-isolation")
    env = _cpu_env()
    ordinary, geometry = prepare_universe(_universe(), config=config, env=env)
    shifted, shifted_geometry = prepare_universe(
        _universe(validation_shift=1_000.0), config=config, env=env
    )
    training, validation = split_training_validation(ordinary, 0.25)
    shifted_training, shifted_validation = split_training_validation(shifted, 0.25)
    assert geometry == shifted_geometry
    assert validation is not None and shifted_validation is not None
    assert torch.equal(training["dH"], shifted_training["dH"])
    assert not torch.equal(validation["dH"], shifted_validation["dH"])

    first = DysanosModel(env, geometry=geometry, config=config)
    second = DysanosModel(env, geometry=geometry, config=config)
    first.calibrate_unconditional(training, batch_size=2)
    second.calibrate_unconditional(shifted_training, batch_size=3)
    assert torch.equal(first.bar_variance, second.bar_variance)
    assert torch.equal(first.bar_c, second.bar_c)
    assert first.n_cost.item() == second.n_cost.item() == 6


def test_checkpoint_round_trip_rng_restore_and_incompatible_restart(tmp_path: Path) -> None:
    universe = _universe()
    train_config = _train_config(tmp_path)
    original_config = _model_config("*stable-checkpoint", user_version="a")
    original = train(
        universe,
        config=original_config,
        train_config=train_config,
        epochs=1,
        env=_cpu_env(),
    )
    checkpoint = torch.load(original.checkpoint_path, map_location="cpu", weights_only=False)
    for name in (
        "model_state",
        "main_optimizer",
        "main_scheduler",
        "rng_state",
        "tracker",
        "model_version",
        "user_version",
    ):
        assert name in checkpoint

    torch.rand(5)
    np.random.random(5)
    loaded = train(
        universe,
        config=original_config,
        train_config=train_config,
        epochs=1,
        just_load=True,
        env=_cpu_env(debug=False),
    )
    assert loaded.restarted
    for name, value in original.model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.model.state_dict()[name].cpu())
    assert torch.equal(torch.random.get_rng_state(), checkpoint["rng_state"]["torch_cpu"])
    restored_numpy = np.random.get_state()
    saved_numpy = checkpoint["rng_state"]["numpy"]
    assert restored_numpy[0] == saved_numpy[0]
    assert np.array_equal(restored_numpy[1], saved_numpy[1])
    assert restored_numpy[2:] == saved_numpy[2:]

    incompatible_config = _model_config("*stable-checkpoint", user_version="b")
    with pytest.warns(UserWarning, match="cold-starting"):
        cold = train(
            universe,
            config=incompatible_config,
            train_config=train_config,
            epochs=1,
            env=_cpu_env(),
        )
    assert not cold.restarted


def test_cuda_rng_restore_moves_states_to_cpu_and_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[torch.Tensor] = []
    state = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": [
            torch.tensor([1], dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
        ],
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.extend)

    _restore_rng_state(state)

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_zero_variance_single_observation_and_nonfinite_inputs(tmp_path: Path) -> None:
    universe = _universe(samples=1, steps=1, zero_returns=True)
    result = train(
        universe,
        config=_model_config("single-observation"),
        train_config=_train_config(tmp_path, validation_fraction=0.0),
        epochs=1,
        env=_cpu_env(),
    )
    assert result.validation is None
    assert result.training["n_cost"] == 1
    assert np.array_equal(result.training["bar_variance"], np.zeros(3))
    assert np.array_equal(result.training["std_dH"], np.zeros(3))
    assert math.isfinite(result.training["loss"])
    assert np.isfinite(result.training["sample_ratio_quantiles"]).all()

    nonfinite = _universe()
    nonfinite["dS"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prepare_universe(
            nonfinite,
            config=_model_config("nonfinite-input"),
            env=_cpu_env(),
        )
    with pytest.raises(ValueError, match="non-finite"):
        conditional_variance(torch.tensor([math.nan]), torch.tensor([1.0]))


def test_configuration_modes_have_distinct_checkpoint_names() -> None:
    modes: list[tuple[bool, bool]] = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    filenames = {
        DysanosConfig(
            label="comparable-run",
            delta_hedged=delta_hedged,
            cash_returns=cash_returns,
        ).filename
        for delta_hedged, cash_returns in modes
    }
    assert len(filenames) == len(modes)


def test_martingale_summary_uses_two_sigma_and_familywise_thresholds() -> None:
    z_score = np.array([0.0, 1.5, -2.1, 2.4])
    output = {
        "mean_dH_over_err": z_score,
        "return_on_cost": np.array([0.0, 0.5, -0.8, 1.2]),
        "density_mean": 1.0,
        "density_std": 0.2,
        "density_err": 0.01,
        "density_mean_minus_one_over_err": 0.0,
        "density_ess": 320.0,
        "n_error_observations": 400,
        "density_normalization_error": 0.0,
        "path_density_mean": 1.0,
        "path_density_ess": 75.0,
        "path_density_normalization_error": 0.0,
    }
    summary = martingale_summary(output)
    assert summary["count_abs_z_gt_2"] == 2
    assert summary["fraction_abs_z_le_2"] == 0.5
    assert summary["max_abs_z"] == 2.4
    assert summary["bonferroni_95_threshold"] > 2.0
    assert summary["within_population_mc_noise_95_familywise"]
    assert summary["fraction_abs_return_on_cost_le_1"] == 0.75
    assert summary["density_ess_fraction"] == 0.8
