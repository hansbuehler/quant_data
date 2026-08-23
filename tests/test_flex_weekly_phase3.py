from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import (
    ForwardMuFlexGeometryConfig,
    JointHierarchicalDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.joint_spot_correlation import (
    evaluate_joint_spot_correlation_under_p,
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.dysanos.flex.spread_cost import (
    LINEAR_FEATURE_NAMES,
    FittedSpreadCostModel,
    SpreadFeedForwardConfig,
    TorchSpreadCostModel,
)
from quantdata.dysanos.flex.weekly_dynamics import (
    PHASE3_TRAIN_END,
    PHASE3_VALIDATION_START,
    STRONG_STATE_SIZES,
    TorchWeeklyDynamics,
    WeeklyDynamicsConfig,
    WeeklyTrainingHistory,
    fit_phase3_training_history,
    fit_weekly_hierarchical_dynamics,
    load_weekly_dynamics_fit,
    save_weekly_dynamics_fit,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    NormalWeeklyMeasureKernel,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainConfig,
    audit_front_week_consistency,
    audit_weekly_roll_line,
    fit_weekly_measure_kernel,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklyOptionGridConfig,
    WeeklySanosSimulator,
    front_week_sanos_inverse_cdf,
    hjm_roll_moment_loss,
    load_strong_calibrated_model,
    make_tradable_option_grid,
)
from quantdata.dysanos.flex.weekly_validation import (
    evaluate_oos_dynamics,
    exact_weekly_pairs,
    project_strong_states,
    propagated_joint_moments,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _history(
    sizes: tuple[int, ...] = (3, 3, 2),
    *,
    observations: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9182)
    blocks = [np.zeros((observations, width)) for width in sizes]
    log_spot = np.zeros(observations)
    for time_index in range(1, observations):
        common = rng.normal()
        for expiry_index, block in enumerate(blocks):
            innovation = 0.025 * rng.normal(size=block.shape[1])
            innovation[0] += 0.04 * common
            own = 0.72 + 0.04 * expiry_index
            value = own * block[time_index - 1] + innovation
            if expiry_index > 0:
                lag_driver = np.resize(
                    blocks[expiry_index - 1][time_index - 1], block.shape[1]
                )
                value += 0.11 * lag_driver
            block[time_index] = value
        log_spot[time_index] = (
            log_spot[time_index - 1] + 0.001 + 0.012 * common + 0.004 * rng.normal()
        )
    return np.concatenate(blocks, axis=1), log_spot


def _fit_and_dynamics(
    env: BaseEnvironment,
    *,
    sizes: tuple[int, ...] = (2, 2, 1),
) -> TorchWeeklyDynamics:
    states, log_spot = _history(sizes, observations=150)
    config = WeeklyDynamicsConfig(
        label="phase3-test",
        state_sizes_by_expiry=sizes,
        weekly_session_stride=5,
        pca_mode="full",
        ridge_penalty=1.0e-5,
    )
    fit = fit_weekly_hierarchical_dynamics(states, log_spot, config=config)
    return TorchWeeklyDynamics(env, fit=fit)


def _simulator(env: BaseEnvironment) -> WeeklySanosSimulator:
    sizes = (2, 2, 1)
    dynamics = _fit_and_dynamics(env, sizes=sizes)
    geometry = ForwardMuFlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=3,
    )
    calibrated = LatentStateSanosModel(
        env,
        num_days=2,
        config=TriangularPcaDecoderConfig(
            label="phase3-small-decoder",
            state_sizes_by_expiry=sizes,
            state_size=sum(sizes),
            geometry=geometry,
        ),
    )
    return WeeklySanosSimulator(
        env,
        dynamics=dynamics,
        calibrated_model=calibrated,
    )


def test_weekly_fit_exposes_full_and_reduced_pca_and_is_stationary() -> None:
    from scipy.linalg import expm

    states, log_spot = _history()
    dates = tuple(get_business_days("2022-01-03", "2022-12-30")[: len(states)])
    reduced_config = WeeklyDynamicsConfig(
        label="phase3-reduced-test",
        state_sizes_by_expiry=(3, 3, 2),
        pca_variance_threshold=0.995,
        ridge_penalty=1.0e-5,
    )
    reduced = fit_weekly_hierarchical_dynamics(
        states, log_spot, config=reduced_config, dates=dates
    )
    expected_transitions = len(dates) - 5
    expected_daily_transitions = len(dates) - 1
    assert reduced.weekly_transition_observations == expected_transitions
    assert reduced.diagnostics["origin_weighting"] == (
        "equal weight per exact daily origin for drift and per exact weekly "
        "origin for innovations"
    )
    assert np.all(np.isfinite(reduced.equilibrium_mean))
    assert reduced.diagnostics["training_score_mean_max_abs"] < 1.0e-12
    assert reduced.diagnostics["equilibrium_distance_over_training_rms"] < 10.0
    assert reduced.diagnostics["pair_endpoint_drift_l2"] > 0.0
    assert reduced.diagnostics["reduced_residual_mean_l2"] > 0.0
    for pca in reduced.pca_by_expiry:
        assert pca.full_components.shape[1] == len(pca.mean)
        assert pca.explained_variance[pca.reduced_rank - 1] >= 0.995
    assert np.max(np.abs(reduced.mean_reversion_generator - np.diag(
        np.diag(reduced.mean_reversion_generator)
    ))) > 0.0
    assert np.min(np.linalg.eigvals(reduced.mean_reversion_generator).real) > 0.0
    assert np.max(np.abs(np.linalg.eigvals(reduced.reduced_transition))) < 1.0
    assert reduced.diagnostics["generator_roundtrip_relative_error"] <= (
        reduced.config.generator_roundtrip_relative_tolerance
    )
    assert reduced.diagnostics["transition_condition_number"] <= (
        reduced.config.maximum_transition_condition_number
    )
    assert reduced.diagnostics["estimator"] == (
        "DYSANOS/grid lag-only daily affine matrix logarithm"
    )
    assert reduced.diagnostics[
        "standardized_transition_maximum_power_norm"
    ] <= reduced.config.maximum_standardized_transition_power_norm
    np.testing.assert_allclose(
        expm(-reduced.mean_reversion_generator),
        reduced.reduced_transition,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert all(
        block.observations == expected_daily_transitions
        for block in reduced.blocks
    )
    assert all(block.design_rank <= block.design_columns for block in reduced.blocks)

    full = fit_weekly_hierarchical_dynamics(
        states,
        log_spot,
        config=replace(reduced_config, label="phase3-full-test", pca_mode="full"),
        dates=dates,
    )
    assert full.reduced_sizes_by_expiry == (3, 3, 2)
    assert full.reduced_state_size == sum((3, 3, 2))


def test_weekly_fit_accepts_direct_global_single_block() -> None:
    states, log_spot = _history((5,), observations=180)
    fit = fit_weekly_hierarchical_dynamics(
        states,
        log_spot,
        config=WeeklyDynamicsConfig(
            label="phase3-direct-global-test",
            state_sizes_by_expiry=(5,),
            pca_mode="full",
            ridge_penalty=1.0e-5,
        ),
    )
    assert fit.reduced_sizes_by_expiry == (5,)
    assert len(fit.blocks) == 1
    assert fit.blocks[0].previous_lag is None
    assert np.max(np.abs(np.linalg.eigvals(fit.reduced_transition))) < 1.0


def test_locked_training_wrapper_rejects_validation_leakage() -> None:
    states, log_spot = _history(observations=30)
    training_dates = tuple(get_business_days("2024-10-01", str(PHASE3_TRAIN_END)))
    training_dates = training_dates[: len(states)]
    history = WeeklyTrainingHistory(
        dates=training_dates,
        states=states[: len(training_dates)],
        log_spot=log_spot[: len(training_dates)],
    )
    config = WeeklyDynamicsConfig(
        label="phase3-locked-split",
        state_sizes_by_expiry=(3, 3, 2),
        pca_mode="full",
    )
    fit = fit_phase3_training_history(history, config=config)
    assert fit.training_rows == len(training_dates)
    invalid_dates = history.dates[:-1] + (PHASE3_VALIDATION_START,)
    invalid = WeeklyTrainingHistory(
        dates=invalid_dates,
        states=history.states,
        log_spot=history.log_spot,
    )
    with pytest.raises(ValueError, match="only training dates"):
        fit_phase3_training_history(invalid, config=config)


def test_torch_weekly_transition_has_exact_long_horizon_and_gradients() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    dynamics = _fit_and_dynamics(env)
    weekly = dynamics.transition_matrix(step_weeks=1)
    annual = dynamics.transition_matrix(step_weeks=52)
    torch.testing.assert_close(annual, torch.linalg.matrix_power(weekly, 52))
    torch.testing.assert_close(
        dynamics.transition_matrix(step_weeks=2),
        torch.linalg.matrix_power(weekly, 2),
    )
    torch.testing.assert_close(
        dynamics.transition_matrix(step_weeks=4),
        torch.linalg.matrix_power(weekly, 4),
    )
    torch.testing.assert_close(
        dynamics.two_week_joint_factor @ dynamics.two_week_joint_factor.T,
        dynamics.two_week_joint_covariance,
    )
    torch.testing.assert_close(
        dynamics.four_week_joint_factor @ dynamics.four_week_joint_factor.T,
        dynamics.four_week_joint_covariance,
    )

    initial = torch.zeros(
        (4, dynamics.reduced_state_size),
        dtype=env.fdtype,
        requires_grad=True,
    )
    log_spot = torch.zeros(4, dtype=env.fdtype, requires_grad=True)
    normal = torch.zeros(
        (4, dynamics.reduced_state_size + 1), dtype=env.fdtype
    )
    step = dynamics.step(initial, log_spot, standard_normals=normal)
    loss = step.reduced_state.square().sum() + step.normal_log_spot.sum()
    loss.backward()
    assert initial.grad is not None and torch.all(torch.isfinite(initial.grad))
    assert log_spot.grad is not None and torch.all(torch.isfinite(log_spot.grad))
    assert dynamics.weekly_joint_factor.shape == (
        dynamics.reduced_state_size + 1,
        dynamics.reduced_state_size + 1,
    )
    torch.testing.assert_close(
        dynamics.stationary_surface_covariance,
        weekly.T
        @ dynamics.stationary_surface_covariance
        @ weekly
        + dynamics.weekly_joint_covariance[1:, 1:],
    )
    invariant = dynamics.sample_stationary_reduced_states(5)
    assert invariant.shape == (5, dynamics.reduced_state_size)


def test_normal_and_implied_spot_paths_reuse_the_same_surface_shocks() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    paths = 4
    steps = 3
    normals = torch.linspace(
        -1.5,
        1.5,
        paths * steps * (simulator.dynamics.reduced_state_size + 1),
        dtype=env.fdtype,
    ).reshape(paths, steps, simulator.dynamics.reduced_state_size + 1)
    initial = torch.zeros(
        simulator.dynamics.reduced_state_size,
        dtype=env.fdtype,
        requires_grad=True,
    )
    spot = torch.tensor(100.0, dtype=env.fdtype, requires_grad=True)
    normal = simulator.simulate_paths(
        initial,
        spot,
        num_paths=paths,
        num_steps=steps,
        spot_mode="normal",
        standard_normals=normals,
    )
    implied = simulator.simulate_paths(
        initial,
        spot,
        num_paths=paths,
        num_steps=steps,
        spot_mode="implied",
        standard_normals=normals,
    )
    torch.testing.assert_close(normal.reduced_states, implied.reduced_states)
    assert not torch.allclose(normal.log_spots, implied.log_spots)
    assert torch.all(torch.isfinite(implied.log_spots))
    implied.log_spots[:, -1].sum().backward()
    assert initial.grad is not None and torch.all(torch.isfinite(initial.grad))
    assert spot.grad is not None and torch.all(torch.isfinite(spot.grad))


def test_front_inverse_cdf_is_monotone_and_differentiable() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    state = torch.zeros((3, simulator.dynamics.reduced_state_size), dtype=env.fdtype)
    surface = simulator.decode_surface(state)
    gaussian = torch.tensor([-1.0, 0.0, 1.0], dtype=env.fdtype, requires_grad=True)
    gross = front_week_sanos_inverse_cdf(surface, gaussian)
    assert torch.all(torch.diff(gross) > 0.0)
    gross.sum().backward()
    assert gaussian.grad is not None
    assert torch.all(torch.isfinite(gaussian.grad) & (gaussian.grad > 0.0))


def test_implied_front_distribution_reproduces_front_call_prices() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    num_quantiles = 4096
    probabilities = (
        torch.arange(num_quantiles, dtype=env.fdtype) + 0.5
    ) / num_quantiles
    gaussian = torch.special.ndtri(probabilities)
    state = torch.zeros(
        (num_quantiles, simulator.dynamics.reduced_state_size), dtype=env.fdtype
    )
    surface = simulator.decode_surface(state)
    gross_return = front_week_sanos_inverse_cdf(surface, gaussian)
    assert torch.mean(gross_return).item() == pytest.approx(1.0, abs=5.0e-4)

    strikes = torch.tensor([0.85, 1.0, 1.15], dtype=env.fdtype)
    sampled_calls = torch.mean(
        torch.relu(gross_return.unsqueeze(-1) - strikes), dim=0
    )
    initial_surface = simulator.decode_surface(state[:1])
    front_expiry = simulator.calibrated_model.sanos_decoder.model_expiries_y[0]
    decoded_calls = simulator.calibrated_model.sanos_decoder.price_calls_batch(
        initial_surface,
        expiries_y=front_expiry.expand(1, len(strikes)),
        strikes=strikes.unsqueeze(0),
    )[0]
    torch.testing.assert_close(sampled_calls, decoded_calls, atol=5.0e-4, rtol=0.0)


def test_antithetic_front_audit_and_q1_only_training(tmp_path: Path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    dynamics = simulator.dynamics
    states = torch.zeros((4, dynamics.reduced_state_size), dtype=env.fdtype)
    strikes = torch.tensor([0.9, 1.0, 1.1], dtype=env.fdtype)
    data = precompute_measure_training_data(simulator, states, strikes)
    front = audit_front_week_consistency(
        data,
        strikes,
        paths_per_origin=512,
        origins_per_batch=2,
        seed=8701,
    )
    assert front["maximum_density_mass_error"] < 1.0e-10
    assert front["maximum_analytic_mean_gross_return_error"] < 1.0e-10
    # The exact density moments above establish the identity; this finite-path
    # assertion only guards the antithetic inverse-CDF implementation.
    assert front["conditional_moment_rmse"] < 5.0e-3

    kernel = NormalWeeklyMeasureKernel(
        env,
        dynamics=dynamics,
        config=WeeklyMeasureConfig(
            label="test-q1-only",
            state_sizes_by_expiry=dynamics.reduced_sizes_by_expiry,
            hierarchical=True,
            context_network=FeedForwardConfig(
                activation="SiLU",
                num_depth=1,
                num_width=8,
            ),
        ),
    )
    gross = torch.ones(len(states), dtype=env.fdtype)
    torch.testing.assert_close(
        kernel.conditional_kl_loss(states, gross, line_index=0),
        torch.zeros((), dtype=env.fdtype),
    )
    config = WeeklyMeasureTrainConfig(
        epochs_per_line=1,
        origins_per_batch=2,
        paths_per_origin=4,
        audit_origins=2,
        audit_paths_per_origin=8,
        report_every=1,
        checkpoint_every=1,
        compile_objective=False,
        antithetic_paths=True,
        maximum_roll_lines=1,
        log_moneyness=tuple(float(torch.log(value)) for value in strikes),
        absolute_audit_tolerance=1.0e6,
    )
    result = fit_weekly_measure_kernel(
        kernel,
        simulator,
        data,
        data,
        checkpoint_path=tmp_path / "q1.pt",
        config=config,
    )
    assert result["status"] == "complete"
    assert len(result["lines"]) == 1
    assert result["lines"][0]["current_expiry_index"] == 1
    audit = audit_weekly_roll_line(
        kernel,
        simulator,
        data,
        strikes,
        current_expiry_index=1,
        paths_per_origin=8,
        origins_per_batch=2,
        seed=8702,
    )
    assert audit["num_origins"] == len(states)
    assert audit["paths_per_origin"] == 8


def test_joint_hierarchical_adapter_is_identity_state_p_law() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=8,
        num_model_expiries=3,
    )
    config = JointHierarchicalDynamicsConfig(
        label="test-joint-q-adapter",
        state_sizes_by_expiry=(3, 2, 2),
        state_size=7,
        geometry=geometry,
    )
    model = JointHierarchicalDynamicsModel(env, num_days=2, config=config)
    adapter = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.zeros(3, dtype=env.fdtype),
        spot_orthogonal_std=1.0,
    )
    assert adapter.reduced_state_size == 7
    assert adapter.reduced_sizes_by_expiry == (3, 2, 2)
    states = torch.linspace(-0.2, 0.2, 14, dtype=env.fdtype).reshape(2, 7)
    torch.testing.assert_close(adapter.decode_full_state(states), states)
    torch.testing.assert_close(
        adapter.weekly_surface_transition,
        model.weekly_transition(),
    )
    torch.testing.assert_close(
        adapter.weekly_joint_covariance[1:, 1:],
        model.innovation_cholesky() @ model.innovation_cholesky().T,
    )
    torch.testing.assert_close(
        adapter.weekly_joint_covariance[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=env.fdtype),
    )


def test_joint_spot_correlation_is_learned_under_p() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=8,
        num_model_expiries=3,
    )
    config = JointHierarchicalDynamicsConfig(
        label="test-joint-spot-correlation",
        state_sizes_by_expiry=(3, 2, 2),
        state_size=7,
        geometry=geometry,
    )
    model = JointHierarchicalDynamicsModel(env, num_days=1000, config=config)
    dates = tuple(get_business_days("2020-01-02", "2024-12-31"))[:1000]
    rng = np.random.default_rng(27013)
    transition = model.weekly_transition().detach().numpy()
    factor = model.innovation_cholesky().detach().numpy()
    equilibrium = model.equilibrium_mean.detach().numpy()
    states = np.zeros((len(dates), config.state_size), dtype=np.float64)
    log_spot = np.zeros(len(dates), dtype=np.float64)
    expected_beta = np.asarray((0.03, -0.02, 0.01), dtype=np.float64)
    expected_drift = 0.001
    expected_orthogonal_std = 0.015
    for index in range(5, len(dates)):
        standard_normal = rng.normal(size=config.state_size)
        states[index] = (
            equilibrium
            + (states[index - 5] - equilibrium) @ transition
            + standard_normal @ factor.T
        )
        spot_noise = rng.normal()
        log_spot[index] = (
            log_spot[index - 5]
            + expected_drift
            + standard_normal[:3] @ expected_beta
            + expected_orthogonal_std * spot_noise
        )
    fit = fit_joint_spot_correlation_under_p(model, states, log_spot, dates)
    np.testing.assert_allclose(fit.front_beta, expected_beta, atol=2.0e-3)
    assert fit.weekly_drift == pytest.approx(expected_drift, abs=1.0e-3)
    assert fit.orthogonal_std == pytest.approx(expected_orthogonal_std, abs=1.0e-3)
    diagnostics = evaluate_joint_spot_correlation_under_p(
        fit,
        model,
        states,
        log_spot,
        dates,
    )
    assert diagnostics["standardized_residual_std"] == pytest.approx(1.0, abs=0.02)

    adapter = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(fit.front_beta, dtype=env.fdtype),
        spot_orthogonal_std=fit.orthogonal_std,
    )
    expected_cross = factor[:, :3] @ fit.front_beta
    np.testing.assert_allclose(
        adapter.weekly_joint_covariance[0, 1:].detach().numpy(),
        expected_cross,
        atol=1.0e-12,
    )


def test_hjm_roll_hook_matches_cash_roll_formula_and_has_gradients() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    paths = 4
    current = torch.zeros(
        (paths, simulator.dynamics.reduced_state_size),
        dtype=env.fdtype,
        requires_grad=True,
    )
    next_state = torch.linspace(
        -0.05,
        0.05,
        paths * simulator.dynamics.reduced_state_size,
        dtype=env.fdtype,
    ).reshape(paths, -1).detach().requires_grad_()
    gross = torch.tensor([0.9, 0.98, 1.03, 1.12], dtype=env.fdtype, requires_grad=True)
    strikes = torch.tensor([0.9, 1.0, 1.1], dtype=env.fdtype)
    residuals = simulator.hjm_roll_residuals(
        current, next_state, gross, strikes
    )
    assert residuals.shape == (paths, simulator.num_model_expiries, len(strikes))

    current_surface = simulator.decode_surface(current)
    front_expiry = simulator.calibrated_model.sanos_decoder.model_expiries_y[0]
    front_calls = simulator.calibrated_model.sanos_decoder.price_calls_batch(
        current_surface,
        expiries_y=front_expiry.expand(paths, len(strikes)),
        strikes=strikes.expand(paths, -1),
    )
    torch.testing.assert_close(
        residuals[:, 0],
        torch.relu(gross.unsqueeze(-1) - strikes) - front_calls,
    )
    for expiry_index in range(simulator.num_model_expiries):
        line = simulator.hjm_roll_line_residuals(
            current,
            next_state,
            gross,
            strikes,
            current_expiry_index=expiry_index,
        )
        torch.testing.assert_close(line, residuals[:, expiry_index])

    instruments = torch.stack((torch.ones_like(gross), current[:, 0]), dim=-1)
    loss = hjm_roll_moment_loss(residuals, instruments)
    loss.backward()
    assert current.grad is not None and torch.all(torch.isfinite(current.grad))
    assert next_state.grad is not None and torch.all(torch.isfinite(next_state.grad))
    assert gross.grad is not None and torch.all(torch.isfinite(gross.grad))


@pytest.mark.parametrize(
    "kernel_type", [NormalWeeklyMeasureKernel, GenericWeeklyMeasureKernel]
)
def test_weekly_measure_kernels_start_at_p_and_train_through_roll_loss(
    kernel_type: type[NormalWeeklyMeasureKernel] | type[GenericWeeklyMeasureKernel],
) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    dynamics = simulator.dynamics
    kernel = kernel_type(
        env,
        dynamics=dynamics,
        config=WeeklyMeasureConfig(
            label=f"test-{kernel_type.__name__}",
            state_sizes_by_expiry=dynamics.reduced_sizes_by_expiry,
            hierarchical=True,
            context_network=FeedForwardConfig(
                activation="SiLU",
                num_depth=1,
                num_width=16,
            ),
        ),
    )
    paths = 8
    current = torch.zeros(
        (paths, dynamics.reduced_state_size), dtype=env.fdtype
    )
    normals = torch.linspace(
        -1.5,
        1.5,
        paths * (dynamics.reduced_state_size + 1),
        dtype=env.fdtype,
    ).reshape(paths, -1)
    current_surface = simulator.decode_surface(current)
    gross = front_week_sanos_inverse_cdf(current_surface, normals[:, 0])
    draw = kernel.step(current, gross, normals)
    torch.testing.assert_close(draw.transformed_standard_normals, normals)
    torch.testing.assert_close(draw.reduced_state, draw.reference_reduced_state)
    residuals = simulator.hjm_roll_residuals(
        current,
        draw.reduced_state,
        gross,
        torch.tensor([0.9, 1.0, 1.1], dtype=env.fdtype),
    )
    loss = hjm_roll_moment_loss(residuals)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in kernel.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.all(torch.isfinite(value)) for value in gradients)
    assert any(torch.any(value != 0.0) for value in gradients)
    if isinstance(kernel, GenericWeeklyMeasureKernel):
        assert kernel.minimum_coordinate_derivative(current, gross, normals) > 0.0

    path_normals = normals[:2].unsqueeze(1).expand(-1, 2, -1).contiguous()
    paths = simulator.simulate_paths(
        current[0],
        torch.tensor(100.0, dtype=env.fdtype),
        num_paths=2,
        num_steps=2,
        spot_mode="implied",
        standard_normals=path_normals,
        measure_kernel=kernel,
    )
    assert paths.reduced_states.shape == (2, 3, dynamics.reduced_state_size)
    assert paths.log_spots.shape == (2, 3)
    assert torch.all(torch.isfinite(paths.reduced_states))
    with pytest.raises(ValueError, match="requires implied"):
        simulator.simulate_paths(
            current[0],
            torch.tensor(100.0, dtype=env.fdtype),
            num_paths=2,
            num_steps=2,
            spot_mode="normal",
            standard_normals=path_normals,
            measure_kernel=kernel,
        )


def test_weekly_measure_training_checkpoints_sequential_lines(tmp_path: Path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    dynamics = simulator.dynamics
    kernel = NormalWeeklyMeasureKernel(
        env,
        dynamics=dynamics,
        config=WeeklyMeasureConfig(
            label="test-sequential-q-training",
            state_sizes_by_expiry=dynamics.reduced_sizes_by_expiry,
            hierarchical=True,
            context_network=FeedForwardConfig(
                activation="SiLU",
                num_depth=1,
                num_width=8,
            ),
        ),
    )
    states = torch.linspace(
        -0.1,
        0.1,
        6 * dynamics.reduced_state_size,
        dtype=env.fdtype,
    ).reshape(6, -1)
    strikes = torch.tensor([0.9, 1.0, 1.1], dtype=env.fdtype)
    data = precompute_measure_training_data(simulator, states, strikes)
    config = WeeklyMeasureTrainConfig(
        epochs_per_line=1,
        origins_per_batch=2,
        paths_per_origin=4,
        audit_origins=2,
        audit_paths_per_origin=8,
        report_every=1,
        checkpoint_every=1,
        compile_objective=False,
        log_moneyness=tuple(float(torch.log(value)) for value in strikes),
        absolute_audit_tolerance=1.0e6,
    )
    checkpoint = tmp_path / "normal-q.pt"
    result = fit_weekly_measure_kernel(
        kernel,
        simulator,
        data,
        data,
        checkpoint_path=checkpoint,
        config=config,
    )
    assert result["status"] == "complete"
    assert len(result["lines"]) == simulator.num_model_expiries - 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["phase"] == "complete"
    loaded = fit_weekly_measure_kernel(
        kernel,
        simulator,
        data,
        data,
        checkpoint_path=checkpoint,
        config=config,
    )
    assert loaded["checkpoint_action"] == "loaded"


@pytest.mark.parametrize("spot_mode", ["normal", "implied"])
def test_option_returns_have_exact_horizons_zero_calls_and_costs(
    spot_mode: str,
) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    paths = 2
    steps = simulator.num_model_expiries
    internal_steps = 2 * steps - 1
    normals = torch.zeros(
        (
            paths,
            internal_steps,
            simulator.dynamics.reduced_state_size + 1,
        ),
        dtype=env.fdtype,
    )
    grid_config = WeeklyOptionGridConfig(
        log_strikes=(-0.1, 0.1),
        proportional_cost=0.01,
        spot_proportional_cost=0.002,
    )
    result = simulator.simulate_option_returns(
        torch.zeros(simulator.dynamics.reduced_state_size, dtype=env.fdtype),
        torch.tensor(100.0, dtype=env.fdtype),
        num_paths=paths,
        num_steps=steps,
        grid_config=grid_config,
        return_mode="expiry",
        spot_mode=spot_mode,
        standard_normals=normals,
    )
    assert result.internal_simulation_steps == 2 * steps - 1
    assert result.returns.shape == (paths, steps, steps * 3)
    assert result.retained_paths.reduced_states.shape[1] == steps + 1
    zero = result.grid.is_zero_strike
    entry_spots = result.retained_paths.spots[:, :-1]
    torch.testing.assert_close(
        result.entry_prices[..., zero],
        entry_spots.unsqueeze(-1).expand(-1, -1, steps),
    )
    expected_rates = torch.where(
        result.grid.is_zero_strike,
        torch.tensor(0.002, dtype=env.fdtype),
        torch.tensor(0.01, dtype=env.fdtype),
    )
    torch.testing.assert_close(result.entry_costs, expected_rates * result.entry_prices)
    torch.testing.assert_close(result.exit_costs, torch.zeros_like(result.exit_costs))
    torch.testing.assert_close(result.transaction_costs, result.entry_costs)
    torch.testing.assert_close(
        result.gross_returns,
        (result.exit_values - result.entry_prices) / result.entry_prices,
    )
    assert torch.all(torch.isfinite(result.returns))

    period_normals = normals[:, :steps]
    per_option_cost = torch.linspace(
        0.0, 0.02, result.grid.num_options, dtype=env.fdtype
    )
    period = simulator.simulate_option_returns(
        torch.zeros(simulator.dynamics.reduced_state_size, dtype=env.fdtype),
        torch.tensor(100.0, dtype=env.fdtype),
        num_paths=paths,
        num_steps=steps,
        grid_config=replace(grid_config, proportional_cost=0.0),
        return_mode="period",
        spot_mode=spot_mode,
        proportional_costs=per_option_cost,
        standard_normals=period_normals,
    )
    assert period.internal_simulation_steps == steps
    torch.testing.assert_close(
        period.transaction_costs,
        per_option_cost * (period.entry_prices + period.exit_values),
    )
    first_expiry = period.grid.expiry_index == 0
    next_spot = period.retained_paths.spots[:, 1:]
    strikes = period.retained_paths.spots[:, :-1].unsqueeze(-1) * torch.exp(
        torch.where(
            period.grid.is_zero_strike,
            torch.zeros_like(period.grid.log_moneyness),
            period.grid.log_moneyness,
        )
    )
    strikes = torch.where(period.grid.is_zero_strike, 0.0, strikes)
    expected_payoff = torch.where(
        period.grid.is_call,
        torch.relu(next_spot.unsqueeze(-1) - strikes),
        torch.relu(strikes - next_spot.unsqueeze(-1)),
    )
    torch.testing.assert_close(
        period.exit_values[..., first_expiry],
        expected_payoff[..., first_expiry],
    )


def test_option_side_selection_and_zero_strike_convention() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    itm = make_tradable_option_grid(
        env,
        num_expiries=2,
        config=WeeklyOptionGridConfig(log_strikes=(-0.2, 0.2), option_mode="itm"),
    )
    assert itm.num_options == 6
    assert torch.all(itm.is_call[itm.is_zero_strike])
    assert itm.is_call.tolist() == [True, True, False, True, True, False]
    otm = make_tradable_option_grid(
        env,
        num_expiries=1,
        config=WeeklyOptionGridConfig(log_strikes=(-0.2, 0.2), option_mode="otm"),
    )
    assert otm.is_call.tolist() == [True, False, True]


def test_weekly_validation_uses_exact_horizons_and_training_pca() -> None:
    sizes = (3, 3, 2)
    states, log_spot = _history(sizes, observations=240)
    training_dates = tuple(get_business_days("2024-01-02", "2024-12-31"))[:180]
    training_history = WeeklyTrainingHistory(
        dates=training_dates,
        states=states[:180],
        log_spot=log_spot[:180],
    )
    fit = fit_weekly_hierarchical_dynamics(
        states[:180],
        log_spot[:180],
        config=WeeklyDynamicsConfig(
            label="phase3-validation-test",
            state_sizes_by_expiry=sizes,
            pca_mode="full",
            ridge_penalty=1.0e-5,
        ),
    )
    validation_dates = tuple(get_business_days("2025-01-02", "2025-03-31"))
    validation_dates = validation_dates[:60]
    history = WeeklyTrainingHistory(
        dates=validation_dates,
        states=states[180 : 180 + len(validation_dates)],
        log_spot=log_spot[180 : 180 + len(validation_dates)],
    )
    reduced = project_strong_states(fit, history.states)
    assert reduced.shape == (len(validation_dates), sum(sizes))
    for weeks in (1, 2, 4):
        origin, endpoint = exact_weekly_pairs(validation_dates, weeks=weeks)
        assert len(origin) == len(validation_dates) - 5 * weeks
        assert np.all(endpoint - origin == 5 * weeks)
        transition, covariance = propagated_joint_moments(fit, weeks=weeks)
        assert transition.shape == covariance.shape == (sum(sizes) + 1,) * 2
        assert np.min(np.linalg.eigvalsh(covariance)) > 0.0

    report = evaluate_oos_dynamics(fit, history, training_history)
    assert report["all_horizons_finite"] is True
    assert report["all_horizons_nondegenerate"] is True
    assert set(report["horizons"]) == {"1w", "2w", "4w"}
    one_week = report["horizons"]["1w"]
    assert len(one_week["coordinate_negative_log_likelihood"]) == sum(sizes) + 1
    assert len(one_week["expiry_conditional_nll_per_dimension"]) == len(sizes)
    assert one_week["nonoverlapping_pairs"] > 0
    assert np.isfinite(one_week["joint_energy_score"])


def test_weekly_dynamics_checkpoint_roundtrip(tmp_path: Path) -> None:
    states, log_spot = _history((3, 3, 2), observations=180)
    fit = fit_weekly_hierarchical_dynamics(
        states,
        log_spot,
        config=WeeklyDynamicsConfig(
            label="phase3-checkpoint-test",
            state_sizes_by_expiry=(3, 3, 2),
            pca_mode="full",
            ridge_penalty=1.0e-5,
        ),
    )
    path = tmp_path / "weekly.pt"
    source = {"kind": "test", "sha256": "abc"}
    save_weekly_dynamics_fit(fit, path, source=source)
    restored, restored_source = load_weekly_dynamics_fit(path)
    assert restored_source == source
    assert restored.config == fit.config
    assert restored.reduced_sizes_by_expiry == fit.reduced_sizes_by_expiry
    np.testing.assert_array_equal(
        restored.mean_reversion_generator,
        fit.mean_reversion_generator,
    )
    np.testing.assert_array_equal(
        restored.weekly_joint_innovation_covariance,
        fit.weekly_joint_innovation_covariance,
    )
    for expected, actual in zip(fit.blocks, restored.blocks, strict=True):
        np.testing.assert_array_equal(actual.own_lag, expected.own_lag)
        assert actual.r2 == expected.r2


def test_option_returns_accept_differentiable_linear_spread_model() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    estimator = LinearRegression().fit(
        np.asarray(
            [
                [0.1, -0.8, -2.0],
                [0.2, 0.5, 0.0],
                [0.3, 0.9, 2.0],
            ]
        ),
        np.asarray([0.0010, 0.0015, 0.0020]),
    )
    spread = FittedSpreadCostModel(
        name="linear-test",
        estimator=estimator,
        feature_names=LINEAR_FEATURE_NAMES,
    )
    initial = torch.zeros(
        simulator.dynamics.reduced_state_size,
        dtype=env.fdtype,
        requires_grad=True,
    )
    result = simulator.simulate_option_returns(
        initial,
        torch.tensor(100.0, dtype=env.fdtype),
        num_paths=2,
        num_steps=2,
        grid_config=WeeklyOptionGridConfig(
            log_strikes=(-0.1, 0.1),
            spot_proportional_cost=0.003,
        ),
        spread_cost_model=spread,
        standard_normals=torch.zeros(
            (2, 2, simulator.dynamics.reduced_state_size + 1),
            dtype=env.fdtype,
        ),
    )
    assert torch.all(result.transaction_costs >= 0.0)
    zero = result.grid.is_zero_strike
    torch.testing.assert_close(
        result.entry_costs[..., zero],
        0.003 * result.entry_prices[..., zero],
    )
    result.returns.sum().backward()
    assert initial.grad is not None and torch.all(torch.isfinite(initial.grad))


def test_option_returns_accept_tick_aware_feed_forward_spread_model() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = _simulator(env)
    spread = TorchSpreadCostModel(
        env,
        config=SpreadFeedForwardConfig(label="simulation-test"),
        feature_mean=torch.zeros(6, dtype=env.fdtype),
        feature_scale=torch.ones(6, dtype=env.fdtype),
        log_target_mean=torch.tensor(-8.0, dtype=env.fdtype),
        log_target_scale=torch.tensor(1.0, dtype=env.fdtype),
    )
    initial = torch.zeros(
        simulator.dynamics.reduced_state_size,
        dtype=env.fdtype,
        requires_grad=True,
    )
    result = simulator.simulate_option_returns(
        initial,
        torch.tensor(100.0, dtype=env.fdtype),
        num_paths=2,
        num_steps=2,
        grid_config=WeeklyOptionGridConfig(
            log_strikes=(-0.1, 0.1),
            spot_proportional_cost=0.003,
        ),
        spread_cost_model=spread,
        standard_normals=torch.zeros(
            (2, 2, simulator.dynamics.reduced_state_size + 1),
            dtype=env.fdtype,
        ),
    )
    assert torch.all(torch.isfinite(result.returns))
    assert torch.all(result.transaction_costs >= 0.0)
    result.returns.sum().backward()
    assert initial.grad is not None and torch.all(torch.isfinite(initial.grad))


def test_load_train_only_phase3_strong_checkpoint(tmp_path: Path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = TriangularPcaDecoderConfig(
        label="phase3-train-only-test",
        state_sizes_by_expiry=STRONG_STATE_SIZES,
        geometry=ForwardMuFlexGeometryConfig(),
    )
    expected = LatentStateSanosModel(env, num_days=2, config=config)
    checkpoint = tmp_path / "strong.pt"
    torch.save(
        {
            "schema_version": 1,
            "phase": "complete",
            "config_version": config.version.unique_id48,
            "model_config": asdict(config),
            "train_market_fingerprints": ["date-1", "date-2"],
            "native_target_sha256": "a" * 64,
            "convergence": {"converged": True},
            "model_state_dict": expected.state_dict(),
        },
        checkpoint,
    )
    restored = load_strong_calibrated_model(env, checkpoint)
    assert restored.config == config
    torch.testing.assert_close(restored.daily_states, expected.daily_states)
