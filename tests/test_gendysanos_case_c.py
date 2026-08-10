from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.gendysanos.case_c import (
    CaseCAction,
    CaseCConfig,
    case_c_numerical_band,
    case_c_objective_and_surrogate,
    case_c_pnl,
    normalized_marginal_weights,
    powerexp_one,
    projected_long_short,
    simplex_long_short,
)
from quantdata.dysanos.grid.gendysanos.case_c_kernel import (
    KernelCaseCAction,
    KernelCaseCConfig,
    normalized_gaussian_kernel_weights,
)
from quantdata.dysanos.grid.gendysanos.case_c_partition import (
    GatedParallelFeedForwardDirections,
    LearnedPartitionDirections,
    ParallelDirectionConfig,
    ParallelFeedForwardDirections,
    ParallelKernelDirections,
    ParallelRadialFeedForwardDirections,
    project_rows_to_l1_ball_,
    project_rows_to_simplex_,
    unit_gross_long_short,
)
from quantdata.dysanos.grid.gendysanos.run_expiry_arbitrage_experiments import (
    DeviceExpiryCase,
)
from quantdata.dysanos.grid.gendysanos.run_implied_cvar import (
    weighted_lower_tail_mean,
)
from quantdata.dysanos.grid.gendysanos.run_implied_cvar_t0_lp import (
    _independent_t0_returns,
    solve_density_cap_feasibility,
    solve_minimum_density_cap,
    solve_strict_sample_arbitrage_margin,
)
from quantdata.dysanos.grid.gendysanos.run_implied_cvar_t6_dual import (
    conditional_moments,
    fit_entropy_density,
    project_weighted_capped_density_,
)
from quantdata.dysanos.grid.gendysanos.run_implied_cvar_t6_partition_lp import (
    combine_partition_densities,
)
from quantdata.dysanos.grid.gendysanos.run_parallel_direction_comparison import (
    _balanced_component_bucket_assignment,
    _bucket_downside_loss,
    _bucket_pnl_audit,
    _exact_expiry_lower_bound,
    _finite_maximum_or_none,
    _model_loss,
    _oce_numpy_statistics,
    _parallel_pnl,
    _recalibrate_oce_translation,
    _squared_downside,
    _to_itm_dollar_portfolio,
    _to_itm_dollar_vector_barrier,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _config() -> CaseCConfig:
    return CaseCConfig(
        label="test-case-c",
        source_cache_uid="test-cache",
        state_features=2,
        notionals=(1.0, 2.0, 4.0),
        expiries=1,
        instruments_per_expiry=3,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="SiLU",
            use_bias=True,
        ),
    )


def _model() -> CaseCAction:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    return CaseCAction(
        env,
        config=_config(),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )


def test_exact_weighted_lower_tail_mean_and_gradient() -> None:
    value = torch.tensor(
        [
            [[3.0], [4.0]],
            [[-1.0], [-2.0]],
            [[2.0], [1.0]],
            [[0.0], [3.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = weighted_lower_tail_mean(
        value,
        torch.ones(4, dtype=torch.float64),
        torch.tensor((1.0, 2.0), dtype=torch.float64),
    )
    assert torch.allclose(
        result[:, 0], torch.tensor((1.0, -0.5), dtype=torch.float64)
    )
    result.sum().backward()
    expected = torch.tensor(
        [
            [[0.25], [0.0]],
            [[0.25], [0.5]],
            [[0.25], [0.5]],
            [[0.25], [0.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(value.grad, expected)


def test_exact_minimum_density_cap_and_arbitrage_zeroing() -> None:
    result, density = solve_minimum_density_cap(
        np.asarray([[-1.0], [0.0], [2.0]]),
        np.full(3, 1.0 / 3.0),
    )
    assert result.success and density is not None
    assert result.fun == pytest.approx(1.2)
    assert density.tolist() == pytest.approx([1.2, 1.2, 0.6])

    arbitrage, arbitrage_density = solve_minimum_density_cap(
        np.asarray([[0.0], [1.0], [2.0]]),
        np.full(3, 1.0 / 3.0),
    )
    assert arbitrage.success and arbitrage_density is not None
    assert arbitrage.fun == pytest.approx(3.0)
    assert arbitrage_density.tolist() == pytest.approx([3.0, 0.0, 0.0])

    strict = solve_strict_sample_arbitrage_margin(
        np.asarray([[1.0], [2.0]], dtype=np.float64)
    )
    assert strict.success and -strict.fun == pytest.approx(1.0)
    symmetric = solve_strict_sample_arbitrage_margin(
        np.asarray([[-1.0], [1.0]], dtype=np.float64)
    )
    assert symmetric.success and -symmetric.fun == pytest.approx(0.0)
    feasible = solve_density_cap_feasibility(
        np.asarray([[-1.0], [0.0], [2.0]]),
        np.full(3, 1.0 / 3.0),
        1.2,
    )
    infeasible = solve_density_cap_feasibility(
        np.asarray([[-1.0], [0.0], [2.0]]),
        np.full(3, 1.0 / 3.0),
        1.1,
    )
    assert feasible.success and not infeasible.success


def test_time_zero_lp_retains_every_expiry_specific_stock_return() -> None:
    raw = torch.tensor(
        [
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            [[[[5.0, 6.0], [7.0, 8.0]]]],
        ],
        dtype=torch.float64,
    ).reshape(2, 1, 2, 2)
    result = _independent_t0_returns(SimpleNamespace(returns=raw))
    flattened = raw[:, 0].numpy().reshape(2, 4)
    expected = flattened / np.sqrt(np.mean(np.square(flattened), axis=0))[None]

    assert result.shape == (2, 4)
    assert result == pytest.approx(expected)


def test_partition_density_combination_has_minimum_global_cap() -> None:
    probability = np.full(4, 0.25)
    assignment = np.array([0, 0, 1, 1])
    cap, density, target_mass = combine_partition_densities(
        probability,
        assignment,
        {
            0: (2.0, np.array([0.0, 2.0])),
            1: (1.0, np.array([1.0, 1.0])),
        },
    )

    assert cap == pytest.approx(4.0 / 3.0)
    assert density == pytest.approx([0.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0])
    assert probability @ density == pytest.approx(1.0)
    assert target_mass == pytest.approx({0: 1.0 / 3.0, 1: 2.0 / 3.0})


def test_weighted_capped_density_projection_and_conditional_moments() -> None:
    probability = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64)
    density = torch.tensor((-2.0, 4.0, 0.2), dtype=torch.float64)
    project_weighted_capped_density_(density, probability, 2.0)
    assert torch.all((density >= 0.0) & (density <= 2.0))
    assert torch.dot(probability, density).item() == pytest.approx(1.0)

    cap_one = torch.tensor((-2.0, 4.0, 0.2), dtype=torch.float64)
    project_weighted_capped_density_(cap_one, probability, 1.0)
    assert torch.allclose(cap_one, torch.ones_like(cap_one))

    kernel = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float64
    )
    returns = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    moment = conditional_moments(density, probability, kernel, returns)
    expected = kernel.mT @ (probability[:, None] * density[:, None] * returns)
    assert torch.allclose(moment, expected)

    symmetric_probability = torch.full((2,), 0.5, dtype=torch.float64)
    symmetric_kernel = torch.ones((2, 1), dtype=torch.float64)
    symmetric_return = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    entropy_density, diagnostics = fit_entropy_density(
        symmetric_probability,
        symmetric_kernel,
        symmetric_return,
        max_iter=20,
    )
    assert torch.allclose(entropy_density, torch.ones_like(entropy_density))
    assert diagnostics["conditional_OCE_maximum_upper_bound"] < 1.0e-12


def test_projection_enforces_separate_nonnegative_unit_gross_actions() -> None:
    raw = torch.tensor(
        [[[[[2.0, -1.0, 1.0]], [[1.0, 0.0, -3.0]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    action, gross, boundary = projected_long_short(raw)
    assert action.shape == raw.shape
    assert torch.all(action >= 0.0)
    assert gross.item() == pytest.approx(1.0)
    assert boundary.item() is True
    assert action[0, 0].sum().item() == pytest.approx(1.0)

    weighted = action * torch.arange(1, 7, dtype=action.dtype).reshape_as(action)
    weighted.sum().backward()
    # The exact radial projection removes only scale.  Tangential gradients
    # remain and can change the portfolio direction outside the ball.
    assert torch.count_nonzero(raw.grad).item() > 0
    positive_raw = torch.clamp_min(raw.detach(), 0.0)
    assert (raw.grad * positive_raw).sum().item() == pytest.approx(0.0, abs=1.0e-12)


def test_zero_action_is_finite_and_has_identity_gradient() -> None:
    raw = torch.zeros((4, 2, 2, 1, 3), dtype=torch.float64, requires_grad=True)
    action, gross, boundary = projected_long_short(raw)
    assert torch.equal(action, torch.zeros_like(action))
    assert torch.equal(gross, torch.zeros_like(gross))
    assert not boundary.any()
    action.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_simplex_map_includes_uninvested_gross_without_penalty() -> None:
    # Four action coordinates plus one no-trade coordinate.
    logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, math.log(4.0)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    action, gross, boundary = simplex_long_short(
        logits, expiries=1, instruments_per_expiry=2
    )
    assert action.shape == (1, 2, 1, 2)
    assert torch.all(action >= 0.0)
    assert gross.item() == pytest.approx(0.5)
    assert boundary.item() is False
    action[..., 0, 0, 0].sum().backward()
    assert torch.count_nonzero(logits.grad).item() == logits.numel()


def test_long_short_pnl_and_cost_do_not_net() -> None:
    returns = torch.tensor([[[[0.4, -0.2, 0.1]]]], dtype=torch.float64)
    costs = torch.tensor([[[[0.01, 0.02, 0.03]]]], dtype=torch.float64)
    action = torch.tensor(
        [[[[[0.2, 0.1, 0.0]], [[0.0, 0.3, 0.1]]]]],
        dtype=torch.float64,
    )
    expected = (
        0.2 * 0.4
        + 0.1 * -0.2
        + 0.3 * 0.2
        - 0.1 * 0.1
        - (0.2 * 0.01 + 0.1 * 0.02 + 0.3 * 0.02 + 0.1 * 0.03)
    )
    assert case_c_pnl(action, returns, costs).item() == pytest.approx(expected)


def test_powerexp_value_and_marginal_are_continuous_at_zero() -> None:
    value = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    utility, marginal = powerexp_one(value)
    assert torch.allclose(
        utility,
        torch.tensor(
            [-4.0, 0.0, 1.0 - torch.exp(torch.tensor(-2.0)).item()],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        marginal,
        torch.tensor(
            [3.0, 1.0, torch.exp(torch.tensor(-2.0)).item()],
            dtype=torch.float64,
        ),
    )


def test_zero_pnl_marginal_weights_equal_physical_weights() -> None:
    pnl = torch.zeros((3, 2), dtype=torch.float64)
    weight = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = normalized_marginal_weights(pnl, weight, notional=64.0)
    expected = (weight / weight.sum())[:, None].expand_as(result)
    assert torch.allclose(result, expected)


def test_log_marginal_computation_remains_finite_for_large_positive_pnl() -> None:
    pnl = torch.tensor([[10.0], [-10.0]], dtype=torch.float64)
    weight = torch.ones(2, dtype=torch.float64)
    result = normalized_marginal_weights(pnl, weight, notional=256.0)
    assert torch.isfinite(result).all()
    assert result.sum().item() == pytest.approx(1.0)


def test_raw_marginal_may_underflow_but_log_gradient_remains_finite() -> None:
    pnl = torch.tensor([[1.0], [-0.01]], dtype=torch.float32)
    _, raw_marginal = powerexp_one(256.0 * pnl)
    assert raw_marginal[0].item() == 0.0
    result = normalized_marginal_weights(
        pnl, torch.ones(2, dtype=torch.float32), notional=256.0
    )
    assert torch.isfinite(result).all()
    assert result.sum().item() == pytest.approx(1.0)


def test_analytic_surrogate_gradient_matches_manual_long_short_gradient() -> None:
    returns = torch.tensor(
        [
            [[[0.4, -0.2]]],
            [[[-0.1, 0.3]]],
            [[[0.2, 0.1]]],
        ],
        dtype=torch.float64,
    )
    costs = torch.full_like(returns, 0.01)
    action = torch.tensor(
        [
            [[[[0.2, 0.1]], [[0.1, 0.0]]]],
            [[[[0.2, 0.1]], [[0.1, 0.0]]]],
            [[[[0.2, 0.1]], [[0.1, 0.0]]]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    weight = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    pnl = case_c_pnl(action, returns, costs)
    _, surrogate, marginal = case_c_objective_and_surrogate(
        pnl, weight, notional=8.0
    )
    surrogate.sum().backward()
    expected_long = marginal[:, :, None, None] * (returns - costs)
    expected_short = marginal[:, :, None, None] * (-returns - costs)
    expected = torch.stack((expected_long, expected_short), dim=-3)
    assert torch.allclose(action.grad, expected, atol=1.0e-12)


def test_surrogate_is_true_gradient_divided_by_common_marginal_scale() -> None:
    pnl = torch.tensor([[0.1], [-0.2], [0.3]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    notional = 7.0
    objective, surrogate, _ = case_c_objective_and_surrogate(
        pnl, weight, notional=notional
    )
    true_gradient = torch.autograd.grad(objective.sum(), pnl, retain_graph=True)[0]
    surrogate_gradient = torch.autograd.grad(surrogate.sum(), pnl)[0]
    _, marginal = powerexp_one(notional * pnl.detach())
    normalized_weight = weight / weight.sum()
    common_scale = notional * (normalized_weight[:, None] * marginal).sum(dim=0)
    assert torch.allclose(
        true_gradient / common_scale,
        surrogate_gradient,
        atol=1.0e-12,
    )


def test_numerical_band_propagates_gross_long_short_exposure() -> None:
    returns = torch.tensor([[[[0.4, -0.2]]]], dtype=torch.float64)
    costs = torch.full_like(returns, 0.01)
    instrument_band = torch.full_like(returns, 2.0e-6)
    action = torch.tensor(
        [[[[[0.2, 0.1]], [[0.1, 0.0]]]]], dtype=torch.float64
    )
    gross = action[..., 0, :, :] + action[..., 1, :, :]
    expected = (gross * instrument_band).sum() + 32.0 * torch.finfo(
        torch.float32
    ).eps * (gross * (returns.abs() + costs)).sum()
    assert case_c_numerical_band(
        action, returns, costs, instrument_band
    ).item() == pytest.approx(expected.item())


def test_case_c_model_uses_quantdata_torch_and_separates_t0_from_t6() -> None:
    model = _model()
    assert isinstance(model, BaseModel)
    assert isinstance(model.conditional, FeedForward)
    assert model.conditional.env is model.env
    state = torch.randn(8, 2, dtype=model.fdtype)
    action, gross, boundary, raw = model(state)
    assert action.shape == (8, 2, 2, 1, 3)
    assert raw.shape == (8, 2, 7)
    assert gross.shape == boundary.shape == (8, 2)
    assert torch.allclose(gross, torch.full_like(gross, 0.5))
    assert torch.allclose(
        action[..., 0, :, :], action[..., 1, :, :]
    )

    returns = torch.ones((8, 2, 1, 3), dtype=model.fdtype) * 0.1
    costs = torch.full_like(returns, 1.0e-4)
    pnl = case_c_pnl(action, returns, costs)
    _, surrogate, _ = case_c_objective_and_surrogate(
        pnl, torch.ones(8, dtype=model.fdtype), notional=1.0
    )
    (-surrogate.mean()).backward()
    assert model.deterministic_logits.grad is not None
    assert torch.count_nonzero(model.deterministic_logits.grad).item() > 0
    conditional_gradients = [
        parameter.grad
        for parameter in model.conditional.parameters()
        if parameter.grad is not None
    ]
    assert conditional_gradients
    assert any(torch.count_nonzero(value).item() > 0 for value in conditional_gradients)


def _kernel_model() -> KernelCaseCAction:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = KernelCaseCConfig(
        label="test-kernel-case-c",
        source_cache_uid="test-cache",
        state_features=2,
        kernels=2,
        notionals=(64.0,),
        expiries=1,
        instruments_per_expiry=3,
    )
    return KernelCaseCAction(
        env,
        config=config,
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
        centers=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=env.fdtype),
        inverse_bandwidth=torch.ones(2, dtype=env.fdtype),
    )


def test_normalized_kernel_weights_are_finite_convex_coefficients() -> None:
    factor = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    weight = normalized_gaussian_kernel_weights(
        factor, centers, torch.ones(2, dtype=torch.float64)
    )
    assert torch.isfinite(weight).all()
    assert torch.all(weight >= 0.0)
    assert torch.allclose(weight.sum(dim=-1), torch.ones(2, dtype=weight.dtype))
    assert weight[0, 0] > weight[0, 1]
    assert weight[1, 0].item() == pytest.approx(0.5)


def test_kernel_templates_remain_feasible_and_can_leave_boundary() -> None:
    model = _kernel_model()
    with torch.no_grad():
        model.deterministic_action.fill_(2.0)
        model.template_action.fill_(2.0)
    model.project_parameters_()
    assert model.deterministic_action.sum().item() == pytest.approx(1.0)
    assert torch.allclose(
        model.template_action.sum(dim=(-3, -2, -1)),
        torch.ones(2, dtype=model.fdtype),
    )

    # Direct feasible coordinates can move radially inward.  There is no
    # saturated no-trade coordinate between the objective and the templates.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    model.template_action.sum().backward()
    optimizer.step()
    model.project_parameters_()
    assert torch.all(model.template_action.sum(dim=(-3, -2, -1)) < 1.0)


def test_kernel_action_is_convex_mixture_of_local_portfolios() -> None:
    model = _kernel_model()
    with torch.no_grad():
        model.deterministic_action[0, 0, 0] = 0.4
        model.template_action[0, 0, 0, 0] = 1.0
        model.template_action[1, 1, 0, 1] = 1.0
    state = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=model.fdtype)
    action, gross, boundary, raw = model(state)
    assert action.shape == (2, 2, 2, 1, 3)
    assert gross.shape == boundary.shape == (2, 2)
    assert raw.shape == (2, 2, 6)
    assert torch.all(action >= 0.0)
    assert torch.all(gross <= 1.0 + 1.0e-12)
    assert gross[0, 1].item() == pytest.approx(1.0)
    assert action[1, 1, 0, 0, 0].item() == pytest.approx(0.5)
    assert action[1, 1, 1, 0, 1].item() == pytest.approx(0.5)


def _parallel_config(
    architecture: str,
    *,
    learn_oce_translation: bool = False,
) -> ParallelDirectionConfig:
    return ParallelDirectionConfig(
        label=f"test-parallel-{architecture}",
        source_cache_uid="test-cache",
        architecture=architecture,
        state_features=2,
        notionals=(1.0, 8.0, 64.0),
        partitions=3,
        expiries=1,
        instruments_per_expiry=2,
        learn_oce_translation=learn_oce_translation,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="SiLU",
            use_bias=True,
        ),
    )


def test_oce_translation_is_one_real_per_independent_problem() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = GatedParallelFeedForwardDirections(
        env,
        config=_parallel_config(
            "gated-feedforward",
            learn_oce_translation=True,
        ),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    assert model.oce_translation is not None
    assert model.oce_translation.shape == (3, 2)
    assert torch.equal(model.oce_translation, torch.zeros_like(model.oce_translation))


def test_oce_translation_gradient_matches_scalar_first_order_condition() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = GatedParallelFeedForwardDirections(
        env,
        config=_parallel_config(
            "gated-feedforward",
            learn_oce_translation=True,
        ),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    paths = 13
    state = torch.randn(paths, 2, dtype=env.fdtype)
    returns = torch.randn(paths, 2, 1, 2, dtype=env.fdtype) * 0.03
    costs = torch.full_like(returns, 1.0e-4)
    path_weight = torch.arange(1, paths + 1, dtype=env.fdtype)
    notionals = torch.tensor((64.0, 128.0, 256.0), dtype=env.fdtype)
    loss, values = _model_loss(
        model,
        state,
        returns,
        costs,
        path_weight,
        notionals,
        balance_coefficient=0.0,
        objective="oce-powerexp",
    )
    loss.backward()
    assert model.oce_translation is not None
    assert model.oce_translation.grad is not None
    expected = -values["oce_translation_first_order_condition"].detach() / 6.0
    assert torch.allclose(model.oce_translation.grad, expected, atol=1.0e-12)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_vector_barrier_uses_signed_net_portfolio_l1_norm_once() -> None:
    returns = torch.tensor([[[[0.4, -0.2]]]], dtype=torch.float64)
    costs = torch.full_like(returns, 0.01)
    action = torch.tensor(
        [0.30, 0.10, 0.05, 0.00], dtype=torch.float64
    ).reshape(1, 1, 1, 2, 1, 2)
    signed = torch.tensor([0.25, 0.10], dtype=torch.float64)
    expected = torch.dot(signed, returns.flatten()) + 0.01 * math.log1p(
        -signed.abs().sum().item()
    )
    value = _parallel_pnl(
        action,
        returns,
        costs,
        portfolio_barrier_rate=0.01,
    )
    assert value.item() == pytest.approx(expected.item())

    # Moving equal long and short notionals does not change the signed dollar
    # portfolio and therefore does not incur another coordinatewise barrier.
    offset = action.clone()
    offset[..., 0, 0, 0] += 0.1
    offset[..., 1, 0, 0] += 0.1
    assert _parallel_pnl(
        offset,
        returns,
        costs,
        portfolio_barrier_rate=0.01,
    ).item() == pytest.approx(value.item())


def test_vector_barrier_is_finite_at_zero_with_finite_action_gradient() -> None:
    returns = torch.tensor([[[[0.4, -0.2]]]], dtype=torch.float64)
    action = torch.zeros((1, 1, 1, 2, 1, 2), dtype=torch.float64, requires_grad=True)
    value = _parallel_pnl(
        action,
        returns,
        torch.full_like(returns, 0.01),
        portfolio_barrier_rate=0.01,
    )
    assert value.item() == pytest.approx(0.0)
    value.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert torch.count_nonzero(action.grad).item() == action.numel()


def test_itm_dollar_normalization_chooses_put_at_strike_above_spot() -> None:
    dtype = torch.float64
    data = DeviceExpiryCase(
        state=torch.zeros((1, 1, 2), dtype=dtype),
        returns=torch.tensor([[[[10.0, 12.0, 1.0]]]], dtype=dtype),
        costs=torch.zeros((1, 1, 1, 3), dtype=dtype),
        numerical_band=torch.full((1, 1, 1, 3), 1.0e-6, dtype=dtype),
        path_weight=torch.ones(1, dtype=dtype),
        component=torch.zeros(1, dtype=torch.int16),
        reference=torch.tensor([0]),
        validation=torch.tensor([0]),
        option_price=torch.tensor([[[[25.0, 2.0]]]], dtype=dtype),
        cash_strike=torch.tensor([[[[80.0, 120.0]]]], dtype=dtype),
        spot=torch.tensor([[100.0]], dtype=dtype),
    )
    linear, _ = _to_itm_dollar_portfolio(
        data,
        cost_rate=0.0,
        vector_barrier=False,
    )
    assert linear.portfolio_barrier_rate is None
    assert torch.count_nonzero(linear.costs).item() == 0
    normalized, report = _to_itm_dollar_vector_barrier(data, cost_rate=1.0e-4)
    assert normalized.option_is_put is not None
    assert normalized.option_is_put.flatten().tolist() == [False, True]
    assert normalized.option_price is not None
    assert normalized.option_price.flatten().tolist() == pytest.approx([25.0, 22.0])
    assert normalized.returns.flatten().tolist() == pytest.approx(
        [0.1, 12.0 / 25.0, -9.0 / 22.0]
    )
    assert torch.all(normalized.costs == 1.0e-4)
    assert normalized.portfolio_barrier_rate == pytest.approx(1.0e-4)
    assert report["put_coordinate_fraction"] == pytest.approx(0.5)

    action = torch.zeros((1, 1, 1, 2, 1, 3), dtype=dtype)
    action[..., 0, 0, 1] = 0.1
    exact = _exact_expiry_lower_bound(action, normalized, torch.tensor([0]))
    assert exact is not None
    lower, slope_ok = exact
    assert lower.shape == slope_ok.shape == (1, 1, 1)
    assert torch.isfinite(lower).all() and slope_ok.all()


def test_oce_zero_pnl_has_zero_translation_and_finite_zero_gradient() -> None:
    value = np.zeros(7, dtype=np.float64)
    weight = np.arange(1, 8, dtype=np.float64)
    normalized_weight = weight / weight.sum()
    translation = _recalibrate_oce_translation(value, normalized_weight, 256.0)
    statistics = _oce_numpy_statistics(value, weight, 256.0, translation)
    assert translation == pytest.approx(0.0, abs=1.0e-15)
    assert statistics["objective"] == pytest.approx(0.0, abs=1.0e-15)
    assert statistics["translation_first_order_condition"] == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert statistics["normalized_notional_derivative"] == pytest.approx(0.0)


def test_oce_recalibration_is_cash_translation_invariant() -> None:
    value = np.array([-0.03, 0.01, 0.04, 0.08], dtype=np.float64)
    weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    normalized_weight = weight / weight.sum()
    notional = 64.0
    cash_shift = 0.017
    base_y = _recalibrate_oce_translation(value, normalized_weight, notional)
    shifted_y = _recalibrate_oce_translation(
        value + cash_shift,
        normalized_weight,
        notional,
    )
    base = _oce_numpy_statistics(value, weight, notional, base_y)
    shifted = _oce_numpy_statistics(
        value + cash_shift,
        weight,
        notional,
        shifted_y,
    )
    assert shifted_y == pytest.approx(base_y - notional * cash_shift)
    assert shifted["objective"] == pytest.approx(
        base["objective"] + notional * cash_shift
    )


def test_simplex_projection_is_exact_and_does_not_freeze_zero_coordinates() -> None:
    value = torch.tensor(
        [[2.0, -1.0, 0.5], [-3.0, -2.0, -1.0]], dtype=torch.float64
    )
    project_rows_to_simplex_(value)
    assert torch.all(value >= 0.0)
    assert torch.allclose(value.sum(dim=-1), torch.ones(2, dtype=value.dtype))

    parameter = torch.nn.Parameter(value.clone())
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    # A zero coordinate remains a direct optimization variable and can recover.
    (-parameter[:, -1].sum()).backward()
    optimizer.step()
    project_rows_to_simplex_(parameter)
    assert torch.all(parameter[:, -1] > 0.0)


def test_direct_l1_ball_projection_preserves_interior_and_zero_rows() -> None:
    value = torch.tensor(
        [[0.2, 0.3, 0.1], [2.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    project_rows_to_l1_ball_(value)
    assert torch.allclose(value[0], torch.tensor([0.2, 0.3, 0.1], dtype=value.dtype))
    assert torch.allclose(
        value[1], torch.tensor([1.0, 0.0, 0.0], dtype=value.dtype)
    )
    assert torch.equal(value[2], torch.zeros(3, dtype=value.dtype))


def test_unit_gross_map_has_no_no_trade_coordinate() -> None:
    logits = torch.zeros((4, 3, 4), dtype=torch.float64)
    action = unit_gross_long_short(
        logits, expiries=1, instruments_per_expiry=2
    )
    assert action.shape == (4, 3, 2, 1, 2)
    assert torch.allclose(
        action.sum(dim=(-3, -2, -1)), torch.ones((4, 3), dtype=action.dtype)
    )


def test_squared_downside_is_c1_at_zero_and_ignores_gains() -> None:
    value = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = _squared_downside(value)
    assert torch.equal(result, torch.tensor([4.0, 0.0, 0.0], dtype=value.dtype))
    result.sum().backward()
    assert torch.equal(value.grad, torch.tensor([-4.0, 0.0, 0.0], dtype=value.dtype))


def test_downside_objective_has_finite_nonzero_direction_gradients() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = ParallelFeedForwardDirections(
        env,
        config=_parallel_config("feedforward"),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    paths = 11
    state = torch.randn(paths, 2, dtype=env.fdtype)
    returns = torch.randn(paths, 2, 1, 2, dtype=env.fdtype) * 0.1
    costs = torch.full_like(returns, 1.0e-4)
    loss, values = _model_loss(
        model,
        state,
        returns,
        costs,
        torch.ones(paths, dtype=env.fdtype),
        torch.tensor((1.0, 8.0, 64.0), dtype=env.fdtype),
        balance_coefficient=0.0,
        objective="downside2",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(values["objective"] <= 0.0)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value).item() > 0 for value in gradients)


def test_bucket_downside_selects_worst_weighted_bucket_and_has_gradient() -> None:
    pnl = torch.tensor(
        [[[-0.1]], [[0.2]], [[-0.3]], [[0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weight = torch.tensor([1.0, 3.0, 2.0, 2.0], dtype=torch.float64)
    bucket = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    loss, worst, by_bucket = _bucket_downside_loss(
        pnl,
        weight,
        bucket,
        buckets=2,
        gain_tiebreak=0.0,
    )
    assert by_bucket[:, 0, 0].tolist() == pytest.approx([0.0025, 0.045])
    assert worst.item() == pytest.approx(0.045)
    assert loss.item() == pytest.approx(0.045)
    loss.backward()
    assert pnl.grad[2].item() == pytest.approx(-0.3)
    assert torch.count_nonzero(pnl.grad).item() == 1


def test_unbounded_payoff_cone_maximum_is_strict_json_compatible() -> None:
    assert _finite_maximum_or_none(torch.tensor([-torch.inf]).numpy()) is None
    assert _finite_maximum_or_none(
        torch.tensor([-torch.inf, -0.2, 0.1]).numpy()
    ) == pytest.approx(0.1)


def test_bucket_pnl_audit_reports_worst_physical_bucket_statistics() -> None:
    pnl = np.array([[[-0.2]], [[0.4]], [[-0.1]], [[0.3]]])
    weight = np.array([1.0, 3.0, 2.0, 2.0])
    assignment = np.array([0, 0, 1, 1])
    report = _bucket_pnl_audit(pnl, weight, assignment, buckets=2)
    assert report["paths_per_bucket_minimum"] == 2
    assert report["paths_per_bucket_maximum"] == 2
    assert report["minimum_bucket_P_mean_pnl_by_notional_and_time"][0][0] == (
        pytest.approx(0.1)
    )
    assert report[
        "maximum_bucket_P_squared_downside_by_notional_and_time"
    ][0][0] == pytest.approx(0.01)
    assert report[
        "maximum_bucket_P_loss_probability_by_notional_and_time"
    ][0][0] == pytest.approx(0.5)


def test_bucket_assignment_balances_weight_within_component() -> None:
    component = np.array([0] * 8 + [1] * 8)
    weight = np.array([10, 9, 8, 7, 6, 5, 4, 3] * 2, dtype=np.float64)
    assignment = _balanced_component_bucket_assignment(
        component, weight, buckets=2, seed=17
    )
    for value in (0, 1):
        mass = [
            weight[(component == value) & (assignment == bucket)].sum()
            for bucket in range(2)
        ]
        assert mass[0] == pytest.approx(mass[1])


def test_parallel_feedforward_has_independent_unit_gross_notional_heads() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = ParallelFeedForwardDirections(
        env,
        config=_parallel_config("feedforward"),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    state = torch.randn(7, 2, dtype=env.fdtype)
    action, logits = model(state)
    assert action.shape == (7, 3, 2, 2, 1, 2)
    assert logits.shape == (7, 3, 4)
    assert torch.allclose(
        action.sum(dim=(-3, -2, -1)),
        torch.ones((7, 3, 2), dtype=env.fdtype),
    )
    assert len(model.conditional) == 3
    first_parameters = {id(value) for value in model.conditional[0].parameters()}
    second_parameters = {id(value) for value in model.conditional[1].parameters()}
    assert first_parameters.isdisjoint(second_parameters)


def test_parallel_radial_feedforward_starts_at_zero_with_local_gradient() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = ParallelRadialFeedForwardDirections(
        env,
        config=_parallel_config("radial-feedforward"),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    state = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]], dtype=env.fdtype
    )
    action, raw = model(state)
    assert action.shape == (2, 3, 2, 2, 1, 2)
    assert raw.shape == (2, 3, 2)
    assert torch.count_nonzero(action) == 0
    raw.retain_grad()
    signed = action[:, :, 1, 0] - action[:, :, 1, 1]
    signed.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad) > 0


def test_parallel_kernel_has_independent_feasible_local_direction_heads() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = _parallel_config("kernel")
    model = ParallelKernelDirections(
        env,
        config=config,
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
        centers=torch.randn(config.kernels, 2, dtype=env.fdtype),
        inverse_bandwidth=torch.ones(config.kernels, dtype=env.fdtype),
    )
    state = torch.randn(7, 2, dtype=env.fdtype)
    action, kernel_weight = model(state)
    assert action.shape == (7, 3, 2, 2, 1, 2)
    assert kernel_weight.shape == (7, 3, config.kernels)
    assert torch.all(action >= 0.0)
    gross = action.sum(dim=(-3, -2, -1))
    assert torch.all(gross <= 1.0)
    assert torch.allclose(gross, torch.full_like(gross, 0.05))
    assert torch.allclose(
        kernel_weight.sum(dim=-1), torch.ones((7, 3), dtype=env.fdtype)
    )

    returns = torch.randn(7, 2, 1, 2, dtype=env.fdtype)
    costs = torch.full_like(returns, 1.0e-4)
    loss, _ = _model_loss(
        model,
        state,
        returns,
        costs,
        torch.ones(7, dtype=env.fdtype),
        torch.tensor(config.notionals, dtype=env.fdtype),
        balance_coefficient=0.0,
        objective="bucket-downside2",
        bucket_index=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        buckets=2,
        gain_tiebreak=1.0e-8,
    )
    loss.backward()
    assert model.template_action.grad is not None
    assert torch.isfinite(model.template_action.grad).all()
    assert torch.count_nonzero(model.template_action.grad).item() > 0
    assert model.centers.grad is not None
    assert model.log_inverse_bandwidth.grad is not None
    assert torch.isfinite(model.centers.grad).all()
    assert torch.isfinite(model.log_inverse_bandwidth.grad).all()
    assert torch.count_nonzero(model.centers.grad).item() > 0
    assert torch.count_nonzero(model.log_inverse_bandwidth.grad).item() > 0


def test_l1_ball_projection_is_euclidean_and_preserves_interior_rows() -> None:
    value = torch.tensor(
        [[0.2, 0.3, -0.4], [1.0, 1.0, 0.0]], dtype=torch.float64
    )
    project_rows_to_l1_ball_(value)
    assert torch.allclose(
        value[0], torch.tensor([0.2, 0.3, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(
        value[1], torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    )


def test_gated_feedforward_can_hold_nearly_zero_and_retains_gate_gradient() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = GatedParallelFeedForwardDirections(
        env,
        config=_parallel_config("gated-feedforward"),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    state = torch.randn(7, 2, dtype=env.fdtype)
    action, raw = model(state)
    gross = action.sum(dim=(-3, -2, -1))
    assert raw.shape == (7, 3, 5)
    assert torch.allclose(gross[:, :, 0], torch.ones_like(gross[:, :, 0]))
    assert torch.allclose(
        gross[:, :, 1],
        torch.full_like(gross[:, :, 1], torch.sigmoid(torch.tensor(-3.0)).item()),
    )
    gross[:, :, 1].sum().backward()
    for network in model.gate:
        output = network.In[-1] if network.In else network.I0
        assert output.bias is not None and output.bias.grad is not None
        assert output.bias.grad.item() > 0.0


def test_learned_partition_has_no_prespecified_state_centres_and_trains_router() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = LearnedPartitionDirections(
        env,
        config=_parallel_config("partition"),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype),
    )
    assert not hasattr(model, "centers")
    paths = 13
    state = torch.randn(paths, 2, dtype=env.fdtype)
    action, responsibility, logits = model(state)
    assert action.shape == (paths, 3, 2, 2, 1, 2)
    assert responsibility.shape == logits.shape == (paths, 3, 3)
    assert torch.allclose(
        responsibility.sum(dim=-1), torch.ones((paths, 3), dtype=env.fdtype)
    )
    assert torch.allclose(
        model.expert_actions().sum(dim=(-3, -2, -1)),
        torch.ones((3, 3), dtype=env.fdtype),
    )

    returns = torch.randn(paths, 2, 1, 2, dtype=env.fdtype) * 0.1
    costs = torch.full_like(returns, 1.0e-4)
    weight = torch.linspace(0.5, 1.5, paths, dtype=env.fdtype)
    loss, values = _model_loss(
        model,
        state,
        returns,
        costs,
        weight,
        torch.tensor((1.0, 8.0, 64.0), dtype=env.fdtype),
        balance_coefficient=1.0,
    )
    loss.backward()
    assert values["partition_mass"].shape == (3, 3)
    assert model.expert_action.grad is not None
    assert torch.count_nonzero(model.expert_action.grad).item() > 0
    router_gradients = [
        parameter.grad
        for network in model.router
        for parameter in network.parameters()
        if parameter.grad is not None
    ]
    assert router_gradients
    assert any(torch.count_nonzero(value).item() > 0 for value in router_gradients)
