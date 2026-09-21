"""Tests for the parallel direct-utility notional frontier."""

from __future__ import annotations

import torch

from quantdata.dysanos.flex.direct_utility_frontier import (
    DirectUtilityForward,
    DirectUtilityTowerPolicy,
    direct_utility_loss,
    normalized_powerexp_marginal,
    powerexp_utility,
)
from quantdata.dysanos.flex.run_direct_utility_frontier import (
    PhysicalTowerDataset,
    _beta_whitened_dataset,
    _run_residual_adversary,
)


def test_normalized_powerexp_marginal_is_stable_and_has_unit_mass() -> None:
    pnl = torch.tensor(
        [[-1.0e6, -10.0, 0.0, 10.0, 1.0e6], [3.0, -2.0, 1.0, -4.0, 0.5]],
        dtype=torch.float64,
    )
    weights = normalized_powerexp_marginal(pnl)

    assert bool(torch.all(torch.isfinite(weights) & (weights >= 0.0)))
    torch.testing.assert_close(
        torch.mean(weights, dim=1),
        torch.ones(2, dtype=torch.float64),
    )


def test_detached_marginal_surrogate_has_exact_normalized_utility_gradient() -> None:
    notionals = torch.tensor([2.0, 8.0], dtype=torch.float64)
    normalized_pnl = torch.tensor(
        [[-0.3, 0.2, 0.5, -0.1], [0.1, -0.4, 0.3, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    leveraged = notionals[:, None] * normalized_pnl
    forward = DirectUtilityForward(
        normalized_pnl=normalized_pnl,
        leveraged_pnl=leveraged,
        action_penalty_by_head=torch.zeros(2, dtype=torch.float64),
        action_l1_by_time=(),
    )
    stable = direct_utility_loss(forward, action_penalty=0.0)
    stable_gradient = torch.autograd.grad(stable.loss, normalized_pnl)[0]

    direct_by_head = torch.mean(powerexp_utility(leveraged), dim=1)
    direct_gradient = torch.stack(
        [
            torch.autograd.grad(
                direct_by_head[head],
                normalized_pnl,
                retain_graph=True,
            )[0][head]
            for head in range(2)
        ]
    )
    marginal_mass = torch.mean(
        torch.exp(
            torch.where(
                leveraged > 0.0,
                -leveraged,
                torch.log1p(-torch.clamp_max(leveraged, 0.0)),
            )
        ),
        dim=1,
    )
    expected_loss_gradient = -direct_gradient / (2.0 * notionals[:, None] * marginal_mass[:, None])
    torch.testing.assert_close(stable_gradient, expected_loss_gradient)


def test_parallel_policy_caps_every_head_and_time_action() -> None:
    dtype = torch.float64
    feature_sizes = (7, 6, 5, 4)
    output_sizes = (8, 6, 4, 2)
    policy = DirectUtilityTowerPolicy(
        notionals=(2.0, 4.0, 8.0),
        feature_means=[torch.zeros(size, dtype=dtype) for size in feature_sizes],
        feature_scales=[torch.ones(size, dtype=dtype) for size in feature_sizes],
        output_sizes=output_sizes,
        hidden_width=5,
        hidden_layers=2,
        seed=17,
    )
    generator = torch.Generator().manual_seed(19)
    features = [torch.randn(11, size, dtype=dtype, generator=generator) for size in feature_sizes]
    returns = [torch.randn(11, size, dtype=dtype, generator=generator) for size in output_sizes]
    result = policy(features, returns)

    assert result.normalized_pnl.shape == (3, 11)
    assert result.leveraged_pnl.shape == (3, 11)
    assert result.action_penalty_by_head.shape == (3,)
    assert len(result.action_l1_by_time) == 4
    notionals = torch.tensor([2.0, 4.0, 8.0], dtype=dtype)[:, None]
    assert all(bool(torch.all(value <= notionals + 1.0e-12)) for value in result.action_l1_by_time)
    torch.testing.assert_close(
        result.leveraged_pnl,
        notionals * result.normalized_pnl,
    )


def test_residual_adversary_leaves_base_measure_policy_unchanged() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(31)
    feature_sizes = (7, 6, 5, 4)
    output_sizes = (8, 6, 4, 2)
    feature_means = [torch.zeros(size, dtype=dtype) for size in feature_sizes]
    feature_scales = [torch.ones(size, dtype=dtype) for size in feature_sizes]
    dataset = PhysicalTowerDataset(
        features_by_time=tuple(
            torch.randn(20, size, dtype=dtype, generator=generator) for size in feature_sizes
        ),
        returns_by_time=tuple(
            0.01 * torch.randn(20, size, dtype=dtype, generator=generator) for size in output_sizes
        ),
        spot_returns=torch.zeros(20, 4, dtype=dtype),
        paths_per_independent_block=2,
        diagnostics={},
    )
    base_policy = DirectUtilityTowerPolicy(
        notionals=(16.0,),
        feature_means=feature_means,
        feature_scales=feature_scales,
        output_sizes=output_sizes,
        hidden_width=5,
        hidden_layers=1,
        seed=37,
    )
    base_before = {name: value.detach().clone() for name, value in base_policy.state_dict().items()}

    report, _, _, combined_weights = _run_residual_adversary(
        dataset=dataset,
        base_q_weights=torch.ones(dataset.num_paths, dtype=dtype),
        base_q_label="unit test",
        feature_means=feature_means,
        feature_scales=feature_scales,
        output_sizes=output_sizes,
        notionals=(8.0,),
        hidden_width=5,
        hidden_layers=1,
        train_hidden_layers=True,
        optimizer_kind="direct-lbfgs",
        lbfgs_iterations_per_epoch=2,
        seed=41,
        epochs=2,
        min_epochs=0,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-7,
        lr_patience=0,
        lr_decay=0.5,
        gradient_rms_tolerance=0.0,
        utility_ceiling=None,
        action_penalty=0.0,
        gradient_clip_norm=100.0,
        monitoring_interval=2,
        bias_natural_gradient_interval=0,
        bias_natural_gradient_ridge=1.0e-2,
        final_bias_root_steps=1,
        final_bias_root_ridge=1.0e-4,
        batch_size=7,
    )

    assert report["base_q_weights_detached"] is True
    assert report["path_scope"] == "identical fitted path set"
    torch.testing.assert_close(
        torch.mean(combined_weights, dim=1),
        torch.ones(1, dtype=dtype),
    )
    for name, value in base_policy.state_dict().items():
        torch.testing.assert_close(value, base_before[name])


def test_fixed_hidden_policy_only_trains_linear_output_maps() -> None:
    policy = DirectUtilityTowerPolicy(
        notionals=(8.0,),
        feature_means=[torch.zeros(size) for size in (7, 6, 5, 4)],
        feature_scales=[torch.ones(size) for size in (7, 6, 5, 4)],
        output_sizes=(8, 6, 4, 2),
        hidden_width=5,
        hidden_layers=2,
        seed=53,
        train_hidden_layers=False,
    )

    for time_policy in policy.policies:
        assert all(not parameter.requires_grad for parameter in time_policy.weights[:-1])
        assert all(not parameter.requires_grad for parameter in time_policy.biases[:-1])
        assert time_policy.weights[-1].requires_grad
        assert time_policy.biases[-1].requires_grad


def test_beta_whitening_is_full_rank_and_preserves_raw_mean_constraints() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(43)
    paths = 40
    feature_sizes = (7, 6, 5, 4)
    output_sizes = (8, 6, 4, 2)
    spot = torch.randn(paths, 4, dtype=dtype, generator=generator)
    raw_returns = []
    for time_index, size in enumerate(output_sizes):
        loadings = torch.linspace(-2.0, 2.0, size, dtype=dtype)
        residual = 0.2 * torch.randn(
            paths,
            size,
            dtype=dtype,
            generator=generator,
        )
        raw_returns.append(spot[:, time_index, None] * loadings[None] + residual)
    dataset = PhysicalTowerDataset(
        features_by_time=tuple(
            torch.randn(paths, size, dtype=dtype, generator=generator) for size in feature_sizes
        ),
        returns_by_time=tuple(raw_returns),
        spot_returns=spot,
        paths_per_independent_block=2,
        diagnostics={},
    )

    transformed, whitening = _beta_whitened_dataset(
        dataset,
        maximum_condition_number=100.0,
        noise_seed=47,
    )

    for time_index, (raw, factor) in enumerate(
        zip(dataset.returns_by_time, transformed.returns_by_time, strict=True)
    ):
        beta = whitening.beta_by_time[time_index]
        eigenvectors = whitening.eigenvectors_by_time[time_index]
        inverse_sqrt = whitening.inverse_sqrt_eigenvalues_by_time[time_index]
        # The added antithetic epsilon has exactly zero full-sample mean, so
        # transformed zero-mean constraints still map back to the raw means.
        factor_mean = torch.mean(factor[:, 1:], dim=0)
        reconstructed_residual_mean = (factor_mean / inverse_sqrt) @ eigenvectors.T
        raw_residual_mean = torch.mean(
            raw - spot[:, time_index, None] * beta[None],
            dim=0,
        )
        torch.testing.assert_close(reconstructed_residual_mean, raw_residual_mean)
        assert factor.shape == (paths, raw.shape[1] + 1)
