from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.dysanos.bates_crc.gambara_teichmann_audit import CrcAuditConfig
from quantdata.dysanos.bates_crc.gambara_teichmann_crc import (
    FiveBucketJumps,
    GeneralizedBatesState,
    RegridResult,
    bates_antithetic_step,
    paired_maturity_boundaries,
    price_calls_generalized_bates,
    regrid_jump_parameters,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_inverse_data import (
    InverseDatasetConfig,
    exact_reset_price_gate,
    generate_surface_shard,
    shard_split,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_inverse_train import (
    CrcInverseNetworkConfig,
    CrcInverseNetworks,
    CrcInverseTrainConfig,
    _checkpoint_payload,
    apply_reset_innovations,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_inverse_validate import (
    verify_training_provenance,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_population import (
    CrcPopulationConfig,
    _black_calls,
    _implied_volatilities,
    _pair_randomness,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_study import (
    apply_reset_innovation,
    estimate_reset_innovation_law,
    select_ten_maturity_nodes,
)
from quantdata.dysanos.bates_crc.gambara_teichmann_torch import (
    price_calls_generalized_bates_batched,
)
from quantdata.dysanos.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.dysanos.bates_crc.real_data_study.fast_bates_pricer import (
    price_calls_ordinary_bates,
)
from quantdata.dysanos.no_arbitrage_audit import (
    RestrictedPopulation,
    crossfit_conditional_audit,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _state() -> GeneralizedBatesState:
    return GeneralizedBatesState(
        log_spot=0.0,
        variance=0.04,
        heston=HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.60),
        jumps=FiveBucketJumps(
            intensity=0.35,
            boundaries=(0.04, 0.08, 0.12, 0.16),
            means=(-0.06, -0.05, -0.04, -0.03, -0.02),
            stds=(0.10, 0.11, 0.12, 0.13, 0.14),
        ),
    )


def test_inverse_training_provenance_binds_dataset_and_target(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    target = tmp_path / "target-fit.json"
    run = tmp_path / "network"
    dataset.mkdir()
    run.mkdir()
    target.write_text("{}", encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    (run / "training-report.json").write_text(
        json.dumps(
            {
                "schema": "gambara-teichmann-crc-inverse-training-report-v1",
                "status": "complete",
                "dataset_directory": str(dataset.resolve()),
                "target_fit_report": str(target.resolve()),
            }
        ),
        encoding="utf-8",
    )

    report = verify_training_provenance(
        checkpoint,
        dataset_directory=dataset,
        target_fit_report=target,
    )
    assert report["status"] == "complete"
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_training_provenance(
            checkpoint,
            dataset_directory=tmp_path / "wrong-dataset",
            target_fit_report=target,
        )


def test_checkpoint_payload_accepts_string_cuda_device(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    cuda_state = [torch.tensor((1, 2, 3), dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: cuda_state)
    stateful = SimpleNamespace(state_dict=lambda: {})
    model = SimpleNamespace(device="cuda:0", state_dict=lambda: {})

    payload = _checkpoint_payload(
        model,
        model_config=CrcInverseNetworkConfig(label="checkpoint-regression"),
        train_config=CrcInverseTrainConfig(forward_epochs=1, inverse_epochs=1),
        phase="forward",
        epoch=1,
        optimizer=stateful,
        scheduler=stateful,
        cpu_generator=torch.Generator(device="cpu"),
    )

    assert payload["cuda_rng_state"] is cuda_state


def test_bucket_durations_partition_maturity_and_compensate_martingale() -> None:
    jumps = _state().jumps
    for tau in (0.01, 0.07, 0.20, 0.50):
        assert np.sum(jumps.bucket_durations(tau)) == pytest.approx(tau)
        exponent = jumps.integrated_exponent(np.asarray((-1j,)), tau)
        assert exponent[0] == pytest.approx(0.0j, abs=2.0e-15)


def test_pair_boundaries_group_ten_adjacent_maturities() -> None:
    maturities = np.arange(1.0, 11.0) / 252.0
    assert paired_maturity_boundaries(maturities) == pytest.approx(
        (2.5 / 252.0, 4.5 / 252.0, 6.5 / 252.0, 8.5 / 252.0)
    )


def test_constant_bucket_law_matches_ordinary_bates() -> None:
    state = _state()
    common_mean, common_std = -0.04, 0.12
    jumps = FiveBucketJumps(
        intensity=state.jumps.intensity,
        boundaries=state.jumps.boundaries,
        means=(common_mean,) * 5,
        stds=(common_std,) * 5,
    )
    generalized = GeneralizedBatesState(
        log_spot=0.0,
        variance=state.variance,
        heston=state.heston,
        jumps=jumps,
    )
    strikes = np.asarray((0.85, 0.95, 1.00, 1.10, 1.20))
    expiries = np.asarray((0.02, 0.06, 0.10, 0.18, 0.30))
    actual = price_calls_generalized_bates(
        generalized, strikes, expiries, n_quad=240, u_max=600.0
    )
    expected = price_calls_ordinary_bates(
        state.heston,
        GaussianJumpParams(lambda_J=jumps.intensity, m_J=common_mean, delta_J=common_std),
        state.variance,
        strikes,
        expiries,
        n_quad=240,
        u_max=600.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_batched_torch_pricer_matches_scalar_reference() -> None:
    torch = pytest.importorskip("torch")
    states = (_state(), _state())
    log_spot = torch.tensor((0.0, np.log(1.07)), dtype=torch.float64)
    variance = torch.tensor((0.04, 0.025), dtype=torch.float64)
    theta = torch.tensor((0.04, 0.03), dtype=torch.float64)
    xi = torch.tensor((0.30, 0.25), dtype=torch.float64)
    rho = torch.tensor((-0.60, -0.35), dtype=torch.float64)
    means = torch.tensor((states[0].jumps.means, states[1].jumps.means), dtype=torch.float64)
    stds = torch.tensor((states[0].jumps.stds, states[1].jumps.stds), dtype=torch.float64)
    strikes = torch.tensor(((0.9, 1.0, 1.1), (0.9, 1.0, 1.1)), dtype=torch.float64)
    actual = price_calls_generalized_bates_batched(
        log_spot=log_spot,
        variance=variance,
        theta=theta,
        xi=xi,
        rho=rho,
        jump_means=means,
        jump_stds=stds,
        strikes=strikes,
        expiry=0.12,
        kappa=2.0,
        intensity=0.35,
        boundaries=states[0].jumps.boundaries,
        n_quad=240,
        u_max=600.0,
    ).numpy()
    expected = []
    for index in range(2):
        state = GeneralizedBatesState(
            log_spot=float(log_spot[index]),
            variance=float(variance[index]),
            heston=HestonParams(
                kappa=2.0,
                theta=float(theta[index]),
                xi=float(xi[index]),
                rho=float(rho[index]),
            ),
            jumps=states[index].jumps,
        )
        expected.append(
            price_calls_generalized_bates(
                state,
                strikes[index].numpy(),
                np.full(3, 0.12),
                n_quad=240,
                u_max=600.0,
            )
        )
    np.testing.assert_allclose(actual, np.stack(expected), rtol=5.0e-11, atol=5.0e-11)
    actual_float32 = price_calls_generalized_bates_batched(
        log_spot=log_spot.float(),
        variance=variance.float(),
        theta=theta.float(),
        xi=xi.float(),
        rho=rho.float(),
        jump_means=means.float(),
        jump_stds=stds.float(),
        strikes=strikes.float(),
        expiry=0.12,
        kappa=2.0,
        intensity=0.35,
        boundaries=states[0].jumps.boundaries,
        n_quad=240,
        u_max=600.0,
    ).numpy()
    np.testing.assert_allclose(actual_float32, np.stack(expected), rtol=2.0e-5, atol=2.0e-7)


def test_crc_population_pair_geometry_and_black_inversion() -> None:
    torch = pytest.importorskip("torch")
    config = CrcPopulationConfig(path_count=4, batch_size=4)
    assert config.n_quad == 240 and config.u_max == 600.0
    diffusion, counts, jump_normals, reset_normals = _pair_randomness(
        config, batch_index=3, intensity=0.35
    )
    np.testing.assert_array_equal(diffusion[:, :, 0], -diffusion[:, :, 1])
    np.testing.assert_array_equal(diffusion[:, :, 2], -diffusion[:, :, 3])
    np.testing.assert_array_equal(jump_normals[:, :, 0], -jump_normals[:, :, 1])
    np.testing.assert_array_equal(reset_normals[:, 0], -reset_normals[:, 1])
    assert counts.shape == (20, 8, 4)

    moneyness = torch.tensor(((0.9, 1.0, 1.1),), dtype=torch.float64)
    expiry = torch.full_like(moneyness, 20.0 / 252.0)
    volatility = torch.tensor(((0.18, 0.20, 0.22),), dtype=torch.float64)
    prices = _black_calls(moneyness, expiry, volatility)
    recovered, clipped = _implied_volatilities(
        prices,
        moneyness,
        20.0 / 252.0,
        iterations=48,
    )
    assert clipped == 0
    torch.testing.assert_close(recovered, volatility, rtol=1.0e-10, atol=1.0e-10)


def test_crc_config_runs_shared_pair_preserving_conditional_audit(tmp_path) -> None:
    population = RestrictedPopulation(
        features=np.asarray(([[0.0]], [[0.0]], [[1.0]], [[1.0]]), dtype=np.float32),
        returns=np.asarray(([[0.01]], [[-0.01]], [[0.02]], [[-0.02]]), dtype=np.float64),
        active_mask=np.ones((1, 1), dtype=bool),
        pair_ids=np.asarray((0, 0, 1, 1), dtype=np.int64),
    )
    result = crossfit_conditional_audit(
        population,
        objective="squared",
        config=CrcAuditConfig(path_count=4, policy_epochs=1, policy_batch_size=2),
        checkpoint_directory=tmp_path,
        device="cpu",
    )
    assert result.path_count == 4
    assert result.pair_count == 2


def test_step_preserves_adjacent_antithetic_pair_geometry() -> None:
    normals = np.asarray(
        (
            ((0.5, -0.25), (-0.5, 0.25)),
            ((-1.0, 0.75), (1.0, -0.75)),
        )
    )
    log_spot, variance = bates_antithetic_step(_state(), dt=1.0 / 252.0, normal_pairs=normals)
    assert log_spot.shape == (2, 2)
    assert variance.shape == (2, 2)
    # With no jumps, pair averages retain exactly the common Euler drift.
    expected_log_mean = (-0.5 * _state().variance - _state().jumps.intensity * (
        np.exp(_state().jumps.means[0] + 0.5 * _state().jumps.stds[0] ** 2) - 1.0
    )) / 252.0
    np.testing.assert_allclose(np.mean(log_spot, axis=1), expected_log_mean, atol=1.0e-15)


def test_regrid_smoke_preserves_fixed_contract_prices() -> None:
    state = _state()
    maturities = np.asarray((0.02, 0.035, 0.055, 0.075, 0.095, 0.115, 0.135, 0.155, 0.18, 0.22))
    strikes = np.asarray((0.90, 1.10) * 5)
    result = regrid_jump_parameters(
        state,
        HestonParams(kappa=2.0, theta=0.04002, xi=0.30005, rho=-0.5999),
        strikes,
        maturities,
        heldout_strikes=np.asarray((0.95, 1.05) * 5),
        heldout_expiries=maturities,
        tolerance=2.0e-6,
        n_quad=100,
        u_max=160.0,
    )
    assert result.accepted, result
    assert result.require_accepted() == result.jumps
    assert result.jumps.intensity == state.jumps.intensity
    assert result.max_abs_price_error <= 2.0e-6
    assert result.heldout_max_abs_price_error <= 2.0e-6


def test_failed_regrid_cannot_be_claimed_as_crc_reset() -> None:
    failed = RegridResult(
        jumps=_state().jumps,
        prices_before=np.asarray((0.1,)),
        prices_after=np.asarray((0.2,)),
        heldout_prices_before=np.asarray((0.1,)),
        heldout_prices_after=np.asarray((0.2,)),
        calibration_max_abs_price_error=0.1,
        heldout_max_abs_price_error=0.1,
        max_abs_price_error=0.1,
        rms_price_error=0.1,
        tolerance=1.0e-6,
        function_evaluations=1,
        accepted=False,
    )
    with pytest.raises(RuntimeError, match="did not preserve"):
        failed.require_accepted()


def test_reset_law_is_strictly_pre_target_and_keeps_kappa_fixed() -> None:
    dates = np.asarray(("2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28"))
    parameters = np.asarray(
        ((0.040, 0.300, -0.60), (0.041, 0.305, -0.59), (0.039, 0.302, -0.61), (0.040, 0.301, -0.60))
    )
    law = estimate_reset_innovation_law(dates, parameters, target_date="2025-08-29")
    updated = apply_reset_innovation(_state().heston, np.asarray((0.01, -0.02, 0.01)), law)
    assert updated.kappa == _state().heston.kappa
    assert 2.0 * updated.kappa * updated.theta > updated.xi**2

    with pytest.raises(ValueError, match="strictly before"):
        estimate_reset_innovation_law(
            np.append(dates, "2025-08-29"),
            np.vstack((parameters, parameters[-1])),
            target_date="2025-08-29",
        )


def test_target_maturity_selection_is_deterministic() -> None:
    expiries = np.arange(1.0, 25.0) / 252.0
    first = select_ten_maturity_nodes(expiries)
    second = select_ten_maturity_nodes(expiries[::-1])
    np.testing.assert_array_equal(first, second)
    assert first.size == 10


def test_inverse_shards_are_deterministic_and_split_without_leakage() -> None:
    config = InverseDatasetConfig(samples_per_shard=2, n_quad=60, u_max=140.0)
    first = generate_surface_shard(config, 3)
    second = generate_surface_shard(config, 3)
    np.testing.assert_array_equal(first["parameters"], second["parameters"])
    np.testing.assert_array_equal(first["implied_volatilities"], second["implied_volatilities"])
    assert first["implied_volatilities"].shape == (2, 130)
    assert int(first["quadrature_nodes"]) == 60
    assert float(first["quadrature_u_max"]) == 140.0
    assert shard_split(8, 10) == "train"
    assert shard_split(9, 10) == "heldout"


def test_exact_reset_gate_checks_independent_heldout_contracts() -> None:
    state = _state()
    calibration_expiries = np.asarray((0.04, 0.08, 0.12, 0.16, 0.20))
    heldout_expiries = calibration_expiries + 0.005
    passed = exact_reset_price_gate(
        state,
        state,
        calibration_strikes=np.asarray((0.9, 0.95, 1.0, 1.05, 1.1)),
        calibration_expiries=calibration_expiries,
        heldout_strikes=np.asarray((0.925, 0.975, 1.025, 1.075, 1.125)),
        heldout_expiries=heldout_expiries,
        tolerance=1.0e-10,
        n_quad=80,
        u_max=160.0,
    )
    assert passed["accepted"] is True
    assert passed["heldout_max_abs_price_error"] == 0.0


def test_inverse_network_geometry_bounds_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    config = CrcInverseNetworkConfig(
        label="test",
        forward_network=FeedForwardConfig(activation="Softplus", num_depth=2, num_width=16),
        inverse_network=FeedForwardConfig(activation="Softplus", num_depth=2, num_width=16),
    )
    model = CrcInverseNetworks(
        BaseEnvironment(device="cpu", precision=32),
        config=config,
        parameter_mean=torch.zeros(14),
        parameter_scale=torch.ones(14),
        surface_mean=torch.zeros(130),
        surface_scale=torch.ones(130),
        jump_lower=torch.tensor((-0.3,) * 5 + (0.005,) * 5),
        jump_upper=torch.tensor((0.15,) * 5 + (0.30,) * 5),
    )
    parameters = torch.zeros((3, 14), requires_grad=True)
    surfaces = model.surface(parameters)
    jumps = model.jump_parameters(parameters[:, :4], surfaces)
    assert surfaces.shape == (3, 130)
    assert jumps.shape == (3, 10)
    assert torch.all(jumps[:, :5] >= -0.3) and torch.all(jumps[:, :5] <= 0.15)
    assert torch.all(jumps[:, 5:] >= 0.005) and torch.all(jumps[:, 5:] <= 0.30)
    (surfaces.square().mean() + jumps.square().mean()).backward()
    assert parameters.grad is not None and torch.isfinite(parameters.grad).all()


def test_batched_reset_innovations_keep_variance_kappa_domain_and_feller() -> None:
    torch = pytest.importorskip("torch")
    state = torch.tensor(((0.04, 0.03, 0.40, -0.8), (0.02, 0.02, 0.30, -0.5)))
    innovation = torch.tensor(((0.2, 0.2, 0.1), (-0.2, -0.2, -0.1)))
    updated = apply_reset_innovations(
        state,
        innovation,
        kappa=2.0,
        relative_change_cap=0.05,
    )
    torch.testing.assert_close(updated[:, 0], state[:, 0])
    assert torch.all(updated[:, 1] >= 0.01)
    assert torch.all(updated[:, 2] >= 0.01)
    assert torch.all(torch.abs(updated[:, 3]) < 1.0)
    assert torch.all(updated[:, 2].square() < 2.0 * 2.0 * updated[:, 1])
