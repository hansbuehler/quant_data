from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from quantdata.dysanos.flex.empirical_risk_neutral import (
    fit_empirical_risk_neutral_density,
    fit_empirical_risk_neutral_density_torch,
)
from quantdata.dysanos.flex.run_empirical_risk_neutral import (
    load_return_matrix,
    run,
)


def test_identity_density_when_empirical_moments_are_inside_cost_bands() -> None:
    returns = np.asarray(
        (
            (-0.02, 0.01),
            (0.01, -0.02),
            (0.02, 0.00),
            (-0.01, 0.01),
        ),
        dtype=np.float64,
    )
    fit = fit_empirical_risk_neutral_density(
        returns,
        np.asarray((0.001, 0.001)),
        chunk_size=2,
    )

    assert fit.optimizer_success
    assert fit.action == pytest.approx(np.zeros(2), abs=1.0e-12)
    assert fit.weights == pytest.approx(np.ones(4), abs=1.0e-12)
    assert fit.weighted_mean == pytest.approx(np.mean(returns, axis=0), abs=1.0e-12)
    assert fit.primal_relative_entropy == pytest.approx(0.0, abs=1.0e-12)
    assert fit.effective_sample_size == pytest.approx(4.0)


def test_one_dimensional_fit_reaches_active_positive_cost_boundary() -> None:
    returns = np.asarray((-0.04, -0.01, 0.02, 0.05, 0.08), dtype=np.float64)[:, None]
    cost = np.asarray((0.0075,))
    fit = fit_empirical_risk_neutral_density(
        returns,
        cost,
        chunk_size=2,
        moment_tolerance=2.0e-9,
    )

    assert fit.optimizer_success
    assert fit.action[0] > 0.0
    assert fit.weighted_mean[0] == pytest.approx(cost[0], abs=2.0e-9)
    assert np.mean(fit.weights) == pytest.approx(1.0, abs=1.0e-14)
    assert fit.primal_relative_entropy >= 0.0
    assert fit.primal_dual_gap == pytest.approx(0.0, abs=2.0e-9)
    assert fit.effective_sample_size < len(returns)


def test_multivariate_fit_handles_opposite_active_signs_and_three_dimensions() -> None:
    one = np.asarray((-0.05, -0.02, 0.01, 0.04, 0.07, 0.10), dtype=np.float64)
    flat = np.column_stack((one, -one, np.roll(one, 1), -np.roll(one, 1)))
    returns = flat.reshape(len(flat), 2, 2)
    cost = np.full(4, 0.005)
    fit = fit_empirical_risk_neutral_density(
        returns,
        cost,
        chunk_size=3,
        moment_tolerance=2.0e-8,
    )

    assert fit.optimizer_success
    assert np.max(np.abs(fit.weighted_mean) - cost) <= 2.0e-8
    assert fit.action[0] > 0.0
    assert fit.action[1] < 0.0
    assert fit.action.shape == (4,)
    assert fit.weights.shape == (6,)


def test_zero_cost_fit_enforces_zero_empirical_moments() -> None:
    generator = np.random.default_rng(123)
    returns = generator.normal(size=(500, 5)) * 0.03
    returns += np.asarray((0.010, -0.008, 0.006, -0.004, 0.002))

    fit = fit_empirical_risk_neutral_density(
        returns,
        np.zeros(5),
        chunk_size=len(returns),
        moment_tolerance=1.0e-10,
    )

    assert fit.optimizer_success
    assert fit.maximum_kkt_residual <= 1.0e-10
    assert fit.weighted_mean == pytest.approx(np.zeros(5), abs=1.0e-10)
    assert np.mean(fit.weights) == pytest.approx(1.0, abs=1.0e-14)
    assert fit.active_set_iterations == 0
    assert "unconstrained zero-cost" in fit.optimizer_message


def test_torch_full_population_solver_matches_streamed_cpu_solver() -> None:
    generator = np.random.default_rng(17)
    returns = generator.normal(size=(400, 8)) * 0.03
    returns += np.asarray((0.01, -0.008, 0.006, -0.004, 0.002, 0.0, 0.003, -0.005))
    cost = np.full(8, 0.001)

    cpu_fit = fit_empirical_risk_neutral_density(
        returns,
        cost,
        chunk_size=73,
        moment_tolerance=2.0e-8,
    )
    torch_fit = fit_empirical_risk_neutral_density_torch(
        returns,
        cost,
        device="cpu",
        upload_chunk_size=73,
        final_chunk_size=73,
        moment_tolerance=2.0e-8,
    )

    assert cpu_fit.optimizer_success
    assert torch_fit.optimizer_success
    assert torch_fit.action == pytest.approx(cpu_fit.action, abs=5.0e-7)
    assert torch_fit.weighted_mean == pytest.approx(cpu_fit.weighted_mean, abs=2.0e-9)
    assert torch_fit.effective_sample_fraction == pytest.approx(
        cpu_fit.effective_sample_fraction, abs=2.0e-9
    )


def test_torch_zero_cost_solver_matches_streamed_cpu_solver() -> None:
    generator = np.random.default_rng(23)
    returns = generator.normal(size=(500, 6)) * 0.03
    returns += np.asarray((0.01, -0.008, 0.006, -0.004, 0.002, -0.003))
    cost = np.zeros(6)

    cpu_fit = fit_empirical_risk_neutral_density(
        returns,
        cost,
        chunk_size=79,
        moment_tolerance=1.0e-9,
    )
    torch_fit = fit_empirical_risk_neutral_density_torch(
        returns,
        cost,
        device="cpu",
        upload_chunk_size=79,
        final_chunk_size=79,
        moment_tolerance=1.0e-9,
    )

    assert cpu_fit.optimizer_success
    assert torch_fit.optimizer_success
    assert torch_fit.action == pytest.approx(cpu_fit.action, abs=1.0e-8)
    assert torch_fit.weighted_mean == pytest.approx(cpu_fit.weighted_mean, abs=1.0e-10)


def test_active_set_drops_a_misleading_empirical_mean_coordinate() -> None:
    generator = np.random.default_rng(0)
    returns = generator.normal(size=(30, 5))
    returns[:, 1] = 0.9 * returns[:, 0] + 0.2 * returns[:, 1]
    returns[:, 3] = -0.8 * returns[:, 2] + 0.3 * returns[:, 3]
    returns += np.asarray((0.30, 0.25, -0.20, 0.15, 0.05))
    cost = np.full(5, 0.05)
    assert np.mean(returns, axis=0)[2] > cost[2]

    fit = fit_empirical_risk_neutral_density(
        returns,
        cost,
        chunk_size=11,
        maximum_iterations=500,
    )

    assert fit.optimizer_success
    assert fit.active_set_iterations >= 2
    assert fit.action[2] == pytest.approx(0.0, abs=1.0e-12)
    assert abs(fit.weighted_mean[2]) < cost[2]
    assert fit.maximum_kkt_residual <= 1.0e-8


def test_solver_rejects_a_coordinate_whose_empirical_range_misses_band() -> None:
    returns = np.asarray(((0.02,), (0.03,), (0.04,)))
    with pytest.raises(ValueError, match="no feasible density"):
        fit_empirical_risk_neutral_density(returns, np.asarray((0.01,)))


def test_cli_loads_memmap_and_writes_report_and_density(tmp_path) -> None:
    base = np.asarray(
        (
            (-0.02, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02),
            (-0.01, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03),
            (0.00, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04),
            (0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05),
            (0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
            (0.03, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07),
        ),
        dtype=np.float32,
    )
    returns = np.concatenate((base, -base), axis=0)[:, None, :]
    input_path = tmp_path / "returns.npy"
    report_path = tmp_path / "report.json"
    weights_path = tmp_path / "weights.npy"
    log_weights_path = tmp_path / "log_weights.npy"
    np.save(input_path, returns)

    loaded = load_return_matrix(input_path, key="returns")
    assert isinstance(loaded, np.memmap)
    arguments = Namespace(
        returns=input_path,
        report=report_path,
        weights=weights_path,
        log_weights=log_weights_path,
        array_key="returns",
        book_expiry_days=[5, 10],
        spot_cost_bps=1.0,
        option_cost_bps=10.0,
        chunk_size=3,
        maximum_iterations=2_000,
        maximum_active_set_iterations=50,
        moment_tolerance=1.0e-8,
        optimizer_gradient_tolerance=1.0e-10,
        optimizer_function_tolerance=1.0e-15,
    )
    fit = run(arguments)

    assert fit.optimizer_success
    assert report_path.is_file()
    assert np.load(weights_path) == pytest.approx(fit.weights)
    assert np.load(log_weights_path) == pytest.approx(fit.log_weights)


def test_cli_zero_cost_report_describes_exact_moment_constraints(tmp_path) -> None:
    returns = np.asarray(
        (
            (-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04),
            (0.02, 0.01, 0.00, -0.01, -0.02, -0.03, -0.04),
        ),
        dtype=np.float32,
    )[:, None, :]
    input_path = tmp_path / "returns.npy"
    report_path = tmp_path / "report.json"
    np.save(input_path, returns)
    arguments = Namespace(
        returns=input_path,
        report=report_path,
        weights=None,
        log_weights=None,
        array_key="returns",
        book_expiry_days=[5, 10],
        spot_cost_bps=0.0,
        option_cost_bps=0.0,
        chunk_size=2,
        maximum_iterations=2_000,
        maximum_active_set_iterations=50,
        moment_tolerance=1.0e-8,
        optimizer_gradient_tolerance=1.0e-10,
        optimizer_function_tolerance=1.0e-15,
    )

    fit = run(arguments)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert fit.optimizer_success
    assert report["definition"]["transaction_cost_band"] == "none"
    assert report["definition"]["primal"].endswith(
        "mean(w[:,None]*returns,axis=0)=0"
    )
    assert "cost_band" not in report["definition"]["signed_dual"]
