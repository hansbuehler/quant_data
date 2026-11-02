from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2 import rolling_audit
from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower
from quantdata.gromm2.market import solve_sanos_step_numpy
from quantdata.gromm2.rolling_audit import (
    UnconditionalAuditConfig,
    antithetic_shocks,
    audit_batch_function,
    audit_worker_command,
    combine_moments,
    moment_statistics,
    paired_samples,
    run_audit,
)
from quantdata.gromm2.simulation import localize_loadings_by_density

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def small_audit_inputs() -> dict[str, object]:
    geometry = Gromm2GeometryConfig(num_inner_strikes=16)
    expiries = np.asarray([1.0, 5.0]) / 252.0
    supports = expanding_supports(0.2**2 * expiries, config=geometry)
    transports = transport_tower(supports)
    dlvsq = np.full((2, 16), 0.18**2)
    calls = np.maximum(1.0 - supports[0], 0.0)
    initial = np.diff(np.diff(calls, prepend=1.0) / np.diff(supports[0], prepend=0.0), append=0.0)
    first = solve_sanos_step_numpy(supports[0], initial, dlvsq[0] / 252.0)
    second = solve_sanos_step_numpy(supports[1], transports[0] @ first, dlvsq[1] * 4.0 / 252.0)
    return {
        "supports": supports, "transports": transports, "expiries_y": expiries,
        "dlvsq": dlvsq, "initial_densities": np.stack((first, second)),
        "loadings": np.zeros((1, 16, 2)), "projected_variances": 0.2**2 * expiries,
        "background_volatility": 0.05,
    }


def test_shocks_and_statistics_keep_antithetic_pairs_together() -> None:
    factors, background = antithetic_shocks(seed=12, completed_paths=0, paths=8, substeps=3, factors=4)
    np.testing.assert_array_equal(factors[:, ::2] + factors[:, 1::2], 0.0)
    np.testing.assert_array_equal(background[::2] + background[1::2], 0.0)
    values = np.arange(24.0).reshape(8, 3)
    actual = paired_samples(values)
    np.testing.assert_array_equal(actual, 0.5 * (values[::2] + values[1::2]))


def test_float32_saved_expiry_settles_front_without_residual_black_time(small_audit_inputs) -> None:
    inputs = dict(small_audit_inputs)
    inputs["expiries_y"] = np.asarray(inputs["expiries_y"], dtype=np.float32).astype(np.float64)
    config = UnconditionalAuditConfig(path_count=4, batch_size=4, substeps=1, nms=(-0.1, 0.0, 0.1))
    inputs["dlvsq"] = np.zeros_like(inputs["dlvsq"])
    front = np.zeros_like(inputs["initial_densities"])
    support = np.asarray(inputs["supports"])
    lower = np.searchsorted(support[0], 1.0) - 1
    weight = (1.0 - support[0, lower]) / (support[0, lower + 1] - support[0, lower])
    front[0, lower:lower + 2] = (1.0 - weight, weight)
    front[1] = inputs["transports"][0] @ front[0]
    inputs["initial_densities"] = front
    simulate, instruments = audit_batch_function(**inputs, config=config)
    factors = np.zeros((1, 4, 2))
    background = np.full(4, 0.5 * inputs["background_volatility"] / np.sqrt(252.0))
    gains, _, _, _, _ = simulate(jnp.asarray(factors), jnp.asarray(background))
    spot = np.asarray(gains)[:, 0] + 1.0
    strikes = instruments["option_strikes"][0]
    terminal = np.where(strikes >= 1.0, np.maximum(spot[:, None] - strikes, 0.0), np.maximum(strikes - spot[:, None], 0.0))
    recovered = np.asarray(gains)[:, 1:4] + instruments["initial_option_prices"][0]
    np.testing.assert_allclose(recovered, terminal, rtol=0.0, atol=1.0e-14)


def test_cpu_worker_does_not_claim_a_gpu(tmp_path) -> None:
    command = audit_worker_command(
        calibration_root=tmp_path, origin="2025-08-29", backend="cpu",
        config=UnconditionalAuditConfig(),
    )
    assert command[:2] == ["env", "JAX_PLATFORMS=cpu"]
    assert "--gpus" not in command
    assert all("gpuctl" not in part for part in command)


def test_single_origin_watcher_ignores_other_completed_dates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rolling_audit, "check_resource_headroom", lambda: None)
    for origin in ("2025-08-28", "2025-08-29"):
        directory = tmp_path / "rolling" / origin
        output = directory / "unconditional_q"
        output.mkdir(parents=True)
        (directory / "report.json").write_text("{}")
        (output / "report.json").write_text(json.dumps({"origin": origin}))
    rolling_audit.watch_audits(calibration_root=tmp_path, config=UnconditionalAuditConfig(), origin="2025-08-29", backend="cpu")
    summary = json.loads((tmp_path / "unconditional_audit_summary.json").read_text())
    assert summary["dates"] == [{"origin": "2025-08-29"}]
    assert summary["ready_rolling_dates"] == 1
    assert summary["backend"] == "cpu"


def test_streamed_moments_and_standard_errors_use_independent_pairs() -> None:
    generator = np.random.default_rng(42)
    observations = generator.normal(size=(17, 4))
    count, mean, squares = combine_moments(0, np.zeros(4), np.zeros(4), observations[:6])
    count, mean, squares = combine_moments(count, mean, squares, observations[6:])
    stats = moment_statistics(count, mean, squares, epsilon=1.0e-8)
    np.testing.assert_allclose(stats["mean_gain_over_spot"], np.mean(observations, axis=0), atol=1.0e-15)
    np.testing.assert_allclose(stats["standard_error"], np.std(observations, axis=0, ddof=1) / np.sqrt(17), atol=1.0e-15)


def test_deterministic_nonzero_drift_is_not_silently_given_zero_z() -> None:
    mean = np.asarray([0.0, 2.0e-10])
    stats = moment_statistics(100, mean, np.zeros(2), epsilon=1.0e-8)
    assert stats["z"][0] == 0.0
    assert np.isposinf(stats["z"][1])
    np.testing.assert_allclose(stats["economic_floor_score"], [0.0, 0.02])


def test_zero_interior_density_switches_off_localized_diffusion() -> None:
    density = jnp.asarray([[0.5, 0.0, 0.0, 0.5]])
    loadings = jnp.ones((1, 2, 3))
    actual = np.asarray(localize_loadings_by_density(loadings, density))
    np.testing.assert_array_equal(actual, 0.0)


def test_one_day_audit_uses_production_q_step_and_preserves_density_invariants(small_audit_inputs) -> None:
    config = UnconditionalAuditConfig(path_count=8, batch_size=8, substeps=2)
    simulate, instruments = audit_batch_function(**small_audit_inputs, config=config)
    factors, background = antithetic_shocks(seed=17, completed_paths=0, paths=8, substeps=2, factors=2)
    gains, taming, mass_error, forward_error, stable = simulate(jnp.asarray(factors), jnp.asarray(background))
    assert gains.shape == (8, 1 + instruments["option_strikes"].size)
    assert bool(stable)
    assert np.all(np.isfinite(np.asarray(gains)))
    assert float(taming) == 0.0
    assert float(mass_error) < 1.0e-10
    assert float(forward_error) < 1.0e-10


def test_resource_stop_prevents_work_but_probe_failure_is_logged_as_warning(monkeypatch, capsys) -> None:
    response = SimpleNamespace(returncode=0, stdout=json.dumps({"checks": [{"health": "STOP", "name": "ram"}]}))
    monkeypatch.setattr(rolling_audit.subprocess, "run", lambda *args, **kwargs: response)
    with pytest.raises(rolling_audit.ResourceHeadroomError):
        rolling_audit.check_resource_headroom()

    def failed_probe(*args, **kwargs):
        raise subprocess.TimeoutExpired("gpuctl", 15)

    monkeypatch.setattr(rolling_audit.subprocess, "run", failed_probe)
    monkeypatch.setattr(rolling_audit.time, "sleep", lambda _: None)
    rolling_audit.check_resource_headroom()
    assert "gpuctl resources could not be read" in capsys.readouterr().out


def test_audit_checkpoints_and_cached_report_keep_the_sampling_identity(tmp_path, monkeypatch, small_audit_inputs) -> None:
    monkeypatch.setattr(rolling_audit, "check_resource_headroom", lambda: None)
    origin = "2024-01-02"
    own = tmp_path / "own_dates" / origin
    rolling = tmp_path / "rolling" / origin
    own.mkdir(parents=True)
    rolling.mkdir(parents=True)
    model = small_audit_inputs
    np.savez(own / "inputs.npz", supports=model["supports"], transports=model["transports"],
             model_expiries_y=model["expiries_y"], projected_total_variances=model["projected_variances"])
    np.savez(own / "fit.npz", dlvsq=model["dlvsq"], densities=model["initial_densities"])
    np.savez(rolling / "ar1.npz", q_log_dlvsq_loadings=np.full_like(model["loadings"], 0.02))
    (own / "report.json").write_text(json.dumps({
        "model_label": "Gromm v2 4,3", "config": {"geometry": {"backbone_annual_volatility": 0.05}},
    }))
    (rolling / "report.json").write_text("{}")
    config = UnconditionalAuditConfig(path_count=8, batch_size=4, substeps=2)
    report = run_audit(calibration_root=tmp_path, origin=origin, config=config)
    assert report["paths"] == 8
    assert report["independent_antithetic_pairs"] == 4
    assert not report["conditional_strategy_training"]
    assert not report["measure_preconditioning"]
    assert report["maximum_density_mass_error"] < 1.0e-10
    output = rolling / "unconditional_q"
    assert (output / "statistics.npz").is_file()
    with np.load(output / "checkpoint.npz") as checkpoint:
        assert int(checkpoint["completed_paths"]) == 8
    assert run_audit(calibration_root=tmp_path, origin=origin, config=config) == json.loads(json.dumps(report))
    with pytest.raises(ValueError, match="different model or sampling identity"):
        run_audit(calibration_root=tmp_path, origin=origin, config=replace(config, seed=config.seed + 1))
