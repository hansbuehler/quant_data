"""Tests for the residual-clock conditional no-arbitrage audit."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import quantdata.dysanos.flex.run_residual_clock_rn_conditional_audit as audit
from quantdata.dysanos.flex.run_residual_clock_rn_conditional_audit import (
    ResidualClockRnPolicy,
    ResidualClockRnPolicyConfig,
    adjacent_pair_test,
    load_residual_clock_rn_path_cache,
    log_mean_exp_negative_gain,
    parser,
    run,
    squared_second_moment_reward,
    streaming_normalizers,
    unconditional_coordinate_diagnostics,
)
from quantdata.torch import BaseEnvironment


def _write_population(
    tmp_path,
    *,
    state_size: int = 7,
    return_value: float = 0.0,
) -> None:
    num_paths = 8
    num_steps = 25
    states = np.arange(
        num_paths * num_steps * state_size,
        dtype=np.float32,
    ).reshape(num_paths, num_steps, state_size)
    prices = np.full((num_paths, num_steps, 4, 12), 0.25, dtype=np.float32)
    returns = np.full(
        (num_paths, num_steps, 49),
        return_value,
        dtype=np.float32,
    )
    feature_valid = np.ones((num_paths, num_steps), dtype=np.uint8)
    return_valid = np.ones((num_paths, num_steps, 49), dtype=np.uint8)
    np.save(tmp_path / "policy-states.npy", states)
    np.save(tmp_path / "policy-option-prices.npy", prices)
    np.save(tmp_path / "returns.npy", returns)
    np.save(tmp_path / "policy-feature-valid.npy", feature_valid)
    np.save(tmp_path / "return-valid.npy", return_valid)
    metadata = {
        "schema": "dysanos-residual-clock-rn-oce-paths-v1-population",
        "shapes": {
            "num_paths": num_paths,
            "issue_days": num_steps,
            "policy_state_size": state_size,
        },
        "files": {
            "policy_states": "policy-states.npy",
            "option_prices": "policy-option-prices.npy",
            "returns": "returns.npy",
            "feature_valid": "policy-feature-valid.npy",
            "return_valid": "return-valid.npy",
        },
        "path_order": (
            "adjacent rows (2k,2k+1) are one antithetic pair; batches are "
            "contiguous global pair intervals"
        ),
    }
    (tmp_path / "population.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_loads_primary_population_manifest_and_memmaps(tmp_path) -> None:
    _write_population(tmp_path)

    cache = load_residual_clock_rn_path_cache(tmp_path)

    assert cache.metadata_path == tmp_path / "population.json"
    assert cache.state_size == 7
    assert cache.num_paths == 8
    assert cache.num_steps == 25
    assert cache.states.shape == (8, 25, 7)
    assert cache.prices.shape == (8, 25, 4, 12)
    assert cache.returns.shape == (8, 25, 49)
    assert isinstance(cache.states, np.memmap)
    assert isinstance(cache.prices, np.memmap)
    assert isinstance(cache.returns, np.memmap)


def test_population_loader_rejects_incomplete_validity_mask(tmp_path) -> None:
    _write_population(tmp_path)
    validity = np.load(tmp_path / "return-valid.npy")
    validity[3, 4, 5] = 0
    np.save(tmp_path / "return-valid.npy", validity)

    with pytest.raises(ValueError, match="incomplete path entries"):
        load_residual_clock_rn_path_cache(tmp_path)


def test_loader_retains_legacy_cache_names(tmp_path) -> None:
    np.save(tmp_path / "states.npy", np.zeros((4, 25, 3), dtype=np.float32))
    np.save(tmp_path / "prices.npy", np.ones((4, 25, 4, 12), dtype=np.float32))
    np.save(tmp_path / "returns.npy", np.zeros((4, 25, 49), dtype=np.float32))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "num_paths": 4,
                "num_steps": 25,
                "policy_state_metadata": {"feature_size": 3},
                "antithetic_pair_ordering": "adjacent",
            }
        ),
        encoding="utf-8",
    )

    cache = load_residual_clock_rn_path_cache(tmp_path)

    assert cache.state_size == 3
    assert cache.num_steps == 25


def test_policy_is_uncapped_shared_deep_plus_expiry_affine_skips() -> None:
    state_size = 7
    env = BaseEnvironment(precision=32, device="cpu", debug=True)
    model = ResidualClockRnPolicy(
        env,
        config=ResidualClockRnPolicyConfig(label="audit-test", state_size=state_size),
        state_mean=torch.zeros(state_size),
        state_scale=torch.ones(state_size),
        log_price_mean=torch.zeros(12),
        log_price_scale=torch.ones(12),
    )
    with torch.no_grad():
        for expiry, layer in enumerate(model.linear_skip):
            layer.bias.fill_(float(expiry + 1))
    action = model(
        torch.zeros(2, state_size),
        torch.ones(2, 4, 12),
        calendar_time=torch.tensor([0.0, 1.0]),
    )

    assert action.shape == (2, 49)
    torch.testing.assert_close(action[:, 0], torch.full((2,), 10.0))
    expected_options = torch.repeat_interleave(
        torch.arange(1.0, 5.0),
        repeats=12,
    )
    torch.testing.assert_close(action[:, 1:], expected_options[None].expand(2, -1))
    assert model.config.network.num_depth == 2
    assert model.config.network.num_width == 128


def test_streaming_normalizers_match_direct_population_moments() -> None:
    generator = np.random.default_rng(101)
    states = generator.normal(size=(8, 3, 5)).astype(np.float32)
    prices = np.exp(generator.normal(size=(8, 3, 4, 12))).astype(np.float32)

    actual = streaming_normalizers(
        states,
        prices,
        start=2,
        stop=8,
        batch_size=2,
    )

    expected_state = states[2:].astype(np.float64)
    expected_log_price = np.log(prices[2:].astype(np.float64))
    np.testing.assert_allclose(
        actual["state_mean"],
        np.mean(expected_state, axis=(0, 1)),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        actual["state_scale"],
        np.std(expected_state, axis=(0, 1)),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        actual["log_price_mean"],
        np.mean(expected_log_price, axis=(0, 1, 2)),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        actual["log_price_scale"],
        np.std(expected_log_price, axis=(0, 1, 2)),
        rtol=2.0e-6,
    )


def test_objective_and_pair_cluster_statistics_match_direct_calculation() -> None:
    gain = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    loss, effective_size = log_mean_exp_negative_gain(gain, batch_size=1)
    weights = np.exp(-gain.reshape(-1))

    assert loss == pytest.approx(math.log(float(np.mean(weights))))
    assert effective_size == pytest.approx(float(np.sum(weights) ** 2 / np.sum(weights**2)))

    result = adjacent_pair_test(np.array([1.0, 3.0, 2.0, 4.0]))
    assert result["antithetic_pairs"] == 2
    assert result["mean_daily_captured_return"] == pytest.approx(2.5)
    assert result["pair_cluster_standard_error"] == pytest.approx(0.5)
    assert result["statistic"] == pytest.approx(5.0)
    assert result["two_sided_normal_p_value"] == pytest.approx(math.erfc(5.0 / math.sqrt(2.0)))


def test_squared_second_moment_reward_matches_main2_formula_and_gradient() -> None:
    actions = torch.tensor(
        [[[0.25, -0.5], [0.75, 0.125]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    returns = torch.tensor(
        [[[0.4, -0.2], [-0.1, 0.3]]],
        dtype=torch.float64,
    )

    reward = squared_second_moment_reward(
        actions,
        returns,
        lambda_second_moment=1.0,
    )
    raw = torch.sum(actions * returns, dim=-1)
    expected = raw - 0.5 * torch.sum(actions.square(), dim=-1) - 0.5 * raw.square()

    torch.testing.assert_close(reward, expected)
    reward.sum().backward()
    expected_gradient = returns - actions - raw[..., None] * returns
    torch.testing.assert_close(actions.grad, expected_gradient)


def test_unconditional_diagnostics_cluster_adjacent_pairs() -> None:
    first_pair = np.array([[1.0, 2.0], [3.0, 4.0]])
    second_pair = first_pair + 2.0
    returns = np.stack((first_pair, first_pair, second_pair, second_pair))

    result = unconditional_coordinate_diagnostics(
        returns,
        start=0,
        stop=4,
        batch_size=2,
    )

    assert result["antithetic_pairs"] == 2
    assert result["coordinates"] == 4
    assert result["maximum_absolute_z"] == pytest.approx(5.0)
    assert result["coordinates_absolute_z_above_1_96"] == 4
    assert result["coordinates_absolute_z_above_2_576"] == 3
    assert result["coordinates_absolute_z_above_5"] == 0
    assert result["worst_coordinate"] == {
        "time_index_zero_based": 1,
        "instrument_index_zero_based": 1,
        "mean": 5.0,
        "pair_cluster_standard_error": 1.0,
        "z_score": 5.0,
    }


def test_parser_uses_dy21_audit_schedule(tmp_path) -> None:
    arguments = parser().parse_args(
        ["--cache", str(tmp_path), "--output", str(tmp_path / "output")]
    )

    assert arguments.affine_epochs == 100
    assert arguments.full_epochs == 200
    assert arguments.lambda_second_moment == 1.0
    assert arguments.backtracking_factor == 0.2
    assert arguments.maximum_line_search_steps == 20


def test_progress_reports_stage_eta_and_resets_for_next_fold(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "_check_resources",
        lambda _output: {"source": "test", "health": "OK", "checks": []},
    )
    progress = audit._Progress(tmp_path, interval_seconds=0.0)
    base = {
        "stage": "train-epoch-complete",
        "diagnostic": "squared",
        "fit_stage": "full",
        "total_epochs": 4,
        "accepted_step_size": 1.0e-3,
    }

    progress.emit({**base, "epoch": 1, "epoch_seconds": 10.0}, force=True)
    progress.emit({**base, "epoch": 2, "epoch_seconds": 20.0}, force=True)
    progress.emit({**base, "epoch": 1, "epoch_seconds": 5.0}, force=True)

    records = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["estimated_remaining_seconds"] == pytest.approx(30.0)
    assert records[1]["estimated_remaining_seconds"] == pytest.approx(30.0)
    assert records[1]["estimated_remaining_epochs"] == 2
    assert records[1]["estimated_remaining_scope"] == "current-fit-stage"
    assert records[2]["estimated_remaining_seconds"] == pytest.approx(15.0)


def test_resource_snapshot_uses_gpuctl_and_stop_health_is_enforced(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        "source": "daemon",
        "timestamp": 123.0,
        "checks": [
            {"name": "ram", "health": "OK", "current": {"free_gib": 80.0}},
            {"name": "data", "health": "UNAVAILABLE", "current": None},
        ],
    }

    class Result:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(audit.subprocess, "run", lambda *_args, **_kwargs: Result())

    snapshot = audit._resource_snapshot(tmp_path)
    assert snapshot["source"] == "daemon"
    assert snapshot["health"] == "STOP"
    with pytest.raises(RuntimeError, match="failing checks: data"):
        audit._check_resources(tmp_path)


def test_resource_command_failure_is_reported_as_warning(tmp_path, monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("gpuctl")

    monkeypatch.setattr(audit.subprocess, "run", unavailable)

    snapshot = audit._check_resources(tmp_path)
    assert snapshot["health"] == "WARN"
    assert "could not be read" in snapshot["error"]


def test_one_epoch_smoke_run_writes_both_cross_fit_reports(tmp_path) -> None:
    cache_path = tmp_path / "cache"
    output_path = tmp_path / "audit"
    cache_path.mkdir()
    _write_population(cache_path, state_size=5)
    arguments = parser().parse_args(
        [
            "--cache",
            str(cache_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--affine-epochs",
            "3",
            "--full-epochs",
            "3",
            "--progress-interval-seconds",
            "0",
        ]
    )

    run(arguments)

    report = json.loads((output_path / "report.json").read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["cache"]["state_size"] == 5
    assert report["pooled_out_of_fold"]["T_sqr"] == 0.0
    assert report["pooled_out_of_fold"]["p_sqr"] == 1.0
    assert report["pooled_out_of_fold"]["T_ent"] == 0.0
    assert report["pooled_out_of_fold"]["p_ent"] == 1.0
    for diagnostic in ("squared", "entropy"):
        result = report["conditional_diagnostics"][diagnostic]
        assert result["fold_A_fit"]["training_antithetic_pairs"] == 2
        assert result["fold_B_fit"]["training_antithetic_pairs"] == 2
        assert result["fold_A_fit"]["affine_epochs"] == 1
        assert result["fold_A_fit"]["full_epochs"] == 1
        assert "quadratic_score" not in result
        assert (output_path / diagnostic / "a" / "affine.pt").exists()
        assert (output_path / diagnostic / "b" / "full.pt").exists()
        checkpoint = torch.load(
            output_path / diagnostic / "a" / "affine.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["complete"] is True
        assert checkpoint["epoch"] == 1
        assert checkpoint["history"][0]["accepted_step_size"] == 0.0
        assert checkpoint["history"][0]["line_search_evaluations"] == 20
        assert checkpoint["optimization_config"] == {
            "method": "exact-gradient-descent-with-backtracking",
            "learning_rate": 1.0e-3,
            "maximum_gradient_norm": 10.0,
            "backtracking_factor": 0.2,
            "maximum_line_search_steps": 20,
            "minimum_line_search_improvement": (1.0e-12 if diagnostic == "entropy" else 1.0e-15),
        }


def test_backtracking_accepts_only_improving_steps_and_keys_checkpoint(tmp_path) -> None:
    cache_path = tmp_path / "cache"
    output_path = tmp_path / "audit"
    cache_path.mkdir()
    _write_population(cache_path, state_size=3, return_value=0.05)
    base_arguments = [
        "--cache",
        str(cache_path),
        "--output",
        str(output_path),
        "--device",
        "cpu",
        "--batch-size",
        "4",
        "--affine-epochs",
        "1",
        "--full-epochs",
        "0",
        "--learning-rate",
        "100",
        "--maximum-line-search-steps",
        "8",
        "--progress-interval-seconds",
        "1000",
    ]

    run(parser().parse_args(base_arguments))

    report = json.loads((output_path / "report.json").read_text(encoding="utf-8"))
    for diagnostic in ("squared", "entropy"):
        for fold in ("fold_A_fit", "fold_B_fit"):
            record = report["conditional_diagnostics"][diagnostic][fold]["affine_final"]
            assert record["line_search_evaluations"] > 1
            assert 0.0 < record["accepted_step_size"] < 100.0
            assert record["candidate_objective_loss"] < record["objective_loss"]

    incompatible = parser().parse_args([*base_arguments, "--backtracking-factor", "0.5"])
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        run(incompatible)
