"""Raw-position replay detects economic, covariance and provenance mismatches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.audit_episode_risk import (
    audit_episode_risk,
    bound_risk_steps,
    replay_risk_step,
)
from quantdata.hedge2.parametric import gaussian_power_risk


def risk_example():
    positions = np.array([[.46, .2], [.184, .08], [.1, .2], [.3, .1], [.5, -.2], [0., 0.]])
    previous = np.zeros_like(positions)
    previous[4] = [11., -120.]
    inputs = {"costs": np.array([0., .03]), "previous_positions": previous,
        "normalized_premium_prices": np.array([1., .1]), "claim_labels": np.array(["claim"]),
        "instrument_exposures": np.array([[1., 0.], [.2, 1.]]),
        "claim_exposures": np.array([[.5, .2]]), "covariance": np.array([[.04, .01], [.01, .09]])}
    inputs["greek_instrument_exposures"] = inputs["instrument_exposures"].copy()
    inputs["greek_claim_exposures"] = inputs["claim_exposures"].copy()
    costs = np.abs(positions-previous) @ inputs["costs"]
    residual = positions @ inputs["instrument_exposures"]-inputs["claim_exposures"]
    variance = np.einsum("ij,jk,ik->i", residual, inputs["covariance"], residual)
    trades = pd.DataFrame({"claim": ["claim"]*6, "side": [-1]*6,
        "method": ["DLV Greek"]*2+["DLV parametric"]*4,
        "objective": ["none"]*2+["mean_variance"]*2+["power_exponential"]*2,
        "risk_aversion": [0., 0., 10., np.inf, 10., 10.], "greek_budget": [1., .4, np.nan, np.nan, np.nan, np.nan],
        "greek_full_entry_cost": [costs[0], costs[0], np.nan, np.nan, np.nan, np.nan],
        "claim_active_before_step": [True]*5+[False], "entry_cost": costs,
        "training_std": np.sqrt(variance), "training_mean_net": -costs, "training_objective": np.zeros(6)})
    trades.loc[2, "training_objective"] = 5.*variance[2]+costs[2]
    trades.loc[3, "training_objective"] = .5*variance[3]
    trades.loc[4, "training_objective"] = gaussian_power_risk(np.sqrt(variance[4]), risk_aversion=10.)[0]+costs[4]
    trades.loc[5, ["training_std", "training_mean_net", "training_objective"]] = np.nan
    return trades, positions, inputs


def test_raw_risk_replay_checks_mixed_objectives_and_keeps_carried_inventory_cost() -> None:
    trades, positions, inputs = risk_example()
    record, candidates = replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)
    assert record["portfolios"] == 6
    assert record["active_portfolios"] == 5
    assert max(record[name] for name in ("cost_error", "std_error", "mean_error", "mv_objective_error")) < 1e-14
    assert len(candidates) == 1
    assert candidates[0]["row"] == 4
    assert trades.loc[4, "entry_cost"] > 3.


@pytest.mark.parametrize("defect,exception,match", [
    ("fee", RuntimeError, "turnover"), ("cap", RuntimeError, "premium limit"),
    ("budget", RuntimeError, "cost budget"), ("greek", RuntimeError, "price sensitivities"),
    ("variance_cost", RuntimeError, "MV objective"), ("covariance", ValueError, "semidefinite"),
    ("mean", RuntimeError, "net mean"), ("label", ValueError, "absent"),
])
def test_raw_risk_replay_rejects_independent_mismatches(defect, exception, match) -> None:
    trades, positions, inputs = risk_example()
    if defect == "fee":
        trades.loc[0, "entry_cost"] += .01
    elif defect == "cap":
        positions[2, 0] = 11.
    elif defect == "budget":
        trades.loc[1, "greek_budget"] = .2
    elif defect == "greek":
        positions[0, 0] += .1
    elif defect == "variance_cost":
        trades.loc[3, "training_objective"] += trades.loc[3, "entry_cost"]
    elif defect == "covariance":
        inputs["covariance"][0, 0] = -1.
    elif defect == "mean":
        trades.loc[2, "training_mean_net"] = .1
    else:
        trades.loc[2, "claim"] = "missing"
    with pytest.raises(exception, match=match):
        replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)


def test_inactive_claims_still_pay_actual_turnover_without_fresh_risk_geometry() -> None:
    trades, positions, inputs = risk_example()
    trades["claim_active_before_step"] = False
    for name in ("instrument_exposures", "claim_exposures", "greek_instrument_exposures",
            "greek_claim_exposures", "covariance"):
        inputs.pop(name)
    record, candidates = replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)
    assert record["active_portfolios"] == 0
    assert not candidates
    trades.loc[4, "entry_cost"] = 0.
    with pytest.raises(RuntimeError, match="turnover"):
        replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)


def write_population(root: Path) -> Path:
    directory = root / "2024-04-19" / "digital-3D"
    step = directory / "steps" / "step-00"
    step.mkdir(parents=True)
    trades, positions, inputs = risk_example()
    trades, positions = trades.iloc[:5].copy(), positions[:5]
    inputs["previous_positions"] = np.zeros_like(positions)
    inputs["active_columns"] = np.array([0, 1])
    actual_cost = np.abs(positions) @ inputs["costs"]
    finite = np.isfinite(trades.risk_aversion.to_numpy())
    trades.loc[finite, "training_objective"] += actual_cost[finite]-trades.entry_cost.to_numpy()[finite]
    trades["entry_cost"], trades["training_mean_net"] = actual_cost, -actual_cost
    trades["elapsed_day"], trades["knockout_exit_at_mid"] = 0, False
    trades["date"] = "2024-04-19"
    trades.to_parquet(step / "trades.parquet", index=False)
    trades.to_parquet(directory / "trades.parquet", index=False)
    pd.DataFrame({"strike": [1.1], "expiry": ["2024-04-24"]}).to_parquet(directory / "initial_book.parquet", index=False)
    np.savez_compressed(directory / "positions.npz", step_0=positions, active_0=np.array([0, 1]))
    np.savez_compressed(step / "completed_step.npz", positions=positions)
    np.savez_compressed(step / "risk_inputs.npz", **inputs)
    hashes = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()}
    (directory / "completed.json").write_text(json.dumps({"artifacts_sha256": hashes}))
    (root / "config.json").write_text(json.dumps({"optimized_premium_cap": 10.}))
    (root / "summary.json").write_text(json.dumps({"status": "complete", "episodes": 1}))
    return directory


def test_population_replay_checks_integrals_hashes_and_completion_scope(tmp_path) -> None:
    root = tmp_path / "input"
    directory = write_population(root)
    summary = audit_episode_risk(input_directory=root, output_directory=tmp_path / "audit")
    assert summary["daily_portfolios"] == 5
    assert summary["gaussian_integrals"] == 1
    assert summary["maximum_integral_error"] < 1e-10
    assert not summary["partial_run_audit"]
    with (directory / "steps" / "step-00" / "risk_inputs.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        audit_episode_risk(input_directory=root, output_directory=tmp_path / "bad")


def test_partial_population_requires_explicit_flag_and_discloses_unsealed_risk_files(tmp_path) -> None:
    root = tmp_path / "input"
    directory = write_population(root)
    (root / "summary.json").unlink()
    with pytest.raises(ValueError, match="unfinished population"):
        audit_episode_risk(input_directory=root, output_directory=tmp_path / "refused")
    result = audit_episode_risk(input_directory=root, output_directory=tmp_path / "partial", completed_only=True)
    assert result["partial_run_audit"]
    path = directory / "completed.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts_sha256"].pop("steps/step-00/risk_inputs.npz")
    path.write_text(json.dumps(manifest))
    captured = audit_episode_risk(input_directory=root, output_directory=tmp_path / "captured", completed_only=True)
    assert str(directory / "steps" / "step-00" / "risk_inputs.npz") in captured["post_completion_capture_sha256"]
    assert captured["risk_replay_conditional_on_captured_geometry"]


def test_full_audit_rejects_mismatched_population_size(tmp_path) -> None:
    root = tmp_path / "input"
    write_population(root)
    (root / "summary.json").write_text(json.dumps({"status": "complete", "episodes": 2}))
    with pytest.raises(ValueError, match="completion count"):
        audit_episode_risk(input_directory=root, output_directory=tmp_path / "audit")


def test_post_completion_capture_still_rejects_changed_prior_inventory(tmp_path) -> None:
    root = tmp_path / "input"
    directory = write_population(root)
    manifest_path = directory / "completed.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts_sha256"] = {name: digest for name, digest in manifest["artifacts_sha256"].items()
        if not name.startswith("steps/")}
    manifest_path.write_text(json.dumps(manifest))
    path = directory / "steps" / "step-00" / "risk_inputs.npz"
    with np.load(path, allow_pickle=False) as saved:
        inputs = {name: saved[name] for name in saved.files}
    inputs["previous_positions"][0, 0] = .1
    np.savez_compressed(path, **inputs)
    with pytest.raises(AssertionError):
        audit_episode_risk(input_directory=root, output_directory=tmp_path / "audit")


def test_nonfinite_computed_variance_and_zero_live_premium_are_rejected() -> None:
    trades, positions, inputs = risk_example()
    inputs["normalized_premium_prices"][1] = 0.
    with pytest.raises(ValueError, match="Premium audit"):
        replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)
    inputs["normalized_premium_prices"][1] = .1
    inputs["instrument_exposures"][:] = 1e308
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="non-finite|overflowed"):
            replay_risk_step(trades=trades, positions=positions, inputs=inputs, premium_cap=10.)


def test_nonfinite_independent_quadrature_cannot_pass_the_audit(tmp_path, monkeypatch) -> None:
    import quantdata.hedge2.audit_episode_risk as module

    root = tmp_path / "input"
    write_population(root)
    monkeypatch.setattr(module, "integrated_normal_power", lambda *args, **kwargs: (np.nan, 0.))
    with pytest.raises(RuntimeError, match="utility integration"):
        audit_episode_risk(input_directory=root, output_directory=tmp_path / "audit")


def test_bound_steps_keep_knockout_clearing_and_expiry_inventory_causal(tmp_path) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    pd.DataFrame({"strike": [1.1]}).to_parquet(episode / "initial_book.parquet", index=False)
    first = pd.DataFrame({"claim": ["knocked", "alive"], "method": ["DLV Greek"]*2,
        "objective": ["none"]*2, "risk_aversion": [0., 0.], "side": [-1, -1], "greek_budget": [1., 1.],
        "elapsed_day": [0, 0], "claim_active_before_step": [True, True],
        "knockout_exit_at_mid": [True, False]})
    second = first.copy()
    second["elapsed_day"] = 1
    second["claim_active_before_step"] = [False, True]
    second["knockout_exit_at_mid"] = False
    positions = (np.array([[.1, .2], [.3, .4]]), np.array([[0.], [.2]]))
    columns = (np.array([0, 1]), np.array([0]))
    previous = (np.zeros((2, 2)), np.array([[0.], [.3]]))
    pd.concat([first, second], ignore_index=True).to_parquet(episode / "trades.parquet", index=False)
    np.savez_compressed(episode / "positions.npz", step_0=positions[0], step_1=positions[1],
        active_0=columns[0], active_1=columns[1])
    for step, frame in enumerate((first, second)):
        directory = episode / "steps" / f"step-{step:02d}"
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "trades.parquet", index=False)
        np.savez_compressed(directory / "completed_step.npz", positions=positions[step])
        np.savez_compressed(directory / "risk_inputs.npz", previous_positions=previous[step],
            active_columns=columns[step], claim_labels=np.array(["knocked", "alive"]))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in episode.iterdir() if path.is_file()}
    steps, captured = bound_risk_steps(episode, hashes)
    assert len(steps) == 2
    assert len(captured) == 6
    path = episode / "steps" / "step-01" / "risk_inputs.npz"
    np.savez_compressed(path, previous_positions=np.array([[.1], [.3]]), active_columns=columns[1],
        claim_labels=np.array(["knocked", "alive"]))
    with pytest.raises(AssertionError):
        bound_risk_steps(episode, hashes)


def test_bound_steps_reject_missing_elapsed_step(tmp_path) -> None:
    directory = write_population(tmp_path / "input")
    (directory / "steps" / "step-00").rename(directory / "steps" / "step-01")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.is_file() and path.name != "completed.json"}
    with pytest.raises(ValueError, match="elapsed-step set"):
        bound_risk_steps(directory, hashes)
