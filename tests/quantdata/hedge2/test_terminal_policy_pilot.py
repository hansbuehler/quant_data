"""Frozen simulation loading and untouched-holdout terminal-policy diagnostics."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.hedge2 import terminal_policy_pilot
from quantdata.hedge2.terminal_market import TerminalSimulationConfig, generate_terminal_paths
from quantdata.hedge2.terminal_policy_pilot import load_simulation_group, run_policy_pilot
from quantdata.hedge2.terminal_train import TerminalTrainConfig
from tests.quantdata.hedge2.test_terminal_market import inputs


def saved_simulation(root: Path) -> tuple[Path, dict[str, Path]]:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=12))
    directory = root / "simulation" / "exotic-3D-5D"
    directory.mkdir(parents=True)
    arguments["book"].to_parquet(directory / "book.parquet", index=False)
    np.savez_compressed(directory / "paths.npz", **asdict(result))
    (directory / "claims.json").write_text(json.dumps({"claims": [asdict(claim) for claim in arguments["claims"]]}))
    (directory / "diagnostics.json").write_text("{}")
    names = ("paths.npz", "book.parquet", "claims.json", "diagnostics.json")
    (directory / "completed.json").write_text(json.dumps({"artifacts_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}))
    sources = {key: str(root / key) for key in ("daily_directory", "dynamics_directory", "market_directory",
        "licensed_fit_directory", "raw_quote_directory")}
    day = result.model_date
    source_files = {"daily_config": Path(sources["daily_directory"]) / "config.json",
        "daily_state": Path(sources["daily_directory"]) / "days" / f"{day}.npz",
        "dynamics": Path(sources["dynamics_directory"]) / "days" / f"{day}.npz",
        "licensed_quotes": Path(sources["market_directory"]) / "days" / f"{day}.parquet",
        "licensed_fit": Path(sources["licensed_fit_directory"]) / "SPX" / f"{day}.pdq",
        "licensed_raw_quotes": Path(sources["raw_quote_directory"]) / "SPX" / f"{day}.pdq",
        "source_model": root / "source.h5"}
    for name, path in source_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    source_config = root / "sources.json"
    source_config.write_text(json.dumps(sources))
    implementation = Path(terminal_policy_pilot.__file__).parent
    configuration = {"source_configuration": str(source_config), "day": day,
        "source_configuration_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "simulation_version": TerminalSimulationConfig.version.unique_id48,
        "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items() if name != "source_model"},
        "source_model": {"source_model": str(source_files["source_model"]),
            "source_sha256": hashlib.sha256(source_files["source_model"].read_bytes()).hexdigest()},
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in implementation.glob("*.py")}}
    (directory.parent / "config.json").write_text(json.dumps(configuration))
    return directory, source_files


def test_saved_simulation_roundtrip_preserves_claim_features_and_fixed_book(tmp_path: Path) -> None:
    directory, _ = saved_simulation(tmp_path)
    result, book, claims = load_simulation_group(directory)
    assert result.instrument_gains.shape == (12, 5, 5)
    assert result.claim_labels == tuple(claim.label for claim in claims)
    assert len(book) == 4
    assert result.for_claim(claims[2], 2).features.shape == (12, 5, 5)
    assert result.seed == 81


def test_loader_rejects_modified_completed_artifact(tmp_path: Path) -> None:
    directory, _ = saved_simulation(tmp_path)
    (directory / "claims.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        load_simulation_group(directory)


def test_loader_rejects_changed_source_state(tmp_path: Path) -> None:
    directory, sources = saved_simulation(tmp_path)
    sources["daily_state"].write_text("different state")
    with pytest.raises(ValueError, match="source input changed: daily_state"):
        load_simulation_group(directory)


def test_loader_rejects_incompatible_simulation_implementation(tmp_path: Path) -> None:
    directory, _ = saved_simulation(tmp_path)
    path = directory.parent / "config.json"
    configuration = json.loads(path.read_text())
    configuration["implementation_sha256"]["terminal_market.py"] = "changed"
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="original implementation: terminal_market.py"):
        load_simulation_group(directory)


def test_changing_only_untouched_holdout_size_cannot_change_training(tmp_path: Path) -> None:
    directory, _ = saved_simulation(tmp_path)
    options = dict(simulation_directory=directory, claim_label="digital", training_paths=6,
        objective="mean_variance", risk_aversion=10., target_side=-1,
        train_config=TerminalTrainConfig(maximum_updates=2, minimum_updates=2), split_seed=19)
    first = run_policy_pilot(**options, output_directory=tmp_path / "fit-a", holdout_paths=2)
    second = run_policy_pilot(**options, output_directory=tmp_path / "fit-b", holdout_paths=4)
    first_saved = torch.load(first["checkpoint"], weights_only=True)
    second_saved = torch.load(second["checkpoint"], weights_only=True)
    assert first["status"] == second["status"] == "training_limit"
    assert first_saved["history"] == second_saved["history"]
    for name in first_saved["best_model"]:
        torch.testing.assert_close(first_saved["best_model"][name], second_saved["best_model"][name], rtol=0., atol=0.)
    assert first["small_sample_diagnostic"] and not first["realized_OOS_results"]
    split = np.load(tmp_path / "fit-a" / "split.npz", allow_pickle=False)
    assert not np.intersect1d(split["training"], split["holdout"]).size
    assert first["holdout_fitted"]["maximum_replay_error"] < 1e-12


def test_inception_pilot_cannot_silently_reset_carried_inventory(monkeypatch, tmp_path: Path) -> None:
    directory, _ = saved_simulation(tmp_path)
    result, book, claims = load_simulation_group(directory)
    monkeypatch.setattr(terminal_policy_pilot, "load_simulation_group",
        lambda directory: (replace(result, elapsed_day=1), book, claims))
    with pytest.raises(ValueError, match="must carry actual inventory"):
        run_policy_pilot(simulation_directory=directory, claim_label="digital", output_directory=tmp_path / "fit",
            training_paths=6, holdout_paths=4, objective="mean_variance", risk_aversion=10., target_side=-1)
