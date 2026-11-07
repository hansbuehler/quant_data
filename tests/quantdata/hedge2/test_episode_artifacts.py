"""Completion seals bind actual accounting records, source inputs and final checkpoints."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.episode_artifacts import publish_episode, verify_episode_artifacts
from quantdata.hedge2.episodes import EpisodeResult


def _episode(tmp_path):
    directory = tmp_path / "2024-04-01/digital-1D"
    (directory / "steps").mkdir(parents=True)
    book = pd.DataFrame({"strike": [100.]})
    hashes = {"frozen_book": hashlib.sha256(pd.util.hash_pandas_object(book, index=True).to_numpy().tobytes()).hexdigest(),
        "observed_marks_spreads_and_carry": "1" * 64, "/test/source": "2" * 64}
    result = EpisodeResult(summary=pd.DataFrame({"origin": ["2024-04-01"], "costed_pnl": [.2]}),
        trades=pd.DataFrame({"hedge_gain": [.2]}), initial_book=book,
        positions=(np.array([[1., 2.]]),), active_indices=(np.array([0, 1]),),
        diagnostics=({"spot_cost": 0.},), input_hashes=hashes)
    identity = {"origin": "2024-04-01", "family": "digital-1D", "input_hashes": hashes}
    np.savez_compressed(directory / "steps/episode_checkpoint.npz", identity_json=json.dumps(identity),
        next_step=1, positions_0=result.positions[0], indices_0=result.active_indices[0])
    return directory, result


def _publish(directory, result):
    publish_episode(directory=directory, result=result, origin="2024-04-01",
        family="digital-1D", configuration_sha256="3" * 64)


def test_completion_seals_inputs_checkpoint_and_all_accounting_files(tmp_path) -> None:
    directory, result = _episode(tmp_path)
    _publish(directory, result)
    manifest = verify_episode_artifacts(directory=directory, configuration_sha256="3" * 64)
    assert manifest["schema_version"] == 1
    assert json.loads((directory / "input_hashes.json").read_text()) == result.input_hashes
    assert "steps/episode_checkpoint.npz" in manifest["artifacts_sha256"]
    assert not (directory / "completed.partial.json").exists()
    with pytest.raises(ValueError, match="immutable"):
        _publish(directory, result)


@pytest.mark.parametrize("artifact", ["summary.parquet", "input_hashes.json", "steps/episode_checkpoint.npz"])
def test_a_mutated_completion_artifact_is_rejected(tmp_path, artifact) -> None:
    directory, result = _episode(tmp_path)
    _publish(directory, result)
    (directory / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_episode_artifacts(directory=directory, configuration_sha256="3" * 64)


@pytest.mark.parametrize("problem", ["input_hashes", "book", "positions", "indices", "horizon"])
def test_inconsistent_final_checkpoint_cannot_publish_completion(tmp_path, problem) -> None:
    directory, result = _episode(tmp_path)
    if problem == "input_hashes":
        result = replace(result, input_hashes=result.input_hashes | {"/test/source": "4" * 64})
    elif problem == "book":
        result = replace(result, initial_book=pd.DataFrame({"strike": [101.]}))
    elif problem == "positions":
        result = replace(result, positions=(np.array([[1., 3.]]),))
    elif problem == "indices":
        result = replace(result, active_indices=(np.array([1, 0]),))
    else:
        with np.load(directory / "steps/episode_checkpoint.npz", allow_pickle=False) as saved:
            values = dict(saved)
        np.savez_compressed(directory / "steps/episode_checkpoint.npz", **(values | {"next_step": 0}))
    with pytest.raises(ValueError):
        _publish(directory, result)
    assert not (directory / "completed.json").exists()


def test_configuration_substitution_and_unsealed_legacy_inputs_are_rejected(tmp_path) -> None:
    directory, result = _episode(tmp_path)
    _publish(directory, result)
    with pytest.raises(ValueError, match="another configuration"):
        verify_episode_artifacts(directory=directory, configuration_sha256="4" * 64)
    manifest_path = directory / "completed.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["artifacts_sha256"]["input_hashes.json"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance is incomplete"):
        verify_episode_artifacts(directory=directory, configuration_sha256="3" * 64)


def test_completion_cannot_read_an_artifact_outside_its_directory(tmp_path) -> None:
    directory, result = _episode(tmp_path)
    _publish(directory, result)
    outside = tmp_path / "foreign"
    outside.write_bytes(b"outside")
    manifest_path = directory / "completed.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts_sha256"]["../../foreign"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escaped its directory"):
        verify_episode_artifacts(directory=directory, configuration_sha256="3" * 64)
