"""Run snapshots must reject corruption and avoid loading mutable project code."""

import hashlib
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import pytest

from quantdata.hedge2.objective_run import _identity
from quantdata.hedge2.snapshot_run import restore_implementation


def test_snapshot_restoration_verifies_exact_sources_and_can_be_repeated(tmp_path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    content = b'"""Frozen test implementation."""\n'
    (parent / "config.json").write_text(json.dumps({"implementation_sha256": {
        "objective_run.py": hashlib.sha256(content).hexdigest()}}))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("objective_run.py", content)
    output = tmp_path / "child"
    restored = restore_implementation(input_directory=parent, output_directory=output)
    assert (restored / "objective_run.py").read_bytes() == content
    assert restore_implementation(input_directory=parent, output_directory=output) == restored
    (restored / "objective_run.py").write_bytes(content + b"\n")
    with pytest.raises(ValueError, match="differs"):
        restore_implementation(input_directory=parent, output_directory=output)


def test_frozen_multiday_runner_uses_its_own_entrypoint_in_a_fresh_process(tmp_path) -> None:
    parent = tmp_path / "multiday"
    parent.mkdir()
    content = (b"def run_episodes(**arguments):\n"
               b"    return {'entrypoint': 'multiday', 'methods': list(arguments['methods'])}\n")
    configuration = {"experiment_type": "multiday", "dates": ["2023-01-09"],
        "implementation_sha256": {"episode_run.py": hashlib.sha256(content).hexdigest()},
        "methods": ["unhedged"], "families": ["ko"], "risks": [1.], "greek_budgets": [1.],
        "shard_index": 0, "shard_count": 1, "objective_gap_tolerance": 1e-4,
        **{name: str(tmp_path) for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
            "market_directory", "daily_directory", "dynamics_directory", "coverage_path")}}
    (parent / "config.json").write_text(json.dumps(configuration))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("episode_run.py", content)
    completed = subprocess.run([sys.executable, "-m", "quantdata.hedge2.snapshot_run", "--input-directory", str(parent),
        "--output-directory", str(tmp_path / "output")], env=os.environ.copy(), capture_output=True, text=True, check=True)
    assert '"entrypoint": "multiday"' in completed.stdout
    assert '"unhedged"' in completed.stdout


def test_snapshot_rejects_a_source_that_does_not_match_its_recorded_digest(tmp_path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text(json.dumps({"implementation_sha256": {"objective_run.py": "wrong"}}))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("objective_run.py", "pass\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_implementation(input_directory=parent, output_directory=tmp_path / "output")


def test_missing_transitive_source_requires_explicit_recorded_supplement(tmp_path) -> None:
    parent, source = tmp_path / "parent", tmp_path / "source"
    parent.mkdir()
    source.mkdir()
    content = b"from quantdata.hedge2.dlv import DlvConfig\n"
    (parent / "config.json").write_text(json.dumps({"implementation_sha256": {
        "objective_run.py": hashlib.sha256(content).hexdigest()}}))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("objective_run.py", content)
    (source / "dlv.py").write_text("DlvConfig = None\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="omitted dependency"):
        restore_implementation(input_directory=parent, output_directory=output)
    restored = restore_implementation(input_directory=parent, output_directory=output, supplemental_directory=source)
    supplements = json.loads((output / "supplemental_sources.json").read_text())
    assert supplements["dlv.py"]["sha256"] == hashlib.sha256((restored / "dlv.py").read_bytes()).hexdigest()
    assert restore_implementation(input_directory=parent, output_directory=output) == restored


@pytest.mark.parametrize("premium_book", (True, False))
def test_frozen_one_day_restart_preserves_the_book_and_return_units(tmp_path, premium_book) -> None:
    parent = tmp_path / "one-day"
    parent.mkdir()
    content = (b"def run_objectives(**arguments):\n"
        b"    return {'itm_premium_book': arguments['itm_premium_book'], 'method': arguments['method']}\n")
    configuration = {"method": "historical", "dates": ["2023-01-09"], "itm_premium_book": premium_book,
        "implementation_sha256": {"objective_run.py": hashlib.sha256(content).hexdigest()},
        "objective_gap_tolerance": 1e-4, "diagnostic_only": False,
        **{name: str(tmp_path) for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
            "market_directory", "daily_directory", "dynamics_directory")}}
    (parent / "config.json").write_text(json.dumps(configuration))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("objective_run.py", content)
    completed = subprocess.run([sys.executable, "-m", "quantdata.hedge2.snapshot_run", "--input-directory", str(parent),
        "--output-directory", str(tmp_path / "output")], env=os.environ.copy(), capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout[completed.stdout.index("{\n"):])["itm_premium_book"] == premium_book


def test_new_one_day_archives_capture_the_full_local_dependency_closure(tmp_path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    configuration = {"method": "statistical", "dates": ["2024-04-19"], "itm_premium_book": True}
    _identity(output_directory=parent, configuration=configuration)
    restored = restore_implementation(input_directory=parent, output_directory=tmp_path / "output")
    assert (restored / "dlv.py").is_file()
    assert (restored / "period.py").is_file()
    assert (restored / "front_joint.py").is_file()
    assert (restored / "front_scenarios.py").is_file()
    assert not (tmp_path / "output/supplemental_sources.json").exists()


@pytest.mark.parametrize("law", ("empirical_rank", "joint_gaussian"))
def test_frozen_front_restart_preserves_explicit_joint_law_and_score_panel(tmp_path, law) -> None:
    parent = tmp_path / "one-day-front"
    parent.mkdir()
    content = (b"def run_objectives(**arguments):\n"
        b"    return {'law': arguments['front_surface_law'], 'scores': str(arguments['front_scores_path'])}\n")
    scores = tmp_path / "scores.parquet"
    configuration = {"method": "statistical", "dates": ["2024-04-19"], "itm_premium_book": True,
        "front_surface_law": law, "front_scores_path": str(scores),
        "implementation_sha256": {"objective_run.py": hashlib.sha256(content).hexdigest()},
        "objective_gap_tolerance": 1e-4, "diagnostic_only": True,
        **{name: str(tmp_path) for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
            "market_directory", "daily_directory", "dynamics_directory")}}
    (parent / "config.json").write_text(json.dumps(configuration))
    with ZipFile(parent / "implementation.zip", "w") as archive:
        archive.writestr("objective_run.py", content)
    completed = subprocess.run([sys.executable, "-m", "quantdata.hedge2.snapshot_run", "--input-directory", str(parent),
        "--output-directory", str(tmp_path / "output")], env=os.environ.copy(), capture_output=True, text=True, check=True)
    output = json.loads(completed.stdout[completed.stdout.index("{\n"):])
    assert output == {"law": law, "scores": str(scores)}
