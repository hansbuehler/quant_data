"""A host-RNG placement fix cannot silently change training state or financial code."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from quantdata.hedge2 import migrate_terminal_checkpoint
from quantdata.hedge2.migrate_terminal_checkpoint import (
    PRE_SCHEDULE_TRAINING_DEPENDENCIES,
    TRAINING_DEPENDENCIES,
    migrate_host_rng_checkpoint,
)
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig


def prior_run(root: Path, *, change: str | None = None, old_loader: bool = True,
              old_scheduler: bool = False) -> Path:
    directory = root / "prior"
    directory.mkdir()
    package = Path(migrate_terminal_checkpoint.__file__).parent
    sources = {path.name: path.read_bytes() for path in package.glob("*.py")}
    if old_scheduler:
        del sources["terminal_schedule.py"]
        sources["terminal_train.py"] = (sources["terminal_train.py"]
            .replace(b'from quantdata.hedge2.terminal_schedule import step_training_scheduler\n', b'')
            .replace(b', "terminal_schedule.py"', b'')
            .replace(b'threshold=train_config.relative_loss_tolerance, threshold_mode="abs")',
                     b'threshold=train_config.relative_loss_tolerance)')
            .replace(b'step_training_scheduler(scheduler, current_loss, tolerance=train_config.relative_loss_tolerance)',
                     b'scheduler.step(current_loss)'))
    if old_loader:
        sources["terminal_train.py"] = sources["terminal_train.py"].replace(
            b'values = torch.load(path, map_location="cpu", weights_only=True)',
            b'values = torch.load(path, map_location=env.device, weights_only=True)')
    if change is not None:
        sources[change] += b"\n"
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    configuration = {"implementation_sha256": hashes, "model_config": {"user_version": "0.1"}}
    (directory / "config.json").write_text(json.dumps(configuration))
    with ZipFile(directory / "implementation.zip", "w") as archive:
        for name, content in sources.items():
            archive.writestr(name, content)
    dependencies = PRE_SCHEDULE_TRAINING_DEPENDENCIES if old_scheduler else TRAINING_DEPENDENCIES
    values = {"identity": {"implementation_sha256": {name: hashes[name] for name in dependencies},
        "data_sha256": "unchanged", "context": {"risk_aversion": 10.}},
        "code_version": TerminalPolicyConfig.version.unique_id48, "user_version": "0.1",
        "updates": 10000, "status": "training_limit", "model": {"weight": torch.tensor([.2, .4])},
        "optimizer": {"momentum": torch.tensor([.1, .3]), "lr": 3.9e-6}, "torch_rng": torch.get_rng_state(),
        "best_loss": -.1, "stable_checks": 4, "train_config": {"relative_loss_tolerance": 1e-6},
        "scheduler": {"mode": "min", "threshold_mode": "rel", "threshold": 1e-6,
            "best": -.09, "num_bad_epochs": 3, "cooldown_counter": 2,
            "last_epoch": 20, "_last_lr": [3.9e-6]}}
    torch.save(values, directory / "policy.pt")
    return directory


def assert_same_state(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_same_state(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same_state(left, right)
    else:
        assert actual == expected


def test_migration_is_copy_only_and_preserves_model_optimizer_rng_and_data_identity(tmp_path: Path) -> None:
    prior = prior_run(tmp_path)
    original = (prior / "policy.pt").read_bytes()
    before = torch.load(prior / "policy.pt", weights_only=True)
    result = migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "migrated")
    after = torch.load(result["checkpoint"], weights_only=True)
    assert (prior / "policy.pt").read_bytes() == original
    for category in ("model", "optimizer"):
        for name in before[category]:
            torch.testing.assert_close(before[category][name], after[category][name], rtol=0., atol=0.)
    torch.testing.assert_close(before["torch_rng"], after["torch_rng"], rtol=0., atol=0.)
    assert before["identity"]["data_sha256"] == after["identity"]["data_sha256"]
    assert before["identity"]["context"] == after["identity"]["context"]
    assert result["preserved_updates"] == 10000
    assert not result["financial_or_optimizer_change"]
    assert (tmp_path / "migrated" / "restart_migration.json").is_file()


@pytest.mark.parametrize("changed", ["terminal_train.py", "terminal_policy.py", "terminal_policy_pilot.py", "terminal_descent.py", "terminal_coordinates.py"])
def test_unreviewed_source_changes_are_not_migrated(tmp_path: Path, changed: str) -> None:
    prior = prior_run(tmp_path, change=changed)
    with pytest.raises(ValueError, match="reviewed|Unreviewed"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("old_loader", (False, True))
def test_explicit_scheduler_migration_resets_only_reviewed_schedule_state(tmp_path: Path, old_loader: bool) -> None:
    prior = prior_run(tmp_path, old_loader=old_loader, old_scheduler=True)
    original = (prior / "policy.pt").read_bytes()
    before = torch.load(prior / "policy.pt", weights_only=True)
    with pytest.raises(ValueError, match="reviewed"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "implicit")
    result = migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "explicit",
        allow_scheduler_correction=True)
    after = torch.load(result["checkpoint"], weights_only=True)
    assert (prior / "policy.pt").read_bytes() == original
    assert result["migration"] == "signed_loss_scheduler_correction"
    assert result["optimizer_schedule_change"] and not result["financial_objective_change"]
    assert result["host_rng_loader_changed"] == old_loader
    for name in ("best_loss", "updates", "stable_checks", "status", "train_config"):
        assert after[name] == before[name]
    for category in ("model", "optimizer"):
        for name in before[category]:
            if isinstance(before[category][name], torch.Tensor):
                torch.testing.assert_close(after[category][name], before[category][name], rtol=0., atol=0.)
            else:
                assert after[category][name] == before[category][name]
    torch.testing.assert_close(after["torch_rng"], before["torch_rng"], rtol=0., atol=0.)
    assert after["identity"]["data_sha256"] == before["identity"]["data_sha256"]
    assert after["scheduler"] == {**before["scheduler"], "best": -.1, "num_bad_epochs": 0,
        "cooldown_counter": 0, "threshold_mode": "abs", "threshold": 1e-6}
    assert set(after["identity"]["implementation_sha256"]) == set(TRAINING_DEPENDENCIES)
    permitted = deepcopy(before)
    permitted["identity"]["implementation_sha256"] = after["identity"]["implementation_sha256"]
    permitted["scheduler"] = after["scheduler"]
    assert_same_state(after, permitted)


@pytest.mark.parametrize("changed", ("terminal_train.py", "terminal_policy.py", "terminal_market.py", "pricing.py"))
def test_scheduler_correction_cannot_cover_other_runtime_changes(tmp_path: Path, changed: str) -> None:
    prior = prior_run(tmp_path, old_scheduler=True, change=changed)
    with pytest.raises(ValueError, match="reviewed|Unreviewed"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "rejected",
            allow_scheduler_correction=True)
    assert not (tmp_path / "rejected").exists()


def test_already_stable_results_are_not_reinterpreted_by_scheduler_migration(tmp_path: Path) -> None:
    prior = prior_run(tmp_path, old_scheduler=True)
    values = torch.load(prior / "policy.pt", weights_only=True)
    values["status"] = "training_stable"
    torch.save(values, prior / "policy.pt")
    with pytest.raises(ValueError, match="unfinished"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "rejected",
            allow_scheduler_correction=True)


def test_scheduler_migration_rejects_unreviewed_new_helper(monkeypatch, tmp_path: Path) -> None:
    prior = prior_run(tmp_path, old_scheduler=True)
    original_reader = Path.read_bytes
    helper = Path(migrate_terminal_checkpoint.__file__).parent / "terminal_schedule.py"
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_reader(path) + (b"\n" if path == helper else b""))
    with pytest.raises(ValueError, match="Unreviewed introduced"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "rejected",
            allow_scheduler_correction=True)
    assert not (tmp_path / "rejected").exists()


def test_an_existing_destination_cannot_be_overwritten(tmp_path: Path) -> None:
    prior = prior_run(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    retained = target / "user.txt"
    retained.write_text("keep")
    with pytest.raises(FileExistsError):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=target)
    assert retained.read_text() == "keep"


def test_unchanged_runtime_copy_requires_explicit_mode_and_preserves_latest_state(tmp_path: Path) -> None:
    prior = prior_run(tmp_path, old_loader=False, change="terminal_roll.py")
    original = (prior / "policy.pt").read_bytes()
    with pytest.raises(ValueError, match="unchanged runtime"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "implicit")
    result = migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "explicit",
        allow_unchanged_runtime=True)
    assert result["migration"] == "unchanged_runtime_copy"
    assert not result["financial_or_optimizer_change"]
    assert (prior / "policy.pt").read_bytes() == original
    before, after = (torch.load(path, weights_only=True) for path in (prior / "policy.pt", Path(result["checkpoint"])))
    torch.testing.assert_close(before["model"]["weight"], after["model"]["weight"], rtol=0., atol=0.)
    torch.testing.assert_close(before["optimizer"]["momentum"], after["optimizer"]["momentum"], rtol=0., atol=0.)
    torch.testing.assert_close(before["torch_rng"], after["torch_rng"], rtol=0., atol=0.)
    assert after["updates"] == before["updates"] == 10000


@pytest.mark.parametrize("changed", ("terminal_train.py", "terminal_market.py", "pricing.py", "terminal_policy_pilot.py"))
def test_unchanged_mode_does_not_admit_changed_runtime(tmp_path: Path, changed: str) -> None:
    prior = prior_run(tmp_path, old_loader=False, change=changed)
    with pytest.raises(ValueError, match="reviewed|Unreviewed|unchanged"):
        migrate_host_rng_checkpoint(prior_directory=prior, output_directory=tmp_path / "rejected",
            allow_unchanged_runtime=True)
    assert not (tmp_path / "rejected").exists()
