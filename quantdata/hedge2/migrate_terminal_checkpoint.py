"""Copy-only checkpoint migrations with exact reviewed runtime transformations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from quantdata.hedge2.terminal_policy import TerminalPolicyConfig

PRE_SCHEDULE_TRAINING_DEPENDENCIES = ("terminal_train.py", "terminal_policy.py", "terminal_features.py", "terminal_torch.py", "terminal.py", "terminal_descent.py", "terminal_coordinates.py")
TRAINING_DEPENDENCIES = (*PRE_SCHEDULE_TRAINING_DEPENDENCIES[:-1], "terminal_schedule.py", "terminal_coordinates.py")
PILOT_RUNTIME_DEPENDENCIES = (*TRAINING_DEPENDENCIES[1:], "terminal_policy_pilot.py",
    "terminal_market.py", "claims.py", "cash.py", "pricing.py", "batch_pricing.py")
REVIEWED_SCHEDULE_SHA256 = "690a2f54ed65881cd0312dedcc1ed422d3678c99baeea87aa47793c531ef9c6f"


def migrate_host_rng_checkpoint(*, prior_directory: Path, output_directory: Path,
                                allow_unchanged_runtime: bool = False,
                                allow_scheduler_correction: bool = False) -> dict[str, object]:
    """Preserve financial/model/Adam/RNG state and the original run.

    The exact known CPU-deserialization change is admissible in the trainer.
    An explicit unchanged-runtime mode also permits copying an already
    repaired checkpoint after unrelated project files change. Training,
    simulation, pricing and pilot dependencies must remain bitwise identical.
    A separate explicit scheduler-correction mode admits only the reviewed
    signed-loss threshold transformation. It resets the scheduler's plateau
    counters/best reference, not the Adam state or its current learning rate.
    The new run records old/new code and checkpoint hashes;
    ordinary resumes continue to reject unreviewed implementation changes.
    """
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Migrated terminal runs must be outside Git working trees")
    if output_directory.exists():
        raise FileExistsError("A terminal checkpoint migration needs a new, untouched output directory")
    configuration = json.loads((prior_directory / "config.json").read_text())
    expected = configuration["implementation_sha256"]
    with ZipFile(prior_directory / "implementation.zip") as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Prior terminal source archive does not match its recorded implementation")
        previous_sources = {name: archive.read(name) for name in expected}
    if any(hashlib.sha256(content).hexdigest() != expected[name] for name, content in previous_sources.items()):
        raise ValueError("A frozen prior terminal source changed")
    package = Path(__file__).parent
    previous_loader = b'values = torch.load(path, map_location=env.device, weights_only=True)'
    current_loader = b'values = torch.load(path, map_location="cpu", weights_only=True)'
    previous_trainer = previous_sources["terminal_train.py"]
    current_trainer = (package / "terminal_train.py").read_bytes()
    unchanged = previous_trainer == current_trainer
    loader_changed = previous_trainer.count(previous_loader) == 1
    transformed = previous_trainer.replace(previous_loader, current_loader) if loader_changed else previous_trainer
    scheduler_changed = False
    if allow_scheduler_correction and transformed != current_trainer:
        changes = (
            (b'from quantdata.hedge2.terminal_torch import (',
             b'from quantdata.hedge2.terminal_schedule import step_training_scheduler\nfrom quantdata.hedge2.terminal_torch import ('),
            (b'for name in ("terminal_train.py", "terminal_policy.py", "terminal_features.py", "terminal_torch.py", "terminal.py", "terminal_descent.py", "terminal_coordinates.py")',
             b'for name in ("terminal_train.py", "terminal_policy.py", "terminal_features.py", "terminal_torch.py", "terminal.py", "terminal_descent.py", "terminal_schedule.py", "terminal_coordinates.py")'),
            (b'threshold=train_config.relative_loss_tolerance)',
             b'threshold=train_config.relative_loss_tolerance, threshold_mode="abs")'),
            (b'                    scheduler.step(current_loss)',
             b'                    step_training_scheduler(scheduler, current_loss, tolerance=train_config.relative_loss_tolerance)'),
        )
        if all(transformed.count(old) == 1 for old, _ in changes):
            for old, new in changes:
                transformed = transformed.replace(old, new)
            scheduler_changed = transformed == current_trainer
    if not (allow_unchanged_runtime and unchanged) and not (
            (loader_changed or scheduler_changed) and transformed == current_trainer):
        raise ValueError("Migration permits only explicit unchanged runtime or exact reviewed host-RNG/scheduler changes")
    if scheduler_changed and hashlib.sha256((package / "terminal_schedule.py").read_bytes()).hexdigest() != REVIEWED_SCHEDULE_SHA256:
        raise ValueError("Unreviewed introduced terminal scheduler implementation")
    for name in PILOT_RUNTIME_DEPENDENCIES:
        if name == "terminal_schedule.py" and scheduler_changed and name not in previous_sources:
            continue
        if name not in previous_sources:
            raise ValueError(f"Unreviewed missing terminal implementation: {name}")
        if previous_sources[name] != (package / name).read_bytes():
            raise ValueError(f"Unreviewed terminal implementation change: {name}")
    candidates = tuple(path for path in prior_directory.glob("*.pt") if not path.name.endswith(".partial.pt"))
    if len(candidates) != 1:
        raise ValueError("A prior terminal policy run must identify exactly one saved checkpoint")
    prior_checkpoint = candidates[0]
    checkpoint_bytes = prior_checkpoint.read_bytes()
    values = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    prior_dependencies = PRE_SCHEDULE_TRAINING_DEPENDENCIES if scheduler_changed else TRAINING_DEPENDENCIES
    if (values["code_version"] != TerminalPolicyConfig.version.unique_id48
            or values["user_version"] != configuration["model_config"]["user_version"]
            or values["identity"]["implementation_sha256"] != {name: expected[name] for name in prior_dependencies}):
        raise ValueError("Prior terminal checkpoint versions or implementation identity do not match")
    scheduler_reset = None
    if scheduler_changed:
        scheduler = values["scheduler"]
        tolerance = values["train_config"]["relative_loss_tolerance"]
        if (values["status"] == "training_stable" or scheduler["mode"] != "min"
                or scheduler["threshold_mode"] != "rel" or not math.isfinite(values["best_loss"])
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Reviewed scheduler correction requires an unfinished minimum-risk relative-threshold checkpoint")
        before = dict(scheduler)
        scheduler.update(threshold_mode="abs", threshold=tolerance * max(1., abs(values["best_loss"])),
            best=values["best_loss"], num_bad_epochs=0, cooldown_counter=0)
        scheduler_reset = {"before": before, "after": dict(scheduler)}
    current_sources = {path.name: path.read_bytes() for path in sorted(package.glob("*.py"))}
    current_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in current_sources.items()}
    configuration["implementation_sha256"] = current_hashes
    values["identity"]["implementation_sha256"] = {name: current_hashes[name] for name in TRAINING_DEPENDENCIES}
    output_directory.mkdir(parents=True)
    checkpoint = output_directory / prior_checkpoint.name
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save(values, temporary)
    temporary.replace(checkpoint)
    (output_directory / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name, content in current_sources.items():
            archive.writestr(name, content)
    for name in ("metrics.json", "split.npz"):
        if (prior_directory / name).exists():
            destination = "prior_metrics.json" if name == "metrics.json" else name
            shutil.copyfile(prior_directory / name, output_directory / destination)
    metadata = {"migration": ("signed_loss_scheduler_correction" if scheduler_changed else
        "unchanged_runtime_copy" if unchanged else "host_rng_checkpoint_deserialization"),
        "prior_directory": str(prior_directory.resolve()), "prior_checkpoint": str(prior_checkpoint),
        "prior_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "prior_implementation_sha256": expected, "implementation_sha256": current_hashes,
        "preserved_updates": values["updates"], "preserved_status": values["status"],
        "financial_or_optimizer_change": scheduler_changed,
        "financial_objective_change": False, "optimizer_schedule_change": scheduler_changed,
        "adam_state_and_current_learning_rate_preserved": True, "scheduler_reset": scheduler_reset,
        "host_rng_loader_changed": loader_changed,
        "reason": ("Align signed-loss scheduler thresholds with max(1, abs(best)); reset plateau/cooldown counters, preserve all Adam/model/RNG/data state"
            if scheduler_changed else "Training, simulation, pricing and pilot runtime are unchanged; refresh unrelated archived source metadata"
            if unchanged else "GPU RNG state must remain a host ByteTensor; model and optimizer move explicitly on restore")}
    (output_directory / "restart_migration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-unchanged-runtime", action="store_true")
    parser.add_argument("--allow-scheduler-correction", action="store_true")
    arguments = parser.parse_args()
    metadata = migrate_host_rng_checkpoint(prior_directory=arguments.prior_directory,
        output_directory=arguments.output_directory, allow_unchanged_runtime=arguments.allow_unchanged_runtime,
        allow_scheduler_correction=arguments.allow_scheduler_correction)
    print(json.dumps({key: metadata[key] for key in ("migration", "checkpoint", "preserved_updates", "preserved_status")}), flush=True)


if __name__ == "__main__":
    main()
