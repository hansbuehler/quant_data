"""Continue bid and ask from their epoch checkpoints on the existing paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, default=500)
    parser.add_argument("--target-epochs", type=int)
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    source, output = args.resume_run.resolve(), args.output.resolve()
    if not output.is_relative_to(Path(os.environ["QUANTDATA_ROOT"]).resolve()) or output == source:
        raise ValueError("use a new output directory inside the external project data root")
    if args.additional_epochs < 1:
        raise ValueError("additional epochs must be positive")
    prior_path = source / "training/report.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior.get("status") != "complete":
            raise ValueError("source training report is incomplete")
        scenarios = Path(prior["scenarios"]).resolve()
        settings = prior["training_config"]
    elif args.resume_incomplete:
        import torch

        prior = json.loads((source / "pipeline.json").read_text())
        scenarios = Path(prior.get("scenarios", source / "scenarios")).resolve()
        settings = torch.load(source / "training/long.checkpoint.pt", map_location="cpu", weights_only=False)["train_config"]
    else:
        raise ValueError("a completed source report or explicit --resume-incomplete is required")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline.json"
    if status_path.exists():
        raise FileExistsError("continuation output already used")
    scenario_identity = json.loads((scenarios / "identity.json").read_text())
    record = dict(status="running", started_unix=time.time(), resume_run=str(source), scenarios=str(scenarios),
                  additional_epochs=args.additional_epochs if args.target_epochs is None else None,
                  trained_policies=["long", "short"],
                  target_epochs=args.target_epochs, recovered_incomplete=args.resume_incomplete,
                  simulation_substeps_per_day=scenario_identity["substeps"], trading_steps_per_day=1, horizon_days=20,
                  early_stopping=False, completed_stages=[],
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    if args.resume_incomplete:
        import torch

        record["resume_epochs"] = {label: torch.load(source / f"training/{label}.checkpoint.pt",
            map_location="cpu", weights_only=False)["epoch"] + 1 for label in ("empty", "long", "short")}
    source_pipeline = json.loads((source / "pipeline.json").read_text())
    if source_pipeline.get("reviewed_q_diagnostics"):
        record["reviewed_q_diagnostics"] = source_pipeline["reviewed_q_diagnostics"]

    def publish(**fields: object) -> None:
        record.update(fields)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(fields), flush=True)

    publish(stage="initializing")
    try:
        publish(stage="full-training")
        options = [sys.executable, "-u", "-m", "quantdata.sdlv.dhgromm1.training",
            "--scenarios", str(scenarios), "--output", str(output / "training"),
            "--resume-from", str(source / "training"), "--policies", "long", "short",
            "--epochs", str(args.additional_epochs), "--patience", "0", "--device", args.device,
            "--learning-rate", str(settings["learning_rate"]), "--batch-size", str(settings["batch_size"]),
            "--seed", str(settings["seed"])]
        if args.target_epochs is not None:
            options += ["--target-epochs", str(args.target_epochs)]
        if args.resume_incomplete:
            options += ["--resume-incomplete"]
        subprocess.run(options, check=True)
        record["completed_stages"].append("full-training")
        publish(stage="plot-losses")
        environment = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        subprocess.run([sys.executable, "-u", "-m", "quantdata.sdlv.dhgromm1.plot_losses",
            "--run", str(output), "--output", str(output / "diagnostics/loss-curves")],
            env=environment, check=True)
        record["completed_stages"].append("plot-losses")
        publish(status="complete", stage="complete", finished_unix=time.time(),
                report=str(output / "training/report.json"), graph=str(output / "diagnostics/loss-curves/losses.png"))
    except BaseException as error:
        publish(status="failed", error=repr(error), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
