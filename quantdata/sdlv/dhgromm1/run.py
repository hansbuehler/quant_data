"""Run expiry-return smoke validation, scenario generation and hedge training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths-per-sample", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warm-start-from", type=Path)
    parser.add_argument("--warm-start-long", type=Path)
    parser.add_argument("--policies", choices=["empty", "long", "short"], nargs="+", default=["empty", "long", "short"])
    parser.add_argument("--audit-report", type=Path, help="record the previously reviewed Q diagnostics, not a pass assertion")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.substeps < 1 or args.patience < 0:
        parser.error("substeps must be positive; patience must be nonnegative")
    output = args.output.resolve()
    if not output.is_relative_to(Path(os.environ["QUANTDATA_ROOT"]).resolve()):
        raise ValueError("pipeline output must be inside the external data root")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline.json"
    if status_path.exists():
        raise FileExistsError("pipeline directory already used; choose a unique run directory")
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    record = dict(status="running", started_unix=time.time(), sources=sources, completed_stages=[],
                  paths_per_sample=args.paths_per_sample, epochs=args.epochs, learning_rate=args.learning_rate,
                  simulation_substeps_per_day=args.substeps, trading_steps_per_day=1, horizon_days=20,
                  trained_policies=args.policies, warm_start_from=str(args.warm_start_from) if args.warm_start_from else None,
                  warm_start_long=str(args.warm_start_long) if args.warm_start_long else None,
                  instrument_menu="15 ITM calls, terminal ATM call, four co-terminal zero-strike calls, spot")
    if args.audit_report:
        audit = json.loads(args.audit_report.read_text())
        if (audit.get("status") != "complete" or audit.get("horizon_days") != 20
                or audit.get("dlv_substeps") != args.substeps):
            raise ValueError("Q diagnostic report has a different time geometry or is incomplete")
        record["reviewed_q_diagnostics"] = dict(path=str(args.audit_report.resolve()),
            sha256=hashlib.sha256(args.audit_report.read_bytes()).hexdigest(),
            unconditional=audit["unconditional"]["all_instruments"],
            conditional_z={key: value["combined_out_of_fold"]["z_score"] for key, value in audit["conditional"].items()},
            interpretation="Reviewed diagnostics only; no no-arbitrage pass is asserted")

    def publish(**fields: object) -> None:
        record.update(fields)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(fields), flush=True)

    def stage(name: str, module: str, options: list[str]) -> None:
        publish(stage=name)
        environment = dict(os.environ)
        environment["JAX_PLATFORMS"] = args.device
        subprocess.run([sys.executable, "-u", "-m", module, *options], env=environment, check=True)
        record["completed_stages"].append(name)
        publish(last_completed_stage=name)

    def scenarios(folder: Path, *, paths: int, batch: int, chunk: int) -> list[str]:
        return ["--calibration", str(args.calibration.resolve()), "--output", str(folder),
                "--paths-per-sample", str(paths), "--batch-size", str(batch), "--chunk-size", str(chunk),
                "--substeps", str(args.substeps)]

    def training(folder: Path, target: Path, *, epochs: int, batch: int) -> list[str]:
        options = ["--scenarios", str(folder), "--output", str(target), "--device", args.device,
                   "--epochs", str(epochs), "--batch-size", str(batch), "--learning-rate", str(args.learning_rate),
                   "--patience", str(args.patience), "--policies", *args.policies]
        if args.warm_start_from:
            options += ["--warm-start-from", str(args.warm_start_from.resolve())]
        if args.warm_start_long:
            options += ["--warm-start-long", str(args.warm_start_long.resolve())]
        return options

    publish(stage="initializing")
    try:
        stage("smoke-scenarios", "quantdata.sdlv.dhgromm1.rolling_simulation",
              scenarios(output / "smoke/scenarios", paths=4, batch=4, chunk=4))
        stage("smoke-training", "quantdata.sdlv.dhgromm1.training",
              training(output / "smoke/scenarios", output / "smoke/training", epochs=2, batch=4))
        smoke = json.loads((output / "smoke/training/report.json").read_text())
        if smoke["status"] != "complete" or not all(math.isfinite(smoke[k]) for k in ("bid", "ask")):
            raise RuntimeError("smoke validation did not produce finite complete hedge results")
        stage("full-scenarios", "quantdata.sdlv.dhgromm1.rolling_simulation",
              scenarios(output / "scenarios", paths=args.paths_per_sample, batch=128, chunk=4000))
        stage("full-training", "quantdata.sdlv.dhgromm1.training",
              training(output / "scenarios", output / "training", epochs=args.epochs, batch=1024))
        publish(stage="plot-losses")
        environment = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        subprocess.run([sys.executable, "-u", "-m", "quantdata.sdlv.dhgromm1.plot_losses",
            "--run", str(output), "--output", str(output / "diagnostics/loss-curves")], env=environment, check=True)
        record["completed_stages"].append("plot-losses")
        publish(status="complete", stage="complete", finished_unix=time.time(),
                report=str(output / "training/report.json"), graph=str(output / "diagnostics/loss-curves/losses.png"))
    except BaseException as error:
        publish(status="failed", error=repr(error), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
