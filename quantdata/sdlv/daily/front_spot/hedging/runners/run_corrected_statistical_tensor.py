"""Run the missing demeaned one-day statistical hedging cells on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.reports.statistical_1d_matrix import (
    RISK_GRID,
)

DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data"))
PROJECT_ROOT = Path(__file__).resolve().parents[6]
DATASET = DATA_ROOT / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1"
DATASET_SHA256 = "58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a"
NOTIONAL_ARTIFACT = DATA_ROOT / "processing/hedging/target-notionals/daily-target-notionals-5b95b76536e96cfd8e271b66fd936ac5d89ebb8b7a4d06819a7a2a316039e6a7.json"
NOTIONAL_SHA256 = "1ba600aa7e5cd017cf79474f0f171361bf7fa19087401bf880089e293e8de3dc"
GPUCTL = Path(os.environ.get("GPUCTL", str(Path.home() / ".local/bin/gpuctl")))
EXISTING = {
    "covariance-entropy-1-demean": DATA_ROOT / "experiments/hedging/rolling-1d-20261017-r2/covariance-entropy-1-demean",
    "historical-sanos-entropy-1-demean": DATA_ROOT / "experiments/hedging/rolling-1d-20261017-historical-r3",
}


def _cases() -> list[tuple[str, str, float, str]]:
    cases = []
    for model in ("covariance", "historical-sanos"):
        for objective, risks in RISK_GRID.items():
            for risk in sorted(risks, key=lambda value: (value != 1.0, value)):
                case_id = f"{model}-{objective}-{risk:g}-demean"
                cases.append((model, objective, risk, case_id))
    return cases


def _resources_ok() -> bool:
    completed = subprocess.run(
        [str(GPUCTL), "resources", "--json"],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        print(f"WARN gpuctl resources unreadable: {completed.stderr.strip()}", flush=True)
        return False
    payload = json.loads(completed.stdout)
    checks = [(check["name"], check["health"]) for check in payload["checks"]]
    print(f"resources source={payload.get('source')} checks={checks}", flush=True)
    if payload.get("source") != "daemon":
        print("WARN gpuctl daemon unreachable; local resource reading used", flush=True)
    return all(health == "OK" for _, health in checks)


def _environment(output_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        QUANTDATA_ROOT=str(DATA_ROOT),
        PYTHONPATH=str(PROJECT_ROOT),
        MPLCONFIGDIR=str(output_root / "matplotlib"),
        PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    return environment


def _command(
    python: Path, *, model: str, objective: str, risk: float, output: Path
) -> list[str]:
    return [
        str(python), "-u", "-m",
        "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
        "--model", model,
        "--target-maturity", "1",
        "--objective", objective,
        "--risk-parameter", f"{risk:g}",
        "--mean-reduction", "demean",
        "--dataset", str(DATASET),
        "--dataset-manifest-sha256", DATASET_SHA256,
        "--target-notional-artifact-sha256", NOTIONAL_SHA256,
        "--start", "2023-01-01",
        "--end", "2025-12-31",
        "--progress-every", "5",
        "--output", str(output),
    ]


def _write_status(path: Path, entries: dict[str, dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2, sort_keys=True))
    temporary.replace(path)


def _report(output_root: Path, python: Path, environment: dict[str, str]) -> None:
    command = [
        str(python), "-u", "-m",
        "quantdata.sdlv.daily.front_spot.hedging.reports.statistical_1d_matrix",
        "--sanos-lv", str(DATA_ROOT / "experiments/hedging/daily/production-matrix-v3/sanos-lv-reference"),
        "--bs-greeks", str(DATA_ROOT / "experiments/hedging/rolling-1d-20261017-bs-greeks-unit-v1"),
        "--output", str(output_root / "reports"),
    ]
    for model, objective, risk, case_id in _cases():
        directory = EXISTING.get(case_id, output_root / "cases" / case_id)
        model_label = "Covariance" if model == "covariance" else "Historical-SANOS"
        command += ["--case", model_label, objective, f"{risk:g}", str(directory)]
    with (output_root / "report.log").open("w") as stream:
        subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers not in (1, 2, 3, 4):
        parser.error("--workers must be between one and four")
    output_root = arguments.output_root.resolve()
    if not output_root.is_relative_to(DATA_ROOT / "experiments/hedging"):
        parser.error("output root must be under the external hedging experiments directory")
    if hashlib.sha256(NOTIONAL_ARTIFACT.read_bytes()).hexdigest() != NOTIONAL_SHA256:
        raise RuntimeError("pinned target-notional artifact changed")
    environment = _environment(output_root)
    if arguments.dry_run:
        for model, objective, risk, case_id in _cases():
            print(case_id, "existing" if case_id in EXISTING else "queued")
        return
    (output_root / "cases").mkdir(parents=True, exist_ok=True)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(arguments.python), "-c", "from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import default_public_candidate_path; print(default_public_candidate_path())"],
        cwd=PROJECT_ROOT, env=environment, check=True,
    )
    if not _resources_ok():
        raise RuntimeError("resources not OK before corrected tensor launch")

    pending = [case for case in _cases() if case[3] not in EXISTING]
    entries: dict[str, dict[str, object]] = {
        case_id: {"status": "complete", "directory": str(directory)}
        for case_id, directory in EXISTING.items()
    }
    entries.update({case[3]: {"status": "queued"} for case in pending})
    status_path = output_root / "status.json"
    _write_status(status_path, entries)
    active: dict[str, tuple[subprocess.Popen[str], object]] = {}
    stop_requested = False
    failed = False
    last_resource_check = 0.0

    def stop_handler(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        print(f"stop requested signal={signum}", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    while pending or active:
        now = time.monotonic()
        if now - last_resource_check >= 300:
            if not _resources_ok():
                stop_requested = True
            last_resource_check = now
        while pending and len(active) < arguments.workers and not (stop_requested or failed):
            if not _resources_ok():
                stop_requested = True
                break
            model, objective, risk, case_id = pending.pop(0)
            directory = output_root / "cases" / case_id
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "job.log").open("a")
            command = _command(arguments.python, model=model, objective=objective, risk=risk, output=directory)
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            active[case_id] = (process, log)
            entries[case_id] = {"status": "running", "pid": process.pid, "directory": str(directory)}
            print(f"launched {case_id} pid={process.pid}", flush=True)
            _write_status(status_path, entries)
        for case_id, (process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            directory = output_root / "cases" / case_id
            records = len(list((directory / "records").glob("*.json")))
            entries[case_id] = {
                "status": "complete" if code == 0 else "failed",
                "exit_code": code,
                "completed_records": records,
                "directory": str(directory),
            }
            print(f"finished {case_id} exit={code} records={records}", flush=True)
            failed |= code != 0
            del active[case_id]
            _write_status(status_path, entries)
        if stop_requested:
            for process, _ in active.values():
                process.terminate()
            deadline = time.monotonic() + 60
            for process, _ in active.values():
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
            for case_id, (process, log) in active.items():
                log.close()
                entries[case_id] = {"status": "stopped", "exit_code": process.returncode}
            _write_status(status_path, entries)
            raise RuntimeError("corrected tensor stopped; completed origins remain resumable")
        if failed and not active:
            raise RuntimeError("corrected tensor has failed cells; inspect their job logs before resuming")
        time.sleep(30)
    _report(output_root, arguments.python, environment)
    print(f"corrected tensor complete; report={output_root / 'reports/STATISTICAL_SUMMARY.pdf'}", flush=True)


if __name__ == "__main__":
    main()
