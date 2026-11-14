"""Causal dated model training for the fixed 44-origin paper evaluation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from filelock import FileLock

from quantdata.hedging.model_audit.run import _hash, _json

REPORT_CASES = ("DYSANOS v1-P", "DYSANOS v2-P", "GROMM v1-P", "GROMM v1-Q", "GROMM v2-P", "GROMM v2-Q")
INITIALIZATION_POLICY = "independent rolling windows; no preceding-origin fitted weights"


def origins(reference: Path) -> tuple[str, ...]:
    selected = tuple(json.loads((reference / "observable_reference.json").read_text())["dates"])
    if len(selected) != 44 or sorted(set(selected)) != list(selected) or selected[0] != "2024-04-01" or selected[-1] != "2024-05-31":
        raise ValueError("Expected the unchanged 44 April-May forecast origins")
    return selected


def gromm_origin(root: Path, reference: Path, *, family: str, day: str, source: Path) -> None:
    """Train one origin; no future representation or all-history weights are loaded."""
    if (root / "INCLUDE_DYSANOS.json").exists():
        dispatch_dysanos(root, reference, source=source, through=day)
    from openqdata.biz_calendar import get_business_days
    from openqdata.models.gromm import gromm1, gromm2
    from openqdata.models.gromm.config import FitConfig, TrainingConfig
    from openqdata.models.gromm.data import LicensedDataset
    from openqdata.models.gromm.dynamics import estimate_origin, load_dynamics
    from openqdata.models.gromm.progress import Heartbeat
    from openqdata.models.gromm.rolling import fit_origin, load_representation

    from quantdata.hedging.model_audit.gromm_probability import run as probability_refit

    selected = origins(reference)
    index = selected.index(day)
    directory = root / f"fits/{family}/{day}"
    directory.mkdir(parents=True, exist_ok=True)
    origin = date.fromisoformat(day)
    dataset = LicensedDataset(source / "data/processed", probe_day=origin, sanos_root=source)
    sessions = get_business_days(date(2020, 1, 1), date(2024, 5, 31))
    config = (gromm1 if family == "gromm1" else gromm2).Config()
    training = TrainingConfig(fit=FitConfig(linear_solver="pcr"), chunk_epochs=1, checkpoint_epochs=10)
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, family=family, origin=day, origin_index=index + 1,
            origins_total=44, elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        emit({"phase": "checkpoint_stop_requested", "signal": signum})

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with FileLock(directory / ".training.lock"), Heartbeat(emit, interval=60.) as progress:
        representation_path = directory / "representation.npz"
        if not representation_path.exists():
            model = fit_origin(dataset, origin, config, training, root / "fits", sessions=sessions,
                previous=None, progress=progress.update, should_stop=lambda: stopping)
        else:
            model = load_representation(representation_path)
        if model.origin != origin or any(after >= origin for before, after in model.endpoints.pairs):
            raise ValueError("Noncausal or incorrectly dated fixed-contract representation")
        dynamics_path = directory / "dynamics-unweighted.npz"
        if not dynamics_path.exists():
            estimate_origin(representation_path, dataset, sessions, dynamics_path,
                previous=None, progress=progress.update)
        else:
            load_dynamics(dynamics_path)
        retained = dynamics_path
        if family == "gromm1":
            retained = directory / "probability-weighted/dynamics.npz"
            if not retained.exists():
                probability_refit(representation_path, dynamics_path, retained.parent,
                    previous_path=None,
                    scope="44-origin causal rolling April-May comparison; not the single-origin or all-history controls")
        fitted = load_dynamics(retained)
        _json(directory / "COMPLETE.json", {"family": family, "origin": day,
            "representation": str(representation_path), "representation_sha256": _hash(representation_path),
            "dynamics": str(retained), "dynamics_sha256": _hash(retained), "identity": fitted.identity,
            "training_dates": [str(value) for value in model.dates],
            "maximum_transition_endpoint": max(str(after) for before, after in model.endpoints.pairs),
            "initialization_origin": None, "initialization_policy": INITIALIZATION_POLICY,
            "worker_source_sha256": _hash(Path(__file__)),
            "scope": "causal rolling; no all-history or later-origin initialization", "measures": ["P", "Q"],
            "forecast_statistics": "pending same-reference evaluation; fit completion is not a statistics pass"})
        progress.update({"phase": "causal_origin_complete", "completed": index + 1, "total": 44})


def dispatch_dysanos(root: Path, reference: Path, *, source: Path, through: str) -> None:
    """Run ready DYSANOS workers inside the same allocation-owner process tree."""
    for day in origins(reference):
        if day > through:
            break
        for family in ("dysanos1", "dysanos2"):
            destination = root / f"fits/{family}/{day}"
            if (destination / "COMPLETE.json").exists():
                continue
            if not (root / f"prepared/{family}/COMPLETE.json").exists():
                continue
            if family == "dysanos2":
                from datetime import date

                from openqdata.biz_calendar import get_business_days

                from quantdata.hedging.model_audit.rolling_dysanos import window_dates

                window = window_dates(get_business_days(date(2020, 1, 1), date.fromisoformat(day)), date.fromisoformat(day))
                if not all((root / f"calibrated/dysanos2/{point}.npz").exists() for point in window):
                    continue
            print(f"dispatch causal family={family} origin={day} from allocation-owned rolling worker", flush=True)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = environment["PYTHONPATH"].replace(
                str(root / "source/dysanos/src"), str(root / "source/dysanos_controlled/src"))
            command = [sys.executable, "-u", "-m", "quantdata.hedging.model_audit.rolling_dysanos",
                "--root", str(root), "--reference", str(reference), "--source", str(source), "--family", family, "--origin", day]
            result = subprocess.run(command, env=environment)
            if result.returncode:
                raise RuntimeError(f"Causal {family} {day} stopped with return code {result.returncode}; preserve exact log")


def resource_health() -> list[str]:
    check = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True)
    if check.returncode:
        print("WARN: gpuctl resources could not be read", flush=True)
        return ["WARN"]
    return [value["health"] for value in json.loads(check.stdout)["checks"]]


def queue(root: Path, reference: Path, *, source: Path, families: tuple[str, ...]) -> None:
    """One isolated GPU worker at a time, independently fitted at every origin."""
    selected = origins(reference)
    if set(families) - {"gromm1", "gromm2"}:
        raise ValueError("This queue trains only the two explicit GROMM versions")
    protocol = {"schema": "causal-versioned-44-origins-v1", "report_cases": REPORT_CASES,
        "origins": selected, "reference_sha256": _hash(reference / "reference_history.npz"),
        "frozen_observable_pca_sha256": _hash(reference / "frozen_observable_pca.npz"),
        "source": str(source), "gromm1_pca": "probability-weighted; market calibration unchanged",
        "gromm2_pca": "native specified innovation PCA", "native_ranks": {"dysanos1": 200, "dysanos2": 16, "gromm1": 16, "gromm2": 16},
        "statistics": "same 49 floating log-IV coordinates, observed outcomes, frozen PCA, covariance/forecast-MSE/tails/quotes/stability definitions as the existing PDF",
        "no_arbitrage_order": "after the rolling comparison; all models, only Q laws screened for dynamic zero means",
        "initialization_policy": INITIALIZATION_POLICY,
        "source_sha256": _hash(Path(__file__))}
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "PROTOCOL.json"
    if destination.exists() and json.loads(destination.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Rolling run identity changed; use a new run directory")
    _json(destination, protocol)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for day in selected:
        for family in families:
            completion = root / f"fits/{family}/{day}/COMPLETE.json"
            if completion.exists():
                continue
            health = resource_health()
            if stopping or any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in health):
                raise InterruptedError("Rolling queue stopped before another origin")
            print(f"launch family={family} origin={day} causal_index={selected.index(day)+1}/44", flush=True)
            command = [sys.executable, "-u", "-m", "quantdata.hedging.model_audit.rolling_versions", "--root", str(root), "--reference", str(reference),
                "--source", str(source), "--worker", family, "--origin", day]
            process = subprocess.Popen(command, env=os.environ.copy())
            while process.poll() is None:
                try:
                    process.wait(timeout=60.)
                except subprocess.TimeoutExpired:
                    health = resource_health()
                    if stopping or any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in health):
                        process.terminate()
                        try:
                            process.wait(timeout=60.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise InterruptedError("Active rolling origin stopped; retain last periodic checkpoint")
            if process.returncode:
                _json(root / "FAILURE.json", {"family": family, "origin": day, "returncode": process.returncode,
                    "action": "stopped; exact worker error is retained in persistent log; not an accepted fit"})
                raise RuntimeError(f"{family} {day} failed; inspect persistent log before resuming")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["gromm1", "gromm2"])
    parser.add_argument("--worker", choices=("gromm1", "gromm2"))
    parser.add_argument("--origin")
    args = parser.parse_args()
    if args.worker:
        gromm_origin(args.root, args.reference, family=args.worker, day=args.origin, source=args.source)
    else:
        queue(args.root, args.reference, source=args.source, families=tuple(args.families))


if __name__ == "__main__":
    main()
