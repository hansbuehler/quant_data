"""Paper-matched restricted Q audits of an explicitly identified wider-grid fit.

This driver does not change the Q dynamics or the policy objectives. It supplies
the wider historical fits to the existing paper estimator and simulator, then
uses the original entropy and squared policy optimizers on the same population.
All outputs are isolated from the retained narrow-grid paper results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import jax
import numpy as np
import torch
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot import run_oce as entropy
from quantdata.sdlv.daily.front_spot import run_quadratic_oce as squared
from quantdata.sdlv.daily.front_spot.calibration import calibrate_front_spot_model
from quantdata.sdlv.daily.front_spot.estimation import (
    JointInnovationFit,
    align_fixed_contract_history,
    fit_joint_innovations,
    rolling_history_indices,
)
from quantdata.sdlv.daily.front_spot.oce import paired_path_average_statistics
from quantdata.sdlv.daily.front_spot.run import _calibration_summary
from quantdata.sdlv.daily.front_spot.simulation import (
    InitialModelState,
    load_initial_model_state,
)

jax.config.update("jax_enable_x64", True)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(**record: object) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


def resource_health() -> list[dict]:
    """Use only the host resource manager for headroom decisions."""
    result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True,
                            text=True, check=True, timeout=15)
    checks = json.loads(result.stdout)["checks"]
    if not checks:
        raise RuntimeError("Resource manager returned no checks")
    return [dict(name=check["name"], health=check["health"]) for check in checks]


class AuditMonitor:
    """Progress during compilation and stop-on-pressure without touching other jobs."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.started = time.monotonic()
        self.done = threading.Event()
        self.record: dict = dict(stage="initializing", completed=0, total=1)
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.done.wait(60):
            elapsed = time.monotonic() - self.started
            record = dict(self.record)
            emit(event="heartbeat", **record, elapsed_seconds=elapsed)
            try:
                unhealthy = [check for check in resource_health() if check["health"] != "OK"]
                if unhealthy:
                    emit(event="resource_alert", checks=unhealthy)
                if any(check["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for check in unhealthy):
                    emit(event="graceful_stop", message="Prior complete population chunks and policy checkpoints are recovery points")
                    os.kill(os.getpid(), signal.SIGTERM)
                    if not self.done.wait(60):
                        os.kill(os.getpid(), signal.SIGKILL)
                    return
            except Exception as error:
                emit(event="resource_warning", message=str(error))

    def stage(self, name: str, completed: int = 0, total: int = 1) -> None:
        self.record = dict(stage=name, completed=completed, total=total)
        entropy._atomic_json(dict(**self.record, elapsed_seconds=time.monotonic() - self.started),
                             self.output / "progress.json")
        emit(**self.record)


def load_source(run_directory: Path, output: Path, snapshot: Path | None) -> tuple[np.ndarray, dict]:
    """Freeze weights, verify dataset identity, and distinguish smoke from final fits."""
    dataset = run_directory / "market.npz"
    dataset_digest = sha256(dataset)
    frozen = output / "initial-fit.npz"
    if snapshot is not None:
        if not frozen.exists():
            shutil.copy2(snapshot, frozen)
        with np.load(frozen, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity"].item()))
            raw = np.asarray(saved["raw"], dtype=np.float32)
            epoch = int(saved["epoch"])
        model_label = identity["model_label"]
        if identity["dataset_sha256"] != dataset_digest:
            raise ValueError("Interim checkpoint does not belong to this wider dataset")
        source_status = "interim"
        report_digest = None
    else:
        report_path = run_directory / "fit/report.json"
        report = json.loads(report_path.read_text())
        if report["status"] != "complete" or report["dataset_sha256"] != dataset_digest:
            raise ValueError("Completed training report and wider dataset do not match")
        source = Path(report["checkpoint"])
        if not frozen.exists():
            shutil.copy2(source, frozen)
        if sha256(source) != sha256(frozen):
            raise ValueError("Completed checkpoint changed since this audit was initialized")
        with np.load(frozen, allow_pickle=False) as saved:
            if str(saved["schema"].item()) != "daily25-jax-weighted-fit-v1":
                raise ValueError("Unexpected wider-fit checkpoint schema")
            if str(saved["dataset_sha256"].item()) != dataset_digest:
                raise ValueError("Checkpoint and dataset hashes do not match")
            raw = np.asarray(saved["raw"], dtype=np.float32)
            model_label = str(saved["model_label"].item())
        epoch = report["completed_epochs"]
        source_status = "complete"
        report_digest = sha256(report_path)
    with np.load(dataset, allow_pickle=False) as data:
        dates = data["dates"].astype(str)
        supports = np.asarray(data["supports"], dtype=np.float64)
        np.testing.assert_allclose(data["model_times_y"], np.arange(1, 26) / DAYS_PER_YEAR, rtol=1e-7)
        if not np.isclose(float(data["backbone_annual_vol"]), 0.05):
            raise ValueError("Paper audit requires the independent 5% Black background")
    if raw.shape != (len(dates), 25, supports.shape[1] - 2) or not np.all(np.isfinite(raw)):
        raise ValueError("Invalid wider fitted grids")
    if dates[-1] != "2025-08-29" or len(dates) != 891:
        raise ValueError("Expected the paper's historical dates and reference date")
    source = dict(model_label=model_label, fit_status=source_status, fit_epoch=epoch,
                  initial_date=str(dates[-1]), dataset=str(dataset.resolve()), dataset_sha256=dataset_digest,
                  checkpoint=str(frozen.resolve()), checkpoint_sha256=sha256(frozen),
                  training_report_sha256=report_digest)
    return raw, source


def conditional_audit(population: entropy.FrontSpotOcePopulation, *, output: Path,
                      objective: str, epochs: int, device: str, monitor: AuditMonitor) -> dict:
    """Use the paper's original optimizers, with complete-pair two-fold separation."""
    if objective not in {"entropy", "squared"}:
        raise ValueError(f"Unknown objective: {objective}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    paths = len(population.features)
    if paths < 4 or paths % 4:
        raise ValueError("Two folds must contain complete antithetic pairs")
    split = paths // 2
    scores = np.empty(paths, dtype=np.float64)
    histories, heldout = {}, {}
    config = (entropy.FrontSpotOceTrainConfig(maximum_epochs=epochs)
              if objective == "entropy" else squared.QuadraticSecondMomentTrainConfig(maximum_epochs=epochs))
    quadratic_population = squared.QuadraticPopulation(
        features=population.features, returns=population.returns, label="Gromm-v1-4,3",
        source=dict(initial_date=population.initial_date), static_active_mask=population.active_mask,
    )
    for fold, start, stop, other_start, other_stop in (("A", 0, split, split, paths), ("B", split, paths, 0, split)):
        initial_state = None
        for affine in (True, False):
            label = f"Gromm-v1-43-{objective}-{'affine' if affine else 'deep'}-{fold}"
            monitor.stage(label)
            checkpoint = output / f"{fold}-{'affine' if affine else 'deep'}.pt"
            if objective == "entropy":
                if checkpoint.exists():
                    policy, saved = entropy._load_policy_checkpoint(checkpoint, device=device, expected_train_config=config)
                    history = saved["history"]
                else:
                    policy, history = entropy._fit_policy(
                        population.features[start:stop], population.returns[start:stop], population.active_mask,
                        label=label, train_config=config, output_path=checkpoint, device=device,
                        affine_only=affine, initial_state_dict=initial_state,
                    )
            else:
                policy, history = squared._fit_policy(
                    quadratic_population, start, stop, label=label, checkpoint_path=checkpoint,
                    train_config=config, device=device, affine_only=affine, initial_state_dict=initial_state,
                )
            histories[label] = len(history)
            initial_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        monitor.stage(f"{objective}-evaluate-{fold}-on-opposite-fold")
        if objective == "entropy":
            summary, _, fold_scores = entropy._evaluate_policy(
                policy, population.features[other_start:other_stop], population.returns[other_start:other_stop],
                population.active_mask, batch_size=config.batch_size, epsilon=config.epsilon,
            )
        else:
            summary, fold_scores = squared._evaluate_policy(
                policy, quadratic_population, other_start, other_stop, batch_size=config.batch_size,
                lambda_second_moment=config.lambda_second_moment, label=f"evaluate-{fold}",
            )
        scores[other_start:other_stop] = fold_scores
        heldout[fold] = summary["paired_conditional_drift"]
    combined, pairs = paired_path_average_statistics(scores)
    np.save(output / "out-of-fold-pair-scores.npy", pairs, allow_pickle=False)
    report = dict(objective=objective, train_config=asdict(config), training_epochs=histories,
                  heldout=heldout, combined_out_of_fold=combined,
                  normalization="Training fold only; complete pairs and whole paths stay together",
                  policy="Uncapped affine plus two-layer width-32 SiLU; affine warm start; original paper optimizer")
    entropy._atomic_json(report, report_path)
    return report


def load_rolling_calibration(directory: Path, raw: np.ndarray, source: dict) -> tuple[
    InitialModelState, JointInnovationFit, dict
]:
    """Validate a prior rolling Q calibration and retain its exact saved arrays.

    Reconstructing the innovations is a consistency check, not a new model fit.
    Population moments cannot supply conditional features, so no old simulated
    observations or policies are imported into the full audit.
    """
    identity = json.loads((directory / "identity.json").read_text())
    summary = json.loads((directory / "calibration.json").read_text())
    expected = dict(schema="gromm-wide-rolling252-unconditional-v1", window_date_count=252,
                    horizon_days=20, factor_rank=8, dlv_substeps=8)
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Incompatible rolling calibration geometry")
    for key in ("model_label", "dataset_sha256", "checkpoint_sha256", "training_report_sha256"):
        if identity["source"].get(key) != source.get(key):
            raise ValueError(f"Rolling calibration source mismatch: {key}")
    # The wrapper is extended here; the numerical model and policy code must be
    # identical to those used for the saved rolling calibration.
    for path in sorted((Path(__file__).parent / "front_spot").glob("*.py")):
        if identity["code_sha256"].get(str(path.resolve())) != sha256(path):
            raise ValueError(f"Rolling calibration numerical code changed: {path.name}")
    dataset = Path(source["dataset"])
    with np.load(dataset, allow_pickle=False) as saved:
        dates = tuple(saved["dates"].astype(str))
    reference = identity["reference_date"]
    if reference != source["initial_date"] or summary["reference_date"] != reference:
        raise ValueError("Rolling calibration reference date mismatch")
    window = rolling_history_indices(dates, reference_date=date.fromisoformat(reference), date_count=252)
    window_dates = [dates[index] for index in window]
    if (identity["window_dates"] != window_dates
            or identity["window_start"] != window_dates[0]
            or identity["window_end"] != window_dates[-1]
            or summary["window_start"] != window_dates[0]
            or summary["window_end"] != window_dates[-1]):
        raise ValueError("Rolling calibration must use exactly 252 strictly past dates")
    history = align_fixed_contract_history(
        dataset_path=dataset, raw_grids=raw,
        training_start=date.fromisoformat(window_dates[0]), training_end=date.fromisoformat(window_dates[-1]),
    )
    if np.any(history.origin_indices < window[0]) or np.any(history.endpoint_indices > window[-1]):
        raise ValueError("Rolling calibration pairs escaped their training window")
    innovations = fit_joint_innovations(history, factor_rank=8, smoothing_sigma=(1.0, 2.0))
    initial = load_initial_model_state(dataset_path=dataset, raw_grids=raw, date_index=dates.index(reference))
    with np.load(directory / "diffusion.npz", allow_pickle=False) as saved:
        for name, value in (("annualized_loadings", innovations.annualized_loadings),
                            ("eigenvalues", innovations.daily_eigenvalues), ("support", initial.support),
                            ("dlvsq", initial.dlvsq), ("density_tower", initial.density_tower)):
            np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Saved rolling calibration changed: {name}")
        np.testing.assert_array_equal(history.origin_indices, saved["origin_indices"])
        np.testing.assert_array_equal(history.endpoint_indices, saved["endpoint_indices"])
        initial = replace(initial, support=saved["support"].copy(), dlvsq=saved["dlvsq"].copy(),
                          density_tower=saved["density_tower"].copy())
        innovations = replace(innovations, annualized_loadings=saved["annualized_loadings"].copy(),
                              daily_eigenvalues=saved["eigenvalues"].copy())
    return initial, innovations, summary


def run(arguments: argparse.Namespace) -> dict:
    """Re-estimate diffusion, generate Q paths, and run all three paper diagnostics."""
    from quantdata import env

    if not all(os.environ.get(key) for key in ("QUANTDATA_ROOT", "PYTHONPATH")):
        raise RuntimeError("Detached project environment is missing")
    if not arguments.run_directory.resolve().is_relative_to(Path(os.environ["QUANTDATA_ROOT"]).resolve()):
        raise ValueError("Training run must be inside the configured external data root")
    emit(event="environment_validated", locations=str(env.locations), jax_backend=jax.default_backend(),
         interpreter=os.sys.executable, working_directory=os.getcwd())
    if arguments.paths < 4 or arguments.paths % 4:
        raise ValueError("Path count must be divisible by four")
    if arguments.chunk_size < 4 or arguments.chunk_size % 4:
        raise ValueError("Chunk size must preserve both complete-pair folds")
    if arguments.generation_batch_size < 2 or arguments.generation_batch_size % 2:
        raise ValueError("Generation batch size must preserve complete pairs")
    if arguments.device == "cpu" and jax.default_backend() != "cpu":
        raise RuntimeError("CPU audits must set JAX_PLATFORMS=cpu")
    if any(check["health"] != "OK" for check in resource_health()):
        raise RuntimeError("Resource checks must all be OK before launching an audit")
    torch.set_num_threads(arguments.torch_threads)
    output = arguments.output.resolve()
    if (output.is_relative_to(Path(__file__).resolve().parents[3])
            or not output.is_relative_to(Path(os.environ["QUANTDATA_ROOT"]).resolve())):
        raise ValueError("Audit outputs must be in the external project data root, outside Git")
    output.mkdir(parents=True, exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "audit.lock", timeout=0):
            raw, source = load_source(arguments.run_directory, output, arguments.snapshot)
            source_files = sorted((Path(__file__).parent / "front_spot").glob("*.py")) + [Path(__file__)]
            identity = dict(source=source, paths=arguments.paths, factor_rank=8, dlv_substeps=8,
                            seed=arguments.seed, horizon_days=20, instrument_count=45,
                            chunk_size=arguments.chunk_size, generation_batch_size=arguments.generation_batch_size,
                            policy_epochs=arguments.policy_epochs, device=arguments.device,
                            code_sha256={str(path.resolve()): sha256(path) for path in source_files})
            if arguments.rolling_calibration is not None:
                directory = arguments.rolling_calibration.resolve()
                identity["rolling_calibration"] = dict(
                    directory=str(directory),
                    sha256={name: sha256(directory / name)
                            for name in ("identity.json", "calibration.json", "diffusion.npz")},
                    population_reused=False, policies_reused=False,
                )
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError("Audit source/configuration/code changed; use a new output directory")
            entropy._atomic_json(identity, identity_path)
            if arguments.rolling_calibration is None:
                monitor.stage("calibrate-wide-historical-diffusion")
                calibration = calibrate_front_spot_model(
                    dataset_path=Path(source["dataset"]), raw_grids=raw, factor_rank=8,
                )
                initial = load_initial_model_state(dataset_path=Path(source["dataset"]), raw_grids=raw)
                innovations = calibration.innovations
                origins, endpoints = calibration.history.origin_indices, calibration.history.endpoint_indices
                calibration_summary = dict(
                    **_calibration_summary(calibration), estimation_end="2024-12-31", smoothing_sigma=[1.0, 2.0],
                    first_origin=calibration.history.dates[origins[0]],
                    last_endpoint=calibration.history.dates[endpoints[-1]],
                )
            else:
                monitor.stage("validate-frozen-rolling252-calibration")
                initial, innovations, calibration_summary = load_rolling_calibration(
                    arguments.rolling_calibration, raw, source,
                )
                with np.load(arguments.rolling_calibration / "diffusion.npz", allow_pickle=False) as saved:
                    origins, endpoints = saved["origin_indices"], saved["endpoint_indices"]
            tower = initial.density_tower
            stability = dict(minimum_probability=float(tower.min()),
                             maximum_mass_error=float(np.max(np.abs(tower.sum(axis=-1) - 1))),
                             maximum_forward_error=float(np.max(np.abs(tower @ initial.support - 1))))
            if not np.all(np.isfinite(tower)) or stability["minimum_probability"] < -1e-13 or max(
                    stability["maximum_mass_error"], stability["maximum_forward_error"]) > 1e-10:
                raise ValueError("Initial wider state fails probability mass/forward validation")
            calibration_summary = dict(calibration_summary, initial_stability=stability)
            entropy._atomic_json(calibration_summary, output / "calibration.json")
            np.savez(output / "diffusion.npz", annualized_loadings=innovations.annualized_loadings,
                     eigenvalues=innovations.daily_eigenvalues,
                     origin_indices=origins, endpoint_indices=endpoints,
                     support=initial.support, dlvsq=initial.dlvsq, density_tower=tower)
            chunks = output / "population-chunks"
            chunks.mkdir(exist_ok=True)
            features, returns = [], []
            generation_seconds = 0.0
            for start in range(0, arguments.paths, arguments.chunk_size):
                count = min(arguments.chunk_size, arguments.paths - start)
                monitor.stage("generate-Q-paths", start, arguments.paths)
                chunk = chunks / f"{start:06d}.npz"
                if not chunk.exists():
                    population = entropy.generate_front_spot_oce_population(
                        None, innovations=innovations, initial_state=initial, path_count=count,
                        batch_size=arguments.generation_batch_size, dlv_substeps=8,
                        seed=arguments.seed + start, measure="Q",
                    )
                    temporary = chunk.with_suffix(".tmp.npz")
                    np.savez(temporary, features=population.features, returns=population.returns,
                             active_mask=population.active_mask, strikes=population.strikes,
                             instrument_labels=np.asarray(population.instrument_labels),
                             generation_seconds=population.generation_seconds)
                    temporary.replace(chunk)
                with np.load(chunk, allow_pickle=False) as data:
                    if data["features"].shape != (count, 20, 58) or data["returns"].shape != (count, 20, 45):
                        raise ValueError("Population chunk geometry mismatch")
                    features.append(data["features"])
                    returns.append(data["returns"])
                    active_mask, strikes = data["active_mask"], data["strikes"]
                    labels = tuple(str(label) for label in data["instrument_labels"])
                    generation_seconds += float(data["generation_seconds"])
            population = entropy.FrontSpotOcePopulation(
                features=np.concatenate(features), returns=np.concatenate(returns), active_mask=active_mask,
                strikes=strikes, instrument_labels=labels, instrument_universe="restricted",
                initial_date=initial.date, generation_seconds=generation_seconds,
            )
            del features, returns
            if not np.all(np.isfinite(population.features)) or not np.all(np.isfinite(population.returns)):
                raise FloatingPointError("Non-finite simulated population")
            monitor.stage("unconditional-audit", arguments.paths, arguments.paths)
            unconditional = entropy.unconditional_spot_scaled_martingale_audit(
                population.returns, population.active_mask, population.instrument_labels,
            )
            entropy._atomic_json(unconditional, output / "unconditional.json")
            conditional = {objective: conditional_audit(
                population, output=output / objective, objective=objective, epochs=arguments.policy_epochs,
                device=arguments.device, monitor=monitor,
            ) for objective in ("squared", "entropy")}
            result = dict(schema="gromm-wide-restricted-q-audit-v1", status="complete", **identity,
                          calibration=calibration_summary, unconditional=unconditional, conditional=conditional,
                          generation_seconds=generation_seconds, elapsed_seconds=time.monotonic() - monitor.started,
                          interpretation="Restricted-universe statistical diagnostics, not a proof over all strategies or tail strikes")
            entropy._atomic_json(result, output / "q-audit.json")
            monitor.stage("complete", arguments.paths, arguments.paths)
            emit(event="complete", output=str(output / "q-audit.json"))
            return result
    finally:
        monitor.done.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="Explicit interim snapshot for a separately labelled smoke audit")
    parser.add_argument("--rolling-calibration", type=Path,
                        help="Reuse the exact validated state/diffusion from a completed rolling252 audit")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--policy-epochs", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
