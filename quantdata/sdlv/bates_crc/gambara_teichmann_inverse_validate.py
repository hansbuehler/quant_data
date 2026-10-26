"""Exact held-out repricing gate for the learned generalized-Bates inverse."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from quantdata.sdlv.bates_crc.gambara_teichmann_crc import (
    FiveBucketJumps,
    GeneralizedBatesState,
    paired_maturity_boundaries,
)
from quantdata.sdlv.bates_crc.gambara_teichmann_inverse_data import (
    exact_reset_price_gate,
)
from quantdata.sdlv.bates_crc.gambara_teichmann_inverse_train import (
    CrcInverseNetworkConfig,
    CrcInverseNetworks,
    _load_grid,
    _load_split,
    apply_reset_innovations,
)
from quantdata.sdlv.bates_crc.heston_riccati import HestonParams
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _network_config(payload: dict[str, object]) -> CrcInverseNetworkConfig:
    values = dict(payload)
    values["forward_network"] = FeedForwardConfig(**values["forward_network"])
    values["inverse_network"] = FeedForwardConfig(**values["inverse_network"])
    known = {field.name for field in fields(CrcInverseNetworkConfig)}
    if set(values) != known:
        raise RuntimeError("inverse checkpoint model configuration has unexpected fields")
    return CrcInverseNetworkConfig(**values)


def load_inverse_network(checkpoint: Path, *, device: str) -> CrcInverseNetworks:
    """Load a version-checked inverse network for valuation only."""

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "gambara-teichmann-crc-inverse-networks-v1":
        raise RuntimeError(f"invalid inverse checkpoint schema in {checkpoint}")
    config = _network_config(payload["model_config"])
    if payload.get("user_version") != config.user_version:
        raise RuntimeError("inverse checkpoint user version does not match its configuration")
    if payload.get("model_version") != config.version.unique_id48:
        raise RuntimeError("inverse checkpoint code version is stale")
    state = payload["model_state_dict"]
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = CrcInverseNetworks(
        env,
        config=config,
        parameter_mean=state["parameter_mean"],
        parameter_scale=state["parameter_scale"],
        surface_mean=state["surface_mean"],
        surface_scale=state["surface_scale"],
        jump_lower=state["jump_lower"],
        jump_upper=state["jump_upper"],
    )
    model.load_state_dict(state)
    model.eval()
    return model


def verify_training_provenance(
    checkpoint: Path,
    *,
    dataset_directory: Path,
    target_fit_report: Path,
) -> dict[str, object]:
    """Require the completed training report matching a valuation checkpoint."""

    report_path = checkpoint.parent / "training-report.json"
    if not report_path.exists():
        raise RuntimeError(f"missing completed inverse-training report {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "gambara-teichmann-crc-inverse-training-report-v1"
        or report.get("status") != "complete"
    ):
        raise RuntimeError(f"inverse-training report is incomplete or incompatible: {report_path}")
    expected = {
        "dataset_directory": dataset_directory.resolve(),
        "target_fit_report": target_fit_report.resolve(),
    }
    for key, path in expected.items():
        recorded = Path(str(report.get(key, ""))).resolve()
        if recorded != path:
            raise RuntimeError(
                f"inverse checkpoint provenance mismatch for {key}: {recorded} != {path}"
            )
    return report


def validate_exact_resets(
    *,
    checkpoint: Path,
    dataset_directory: Path,
    target_fit_report: Path,
    output_directory: Path,
    sample_count: int,
    tolerance: float,
    device: str,
    seed: int = 19_871,
) -> dict[str, object]:
    """Apply exact Fourier gates to independently held-out stochastic resets."""

    if sample_count <= 0 or tolerance <= 0.0:
        raise ValueError("sample_count and tolerance must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = json.loads(target_fit_report.read_text(encoding="utf-8"))
    verify_training_provenance(
        checkpoint,
        dataset_directory=dataset_directory,
        target_fit_report=target_fit_report,
    )
    reset_law = report["reset_innovation_law"]
    target_fit = report["target_fit"]
    kappa = float(target_fit["kappa_fixed_from_history"])
    intensity = float(target_fit["intensity_fixed_from_history"])
    parameters, surfaces = _load_split(dataset_directory, "heldout")
    if sample_count > len(parameters):
        raise ValueError(f"requested {sample_count} gates from only {len(parameters)} held-out rows")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(parameters), size=sample_count, replace=False)
    parameter_rows = parameters[selected]
    surface_rows = surfaces[selected]
    innovations = rng.multivariate_normal(
        np.asarray(reset_law["mean"], dtype=np.float64),
        np.asarray(reset_law["covariance"], dtype=np.float64),
        size=sample_count,
    ).astype(np.float32)
    old_heston = torch.from_numpy(parameter_rows[:, :4])
    new_heston = apply_reset_innovations(
        old_heston,
        torch.from_numpy(innovations),
        kappa=kappa,
        relative_change_cap=float(reset_law["relative_change_cap"]),
    )
    model = load_inverse_network(checkpoint, device=device)
    with torch.no_grad():
        candidate_jumps = model.jump_parameters(
            new_heston.to(model.device),
            torch.from_numpy(surface_rows).to(model.device),
        ).cpu().numpy()

    (
        maturities,
        moneyness,
        dataset_kappa,
        dataset_intensity,
        dataset_quadrature_nodes,
        dataset_quadrature_u_max,
    ) = _load_grid(dataset_directory)
    if dataset_kappa != kappa or dataset_intensity != intensity:
        raise RuntimeError(
            "inverse-map fixed kappa/intensity do not match the target calibration"
        )
    if dataset_quadrature_nodes < 240 or dataset_quadrature_u_max < 600.0:
        raise RuntimeError("inverse-map validation requires the accurate Fourier dataset")
    boundaries = paired_maturity_boundaries(maturities)
    calibration_expiries = np.repeat(maturities, len(moneyness))
    calibration_strikes = np.tile(moneyness, len(maturities))
    midpoint_strikes = 0.5 * (moneyness[:-1] + moneyness[1:])
    midpoint_expiries = 0.5 * (maturities[:-1] + maturities[1:])
    heldout_strikes = np.concatenate(
        (
            np.tile(midpoint_strikes, len(maturities)),
            np.tile(moneyness, len(midpoint_expiries)),
        )
    )
    heldout_expiries = np.concatenate(
        (
            np.repeat(maturities, len(midpoint_strikes)),
            np.repeat(midpoint_expiries, len(moneyness)),
        )
    )

    calibration_errors = np.empty(sample_count, dtype=np.float64)
    heldout_errors = np.empty(sample_count, dtype=np.float64)
    accepted = np.empty(sample_count, dtype=np.bool_)
    progress_path = output_directory / "exact-gate-progress.jsonl"
    for sample in range(sample_count):
        old = parameter_rows[sample]
        new = new_heston[sample].numpy()
        before = GeneralizedBatesState(
            log_spot=0.0,
            variance=float(old[0]),
            heston=HestonParams(kappa=kappa, theta=float(old[1]), xi=float(old[2]), rho=float(old[3])),
            jumps=FiveBucketJumps(
                intensity=intensity,
                boundaries=boundaries,
                means=tuple(float(value) for value in old[4:9]),
                stds=tuple(float(value) for value in old[9:14]),
            ),
        )
        after = GeneralizedBatesState(
            log_spot=0.0,
            variance=float(new[0]),
            heston=HestonParams(kappa=kappa, theta=float(new[1]), xi=float(new[2]), rho=float(new[3])),
            jumps=FiveBucketJumps(
                intensity=intensity,
                boundaries=boundaries,
                means=tuple(float(value) for value in candidate_jumps[sample, :5]),
                stds=tuple(float(value) for value in candidate_jumps[sample, 5:]),
            ),
        )
        gate = exact_reset_price_gate(
            before,
            after,
            calibration_strikes=calibration_strikes,
            calibration_expiries=calibration_expiries,
            heldout_strikes=heldout_strikes,
            heldout_expiries=heldout_expiries,
            tolerance=tolerance,
            n_quad=240,
            u_max=600.0,
        )
        calibration_errors[sample] = gate["calibration_max_abs_price_error"]
        heldout_errors[sample] = gate["heldout_max_abs_price_error"]
        accepted[sample] = gate["accepted"]
        if sample == 0 or (sample + 1) % 10 == 0 or sample + 1 == sample_count:
            elapsed = time.monotonic() - started
            completed = sample + 1
            record = {
                "stage": "exact_reset_gate",
                "elapsed_seconds": elapsed,
                "completed_work_units": completed,
                "total_work_units": sample_count,
                "estimated_remaining_seconds": elapsed * (sample_count - completed) / completed,
                "accepted_so_far": int(np.count_nonzero(accepted[:completed])),
            }
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

    maximum = np.maximum(calibration_errors, heldout_errors)
    result: dict[str, object] = {
        "schema": "gambara-teichmann-crc-exact-reset-validation-v1",
        "checkpoint": str(checkpoint.resolve()),
        "dataset_directory": str(dataset_directory.resolve()),
        "target_fit_report": str(target_fit_report.resolve()),
        "sample_count": sample_count,
        "tolerance": tolerance,
        "accepted_count": int(np.count_nonzero(accepted)),
        "accepted_fraction": float(np.mean(accepted)),
        "maximum_abs_price_error": float(np.max(maximum)),
        "p99_abs_price_error": float(np.quantile(maximum, 0.99)),
        "median_abs_price_error": float(np.median(maximum)),
        "maximum_calibration_error": float(np.max(calibration_errors)),
        "maximum_heldout_error": float(np.max(heldout_errors)),
        "status": "accepted" if np.all(accepted) else "rejected",
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary = output_directory / ".exact-reset-validation.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_directory / "exact-reset-validation.json")
    np.savez_compressed(
        output_directory / "exact-reset-errors.npz",
        selected_indices=selected,
        calibration_max_abs_price_error=calibration_errors,
        heldout_max_abs_price_error=heldout_errors,
        accepted=accepted,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--target-fit-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1_000)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    result = validate_exact_resets(
        checkpoint=arguments.checkpoint,
        dataset_directory=arguments.dataset_directory,
        target_fit_report=arguments.target_fit_report,
        output_directory=arguments.output_directory,
        sample_count=arguments.sample_count,
        tolerance=arguments.tolerance,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
