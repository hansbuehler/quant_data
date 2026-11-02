"""Audit a saved GROMM2 model at its origin without recalibrating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model
from quantdata.gromm2.rolling_audit import UnconditionalAuditConfig, run_audit
from quantdata.gromm2.rolling_calibration import write_json

AUDIT_NMS = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5)


def initial_ar1_loadings(arrays: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saved predictable first-day AR(1) amplitudes exactly once."""
    parameters = arrays["ar1_parameters"]
    predicted = parameters["variance_intercept"] + parameters["variance_slope"] * parameters["terminal_log_squared"]
    bound = float(parameters["log_scale_bound"])
    scales = parameters["fixed_factor_scales"] * parameters["scale_calibration"] * np.exp(np.clip(0.5 * predicted, -bound, bound))
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("saved first-day AR(1) amplitudes must be finite and positive")
    return np.asarray(arrays["loadings"]) * scales, scales


def prepare_legacy_audit(
    *, history: Path, estimate: Path, factor_law: Path, reference_inputs: Path,
    output_directory: Path, origin: str,
) -> dict[str, object]:
    """Create a provenance-labelled input adapter for the existing origin audit."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("this audit requires explicitly disabled GPU visibility")
    arrays, metadata = load_legacy_ar1_model(history=history, estimate=estimate, factor_law=factor_law, origin=origin, backend="cpu")
    if jax.default_backend() != "cpu" or np.asarray(arrays["loadings"]).shape[-1] != 16:
        raise RuntimeError("the requested audit requires the saved 16-factor model on CPU")
    with np.load(reference_inputs, allow_pickle=False) as saved:
        expiries = np.asarray(saved["model_expiries_y"])
        variances = np.asarray(saved["projected_total_variances"])
    np.testing.assert_array_equal(np.rint(252.0 * expiries), np.rint(252.0 * arrays["expiries_y"]))
    if variances.shape != expiries.shape or np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("the common reference ATM variance curve is invalid")
    loadings, scales = initial_ar1_loadings(arrays)
    own = output_directory / "own_dates" / origin
    rolling = output_directory / "rolling" / origin
    own.mkdir(parents=True, exist_ok=True)
    rolling.mkdir(parents=True, exist_ok=True)
    provenance = {
        **metadata, "scope": "one-day unconditional origin audit only; not a pathwise conditional test",
        "adapter": "unchanged saved model; predicted first-day AR(1) amplitudes multiplied into the loading field",
        "reference_inputs": str(reference_inputs.resolve()),
        "reference_inputs_sha256": hashlib.sha256(reference_inputs.read_bytes()).hexdigest(),
        "initial_factor_scales": scales.tolist(),
        "measure_preconditioning": False, "source_refitted": False,
    }
    report_path = output_directory / "adapter.json"
    if report_path.exists():
        if json.loads(report_path.read_text()) != provenance:
            raise ValueError("existing audit adapter has different saved-model provenance")
        return provenance
    _atomic_npz(own / "inputs.npz", {
        "supports": np.asarray(arrays["supports"]), "transports": np.asarray(arrays["transports"]),
        "model_expiries_y": np.asarray(arrays["expiries_y"]), "projected_total_variances": variances,
    })
    _atomic_npz(own / "fit.npz", {"dlvsq": np.asarray(arrays["dlvsq"]), "densities": np.asarray(arrays["densities"])})
    _atomic_npz(rolling / "ar1.npz", {"q_log_dlvsq_loadings": loadings})
    write_json(own / "report.json", {
        "model_label": metadata["model_label"], "adapter_only": True,
        "config": {"geometry": {"backbone_annual_volatility": metadata["background_volatility"]}},
    })
    write_json(rolling / "report.json", {"model_label": metadata["model_label"], "adapter_only": True, "provenance": provenance})
    write_json(report_path, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--factor-law", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--origin", default="2025-08-29")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--substeps", type=int, choices=(1, 8), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    provenance = prepare_legacy_audit(
        history=arguments.history, estimate=arguments.estimate, factor_law=arguments.factor_law,
        reference_inputs=arguments.reference_inputs, output_directory=arguments.output_directory,
        origin=arguments.origin,
    )
    config = UnconditionalAuditConfig(path_count=arguments.paths, batch_size=arguments.batch_size, substeps=arguments.substeps, nms=AUDIT_NMS)
    print(json.dumps({"event": "saved_model_audit_preflight", "model": provenance["model_label"], "backend": jax.default_backend(), "config": asdict(config)}), flush=True)
    if not arguments.prepare_only:
        report = run_audit(calibration_root=arguments.output_directory, origin=arguments.origin, config=config)
        write_json(arguments.output_directory / "result.json", {"provenance": provenance, "audit": report})


if __name__ == "__main__":
    main()
