"""Fit a CPU empirical risk-neutral density to a fixed return-path matrix."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from quantdata.dysanos.flex.empirical_risk_neutral import (
    EmpiricalRiskNeutralFit,
    ReturnMatrix,
    fit_empirical_risk_neutral_density,
    fit_empirical_risk_neutral_density_torch,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _book,
    _nms_book,
)

_INSTRUMENT_LABELS = _book((5, 10)).labels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--log-weights", type=Path)
    parser.add_argument("--array-key", default="returns")
    parser.add_argument(
        "--book-expiry-days",
        type=int,
        nargs="+",
        default=[5, 10],
    )
    strike_group = parser.add_mutually_exclusive_group()
    strike_group.add_argument(
        "--relative-strikes",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional increasing relative-strike book. The ITM call is used at "
            "strikes no greater than one and the ITM put above one."
        ),
    )
    strike_group.add_argument(
        "--normalized-moneyness",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional increasing NMS book used to label and cost the return matrix."
        ),
    )
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=32_768)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Use CPU streaming when 'cpu'; otherwise upload the fixed matrix to Torch.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--upload-chunk-size", type=int, default=32_768)
    parser.add_argument("--maximum-iterations", type=int, default=2_000)
    parser.add_argument("--maximum-active-set-iterations", type=int, default=50)
    parser.add_argument("--moment-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--optimizer-gradient-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--optimizer-function-tolerance", type=float, default=1.0e-15)
    return parser


def _mapping_array(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"input mapping has no array key {key!r}")
    return value[key]


def load_return_matrix(path: Path, *, key: str) -> ReturnMatrix:
    """Load supported return artifacts, memory-mapping an ``.npy`` input."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        value: Any = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key in archive.files:
                value = archive[key]
            elif len(archive.files) == 1:
                value = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"npz input has keys {archive.files}, but no requested key {key!r}"
                )
    elif suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        value = _mapping_array(payload, key) if isinstance(payload, Mapping) else payload
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    else:
        raise ValueError("returns input must have suffix .npy, .npz, .pt, or .pth")
    if not isinstance(value, np.ndarray):
        raise ValueError("returns artifact does not contain a NumPy array or Torch tensor")
    if value.ndim not in {2, 3}:
        raise ValueError("returns array must be two- or three-dimensional")
    return value


def _atomic_save_npy(values: np.ndarray, path: Path) -> None:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"density output must use the .npy suffix, found {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _cost_band(
    coordinates: int,
    *,
    instrument_labels: tuple[str, ...],
    spot_bps: float,
    option_bps: float,
) -> np.ndarray:
    if coordinates % len(instrument_labels):
        raise ValueError(
            f"coordinate count {coordinates} is not divisible by {len(instrument_labels)}"
        )
    if not np.isfinite(spot_bps) or spot_bps < 0.0:
        raise ValueError("spot cost must be finite and nonnegative")
    if not np.isfinite(option_bps) or option_bps < 0.0:
        raise ValueError("option cost must be finite and nonnegative")
    one_step = np.asarray(
        (spot_bps, *(option_bps for _ in range(len(instrument_labels) - 1))),
        dtype=np.float64,
    ) / 1.0e4
    return np.tile(one_step, coordinates // len(one_step))


def _report(
    fit: EmpiricalRiskNeutralFit,
    *,
    input_path: Path,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype[Any],
    instrument_labels: tuple[str, ...],
    solver_description: str,
) -> dict[str, object]:
    paths = input_shape[0]
    coordinates = int(np.prod(input_shape[1:]))
    steps = coordinates // len(instrument_labels)
    has_cost_band = bool(np.any(fit.cost_band))
    primal_definition = (
        "min mean(w*log(w)); mean(w)=1; w>=0; "
        "abs(mean(w[:,None]*returns,axis=0))<=cost_band"
        if has_cost_band
        else "min mean(w*log(w)); mean(w)=1; w>0; mean(w[:,None]*returns,axis=0)=0"
    )
    dual_definition = (
        "log(mean(exp(-returns@action))) + cost_band@abs(action)"
        if has_cost_band
        else "log(mean(exp(-returns@action)))"
    )
    return {
        "schema": "dysanos-fixed-empirical-risk-neutral-v1",
        "status": "complete" if fit.optimizer_success else "optimizer_not_converged",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            "path": str(input_path),
            "shape": list(input_shape),
            "dtype": str(input_dtype),
            "num_paths": paths,
            "num_steps": steps,
            "num_instruments": len(instrument_labels),
            "num_coordinates": coordinates,
            "flattening": "C order: step-major, then instrument",
            "instrument_labels": list(instrument_labels),
        },
        "definition": {
            "reference_measure": "uniform empirical measure on the supplied training paths",
            "density_normalization": "weights are dQ/dP with sample mean one",
            "primal": primal_definition,
            "signed_dual": dual_definition,
            "transaction_cost_band": "nonzero" if has_cost_band else "none",
            "solver": solver_description,
            "scope": (
                f"{coordinates} unconditional training-population return moments; "
                "this is not a conditional martingale or out-of-sample validation claim"
            ),
        },
        "fit": {
            "optimizer_success": fit.optimizer_success,
            "optimizer_status": fit.optimizer_status,
            "optimizer_message": fit.optimizer_message,
            "optimizer_iterations": fit.optimizer_iterations,
            "objective_evaluations": fit.objective_evaluations,
            "active_coordinates": fit.active_coordinates,
            "active_set_iterations": fit.active_set_iterations,
            "primal_relative_entropy": fit.primal_relative_entropy,
            "dual_objective": fit.dual_objective,
            "primal_dual_gap": fit.primal_dual_gap,
            "effective_sample_size": fit.effective_sample_size,
            "effective_sample_fraction": fit.effective_sample_fraction,
            "maximum_constraint_violation": fit.maximum_constraint_violation,
            "maximum_kkt_residual": fit.maximum_kkt_residual,
            "minimum_weight": float(np.min(fit.weights)),
            "maximum_weight": float(np.max(fit.weights)),
            "zero_weight_count": int(np.sum(fit.weights == 0.0)),
            "density_mass": float(np.mean(fit.weights)),
            "action": fit.action.tolist(),
            "weighted_mean": fit.weighted_mean.tolist(),
            "cost_band": fit.cost_band.tolist(),
        },
    }


def run(arguments: argparse.Namespace) -> EmpiricalRiskNeutralFit:
    """Fit and serialize a fixed-training-population empirical measure."""

    returns = load_return_matrix(arguments.returns, key=arguments.array_key)
    argument_nms = getattr(arguments, "normalized_moneyness", None)
    if argument_nms is not None:
        instrument_labels = _nms_book(
            tuple(arguments.book_expiry_days),
            normalized_moneyness=tuple(argument_nms),
        ).labels
    else:
        argument_strikes = getattr(arguments, "relative_strikes", None)
        relative_strikes = (
            tuple(argument_strikes)
            if argument_strikes is not None
            else (0.99, 1.0, 1.01)
        )
        instrument_labels = _book(
            tuple(arguments.book_expiry_days), relative_strikes=relative_strikes
        ).labels
    coordinates = int(np.prod(returns.shape[1:]))
    cost_band = _cost_band(
        coordinates,
        instrument_labels=instrument_labels,
        spot_bps=arguments.spot_cost_bps,
        option_bps=arguments.option_cost_bps,
    )
    device = getattr(arguments, "device", "cpu")
    if device == "cpu":
        fit = fit_empirical_risk_neutral_density(
            returns,
            cost_band,
            chunk_size=arguments.chunk_size,
            maximum_iterations=arguments.maximum_iterations,
            moment_tolerance=arguments.moment_tolerance,
            optimizer_gradient_tolerance=arguments.optimizer_gradient_tolerance,
            optimizer_function_tolerance=arguments.optimizer_function_tolerance,
            maximum_active_set_iterations=arguments.maximum_active_set_iterations,
        )
        if np.any(cost_band):
            solver_description = (
                "exact signed active-set dual; each smooth nonnegative orthant is "
                "solved by CPU L-BFGS-B using streamed stable log-sum-exp and "
                "analytic gradients"
            )
        else:
            solver_description = (
                "smooth unconstrained zero-cost entropy dual solved by streamed "
                "CPU trust-region Newton-CG iterations using analytic gradients and "
                "exact covariance Hessian-vector products"
            )
    else:
        torch_dtype = (
            torch.float32
            if getattr(arguments, "torch_dtype", "float64") == "float32"
            else torch.float64
        )
        fit = fit_empirical_risk_neutral_density_torch(
            returns,
            cost_band,
            device=device,
            dtype=torch_dtype,
            upload_chunk_size=getattr(arguments, "upload_chunk_size", 32_768),
            final_chunk_size=arguments.chunk_size,
            maximum_iterations=arguments.maximum_iterations,
            moment_tolerance=arguments.moment_tolerance,
            optimizer_gradient_tolerance=arguments.optimizer_gradient_tolerance,
            optimizer_function_tolerance=arguments.optimizer_function_tolerance,
            maximum_active_set_iterations=arguments.maximum_active_set_iterations,
        )
        if np.any(cost_band):
            solver_description = (
                "exact signed active-set dual; SciPy L-BFGS-B controls each orthant, "
                f"while full-population objective and gradient evaluations use {device} "
                f"in {str(torch_dtype).removeprefix('torch.')} arithmetic; final weights "
                "and KKT diagnostics are recomputed by streamed CPU float64 arithmetic"
            )
        else:
            solver_description = (
                "smooth unconstrained zero-cost entropy dual solved by SciPy "
                f"trust-region Newton-CG iterations, with {device} "
                f"{str(torch_dtype).removeprefix('torch.')} gradients and exact "
                "covariance Hessian-vector products; final weights and KKT "
                "diagnostics are recomputed by streamed CPU float64 arithmetic"
            )
    if arguments.weights is not None:
        _atomic_save_npy(fit.weights, arguments.weights)
    if arguments.log_weights is not None:
        _atomic_save_npy(fit.log_weights, arguments.log_weights)
    write_json_atomic(
        _report(
            fit,
            input_path=arguments.returns,
            input_shape=tuple(returns.shape),
            input_dtype=returns.dtype,
            instrument_labels=instrument_labels,
            solver_description=solver_description,
        ),
        arguments.report,
    )
    return fit


def main() -> None:
    arguments = _parser().parse_args()
    fit = run(arguments)
    print(
        f"success={fit.optimizer_success} KL={fit.primal_relative_entropy:.9g} "
        f"ESS={fit.effective_sample_size:.1f} "
        f"max_violation={fit.maximum_constraint_violation:.3g} "
        f"max_kkt={fit.maximum_kkt_residual:.3g}",
        flush=True,
    )
    if not fit.optimizer_success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
