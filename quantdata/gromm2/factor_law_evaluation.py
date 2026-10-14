"""Rolling-origin evaluation for fitted GROMM2 factor-law architectures."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.gromm2.factor_law import (
    FactorLawConfig,
    FactorLawDataset,
    FactorLawFit,
    _bounded_spot_correlations,
    _lstm_forward,
    _mamba_forward,
    _q_point_spot_correlated_negative_log_likelihood,
    build_factor_law_dataset,
    fit_ar1_factor_law,
    fit_fixed_factor_law,
    fit_recurrent_factor_law,
)

FloatArray = NDArray[np.float64]
MODEL_NAMES = (
    "fixed",
    "ar1",
    "lstm",
    "mamba",
    "lstm_corr",
    "mamba_corr",
)


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class FactorLawEvaluationConfig:
    """Controls for rolling-origin fitting and paired block inference."""

    fold_count: int = 4
    fold_size: int = 63
    selection_fraction: float = 0.20
    block_length: int = 10
    bootstrap_samples: int = 20_000
    seed: int = 20261013

    def __post_init__(self) -> None:
        verify_inp(self.fold_count > 0, lambda: "fold_count must be positive")
        verify_inp(self.fold_size >= 20, lambda: "fold_size must be at least 20")
        verify_inp(
            0.10 <= self.selection_fraction <= 0.40,
            lambda: "selection_fraction must lie in [0.10, 0.40]",
        )
        verify_inp(
            2 <= self.block_length <= self.fold_size,
            lambda: "block_length must lie between 2 and fold_size",
        )
        verify_inp(
            self.bootstrap_samples >= 1_000,
            lambda: "bootstrap_samples must be at least 1000",
        )


def _renormalize_for_origin(
    data: FactorLawDataset,
    *,
    fit_stop: int,
    selection_fraction: float,
) -> tuple[FactorLawDataset, FactorLawDataset]:
    raw_inputs = data.inputs * data.input_scale + data.input_mean
    train_count = int(math.floor((1.0 - selection_fraction) * fit_stop))
    if train_count <= max(32, data.factor_count + 2):
        raise ValueError("rolling-origin training prefix is too short")
    input_mean = np.mean(raw_inputs[:train_count], axis=0)
    input_scale = np.maximum(
        np.std(raw_inputs[:train_count], axis=0, ddof=1), 1.0e-6
    )
    inputs = (raw_inputs - input_mean) / input_scale
    fixed_factor_scales = np.maximum(
        np.sqrt(np.mean(data.factor_scores[:train_count] ** 2, axis=0)),
        1.0e-6,
    )
    risk_price_mean = np.mean(data.physical_risk_prices[:train_count], axis=0)
    risk_price_scale = np.maximum(
        np.std(data.physical_risk_prices[:train_count], axis=0, ddof=1),
        1.0e-6,
    )
    normalized = replace(
        data,
        inputs=inputs,
        fixed_factor_scales=fixed_factor_scales,
        risk_price_mean=risk_price_mean,
        risk_price_scale=risk_price_scale,
        input_mean=input_mean,
        input_scale=input_scale,
        train_count=train_count,
    )
    fitting = replace(
        normalized,
        dates=normalized.dates[:fit_stop],
        origin_indices=normalized.origin_indices[:fit_stop],
        inputs=normalized.inputs[:fit_stop],
        reset=normalized.reset[:fit_stop],
        factor_scores=normalized.factor_scores[:fit_stop],
        physical_risk_prices=normalized.physical_risk_prices[:fit_stop],
    )
    return normalized, fitting


def _predict(
    fit: FactorLawFit,
    data: FactorLawDataset,
    *,
    config: FactorLawConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    factor_count = data.factor_count
    zero_correlations = np.zeros(
        (len(data.factor_scores), factor_count - 1),
        dtype=np.float64,
    )
    if fit.architecture == "fixed":
        scales = np.broadcast_to(
            fit.parameters["factor_scale"], data.factor_scores.shape
        )
        drift = np.broadcast_to(
            fit.parameters["risk_price"], data.physical_risk_prices.shape
        )
        return scales, drift, zero_correlations

    if fit.architecture == "ar1":
        log_squared = np.log(
            (data.factor_scores / data.fixed_factor_scales) ** 2
            + config.squared_score_floor
        )
        previous_log_squared = np.vstack(
            (np.zeros((1, factor_count)), log_squared[:-1])
        )
        previous_log_squared[data.reset] = 0.0
        predicted_log_squared = (
            fit.parameters["variance_intercept"]
            + fit.parameters["variance_slope"] * previous_log_squared
        )
        scales = data.fixed_factor_scales * np.exp(
            np.clip(
                0.5 * predicted_log_squared,
                -config.log_scale_bound,
                config.log_scale_bound,
            )
        )
        scales *= fit.parameters["scale_calibration"]

        standardized = (
            data.physical_risk_prices - data.risk_price_mean
        ) / data.risk_price_scale
        previous_drift = np.vstack((np.zeros((1, factor_count)), standardized[:-1]))
        previous_drift[data.reset] = 0.0
        predicted_standardized = (
            fit.parameters["drift_intercept"]
            + fit.parameters["drift_slope"] * previous_drift
        )
        drift = (
            data.risk_price_mean
            + data.risk_price_scale * predicted_standardized
        )
        return scales, drift, zero_correlations

    parameters = {name: jnp.asarray(value) for name, value in fit.parameters.items()}
    base_architecture = fit.architecture.removesuffix("_corr")
    forward = _lstm_forward if base_architecture == "lstm" else _mamba_forward
    outputs, _ = forward(
        parameters,
        jnp.asarray(data.inputs),
        jnp.asarray(data.reset),
    )
    outputs_array = np.asarray(outputs)
    drift = (
        data.risk_price_mean
        + data.risk_price_scale * outputs_array[:, :factor_count]
    )
    scales = data.fixed_factor_scales * np.exp(
        np.clip(
            outputs_array[:, factor_count : 2 * factor_count],
            -config.log_scale_bound,
            config.log_scale_bound,
        )
    )
    correlations = (
        _bounded_spot_correlations(
            outputs_array[:, 2 * factor_count :],
            bound=config.spot_correlation_bound,
        )
        if fit.architecture.endswith("_corr")
        else zero_correlations
    )
    return scales, drift, correlations


def _fit_models(
    data: FactorLawDataset,
    *,
    config: FactorLawConfig,
) -> tuple[FactorLawFit, ...]:
    return (
        fit_fixed_factor_law(data),
        fit_ar1_factor_law(data, config=config),
        fit_recurrent_factor_law(data, architecture="lstm", config=config),
        fit_recurrent_factor_law(data, architecture="mamba", config=config),
        fit_recurrent_factor_law(
            data,
            architecture="lstm",
            config=config,
            learn_spot_correlation=True,
        ),
        fit_recurrent_factor_law(
            data,
            architecture="mamba",
            config=config,
            learn_spot_correlation=True,
        ),
    )


def _moving_block_means(
    values: FloatArray,
    *,
    block_length: int,
    samples: int,
    generator: np.random.Generator,
) -> FloatArray:
    observations = len(values)
    block_count = math.ceil(observations / block_length)
    starts = generator.integers(
        0,
        observations - block_length + 1,
        size=(samples, block_count),
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(samples, -1)
    indices = indices[:, :observations]
    return np.mean(values[indices], axis=1)


def _skill_summary(
    baseline: FloatArray,
    candidate: FloatArray,
    *,
    block_length: int,
    samples: int,
    generator: np.random.Generator,
) -> dict[str, float]:
    paired_improvement = baseline - candidate
    bootstrapped = _moving_block_means(
        paired_improvement,
        block_length=block_length,
        samples=samples,
        generator=generator,
    )
    return {
        "mean_improvement": float(np.mean(paired_improvement)),
        "confidence_interval_95_lower": float(np.quantile(bootstrapped, 0.025)),
        "confidence_interval_95_upper": float(np.quantile(bootstrapped, 0.975)),
        "bootstrap_probability_positive": float(np.mean(bootstrapped > 0.0)),
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def evaluate_factor_laws(
    *,
    base_estimate: Path,
    history_path: Path,
    dataset: Path,
    output: Path,
    model_config: FactorLawConfig = FactorLawConfig(),
    evaluation_config: FactorLawEvaluationConfig = FactorLawEvaluationConfig(),
) -> dict[str, Any]:
    """Run expanding-prefix fits and paired block-bootstrap comparisons."""
    data = build_factor_law_dataset(
        base_estimate=base_estimate,
        history_path=history_path,
        dataset=dataset,
        config=model_config,
    )
    first_fold = len(data.factor_scores) - (
        evaluation_config.fold_count * evaluation_config.fold_size
    )
    if first_fold <= max(64, 2 * data.factor_count):
        raise ValueError("too little initial history for requested rolling folds")
    q_scores = {name: [] for name in MODEL_NAMES}
    p_errors = {name: [] for name in MODEL_NAMES}
    spot_correlations = {name: [] for name in MODEL_NAMES}
    evaluation_dates: list[NDArray[np.str_]] = []
    folds: list[dict[str, Any]] = []
    for fold in range(evaluation_config.fold_count):
        start = first_fold + fold * evaluation_config.fold_size
        stop = start + evaluation_config.fold_size
        normalized, fitting = _renormalize_for_origin(
            data,
            fit_stop=start,
            selection_fraction=evaluation_config.selection_fraction,
        )
        fits = _fit_models(fitting, config=model_config)
        evaluation_dates.append(normalized.dates[start:stop])
        fold_report: dict[str, Any] = {
            "fold": fold,
            "fit_end_date": str(normalized.dates[start - 1]),
            "evaluation_start_date": str(normalized.dates[start]),
            "evaluation_end_date": str(normalized.dates[stop - 1]),
            "fit_count": start,
            "selection_train_count": fitting.train_count,
            "selected_epochs": {},
        }
        for fit in fits:
            scales, drift, correlations = _predict(
                fit,
                normalized,
                config=model_config,
            )
            q_point = _q_point_spot_correlated_negative_log_likelihood(
                normalized.factor_scores,
                scales,
                correlations,
            )
            standardized_error = (
                drift - normalized.physical_risk_prices
            ) / normalized.risk_price_scale
            p_point = np.mean(standardized_error**2, axis=1)
            q_scores[fit.architecture].append(q_point[start:stop])
            p_errors[fit.architecture].append(p_point[start:stop])
            spot_correlations[fit.architecture].append(
                correlations[start:stop]
            )
            fold_report["selected_epochs"][fit.architecture] = fit.epochs
        folds.append(fold_report)
        print(
            f"factor-law-evaluation fold={fold + 1}/{evaluation_config.fold_count} "
            f"fit_end={normalized.dates[start - 1]} "
            f"evaluation={normalized.dates[start]}:{normalized.dates[stop - 1]}",
            flush=True,
        )

    dates = np.concatenate(evaluation_dates)
    q_values = {name: np.concatenate(values) for name, values in q_scores.items()}
    p_values = {name: np.concatenate(values) for name, values in p_errors.items()}
    correlation_values = {
        name: np.concatenate(values) for name, values in spot_correlations.items()
    }
    generator = np.random.default_rng(evaluation_config.seed)
    report: dict[str, Any] = {
        "schema": "gromm2-factor-law-rolling-evaluation-v2",
        "base_estimate": str(base_estimate.resolve()),
        "history_path": str(history_path.resolve()),
        "dataset": str(dataset.resolve()),
        "model_config": asdict(model_config),
        "evaluation_config": asdict(evaluation_config),
        "evaluation_count": len(dates),
        "evaluation_start_date": str(dates[0]),
        "evaluation_end_date": str(dates[-1]),
        "folds": folds,
        "metrics": {
            "q": "Gaussian negative log score per factor and day",
            "p": (
                "standardized squared error against the estimated attainable "
                "P-drift coefficient target; this is not a realized hedge P&L"
            ),
        },
        "models": {
            name: {
                "mean_q_negative_log_score": float(np.mean(q_values[name])),
                "mean_p_drift_target_squared_error": float(
                    np.mean(p_values[name])
                ),
                "mean_spot_correlation_vector_norm": float(
                    np.mean(np.linalg.norm(correlation_values[name], axis=1))
                ),
                "maximum_spot_correlation_vector_norm": float(
                    np.max(np.linalg.norm(correlation_values[name], axis=1))
                ),
            }
            for name in MODEL_NAMES
        },
        "skill_vs_ar1": {
            name: {
                "q": _skill_summary(
                    q_values["ar1"],
                    q_values[name],
                    block_length=evaluation_config.block_length,
                    samples=evaluation_config.bootstrap_samples,
                    generator=generator,
                ),
                "p": _skill_summary(
                    p_values["ar1"],
                    p_values[name],
                    block_length=evaluation_config.block_length,
                    samples=evaluation_config.bootstrap_samples,
                    generator=generator,
                ),
            }
            for name in MODEL_NAMES
            if name != "ar1"
        },
        "correlation_skill_vs_diagonal": {
            correlated: {
                "q": _skill_summary(
                    q_values[diagonal],
                    q_values[correlated],
                    block_length=evaluation_config.block_length,
                    samples=evaluation_config.bootstrap_samples,
                    generator=generator,
                ),
                "p": _skill_summary(
                    p_values[diagonal],
                    p_values[correlated],
                    block_length=evaluation_config.block_length,
                    samples=evaluation_config.bootstrap_samples,
                    generator=generator,
                ),
            }
            for diagonal, correlated in (
                ("lstm", "lstm_corr"),
                ("mamba", "mamba_corr"),
            )
        },
    }
    arrays: dict[str, np.ndarray] = {"dates": dates.astype("S10")}
    for name in MODEL_NAMES:
        arrays[f"{name}_q_negative_log_score"] = q_values[name]
        arrays[f"{name}_p_drift_target_squared_error"] = p_values[name]
        arrays[f"{name}_spot_factor_correlations"] = correlation_values[name]
    _atomic_save_npz(output, arrays)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-estimate", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--fold-size", type=int, default=63)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-epochs", type=int, default=2_000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261013)
    arguments = parser.parse_args()
    report = evaluate_factor_laws(
        base_estimate=arguments.base_estimate,
        history_path=arguments.history,
        dataset=arguments.dataset,
        output=arguments.output,
        model_config=FactorLawConfig(
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            hidden_size=arguments.hidden_size,
            seed=arguments.seed,
        ),
        evaluation_config=FactorLawEvaluationConfig(
            fold_count=arguments.fold_count,
            fold_size=arguments.fold_size,
            block_length=arguments.block_length,
            bootstrap_samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
