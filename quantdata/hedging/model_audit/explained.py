"""Identical training/holdout observable-span variance scores for every model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.run import COMMON_INDICES, COMMON_NMS, GRID, _hash, _json
from quantdata.hedging.model_audit.versioned import MODELS


def straddle_risk_weights(log_iv: np.ndarray) -> np.ndarray:
    """Derivative of normalized Black straddle / sqrt(T) with respect to log IV."""
    if log_iv.shape != (7, 7) or not np.isfinite(log_iv).all():
        raise ValueError("Common risk weights require seven tenors and seven finite NMS nodes")
    volatility = np.exp(log_iv)
    total_volatility = volatility * np.sqrt(np.asarray(GRID.expiries_y))[:, None]
    log_strikes = np.log(NMS.from_nms(COMMON_NMS[None, :], np.asarray(GRID.expiries_y)[:, None], vol=volatility[:, [3]]))
    d1 = -log_strikes / total_volatility + .5 * total_volatility
    return (2. * np.exp(-.5 * d1**2) / np.sqrt(2. * np.pi) * volatility).reshape(-1)


def covariance_span(covariance: np.ndarray, weights: np.ndarray, *, rank: int = 8) -> np.ndarray:
    """Eight leading directions of model covariance in the declared observable metric."""
    covariance, weights = np.asarray(covariance), np.asarray(weights)
    if (covariance.shape != (len(weights), len(weights)) or not np.isfinite(covariance).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0.) or not 0 < rank <= len(weights)):
        raise ValueError("Covariance and positive risk weights must align")
    metric = weights[:, None] * .5 * (covariance + covariance.T) * weights[None, :]
    values, vectors = np.linalg.eigh(metric)
    tolerance = 1e-10 * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[0] < -tolerance:
        raise ValueError("The saved model covariance is not positive semidefinite")
    if values[-rank] <= np.finfo(float).eps * max(float(values[-1]), np.finfo(float).tiny):
        raise ValueError("The requested observable span is not identified; do not invent directions")
    return vectors[:, -rank:]


def span_reconstruction(samples: np.ndarray, training_mean: np.ndarray,
                        weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return a weighted-space reconstruction using only the origin's fixed map and mean."""
    centered = (samples - training_mean) * weights
    return training_mean * weights + (centered @ basis) @ basis.T


def centered_energies(actual: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    """Sum centered coordinate energies; the sample-variance denominator cancels."""
    if (actual.shape != fitted.shape or actual.ndim != 2 or len(actual) < 2
            or not np.isfinite(actual).all() or not np.isfinite(fitted).all()):
        raise ValueError("Variance explained requires matching finite observation-by-coordinate arrays")
    error = actual - fitted
    return (float(np.sum((actual - actual.mean(axis=0))**2)),
            float(np.sum((error - error.mean(axis=0))**2)))


def explained_percent(target: float, residual: float) -> float:
    if not np.isfinite([target, residual]).all() or target <= 0. or residual < 0.:
        raise ValueError("Explained variance requires positive target and nonnegative residual energy")
    return 100. * (1. - residual / target)


def run(output: Path) -> dict:
    """Use the original reference/calendar throughout; no missing-date series is substituted."""
    with np.load(output / "reference_history.npz", allow_pickle=False) as saved:
        dates = saved["dates"].astype(str).tolist()
        common = saved["surfaces"][:, :, COMMON_INDICES].reshape(len(dates), -1)
    origins = json.loads((output / "observable_reference.json").read_text())["dates"]
    index_lookup = {text: index for index, text in enumerate(dates)}
    from quantdata.biz_calendar import count_business_days_between

    session_lengths = np.array([0, *[count_business_days_between(pd.Timestamp(left).date(), pd.Timestamp(right).date())
                                    for left, right in zip(dates, dates[1:])]])
    changes = np.diff(common, axis=0, prepend=common[:1])
    rows, details, inputs = [], [], {Path(__file__), output / "reference_history.npz",
                                    output / "observable_reference.json"}
    protocol = {"schema": "common-observable-variance-explained-v1", "models": list(MODELS),
        "origins": origins, "ranks": [8, 16], "training_transitions": 252,
        "training_calendar": "latest 252 exact one-session pairs in the unchanged reference history, through each origin",
        "reference_sha256": _hash(output / "reference_history.npz"),
        "metrics": ["log_iv", "straddle_risk"],
        "straddle_risk": "local fixed-spot/strike/tenor Black straddle log-IV sensitivity divided by sqrt(T); first-order surface return metric, not exact realized dollar P&L",
        "definition": "100 * (1 - sum coordinate centered residual variances / sum coordinate centered target variances)",
        "basis": "leading eight or sixteen directions of each model's conditional observable covariance, in the declared metric; no hindsight fitted coefficients",
        "in_sample": "center within each origin's 252 training returns; pool target/residual sums of squares, not percentages",
        "out_of_sample": "one next-session realization per origin, then center each coordinate over the 44 origins",
        "mean": "trailing training mean only in reconstruction; heldout centering is used only to evaluate variance",
        "not_a_forecast": "future realized scores are projected for representation; predictive covariance scale is reported separately",
        "native_factors": "all rows are scored at both eight and sixteen observable covariance directions; these are not native factor counts or heterogeneous native PCA percentages",
        "data_policy": "original saved reference only; missing native-calendar dates are not filled from another spot series"}
    _json(output / "EXPLAINED_PROTOCOL.json", protocol)
    started = time.monotonic()
    for model in MODELS:
        for metric, rank in ((metric, rank) for metric in ("log_iv", "straddle_risk") for rank in (8, 16)):
            target_energy, error_energy, targets, fitted = 0., 0., [], []
            for text in origins:
                index = index_lookup[text]
                training = np.flatnonzero((session_lengths == 1) & (np.arange(len(dates)) <= index))[-252:]
                if len(training) != 252 or index + 1 >= len(dates) or session_lengths[index + 1] != 1:
                    raise ValueError(f"Original reference has insufficient exact-session coverage for {text}")
                path = output / f"daily/{model.replace(' ', '_')}/{text}.json"
                inputs.add(path)
                record = json.loads(path.read_text())
                if (record["status"] != "scored" or record["origin_date"] != text
                        or record["endpoint"] != dates[index + 1]):
                    raise ValueError(f"Mismatched model forecast {model} {text}")
                actual = changes[index + 1]
                np.testing.assert_allclose(record["actual_change"], actual, atol=1e-14, rtol=0.)
                samples = changes[training]
                mean = samples.mean(axis=0)
                weights = np.ones(common.shape[1]) if metric == "log_iv" else straddle_risk_weights(common[index].reshape(7, 7))
                basis = covariance_span(np.asarray(record["surface_covariance"]), weights, rank=rank)
                reconstruction = span_reconstruction(samples, mean, weights, basis)
                total, residual = centered_energies(samples * weights, reconstruction)
                target_energy += total
                error_energy += residual
                targets.append(actual * weights)
                predicted = span_reconstruction(actual[None], mean, weights, basis)[0]
                fitted.append(predicted)
                details.append({"model": model, "metric": metric, "rank": rank, "origin_date": text,
                    "training_first_endpoint": dates[int(training[0])], "training_last_endpoint": dates[int(training[-1])],
                    "training_pairs": len(training), "train_target_energy": total, "train_error_energy": residual,
                    "train_explained_percent": explained_percent(total, residual),
                    "heldout_target_squared_norm": float(np.sum((actual * weights)**2)),
                    "heldout_error_squared_norm": float(np.sum((actual * weights - predicted)**2))})
            total, residual = centered_energies(np.asarray(targets), np.asarray(fitted))
            rows.append({"model": model, "metric": metric, "rank": rank, "origins": len(origins),
                "train_explained_percent": explained_percent(target_energy, error_energy),
                "oos_explained_percent": explained_percent(total, residual),
                "train_target_energy": target_energy, "train_error_energy": error_energy,
                "oos_target_energy": total, "oos_error_energy": residual,
                "oos_residual_mean_rms": float(np.sqrt(np.mean((np.asarray(targets) - fitted).mean(axis=0)**2)))})
            print(f"common explained {model} {metric} rank={rank} completed={len(rows)}/{4*len(MODELS)} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(output / "explained_variance.csv", index=False)
    pd.DataFrame(details).to_csv(output / "explained_daily.csv", index=False)
    result = {"protocol": protocol, "scores": rows}
    _json(output / "explained_variance.json", result)
    _json(output / "EXPLAINED_INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
