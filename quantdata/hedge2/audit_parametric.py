"""Independent covariance, objective and Gaussian-integral replay of saved hedges."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from quantdata.hedge2.audit_daily import completed_audit_dates


def integrated_normal_power(std: float, *, risk_aversion: float) -> tuple[float, float]:
    """Adaptive integration of utility, without production truncated-normal formulas.

    Integrate the standard normal on [-12,12], splitting at the cash shift.
    The omitted probability is below 4e-33. A dimensionless cash condition
    divided by lambda*std remains meaningful at small risk aversion.
    """
    if not np.isfinite(std) or std < 0 or risk_aversion <= 0 or not np.isfinite(risk_aversion):
        raise ValueError("Independent Gaussian integration needs nonnegative finite std and positive finite lambda")
    if std == 0:
        return 0., 0.
    scaled = risk_aversion * std
    density_scale = 1. / np.sqrt(2. * np.pi)

    def density(value: float) -> float:
        return density_scale * np.exp(-.5 * value**2)

    def cash_condition(shift: float) -> float:
        below = quad(lambda value: -(value - shift) * density(value), -12., shift, epsabs=1e-11)[0]
        above = quad(lambda value: np.expm1(-scaled * (value - shift)) / scaled * density(value), shift, 12., epsabs=1e-11)[0]
        return below + above

    shift = brentq(cash_condition, -12., 12., xtol=1e-12)
    below = quad(lambda value: ((value - shift) - .5 * scaled * (value - shift)**2) * density(value),
                 -12., shift, epsabs=1e-11)[0]
    above = quad(lambda value: -np.expm1(-scaled * (value - shift)) / scaled * density(value),
                 shift, 12., epsabs=1e-11)[0]
    risk = -std * (shift + below + above)
    lower_derivative = quad(lambda value: -scaled * (value - shift) * value * density(value), -12., shift, epsabs=1e-11)[0]
    upper_derivative = quad(lambda value: np.expm1(-scaled * (value - shift)) * value * density(value), shift, 12., epsabs=1e-11)[0]
    return float(risk), float(-lower_derivative - upper_derivative)


def audit_parametric(*, input_directory: Path, output_directory: Path,
                     completed_only: bool = False) -> dict[str, object]:
    config = json.loads((input_directory / "config.json").read_text())
    if config["method"] != "parametric":
        raise ValueError("This replay checks the full-covariance Gaussian parametric model")
    dates = completed_audit_dates(input_directory=input_directory, configuration=config,
        completed_only=completed_only)
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, power_candidates = [], []
    for index, day in enumerate(dates):
        directory = input_directory / day
        frame = pd.read_parquet(directory / "results.parquet")
        book = pd.read_parquet(directory / "book.parquet")
        metadata = json.loads((directory / "inputs.json").read_text())
        claim_lookup = {claim["label"]: position for position, claim in enumerate(metadata["claims"])}
        with np.load(directory / "positions.npz", allow_pickle=False) as saved:
            positions = saved["positions"]
        with np.load(directory / "risk_inputs.npz", allow_pickle=False) as saved:
            covariance = saved["covariance"]
            exposures, targets = saved["instrument_exposures"], saved["claim_exposures"]
        if (not all(np.isfinite(value).all() for value in (positions, covariance, exposures, targets))
                or np.linalg.eigvalsh(covariance).min() < -1e-12):
            raise ValueError(f"Non-finite or indefinite saved risk geometry on {day}")
        residuals = positions @ exposures + frame["side"].to_numpy()[:, None] * targets[[claim_lookup[name] for name in frame["claim"]]]
        variance = np.einsum("ij,jk,ik->i", residuals, covariance, residuals)
        std = np.sqrt(np.maximum(variance, 0.))
        spot = float(metadata["claims"][0]["initial_spot"])
        prices = np.r_[spot, book["observed_mid"].to_numpy(dtype=float)]
        divisor = prices if config.get("itm_premium_book", False) else np.full(len(prices), spot)
        costs = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float)] / divisor
        entry_costs = np.abs(positions) @ costs
        std_error = float(np.max(np.abs(std - frame["training_std"].to_numpy())))
        cost_error = float(np.max(np.abs(entry_costs - frame["entry_cost"].to_numpy())))
        mean_error = float(np.max(np.abs(frame["training_mean"].to_numpy() + entry_costs)))
        mv = frame["objective"].eq("mean_variance").to_numpy()
        finite = np.isfinite(frame["risk_aversion"].to_numpy())
        multiplier = np.where(finite, frame["risk_aversion"].to_numpy(), 1.)
        predicted = .5 * multiplier * variance + np.where(finite, entry_costs, 0.)
        value_error = float(np.max(np.abs(predicted[mv] - frame["training_objective"].to_numpy()[mv])))
        for row_index in np.flatnonzero(~mv):
            row = frame.iloc[row_index]
            power_candidates.append({"date": day, "row": int(row_index), "std": float(std[row_index]),
                "risk_aversion": float(row["risk_aversion"]), "risk": float(row["training_objective"] - entry_costs[row_index]),
                "effective_mv_aversion": float(row["effective_mv_aversion"])})
        if max(std_error, cost_error, mean_error, value_error) > 1e-8:
            raise RuntimeError(f"Saved risk/objective replay differs on {day}")
        records.append({"date": day, "cells": len(frame), "std_error": std_error, "cost_error": cost_error,
                        "mean_error": mean_error, "mv_objective_error": value_error})
        if (index + 1) % 100 == 0:
            print(f"risk replay dates={index+1}/{len(dates)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    candidates = pd.DataFrame(power_candidates).sort_values(["risk_aversion", "std"])
    selected = candidates.iloc[np.unique(np.linspace(0, len(candidates) - 1, min(100, len(candidates))).astype(int))].copy()
    quadrature = []
    for row in selected.itertuples():
        risk, derivative = integrated_normal_power(row.std, risk_aversion=row.risk_aversion)
        risk_error = abs(risk - row.risk)
        derivative_residual = abs(derivative - row.effective_mv_aversion * row.std)
        if risk_error > 1e-7 or derivative_residual > 1e-6:
            raise RuntimeError(f"Independent Gaussian utility integral differs on {row.date}: {risk_error}, {derivative_residual}")
        quadrature.append({"date": row.date, "row": row.row, "std": row.std, "lambda": row.risk_aversion,
                           "risk_error": risk_error, "derivative_residual": derivative_residual})
    pd.DataFrame(records).to_parquet(output_directory / "risk_replay.parquet", index=False)
    pd.DataFrame(quadrature).to_parquet(output_directory / "gaussian_integral_replay.parquet", index=False)
    summary = {"dates": len(records), "cells": sum(row["cells"] for row in records),
        "requested_dates": len(config["dates"]), "partial_run_audit": len(dates) < len(config["dates"]),
        "maximum_risk_replay_error": max(max(row[name] for name in ("std_error", "cost_error", "mean_error", "mv_objective_error")) for row in records),
        "independent_gaussian_integrals": len(quadrature),
        "maximum_gaussian_risk_error": max(row["risk_error"] for row in quadrature),
        "maximum_gaussian_derivative_residual": max(row["derivative_residual"] for row in quadrature),
        "input_directory": str(input_directory), "elapsed_seconds": time.monotonic()-started}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true")
    print(json.dumps(audit_parametric(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
