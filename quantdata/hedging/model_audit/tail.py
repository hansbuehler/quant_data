"""Causal rolling Tail Diffusion for the retained floating-DLV PCA/AR fits."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    filter_log_variance_diffusion,
    fit_log_variance_diffusion,
)
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedging.model_audit.run import _hash, _json


def conditional_factor_innovations(dynamics: FittedDynamics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate normal spot innovations from spot-conditioned surface innovations."""
    covariance = dynamics.joint_innovation_covariance
    spot_variance = float(covariance[0, 0])
    if spot_variance <= 0:
        raise ValueError("Tail diffusion requires positive historical spot variance")
    loading = covariance[1:, 0] / spot_variance
    conditional = covariance[1:, 1:] - np.outer(covariance[1:, 0], covariance[0, 1:]) / spot_variance
    factor = np.linalg.cholesky(.5 * (conditional + conditional.T))
    residuals = dynamics.residuals[:, 1:] - dynamics.residuals[:, :1] * loading
    return loading, factor, residuals


@lru_cache(maxsize=1)
def _training_pairs(root: Path) -> tuple[tuple[date, date, int], ...]:
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    observations = pd.read_parquet(directory / "observations.parquet").set_index("date")
    observations.index = pd.to_datetime(observations.index).date
    dates = []
    for path in sorted((directory / "days").glob("*.npz")):
        day = date.fromisoformat(path.stem)
        if day < date(2022, 1, 1):
            continue
        with np.load(path, allow_pickle=False) as saved:
            if bool(saved["reconstruction_passed"]):
                dates.append(day)
    sessions = observations.loc[dates, "session_index"].to_numpy(dtype=int)
    return tuple((dates[index - 1], dates[index], int(sessions[index])) for index in range(1, len(dates))
                 if sessions[index] - sessions[index - 1] == 1)


def fit_tail(dynamics: FittedDynamics, *, root: Path, output: Path, model: str) -> dict:
    """Fit only this origin's 252 innovations, with the prior origin as an extra start."""
    directory = output / "tail_fits" / model.replace(" ", "_")
    destination = directory / f"{dynamics.as_of_date}.json"
    if destination.exists():
        record = json.loads(destination.read_text())
        if not record["fit"]["success"]:
            review_tail_optimizer(dynamics, record, root=root, output=output)
        return record
    loading, factor, residuals = conditional_factor_innovations(dynamics)
    white = solve_triangular(factor, residuals.T, lower=True).T
    radii = np.mean(white**2, axis=1)
    pairs = [pair for pair in _training_pairs(root) if pair[1] <= dynamics.as_of_date][-len(radii):]
    if len(pairs) != len(radii) or pairs[0][0] != dynamics.training_start or pairs[-1][1] != dynamics.training_end:
        raise ValueError("Tail residual chronology does not match its retained rolling AR fit")
    starts = np.r_[True, np.diff([pair[2] for pair in pairs]) != 1]
    starts[0] = True
    previous = sorted(path for path in directory.glob("*.json") if date.fromisoformat(path.stem) < dynamics.as_of_date)
    initial = None
    if previous:
        initial = LogVarianceDiffusionConfig(**json.loads(previous[-1].read_text())["fit"]["config"])
    fit = fit_log_variance_diffusion(radii, starts, dimension=white.shape[1], initial_config=initial)
    record = {"model": model, "origin_date": str(dynamics.as_of_date),
        "training_start": str(dynamics.training_start), "training_end": str(dynamics.training_end),
        "innovation_count": len(radii), "dimension": white.shape[1], "fit": fit.to_dict(step_sessions=1),
        "innovation_endpoints": [str(pair[1]) for pair in pairs],
        "initialisation_date": previous[-1].stem if previous else None,
        "initialisation_sha256": _hash(previous[-1]) if previous else None,
        "initial_next_log_variance_mean": float(fit.filter.next_predicted_means[-1]),
        "initial_next_log_variance_variance": float(fit.filter.next_predicted_variances[-1]),
        "spot_policy": "normal spot; tail scaling on spot-conditioned Gaussian surface innovations only",
        "conditional_spot_loading": loading.tolist(), "conditional_factor": factor.tolist(),
        "segments": int(starts.sum()), "source": "existing fit_log_variance_diffusion; Gaussian log-chi-square measurement approximation"}
    _json(destination, record)
    if not fit.success:
        review_tail_optimizer(dynamics, record, root=root, output=output)
    return record


def review_tail_optimizer(dynamics: FittedDynamics, record: dict, *, root: Path, output: Path) -> dict:
    """Independently check first-order accuracy while retaining the exact solver warning."""
    destination = output / "tail_solver_reviews" / record["model"].replace(" ", "_") / f"{dynamics.as_of_date}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    _, factor, residuals = conditional_factor_innovations(dynamics)
    white = solve_triangular(factor, residuals.T, lower=True).T
    radius = np.mean(white**2, axis=1)
    pairs = [pair for pair in _training_pairs(root)
             if pair[1] <= dynamics.as_of_date][-len(radius):]
    starts = np.r_[True, np.diff([pair[2] for pair in pairs]) != 1]
    names = ("persistence", "stationary_log_variance", "unconditional_scale")
    values = np.asarray([record["fit"]["config"][name] for name in names])
    lower, upper = np.array([0., 1e-8, .05]), np.array([.999, 10., 20.])

    def objective(parameters: np.ndarray) -> float:
        config = LogVarianceDiffusionConfig(**dict(zip(names, parameters, strict=True)))
        return -float(np.mean(filter_log_variance_diffusion(radius, starts,
            dimension=white.shape[1], config=config).log_likelihood))

    gradients = []
    for step in (1e-4, 1e-5):
        gradient = []
        for coordinate in range(len(values)):
            above, below = values.copy(), values.copy()
            above[coordinate] = min(values[coordinate] + step, upper[coordinate])
            below[coordinate] = max(values[coordinate] - step, lower[coordinate])
            derivative = (objective(above) - objective(below)) / (above[coordinate] - below[coordinate])
            if values[coordinate] <= lower[coordinate] + 1e-10 and derivative > 0:
                derivative = 0.
            if values[coordinate] >= upper[coordinate] - 1e-10 and derivative < 0:
                derivative = 0.
            gradient.append(float(derivative))
        gradients.append(gradient)
    norm = float(np.max(np.abs(gradients)))
    result = {"origin_date": str(dynamics.as_of_date), "model": record["model"],
        "exact_solver_message": record["fit"]["message"], "original_solver_success": record["fit"]["success"],
        "projected_gradients": gradients, "finite_difference_steps": [1e-4, 1e-5],
        "maximum_projected_gradient": norm, "accuracy_tolerance": 1e-6,
        "local_accuracy_verified": bool(np.isfinite(norm) and norm <= 1e-6),
        "parameters_changed": False, "scope": "local first-order accuracy of mean log-likelihood, not a global-optimum certificate"}
    _json(destination, result)
    print(f"Tail solver review {record['model']} {dynamics.as_of_date}: message={record['fit']['message']!r} "
          f"projected_gradient={norm:.3g} accuracy_verified={result['local_accuracy_verified']}", flush=True)
    return result


class TailSampler:
    """The paper's scalar log-variance OU, advanced exactly at daily intervals."""

    def __init__(self, dynamics: FittedDynamics, record: dict, *, paths: int, seed: int):
        self.config = LogVarianceDiffusionConfig(**record["fit"]["config"])
        self.loading = np.asarray(record["conditional_spot_loading"])
        self.factor = np.asarray(record["conditional_factor"])
        self.spot_scale = np.sqrt(dynamics.joint_innovation_covariance[0, 0])
        self.random = np.random.default_rng(seed)
        self.scale_random = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.latent = self.scale_random.normal(record["initial_next_log_variance_mean"],
            np.sqrt(record["initial_next_log_variance_variance"]), size=paths)
        self.paths = paths

    def draw(self) -> np.ndarray:
        """Draw one joint spot/factor innovation per path, then advance latent variance."""
        normals = self.random.standard_normal((self.paths, 1 + len(self.loading)))
        spot = self.spot_scale * normals[:, 0]
        multiplier = np.sqrt(self.config.unconditional_scale * np.exp(
            self.latent - .5 * self.config.stationary_log_variance))
        factors = spot[:, None] * self.loading + multiplier[:, None] * (normals[:, 1:] @ self.factor.T)
        self.latent = (self.config.persistence * self.latent
            + self.config.log_variance_innovation_standard_deviation * self.scale_random.standard_normal(self.paths))
        return np.column_stack((spot, factors))
