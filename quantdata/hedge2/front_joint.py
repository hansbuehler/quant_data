"""Explicit joint front-spot/surface laws; neither candidate is selected by default."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from functools import cached_property
from typing import Self

import numpy as np
import pandas as pd
from scipy.special import ndtri

from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.front_scores import aligned_front_scores
from quantdata.hedge2.front_spot import DiscreteFrontLaw


def _readonly(values: np.ndarray, *, dtype: type = float) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _samples(scores: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, residuals = _readonly(scores), _readonly(residuals)
    if (scores.ndim != 1 or len(scores) < 2 or residuals.ndim != 2
            or residuals.shape[0] != len(scores) or residuals.shape[1] == 0
            or not np.isfinite(scores).all() or not np.isfinite(residuals).all()):
        raise ValueError("Joint front dependence needs aligned finite score/residual rows")
    return scores, residuals


def _sampling_inputs(reference_forward: float, count: int) -> None:
    if (not np.isfinite(reference_forward) or reference_forward <= 0.
            or isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1):
        raise ValueError("Joint front sampling needs a positive financing forward and integer draw count")


@dataclass(frozen=True, kw_only=True)
class JointFrontDraws:
    next_spots: np.ndarray
    physical_returns: np.ndarray
    factor_residuals: np.ndarray
    normal_spot_driver: np.ndarray
    historical_indices: np.ndarray | None


@dataclass(frozen=True, kw_only=True)
class JointFrontBranches:
    next_spots: np.ndarray
    physical_returns: np.ndarray
    factor_residuals: np.ndarray
    probabilities: np.ndarray
    front_indices: np.ndarray
    historical_indices: np.ndarray


class _FrontDependence:
    """Shared provenance and physical-unit checks; not a default joint-law choice."""

    historical_scores: np.ndarray
    factor_residuals: np.ndarray
    as_of_date: date
    law_kind: str

    @classmethod
    def from_dynamics(cls, *, model: FittedDynamics, scores: pd.DataFrame) -> Self:
        normal = aligned_front_scores(model=model, scores=scores)
        return cls(historical_scores=normal, factor_residuals=model.residuals[:, 1:], as_of_date=model.as_of_date)

    def validate_model(self, model: FittedDynamics) -> None:
        if (self.as_of_date != model.as_of_date
                or not np.array_equal(self.factor_residuals, model.residuals[:, 1:])):
            raise ValueError("The front coupling must belong to the same dated surface innovations")

    @cached_property
    def identity(self) -> str:
        digest = hashlib.sha256(f"hedge2.front_joint.v1/{self.law_kind}/{self.as_of_date.isoformat()}".encode())
        for values in (self.historical_scores, self.factor_residuals):
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _draws(self, *, front: DiscreteFrontLaw, reference_forward: float, normal: np.ndarray,
               factors: np.ndarray, indices: np.ndarray | None) -> JointFrontDraws:
        spots = front.from_normal(normal)
        physical = (spots - front.mean) / reference_forward
        if not np.isfinite(factors).all() or not np.isfinite(physical).all():
            raise ValueError("Joint front sampling produced nonfinite surface or physical-return innovations")
        return JointFrontDraws(next_spots=spots, physical_returns=physical,
            factor_residuals=factors, normal_spot_driver=normal, historical_indices=indices)


@dataclass(frozen=True, kw_only=True)
class EmpiricalFrontCopula(_FrontDependence):
    """Rank-couple a standard-normal spot driver to unchanged empirical factor rows.

    Each historical rank occupies an equal-probability interval. Tied scores
    share their combined interval and select a member uniformly, independently
    of the spot driver inside that interval. This preserves both marginals;
    it is an empirical rank-copula assumption, not a Gaussian diffusion.
    """

    historical_scores: np.ndarray
    factor_residuals: np.ndarray
    as_of_date: date
    law_kind = "front_empirical_rank"

    def __post_init__(self) -> None:
        if not isinstance(self.as_of_date, date):
            raise ValueError("Front dependence requires its fitted model date")
        scores, residuals = _samples(self.historical_scores, self.factor_residuals)
        object.__setattr__(self, "historical_scores", scores)
        object.__setattr__(self, "factor_residuals", residuals)

    @cached_property
    def rank_groups(self) -> tuple[np.ndarray, ...]:
        ordered = np.argsort(self.historical_scores, kind="stable")
        changes = np.flatnonzero(np.diff(self.historical_scores[ordered]) != 0.) + 1
        return tuple(_readonly(group, dtype=np.int64) for group in np.split(ordered, changes))

    def sample(self, *, front: DiscreteFrontLaw, reference_forward: float,
               count: int, rng: np.random.Generator) -> JointFrontDraws:
        _sampling_inputs(reference_forward, count)
        normal = rng.standard_normal(count)
        boundaries = np.cumsum([len(group) for group in self.rank_groups])[:-1]
        lower = boundaries / len(self.historical_scores)
        upper = (len(self.historical_scores) - boundaries) / len(self.historical_scores)
        quantiles = ndtri(np.minimum(lower, upper))
        group_indices = np.searchsorted(np.where(lower <= upper, quantiles, -quantiles), normal, side="left")
        historical = np.empty(count, dtype=np.int64)
        for index, group in enumerate(self.rank_groups):
            selected = np.flatnonzero(group_indices == index)
            if len(group) == 1:
                historical[selected] = group[0]
            else:
                historical[selected] = group[rng.integers(len(group), size=len(selected))]
        return self._draws(front=front, reference_forward=reference_forward, normal=normal,
            factors=self.factor_residuals[historical], indices=historical)

    def branches(self, *, front: DiscreteFrontLaw, reference_forward: float) -> JointFrontBranches:
        """Integrate the finite copula exactly, including tiny tail atoms and ties.

        Wide decimal CDF boundaries prevent tiny upper atoms disappearing when
        added to a cumulative probability near one. Final branch weights remain
        float64; unrepresentable underflow is an error, never silent truncation.
        """
        _sampling_inputs(reference_forward, 1)
        atom_indices, historical_indices, weights = [], [], []
        precision = max(50, int(math.ceil(-math.log10(float(front.probabilities.min())))) + 50)
        with localcontext() as context:
            context.prec = precision
            masses = [Decimal.from_float(float(value)) for value in front.probabilities]
            total = sum(masses, Decimal(0))
            front_bounds = [Decimal(0)]
            for mass in masses:
                front_bounds.append(front_bounds[-1] + mass / total)
            front_bounds[-1] = Decimal(1)
            group_bounds = [Decimal(0)]
            cumulative = 0
            for group in self.rank_groups:
                cumulative += len(group)
                group_bounds.append(Decimal(cumulative) / Decimal(len(self.historical_scores)))
            atom = group_index = 0
            while atom < len(masses) and group_index < len(self.rank_groups):
                left = max(front_bounds[atom], group_bounds[group_index])
                right = min(front_bounds[atom + 1], group_bounds[group_index + 1])
                if right > left:
                    group = self.rank_groups[group_index]
                    weight = float((right - left) / Decimal(len(group)))
                    if not weight > 0.:
                        raise ValueError("A positive joint front branch underflows float64 probability")
                    atom_indices.extend([atom] * len(group))
                    historical_indices.extend(group.tolist())
                    weights.extend([weight] * len(group))
                front_ends = front_bounds[atom + 1] <= group_bounds[group_index + 1]
                group_ends = group_bounds[group_index + 1] <= front_bounds[atom + 1]
                atom += int(front_ends)
                group_index += int(group_ends)
        atoms = np.asarray(atom_indices, dtype=np.int64)
        historical = np.asarray(historical_indices, dtype=np.int64)
        probabilities = np.asarray(weights, dtype=float)
        front_mass = np.bincount(atoms, weights=probabilities, minlength=len(front.cash_support))
        factor_mass = np.bincount(historical, weights=probabilities, minlength=len(self.historical_scores))
        if (not np.isfinite(probabilities).all() or abs(probabilities.sum() - 1.) > 1e-12
                or not np.allclose(front_mass, front.probabilities, rtol=2e-12, atol=0.)
                or not np.allclose(factor_mass, 1. / len(self.historical_scores), rtol=2e-12, atol=0.)):
            raise ValueError("Joint front branch probabilities do not preserve both marginal laws")
        spots = front.cash_support[atoms]
        return JointFrontBranches(next_spots=spots, physical_returns=(spots - front.mean) / reference_forward,
            factor_residuals=self.factor_residuals[historical], probabilities=probabilities,
            front_indices=atoms, historical_indices=historical)

    def innovation_covariance(self, *, front: DiscreteFrontLaw, reference_forward: float) -> np.ndarray:
        branches = self.branches(front=front, reference_forward=reference_forward)
        values = np.c_[branches.physical_returns, branches.factor_residuals]
        values -= branches.probabilities @ values
        return values.T @ (branches.probabilities[:, None] * values)


@dataclass(frozen=True, kw_only=True)
class GaussianFrontCopula(_FrontDependence):
    """Jointly Gaussian spot score/factor innovations, not Gaussian spot returns.

    Standardizing historical scores estimates dependence only. New spot scores
    are genuinely N(0,1); physical spot still comes from the fitted front CDF.
    The factor marginal preserves its empirical mean and population covariance.
    """

    historical_scores: np.ndarray
    factor_residuals: np.ndarray
    as_of_date: date
    law_kind = "front_joint_gaussian"

    def __post_init__(self) -> None:
        if not isinstance(self.as_of_date, date):
            raise ValueError("Front dependence requires its fitted model date")
        scores, residuals = _samples(self.historical_scores, self.factor_residuals)
        if not np.isfinite(scores.std()) or scores.std() <= 0.:
            raise ValueError("Gaussian front dependence requires nonzero historical score variance")
        object.__setattr__(self, "historical_scores", scores)
        object.__setattr__(self, "factor_residuals", residuals)

    @cached_property
    def conditional_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = self.factor_residuals.mean(axis=0)
        centered = self.factor_residuals - mean
        standardized = (self.historical_scores - self.historical_scores.mean()) / self.historical_scores.std()
        loading = standardized @ centered / len(standardized)
        remainder = centered - standardized[:, None] * loading
        _, singular, directions = np.linalg.svd(remainder / np.sqrt(len(remainder)), full_matrices=False)
        root = directions.T * singular
        if not all(np.isfinite(values).all() for values in (mean, loading, root)):
            raise ValueError("Gaussian front dependence produced nonfinite conditional parameters")
        return _readonly(mean), _readonly(loading), _readonly(root)

    def sample(self, *, front: DiscreteFrontLaw, reference_forward: float,
               count: int, rng: np.random.Generator) -> JointFrontDraws:
        _sampling_inputs(reference_forward, count)
        mean, loading, root = self.conditional_parameters
        normal = rng.standard_normal(count)
        independent = rng.standard_normal((count, root.shape[1]))
        factors = mean + normal[:, None] * loading + independent @ root.T
        return self._draws(front=front, reference_forward=reference_forward, normal=normal,
            factors=factors, indices=None)

    def innovation_covariance(self, *, front: DiscreteFrontLaw, reference_forward: float) -> np.ndarray:
        """Exact physical spot/factor covariance, using truncated-normal first moments."""
        _sampling_inputs(reference_forward, 1)
        _, loading, root = self.conditional_parameters
        lower = np.cumsum(front.probabilities)[:-1]
        upper = np.cumsum(front.probabilities[::-1])[::-1][1:]
        thresholds = ndtri(np.minimum(lower, upper))
        density = np.exp(-.5 * thresholds**2) / np.sqrt(2. * np.pi)
        normal_moments = np.r_[0., density] - np.r_[density, 0.]
        spot_normal_covariance = float((front.cash_support - front.mean) @ normal_moments / reference_forward)
        covariance = np.empty((1 + len(loading), 1 + len(loading)))
        covariance[0, 0] = front.variance / reference_forward**2
        covariance[0, 1:] = covariance[1:, 0] = loading * spot_normal_covariance
        covariance[1:, 1:] = np.outer(loading, loading) + root @ root.T
        return covariance
