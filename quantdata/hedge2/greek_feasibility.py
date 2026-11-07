"""Independent primal/dual certificates for exact Greek matching in a position box."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True, kw_only=True)
class GreekBoxCertificate:
    minimum_maximum_position: float
    primal_positions: np.ndarray
    normalized_primal_residual: float
    normalized_separator: np.ndarray
    dual_lower_bound: float
    duality_gap: float
    declared_cap: float
    separator_target: float
    separator_box_support: float
    separation_margin: float
    tolerance_allowance: float
    certified_infeasible: bool
    solver_message: str


def certify_greek_box(*, exposures: np.ndarray, target_exposures: np.ndarray,
                      position_cap: float, target_side: int = -1,
                      normalized_exposure_tolerance: float = 1e-6) -> GreekBoxCertificate:
    """Minimize maximum absolute holding, then verify a separating inequality.

    Exposures are instrument-by-risk in the SAME position units as the cap.
    This diagnostic neither changes the hedge nor substitutes a relaxed cap.
    It certifies infeasibility only when the independently evaluated box
    support is smaller than the target projection, including the declared
    row-normalized matching tolerance and a floating-point allowance.
    An unspanned target or an uncertified auxiliary solve remains an error,
    not an automatically accepted economic infeasibility classification.
    """
    if (exposures.ndim != 2 or min(exposures.shape) < 1
            or target_exposures.shape != (exposures.shape[1],)
            or not np.isfinite(exposures).all() or not np.isfinite(target_exposures).all()
            or not np.isfinite(position_cap) or position_cap <= 0.
            or target_side not in (-1, 1)
            or not np.isfinite(normalized_exposure_tolerance) or normalized_exposure_tolerance < 0.):
        raise ValueError("Greek-box certificates need finite aligned exposures, positive cap and nonnegative tolerance")
    matrix = exposures.T
    target = -target_side * target_exposures
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(target))
    active = row_scale > 1e-14
    scaled = matrix[active] / row_scale[active, None]
    rhs = target[active] / row_scale[active]
    instruments = exposures.shape[0]
    identity = np.eye(instruments)
    inequalities = np.c_[np.r_[identity, -identity], -np.ones(2 * instruments)]
    equality = np.c_[scaled, np.zeros(len(rhs))]
    result = linprog(c=np.r_[np.zeros(instruments), 1.], A_ub=inequalities,
        b_ub=np.zeros(2 * instruments), A_eq=equality, b_eq=rhs,
        bounds=[(None, None)] * instruments + [(0., None)], method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not result.success or result.x is None:
        raise RuntimeError(f"Independent Greek-box certificate needs review: status={result.status}; {result.message}")
    arguments = dict(scaled=scaled, rhs=rhs, active=active, risk_count=len(target),
        position_cap=position_cap, normalized_exposure_tolerance=normalized_exposure_tolerance)
    try:
        return _checked_box_certificate(result=result, **arguments)
    except RuntimeError as original_error:
        failures = [str(original_error)]
    transforms = [("tighter original rows", np.eye(len(rhs)), "highs-ds")]
    left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    rank_tolerance = np.finfo(float).eps * max(scaled.shape, default=1) * float(singular.max(initial=0.))
    if len(singular) == len(rhs) and len(singular) and np.all(singular > rank_tolerance):
        transforms.append(("invertible SVD row whitening", left.T / singular[:, None], "highs-ipm"))
    for label, transform, method in transforms:
        equality = np.c_[transform @ scaled, np.zeros(len(rhs))]
        refined = linprog(c=np.r_[np.zeros(instruments), 1.], A_ub=inequalities,
            b_ub=np.zeros(2 * instruments), A_eq=equality, b_eq=transform @ rhs,
            bounds=[(None, None)] * instruments + [(0., None)], method=method,
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                     "ipm_optimality_tolerance": 1e-12})
        if not refined.success or refined.x is None:
            failures.append(f"{label}: status={refined.status}; {refined.message}")
            continue
        try:
            return _checked_box_certificate(result=refined, separator_transform=transform,
                original_failure="; ".join(failures), **arguments)
        except RuntimeError as error:
            failures.append(f"{label}: {error}")
    raise RuntimeError("; ".join(failures))


def _checked_box_certificate(*, result, scaled: np.ndarray, rhs: np.ndarray,
                             active: np.ndarray, risk_count: int, position_cap: float,
                             normalized_exposure_tolerance: float,
                             separator_transform: np.ndarray | None = None,
                             original_failure: str | None = None) -> GreekBoxCertificate:
    """Check every candidate in original normalized rows, including mapped duals."""
    instruments = scaled.shape[1]
    positions = np.asarray(result.x[:-1], dtype=float)
    separator = np.asarray(result.eqlin.marginals, dtype=float)
    if (positions.shape != (instruments,) or separator.shape != rhs.shape
            or not np.isfinite(positions).all() or not np.isfinite(separator).all()):
        raise RuntimeError("Independent Greek-box certificate returned non-finite or misaligned evidence")
    if separator_transform is not None:
        separator = separator_transform.T @ separator
        if not np.isfinite(separator).all():
            raise RuntimeError("Mapped Greek-box separator is non-finite")
    maximum_position = float(np.max(np.abs(positions), initial=0.))
    primal_residual = float(np.max(np.abs(scaled @ positions - rhs), initial=0.))
    projected_instruments = scaled.T @ separator
    support_norm = float(np.abs(projected_instruments).sum())
    projected_target = float(separator @ rhs)
    lower_bound = projected_target / max(support_norm, 1.)
    gap = maximum_position - lower_bound
    certificate_tolerance = 1e-7 * max(1., maximum_position, abs(lower_bound))
    if (primal_residual > 1e-7 or gap < -certificate_tolerance or gap > certificate_tolerance
            or not np.isfinite([maximum_position, primal_residual, support_norm,
                               projected_target, lower_bound, gap]).all()):
        raise RuntimeError(f"Independent Greek-box primal/dual check failed: residual={primal_residual}, gap={gap}")
    box_support = position_cap * support_norm
    margin = projected_target - box_support
    roundoff = 128. * np.finfo(float).eps * (
        float(np.abs(separator) @ np.abs(rhs))
        + position_cap * float((np.abs(scaled).T @ np.abs(separator)).sum()))
    allowance = normalized_exposure_tolerance * float(np.abs(separator).sum()) + roundoff
    if not np.isfinite([box_support, margin, roundoff, allowance]).all():
        raise RuntimeError("Independent Greek-box separation arithmetic is non-finite")
    full_separator = np.zeros(risk_count)
    full_separator[active] = separator
    message = str(result.message) if original_failure is None else (
        f"Original numerical certificate failure retained: {original_failure}; "
        f"equivalent-row recovery: {result.message}")
    return GreekBoxCertificate(minimum_maximum_position=maximum_position, primal_positions=positions,
        normalized_primal_residual=primal_residual, normalized_separator=full_separator,
        dual_lower_bound=float(lower_bound), duality_gap=max(float(gap), 0.), declared_cap=position_cap,
        separator_target=projected_target, separator_box_support=box_support,
        separation_margin=float(margin), tolerance_allowance=float(allowance),
        certified_infeasible=bool(margin > allowance), solver_message=message)
