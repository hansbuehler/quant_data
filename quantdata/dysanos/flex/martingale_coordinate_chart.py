"""Semi-analytic martingale coordinates for a native DYSANOS surface.

For one expiry with ``N`` interior DLV nodes, fix ``N`` cash strikes.  Given
spot, forward/backbone volatilities and the corresponding *cash* call prices,
the discrete density is the solution of the square linear system

``A(S, sigma, eta) q = (1, 1, C_1 / S, ..., C_N / S)``.

The first two rows of ``A`` impose unit mass and unit normalized mean.  Its
remaining rows are Black call values for every discrete support node.  Once
the density tower is known, each DLV row follows algebraically from the SANOS
recursion.  Consequently spot and fixed-contract call prices may be modeled
as driftless coordinates while forward volatilities diffuse freely; the
induced density and DLV drift is whatever this explicit chart requires.

The chart is local.  It is valid while its pricing matrices are nonsingular,
all recovered densities are positive and the recovered DLVs stay inside the
configured open bounds.  These are direct numerical gates, not assumptions.

For a finite one-step transition the companion bounded completion uses the
log-affine forward-volatility law

``f'_j = fbar_j * exp(alpha_j * centered_log_spot + gamma_j * sqrt(h) * z)``.

The Black-backbone forward vol has the same form with its own spot elasticity;
``eta'`` is their ratio.  Conditional density rows are then the solution of a
convex projection subject to branchwise mass and mean, cross-branch fixed-cash
call martingales, positivity and DLV bounds.  The DLV inequalities are linear:

``v_min * box_i * q_i <= C_j(k_i)-C_{j-1}(k_i) <= v_max * box_i * q_i``.

Finally each DLV is recovered by dividing the middle call difference by
``box_i * q_i``.  Thus the finite-step drift map consists only of an explicit
log-affine volatility law, one convex program and an algebraic inverse; it
does not differentiate through the SANOS decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import diags, eye, lil_matrix, vstack

from quantdata.dysanos.flex.config import CappedUnboundedDlvFlexGeometryConfig
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface


@dataclass(frozen=True)
class MartingaleChartDiagnostics:
    """Numerical and admissibility certificates for one chart evaluation."""

    minimum_density: torch.Tensor
    minimum_dlvsq_margin: torch.Tensor
    maximum_dlvsq_margin: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor
    maximum_density_recursion_residual: torch.Tensor
    maximum_absolute_cash_repricing_error: torch.Tensor
    density_system_condition_numbers: torch.Tensor


@dataclass(frozen=True)
class MartingaleChartSurface:
    """A decoded surface recovered from martingale price coordinates."""

    spot: torch.Tensor
    cash_strikes: torch.Tensor
    target_cash_calls: torch.Tensor
    surface: SanosSurface
    diagnostics: MartingaleChartDiagnostics


@dataclass(frozen=True)
class BranchDensityCompletion:
    """Joint minimum-relative-change completion across finite Q branches."""

    densities: torch.Tensor
    condition_numbers: torch.Tensor
    maximum_absolute_relative_change: torch.Tensor
    maximum_constraint_residual: torch.Tensor
    newton_iterations: tuple[int, ...]


@dataclass(frozen=True)
class BoundedBranchDlvCompletion:
    """Linear-program completion with positive bounded DLVs."""

    densities: torch.Tensor
    dlvsqs: torch.Tensor
    selection_method: str
    projection_solver_status: str | None
    weighted_absolute_density_change: float
    weighted_quadratic_relative_density_change: float | None
    weighted_root_mean_square_dlvsq_change: float
    maximum_equality_residual: float
    maximum_inequality_residual: float


@dataclass(frozen=True)
class ImplicitLinearSystemDirectionalDerivatives:
    """First and second derivatives of a parameterized linear system.

    The chart is ``A(y) q(y) = b(y)``.  Derivatives are supplied along a
    finite collection of scalar directions in ``y`` and are recovered from
    the differentiated linear system; no differentiation through a solver is
    required.
    """

    solution: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    condition_numbers: torch.Tensor
    maximum_first_order_equation_residual: torch.Tensor
    maximum_second_order_equation_residual: torch.Tensor


def implicit_linear_system_directional_derivatives(
    *,
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    matrix_first: torch.Tensor,
    matrix_second: torch.Tensor,
    right_hand_side_first: torch.Tensor,
    right_hand_side_second: torch.Tensor,
) -> ImplicitLinearSystemDirectionalDerivatives:
    r"""Differentiate ``A(y)q(y)=b(y)`` once and twice by direction.

    Direction ``f`` satisfies

    ``A q_f = b_f - A_f q``

    and

    ``A q_ff = b_ff - A_ff q - 2 A_f q_f``.

    ``matrix`` may contain leading batch dimensions.  The derivative tensors
    add one leading direction dimension to the corresponding base tensor.
    """

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in one square linear system")
    if right_hand_side.shape != matrix.shape[:-1]:
        raise ValueError("right_hand_side must align with matrix rows")
    expected_matrix_derivative = (matrix_first.shape[0],) + matrix.shape
    expected_rhs_derivative = (matrix_first.shape[0],) + right_hand_side.shape
    if (
        matrix_first.ndim != matrix.ndim + 1
        or matrix_first.shape != expected_matrix_derivative
        or matrix_second.shape != expected_matrix_derivative
        or right_hand_side_first.shape != expected_rhs_derivative
        or right_hand_side_second.shape != expected_rhs_derivative
    ):
        raise ValueError("directional derivatives do not align with the base system")
    values = (
        matrix,
        right_hand_side,
        matrix_first,
        matrix_second,
        right_hand_side_first,
        right_hand_side_second,
    )
    if not all(bool(torch.all(torch.isfinite(value))) for value in values):
        raise ValueError("pricing systems and their derivatives must be finite")

    solution = torch.linalg.solve_ex(matrix, right_hand_side.unsqueeze(-1))
    if not bool(torch.all(solution.info == 0)):
        raise RuntimeError("base martingale-coordinate system is singular")
    base_solution = solution.result.squeeze(-1)
    first_values: list[torch.Tensor] = []
    second_values: list[torch.Tensor] = []
    maximum_first_residual = torch.zeros(
        (), dtype=matrix.dtype, device=matrix.device
    )
    maximum_second_residual = torch.zeros_like(maximum_first_residual)
    for direction in range(matrix_first.shape[0]):
        first_rhs = right_hand_side_first[direction] - torch.einsum(
            "...ij,...j->...i", matrix_first[direction], base_solution
        )
        first_solution = torch.linalg.solve_ex(matrix, first_rhs.unsqueeze(-1))
        if not bool(torch.all(first_solution.info == 0)):
            raise RuntimeError(f"first derivative system {direction} is singular")
        first = first_solution.result.squeeze(-1)
        second_rhs = (
            right_hand_side_second[direction]
            - torch.einsum(
                "...ij,...j->...i", matrix_second[direction], base_solution
            )
            - 2.0
            * torch.einsum(
                "...ij,...j->...i", matrix_first[direction], first
            )
        )
        second_solution = torch.linalg.solve_ex(matrix, second_rhs.unsqueeze(-1))
        if not bool(torch.all(second_solution.info == 0)):
            raise RuntimeError(f"second derivative system {direction} is singular")
        second = second_solution.result.squeeze(-1)
        first_residual = (
            torch.einsum("...ij,...j->...i", matrix, first)
            + torch.einsum(
                "...ij,...j->...i", matrix_first[direction], base_solution
            )
            - right_hand_side_first[direction]
        )
        second_residual = (
            torch.einsum("...ij,...j->...i", matrix, second)
            + torch.einsum(
                "...ij,...j->...i", matrix_second[direction], base_solution
            )
            + 2.0
            * torch.einsum(
                "...ij,...j->...i", matrix_first[direction], first
            )
            - right_hand_side_second[direction]
        )
        maximum_first_residual = torch.maximum(
            maximum_first_residual, torch.amax(torch.abs(first_residual))
        )
        maximum_second_residual = torch.maximum(
            maximum_second_residual, torch.amax(torch.abs(second_residual))
        )
        first_values.append(first)
        second_values.append(second)

    singular_values = torch.linalg.svdvals(matrix)
    condition = singular_values[..., 0] / torch.clamp_min(
        singular_values[..., -1], torch.finfo(matrix.dtype).tiny
    )
    return ImplicitLinearSystemDirectionalDerivatives(
        solution=base_solution,
        first=torch.stack(first_values),
        second=torch.stack(second_values),
        condition_numbers=condition,
        maximum_first_order_equation_residual=maximum_first_residual,
        maximum_second_order_equation_residual=maximum_second_residual,
    )


def black_call_matrix(
    support: torch.Tensor,
    relative_strikes: torch.Tensor,
    backbone_total_vols: torch.Tensor,
) -> torch.Tensor:
    """Return Black call values with rows=strikes and columns=support nodes."""

    if support.ndim != 3:
        raise ValueError("support must have shape (batch, expiries, nodes)")
    if relative_strikes.ndim != 3 or relative_strikes.shape[:2] != support.shape[:2]:
        raise ValueError("relative strikes must align with batch and expiries")
    if backbone_total_vols.shape != support.shape[:2]:
        raise ValueError("backbone total vols must align with batch and expiries")
    if not bool(
        torch.all(torch.isfinite(support) & (support > 0.0))
        and torch.all(torch.isfinite(relative_strikes) & (relative_strikes > 0.0))
        and torch.all(torch.isfinite(backbone_total_vols) & (backbone_total_vols >= 0.0))
    ):
        raise ValueError("Black matrix inputs must be finite and in their positive domains")
    safe_vol = torch.clamp_min(
        backbone_total_vols,
        torch.finfo(support.dtype).tiny,
    )
    node = support.unsqueeze(-2)
    strike = relative_strikes.unsqueeze(-1)
    volatility = safe_vol.unsqueeze(-1).unsqueeze(-1)
    d1 = (torch.log(node / strike) + 0.5 * volatility.square()) / volatility
    black = node * torch.special.ndtr(d1) - strike * torch.special.ndtr(
        d1 - volatility
    )
    return torch.where(
        backbone_total_vols[..., None, None] > 0.0,
        black,
        torch.relu(node - strike),
    )


def recover_density_from_cash_calls(
    *,
    spot: torch.Tensor,
    support: torch.Tensor,
    backbone_total_vols: torch.Tensor,
    cash_strikes: torch.Tensor,
    cash_calls: torch.Tensor,
    reference_density: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert mass, mean and fixed-cash call prices into discrete densities.

    When fewer than ``N`` calls are supplied for ``N`` interior DLV nodes,
    ``reference_density`` selects the unique minimum-relative-change solution
    ``q = q_ref * (1 + y)``.  This is the closed weighted right inverse of the
    exact linear constraints and uses the unused density directions rather
    than trying to identify tail masses from the traded sieve.
    """

    if spot.ndim != 1 or support.ndim != 3:
        raise ValueError("spot and support must have shapes (batch,) and (batch,T,N+2)")
    batch, expiries, support_size = support.shape
    num_inner = support_size - 2
    if (
        cash_strikes.ndim != 3
        or cash_strikes.shape[:2] != (batch, expiries)
        or cash_calls.shape != cash_strikes.shape
    ):
        raise ValueError(
            "cash strikes and calls must align with batch and expiries, found "
            f"{cash_strikes.shape} and {cash_calls.shape}"
        )
    num_calls = cash_strikes.shape[-1]
    if num_calls <= 0 or num_calls > num_inner:
        raise ValueError("the chart needs between one and N fixed calls per expiry")
    if backbone_total_vols.shape != (batch, expiries):
        raise ValueError("backbone total vols have an invalid shape")
    if not bool(
        torch.all(torch.isfinite(spot) & (spot > 0.0))
        and torch.all(torch.isfinite(cash_strikes) & (cash_strikes > 0.0))
        and torch.all(torch.isfinite(cash_calls) & (cash_calls >= 0.0))
    ):
        raise ValueError("spot, strikes and calls must be finite and nonnegative")

    relative_strikes = cash_strikes / spot[:, None, None]
    normalized_calls = cash_calls / spot[:, None, None]
    pricing = black_call_matrix(
        support,
        relative_strikes,
        backbone_total_vols,
    )
    matrix = torch.cat(
        (
            torch.ones_like(support[:, :, None, :]),
            support[:, :, None, :],
            pricing,
        ),
        dim=2,
    )
    right_hand_side = torch.cat(
        (
            torch.ones(
                (batch, expiries, 2),
                dtype=support.dtype,
                device=support.device,
            ),
            normalized_calls,
        ),
        dim=-1,
    )
    if reference_density is None:
        if num_calls != num_inner:
            raise ValueError("an underdetermined chart requires reference_density")
        solution = torch.linalg.solve_ex(matrix, right_hand_side.unsqueeze(-1))
        if not bool(torch.all(solution.info == 0)):
            failed = torch.nonzero(solution.info != 0).tolist()
            raise RuntimeError(
                f"martingale-coordinate density systems are singular at {failed}"
            )
        density = solution.result.squeeze(-1)
        condition_matrix = matrix
    else:
        if reference_density.shape != support.shape or not bool(
            torch.all(torch.isfinite(reference_density) & (reference_density > 0.0))
        ):
            raise ValueError("reference density must be finite, positive and match support")
        residual = right_hand_side - torch.einsum(
            "btrn,btn->btr",
            matrix,
            reference_density,
        )
        weighted = matrix * reference_density.unsqueeze(-2)
        left, singular_values, right_t = torch.linalg.svd(
            weighted,
            full_matrices=False,
        )
        tolerance = (
            torch.finfo(weighted.dtype).eps
            * max(weighted.shape[-2:])
            * singular_values[..., :1]
        )
        if not bool(torch.all(singular_values[..., -1:] > tolerance)):
            raise RuntimeError("weighted martingale-coordinate systems are rank deficient")
        coordinates = (left.mT @ residual.unsqueeze(-1)).squeeze(-1)
        relative_change = (
            right_t.mT @ (coordinates / singular_values).unsqueeze(-1)
        ).squeeze(-1)
        density = reference_density * (1.0 + relative_change)
        condition_matrix = weighted
    singular_values = torch.linalg.svdvals(condition_matrix)
    condition = singular_values[..., 0] / torch.clamp_min(
        singular_values[..., -1],
        torch.finfo(support.dtype).tiny,
    )
    return density, condition


def complete_branch_densities_to_martingales(
    *,
    branch_probabilities: torch.Tensor,
    spot: torch.Tensor,
    support: torch.Tensor,
    backbone_total_vols: torch.Tensor,
    reference_densities: torch.Tensor,
    cash_strikes: torch.Tensor,
    target_mean_cash_calls: torch.Tensor,
    maximum_newton_iterations: int = 50,
    constraint_tolerance: float = 2.0e-12,
) -> BranchDensityCompletion:
    """Complete finite branches so spot and selected cash calls are martingales.

    Reference branches may have arbitrary stochastic spot, forward-volatility
    and DLV moves. This routine computes their minimum-relative-entropy
    projection subject to, branch by branch, unit mass and unit normalized
    mean, and, across branches, the requested mean cash call prices. The dual
    gradient and Hessian are available analytically, and the exponential
    family keeps every density strictly positive without clipping.
    """

    if spot.ndim != 1 or support.ndim != 3:
        raise ValueError("spot and support must have shapes (branches,) and (branches,T,M)")
    branches, expiries, support_size = support.shape
    if branch_probabilities.shape != (branches,) or backbone_total_vols.shape != (
        branches,
        expiries,
    ):
        raise ValueError("branch probabilities or backbone vols have invalid shapes")
    if reference_densities.shape != support.shape:
        raise ValueError("reference densities must align with branch supports")
    if (
        cash_strikes.ndim != 3
        or cash_strikes.shape[:2] != (branches, expiries)
        or target_mean_cash_calls.shape != (expiries, cash_strikes.shape[-1])
    ):
        raise ValueError("cash strike or target-call geometry is invalid")
    if not bool(
        torch.all(torch.isfinite(branch_probabilities) & (branch_probabilities > 0.0))
        and torch.abs(torch.sum(branch_probabilities) - 1.0)
        <= 64.0 * torch.finfo(support.dtype).eps
        and torch.all(torch.isfinite(reference_densities) & (reference_densities > 0.0))
    ):
        raise ValueError("branch probabilities and reference densities must be positive")
    if maximum_newton_iterations <= 0 or constraint_tolerance <= 0.0:
        raise ValueError("Newton iteration count and tolerance must be positive")

    relative_strikes = cash_strikes / spot[:, None, None]
    pricing = black_call_matrix(support, relative_strikes, backbone_total_vols)
    completed: list[torch.Tensor] = []
    conditions: list[torch.Tensor] = []
    maximum_change = torch.zeros((), dtype=support.dtype, device=support.device)
    maximum_residual = torch.zeros_like(maximum_change)
    iterations: list[int] = []
    num_calls = cash_strikes.shape[-1]
    variables = branches * support_size
    for expiry_index in range(expiries):
        constraint = torch.zeros(
            (2 * branches + num_calls, variables),
            dtype=support.dtype,
            device=support.device,
        )
        right_hand_side = torch.zeros(
            2 * branches + num_calls,
            dtype=support.dtype,
            device=support.device,
        )
        for branch in range(branches):
            block = slice(branch * support_size, (branch + 1) * support_size)
            density = reference_densities[branch, expiry_index]
            constraint[2 * branch, block] = 1.0
            constraint[2 * branch + 1, block] = support[branch, expiry_index]
            right_hand_side[2 * branch : 2 * branch + 2] = 1.0
            cash_kernel = (
                branch_probabilities[branch]
                * spot[branch]
                * pricing[branch, expiry_index]
            )
            constraint[2 * branches :, block] = cash_kernel
        right_hand_side[2 * branches :] = target_mean_cash_calls[expiry_index]
        reference = reference_densities[:, expiry_index].reshape(-1)
        row_scale = torch.sqrt(
            torch.sum(constraint.square() * reference[None], dim=-1)
        )
        if not bool(torch.all(row_scale > 0.0)):
            raise RuntimeError("branch completion contains an empty constraint row")
        scaled_constraint = constraint / row_scale[:, None]
        scaled_target = right_hand_side / row_scale
        weighted_constraint = scaled_constraint * torch.sqrt(reference)[None]
        left, singular_values, _ = torch.linalg.svd(
            weighted_constraint,
            full_matrices=False,
        )
        rank_tolerance = (
            torch.finfo(support.dtype).eps
            * max(weighted_constraint.shape)
            * singular_values[0]
        )
        if not bool(torch.all(singular_values > rank_tolerance)):
            raise RuntimeError(
                f"branch constraints are rank deficient at expiry {expiry_index}"
            )
        whitener = (left / singular_values[None]).mT
        dual_constraint = whitener @ scaled_constraint
        dual_target = whitener @ scaled_target
        multiplier = torch.zeros_like(dual_target)

        def evaluate(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            exponent = dual_constraint.mT @ candidate
            if not bool(torch.all(torch.isfinite(exponent))):
                raise RuntimeError("branch entropy exponent became nonfinite")
            density_value = reference * torch.exp(exponent)
            residual_value = dual_constraint @ density_value - dual_target
            dual_value = torch.sum(density_value) - torch.dot(dual_target, candidate)
            return density_value, residual_value, dual_value

        density_flat, residual, dual = evaluate(multiplier)
        converged = False
        for iteration in range(maximum_newton_iterations + 1):
            original_residual = constraint @ density_flat - right_hand_side
            if bool(torch.amax(torch.abs(original_residual)) <= constraint_tolerance):
                converged = True
                iterations.append(iteration)
                break
            if iteration == maximum_newton_iterations:
                break
            hessian = (dual_constraint * density_flat[None]) @ dual_constraint.mT
            step = torch.linalg.solve_ex(hessian, residual.unsqueeze(-1))
            if not bool(step.info == 0):
                raise RuntimeError(
                    f"branch entropy Hessian is singular at expiry {expiry_index}"
                )
            direction = step.result.squeeze(-1)
            damping = 1.0
            accepted = False
            directional_decrease = torch.dot(residual, direction)
            for _ in range(20):
                trial = multiplier - damping * direction
                try:
                    trial_density, trial_residual, trial_dual = evaluate(trial)
                except RuntimeError:
                    damping *= 0.5
                    continue
                if bool(
                    torch.isfinite(trial_dual)
                    & (
                        trial_dual
                        <= dual - 1.0e-4 * damping * directional_decrease
                    )
                ):
                    multiplier = trial
                    density_flat = trial_density
                    residual = trial_residual
                    dual = trial_dual
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"branch entropy line search failed at expiry {expiry_index}"
                )
        if not converged:
            worst = float(
                torch.amax(torch.abs(constraint @ density_flat - right_hand_side))
            )
            raise RuntimeError(
                f"branch entropy solve did not converge at expiry {expiry_index}; "
                f"residual={worst:.6g}"
            )
        density = density_flat.reshape(branches, support_size)
        relative_change = density / reference_densities[:, expiry_index] - 1.0
        completed.append(density)
        hessian = (dual_constraint * density_flat[None]) @ dual_constraint.mT
        singular = torch.linalg.svdvals(hessian)
        conditions.append(singular[0] / singular[-1])
        maximum_change = torch.maximum(
            maximum_change,
            torch.amax(torch.abs(relative_change)),
        )
        maximum_residual = torch.maximum(
            maximum_residual,
            torch.amax(torch.abs(constraint @ density_flat - right_hand_side)),
        )
    return BranchDensityCompletion(
        densities=torch.stack(completed, dim=1),
        condition_numbers=torch.stack(conditions),
        maximum_absolute_relative_change=maximum_change,
        maximum_constraint_residual=maximum_residual,
        newton_iterations=tuple(iterations),
    )


def complete_branch_dlvs_to_martingales(
    decoder: FlexSanosDecoder,
    *,
    branch_probabilities: torch.Tensor,
    spot: torch.Tensor,
    support: torch.Tensor,
    density0: torch.Tensor,
    backbone_total_vols: torch.Tensor,
    reference_densities: torch.Tensor,
    cash_strikes: torch.Tensor,
    target_mean_cash_calls: torch.Tensor,
    density_floor: float = 1.0e-5,
    dlvsq_bound_margin: float = 1.0e-4,
    smooth_quadratic_projection: bool = False,
    dlvsq_projection_weight: float = 0.0,
) -> BoundedBranchDlvCompletion:
    """Complete a finite branch law by one exact linear program.

    In addition to branchwise unit mass and mean and aggregate call-martingale
    equalities, this imposes the SANOS DLV lower and upper bounds directly.
    Those bounds are linear in consecutive density rows because call values
    and the density recursion are linear in the densities. A weighted-L1
    solve supplies a feasible point. Optionally, a convex chi-square
    projection selects a smoother point close to the reference density tower.
    """

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("bounded branch completion requires capped DLV geometry")
    if support.device.type != "cpu":
        raise ValueError("the linear-program completion currently requires CPU tensors")
    branches, expiries, support_size = support.shape
    num_inner = support_size - 2
    num_calls = cash_strikes.shape[-1]
    expected_surface_shape = (branches, expiries, support_size)
    if reference_densities.shape != expected_surface_shape:
        raise ValueError("reference densities must align with branch supports")
    if density0.shape != (branches, support_size):
        raise ValueError("density0 must align with the first branch support")
    if backbone_total_vols.shape != (branches, expiries):
        raise ValueError("backbone total vols must align with branch supports")
    if cash_strikes.shape != (branches, expiries, num_calls):
        raise ValueError("cash strikes must align with branches and expiries")
    if target_mean_cash_calls.shape != (expiries, num_calls):
        raise ValueError("target calls must align with expiries and strikes")
    if branch_probabilities.shape != (branches,) or spot.shape != (branches,):
        raise ValueError("branch probabilities and spot have invalid shapes")
    if (
        density_floor <= 0.0
        or dlvsq_bound_margin <= 0.0
        or not np.isfinite(dlvsq_projection_weight)
        or dlvsq_projection_weight < 0.0
    ):
        raise ValueError("density floor and DLV bound margin must be positive")
    lower_dlvsq = geometry.min_dlvsq + dlvsq_bound_margin
    upper_dlvsq = geometry.max_dlvsq - dlvsq_bound_margin
    if lower_dlvsq >= upper_dlvsq:
        raise ValueError("DLV bound margin leaves an empty interval")

    relative_strikes = cash_strikes / spot[:, None, None]
    pricing = black_call_matrix(support, relative_strikes, backbone_total_vols)
    reference_dlvsqs = recover_dlvsqs_from_densities(
        decoder,
        support=support,
        density0=density0,
        densities=reference_densities,
    )
    num_densities = branches * expiries * support_size
    num_variables = 2 * num_densities

    def density_index(expiry: int, branch: int, node: int = 0) -> int:
        return (expiry * branches + branch) * support_size + node

    equality_rows = expiries * (2 * branches + num_calls)
    equality = lil_matrix((equality_rows, num_variables), dtype=np.float64)
    equality_target = np.zeros(equality_rows, dtype=np.float64)
    for expiry in range(expiries):
        row0 = expiry * (2 * branches + num_calls)
        for branch in range(branches):
            block = slice(
                density_index(expiry, branch),
                density_index(expiry, branch) + support_size,
            )
            equality[row0 + 2 * branch, block] = 1.0
            equality[row0 + 2 * branch + 1, block] = (
                support[branch, expiry].detach().numpy()
            )
            equality_target[row0 + 2 * branch : row0 + 2 * branch + 2] = 1.0
            cash_kernel = (
                branch_probabilities[branch]
                * spot[branch]
                * pricing[branch, expiry]
            )
            equality[
                row0 + 2 * branches : row0 + 2 * branches + num_calls,
                block,
            ] = cash_kernel.detach().numpy()[:, None]
        equality_target[
            row0 + 2 * branches : row0 + 2 * branches + num_calls
        ] = target_mean_cash_calls[expiry].detach().numpy()

    dlv_rows = 2 * branches * expiries * num_inner
    dlv_projection_rows = branches * expiries * num_inner
    dlv_projection = lil_matrix(
        (dlv_projection_rows, num_densities),
        dtype=np.float64,
    )
    dlv_projection_target = np.zeros(dlv_projection_rows, dtype=np.float64)
    dlv_projection_scale = np.zeros(dlv_projection_rows, dtype=np.float64)
    inequality_rows = dlv_rows + 2 * num_densities
    inequality = lil_matrix((inequality_rows, num_variables), dtype=np.float64)
    inequality_target = np.zeros(inequality_rows, dtype=np.float64)
    row = 0
    for expiry in range(expiries):
        for branch in range(branches):
            current_support = support[branch, expiry].detach().numpy()
            inner = current_support[1:-1]
            current_call_kernel = np.maximum(
                current_support[None, :] - inner[:, None],
                0.0,
            )
            box = (
                inner**2
                * float(decoder.model_dts_y[expiry])
                / (current_support[2:] - current_support[:-2])
            )
            if expiry == 0:
                previous_support = support[branch, 0].detach().numpy()
                previous_call = np.maximum(
                    previous_support[None, :] - inner[:, None],
                    0.0,
                ) @ density0[branch].detach().numpy()
            else:
                previous_support = support[branch, expiry - 1].detach().numpy()
                previous_call_kernel = np.maximum(
                    previous_support[None, :] - inner[:, None],
                    0.0,
                )
                previous_block = slice(
                    density_index(expiry - 1, branch),
                    density_index(expiry - 1, branch) + support_size,
                )
            current_block = slice(
                density_index(expiry, branch),
                density_index(expiry, branch) + support_size,
            )
            for node in range(num_inner):
                projection_row = (
                    (expiry * branches + branch) * num_inner + node
                )
                projection = np.array(current_call_kernel[node], copy=True)
                projection[node + 1] -= (
                    float(reference_dlvsqs[branch, expiry, node]) * box[node]
                )
                dlv_projection[projection_row, current_block] = projection
                dlv_projection_scale[projection_row] = (
                    box[node]
                    * float(reference_densities[branch, expiry, node + 1])
                )
                if expiry == 0:
                    dlv_projection_target[projection_row] = previous_call[node]
                else:
                    dlv_projection[projection_row, previous_block] = (
                        -previous_call_kernel[node]
                    )

                lower = np.array(-current_call_kernel[node], copy=True)
                lower[node + 1] += lower_dlvsq * box[node]
                inequality[row, current_block] = lower
                if expiry == 0:
                    inequality_target[row] = -previous_call[node]
                else:
                    inequality[row, previous_block] = previous_call_kernel[node]
                row += 1

                upper = np.array(current_call_kernel[node], copy=True)
                upper[node + 1] -= upper_dlvsq * box[node]
                inequality[row, current_block] = upper
                if expiry == 0:
                    inequality_target[row] = previous_call[node]
                else:
                    inequality[row, previous_block] = -previous_call_kernel[node]
                row += 1

    reference = (
        reference_densities.permute(1, 0, 2).reshape(-1).detach().numpy()
    )
    identity = eye(num_densities, format="csr")
    inequality[row : row + num_densities, :num_densities] = identity
    inequality[row : row + num_densities, num_densities:] = -identity
    inequality_target[row : row + num_densities] = reference
    row += num_densities
    inequality[row : row + num_densities, :num_densities] = -identity
    inequality[row : row + num_densities, num_densities:] = -identity
    inequality_target[row : row + num_densities] = -reference

    probability_weights = np.tile(
        np.repeat(branch_probabilities.detach().numpy(), support_size),
        expiries,
    )
    objective = np.concatenate((np.zeros(num_densities), probability_weights))
    equality_csr = equality.tocsr()
    equality_scale = np.sqrt(
        np.asarray(equality_csr.multiply(equality_csr).sum(axis=1)).reshape(-1)
    )
    scaled_equality = diags(1.0 / equality_scale) @ equality_csr
    scaled_equality_target = equality_target / equality_scale
    dlv_inequality = inequality[:dlv_rows].tocsr()
    dlv_scale = np.sqrt(
        np.asarray(dlv_inequality.multiply(dlv_inequality).sum(axis=1)).reshape(-1)
    )
    scaled_inequality = vstack(
        (
            diags(1.0 / dlv_scale) @ dlv_inequality,
            inequality[dlv_rows:].tocsr(),
        ),
        format="csr",
    )
    scaled_inequality_target = np.concatenate(
        (
            inequality_target[:dlv_rows] / dlv_scale,
            inequality_target[dlv_rows:],
        )
    )
    result = linprog(
        objective,
        A_ub=scaled_inequality,
        b_ub=scaled_inequality_target,
        A_eq=scaled_equality,
        b_eq=scaled_equality_target,
        bounds=[(density_floor, None)] * num_densities
        + [(0.0, None)] * num_densities,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1.0e-10,
            "primal_feasibility_tolerance": 1.0e-10,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "bounded martingale completion linear program failed: "
            f"{result.message}"
        )
    density_values = result.x[:num_densities]
    quadratic_weights = probability_weights / np.maximum(reference, density_floor)
    projection_solver_status: str | None = None
    if smooth_quadratic_projection:
        density_variable = cp.Variable(num_densities)
        density_variable.value = density_values
        quadratic_objective = 0.5 * cp.sum(
            cp.multiply(
                quadratic_weights,
                cp.square(density_variable - reference),
            )
        )
        if dlvsq_projection_weight > 0.0:
            normalized_dlv_projection = (
                diags(1.0 / dlv_projection_scale) @ dlv_projection.tocsr()
            )
            normalized_dlv_target = (
                dlv_projection_target / dlv_projection_scale
            )
            projection_weights = np.tile(
                np.repeat(branch_probabilities.detach().numpy(), num_inner),
                expiries,
            ) / expiries / num_inner
            quadratic_objective += (
                0.5
                * dlvsq_projection_weight
                * cp.sum(
                    cp.multiply(
                        projection_weights,
                        cp.square(
                            normalized_dlv_projection @ density_variable
                            - normalized_dlv_target
                        ),
                    )
                )
            )
        quadratic_problem = cp.Problem(
            cp.Minimize(quadratic_objective),
            (
                scaled_equality[:, :num_densities] @ density_variable
                == scaled_equality_target,
                scaled_inequality[:dlv_rows, :num_densities] @ density_variable
                <= scaled_inequality_target[:dlv_rows],
                density_variable >= density_floor,
            ),
        )
        quadratic_problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1.0e-9,
            tol_feas=1.0e-9,
            max_iter=500,
            warm_start=True,
            verbose=False,
        )
        projection_solver_status = quadratic_problem.status
        if quadratic_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                "bounded martingale completion quadratic projection failed: "
                f"{quadratic_problem.status}"
            )
        density_values = np.asarray(density_variable.value)
    density = torch.as_tensor(
        density_values.reshape(expiries, branches, support_size)
        .transpose(1, 0, 2)
        .copy(),
        dtype=support.dtype,
        device=support.device,
    )
    dlvsqs = recover_dlvsqs_from_densities(
        decoder,
        support=support,
        density0=density0,
        densities=density,
    )
    equality_residual = equality[:, :num_densities] @ density_values - equality_target
    inequality_residual = (
        inequality[:dlv_rows, :num_densities] @ density_values
        - inequality_target[:dlv_rows]
    )
    return BoundedBranchDlvCompletion(
        densities=density,
        dlvsqs=dlvsqs,
        selection_method=(
            (
                "convex-dlv-chi-square"
                if dlvsq_projection_weight > 0.0
                else "convex-chi-square"
            )
            if smooth_quadratic_projection
            else "weighted-L1"
        ),
        projection_solver_status=projection_solver_status,
        weighted_absolute_density_change=float(
            np.dot(probability_weights, np.abs(density_values - reference))
        ),
        weighted_quadratic_relative_density_change=(
            float(
                0.5
                * np.dot(
                    quadratic_weights,
                    np.square(density_values - reference),
                )
            )
            if smooth_quadratic_projection
            else None
        ),
        weighted_root_mean_square_dlvsq_change=float(
            torch.sqrt(
                torch.sum(
                    branch_probabilities[:, None, None]
                    * (dlvsqs - reference_dlvsqs).square()
                )
                / expiries
                / num_inner
            )
        ),
        maximum_equality_residual=float(np.max(np.abs(equality_residual))),
        maximum_inequality_residual=float(np.max(inequality_residual)),
    )


def _calls_on_nodes(
    source_support: torch.Tensor,
    source_density: torch.Tensor,
    target_strikes: torch.Tensor,
) -> torch.Tensor:
    """Price unsmoothed calls from one discrete law at target relative strikes."""

    return torch.sum(
        torch.relu(source_support.unsqueeze(-2) - target_strikes.unsqueeze(-1))
        * source_density.unsqueeze(-2),
        dim=-1,
    )


def _price_cash_call_grid(
    decoder: FlexSanosDecoder,
    surface: SanosSurface,
    *,
    spot: torch.Tensor,
    cash_strikes: torch.Tensor,
) -> torch.Tensor:
    """Price an expiry-by-strike cash grid through the flat decoder API."""

    batch, expiries, num_strikes = cash_strikes.shape
    relative = cash_strikes / spot[:, None, None]
    query_expiries = decoder.model_expiries_y.reshape(1, expiries, 1).expand(
        batch,
        expiries,
        num_strikes,
    )
    normalized = decoder.price_calls_batch(
        surface,
        expiries_y=query_expiries.reshape(batch, -1),
        strikes=relative.reshape(batch, -1),
    ).reshape(batch, expiries, num_strikes)
    return spot[:, None, None] * normalized


def recover_dlvsqs_from_densities(
    decoder: FlexSanosDecoder,
    *,
    support: torch.Tensor,
    density0: torch.Tensor,
    densities: torch.Tensor,
) -> torch.Tensor:
    """Recover the exact DLV tower from a density tower without differentiation."""

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("the martingale chart requires open bounded DLV coordinates")
    if support.ndim != 3 or densities.shape != support.shape:
        raise ValueError("support and densities must share shape (batch,T,N+2)")
    batch, expiries, support_size = support.shape
    if density0.shape != (batch, support_size):
        raise ValueError("density0 must align with the first support")
    if expiries != geometry.num_model_expiries or support_size != (
        geometry.num_inner_strikes + 2
    ):
        raise ValueError("density tower does not match the decoder geometry")

    dlvsq_rows: list[torch.Tensor] = []
    previous_support = support[:, 0]
    previous_density = density0
    for expiry_index in range(expiries):
        current_support = support[:, expiry_index]
        current_density = densities[:, expiry_index]
        inner = current_support[:, 1:-1]
        current_calls = _calls_on_nodes(current_support, current_density, inner)
        previous_calls = _calls_on_nodes(previous_support, previous_density, inner)
        box = (
            inner.square()
            * decoder.model_dts_y[expiry_index]
            / (current_support[:, 2:] - current_support[:, :-2])
        )
        denominator = box * current_density[:, 1:-1]
        if not bool(torch.all(denominator > 0.0)):
            failed = torch.nonzero(denominator <= 0.0).tolist()
            raise RuntimeError(
                "DLV recovery requires positive interior density; failed at "
                f"{failed[:16]}"
            )
        dlvsq_rows.append((current_calls - previous_calls) / denominator)
        previous_support = current_support
        previous_density = current_density
    return torch.stack(dlvsq_rows, dim=1)


def decode_martingale_coordinates(
    decoder: FlexSanosDecoder,
    *,
    spot: torch.Tensor,
    forward_vols: torch.Tensor,
    eta: torch.Tensor,
    cash_strikes: torch.Tensor,
    cash_calls: torch.Tensor,
    reference_densities: torch.Tensor | None = None,
) -> MartingaleChartSurface:
    """Decode spot, forward vols and driftless cash calls into DYSANOS state."""

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("the martingale chart requires capped-unbounded DLV geometry")
    batch = len(spot)
    expiries = geometry.num_model_expiries
    if forward_vols.shape != (batch, expiries) or eta.shape not in {
        (batch,),
        (batch, expiries),
    }:
        raise ValueError("forward vols or eta do not match the decoder geometry")
    if not bool(
        torch.all(
            torch.isfinite(forward_vols)
            & (forward_vols > geometry.min_forward_vol)
            & (forward_vols < geometry.max_forward_vol)
        )
        and torch.all(
            torch.isfinite(eta)
            & (eta > geometry.min_eta)
            & (eta < geometry.max_eta)
        )
    ):
        raise ValueError("forward vols and eta must lie in their open model bounds")

    _, support = decoder._support_strikes_from_forward_vols(forward_vols)
    density0 = decoder._density_from_calls(
        support[:, 0],
        torch.relu(1.0 - support[:, 0]),
    )
    backbone_forward = (
        forward_vols * eta
        if eta.ndim == 2
        else forward_vols * eta.unsqueeze(-1)
    )
    backbone_total_vols = decoder._total_vols(backbone_forward)
    densities, condition = recover_density_from_cash_calls(
        spot=spot,
        support=support,
        backbone_total_vols=backbone_total_vols,
        cash_strikes=cash_strikes,
        cash_calls=cash_calls,
        reference_density=reference_densities,
    )
    dlvsqs = recover_dlvsqs_from_densities(
        decoder,
        support=support,
        density0=density0,
        densities=densities,
    )
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    cash_reprice = _price_cash_call_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=cash_strikes,
    )
    mass_residual = torch.amax(torch.abs(torch.sum(densities, dim=-1) - 1.0))
    mean_residual = torch.amax(
        torch.abs(torch.sum(densities * support, dim=-1) - 1.0)
    )
    recursion_residual = torch.amax(torch.abs(surface.densities - densities))
    return MartingaleChartSurface(
        spot=spot,
        cash_strikes=cash_strikes,
        target_cash_calls=cash_calls,
        surface=surface,
        diagnostics=MartingaleChartDiagnostics(
            minimum_density=torch.amin(densities),
            minimum_dlvsq_margin=torch.amin(dlvsqs - geometry.min_dlvsq),
            maximum_dlvsq_margin=torch.amin(geometry.max_dlvsq - dlvsqs),
            maximum_mass_residual=mass_residual,
            maximum_mean_residual=mean_residual,
            maximum_density_recursion_residual=recursion_residual,
            maximum_absolute_cash_repricing_error=torch.amax(
                torch.abs(cash_reprice - cash_calls)
            ),
            density_system_condition_numbers=condition,
        ),
    )


__all__ = [
    "BoundedBranchDlvCompletion",
    "BranchDensityCompletion",
    "ImplicitLinearSystemDirectionalDerivatives",
    "MartingaleChartDiagnostics",
    "MartingaleChartSurface",
    "black_call_matrix",
    "complete_branch_dlvs_to_martingales",
    "complete_branch_densities_to_martingales",
    "decode_martingale_coordinates",
    "implicit_linear_system_directional_derivatives",
    "recover_density_from_cash_calls",
    "recover_dlvsqs_from_densities",
]
