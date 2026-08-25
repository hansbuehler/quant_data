"""P-free finite-support martingale transport between two option marginals.

The module constructs a joint law ``pi[i, j]`` between supplied one-week and
two-week spot marginals.  It deliberately has no empirical transition law and
no entropy objective: risk neutrality is imposed directly by the marginal and
row-wise martingale equalities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded
from scipy.optimize import linprog
from scipy.special import logsumexp, ndtr


@dataclass(frozen=True)
class ConvexOrderDiagnostics:
    """Finite-support convex-order certificate for two marginals."""

    source_mean: float
    target_mean: float
    mean_difference: float
    minimum_call_slack: float
    minimum_call_slack_strike: float
    tolerance: float
    is_feasible: bool


@dataclass(frozen=True)
class MartingaleCouplingDiagnostics:
    """Solver and exact-constraint diagnostics for a martingale coupling."""

    convex_order: ConvexOrderDiagnostics
    solver_status: int
    solver_message: str
    maximum_source_marginal_residual: float
    maximum_target_marginal_residual: float
    maximum_martingale_residual: float
    minimum_joint_mass: float
    number_positive_cells: int
    minimum_active_conditional_variance: float
    maximum_active_conditional_variance: float


class MartingaleCouplingInfeasibleError(RuntimeError):
    """Raised when the supplied marginals admit no martingale coupling."""

    def __init__(
        self,
        message: str,
        *,
        convex_order: ConvexOrderDiagnostics,
    ) -> None:
        super().__init__(message)
        self.convex_order = convex_order


@dataclass(frozen=True)
class FiniteMartingaleCoupling:
    """A joint finite-support martingale law and its validated diagnostics.

    Rows with zero source probability have zero transition weights because
    their conditional law is immaterial.  All positive-probability rows are
    normalized conditional distributions.
    """

    source_support: np.ndarray
    source_probabilities: np.ndarray
    target_support: np.ndarray
    target_probabilities: np.ndarray
    joint_mass: np.ndarray
    transition_probabilities: np.ndarray
    diagnostics: MartingaleCouplingDiagnostics

    def conditional_call_prices(self, strikes: np.ndarray) -> np.ndarray:
        """Return ``E[(Y-K)+ | X=x_i]`` for every row and strike."""
        strikes = _validate_strikes(strikes)
        payoffs = np.maximum(
            self.target_support[:, None] - strikes[None, :],
            0.0,
        )
        return self.transition_probabilities @ payoffs

    def reconstructed_target_call_prices(self, strikes: np.ndarray) -> np.ndarray:
        """Aggregate conditional calls back to the supplied target marginal."""
        return self.source_probabilities @ self.conditional_call_prices(strikes)

    def target_call_prices(self, strikes: np.ndarray) -> np.ndarray:
        """Price calls directly from the supplied target marginal."""
        return call_prices_from_distribution(
            self.target_support,
            self.target_probabilities,
            strikes,
        )

    def normalized_conditional_distributions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return rows as next-week spot ratios ``Y/X`` and probabilities.

        This is the narrow integration seam to a front-expiry decoder.  For a
        realized one-week spot ``X=x_i``, row ``i`` is a mean-one distribution
        for ``Y/x_i``.  Converting that arbitrary row into the restricted FLEX
        SANOS geometry is a separate approximation problem.
        """
        if np.any(self.source_support <= 0.0):
            raise ValueError("Normalized conditional distributions require positive source support")
        return (
            self.target_support[None, :] / self.source_support[:, None],
            self.transition_probabilities.copy(),
        )


@dataclass(frozen=True)
class SanosStepTransitionDiagnostics:
    """Exact Markov and marginal residuals of one native SANOS step."""

    maximum_row_mass_residual: float
    maximum_row_martingale_residual: float
    maximum_density_propagation_residual: float
    minimum_transition_probability: float
    number_positive_cells: int


@dataclass(frozen=True)
class SanosStepMartingaleTransition:
    """The native Markov transition implicit in one shared-support SANOS step."""

    support: np.ndarray
    previous_density: np.ndarray
    next_density: np.ndarray
    transition_probabilities: np.ndarray
    diagnostics: SanosStepTransitionDiagnostics


@dataclass(frozen=True)
class SanosPosteriorConditioningDiagnostics:
    """Normalization and martingale residuals after conditioning on spot."""

    posterior_mass_residual: float
    component_mass_residual: float
    conditional_spot_mean_residual: float


@dataclass(frozen=True)
class SanosPosteriorConditionalLaw:
    """Next spot law conditional on the observed front-expiry total spot.

    Rows index the posterior latent node of the observed front SANOS mixture;
    columns index its next latent node.  ``component_forward_spots`` are the
    conditional means before applying the common mean-one lognormal backbone
    increment.
    """

    observed_spot: float
    latent_posterior_probabilities: np.ndarray
    component_probabilities: np.ndarray
    component_forward_spots: np.ndarray
    backbone_increment_total_vol: float
    diagnostics: SanosPosteriorConditioningDiagnostics

    def call_prices(self, strikes: np.ndarray) -> np.ndarray:
        """Return next-expiry calls conditional on the observed current spot."""
        strikes = _validate_strikes(strikes)
        if np.any(strikes <= 0.0):
            raise ValueError("Black conditional calls require positive strikes")
        component_calls = _black_calls(
            self.component_forward_spots.reshape(-1),
            strikes,
            self.backbone_increment_total_vol,
        )
        return self.component_probabilities.reshape(-1) @ component_calls


@dataclass(frozen=True)
class RevealedNodeRolledSanosChain:
    """Exact custom-grid rolled surface after revealing a front latent node."""

    source_node_index: int
    normalized_support: np.ndarray
    density0: np.ndarray
    densities: np.ndarray
    dlvsqs: np.ndarray
    dlvsq_time_scale_factors: np.ndarray
    backbone_total_vols: np.ndarray
    maximum_mass_residual: float
    maximum_mean_residual: float


def _sanos_step_matrix(
    w_minus: np.ndarray,
    w_plus: np.ndarray,
) -> np.ndarray:
    """Build the matrix solved by ``FlexSanosDecoder._solve_sanos_step``."""
    w_minus = np.asarray(w_minus, dtype=np.float64)
    w_plus = np.asarray(w_plus, dtype=np.float64)
    if w_minus.ndim != 1 or w_plus.shape != w_minus.shape or len(w_minus) == 0:
        raise ValueError("SANOS step weights must be aligned non-empty vectors")
    if not np.all(np.isfinite(w_minus)) or not np.all(np.isfinite(w_plus)):
        raise ValueError("SANOS step weights must be finite")
    if np.min(w_minus) < 0.0 or np.min(w_plus) < 0.0:
        raise ValueError("SANOS step weights must be non-negative")
    n = len(w_minus) + 2
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[0, 1] = -w_minus[0]
    matrix[-1, -2] = -w_plus[-1]
    matrix[-1, -1] = 1.0
    for index in range(1, n - 1):
        if index > 1:
            matrix[index, index - 1] = -w_plus[index - 2]
        matrix[index, index] = 1.0 + w_minus[index - 1] + w_plus[index - 1]
        if index < n - 2:
            matrix[index, index + 1] = -w_minus[index]
    return matrix


def solve_shared_sanos_step_transition(
    support: np.ndarray,
    previous_density: np.ndarray,
    next_density: np.ndarray,
    *,
    w_minus: np.ndarray,
    w_plus: np.ndarray,
    tolerance: float = 1.0e-9,
) -> SanosStepMartingaleTransition:
    """Expose and validate the native P-free transition of a SANOS step.

    If ``A q_next = q_previous`` is the shared-support SANOS solve, then
    ``P=A^{-T}`` is a row-stochastic martingale transition and
    ``q_previous @ P = q_next``.  The result depends only on the current SANOS
    surface parameters; it is unrelated to the empirical P dynamics.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"'tolerance' must be finite and positive, found {tolerance}")
    support = np.asarray(support, dtype=np.float64)
    previous_density = np.asarray(previous_density, dtype=np.float64)
    next_density = np.asarray(next_density, dtype=np.float64)
    if support.ndim != 1 or len(support) < 3:
        raise ValueError("'support' must contain at least three points")
    if previous_density.shape != support.shape or next_density.shape != support.shape:
        raise ValueError("SANOS support and densities must be aligned vectors")
    if not np.all(np.isfinite(support)) or not np.all(np.diff(support) > 0.0):
        raise ValueError("SANOS support must be finite and strictly increasing")
    for name, density in (
        ("previous_density", previous_density),
        ("next_density", next_density),
    ):
        if not np.all(np.isfinite(density)) or np.min(density) < -tolerance:
            raise ValueError(f"'{name}' must be finite and non-negative")
        if abs(float(np.sum(density)) - 1.0) > tolerance:
            raise ValueError(f"'{name}' must have unit mass")

    matrix = _sanos_step_matrix(w_minus, w_plus)
    if matrix.shape != (len(support), len(support)):
        raise ValueError(
            "SANOS weight count does not match support: "
            f"matrix {matrix.shape}, support {support.shape}"
        )
    # q_next=A^-1 q_previous, while row-law propagation is q_prev@P.  A is
    # tridiagonal, so solve all columns of A^T P=I in O(N^2), rather than using
    # a dense O(N^3) factorization for every date and expiry.
    transpose = matrix.T
    bands = np.zeros((3, len(support)), dtype=np.float64)
    bands[0, 1:] = np.diag(transpose, k=1)
    bands[1] = np.diag(transpose)
    bands[2, :-1] = np.diag(transpose, k=-1)
    transition = solve_banded(
        (1, 1),
        bands,
        np.eye(len(support)),
        overwrite_ab=True,
        overwrite_b=True,
        check_finite=False,
    )
    minimum_probability = float(np.min(transition))
    if minimum_probability < -tolerance:
        raise RuntimeError(
            f"SANOS inverse is not a non-negative transition: minimum={minimum_probability:.6g}"
        )
    transition = np.maximum(transition, 0.0)
    row_residual = transition.sum(axis=1) - 1.0
    martingale_residual = transition @ support - support
    propagation_residual = previous_density @ transition - next_density
    maximum_residual = max(
        float(np.max(np.abs(row_residual))),
        float(np.max(np.abs(martingale_residual))),
        float(np.max(np.abs(propagation_residual))),
    )
    if maximum_residual > 10.0 * tolerance:
        raise RuntimeError(
            "SANOS step is not a validated martingale transition: "
            f"maximum residual={maximum_residual:.6g}"
        )
    return SanosStepMartingaleTransition(
        support=support.copy(),
        previous_density=previous_density.copy(),
        next_density=next_density.copy(),
        transition_probabilities=transition,
        diagnostics=SanosStepTransitionDiagnostics(
            maximum_row_mass_residual=float(np.max(np.abs(row_residual))),
            maximum_row_martingale_residual=float(np.max(np.abs(martingale_residual))),
            maximum_density_propagation_residual=float(np.max(np.abs(propagation_residual))),
            minimum_transition_probability=float(np.min(transition)),
            number_positive_cells=int(np.sum(transition > tolerance)),
        ),
    )


def solve_shared_sanos_step_transition_from_dlvsqs(
    support: np.ndarray,
    previous_density: np.ndarray,
    next_density: np.ndarray,
    *,
    dlvsqs: np.ndarray,
    time_increment_y: float,
    tolerance: float = 1.0e-9,
) -> SanosStepMartingaleTransition:
    """Expose a shared-strike FLEX step directly from its decoded DLV values.

    A decoded ``SanosSurface`` supplies exactly these arrays as
    ``support_strikes[...,j,:]``, ``densities[...,j-1,:]``,
    ``densities[...,j,:]``, and ``dlvsqs[...,j,:]``.  This function is kept
    NumPy-only so the risk-neutral construction has no dependency on a P model
    or a Torch training graph.
    """
    support = np.asarray(support, dtype=np.float64)
    dlvsqs = np.asarray(dlvsqs, dtype=np.float64)
    if support.ndim != 1 or len(support) < 3:
        raise ValueError("'support' must contain at least three points")
    if dlvsqs.shape != (len(support) - 2,):
        raise ValueError(
            "'dlvsqs' must contain one value per inner support node, "
            f"found {dlvsqs.shape} for support {support.shape}"
        )
    if not np.all(np.isfinite(dlvsqs)) or np.min(dlvsqs) < 0.0:
        raise ValueError("'dlvsqs' must be finite and non-negative")
    if not np.isfinite(time_increment_y) or time_increment_y <= 0.0:
        raise ValueError("'time_increment_y' must be finite and positive")
    base_weights = dlvsqs * support[1:-1] ** 2 * time_increment_y / (support[2:] - support[:-2])
    return solve_shared_sanos_step_transition(
        support,
        previous_density,
        next_density,
        w_minus=base_weights / (support[1:-1] - support[:-2]),
        w_plus=base_weights / (support[2:] - support[1:-1]),
        tolerance=tolerance,
    )


def solve_shared_sanos_density_step_from_dlvsqs(
    support: np.ndarray,
    previous_density: np.ndarray,
    *,
    dlvsqs: np.ndarray,
    time_increment_y: float,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Propagate one density through a shared-support SANOS DLV step.

    This is the NumPy/banded counterpart of the native Torch tridiagonal solve.
    It is useful for deterministic parameter-projection controls where the
    resulting next density is not known before the solve.
    """
    support = np.asarray(support, dtype=np.float64)
    previous_density = np.asarray(previous_density, dtype=np.float64)
    dlvsqs = np.asarray(dlvsqs, dtype=np.float64)
    if support.ndim != 1 or len(support) < 3:
        raise ValueError("'support' must contain at least three points")
    if previous_density.shape != support.shape:
        raise ValueError("Previous density and support must be aligned vectors")
    if dlvsqs.shape != (len(support) - 2,):
        raise ValueError("'dlvsqs' must contain one value per inner support node")
    if (
        not np.all(np.isfinite(support))
        or not np.all(np.diff(support) > 0.0)
        or not np.all(np.isfinite(previous_density))
        or np.min(previous_density) < -tolerance
        or abs(float(np.sum(previous_density)) - 1.0) > tolerance
        or not np.all(np.isfinite(dlvsqs))
        or np.min(dlvsqs) < 0.0
        or not np.isfinite(time_increment_y)
        or time_increment_y <= 0.0
    ):
        raise ValueError("Invalid SANOS density-step inputs")
    base_weights = dlvsqs * support[1:-1] ** 2 * time_increment_y / (support[2:] - support[:-2])
    matrix = _sanos_step_matrix(
        base_weights / (support[1:-1] - support[:-2]),
        base_weights / (support[2:] - support[1:-1]),
    )
    bands = np.zeros((3, len(support)), dtype=np.float64)
    bands[0, 1:] = np.diag(matrix, k=1)
    bands[1] = np.diag(matrix)
    bands[2, :-1] = np.diag(matrix, k=-1)
    density = solve_banded(
        (1, 1),
        bands,
        previous_density.copy(),
        overwrite_ab=True,
        overwrite_b=True,
        check_finite=False,
    )
    if np.min(density) < -tolerance:
        raise RuntimeError("SANOS density step produced negative mass")
    mass_residual = abs(float(np.sum(density)) - 1.0)
    mean_residual = abs(float(density @ support) - float(previous_density @ support))
    if max(mass_residual, mean_residual) > 10.0 * tolerance:
        raise RuntimeError("SANOS density step failed its mass or mean identity")
    return np.maximum(density, 0.0)


def build_revealed_node_rolled_sanos_chain(
    support: np.ndarray,
    transitions: tuple[SanosStepMartingaleTransition, ...],
    *,
    source_node_index: int,
    old_dlvsqs: np.ndarray,
    old_time_increments_y: np.ndarray,
    old_backbone_total_vols: np.ndarray,
    tolerance: float = 1.0e-9,
) -> RevealedNodeRolledSanosChain:
    """Build the exact revealed-node rolled surface on its custom grid.

    If the front discrete SANOS node is revealed as ``I=i``, the rolled grid is
    ``support/support[i]`` and time-zero density is the atom at ``i``.  Its
    first density is row ``i`` of the W1-to-W2 kernel; later densities propagate
    that row through the subsequent native kernels.

    FLEX places a new front expiry at 4.8 business days although the old
    W1-to-W2 transition spans five days.  Since the SANOS operator contains
    ``DLVsq * dt``, the first rolled DLVsq is multiplied by ``5/4.8``.  The
    returned factors are computed from the supplied time grid rather than
    hard-coded; later factors equal one on the equidistant weekly grid.
    """
    support = np.asarray(support, dtype=np.float64)
    old_dlvsqs = np.asarray(old_dlvsqs, dtype=np.float64)
    old_time_increments_y = np.asarray(old_time_increments_y, dtype=np.float64)
    old_backbone_total_vols = np.asarray(old_backbone_total_vols, dtype=np.float64)
    steps = len(transitions)
    if steps == 0:
        raise ValueError("At least one SANOS transition is required")
    if source_node_index < 0 or source_node_index >= len(support):
        raise ValueError("'source_node_index' lies outside the support")
    if np.any(support <= 0.0) or not np.all(np.diff(support) > 0.0):
        raise ValueError("Revealed-node support must be positive and increasing")
    if (
        old_dlvsqs.ndim != 2
        or old_dlvsqs.shape[0] < steps + 1
        or old_dlvsqs.shape[1] != len(support) - 2
        or old_time_increments_y.ndim != 1
        or len(old_time_increments_y) < steps + 1
        or old_backbone_total_vols.ndim != 1
        or len(old_backbone_total_vols) < steps + 1
    ):
        raise ValueError("Old SANOS arrays do not cover every rolled transition")
    if not np.all(np.isfinite(old_time_increments_y[: steps + 1])) or np.any(
        old_time_increments_y[: steps + 1] <= 0.0
    ):
        raise ValueError("Old SANOS time increments must be finite and positive")
    for transition in transitions:
        if not np.array_equal(transition.support, support):
            raise ValueError("Every rolled transition must use the shared support")

    normalized_support = support / support[source_node_index]
    density0 = np.zeros_like(support)
    density0[source_node_index] = 1.0
    density = density0.copy()
    rolled_densities: list[np.ndarray] = []
    for transition in transitions:
        density = density @ transition.transition_probabilities
        rolled_densities.append(density.copy())
    densities = np.stack(rolled_densities)

    rolled_time_increments = np.concatenate(
        (
            old_time_increments_y[:1],
            old_time_increments_y[2 : steps + 1],
        )
    )
    assert rolled_time_increments.shape == (steps,)
    scale_factors = old_time_increments_y[1 : steps + 1] / rolled_time_increments
    rolled_dlvsqs = old_dlvsqs[1 : steps + 1] * scale_factors[:, None]
    backbone_variance = (
        old_backbone_total_vols[1 : steps + 1] ** 2 - old_backbone_total_vols[0] ** 2
    )
    if np.min(backbone_variance) < -tolerance:
        raise ValueError("Rolled backbone variance is negative")
    rolled_backbone = np.sqrt(np.maximum(backbone_variance, 0.0))
    mass_residual = np.sum(densities, axis=1) - 1.0
    mean_residual = densities @ normalized_support - 1.0
    maximum_mass_residual = float(np.max(np.abs(mass_residual)))
    maximum_mean_residual = float(np.max(np.abs(mean_residual)))
    if max(maximum_mass_residual, maximum_mean_residual) > 10.0 * tolerance:
        raise RuntimeError("Revealed-node rolled SANOS chain is not a martingale law")
    return RevealedNodeRolledSanosChain(
        source_node_index=source_node_index,
        normalized_support=normalized_support,
        density0=density0,
        densities=densities,
        dlvsqs=rolled_dlvsqs,
        dlvsq_time_scale_factors=scale_factors,
        backbone_total_vols=rolled_backbone,
        maximum_mass_residual=maximum_mass_residual,
        maximum_mean_residual=maximum_mean_residual,
    )


def condition_sanos_step_on_spot(
    step: SanosStepMartingaleTransition,
    observed_spot: float,
    *,
    previous_backbone_total_vol: float,
    next_backbone_total_vol: float,
) -> SanosPosteriorConditionalLaw:
    """Condition the native SANOS transition on total observed front spot.

    Write the observed front spot as ``S_1=Z_1 L_1``, where ``Z_1`` has the
    discrete SANOS density and ``L_1`` is its independent mean-one lognormal
    backbone.  Bayes' rule gives the posterior over ``Z_1``.  Conditional on
    nodes ``i,j``, the next spot is

    ``S_2 = S_1 * support[j]/support[i] * G``,

    where ``G`` is the independent mean-one lognormal backbone increment.
    This integrates out the latent front node: the returned law conditions
    only on the observable spot and the supplied current SANOS surface.
    """
    if not np.isfinite(observed_spot) or observed_spot <= 0.0:
        raise ValueError("'observed_spot' must be finite and positive")
    if np.any(step.support <= 0.0):
        raise ValueError("Posterior SANOS conditioning requires positive support")
    if (
        not np.isfinite(previous_backbone_total_vol)
        or not np.isfinite(next_backbone_total_vol)
        or previous_backbone_total_vol <= 0.0
        or next_backbone_total_vol < previous_backbone_total_vol
    ):
        raise ValueError("Posterior conditioning requires positive, increasing backbone total vols")
    log_prior = np.full_like(step.previous_density, -np.inf)
    active = step.previous_density > 0.0
    log_prior[active] = np.log(step.previous_density[active])
    gaussian_coordinate = (
        np.log(observed_spot / step.support) + 0.5 * previous_backbone_total_vol**2
    ) / previous_backbone_total_vol
    # The lognormal density's 1/(S_1 v sqrt(2*pi)) factor is common to all
    # latent nodes and cancels from Bayes' rule.
    log_weight = log_prior - 0.5 * gaussian_coordinate**2
    normalizer = float(logsumexp(log_weight))
    if not np.isfinite(normalizer):
        raise RuntimeError("Observed spot has zero numerical posterior mass")
    posterior = np.exp(log_weight - normalizer)
    component_probability = posterior[:, None] * step.transition_probabilities
    component_forward = observed_spot * step.support[None, :] / step.support[:, None]
    conditional_mean = float(np.sum(component_probability * component_forward))
    increment = float(
        np.sqrt(
            max(
                0.0,
                next_backbone_total_vol**2 - previous_backbone_total_vol**2,
            )
        )
    )
    return SanosPosteriorConditionalLaw(
        observed_spot=float(observed_spot),
        latent_posterior_probabilities=posterior,
        component_probabilities=component_probability,
        component_forward_spots=component_forward,
        backbone_increment_total_vol=increment,
        diagnostics=SanosPosteriorConditioningDiagnostics(
            posterior_mass_residual=float(np.sum(posterior) - 1.0),
            component_mass_residual=float(np.sum(component_probability) - 1.0),
            conditional_spot_mean_residual=conditional_mean - observed_spot,
        ),
    )


def _black_calls(
    spots: np.ndarray,
    strikes: np.ndarray,
    sqrt_variance: float,
) -> np.ndarray:
    spots = np.asarray(spots, dtype=np.float64).reshape(-1, 1)
    strikes = np.asarray(strikes, dtype=np.float64).reshape(1, -1)
    if sqrt_variance == 0.0:
        return np.maximum(spots - strikes, 0.0)
    d1 = (np.log(spots / strikes) + 0.5 * sqrt_variance**2) / sqrt_variance
    d2 = d1 - sqrt_variance
    return spots * ndtr(d1) - strikes * ndtr(d2)


def reconstruct_sanos_step_calls(
    step: SanosStepMartingaleTransition,
    strikes: np.ndarray,
    *,
    previous_backbone_total_vol: float,
    next_backbone_total_vol: float,
    quadrature_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct next-expiry calls through the full native transition.

    The previous lognormal backbone is retained and an independent mean-one
    lognormal increment is applied.  The first result integrates the resulting
    conditional transition; the second prices the next SANOS mixture directly.
    Equality validates the complete discrete-plus-backbone construction.
    """
    strikes = _validate_strikes(strikes)
    if np.any(strikes <= 0.0):
        raise ValueError("Black-call reconstruction requires positive strikes")
    if (
        not np.isfinite(previous_backbone_total_vol)
        or not np.isfinite(next_backbone_total_vol)
        or previous_backbone_total_vol < 0.0
        or next_backbone_total_vol < previous_backbone_total_vol
    ):
        raise ValueError("Backbone total volatilities must be finite and increasing")
    if quadrature_points < 8:
        raise ValueError("'quadrature_points' must be at least eight")
    increment = float(
        np.sqrt(
            max(
                0.0,
                next_backbone_total_vol**2 - previous_backbone_total_vol**2,
            )
        )
    )
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    previous_backbone = np.exp(
        -0.5 * previous_backbone_total_vol**2 + np.sqrt(2.0) * previous_backbone_total_vol * nodes
    )
    weights = weights / np.sqrt(np.pi)
    reconstructed = np.zeros_like(strikes)
    for multiplier, weight in zip(previous_backbone, weights, strict=True):
        next_node_calls = _black_calls(
            step.support * multiplier,
            strikes,
            increment,
        )
        conditional_node_calls = step.transition_probabilities @ next_node_calls
        reconstructed += weight * (step.previous_density @ conditional_node_calls)
    direct = step.next_density @ _black_calls(
        step.support,
        strikes,
        next_backbone_total_vol,
    )
    return reconstructed, direct


def _validate_distribution(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if support.ndim != 1 or len(support) == 0:
        raise ValueError(f"'{name}_support' must be a non-empty vector")
    if probabilities.shape != support.shape:
        raise ValueError(
            f"'{name}_probabilities' must have shape {support.shape}, found {probabilities.shape}"
        )
    if not np.all(np.isfinite(support)):
        raise ValueError(f"'{name}_support' must contain only finite values")
    if len(support) > 1 and not np.all(np.diff(support) > 0.0):
        raise ValueError(f"'{name}_support' must be strictly increasing")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"'{name}_probabilities' must contain only finite values")
    if np.min(probabilities) < 0.0:
        raise ValueError(f"'{name}_probabilities' must be non-negative")
    mass = float(np.sum(probabilities))
    if abs(mass - 1.0) > tolerance:
        raise ValueError(
            f"'{name}_probabilities' must sum to one within {tolerance:g}, found {mass:.17g}"
        )
    # Remove only the permitted floating-point mass error.  Subsequent LP
    # equalities and diagnostics are then expressed against exact unit masses.
    return support.copy(), probabilities / mass


def _validate_strikes(strikes: np.ndarray) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=np.float64)
    if strikes.ndim != 1:
        raise ValueError("'strikes' must be a vector")
    if not np.all(np.isfinite(strikes)):
        raise ValueError("'strikes' must contain only finite values")
    return strikes


def call_prices_from_distribution(
    support: np.ndarray,
    probabilities: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Return undiscounted call prices from a finite spot distribution."""
    support = np.asarray(support, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    strikes = _validate_strikes(strikes)
    if support.ndim != 1 or probabilities.shape != support.shape:
        raise ValueError("'support' and 'probabilities' must be aligned vectors")
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Distribution inputs must contain only finite values")
    return probabilities @ np.maximum(support[:, None] - strikes[None, :], 0.0)


def convex_order_diagnostics(
    source_support: np.ndarray,
    source_probabilities: np.ndarray,
    target_support: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> ConvexOrderDiagnostics:
    """Check the necessary and sufficient finite-support convex-order test."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"'tolerance' must be finite and positive, found {tolerance}")
    x, p = _validate_distribution(
        source_support,
        source_probabilities,
        name="source",
        tolerance=tolerance,
    )
    y, q = _validate_distribution(
        target_support,
        target_probabilities,
        name="target",
        tolerance=tolerance,
    )
    strikes = np.unique(np.concatenate((x, y)))
    source_calls = call_prices_from_distribution(x, p, strikes)
    target_calls = call_prices_from_distribution(y, q, strikes)
    call_slack = target_calls - source_calls
    minimum_index = int(np.argmin(call_slack))
    source_mean = float(p @ x)
    target_mean = float(q @ y)
    scale = max(1.0, float(np.max(np.abs(np.concatenate((x, y))))))
    scaled_tolerance = tolerance * scale
    mean_difference = target_mean - source_mean
    minimum_call_slack = float(call_slack[minimum_index])
    return ConvexOrderDiagnostics(
        source_mean=source_mean,
        target_mean=target_mean,
        mean_difference=mean_difference,
        minimum_call_slack=minimum_call_slack,
        minimum_call_slack_strike=float(strikes[minimum_index]),
        tolerance=scaled_tolerance,
        is_feasible=(
            abs(mean_difference) <= scaled_tolerance and minimum_call_slack >= -scaled_tolerance
        ),
    )


def solve_finite_martingale_coupling(
    source_support: np.ndarray,
    source_probabilities: np.ndarray,
    target_support: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    cost_matrix: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
) -> FiniteMartingaleCoupling:
    """Solve a P-free finite-support martingale coupling LP.

    The joint masses satisfy

    ``sum_j pi[i,j]=p[i]``, ``sum_i pi[i,j]=q[j]``, and
    ``sum_j pi[i,j]*y[j]=p[i]*x[i]``.

    With no ``cost_matrix``, HiGHS returns any feasible law.  An optional
    finite linear transport cost can select among incomplete-market couplings;
    it is supplied directly and is not an entropy or empirical-law distance.
    """
    certificate = convex_order_diagnostics(
        source_support,
        source_probabilities,
        target_support,
        target_probabilities,
        tolerance=tolerance,
    )
    if not certificate.is_feasible:
        raise MartingaleCouplingInfeasibleError(
            "The supplied marginals fail equal-mean convex order: "
            f"mean difference={certificate.mean_difference:.6g}, "
            f"minimum call slack={certificate.minimum_call_slack:.6g} at "
            f"K={certificate.minimum_call_slack_strike:.6g}",
            convex_order=certificate,
        )
    x, p = _validate_distribution(
        source_support,
        source_probabilities,
        name="source",
        tolerance=tolerance,
    )
    y, q = _validate_distribution(
        target_support,
        target_probabilities,
        name="target",
        tolerance=tolerance,
    )
    n, m = len(x), len(y)
    cell = np.arange(n * m, dtype=np.int64).reshape(n, m)

    row_index: list[np.ndarray] = []
    column_index: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rhs_parts: list[np.ndarray] = []
    constraint_offset = 0

    # Source marginal rows.
    row_index.append(np.repeat(np.arange(n) + constraint_offset, m))
    column_index.append(cell.reshape(-1))
    values.append(np.ones(n * m, dtype=np.float64))
    rhs_parts.append(p)
    constraint_offset += n

    # Target marginal columns.
    row_index.append(np.tile(np.arange(m) + constraint_offset, n))
    column_index.append(cell.reshape(-1))
    values.append(np.ones(n * m, dtype=np.float64))
    rhs_parts.append(q)
    constraint_offset += m

    # Row-wise martingale constraints use the target support Y.
    row_index.append(np.repeat(np.arange(n) + constraint_offset, m))
    column_index.append(cell.reshape(-1))
    values.append(np.tile(y, n))
    rhs_parts.append(p * x)
    constraint_offset += n

    equality = sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(row_index), np.concatenate(column_index)),
        ),
        shape=(constraint_offset, n * m),
    ).tocsr()
    if cost_matrix is None:
        costs = np.zeros((n, m), dtype=np.float64)
    else:
        costs = np.asarray(cost_matrix, dtype=np.float64)
        if costs.shape != (n, m) or not np.all(np.isfinite(costs)):
            raise ValueError(
                f"'cost_matrix' must be a finite matrix of shape {(n, m)}, found {costs.shape}"
            )
    solution = linprog(
        costs.reshape(-1),
        A_eq=equality,
        b_eq=np.concatenate(rhs_parts),
        bounds=(0.0, None),
        method="highs",
        options={"primal_feasibility_tolerance": max(tolerance, 1.0e-10)},
    )
    if not solution.success or solution.x is None:
        raise MartingaleCouplingInfeasibleError(
            "HiGHS did not produce a martingale coupling despite the convex-order "
            f"certificate: status={solution.status}, message={solution.message}",
            convex_order=certificate,
        )

    joint = solution.x.reshape(n, m)
    # HiGHS can return signed zero/tolerance-scale negative entries.  Values
    # beyond the stated validation tolerance are a solver failure, not masses
    # to be silently repaired.
    scaled_tolerance = certificate.tolerance
    if float(np.min(joint)) < -scaled_tolerance:
        raise RuntimeError(
            f"Martingale LP returned materially negative joint mass: {np.min(joint):.6g}"
        )
    joint = np.maximum(joint, 0.0)
    transition = np.zeros_like(joint)
    active_rows = p > scaled_tolerance
    transition[active_rows] = joint[active_rows] / p[active_rows, None]

    source_residual = np.sum(joint, axis=1) - p
    target_residual = np.sum(joint, axis=0) - q
    martingale_residual = joint @ y - p * x
    conditional_means = transition[active_rows] @ y
    conditional_variances = np.sum(
        transition[active_rows] * (y[None, :] - conditional_means[:, None]) ** 2,
        axis=1,
    )
    maximum_residual = max(
        float(np.max(np.abs(source_residual))),
        float(np.max(np.abs(target_residual))),
        float(np.max(np.abs(martingale_residual))),
    )
    if maximum_residual > 10.0 * scaled_tolerance:
        raise RuntimeError(
            "Martingale LP failed post-solve residual validation: "
            f"maximum residual={maximum_residual:.6g}, "
            f"tolerance={10.0 * scaled_tolerance:.6g}"
        )
    diagnostics = MartingaleCouplingDiagnostics(
        convex_order=certificate,
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        maximum_source_marginal_residual=float(np.max(np.abs(source_residual))),
        maximum_target_marginal_residual=float(np.max(np.abs(target_residual))),
        maximum_martingale_residual=float(np.max(np.abs(martingale_residual))),
        minimum_joint_mass=float(np.min(joint)),
        number_positive_cells=int(np.sum(joint > scaled_tolerance)),
        minimum_active_conditional_variance=float(np.min(conditional_variances)),
        maximum_active_conditional_variance=float(np.max(conditional_variances)),
    )
    return FiniteMartingaleCoupling(
        source_support=x,
        source_probabilities=p,
        target_support=y,
        target_probabilities=q,
        joint_mass=joint,
        transition_probabilities=transition,
        diagnostics=diagnostics,
    )
