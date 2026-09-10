"""Plain weekly AR(1) dynamics for the floating four-week DLV boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import torch

from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs


@dataclass(frozen=True)
class LastExpiryAr1Diagnostics:
    """Fit and temporal-validation evidence for the isolated boundary law."""

    training_end: str
    rank: int
    num_training_levels: int
    num_training_pairs: int
    num_validation_pairs: int
    explained_variance_fraction: float
    weekly_spectral_radius: float
    training_forecast_rmse: float
    training_persistence_rmse: float
    validation_forecast_rmse: float
    validation_persistence_rmse: float
    validation_unconditional_rmse: float
    validation_improvement_over_persistence_fraction: float
    minimum_innovation_covariance_eigenvalue: float
    maximum_innovation_covariance_eigenvalue: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class LastExpiryAr1Fit:
    r"""PCA curve representation and row-vector VAR(1) coefficients.

    If ``z_t`` is one row of scores, the fitted transition is

    ``z_(t+1) = intercept + z_t @ transition + innovation``.
    """

    canonical_inner_strikes: np.ndarray
    raw_curve_mean: np.ndarray
    training_raw_curve_minimum: np.ndarray
    training_raw_curve_maximum: np.ndarray
    components: np.ndarray
    intercept: np.ndarray
    transition: np.ndarray
    innovation_cholesky: np.ndarray
    diagnostics: LastExpiryAr1Diagnostics


@dataclass(frozen=True)
class TorchLastExpiryAr1:
    """Torch tensors needed to simulate the fitted floating boundary."""

    canonical_inner_strikes: torch.Tensor
    canonical_log_strikes: torch.Tensor
    raw_curve_mean: torch.Tensor
    training_raw_curve_minimum: torch.Tensor
    training_raw_curve_maximum: torch.Tensor
    components: torch.Tensor
    intercept: torch.Tensor
    transition: torch.Tensor
    innovation_cholesky: torch.Tensor

    @property
    def rank(self) -> int:
        """Number of retained curve factors."""

        return self.components.shape[0]

    def project(self, raw_curves: torch.Tensor) -> torch.Tensor:
        """Project canonical-grid raw log-DLV curves into AR scores."""

        if raw_curves.shape[-1] != self.components.shape[-1]:
            raise ValueError("raw curves have the wrong strike dimension")
        return (raw_curves - self.raw_curve_mean) @ self.components.transpose(0, 1)

    def step(
        self,
        scores: torch.Tensor,
        standard_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance scores once and reconstruct a canonical-grid raw curve."""

        if scores.shape != standard_normals.shape or scores.shape[-1] != self.rank:
            raise ValueError("AR scores and innovations must have shape (batch, rank)")
        next_scores = (
            self.intercept
            + scores @ self.transition
            + standard_normals @ self.innovation_cholesky.transpose(0, 1)
        )
        raw_curves = self.raw_curve_mean + next_scores @ self.components
        return next_scores, raw_curves

    def clamp_to_training_envelope(
        self,
        raw_curves: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep Gaussian forecasts inside the observed training curve envelope."""

        clipped = (raw_curves < self.training_raw_curve_minimum) | (
            raw_curves > self.training_raw_curve_maximum
        )
        return torch.maximum(
            torch.minimum(raw_curves, self.training_raw_curve_maximum),
            self.training_raw_curve_minimum,
        ), clipped


def _root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def fit_last_expiry_ar1(
    raw_log_dlvsq_curves: np.ndarray,
    dates: tuple[date, ...],
    canonical_inner_strikes: np.ndarray,
    *,
    rank: int = 25,
    training_end: date = date(2024, 12, 31),
) -> LastExpiryAr1Fit:
    """Fit a training-only PCA plus unrestricted weekly VAR(1)."""

    curves = np.asarray(raw_log_dlvsq_curves, dtype=np.float64)
    strikes = np.asarray(canonical_inner_strikes, dtype=np.float64)
    if (
        curves.ndim != 2
        or curves.shape != (len(dates), len(strikes))
        or not np.all(np.isfinite(curves))
        or strikes.ndim != 1
        or not np.all(np.isfinite(strikes))
        or not np.all(np.diff(strikes) > 0.0)
    ):
        raise ValueError("last-expiry AR inputs are malformed")
    if not 1 <= rank < min(curves.shape):
        raise ValueError("last-expiry AR rank is outside the admissible range")

    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    training_pairs = np.asarray(
        [dates[int(index)] <= training_end for index in endpoint],
        dtype=bool,
    )
    validation_pairs = np.asarray(
        [dates[int(index)] > training_end for index in origin],
        dtype=bool,
    )
    if np.sum(training_pairs) <= rank or not np.any(validation_pairs):
        raise ValueError("last-expiry AR fit needs training and validation pairs")
    training_levels = np.flatnonzero(
        np.asarray([value <= training_end for value in dates], dtype=bool)
    )
    mean = np.mean(curves[training_levels], axis=0)
    training_minimum = np.min(curves[training_levels], axis=0)
    training_maximum = np.max(curves[training_levels], axis=0)
    centered_training = curves[training_levels] - mean
    _, singular_values, right = np.linalg.svd(
        centered_training,
        full_matrices=False,
    )
    components = right[:rank]
    explained = float(
        np.sum(np.square(singular_values[:rank]))
        / np.sum(np.square(singular_values))
    )
    scores = (curves - mean) @ components.T
    train_origin = scores[origin[training_pairs]]
    train_endpoint = scores[endpoint[training_pairs]]
    design = np.concatenate(
        (np.ones((len(train_origin), 1), dtype=np.float64), train_origin),
        axis=1,
    )
    coefficients = np.linalg.lstsq(design, train_endpoint, rcond=None)[0]
    intercept = coefficients[0]
    transition = coefficients[1:]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0:
        raise ValueError(
            "last-expiry AR transition is not stationary: "
            f"spectral radius={spectral_radius:.6g}"
        )

    fitted_train_scores = intercept + train_origin @ transition
    residuals = train_endpoint - fitted_train_scores
    covariance = residuals.T @ residuals / len(residuals)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = (
        np.finfo(np.float64).eps
        * max(float(np.max(eigenvalues)), 1.0)
        * covariance.shape[0]
    )
    regularized = (eigenvectors * np.maximum(eigenvalues, floor)[None]) @ (
        eigenvectors.T
    )
    innovation_cholesky = np.linalg.cholesky(regularized)

    def forecast(indices: np.ndarray) -> tuple[float, float, float]:
        source = origin[indices]
        target = endpoint[indices]
        predicted_scores = intercept + scores[source] @ transition
        predicted_curves = np.clip(
            mean + predicted_scores @ components,
            training_minimum,
            training_maximum,
        )
        return (
            _root_mean_square(predicted_curves - curves[target]),
            _root_mean_square(curves[source] - curves[target]),
            _root_mean_square(mean - curves[target]),
        )

    train_rmse, train_persistence, _ = forecast(training_pairs)
    validation_rmse, validation_persistence, validation_unconditional = forecast(
        validation_pairs
    )
    diagnostics = LastExpiryAr1Diagnostics(
        training_end=training_end.isoformat(),
        rank=rank,
        num_training_levels=len(training_levels),
        num_training_pairs=int(np.sum(training_pairs)),
        num_validation_pairs=int(np.sum(validation_pairs)),
        explained_variance_fraction=explained,
        weekly_spectral_radius=spectral_radius,
        training_forecast_rmse=train_rmse,
        training_persistence_rmse=train_persistence,
        validation_forecast_rmse=validation_rmse,
        validation_persistence_rmse=validation_persistence,
        validation_unconditional_rmse=validation_unconditional,
        validation_improvement_over_persistence_fraction=(
            1.0 - validation_rmse / validation_persistence
        ),
        minimum_innovation_covariance_eigenvalue=float(np.min(eigenvalues)),
        maximum_innovation_covariance_eigenvalue=float(np.max(eigenvalues)),
    )
    return LastExpiryAr1Fit(
        canonical_inner_strikes=strikes.copy(),
        raw_curve_mean=mean,
        training_raw_curve_minimum=training_minimum,
        training_raw_curve_maximum=training_maximum,
        components=components,
        intercept=intercept,
        transition=transition,
        innovation_cholesky=innovation_cholesky,
        diagnostics=diagnostics,
    )


def torch_last_expiry_ar1(
    fit: LastExpiryAr1Fit,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> TorchLastExpiryAr1:
    """Move a fitted boundary law to one Torch dtype and device."""

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    strikes = tensor(fit.canonical_inner_strikes)
    return TorchLastExpiryAr1(
        canonical_inner_strikes=strikes,
        canonical_log_strikes=torch.log(strikes),
        raw_curve_mean=tensor(fit.raw_curve_mean),
        training_raw_curve_minimum=tensor(fit.training_raw_curve_minimum),
        training_raw_curve_maximum=tensor(fit.training_raw_curve_maximum),
        components=tensor(fit.components),
        intercept=tensor(fit.intercept),
        transition=tensor(fit.transition),
        innovation_cholesky=tensor(fit.innovation_cholesky),
    )


def make_antithetic_ar1_normals(
    *,
    num_states: int,
    pairs_per_state: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return adjacent ``Z,-Z`` AR innovations in state-major order."""

    if min(num_states, pairs_per_state, rank) <= 0:
        raise ValueError("AR innovation dimensions must be positive")
    normal = torch.randn(
        (num_states, pairs_per_state, rank),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return torch.stack((normal, -normal), dim=2).reshape(
        2 * num_states * pairs_per_state,
        rank,
    )


def interpolate_ar1_raw_curves(
    model: TorchLastExpiryAr1,
    raw_curves: torch.Tensor,
    target_inner_strikes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate canonical raw curves onto path-specific relative strikes."""

    if (
        raw_curves.ndim != 2
        or target_inner_strikes.shape != raw_curves.shape
        or raw_curves.shape[-1] != len(model.canonical_inner_strikes)
    ):
        raise ValueError("AR interpolation inputs have incompatible shapes")
    query = torch.log(target_inner_strikes)
    grid = model.canonical_log_strikes
    extrapolated = (query < grid[0]) | (query > grid[-1])
    upper = torch.searchsorted(grid, query).clamp(1, len(grid) - 1)
    lower = upper - 1
    weight = ((query - grid[lower]) / (grid[upper] - grid[lower])).clamp(0.0, 1.0)
    lower_values = torch.gather(raw_curves, 1, lower)
    upper_values = torch.gather(raw_curves, 1, upper)
    return torch.lerp(lower_values, upper_values, weight), extrapolated


def physical_dlvsq_from_raw(
    raw_log_dlvsq: torch.Tensor,
    *,
    minimum_dlvsq: float,
    maximum_dlvsq: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map log-excess coordinates into strict finite DLV bounds."""

    if not 0.0 < minimum_dlvsq < maximum_dlvsq:
        raise ValueError("DLV bounds must be positive and increasing")
    epsilon = torch.finfo(raw_log_dlvsq.dtype).eps
    minimum_excess = epsilon * max(1.0, minimum_dlvsq)
    maximum_excess = (maximum_dlvsq - minimum_dlvsq) * (1.0 - 4.0 * epsilon)
    lower = torch.log(
        torch.as_tensor(
            minimum_excess,
            dtype=raw_log_dlvsq.dtype,
            device=raw_log_dlvsq.device,
        )
    )
    upper = torch.log(
        torch.as_tensor(
            maximum_excess,
            dtype=raw_log_dlvsq.dtype,
            device=raw_log_dlvsq.device,
        )
    )
    clipped = (raw_log_dlvsq < lower) | (raw_log_dlvsq > upper)
    physical = minimum_dlvsq + torch.exp(raw_log_dlvsq.clamp(lower, upper))
    return physical, clipped


__all__ = [
    "LastExpiryAr1Diagnostics",
    "LastExpiryAr1Fit",
    "TorchLastExpiryAr1",
    "fit_last_expiry_ar1",
    "interpolate_ar1_raw_curves",
    "make_antithetic_ar1_normals",
    "physical_dlvsq_from_raw",
    "torch_last_expiry_ar1",
]
