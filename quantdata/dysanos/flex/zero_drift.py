"""Covariance-aware tests and objectives for finite Q martingale moments."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class ZeroDriftStatistics(NamedTuple):
    """Mean, regularized covariance, precision, and quadratic zero-drift score."""

    mean: torch.Tensor
    covariance_of_mean: torch.Tensor
    precision: torch.Tensor
    quadratic: torch.Tensor
    quadratic_per_moment: torch.Tensor


class SpectralZeroDriftStatistics(NamedTuple):
    """Zero-drift statistics using a truncated covariance eigenspace."""

    mean: torch.Tensor
    covariance_of_mean: torch.Tensor
    precision: torch.Tensor
    quadratic: torch.Tensor
    quadratic_per_moment: torch.Tensor
    correlation_eigenvalues: torch.Tensor
    retained_rank: torch.Tensor


class BetaFactorCovariance(NamedTuple):
    """Spot-factor and regularized residual components of a mean covariance."""

    covariance_of_mean: torch.Tensor
    spot_factor_covariance_of_mean: torch.Tensor
    residual_covariance_of_mean: torch.Tensor


class ConditionedPrecision(NamedTuple):
    """Full-rank correlation-shrunk covariance and its capped precision."""

    covariance_of_mean: torch.Tensor
    precision: torch.Tensor
    correlation_eigenvalues: torch.Tensor
    condition_number: torch.Tensor


def correlation_conditioned_precision(
    covariance_of_mean: torch.Tensor,
    *,
    diagonal_shrinkage: float = 0.10,
    maximum_condition_number: float = 100.0,
) -> ConditionedPrecision:
    """Return a full-rank precision after shrinkage and eigenvalue flooring.

    Conditioning is performed in correlation space so the result is invariant
    to instrument units.  Unlike the inference-only spectral statistic, no
    direction is discarded: weak directions are retained with a precision cap.
    """

    if (
        covariance_of_mean.ndim < 2
        or covariance_of_mean.shape[-1] != covariance_of_mean.shape[-2]
    ):
        raise ValueError("covariance must end in a square matrix")
    if not 0.0 <= diagonal_shrinkage <= 1.0:
        raise ValueError("diagonal shrinkage must lie in [0, 1]")
    if (
        not math.isfinite(maximum_condition_number)
        or maximum_condition_number <= 1.0
    ):
        raise ValueError("maximum condition number must be finite and above one")

    covariance = covariance_of_mean.double()
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    raw_diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    typical_variance = torch.median(torch.abs(raw_diagonal), dim=-1).values
    diagonal_floor = torch.maximum(
        typical_variance * 1.0e-12,
        torch.full_like(typical_variance, torch.finfo(torch.float64).tiny),
    )
    diagonal = torch.maximum(raw_diagonal, diagonal_floor[..., None])
    scale = torch.sqrt(diagonal)
    correlation = covariance / (scale[..., :, None] * scale[..., None, :])
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    correlation = correlation.clamp(-1.0, 1.0)
    correlation = correlation.clone()
    correlation.diagonal(dim1=-2, dim2=-1).copy_(torch.ones_like(diagonal))

    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    eigenvalues = eigenvalues.clamp_min(0.0)
    shrunk_eigenvalues = (
        (1.0 - diagonal_shrinkage) * eigenvalues + diagonal_shrinkage
    )
    maximum_eigenvalue = torch.amax(shrunk_eigenvalues, dim=-1, keepdim=True)
    minimum_eigenvalue = maximum_eigenvalue / maximum_condition_number
    conditioned_eigenvalues = torch.maximum(shrunk_eigenvalues, minimum_eigenvalue)
    conditioned_correlation = torch.matmul(
        eigenvectors * conditioned_eigenvalues[..., None, :],
        eigenvectors.transpose(-1, -2),
    )
    correlation_precision = torch.matmul(
        eigenvectors * torch.reciprocal(conditioned_eigenvalues)[..., None, :],
        eigenvectors.transpose(-1, -2),
    )
    conditioned_covariance = conditioned_correlation * (
        scale[..., :, None] * scale[..., None, :]
    )
    precision = correlation_precision / (
        scale[..., :, None] * scale[..., None, :]
    )
    condition_number = maximum_eigenvalue[..., 0] / torch.amin(
        conditioned_eigenvalues, dim=-1
    )
    return ConditionedPrecision(
        covariance_of_mean=conditioned_covariance.detach(),
        precision=precision.detach(),
        correlation_eigenvalues=conditioned_eigenvalues.detach(),
        condition_number=condition_number.detach(),
    )


def regularize_covariance(
    covariance: torch.Tensor,
    *,
    diagonal_shrinkage: float,
    ridge_fraction: float,
) -> torch.Tensor:
    """Return a positive-definite diagonal-shrinkage covariance estimate."""

    if covariance.ndim < 2 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("covariance must end in a square matrix")
    if not 0.0 <= diagonal_shrinkage <= 1.0:
        raise ValueError("diagonal shrinkage must lie in [0, 1]")
    if not math.isfinite(ridge_fraction) or ridge_fraction <= 0.0:
        raise ValueError("ridge fraction must be finite and positive")
    values = covariance.double()
    values = 0.5 * (values + values.transpose(-1, -2))
    raw_diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    typical_variance = torch.median(torch.abs(raw_diagonal), dim=-1).values
    diagonal_floor = torch.maximum(
        typical_variance * 1.0e-12,
        torch.full_like(typical_variance, torch.finfo(torch.float64).tiny),
    )
    diagonal = torch.maximum(raw_diagonal, diagonal_floor[..., None])
    scale = torch.sqrt(diagonal)
    correlation = values / (scale[..., :, None] * scale[..., None, :])
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    correlation = correlation.clamp(-1.0, 1.0)
    identity = torch.eye(
        values.shape[-1],
        dtype=torch.float64,
        device=values.device,
    )
    correlation = correlation.clone()
    correlation.diagonal(dim1=-2, dim2=-1).copy_(
        torch.ones_like(diagonal)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    correlation = torch.matmul(
        eigenvectors * eigenvalues.clamp_min(0.0)[..., None, :],
        eigenvectors.transpose(-1, -2),
    )
    regularized_correlation = (
        (1.0 - diagonal_shrinkage) * correlation
        + diagonal_shrinkage * identity
        + ridge_fraction * identity
    )
    return regularized_correlation * (
        scale[..., :, None] * scale[..., None, :]
    )


def beta_factor_covariance(
    *,
    beta: torch.Tensor,
    spot_variance_of_mean: torch.Tensor,
    residual_covariance_of_mean: torch.Tensor,
    residual_diagonal_shrinkage: float = 0.10,
    residual_ridge_fraction: float = 1.0e-6,
) -> BetaFactorCovariance:
    """Build ``beta beta' Var(mean(S))`` plus a shrunk residual covariance."""

    moments = beta.shape[-1]
    if (
        residual_covariance_of_mean.shape != beta.shape + (moments,)
        or spot_variance_of_mean.shape != beta.shape[:-1]
    ):
        raise ValueError("beta-factor covariance shapes are incompatible")
    if bool(torch.any(spot_variance_of_mean < 0.0)):
        raise ValueError("spot variance of the mean cannot be negative")
    beta64 = beta.double()
    spot_factor = (
        spot_variance_of_mean.double()[..., None, None]
        * beta64[..., :, None]
        * beta64[..., None, :]
    )
    residual = regularize_covariance(
        residual_covariance_of_mean,
        diagonal_shrinkage=residual_diagonal_shrinkage,
        ridge_fraction=residual_ridge_fraction,
    )
    return BetaFactorCovariance(
        covariance_of_mean=spot_factor + residual,
        spot_factor_covariance_of_mean=spot_factor,
        residual_covariance_of_mean=residual,
    )


def spectral_zero_drift(
    mean: torch.Tensor,
    covariance_of_mean: torch.Tensor,
    *,
    correlation_eigenvalue_cutoff: float,
    detach_covariance: bool = True,
) -> SpectralZeroDriftStatistics:
    """Test zero drift after discarding weak covariance-correlation directions."""

    moments = mean.shape[-1]
    if covariance_of_mean.shape != mean.shape + (moments,):
        raise ValueError("mean and covariance shapes are incompatible")
    if not 0.0 <= correlation_eigenvalue_cutoff < 1.0:
        raise ValueError("the correlation eigenvalue cutoff must lie in [0, 1)")
    covariance = covariance_of_mean.double()
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    if bool(torch.any(diagonal <= 0.0)):
        raise ValueError("spectral zero-drift covariance must have positive diagonal")
    scale = torch.sqrt(diagonal)
    correlation = covariance / (scale[..., :, None] * scale[..., None, :])
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    eigenvalues = eigenvalues.clamp_min(0.0)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    retained_rank = torch.sum(retained, dim=-1)
    if bool(torch.any(retained_rank == 0)):
        raise ValueError("the spectral cutoff discarded every covariance direction")
    inverse_eigenvalues = torch.where(
        retained,
        torch.reciprocal(eigenvalues.clamp_min(torch.finfo(torch.float64).tiny)),
        torch.zeros_like(eigenvalues),
    )
    correlation_precision = torch.matmul(
        eigenvectors * inverse_eigenvalues[..., None, :],
        eigenvectors.transpose(-1, -2),
    )
    precision = correlation_precision / (
        scale[..., :, None] * scale[..., None, :]
    )
    if detach_covariance:
        covariance = covariance.detach()
        precision = precision.detach()
    mean64 = mean.double()
    quadratic = torch.einsum("...i,...ij,...j->...", mean64, precision, mean64)
    return SpectralZeroDriftStatistics(
        mean=mean64,
        covariance_of_mean=covariance,
        precision=precision,
        quadratic=quadratic,
        quadratic_per_moment=quadratic / retained_rank,
        correlation_eigenvalues=eigenvalues,
        retained_rank=retained_rank,
    )


def beta_factor_block_zero_drift_statistics(
    raw_block_means: torch.Tensor,
    spot_block_means: torch.Tensor,
    beta: torch.Tensor,
    *,
    residual_diagonal_shrinkage: float = 0.10,
    residual_ridge_fraction: float = 1.0e-6,
    correlation_eigenvalue_cutoff: float = 0.01,
) -> SpectralZeroDriftStatistics:
    """Estimate a spot-factor covariance from independent simulation blocks."""

    if raw_block_means.ndim < 2 or len(raw_block_means) < 3:
        raise ValueError("beta-factor inference needs at least three blocks")
    if (
        spot_block_means.shape != raw_block_means.shape[:-1]
        or beta.shape != raw_block_means.shape[1:]
    ):
        raise ValueError("beta-factor block shapes are incompatible")
    raw = raw_block_means.double()
    spot = spot_block_means.double()
    beta64 = beta.double()
    mean = torch.mean(raw, dim=0) - beta64 * torch.mean(spot, dim=0)[..., None]
    residual = raw - beta64[None] * spot[..., None]
    centered_residual = residual - torch.mean(residual, dim=0)
    centered_spot = spot - torch.mean(spot, dim=0)
    blocks = len(raw)
    residual_covariance_of_mean = torch.einsum(
        "b...i,b...j->...ij",
        centered_residual,
        centered_residual,
    ) / ((blocks - 1) * blocks)
    spot_variance_of_mean = torch.sum(centered_spot.square(), dim=0) / (
        (blocks - 1) * blocks
    )
    factor = beta_factor_covariance(
        beta=beta64,
        spot_variance_of_mean=spot_variance_of_mean,
        residual_covariance_of_mean=residual_covariance_of_mean,
        residual_diagonal_shrinkage=residual_diagonal_shrinkage,
        residual_ridge_fraction=residual_ridge_fraction,
    )
    return spectral_zero_drift(
        mean,
        factor.covariance_of_mean,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )


def quadratic_zero_drift(
    mean: torch.Tensor,
    covariance_of_mean: torch.Tensor,
    *,
    diagonal_shrinkage: float = 0.10,
    ridge_fraction: float = 1.0e-6,
    detach_covariance: bool = True,
) -> ZeroDriftStatistics:
    """Compute the covariance-aware Gaussian/GMM zero-mean quadratic."""

    if covariance_of_mean.shape != mean.shape + (mean.shape[-1],):
        raise ValueError("mean and covariance shapes are incompatible")
    covariance = regularize_covariance(
        covariance_of_mean,
        diagonal_shrinkage=diagonal_shrinkage,
        ridge_fraction=ridge_fraction,
    )
    if detach_covariance:
        covariance = covariance.detach()
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if bool(torch.any(info != 0)):
        raise ValueError("regularized covariance is not positive definite")
    precision = torch.cholesky_inverse(cholesky)
    mean64 = mean.double()
    quadratic = torch.einsum("...i,...ij,...j->...", mean64, precision, mean64)
    return ZeroDriftStatistics(
        mean=mean64,
        covariance_of_mean=covariance,
        precision=precision,
        quadratic=quadratic,
        quadratic_per_moment=quadratic / mean.shape[-1],
    )


def block_zero_drift_statistics(
    block_means: torch.Tensor,
    *,
    diagonal_shrinkage: float = 0.10,
    ridge_fraction: float = 1.0e-6,
    detach_covariance: bool = True,
) -> ZeroDriftStatistics:
    """Estimate covariance from independent block means and test zero drift."""

    if block_means.ndim < 2 or len(block_means) < 3:
        raise ValueError("zero-drift inference needs at least three blocks")
    values = block_means.double()
    mean = torch.mean(values, dim=0)
    centered = values - mean
    covariance_between_blocks = torch.einsum(
        "b...i,b...j->...ij",
        centered,
        centered,
    ) / (len(values) - 1)
    covariance_of_mean = covariance_between_blocks / len(values)
    return quadratic_zero_drift(
        mean,
        covariance_of_mean,
        diagonal_shrinkage=diagonal_shrinkage,
        ridge_fraction=ridge_fraction,
        detach_covariance=detach_covariance,
    )


def adjusted_path_mean_covariance(
    *,
    path_sum: torch.Tensor,
    path_cross_product_sum: torch.Tensor,
    spot_sum: torch.Tensor,
    spot_square_sum: torch.Tensor,
    spot_return_product_sum: torch.Tensor,
    beta: torch.Tensor,
    num_paths: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adjusted means and their full cross-instrument covariance."""

    instruments = path_sum.shape[-1]
    expected_cross_shape = path_sum.shape + (instruments,)
    if (
        path_cross_product_sum.shape != expected_cross_shape
        or spot_return_product_sum.shape != path_sum.shape
        or beta.shape != path_sum.shape
        or spot_sum.shape != path_sum.shape[:-1]
        or spot_square_sum.shape != path_sum.shape[:-1]
        or num_paths <= 1
    ):
        raise ValueError("adjusted-return sufficient statistics are incompatible")
    sums = path_sum.double()
    cross = path_cross_product_sum.double()
    spot_products = spot_return_product_sum.double()
    beta64 = beta.double()
    spot_sum64 = spot_sum.double()
    spot_square64 = spot_square_sum.double()
    adjusted_sum = sums - beta64 * spot_sum64[..., None]
    adjusted_cross = (
        cross
        - spot_products[..., :, None] * beta64[..., None, :]
        - beta64[..., :, None] * spot_products[..., None, :]
        + beta64[..., :, None]
        * beta64[..., None, :]
        * spot_square64[..., None, None]
    )
    centered_cross = adjusted_cross - (
        adjusted_sum[..., :, None] * adjusted_sum[..., None, :] / num_paths
    )
    path_covariance = centered_cross / (num_paths - 2)
    covariance_of_mean = path_covariance / num_paths
    return adjusted_sum / num_paths, covariance_of_mean


def raw_path_mean_covariance(
    *,
    path_sum: torch.Tensor,
    path_cross_product_sum: torch.Tensor,
    num_paths: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw option-return means and their full covariance."""

    instruments = path_sum.shape[-1]
    if (
        path_cross_product_sum.shape != path_sum.shape + (instruments,)
        or num_paths <= 1
    ):
        raise ValueError("raw-return sufficient statistics are incompatible")
    sums = path_sum.double()
    centered_cross = path_cross_product_sum.double() - (
        sums[..., :, None] * sums[..., None, :] / num_paths
    )
    path_covariance = centered_cross / (num_paths - 1)
    return sums / num_paths, path_covariance / num_paths


def sign_flip_zero_drift_bootstrap(
    block_means: torch.Tensor,
    *,
    precision: torch.Tensor,
    covariance_of_mean: torch.Tensor,
    samples: int,
    seed: int,
    tail_fraction: float = 0.10,
    independent_states: bool = True,
    chunk_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Calibrate dense, maximum, and tail quadratics under the zero-drift null."""

    if block_means.ndim != 3:
        raise ValueError("bootstrap block means must have shape [blocks, states, moments]")
    blocks, states, moments = block_means.shape
    expected_covariance_shape = (states, moments, moments)
    if (
        precision.shape != expected_covariance_shape
        or covariance_of_mean.shape != expected_covariance_shape
    ):
        raise ValueError("bootstrap covariance shapes differ from block means")
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap sample and chunk counts must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    values = block_means.double()
    centered = values - torch.mean(values, dim=0)
    precision64 = precision.detach().double()
    generator = torch.Generator(device=values.device).manual_seed(seed)
    dense_values: list[torch.Tensor] = []
    maximum_values: list[torch.Tensor] = []
    tail_values: list[torch.Tensor] = []
    coordinate_values: list[torch.Tensor] = []
    tail_count = max(1, math.ceil(tail_fraction * states))
    standard_error = torch.sqrt(
        torch.diagonal(covariance_of_mean.double(), dim1=-2, dim2=-1)
    ).clamp_min(torch.finfo(torch.float64).tiny)
    remaining = samples
    while remaining:
        count = min(chunk_size, remaining)
        sign_states = states if independent_states else 1
        signs = 2.0 * torch.randint(
            0,
            2,
            (count, blocks, sign_states),
            device=values.device,
            generator=generator,
        ).double() - 1.0
        if not independent_states:
            signs = signs.expand(-1, -1, states)
        pseudo_mean = torch.einsum("kbs,bsp->ksp", signs, centered) / blocks
        quadratic = torch.einsum(
            "ksi,sij,ksj->ks",
            pseudo_mean,
            precision64,
            pseudo_mean,
        )
        dense_values.append(torch.sum(quadratic, dim=1))
        maximum_values.append(torch.max(quadratic, dim=1).values)
        tail_values.append(
            torch.mean(torch.topk(quadratic, tail_count, dim=1).values, dim=1)
        )
        coordinate_values.append(
            torch.amax(torch.abs(pseudo_mean) / standard_error[None], dim=(1, 2))
        )
        remaining -= count
    return {
        "dense_sum": torch.cat(dense_values),
        "state_maximum": torch.cat(maximum_values),
        "state_tail_mean": torch.cat(tail_values),
        "coordinate_maximum": torch.cat(coordinate_values),
    }


__all__ = [
    "BetaFactorCovariance",
    "SpectralZeroDriftStatistics",
    "ZeroDriftStatistics",
    "adjusted_path_mean_covariance",
    "beta_factor_block_zero_drift_statistics",
    "beta_factor_covariance",
    "block_zero_drift_statistics",
    "quadratic_zero_drift",
    "raw_path_mean_covariance",
    "regularize_covariance",
    "sign_flip_zero_drift_bootstrap",
    "spectral_zero_drift",
]
