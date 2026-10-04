"""Covariance-supported trading coordinates for fixed empirical path sets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SpectralTradingTransform:
    """Per-time retained covariance modes and their whitening scales."""

    eigenvectors_by_time: tuple[torch.Tensor, ...]
    inverse_sqrt_eigenvalues_by_time: tuple[torch.Tensor, ...]
    diagnostics: tuple[dict[str, float | int | list[float]], ...]


def _mean_and_covariance(
    values: torch.Tensor,
    *,
    computation_device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a sample covariance in two passes with float32 GPU products."""

    if values.ndim != 2 or len(values) < 2:
        raise ValueError("spectral returns must have shape (paths,coordinates) with paths >= 2")
    if batch_size <= 0:
        raise ValueError("spectral covariance batch size must be positive")
    coordinates = values.shape[1]
    total = torch.zeros(coordinates, dtype=torch.float64, device=computation_device)
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(
            computation_device,
            dtype=torch.float32,
            non_blocking=values.device.type == "cpu",
        )
        total += torch.sum(batch, dim=0, dtype=torch.float64)
    mean = total / len(values)

    centered_cross_product = torch.zeros(
        (coordinates, coordinates),
        dtype=torch.float32,
        device=computation_device,
    )
    mean_float = mean.float()
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(
            computation_device,
            dtype=torch.float32,
            non_blocking=values.device.type == "cpu",
        )
        centered = batch - mean_float
        centered_cross_product.addmm_(centered.T, centered)
    covariance = centered_cross_product.double() / (len(values) - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def covariance_supported_coordinates(
    returns_by_time: Sequence[torch.Tensor],
    *,
    num_independent_blocks: int,
    standard_error_multiple: float,
    computation_device: torch.device | str,
    batch_size: int,
) -> tuple[tuple[torch.Tensor, ...], SpectralTradingTransform]:
    r"""Project returns onto reliably non-null covariance modes and whiten them.

    The variance floor at each trade time is

    ``multiple * sqrt(2/(B-1)) * mean(diag(covariance))``,

    where ``B`` is the number of independent path blocks. Returns are not
    centered when transformed, so their physical drift remains present.
    """

    if not returns_by_time:
        raise ValueError("at least one return slab is required")
    if num_independent_blocks < 2:
        raise ValueError("at least two independent blocks are required")
    if not math.isfinite(standard_error_multiple) or standard_error_multiple <= 0.0:
        raise ValueError("standard_error_multiple must be finite-positive")
    device = torch.device(computation_device)
    relative_standard_error = math.sqrt(2.0 / (num_independent_blocks - 1))
    transformed_by_time: list[torch.Tensor] = []
    eigenvectors_by_time: list[torch.Tensor] = []
    inverse_sqrt_by_time: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int | list[float]]] = []

    for trade_week, returns in enumerate(returns_by_time):
        if returns.ndim != 2 or len(returns) < 2 or returns.shape[1] < 1:
            raise ValueError("every return slab must be a non-empty path-coordinate matrix")
        _, covariance = _mean_and_covariance(
            returns,
            computation_device=device,
            batch_size=batch_size,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        # The cross-product is accumulated in float32 for GPU throughput, then
        # diagonalized in float64. Judge tiny negative values at the precision
        # of the accumulation which produced them, not at float64 epsilon.
        tolerance = (
            4.0
            * torch.finfo(torch.float32).eps
            * max(float(torch.amax(torch.abs(eigenvalues))), 1.0)
            * len(eigenvalues)
        )
        if float(torch.amin(eigenvalues)) < -tolerance:
            raise FloatingPointError("return covariance is not positive semidefinite")
        eigenvalues = torch.clamp_min(eigenvalues, 0.0)
        average_variance = float(torch.mean(torch.diagonal(covariance)))
        variance_floor = (
            standard_error_multiple * relative_standard_error * average_variance
        )
        retained = eigenvalues >= variance_floor
        if not bool(torch.any(retained)):
            raise RuntimeError("the covariance reliability floor removed every trading mode")
        selected_values = eigenvalues[retained].flip(0)
        selected_vectors = eigenvectors[:, retained].flip(1)
        inverse_sqrt = torch.rsqrt(selected_values)
        transformed = torch.empty(
            (len(returns), len(selected_values)),
            dtype=returns.dtype,
            device=returns.device,
        )
        vectors_float = selected_vectors.float()
        inverse_sqrt_float = inverse_sqrt.float()
        for start in range(0, len(returns), batch_size):
            stop = min(start + batch_size, len(returns))
            batch = returns[start:stop].to(
                device,
                dtype=torch.float32,
                non_blocking=returns.device.type == "cpu",
            )
            factors = (batch @ vectors_float) * inverse_sqrt_float[None]
            transformed[start:stop] = factors.to(
                returns.device,
                dtype=returns.dtype,
                non_blocking=returns.device.type == "cpu",
            )

        total_variance = float(torch.sum(eigenvalues))
        retained_variance = float(torch.sum(selected_values))
        discarded_values = eigenvalues[~retained]
        diagnostics.append(
            {
                "trade_week": trade_week,
                "num_input_coordinates": returns.shape[1],
                "num_retained_modes": len(selected_values),
                "num_discarded_modes": int(torch.sum(~retained)),
                "num_independent_blocks": num_independent_blocks,
                "relative_standard_error": relative_standard_error,
                "standard_error_multiple": standard_error_multiple,
                "average_coordinate_variance": average_variance,
                "variance_floor": variance_floor,
                "minimum_covariance_eigenvalue": float(torch.amin(eigenvalues)),
                "maximum_covariance_eigenvalue": float(torch.amax(eigenvalues)),
                "minimum_retained_eigenvalue": float(torch.amin(selected_values)),
                "maximum_discarded_eigenvalue": (
                    float(torch.amax(discarded_values)) if len(discarded_values) else 0.0
                ),
                "retained_variance_fraction": retained_variance / total_variance,
                "eigenvalues_descending": eigenvalues.flip(0).cpu().tolist(),
            }
        )
        transformed_by_time.append(transformed)
        eigenvectors_by_time.append(selected_vectors.cpu())
        inverse_sqrt_by_time.append(inverse_sqrt.cpu())

    return (
        tuple(transformed_by_time),
        SpectralTradingTransform(
            eigenvectors_by_time=tuple(eigenvectors_by_time),
            inverse_sqrt_eigenvalues_by_time=tuple(inverse_sqrt_by_time),
            diagnostics=tuple(diagnostics),
        ),
    )


__all__ = ["SpectralTradingTransform", "covariance_supported_coordinates"]
