"""Estimate the constant joint spot/log-DLV diffusion on a ragged GROMM2 grid."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, ndtri

FloatArray = NDArray[np.float64]


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class RaggedDiffusionConfig:
    """Smoothing and factor-rank controls for the ragged diffusion estimate."""

    factor_count: int = 16
    expiry_sigma: float = 1.0
    strike_sigma: float = 2.0
    fit_nms_range: tuple[float, float] = (-2.5, 1.5)
    dt: float = 1.0 / 252.0

    def __post_init__(self) -> None:
        verify_inp(
            self.factor_count >= 2,
            lambda: "factor_count must be at least two",
        )
        verify_inp(
            self.expiry_sigma >= 0.0 and self.strike_sigma >= 0.0,
            lambda: "smoothing sigmas must be non-negative",
        )
        verify_inp(
            self.fit_nms_range[0] < 0.0 < self.fit_nms_range[1],
            lambda: "fit_nms_range must contain zero",
        )
        verify_inp(self.dt > 0.0, lambda: "dt must be positive")


def _positive_curve_value(
    business_dtes: FloatArray,
    values: FloatArray,
    target_business_dte: float,
) -> float:
    return float(
        np.exp(
            np.interp(
                target_business_dte,
                business_dtes,
                np.log(values),
                left=np.log(values[0]),
                right=np.log(values[-1]),
            )
        )
    )


def _split_flat(
    values: FloatArray,
    offsets: NDArray[np.int64],
) -> tuple[FloatArray, ...]:
    return tuple(
        values[..., offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    )


def _fixed_contract_endpoint(
    *,
    endpoint_log_dlvsq: tuple[FloatArray, ...],
    supports: tuple[FloatArray, ...],
    business_dtes: FloatArray,
    origin_forward_over_spot: FloatArray,
    endpoint_forward_over_spot: FloatArray,
    gross_spot_return: float,
) -> tuple[FloatArray, int]:
    aligned: list[FloatArray] = []
    clipped = 0
    for origin_index in range(1, len(business_dtes)):
        target_dte = float(business_dtes[origin_index] - 1.0)
        upper = min(
            int(np.searchsorted(business_dtes, target_dte, side="left")),
            len(business_dtes) - 1,
        )
        lower = max(upper - 1, 0)
        if business_dtes[upper] == target_dte:
            lower = upper
        weight = (
            0.0
            if lower == upper
            else (target_dte - business_dtes[lower])
            / (business_dtes[upper] - business_dtes[lower])
        )
        origin_forward = _positive_curve_value(
            business_dtes,
            origin_forward_over_spot,
            float(business_dtes[origin_index]),
        )
        endpoint_forward = _positive_curve_value(
            business_dtes,
            endpoint_forward_over_spot,
            target_dte,
        )
        query = (
            supports[origin_index][1:-1]
            * origin_forward
            / (gross_spot_return * endpoint_forward)
        )
        lower_support = supports[lower][1:-1]
        upper_support = supports[upper][1:-1]
        clipped += int(
            np.count_nonzero(
                (query < lower_support[0])
                | (query > lower_support[-1])
                | (query < upper_support[0])
                | (query > upper_support[-1])
            )
        )
        lower_values = np.interp(query, lower_support, endpoint_log_dlvsq[lower])
        upper_values = np.interp(query, upper_support, endpoint_log_dlvsq[upper])
        aligned.append((1.0 - weight) * lower_values + weight * upper_values)
    return np.concatenate(aligned), clipped


def _spot_score(
    *,
    normalized_spot: float,
    support: FloatArray,
    density: FloatArray,
    backbone_total_variance: float,
) -> tuple[float, float]:
    standard_deviation = math.sqrt(backbone_total_variance)
    z = (
        np.log(normalized_spot / support) + 0.5 * backbone_total_variance
    ) / standard_deviation
    cumulative = float(density @ ndtr(z))
    clipped = float(np.clip(cumulative, 1.0e-10, 1.0 - 1.0e-10))
    return float(ndtri(clipped)), cumulative


def _coordinatewise_ar1(
    origins: FloatArray,
    endpoints: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    origin_mean = np.mean(origins, axis=0)
    endpoint_mean = np.mean(endpoints, axis=0)
    centered_origin = origins - origin_mean
    centered_endpoint = endpoints - endpoint_mean
    denominator = np.sum(centered_origin**2, axis=0)
    scale = np.maximum(np.sum(origins**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * len(origins)
    slope = np.zeros_like(denominator)
    slope[variable] = (
        np.sum(centered_origin * centered_endpoint, axis=0)[variable]
        / denominator[variable]
    )
    intercept = endpoint_mean - slope * origin_mean
    residual = endpoints - (intercept + slope * origins)
    residual -= np.mean(residual, axis=0, keepdims=True)
    return intercept, slope, residual


def _smooth_residuals(
    residuals: FloatArray,
    *,
    dlvsq_offsets: NDArray[np.int64],
    supports: tuple[FloatArray, ...],
    reference_total_variances: FloatArray,
    config: RaggedDiffusionConfig,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    samples = len(residuals)
    common_nms = np.linspace(
        config.fit_nms_range[0], config.fit_nms_range[1], 161
    )
    rectangular = np.empty((samples, len(supports) - 1, len(common_nms)))
    row_nms: list[FloatArray] = []
    for output_row, expiry_index in enumerate(range(1, len(supports))):
        start = dlvsq_offsets[expiry_index] - dlvsq_offsets[1]
        stop = dlvsq_offsets[expiry_index + 1] - dlvsq_offsets[1]
        nms = np.log(supports[expiry_index][1:-1]) / math.sqrt(
            reference_total_variances[expiry_index]
        )
        row_nms.append(nms)
        for sample in range(samples):
            rectangular[sample, output_row] = np.interp(
                common_nms, nms, residuals[sample, start:stop]
            )
    smoothed = gaussian_filter(
        rectangular,
        sigma=(0.0, config.expiry_sigma, config.strike_sigma),
        mode="nearest",
    )
    result = np.zeros_like(residuals)
    core_mask = np.zeros(residuals.shape[1], dtype=bool)
    for output_row, expiry_index in enumerate(range(1, len(supports))):
        start = dlvsq_offsets[expiry_index] - dlvsq_offsets[1]
        stop = dlvsq_offsets[expiry_index + 1] - dlvsq_offsets[1]
        nms = row_nms[output_row]
        core = (nms >= config.fit_nms_range[0]) & (
            nms <= config.fit_nms_range[1]
        )
        core_mask[start:stop] = core
        for sample in range(samples):
            result[sample, start:stop] = np.interp(
                nms, common_nms, smoothed[sample, output_row]
            )
        block = result[:, start:stop]
        block[:, ~core] = 0.0
    result -= np.mean(result, axis=0, keepdims=True)
    return result, core_mask


def _low_rank_joint_diffusion(
    innovations: FloatArray,
    spot_scores: FloatArray,
    *,
    factor_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    samples = len(innovations)
    centered_spot = spot_scores - np.mean(spot_scores)
    spot_standard_deviation = float(np.std(centered_spot, ddof=1))
    standardized_spot = centered_spot / spot_standard_deviation
    spot_loading = standardized_spot @ innovations / (samples - 1)
    orthogonal = innovations - standardized_spot[:, None] * spot_loading[None]
    left, singular, right_transpose = np.linalg.svd(
        orthogonal, full_matrices=False
    )
    residual_rank = factor_count - 1
    residual_loadings = (
        right_transpose[:residual_rank].T
        * singular[:residual_rank][None]
        / math.sqrt(samples - 1)
    )
    residual_scores = left[:, :residual_rank] * math.sqrt(samples - 1)
    loadings = np.column_stack((spot_loading, residual_loadings))
    scores = np.column_stack((standardized_spot, residual_scores))
    residual_eigenvalues = singular**2 / (samples - 1)
    factor_variances = np.concatenate(
        ([float(spot_loading @ spot_loading)], residual_eigenvalues)
    )
    return loadings, scores, factor_variances, standardized_spot


def _weighted_quantile(
    values: FloatArray,
    weights: FloatArray,
    probability: float,
) -> float:
    selected = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    ordered = np.argsort(values[selected])
    ordered_values = values[selected][ordered]
    ordered_weights = weights[selected][ordered]
    cumulative = np.cumsum(ordered_weights)
    index = np.searchsorted(cumulative, probability * cumulative[-1])
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def estimate_ragged_joint_diffusion(
    *,
    history_path: Path,
    dataset: Path,
    output: Path,
    config: RaggedDiffusionConfig = RaggedDiffusionConfig(),
) -> dict[str, object]:
    """Estimate the constant joint diffusion from strictly pre-origin pairs."""
    with h5py.File(history_path, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-causal-multiscale-history-v1":
            raise ValueError("history is not a causal multiscale GROMM2 panel")
        reference_date = str(stream.attrs["reference_date"])
        backbone_fraction = float(stream.attrs["backbone_atm_fraction"])
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        support_offsets = np.asarray(stream["support_offsets"], dtype=np.int64)
        supports = _split_flat(
            np.asarray(stream["supports"], dtype=np.float64), support_offsets
        )
        dlvsq_offsets = np.asarray(stream["dlvsq_offsets"], dtype=np.int64)
        dlvsq = np.asarray(stream["dlvsq"], dtype=np.float64)
        densities = np.asarray(stream["densities"], dtype=np.float64)
        total_variances = np.asarray(
            stream["projected_atm_total_variances"], dtype=np.float64
        )
    observations = pl.read_parquet(
        dataset / "observations.parquet",
        columns=[
            "date",
            "business_sessions_from_previous_observation",
            "forward_carry_from_previous_observation",
            "forward_adjusted_spot_return_into_date",
        ],
    )
    expiries = pl.read_parquet(
        dataset / "expiries.parquet",
        columns=["date", "business_dtes", "forward_over_spot"],
    )
    observation_map = {
        row["date"].isoformat(): row for row in observations.iter_rows(named=True)
    }
    expiry_map = {
        row["date"].isoformat(): row for row in expiries.iter_rows(named=True)
    }
    log_dlvsq = np.log(dlvsq)
    first_future = int(dlvsq_offsets[1])
    origins: list[FloatArray] = []
    endpoints: list[FloatArray] = []
    raw_spot_scores: list[float] = []
    front_cdfs: list[float] = []
    origin_indices: list[int] = []
    clipped_queries = 0
    total_queries = 0
    for endpoint_index in range(1, len(dates)):
        endpoint_date = str(dates[endpoint_index])
        if endpoint_date >= reference_date:
            continue
        observation = observation_map[endpoint_date]
        if observation["business_sessions_from_previous_observation"] != 1:
            continue
        origin_index = endpoint_index - 1
        origin_curve = expiry_map[str(dates[origin_index])]
        endpoint_curve = expiry_map[endpoint_date]
        origin_forward = np.asarray(
            [
                _positive_curve_value(
                    np.asarray(origin_curve["business_dtes"], dtype=np.float64),
                    np.asarray(origin_curve["forward_over_spot"], dtype=np.float64),
                    float(target),
                )
                for target in business_dtes
            ]
        )
        endpoint_forward = np.asarray(
            [
                _positive_curve_value(
                    np.asarray(endpoint_curve["business_dtes"], dtype=np.float64),
                    np.asarray(endpoint_curve["forward_over_spot"], dtype=np.float64),
                    float(target),
                )
                for target in business_dtes
            ]
        )
        gross_spot_return = float(
            observation["forward_carry_from_previous_observation"]
        ) * (1.0 + float(observation["forward_adjusted_spot_return_into_date"]))
        aligned, clipped = _fixed_contract_endpoint(
            endpoint_log_dlvsq=_split_flat(log_dlvsq[endpoint_index], dlvsq_offsets),
            supports=supports,
            business_dtes=business_dtes,
            origin_forward_over_spot=origin_forward,
            endpoint_forward_over_spot=endpoint_forward,
            gross_spot_return=gross_spot_return,
        )
        origins.append(log_dlvsq[origin_index, first_future:])
        endpoints.append(aligned)
        clipped_queries += clipped
        total_queries += len(aligned)
        front_density = densities[
            origin_index, support_offsets[0] : support_offsets[1]
        ]
        score, cdf = _spot_score(
            normalized_spot=(
                1.0
                + float(observation["forward_adjusted_spot_return_into_date"])
            ),
            support=supports[0],
            density=front_density,
            backbone_total_variance=(
                backbone_fraction**2 * total_variances[origin_index, 0]
            ),
        )
        raw_spot_scores.append(score)
        front_cdfs.append(cdf)
        origin_indices.append(origin_index)
    origin_values = np.stack(origins)
    endpoint_values = np.stack(endpoints)
    raw_scores = np.asarray(raw_spot_scores)
    if len(origin_values) <= config.factor_count + 2:
        raise ValueError("too few strict pre-origin pairs for factor count")
    intercept, slope, residual = _coordinatewise_ar1(
        origin_values, endpoint_values
    )
    reference_index = int(np.flatnonzero(dates == reference_date)[0])
    smoothed, core_mask = _smooth_residuals(
        residual,
        dlvsq_offsets=dlvsq_offsets,
        supports=supports,
        reference_total_variances=total_variances[reference_index],
        config=config,
    )
    reference_density_parts = _split_flat(
        densities[reference_index], support_offsets
    )
    localization_parts: list[FloatArray] = []
    density_weight_parts: list[FloatArray] = []
    for expiry_index in range(1, len(supports)):
        interior = np.maximum(reference_density_parts[expiry_index][1:-1], 0.0)
        density_weight_parts.append(interior / max(np.sum(interior), 1.0e-300))
        floor = 1.0e-4 * max(float(np.max(interior)), 1.0e-300)
        localization_parts.append(interior / (interior + floor))
    localization = np.concatenate(localization_parts) * core_mask
    density_weights = np.concatenate(density_weight_parts) / (len(supports) - 1)
    density_weights *= core_mask
    density_weights /= np.sum(density_weights)
    innovations = smoothed * localization[None]
    loadings, scores, factor_variances, standardized_spot = (
        _low_rank_joint_diffusion(
            innovations,
            raw_scores,
            factor_count=config.factor_count,
        )
    )
    annualized_loadings = loadings / math.sqrt(config.dt)
    annualized_volatility = np.sqrt(np.sum(annualized_loadings**2, axis=1))
    spot_correlation = np.divide(
        annualized_loadings[:, 0],
        annualized_volatility,
        out=np.zeros_like(annualized_volatility),
        where=annualized_volatility > 0.0,
    )
    total_variance = float(np.sum(np.var(innovations, axis=0, ddof=1)))
    retained = np.cumsum(factor_variances[: config.factor_count])
    origin_index_array = np.asarray(origin_indices, dtype=np.int64)
    arrays = {
        "schema": np.asarray("gromm2-causal-joint-diffusion-v1"),
        "config": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "reference_date": np.asarray(reference_date),
        "origin_indices": origin_index_array,
        "raw_spot_scores": raw_scores,
        "standardized_spot_scores": standardized_spot,
        "front_cdf_values": np.asarray(front_cdfs),
        "ar_intercept": intercept,
        "ar_slope": slope,
        "annualized_log_dlvsq_loadings": annualized_loadings,
        "annualized_log_dlvsq_volatility": annualized_volatility,
        "spot_correlation": spot_correlation,
        "factor_scores": scores,
        "factor_variances": factor_variances,
        "localization": localization,
        "density_weights": density_weights,
        "core_mask": core_mask,
        "dlvsq_offsets": dlvsq_offsets[1:] - first_future,
    }
    _atomic_save_npz(output, arrays)
    report: dict[str, object] = {
        "schema": "gromm2-causal-joint-diffusion-v1",
        "reference_date": reference_date,
        "strictly_pre_origin": True,
        "pair_count": len(origin_values),
        "factor_count": config.factor_count,
        "spot_score_mean": float(np.mean(raw_scores)),
        "spot_score_standard_deviation": float(np.std(raw_scores, ddof=1)),
        "spot_score_minimum": float(np.min(raw_scores)),
        "spot_score_maximum": float(np.max(raw_scores)),
        "fixed_contract_interpolation_clip_fraction": (
            clipped_queries / total_queries
        ),
        "explained_variance_fraction_rank_8": float(
            retained[min(7, len(retained) - 1)] / total_variance
        ),
        "explained_variance_fraction_rank_16": float(retained[-1] / total_variance),
        "annualized_log_dlvsq_volatility": {
            f"density_weighted_p{probability:02d}": _weighted_quantile(
                annualized_volatility, density_weights, probability / 100.0
            )
            for probability in (10, 50, 90, 99)
        },
        "spot_correlation": {
            **{
                f"density_weighted_p{probability:02d}": _weighted_quantile(
                    spot_correlation, density_weights, probability / 100.0
                )
                for probability in (10, 50, 90)
            },
            "density_weighted_mean": float(density_weights @ spot_correlation),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-count", type=int, default=16)
    arguments = parser.parse_args()
    report = estimate_ragged_joint_diffusion(
        history_path=arguments.history,
        dataset=arguments.dataset,
        output=arguments.output,
        config=RaggedDiffusionConfig(factor_count=arguments.factor_count),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
