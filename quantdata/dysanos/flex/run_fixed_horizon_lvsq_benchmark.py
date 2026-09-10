"""Benchmark the finite-h exponential LVsq martingale step on fitted states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.lvsq_drift import (
    FixedHorizonLvsqStep,
    fixed_contract_lvsq_drift,
    fixed_horizon_lvsq_exponential_step,
    smooth_lvsq_log_loadings,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FixedHorizonLvsqBenchmarkConfig:
    """Logical finite-step and processing specification."""

    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    backbone_annualized_volatility: float = 0.10
    benchmark_batch_sizes: tuple[int, ...] = (1, 4, 16, 64, 128, 256)
    all_state_batch_size: int = 256
    moment_tolerance: float = 1.0e-14

    def __post_init__(self) -> None:
        verify_inp(
            self.calendar_business_days > 0,
            lambda: "calendar_business_days must be positive",
        )
        verify_inp(
            np.isfinite(self.dlvsq_annualized_log_volatility)
            and self.dlvsq_annualized_log_volatility > 0.0,
            lambda: "DLVsq log volatility must be finite and positive",
        )
        verify_inp(
            np.isfinite(self.backbone_annualized_volatility)
            and self.backbone_annualized_volatility > 0.0,
            lambda: "backbone volatility must be finite and positive",
        )
        verify_inp(
            self.benchmark_batch_sizes
            and all(value > 0 for value in self.benchmark_batch_sizes)
            and self.all_state_batch_size > 0,
            lambda: "all batch sizes must be positive",
        )
        verify_inp(
            np.isfinite(self.moment_tolerance) and self.moment_tolerance > 0.0,
            lambda: "moment_tolerance must be finite and positive",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--all-state-batch-size", type=int, default=256)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_surface(decoder: Any, raw_states: torch.Tensor) -> Any:
    parameters = bound_native_raw(raw_states, geometry=decoder.geometry)
    return decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )


def _black_call_kernel(
    support: torch.Tensor,
    strikes: torch.Tensor,
    total_volatility: torch.Tensor,
) -> torch.Tensor:
    spot = support[..., None]
    strike = strikes[None, None, None]
    sqrt_variance = total_volatility[None, :, None, None]
    d1 = (
        torch.log(spot / strike) + 0.5 * sqrt_variance.square()
    ) / sqrt_variance
    d2 = d1 - sqrt_variance
    return spot * torch.special.ndtr(d1) - strike * torch.special.ndtr(d2)


def _mean_itm_option_residual_bps(
    *,
    density_mean: torch.Tensor,
    target_density: torch.Tensor,
    support: torch.Tensor,
    model_expiries_y: torch.Tensor,
    backbone_annualized_volatility: float,
) -> float:
    strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=support.dtype, device=support.device)
    )
    total_volatility = backbone_annualized_volatility * torch.sqrt(model_expiries_y)
    calls = _black_call_kernel(support, strikes, total_volatility)
    puts = calls - support[..., None] + strikes
    payoff = torch.where((strikes <= 1.0)[None, None, None], calls, puts)
    option_residual = torch.sum(
        (density_mean - target_density)[..., None] * payoff,
        dim=2,
    )
    return float((10_000.0 * torch.max(torch.abs(option_residual))).cpu())


def _local_exponential_benchmark(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    loadings: torch.Tensor,
    innovations: torch.Tensor,
    probabilities: torch.Tensor,
    horizon_y: float,
    backbone_annualized_volatility: float,
) -> dict[str, float]:
    _synchronize(raw_states.device)
    started = perf_counter()
    local = fixed_contract_lvsq_drift(
        decoder,
        raw_states,
        raw_log_loadings=loadings,
    )
    shocks = torch.einsum("bsf,rf->brs", local.raw_diffusion, innovations)
    branch_raw = (
        raw_states[:, None]
        + horizon_y * local.raw_drift[:, None]
        + horizon_y**0.5 * shocks
    )
    branch_surface = _decode_surface(
        decoder,
        branch_raw.reshape(-1, branch_raw.shape[-1]),
    )
    branch_density = branch_surface.densities.reshape(
        len(raw_states),
        len(innovations),
        decoder.geometry.num_model_expiries,
        decoder.geometry.num_inner_strikes + 2,
    )
    density_mean = torch.sum(
        probabilities[None, :, None, None] * branch_density,
        dim=1,
    )
    _synchronize(raw_states.device)
    seconds = perf_counter() - started
    target_surface = _decode_surface(decoder, raw_states)
    return {
        "seconds": seconds,
        "states_per_second": len(raw_states) / seconds,
        "maximum_density_mean_residual": float(
            torch.max(torch.abs(density_mean[:, 1:] - target_surface.densities[:, 1:])).cpu()
        ),
        "maximum_itm_option_mean_residual_bps_spot": _mean_itm_option_residual_bps(
            density_mean=density_mean[:, 1:],
            target_density=target_surface.densities[:, 1:],
            support=target_surface.support_strikes[:, 1:],
            model_expiries_y=decoder.model_expiries_y[1:],
            backbone_annualized_volatility=backbone_annualized_volatility,
        ),
    }


def _exact_benchmark(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    loadings: torch.Tensor,
    horizon_y: float,
    moment_tolerance: float,
    backbone_annualized_volatility: float,
) -> tuple[dict[str, object], FixedHorizonLvsqStep]:
    _synchronize(raw_states.device)
    started = perf_counter()
    result = fixed_horizon_lvsq_exponential_step(
        decoder,
        raw_states,
        raw_log_loadings=loadings,
        calendar_time_increment_y=horizon_y,
        moment_tolerance=moment_tolerance,
    )
    _synchronize(raw_states.device)
    seconds = perf_counter() - started
    density_mean = torch.sum(
        result.branch_probabilities[None, :, None, None]
        * result.densities_by_branch,
        dim=1,
    )
    target_surface = _decode_surface(decoder, raw_states)
    return (
        {
            "seconds": seconds,
            "states_per_second": len(raw_states) / seconds,
            "maximum_density_mean_residual": float(
                result.maximum_density_mean_residual.cpu()
            ),
            "maximum_density_dlvsq_mean_residual": float(
                result.maximum_density_dlvsq_mean_residual.cpu()
            ),
            "maximum_itm_option_mean_residual_bps_spot": (
                _mean_itm_option_residual_bps(
                    density_mean=density_mean[:, 1:],
                    target_density=target_surface.densities[:, 1:],
                    support=target_surface.support_strikes[:, 1:],
                    model_expiries_y=decoder.model_expiries_y[1:],
                    backbone_annualized_volatility=backbone_annualized_volatility,
                )
            ),
            "newton_iterations_by_expiry": list(
                result.newton_iterations_by_expiry
            ),
            "minimum_raw_cap_margin": float(result.minimum_raw_cap_margin.cpu()),
            "maximum_effective_minus_local_raw_drift": float(
                result.maximum_effective_minus_local_raw_drift.cpu()
            ),
        },
        result,
    )


def run(arguments: argparse.Namespace) -> None:
    """Run parallel scaling benchmarks and the complete state audit."""

    config = FixedHorizonLvsqBenchmarkConfig(
        all_state_batch_size=arguments.all_state_batch_size
    )
    env = BaseEnvironment(
        precision=arguments.precision,
        device=arguments.device,
        debug=False,
    )
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    with torch.no_grad():
        raw_states = model.raw_states_from_states(latent_states)
        first_surface = _decode_surface(model.sanos_decoder, raw_states[:1])
        loadings = smooth_lvsq_log_loadings(
            first_surface.support_strikes[0, 0],
            num_model_expiries=model.config.geometry.num_model_expiries,
            annualized_log_volatility=config.dlvsq_annualized_log_volatility,
        )
        innovations, probabilities = symmetric_rademacher_cubature(
            loadings.shape[-1],
            dtype=env.fdtype,
            device=env.device,
        )
        # Warm the CUDA kernels before recording either method's timings.
        _local_exponential_benchmark(
            decoder=model.sanos_decoder,
            raw_states=raw_states[:1],
            loadings=loadings,
            innovations=innovations,
            probabilities=probabilities,
            horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
            backbone_annualized_volatility=config.backbone_annualized_volatility,
        )
        _exact_benchmark(
            decoder=model.sanos_decoder,
            raw_states=raw_states[:1],
            loadings=loadings,
            horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
            moment_tolerance=config.moment_tolerance,
            backbone_annualized_volatility=config.backbone_annualized_volatility,
        )

        scaling: list[dict[str, object]] = []
        for batch_size in config.benchmark_batch_sizes:
            if batch_size > len(raw_states):
                continue
            indices = np.linspace(0, len(raw_states) - 1, batch_size, dtype=np.int64)
            selected = raw_states[
                torch.as_tensor(indices, dtype=torch.long, device=env.device)
            ]
            local_metrics = _local_exponential_benchmark(
                decoder=model.sanos_decoder,
                raw_states=selected,
                loadings=loadings,
                innovations=innovations,
                probabilities=probabilities,
                horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
                backbone_annualized_volatility=config.backbone_annualized_volatility,
            )
            exact_metrics, exact_result = _exact_benchmark(
                decoder=model.sanos_decoder,
                raw_states=selected,
                loadings=loadings,
                horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
                moment_tolerance=config.moment_tolerance,
                backbone_annualized_volatility=config.backbone_annualized_volatility,
            )
            scaling.append(
                {
                    "batch_size": batch_size,
                    "state_indices": indices.tolist(),
                    "local_equation_15_exponential": local_metrics,
                    "finite_h_exact_exponential": exact_metrics,
                    "exact_over_local_wall_time_ratio": (
                        float(exact_metrics["seconds"])
                        / local_metrics["seconds"]
                    ),
                }
            )
            print(
                f"batch={batch_size} local={local_metrics['seconds']:.4f}s "
                f"exact={exact_metrics['seconds']:.4f}s "
                f"exact_throughput={exact_metrics['states_per_second']:.1f}/s",
                flush=True,
            )
            del selected, exact_result
            if raw_states.device.type == "cuda":
                torch.cuda.empty_cache()

        all_state_chunks: list[dict[str, object]] = []
        all_state_started = perf_counter()
        for start in range(0, len(raw_states), config.all_state_batch_size):
            stop = min(start + config.all_state_batch_size, len(raw_states))
            metrics, result = _exact_benchmark(
                decoder=model.sanos_decoder,
                raw_states=raw_states[start:stop],
                loadings=loadings,
                horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
                moment_tolerance=config.moment_tolerance,
                backbone_annualized_volatility=config.backbone_annualized_volatility,
            )
            metrics.update({"start_state_index": start, "stop_state_index": stop})
            all_state_chunks.append(metrics)
            print(
                f"all states [{start}:{stop}] exact={metrics['seconds']:.4f}s "
                f"density_residual={metrics['maximum_density_mean_residual']:.3e}",
                flush=True,
            )
            del result
            if raw_states.device.type == "cuda":
                torch.cuda.empty_cache()
        all_state_seconds = perf_counter() - all_state_started

    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "fixed_horizon_lvsq_exponential_benchmark",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "num_states": len(raw_states),
        "num_cubature_branches": len(innovations),
        "sources": sources,
        "parallel_scaling": scaling,
        "all_state_exact": {
            "seconds_including_chunk_overhead": all_state_seconds,
            "states_per_second": len(raw_states) / all_state_seconds,
            "maximum_density_mean_residual": max(
                float(value["maximum_density_mean_residual"])
                for value in all_state_chunks
            ),
            "maximum_density_dlvsq_mean_residual": max(
                float(value["maximum_density_dlvsq_mean_residual"])
                for value in all_state_chunks
            ),
            "maximum_itm_option_mean_residual_bps_spot": max(
                float(value["maximum_itm_option_mean_residual_bps_spot"])
                for value in all_state_chunks
            ),
            "minimum_raw_cap_margin": min(
                float(value["minimum_raw_cap_margin"])
                for value in all_state_chunks
            ),
            "chunks": all_state_chunks,
        },
        "interpretation": {
            "local_method": (
                "Equation (15) with coefficients frozen at the start and one "
                "shifted-exponential DLV step."
            ),
            "exact_method": (
                "The same exponential branches with an expiry-triangular finite-h "
                "drift correction that exactly matches density-DLV moments."
            ),
            "backbone": "independent Black-Scholes factor fixed at 10% annualized",
        },
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
    }
    write_json_atomic(report, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    """Command-line entry point."""

    run(_parser().parse_args())


if __name__ == "__main__":
    main()
