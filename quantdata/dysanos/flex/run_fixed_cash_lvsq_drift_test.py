"""Run the one-week fixed-cash LVsq drift test before endpoint regridding."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lvsq_drift import (
    fixed_contract_lvsq_drift,
    fixed_contract_lvsq_euler_step,
    smooth_lvsq_log_loadings,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_four_step_lvsq_comparison import (
    _decode_surface,
    _risk_neutrality_audit,
    _set_constant_bs_backbone,
    _surface_option_values,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FixedCashLvsqDriftTestConfig:
    """Logical test and processing configuration."""

    num_paths: int = 10_000
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    state_batch_size: int = 4_096
    surface_batch_size: int = 2_048
    covariance_eigenvalue_fraction: float = 1.0e-8
    seed: int = 2_026_090_8

    def __post_init__(self) -> None:
        verify_inp(self.num_paths > 1, lambda: "num_paths must exceed one")
        verify_inp(
            self.calendar_business_days > 0,
            lambda: "calendar_business_days must be positive",
        )
        verify_inp(
            min(self.state_batch_size, self.surface_batch_size) > 0,
            lambda: "batch sizes must be positive",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-paths", type=int, default=10_000)
    return parser


def _synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(arguments: argparse.Namespace) -> None:
    """Simulate exactly one fixed-cash week with no spot input."""

    config = FixedCashLvsqDriftTestConfig(num_paths=arguments.num_paths)
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
    geometry = model.config.geometry
    if not isinstance(
        geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig
    ):
        raise ValueError("fixed-cash drift test requires the fixed relative grid")
    rng = np.random.default_rng(config.seed)
    sampled_indices_numpy = rng.integers(
        0, len(latent_states), size=config.num_paths
    )
    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        sampled_indices = torch.as_tensor(
            sampled_indices_numpy, dtype=torch.long, device=env.device
        )
        initial_raw = all_raw[sampled_indices]
        initial_raw, forward_parameters = _set_constant_bs_backbone(
            initial_raw,
            decoder=model.sanos_decoder,
            annualized_volatility=config.bs_backbone_annualized_volatility,
        )
        first_surface = _decode_surface(model.sanos_decoder, initial_raw[:1])
        loadings = smooth_lvsq_log_loadings(
            first_surface.support_strikes[0, 0],
            num_model_expiries=geometry.num_model_expiries,
            annualized_log_volatility=config.dlvsq_annualized_log_volatility,
        )
        innovations, probabilities = symmetric_rademacher_cubature(
            loadings.shape[-1], dtype=env.fdtype, device=env.device
        )
        del probabilities
        branch_indices = torch.as_tensor(
            rng.integers(0, len(innovations), size=config.num_paths),
            dtype=torch.long,
            device=env.device,
        )
        path_innovations = innovations[branch_indices]
        strikes = torch.exp(
            torch.linspace(
                -0.30,
                0.30,
                config.num_option_strikes,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        unit_spot = torch.ones(
            config.num_paths, dtype=env.fdtype, device=env.device
        )
        initial_options = _surface_option_values(
            decoder=model.sanos_decoder,
            raw_states=initial_raw,
            cumulative_spot=unit_spot,
            fixed_cash_strikes=strikes,
            batch_size=config.surface_batch_size,
            fixed_cash=False,
        )
        next_raw = torch.empty_like(initial_raw)
        maximum_volatility_residual = 0.0
        maximum_drift_residual = 0.0
        minimum_cap_margin = float("inf")
        _synchronize(env.device)
        started = perf_counter()
        for start in range(0, config.num_paths, config.state_batch_size):
            stop = min(start + config.state_batch_size, config.num_paths)
            restriction = fixed_contract_lvsq_drift(
                model.sanos_decoder,
                initial_raw[start:stop],
                raw_log_loadings=loadings,
            )
            next_raw[start:stop] = fixed_contract_lvsq_euler_step(
                restriction,
                initial_raw[start:stop],
                calendar_time_increment_y=(
                    config.calendar_business_days / DAYS_PER_YEAR
                ),
                standard_normals=path_innovations[start:stop],
            )
            maximum_volatility_residual = max(
                maximum_volatility_residual,
                float(restriction.maximum_volatility_equation_residual.cpu()),
            )
            maximum_drift_residual = max(
                maximum_drift_residual,
                float(restriction.maximum_drift_equation_residual.cpu()),
            )
            minimum_cap_margin = min(
                minimum_cap_margin,
                float(restriction.minimum_raw_cap_margin.cpu()),
            )
        _synchronize(env.device)
        transition_seconds = perf_counter() - started
        next_options = _surface_option_values(
            decoder=model.sanos_decoder,
            raw_states=next_raw,
            cumulative_spot=unit_spot,
            fixed_cash_strikes=strikes,
            batch_size=config.surface_batch_size,
            fixed_cash=False,
        )
        raw_dlvsq = initial_raw.reshape(
            config.num_paths,
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )[..., : geometry.num_inner_strikes].reshape(config.num_paths, -1)
        features = (
            raw_dlvsq @ loadings.reshape(-1, loadings.shape[-1])
        ).cpu().double().numpy()
        features = (features - np.mean(features, axis=0)) / np.std(
            features, axis=0, ddof=1
        )
        instruments = np.concatenate(
            (np.ones((config.num_paths, 1)), features), axis=1
        )
        risk_neutrality = _risk_neutrality_audit(
            next_options - initial_options,
            state_instruments=instruments,
            eigenvalue_fraction=config.covariance_eigenvalue_fraction,
        )
        realized_factor_covariance = torch.cov(path_innovations.T)
        factor_identity = torch.eye(
            path_innovations.shape[-1],
            dtype=env.fdtype,
            device=env.device,
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "one_week_fixed_cash_lvsq_drift_test",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "num_source_states": len(latent_states),
        "num_unique_sampled_states": int(len(np.unique(sampled_indices_numpy))),
        "interpretation": (
            "Every strike is fixed in S0 cash units for the complete weekly "
            "transition. Spot is not sampled and no endpoint regrid is applied."
        ),
        "backbone_forward_parameters": forward_parameters.cpu().tolist(),
        "transition_seconds": transition_seconds,
        "path_steps_per_second": config.num_paths / transition_seconds,
        "maximum_volatility_equation_residual": maximum_volatility_residual,
        "maximum_drift_equation_residual": maximum_drift_residual,
        "minimum_raw_cap_margin": minimum_cap_margin,
        "risk_neutrality": risk_neutrality,
        "realized_factor_covariance": realized_factor_covariance.cpu().tolist(),
        "maximum_absolute_factor_covariance_error": float(
            torch.max(torch.abs(realized_factor_covariance - factor_identity)).cpu()
        ),
    }
    write_json_atomic(report, arguments.output)
    print(
        f"max_mean={risk_neutrality['maximum_absolute_mean_bps_spot']:.6f}bp "
        f"max_z={risk_neutrality['maximum_absolute_coordinate_z']:.4f} "
        f"p={risk_neutrality['unconditional_omnibus']['p_value']:.6f}",
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
