"""Fit one-step cutoff power-exponential OCE measures on the weekly DLV tower."""

from __future__ import annotations

import argparse
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.cutoff_powerexp_oce import (
    cutoff_powerexp_utility,
    solve_cutoff_powerexp_density,
)
from quantdata.dysanos.flex.direct_utility_frontier import DirectUtilityTowerPolicy
from quantdata.dysanos.flex.fixed_expiry_ar1 import (
    fit_fixed_expiry_ar1,
    torch_fixed_expiry_raw_log_loadings,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    IndependentDlvNoiseCalibration,
    TriangularFixedGridState,
    calibrate_independent_dlv_noise,
    initial_dlv_measure_switch_diagnostics,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_direct_utility_frontier import (
    PhysicalTowerDataset,
    _dataset_batch,
    _feature_statistics,
    _physical_tower_dataset,
    _weighted_martingale_audit,
)
from quantdata.dysanos.flex.spectral_trading import covariance_supported_coordinates
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=44)
    parser.add_argument("--cutoffs", default="0.25,0.5,1,2")
    parser.add_argument("--action-cap", type=float, default=65_536.0)
    parser.add_argument("--training-paths", type=int, default=1_000_000)
    parser.add_argument("--path-generation-batch-size", type=int, default=10_000)
    parser.add_argument("--training-batch-size", type=int, default=100_000)
    parser.add_argument("--antithetic-pairs-per-block", type=int, default=50)
    parser.add_argument("--innovation-rank", type=int, default=25)
    parser.add_argument(
        "--independent-dlv-option-variance-fraction",
        type=float,
        default=0.0,
        help=(
            "target fraction of each surface-sensitive option's instantaneous variance "
            "supplied by independent non-front log-DLV coordinate noise"
        ),
    )
    parser.add_argument(
        "--path-law",
        choices=("physical", "ar1", "ar1-heun", "achievable-heun", "analytic"),
        default="achievable-heun",
    )
    parser.add_argument("--analytic-num-substeps", type=int, default=10)
    parser.add_argument("--state-feature-mode", choices=("density", "full"), default="full")
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--minimum-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-patience", type=int, default=50)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--gradient-rms-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--action-penalty", type=float, default=1.0e-3)
    parser.add_argument(
        "--spectral-standard-error-multiple",
        type=float,
        default=2.0,
        help=(
            "retain covariance modes above this multiple of the independent-block "
            "variance-noise scale"
        ),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--monitoring-interval", type=int, default=10)
    parser.add_argument("--final-bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--training-seed", type=int, default=2_026_092_4)
    parser.add_argument("--model-seed", type=int, default=2_026_092_5)
    return parser


def _positive_grid(value: str, *, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{label} must be comma-separated finite-positive values")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _validate_arguments(arguments: argparse.Namespace) -> tuple[float, ...]:
    cutoffs = _positive_grid(arguments.cutoffs, label="cutoffs")
    paths_per_block = 2 * arguments.antithetic_pairs_per_block
    positive_integers = (
        arguments.training_paths,
        arguments.path_generation_batch_size,
        arguments.training_batch_size,
        arguments.antithetic_pairs_per_block,
        arguments.innovation_rank,
        arguments.analytic_num_substeps,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.epochs,
        arguments.minimum_epochs,
        arguments.monitoring_interval,
        arguments.final_bootstrap_samples,
    )
    if min(positive_integers) <= 0:
        raise ValueError("cutoff OCE integer controls must be positive")
    if arguments.minimum_epochs > arguments.epochs:
        raise ValueError("minimum epochs cannot exceed maximum epochs")
    if arguments.training_paths % paths_per_block:
        raise ValueError("training paths must contain complete independent blocks")
    if arguments.path_generation_batch_size % paths_per_block or (
        arguments.training_paths % arguments.path_generation_batch_size
    ):
        raise ValueError("generation batches must contain complete blocks and divide paths")
    finite_positive = (
        arguments.action_cap,
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.gradient_rms_tolerance,
        arguments.gradient_clip_norm,
        arguments.spectral_standard_error_multiple,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
        raise ValueError("cutoff OCE scalar controls must be finite-positive")
    if (
        arguments.minimum_learning_rate > arguments.learning_rate
        or arguments.lr_patience < 0
        or not math.isfinite(arguments.lr_decay)
        or not 0.0 < arguments.lr_decay < 1.0
        or not math.isfinite(arguments.action_penalty)
        or arguments.action_penalty < 0.0
    ):
        raise ValueError("cutoff OCE optimizer controls are invalid")
    if not 0.0 <= arguments.independent_dlv_option_variance_fraction < 1.0:
        raise ValueError("independent DLV option variance fraction must lie in [0, 1)")
    return cutoffs


def _json_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _scaled_trading_dataset(
    dataset: PhysicalTowerDataset,
) -> tuple[PhysicalTowerDataset, tuple[torch.Tensor, ...]]:
    """Append spot and apply an invertible RMS scaling at every trade week."""

    scaled: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for trade_week, option_returns in enumerate(dataset.returns_by_time):
        combined = torch.cat(
            (option_returns, dataset.spot_returns[:, trade_week : trade_week + 1]),
            dim=1,
        )
        rms = torch.sqrt(torch.mean(combined.double().square(), dim=0)).clamp_min(1.0e-8)
        scaled.append(combined / rms.to(dtype=combined.dtype)[None])
        scales.append(rms)
    diagnostics = dict(dataset.diagnostics)
    diagnostics.update(
        {
            "trading_coordinates": (
                "all live fixed-cash option returns plus contemporaneous spot return"
            ),
            "coordinate_preconditioner": (
                "invertible per-coordinate population RMS scaling on the fixed path set"
            ),
            "num_trading_coordinates": sum(value.shape[1] for value in scaled),
        }
    )
    return (
        PhysicalTowerDataset(
            features_by_time=dataset.features_by_time,
            returns_by_time=tuple(scaled),
            spot_returns=dataset.spot_returns,
            paths_per_independent_block=dataset.paths_per_independent_block,
            diagnostics=diagnostics,
        ),
        tuple(scales),
    )


def _full_policy_values(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    cutoffs: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    pnl_parts: list[torch.Tensor] = []
    action_norm_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, dataset.num_paths, batch_size):
            stop = min(start + batch_size, dataset.num_paths)
            indices = torch.arange(start, stop, device=policy.notionals.device)
            features, returns = _dataset_batch(dataset, indices, device=policy.notionals.device)
            forward = policy(features, returns)
            pnl_parts.append(forward.leveraged_pnl)
            for trade_week, norms in enumerate(forward.action_l1_by_time):
                action_norm_parts[trade_week].append(norms)
        pnl = torch.cat(pnl_parts, dim=1)
        density = solve_cutoff_powerexp_density(pnl, cutoffs)
        objective = (
            torch.mean(cutoff_powerexp_utility(density.shifted_pnl, cutoffs), dim=1)
            - density.translation
        )
        effective_sample_size = dataset.num_paths**2 / torch.sum(
            density.weights.square(), dim=1
        )
    return {
        "pnl": pnl,
        "translation": density.translation,
        "weights": density.weights,
        "objective": objective,
        "effective_sample_size": effective_sample_size,
        "maximum_weight": torch.amax(density.weights, dim=1),
        "active_fraction": torch.mean((density.weights > 0.0).float(), dim=1),
        "mass_before_boundary_subgradient": density.mass_before_boundary_subgradient,
        "boundary_subgradient_weight": density.boundary_subgradient_weight,
        "action_l1_by_time": tuple(
            torch.cat(parts, dim=1) for parts in action_norm_parts
        ),
    }


def _gradient_statistics(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    weights: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, float | int]]:
    policy.zero_grad(set_to_none=True)
    for start in range(0, dataset.num_paths, batch_size):
        stop = min(start + batch_size, dataset.num_paths)
        indices = torch.arange(start, stop, device=policy.notionals.device)
        features, returns = _dataset_batch(dataset, indices, device=policy.notionals.device)
        forward = policy(features, returns)
        loss = -torch.sum(
            weights[:, start:stop].detach() * forward.leveraged_pnl
        ) / (dataset.num_paths * len(policy.notionals))
        loss.backward()
    squared_norm = torch.zeros_like(policy.notionals)
    maximum_absolute = torch.zeros_like(policy.notionals)
    parameters_per_head = 0
    for parameter in policy.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.shape[0] != len(policy.notionals):
            raise RuntimeError("cutoff OCE parameters must retain an independent head axis")
        flat = parameter.grad.reshape(len(policy.notionals), -1)
        squared_norm += torch.sum(flat.square(), dim=1)
        maximum_absolute = torch.maximum(maximum_absolute, torch.amax(torch.abs(flat), dim=1))
        parameters_per_head += flat.shape[1]
    result = [
        {
            "parameters_per_head": parameters_per_head,
            "l2_norm": float(torch.sqrt(squared_norm[head])),
            "rms_per_parameter": float(torch.sqrt(squared_norm[head] / parameters_per_head)),
            "maximum_absolute_coordinate": float(maximum_absolute[head]),
        }
        for head in range(len(policy.notionals))
    ]
    policy.zero_grad(set_to_none=True)
    return result


def _action_summaries(
    values: dict[str, Any],
    *,
    action_cap: float,
) -> list[list[dict[str, float | int]]]:
    summaries: list[list[dict[str, float | int]]] = []
    for head in range(values["pnl"].shape[0]):
        head_summaries: list[dict[str, float | int]] = []
        for trade_week, norms in enumerate(values["action_l1_by_time"]):
            head_norms = norms[head]
            quantiles = torch.quantile(
                head_norms,
                torch.tensor((0.5, 0.9, 0.99), dtype=head_norms.dtype, device=head_norms.device),
            )
            head_summaries.append(
                {
                    "trade_week": trade_week,
                    "median": float(quantiles[0]),
                    "q90": float(quantiles[1]),
                    "q99": float(quantiles[2]),
                    "maximum": float(torch.max(head_norms)),
                    "fraction_at_action_cap": float(
                        torch.mean((head_norms >= action_cap * (1.0 - 1.0e-5)).float())
                    ),
                }
            )
        summaries.append(head_summaries)
    return summaries


def _checkpoint_payload(
    *,
    arguments: argparse.Namespace,
    cutoffs: Sequence[float],
    policy: DirectUtilityTowerPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "experiment_kind": "weekly_spectral_cutoff_powerexp_cash_oce",
        "arguments": _json_arguments(arguments),
        "cutoffs": list(cutoffs),
        "epoch": epoch,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }


def run(arguments: argparse.Namespace) -> None:
    """Generate the common weekly paths and fit all cutoff heads jointly."""

    cutoffs = _validate_arguments(arguments)
    env = BaseEnvironment(precision=64, device=arguments.device, debug=False)
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("cutoff OCE requires the fitted fixed cash grid")
    if not 0 <= arguments.state_index < len(latent_states):
        raise ValueError("state index lies outside the fitted static state table")
    independent_noise: IndependentDlvNoiseCalibration | None = None
    measure_switch_diagnostics: dict[str, float | int] | None = None
    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        fixed_expiry_fit = fit_fixed_expiry_ar1(
            all_raw.cpu().double().numpy(),
            dates,
            num_inner_strikes=geometry.num_inner_strikes,
            rank=arguments.innovation_rank,
            training_end=date(2024, 12, 31),
        )
        raw_log_loadings = torch_fixed_expiry_raw_log_loadings(
            fixed_expiry_fit,
            dtype=env.fdtype,
            device=env.device,
        )
        ar1_intercept = torch.as_tensor(
            fixed_expiry_fit.conditional_intercept,
            dtype=env.fdtype,
            device=env.device,
        )
        ar1_slope = torch.as_tensor(
            fixed_expiry_fit.conditional_slope,
            dtype=env.fdtype,
            device=env.device,
        )
        selected_raw = all_raw[arguments.state_index : arguments.state_index + 1]
        parameters = bound_native_raw(selected_raw, geometry=geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        fixed_cash_strikes = torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=env.fdtype,
            device=env.device,
        )
        if arguments.independent_dlv_option_variance_fraction > 0.0:
            initial_state = TriangularFixedGridState(
                support=surface.support_strikes[:, 0],
                densities=surface.densities,
                dlvsqs=parameters.dlvsqs,
            )
            initial_option_signs = torch.where(
                fixed_cash_strikes[None] <= 1.0,
                1.0,
                -1.0,
            )
            independent_noise = calibrate_independent_dlv_noise(
                model.sanos_decoder,
                initial_state,
                raw_log_loadings=raw_log_loadings,
                minimum_dlvsq=geometry.min_dlvsq,
                expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
                fixed_cash_strikes=fixed_cash_strikes,
                target_option_variance_fraction=(
                    arguments.independent_dlv_option_variance_fraction
                ),
                option_signs=initial_option_signs,
            )
            raw_log_loadings = independent_noise.augmented_raw_log_loadings
            measure_switch_diagnostics = initial_dlv_measure_switch_diagnostics(
                model.sanos_decoder,
                initial_state,
                raw_log_loadings=raw_log_loadings,
                conditional_intercept=ar1_intercept,
                conditional_slope=ar1_slope,
                minimum_dlvsq=geometry.min_dlvsq,
                expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
                calendar_time_increment_y=5.0 / DAYS_PER_YEAR,
                fixed_cash_strikes=fixed_cash_strikes,
                option_signs=initial_option_signs,
            )
            print(
                "independent DLV calibration: "
                f"target={arguments.independent_dlv_option_variance_fraction:.3%}, "
                f"median={independent_noise.diagnostics['achieved_fraction_median']:.3%}, "
                f"range=[{independent_noise.diagnostics['achieved_fraction_minimum']:.3%}, "
                f"{independent_noise.diagnostics['achieved_fraction_maximum']:.3%}], "
                f"factors={raw_log_loadings.shape[-1]}",
                flush=True,
            )
            print(
                "initial exact measure switch: "
                f"DLV rank={measure_switch_diagnostics['dlv_loading_numerical_rank']}/"
                f"{measure_switch_diagnostics['num_dlv_coordinates']}, "
                f"option rank={measure_switch_diagnostics['option_loading_numerical_rank']}/"
                f"{measure_switch_diagnostics['num_surface_sensitive_option_coordinates']}, "
                f"ESS={measure_switch_diagnostics['one_week_lognormal_ess_fraction']:.3g}, "
                f"|a|1={measure_switch_diagnostics['minimum_norm_traded_action_l1']:.3g}",
                flush=True,
            )
        raw_dataset = _physical_tower_dataset(
            model=model,
            geometry=geometry,
            surface=surface,
            parameters=parameters,
            raw_log_loadings=raw_log_loadings,
            fixed_cash_strikes=fixed_cash_strikes,
            num_paths=arguments.training_paths,
            path_batch_size=arguments.path_generation_batch_size,
            pairs_per_block=arguments.antithetic_pairs_per_block,
            seed=arguments.training_seed,
            state_feature_mode=arguments.state_feature_mode,
            path_law=arguments.path_law,
            analytic_num_substeps=arguments.analytic_num_substeps,
            dataset_storage="cpu",
            ar1_conditional_intercept=ar1_intercept,
            ar1_conditional_slope=ar1_slope,
            bs_y_vol=0.0,
        )
        raw_dataset.diagnostics["independent_dlv_noise"] = (
            None if independent_noise is None else independent_noise.diagnostics
        )
    scaled_dataset, return_rms = _scaled_trading_dataset(raw_dataset)
    spectral_returns, spectral_transform = covariance_supported_coordinates(
        scaled_dataset.returns_by_time,
        num_independent_blocks=(
            scaled_dataset.num_paths // scaled_dataset.paths_per_independent_block
        ),
        standard_error_multiple=arguments.spectral_standard_error_multiple,
        computation_device=env.device,
        batch_size=arguments.training_batch_size,
    )
    spectral_diagnostics = dict(scaled_dataset.diagnostics)
    spectral_diagnostics.update(
        {
            "trading_coordinates": (
                "retained covariance eigenmodes of RMS-scaled live option and spot returns"
            ),
            "spectral_variance_floor": (
                "multiple*sqrt(2/(independent_blocks-1))*average coordinate variance"
            ),
            "spectral_standard_error_multiple": (
                arguments.spectral_standard_error_multiple
            ),
            "num_trading_coordinates": sum(
                values.shape[1] for values in spectral_returns
            ),
        }
    )
    trading_dataset = PhysicalTowerDataset(
        features_by_time=scaled_dataset.features_by_time,
        returns_by_time=spectral_returns,
        spot_returns=scaled_dataset.spot_returns,
        paths_per_independent_block=scaled_dataset.paths_per_independent_block,
        diagnostics=spectral_diagnostics,
    )
    uniform_audit = _weighted_martingale_audit(
        raw_dataset,
        torch.ones((1, raw_dataset.num_paths), dtype=torch.float32),
        bootstrap_samples=arguments.final_bootstrap_samples,
        bootstrap_seed=arguments.training_seed + 500_000,
    )
    print(
        "uniform P audit: "
        f"p={uniform_audit['heads'][0]['weighted_option_zero_drift']['combined_p_value']:.6g}, "
        f"max|t|={uniform_audit['heads'][0]['weighted_option_zero_drift']['maximum_absolute_t']:.6g}",
        flush=True,
    )

    feature_means, feature_scales = _feature_statistics(trading_dataset)
    feature_means = tuple(value.to(env.device) for value in feature_means)
    feature_scales = tuple(value.to(env.device) for value in feature_scales)
    policy = DirectUtilityTowerPolicy(
        notionals=(arguments.action_cap,) * len(cutoffs),
        feature_means=feature_means,
        feature_scales=feature_scales,
        output_sizes=[value.shape[1] for value in trading_dataset.returns_by_time],
        hidden_width=arguments.hidden_width,
        hidden_layers=arguments.hidden_layers,
        seed=arguments.model_seed,
    )
    cutoff_tensor = torch.as_tensor(
        cutoffs,
        dtype=policy.notionals.dtype,
        device=policy.notionals.device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    history: list[dict[str, object]] = []
    start_epoch = 1
    if arguments.resume:
        payload = torch.load(arguments.checkpoint, map_location=env.device, weights_only=False)
        if payload.get("schema_version") != 3 or payload.get("cutoffs") != list(cutoffs):
            raise ValueError("cutoff OCE resume checkpoint is incompatible")
        checkpoint_fraction = payload.get("arguments", {}).get(
            "independent_dlv_option_variance_fraction",
            0.0,
        )
        if checkpoint_fraction != arguments.independent_dlv_option_variance_fraction:
            raise ValueError("cutoff OCE checkpoint has a different independent DLV noise level")
        policy.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = arguments.learning_rate
        history = payload["history"]
        start_epoch = int(payload["epoch"]) + 1

    best_mean_objective = -math.inf
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = "maximum epochs reached"
    training_started = perf_counter()
    for epoch in range(start_epoch, arguments.epochs + 1):
        values = _full_policy_values(
            policy,
            trading_dataset,
            cutoff_tensor,
            batch_size=arguments.training_batch_size,
        )
        weights = values["weights"]
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for start in range(0, trading_dataset.num_paths, arguments.training_batch_size):
            stop = min(start + arguments.training_batch_size, trading_dataset.num_paths)
            indices = torch.arange(start, stop, device=env.device)
            features, returns = _dataset_batch(trading_dataset, indices, device=env.device)
            forward = policy(features, returns)
            surrogate = -torch.sum(
                weights[:, start:stop].detach() * forward.leveraged_pnl,
                dim=1,
            ) / trading_dataset.num_paths
            batch_fraction = (stop - start) / trading_dataset.num_paths
            loss_by_head = surrogate + (
                arguments.action_penalty
                * batch_fraction
                * forward.action_penalty_by_head
            )
            loss = torch.mean(loss_by_head)
            loss.backward()
            epoch_loss += float(loss.detach())
        raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), arguments.gradient_clip_norm
        )
        optimizer.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "pre_update_objective": values["objective"].cpu().tolist(),
            "surrogate_loss": epoch_loss,
            "pre_clip_gradient_norm": float(raw_gradient_norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % arguments.monitoring_interval == 0 or epoch == arguments.epochs:
            monitored = _full_policy_values(
                policy,
                trading_dataset,
                cutoff_tensor,
                batch_size=arguments.training_batch_size,
            )
            gradients = _gradient_statistics(
                policy,
                trading_dataset,
                monitored["weights"],
                batch_size=arguments.training_batch_size,
            )
            record.update(
                {
                    "objective": monitored["objective"].cpu().tolist(),
                    "effective_sample_size": monitored["effective_sample_size"].cpu().tolist(),
                    "active_fraction": monitored["active_fraction"].cpu().tolist(),
                    "maximum_weight": monitored["maximum_weight"].cpu().tolist(),
                    "gradient": gradients,
                }
            )
            mean_objective = float(torch.mean(monitored["objective"]))
            if mean_objective > best_mean_objective * (1.0 + 1.0e-6) + 1.0e-10:
                best_mean_objective = mean_objective
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += arguments.monitoring_interval
            if (
                arguments.lr_patience
                and epochs_without_improvement >= arguments.lr_patience
                and optimizer.param_groups[0]["lr"] > arguments.minimum_learning_rate
            ):
                before = optimizer.param_groups[0]["lr"]
                after = max(arguments.minimum_learning_rate, before * arguments.lr_decay)
                for group in optimizer.param_groups:
                    group["lr"] = after
                record["learning_rate_decay"] = {"before": before, "after": after}
                epochs_without_improvement = 0
            print(
                f"epoch {epoch}: "
                + ", ".join(
                    f"x0={cutoff:g}:oce={objective:.6g},"
                    f"ESS={ess / trading_dataset.num_paths:.3g},"
                    f"active={active:.3g},grad={gradient['rms_per_parameter']:.3g}"
                    for cutoff, objective, ess, active, gradient in zip(
                        cutoffs,
                        monitored["objective"].cpu().tolist(),
                        monitored["effective_sample_size"].cpu().tolist(),
                        monitored["active_fraction"].cpu().tolist(),
                        gradients,
                        strict=True,
                    )
                ),
                flush=True,
            )
            if epoch >= arguments.minimum_epochs and max(
                item["rms_per_parameter"] for item in gradients
            ) <= arguments.gradient_rms_tolerance:
                stopped_early = True
                stop_reason = "all cutoff-head gradient RMS values reached tolerance"
        history.append(record)
        if epoch % arguments.monitoring_interval == 0 or epoch == arguments.epochs:
            _atomic_torch_save(
                _checkpoint_payload(
                    arguments=arguments,
                    cutoffs=cutoffs,
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                ),
                arguments.checkpoint,
            )
            write_json_atomic(
                {
                    "schema_version": 2,
                    "experiment_kind": "weekly_spectral_cutoff_powerexp_cash_oce",
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "cutoffs": list(cutoffs),
                    "completed_epoch": epoch,
                    "history": history,
                },
                arguments.output,
            )
        if stopped_early:
            break

    final_values = _full_policy_values(
        policy,
        trading_dataset,
        cutoff_tensor,
        batch_size=arguments.training_batch_size,
    )
    final_gradients = _gradient_statistics(
        policy,
        trading_dataset,
        final_values["weights"],
        batch_size=arguments.training_batch_size,
    )
    final_audit = _weighted_martingale_audit(
        raw_dataset,
        final_values["weights"].detach().cpu(),
        bootstrap_samples=arguments.final_bootstrap_samples,
        bootstrap_seed=arguments.training_seed + 1_000_000,
    )
    final_factor_audit = _weighted_martingale_audit(
        trading_dataset,
        final_values["weights"].detach().cpu(),
        bootstrap_samples=arguments.final_bootstrap_samples,
        bootstrap_seed=arguments.training_seed + 1_500_000,
    )
    action_summaries = _action_summaries(final_values, action_cap=arguments.action_cap)
    heads: list[dict[str, object]] = []
    for head, cutoff in enumerate(cutoffs):
        pnl = final_values["pnl"][head]
        heads.append(
            {
                "cutoff": cutoff,
                "cash_translation": float(final_values["translation"][head]),
                "oce": float(final_values["objective"][head]),
                "mean_strategy_pnl_under_p": float(torch.mean(pnl)),
                "strategy_pnl_standard_deviation_under_p": float(torch.std(pnl, correction=1)),
                "density_effective_sample_size": float(
                    final_values["effective_sample_size"][head]
                ),
                "density_effective_sample_fraction": float(
                    final_values["effective_sample_size"][head] / raw_dataset.num_paths
                ),
                "density_active_fraction": float(final_values["active_fraction"][head]),
                "maximum_density_weight": float(final_values["maximum_weight"][head]),
                "density_mass_before_boundary_subgradient": float(
                    final_values["mass_before_boundary_subgradient"][head]
                ),
                "boundary_subgradient_weight": float(
                    final_values["boundary_subgradient_weight"][head]
                ),
                "objective_gradient": final_gradients[head],
                "action_l1_by_trade_week": action_summaries[head],
                "martingale_audit": final_audit["heads"][head],
                "retained_factor_martingale_audit": final_factor_audit["heads"][head],
            }
        )
    report = {
        "schema_version": 3,
        "experiment_kind": "weekly_spectral_cutoff_powerexp_cash_oce",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": (
            "one joint cash OCE per cutoff: sup_{theta,y} E[u(min(y+pi_theta,x0))]-y; "
            "the empirical density is a selected nonnegative cutoff subgradient"
        ),
        "absolute_continuity": (
            "Q is absolutely continuous with respect to the fixed empirical P path law; "
            "zero density is permitted above the utility cutoff"
        ),
        "source_state_index": arguments.state_index,
        "source_state_date": dates[arguments.state_index].isoformat(),
        "sources": sources,
        "arguments": _json_arguments(arguments),
        "path_simulation": (
            "X-only fixed cash grid; initial 1W/2W/3W/4W tower; four weekly rolls; "
            "spot drawn from the current front density; no replacement options; "
            "the last roll only draws terminal spot and settles the original 4W claims; "
            "optional independent noise acts on non-front log-DLV coordinates before pricing"
        ),
        "return_definition": (
            "2,000 one-step same-claim ITM option returns (800+600+400+200), "
            "plus four contemporaneous spot returns"
        ),
        "strategy_class": {
            "state_features": raw_dataset.diagnostics["state_feature_definition"],
            "hidden_width": arguments.hidden_width,
            "hidden_layers": arguments.hidden_layers,
            "total_parameters": sum(value.numel() for value in policy.parameters()),
            "coordinate_scaling": trading_dataset.diagnostics["coordinate_preconditioner"],
            "action_cap": arguments.action_cap,
            "action_coordinates": (
                "whitened retained covariance modes; the cap applies to factor exposures"
            ),
        },
        "spectral_transform": {
            "definition": (
                "uncentered RMS-scaled returns projected onto retained covariance "
                "eigenvectors and divided by square-root eigenvalue"
            ),
            "diagnostics_by_trade_week": spectral_transform.diagnostics,
        },
        "return_rms_by_trade_week": [value.tolist() for value in return_rms],
        "training_dataset": raw_dataset.diagnostics,
        "independent_dlv_noise_calibration": (
            None if independent_noise is None else independent_noise.diagnostics
        ),
        "initial_exact_measure_switch": measure_switch_diagnostics,
        "uniform_p_martingale_audit": uniform_audit,
        "completed_epochs": history[-1]["epoch"] if history else 0,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "heads": heads,
        "history": history,
        "speed": {"training_and_audit_seconds": perf_counter() - training_started},
    }
    write_json_atomic(report, arguments.output)
    _atomic_torch_save(
        {
            **_checkpoint_payload(
                arguments=arguments,
                cutoffs=cutoffs,
                policy=policy,
                optimizer=optimizer,
                epoch=int(report["completed_epochs"]),
                history=history,
            ),
            "empirical_q_weights": final_values["weights"].detach().cpu(),
            "spectral_eigenvectors_by_time": spectral_transform.eigenvectors_by_time,
            "spectral_inverse_sqrt_eigenvalues_by_time": (
                spectral_transform.inverse_sqrt_eigenvalues_by_time
            ),
            "final_report": report,
        },
        arguments.checkpoint,
    )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    arguments = _parser().parse_args()
    lock = FileLock(f"{arguments.checkpoint}.lock")
    with lock:
        run(arguments)


if __name__ == "__main__":
    main()
