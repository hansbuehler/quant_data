"""Paper-geometry CRC population for restricted dynamic-arbitrage audits."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.version import version
from filelock import FileLock
from scipy.special import ndtri
from scipy.stats import poisson

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sdlv.bates_crc.gambara_teichmann_inverse_train import (
    _load_grid,
    apply_reset_innovations,
)
from quantdata.sdlv.bates_crc.gambara_teichmann_inverse_validate import (
    load_inverse_network,
    verify_training_provenance,
)
from quantdata.sdlv.bates_crc.gambara_teichmann_torch import (
    price_calls_generalized_bates_batched,
)

NORMALIZED_MONEYNESS = np.asarray(
    (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5),
    dtype=np.float64,
)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CrcPopulationConfig:
    """Numerical geometry of the restricted CRC population."""

    path_count: int = 100_000
    batch_size: int = 1_000
    horizon_days: int = 20
    steps_per_day: int = 8
    option_expiry_days: tuple[int, ...] = (5, 10, 15, 20)
    options_per_expiry: int = 11
    n_quad: int = 240
    u_max: float = 600.0
    implied_vol_iterations: int = 36
    seed: int = 84_113

    def __post_init__(self) -> None:
        if (
            self.path_count < 4
            or self.path_count % 4 != 0
            or self.batch_size < 4
            or self.batch_size % 2 != 0
            or self.path_count % self.batch_size != 0
        ):
            raise ValueError("paths must preserve pairs/folds and batch size must divide paths")
        if self.horizon_days != 20 or self.steps_per_day != 8:
            raise ValueError("the paper comparison requires 20 days and eight steps per day")
        if self.option_expiry_days != (5, 10, 15, 20) or self.options_per_expiry != 11:
            raise ValueError("the restricted comparison requires the fixed 4-by-11 option grid")
        if self.n_quad <= 0 or self.u_max <= 0.0 or self.implied_vol_iterations <= 0:
            raise ValueError("pricing geometry must be positive")

    @property
    def instrument_count(self) -> int:
        return 1 + len(self.option_expiry_days) * self.options_per_expiry


@dataclass(frozen=True)
class CrcRestrictedUniverse:
    expiry_days: np.ndarray
    strikes: np.ndarray
    is_put: np.ndarray
    labels: tuple[str, ...]


def _black_calls(
    strikes: torch.Tensor,
    expiries: torch.Tensor,
    volatility: torch.Tensor,
) -> torch.Tensor:
    total = volatility * torch.sqrt(expiries)
    d1 = -torch.log(strikes) / total + 0.5 * total
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.cdf(d1) - strikes * normal.cdf(d1 - total)


def _implied_volatilities(
    normalized_prices: torch.Tensor,
    moneyness: torch.Tensor,
    expiry: float,
    *,
    iterations: int,
) -> tuple[torch.Tensor, int]:
    """Invert zero-rate Black calls after bounded Fourier-tail clipping."""

    intrinsic = torch.clamp(1.0 - moneyness, min=0.0)
    invalid = (normalized_prices <= intrinsic) | (normalized_prices >= 1.0)
    prices = torch.minimum(
        torch.maximum(normalized_prices, intrinsic + 1.0e-8),
        torch.full_like(normalized_prices, 1.0 - 1.0e-8),
    )
    lower = torch.full_like(prices, 1.0e-5)
    upper = torch.full_like(prices, 5.0)
    expiries = torch.full_like(prices, expiry)
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        below = _black_calls(moneyness, expiries, middle) < prices
        lower = torch.where(below, middle, lower)
        upper = torch.where(below, upper, middle)
    return 0.5 * (lower + upper), int(torch.count_nonzero(invalid).cpu())


def _universe(
    *,
    variance: float,
    theta: float,
    xi: float,
    rho: float,
    jump_means: tuple[float, ...],
    jump_stds: tuple[float, ...],
    kappa: float,
    intensity: float,
    boundaries: tuple[float, float, float, float],
    config: CrcPopulationConfig,
) -> CrcRestrictedUniverse:
    dtype = torch.float64
    expiry_days = np.asarray(config.option_expiry_days, dtype=np.int64)
    atms = []
    for days in expiry_days:
        price = price_calls_generalized_bates_batched(
            log_spot=torch.zeros(1, dtype=dtype),
            variance=torch.full((1,), variance, dtype=dtype),
            theta=torch.full((1,), theta, dtype=dtype),
            xi=torch.full((1,), xi, dtype=dtype),
            rho=torch.full((1,), rho, dtype=dtype),
            jump_means=torch.tensor((jump_means,), dtype=dtype),
            jump_stds=torch.tensor((jump_stds,), dtype=dtype),
            strikes=torch.ones(1, dtype=dtype),
            expiry=float(days) / DAYS_PER_YEAR,
            kappa=kappa,
            intensity=intensity,
            boundaries=boundaries,
            n_quad=config.n_quad,
            u_max=config.u_max,
        )
        atms.append(float(price[0, 0]))
    probabilities = np.clip(0.5 * (np.asarray(atms) + 1.0), 0.5, 1.0 - 1.0e-14)
    total_volatility = 2.0 * ndtri(probabilities)
    epsilon_volatility = total_volatility[0] / math.sqrt(
        expiry_days[0] / DAYS_PER_YEAR
    )
    nms_scale = np.sqrt(
        (
            epsilon_volatility**2 * float(NMS.EPS)
            + np.square(total_volatility)
        )
        / (1.0 + float(NMS.EPS))
    )
    strikes = np.exp(nms_scale[:, None] * NORMALIZED_MONEYNESS[None])
    is_put = np.broadcast_to(NORMALIZED_MONEYNESS[None] < 0.0, strikes.shape).copy()
    labels = tuple(
        f"{expiry}D NMS {score:g} {'put' if score < 0.0 else 'call'}"
        for expiry in expiry_days
        for score in NORMALIZED_MONEYNESS
    )
    return CrcRestrictedUniverse(
        expiry_days=expiry_days,
        strikes=strikes,
        is_put=is_put,
        labels=("spot", *labels),
    )


def _active_mask(config: CrcPopulationConfig) -> np.ndarray:
    mask = np.zeros((config.horizon_days, config.instrument_count), dtype=bool)
    mask[:, 0] = True
    for index, expiry in enumerate(config.option_expiry_days):
        start = 1 + index * config.options_per_expiry
        mask[:expiry, start : start + config.options_per_expiry] = True
    return mask


def _option_values(
    *,
    log_spot: torch.Tensor,
    variance: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
    jump_means: torch.Tensor,
    jump_stds: torch.Tensor,
    universe: CrcRestrictedUniverse,
    day_index: int,
    jump_start: float,
    kappa: float,
    intensity: float,
    boundaries: tuple[float, float, float, float],
    config: CrcPopulationConfig,
) -> torch.Tensor:
    spot = torch.exp(log_spot)
    values = []
    for expiry_index, expiry_day in enumerate(universe.expiry_days):
        remaining_days = int(expiry_day) - day_index
        strikes = torch.as_tensor(
            universe.strikes[expiry_index],
            dtype=log_spot.dtype,
            device=log_spot.device,
        )
        if remaining_days == 0:
            calls = torch.clamp(spot[:, None] - strikes[None], min=0.0)
        elif remaining_days < 0:
            calls = torch.zeros(
                (len(spot), len(strikes)), dtype=log_spot.dtype, device=log_spot.device
            )
        else:
            calls = price_calls_generalized_bates_batched(
                log_spot=log_spot,
                variance=variance,
                theta=theta,
                xi=xi,
                rho=rho,
                jump_means=jump_means,
                jump_stds=jump_stds,
                strikes=strikes,
                expiry=remaining_days / DAYS_PER_YEAR,
                kappa=kappa,
                intensity=intensity,
                boundaries=boundaries,
                jump_start=jump_start,
                n_quad=config.n_quad,
                u_max=config.u_max,
            )
        puts = calls - spot[:, None] + strikes[None]
        selector = torch.as_tensor(
            universe.is_put[expiry_index], dtype=torch.bool, device=log_spot.device
        )
        values.append(torch.where(selector[None], puts, calls))
    return torch.stack(values, dim=1)


def _target_surface(
    *,
    log_spot: torch.Tensor,
    variance: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
    jump_means: torch.Tensor,
    jump_stds: torch.Tensor,
    maturities: np.ndarray,
    moneyness: np.ndarray,
    kappa: float,
    intensity: float,
    boundaries: tuple[float, float, float, float],
    config: CrcPopulationConfig,
) -> tuple[torch.Tensor, int]:
    spot = torch.exp(log_spot)
    relative_strikes = torch.as_tensor(
        moneyness, dtype=log_spot.dtype, device=log_spot.device
    )
    surfaces = []
    clipped = 0
    for maturity in maturities:
        strikes = spot[:, None] * relative_strikes[None]
        calls = price_calls_generalized_bates_batched(
            log_spot=log_spot,
            variance=variance,
            theta=theta,
            xi=xi,
            rho=rho,
            jump_means=jump_means,
            jump_stds=jump_stds,
            strikes=strikes,
            expiry=float(maturity),
            kappa=kappa,
            intensity=intensity,
            boundaries=boundaries,
            jump_start=1.0 / DAYS_PER_YEAR,
            n_quad=config.n_quad,
            u_max=config.u_max,
        )
        volatility, count = _implied_volatilities(
            calls / spot[:, None],
            relative_strikes[None].expand(len(spot), -1),
            float(maturity),
            iterations=config.implied_vol_iterations,
        )
        surfaces.append(volatility)
        clipped += count
    return torch.cat(surfaces, dim=1), clipped


def _pair_randomness(
    config: CrcPopulationConfig,
    *,
    batch_index: int,
    intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate adjacent antithetic diffusion, jump, and reset innovations."""

    rng = np.random.default_rng(np.random.SeedSequence((config.seed, batch_index)))
    pair_count = config.batch_size // 2
    diffusion_base = rng.standard_normal(
        (config.horizon_days, config.steps_per_day, pair_count, 2)
    )
    diffusion = np.stack((diffusion_base, -diffusion_base), axis=3).reshape(
        config.horizon_days, config.steps_per_day, config.batch_size, 2
    )
    uniforms = np.clip(
        rng.random((config.horizon_days, config.steps_per_day, pair_count)),
        1.0e-12,
        1.0 - 1.0e-12,
    )
    poisson_mean = intensity / (DAYS_PER_YEAR * config.steps_per_day)
    count_left = poisson.ppf(uniforms, poisson_mean).astype(np.int64)
    count_right = poisson.ppf(1.0 - uniforms, poisson_mean).astype(np.int64)
    counts = np.stack((count_left, count_right), axis=3).reshape(
        config.horizon_days, config.steps_per_day, config.batch_size
    )
    jump_base = rng.standard_normal(
        (config.horizon_days, config.steps_per_day, pair_count)
    )
    jump_normals = np.stack((jump_base, -jump_base), axis=3).reshape(
        config.horizon_days, config.steps_per_day, config.batch_size
    )
    reset_base = rng.standard_normal((config.horizon_days, pair_count, 3))
    reset_normals = np.stack((reset_base, -reset_base), axis=2).reshape(
        config.horizon_days, config.batch_size, 3
    )
    return diffusion, counts, jump_normals, reset_normals


def _atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def generate_population(
    *,
    checkpoint: Path,
    dataset_directory: Path,
    target_fit_report: Path,
    output_directory: Path,
    config: CrcPopulationConfig,
    device: str,
) -> Path:
    """Generate resumable batches and combine them into one audit population."""

    output_directory.mkdir(parents=True, exist_ok=True)
    report = json.loads(target_fit_report.read_text(encoding="utf-8"))
    verify_training_provenance(
        checkpoint,
        dataset_directory=dataset_directory,
        target_fit_report=target_fit_report,
    )
    fit = report["target_fit"]
    reset_law = report["reset_innovation_law"]
    kappa = float(fit["kappa_fixed_from_history"])
    intensity = float(fit["intensity_fixed_from_history"])
    (
        maturities,
        moneyness,
        dataset_kappa,
        dataset_intensity,
        dataset_quadrature_nodes,
        dataset_quadrature_u_max,
    ) = _load_grid(dataset_directory)
    if not np.array_equal(maturities, np.asarray(fit["maturity_nodes"])):
        raise RuntimeError("CRC inverse grid does not match the fitted maturity nodes")
    if dataset_kappa != kappa or dataset_intensity != intensity:
        raise RuntimeError("CRC inverse data do not match fitted kappa/intensity")
    if (
        config.n_quad != dataset_quadrature_nodes
        or config.u_max != dataset_quadrature_u_max
    ):
        raise RuntimeError("CRC population pricing must match the inverse-data quadrature")
    boundaries = tuple(float(value) for value in fit["boundaries"])
    model = load_inverse_network(checkpoint, device=device)
    dtype = model.fdtype
    torch_device = model.device
    universe = _universe(
        variance=float(fit["variance"]),
        theta=float(fit["theta"]),
        xi=float(fit["xi"]),
        rho=float(fit["rho"]),
        jump_means=tuple(float(value) for value in fit["jump_means"]),
        jump_stds=tuple(float(value) for value in fit["jump_stds"]),
        kappa=kappa,
        intensity=intensity,
        boundaries=boundaries,
        config=config,
    )
    active_mask = _active_mask(config)
    innovation_mean = torch.as_tensor(
        reset_law["mean"], dtype=dtype, device=torch_device
    )
    innovation_root = torch.as_tensor(
        np.linalg.cholesky(np.asarray(reset_law["covariance"])),
        dtype=dtype,
        device=torch_device,
    )
    feature_count = 5 + 10 + 44 + 1 + 4
    progress_path = output_directory / "population-progress.jsonl"
    started = time.monotonic()
    clipped_total = 0
    reset_maximum = 0.0
    reset_squared_sum = 0.0
    reset_observations = 0
    batch_count = config.path_count // config.batch_size

    with FileLock(output_directory / "population.lock"), torch.no_grad():
        for batch_index in range(batch_count):
            start = batch_index * config.batch_size
            stop = start + config.batch_size
            destination = output_directory / f"population-{start:06d}-{stop:06d}.npz"
            if destination.exists():
                continue
            diffusion_np, counts_np, jump_normals_np, reset_normals_np = _pair_randomness(
                config, batch_index=batch_index, intensity=intensity
            )
            log_spot = torch.zeros(config.batch_size, dtype=dtype, device=torch_device)
            variance = torch.full(
                (config.batch_size,), float(fit["variance"]), dtype=dtype, device=torch_device
            )
            theta = torch.full(
                (config.batch_size,), float(fit["theta"]), dtype=dtype, device=torch_device
            )
            xi = torch.full(
                (config.batch_size,), float(fit["xi"]), dtype=dtype, device=torch_device
            )
            rho = torch.full(
                (config.batch_size,), float(fit["rho"]), dtype=dtype, device=torch_device
            )
            jump_means = torch.tensor(
                fit["jump_means"], dtype=dtype, device=torch_device
            )[None].expand(config.batch_size, -1).clone()
            jump_stds = torch.tensor(
                fit["jump_stds"], dtype=dtype, device=torch_device
            )[None].expand(config.batch_size, -1).clone()
            options = _option_values(
                log_spot=log_spot,
                variance=variance,
                theta=theta,
                xi=xi,
                rho=rho,
                jump_means=jump_means,
                jump_stds=jump_stds,
                universe=universe,
                day_index=0,
                jump_start=0.0,
                kappa=kappa,
                intensity=intensity,
                boundaries=boundaries,
                config=config,
            )
            features = np.empty(
                (config.batch_size, config.horizon_days, feature_count), dtype=np.float32
            )
            returns = np.empty(
                (config.batch_size, config.horizon_days, config.instrument_count),
                dtype=np.float64,
            )
            batch_clipped = 0
            batch_reset_maximum = 0.0
            batch_reset_squared_sum = 0.0
            batch_reset_observations = 0
            dt = 1.0 / (DAYS_PER_YEAR * config.steps_per_day)
            sqrt_dt = math.sqrt(dt)
            for day_index in range(config.horizon_days):
                spot = torch.exp(log_spot)
                live = torch.as_tensor(
                    universe.expiry_days > day_index,
                    dtype=torch.bool,
                    device=torch_device,
                )
                normalized = options / spot[:, None, None]
                log_options = torch.where(
                    live[None, :, None],
                    torch.log(torch.clamp(normalized, min=1.0e-12)),
                    torch.zeros_like(normalized),
                )
                time_to_expiry = torch.as_tensor(
                    np.maximum(universe.expiry_days - day_index, 0) / config.horizon_days,
                    dtype=dtype,
                    device=torch_device,
                )[None].expand(config.batch_size, -1)
                state_features = torch.cat(
                    (
                        torch.stack((log_spot, variance, theta, xi, rho), dim=1),
                        jump_means,
                        jump_stds,
                        log_options.reshape(config.batch_size, -1),
                        torch.full(
                            (config.batch_size, 1),
                            day_index / config.horizon_days,
                            dtype=dtype,
                            device=torch_device,
                        ),
                        time_to_expiry,
                    ),
                    dim=1,
                )
                features[:, day_index] = state_features.float().cpu().numpy()
                previous_spot = spot
                previous_options = options
                for substep in range(config.steps_per_day):
                    normals = torch.as_tensor(
                        diffusion_np[day_index, substep], dtype=dtype, device=torch_device
                    )
                    counts = torch.as_tensor(
                        counts_np[day_index, substep], dtype=dtype, device=torch_device
                    )
                    jump_normal = torch.as_tensor(
                        jump_normals_np[day_index, substep], dtype=dtype, device=torch_device
                    )
                    positive_variance = torch.clamp(variance, min=0.0)
                    root_variance_dt = torch.sqrt(positive_variance) * sqrt_dt
                    variance_normal = normals[:, 1]
                    spot_normal = (
                        rho * variance_normal
                        + torch.sqrt(torch.clamp(1.0 - rho.square(), min=0.0))
                        * normals[:, 0]
                    )
                    mean = jump_means[:, 0]
                    std = jump_stds[:, 0]
                    compensator = intensity * (torch.exp(mean + 0.5 * std.square()) - 1.0)
                    aggregate_jump = counts * mean + torch.sqrt(counts) * std * jump_normal
                    log_spot = (
                        log_spot
                        + (-0.5 * positive_variance - compensator) * dt
                        + root_variance_dt * spot_normal
                        + aggregate_jump
                    )
                    variance = torch.clamp(
                        variance
                        + kappa * (theta - positive_variance) * dt
                        + xi * root_variance_dt * variance_normal,
                        min=0.0,
                    )

                old_options = _option_values(
                    log_spot=log_spot,
                    variance=variance,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    jump_means=jump_means,
                    jump_stds=jump_stds,
                    universe=universe,
                    day_index=day_index + 1,
                    jump_start=1.0 / DAYS_PER_YEAR,
                    kappa=kappa,
                    intensity=intensity,
                    boundaries=boundaries,
                    config=config,
                )
                target_surface, clipped = _target_surface(
                    log_spot=log_spot,
                    variance=variance,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    jump_means=jump_means,
                    jump_stds=jump_stds,
                    maturities=maturities,
                    moneyness=moneyness,
                    kappa=kappa,
                    intensity=intensity,
                    boundaries=boundaries,
                    config=config,
                )
                batch_clipped += clipped
                reset_normal = torch.as_tensor(
                    reset_normals_np[day_index], dtype=dtype, device=torch_device
                )
                innovations = innovation_mean + reset_normal @ innovation_root.T
                old_heston = torch.stack((variance, theta, xi, rho), dim=1)
                new_heston = apply_reset_innovations(
                    old_heston,
                    innovations,
                    kappa=kappa,
                    relative_change_cap=float(reset_law["relative_change_cap"]),
                )
                candidate = model.jump_parameters(new_heston, target_surface)
                theta, xi, rho = new_heston[:, 1], new_heston[:, 2], new_heston[:, 3]
                jump_means, jump_stds = candidate[:, :5], candidate[:, 5:]
                options = _option_values(
                    log_spot=log_spot,
                    variance=variance,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    jump_means=jump_means,
                    jump_stds=jump_stds,
                    universe=universe,
                    day_index=day_index + 1,
                    jump_start=0.0,
                    kappa=kappa,
                    intensity=intensity,
                    boundaries=boundaries,
                    config=config,
                )
                reset_error = options - old_options
                live_after = torch.as_tensor(
                    universe.expiry_days > day_index + 1,
                    dtype=torch.bool,
                    device=torch_device,
                )[None, :, None]
                live_error = reset_error[live_after.expand_as(reset_error)]
                if live_error.numel():
                    batch_reset_maximum = max(
                        batch_reset_maximum, float(torch.max(torch.abs(live_error)).cpu())
                    )
                    batch_reset_squared_sum += float(torch.sum(live_error.square()).cpu())
                    batch_reset_observations += live_error.numel()
                option_returns = (options - previous_options) / previous_spot[:, None, None]
                returns[:, day_index, 0] = (
                    (torch.exp(log_spot) - previous_spot) / previous_spot
                ).double().cpu().numpy()
                returns[:, day_index, 1:] = option_returns.reshape(
                    config.batch_size, -1
                ).double().cpu().numpy()

            _atomic_npz(
                destination,
                features=features,
                returns=returns,
                reset_maximum_absolute_price_error=np.asarray(batch_reset_maximum),
                reset_squared_price_error_sum=np.asarray(batch_reset_squared_sum),
                reset_price_error_observations=np.asarray(batch_reset_observations),
                clipped_surface_nodes=np.asarray(batch_clipped),
            )
            clipped_total += batch_clipped
            reset_maximum = max(reset_maximum, batch_reset_maximum)
            reset_squared_sum += batch_reset_squared_sum
            reset_observations += batch_reset_observations
            completed = stop
            elapsed = time.monotonic() - started
            record = {
                "stage": "crc_restricted_population",
                "elapsed_seconds": elapsed,
                "completed_work_units": completed,
                "total_work_units": config.path_count,
                "estimated_remaining_seconds": elapsed * (config.path_count - completed) / completed,
                "maximum_reset_price_error": reset_maximum,
                "clipped_surface_nodes": clipped_total,
            }
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

        paths = sorted(output_directory.glob("population-*-*.npz"))
        if len(paths) != batch_count:
            raise RuntimeError(f"found {len(paths)} population batches, expected {batch_count}")
        feature_parts = []
        return_parts = []
        clipped_total = 0
        reset_maximum = 0.0
        reset_squared_sum = 0.0
        reset_observations = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                feature_parts.append(np.asarray(archive["features"], dtype=np.float32))
                return_parts.append(np.asarray(archive["returns"], dtype=np.float64))
                clipped_total += int(archive["clipped_surface_nodes"])
                reset_maximum = max(
                    reset_maximum, float(archive["reset_maximum_absolute_price_error"])
                )
                reset_squared_sum += float(archive["reset_squared_price_error_sum"])
                reset_observations += int(archive["reset_price_error_observations"])
        combined = output_directory / "population.npz"
        _atomic_npz(
            combined,
            features=np.concatenate(feature_parts),
            returns=np.concatenate(return_parts),
            active_mask=active_mask,
            pair_ids=np.repeat(np.arange(config.path_count // 2, dtype=np.int64), 2),
            expiry_days=universe.expiry_days,
            strikes=universe.strikes,
            is_put=universe.is_put,
        )
        summary = {
            "schema": "gambara-teichmann-crc-restricted-population-v1",
            "config": asdict(config),
            "checkpoint": str(checkpoint.resolve()),
            "dataset_directory": str(dataset_directory.resolve()),
            "target_fit_report": str(target_fit_report.resolve()),
            "population": str(combined.resolve()),
            "maximum_reset_price_error": reset_maximum,
            "rms_reset_price_error": math.sqrt(reset_squared_sum / reset_observations),
            "reset_price_error_observations": reset_observations,
            "clipped_surface_nodes": clipped_total,
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary = output_directory / ".population-report.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_directory / "population-report.json")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--target-fit-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--path-count", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--n-quad", type=int, default=240)
    parser.add_argument("--u-max", type=float, default=600.0)
    arguments = parser.parse_args()
    population = generate_population(
        checkpoint=arguments.checkpoint,
        dataset_directory=arguments.dataset_directory,
        target_fit_report=arguments.target_fit_report,
        output_directory=arguments.output_directory,
        config=CrcPopulationConfig(
            path_count=arguments.path_count,
            batch_size=arguments.batch_size,
            n_quad=arguments.n_quad,
            u_max=arguments.u_max,
        ),
        device=arguments.device,
    )
    print(population)


if __name__ == "__main__":
    main()
