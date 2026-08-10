"""Run expiry-separated conditional-arbitrage searches on cached markets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.torch import BaseEnvironment

from .expiry_arbitrage import (
    ExpiryArbitrageConfig,
    ExpiryArbitrageMethod,
    ExpiryArbitrageModel,
    ExpiryArbitrageTrainConfig,
    exact_payoff_cone_lower_bound,
    expiry_portfolio_pnl,
    powerexp_oce_by_expiry,
    propagated_expiry_band,
    smooth_payoff_cone_lower_bound,
    weighted_expiry_mean,
)
from .matched_cache import MatchedUniverseBlockCache
from .required_runs import (
    RequiredCase,
    clopper_pearson_one_sided,
    derive_case_arrays,
    stratified_path_split,
)

_DEFAULT_CACHE_ROOT = Path("/media/storage/tmp/quantdata/dysanos/required-runs/universes")
_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/expiry-arbitrage"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/expiry-arbitrage/checkpoints"
)


@dataclass
class LoadedExpiryCase:
    """Dense expiry-separated arrays loaded once before GPU fitting."""

    state: np.ndarray
    returns: np.ndarray
    costs: np.ndarray
    numerical_band: np.ndarray
    log_path_weight: np.ndarray
    component: np.ndarray
    source_cache: Path
    source_cache_uid: str
    option_price: np.ndarray | None = None
    delta: np.ndarray | None = None
    log_strike: np.ndarray | None = None
    spot: np.ndarray | None = None


@dataclass
class DeviceExpiryCase:
    """The tensors used by one GPU worker."""

    state: torch.Tensor
    returns: torch.Tensor
    costs: torch.Tensor
    numerical_band: torch.Tensor
    path_weight: torch.Tensor
    component: torch.Tensor
    reference: torch.Tensor
    validation: torch.Tensor
    option_price: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    cash_strike: torch.Tensor | None = None
    spot: torch.Tensor | None = None
    option_is_put: torch.Tensor | None = None
    portfolio_barrier_rate: float | None = None
    portfolio_normalization: str = "native"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _find_cache(case: RequiredCase, cache_root: Path) -> MatchedUniverseBlockCache:
    matches: list[MatchedUniverseBlockCache] = []
    incompatible: list[Path] = []
    for manifest_path in cache_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        configuration = manifest.get("configuration", {})
        if (
            manifest.get("status") == "complete"
            and manifest.get("num_samples") == case.path_count
            and configuration.get("generator") == case.generator
            and configuration.get("trading_frequency") == case.trading_frequency
        ):
            cache = MatchedUniverseBlockCache(
                cache_root,
                cache_uid=manifest["cache_uid"],
                configuration=configuration,
                num_samples=case.path_count,
            )
            if cache.is_complete:
                matches.append(cache)
            else:
                incompatible.append(manifest_path.parent)
    verify_inp(
        len(matches) == 1,
        lambda: (
            "expected one complete base cache for "
            f"{case.generator}/{case.trading_frequency}/{case.path_count}, "
            f"found {[str(cache.directory) for cache in matches]}; "
            f"ignored incompatible caches {[str(path) for path in incompatible]}"
        ),
    )
    return matches[0]


def _expiry_instrument_indices(derived) -> np.ndarray:
    """Return one matched spot plus all option strikes for each expiry."""

    expiry_count = int(np.max(derived.expiry_index)) + 1
    rows: list[np.ndarray] = []
    for expiry in range(expiry_count):
        spot = np.flatnonzero(
            (np.asarray(derived.instrument_kind) == "spot")
            & ((derived.expiry_index == expiry) | (derived.expiry_index == -1))
        )
        options = np.flatnonzero(
            (np.asarray(derived.instrument_kind) == "option")
            & (derived.expiry_index == expiry)
        )
        verify_inp(
            spot.size >= 1 and options.size >= 1,
            lambda expiry=expiry: f"expiry {expiry} has no matched spot or options",
        )
        rows.append(np.concatenate((spot[:1], options)))
    result = np.stack(rows)
    verify_inp(
        np.all(result[:, 1:] >= 0) and np.unique(result[:, 1:]).size == result[:, 1:].size,
        "expiry option slices overlap or contain invalid indices",
    )
    return result


def _load_case(
    case: RequiredCase,
    *,
    cache_root: Path,
    need_primitives: bool,
) -> LoadedExpiryCase:
    """Stream a compressed base cache into dense expiry-separated arrays."""

    cache = _find_cache(case, cache_root)
    arrays: dict[str, np.ndarray] | None = None
    primitive: dict[str, np.ndarray] = {}
    selection: np.ndarray | None = None
    last_stop = 0
    started = time.monotonic()
    for block_number, block in enumerate(cache.iter_blocks(), start=1):
        universe = getattr(block, case.return_horizon)
        derived = derive_case_arrays(
            universe,
            hedge=case.hedge,
            cost_rate=case.cost_rate,
        )
        if arrays is None:
            selection = _expiry_instrument_indices(derived)
            S = case.path_count
            state_shape = np.asarray(universe["xstate"]).shape[1:]
            case_shape = (*derived.returns.shape[1:2], *selection.shape)
            arrays = {
                "state": np.empty((S, *state_shape), dtype=np.float32),
                "returns": np.empty((S, *case_shape), dtype=np.float32),
                "costs": np.empty((S, *case_shape), dtype=np.float32),
                "numerical_band": np.empty((S, *case_shape), dtype=np.float32),
                "log_path_weight": np.empty(S, dtype=np.float64),
                "component": np.empty(S, dtype=np.int16),
            }
            if need_primitives:
                option_shape = np.asarray(universe["C"]).shape[1:]
                primitive = {
                    "option_price": np.empty((S, *option_shape), dtype=np.float32),
                    "delta": np.empty((S, *option_shape), dtype=np.float32),
                    "log_strike": np.empty((S, *option_shape), dtype=np.float32),
                    "spot": np.empty((S, *np.asarray(universe["spot"]).shape[1:]), dtype=np.float32),
                }
        assert arrays is not None and selection is not None
        verify_inp(block.start == last_stop, "cached blocks are not contiguous")
        interval = slice(block.start, block.stop)
        arrays["state"][interval] = universe["xstate"]
        for name, value in (
            ("returns", derived.returns),
            ("costs", derived.costs),
            ("numerical_band", derived.numerical_band),
        ):
            selected = np.take(value, selection.reshape(-1), axis=-1).reshape(
                value.shape[0], value.shape[1], *selection.shape
            )
            arrays[name][interval] = selected
        arrays["log_path_weight"][interval] = universe.get(
            "log_path_weight", np.zeros(block.stop - block.start)
        )
        arrays["component"][interval] = universe.get(
            "importance_component", np.zeros(block.stop - block.start)
        )
        if need_primitives:
            primitive["option_price"][interval] = universe["C"]
            primitive["delta"][interval] = universe["delta"]
            primitive["log_strike"][interval] = universe["log_strikes"]
            primitive["spot"][interval] = universe["spot"]
        last_stop = block.stop
        if block_number == 1 or block.stop == case.path_count or block_number % 10 == 0:
            print(
                f"[{case.generator}/{case.trading_frequency}/{case.path_count}] "
                f"loaded {block.stop:,} paths in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    verify_inp(arrays is not None and last_stop == case.path_count, "base cache is incomplete")
    for name, value in {**arrays, **primitive}.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"loaded field {name!r} contains NaN or infinity")
    verify_inp(np.all(arrays["costs"] >= 0.0), "loaded costs are negative")
    verify_inp(np.all(arrays["numerical_band"] > 0.0), "loaded bands are not positive")
    return LoadedExpiryCase(
        **arrays,
        source_cache=cache.directory,
        source_cache_uid=cache.cache_uid,
        **primitive,
    )


def _to_device(
    loaded: LoadedExpiryCase,
    *,
    case: RequiredCase,
    env: BaseEnvironment,
) -> tuple[DeviceExpiryCase, dict[str, Any]]:
    split = stratified_path_split(
        loaded.component,
        audit_fraction=0.2,
        seed=case.path_count + 97,
    )
    log_weight = loaded.log_path_weight
    # Values below float32's smallest subnormal carry negligible total mass.
    path_weight = np.exp(
        np.maximum(log_weight, math.log(np.nextafter(np.float32(0.0), np.float32(1.0))))
    ).astype(np.float32)
    if not np.all(np.isfinite(path_weight)) or not np.all(path_weight > 0.0):
        raise FloatingPointError("importance weights are not finite-positive")

    def tensor(value: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(value),
            dtype=env.fdtype if dtype is None else dtype,
            device=env.device,
        )

    option_price = tensor(loaded.option_price) if loaded.option_price is not None else None
    delta = tensor(loaded.delta) if loaded.delta is not None else None
    spot = tensor(loaded.spot) if loaded.spot is not None else None
    cash_strike = None
    if loaded.log_strike is not None:
        assert spot is not None
        cash_strike = torch.exp(tensor(loaded.log_strike)) * spot[..., None, None]
    result = DeviceExpiryCase(
        state=tensor(loaded.state),
        returns=tensor(loaded.returns),
        costs=tensor(loaded.costs),
        numerical_band=tensor(loaded.numerical_band),
        path_weight=tensor(path_weight),
        component=tensor(loaded.component, dtype=torch.int16),
        reference=tensor(split.reference, dtype=torch.int64),
        validation=tensor(split.audit, dtype=torch.int64),
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        spot=spot,
    )
    del loaded
    gc.collect()
    diagnostics = {
        "split_seed": split.seed,
        "validation_fraction": split.audit_fraction,
        "component_counts": split.component_counts,
        "weight_definition": "exp(log(p/q)); physical estimates are N^-1 sum_r (p/q)_r f_r",
        "full_weight_mean": float(path_weight.mean(dtype=np.float64)),
        "full_weight_effective_sample_size": float(
            np.square(path_weight.sum(dtype=np.float64))
            / np.square(path_weight.astype(np.float64)).sum()
        ),
    }
    return result, diagnostics


def _state_normalization(data: DeviceExpiryCase) -> tuple[torch.Tensor, torch.Tensor]:
    state = data.state[data.reference]
    mean = state.mean(dim=(0, 1))
    scale = state.std(dim=(0, 1), correction=0).clamp_min(1.0e-6)
    if not torch.isfinite(mean).all().item() or not torch.isfinite(scale).all().item():
        raise FloatingPointError("reference-only state normalization is non-finite")
    return mean, scale


def _pnl_scale(data: DeviceExpiryCase) -> torch.Tensor:
    sample = data.reference[: min(50_000, data.reference.numel())]
    weight = data.path_weight[sample]
    square = data.returns[sample].square().mean(dim=-1).sum(dim=1)
    scale = torch.sqrt(weighted_expiry_mean(square, weight)).clamp_min(1.0e-4)
    return scale.detach()


def _batch_loss(
    model: ExpiryArbitrageModel,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    method: ExpiryArbitrageMethod,
    pnl_scale: torch.Tensor,
    dual: torch.Tensor,
    rho: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state = data.state[indices]
    weight = data.path_weight[indices]
    action, raw_l1, cap_penalty = model(state)
    cap = weighted_expiry_mean(cap_penalty, weight)
    if method == "payoff-cone":
        assert data.spot is not None
        assert data.option_price is not None
        assert data.delta is not None
        assert data.cash_strike is not None
        _, path_value, tail_slope = smooth_payoff_cone_lower_bound(
            action,
            spot=data.spot[indices],
            option_price=data.option_price[indices],
            delta=data.delta[indices],
            cash_strike=data.cash_strike[indices],
            costs=data.costs[indices],
            hedge=model.config.hedge,
            temperature=temperature,
        )
        scaled = path_value / pnl_scale
        mean = weighted_expiry_mean(scaled, weight)
        violation = weighted_expiry_mean(torch.relu(-scaled).square(), weight)
        tail_violation = weighted_expiry_mean(
            torch.relu(-tail_slope).square(), weight
        )
        constraint = violation + tail_violation
        loss = (
            -mean
            + dual * constraint
            + 0.5 * rho * constraint.square()
            + model.config.raw_norm_penalty * cap
        ).mean()
        values = {
            "objective": mean,
            "constraint": constraint,
            "cap_penalty": cap,
            "tail_constraint": tail_violation,
            "batch_loss_count": (path_value < 0.0).sum(dim=0),
            "batch_gain_count": (path_value > 0.0).sum(dim=0),
        }
    else:
        _, path_pnl = expiry_portfolio_pnl(
            action, data.returns[indices], data.costs[indices]
        )
        if method == "negative-part":
            _, path_band = propagated_expiry_band(
                action,
                data.returns[indices],
                data.costs[indices],
                data.numerical_band[indices],
            )
            scaled = path_pnl / pnl_scale
            negative_excess = torch.relu(-(path_pnl + path_band) / pnl_scale)
            mean = weighted_expiry_mean(scaled, weight)
            constraint = weighted_expiry_mean(negative_excess.square(), weight)
            loss = (
                -mean
                + dual * constraint
                + 0.5 * rho * constraint.square()
                + model.config.raw_norm_penalty * cap
            ).mean()
            values = {
                "objective": mean,
                "constraint": constraint,
                "cap_penalty": cap,
                "batch_loss_count": (path_pnl < -path_band).sum(dim=0),
                "batch_gain_count": (path_pnl > path_band).sum(dim=0),
            }
        else:
            oce, y_residual, _ = powerexp_oce_by_expiry(model, path_pnl, weight)
            loss = (-oce + model.config.raw_norm_penalty * cap).mean()
            values = {
                "objective": oce,
                "constraint": torch.zeros_like(oce),
                "cap_penalty": cap,
                "y_residual": y_residual,
                "batch_loss_count": (path_pnl < 0.0).sum(dim=0),
                "batch_gain_count": (path_pnl > 0.0).sum(dim=0),
            }
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"{method} objective became NaN or infinity")
    return loss, values


@torch.no_grad()
def _evaluate_arrays(
    model: ExpiryArbitrageModel,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    method: ExpiryArbitrageMethod,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    paths = indices.numel()
    E = model.config.expiries
    T = model.config.trading_times
    pnl = np.empty((paths, E), dtype=np.float64)
    band = np.empty((paths, E), dtype=np.float64)
    cone = np.empty((paths, E), dtype=np.float64) if method == "payoff-cone" else None
    cone_band = np.empty((paths, E), dtype=np.float64) if method == "payoff-cone" else None
    time_loss = np.zeros((T, E), dtype=np.int64)
    time_gain = np.zeros((T, E), dtype=np.int64)
    raw_above = np.zeros(E, dtype=np.int64)
    action_boundary = np.zeros(E, dtype=np.int64)
    raw_count = 0
    raw_max = np.zeros(E, dtype=np.float64)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action, raw_l1, _ = model(data.state[batch])
        time_pnl, path_pnl = expiry_portfolio_pnl(
            action, data.returns[batch], data.costs[batch]
        )
        time_band, path_band = propagated_expiry_band(
            action,
            data.returns[batch],
            data.costs[batch],
            data.numerical_band[batch],
        )
        pnl[start:stop] = path_pnl.cpu().double().numpy()
        band[start:stop] = path_band.cpu().double().numpy()
        time_loss += (time_pnl < -time_band).sum(dim=0).cpu().numpy()
        time_gain += (time_pnl > time_band).sum(dim=0).cpu().numpy()
        raw_above += (raw_l1 > 1.0).sum(dim=(0, 1)).cpu().numpy()
        action_boundary += (
            action.abs().sum(dim=-1) >= 0.999
        ).sum(dim=(0, 1)).cpu().numpy()
        raw_count += raw_l1.shape[0] * raw_l1.shape[1]
        raw_max = np.maximum(raw_max, raw_l1.amax(dim=(0, 1)).cpu().double().numpy())
        if method == "payoff-cone":
            assert cone is not None and cone_band is not None
            assert data.spot is not None
            assert data.option_price is not None
            assert data.delta is not None
            assert data.cash_strike is not None
            _, path_worst, _ = exact_payoff_cone_lower_bound(
                action,
                spot=data.spot[batch],
                option_price=data.option_price[batch],
                delta=data.delta[batch],
                cash_strike=data.cash_strike[batch],
                costs=data.costs[batch],
                hedge=model.config.hedge,
            )
            option_scale = (
                data.option_price[batch].abs()
                + data.cash_strike[batch].abs()
                + data.spot[batch, :, None, None].abs()
            )
            operand_scale = torch.cat(
                (data.spot[batch, :, None, None].abs().expand(-1, -1, E, 1), option_scale),
                dim=-1,
            )
            time_cone_band = 32.0 * torch.finfo(torch.float32).eps * (
                1.0 + (action.abs() * (operand_scale + data.costs[batch])).sum(dim=-1)
            )
            cone[start:stop] = path_worst.cpu().double().numpy()
            cone_band[start:stop] = time_cone_band.sum(dim=1).cpu().double().numpy()
    return {
        "pnl": pnl,
        "band": band,
        "cone": cone,
        "cone_band": cone_band,
        "time_loss_count": time_loss,
        "time_gain_count": time_gain,
        "raw_norm_above_one_fraction": raw_above / raw_count,
        "action_l1_boundary_fraction": action_boundary / raw_count,
        "maximum_raw_l1_norm": raw_max,
    }


def _expiry_metrics(
    arrays: dict[str, np.ndarray | float],
    *,
    weight: np.ndarray,
    component: np.ndarray,
    method: ExpiryArbitrageMethod,
) -> dict[str, Any]:
    pnl = np.asarray(arrays["pnl"])
    band = np.asarray(arrays["band"])
    certificate_value = pnl
    certificate_band = band
    basis = "realized path P&L"
    if method == "payoff-cone":
        certificate_value = np.asarray(arrays["cone"])
        certificate_band = np.asarray(arrays["cone_band"])
        basis = "exact piecewise-linear terminal-payoff lower bound"
    direct = component == 0
    reports: list[dict[str, Any]] = []
    for expiry in range(pnl.shape[1]):
        value = certificate_value[:, expiry]
        tolerance = certificate_band[:, expiry]
        loss = value < -tolerance
        gain = value > tolerance
        direct_loss = loss & direct
        direct_gain = gain & direct
        minimum = float(np.min(value + tolerance))
        gain_lower, _ = clopper_pearson_one_sided(
            int(direct_gain.sum()), int(direct.sum())
        )
        weighted_value = weight * pnl[:, expiry]
        mean = float(weighted_value.mean())
        standard_error = float(weighted_value.std(ddof=0) / math.sqrt(weight.size))
        reports.append(
            {
                "expiry_index": expiry,
                "certificate_basis": basis,
                "proposal_loss_count": int(loss.sum()),
                "proposal_gain_count": int(gain.sum()),
                "direct_P_paths": int(direct.sum()),
                "direct_P_loss_count": int(direct_loss.sum()),
                "direct_P_gain_count": int(direct_gain.sum()),
                "direct_P_gain_probability_lower_95": gain_lower,
                "upper_tail_unbounded_count": int(np.isneginf(value).sum()),
                "minimum_certificate_value_minus_negative_band": (
                    minimum if math.isfinite(minimum) else None
                ),
                "weighted_P_mean_path_pnl": mean,
                "weighted_P_path_standard_error": standard_error,
                "weighted_P_mean_over_standard_error": (
                    mean / standard_error if standard_error > 0.0 else None
                ),
                "realized_path_loss_count": int((pnl[:, expiry] < -band[:, expiry]).sum()),
                "realized_path_gain_count": int((pnl[:, expiry] > band[:, expiry]).sum()),
                "confirmed": bool(
                    loss.sum() == 0
                    and direct_loss.sum() == 0
                    and direct_gain.sum() > 0
                    and direct.sum() >= 20
                ),
            }
        )
    return {
        "paths": int(pnl.shape[0]),
        "weight_mean": float(weight.mean()),
        "weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "candidate_expiry_count": sum(report["confirmed"] for report in reports),
        "expiries": reports,
        "time_loss_count": np.asarray(arrays["time_loss_count"]).tolist(),
        "time_gain_count": np.asarray(arrays["time_gain_count"]).tolist(),
        "raw_norm_above_one_fraction": np.asarray(
            arrays["raw_norm_above_one_fraction"]
        ).tolist(),
        "action_l1_boundary_fraction": np.asarray(
            arrays["action_l1_boundary_fraction"]
        ).tolist(),
        "maximum_raw_l1_norm": np.asarray(arrays["maximum_raw_l1_norm"]).tolist(),
    }


def _gradient_diagnostics(
    model: ExpiryArbitrageModel,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    method: ExpiryArbitrageMethod,
    pnl_scale: torch.Tensor,
    dual: torch.Tensor,
    train_config: ExpiryArbitrageTrainConfig,
    temperature: float,
) -> dict[str, Any]:
    sample = indices[: min(indices.numel(), train_config.valuation_batch_size)]
    model.zero_grad(set_to_none=True)
    loss, values = _batch_loss(
        model,
        data,
        sample,
        method=method,
        pnl_scale=pnl_scale,
        dual=dual,
        rho=train_config.augmented_rho,
        temperature=temperature,
    )
    loss.backward()
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    flat = torch.cat(gradients) if gradients else torch.zeros(1, device=data.state.device)
    report = {
        "sample_paths": int(sample.numel()),
        "loss": float(loss.detach().cpu()),
        "parameter_gradient_l2": float(torch.linalg.vector_norm(flat).cpu()),
        "parameter_gradient_maximum_absolute": float(flat.abs().max().cpu()),
        "objective_by_expiry": values["objective"].detach().cpu().tolist(),
        "constraint_by_expiry": values["constraint"].detach().cpu().tolist(),
    }
    if "y_residual" in values:
        report["oce_offset_first_order_residual"] = (
            values["y_residual"].detach().cpu().tolist()
        )
    model.zero_grad(set_to_none=True)
    return report


def _checkpoint(
    path: Path,
    *,
    model: ExpiryArbitrageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    dual: torch.Tensor,
    tracker: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = {
        "model_code_version": ExpiryArbitrageConfig.version.unique_id48,
        "user_version": model.config.user_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "dual": dual,
        "tracker": tracker,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(model.device),
    }
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _train(
    data: DeviceExpiryCase,
    *,
    case: RequiredCase,
    method: ExpiryArbitrageMethod,
    model_config: ExpiryArbitrageConfig,
    train_config: ExpiryArbitrageTrainConfig,
    checkpoint_path: Path,
    env: BaseEnvironment,
) -> tuple[ExpiryArbitrageModel, dict[str, Any], torch.Tensor, torch.Tensor]:
    state_mean, state_scale = _state_normalization(data)
    pnl_scale = _pnl_scale(data)
    model = ExpiryArbitrageModel(
        env,
        config=model_config,
        state_mean=state_mean,
        state_scale=state_scale,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    dual = torch.zeros(model_config.expiries, dtype=env.fdtype, device=env.device)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        if (
            saved.get("model_code_version") != ExpiryArbitrageConfig.version.unique_id48
            or saved.get("user_version") != model_config.user_version
        ):
            print("Checkpoint version changed; cold-starting the expiry model.", flush=True)
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            dual = saved["dual"].to(env.device)
            tracker = list(saved.get("tracker", []))
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), env.device)
            print(f"Warm-started {method} from step {start_step:,}.", flush=True)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(train_config.seed)
    callable_model = torch.compile(model, dynamic=False) if train_config.compile else model
    torch.cuda.reset_peak_memory_stats(env.device)
    started = time.monotonic()
    converged = False
    final_temperature = max(1.0e-6, 1.0e-4 * float(torch.median(pnl_scale).cpu()))
    initial_temperature = max(
        final_temperature,
        5.0e-2 * float(torch.median(pnl_scale).cpu()),
    )
    for step in range(start_step + 1, train_config.epochs + 1):
        draw = torch.randint(
            data.reference.numel(),
            (train_config.batch_size,),
            generator=generator,
            device=env.device,
        )
        indices = data.reference[draw]
        progress = step / train_config.epochs
        temperature = initial_temperature * math.pow(
            final_temperature / initial_temperature, progress
        )
        optimizer.zero_grad(set_to_none=True)
        loss, values = _batch_loss(
            callable_model,
            data,
            indices,
            method=method,
            pnl_scale=pnl_scale,
            dual=dual,
            rho=train_config.augmented_rho,
            temperature=temperature,
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"{method} gradient became NaN or infinity")
        optimizer.step()
        if step % train_config.value_update == 0 or step == train_config.epochs:
            constraint = values["constraint"].detach()
            if method != "powerexp-oce":
                dual = torch.clamp(
                    dual + train_config.augmented_rho * constraint,
                    min=0.0,
                    max=1.0e8,
                )
            item = {
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "loss": float(loss.detach().cpu()),
                "objective_by_expiry": values["objective"].detach().cpu().tolist(),
                "constraint_by_expiry": constraint.cpu().tolist(),
                "cap_penalty_by_expiry": values["cap_penalty"].detach().cpu().tolist(),
                "dual_by_expiry": dual.detach().cpu().tolist(),
                "temperature": temperature if method == "payoff-cone" else None,
                "provisional_batch_loss_count_by_expiry": values[
                    "batch_loss_count"
                ].detach().cpu().tolist(),
                "provisional_batch_gain_count_by_expiry": values[
                    "batch_gain_count"
                ].detach().cpu().tolist(),
            }
            if "y_residual" in values:
                item["oce_offset_first_order_residual"] = (
                    values["y_residual"].detach().cpu().tolist()
                )
            tracker.append(item)
            recent = tracker[-train_config.convergence_points :]
            if step >= train_config.minimum_epochs and len(recent) == train_config.convergence_points:
                scores = np.asarray([sum(point["objective_by_expiry"]) for point in recent])
                scale = max(1.0, float(np.max(np.abs(scores))))
                stable = float(np.ptp(scores)) <= train_config.convergence_relative_tolerance * scale
                feasible = method == "powerexp-oce" or max(item["constraint_by_expiry"]) <= (
                    10.0 * train_config.feasibility_tolerance
                )
                converged = stable and feasible
            print(
                f"[{case.generator}/{method}] step {step:,}: loss={item['loss']:.6g}, "
                f"provisional no-loss expiries on batch="
                f"{sum(losses == 0 and gains > 0 for losses, gains in zip(item['provisional_batch_loss_count_by_expiry'], item['provisional_batch_gain_count_by_expiry'], strict=True))}/7, "
                f"max constraint={max(item['constraint_by_expiry']):.3g}",
                flush=True,
            )
            if step % train_config.checkpoint_update == 0 or converged:
                _checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    dual=dual,
                    tracker=tracker,
                )
            if converged:
                break
    final_step = tracker[-1]["step"] if tracker else start_step
    if not checkpoint_path.exists() or final_step % train_config.checkpoint_update != 0:
        _checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=final_step,
            dual=dual,
            tracker=tracker,
        )
    training = {
        "steps": final_step,
        "converged": converged,
        "elapsed_seconds": time.monotonic() - started,
        "pnl_scale_by_expiry": pnl_scale.cpu().tolist(),
        "state_mean": state_mean.cpu().tolist(),
        "state_scale": state_scale.cpu().tolist(),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(env.device)),
        "tracker": tracker,
    }
    return model, training, pnl_scale, dual


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = RequiredCase(
        generator=args.generator,
        trading_frequency=args.frequency,
        return_horizon="expiry",
        hedge=args.hedge,
        cost_rate=args.cost_rate,
        path_count=args.paths,
    )
    method: ExpiryArbitrageMethod = args.method
    loaded = _load_case(
        case,
        cache_root=Path(args.cache_root),
        need_primitives=method == "payoff-cone",
    )
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=32, device=args.device, debug=args.debug)
    data, split_diagnostics = _to_device(loaded, case=case, env=env)
    torch.cuda.synchronize(env.device)
    label = (
        f"{method}-{case.generator}-{case.trading_frequency}-{case.path_count}-"
        f"expiry-{case.hedge}-{case.cost_rate:.0e}"
    )
    model_config = ExpiryArbitrageConfig(
        label=label,
        group=(
            f"gendysanos/expiry-arbitrage/{method}/{case.generator}/"
            f"{case.trading_frequency}/{case.path_count}"
        ),
        user_version=args.user_version,
        state_features=data.state.shape[-1],
        method=method,
        hedge=case.hedge,
        cost_rate=case.cost_rate,
        lmbda=args.lmbda,
        raw_norm_penalty=args.raw_norm_penalty,
    )
    train_config = ExpiryArbitrageTrainConfig(
        batch_size=args.batch_size,
        valuation_batch_size=args.valuation_batch_size,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        value_update=args.value_update,
        checkpoint_update=args.checkpoint_update,
        lr=args.lr,
        compile=not args.no_compile,
        augmented_rho=args.augmented_rho,
        feasibility_tolerance=args.feasibility_tolerance,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        seed=args.seed,
    )
    checkpoint_path = Path(args.checkpoint_root) / model_config.group / (
        f"{model_config.filename}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output_root) / f"{label}.json"
    started = time.monotonic()
    with FileLock(str(checkpoint_path) + ".lock"):
        model, training, pnl_scale, dual = _train(
            data,
            case=case,
            method=method,
            model_config=model_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            env=env,
        )
        final_temperature = max(
            1.0e-6, 1.0e-4 * float(torch.median(pnl_scale).cpu())
        )
        reference_arrays = _evaluate_arrays(
            model,
            data,
            data.reference,
            method=method,
            batch_size=train_config.valuation_batch_size,
        )
        validation_arrays = _evaluate_arrays(
            model,
            data,
            data.validation,
            method=method,
            batch_size=train_config.valuation_batch_size,
        )
        reference_index = data.reference.cpu().numpy()
        validation_index = data.validation.cpu().numpy()
        weight = data.path_weight.cpu().double().numpy()
        component = data.component.cpu().numpy()
        reference = _expiry_metrics(
            reference_arrays,
            weight=weight[reference_index],
            component=component[reference_index],
            method=method,
        )
        validation = _expiry_metrics(
            validation_arrays,
            weight=weight[validation_index],
            component=component[validation_index],
            method=method,
        )
        gradient = {
            "training": _gradient_diagnostics(
                model,
                data,
                data.reference,
                method=method,
                pnl_scale=pnl_scale,
                dual=dual,
                train_config=train_config,
                temperature=final_temperature,
            ),
            "validation": _gradient_diagnostics(
                model,
                data,
                data.validation,
                method=method,
                pnl_scale=pnl_scale,
                dual=dual,
                train_config=train_config,
                temperature=final_temperature,
            ),
        }
        report = {
            "status": "complete",
            "schema_version": "1.0",
            "experiment": "expiry-separated conditional-arbitrage search",
            "configuration": asdict(case),
            "method": method,
            "method_definition": {
                "expiry_separation": (
                    "seven independent problems; each contains one matched spot leg and "
                    "20 strikes; no cross-expiry position is permitted"
                ),
                "action": "a_e(t,x_t), with ||a_e(t,x_t)||_1 <= 1 at every trading time",
                "path_pnl": "X_e=sum_t[a_e' dH_e-|a_e|' c_e]",
                "payoff_cone": (
                    "maximize the exact piecewise-linear terminal-payoff lower bound; "
                    "terminal spot kinks and the upper-tail slope characterize the cone"
                ),
                "negative_part": "maximize E[X_e] subject to E[((-X_e-eta_e)_+)^2]=0",
                "powerexp_oce": "sup_{a_e,y_e} E[u_powerexp(X_e+y_e)-y_e]",
                "radial_budget": (
                    "a=v/max(1,||v||_1), with (||v||_1-1)_+^2 exterior penalty; "
                    "there is no tanh action cap"
                ),
                "cost": (
                    "gross, without option/spot netting; delta option cost is "
                    "q*(|C|+|Delta*S|), and an explicit spot leg costs q*|S|"
                ),
            },
            "source_cache": source_cache,
            "source_cache_uid": source_cache_uid,
            "split": split_diagnostics,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "training": training,
            "training_set": reference,
            "validation_set": validation,
            "gradient_diagnostics": gradient,
            "validation_confirmed_expiries": [
                item["expiry_index"]
                for item in validation["expiries"]
                if item["confirmed"]
            ],
            "has_validation_confirmed_arbitrage": bool(
                validation["candidate_expiry_count"] > 0
            ),
            "runtime_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path),
            "device": str(env.device),
        }
        _atomic_json(report_path, report)
        print(
            f"Completed {label}: validation-confirmed expiries "
            f"{report['validation_confirmed_expiries']}; report={report_path}",
            flush=True,
        )
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("payoff-cone", "negative-part", "powerexp-oce"), required=True)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--frequency", choices=("daily", "weekly"), default="weekly")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--hedge", choices=("raw", "delta"), default="delta")
    parser.add_argument("--cost-rate", type=float, choices=(0.0, 1.0e-4, 1.0e-3), default=1.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30_000)
    parser.add_argument("--minimum-epochs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=16_384)
    parser.add_argument("--value-update", type=int, default=500)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--raw-norm-penalty", type=float, default=1.0)
    parser.add_argument("--augmented-rho", type=float, default=100.0)
    parser.add_argument("--feasibility-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--convergence-points", type=int, default=8)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=82_061)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
