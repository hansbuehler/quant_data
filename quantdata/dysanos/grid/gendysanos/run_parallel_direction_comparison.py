"""Compare learned state partitions with independent feed-forward directions."""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock
from scipy.special import logsumexp
from torch import Tensor

from quantdata.torch import BaseEnvironment

from .arb_tracing import fit_physical_state_whitener, weighted_quantile
from .case_c_partition import (
    GatedParallelFeedForwardDirections,
    LearnedPartitionDirections,
    ParallelDirectionConfig,
    ParallelFeedForwardDirections,
    ParallelKernelDirections,
)
from .expiry_arbitrage import payoff_cone_values
from .required_runs import RequiredCase, stratified_path_split
from .run_case_c import _fit_kernel_basis
from .run_expiry_arbitrage_experiments import (
    _DEFAULT_CACHE_ROOT,
    DeviceExpiryCase,
    _load_case,
    _to_device,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/parallel-direction-comparison"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/parallel-direction-comparison/checkpoints"
)

DirectionModel = (
    ParallelFeedForwardDirections
    | GatedParallelFeedForwardDirections
    | ParallelKernelDirections
    | LearnedPartitionDirections
)


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


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def _initialize_gated_direction_from_report(
    model: GatedParallelFeedForwardDirections,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Copy a trained feed-forward direction while retaining fresh gate rows."""

    source_config = report.get("model_config", {})
    verify_inp(
        source_config.get("architecture") == "feedforward",
        "gated initialization requires a feed-forward source report",
    )
    verify_inp(
        len(source_config.get("notionals", ())) == len(model.config.notionals),
        "source and target must contain the same number of independent heads",
    )
    checkpoint_path = Path(report["checkpoint"])
    saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    source = saved["model"]
    target = model.state_dict()
    copied: list[str] = []
    for name, destination in target.items():
        if name not in source:
            continue
        value = source[name].to(dtype=destination.dtype, device=destination.device)
        if value.shape == destination.shape:
            destination.copy_(value)
            copied.append(name)
    model.load_state_dict(target)
    verify_inp(
        any(name.startswith("conditional.") for name in copied),
        "source did not contain compatible direction outputs",
    )
    verify_inp(
        all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
        "gated direction initialization produced non-finite parameters",
    )
    return {
        "source_report": report.get("report_path"),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": int(saved.get("step", -1)),
        "copied_parameter_tensors": copied,
        "separate_gate_parameter_tensors_retained": [
            name for name in target if name.startswith("gate.")
        ],
        "gate_initialization_retained": "sigmoid(-3)",
    }


def _select_times_and_fold(
    data: DeviceExpiryCase,
    *,
    deterministic_time: int,
    conditional_time: int,
    fold: int,
    seed: int,
) -> tuple[DeviceExpiryCase, dict[str, Any]]:
    """Select t0/t6 and construct reciprocal component-stratified half samples."""

    split = stratified_path_split(
        data.component.cpu().numpy(), audit_fraction=0.5, seed=seed
    )
    first = torch.as_tensor(split.reference, dtype=torch.int64, device=data.state.device)
    second = torch.as_tensor(split.audit, dtype=torch.int64, device=data.state.device)
    reference, validation = (first, second) if fold == 0 else (second, first)
    times = torch.as_tensor(
        (deterministic_time, conditional_time),
        dtype=torch.int64,
        device=data.state.device,
    )
    result = DeviceExpiryCase(
        state=data.state.index_select(1, times),
        returns=data.returns.index_select(1, times),
        costs=data.costs.index_select(1, times),
        numerical_band=data.numerical_band.index_select(1, times),
        path_weight=data.path_weight,
        component=data.component,
        reference=reference,
        validation=validation,
        option_price=(
            None
            if data.option_price is None
            else data.option_price.index_select(1, times)
        ),
        delta=None if data.delta is None else data.delta.index_select(1, times),
        cash_strike=(
            None
            if data.cash_strike is None
            else data.cash_strike.index_select(1, times)
        ),
        spot=None if data.spot is None else data.spot.index_select(1, times),
        option_is_put=(
            None
            if data.option_is_put is None
            else data.option_is_put.index_select(1, times)
        ),
        portfolio_barrier_rate=data.portfolio_barrier_rate,
        portfolio_normalization=data.portfolio_normalization,
    )
    diagnostics = {
        "definition": (
            "component-stratified reciprocal 50/50 folds; fold 0 trains on "
            "the first half and fold 1 trains on the second"
        ),
        "seed": seed,
        "fold": fold,
        "training_paths": int(reference.numel()),
        "validation_paths": int(validation.numel()),
        "training_direct_P_paths": int((data.component[reference] == 0).sum().cpu()),
        "validation_direct_P_paths": int(
            (data.component[validation] == 0).sum().cpu()
        ),
        "component_counts": split.component_counts,
    }
    return result, diagnostics


def _to_itm_dollar_portfolio(
    data: DeviceExpiryCase,
    *,
    cost_rate: float,
    vector_barrier: bool,
) -> tuple[DeviceExpiryCase, dict[str, Any]]:
    """Use entry-time ITM options and express every position in dollars.

    Calls below spot are retained. Calls above spot are replaced by puts using
    zero-rate put-call parity, ``P=C+K-S`` and ``dP=dC-dS``. The choice is made
    at entry and remains fixed through expiry. Dividing every return by its
    entry price makes a signed action coordinate a dollar investment.
    """

    verify_inp(
        data.option_price is not None
        and data.cash_strike is not None
        and data.spot is not None,
        "ITM dollar normalization requires expiry option primitives",
    )
    verify_inp(
        data.returns.shape == data.costs.shape == data.numerical_band.shape,
        "ITM dollar normalization received inconsistent market arrays",
    )
    verify_inp(
        data.returns.shape[-1] == data.option_price.shape[-1] + 1,
        "expiry portfolios must contain one spot followed by their option strikes",
    )
    verify_inp(
        math.isfinite(cost_rate) and cost_rate >= 0.0,
        lambda: f"ITM dollar cost rate must be nonnegative, got {cost_rate!r}",
    )
    verify_inp(
        not vector_barrier or cost_rate > 0.0,
        "the logarithmic vector barrier requires a positive cost rate",
    )

    spot = data.spot[..., None, None]
    call_price = data.option_price
    put_price = call_price + data.cash_strike - spot
    option_is_put = data.cash_strike >= spot
    selected_price = torch.where(option_is_put, put_price, call_price)
    price_scale = torch.maximum(
        torch.maximum(call_price.abs(), data.cash_strike.abs()), spot.abs()
    )
    price_floor = 64.0 * torch.finfo(selected_price.dtype).eps * price_scale
    verify_inp(
        bool(torch.isfinite(selected_price).all().item())
        and bool(torch.all(selected_price > price_floor).item()),
        lambda: (
            "entry-time ITM option prices must be finite and numerically positive; "
            f"minimum price={float(selected_price.min().item()):g}, "
            f"maximum floor={float(price_floor.max().item()):g}"
        ),
    )

    spot_return = data.returns[..., :1]
    call_return = data.returns[..., 1:]
    put_return = call_return - spot_return
    selected_return = torch.where(option_is_put, put_return, call_return)
    normalized_spot_return = spot_return / spot
    normalized_option_return = selected_return / selected_price
    normalized_return = torch.cat(
        (normalized_spot_return, normalized_option_return), dim=-1
    )

    spot_band = data.numerical_band[..., :1]
    call_band = data.numerical_band[..., 1:]
    put_band = call_band + spot_band
    selected_band = torch.where(option_is_put, put_band, call_band)
    normalized_band = torch.cat(
        (spot_band / spot, selected_band / selected_price), dim=-1
    )
    normalized_cost = torch.full_like(normalized_return, cost_rate)
    for name, value in (
        ("returns", normalized_return),
        ("costs", normalized_cost),
        ("numerical_band", normalized_band),
    ):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(
                f"ITM dollar-normalized {name} contains NaN or infinity"
            )

    result = DeviceExpiryCase(
        state=data.state,
        returns=normalized_return,
        costs=normalized_cost,
        numerical_band=normalized_band,
        path_weight=data.path_weight,
        component=data.component,
        reference=data.reference,
        validation=data.validation,
        option_price=selected_price,
        delta=None,
        cash_strike=data.cash_strike,
        spot=data.spot,
        option_is_put=option_is_put,
        portfolio_barrier_rate=cost_rate if vector_barrier else None,
        portfolio_normalization="entry-time ITM call-or-put dollars",
    )
    paths, times, expiries, strikes = selected_price.shape
    return result, {
        "definition": (
            "calls with K<S are retained; calls with K>=S are replaced at entry "
            "by puts through P=C+K-S and dP=dC-dS; spot and option returns are "
            "divided by their positive entry prices"
        ),
        "paths": paths,
        "times": times,
        "expiries": expiries,
        "strikes_per_expiry": strikes,
        "put_coordinate_fraction": float(option_is_put.float().mean().item()),
        "minimum_selected_option_price": float(selected_price.min().item()),
        "maximum_selected_option_price": float(selected_price.max().item()),
        "cost": (
            (
                "one vector barrier B_X(a)=-c X log(1-||a||_1/X), applied "
                "to the signed net dollar portfolio rather than coordinatewise"
            )
            if vector_barrier
            else "coordinatewise proportional cost after entry-value normalization"
        ),
        "cost_rate": cost_rate,
    }


def _to_itm_dollar_vector_barrier(
    data: DeviceExpiryCase,
    *,
    cost_rate: float,
) -> tuple[DeviceExpiryCase, dict[str, Any]]:
    """Use ITM dollar instruments with one barrier on signed net investment."""

    return _to_itm_dollar_portfolio(
        data,
        cost_rate=cost_rate,
        vector_barrier=True,
    )


def _parallel_pnl(
    action: Tensor,
    returns: Tensor,
    costs: Tensor,
    *,
    portfolio_barrier_rate: float | None = None,
) -> Tensor:
    """Return unit P&L with shape ``(S,N,T)`` for parallel directions."""

    assert action.ndim == 6 and action.shape[-3] == 2
    assert returns.shape == costs.shape
    assert action.shape[0] == returns.shape[0]
    assert action.shape[2] == returns.shape[1]
    market_return = returns[:, None]
    market_cost = costs[:, None]
    long = action[..., 0, :, :]
    short = action[..., 1, :, :]
    if portfolio_barrier_rate is not None:
        signed = long - short
        relative_notional = signed.abs().sum(dim=(-1, -2))
        if not torch.all(relative_notional < 1.0).item():
            raise FloatingPointError(
                "vector-barrier relative notional reached or exceeded one"
            )
        result = (signed * market_return).sum(dim=(-1, -2))
        result = result + portfolio_barrier_rate * torch.log1p(
            -relative_notional
        )
        if not torch.isfinite(result).all().item():
            raise FloatingPointError("vector-barrier P&L contains NaN or infinity")
        return result
    result = (
        (long - short) * market_return - (long + short) * market_cost
    ).sum(dim=(-1, -2))
    return result


def _parallel_band(
    action: Tensor,
    returns: Tensor,
    costs: Tensor,
    instrument_band: Tensor,
    *,
    portfolio_barrier_rate: float | None = None,
) -> Tensor:
    """Propagate observation errors through all parallel unit-gross directions."""

    assert returns.shape == costs.shape == instrument_band.shape
    gross = (
        (action[..., 0, :, :] - action[..., 1, :, :]).abs()
        if portfolio_barrier_rate is not None
        else action[..., 0, :, :] + action[..., 1, :, :]
    )
    accumulation = 32.0 * torch.finfo(torch.float32).eps * (
        gross * (returns[:, None].abs() + costs[:, None])
    ).sum(dim=(-1, -2))
    return (gross * instrument_band[:, None]).sum(dim=(-1, -2)) + accumulation


def _exact_expiry_lower_bound(
    action: Tensor,
    data: DeviceExpiryCase,
    indices: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Return the all-spot payoff-cone lower bound for every action head."""

    if (
        data.option_price is None
        or data.cash_strike is None
        or data.spot is None
        or (data.option_is_put is None and data.delta is None)
    ):
        return None
    if data.option_is_put is not None:
        paths, heads, times = action.shape[:3]
        signed = action[..., 0, :, :] - action[..., 1, :, :]
        relative_notional = signed.abs().sum(dim=(-1, -2))
        if data.portfolio_barrier_rate is not None:
            verify_inp(
                bool(torch.all(relative_notional < 1.0).item()),
                "exact vector-barrier audit reached its open notional boundary",
            )
            lower = data.portfolio_barrier_rate * torch.log1p(
                -relative_notional
            )
        else:
            representation_gross = action[..., 0, :, :] + action[..., 1, :, :]
            lower = -(
                representation_gross * data.costs[indices, None]
            ).sum(dim=(-1, -2))
        slope_ok = torch.ones_like(lower, dtype=torch.bool)
        for time_index in range(times):
            selected_action = signed[:, :, time_index]
            selected_spot = data.spot[indices, time_index]
            selected_option = data.option_price[indices, time_index]
            selected_strike = data.cash_strike[indices, time_index]
            selected_put = data.option_is_put[indices, time_index]
            for expiry_index in range(selected_option.shape[1]):
                strike = selected_strike[:, expiry_index]
                option_price = selected_option[:, expiry_index]
                is_put = selected_put[:, expiry_index]
                terminal_spot = torch.cat(
                    (
                        torch.zeros_like(strike[:, :1]),
                        strike,
                    ),
                    dim=-1,
                )
                call_payoff = torch.clamp_min(
                    terminal_spot[:, :, None] - strike[:, None, :], 0.0
                )
                put_payoff = torch.clamp_min(
                    strike[:, None, :] - terminal_spot[:, :, None], 0.0
                )
                payoff = torch.where(is_put[:, None, :], put_payoff, call_payoff)
                normalized_option_return = (
                    payoff - option_price[:, None, :]
                ) / option_price[:, None, :]
                normalized_spot_return = (
                    terminal_spot - selected_spot[:, None]
                ) / selected_spot[:, None]
                expiry_action = selected_action[:, :, expiry_index]
                scenario_value = (
                    expiry_action[..., :1]
                    * normalized_spot_return[:, None]
                    + (
                        expiry_action[..., 1:, None]
                        * normalized_option_return[:, None].transpose(-1, -2)
                    ).sum(dim=-2)
                )
                lower[:, :, time_index] += scenario_value.amin(dim=-1)

                tail_slope = expiry_action[..., 0] / selected_spot[:, None]
                call_slope = torch.where(
                    is_put,
                    torch.zeros_like(option_price),
                    option_price.reciprocal(),
                )
                tail_slope = tail_slope + (
                    expiry_action[..., 1:] * call_slope[:, None]
                ).sum(dim=-1)
                slope_tolerance = (
                    32.0
                    * torch.finfo(action.dtype).eps
                    * expiry_action.abs().sum(dim=-1)
                    / selected_spot[:, None]
                )
                valid_slope = tail_slope >= -slope_tolerance
                lower[:, :, time_index] = torch.where(
                    valid_slope,
                    lower[:, :, time_index],
                    torch.full_like(lower[:, :, time_index], -torch.inf),
                )
                slope_ok[:, :, time_index] &= valid_slope
        return lower, slope_ok
    paths, heads, times = action.shape[:3]
    lower = torch.empty(
        (paths, heads, times), dtype=action.dtype, device=action.device
    )
    slope_ok = torch.empty(
        (paths, heads, times), dtype=torch.bool, device=action.device
    )
    signed = action[..., 0, :, :] - action[..., 1, :, :]
    for time_index in range(times):
        selected_action = signed[:, :, time_index]
        selected_spot = data.spot[indices, time_index, None].expand(-1, heads)
        selected_option = data.option_price[
            indices, time_index, None
        ].expand(-1, heads, -1, -1)
        selected_delta = data.delta[
            indices, time_index, None
        ].expand_as(selected_option)
        selected_strike = data.cash_strike[
            indices, time_index, None
        ].expand_as(selected_option)
        selected_cost = data.costs[
            indices, time_index, None
        ].expand_as(selected_action)
        values, tail_slope = payoff_cone_values(
            selected_action,
            spot=selected_spot,
            option_price=selected_option,
            delta=selected_delta,
            cash_strike=selected_strike,
            costs=selected_cost,
            hedge="raw",
        )
        slope_tolerance = (
            32.0
            * torch.finfo(action.dtype).eps
            * selected_action.abs().sum(dim=-1)
        )
        valid_slope = tail_slope >= -slope_tolerance
        expiry_lower = values.amin(dim=-1)
        expiry_lower = torch.where(
            valid_slope,
            expiry_lower,
            torch.full_like(expiry_lower, -torch.inf),
        )
        lower[:, :, time_index] = expiry_lower.sum(dim=-1)
        slope_ok[:, :, time_index] = valid_slope.all(dim=-1)
    return lower, slope_ok


def _powerexp(value: Tensor) -> tuple[Tensor, Tensor]:
    positive = torch.clamp_min(value, 0.0)
    negative = torch.clamp_max(value, 0.0)
    utility = torch.where(
        value > 0.0,
        -torch.expm1(-positive),
        negative - 0.5 * negative.square(),
    )
    marginal = torch.where(value > 0.0, torch.exp(-positive), 1.0 - negative)
    return utility, marginal


def _squared_downside(value: Tensor) -> Tensor:
    """Return the C1 squared loss below zero and zero otherwise."""

    return torch.clamp_max(value, 0.0).square()


def _bucket_downside_loss(
    pnl: Tensor,
    path_weight: Tensor,
    bucket_index: Tensor,
    *,
    buckets: int,
    gain_tiebreak: float,
    relative_temperature: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Minimize the worst weighted squared downside across path buckets."""

    assert pnl.shape[0] == path_weight.shape[0] == bucket_index.shape[0]
    assert bucket_index.dtype == torch.int64
    assert buckets > 1
    tail = pnl.shape[1:]
    weight = path_weight.reshape(-1, *(1 for _ in tail))
    expanded_bucket = bucket_index.reshape(-1, *(1 for _ in tail)).expand_as(pnl)
    bucket_sum = torch.zeros(
        (buckets, *tail), dtype=pnl.dtype, device=pnl.device
    )
    bucket_sum.scatter_add_(
        0, expanded_bucket, weight * _squared_downside(pnl)
    )
    bucket_weight = torch.zeros(
        buckets, dtype=pnl.dtype, device=pnl.device
    ).scatter_add_(0, bucket_index, path_weight)
    assert torch.all(bucket_weight > 0.0)
    bucket_risk = bucket_sum / bucket_weight.reshape(
        buckets, *(1 for _ in tail)
    )
    if relative_temperature > 0.0:
        scale = bucket_risk.mean(dim=0).detach().clamp_min(
            torch.finfo(pnl.dtype).tiny
        )
        temperature = relative_temperature * scale
        worst_risk = temperature * (
            torch.logsumexp(bucket_risk / temperature, dim=0)
            - math.log(buckets)
        )
    else:
        worst_risk = bucket_risk.amax(dim=0)
    physical = path_weight / path_weight.sum()
    mean_gain = (
        physical.reshape(-1, *(1 for _ in tail)) * torch.clamp_min(pnl, 0.0)
    ).sum(dim=0)
    loss = worst_risk.mean() - gain_tiebreak * mean_gain.mean()
    return loss, worst_risk, bucket_risk


def _normalized_marginal_weight(
    pnl: Tensor,
    path_weight: Tensor,
    notionals: Tensor,
    *,
    translation: Tensor | None = None,
    responsibility: Tensor | None = None,
) -> Tensor:
    """Normalize physical marginal-utility weights over the sample axis."""

    assert pnl.shape[0] == path_weight.shape[0]
    reshape = (1, notionals.numel(), *(1 for _ in range(pnl.ndim - 2)))
    scaled = pnl * notionals.reshape(reshape)
    if translation is not None:
        assert translation.shape == pnl.shape[1:]
        scaled = scaled + translation[None]
    nonpositive = torch.clamp_max(scaled, 0.0)
    log_marginal = torch.where(
        scaled > 0.0,
        -scaled,
        torch.log1p(-nonpositive),
    )
    extra = 0.0
    if responsibility is not None:
        assert responsibility.shape == pnl.shape
        extra = torch.log(responsibility.clamp_min(torch.finfo(pnl.dtype).tiny))
    log_density = (
        torch.log(path_weight)[:, None]
        .reshape(path_weight.shape[0], 1, *(1 for _ in range(pnl.ndim - 2)))
        + log_marginal
        + extra
    )
    return torch.softmax(log_density, dim=0)


def _feedforward_loss(
    model: (
        ParallelFeedForwardDirections
        | GatedParallelFeedForwardDirections
        | ParallelKernelDirections
    ),
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    notionals: Tensor,
    *,
    objective: str,
    bucket_index: Tensor | None,
    buckets: int,
    gain_tiebreak: float,
    bucket_temperature: float,
    portfolio_barrier_rate: float | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    action, logits = model(state)
    pnl = _parallel_pnl(
        action,
        returns,
        costs,
        portfolio_barrier_rate=portfolio_barrier_rate,
    )
    physical = path_weight / path_weight.sum()
    bucket_risk: Tensor | None = None
    if objective in {"powerexp", "oce-powerexp"}:
        use_translation = objective == "oce-powerexp"
        translation = model.oce_translation if use_translation else None
        assert (translation is not None) == use_translation
        marginal_weight = _normalized_marginal_weight(
            pnl,
            path_weight,
            notionals,
            translation=translation,
        )
        surrogate = (marginal_weight.detach() * pnl).sum(dim=0)
        scaled = pnl * notionals[None, :, None]
        if translation is not None:
            scaled = scaled + translation[None]
        utility, _ = _powerexp(scaled)
        reported_objective = (physical[:, None, None] * utility).sum(dim=0)
        loss = -surrogate.mean()
        translation_first_order_condition: Tensor | None = None
        if translation is not None:
            # Detaching P&L makes this term train y only.  The normalized
            # marginal surrogate above trains a without exponentially small
            # gradients and has the same action first-order condition.
            translated_detached = (
                pnl.detach() * notionals[None, :, None] + translation[None]
            )
            translation_utility, translation_marginal = _powerexp(
                translated_detached
            )
            translation_objective = (
                physical[:, None, None] * translation_utility
            ).sum(dim=0) - translation
            loss = loss - translation_objective.mean()
            reported_objective = reported_objective - translation
            translation_first_order_condition = (
                physical[:, None, None] * translation_marginal
            ).sum(dim=0) - 1.0
    elif objective == "downside2":
        downside = _squared_downside(pnl)
        risk = (physical[:, None, None] * downside).sum(dim=0)
        reported_objective = -risk
        loss = risk.mean()
    else:
        assert objective == "bucket-downside2" and bucket_index is not None
        loss, risk, bucket_risk = _bucket_downside_loss(
            pnl,
            path_weight,
            bucket_index,
            buckets=buckets,
            gain_tiebreak=gain_tiebreak,
            relative_temperature=bucket_temperature,
        )
        reported_objective = -risk
    result = {
        "action": action,
        "objective": reported_objective,
        "pnl": pnl,
        "raw_maximum": logits.abs().amax(dim=-1),
    }
    if objective == "oce-powerexp":
        assert model.oce_translation is not None
        assert translation_first_order_condition is not None
        result["oce_translation"] = model.oce_translation
        result["oce_translation_first_order_condition"] = (
            translation_first_order_condition
        )
    if bucket_risk is not None:
        result["bucket_downside"] = bucket_risk
    return loss, result


def _expert_pnl(
    expert_action: Tensor,
    returns: Tensor,
    costs: Tensor,
) -> Tensor:
    """Return conditional expert P&L with shape ``(S,N,K)``."""

    long = expert_action[..., 0, :, :]
    short = expert_action[..., 1, :, :]
    return (
        (long[None] - short[None]) * returns[:, None, None]
        - (long[None] + short[None]) * costs[:, None, None]
    ).sum(dim=(-1, -2))


def _partition_loss(
    model: LearnedPartitionDirections,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    notionals: Tensor,
    *,
    balance_coefficient: float,
    objective: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    action, responsibility, logits = model(state)
    deterministic_pnl = _parallel_pnl(
        action[:, :, :1], returns[:, :1], costs[:, :1]
    )[..., 0]
    expert_pnl = _expert_pnl(
        model.expert_actions(), returns[:, 1], costs[:, 1]
    )
    physical = path_weight / path_weight.sum()
    physical_responsibility = physical[:, None, None] * responsibility
    mass = physical_responsibility.sum(dim=0)
    balance = (
        model.config.partitions * mass - 1.0
    ).square().mean()
    actual_pnl = _parallel_pnl(action, returns, costs)
    if objective == "powerexp":
        deterministic_weight = _normalized_marginal_weight(
            deterministic_pnl, path_weight, notionals
        )
        deterministic_surrogate = (
            deterministic_weight.detach() * deterministic_pnl
        ).sum(dim=0)
        expert_weight = _normalized_marginal_weight(
            expert_pnl,
            path_weight,
            notionals,
            responsibility=responsibility,
        )
        expert_surrogate = (expert_weight.detach() * expert_pnl).sum(dim=0)
        expert_utility, _ = _powerexp(expert_pnl * notionals[None, :, None])
        router_objective = (
            physical_responsibility * expert_utility.detach()
        ).sum(dim=0) / mass.clamp_min(torch.finfo(mass.dtype).tiny)
        actual_utility, _ = _powerexp(actual_pnl * notionals[None, :, None])
        actual_objective = (physical[:, None, None] * actual_utility).sum(dim=0)
        loss = (
            -deterministic_surrogate.mean()
            -expert_surrogate.mean()
            -router_objective.mean()
            + balance_coefficient * balance
        )
    else:
        assert objective == "downside2"
        deterministic_risk = (
            physical[:, None] * _squared_downside(deterministic_pnl)
        ).sum(dim=0)
        expert_downside = _squared_downside(expert_pnl)
        expert_risk = (
            physical_responsibility * expert_downside
        ).sum(dim=0) / mass.clamp_min(torch.finfo(mass.dtype).tiny)
        # The detached local risk lets the state-only router learn regions while
        # the expert directions independently minimize downside within them.
        router_risk = (
            physical_responsibility * expert_downside.detach()
        ).sum(dim=0) / mass.clamp_min(torch.finfo(mass.dtype).tiny)
        actual_risk = (
            physical[:, None, None] * _squared_downside(actual_pnl)
        ).sum(dim=0)
        actual_objective = -actual_risk
        loss = (
            deterministic_risk.mean()
            + expert_risk.mean()
            + router_risk.mean()
            + balance_coefficient * balance
        )
    return loss, {
        "action": action,
        "objective": actual_objective,
        "pnl": actual_pnl,
        "raw_maximum": logits.abs().amax(dim=-1),
        "partition_mass": mass,
        "partition_balance": balance,
        "responsibility": responsibility,
    }


def _model_loss(
    model: DirectionModel,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    notionals: Tensor,
    *,
    balance_coefficient: float,
    objective: str = "powerexp",
    bucket_index: Tensor | None = None,
    buckets: int = 0,
    gain_tiebreak: float = 0.0,
    bucket_temperature: float = 0.0,
    portfolio_barrier_rate: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    assert objective in {
        "powerexp",
        "oce-powerexp",
        "downside2",
        "bucket-downside2",
    }
    if model.config.architecture == "partition":
        assert portfolio_barrier_rate is None
        assert objective not in {"oce-powerexp", "bucket-downside2"}
        return _partition_loss(
            model,
            state,
            returns,
            costs,
            path_weight,
            notionals,
            balance_coefficient=balance_coefficient,
            objective=objective,
        )
    return _feedforward_loss(
        model,
        state,
        returns,
        costs,
        path_weight,
        notionals,
        objective=objective,
        bucket_index=bucket_index,
        buckets=buckets,
        gain_tiebreak=gain_tiebreak,
        bucket_temperature=bucket_temperature,
        portfolio_barrier_rate=portfolio_barrier_rate,
    )


def _checkpoint(
    path: Path,
    *,
    model: DirectionModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    tracker: list[dict[str, Any]],
    completed: bool,
) -> None:
    _atomic_torch_save(
        path,
        {
            "model_code_version": type(model).version.unique_id48,
            "config_code_version": type(model.config).version.unique_id48,
            "user_version": model.config.user_version,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
            "tracker": tracker,
            "completed": completed,
        },
    )


def _stratified_bucket_assignment(
    data: DeviceExpiryCase,
    indices: Tensor,
    *,
    buckets: int,
    seed: int,
) -> Tensor:
    """Assign positions to physical-weight-balanced proposal-component buckets."""

    assert buckets > 1 and indices.numel() >= buckets
    component = data.component[indices].cpu().numpy()
    weight = data.path_weight[indices].double().cpu().numpy()
    assignment = _balanced_component_bucket_assignment(
        component,
        weight,
        buckets=buckets,
        seed=seed,
    )
    return torch.as_tensor(
        assignment, dtype=torch.int64, device=indices.device
    )


def _balanced_component_bucket_assignment(
    component: np.ndarray,
    path_weight: np.ndarray,
    *,
    buckets: int,
    seed: int,
) -> np.ndarray:
    """Greedily balance physical mass within each proposal component."""

    assert component.ndim == path_weight.ndim == 1
    assert component.shape == path_weight.shape and component.size >= buckets
    assert np.all(np.isfinite(path_weight)) and np.all(path_weight > 0.0)
    generator = np.random.default_rng(seed)
    assignment = np.full(component.shape, -1, dtype=np.int64)
    for value in np.unique(component):
        positions = np.flatnonzero(component == value)
        generator.shuffle(positions)
        order = np.argsort(-path_weight[positions], kind="stable")
        positions = positions[order]
        priority = generator.permutation(buckets)
        heap = [(0.0, int(priority[bucket]), bucket) for bucket in range(buckets)]
        heapq.heapify(heap)
        for position in positions:
            mass, tie_break, bucket = heapq.heappop(heap)
            assignment[position] = bucket
            heapq.heappush(
                heap,
                (mass + float(path_weight[position]), tie_break, bucket),
            )
    assert np.all(assignment >= 0)
    assert np.unique(assignment).size == buckets
    return assignment


def _stratified_bucket_members(
    data: DeviceExpiryCase,
    *,
    buckets: int,
    seed: int,
) -> list[Tensor]:
    """Partition training positions while spreading proposal components."""

    assignment = _stratified_bucket_assignment(
        data, data.reference, buckets=buckets, seed=seed
    )
    members = [
        torch.nonzero(assignment == bucket, as_tuple=False).flatten()
        for bucket in range(buckets)
    ]
    assert all(value.numel() > 0 for value in members)
    return members


def _draw_bucketed_indices(
    reference: Tensor,
    members: list[Tensor],
    *,
    size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Draw an equal-size weighted-training sample from every fixed bucket."""

    buckets = len(members)
    assert size >= buckets
    quotient, remainder = divmod(size, buckets)
    draws: list[Tensor] = []
    labels: list[Tensor] = []
    for bucket, positions in enumerate(members):
        count = quotient + int(bucket < remainder)
        selected = positions[
            torch.randint(
                positions.numel(),
                (count,),
                generator=generator,
                device=positions.device,
            )
        ]
        draws.append(reference[selected])
        labels.append(
            torch.full(
                (count,), bucket, dtype=torch.int64, device=reference.device
            )
        )
    return torch.cat(draws), torch.cat(labels)


def _train(
    model: DirectionModel,
    callable_model: DirectionModel,
    data: DeviceExpiryCase,
    notionals: Tensor,
    *,
    checkpoint_path: Path,
    batch_size: int,
    valuation_batch_size: int,
    epochs: int,
    value_update: int,
    checkpoint_update: int,
    lr: float,
    seed: int,
    balance_coefficient: float,
    objective: str,
    buckets: int,
    gain_tiebreak: float,
    bucket_temperature: float,
) -> dict[str, Any]:
    """Cold-train every notional simultaneously on one half sample."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        compatible = (
            saved.get("model_code_version") == type(model).version.unique_id48
            and saved.get("config_code_version")
            == type(model.config).version.unique_id48
            and saved.get("user_version") == model.config.user_version
        )
        if compatible:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator_state"].cpu())
            tracker = list(saved.get("tracker", []))
            start_step = int(saved["step"])
            if bool(saved.get("completed")) and start_step >= epochs:
                calibration = (
                    _calibrate_model_oce_translation(
                        model,
                        data,
                        data.reference,
                        tuple(float(value) for value in notionals.cpu().tolist()),
                        batch_size=valuation_batch_size,
                    )
                    if objective == "oce-powerexp"
                    else None
                )
                if calibration is not None:
                    _checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        generator=generator,
                        step=start_step,
                        tracker=tracker,
                        completed=True,
                    )
                return {
                    "steps": start_step,
                    "resumed_complete_checkpoint": True,
                    "final_oce_translation_calibration": calibration,
                    "tracker": tracker,
                }
            if bool(saved.get("completed")):
                print(
                    f"Extending completed checkpoint {checkpoint_path} from "
                    f"{start_step:,} to {epochs:,} steps",
                    flush=True,
                )
        else:
            print(f"Ignoring incompatible checkpoint {checkpoint_path}", flush=True)

    bucket_members = (
        _stratified_bucket_members(data, buckets=buckets, seed=seed + 20_000)
        if objective == "bucket-downside2"
        else None
    )
    fixed_generator = torch.Generator(device=model.device)
    fixed_generator.manual_seed(seed + 10_000)
    fixed_size = min(valuation_batch_size, data.reference.numel())
    if bucket_members is None:
        fixed_draw = torch.randperm(
            data.reference.numel(), generator=fixed_generator, device=model.device
        )[:fixed_size]
        fixed = data.reference[fixed_draw]
        fixed_bucket = None
    else:
        fixed, fixed_bucket = _draw_bucketed_indices(
            data.reference,
            bucket_members,
            size=fixed_size,
            generator=fixed_generator,
        )
    started = time.monotonic()
    for step in range(start_step + 1, epochs + 1):
        if bucket_members is None:
            draw = torch.randint(
                data.reference.numel(),
                (batch_size,),
                generator=generator,
                device=model.device,
            )
            indices = data.reference[draw]
            batch_bucket = None
        else:
            indices, batch_bucket = _draw_bucketed_indices(
                data.reference,
                bucket_members,
                size=batch_size,
                generator=generator,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _model_loss(
            callable_model,
            data.state[indices, 1],
            data.returns[indices],
            data.costs[indices],
            data.path_weight[indices],
            notionals,
            balance_coefficient=balance_coefficient,
            objective=objective,
            bucket_index=batch_bucket,
            buckets=buckets,
            gain_tiebreak=gain_tiebreak,
            bucket_temperature=bucket_temperature,
            portfolio_barrier_rate=data.portfolio_barrier_rate,
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("parallel direction gradient is non-finite")
        optimizer.step()
        model.project_parameters_()

        if step % value_update == 0 or step == epochs:
            with torch.no_grad():
                fixed_loss, values = _model_loss(
                    model,
                    data.state[fixed, 1],
                    data.returns[fixed],
                    data.costs[fixed],
                    data.path_weight[fixed],
                    notionals,
                    balance_coefficient=balance_coefficient,
                    objective=objective,
                    bucket_index=fixed_bucket,
                    buckets=buckets,
                    gain_tiebreak=gain_tiebreak,
                    bucket_temperature=bucket_temperature,
                    portfolio_barrier_rate=data.portfolio_barrier_rate,
                )
            item: dict[str, Any] = {
                "step": step,
                "loss": float(fixed_loss.cpu()),
                "objective_by_notional_and_time": values["objective"].cpu().tolist(),
                "raw_maximum_by_notional": values["raw_maximum"].amax(dim=0).cpu().tolist(),
                "elapsed_seconds": time.monotonic() - started,
            }
            if "partition_mass" in values:
                item["partition_mass_minimum_by_notional"] = (
                    values["partition_mass"].amin(dim=-1).cpu().tolist()
                )
                item["partition_mass_maximum_by_notional"] = (
                    values["partition_mass"].amax(dim=-1).cpu().tolist()
                )
                item["partition_balance"] = float(
                    values["partition_balance"].cpu()
                )
            if "bucket_downside" in values:
                item["maximum_bucket_downside_by_notional_and_time"] = (
                    values["bucket_downside"].amax(dim=0).cpu().tolist()
                )
            if "oce_translation" in values:
                item["oce_translation_by_notional_and_time"] = (
                    values["oce_translation"].cpu().tolist()
                )
                item["oce_translation_first_order_condition"] = (
                    values["oce_translation_first_order_condition"].cpu().tolist()
                )
            tracker.append(item)
            print(
                f"step {step:,}: loss={item['loss']:.7g}, "
                f"objectives={item['objective_by_notional_and_time']}",
                flush=True,
            )
            if step % checkpoint_update == 0:
                _checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    step=step,
                    tracker=tracker,
                    completed=False,
                )

    calibration = (
        _calibrate_model_oce_translation(
            model,
            data,
            data.reference,
            tuple(float(value) for value in notionals.cpu().tolist()),
            batch_size=valuation_batch_size,
        )
        if objective == "oce-powerexp"
        else None
    )
    _checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        step=epochs,
        tracker=tracker,
        completed=True,
    )
    return {
        "steps": epochs,
        "resumed_complete_checkpoint": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "constant_learning_rate": lr,
        "gradient_clipping": False,
        "cold_independent_notional_parameters": True,
        "optimization_objective": objective,
        "buckets": buckets if objective == "bucket-downside2" else None,
        "gain_tiebreak": gain_tiebreak,
        "bucket_relative_temperature": bucket_temperature,
        "final_oce_translation_calibration": calibration,
        "tracker": tracker,
    }


def _weighted_mean_error(
    value: np.ndarray, normalized_weight: np.ndarray
) -> tuple[float, float]:
    mean = float(np.dot(normalized_weight, value))
    error = float(
        np.sqrt(np.sum(np.square(normalized_weight * (value - mean))))
    )
    return mean, error


def _finite_maximum_or_none(value: np.ndarray) -> float | None:
    """Return the largest finite value, leaving unbounded diagnostics as JSON null."""

    finite = np.isfinite(value)
    return float(np.max(value[finite])) if np.any(finite) else None


def _bucket_pnl_audit(
    pnl: np.ndarray,
    weight: np.ndarray,
    assignment: np.ndarray,
    *,
    buckets: int,
) -> dict[str, Any]:
    """Audit P&L means and squared downside within every fixed path bucket."""

    assert pnl.ndim == 3 and weight.shape == assignment.shape == pnl.shape[:1]
    bucket_mean = np.empty((buckets, *pnl.shape[1:]), dtype=np.float64)
    bucket_downside = np.empty_like(bucket_mean)
    bucket_loss_probability = np.empty_like(bucket_mean)
    bucket_paths: list[int] = []
    for bucket in range(buckets):
        selected = assignment == bucket
        bucket_paths.append(int(selected.sum()))
        assert selected.any()
        local_weight = weight[selected]
        local_weight = local_weight / local_weight.sum()
        local_pnl = pnl[selected]
        shaped_weight = local_weight.reshape(-1, 1, 1)
        bucket_mean[bucket] = np.sum(shaped_weight * local_pnl, axis=0)
        bucket_downside[bucket] = np.sum(
            shaped_weight * np.square(np.minimum(local_pnl, 0.0)), axis=0
        )
        bucket_loss_probability[bucket] = np.sum(
            shaped_weight * (local_pnl < 0.0), axis=0
        )
    return {
        "paths_per_bucket_minimum": min(bucket_paths),
        "paths_per_bucket_maximum": max(bucket_paths),
        "minimum_bucket_P_mean_pnl_by_notional_and_time": bucket_mean.min(
            axis=0
        ).tolist(),
        "maximum_bucket_P_squared_downside_by_notional_and_time": (
            bucket_downside.max(axis=0).tolist()
        ),
        "maximum_bucket_P_loss_probability_by_notional_and_time": (
            bucket_loss_probability.max(axis=0).tolist()
        ),
    }


def _powerexp_numpy(value: np.ndarray) -> np.ndarray:
    positive = np.maximum(value, 0.0)
    negative = np.minimum(value, 0.0)
    return np.where(
        value > 0.0,
        -np.expm1(-positive),
        negative - 0.5 * np.square(negative),
    )


def _powerexp_marginal_numpy(value: np.ndarray) -> np.ndarray:
    """Evaluate PowerExp's positive marginal without eager branch overflow."""

    result = np.empty_like(value, dtype=np.float64)
    positive = value > 0.0
    result[positive] = np.exp(-value[positive])
    result[~positive] = 1.0 - value[~positive]
    return result


def _recalibrate_oce_translation(
    value: np.ndarray,
    normalized_weight: np.ndarray,
    notional: float,
) -> float:
    """Solve E_P[u'(y+n*pi)]=1 for one scalar translation y."""

    assert value.ndim == normalized_weight.ndim == 1
    assert value.shape == normalized_weight.shape
    scaled = notional * value.astype(np.float64, copy=False)
    lower = float(-np.max(scaled) - 1.0)
    upper = float(-np.min(scaled) + 1.0)

    def condition(translation: float) -> float:
        marginal = _powerexp_marginal_numpy(translation + scaled)
        return float(np.dot(normalized_weight, marginal) - 1.0)

    assert condition(lower) > 0.0 and condition(upper) < 0.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if condition(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _oce_numpy_statistics(
    value: np.ndarray,
    weight: np.ndarray,
    notional: float,
    translation: float,
) -> dict[str, float | np.ndarray]:
    """Return OCE value and both scalar and action first-order conditions."""

    normalized_weight = weight / weight.sum()
    shifted = translation + notional * value
    marginal = _powerexp_marginal_numpy(shifted)
    marginal_weight = normalized_weight * marginal
    marginal_expectation = float(marginal_weight.sum())
    marginal_weight /= marginal_expectation
    derivative, derivative_error = _weighted_mean_error(value, marginal_weight)
    objective, objective_error = _weighted_mean_error(
        _powerexp_numpy(shifted), normalized_weight
    )
    return {
        "objective": objective - translation,
        "objective_standard_error": objective_error,
        "translation_first_order_condition": marginal_expectation - 1.0,
        "normalized_notional_derivative": derivative,
        "normalized_notional_derivative_standard_error": derivative_error,
        "marginal_weight": marginal_weight,
    }


@torch.no_grad()
def _calibrate_model_oce_translation(
    model: DirectionModel,
    data: DeviceExpiryCase,
    indices: Tensor,
    notionals: tuple[float, ...],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Exactly calibrate each scalar y on the full frozen-action training set."""

    verify_inp(model.oce_translation is not None, "model has no OCE translation")
    paths = int(indices.numel())
    pnl = np.empty((paths, len(notionals), 2), dtype=np.float64)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(data.state[batch, 1])[0]
        pnl[start:stop] = _parallel_pnl(
            action,
            data.returns[batch],
            data.costs[batch],
            portfolio_barrier_rate=data.portfolio_barrier_rate,
        ).cpu().double().numpy()
    weight = data.path_weight[indices].cpu().double().numpy()
    normalized_weight = weight / weight.sum()
    before = model.oce_translation.detach().cpu().double().numpy()
    calibrated = np.empty_like(before)
    residual = np.empty_like(before)
    for notional_index, notional in enumerate(notionals):
        for time_index in range(2):
            value = pnl[:, notional_index, time_index]
            calibrated[notional_index, time_index] = _recalibrate_oce_translation(
                value,
                normalized_weight,
                notional,
            )
            residual[notional_index, time_index] = _oce_numpy_statistics(
                value,
                weight,
                notional,
                calibrated[notional_index, time_index],
            )["translation_first_order_condition"]
    model.oce_translation.copy_(
        torch.as_tensor(
            calibrated,
            dtype=model.fdtype,
            device=model.device,
        )
    )
    return {
        "definition": (
            "one-dimensional full-training-set solution of E_P[u'(y+n*pi)]=1 "
            "after freezing the action"
        ),
        "before": before.tolist(),
        "after": calibrated.tolist(),
        "maximum_absolute_first_order_residual": float(np.max(np.abs(residual))),
    }


def _activity_threshold_report(
    value: np.ndarray,
    tolerance: np.ndarray,
    activity: np.ndarray,
    weight: np.ndarray,
    direct: np.ndarray,
    *,
    notional: float,
    exact_lower: np.ndarray | None,
    use_oce_translation: bool,
) -> list[dict[str, Any]]:
    """Audit the admissible hard gate obtained from a learned soft activity."""

    normalized_weight = weight / weight.sum()
    result: list[dict[str, Any]] = []
    for threshold in (0.01, 0.1, 0.5, 0.9, 0.99):
        active = activity >= threshold
        gated_value = np.where(active, value, 0.0)
        gated_tolerance = np.where(active, tolerance, 0.0)
        loss = gated_value < -gated_tolerance
        active_mass = float(np.dot(normalized_weight, active.astype(np.float64)))
        if active.any() and active_mass > 0.0:
            active_weight = normalized_weight[active] / active_mass
            conditional_mean, conditional_mean_error = _weighted_mean_error(
                value[active], active_weight
            )
            conditional_loss, conditional_loss_error = _weighted_mean_error(
                (value[active] < -tolerance[active]).astype(np.float64),
                active_weight,
            )
        else:
            conditional_mean = conditional_mean_error = 0.0
            conditional_loss = conditional_loss_error = 0.0
        translation = (
            _recalibrate_oce_translation(
                gated_value,
                normalized_weight,
                notional,
            )
            if use_oce_translation
            else 0.0
        )
        utility_statistics = _oce_numpy_statistics(
            gated_value,
            weight,
            notional,
            translation,
        )
        result.append(
            {
                "activity_threshold": threshold,
                "active_path_count": int(active.sum()),
                "direct_P_active_path_count": int(np.sum(active & direct)),
                "P_active_probability": active_mass,
                "P_mean_pnl": _weighted_mean_error(
                    gated_value, normalized_weight
                )[0],
                "P_active_conditional_mean_pnl": conditional_mean,
                "P_active_conditional_mean_pnl_standard_error": (
                    conditional_mean_error
                ),
                "P_loss_probability": _weighted_mean_error(
                    loss.astype(np.float64), normalized_weight
                )[0],
                "P_active_conditional_loss_probability": conditional_loss,
                "P_active_conditional_loss_probability_standard_error": (
                    conditional_loss_error
                ),
                "objective": utility_statistics["objective"],
                "objective_standard_error": utility_statistics[
                    "objective_standard_error"
                ],
                "oce_translation": translation if use_oce_translation else None,
                "oce_translation_first_order_condition": (
                    utility_statistics["translation_first_order_condition"]
                    if use_oce_translation
                    else None
                ),
                "normalized_notional_derivative": utility_statistics[
                    "normalized_notional_derivative"
                ],
                "normalized_notional_derivative_standard_error": (
                    utility_statistics[
                        "normalized_notional_derivative_standard_error"
                    ]
                ),
                "proposal_minimum_pnl_plus_band": float(
                    np.min(gated_value + gated_tolerance)
                ),
                "exact_payoff_cone": (
                    None
                    if exact_lower is None
                    else {
                        "active_numerically_nonnegative_count": int(
                            np.sum(active & (exact_lower >= -tolerance))
                        ),
                        "active_strict_nonnegative_count": int(
                            np.sum(active & (exact_lower >= 0.0))
                        ),
                        "all_active_numerically_nonnegative": bool(
                            active.any()
                            and np.all(exact_lower[active] >= -tolerance[active])
                        ),
                    }
                ),
            }
        )
    return result


@torch.no_grad()
def _evaluate_partition_candidates(
    model: LearnedPartitionDirections,
    data: DeviceExpiryCase,
    indices: Tensor,
    notionals: tuple[float, ...],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Audit each hard-routed local direction as a separate conditional strategy."""

    paths = int(indices.numel())
    num_notionals = len(notionals)
    partitions = model.config.partitions
    pnl = np.empty((paths, num_notionals, partitions), dtype=np.float64)
    band = np.empty_like(pnl)
    cone_lower = (
        np.empty_like(pnl) if data.option_price is not None else None
    )
    assignment = np.empty((paths, num_notionals), dtype=np.int16)
    expert_action = model.expert_actions()
    expert_gross = expert_action[..., 0, :, :] + expert_action[..., 1, :, :]
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        factor = model.factor(data.state[batch, 1])
        responsibility, _ = model.responsibilities(factor)
        assignment[start:stop] = responsibility.argmax(dim=-1).cpu().numpy()
        batch_returns = data.returns[batch, 1]
        batch_costs = data.costs[batch, 1]
        pnl[start:stop] = _expert_pnl(
            expert_action, batch_returns, batch_costs
        ).cpu().double().numpy()
        accumulation = 32.0 * torch.finfo(torch.float32).eps * (
            expert_gross[None]
            * (batch_returns[:, None, None].abs() + batch_costs[:, None, None])
        ).sum(dim=(-1, -2))
        batch_band = (
            expert_gross[None] * data.numerical_band[batch, 1, None, None]
        ).sum(dim=(-1, -2)) + accumulation
        band[start:stop] = batch_band.cpu().double().numpy()
        if cone_lower is not None:
            assert (
                data.option_price is not None
                and data.delta is not None
                and data.cash_strike is not None
                and data.spot is not None
            )
            flat_action = (
                expert_action[..., 0, :, :] - expert_action[..., 1, :, :]
            ).reshape(num_notionals * partitions, *expert_action.shape[-2:])
            heads = num_notionals * partitions
            flat_lower = torch.empty(
                (batch.numel(), heads),
                dtype=expert_action.dtype,
                device=expert_action.device,
            )
            for head_start in range(0, heads, 16):
                head_stop = min(head_start + 16, heads)
                selected_action = flat_action[None, head_start:head_stop].expand(
                    batch.numel(), -1, -1, -1
                )
                selected_spot = data.spot[batch, 1, None].expand(
                    -1, head_stop - head_start
                )
                selected_option = data.option_price[
                    batch, 1, None
                ].expand(-1, head_stop - head_start, -1, -1)
                values, tail_slope = payoff_cone_values(
                    selected_action,
                    spot=selected_spot,
                    option_price=selected_option,
                    delta=data.delta[batch, 1, None].expand_as(selected_option),
                    cash_strike=data.cash_strike[batch, 1, None].expand_as(
                        selected_option
                    ),
                    costs=data.costs[batch, 1, None].expand_as(selected_action),
                    hedge="raw",
                )
                slope_tolerance = (
                    32.0
                    * torch.finfo(expert_action.dtype).eps
                    * selected_action.abs().sum(dim=-1)
                )
                expiry_lower = values.amin(dim=-1)
                expiry_lower = torch.where(
                    tail_slope >= -slope_tolerance,
                    expiry_lower,
                    torch.full_like(expiry_lower, -torch.inf),
                )
                flat_lower[:, head_start:head_stop] = expiry_lower.sum(dim=-1)
            cone_lower[start:stop] = (
                flat_lower.reshape(batch.numel(), num_notionals, partitions)
                .cpu()
                .double()
                .numpy()
            )

    weight = data.path_weight[indices].cpu().double().numpy()
    component = data.component[indices].cpu().numpy()
    total_weight = weight.sum()
    reports: list[list[dict[str, Any]]] = []
    for notional_index, notional in enumerate(notionals):
        by_partition: list[dict[str, Any]] = []
        for partition in range(partitions):
            selected = assignment[:, notional_index] == partition
            selected_paths = int(selected.sum())
            if selected_paths == 0:
                by_partition.append(
                    {
                        "partition": partition,
                        "paths": 0,
                        "physical_mass": 0.0,
                        "physical_ESS": 0.0,
                        "numerically_nonnegative": False,
                    }
                )
                continue
            local_weight = weight[selected]
            normalized_weight = local_weight / local_weight.sum()
            value = pnl[selected, notional_index, partition]
            tolerance = band[selected, notional_index, partition]
            loss = value < -tolerance
            gain = value > tolerance
            scaled = notional * value
            log_marginal = np.where(
                scaled > 0.0,
                -scaled,
                np.log1p(-np.minimum(scaled, 0.0)),
            )
            log_density = np.log(local_weight) + log_marginal
            log_density -= logsumexp(log_density)
            derivative, derivative_error = _weighted_mean_error(
                value, np.exp(log_density)
            )
            mean, mean_error = _weighted_mean_error(value, normalized_weight)
            loss_probability, loss_error = _weighted_mean_error(
                loss.astype(np.float64), normalized_weight
            )
            by_partition.append(
                {
                    "partition": partition,
                    "paths": selected_paths,
                    "direct_P_paths": int(np.sum(component[selected] == 0)),
                    "physical_mass": float(local_weight.sum() / total_weight),
                    "physical_ESS": float(
                        np.square(local_weight.sum()) / np.square(local_weight).sum()
                    ),
                    "P_mean_unit_pnl": mean,
                    "P_mean_unit_pnl_standard_error": mean_error,
                    "P_loss_probability": loss_probability,
                    "P_loss_probability_standard_error": loss_error,
                    "proposal_loss_count": int(loss.sum()),
                    "direct_P_loss_count": int(
                        np.sum(loss & (component[selected] == 0))
                    ),
                    "P_gain_probability": _weighted_mean_error(
                        gain.astype(np.float64), normalized_weight
                    )[0],
                    "proposal_minimum_unit_pnl_plus_band": float(
                        np.min(value + tolerance)
                    ),
                    "numerically_nonnegative": bool(np.all(~loss)),
                    "normalized_notional_derivative": derivative,
                    "normalized_notional_derivative_standard_error": (
                        derivative_error
                    ),
                    "exact_payoff_cone": (
                        None
                        if cone_lower is None
                        else {
                            "unbounded_below_path_count": int(
                                np.sum(
                                    ~np.isfinite(
                                        cone_lower[
                                            selected, notional_index, partition
                                        ]
                                    )
                                )
                            ),
                            "nonnegative_path_count": int(
                                np.sum(
                                    cone_lower[
                                        selected, notional_index, partition
                                    ]
                                    >= -tolerance
                                )
                            ),
                            "strict_nonnegative_path_count": int(
                                np.sum(
                                    cone_lower[
                                        selected, notional_index, partition
                                    ]
                                    >= 0.0
                                )
                            ),
                            "maximum_finite_lower_bound": _finite_maximum_or_none(
                                cone_lower[selected, notional_index, partition]
                            ),
                            "P_numerically_nonnegative_probability": (
                                _weighted_mean_error(
                                    (
                                        cone_lower[
                                            selected, notional_index, partition
                                        ]
                                        >= -tolerance
                                    ).astype(np.float64),
                                    normalized_weight,
                                )[0]
                            ),
                            "all_selected_paths_nonnegative": bool(
                                np.all(
                                    cone_lower[
                                        selected, notional_index, partition
                                    ]
                                    >= -tolerance
                                )
                            ),
                        }
                    ),
                }
            )
        reports.append(by_partition)
    return {
        "definition": (
            "for each notional and learned partition, trade its unit-gross local "
            "direction only when that partition has the largest state-only router "
            "probability; all other states hold zero"
        ),
        "by_notional_and_partition": reports,
    }


@torch.no_grad()
def _evaluate(
    model: DirectionModel,
    data: DeviceExpiryCase,
    indices: Tensor,
    notionals: tuple[float, ...],
    *,
    batch_size: int,
    objective: str,
    evaluation_notionals: tuple[float, ...] | None = None,
    audit_buckets: int | None = None,
    bucket_seed: int = 0,
) -> dict[str, Any]:
    paths = int(indices.numel())
    num_notionals = len(notionals)
    pnl = np.empty((paths, num_notionals, 2), dtype=np.float64)
    band = np.empty_like(pnl)
    gross = np.empty_like(pnl)
    exact_lower = (
        np.empty_like(pnl) if data.option_price is not None else None
    )
    exact_slope_ok = (
        np.empty(pnl.shape, dtype=np.bool_) if data.option_price is not None else None
    )
    responsibility: np.ndarray | None = None
    if isinstance(model, LearnedPartitionDirections):
        responsibility = np.empty(
            (paths, num_notionals, model.config.partitions), dtype=np.float32
        )
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        if isinstance(model, LearnedPartitionDirections):
            action, batch_responsibility, _ = model(data.state[batch, 1])
            assert responsibility is not None
            responsibility[start:stop] = batch_responsibility.cpu().numpy()
        else:
            action, _ = model(data.state[batch, 1])
        pnl[start:stop] = _parallel_pnl(
            action,
            data.returns[batch],
            data.costs[batch],
            portfolio_barrier_rate=data.portfolio_barrier_rate,
        ).cpu().double().numpy()
        gross_action = (
            (action[..., 0, :, :] - action[..., 1, :, :]).abs()
            if data.portfolio_barrier_rate is not None
            else action[..., 0, :, :] + action[..., 1, :, :]
        )
        gross[start:stop] = gross_action.sum(
            dim=(-1, -2)
        ).cpu().double().numpy()
        band[start:stop] = _parallel_band(
            action,
            data.returns[batch],
            data.costs[batch],
            data.numerical_band[batch],
            portfolio_barrier_rate=data.portfolio_barrier_rate,
        ).cpu().double().numpy()
        exact = _exact_expiry_lower_bound(action, data, batch)
        if exact is not None:
            assert exact_lower is not None and exact_slope_ok is not None
            batch_lower, batch_slope_ok = exact
            exact_lower[start:stop] = batch_lower.cpu().double().numpy()
            exact_slope_ok[start:stop] = batch_slope_ok.cpu().numpy()

    weight = data.path_weight[indices].cpu().double().numpy()
    normalized_weight = weight / weight.sum()
    component = data.component[indices].cpu().numpy()
    direct = component == 0
    use_oce_translation = objective == "oce-powerexp"
    learned_translation = (
        model.oce_translation.detach().cpu().double().numpy()
        if use_oce_translation
        else np.zeros((num_notionals, 2), dtype=np.float64)
    )
    assert learned_translation.shape == (num_notionals, 2)
    by_notional: list[dict[str, Any]] = []
    for notional_index, notional in enumerate(notionals):
        by_time: list[dict[str, Any]] = []
        for time_index, name in enumerate(("t0_deterministic", "t6_conditional")):
            value = pnl[:, notional_index, time_index]
            tolerance = band[:, notional_index, time_index]
            loss = value < -tolerance
            gain = value > tolerance
            translation = float(learned_translation[notional_index, time_index])
            learned_statistics = _oce_numpy_statistics(
                value,
                weight,
                notional,
                translation,
            )
            recalibrated_translation = (
                _recalibrate_oce_translation(value, normalized_weight, notional)
                if use_oce_translation
                else 0.0
            )
            recalibrated_statistics = _oce_numpy_statistics(
                value,
                weight,
                notional,
                recalibrated_translation,
            )
            mean, mean_error = _weighted_mean_error(value, normalized_weight)
            loss_probability, loss_error = _weighted_mean_error(
                loss.astype(np.float64), normalized_weight
            )
            marginal_weight = learned_statistics["marginal_weight"]
            assert isinstance(marginal_weight, np.ndarray)
            by_time.append(
                {
                    "name": name,
                    "objective": learned_statistics["objective"],
                    "objective_standard_error": learned_statistics[
                        "objective_standard_error"
                    ],
                    "learned_oce_translation": (
                        translation if use_oce_translation else None
                    ),
                    "learned_oce_translation_first_order_condition": (
                        learned_statistics["translation_first_order_condition"]
                        if use_oce_translation
                        else None
                    ),
                    "recalibrated_oce_translation": (
                        recalibrated_translation if use_oce_translation else None
                    ),
                    "recalibrated_oce_objective": (
                        recalibrated_statistics["objective"]
                        if use_oce_translation
                        else None
                    ),
                    "recalibrated_oce_translation_first_order_condition": (
                        recalibrated_statistics[
                            "translation_first_order_condition"
                        ]
                        if use_oce_translation
                        else None
                    ),
                    "recalibrated_normalized_notional_derivative": (
                        recalibrated_statistics["normalized_notional_derivative"]
                        if use_oce_translation
                        else None
                    ),
                    "recalibrated_normalized_notional_derivative_standard_error": (
                        recalibrated_statistics[
                            "normalized_notional_derivative_standard_error"
                        ]
                        if use_oce_translation
                        else None
                    ),
                    "P_mean_unit_pnl": mean,
                    "P_mean_unit_pnl_standard_error": mean_error,
                    "P_loss_probability": loss_probability,
                    "P_loss_probability_standard_error": loss_error,
                    "P_gain_probability": _weighted_mean_error(
                        gain.astype(np.float64), normalized_weight
                    )[0],
                    "P_mean_gross_activity": _weighted_mean_error(
                        gross[:, notional_index, time_index], normalized_weight
                    )[0],
                    "P_gross_activity_01_quantile": weighted_quantile(
                        gross[:, notional_index, time_index], weight, 0.01
                    ),
                    "P_gross_activity_median": weighted_quantile(
                        gross[:, notional_index, time_index], weight, 0.50
                    ),
                    "P_gross_activity_99_quantile": weighted_quantile(
                        gross[:, notional_index, time_index], weight, 0.99
                    ),
                    "activity_threshold_audit": _activity_threshold_report(
                        value,
                        tolerance,
                        gross[:, notional_index, time_index],
                        weight,
                        direct,
                        notional=notional,
                        use_oce_translation=use_oce_translation,
                        exact_lower=(
                            None
                            if exact_lower is None
                            else exact_lower[:, notional_index, time_index]
                        ),
                    ),
                    "P_unit_pnl_01_quantile": weighted_quantile(
                        value, weight, 0.01
                    ),
                    "P_unit_pnl_05_quantile": weighted_quantile(
                        value, weight, 0.05
                    ),
                    "P_unit_pnl_median": weighted_quantile(value, weight, 0.50),
                    "proposal_minimum_unit_pnl_plus_band": float(
                        np.min(value + tolerance)
                    ),
                    "proposal_numerically_nonnegative": bool(np.all(~loss)),
                    "direct_P_loss_count": int(np.sum(loss & direct)),
                    "direct_P_gain_count": int(np.sum(gain & direct)),
                    "normalized_notional_derivative": learned_statistics[
                        "normalized_notional_derivative"
                    ],
                    "normalized_notional_derivative_standard_error": (
                        learned_statistics[
                            "normalized_notional_derivative_standard_error"
                        ]
                    ),
                    "marginal_density_effective_sample_size": float(
                        1.0 / np.square(marginal_weight).sum()
                    ),
                    "exact_payoff_cone": (
                        None
                        if exact_lower is None
                        else {
                            "unbounded_below_path_count": int(
                                np.sum(
                                    ~np.isfinite(
                                        exact_lower[
                                            :, notional_index, time_index
                                        ]
                                    )
                                )
                            ),
                            "minimum_finite_lower_bound": (
                                None
                                if not np.any(
                                    np.isfinite(
                                        exact_lower[
                                            :, notional_index, time_index
                                        ]
                                    )
                                )
                                else float(
                                    np.min(
                                        exact_lower[
                                            :, notional_index, time_index
                                        ][
                                            np.isfinite(
                                                exact_lower[
                                                    :, notional_index, time_index
                                                ]
                                            )
                                        ]
                                    )
                                )
                            ),
                            "nonnegative_path_count": int(
                                np.sum(
                                    exact_lower[:, notional_index, time_index]
                                    >= -band[:, notional_index, time_index]
                                )
                            ),
                            "strict_nonnegative_path_count": int(
                                np.sum(
                                    exact_lower[:, notional_index, time_index]
                                    >= 0.0
                                )
                            ),
                            "maximum_finite_lower_bound": _finite_maximum_or_none(
                                exact_lower[:, notional_index, time_index]
                            ),
                            "P_numerically_nonnegative_probability": (
                                _weighted_mean_error(
                                    (
                                        exact_lower[
                                            :, notional_index, time_index
                                        ]
                                        >= -band[
                                            :, notional_index, time_index
                                        ]
                                    ).astype(np.float64),
                                    normalized_weight,
                                )[0]
                            ),
                            "direct_P_nonnegative_path_count": int(
                                np.sum(
                                    (
                                        exact_lower[:, notional_index, time_index]
                                        >= -band[:, notional_index, time_index]
                                    )
                                    & direct
                                )
                            ),
                            "all_expiry_tail_slopes_valid_count": int(
                                np.sum(
                                    exact_slope_ok[
                                        :, notional_index, time_index
                                    ]
                                )
                            ),
                        }
                    ),
                }
            )
        by_notional.append({"notional": notional, "times": by_time})

    evaluation_notionals = (
        notionals if evaluation_notionals is None else evaluation_notionals
    )
    cross_objective: list[list[list[float]]] = []
    cross_derivative: list[list[list[float]]] = []
    cross_derivative_error: list[list[list[float]]] = []
    for direction_index in range(num_notionals):
        by_scale: list[list[float]] = []
        derivative_by_scale: list[list[float]] = []
        derivative_error_by_scale: list[list[float]] = []
        for scale in evaluation_notionals:
            scale_objective: list[float] = []
            scale_derivative: list[float] = []
            scale_derivative_error: list[float] = []
            for time_index in range(2):
                value = pnl[:, direction_index, time_index]
                translation = (
                    _recalibrate_oce_translation(
                        value,
                        normalized_weight,
                        scale,
                    )
                    if use_oce_translation
                    else 0.0
                )
                statistics = _oce_numpy_statistics(
                    value,
                    weight,
                    scale,
                    translation,
                )
                scale_objective.append(float(statistics["objective"]))
                scale_derivative.append(
                    float(statistics["normalized_notional_derivative"])
                )
                scale_derivative_error.append(
                    float(
                        statistics[
                            "normalized_notional_derivative_standard_error"
                        ]
                    )
                )
            by_scale.append(scale_objective)
            derivative_by_scale.append(scale_derivative)
            derivative_error_by_scale.append(scale_derivative_error)
        cross_objective.append(by_scale)
        cross_derivative.append(derivative_by_scale)
        cross_derivative_error.append(derivative_error_by_scale)

    partition: dict[str, Any] | None = None
    if responsibility is not None:
        weighted_responsibility = normalized_weight[:, None, None] * responsibility
        mass = weighted_responsibility.sum(axis=0)
        local_weight = weighted_responsibility / np.maximum(
            mass[None], np.finfo(np.float64).tiny
        )
        entropy = -np.sum(
            responsibility * np.log(np.maximum(responsibility, np.finfo(np.float32).tiny)),
            axis=-1,
        )
        partition = {
            "physical_mass_by_notional_and_partition": mass.tolist(),
            "physical_ESS_by_notional_and_partition": (
                1.0 / np.square(local_weight).sum(axis=0)
            ).tolist(),
            "mean_router_entropy_by_notional": np.sum(
                normalized_weight[:, None] * entropy, axis=0
            ).tolist(),
            "effective_soft_partition_count_by_notional": np.exp(
                np.sum(normalized_weight[:, None] * entropy, axis=0)
            ).tolist(),
        }

    candidates = None
    if isinstance(model, LearnedPartitionDirections):
        candidates = _evaluate_partition_candidates(
            model,
            data,
            indices,
            notionals,
            batch_size=batch_size,
        )

    bucket_audit = None
    if audit_buckets is not None:
        assignment = _stratified_bucket_assignment(
            data,
            indices,
            buckets=audit_buckets,
            seed=bucket_seed,
        ).cpu().numpy()
        bucket_audit = {
            "definition": (
                "fixed buckets balanced by physical probability mass within each "
                "importance-sampling component; all reported moments use normalized "
                "physical weights within each bucket"
            ),
            "buckets": audit_buckets,
            "seed": bucket_seed,
            **_bucket_pnl_audit(
                pnl,
                weight,
                assignment,
                buckets=audit_buckets,
            ),
        }

    return {
        "paths": paths,
        "direct_P_paths": int(direct.sum()),
        "physical_weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "by_notional": by_notional,
        "cross_objective_definition": (
            "[direction_notional_index][evaluation_notional_index][time_index]; "
            + (
                "the scalar OCE translation is recalibrated separately for every "
                "evaluation notional and sample set"
                if use_oce_translation
                else "no OCE translation"
            )
        ),
        "cross_objective_evaluation_notionals": list(evaluation_notionals),
        "cross_objective": cross_objective,
        "cross_normalized_notional_derivative": cross_derivative,
        "cross_normalized_notional_derivative_standard_error": (
            cross_derivative_error
        ),
        "partition": partition,
        "hard_partition_candidates": candidates,
        "bucket_audit": bucket_audit,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_inp(
        args.objective != "bucket-downside2"
        or args.architecture in {"feedforward", "gated-feedforward", "kernel"},
        "bucket-downside2 is defined for feed-forward and kernel architectures",
    )
    verify_inp(
        args.objective != "oce-powerexp" or args.architecture != "partition",
        "oce-powerexp is defined for feed-forward and kernel architectures",
    )
    verify_inp(
        isinstance(args.buckets, int) and 1 < args.buckets <= args.batch_size,
        lambda: f"invalid bucket count {args.buckets} for batch {args.batch_size}",
    )
    verify_inp(
        math.isfinite(args.gain_tiebreak) and args.gain_tiebreak >= 0.0,
        lambda: f"invalid gain tie-break {args.gain_tiebreak}",
    )
    verify_inp(
        math.isfinite(args.bucket_temperature) and args.bucket_temperature >= 0.0,
        lambda: f"invalid bucket temperature {args.bucket_temperature}",
    )
    vector_barrier = args.portfolio_convention == "itm-dollar-vector-barrier"
    itm_dollar = args.portfolio_convention in {
        "itm-dollar-linear-cost",
        "itm-dollar-vector-barrier",
    }
    verify_inp(
        not vector_barrier
        or (
            args.return_horizon == "expiry"
            and args.hedge == "raw"
            and args.architecture != "partition"
            and args.cost_bps > 0.0
        ),
        (
            "ITM dollar vector-barrier experiments require raw expiry returns, "
            "a positive cost rate, and a non-partition architecture"
        ),
    )
    verify_inp(
        math.isfinite(args.barrier_interior_epsilon)
        and 0.0 < args.barrier_interior_epsilon < 0.1,
        lambda: (
            "barrier interior epsilon must lie in (0,0.1), got "
            f"{args.barrier_interior_epsilon!r}"
        ),
    )
    initialization_report: dict[str, Any] | None = None
    initialization_id = ""
    if args.initialize_report:
        initialization_path = Path(args.initialize_report)
        initialization_report = json.loads(
            initialization_path.read_text(encoding="utf-8")
        )
        initialization_report["report_path"] = str(initialization_path)
        verify_inp(
            args.architecture == "gated-feedforward",
            "direction initialization is only valid for gated-feedforward",
        )
        initialization_id = Path(initialization_report["checkpoint"]).stem.split()[-1]
    verify_inp(
        not args.freeze_initialized_direction or initialization_report is not None,
        "freezing an initialized direction requires --initialize-report",
    )
    notionals = tuple(float(value) for value in args.notionals)
    audit_notionals = (
        tuple(float(value) for value in args.audit_notionals)
        if args.audit_notionals
        else None
    )
    verify_inp(
        audit_notionals is None
        or (
            all(math.isfinite(value) and value > 0.0 for value in audit_notionals)
            and all(
                left < right
                for left, right in zip(audit_notionals, audit_notionals[1:])
            )
        ),
        lambda: f"audit notionals must be finite-positive and increasing: {audit_notionals}",
    )
    case = RequiredCase(
        generator=args.generator,
        trading_frequency=args.frequency,
        return_horizon=args.return_horizon,
        hedge=args.hedge,
        cost_rate=args.cost_bps * 1.0e-4,
        path_count=args.paths,
    )
    loaded = _load_case(
        case,
        cache_root=Path(args.cache_root),
        need_primitives=args.return_horizon == "expiry",
    )
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    if initialization_report is not None:
        verify_inp(
            initialization_report.get("source_cache_uid") == source_cache_uid
            and initialization_report.get("fold", {}).get("fold") == args.fold,
            "direction initialization must use the same universe and reciprocal fold",
        )
    env = BaseEnvironment(
        precision=args.precision, device=args.device, debug=args.debug
    )
    full_data, source_split = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    data, fold_diagnostics = _select_times_and_fold(
        full_data,
        deterministic_time=0,
        conditional_time=6,
        fold=args.fold,
        seed=args.split_seed,
    )
    del full_data
    portfolio_normalization = None
    if vector_barrier:
        data, portfolio_normalization = _to_itm_dollar_vector_barrier(
            data,
            cost_rate=case.cost_rate,
        )
    elif itm_dollar:
        data, portfolio_normalization = _to_itm_dollar_portfolio(
            data,
            cost_rate=case.cost_rate,
            vector_barrier=False,
        )

    torch.manual_seed(args.seed)
    if torch.device(env.device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(env.device)
    horizon_label = "period" if args.return_horizon == "incremental" else "expiry"
    default_notionals = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    notional_label = ""
    if notionals != default_notionals:
        notional_label = "-n-" + "-".join(f"{value:g}" for value in notionals)
    partition_label = ""
    if args.architecture == "partition" and args.partitions != 16:
        partition_label = f"-k-{args.partitions}"
    if args.architecture == "kernel":
        partition_label = f"-bounded-learned-k-{args.kernels}"
    objective_label = "" if args.objective == "powerexp" else f"-{args.objective}"
    if args.objective == "bucket-downside2":
        objective_label += (
            f"-balanced-m-{args.buckets}-g-{args.gain_tiebreak:g}"
            f"-tau-{args.bucket_temperature:g}"
        )
    label = (
        f"parallel-directions-{args.generator}-{args.frequency}-{horizon_label}-"
        f"{args.hedge}-{args.cost_bps:g}bps-{args.paths}-{args.architecture}-"
        f"fold-{args.fold}"
        f"{partition_label}{objective_label}{notional_label}"
    )
    if vector_barrier:
        label += "-itm-dollar-vector-barrier"
    elif itm_dollar:
        label += "-itm-dollar-linear-cost"
    if initialization_report is not None:
        label += f"-init-{initialization_id}"
    if args.freeze_initialized_direction:
        label += "-freeze-direction"
    config = ParallelDirectionConfig(
        label=label,
        source_cache_uid=source_cache_uid,
        architecture=args.architecture,
        state_features=data.state.shape[-1],
        notionals=notionals,
        partitions=args.partitions,
        kernels=args.kernels,
        learn_oce_translation=args.objective == "oce-powerexp",
        action_radius=(
            1.0 - args.barrier_interior_epsilon if vector_barrier else 1.0
        ),
        group=(
            "gendysanos/parallel-direction-comparison/"
            f"{args.generator}/{args.frequency}/{args.architecture}/"
            f"{args.cost_bps:g}bps/fold-{args.fold}"
        ),
        user_version=args.user_version,
    )
    state_mean, state_whitener, whitening = fit_physical_state_whitener(
        data.state,
        data.path_weight,
        data.reference,
        relative_eigenvalue_floor=args.covariance_relative_floor,
    )
    kernel_basis = None
    if args.architecture == "partition":
        model: DirectionModel = (
            LearnedPartitionDirections(
                env,
                config=config,
                state_mean=state_mean[1],
                state_whitener=state_whitener[1],
            )
        )
    elif args.architecture == "gated-feedforward":
        model = GatedParallelFeedForwardDirections(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
        )
    elif args.architecture == "kernel":
        centers, inverse_bandwidth, kernel_basis = _fit_kernel_basis(
            data,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            kernels=args.kernels,
            seed=args.seed,
        )
        model = ParallelKernelDirections(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            centers=centers,
            inverse_bandwidth=inverse_bandwidth,
        )
    else:
        model = ParallelFeedForwardDirections(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
        )
    initialization = (
        _initialize_gated_direction_from_report(model, initialization_report)
        if isinstance(model, GatedParallelFeedForwardDirections)
        and initialization_report is not None
        else None
    )
    if args.freeze_initialized_direction:
        assert isinstance(model, GatedParallelFeedForwardDirections)
        for parameter in model.conditional.parameters():
            parameter.requires_grad_(False)
        assert all(parameter.requires_grad for parameter in model.gate.parameters())
        assert initialization is not None
        initialization["direction_frozen_during_training"] = True
    notionals_tensor = torch.as_tensor(
        notionals, dtype=env.fdtype, device=env.device
    )
    callable_model = torch.compile(model, dynamic=False) if not args.no_compile else model
    checkpoint_path = Path(args.checkpoint_root) / config.group / f"{config.filename}.pt"
    checkpoint_lock = FileLock(f"{checkpoint_path}.lock")
    with checkpoint_lock:
        training = _train(
            model,
            callable_model,
            data,
            notionals_tensor,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            valuation_batch_size=args.valuation_batch_size,
            epochs=args.epochs,
            value_update=args.value_update,
            checkpoint_update=args.checkpoint_update,
            lr=args.lr,
            seed=args.seed,
            balance_coefficient=args.balance_coefficient,
            objective=args.objective,
            buckets=args.buckets,
            gain_tiebreak=args.gain_tiebreak,
            bucket_temperature=args.bucket_temperature,
        )

    training_set = _evaluate(
        model,
        data,
        data.reference,
        notionals,
        batch_size=args.valuation_batch_size,
        objective=args.objective,
        evaluation_notionals=(
            audit_notionals
        ),
        audit_buckets=(args.buckets if args.objective == "bucket-downside2" else None),
        bucket_seed=args.seed + 20_000,
    )
    validation_set = _evaluate(
        model,
        data,
        data.validation,
        notionals,
        batch_size=args.valuation_batch_size,
        objective=args.objective,
        evaluation_notionals=(
            audit_notionals
        ),
        audit_buckets=(args.buckets if args.objective == "bucket-downside2" else None),
        bucket_seed=args.seed + 30_000,
    )
    fitted_kernel = None
    if isinstance(model, ParallelKernelDirections):
        with torch.no_grad():
            fitted_inverse_bandwidth = torch.exp(model.log_inverse_bandwidth)
            fitted_kernel = {
                "definition": (
                    "physical-weighted K-means initializes z_k and w_k; both are "
                    "subsequently learned jointly with the local directions"
                ),
                "center_root_mean_square_displacement": float(
                    torch.sqrt(
                        torch.mean((model.centers - model.initial_centers).square())
                    ).cpu()
                ),
                "minimum_inverse_bandwidth": float(
                    fitted_inverse_bandwidth.min().cpu()
                ),
                "median_inverse_bandwidth": float(
                    fitted_inverse_bandwidth.median().cpu()
                ),
                "maximum_inverse_bandwidth": float(
                    fitted_inverse_bandwidth.max().cpu()
                ),
            }
    report = {
        "status": "complete",
        "schema_version": "1.2" if vector_barrier else "1.1",
        "experiment": (
            "independent fixed-notional conditional directions trained in parallel"
        ),
        "definition": {
            "objective": (
                (
                    "OCE power-exp lambda=1: sup_y E[u(y+n*pi)]-y, with one "
                    "learned real y per independent (n,trading-time) portfolio; "
                    "after action training each y is solved exactly on the full "
                    "training half before validation"
                    if args.objective == "oce-powerexp"
                    else "power-exp lambda=1 applied to n times unit-gross net P&L"
                )
                if args.objective in {"powerexp", "oce-powerexp"}
                else (
                    (
                        "negative worst fixed-bucket physical expected squared "
                        "downside, with a full-sample positive-gain tie-break"
                        if args.objective == "bucket-downside2"
                        else (
                            "negative physical expected squared downside of "
                            "unit-gross net P&L"
                        )
                    )
                    + "; the repeated heads are independent cold restarts"
                )
            ),
            "action": (
                (
                    "the signed net action is an entry-value dollar portfolio; "
                    "its L1 norm is strictly below one and one logarithmic barrier "
                    "is applied to that full vector"
                    if vector_barrier
                    else (
                        (
                            "the signed action is an entry-value dollar portfolio "
                            "with separate nonnegative long/short parameters"
                            if itm_dollar
                            else (
                                "separate nonnegative long/short positions; gross sum "
                                "is at most one for kernel and gated-feedforward, and "
                                "exactly one for the other direction-only architectures"
                            )
                        )
                    )
                )
            ),
            "notional_independence": (
                "every notional has cold-started parameters and no gradient sharing"
            ),
            "OCE_validation": (
                "the action and training y are frozen on validation; the report also "
                "recalibrates one scalar y on each sample set for a proper fixed-action "
                "OCE valuation"
                if args.objective == "oce-powerexp"
                else None
            ),
            "partition": (
                "state-only learned soft router; local direction objectives are summed "
                "and physical partition masses are regularized towards equal mass"
            ),
            "validation": (
                "the entire fitted action is frozen and evaluated on the reciprocal half"
            ),
        },
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "source_split_ignored_for_this_experiment": source_split,
        "fold": fold_diagnostics,
        "portfolio_normalization": portfolio_normalization,
        "model_config": asdict(config),
        "training_configuration": {
            "batch_size": args.batch_size,
            "valuation_batch_size": args.valuation_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "compile": not args.no_compile,
            "seed": args.seed,
            "balance_coefficient": args.balance_coefficient,
            "optimization_objective": args.objective,
            "buckets": args.buckets if args.objective == "bucket-downside2" else None,
            "bucket_construction": (
                "fixed training-fold buckets balanced by physical probability mass "
                "within each importance component; every minibatch draws equally "
                "from every bucket"
                if args.objective == "bucket-downside2"
                else None
            ),
            "gain_tiebreak": args.gain_tiebreak,
            "bucket_relative_temperature": args.bucket_temperature,
            "freeze_initialized_direction": args.freeze_initialized_direction,
        },
        "initialization": initialization,
        "whitening": whitening,
        "kernel_basis": kernel_basis,
        "fitted_kernel": fitted_kernel,
        "training": training,
        "training_set": training_set,
        "validation_set": validation_set,
        "checkpoint": str(checkpoint_path),
        "device": str(env.device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(env.device))
            if torch.device(env.device).type == "cuda"
            else 0
        ),
    }
    audit_label = ""
    if audit_notionals:
        audit_label = "-audit-n-" + "-".join(
            f"{value:g}" for value in audit_notionals
        )
    report_path = Path(args.output_root) / f"{config.label}{audit_label}.json"
    _atomic_json(report_path, report)
    print(f"Completed parallel-direction report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), default="pca-iv")
    parser.add_argument("--frequency", choices=("daily", "weekly"), default="weekly")
    parser.add_argument("--paths", type=int, choices=(100_000, 1_000_000), default=100_000)
    parser.add_argument(
        "--architecture",
        choices=("feedforward", "gated-feedforward", "kernel", "partition"),
        required=True,
    )
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cost-bps", type=float, choices=(0.0, 1.0, 10.0), required=True)
    parser.add_argument("--hedge", choices=("raw", "delta"), default="raw")
    parser.add_argument(
        "--return-horizon", choices=("incremental", "expiry"), default="incremental"
    )
    parser.add_argument(
        "--objective",
        choices=("powerexp", "oce-powerexp", "downside2", "bucket-downside2"),
        default="powerexp",
    )
    parser.add_argument(
        "--portfolio-convention",
        choices=("native", "itm-dollar-linear-cost", "itm-dollar-vector-barrier"),
        default="native",
    )
    parser.add_argument("--barrier-interior-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument(
        "--notionals",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    )
    parser.add_argument("--audit-notionals", type=float, nargs="+")
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=16_384)
    parser.add_argument("--value-update", type=int, default=500)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--balance-coefficient", type=float, default=1.0)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--gain-tiebreak", type=float, default=1.0e-8)
    parser.add_argument("--bucket-temperature", type=float, default=0.0)
    parser.add_argument("--initialize-report")
    parser.add_argument("--freeze-initialized-direction", action="store_true")
    parser.add_argument("--covariance-relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=182_071)
    parser.add_argument("--split-seed", type=int, default=100_097)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
