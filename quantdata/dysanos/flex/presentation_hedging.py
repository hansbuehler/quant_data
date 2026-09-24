"""Five-day hedging paths from the presentation Direct-200 DYSANOS model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    JointDirectDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    bound_native_raw,
)
from quantdata.dysanos.flex.weekly_simulation import (
    front_week_sanos_inverse_cdf,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@dataclass(frozen=True)
class PresentationInstrument:
    """One observed fixed-cash option used in the presentation hedge book."""

    expiry_days: int
    is_call: bool
    strike: float
    mid: float
    halfspread: float

    @property
    def label(self) -> str:
        """Return the compact label used by the original presentation run."""
        side = "C" if self.is_call else "P"
        return f"{self.expiry_days}d {side} {self.strike:.4f}"


# These are the exact normalized strikes and observed bid/ask-derived inputs
# used by the 2025-03-18 presentation experiment.  The two target calls are
# excluded from the hedge book.
PRESENTATION_INSTRUMENTS = (
    PresentationInstrument(1, False, 0.989078938961029, 0.001745122135616839, 1.7807353287935257e-05),
    PresentationInstrument(1, False, 0.9997620582580566, 0.005386729026213288, 2.6711029931902885e-05),
    PresentationInstrument(1, True, 1.0006523132324219, 0.0051819439977407455, 3.5614706575869934e-05),
    PresentationInstrument(1, True, 1.0113354921340942, 0.001460204366594553, 1.7807353287935257e-05),
    PresentationInstrument(2, False, 0.988176703453064, 0.003285819082520902, 2.6714173145592213e-05),
    PresentationInstrument(2, False, 0.9997498989105225, 0.007479913299903274, 3.561866469681263e-05),
    PresentationInstrument(2, True, 1.0006401538848877, 0.007275106385350227, 2.6714056722170208e-05),
    PresentationInstrument(2, True, 1.0122133493423462, 0.0027604441856965423, 3.56185482814873e-05),
    PresentationInstrument(5, False, 0.9858283996582031, 0.006269084755331278, 3.56198288500309e-05),
    PresentationInstrument(5, False, 0.9991744160652161, 0.011211331933736801, 4.45246696472168e-05),
    PresentationInstrument(5, True, 1.014299988746643, 0.005227205576375127, 2.6714755222201347e-05),
)

TARGET_STRIKES = (0.9991744160652161, 1.0000641345977783)
TARGET_MARKET_PRICE = 0.5306955257730696
TARGET_DATE = "2025-03-18"


def _checkpoint_config(payload: Mapping[str, Any]) -> JointDirectDynamicsConfig:
    values = payload.get("model_config")
    if not isinstance(values, Mapping):
        raise ValueError("Direct-200 checkpoint has no model configuration")
    geometry_values = values.get("geometry")
    network_values = values.get("decoder_network")
    if not isinstance(geometry_values, Mapping) or not isinstance(network_values, Mapping):
        raise ValueError("Direct-200 checkpoint geometry/network is invalid")
    geometry_payload = dict(geometry_values)
    for key in ("nms_range", "explicit_model_business_dtes"):
        if geometry_payload.get(key) is not None:
            geometry_payload[key] = tuple(geometry_payload[key])
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(**geometry_payload)
    return JointDirectDynamicsConfig(
        label=str(values["label"]),
        state_size=int(values["state_size"]),
        group=str(values["group"]),
        user_version=str(values["user_version"]),
        geometry=geometry,
        decoder_network=FeedForwardConfig(**dict(network_values)),
        minimum_mean_reversion_per_week=float(
            values["minimum_mean_reversion_per_week"]
        ),
        innovation_std_floor=float(values["innovation_std_floor"]),
    )


def load_presentation_model(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[BaseEnvironment, JointDirectDynamicsModel]:
    """Load the exact affine Direct-200 checkpoint used by the presentation."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "flex-joint-direct-dynamics-v1" or payload.get(
        "phase"
    ) != "complete":
        raise ValueError(f"not a completed Direct-200 checkpoint: {checkpoint_path}")
    config = _checkpoint_config(payload)
    if payload.get("model_config_version") != config.version.unique_id48:
        raise ValueError("Direct-200 checkpoint code version is stale")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Direct-200 checkpoint has no model state")
    daily_state = state_dict.get("daily_state_embedding.weight")
    if not isinstance(daily_state, torch.Tensor) or daily_state.ndim != 2:
        raise ValueError("Direct-200 checkpoint has no daily state history")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = JointDirectDynamicsModel(
        env,
        num_days=daily_state.shape[0],
        config=config,
    )
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    model.eval()
    return env, model


def exact_daily_transition(
    model: JointDirectDynamicsModel,
    *,
    sessions_per_week: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Factor the fitted exact-week Gaussian AR law into equal daily laws.

    The model uses row states, ``h' = m + (h-m) A + eps``.  Consequently its
    stationary covariance solves ``C = A.T C A + Q``.
    """
    if sessions_per_week <= 0:
        raise ValueError("sessions_per_week must be positive")
    with torch.no_grad():
        weekly_transition = model.weekly_transition().cpu().double().numpy()
        weekly_factor = model.innovation_cholesky().cpu().double().numpy()
        generator = model.mean_reversion_generator().cpu().double()
        daily_transition = torch.matrix_exp(-generator / sessions_per_week).numpy()
    weekly_covariance = weekly_factor @ weekly_factor.T
    stationary = solve_discrete_lyapunov(
        weekly_transition.T,
        weekly_covariance,
    )
    stationary = 0.5 * (stationary + stationary.T)
    daily_covariance = stationary - daily_transition.T @ stationary @ daily_transition
    daily_covariance = 0.5 * (daily_covariance + daily_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(daily_covariance)
    eigenvalue_floor = 64.0 * np.finfo(np.float64).eps * max(
        float(np.max(eigenvalues)), 1.0
    )
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    daily_factor = eigenvectors @ np.diag(np.sqrt(clipped))

    composed_transition = np.linalg.matrix_power(daily_transition, sessions_per_week)
    composed_covariance = np.zeros_like(daily_covariance)
    propagation = np.eye(daily_transition.shape[0])
    for _ in range(sessions_per_week):
        composed_covariance += propagation.T @ daily_covariance @ propagation
        propagation = propagation @ daily_transition
    diagnostics = {
        "transition_max_abs_error": float(
            np.max(np.abs(composed_transition - weekly_transition))
        ),
        "covariance_max_abs_error": float(
            np.max(np.abs(composed_covariance - weekly_covariance))
        ),
        "daily_covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
        "daily_covariance_clipped_eigenvalues": int(np.sum(eigenvalues < eigenvalue_floor)),
    }
    return (
        torch.as_tensor(daily_transition, dtype=model.fdtype, device=model.device),
        torch.as_tensor(daily_factor, dtype=model.fdtype, device=model.device),
        diagnostics,
    )


def _decode_surface(model: JointDirectDynamicsModel, states: torch.Tensor):
    raw = model.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    return model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )


def _call_prices(
    model: JointDirectDynamicsModel,
    states: torch.Tensor,
    spots: torch.Tensor,
    *,
    remaining_days: torch.Tensor,
    cash_strikes: torch.Tensor,
) -> torch.Tensor:
    surface = _decode_surface(model, states)
    normalized_strikes = cash_strikes.unsqueeze(0) / spots.unsqueeze(1)
    expiries = remaining_days.unsqueeze(0).expand_as(normalized_strikes) / DAYS_PER_YEAR
    normalized_calls = model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=expiries,
        strikes=normalized_strikes,
    )
    return spots.unsqueeze(1) * normalized_calls


def generate_presentation_paths(
    *,
    checkpoint_path: Path,
    target_history_path: Path,
    output_path: Path,
    num_paths: int,
    batch_size: int,
    seed: int = 58123,
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    """Generate the five-day path dataset used by the presentation objectives."""
    if num_paths <= 0 or batch_size <= 0:
        raise ValueError("num_paths and batch_size must be positive")
    env, model = load_presentation_model(checkpoint_path, device=device)
    history = torch.load(target_history_path, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states = history.get("states")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError("Direct-200 target history is invalid")
    try:
        target_index = dates.index(TARGET_DATE)
    except ValueError as error:
        raise ValueError(f"target date {TARGET_DATE} is absent from state history") from error
    initial_state = states[target_index].to(dtype=env.fdtype, device=env.device)
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    equilibrium = model.equilibrium_mean.detach()

    num_steps = 5
    num_assets = len(PRESENTATION_INSTRUMENTS)
    output_states = np.empty((num_paths, num_steps + 1, model.config.state_size), np.float32)
    output_cumulative = np.empty((num_paths, num_steps), np.float32)
    output_values = np.empty((num_paths, num_steps + 1, num_assets), np.float32)
    output_claim = np.empty((num_paths, num_steps + 1), np.float32)

    expiries = torch.tensor(
        [instrument.expiry_days for instrument in PRESENTATION_INSTRUMENTS],
        dtype=env.fdtype,
        device=env.device,
    )
    strikes = torch.tensor(
        [instrument.strike for instrument in PRESENTATION_INSTRUMENTS],
        dtype=env.fdtype,
        device=env.device,
    )
    is_call = torch.tensor(
        [instrument.is_call for instrument in PRESENTATION_INSTRUMENTS],
        dtype=torch.bool,
        device=env.device,
    )
    target_strikes = torch.tensor(TARGET_STRIKES, dtype=env.fdtype, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    initial_model_prices: np.ndarray | None = None
    initial_model_target_price: float | None = None

    with torch.inference_mode():
        for first in range(0, num_paths, batch_size):
            last = min(first + batch_size, num_paths)
            count = last - first
            state = initial_state.unsqueeze(0).expand(count, -1).clone()
            spot = torch.ones(count, dtype=env.fdtype, device=env.device)
            settled = torch.zeros((count, num_assets), dtype=env.fdtype, device=env.device)
            for step in range(num_steps + 1):
                output_states[first:last, step] = state.cpu().numpy()
                remaining = torch.clamp(expiries - step, min=1.0)
                calls = _call_prices(
                    model,
                    state,
                    spot,
                    remaining_days=remaining,
                    cash_strikes=strikes,
                )
                live_values = torch.where(
                    is_call.unsqueeze(0),
                    calls,
                    calls - spot.unsqueeze(1) + strikes.unsqueeze(0),
                )
                expired = expiries <= step
                values = torch.where(expired.unsqueeze(0), settled, live_values)
                if step == 0:
                    initial_model_prices = values[0].cpu().double().numpy()
                    # Preserve the exact observed initial hedge prices, as in
                    # the original presentation experiment.
                    observed = torch.tensor(
                        [instrument.mid for instrument in PRESENTATION_INSTRUMENTS],
                        dtype=env.fdtype,
                        device=env.device,
                    )
                    values = observed.unsqueeze(0).expand(count, -1)
                output_values[first:last, step] = values.cpu().numpy()

                target_remaining = torch.full(
                    (2,), max(5 - step, 1), dtype=env.fdtype, device=env.device
                )
                target_calls = _call_prices(
                    model,
                    state,
                    spot,
                    remaining_days=target_remaining,
                    cash_strikes=target_strikes,
                )
                width = TARGET_STRIKES[1] - TARGET_STRIKES[0]
                claim = (target_calls[:, 0] - target_calls[:, 1]) / width
                if step == 0:
                    initial_model_target_price = float(claim[0].cpu())
                    claim = torch.full_like(claim, TARGET_MARKET_PRICE)
                if step == num_steps:
                    claim = (
                        torch.relu(spot - TARGET_STRIKES[0])
                        - torch.relu(spot - TARGET_STRIKES[1])
                    ) / width
                output_claim[first:last, step] = claim.cpu().numpy()
                if step == num_steps:
                    break

                surface = _decode_surface(model, state)
                spot_normal = torch.randn(
                    count, dtype=env.fdtype, device=env.device, generator=generator
                )
                gross_spot_return = front_week_sanos_inverse_cdf(
                    surface, spot_normal
                )
                previous_spot = spot
                spot = spot * gross_spot_return
                output_cumulative[first:last, step] = torch.log(spot).cpu().numpy()

                newly_expired = expiries == (step + 1)
                if torch.any(newly_expired):
                    expiry_strikes = strikes[newly_expired]
                    expiry_calls = torch.relu(spot.unsqueeze(1) - expiry_strikes.unsqueeze(0))
                    expiry_puts = torch.relu(expiry_strikes.unsqueeze(0) - spot.unsqueeze(1))
                    expiry_values = torch.where(
                        is_call[newly_expired].unsqueeze(0), expiry_calls, expiry_puts
                    )
                    settled[:, newly_expired] = expiry_values

                state_normal = torch.randn(
                    (count, model.config.state_size),
                    dtype=env.fdtype,
                    device=env.device,
                    generator=generator,
                )
                state = equilibrium + (state - equilibrium) @ daily_transition
                state = state + state_normal @ daily_factor.T
                if torch.any(~torch.isfinite(state)) or torch.any(~torch.isfinite(spot)):
                    raise RuntimeError("DYSANOS path generation produced a non-finite value")
                if torch.any(previous_spot <= 0.0):
                    raise RuntimeError("DYSANOS path generation lost positive spot")

    if initial_model_prices is None or initial_model_target_price is None:
        raise RuntimeError("path generation produced no initial prices")
    labels = np.asarray([instrument.label for instrument in PRESENTATION_INSTRUMENTS])
    halfspread = np.asarray(
        [instrument.halfspread for instrument in PRESENTATION_INSTRUMENTS],
        dtype=np.float64,
    )
    expiry_days = np.asarray(
        [instrument.expiry_days for instrument in PRESENTATION_INSTRUMENTS],
        dtype=np.int64,
    )
    np.savez_compressed(
        output_path,
        states=output_states,
        cumulative=output_cumulative,
        values=output_values,
        claim_values=output_claim,
        halfspread=halfspread,
        expiry_days=expiry_days,
        labels=labels,
        strikes=np.asarray(TARGET_STRIKES),
        market_price=np.asarray(TARGET_MARKET_PRICE),
        initial_model_prices=initial_model_prices,
        initial_model_target_price=np.asarray(initial_model_target_price),
    )
    return {
        "output": str(output_path),
        "num_paths": num_paths,
        "num_states": model.config.state_size,
        "num_native_coordinates": model.config.geometry.native_state_size,
        "target_date": TARGET_DATE,
        "target_index": target_index,
        "transition": transition_diagnostics,
        "initial_model_prices": initial_model_prices.tolist(),
        "initial_observed_prices": [
            instrument.mid for instrument in PRESENTATION_INSTRUMENTS
        ],
        "initial_model_target_price": initial_model_target_price,
        "initial_observed_target_price": TARGET_MARKET_PRICE,
        "terminal_spot_mean": float(np.mean(np.exp(output_cumulative[:, -1]))),
        "terminal_spot_standard_deviation": float(
            np.std(np.exp(output_cumulative[:, -1]), ddof=0)
        ),
    }


def generate_presentation_rolling_market(
    *,
    checkpoint_path: Path,
    path_dataset: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    """Value the presentation's fresh daily ATM 1D/2D rolling hedge book."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    env, model = load_presentation_model(checkpoint_path, device=device)
    data = np.load(path_dataset, allow_pickle=False)
    states = data["states"]
    cumulative = data["cumulative"]
    if states.ndim != 3 or states.shape[1:] != (6, model.config.state_size):
        raise ValueError(f"path state geometry is invalid: {states.shape}")
    num_paths = len(states)
    forward = np.column_stack(
        (np.ones(num_paths, dtype=np.float32), np.exp(cumulative))
    ).astype(np.float32, copy=False)
    rolling_returns = np.zeros((num_paths, 5, 4), dtype=np.float32)
    rolling_costs = np.zeros_like(rolling_returns)
    one_day_spread = float(np.median(data["halfspread"][data["expiry_days"] == 1]))
    two_day_spread = float(np.median(data["halfspread"][data["expiry_days"] == 2]))

    with torch.inference_mode():
        for first in range(0, num_paths, batch_size):
            last = min(first + batch_size, num_paths)
            batch_states = torch.as_tensor(
                states[first:last], dtype=env.fdtype, device=env.device
            )
            batch_forward = torch.as_tensor(
                forward[first:last], dtype=env.fdtype, device=env.device
            )
            count = last - first
            for step in range(5):
                issue_state = batch_states[:, step]
                issue_spot = batch_forward[:, step]
                next_spot = batch_forward[:, step + 1]
                for tenor_index, tenor in enumerate((1, 2)):
                    if step + tenor > 5:
                        continue
                    # Each fresh contract has cash strike equal to pathwise
                    # issue spot, hence normalized strike one at inception.
                    issue_surface = _decode_surface(model, issue_state)
                    expiry = torch.full_like(issue_spot, tenor / DAYS_PER_YEAR)
                    normalized = torch.ones_like(issue_spot)
                    issue_call = issue_spot * model.sanos_decoder.price_calls_batch(
                        issue_surface,
                        expiries_y=expiry.unsqueeze(1),
                        strikes=normalized.unsqueeze(1),
                    )[:, 0]
                    issue_put = issue_call
                    if tenor == 1:
                        next_call = torch.relu(next_spot - issue_spot)
                        next_put = torch.relu(issue_spot - next_spot)
                        cost = one_day_spread * issue_spot
                    else:
                        next_state = batch_states[:, step + 1]
                        next_surface = _decode_surface(model, next_state)
                        next_strike = issue_spot / next_spot
                        one_day = torch.full_like(next_spot, 1.0 / DAYS_PER_YEAR)
                        normalized_call = model.sanos_decoder.price_calls_batch(
                            next_surface,
                            expiries_y=one_day.unsqueeze(1),
                            strikes=next_strike.unsqueeze(1),
                        )[:, 0]
                        next_call = next_spot * normalized_call
                        next_put = next_call - next_spot + issue_spot
                        cost = two_day_spread * issue_spot + one_day_spread * next_spot
                    base = 2 * tenor_index
                    rolling_returns[first:last, step, base] = (
                        next_put - issue_put
                    ).cpu().numpy()
                    rolling_returns[first:last, step, base + 1] = (
                        next_call - issue_call
                    ).cpu().numpy()
                    cost_values = cost.cpu().numpy()
                    rolling_costs[first:last, step, base] = cost_values
                    rolling_costs[first:last, step, base + 1] = cost_values
            if count == 0:
                raise RuntimeError("rolling-market batch is unexpectedly empty")

    if not np.all(np.isfinite(rolling_returns)) or not np.all(
        np.isfinite(rolling_costs)
    ):
        raise RuntimeError("rolling-market valuation produced a non-finite value")
    np.savez_compressed(
        output_path,
        states=states,
        cumulative=cumulative,
        forward=forward,
        rolling_returns=rolling_returns,
        rolling_costs=rolling_costs,
    )
    return {
        "output": str(output_path),
        "num_paths": num_paths,
        "num_steps": 5,
        "num_rolling_options_per_step": 4,
        "one_day_halfspread": one_day_spread,
        "two_day_halfspread": two_day_spread,
        "maximum_absolute_return": float(np.max(np.abs(rolling_returns))),
        "maximum_cost": float(np.max(rolling_costs)),
    }


__all__ = [
    "PRESENTATION_INSTRUMENTS",
    "TARGET_DATE",
    "TARGET_MARKET_PRICE",
    "TARGET_STRIKES",
    "exact_daily_transition",
    "generate_presentation_paths",
    "generate_presentation_rolling_market",
    "load_presentation_model",
]
