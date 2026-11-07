"""Remaining-horizon AR market paths with exact conditional asset-alpha removal."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedge2.batch_pricing import BatchDlvPricer, decode_batch
from quantdata.hedge2.cash import CashCurve, zero_strike_mask
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal import TerminalPaths


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class TerminalSimulationConfig:
    paths: int | None = None
    integration_parent_batch: int = 16

    def __post_init__(self) -> None:
        verify_inp((self.paths is None or isinstance(self.paths, int) and self.paths > 0)
                   and isinstance(self.integration_parent_batch, int) and self.integration_parent_batch > 0,
                   lambda: "Terminal path count and conditional-integration batch must be positive integers")


@dataclass(frozen=True, kw_only=True)
class TerminalMarketPaths:
    instrument_gains: np.ndarray
    raw_instrument_gains: np.ndarray
    removed_conditional_mean: np.ndarray
    costs: np.ndarray
    instrument_prices: np.ndarray
    spots: np.ndarray
    state_features: np.ndarray
    history_ids: np.ndarray
    live_contracts: np.ndarray
    claim_gains: np.ndarray
    claim_payoffs: np.ndarray
    terminate_after_step: np.ndarray
    initial_active: np.ndarray
    weights: np.ndarray
    innovation_indices: np.ndarray
    lower_clips: np.ndarray
    upper_clips: np.ndarray
    conditional_mean_residual: np.ndarray
    integrated_parent_counts: np.ndarray
    zero_live_model_prices: np.ndarray
    model_date: str
    seed: int
    elapsed_day: int
    claim_labels: tuple[str, ...]
    future_spread_convention: str = "freeze today's absolute half-spreads; no future observed quotes"
    alpha_convention: str = "exact uniform integration of every fitted joint innovation at each visited state"

    def for_claim(self, claim: Claim, index: int) -> TerminalPaths:
        """Add only observed reset/barrier information to the policy's current-state inputs."""
        count, steps, _ = self.instrument_gains.shape
        if not 0 <= index < len(self.claim_labels) or self.claim_labels[index] != claim.label:
            raise ValueError("The terminal claim index must match the saved payoff label")
        reset = np.zeros((count, steps))
        observed = np.zeros((count, steps))
        if claim.kind in ("ko", "ratio_straddle"):
            observation = claim.observation_day - self.elapsed_day
            for step in range(steps):
                if step >= observation:
                    observed[:, step] = 1.
                    if claim.kind == "ratio_straddle":
                        value = (np.full(count, claim.reset_spot) if observation <= 0
                                 else self.spots[:, observation])
                        if claim.reset_spot is None and observation <= 0:
                            raise ValueError("A previously reset claim must carry its actual reset spot")
                        reset[:, step] = np.log(value / claim.initial_spot)
        features = np.concatenate([self.state_features, observed[:, :, None], reset[:, :, None]], axis=2)
        return TerminalPaths(gains=self.instrument_gains, costs=self.costs, features=features,
            history_ids=self.history_ids, live_contracts=self.live_contracts,
            terminate_after_step=self.terminate_after_step[:, :, index], claim_gain=self.claim_gains[:, index],
            weights=self.weights, initially_active=bool(self.initial_active[index]))


def cash_europeans_batch(*, decoded: BatchDlvPricer | None, template: CashCurve,
                         spots: np.ndarray, business_dtes: np.ndarray,
                         cash_strikes: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Frozen cash contracts; expired assets are absent and zero DTE uses cash settlement."""
    days, strikes, calls = np.asarray(business_dtes), np.asarray(cash_strikes), np.asarray(is_call, dtype=bool)
    if (days.ndim != 1 or strikes.shape != days.shape or calls.shape != days.shape
            or spots.ndim != 1 or not np.isfinite(spots).all() or np.any(spots <= 0)):
        raise ValueError("Batch cash-price arguments must align fixed instruments and positive spots")
    values = np.zeros((len(spots), len(days)))
    settled = np.abs(days) <= 1e-6
    difference = spots[:, None] - strikes[settled]
    values[:, settled] = np.maximum(np.where(calls[settled], difference, -difference), 0.)
    live = days > 1e-6
    if live.any():
        if decoded is None:
            raise ValueError("Live options require a decoded conditional pricing state")
        forward, _, discount = template.at(days[live])
        forward = spots[:, None] / template.spot * forward[None, :]
        normalized = decoded.europeans(business_dtes=days[live], strikes=strikes[live] / forward,
            is_call=calls[live])
        values[:, live] = discount * forward * normalized
    return values


def _forecast(native: np.ndarray, model: FittedDynamics) -> np.ndarray:
    scores = (native - model.state_mean) @ model.directions
    return native + (model.ar_intercept + scores @ model.ar_matrix.T - scores) @ model.directions.T


def generate_terminal_paths(*, state: DailyState, dynamics: FittedDynamics, curve: CashCurve,
                            book: pd.DataFrame, claims: tuple[Claim, ...],
                            initial_spot: float, current_claim_marks: np.ndarray,
                            elapsed_day: int = 0, discount_to_current: float = 1., seed: int = 1,
                            config: TerminalSimulationConfig = TerminalSimulationConfig(),
                            progress: Callable[[str], None] | None = None) -> TerminalMarketPaths:
    """Freeze today's fitted model inside the forecast, never use future historical fits.

    Conditional means integrate the full empirical innovation support at
    each visited state. Future state-dependent means are not replaced by a
    single across-path mean. Asset gains and costs use one inception unit.
    The returned prices support premium-unit policies without redefining
    fixed-contract P&L or treating a repricing as a trade.
    """
    if (not claims or state.day != dynamics.as_of_date or initial_spot <= 0 or discount_to_current <= 0
            or not np.isfinite([initial_spot, discount_to_current]).all()
            or current_claim_marks.shape != (len(claims),) or not np.isfinite(current_claim_marks).all()
            or len({claim.terminal_day for claim in claims}) != 1 or len({claim.label for claim in claims}) != len(claims)
            or any(not np.isclose(claim.initial_spot, initial_spot, rtol=1e-12) for claim in claims)):
        raise ValueError("Terminal simulation needs aligned causal inputs and co-terminal targets")
    horizon = claims[0].terminal_day - elapsed_day
    if horizon < 1 or horizon > 5 or not isinstance(elapsed_day, int) or elapsed_day < 0:
        raise ValueError("The remaining benchmark horizon must be one through five sessions")
    if any(claim.kind != "digital" and claim.observation_done != (elapsed_day >= claim.observation_day)
           for claim in claims):
        raise ValueError("Barrier/reset observation state must match the actual elapsed episode time")
    zero_strike_mask(book)
    days = book["bdte"].to_numpy(dtype=float)
    strikes = book["strike"].to_numpy(dtype=float)
    calls = (book["option_type"] == "call").to_numpy()
    if (np.any(days < 1. - 1e-6) or not np.isfinite(days).all()
            or not np.isfinite(strikes).all() or np.any(strikes < 0.)):
        raise ValueError("The frozen terminal book must start with live finite cash contracts")
    count = max(10000, 10 * (1 + len(book))) if config.paths is None else config.paths
    empirical = np.asarray(dynamics.residuals)
    if (empirical.ndim != 2 or empirical.shape[1] != 1 + dynamics.directions.shape[1]
            or not np.isfinite(empirical).all() or np.any(empirical[:, 0] <= -1.)):
        raise ValueError("The empirical AR law must preserve finite states and positive spot")
    noise = empirical[:, 1:] @ dynamics.directions.T + empirical[:, :1] * dynamics.spot_state_loading
    support_count = len(empirical)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, support_count, size=(count, horizon))
    _, innovation_labels = np.unique(empirical, axis=0, return_inverse=True)
    spots = np.full((count, horizon + 1), curve.spot)
    native = np.broadcast_to(state.native, (count, len(state.native))).copy()
    instrument_count = 1 + len(book)
    prices = np.zeros((count, horizon + 1, instrument_count))
    prices[:, 0] = np.r_[curve.spot, book["observed_mid"].to_numpy(dtype=float)] / initial_spot
    if not np.isfinite(prices[:, 0]).all() or np.any(prices[:, 0] <= 0):
        raise ValueError("Initial ITM/forward marks must be positive without substituting a premium floor")
    fees = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float)] / initial_spot
    if not np.isfinite(fees).all() or np.any(fees < 0):
        raise ValueError("Forecast half-spreads must be finite and nonnegative")
    gains, raw, removed, costs = (np.zeros((count, horizon, instrument_count)) for _ in range(4))
    features = np.empty((count, horizon, dynamics.directions.shape[1] + 2))
    histories = np.zeros((count, horizon + 1), dtype=int)
    live = np.c_[np.ones(horizon + 1, dtype=bool), days[None, :] > np.arange(horizon + 1)[:, None] + 1e-6]
    lower_clips, upper_clips = np.zeros((count, horizon), dtype=int), np.zeros((count, horizon), dtype=int)
    mean_residual, zero_prices = np.zeros(horizon), np.zeros(horizon, dtype=int)
    integrated_parents = np.zeros(horizon, dtype=int)
    template = curve
    cumulative_discount = discount_to_current
    started, last_progress = time.monotonic(), 0.
    lower = np.r_[np.full(state.config.num_inner_strikes, 2. * np.log(state.config.min_local_vol)),
                  np.log(state.config.min_local_vol)]
    upper = np.r_[np.full(state.config.num_inner_strikes, 2. * np.log(state.config.max_local_vol)),
                  np.log(state.config.max_local_vol)]
    for step in range(horizon):
        scores = ((native - dynamics.state_mean) @ dynamics.directions - dynamics.current_scores) / dynamics.daily_factor_scales
        beta = dynamics.spot_state_loading
        outside = beta - dynamics.directions @ (dynamics.directions.T @ beta)
        outside_scale = float(np.linalg.norm(outside))
        component = (np.zeros(count) if outside_scale < 1e-14
                     else (native - state.native) @ (outside / outside_scale))
        features[:, step] = np.c_[np.log(spots[:, step] / curve.spot), scores, component]
        forecast = _forecast(native, dynamics)
        _, reference, one_day_discount = template.at(1.)
        reference = spots[:, step] * reference / template.spot
        remaining = days - (step + 1)
        maximum_remaining = max(float(np.max(remaining, initial=0.)), 0.)
        next_template = (template.roll(1., new_spot=curve.spot)
                         if step + 1 < horizon or maximum_remaining > 1e-6 else template)
        conditional_prices = np.empty((count, instrument_count))
        selected_prices = np.empty_like(conditional_prices)
        _, representatives, parent_inverse = np.unique(histories[:, step], return_index=True, return_inverse=True)
        integrated_parents[step] = len(representatives)
        path_order = np.argsort(parent_inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(parent_inverse))]
        for begin in range(0, len(representatives), config.integration_parent_batch):
            end = min(begin + config.integration_parent_batch, len(representatives))
            parents = end - begin
            selected_parents = representatives[begin:end]
            branch_spots = reference[selected_parents, None] * (1. + empirical[None, :, 0])
            branch_native = forecast[selected_parents, None, :] + noise[None, :, :]
            decoded = (decode_batch(state, branch_native.reshape(-1, len(state.native)), max_business_dte=maximum_remaining)
                       if maximum_remaining > 1e-6 else None)
            option_prices = cash_europeans_batch(decoded=decoded, template=next_template,
                spots=branch_spots.reshape(-1), business_dtes=remaining, cash_strikes=strikes, is_call=calls)
            branch_prices = np.concatenate([branch_spots[:, :, None],
                option_prices.reshape(parents, support_count, len(book))], axis=2) / initial_spot
            means = branch_prices.mean(axis=1)
            selected_paths = path_order[offsets[begin]:offsets[end]]
            local_parent = parent_inverse[selected_paths] - begin
            conditional_prices[selected_paths] = means[local_parent]
            selected_prices[selected_paths] = branch_prices[local_parent, draws[selected_paths, step]]
            centered = branch_prices - means[:, None, :]
            mean_residual[step] = max(mean_residual[step], float(np.max(np.abs(centered.mean(axis=1)), initial=0.)))
            if progress is not None and (time.monotonic() - last_progress > 30. or end == len(representatives)):
                elapsed = time.monotonic() - started
                completed = step + end / len(representatives)
                progress(f"terminal paths step={step+1}/{horizon} unique_parents={end}/{len(representatives)} paths={count} "
                    f"conditional_branches={support_count} elapsed_s={elapsed:.1f} "
                    f"remaining_s~{elapsed*(horizon-completed)/max(completed,1e-10):.0f}")
                last_progress = time.monotonic()
        spots[:, step + 1] = selected_prices[:, 0] * initial_spot
        native = forecast + noise[draws[:, step]]
        packed = native.reshape(count, len(state.config.business_dtes), state.config.num_inner_strikes + 1)
        lower_clips[:, step] = np.sum(packed < lower, axis=(1, 2))
        upper_clips[:, step] = np.sum(packed > upper, axis=(1, 2))
        prices[:, step + 1] = selected_prices
        current = prices[:, step]
        raw[:, step, 0] = cumulative_discount * one_day_discount * (spots[:, step + 1] - reference) / initial_spot
        raw[:, step, 1:] = cumulative_discount * (one_day_discount * selected_prices[:, 1:] - current[:, 1:])
        removed[:, step, 0] = cumulative_discount * one_day_discount * (conditional_prices[:, 0] - reference / initial_spot)
        removed[:, step, 1:] = cumulative_discount * (one_day_discount * conditional_prices[:, 1:] - current[:, 1:])
        raw[:, step] *= live[step]
        removed[:, step] *= live[step]
        gains[:, step] = raw[:, step] - removed[:, step]
        costs[:, step] = cumulative_discount * fees * live[step]
        zero_prices[step] = int(np.sum((current <= 0.) & live[step]))
        _, histories[:, step + 1] = np.unique(np.c_[histories[:, step], innovation_labels[draws[:, step]]],
            axis=0, return_inverse=True)
        cumulative_discount *= one_day_discount
        template = next_template
    payoffs = np.empty((count, len(claims)))
    terminate = np.zeros((count, horizon, len(claims)), dtype=bool)
    active = np.ones(len(claims), dtype=bool)
    for index, claim in enumerate(claims):
        if claim.kind == "digital":
            payoffs[:, index] = spots[:, -1] > claim.strike if claim.is_call else spots[:, -1] <= claim.strike
        elif claim.kind == "ko":
            observation = claim.observation_day - elapsed_day
            if observation > 0:
                observed = spots[:, observation]
                survives = observed > claim.barrier if claim.is_call else observed < claim.barrier
                terminate[:, observation - 1, index] = ~survives
            else:
                if not claim.observation_done:
                    raise ValueError("A past KO observation must be carried into the daily restart")
                survives = np.full(count, not claim.knocked_out)
                active[index] = not claim.knocked_out
            difference = spots[:, -1] - claim.strike
            payoffs[:, index] = survives * np.maximum(difference if claim.is_call else -difference, 0.) / claim.initial_spot
        else:
            observation = claim.observation_day - elapsed_day
            if observation <= 0 and claim.reset_spot is None:
                raise ValueError("A past ratio reset must carry its observed reset spot")
            reset = claim.reset_spot if observation <= 0 else spots[:, observation]
            payoffs[:, index] = np.abs(spots[:, -1] / reset - claim.strike)
    claim_gains = cumulative_discount * payoffs - discount_to_current * current_claim_marks
    if (not np.isfinite(gains).all() or not np.isfinite(claim_gains).all()
            or np.max(mean_residual, initial=0.) > 1e-10):
        raise ValueError("Full-horizon forecast gains or conditional mean integration failed")
    return TerminalMarketPaths(instrument_gains=gains, raw_instrument_gains=raw,
        removed_conditional_mean=removed, costs=costs, instrument_prices=prices,
        spots=spots, state_features=features, history_ids=histories[:, :-1], live_contracts=live,
        claim_gains=claim_gains, claim_payoffs=payoffs, terminate_after_step=terminate,
        initial_active=active, weights=np.full(count, 1. / count), innovation_indices=draws,
        lower_clips=lower_clips, upper_clips=upper_clips, conditional_mean_residual=mean_residual,
        integrated_parent_counts=integrated_parents,
        zero_live_model_prices=zero_prices, model_date=state.day.isoformat(), seed=seed, elapsed_day=elapsed_day,
        claim_labels=tuple(claim.label for claim in claims))
