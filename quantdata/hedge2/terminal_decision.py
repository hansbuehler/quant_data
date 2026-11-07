"""Actual-inventory decision boundary for daily restarted terminal-risk hedges."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.cash import CONTRACT_KEY
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_adequacy import assess_zero_action_reference
from quantdata.hedge2.terminal_features import rolling_feature_mapping
from quantdata.hedge2.terminal_market import TerminalMarketPaths
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig, policy_book
from quantdata.hedge2.terminal_train import (
    TerminalPolicyFit,
    TerminalTrainConfig,
    audit_fitted_policy,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TerminalDecision:
    fit: TerminalPolicyFit
    state: DailyState
    dynamics: FittedDynamics
    root_spot: float
    elapsed_day: int
    claim_label: str
    initial_spot: float
    claim: Claim
    contracts: pd.DataFrame
    positions: np.ndarray
    premium_positions: np.ndarray
    accepted_for_execution: bool


def fit_terminal_decision(*, simulations: TerminalMarketPaths, contracts: pd.DataFrame,
                          claim: Claim, claim_index: int, state: DailyState, dynamics: FittedDynamics,
                          initial_inventory: np.ndarray, model_config: TerminalPolicyConfig,
                          train_config: TerminalTrainConfig, env: BaseEnvironment,
                          objective: str, risk_aversion: float, target_side: int,
                          output_directory: Path, previous_decision: TerminalDecision | None = None,
                          previous_origin_decision: TerminalDecision | None = None,
                          diagnostic_only: bool = False) -> TerminalDecision:
    """Fit today's remaining horizon and return only its common-root action.

    The caller must retain the inception contracts, cash strikes, observation
    state and numeraire. Inventory is the ACTUAL carried fixed-contract
    coefficient vector, never tomorrow's desired allocation. Exogenous
    simulations freeze today's fitted AR law, not tomorrow's realized fit.
    The next actual date gets its newly fitted state/dynamics and initializes
    the neural weights from this decision with an explicit feature map.

    A budget-limited fit is saved and returned as a candidate, but is NOT
    accepted for production execution. No untouched validation sample is
    consumed by this training boundary.
    """
    paths = simulations.for_claim(claim, claim_index)
    root_spot = float(simulations.spots[0, 0])
    normalized_root_price = float(simulations.instrument_prices[0, 0, 0])
    if (simulations.model_date != state.day.isoformat() or state.day != dynamics.as_of_date
            or dynamics.training_end > state.day or dynamics.training_start.year < 2022
            or model_config.feature_count != paths.features.shape[2] or not paths.initially_active
            or paths.gains.shape[1] != claim.terminal_day - simulations.elapsed_day
            or not np.isfinite([root_spot, normalized_root_price]).all() or min(root_spot, normalized_root_price) <= 0
            or not np.isclose(root_spot / normalized_root_price, claim.initial_spot, rtol=1e-12)
            or not np.isclose(claim.initial_spot, root_spot) and simulations.elapsed_day == 0):
        raise ValueError("Terminal decisions need today's causal model, compatible claim units and an active episode")
    minimum_paths = max(10000, 10 * paths.gains.shape[2])
    if len(paths.weights) < minimum_paths and not diagnostic_only:
        raise ValueError(f"Production terminal training requires at least {minimum_paths} complete simulated paths")
    if len(dynamics.residuals) != 252 and not diagnostic_only:
        raise ValueError("Production terminal models must use exactly 252 fitted one-session innovations")
    if previous_decision is not None and previous_origin_decision is not None:
        raise ValueError("Choose same-episode or previous-origin initialization, not both")
    if any(name not in contracts for name in (*CONTRACT_KEY, "instrument_kind", "bdte")):
        raise ValueError("Terminal decisions must retain all fixed contract identifiers")
    book = policy_book(book=contracts, root_spot=root_spot, prices=simulations.instrument_prices[:, :-1])
    mapping, previous_policy = None, None
    if previous_decision is not None:
        expected_claim = previous_decision.claim.observe(elapsed_day=simulations.elapsed_day, spot=root_spot)
        old_book = previous_decision.contracts
        expected_book = old_book.loc[old_book["bdte"] > 1. + 1e-6].reset_index(drop=True)
        keys = [*CONTRACT_KEY, "instrument_kind"]
        expected_keys, current_keys = expected_book[keys].copy(), contracts[keys].reset_index(drop=True).copy()
        expected_keys["expiry"] = pd.to_datetime(expected_keys["expiry"]).astype("datetime64[ns]")
        current_keys["expiry"] = pd.to_datetime(current_keys["expiry"]).astype("datetime64[ns]")
        if (expected_claim != claim or previous_decision.elapsed_day + 1 != simulations.elapsed_day
                or len(get_business_days(previous_decision.state.day, state.day)) != 2
                or not expected_keys.equals(current_keys)
                or not np.allclose(expected_book["bdte"] - 1., contracts["bdte"], rtol=0., atol=1e-6)):
            raise ValueError("Daily restart initialization must come from the previous decision of this same episode")
        mapping = rolling_feature_mapping(previous_state=previous_decision.state, previous_model=previous_decision.dynamics,
            current_state=state, current_model=dynamics, previous_root_spot=previous_decision.root_spot,
            current_root_spot=root_spot, elapsed_step_offset=1)
        previous_policy = previous_decision.fit.policy
    elif previous_origin_decision is not None:
        if (simulations.elapsed_day != 0 or previous_origin_decision.elapsed_day != 0
                or previous_origin_decision.claim_label != claim.label):
            raise ValueError("Previous-origin initialization requires an earlier inception policy for the same target label")
        mapping = rolling_feature_mapping(previous_state=previous_origin_decision.state,
            previous_model=previous_origin_decision.dynamics, current_state=state, current_model=dynamics,
            previous_root_spot=previous_origin_decision.root_spot, current_root_spot=root_spot)
        matrix = mapping.matrix.copy()
        if claim.kind == "ratio_straddle":
            matrix[-1, -2] += np.log(claim.initial_spot / previous_origin_decision.initial_spot)
        mapping = replace(mapping, matrix=matrix)
        previous_policy = previous_origin_decision.fit.policy
    elif simulations.elapsed_day > 0 and not (output_directory / f"{model_config.filename}.pt").exists():
        raise ValueError("A restarted episode must supply the preceding fitted policy; no silent cold start")
    fit = train_terminal_policy(paths=paths, book=book, initial_inventory=initial_inventory,
        model_config=model_config, train_config=train_config, env=env, objective=objective,
        risk_aversion=risk_aversion, target_side=target_side, as_of_date=state.day, target_key=claim.label,
        output_directory=output_directory, previous_policy=previous_policy, feature_mapping=mapping,
        progress=lambda message: print(message, flush=True))
    audit = (None if fit.status == "checkpointed_stop" else audit_fitted_policy(policy=fit.policy,
        paths=paths, book=book, initial_inventory=initial_inventory, objective=objective,
        risk_aversion=risk_aversion, target_side=target_side))
    adequacy = (None if audit is None else assess_zero_action_reference(paths=paths,
        initial_inventory=initial_inventory, fitted_risk=audit["risk"], objective=objective,
        risk_aversion=risk_aversion, target_side=target_side, precision=torch.finfo(env.fdtype).bits))
    root_book = replace(book, prices=book.prices[:1]).tensor(env)
    with torch.no_grad():
        positions = fit.policy(torch.tensor(paths.features[:1, 0], dtype=env.fdtype, device=env.device),
            torch.tensor(initial_inventory[None, :], dtype=env.fdtype, device=env.device), 0,
            book=root_book, costs=torch.tensor(paths.costs[:1, 0], dtype=env.fdtype, device=env.device))[0].cpu().numpy()
    positions = np.where(paths.live_contracts[0], positions, 0.)
    allocations = positions * book.prices[0, 0]
    if (not np.isfinite(positions).all()
            or np.max(np.abs(allocations)) > model_config.instrument_cap * (1. + 64. * torch.finfo(env.fdtype).eps)):
        raise RuntimeError("A terminal root candidate violates finite holdings or its premium-dollar limit")
    accepted = (fit.status == "training_stable" and not diagnostic_only and adequacy is not None
        and adequacy["no_worse_than_zero_reference"])
    metadata = {"model_date": simulations.model_date, "elapsed_day": simulations.elapsed_day,
        "claim": claim.label, "target_side": target_side, "objective": objective, "risk_aversion": risk_aversion,
        "fit_status": fit.status, "accepted_for_execution": accepted, "diagnostic_only": diagnostic_only,
        "training_paths": len(paths.weights), "required_paths": minimum_paths,
        "training_transitions": len(dynamics.residuals),
        "latest_fitted_transition": dynamics.training_end.isoformat(),
        "latest_transition_session_lag": len(get_business_days(dynamics.training_end, state.day)) - 1,
        "initialization_scope": ("previous_episode_step" if previous_decision is not None
            else "previous_origin" if previous_origin_decision is not None else "checkpoint_or_initial_fit"),
        "initialisation_date": None if fit.initialisation_date is None else fit.initialisation_date.isoformat(),
        "current_root_spot": root_spot, "episode_initial_spot": claim.initial_spot,
        "coefficient_units": "cash-unit holdings multiplied by episode initial spot",
        "frozen_training_policy_audit": audit,
        "zero_action_reference_audit": adequacy,
        "old_subspace_projection_residual": None if mapping is None else mapping.old_subspace_projection_residual}
    temporary = output_directory / "root_decision.partial.npz"
    np.savez_compressed(temporary, positions=positions, premium_positions=allocations,
        cash_unit_holdings=positions / claim.initial_spot, initial_inventory=initial_inventory,
        current_prices_in_initial_spot_units=book.prices[0, 0], metadata_json=json.dumps(metadata))
    temporary.replace(output_directory / "root_decision.npz")
    return TerminalDecision(fit=fit, state=state, dynamics=dynamics, root_spot=root_spot,
        elapsed_day=simulations.elapsed_day, claim_label=claim.label, initial_spot=claim.initial_spot,
        claim=claim, contracts=contracts.copy(),
        positions=positions, premium_positions=allocations, accepted_for_execution=accepted)
