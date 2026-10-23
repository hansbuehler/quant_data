"""Audit the capped weekly-Normal model's decoded leverage and front spot law."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import kstest

from quantdata.dysanos.flex.front_spot_copula import (
    antithetic_normals,
    fit_front_spot_copula_from_history,
    sample_correlated_front_scenarios,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import exact_daily_transition
from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _decode_atm_volatilities,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    _leverage_values,
)
from quantdata.dysanos.v11.native_atm_history import load_atm_history
from quantdata.dysanos.v11.radius_cap_diagnostics import decoder_cap_diagnostics


def _pit_summary(scores: np.ndarray) -> dict[str, object]:
    probabilities = ndtr(scores)
    ks = kstest(probabilities, "uniform")
    return {
        "observations": len(scores),
        "gaussian_score_mean": float(np.mean(scores)),
        "gaussian_score_standard_deviation": float(np.std(scores, ddof=1)),
        "pit_mean": float(np.mean(probabilities)),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_iid_naive_pvalue": float(ks.pvalue),
        "pit_quantiles": {
            str(p): float(np.quantile(probabilities, p))
            for p in (0.01, 0.05, 0.50, 0.95, 0.99)
        },
        "tail_frequency": {
            "below_0_01": float(np.mean(probabilities < 0.01)),
            "below_0_05": float(np.mean(probabilities < 0.05)),
            "above_0_95": float(np.mean(probabilities > 0.95)),
            "above_0_99": float(np.mean(probabilities > 0.99)),
        },
    }


def run(
    *,
    checkpoint: Path,
    training_state_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    observation_report: Path,
    output: Path,
    path_count: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Run a one-step model audit at the final pre-2025 fitted state."""
    if path_count <= 0 or path_count % 2 or batch_size <= 0:
        raise ValueError("path count must be even and batch size positive")
    model, checkpoint_audit = load_capped_joint_model(checkpoint, device=device)
    history = torch.load(training_state_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_tensor = history.get("states")
    if not isinstance(dates, list) or not isinstance(states_tensor, torch.Tensor):
        raise ValueError("invalid pre-2025 state history")
    states = states_tensor.double().numpy()
    if states.shape != tuple(model.daily_states.shape):
        raise ValueError("pre-2025 states do not match the checkpoint")
    observed_dates, direct_dtes, _, simple_spot = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if list(observed_dates[: len(dates)]) != dates:
        raise ValueError("observed direct-SANOS dates do not match training states")
    gross_train = 1.0 + simple_spot[: len(dates) - 1]
    normalized_spots = np.concatenate(([1.0], np.cumprod(gross_train)))
    daily_transition_t, daily_factor_t, transition_diagnostics = exact_daily_transition(model)
    daily_transition = daily_transition_t.detach().cpu().double().numpy()
    daily_factor = daily_factor_t.detach().cpu().double().numpy()
    copula, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=normalized_spots,
        dates=dates,
        daily_transition=daily_transition,
        daily_factor=daily_factor,
        fit_before=date(2025, 1, 1),
    )
    print("front spot copula fit complete; decoding training ATM IV", flush=True)
    decoded_atm = _decode_atm_volatilities(model, states, batch_size=batch_size)
    model_dtes = np.asarray(model.config.geometry.model_expiries_y) * 252.0
    indices = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(model_dtes, tenor, rtol=0.0, atol=1.0e-9))
        if len(matches) != 1 or not np.any(np.isclose(direct_dtes, tenor)):
            raise ValueError(f"audit tenor {tenor}D is unavailable")
        indices.append(int(matches[0]))
    decoded = decoded_atm[:, indices]
    observed_leverage = _leverage_values(
        np.log(gross_train), np.diff(np.log(decoded), axis=0)
    )
    scores = _historical_spot_scores(
        model, states[:-1], gross_train, batch_size=batch_size
    )
    print("front PIT calculated; generating terminal one-step scenarios", flush=True)
    state_normals, spot_normals = antithetic_normals(
        count=path_count, dimension=model.config.state_size, seed=seed
    )
    next_states, gross_returns = sample_correlated_front_scenarios(
        model=model,
        current_state=states[-1],
        daily_transition=daily_transition,
        daily_factor=daily_factor,
        fit=copula,
        state_normals=state_normals,
        independent_spot_normals=spot_normals,
    )
    next_atm = _decode_atm_volatilities(model, next_states, batch_size=batch_size)
    spot_log_returns = np.log(gross_returns)
    log_iv_changes = np.log(next_atm[:, indices] / decoded[-1][None, :])
    simulated_leverage = _leverage_values(spot_log_returns, log_iv_changes)
    observed_report = json.loads(observation_report.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "schema": "dysanos-v11-capped-weekly-normal-one-step-v1",
        "checkpoint": checkpoint_audit,
        "training_state_history": str(training_state_history.resolve()),
        "observed_atm_history": str(historical_sanos_history.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "simulation_origin_date": dates[-1],
        "simulation_path_count": path_count,
        "simulation_independent_antithetic_pairs": path_count // 2,
        "simulation_seed": seed,
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "historical_pre2025_correlation": observed_report["pre_evaluation"]["leverage"]["correlation"],
        "historical_2025_correlation": observed_report["forward_evaluation"]["leverage"]["correlation"],
        "decoded_training_leverage": {
            "correlation": observed_leverage[0].tolist(),
            "covariance": observed_leverage[1].tolist(),
            "slope": observed_leverage[2].tolist(),
        },
        "simulated_terminal_one_step_leverage": {
            "correlation": simulated_leverage[0].tolist(),
            "covariance": simulated_leverage[1].tolist(),
            "slope": simulated_leverage[2].tolist(),
            "antithetic_pair_correlation_standard_error": (
                _antithetic_correlation_standard_errors(
                    spot_log_returns, log_iv_changes
                ).tolist()
            ),
        },
        "historical_training_front_pit": _pit_summary(scores),
        "simulated_spot_log_return_quantiles": {
            str(p): float(np.quantile(spot_log_returns, p))
            for p in (0.01, 0.05, 0.50, 0.95, 0.99)
        },
        "simulated_next_state_decoder_cap": decoder_cap_diagnostics(model, next_states),
        "transition_diagnostics": transition_diagnostics,
        "front_spot_copula": copula.to_dict(),
        "front_spot_copula_diagnostics": copula_diagnostics,
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"capped weekly-Normal leverage audit complete: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
