"""Fit a small direct daily state/spot transition at the 2024-12-31 origin.

The quote decoder and its 200-dimensional latent-state gauge are frozen.  This
is intentionally a one-origin experiment: only the preceding 252 dates enter
the dynamics fit, and the next session is never used for fitting or tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.v11.atm_spot_correlation import _decode_atm_volatilities
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _terminal_front_gross_returns
from quantdata.torch import BaseEnvironment, BaseModel


@dataclass(frozen=True, kw_only=True)
class DirectPilotConfig:
    """Small direct map geometry, fixed before observing the terminal draw."""

    factor_rank: int = 8
    hidden_width: int = 16
    validation_transitions: int = 42
    max_epochs: int = 2_000
    learning_rate: float = 2.0e-3
    weight_decay: float = 2.0e-3
    patience: int = 120


class DirectDailySpotMap(BaseModel):
    """Direct conditional mean of the next low-rank DYSANOS state."""

    def __init__(
        self,
        env: BaseEnvironment,
        input_size: int,
        output_size: int,
        hidden_width: int,
    ) -> None:
        super().__init__(env)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, output_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def run(
    *,
    checkpoint: Path,
    target_history: Path,
    checkpoint_output: Path,
    report_output: Path,
    lookback_dates: int = 252,
    paths: int = 10_000,
    seed: int = 20261019,
    config: DirectPilotConfig = DirectPilotConfig(),
) -> dict[str, object]:
    """Select then refit a direct transition using only the trailing 252 dates."""
    if paths <= 0 or paths % 2 or lookback_dates < 64:
        raise ValueError("paths must be positive/even and lookback_dates at least 64")
    torch.manual_seed(seed)
    model, source_audit = load_capped_joint_model(checkpoint, device="cpu")
    source = torch.load(target_history, map_location="cpu", weights_only=False)
    states = source["states"].double().numpy()
    dates = source["dates"]
    scores = np.asarray(source["front_spot_scores"], dtype=np.float64)
    if len(states) < lookback_dates or scores.shape != (lookback_dates - 1,):
        raise ValueError("pilot expects the 251 spot scores saved for the 252-date daily window")
    states = states[-lookback_dates:]
    dates = dates[-lookback_dates:]
    origin, endpoint = states[:-1], states[1:]
    state_mean = origin.mean(axis=0)
    _, _, right = np.linalg.svd(origin - state_mean, full_matrices=False)
    basis = right[: config.factor_rank].T
    current = (origin - state_mean) @ basis
    next_value = (endpoint - state_mean) @ basis
    input_mean = np.concatenate((current.mean(axis=0), [scores.mean()]))
    input_std = np.maximum(np.concatenate((current.std(axis=0), [scores.std()])), 1.0e-6)
    target_mean = next_value.mean(axis=0)
    target_std = np.maximum(next_value.std(axis=0), 1.0e-6)
    inputs = np.column_stack((current, scores))
    inputs = (inputs - input_mean) / input_std
    targets = (next_value - target_mean) / target_std
    split = len(inputs) - config.validation_transitions
    if split <= config.factor_rank + 20:
        raise ValueError("insufficient training transitions after validation split")

    def fit(*, train_count: int, epochs: int, select: bool) -> tuple[DirectDailySpotMap, int, float]:
        net = DirectDailySpotMap(
            BaseEnvironment(precision=32, device="cpu", debug=False),
            config.factor_rank + 1,
            config.factor_rank,
            config.hidden_width,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        x = torch.as_tensor(inputs[:train_count], dtype=torch.float32)
        y = torch.as_tensor(targets[:train_count], dtype=torch.float32)
        xv = torch.as_tensor(inputs[train_count:], dtype=torch.float32)
        yv = torch.as_tensor(targets[train_count:], dtype=torch.float32)
        best_state, best_value, best_epoch, stale = None, float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(x) - y).square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                value = float(torch.mean((net(xv) - yv).square())) if select else float(loss)
            if value < best_value - 1.0e-6:
                best_state = {key: item.detach().clone() for key, item in net.state_dict().items()}
                best_value, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if select and stale >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("direct pilot did not produce a finite state")
        net.load_state_dict(best_state)
        return net, best_epoch, best_value

    selected, selected_epochs, validation_mse = fit(train_count=split, epochs=config.max_epochs, select=True)
    # Refit for exactly the validation-selected horizon on all 251 transitions.
    final, _, train_mse = fit(train_count=len(inputs), epochs=selected_epochs, select=False)
    with torch.no_grad():
        fitted_next = final(torch.as_tensor(inputs, dtype=torch.float32)).double().numpy()
    residual = (targets - fitted_next) * target_std
    covariance = np.cov(residual, rowvar=False, ddof=1) + 1.0e-8 * np.eye(config.factor_rank)
    residual_factor = np.linalg.cholesky(covariance)

    generator = np.random.default_rng(seed + 1)
    half = paths // 2
    spot_half = generator.standard_normal(half)
    spot_standardized = np.concatenate((spot_half, -spot_half))
    terminal_scores = scores.mean() + scores.std(ddof=1) * spot_standardized
    terminal_current = (states[-1] - state_mean) @ basis
    terminal_input = (np.column_stack((np.repeat(terminal_current[None], paths, axis=0), terminal_scores)) - input_mean) / input_std
    with torch.no_grad():
        terminal_factor = final(torch.as_tensor(terminal_input, dtype=torch.float32)).double().numpy()
    shock_half = generator.standard_normal((half, config.factor_rank)) @ residual_factor.T
    shock = np.concatenate((shock_half, -shock_half))
    next_states = state_mean + (terminal_factor * target_std + target_mean + shock) @ basis.T
    current_atm = _decode_atm_volatilities(model, states[-1:], batch_size=1)[0]
    next_atm = _decode_atm_volatilities(model, next_states, batch_size=256)
    log_spot = np.log(_terminal_front_gross_returns(model, states[-1], terminal_scores))
    leverage = [_correlation(log_spot, np.log(next_atm[:, j]) - np.log(current_atm[j])) for j in range(next_atm.shape[1])]

    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "dysanos-v11-direct-daily-spot-pilot-v1", "config": asdict(config), "state_dict": final.state_dict(), "basis": basis, "state_mean": state_mean, "input_mean": input_mean, "input_std": input_std, "target_mean": target_mean, "target_std": target_std, "residual_factor": residual_factor, "dates": dates}, checkpoint_output)
    report: dict[str, object] = {"schema": "dysanos-v11-direct-daily-spot-pilot-v1", "status": "complete", "scope": "252 dates through 2024-12-31; terminal 2025-01-02 outcome excluded", "source_audit": source_audit, "window": [dates[0], dates[-1]], "transitions": len(inputs), "validation_transitions": config.validation_transitions, "selected_epochs": selected_epochs, "validation_mse_standardized": validation_mse, "full_window_mse_standardized": train_mse, "config": asdict(config), "paths": paths, "terminal_atm_spot_correlations": leverage, "business_dtes": (np.asarray(model.config.geometry.model_expiries_y) * 252.0).tolist(), "checkpoint": str(checkpoint_output.resolve())}
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=10_000)
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"direct daily spot pilot complete: {report['report_output'] if 'report_output' in report else arguments.report_output}", flush=True)


if __name__ == "__main__":
    main()
