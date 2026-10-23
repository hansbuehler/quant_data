"""Audit the fitted front-SANOS next-session spot law before and during 2025."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.factorial_statistics import transition_sample_indices
from quantdata.dysanos.v11.native_atm_history import load_atm_history
from quantdata.dysanos.v11.run_capped_baseline_leverage import _pit_summary
from quantdata.dysanos.v11.run_predictive_panels import _history


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output: Path,
    forward_state_history: Path | None = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, object]:
    """Use each origin's fitted state, never the endpoint state, for its PIT."""
    model, audit = load_capped_joint_model(joint_checkpoint, device=device)
    dates, states = _history(training_state_history)
    if forward_state_history is not None:
        later_dates, later_states = _history(forward_state_history)
        if later_dates[0] <= dates[-1]:
            raise ValueError("forward dates must follow training dates")
        dates += later_dates
        states = np.concatenate((states, later_states), axis=0)
    observed_dates, _, _, simple_spot = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if list(observed_dates[: len(dates)]) != dates:
        raise ValueError("front PIT dates and fitted-state dates do not align")
    gross_returns = 1.0 + simple_spot[: len(dates) - 1]
    scores = _historical_spot_scores(
        model, states[:-1], gross_returns, batch_size=batch_size
    )
    if forward_state_history is None:
        pre = np.arange(1, len(dates))
        forward = np.empty(0, dtype=np.int64)
    else:
        pre, forward = transition_sample_indices(np.asarray(dates))
    report: dict[str, object] = {
        "schema": "dysanos-v11-front-spot-pit-v1",
        "checkpoint": audit,
        "pre_evaluation": _pit_summary(scores[pre - 1]),
        "forward_evaluation": (
            None if not len(forward) else _pit_summary(scores[forward - 1])
        ),
        "forward_state_history": (
            None if forward_state_history is None else str(forward_state_history.resolve())
        ),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-state-history", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"front PIT audit complete: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
