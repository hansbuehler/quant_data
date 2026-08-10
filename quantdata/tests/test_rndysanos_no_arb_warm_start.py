from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from quantdata.dysanos.grid.rndysanos import (
    DysanosConfig,
    DysanosModel,
    TrainConfig,
    UniverseGeometry,
    initialize_no_arb_action,
)
from quantdata.dysanos.grid.rndysanos.run_cached_expiry_oce import (
    _validate_generated_market_report,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _report(path: Path) -> None:
    candidate = {
        "state_region": {"center": [0.5, -0.5, 1.0]},
        "reference_solution": {"action": [0.4, 0.1, -0.1]},
        # Held-out fields must not enter initialization.
        "audit_loss_count": 99,
        "certified": False,
    }
    value = {
        "configuration": {"return_horizon": "expiry", "hedge": "raw"},
        "conditional": {
            "times": [
                {
                    "state_normalization": {
                        "reference_only_mean": [1.0, 2.0, 3.0],
                        "reference_only_population_scale": [2.0, 4.0, 0.5],
                    },
                    "local_conditional_baskets": {
                        "surviving": [],
                        "rejected": [candidate],
                        "unresolved": [],
                    },
                }
            ]
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_no_arb_basket_warm_start_is_interior_and_learns_direction(
    tmp_path: Path,
) -> None:
    report = tmp_path / "no-arb.json"
    _report(report)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = UniverseGeometry(
        samples=4,
        steps=1,
        expiries=1,
        strikes=2,
        state_variables=3,
        include_contract_features=False,
    )
    config = DysanosConfig(
        label="test-no-arb-init",
        state_mode="raw",
        trade_spot=False,
        cost_cond=0.0,
        pf_config=FeedForwardConfig(
            num_depth=1,
            num_width=8,
            activation="LeakyReLU",
            use_bias=True,
        ),
    )
    model = DysanosModel(env, geometry=geometry, config=config)

    diagnostics = initialize_no_arb_action(
        model,
        report,
        epochs=300,
        batch_size=None,
        lr=1.0e-2,
        maximum_cap_fraction=0.25,
    )

    assert diagnostics["target_count"] == 1
    assert diagnostics["source_rejected_candidates"] == 1
    assert diagnostics["source_audit_outcomes_used"] is False
    assert diagnostics["maximum_target_cap_fraction"] <= 0.25
    assert diagnostics["minimum_target_tanh_derivative"] >= 0.9375
    assert diagnostics["final_relative_mse"] < diagnostics["first_relative_mse"]
    assert diagnostics["mean_direction_cosine"] > 0.99
    assert diagnostics["minimum_fitted_tanh_derivative"] > 0.8

    state = torch.tensor([[2.0, 0.0, 3.5]], dtype=env.fdtype)
    action = model.action(state)
    assert action[0, 0].abs().item() < 1.0e-2
    assert action[0, 1].item() > 0.0
    assert action[0, 2].item() < 0.0


def test_no_arb_report_requires_initialization_epochs() -> None:
    with pytest.raises(ValueError, match="requires positive initialization epochs"):
        TrainConfig(action_initialization_report="report.json")


@pytest.mark.parametrize("frequency", ["daily", "weekly"])
def test_cached_expiry_oce_accepts_supported_frequencies(frequency: str) -> None:
    report = {
        "status": "complete",
        "configuration": {
            "generator": "dysanos",
            "trading_frequency": frequency,
            "return_horizon": "expiry",
            "hedge": "delta",
            "cost_rate": 1.0e-3,
            "path_count": 1_000_000,
        },
    }

    configuration = _validate_generated_market_report(
        report,
        expected_hedge="delta",
        cost_prop=1.0e-3,
    )

    assert configuration["trading_frequency"] == frequency


def test_cached_expiry_oce_rejects_cost_mismatch() -> None:
    report = {
        "status": "complete",
        "configuration": {
            "generator": "dysanos",
            "trading_frequency": "weekly",
            "return_horizon": "expiry",
            "hedge": "delta",
            "cost_rate": 1.0e-4,
            "path_count": 1_000_000,
        },
    }

    with pytest.raises(ValueError, match="cost 0.001"):
        _validate_generated_market_report(
            report,
            expected_hedge="delta",
            cost_prop=1.0e-3,
        )
