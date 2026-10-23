"""Exact weekly-radius and raw-coordinate cap checks."""

from datetime import date

import numpy as np
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import JointDirectDynamicsConfig
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.v11.radius_cap_diagnostics import (
    decoder_cap_diagnostics,
    innovation_radius_panels,
)
from quantdata.torch import BaseEnvironment


def test_weekly_radii_use_horizon_covariance_and_endpoint_split() -> None:
    dates = tuple(get_business_days(date(2024, 11, 1), date(2025, 3, 31)))
    states = np.column_stack((np.sin(np.arange(len(dates))), np.cos(np.arange(len(dates)))))
    report = innovation_radius_panels(
        states,
        dates,
        mean=np.zeros(2),
        weekly_transition=0.5 * np.eye(2),
        weekly_innovation_factor=np.eye(2),
    )
    assert set(report) == {1, 2, 3, 4}
    assert report[1]["pre_evaluation"]["observations"] > 0
    assert report[4]["forward_evaluation"]["observations"] > 0
    assert all(report[week]["gaussian_p99"] > 0.0 for week in report)


def test_decoder_cap_audit_counts_the_full_dlv_book() -> None:
    model = JointDirectDynamicsModel(
        BaseEnvironment(precision=32, device="cpu", debug=False),
        num_days=2,
        config=JointDirectDynamicsConfig(label="cap-audit-test", state_size=2),
    )
    result = decoder_cap_diagnostics(model, torch.zeros((3, 2)), batch_size=2)
    assert result["states"] == 3
    assert result["dlv_coordinates"] == (
        3 * model.config.geometry.num_model_expiries * model.config.geometry.num_inner_strikes
    )
    assert 0.0 <= result["fraction_at_hard_cap"] <= 1.0
