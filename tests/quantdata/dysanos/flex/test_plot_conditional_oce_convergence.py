"""Tests for conditional-OCE convergence plots."""

from __future__ import annotations

import json

import numpy as np

from quantdata.dysanos.flex.plot_conditional_oce_convergence import load_stage


def test_load_stage_counts_full_population_evaluations(tmp_path) -> None:
    report = tmp_path / "convergence.json"
    report.write_text(
        json.dumps(
            {
                "training_history": [
                    {
                        "epoch": 1,
                        "full_population_oce": 0.1,
                        "gradient_rms": 0.03,
                    },
                    {
                        "epoch": 2,
                        "full_population_log_loss": -0.2,
                        "gradient_rms": 0.01,
                        "lbfgs_closure_evaluations": 7,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    stage = load_stage("combined", report)

    np.testing.assert_array_equal(stage.cumulative_evaluations, [1.0, 8.0])
    np.testing.assert_array_equal(stage.oce_objective, [0.1, 0.2])
    np.testing.assert_array_equal(stage.gradient_rms, [0.03, 0.01])
