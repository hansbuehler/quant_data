"""Tests for the short fixed-chart path experiment."""

from __future__ import annotations

from quantdata.dysanos.flex.run_fixed_chart_multistep_paths import Config, run


def test_two_step_path_smoke_run_is_conditionally_admissible(tmp_path) -> None:
    report = run(
        Config(
            output=tmp_path / "fixed-chart-paths",
            num_paths=32,
            num_steps=2,
            path_batch_size=16,
        )
    )

    assert report["terminal_admissible_paths"] == 32
    assert len(report["steps"]) == 2
    for step in report["steps"]:
        assert step["strictly_admissible_conditional_state_fraction"] == 1.0
        assert step["sampled_admissible_branch_fraction"] == 1.0
        assert step["maximum_conditional_selected_call_error"] < 1.0e-12
        assert step["minimum_density"] > 0.0
        assert step["minimum_dlvsq_margin"] > 0.0
