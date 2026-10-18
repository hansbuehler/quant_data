"""Checks for the corrected statistical tensor launch specification."""

from __future__ import annotations

from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.runners.run_corrected_statistical_tensor import (
    EXISTING,
    _cases,
    _command,
)


def test_corrected_tensor_has_one_case_per_statistical_grid_point() -> None:
    cases = _cases()
    assert len(cases) == 26
    assert len({case[3] for case in cases}) == 26
    assert len([case for case in cases if case[3] not in EXISTING]) == 24
    assert all(case_id.endswith("-demean") for _, _, _, case_id in cases)


def test_case_command_pins_public_data_and_mean_reduction() -> None:
    command = _command(
        Path("/project/bin/python"),
        model="covariance",
        objective="mean-variance",
        risk=1.0,
        output=Path("/data/case"),
    )
    assert command[0] == "/project/bin/python"
    assert command[command.index("--mean-reduction") + 1] == "demean"
    assert command[command.index("--target-maturity") + 1] == "1"
    assert command[command.index("--target-notional-artifact-sha256") + 1]
    assert command[command.index("--dataset-manifest-sha256") + 1]
