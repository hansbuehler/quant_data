from __future__ import annotations

from pathlib import Path

from quantdata.gromm2.million_forward_experiment import experiment_stages


def test_stage_plan_has_matched_randomness_and_the_requested_order(tmp_path: Path) -> None:
    stages = experiment_stages(run_directory=tmp_path, history=tmp_path / "history.h5",
        estimate=tmp_path / "estimate.npz", factor_law=tmp_path / "law.npz", paths=1_000_000,
        batch_size=1024, seed=2026102801, maximum_epochs=2000, compile_policies=True)
    assert [name for name, _ in stages] == ["black5-sampling", "black0-sampling",
        "black5-with_fixing_options", "black5-without_fixing_options",
        "black0-with_fixing_options", "black0-without_fixing_options"]
    first, second = stages[0][1], stages[1][1]
    assert first[first.index("--seed") + 1] == second[second.index("--seed") + 1]
    assert "--background-volatility" not in first
    assert second[second.index("--background-volatility") + 1] == "0"
    assert "--without-fixing-options" not in stages[2][1]
    assert "--without-fixing-options" in stages[3][1]
    assert all("--compile" in command for _, command in stages[2:])
