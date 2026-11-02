from __future__ import annotations

import json
import sys

import pytest

from quantdata.gromm2 import plot_hedging_training as plotting


def test_subtasks_cli_renders_three_independent_portfolios(tmp_path, monkeypatch, capsys) -> None:
    report = {
        "epoch": 1,
        "config": {"label": "combined-test", "risk_aversion": 100.},
        "scenario_model": {"background_volatility": .05, "simulated_background_volatility": 0.},
        "training_paths": 8,
        "validation_paths": 8,
        "policies": {name: {"best_epoch": 0} for name in ("zero", "short", "long")},
    }
    rows = [{"epoch": epoch, **{split: {
        name: {"risk_at_training_translation": value + epoch * .001}
        for name, value in (("zero", 0.), ("short", .02), ("long", -.015))}
        for split in ("train", "validation")}} for epoch in (0, 1)]
    (tmp_path / "training_report.json").write_text(json.dumps(report))
    (tmp_path / "training_history.json").write_text(json.dumps({"epochs": rows}))
    output = tmp_path / "independent.png"
    monkeypatch.setattr(sys, "argv", ["plot_hedging_training", "--policy-directory", str(tmp_path),
        "--output", str(output), "--subtasks"])

    plotting.main()

    result = json.loads(capsys.readouterr().out)
    assert result["completed_epochs"] == 1
    assert result["best_epochs"] == {"zero": 0, "short": 0, "long": 0}
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(output.with_suffix(".json").read_text()) == result


def test_subtasks_cli_keeps_joint_two_portfolio_dispatch(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "training_report.json").write_text(json.dumps({"shared_parameters": True}))
    output = tmp_path / "joint.png"
    observed = []

    def joint_plot(directory, *, output):
        observed.append((directory, output))
        return {"plot": str(output)}

    monkeypatch.setattr(plotting, "plot_joint_losses", joint_plot)
    monkeypatch.setattr(plotting, "plot_independent_losses", lambda *args, **kwargs: pytest.fail("wrong portfolio schema"))
    monkeypatch.setattr(sys, "argv", ["plot_hedging_training", "--policy-directory", str(tmp_path),
        "--output", str(output), "--subtasks"])

    plotting.main()

    assert observed == [(tmp_path, output)]
    assert json.loads(capsys.readouterr().out) == {"plot": str(output)}
