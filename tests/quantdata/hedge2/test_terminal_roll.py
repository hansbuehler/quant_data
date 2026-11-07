"""Rolling orchestration preserves causal inception warm starts and explicit exclusions."""

import errno
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2 import terminal_roll
from quantdata.hedge2.terminal_market import TerminalSimulationConfig
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig
from quantdata.hedge2.terminal_roll import (
    agreed_terminal_claims,
    market_data_unavailable,
    terminal_rules,
)
from quantdata.hedge2.terminal_train import TerminalTrainConfig
from quantdata.torch import BaseEnvironment


def test_default_tensor_covers_all_48_targets_both_sides_and_requested_objectives() -> None:
    labels = agreed_terminal_claims()
    rules = terminal_rules(claims=labels)
    assert len(labels) == 48
    assert len(rules) == 48 * 2 * 9
    assert len({rule.filename for rule in rules}) == len(rules)
    assert all(rule.objective == "mean_variance" for rule in rules if np.isinf(rule.risk_aversion))


@pytest.mark.parametrize("changes", [
    {"claims": ("unknown",)}, {"claims": ()}, {"risk_aversions": (0.,)},
    {"risk_aversions": (np.nan,)}, {"objectives": ("unknown",)}, {"sides": (0,)},
    {"objectives": ("power_exponential",), "risk_aversions": (np.inf,)},
])
def test_tensor_rejects_unagreed_or_empty_scope(changes) -> None:
    with pytest.raises(ValueError):
        terminal_rules(**{"claims": agreed_terminal_claims()[:1], **changes})


def test_only_missing_external_sources_and_defined_quote_gaps_are_data_exclusions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    assert market_data_unavailable(FileNotFoundError(errno.ENOENT, "missing", str(source / "2024-04-22.npz")),
        source_directories=(source,))
    assert not market_data_unavailable(FileNotFoundError(errno.ENOENT, "missing", str(source / "config.json")),
        source_directories=(source,))
    assert not market_data_unavailable(FileNotFoundError(errno.ENOENT, "missing", str(tmp_path / "policy.pt")),
        source_directories=(source,))
    assert market_data_unavailable(ValueError("Missing observed mids on 2024-04-22: contract"))
    assert market_data_unavailable(ValueError("No PM expiry at or after five business days"))
    assert not market_data_unavailable(ValueError("Non-finite gradient"))
    assert not market_data_unavailable(ValueError("Saved terminal forecast inputs changed"))


def rolling_inputs(monkeypatch, root: Path):
    sources = {}
    for name in ("source_root", "market_directory", "licensed_fit_directory", "raw_quote_directory",
                 "daily_directory", "dynamics_directory"):
        directory = root / name
        directory.mkdir()
        sources[name] = str(directory)
    days = (date(2024, 4, 18), date(2024, 4, 19), date(2024, 4, 22))
    for name in ("daily_directory", "dynamics_directory"):
        directory = Path(sources[name]) / "days"
        directory.mkdir()
        for day in days:
            (directory / f"{day}.npz").write_text(day.isoformat())
    configuration = root / "source.json"
    configuration.write_text(json.dumps(sources))
    calls, loaded, failures = [], [], {}
    monkeypatch.setattr(terminal_roll, "MarketStore", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(terminal_roll, "SourceArchive", lambda **kwargs: SimpleNamespace())

    def replay(**kwargs):
        origin, directory = kwargs["origin"], kwargs["output_directory"]
        previous = kwargs["previous_origin_decision"]
        calls.append((origin, kwargs["objective"], kwargs["risk_aversion"], kwargs["target_side"], previous))
        if previous is not None:
            assert previous.state.day < origin
            assert previous.elapsed_day == 0
            assert previous.target_key == (kwargs["claim_label"], kwargs["objective"], kwargs["risk_aversion"], kwargs["target_side"])
        directory.mkdir(parents=True, exist_ok=True)
        record = {"date": origin.isoformat(), "target": [kwargs["claim_label"], kwargs["objective"],
            kwargs["risk_aversion"], kwargs["target_side"]]}
        (directory / "episode.json").write_text(json.dumps({"origin_decision": record}))
        if origin in failures:
            problem = failures[origin]
            if isinstance(problem, Exception):
                raise problem
            return SimpleNamespace(status=problem, summary={"pending_fit_status": "training_limit"})
        return SimpleNamespace(status="complete", summary={"completed_steps": 3, "diagnostic_only": True,
            "costed_pnl": float(origin.day), "entry_cost": .001},
            origin_decision=SimpleNamespace(state=SimpleNamespace(day=origin + timedelta(days=5))))

    def load(directory, **kwargs):
        record = json.loads((directory / "episode.json").read_text())["origin_decision"]
        loaded.append((directory, record["date"]))
        return SimpleNamespace(state=SimpleNamespace(day=date.fromisoformat(record["date"])),
            elapsed_day=0, target_key=tuple(record["target"]))

    monkeypatch.setattr(terminal_roll, "replay_terminal_episode", replay)
    monkeypatch.setattr(terminal_roll, "load_origin_decision", load)
    monkeypatch.setattr(terminal_roll, "audit_terminal_replay", lambda **kwargs: {"status": "passed"})
    arguments = dict(source_configuration=configuration, output_directory=root / "output",
        start_date=days[0], end_date=days[-1], claims=(agreed_terminal_claims()[0],),
        objectives=("mean_variance", "power_exponential"), risk_aversions=(1., 10.), sides=(-1, 1),
        env=BaseEnvironment(device="cpu", precision=64), diagnostic_only=True,
        train_config=TerminalTrainConfig(maximum_updates=2, minimum_updates=1),
        simulation_config=TerminalSimulationConfig(paths=8))
    return arguments, days, calls, loaded, failures


def test_each_tensor_cell_warm_starts_from_earlier_inception_not_later_episode_weights(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, _ = rolling_inputs(monkeypatch, tmp_path)
    result = terminal_roll.run_terminal_roll(**arguments)
    assert result["completed_episodes"] == 24
    assert result["processed_cells"] == result["total_cells"] == 24
    assert [record[0] for record in calls] == [day for day in days for _ in range(8)]
    assert all(record[-1] is None for record in calls[:8])
    assert all(record[-1].state.day == days[0] for record in calls[8:16])
    assert all(record[-1].state.day == days[1] for record in calls[16:])
    assert (arguments["output_directory"] / "implementation.zip").is_file()
    terminal_roll.run_terminal_roll(**arguments)
    assert len(calls) == 48


def test_missing_quote_episode_keeps_its_causal_inception_weights_for_next_origin(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, failures = rolling_inputs(monkeypatch, tmp_path)
    failures[days[0]] = ValueError("Missing observed mids on next date: contract")
    result = terminal_roll.run_terminal_roll(**arguments)
    assert result["cell_status_counts"] == {"complete": 16, "market_data_unavailable": 8}
    assert all(record[-1].state.day == days[0] for record in calls[8:16])
    saved = pd.read_parquet(arguments["output_directory"] / "population.parquet")
    assert saved.exact_error.dropna().eq("Missing observed mids on next date: contract").all()


def test_explicit_policy_parameterization_is_forwarded_and_cannot_change_on_resume(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, _, _ = rolling_inputs(monkeypatch, tmp_path)
    model = TerminalPolicyConfig(label="fee-roll", feature_count=12, fee_preconditioning=True)
    original, forwarded = terminal_roll.replay_terminal_episode, []

    def replay(**kwargs):
        forwarded.append(kwargs["model_config"])
        return original(**kwargs)

    monkeypatch.setattr(terminal_roll, "replay_terminal_episode", replay)
    terminal_roll.run_terminal_roll(**arguments, model_config=model)
    assert len(forwarded) == 24 and all(config is model for config in forwarded)
    saved = json.loads((arguments["output_directory"] / "config.json").read_text())
    assert saved["policy_config"]["fee_preconditioning"]
    assert saved["policy_config_version"] == TerminalPolicyConfig.version.unique_id48
    with pytest.raises(ValueError, match="Rolling scope"):
        terminal_roll.run_terminal_roll(**arguments, model_config=None)


def test_incomplete_training_stops_without_silently_running_later_origins(monkeypatch, tmp_path: Path) -> None:
    arguments, _, calls, _, failures = rolling_inputs(monkeypatch, tmp_path)
    failures[arguments["start_date"]] = "training_incomplete"
    result = terminal_roll.run_terminal_roll(**arguments)
    assert result["status"] == "training_incomplete"
    assert result["completed_episodes"] == 0
    assert len(calls) == 1
    assert result["processed_cells"] == 0


def test_independent_chains_continue_but_never_bypass_an_unfinished_origin(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, loaded, _ = rolling_inputs(monkeypatch, tmp_path)
    original = terminal_roll.replay_terminal_episode

    def replay(**kwargs):
        result = original(**kwargs)
        if kwargs["origin"] == days[0] and kwargs["objective"] == "mean_variance":
            return SimpleNamespace(status="training_incomplete", summary={"pending_fit_status": "training_limit"})
        return result

    monkeypatch.setattr(terminal_roll, "replay_terminal_episode", replay)
    result = terminal_roll.run_terminal_roll(**arguments, continue_independent_chains=True)
    assert result["status"] == "training_incomplete"
    assert result["completed_episodes"] == result["processed_cells"] == 12
    assert result["deferred_training_chains"] == 4
    assert result["deferred_later_cells"] == 8
    assert result["total_cells"] == 24
    assert len(calls) == 16
    assert all(record[1] == "power_exponential" for record in calls if record[0] > days[0])
    assert len(loaded) == 12
    assert all(record[-1].state.day == days[0] for record in calls[8:12])
    assert all(record[-1].state.day == days[1] for record in calls[12:])
    deferred = pd.read_parquet(arguments["output_directory"] / "deferred_cells.parquet")
    assert deferred.awaiting_origin.eq(days[0].isoformat()).all()
    assert set(deferred.origin) == {day.isoformat() for day in days[1:]}
    with pytest.raises(ValueError, match="Rolling scope"):
        terminal_roll.run_terminal_roll(**arguments)
    monkeypatch.setattr(terminal_roll, "replay_terminal_episode", original)
    resumed = terminal_roll.run_terminal_roll(**arguments, continue_independent_chains=True)
    assert resumed["status"] == "complete"
    assert resumed["completed_episodes"] == 24
    assert resumed["deferred_training_chains"] == resumed["deferred_later_cells"] == 0
    assert pd.read_parquet(arguments["output_directory"] / "deferred_cells.parquet").empty


def test_independent_mode_still_stops_on_numerical_error(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, failures = rolling_inputs(monkeypatch, tmp_path)
    failures[days[0]] = RuntimeError("Original numerical warning and failure")
    with pytest.raises(RuntimeError, match="Original numerical warning and failure"):
        terminal_roll.run_terminal_roll(**arguments, continue_independent_chains=True)
    assert len(calls) == 1
    saved = json.loads((arguments["output_directory"] / "summary.json").read_text())
    assert saved["status"] == "review_required"
    assert saved["deferred_training_chains"] == 0


def test_independent_mode_respects_a_checkpointed_stop(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, failures = rolling_inputs(monkeypatch, tmp_path)
    failures[days[0]] = "checkpointed_stop"
    result = terminal_roll.run_terminal_roll(**arguments, continue_independent_chains=True)
    assert result["status"] == "checkpointed_stop"
    assert result["deferred_training_chains"] == 0
    assert result["completed_episodes"] == result["processed_cells"] == 0
    assert len(calls) == 1


def test_deferred_chains_are_not_counted_as_processed_on_later_missing_model_dates(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, _ = rolling_inputs(monkeypatch, tmp_path)
    original = terminal_roll.replay_terminal_episode

    def replay(**kwargs):
        result = original(**kwargs)
        if kwargs["origin"] == days[0] and kwargs["objective"] == "mean_variance":
            return SimpleNamespace(status="training_incomplete", summary={"pending_fit_status": "training_limit"})
        return result

    monkeypatch.setattr(terminal_roll, "replay_terminal_episode", replay)
    (tmp_path / "dynamics_directory" / "days" / f"{days[1]}.npz").unlink()
    result = terminal_roll.run_terminal_roll(**arguments, continue_independent_chains=True)
    assert result["status"] == "training_incomplete"
    assert result["completed_episodes"] == 8
    assert result["processed_cells"] == 12
    assert result["unavailable_date_cells"] == 4
    assert result["deferred_later_cells"] == 8
    assert result["remaining_seconds_estimate"] is None
    assert len(calls) == 12
    assert all(record[-1].state.day == days[0] for record in calls[8:])
    deferred = pd.read_parquet(arguments["output_directory"] / "deferred_cells.parquet")
    assert len(deferred.loc[deferred.origin == days[1].isoformat()]) == 4
    excluded = pd.read_parquet(arguments["output_directory"] / "date_exclusions.parquet")
    assert excluded.unavailable_rule_cells.tolist() == [4]


def test_numerical_or_missing_checkpoint_errors_stop_and_preserve_exact_error(monkeypatch, tmp_path: Path) -> None:
    arguments, _, calls, _, failures = rolling_inputs(monkeypatch, tmp_path)
    failures[arguments["start_date"]] = FileNotFoundError(errno.ENOENT, "missing", str(tmp_path / "policy.pt"))
    with pytest.raises(FileNotFoundError):
        terminal_roll.run_terminal_roll(**arguments)
    assert len(calls) == 1
    summary = json.loads((arguments["output_directory"] / "summary.json").read_text())
    assert summary["status"] == "review_required"
    assert summary["cell_status_counts"] == {"review_required": 1}


def test_requested_dates_without_models_are_explicit_not_silently_shortened(monkeypatch, tmp_path: Path) -> None:
    arguments, days, calls, _, _ = rolling_inputs(monkeypatch, tmp_path)
    (tmp_path / "dynamics_directory" / "days" / f"{days[1]}.npz").unlink()
    result = terminal_roll.run_terminal_roll(**arguments)
    assert result["unavailable_origin_dates"] == 1
    assert result["unavailable_date_cells"] == 8
    assert result["completed_episodes"] == 16
    assert not any(record[0] == days[1] for record in calls)
    assert all(record[-1].state.day == days[0] for record in calls[8:])
    missing = pd.read_parquet(arguments["output_directory"] / "date_exclusions.parquet")
    assert missing.origin.tolist() == [days[1].isoformat()]


def test_changed_source_configuration_cannot_mix_completed_rolling_results(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, _, _ = rolling_inputs(monkeypatch, tmp_path)
    terminal_roll.run_terminal_roll(**arguments)
    arguments["source_configuration"].write_text(arguments["source_configuration"].read_text() + "\n")
    with pytest.raises(ValueError, match="scope, source or implementation changed"):
        terminal_roll.run_terminal_roll(**arguments)
