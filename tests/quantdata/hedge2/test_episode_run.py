"""Episode sharding covers every eligible family without duplication or scope loss."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock, Timeout

from quantdata.hedge2.episode_run import (
    black_coverage_unavailable,
    eligible_episodes,
    episode_rules,
    run_episodes,
)


def test_episode_shards_cover_all_families_and_dates() -> None:
    coverage = pd.DataFrame({"origin": ["2023-01-09"] * 3, "group": ["digital-3D", "digital-5D", "exotic-3D-5D"],
                             "eligible": [True, False, True]})
    arguments = {"coverage": coverage, "families": ("digital-3D", "digital-5D", "ko", "ratio_straddle")}
    first = eligible_episodes(**arguments, shard_index=0, shard_count=2)
    second = eligible_episodes(**arguments, shard_index=1, shard_count=2)
    assert set(first).isdisjoint(second)
    assert set(first + second) == {(date(2023, 1, 9), family) for family in ("digital-3D", "ko", "ratio_straddle")}


def test_one_day_coverage_is_not_replaced_by_a_co_terminal_or_multiday_population() -> None:
    coverage = pd.DataFrame({"origin": ["2023-01-06", "2023-01-06", "2023-01-09"],
        "group": ["digital-1D", "digital-3D", "digital-1D"], "eligible": [True, False, True]})
    selected = eligible_episodes(coverage, families=("digital-1D",), shard_index=0, shard_count=1)
    assert selected == ((date(2023, 1, 6), "digital-1D"), (date(2023, 1, 9), "digital-1D"))


def test_full_rules_keep_sides_and_infinite_variance_without_infinite_power() -> None:
    rules = episode_rules(methods=("parametric", "statistical"), risks=(1., 10., 100., 1000., np.inf), greek_budgets=(1.,))
    assert len(rules) == 36
    assert sum(rule.objective == "mean_variance" and np.isinf(rule.risk_aversion) for rule in rules) == 4
    assert all(np.isfinite(rule.risk_aversion) for rule in rules if rule.objective == "power_exponential")


def test_output_lock_precedes_any_initialization_write(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    arguments = {name: tmp_path for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
        "market_directory", "daily_directory", "dynamics_directory", "coverage_path")}
    with FileLock(str(output / "run.lock"), timeout=0):
        with pytest.raises(Timeout):
            run_episodes(**arguments, output_directory=output, families=("ko",), methods=("unhedged",),
                         risks=(1.,), greek_budgets=(1.,))
    assert not (output / "config.json").exists()
    assert not (output / "implementation.zip").exists()


def test_only_prescribed_black_geometry_errors_are_coverage_exclusions() -> None:
    assert black_coverage_unavailable(ValueError("Black exotic hedge needs observed strikes surrounding 4050 at 1D"))
    assert black_coverage_unavailable(ValueError("The Black vega hedge needs two surrounding observed option strikes"))
    assert not black_coverage_unavailable(ValueError("Non-finite Black price"))
    assert not black_coverage_unavailable(ValueError("Position optimization failed"))


def test_current_reference_settings_are_forwarded_under_the_output_lock(monkeypatch, tmp_path) -> None:
    captured = {}

    def locked_runner(**kwargs):
        captured.update(kwargs)
        assert kwargs["run_lock"].is_locked
        return {"status": "test"}

    monkeypatch.setattr("quantdata.hedge2.episode_run._run_episodes_locked", locked_runner)
    arguments = {name: tmp_path for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
        "market_directory", "daily_directory", "dynamics_directory", "coverage_path")}
    result = run_episodes(**arguments, output_directory=tmp_path / "run", families=("ko",), methods=("black",),
        risks=(1.,), greek_budgets=(1.,), option_side="itm", include_co_terminal_forwards=True,
        reference_premium_cap=10.)
    assert result == {"status": "test"}
    assert captured["option_side"] == "itm"
    assert captured["include_co_terminal_forwards"] is True
    assert captured["reference_premium_cap"] == 10.
    assert "resolved" not in captured


def test_current_optimizer_premium_limit_is_forwarded_without_changing_statistical_horizon(monkeypatch, tmp_path) -> None:
    captured = {}

    def locked_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "test"}

    monkeypatch.setattr("quantdata.hedge2.episode_run._run_episodes_locked", locked_runner)
    arguments = {name: tmp_path for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
        "market_directory", "daily_directory", "dynamics_directory", "coverage_path")}
    run_episodes(**arguments, output_directory=tmp_path / "run", families=("ko",), methods=("parametric",),
        risks=(10.,), greek_budgets=(1.,), option_side="itm", include_co_terminal_forwards=True,
        optimized_premium_cap=10.)
    assert captured["optimized_premium_cap"] == 10.
    with pytest.raises(ValueError, match="terminal-horizon"):
        run_episodes(**arguments, output_directory=tmp_path / "statistical", families=("ko",),
            methods=("statistical",), risks=(10.,), greek_budgets=(1.,), option_side="itm",
            include_co_terminal_forwards=True, optimized_premium_cap=10.)
