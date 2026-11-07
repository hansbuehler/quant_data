"""A running population must retain its full scope and independently checked outcomes."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2 import frontier_population
from quantdata.hedge2.episode_artifacts import publish_episode
from quantdata.hedge2.episodes import EpisodeResult
from quantdata.hedge2.frontier_metrics import PORTFOLIO_KEYS
from quantdata.hedge2.frontier_population import (
    external_output_directory,
    load_frontier_population,
    load_reference_population,
)


def _requested():
    return pd.DataFrame([{"variant": variant, "origin": day, "family": "digital-3D", "claim": "target",
        "side": -1, "greek_budget": budget} for variant in ("ordinary", "orthogonal")
        for day in ("2024-04-01", "2024-04-02") for budget in (.2, 1.)])


def _write_frontier(root, requested):
    row = requested.iloc[0]
    episode = root / row.origin / row.family
    episode.mkdir(parents=True)
    configuration = {"origin": row.origin, "family": row.family, "claim_labels": ["target"],
        "identity": {"dynamics_directory": "/test/dynamics", "optimized_premium_cap": 10.},
        "implementation_sha256": {"claims.py": "2" * 64},
        "rules": [{"method": "DLV Greek", "side": -1, "objective": "none", "risk_aversion": 0.,
                   "greek_budget": budget} for budget in (.2, 1.)]}
    (episode / "config.json").write_text(json.dumps(configuration))
    shared = {"/test/source": "1" * 64, "frozen_book": "3" * 64, "observed_marks_spreads_and_carry": "4" * 64}
    (episode / "shared_input_hashes.json").write_text(json.dumps(shared))
    completed_episode = episode / "batches/test" / row.origin / row.family
    (completed_episode / "steps").mkdir(parents=True)
    (completed_episode / "completed.json").write_text("{}")
    np.savez_compressed(completed_episode / "steps/episode_checkpoint.npz", identity_json=json.dumps({
        "origin": row.origin, "family": row.family, "input_hashes": shared, "initial_spot": 1.,
        "option_side": "itm", "include_co_terminal_forwards": True}))
    completed = requested.loc[requested.origin.eq(row.origin)].copy()
    completed["costed_pnl"], completed["entry_cost"], completed["initial_price"] = .1, .001, .5
    completed.to_parquet(episode / "results.parquet", index=False)
    pd.DataFrame(columns=[*PORTFOLIO_KEYS, "costed_pnl"]).to_parquet(episode / "unavailable.parquet", index=False)
    (episode / "summary.json").write_text(json.dumps({"status": "complete"}))
    return episode


def test_pending_rows_preserve_full_requested_scope_and_audit_every_closed_frontier(tmp_path, monkeypatch) -> None:
    requested = _requested()
    ordinary = tmp_path / "ordinary"
    subset = requested.loc[requested.variant.eq("ordinary")]
    episode = _write_frontier(ordinary, subset)
    calls = []

    def audit(*, input_directory, output_directory):
        calls.append((input_directory, output_directory))
        return {"test_only_auditor": True}

    monkeypatch.setattr(frontier_population, "audit_isolated_frontier", audit)
    result = load_frontier_population(requested=requested,
        sources={"ordinary": (ordinary,), "orthogonal": (tmp_path / "orthogonal",)},
        audit_directory=tmp_path / "audit")
    assert len(result.requested) == 8 and len(result.available) == 2 and len(result.pending) == 6
    assert result.unavailable.empty and result.pending.costed_pnl.isna().all()
    assert calls == [(episode, tmp_path / "audit/ordinary/2024-04-01/digital-3D")]
    assert result.episodes.pending_portfolios.sum() == 6
    digest = result.evidence[str(episode)]["artifacts_sha256"]["results.parquet"]
    assert digest == hashlib.sha256((episode / "results.parquet").read_bytes()).hexdigest()


def test_a_failed_independent_audit_stops_reporting(tmp_path, monkeypatch) -> None:
    requested = _requested().loc[lambda frame: frame.variant.eq("ordinary")]
    root = tmp_path / "source"
    _write_frontier(root, requested)

    def fail(**kwargs):
        raise RuntimeError("Raw certificate failed")

    monkeypatch.setattr(frontier_population, "audit_isolated_frontier", fail)
    with pytest.raises(RuntimeError, match="Raw certificate failed"):
        load_frontier_population(requested=requested, sources={"ordinary": (root,)}, audit_directory=tmp_path / "audit")


def test_duplicate_completed_copies_cannot_be_silently_selected(tmp_path) -> None:
    requested = _requested().loc[lambda frame: frame.variant.eq("ordinary")]
    for name in ("old", "new"):
        _write_frontier(tmp_path / name, requested)
    with pytest.raises(ValueError, match="Multiple completed"):
        load_frontier_population(requested=requested, sources={"ordinary": (tmp_path / "old", tmp_path / "new")},
            audit_directory=tmp_path / "audit")


@pytest.mark.parametrize("change", ["claim", "budget", "method", "origin"])
def test_completed_frontier_cannot_substitute_targets_rules_or_identity(tmp_path, change) -> None:
    requested = _requested().loc[lambda frame: frame.variant.eq("ordinary")]
    root = tmp_path / "source"
    episode = _write_frontier(root, requested)
    configuration = json.loads((episode / "config.json").read_text())
    if change == "claim":
        configuration["claim_labels"] = ["other-target"]
    elif change == "budget":
        configuration["rules"][0]["greek_budget"] = .4
    elif change == "method":
        configuration["rules"][0]["method"] = "DLV parametric"
    else:
        configuration["origin"] = "2024-04-03"
    (episode / "config.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError):
        load_frontier_population(requested=requested, sources={"ordinary": (root,)}, audit_directory=tmp_path / "audit")


@pytest.mark.parametrize("problem", ["omission", "duplicate", "date", "family", "variant", "unsafe_variant", "budget", "side"])
def test_invalid_requested_population_is_rejected_before_audit(tmp_path, problem) -> None:
    requested = _requested()
    if problem == "omission":
        requested = requested.drop(columns="claim")
    elif problem == "duplicate":
        requested = pd.concat([requested, requested.iloc[:1]], ignore_index=True)
    elif problem == "date":
        requested.loc[0, "origin"] = "../2024-04-01"
    elif problem == "family":
        requested.loc[0, "family"] = "../digital-3D"
    elif problem == "variant":
        requested.loc[0, "variant"] = None
    elif problem == "unsafe_variant":
        requested.loc[0, "variant"] = "../ordinary"
    elif problem == "budget":
        requested.loc[0, "greek_budget"] = np.inf
    else:
        requested.loc[0, "side"] = 0
    with pytest.raises(ValueError):
        load_frontier_population(requested=requested,
            sources={"ordinary": (tmp_path / "ordinary",), "orthogonal": (tmp_path / "orthogonal",)},
            audit_directory=tmp_path / "audit")


def test_generated_output_cannot_enter_a_git_worktree(tmp_path) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (repository / ".git/HEAD").write_text("ref: refs/heads/main")
    with pytest.raises(ValueError, match="outside Git"):
        external_output_directory(repository / "reports")
    external_output_directory(tmp_path / "external")


def _reference(root):
    episode = root / "2024-04-01/digital-3D"
    episode.mkdir(parents=True)
    frame = pd.DataFrame([{"origin": "2024-04-01", "claim": "target", "side": -1,
        "costed_pnl": .1, "entry_cost": 0., "initial_price": .5}])
    path = episode / "summary.parquet"
    (root / "config.json").write_text(json.dumps({"dynamics_directory": "/test/dynamics",
        "implementation_sha256": {"claims.py": "2" * 64}}))
    book = pd.DataFrame({"strike": [100.]})
    shared = {"/test/source": "1" * 64,
        "frozen_book": hashlib.sha256(pd.util.hash_pandas_object(book, index=True).to_numpy().tobytes()).hexdigest(),
        "observed_marks_spreads_and_carry": "4" * 64}
    (episode / "steps").mkdir()
    np.savez_compressed(episode / "steps/episode_checkpoint.npz", identity_json=json.dumps({
        "origin": "2024-04-01", "family": "digital-3D", "input_hashes": shared, "initial_spot": 1.,
        "option_side": "itm", "include_co_terminal_forwards": True}), next_step=3,
        **{f"positions_{step}": np.zeros((1, 2)) for step in range(3)},
        **{f"indices_{step}": np.array([0, 1]) for step in range(3)})
    result = EpisodeResult(summary=frame, trades=pd.DataFrame({"hedge_gain": [0.]}), initial_book=book,
        positions=tuple(np.zeros((1, 2)) for _ in range(3)),
        active_indices=tuple(np.array([0, 1]) for _ in range(3)), diagnostics=(), input_hashes=shared)
    publish_episode(directory=episode, result=result, origin="2024-04-01", family="digital-3D",
        configuration_sha256=hashlib.sha256((root / "config.json").read_bytes()).hexdigest())
    (root / "summary.json").write_text(json.dumps({"status": "complete", "episodes": 1, "cells": 1}))
    (root / "independent_cash_audit").mkdir()
    (root / "independent_cash_audit/summary.json").write_text(json.dumps({"completed_episodes": 1,
        "portfolios": 1, "maximum_cash_error": 0.}))
    return path


def test_reference_needs_complete_counts_cash_checks_and_unchanged_sealed_data(tmp_path) -> None:
    root = tmp_path / "reference"
    path = _reference(root)
    frame, evidence = load_reference_population(directory=root)
    assert len(frame) == 1 and frame.family.iloc[0] == "digital-3D"
    assert evidence["cash_audit"]["maximum_cash_error"] == 0.
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_reference_population(directory=root)


def test_report_expected_economic_identity_is_enforced(tmp_path) -> None:
    requested = _requested().loc[lambda frame: frame.variant.eq("ordinary")]
    root = tmp_path / "source"
    _write_frontier(root, requested)
    with pytest.raises(ValueError, match="economic/model identity"):
        load_frontier_population(requested=requested, sources={"ordinary": (root,)},
            audit_directory=tmp_path / "audit", expected_identities={"ordinary": {"optimized_premium_cap": 1000.}})


@pytest.mark.parametrize("error", [1., float("nan"), float("inf")])
def test_reference_cash_failure_or_nonfinite_error_is_not_accepted(tmp_path, error) -> None:
    root = tmp_path / "reference"
    _reference(root)
    path = root / "independent_cash_audit/summary.json"
    audit = json.loads(path.read_text())
    audit["maximum_cash_error"] = error
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="unaudited reference"):
        load_reference_population(directory=root)
