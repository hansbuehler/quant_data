"""Partial rolling portfolios are atomic, aligned and usable without pickle."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.certification import certify_candidate
from quantdata.hedge2.objective_run import (
    load_checkpoint,
    save_checkpoint,
    save_verified_array_inputs,
    save_verified_book,
    save_verified_metadata,
)
from quantdata.hedge2.optimization import ScenarioOptimizer
from quantdata.hedge2.resume_objectives import migrate_checkpoints


def test_atomic_checkpoint_retains_infinity_and_row_portfolio_alignment(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    rows = [{"claim": "d", "risk_aversion": np.inf, "optimization_certified": True}]
    positions = [np.array([1., 2.])]
    assert load_checkpoint(path) == ([], [])
    save_checkpoint(path, rows, positions)
    saved_rows, saved_positions = load_checkpoint(path)
    assert saved_rows == rows
    np.testing.assert_array_equal(saved_positions, positions)
    assert not path.with_suffix(".partial.npz").exists()


def test_statistical_acceptance_checks_economic_gap_not_warning_text() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=np.full(4, .25), costs=np.array([.0001]))
    for objective in ("mean_variance", "power_exponential"):
        candidate = optimizer.solve(claim_gains=2. * gains[:, 0], objective=objective, risk_aversion=10.)
        hedge, certificate = certify_candidate(optimizer=optimizer, candidate=candidate,
            claim_gains=2. * gains[:, 0], objective=objective, risk_aversion=10., target_side=-1)
        assert certificate["optimization_certified"]
        assert hedge.warnings == candidate.warnings


def test_numerical_resume_preserves_completed_cells_with_their_original_identity(tmp_path) -> None:
    parent = tmp_path / "parent"
    directory = parent / "2023-03-10"
    directory.mkdir(parents=True)
    (parent / "config.json").write_text(json.dumps({"method": "parametric"}))
    rows = [{"claim": "d", "risk_aversion": 10., "optimization_certified": True,
             "warnings": ["Solution may be inaccurate."]}]
    positions = [np.array([1., 2.])]
    save_checkpoint(directory / "checkpoint.npz", rows, positions)
    output = tmp_path / "child"
    record = migrate_checkpoints(input_directory=parent, output_directory=output)
    saved_rows, saved_positions = load_checkpoint(output / "2023-03-10/checkpoint.npz")
    assert record["inherited_certified_cells"] == 1
    assert saved_rows[0]["warnings"] == rows[0]["warnings"]
    assert saved_rows[0]["inherited_implementation_run"] == str(parent)
    np.testing.assert_array_equal(saved_positions, positions)
    assert load_checkpoint(directory / "checkpoint.npz")[0] == rows


def test_resumed_inputs_must_match_saved_arrays_book_and_metadata(tmp_path) -> None:
    arrays = {"weights": np.array([.25, .75]), "source": "saved0.1"}
    path = tmp_path / "inputs.npz"
    save_verified_array_inputs(path, arrays)
    save_verified_array_inputs(path, arrays)
    with pytest.raises(ValueError, match="risk inputs changed"):
        save_verified_array_inputs(path, dict(arrays, weights=np.array([.3, .7])))
    book = pd.DataFrame({"strike": [100.], "bdte": [1.]})
    save_verified_book(tmp_path / "book.parquet", book)
    save_verified_book(tmp_path / "book.parquet", book)
    with pytest.raises(ValueError, match="hedge book changed"):
        save_verified_book(tmp_path / "book.parquet", book.assign(strike=101.))
    metadata = {"date": "2024-04-19", "source_hash": "first"}
    save_verified_metadata(tmp_path / "metadata.json", metadata)
    save_verified_metadata(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="metadata changed"):
        save_verified_metadata(tmp_path / "metadata.json", dict(metadata, source_hash="different"))


@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
def test_premium_dollar_units_preserve_hedge_cash_objective_when_limits_do_not_bind(objective) -> None:
    gains = np.array([[-.03, -.02], [-.01, .01], [.01, -.01], [.03, .02]])
    prices = np.array([1., .5])
    target = gains @ np.array([.3, -.4])
    costs = np.array([0., .0001])
    original = ScenarioOptimizer(instrument_gains=gains, weights=np.full(4, .25), costs=costs,
        position_cap=1000.).solve(claim_gains=target, objective=objective, risk_aversion=100., target_side=-1)
    premium = ScenarioOptimizer(instrument_gains=gains / prices[None, :], weights=np.full(4, .25),
        costs=costs / prices, position_cap=10.).solve(claim_gains=target, objective=objective,
        risk_aversion=100., target_side=-1)
    np.testing.assert_allclose(gains @ original.positions, gains @ (premium.positions / prices), atol=2e-5)
    assert premium.training_objective == pytest.approx(original.training_objective, abs=1e-7)
    assert premium.entry_cost == pytest.approx(original.entry_cost, abs=1e-7)


def test_reviewed_power_migration_retains_historical_premium_cells_and_original_units(tmp_path) -> None:
    parent = tmp_path / "parent"
    directory = parent / "2024-04-19"
    directory.mkdir(parents=True)
    configuration = {"method": "historical", "itm_premium_book": True, "position_cap": 10.}
    (parent / "config.json").write_text(json.dumps(configuration))
    rows = [{"claim": "d", "risk_aversion": 100., "optimization_certified": True}]
    save_checkpoint(directory / "checkpoint.npz", rows, [np.array([1., 2.])])
    record = migrate_checkpoints(input_directory=parent, output_directory=tmp_path / "child",
        numerical_fix="cash_unit_power_cone")
    assert record["inherited_certified_cells"] == 1
    assert record["numerical_fix"] == "cash_unit_power_cone"
    inherited_rows, inherited_positions = load_checkpoint(tmp_path / "child/2024-04-19/checkpoint.npz")
    np.testing.assert_array_equal(inherited_positions, [[1., 2.]])
    assert inherited_rows[0]["inherited_implementation_run"] == str(parent)
    assert json.loads((parent / "config.json").read_text()) == configuration


def test_completed_migration_rebinds_changed_checkpoint_hash_without_changing_parent(tmp_path) -> None:
    parent = tmp_path / "parent"
    directory = parent / "2024-04-19"
    directory.mkdir(parents=True)
    (parent / "config.json").write_text(json.dumps({"method": "historical"}))
    save_checkpoint(directory / "checkpoint.npz", [{"optimization_certified": True}], [np.array([1., 2.])])
    digest = hashlib.sha256((directory / "checkpoint.npz").read_bytes()).hexdigest()
    manifest = {"certified_cells": 1, "artifacts_sha256": {"checkpoint.npz": digest}}
    (directory / "completed.json").write_text(json.dumps(manifest))
    output = tmp_path / "child"
    migrate_checkpoints(input_directory=parent, output_directory=output, numerical_fix="cash_unit_power_cone")
    inherited = json.loads((output / "2024-04-19/completed.json").read_text())
    assert inherited["artifacts_sha256"]["checkpoint.npz"] == hashlib.sha256((output / "2024-04-19/checkpoint.npz").read_bytes()).hexdigest()
    assert inherited["inherited_completed_manifest_sha256"] == hashlib.sha256((directory / "completed.json").read_bytes()).hexdigest()
    assert inherited["artifacts_sha256"]["checkpoint.npz"] != digest
    assert json.loads((directory / "completed.json").read_text()) == manifest
