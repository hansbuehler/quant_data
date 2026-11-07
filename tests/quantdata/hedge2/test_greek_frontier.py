"""Unavailable portfolios cannot stop, replace or contaminate independent chains."""

from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.episodes import EpisodeResult, HedgeRule
from quantdata.hedge2.greek_frontier import partition_failed_portfolio, run_isolated_greek_frontier
from quantdata.hedge2.premium import GreekBudgetInfeasible


def _result(labels, rules) -> EpisodeResult:
    rows = [{"claim": label, **asdict(rule), "costed_pnl": float({"bad": 0, "good": 1, "target": 2}[label] + rule.side),
        "origin": "2024-04-19"} for label in labels for rule in rules]
    return EpisodeResult(summary=pd.DataFrame(rows), trades=pd.DataFrame({"prefix_only": [True]}),
        initial_book=pd.DataFrame({"strike": [100.]}), positions=(np.ones((len(rows), 2)),),
        active_indices=(np.array([0, 1]),), diagnostics=(), input_hashes={"market": "a" * 64})


def test_partition_is_disjoint_and_covers_every_other_portfolio() -> None:
    rules = tuple(HedgeRule(method="DLV Greek", side=side, greek_budget=budget)
                  for side in (-1, 1) for budget in (1., .6, .2))
    labels = ("first", "second", "third")
    parts = partition_failed_portfolio(claim_labels=labels, rules=rules,
        failed_claim="second", failed_rule=rules[2])
    flattened = [(label, rule) for chosen_labels, chosen_rules in parts for label in chosen_labels for rule in chosen_rules]
    expected = {(label, rule) for label in labels for rule in rules} - {("second", rules[2])}
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == expected


@pytest.mark.parametrize("side", [-1, 1])
def test_unavailable_chain_is_not_scored_or_liquidated_and_resume_reuses_evidence(tmp_path, side) -> None:
    rules = (HedgeRule(method="DLV Greek", side=side, greek_budget=.2),
             HedgeRule(method="DLV Greek", side=side, greek_budget=1.))
    calls = []

    def runner(labels, selected, checkpoint):
        calls.append((labels, selected))
        if "bad" in labels and rules[0] in selected:
            checkpoint.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(checkpoint / "last_feasible_inventory.npz", holdings=np.array([2., -3.]))
            error = GreekBudgetInfeasible(full_entry_cost=2., mandatory_turnover_cost=1., budget_fraction=.2)
            error.portfolio_context = {"origin": "2024-04-19", "family": "digital-3D", "date": "2024-04-22",
                "claim": "bad", **asdict(rules[0]), "executed_at_failure": False,
                "input_hashes": {"market": "a" * 64},
                "error_type": type(error).__name__, "exact_error": str(error)}
            raise error
        return _result(labels, selected)

    arguments = dict(origin=date(2024, 4, 19), family="digital-3D", claim_labels=("bad", "good"),
        rules=rules, identity={"optimized_premium_cap": 10.}, run_batch=runner, output_directory=tmp_path / "run")
    result = run_isolated_greek_frontier(**arguments)
    assert result.summary["status"] == "processing_complete_with_unavailable"
    assert result.summary["requested_portfolios"] == 4
    assert len(result.results) == 3 and len(result.unavailable) == 1
    assert result.unavailable.costed_pnl.isna().all()
    assert not result.summary["substitute_trades"]
    assert not result.summary["complete_feasible_frontier"]
    for row in result.results.itertuples():
        single_rule = HedgeRule(method=row.method, side=row.side, objective=row.objective,
            risk_aversion=row.risk_aversion, greek_budget=row.greek_budget)
        standalone = _result((row.claim,), (single_rule,))
        assert row.costed_pnl == standalone.summary.costed_pnl.iloc[0]
    prefix = next((tmp_path / "run").rglob("last_feasible_inventory.npz"))
    with np.load(prefix) as saved:
        np.testing.assert_array_equal(saved["holdings"], [2., -3.])
    call_count = len(calls)
    restored = run_isolated_greek_frontier(**arguments)
    assert len(calls) == call_count
    pd.testing.assert_frame_equal(restored.results, result.results)
    pd.testing.assert_frame_equal(restored.unavailable, result.unavailable)
    prefix.write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        run_isolated_greek_frontier(**arguments)


def test_generic_solver_error_still_stops_and_is_not_a_coverage_exclusion(tmp_path) -> None:
    def fail(*args):
        raise RuntimeError("original numerical error")

    with pytest.raises(RuntimeError, match="original numerical error"):
        run_isolated_greek_frontier(origin=date(2024, 4, 19), family="digital-3D", claim_labels=("target",),
            rules=(HedgeRule(method="DLV Greek", side=-1),), identity={}, run_batch=fail,
            output_directory=tmp_path / "run")
    assert not (tmp_path / "run" / "summary.json").exists()


def test_checkpointed_interruption_can_resume_without_an_unavailable_classification(tmp_path) -> None:
    calls = []

    def runner(labels, rules, checkpoint):
        checkpoint.mkdir(parents=True, exist_ok=True)
        marker = checkpoint / "saved.txt"
        calls.append(marker.exists())
        if not marker.exists():
            marker.write_text("last completed daily step")
            raise InterruptedError("checkpointed")
        return _result(labels, rules)

    arguments = dict(origin=date(2024, 4, 19), family="digital-3D", claim_labels=("target",),
        rules=(HedgeRule(method="DLV Greek", side=-1),), identity={}, run_batch=runner,
        output_directory=tmp_path / "run")
    with pytest.raises(InterruptedError, match="checkpointed"):
        run_isolated_greek_frontier(**arguments)
    result = run_isolated_greek_frontier(**arguments)
    assert calls == [False, True]
    assert result.summary["status"] == "complete"
    assert result.unavailable.empty


def test_missing_portfolio_context_is_not_silently_accepted(tmp_path) -> None:
    def fail(*args):
        raise GreekBudgetInfeasible(full_entry_cost=2., mandatory_turnover_cost=1., budget_fraction=.2)

    with pytest.raises(GreekBudgetInfeasible):
        run_isolated_greek_frontier(origin=date(2024, 4, 19), family="digital-3D", claim_labels=("target",),
            rules=(HedgeRule(method="DLV Greek", side=-1),), identity={}, run_batch=fail,
            output_directory=tmp_path / "run")


@pytest.mark.parametrize("all_unavailable", [False, True])
def test_multiple_first_step_failures_remain_disjoint_without_fabricated_pnl(tmp_path, all_unavailable) -> None:
    rules = tuple(HedgeRule(method="DLV Greek", side=side, greek_budget=budget)
        for side in (-1, 1) for budget in (.2, 1.))
    labels = ("bad", "good")
    failures = {(label, rule) for label in labels for rule in rules} if all_unavailable else {
        ("bad", rules[0]), ("bad", rules[2]), ("good", rules[1])}

    def runner(chosen_labels, chosen_rules, checkpoint):
        for label in chosen_labels:
            for rule in chosen_rules:
                if (label, rule) in failures:
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    error = GreekBudgetInfeasible(full_entry_cost=2., mandatory_turnover_cost=1., budget_fraction=.2)
                    error.portfolio_context = {"origin": "2024-04-19", "family": "digital-3D", "date": "2024-04-19",
                        "claim": label, **asdict(rule), "executed_at_failure": False,
                        "input_hashes": {"market": "a" * 64}, "error_type": type(error).__name__, "exact_error": str(error)}
                    raise error
        return _result(chosen_labels, chosen_rules)

    result = run_isolated_greek_frontier(origin=date(2024, 4, 19), family="digital-3D", claim_labels=labels,
        rules=rules, identity={}, run_batch=runner, output_directory=tmp_path / "run")
    assert len(result.unavailable) == len(failures)
    assert len(result.results) == len(labels) * len(rules) - len(failures)
    assert result.unavailable.costed_pnl.isna().all()
    assert list((tmp_path / "run").glob("batches/*/*/*/unavailable_context.json"))
    observed = {(row.claim, HedgeRule(method=row.method, side=row.side,
        greek_budget=row.greek_budget)) for row in result.unavailable.itertuples()}
    assert observed == failures


def test_input_change_between_independent_replays_is_rejected(tmp_path) -> None:
    from dataclasses import replace

    calls = []

    def runner(labels, rules, checkpoint):
        calls.append(True)
        result = _result(labels, rules)
        return replace(result, input_hashes={"market": "b" * 64})

    output = tmp_path / "run"
    output.mkdir()
    (output / "shared_input_hashes.json").write_text('{"market":"' + "a" * 64 + '"}')
    with pytest.raises(ValueError, match="different market/model"):
        run_isolated_greek_frontier(origin=date(2024, 4, 19), family="digital-3D", claim_labels=("target",),
            rules=(HedgeRule(method="DLV Greek", side=-1),), identity={}, run_batch=runner,
            output_directory=output)
