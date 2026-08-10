from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.gendysanos.candidate_replay import (
    summarize_frozen_replay,
)
from quantdata.dysanos.grid.gendysanos.case_store import DerivedCaseStore
from quantdata.dysanos.grid.gendysanos.conditional import (
    _fit_and_audit_local_baskets,
    run_time_separated_local_tail,
)
from quantdata.dysanos.grid.gendysanos.matched_cache import (
    MatchedUniverseBlockCache,
    migrate_legacy_option_settlement_cache,
)
from quantdata.dysanos.grid.gendysanos.required_runs import (
    REPORTED_COST_RATES,
    DerivedCaseArrays,
    OptionCoverageAccumulator,
    PathSplit,
    RequiredCase,
    TimeSeparatedUnconditionalAccumulator,
    audit_constant_basket,
    certify_unconditional_directions,
    convergence_report,
    derive_case_arrays,
    float64_replay_report,
    functional_gate_report,
    propagated_basket_band,
    required_cases,
    solve_constant_basket,
    stratified_path_split,
    time_separated_unconditional_report,
)
from quantdata.dysanos.grid.gendysanos.run_required_experiments import (
    _REPORT_SCHEMA_VERSION,
    _base_configuration,
    _coverage_gate,
    _dynamic_arbitrage_score,
    _load_completed_case_report,
)
from quantdata.dysanos.grid.gendysanos.static_spread_strategy import (
    StateDependentStaticSpreadAccumulator,
)
from quantdata.dysanos.implied.universe import PCAMatchedUniverseBlock


def _expiry_universe() -> dict[str, np.ndarray]:
    return {
        "dC": np.array([[[[2.0], [7.0]]]], dtype=np.float32),
        "dS": np.array([[1.0]], dtype=np.float32),
        "expiry_dS": np.array([[[3.0, 8.0]]], dtype=np.float32),
        "C": np.array([[[[5.0], [10.0]]]], dtype=np.float32),
        "spot": np.array([[100.0]], dtype=np.float32),
        "delta": np.array([[[[0.5], [0.25]]]], dtype=np.float32),
        "log_strikes": np.log(
            np.array([[[[0.9], [1.1]]]], dtype=np.float32)
        ),
        "option_expired": np.ones((1, 1, 2), dtype=bool),
        "expiries": np.array([[[0.1, 0.2]]], dtype=np.float32),
        "xstate": np.zeros((1, 1, 3), dtype=np.float32),
    }


def _incremental_universe() -> dict[str, np.ndarray]:
    universe = _expiry_universe()
    universe.pop("expiry_dS")
    universe["dS"] = np.array([[4.0]], dtype=np.float32)
    universe["option_dS"] = np.array([[[4.0, 4.0]]], dtype=np.float32)
    universe["option_settlement_dS"] = universe["option_dS"].copy()
    universe["option_expired"] = np.zeros((1, 1, 2), dtype=bool)
    return universe


def test_required_matrix_has_48_cases_and_fixed_gpu_mapping() -> None:
    cases = required_cases(100_000)

    assert len(cases) == 48
    assert len({tuple(sorted(case.__dict__.items())) for case in cases}) == 48
    assert {case.cost_rate for case in cases} == set(REPORTED_COST_RATES)
    assert {case.gpu for case in cases} == {0, 1, 2, 3}
    assert len({case.base_key for case in cases}) == 4
    assert {case.return_times for case in cases} == {9, 10}


def test_lower_transaction_cost_improves_frozen_basket_pnl() -> None:
    universe = _expiry_universe()
    low_cost = derive_case_arrays(universe, hedge="raw", cost_rate=1.0e-4)
    high_cost = derive_case_arrays(universe, hedge="raw", cost_rate=1.0e-3)
    action = np.full(low_cost.returns.shape[-1], 1.0 / low_cost.returns.shape[-1])

    low_pnl = (
        low_cost.returns @ action - low_cost.costs @ np.abs(action)
    )
    high_pnl = (
        high_cost.returns @ action - high_cost.costs @ np.abs(action)
    )
    low_band = propagated_basket_band(
        action,
        low_cost.returns,
        low_cost.costs,
        low_cost.numerical_band,
    )
    high_band = propagated_basket_band(
        action,
        high_cost.returns,
        high_cost.costs,
        high_cost.numerical_band,
    )

    assert np.all(low_pnl >= high_pnl)
    assert np.all(low_band <= high_band)


def test_base_configuration_targets_all_ten_action_cohorts() -> None:
    configuration = _base_configuration(
        generator="dysanos",
        frequency="weekly",
        paths=100_000,
        seed=7,
        block_size=10_000,
    )

    assert configuration["importance_sampling"]["target_action_count"] == 10
    assert configuration["return_horizons_derived"] == ["expiry", "incremental"]
    assert configuration["cost_rates_derived"] == [0.0, 1.0e-4, 1.0e-3]


def test_expiry_case_uses_one_matching_spot_instrument_per_expiry() -> None:
    case = derive_case_arrays(_expiry_universe(), hedge="raw", cost_rate=0.0)

    assert case.return_horizon == "expiry"
    assert case.returns.shape == (1, 1, 4)
    assert case.returns[0, 0] == pytest.approx([3.0, 8.0, 2.0, 7.0])
    assert case.instrument_kind == ("spot", "spot", "option", "option")
    assert case.expiry_index.tolist() == [0, 1, 0, 1]


def test_incremental_case_has_one_matching_spot_instrument_per_expiry() -> None:
    case = derive_case_arrays(
        _incremental_universe(), hedge="raw", cost_rate=0.0
    )

    assert case.return_horizon == "incremental"
    assert case.returns.shape == (1, 1, 4)
    assert case.returns[0, 0] == pytest.approx([4.0, 4.0, 2.0, 7.0])
    assert case.instrument_kind == ("spot", "spot", "option", "option")
    assert case.expiry_index.tolist() == [0, 1, 0, 1]


def test_incremental_delta_hedge_closes_at_early_expiry() -> None:
    universe = _incremental_universe()
    universe["option_expired"] = np.array([[[True, False]]])
    universe["option_settlement_dS"] = np.array(
        [[[1.0, 4.0]]], dtype=np.float32
    )
    first = derive_case_arrays(universe, hedge="delta", cost_rate=0.0)

    universe["option_settlement_dS"][0, 0, 0] = 40.0
    second = derive_case_arrays(universe, hedge="delta", cost_rate=0.0)

    assert first.returns[0, 0, :2] == pytest.approx([1.0, 4.0])
    assert second.returns[0, 0, :2] == pytest.approx([40.0, 4.0])
    assert first.returns[0, 0, 2] == pytest.approx(2.0 - 0.5 * 1.0)
    assert second.returns[0, 0, 2] == pytest.approx(2.0 - 0.5 * 40.0)
    assert first.returns[0, 0, 3] == second.returns[0, 0, 3]
    assert first.numerical_band[0, 0, 2] < second.numerical_band[0, 0, 2]
    assert first.numerical_band[0, 0, 3] == second.numerical_band[0, 0, 3]


def test_raw_and_delta_costs_are_derived_from_same_primitive_block() -> None:
    universe = _expiry_universe()
    raw = derive_case_arrays(universe, hedge="raw", cost_rate=1.0e-4)
    hedged = derive_case_arrays(universe, hedge="delta", cost_rate=1.0e-4)

    assert raw.returns[0, 0] == pytest.approx([3.0, 8.0, 2.0, 7.0])
    assert hedged.returns[0, 0] == pytest.approx(
        [3.0, 8.0, 2.0 - 0.5 * 3.0, 7.0 - 0.25 * 8.0]
    )
    assert raw.costs[0, 0] == pytest.approx(
        [0.01, 0.01, 0.0005, 0.001]
    )
    assert hedged.costs[0, 0] == pytest.approx(
        [0.01, 0.01, 0.0055, 0.0035]
    )
    assert np.shares_memory(universe["dC"], universe["dC"])


def test_numerical_band_uses_unreduced_scale_without_cash_floor() -> None:
    universe = _expiry_universe()
    universe["spot"] *= 1.0e-9
    universe["C"] *= 1.0e-9
    universe["dC"] *= 1.0e-9
    universe["expiry_dS"] *= 1.0e-9
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)

    assert np.max(case.numerical_band) < 1.0e-12
    assert np.all(case.numerical_band > 0.0)


def test_float64_replay_recovers_expiry_return_hedge_and_cost() -> None:
    universe = _expiry_universe()
    # Make dC internally consistent with the fixed-strike payoff identity.
    cash_strike = np.exp(universe["log_strikes"]) * universe["spot"][..., None, None]
    payoff = np.maximum(
        universe["spot"][..., None, None]
        + universe["expiry_dS"][..., None]
        - cash_strike,
        0.0,
    )
    universe["dC"] = payoff - universe["C"]
    case = derive_case_arrays(universe, hedge="delta", cost_rate=1.0e-4)

    report = float64_replay_report(universe, case)

    assert report["passed"]


def test_float64_replay_recovers_incremental_early_expiry_payoff() -> None:
    universe = _incremental_universe()
    universe["option_expired"] = np.array([[[True, False]]])
    universe["option_settlement_dS"] = np.array(
        [[[1.0, 4.0]]], dtype=np.float32
    )
    cash_strike = np.exp(universe["log_strikes"]) * universe["spot"][
        ..., None, None
    ]
    payoff = np.maximum(
        universe["spot"][..., None, None]
        + universe["option_settlement_dS"][..., None]
        - cash_strike,
        0.0,
    )
    universe["dC"][:, :, 0] = payoff[:, :, 0] - universe["C"][:, :, 0]
    case = derive_case_arrays(universe, hedge="delta", cost_rate=1.0e-4)

    report = float64_replay_report(universe, case)

    assert report["passed"]


def test_stratified_split_is_exact_disjoint_and_reproducible() -> None:
    components = np.repeat(np.arange(4), [50, 25, 15, 10])
    first = stratified_path_split(components, seed=17)
    second = stratified_path_split(components, seed=17)

    assert first.reference.size == 80
    assert first.audit.size == 20
    assert np.intersect1d(first.reference, first.audit).size == 0
    assert np.array_equal(first.reference, second.reference)
    assert np.array_equal(first.audit, second.audit)
    assert [first.component_counts[index]["audit"] for index in range(4)] == [
        10,
        5,
        3,
        2,
    ]


def test_unconditional_report_keeps_trading_times_separate() -> None:
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, 4, axis=0)
    universe["dC"] = np.repeat(universe["dC"], 2, axis=1)
    universe["dS"] = np.repeat(universe["dS"], 2, axis=1)
    universe["option_dS"] = np.repeat(universe["option_dS"], 2, axis=1)
    universe["option_settlement_dS"] = np.repeat(
        universe["option_settlement_dS"], 2, axis=1
    )
    universe["C"] = np.repeat(universe["C"], 2, axis=1)
    universe["spot"] = np.repeat(universe["spot"], 2, axis=1)
    universe["delta"] = np.repeat(universe["delta"], 2, axis=1)
    universe["log_strikes"] = np.repeat(universe["log_strikes"], 2, axis=1)
    universe["xstate"] = np.repeat(universe["xstate"], 2, axis=1)
    universe["dC"][:, 1] *= -1.0
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)

    report = time_separated_unconditional_report(
        case,
        importance_components=np.array([0, 0, 1, 1]),
    )

    assert report["trading_times"] == 2
    assert len(report["times"]) == 2
    first = report["times"][0]["directions"]["long"][2]
    second = report["times"][1]["directions"]["long"][2]
    assert first["proposal_gain_count"] == 4
    assert second["proposal_loss_count"] == 4


def test_propagated_basket_band_and_constraint_generation() -> None:
    S = 20
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, S, axis=0)
    # Make the first option a genuine nonnegative reference direction.
    universe["dC"][:, 0, 0, 0] = np.linspace(0.0, 1.0, S)
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    action = np.array([0.0, 0.0, 1.0, 0.0])
    band = propagated_basket_band(
        action,
        case.returns[:, 0],
        case.costs[:, 0],
        case.numerical_band[:, 0],
    )
    solution = solve_constant_basket(
        case,
        time_index=0,
        reference_paths=np.arange(S),
    )

    assert np.all(band >= case.numerical_band[:, 0, 2])
    assert solution["l1_norm"] <= 1.0 + 1.0e-9
    assert solution["minimum_pnl_minus_negative_band"] >= -1.0e-9
    assert solution["objective_mean"] > 0.0


def test_zero_cost_constant_basket_allows_short_actions() -> None:
    paths = 20
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, paths, axis=0)
    # Only the short direction of the first option is nonnegative.
    universe["dS"][:] = 0.0
    universe["option_dS"][:] = 0.0
    universe["option_settlement_dS"][:] = 0.0
    universe["dC"][:] = 0.0
    universe["dC"][:, 0, 0, 0] = -np.linspace(0.1, 1.0, paths)
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)

    solution = solve_constant_basket(
        case,
        time_index=0,
        reference_paths=np.arange(paths),
    )

    assert solution["action"][2] < 0.0
    assert solution["l1_norm"] <= 1.0 + 1.0e-9
    assert solution["minimum_economic_pnl"] >= -1.0e-9
    assert solution["objective_mean"] > 0.0


def test_derived_case_rejects_nonfinite_values() -> None:
    universe = _expiry_universe()
    universe["dC"][0, 0, 0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        derive_case_arrays(universe, hedge="raw", cost_rate=0.0)


def test_convergence_report_checks_rates_not_raw_counts() -> None:
    small = {
        "paths": 100_000,
        "weight_mean": 1.001,
        "weight_mean_standard_error": 0.001,
        "effective_sample_size": 50_000,
        "mean_standard_error": 0.01,
        "conditional_radius": 1.0,
        "conditional_effective_sample_size": 800,
    }
    large = {
        "paths": 1_000_000,
        "weight_mean": 1.0001,
        "weight_mean_standard_error": 0.0001,
        "effective_sample_size": 500_000,
        "mean_standard_error": 0.0032,
        "conditional_radius": 0.7,
        "conditional_effective_sample_size": 3_000,
    }

    report = convergence_report(small, large)

    assert report["passed"]
    assert report["expected_standard_error_reduction"] == pytest.approx(
        np.sqrt(10.0)
    )


def test_matched_cache_stores_common_path_primitives_once(tmp_path) -> None:
    expiry = _expiry_universe()
    for name, value in list(expiry.items()):
        if name not in {"expiries", "option_expired"}:
            expiry[name] = np.repeat(value, 10, axis=1)
    incremental = {
        name: value[:, :9]
        for name, value in expiry.items()
        if name not in {"expiries", "option_expired", "expiry_dS", "dS"}
    }
    incremental["expiries"] = expiry["expiries"]
    incremental["option_expired"] = np.zeros((1, 1, 2), dtype=bool)
    incremental["dC"] = expiry["dC"][:, :9].copy()
    incremental["dS"] = np.ones((1, 9), dtype=np.float32)
    incremental["option_dS"] = np.ones((1, 9, 2), dtype=np.float32)
    incremental["option_settlement_dS"] = incremental["option_dS"].copy()
    with pytest.raises(ValueError, match="at least two"):
        MatchedUniverseBlockCache(
            tmp_path,
            cache_uid="test-matched",
            configuration={"generator": "pca-iv"},
            num_samples=1,
        )

    second_expiry = {
        name: (
            value if name in {"expiries", "option_expired"} else np.repeat(value, 2, axis=0)
        )
        for name, value in expiry.items()
    }
    second_incremental = {
        name: (
            value if name in {"expiries", "option_expired"} else np.repeat(value, 2, axis=0)
        )
        for name, value in incremental.items()
    }
    cache = MatchedUniverseBlockCache(
        tmp_path,
        cache_uid="test-matched",
        configuration={"generator": "pca-iv"},
        num_samples=2,
    )
    cache.begin()
    cache.write(
        PCAMatchedUniverseBlock(
            start=0,
            stop=2,
            expiry=second_expiry,
            incremental=second_incremental,
        )
    )
    cache.complete(diagnostics={"status": "tested"})

    restored = next(cache.iter_blocks())
    assert np.array_equal(restored.incremental["C"], restored.expiry["C"][:, :9])
    assert cache.manifest()["storage_semantics"].startswith("one coherent")


def test_migrate_cache_closes_option_hedge_at_settlement_endpoint(
    tmp_path,
) -> None:
    expiry = _expiry_universe()
    for name, value in list(expiry.items()):
        if name not in {"expiries", "option_expired"}:
            expiry[name] = np.repeat(np.repeat(value, 2, axis=0), 10, axis=1)
    incremental = {
        name: value[:, :9]
        for name, value in expiry.items()
        if name not in {"expiries", "option_expired", "expiry_dS", "dS"}
    }
    incremental["expiries"] = expiry["expiries"]
    incremental["option_expired"] = np.zeros((1, 1, 2), dtype=bool)
    incremental["dC"] = expiry["dC"][:, :9].copy()
    incremental["dS"] = np.ones((2, 9), dtype=np.float32)
    incremental["option_dS"] = np.ones((2, 9, 2), dtype=np.float32)
    incremental["option_settlement_dS"] = np.full(
        (2, 9, 2), 2.0, dtype=np.float32
    )
    source = MatchedUniverseBlockCache(
        tmp_path,
        cache_uid="legacy",
        configuration={"schema_version": "1.2", "generator": "pca-iv"},
        num_samples=2,
    )
    source.begin()
    source.write(
        PCAMatchedUniverseBlock(
            start=0, stop=2, expiry=expiry, incremental=incremental
        )
    )
    source.complete(diagnostics={"status": "preserved"})

    block_path = next(source.directory.glob("block-*.npz"))
    with np.load(block_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    manifest = json.loads(source.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.2"
    manifest["blocks"][0]["fields"] = sorted(payload)
    source.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    migrated = migrate_legacy_option_settlement_cache(
        root=tmp_path,
        source_uid="legacy",
        destination_uid="current",
        destination_configuration={
            "schema_version": "1.3",
            "generator": "pca-iv",
        },
        num_samples=2,
        source_schema_version="1.2",
    )
    restored = next(migrated.iter_blocks())

    assert np.all(restored.incremental["option_dS"] == 2.0)
    assert np.all(restored.incremental["option_settlement_dS"] == 2.0)
    assert migrated.manifest()["diagnostics"] == {"status": "preserved"}


def test_streaming_unconditional_accumulator_matches_full_report_geometry() -> None:
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, 4, axis=0)
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    accumulator = TimeSeparatedUnconditionalAccumulator()
    for indices in (slice(0, 2), slice(2, 4)):
        block = derive_case_arrays(
            {
                name: (
                    value
                    if name in {"expiries", "option_expired"}
                    else value[indices]
                )
                for name, value in universe.items()
            },
            hedge="raw",
            cost_rate=0.0,
        )
        accumulator.update(
            block,
            importance_components=np.array([0, 1]),
        )
    report = accumulator.report()

    assert report["paths"] == 4
    assert report["direct_P_paths"] == 2
    assert report["trading_times"] == case.returns.shape[1]
    item = report["times"][0]["directions"]["long"][2]
    assert item["proposal_gain_count"] == 4


def test_option_coverage_reports_strike_and_both_hedge_sign_regions() -> None:
    universe = _expiry_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, 4, axis=0)
    accumulator = OptionCoverageAccumulator()

    accumulator.update(universe)
    report = accumulator.report(target_count=1)

    assert report["trading_times"] == 1
    assert report["expiries"] == 2
    assert set(report["minimum_counts"]) == {
        "crossing",
        "noncrossing",
        "raw_positive",
        "raw_negative",
        "delta_positive",
        "delta_negative",
    }


def test_incremental_coverage_gate_does_not_require_expiry_strike_crossing() -> None:
    support = {
        "minimum_counts": {
            "crossing": 0,
            "noncrossing": 0,
            "raw_positive": 2_000,
            "raw_negative": 2_001,
            "delta_positive": 3_000,
            "delta_negative": 3_001,
        }
    }
    unconditional = {
        "times": [
            {
                "directions": {
                    direction: [
                        {
                            "kind": "option",
                            "proposal_loss_count": 4_000,
                            "proposal_gain_count": 5_000,
                        }
                    ]
                    for direction in ("long", "short")
                }
            }
        ]
    }

    incremental = _coverage_gate(
        unconditional,
        support=support,
        target=1_000,
        return_horizon="incremental",
        hedge="delta",
    )
    expiry = _coverage_gate(
        unconditional,
        support=support,
        target=1_000,
        return_horizon="expiry",
        hedge="delta",
    )

    assert incremental["passed"]
    assert incremental["applicable_support_counts"] == {
        "delta_positive": 3_000,
        "delta_negative": 3_001,
    }
    assert not expiry["passed"]


def test_functional_gate_fails_closed_for_missing_check() -> None:
    names = (
        "finite_arrays",
        "timing_identities",
        "float64_accounting_replay",
        "decoder_identity",
        "strike_and_sign_boundary_coverage",
        "importance_normalization_and_moment_recovery",
        "stratified_disjoint_split",
        "conditional_radius_and_ess",
        "extremes_replayed",
        "memory_within_limits",
    )
    passed = functional_gate_report({name: True for name in names})
    failed = functional_gate_report(
        {name: True for name in names if name != "decoder_identity"}
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert failed["blockers"] == ["decoder_identity"]


def test_dynamic_arbitrage_score_is_ternary_and_fail_closed() -> None:
    complete_candidates = {
        "surviving": [],
        "conditional_regions": {
            "status": "complete",
            "surviving": [],
            "unresolved": [],
        },
        "local_conditional_baskets": {
            "status": "complete",
            "surviving": [],
            "unresolved": [],
        },
    }
    clear = _dynamic_arbitrage_score(
        gate={"passed": True},
        candidate_certification=complete_candidates,
        constant_baskets=[],
        static_spread_strategy={"has_pathwise_dynamic_arbitrage": False},
    )
    unresolved = _dynamic_arbitrage_score(
        gate={"passed": False},
        candidate_certification=complete_candidates,
        constant_baskets=[],
        static_spread_strategy={"has_pathwise_dynamic_arbitrage": False},
    )
    detected = _dynamic_arbitrage_score(
        gate={"passed": False},
        candidate_certification={
            **complete_candidates,
            "surviving": [{"certified": True}],
        },
        constant_baskets=[],
        static_spread_strategy={"has_pathwise_dynamic_arbitrage": False},
    )

    assert clear["has_dynamic_arbitrage"] is False
    assert unresolved["has_dynamic_arbitrage"] is None
    assert detected["has_dynamic_arbitrage"] is True


def test_completed_case_report_is_reused_only_when_identity_matches(
    tmp_path,
) -> None:
    case = RequiredCase(
        generator="dysanos",
        trading_frequency="daily",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=100_000,
    )
    report_path = tmp_path / (
        "dysanos-daily-100000-expiry-delta-1e-04.json"
    )
    report_path.write_text(
        __import__("json").dumps(
            {
                "status": "complete",
                "schema_version": _REPORT_SCHEMA_VERSION,
                    "configuration": case.__dict__,
                    "base_cache_uid": "base-a",
                    "functional_gate": {"passed": True},
                    "dynamic_arbitrage": {"has_dynamic_arbitrage": False},
            }
        ),
        encoding="utf-8",
    )

    assert (
        _load_completed_case_report(
            report_root=tmp_path, case=case, base_cache_uid="base-a"
        )
        is not None
    )
    assert (
        _load_completed_case_report(
            report_root=tmp_path, case=case, base_cache_uid="base-b"
        )
        is None
    )


def test_completed_case_report_rejects_placeholder_conditional_baskets(
    tmp_path,
) -> None:
    case = RequiredCase(
        generator="pca-iv",
        trading_frequency="daily",
        return_horizon="expiry",
        hedge="raw",
        cost_rate=1.0e-4,
        path_count=100_000,
    )
    report_path = tmp_path / "pca-iv-daily-100000-expiry-raw-1e-04.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "status": "complete",
                "schema_version": _REPORT_SCHEMA_VERSION,
                "configuration": case.__dict__,
                "base_cache_uid": "base-a",
                "functional_gate": {"passed": True},
                "dynamic_arbitrage": {"has_dynamic_arbitrage": False},
                "conditional": {
                    "status": "complete",
                    "times": [
                        {"local_conditional_baskets": {"fitted": 0}}
                        for _ in range(case.return_times)
                    ],
                },
                "candidate_certification": {
                    "conditional_regions": {"status": "complete"},
                    "local_conditional_baskets": {"status": "complete"},
                },
                "constant_baskets": [{} for _ in range(case.return_times)],
            }
        ),
        encoding="utf-8",
    )

    assert (
        _load_completed_case_report(
            report_root=tmp_path,
            case=case,
            base_cache_uid="base-a",
            require_conditional=True,
            require_baskets=True,
        )
        is None
    )


def test_conditional_neighbors_are_time_separated_and_leave_path_out() -> None:
    paths = 10
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, paths, axis=0)
    for name in (
        "dC",
        "dS",
        "option_dS",
        "option_settlement_dS",
        "C",
        "spot",
        "delta",
        "log_strikes",
        "xstate",
    ):
        universe[name] = np.repeat(universe[name], 2, axis=1)
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    state = np.zeros((paths, 2, 2), dtype=np.float32)
    state[:, 0, 0] = np.arange(paths)
    state[:, 1, 0] = 100.0 + np.arange(paths)
    components = np.zeros(paths, dtype=np.int16)
    split = stratified_path_split(components, seed=5)

    log_weights = np.linspace(-1_000.0, 0.0, paths)
    report = run_time_separated_local_tail(
        state=state,
        case=case,
        log_path_weights=log_weights,
        importance_components=components,
        split=split,
        neighbors=4,
        query_paths_per_split=2,
        instrument_block_size=2,
        seed=9,
        device=torch.device("cpu"),
        tail_ranks=(1, 2),
        progress_interval_seconds=1.0e9,
    )

    assert not report["time_pooling"]
    assert len(report["times"]) == 2
    assert report["split"]["reference_queries_leave_one_complete_path_out"]
    assert all(
        item["reference"]["reference_observations"] == 8
        for item in report["times"]
    )
    assert all(
        "frozen_candidate_region_audit" in item for item in report["times"]
    )
    assert all(
        "held_out_calibration" in item["audit"] for item in report["times"]
    )
    assert all("local_conditional_baskets" in item for item in report["times"])


def test_conditional_basket_search_does_not_depend_on_individual_directions() -> None:
    paths = 100
    returns = np.empty((paths, 1, 2), dtype=np.float32)
    returns[0::2, 0] = (1.0, -0.5)
    returns[1::2, 0] = (-1.0, 1.5)
    assert np.all(np.min(returns[:, 0], axis=0) < 0.0)
    assert np.all(np.max(returns[:, 0], axis=0) > 0.0)
    case = DerivedCaseArrays(
        returns=returns,
        costs=np.zeros_like(returns),
        numerical_band=np.full_like(returns, 1.0e-7),
        instrument_kind=("option", "option"),
        expiry_index=np.zeros(2, dtype=np.int32),
        strike_index=np.arange(2, dtype=np.int32),
        return_horizon="expiry",
        hedge="raw",
        cost_rate=0.0,
    )
    split = PathSplit(
        reference=np.arange(80, dtype=np.int64),
        audit=np.arange(80, paths, dtype=np.int64),
        component_counts={0: {"total": paths, "reference": 80, "audit": 20}},
        audit_fraction=0.2,
        seed=1,
    )

    report = _fit_and_audit_local_baskets(
        centers=np.zeros((1, 2), dtype=np.float32),
        reference_neighbor_indices=np.arange(80, dtype=np.int64)[None],
        neighbor_radius=np.ones(1, dtype=np.float32),
        audit_state=np.zeros((20, 2), dtype=np.float32),
        case=case,
        split=split,
        log_path_weights=np.zeros(paths, dtype=np.float64),
        importance_components=np.zeros(paths, dtype=np.int16),
        time_index=0,
        neighbors=80,
        family_size=1,
        prerequisite_checks_passed=True,
        progress_interval_seconds=1.0e9,
        solver_workers=4,
    )

    assert report["searched_all_query_states_independent_of_individual_screens"]
    assert report["searched_query_states"] == 1
    assert report["reference_candidates"] == 1
    assert report["zero_reference_solutions"] == 0
    assert len(report["surviving"]) == 1
    action = np.asarray(
        report["surviving"][0]["reference_solution"]["action"]
    )
    pnl = returns[:, 0] @ action
    assert np.all(pnl >= -1.0e-12)
    assert np.any(pnl > 0.0)


def test_frozen_candidate_replay_uses_global_target_probability() -> None:
    pnl = np.zeros(100, dtype=np.float64)
    pnl[:10] = 0.01
    report = summarize_frozen_replay(
        activated_pnl=pnl,
        activated_band=np.full(100, 1.0e-8),
        in_region=np.arange(100) < 10,
        log_path_weights=np.zeros(100, dtype=np.float64),
        importance_components=np.zeros(100, dtype=np.int16),
        family_size=1,
        prerequisite_checks_passed=True,
    )

    assert report["paths_in_region"] == 10
    assert report["gain_count"] == 10
    assert report["loss_count"] == 0
    assert report["global_weighted_P_gain_probability"] == pytest.approx(0.1)
    assert report["economic_certified"] is True


def test_state_dependent_static_butterfly_is_replayed_as_dynamic_arbitrage() -> None:
    paths = 100
    terminal_spot = np.resize(np.array([80.0, 100.0, 120.0]), paths)
    strikes = np.array([90.0, 100.0, 110.0])
    payoff = np.maximum(terminal_spot[:, None] - strikes[None], 0.0)
    calls = np.broadcast_to(
        np.array([15.0, 12.0, 5.0]), (paths, 1, 1, 3)
    ).copy()
    universe = {
        "dC": (payoff[:, None, None] - calls).astype(np.float32),
        "dS": (terminal_spot - 100.0)[:, None].astype(np.float32),
        "expiry_dS": (terminal_spot - 100.0)[:, None, None].astype(np.float32),
        "C": calls.astype(np.float32),
        "spot": np.full((paths, 1), 100.0, dtype=np.float32),
        "delta": np.zeros((paths, 1, 1, 3), dtype=np.float32),
        "log_strikes": np.broadcast_to(
            np.log(strikes / 100.0), (paths, 1, 1, 3)
        ).astype(np.float32),
        "option_expired": np.ones((1, 1, 1), dtype=bool),
        "expiries": np.full((paths, 1, 1), 0.1, dtype=np.float32),
        "xstate": np.zeros((paths, 1, 2), dtype=np.float32),
        "log_path_weight": np.zeros(paths, dtype=np.float64),
        "importance_component": np.zeros(paths, dtype=np.int16),
    }
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    accumulator = StateDependentStaticSpreadAccumulator()
    accumulator.update(universe, case)
    report = accumulator.report()

    assert report["has_pathwise_dynamic_arbitrage"]
    assert report["certified_trading_times"] == [0]
    time_report = report["times"][0]
    assert time_report["active_paths"] == paths
    assert time_report["loss_count"] == 0
    assert time_report["minimum_economic_pnl"] == pytest.approx(1.0)
    assert time_report["active_count_by_spread"]["strike_convexity"] == paths


def test_derived_case_store_streams_from_matched_cache(tmp_path) -> None:
    expiry = _expiry_universe()
    for name, value in list(expiry.items()):
        if name not in {"expiries", "option_expired"}:
            expiry[name] = np.repeat(value, 10, axis=1)
    incremental = {
        name: value[:, :9]
        for name, value in expiry.items()
        if name not in {"expiries", "option_expired", "expiry_dS", "dS"}
    }
    incremental.update(
        expiries=expiry["expiries"],
        option_expired=np.zeros((1, 1, 2), dtype=bool),
        dC=expiry["dC"][:, :9],
        dS=np.ones((1, 9), dtype=np.float32),
        option_dS=np.ones((1, 9, 2), dtype=np.float32),
        option_settlement_dS=np.ones((1, 9, 2), dtype=np.float32),
    )
    blocks = []
    for index in range(2):
        blocks.append(
            PCAMatchedUniverseBlock(
                start=index,
                stop=index + 1,
                expiry=expiry,
                incremental=incremental,
            )
        )
    case = RequiredCase(
        generator="pca-iv",
        trading_frequency="daily",
        return_horizon="incremental",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=2,
    )
    store = DerivedCaseStore(
        tmp_path, case=case, source_cache_uid="source-cache-test"
    )
    store.build(blocks)

    state, derived, weights, components = store.arrays()
    assert state.shape == (2, 9, 3)
    assert derived.returns.shape == (2, 9, 4)
    assert weights.shape == components.shape == (2,)
    assert np.all(np.isfinite(derived.numerical_band))
    del state, derived, weights, components
    store.cleanup()
    assert not store.directory.exists()
    store.cleanup()


def test_unconditional_candidate_is_frozen_then_certified_on_audit() -> None:
    paths = 40
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, paths, axis=0)
    universe["dC"][:, 0, 0, 0] = 1.0
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    components = np.zeros(paths, dtype=np.int16)
    split = stratified_path_split(components, seed=31)

    report = certify_unconditional_directions(
        case,
        split=split,
        log_path_weights=np.zeros(paths),
        importance_components=components,
        family_size=1,
        bootstrap_repetitions=20,
        seed=11,
        prerequisite_checks_passed=True,
    )

    assert report["reference_screened_candidates"] > 0
    assert report["surviving"]
    assert all(item["audit_loss_count"] == 0 for item in report["surviving"])


def test_constant_basket_is_frozen_then_evaluated_on_audit() -> None:
    paths = 40
    universe = _incremental_universe()
    for name, value in list(universe.items()):
        if name not in {"expiries", "option_expired"}:
            universe[name] = np.repeat(value, paths, axis=0)
    universe["dC"][:, 0, 0, 0] = np.linspace(0.1, 1.0, paths)
    case = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
    components = np.zeros(paths, dtype=np.int16)
    split = stratified_path_split(components, seed=71)
    solution = solve_constant_basket(
        case,
        time_index=0,
        reference_paths=split.reference,
    )

    audit = audit_constant_basket(
        solution,
        case,
        split=split,
        log_path_weights=np.zeros(paths),
        importance_components=components,
        family_size=1,
        prerequisite_checks_passed=True,
    )

    assert audit["audit_status"] == "evaluated frozen reference basket"
    assert audit["certified"]
    assert audit["audit_loss_count"] == 0
