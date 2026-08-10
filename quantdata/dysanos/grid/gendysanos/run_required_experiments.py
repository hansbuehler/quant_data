"""Execute the staged experiments required by :mod:`GENDYSANOS.md`."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.verbose import Context
from scipy.stats import norm

from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig

from .case_store import DerivedCaseStore
from .conditional import run_time_separated_local_tail
from .matched_cache import (
    MatchedUniverseBlockCache,
    migrate_legacy_option_settlement_cache,
)
from .required_runs import (
    REPORTED_COST_RATES,
    REPORTED_PATH_COUNTS,
    OptionCoverageAccumulator,
    RequiredCase,
    TimeSeparatedUnconditionalAccumulator,
    audit_constant_basket,
    certify_unconditional_directions,
    convergence_report,
    derive_case_arrays,
    float64_replay_report,
    functional_gate_report,
    required_cases,
    solve_constant_basket,
    stratified_path_split,
)
from .static_spread_strategy import StateDependentStaticSpreadAccumulator

_BASE_SCHEMA_VERSION = "1.3"
_REPORT_SCHEMA_VERSION = "2.8"
_DEFAULT_CACHE_ROOT = "/media/storage/tmp/quantdata/dysanos/required-runs"
_DEFAULT_REPORT_ROOT = "quantdata/dysanos/grid/gendysanos/reports/required"


def _required_importance_config() -> GaussianSpotImportanceConfig:
    """Return the common proposal spanning all ten action cohorts."""

    return GaussianSpotImportanceConfig(
        spot_std_multiplier=8.0,
        base_probability=0.1,
        target_action_count=10,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _memory_snapshot() -> dict[str, Any]:
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1_024
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = int(torch.cuda.max_memory_allocated())
        gpu_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "maximum_resident_bytes": int(maximum_rss),
        "maximum_cuda_allocated_bytes": gpu_allocated,
        "maximum_cuda_reserved_bytes": gpu_reserved,
    }


def _base_configuration(
    *, generator: str, frequency: str, paths: int, seed: int, block_size: int
) -> dict[str, Any]:
    return {
        "schema_version": _BASE_SCHEMA_VERSION,
        "generator": generator,
        "trading_frequency": frequency,
        "num_samples": paths,
        "num_trading_times": 10,
        "daily_intermediate_path": True,
        "return_horizons_derived": ["expiry", "incremental"],
        "hedges_derived": ["delta", "raw"],
        "cost_rates_derived": list(REPORTED_COST_RATES),
        "surface_first": True,
        "surface_state_dimension": 20,
        "num_alpha": None,
        "training_grid_strikes": 100,
        "traded_strikes_per_expiry": 20,
        "use_matrix_exp": True,
        "bandwidth": 1.0,
        "seed": seed,
        "block_size": block_size,
        "importance_sampling": asdict(_required_importance_config()),
    }


def _base_uid(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return (
        f"{configuration['generator']}-{configuration['trading_frequency']}-"
        f"{configuration['num_samples']}-{digest}"
    )


def _generate_base_cache(
    *,
    configuration: dict[str, Any],
    cache_root: Path,
    verbose: Context,
) -> MatchedUniverseBlockCache:
    """Generate one path sample shared by all twelve derived cases."""

    generator = configuration["generator"]
    frequency = configuration["trading_frequency"]
    paths = configuration["num_samples"]
    cache = MatchedUniverseBlockCache(
        cache_root / "universes",
        cache_uid=_base_uid(configuration),
        configuration=configuration,
        num_samples=paths,
    )
    if cache.is_complete:
        return cache
    for source_schema_version in ("1.2", "1.0"):
        legacy_configuration = dict(configuration)
        legacy_configuration["schema_version"] = source_schema_version
        legacy_uid = _base_uid(legacy_configuration)
        legacy_manifest = cache_root / "universes" / legacy_uid / "manifest.json"
        if legacy_manifest.exists():
            verbose.write(
                lambda: (
                    f"Migrating verified matched cache {legacy_uid} to "
                    f"{cache.cache_uid}"
                )
            )

            def migration_progress(block: int, total: int) -> None:
                if block == 1 or block == total or block % 10 == 0:
                    verbose.write(
                        lambda block=block, total=total: (
                            f"Migrated matched cache blocks {block:,}/{total:,}"
                        )
                    )

            return migrate_legacy_option_settlement_cache(
                root=cache_root / "universes",
                source_uid=legacy_uid,
                destination_uid=cache.cache_uid,
                destination_configuration=configuration,
                num_samples=paths,
                source_schema_version=source_schema_version,
                progress=migration_progress,
            )
    cache.begin()
    importance = _required_importance_config()
    if generator == "dysanos":
        from .torch_prep import (
            DysanosUniverseStreamDiagnostics,
            GenDysanosExperiment,
            finalize_dysanos_stream_diagnostics,
            iter_dysanos_matched_universe_blocks,
        )

        experiment = GenDysanosExperiment(
            num_samples=paths,
            trading_frequency=frequency,
            return_horizon="expiry",
            num_trading_times=10,
            num_alpha=None,
            use_matrix_exp=True,
            bandwidth=1.0,
            seed=configuration["seed"],
            block_size=configuration["block_size"],
            importance_sampling=importance,
        )
        diagnostics = DysanosUniverseStreamDiagnostics()
        for block in iter_dysanos_matched_universe_blocks(
            experiment, diagnostics=diagnostics, verbose=verbose
        ):
            cache.write(block)
        path_diagnostics = finalize_dysanos_stream_diagnostics(diagnostics)
        generated_diagnostics: dict[str, Any] = {
            "decoder": {
                "name": "ML-SANOS",
                "static_arbitrage_repair": False,
                "cross_section_static_arbitrage_free_by_construction": True,
                "incremental_call_roundoff_clip_count": (
                    diagnostics.incremental_call_roundoff_clip_count
                ),
                "minimum_unclipped_incremental_call": (
                    diagnostics.minimum_unclipped_incremental_call
                ),
                "roundoff_stabilization": (
                    "alive incremental normalized calls in [-32 eps32,0) are "
                    "set to the exact nonnegative boundary; larger negatives fail"
                ),
            },
            "sampling": asdict(path_diagnostics),
        }
    elif generator == "pca-iv":
        from quantdata.dysanos.implied.decoder import PCATotalVarianceDecoder
        from quantdata.dysanos.implied.experiment import ImpliedPCAExperiment
        from quantdata.dysanos.implied.market import load_fitted_pca
        from quantdata.dysanos.implied.simulator import ImpliedPCASampler
        from quantdata.dysanos.implied.static_arbitrage import (
            StaticArbitrageAccumulator,
        )
        from quantdata.dysanos.implied.universe import (
            PCAUniverseStreamDiagnostics,
            default_static_strike_mesh,
            finalize_stream_diagnostics,
            iter_pca_matched_universe_blocks,
        )

        experiment = ImpliedPCAExperiment(
            num_samples=paths,
            trading_frequency=frequency,
            return_horizon="expiry",
            num_trading_times=10,
            num_alpha=None,
            use_matrix_exp=True,
            bandwidth=1.0,
            seed=configuration["seed"],
            block_size=configuration["block_size"],
            importance_sampling=importance,
        )
        _, market, pca, pca_diagnostics = load_fitted_pca(
            experiment.fit_config, verbose=verbose
        )
        decoder = PCATotalVarianceDecoder(pca)
        sampler = ImpliedPCASampler(
            pca=pca,
            scores=pca_diagnostics.scores,
            time_y=market.time_y,
            spot_ref=market.spot_ref,
            num_alpha=experiment.num_alpha,
            seed=experiment.seed,
            verbose=verbose,
        )
        static_strikes = default_static_strike_mesh(
            experiment.static_mesh_points
        )
        static = StaticArbitrageAccumulator(
            strikes=static_strikes,
            expiries=pca.expiries,
            roundoff_multipliers=experiment.roundoff_multipliers,
            seed=experiment.seed + 911,
        )
        diagnostics = PCAUniverseStreamDiagnostics()
        for block in iter_pca_matched_universe_blocks(
            experiment,
            decoder=decoder,
            sampler=sampler,
            diagnostics=diagnostics,
            static_accumulator=static,
            static_strikes=static_strikes,
            verbose=verbose,
        ):
            cache.write(block)
        path_diagnostics = finalize_stream_diagnostics(diagnostics)
        generated_diagnostics = {
            "decoder": {
                "name": "PCA-IV",
                "static_arbitrage_repair": False,
                "sample_rejection": False,
            },
            "sampling": asdict(path_diagnostics),
            "static_arbitrage": static.report(),
        }
    else:
        raise ValueError(f"unknown generator {generator!r}")
    cache.complete(diagnostics=_json_safe(generated_diagnostics))
    return cache


def _importance_gate(
    sampling: dict[str, Any],
    *,
    paths: int,
    unconditional: dict[str, Any],
) -> dict[str, Any]:
    importance = sampling.get("importance_sampling")
    if importance is None:
        return {"passed": False, "reason": "importance sampling is absent"}
    path = importance["path_weights"]
    mean = float(path["mean_density"])
    ess = float(path["independent_path_effective_sample_size"])
    second = mean * mean * paths / ess
    standard_error = math.sqrt(max(second - mean * mean, 0.0) / paths)
    controls = importance["log_spot_control_at_target_steps"]
    targets = max(1, len(importance["target_steps"]))
    quantile = float(norm.ppf(1.0 - 0.05 / (2.0 * targets)))
    mean_tolerance = quantile * math.sqrt(1.0 / paths + 1.0 / ess)
    variance_tolerance = quantile * math.sqrt(
        2.0 / max(paths - 1, 1) + 2.0 / max(ess - 1.0, 1.0)
    )
    return_items = [
        item
        for time_report in unconditional["times"]
        for direction in ("long", "short")
        for item in time_report["directions"][direction]
    ]
    return_quantile = float(
        norm.ppf(1.0 - 0.05 / (2.0 * max(1, len(return_items))))
    )
    return_failures = 0
    maximum_return_z = 0.0
    for item in return_items:
        error = math.hypot(
            float(item["weighted_P_path_standard_error"]),
            float(item["direct_P_path_standard_error"]),
        )
        difference = abs(
            float(item["weighted_P_mean"]) - float(item["direct_P_mean"])
        )
        z_score = difference / error if error > 0.0 else (math.inf if difference else 0.0)
        maximum_return_z = max(maximum_return_z, z_score)
        return_failures += int(difference > return_quantile * error)
    checks = {
        "density_normalization": abs(mean - 1.0) <= 3.0 * standard_error,
        "target_log_spot_mean_recovery": float(
            controls["maximum_mean_difference_in_direct_P_std"]
        )
        <= mean_tolerance,
        "target_log_spot_variance_recovery": float(
            controls["maximum_relative_variance_difference"]
        )
        <= variance_tolerance,
        "positive_effective_sample_size": 1.0 < ess <= paths,
        "direct_P_and_weighted_return_mean_recovery": return_failures == 0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mean_density": mean,
        "mean_density_standard_error": standard_error,
        "effective_sample_size": ess,
        "simultaneous_mean_tolerance_in_std": mean_tolerance,
        "simultaneous_relative_variance_tolerance": variance_tolerance,
        "return_mean_simultaneous_normal_quantile": return_quantile,
        "maximum_return_mean_difference_z": maximum_return_z,
        "return_mean_recovery_failures": return_failures,
    }


def _coverage_gate(
    report: dict[str, Any],
    *,
    support: dict[str, Any],
    target: int,
    return_horizon: str,
    hedge: str,
) -> dict[str, Any]:
    minimum = math.inf
    for time_report in report["times"]:
        for direction in ("long", "short"):
            for item in time_report["directions"][direction]:
                if item["kind"] != "option":
                    continue
                minimum = min(
                    minimum,
                    item["proposal_loss_count"],
                    item["proposal_gain_count"],
                )
    minimum = int(minimum) if math.isfinite(minimum) else 0
    required_support = [
        f"{hedge}_positive",
        f"{hedge}_negative",
    ]
    if return_horizon == "expiry":
        required_support.extend(("crossing", "noncrossing"))
    applicable_counts = {
        name: int(support["minimum_counts"][name]) for name in required_support
    }
    return {
        "passed": minimum >= target
        and all(value >= target for value in applicable_counts.values()),
        "target_count_per_cost_adjusted_sign_side": target,
        "minimum_cost_adjusted_sign_count": minimum,
        "applicable_support_counts": applicable_counts,
        "strike_crossing_required": return_horizon == "expiry",
        "minimum_strike_and_precost_sign_counts": support["minimum_counts"],
    }


def _conditional_gate(report: dict[str, Any]) -> dict[str, Any]:
    radii: list[float] = []
    effective_sizes: list[float] = []
    for item in report["times"]:
        for split in ("reference", "audit"):
            radii.append(float(item[split]["neighbor_radius"]["mean"]))
            effective_sizes.append(
                float(item[split]["local_effective_sample_size"]["mean"])
            )
    finite = np.all(np.isfinite(radii)) and np.all(np.isfinite(effective_sizes))
    return {
        "passed": bool(
            finite
            and np.all(np.asarray(radii) > 0.0)
            and np.all(np.asarray(effective_sizes) > 1.0)
        ),
        "mean_radii_by_time_and_split": radii,
        "mean_effective_sizes_by_time_and_split": effective_sizes,
    }


def _dynamic_arbitrage_score(
    *,
    gate: dict[str, Any],
    candidate_certification: dict[str, Any],
    constant_baskets: list[dict[str, Any]],
    static_spread_strategy: dict[str, Any],
) -> dict[str, Any]:
    """Summarize independently audited candidates without hiding uncertainty."""

    conditional = candidate_certification.get("conditional_regions", {})
    local_baskets = candidate_certification.get(
        "local_conditional_baskets", {}
    )
    surviving_by_class = {
        "unconditional_unit_direction": len(
            candidate_certification.get("surviving", [])
        ),
        "unconditional_constant_basket": sum(
            item.get("certified") is True for item in constant_baskets
        ),
        "conditional_unit_region": len(conditional.get("surviving", [])),
        "conditional_local_basket": len(local_baskets.get("surviving", [])),
        "state_dependent_static_spread": int(
            static_spread_strategy.get("has_pathwise_dynamic_arbitrage") is True
        ),
    }
    unresolved_by_class = {
        "conditional_unit_region": len(conditional.get("unresolved", [])),
        "conditional_local_basket": len(local_baskets.get("unresolved", [])),
    }
    surviving = sum(surviving_by_class.values())
    unresolved = sum(unresolved_by_class.values())
    diagnostics_complete = (
        gate.get("passed") is True
        and conditional.get("status") == "complete"
        and local_baskets.get("status") == "complete"
    )
    if surviving:
        score: bool | None = True
        assessment = "dynamic arbitrage detected within a tested strategy class"
    elif diagnostics_complete and not unresolved:
        score = False
        assessment = (
            "no per-trading-time conditional arbitrage detected within the "
            "tested strategy classes and finite-sample resolution"
        )
    else:
        score = None
        assessment = "unresolved"
    return {
        "has_dynamic_arbitrage": score,
        "assessment": assessment,
        "strategy_classes": tuple(surviving_by_class),
        "surviving_candidates_by_class": surviving_by_class,
        "unresolved_candidates_by_class": unresolved_by_class,
        "functional_gate_passed": gate.get("passed") is True,
        "all_required_candidate_audits_resolved": diagnostics_complete
        and not unresolved,
        "qualification": (
            "the score concerns the tested strategy classes only and is not a "
            "proof over all predictable self-financing strategies"
        ),
    }


def _timing_replay(block, *, frequency: str) -> dict[str, Any]:
    """Replay matched timestamp and exact-expiry identities for one block."""

    expiry = block.expiry
    incremental = block.incremental
    stride = 5 if frequency == "weekly" else 1
    expiries = np.asarray(expiry["expiries"], dtype=np.float64).reshape(-1)
    offsets = np.rint(expiries * 252.0).astype(np.int64)
    expected_expired = offsets <= stride
    checks = {
        "ten_expiry_cohorts": expiry["C"].shape[1] == 10,
        "nine_incremental_intervals": incremental["C"].shape[1] == 9,
        "shared_trading_states": np.array_equal(
            incremental["xstate"], expiry["xstate"][:, :9]
        ),
        "shared_entry_prices": np.array_equal(
            incremental["C"], expiry["C"][:, :9]
        ),
        "shared_entry_deltas": np.array_equal(
            incremental["delta"], expiry["delta"][:, :9]
        ),
        "incremental_spot_endpoint": np.allclose(
            incremental["dS"],
            expiry["spot"][:, 1:] - expiry["spot"][:, :9],
            rtol=0.0,
            atol=32.0
            * np.finfo(np.float32).eps
            * np.maximum(np.abs(expiry["spot"][:, 1:]), 1.0e-30),
        ),
        "incremental_option_hedge_endpoint": np.allclose(
            incremental["option_dS"],
            np.where(
                expected_expired[None, None, :],
                expiry["expiry_dS"][:, :9],
                incremental["dS"][..., None],
            ),
            rtol=0.0,
            atol=32.0
            * np.finfo(np.float32).eps
            * np.maximum(np.abs(expiry["spot"][:, 1:, None]), 1.0e-30),
        ),
        "incremental_option_settlement_endpoint": np.allclose(
            incremental["option_settlement_dS"],
            np.where(
                expected_expired[None, None, :],
                expiry["expiry_dS"][:, :9],
                incremental["dS"][..., None],
            ),
            rtol=0.0,
            atol=32.0
            * np.finfo(np.float32).eps
            * np.maximum(np.abs(expiry["spot"][:, 1:, None]), 1.0e-30),
        ),
        "inside_interval_expiry_marker": np.array_equal(
            np.asarray(incremental["option_expired"]).reshape(-1),
            expected_expired,
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "stride": stride}


def _run_case(
    *,
    case: RequiredCase,
    cache: MatchedUniverseBlockCache,
    cache_root: Path,
    report_root: Path,
    device: torch.device,
    run_conditional: bool,
    run_baskets: bool,
) -> dict[str, Any]:
    label = (
        f"{case.generator}-{case.trading_frequency}-{case.path_count}-"
        f"{case.return_horizon}-{case.hedge}-{case.cost_rate:.0e}"
    )
    report_path = report_root / f"{label}.json"
    report: dict[str, Any] = {
        "status": "running",
        "schema_version": _REPORT_SCHEMA_VERSION,
        "configuration": asdict(case),
        "base_cache": str(cache.directory),
        "base_cache_uid": cache.cache_uid,
        "return_definition": case.return_horizon,
        "hedge_definition": (
            "raw option return"
            if case.hedge == "raw"
            else (
                "option return minus entry delta times spot return to the "
                "option close: exact expiry inside an interval, otherwise "
                "the next trading time"
            )
        ),
        "cost_definition": (
            "spot q*abs(S); raw option q*abs(C); delta-hedged option "
            "q*(abs(C)+abs(delta*S)); no option/spot netting"
        ),
        "numerical_tolerance": (
            "eta=32*finfo(float32).eps*max(unreduced arithmetic scale+cost, "
            "finfo(float32).tiny); no cash floor"
        ),
    }
    _atomic_json(report_path, report)
    accumulator = TimeSeparatedUnconditionalAccumulator()
    support_accumulator = OptionCoverageAccumulator()
    static_spread_accumulator = (
        StateDependentStaticSpreadAccumulator()
        if case.return_horizon == "expiry" and case.hedge == "raw"
        else None
    )
    replay_passed = True
    replay_has_expiry_payoff = False
    replay_max_return = replay_max_cost = replay_max_expiry = 0.0
    components = np.empty(case.path_count, dtype=np.int16)
    timing_checks: list[dict[str, Any]] = []
    for block in cache.iter_blocks():
        timing_checks.append(
            _timing_replay(block, frequency=case.trading_frequency)
        )
        universe = getattr(block, case.return_horizon)
        derived = derive_case_arrays(
            universe, hedge=case.hedge, cost_rate=case.cost_rate
        )
        if static_spread_accumulator is not None:
            static_spread_accumulator.update(universe, derived)
        log_weight = np.asarray(
            universe.get("log_path_weight", np.zeros(block.stop - block.start))
        )
        component = np.asarray(
            universe.get(
                "importance_component", np.zeros(block.stop - block.start)
            )
        )
        accumulator.update(
            derived,
            log_path_weights=log_weight,
            importance_components=component,
        )
        support_accumulator.update(universe, log_path_weights=log_weight)
        components[block.start : block.stop] = component
        replay = float64_replay_report(universe, derived)
        replay_passed &= replay["passed"]
        replay_max_return = max(
            replay_max_return, replay["maximum_absolute_return_identity_error"]
        )
        replay_max_cost = max(
            replay_max_cost, replay["maximum_absolute_cost_identity_error"]
        )
        replay_expiry = replay["maximum_absolute_expiry_payoff_identity_error"]
        if replay_expiry is not None:
            replay_has_expiry_payoff = True
            replay_max_expiry = max(replay_max_expiry, replay_expiry)
    unconditional = accumulator.report()
    option_coverage = support_accumulator.report()
    split = stratified_path_split(components, seed=case.path_count + case.gpu)
    split_report = {
        "reference_paths": int(split.reference.size),
        "audit_paths": int(split.audit.size),
        "disjoint": np.intersect1d(split.reference, split.audit).size == 0,
        "mixture_stratified": True,
        "component_counts": split.component_counts,
    }
    float64_replay = {
        "passed": replay_passed,
        "maximum_absolute_return_identity_error": replay_max_return,
        "maximum_absolute_cost_identity_error": replay_max_cost,
        "maximum_absolute_expiry_payoff_identity_error": (
            replay_max_expiry if replay_has_expiry_payoff else None
        ),
    }
    timing_replay = {
        "passed": bool(all(item["passed"] for item in timing_checks)),
        "blocks_checked": len(timing_checks),
        "failed_blocks": [
            index for index, item in enumerate(timing_checks) if not item["passed"]
        ],
        "checks": timing_checks[0]["checks"] if timing_checks else {},
    }
    report.update(
        unconditional=unconditional,
        option_coverage=option_coverage,
        split=split_report,
        float64_replay=float64_replay,
        timing_replay=timing_replay,
    )
    static_spread_strategy = (
        static_spread_accumulator.report()
        if static_spread_accumulator is not None
        else {
            "applicability": "raw option returns held to expiry",
            "status": "not applicable to this return/hedge case",
            "has_pathwise_dynamic_arbitrage": False,
        }
    )
    report["state_dependent_static_spread_strategy"] = static_spread_strategy
    base_diagnostics = cache.manifest()["diagnostics"]
    sampling = base_diagnostics["sampling"]
    coverage_check = _coverage_gate(
        unconditional,
        support=option_coverage,
        target=1_000,
        return_horizon=case.return_horizon,
        hedge=case.hedge,
    )
    importance_check = _importance_gate(
        sampling,
        paths=case.path_count,
        unconditional=unconditional,
    )
    prerequisite_checks_passed = (
        coverage_check["passed"]
        and importance_check["passed"]
        and float64_replay["passed"]
    )

    conditional: dict[str, Any] = {"status": "not-run"}
    baskets: list[dict[str, Any]] = []
    stored_case = None
    stored_log_weight = None
    stored_components = None
    store = None
    if run_conditional or run_baskets:
        store = DerivedCaseStore(
            cache_root / "case-work",
            case=case,
            source_cache_uid=cache.cache_uid,
        )
        if not store.is_complete:
            store.build(cache.iter_blocks())
        state, stored_case, log_weight, stored_components = store.arrays()
        stored_log_weight = log_weight
        if run_conditional:
            neighbors = 4_096 if case.path_count >= 1_000_000 else 1_024
            conditional = run_time_separated_local_tail(
                state=state,
                case=stored_case,
                log_path_weights=log_weight,
                importance_components=stored_components,
                split=split,
                neighbors=neighbors,
                query_paths_per_split=min(1_024, split.audit.size),
                instrument_block_size=20,
                seed=case.path_count + 101,
                device=device,
                certification_prerequisites_passed=(
                    prerequisite_checks_passed
                ),
                basket_solver_workers=min(12, max(1, (os.cpu_count() or 1) // 2)),
            )
        if run_baskets:
            for time_index in range(stored_case.returns.shape[1]):
                baskets.append(
                    solve_constant_basket(
                        stored_case,
                        time_index=time_index,
                        reference_paths=split.reference,
                        log_path_weights=log_weight,
                    )
                )
    report["conditional"] = conditional
    report["trained_strategy_class"] = {
        "individual": "unit long and short directions",
        "constant_basket": "state-independent L1-bounded linear program",
        "conditional_unit": "single-instrument k-nearest-neighbour state policy",
        "conditional_basket": (
            "independent all-instrument L1-bounded linear program at every "
            "queried state and trading time"
        ),
    }
    if (
        baskets
        and stored_case is not None
        and stored_log_weight is not None
        and stored_components is not None
    ):
        baskets = [
            audit_constant_basket(
                solution,
                stored_case,
                split=split,
                log_path_weights=stored_log_weight,
                importance_components=stored_components,
                prerequisite_checks_passed=prerequisite_checks_passed,
            )
            for solution in baskets
        ]
    report["constant_baskets"] = baskets
    if (
        stored_case is not None
        and stored_log_weight is not None
        and stored_components is not None
    ):
        candidate_certification = certify_unconditional_directions(
            stored_case,
            split=split,
            log_path_weights=stored_log_weight,
            importance_components=stored_components,
            seed=case.path_count + 313,
            prerequisite_checks_passed=prerequisite_checks_passed,
        )
        conditional_surviving: list[dict[str, Any]] = []
        conditional_rejected: list[dict[str, Any]] = []
        conditional_unresolved: list[dict[str, Any]] = []
        basket_surviving: list[dict[str, Any]] = []
        basket_rejected: list[dict[str, Any]] = []
        basket_unresolved: list[dict[str, Any]] = []
        for time_report in conditional.get("times", []):
            audit = time_report["frozen_candidate_region_audit"]
            conditional_surviving.extend(audit["surviving"])
            conditional_rejected.extend(audit["rejected"])
            conditional_unresolved.extend(audit["unresolved"])
            local_basket = time_report["local_conditional_baskets"]
            basket_surviving.extend(local_basket["surviving"])
            basket_rejected.extend(local_basket["rejected"])
            basket_unresolved.extend(local_basket["unresolved"])
        candidate_certification["conditional_regions"] = {
            "status": (
                "complete" if conditional.get("status") == "complete" else "not-run"
            ),
            "surviving": conditional_surviving,
            "rejected": conditional_rejected,
            "unresolved": conditional_unresolved,
        }
        candidate_certification["local_conditional_baskets"] = {
            "status": (
                "complete" if conditional.get("status") == "complete" else "not-run"
            ),
            "surviving": basket_surviving,
            "rejected": basket_rejected,
            "unresolved": basket_unresolved,
        }
    else:
        candidate_certification = {
            "status": "not-run",
            "surviving": [],
            "rejected": [],
        }
    report["candidate_certification"] = candidate_certification
    memory = _memory_snapshot()
    gate = functional_gate_report(
        {
            "finite_arrays": True,
            "timing_identities": timing_replay,
            "float64_accounting_replay": float64_replay,
            "decoder_identity": {
                "passed": base_diagnostics["decoder"]["static_arbitrage_repair"]
                is False,
                **base_diagnostics["decoder"],
            },
            "strike_and_sign_boundary_coverage": coverage_check,
            "importance_normalization_and_moment_recovery": importance_check,
            "stratified_disjoint_split": {
                "passed": split_report["disjoint"]
                and split.reference.size == 4 * case.path_count // 5,
                **split_report,
            },
            "conditional_radius_and_ess": (
                _conditional_gate(conditional)
                if conditional.get("status") == "complete"
                else {"passed": False, "reason": "conditional diagnostics not run"}
            ),
            "extremes_replayed": float64_replay,
            "memory_within_limits": {
                "passed": memory["maximum_resident_bytes"] < 900 * 2**30
                and memory["maximum_cuda_reserved_bytes"] < 92 * 2**30,
                **memory,
            },
        }
    )
    if case.path_count == 1_000:
        gate["reported_stage_applicability"] = (
            "mechanical smoke only; coverage thresholds are intentionally not "
            "relaxed and this gate does not authorize a reported stage"
        )
    dynamic_arbitrage = _dynamic_arbitrage_score(
        gate=gate,
        candidate_certification=candidate_certification,
        constant_baskets=baskets,
        static_spread_strategy=static_spread_strategy,
    )
    if store is not None:
        del state, log_weight, stored_case, stored_log_weight, stored_components
        gc.collect()
        store.cleanup()
        report["temporary_case_store"] = {
            "path": str(store.directory),
            "removed_after_diagnostics": True,
        }
    report.update(
        status="complete",
        functional_gate=gate,
        dynamic_arbitrage=dynamic_arbitrage,
        peak_memory=memory,
        sampling_diagnostics=sampling,
        generator_diagnostics=base_diagnostics,
    )
    _atomic_json(report_path, report)
    return report


def run_worker(
    *,
    generator: str,
    frequency: str,
    paths: int,
    horizons: tuple[str, ...],
    hedges: tuple[str, ...],
    costs: tuple[float, ...],
    cache_root: Path,
    report_root: Path,
    seed: int,
    block_size: int,
    run_conditional: bool,
    run_baskets: bool,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run one generator-frequency worker on its assigned visible GPU."""

    configuration = _base_configuration(
        generator=generator,
        frequency=frequency,
        paths=paths,
        seed=seed,
        block_size=block_size,
    )
    cache = _generate_base_cache(
        configuration=configuration,
        cache_root=cache_root,
        verbose=Context.all,
    )
    reports = []
    for horizon in horizons:
        for hedge in hedges:
            for cost in costs:
                case = RequiredCase(
                    generator=generator,
                    trading_frequency=frequency,
                    return_horizon=horizon,
                    hedge=hedge,
                    cost_rate=cost,
                    path_count=paths,
                )
                completed = _load_completed_case_report(
                    report_root=report_root,
                    case=case,
                    base_cache_uid=cache.cache_uid,
                    require_conditional=run_conditional,
                    require_baskets=run_baskets,
                )
                if completed is not None:
                    reports.append(completed)
                    continue
                reports.append(
                    _run_case(
                        case=case,
                        cache=cache,
                        cache_root=cache_root,
                        report_root=report_root,
                        device=device,
                        run_conditional=run_conditional,
                        run_baskets=run_baskets,
                    )
                )
    return reports


def _case_report_path(report_root: Path, case: RequiredCase) -> Path:
    return report_root / (
        f"{case.generator}-{case.trading_frequency}-{case.path_count}-"
        f"{case.return_horizon}-{case.hedge}-{case.cost_rate:.0e}.json"
    )


def _load_completed_case_report(
    *,
    report_root: Path,
    case: RequiredCase,
    base_cache_uid: str,
    require_conditional: bool = False,
    require_baskets: bool = False,
) -> dict[str, Any] | None:
    """Return a configuration-matched completed report for restart recovery."""

    path = _case_report_path(report_root, case)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    conditional = report.get("conditional", {})
    certification = report.get("candidate_certification", {})
    conditional_times = conditional.get("times", [])
    has_current_conditional = (
        conditional.get("status") == "complete"
        and certification.get("conditional_regions", {}).get("status")
        == "complete"
        and certification.get("local_conditional_baskets", {}).get("status")
        == "complete"
        and len(conditional_times) == case.return_times
        and all(
            "searched_query_states"
            in item.get("local_conditional_baskets", {})
            for item in conditional_times
        )
    )
    has_current_baskets = len(report.get("constant_baskets", [])) == (
        case.return_times
    )
    if (
        report.get("status") != "complete"
        or report.get("schema_version") != _REPORT_SCHEMA_VERSION
        or report.get("configuration") != asdict(case)
        or report.get("base_cache_uid") != base_cache_uid
        or "dynamic_arbitrage" not in report
        or (require_conditional and not has_current_conditional)
        or (require_baskets and not has_current_baskets)
        or (
            case.cost_rate == 0.0
            and (
                not report.get("constant_baskets")
                or not all(
                    item.get("zero_cost_signed_action_formulation")
                    == "free-signed-action-v2"
                    for item in report.get("constant_baskets", [])
                )
            )
        )
        or (
            case.path_count in REPORTED_PATH_COUNTS
            and not report.get("functional_gate", {}).get("passed", False)
        )
    ):
        return None
    return report


def _stage_passed(
    *,
    report_root: Path,
    paths: int,
    horizon: str,
    hedge: str,
    cost: float,
) -> bool:
    for generator, frequency in (
        ("dysanos", "daily"),
        ("dysanos", "weekly"),
        ("pca-iv", "daily"),
        ("pca-iv", "weekly"),
    ):
        case = RequiredCase(
            generator=generator,
            trading_frequency=frequency,
            return_horizon=horizon,
            hedge=hedge,
            cost_rate=cost,
            path_count=paths,
        )
        path = _case_report_path(report_root, case)
        if not path.exists():
            return False
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or not report.get(
            "functional_gate", {}
        ).get("passed", False):
            return False
    return True


def run_four_gpu_stage(
    *,
    paths: int,
    horizon: str,
    hedge: str,
    costs: tuple[float, ...],
    cache_root: Path,
    report_root: Path,
    seed: int,
    block_size: int,
    run_conditional: bool,
    run_baskets: bool,
) -> None:
    """Run the four prescribed generator-frequency workers concurrently."""

    processes: list[subprocess.Popen] = []
    for gpu, (generator, frequency) in enumerate(
        (
            ("dysanos", "daily"),
            ("dysanos", "weekly"),
            ("pca-iv", "daily"),
            ("pca-iv", "weekly"),
        )
    ):
        command = [
            sys.executable,
            "-m",
            "quantdata.dysanos.grid.gendysanos.run_required_experiments",
            "--worker",
            "--generator",
            generator,
            "--frequency",
            frequency,
            "--paths",
            str(paths),
            "--horizons",
            horizon,
            "--hedges",
            hedge,
            "--costs",
            *(str(value) for value in costs),
            "--cache-root",
            str(cache_root),
            "--report-root",
            str(report_root),
            "--seed",
            str(seed),
            "--block-size",
            str(block_size),
            "--device",
            "cuda",
        ]
        if run_conditional:
            command.append("--conditional")
        if run_baskets:
            command.append("--baskets")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes.append(subprocess.Popen(command, env=environment))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"four-GPU stage failed with exit codes {failures}")


def run_required_sequence(args: argparse.Namespace) -> None:
    """Run smoke, 100,000, and gated one-million stages in required order."""

    cache_root = Path(args.cache_root)
    report_root = Path(args.report_root)
    for horizon in ("expiry", "incremental"):
        for hedge in ("delta", "raw"):
            run_four_gpu_stage(
                paths=1_000,
                horizon=horizon,
                hedge=hedge,
                costs=(1.0e-4,),
                cache_root=cache_root,
                report_root=report_root,
                seed=args.seed,
                block_size=min(args.block_size, 1_000),
                run_conditional=True,
                run_baskets=True,
            )
            run_four_gpu_stage(
                paths=100_000,
                horizon=horizon,
                hedge=hedge,
                costs=(1.0e-4,),
                cache_root=cache_root,
                report_root=report_root,
                seed=args.seed,
                block_size=args.block_size,
                run_conditional=True,
                run_baskets=True,
            )
            if not _stage_passed(
                report_root=report_root,
                paths=100_000,
                horizon=horizon,
                hedge=hedge,
                cost=1.0e-4,
            ):
                raise RuntimeError(
                    "functional gate failed; the one-million stage was not launched"
                )
            run_four_gpu_stage(
                paths=1_000_000,
                horizon=horizon,
                hedge=hedge,
                costs=(1.0e-4,),
                cache_root=cache_root,
                report_root=report_root,
                seed=args.seed,
                block_size=args.block_size,
                run_conditional=True,
                run_baskets=True,
            )
    for paths in (100_000, 1_000_000):
        for horizon in ("expiry", "incremental"):
            for hedge in ("delta", "raw"):
                run_four_gpu_stage(
                    paths=paths,
                    horizon=horizon,
                    hedge=hedge,
                    costs=(0.0, 1.0e-3),
                    cache_root=cache_root,
                    report_root=report_root,
                    seed=args.seed,
                    block_size=args.block_size,
                    run_conditional=True,
                    run_baskets=True,
                )
    write_aggregate_report(report_root)


def write_aggregate_report(report_root: Path) -> dict[str, Any]:
    """Write a complete matrix report and 100k-to-one-million comparisons."""

    reports: dict[int, list[dict[str, Any]]] = {}
    missing: dict[int, list[dict[str, Any]]] = {}
    for paths in (100_000, 1_000_000):
        reports[paths] = []
        missing[paths] = []
        for case in required_cases(paths):
            path = _case_report_path(report_root, case)
            if path.exists():
                reports[paths].append(json.loads(path.read_text(encoding="utf-8")))
            else:
                missing[paths].append(asdict(case))
    convergence: list[dict[str, Any]] = []
    large_by_key = {
        _comparison_key(item["configuration"]): item
        for item in reports[1_000_000]
    }
    for small in reports[100_000]:
        key = _comparison_key(small["configuration"])
        large = large_by_key.get(key)
        if large is None:
            continue
        small_metrics = _convergence_metrics(small)
        large_metrics = _convergence_metrics(large)
        if small_metrics is None or large_metrics is None:
            convergence.append(
                {"case": key, "status": "unresolved-incomplete-diagnostics"}
            )
            continue
        convergence.append(
            {
                "case": key,
                "status": "complete",
                "assessment": convergence_report(
                    small_metrics, large_metrics
                ),
                "candidate_rate_100000": small_metrics["candidate_rate"],
                "candidate_rate_1000000": large_metrics["candidate_rate"],
            }
        )
    summaries = {
        str(paths): [
            _aggregate_case_summary(item, report_root=report_root)
            for item in reports[paths]
        ]
        for paths in (100_000, 1_000_000)
    }
    score_counts = {
        str(paths): {
            "has_dynamic_arbitrage": sum(
                item["dynamic_arbitrage"]["has_dynamic_arbitrage"] is True
                for item in reports[paths]
            ),
            "no_dynamic_arbitrage_detected": sum(
                item["dynamic_arbitrage"]["has_dynamic_arbitrage"] is False
                for item in reports[paths]
            ),
            "unresolved": sum(
                item["dynamic_arbitrage"]["has_dynamic_arbitrage"] is None
                for item in reports[paths]
            ),
        }
        for paths in (100_000, 1_000_000)
    }
    aggregate = {
        "status": (
            "complete" if not missing[100_000] and not missing[1_000_000] else "incomplete"
        ),
        "expected_cases_per_path_count": 48,
        "completed_cases": {str(key): len(value) for key, value in reports.items()},
        "missing_cases": missing,
        "qualification": (
            "finite Monte Carlo can report no per-trading-time conditional "
            "arbitrage detected within the tested strategy class and resolution; "
            "it cannot prove mathematical absence of arbitrary dynamic arbitrage"
        ),
        "score_counts": score_counts,
        "cases": summaries,
        "convergence": convergence,
    }
    _atomic_json(report_root / "aggregate.json", aggregate)
    return aggregate


def _aggregate_case_summary(
    report: dict[str, Any], *, report_root: Path
) -> dict[str, Any]:
    """Return compact comparison fields while retaining the detailed report."""

    configuration = report["configuration"]
    case = RequiredCase(**configuration)
    static = (
        report.get("generator_diagnostics", {})
        .get("static_arbitrage", {})
        .get("primary", {})
        .get("aggregate")
    )
    importance = report.get("functional_gate", {}).get("checks", {}).get(
        "importance_normalization_and_moment_recovery"
    )
    return {
        "status": report.get("status"),
        "configuration": configuration,
        "report_file": str(_case_report_path(report_root, case)),
        "base_cache_uid": report.get("base_cache_uid"),
        "functional_gate_passed": report.get("functional_gate", {}).get("passed"),
        "dynamic_arbitrage": report.get("dynamic_arbitrage"),
        "importance": importance,
        "static_arbitrage_primary": static,
        "peak_memory": report.get("peak_memory"),
    }


def _comparison_key(configuration: dict[str, Any]) -> str:
    """Return the path-count-independent identity of a reported case."""

    return "|".join(
        str(configuration[name])
        for name in (
            "generator",
            "trading_frequency",
            "return_horizon",
            "hedge",
            "cost_rate",
        )
    )


def _convergence_metrics(report: dict[str, Any]) -> dict[str, float] | None:
    """Extract rate-based metrics used by the required convergence check."""

    conditional = report.get("conditional", {})
    if conditional.get("status") != "complete":
        return None
    sampling = report["sampling_diagnostics"]["importance_sampling"]
    path = sampling["path_weights"]
    paths = float(report["configuration"]["path_count"])
    weight_mean = float(path["mean_density"])
    effective_size = float(path["independent_path_effective_sample_size"])
    second = weight_mean * weight_mean * paths / effective_size
    weight_error = math.sqrt(
        max(second - weight_mean * weight_mean, 0.0) / paths
    )
    mean_errors: list[float] = []
    for time_report in report["unconditional"]["times"]:
        for direction in ("long", "short"):
            mean_errors.extend(
                float(item["weighted_P_path_standard_error"])
                for item in time_report["directions"][direction]
            )
    radii: list[float] = []
    local_sizes: list[float] = []
    queries = 0
    screened = 0
    for time_report in conditional["times"]:
        audit = time_report["audit"]
        radii.append(float(audit["neighbor_radius"]["mean"]))
        local_sizes.append(float(audit["local_effective_sample_size"]["mean"]))
        queries += int(audit["queries"])
        screened += int(
            time_report["frozen_candidate_region_audit"][
                "reference_screened_candidates"
            ]
        )
    return {
        "paths": paths,
        "weight_mean": weight_mean,
        "weight_mean_standard_error": weight_error,
        "effective_sample_size": effective_size,
        "mean_standard_error": float(np.median(mean_errors)),
        "conditional_radius": float(np.mean(radii)),
        "conditional_effective_sample_size": float(np.mean(local_sizes)),
        "candidate_rate": screened / max(queries, 1),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"))
    parser.add_argument("--frequency", choices=("daily", "weekly"))
    parser.add_argument("--paths", type=int, default=1_000)
    parser.add_argument(
        "--horizons",
        nargs="+",
        choices=("expiry", "incremental"),
        default=("expiry",),
    )
    parser.add_argument(
        "--hedges", nargs="+", choices=("delta", "raw"), default=("delta",)
    )
    parser.add_argument(
        "--costs", nargs="+", type=float, default=(1.0e-4,)
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--baskets", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--cache-root", default=_DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", default=_DEFAULT_REPORT_ROOT)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.aggregate_only:
        write_aggregate_report(Path(args.report_root))
        return
    if args.worker:
        if args.generator is None or args.frequency is None:
            raise ValueError("worker mode requires --generator and --frequency")
        invalid_costs = set(args.costs) - set(REPORTED_COST_RATES)
        if invalid_costs:
            raise ValueError(f"unsupported cost rates {sorted(invalid_costs)}")
        device = torch.device(
            "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        run_worker(
            generator=args.generator,
            frequency=args.frequency,
            paths=args.paths,
            horizons=tuple(args.horizons),
            hedges=tuple(args.hedges),
            costs=tuple(args.costs),
            cache_root=Path(args.cache_root),
            report_root=Path(args.report_root),
            seed=args.seed,
            block_size=args.block_size,
            run_conditional=args.conditional,
            run_baskets=args.baskets,
            device=device,
        )
        return
    run_required_sequence(args)


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(f"required-run command completed in {time.monotonic() - started:.1f}s")
