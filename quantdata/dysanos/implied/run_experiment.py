"""Run one direct PCA-IV-AR(1) benchmark with blockwise diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig

from .cache import PCAUniverseBlockCache
from .conditional_diagnostics import (
    CompactLocalTailStore,
    run_conditional_local_tail,
)
from .decoder import PCATotalVarianceDecoder
from .diagnostics import (
    DATA_QUALIFICATION,
    DYNAMIC_ARBITRAGE_QUALIFICATION,
    historical_static_arbitrage_diagnostics,
    invariant_surface_diagnostics,
    sampler_fit_report,
    weekly_path_diagnostics,
)
from .experiment import ImpliedPCAExperiment
from .market import load_fitted_pca
from .plotting import (
    write_distribution_fidelity_figure,
    write_iv_eigensurface_figure,
    write_local_candidate_figure,
    write_minority_tail_figure,
    write_pca_reconstruction_figure,
    write_static_violation_figure,
)
from .return_diagnostics import ReturnDiagnosticsAccumulator
from .simulator import ImpliedPCASampler
from .static_arbitrage import StaticArbitrageAccumulator
from .universe import (
    PCAUniverseStreamDiagnostics,
    default_static_strike_mesh,
    finalize_stream_diagnostics,
    iter_pca_universe_blocks,
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


def _slice_return_universe(
    universe: dict[str, np.ndarray], start: int, stop: int
) -> dict[str, np.ndarray]:
    """Select independent paths without copying unrelated cached fields."""

    names = {
        "dC",
        "dS",
        "expiry_dS",
        "option_dS",
        "C",
        "spot",
        "delta",
        "log_strikes",
        "log_path_weight",
    }
    result = {
        name: value[start:stop]
        for name, value in universe.items()
        if name in names
    }
    if "option_expired" in universe:
        result["option_expired"] = universe["option_expired"]
    return result


def _write_generated_figures(
    *,
    output_root: Path,
    filename: str,
    historical_static: dict[str, Any],
    generated: dict[str, Any],
) -> dict[str, str]:
    """Write figures which require the completed streamed diagnostics."""

    paths = {
        "static_violations": output_root / f"{filename}-static-violations.png",
        "minority_tail": output_root / f"{filename}-minority-tail.png",
    }
    write_static_violation_figure(
        historical_static,
        generated["generated_surface_static_arbitrage"],
        paths["static_violations"],
    )
    write_minority_tail_figure(
        generated["return_diagnostics"], paths["minority_tail"]
    )
    local = generated.get("conditional_local_tail", {})
    if local.get("status") == "complete":
        paths["local_candidates"] = (
            output_root / f"{filename}-local-candidates.png"
        )
        write_local_candidate_figure(local, paths["local_candidates"])
    return {name: str(path) for name, path in paths.items()}


def run(
    experiment: ImpliedPCAExperiment,
    *,
    output_root: str | Path = "quantdata/dysanos/implied/reports",
    cache_root: str | Path = (
        "/media/storage/tmp/quantdata/dysanos/implied-pca-universes"
    ),
    invariant_samples: int = 100_000,
    static_mesh_points: int | None = None,
    run_local_tail: bool = True,
    local_tail_device: torch.device | None = None,
    verbose: Context = Context.all,
) -> dict[str, Any]:
    """Fit, simulate, and report one versioned PCA benchmark experiment."""

    output_root = Path(output_root)
    report_path = output_root / f"{experiment.filename}.json"
    report: dict[str, Any] = {
        "status": "loading",
        "benchmark": "PCA-IV-AR(1)",
        "configuration": asdict(experiment),
        "diagnostic_configuration": {
            "invariant_and_weekly_samples": invariant_samples,
            "run_conditional_local_tail": run_local_tail,
            "local_tail_device": (
                "auto" if local_tail_device is None else str(local_tail_device)
            ),
        },
        "primary_path_count": experiment.is_primary_path_count,
        "data_qualification": DATA_QUALIFICATION,
        "dynamic_arbitrage_qualification": DYNAMIC_ARBITRAGE_QUALIFICATION,
    }
    _atomic_json(report_path, report)

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
    if static_mesh_points is not None:
        verify_inp(
            static_mesh_points == experiment.static_mesh_points,
            lambda: (
                "run-time static mesh differs from the versioned experiment: "
                f"{static_mesh_points}/{experiment.static_mesh_points}"
            ),
        )
    static_strikes = default_static_strike_mesh(
        experiment.static_mesh_points
    )
    historical_static, historical_wing_hits = (
        historical_static_arbitrage_diagnostics(
            decoder,
            pca_diagnostics.scores,
            strikes=static_strikes,
            roundoff_multipliers=experiment.roundoff_multipliers,
        )
    )
    invariant = invariant_surface_diagnostics(
        sampler, num_samples=invariant_samples
    )
    weekly = weekly_path_diagnostics(sampler, num_samples=invariant_samples)
    report.update(
        status="fitted",
        pca=pca_diagnostics.report(pca),
        ar1=sampler_fit_report(sampler),
        invariant_surface_distribution=invariant,
        weekly_path_fidelity=weekly,
        historical_PCA_reconstruction_static_arbitrage=historical_static,
        historical_interpolation_wing_hits=asdict(historical_wing_hits),
    )
    eigensurface_path = output_root / f"{experiment.filename}-iv-eigensurfaces.png"
    reconstruction_path = output_root / f"{experiment.filename}-pca-reconstruction.png"
    fidelity_path = output_root / f"{experiment.filename}-distribution-fidelity.png"
    write_iv_eigensurface_figure(pca, pca_diagnostics, eigensurface_path)
    write_pca_reconstruction_figure(pca, pca_diagnostics, reconstruction_path)
    write_distribution_fidelity_figure(invariant, weekly, fidelity_path)
    report["pca"]["iv_eigensurface_figure"] = str(eigensurface_path)
    report["pca"]["reconstruction_figure"] = str(reconstruction_path)
    report["fidelity_figure"] = str(fidelity_path)
    _atomic_json(report_path, report)

    cache = PCAUniverseBlockCache(cache_root, experiment=experiment)
    if cache.is_complete:
        cached = cache.manifest()["diagnostics"]
        if run_local_tail and cached.get("conditional_local_tail", {}).get(
            "status"
        ) != "complete":
            local_store = CompactLocalTailStore(
                cache.directory, experiment=experiment
            )
            if not local_store.is_complete:
                local_store.build(cache.iter_blocks())
            cached["conditional_local_tail"] = run_conditional_local_tail(
                local_store, device=local_tail_device
            )
        cached["figures"] = _write_generated_figures(
            output_root=output_root,
            filename=experiment.filename,
            historical_static=historical_static,
            generated=cached,
        )
        cache.update_diagnostics(cached)
        report.update(status="complete", universe_cache=str(cache.directory), **cached)
        _atomic_json(report_path, report)
        return report

    cache.begin()
    stream_diagnostics = PCAUniverseStreamDiagnostics()
    static_accumulator = StaticArbitrageAccumulator(
        strikes=static_strikes,
        expiries=pca.expiries,
        roundoff_multipliers=experiment.roundoff_multipliers,
        seed=experiment.seed + 911,
    )
    return_accumulators = {
        split: ReturnDiagnosticsAccumulator(
            expiries=pca.expiries.size,
            strikes=experiment.num_traded_nms,
            cost_proportion=experiment.cost_proportion,
        )
        for split in ("training", "validation")
    }
    split_path = experiment.reference_paths
    for block in iter_pca_universe_blocks(
        experiment,
        decoder=decoder,
        sampler=sampler,
        diagnostics=stream_diagnostics,
        static_accumulator=static_accumulator,
        static_strikes=static_strikes,
        verbose=verbose,
    ):
        training_stop = min(block.stop, split_path)
        if block.start < training_stop:
            return_accumulators["training"].update(
                _slice_return_universe(
                    block.universe,
                    0,
                    training_stop - block.start,
                )
            )
        validation_start = max(block.start, split_path)
        if validation_start < block.stop:
            return_accumulators["validation"].update(
                _slice_return_universe(
                    block.universe,
                    validation_start - block.start,
                    block.stop - block.start,
                )
            )
        cache.write(block)
        report.update(
            status="simulating",
            completed_paths=block.stop,
            universe_cache=str(cache.directory),
        )
        _atomic_json(report_path, report)

    path_diagnostics = finalize_stream_diagnostics(stream_diagnostics)
    generated: dict[str, Any] = {
        "sampling_diagnostics": asdict(path_diagnostics),
        "interpolation_wing_hits": asdict(stream_diagnostics.wing_hits),
        "generated_surface_static_arbitrage": static_accumulator.report(),
        "return_diagnostics": {
            split: accumulator.report()
            for split, accumulator in return_accumulators.items()
        },
        "conditional_local_tail": {
            "status": "pending" if run_local_tail else "not-run",
            "state_definition": "log-spot plus 20 PCA surface scores",
            "training_validation_split": "first/second 50% independent paths",
            "neighbors": experiment.neighbors,
            "candidate_neighbors": experiment.candidate_neighbors,
            "query_paths_per_split": experiment.query_paths_per_split,
            "tail_ranks": list(experiment.tail_ranks),
            "signal_thresholds": list(experiment.signal_thresholds),
            "instrument_block_size": experiment.instrument_block_size,
            "note": (
                "The versioned cache contains only the state, return, roundoff, "
                "weight, price, hedge, and instrument descriptors required by "
                "the shared SANOS local-tail valuation path."
            ),
        },
    }
    generated = _json_safe(generated)
    cache.complete(diagnostics=generated)
    if run_local_tail:
        local_store = CompactLocalTailStore(
            cache.directory, experiment=experiment
        )
        local_store.build(cache.iter_blocks())
        generated["conditional_local_tail"] = _json_safe(
            run_conditional_local_tail(
                local_store, device=local_tail_device
            )
        )
    generated["figures"] = _write_generated_figures(
        output_root=output_root,
        filename=experiment.filename,
        historical_static=historical_static,
        generated=generated,
    )
    cache.update_diagnostics(generated)
    report.update(
        status="complete",
        completed_paths=experiment.num_samples,
        universe_cache=str(cache.directory),
        **generated,
    )
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument(
        "--mode",
        choices=("one-step", "one-step-weekly", "path", "weekly"),
        default=None,
        help="Deprecated legacy geometry alias",
    )
    parser.add_argument(
        "--option-return-horizon",
        choices=("expiry", "one-day"),
        default=None,
        help="Deprecated legacy return-horizon alias",
    )
    parser.add_argument(
        "--trading-frequency", choices=("daily", "weekly"), default="daily"
    )
    parser.add_argument(
        "--return-horizon", choices=("expiry", "incremental"), default="expiry"
    )
    parser.add_argument("--num-trading-times", type=int, default=10)
    parser.add_argument("--num-alpha", choices=("5", "all"), default="5")
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument(
        "--num-steps", type=int, default=None, help="Deprecated legacy path length"
    )
    parser.add_argument("--direct-p", action="store_true")
    parser.add_argument("--invariant-samples", type=int, default=100_000)
    parser.add_argument("--static-mesh-points", type=int, default=201)
    parser.add_argument("--skip-local-tail", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-root", default="quantdata/dysanos/implied/reports"
    )
    parser.add_argument(
        "--cache-root",
        default="/media/storage/tmp/quantdata/dysanos/implied-pca-universes",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    importance = None if args.direct_p else GaussianSpotImportanceConfig()
    experiment = ImpliedPCAExperiment(
        num_samples=args.num_samples,
        mode=args.mode,
        option_return_horizon=args.option_return_horizon,
        trading_frequency=args.trading_frequency,
        return_horizon=args.return_horizon,
        num_trading_times=args.num_trading_times,
        num_alpha=None if args.num_alpha == "all" else 5,
        num_steps=args.num_steps,
        seed=args.seed,
        block_size=args.block_size,
        importance_sampling=importance,
        static_mesh_points=args.static_mesh_points,
    )
    report = run(
        experiment,
        output_root=args.output_root,
        cache_root=args.cache_root,
        invariant_samples=args.invariant_samples,
        run_local_tail=not args.skip_local_tail,
        local_tail_device=(
            None if args.device == "auto" else torch.device(args.device)
        ),
    )
    print(json.dumps({"status": report["status"], "configuration": report["configuration"]}, indent=2))


if __name__ == "__main__":
    main()
