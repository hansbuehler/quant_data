"""Export fixed one-day return books for repeated generator hedge objectives."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    build_generated_daily_scenarios,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    generate_gromm2_daily_scenarios,
    load_gromm2_market_state,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    load_rolling_gromm2,
)


def _reference_quantities(reference_run: Path) -> tuple[tuple[str, ...], np.ndarray]:
    source = json.loads(reference_run.read_text(encoding="utf-8"))
    labels = tuple(target.label for target in daily_digital_targets(maturity_business_days=1))
    quantities = np.asarray(source["target_quantities"], dtype=np.float64)
    if tuple(source["target_labels"]) != labels or quantities.shape != (len(labels),):
        raise ValueError("reference one-day target catalog changed")
    return labels, quantities


def _diffusion_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        if (
            metadata["schema"] != "diffusion-generated-scenarios-v1"
            or metadata["origin_date"] != origin_date.isoformat()
            or metadata["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
            or metadata["path_count"] != 10_000
        ):
            raise ValueError(f"diffusion scenario identity changed: {source_path}")
        log_returns = np.asarray(source["forward_log_returns"], dtype=np.float64)
        surfaces = np.asarray(source["next_log_implied_volatilities"], dtype=np.float64)
    targets = daily_digital_targets(maturity_business_days=1)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    grid = GenerativeSurfaceGrid.short_end_nms()
    scenarios = tuple(
        build_generated_daily_scenarios(
            sample,
            forward_log_returns=log_returns,
            next_log_implied_volatilities=surfaces,
            grid=grid,
        )
        for sample in samples
    )
    instruments = scenarios[0].instrument_returns
    if any(not np.array_equal(item.instrument_returns, instruments) for item in scenarios[1:]):
        raise RuntimeError(f"diffusion targets do not share one hedge book: {origin_date}")
    targets_returns = np.column_stack(tuple(item.claim_returns for item in scenarios))
    provenance = {
        "source_scenario_sha256": file_sha256(source_path),
        "fit_checkpoint_sha256": metadata["fit_checkpoint_sha256"],
        "configuration": metadata["configuration"],
    }
    return instruments, targets_returns, provenance


def _gromm_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    artifact_root: Path,
    measure: GrommMeasure,
    path_count: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    artifacts = load_rolling_gromm2(
        artifact_root,
        as_of_date=origin_date,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
    )
    state = load_gromm2_market_state(artifacts)
    config = GrommConfig(
        measure=measure,
        path_count=path_count,
        batch_size=min(1_000, path_count),
    )
    targets = daily_digital_targets(maturity_business_days=1)
    generated = generate_gromm2_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=targets[0],
        market_state=state,
        config=config,
    )
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    if any(sample.hedge_quotes != generated.sample.hedge_quotes for sample in samples):
        raise RuntimeError(f"GROMM targets do not share one hedge book: {origin_date}")
    bridge = (
        generated.sample.transition.discount_factor
        * generated.sample.transition.forward_over_spot
    )
    terminal_spot = 1.0 + generated.instrument_returns[:, 0] / bridge
    targets_returns = np.column_stack(
        tuple(
            sample.target_discount_factor
            * (
                terminal_spot > sample.target_strike_over_forward
                if target.right is OptionRight.CALL
                else terminal_spot <= sample.target_strike_over_forward
            )
            - sample.market_price
            for target, sample in zip(targets, samples, strict=True)
        )
    )
    if not np.array_equal(targets_returns[:, 0], generated.claim_returns):
        raise RuntimeError(f"GROMM sampled claim changed: {origin_date}")
    provenance = {
        "source_fit_manifest_sha256": file_sha256(artifacts.paths.manifest),
        "simulation_config": {
            "measure": measure.value,
            "path_count": config.path_count,
            "batch_size": config.batch_size,
            "dlv_substeps": config.dlv_substeps,
            "seed": config.seed,
        },
        "physical_selected_model": state.physical_selected_model,
        "physical_accepted": state.physical_accepted,
    }
    return generated.instrument_returns, targets_returns, provenance


def export_scenarios(
    *,
    model: str,
    source_root: Path,
    output_directory: Path,
    reference_run: Path,
    start: date,
    end: date,
    stop_after: int | None = None,
) -> None:
    """Save one authenticated return tensor per date, reused by every objective."""
    panel = PublicHedgingPanel(default_public_candidate_path(), validate=False)
    labels, quantities = _reference_quantities(reference_run)
    if json.loads(reference_run.read_text(encoding="utf-8"))["dataset_manifest_sha256"] != panel.dataset_manifest_sha256:
        raise ValueError("reference target notionals use another dataset")
    origins = tuple(origin for origin in panel.dates if start <= origin <= end)
    if stop_after is not None:
        origins = origins[:stop_after]
    if not origins:
        raise ValueError("no origin dates selected")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    for index, origin_date in enumerate(origins, start=1):
        destination = output_directory / f"{origin_date.isoformat()}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                existing = json.loads(str(saved["metadata_json"].item()))
                if (
                    existing["schema"] != "hedging-generated-return-book-v1"
                    or existing["model"] != model
                    or existing["origin_date"] != origin_date.isoformat()
                    or existing["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
                    or existing["reference_run_sha256"] != file_sha256(reference_run)
                    or existing["target_labels"] != list(labels)
                    or existing["target_quantities"] != quantities.tolist()
                ):
                    raise RuntimeError(f"existing return book has another identity: {destination}")
                source_path = (
                    source_root / origin_date.isoformat() / "manifest.json"
                    if model.startswith("gromm-")
                    else source_root / f"{origin_date.isoformat()}.npz"
                )
                hash_field = (
                    "source_fit_manifest_sha256"
                    if model.startswith("gromm-")
                    else "source_scenario_sha256"
                )
                if existing["provenance"][hash_field] != file_sha256(source_path):
                    raise RuntimeError(f"source model changed after return export: {origin_date}")
            print(f"return book {index}/{len(origins)} {origin_date} reused", flush=True)
            continue
        if model.startswith("gromm-"):
            measure = GrommMeasure(model.removeprefix("gromm-"))
            instruments, targets_returns, provenance = _gromm_book(
                panel,
                origin_date=origin_date,
                artifact_root=source_root,
                measure=measure,
            )
        elif model in {"diffusion-base", "diffusion-ft"}:
            source_path = source_root / f"{origin_date.isoformat()}.npz"
            instruments, targets_returns, provenance = _diffusion_book(
                panel,
                origin_date=origin_date,
                source_path=source_path,
            )
        else:
            raise ValueError(f"unknown generator: {model}")
        if (
            instruments.shape[0] != 10_000
            or targets_returns.shape != (len(instruments), len(labels))
            or not np.all(np.isfinite(instruments))
            or not np.all(np.isfinite(targets_returns))
        ):
            raise ValueError(f"invalid generated return book: {origin_date}")
        metadata = {
            "schema": "hedging-generated-return-book-v1",
            "model": model,
            "origin_date": origin_date.isoformat(),
            "dataset_manifest_sha256": panel.dataset_manifest_sha256,
            "reference_run_sha256": file_sha256(reference_run),
            "target_labels": list(labels),
            "target_quantities": quantities.tolist(),
            "provenance": provenance,
        }
        with tempfile.NamedTemporaryFile(
            dir=output_directory,
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(
                    handle,
                    metadata_json=json.dumps(metadata, sort_keys=True),
                    instrument_returns=instruments,
                    target_returns=targets_returns,
                )
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(destination)
        elapsed = monotonic() - started
        eta = elapsed * (len(origins) - index) / index
        print(
            f"return book {index}/{len(origins)} {origin_date} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--stop-after", type=int)
    export_scenarios(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
