"""Merge complete daily date shards into one authenticated full-range run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DAILY_EVALUATION_SCHEMA,
    DailyEvaluationSpec,
    DailyEvaluationSummary,
    _atomic_write_json,
    _write_daily_record,
    daily_numerical_diagnostics,
    eligible_daily_origins,
    read_daily_record,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    authenticate_completed_daily_results,
    summarize_completed_daily_run,
)

DAILY_SHARD_MERGE_SCHEMA = "hedging-daily-shard-merge-v1"


def merge_daily_shards(
    panel: PublicHedgingPanel,
    *,
    destination: Path,
    shards: tuple[Path, ...],
    require_aggregates: bool = True,
) -> DailyEvaluationSummary:
    """Validate disjoint complete shards and materialize the full-range records."""
    if not shards:
        raise ValueError("daily shard merge requires at least one shard")
    destination = Path(destination)
    destination_spec = _load_spec(destination / "run.json")
    eligible = eligible_daily_origins(panel, spec=destination_spec)
    eligible_set = set(eligible)
    if not eligible:
        raise ValueError("daily shard destination has no eligible origins")

    existing = _load_destination_records(
        destination,
        spec=destination_spec,
        eligible=eligible_set,
    )
    shard_results: dict[date, tuple[HedgeSmokeResult, ...]] = {}
    shard_evidence: list[dict[str, object]] = []
    for shard in shards:
        directory = Path(shard)
        shard_spec = _load_spec(directory / "run.json")
        _require_compatible_specs(destination_spec, shard_spec)
        shard_eligible = eligible_daily_origins(panel, spec=shard_spec)
        if not shard_eligible or not set(shard_eligible).issubset(eligible_set):
            raise ValueError(f"daily shard lies outside the destination range: {directory}")
        if set(shard_eligible).intersection(shard_results):
            raise ValueError(f"daily shards overlap: {directory}")
        results = _load_complete_shard(
            panel,
            directory=directory,
            spec=shard_spec,
            eligible=shard_eligible,
            require_aggregate=require_aggregates,
        )
        shard_results.update(results)
        shard_evidence.append(
            {
                "directory": str(directory.resolve()),
                "run_identity": shard_spec.run_identity,
                "run_sha256": _file_sha256(directory / "run.json"),
                "summary_sha256": _file_sha256(directory / "summary.json"),
                "results_sha256": (
                    _file_sha256(directory / "results.json")
                    if require_aggregates
                    else None
                ),
                "origin_count": len(shard_eligible),
                "first_origin": shard_eligible[0].isoformat(),
                "last_origin": shard_eligible[-1].isoformat(),
            }
        )

    native_origins = tuple(value for value in eligible if value not in shard_results)
    missing_native = tuple(value for value in native_origins if value not in existing)
    if missing_native:
        raise ValueError(
            "daily shard merge leaves uncovered destination origins: "
            + ", ".join(value.isoformat() for value in missing_native[:5])
        )
    for origin, results in shard_results.items():
        if origin in existing and existing[origin] != results:
            raise ValueError(
                f"daily shard disagrees with an existing destination record: {origin}"
            )
    combined = {**existing, **shard_results}
    if set(combined) != eligible_set:
        raise ValueError("daily shard merge does not exactly cover the destination range")

    for origin in eligible:
        if origin in shard_results and origin not in existing:
            _write_daily_record(
                destination / "records" / f"{origin.isoformat()}.json",
                spec=destination_spec,
                origin_date=origin,
                results=combined[origin],
            )
    summary = _complete_summary(
        spec=destination_spec,
        eligible=eligible,
        records=combined,
    )
    _atomic_write_json(destination / "summary.json", summary.to_dict())
    _atomic_write_json(
        destination / "merge.json",
        {
            "schema": DAILY_SHARD_MERGE_SCHEMA,
            "evaluation_schema": DAILY_EVALUATION_SCHEMA,
            "destination": str(destination.resolve()),
            "destination_run_identity": destination_spec.run_identity,
            "destination_run_sha256": _file_sha256(destination / "run.json"),
            "eligible_origin_count": len(eligible),
            "native_origin_count": len(native_origins),
            "native_record_sha256": {
                origin.isoformat(): _file_sha256(
                    destination / "records" / f"{origin.isoformat()}.json"
                )
                for origin in native_origins
            },
            "shards": shard_evidence,
        },
    )
    return summary


def _load_spec(path: Path) -> DailyEvaluationSpec:
    payload = _read_json_object(path)
    return DailyEvaluationSpec.from_dict(payload)


def _require_compatible_specs(
    destination: DailyEvaluationSpec,
    shard: DailyEvaluationSpec,
) -> None:
    destination_payload = destination.to_dict()
    shard_payload = shard.to_dict()
    for name in ("evaluation_start", "evaluation_end"):
        destination_payload.pop(name)
        shard_payload.pop(name)
    if shard_payload != destination_payload:
        raise ValueError("daily shard model identity differs from the destination")
    if (
        shard.evaluation_start < destination.evaluation_start
        or shard.evaluation_end > destination.evaluation_end
    ):
        raise ValueError("daily shard date range exceeds the destination range")


def _load_destination_records(
    destination: Path,
    *,
    spec: DailyEvaluationSpec,
    eligible: set[date],
) -> dict[date, tuple[HedgeSmokeResult, ...]]:
    records: dict[date, tuple[HedgeSmokeResult, ...]] = {}
    for path in sorted((destination / "records").glob("*.json")):
        try:
            origin = date.fromisoformat(path.stem)
        except ValueError as error:
            raise ValueError(f"daily destination has a malformed record name: {path}") from error
        if origin not in eligible:
            raise ValueError(f"daily destination has an ineligible record: {origin}")
        records[origin] = read_daily_record(path, spec=spec, origin_date=origin)
    return records


def _load_complete_shard(
    panel: PublicHedgingPanel,
    *,
    directory: Path,
    spec: DailyEvaluationSpec,
    eligible: tuple[date, ...],
    require_aggregate: bool,
) -> dict[date, tuple[HedgeSmokeResult, ...]]:
    summary = DailyEvaluationSummary.from_dict(
        _read_json_object(directory / "summary.json")
    )
    expected_origins = tuple(value.isoformat() for value in eligible)
    if (
        summary.run_identity != spec.run_identity
        or summary.eligible_origins != expected_origins
        or summary.completed_origins != expected_origins
    ):
        raise ValueError(f"daily shard is incomplete or has the wrong identity: {directory}")
    paths = tuple(sorted((directory / "records").glob("*.json")))
    if tuple(path.stem for path in paths) != expected_origins:
        raise ValueError(f"daily shard record coverage differs: {directory}")
    records = {
        origin: read_daily_record(
            directory / "records" / f"{origin.isoformat()}.json",
            spec=spec,
            origin_date=origin,
        )
        for origin in eligible
    }
    rebuilt = _complete_summary(spec=spec, eligible=eligible, records=records)
    if (
        rebuilt.run_identity != summary.run_identity
        or rebuilt.eligible_origins != summary.eligible_origins
        or rebuilt.completed_origins != summary.completed_origins
        or rebuilt.result_count != summary.result_count
        or rebuilt.nonconverged_results != summary.nonconverged_results
        or rebuilt.cap_hits != summary.cap_hits
        or rebuilt.unregularized_recession_direction_fits
        != summary.unregularized_recession_direction_fits
    ):
        raise ValueError(f"daily shard summary does not match its records: {directory}")
    if require_aggregate:
        authenticate_completed_daily_results(
            panel,
            results_path=directory / "results.json",
        )
    return records


def _complete_summary(
    *,
    spec: DailyEvaluationSpec,
    eligible: tuple[date, ...],
    records: dict[date, tuple[HedgeSmokeResult, ...]],
) -> DailyEvaluationSummary:
    results = tuple(result for origin in eligible for result in records[origin])
    return DailyEvaluationSummary(
        run_identity=spec.run_identity,
        eligible_origins=tuple(value.isoformat() for value in eligible),
        completed_origins=tuple(value.isoformat() for value in eligible),
        reused_origins=tuple(value.isoformat() for value in eligible),
        result_count=len(results),
        nonconverged_results=sum(not result.converged for result in results),
        cap_hits=sum(result.cap_hits for result in results),
        unregularized_recession_direction_fits=sum(
            int(
                daily_numerical_diagnostics(records[origin])[
                    "unregularized_recession_direction_fits"
                ]
            )
            for origin in eligible
        ),
    )


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Parse a complete shard merge, authenticate it, and write aggregation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(
        arguments.dataset,
        validate=arguments.validate_dataset,
    )
    summary = merge_daily_shards(
        panel,
        destination=arguments.destination,
        shards=tuple(arguments.shard),
    )
    report = summarize_completed_daily_run(
        panel,
        run_directory=arguments.destination,
    )
    results_path = arguments.destination / "results.json"
    report.write_json(results_path)
    authenticate_completed_daily_results(panel, results_path=results_path)
    print(
        json.dumps(
            {
                "destination": str(arguments.destination),
                "run_identity": summary.run_identity,
                "result_count": summary.result_count,
                "results_sha256": _file_sha256(results_path),
                "merge_sha256": _file_sha256(arguments.destination / "merge.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
