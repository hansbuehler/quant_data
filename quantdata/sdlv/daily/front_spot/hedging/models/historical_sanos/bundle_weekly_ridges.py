"""Bundle authenticated objective/risk-specific weekly ridge selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_ridge import (
    WeeklyRidgeBundle,
    WeeklyRidgeSelection,
    write_weekly_ridge_bundle,
)


def main() -> None:
    """Authenticate individual selections and write one portable bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument(
        "--selection-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--quant-review", type=Path, required=True)
    parser.add_argument("--quant-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.selection) != len(arguments.selection_sha256):
        parser.error("selection paths and SHA-256 values must align")
    if _file_sha256(arguments.quant_review) != arguments.quant_review_sha256:
        raise RuntimeError("weekly ridge quant-review SHA-256 mismatch")

    selections = tuple(
        _load_selection(path, expected_sha256=expected_sha256)
        for path, expected_sha256 in zip(
            arguments.selection,
            arguments.selection_sha256,
            strict=True,
        )
    )
    bundle = WeeklyRidgeBundle(
        dataset_manifest_sha256=selections[0].dataset_manifest_sha256,
        target_notional_artifact_sha256=(
            selections[0].target_notional_artifact_sha256
        ),
        quant_review_sha256=arguments.quant_review_sha256,
        selections=selections,
    )
    write_weekly_ridge_bundle(bundle, output=arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "file_sha256": _file_sha256(arguments.output),
                "bundle_sha256": bundle.sha256,
                "quant_review_sha256": bundle.quant_review_sha256,
                "scope_count": len(bundle.selections),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_selection(path: Path, *, expected_sha256: str) -> WeeklyRidgeSelection:
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"weekly ridge selection SHA-256 mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"weekly ridge selection must contain an object: {path}")
    declared_sha256 = payload.pop("selection_sha256", None)
    selection = WeeklyRidgeSelection.from_dict(payload)
    if declared_sha256 != selection.sha256:
        raise ValueError(f"weekly ridge selection digest is invalid: {path}")
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"weekly ridge selection changed while loading: {path}")
    return selection


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
