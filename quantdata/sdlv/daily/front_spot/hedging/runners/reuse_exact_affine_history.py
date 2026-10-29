"""Reuse an affine GROMM fit only when every dated model input is identical."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
from filelock import FileLock

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256


def _same_dataset_values(source: Path, target: Path, *, ignored_attributes: frozenset[str] = frozenset()) -> int:
    with h5py.File(source, "r") as older, h5py.File(target, "r") as newer:
        if set(older) != set(newer):
            raise ValueError(f"HDF5 dataset catalog changed: {source} versus {target}")
        for name in older:
            left = older[name]
            right = newer[name]
            if (
                not isinstance(left, h5py.Dataset)
                or not isinstance(right, h5py.Dataset)
                or left.dtype != right.dtype
                or left.shape != right.shape
                or not np.array_equal(left[()], right[()])
            ):
                raise ValueError(f"HDF5 model input changed: {name}")
        old_attributes = set(older.attrs) - ignored_attributes
        new_attributes = set(newer.attrs) - ignored_attributes
        if old_attributes != new_attributes or any(
            not np.array_equal(older.attrs[name], newer.attrs[name])
            for name in old_attributes
        ):
            raise ValueError(f"HDF5 model metadata changed: {source} versus {target}")
        return len(older)


def _normalized_calibration(value: dict) -> dict:
    normalized = dict(value)
    normalized.setdefault("plateau_review_epochs", 0)
    return normalized


def _link_verified(source: Path, target: Path, expected_sha256: str) -> None:
    if file_sha256(source) != expected_sha256:
        raise ValueError(f"source affine artifact changed: {source}")
    if target.exists():
        if file_sha256(target) != expected_sha256:
            raise ValueError(f"target affine artifact conflicts: {target}")
        return
    temporary = target.with_suffix(target.suffix + ".reuse.tmp")
    os.link(source, temporary)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def reuse_one_date(*, source_root: Path, target_root: Path, as_of_date: date) -> bool:
    """Authenticate exact old/current books and hard-link an accepted affine fit."""
    source = source_root / as_of_date.isoformat()
    target = target_root / as_of_date.isoformat()
    lock = FileLock(str(target / ".exact-affine-reuse.lock"))
    with lock:
        old_manifest_path = source / "manifest.json"
        new_manifest_path = target / "manifest.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("status") != "complete" or new_manifest.get("status") != "prepared":
            raise ValueError(f"GROMM fit is not complete/prepared on {as_of_date}")
        if (
            old_manifest.get("as_of_date") != as_of_date.isoformat()
            or new_manifest.get("as_of_date") != as_of_date.isoformat()
            or old_manifest.get("training_origin_dates") != new_manifest.get("training_origin_dates")
            or old_manifest.get("training_endpoint_dates") != new_manifest.get("training_endpoint_dates")
        ):
            raise ValueError(f"GROMM rolling dates changed on {as_of_date}")
        for field in ("geometry", "window_transitions", "minimum_inside_spread_percent"):
            if old_manifest["config"][field] != new_manifest["config"][field]:
                raise ValueError(f"GROMM {field} changed on {as_of_date}")
        if _normalized_calibration(old_manifest["config"]["calibration"]) != _normalized_calibration(new_manifest["config"]["calibration"]):
            raise ValueError(f"GROMM calibration specification changed on {as_of_date}")
        if old_manifest["percent_inside_spread"] < new_manifest["config"]["minimum_inside_spread_percent"]:
            raise ValueError(f"GROMM affine fit fails current market gate on {as_of_date}")
        if any(
            file_sha256(source / filename) != old_manifest[sha_field]
            or file_sha256(target / filename) != new_manifest[sha_field]
            for filename, sha_field in (
                ("source-history.h5", "source_history_sha256"),
                ("prepared-books.h5", "prepared_books_sha256"),
            )
        ):
            raise ValueError(f"GROMM prepared input provenance changed on {as_of_date}")
        history_count = _same_dataset_values(
            source / "source-history.h5",
            target / "source-history.h5",
        )
        book_count = _same_dataset_values(
            source / "prepared-books.h5",
            target / "prepared-books.h5",
            ignored_attributes=frozenset({"source_history"}),
        )
        affine_path = source / "affine-200-history.h5"
        affine_sha256 = old_manifest["affine_history_sha256"]
        if file_sha256(affine_path) != affine_sha256:
            raise ValueError(f"GROMM affine history changed on {as_of_date}")
        with h5py.File(affine_path, "r") as affine, h5py.File(target / "prepared-books.h5", "r") as books:
            if (
                affine.attrs.get("schema") != "gromm2-affine-dlv-history-v1"
                or not np.array_equal(affine["dates"][()], books["dates"][()])
                or not np.array_equal(
                    affine["nms_nodes"][()],
                    np.asarray(books["nms_nodes"][()], dtype=np.float32),
                )
                or float(affine.attrs["percent_inside_spread"]) != old_manifest["percent_inside_spread"]
                or _normalized_calibration(json.loads(str(affine.attrs["calibration_config"])))
                != _normalized_calibration(old_manifest["affine_calibration_config"])
            ):
                raise ValueError(f"GROMM affine history geometry or fit changed on {as_of_date}")
        checkpoint_path = source / "affine-200-checkpoint.npz"
        checkpoint_sha256 = file_sha256(checkpoint_path) if checkpoint_path.exists() else None
        if checkpoint_sha256 is not None:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                if (
                    str(checkpoint["schema"].item()) != "gromm2-affine-jax-checkpoint-v1"
                    or int(checkpoint["completed_epochs"]) < old_manifest["calibration_epochs_completed"]
                    or checkpoint["parameter_codes"].shape[0] != len(new_manifest["training_origin_dates"]) + 1
                ):
                    raise ValueError(f"GROMM affine checkpoint changed on {as_of_date}")
        provenance = {
            "schema": "gromm-exact-input-affine-reuse-v1",
            "source_manifest_sha256": file_sha256(old_manifest_path),
            "source_dataset_manifest_sha256": old_manifest["dataset_manifest_sha256"],
            "source_affine_warm_start_date": old_manifest.get("warm_start_date"),
            "source_affine_warm_start_checkpoint_sha256": old_manifest.get("warm_start_checkpoint_sha256"),
            "source_history_sha256": old_manifest["source_history_sha256"],
            "source_prepared_books_sha256": old_manifest["prepared_books_sha256"],
            "target_history_sha256": new_manifest["source_history_sha256"],
            "target_prepared_books_sha256": new_manifest["prepared_books_sha256"],
            "history_dataset_count": history_count,
            "prepared_book_dataset_count": book_count,
            "affine_history_sha256": affine_sha256,
            "affine_checkpoint_sha256": checkpoint_sha256,
        }
        if "exact_input_affine_reuse" in new_manifest:
            if new_manifest["exact_input_affine_reuse"] != provenance:
                raise ValueError(f"GROMM affine reuse provenance changed on {as_of_date}")
            return True
        _link_verified(affine_path, target / affine_path.name, affine_sha256)
        if checkpoint_sha256 is not None:
            _link_verified(checkpoint_path, target / checkpoint_path.name, checkpoint_sha256)
        new_manifest["exact_input_affine_reuse"] = provenance
        temporary = new_manifest_path.with_suffix(".json.reuse.tmp")
        temporary.write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(new_manifest_path)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--stop-after", type=int)
    arguments = parser.parse_args()
    dates = tuple(
        date.fromisoformat(path.name)
        for path in sorted(arguments.target_root.iterdir())
        if path.is_dir()
        and path.name.startswith("20")
        and arguments.start <= date.fromisoformat(path.name) <= arguments.end
    )
    if arguments.stop_after is not None:
        dates = dates[:arguments.stop_after]
    started = monotonic()
    for index, as_of_date in enumerate(dates, start=1):
        reused = reuse_one_date(
            source_root=arguments.source_root,
            target_root=arguments.target_root,
            as_of_date=as_of_date,
        )
        elapsed = monotonic() - started
        eta = elapsed / index * (len(dates) - index)
        print(
            f"exact affine reuse {index}/{len(dates)} {as_of_date} "
            f"existing={reused} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
