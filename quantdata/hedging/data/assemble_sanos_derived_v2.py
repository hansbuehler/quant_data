"""Assemble the fitted SANOS-to-DYSANOS grid into a standalone v2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from quantdata.hedging.data.build_sanos_derived_grid import BUSINESS_DTES, GEOMETRY

DERIVED_FILE = "sanos_dysanos_grid.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(*, source: Path, day_directory: Path, destination: Path) -> dict:
    """Copy the v1 release and add one validated derived daily state file."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    source_manifest = json.loads((source / "manifest.json").read_text())
    observation_dates = pl.read_parquet(source / "observations.parquet")["date"].to_list()
    rows: list[dict[str, object]] = []
    for observation_date in observation_dates:
        day_path = day_directory / f"{observation_date.isoformat()}.npz"
        if not day_path.is_file():
            raise FileNotFoundError(f"Missing derived fit for {observation_date}: {day_path}")
        with np.load(day_path, allow_pickle=False) as stored:
            saved_date = date.fromisoformat(str(stored["date"].item()))
            if saved_date != observation_date:
                raise ValueError(f"Derived date mismatch: {saved_date} != {observation_date}")
            row: dict[str, object] = {"date": observation_date}
            for name in (
                "support_strikes",
                "source_call_prices",
                "fitted_call_prices",
                "decoded_call_prices",
                "densities",
                "dlvsqs",
                "forward_vols",
                "raw_u",
            ):
                values = np.asarray(stored[name], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{saved_date}: {name} contains non-finite values")
                row[name] = values.tolist()
            for name in (
                "eta_front",
                "eta_term",
                "fit_max_abs_core",
                "fit_mean_abs_core",
                "fit_max_weighted_error",
                "decode_max_abs_core",
                "decode_max_weighted_error",
            ):
                row[name] = float(stored[name].item())
            row["variance_increments_repaired"] = int(
                stored["variance_increments_repaired"].item()
            )
            if np.shape(row["support_strikes"]) != (202,):
                raise ValueError(f"{saved_date}: support must have 202 strikes")
            if np.shape(row["source_call_prices"]) != (9, 200):
                raise ValueError(f"{saved_date}: source prices must have shape (9,200)")
            if np.shape(row["raw_u"]) != (GEOMETRY.native_state_size,):
                raise ValueError(f"{saved_date}: raw state has incorrect length")
            rows.append(row)

    destination.mkdir(parents=True)
    for filename in source_manifest["checksums"]:
        shutil.copy2(source / filename, destination / filename)
    frame = pl.DataFrame(rows)
    frame.write_parquet(destination / DERIVED_FILE, compression="zstd")

    schema = "hedging-public-candidate-v2"
    dataset_name = destination.name
    config = json.loads((destination / "config.json").read_text())
    config.update(
        schema=schema,
        dataset_name=dataset_name,
        legacy_dlv_fit_label=config["fit_label"],
        fit_label="sanos-derived-shared-9x200",
        sanos_derived_grid_business_dtes=list(BUSINESS_DTES),
        sanos_derived_grid_nms_range=[-4.0, 3.0],
        sanos_derived_grid_inner_strikes=200,
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    geometry = json.loads((destination / "geometry.json").read_text())
    geometry.update(
        schema=schema,
        sanos_derived_grid_shape=[9, 200],
        sanos_derived_state_length=GEOMETRY.native_state_size,
        sanos_derived_etas="front and later-term multipliers",
        sanos_derived_grid_shared_within_day=True,
        sanos_derived_grid_floats_between_days=True,
    )
    (destination / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    with (destination / "README.md").open("a") as stream:
        stream.write(
            "\n## SANOS-derived DYSANOS grid (v2)\n\n"
            "The daily 9-by-200 shared strike grid uses business DTEs "
            "1, 2, 3, 4, 5, 10, 15, 20, and 25 and endpoint NMS range "
            "(-4, 3). The same strike support is used at every expiry on a day; "
            "it floats between days. `sanos_dysanos_grid.parquet` stores SANOS "
            "source call prices, SANOS-fitter call prices, decoded DYSANOS call "
            "prices, fitted densities, DLV squares, and the flattened native "
            "state `raw_u`. All prices use `C/(DF*F)`; none is a market midpoint. "
            "The fitter's synthetic corridors are numerical weights, not bid/ask "
            "quotes: observations outside expiry-specific NMS (-3.5, 2.5) "
            "receive 10% of the core weight. The two eta coordinates match the "
            "front and final SANOS smoothing scales approximately; the LP then "
            "fits the complete source-price grid under SANOS martingale and "
            "DLV constraints. The three price arrays and their errors remain "
            "separate because the transformation need not be exact. This is a "
            "daily SANOS-derived coordinate conversion, not a rolling "
            "DYSANOS AR(1) fit. The Greek, parametric, statistical, and "
            "normal-statistical hedge methods applied to these states are "
            "all SANOS methods.\n"
            "The v1 `dlv_geometry.parquet`, `dlv_state.parquet`, and "
            "`dlv_prices_over_discounted_forward` fields are retained only "
            "for backward compatibility; they describe the earlier affine-200 "
            "fit and are not the new SANOS-derived DYSANOS state.\n"
        )
    with (destination / "DATA_DICTIONARY.md").open("a") as stream:
        stream.write(
            "\n## `sanos_dysanos_grid.parquet` (v2)\n\n"
            "One row per date. `support_strikes` is 202 forward-normalized "
            "nodes, including two exterior boundaries. `source_call_prices`, "
            "`fitted_call_prices`, and `decoded_call_prices` are [9,200] arrays "
            "at the interior nodes. `densities` is [9,202], `dlvsqs` is [9,200], "
            "`forward_vols` is [9], and `raw_u` is the 1811-coordinate "
            "expiry-major DYSANOS state with two trailing eta coordinates. "
            "`eta_front` and `eta_term` are the bounded eta values. "
            "`fit_max_abs_core`, `fit_mean_abs_core`, and "
            "`fit_max_weighted_error` compare fitter calls with SANOS calls; "
            "`decode_max_abs_core` and `decode_max_weighted_error` compare "
            "decoded calls with SANOS calls. `variance_increments_repaired` "
            "counts forward-variance intervals raised to the positive minimum "
            "required by the DYSANOS parameter bounds.\n"
        )

    manifest = source_manifest.copy()
    manifest.update(
        schema=schema,
        dataset_name=dataset_name,
        source_model="retained daily smooth SANOS plus SANOS-fitter-derived shared-grid DYSANOS",
        source_dataset_name=source_manifest["dataset_name"],
        source_manifest_sha256=_sha256(source / "manifest.json"),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        generator_worktree_dirty=True,
    )
    manifest["capabilities"] = {
        **source_manifest["capabilities"],
        "sanos_derived_dysanos_state": True,
    }
    builder_path = Path(__file__).with_name("build_sanos_derived_grid.py")
    manifest["generator_code_sha256"] = {
        **source_manifest["generator_code_sha256"],
        "quantdata/hedging/data/build_sanos_derived_grid.py": _sha256(builder_path),
        "quantdata/hedging/data/assemble_sanos_derived_v2.py": _sha256(Path(__file__)),
    }
    manifest["checksums"] = {
        filename: _sha256(destination / filename)
        for filename in (*source_manifest["checksums"], DERIVED_FILE)
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--days", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = assemble(
        source=arguments.source,
        day_directory=arguments.days,
        destination=arguments.destination,
    )
    print(json.dumps({"dataset_name": manifest["dataset_name"], "num_dates": manifest["num_dates"]}))


if __name__ == "__main__":
    main()
