"""Access the retained daily 25-by-200 SDLV fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = (
    PACKAGE_DIRECTORY
    / "data"
    / "spx-2022-2025-daily25-shared200-fixed5-market-v1.npz"
)


@dataclass(frozen=True)
class FitArtifact:
    """Paths and geometry for one retained fitted state representation."""

    label: str
    directory: Path
    rank: int | None

    @property
    def checkpoint(self) -> Path:
        return self.directory / "checkpoint.npz"

    @property
    def report(self) -> Path:
        return self.directory / "report.json"


FITS = {
    "full": FitArtifact(
        label="full",
        directory=PACKAGE_DIRECTORY / "artifacts" / "direct-continuation-1000-v1",
        rank=None,
    ),
    "affine-100": FitArtifact(
        label="affine-100",
        directory=(
            PACKAGE_DIRECTORY / "artifacts" / "affine-100-continuation-1000-v1"
        ),
        rank=100,
    ),
    "affine-200": FitArtifact(
        label="affine-200",
        directory=(
            PACKAGE_DIRECTORY / "artifacts" / "affine-200-continuation-1000-v1"
        ),
        rank=200,
    ),
    "affine-400": FitArtifact(
        label="affine-400",
        directory=(
            PACKAGE_DIRECTORY / "artifacts" / "affine-400-continuation-1000-v1"
        ),
        rank=400,
    ),
}


def load_raw_grids(label: str) -> NDArray[np.float32]:
    """Load all 891 fitted raw DLV grids with shape ``(891, 25, 200)``."""
    artifact = FITS[label]
    with np.load(artifact.checkpoint, allow_pickle=False) as checkpoint:
        schema = str(checkpoint["schema"].item())
        mode = str(checkpoint["mode"].item())
        rank = int(checkpoint["rank"].item())
        if schema != "daily25-jax-end-to-end-v1":
            raise ValueError(f"Unexpected checkpoint schema {schema!r}")
        if artifact.rank is None:
            if mode != "direct" or rank != 0:
                raise ValueError(f"Unexpected full-fit architecture {(mode, rank)!r}")
            grids = np.asarray(checkpoint["raw"], dtype=np.float32)
        else:
            if mode != "affine" or rank != artifact.rank:
                raise ValueError(f"Unexpected affine architecture {(mode, rank)!r}")
            grids = np.asarray(
                checkpoint["mean"]
                + np.einsum(
                    "dr,rij->dij",
                    checkpoint["codes"],
                    checkpoint["loadings"],
                    optimize=True,
                ),
                dtype=np.float32,
            )
    if grids.shape != (891, 25, 200):
        raise ValueError(f"Unexpected fitted grid shape {grids.shape!r}")
    if not np.all(np.isfinite(grids)):
        raise ValueError("Fitted grids contain non-finite values")
    return grids


def load_raw_grid(label: str, date_index: int) -> NDArray[np.float32]:
    """Load one fitted raw DLV grid with shape ``(25, 200)``."""
    return load_raw_grids(label)[date_index]
