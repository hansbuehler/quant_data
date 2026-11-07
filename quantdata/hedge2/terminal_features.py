"""Explicit causal remapping between successive rolling terminal-state coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.state import DailyState


@dataclass(frozen=True, kw_only=True)
class TerminalFeatureMapping:
    matrix: np.ndarray
    offset: np.ndarray
    elapsed_step_offset: int = 0
    old_subspace_projection_residual: float = 0.

    def __post_init__(self) -> None:
        if (self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]
                or self.offset.shape != (len(self.matrix),) or not np.isfinite(self.matrix).all()
                or not np.isfinite(self.offset).all() or self.elapsed_step_offset < 0
                or not np.isfinite(self.old_subspace_projection_residual)):
            raise ValueError("Terminal feature remapping needs finite square coordinates and a causal time offset")


def _native_maps(model: FittedDynamics) -> tuple[np.ndarray, np.ndarray]:
    outside = model.spot_state_loading - model.directions @ (model.directions.T @ model.spot_state_loading)
    norm = float(np.linalg.norm(outside))
    outside = outside / norm if norm >= 1e-14 else np.zeros_like(outside)
    reconstruction = np.c_[model.directions * model.daily_factor_scales, outside]
    projection = np.c_[model.directions / model.daily_factor_scales, outside]
    return reconstruction, projection


def rolling_feature_mapping(*, previous_state: DailyState, previous_model: FittedDynamics,
                            current_state: DailyState, current_model: FittedDynamics,
                            previous_root_spot: float, current_root_spot: float,
                            elapsed_step_offset: int = 0) -> TerminalFeatureMapping:
    """Map NEW forecast features to OLD features on the new model's native-state span.

    Recovered DLV labels must retain the same floating coordinate geometry.
    The old PCA span need not equal the new one: its projection residual is
    recorded, not silently called an exact transformation on arbitrary
    native states. The mapping is exact for new-law states reachable from
    the current root through its retained PCs and full spot-loading axis.
    Use a positive time offset when restarting the SAME episode; use zero
    for a new inception. Observation/reset flags retain their known meaning.
    """
    if (previous_state.day != previous_model.as_of_date or current_state.day != current_model.as_of_date
            or previous_state.day >= current_state.day or previous_state.config != current_state.config
            or previous_model.directions.shape != current_model.directions.shape
            or previous_state.native.shape != current_state.native.shape
            or min(previous_root_spot, current_root_spot) <= 0
            or not np.isfinite([previous_root_spot, current_root_spot]).all()):
        raise ValueError("Rolling policy features need earlier compatible native geometry and positive root spots")
    current_reconstruction, _ = _native_maps(current_model)
    previous_reconstruction, previous_projection = _native_maps(previous_model)
    rank = previous_model.directions.shape[1]
    matrix, offset = np.eye(rank + 4), np.zeros(rank + 4)
    matrix[1:rank+2, 1:rank+2] = previous_projection.T @ current_reconstruction
    offset[0] = np.log(current_root_spot / previous_root_spot)
    offset[1:rank+2] = (current_state.native - previous_state.native) @ previous_projection
    represented = current_reconstruction @ np.linalg.lstsq(current_reconstruction, previous_reconstruction, rcond=None)[0]
    residual = np.linalg.norm(previous_reconstruction - represented) / max(np.linalg.norm(previous_reconstruction), 1e-30)
    return TerminalFeatureMapping(matrix=matrix, offset=offset, elapsed_step_offset=elapsed_step_offset,
        old_subspace_projection_residual=float(residual))
