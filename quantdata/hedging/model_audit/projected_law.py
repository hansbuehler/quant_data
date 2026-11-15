"""Native simulation of a retained affine/tangent factor law with full spot loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedging.model_audit.projection_metrics import projected_scenarios


@dataclass(frozen=True, kw_only=True)
class ProjectedLaw:
    dynamics: FittedDynamics
    encoder: np.ndarray
    decoder: np.ndarray
    spot_loading: np.ndarray

    @property
    def factor_decoder(self) -> np.ndarray:
        return self.decoder @ self.dynamics.directions

    @property
    def factor_spot_loading(self) -> np.ndarray:
        return self.dynamics.directions.T @ (self.encoder.T @ self.spot_loading)

    @property
    def perpendicular_spot_loading(self) -> np.ndarray:
        return self.spot_loading - self.factor_decoder @ self.factor_spot_loading

    def next_native(self, current: np.ndarray) -> np.ndarray:
        return projected_scenarios(current, self.encoder, self.decoder, self.dynamics,
                                   native_spot_loading=self.spot_loading)

    def recursive_paths(self, current: np.ndarray, *, paths: int, steps: int, seed: int,
                        snapshot_steps: tuple[int, ...] | None = None) -> np.ndarray:
        """Keep empirical spot/factor shocks paired and restore the complete native spot change."""
        if paths < 1 or steps < 1 or current.shape != (self.encoder.shape[0],):
            raise ValueError("Recursive law needs positive path/horizon counts and matching native geometry")
        selected = set(range(1, steps + 1) if snapshot_steps is None else snapshot_steps)
        if not selected or min(selected) < 1 or max(selected) > steps:
            raise ValueError("Snapshots must fall within the simulated horizon")
        rng = np.random.default_rng(seed)
        fit = self.dynamics
        initial = (current @ self.encoder - fit.state_mean) @ fit.directions
        scores = np.broadcast_to(initial, (paths, len(initial))).copy()
        anchor = current - self.factor_decoder @ initial
        accumulated_spot = np.zeros(paths)
        result = []
        for position in range(steps):
            shock = fit.residuals[rng.integers(0, len(fit.residuals), paths)]
            scores = (scores @ fit.ar_matrix.T + fit.ar_intercept + shock[:, 1:]
                      + shock[:, :1] * self.factor_spot_loading)
            accumulated_spot += shock[:, 0]
            if position + 1 in selected:
                result.append(anchor + scores @ self.factor_decoder.T
                              + accumulated_spot[:, None] * self.perpendicular_spot_loading)
        return np.stack(result, axis=1)


def load_law(output: Path, model: str, day: date) -> ProjectedLaw:
    directory = output / f"fits/{model.replace(' ', '_')}/{day}"
    fit = load_dynamics(directory / "dynamics.npz")
    with np.load(directory / "projection.npz", allow_pickle=False) as saved:
        return ProjectedLaw(dynamics=fit, encoder=saved["encoder"], decoder=saved["decoder"],
                            spot_loading=saved["native_spot_loading"])
