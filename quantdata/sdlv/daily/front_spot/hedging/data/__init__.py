"""Interim public-candidate data materialization for the hedging benchmark."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantdata.sdlv.daily.front_spot.hedging.data.export_public_candidate import (
        InterimPublicDatasetConfig,
    )

__all__ = [
    "InterimPublicDatasetConfig",
    "GenerativeSurfaceGrid",
    "RollingDlvCalibrationConfig",
    "RollingDlvFactorConfig",
    "RollingFactorAlignmentConfig",
    "align_rolling_dlv_factors",
    "build_sanos_fixed_grid_history",
    "export_public_candidate",
    "fit_rolling_dlv_calibration",
    "fit_rolling_dlv_factors",
    "fixed_grid_sanos_day",
    "require_public_hedging_panel",
]


def __getattr__(name: str) -> Any:
    """Load exporter symbols without preloading its executable module."""
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "GenerativeSurfaceGrid",
        "build_sanos_fixed_grid_history",
        "fixed_grid_sanos_day",
    }:
        module = import_module(
            "quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces"
        )
        return getattr(module, name)
    if name in {
        "RollingDlvFactorConfig",
        "RollingFactorAlignmentConfig",
        "align_rolling_dlv_factors",
        "fit_rolling_dlv_factors",
    }:
        module = import_module(
            "quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors"
        )
        return getattr(module, name)
    if name in {"RollingDlvCalibrationConfig", "fit_rolling_dlv_calibration"}:
        module = import_module(
            "quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration"
        )
        return getattr(module, name)
    if name == "require_public_hedging_panel":
        module = import_module(
            "quantdata.sdlv.daily.front_spot.hedging.data.public_only"
        )
        return getattr(module, name)
    module = import_module(
        "quantdata.sdlv.daily.front_spot.hedging.data.export_public_candidate"
    )
    return getattr(module, name)
