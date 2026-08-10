"""Exact historical-market route and fitted-PCA cache."""

from __future__ import annotations

from datetime import date

import numpy as np
from cdxcore.subdir import SubDir
from cdxcore.verbose import Context

from quantdata.biz_calendar import DateRange
from quantdata.dysanos import _locations_dysanos
from quantdata.dysanos.grid.torch_market import GridConfig, GridMarketDefinition
from quantdata.pool import PoolConfig

from .experiment import PCAFitConfig
from .pca import (
    LogIVPCA,
    PCAFitDiagnostics,
    ValidatedPCAMarket,
    fit_log_iv_pca,
    validate_pca_market,
)

_fit_cache = _locations_dysanos.training("implied/pca-level-fit")


def grid_market_definition(fit_config: PCAFitConfig) -> GridMarketDefinition:
    """Return the fixed SPX historical scope required by the benchmark."""

    return GridMarketDefinition(
        ticker="SPX",
        date_range=DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30)),
        grid_config=GridConfig(grid_num_strikes=fit_config.grid_strikes),
    )


def load_pca_market(
    fit_config: PCAFitConfig,
    *,
    pool_config: PoolConfig = PoolConfig(num_workers=4, threading=False),
    verbose: Context = Context.all,
) -> tuple[object, ValidatedPCAMarket]:
    """Load and validate the exact nms-grid route shared with ML-SANOS."""

    print("** Starting to load market data...")
    market_definition = grid_market_definition(fit_config)
    market = market_definition.load(
        pool_config=pool_config,
        override_cache_mode="cacheonly",
        verbose=verbose,
    )
    validated = validate_pca_market(market)
    print(f"nms /done - found {len(validated.dates)} dates.")
    return market, validated


def _fit_cache_uid(fit_config: PCAFitConfig) -> str:
    return f"SPX-2020-01-01--2025-08-30/{fit_config.filename}"


@_fit_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.1",
    dependencies=[PCAFitConfig, fit_log_iv_pca, validate_pca_market],
    uid=_fit_cache_uid,
    exclude_args=["pool_config", "verbose"],
)
def _cached_pca_state(
    fit_config: PCAFitConfig,
    *,
    pool_config: PoolConfig,
    verbose: Context,
) -> dict[str, np.ndarray]:
    _, market = load_pca_market(
        fit_config, pool_config=pool_config, verbose=verbose
    )
    model, _ = fit_log_iv_pca(
        market.mid_vols,
        expiries=market.expiries[0],
        nms=market.nms[0],
        num_components=fit_config.num_surface_state,
    )
    return model.to_state_dict()


def load_fitted_pca(
    fit_config: PCAFitConfig,
    *,
    pool_config: PoolConfig = PoolConfig(num_workers=4, threading=False),
    verbose: Context = Context.all,
) -> tuple[object, ValidatedPCAMarket, LogIVPCA, PCAFitDiagnostics]:
    """Load the market and its versioned cached direct log-IV PCA fit."""

    market_object, market = load_pca_market(
        fit_config, pool_config=pool_config, verbose=verbose
    )
    state = _cached_pca_state(
        fit_config, pool_config=pool_config, verbose=verbose
    )
    model = LogIVPCA.from_state_dict(state)

    # Reconstruction diagnostics are cheap relative to fitting and deliberately
    # recomputed against the currently loaded observations.
    scores = model.encode(market.mid_vols)
    reconstructed_log = model.decode_log_iv(scores)
    reconstructed_iv = np.exp(reconstructed_log)
    log_error = reconstructed_log - np.log(market.mid_vols)
    iv_error = reconstructed_iv - market.mid_vols

    def summary(value: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(np.mean(np.square(value)))),
            "maximum_absolute_error": float(np.max(np.abs(value))),
        }

    errors_by_expiry = [
        {
            "expiry": float(model.expiries[index]),
            **{
                f"iv_{name}": value
                for name, value in summary(iv_error[:, index]).items()
            },
            **{
                f"log_iv_{name}": value
                for name, value in summary(log_error[:, index]).items()
            },
        }
        for index in range(model.expiries.size)
    ]
    region_masks = {
        "left_wing_z_lt_minus_0.5": model.nms < -0.5,
        "center_abs_z_le_0.5": np.abs(model.nms) <= 0.5,
        "right_wing_z_gt_0.5": model.nms > 0.5,
    }
    region_errors: dict[str, dict[str, float | int]] = {}
    for name, mask in region_masks.items():
        region_errors[name] = (
            {
                "grid_points": int(np.count_nonzero(mask)),
                "iv_rmse": float(np.sqrt(np.mean(np.square(iv_error[:, mask])))),
                "iv_maximum_absolute_error": float(np.max(np.abs(iv_error[:, mask]))),
                "log_iv_rmse": float(
                    np.sqrt(np.mean(np.square(log_error[:, mask])))
                ),
                "log_iv_maximum_absolute_error": float(
                    np.max(np.abs(log_error[:, mask]))
                ),
            }
            if np.any(mask)
            else {
                "grid_points": 0,
                "iv_rmse": float("nan"),
                "log_iv_rmse": float("nan"),
            }
        )
    component_count = min(5, model.num_components)
    shifted_log = (
        model.mean_log_iv[None]
        + np.sqrt(model.explained_variance[:component_count, None])
        * model.components[:component_count]
    )
    eigensurfaces = np.exp(shifted_log).reshape(
        component_count, *model.surface_shape
    ) - np.exp(model.mean_log_iv).reshape(model.surface_shape)
    diagnostics = PCAFitDiagnostics(
        scores=scores,
        reconstruction={
            **{f"iv_{name}": value for name, value in summary(iv_error).items()},
            **{
                f"log_iv_{name}": value
                for name, value in summary(log_error).items()
            },
        },
        errors_by_expiry=errors_by_expiry,
        errors_by_nms_region=region_errors,
        first_five_iv_eigensurfaces=eigensurfaces,
    )
    return market_object, market, model, diagnostics


__all__ = [
    "grid_market_definition",
    "load_fitted_pca",
    "load_pca_market",
]
