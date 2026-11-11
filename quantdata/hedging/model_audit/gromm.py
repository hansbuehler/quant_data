"""Observable one-day GROMM forecasts from the retained dated fits."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.metrics import covariance, quote_scores
from quantdata.hedging.model_audit.run import (
    COMMON_INDICES,
    COMMON_NMS,
    GRID,
    QUANTILES,
    _hash,
    _json,
    _summarize,
    build_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    load_gromm2_market_state,
)


def kernel_prices(support: np.ndarray, density: np.ndarray, background: np.ndarray,
                  spot: np.ndarray, strikes: np.ndarray, *, expiry: float) -> np.ndarray:
    """Native 5% Black backbone, evaluated on the OTM side before IV inversion."""
    latent = background[:, None] * support[None]
    cash_strikes = spot[:, None] * strikes
    calls = strikes >= 1.
    kernels = bs.price(k=cash_strikes[:, :, None] / latent[:, None, :],
        vol=.05, sqrtT=math.sqrt(expiry), is_call=calls[:, :, None], eps=1e-14)
    return np.sum(kernels * latent[:, None] * density[:, None], axis=-1) / spot[:, None]


def observable_surface(market, densities: np.ndarray, background: np.ndarray,
                       spot: np.ndarray, *, elapsed: int) -> tuple[np.ndarray, np.ndarray]:
    surfaces, validities = [], []
    for tenor in np.asarray(GRID.expiries_y) * 252.:
        selected = np.flatnonzero(np.isclose(market.business_dtes, tenor + elapsed, atol=1e-6, rtol=0))
        if len(selected) != 1:
            raise ValueError(f"No unique GROMM expiry at {tenor + elapsed} business days")
        row = int(selected[0])
        expiry = tenor / 252.
        density = densities[:, row]
        atm_price = kernel_prices(market.supports[row], density, background, spot,
                                   np.ones((len(spot), 1)), expiry=expiry)
        atm_vol, atm_valid = implied_volatilities_with_validity(expiries_y=np.full(len(spot), expiry),
            normalized_strikes=np.ones(len(spot)), normalized_prices=atm_price[:, 0], is_call=np.ones(len(spot), dtype=bool))
        if not atm_valid.all():
            raise ValueError("GROMM ATM prices do not identify finite IVs")
        strikes = NMS.from_nms(COMMON_NMS[None], expiry, vol=atm_vol[:, None])
        prices = kernel_prices(market.supports[row], density, background, spot, strikes, expiry=expiry)
        vols, valid = implied_volatilities_with_validity(expiries_y=np.full(prices.size, expiry),
            normalized_strikes=strikes.ravel(), normalized_prices=prices.ravel(), is_call=(strikes >= 1.).ravel())
        vols, valid = vols.reshape(prices.shape), valid.reshape(prices.shape)
        values = np.full_like(vols, np.nan)
        np.log(vols, out=values, where=valid & (vols > 0))
        surfaces.append(values)
        validities.append(valid)
    return np.stack(surfaces, axis=1), np.stack(validities, axis=1)


def reconstruction(root: Path, market, day: date) -> dict:
    path = root / f"experiments/hedge2/market-full-20261106T0001/market/days/{day}.parquet"
    quotes = pd.read_parquet(path)
    quotes = quotes.loc[(quotes.bdte >= 1.) & (quotes.bdte <= 10.) & (quotes.nms.abs() <= 1.5)].copy()
    prices = np.full(len(quotes), np.nan)
    for tenor in quotes.bdte.unique():
        matching = np.flatnonzero(np.isclose(market.business_dtes, tenor, atol=1e-6, rtol=0))
        if len(matching) != 1:
            continue
        selected = (quotes.bdte == tenor).to_numpy()
        block = quotes.loc[selected]
        row = int(matching[0])
        strikes = (block.strike / block.fwd).to_numpy()[None]
        normalized = kernel_prices(market.supports[row], market.density_tower[None, row],
            np.ones(1), np.ones(1), strikes, expiry=tenor / 252.)[0]
        own_call = strikes[0] >= 1.
        target_call = (block.option_type == "call").to_numpy()
        normalized += np.where(target_call & ~own_call, 1. - strikes[0], 0.)
        normalized += np.where(~target_call & own_call, strikes[0] - 1., 0.)
        prices[selected] = normalized * block.df.to_numpy() * block.fwd.to_numpy()
    result = quote_scores(prices, quotes.observed_bid, quotes.observed_ask)
    result["status"] = "scored"
    result["mode"] = "same-day reconstruction against licensed quotes; not stored training-window fit rate"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=512)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    panel = PublicHedgingPanel(arguments.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, arguments.output)
    with np.load(arguments.output / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    common = history.log_implied_volatilities[:, :, COMMON_INDICES].reshape(len(history.dates), -1)
    origins = [index for index, day in enumerate(history.dates) if date(2024, 4, 1) <= day <= date(2024, 5, 31)]
    step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    started = time.monotonic()
    for position, index in enumerate(origins):
        day = history.dates[index]
        if all((arguments.output / "daily" / f"GROMM-{measure}" / f"{day}.json").exists() for measure in ("P", "Q")):
            continue
        directory = arguments.root / f"training/hedging/gromm-current-public-20261024-v1/artifacts/{day}"
        manifest = json.loads((directory / "manifest.json").read_text())
        paths = SimpleNamespace(affine_history=directory / "affine-200-history.h5", dynamics=directory / "rank16-ar1-dynamics.npz")
        for name, path in (("affine_history_sha256", paths.affine_history), ("dynamics_sha256", paths.dynamics)):
            if _hash(path) != manifest[name]:
                raise ValueError(f"GROMM {day} input hash mismatch: {name}")
        names = ("percent_inside_spread", "explained_variance_ratio", "physical_projection_fraction", "candidate_physical_projection_fraction",
                 "physical_fallback_used", "physical_selected_model", "physical_accepted")
        market = load_gromm2_market_state(SimpleNamespace(paths=paths, as_of_date=day,
            **{name: manifest[name] for name in names}))
        current, _ = observable_surface(market, market.density_tower[None], np.ones(1), np.ones(1), elapsed=0)
        quotes = reconstruction(arguments.root, market, day)
        random = np.random.default_rng(20_261_111)
        half = random.standard_normal((8, arguments.paths // 2, market.dynamics.factor_count))
        normals = np.stack((half, -half), axis=2).reshape(8, arguments.paths, -1)
        background_half = random.standard_normal(arguments.paths // 2)
        background = np.stack((background_half, -background_half), axis=1).reshape(arguments.paths)
        state = SimulationState(front_density=jnp.broadcast_to(jnp.asarray(market.density_tower[0]), (arguments.paths, market.supports.shape[1])),
            dlvsq=jnp.broadcast_to(jnp.asarray(market.dlvsq), (arguments.paths, *market.dlvsq.shape)), background_level=jnp.ones(arguments.paths))
        for measure in ("Q", "P"):
            path = arguments.output / "daily" / f"GROMM-{measure}" / f"{day}.json"
            if path.exists():
                continue
            print(f"GROMM-{measure} {position+1}/{len(origins)} {day} simulation start paths={arguments.paths} elapsed={time.monotonic()-started:.1f}s", flush=True)
            shocks = normals if measure == "Q" else normals + market.physical_market_price_of_risk[None, None] * np.sqrt(1. / (252. * 8.))
            result = step(jnp.asarray(market.supports), jnp.asarray(market.transports),
                jnp.asarray(np.diff(market.expiries_y, prepend=0.)[1:]), state,
                jnp.asarray(market.dynamics.log_dlvsq_loadings), jnp.asarray(shocks), jnp.asarray(background),
                remaining_front_days=jnp.asarray(1), substeps=8)
            spot = np.asarray(result.spot)
            densities = np.asarray(result.density_tower)
            surfaces, validity = observable_surface(market, densities, np.asarray(result.state.background_level), spot, elapsed=1)
            valid = validity.all(axis=(1, 2)) & np.isfinite(surfaces).all(axis=(1, 2)) & np.isfinite(spot) & (spot > 0)
            if valid.sum() < 4:
                raise ValueError(f"GROMM {day} has too few valid observable scenarios")
            changes = surfaces[valid].reshape(valid.sum(), -1) - current[0].reshape(-1)
            returns = np.log(spot[valid])
            targets = np.column_stack((changes @ basis, returns))
            joint = np.column_stack((returns, changes @ basis))
            quantiles = np.quantile(targets, QUANTILES, axis=0)
            actual = common[index + 1] - common[index]
            actual_spot = float(history.forward_log_returns_into_date[index + 1])
            record = {"model": f"GROMM-{measure}", "origin_date": str(day), "endpoint": str(history.dates[index+1]), "status": "scored",
                "scenario_count": arguments.paths, "valid_scenario_count": int(valid.sum()),
                "surface_mean": changes.mean(axis=0).tolist(), "surface_covariance": covariance(changes).tolist(),
                "joint_mean": joint.mean(axis=0).tolist(), "joint_covariance": covariance(joint).tolist(),
                "actual_change": actual.tolist(), "actual_spot": actual_spot,
                "actual_targets": np.r_[actual @ basis, actual_spot].tolist(), "quantiles": quantiles.tolist(),
                "es_lower": [float(targets[targets[:, column] <= quantiles[1, column], column].mean()) for column in range(4)],
                "invalid_surface_rate": float(np.mean(~valid)), "extreme_surface_rate": float(np.mean(np.any(surfaces > np.log(3.), axis=(1, 2)))),
                "forecast_rmse": float(np.sqrt(np.mean((changes.mean(axis=0) - actual)**2))), "bidask": quotes,
                "diagnostics": {"fit_manifest": str(directory / "manifest.json"), "fit_manifest_sha256": _hash(directory / "manifest.json"),
                    "measure": measure, "substeps": 8, "antithetic_pairs": arguments.paths // 2, "seed": 20_261_111,
                    "stored_training_window_inside_spread_percent": manifest["percent_inside_spread"],
                    "physical_projection_fraction": manifest["physical_projection_fraction"],
                    "density_max_mass_error": float(np.max(np.abs(densities.sum(axis=-1) - 1.))),
                    "note": "Q versus historical P is a distributional comparison, not a Q calibration test"}}
            _json(path, record)
            print(f"GROMM-{measure} {position+1}/{len(origins)} scored elapsed={time.monotonic()-started:.1f}s", flush=True)
    summaries = json.loads((arguments.output / "summary.json").read_text())
    for measure in ("P", "Q"):
        records = [json.loads(path.read_text()) for path in sorted((arguments.output / "daily" / f"GROMM-{measure}").glob("*.json"))]
        summaries[f"GROMM-{measure}"] = _summarize(records, basis)
    _json(arguments.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
