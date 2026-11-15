"""Observe the repository GROMM2 law on the unchanged common log-IV chart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.metrics import covariance
from quantdata.hedging.model_audit.run import COMMON_NMS, GRID, QUANTILES, _json, _summarize


def observable(simulator, state, *, elapsed: int) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm.config import DAY

    logsig, boundaries, brownian, log_y = state
    front, loading = jax.vmap(simulator.front, in_axes=(0, 0, None))(boundaries, brownian, jnp.asarray(DAY))
    tower, _ = jax.vmap(simulator.chain, in_axes=(0, 0, 0, None))(logsig, front, loading, elapsed)
    densities, levels = np.asarray(tower), np.asarray(log_y)
    supports = np.asarray(simulator.strikes)
    spot = np.exp(levels) * np.sum(densities[:, elapsed] * supports[elapsed], axis=1)
    if not np.isfinite(densities).all() or np.any(spot <= 0.) or not np.isfinite(spot).all():
        raise ValueError("GROMM2 observable received an invalid density or spot; no paths removed")
    times = np.asarray(GRID.expiries_y)
    tenors = np.rint(times * 252.).astype(int)
    rows = tenors + elapsed - 1
    variances = np.asarray(simulator.yvar)
    result = np.empty((len(spot), len(times), len(COMMON_NMS)))
    for position, (tenor, row, expiry) in enumerate(zip(tenors, rows, times, strict=True)):
        nodes = supports[row][None] * np.exp(levels[:, None]) / spot[:, None]
        sd = np.sqrt(variances[tenor + elapsed] - variances[elapsed])

        def prices(strikes: np.ndarray) -> np.ndarray:
            kernels = bs.price(k=strikes[:, :, None] / nodes[:, None], vol=sd, sqrtT=1.) * nodes[:, None]
            return np.sum(kernels * densities[:, row, None], axis=-1)

        atm_price = prices(np.ones((len(spot), 1)))[:, 0]
        atm, identified = implied_volatilities_with_validity(expiries_y=np.full(len(spot), expiry),
            normalized_strikes=np.ones(len(spot)), normalized_prices=atm_price, is_call=np.ones(len(spot), dtype=bool))
        if not identified.all():
            raise ValueError(f"GROMM2 {int((~identified).sum())} unpriced ATM observations; no paths removed")
        strikes = NMS.from_nms(COMMON_NMS[None], expiry, vol=atm[:, None])
        values = prices(strikes)
        vols, identified = implied_volatilities_with_validity(expiries_y=np.full(values.size, expiry),
            normalized_strikes=strikes.ravel(), normalized_prices=values.ravel(), is_call=np.ones(values.size, dtype=bool))
        if not identified.all():
            raise ValueError(f"GROMM2 {int((~identified).sum())} unpriced common IV observations; no paths removed")
        result[:, position] = np.log(vols.reshape(values.shape))
    return result.reshape(len(spot), -1), spot


def score_origin(run: Path, reference: Path, *, origin: str, paths: int = 1000) -> None:
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm import estimate, pipeline, simulate

    directory = run / f"gromm2/{origin}"
    dynamics = estimate.load_dynamics(directory)
    common = json.loads((reference / f"daily/FuNVol/{origin}.json").read_text())
    with np.load(reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    initial = simulate.Model(dynamics, np.array([1], dtype=int), np.array([1.]))
    current, _ = observable(initial, initial.initial(1), elapsed=0)
    risk_price, physical = pipeline.market_price_of_risk(initial, dynamics)
    for measure in ("P", "Q"):
        simulator = simulate.Model(dynamics, np.array([1], dtype=int), np.array([1.]),
            lam=risk_price if measure == "P" else None)
        following, spot_returns = [], []
        for begin in range(0, paths, 100):
            count = min(100, paths - begin)
            state, _, _, _ = simulator.day(simulator.initial(count), jnp.int32(0),
                jax.random.PRNGKey(20_261_112 + begin), 8)
            surface, spot = observable(simulator, state, elapsed=1)
            following.append(surface)
            spot_returns.append(np.log(spot))
            print(f"GROMM2 {measure} {origin} matched forecast completed={begin+count}/{paths}", flush=True)
        future = np.concatenate(following)
        changes = future - current
        spot = np.concatenate(spot_returns)
        targets = np.c_[changes @ basis, spot]
        quantiles = np.quantile(targets, QUANTILES, axis=0)
        joint = np.c_[spot, changes @ basis]
        record = {name: common[name] for name in ("origin_date", "endpoint", "actual_change", "actual_spot", "actual_targets")}
        record.update(model=f"repository GROMM2-{measure}", status="scored", scenario_count=paths, valid_scenario_count=paths,
            surface_mean=changes.mean(axis=0).tolist(), surface_covariance=covariance(changes, ddof=0).tolist(),
            joint_mean=joint.mean(axis=0).tolist(), joint_covariance=covariance(joint, ddof=0).tolist(),
            quantiles=quantiles.tolist(), es_lower=[float(targets[targets[:, column] <= quantiles[1, column], column].mean()) for column in range(4)],
            scenario_covariance_ddof=0, invalid_surface_rate=0., extreme_surface_rate=float(np.any(future > np.log(3.), axis=1).mean()),
            forecast_rmse=float(np.sqrt(np.mean((changes.mean(axis=0) - np.asarray(common["actual_change"]))**2))),
            diagnostics={"sampling": "standalone GROMM2 Model.day, eight substeps, antithetic pairs and matched P/Q keys",
                "independent_backbone_preserved": True, "physical_measure": physical, "model_directory": str(directory)},
            bidask={"status": "not_scored", "reason": "Matched quote audit pending; repository fit statistics retained separately"})
        destination = run / f"daily/GROMM2-{measure}"
        _json(destination / f"{origin}.json", record)
        records = [json.loads(path.read_text()) for path in sorted(destination.glob("*.json"))]
        if len(records) >= 2:
            _json(run / f"GROMM2-{measure}-summary.json", _summarize(records, basis))
    _json(directory / "FORECAST_COMPLETE.json", {"origin": origin, "measures": ["P", "Q"], "paths": paths,
        "status": "matched one-day statistics complete; quote and no-arbitrage diagnostics pending"})
