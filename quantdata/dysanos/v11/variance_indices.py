"""Analytic, business-time variance-swap indices for SANOS/DYSANOS.

For a normalized martingale marginal X*exp(-v/2+sqrt(v)*Z), its log-contract
total variance is v - 2 E[log X]. This is the continuous-strike model-free
variance integral, not Cboe's discrete, calendar-time VIX index.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import _decode_surface
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import _market_dates_from_completion
from quantdata.dysanos.v11.run_rolling_direct_ar import (
    _historical_sanos_directory,
    historical_sanos_cache,
)

BUSINESS_DTES = tuple(range(5, 71, 5))


def log_contract_total_variance(
    support: torch.Tensor, density: torch.Tensor, backbone_total_variance: torch.Tensor,
) -> torch.Tensor:
    """Exact log contract of a mean-one discrete/Black mixture; differentiable."""
    if support.shape != density.shape or support.shape[:-1] != backbone_total_variance.shape:
        raise ValueError("Support, density and backbone shapes do not match")
    return backbone_total_variance - 2 * torch.sum(density * torch.log(support), dim=-1)


def decoded_variance_indices(model: object, states: torch.Tensor, *, business_dtes: tuple[int, ...] = BUSINESS_DTES) -> torch.Tensor:
    """Return VIX-type vol points at requested existing DYSANOS model nodes."""
    surface = _decode_surface(model, states)
    indices = []
    nodes = np.asarray(model.config.geometry.model_business_dtes)
    for tenor in business_dtes:
        matches = np.flatnonzero(np.isclose(nodes, tenor, atol=1e-8, rtol=0))
        if len(matches) != 1:
            raise ValueError(f"Requested tenor {tenor} is not a model node")
        indices.append(int(matches[0]))
    tv = log_contract_total_variance(
        surface.support_strikes[:, indices], surface.densities[:, indices],
        surface.backbone_total_vols[:, indices].square(),
    )
    times = surface.expiries_y[:, indices]
    return 100 * torch.sqrt(torch.clamp(tv, min=torch.finfo(tv.dtype).tiny) / times)


def historic_variance_indices(
    pricer: dict, *, business_dtes: tuple[int, ...] = BUSINESS_DTES,
) -> np.ndarray:
    """Exact indices of a stored historical SANOS surface, without extrapolation.

    Every supported tenor uses the historical pricer's interpolation of the
    discrete log moment and its Black-backbone variance. Unsupported maturities
    are NaN and retain their own per-tenor observation counts.
    """
    times = np.asarray(pricer["expiries"], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Historical SANOS expiries are not strictly increasing")
    supports, densities = pricer["mstrikes"], pricer["mdensities"]
    moments = []
    for k, p in zip(supports, densities, strict=True):
        k, p = np.asarray(k, dtype=np.float64), np.asarray(p, dtype=np.float64)
        if not (np.all(np.isfinite(k)) and np.all(k > 0) and np.all(np.isfinite(p))
                and p.min() >= -1e-7 and abs(p.sum() - 1) < 1e-5
                and abs(p @ k - 1) < 1e-5):
            raise ValueError("Invalid positive, mean-one historical SANOS marginal")
        moments.append(float(p @ np.log(k)))
    query = np.asarray(business_dtes, dtype=np.float64) / 252
    valid = (query >= times[0] - 1e-8) & (query <= times[-1] + 1e-8)
    smoothness = pricer["smoothness"]
    backbone = 0.0 if smoothness is None else float(smoothness) * np.interp(query, times, pricer["variances"])
    total_variance = backbone - 2 * np.interp(query, times, moments)
    if np.any(total_variance[valid] <= 0) or np.any(~np.isfinite(total_variance[valid])):
        raise ValueError("Invalid historical log-contract variance")
    result = np.full(len(query), np.nan)
    result[valid] = 100 * np.sqrt(total_variance[valid] / query[valid])
    return result


def prepare_history(*, run_directory: Path, output: Path, workers: int = 4) -> dict:
    """Read all source-panel SANOS surfaces, including dates outside the AR fit."""
    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four input readers")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run_directory / "reports/run-manifest.json").read_text())
    dates = _market_dates_from_completion(Path(manifest["native_completion_source"]))
    directory = _historical_sanos_directory(ticker="SPX", dates=dates)
    root = Path(historical_sanos_cache.path)

    def read(trading_date: date) -> np.ndarray:
        payload = historical_sanos_cache.read(
            str((directory / str(trading_date)).relative_to(root)),
            delete_wrong_version=False, raise_on_error=True,
        )
        if payload["d"] != trading_date:
            raise RuntimeError("Cached historical date mismatch")
        return historic_variance_indices(payload["af_pricer"])

    values = []
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for count, row in enumerate(executor.map(read, dates), start=1):
            values.append(row)
            if count == 1 or count % 50 == 0 or count == len(dates):
                elapsed = perf_counter() - started
                print(f"analytic history={count}/{len(dates)} elapsed={elapsed:.1f}s "
                      f"remaining={elapsed * (len(dates) - count) / count:.1f}s", flush=True)
    values = np.stack(values)
    data = {"date": list(dates), **{f"vix_{tenor}d": values[:, j] for j, tenor in enumerate(BUSINESS_DTES)}}
    pl.DataFrame(data).write_parquet(output / "historic_variance_indices.parquet")
    report = {
        "schema": "dysanos-v11-analytic-variance-indices-v1", "status": "complete",
        "dates": len(dates), "first_date": str(dates[0]), "last_date": str(dates[-1]),
        "source": str(directory), "initial_model_as_of": manifest["as_of"],
        "definition": "100 sqrt((backbone total variance - 2 E[log X_T])/(business days/252)); exact continuous-strike SANOS log contract, not discrete Cboe VIX",
        "coverage": {str(tenor): int(np.isfinite(values[:, j]).sum()) for j, tenor in enumerate(BUSINESS_DTES)},
        "no_extrapolation": True, "whole_available_history_used": True,
    }
    write_json_atomic(report, output / "manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    print(json.dumps(prepare_history(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
