"""Positive log-IV grids from frozen market surfaces; no arbitrage guarantee.

NMS geometry uses the saved source's unsmoothed ATM volatility curve. The
geometry remains fixed when log-IV states are changed. Across NMS, interpolate
log-IV with constant boundary wings; across tenor, interpolate total variance
at the queried NMS. The zero-tenor variance is zero, and queries beyond the
last grid tenor are rejected. These conventions define European prices only,
not a joint law for path-dependent claims or digital strike derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.historical import licensed_curve, sanos_europeans
from quantdata.hedge2.source import SourceArchive, surface_time_values
from quantdata.sanos.arbfree import ArbFreePricer


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class IVGridConfig:
    business_dtes: tuple[float, ...] = (1., 2., 3., 4., 5., 10., 15., 20., 25.)
    num_nms_nodes: int = 29
    nms_range: tuple[float, float] = (-4., 3.)
    vol_min: float = 0.01
    vol_max: float = 5.
    price_tolerance: float = 1e-8
    low_vega_threshold: float = 1e-8

    def __post_init__(self) -> None:
        times = np.asarray(self.business_dtes, dtype=float)
        bounds = np.asarray(self.nms_range, dtype=float)
        verify_inp(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
                   and times[0] > 0 and np.all(np.diff(times) > 0),
                   lambda: f"Business DTEs must be positive and increasing: {self.business_dtes}")
        verify_inp(isinstance(self.num_nms_nodes, int) and not isinstance(self.num_nms_nodes, bool)
                   and self.num_nms_nodes >= 2 and bounds.shape == (2,)
                   and np.isfinite(bounds).all() and bounds[0] < 0 < bounds[1],
                   lambda: f"Invalid NMS geometry: {self.num_nms_nodes}, {self.nms_range}")
        verify_inp(np.isfinite([self.vol_min, self.vol_max]).all()
                   and 0 < self.vol_min < self.vol_max,
                   lambda: f"Inversion volatility bounds must be positive and ordered: {self.vol_min}, {self.vol_max}")
        verify_inp(np.isfinite(self.price_tolerance) and self.price_tolerance >= 1e-8
                   and np.isfinite(self.low_vega_threshold) and self.low_vega_threshold > 0,
                   lambda: f"Invalid inversion tolerances: {self.price_tolerance}, {self.low_vega_threshold}")

    @property
    def expiries_y(self) -> np.ndarray:
        return np.asarray(self.business_dtes, dtype=float) / DAYS_PER_YEAR

    @property
    def nms_nodes(self) -> np.ndarray:
        return np.linspace(*self.nms_range, self.num_nms_nodes)


@dataclass(frozen=True, kw_only=True)
class IVInterpolation:
    implied_vols: np.ndarray
    nms: np.ndarray
    nms_extrapolated: np.ndarray
    below_first_expiry: np.ndarray


@dataclass(frozen=True, kw_only=True)
class IVPriceResult:
    prices: np.ndarray
    implied_vols: np.ndarray
    nms: np.ndarray
    nms_extrapolated: np.ndarray
    below_first_expiry: np.ndarray
    forwards: np.ndarray
    discounts: np.ndarray


@dataclass(frozen=True, kw_only=True)
class IVGrid:
    day: date
    config: IVGridConfig
    reference_vols: np.ndarray
    log_implied_vols: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.config.business_dtes), self.config.num_nms_nodes)
        if (self.reference_vols.shape != (shape[0],) or self.log_implied_vols.shape != shape
                or not np.isfinite(self.reference_vols).all() or np.any(self.reference_vols <= 0)
                or not np.isfinite(self.log_implied_vols).all()):
            raise ValueError(f"IV state and reference geometry must be finite and match {shape}")
        with np.errstate(over="ignore", under="ignore"):
            implied_vols = np.exp(self.log_implied_vols)
        if not np.isfinite(implied_vols).all() or np.any(implied_vols <= 0):
            raise ValueError("Log-IV state must exponentiate to finite positive volatility")

    @property
    def state(self) -> np.ndarray:
        """Expiry-major positive-volatility state in logarithmic coordinates."""
        return self.log_implied_vols.reshape(-1).copy()

    def with_state(self, state: np.ndarray) -> IVGrid:
        """Replace the log-IV state while retaining the date's NMS geometry."""
        values = np.asarray(state, dtype=float)
        if values.shape != self.state.shape:
            raise ValueError(f"Log-IV state must have shape {self.state.shape}; got {values.shape}")
        return replace(self, log_implied_vols=values.reshape(self.log_implied_vols.shape).copy())

    def interpolate(self, *, business_dtes: np.ndarray | float,
                    strikes_over_forward: np.ndarray | float) -> IVInterpolation:
        """Interpolate at supplied K/F; return every constant-wing extrapolation."""
        days, strikes = np.broadcast_arrays(np.atleast_1d(np.asarray(business_dtes, dtype=float)),
                                            np.atleast_1d(np.asarray(strikes_over_forward, dtype=float)))
        if (days.ndim != 1 or not np.isfinite(days).all() or np.any(days <= 0)
                or np.any(days > self.config.business_dtes[-1])
                or not np.isfinite(strikes).all() or np.any(strikes <= 0)):
            raise ValueError("IV queries need finite positive K/F and DTE within the grid horizon")
        times = days / DAYS_PER_YEAR
        nodes = self.config.expiries_y
        reference_variance = np.interp(times, np.r_[0., nodes],
                                       np.r_[0., nodes * self.reference_vols**2])
        reference = np.sqrt(reference_variance / times)
        nms = NMS.to_nms(strikes, times, vol=reference, eps_vol=reference)
        log_vols = np.stack([np.interp(nms, self.config.nms_nodes, row)
                             for row in self.log_implied_vols])
        variances = np.exp(2. * log_vols) * nodes[:, None]
        upper = np.searchsorted(nodes, times)
        lower = np.maximum(upper - 1, 0)
        columns = np.arange(len(times))
        lower_times = np.where(upper == 0, 0., nodes[lower])
        lower_variance = np.where(upper == 0, 0., variances[lower, columns])
        weight = (times - lower_times) / (nodes[upper] - lower_times)
        total_variance = (1. - weight) * lower_variance + weight * variances[upper, columns]
        implied_vols = np.sqrt(total_variance / times)
        if not np.isfinite(implied_vols).all() or np.any(implied_vols <= 0):
            raise ValueError("Interpolated volatility is not finite and positive")
        return IVInterpolation(implied_vols=implied_vols, nms=nms,
            nms_extrapolated=(nms < self.config.nms_range[0]) | (nms > self.config.nms_range[1]),
            below_first_expiry=days < self.config.business_dtes[0])

    def price_europeans(self, *, curve: CashCurve, business_dtes: np.ndarray | float,
                        cash_strikes: np.ndarray | float,
                        is_call: np.ndarray | bool = True) -> IVPriceResult:
        """Price the original cash strikes using the supplied source forward and DF.

        Settlement uses the actual spot. A live zero-strike call is DF*F.
        IV and NMS are undefined (NaN) for settlement and zero-strike calls.
        The financing reference forward is never used as the pricing forward.
        """
        days, strikes, calls = np.broadcast_arrays(np.atleast_1d(np.asarray(business_dtes, dtype=float)),
            np.atleast_1d(np.asarray(cash_strikes, dtype=float)), np.atleast_1d(np.asarray(is_call)))
        if (days.ndim != 1 or not np.isfinite(days).all() or np.any(days < 0)
                or np.any(days > self.config.business_dtes[-1])
                or not np.isfinite(strikes).all() or np.any(strikes < 0)
                or calls.dtype != np.bool_ or np.any((strikes == 0) & ~calls)):
            raise ValueError("European queries need valid DTE, cash strike and boolean call/put side")
        forwards, _, discounts = curve.at(days)
        prices = np.maximum(np.where(calls, curve.spot - strikes, strikes - curve.spot), 0.)
        implied_vols = np.full(days.shape, np.nan)
        nms = np.full(days.shape, np.nan)
        outside = np.zeros(days.shape, dtype=bool)
        below_first = np.zeros(days.shape, dtype=bool)
        zero_calls = (days > 0) & (strikes == 0)
        prices[zero_calls] = discounts[zero_calls] * forwards[zero_calls]
        live = (days > 0) & (strikes > 0)
        if live.any():
            interpolated = self.interpolate(business_dtes=days[live],
                                            strikes_over_forward=strikes[live] / forwards[live])
            implied_vols[live] = interpolated.implied_vols
            nms[live] = interpolated.nms
            outside[live] = interpolated.nms_extrapolated
            below_first[live] = interpolated.below_first_expiry
            prices[live] = discounts[live] * forwards[live] * bs.price(
                k=strikes[live] / forwards[live], vol=implied_vols[live],
                sqrtT=np.sqrt(days[live] / DAYS_PER_YEAR), is_call=calls[live])
        if not np.isfinite(prices).all() or np.any(prices < 0):
            raise ValueError("Black European price must be finite and nonnegative")
        return IVPriceResult(prices=prices, implied_vols=implied_vols, nms=nms,
            nms_extrapolated=outside, below_first_expiry=below_first,
            forwards=forwards, discounts=discounts)


@dataclass(frozen=True, kw_only=True)
class IVGridExtraction:
    grid: IVGrid
    strikes_over_forward: np.ndarray
    source_calls: np.ndarray
    source_otm_prices: np.ndarray
    roundtrip_errors: np.ndarray
    otm_roundtrip_errors: np.ndarray
    source_parity_errors: np.ndarray
    inversion_floor: np.ndarray
    inversion_cap: np.ndarray
    inversion_failed: np.ndarray
    intrinsic_within_tolerance: np.ndarray
    low_vega: np.ndarray

    def summary(self) -> dict[str, object]:
        """Normalized price errors use cash price/(DF*F), uniformly over grid cells."""
        return {"date": self.grid.day.isoformat(), "grid_cells": self.source_calls.size,
            "inversion_floor_count": int(self.inversion_floor.sum()),
            "inversion_cap_count": int(self.inversion_cap.sum()),
            "inversion_failed_count": int(self.inversion_failed.sum()),
            "intrinsic_within_tolerance_count": int(self.intrinsic_within_tolerance.sum()),
            "low_vega_count": int(self.low_vega.sum()),
            "max_abs_roundtrip_error": float(np.max(np.abs(self.roundtrip_errors))),
            "canonical_call_roundtrip_tolerance_exceeded_count": int(np.sum(
                np.abs(self.roundtrip_errors) > self.grid.config.price_tolerance)),
            "max_abs_otm_roundtrip_error": float(np.max(np.abs(self.otm_roundtrip_errors))),
            "roundtrip_rmse": float(np.sqrt(np.mean(self.roundtrip_errors**2))),
            "max_abs_source_parity_error": float(np.max(np.abs(self.source_parity_errors))),
            "minimum_iv": float(np.exp(self.grid.log_implied_vols).min()),
            "maximum_iv": float(np.exp(self.grid.log_implied_vols).max())}


def extract_iv_grid(*, source: ArbFreePricer, day: date,
                    config: IVGridConfig = IVGridConfig()) -> IVGridExtraction:
    """Invert stable OTM prices from a frozen source; do not normalize or refit it.

    Prices within the configured inversion tolerance of intrinsic receive the
    explicit positive lower bound. All such cells and repriced OTM errors above
    tolerance are retained as diagnostics, including errors at inversion bounds.
    """
    times = config.expiries_y
    reference = np.asarray(source.interpolate_atm_vols(times), dtype=float)
    strikes = NMS.from_nms(config.nms_nodes[None, :], times[:, None],
                           vol=reference[:, None], eps_vol=reference[:, None])
    calls = np.asarray(source.price(expiries=times[:, None], strikes=strikes, mode="surface"))
    otm = np.stack([surface_time_values(source, expiries_y=np.array([expiry]), strikes=row)[0]
                    for expiry, row in zip(times, strikes, strict=True)])
    if not np.isfinite(calls).all() or not np.isfinite(otm).all() or np.any(otm < 0):
        raise ValueError("Frozen source prices must be finite with nonnegative OTM values")
    vols = bs.implied(k=strikes, prices=otm, is_call=strikes >= 1.,
        sqrtT=np.sqrt(times[:, None]), vol_min=config.vol_min, vol_max=config.vol_max,
        default_vol=config.vol_min, price_tol=config.price_tolerance)
    grid = IVGrid(day=day, config=config, reference_vols=reference, log_implied_vols=np.log(vols))
    reconstructed = bs.price(k=strikes, vol=vols, sqrtT=np.sqrt(times[:, None]))
    otm_errors = bs.price(k=strikes, vol=vols, sqrtT=np.sqrt(times[:, None]), is_call=strikes >= 1.) - otm
    vega = bs.vega(k=strikes, vol=vols, sqrtT=np.sqrt(times[:, None]))
    return IVGridExtraction(grid=grid, strikes_over_forward=strikes,
        source_calls=calls, source_otm_prices=otm, roundtrip_errors=reconstructed - calls,
        otm_roundtrip_errors=otm_errors,
        source_parity_errors=otm + np.maximum(1. - strikes, 0.) - calls,
        inversion_floor=vols == config.vol_min, inversion_cap=vols == config.vol_max,
        inversion_failed=np.abs(otm_errors) > config.price_tolerance,
        intrinsic_within_tolerance=otm <= config.price_tolerance,
        low_vega=vega <= config.low_vega_threshold)


def run_iv_grid_audit(*, source_root: Path, licensed_fit_directory: Path,
                      output_directory: Path, days: tuple[date, ...],
                      config: IVGridConfig = IVGridConfig(), max_contracts: int = 200) -> dict[str, object]:
    """Audit grid extraction and actual licensed cash strikes against source model prices.

    The comparison reference is the saved market interpolator, not observed
    option mids. Saved arrays are licensed derived data and remain outside Git.
    The output directory must be new, keeping concurrent diagnostic runs separate.
    """
    verify_inp(len(days) > 0 and max_contracts > 0, lambda: "Audit needs dates and positive contract count")
    output_directory.mkdir(parents=True, exist_ok=False)
    archive = SourceArchive(root=source_root)
    results = []
    (output_directory / "config.json").write_text(json.dumps({"config": asdict(config),
        "source_root": str(source_root), "licensed_fit_directory": str(licensed_fit_directory),
        "dates": [day.isoformat() for day in days], "max_contracts": max_contracts,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "price_reference": "frozen source SANOS; not observed mids", "arbitrage_free_guarantee": False}, indent=2) + "\n")
    for day in days:
        try:
            source, metadata = archive.load(day)
            curve = licensed_curve(licensed_fit_directory, day)
            extraction = extract_iv_grid(source=source, day=day, config=config)
            frame = pd.read_parquet(licensed_fit_directory / "SPX" / f"{day.isoformat()}.pdq",
                                    columns=["strike", "bdte", "is_am"])
            frame = frame.loc[(~frame["is_am"]) & (frame["bdte"] > 0)
                              & (frame["bdte"] <= config.business_dtes[-1])]
            frame = frame.drop_duplicates(["bdte", "strike"]).sort_values(["bdte", "strike"])
            if frame.empty:
                raise ValueError("No actual PM contracts within the IV grid horizon")
            indices = np.linspace(0, len(frame) - 1, min(len(frame), max_contracts), dtype=int)
            selected = frame.iloc[indices]
            query_days = np.repeat(selected["bdte"].to_numpy(dtype=float), 2)
            cash_strikes = np.repeat(selected["strike"].to_numpy(dtype=float), 2)
            calls = np.tile(np.array([True, False]), len(selected))
            priced = extraction.grid.price_europeans(curve=curve, business_dtes=query_days,
                                                     cash_strikes=cash_strikes, is_call=calls)
            source_prices = sanos_europeans(source=source, curve=curve, business_dtes=query_days,
                                            cash_strikes=cash_strikes, is_call=calls)
            normalized_errors = (priced.prices - source_prices) / curve.spot
            arrays = {name: value for name, value in vars(extraction).items() if isinstance(value, np.ndarray)}
            arrays.update({"nms_nodes": config.nms_nodes, "business_dtes": np.asarray(config.business_dtes),
                "reference_vols": extraction.grid.reference_vols, "log_implied_vols": extraction.grid.log_implied_vols,
                "query_business_dtes": query_days, "query_cash_strikes": cash_strikes, "query_is_call": calls,
                "query_source_prices": source_prices, "query_normalized_errors": normalized_errors})
            arrays.update({f"query_{name}": value for name, value in vars(priced).items()})
            np.savez_compressed(output_directory / f"{day.isoformat()}.npz", **arrays)
            result = {**extraction.summary(), **metadata, "status": "complete", "query_sides": len(calls),
                "source_first_business_dte": float(source.expiries[0] * DAYS_PER_YEAR),
                "nms_extrapolated_sides": int(priced.nms_extrapolated.sum()),
                "below_first_grid_expiry_sides": int(priced.below_first_expiry.sum()),
                "max_abs_query_error_bps_spot": float(np.max(np.abs(normalized_errors)) * 10000.),
                "query_rmse_bps_spot": float(np.sqrt(np.mean(normalized_errors**2)) * 10000.)}
            inside = ~priced.nms_extrapolated
            result["inside_nms_query_sides"] = int(inside.sum())
            if inside.any():
                result["inside_nms_max_abs_query_error_bps_spot"] = float(np.max(np.abs(normalized_errors[inside])) * 10000.)
                result["inside_nms_query_rmse_bps_spot"] = float(np.sqrt(np.mean(normalized_errors[inside]**2)) * 10000.)
        except (FileNotFoundError, ValueError) as error:
            result = {"date": day.isoformat(), "status": "failed", "issue": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {"schema": "hedge2-iv-grid-audit-v1", "dates": results,
               "price_reference": "frozen source SANOS; not observed mids", "arbitrage_free_guarantee": False}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--licensed-fit-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    args = parser.parse_args()
    run_iv_grid_audit(source_root=args.source_root, licensed_fit_directory=args.licensed_fit_directory,
                      output_directory=args.output_directory, days=tuple(args.dates))


if __name__ == "__main__":
    main()
