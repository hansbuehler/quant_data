"""Load the saved qdata SANOS interpolators without market recalibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.bs import bs
from cdxcore.subdir import SubDir
from scipy.optimize import brentq
from scipy.special import ndtr

from quantdata.biz_calendar import ONE_MINUTE
from quantdata.sanos.arbfree import ArbFreePricer


def normalize_surface_moments(pricer: ArbFreePricer) -> dict[str, float]:
    """Remove certified source roundoff in mass/mean without fitting prices."""
    mass_errors = []
    mean_errors = []
    for index, (support, density) in enumerate(zip(pricer.mstrikes, pricer.mdensities, strict=True)):
        mass = float(density.sum())
        mean = float(density @ support / mass)
        mass_errors.append(abs(mass - 1.))
        mean_errors.append(abs(mean - 1.))
        pricer.mdensities[index] = density / mass
        pricer.mstrikes[index] = support / mean
    return {"source_mass_error": max(mass_errors), "source_mean_error": max(mean_errors)}


def surface_time_values(pricer: ArbFreePricer, *, expiries_y: np.ndarray,
                        strikes: np.ndarray) -> np.ndarray:
    """Evaluate the saved interpolation's OTM side directly for stable differences."""
    output = []
    for expiry in expiries_y:
        total_vol = float(pricer.interpolate_atm_vols(float(expiry))) * np.sqrt(pricer.smoothness * expiry)

        def node_price(index: int) -> np.ndarray:
            support = pricer.mstrikes[index]
            kernel = bs.price(k=strikes[:, None] / support[None, :], vol=total_vol,
                              is_call=(strikes >= 1.)[:, None], eps=pricer._eps_price)
            return (kernel * support[None, :]) @ pricer.mdensities[index]

        upper = int(np.searchsorted(pricer.expiries, expiry))
        if upper == 0:
            output.append(node_price(0))
        elif upper == len(pricer.expiries):
            output.append(node_price(-1))
        else:
            lower = upper - 1
            distances = (abs(expiry - pricer.expiries[lower]), abs(expiry - pricer.expiries[upper]))
            if min(distances) < ONE_MINUTE:
                output.append(node_price(upper if distances[1] < distances[0] else lower))
            else:
                weight = (expiry - pricer.expiries[lower]) / (pricer.expiries[upper] - pricer.expiries[lower])
                output.append((1. - weight) * node_price(lower) + weight * node_price(upper))
    return np.asarray(output)


def implied_vols_from_time_values(*, expiries_y: np.ndarray, strikes: np.ndarray,
                                  time_values: np.ndarray) -> np.ndarray:
    """Invert positive OTM values relatively using cdxcore Black prices.

    The bulk implied-vol API's absolute tolerance identifies small positive
    prices as intrinsic. That is unsuitable for tail vegas. Zero OTM value
    returns zero, an unidentified boundary rather than an invented vol floor.
    Subnormal prices also allow 32 ULPs for representation, not relative accuracy.
    Put-call symmetry evaluates OTM puts as scaled reciprocal-strike calls,
    avoiding the pricing API's cancellation when it applies put-call parity.
    """
    times, levels, values = np.broadcast_arrays(np.asarray(expiries_y, dtype=float),
        np.asarray(strikes, dtype=float), np.asarray(time_values, dtype=float))
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times <= 0.)
            or not np.isfinite(levels).all() or np.any(levels <= 0.)
            or not np.isfinite(values).all() or np.any(values < 0.)
            or np.any(values >= np.minimum(1., levels))):
        raise ValueError("OTM inversion needs positive finite tenors/strikes and valid nonnegative prices")
    output = np.zeros_like(values)
    for index, (expiry, strike, target) in enumerate(zip(times, levels, values, strict=True)):
        if target == 0.:
            continue
        log_target = float(np.log(target))

        def price(total_vol: float) -> float:
            call_strike = strike if strike >= 1. else 1. / strike
            multiplier = 1. if strike >= 1. else strike
            return multiplier * float(bs.price(k=np.array([call_strike]), vol=total_vol, is_call=True, eps=1e-16)[0])

        def residual(total_vol: float) -> float:
            value = price(total_vol)
            return float(np.log(value) - log_target) if value > 0. else -1e6

        upper = 1.
        while price(upper) < target and upper < 32.:
            upper *= 2.
        if price(upper) < target:
            raise ValueError("OTM inversion exceeded the numerical total-volatility bound")
        total_vol = brentq(residual, 0., upper, xtol=1e-14, rtol=1e-14, maxiter=200)
        reconstructed = price(total_vol)
        if (not np.isfinite(reconstructed) or reconstructed <= 0.
                or abs(reconstructed - target) > 1e-8 * target + 32. * np.nextafter(0., 1.)):
            raise RuntimeError(f"Relative OTM implied-vol reconstruction failed: strike={strike:.17g}, "
                f"expiry={expiry:.17g}, target={target:.17g}, reconstructed={reconstructed:.17g}")
        output[index] = total_vol / np.sqrt(expiry)
    return output


def surface_implied_vols(pricer: ArbFreePricer, *, expiries_y: np.ndarray,
                         strikes: np.ndarray) -> np.ndarray:
    """Accurate Black IVs of the same saved surface, without changing its prices."""
    times, levels = np.broadcast_arrays(np.asarray(expiries_y, dtype=float), np.asarray(strikes, dtype=float))
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times <= 0.) or not np.isfinite(levels).all() or np.any(levels <= 0.):
        raise ValueError("Surface IV queries require aligned positive finite tenors and strikes")
    time_values = np.empty_like(levels)
    for expiry in np.unique(times):
        selected = times == expiry
        time_values[selected] = surface_time_values(pricer, expiries_y=np.array([expiry]), strikes=levels[selected])[0]
    return implied_vols_from_time_values(expiries_y=times, strikes=levels, time_values=time_values)


def surface_digitals(pricer: ArbFreePricer, *, expiry_y: float,
                     strikes: np.ndarray, is_call: bool = True) -> np.ndarray:
    """Differentiate the saved smooth SANOS European interpolator, without refitting."""
    strikes = np.asarray(strikes, dtype=float)
    if strikes.ndim != 1 or not np.isfinite(strikes).all() or np.any(strikes <= 0) or expiry_y < 0:
        raise ValueError("Digital strikes must be finite, positive and one-dimensional")
    if expiry_y < ONE_MINUTE:
        calls = (1. > strikes).astype(float)
    else:
        total_vol = float(pricer.interpolate_atm_vols(expiry_y)) * np.sqrt(pricer.smoothness * expiry_y)

        def at_index(index: int) -> np.ndarray:
            support = pricer.mstrikes[index]
            if total_vol == 0:
                probabilities = (support[None, :] > strikes[:, None]).astype(float)
            else:
                probabilities = ndtr(np.log(support[None, :] / strikes[:, None]) / total_vol - .5 * total_vol)
            return probabilities @ pricer.mdensities[index]

        upper = int(np.searchsorted(pricer.expiries, expiry_y))
        if upper == 0:
            calls = at_index(0)
        elif upper == len(pricer.expiries):
            calls = at_index(-1)
        else:
            lower = upper - 1
            distances = (abs(expiry_y - pricer.expiries[lower]), abs(expiry_y - pricer.expiries[upper]))
            if min(distances) < ONE_MINUTE:
                calls = at_index(upper if distances[1] < distances[0] else lower)
            else:
                weight = (expiry_y - pricer.expiries[lower]) / (pricer.expiries[upper] - pricer.expiries[lower])
                calls = (1. - weight) * at_index(lower) + weight * at_index(upper)
    if not np.isfinite(calls).all() or calls.min() < -1e-8 or calls.max() > 1. + 1e-5:
        raise ValueError("The source digital price lies outside its probability bounds")
    return calls if is_call else 1. - calls


@dataclass(frozen=True, kw_only=True)
class SourceArchive:
    """Saved SANOS fits of one smoothness: 0 (DLV), 0.1 (SANOS benchmark) or 0.25."""

    root: Path
    smoothness: float = 0.1

    def __post_init__(self) -> None:
        if self.smoothness not in (0., .1, .25):
            raise ValueError(f"Saved SANOS fits exist for smoothness 0, 0.1 and 0.25, not {self.smoothness}")

    @property
    def model_name(self) -> str:
        return f"all-{self.smoothness:g}"

    def model_path(self, day: date) -> Path:
        candidates = [
            self.root / location / day.isoformat() / "models" / self.model_name / "model.h5"
            for location in ("days", "pilot")
        ]
        found = [path for path in candidates if path.is_file()]
        if not found:
            raise FileNotFoundError(f"Missing saved {self.model_name} model for {day}")
        return found[0]

    def dates(self) -> tuple[date, ...]:
        return tuple(sorted({
            date.fromisoformat(path.parents[2].name)
            for location in ("days", "pilot")
            for path in (self.root / location).glob(f"*/models/{self.model_name}/model.h5")
        }))

    def load(self, day: date) -> tuple[ArbFreePricer, dict[str, object]]:
        path = self.model_path(day)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        certificate_path = path.with_name("certificate.json")
        certificate = json.loads(certificate_path.read_text())
        if certificate.get("status") != "passed" or certificate.get("model_sha256") != checksum:
            raise ValueError(f"Source certificate/checksum failed for {day}")
        directory = SubDir(str(path.parent), fmt=SubDir.PYTREE_HDF5, create_directory=False)
        pricer = ArbFreePricer.from_dict(directory.read("model"))
        if pricer.smoothness != self.smoothness or pricer.mdensities is None:
            raise ValueError(f"Expected the stored {self.model_name} surface model on {day}")
        for support, density in zip(pricer.mstrikes, pricer.mdensities, strict=True):
            if (
                not np.isfinite(density).all() or not np.isfinite(support).all()
                or np.min(density) < -1e-10
                or abs(density.sum() - 1.0) > 1e-5
                or abs(density @ support - 1.0) > 1e-5
            ):
                raise ValueError(f"Invalid saved SANOS density on {day}")
        return pricer, {"source_model": str(path), "source_sha256": checksum,
                        "source_smoothness": pricer.smoothness}
