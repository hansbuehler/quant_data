"""Algebraic zero-backbone DLV initialization from a saved SANOS surface."""

from __future__ import annotations

import numpy as np
import torch
from cdxcore.bs import bs

from quantdata.biz_calendar import ONE_MINUTE
from quantdata.dysanos.flex.config import FlexGeometryConfig
from quantdata.dysanos.flex.model import (
    NativeParameters,
    forward_vols_from_total_vols,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.sanos import make_supports_from_total_vols
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.sanos.tridiag import solve_sanos_tridiagonal


def recover_initial_state(pricer: ArbFreePricer, *, geometry: FlexGeometryConfig) -> tuple[np.ndarray, dict[str, float | int]]:
    """Recover DLV squares on the decoder's actual shared grid from source prices.

    The recovered law has zero backbone. The current decoder imposes a
    positive minimum backbone and DLV floor, so the encoded state is only an
    initialization and all numerical floors/caps are reported.
    """
    times = geometry.model_expiries_y
    atm = np.asarray(pricer.interpolate_atm_vols(times), dtype=np.float64)
    target_total = atm * np.sqrt(times)
    vols = forward_vols_from_total_vols(torch.from_numpy(target_total), geometry=geometry).numpy()
    vols = np.clip(vols, geometry.min_forward_vol + 1e-6, geometry.max_forward_vol - 1e-6)
    actual_total = np.cumsum(vols * np.sqrt(np.diff(times, prepend=0.)))
    supports = make_supports_from_total_vols(actual_total, geometry=geometry)
    if not np.allclose(supports, supports[:1], rtol=0., atol=1e-13):
        raise ValueError("Algebraic initialization requires the shared strike grid")
    support = supports[0]
    inner = support[1:-1]
    time_values = []
    for expiry in times:
        std = float(pricer.interpolate_atm_vols(float(expiry))) * np.sqrt(pricer.smoothness * expiry)
        def node(index):
            strikes = pricer.mstrikes[index]
            values = bs.price(k=inner[:, None] / strikes[None, :], vol=std,
                              is_call=(inner >= 1.)[:, None], eps=pricer._eps_price)
            return (values * strikes[None, :]) @ pricer.mdensities[index]
        upper = int(np.searchsorted(pricer.expiries, expiry))
        if upper == 0:
            value = node(0)
        elif upper == len(pricer.expiries):
            value = node(-1)
        else:
            lower = upper - 1
            distances = (abs(expiry-pricer.expiries[lower]), abs(expiry-pricer.expiries[upper]))
            if min(distances) < ONE_MINUTE:
                value = node(upper if distances[1] < distances[0] else lower)
            else:
                w = (expiry-pricer.expiries[lower]) / (pricer.expiries[upper]-pricer.expiries[lower])
                value = (1.-w)*node(lower) + w*node(upper)
        time_values.append(value)
    time_values = np.asarray(time_values)
    if np.min(np.diff(time_values, axis=0, prepend=np.zeros((1, len(inner))))) < -1e-5:
        raise ValueError("Source calendar inconsistency exceeds tolerance")
    original_values = time_values.copy()
    time_values = np.maximum.accumulate(np.vstack((np.zeros(len(inner)), time_values)), axis=0)[1:]
    tv = np.c_[np.zeros(len(times)), time_values, np.zeros(len(times))]
    slopes = np.diff(tv, axis=1) / np.diff(support)
    density = np.diff(slopes, axis=1, prepend=np.zeros((len(times), 1)), append=np.zeros((len(times), 1)))
    atm_node = int(np.argmin(np.abs(support-1.)))
    if abs(support[atm_node]-1.) > 1e-12:
        raise ValueError("Grid has no exact ATM node")
    density[:, atm_node] += 1.
    theta = np.diff(tv[:, 1:-1], axis=0, prepend=np.zeros((1, len(inner))))
    box = inner**2 * np.diff(times, prepend=0.)[:, None] / (support[2:] - support[:-2])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.divide(theta, box*density[:, 1:-1], out=np.full_like(theta, np.nan), where=density[:, 1:-1] > 1e-14)
    valid = np.isfinite(raw) & (raw >= 0.)
    recovered = np.maximum(np.where(valid, raw, geometry.min_dlvsq), geometry.min_dlvsq + 1e-6)
    seed = np.minimum(recovered, geometry.max_dlvsq)
    payoff = np.maximum(support[None, :] - support[:, None], 0.)
    def repricing_error(variances):
        law = np.zeros_like(support)
        law[atm_node] = 1.
        error = 0.
        for i, step in enumerate(np.diff(times, prepend=0.)):
            weight = variances[i] * inner**2 * step / (support[2:] - support[:-2])
            law = solve_sanos_tridiagonal(w_minus=weight/(inner-support[:-2]), w_plus=weight/(support[2:]-inner), rhs=law)
            target = original_values[i] + np.maximum(1.-inner, 0.)
            error = max(error, float(np.max(np.abs((payoff @ law)[1:-1]-target))))
        return error
    dtype = torch.float64
    params = NativeParameters(dlvsqs=torch.tensor(seed, dtype=dtype), forward_vols=torch.tensor(vols, dtype=dtype),
                              eta=torch.full((len(times),), geometry.min_eta + 1e-5, dtype=dtype))
    raw_state = raw_from_native_parameters(params, geometry=geometry).numpy()
    return raw_state, {"source_smoothness": float(pricer.smoothness), "recovery_backbone": 0.,
                       "decoder_minimum_eta": float(geometry.min_eta), "undefined_nodes": int((~valid).sum()),
                       "capped_nodes": int((valid & (raw > geometry.max_dlvsq)).sum()),
                       "floored_nodes": int((~valid | (raw <= geometry.min_dlvsq)).sum()),
                       "calendar_adjustment": float(np.max(np.abs(time_values-original_values))),
                       "max_price_error_uncapped": repricing_error(recovered),
                       "max_price_error_after_guards": repricing_error(seed)}
