"""Batched floating DLV decoding and fixed-strike European payoff integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.state import DailyState
from quantdata.sanos.tridiag import solve_sanos_tridiagonal


def floating_inner_strikes(forward_vols: np.ndarray, *, config: DlvConfig) -> np.ndarray:
    """Exact batch counterpart of shared_inner_strikes, including its nonuniform spacing."""
    if (forward_vols.ndim != 2 or forward_vols.shape[1] != len(config.business_dtes)
            or not np.isfinite(forward_vols).all() or np.any(forward_vols <= 0)):
        raise ValueError("Batched forward vols must be finite positive path-by-expiry arrays")
    increments = np.diff(config.expiries_y, prepend=0.)
    total = np.cumsum(forward_vols * np.sqrt(increments), axis=1)
    scales = np.sqrt((forward_vols[:, :1]**2 * NMS.EPS + total**2) / (1. + NMS.EPS))
    half = config.num_inner_strikes // 2
    lower, upper = config.nms_range
    left_count = min(max(2, int(half * abs(lower) // (upper - lower))), half - 3)
    front = np.r_[np.linspace(lower, 0., left_count, endpoint=False), np.linspace(0., upper, half - left_count)]
    final = np.linspace(lower, upper, half)
    if np.any(np.isclose(final, 0., atol=1e-14, rtol=0.)):
        edges = np.linspace(lower, upper, half + 1)
        final = .5 * (edges[1:] + edges[:-1])
    ordered = np.sort(np.c_[scales[:, :1] * front, scales[:, -1:] * final], axis=1)
    atm_index = int(np.sum(front < 0.) + np.sum(final < 0.))
    if not np.all(ordered[:, atm_index] == 0.):
        raise ValueError("Every floating batch grid requires one exact ATM node")
    gap = config.minimum_spacing_fraction * (ordered[:, -1:] - ordered[:, :1]) / (ordered.shape[1] - 1)
    left = -np.cumsum(np.maximum(np.diff(ordered[:, :atm_index + 1], axis=1), gap)[:, ::-1], axis=1)[:, ::-1]
    right = np.cumsum(np.maximum(np.diff(ordered[:, atm_index:], axis=1), gap), axis=1)
    return np.exp(np.c_[left, np.zeros(len(ordered)), right])


def row_searchsorted_right(support: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Binary search each row without allocating a path-by-strike-by-node payoff cube."""
    lower, upper = np.zeros(strikes.shape, dtype=int), np.full(strikes.shape, support.shape[1], dtype=int)
    for _ in range(support.shape[1].bit_length() + 1):
        active = lower < upper
        middle = (lower + upper) // 2
        values = np.take_along_axis(support, np.minimum(middle, support.shape[1] - 1), axis=1)
        right = active & (values <= strikes)
        upper = np.where(active & ~right, middle, upper)
        lower = np.where(right, middle + 1, lower)
    return lower


@dataclass(frozen=True, kw_only=True)
class BatchDlvPricer:
    support: np.ndarray
    densities: np.ndarray
    business_dtes: np.ndarray
    lower_clips: np.ndarray
    upper_clips: np.ndarray

    def density_at(self, business_dte: float) -> np.ndarray:
        if (not np.isfinite(business_dte) or business_dte < 0.
                or business_dte > self.business_dtes[-1] + 1e-10):
            raise ValueError("The batch expiry lies outside the model horizon")
        initial = (self.support == 1.).astype(float)
        times = np.r_[0., self.business_dtes]
        upper = min(int(np.searchsorted(times, business_dte, side="right")), len(times) - 1)
        weight = (business_dte - times[upper - 1]) / (times[upper] - times[upper - 1])
        previous = initial if upper == 1 else self.densities[:, upper - 2]
        return (1. - weight) * previous + weight * self.densities[:, upper - 1]

    def europeans(self, *, business_dtes: np.ndarray, strikes: np.ndarray,
                  is_call: np.ndarray | bool = True, digital: bool = False) -> np.ndarray:
        """Normalized call/put prices; digitals use > for calls and <= for puts."""
        days = np.asarray(business_dtes, dtype=float)
        query = np.asarray(strikes, dtype=float)
        if query.ndim == 1:
            query = np.broadcast_to(query, (len(self.support), len(query)))
        calls = np.broadcast_to(np.asarray(is_call, dtype=bool), query.shape)
        if (days.ndim != 1 or query.shape != (len(self.support), len(days))
                or not np.isfinite(query).all() or np.any(query < 0.) or np.any((query == 0.) & ~calls)
                or not np.isfinite(days).all()):
            raise ValueError("Batch option queries must align paths, strikes and remaining expiries")
        output = np.empty_like(query)
        for tenor in np.unique(days):
            selected = days == tenor
            density = self.density_at(float(tenor))
            indices = row_searchsorted_right(self.support, query[:, selected])
            probabilities = np.c_[np.zeros(len(density)), np.cumsum(density, axis=1)]
            moments = np.c_[np.zeros(len(density)), np.cumsum(density * self.support, axis=1)]
            survival = np.c_[np.cumsum(density[:, ::-1], axis=1)[:, ::-1], np.zeros(len(density))]
            upper_moments = np.c_[np.cumsum((density * self.support)[:, ::-1], axis=1)[:, ::-1], np.zeros(len(density))]
            put_mass = np.take_along_axis(probabilities, indices, axis=1)
            call_mass = np.take_along_axis(survival, indices, axis=1)
            if digital:
                output[:, selected] = np.where(calls[:, selected], call_mass, put_mass)
            else:
                call = np.take_along_axis(upper_moments, indices, axis=1) - query[:, selected] * call_mass
                put = query[:, selected] * put_mass - np.take_along_axis(moments, indices, axis=1)
                output[:, selected] = np.maximum(np.where(calls[:, selected], call, put), 0.)
        return output


def decode_batch(state: DailyState, native: np.ndarray, *, max_business_dte: float | None = None) -> BatchDlvPricer:
    """Same guards, shared floating geometry and canonical implicit solver as scalar decoding."""
    values = np.asarray(native, dtype=float)
    config = state.config
    expiries, strikes = len(config.business_dtes), config.num_inner_strikes
    if (values.ndim != 2 or values.shape[1] != expiries * (strikes + 1)
            or not len(values) or not np.isfinite(values).all()):
        raise ValueError("Batch native states must be finite and have the original geometry")
    packed = values.reshape(len(values), expiries, strikes + 1)
    lower = np.r_[np.full(strikes, 2. * np.log(config.min_local_vol)), np.log(config.min_local_vol)]
    upper = np.r_[np.full(strikes, 2. * np.log(config.max_local_vol)), np.log(config.max_local_vol)]
    guarded = np.clip(packed, lower, upper)
    forward_vols = np.exp(guarded[:, :, -1])
    increments = np.diff(config.expiries_y, prepend=0.)
    geometry_bound = max(abs(value) for value in config.nms_range) * np.sum(forward_vols * np.sqrt(increments), axis=1)
    if np.any(geometry_bound > 30.):
        raise ValueError("The shocked forward-vol geometry exceeds the numerical log-strike bound of 30")
    inner = floating_inner_strikes(forward_vols, config=config)
    support = np.c_[state.reference_support[0] / state.reference_support[1] * inner[:, 0], inner,
                    state.reference_support[-1] / state.reference_support[-2] * inner[:, -1]]
    if max_business_dte is not None:
        if not np.isfinite(max_business_dte) or not 0 <= max_business_dte <= config.business_dtes[-1]:
            raise ValueError("The requested partial batch horizon must lie inside the original model")
        expiries = min(expiries, int(np.searchsorted(config.business_dtes, max_business_dte)) + 1)
    density = (support == 1.).astype(float)
    marginals = np.empty((len(values), expiries, strikes + 2))
    variances = np.exp(guarded[:, :, :-1])
    for index, increment in enumerate(increments[:expiries]):
        box = variances[:, index] * support[:, 1:-1]**2 * increment / (support[:, 2:] - support[:, :-2])
        density = solve_sanos_tridiagonal(w_minus=box / (support[:, 1:-1] - support[:, :-2]),
            w_plus=box / (support[:, 2:] - support[:, 1:-1]), rhs=density)
        marginals[:, index] = density
    if (not np.isfinite(marginals).all() or np.min(marginals) < -1e-12
            or np.max(np.abs(marginals.sum(axis=2) - 1.)) > 1e-8
            or np.max(np.abs(np.sum(marginals * support[:, None, :], axis=2) - 1.)) > 1e-8):
        raise ValueError("Batched DLV marginals lost positivity, mass or first moment")
    return BatchDlvPricer(support=support, densities=marginals,
        business_dtes=np.asarray(config.business_dtes[:expiries], dtype=float),
        lower_clips=np.sum(packed < lower, axis=(1, 2)), upper_clips=np.sum(packed > upper, axis=(1, 2)))
