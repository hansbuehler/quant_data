"""Terminal-payoff sampling from decoded SANOS surfaces.

SANOS represents a normalized terminal spot distribution as a discrete
mixture.  With smoothing, every mixture atom is replaced by a driftless
lognormal component.  Sampling that same mixture makes the Monte Carlo call
payoff consistent with the price returned by :class:`SanosPricer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from cdxcore.err import verify_inp


@dataclass(frozen=True, kw_only=True)
class TerminalSamplingDiagnostics:
    """Numerical corrections applied to SANOS probability weights."""

    minimum_raw_weight: float
    maximum_mass_error: float
    clipped_negative_weight_count: int


@dataclass(frozen=True, kw_only=True)
class TerminalPathDiagnostics:
    """Geometry and numerical range of a coherent simulated state path."""

    mode: str
    option_return_horizon: str
    trading_times: int
    trading_step_stride: int
    simulated_state_steps: int
    expiry_step_offsets: tuple[int, ...]
    dt: float
    maximum_expiry_time_error: float
    minimum_log_spot: float
    maximum_log_spot: float
    importance_sampling: dict[str, Any] | None = None


def _validate_surface_arrays(
    *,
    component_spots: np.ndarray,
    component_weights: np.ndarray,
    sqrt_variances: np.ndarray,
) -> tuple[int, int, int]:
    verify_inp(
        component_spots.ndim == 3,
        lambda: (
            "'component_spots' must have shape (S, T, M), got "
            f"{component_spots.shape}"
        ),
    )
    verify_inp(
        component_weights.shape == component_spots.shape,
        lambda: (
            "component spot/weight shapes differ: "
            f"{component_spots.shape}/{component_weights.shape}"
        ),
    )
    S, T, M = component_spots.shape
    verify_inp(
        sqrt_variances.shape == (S, T),
        lambda: (
            f"'sqrt_variances' must have shape {(S, T)}, got "
            f"{sqrt_variances.shape}"
        ),
    )
    verify_inp(min(S, T, M) > 0, "surface dimensions must be positive")
    for name, value in (
        ("component_spots", component_spots),
        ("component_weights", component_weights),
        ("sqrt_variances", sqrt_variances),
    ):
        verify_inp(
            np.issubdtype(value.dtype, np.floating),
            lambda name=name, value=value: (
                f"'{name}' must have floating dtype, got {value.dtype}"
            ),
        )
        verify_inp(
            np.all(np.isfinite(value)),
            lambda name=name: f"'{name}' contains non-finite values",
        )
    verify_inp(
        np.all(component_spots > 0.0),
        lambda: (
            "all SANOS component spots must be positive; found minimum "
            f"{float(np.min(component_spots)):g}"
        ),
    )
    verify_inp(
        np.all(sqrt_variances >= 0.0),
        lambda: (
            "SANOS square-root variances must be nonnegative; found minimum "
            f"{float(np.min(sqrt_variances)):g}"
        ),
    )
    return S, T, M


def sample_sanos_terminal_ratios(
    *,
    component_spots: np.ndarray,
    component_weights: np.ndarray,
    sqrt_variances: np.ndarray,
    smoothness: float | None,
    rng: np.random.Generator,
    negative_weight_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, TerminalSamplingDiagnostics]:
    """Draw one normalized terminal spot per path and expiry.

    For component spot ``x`` and smoothed standard deviation ``v``, the draw is
    ``x * exp(-v**2/2 + v*Z)``.  Its conditional mean is ``x`` and its call
    expectation is exactly the Black--Scholes component used by SANOS.
    """

    S, T, _ = _validate_surface_arrays(
        component_spots=component_spots,
        component_weights=component_weights,
        sqrt_variances=sqrt_variances,
    )
    verify_inp(
        smoothness is None or (math.isfinite(smoothness) and smoothness >= 0.0),
        lambda: f"'smoothness' must be None or finite and nonnegative, got {smoothness}",
    )
    verify_inp(
        math.isfinite(negative_weight_tolerance)
        and negative_weight_tolerance >= 0.0,
        lambda: (
            "'negative_weight_tolerance' must be finite and nonnegative, got "
            f"{negative_weight_tolerance}"
        ),
    )

    minimum_raw_weight = float(np.min(component_weights))
    verify_inp(
        minimum_raw_weight >= -negative_weight_tolerance,
        lambda: (
            "SANOS component probability is materially negative: minimum "
            f"{minimum_raw_weight:g} below tolerance {-negative_weight_tolerance:g}"
        ),
    )
    clipped_negative_weight_count = int(np.count_nonzero(component_weights < 0.0))
    weights = np.maximum(component_weights.astype(np.float64, copy=False), 0.0)
    mass = np.sum(weights, axis=-1, keepdims=True)
    verify_inp(
        np.all(np.isfinite(mass)) and np.all(mass > 0.0),
        "SANOS component probabilities have non-positive total mass",
    )
    maximum_mass_error = float(np.max(np.abs(mass - 1.0)))
    weights = weights / mass

    cdf = np.cumsum(weights, axis=-1)
    cdf[..., -1] = 1.0
    uniforms = rng.random((S, T, 1))
    component_index = np.sum(uniforms > cdf, axis=-1)
    selected_spot = np.take_along_axis(
        component_spots, component_index[..., None], axis=-1
    )[..., 0].astype(np.float64, copy=False)

    smoothing_scale = 0.0 if smoothness is None else math.sqrt(smoothness)
    smoothed_std = smoothing_scale * sqrt_variances.astype(np.float64, copy=False)
    normal = rng.standard_normal((S, T))
    terminal = selected_spot * np.exp(
        -0.5 * smoothed_std**2 + smoothed_std * normal
    )
    if not np.all(np.isfinite(terminal)) or not np.all(terminal > 0.0):
        raise FloatingPointError("sampled SANOS terminal spots are non-finite or non-positive")
    return terminal, TerminalSamplingDiagnostics(
        minimum_raw_weight=minimum_raw_weight,
        maximum_mass_error=maximum_mass_error,
        clipped_negative_weight_count=clipped_negative_weight_count,
    )


def build_one_time_terminal_universe(
    *,
    xstate: np.ndarray,
    expiries: np.ndarray,
    normalized_strikes: np.ndarray,
    normalized_calls: np.ndarray,
    component_spots: np.ndarray,
    component_weights: np.ndarray,
    sqrt_variances: np.ndarray,
    smoothness: float | None,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], TerminalSamplingDiagnostics]:
    """Construct one-trade-time option returns from SANOS terminal payoffs.

    Calls are entered at the decoded cash price and held to their actual
    expiry.  All strikes for one path/expiry share the same terminal spot.
    The first-expiry terminal spot also supplies the single spot return.
    """

    terminal_ratio, diagnostics = sample_sanos_terminal_ratios(
        component_spots=component_spots,
        component_weights=component_weights,
        sqrt_variances=sqrt_variances,
        smoothness=smoothness,
        rng=rng,
    )
    S, T, _ = component_spots.shape
    verify_inp(
        xstate.ndim == 2 and xstate.shape[0] == S and xstate.shape[1] > 1,
        lambda: f"'xstate' must have shape ({S}, XS>1), got {xstate.shape}",
    )
    verify_inp(
        expiries.shape == (T,),
        lambda: f"'expiries' must have shape {(T,)}, got {expiries.shape}",
    )
    verify_inp(
        np.all(np.isfinite(expiries)) and np.all(expiries > 0.0),
        "expiries must be finite and positive",
    )
    verify_inp(
        normalized_strikes.ndim == 3
        and normalized_strikes.shape[:2] == (S, T),
        lambda: (
            "'normalized_strikes' must have shape (S, T, N), got "
            f"{normalized_strikes.shape}"
        ),
    )
    verify_inp(
        normalized_calls.shape == normalized_strikes.shape,
        lambda: (
            "normalized call/strike shapes differ: "
            f"{normalized_calls.shape}/{normalized_strikes.shape}"
        ),
    )
    for name, value in (
        ("xstate", xstate),
        ("expiries", expiries),
        ("normalized_strikes", normalized_strikes),
        ("normalized_calls", normalized_calls),
    ):
        verify_inp(
            np.issubdtype(value.dtype, np.floating),
            lambda name=name, value=value: (
                f"'{name}' must have floating dtype, got {value.dtype}"
            ),
        )
        verify_inp(
            np.all(np.isfinite(value)),
            lambda name=name: f"'{name}' contains non-finite values",
        )
    verify_inp(
        np.all(normalized_strikes > 0.0),
        "normalized strikes must be strictly positive",
    )
    verify_inp(
        np.all(normalized_calls > 0.0),
        lambda: (
            "cash-normalized OCE requires strictly positive call premiums; "
            f"found minimum {float(np.min(normalized_calls)):g}"
        ),
    )

    dtype = np.result_type(xstate.dtype, normalized_calls.dtype, np.float32)
    spot = np.exp(xstate[:, 0].astype(np.float64)).astype(dtype)
    cash_calls = normalized_calls.astype(dtype, copy=False) * spot[:, None, None]
    cash_payoff = np.maximum(
        terminal_ratio[:, :, None] - normalized_strikes, 0.0
    ).astype(dtype) * spot[:, None, None]
    dC = cash_payoff - cash_calls
    dS = spot * (terminal_ratio[:, 0].astype(dtype) - 1.0)

    universe = {
        "dC": dC[:, None, :, :],
        "dS": dS[:, None],
        "C": cash_calls[:, None, :, :],
        "spot": spot[:, None],
        "delta": np.zeros_like(dC[:, None, :, :]),
        "expiries": expiries.astype(dtype, copy=False)[None, None, :],
        "log_strikes": np.log(normalized_strikes).astype(dtype)[:, None, :, :],
        "xstate": xstate.astype(dtype, copy=False)[:, None, :],
    }
    for name, value in universe.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"generated universe field {name!r} is non-finite")
    return universe, diagnostics


def expiry_step_offsets(
    expiries: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, float]:
    """Map year-fraction expiries to an increasing daily path grid."""

    verify_inp(
        expiries.ndim == 1
        and expiries.size > 0
        and np.all(np.isfinite(expiries))
        and np.all(expiries > 0.0)
        and np.all(np.diff(expiries) > 0.0),
        lambda: f"'expiries' must be finite, positive, and increasing; got {expiries}",
    )
    verify_inp(
        math.isfinite(dt) and dt > 0.0,
        lambda: f"'dt' must be finite and positive, got {dt}",
    )
    offsets = np.rint(expiries / dt).astype(np.int64)
    verify_inp(
        np.all(offsets > 0) and np.all(np.diff(offsets) > 0),
        lambda: (
            "expiry offsets must be positive and increasing on the selected "
            f"path grid; got {offsets}"
        ),
    )
    maximum_error = float(np.max(np.abs(offsets.astype(np.float64) * dt - expiries)))
    return offsets, maximum_error


def build_terminal_path_universe(
    *,
    xstate_path: np.ndarray,
    trading_times: int,
    trading_step_stride: int = 1,
    expiries: np.ndarray,
    expiry_offsets: np.ndarray,
    normalized_strikes: np.ndarray,
    normalized_calls: np.ndarray,
    normalized_deltas: np.ndarray,
    normalized_calls_tp1: np.ndarray | None = None,
    option_return_step_offset: int = 1,
    require_bounded_deltas: bool = True,
) -> dict[str, np.ndarray]:
    """Build expiry or repriced option P&Ls from one common state path.

    Contract ``(t, j, k)`` is entered at daily-path index
    ``t * trading_step_stride``.  With no ``normalized_calls_tp1``, its payoff
    uses the same sample's spot at the contract expiry.  When endpoint calls
    are supplied, they must be prices at ``t + option_return_step_offset`` of
    the original cash strike and remaining maturity.  An option expiring on
    or before that endpoint is instead settled at its exact daily expiry and
    its cash payoff is carried to the return endpoint.  The option's stock
    hedge is closed at the same exact expiry; otherwise it is closed at the
    common return endpoint.
    ``normalized_deltas`` are the initial cash-call deltas.
    SANOS deltas are bounded by default. Direct surface decoders may set
    ``require_bounded_deltas=False`` so smile-induced violations remain visible
    rather than being rejected or clipped.
    """

    verify_inp(
        xstate_path.ndim == 3 and xstate_path.shape[-1] > 1,
        lambda: (
            "'xstate_path' must have shape (S, path_steps, XS>1), got "
            f"{xstate_path.shape}"
        ),
    )
    S, path_steps, XS = xstate_path.shape
    verify_inp(
        0 < trading_times < path_steps,
        lambda: (
            f"'trading_times' must be in [1, {path_steps - 1}], got "
            f"{trading_times}"
        ),
    )
    verify_inp(
        isinstance(trading_step_stride, int) and trading_step_stride > 0,
        lambda: (
            "'trading_step_stride' must be a positive integer, got "
            f"{trading_step_stride!r}"
        ),
    )
    verify_inp(
        isinstance(option_return_step_offset, int)
        and option_return_step_offset > 0,
        lambda: (
            "'option_return_step_offset' must be a positive integer, got "
            f"{option_return_step_offset!r}"
        ),
    )
    verify_inp(
        isinstance(require_bounded_deltas, bool),
        lambda: (
            "'require_bounded_deltas' must be a bool, got "
            f"{type(require_bounded_deltas)}"
        ),
    )
    verify_inp(
        expiries.ndim == 1 and expiry_offsets.shape == expiries.shape,
        lambda: (
            "expiry values/offsets must be one-dimensional with equal shape; "
            f"got {expiries.shape}/{expiry_offsets.shape}"
        ),
    )
    T = expiries.size
    verify_inp(
        np.issubdtype(expiry_offsets.dtype, np.integer)
        and np.all(expiry_offsets > 0)
        and np.all(np.diff(expiry_offsets) > 0),
        lambda: f"expiry offsets must be positive increasing integers, got {expiry_offsets}",
    )
    last_trade_index = (trading_times - 1) * trading_step_stride
    maximum_return_offset = (
        int(expiry_offsets[-1])
        if normalized_calls_tp1 is None
        else option_return_step_offset
    )
    required_steps = last_trade_index + maximum_return_offset + 1
    verify_inp(
        path_steps >= required_steps,
        lambda: (
            f"state path has {path_steps} steps but {required_steps} are required "
            "to reach every option return horizon"
        ),
    )
    verify_inp(
        normalized_strikes.ndim == 4
        and normalized_strikes.shape[:3] == (S, trading_times, T),
        lambda: (
            "'normalized_strikes' must have shape "
            f"({S}, {trading_times}, {T}, N), got {normalized_strikes.shape}"
        ),
    )
    verify_inp(
        normalized_calls.shape
        == normalized_strikes.shape
        == normalized_deltas.shape,
        lambda: (
            "normalized call/strike/delta shapes differ: "
            f"{normalized_calls.shape}/{normalized_strikes.shape}/"
            f"{normalized_deltas.shape}"
        ),
    )
    if normalized_calls_tp1 is not None:
        verify_inp(
            normalized_calls_tp1.shape == normalized_calls.shape,
            lambda: (
                "next/current normalized call shapes differ: "
                f"{normalized_calls_tp1.shape}/{normalized_calls.shape}"
            ),
        )
    surface_arrays = (
        ("xstate_path", xstate_path),
        ("expiries", expiries),
        ("normalized_strikes", normalized_strikes),
        ("normalized_calls", normalized_calls),
        ("normalized_deltas", normalized_deltas),
    )
    if normalized_calls_tp1 is not None:
        surface_arrays += (("normalized_calls_tp1", normalized_calls_tp1),)
    for name, value in surface_arrays:
        verify_inp(
            np.issubdtype(value.dtype, np.floating) and np.all(np.isfinite(value)),
            lambda name=name, value=value: (
                f"'{name}' must have a finite floating array, got dtype "
                f"{value.dtype}"
            ),
        )
    verify_inp(
        np.all(normalized_strikes > 0.0),
        "normalized strikes must be strictly positive",
    )
    verify_inp(
        np.all(normalized_calls > 0.0),
        lambda: (
            "cash-normalized OCE requires strictly positive call premiums; "
            f"found minimum {float(np.min(normalized_calls)):g}"
        ),
    )
    if normalized_calls_tp1 is not None:
        verify_inp(
            np.all(normalized_calls_tp1 >= 0.0),
            lambda: (
                "next normalized call prices must be nonnegative; found minimum "
                f"{float(np.min(normalized_calls_tp1)):g}"
            ),
        )
    if require_bounded_deltas:
        verify_inp(
            np.all((normalized_deltas >= 0.0) & (normalized_deltas <= 1.0)),
            lambda: (
                "normalized call deltas must be in [0, 1]; found range "
                f"[{float(np.min(normalized_deltas)):g}, "
                f"{float(np.max(normalized_deltas)):g}]"
            ),
        )

    # The established universe schema is float32.  Convert blockwise here so
    # million-path path-mode arrays remain tractable in CPU and GPU memory.
    dtype = np.dtype(np.float32)
    log_spot = xstate_path[..., 0].astype(np.float64, copy=False)
    maximum_finite_log_spot = math.log(np.finfo(dtype).max)
    verify_inp(
        float(np.max(log_spot)) <= maximum_finite_log_spot,
        lambda: (
            "simulated log-spot overflows the universe dtype: maximum "
            f"{float(np.max(log_spot)):g}, limit {maximum_finite_log_spot:g}"
        ),
    )
    spot_path = np.exp(log_spot).astype(dtype)
    trading_indices = (
        np.arange(trading_times, dtype=np.int64) * trading_step_stride
    )
    spot = spot_path[:, trading_indices]
    cash_calls = normalized_calls.astype(dtype, copy=False) * spot[:, :, None, None]
    cash_strikes = (
        normalized_strikes.astype(dtype, copy=False) * spot[:, :, None, None]
    )
    if normalized_calls_tp1 is None:
        expiry_indices = trading_indices[:, None] + expiry_offsets[None, :]
        settlement_spot = spot_path[:, expiry_indices]
        endpoint_option_value = np.maximum(
            settlement_spot[:, :, :, None] - cash_strikes,
            0.0,
        )
        dC = endpoint_option_value - cash_calls
        dS = spot_path[:, trading_indices + trading_step_stride] - spot
        option_dS = settlement_spot - spot[:, :, None]
        option_dS_name = "expiry_dS"
        option_expired = np.ones((1, 1, T), dtype=bool)
    else:
        return_spot = spot_path[:, trading_indices + option_return_step_offset]
        cash_calls_tp1 = (
            normalized_calls_tp1.astype(dtype, copy=False)
            * return_spot[:, :, None, None]
        )
        expired = expiry_offsets <= option_return_step_offset
        settlement_spot = np.broadcast_to(
            return_spot[:, :, None], (S, trading_times, T)
        ).copy()
        if np.any(expired):
            exact_expiry_indices = (
                trading_indices[:, None] + expiry_offsets[None, expired]
            )
            settlement_spot[:, :, expired] = spot_path[:, exact_expiry_indices]
        cash_payoff = np.maximum(
            settlement_spot[:, :, :, None] - cash_strikes,
            0.0,
        )
        endpoint_option_value = cash_calls_tp1.copy()
        endpoint_option_value[:, :, expired, :] = cash_payoff[:, :, expired, :]
        dC = endpoint_option_value - cash_calls
        dS = return_spot - spot
        option_settlement_dS = settlement_spot - spot[:, :, None]
        option_dS = option_settlement_dS.copy()
        option_dS_name = "option_dS"
        option_expired = expired[None, None, :]
    universe = {
        "dC": dC,
        "dS": dS,
        option_dS_name: option_dS,
        "C": cash_calls,
        "spot": spot,
        "delta": normalized_deltas.astype(dtype, copy=False),
        "expiries": expiries.astype(dtype, copy=False)[None, None, :],
        "option_expired": option_expired,
        "log_strikes": np.log(normalized_strikes).astype(dtype, copy=False),
        "xstate": xstate_path[:, trading_indices, :].astype(dtype, copy=False),
    }
    if normalized_calls_tp1 is not None:
        universe["option_settlement_dS"] = option_settlement_dS
    expected_shapes = {
        "dC": normalized_calls.shape,
        "dS": (S, trading_times),
        option_dS_name: (S, trading_times, T),
        "C": normalized_calls.shape,
        "spot": (S, trading_times),
        "delta": normalized_calls.shape,
        "expiries": (1, 1, T),
        "option_expired": (1, 1, T),
        "log_strikes": normalized_calls.shape,
        "xstate": (S, trading_times, XS),
    }
    if normalized_calls_tp1 is not None:
        expected_shapes["option_settlement_dS"] = (S, trading_times, T)
    for name, value in universe.items():
        if value.shape != expected_shapes[name]:
            raise RuntimeError(
                f"generated universe field {name!r} has shape {value.shape}, "
                f"expected {expected_shapes[name]}"
            )
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"generated universe field {name!r} is non-finite")
    return universe
