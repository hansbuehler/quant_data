"""Blockwise coherent spot/option universes for direct PCA IV surfaces."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.dysanos.grid.gendysanos.terminal_universe import (
    TerminalPathDiagnostics,
    build_terminal_path_universe,
    expiry_step_offsets,
)
from quantdata.dysanos.grid.gendysanos.torch_prep import (
    _importance_control_diagnostics,
    _importance_target_steps,
)

from .decoder import PCATotalVarianceDecoder, WingHitDiagnostics
from .experiment import ImpliedPCAExperiment
from .simulator import ImpliedPCASampler
from .static_arbitrage import StaticArbitrageAccumulator


@dataclass(frozen=True, kw_only=True)
class PCAUniverseBlock:
    """One independently simulated path block and its global row interval."""

    start: int
    stop: int
    universe: dict[str, np.ndarray]


@dataclass(frozen=True, kw_only=True)
class PCAMatchedUniverseBlock:
    """Expiry and incremental markets derived from one simulated path block."""

    start: int
    stop: int
    expiry: dict[str, np.ndarray]
    incremental: dict[str, np.ndarray]


@dataclass
class PCAUniverseStreamDiagnostics:
    """Mutable online diagnostics finalized after a universe stream is consumed."""

    wing_hits: WingHitDiagnostics = WingHitDiagnostics(
        queries=0, left_hits=0, right_hits=0
    )
    minimum_log_spot: float = math.inf
    maximum_log_spot: float = -math.inf
    importance_accumulator: dict[str, np.ndarray | float] | None = None
    importance_component_counts: np.ndarray | None = None
    minimum_log_weight: float = math.inf
    maximum_log_weight: float = -math.inf
    _path_geometry: TerminalPathDiagnostics | None = field(
        default=None, init=False, repr=False
    )

    def add_wing_hits(self, value: WingHitDiagnostics) -> None:
        self.wing_hits = self.wing_hits + value


def traded_nms_indices(grid_strikes: int, num_traded_nms: int) -> np.ndarray:
    """Return the deterministic evenly spaced subset of the training grid."""

    verify_inp(
        isinstance(grid_strikes, int)
        and isinstance(num_traded_nms, int)
        and 0 < num_traded_nms <= grid_strikes,
        lambda: f"invalid grid/traded strike counts {grid_strikes}/{num_traded_nms}",
    )
    indices = np.rint(
        np.linspace(0, grid_strikes - 1, num_traded_nms)
    ).astype(np.int64)
    verify_inp(
        np.unique(indices).size == num_traded_nms,
        lambda: f"deterministic traded-grid selection is not unique: {indices}",
    )
    return indices


def default_static_strike_mesh(points: int = 201) -> np.ndarray:
    """Dense fixed relative-strike mesh containing ATM exactly."""

    verify_inp(
        isinstance(points, int) and points >= 5 and points % 2 == 1,
        lambda: f"static strike mesh requires an odd integer >=5, got {points!r}",
    )
    half = points // 2
    return np.concatenate(
        (
            np.linspace(0.5, 1.0, half, endpoint=False),
            np.linspace(1.0, 1.5, half + 1),
        )
    )


def _update_importance_accumulator(
    diagnostics: PCAUniverseStreamDiagnostics,
    *,
    importance_sample,
) -> None:
    log_weight = importance_sample.log_weights
    weight = np.exp(log_weight)
    log_spot = importance_sample.paths[..., 0].astype(np.float64, copy=False)
    realized_variance = np.square(np.diff(log_spot, axis=1)).sum(axis=1)
    block: dict[str, np.ndarray | float] = {
        "weight_sum": float(weight.sum()),
        "weight_square_sum": float(np.square(weight).sum()),
        "reference_log_spot_sum": importance_sample.reference_log_spot_sum,
        "reference_log_spot_square_sum": (
            importance_sample.reference_log_spot_square_sum
        ),
        "weighted_log_spot_sum": (weight[:, None] * log_spot).sum(axis=0),
        "weighted_log_spot_square_sum": (
            weight[:, None] * np.square(log_spot)
        ).sum(axis=0),
        "reference_rv_sum": importance_sample.reference_realized_variance_sum,
        "reference_rv_square_sum": (
            importance_sample.reference_realized_variance_square_sum
        ),
        "weighted_rv_sum": float(np.dot(weight, realized_variance)),
        "weighted_rv_square_sum": float(
            np.dot(weight, np.square(realized_variance))
        ),
    }
    if diagnostics.importance_accumulator is None:
        diagnostics.importance_accumulator = block
    else:
        for name, value in block.items():
            diagnostics.importance_accumulator[name] += value
    assert diagnostics.importance_component_counts is not None
    diagnostics.importance_component_counts += np.bincount(
        importance_sample.proposal_components,
        minlength=diagnostics.importance_component_counts.size,
    )
    diagnostics.minimum_log_weight = min(
        diagnostics.minimum_log_weight, float(np.min(log_weight))
    )
    diagnostics.maximum_log_weight = max(
        diagnostics.maximum_log_weight, float(np.max(log_weight))
    )


def iter_pca_universe_blocks(
    experiment: ImpliedPCAExperiment,
    *,
    decoder: PCATotalVarianceDecoder,
    sampler: ImpliedPCASampler,
    diagnostics: PCAUniverseStreamDiagnostics,
    static_accumulator: StaticArbitrageAccumulator | None = None,
    static_strikes: np.ndarray | None = None,
    verbose: Context = Context.all,
) -> Iterator[PCAUniverseBlock]:
    """Yield coherent path blocks without materializing a full second copy."""

    verify_inp(
        decoder.pca is sampler.pca,
        "decoder and sampler must refer to the same fitted PCA representation",
    )
    expiries = decoder.pca.expiries
    T, grid_strikes = decoder.pca.surface_shape
    verify_inp(
        grid_strikes == experiment.grid_strikes
        and decoder.pca.num_components == experiment.num_surface_state,
        lambda: (
            "PCA geometry differs from experiment: "
            f"grid {grid_strikes}/{experiment.grid_strikes}, states "
            f"{decoder.pca.num_components}/{experiment.num_surface_state}"
        ),
    )
    expiry_offsets, maximum_expiry_time_error = expiry_step_offsets(
        expiries, dt=experiment.dt
    )
    trading_times = experiment.trading_times
    stride = experiment.trading_step_stride
    option_return_offset = (
        int(expiry_offsets[-1])
        if experiment.return_horizon == "expiry"
        else stride
    )
    last_action = (trading_times - 1) * stride
    simulated_steps = last_action + max(option_return_offset, stride) + 1
    traded_indices = traded_nms_indices(
        grid_strikes, experiment.num_traded_nms
    )
    traded_nms = decoder.pca.nms[:, traded_indices]
    trading_indices = np.arange(trading_times, dtype=np.int64) * stride

    importance_config = experiment.importance_sampling
    target_steps = (
        _importance_target_steps(
            trading_times=trading_times,
            trading_step_stride=stride,
            return_offsets=(
                expiry_offsets
                if experiment.return_horizon == "expiry"
                else np.asarray([option_return_offset], dtype=np.int64)
            ),
            action_count=importance_config.target_action_count,
        )
        if importance_config is not None
        else ()
    )
    importance_rng = np.random.default_rng(experiment.seed + 8_000_003)
    if importance_config is not None:
        diagnostics.importance_component_counts = np.zeros(
            len(target_steps) + 1, dtype=np.int64
        )

    if static_accumulator is not None:
        verify_inp(
            static_strikes is not None,
            "a static strike mesh is required with a static accumulator",
        )
        static_strikes = np.asarray(static_strikes, dtype=np.float64)
        verify_inp(
            np.array_equal(static_accumulator.strikes, static_strikes)
            and np.array_equal(static_accumulator.expiries, expiries),
            "static accumulator geometry differs from the PCA decoder",
        )

    for start in range(0, experiment.num_samples, experiment.block_size):
        stop = min(start + experiment.block_size, experiment.num_samples)
        block_samples = stop - start
        if importance_config is None:
            importance_sample = None
            xstate_path = sampler.free_sample_xstates(
                num_steps=simulated_steps,
                num_samples=block_samples,
                dt=experiment.dt,
                bandwidth=1.0,
            )
        else:
            importance_sample = sampler.importance_sample_xstates(
                num_steps=simulated_steps,
                num_samples=block_samples,
                dt=experiment.dt,
                target_steps=target_steps,
                config=importance_config,
                proposal_rng=importance_rng,
            )
            xstate_path = importance_sample.paths
            _update_importance_accumulator(
                diagnostics, importance_sample=importance_sample
            )
        if not np.all(np.isfinite(xstate_path)):
            raise FloatingPointError("PCA state sampler produced non-finite paths")
        diagnostics.minimum_log_spot = min(
            diagnostics.minimum_log_spot,
            float(np.min(xstate_path[..., 0])),
        )
        diagnostics.maximum_log_spot = max(
            diagnostics.maximum_log_spot,
            float(np.max(xstate_path[..., 0])),
        )

        action_scores = xstate_path[:, trading_indices, 1:]
        action_log_spot = xstate_path[:, trading_indices, 0]
        action_spot = np.exp(action_log_spot)
        maturity_query = expiries[:, None]
        nms_query = traded_nms
        relative_strikes = decoder.relative_strikes_from_nms(
            action_scores, maturity_query, nms_query
        )
        normalized_calls, call_hits = decoder.normalized_call_prices(
            action_scores, maturity_query, relative_strikes
        )
        cash_strikes = relative_strikes * action_spot[:, :, None, None]
        deltas, delta_hits = decoder.cash_deltas(
            action_scores,
            action_spot[:, :, None, None],
            maturity_query,
            cash_strikes,
        )
        diagnostics.add_wing_hits(call_hits + delta_hits)

        calls_tp1 = None
        if experiment.return_horizon == "incremental":
            next_indices = trading_indices + option_return_offset
            next_scores = xstate_path[:, next_indices, 1:]
            next_spot = np.exp(xstate_path[:, next_indices, 0])
            remaining_maturity = (
                expiries - option_return_offset * experiment.dt
            )
            alive = expiry_offsets > option_return_offset
            calls_tp1 = np.zeros_like(normalized_calls)
            if np.any(alive):
                relative_strikes_tp1 = (
                    cash_strikes[:, :, alive, :]
                    / next_spot[:, :, None, None]
                )
                alive_calls, repricing_hits = decoder.normalized_call_prices(
                    next_scores,
                    remaining_maturity[alive][None, None, :, None],
                    relative_strikes_tp1,
                )
                calls_tp1[:, :, alive, :] = alive_calls
                diagnostics.add_wing_hits(repricing_hits)

        if static_accumulator is not None:
            assert static_strikes is not None
            static_prices, static_hits = decoder.normalized_call_prices(
                action_scores,
                expiries[:, None],
                static_strikes[None, :],
            )
            static_nms = decoder.nms_from_relative_strikes(
                action_scores,
                expiries[:, None],
                static_strikes[None, :],
            )
            diagnostics.add_wing_hits(static_hits)
            static_accumulator.update(
                static_prices,
                normalized_moneyness=static_nms,
                log_path_weights=(
                    None
                    if importance_sample is None
                    else importance_sample.log_weights
                ),
            )

        universe = build_terminal_path_universe(
            xstate_path=xstate_path,
            trading_times=trading_times,
            trading_step_stride=stride,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            normalized_strikes=relative_strikes,
            normalized_calls=normalized_calls,
            normalized_deltas=deltas,
            normalized_calls_tp1=calls_tp1,
            option_return_step_offset=option_return_offset,
            require_bounded_deltas=False,
        )
        if importance_sample is not None:
            universe["log_path_weight"] = importance_sample.log_weights
            universe["importance_component"] = (
                importance_sample.proposal_components
            )
        for name, value in universe.items():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"generated PCA universe field {name!r} is non-finite"
                )
        verbose.write(
            lambda start=start, stop=stop: (
                f"Generated PCA {experiment.mode}/{experiment.option_return_horizon} "
                f"paths {start:,}:{stop:,}/{experiment.num_samples:,}"
            )
        )
        yield PCAUniverseBlock(start=start, stop=stop, universe=universe)

    diagnostics._path_geometry = TerminalPathDiagnostics(
        mode=experiment.mode,
        option_return_horizon=experiment.option_return_horizon,
        trading_times=trading_times,
        trading_step_stride=stride,
        simulated_state_steps=simulated_steps,
        expiry_step_offsets=tuple(int(value) for value in expiry_offsets),
        dt=experiment.dt,
        maximum_expiry_time_error=maximum_expiry_time_error,
        minimum_log_spot=diagnostics.minimum_log_spot,
        maximum_log_spot=diagnostics.maximum_log_spot,
        importance_sampling=(
            None
            if importance_config is None
            else _importance_control_diagnostics(
                diagnostics.importance_accumulator,
                samples=experiment.num_samples,
                config=importance_config,
                target_steps=target_steps,
                component_counts=diagnostics.importance_component_counts,
                minimum_log_weight=diagnostics.minimum_log_weight,
                maximum_log_weight=diagnostics.maximum_log_weight,
            )
        ),
    )


def iter_pca_matched_universe_blocks(
    experiment: ImpliedPCAExperiment,
    *,
    decoder: PCATotalVarianceDecoder,
    sampler: ImpliedPCASampler,
    diagnostics: PCAUniverseStreamDiagnostics,
    static_accumulator: StaticArbitrageAccumulator | None = None,
    static_strikes: np.ndarray | None = None,
    verbose: Context = Context.all,
) -> Iterator[PCAMatchedUniverseBlock]:
    """Yield both return horizons from one sufficiently long coherent path.

    The experiment's ``return_horizon`` is deliberately ignored here.  It is
    not a simulation input: the same ten decoded trading surfaces and daily
    spot path determine the ten expiry cohorts and nine incremental intervals.
    """

    verify_inp(
        experiment.num_trading_times == 10,
        "matched reported markets require ten trading timestamps",
    )
    verify_inp(
        decoder.pca is sampler.pca,
        "decoder and sampler must refer to the same fitted PCA representation",
    )
    expiries = decoder.pca.expiries
    T, grid_strikes = decoder.pca.surface_shape
    verify_inp(
        grid_strikes == experiment.grid_strikes
        and decoder.pca.num_components == experiment.num_surface_state,
        "PCA geometry differs from the versioned experiment",
    )
    expiry_offsets, maximum_expiry_time_error = expiry_step_offsets(
        expiries, dt=experiment.dt
    )
    action_times = experiment.num_trading_times
    stride = experiment.trading_step_stride
    trading_indices = np.arange(action_times, dtype=np.int64) * stride
    simulated_steps = int(trading_indices[-1] + expiry_offsets[-1] + 1)
    traded_indices = traded_nms_indices(
        grid_strikes, experiment.num_traded_nms
    )
    traded_nms = decoder.pca.nms[:, traded_indices]

    importance_config = experiment.importance_sampling
    target_steps = (
        _importance_target_steps(
            trading_times=action_times,
            trading_step_stride=stride,
            return_offsets=np.unique(
                np.concatenate((expiry_offsets, np.asarray([stride])))
            ),
            action_count=importance_config.target_action_count,
        )
        if importance_config is not None
        else ()
    )
    importance_rng = np.random.default_rng(experiment.seed + 8_000_003)
    if importance_config is not None:
        diagnostics.importance_component_counts = np.zeros(
            len(target_steps) + 1, dtype=np.int64
        )
    if static_accumulator is not None:
        verify_inp(
            static_strikes is not None,
            "a static strike mesh is required with a static accumulator",
        )
        static_strikes = np.asarray(static_strikes, dtype=np.float64)
        verify_inp(
            np.array_equal(static_accumulator.strikes, static_strikes)
            and np.array_equal(static_accumulator.expiries, expiries),
            "static accumulator geometry differs from the PCA decoder",
        )

    for start in range(0, experiment.num_samples, experiment.block_size):
        stop = min(start + experiment.block_size, experiment.num_samples)
        block_samples = stop - start
        if importance_config is None:
            importance_sample = None
            xstate_path = sampler.free_sample_xstates(
                num_steps=simulated_steps,
                num_samples=block_samples,
                dt=experiment.dt,
                bandwidth=1.0,
            )
        else:
            importance_sample = sampler.importance_sample_xstates(
                num_steps=simulated_steps,
                num_samples=block_samples,
                dt=experiment.dt,
                target_steps=target_steps,
                config=importance_config,
                proposal_rng=importance_rng,
            )
            xstate_path = importance_sample.paths
            _update_importance_accumulator(
                diagnostics, importance_sample=importance_sample
            )
        if not np.all(np.isfinite(xstate_path)):
            raise FloatingPointError("PCA state sampler produced non-finite paths")
        diagnostics.minimum_log_spot = min(
            diagnostics.minimum_log_spot, float(np.min(xstate_path[..., 0]))
        )
        diagnostics.maximum_log_spot = max(
            diagnostics.maximum_log_spot, float(np.max(xstate_path[..., 0]))
        )

        action_scores = xstate_path[:, trading_indices, 1:]
        action_spot = np.exp(xstate_path[:, trading_indices, 0])
        relative_strikes = decoder.relative_strikes_from_nms(
            action_scores, expiries[:, None], traded_nms
        )
        normalized_calls, call_hits = decoder.normalized_call_prices(
            action_scores, expiries[:, None], relative_strikes
        )
        cash_strikes = relative_strikes * action_spot[:, :, None, None]
        deltas, delta_hits = decoder.cash_deltas(
            action_scores,
            action_spot[:, :, None, None],
            expiries[:, None],
            cash_strikes,
        )
        diagnostics.add_wing_hits(call_hits + delta_hits)

        incremental_calls = np.zeros_like(normalized_calls[:, :-1])
        alive = expiry_offsets > stride
        if np.any(alive):
            next_scores = action_scores[:, 1:]
            next_spot = action_spot[:, 1:]
            next_relative_strikes = (
                cash_strikes[:, :-1, alive]
                / next_spot[:, :, None, None]
            )
            alive_calls, repricing_hits = decoder.normalized_call_prices(
                next_scores,
                (expiries[alive] - stride * experiment.dt)[None, None, :, None],
                next_relative_strikes,
            )
            incremental_calls[:, :, alive] = alive_calls
            diagnostics.add_wing_hits(repricing_hits)

        if static_accumulator is not None:
            assert static_strikes is not None
            static_prices, static_hits = decoder.normalized_call_prices(
                action_scores,
                expiries[:, None],
                static_strikes[None, :],
            )
            static_nms = decoder.nms_from_relative_strikes(
                action_scores,
                expiries[:, None],
                static_strikes[None, :],
            )
            diagnostics.add_wing_hits(static_hits)
            static_accumulator.update(
                static_prices,
                normalized_moneyness=static_nms,
                log_path_weights=(
                    None
                    if importance_sample is None
                    else importance_sample.log_weights
                ),
            )

        expiry = build_terminal_path_universe(
            xstate_path=xstate_path,
            trading_times=action_times,
            trading_step_stride=stride,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            normalized_strikes=relative_strikes,
            normalized_calls=normalized_calls,
            normalized_deltas=deltas,
            require_bounded_deltas=False,
        )
        incremental = build_terminal_path_universe(
            xstate_path=xstate_path,
            trading_times=action_times - 1,
            trading_step_stride=stride,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            normalized_strikes=relative_strikes[:, :-1],
            normalized_calls=normalized_calls[:, :-1],
            normalized_deltas=deltas[:, :-1],
            normalized_calls_tp1=incremental_calls,
            option_return_step_offset=stride,
            require_bounded_deltas=False,
        )
        if importance_sample is not None:
            for universe in (expiry, incremental):
                universe["log_path_weight"] = importance_sample.log_weights
                universe["importance_component"] = (
                    importance_sample.proposal_components
                )
        for horizon, universe in (
            ("expiry", expiry),
            ("incremental", incremental),
        ):
            for name, value in universe.items():
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError(
                        f"generated PCA {horizon} field {name!r} is non-finite"
                    )
        verbose.write(
            lambda start=start, stop=stop: (
                f"Generated matched PCA {experiment.trading_frequency} paths "
                f"{start:,}:{stop:,}/{experiment.num_samples:,}"
            )
        )
        yield PCAMatchedUniverseBlock(
            start=start,
            stop=stop,
            expiry=expiry,
            incremental=incremental,
        )

    diagnostics._path_geometry = TerminalPathDiagnostics(
        mode=f"matched-{experiment.trading_frequency}",
        option_return_horizon="expiry+incremental",
        trading_times=action_times,
        trading_step_stride=stride,
        simulated_state_steps=simulated_steps,
        expiry_step_offsets=tuple(int(value) for value in expiry_offsets),
        dt=experiment.dt,
        maximum_expiry_time_error=maximum_expiry_time_error,
        minimum_log_spot=diagnostics.minimum_log_spot,
        maximum_log_spot=diagnostics.maximum_log_spot,
        importance_sampling=(
            None
            if importance_config is None
            else _importance_control_diagnostics(
                diagnostics.importance_accumulator,
                samples=experiment.num_samples,
                config=importance_config,
                target_steps=target_steps,
                component_counts=diagnostics.importance_component_counts,
                minimum_log_weight=diagnostics.minimum_log_weight,
                maximum_log_weight=diagnostics.maximum_log_weight,
            )
        ),
    )


def finalize_stream_diagnostics(
    diagnostics: PCAUniverseStreamDiagnostics,
) -> TerminalPathDiagnostics:
    """Return path diagnostics after fully consuming the block iterator."""

    verify_inp(
        diagnostics._path_geometry is not None,
        "PCA universe stream must be fully consumed before finalization",
    )
    assert diagnostics._path_geometry is not None
    return diagnostics._path_geometry


__all__ = [
    "PCAMatchedUniverseBlock",
    "PCAUniverseBlock",
    "PCAUniverseStreamDiagnostics",
    "default_static_strike_mesh",
    "finalize_stream_diagnostics",
    "iter_pca_matched_universe_blocks",
    "iter_pca_universe_blocks",
    "traded_nms_indices",
]
