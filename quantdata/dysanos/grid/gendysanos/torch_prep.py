"""Prepare terminal-payoff universes from the trained SANOS decoder."""

from __future__ import annotations

import gc
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
)
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig

from .terminal_universe import (
    TerminalPathDiagnostics,
    build_terminal_path_universe,
    expiry_step_offsets,
)


@version("1.0", dependencies=[GaussianSpotImportanceConfig])
@dataclass(frozen=True, kw_only=True)
class GenDysanosExperiment:
    """Logical inputs for one coherent generated DYSANOS market case."""

    generator: str = "dysanos"
    num_samples: int = 100_000
    grid_strikes: int = 100
    num_nms: int = 20
    num_alpha: int | None = None
    use_matrix_exp: bool = True
    bandwidth: float = 1.0
    trading_frequency: str = "daily"
    return_horizon: str = "expiry"
    # Deprecated constructor aliases retained for reproducibility of existing
    # experiment scripts. New reports use the explicit fields above.
    mode: str | None = None
    option_return_horizon: str | None = None
    num_steps: int | None = None
    delta_hedged: bool = True
    cost_rate: float = 1.0e-4
    num_trading_times: int = 10
    dt: float = 1.0 / DAYS_PER_YEAR
    seed: int = 500_000
    block_size: int = 25_000
    importance_sampling: GaussianSpotImportanceConfig | None = None

    def __post_init__(self) -> None:
        if self.mode is not None:
            verify_inp(
                self.mode in {"one-step", "one-step-weekly", "path", "weekly"},
                lambda: f"invalid legacy 'mode' {self.mode!r}",
            )
            object.__setattr__(
                self,
                "trading_frequency",
                "weekly" if self.mode in {"one-step-weekly", "weekly"} else "daily",
            )
        if self.option_return_horizon is not None:
            verify_inp(
                self.option_return_horizon in {"expiry", "one-day"},
                lambda: (
                    "'option_return_horizon' must be 'expiry' or 'one-day', got "
                    f"{self.option_return_horizon!r}"
                ),
            )
            object.__setattr__(
                self,
                "return_horizon",
                "incremental"
                if self.option_return_horizon == "one-day"
                else "expiry",
            )
        if self.mode is not None:
            if self.mode in {"one-step", "one-step-weekly"}:
                object.__setattr__(self, "num_trading_times", 1)
            elif self.num_steps is not None:
                legacy_times = (
                    self.num_steps
                    if self.return_horizon == "incremental"
                    else self.num_steps - 1
                )
                object.__setattr__(self, "num_trading_times", legacy_times)
        elif self.num_steps is not None:
            verify_inp(
                self.num_steps == self.num_trading_times,
                lambda: (
                    "legacy 'num_steps' is only meaningful with legacy 'mode'; "
                    f"got {self.num_steps} versus num_trading_times="
                    f"{self.num_trading_times}"
                ),
            )
        if self.mode is None:
            object.__setattr__(
                self,
                "mode",
                (
                    "one-step-weekly"
                    if self.num_trading_times == 1
                    and self.trading_frequency == "weekly"
                    else "one-step"
                    if self.num_trading_times == 1
                    else "weekly"
                    if self.trading_frequency == "weekly"
                    else "path"
                ),
            )
        if self.option_return_horizon is None:
            object.__setattr__(
                self,
                "option_return_horizon",
                "one-day" if self.return_horizon == "incremental" else "expiry",
            )
        for name in (
            "num_samples",
            "grid_strikes",
            "num_nms",
            "num_trading_times",
            "block_size",
        ):
            value = getattr(self, name)
            verify_inp(value > 0, lambda name=name, value=value: f"'{name}' must be positive, got {value}")
        verify_inp(
            self.generator == "dysanos",
            lambda: f"GenDysanosExperiment requires generator='dysanos', got {self.generator!r}",
        )
        verify_inp(
            self.trading_frequency in {"daily", "weekly"},
            lambda: (
                "'trading_frequency' must be 'daily' or 'weekly', got "
                f"{self.trading_frequency!r}"
            ),
        )
        verify_inp(
            self.return_horizon in {"expiry", "incremental"},
            lambda: (
                "'return_horizon' must be 'expiry' or 'incremental', got "
                f"{self.return_horizon!r}"
            ),
        )
        verify_inp(
            isinstance(self.delta_hedged, bool),
            lambda: f"'delta_hedged' must be a bool, got {type(self.delta_hedged)}",
        )
        verify_inp(
            self.cost_rate in {0.0, 1.0e-4, 1.0e-3},
            lambda: (
                "'cost_rate' must be zero, one basis point, or ten basis "
                f"points, got {self.cost_rate}"
            ),
        )
        verify_inp(
            self.return_horizon == "expiry" or self.num_trading_times >= 2,
            "incremental returns require at least two trading timestamps",
        )
        verify_inp(
            self.num_alpha is None or self.num_alpha > 0,
            lambda: f"'num_alpha' must be None or positive, got {self.num_alpha}",
        )
        verify_inp(
            0.0 <= self.bandwidth <= 1.0,
            lambda: f"'bandwidth' must be in [0, 1], got {self.bandwidth}",
        )
        verify_inp(
            math.isfinite(self.dt) and self.dt > 0.0,
            lambda: f"'dt' must be finite and positive, got {self.dt}",
        )
        verify_inp(isinstance(self.seed, int), lambda: f"'seed' must be an int, got {self.seed!r}")
        if self.importance_sampling is not None:
            verify_inp(
                self.use_matrix_exp and self.bandwidth == 1.0,
                lambda: (
                    "importance sampling requires use_matrix_exp=True, "
                    "and bandwidth=1; got "
                    f"{self.use_matrix_exp}/{self.bandwidth}"
                ),
            )

    @property
    def trading_times(self) -> int:
        """Number of return start times represented in the universe."""

        return (
            self.num_trading_times
            if self.return_horizon == "expiry"
            else self.num_trading_times - 1
        )

    @property
    def trading_step_stride(self) -> int:
        """Spacing of consecutive action dates on the daily state path."""

        return 5 if self.trading_frequency == "weekly" else 1

    @property
    def reference_paths(self) -> int:
        """Paths used to fit diagnostic estimators (80% of the universe)."""

        return min(self.num_samples - 1, 4 * self.num_samples // 5)

    @property
    def audit_paths(self) -> int:
        """Held-out paths used only to audit fitted diagnostics."""

        return self.num_samples - self.reference_paths


@dataclass(frozen=True, kw_only=True)
class DysanosMatchedUniverseBlock:
    """Expiry and incremental DYSANOS markets from one simulated path block."""

    start: int
    stop: int
    expiry: dict[str, np.ndarray]
    incremental: dict[str, np.ndarray]


@dataclass
class DysanosUniverseStreamDiagnostics:
    """Online sampling diagnostics for a matched DYSANOS path stream."""

    minimum_log_spot: float = math.inf
    maximum_log_spot: float = -math.inf
    importance_accumulator: dict[str, np.ndarray | float] | None = None
    importance_component_counts: np.ndarray | None = None
    minimum_log_weight: float = math.inf
    maximum_log_weight: float = -math.inf
    incremental_call_roundoff_clip_count: int = 0
    minimum_unclipped_incremental_call: float = math.inf
    _path_geometry: TerminalPathDiagnostics | None = field(
        default=None, init=False, repr=False
    )


def _stabilize_incremental_calls(
    calls,
    *,
    expiry_offsets: np.ndarray,
    stride: int,
    diagnostics: DysanosUniverseStreamDiagnostics,
):
    """Set expired marks and roundoff-negative alive calls to exact zero."""

    import torch

    expired = torch.as_tensor(
        expiry_offsets <= stride,
        dtype=torch.bool,
        device=calls.device,
    )
    if torch.any(expired).item():
        # These values are not endpoint marks: the shared universe builder
        # settles them at exact daily expiry. The interpolation rule assumes
        # that ``dt`` does not cross an expiry and its unused output may be NaN.
        calls[:, :, expired, :] = 0.0
    alive = ~expired
    if torch.any(alive).item():
        alive_calls = calls[:, :, alive, :]
        diagnostics.minimum_unclipped_incremental_call = min(
            diagnostics.minimum_unclipped_incremental_call,
            float(torch.min(alive_calls).item()),
        )
        negative = alive_calls < 0.0
        clip_count = int(torch.count_nonzero(negative).item())
        if clip_count:
            normalized_price_roundoff = (
                DEFAULT_ROUNDOFF_MULTIPLIER * float(np.finfo(np.float32).eps)
            )
            minimum = float(torch.min(alive_calls).item())
            if minimum < -normalized_price_roundoff:
                raise FloatingPointError(
                    "DYSANOS incremental call is negative beyond its normalized "
                    f"float32 band: {minimum:g} < {-normalized_price_roundoff:g}"
                )
            diagnostics.incremental_call_roundoff_clip_count += clip_count
            calls[:, :, alive, :] = torch.clamp_min(alive_calls, 0.0)
    return calls

def _load_generator_components(
    experiment: GenDysanosExperiment,
    *,
    verbose: Context,
):
    """Load the historical grid, trained decoder, and Gaussian state sampler."""

    import torch

    from quantdata.dysanos.grid.torch_gfx import VizConfig
    from quantdata.dysanos.grid.torch_market import (
        DateRange,
        GridConfig,
        GridMarketDefinition,
        PoolConfig,
    )
    from quantdata.dysanos.grid.torch_model import (
        BaseEnvironment,
        FeedForward,
        SanosDecoderConfig,
        SanosModelConfig,
    )
    from quantdata.dysanos.grid.torch_simulator import ARSampler, ARSamplerConfig
    from quantdata.dysanos.grid.torch_train import TrainConfig, train

    grid_config = GridConfig(grid_num_strikes=experiment.grid_strikes)
    market_definition = GridMarketDefinition(
        ticker="SPX",
        date_range=DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30)),
        grid_config=grid_config,
    )
    market = market_definition.load(
        pool_config=PoolConfig(num_workers=4, threading=False),
        override_cache_mode="cacheonly",
        verbose=verbose(1),
    )
    model_config = SanosModelConfig(
        label="ac_state20",
        user_version="0.2",
        group="sm25_100",
        encoder_config=FeedForward.Config(
            activation="SELU",
            num_depth=2,
            num_width=100,
            use_bias=True,
        ),
        decoder_config=SanosDecoderConfig(
            smoothness=0.25,
            num_state=20,
            num_nms=experiment.num_nms,
            nn_config=FeedForward.Config(
                activation="SELU",
                num_depth=2,
                num_width=100,
                use_bias=True,
            ),
        ),
    )
    env = BaseEnvironment(device="cpu", debug=False)
    training_config = TrainConfig(
        num_epochs=100_000,
        lr=1.0e-4,
        final_lr=1.0e-6,
        weight_decay=1.0e-4,
        batch_size=400,
        compile=True,
        epoch_interval=10_000,
        clip_grad_norm=None,
    )
    _, model, _ = train(
        market,
        mdl_config=model_config,
        trn_config=training_config,
        viz_config=VizConfig(num_viz_dates=4),
        just_load="plot",
        env=env,
    )
    # Encoding the historical surfaces only needs the seven input feature
    # fields. SanosModelConfig.torch_data additionally builds dense transition
    # and pricing tensors used by the training loss; for the 100-strike grid
    # those tensors dominate setup time and memory but are irrelevant here.
    feature_names = (
        "mid_vols",
        "xstrikes",
        "expiries",
        "variances",
        "nms",
        "mid_vega",
        "mid_delta",
    )
    feature_data = {
        name: torch.as_tensor(
            np.asarray(market[name]), dtype=env.fdtype, device=env.device
        )
        for name in feature_names
    }
    model.eval()
    with torch.no_grad():
        features = model.build_default_features(feature_data, training=False)
        surface_states = model.encoder(features).detach().cpu().numpy()
    del feature_data, features
    sampler = ARSampler(
        market,
        model,
        surface_states=surface_states,
        config=ARSamplerConfig(
            num_alpha=experiment.num_alpha,
            use_matrix_exp=experiment.use_matrix_exp,
            surface_first=True,
            seed=experiment.seed,
        ),
        verbose=verbose(1),
    )
    return grid_config, market, model, sampler


def _terminal_call_delta(pricer, strikes):
    """Return cash-call delta for the decoder's scale-family distribution.

    For normalized component spot ``x`` and current cash spot ``S``, the
    terminal component is ``S*x``.  Differentiating the smoothed component
    call with respect to ``S`` gives ``x*N(d1)``.  The unsmoothed limit is
    ``x*1{x>k}``.
    """

    import torch

    component_spot = pricer.xstrikes[:, :, None, :]
    weights = pricer.xdensities[:, :, None, :]
    strike = strikes[:, :, :, None]
    if pricer.smoothness is None or pricer.smoothness == 0.0:
        component_delta = component_spot * (component_spot > strike).to(
            component_spot.dtype
        )
    else:
        sqrt_variance = (
            math.sqrt(pricer.smoothness)
            * pricer.sqrtVars[:, :, None, None]
        )
        d1 = -torch.log(strike / component_spot) / sqrt_variance
        d1 = d1 + 0.5 * sqrt_variance
        component_delta = component_spot * torch.special.ndtr(d1)
    delta = (weights * component_delta).sum(dim=-1)
    if not torch.isfinite(delta).all().item():
        raise FloatingPointError("terminal call delta is non-finite")
    tolerance = 1.0e-5
    if torch.any((delta < -tolerance) | (delta > 1.0 + tolerance)).item():
        raise RuntimeError(
            "terminal call delta lies outside [0, 1]: "
            f"range [{delta.min().item():g}, {delta.max().item():g}]"
        )
    return delta.clamp(0.0, 1.0)


def _importance_target_steps(
    *,
    trading_times: int,
    trading_step_stride: int,
    return_offsets: np.ndarray,
    action_count: int,
) -> tuple[int, ...]:
    """Select first, evenly spaced, and last action-return coordinates."""

    count = min(action_count, trading_times)
    action = np.unique(
        np.rint(np.linspace(0, trading_times - 1, count)).astype(np.int64)
    )
    action *= trading_step_stride
    return tuple(
        sorted(
            {
                int(action_step + expiry_step)
                for action_step in action
                for expiry_step in return_offsets
            }
        )
    )


def _importance_control_diagnostics(
    accumulator: dict[str, np.ndarray | float],
    *,
    samples: int,
    config: GaussianSpotImportanceConfig,
    target_steps: tuple[int, ...],
    component_counts: np.ndarray,
    minimum_log_weight: float,
    maximum_log_weight: float,
) -> dict[str, object]:
    """Compare direct-P and likelihood-weighted proposal stock moments."""

    weight_sum = float(accumulator["weight_sum"])
    weight_square_sum = float(accumulator["weight_square_sum"])
    reference_sum = np.asarray(accumulator["reference_log_spot_sum"])
    reference_square_sum = np.asarray(
        accumulator["reference_log_spot_square_sum"]
    )
    weighted_sum = np.asarray(accumulator["weighted_log_spot_sum"])
    weighted_square_sum = np.asarray(
        accumulator["weighted_log_spot_square_sum"]
    )
    reference_mean = reference_sum / samples
    reference_variance = np.maximum(
        reference_square_sum / samples - np.square(reference_mean), 0.0
    )
    weighted_mean = weighted_sum / weight_sum
    weighted_variance = np.maximum(
        weighted_square_sum / weight_sum - np.square(weighted_mean), 0.0
    )
    selected = np.asarray(target_steps, dtype=np.int64)

    reference_rv_mean = float(accumulator["reference_rv_sum"]) / samples
    reference_rv_variance = max(
        float(accumulator["reference_rv_square_sum"]) / samples
        - reference_rv_mean**2,
        0.0,
    )
    weighted_rv_mean = float(accumulator["weighted_rv_sum"]) / weight_sum
    weighted_rv_variance = max(
        float(accumulator["weighted_rv_square_sum"]) / weight_sum
        - weighted_rv_mean**2,
        0.0,
    )
    variance_scale = np.maximum(reference_variance[selected], 1.0e-16)
    return {
        "config": asdict(config),
        "proposal": (
            "mixture of the direct-P Gaussian path and exact conditional "
            "Gaussian bridges with one widened future log-spot marginal"
        ),
        "target_steps": list(target_steps),
        "proposal_component_counts": component_counts.astype(int).tolist(),
        "path_weights": {
            "minimum_log_weight": minimum_log_weight,
            "maximum_log_weight": maximum_log_weight,
            "mean_density": weight_sum / samples,
            "normalization_error": weight_sum / samples - 1.0,
            "independent_path_effective_sample_size": (
                weight_sum**2 / weight_square_sum
            ),
            "effective_sample_fraction": (
                weight_sum**2 / weight_square_sum / samples
            ),
        },
        "log_spot_control_at_target_steps": {
            "direct_P_mean": reference_mean[selected].tolist(),
            "weighted_proposal_mean": weighted_mean[selected].tolist(),
            "direct_P_variance": reference_variance[selected].tolist(),
            "weighted_proposal_variance": weighted_variance[selected].tolist(),
            "maximum_absolute_mean_difference": float(
                np.max(np.abs(weighted_mean[selected] - reference_mean[selected]))
            ),
            "maximum_mean_difference_in_direct_P_std": float(
                np.max(
                    np.abs(weighted_mean[selected] - reference_mean[selected])
                    / np.sqrt(variance_scale)
                )
            ),
            "maximum_relative_variance_difference": float(
                np.max(
                    np.abs(weighted_variance[selected] - reference_variance[selected])
                    / variance_scale
                )
            ),
        },
        "realized_log_variance_control": {
            "definition": "sum of squared daily log-spot changes along each path",
            "direct_P_mean": reference_rv_mean,
            "weighted_proposal_mean": weighted_rv_mean,
            "direct_P_variance": reference_rv_variance,
            "weighted_proposal_variance": weighted_rv_variance,
            "relative_mean_difference": (
                (weighted_rv_mean - reference_rv_mean) / reference_rv_mean
                if reference_rv_mean > 0.0
                else math.nan
            ),
            "relative_variance_difference": (
                (weighted_rv_variance - reference_rv_variance)
                / reference_rv_variance
                if reference_rv_variance > 0.0
                else math.nan
            ),
        },
    }


def _update_dysanos_importance_diagnostics(
    diagnostics: DysanosUniverseStreamDiagnostics,
    *,
    importance_sample,
) -> None:
    """Accumulate path-likelihood and direct-P control moments blockwise."""

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


def iter_dysanos_matched_universe_blocks(
    experiment: GenDysanosExperiment,
    *,
    diagnostics: DysanosUniverseStreamDiagnostics,
    verbose: Context = Context.all,
) -> Iterator[DysanosMatchedUniverseBlock]:
    """Yield both required return horizons from one coherent DYSANOS path."""

    import torch

    verify_inp(
        experiment.num_trading_times == 10,
        "matched reported markets require ten trading timestamps",
    )
    grid_config, market, model, sampler = _load_generator_components(
        experiment, verbose=verbose
    )
    sampler.rng = np.random.default_rng(experiment.seed)
    expiries = np.asarray(
        market.grid_market_def.grid_config.grid_expiries, dtype=np.float32
    )
    expiry_offsets, maximum_expiry_time_error = expiry_step_offsets(
        expiries, dt=experiment.dt
    )
    action_times = experiment.num_trading_times
    stride = experiment.trading_step_stride
    trading_indices = np.arange(action_times, dtype=np.int64) * stride
    simulated_steps = int(trading_indices[-1] + expiry_offsets[-1] + 1)
    T, N = grid_config.TN[0], experiment.num_nms
    verify_inp(expiries.shape == (T,), "DYSANOS expiry geometry is inconsistent")

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

    model.eval()
    with torch.no_grad():
        for start in range(0, experiment.num_samples, experiment.block_size):
            stop = min(start + experiment.block_size, experiment.num_samples)
            block_samples = stop - start
            if importance_config is None:
                importance_sample = None
                xstate_path = sampler.free_sample_xstates(
                    num_steps=simulated_steps,
                    num_samples=block_samples,
                    dt=experiment.dt,
                    bandwidth=experiment.bandwidth,
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
                _update_dysanos_importance_diagnostics(
                    diagnostics, importance_sample=importance_sample
                )
            if not np.all(np.isfinite(xstate_path)):
                raise FloatingPointError("DYSANOS sampler produced non-finite paths")
            diagnostics.minimum_log_spot = min(
                diagnostics.minimum_log_spot,
                float(np.min(xstate_path[..., 0])),
            )
            diagnostics.maximum_log_spot = max(
                diagnostics.maximum_log_spot,
                float(np.max(xstate_path[..., 0])),
            )

            action_state = torch.as_tensor(
                xstate_path[:, trading_indices, 1:].reshape(
                    block_samples * action_times, -1
                ),
                dtype=model.env.fdtype,
                device=model.env.device,
            )
            torch_expiries = torch.as_tensor(
                expiries[None, :],
                dtype=model.env.fdtype,
                device=model.env.device,
            )
            pricer = model.decoder(
                state=action_state,
                expiries=torch_expiries,
                training=False,
            )
            calls_tensor = pricer(pricer.strikes, training=False)
            strikes_tensor = pricer.strikes.reshape(
                block_samples, action_times, T, N
            )
            calls_tensor = calls_tensor.reshape(
                block_samples, action_times, T, N
            )
            deltas_tensor = _terminal_call_delta(
                pricer, pricer.strikes
            ).reshape(block_samples, action_times, T, N)

            next_state = torch.as_tensor(
                xstate_path[:, trading_indices[1:], 1:].reshape(
                    block_samples * (action_times - 1), -1
                ),
                dtype=model.env.fdtype,
                device=model.env.device,
            )
            next_pricer = model.decoder(
                state=next_state,
                expiries=torch_expiries,
                training=False,
            )
            spot_ratio = np.exp(
                xstate_path[:, trading_indices[:-1], 0]
                - xstate_path[:, trading_indices[1:], 0]
            )
            incremental_strikes = (
                strikes_tensor[:, :-1]
                * torch.as_tensor(
                    spot_ratio[:, :, None, None],
                    dtype=model.env.fdtype,
                    device=model.env.device,
                )
            )
            incremental_calls_tensor = next_pricer(
                incremental_strikes.reshape(
                    block_samples * (action_times - 1), T, N
                ),
                dt=stride * experiment.dt,
                training=False,
            ).reshape(block_samples, action_times - 1, T, N)
            incremental_calls_tensor = _stabilize_incremental_calls(
                incremental_calls_tensor,
                expiry_offsets=expiry_offsets,
                stride=stride,
                diagnostics=diagnostics,
            )
            for name, value in (
                ("calls", calls_tensor),
                ("strikes", strikes_tensor),
                ("deltas", deltas_tensor),
                ("incremental_calls", incremental_calls_tensor),
            ):
                if not torch.isfinite(value).all().item():
                    raise FloatingPointError(
                        f"DYSANOS decoder produced non-finite {name}"
                    )

            strikes = strikes_tensor.detach().cpu().numpy()
            calls = calls_tensor.detach().cpu().numpy()
            deltas = deltas_tensor.detach().cpu().numpy()
            incremental_calls = incremental_calls_tensor.detach().cpu().numpy()
            expiry = build_terminal_path_universe(
                xstate_path=xstate_path,
                trading_times=action_times,
                trading_step_stride=stride,
                expiries=expiries,
                expiry_offsets=expiry_offsets,
                normalized_strikes=strikes,
                normalized_calls=calls,
                normalized_deltas=deltas,
            )
            incremental = build_terminal_path_universe(
                xstate_path=xstate_path,
                trading_times=action_times - 1,
                trading_step_stride=stride,
                expiries=expiries,
                expiry_offsets=expiry_offsets,
                normalized_strikes=strikes[:, :-1],
                normalized_calls=calls[:, :-1],
                normalized_deltas=deltas[:, :-1],
                normalized_calls_tp1=incremental_calls,
                option_return_step_offset=stride,
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
                            f"generated DYSANOS {horizon} field {name!r} is non-finite"
                        )
            verbose.write(
                lambda start=start, stop=stop: (
                    f"Generated matched DYSANOS {experiment.trading_frequency} "
                    f"paths {start:,}:{stop:,}/{experiment.num_samples:,}"
                )
            )
            yield DysanosMatchedUniverseBlock(
                start=start,
                stop=stop,
                expiry=expiry,
                incremental=incremental,
            )
            del (
                xstate_path,
                action_state,
                next_state,
                pricer,
                next_pricer,
                calls_tensor,
                strikes_tensor,
                deltas_tensor,
                incremental_calls_tensor,
                calls,
                strikes,
                deltas,
                incremental_calls,
                expiry,
                incremental,
                importance_sample,
            )
            gc.collect()

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


def finalize_dysanos_stream_diagnostics(
    diagnostics: DysanosUniverseStreamDiagnostics,
) -> TerminalPathDiagnostics:
    """Return matched path diagnostics after fully consuming the stream."""

    verify_inp(
        diagnostics._path_geometry is not None,
        "DYSANOS universe stream must be fully consumed before finalization",
    )
    assert diagnostics._path_geometry is not None
    return diagnostics._path_geometry


def prepare_terminal_path_experiment(
    experiment: GenDysanosExperiment,
    *,
    verbose: Context = Context.all,
) -> tuple[dict[str, np.ndarray], TerminalPathDiagnostics]:
    """Generate terminal payoffs from one extended state path per sample."""

    import torch

    grid_config, market, model, sampler = _load_generator_components(
        experiment, verbose=verbose
    )
    sampler.rng = np.random.default_rng(experiment.seed)
    expiries = np.asarray(
        market.grid_market_def.grid_config.grid_expiries, dtype=np.float32
    )
    offsets, maximum_expiry_time_error = expiry_step_offsets(
        expiries,
        dt=experiment.dt,
    )
    trading_times = experiment.trading_times
    trading_step_stride = experiment.trading_step_stride
    option_return_step_offset = (
        int(offsets[-1])
        if experiment.return_horizon == "expiry"
        else trading_step_stride
    )
    simulated_state_steps = (
        (trading_times - 1) * trading_step_stride
        + option_return_step_offset
        + 1
    )
    T, N = grid_config.TN[0], experiment.num_nms
    verify_inp(
        expiries.shape == (T,),
        lambda: f"market expiries have shape {expiries.shape}, expected {(T,)}",
    )
    importance_config = experiment.importance_sampling
    importance_target_steps = (
        _importance_target_steps(
            trading_times=trading_times,
            trading_step_stride=trading_step_stride,
            return_offsets=(
                offsets
                if experiment.return_horizon == "expiry"
                else np.asarray([option_return_step_offset], dtype=np.int64)
            ),
            action_count=importance_config.target_action_count,
        )
        if importance_config is not None
        else ()
    )
    importance_rng = np.random.default_rng(experiment.seed + 8_000_003)
    importance_accumulator: dict[str, np.ndarray | float] | None = None
    importance_component_counts = (
        np.zeros(len(importance_target_steps) + 1, dtype=np.int64)
        if importance_config is not None
        else None
    )
    minimum_log_weight = math.inf
    maximum_log_weight = -math.inf
    universe: dict[str, np.ndarray] | None = None
    minimum_log_spot = math.inf
    maximum_log_spot = -math.inf
    model.eval()
    with torch.no_grad():
        for start in range(0, experiment.num_samples, experiment.block_size):
            end = min(start + experiment.block_size, experiment.num_samples)
            block_samples = end - start
            if importance_config is None:
                importance_sample = None
                xstate_path = sampler.free_sample_xstates(
                    num_steps=simulated_state_steps,
                    num_samples=block_samples,
                    dt=experiment.dt,
                    bandwidth=experiment.bandwidth,
                )
            else:
                importance_sample = sampler.importance_sample_xstates(
                    num_steps=simulated_state_steps,
                    num_samples=block_samples,
                    dt=experiment.dt,
                    target_steps=importance_target_steps,
                    config=importance_config,
                    proposal_rng=importance_rng,
                )
                xstate_path = importance_sample.paths
                log_weight = importance_sample.log_weights
                weight = np.exp(log_weight)
                log_spot = xstate_path[..., 0].astype(np.float64, copy=False)
                realized_variance = np.square(
                    np.diff(log_spot, axis=1)
                ).sum(axis=1)
                block_accumulator: dict[str, np.ndarray | float] = {
                    "weight_sum": float(weight.sum()),
                    "weight_square_sum": float(np.square(weight).sum()),
                    "reference_log_spot_sum": (
                        importance_sample.reference_log_spot_sum
                    ),
                    "reference_log_spot_square_sum": (
                        importance_sample.reference_log_spot_square_sum
                    ),
                    "weighted_log_spot_sum": (
                        (weight[:, None] * log_spot).sum(axis=0)
                    ),
                    "weighted_log_spot_square_sum": (
                        (weight[:, None] * np.square(log_spot)).sum(axis=0)
                    ),
                    "reference_rv_sum": (
                        importance_sample.reference_realized_variance_sum
                    ),
                    "reference_rv_square_sum": (
                        importance_sample.reference_realized_variance_square_sum
                    ),
                    "weighted_rv_sum": float(
                        np.dot(weight, realized_variance)
                    ),
                    "weighted_rv_square_sum": float(
                        np.dot(weight, np.square(realized_variance))
                    ),
                }
                if importance_accumulator is None:
                    importance_accumulator = block_accumulator
                else:
                    for name, value in block_accumulator.items():
                        importance_accumulator[name] += value
                assert importance_component_counts is not None
                importance_component_counts += np.bincount(
                    importance_sample.proposal_components,
                    minlength=importance_component_counts.size,
                )
                minimum_log_weight = min(
                    minimum_log_weight, float(np.min(log_weight))
                )
                maximum_log_weight = max(
                    maximum_log_weight, float(np.max(log_weight))
                )
            if not np.all(np.isfinite(xstate_path)):
                raise FloatingPointError("state sampler produced non-finite values")
            minimum_log_spot = min(
                minimum_log_spot,
                float(np.min(xstate_path[..., 0])),
            )
            maximum_log_spot = max(
                maximum_log_spot,
                float(np.max(xstate_path[..., 0])),
            )
            trading_indices = (
                np.arange(trading_times, dtype=np.int64)
                * trading_step_stride
            )
            state = torch.as_tensor(
                xstate_path[
                    :,
                    trading_indices,
                    1:,
                ].reshape(
                    block_samples * trading_times,
                    -1,
                ),
                dtype=model.env.fdtype,
                device=model.env.device,
            )
            torch_expiries = torch.as_tensor(
                expiries[None, :], dtype=model.env.fdtype, device=model.env.device
            )
            pricer = model.decoder(
                state=state,
                expiries=torch_expiries,
                training=False,
            )
            calls = pricer(pricer.strikes, training=False)
            strikes = pricer.strikes.reshape(
                block_samples,
                trading_times,
                T,
                N,
            )
            deltas = _terminal_call_delta(pricer, pricer.strikes).reshape(
                block_samples,
                trading_times,
                T,
                N,
            )
            calls = calls.reshape(block_samples, trading_times, T, N)
            calls_tp1 = None
            next_pricer = None
            next_state = None
            if experiment.return_horizon == "incremental":
                next_state = torch.as_tensor(
                    xstate_path[
                        :,
                        trading_indices + option_return_step_offset,
                        1:,
                    ].reshape(block_samples * trading_times, -1),
                    dtype=model.env.fdtype,
                    device=model.env.device,
                )
                next_pricer = model.decoder(
                    state=next_state,
                    expiries=torch_expiries,
                    training=False,
                )
                spot_ratio = np.exp(
                    xstate_path[:, trading_indices, 0]
                    - xstate_path[
                        :,
                        trading_indices + option_return_step_offset,
                        0,
                    ]
                )
                torch_spot_ratio = torch.as_tensor(
                    spot_ratio[:, :, None, None],
                    dtype=model.env.fdtype,
                    device=model.env.device,
                )
                strikes_tp1 = strikes * torch_spot_ratio
                calls_tp1 = next_pricer(
                    strikes_tp1.reshape(block_samples * trading_times, T, N),
                    dt=option_return_step_offset * experiment.dt,
                    training=False,
                ).reshape(block_samples, trading_times, T, N)
                if not torch.isfinite(calls_tp1).all().item():
                    raise FloatingPointError(
                        "incremental option repricing produced non-finite calls"
                    )
            block = build_terminal_path_universe(
                xstate_path=xstate_path,
                trading_times=trading_times,
                trading_step_stride=trading_step_stride,
                expiries=expiries,
                expiry_offsets=offsets,
                normalized_strikes=strikes.detach().cpu().numpy(),
                normalized_calls=calls.detach().cpu().numpy(),
                normalized_deltas=deltas.detach().cpu().numpy(),
                normalized_calls_tp1=(
                    calls_tp1.detach().cpu().numpy()
                    if calls_tp1 is not None
                    else None
                ),
                option_return_step_offset=option_return_step_offset,
            )
            if importance_sample is not None:
                block["log_path_weight"] = importance_sample.log_weights
                block["importance_component"] = (
                    importance_sample.proposal_components
                )
            if universe is None:
                constant_fields = {"expiries", "option_expired"}
                universe = {
                    name: (
                        value.copy()
                        if name in constant_fields
                        else np.empty(
                            (experiment.num_samples, *value.shape[1:]),
                            dtype=value.dtype,
                        )
                    )
                    for name, value in block.items()
                }
            for name, value in block.items():
                if name not in constant_fields:
                    universe[name][start:end] = value
            verbose.write(
                lambda start=start, end=end: (
                    f"Generated {experiment.trading_frequency}/"
                    f"{experiment.return_horizon} "
                    "paths "
                    f"{start:,}:{end:,}/{experiment.num_samples:,}"
                )
            )
            del (
                xstate_path,
                state,
                pricer,
                strikes,
                calls,
                calls_tp1,
                deltas,
                next_pricer,
                next_state,
                block,
                importance_sample,
            )
            gc.collect()

    if universe is None:
        raise RuntimeError("terminal path generator did not produce a universe")
    S = experiment.num_samples
    assert universe["dC"].shape == (S, trading_times, T, N)
    assert universe["C"].shape == (S, trading_times, T, N)
    assert universe["dS"].shape == (S, trading_times)
    option_dS_name = (
        "expiry_dS"
        if experiment.return_horizon == "expiry"
        else "option_dS"
    )
    assert universe[option_dS_name].shape == (S, trading_times, T)
    if experiment.return_horizon == "incremental":
        assert universe["option_settlement_dS"].shape == (S, trading_times, T)
    assert universe["xstate"].shape == (
        S,
        trading_times,
        sampler.num_xstate,
    )
    importance_diagnostics = None
    if importance_config is not None:
        assert importance_accumulator is not None
        assert importance_component_counts is not None
        importance_diagnostics = _importance_control_diagnostics(
            importance_accumulator,
            samples=S,
            config=importance_config,
            target_steps=importance_target_steps,
            component_counts=importance_component_counts,
            minimum_log_weight=minimum_log_weight,
            maximum_log_weight=maximum_log_weight,
        )
        assert universe["log_path_weight"].shape == (S,)
    diagnostics = TerminalPathDiagnostics(
        mode=experiment.mode,
        option_return_horizon=experiment.return_horizon,
        trading_times=trading_times,
        trading_step_stride=trading_step_stride,
        simulated_state_steps=simulated_state_steps,
        expiry_step_offsets=tuple(int(value) for value in offsets),
        dt=experiment.dt,
        maximum_expiry_time_error=maximum_expiry_time_error,
        minimum_log_spot=minimum_log_spot,
        maximum_log_spot=maximum_log_spot,
        importance_sampling=importance_diagnostics,
    )
    return universe, diagnostics


def prepare_one_time_terminal_experiment(
    experiment: GenDysanosExperiment,
    *,
    verbose: Context = Context.all,
) -> tuple[dict[str, np.ndarray], TerminalPathDiagnostics]:
    """Compatibility entry point for coherent one-step path generation."""

    verify_inp(
        experiment.num_trading_times == 1
        and experiment.trading_frequency == "daily"
        and experiment.return_horizon == "expiry",
        lambda: (
            "prepare_one_time_terminal_experiment requires one daily expiry "
            "trading time; got "
            f"{experiment.trading_frequency}/{experiment.return_horizon}/"
            f"{experiment.num_trading_times}"
        ),
    )
    return prepare_terminal_path_experiment(experiment, verbose=verbose)


__all__ = [
    "DysanosMatchedUniverseBlock",
    "DysanosUniverseStreamDiagnostics",
    "GenDysanosExperiment",
    "finalize_dysanos_stream_diagnostics",
    "iter_dysanos_matched_universe_blocks",
    "prepare_one_time_terminal_experiment",
    "prepare_terminal_path_experiment",
]
