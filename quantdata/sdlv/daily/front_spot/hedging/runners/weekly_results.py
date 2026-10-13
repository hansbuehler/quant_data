"""Auditable statistics for completed overlapping weekly benchmark runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedging.inference import (
    EpisodeInferenceConfig,
    EpisodeLossSummary,
    summarize_episode_losses,
)
from quantdata.hedging.metrics import (
    entropy_risk,
    mean_variance_risk,
    mean_volatility_risk,
    power_exponential_risk,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyEpisodePath,
    build_weekly_episode_path,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyModelRun,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
    WeeklyEvaluationSummary,
    eligible_weekly_origins,
    read_weekly_record,
)

WEEKLY_RESULTS_SCHEMA = "hedging-weekly-results-v1"


@version("0.1", dependencies=[EpisodeInferenceConfig, WeeklyEvaluationSpec])
@dataclass(frozen=True, kw_only=True)
class WeeklyResultsConfig:
    """Precommitted weekly aggregation, inference, and risk controls."""

    volatility_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    inference: EpisodeInferenceConfig = EpisodeInferenceConfig()
    mean_variance_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    mean_volatility_parameters: tuple[float, ...] = (0.1, 1.0, 2.0, 3.0)
    entropy_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    power_exponential_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)

    def __post_init__(self) -> None:
        lower, upper = self.volatility_quantiles
        verify_inp(
            0.0 < lower < upper < 1.0,
            lambda: "weekly volatility quantiles must increase inside (0,1)",
        )
        for values in (
            self.mean_variance_parameters,
            self.mean_volatility_parameters,
            self.entropy_parameters,
            self.power_exponential_parameters,
        ):
            verify_inp(
                bool(values)
                and len(set(values)) == len(values)
                and all(math.isfinite(value) and value > 0.0 for value in values),
                lambda: "weekly risk parameters must be unique finite-positive values",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete aggregation and inference configuration."""
        return {
            "code_version": str(type(self).version),
            "volatility_quantiles": list(self.volatility_quantiles),
            "inference": {
                **asdict(self.inference),
                "hac_kernel": self.inference.hac_kernel.value,
            },
            "mean_variance_parameters": list(self.mean_variance_parameters),
            "mean_volatility_parameters": list(self.mean_volatility_parameters),
            "entropy_parameters": list(self.entropy_parameters),
            "power_exponential_parameters": list(
                self.power_exponential_parameters
            ),
        }


@dataclass(frozen=True, kw_only=True)
class WeeklyLossStatistics:
    """Loss distribution, overlap-aware inference, and risk scores."""

    observations: int
    mean: float
    median: float
    sample_standard_deviation: float
    root_mean_square: float
    mean_absolute_loss: float
    quantile_95: float
    quantile_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    hac_standard_error: float
    raw_hac_long_run_variance: float
    hac_long_run_variance: float
    hac_variance_was_floored: bool
    hac_kernel: str
    hac_lags: int
    hac_confidence_interval_95: tuple[float, float]
    hac_confidence_interval_99: tuple[float, float]
    bootstrap_confidence_interval_95: tuple[float, float]
    bootstrap_confidence_interval_99: tuple[float, float]
    mean_variance: dict[str, float]
    mean_volatility: dict[str, float]
    entropy: dict[str, float]
    power_exponential: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class WeeklyResultSlice:
    """One weekly target's statistics for one dated volatility stratum."""

    claim: str
    slice_label: str
    year: int | None
    volatility_regime: str | None
    unhedged_loss: WeeklyLossStatistics
    economic_loss: WeeklyLossStatistics
    pure_hedge_loss: WeeklyLossStatistics
    pricing_loss: WeeklyLossStatistics
    mean_transaction_cost: float
    mean_spot_transaction_cost: float
    mean_option_transaction_cost: float
    mean_total_turnover_notional: float
    mean_entry_turnover_notional: float
    mean_gross_exposure: float
    mean_net_exposure: float
    mean_absolute_net_exposure: float
    mean_nonzero_actions_per_decision: float
    cap_hits: int
    numerical_cap_hit_result_rate: float


@dataclass(frozen=True, kw_only=True)
class WeeklyRunResults:
    """Complete model-neutral report for one immutable weekly run."""

    run_identity: str
    model: str
    measure: str
    schedule: str
    dataset_manifest_sha256: str
    result_count: int
    aggregation_config: dict[str, object]
    volatility_thresholds: tuple[float, float]
    exposure_convention: str
    source_run_sha256: str
    source_summary_sha256: str
    record_sha256_by_origin: dict[str, str]
    slices: tuple[WeeklyResultSlice, ...]

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.run_identity,
                self.model,
                self.measure,
                self.schedule,
                self.dataset_manifest_sha256,
                self.exposure_convention,
            )
        ):
            raise ValueError("weekly result identities and labels must be non-empty")
        for name, value in (
            ("run_identity", self.run_identity),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("source_run_sha256", self.source_run_sha256),
            ("source_summary_sha256", self.source_summary_sha256),
        ):
            _validate_sha256(value, name=name)
        if self.result_count <= 0:
            raise ValueError("weekly result count must be positive")
        lower, upper = self.volatility_thresholds
        if not 0.0 < lower <= upper:
            raise ValueError("weekly result volatility thresholds are invalid")
        origins = tuple(self.record_sha256_by_origin)
        if origins != tuple(sorted(origins)) or not origins:
            raise ValueError("weekly source records must be non-empty and chronological")
        for origin, digest in self.record_sha256_by_origin.items():
            date.fromisoformat(origin)
            _validate_sha256(digest, name=f"record hash for {origin}")
        if not self.slices:
            raise ValueError("weekly result slices must be non-empty")
        slice_ids = tuple(
            (value.claim, value.slice_label) for value in self.slices
        )
        if len(set(slice_ids)) != len(slice_ids):
            raise ValueError("weekly result slices must have unique claim/label pairs")
        expected_claims = {target.label for target in weekly_targets()}
        overall = tuple(
            value
            for value in self.slices
            if value.slice_label == "overall"
            and value.year is None
            and value.volatility_regime is None
        )
        if {value.claim for value in overall} != expected_claims:
            raise ValueError("weekly results require one overall slice per target")
        observations = {value.pure_hedge_loss.observations for value in overall}
        if observations != {len(self.record_sha256_by_origin)}:
            raise ValueError("weekly overall slices do not match source origins")
        observation_count = next(iter(observations))
        if observation_count * len(overall) != self.result_count:
            raise ValueError("weekly overall slices do not match the result count")

    @property
    def slices_sha256(self) -> str:
        """Return the deterministic hash of every reported slice."""
        return _json_sha256([asdict(value) for value in self.slices])

    @property
    def aggregation_identity(self) -> str:
        """Bind source bytes, aggregation controls, and reported slices."""
        return _json_sha256(
            {
                "run_identity": self.run_identity,
                "model": self.model,
                "measure": self.measure,
                "schedule": self.schedule,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "result_count": self.result_count,
                "volatility_thresholds": self.volatility_thresholds,
                "exposure_convention": self.exposure_convention,
                "source_run_sha256": self.source_run_sha256,
                "source_summary_sha256": self.source_summary_sha256,
                "record_sha256_by_origin": self.record_sha256_by_origin,
                "aggregation_config": self.aggregation_config,
                "slices_sha256": self.slices_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report payload."""
        return {
            "schema": WEEKLY_RESULTS_SCHEMA,
            "run_identity": self.run_identity,
            "model": self.model,
            "measure": self.measure,
            "schedule": self.schedule,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "result_count": self.result_count,
            "aggregation_config": self.aggregation_config,
            "volatility_thresholds": list(self.volatility_thresholds),
            "exposure_convention": self.exposure_convention,
            "source_run_sha256": self.source_run_sha256,
            "source_summary_sha256": self.source_summary_sha256,
            "record_sha256_by_origin": self.record_sha256_by_origin,
            "slices_sha256": self.slices_sha256,
            "aggregation_identity": self.aggregation_identity,
            "slices": [asdict(value) for value in self.slices],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        *,
        expected_config: WeeklyResultsConfig = WeeklyResultsConfig(),
    ) -> WeeklyRunResults:
        """Reconstruct and authenticate a completed weekly aggregate."""
        if payload.get("schema") != WEEKLY_RESULTS_SCHEMA:
            raise ValueError("weekly aggregate has an incompatible schema")
        aggregation_config = _required_object(payload, "aggregation_config")
        if aggregation_config != expected_config.to_dict():
            raise ValueError("weekly aggregate configuration is incompatible")
        slices_payload = payload.get("slices")
        if not isinstance(slices_payload, list):
            raise ValueError("weekly aggregate slices must be a list")
        records_payload = _required_object(payload, "record_sha256_by_origin")
        if not all(
            isinstance(origin, str) and isinstance(digest, str)
            for origin, digest in records_payload.items()
        ):
            raise ValueError("weekly aggregate record hashes are invalid")
        result = cls(
            run_identity=_required_string(payload, "run_identity"),
            model=_required_string(payload, "model"),
            measure=_required_string(payload, "measure"),
            schedule=_required_string(payload, "schedule"),
            dataset_manifest_sha256=_required_string(
                payload,
                "dataset_manifest_sha256",
            ),
            result_count=_required_integer(payload, "result_count"),
            aggregation_config=aggregation_config,
            volatility_thresholds=_required_float_pair(
                payload,
                "volatility_thresholds",
            ),
            exposure_convention=_required_string(payload, "exposure_convention"),
            source_run_sha256=_required_string(payload, "source_run_sha256"),
            source_summary_sha256=_required_string(
                payload,
                "source_summary_sha256",
            ),
            record_sha256_by_origin={
                str(origin): str(digest)
                for origin, digest in records_payload.items()
            },
            slices=tuple(
                _weekly_result_slice_from_dict(value, config=expected_config)
                for value in slices_payload
            ),
        )
        if payload.get("slices_sha256") != result.slices_sha256:
            raise ValueError("weekly aggregate slice hash is not authentic")
        if payload.get("aggregation_identity") != result.aggregation_identity:
            raise ValueError("weekly aggregate identity is not authentic")
        return result

    def write_json(self, path: Path) -> None:
        """Write the report atomically with deterministic formatting."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


@dataclass(frozen=True, kw_only=True)
class WeeklyExposure:
    """Normalized executable turnover and exposure for one weekly target."""

    total_turnover_notional: float
    entry_turnover_notional: float
    mean_gross_exposure: float
    mean_net_exposure: float
    mean_absolute_net_exposure: float
    mean_nonzero_actions_per_decision: float


def summarize_completed_weekly_run(
    panel: PublicHedgingPanel,
    *,
    run_directory: Path,
    config: WeeklyResultsConfig = WeeklyResultsConfig(),
    progress: Callable[[int, int, date], None] | None = None,
) -> WeeklyRunResults:
    """Load, revalidate, and summarize one complete weekly evaluation."""
    require_public_hedging_panel(panel)
    directory = Path(run_directory)
    run_path = directory / "run.json"
    summary_path = directory / "summary.json"
    run_payload = _read_json_object(run_path)
    summary_payload = _read_json_object(summary_path)
    spec = WeeklyEvaluationSpec.from_dict(run_payload)
    summary = WeeklyEvaluationSummary.from_dict(summary_payload)
    if summary.run_identity != spec.run_identity:
        raise ValueError("weekly run and summary identities do not match")
    if spec.dataset_manifest_sha256 != dataset_manifest_sha256(panel):
        raise ValueError("weekly run belongs to another public dataset manifest")
    eligible = tuple(
        value.isoformat() for value in eligible_weekly_origins(panel, spec=spec)
    )
    if summary.eligible_origins != eligible:
        raise ValueError("weekly summary origins do not match the run specification")
    if not eligible or summary.completed_origins != eligible:
        raise ValueError("weekly aggregation requires a complete evaluation run")

    runs: list[WeeklyModelRun] = []
    exposures: dict[tuple[str, str], WeeklyExposure] = {}
    record_sha256_by_origin: dict[str, str] = {}
    cap_hits = 0
    for index, origin in enumerate(summary.completed_origins, start=1):
        origin_date = date.fromisoformat(origin)
        record_path = directory / "records" / f"{origin}.json"
        origin_runs = read_weekly_record(
            record_path,
            panel=panel,
            spec=spec,
            origin_date=origin_date,
        )
        record_sha256_by_origin[origin] = _file_sha256(record_path)
        episode = build_weekly_episode_path(
            panel,
            origin_date=origin_date,
            schedule=spec.schedule,
            rolling_window_business_days=spec.rolling_observations,
        )
        for run in origin_runs:
            key = (origin, run.result.target_label)
            if key in exposures:
                raise ValueError("weekly run contains duplicate origin/claim results")
            exposures[key] = weekly_exposure(run, episode=episode)
            cap_hits += _run_cap_hits(run, position_cap=spec.position_cap)
        runs.extend(origin_runs)
        if progress is not None:
            progress(index, len(summary.completed_origins), origin_date)
    if len(runs) != summary.result_count:
        raise ValueError("weekly record count does not match the completed summary")
    if summary.result_count != len(eligible) * len(spec.target_labels):
        raise ValueError("weekly result count does not match the immutable target book")
    if cap_hits != summary.cap_hits:
        raise ValueError("weekly cap-hit count does not match its records")

    thresholds = initialization_weekly_volatility_thresholds(panel, config=config)
    volatility_by_origin = {
        origin: panel.market_day(date.fromisoformat(origin)).atm_vol_at_business_dte(
            WEEKLY_HORIZON
        )
        for origin in summary.completed_origins
    }
    slices = summarize_weekly_results(
        tuple(runs),
        volatility_by_origin=volatility_by_origin,
        volatility_thresholds=thresholds,
        exposure_by_result=exposures,
        position_cap=spec.position_cap,
        config=config,
    )
    return WeeklyRunResults(
        run_identity=spec.run_identity,
        model=spec.model,
        measure=spec.measure,
        schedule=spec.schedule.value,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        result_count=len(runs),
        aggregation_config=config.to_dict(),
        volatility_thresholds=thresholds,
        exposure_convention=(
            "discounted episode-origin PV notional; turnover includes all five "
            "rebalances and terminal liquidation, while exposures are averaged "
            "over the five predictable decision books"
        ),
        source_run_sha256=_file_sha256(run_path),
        source_summary_sha256=_file_sha256(summary_path),
        record_sha256_by_origin=record_sha256_by_origin,
        slices=slices,
    )


def initialization_weekly_volatility_thresholds(
    panel: PublicHedgingPanel,
    *,
    config: WeeklyResultsConfig = WeeklyResultsConfig(),
) -> tuple[float, float]:
    """Fix five-session ATM-volatility regimes from the 2022 initialization set."""
    require_public_hedging_panel(panel)
    initialization_dates = tuple(value for value in panel.dates if value.year == 2022)
    if not initialization_dates:
        raise ValueError("public panel has no 2022 initialization dates")
    values = np.asarray(
        [
            panel.market_day(trading_date).atm_vol_at_business_dte(WEEKLY_HORIZON)
            for trading_date in initialization_dates
        ],
        dtype=np.float64,
    )
    return tuple(
        float(value)
        for value in np.quantile(
            values,
            config.volatility_quantiles,
            method="linear",
        )
    )


def summarize_weekly_results(
    runs: tuple[WeeklyModelRun, ...],
    *,
    volatility_by_origin: dict[str, float],
    volatility_thresholds: tuple[float, float],
    exposure_by_result: dict[tuple[str, str], WeeklyExposure],
    position_cap: float,
    config: WeeklyResultsConfig = WeeklyResultsConfig(),
) -> tuple[WeeklyResultSlice, ...]:
    """Return overall, annual, regime, and annual-by-regime weekly slices."""
    if not runs:
        raise ValueError("weekly results must be non-empty")
    known_claims = {target.label for target in weekly_targets()}
    unknown = {run.result.target_label for run in runs} - known_claims
    if unknown:
        raise ValueError(f"weekly results contain unknown claims {sorted(unknown)}")
    if not math.isfinite(position_cap) or position_cap <= 0.0:
        raise ValueError("weekly reporting position cap must be finite-positive")

    grouped: dict[str, list[tuple[WeeklyModelRun, int, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for run in runs:
        origin = run.result.origin_date.isoformat()
        key = (origin, run.result.target_label)
        if key in seen:
            raise ValueError("weekly results contain duplicate origin/claim rows")
        seen.add(key)
        try:
            volatility = volatility_by_origin[origin]
            exposure_by_result[key]
        except KeyError as error:
            raise ValueError("weekly volatility or exposure metadata is incomplete") from error
        regime = _volatility_regime(volatility, volatility_thresholds)
        grouped.setdefault(run.result.target_label, []).append(
            (run, run.result.origin_date.year, regime)
        )

    slices: list[WeeklyResultSlice] = []
    for claim in sorted(grouped):
        rows = sorted(grouped[claim], key=lambda value: value[0].result.origin_date)
        years = sorted({year for _, year, _ in rows})
        regimes = ("low", "medium", "high")
        specifications: list[tuple[str, int | None, str | None]] = [
            ("overall", None, None)
        ]
        specifications.extend((f"year:{year}", year, None) for year in years)
        specifications.extend((f"regime:{regime}", None, regime) for regime in regimes)
        specifications.extend(
            (f"year:{year}/regime:{regime}", year, regime)
            for year in years
            for regime in regimes
        )
        for label, selected_year, selected_regime in specifications:
            domain = tuple(
                run
                for run, year, _ in rows
                if selected_year is None or year == selected_year
            )
            domain_index = {
                run.result.origin_date: index for index, run in enumerate(domain)
            }
            selected = tuple(
                run
                for run, year, regime in rows
                if (selected_year is None or year == selected_year)
                and (selected_regime is None or regime == selected_regime)
            )
            if len(selected) < 2:
                continue
            slices.append(
                _summarize_slice(
                    selected,
                    claim=claim,
                    slice_label=label,
                    year=selected_year,
                    volatility_regime=selected_regime,
                    exposure_by_result=exposure_by_result,
                    position_cap=position_cap,
                    episode_indices=tuple(
                        domain_index[run.result.origin_date] for run in selected
                    ),
                    full_episode_count=len(domain),
                    config=config,
                )
            )
    return tuple(slices)


def weekly_exposure(
    run: WeeklyModelRun,
    *,
    episode: WeeklyEpisodePath,
) -> WeeklyExposure:
    """Reconstruct executable PV turnover and decision-book exposure."""
    if run.result.origin_date != episode.spec.origin_date:
        raise ValueError("weekly exposure episode does not match the model run")
    spot = np.asarray(run.spot_positions, dtype=np.float64)
    options = np.asarray(run.option_positions, dtype=np.float64)
    previous_spot = 0.0
    previous_options = np.zeros(options.shape[1], dtype=np.float64)
    total_turnover = 0.0
    entry_turnover = 0.0
    gross: list[float] = []
    net: list[float] = []
    nonzero: list[int] = []
    for step in range(WEEKLY_HORIZON):
        previous_options = np.where(
            episode.option_alive[step], previous_options, 0.0
        )
        spot_mark = (
            episode.discount_to_origin[step]
            * episode.spot_over_episode_origin[step]
        )
        option_marks = episode.option_marks_pv[step]
        turnover = abs(float(spot[step] - previous_spot)) * spot_mark + float(
            np.sum(np.abs(options[step] - previous_options) * option_marks)
        )
        if step == 0:
            entry_turnover = turnover
        total_turnover += turnover
        gross.append(
            abs(float(spot[step])) * spot_mark
            + float(np.sum(np.abs(options[step]) * option_marks))
        )
        net.append(
            float(spot[step]) * spot_mark
            + float(options[step] @ option_marks)
        )
        nonzero.append(
            int(abs(float(spot[step])) > 1.0e-14)
            + int(np.count_nonzero(np.abs(options[step]) > 1.0e-14))
        )
        previous_spot = float(spot[step])
        previous_options = np.where(
            episode.option_settles_next[step], 0.0, options[step]
        )
    terminal_spot_mark = (
        episode.discount_to_origin[-1] * episode.spot_over_episode_origin[-1]
    )
    total_turnover += abs(previous_spot) * terminal_spot_mark + float(
        np.sum(
            np.abs(previous_options)
            * episode.option_marks_pv[-1]
            * episode.option_alive[-1]
        )
    )
    return WeeklyExposure(
        total_turnover_notional=total_turnover,
        entry_turnover_notional=entry_turnover,
        mean_gross_exposure=float(np.mean(gross)),
        mean_net_exposure=float(np.mean(net)),
        mean_absolute_net_exposure=float(np.mean(np.abs(net))),
        mean_nonzero_actions_per_decision=float(np.mean(nonzero)),
    )


def _summarize_slice(
    runs: tuple[WeeklyModelRun, ...],
    *,
    claim: str,
    slice_label: str,
    year: int | None,
    volatility_regime: str | None,
    exposure_by_result: dict[tuple[str, str], WeeklyExposure],
    position_cap: float,
    episode_indices: tuple[int, ...],
    full_episode_count: int,
    config: WeeklyResultsConfig,
) -> WeeklyResultSlice:
    dates = tuple(run.result.origin_date for run in runs)
    breakdowns = tuple(run.result.breakdown for run in runs)
    exposures = tuple(
        exposure_by_result[(run.result.origin_date.isoformat(), claim)] for run in runs
    )
    cap_hits = sum(_run_cap_hits(run, position_cap=position_cap) for run in runs)
    cap_hit_results = sum(
        _run_cap_hits(run, position_cap=position_cap) > 0 for run in runs
    )
    unhedged_losses = np.asarray(
        [
            value.claim_quantity * (value.market_price - value.payoff)
            for value in breakdowns
        ],
        dtype=np.float64,
    )
    return WeeklyResultSlice(
        claim=claim,
        slice_label=slice_label,
        year=year,
        volatility_regime=volatility_regime,
        unhedged_loss=_loss_statistics(
            unhedged_losses,
            dates=dates,
            episode_indices=episode_indices,
            full_episode_count=full_episode_count,
            config=config,
        ),
        economic_loss=_loss_statistics(
            -np.asarray([value.economic_pnl for value in breakdowns]),
            dates=dates,
            episode_indices=episode_indices,
            full_episode_count=full_episode_count,
            config=config,
        ),
        pure_hedge_loss=_loss_statistics(
            -np.asarray([value.pure_hedge_error for value in breakdowns]),
            dates=dates,
            episode_indices=episode_indices,
            full_episode_count=full_episode_count,
            config=config,
        ),
        pricing_loss=_loss_statistics(
            -np.asarray([value.pricing_contribution for value in breakdowns]),
            dates=dates,
            episode_indices=episode_indices,
            full_episode_count=full_episode_count,
            config=config,
        ),
        mean_transaction_cost=float(
            np.mean([run.result.transaction_cost for run in runs])
        ),
        mean_spot_transaction_cost=float(
            np.mean([run.result.spot_transaction_cost for run in runs])
        ),
        mean_option_transaction_cost=float(
            np.mean([run.result.option_transaction_cost for run in runs])
        ),
        mean_total_turnover_notional=float(
            np.mean([value.total_turnover_notional for value in exposures])
        ),
        mean_entry_turnover_notional=float(
            np.mean([value.entry_turnover_notional for value in exposures])
        ),
        mean_gross_exposure=float(
            np.mean([value.mean_gross_exposure for value in exposures])
        ),
        mean_net_exposure=float(
            np.mean([value.mean_net_exposure for value in exposures])
        ),
        mean_absolute_net_exposure=float(
            np.mean([value.mean_absolute_net_exposure for value in exposures])
        ),
        mean_nonzero_actions_per_decision=float(
            np.mean([value.mean_nonzero_actions_per_decision for value in exposures])
        ),
        cap_hits=cap_hits,
        numerical_cap_hit_result_rate=cap_hit_results / len(runs),
    )


def _loss_statistics(
    losses: np.ndarray,
    *,
    dates: tuple[date, ...],
    episode_indices: tuple[int, ...],
    full_episode_count: int,
    config: WeeklyResultsConfig,
) -> WeeklyLossStatistics:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("weekly losses must contain at least two finite observations")
    inference = EpisodeInferenceConfig(
        hac_lags=min(config.inference.hac_lags, full_episode_count - 1),
        hac_kernel=config.inference.hac_kernel,
        bootstrap_block_length=min(
            config.inference.bootstrap_block_length,
            full_episode_count,
        ),
        bootstrap_samples=config.inference.bootstrap_samples,
        bootstrap_seed=config.inference.bootstrap_seed,
    )
    summary = summarize_episode_losses(
        values,
        episode_dates=dates,
        episode_indices=episode_indices,
        full_episode_count=full_episode_count,
        config=inference,
    )
    return _with_risk_scores(summary, values=values, config=config)


def _with_risk_scores(
    summary: EpisodeLossSummary,
    *,
    values: np.ndarray,
    config: WeeklyResultsConfig,
) -> WeeklyLossStatistics:
    inference = summary.inference
    return WeeklyLossStatistics(
        observations=summary.observations,
        mean=summary.mean,
        median=summary.median,
        sample_standard_deviation=summary.sample_standard_deviation,
        root_mean_square=summary.root_mean_square,
        mean_absolute_loss=summary.mean_absolute_loss,
        quantile_95=summary.quantile_95,
        quantile_99=summary.quantile_99,
        expected_shortfall_95=summary.expected_shortfall_95,
        expected_shortfall_99=summary.expected_shortfall_99,
        hac_standard_error=inference.hac_standard_error,
        raw_hac_long_run_variance=inference.raw_hac_long_run_variance,
        hac_long_run_variance=inference.hac_long_run_variance,
        hac_variance_was_floored=inference.hac_variance_was_floored,
        hac_kernel=inference.hac_kernel.value,
        hac_lags=inference.hac_lags,
        hac_confidence_interval_95=inference.hac_95,
        hac_confidence_interval_99=inference.hac_99,
        bootstrap_confidence_interval_95=inference.bootstrap_95,
        bootstrap_confidence_interval_99=inference.bootstrap_99,
        mean_variance={
            f"{parameter:g}": mean_variance_risk(values, risk_parameter=parameter)
            for parameter in config.mean_variance_parameters
        },
        mean_volatility={
            f"{parameter:g}": mean_volatility_risk(values, risk_parameter=parameter)
            for parameter in config.mean_volatility_parameters
        },
        entropy={
            f"{parameter:g}": entropy_risk(values, risk_parameter=parameter)
            for parameter in config.entropy_parameters
        },
        power_exponential={
            f"{parameter:g}": power_exponential_risk(
                values,
                risk_parameter=parameter,
            )
            for parameter in config.power_exponential_parameters
        },
    )


def _run_cap_hits(run: WeeklyModelRun, *, position_cap: float) -> int:
    threshold = position_cap * (1.0 - 1.0e-10)
    return int(np.count_nonzero(np.abs(run.spot_positions) >= threshold)) + int(
        np.count_nonzero(np.abs(run.option_positions) >= threshold)
    )


def _volatility_regime(
    volatility: float,
    thresholds: tuple[float, float],
) -> str:
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("weekly ATM volatility must be finite-positive")
    lower, upper = thresholds
    if not 0.0 < lower <= upper:
        raise ValueError("weekly volatility thresholds are invalid")
    if volatility <= lower:
        return "low"
    if volatility <= upper:
        return "medium"
    return "high"


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"weekly JSON artifact must be an object: {path}")
    return payload


def _weekly_result_slice_from_dict(
    value: object,
    *,
    config: WeeklyResultsConfig,
) -> WeeklyResultSlice:
    if not isinstance(value, dict):
        raise ValueError("weekly aggregate slice must be a JSON object")
    claim = _required_string(value, "claim")
    if claim not in {target.label for target in weekly_targets()}:
        raise ValueError(f"weekly aggregate has unknown claim {claim}")
    slice_label = _required_string(value, "slice_label")
    year_value = value.get("year")
    if year_value is not None and (
        isinstance(year_value, bool) or not isinstance(year_value, int)
    ):
        raise ValueError("weekly aggregate slice year must be an integer or null")
    regime_value = value.get("volatility_regime")
    if regime_value not in {None, "low", "medium", "high"}:
        raise ValueError("weekly aggregate volatility regime is invalid")
    expected_label = (
        "overall"
        if year_value is None and regime_value is None
        else (
            f"year:{year_value}"
            if regime_value is None
            else (
                f"regime:{regime_value}"
                if year_value is None
                else f"year:{year_value}/regime:{regime_value}"
            )
        )
    )
    if slice_label != expected_label:
        raise ValueError("weekly aggregate slice label does not match its selectors")
    losses = tuple(
        _weekly_loss_statistics_from_dict(
            _required_object(value, name),
            config=config,
        )
        for name in (
            "unhedged_loss",
            "economic_loss",
            "pure_hedge_loss",
            "pricing_loss",
        )
    )
    if len({loss.observations for loss in losses}) != 1:
        raise ValueError("weekly aggregate slice loss counts do not align")
    nonnegative_names = (
        "mean_transaction_cost",
        "mean_spot_transaction_cost",
        "mean_option_transaction_cost",
        "mean_total_turnover_notional",
        "mean_entry_turnover_notional",
        "mean_gross_exposure",
        "mean_absolute_net_exposure",
        "mean_nonzero_actions_per_decision",
    )
    nonnegative = {
        name: _required_float(value, name) for name in nonnegative_names
    }
    if any(number < 0.0 for number in nonnegative.values()):
        raise ValueError("weekly aggregate costs and absolute exposures must be non-negative")
    cap_hits = _required_integer(value, "cap_hits")
    cap_rate = _required_float(value, "numerical_cap_hit_result_rate")
    if cap_hits < 0 or not 0.0 <= cap_rate <= 1.0:
        raise ValueError("weekly aggregate cap statistics are invalid")
    return WeeklyResultSlice(
        claim=claim,
        slice_label=slice_label,
        year=year_value,
        volatility_regime=regime_value,
        unhedged_loss=losses[0],
        economic_loss=losses[1],
        pure_hedge_loss=losses[2],
        pricing_loss=losses[3],
        mean_transaction_cost=nonnegative["mean_transaction_cost"],
        mean_spot_transaction_cost=nonnegative["mean_spot_transaction_cost"],
        mean_option_transaction_cost=nonnegative["mean_option_transaction_cost"],
        mean_total_turnover_notional=nonnegative["mean_total_turnover_notional"],
        mean_entry_turnover_notional=nonnegative["mean_entry_turnover_notional"],
        mean_gross_exposure=nonnegative["mean_gross_exposure"],
        mean_net_exposure=_required_float(value, "mean_net_exposure"),
        mean_absolute_net_exposure=nonnegative["mean_absolute_net_exposure"],
        mean_nonzero_actions_per_decision=nonnegative[
            "mean_nonzero_actions_per_decision"
        ],
        cap_hits=cap_hits,
        numerical_cap_hit_result_rate=cap_rate,
    )


def _weekly_loss_statistics_from_dict(
    payload: dict[str, object],
    *,
    config: WeeklyResultsConfig,
) -> WeeklyLossStatistics:
    observations = _required_integer(payload, "observations")
    if observations < 2:
        raise ValueError("weekly aggregate losses require at least two observations")
    scalar_names = (
        "mean",
        "median",
        "sample_standard_deviation",
        "root_mean_square",
        "mean_absolute_loss",
        "quantile_95",
        "quantile_99",
        "expected_shortfall_95",
        "expected_shortfall_99",
        "hac_standard_error",
        "raw_hac_long_run_variance",
        "hac_long_run_variance",
    )
    scalars = {name: _required_float(payload, name) for name in scalar_names}
    if any(
        scalars[name] < 0.0
        for name in (
            "sample_standard_deviation",
            "root_mean_square",
            "mean_absolute_loss",
            "hac_standard_error",
            "hac_long_run_variance",
        )
    ):
        raise ValueError("weekly aggregate dispersion statistics are invalid")
    floored = payload.get("hac_variance_was_floored")
    if not isinstance(floored, bool):
        raise ValueError("weekly aggregate HAC floor flag must be boolean")
    hac_kernel = _required_string(payload, "hac_kernel")
    if hac_kernel != config.inference.hac_kernel.value:
        raise ValueError("weekly aggregate HAC kernel is incompatible")
    hac_lags = _required_integer(payload, "hac_lags")
    if hac_lags < 0:
        raise ValueError("weekly aggregate HAC lags must be non-negative")
    risk_values = {
        "mean_variance": _required_risk_scores(
            payload,
            "mean_variance",
            config.mean_variance_parameters,
        ),
        "mean_volatility": _required_risk_scores(
            payload,
            "mean_volatility",
            config.mean_volatility_parameters,
        ),
        "entropy": _required_risk_scores(
            payload,
            "entropy",
            config.entropy_parameters,
        ),
        "power_exponential": _required_risk_scores(
            payload,
            "power_exponential",
            config.power_exponential_parameters,
        ),
    }
    return WeeklyLossStatistics(
        observations=observations,
        **scalars,
        hac_variance_was_floored=floored,
        hac_kernel=hac_kernel,
        hac_lags=hac_lags,
        hac_confidence_interval_95=_required_float_pair(
            payload,
            "hac_confidence_interval_95",
        ),
        hac_confidence_interval_99=_required_float_pair(
            payload,
            "hac_confidence_interval_99",
        ),
        bootstrap_confidence_interval_95=_required_float_pair(
            payload,
            "bootstrap_confidence_interval_95",
        ),
        bootstrap_confidence_interval_99=_required_float_pair(
            payload,
            "bootstrap_confidence_interval_99",
        ),
        **risk_values,
    )


def _required_risk_scores(
    payload: dict[str, object],
    name: str,
    parameters: tuple[float, ...],
) -> dict[str, float]:
    scores = _required_object(payload, name)
    expected = {f"{parameter:g}" for parameter in parameters}
    if set(scores) != expected:
        raise ValueError(f"weekly aggregate {name} parameters are incompatible")
    return {label: _finite_float(value, name=f"{name}[{label}]") for label, value in scores.items()}


def _required_object(
    payload: dict[str, object],
    name: str,
) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_float(payload: dict[str, object], name: str) -> float:
    return _finite_float(payload.get(name), name=name)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_float_pair(
    payload: dict[str, object],
    name: str,
) -> tuple[float, float]:
    value = payload.get(name)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    return tuple(_finite_float(item, name=name) for item in value)


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
