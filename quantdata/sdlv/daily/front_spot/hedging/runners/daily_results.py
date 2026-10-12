"""Auditable aggregate statistics for completed daily benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.hedging.inference import HacKernel, hac_standard_error_of_mean
from quantdata.hedging.metrics import (
    HedgeSmokeResult,
    entropy_risk,
    mean_variance_risk,
    mean_volatility_risk,
    power_exponential_risk,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyEvaluationSpec,
    DailyEvaluationSummary,
    dataset_manifest_sha256,
    eligible_daily_origins,
    read_daily_record,
)

FloatArray = NDArray[np.float64]

DAILY_RESULTS_SCHEMA = "hedging-daily-results-v2"
_NORMAL_95 = 1.959963984540054
_NORMAL_99 = 2.5758293035489004


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class DailyResultsConfig:
    """Precommitted daily aggregation and risk-score controls."""

    volatility_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    hac_lags: int = 5
    hac_kernel: HacKernel = HacKernel.BARTLETT
    mean_variance_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    mean_volatility_parameters: tuple[float, ...] = (0.1, 1.0, 2.0, 3.0)
    entropy_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    power_exponential_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)

    def __post_init__(self) -> None:
        lower, upper = self.volatility_quantiles
        verify_inp(
            0.0 < lower < upper < 1.0,
            lambda: "volatility quantiles must be increasing inside (0,1)",
        )
        verify_inp(self.hac_lags >= 0, lambda: "hac_lags must be non-negative")
        verify_inp(
            isinstance(self.hac_kernel, HacKernel),
            lambda: "hac_kernel must be a HacKernel",
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
                lambda: "risk parameters must be unique finite-positive values",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete aggregation and inference configuration."""
        return {
            "code_version": str(type(self).version),
            "volatility_quantiles": list(self.volatility_quantiles),
            "hac_lags": self.hac_lags,
            "hac_kernel": self.hac_kernel.value,
            "hac_lag_unit": "selected-observation order within each reported slice",
            "mean_variance_parameters": list(self.mean_variance_parameters),
            "mean_volatility_parameters": list(self.mean_volatility_parameters),
            "entropy_parameters": list(self.entropy_parameters),
            "power_exponential_parameters": list(
                self.power_exponential_parameters
            ),
        }


@dataclass(frozen=True, kw_only=True)
class DailyLossStatistics:
    """Loss-distribution, inference, and risk-functional statistics."""

    observations: int
    mean: float
    standard_error: float
    standard_error_method: str
    hac_kernel: str
    hac_lags: int
    hac_lag_unit: str
    confidence_interval_95: tuple[float, float]
    confidence_interval_99: tuple[float, float]
    median: float
    sample_standard_deviation: float
    root_mean_square: float
    mean_absolute_loss: float
    quantile_95: float
    quantile_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    mean_variance: dict[str, float]
    mean_volatility: dict[str, float]
    entropy: dict[str, float]
    power_exponential: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class DailyResultSlice:
    """One claim's statistics for one dated volatility stratum."""

    claim: str
    slice_label: str
    year: int | None
    volatility_regime: str | None
    unhedged_loss: DailyLossStatistics
    economic_loss: DailyLossStatistics
    pure_hedge_loss: DailyLossStatistics
    pricing_loss: DailyLossStatistics
    mean_transaction_cost: float
    mean_action_l1: float
    mean_entry_turnover_notional: float
    mean_gross_exposure: float
    mean_net_exposure: float
    mean_absolute_net_exposure: float
    mean_nonzero_actions: float
    nonconverged_results: int
    nonconvergence_rate: float
    cap_hits: int
    numerical_cap_hit_result_rate: float


@dataclass(frozen=True, kw_only=True)
class DailyRunResults:
    """Complete model-neutral report for one immutable daily run."""

    run_identity: str
    model: str
    measure: str
    dataset_manifest_sha256: str
    result_count: int
    aggregation_config: dict[str, object]
    exposure_convention: str
    volatility_thresholds: dict[str, tuple[float, float]]
    slices: tuple[DailyResultSlice, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report payload."""
        return {
            "schema": DAILY_RESULTS_SCHEMA,
            "run_identity": self.run_identity,
            "model": self.model,
            "measure": self.measure,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "result_count": self.result_count,
            "aggregation_config": self.aggregation_config,
            "exposure_convention": self.exposure_convention,
            "volatility_thresholds": {
                label: list(values)
                for label, values in self.volatility_thresholds.items()
            },
            "slices": [asdict(value) for value in self.slices],
        }

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


def summarize_completed_daily_run(
    panel: PublicHedgingPanel,
    *,
    run_directory: Path,
    config: DailyResultsConfig = DailyResultsConfig(),
) -> DailyRunResults:
    """Load, validate, and summarize one completed daily evaluation directory."""
    require_public_hedging_panel(panel)
    directory = Path(run_directory)
    run_payload = _read_json_object(directory / "run.json")
    summary_payload = _read_json_object(directory / "summary.json")
    spec = DailyEvaluationSpec.from_dict(run_payload)
    summary = DailyEvaluationSummary.from_dict(summary_payload)
    if summary.run_identity != spec.run_identity:
        raise ValueError("daily run and summary identities do not match")
    if spec.dataset_manifest_sha256 != dataset_manifest_sha256(panel):
        raise ValueError("daily run belongs to another public dataset manifest")
    eligible = tuple(
        value.isoformat() for value in eligible_daily_origins(panel, spec=spec)
    )
    if summary.eligible_origins != eligible:
        raise ValueError("daily summary eligible origins do not match the run specification")
    if not eligible or summary.completed_origins != eligible:
        raise ValueError("daily aggregation requires a complete evaluation run")

    results: list[HedgeSmokeResult] = []
    for origin in summary.completed_origins:
        results.extend(
            read_daily_record(
                directory / "records" / f"{origin}.json",
                spec=spec,
                origin_date=date.fromisoformat(origin),
            )
        )
    if len(results) != summary.result_count:
        raise ValueError("daily record count does not match the completed summary")
    if summary.result_count != len(eligible) * len(spec.target_labels):
        raise ValueError("daily result count does not match the immutable target book")
    if summary.nonconverged_results != sum(not result.converged for result in results):
        raise ValueError("daily nonconvergence count does not match its records")
    if summary.cap_hits != sum(result.cap_hits for result in results):
        raise ValueError("daily cap-hit count does not match its records")

    thresholds = initialization_volatility_thresholds(panel, config=config)
    volatility_by_origin = _volatility_by_origin(panel, results=tuple(results))
    exposure_by_result = _entry_exposure_by_result(panel, results=tuple(results))
    slices = summarize_daily_results(
        tuple(results),
        volatility_by_origin=volatility_by_origin,
        volatility_thresholds=thresholds,
        exposure_by_result=exposure_by_result,
        config=config,
    )
    return DailyRunResults(
        run_identity=spec.run_identity,
        model=spec.model,
        measure=spec.measure,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        result_count=len(results),
        aggregation_config=config.to_dict(),
        exposure_convention=(
            "entry positions marked at the origin normalized spot or SANOS option "
            "mark; every daily hedge starts from zero, so entry turnover equals "
            "entry gross exposure"
        ),
        volatility_thresholds={
            f"{business_dte}d": value
            for business_dte, value in thresholds.items()
        },
        slices=slices,
    )


def initialization_volatility_thresholds(
    panel: PublicHedgingPanel,
    *,
    config: DailyResultsConfig = DailyResultsConfig(),
) -> dict[int, tuple[float, float]]:
    """Fix target-maturity volatility regimes from the 2022 initialization set."""
    require_public_hedging_panel(panel)
    initialization_dates = tuple(value for value in panel.dates if value.year == 2022)
    if not initialization_dates:
        raise ValueError("public panel has no 2022 initialization dates")
    maturities = tuple(
        sorted({target.maturity_business_days for target in daily_digital_targets()})
    )
    return {
        maturity: tuple(
            float(value)
            for value in np.quantile(
                np.asarray(
                    [
                        panel.market_day(trading_date).atm_vol_at_business_dte(maturity)
                        for trading_date in initialization_dates
                    ]
                ),
                config.volatility_quantiles,
                method="linear",
            )
        )
        for maturity in maturities
    }


def summarize_daily_results(
    results: tuple[HedgeSmokeResult, ...],
    *,
    volatility_by_origin: dict[tuple[str, int], float],
    volatility_thresholds: dict[int, tuple[float, float]],
    exposure_by_result: dict[tuple[str, str], tuple[float, float, float]] | None = None,
    config: DailyResultsConfig = DailyResultsConfig(),
) -> tuple[DailyResultSlice, ...]:
    """Return overall, annual, regime, and annual-by-regime claim slices."""
    if not results:
        raise ValueError("daily results must be non-empty")
    target_maturity = {
        target.label: target.maturity_business_days for target in daily_digital_targets()
    }
    unknown = {result.claim for result in results} - set(target_maturity)
    if unknown:
        raise ValueError(f"daily results contain unknown claims {sorted(unknown)}")
    if exposure_by_result is None:
        exposure_by_result = {
            (result.origin_date, result.claim): (
                sum(abs(value) for value in result.actions.values()),
                sum(abs(value) for value in result.actions.values()),
                sum(result.actions.values()),
            )
            for result in results
        }
    grouped: dict[str, list[tuple[HedgeSmokeResult, int, str]]] = {}
    for result in results:
        maturity = target_maturity[result.claim]
        try:
            volatility = volatility_by_origin[(result.origin_date, maturity)]
            thresholds = volatility_thresholds[maturity]
        except KeyError as error:
            raise ValueError("daily volatility metadata is incomplete") from error
        regime = _volatility_regime(volatility, thresholds)
        grouped.setdefault(result.claim, []).append(
            (result, date.fromisoformat(result.origin_date).year, regime)
        )

    slices: list[DailyResultSlice] = []
    for claim in sorted(grouped):
        rows = grouped[claim]
        years = sorted({year for _, year, _ in rows})
        regimes = ("low", "medium", "high")
        specifications: list[tuple[str, int | None, str | None]] = [("overall", None, None)]
        specifications.extend((f"year:{year}", year, None) for year in years)
        specifications.extend((f"regime:{regime}", None, regime) for regime in regimes)
        specifications.extend(
            (f"year:{year}/regime:{regime}", year, regime)
            for year in years
            for regime in regimes
        )
        for label, selected_year, selected_regime in specifications:
            selected = tuple(
                result
                for result, year, regime in rows
                if (selected_year is None or year == selected_year)
                and (selected_regime is None or regime == selected_regime)
            )
            if not selected:
                continue
            slices.append(
                _summarize_slice(
                    selected,
                    claim=claim,
                    slice_label=label,
                    year=selected_year,
                    volatility_regime=selected_regime,
                    exposure_by_result=exposure_by_result,
                    config=config,
                )
            )
    return tuple(slices)


def _summarize_slice(
    results: tuple[HedgeSmokeResult, ...],
    *,
    claim: str,
    slice_label: str,
    year: int | None,
    volatility_regime: str | None,
    exposure_by_result: dict[tuple[str, str], tuple[float, float, float]],
    config: DailyResultsConfig,
) -> DailyResultSlice:
    actions = tuple(result.actions for result in results)
    exposures = tuple(
        exposure_by_result[(result.origin_date, result.claim)] for result in results
    )
    cap_hit_results = sum(result.cap_hits > 0 for result in results)
    nonconverged_results = sum(not result.converged for result in results)
    return DailyResultSlice(
        claim=claim,
        slice_label=slice_label,
        year=year,
        volatility_regime=volatility_regime,
        unhedged_loss=_loss_statistics(
            -np.asarray([result.unhedged_pnl for result in results]),
            config=config,
        ),
        economic_loss=_loss_statistics(
            -np.asarray([result.hedged_pnl for result in results]),
            config=config,
        ),
        pure_hedge_loss=_loss_statistics(
            -np.asarray([result.pure_hedge_error for result in results]),
            config=config,
        ),
        pricing_loss=_loss_statistics(
            -np.asarray([result.pricing_difference for result in results]),
            config=config,
        ),
        mean_transaction_cost=float(
            np.mean([result.transaction_cost for result in results])
        ),
        mean_action_l1=float(
            np.mean([sum(abs(value) for value in action.values()) for action in actions])
        ),
        mean_entry_turnover_notional=float(np.mean([value[0] for value in exposures])),
        mean_gross_exposure=float(np.mean([value[1] for value in exposures])),
        mean_net_exposure=float(np.mean([value[2] for value in exposures])),
        mean_absolute_net_exposure=float(
            np.mean([abs(value[2]) for value in exposures])
        ),
        mean_nonzero_actions=float(
            np.mean([sum(abs(value) > 1.0e-14 for value in action.values()) for action in actions])
        ),
        nonconverged_results=nonconverged_results,
        nonconvergence_rate=nonconverged_results / len(results),
        cap_hits=sum(result.cap_hits for result in results),
        numerical_cap_hit_result_rate=cap_hit_results / len(results),
    )


def _loss_statistics(
    losses: FloatArray,
    *,
    config: DailyResultsConfig,
) -> DailyLossStatistics:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("daily losses must be a non-empty finite vector")
    mean = float(np.mean(values))
    sample_standard_deviation = (
        0.0 if len(values) == 1 else float(np.std(values, ddof=1))
    )
    standard_error = (
        0.0
        if len(values) == 1
        else hac_standard_error_of_mean(
            values,
            lags=min(config.hac_lags, len(values) - 1),
            kernel=config.hac_kernel,
        )
    )
    hac_lags = min(config.hac_lags, len(values) - 1)
    quantile_95 = float(np.quantile(values, 0.95, method="linear"))
    quantile_99 = float(np.quantile(values, 0.99, method="linear"))
    return DailyLossStatistics(
        observations=len(values),
        mean=mean,
        standard_error=standard_error,
        standard_error_method="HAC standard error of the sample mean",
        hac_kernel=config.hac_kernel.value,
        hac_lags=hac_lags,
        hac_lag_unit="selected-observation order within this reported slice",
        confidence_interval_95=(
            mean - _NORMAL_95 * standard_error,
            mean + _NORMAL_95 * standard_error,
        ),
        confidence_interval_99=(
            mean - _NORMAL_99 * standard_error,
            mean + _NORMAL_99 * standard_error,
        ),
        median=float(np.median(values)),
        sample_standard_deviation=sample_standard_deviation,
        root_mean_square=float(np.sqrt(np.mean(values**2))),
        mean_absolute_loss=float(np.mean(np.abs(values))),
        quantile_95=quantile_95,
        quantile_99=quantile_99,
        expected_shortfall_95=float(np.mean(values[values >= quantile_95])),
        expected_shortfall_99=float(np.mean(values[values >= quantile_99])),
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


def _volatility_by_origin(
    panel: PublicHedgingPanel,
    *,
    results: tuple[HedgeSmokeResult, ...],
) -> dict[tuple[str, int], float]:
    maturities_by_origin: dict[str, set[int]] = {}
    target_maturity = {
        target.label: target.maturity_business_days for target in daily_digital_targets()
    }
    for result in results:
        maturities_by_origin.setdefault(result.origin_date, set()).add(
            target_maturity[result.claim]
        )
    return {
        (origin, maturity): panel.market_day(
            date.fromisoformat(origin)
        ).atm_vol_at_business_dte(maturity)
        for origin, maturities in maturities_by_origin.items()
        for maturity in maturities
    }


def _entry_exposure_by_result(
    panel: PublicHedgingPanel,
    *,
    results: tuple[HedgeSmokeResult, ...],
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Mark each flat-book entry action in normalized origin-date notional."""
    snapshots = {
        origin: panel.market_day(date.fromisoformat(origin)).hedge_snapshot()
        for origin in {result.origin_date for result in results}
    }
    exposures: dict[tuple[str, str], tuple[float, float, float]] = {}
    for result in results:
        snapshot = snapshots[result.origin_date]
        quotes = snapshot.quote_by_id()
        signed_values: list[float] = []
        for instrument_id, quantity in result.actions.items():
            if instrument_id == "spot":
                price = snapshot.normalized_spot
            else:
                try:
                    price = quotes[instrument_id].mark
                except KeyError as error:
                    raise ValueError(
                        f"daily action references unknown origin quote {instrument_id!r}"
                    ) from error
            signed_values.append(quantity * price)
        gross = float(sum(abs(value) for value in signed_values))
        net = float(sum(signed_values))
        key = (result.origin_date, result.claim)
        if key in exposures:
            raise ValueError("daily results contain duplicate origin/claim records")
        exposures[key] = (gross, gross, net)
    return exposures


def _volatility_regime(
    volatility: float,
    thresholds: tuple[float, float],
) -> str:
    lower, upper = thresholds
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("daily ATM volatility must be finite-positive")
    if not 0.0 < lower <= upper:
        raise ValueError("daily volatility thresholds are invalid")
    if volatility <= lower:
        return "low"
    if volatility <= upper:
        return "medium"
    return "high"


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload
