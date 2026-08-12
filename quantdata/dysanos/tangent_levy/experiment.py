"""End-to-end orchestration for Tangent Lévy historical experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify, verify_inp
from cdxcore.subdir import SubDir
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.verbose import Context

from quantdata.biz_calendar import get_half_days, to_date
from quantdata.dysanos import _locations_dysanos

from .config import TangentLevyConfig
from .double_exponential import DoubleExponentialParameters, call_prices
from .dynamics import DynamicsFit, fit_dynamics
from .markets import (
    MarketAudit,
    TangentLevyMarketDay,
    load_market_day,
    market_dates,
)
from .pricing import density_call_prices
from .simulator import SimulationResult, simulate_codebooks
from .static_fit import (
    DayFit,
    MaturityFit,
    SymmetryCurve,
    fit_market_day,
    fitted_average_density,
    reconstruct_forward_density,
)

_history_cache = _locations_dysanos.training("tangent-levy/history")


@dataclass(frozen=True)
class HistoricalCalibration:
    """Successful daily code-books with failures preserved in the audit."""

    ticker: str
    requested_dates: tuple[date, ...]
    dates: tuple[date, ...]
    day_fits: tuple[DayFit, ...]
    symmetry_curve: SymmetryCurve
    forward_density: np.ndarray
    consecutive_mask: np.ndarray
    reconstruction_relative_error: np.ndarray
    reconstruction_l1_error: np.ndarray
    reconstruction_repricing_rmse: np.ndarray
    reconstruction_maximum_price_error: np.ndarray
    reconstruction_passed: np.ndarray
    audits: tuple[MarketAudit, ...]
    failure_reasons: dict[str, str]

    @property
    def successful_date_fraction(self) -> float:
        """Fraction of requested sessions producing a code-book."""

        return len(self.dates) / len(self.requested_dates)

    def to_state_dict(self) -> dict[str, object]:
        """Return an HDF5-safe logical artifact."""

        fit_states: list[dict[str, object]] = []
        for day_fit in self.day_fits:
            fit_states.append(
                {
                    "market_date": str(day_fit.market_date),
                    "initial_symmetry_fit": day_fit.initial_symmetry_fit,
                    "maturities": day_fit.maturities,
                    "parameters": np.asarray(
                        [
                            (
                                fit.parameters.intensity,
                                fit.parameters.positive_rate,
                                fit.parameters.negative_rate,
                                fit.parameters.positive_probability,
                            )
                            for fit in day_fit.fits
                        ],
                        dtype=np.float64,
                    ),
                    "objective": np.asarray(
                        [fit.objective for fit in day_fit.fits],
                        dtype=np.float64,
                    ),
                    "success": np.asarray(
                        [fit.success for fit in day_fit.fits],
                        dtype=bool,
                    ),
                    "iterations": np.asarray(
                        [fit.iterations for fit in day_fit.fits],
                        dtype=np.int32,
                    ),
                    "selected_start": np.asarray(
                        [fit.selected_start for fit in day_fit.fits],
                        dtype=np.int32,
                    ),
                    "start_objectives": [
                        fit.start_objectives for fit in day_fit.fits
                    ],
                    "num_quotes": np.asarray(
                        [fit.num_quotes for fit in day_fit.fits],
                        dtype=np.int32,
                    ),
                    "weighted_rmse": np.asarray(
                        [fit.weighted_rmse for fit in day_fit.fits],
                        dtype=np.float64,
                    ),
                    "maximum_absolute_error": np.asarray(
                        [fit.maximum_absolute_error for fit in day_fit.fits],
                        dtype=np.float64,
                    ),
                    "messages": [fit.message for fit in day_fit.fits],
                }
            )
        return {
            "ticker": self.ticker,
            "requested_dates": [value.isoformat() for value in self.requested_dates],
            "dates": [value.isoformat() for value in self.dates],
            "day_fits": fit_states,
            "symmetry_maturities": self.symmetry_curve.maturities,
            "symmetry_values": self.symmetry_curve.values,
            "forward_density": self.forward_density,
            "consecutive_mask": self.consecutive_mask,
            "reconstruction_relative_error": self.reconstruction_relative_error,
            "reconstruction_l1_error": self.reconstruction_l1_error,
            "reconstruction_repricing_rmse": self.reconstruction_repricing_rmse,
            "reconstruction_maximum_price_error": (
                self.reconstruction_maximum_price_error
            ),
            "reconstruction_passed": self.reconstruction_passed,
            "audits": [
                {**audit.__dict__, "market_date": audit.market_date.isoformat()}
                for audit in self.audits
            ],
            "failure_reasons": self.failure_reasons,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> HistoricalCalibration:
        """Reconstruct a cached historical calibration artifact."""

        day_fits: list[DayFit] = []
        for day_state_object in state["day_fits"]:
            day_state = dict(day_state_object)
            maturities = np.asarray(day_state["maturities"], dtype=np.float64)
            parameter_values = np.asarray(day_state["parameters"], dtype=np.float64)
            objectives = np.asarray(day_state["objective"], dtype=np.float64)
            successes = np.asarray(day_state["success"], dtype=bool)
            iterations = np.asarray(day_state["iterations"], dtype=np.int32)
            selected_starts = np.asarray(day_state["selected_start"], dtype=np.int32)
            num_quotes = np.asarray(day_state["num_quotes"], dtype=np.int32)
            weighted_rmse = np.asarray(day_state["weighted_rmse"], dtype=np.float64)
            maximum_error = np.asarray(
                day_state["maximum_absolute_error"], dtype=np.float64
            )
            start_objectives = day_state["start_objectives"]
            messages = day_state["messages"]
            maturity_fits: list[MaturityFit] = []
            for index, tau in enumerate(maturities):
                parameters = DoubleExponentialParameters(
                    intensity=float(parameter_values[index, 0]),
                    positive_rate=float(parameter_values[index, 1]),
                    negative_rate=float(parameter_values[index, 2]),
                    positive_probability=float(parameter_values[index, 3]),
                )
                maturity_fits.append(
                    MaturityFit(
                        tau=float(tau),
                        parameters=parameters,
                        objective=float(objectives[index]),
                        success=bool(successes[index]),
                        iterations=int(iterations[index]),
                        selected_start=int(selected_starts[index]),
                        start_objectives=np.asarray(
                            start_objectives[index], dtype=np.float64
                        ),
                        num_quotes=int(num_quotes[index]),
                        weighted_rmse=float(weighted_rmse[index]),
                        maximum_absolute_error=float(maximum_error[index]),
                        message=str(messages[index]),
                    )
                )
            day_fits.append(
                DayFit(
                    market_date=to_date(day_state["market_date"]),
                    fits=tuple(maturity_fits),
                    initial_symmetry_fit=bool(day_state["initial_symmetry_fit"]),
                )
            )
        audits = tuple(
            MarketAudit(
                **{
                    **dict(audit_state),
                    "market_date": to_date(dict(audit_state)["market_date"]),
                }
            )
            for audit_state in state["audits"]
        )
        return cls(
            ticker=str(state["ticker"]),
            requested_dates=tuple(to_date(value) for value in state["requested_dates"]),
            dates=tuple(to_date(value) for value in state["dates"]),
            day_fits=tuple(day_fits),
            symmetry_curve=SymmetryCurve(
                maturities=np.asarray(state["symmetry_maturities"], dtype=np.float64),
                values=np.asarray(state["symmetry_values"], dtype=np.float64),
            ),
            forward_density=np.asarray(state["forward_density"], dtype=np.float64),
            consecutive_mask=np.asarray(state["consecutive_mask"], dtype=bool),
            reconstruction_relative_error=np.asarray(
                state["reconstruction_relative_error"], dtype=np.float64
            ),
            reconstruction_l1_error=np.asarray(
                state["reconstruction_l1_error"], dtype=np.float64
            ),
            reconstruction_repricing_rmse=np.asarray(
                state["reconstruction_repricing_rmse"], dtype=np.float64
            ),
            reconstruction_maximum_price_error=np.asarray(
                state["reconstruction_maximum_price_error"], dtype=np.float64
            ),
            reconstruction_passed=np.asarray(
                state["reconstruction_passed"], dtype=bool
            ),
            audits=audits,
            failure_reasons={
                str(key): str(value)
                for key, value in dict(state["failure_reasons"]).items()
            },
        )


@dataclass(frozen=True)
class CoreExperimentResult:
    """Calibrated history, fitted dynamics, and optional Q simulation."""

    history: HistoricalCalibration
    dynamics: DynamicsFit
    simulation: SimulationResult | None
    dynamics_end: date


def _missing_audit(market_date: date, reason: str) -> MarketAudit:
    return MarketAudit(
        market_date=market_date,
        early_close=market_date in get_half_days(market_date, market_date),
        available=False,
        input_expiries=0,
        retained_expiries=0,
        retained_min_tau=float("nan"),
        retained_max_tau=float("nan"),
        input_quotes=0,
        active_quotes=0,
        excluded_american_quotes=0,
        option_cap_bound=False,
        reason=reason,
    )


def _reconstruction_repricing_error(
    market: TangentLevyMarketDay,
    day_fit: DayFit,
    *,
    projected_average_density: np.ndarray,
    jump_x: np.ndarray,
    config: TangentLevyConfig,
) -> tuple[float, float]:
    """Measure the price change caused only by density reconstruction."""

    normalized_errors: list[np.ndarray] = []
    price_errors: list[np.ndarray] = []
    for fit_index, fit in enumerate(day_fit.fits):
        matches = np.flatnonzero(
            np.isclose(market.expiries, fit.tau, rtol=0.0, atol=1.0e-8)
        )
        verify(
            matches.size == 1,
            lambda fit=fit: (
                f"fit maturity {fit.tau} does not uniquely match the market grid"
            ),
        )
        market_index = int(matches[0])
        strikes = market.strikes[market_index]
        log_strikes = np.log(strikes)
        fitted_calls = call_prices(
            tau=fit.tau,
            log_strikes=log_strikes,
            parameters=fit.parameters,
            config=config.static_fit.pricing,
        )
        reconstructed_calls = density_call_prices(
            tau=fit.tau,
            log_strikes=log_strikes,
            jump_x=jump_x,
            average_density=projected_average_density[fit_index],
            config=config.static_fit.pricing,
        )
        active = market.model_masks[market_index]
        half_spread = np.maximum(
            0.5
            * (
                market.pure_asks[market_index]
                - market.pure_bids[market_index]
            ),
            config.static_fit.half_spread_floor,
        )
        errors = reconstructed_calls[active] - fitted_calls[active]
        normalized_errors.append(errors / half_spread[active])
        price_errors.append(errors)
    normalized = np.concatenate(normalized_errors)
    price = np.concatenate(price_errors)
    return (
        float(np.sqrt(np.mean(np.square(normalized)))),
        float(np.max(np.abs(price))),
    )


def calibrate_history(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: TangentLevyConfig,
    verbose: Context | None = Context.all,
    maximum_dates: int | None = None,
) -> HistoricalCalibration:
    """Calibrate all available days and reconstruct a fixed-grid code-book panel."""

    verify_inp(
        isinstance(ticker, str) and len(ticker) > 0,
        lambda: f"'ticker' must be nonempty, got {ticker!r}",
    )
    if maximum_dates is not None:
        verify_inp(
            isinstance(maximum_dates, int) and maximum_dates > 0,
            lambda: f"'maximum_dates' must be positive, got {maximum_dates!r}",
        )
    requested = market_dates(start, end)
    if maximum_dates is not None:
        requested = requested[:maximum_dates]
    verify_inp(
        len(requested) > 0,
        lambda: f"no exchange sessions in requested range {start} through {end}",
    )
    requested_positions = {market_date: index for index, market_date in enumerate(requested)}
    tau = np.asarray(config.dynamics.tau_grid, dtype=np.float64)
    jump_x = np.asarray(config.dynamics.jump_x_grid, dtype=np.float64)
    symmetry_curve: SymmetryCurve | None = None
    previous_fit: DayFit | None = None
    dates: list[date] = []
    fits: list[DayFit] = []
    densities: list[np.ndarray] = []
    relative_errors: list[float] = []
    l1_errors: list[float] = []
    repricing_errors: list[float] = []
    maximum_price_errors: list[float] = []
    reconstruction_passed: list[bool] = []
    audits: list[MarketAudit] = []
    failure_reasons: dict[str, str] = {}

    for market_date in requested:
        try:
            market = load_market_day(
                ticker,
                market_date,
                config=config.market,
                verbose=verbose,
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            reason = f"market generation failed: {error}"
            audits.append(_missing_audit(market_date, reason))
            failure_reasons[market_date.isoformat()] = reason
            continue
        if market is None:
            reason = "canonical smoothed market unavailable or outside configured domain"
            audits.append(_missing_audit(market_date, reason))
            failure_reasons[market_date.isoformat()] = reason
            continue
        audits.append(market.audit)
        try:
            day_fit = fit_market_day(
                market,
                config=config.static_fit,
                symmetry_curve=symmetry_curve,
                warm_start=previous_fit,
            )
            candidate_symmetry_curve = (
                day_fit.symmetry_curve()
                if symmetry_curve is None
                else symmetry_curve
            )
            eta = fitted_average_density(day_fit, jump_x)
            reconstruction = reconstruct_forward_density(
                fitted_tau=day_fit.maturities,
                fitted_average_density=eta,
                target_tau=tau,
                jump_x=jump_x,
                relative_tolerance=config.static_fit.reconstruction_relative_tolerance,
                forward_density_floor=config.static_fit.forward_density_floor,
            )
            repricing_rmse, maximum_price_error = _reconstruction_repricing_error(
                market,
                day_fit,
                projected_average_density=(
                    reconstruction.projected_fitted_average_density
                ),
                jump_x=jump_x,
                config=config,
            )
        except (RuntimeError, ValueError) as error:
            failure_reasons[market_date.isoformat()] = str(error)
            previous_fit = None
            continue
        dates.append(market_date)
        fits.append(day_fit)
        densities.append(reconstruction.forward_density)
        relative_errors.append(reconstruction.maximum_relative_projection)
        l1_errors.append(reconstruction.relative_l1_projection)
        repricing_errors.append(repricing_rmse)
        maximum_price_errors.append(maximum_price_error)
        reconstruction_passed.append(
            reconstruction.passed_tolerance
            and repricing_rmse
            <= config.static_fit.reconstruction_repricing_tolerance
        )
        symmetry_curve = candidate_symmetry_curve
        previous_fit = day_fit

    verify(
        symmetry_curve is not None and len(dates) >= 3,
        lambda: (
            "history calibration requires at least three successful dates, got "
            f"{len(dates)}; failures={failure_reasons}"
        ),
    )
    positions = np.asarray([requested_positions[value] for value in dates], dtype=np.int64)
    consecutive = np.diff(positions) == 1
    return HistoricalCalibration(
        ticker=ticker,
        requested_dates=tuple(requested),
        dates=tuple(dates),
        day_fits=tuple(fits),
        symmetry_curve=symmetry_curve,
        forward_density=np.stack(densities, axis=0),
        consecutive_mask=consecutive,
        reconstruction_relative_error=np.asarray(relative_errors),
        reconstruction_l1_error=np.asarray(l1_errors),
        reconstruction_repricing_rmse=np.asarray(repricing_errors),
        reconstruction_maximum_price_error=np.asarray(maximum_price_errors),
        reconstruction_passed=np.asarray(reconstruction_passed, dtype=bool),
        audits=tuple(audits),
        failure_reasons=failure_reasons,
    )


def _history_cache_uid(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: TangentLevyConfig,
    maximum_dates: int | None,
) -> str:
    scope = f"{ticker}/{to_date(start)}--{to_date(end)}"
    if maximum_dates is not None:
        scope += f"-first-{maximum_dates}"
    history_key = named_unique_filename48_8(
        config.label,
        config.user_version,
        config.market,
        config.static_fit,
        config.dynamics,
    )
    return f"{scope}/{history_key}"


@_history_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.7",
    dependencies=[TangentLevyConfig],
    uid=_history_cache_uid,
    exclude_args=["verbose"],
)
def _cached_history_state(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: TangentLevyConfig,
    verbose: Context | None,
    maximum_dates: int | None,
) -> dict[str, object]:
    return calibrate_history(
        ticker,
        start,
        end,
        config=config,
        verbose=verbose,
        maximum_dates=maximum_dates,
    ).to_state_dict()


def load_calibrated_history(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: TangentLevyConfig,
    verbose: Context | None = Context.all,
    maximum_dates: int | None = None,
) -> HistoricalCalibration:
    """Load or generate the versioned cached historical calibration."""

    state = _cached_history_state(
        ticker,
        start,
        end,
        config=config,
        verbose=verbose,
        maximum_dates=maximum_dates,
    )
    return HistoricalCalibration.from_state_dict(state)


def validate_reconstruction_gate(
    history: HistoricalCalibration,
    *,
    dynamics_position: int,
    include_simulation_origin: bool,
) -> None:
    """Fail before dynamics consume a materially altered static code-book."""

    verify_inp(
        0 < dynamics_position <= len(history.dates),
        lambda: (
            "'dynamics_position' must identify a nonempty history prefix, got "
            f"{dynamics_position} for {len(history.dates)} dates"
        ),
    )
    required = np.arange(dynamics_position, dtype=np.int64)
    if include_simulation_origin:
        required = np.unique(
            np.concatenate([required, np.asarray([len(history.dates) - 1])])
        )
    failed = required[~history.reconstruction_passed[required]]
    verify(
        failed.size == 0,
        lambda: (
            "cumulative-density reconstruction gate failed for "
            f"{failed.size} of {required.size} required dates; examples="
            f"{[history.dates[index].isoformat() for index in failed[:5]]}, "
            "maximum relative-L1 projection="
            f"{float(np.max(history.reconstruction_l1_error[failed])):.6g}, "
            "maximum repricing RMSE="
            f"{float(np.max(history.reconstruction_repricing_rmse[failed])):.6g} "
            "half-spreads. Set DynamicsConfig(require_reconstruction_pass=False) "
            "only for an explicitly labeled diagnostic run."
        ),
    )


def run_core_experiment(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: TangentLevyConfig,
    verbose: Context | None = Context.all,
    maximum_dates: int | None = None,
    dynamics_end: date | str | None = None,
    simulate: bool = True,
) -> CoreExperimentResult:
    """Calibrate history, fit constrained factors, and optionally simulate Q."""

    history = load_calibrated_history(
        ticker,
        start,
        end,
        config=config,
        verbose=verbose,
        maximum_dates=maximum_dates,
    )
    dynamics_position = (
        len(history.dates)
        if dynamics_end is None
        else sum(value <= to_date(dynamics_end) for value in history.dates)
    )
    verify_inp(
        dynamics_position >= config.dynamics.minimum_observations + 1,
        lambda: (
            "dynamics cutoff leaves too few calibrated dates: "
            f"{dynamics_position} through {dynamics_end}"
        ),
    )
    if config.dynamics.require_reconstruction_pass:
        validate_reconstruction_gate(
            history,
            dynamics_position=dynamics_position,
            include_simulation_origin=simulate,
        )
    dynamics = fit_dynamics(
        history.forward_density[:dynamics_position],
        config=config.dynamics,
        consecutive_mask=history.consecutive_mask[: dynamics_position - 1],
    )
    simulation = (
        simulate_codebooks(
            history.forward_density[-1],
            dynamics,
            config=config.simulation,
        )
        if simulate
        else None
    )
    return CoreExperimentResult(
        history=history,
        dynamics=dynamics,
        simulation=simulation,
        dynamics_end=history.dates[dynamics_position - 1],
    )
