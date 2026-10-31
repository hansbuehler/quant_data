"""Fit one as-of-date rolling DYSANOS Direct-200 Gaussian AR(1) model.

The model uses exactly ``lookback_sessions`` quote dates ending at ``as_of``.
This is an as-of-close calibration: the final date's quote surface is in the
fit, while no later date is loaded.  It is the first isolated origin; a
subsequent rolling driver must map in the prior origin's fitted weights.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    JointDirectDynamicsConfig,
    NativeStateConfig,
    PcaDecoderConfig,
)
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    compact_evaluation,
    fit_joint_direct_model,
    initialize_joint_model,
    standardize_affine_direct_model,
)
from quantdata.dysanos.flex.market import FlexMarketDay, market_day_from_payload
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.metrics import expiry_excess_statistics, surface_fit_statistics
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import (
    FitEvaluation,
    FlexTrainConfig,
    TrainingConvergenceError,
    evaluate_model,
    fit_latent_states,
    fit_native_states,
    materialize_day,
    model_checkpoint_phase,
)
from quantdata.dysanos.market import _cache as historical_sanos_cache
from quantdata.dysanos.v11.run_strict_pre2025_direct import _capped_joint_checkpoint
from quantdata.torch import BaseEnvironment


def _geometry(base: FlexGeometryConfig, *, lower: float, upper: float) -> FlexGeometryConfig:
    if not lower < 0.0 < upper:
        raise ValueError("NMS endpoints must straddle zero")
    return replace(base, nms_range=(lower, upper))


def _selected_dates(
    all_dates: tuple[date, ...], *, as_of: date, lookback_sessions: int
) -> tuple[date, ...]:
    if lookback_sessions < 52:
        raise ValueError("lookback_sessions must be at least 52")
    try:
        endpoint = all_dates.index(as_of)
    except ValueError as error:
        raise ValueError(f"as_of {as_of} is not an available market date") from error
    start = endpoint - lookback_sessions + 1
    if start < 0:
        raise ValueError("lookback precedes the available market history")
    return all_dates[start : endpoint + 1]


def _fit_config(*, epochs: int, device: str, stage: Literal["native", "decoder"]) -> FlexTrainConfig:
    """Keep the successful native and affine quote-calibration rates distinct."""
    return FlexTrainConfig(
        num_epochs=epochs,
        day_batch_size=128,
        learning_rate=0.02 if stage == "native" else 1.0e-4,
        decoder_learning_rate=None if stage == "native" else 1.0e-4,
        learning_rate_schedule="cosine",
        minimum_learning_rate_fraction=0.0,
        device=device,
        precision=32,
        checkpoint_every=5,
        report_every=25,
        compile_sanos_solver=device.startswith("cuda"),
    )


def _primary_fit_metrics(evaluation: FitEvaluation) -> dict[str, int | float]:
    """Expose the quote-containment criterion ahead of auxiliary losses.

    Excess is measured in observed half-spread units and is conditional on a
    quote falling outside its bid--ask interval.  The full evaluation remains
    in the report as well; this small panel makes the selection criterion
    unambiguous for rolling-model comparisons.
    """
    statistics = evaluation.statistics.all
    return {
        "num_quotes": statistics.num_options,
        "inside_bid_ask_percent": statistics.percent_zero,
        "num_outside_bid_ask": statistics.num_positive,
        "conditional_mean_excess_half_spreads": statistics.positive_mean,
        "conditional_median_excess_half_spreads": statistics.positive_median,
    }


def _date_evaluation(
    evaluation: FitEvaluation,
    days: tuple[FlexMarketDay, ...],
    *,
    index: int,
    objective: FlexObjectiveConfig,
) -> FitEvaluation:
    """Select already priced quotes, preserving the fitted state's date index."""
    if len(evaluation.fits) != len(days):
        raise ValueError("Evaluation and market dates must have the same length")
    fit = evaluation.fits[index]
    day = days[index]
    half_spread = 0.5 * (day.asks - day.bids)
    values = (
        np.maximum(fit - day.asks, 0.0)
        + np.maximum(day.bids - fit, 0.0)
        + objective.fit_weight * np.abs(fit - day.fit_prices)
    ) / half_spread
    return FitEvaluation(
        objective=float(np.mean(values)),
        statistics=expiry_excess_statistics(fit, day.bids, day.asks, day.quote_business_dtes),
        surface_fit=surface_fit_statistics(fit, day.fit_prices, day.bids, day.asks),
        fits=(fit,),
    )


def _progress_reporter(
    *,
    stage: str,
    epochs: int,
    report_directory: Path,
    days: tuple[FlexMarketDay, ...],
    objective: FlexObjectiveConfig,
) -> Callable[[int, FitEvaluation], None]:
    """Persist exact full-panel fit metrics at each scheduled progress report."""
    started = perf_counter()
    path = report_directory / f"{stage}-progress.json"
    history: list[dict[str, object]] = []

    def report(epoch: int, evaluation: FitEvaluation) -> None:
        metrics = _primary_fit_metrics(evaluation)
        as_of_metrics = _primary_fit_metrics(
            _date_evaluation(evaluation, days, index=-1, objective=objective)
        )
        elapsed = perf_counter() - started
        history.append({
            "epoch": epoch, "total_epochs": epochs, "elapsed_seconds": elapsed,
            "objective": evaluation.objective, **metrics, "as_of": as_of_metrics,
        })
        write_json_atomic({"stage": stage, "history": history}, path)
        print(
            f"fit-progress stage={stage} epoch={epoch}/{epochs} "
            f"inside_bid_ask={metrics['inside_bid_ask_percent']:.4f}% "
            f"conditional_mean_excess={metrics['conditional_mean_excess_half_spreads']:.6g} "
            f"conditional_median_excess={metrics['conditional_median_excess_half_spreads']:.6g} "
            f"as_of_inside={as_of_metrics['inside_bid_ask_percent']:.4f}% "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return report


def _historical_sanos_directory(
    *, ticker: str, dates: tuple[date, ...]
) -> Path:
    """Select the stored SANOS cache covering the complete rolling panel.

    Historical surfaces are immutable input data.  In particular, a rolling
    DYSANOS estimate must never trigger an arbitrage-free surface refit merely
    because the current cache-code version differs from the historical one.
    """
    cache_root = Path(historical_sanos_cache.path)
    candidates = tuple(cache_root.glob(f"*/sm_*/{ticker}"))
    coverage = [
        (sum((directory / f"{trading_date}.h5").is_file() for trading_date in dates), directory)
        for directory in candidates
    ]
    if not coverage:
        raise FileNotFoundError(f"No stored SANOS cache directory for {ticker}")
    maximum = max(count for count, _ in coverage)
    best = tuple(directory for count, directory in coverage if count == maximum)
    if maximum != len(dates):
        missing = [
            trading_date.isoformat()
            for trading_date in dates
            if not (best[0] / f"{trading_date}.h5").is_file()
        ]
        raise FileNotFoundError(
            "No single stored SANOS cache covers the requested rolling panel; "
            f"best coverage={maximum}/{len(dates)}, missing={missing}"
        )
    if len(best) != 1:
        raise RuntimeError(
            "Ambiguous historical SANOS cache directories with identical full coverage: "
            f"{[str(item) for item in best]}"
        )
    return best[0]


def _load_stored_sanos_markets(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
) -> tuple[FlexMarketDay, ...]:
    """Deserialize the certified historical SANOS surfaces without refitting."""
    directory = _historical_sanos_directory(ticker=ticker, dates=dates)
    cache_root = Path(historical_sanos_cache.path)
    days: list[FlexMarketDay] = []
    for index, trading_date in enumerate(dates, start=1):
        path = directory / f"{trading_date}.h5"
        payload = historical_sanos_cache.read(
            str(path.relative_to(cache_root).with_suffix("")),
            raise_on_error=True,
            delete_wrong_version=False,
        )
        if not isinstance(payload, dict) or payload.get("d") != trading_date:
            raise RuntimeError(f"Invalid stored SANOS payload {path}")
        day = market_day_from_payload(
            payload,
            ticker=ticker,
            geometry=geometry,
            config=market_config,
        )
        if day is None:
            raise RuntimeError(f"Stored SANOS payload is ineligible for {trading_date}")
        days.append(day)
        if index % 50 == 0 or index == len(dates):
            print(
                f"stored SANOS markets={index}/{len(dates)} date={trading_date} "
                f"source={directory}",
                flush=True,
            )
    return tuple(days)


def run(
    *,
    native_completion: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    as_of: date,
    lookback_sessions: int,
    nms_lower: float,
    nms_upper: float,
    native_epochs: int,
    decoder_epochs: int,
    joint_epochs: int,
    native_weights_warm_start_checkpoint: Path | None,
    workers: int,
    device: str,
) -> dict[str, object]:
    """Fit the quote and one-session dynamics objectives on one rolling panel."""
    completion = json.loads(native_completion.read_text(encoding="utf-8"))
    base_geometry = _geometry_from_completion(
        completion, requested_model_business_dtes=None
    )
    geometry = _geometry(base_geometry, lower=nms_lower, upper=nms_upper)
    dates = _selected_dates(
        _market_dates_from_completion(native_completion),
        as_of=as_of,
        lookback_sessions=lookback_sessions,
    )
    market_config = _market_config_from_completion(completion)
    days = _load_stored_sanos_markets(
        ticker="SPX",
        dates=dates,
        geometry=geometry,
        market_config=market_config,
    )
    env = BaseEnvironment(precision=32, device=device, debug=False)
    objective = FlexObjectiveConfig(fit_weight=0.05)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    native_checkpoint = checkpoint_directory / "native.pt"
    native_train = _fit_config(epochs=native_epochs, device=device, stage="native")
    decoder_train = _fit_config(epochs=decoder_epochs, device=device, stage="decoder")
    if native_checkpoint.exists() and model_checkpoint_phase(native_checkpoint) == "complete":
        native_train = replace(native_train, just_load=True)
    write_json_atomic({
        "as_of": as_of.isoformat(), "training_dates": [str(value) for value in dates],
        "geometry": asdict(geometry), "objective": asdict(objective),
        "native_train_config": asdict(native_train),
        "decoder_train_config": asdict(decoder_train),
        "native_completion_source": str(native_completion),
        "native_weights_warm_start": (
            None if native_weights_warm_start_checkpoint is None
            else str(native_weights_warm_start_checkpoint)
        ),
    }, report_directory / "run-manifest.json")
    native_fit = fit_native_states(
        days,
        config=NativeStateConfig(label="rolling-native", geometry=geometry),
        objective_config=objective,
        train_config=native_train,
        checkpoint_path=native_checkpoint,
        weights_only_warm_start_path=native_weights_warm_start_checkpoint,
        evaluation_callback=_progress_reporter(
            stage="native", epochs=native_epochs, report_directory=report_directory,
            days=days, objective=objective,
        ),
    )
    write_json_atomic({
        "primary_fit": _primary_fit_metrics(native_fit.evaluation),
        "convergence": native_fit.convergence, "evaluation": native_fit.evaluation.to_dict(),
    }, report_directory / "native-complete.json")
    decoder_checkpoint = checkpoint_directory / "direct-200.pt"
    if decoder_checkpoint.exists() and model_checkpoint_phase(decoder_checkpoint) == "complete":
        decoder_train = replace(decoder_train, just_load=True)
    decoder_fit = fit_latent_states(
        days,
        native_fit,
        config=PcaDecoderConfig(
            label="rolling-direct-200",
            group="dysanos/v11/rolling-direct-ar",
            state_size=200,
            geometry=geometry,
        ),
        objective_config=objective,
        supervised_train_config=decoder_train,
        market_train_config=decoder_train,
        checkpoint_path=decoder_checkpoint,
        evaluation_callback=_progress_reporter(
            stage="direct-200", epochs=decoder_epochs, report_directory=report_directory,
            days=days, objective=objective,
        ),
    )
    write_json_atomic({
        "primary_fit": _primary_fit_metrics(decoder_fit.evaluation),
        "convergence": decoder_fit.market_convergence,
        "evaluation": decoder_fit.evaluation.to_dict(),
    }, report_directory / "direct-200-complete.json")
    initialization = standardize_affine_direct_model(decoder_fit.model)
    model_config = JointDirectDynamicsConfig(
        label="rolling-daily-direct-200-gaussian-ar",
        group="dysanos/v11/rolling-direct-ar",
        state_size=200,
        geometry=geometry,
        minimum_mean_reversion_per_week=2.0e-4,
    )
    model = JointDirectDynamicsModel(env, num_days=len(days), config=model_config)
    origin = np.arange(len(days) - 1, dtype=np.int64)
    endpoint = origin + 1
    initialization_audit = initialize_joint_model(
        model, initialization, origin_indices=origin, endpoint_indices=endpoint,
        diagonal_transition=True, covariance_shrinkage=0.5,
    )
    with torch.no_grad():
        previous = np.asarray([day.spot_ref for day in days[:-1]], dtype=np.float64)
        following = np.asarray([day.spot_ref for day in days[1:]], dtype=np.float64)
        raw_returns = following / previous
        # These states use the whitened coordinates of the initialized joint
        # model. Its affine decoder preserves the original decoded surfaces.
        from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores

        spot_scores = _historical_spot_scores(
            model,
            initialization.states.numpy()[origin],
            raw_returns,
            batch_size=128,
        )
    joint_config = JointDirectTrainConfig(
        num_epochs=joint_epochs,
        day_batch_size=16,
        state_learning_rate=3.0e-5,
        decoder_learning_rate=1.0e-5,
        dynamics_learning_rate=1.0e-4,
        warmup_epochs=min(20, joint_epochs - 1),
        dynamics_ramp_epochs=min(50, joint_epochs),
        report_every=1,
        checkpoint_every=5,
        compile_sanos_solver=device.startswith("cuda"),
    )
    joint_checkpoint = checkpoint_directory / "joint-direct-200.pt"
    source_digest = checkpoint_digest(decoder_checkpoint)
    with FileLock(str(joint_checkpoint.with_suffix(".lock"))):
        try:
            history, convergence = fit_joint_direct_model(
                model,
                days,
                dates=dates,
                objective_config=objective,
                train_config=joint_config,
                checkpoint_path=joint_checkpoint,
                source_decoder_sha256=source_digest,
                env=env,
                transition_pair_indices=(origin, endpoint),
                transition_spot_scores=spot_scores,
            )
        except TrainingConvergenceError as error:
            weights, history, convergence = _capped_joint_checkpoint(
                joint_checkpoint,
                train_config=joint_config,
                model_config=model_config,
                market_fingerprints=tuple(day.training_fingerprint for day in days),
                source_decoder_sha256=source_digest,
                error=error,
            )
            model.load_state_dict(weights)

    torch_days = tuple(materialize_day(day, env=env) for day in days)
    panel_fit = evaluate_model(
        model, torch_days, env=env, objective_config=objective,
        day_batch_size=16, equal_date_weighting=True,
    )
    as_of_fit = _date_evaluation(panel_fit, days, index=-1, objective=objective)
    report: dict[str, object] = {
        "schema": "dysanos-v11-rolling-direct-ar-v1",
        "status": "complete" if convergence.get("converged") is True else "capped",
        "as_of": as_of.isoformat(),
        "training_dates": [item.isoformat() for item in dates],
        "lookback_sessions": len(dates),
        "initialization": (
            "fresh; no earlier rolling-origin checkpoint exists"
            if native_weights_warm_start_checkpoint is None
            else f"native weights warm-started from {native_weights_warm_start_checkpoint}"
        ),
        "geometry": asdict(geometry),
        "native_fit": native_fit.to_dict(),
        "native_primary_fit": _primary_fit_metrics(native_fit.evaluation),
        "decoder_fit": decoder_fit.to_dict(),
        "decoder_primary_fit": _primary_fit_metrics(decoder_fit.evaluation),
        "joint_initialization": initialization_audit,
        "joint_convergence": convergence,
        "joint_history_tail": history[-50:],
        "panel_fit": compact_evaluation(panel_fit),
        "panel_primary_fit": _primary_fit_metrics(panel_fit),
        "as_of_fit": compact_evaluation(as_of_fit),
        "as_of_primary_fit": _primary_fit_metrics(as_of_fit),
        "checkpoints": {
            "native": str(native_checkpoint),
            "decoder": str(decoder_checkpoint),
            "joint": str(joint_checkpoint),
        },
    }
    write_json_atomic(report, report_directory / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--lookback-sessions", type=int, default=252)
    parser.add_argument("--nms-lower", type=float, required=True)
    parser.add_argument("--nms-upper", type=float, required=True)
    parser.add_argument("--native-epochs", type=int, default=2000)
    parser.add_argument("--decoder-epochs", type=int, default=2000)
    parser.add_argument("--joint-epochs", type=int, default=2000)
    parser.add_argument("--native-weights-warm-start-checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(arguments.checkpoint_directory / "run.lock")):
        print(json.dumps(run(**vars(arguments)), indent=2), flush=True)


if __name__ == "__main__":
    main()
