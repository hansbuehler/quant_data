"""Matched daily SANOS hedges in the infinite risk-aversion limit."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear

from quantdata.hedging.mean_reduction import MeanReduction, reduce_one_day_means
from quantdata.hedging.objectives import regularized_tyler_covariance
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.hedger import (
    _robust_target_scatter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.pca import (
    spot_residual_pca_covariance,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
    build_historical_return_panel,
)


def minimum_variance_actions(
    instruments: np.ndarray,
    targets: np.ndarray,
    *,
    position_cap: float,
    linear_transaction_costs: np.ndarray | None = None,
    quadratic_ridge: float = 1.0e-4,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Minimize variance, then original spread/ridge on its minimizer set."""
    instruments = np.asarray(instruments, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if instruments.ndim != 2 or targets.ndim != 2 or len(instruments) != len(targets):
        raise ValueError("Instruments and targets must be aligned scenario matrices")
    if not np.all(np.isfinite(instruments)) or not np.all(np.isfinite(targets)):
        raise ValueError("Scenario matrices must be finite")
    if not np.isfinite(position_cap) or position_cap <= 0.0:
        raise ValueError("Position cap must be positive and finite")
    costs = (
        None if linear_transaction_costs is None
        else np.asarray(linear_transaction_costs, dtype=np.float64)
    )
    if costs is not None and (
        costs.shape != (instruments.shape[1],)
        or np.any(~np.isfinite(costs) | (costs < 0.0))
    ):
        raise ValueError("Transaction costs must be nonnegative and instrument-aligned")
    if not np.isfinite(quadratic_ridge) or quadratic_ridge < 0.0:
        raise ValueError("Quadratic ridge must be finite and nonnegative")
    centered_instruments = instruments - np.mean(instruments, axis=0)
    centered_targets = targets - np.mean(targets, axis=0)
    scales = np.maximum(np.sqrt(np.mean(centered_instruments**2, axis=0)), 1.0e-12)
    design = centered_instruments / scales
    scaled_actions, _, rank, singular_values = lstsq(
        design, -centered_targets, cond=1.0e-12, lapack_driver="gelsd"
    )
    actions = scaled_actions / scales[:, None]
    right_basis = None
    if costs is not None and rank < instruments.shape[1]:
        _, _, right_vectors = np.linalg.svd(design, full_matrices=False)
        right_basis = right_vectors[:rank]
    diagnostics: list[dict[str, float | int | str]] = []
    for target_index in range(targets.shape[1]):
        action = actions[:, target_index]
        solver = "lstsq"
        if np.max(np.abs(action)) > position_cap:
            bounded = lsq_linear(
                design,
                -centered_targets[:, target_index],
                bounds=(-position_cap * scales, position_cap * scales),
                method="bvls",
                tol=1.0e-10,
                max_iter=10_000,
            )
            if not bounded.success:
                raise RuntimeError(
                    f"Minimum-variance constrained solve failed for target "
                    f"{target_index}: {bounded.message}"
                )
            action = bounded.x / scales
            actions[:, target_index] = action
            solver = "bounded-lsq"
        residual = centered_targets[:, target_index] + centered_instruments @ action
        minimum_variance = float(np.mean(residual**2))
        if right_basis is not None:
            candidate = cp.Variable(instruments.shape[1])
            scaled_candidate = cp.multiply(scales, candidate)
            original_ridge_scale = np.clip(np.maximum(scales, 1.0e-8), 1.0e-6, 1.0e6)
            tie_break = cp.Problem(
                cp.Minimize(
                    costs @ cp.abs(candidate)
                    + 0.5 * quadratic_ridge
                    * cp.sum_squares(cp.multiply(original_ridge_scale, candidate))
                ),
                [
                    right_basis @ scaled_candidate == right_basis @ (scales * action),
                    candidate >= -position_cap,
                    candidate <= position_cap,
                ],
            )
            tie_break.solve(solver=cp.CLARABEL)
            if tie_break.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(
                    f"Minimum-variance tie-break failed for target {target_index}: "
                    f"{tie_break.status}"
                )
            candidate_action = np.asarray(candidate.value, dtype=np.float64)
            if not np.all(np.isfinite(candidate_action)):
                raise FloatingPointError("Minimum-variance tie-break is non-finite")
            candidate_residual = (
                centered_targets[:, target_index]
                + centered_instruments @ candidate_action
            )
            candidate_variance = float(np.mean(candidate_residual**2))
            if candidate_variance > minimum_variance + max(1.0e-12, 1.0e-6 * minimum_variance):
                raise RuntimeError(
                    f"Minimum-variance tie-break changed the primary objective "
                    f"for target {target_index}: {minimum_variance} -> {candidate_variance}"
                )
            action = candidate_action
            actions[:, target_index] = action
            solver += "+cost-ridge-tie-break"
            residual = candidate_residual
        objective = float(np.mean(residual**2))
        if not np.isfinite(objective):
            raise FloatingPointError("Minimum-variance objective is not finite")
        diagnostics.append({
            "solver": solver,
            "matrix_rank": int(rank),
            "minimum_retained_singular_value": float(singular_values[rank - 1]),
            "training_variance": objective,
            "maximum_absolute_action": float(np.max(np.abs(action))),
            "cap_hits": int(np.count_nonzero(np.isclose(np.abs(action), position_cap))),
            "tie_break_entry_cost": (
                0.0 if costs is None else float(costs @ np.abs(action))
            ),
        })
    return actions, diagnostics


def minimum_normal_actions(
    instruments: np.ndarray,
    targets: np.ndarray,
    *,
    position_cap: float,
    spot_residual_pca_rank: int | None = None,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Minimize the existing robust Tyler target-scatter variance directly."""
    estimate = regularized_tyler_covariance(instruments)
    if not estimate.converged:
        raise RuntimeError("Robust Tyler covariance did not converge")
    pca = (
        None
        if spot_residual_pca_rank is None
        else spot_residual_pca_covariance(
            estimate.covariance, rank=spot_residual_pca_rank
        )
    )
    covariance = estimate.covariance if pca is None else pca.covariance
    cholesky = np.linalg.cholesky(covariance)
    design = cholesky.T
    actions = np.empty((instruments.shape[1], targets.shape[1]), dtype=np.float64)
    diagnostics: list[dict[str, float | int | str]] = []
    for target_index in range(targets.shape[1]):
        cross, claim_variance = _robust_target_scatter(
            instruments,
            targets[:, target_index],
            estimate=estimate,
        )
        if pca is not None:
            cross = pca.project_claim_cross_covariance(cross)
        action = -np.linalg.solve(covariance, cross)
        solver = "direct-covariance"
        if np.max(np.abs(action)) > position_cap:
            response = np.linalg.solve(cholesky, cross)
            bounded = lsq_linear(
                design,
                -response,
                bounds=(-position_cap, position_cap),
                tol=1.0e-10,
                max_iter=1_000,
            )
            if not bounded.success:
                raise RuntimeError(
                    f"Robust-normal constrained solve failed for target "
                    f"{target_index}: {bounded.message}"
                )
            action = bounded.x
            solver = "bounded-covariance"
        actions[:, target_index] = action
        variance = float(
            claim_variance + 2.0 * cross @ action + action @ covariance @ action
        )
        if variance < -1.0e-10 or not np.isfinite(variance):
            raise FloatingPointError("Robust-normal residual variance is invalid")
        diagnostics.append({
            "solver": solver,
            "training_variance": max(variance, 0.0),
            "tyler_shrinkage": float(estimate.shrinkage),
            "tyler_iterations": int(estimate.iterations),
            "spot_residual_pca_rank": (
                0 if pca is None else spot_residual_pca_rank
            ),
            "explained_residual_variance": (
                1.0 if pca is None else pca.explained_residual_variance
            ),
            "maximum_absolute_action": float(np.max(np.abs(action))),
            "cap_hits": int(np.count_nonzero(np.isclose(np.abs(action), position_cap))),
        })
    return actions, diagnostics


def _reference_quantities(reference_run: Path, origin_date: date) -> dict[str, float]:
    payload = json.loads(
        (reference_run / "records" / f"{origin_date.isoformat()}.json").read_text()
    )
    results = payload if isinstance(payload, list) else payload["results"]
    return {result["claim"]: float(result["claim_quantity"]) for result in results}


def _greek_limit(greek_run: Path, origin_date: date) -> list[dict]:
    records = json.loads((greek_run / f"{origin_date.isoformat()}.json").read_text())
    result = [
        record for record in records
        if not record["model"].startswith("sanos-parametric-")
    ]
    for record in records:
        if not record["model"].startswith("sanos-greek-"):
            continue
        residual = float(record["diagnostics"]["factor_residual_norm"])
        if residual > 1.0e-8 or record["cap_hits"] != 0:
            raise RuntimeError(
                f"{origin_date}: Greek hedge is not a certified zero-variance limit"
            )
        copy = json.loads(json.dumps(record))
        copy["model"] = record["model"].replace("sanos-greek-", "sanos-parametric-")
        copy["objective_value"] = 0.0
        copy["diagnostics"]["limiting_objective"] = (
            "minimum spot/eight-factor variance; exact zero Greek residual"
        )
        result.append(copy)
    return result


def run_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    reference_run: Path,
    greek_run: Path,
    position_cap: float = 1_000.0,
) -> dict[str, list[dict]]:
    """Evaluate empirical, robust-normal, and factor limits on one market date."""
    targets = daily_digital_targets(maturity_business_days=1)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != quotes for sample in samples[1:]):
        raise RuntimeError("Targets do not share one hedge book")
    quantities_by_label = _reference_quantities(reference_run, origin_date)
    if set(quantities_by_label) != {target.label for target in targets}:
        raise ValueError("Reference target quantities do not align")
    quantities = np.asarray([quantities_by_label[target.label] for target in targets])
    returns = build_historical_return_panel(
        panel,
        origin_date=origin_date,
        option_quotes=quotes,
        targets=targets,
        count=252,
        price_source="sanos",
    )
    scaled_returns = HistoricalReturnPanel(
        training_origin_dates=returns.training_origin_dates,
        training_terminal_dates=returns.training_terminal_dates,
        instrument_ids=returns.instrument_ids,
        target_labels=returns.target_labels,
        instrument_returns=returns.instrument_returns,
        target_returns=returns.target_returns * quantities[None, :],
    )
    reduced = reduce_one_day_means(
        scaled_returns.instrument_returns,
        scaled_returns.target_returns,
        method=MeanReduction.DEMEAN,
    )
    empirical_actions, diagnostics = minimum_variance_actions(
        reduced.instrument_returns,
        reduced.target_returns,
        position_cap=position_cap,
        linear_transaction_costs=np.asarray(
            [0.0, *(quote.half_spread for quote in quotes)], dtype=np.float64
        ),
    )
    normal_actions, normal_diagnostics = minimum_normal_actions(
        scaled_returns.instrument_returns,
        scaled_returns.target_returns,
        position_cap=position_cap,
    )
    results: dict[str, list[dict]] = {"historical": [], "normal": []}
    for index, sample in enumerate(samples):
        action = empirical_actions[:, index]
        results["historical"].append(asdict(evaluate_daily_digital_hedge(
            sample,
            model="historical-sanos",
            measure="historical-P",
            spot_action=float(action[0]),
            option_actions=action[1:],
            model_price=sample.market_price,
            objective_value=float(diagnostics[index]["training_variance"]),
            converged=True,
            cap_hits=int(diagnostics[index]["cap_hits"]),
            claim_quantity=float(quantities[index]),
            diagnostics={
                **diagnostics[index],
                "risk_parameter": "infinity",
                "objective": "minimum-variance",
                "mean_reduction": "demean",
                "training_observations": len(returns.training_origin_dates),
                "instrument_count": returns.instrument_returns.shape[1],
            },
        )))
        normal_action = normal_actions[:, index]
        results["normal"].append(asdict(evaluate_daily_digital_hedge(
            sample,
            model="sanos-normal-statistical",
            measure="historical-P",
            spot_action=float(normal_action[0]),
            option_actions=normal_action[1:],
            model_price=sample.market_price,
            objective_value=float(normal_diagnostics[index]["training_variance"]),
            converged=True,
            cap_hits=int(normal_diagnostics[index]["cap_hits"]),
            claim_quantity=float(quantities[index]),
            diagnostics={
                **normal_diagnostics[index],
                "risk_parameter": "infinity",
                "objective": "minimum-variance",
                "training_observations": len(returns.training_origin_dates),
                "instrument_count": returns.instrument_returns.shape[1],
            },
        )))
    results["greek"] = _greek_limit(greek_run, origin_date)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dates-file", type=Path, required=True)
    parser.add_argument("--reference-statistical-run", type=Path, required=True)
    parser.add_argument("--greek-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-greek-only", action="store_true")
    arguments = parser.parse_args()
    dates = tuple(date.fromisoformat(value) for value in arguments.dates_file.read_text().splitlines())
    if not dates:
        raise ValueError("No evaluation dates")
    panel = PublicHedgingPanel(arguments.dataset)
    started = time.monotonic()
    for completed, origin_date in enumerate(dates, start=1):
        paths = {
            "historical": arguments.output / "historical" / "records" / f"{origin_date}.json",
            "normal": arguments.output / "normal" / "records" / f"{origin_date}.json",
            "greek": arguments.output / "greek" / f"{origin_date}.json",
        }
        if arguments.repair_greek_only:
            if not paths["historical"].exists() or not paths["normal"].exists():
                raise RuntimeError(f"{origin_date}: variance records are missing")
            results = {"greek": _greek_limit(arguments.greek_run, origin_date)}
        elif not all(path.exists() for path in paths.values()):
            results = run_day(
                panel,
                origin_date=origin_date,
                reference_run=arguments.reference_statistical_run,
                greek_run=arguments.greek_run,
            )
        else:
            results = {}
        for name, records in results.items():
            path = paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial.json")
            payload = records if name == "greek" else {"results": records}
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        elapsed = time.monotonic() - started
        remaining = elapsed * (len(dates) - completed) / completed
        print(
            f"minimum-variance progress {completed}/{len(dates)} "
            f"origin={origin_date} elapsed_s={elapsed:.1f} eta_s={remaining:.1f}",
            flush=True,
        )
    (arguments.output / "greek" / "summary.json").write_text(
        json.dumps({"dates": [value.isoformat() for value in dates], "risk_parameter": "infinity"}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
