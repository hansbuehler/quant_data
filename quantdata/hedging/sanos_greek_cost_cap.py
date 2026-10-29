"""Reoptimize the ordinary eight-factor SANOS Greek hedge under entry-cost caps."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from quantdata.hedging.data.rolling_sanos_factors import fit_rolling_sanos_factors
from quantdata.hedging.data.sanos_state_greeks import finite_difference_sanos_state_greeks
from quantdata.hedging.sanos_greek_week import (
    _greek_block,
    _load_raw_state,
    _price_function,
    _rolling_window,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel

FloatArray = NDArray[np.float64]
PERCENTAGES = tuple(range(100, -1, -10))


class FactorCostCapSolver:
    """Minimize predicted factor P&L variance with a hard entry-cost budget."""

    def __init__(
        self,
        *,
        option_factor_exposures: FloatArray,
        option_spot_deltas: FloatArray,
        factor_covariance: FloatArray,
        option_costs_kusd: FloatArray,
    ) -> None:
        exposures = np.asarray(option_factor_exposures, dtype=np.float64)
        deltas = np.asarray(option_spot_deltas, dtype=np.float64)
        covariance = np.asarray(factor_covariance, dtype=np.float64)
        costs = np.asarray(option_costs_kusd, dtype=np.float64)
        instrument_count, factor_rank = exposures.shape
        if (
            deltas.shape != (instrument_count,)
            or covariance.shape != (factor_rank, factor_rank)
            or costs.shape != (instrument_count,)
            or np.any(costs <= 0.0)
            or not all(np.all(np.isfinite(values)) for values in (exposures, deltas, covariance, costs))
        ):
            raise ValueError("Factor, spot, covariance, and cost inputs must be finite and aligned")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.min(eigenvalues) <= 1.0e-10:
            raise ValueError("Factor covariance must be positive definite")
        covariance_square_root = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        self.covariance_square_root = covariance_square_root
        self.weighted_options = 1_000.0 * covariance_square_root @ exposures.T
        self.weighted_target = cp.Parameter(factor_rank)
        self.target_spot_delta = cp.Parameter()
        self.budget = cp.Parameter(nonneg=True)
        self.position_limit = cp.Parameter(nonneg=True)
        self.option_costs_kusd = costs
        self.option_spot_deltas = deltas
        self.action = cp.Variable(instrument_count)
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.weighted_target + self.weighted_options @ self.action)),
            [
                cp.norm1(cp.multiply(costs, self.action)) <= self.budget,
                cp.abs(self.action) <= self.position_limit,
                cp.abs(self.target_spot_delta + deltas @ self.action) <= self.position_limit,
            ],
        )

    def risk(self, action: FloatArray) -> float:
        residual = self.weighted_target.value + self.weighted_options @ action
        return float(residual @ residual)

    def solve(
        self,
        *,
        target_factor_exposures: FloatArray,
        target_spot_delta: float,
        claim_quantity: float,
        original_action_per_claim: FloatArray,
        budget_kusd: float,
        position_cap: float = 1_000.0,
    ) -> tuple[FloatArray, str, float]:
        if not (math.isfinite(claim_quantity) and claim_quantity > 0.0):
            raise ValueError("Claim quantity must be positive")
        if not (math.isfinite(budget_kusd) and budget_kusd >= 0.0):
            raise ValueError("The entry-cost budget must be non-negative")
        original = np.asarray(original_action_per_claim, dtype=np.float64)
        if original.shape != (len(self.option_costs_kusd),) or not np.all(np.isfinite(original)):
            raise ValueError("Original option action has the wrong shape")
        original_cost = float(self.option_costs_kusd @ np.abs(original))
        self.weighted_target.value = 1_000.0 * self.covariance_square_root @ np.asarray(
            target_factor_exposures, dtype=np.float64
        )
        self.target_spot_delta.value = float(target_spot_delta)
        self.budget.value = float(budget_kusd)
        self.position_limit.value = float(position_cap / claim_quantity)
        if budget_kusd >= original_cost - 1.0e-8:
            return original.copy(), "original-exact-hedge", self.risk(original)
        if budget_kusd == 0.0:
            action = np.zeros_like(original)
            return action, "spot-only", self.risk(action)

        scaled = original * (budget_kusd / original_cost)
        reference_risk = self.risk(scaled)
        self.action.value = scaled
        solver_status = ""
        for solver, options in (
            (cp.CLARABEL, {"tol_gap_abs": 1.0e-8, "tol_feas": 1.0e-8,
                           "max_iter": 500}),
            (cp.OSQP, {"eps_abs": 1.0e-7, "eps_rel": 1.0e-7, "max_iter": 100_000,
                       "polishing": True}),
        ):
            try:
                self.problem.solve(solver=solver, warm_start=True, **options)
            except cp.SolverError as error:
                solver_status = f"{solver}: {error}"
                continue
            solver_status = f"{solver}: {self.problem.status}"
            if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                continue
            action = np.asarray(self.action.value, dtype=np.float64)
            predicted_risk = self.risk(action)
            cost = float(self.option_costs_kusd @ np.abs(action))
            limit = position_cap / claim_quantity
            spot_action = -target_spot_delta - self.option_spot_deltas @ action
            if (
                np.all(np.isfinite(action))
                and cost <= budget_kusd + 1.0e-5
                and np.max(np.abs(action)) <= limit + 1.0e-5
                and abs(spot_action) <= limit + 1.0e-5
                and predicted_risk <= reference_risk + 1.0e-3 * max(1.0, reference_risk)
            ):
                return action, solver_status, predicted_risk
        raise RuntimeError(
            f"Budgeted Greek hedge failed: {solver_status}; budget={budget_kusd:.8g}; "
            f"reference risk={reference_risk:.8g}"
        )


def load_original_rows(directory: Path, trading_date: date) -> dict[str, dict]:
    rows = json.loads((directory / f"{trading_date.isoformat()}.json").read_text())
    selected = {row["claim"]: row for row in rows if row["model"] == "sanos-greek-ordinary"}
    if len(selected) != 10 or any(not row["converged"] or row["cap_hits"] for row in selected.values()):
        raise ValueError(f"Original ordinary Greek cohort is invalid on {trading_date}")
    return selected


def run_day(
    panel: PublicHedgingPanel,
    *,
    trading_date: date,
    state_directory: Path,
    original_directory: Path,
    maximum_original_cost_kusd: float,
) -> list[dict]:
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=trading_date, target=target)
        for target in daily_digital_targets(maturity_business_days=1)
    )
    window = _rolling_window(panel, state_directory=state_directory, origin_date=trading_date)
    factors = fit_rolling_sanos_factors(window, method="ordinary", rank=8)
    greeks = finite_difference_sanos_state_greeks(
        price=_price_function(samples),
        spot=1.0,
        raw_state=_load_raw_state(state_directory, trading_date),
        factors=factors,
    )
    option_count = len(samples[0].hedge_quotes)
    options = _greek_block(greeks, selection=slice(0, option_count), factors=factors)
    option_costs = np.asarray([quote.half_spread for quote in samples[0].hedge_quotes])
    solver = FactorCostCapSolver(
        option_factor_exposures=options.factor_vegas,
        option_spot_deltas=options.spot_deltas,
        factor_covariance=factors.spot_factor_covariance[1:, 1:],
        option_costs_kusd=1_000.0 * option_costs,
    )
    original_rows = load_original_rows(original_directory, trading_date)
    results = []
    for index, sample in enumerate(samples):
        original_row = original_rows[sample.target.label]
        quantity = float(original_row["claim_quantity"])
        target = _greek_block(
            greeks,
            selection=slice(option_count + index, option_count + index + 1),
            factors=factors,
        )
        option_action_per_claim = np.asarray([
            original_row["actions"].get(quote.contract.contract_id, 0.0) / quantity
            for quote in sample.hedge_quotes
        ], dtype=np.float64)
        original_cost_kusd = float(1_000.0 * option_costs @ np.abs(option_action_per_claim))
        if abs(original_cost_kusd - 1_000.0 * original_row["transaction_cost"] / quantity) > 1.0e-6:
            raise RuntimeError(f"Original entry costs do not match on {trading_date}: {sample.target.label}")
        original_spot_per_claim = float(original_row["actions"]["spot"] / quantity)
        original_delta_residual = (
            target.spot_deltas[0]
            + option_action_per_claim @ options.spot_deltas
            + original_spot_per_claim
        )
        original_factor_residual = target.factor_vegas[0] + option_action_per_claim @ options.factor_vegas
        if abs(original_delta_residual) > 1.0e-5 or np.linalg.norm(original_factor_residual) > 1.0e-5:
            raise RuntimeError(f"Original Greek actions do not match recomputed exposures on {trading_date}")
        original_replayed_pnl = quantity * (
            sample.claim_return
            + original_spot_per_claim * sample.transition.discounted_spot_gain
            + option_action_per_claim @ sample.hedge_returns
            - original_cost_kusd / 1_000.0
        )
        if abs(original_replayed_pnl - original_row["hedged_pnl"]) > 1.0e-7:
            raise RuntimeError(f"Original P&L does not replay on {trading_date}: {sample.target.label}")
        for method in ("global_cap", "per_hedge_fraction"):
            for percentage in PERCENTAGES:
                budget_kusd = percentage / 100.0 * (
                    maximum_original_cost_kusd if method == "global_cap" else original_cost_kusd
                )
                action, status, factor_risk = solver.solve(
                    target_factor_exposures=target.factor_vegas[0],
                    target_spot_delta=float(target.spot_deltas[0]),
                    claim_quantity=quantity,
                    original_action_per_claim=option_action_per_claim,
                    budget_kusd=budget_kusd,
                )
                spot_action_per_claim = -float(
                    target.spot_deltas[0] + action @ options.spot_deltas
                )
                entry_cost_kusd = float(1_000.0 * option_costs @ np.abs(action))
                pnl_kusd = 1_000.0 * (
                    sample.claim_return
                    + spot_action_per_claim * sample.transition.discounted_spot_gain
                    + action @ sample.hedge_returns
                    - entry_cost_kusd / 1_000.0
                )
                results.append({
                    "origin_date": trading_date.isoformat(),
                    "claim": sample.target.label,
                    "method": method,
                    "cap_percentage": percentage,
                    "budget_kusd": budget_kusd,
                    "original_entry_cost_kusd": original_cost_kusd,
                    "entry_cost_kusd": entry_cost_kusd,
                    "costed_pnl_kusd": float(pnl_kusd),
                    "predicted_factor_variance_kusd2": factor_risk,
                    "spot_action_per_claim": spot_action_per_claim,
                    "option_action_per_claim": {
                        quote.contract.contract_id: float(value)
                        for quote, value in zip(sample.hedge_quotes, action, strict=True)
                        if abs(value) > 1.0e-10
                    },
                    "solver_status": status,
                })
    return results


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["method"], row["cap_percentage"])][row["claim"]].append(row)
    summary = []
    for (method, percentage), targets in sorted(grouped.items()):
        if len(targets) != 10 or {len(cohort) for cohort in targets.values()} != {44}:
            raise ValueError(f"Incomplete cost-cap cohort: {method}, {percentage}")
        summary.append({
            "method": method,
            "cap_percentage": percentage,
            "mean_entry_cost_kusd": float(np.mean([
                np.mean([row["entry_cost_kusd"] for row in cohort])
                for cohort in targets.values()
            ])),
            "pnl_sd_kusd": float(np.mean([
                np.std([row["costed_pnl_kusd"] for row in cohort], ddof=1)
                for cohort in targets.values()
            ])),
            "mean_downside_loss_kusd": float(np.mean([
                np.mean([max(-row["costed_pnl_kusd"], 0.0) for row in cohort])
                for cohort in targets.values()
            ])),
            "mean_absolute_error_kusd": float(np.mean([
                np.mean([abs(row["costed_pnl_kusd"]) for row in cohort])
                for cohort in targets.values()
            ])),
            "mean_predicted_factor_variance_kusd2": float(np.mean([
                row["predicted_factor_variance_kusd2"]
                for cohort in targets.values() for row in cohort
            ])),
            "binding_fraction": float(np.mean([
                row["entry_cost_kusd"] < row["original_entry_cost_kusd"] - 1.0e-6
                for cohort in targets.values() for row in cohort
            ])),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--dates-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    arguments = parser.parse_args()
    dates = tuple(date.fromisoformat(value) for value in arguments.dates_file.read_text().splitlines())
    if len(dates) != 44 or len(set(dates)) != 44:
        raise ValueError("This matched cohort must contain exactly 44 distinct dates")
    original_costs = []
    for trading_date in dates:
        original_costs.extend(
            1_000.0 * row["transaction_cost"] / row["claim_quantity"]
            for row in load_original_rows(arguments.original_results, trading_date).values()
        )
    maximum_original_cost = float(np.max(original_costs))
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "records").mkdir(exist_ok=True)
    panel = PublicHedgingPanel(arguments.dataset)
    started = time.monotonic()
    for index, trading_date in enumerate(dates, start=1):
        path = arguments.output / "records" / f"{trading_date.isoformat()}.json"
        if not path.exists():
            records = run_day(
                panel,
                trading_date=trading_date,
                state_directory=arguments.states,
                original_directory=arguments.original_results,
                maximum_original_cost_kusd=maximum_original_cost,
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(records, indent=2) + "\n")
            temporary.replace(path)
        elapsed = time.monotonic() - started
        remaining = elapsed / index * (len(dates) - index)
        print(
            f"cost-cap {index}/{len(dates)} date={trading_date} "
            f"elapsed_s={elapsed:.1f} eta_s={remaining:.1f}",
            flush=True,
        )
        if arguments.stop_after is not None and index >= arguments.stop_after:
            return
    all_rows = [
        row for trading_date in dates
        for row in json.loads((arguments.output / "records" / f"{trading_date.isoformat()}.json").read_text())
    ]
    summary = {
        "schema": "reoptimized-sanos-greek-cost-cap-v1",
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "source_original_results": str(arguments.original_results),
        "state_directory": str(arguments.states),
        "dates": [value.isoformat() for value in dates],
        "maximum_original_entry_cost_kusd": maximum_original_cost,
        "method": "minimize ordinary eight-factor linearized variance, exactly neutralize spot delta, constrain option entry cost",
        "curves": summarize(all_rows),
    }
    (arguments.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
