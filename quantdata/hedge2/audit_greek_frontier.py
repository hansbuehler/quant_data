"""Independent evidence checks for unavailable Greek portfolios and retained prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from quantdata.hedge2.audit_episode_risk import replay_risk_step


def verify_unavailable_geometry(*, context: dict[str, object], evidence: dict[str, np.ndarray],
                                premium_cap: float) -> dict[str, float | str]:
    """Verify the reason without trusting the original solver's status string.

    Box infeasibility uses the saved separator evaluated on raw geometry.
    Budget infeasibility independently reprices mandatory trimming and
    constructs a feasible exact-match cost upper bound with a different LP
    formulation. Neither check executes the diagnostic witness as a hedge.
    All exposure matches use the original row normalization/tolerances.
    """
    prices, costs = evidence["normalized_premium_prices"], evidence["costs"]
    previous, exposures = evidence["previous_contract_positions"], evidence["instrument_exposures"]
    target = evidence["target_exposures"]
    side, fraction = context["side"], context["greek_budget"]
    if (prices.ndim != 1 or not len(prices) or costs.shape != prices.shape or previous.shape != prices.shape
            or exposures.ndim != 2 or exposures.shape[0] != len(prices) or target.shape != (exposures.shape[1],)
            or not all(np.isfinite(values).all() for values in (prices, costs, previous, exposures, target))
            or np.any(prices <= 0.) or np.any(costs < 0.) or side not in (-1, 1)
            or not np.isfinite(fraction) or not 0. <= fraction <= 1.
            or not np.isfinite(premium_cap) or premium_cap <= 0.
            or context.get("executed_at_failure") is not False
            or context.get("terminal_performance_available") is not False):
        raise ValueError("Unavailable Greek evidence needs finite aligned geometry and explicit nonexecution")
    matrix = (exposures / prices[:, None]).T
    signed_target = -side * target
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(signed_target))
    active = row_scale > 1e-14
    normalized = matrix[active] / row_scale[active, None]
    rhs = signed_target[active] / row_scale[active]
    if context["error_type"] == "GreekBoxInfeasible":
        separator = evidence["normalized_separator"]
        witness = evidence["primal_positions"]
        if (separator.shape != target.shape or witness.shape != prices.shape
                or not np.isfinite(separator).all() or not np.isfinite(witness).all()
                or np.any(separator[~active] != 0.)):
            raise ValueError("Box evidence requires finite correctly normalized separator and primal witness")
        direction = separator[active]
        projected_target = float(direction @ rhs)
        support_norm = float(np.abs(normalized.T @ direction).sum())
        support = premium_cap * support_norm
        roundoff = 128. * np.finfo(float).eps * (float(np.abs(direction) @ np.abs(rhs))
            + premium_cap * float((np.abs(normalized).T @ np.abs(direction)).sum()))
        allowance = 1e-6 * float(np.abs(direction).sum()) + roundoff
        margin = projected_target - support
        primal_residual = float(np.max(np.abs(normalized @ witness - rhs), initial=0.))
        lower_bound = projected_target / max(support_norm, 1.)
        maximum_position = float(np.max(np.abs(witness), initial=0.))
        if (not np.isfinite([margin, allowance, primal_residual, lower_bound, maximum_position]).all()
                or margin <= allowance or primal_residual > 1e-7
                or abs(maximum_position - lower_bound) > 1e-7 * max(1., maximum_position, abs(lower_bound))):
            raise RuntimeError("Raw geometry does not independently certify the recorded box infeasibility")
        saved = context["box_certificate"]
        if (saved["declared_cap"] != premium_cap
                or abs(saved["dual_lower_bound"] - lower_bound) > 1e-8 * max(1., abs(lower_bound))):
            raise RuntimeError("Saved box certificate disagrees with independently evaluated support")
        return {"reason": "exact_reference_outside_box", "minimum_cap_lower_bound": lower_bound,
            "separation_margin": margin, "matching_tolerance_allowance": allowance,
            "normalized_witness_residual": primal_residual}
    if context["error_type"] != "GreekBudgetInfeasible":
        raise ValueError("Generic numerical errors cannot become unavailable Greek portfolios")
    old_premium = previous * prices
    premium_costs = costs / prices
    mandatory = float(premium_costs @ np.maximum(np.abs(old_premium) - premium_cap, 0.))
    full_cost = float(context["full_entry_cost"])
    recorded_budget = float(context["turnover_budget"])
    if (not np.isfinite([mandatory, full_cost, recorded_budget]).all() or full_cost < 0.
            or context["mandatory_turnover_cost"] < 0.):
        raise ValueError("Unavailable budget metadata must be finite and nonnegative")
    instruments = len(prices)
    identity = np.eye(instruments)
    inequalities = np.r_[np.c_[identity, -identity], np.c_[-identity, -identity]]
    cost_scale = max(float(premium_costs.max()), np.finfo(float).tiny)
    result = linprog(c=np.r_[np.zeros(instruments), premium_costs / cost_scale], A_ub=inequalities,
        b_ub=np.r_[old_premium, -old_premium], A_eq=np.c_[normalized, np.zeros_like(normalized)],
        b_eq=rhs, bounds=[(-premium_cap, premium_cap)] * instruments + [(0., None)] * instruments,
        method="highs", options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not result.success or result.x is None:
        raise RuntimeError(f"Independent full-cost budget witness needs review: {result.message}")
    witness = np.asarray(result.x[:instruments], dtype=float)
    upper_cost = float(premium_costs @ np.abs(witness - old_premium))
    residual = float(np.max(np.abs(normalized @ witness - rhs), initial=0.))
    tolerance = 256. * np.finfo(float).eps * max(mandatory, upper_cost, full_cost, np.finfo(float).tiny)
    if (not np.isfinite(witness).all() or not np.isfinite([upper_cost, residual, tolerance]).all()
            or np.max(np.abs(witness), initial=0.) > premium_cap * (1. + 1e-9) or residual > 1e-7
            or abs(mandatory - context["mandatory_turnover_cost"]) > tolerance
            or abs(recorded_budget - fraction * full_cost) > tolerance
            or mandatory <= max(recorded_budget, fraction * upper_cost) + tolerance):
        raise RuntimeError("Raw inventory/full-match witness does not certify mandatory-cost budget infeasibility")
    return {"reason": "mandatory_turnover_exceeds_budget", "mandatory_turnover_cost": mandatory,
        "recorded_budget": recorded_budget, "independent_full_cost_upper_bound": upper_cost,
        "budget_shortfall": mandatory - recorded_budget, "normalized_witness_residual": residual}


def _verify_files(directory: Path, hashes: dict[str, str]) -> None:
    if not hashes:
        raise ValueError("An unavailable chain needs immutable original failure evidence")
    for name, digest in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unavailable chain evidence changed: {path}")


def verify_failed_prefix(*, episode: Path, context: dict[str, object],
                         premium_cap: float) -> dict[str, int | float]:
    """Verify saved prefix turnover/risk and absence of any post-failure execution."""
    elapsed, portfolio = int(context["elapsed_day"]), int(context["portfolio_index"])
    if elapsed < 0 or portfolio < 0:
        raise ValueError("A failed prefix needs nonnegative elapsed step and portfolio index")
    directories = sorted((episode / "steps").glob("step-*"))
    if [path.name for path in directories] != [f"step-{step:02d}" for step in range(elapsed + 1)]:
        raise ValueError("A failed prefix must stop at exactly its recorded first unavailable step")
    risk_path = directories[-1] / "risk_inputs.npz"
    with np.load(risk_path, allow_pickle=False) as saved:
        failed_inputs = {name: saved[name] for name in saved.files}
    if portfolio >= len(failed_inputs["previous_positions"]):
        raise ValueError("Unavailable portfolio index is outside the preserved inventory")
    if (directories[-1] / "trades.parquet").exists() or (directories[-1] / "completed_step.npz").exists():
        raise ValueError("An unavailable daily decision was incorrectly recorded as executed")
    if (episode / "summary.parquet").exists():
        raise ValueError("An unavailable batch must not publish fabricated terminal performance")
    checkpoint = episode / "steps" / "episode_checkpoint.npz"
    maximum_error = 0.
    if elapsed == 0:
        if checkpoint.exists() or np.any(failed_inputs["previous_positions"] != 0.):
            raise ValueError("A first-step failure cannot have executed prior inventory")
        return {"executed_prefix_steps": 0, "maximum_prefix_cost_error": 0.}
    with np.load(checkpoint, allow_pickle=False) as saved:
        if int(saved["next_step"]) != elapsed:
            raise ValueError("The checkpoint advanced beyond or stopped before the unavailable decision")
        np.testing.assert_array_equal(saved["holdings"][:, failed_inputs["active_columns"]],
            failed_inputs["previous_positions"])
        if saved["terminated"][portfolio]:
            raise ValueError("A previously knocked-out portfolio cannot require a fresh Greek decision")
        rows = pd.DataFrame(json.loads(str(saved["rows_json"])))
        count = len(failed_inputs["previous_positions"])
        if len(rows) != elapsed * count:
            raise ValueError("The failed prefix checkpoint lost executed trading rows")
        inventory = np.zeros_like(saved["holdings"])
        first_book = pd.read_parquet(directories[0] / "book.parquet")
        if inventory.shape != (count, 1 + len(first_book)):
            raise ValueError("The failed prefix does not preserve its frozen inception book")
        for step, directory in enumerate(directories[:-1]):
            daily = pd.read_parquet(directory / "trades.parquet")
            pd.testing.assert_frame_equal(daily.reset_index(drop=True),
                rows.loc[rows.elapsed_day == step].reset_index(drop=True), check_dtype=False)
            with np.load(directory / "completed_step.npz", allow_pickle=False) as positions:
                holdings = positions["positions"]
            np.testing.assert_array_equal(holdings, saved[f"positions_{step}"])
            with np.load(directory / "risk_inputs.npz", allow_pickle=False) as inputs:
                columns = inputs["active_columns"]
                np.testing.assert_array_equal(inputs["previous_positions"], inventory[:, columns])
                record, _ = replay_risk_step(trades=daily, positions=holdings,
                    inputs={name: inputs[name] for name in inputs.files}, premium_cap=premium_cap)
            inventory[:, columns] = holdings
            inventory[daily.knockout_exit_at_mid.to_numpy(dtype=bool)] = 0.
            settled = np.flatnonzero(first_book.bdte.to_numpy(dtype=float) <= step + 1. + 1e-6)
            inventory[:, 1 + settled] = 0.
            maximum_error = max(maximum_error, record["cost_error"])
        np.testing.assert_array_equal(inventory, saved["holdings"])
    return {"executed_prefix_steps": elapsed, "maximum_prefix_cost_error": maximum_error}


def audit_isolated_frontier(*, input_directory: Path, output_directory: Path) -> dict[str, object]:
    """Audit a fully processed origin/family, including explicit unavailable cells.

    Successful batches must retain their existing independent cash/risk
    audits. This adds raw infeasibility and prefix checks; it is not a
    global neural-optimality or a model-calibration certificate.
    """
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Frontier audit outputs must remain outside Git working trees")
    configuration = json.loads((input_directory / "config.json").read_text())
    completion = json.loads((input_directory / "summary.json").read_text())
    if completion["status"] not in ("complete", "processing_complete_with_unavailable"):
        raise ValueError("An unfinished frontier cannot enter the processed-frontier audit")
    cap = float(configuration["identity"]["optimized_premium_cap"])
    shared = json.loads((input_directory / "shared_input_hashes.json").read_text())
    available = pd.read_parquet(input_directory / "results.parquet")
    unavailable = pd.read_parquet(input_directory / "unavailable.parquet")
    identifiers = ["claim", "method", "side", "objective", "risk_aversion", "greek_budget"]
    expected = {(label, *[rule[name] for name in identifiers[1:]])
        for label in configuration["claim_labels"] for rule in configuration["rules"]}
    observed = pd.concat([frame[identifiers] for frame in (available, unavailable) if len(frame)], ignore_index=True)
    if (len(observed) != len(expected) or set(observed.itertuples(index=False, name=None)) != expected
            or completion["requested_portfolios"] != len(expected)
            or completion["completed_portfolios"] != len(available)
            or completion["unavailable_portfolios"] != len(unavailable)
            or len(available) and not np.isfinite(available.costed_pnl).all()
            or len(unavailable) and not unavailable.costed_pnl.isna().all()):
        raise ValueError("Frontier availability/performance does not account for every requested chain exactly once")
    records, completed_frames, evidence = [], [], {}
    for path in sorted((input_directory / "batches").glob("*/*/*/completed.json")):
        manifest = json.loads(path.read_text())
        _verify_files(path.parent, manifest["artifacts_sha256"])
        if json.loads((path.parent / "input_hashes.json").read_text()) != shared:
            raise ValueError("Completed batches do not share the declared market/model inputs")
        batch = path.parent.parent.parent
        cash = json.loads((batch / "independent_cash_audit" / "summary.json").read_text())
        risk = json.loads((batch / "independent_risk_audit" / "summary.json").read_text())
        if (cash["maximum_cash_error"] > 1e-10 or risk["maximum_replay_error"] > 1e-8
                or not np.isfinite([cash["maximum_cash_error"], risk["maximum_replay_error"]]).all()):
            raise RuntimeError("Completed Greek batches have not passed their independent accounting/risk checks")
        completed_frames.append(pd.read_parquet(path.parent / "summary.parquet"))
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if completed_frames:
        combined = pd.concat(completed_frames, ignore_index=True).sort_values(identifiers).reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, available.sort_values(identifiers).reset_index(drop=True))
    elif len(available):
        raise ValueError("Available performance has no completed batch evidence")
    for row in unavailable.itertuples():
        batch = Path(row.evidence_directory)
        if not batch.resolve().is_relative_to((input_directory / "batches").resolve()):
            raise ValueError("Unavailable performance references evidence outside its own frontier")
        outcome_path = batch / "unavailable.json"
        outcome = json.loads(outcome_path.read_text())
        episode = batch / configuration["origin"] / configuration["family"]
        _verify_files(episode, outcome["artifacts_sha256"])
        context = json.loads((episode / "unavailable_context.json").read_text())
        if (context != outcome["context"] or context["input_hashes"] != shared
                or context["origin"] != configuration["origin"] or context["family"] != configuration["family"]
                or any(context[name] != getattr(row, name) for name in identifiers)
                or context["date"] != row.failure_date):
            raise ValueError("Unavailable rows do not match their immutable episode and input identity")
        prefix = Path(context["failure_evidence"])
        if not prefix.resolve().is_relative_to(episode.resolve()):
            raise ValueError("The failure snapshot is outside its own episode")
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            geometry = {name: saved[name] for name in saved.files}
        with np.load(prefix.parent / "risk_inputs.npz", allow_pickle=False) as saved:
            if float(saved["optimized_premium_cap"]) != cap:
                raise ValueError("Unavailable geometry uses a different premium limit")
            for name in ("normalized_premium_prices", "costs", "active_columns"):
                np.testing.assert_array_equal(geometry[name], saved[name])
            np.testing.assert_array_equal(geometry["previous_contract_positions"],
                saved["previous_positions"][context["portfolio_index"]])
            np.testing.assert_array_equal(geometry["instrument_exposures"], saved["greek_instrument_exposures"])
            labels = list(saved["claim_labels"])
            if labels.count(context["claim"]) != 1:
                raise ValueError("An unavailable target is absent or duplicated in the captured risk geometry")
            np.testing.assert_array_equal(geometry["target_exposures"],
                saved["greek_claim_exposures"][labels.index(context["claim"])])
        record = verify_unavailable_geometry(context=context, evidence=geometry, premium_cap=cap)
        history = verify_failed_prefix(episode=episode, context=context, premium_cap=cap)
        records.append({"claim": row.claim, "side": row.side, "greek_budget": row.greek_budget,
            "failure_date": row.failure_date, **record, **history})
        evidence[str(outcome_path)] = hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output_directory / "unavailable_replay.parquet", index=False)
    summary = {"origin": configuration["origin"], "family": configuration["family"],
        "requested_portfolios": len(expected), "completed_portfolios": len(available),
        "verified_unavailable_portfolios": len(frame), "complete_feasible_frontier": not len(unavailable),
        "reason_counts": frame.reason.value_counts().to_dict() if len(frame) else {},
        "configuration_sha256": hashlib.sha256((input_directory / "config.json").read_bytes()).hexdigest(),
        "completion_sha256": hashlib.sha256((input_directory / "summary.json").read_bytes()).hexdigest(),
        "evidence_sha256": evidence, "scope": "recorded geometry, accounting and infeasibility; not model calibration or global policy optimality",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    print(json.dumps(audit_isolated_frontier(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
