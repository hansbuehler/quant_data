"""Independent raw-position Greek, covariance and utility replay of dated episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedge2.audit_parametric import integrated_normal_power


def replay_risk_step(*, trades: pd.DataFrame, positions: np.ndarray,
                     inputs: dict[str, np.ndarray], premium_cap: float | None
                     ) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Recompute risk in original contract units, not the optimizer's premium map.

    These are local Gaussian/Greek checks, not terminal-policy optimality.
    Actual turnover includes carried positions outside today's premium box.
    Knocked-out rows have no fresh risk optimization but retain cash checks.
    """
    costs, previous = inputs["costs"], inputs["previous_positions"]
    if (positions.ndim != 2 or not len(positions) or len(trades) != len(positions)
            or costs.shape != (positions.shape[1],) or previous.shape != positions.shape
            or not all(np.isfinite(value).all() for value in (positions, costs, previous))
            or np.any(costs < 0.) or not trades["side"].isin((-1, 1)).all()
            or not trades["method"].isin(("DLV Greek", "DLV parametric")).all()):
        raise ValueError("Episode risk replay needs aligned finite optimized positions and nonnegative fees")
    actual_cost = np.abs(positions - previous) @ costs
    recorded_cost = trades["entry_cost"].to_numpy(dtype=float)
    if not np.isfinite(recorded_cost).all() or np.any(recorded_cost < 0.):
        raise ValueError("Saved entry costs must be finite and nonnegative")
    cost_error = float(np.max(np.abs(actual_cost - recorded_cost)))
    if not np.isfinite(actual_cost).all() or not np.isfinite(cost_error) or cost_error > 1e-8:
        raise RuntimeError("Saved episode entry cost differs from raw carried-inventory turnover")
    if premium_cap is not None:
        prices = inputs["normalized_premium_prices"]
        if (not np.isfinite(premium_cap) or premium_cap <= 0. or prices.shape != costs.shape
                or not np.isfinite(prices).all() or np.any(prices <= 0.)):
            raise ValueError("Premium audit requires current undiscounted prices and a positive cap")
        if np.max(np.abs(positions * prices)) > premium_cap * (1. + 1e-8):
            raise RuntimeError("Raw episode positions exceed the current-premium limit")
    active = trades["claim_active_before_step"].to_numpy(dtype=bool)
    labels = list(inputs["claim_labels"])
    if len(set(labels)) != len(labels):
        raise ValueError("Saved claim labels must be unique")
    lookup = {label: index for index, label in enumerate(labels)}
    if not trades["claim"].isin(lookup).all():
        raise ValueError("A saved target is absent from the raw risk geometry")
    target_indices = np.array([lookup[label] for label in trades["claim"]])
    record = {"portfolios": len(trades), "active_portfolios": int(active.sum()),
        "cost_error": cost_error, "std_error": 0., "mean_error": 0.,
        "mv_objective_error": 0., "maximum_normalized_full_greek_residual": 0.}
    power_candidates = []
    for method in ("DLV parametric", "DLV Greek"):
        selected = active & trades["method"].eq(method).to_numpy()
        if not selected.any():
            continue
        prefix = "greek_" if method == "DLV Greek" else ""
        exposures, targets = inputs[f"{prefix}instrument_exposures"], inputs[f"{prefix}claim_exposures"]
        if (exposures.ndim != 2 or exposures.shape[0] != positions.shape[1]
                or targets.shape != (len(labels), exposures.shape[1])
                or not np.isfinite(exposures).all() or not np.isfinite(targets).all()):
            raise ValueError("Saved raw Greek exposures and targets must align and be finite")
        target = targets[target_indices]
        residual = positions @ exposures + trades["side"].to_numpy()[:, None] * target
        if not np.isfinite(residual).all():
            raise RuntimeError("Raw risk exposures overflowed despite finite saved inputs")
        if method == "DLV Greek":
            fractions = trades.loc[selected, "greek_budget"].to_numpy(dtype=float)
            full_cost = trades.loc[selected, "greek_full_entry_cost"].to_numpy(dtype=float)
            if (not np.isfinite(fractions).all() or np.any((fractions < 0.) | (fractions > 1.))
                    or not np.isfinite(full_cost).all() or np.any(full_cost < 0.)):
                raise ValueError("Greek budget fractions and full costs must be finite and valid")
            allowance = np.maximum(1e-6 * full_cost, 32. * np.nextafter(0., 1.))
            if np.any(actual_cost[selected] > fractions * full_cost + allowance):
                raise RuntimeError("Greek raw-position turnover exceeds its stated cost budget")
            exact = selected & trades["greek_budget"].eq(1.).to_numpy()
            scale = np.maximum(np.max(np.abs(exposures), axis=0)[None, :], np.abs(target))
            scale = np.where(scale > 0., scale, 1.)
            error = float(np.max(np.abs(residual[exact]) / scale[exact], initial=0.))
            record["maximum_normalized_full_greek_residual"] = error
            if error > 1e-6:
                raise RuntimeError("Full-budget Greek hedge does not match its saved price sensitivities")
            continue
        covariance = inputs["covariance"]
        if covariance.shape != (exposures.shape[1], exposures.shape[1]) or not np.isfinite(covariance).all():
            raise ValueError("Saved factor covariance must be finite and match the exposure axes")
        tolerance = 1e-12 * max(1., float(np.max(np.abs(covariance))))
        if (np.max(np.abs(covariance - covariance.T)) > tolerance
                or np.linalg.eigvalsh(covariance).min() < -tolerance):
            raise ValueError("Saved factor covariance must be symmetric and positive semidefinite")
        variance = np.einsum("ij,jk,ik->i", residual, covariance, residual)
        if not np.isfinite(variance[selected]).all():
            raise RuntimeError("Raw-contract covariance variance is non-finite")
        std = np.sqrt(np.maximum(variance, 0.))
        saved_std = trades["training_std"].to_numpy(dtype=float)
        saved_mean = trades["training_mean_net"].to_numpy(dtype=float)
        saved_risk = trades["training_objective"].to_numpy(dtype=float)
        if not all(np.isfinite(value[selected]).all() for value in (saved_std, saved_mean, saved_risk)):
            raise ValueError("Saved active parametric risk statistics must be finite")
        record["std_error"] = float(np.max(np.abs(std[selected] - saved_std[selected])))
        record["mean_error"] = float(np.max(np.abs(saved_mean[selected] + actual_cost[selected])))
        risk_aversion = trades["risk_aversion"].to_numpy(dtype=float)
        mv = selected & trades["objective"].eq("mean_variance").to_numpy()
        power = selected & trades["objective"].eq("power_exponential").to_numpy()
        if (np.any(selected & ~(mv | power)) or np.any(np.isnan(risk_aversion[selected]))
                or np.any(risk_aversion[selected] <= 0.) or not np.isfinite(risk_aversion[power]).all()):
            raise ValueError("Parametric replay requires supported positive risk aversions and objectives")
        finite = np.isfinite(risk_aversion)
        expected = .5 * np.where(finite, risk_aversion, 1.) * variance + np.where(finite, actual_cost, 0.)
        if not np.isfinite(expected[mv]).all() or not np.isfinite(std[selected]).all():
            raise RuntimeError("Raw-contract objective or standard deviation is non-finite")
        record["mv_objective_error"] = float(np.max(np.abs(expected[mv] - saved_risk[mv]), initial=0.))
        if max(record[name] for name in ("std_error", "mean_error", "mv_objective_error")) > 1e-8:
            raise RuntimeError("Raw-contract covariance, net mean or MV objective differs from saved optimization")
        for index in np.flatnonzero(power):
            power_candidates.append({"row": int(index), "std": float(std[index]),
                "risk_aversion": float(risk_aversion[index]),
                "saved_centered_risk": float(saved_risk[index] - actual_cost[index])})
    return record, power_candidates


def bound_risk_steps(episode: Path, hashes: dict[str, str]
                     ) -> tuple[list[tuple[Path, pd.DataFrame, np.ndarray, dict[str, np.ndarray]]], dict[str, str]]:
    """Bind step files to originally sealed positions/trades; capture unsealed geometry separately."""
    for name in ("trades.parquet", "positions.npz", "initial_book.parquet"):
        if name not in hashes:
            raise ValueError(f"The original completion must seal the root accounting artifact: {name}")
    trades = pd.read_parquet(episode / "trades.parquet")
    book = pd.read_parquet(episode / "initial_book.parquet")
    elapsed = sorted(trades.elapsed_day.unique())
    if elapsed != list(range(len(elapsed))):
        raise ValueError("Saved elapsed steps must form a complete sequence from inception")
    expected_names = [f"step-{step:02d}" for step in elapsed]
    directories = sorted((episode / "steps").glob("step-*"))
    if [path.name for path in directories] != expected_names:
        raise ValueError("Risk step directories do not match the complete sealed elapsed-step set")
    records, captured = [], {}
    inventory, identities = None, None
    active_claims = None
    with np.load(episode / "positions.npz", allow_pickle=False) as root_positions:
        for step, directory in zip(elapsed, directories, strict=True):
            daily = trades.loc[trades.elapsed_day.eq(step)].reset_index(drop=True)
            saved_trades = pd.read_parquet(directory / "trades.parquet")
            pd.testing.assert_frame_equal(saved_trades, daily)
            with np.load(directory / "completed_step.npz", allow_pickle=False) as saved:
                positions = saved["positions"]
            np.testing.assert_array_equal(positions, root_positions[f"step_{step}"])
            with np.load(directory / "risk_inputs.npz", allow_pickle=False) as saved:
                inputs = {name: saved[name] for name in saved.files}
            columns = root_positions[f"active_{step}"]
            np.testing.assert_array_equal(columns, inputs["active_columns"])
            if (columns.ndim != 1 or len(columns) == 0 or not np.issubdtype(columns.dtype, np.integer)
                    or len(columns) != positions.shape[1] or columns[0] != 0
                    or np.any(np.diff(columns) <= 0) or columns[-1] > len(book)):
                raise ValueError("Sealed active columns do not identify the frozen inception book")
            if inventory is None:
                inventory = np.zeros((len(daily), 1 + len(book)))
                active_claims = np.ones(len(daily), dtype=bool)
                identities = daily[["claim", "method", "objective", "risk_aversion", "side", "greek_budget"]]
            if len(daily) != len(inventory):
                raise ValueError("The number of hedge portfolios changed inside the frozen episode")
            pd.testing.assert_frame_equal(daily[identities.columns], identities)
            np.testing.assert_array_equal(inputs["previous_positions"], inventory[:, columns])
            np.testing.assert_array_equal(daily.claim_active_before_step, active_claims)
            if set(inputs["claim_labels"]) != set(trades.claim):
                raise ValueError("Captured risk labels do not match the sealed target universe")
            inventory[:, columns] = positions
            active_claims &= ~daily.knockout_exit_at_mid.to_numpy(dtype=bool)
            inventory[~active_claims] = 0.
            for name in ("trades.parquet", "completed_step.npz", "risk_inputs.npz"):
                path = directory / name
                relative = str(path.relative_to(episode))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if relative in hashes and hashes[relative] != digest:
                    raise ValueError(f"Originally sealed risk-step evidence changed: {path}")
                if relative not in hashes:
                    captured[str(path)] = digest
            records.append((directory, daily, positions, inputs))
    return records, captured


def audit_episode_risk(*, input_directory: Path, output_directory: Path,
                       completed_only: bool = False, maximum_gaussian_integrals: int = 100) -> dict[str, object]:
    """Audit immutable completed episodes, explicitly distinguishing a partial capture."""
    if (not isinstance(completed_only, bool) or not isinstance(maximum_gaussian_integrals, int)
            or isinstance(maximum_gaussian_integrals, bool) or maximum_gaussian_integrals < 1):
        raise ValueError("Specify a boolean completion mode and a positive Gaussian integration count")
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
            for parent in (resolved, *resolved.parents)):
        raise ValueError("Episode risk audit outputs must be outside Git working trees")
    configuration_path = input_directory / "config.json"
    configuration = json.loads(configuration_path.read_text())
    completion_path = input_directory / "summary.json"
    completion = (json.loads(completion_path.read_text()) if completion_path.exists()
        else {"status": "running"})
    if not completed_only and completion["status"] != "complete":
        raise ValueError("An unfinished population requires an explicitly partial risk audit")
    manifests = sorted(input_directory.glob("*/*/completed.json"))
    if not manifests:
        raise ValueError("No completed episode manifests are available")
    if not completed_only and len(manifests) != completion["episodes"]:
        raise ValueError("The population completion count disagrees with its episode manifests")
    records, candidates, evidence, captured = [], [], {}, {}
    started = time.monotonic()
    for episode_index, manifest_path in enumerate(manifests):
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        evidence[str(manifest_path)] = hashlib.sha256(manifest_bytes).hexdigest()
        hashes = manifest["artifacts_sha256"]
        if not hashes:
            raise ValueError("A completed episode must retain hashed accounting and risk evidence")
        for name, digest in hashes.items():
            if hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Completed episode evidence changed: {manifest_path.parent / name}")
        steps, supplements = bound_risk_steps(manifest_path.parent, hashes)
        captured.update(supplements)
        for directory, trades, positions, inputs in steps:
            record, power = replay_risk_step(trades=trades, positions=positions, inputs=inputs,
                premium_cap=configuration.get("optimized_premium_cap"))
            labels = {"origin": manifest_path.parent.parent.name,
                "family": manifest_path.parent.name, "step": directory.name}
            records.append({**labels, **record})
            candidates.extend({**labels, **row} for row in power)
        if (episode_index + 1) % 50 == 0:
            print(f"episode risk replay complete={episode_index+1}/{len(manifests)} "
                f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    if not records:
        raise ValueError("Completed episodes contain no saved raw-risk steps")
    integrals = pd.DataFrame(candidates)
    if len(integrals):
        integrals = integrals.sort_values(["risk_aversion", "std"])
        selected = np.unique(np.linspace(0, len(integrals)-1,
            min(maximum_gaussian_integrals, len(integrals))).astype(int))
        integrals = integrals.iloc[selected].copy()
        integrals["integrated_risk"] = [integrated_normal_power(row.std, risk_aversion=row.risk_aversion)[0]
            for row in integrals.itertuples()]
        integrals["absolute_risk_error"] = np.abs(integrals.integrated_risk - integrals.saved_centered_risk)
        if (not np.isfinite(integrals[["integrated_risk", "absolute_risk_error"]].to_numpy()).all()
                or integrals.absolute_risk_error.max() > 1e-7):
            raise RuntimeError("Saved Gaussian power objective differs from independent utility integration")
    frame = pd.DataFrame(records)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_directory / "raw_risk_replay.parquet", index=False)
    integrals.to_parquet(output_directory / "gaussian_integrals.parquet", index=False)
    summary = {"input_directory": str(input_directory), "completed_episodes": len(manifests),
        "configuration_sha256": hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
        "partial_run_audit": completed_only, "steps": len(frame), "daily_portfolios": int(frame.portfolios.sum()),
        "maximum_replay_error": float(frame[["cost_error", "std_error", "mean_error", "mv_objective_error"]].max().max()),
        "maximum_normalized_full_greek_residual": float(frame.maximum_normalized_full_greek_residual.max()),
        "gaussian_integrals": len(integrals), "maximum_integral_error": (
            float(integrals.absolute_risk_error.max()) if len(integrals) else None),
        "completion_manifest_sha256": evidence,
        "post_completion_capture_sha256": captured,
        "provenance_scope": "root accounting originally sealed; unsealed risk geometry explicitly captured at audit time",
        "risk_replay_conditional_on_captured_geometry": True,
        "greek_scope": "exposure and saved-budget feasibility, not minimum-cost or frontier optimality",
        "power_scope": "raw covariance/std for all rows; independent utility integration on the stated sample only",
        "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("audit_episode_risk.py", "audit_parametric.py")}}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--maximum-gaussian-integrals", type=int, default=100)
    print(json.dumps(audit_episode_risk(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
