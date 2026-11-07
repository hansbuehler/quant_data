"""Dated CPU validation of final-horizon market paths, without fitting hedge policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import CashCurve, MarketStore, select_book
from quantdata.hedge2.claims import Claim, make_claims
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive, surface_digitals
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_market import (
    TerminalMarketPaths,
    TerminalSimulationConfig,
    generate_terminal_paths,
)


def claim_groups(claims: tuple[Claim, ...]) -> dict[str, tuple[Claim, ...]]:
    """Separate each prescribed frozen book; only the exotics share 3D/5D contracts."""
    return {**{f"digital-{tenor}D": tuple(claim for claim in claims
                if claim.kind == "digital" and claim.terminal_day == tenor) for tenor in (1, 3, 5)},
            "exotic-3D-5D": tuple(claim for claim in claims if claim.kind != "digital")}


def audit_terminal_cash(*, paths: TerminalMarketPaths, book: pd.DataFrame,
                        curve: CashCurve, claims: tuple[Claim, ...],
                        current_marks: np.ndarray) -> dict[str, float]:
    """Independently replay fixed-contract settlement and target cashflows, without demeaning."""
    count, horizon, _ = paths.instrument_gains.shape
    maximum_error = 0.
    for index, row in enumerate(book.itertuples()):
        end = min(horizon, int(round(row.bdte)))
        final_mark = paths.instrument_prices[:, end, index + 1]
        expected = curve.at(float(end))[2] * final_mark - row.observed_mid / curve.spot
        actual = paths.raw_instrument_gains[:, :, index + 1].sum(axis=1)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    spot_error = 0.
    for step in range(horizon):
        reference_ratio = curve.at(float(step + 1))[1] / curve.at(float(step))[1]
        expected = curve.at(float(step + 1))[2] * (paths.spots[:, step + 1]
            - paths.spots[:, step] * reference_ratio) / curve.spot
        spot_error = max(spot_error, float(np.max(np.abs(paths.raw_instrument_gains[:, step, 0] - expected))))
    payoff_error = 0.
    for index, claim in enumerate(claims):
        expected = []
        for path in range(count):
            observed = claim if claim.kind == "digital" else claim.observe(
                elapsed_day=claim.observation_day, spot=float(paths.spots[path, claim.observation_day]))
            expected.append(observed.payoff(float(paths.spots[path, -1])))
        payoff_error = max(payoff_error, float(np.max(np.abs(paths.claim_payoffs[:, index] - expected))))
    expected_gains = curve.at(float(horizon))[2] * paths.claim_payoffs - current_marks
    claim_error = float(np.max(np.abs(expected_gains - paths.claim_gains)))
    if max(maximum_error, spot_error, payoff_error, claim_error) > 1e-10:
        raise ValueError("Independent full-horizon cash replay failed")
    return {"maximum_option_cash_error": maximum_error, "maximum_spot_cash_error": spot_error,
            "maximum_payoff_error": payoff_error, "maximum_claim_cash_error": claim_error}


def run_pilot(*, source_configuration: Path, output_directory: Path, day: date,
              paths: int = 16, integration_parent_batch: int = 16, seed: int = 1,
              groups: tuple[str, ...] | None = None) -> dict[str, object]:
    """Use the saved current market, native state and rolling AR fit for every target family."""
    output_directory.mkdir(parents=True, exist_ok=True)
    simulation_config = TerminalSimulationConfig(paths=paths, integration_parent_batch=integration_parent_batch)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        source_configuration_data = source_configuration.read_bytes()
        sources = json.loads(source_configuration_data)
        store = MarketStore(market_directory=Path(sources["market_directory"]),
            licensed_fit_directory=Path(sources["licensed_fit_directory"]),
            raw_quote_directory=Path(sources["raw_quote_directory"]))
        market = store.load(day)
        source, source_metadata = SourceArchive(root=Path(sources["source_root"])).load(day)
        state_directory, dynamics_directory = Path(sources["daily_directory"]), Path(sources["dynamics_directory"])
        state = DailyState.load(state_directory, day)
        dynamics = load_dynamics(dynamics_directory / "days" / f"{day.isoformat()}.npz")
        implementation = sorted(Path(__file__).parent.glob("*.py"))
        inputs = {"daily_config": state_directory / "config.json",
            "daily_state": state_directory / "days" / f"{day.isoformat()}.npz",
            "dynamics": dynamics_directory / "days" / f"{day.isoformat()}.npz",
            "licensed_quotes": Path(sources["market_directory"]) / "days" / f"{day.isoformat()}.parquet",
            "licensed_fit": Path(sources["licensed_fit_directory"]) / "SPX" / f"{day.isoformat()}.pdq",
            "licensed_raw_quotes": Path(sources["raw_quote_directory"]) / "SPX" / f"{day.isoformat()}.pdq"}
        configuration = {"source_configuration": str(source_configuration),
            "source_configuration_sha256": hashlib.sha256(source_configuration_data).hexdigest(),
            "source_model": source_metadata, "day": day.isoformat(), "seed": seed,
            "simulation": asdict(simulation_config), "simulation_version": TerminalSimulationConfig.version.unique_id48,
            "diagnostic_only": True, "OOS_hedging_results": False,
            "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
            "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}}
        if groups is not None:
            available = ("digital-1D", "digital-3D", "digital-5D", "exotic-3D-5D")
            if not groups or len(set(groups)) != len(groups) or any(group not in available for group in groups):
                raise ValueError("Choose distinct prescribed target/book groups")
            configuration["groups"] = list(groups)
        config_path = output_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError("This terminal pilot directory belongs to a different model, source, or implementation")
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as saved:
                for path in implementation:
                    saved.write(path, arcname=path.name)
        claims = make_claims(initial_spot=market.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
        vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
        baseline = state.decode().pricer
        rows, started = [], time.monotonic()
        for label, group in claim_groups(claims).items():
            if groups is not None and label not in groups:
                continue
            directory = output_directory / label
            directory.mkdir(parents=True, exist_ok=True)
            completion = directory / "completed.json"
            if completion.exists():
                marker = json.loads(completion.read_text())
                for name, digest in marker["artifacts_sha256"].items():
                    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"A completed terminal pilot artifact changed: {directory / name}")
                rows.append(json.loads((directory / "diagnostics.json").read_text()))
                continue
            coterminal = None if label == "digital-1D" else ((3, 5) if label == "exotic-3D-5D" else (group[0].terminal_day,))
            book = select_book(market, atm_vols=vols, co_terminal_business_dtes=coterminal,
                option_side="itm", include_co_terminal_forwards=True)
            marks = []
            for claim in group:
                if claim.kind == "digital":
                    forward, _, discount = market.curve.at(float(claim.terminal_day))
                    mark = discount * surface_digitals(source, expiry_y=claim.terminal_day / DAYS_PER_YEAR,
                        strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0]
                else:
                    mark = claim.price(baseline, market.curve, elapsed_day=0)
                marks.append(mark)
            marks = np.array(marks)
            print(f"terminal pilot group={label} paths={paths} instruments={1+len(book)} claims={len(group)}", flush=True)
            group_started = time.monotonic()
            result = generate_terminal_paths(state=state, dynamics=dynamics, curve=market.curve,
                book=book, claims=group, initial_spot=market.curve.spot, current_claim_marks=marks,
                config=simulation_config, seed=seed + group[0].terminal_day,
                progress=lambda message: print(message, flush=True))
            for index, claim in enumerate(group):
                result.for_claim(claim, index)
            audited = audit_terminal_cash(paths=result, book=book, curve=market.curve, claims=group, current_marks=marks)
            book.to_parquet(directory / "book.parquet", index=False)
            np.savez_compressed(directory / "paths.npz", **asdict(result))
            (directory / "claims.json").write_text(json.dumps({"claims": [asdict(claim) for claim in group],
                "initial_marks": marks.tolist(), "digital_mark": "saved smoothness-0.1 SANOS derivative",
                "exotic_mark": "inception decoded DLV joint-law price; no observed exotic quote"}, indent=2) + "\n")
            diagnostic = {"group": label, "paths": len(result.weights), "instruments": 1 + len(book), "claims": len(group),
                "horizon": result.instrument_gains.shape[1], "elapsed_seconds": time.monotonic()-group_started,
                "integrated_parents": result.integrated_parent_counts.tolist(),
                "maximum_conditional_mean_residual": float(result.conditional_mean_residual.max()),
                "zero_live_model_prices": result.zero_live_model_prices.tolist(),
                "selected_lower_guard_count": int(result.lower_clips.sum()),
                "selected_upper_guard_count": int(result.upper_clips.sum()), **audited}
            (directory / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
            names = ("book.parquet", "paths.npz", "claims.json", "diagnostics.json")
            temporary = completion.with_suffix(".partial.json")
            temporary.write_text(json.dumps({"artifacts_sha256": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}, indent=2) + "\n")
            temporary.replace(completion)
            rows.append(diagnostic)
            print(f"terminal pilot completed group={label} elapsed_s={diagnostic['elapsed_seconds']:.1f}", flush=True)
        pd.DataFrame(rows).to_parquet(output_directory / "diagnostics.parquet", index=False)
        summary = {"groups": rows, "elapsed_seconds": time.monotonic()-started,
            "diagnostic_only": True, "OOS_hedging_results": False}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--paths", type=int, default=16)
    parser.add_argument("--integration-parent-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", nargs="+", choices=("digital-1D", "digital-3D", "digital-5D", "exotic-3D-5D"))
    arguments = vars(parser.parse_args())
    if arguments["groups"] is not None:
        arguments["groups"] = tuple(arguments["groups"])
    print(json.dumps(run_pilot(**arguments), indent=2), flush=True)


if __name__ == "__main__":
    main()
