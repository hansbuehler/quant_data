"""Run the reduced deterministic-codebook Tangent Levy v2 Q audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dynamics import DynamicsFit
from .run_v2 import DEFAULT_OUTPUT, _load_day
from .v2_audit import (
    RestrictedPopulation,
    crossfit_conditional_audit,
    unconditional_coordinate_audit,
)
from .v2_config import TangentLevyV2AuditConfig
from .v2_population import simulate_restricted_population

INITIAL_DATE = "2022-06-01"


def _zero_factor_dynamics(initial: np.ndarray) -> DynamicsFit:
    tau = np.arange(71, dtype=np.float64) / 252.0
    jump_x = np.linspace(-0.75, 0.75, 301)
    return DynamicsFit(
        tau=tau,
        jump_x=jump_x,
        beta=np.zeros((0, len(tau), len(jump_x))),
        eigenvalues_one_session=np.zeros(0),
        explained_variance_ratio=np.zeros(0),
        mean_change=np.zeros_like(initial),
        observations=0,
        maximum_constraint_residual=0.0,
        covariance_shrinkage=0.0,
        dt=1.0 / 252.0,
        active_mask=np.ones_like(initial, dtype=bool),
    )


def _save_json(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate(output: Path) -> None:
    config = TangentLevyV2AuditConfig()
    _, initial, reconstruction = _load_day(
        output / "days" / f"{INITIAL_DATE}.npz"
    )
    population, universe, diagnostics = simulate_restricted_population(
        initial,
        _zero_factor_dynamics(initial),
        config=config,
        path_count=config.path_count,
        batch_size=2_000,
    )
    population_directory = output / "q-reduced-population"
    population_directory.mkdir(parents=True, exist_ok=True)
    np.save(population_directory / "features.npy", population.features)
    np.save(population_directory / "returns.npy", population.returns)
    np.save(
        population_directory / "returns-training-f32.npy",
        population.returns.astype(np.float32),
    )
    np.save(population_directory / "active-mask.npy", population.active_mask)
    np.save(population_directory / "pair-ids.npy", population.pair_ids)
    unconditional = unconditional_coordinate_audit(
        population.returns, active_mask=population.active_mask
    )
    _save_json(
        {
            "schema": "tangent-levy-v2-reduced-q-audit",
            "model": "deterministic Musiela codebook, native short-end TL spot",
            "initial_date": INITIAL_DATE,
            "paths": config.path_count,
            "pairs": config.path_count // 2,
            "reconstruction": asdict(reconstruction),
            "population_diagnostics": asdict(diagnostics),
            "unconditional": asdict(unconditional),
            "instrument_labels": universe.labels,
        },
        output / "q-reduced-audit.json",
    )


def conditional(output: Path, objective: str, device: str) -> None:
    config = TangentLevyV2AuditConfig()
    directory = output / "q-reduced-population"
    population = RestrictedPopulation(
        features=np.load(directory / "features.npy", mmap_mode="r"),
        returns=np.load(directory / "returns-training-f32.npy", mmap_mode="r"),
        active_mask=np.load(directory / "active-mask.npy"),
        pair_ids=np.load(directory / "pair-ids.npy"),
    )
    result = crossfit_conditional_audit(
        population,
        objective=objective,
        config=config,
        checkpoint_directory=output / "q-reduced-conditional-checkpoints",
        device=device,
    )
    _save_json(asdict(result), output / f"q-reduced-{objective}.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("population", "conditional"))
    parser.add_argument("--objective", choices=("squared", "entropy"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.stage == "population":
        generate(arguments.output)
    else:
        if arguments.objective is None:
            parser.error("conditional stage requires --objective")
        conditional(arguments.output, arguments.objective, arguments.device)


if __name__ == "__main__":
    main()
