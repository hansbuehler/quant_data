"""Run the paper's restricted unconditional and conditional CRC drift tests."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from cdxcore.version import version

from quantdata.dysanos.no_arbitrage_audit import (
    RestrictedPopulation,
    crossfit_conditional_audit,
    unconditional_coordinate_audit,
)
from quantdata.torch.ffw import FeedForwardConfig


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CrcAuditConfig:
    """Conditional-policy geometry matched to the main paper."""

    path_count: int = 100_000
    policy_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=32,
    )
    policy_epochs: int = 200
    policy_batch_size: int = 256
    policy_learning_rate: float = 1.0e-3
    lambda_second_moment: float = 1.0
    seed: int = 84_113
    checkpoint_schema: str = "gambara-teichmann-crc-conditional-policy-v1"

    def __post_init__(self) -> None:
        if self.path_count < 4 or self.path_count % 4 != 0:
            raise ValueError("path count must be divisible by four")
        if self.policy_epochs <= 0 or self.policy_batch_size <= 0:
            raise ValueError("policy epochs and batch size must be positive")
        if self.policy_learning_rate <= 0.0 or self.lambda_second_moment < 0.0:
            raise ValueError("policy learning rate/penalty are invalid")


def run_q_audit(
    *,
    population_path: Path,
    output_directory: Path,
    config: CrcAuditConfig,
    device: str,
) -> dict[str, object]:
    """Evaluate all 570 coordinates and two out-of-fold strategy tests."""

    with np.load(population_path, allow_pickle=False) as archive:
        population = RestrictedPopulation(
            features=np.asarray(archive["features"], dtype=np.float32),
            returns=np.asarray(archive["returns"], dtype=np.float64),
            active_mask=np.asarray(archive["active_mask"], dtype=bool),
            pair_ids=np.asarray(archive["pair_ids"], dtype=np.int64),
        )
    if population.features.shape[0] != config.path_count:
        raise RuntimeError(
            f"population has {population.features.shape[0]} paths, expected {config.path_count}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    unconditional = unconditional_coordinate_audit(
        population.returns,
        active_mask=population.active_mask,
    )
    conditional = {}
    for objective in ("squared", "entropy"):
        conditional[objective] = crossfit_conditional_audit(
            population,
            objective=objective,
            config=config,
            checkpoint_directory=output_directory / "conditional-checkpoints",
            device=device,
        )
    result: dict[str, object] = {
        "schema": "gambara-teichmann-crc-q-audit-v1",
        "population": str(population_path.resolve()),
        "config": asdict(config),
        "unconditional": asdict(unconditional),
        "conditional": {
            objective: asdict(value) for objective, value in conditional.items()
        },
    }
    temporary = output_directory / ".q-audit.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_directory / "q-audit.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--path-count", type=int, default=100_000)
    arguments = parser.parse_args()
    result = run_q_audit(
        population_path=arguments.population,
        output_directory=arguments.output_directory,
        config=CrcAuditConfig(path_count=arguments.path_count),
        device=arguments.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
