"""Report pre-2025 and forward-2025 observed spot/ATM-IV diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.v11.factorial_statistics import historical_observation_panels
from quantdata.dysanos.v11.native_atm_history import load_atm_history


def run(
    *,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output: Path,
    bootstrap_replicates: int,
    bootstrap_block_length: int,
    seed: int,
) -> dict[str, object]:
    """Compute observed panels without using model fits or the evaluation split for tuning."""
    dates, business_dtes, atm_iv, spot_returns = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    report = historical_observation_panels(
        dates,
        business_dtes,
        atm_iv,
        spot_returns,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_block_length=bootstrap_block_length,
        seed=seed,
    )
    report["historical_sanos_history"] = str(historical_sanos_history.resolve())
    report["historical_dataset"] = str(historical_dataset.resolve())
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--bootstrap-block-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261018)
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(
        "observed panels complete: "
        f"pre-2025={report['pre_evaluation']['observations']} "
        f"2025={report['forward_evaluation']['observations']} "
        f"output={arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
