"""Generate Direct-200 DYSANOS paths for the CBOE hedging recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantdata.dysanos.flex.presentation_hedging import generate_presentation_paths

DEFAULT_ROOT = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/"
    "dte1-5-weekly-no-am-all-history-v1/checkpoints"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "joint-direct-200.pt",
    )
    parser.add_argument(
        "--target-history",
        type=Path,
        default=DEFAULT_ROOT / "joint-direct-200-targets.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-paths", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=58_123)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    result = generate_presentation_paths(
        checkpoint_path=arguments.checkpoint,
        target_history_path=arguments.target_history,
        output_path=arguments.output,
        num_paths=arguments.num_paths,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
