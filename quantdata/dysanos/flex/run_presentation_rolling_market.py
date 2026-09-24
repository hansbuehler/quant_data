"""Build fresh daily ATM 1D/2D hedge returns from Direct-200 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantdata.dysanos.flex.presentation_hedging import (
    generate_presentation_rolling_market,
)

DEFAULT_CHECKPOINT = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/"
    "dte1-5-weekly-no-am-all-history-v1/checkpoints/joint-direct-200.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--path-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    result = generate_presentation_rolling_market(
        checkpoint_path=arguments.checkpoint,
        path_dataset=arguments.path_dataset,
        output_path=arguments.output,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
