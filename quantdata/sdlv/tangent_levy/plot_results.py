"""Plot an existing Tangent Lévy history artifact without running simulation."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from cdxcore.verbose import Context

from .config import TangentLevyConfig
from .dynamics import fit_dynamics
from .experiment import load_calibrated_history, validate_reconstruction_gate
from .plotting import write_diagnostic_figure


def main() -> None:
    """Load the versioned history cache and write its diagnostic figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 8, 30))
    parser.add_argument(
        "--dynamics-end",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
    )
    parser.add_argument("--label", default="spx-2022-2025")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tangent-levy-diagnostics.png"),
    )
    args = parser.parse_args()
    config = TangentLevyConfig(label=args.label)
    history = load_calibrated_history(
        args.ticker,
        args.start,
        args.end,
        config=config,
        verbose=Context.quiet,
    )
    dynamics_position = sum(value <= args.dynamics_end for value in history.dates)
    if config.dynamics.require_reconstruction_pass:
        validate_reconstruction_gate(
            history,
            dynamics_position=dynamics_position,
            include_simulation_origin=False,
        )
    dynamics = fit_dynamics(
        history.forward_density[:dynamics_position],
        config=config.dynamics,
        consecutive_mask=history.consecutive_mask[: dynamics_position - 1],
    )
    output = write_diagnostic_figure(
        history,
        dynamics,
        output=args.output,
        reconstruction_tolerance=(
            config.static_fit.reconstruction_relative_tolerance
        ),
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
