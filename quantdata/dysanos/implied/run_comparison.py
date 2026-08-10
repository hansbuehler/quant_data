"""Run the complete PCA-IV-AR(1) factor/path/mode comparison matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig

from .comparison import write_aggregate_comparison
from .experiment import ImpliedPCAExperiment
from .run_experiment import run

_MODES = ("one-step", "one-step-weekly", "path", "weekly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-counts", type=int, nargs="+", default=(100_000, 1_000_000)
    )
    parser.add_argument("--modes", choices=_MODES, nargs="+", default=_MODES)
    parser.add_argument("--num-alpha", choices=("5", "all"), nargs="+", default=("5", "all"))
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--direct-p", action="store_true")
    parser.add_argument("--skip-local-tail", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--invariant-samples", type=int, default=100_000)
    parser.add_argument("--static-mesh-points", type=int, default=201)
    parser.add_argument(
        "--output-root", default="quantdata/dysanos/implied/reports"
    )
    parser.add_argument(
        "--cache-root",
        default="/media/storage/tmp/quantdata/dysanos/implied-pca-universes",
    )
    parser.add_argument("--sanos-reports", nargs="*", default=())
    args = parser.parse_args()

    importance = None if args.direct_p else GaussianSpotImportanceConfig()
    pca_reports = []
    for alpha_name in args.num_alpha:
        alpha = None if alpha_name == "all" else 5
        for paths in args.path_counts:
            for mode in args.modes:
                experiment = ImpliedPCAExperiment(
                    num_samples=paths,
                    mode=mode,
                    num_alpha=alpha,
                    num_steps=args.num_steps,
                    seed=args.seed + paths + 1_009 * _MODES.index(mode),
                    block_size=args.block_size,
                    importance_sampling=importance,
                    static_mesh_points=args.static_mesh_points,
                )
                pca_reports.append(
                    run(
                        experiment,
                        output_root=args.output_root,
                        cache_root=args.cache_root,
                        invariant_samples=args.invariant_samples,
                        run_local_tail=not args.skip_local_tail,
                        local_tail_device=(
                            None
                            if args.device == "auto"
                            else torch.device(args.device)
                        ),
                    )
                )
    comparison_path = Path(args.output_root) / "pca-sanos-comparison.json"
    comparison = write_aggregate_comparison(
        pca_reports=pca_reports,
        sanos_reports=args.sanos_reports,
        output=comparison_path,
    )
    print(
        json.dumps(
            {
                "status": comparison["status"],
                "PCA_runs": len(pca_reports),
                "report": str(comparison_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
