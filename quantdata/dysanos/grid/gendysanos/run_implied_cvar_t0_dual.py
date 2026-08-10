"""Bracket the empirical time-zero CVaR density cap on the GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
from cdxcore.err import verify_inp

from quantdata.torch import BaseEnvironment

from .required_runs import RequiredCase
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case, _to_device
from .run_implied_cvar_t6_dual import (
    _fit_one_cap,
    conditional_moments,
    fit_entropy_density,
    project_weighted_capped_density_,
)
from .run_parallel_direction_comparison import (
    _select_times_and_fold,
    _to_itm_dollar_portfolio,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/implied-cvar"
)


def run(args: argparse.Namespace) -> dict:
    lambdas = tuple(float(value) for value in args.lambdas)
    verify_inp(
        bool(lambdas)
        and all(math.isfinite(value) and value > 0.0 for value in lambdas)
        and all(left < right for left, right in zip(lambdas, lambdas[1:])),
        lambda: f"lambdas must be finite, positive, and increasing: {lambdas}",
    )
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="raw",
        cost_rate=0.0,
        path_count=100_000,
    )
    loaded = _load_case(case, cache_root=Path(args.cache_root), need_primitives=True)
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=32, device=args.device, debug=False)
    full_data, source_split = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    data, _ = _select_times_and_fold(
        full_data,
        deterministic_time=0,
        conditional_time=6,
        fold=0,
        seed=100_097,
    )
    del full_data
    data, normalization = _to_itm_dollar_portfolio(
        data, cost_rate=0.0, vector_barrier=False
    )
    native_return = data.returns[:, 0]
    returns = native_return.reshape(native_return.shape[0], -1).double()
    probability = data.path_weight.double()
    probability /= probability.sum()
    probability = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    probability /= probability.sum()
    kernel_weight = torch.ones(
        (returns.shape[0], 1), dtype=returns.dtype, device=returns.device
    )
    moment_scale = torch.sqrt(
        kernel_weight.square().mT
        @ (probability[:, None] * returns.square())
    ).clamp_min(1.0e-12)
    entropy_density, entropy = fit_entropy_density(
        probability,
        kernel_weight,
        returns,
        max_iter=args.entropy_steps,
    )
    print(f"Entropy density: {entropy}", flush=True)

    density = torch.ones_like(probability)
    fits: list[dict] = []
    for lmbda in lambdas:
        entropy_start = entropy_density.clone()
        project_weighted_capped_density_(
            entropy_start, probability, 1.0 + lmbda
        )
        with torch.no_grad():
            warm_loss = torch.square(
                conditional_moments(
                    density, probability, kernel_weight, returns
                )
                / moment_scale
            ).sum()
            entropy_loss = torch.square(
                conditional_moments(
                    entropy_start, probability, kernel_weight, returns
                )
                / moment_scale
            ).sum()
        if entropy_loss < warm_loss:
            density = entropy_start
        density, fit = _fit_one_cap(
            density,
            density_cap=1.0 + lmbda,
            probability=probability,
            kernel_weight=kernel_weight,
            returns=returns,
            moment_scale=moment_scale,
            steps=args.steps,
            batch_size=100_000,
            learning_rate=args.lr,
            value_update=args.value_update,
            tolerance=args.tolerance,
        )
        fits.append(fit)

    report = {
        "status": "complete",
        "experiment": "empirical time-zero CVaR density-cap bracket",
        "definition": (
            "D is constrained by E_p[D]=1 and 0<=D<=1+lambda; the reported "
            "upper bound is max_j |E_p[D R_j]| for the deterministic portfolio class"
        ),
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "source_split_unused_for_fitting": source_split,
        "portfolio_normalization": normalization,
        "entropy_density_initializer": entropy,
        "fits": fits,
        "smallest_tested_lambda_certified_zero": next(
            (fit["lambda"] for fit in fits if fit["converged"]), None
        ),
    }
    report_path = Path(args.output_root) / (
        f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-t0-dual.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, report_path)
    print(f"Completed time-zero dual report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 100),
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--value-update", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--entropy-steps", type=int, default=1_000)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
