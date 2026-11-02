"""Measure bounded streaming and compiled valuation on isolated checkpoint copies."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from quantdata.gromm2.hedging_policy import HedgingNetworkConfig, _Progress, training_environment
from quantdata.gromm2.joint_hedging import JointTrainConfig
from quantdata.gromm2.market_batching import GPU_CAPTURE_LOCK, MarketBatchIterator
from quantdata.gromm2.market_hedging import (
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
    load_market_scenarios,
)
from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.portfolio_hedging import (
    LIABILITIES,
    IndependentHedges,
    portfolio_evaluation_forward,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom


def benchmark(*, checkpoint: Path, run_directory: Path, output: Path, batches: int = 48) -> None:
    """Keep production checkpoints immutable and retain no generated path arrays."""
    if batches < 16:
        raise ValueError("at least sixteen benchmark batches are required")
    check_resource_headroom()
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = copy.deepcopy(saved["identity"]["config"])
    configuration["deep"] = HedgingNetworkConfig(**configuration["deep"])
    config = UnitTurnoverBookConfig(**configuration)
    train_config = JointTrainConfig(**saved["report"]["train_config"])
    torch.set_num_threads(train_config.cpu_threads)
    env = training_environment(train_config)
    normalization = tuple(saved["model"][f"policies.zero.normalization_{name}"].numpy()
        for name in ("mean", "scale"))
    model = IndependentHedges(env, config=config, normalization=normalization, policy_class=UnitTurnoverBookPolicy)
    model.load_state_dict(saved["model"])
    optimizers = {name: torch.optim.Adam(policy.parameters(), lr=train_config.learning_rate)
        for name, policy in model.policies.items()}
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][name])
    step = IndependentPortfolioStep(model, optimizers, gradient_clip=train_config.gradient_clip)
    data, _ = load_market_scenarios(run_directory / "train.h5", split="train",
        case=config.label.split("-", 1)[0], kind=config.kind)
    order = np.random.default_rng(20261028).permutation(len(data.payoff))[:batches * train_config.batch_size]
    progress = _Progress(label="isolated-training-pipeline-review", interval=60.0)
    progress.state.update(stage="warming-captured-training", total=4 * batches + 3)
    previous = {sig: signal.signal(sig, lambda *_: progress.stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    progress.thread.start()
    try:
        with MarketBatchIterator(data, env=env, batch_size=train_config.batch_size,
                order=order[:train_config.batch_size]) as iterator:
            resident = next(iterator)
        for _ in range(3):
            step(*resident, tuple(LIABILITIES))
            progress.state["completed"] += 1
        torch.cuda.synchronize(env.device)
        measurements = []
        for trial, workers in enumerate((1, 2, 2, 1)):
            check_resource_headroom()
            progress.state.update(stage=f"streaming-workers-{workers}-trial-{trial}")
            durations = []
            started = previous_batch = time.monotonic()
            with MarketBatchIterator(data, env=env, batch_size=train_config.batch_size,
                    order=order, prefetch_workers=workers) as iterator:
                for features, prices, payoff in iterator:
                    step(features, prices, payoff, tuple(LIABILITIES))
                    now = time.monotonic()
                    durations.append(1000 * (now - previous_batch))
                    previous_batch = now
                    progress.state["completed"] += 1
                    if progress.stop.is_set():
                        raise InterruptedError("isolated pipeline benchmark stopped")
            torch.cuda.synchronize(env.device)
            elapsed = time.monotonic() - started
            record = dict(trial=trial, prefetch_workers=workers, paths=len(order), elapsed_seconds=elapsed,
                paths_per_second=len(order) / elapsed, batch_median_ms=float(np.median(durations)),
                batch_p90_ms=float(np.quantile(durations, 0.9)))
            measurements.append(record)
            progress.emit(measurement=record)
        progress.state.update(stage="compiled-population-evaluation", completed=0, total=0)
        model.eval()
        eager = portfolio_evaluation_forward(model)
        compiled = portfolio_evaluation_forward(model, compile=True)
        evaluation = []
        with torch.no_grad():
            for count in (4096, 576):
                with MarketBatchIterator(data, env=env, batch_size=count, order=order[:count]) as iterator:
                    features, prices, _ = next(iterator)
                reference = eager(features, prices).cpu()
                with GPU_CAPTURE_LOCK:
                    for _ in range(3):
                        torch.compiler.cudagraph_mark_step_begin()
                        result = compiled(features, prices)
                candidate = result.cpu()
                torch.testing.assert_close(candidate, reference, rtol=2e-5, atol=2e-7)
                del result
                torch.cuda.synchronize(env.device)
                started = time.monotonic()
                for _ in range(8):
                    with GPU_CAPTURE_LOCK:
                        torch.compiler.cudagraph_mark_step_begin()
                        compiled(features, prices).cpu()
                elapsed = time.monotonic() - started
                evaluation.append(dict(paths=count, batch_mean_ms=1000 * elapsed / 8,
                    maximum_absolute_error=float((candidate - reference).abs().max())))
                progress.emit(evaluation=evaluation[-1])
                progress.state["completed"] += 1
        report = dict(checkpoint=str(checkpoint), checkpoint_written=False, input_paths_reused=True,
            batch_size=train_config.batch_size, measurements=measurements, evaluation=evaluation)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        progress.closed.set()
        progress.thread.join(timeout=2)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=48)
    args = parser.parse_args()
    benchmark(checkpoint=args.checkpoint, run_directory=args.run_directory, output=args.output, batches=args.batches)


if __name__ == "__main__":
    main()
