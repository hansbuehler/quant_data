"""Joint summed-loss training of separate full-state bid/ask hedge networks."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu
from quantdata.sdlv.dhgromm1.feedback import packed_geometry
from quantdata.sdlv.dhgromm1.feedback_data import replay_states, state_normalization
from quantdata.sdlv.dhgromm1.joint import JointConfig, train_joint, verify_joint_gpu
from quantdata.sdlv.dhgromm1.training import (
    TrainConfig,
    _economic_utility,
    day10_straddle_benchmark,
    optimize_translation,
    pair_summary,
    read_sample,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ResidentSample:
    """One upload; ragged full states are gathered/padded entirely on GPU."""

    def __init__(self, *, arrays: tuple[np.ndarray, ...], states: np.ndarray,
                 mean: np.ndarray, scale: np.ndarray, device: str) -> None:
        self.arrays = tuple(torch.as_tensor(x, device=device, dtype=torch.float32) for x in arrays)
        self.states = torch.as_tensor(states, device=device)
        self.mean = torch.as_tensor(mean, device=device)
        self.scale = torch.as_tensor(scale, device=device)
        _, index, valid = packed_geometry()
        self.index = torch.as_tensor(index.ravel(), device=device)
        self.valid = torch.as_tensor(valid, device=device)
        self.size = len(states)

    def batch(self, selection: slice | torch.Tensor) -> tuple[torch.Tensor, ...]:
        prices, returns, payoff = (x[selection] for x in self.arrays)
        packed = (self.states[selection] - self.mean) / self.scale
        state = packed.index_select(1, self.index).reshape(-1, 20, 5003)
        state = state.masked_fill(~self.valid[None], 0)
        return prices, returns, payoff, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500, help="joint epochs, each visiting the full sample for BOTH networks")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--claim-gpu-after-replay", action="store_true")
    args = parser.parse_args()
    train = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                        patience=0, compile_model=True)
    output = args.output.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a unique run directory in the external project data root")
    if args.paths < 4 or args.paths % 2:
        raise ValueError("training/evaluation samples must contain complete antithetic pairs")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("detached environment did not discover the external project data root")
    print(json.dumps(dict(stage="environment-validated", interpreter=os.sys.executable,
        working_directory=str(Path.cwd()), data_root=str(root),
        temporary_root=os.environ.get("QUANTDATA_TEMP_DATA"))), flush=True)
    checks = resource_health()
    if any(x["health"] in {"STOP", "UNAVAILABLE", "ERROR"} for x in checks):
        raise RuntimeError("resource manager blocks launch")
    if any(x["health"] != "OK" for x in checks):
        print(json.dumps(dict(stage="resource-warning", checks=checks)), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "training.lock", timeout=0):
            torch.set_num_threads(4)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            if args.verify_only:
                if args.claim_gpu_after_replay:
                    claim_training_gpu(output)
                monitor.stage("verify-joint-compiled-scans")
                verification = verify_joint_gpu(batch_size=args.batch_size)
                atomic_json(output / "gpu-verification.json", verification)
                print(json.dumps(verification), flush=True)
                return
            identity = json.loads((args.scenarios / "identity.json").read_text())
            if identity["substeps"] != 1 or identity["paths_per_sample"] < args.paths:
                raise ValueError("the accepted one-step/day complete path sample is required")
            if sha256(Path(identity["calibration"]) / "diffusion.npz") != identity["calibration_sha256"]:
                raise ValueError("calibration changed since the original one-step paths were generated")
            files = [Path(__file__), *(Path(__file__).with_name(name) for name in
                     ("feedback.py", "feedback_data.py", "rolling_simulation.py", "allocation.py", "joint.py"))]
            provenance = dict(scenarios=str(args.scenarios.resolve()), scenario_identity=identity,
                scenario_identity_sha256=sha256(args.scenarios / "identity.json"), joint_epochs=args.epochs,
                epochs_per_network=args.epochs, training_mode="sum of bid/ask losses, separate networks/translations",
                checkpoint_selection="minimum training SUM at a common epoch",
                joint_model_config=asdict(JointConfig(label="bid-ask")),
                joint_model_code_version=JointConfig.version.unique_id48,
                paths_per_sample=args.paths, simulation_substeps_per_day=1, trading_days=20,
                train_config=asdict(train), latent_encoding="front density; log1p future DLVSQ; log background; FP32",
                replay_tolerance=dict(atol=5e-8, rtol=2e-7, original_training_marks_unchanged=True),
                replay_backend="cpu", claim_gpu_after_replay=args.claim_gpu_after_replay,
                warm_start=str(args.warm_start.resolve()),
                inputs="t, 21 prices, full GROMM state, previous trades, live cohort units, LSTM memory",
                recurrence="two compiled differentiable scans",
                time_gate="1+tanh(G(t)-G0(t)); identity initially; raw t bypass; action uncapped",
                empty="frozen zero-trading baseline, OCE exactly zero",
                q_warning="Accepted one-step unconditional-drift warning; not an unconditional no-arbitrage pass",
                code_sha256={str(p): sha256(p) for p in files},
                source_checkpoints={label: sha256(args.warm_start / f"{label}.checkpoint.pt") for label in ("long", "short")})
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != provenance:
                raise ValueError("run identity changed; choose a new output directory")
            atomic_json(identity_path, provenance)
            monitor.stage("reconstruct-full-state", 0, 2 * args.paths)
            train_states, evaluation_states, replay = replay_states(scenarios=args.scenarios, paths=args.paths,
                progress=lambda n: monitor.record.update(completed=n))
            atomic_json(output / "replay-verification.json", dict(status="pass", chunks=replay,
                state_storage="RAM only, FP32 ragged packing; no PCA projection"))
            monitor.stage("normalize-training-state")
            mean, scale = state_normalization(train_states)
            np.savez(output / "state-normalization.npz", mean=mean, scale=scale)
            total_paths = identity["paths_per_sample"]
            training = tuple(x[:args.paths] for x in read_sample(args.scenarios / "train", paths=total_paths))
            evaluation = tuple(x[:args.paths] for x in read_sample(args.scenarios / "evaluation", paths=total_paths))
            if args.claim_gpu_after_replay:
                if torch.cuda.is_initialized():
                    raise RuntimeError("CUDA initialized before a GPU was allocated")
                monitor.stage("waiting-for-gpu")
                claim_training_gpu(output)
            monitor.stage("verify-joint-compiled-scans")
            atomic_json(output / "gpu-verification.json", verify_joint_gpu(batch_size=args.batch_size))
            monitor.stage("upload-training-state")
            sample = ResidentSample(arrays=training, states=train_states, mean=mean, scale=scale, device="cuda")
            normalizers = dict(mean=mean, scale=scale, representation="front density, log1p future DLVSQ, log background")
            policies = train_joint(source=args.warm_start, output=output, train=train, sample=sample,
                                   monitor=monitor, state_normalizers=normalizers)
            del sample, train_states
            gc.collect()
            torch.cuda.empty_cache()
            monitor.stage("upload-independent-evaluation")
            sample = ResidentSample(arrays=evaluation, states=evaluation_states, mean=mean, scale=scale, device="cuda")
            statistics = {}
            for label, sign in (("long", 1), ("short", -1)):
                policy = policies[label]
                execute = torch.compile(policy, fullgraph=True)
                gains = torch.empty(args.paths, device="cuda")
                penalties = torch.empty_like(gains)
                with torch.no_grad():
                    for begin in range(0, args.paths, args.batch_size):
                        monitor.record = dict(stage=f"evaluate-{label}", completed=begin, total=args.paths)
                        price, ret, _, latent = sample.batch(slice(begin, begin + args.batch_size))
                        result = execute(price, latent, returns=ret)
                        gains[begin:begin + len(price)] = result.gain
                        penalties[begin:begin + len(price)] = result.penalty
                pnl = gains.cpu().numpy().astype(np.float64) + sign * evaluation[2]
                utility = _economic_utility(pnl, float(policy.translation.detach()), 100)
                statistics[label] = dict(economic_oce_frozen_translation=pair_summary(utility),
                    mean_regularizer=pair_summary(penalties.cpu().numpy()), fit=policy.fit_summary)
                np.savez_compressed(output / f"{label}-evaluation.npz", pnl=pnl, utility=utility)
            benchmark_gain, benchmark_penalty, benchmark_error = day10_straddle_benchmark(*training)
            benchmark = {}
            for label, sign in (("long", 1), ("short", -1)):
                pnl = sign * training[2] - sign * benchmark_gain
                m = optimize_translation(pnl, 100)
                benchmark[label] = dict(penalized_training_loss=float(-_economic_utility(pnl, m, 100).mean()
                                                                      + benchmark_penalty.mean()))
            report = dict(status="complete", **provenance, statistics=statistics,
                q_expectation=pair_summary(evaluation[2]),
                bid=statistics["long"]["economic_oce_frozen_translation"]["mean"],
                ask=-statistics["short"]["economic_oce_frozen_translation"]["mean"], empty_oce=0,
                benchmark=benchmark, benchmark_identity_max_error=benchmark_error,
                convergence_claim="not assumed; compare benchmark and independent-sample performance")
            atomic_json(output / "report.json", report)
            monitor.stage("complete", args.epochs, args.epochs)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
