"""Train shared-plus-day-0/day-10 policies on the original training paths only."""

from __future__ import annotations

import argparse
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
from quantdata.sdlv.dhgromm1.events import (
    EventJointConfig,
    EventJointHedges,
    exercise_event_heads,
    initialize_events,
)
from quantdata.sdlv.dhgromm1.feedback_data import replay_training_states
from quantdata.sdlv.dhgromm1.feedback_run import ResidentSample, atomic_json
from quantdata.sdlv.dhgromm1.joint import (
    JointConfig,
    load_joint_checkpoint,
    train_joint,
    verify_joint_gpu,
)
from quantdata.sdlv.dhgromm1.training import (
    TrainConfig,
    _economic_utility,
    pair_summary,
    read_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True, help="completed base joint.checkpoint.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--claim-gpu-after-replay", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    output, source = args.output.resolve(), args.warm_start.resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a unique external project run directory")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("detached environment did not discover the external project data root")
    checks = resource_health()
    if any(c["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks launch")
    if any(c["health"] != "OK" for c in checks):
        print(json.dumps(dict(stage="resource-warning", checks=checks)), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage="environment-validated", interpreter=os.sys.executable,
        working_directory=str(Path.cwd()), data_root=str(root), temporary_root=os.environ.get("QUANTDATA_TEMP_DATA"),
        sample="training only", event_days=[0, 10])), flush=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "training.lock", timeout=0):
            torch.set_num_threads(4)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            config = EventJointConfig(label="bid-ask")
            if args.verify_only:
                if args.claim_gpu_after_replay:
                    claim_training_gpu(output)
                monitor.stage("verify-event-compiled-scans")
                verification = verify_joint_gpu(batch_size=args.batch_size, config=config,
                    model_type=EventJointHedges, prepare=exercise_event_heads)
                atomic_json(output / "gpu-verification.json", verification)
                print(json.dumps(verification), flush=True)
                return
            old = load_joint_checkpoint(source, config=JointConfig(label="bid-ask"))
            old_identity = json.loads((source.parent / "identity.json").read_text())
            scenarios = Path(old_identity["scenarios"])
            scenario = json.loads((scenarios / "identity.json").read_text())
            if scenario["substeps"] != 1 or old["training_paths"] != scenario["paths_per_sample"]:
                raise ValueError("the original full one-step training population is required")
            if sha256(scenarios / "identity.json") != old_identity["scenario_identity_sha256"]:
                raise ValueError("original scenario identity changed")
            if sha256(Path(scenario["calibration"]) / "diffusion.npz") != scenario["calibration_sha256"]:
                raise ValueError("calibration changed since scenario generation")
            train = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                patience=0, compile_model=True)
            files = [Path(__file__).with_name(name) for name in (
                "event_run.py", "events.py", "feedback.py", "feedback_data.py", "joint.py", "rolling_simulation.py", "allocation.py")]
            provenance = dict(scenarios=str(scenarios), scenario_identity= scenario,
                scenario_identity_sha256=sha256(scenarios / "identity.json"),
                warm_start=str(source), warm_start_sha256=sha256(source), warm_start_selected_epoch=old["selected_epoch"],
                model_config=asdict(config), model_code_version=type(config).version.unique_id48,
                joint_epochs=args.epochs, training_config=asdict(train), event_days=[0, 10],
                action="shared + 1[t=0]*head0 + 1[t=10]*head10; no cap; each head affine+deep minus frozen initial maps",
                objective="sum of separate long/short penalized negative OCEs; penalty on combined executed trades",
                checkpoint_selection="minimum full-training-set summed loss; common epoch; no validation",
                sample="TRAINING ONLY", validation_used=False, paths=scenario["paths_per_sample"],
                simulation_substeps_per_day=1, replay_backend="cpu", replay_tolerance=dict(atol=5e-8, rtol=2e-7),
                latent_normalization="original training-only means/scales retained without refitting",
                optimizer="fresh Adam for explicit architecture migration; all shared and dated trainable weights update",
                q_warning=old_identity["q_warning"], code_sha256={str(p):sha256(p) for p in files})
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != provenance:
                raise ValueError("event run identity changed; choose a new output directory")
            atomic_json(identity_path, provenance)
            monitor.stage("reconstruct-training-state", 0, old["training_paths"])
            states, records = replay_training_states(scenarios=scenarios, paths=old["training_paths"],
                progress=lambda n:monitor.record.update(completed=n))
            atomic_json(output / "replay-verification.json", dict(status="pass", sample="training only", chunks=records))
            mean, scale = (old["state_normalization"][key] for key in ("mean", "scale"))
            normalizers = old["state_normalization"]
            np.savez(output / "state-normalization.npz", mean=mean, scale=scale)
            training = read_sample(scenarios / "train", paths=old["training_paths"])
            del old
            if args.claim_gpu_after_replay:
                if torch.cuda.is_initialized():
                    raise RuntimeError("CUDA initialized before allocation")
                monitor.stage("waiting-for-gpu")
                claim_training_gpu(output)
            monitor.stage("verify-event-compiled-scans")
            atomic_json(output / "gpu-verification.json", verify_joint_gpu(batch_size=args.batch_size, config=config,
                model_type=EventJointHedges, prepare=exercise_event_heads))
            monitor.stage("upload-training-state")
            sample = ResidentSample(arrays=training, states=states, mean=mean, scale=scale, device="cuda")
            policies = train_joint(source=source, output=output, train=train, sample=sample, monitor=monitor,
                state_normalizers=normalizers, config=config,
                initializer=lambda env,cfg:initialize_events(source, env=env, config=cfg))
            monitor.stage("summarize-selected-training-policy")
            statistics = {}
            for label, sign in (("long", 1), ("short", -1)):
                policy = policies[label]
                execute = torch.compile(policy, fullgraph=True)
                gains, penalties = np.empty(sample.size), np.empty(sample.size)
                for begin in range(0, sample.size, args.batch_size):
                    with torch.no_grad():
                        price, ret, _, latent = sample.batch(slice(begin, begin + args.batch_size))
                        result = execute(price, latent, returns=ret)
                        gains[begin:begin+len(price)] = result.gain.cpu().numpy()
                        penalties[begin:begin+len(price)] = result.penalty.cpu().numpy()
                utility = _economic_utility(gains + sign*training[2], float(policy.translation.detach()), 100)
                statistics[label] = dict(fit=policy.fit_summary, training_economic_oce=pair_summary(utility),
                    mean_trade_penalty=pair_summary(penalties), mean_gain=pair_summary(gains))
                # Keep concrete training-path decisions for the case previously diagnosed.
                with torch.no_grad():
                    price, ret, _, latent = sample.batch(slice(2412, 2413))
                    point = policy(price, latent, returns=ret, eager=True)
                    statistics[label]["training_path_2412_trades"] = {
                        str(day):point.trades[0,day].cpu().tolist() for day in (0,10)}
            atomic_json(output / "report.json", dict(status="complete", **provenance, statistics=statistics,
                training_payoff_expectation=pair_summary(training[2]),
                interpretation="training statistics only, not independent evaluation or validated bid/ask quotes"))
            monitor.stage("complete", args.epochs, args.epochs)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
