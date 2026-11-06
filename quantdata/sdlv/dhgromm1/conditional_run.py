"""CPU pilot: probability-weighted absorbing branches, ordinary DH and DH+L1/L2.

The surviving spine is sampled conditional on remaining interior at each date.
Before every transition, add both absorbing endpoint continuations with weight
W_t*p_endpoint/M, and continue the spine with W_(t+1)=W_t*p_interior.
The remaining spine gets W_20. Each outer path's weights sum to one. The plain
MC comparator uses the same ORIGINAL normal draws without conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    one_day_risk_neutral_substepped,
)
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.conditional_oce import pair_statistic, summarize
from quantdata.sdlv.dhgromm1.conditional_paths import (
    MarketPaths,
    absorbed_paths,
    boundary_probabilities,
    interior_normals,
)
from quantdata.sdlv.dhgromm1.floating import LABELS, SIGNS, FloatingTriple, FloatingTripleConfig
from quantdata.sdlv.dhgromm1.floating_run import load_floating_checkpoint
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.regularized_run import restore_config
from quantdata.torch import BaseEnvironment

jax.config.update("jax_enable_x64", True)


def paired_normals(rng: np.random.Generator, paths: int, *shape: int) -> np.ndarray:
    """Adjacent outer partners share opposite normals; different pairs independent."""
    x = rng.standard_normal((paths//2, *shape))
    return np.stack((x, -x), 1).reshape(paths, *shape)


class FrozenScorer:
    """No new Torch architecture, no policy fitting, no new information channel."""

    def __init__(self, baseline: Path, regularized: Path, output: Path) -> None:
        self.models, self.penalties, self.sources = [], [], []
        for name, source in (("dh", baseline), ("dh_l1_l2", regularized)):
            status = json.loads((source/"status.json").read_text())
            if status.get("status") != "complete":
                raise ValueError("only completed immutable ordinary/regularized policies are considered")
            folder = output/name
            folder.mkdir()
            frozen = folder/"floating.checkpoint.pt"
            before = sha256(source/"floating.checkpoint.pt")
            shutil.copyfile(source/"floating.checkpoint.pt", frozen)
            if sha256(frozen) != before or sha256(source/"floating.checkpoint.pt") != before:
                raise RuntimeError("source checkpoint changed while freezing it")
            raw = torch.load(frozen, map_location="cpu", weights_only=False)
            config = restore_config(FloatingTripleConfig(label=raw["model_config"]["label"]), raw["model_config"])
            reg = ActionRegularizerConfig(**raw["action_regularizer"]) if raw.get("action_regularizer") else None
            if (name == "dh") != (reg is None) or config.policy.trade_l1 or config.policy.trade_l2:
                raise ValueError("expected ordinary zero-cost DH and zero-cost DH with outside-utility regularizers")
            saved = load_floating_checkpoint(frozen, config=config, regularizer=reg)
            state = saved["state_dict"]
            model = FloatingTriple(BaseEnvironment(device="cpu", precision=32), config=config,
                mean=state["empty.price_mean"], scale=state["empty.price_scale"],
                return_mean=state["empty.return_mean"], payoff_mean=float(state["payoff_mean"]))
            model.load_state_dict(state)
            for label in LABELS:
                getattr(model, label).load_state_dict(saved["best_states"][label])
            model.eval().requires_grad_(False)
            self.models.append(model)
            self.penalties.append(reg)
            self.sources.append(dict(name=name, source=str(source), checkpoint_sha256=before,
                selected_epochs=saved["selected_epochs"], regularizer=raw.get("action_regularizer")))
        self.cash = np.array([float(getattr(m, k).translation) for m in self.models for k in LABELS])
        self.payoff_mean = float(self.models[0].payoff_mean)
        if self.payoff_mean != float(self.models[1].payoff_mean):
            raise ValueError("the two models have different frozen payoff centering")

    def __call__(self, paths: MarketPaths, arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
        sample = ObservableSample(arrays=arrays, changes=paths.packed, device="cpu")
        net, penalties = np.empty((sample.size, 6)), np.zeros((sample.size, 6))
        with torch.no_grad():
            for begin in range(0, sample.size, 512):
                price, returns, _, changes = sample.batch(slice(begin, begin+512))
                end = begin+len(price)
                for model_index, (model, reg) in enumerate(zip(self.models, self.penalties, strict=True)):
                    for i, (label, sign) in enumerate(zip(LABELS, SIGNS, strict=True)):
                        result = getattr(model, label)(price, changes, returns=returns, eager=True)
                        col = 3*model_index+i
                        net[begin:end, col] = result.gain.numpy().astype(float)+sign*(arrays[2][begin:end]-float(model.payoff_mean))
                        if reg is not None:
                            penalties[begin:end, col] = (reg.l1*result.trades.abs()+reg.l2*result.trades.square()).sum((1, 2)).numpy()
        return net, penalties


def calibration_arrays(directory: Path) -> tuple:
    with np.load(directory/"diffusion.npz") as f:
        return tuple(jnp.asarray(f[k], dtype=jnp.float64) for k in
                     ("support", "density_tower", "dlvsq", "annualized_loadings"))


def plain_market(calibration: tuple, normals: np.ndarray, black: np.ndarray) -> tuple[MarketPaths, tuple]:
    """Original one-step law, also used for the saved-scenario reconstruction gate."""
    support, tower0, dlv0, loadings = calibration
    count = normals.shape[1]
    state = RiskNeutralState(jnp.broadcast_to(tower0[0], (count, len(support))),
        jnp.broadcast_to(dlv0[1:], (count, *dlv0[1:].shape)), jnp.ones(count))
    tower = jnp.broadcast_to(tower0, (count, *tower0.shape))
    spot = np.ones(count)
    paths = MarketPaths.empty(count)
    step = jax.jit(one_day_risk_neutral_substepped, static_argnames=("expiry_dt", "time_power"))
    for day in range(20):
        paths.observe(day, spot, support=support, tower=tower, background=state.background_level)
        result = step(support, state, loadings[:state.future_dlvsq.shape[-2]], jnp.asarray(normals[day:day+1]),
                      jnp.asarray(black[day]), expiry_dt=1/252, time_power=1.)
        spot, tower, state = np.asarray(result.spot), result.density_tower, result.state
    paths.observe(20, spot, support=support, tower=tower, background=state.background_level)
    return paths, paths.arrays(support=support, tower=tower, background=state.background_level)


def conditional_batch(calibration: tuple, *, count: int, inner: int, seed: int, start: int,
                      scorer: FrozenScorer, monitor: AuditMonitor) -> tuple[dict, dict]:
    support, tower0, dlv0, loadings = calibration
    rng = np.random.default_rng(np.random.SeedSequence([seed, start, 0]))
    normals = paired_normals(rng, count, 20, 8).transpose(1, 0, 2)
    black = paired_normals(rng, count, 20).T
    state = RiskNeutralState(jnp.broadcast_to(tower0[0], (2*count, len(support))),
        jnp.broadcast_to(dlv0[1:], (2*count, *dlv0[1:].shape)), jnp.ones(2*count))
    tower = jnp.broadcast_to(tower0, (2*count, *tower0.shape))
    spot, survival = np.ones(2*count), np.ones(count)
    market = MarketPaths.empty(2*count)
    step = jax.jit(one_day_risk_neutral_substepped, static_argnames=("expiry_dt", "time_power"))
    atoms = {key:[] for key in ("net", "penalty", "weights", "outer", "payoff", "day", "endpoint")}
    for day in range(20):
        monitor.stage("conditional-boundary-prefix", start*20+day*count, monitor.record.get("total", 1))
        market.observe(day, spot, support=support, tower=tower, background=state.background_level)
        lower, upper, interior = boundary_probabilities(np.asarray(state.front_density)[:count])
        branch_rng = np.random.default_rng(np.random.SeedSequence([seed, start, day, 1]))
        branch_black = paired_normals(branch_rng, count, inner, 20-day)
        prefix = market.fork(np.arange(count))
        branch = absorbed_paths(prefix, day=day, background=np.asarray(state.background_level)[:count],
            endpoints=np.asarray(support)[[0, -1]], normals=branch_black)
        arrays = branch.arrays()
        net, penalty = scorer(branch, arrays)
        weights = np.repeat(survival[:, None]*np.stack((lower, upper), -1)/inner, inner, axis=1).reshape(-1)
        atoms["net"].append(net)
        atoms["penalty"].append(penalty)
        atoms["weights"].append(weights)
        atoms["outer"].append(np.repeat(np.arange(count)+start, 2*inner))
        atoms["payoff"].append(arrays[2])
        atoms["day"].append(np.full(len(weights), day, np.int8))
        atoms["endpoint"].append(np.tile(np.repeat([0, 1], inner), count).astype(np.int8))
        survival *= interior
        conditioned = interior_normals(normals[day], lower, interior)
        joint = np.concatenate((conditioned, normals[day]))[None]
        result = step(support, state, loadings[:state.future_dlvsq.shape[-2]], jnp.asarray(joint),
                      jnp.asarray(np.r_[black[day], black[day]]), expiry_dt=1/252, time_power=1.)
        sampled = np.asarray(result.sampled_front_index)[:count]
        if np.any((survival > 0) & ((sampled == 0) | (sampled == len(support)-1))):
            raise FloatingPointError("positive-weight conditional spine did not remain interior")
        spot, tower, state = np.asarray(result.spot), result.density_tower, result.state
    market.observe(20, spot, support=support, tower=tower, background=state.background_level)
    arrays = market.arrays(support=support, tower=tower, background=state.background_level)
    net, penalty = scorer(market, arrays)
    for key, value in dict(net=net[:count], penalty=penalty[:count], weights=survival,
            outer=np.arange(count)+start, payoff=arrays[2][:count], day=np.full(count, 20, np.int8),
            endpoint=np.full(count, -1, np.int8)).items():
        atoms[key].append(value)
    plain = dict(net=net[count:], penalty=penalty[count:], weights=np.ones(count),
        outer=np.arange(count)+start, payoff=arrays[2][count:])
    return {k:np.concatenate(v) for k, v in atoms.items()}, plain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--regularized-run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.paths < 4 or args.paths % 2 or args.batch_size < 2 or args.batch_size % 2 or args.inner < 1:
        raise ValueError("use even outer path/batch counts and positive inner count")
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(allowed[-min(8, len(allowed)):]))
    torch.set_num_threads(4)
    root, output = Path(os.environ["QUANTDATA_ROOT"]).resolve(), args.output.resolve()
    if quantdata.env.test_mode or not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("explicit production root and unique external output required")
    if torch.cuda.is_initialized() or jax.default_backend() != "cpu":
        raise RuntimeError("this pilot is CPU only, with no GPU allocation")
    checks = resource_health()
    if any(c["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks conditional pilot")
    print(json.dumps(dict(stage="conditional-detached-preflight-passed",training_root=str(quantdata.env.locations.training.path),
        cpu_only=True,checks=checks)),flush=True)
    output.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate_report.read_text())
    if gate.get("status") != "passed" or gate["calibration_sha256"] != sha256(args.calibration/"diffusion.npz"):
        raise ValueError("the exact contract/causal-policy reconstruction gate must pass first")
    if any(sha256(Path(path)) != digest for path,digest in gate["code_sha256"].items()):
        raise ValueError("conditional pilot code changed after the reconstruction gate")
    monitor = AuditMonitor(output)
    monitor.thread.start()
    started = time.monotonic()
    try:
        with FileLock(output/"pilot.lock", timeout=0):
            scorer = FrozenScorer(args.baseline_run, args.regularized_run, output)
            if scorer.sources != gate["sources"]:
                raise ValueError("frozen checkpoints differ from the reconstruction gate")
            calibration = calibration_arrays(args.calibration)
            for source in (args.baseline_run, args.regularized_run):
                identity = json.loads((source/"identity.json").read_text())
                scenarios = json.loads((Path(identity["scenarios"])/"identity.json").read_text())
                if scenarios["substeps"] != 1 or scenarios["calibration_sha256"] != sha256(args.calibration/"diffusion.npz"):
                    raise ValueError("checkpoint and conditional simulator reference different models")
            provenance = dict(schema="dhgromm1-conditional-absorption-v1",paths=args.paths,inner=args.inner,
                batch_size=args.batch_size,seed=args.seed,device="CPU only",sources=scorer.sources,
                calibration=str(args.calibration),calibration_sha256=sha256(args.calibration/"diffusion.npz"),
                simulation_substeps=1,hedge_decisions_per_day=1,objective="ordinary OCE only; DH and DH+L1/L2; no robust case",
                centering="unchanged original training return/payoff means; no new means fitted",
                conditional_law="Gaussian factor zero truncated to interior CDF interval; both absorbing endpoint branches use exact 5% Black continuation",
                branch_weights="W_t*p_endpoint/M; W_(t+1)=W_t*p_interior; terminal survivor W_20",
                training=False,validation_checkpoint_selection=False,
                reconstruction_gate=str(args.gate_report),reconstruction_gate_sha256=sha256(args.gate_report),
                code_sha256={str(p):sha256(p) for p in (Path(__file__),Path(__file__).with_name("conditional_paths.py"),Path(__file__).with_name("conditional_oce.py"))})
            atomic_json(output/"identity.json",provenance)
            atomic_json(output/"status.json",dict(status="running",stage="conditional-pilot"))
            monitor.stage("conditional-boundary-prefix",0,args.paths*20)
            for start in range(0, args.paths, args.batch_size):
                count = min(args.batch_size, args.paths-start)
                atoms,plain = conditional_batch(calibration,count=count,inner=args.inner,seed=args.seed,start=start,
                                                scorer=scorer,monitor=monitor)
                np.savez_compressed(output/f"conditional-{start:06d}.npz",**atoms)
                np.savez_compressed(output/f"plain-{start:06d}.npz",**plain)
                elapsed,completed = time.monotonic()-started,start+count
                print(json.dumps(dict(stage="conditional-batch-complete",completed=completed,total=args.paths,
                    elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(args.paths-completed)/completed)),flush=True)
            reports,retained = {},{}
            for estimator in ("conditional","plain"):
                chunks = []
                for file in sorted(output.glob(f"{estimator}-[0-9]*.npz")):
                    with np.load(file) as f:
                        chunks.append({k:f[k] for k in ("net","penalty","weights","outer","payoff")})
                arrays = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
                reports[estimator],values = summarize(arrays["net"],arrays["penalty"],arrays["weights"],arrays["outer"],
                    outer_paths=args.paths,training_cash=scorer.cash,payoff_mean=scorer.payoff_mean)
                eq = np.bincount(arrays["outer"],weights=arrays["weights"]*arrays["payoff"],minlength=args.paths)
                reports[estimator]["EQ_payoff"] = pair_statistic(eq)
                reports[estimator]["EQ_payoff"]["mean"] = reports[estimator]["EQ_payoff"].pop("oce")
                retained.update({estimator+"_"+k:v for k,v in values.items()})
            difference = retained["conditional_fixed_training_cash_outer_utilities"]-retained["plain_fixed_training_cash_outer_utilities"]
            reports["paired_fixed_cash_difference"] = {name:pair_statistic(difference[:,i]) for i,name in
                enumerate(("dh_empty","dh_long","dh_short","dh_l1_l2_empty","dh_l1_l2_long","dh_l1_l2_short"))}
            reports["provenance"] = provenance
            reports["elapsed_seconds"] = time.monotonic()-started
            np.savez_compressed(output/"outer-cluster-utilities.npz",**retained)
            atomic_json(output/"report.json",reports)
            atomic_json(output/"status.json",dict(status="complete",outer_paths=args.paths,elapsed_seconds=reports["elapsed_seconds"]))
            monitor.stage("complete",args.paths,args.paths)
    except Exception as error:
        atomic_json(output/"status.json",dict(status="failed",error=repr(error)))
        raise
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
