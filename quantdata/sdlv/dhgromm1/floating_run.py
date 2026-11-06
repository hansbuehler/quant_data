"""Train centered floating-option hedges, retaining the exact day-10 benchmark."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from scipy.optimize import brentq

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache
from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    SIGNS,
    FloatingTriple,
    FloatingTripleConfig,
    fit_return_means,
    read_floating_sample,
)
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.persistence import atomic_json, write_history
from quantdata.sdlv.dhgromm1.regularization import (
    ActionRegularizerConfig,
    action_metrics,
    regularized_losses,
)
from quantdata.sdlv.dhgromm1.time_decider import (
    TimeDeciderTriple,
    TimeDeciderTripleConfig,
    TimeDeciderWithA10Config,
    TimeDeciderWithA10Triple,
)
from quantdata.sdlv.dhgromm1.training import (
    TrainConfig,
    _economic_utility,
    optimize_translation,
    pair_summary,
)
from quantdata.sdlv.dhgromm1.triple_plot import plot_triple_history
from quantdata.torch import BaseEnvironment


def scalar_benchmarks(*, prices: np.ndarray, returns: np.ndarray, payoff: np.ndarray,
                      means: np.ndarray, config: FloatingTripleConfig, payoff_mean: float | None = None) -> dict:
    """Solve the contained constant-alpha benchmark, both with and without costs."""
    b = payoff-(payoff.mean() if payoff_mean is None else payoff_mean)
    centered = returns-means
    d = centered[:, 10, 17]-2*centered[:, 10, 1]
    premium = 1+(2*prices[:, 10, 1]-prices[:, 10, 17])/prices[:, 10, 0]
    error = float(np.max(np.abs(d-((premium-payoff)-(means[10,17]-2*means[10,1])))))
    if error > 1e-10:
        raise ValueError(f"floating day-10 straddle identity failed: {error}")
    lam = config.policy.risk_aversion
    report = dict(replication_max_error=error, mean_H=float(payoff.mean()), mean_P10=float(premium.mean()), policies={})
    for label, sign in zip(LABELS, SIGNS, strict=True):
        report["policies"][label] = {}
        for costs in (False, True):
            def profile(a):
                penalty = (3*config.policy.trade_l1*abs(a)+5*config.policy.trade_l2*a*a) if costs else 0.
                net = sign*(b+a*d)-penalty
                cash = optimize_translation(net, lam)
                z = net-cash
                deriv = np.exp(-lam*np.maximum(z, 0))+lam*np.maximum(-z, 0)
                cost_deriv = (3*config.policy.trade_l1*np.sign(a)+10*config.policy.trade_l2*a) if costs else 0.
                gradient = float(np.mean(deriv*(sign*d-cost_deriv)))
                return float(_economic_utility(net, cash, lam).mean()), cash, gradient
            if sign:
                lo, hi = 0., 2.
                for _ in range(20):
                    if profile(lo)[2] > 0 and profile(hi)[2] < 0:
                        break
                    lo, hi = lo-(hi-lo), hi+(hi-lo)
                else:
                    raise RuntimeError("could not bracket the unrestricted scalar benchmark")
                alpha = float(brentq(lambda a:profile(a)[2], lo, hi, xtol=1e-13))
            else:
                alpha = 0.
            value, cash, gradient = profile(alpha)
            report["policies"][label]["net" if costs else "economic"] = dict(
                alpha=alpha, cash=cash, y=-cash, oce=value, gradient=gradient,
                fixed_a1_oce=profile(1.)[0] if sign else 0.)
    return report


def load_floating_checkpoint(path: Path, *, config: FloatingTripleConfig,
                             regularizer: ActionRegularizerConfig | None = None) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("model_code_version") != type(config).version.unique_id48
            or saved.get("model_config") != asdict(config) or saved.get("user_version") != config.user_version
            or saved.get("cost_treatment") != "inside_utility"):
        raise ValueError("floating checkpoint gain/architecture/objective version differs")
    expected = asdict(regularizer) if regularizer is not None else None
    code = ActionRegularizerConfig.version.unique_id48 if regularizer is not None else None
    if saved.get("action_regularizer") != expected or saved.get("regularizer_code_version") != code:
        raise ValueError("checkpoint action regularizer/objective version differs")
    return saved


def load_cost_continuation(path: Path, *, config: FloatingTripleConfig) -> dict:
    """Explicitly authorize ONLY a cost-coefficient change; preserve optimizer/RNG.

    Do not weaken the normal loader or reinterpret its old best losses as
    new-objective losses. A new run must rescore weights and reset selection.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    old_policy = raw["model_config"]["policy"]
    source_config = replace(config, policy=replace(config.policy,
        trade_l1=old_policy["trade_l1"], trade_l2=old_policy["trade_l2"]))
    return load_floating_checkpoint(path, config=source_config)


def load_floating_model(path: Path, *, config: FloatingTripleConfig, env: BaseEnvironment,
                        regularizer: ActionRegularizerConfig | None = None) -> FloatingTriple:
    """Read selected per-policy weights without fitting or holding a training lock."""
    saved = load_floating_checkpoint(path, config=config, regularizer=regularizer)
    state = saved["state_dict"]
    model_class = (TimeDeciderWithA10Triple if isinstance(config,TimeDeciderWithA10Config) else
                   TimeDeciderTriple if isinstance(config,TimeDeciderTripleConfig) else FloatingTriple)
    model = model_class(env, config=config, mean=state["empty.price_mean"], scale=state["empty.price_scale"],
        return_mean=state["empty.return_mean"], payoff_mean=float(state["payoff_mean"]))
    model.load_state_dict(state)
    for label in LABELS:
        getattr(model, label).load_state_dict(saved["best_states"][label])
    model.requires_grad_(False)
    return model


def verify_compiled(model: FloatingTriple, sample: ObservableSample, *, batch_size: int,
                    regularizer: ActionRegularizerConfig | None = None) -> dict:
    """Compare GPU scan with same-device eager; report CPU rounding separately."""
    original = {k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        for label in LABELS:
            policy = getattr(model, label)
            policy.trade_head.weight.add_(torch.randn_like(policy.trade_head.weight)*1e-5)
            policy.day0.add_(.0001)
            if hasattr(policy, "day10"):
                policy.day10.add_(.0001)
    reference = copy.deepcopy(model)
    cpu_reference = copy.deepcopy(model).cpu()
    price, ret, payoff, changes = sample.batch(slice(0, batch_size))
    functions = model.policy_functions(compile_model=True)
    actual_results = tuple(f(price, changes, returns=ret) for f in functions)
    def objective(which, results, claim):
        raw = which.losses(results, claim)
        return regularized_losses(raw, results, regularizer) if regularizer is not None else raw

    actual = objective(model, actual_results, payoff)
    expected = objective(reference, tuple(getattr(reference,k)(price,changes,returns=ret,eager=True)
                                         for k in LABELS),payoff)
    cpu_expected = objective(cpu_reference, tuple(getattr(cpu_reference,k)(price.cpu(),changes.cpu(),returns=ret.cpu(),eager=True)
                                                 for k in LABELS),payoff.cpu())
    actual.sum().backward()
    expected.sum().backward()
    cpu_expected.sum().backward()
    analytic_cash = {}
    for label,sign,out in zip(LABELS,SIGNS,actual_results,strict=True):
        cash = getattr(model,label).translation
        z = (out.gain+sign*(payoff-model.payoff_mean)-out.penalty-cash).detach().double()
        derivative = torch.exp(-100*z.clamp_min(0))+100*(-z).clamp_min(0)
        if hasattr(getattr(model,label),"tail_threshold") and model.config.tail_fraction<1:
            # The shared lower-tail threshold is a training-only scalar.
            # Match the actual FP32 hinge boundary and its valid q tie slope.
            utility = model._utility(z.to(out.gain.dtype),torch.zeros_like(out.gain)).detach()
            threshold = getattr(model,label).tail_threshold
            weight = ((utility<threshold.detach()).double()/model.config.tail_fraction
                      +(utility==threshold.detach()).double())
            value = (weight*derivative).mean()-1
            torch.testing.assert_close(threshold.grad.double(),weight.mean()-1,atol=2e-6,rtol=1e-4)
        else:
            value = derivative.mean()-1
        torch.testing.assert_close(cash.grad.double(),value,atol=2e-6,rtol=1e-4)
        analytic_cash[label] = float(value)
    diagnostics = dict(gpu_compiled_losses=actual.detach().cpu().tolist(),
        gpu_eager_losses=expected.detach().cpu().tolist(),cpu_eager_losses=cpu_expected.detach().tolist(),
        cash_gradients={k:dict(compiled=float(getattr(model,k).translation.grad),
            gpu_eager=float(getattr(reference,k).translation.grad),
            cpu_eager=float(getattr(cpu_reference,k).translation.grad),analytic=analytic_cash[k]) for k in LABELS})
    print(json.dumps(dict(stage="floating-verification-diagnostics",**diagnostics)),flush=True)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-4)
    worst = 0.
    cpu_worst = 0.
    for (name,p),(_,q),(_,c) in zip(model.named_parameters(),reference.named_parameters(),cpu_reference.named_parameters(),strict=True):
        if p.grad is not None:
            if q.grad is None:
                raise RuntimeError(f"missing eager gradient: {name}")
            torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=1e-3,msg=name)
            torch.testing.assert_close(p.grad.cpu(),c.grad,atol=3e-5,rtol=1e-3,msg=f"CPU {name}")
            worst = max(worst,float((p.grad-q.grad).abs().max()))
            cpu_worst = max(cpu_worst,float((p.grad.cpu()-c.grad).abs().max()))
    report = dict(status="pass",batch_size=batch_size,
        maximum_loss_error=float((actual-expected).detach().abs().max()),maximum_gradient_error=worst,
        cpu_maximum_loss_difference=float((actual.cpu()-cpu_expected).detach().abs().max()),
        cpu_maximum_gradient_difference=cpu_worst,
        diagnostics=diagnostics,
        recurrent_execution="three separately compiled differentiable scans; same-device eager reference; CPU diagnostic")
    model.load_state_dict(original)
    model.zero_grad(set_to_none=True)
    return report


def train_floating(*, model: FloatingTriple, sample: ObservableSample, config: FloatingTripleConfig,
                   train: TrainConfig, output: Path, monitor: AuditMonitor, stop: dict,
                   initial_benchmark: np.ndarray | None, continuation: dict | None = None,
                   regularizer: ActionRegularizerConfig | None = None,
                   warm_start: dict | None = None,
                   allow_selected_cost_warm_start: bool = False) -> dict:
    if regularizer is not None and (config.policy.trade_l1 or config.policy.trade_l2):
        raise ValueError("this regularization experiment requires zero transaction costs")
    if continuation is not None and warm_start is not None:
        raise ValueError("choose cost continuation or selected-weight regularizer warm start")
    checkpoint = output/"floating.checkpoint.pt"
    device = sample.arrays[0].device
    on_cuda = device.type == "cuda"
    optimizer = torch.optim.Adam([dict(params=list(getattr(model,k).parameters()),name=k) for k in LABELS],lr=train.learning_rate)
    start, offset, order = 0, 0, None
    functions = model.policy_functions(compile_model=train.compile_model)

    score_metrics = None

    def execute(prices,changes,returns,payoff, *, diagnostics=False):
        results = tuple(f(prices,changes,returns=returns) for f in functions)
        raw = model.losses(results,payoff)
        if regularizer is None:
            return raw
        values = regularized_losses(raw,results,regularizer)
        return (values,action_metrics(raw,results,regularizer)) if diagnostics else values

    def score():
        nonlocal score_metrics
        total = torch.zeros(3,device=device,dtype=torch.float64)
        metrics = torch.zeros((3,5),device=device,dtype=torch.float64) if regularizer is not None else None
        with torch.no_grad():
            for begin in range(0,sample.size,train.batch_size):
                p,r,h,m = sample.batch(slice(begin,begin+train.batch_size))
                if regularizer is None:
                    total += execute(p,m,r,h).double()*len(p)
                else:
                    values,details = execute(p,m,r,h,diagnostics=True)
                    total += values.double()*len(p)
                    metrics[:,:4] += details[:,:4].double()*len(p)
                    metrics[:,4] = torch.maximum(metrics[:,4],details[:,4])
        if metrics is not None:
            metrics[:,:4] /= sample.size
            score_metrics = metrics.cpu().numpy()
        values = (total/sample.size).cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite centered floating OCE; prior checkpoint remains recovery point")
        return values

    def snapshot_policy(label):
        return {k:v.detach().cpu().clone() for k,v in getattr(model,label).state_dict().items()}

    def record(epoch,values):
        row = dict(epoch=epoch,**{f"{k}_loss":float(v) for k,v in zip(LABELS,values,strict=True)},
                   sum_loss=float(values.sum()),learning_rate=train.learning_rate)
        if regularizer is not None:
            for label,detail in zip(LABELS,score_metrics,strict=True):
                row.update({f"{label}_{k}":float(v) for k,v in zip(
                    ("raw_oce","regularizer","mean_sum_abs_trades","mean_sum_squared_trades","max_abs_trade"),detail,strict=True)})
        return row

    if checkpoint.exists():
        saved = load_floating_checkpoint(checkpoint,config=config,regularizer=regularizer)
        if saved["training_paths"] != sample.size:
            raise ValueError("floating checkpoint training population differs")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = train.learning_rate
        start,offset = saved["resume_epoch"],saved["resume_offset"]
        order = saved["pair_order"].to(device) if saved["pair_order"] is not None else None
        best,best_states,selected,history = saved["best_losses"],saved["best_states"],saved["selected_epochs"],saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if on_cuda:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif warm_start is not None:
        selected_cost = (allow_selected_cost_warm_start
            and getattr(config.policy,"transaction_cost_rate",0)>0 and regularizer is None)
        if (regularizer is None and not selected_cost) or warm_start["training_paths"] != sample.size:
            raise ValueError("selected-weight objective warm start requires explicit authorization and the same training population")
        model.load_state_dict(warm_start["state_dict"])
        for label in LABELS:
            getattr(model,label).load_state_dict(warm_start["best_states"][label])
        # A changed objective and independently selected policy epochs have no
        # matching joint Adam moments. Keep learned weights/cash, use fresh Adam.
        torch.set_rng_state(warm_start["torch_rng"])
        if on_cuda:
            torch.cuda.set_rng_state_all(warm_start["cuda_rng"])
        initial = score()
        best = initial.copy()
        best_states = {k:snapshot_policy(k) for k in LABELS}
        selected = {k:0 for k in LABELS}
        history = [record(0,initial)]
    elif continuation is not None:
        if continuation["training_paths"] != sample.size:
            raise ValueError("cost continuation requires the same training population")
        model.load_state_dict(continuation["state_dict"])
        optimizer.load_state_dict(continuation["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = train.learning_rate
        offset = continuation["resume_offset"]
        order = continuation["pair_order"].to(device) if continuation["pair_order"] is not None else None
        torch.set_rng_state(continuation["torch_rng"])
        if on_cuda:
            torch.cuda.set_rng_state_all(continuation["cuda_rng"])
        # Keep the CURRENT optimizer trajectory. Old best weights are rescored
        # diagnostically, not silently substituted under unrelated Adam moments.
        initial = score()
        current_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        for label in LABELS:
            getattr(model,label).load_state_dict(continuation["best_states"][label])
        old_best_rescored = score()
        model.load_state_dict(current_state)
        torch.set_rng_state(continuation["torch_rng"])
        if on_cuda:
            torch.cuda.set_rng_state_all(continuation["cuda_rng"])
        atomic_json(output/"objective-change-rescore.json",dict(
            current_zero_cost_losses=dict(zip(LABELS,map(float,initial),strict=True)),
            old_best_zero_cost_losses=dict(zip(LABELS,map(float,old_best_rescored),strict=True)),
            scalar_zero_cost_losses=dict(zip(LABELS,map(float,initial_benchmark),strict=True)),
            old_selected_epochs=continuation["selected_epochs"],
            initialization="latest current weights, translations and optimizer; old best retained in source",
            history="new objective; zero-cost phase epoch 0; no old penalized losses copied"))
        best = initial.copy()
        best_states = {k:snapshot_policy(k) for k in LABELS}
        selected = {k:0 for k in LABELS}
        history = [record(0,initial)]
    else:
        if initial_benchmark is None:
            raise ValueError("cold starts require the scalar reconstruction benchmark")
        initial = score()
        np.testing.assert_allclose(initial,initial_benchmark,atol=3e-6,rtol=1e-4,
            err_msg="production policy must reproduce its contained scalar benchmark before training")
        best = initial.copy()
        best_states = {k:snapshot_policy(k) for k in LABELS}
        selected = {k:0 for k in LABELS}
        history = [record(0,initial)]
    write_history(output,history)
    started = time.monotonic()

    def save(resume_epoch,resume_offset,pair_order):
        payload = dict(model_code_version=type(config).version.unique_id48,model_config=asdict(config),
            user_version=config.user_version,cost_treatment="inside_utility",train_config=asdict(train),
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            best_states=best_states,best_losses=best,selected_epochs=selected,history=history,
            training_paths=sample.size,resume_epoch=resume_epoch,resume_offset=resume_offset,
            pair_order=pair_order.cpu() if pair_order is not None else None,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if on_cuda else [],scheduler=None,scaler=None)
        if regularizer is not None:
            payload.update(action_regularizer=asdict(regularizer),
                           regularizer_code_version=ActionRegularizerConfig.version.unique_id48)
        temporary = checkpoint.with_suffix(".tmp.pt")
        torch.save(payload,temporary)
        temporary.replace(checkpoint)

    print(json.dumps(dict(stage="floating-initial",**history[-1],selected_epochs=selected)),flush=True)
    if not checkpoint.exists():
        save(start,offset,order)
    for epoch in range(start,train.epochs):
        if offset == 0:
            order = torch.randperm(sample.size//2,device=device)
        indices = (2*order[:,None]+torch.arange(2,device=device)).reshape(-1)
        bad_norm = torch.zeros((),device=device,dtype=torch.bool)
        monitor.stage("train-centered-floating-oce",epoch*sample.size+offset,train.epochs*sample.size)
        for begin in range(offset,sample.size,train.batch_size):
            # Only the 60-second monitor writes progress while minibatches run;
            # do not add a filesystem write, log line or GPU transfer per batch.
            completed = epoch*sample.size+begin
            progressed = completed-start*sample.size
            elapsed = time.monotonic()-started
            monitor.record.update(completed=completed,estimated_remaining_seconds=
                elapsed*(train.epochs*sample.size-completed)/progressed if progressed > 0 else None)
            p,r,h,m = sample.batch(indices[begin:begin+train.batch_size])
            optimizer.zero_grad(set_to_none=True)
            losses = execute(p,m,r,h)
            losses.sum().backward()
            for label in LABELS:
                norm = torch.nn.utils.clip_grad_norm_(getattr(model,label).parameters(),train.gradient_clip)
                bad_norm |= ~torch.isfinite(norm)
            optimizer.step()
            if stop["requested"]:
                if bool(bad_norm):
                    raise FloatingPointError("nonfinite gradient at stop: only the prior checkpoint is trustworthy")
                save(epoch,min(begin+len(p),sample.size),order)
                print(json.dumps(dict(stage="checkpointed-stop",epoch=epoch,completed_in_epoch=begin+len(p))),flush=True)
                return dict(status="stopped",selected_epochs=selected)
        if bool(bad_norm):
            raise FloatingPointError("nonfinite gradient; prior atomic checkpoint is trustworthy")
        offset = 0
        monitor.stage("score-centered-floating-oce",epoch+1,train.epochs)
        values = score()
        for i,label in enumerate(LABELS):
            if values[i] < best[i]:
                best[i],best_states[label],selected[label] = values[i],snapshot_policy(label),epoch+1
        history.append(record(epoch+1,values))
        save(epoch+1,0,None)
        write_history(output,history)
        elapsed = time.monotonic()-started
        print(json.dumps(dict(stage="floating-epoch",**history[-1],selected_epochs=selected,
            best_losses={k:float(v) for k,v in zip(LABELS,best,strict=True)},completed=epoch+1,total=train.epochs,
            elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(train.epochs-epoch-1)/(epoch+1-start))),flush=True)
    for label in LABELS:
        getattr(model,label).load_state_dict(best_states[label])
    return dict(status="complete",selected_epochs=selected,best_losses={k:float(v) for k,v in zip(LABELS,best,strict=True)},
                initial=history[0],latest=history[-1],selection=(
                    "independent training-best regularized objective; disjoint weights" if regularizer is not None else
                    "independent training-best policy; disjoint weights"))


def summarize(model: FloatingTriple,sample: ObservableSample,*,payoff: np.ndarray,batch_size: int) -> dict:
    functions = model.policy_functions(compile_model=True)
    gains = torch.empty((3,sample.size),device="cuda")
    costs = torch.empty_like(gains)
    with torch.no_grad():
        for begin in range(0,sample.size,batch_size):
            p,r,_,m = sample.batch(slice(begin,begin+batch_size))
            for i,f in enumerate(functions):
                out = f(p,m,returns=r)
                gains[i,begin:begin+len(p)] = out.gain
                costs[i,begin:begin+len(p)] = out.penalty
    gains,costs = gains.cpu().numpy().astype(np.float64),costs.cpu().numpy().astype(np.float64)
    stats,utilities = {},{}
    centered = payoff-float(model.payoff_mean)
    for i,(label,sign) in enumerate(zip(LABELS,SIGNS,strict=True)):
        policy = getattr(model,label)
        economic = gains[i]+sign*centered
        net = economic-costs[i]
        cash = float(policy.translation.detach())
        utilities[label] = _economic_utility(net,cash,100)
        profiled = optimize_translation(net,100)
        stats[label] = dict(trained_net_oce=pair_summary(utilities[label]),
            profiled_net_oce=pair_summary(_economic_utility(net,profiled,100)),
            economic_oce=pair_summary(_economic_utility(economic,optimize_translation(economic,100),100)),
            mean_trade_penalty=pair_summary(costs[i]),mean_trading_gain=pair_summary(gains[i]),
            day0=policy.day0.detach().cpu().tolist(),
            day10=policy.day10.detach().cpu().tolist() if hasattr(policy,"day10") else None)
    return dict(status="complete",statistics=stats,
        centered_bid=pair_summary(utilities["long"]-utilities["empty"]),
        centered_ask=pair_summary(utilities["empty"]-utilities["short"]),
        bid=pair_summary(utilities["long"]-utilities["empty"]+float(model.payoff_mean)),
        ask=pair_summary(utilities["empty"]-utilities["short"]+float(model.payoff_mean)),
        interpretation="TRAINING ONLY, conditional on fitted means and selected policy; not a Q no-arbitrage proof")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=500)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--trade-l1",type=float,default=1e-4)
    parser.add_argument("--trade-l2",type=float,default=1e-8)
    parser.add_argument("--continue-from",type=Path,help="explicit cost-only objective change from current checkpoint")
    parser.add_argument("--without-a10",action="store_true",help="new rich-clock decider; no explicit day-10 vector")
    parser.add_argument("--clock-with-a10",action="store_true",help="matched rich-clock decider WITH explicit a10")
    parser.add_argument("--centering-from",type=Path,help="frozen Q training-centering.npz; REQUIRED for P, never fit P means")
    args = parser.parse_args()
    output,root = args.output.resolve(),Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a unique external run directory")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("detached imports did not discover the external project data root")
    checks = resource_health()
    if any(c["health"] in {"STOP","UNAVAILABLE","ERROR"} for c in checks):
        raise RuntimeError("resource manager blocks launch")
    if any(c["health"] != "OK" for c in checks):
        print(json.dumps(dict(stage="resource-warning",checks=checks)),flush=True)
    print(json.dumps(dict(stage="environment-validated",interpreter=os.sys.executable,cwd=str(Path.cwd()),data_root=str(root))),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    stop = {"requested":False}
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True))
    try:
        with FileLock(output/"training.lock",timeout=0):
            torch.set_num_threads(4)
            torch.manual_seed(20261105)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            if args.without_a10 and args.clock_with_a10:
                raise ValueError("choose either no-a10 or clock-with-a10")
            config_class = (TimeDeciderWithA10Config if args.clock_with_a10 else
                            TimeDeciderTripleConfig if args.without_a10 else FloatingTripleConfig)
            model_class = (TimeDeciderWithA10Triple if args.clock_with_a10 else
                           TimeDeciderTriple if args.without_a10 else FloatingTriple)
            config = config_class(label="centered-floating-no-a10" if args.without_a10 else "centered-floating-empty-long-short")
            config = replace(config,policy=replace(config.policy,trade_l1=args.trade_l1,trade_l2=args.trade_l2))
            continuation = load_cost_continuation(args.continue_from,config=config) if args.continue_from else None
            train = TrainConfig(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
                                patience=0,compile_model=True)
            scenario = json.loads((args.scenarios/"identity.json").read_text())
            measure = scenario.get("measure","Q")
            if measure not in {"P","Q"} or (measure == "P" and args.centering_from is None):
                raise ValueError("P scenarios require frozen Q centering; do NOT subtract P risk premia")
            if scenario["substeps"] != 1 or scenario["paths_per_sample"] != 100_000:
                raise ValueError("use the original 100000-path one-step/day sample")
            if sha256(Path(scenario["calibration"])/"diffusion.npz") != scenario["calibration_sha256"]:
                raise ValueError("original model calibration changed")
            monitor.stage("convert-existing-training-contract-gains")
            arrays = read_floating_sample(args.scenarios/"train",paths=100_000)
            prices,returns,payoff = arrays
            means = fit_return_means(returns)
            normalized = prices[:, :20]/prices[:, :20, :1]
            normalized[:, :, 0] = prices[:, :20, 0]
            mean,scale = normalized.mean((0,1)),np.maximum(normalized.std((0,1)),1e-4)
            payoff_mean = float(payoff.mean())
            if args.centering_from is not None:
                source_identity = json.loads((args.centering_from.parent/"identity.json").read_text())
                source_scenario = json.loads((Path(source_identity["scenarios"])/"identity.json").read_text())
                if (source_scenario.get("measure","Q") != "Q"
                        or source_scenario["calibration_sha256"] != scenario["calibration_sha256"]
                        or source_scenario["substeps"] != scenario["substeps"]):
                    raise ValueError("centering reference must be Q with the same calibration and time step")
                with np.load(args.centering_from,allow_pickle=False) as saved:
                    means,mean,scale = (saved[k].copy() for k in ("return_mean","price_mean","price_scale"))
                    payoff_mean = float(saved["payoff_mean"])
            benchmark = scalar_benchmarks(prices=prices,returns=returns,payoff=payoff,means=means,config=config,payoff_mean=payoff_mean)
            atomic_json(output/"scalar-benchmarks.json",benchmark)
            np.savez(output/"training-centering.npz",return_mean=means,payoff_mean=payoff_mean,price_mean=mean,price_scale=scale)
            provenance = dict(objective="oce",policy_kind=("centered floating contracts; rich-clock LSTM + a0; NO a10" if args.without_a10
                else "centered floating contracts; generic LSTM + deterministic a0/a10"),
                cost_treatment="inside_utility",epochs=args.epochs,model_config=asdict(config),train_config=asdict(train),
                checkpoint_selection="independent_policy_best",
                measure=measure, centering_reference=str(args.centering_from.resolve()) if args.centering_from else None,
                centering_reference_sha256=sha256(args.centering_from) if args.centering_from else None,
                p_interpretation="P expectation with unchanged Q option prices and frozen Q corrections; no Q-bracketing guarantee" if measure=="P" else None,
                compiler_options={"eager_numerics.division_rounding":True},
                model_code_version=type(config).version.unique_id48,scenarios=str(args.scenarios.resolve()),
                scenario_identity_sha256=sha256(args.scenarios/"identity.json"),cache=str(args.cache.resolve()),
                return_definition="(close-open)/S_open minus frozen date/slot training mean; same convention for options and spot",
                claim="s*(H-frozen_training_mean(H))",initialization=(
                    "current saved policy/Adam/RNG/minibatch; cost-only objective change" if continuation is not None else
                    "cold start; zero generic action and a0; unhedged optimal cash; NO a10" if args.without_a10 else
                    "zero generic action; scalar-optimal constant a10; zero a0/empty"),
                objective_formula="SUM_s[-c_s-mean(u_100(G_s+s*(H-mean_H)-C_s-c_s))]",
                regularizer=f"{args.trade_l1:g}*sum(abs(floating quantities))+{args.trade_l2:g}*sum(quantities**2), inside utility",
                paths=100_000,simulation_substeps_per_day=1,validation_used=False,
                q_warning="Marginal centering does not imply conditional drift zero; accepted one-step Q warning remains",
                code_sha256={str(p):sha256(p) for p in [
                    *(Path(__file__).with_name(n) for n in ("floating_run.py","floating.py","time_decider.py","contracts.py",
                        "feedback.py","hedge.py","observable_data.py","cache_load.py","persistence.py","triple_plot.py","training.py","allocation.py")),
                    Path(__file__).parents[2]/"torch"/"utility.py"]})
            if continuation is not None:
                source_identity = json.loads((args.continue_from.parent/"identity.json").read_text())
                if source_identity["scenario_identity_sha256"] != provenance["scenario_identity_sha256"]:
                    raise ValueError("cost continuation cannot change the underlying scenarios")
                provenance["continuation"] = dict(source=str(args.continue_from.resolve()),
                    source_sha256=sha256(args.continue_from),source_resume_epoch=continuation["resume_epoch"],
                    source_resume_offset=continuation["resume_offset"],optimizer_preserved=True,
                    original_total_epoch_target=continuation["resume_epoch"]+args.epochs,
                    old_costs={k:continuation["model_config"]["policy"][k] for k in ("trade_l1","trade_l2")},
                    new_costs=dict(trade_l1=args.trade_l1,trade_l2=args.trade_l2))
            identity = output/"identity.json"
            if identity.exists():
                previous = json.loads(identity.read_text())
                operational = {"epochs","train_config"}
                if {k:v for k,v in previous.items() if k not in operational} != {
                        k:v for k,v in provenance.items() if k not in operational}:
                    raise ValueError("logical run identity changed; choose a new directory")
            atomic_json(identity,provenance)
            changes,cache_report = load_retained_market_cache(cache=args.cache,scenarios=args.scenarios,paths=100_000,monitor=monitor)
            atomic_json(output/"cache-verification.json",cache_report)
            if stop["requested"]:
                return
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA initialized before managed GPU claim")
            claim_training_gpu(output)
            monitor.stage("upload-once-floating-training-sample")
            sample = ObservableSample(arrays=arrays,changes=changes,device="cuda")
            del changes,normalized
            gc.collect()
            model = model_class(BaseEnvironment(device="cuda",precision=32),config=config,
                mean=torch.tensor(mean),scale=torch.tensor(scale),return_mean=torch.tensor(means),payoff_mean=payoff_mean)
            initial_losses = []
            with torch.no_grad():
                for label,sign in zip(LABELS,SIGNS,strict=True):
                    item = benchmark["policies"][label]["net"]
                    policy = getattr(model,label)
                    if args.without_a10:
                        claim = sign*(payoff-payoff_mean)
                        cash = optimize_translation(claim,config.policy.risk_aversion)
                        initial_losses.append(-float(_economic_utility(claim,cash,config.policy.risk_aversion).mean()))
                        policy.translation.fill_(cash)
                    else:
                        policy.day10[1],policy.day10[17] = -2*sign*item["alpha"],sign*item["alpha"]
                        policy.translation.fill_(item["cash"])
                        initial_losses.append(-item["oce"])
            if continuation is not None:
                target = model.state_dict()
                for label in LABELS:
                    for name in ("price_mean","price_scale","return_mean"):
                        key = f"{label}.{name}"
                        torch.testing.assert_close(target[key].cpu(),continuation["state_dict"][key],atol=0,rtol=0)
                torch.testing.assert_close(target["payoff_mean"].cpu(),continuation["state_dict"]["payoff_mean"],atol=0,rtol=0)
                model.load_state_dict(continuation["state_dict"])
            monitor.stage("verify-floating-gpu-values-and-gradients")
            verified = verify_compiled(model,sample,batch_size=train.batch_size)
            atomic_json(output/"gpu-verification.json",verified)
            print(json.dumps(dict(stage="floating-verification",**verified)),flush=True)
            fit = train_floating(model=model,sample=sample,config=config,train=train,output=output,monitor=monitor,stop=stop,
                initial_benchmark=np.array(initial_losses),continuation=continuation)
            if fit["status"] == "complete":
                monitor.stage("summarize-selected-floating-hedges")
                report = summarize(model,sample,payoff=payoff,batch_size=train.batch_size)
                report["fit"] = fit
                atomic_json(output/"report.json",report)
                plot_triple_history(output,output/"figures")
                monitor.stage("complete",train.epochs,train.epochs)
            atomic_json(output/"status.json",fit)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
