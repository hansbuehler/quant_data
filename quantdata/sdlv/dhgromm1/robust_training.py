"""Restartable global-tail OCE SGD; no minibatch trimming or additional penalties."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from quantdata.sdlv.dhgromm1.floating import LABELS
from quantdata.sdlv.dhgromm1.persistence import write_history
from quantdata.sdlv.dhgromm1.robust import RobustTriple, RobustTripleConfig, exact_population_scores
from quantdata.sdlv.dhgromm1.training import TrainConfig


def load_robust_checkpoint(path: Path, *, config: RobustTripleConfig) -> dict:
    """Do not reinterpret the older ordinary-OCE or regularized checkpoints."""
    saved = torch.load(path,map_location="cpu",weights_only=False)
    if (saved.get("schema") != "dhgromm1-global-tail-oce-v1"
            or saved.get("model_code_version") != RobustTripleConfig.version.unique_id48
            or saved.get("model_config") != asdict(config) or saved.get("user_version") != config.user_version):
        raise ValueError("robust checkpoint architecture/objective version differs")
    return saved


def collect_gains(model: RobustTriple, sample, *, functions: tuple, batch_size: int, monitor=None) -> torch.Tensor:
    """GPU-resident population outputs, not population autograd graphs."""
    gains = torch.empty((3,sample.size),device=sample.arrays[0].device,dtype=sample.arrays[0].dtype)
    with torch.no_grad():
        for begin in range(0,sample.size,batch_size):
            prices,returns,_,changes = sample.batch(slice(begin,begin+batch_size))
            for i,function in enumerate(functions):
                gains[i,begin:begin+len(prices)] = function(prices,changes,returns=returns).gain
            if monitor is not None:
                monitor.record.update(completed=begin+len(prices),total=sample.size)
    if not bool(torch.isfinite(gains).all()):
        raise FloatingPointError("nonfinite robust hedge gains; prior checkpoint remains recovery point")
    return gains


def train_robust(*, model: RobustTriple, sample, train: TrainConfig, output: Path, monitor, stop: dict) -> dict:
    """Minibatches estimate a shared variational objective; scoring uses all N.

    Antithetic partners are shuffled together but the tail consists of
    individual paths, never pair averages. No additional penalty, weight decay,
    action cap, new path simulation or market-state reconstruction is applied.
    """
    config,device = model.config,sample.arrays[0].device
    on_cuda = device.type=="cuda"
    checkpoint = output/"robust.checkpoint.pt"
    optimizer = torch.optim.Adam([dict(params=list(getattr(model,k).parameters()),name=k) for k in LABELS],
                                 lr=train.learning_rate,weight_decay=0.)
    functions = model.policy_functions(compile_model=train.compile_model)
    start,offset,order = 0,0,None

    def snapshot(label):
        return {k:v.detach().cpu().clone() for k,v in getattr(model,label).state_dict().items()}

    def score(epoch):
        monitor.stage("score-full-population-bottom90",0,sample.size)
        gains = collect_gains(model,sample,functions=functions,batch_size=train.batch_size,monitor=monitor)
        with torch.no_grad():
            details = exact_population_scores(model,gains,sample.arrays[2])
        row = dict(epoch=epoch,learning_rate=train.learning_rate)
        for label in LABELS:
            row[f"{label}_loss"] = -details[label]["robust_oce"]
            row.update({f"{label}_{k}":v for k,v in details[label].items()})
        row["sum_loss"] = sum(row[f"{k}_loss"] for k in LABELS)
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError("nonfinite global-tail score; prior checkpoint remains recovery point")
        return row

    if checkpoint.exists():
        saved = load_robust_checkpoint(checkpoint,config=config)
        if saved["training_paths"] != sample.size:
            raise ValueError("robust restart population changed")
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
    else:
        initial = score(0)
        history = [initial]
        best = {k:initial[f"{k}_loss"] for k in LABELS}
        best_states,selected = {k:snapshot(k) for k in LABELS},dict.fromkeys(LABELS,0)

    def save(epoch,completed,pair_order):
        payload = dict(schema="dhgromm1-global-tail-oce-v1",model_code_version=RobustTripleConfig.version.unique_id48,
            model_config=asdict(config),user_version=config.user_version,train_config=asdict(train),
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            best_losses=best,best_states=best_states,selected_epochs=selected,history=history,training_paths=sample.size,
            resume_epoch=epoch,resume_offset=completed,pair_order=pair_order.cpu() if pair_order is not None else None,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if on_cuda else [],
            scheduler=None,scaler=None,transaction_costs=0,action_regularization=0)
        temporary = checkpoint.with_suffix(".tmp.pt")
        torch.save(payload,temporary)
        temporary.replace(checkpoint)

    write_history(output,history)
    if not checkpoint.exists():
        save(start,offset,order)
    print(json.dumps(dict(stage="robust-initial",**history[-1],selected_epochs=selected)),flush=True)
    started = time.monotonic()
    for epoch in range(start,train.epochs):
        if offset==0:
            order = torch.randperm(sample.size//2,device=device)
        indices = (2*order[:,None]+torch.arange(2,device=device)).reshape(-1)
        finite = torch.ones((),device=device,dtype=torch.bool)
        monitor.stage("train-global-tail-variational-oce",epoch*sample.size+offset,train.epochs*sample.size)
        for begin in range(offset,sample.size,train.batch_size):
            done = epoch*sample.size+begin
            elapsed = time.monotonic()-started
            completed = done-start*sample.size
            monitor.record.update(completed=done,estimated_remaining_seconds=
                elapsed*(train.epochs*sample.size-done)/completed if completed>0 else None)
            prices,returns,payoff,changes = sample.batch(indices[begin:begin+train.batch_size])
            optimizer.zero_grad(set_to_none=True)
            results = tuple(f(prices,changes,returns=returns) for f in functions)
            losses = model.losses(results,payoff)
            losses.sum().backward()
            for label in LABELS:
                norm = torch.nn.utils.clip_grad_norm_(getattr(model,label).parameters(),train.gradient_clip)
                finite &= torch.isfinite(norm)
            optimizer.step()
            if stop["requested"]:
                if not bool(finite):
                    raise FloatingPointError("nonfinite gradient at stop; only prior checkpoint is trustworthy")
                save(epoch,min(begin+len(prices),sample.size),order)
                print(json.dumps(dict(stage="robust-checkpointed-stop",epoch=epoch,completed_in_epoch=begin+len(prices))),flush=True)
                return dict(status="stopped",selected_epochs=selected)
        if not bool(finite):
            raise FloatingPointError("nonfinite robust gradient; prior checkpoint remains trustworthy")
        offset = 0
        row = score(epoch+1)
        for label in LABELS:
            value = row[f"{label}_loss"]
            if value<best[label]:
                best[label],best_states[label],selected[label] = value,snapshot(label),epoch+1
        history.append(row)
        save(epoch+1,0,None)
        write_history(output,history)
        elapsed = time.monotonic()-started
        print(json.dumps(dict(stage="robust-epoch",**row,selected_epochs=selected,completed=epoch+1,total=train.epochs,
            elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(train.epochs-epoch-1)/(epoch+1-start))),flush=True)
    for label in LABELS:
        getattr(model,label).load_state_dict(best_states[label])
    return dict(status="complete",selected_epochs=selected,best_losses=best,initial=history[0],latest=history[-1],
        selection="independent policy best exact FULL-POPULATION bottom-q utility at trained cash; never batch top-k")
