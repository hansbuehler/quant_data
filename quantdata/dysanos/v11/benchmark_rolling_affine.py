"""Benchmark two sequential affine-200 quote fits on rolling market windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from quantdata.dysanos.flex.config import FlexObjectiveConfig, PcaDecoderConfig
from quantdata.dysanos.flex.model import LatentStateSanosModel
from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.training import (
    _evaluation_score,
    _market_training_phase,
    evaluate_model,
    initialize_pca_decoder_,
    materialize_day,
    training_convergence_audit,
)
from quantdata.dysanos.v11.dlv_initialization import recover_initial_state
from quantdata.dysanos.v11.licensed_market import licensed_market_day, load_certified_pricer
from quantdata.dysanos.v11.run_rolling_direct_ar import (
    _fit_config,
    _geometry,
    _primary_fit_metrics,
)
from quantdata.torch import BaseEnvironment


def install_rolling_weights(model, source, old_dates, new_dates):
    """Copy shared decoder and match states by date; new states use last known state."""
    if sorted(old_dates) != list(old_dates) or sorted(new_dates) != list(new_dates):
        raise ValueError("Dates must be ordered")
    lookup = {day: i for i, day in enumerate(old_dates)}
    if len(lookup) != len(old_dates) or old_dates[-1] >= new_dates[-1]:
        raise ValueError("Warm-start dates must be unique and strictly earlier")
    current = dict(model.named_parameters())
    previous = source["parameters"]
    if set(current) != set(previous):
        raise ValueError("Warm-start parameter names differ")
    with torch.no_grad():
        for name, target in current.items():
            tensor = previous[name].to(device=target.device, dtype=target.dtype)
            if name == "daily_state_embedding.weight":
                if tensor.shape[1:] != target.shape[1:] or tensor.shape[0] != len(old_dates):
                    raise ValueError("Incompatible state geometry")
                for index, day in enumerate(new_dates):
                    if day in lookup:
                        target[index].copy_(tensor[lookup[day]])
                    elif day > old_dates[-1]:
                        target[index].copy_(tensor[-1])
                    else:
                        raise ValueError(f"Missing historical state for {day}")
            else:
                if target.shape != tensor.shape:
                    raise ValueError(f"Incompatible parameter {name}")
                target.copy_(tensor)


def run(completion_path: Path, output: Path, epochs: int, panel: Path, archive: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    completion = json.loads(completion_path.read_text())
    geometry = _geometry(_geometry_from_completion(completion, requested_model_business_dtes=None), lower=-4, upper=3)
    all_dates = tuple(sorted(date.fromisoformat(path.stem) for path in (panel / "days").glob("*.parquet")))
    last = all_dates.index(date(2025, 8, 29))
    windows = [all_dates[end - 249:end + 1] for end in (last - 5, last)]
    if any(len(window) != 250 for window in windows):
        raise ValueError("Insufficient market dates")
    env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
    objective = FlexObjectiveConfig(fit_weight=0.05)
    config = _fit_config(epochs=epochs, device="cuda:0", stage="decoder")
    previous = None
    for number, dates in enumerate(windows, start=1):
        started = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        loaded = []
        for i, day in enumerate(dates, 1):
            loaded.append(licensed_market_day(day, panel=panel, archive=archive, geometry=geometry))
            if i % 25 == 0 or i == len(dates):
                print(f"window={number}/2 loaded={i}/{len(dates)} elapsed={perf_counter() - started:.1f}s", flush=True)
        days = tuple(loaded)
        materialized = tuple(materialize_day(day, env=env) for day in days)
        data_bytes = sum(t.numel() * t.element_size() for day in materialized for t in vars(day).values() if isinstance(t, torch.Tensor))
        model = LatentStateSanosModel(env, num_days=len(days), config=PcaDecoderConfig(label="rolling-affine-benchmark", state_size=200, geometry=geometry))
        if previous is None:
            states, audits = [], []
            for i, day in enumerate(dates, 1):
                pricer, _ = load_certified_pricer(day, archive=archive)
                state, audit = recover_initial_state(pricer, geometry=geometry)
                states.append(state)
                audits.append({"date": str(day), **audit})
                if i % 25 == 0 or i == len(dates):
                    print(f"window={number}/2 recovered_dlvs={i}/{len(dates)} elapsed={perf_counter()-started:.1f}s", flush=True)
            (output / "dlv-initialization-audit.json").write_text(json.dumps(audits, indent=2))
            raw = torch.as_tensor(np.stack(states), dtype=env.fdtype, device=env.device)
            initialize_pca_decoder_(model, raw)
            del raw
        else:
            install_rolling_weights(model, previous, windows[number - 2], dates)
        initial = evaluate_model(model, materialized, env=env, objective_config=objective, day_batch_size=config.day_batch_size, equal_date_weighting=True)
        model.sanos_decoder.enable_solver_compile()
        def progress(epoch, evaluation):
            elapsed = perf_counter() - started
            print(f"window={number}/2 epoch={epoch}/{epochs} elapsed={elapsed:.1f}s remaining_est={(epochs-epoch)*elapsed/epoch:.1f}s inside={_primary_fit_metrics(evaluation)['inside_bid_ask_percent']:.4f} peak_gib={torch.cuda.max_memory_reserved()/2**30:.3f}", flush=True)
        def checkpoint(state):
            path = output / f"window-{number}-restart.pt"
            temporary = path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "training_state": state, "dates": [str(day) for day in dates], "geometry": asdict(geometry)}, temporary)
            temporary.replace(path)
        history, state = _market_training_phase(model, materialized, env=env, objective_config=objective, train_config=config, pre_training_score=_evaluation_score(initial), evaluation_callback=progress, checkpoint_callback=checkpoint, early_stop_min_epochs=100)
        model.sanos_decoder.disable_solver_compile()
        final = evaluate_model(model, materialized, env=env, objective_config=objective, day_batch_size=config.day_batch_size, equal_date_weighting=True)
        previous = {"parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()}, "dates": [str(day) for day in dates], "geometry": asdict(geometry)}
        torch.save(previous, output / f"window-{number}.pt")
        report = {"window": number, "dates": [str(dates[0]), str(dates[-1])], "warm_start_as_of": None if number == 1 else str(windows[0][-1]), "market_panel": str(panel), "surface_archive": str(archive), "surface_smoothness": 0.1, "price_target": "licensed bid/ask and midpoint", "data_tensor_bytes": data_bytes, "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "elapsed_seconds": perf_counter() - started, "initial_fit": _primary_fit_metrics(initial), "final_fit": _primary_fit_metrics(final), "convergence": training_convergence_audit(history, metric="objective", train_config=config), "history": history}
        (output / f"window-{number}.json").write_text(json.dumps(report, indent=2))
        print(f"window={number} complete elapsed={report['elapsed_seconds']:.1f}s convergence={report['convergence']}", flush=True)
        del model, materialized, days, state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    run(args.completion, args.output, args.epochs, args.panel, args.archive)
