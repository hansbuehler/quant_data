"""Checkpointed static-arbitrage simulation of the retained diffusion models."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.inference import model_from_export
from quantdata.hedging.model_audit.recursive import (
    SEED,
    _resume,
    _save_arrays,
    append_diffusion_context,
)
from quantdata.hedging.model_audit.rolling_versions import resource_health
from quantdata.hedging.model_audit.run import GRID, _hash, _json, build_history
from quantdata.hedging.model_audit.static_arb import (
    HORIZONS,
    NODES,
    TENORS,
    call_violations,
    generated_calls,
    iv_stability,
    next_tenor,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    diffusion_live_context_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    _sample_diffusion_next_states,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    interpolate_generated_implied_volatility,
)
from quantdata.torch import BaseEnvironment, torch


def simulate(output: Path, catalog: Path, model: str, day: date, history, index: int, *,
             paths: int, horizons: tuple[int, ...], batch_size: int, device: str, engine: str, stop_requested) -> dict | None:
    """Simulate fixed originating model; persist path states and exact RNG at session boundaries."""
    entries = json.loads(catalog.read_text())["models"].values()
    variant = "ad-seq-vol-ft" if model.endswith("FT") else "ad-seq-vol"
    selected = [entry for entry in entries if entry["origin_date"] == str(day) and entry["variant"] == variant]
    if len(selected) != 1 or not selected[0]["verified_inference"]:
        raise ValueError(f"No unique verified inference export for {model} {day}")
    source = selected[0]
    environment = BaseEnvironment(precision=32, device=device)
    network = model_from_export(Path(source["manifest"]), environment=environment, expected_sha256=source["manifest_sha256"])
    window = replace(history, dates=history.dates[:index + 1],
        log_implied_volatilities=history.log_implied_volatilities[:index + 1],
        implied_vol_is_interior=history.implied_vol_is_interior[:index + 1],
        forward_log_returns_into_date=history.forward_log_returns_into_date[:index + 1],
        sessions_from_previous_observation=history.sessions_from_previous_observation[:index + 1])
    initial = diffusion_live_context_from_history(window, as_of_date=day, forecast_date=history.dates[index + 1])
    contexts = np.broadcast_to(initial.sequence, (paths, *initial.sequence.shape)).copy()
    surfaces = np.full((paths, len(horizons), len(GRID.expiries_y), GRID.strike_count), np.nan, dtype=np.float32)
    alive = np.ones(paths, dtype=bool)
    if engine == "jax":
        import jax
        import jax.numpy as jnp

        from quantdata.hedging.model_audit.diffusion_jax import (
            make_denoiser,
            make_random_sampler,
            parameters_from_torch,
        )
        from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.process import (
            cosine_beta_schedule,
        )

        config = network.config
        weights = parameters_from_torch(network)
        denoiser = make_denoiser(layers=config.transformer_layers, heads=config.attention_heads,
            dimension=config.model_dimension, adapter=model.endswith("FT"))
        sampler = make_random_sampler(denoiser, diffusion_steps=config.diffusion_steps)
        betas = jnp.asarray(cosine_beta_schedule(config.diffusion_steps, dtype=torch.float32, device=network.device).cpu().numpy())
        random = jax.random.PRNGKey(SEED)
    else:
        random = torch.Generator(device=device).manual_seed(SEED)
    identity = {"schema": 1, "model": model, "date": str(day), "paths": paths, "horizons": horizons,
        "batch_size": batch_size, "device": device, "engine": engine, "seed": SEED, "manifest_sha256": source["manifest_sha256"]}
    identity = json.loads(json.dumps(identity))
    checkpoint = output / "checkpoints" / model / f"{day}.npz"
    resumed = _resume(checkpoint, identity)
    begin = int(resumed["completed"]) if resumed else 0
    if resumed:
        contexts, surfaces, alive = resumed["contexts"], resumed["surfaces"], resumed["alive"]
        if engine == "jax":
            random = jnp.asarray(resumed["rng_state"])
        else:
            random.set_state(torch.from_numpy(resumed["rng_state"]))
    training = np.flatnonzero((history.sessions_from_previous_observation == 1) & (np.arange(len(history.dates)) <= index))[-252:]
    if len(training) != 252:
        raise ValueError("Missing causal training reference")

    def score(horizon: int, surface: np.ndarray) -> list[dict]:
        results = []
        for tenor in TENORS:
            row = GRID.expiries_y.index(tenor / 252.)
            atm = np.exp(history.log_implied_volatilities[index, row, GRID.nms_nodes.index(0.)])
            strikes = NMS.from_nms(NODES, tenor / 252., vol=atm)
            following = next_tenor(tenor, np.asarray(GRID.expiries_y) * 252.)
            calls = generated_calls(surface, strikes, tenor)
            longer = None if following is None else generated_calls(surface, strikes, following)
            historical = np.stack([interpolate_generated_implied_volatility(history.log_implied_volatilities[training],
                query_expiry_y=tenor / 252., query_strikes_over_forward=np.full(len(training), strike), grid=GRID)
                for strike in strikes], axis=1)
            results.append({"simulation_day": horizon, "business_dte": tenor, "next_business_dte": following,
                **call_violations(strikes, calls, longer_prices=longer, zero_strike=np.ones(paths)),
                **iv_stability(calls, strikes, tenor, historical)})
        return results

    started = time.monotonic()
    last_health = 0.
    for position in range(begin, max(horizons)):
        for start in range(0, paths, batch_size):
            if time.monotonic() - last_health >= 240.:
                health = resource_health()
                if any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in health):
                    raise InterruptedError("Managed resource stop; retained session checkpoint remains the recovery point")
                if "WARN" in health:
                    print("WARN: resource headroom or gpuctl is unavailable; review owned storage and processes", flush=True)
                last_health = time.monotonic()
            indices = np.arange(start, min(start + batch_size, paths))
            indices = indices[alive[indices]]
            if len(indices):
                if engine == "jax":
                    generated, random = sampler(weights, jnp.asarray(contexts[indices], dtype=jnp.float32), random, betas)
                    generated = np.asarray(generated)
                else:
                    generated = _sample_diffusion_next_states(SimpleNamespace(ema_model=network),
                        observed_sequences=contexts[indices], generator=random)
                valid = np.isfinite(generated).all(axis=(1, 2, 3))
                alive[indices[~valid]] = False
                if valid.any():
                    active = indices[valid]
                    contexts[active] = append_diffusion_context(contexts[active], generated[valid])
                    if position + 1 in horizons:
                        surfaces[active, horizons.index(position + 1)] = generated[valid, 0]
            elapsed = time.monotonic() - started
            finished = (position - begin) * paths + min(start + batch_size, paths)
            remaining = (max(horizons) - begin) * paths - finished
            print(json.dumps({"model": model, "origin": str(day), "session": position + 1,
                "total_sessions": max(horizons), "paths_completed_in_session": min(start + batch_size, paths),
                "paths": paths, "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(elapsed * remaining / finished, 1), "alive_paths": int(alive.sum())}), flush=True)
        _save_arrays(checkpoint, identity_json=json.dumps(identity), completed=position + 1,
            contexts=contexts, surfaces=surfaces, alive=alive,
            rng_state=np.asarray(random) if engine == "jax" else random.get_state().cpu().numpy())
        if position + 1 in horizons:
            partial = output / "static/partial" / model.replace(" ", "_") / str(day) / f"day-{position + 1}.json"
            partial.parent.mkdir(parents=True, exist_ok=True)
            _json(partial, {"model": model, "origin_date": str(day), "results": score(position + 1, surfaces[:, horizons.index(position + 1)])})
        if stop_requested():
            return None
    results = [row for horizon_index, horizon in enumerate(horizons) for row in score(horizon, surfaces[:, horizon_index])]
    return {"model": model, "origin_date": str(day), "paths": paths, "horizons": horizons,
        "seed": SEED, "status": "scored", "results": results, "inference_manifest": source["manifest"],
        "inference_manifest_sha256": source["manifest_sha256"], "batch_size": batch_size, "device": device, "engine": engine}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("AD-Seq-Vol", "AD-Seq-Vol-FT"), required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--engine", choices=("torch", "jax"), default="torch")
    args = parser.parse_args()
    horizons = tuple(args.horizons)
    if args.paths < 1 or args.batch_size < 1 or not horizons or tuple(sorted(set(horizons))) != horizons or horizons[0] < 1:
        raise ValueError("Invalid simulation size or horizons")
    args.output.mkdir(parents=True, exist_ok=True)
    panel = PublicHedgingPanel(args.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, args.output)
    dates = json.loads((args.reference / "observable_reference.json").read_text())["dates"]
    selected = [date.fromisoformat(text) for text in dates if args.dates is None or date.fromisoformat(text) in args.dates]
    if args.dates and set(selected) != set(args.dates):
        raise ValueError("Requested dates do not match reference origins")
    catalog = args.reference / "INFERENCE_MODELS.json"
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print(f"signal {signum}: saving at session boundary", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    protocol = {"schema": "diffusion-static-v1", "models": args.models,
        "dates": [str(day) for day in selected], "paths": args.paths, "horizons": horizons,
        "batch_size": args.batch_size, "device": args.device, "engine": args.engine, "seed": SEED,
        "catalog_sha256": _hash(catalog), "source_sha256": _hash(Path(__file__)),
        "jax_code_sha256": _hash(Path(__file__).with_name("diffusion_jax.py")) if args.engine == "jax" else None}
    protocol_path = args.output / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Diffusion static run identity changed; choose a new output directory")
    _json(protocol_path, protocol)
    for model in args.models:
        for day in selected:
            destination = args.output / "static/daily" / model.replace(" ", "_") / f"{day}.json"
            if destination.exists():
                continue
            result = simulate(args.output, catalog, model, day, history, history.dates.index(day),
                paths=args.paths, horizons=horizons, batch_size=args.batch_size, device=args.device,
                engine=args.engine, stop_requested=lambda: stop)
            if result is None:
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            _json(destination, result)


if __name__ == "__main__":
    main()
