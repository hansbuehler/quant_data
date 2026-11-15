"""Causal rolling GROMM2 using the user's standalone repository, with restart state."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedging.model_audit.gromm_paper_objective import PAPER_OBJECTIVE
from quantdata.hedging.model_audit.legacy_queue import health_check, run_owned, wait_gpu
from quantdata.hedging.model_audit.legacy_rolling import Progress
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import _hash, _json


def checkpointed_fit(directory: Path, original):
    """Retain the repository's exact Adam step, saving moments and iteration too."""
    import jax.numpy as jnp
    from openqdata.models.gromm import calibrate as calibration

    counter = 0

    def fit(batch, epochs=1000, lr=.05, x0=None, log_every=50, log=print, prior=None, prior_weight=0.):
        nonlocal counter
        checkpoint = directory / f"chunks/{counter:04d}.npz"
        counter += 1
        shape = (len(batch.views), len(batch.geom.config.fit_blocks), batch.geom.strikes.shape[1])
        initial = np.broadcast_to(calibration.initial_x(batch.geom) if x0 is None else x0, shape).copy()
        identity = {"epochs": epochs, "learning_rate": lr, "prior_weight": prior_weight,
            "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "shape": list(shape), "objective": PAPER_OBJECTIVE}
        reference = jnp.asarray(initial if prior is None else np.broadcast_to(prior, shape))
        weight = jnp.asarray(prior_weight if prior is not None else 0.)
        args = calibration._select(batch, np.arange(len(batch.views)))
        state = (jnp.asarray(initial), jnp.zeros_like(reference), jnp.zeros_like(reference))
        history = {"epoch": [], "loss": [], "coverage": []}
        start = 0
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if json.loads(str(saved["identity"])) != identity:
                    raise ValueError(f"GROMM2 same-origin restart identity changed: {checkpoint}")
                state = tuple(jnp.asarray(saved[name]) for name in ("x", "adam_m", "adam_v"))
                start = int(saved["completed_epochs"])
                history = {name: list(saved[f"history_{name}"]) for name in history}
            log(f"restore {checkpoint.name} epoch={start}/{epochs}")
        started = time.monotonic()
        for epoch in range(start, epochs):
            state, losses = calibration._adam_step(state, jnp.asarray(epoch), jnp.asarray(epochs),
                jnp.asarray(lr), reference, weight, *args)
            if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
                coverage = calibration.coverage(batch, np.asarray(state[0]))
                history["epoch"].append(epoch + 1)
                history["loss"].append(np.asarray(losses))
                history["coverage"].append(coverage)
                elapsed = time.monotonic() - started
                log(f"chunk={counter} epoch={epoch+1}/{epochs} loss={float(jnp.mean(losses)):.6g} "
                    f"coverage={coverage.mean():.4f} elapsed={elapsed:.1f}s eta={elapsed*(epochs-epoch-1)/(epoch+1-start):.1f}s")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez(temporary, identity=np.asarray(json.dumps(identity)), completed_epochs=np.asarray(epoch + 1),
                    **dict(zip(("x", "adam_m", "adam_v"), map(np.asarray, state), strict=True)),
                    **{f"history_{name}": np.asarray(values) for name, values in history.items()})
                temporary.replace(checkpoint)
                health_check()
        return np.asarray(state[0]), history

    return fit


def prepare(run: Path, reference: Path) -> None:
    from openqdata.models.gromm import config, data

    selected = [date.fromisoformat(value) for value in origins(reference)]
    available = data.stored_days(data.sanos_run())
    required = sorted({day for origin in selected for day in data.history_window(available, origin, config.GROMM2.history) + [origin]})
    if not set(selected) <= set(available):
        raise ValueError("The approved 44 origins are not all present in the certified SANOS source")
    protocol = {"model": "repository GROMM2", "source_repository": "/home/hans/python/openqdata_models_gromm",
        "source_snapshot": str(Path(config.__file__).parents[3]), "configuration": asdict(config.GROMM2),
        "origins": list(map(str, selected)), "initialization": "independent origins; previous-origin weights are never supplied",
        "historical_window": "repository definition: 250 stored sessions strictly before origin, plus current state",
        "window_difference": "DYSANOS comparison has 250 closes including origin; GROMM2 retains its own public 250-prior-session recipe",
        "targets": "observed bid/ask intervals; certified SANOS supplies geometry and numerical initialization only",
        "objective": PAPER_OBJECTIVE,
        "required_cache_days": list(map(str, required)), "source": str(data.sanos_run())}
    _json(run / "PROTOCOL.json", protocol)
    data.build_cache(run / "data-cache", required, workers=2)
    for origin in selected:
        prior = data.history_window(available, origin, config.GROMM2.history)
        for before, after in zip(prior[:-1], prior[1:], strict=True):
            if not data.adjacent_sessions(before, after):
                raise ValueError(f"GROMM2 stored-history gap at {before}->{after}; no date is silently skipped")
    _json(run / "INPUTS_READY.json", {"origins": len(selected), "cached_days": len(required)})


def train(run: Path, reference: Path, *, chunk: int, origin: str | None = None, fit_only: bool = False) -> None:
    import jax
    from openqdata.models.gromm import calibrate, pipeline

    from quantdata.hedging.model_audit.gromm_paper_objective import install_paper_objective
    from quantdata.hedging.model_audit.repository_gromm2_forecast import score_origin
    from quantdata.hedging.model_audit.repository_gromm2_inputs import prepared_views

    if jax.default_backend() not in ("gpu", "cuda"):
        raise ValueError("This worker must run inside its managed GPU allocation")
    install_paper_objective()
    if not (run / "INPUTS_READY.json").exists():
        raise ValueError("Complete the detached cache-only preflight before training")
    available = origins(reference)
    selected = (origin,) if origin is not None else available
    if not set(selected) <= set(available):
        raise ValueError("GROMM2 origin is outside the approved comparison")
    if origin is None:
        prepared = json.loads((run / "CALIBRATION_INPUTS_READY.json").read_text())
        if not set(selected) <= set(prepared["origins"]):
            raise ValueError("All requested calibration inputs must be prepared before GPU training")
    elif not (pipeline.origin_dir(run, "gromm2", date.fromisoformat(origin)) / "prepared/COMPLETE.json").exists():
        raise ValueError("This origin must finish CPU preparation before its GPU training")
    for index, origin in enumerate(selected):
        health_check()
        directory = pipeline.origin_dir(run, "gromm2", date.fromisoformat(origin))
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "ESTIMATION_COMPLETE.json").exists():
            if not fit_only and not (directory / "FORECAST_COMPLETE.json").exists():
                score_origin(run, reference, origin=origin)
            continue
        with FileLock(directory / ".training.lock"), Progress(directory, family="repository GROMM2", origin=origin) as progress:
            progress.phase = "independent market-surface fit, published quote objective and 1000 epochs"
            progress.completed = 1
            original = calibrate.fit
            original_views, original_extract = calibrate.views_for_origin, calibrate.extract_x
            if not (directory / "fit.json").exists():
                views, initials = prepared_views(directory / "prepared")
                calibrate.views_for_origin = lambda *_: views
                calibrate.extract_x = lambda geom, view, day: initials[(view.date, view.kind)].copy()
            calibrate.fit = checkpointed_fit(directory, original)
            try:
                if (directory / "fit.json").exists():
                    fitted = json.loads((directory / "fit.json").read_text())
                else:
                    fitted = pipeline.fit_origin(run, "gromm2", run / "data-cache", date.fromisoformat(origin),
                        epochs=1000, chunk=chunk, log=lambda value: print(value, flush=True))
            finally:
                calibrate.fit = original
                calibrate.views_for_origin, calibrate.extract_x = original_views, original_extract
            if not fitted["accepted"]:
                _json(directory / "REQUIRES_REVIEW.json", {"reason": "repository calibration coverage gate", "fit": fitted})
                raise RuntimeError(f"GROMM2 {origin} failed its original coverage gate; exact fit retained for review")
            progress.phase = "repository latent-X likelihood, density-weighted factors and compatible P/Q coefficients"
            progress.completed = 2
            dynamics = pipeline.estimate_origin(run, "gromm2", run / "data-cache", date.fromisoformat(origin))
            spot = dynamics["spot"]
            if not spot["converged"]:
                _json(directory / "REQUIRES_REVIEW.json", {"reason": "latent-X optimizer requires independent accuracy review", "dynamics": dynamics})
                raise RuntimeError(f"GROMM2 {origin} latent optimizer: {spot}")
            _json(directory / "ESTIMATION_COMPLETE.json", {"model": "repository GROMM2", "origin": origin,
                "fit_sha256": _hash(directory / "fit.npz"), "dynamics_sha256": _hash(directory / "dynamics.npz"),
                "status": "trained and estimated; matched forecast and no-arbitrage diagnostics still pending"})
            if fit_only:
                progress.completed = 4
                progress.phase = "fit and dynamics complete; statistics deferred until both 44-origin fits finish"
                progress.emit()
                continue
            progress.completed = 3
            progress.phase = "matched one-day P/Q observable forecasts"
            score_origin(run, reference, origin=origin)
            progress.completed = 4
            progress.phase = "fit, dynamics and one-day observable forecasts complete"
            progress.emit()
        print(f"repository GROMM2 estimated={index+1}/{len(selected)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-calibration-only", action="store_true")
    parser.add_argument("--wait-and-train", action="store_true")
    parser.add_argument("--origin")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--reuse-prepared-from", type=Path)
    parser.add_argument("--prepare-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.run, args.reference)
    elif args.wait_and_train or args.prepare_calibration_only:
        from quantdata.hedging.model_audit.repository_gromm2_inputs import prepare_origin

        if not (args.run / "INPUTS_READY.json").exists():
            prepare(args.run, args.reference)
        selected = origins(args.reference)
        if args.origin is not None:
            selected = (selected[selected.index(args.origin)],)
        prepare_one = partial(prepare_origin, args.run, reuse_from=args.reuse_prepared_from)
        if args.prepare_workers == 1:
            for index, origin in enumerate(selected):
                health_check()
                prepare_one(origin)
                print(f"CPU calibration preparation origins={index+1}/{len(selected)}; GPU not allocated", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                for index, _ in enumerate(pool.map(prepare_one, selected)):
                    health_check()
                    print(f"CPU calibration preparation origins={index+1}/{len(selected)}; GPU not allocated", flush=True)
        _json(args.run / "CALIBRATION_INPUTS_READY.json", {"origins": list(selected),
            "policy": "all requested origins prepared on CPU before allocating a GPU"})
        if args.prepare_calibration_only:
            return
        wait_gpu(args.run, origin="repository GROMM2 rolling April-May", completed=0, total=44, poll_seconds=180.)
        command = [os.sys.executable, "-u", "-m", "quantdata.hedging.model_audit.repository_gromm2",
            "--run", str(args.run), "--reference", str(args.reference), "--chunk", str(args.chunk)]
        if args.origin is not None:
            command.extend(["--origin", args.origin])
        environment = dict(os.environ)
        environment.pop("JAX_PLATFORMS", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        run_owned(["gpuctl", "run", "--gpus", "1", "--task-name", "Hedging GROMM2 repository rolling fits",
            "--task-id", "01a12f09-4975-7431-a499-7c321ab35fa5", "--label", "GROMM2 44 independent origins", "--", *command],
            environment=environment)
    else:
        train(args.run, args.reference, chunk=args.chunk, origin=args.origin, fit_only=args.fit_only)


if __name__ == "__main__":
    main()
