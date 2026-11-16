"""Owned DYSANOS-first execution with measured numerical parity and throughput."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.rolling_dysanos import (
    calibrate_v2,
    prepare,
    quote_panel,
    train_v1,
    train_v2,
    window_dates,
)
from quantdata.hedging.model_audit.rolling_versions import (
    INITIALIZATION_POLICY,
    origins,
    resource_health,
)
from quantdata.hedging.model_audit.run import _hash, _json

STOP_REQUESTED = threading.Event()


def run_child(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    """Check managed headroom while a single owned worker is computing."""
    check_resources()
    if STOP_REQUESTED.is_set():
        raise InterruptedError("Owned queue stopped before another worker")
    process = subprocess.Popen(command, env=environment)
    while process.poll() is None:
        try:
            process.wait(timeout=60.)
        except subprocess.TimeoutExpired:
            health = resource_health()
            if STOP_REQUESTED.is_set() or any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in health):
                process.terminate()
                try:
                    process.wait(timeout=60.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise InterruptedError("Owned worker stopped; retain its last complete periodic checkpoint")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)

def check_resources() -> None:
    if any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in resource_health()):
        raise RuntimeError("Resource stop; no additional owned work is launched")


def benchmark(root: Path, reference: Path) -> None:
    """Full-geometry solves and unchanged quote gradients, not smaller models."""
    from datetime import date

    import jax
    import jax.numpy as jnp
    from openqdata.arbfree.tridiag import solve_sanos_tridiagonal
    from openqdata.biz_calendar import get_business_days
    from openqdata.models import dysanos1
    from openqdata.models.dysanos._tridiagonal import solve
    from openqdata.models.dysanos1.train import quote_loss
    from openqdata.models.gromm.progress import Heartbeat

    check_resources()
    selected = origins(reference)
    calendar = get_business_days(date(2020, 1, 1), date.fromisoformat(selected[0]))
    dates = window_dates(calendar, date.fromisoformat(selected[0]))
    panel, raw = quote_panel(root, "dysanos1", dates)
    config = dysanos1.Config.paper()
    report = {"schema": "dysanos-numerical-throughput-v1", "origin": selected[0],
        "observations": len(dates), "quote_capacity": panel.times.shape[1], "state_dimension": 200,
        "native_size": raw.shape[1], "jax_devices": [str(device) for device in jax.devices()],
        "initialization_policy": INITIALIZATION_POLICY, "runs": {}}
    generator = np.random.default_rng(9281)
    minus, plus = np.exp(generator.normal(size=(2, 200)))
    rhs = generator.uniform(size=202)
    rhs /= rhs.sum()
    expected = solve_sanos_tridiagonal(w_minus=minus, w_plus=plus, rhs=rhs)
    actual = np.asarray(jax.jit(solve)(jnp.asarray(minus), jnp.asarray(plus), jnp.asarray(rhs)))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    direction = generator.normal(size=200)
    payoff = generator.normal(size=202)
    gradient = np.asarray(jax.grad(lambda values: jnp.asarray(payoff) @ solve(values, jnp.asarray(plus), jnp.asarray(rhs)))(jnp.asarray(minus)))
    epsilon = 1e-5
    positive = solve_sanos_tridiagonal(w_minus=minus + epsilon * direction, w_plus=plus, rhs=rhs)
    negative = solve_sanos_tridiagonal(w_minus=minus - epsilon * direction, w_plus=plus, rhs=rhs)
    derivative = payoff @ (positive - negative) / (2 * epsilon)
    np.testing.assert_allclose(gradient @ direction, derivative, atol=2e-9, rtol=2e-6)
    report.update(solve_max_error=float(np.max(np.abs(actual - expected))),
        directional_gradient_error=float(abs(gradient @ direction - derivative)))
    inputs = tuple(jnp.asarray(getattr(panel, name)) for name in ("times", "strikes", "bids", "asks", "mask"))
    states = jnp.asarray(raw)

    def loss(native, *quotes):
        return jax.vmap(lambda state, *values: quote_loss(state, *values, config=config))(native, *quotes).mean()

    operation = jax.jit(jax.value_and_grad(loss))
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, elapsed_seconds=time.monotonic() - started)), flush=True)

    with Heartbeat(emit, interval=60.) as progress:
        for index, batch in enumerate((1, 4, 8, 16, 32)):
            check_resources()
            progress.update({"phase": "full_geometry_gradient_benchmark", "completed": index, "total": 5, "batch_size": batch})
            arguments = (states[:batch], *(values[:batch] for values in inputs))
            before = time.monotonic()
            result = operation(*arguments)
            jax.block_until_ready(result)
            compilation = time.monotonic() - before
            timings = []
            for repetition in range(3):
                before = time.monotonic()
                result = operation(*arguments)
                jax.block_until_ready(result)
                timings.append(time.monotonic() - before)
            if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(result)):
                raise FloatingPointError("Benchmark produced nonfinite objective/gradient")
            report["runs"][str(batch)] = {"compile_and_first_seconds": compilation,
                "steady_seconds": timings, "seconds_per_date": float(np.median(timings) / batch),
                "objective": float(result[0])}
            _json(root / "benchmarks/dysanos1.json", report)
            progress.update({"phase": "gradient_benchmark_complete", "completed": index + 1, "total": 5,
                "batch_size": batch, **report["runs"][str(batch)]})
    report["selected_batch_size"] = int(min(report["runs"], key=lambda key: report["runs"][key]["seconds_per_date"]))
    _json(root / "benchmarks/dysanos1.json", report)


def run_queue(root: Path, reference: Path, source: Path) -> None:
    """One allocation owns child lifetimes; independently fit all agreed origins."""
    selected = origins(reference)
    benchmark_path = root / "benchmarks/dysanos1.json"
    batch_size = json.loads(benchmark_path.read_text())["selected_batch_size"]
    check_resources()
    for family in ("dysanos2", "dysanos1", "gromm1"):
        if family == "dysanos2" and not (root / "calibrated/dysanos2/COMPLETE.json").exists():
            command = [sys.executable, "-u", "-m", __spec__.name, "--root", str(root), "--reference", str(reference),
                "--source", str(source), "--mode", "shapes"]
            run_child(command)
        for index, day in enumerate(selected):
            check_resources()
            if (root / f"fits/{family}/{day}/COMPLETE.json").exists():
                continue
            print(json.dumps({"phase": "launch_origin", "family": family, "origin": day,
                "completed": index, "total": len(selected)}), flush=True)
            command = [sys.executable, "-u", "-m", __spec__.name, "--root", str(root), "--reference", str(reference),
                "--source", str(source), "--mode", "origin", "--family", family, "--origin", day,
                "--batch-size", str(batch_size)]
            run_child(command)
        _json(root / f"{family}-TRAINING_COMPLETE.json", {"family": family, "origins": len(selected),
            "forecast_statistics": "pending matched common-reference evaluation"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("pipeline", "prepare", "benchmark", "queue", "shapes", "origin"), required=True)
    parser.add_argument("--family", choices=("dysanos1", "dysanos2", "gromm1"))
    parser.add_argument("--origin")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("Configured QUANTDATA_ROOT is unavailable")
    if args.mode in ("pipeline", "queue"):
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: STOP_REQUESTED.set())
    if args.mode == "pipeline":
        command = [sys.executable, "-u", "-m", __spec__.name, "--root", str(args.root),
            "--reference", str(args.reference), "--source", str(args.source)]
        environment = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        run_child([*command, "--mode", "prepare"], environment=environment)
        run_child([*command, "--mode", "benchmark"])
        run_queue(args.root, args.reference, args.source)
    elif args.mode == "prepare":
        for family in ("dysanos1", "dysanos2"):
            check_resources()
            prepare(args.root, args.reference, args.source, family=family)
        _json(args.root / "SOURCE_RUNTIME.json", {"files": {str(path.relative_to(args.root)): _hash(path)
            for path in (args.root / "source").rglob("*.py")}})
    elif args.mode == "benchmark":
        benchmark(args.root, args.reference)
    elif args.mode == "queue":
        run_queue(args.root, args.reference, args.source)
    elif args.mode == "shapes":
        calibrate_v2(args.root, args.reference)
    elif args.family == "dysanos1":
        train_v1(args.root, args.reference, day=args.origin, batch_size=args.batch_size)
    elif args.family == "dysanos2":
        train_v2(args.root, args.reference, args.source, day=args.origin)
    else:
        from quantdata.hedging.model_audit.rolling_versions import gromm_origin

        gromm_origin(args.root, args.reference, family="gromm1", day=args.origin, source=args.source)


if __name__ == "__main__":
    main()
