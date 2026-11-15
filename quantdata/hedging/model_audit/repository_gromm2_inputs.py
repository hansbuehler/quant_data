"""Prepare the repository's exact calibration views and initializers without a GPU."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedging.model_audit.gromm_paper_objective import PAPER_OBJECTIVE, observed_view
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.run import _hash, _json


def write_view(path: Path, view, initial: np.ndarray) -> None:
    """Save the unmodified view and initializer with exact dtypes, without pickle."""
    values = {field.name: np.asarray(str(getattr(view, field.name)) if field.name == "date"
        else getattr(view, field.name)) for field in fields(view)}
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values, initial=initial)
    temporary.replace(path)


def read_view(path: Path):
    """Restore a repository View and its original implicit-Dupire initializer."""
    from openqdata.models.gromm.calibrate import View

    with np.load(path, allow_pickle=False) as saved:
        values = {field.name: saved[field.name] for field in fields(View)}
        values.update(date=date.fromisoformat(str(values["date"])), kind=str(values["kind"]),
            ratio=float(values["ratio"]))
        initial = saved["initial"].copy()
    return View(**values), initial


def prepare_origin(run: Path, origin: str, *, reuse_from: Path | None = None) -> dict:
    """Cache every CPU view and initializer before this origin may claim a GPU."""
    import jax
    from openqdata.models.gromm import calibrate, config, data, pipeline

    if jax.default_backend() != "cpu":
        raise ValueError("Calibration input preparation must run on CPU without a GPU allocation")
    current = date.fromisoformat(origin)
    history = data.history_window(data.stored_days(data.sanos_run()), current, config.GROMM2.history)
    geometry = pipeline.geometry_for("gromm2", run / "data-cache", current)
    directory = pipeline.origin_dir(run, "gromm2", current) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    specification = [(current, "start", current)] + [(day, "start", day) for day in history]
    specification += [(before, "end", after) for before, after in zip(history[:-1], history[1:], strict=True)]
    identity = {"schema": "paper-objective-calibration-inputs-v1", "origin": origin,
        "configuration": asdict(config.GROMM2), "source_snapshot_sha256": _hash(run / "SOURCE.json"),
        "data_sha256": {str(day): _hash(run / f"data-cache/{day}.npz") for day in [*history, current]},
        "views": len(specification), "objective": PAPER_OBJECTIVE}
    identity = json.loads(json.dumps(identity))
    with FileLock(directory / ".preparation.lock"):
        if (directory / "IDENTITY.json").exists():
            if json.loads((directory / "IDENTITY.json").read_text()) != identity:
                raise ValueError(f"Prepared GROMM2 data identity changed: {directory}")
        else:
            _json(directory / "IDENTITY.json", identity)
        if (directory / "COMPLETE.json").exists():
            return json.loads((directory / "COMPLETE.json").read_text())
        started = time.monotonic()
        days = {day: data.load_day(run / "data-cache", day) for day in [*history, current]}
        previous = None if reuse_from is None else pipeline.origin_dir(reuse_from, "gromm2", current) / "prepared"
        if previous is not None and (previous / "IDENTITY.json").exists():
            old_identity = json.loads((previous / "IDENTITY.json").read_text())
            for name in ("configuration", "source_snapshot_sha256", "data_sha256", "views"):
                if identity[name] != old_identity[name]:
                    raise ValueError(f"Prepared geometry/initializer reuse has incompatible {name}: {previous}")
        hashes = {}
        for index, (day, kind, quote_day) in enumerate(specification):
            if kind == "end" and not data.adjacent_sessions(day, quote_day):
                raise ValueError(f"GROMM2 history is not session-adjacent: {day}->{quote_day}")
            path = directory / f"{index:04d}.npz"
            if not path.exists():
                ratio = float(days[quote_day]["spot"]) / data.forward_1d(days[day]) if kind == "end" else 1.
                old_path = None if previous is None else previous / path.name
                if old_path is not None and old_path.exists():
                    view, initial = read_view(old_path)
                    if (view.date, view.kind, view.ratio) != (day, kind, ratio):
                        raise ValueError(f"Prepared view chronology/geometry changed: {old_path}")
                else:
                    view = calibrate.build_view(geometry, days[quote_day], day, kind, ratio)
                    initial = calibrate.extract_x(geometry, view, days[quote_day])
                view = observed_view(view, geometry, days[quote_day])
                if not np.isfinite(initial).all():
                    raise ValueError(f"Nonfinite original GROMM2 initializer: {origin} {day} {kind}")
                write_view(path, view, initial)
            hashes[path.name] = _hash(path)
            if (index + 1) % 5 == 0 or index + 1 == len(specification):
                elapsed = time.monotonic() - started
                event = {"phase": "CPU calibration input preparation; GPU not allocated", "origin": origin,
                    "completed_views": index + 1, "total_views": len(specification), "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed * (len(specification) - index - 1) / (index + 1)}
                _json(directory / "progress.json", event)
                print(json.dumps(event), flush=True)
                health_check()
        manifest = {"origin": origin, "views": len(specification), "identity_sha256": _hash(directory / "IDENTITY.json"),
            "files_sha256": hashes, "method": "original geometry/kernels/initializer; actual observed quote targets and paper loss",
            "reused_untrained_geometry_from": None if previous is None else str(previous)}
        _json(directory / "COMPLETE.json", manifest)
        return manifest


def prepared_views(directory: Path):
    """Return exact cached views and initializers for a completed origin preparation."""
    manifest = json.loads((directory / "COMPLETE.json").read_text())
    if _hash(directory / "IDENTITY.json") != manifest["identity_sha256"]:
        raise ValueError(f"Prepared GROMM2 identity checksum changed: {directory}")
    views, initials = [], {}
    started = time.monotonic()
    for index, (filename, digest) in enumerate(sorted(manifest["files_sha256"].items())):
        path = directory / filename
        if _hash(path) != digest:
            raise ValueError(f"Prepared GROMM2 input checksum changed: {path}")
        view, initial = read_view(path)
        views.append(view)
        initials[(view.date, view.kind)] = initial
        if (index + 1) % 50 == 0:
            print(f"load prepared calibration inputs views={index+1}/{manifest['views']} "
                f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    if len(views) != manifest["views"]:
        raise ValueError("Prepared GROMM2 view count is incomplete")
    return views, initials
