"""Causal DYSANOS v1/v2 inputs and P-law training; never invent a Q law."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import date
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedging.model_audit.rolling_versions import INITIALIZATION_POLICY, origins
from quantdata.hedging.model_audit.run import _hash, _json


def window_dates(calendar: list[date], origin: date, *, count: int = 250) -> tuple[date, ...]:
    """Declared standalone windows include the observed close, never a future close."""
    index = calendar.index(origin)
    result = tuple(calendar[max(0, index - count + 1):index + 1])
    if len(result) != count or result[0] < date(2020, 1, 1):
        raise ValueError("A complete post-2020 causal session window is required")
    return result


def prepare(root: Path, reference: Path, source: Path, *, family: str) -> None:
    """Reuse certified daily SANOS; no provider access or source SANOS refitting."""
    from openqdata.biz_calendar import get_business_days
    from openqdata.models import dysanos1, dysanos2
    from openqdata.models.dysanos.data import recover_initial_state
    from openqdata.models.dysanos.market import CachedMarketDataset

    selected = origins(reference)
    calendar = get_business_days(date(2020, 1, 1), date.fromisoformat(selected[-1]))
    required = sorted({day for text in selected for day in window_dates(calendar, date.fromisoformat(text))})
    dataset = CachedMarketDataset()
    available = set(dataset.dates(required[0], required[-1]))
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"Missing required licensed daily inputs, not silently excluded: {missing}")
    config = dysanos1.Config.paper() if family == "dysanos1" else dysanos2.Config()
    _json(root / f"prepared/{family}/PROTOCOL.json", {"schema": "causal-dysanos-preparation-v1", "family": family,
        "origins": selected, "sessions_per_window": 250, "includes_origin": True,
        "required_dates": [str(day) for day in required], "source_root": str(source),
        "config": config.to_dict(), "source_sha256": _hash(Path(__file__)),
        "native_initialization": "per-date SANOS-derived chart seed, not final calibrated states or all-history factor weights",
        "measure": "P only; no Q law is asserted"})
    started = time.monotonic()
    for index, day in enumerate(required):
        destination = root / f"prepared/{family}/days/{day}.npz"
        if destination.exists():
            continue
        from dataclasses import replace

        market = dataset.day(day, replace(config.geometry, nms_range=(-1e6, 1e6)), initialize=False)
        raw, initialization_audit = recover_initial_state(market.source, config.geometry)
        if family == "dysanos2":
            blocks = raw[:-2].reshape(len(config.geometry.times), config.geometry.num_dlvs + 1).copy()
            forward = config.geometry.min_forward_vol + (config.geometry.max_forward_vol - config.geometry.min_forward_vol) / (1 + np.exp(-blocks[:, -1]))
            intervals = np.diff(config.geometry.times, prepend=0.)
            total = np.cumsum(forward * np.sqrt(intervals))
            blocks[:, -1] = np.log(np.diff(total**2, prepend=0.) / intervals)
            raw = blocks.ravel()
        active = (market.maturities >= config.geometry.times[0] - 1e-8) & (market.maturities <= config.geometry.times[-1] + 1e-8)
        primary = active & (market.nms >= config.geometry.nms_range[0]) & (market.nms <= config.geometry.nms_range[1])
        if not primary.any():
            raise ValueError(f"No eligible native-tenor quote on {day}")
        metadata = {"date": str(day), "source_model_sha256": market.source_sha256,
            "quote_population": market.quote_population, "initialization_audit": dict(market.initialization_audit, seed=initialization_audit),
            "config": config.to_dict(), "family": family}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, initial_native=raw, times=np.clip(market.maturities[active], config.geometry.times[0], config.geometry.times[-1]),
            strikes=market.strikes[active], bids=market.bids[active], asks=market.asks[active],
            mask=np.where(primary[active], 1., .01),
            metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        temporary.replace(destination)
        elapsed = time.monotonic() - started
        print(f"prepare {family} completed={index+1}/{len(required)} date={day} quotes={int(active.sum())} "
              f"elapsed={elapsed:.1f}s eta={elapsed*(len(required)-index-1)/(index+1):.1f}s", flush=True)
    _json(root / f"prepared/{family}/COMPLETE.json", {"dates": len(required), "origins": 44, "family": family})


def quote_panel(root: Path, family: str, dates: tuple[date, ...]):
    from openqdata.models.dysanos._reference import QuotePanel

    rows, raw, source_ids = [], [], []
    for day in dates:
        path = root / f"prepared/{family}/days/{day}.npz"
        with np.load(path, allow_pickle=False) as data:
            rows.append({key: data[key].copy() for key in ("times", "strikes", "bids", "asks", "mask")})
            raw.append(data["initial_native"].copy())
            source_ids.append(_hash(path))
    capacity = max(len(row["times"]) for row in rows)
    arrays = {key: np.full((len(rows), capacity), 1. if key in ("strikes", "asks") else 0.) for key in rows[0]}
    for index, row in enumerate(rows):
        for key, value in row.items():
            arrays[key][index, :len(value)] = value
    return QuotePanel(dates=tuple(str(day) for day in dates),
        source_ids=tuple(source_ids), **arrays), np.stack(raw)


def train_v1(root: Path, reference: Path, *, day: str, batch_size: int = 16) -> None:
    """Fresh causal affine market refinement followed by the full joint VAR fit."""
    from openqdata.biz_calendar import get_business_days
    from openqdata.models import dysanos1
    from openqdata.models.gromm.progress import Heartbeat

    origins(reference).index(day)
    calendar = get_business_days(date(2020, 1, 1), date.fromisoformat(day))
    dates = window_dates(calendar, date.fromisoformat(day))
    panel, initial_native = quote_panel(root, "dysanos1", dates)
    directory = root / f"fits/dysanos1/{day}"
    directory.mkdir(parents=True, exist_ok=True)
    config, training = dysanos1.Config.paper(), dysanos1.TrainConfig(batch_size=batch_size)
    _json(directory / "PROTOCOL.json", {"schema": "causal-paper-dysanos1-v1", "origin": day,
        "dates": panel.dates, "source_ids": panel.source_ids, "config": config.to_dict(),
        "training": vars(training), "initialization": INITIALIZATION_POLICY,
        "measure": "P", "stages": "native market -> affine market -> joint five-session VAR", "source_sha256": _hash(Path(__file__))})
    stopping, last_checkpoint = False, 0.

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, origin=day, family="dysanos1", job_elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)

    def checkpoint(params, stage, *, force=False):
        nonlocal last_checkpoint
        if not force and time.monotonic() - last_checkpoint < 240.:
            return
        path = directory / "work-snapshot.partial.npz"
        np.savez_compressed(path, metadata=np.asarray(json.dumps(dict(stage, source_ids=panel.source_ids,
            resumability="parameter work snapshot; optimizer moments are not exposed by the canonical fit API"))),
            **{key: np.asarray(value) for key, value in params.items()})
        path.replace(directory / "work-snapshot.npz")
        last_checkpoint = time.monotonic()

    with FileLock(directory / ".training.lock"), Heartbeat(emit, interval=60.) as progress:
        model, history = dysanos1.fit(panel, initial_native, config=config, training=training,
            progress=progress.update, checkpoint=checkpoint, stop_requested=lambda: stopping)
        if model.dates[-1] != day:
            raise ValueError("Model origin does not match its actual training cutoff")
        model.save(directory / "model.npz")
        dysanos1.Model.load(directory / "model.npz")
        _json(directory / "training-history.json", history)
        from openqdata.models.dysanos.reference_calibration import Calibration, classify, market_fit

        surface_fit = market_fit(panel, model.params["b"] + model.params["h"] @ model.params["B"],
            family="dysanos1", config=config, weights=np.ones(len(dates)))
        status = classify(surface_fit, Calibration())
        _json(directory / "market-fit.json", dict(surface_fit, status=status))
        if status == "review_required":
            raise RuntimeError("DYSANOS v1 market-fit gate requires review; model and exact diagnostics are retained")
        _json(directory / "COMPLETE.json", {"family": "dysanos1", "origin": day, "measure": "P",
            "model_sha256": _hash(directory / "model.npz"), "training_start": str(dates[0]), "training_end": str(dates[-1]),
            "state_dimension": 200, "transition_sessions": 5,
            "initialization_origin": None, "initialization_policy": INITIALIZATION_POLICY,
            "scope": "causal rolling fit; not the all-history checkpoint", "forecast_statistics": "pending matched evaluation"})


def calibrate_v2(root: Path, reference: Path) -> None:
    """Cache independent daily shape fits; never share a fitted latent/VAR window."""
    from openqdata.models import dysanos2
    from openqdata.models.gromm.progress import Heartbeat

    manifest = json.loads((root / "prepared/dysanos2/PROTOCOL.json").read_text())
    days = tuple(date.fromisoformat(value) for value in manifest["required_dates"])
    config = dysanos2.Config()
    destination = root / "calibrated/dysanos2"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, family="dysanos2", elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)

    with Heartbeat(emit, interval=60.) as progress:
        for offset in range(0, len(days), 2):
            pair = days[offset:offset + 2] if offset + 1 < len(days) else days[-2:]
            if all((destination / f"{day}.npz").exists() for day in pair):
                continue
            panel, initial = quote_panel(root, "dysanos2", pair)
            progress.update({"phase": "market_shapes", "completed": offset, "total": len(days), "dates": panel.dates})
            fitted, audit = dysanos2.calibrate_shapes(panel, initial, config=config, progress=progress.update)
            for index, day in enumerate(pair):
                if (destination / f"{day}.npz").exists():
                    continue
                temporary = destination / f"{day}.partial.npz"
                np.savez_compressed(temporary, native=fitted[index],
                    audit=np.asarray(json.dumps(audit[index], allow_nan=False)), source_id=np.asarray(panel.source_ids[index]))
                temporary.replace(destination / f"{day}.npz")
            progress.update({"phase": "market_shapes_saved", "completed": min(offset + 2, len(days)), "total": len(days)})
    _json(destination / "COMPLETE.json", {"family": "dysanos2", "dates": len(days), "independent_daily_shapes": True})


def train_v2(root: Path, reference: Path, source: Path, *, day: str) -> None:
    """Daily STRT-16/Tail Diffusion with the actual fitted front-law spot scores."""
    import jax
    import jax.numpy as jnp
    from openqdata.biz_calendar import get_business_days
    from openqdata.models import dysanos2
    from openqdata.models.dysanos._reference import session_pairs
    from openqdata.models.gromm.progress import Heartbeat

    origins(reference).index(day)
    calendar = get_business_days(date(2020, 1, 1), date.fromisoformat(day))
    dates = window_dates(calendar, date.fromisoformat(day))
    config = dysanos2.Config()
    levels, source_ids, audits = [], [], []
    for point in dates:
        path = root / f"calibrated/dysanos2/{point}.npz"
        with np.load(path, allow_pickle=False) as saved:
            levels.append(saved["native"].copy())
            audits.append(json.loads(str(saved["audit"])))
        source_ids.append(_hash(path))
    levels = np.stack(levels)
    pairs, _ = session_pairs(tuple(str(point) for point in dates))
    financing = []
    for point in dates:
        with np.load(root / f"prepared/dysanos2/days/{point}.npz", allow_pickle=False) as saved:
            financing.append(json.loads(str(saved["metadata"]))["initialization_audit"])
    gross = np.asarray([financing[after]["spot"] / financing[before]["front_forward"] for before, after in pairs])
    scores = np.asarray(jax.jit(jax.vmap(lambda raw, ratio: dysanos2.front_score(raw, ratio, config=config)))(
        jnp.asarray(levels[pairs[:, 0]]), jnp.asarray(gross)))
    directory = root / f"fits/dysanos2/{day}"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, family="dysanos2", origin=day, elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)

    with FileLock(directory / ".training.lock"), Heartbeat(emit, interval=60.) as progress:
        progress.update({"phase": "STRT_projection_and_daily_VAR", "completed": 0, "total": 1})
        model, diagnostics = dysanos2.fit(levels, tuple(str(point) for point in dates),
            source_ids=tuple(source_ids), spot_scores=scores, config=config)
        model.save(directory / "model.npz")
        dysanos2.Model.load(directory / "model.npz")
        _json(directory / "shape-fit-quality.json", audits)
        from openqdata.models.dysanos.reference_calibration import Calibration, classify, market_fit

        panel, _ = quote_panel(root, "dysanos2", dates)
        _, sessions = session_pairs(tuple(str(point) for point in dates))
        date_weights = config.oldest_weight ** ((sessions[-1] - sessions) / 249.)
        surface_fit = market_fit(panel, levels, family="dysanos2", config=config, weights=date_weights)
        status = classify(surface_fit, Calibration())
        _json(directory / "market-fit.json", dict(surface_fit, status=status))
        if status == "review_required":
            raise RuntimeError("DYSANOS v2 market-fit gate requires review; model and exact diagnostics are retained")
        _json(directory / "COMPLETE.json", {"family": "dysanos2", "origin": day, "measure": "P",
            "model_sha256": _hash(directory / "model.npz"), "source_ids": source_ids,
            "training_start": str(dates[0]), "training_end": str(dates[-1]), "surface_rank": 16,
            "initialization_origin": None, "initialization_policy": INITIALIZATION_POLICY,
            "transition_sessions": 1, "metric": "STRT", "innovation": "TAIL", "spot_score": "fitted native front PIT",
            "diagnostics": json.loads(json.dumps(diagnostics, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())),
            "forecast_statistics": "pending matched common-reference evaluation"})
        progress.update({"phase": "causal_dysanos2_complete", "completed": 1, "total": 1})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--family", choices=("dysanos1", "dysanos2"), required=True)
    parser.add_argument("--origin")
    parser.add_argument("--calibrate-shapes", action="store_true")
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("QUANTDATA_ROOT is unavailable")
    if args.calibrate_shapes:
        if args.family != "dysanos2":
            raise ValueError("Independent fixed-smoothness shape calibration is a v2 stage")
        calibrate_v2(args.root, args.reference)
    elif args.origin:
        if args.family == "dysanos1":
            train_v1(args.root, args.reference, day=args.origin)
        else:
            train_v2(args.root, args.reference, args.source, day=args.origin)
    else:
        prepare(args.root, args.reference, args.source, family=args.family)


if __name__ == "__main__":
    main()
