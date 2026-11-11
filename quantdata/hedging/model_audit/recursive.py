"""Resumable frozen-fit recursive diagnostics for diffusion and GROMM models."""

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
from quantdata.hedging.model_audit.gromm import kernel_prices
from quantdata.hedging.model_audit.implied import (
    implied_volatilities_with_validity,
    summarize_inversions,
)
from quantdata.hedging.model_audit.inference import model_from_export
from quantdata.hedging.model_audit.persistence import path_acf
from quantdata.hedging.model_audit.run import (
    COMMON_INDICES,
    COMMON_NMS,
    GRID,
    _hash,
    _json,
    build_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    diffusion_live_context_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    _sample_diffusion_next_states,
)
from quantdata.torch import BaseEnvironment, torch

SEED = 20_261_111


def append_diffusion_context(context: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Use the original adapter's spatially averaged spot return in the next context."""
    if context.ndim != 5 or generated.shape != (context.shape[0], *context.shape[2:]):
        raise ValueError("Recursive diffusion contexts and generated states must align")
    returns = generated[:, 1].mean(axis=(-2, -1))
    state = np.stack((generated[:, 0], np.broadcast_to(returns[:, None, None], generated[:, 0].shape)), axis=1)
    return np.concatenate((context[:, 1:], state[:, None]), axis=1)


def available_path_steps(business_dtes: np.ndarray, requested: int) -> int:
    """Do not invent new fixed expiries or extrapolate beyond the last saved expiry."""
    maximum_tenor = float(np.max(np.asarray(GRID.expiries_y) * 252.))
    available = int(np.floor(float(np.max(business_dtes)) - maximum_tenor + 1e-8))
    if available < 2:
        raise ValueError("Saved GROMM fixed expiries cannot support a recursive common-grid test")
    return min(requested, available)


def aged_observable_surface(supports: np.ndarray, densities: np.ndarray, background: np.ndarray,
                            spot: np.ndarray, remaining_dtes: np.ndarray,
                            *, inversion_diagnostics: list[dict] | None = None) -> np.ndarray:
    """Decode floating IVs by total-variance interpolation at fixed normalized strikes.

    Expired rows are discarded. The zero-maturity total-variance anchor is zero;
    between zero and the first live expiry this means flat implied volatility.
    There is no extension beyond the last native expiry.
    """
    alive = np.flatnonzero(remaining_dtes > 1e-8)
    times = remaining_dtes[alive] / 252.
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Active GROMM expiries must be strictly increasing")
    paths = len(spot)

    def native_vol(row: int, strikes: np.ndarray) -> np.ndarray:
        prices = kernel_prices(supports[row], densities[:, row], background, spot, strikes,
                               expiry=float(remaining_dtes[row] / 252.))
        vols, valid = implied_volatilities_with_validity(
            expiries_y=np.full(prices.size, remaining_dtes[row] / 252.),
            normalized_strikes=strikes.ravel(), normalized_prices=prices.ravel(),
            is_call=(strikes >= 1.).ravel(), diagnostics=inversion_diagnostics)
        return np.where(valid, vols, np.nan).reshape(strikes.shape)

    def interpolated_vol(expiry: float, strikes: np.ndarray) -> np.ndarray:
        if expiry > times[-1] + 1e-10:
            raise ValueError("Recursive GROMM observable expiry exceeds the saved tower")
        upper = min(int(np.searchsorted(times, expiry)), len(times) - 1)
        high = native_vol(int(alive[upper]), strikes)
        if upper == 0 or np.isclose(times[upper], expiry, atol=1e-10, rtol=0):
            return high
        lower = upper - 1
        low = native_vol(int(alive[lower]), strikes)
        weight = (expiry - times[lower]) / (times[upper] - times[lower])
        return np.sqrt(((1. - weight) * times[lower] * low**2 + weight * times[upper] * high**2) / expiry)

    surfaces = []
    for expiry in GRID.expiries_y:
        atm = interpolated_vol(expiry, np.ones((paths, 1)))
        strikes = NMS.from_nms(COMMON_NMS[None], expiry, vol=atm)
        surfaces.append(np.log(interpolated_vol(expiry, strikes)))
    return np.stack(surfaces, axis=1)


def _save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _resume(path: Path, identity: dict) -> dict:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["identity_json"])) != identity:
            raise ValueError("Recursive checkpoint identity differs; do not silently reuse it")
        return {name: saved[name] for name in saved.files if name != "identity_json"}


def _progress(model: str, day: date, completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    remaining = elapsed * (total - completed) / max(completed, 1)
    print(f"recursive {model} {day} completed={completed}/{total} elapsed={elapsed:.1f}s remaining={remaining:.1f}s", flush=True)


def _diffusion(output: Path, model: str, day: date, history, index: int, *, paths: int,
               steps: int, device: str, stop_requested) -> tuple[np.ndarray, dict] | None:
    catalog = json.loads((output / "INFERENCE_MODELS.json").read_text())
    variant = "ad-seq-vol-ft" if model.endswith("FT") else "ad-seq-vol"
    selected = [entry for entry in catalog["models"].values()
                if entry["origin_date"] == str(day) and entry["variant"] == variant]
    if len(selected) != 1 or not selected[0]["verified_inference"]:
        raise ValueError("No unique verified dated diffusion inference export")
    source = selected[0]
    environment = BaseEnvironment(precision=32, device=device)
    network = model_from_export(Path(source["manifest"]), environment=environment,
                                expected_sha256=source["manifest_sha256"])
    window = replace(history, dates=history.dates[:index + 1],
        log_implied_volatilities=history.log_implied_volatilities[:index + 1],
        implied_vol_is_interior=history.implied_vol_is_interior[:index + 1],
        forward_log_returns_into_date=history.forward_log_returns_into_date[:index + 1],
        sessions_from_previous_observation=history.sessions_from_previous_observation[:index + 1])
    context = diffusion_live_context_from_history(window, as_of_date=day, forecast_date=history.dates[index + 1])
    contexts = np.broadcast_to(context.sequence, (paths, *context.sequence.shape)).copy()
    random = torch.Generator(device=device).manual_seed(SEED)
    identity = {"model": model, "date": str(day), "paths": paths, "steps": steps, "seed": SEED,
                "manifest_sha256": source["manifest_sha256"], "device": device, "schema": 1}
    checkpoint = output / "completion/recursive" / model / f"{day}.npz"
    resumed = _resume(checkpoint, identity)
    surfaces = np.full((paths, steps, len(GRID.expiries_y) * len(COMMON_INDICES)), np.nan)
    begin = 0
    max_dispersion = 0.
    if resumed:
        contexts, surfaces = resumed["contexts"], resumed["surfaces"]
        random.set_state(torch.from_numpy(resumed["rng_state"]))
        begin = int(resumed["completed"])
        max_dispersion = float(resumed["max_return_dispersion"])
    started = time.monotonic()
    for position in range(begin, steps):
        generated = _sample_diffusion_next_states(SimpleNamespace(ema_model=network),
            observed_sequences=contexts, generator=random)
        max_dispersion = max(max_dispersion, float(np.max(np.ptp(generated[:, 1], axis=(-2, -1)))))
        surfaces[:, position] = generated[:, 0][:, :, COMMON_INDICES].reshape(paths, -1)
        contexts = append_diffusion_context(contexts, generated)
        _save_arrays(checkpoint, identity_json=json.dumps(identity), completed=position + 1,
            contexts=contexts, surfaces=surfaces, rng_state=random.get_state().cpu().numpy(),
            max_return_dispersion=max_dispersion)
        _progress(model, day, position + 1, steps, started)
        if stop_requested():
            return None
    return surfaces, {"lower_clip_coordinates": 0, "upper_clip_coordinates": 0,
        "inference_manifest": source["manifest"], "inference_manifest_sha256": source["manifest_sha256"],
        "return_channel_max_dispersion": max_dispersion, "device": device,
        "context_policy": "own generated history; spatially averaged return channel rebroadcast as original weekly adapter",
        "diffusion_steps_per_session": network.config.diffusion_steps}


def _gromm(root: Path, output: Path, model: str, day: date, *, paths: int,
           steps: int, stop_requested) -> tuple[np.ndarray, dict] | None:
    import jax
    import jax.numpy as jnp

    from quantdata.gromm2.dynamics import interpolate_loading_expiries
    from quantdata.gromm2.simulation import SimulationState
    from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
        load_gromm2_market_state,
    )

    jax.config.update("jax_enable_x64", True)
    directory = root / f"training/hedging/gromm-current-public-20261024-v1/artifacts/{day}"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = SimpleNamespace(affine_history=directory / "affine-200-history.h5",
                              dynamics=directory / "rank16-ar1-dynamics.npz")
    for name, source in (("affine_history_sha256", sources.affine_history), ("dynamics_sha256", sources.dynamics)):
        if _hash(source) != manifest[name]:
            raise ValueError(f"GROMM saved fit checksum mismatch: {name}")
    names = ("percent_inside_spread", "explained_variance_ratio", "physical_projection_fraction",
             "candidate_physical_projection_fraction", "physical_fallback_used", "physical_selected_model", "physical_accepted")
    market = load_gromm2_market_state(SimpleNamespace(paths=sources, as_of_date=day,
        **{name: manifest[name] for name in names}))
    if model.endswith("P") and not market.physical_accepted:
        raise ValueError("GROMM P requires the retained accepted physical fit")
    requested_steps = steps
    steps = available_path_steps(market.business_dtes, steps)
    intervals = np.diff(market.expiries_y, prepend=0.)
    reference_dtes = np.asarray(market.business_dtes[1:])
    random = np.random.default_rng(SEED)
    state = SimulationState(
        front_density=jnp.broadcast_to(jnp.asarray(market.density_tower[0]), (paths, market.supports.shape[1])),
        dlvsq=jnp.broadcast_to(jnp.asarray(market.dlvsq), (paths, *market.dlvsq.shape)),
        background_level=jnp.ones(paths))
    identity = {"model": model, "date": str(day), "paths": paths, "steps": steps, "seed": SEED,
                "manifest_sha256": _hash(manifest_path), "schema": 1}
    checkpoint = output / "completion/recursive" / model / f"{day}.npz"
    resumed = _resume(checkpoint, identity)
    surfaces = np.full((paths, steps, len(GRID.expiries_y) * len(COMMON_INDICES)), np.nan)
    begin, tamed, inversion_records = 0, [], []
    if resumed:
        state = SimulationState(*(jnp.asarray(resumed[name]) for name in ("front_density", "dlvsq", "background_level")))
        random.bit_generator.state = json.loads(str(resumed["rng_json"]))
        surfaces, begin, tamed = resumed["surfaces"], int(resumed["completed"]), resumed["tamed"].tolist()
        inversion_records = json.loads(str(resumed["inversion_json"])) if "inversion_json" in resumed else []
    started = time.monotonic()
    for position in range(begin, steps):
        local_day = position + 1
        front = int(np.searchsorted(market.business_dtes, local_day, side="left"))
        remaining_front = int(round(market.business_dtes[front] - position))
        loadings = interpolate_loading_expiries(market.dynamics.log_dlvsq_loadings,
            reference_dtes, market.business_dtes[front + 1:] - position)
        factors = random.standard_normal((8, paths, market.dynamics.factor_count))
        background_noise = random.standard_normal(paths)
        if model.endswith("P"):
            factors += market.physical_market_price_of_risk[None, None] * np.sqrt(1. / (252. * 8.))
        result = _gromm_step(jnp.asarray(market.supports[front:]), jnp.asarray(market.transports[front:]),
            jnp.asarray(intervals[front + 1:]), SimulationState(state.front_density, state.dlvsq[..., front:, :], state.background_level),
            jnp.asarray(loadings)[None], jnp.asarray(factors), jnp.asarray(background_noise),
            remaining_front_days=jnp.asarray(remaining_front), substeps=8)
        spot = np.asarray(result.spot)
        densities = np.asarray(result.density_tower)
        decoded = aged_observable_surface(market.supports[front:], densities,
            np.asarray(result.state.background_level), spot, market.business_dtes[front:] - local_day,
            inversion_diagnostics=inversion_records)
        surfaces[:, position] = decoded.reshape(paths, -1)
        state = SimulationState(result.state.front_density,
            state.dlvsq.at[..., front:, :].set(result.state.dlvsq), result.state.background_level)
        tamed.append(float(result.drift_taming_fraction))
        _save_arrays(checkpoint, identity_json=json.dumps(identity), completed=local_day,
            front_density=np.asarray(state.front_density), dlvsq=np.asarray(state.dlvsq),
            background_level=np.asarray(state.background_level), surfaces=surfaces,
            rng_json=json.dumps(random.bit_generator.state), tamed=np.asarray(tamed),
            inversion_json=json.dumps(inversion_records))
        _progress(model, day, local_day, steps, started)
        if stop_requested():
            return None
    return surfaces, {"lower_clip_coordinates": 0, "upper_clip_coordinates": 0,
        "fit_manifest": str(manifest_path), "fit_manifest_sha256": _hash(manifest_path),
        "requested_steps": requested_steps, "available_steps": steps,
        "horizon_reason": "last saved fixed expiry minus longest common observable tenor; no expiry replenishment",
        "observable_interpolation": "total variance at fixed K/S; zero-maturity anchor; no long-end extrapolation",
        "drift_taming_fraction_mean": float(np.mean(tamed)), "substeps": 8,
        "background_annual_volatility": .05, "physical_state_policy": "retained accepted constant market-price-of-risk",
        "iv_inversion": summarize_inversions(inversion_records)}


def _gromm_step(*args, **kwargs):
    import jax

    from quantdata.gromm2.simulation import one_day_risk_neutral_step

    global _compiled_step
    if _compiled_step is None:
        _compiled_step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    return _compiled_step(*args, **kwargs)


_compiled_step = None


def summarize_paths(surfaces: np.ndarray, basis: np.ndarray) -> dict:
    """Score every finite whole path, recording rather than hiding invalid paths."""
    valid = np.isfinite(surfaces).all(axis=(1, 2))
    diagnostics = {"invalid_path_rate": float(np.mean(~valid)),
                   "extreme_surface_rate": float(np.mean(np.any(surfaces > np.log(3.), axis=2)))}
    if not valid.any():
        return {"status": "model_invalid", "diagnostics": diagnostics,
                "reason": "Recursive model produced no finite complete paths"}
    selected = surfaces[valid]
    scores = selected @ basis
    return {"status": "scored", "diagnostics": diagnostics,
        "pc_level_acf": path_acf(scores), "pc_level_acf_matched44": path_acf(scores[:, :44]),
        "mean_log_iv_first": float(selected[:, 0].mean()), "mean_log_iv_last": float(selected[:, -1].mean()),
        "variance_log_iv_first": float(selected[:, 0].var(axis=0).mean()),
        "variance_log_iv_last": float(selected[:, -1].var(axis=0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("AD-Seq-Vol", "AD-Seq-Vol-FT", "GROMM-P", "GROMM-Q"), required=True)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--steps", type=int, default=115)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dates", nargs="*", type=date.fromisoformat)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--refresh-report", action="store_true")
    arguments = parser.parse_args()
    if arguments.paths < 2 or arguments.steps < 2:
        raise ValueError("Recursive diagnostics require at least two paths and two sessions")
    panel = PublicHedgingPanel(arguments.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, arguments.output)
    with np.load(arguments.output / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    origins = [index for index, day in enumerate(history.dates) if date(2024, 4, 1) <= day <= date(2024, 5, 31)
               and (arguments.dates is None or day in arguments.dates)]
    if arguments.preflight:
        print(f"preflight OK imports/root/history/PCA; dates={len(origins)} models={arguments.models} device={arguments.device}", flush=True)
        return
    interrupted = False

    def request_stop(signum, frame) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"signal {signum}: checkpointing after current recursive day", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for model in arguments.models:
        for index in origins:
            day = history.dates[index]
            destination = arguments.output / "persistence" / model / f"{day}.json"
            if destination.exists() and json.loads(destination.read_text()).get("status") in ("scored", "model_invalid"):
                continue
            if model.startswith("GROMM"):
                result = _gromm(arguments.root, arguments.output, model, day,
                    paths=arguments.paths, steps=arguments.steps, stop_requested=lambda: interrupted)
            else:
                result = _diffusion(arguments.output, model, day, history, index,
                    paths=arguments.paths, steps=arguments.steps, device=arguments.device, stop_requested=lambda: interrupted)
            if result is None:
                return
            surfaces, diagnostics = result
            record = summarize_paths(surfaces, basis)
            record["diagnostics"].update(diagnostics)
            record.update(model=model, origin_date=str(day), path_count=arguments.paths,
                steps=surfaces.shape[1], requested_steps=arguments.steps, seed=SEED,
                fit_frozen_through_path=True)
            path_arrays = destination.with_suffix(".npz")
            _save_arrays(path_arrays, log_iv_surfaces=surfaces)
            record["surface_artifact"] = str(path_arrays)
            record["surface_artifact_sha256"] = _hash(path_arrays)
            _json(destination, record)
            checkpoint = arguments.output / "completion/recursive" / model / f"{day}.npz"
            checkpoint.unlink(missing_ok=True)
            print(f"recursive origin complete model={model} date={day} status={record['status']} steps={surfaces.shape[1]}", flush=True)
            if arguments.refresh_report:
                from quantdata.hedging.model_audit.refresh import refresh_report

                refresh_report(arguments.output)


if __name__ == "__main__":
    main()
