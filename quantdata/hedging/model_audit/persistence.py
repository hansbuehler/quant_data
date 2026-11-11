"""Finite-horizon, frozen-fit recursive persistence checks on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.implied import summarize_inversions
from quantdata.hedging.model_audit.run import (
    COMMON_INDICES,
    GRID,
    LAGS,
    _dlv_surface,
    _json,
    _record,
    build_history,
    retained_dynamics_path,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import FunvolConfig
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.model import FunvolNeuralSde
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import VolganPcaModelConfig
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.model import VolganGenerator
from quantdata.torch import BaseEnvironment, torch


def path_acf(scores: np.ndarray) -> list[dict]:
    """Mean within-path Pearson ACF; no cross-path or cross-model concatenation."""
    output = []
    for lag in LAGS:
        if lag > scores.shape[1] - 3:
            output.append({"lag": lag, "pairs_per_path": max(0, scores.shape[1] - lag),
                           "correlations": [None] * scores.shape[2]})
            continue
        before, after = scores[:, :-lag], scores[:, lag:]
        before = before - before.mean(axis=1, keepdims=True)
        after = after - after.mean(axis=1, keepdims=True)
        denominator = np.sqrt((before**2).sum(axis=1) * (after**2).sum(axis=1))
        correlations = np.divide((before * after).sum(axis=1), denominator,
                                 out=np.full_like(denominator, np.nan), where=denominator > 0)
        output.append({"lag": lag, "pairs_per_path": scores.shape[1] - lag,
                       "correlations": np.nanmean(correlations, axis=0).tolist()})
    return output


def _dysanos(root: Path, model: str, day: date, *, paths: int, steps: int,
              output: Path | None = None) -> tuple[np.ndarray, dict]:
    base = root / "quant_data/hedge2/runs"
    fit = load_dynamics(retained_dynamics_path(root, model, day, output=output))
    if "(16 factors)" in model and fit.directions.shape[1] != 16:
        raise ValueError("The requested sixteen-factor checkpoint has a different rank")
    directory = base / "dlv-full-v4-20261106T1130"
    config_values = json.loads((directory / "config.json").read_text())
    config = DlvConfig(**{field.name: config_values[field.name] for field in fields(DlvConfig) if field.name in config_values})
    with np.load(directory / f"days/{day}.npz", allow_pickle=False) as saved:
        state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)
    rng = np.random.default_rng(20_261_111)
    tail_sampler, tail_record = None, None
    if model.endswith("(tail)"):
        from quantdata.hedging.model_audit.tail import TailSampler, fit_tail

        if output is None:
            raise ValueError("Tail Diffusion requires its dedicated output directory")
        tail_record = fit_tail(fit, root=root, output=output, model=model)
        tail_sampler = TailSampler(fit, tail_record, paths=paths, seed=20_261_111)
    scores = np.broadcast_to(fit.current_scores, (paths, len(fit.current_scores))).copy()
    projected_spot = fit.spot_state_loading @ fit.directions
    perpendicular = fit.spot_state_loading - fit.directions @ projected_spot
    base_state = state.native - fit.directions @ fit.current_scores
    accumulated_spot = np.zeros(paths)
    natives = []
    for _ in range(steps):
        innovations = (fit.residuals[rng.integers(0, len(fit.residuals), paths)]
                       if tail_sampler is None else tail_sampler.draw())
        scores = scores @ fit.ar_matrix.T + fit.ar_intercept + innovations[:, 1:] + innovations[:, :1] * projected_spot
        accumulated_spot += innovations[:, 0]
        natives.append(base_state + scores @ fit.directions.T + accumulated_spot[:, None] * perpendicular)
    natives = np.stack(natives, axis=1).reshape(paths * steps, -1)
    surfaces, lower, upper, invalid, inversion_records = [], 0, 0, 0, []
    for begin in range(0, len(natives), 256):
        decoded, guards = _dlv_surface(state, natives[begin:begin + 256])
        surfaces.append(decoded.reshape(len(decoded), -1))
        lower += guards["lower_clips"]
        upper += guards["upper_clips"]
        invalid += int(round(guards["invalid_path_rate"] * len(decoded)))
        inversion_records.append(guards["iv_inversion"])
    ratio = float(np.linalg.norm(perpendicular) / np.linalg.norm(fit.spot_state_loading)) if np.linalg.norm(fit.spot_state_loading) > 0 else 0.
    diagnostics = {
        "lower_clip_coordinates": lower, "upper_clip_coordinates": upper,
        "invalid_surface_count": invalid, "spot_loading_outside_factor_span_fraction": ratio,
        "full_native_unit_root_from_spot_loading": bool(ratio > 1e-8),
        "iv_inversion": summarize_inversions(inversion_records)}
    if tail_record is not None:
        diagnostics["tail_diffusion"] = tail_record
    return np.concatenate(surfaces).reshape(paths, steps, -1), diagnostics


def _neural(root: Path, model: str, day: date, history, index: int,
            *, paths: int, steps: int) -> tuple[np.ndarray, dict]:
    env = BaseEnvironment(precision=64, device="cpu")
    random = torch.Generator(device="cpu").manual_seed(20_261_111)
    surfaces = []
    if model == "FuNVol":
        checkpoint_path = root / f"training/hedging/funvol-causal-chain-20261025-warning-v1/aprmay/checkpoints/{day}.pt"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = FunvolConfig()
        if saved["version"] != 2 or saved["config"] != repr(config) or saved["as_of_date"] != str(day):
            raise ValueError("FuNVol read-only checkpoint compatibility check failed")
        network = FunvolNeuralSde(env, config=config)
        network.load_state_dict(saved["model"])
        network.eval()
        last_surfaces = history.log_implied_volatilities[index - config.lag_count + 1:index + 1]
        factor_scores = saved["fpc"].encode(np.exp(last_surfaces))
        log_spot = saved["spot_log_anchor"] + np.cumsum(np.nan_to_num(np.asarray(saved["history_returns"]), nan=0.))
        states = np.column_stack((factor_scores, log_spot[-config.lag_count:]))
        center, scale = saved["state_center"], saved["state_scale"]
        context = torch.as_tensor(np.broadcast_to((states - center) / scale, (paths, *states.shape)).copy(), dtype=env.fdtype)
        with torch.no_grad():
            for _ in range(steps):
                drift, lower = network(context)
                noise = torch.randn(paths, config.state_dimension, generator=random, dtype=env.fdtype)
                next_state = context[:, -1] + drift / 252. + (lower @ noise[..., None]).squeeze(-1) / np.sqrt(252.)
                context = torch.cat((context[:, 1:], next_state[:, None]), dim=1)
                raw_state = next_state.numpy() * scale + center
                decoded = saved["fpc"].decode(raw_state[:, :config.factor_count])
                surfaces.append(np.log(decoded[:, :, COMMON_INDICES]).reshape(paths, -1))
    else:
        record_path = root / f"experiments/hedging/volgan-pca8-rolling-aprmay-20261024-v1/run/records/{day}.json"
        saved = torch.load(_record(record_path)["diagnostics"]["fit_checkpoint_path"], map_location="cpu", weights_only=False)
        config = VolganPcaModelConfig(label="volgan-short-end-nms35-pca8-2022", grid=GRID)
        if saved["model_code_version"] != config.version.unique_id48 or saved["training_end_date"] != str(day):
            raise ValueError("VolGAN read-only checkpoint compatibility check failed")
        weights = saved["generator"]
        names = ("condition_mean", "condition_scale", "target_mean", "target_scale", "level_mean",
                 "level_components", "increment_mean", "increment_components")
        network = VolganGenerator(env, config=config, **{name: weights[name] for name in names})
        network.load_state_dict(weights)
        network.eval()
        current = np.broadcast_to(history.log_implied_volatilities[index], (paths, 7, 19)).copy()
        returns = np.broadcast_to(history.forward_log_returns_into_date[index - 20:index + 1], (paths, 21)).copy()
        sessions = np.broadcast_to(history.sessions_from_previous_observation[index - 20:index + 1], (paths, 21)).copy()
        with torch.no_grad():
            for _ in range(steps):
                conditions = np.column_stack((returns[:, -1] * np.sqrt(252.), returns[:, -2] * np.sqrt(252.),
                    np.std(returns / np.sqrt(sessions), axis=1) * np.sqrt(252.), current.reshape(paths, -1)))
                condition_tensor = torch.as_tensor(conditions, dtype=env.fdtype)
                noise = torch.randn(paths, config.noise_dimension, generator=random, dtype=env.fdtype)
                decoded = network.decode_full_targets(network(noise, network.encode_full_conditions(condition_tensor))).numpy()
                current += decoded[:, 1:].reshape(paths, 7, 19)
                returns = np.column_stack((returns[:, 1:], decoded[:, 0] / np.sqrt(252.)))
                sessions = np.column_stack((sessions[:, 1:], np.ones(paths)))
                surfaces.append(current[:, :, COMMON_INDICES].reshape(paths, -1).copy())
    return np.stack(surfaces, axis=1), {"lower_clip_coordinates": 0, "upper_clip_coordinates": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--steps", type=int, default=115)
    parser.add_argument("--models", nargs="*", default=["DYSANOS ordinary", "DYSANOS spot-orthogonal", "FuNVol", "VolGAN PCA8"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-report", action="store_true")
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, arguments.output)
    with np.load(arguments.output / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    origins = [index for index, day in enumerate(history.dates) if date(2024, 4, 1) <= day <= date(2024, 5, 31)]
    started = time.monotonic()
    for model in arguments.models:
        for position, index in enumerate(origins):
            day = history.dates[index]
            path = arguments.output / "persistence" / model.replace(" ", "_") / f"{day}.json"
            if (path.exists() and not arguments.refresh
                    and json.loads(path.read_text()).get("status") in ("scored", "model_invalid")):
                continue
            try:
                if model.startswith("DYSANOS"):
                    surfaces, diagnostics = _dysanos(arguments.root, model, day, paths=arguments.paths, steps=arguments.steps, output=arguments.output)
                else:
                    surfaces, diagnostics = _neural(arguments.root, model, day, history, index,
                        paths=arguments.paths, steps=arguments.steps)
                from quantdata.hedging.model_audit.recursive import _save_arrays, summarize_paths

                record = summarize_paths(surfaces, basis)
                record["diagnostics"].update(diagnostics)
                record.update(model=model, origin_date=str(day), path_count=arguments.paths,
                    steps=arguments.steps, seed=20_261_111, fit_frozen_through_path=True)
                _save_arrays(path.with_suffix(".npz"), log_iv_surfaces=surfaces)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                record = {"model": model, "origin_date": str(day), "status": "error", "reason": str(error)}
            _json(path, record)
            print(f"persistence {model} {position+1}/{len(origins)} {day} {record['status']} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if arguments.refresh_report and (position % 5 == 4 or position == len(origins) - 1):
                from quantdata.hedging.model_audit.refresh import refresh_report

                refresh_report(arguments.output)


if __name__ == "__main__":
    main()
