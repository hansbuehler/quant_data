"""Realized fixed-contract straddle-return variance retained by causal state projections."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, fields
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.density_weighted import _load
from quantdata.hedging.model_audit.grid_shape import load_model
from quantdata.hedging.model_audit.run import _hash, _json


@dataclass(frozen=True, kw_only=True)
class ReturnBasis:
    encoder: np.ndarray
    decoder: np.ndarray
    spot_loading: np.ndarray
    label: str

    def projected_changes(self, change: np.ndarray, spot_return: float) -> np.ndarray:
        """Nested native reconstructions with actual encoded changes, not hindsight regression."""
        linked = self.spot_loading * spot_return
        scores = (change - linked) @ self.encoder
        increments = self.decoder.T * scores[:, None]
        return np.vstack((np.zeros_like(change), np.cumsum(increments, axis=0))) + linked


def fixed_straddle_values(state: DailyState, natives: np.ndarray, *, strikes: np.ndarray,
                          business_dtes: np.ndarray, spot_ratios: np.ndarray) -> tuple[np.ndarray, dict]:
    """Original cash strikes and terminal dates; expiry uses exact cash settlement."""
    spots = np.asarray(spot_ratios, dtype=float)
    if spots.shape != (len(natives),) or not np.isfinite(spots).all() or np.any(spots <= 0.):
        raise ValueError("Every scenario requires one finite positive spot ratio")
    days = np.repeat(np.asarray(business_dtes, dtype=float), strikes.shape[1])
    if np.any(days < 0.) or strikes.shape[0] != len(business_dtes):
        raise ValueError("Fixed straddle tenors and strikes must align and not precede settlement")
    decoded = decode_batch(state, natives, max_business_dte=float(np.max(days)))
    cash = strikes.reshape(-1)
    relative = cash[None] / spots[:, None]
    calls = decoded.europeans(business_dtes=days, strikes=relative)
    puts = decoded.europeans(business_dtes=days, strikes=relative, is_call=False)
    prices = spots[:, None] * (calls + puts)
    return prices.reshape(len(natives), *strikes.shape), {
        "lower_clips": int(decoded.lower_clips.sum()), "upper_clips": int(decoded.upper_clips.sum())}


def variance_capture(actual: np.ndarray, predicted: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    """One minus centered residual energy over centered return energy, across origin dates."""
    observed, fitted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if observed.shape != fitted.shape or observed.ndim != 3 or not np.isfinite(observed).all() or not np.isfinite(fitted).all():
        raise ValueError("Variance capture needs finite matching date-by-tenor-by-strike arrays")
    eligible = np.ones(observed.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if eligible.shape != observed.shape:
        raise ValueError("The origin-date strike screen must match every return coordinate")
    counts = eligible.sum(axis=0)
    enough = counts >= 2
    means = np.divide(np.sum(np.where(eligible, observed, 0.), axis=0), counts, out=np.zeros(counts.shape), where=counts > 0)
    residuals = observed - fitted
    biases = np.divide(np.sum(np.where(eligible, residuals, 0.), axis=0), counts, out=np.zeros(counts.shape), where=counts > 0)
    target_energy = float(np.sum(np.where(eligible & enough, (observed - means)**2, 0.)))
    error_energy = float(np.sum(np.where(eligible & enough, (residuals - biases)**2, 0.)))
    target_second_moment = float(np.sum(np.where(eligible, observed**2, 0.)))
    error_second_moment = float(np.sum(np.where(eligible, residuals**2, 0.)))
    return {"variance_explained": 1. - error_energy / target_energy if target_energy > 0. else None,
        "target_centered_energy": target_energy, "residual_centered_energy": error_energy,
        "zero_benchmark_energy_explained": 1. - error_second_moment / target_second_moment if target_second_moment > 0. else None,
        "residual_mean_rms": float(np.sqrt(np.mean(biases[enough]**2))) if enough.any() else None,
        "observations": int(eligible.sum()), "coordinates_with_two_observations": int(enough.sum())}


def _basis(method: str, variant: str, day: str, *, weighted: Path, split: Path,
           projections: Path, full_beta: np.ndarray) -> ReturnBasis:
    orthogonal = variant == "orthogonal"
    if method == "weighted_native":
        model = _load(weighted / f"fits/{variant}/{day}")
        fit = model.weighted_fit
        order = np.argsort(fit.explained_variance_fraction)[::-1]
        encoder = model.scale[:, None] * fit.directions[:, order]
        decoder = model.unscale_changes(fit.directions[:, order].T).T
    elif method == "grid3_shape13":
        model = load_model(split / f"fits/{variant}/{day}.npz")
        encoder, decoder = model.encoder, model.decoder
    else:
        directory = projections / f"fits/{variant}/{method}/{day}"
        fit = load_dynamics(directory / "dynamics.npz")
        with np.load(directory / "projection.npz", allow_pickle=False) as saved:
            encoder, decoder = saved["encoder"], saved["decoder"]
        if method.startswith("affine"):
            order = np.argsort(fit.explained_variance_fraction)[::-1]
            encoder, decoder = encoder @ fit.directions[:, order], decoder @ fit.directions[:, order]
    if encoder.shape != decoder.shape or encoder.shape[1] != 16:
        raise ValueError("Every retained return basis must contain exactly sixteen native factors")
    np.testing.assert_allclose(encoder.T @ decoder, np.eye(16), atol=2e-7, rtol=2e-7)
    return ReturnBasis(encoder=encoder, decoder=decoder,
        spot_loading=full_beta if orthogonal else np.zeros_like(full_beta), label=f"{variant} {method}")


def run(root: Path, source: Path, weighted: Path, split: Path, projections: Path, output: Path) -> dict:
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    raw = json.loads((directory / "config.json").read_text())
    config = DlvConfig(**{field.name: raw[field.name] for field in fields(DlvConfig) if field.name in raw})
    origins = json.loads((source / "observable_reference.json").read_text())["dates"]
    methods = ("weighted_native", "grid3_shape13", "affine200_unweighted", "affine200_probability",
               "itm_premium", "straddle_spot", "straddle_sqrt_time", "probability_changes")
    protocol = {"schema": "fixed-straddle-reconstruction-v1", "origin_dates": origins, "source": str(source),
        "weighted_fits": str(weighted), "split_fits": str(split), "projection_fits": str(projections),
        "native_source": str(directory), "source_smoothness": .1,
        "contract": "same origin cash strike and terminal expiry, one-day passage of time and actual normalized spot ratio",
        "return": "(next_price - origin_price)/(origin_spot * sqrt(origin_T_years))",
        "truth": "next day's saved decoded 0.1 source surface; 1D instruments cash settle",
        "primary_statistic": "1 - centered reconstruction-error energy / centered realized return energy",
        "second_statistic": "same statistic after removing exact frozen-surface spot/time P&L",
        "factor_count": 16, "factors_reported": list(range(17)), "rolling_training_transitions": 252,
        "factor_values": "actual next native change through the origin's saved encoder; no ex-post least-squares coefficients",
        "factor_order": "current metric PCA order; affine/native factors sorted by saved variance share; split keeps three grid factors first",
        "spot_surface_loading": "full fitted native loading retained for orthogonal models, including its part outside factor span",
        "maturity_screen": "all nine source tenors; surviving-expiry panel excludes settled 1D options",
        "strike_screen": "all 200 native interior nodes, plus a separately reported origin ATM-scaled NMS (-2,2) screen",
        "forecast_test": False, "source_code_sha256": _hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("This run directory belongs to a different straddle reconstruction protocol")
    _json(output / "PROTOCOL.json", protocol)
    inputs = {directory / "config.json", source / "observable_reference.json", weighted / "PROTOCOL.json",
              split / "PROTOCOL.json", projections / "PROTOCOL.json", Path(__file__)}

    @lru_cache(maxsize=100)
    def daily(text: str) -> tuple[np.ndarray, np.ndarray]:
        path = directory / f"days/{text}.npz"
        inputs.add(path)
        with np.load(path, allow_pickle=False) as saved:
            return saved["raw_u"], saved["support_strikes"]

    started = time.monotonic()
    actual, frozen, masks, results, stages = [], [], [], {}, {}
    for index, text in enumerate(origins):
        baseline = json.loads((weighted / "daily/DYSANOS_ordinary_(16_factors)_density-weighted" / f"{text}.json").read_text())
        current, support = daily(text)
        following, next_support = daily(baseline["endpoint"])
        state = DailyState(day=date.fromisoformat(text), native=current, reference_support=support, config=config)
        next_state = DailyState(day=date.fromisoformat(baseline["endpoint"]), native=following,
                               reference_support=next_support, config=config)
        model = _load(weighted / f"fits/orthogonal/{text}")
        full_beta = model.unscale_changes(model.weighted_fit.spot_state_loading)
        ratio = float(np.exp(baseline["actual_spot"]))
        strikes = np.broadcast_to(state.decode().pricer.support[1:-1], (len(config.business_dtes), config.num_inner_strikes))
        days = np.asarray(config.business_dtes)
        base, _ = fixed_straddle_values(state, current[None], strikes=strikes, business_dtes=days, spot_ratios=np.ones(1))
        truth, _ = fixed_straddle_values(next_state, following[None], strikes=strikes, business_dtes=days - 1., spot_ratios=np.array([ratio]))
        carry_spot, _ = fixed_straddle_values(state, current[None], strikes=strikes, business_dtes=days - 1., spot_ratios=np.array([ratio]))
        scale = np.sqrt(days[:, None] / 252.)
        target = (truth[0] - base[0]) / scale
        no_surface = (carry_spot[0] - base[0]) / scale
        packed = current.reshape(len(days), -1)
        curve_vol = np.sqrt(np.cumsum(np.exp(2. * packed[:, -1]) * np.diff(config.expiries_y, prepend=0.)) / config.expiries_y)
        nms = np.log(strikes) / (curve_vol[:, None] * np.sqrt(config.expiries_y[:, None]))
        masks.append((nms > -2.) & (nms < 2.))
        actual.append(target)
        frozen.append(no_surface)
        decoded_next, _ = fixed_straddle_values(state, following[None], strikes=strikes,
            business_dtes=days - 1., spot_ratios=np.array([ratio]))
        stages.setdefault("full_native_state", []).append((decoded_next[0] - base[0]) / scale)
        for variant in ("ordinary", "orthogonal"):
            for method in ("affine200_unweighted", "affine200_probability"):
                with np.load(projections / f"fits/{variant}/{method}/{text}/projection.npz", allow_pickle=False) as saved:
                    encoder, decoder = saved["encoder"], saved["decoder"]
                linked = full_beta * (ratio - 1.) if variant == "orthogonal" else np.zeros_like(current)
                reconstruction = current + linked + ((following - current - linked) @ encoder) @ decoder.T
                values, _ = fixed_straddle_values(state, reconstruction[None], strikes=strikes,
                    business_dtes=days - 1., spot_ratios=np.array([ratio]))
                stages.setdefault(f"{variant} {method}", []).append((values[0] - base[0]) / scale)
            for method in methods:
                model = _basis(method, variant, text, weighted=weighted, split=split,
                               projections=projections, full_beta=full_beta)
                reconstructed = current + model.projected_changes(following - current, ratio - 1.)
                values, guards = fixed_straddle_values(state, reconstructed, strikes=strikes,
                    business_dtes=days - 1., spot_ratios=np.full(len(reconstructed), ratio))
                prediction = (values - base) / scale
                results.setdefault(model.label, []).append(prediction)
                destination = output / f"daily/{model.label.replace(' ', '_')}/{text}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial.npz")
                np.savez_compressed(temporary, actual=target, frozen_spot_time=no_surface, predictions=prediction,
                    central_mask=masks[-1], lower_clips=guards["lower_clips"], upper_clips=guards["upper_clips"])
                temporary.replace(destination)
                if method == "weighted_native":
                    inputs.add(weighted / f"fits/{variant}/{text}/dynamics.npz")
                    inputs.add(weighted / f"fits/{variant}/{text}/metric.npz")
                elif method == "grid3_shape13":
                    inputs.add(split / f"fits/{variant}/{text}.npz")
                else:
                    inputs.add(projections / f"fits/{variant}/{method}/{text}/dynamics.npz")
                    inputs.add(projections / f"fits/{variant}/{method}/{text}/projection.npz")
        elapsed = time.monotonic() - started
        print(f"fixed straddles {text} completed={index+1}/{len(origins)} elapsed={elapsed:.1f}s "
              f"eta={elapsed*(len(origins)-index-1)/(index+1):.1f}s", flush=True)
    observed, spot_time, central = np.asarray(actual), np.asarray(frozen), np.asarray(masks)
    selected_masks = {"all": np.ones(observed.shape, dtype=bool), "central": central,
                      "surviving": np.broadcast_to((days > 1.)[None, :, None], observed.shape)}
    summaries = {}
    for name, predictions in results.items():
        values = np.asarray(predictions)
        summaries[name] = {}
        for screen, mask in selected_masks.items():
            summaries[name][screen] = {"frozen_spot_time": variance_capture(observed, spot_time, mask=mask),
                "by_factor_count": [{"factors": count,
                    "total": variance_capture(observed, values[:, count], mask=mask),
                    "surface": variance_capture(observed - spot_time, values[:, count] - spot_time, mask=mask)}
                    for count in range(17)]}
        summaries[name]["by_tenor"] = [{"business_dte": float(tenor),
            "spot_time": variance_capture(observed[:, row:row+1], spot_time[:, row:row+1]),
            "sixteen": variance_capture(observed[:, row:row+1], values[:, 16, row:row+1]),
            "surface_sixteen": variance_capture((observed - spot_time)[:, row:row+1], (values[:, 16] - spot_time)[:, row:row+1])}
            for row, tenor in enumerate(days)]
    _json(output / "summary.json", summaries)
    stage_summaries = {name: {screen: {"total": variance_capture(observed, np.asarray(values), mask=mask),
        "surface": variance_capture(observed - spot_time, np.asarray(values) - spot_time, mask=mask)}
        for screen, mask in selected_masks.items()} for name, values in stages.items()}
    _json(output / "pipeline_stages.json", stage_summaries)
    _json(output / "INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)])
    lines = ["# Fixed-straddle return variance explained", "",
        "All 44 April–May origins. At each origin hold the same cash strikes and terminal dates for one day, "
        "including exact settlement of 1D options. Divide dollar price changes by initial spot and sqrt(initial maturity in years). "
        "Both endpoints use the saved decoded 0.1 source surfaces. This is not the earlier log-IV forecast RMSE.", "",
        "Fractions are 1 minus centered reconstruction-error variance divided by centered realized return variance, "
        "summed across tenor/strike coordinates. Each mean is computed over the 44 dates for that coordinate. "
        "Future realized native changes provide factor values through the origin's encoder; neither the basis nor any "
        "least-squares coefficient is fitted on the next return. This measures representation, not advance forecasting.", ""]
    lines.extend(["## Affine level projection before the sixteen change factors", "",
        "| Stage | Total return variance explained | Remaining surface variance explained |",
        "|---|---:|---:|"])
    for name, groups in stage_summaries.items():
        total, surface = groups["all"]["total"]["variance_explained"], groups["all"]["surface"]["variance_explained"]
        lines.append(f"| {name} | {100.*total:.4f}% | {100.*surface:.4f}% |")
    lines.append("")
    for screen in selected_masks:
        lines.extend([f"## {screen} strike/tenor screen", "",
            "| Method | Spot/time only | +4 factors | +8 factors | +16 factors | Surface-only, 16 factors |",
            "|---|---:|---:|---:|---:|---:|"])
        for name, summary in summaries.items():
            group = summary[screen]
            fractions = [group["frozen_spot_time"]["variance_explained"],
                         *(group["by_factor_count"][count]["total"]["variance_explained"] for count in (4, 8, 16)),
                         group["by_factor_count"][16]["surface"]["variance_explained"]]
            printed = ["n/a" if value is None else f"{100.*value:.2f}%" for value in fractions]
            lines.append(f"| {name} | " + " | ".join(printed) + " |")
        lines.append("")
    lines.extend(["Full tables retain every prefix from zero through sixteen factors, daily arrays and a tenor breakdown. "
        "Surface-only percentages remove exact spot/time repricing with frozen u before measuring remaining variance. "
        "The central screen uses origin-date ATM-curve-scaled NMS strictly between -2 and 2. "
        "Surviving excludes options settling in the measured step. Native clipping counts and residual mean errors are retained. "
        "The orthogonal reconstruction preserves the complete separately fitted spot/surface loading, not just its factor projection."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return {"output": str(output), "origins": len(origins), "models": len(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    for name in ("source", "weighted", "split", "projections", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / ".experiment.lock"):
        run(arguments.root, arguments.source, arguments.weighted, arguments.split, arguments.projections, arguments.output)


if __name__ == "__main__":
    main()
