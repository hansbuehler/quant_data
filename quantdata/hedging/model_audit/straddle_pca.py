"""Price-only rolling PCA covariance forecasts of fixed-instrument straddle gains."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedge2.source import SourceArchive, normalize_surface_moments, surface_time_values
from quantdata.hedging.model_audit.run import _hash, _json, build_history
from quantdata.hedging.model_audit.straddle_explained import variance_capture
from quantdata.hedging.model_audit.tail import _training_pairs
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel


@dataclass(frozen=True, kw_only=True)
class ReturnPCA:
    mean: np.ndarray
    directions: np.ndarray
    eigenvalues: np.ndarray
    full_variance_diagonal: np.ndarray

    def variance_diagonal(self, rank: int) -> np.ndarray:
        if not 0 < rank <= len(self.eigenvalues):
            raise ValueError("The requested return covariance rank is unavailable")
        return (self.directions[:, :rank]**2) @ self.eigenvalues[:rank]

    def reconstruct(self, values: np.ndarray, rank: int) -> np.ndarray:
        directions = self.directions[:, :rank]
        return self.mean + ((values - self.mean) @ directions) @ directions.T


def fit_return_pca(values: np.ndarray, *, rank: int = 16, previous: ReturnPCA | None = None) -> ReturnPCA:
    """Closed-form sample covariance PCA; no DLV coordinates or temporal AR model."""
    samples = np.asarray(values, dtype=float)
    if (samples.ndim != 2 or len(samples) <= rank or not 0 < rank <= samples.shape[1]
            or not np.isfinite(samples).all()):
        raise ValueError("Return PCA needs finite training rows and a valid rank")
    mean = samples.mean(axis=0)
    centered = samples - mean
    spectrum, vectors = np.linalg.eigh(centered @ centered.T / (len(samples) - 1.))
    order = np.argsort(spectrum)[::-1][:rank]
    eigenvalues = spectrum[order]
    if eigenvalues[-1] <= np.finfo(float).eps * max(float(spectrum.max()), np.finfo(float).tiny):
        raise ValueError("The requested return PCA rank is numerically unidentified")
    directions = centered.T @ vectors[:, order] / np.sqrt((len(samples) - 1.) * eigenvalues)
    if previous is not None:
        if previous.directions.shape != directions.shape:
            raise ValueError("Previous observable PCA geometry is incompatible")
        signs = np.sign(np.sum(previous.directions * directions, axis=0))
    else:
        signs = np.sign(directions[np.argmax(np.abs(directions), axis=0), np.arange(rank)])
    directions *= np.where(signs == 0., 1., signs)
    return ReturnPCA(mean=mean, directions=directions, eigenvalues=eigenvalues,
                     full_variance_diagonal=np.sum(centered**2, axis=0) / (len(samples) - 1.))


def source_straddles(pricer: ArbFreePricer, *, business_dtes: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Vector SANOS market interpolation, with exact zero-time settlement."""
    days, cash = np.asarray(business_dtes, dtype=float), np.asarray(strikes, dtype=float)
    if (days.ndim != 1 or cash.ndim != 1 or np.any(days < 0.) or np.any(cash <= 0.)
            or not np.isfinite(days).all() or not np.isfinite(cash).all()):
        raise ValueError("Straddle tenors must be nonnegative and strikes finite positive vectors")
    intrinsic = np.abs(1. - cash)
    values = np.broadcast_to(intrinsic, (len(days), len(cash))).copy()
    alive = days > 0.
    if alive.any():
        values[alive] += 2. * surface_time_values(pricer, expiries_y=days[alive] / 252., strikes=cash)
    return values


def select_window(pairs: tuple, as_of: date, *, count: int = 252) -> list:
    selected = [pair for pair in pairs if pair[1] <= as_of][-count:]
    if len(selected) != count:
        raise ValueError(f"Insufficient exact one-session training pairs through {as_of}")
    return selected


def forecast_summary(means: np.ndarray, variances: np.ndarray, actual: np.ndarray) -> dict:
    """Calibration ratios, not fractions of explained variance; values may exceed one."""
    if (means.shape != actual.shape or variances.shape != actual.shape or actual.ndim != 2
            or len(actual) < 2 or np.any(variances < 0.)
            or not all(np.isfinite(value).all() for value in (means, variances, actual))):
        raise ValueError("Forecast mean, variance diagonal and realised returns must align")
    predicted = float(np.mean(variances.sum(axis=1)))
    mse = float(np.mean(np.sum((actual - means)**2, axis=1)))
    observed = float(np.sum(np.var(actual, axis=0, ddof=1)))
    pooled = predicted + float(np.sum(np.var(means, axis=0, ddof=1)))
    if min(mse, observed) <= 0.:
        raise ValueError("Variance calibration requires positive realised variation")
    return {"origins": len(actual), "predicted_conditional_variance_trace": predicted,
        "realized_forecast_error_squared_norm_mean": mse, "variance_over_forecast_mse": predicted / mse,
        "observed_centered_variance_trace": observed, "pooled_predicted_variance_trace": pooled,
        "pooled_over_observed_variance": pooled / observed,
        "conditional_over_unconditional_variance": predicted / observed,
        "forecast_rmse": float(np.sqrt(np.mean((actual - means)**2)))}


def run(root: Path, source: Path, output: Path) -> dict:
    book = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    origins = json.loads((source / "observable_reference.json").read_text())["dates"]
    pairs = _training_pairs(root)
    windows = {text: select_window(pairs, date.fromisoformat(text)) for text in origins}
    endpoints = {str(pair[0]): pair for pair in pairs}
    required = {pair for window in windows.values() for pair in window}
    required.update(endpoints[text] for text in origins)
    initial = origins[0]
    with np.load(book / f"days/{initial}.npz", allow_pickle=False) as saved:
        archive = SourceArchive(root=Path(str(saved["source_model"])).parents[4], smoothness=.1)
        days = saved["expiries_y"] * 252.
    panel = PublicHedgingPanel(root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, source)
    spot_returns = {day: float(np.expm1(value)) for day, value in
                    zip(history.dates, history.forward_log_returns_into_date, strict=True)}
    protocol = {"schema": "price-only-straddle-pca-covariance-v1", "origins": origins,
        "training_pairs": 252, "ranks": [1, 2, 4, 8, 16], "source_smoothness": .1,
        "market_archive": str(archive.root), "observable_reference": str(source),
        "spot_series": "same public-panel forward-adjusted returns as the preceding audit; not the separate source observations series",
        "input": "SANOS market straddle prices only; no u, DLV probabilities, DLV pricing, native factors or VAR",
        "contract_book": "same nine-by-200 cash-strike book, read as instrument identifiers only",
        "return": "fixed-contract one-day price gain divided by initial spot and sqrt(initial maturity in years)",
        "across_pairs": "floating book labels across days; each one-day contract keeps its cash strike and expiry",
        "controls": "truncated PCA covariance, full empirical covariance, PCA with discarded diagonal variance retained",
        "mean_forecast": "causal 252-sample mean; no future centering",
        "covariance_ddof": 1, "dynamics": "rolling covariance forecast; no AR imposed on changing contracts",
        "price_panel_future_allowed_in_fit": False, "source_code_sha256": _hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Output directory belongs to a different direct-return covariance protocol")
    _json(output / "PROTOCOL.json", protocol)
    inputs = {Path(__file__), source / "observable_reference.json", panel.dataset.directory / "manifest.json",
              panel.dataset.directory / "observations.parquet"}

    @lru_cache(maxsize=400)
    def market(day: date):
        pricer, metadata = archive.load(day)
        normalize_surface_moments(pricer)
        inputs.add(Path(metadata["source_model"]))
        inputs.add(Path(metadata["source_model"]).with_name("certificate.json"))
        return pricer

    started = time.monotonic()
    for index, pair in enumerate(sorted(required)):
        start, end, _ = pair
        destination = output / f"price_returns/{start}.npz"
        path = book / f"days/{start}.npz"
        end_path = book / f"days/{end}.npz"
        source_paths = [archive.model_path(point) for point in (start, end)]
        identities = [_hash(path), *(_hash(item) for item in source_paths)]
        inputs.update((path, end_path, *source_paths, *(item.with_name("certificate.json") for item in source_paths)))
        ratio = 1. + spot_returns[end]
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if (saved["source_hashes"].tolist() != identities or float(saved["spot_ratio"]) != ratio
                        or str(saved["endpoint"]) != str(end)):
                    raise ValueError(f"Cached price-only sample input identity changed for {start}")
        if not destination.exists():
            with np.load(path, allow_pickle=False) as saved:
                strikes = saved["support_strikes"][1:-1]
                if float(saved["source_smoothness"]) != .1:
                    raise ValueError("The matched instrument book is not the 0.1 market reference")
            inputs.add(path)
            initial_market, next_market = market(start), market(end)
            origin_price = source_straddles(initial_market, business_dtes=days, strikes=strikes)
            next_price = ratio * source_straddles(next_market, business_dtes=days - 1., strikes=strikes / ratio)
            frozen_price = ratio * source_straddles(initial_market, business_dtes=days - 1., strikes=strikes / ratio)
            scale = np.sqrt(days[:, None] / 252.)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".partial.npz")
            np.savez_compressed(temporary, total=(next_price - origin_price) / scale,
                surface=(next_price - frozen_price) / scale, endpoint=str(end), spot_ratio=ratio,
                business_dtes=days, origin_strikes=strikes, source_hashes=np.asarray(identities))
            temporary.replace(destination)
        if (index + 1) % 10 == 0 or index + 1 == len(required):
            elapsed = time.monotonic() - started
            print(f"direct straddle price panel completed={index+1}/{len(required)} elapsed={elapsed:.1f}s "
                  f"eta={elapsed*(len(required)-index-1)/(index+1):.1f}s", flush=True)
    pricing_seconds = time.monotonic() - started

    @lru_cache(maxsize=400)
    def returns(day: date, target: str) -> np.ndarray:
        with np.load(output / f"price_returns/{day}.npz", allow_pickle=False) as saved:
            return saved[target].reshape(-1)

    records, previous = {}, {}
    fit_times = []
    observed = {target: np.stack([returns(date.fromisoformat(text), target) for text in origins])
                for target in ("total", "surface")}
    for position, text in enumerate(origins):
        for target in ("total", "surface"):
            selected = windows[text]
            samples = np.stack([returns(pair[0], target) for pair in selected])
            started_fit = time.perf_counter()
            fit = fit_return_pca(samples, previous=previous.get(target))
            fit_times.append(time.perf_counter() - started_fit)
            previous[target] = fit
            destination = output / f"fits/{target}/{text}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".partial.npz")
            np.savez_compressed(temporary, mean=fit.mean, directions=fit.directions, eigenvalues=fit.eigenvalues,
                full_variance_diagonal=fit.full_variance_diagonal, as_of_date=text,
                training_start=str(selected[0][0]), training_end=str(selected[-1][1]),
                initialisation_date=origins[position-1] if position else "")
            temporary.replace(destination)
            actual = observed[target][position]
            for rank in (1, 2, 4, 8, 16):
                label = f"{target} PCA{rank}"
                records.setdefault(label, []).append({"mean": fit.mean, "diagonal": fit.variance_diagonal(rank),
                    "reconstructed": fit.reconstruct(actual, rank),
                    "training_share": float(fit.eigenvalues[:rank].sum() / fit.full_variance_diagonal.sum())})
            for label in ("full covariance", "PCA8+diagonal", "PCA16+diagonal"):
                records.setdefault(f"{target} {label}", []).append({"mean": fit.mean,
                    "diagonal": fit.full_variance_diagonal, "training_share": 1.})
        elapsed = time.monotonic() - started
        print(f"direct straddle PCA origin={text} completed={position+1}/{len(origins)} elapsed={elapsed:.1f}s", flush=True)
    summaries = {}
    for label, values in records.items():
        target = label.split()[0]
        means = np.stack([value["mean"] for value in values])
        diagonals = np.stack([value["diagonal"] for value in values])
        summary = forecast_summary(means, diagonals, observed[target])
        summary["mean_training_variance_share"] = float(np.mean([value["training_share"] for value in values]))
        if "reconstructed" in values[0]:
            reconstructed = np.stack([value["reconstructed"] for value in values])
            summary["held_out_representation"] = variance_capture(observed[target].reshape(-1, len(days), 200),
                reconstructed.reshape(-1, len(days), 200))
        summaries[label] = summary
    _json(output / "summary.json", summaries)
    _json(output / "TIMINGS.json", {"new_price_panel_seconds": pricing_seconds, "price_pairs": len(required),
        "median_return_pca_seconds": float(np.median(fit_times)), "fitted_origins": len(origins),
        "median_timing_scope": "252 resident price-return rows, sixteen eigenvectors; no surface valuation or I/O"})
    existing_inputs = sorted(path for path in inputs if path.exists())
    _json(output / "INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in existing_inputs])
    lines = ["# Direct straddle-return PCA: forecast baseline", "",
        "Price-only baseline using saved 0.1 SANOS market interpolation, not a DLV/native state. "
        "The same nine-by-200 book rolls between origins but each measured daily contract keeps its cash strike and terminal date. "
        "The spot series matches the preceding public-panel audit. Every origin uses 252 complete transitions ending through that origin.", "",
        "Forecast variance/error compares mean predicted covariance trace with mean squared held-out error about the causal sample mean. "
        "Pooled/observed adds across-origin forecast-mean variance and divides by centered realized variance. "
        "Both ratios have a 100% reference, can exceed 100%, and are not explained-variance scores. "
        "The representation column instead uses realized factor values to reconstruct the untouched next return.", "",
        "| Target and baseline | Training variance share | Held-out representation | Forecast variance/error | Pooled/observed variance |",
        "|---|---:|---:|---:|---:|"]
    for label, summary in summaries.items():
        representation = summary.get("held_out_representation", {}).get("variance_explained")
        score = "n/a" if representation is None else f"{100.*representation:.2f}%"
        lines.append(f"| {label} | {100.*summary['mean_training_variance_share']:.2f}% | {score} | "
            f"{100.*summary['variance_over_forecast_mse']:.2f}% | {100.*summary['pooled_over_observed_variance']:.2f}% |")
    lines.extend(["", "Total and surface targets are fitted separately. Surface gains remove exact source-market "
        "frozen-surface spot/time repricing, including zero surface risk after 1D settlement. "
        "The PCA-plus-diagonal controls keep discarded idiosyncratic variance, so their trace equals the full empirical covariance. "
        "These are covariance forecasts, not an arbitrage-free surface generator or a fitted AR model. "
        "Only 44 held-out origins are available; ratios are descriptive short-window estimates, not pass/fail conclusions."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return {"output": str(output), "origins": len(origins), "price_pairs": len(required)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / ".experiment.lock"):
        run(arguments.root, arguments.source, arguments.output)


if __name__ == "__main__":
    main()
