"""Causal front-density PIT scores, retaining smooth-SANOS and recovered-DLV laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.front_spot import DiscreteFrontLaw, gaussian_scores, historical_gaussian_score
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedge2.source import SourceArchive, normalize_surface_moments
from quantdata.hedge2.state import DailyState

if TYPE_CHECKING:
    from quantdata.hedge2.dynamics import FittedDynamics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def front_score_record(*, archive: SourceArchive, licensed_fit_directory: Path,
                       daily_directory: Path, before: date, after: date, realized_spot: float,
                       previous_spot: float, probability_floor: float = 1e-7) -> dict[str, object]:
    """Evaluate only the preceding surface and carry at the next exact-session spot.

    The smooth source CDF and the recovered atomic DLV CDF are distinct.
    Neither historical score law is forced to have sample mean zero or
    variance one. Tail clipping and short-expiry extrapolation are explicit.
    """
    sessions = get_business_days(before, after) if before < after else []
    if len(sessions) != 2 or sessions[0] != before or sessions[-1] != after:
        raise ValueError("Front scores require one exact business-session transition")
    curve = licensed_curve(licensed_fit_directory, before)
    if not np.isfinite(previous_spot) or abs(curve.spot - previous_spot) > 1e-6:
        raise ValueError(f"Historical observation and licensed spot disagree on {before}")
    source, provenance = archive.load(before)
    normalization = normalize_surface_moments(source)
    parity, reference, discount = (float(value) for value in curve.at(1.))
    smooth = historical_gaussian_score(previous_source=source, realized_spot=realized_spot,
        previous_parity_forward=parity, probability_floor=probability_floor)
    decoded = DailyState.load(daily_directory, before).decode()
    if decoded.lower_clips or decoded.upper_clips:
        raise ValueError(f"Historical front state needs an unmodified valid decode on {before}")
    law = DiscreteFrontLaw.from_pricer(pricer=decoded.pricer, parity_forward=parity)
    atomic = gaussian_scores(np.array([float(law.cdf(realized_spot))]), probability_floor=probability_floor)
    inputs = {
        "source_model": {"path": provenance["source_model"], "sha256": provenance["source_sha256"]},
        "licensed_curve": {"path": str(licensed_fit_directory / "SPX" / f"{before.isoformat()}.pdq")},
        "daily_state": {"path": str(daily_directory / "days" / f"{before.isoformat()}.npz")},
    }
    for name in ("licensed_curve", "daily_state"):
        inputs[name]["sha256"] = _sha256(Path(inputs[name]["path"]))
    row = {"date": after.isoformat(), "pair_start": before.isoformat(), "spot": realized_spot,
        "previous_spot": previous_spot, "previous_parity_forward_1d": parity,
        "previous_reference_forward_1d": reference, "previous_discount_1d": discount,
        "normalized_spot_return_into_date": realized_spot / reference - 1.,
        "source_first_business_dte": float(source.expiries[0] * DAYS_PER_YEAR),
        "carry_first_business_dte": float(curve.business_dtes[0]),
        "source_short_expiry_extrapolation": bool(source.expiries[0] * DAYS_PER_YEAR > 1. + 1e-6),
        "dlv_front_mean": law.mean, "dlv_front_variance": law.variance,
        "dlv_front_relative_mean_residual": (law.mean - parity) / parity,
        "dlv_front_return_std": float(np.sqrt(law.variance) / reference),
        "dlv_realization_outside_support": bool(realized_spot < law.cash_support[0]
            or realized_spot > law.cash_support[-1]),
        "probability_floor": probability_floor, **normalization, "inputs": inputs}
    for name, result in (("sanos", smooth), ("dlv", atomic)):
        row.update({f"{name}_cdf": float(result.cdf_probabilities[0]),
            f"{name}_gaussian_score": float(result.scores[0]),
            f"{name}_lower_clipped": bool(result.lower_clipped[0]),
            f"{name}_upper_clipped": bool(result.upper_clipped[0])})
    return row


def aligned_front_scores(*, model: FittedDynamics, scores: pd.DataFrame) -> np.ndarray:
    """Match raw smooth-SANOS scores to every fitted residual row, excluding future pairs.

    Atomic DLV CDF values are intentionally not used as continuous PIT inputs.
    Historical score mean/variance are not normalized here; a future standard
    normal simulation driver must not be replaced by resampled raw scores.
    """
    frame = scores.copy()
    for name in ("date", "pair_start"):
        frame[name] = pd.to_datetime(frame[name]).dt.date
    if (frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing
            or not model.training_start < model.training_end <= model.as_of_date):
        raise ValueError("Front-score alignment needs ordered unique dates and a causal training range")
    frame = frame.loc[(frame["pair_start"] >= model.training_start) & (frame["date"] <= model.training_end)]
    if (len(frame) != len(model.residuals) or frame.empty
            or frame.iloc[0]["pair_start"] != model.training_start or frame.iloc[-1]["date"] != model.training_end):
        raise ValueError("Front-score pairs do not match every fitted innovation in the rolling window")
    sessions = {day: index for index, day in enumerate(get_business_days(model.training_start, model.training_end))}
    if any(before not in sessions or after not in sessions or sessions[after] - sessions[before] != 1
           for before, after in zip(frame["pair_start"], frame["date"], strict=True)):
        raise ValueError("Aligned front scores must represent exact business-session pairs")
    values = frame["sanos_gaussian_score"].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Every fitted innovation needs a finite smooth-SANOS Gaussian score")
    values.setflags(write=False)
    return values


def prepare_front_scores(*, archive: SourceArchive, licensed_fit_directory: Path,
                         daily_directory: Path, observations_path: Path, output_directory: Path,
                         start: date, end: date, probability_floor: float = 1e-7) -> dict[str, object]:
    """Write immutable per-pair records and a resumable score panel, never refitting SANOS."""
    if end < start:
        raise ValueError("The score range must be ordered")
    gaussian_scores(np.array([.5]), probability_floor=probability_floor)
    observations = pd.read_parquet(observations_path)
    observations["date"] = pd.to_datetime(observations["date"]).dt.date
    if observations["date"].duplicated().any() or not observations["date"].is_monotonic_increasing:
        raise ValueError("Spot observation dates must be unique and sorted")
    observations = observations.loc[observations["date"].between(start, end)].reset_index(drop=True)
    if observations.empty:
        raise ValueError("No spot observations are available in the requested range")
    sources = ("front_scores.py", "front_spot.py", "historical.py", "source.py", "state.py", "pricing.py", "dlv.py", "cash.py")
    configuration = {"version": 2, "source_root": str(archive.root),
        "licensed_fit_directory": str(licensed_fit_directory), "daily_directory": str(daily_directory),
        "observations_path": str(observations_path), "observations_sha256": _sha256(observations_path),
        "daily_configuration_sha256": _sha256(daily_directory / "config.json"),
        "start": start.isoformat(), "end": end.isoformat(), "probability_floor": probability_floor,
        "implementation_sha256": {name: _sha256(Path(__file__).parent / name) for name in sources},
        "score_convention": "raw preceding-front CDF mapped by inverse standard normal; clipping flags retained",
        "marginal_convention": "smoothness-0.1 SANOS and recovered zero-backbone DLV recorded separately",
        "historical_dependence_input": "smooth-SANOS Gaussian scores; atomic DLV scores are diagnostic, not uniform PITs",
        "source_moment_treatment": "recorded source mass/mean roundoff normalization before CDF evaluation; no refit",
        "license": "private derived diagnostics; no redistribution authorization"}
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("The score output belongs to different inputs or implementation")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    (output_directory / "days").mkdir(exist_ok=True)
    sessions = {day: index for index, day in enumerate(get_business_days(start, end))}
    pairs = [(observations.iloc[index - 1], observations.iloc[index]) for index in range(1, len(observations))
        if sessions[observations.iloc[index]["date"]] - sessions[observations.iloc[index - 1]["date"]] == 1]
    rows = []
    started = last_log = time.monotonic()
    for previous, current in pairs:
        before, after = previous["date"], current["date"]
        path = output_directory / "days" / f"{after.isoformat()}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if (row["pair_start"] != before.isoformat() or row["date"] != after.isoformat()
                    or row["spot"] != float(current["spot"]) or row["previous_spot"] != float(previous["spot"])):
                raise ValueError(f"Cached front-score pair differs from observations: {after}")
            for item in row["inputs"].values():
                if _sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError(f"Cached front-score input changed: {item['path']}")
        else:
            row = front_score_record(archive=archive, licensed_fit_directory=licensed_fit_directory,
                daily_directory=daily_directory, before=before, after=after,
                realized_spot=float(current["spot"]), previous_spot=float(previous["spot"]),
                probability_floor=probability_floor)
            temporary = path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        rows.append(row)
        now = time.monotonic()
        if len(rows) % 25 == 0 or now - last_log >= 60. or len(rows) == len(pairs):
            elapsed = now - started
            print(f"front scores {len(rows)}/{len(pairs)} date={after} elapsed_s={elapsed:.1f} "
                f"eta_s={elapsed / len(rows) * (len(pairs) - len(rows)):.1f}", flush=True)
            last_log = now
    frame = pd.DataFrame([{name: value for name, value in row.items() if name != "inputs"} for row in rows])
    if frame.empty:
        raise ValueError("No exact-session pairs are available in the requested range")
    temporary = output_directory / "scores.partial.parquet"
    frame.to_parquet(temporary, index=False)
    temporary.replace(output_directory / "scores.parquet")
    result = {"pairs": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "source_short_expiry_extrapolations": int(frame["source_short_expiry_extrapolation"].sum()),
        "dlv_realizations_outside_support": int(frame["dlv_realization_outside_support"].sum()),
        "maximum_absolute_front_relative_mean_residual": float(frame["dlv_front_relative_mean_residual"].abs().max()),
        "elapsed_seconds": time.monotonic() - started,
        "score_statistics_are_historical_not_a_test_of_risk_neutral_sampling": True}
    for name in ("sanos", "dlv"):
        result[name] = {"mean": float(frame[f"{name}_gaussian_score"].mean()),
            "std_population": float(frame[f"{name}_gaussian_score"].std(ddof=0)),
            "lower_clipped": int(frame[f"{name}_lower_clipped"].sum()),
            "upper_clipped": int(frame[f"{name}_upper_clipped"].sum())}
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--licensed-fit-directory", type=Path, required=True)
    parser.add_argument("--daily-directory", type=Path, required=True)
    parser.add_argument("--observations-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    arguments = vars(parser.parse_args())
    arguments["archive"] = SourceArchive(root=arguments.pop("source_root"))
    print(json.dumps(prepare_front_scores(**arguments), indent=2), flush=True)


if __name__ == "__main__":
    main()
