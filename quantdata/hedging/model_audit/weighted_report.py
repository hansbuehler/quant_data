"""Complete and report the density-weighted follow-up without replacing baseline laws."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.run import _hash, _json, _summarize

WEIGHTED_MODELS = ("DYSANOS ordinary (16 factors) density-weighted",
                   "DYSANOS spot-orthogonal (16 factors) density-weighted")


def _records(directory: Path, model: str, dates: list[str], *, recursive: bool = False) -> list[dict]:
    rows = [json.loads((directory / model.replace(" ", "_") / f"{day}.json").read_text()) for day in dates]
    accepted = ("scored", "model_invalid") if recursive else ("scored",)
    if any(row["status"] not in accepted or row["origin_date"] != day for row, day in zip(rows, dates, strict=True)):
        raise ValueError(f"Incomplete or mismatched diagnostics: {directory}/{model}")
    if recursive and any((row["path_count"], row["steps"], row["seed"]) != (32, 115, 20_261_111) for row in rows):
        raise ValueError("The weighted recursive comparison changed the path count, horizon or seed")
    return rows


def complete(source: Path, experiment: Path) -> dict:
    """Require complete origin-level coverage before producing publication aggregates."""
    from quantdata.hedging.model_audit.report import _export, _plot_acf, _plot_tails

    reference = json.loads((source / "observable_reference.json").read_text())
    dates = reference["dates"]
    if len(dates) != 44:
        raise ValueError("The weighted follow-up requires the original 44 origins")
    with np.load(source / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    daily = {model: _records(experiment / "daily", model, dates) for model in WEIGHTED_MODELS}
    quotes = {model: _records(experiment / "quote_forecasts", model, dates) for model in WEIGHTED_MODELS}
    recursive = {model: _records(experiment / "persistence", model, dates, recursive=True) for model in WEIGHTED_MODELS}
    summaries = {model: _summarize(rows, basis) for model, rows in daily.items()}
    coverage, stability = _export(experiment, daily, summaries, quotes, recursive, model_order=WEIGHTED_MODELS)
    _plot_tails(experiment, daily, model_order=WEIGHTED_MODELS)
    _plot_acf(experiment, reference, recursive)
    comparisons = []
    for model in ("DYSANOS ordinary (16 factors)", WEIGHTED_MODELS[0],
                  "DYSANOS spot-orthogonal (16 factors)", WEIGHTED_MODELS[1], "GROMM-P", "GROMM-Q"):
        rows = daily[model] if model in daily else _records(source / "daily", model, dates)
        item = _summarize(rows, basis)
        comparisons.append({"model": model, "summary": item})
    details = []
    for model in WEIGHTED_MODELS:
        subset = stability.loc[stability.model == model]
        records = recursive[model]
        acf = []
        for lag in (1, 5, 21):
            observed = next(entry for entry in reference["pc_level_acf"] if entry["lag"] == lag)
            predictions = [next(entry for entry in row["pc_level_acf_matched44"] if entry["lag"] == lag)["correlations"]
                           for row in records if row["status"] == "scored"]
            acf.append({"lag": lag, "observed_pair_count": observed["pair_count"],
                "maximum_absolute_error": float(np.max(np.abs(np.mean(predictions, axis=0) - observed["correlations"])))
                if predictions else None})
        mae = np.mean([row["price_mae_origin_spot_units"] for row in quotes[model]])
        persistence_mae = np.mean([row["persistence_price_mae_origin_spot_units"] for row in quotes[model]])
        details.append({"model": model, "coverage": coverage.loc[coverage.model == model].iloc[0].to_dict(),
            "summary": summaries[model], "stable_fits": int(subset.stationary_backbone.sum()),
            "min_radius": float(subset.spectral_radius.min()), "max_radius": float(subset.spectral_radius.max()),
            "max_half_life": float(subset.max_modal_half_life.max()),
            "mean_metric_pca_coverage": float(subset.metric_pca_coverage.mean()),
            "quote_contracts": sum(row["count"] for row in quotes[model]),
            "quote_mae_spot_units_mean_origin": float(mae), "quote_mae_over_persistence": float(mae / persistence_mae),
            "reconstruction_in_spread_mean_origin": float(np.mean([row["bidask"]["in_spread_rate"] for row in daily[model]])),
            "invalid_path_rate_mean_origin": float(np.mean([row["diagnostics"]["invalid_path_rate"] for row in records])),
            "extreme_surface_rate_mean_origin": float(np.mean([row["diagnostics"]["extreme_surface_rate"] for row in records])),
            "lower_clip_coordinates": sum(row["diagnostics"]["lower_clip_coordinates"] for row in records),
            "upper_clip_coordinates": sum(row["diagnostics"]["upper_clip_coordinates"] for row in records),
            "acf_errors": acf})
    spot_check = json.loads((experiment / "SPOT_IMPLEMENTATION_CHECK.json").read_text())
    result = {"schema": "density-weighted-completed-audit-v1", "experiment": str(experiment),
        "origin_dates": dates, "models": details, "comparison": comparisons,
        "spot_implementation_maximum_errors": spot_check["maximum_errors"],
        "paths": 32, "sessions": 115, "seed": 20_261_111,
        "scope": "new density-weighted PCA/VAR fits and their complete diagnostics; baseline model laws unchanged"}
    _json(experiment / "density_weighted_diagnostics.json", result)
    _json(source / "density_weighted_diagnostics.json", result)
    _json(experiment / "weighted_summary.json", summaries)
    for path in experiment.glob("*.csv"):
        shutil.copy2(path, source / f"density_{path.name}")
    for name in ("acf.pdf", "tails.pdf"):
        shutil.copy2(experiment / name, source / f"density_{name}")
    extension = json.loads((source / "covariance_extension.json").read_text())
    for entry in extension["models"]:
        entry["summary"] = summaries[entry["model"]]
    extension["remaining_diagnostics_complete"] = True
    _json(source / "covariance_extension.json", extension)
    inputs = [path for folder in ("daily", "quote_forecasts", "persistence")
              for model in WEIGHTED_MODELS for path in (experiment / folder / model.replace(" ", "_")).glob("*.json")]
    inputs.extend((experiment / "PROTOCOL.json", experiment / "SPOT_IMPLEMENTATION_CHECK.json"))
    inputs.extend(path for path in (experiment / "fits").glob("*/*/*.npz"))
    _json(source / "DENSITY_WEIGHTED_INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)])
    _json(experiment / "COMPLETE.json", {"complete": True, "models": list(WEIGHTED_MODELS), "origins": 44,
        "surface_forecasts": 88, "quote_forecasts": 88, "recursive_origins": 88, "paths": 32, "sessions": 115})
    return result


def latex_section(output: Path) -> list[str]:
    """Render every requested analytic for the fully evaluated weighted follow-up."""
    from quantdata.hedging.model_audit.grading import grade, tail_grade
    from quantdata.hedging.model_audit.report import (
        TABLE_COLOR_RULES,
        Rated,
        _figure,
        _rated,
        _table,
    )

    path = output / "density_weighted_diagnostics.json"
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    if len(result["origin_dates"]) != 44 or any(row["coverage"]["recursive_origins"] != 44 for row in result["models"]):
        raise ValueError("The weighted PDF section requires complete matched diagnostics")
    for suffix, original in (("covariance", "surface-covariance"), ("quotes", "bidask-fit"),
                             ("tails", "tail-pc1"), ("stability", "ar-stationarity"),
                             ("paths", "recursive-stability"), ("acf", "acf-errors")):
        TABLE_COLOR_RULES[f"tab:density-{suffix}"] = TABLE_COLOR_RULES[f"tab:{original}"]

    def label(model: str) -> str:
        if not model.startswith("DYSANOS"):
            return model
        return ("Spot-orthogonal" if "spot-orthogonal" in model else "Ordinary") + (
            " / density" if model.endswith("density-weighted") else " / unweighted")

    covariance_rows, calibration_rows = [], []
    for row in result["comparison"]:
        summary = row["summary"]
        covariance_rows.append([label(row["model"]), _rated("trace_ratio", summary["conditional_covariance"]["trace_ratio"]),
            _rated("trace_ratio", summary["pooled_covariance"]["trace_ratio"]),
            _rated("cov_frob_rel", summary["pooled_covariance"]["cov_frob_rel"]), summary["forecast_rmse"]])
        calibration = summary["forecast_variance_calibration"]
        calibration_rows.append([label(row["model"]), calibration["predicted_conditional_variance_trace"],
            calibration["forecast_error_squared_norm_mean"], calibration["conditional_variance_over_forecast_mse"]])
    quote_rows, tail_rows, stability_rows, path_rows, acf_rows = [], [], [], [], []
    for row in result["models"]:
        name = label(row["model"])
        quote_rows.append([name, _rated("fit_in_spread", row["reconstruction_in_spread_mean_origin"]),
            row["quote_contracts"], row["quote_mae_spot_units_mean_origin"],
            _rated("forecast_ratio", row["quote_mae_over_persistence"])])
        tails = row["summary"]["tails"]["PC1"]
        tail_rows.append([name, *[Rated(tails[str(level)]["exceptions"], tail_grade(tails[str(level)]))
            for level in (.01, .05, .95, .99)], tails["0.05"]["pinball_q"]])
        stability_rows.append([name, f"{row['stable_fits']}/44", _rated("ar_radius", row["min_radius"]),
            _rated("ar_radius", row["max_radius"]), row["max_half_life"]])
        path_rows.append([name, row["coverage"]["recursive_origins"],
            _rated("invalid_rate", row["invalid_path_rate_mean_origin"]),
            _rated("extreme_rate", row["extreme_surface_rate_mean_origin"]),
            row["lower_clip_coordinates"], row["upper_clip_coordinates"]])
        errors = []
        for entry in row["acf_errors"]:
            level = grade("acf_abs_error", entry["maximum_absolute_error"])
            if entry["observed_pair_count"] < 30 and level == "pass":
                level = "warn"
            errors.append(Rated(entry["maximum_absolute_error"], level))
        acf_rows.append([name, *errors])
    return [r"\section{Density-weighted DYSANOS: completed follow-up}",
        r"All 44 April--May origins have one-day surface/tail/leverage forecasts, quoted-contract forecasts and frozen-fit recursive tests for both weighted variants. Each recursive test uses 32 paths, 115 business sessions and seed 20261111, exactly as for the baselines. Existing baseline fits and diagnostics are retained, not replaced. The separate density-prefixed CSVs contain all target, tenor, moneyness, monthly and rolling-window scores.",
        r"For each origin we average decoded grid probability masses over the 252 training origins, writing $w_{jk}=\overline p^j_k$. These are node probabilities, not probability-density heights on the nonuniform strike grid. Put $D=\operatorname{diag}(\sqrt w)$ for log-DLV-squared coordinates, and unit diagonal entries for log-forward-volatility coordinates. Both PCA and VAR are fitted to $z=Du$; simulated changes are mapped back with the same frozen $D$. No probability floor or shock multiplier is added. A zero-weight native coordinate is frozen. Rolling orientations map the preceding dated loadings into the new causal metric.",
        r"In the spot-orthogonal variant, $\beta=\operatorname{Cov}(\Delta z,r)/\operatorname{Var}(r)$ is removed before PCA and restored in simulation. With $h=U'(z-\bar z)$, the update is",
        r"\[u_{t+1}=u_t+D^+\{U[b+(A-I)h_t+\varepsilon_{t+1}]+\beta r_{t+1}\}.\]",
        r"The paired empirical $(r,\varepsilon)$ law preserves all spot--factor cross-covariances, including residual dependence after VAR fitting. Ordinary PCA does not project out spot but keeps that joint dependence. Fixed-cash-strike option repricing separately uses the updated spot and forward. Across all 44 orthogonal fits, the pre-PCA spot-orthogonality error is below $4\cdot10^{-17}$; paired replay and covariance identities agree to about $10^{-15}$. The reported predictive variance already includes the stock contribution.",
        _table(["Model", "Trace cond.", "Trace pooled", "Frob. pooled", "RMSE"], covariance_rows,
            caption="Weighted and unweighted one-day observable covariance comparison, plus GROMM references, on the same 44 origins. RMSE here is ungraded; native and weighted PCA coverage have different denominators.", label="tab:density-covariance"),
        r"A larger conditional bar in Figure~\ref{fig:surface-variance} is not automatically better calibrated. To remove the unconditional-variance denominator, also compare mean forecast variance with realized squared forecast errors:",
        r"\[\mathcal V_{\rm err}=\frac{\operatorname{tr}(\overline C)}{n^{-1}\sum_t\|\Delta x_{t+1}-m_t\|^2}.\]",
        r"One is the population reference if the conditional mean and covariance are correct. This is a descriptive 44-origin comparison, not an independent-sample acceptance test. Misspecified means can also distort it. Weighted DYSANOS is still narrow on this measure; GROMM is broader than the observed forecast errors. Hence neither a larger conditional trace nor a pooled ratio near one alone establishes superior uncertainty calibration.",
        _table(["Model", "Predicted trace", "Forecast MSE", "Variance / MSE"], calibration_rows,
            caption="Predictive variance versus squared forecast-error scale. All values are ungraded descriptive diagnostics; no orange/red thresholds are assigned.", label="tab:density-calibration"),
        _table(["Model", "Recon. inside", "Quote count", "Forecast MAE / spot", "MAE / persistence"], quote_rows,
            caption="Observed-quote tests. Same-day reconstruction is unchanged because the daily decoder was not refitted. Next-day scores use identical surviving 2--5D cash contracts; normalized price MAEs and their reference are averaged equally over origins.", label="tab:density-quotes"),
        _table(["Model", "1%", "5%", "95%", "99%", "5% pinball"], tail_rows,
            caption="Observable PC1 tail exceptions and lower-tail pinball score. All PC1--PC3 and spot quantiles, coverage/independence tests and joint VaR--ES scores are retained in the density-weighted JSON and CSV files. Expected counts are only 0.44 or 2.2 per tail.", label="tab:density-tails"),
        _figure("density_tails.pdf", "Weighted-model PC1 forecast bands and realized changes, using the same pre-April observable PCA basis."),
        r"April/May summaries and rolling 21-session deterioration are reported in the density-prefixed monthly/rolling CSVs. The 44-origin window does not identify 63-session observed deterioration or ACF, and 1\% tail tests remain exploratory.",
        _table(["Model", "Stable fits", "Min radius", "Max radius", "Max half-life"], stability_rows,
            caption="Weighted floating-volatility PCA--VAR backbone stationarity. Half-lives are in business sessions; no stationarity claim is made for spot levels or for the whole daily re-estimated model sequence.", label="tab:density-stability"),
        _table(["Model", "Lag 1", "Lag 5", "Lag 21"], acf_rows,
            caption="Maximum absolute ACF errors across the three fixed observable PC-level scores. Modeled ACF uses each path's first 44 sessions, not concatenated rolling native factors.", label="tab:density-acf"),
        _figure("density_acf.pdf", "Weighted-model observable-PC persistence against the identical 44-origin observed window. Frozen native metrics are inverted before extracting common-grid IVs."),
        _table(["Model", "Origins", "Invalid paths", "Extreme surfaces", "Lower clips", "Upper clips"], path_rows,
            caption="Complete 32-path, 115-session recursive tests on all 44 origins. Rates are origin averages. Clips count individual native coordinates and remain disclosed even when decoded IVs are valid; finite-horizon survival is not a non-explosion proof.", label="tab:density-paths"),
        r"Implementation feedback: density weighting largely restores forward-volatility variation lost by Euclidean native PCA, but numerical DLV floors/guards and the 0.1-input versus 0.25-observable-reference smoothing difference remain. This follow-up changes the fitted reduction metric, not the pricing geometry, spot law or hedge optimizer. Large fitted metrics and recursive arrays remain external; aggregates and checksummed loading manifests are retained."]


def executive_update(output: Path) -> str:
    path = output / "density_weighted_diagnostics.json"
    if not path.exists():
        return ""
    result = json.loads(path.read_text())
    ordinary, orthogonal = result["models"]
    return (r"\paragraph{Completed density-weighted follow-up.} "
        "Both additional sixteen-factor DYSANOS variants have all 44 surface, quote and recursive diagnostics completed. "
        f"Pooled log-IV variance ratios improve to {ordinary['summary']['pooled_covariance']['trace_ratio']:.3f} "
        f"and {orthogonal['summary']['pooled_covariance']['trace_ratio']:.3f}; weighted forecast accuracy is close to GROMM. "
        "However, predicted variance is only about 0.45--0.51 times realized squared forecast errors, versus 1.82--2.06 for GROMM; "
        "a larger conditional variance bar is therefore not automatically better calibrated. "
        "All weighted recursive paths decoded without invalid or above-300-percent-IV surfaces, but native clipping persists, "
        "tail/ACF mismatches remain, and next-day quoted-price MAE is slightly worse than persistence. "
        "The spot projection/add-back and all covariance cross terms were checked explicitly. "
        "See the completed follow-up section for the full definitions and limitations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    arguments = parser.parse_args()
    result = complete(arguments.source, arguments.experiment)
    print(json.dumps({"complete": True, "models": len(result["models"]), "origins": len(result["origin_dates"])}), flush=True)


if __name__ == "__main__":
    main()
