"""Compile a reproducible PDF, tables and model-choice appendix for the audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantdata.hedging.model_audit.grading import RULES, grade, tail_grade
from quantdata.hedging.model_audit.run import _hash, _json, _summarize

MODEL_ORDER = ("DYSANOS ordinary", "DYSANOS spot-orthogonal", "FuNVol", "VolGAN PCA8",
               "AD-Seq-Vol", "AD-Seq-Vol-FT", "GROMM-P", "GROMM-Q",
               "DYSANOS ordinary (tail)", "DYSANOS spot-orthogonal (tail)")

COVARIANCE_COLORS = (
    "Frobenius error: black at most 0.25, orange above 0.25 through 0.50, red above 0.50. "
    "Trace ratio: black 0.80--1.25; orange 0.50--2.00 outside that band; red outside 0.50--2.00. "
    "These are descriptive review tolerances, not statistical acceptance tests."
)
TABLE_COLOR_RULES = {
    "tab:audit-coverage": "Black: all 44 requested origins evaluated. Orange: fewer than 44. No red coverage grade; coverage is not model quality.",
    "tab:surface-covariance": COVARIANCE_COLORS,
    "tab:matched-covariance": COVARIANCE_COLORS,
    "tab:forecast-subspace": (
        "RMSE: black at most the persistence RMSE, orange above persistence through 1.10 times persistence, red above 1.10 times. "
        "Maximum angle: black at most 15 degrees, orange above 15 through 30, red above 30. "
        "Eigen-share L1 error: black at most 0.10, orange above 0.10 through 0.25, red above 0.25."),
    "tab:spot-leverage": "Absolute correlation error versus the observed reference: black at most 0.10, orange above 0.10 through 0.20, red above 0.20. Reference correlations and counts are neutral.",
    "tab:bidask-fit": (
        "Same-day in-spread share: black at least 95%, orange 80% to below 95%, red below 80%. "
        "Forecast MAE / same-contract persistence MAE: black at most 1.00, orange above 1.00 through 1.10, red above 1.10. "
        "Other forecast statistics are ungraded; next-day in-spread share is not a reconstruction test."),
    "tab:monthly-errors": (
        "Monthly RMSE / monthly persistence RMSE: black at most 1.00, orange above 1.00 through 1.10, red above 1.10. "
        "May/April persistence-normalized deterioration: black at most 1.10, orange above 1.10 through 1.25, red above 1.25. Counts are neutral."),
    "tab:tail-pc1": (
        "Exception-count colors use pointwise two-sided exact-binomial coverage p-values: red below 0.01; "
        "orange from 0.01 to below 0.05, or whenever expected exceptions are fewer than 10; "
        "black requires p at least 0.05 and at least 10 expected exceptions. Pinball loss is ungraded."),
    "tab:acf-errors": (
        "Maximum absolute PC-ACF error: black at most 0.10, orange above 0.10 through 0.20, red above 0.20. "
        "Fewer than 30 observed lag pairs forces at least orange, even for a small error."),
    "tab:ar-stationarity": "AR spectral radius: black below 0.99, orange from 0.99 to below 1, red at least 1. Stable-fit counts and half-lives are ungraded.",
    "tab:recursive-stability": (
        "Extreme-surface rate (any common-grid IV above 300%): black zero, orange above zero through 0.1%, red above 0.1%. "
        "Invalid-path rate: black zero, orange above zero through 1%, red above 1%. Evaluated-origin counts are neutral."),
    "tab:variance-reconciliation": "Black: ungraded diagnostic quantities. Orange and red grades are not assigned; this ablation is not a calibration pass/fail test.",
    "tab:rank-comparison": COVARIANCE_COLORS + " AR spectral radius: black below 0.99, orange from 0.99 to below 1, red at least 1. Native coverage and counts are ungraded.",
    "tab:rank-reconciliation": "Black: ungraded same-shock ablation quantities. Orange and red grades are not assigned; variance retention is not a calibration pass/fail test.",
}


@dataclass(frozen=True)
class Rated:
    value: object
    level: str


def _rated(metric: str, value: float | None) -> Rated:
    return Rated(value, grade(metric, value))


def _rows(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _completed_recursive(rows: list[dict]) -> list[dict]:
    """Count evaluated model-law failures as completed tests, never as passes."""
    return [row for row in rows if row.get("status") in ("scored", "model_invalid")]


def _rank_comparison_section(output: Path) -> list[str]:
    """Include only a completed, separately identified controlled rank experiment."""
    path = output / "rank_comparison.json"
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    if any(row["origins"] != 44 or row["recursive_origins"] != 44 for row in result["models"]):
        raise ValueError("The published rank comparison does not cover every origin")
    rows = [[f"{'Spot-orthogonal' if 'spot-orthogonal' in row['model'] else 'Ordinary'} / {row['rank']}", row["origins"], 100 * row["native_coverage_mean"],
        _rated("trace_ratio", row["conditional_trace_ratio"]), _rated("trace_ratio", row["pooled_trace_ratio"]),
        _rated("cov_frob_rel", row["pooled_covariance_error"]), _rated("ar_radius", row["spectral_radius_max"])]
        for row in result["models"]]
    reconciliation = []
    for record in result["reconciliation"]:
        rank = 16 if record["model"].endswith("(16 factors)") else 8
        groups = record["groups"]
        name = "Spot-orthogonal" if "spot-orthogonal" in record["model"] else "Ordinary"
        reconciliation.append([f"{name} / {rank}", record["origin_date"],
            100 * groups[f"pca{rank}_centered_changes"]["native_energy_fraction_of_full"],
            100 * groups[f"pca{rank}_centered_changes"]["decoded_variance_fraction_of_full"],
            100 * groups[f"pca{rank}_ar_innovations"]["decoded_variance_fraction_of_full"]])
    return [r"\subsection{Controlled eight-versus-sixteen-factor DYSANOS comparison}",
        r"Only the retained rank changes. Both variants use the same 44 origins, 252 exact causal training transitions, empirical paired spot innovations, native states, decoder guards and vector cdxcore IV inversion. Each new rolling basis is aligned with its preceding origin; March 28 supplies a causal alignment seed before scoring starts. The old eight-dimensional subspace is verified to lie inside the new sixteen-dimensional span at every origin. No shock inflation or tail multiplier is introduced. The separate recursive tests retain the common 32-path, 115-session protocol. This does not resolve the source/reference smoothness difference or change the native density fit.",
        _table(["Variant / rank", "Days", "Native %", "Trace cond.", "Trace pooled", "Frob. pooled", "Max radius"], rows,
            caption="Controlled rank comparison on all 44 origins. Native coverage is averaged across origins; decoded covariance is compared against the unchanged observable reference. Counts, native coverage and covariance scale are distinct quantities.", label="tab:rank-comparison"),
        _table(["Variant / rank", "Origin", "Native retained %", "PCA IV %", "AR innovation IV %"], reconciliation,
            caption="Eight- and sixteen-factor same-shock ablations. Percentages use the same full native-shock comparator at each origin. Decoder guards remain active, so this is not an attribution to linear PCA alone.", label="tab:rank-reconciliation"),
        r"Full rank-comparison results, including PC1 tail counts, recursive invalid/extreme rates and decoder clipping counts, are retained in \texttt{rank\_comparison.json}. Native energy retention and a stable AR matrix do not by themselves establish observable covariance calibration."]


def _escape(value: object) -> str:
    text = str(value)
    for before, after in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(before, after)
    return text


def _format(value: object, decimals: int = 3) -> str:
    if isinstance(value, Rated):
        color = {"pass": "black", "warn": "orange", "fail": "red"}[value.level]
        return f"\\textcolor{{{color}}}{{{_format(value.value, decimals)}}}"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, float | np.floating):
        return f"{value:.6f}" if 0 < abs(value) < .001 else f"{value:.{decimals}f}"
    return _escape(value)


def _table(headers: list[str], rows: list[list[object]], *, caption: str, label: str) -> str:
    columns = "l" + "r" * (len(headers) - 1)
    rules = TABLE_COLOR_RULES.get(label, "Black: ungraded entries. No orange or red grades are assigned.")
    caption_text = _escape(caption + " Color rules: " + rules + " Missing graded entries are orange; other raw quantities are ungraded.")
    body = [r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}", f"\\caption{{{caption_text}}}\\label{{{label}}}",
            f"\\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
            " & ".join(_escape(header) for header in headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(_format(value) for value in row) + r" \\" for row in rows)
    return "\n".join(body + [r"\bottomrule\end{tabular}\end{table}"])


def _figure(path: str, caption: str) -> str:
    label = r"\label{fig:surface-variance}" if path == "covariance.pdf" else ""
    return f"\\begin{{figure}}[H]\\centering\\includegraphics[width=\\linewidth]{{{path}}}\\caption{{{caption}}}{label}\\end{{figure}}"


def _executive_summary(summaries: dict, coverage: pd.DataFrame, stability: pd.DataFrame) -> str:
    """Summarize the current evidence without equating diagnostic coverage with a pass."""
    expected = int(coverage.requested_origins.sum())
    counts = {name: int(coverage[name].sum()) for name in
              ("surface_origins", "quote_forecast_origins", "recursive_origins")}
    complete = all(count == expected for count in counts.values())
    coverage_status = ("All requested origin-level diagnostics are complete; completion does not imply a model-quality pass."
                       if complete else "Coverage is still being completed; unfinished diagnostics are not passes.")
    invalid_origins = int(coverage.get("recursive_invalid_origins", pd.Series(dtype=int)).sum())
    if invalid_origins:
        coverage_status += f" {invalid_origins} completed origin tests have no wholly valid decoded IV paths; these are failures, not missing tests."
    ordinary = summaries[MODEL_ORDER[0]]["pooled_covariance"]["trace_ratio"]
    orthogonal = summaries[MODEL_ORDER[1]]["pooled_covariance"]["trace_ratio"]
    volgan = summaries["VolGAN PCA8"]
    base_stability = (stability.loc[stability.model.isin(MODEL_ORDER[:2])]
                      if "model" in stability else stability)
    stable = int(base_stability.stationary_backbone.sum())
    radius = float(base_stability.spectral_radius.max())
    return "\n".join([
        r"\section*{Executive summary}",
        r"\textbf{Purpose.} Assess the surface dynamics of retained rolling models on 44 April--May 2024 forecast origins, using one common log-IV grid and a PCA basis fixed before April. This is a model-law audit, not a rerun of hedging performance or a proof of no arbitrage.",
        r"\begin{itemize}",
        f"\\item \\textbf{{DYSANOS: stationary backbone, weak observable dispersion.}} All {stable}/{len(base_stability)} distinct saved floating-volatility PCA--AR(1) backbones are stationary (maximum spectral radius {radius:.3f}). Nevertheless, pooled modeled log-IV-change variance is only {100*ordinary:.1f}\\% of observed variance for ordinary PCA and {100*orthogonal:.1f}\\% for spot-orthogonal PCA. Stationarity is not the same as calibration.",
        f"\\item \\textbf{{VolGAN: aggregate agreement hides conditional collapse.}} Pooled modeled variance is {100*volgan['pooled_covariance']['trace_ratio']:.1f}\\% of observed variance, but average within-origin forecast variance is only {100*volgan['conditional_covariance']['trace_ratio']:.3f}\\%. A pooled variance ratio near one can therefore be misleading.",
        r"\item \textbf{Judge more than variance scale.} Compare covariance shape, spot leverage, forecast skill relative to persistence, and recursive behavior. A trace ratio near one is not an overall pass, and the conditional ratio need not equal one because its observed denominator is unconditional. GROMM-Q comparisons with historical P outcomes are descriptive, not P-calibration requirements.",
        r"\item \textbf{Tail conclusions remain preliminary.} Forty-four realized dates imply only 0.44 expected exceptions at a 1\% tail and 2.2 at a 5\% tail. Non-rejection cannot establish tail calibration. Stable AR coefficients and finite simulated paths do not prove non-explosion of the complete decoded model.",
        f"\\item \\textbf{{Coverage at this report build.}} Completed model--origin checks: surface forecasts {counts['surface_origins']}/{expected}; quote forecasts {counts['quote_forecast_origins']}/{expected}; recursive path tests {counts['recursive_origins']}/{expected}. {coverage_status}",
        r"\end{itemize}",
        r"\textbf{Interpretation.} These are diagnostics of retained implementations, including disclosed numerical guards and differing SANOS input smoothnesses, not a controlled architecture-only comparison. Black/orange/red colors use the explicit review tolerances in each table caption; they are not universal statistical acceptance criteria.",
        r"\clearpage",
    ])


def _plot_covariance(output: Path, summaries: dict) -> None:
    labels = [model for model in MODEL_ORDER if summaries.get(model, {}).get("surface_days", 0) >= 2]
    conditional = [summaries[model]["conditional_covariance"]["trace_ratio"] for model in labels]
    pooled = [summaries[model]["pooled_covariance"]["trace_ratio"] for model in labels]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(positions - .18, conditional, .36, label="Average conditional variance / observed variance")
    axis.bar(positions + .18, pooled, .36, label="Pooled variance / observed variance")
    axis.axhline(1., color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Trace ratio (log scale)")
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "covariance.pdf")
    plt.close(figure)


def _plot_tails(output: Path, records: dict) -> None:
    figure, axes = plt.subplots((len(MODEL_ORDER) + 1) // 2, 2,
                               figsize=(11, 2.5 * ((len(MODEL_ORDER) + 1) // 2)), sharex=False)
    for axis, model in zip(axes.ravel(), MODEL_ORDER, strict=True):
        rows = [row for row in records[model] if row.get("status") == "scored"]
        if rows:
            positions = np.arange(len(rows))
            quantiles = np.array([row["quantiles"] for row in rows])[:, :, 0]
            actual = [row["actual_targets"][0] for row in rows]
            axis.fill_between(positions, quantiles[:, 0], quantiles[:, 3], color="#9ecae1", label="1--99% forecast")
            axis.fill_between(positions, quantiles[:, 1], quantiles[:, 2], color="#3182bd", alpha=.6, label="5--95% forecast")
            axis.plot(positions, actual, color="#a63603", marker=".", linewidth=.8, label="Observed")
        axis.set_title(textwrap.fill(model + (" (not P-calibration)" if model == "GROMM-Q" else ""), width=38), fontsize=11)
        axis.set_ylabel("PC1 log-IV change")
        axis.set_xlabel("Scored origin index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output / "tails.pdf")
    plt.close(figure)


def _plot_acf(output: Path, reference: dict, persistence: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    for component, axis in enumerate(axes):
        observed = [row for row in reference["pc_level_acf"] if row["lag"] <= 21]
        axis.plot([row["lag"] for row in observed], [row["correlations"][component] for row in observed],
                  color="black", marker="o", linewidth=2, label="Observed 44-date window")
        for model, records in persistence.items():
            scored = [row for row in records if row.get("status") == "scored" and "pc_level_acf_matched44" in row]
            if not scored:
                continue
            values = np.array([[entry["correlations"][component] for entry in row["pc_level_acf_matched44"] if entry["lag"] <= 21] for row in scored])
            axis.plot([1, 2, 5, 10, 21], values.mean(axis=0), marker=".", label=model)
        axis.axhline(0., color="grey", linewidth=.7)
        axis.set_xlabel("Lag (business sessions)")
        axis.set_title(f"Frozen observable PC{component+1} levels")
    axes[0].set_ylabel("Within-window Pearson autocorrelation")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=10)
    figure.tight_layout(rect=(0, .19, 1, 1))
    figure.savefig(output / "acf.pdf")
    plt.close(figure)


def _export(output: Path, records: dict, summaries: dict, quotes: dict, persistence: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, tails, monthly, coverage, stability, iv_buckets = [], [], [], [], [], []
    with np.load(output / "reference_history.npz", allow_pickle=False) as saved:
        reference_dates = list(saved["dates"])
        reference_surfaces = saved["surfaces"][:, :, [4, 5, 7, 9, 11, 13, 14]].reshape(len(reference_dates), -1)
    initial_quote_rows, forecast_quote_rows = [], []
    for model in MODEL_ORDER:
        rows = records[model]
        scored = [row for row in rows if row.get("status") == "scored"]
        completed_recursive = _completed_recursive(persistence[model])
        coverage.append({"model": model, "requested_origins": 44, "surface_origins": len(scored),
            "quote_forecast_origins": sum(row.get("status") == "scored" for row in quotes[model]),
            "recursive_origins": len(completed_recursive),
            "recursive_invalid_origins": sum(row["status"] == "model_invalid" for row in completed_recursive),
            "recursive_min_sessions": min((row["steps"] for row in completed_recursive), default=None),
            "recursive_max_sessions": max((row["steps"] for row in completed_recursive), default=None)})
        for row in scored:
            daily.append({"model": model, "origin_date": row["origin_date"], "endpoint": row["endpoint"],
                "forecast_rmse_log_iv": row["forecast_rmse"], "scenario_count": row["scenario_count"],
                "persistence_rmse_log_iv": float(np.sqrt(np.mean(np.asarray(row["actual_change"])**2))),
                "invalid_surface_rate": row["invalid_surface_rate"], "extreme_surface_rate": row["extreme_surface_rate"]})
            if "stability" in row["diagnostics"]:
                native = row["diagnostics"]
                stability.append({"model": model, "origin_date": row["origin_date"],
                    "spectral_radius": native["stability"]["spectral_radius"], "raw_radius": native["raw_spectral_radius"],
                    "stationary_backbone": native["stability"]["stable_linear_core"],
                    "stationary_variance_trace": native["stability"]["stationary_variance_trace"],
                    "lyapunov_residual_relative": native["stability"]["lyapunov_residual_relative"],
                    "native_pca_coverage": native["native_pca_coverage"], "floored_nodes": native["floored_nodes"],
                    "undefined_nodes": native["undefined_nodes"], "capped_nodes": native["capped_nodes"],
                    "max_modal_half_life": max(value for value in native["stability"]["modal_half_lives"] if value is not None)})
        if scored:
            difference = np.array([row["surface_mean"] for row in scored]) - np.array([row["actual_change"] for row in scored])
            error = difference.reshape(len(scored), 7, 7)
            annual_errors = []
            for row in scored:
                index = reference_dates.index(row["origin_date"])
                forecast = np.exp(reference_surfaces[index] + np.asarray(row["surface_mean"]))
                annual_errors.append(forecast - np.exp(reference_surfaces[index + 1]))
            annual_errors = np.asarray(annual_errors).reshape(len(scored), 7, 7)
            for tenor_index, tenor in enumerate((1, 2, 3, 4, 5, 7, 10)):
                for nms_index, nms in enumerate((-1.5, -1., -.5, 0., .5, 1., 1.5)):
                    iv_buckets.append({"model": model, "business_dte": tenor, "nms": nms,
                        "rmse_log_iv": float(np.sqrt(np.mean(error[:, tenor_index, nms_index]**2))),
                        "iv_rmse_geometric_point": float(np.sqrt(np.mean(annual_errors[:, tenor_index, nms_index]**2))),
                        "point_forecast": "exp(observed log-IV at origin + model mean log-IV change)", "origins": len(scored)})
        for row in scored:
            if row.get("bidask", {}).get("status") == "scored":
                initial_quote_rows.append({"model": model, "origin_date": row["origin_date"],
                    **{key: value for key, value in row["bidask"].items() if isinstance(value, str | int | float)}})
        for row in quotes[model]:
            forecast_quote_rows.append({key: value for key, value in row.items() if isinstance(value, str | int | float)})
        for target, quantiles in summaries.get(model, {}).get("tails", {}).items():
            for level, values in quantiles.items():
                if level != "ES":
                    tails.append({"model": model, "target": target, "quantile": float(level),
                                  **{key: value for key, value in values.items() if key != "transition_counts"}})
    frame = pd.DataFrame(daily)
    rolling_frames = []
    for model, group in frame.groupby("model", sort=False):
        group = group.sort_values("origin_date").copy()
        group["rolling_21d_rmse"] = np.sqrt((group.forecast_rmse_log_iv**2).rolling(21, min_periods=21).mean())
        baseline = float(np.sqrt(np.mean(group.forecast_rmse_log_iv.iloc[:21]**2))) if len(group) >= 21 else np.nan
        group["rolling_21d_delta_from_first21"] = group.rolling_21d_rmse - baseline
        group["rolling_63d_delta"] = np.nan
        rolling_frames.append(group)
        for month, month_group in group.groupby(group.origin_date.str[:7]):
            monthly.append({"model": model, "month": month, "origins": len(month_group),
                            "pooled_log_iv_rmse": float(np.sqrt(np.mean(month_group.forecast_rmse_log_iv**2))),
                            "persistence_rmse_log_iv": float(np.sqrt(np.mean(month_group.persistence_rmse_log_iv**2)))})
    frame.to_csv(output / "daily_scores.csv", index=False)
    pd.concat(rolling_frames).to_csv(output / "rolling_scores.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_scores.csv", index=False)
    pd.DataFrame(tails).to_csv(output / "tail_scores.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "linear_stability.csv", index=False)
    pd.DataFrame(iv_buckets).to_csv(output / "iv_rmse_buckets.csv", index=False)
    pd.DataFrame(initial_quote_rows).to_csv(output / "reconstruction_scores.csv", index=False)
    pd.DataFrame(forecast_quote_rows).to_csv(output / "quote_forecast_scores.csv", index=False)
    return pd.DataFrame(coverage), pd.DataFrame(stability)


def main() -> None:
    global MODEL_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output
    selection_path = output / "MODEL_SELECTION.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        MODEL_ORDER = tuple(selection["models"])
    dysanos_rank = 16 if "(16 factors)" in MODEL_ORDER[0] else 8
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "pdf.fonttype": 42})
    reference = json.loads((output / "observable_reference.json").read_text())
    with np.load(output / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    records = {model: _rows(output / "daily" / model.replace(" ", "_")) for model in MODEL_ORDER}
    quotes = {model: _rows(output / "quote_forecasts" / model.replace(" ", "_")) for model in MODEL_ORDER}
    persistence = {model: _rows(output / "persistence" / model.replace(" ", "_")) for model in MODEL_ORDER}
    summaries = {model: _summarize(rows, basis) for model, rows in records.items()}
    _json(output / "summary.json", summaries)
    common_dates = sorted(set.intersection(*[{row["origin_date"] for row in rows if row.get("status") == "scored"} for rows in records.values()]))
    matched = {model: _summarize([row for row in rows if row.get("status") == "scored" and row["origin_date"] in common_dates], basis)
               for model, rows in records.items()}
    _json(output / "summary_matched_dates.json", {"dates": common_dates, "models": matched})
    coverage, stability = _export(output, records, summaries, quotes, persistence)
    monthly = pd.read_csv(output / "monthly_scores.csv")
    _json(output / "TRAFFIC_LIGHT_RULES.json", {"interpretation": "retrospective descriptive review tolerances, not model-selection or arbitrage tests",
        "colors": {"pass": "black", "warn": "orange", "fail": "red"},
        "rules": {name: asdict(rule) for name, rule in RULES.items()},
        "tail_coverage": {"pass_p_min": .05, "warn_p_min": .01, "minimum_expected_exceptions_for_pass": 10,
                          "test": "two-sided exact binomial; pointwise, exploratory, no across-model multiplicity correction"}})
    _plot_covariance(output, summaries)
    _plot_tails(output, records)
    _plot_acf(output, reference, persistence)
    covariance_rows, forecast_rows, leverage_rows, quote_rows, tail_rows = [], [], [], [], []
    for model in MODEL_ORDER:
        summary = summaries[model]
        if summary.get("surface_days", 0) >= 2:
            conditional, pooled = summary["conditional_covariance"], summary["pooled_covariance"]
            covariance_rows.append([model, summary["surface_days"], _rated("cov_frob_rel", conditional["cov_frob_rel"]), _rated("cov_frob_rel", pooled["cov_frob_rel"]),
                _rated("trace_ratio", conditional["trace_ratio"]), _rated("trace_ratio", pooled["trace_ratio"])])
            forecast_ratio = summary["forecast_rmse"] / summary["persistence_rmse"]
            forecast_rows.append([model, Rated(summary["forecast_rmse"], grade("forecast_ratio", forecast_ratio)),
                summary["persistence_rmse"], _rated("subspace_angle_max", pooled["subspace_angle_max"]), _rated("eig_share_err_k", pooled["eig_share_err_k"])])
            leverage_rows.append([model, summary["surface_days"], *[Rated(value, grade("spot_corr_abs_error", error))
                for value, error in zip(summary["spot_pc_correlations_model"], summary["spot_pc_corr_err_j"], strict=True)]])
            tail = summary["tails"]["PC1"]
            tail_rows.append([model, tail["0.05"]["count"], *[Rated(tail[level]["exceptions"], tail_grade(tail[level]))
                for level in ("0.01", "0.05", "0.95", "0.99")], tail["0.05"]["pinball_q"]])
        initial = [row["bidask"] for row in records[model] if row.get("status") == "scored" and row.get("bidask", {}).get("status") == "scored"]
        predicted = [row for row in quotes[model] if row.get("status") == "scored"]
        predicted_baseline = [row for row in predicted if "persistence_price_mae_origin_spot_units" in row]
        price_ratio = (float(np.mean([row["price_mae_origin_spot_units"] for row in predicted_baseline]))
            / float(np.mean([row["persistence_price_mae_origin_spot_units"] for row in predicted_baseline]))) if predicted_baseline else None
        quote_rows.append([model, _rated("fit_in_spread", float(np.mean([row["in_spread_rate"] for row in initial])) if initial else None),
            len(predicted), float(np.mean([row["in_spread_rate"] for row in predicted])) if predicted else None,
            float(np.mean([row["out_dist_mean"] for row in predicted])) if predicted else None,
            float(np.mean([row["price_mae_origin_spot_units"] for row in predicted])) if predicted else None, _rated("forecast_ratio", price_ratio)])
    observed_correlations = next((summaries[model]["spot_pc_correlations_data"] for model in MODEL_ORDER if summaries[model].get("surface_days") == 44), [])
    leverage_rows.insert(0, ["Observed reference", 44, *observed_correlations])
    monthly_rows = []
    for model in MODEL_ORDER:
        selected = monthly.loc[monthly.model == model].set_index("month")
        values = []
        skills = []
        for month in ("2024-04", "2024-05"):
            if month not in selected.index:
                values.extend([0, Rated(None, "warn")])
                skills.append(None)
                continue
            row = selected.loc[month]
            skill = float(row.pooled_log_iv_rmse / row.persistence_rmse_log_iv)
            values.extend([int(row.origins), Rated(float(row.pooled_log_iv_rmse), grade("forecast_ratio", skill))])
            skills.append(skill)
        deterioration = skills[1] / skills[0] if all(value is not None and value > 0 for value in skills) else None
        monthly_rows.append([model, *values, _rated("window_deterioration", deterioration)])
    acf_rows = []
    for model in MODEL_ORDER:
        scored = [row for row in persistence[model] if row.get("status") == "scored"]
        errors = []
        for lag in (1, 5, 21):
            observed = next(row for row in reference["pc_level_acf"] if row["lag"] == lag)
            predictions = [next(row for row in entry["pc_level_acf_matched44"] if row["lag"] == lag)["correlations"] for entry in scored]
            error = float(np.max(np.abs(np.mean(predictions, axis=0) - observed["correlations"]))) if predictions else None
            level = grade("acf_abs_error", error)
            if observed["pair_count"] < 30 and level == "pass":
                level = "warn"
            errors.append(Rated(error, level))
        acf_rows.append([model, *errors])
    may_ft_fits = len(list((output / "completion/ft-may/dates").glob("*.json")))
    inference_path = output / "INFERENCE_MODELS.json"
    inference = json.loads(inference_path.read_text()) if inference_path.exists() else {"models": {}}
    retired_models = [entry for entry in inference["models"].values() if entry.get("source_retired")]
    inference_gib = inference.get("inference_bytes", 0) / 2**30
    retired_gib = sum(entry["source_bytes"] for entry in retired_models) / 2**30
    reconciliation_rows = []
    for path in sorted((output / "variance_reconciliation").glob("*/*.json")):
        record = json.loads(path.read_text())
        groups = record["groups"]
        rank = 16 if "(16 factors)" in record["model"] else 8
        reconciliation_rows.append([record["model"], record["origin_date"],
            100 * groups[f"pca{rank}_centered_changes"]["native_energy_fraction_of_full"],
            100 * groups[f"pca{rank}_centered_changes"]["decoded_variance_fraction_of_full"],
            100 * groups[f"pca{rank}_ar_innovations"]["decoded_variance_fraction_of_full"]])
    tail_fits = [row for model in MODEL_ORDER if model.endswith("(tail)")
                 for row in _rows(output / "tail_fits" / model.replace(" ", "_"))]
    tail_warnings = [row for row in tail_fits if not row["fit"]["success"]]
    tail_reviews = [json.loads(path.read_text()) for path in (output / "tail_solver_reviews").glob("*/*.json")]
    rule_rows = [
        ("Covariance Frobenius error", "at most 0.25", "above 0.25, at most 0.50", "above 0.50"),
        ("Variance trace ratio", "0.80 to 1.25", "0.50 to 2.00, outside pass", "outside 0.50 to 2.00"),
        ("Leading-share L1 error", "at most 0.10", "above 0.10, at most 0.25", "above 0.25"),
        ("Subspace angle", "at most 15 degrees", "above 15, at most 30 degrees", "above 30 degrees"),
        ("Spot-correlation or IV-ACF error", "at most 0.10", "above 0.10, at most 0.20", "above 0.20"),
        ("Forecast loss / persistence", "at most 1.00", "above 1.00, at most 1.10", "above 1.10"),
        ("Reconstruction inside spread", "at least 95%", "80% to below 95%", "below 80%"),
        ("Reconstruction mean outside distance", "at most 0.25 spreads", "above 0.25, at most 1 spread", "above 1 spread"),
        ("Reconstruction q95 outside distance", "at most 1 spread", "above 1, at most 2 spreads", "above 2 spreads"),
        ("May/April relative-skill deterioration", "at most 1.10", "above 1.10, at most 1.25", "above 1.25"),
        ("Exact-binomial tail coverage", "p at least .05 and 10 expected exceptions", "p at least .01 but not pass", "p below .01"),
        ("PCA AR spectral radius", "below .99", ".99 to below 1", "at least 1"),
        ("Invalid decoded paths", "zero", "above zero, at most 1%", "above 1%"),
        ("Surfaces with IV above 300%", "zero", "above zero, at most 0.1%", "above 0.1%"),
        ("Diagnostic coverage", "all requested dates", "partial coverage or not run", "not a model-quality failure"),
    ]
    threshold_table = "\n".join([r"{\small\setlength{\tabcolsep}{3pt}\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.25\linewidth}>{\raggedright\arraybackslash}p{.22\linewidth}>{\raggedright\arraybackslash}p{.25\linewidth}>{\raggedright\arraybackslash}p{.22\linewidth}@{}}",
        r"\caption{Retrospective traffic-light rules. Black: pass under the stated review tolerance; orange: warning or insufficient evidence; red: outside tolerance or a rejected diagnostic null. Missing evidence is orange. Boundaries are inclusive except where stated.}\label{tab:grading-rules}\\",
        r"\toprule Diagnostic & Pass (black) & Warn (orange) & Fail (red)\\\midrule\endfirsthead",
        r"\toprule Diagnostic & Pass (black) & Warn (orange) & Fail (red)\\\midrule\endhead",
        *[" & ".join(_escape(value) for value in row) + r" \\" for row in rule_rows], r"\bottomrule\end{longtable}}"])
    body = [r"\documentclass[11pt]{article}", r"\usepackage[a4paper,margin=20mm]{geometry}",
        r"\usepackage{amsmath,amssymb,array,booktabs,graphicx,float,longtable,hyperref,xcolor}",
        r"\definecolor{orange}{RGB}{190,95,0}",
        r"\hypersetup{colorlinks=true,urlcolor=blue}", r"\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}",
        r"\title{Retained Hedging Models: Surface-Dynamics Audit}", r"\author{AI-agent implementation and diagnostic feedback}",
        r"\date{11 November 2026}", r"\begin{document}\maketitle",
        _executive_summary(summaries, coverage, stability),
        r"\section{Scope and headline findings}",
        r"The audit covers 44 origins from 1 April through 31 May 2024; the final one-session forecast settles on 3 June. Existing fits are retained unchanged. DYSANOS means our own saved floating-DLV PCA/AR(1) fits, not the latest external SANOS package. Results are diagnostics of retained legacy fits, not validation of the corrected hedge2 specification.",
        r"The observable reference is the retained public-candidate SANOS interpolation (smoothness 0.25), manifest \texttt{2add65d6\ldots}. Our DYSANOS coordinates instead came from the licensed smoothness-0.1 interpolation. These are not identical input pipelines; the comparison therefore measures retained-system behaviour, not architecture alone.",
        r"\textbf{Main distinction:} pooled covariance can look realistic even when a model has almost no conditional randomness. VolGAN is the clearest example. Persistence is measured using one pre-April observable PCA basis, never by concatenating rolling native factor scores. The DYSANOS stationarity assessment concerns only its floating-volatility PCA--AR(1) backbone; spot-level stationarity is neither required nor tested. All 44 fitted backbones in each DYSANOS variant admit a stationary distribution.",
        _table(["Model", "Requested dates", "Surface dates", "Quote dates", "Path-test dates"],
            [[row.model, row.requested_origins, *[Rated(count, "pass" if count == row.requested_origins else "warn")
                for count in (row.surface_origins, row.quote_forecast_origins, row.recursive_origins)]] for row in coverage.itertuples()],
            caption="Diagnostic coverage: dates evaluated out of the 44 requested April--May 2024 origins.", label="tab:audit-coverage"),
        r"Table~\ref{tab:audit-coverage} is a coverage inventory, not a performance or pass/fail table. Surface dates count origins with evaluated next-day IV forecasts for covariance and tail diagnostics. Quote dates count origins with exact next-day contract matches for price-forecast scoring. Path-test dates count completed recursive floating-IV tests, including tests that produced no wholly valid decoded IV paths. Such failures contribute their actual invalid rates to Table~\ref{tab:recursive-stability}; they are not hidden as missing coverage. Each entry counts dates, not Monte Carlo paths; zero means no completed test, not a model-quality failure.",
        f"Covariance/tail availability follows retained next-surface arrays or newly sampled saved fits. The base-diffusion first next-IV artifact was absent: 512 new CPU scenarios were generated from its existing 1 April EMA checkpoint, with no retraining; its other 43 dates retain their 10,000-path forecasts. At the user's request, {may_ft_fits} additional May FT fits now continue the original April chain sequentially, using the unchanged selected one-epoch rolling update and causal 252-observation windows. Training completion is distinct from forecast-scenario availability in Table~\\ref{{tab:audit-coverage}}. Quote forecasts can use retained price-return books independently of IV arrays. The previously missing 20 May quote join was a normalized-strike float32 round-trip identity bug, not absent quotes. The corrected join accepts a source cash strike only when it is the unique candidate within four float32 machine epsilons times the reconstructed cash-strike scale. Ambiguous and more distant contracts remain excluded; per-date maximum identity errors are retained.",
        r"Every model uses the same 32-path, 115-business-session protocol on the same 44 origins, with seed 20261111. Completed neural paths reuse their identical first 115 sessions; partial diffusion checkpoints preserve the original context and RNG state. Native models retain their different stochastic laws, not artificially identical generated prices. GROMM's final saved expiry is 125D and the observable grid extends to 10D, so 115 sessions is the common horizon without inventing expiries or long-end extrapolation. Its aged maturities are decoded by total-variance interpolation at fixed normalized strike, with a zero-maturity variance anchor (flat short-end IV). All models use their own generated history, never future observed states. Coverage counts only completed tests; queued work is not a pass.",
        r"\section{Traffic-light interpretation and origins of thresholds}",
        r"Black means within the stated review tolerance, orange means warning or insufficient evidence, and red means outside tolerance or a rejected diagnostic null. These are diagnostic-specific labels, not a proof of arbitrage-freedom and not a ranking of trading profitability. Counts, reference values and proper scores without a comparison baseline are shown neutrally. Missing evidence is orange, never a pass.",
        threshold_table,
        r"Only the AR boundary $\rho(A)=1$ comes from the mathematical stationarity condition. The nearby .99 warning is a practical slow-reversion flag: $\log(1/2)/\log(.99)\simeq69$ business sessions. Tail cutoffs .05 and .01 are conventional pointwise type-I error levels for the exact binomial coverage test, not Gaussian standard-error approximations. Fewer than ten expected exceptions prevents a black tail result even if the null is not rejected. An IV-ACF comparison with fewer than 30 observed lag pairs likewise cannot receive black.",
        r"All remaining boundaries are explicit, round-number engineering tolerances introduced for this retrospective review, not confidence bounds, published universal standards, or thresholds selected on a separate validation set. Frobenius bounds permit 25\%/50\% norm error; trace bounds permit a factor 1.25/two mismatch; eigen-share bounds permit 10/25 percentage points of summed error; correlation bounds permit .10/.20 absolute error. The spread and forecast tolerances express stated reconstruction goals and allowed deterioration against a simple persistence forecast. They were not used to choose or retrain the models and should be sensitivity-tested before any production acceptance decision.",
        r"Next-day forecast in-spread rates and mean outside distances are not graded as reconstruction: a correct conditional expectation need not lie in a narrow realized next-day spread. Price-prediction grading instead compares MAE with the same contracts' origin observed midpoints carried by the origin-only one-day discount factor. Log-IV forecast RMSE is compared with the no-change IV forecast. A ratio at most one means no deterioration; up to 1.10 is the explicit 10\% warning band. May/April deterioration compares these persistence-normalized log-IV losses, reducing confusion with changing market volatility. Pinball and joint VaR--ES scores are proper-score comparisons, not zero-centered tests; tail colors refer to coverage, not their raw score magnitudes.",
        r"Colors on conditional covariance/trace are descriptive dispersion screens against realized total covariance, not formal conditional-covariance tests; between-origin predicted means can explain a difference. Tail p-values and colors are pointwise and exploratory; no across-model multiplicity correction is claimed. Historical-P comparisons for GROMM-Q are descriptive, not a demand that a Q generator reproduce P outcomes. The complete numeric rules are saved in \texttt{TRAFFIC\_LIGHT\_RULES.json}.",
        r"\section{Surface covariance and spot leverage}",
        r"The common grid has business-day tenors $(1,2,3,4,5,7,10)$ and NMS $(-1.5,-1,-0.5,0,0.5,1,1.5)$, giving 49 equally weighted log-IV coordinates. The central common domain avoids comparing different native grids or extreme-wing inversion conventions. All retained reference nodes in this common domain have interior IVs; the broader native reference grid does contain repaired wing nodes. This is not a test of the omitted wings. Native grids remain unchanged.",
        f"The fixed three-PC basis uses 252 exact one-session training transitions with endpoints {_escape(reference['training_first_endpoint'])}--{_escape(reference['training_last_endpoint'])}, strictly before April. These are eligible retained transitions, not an assertion of 252 consecutive calendar sessions. Their variance shares are {', '.join(f'{100*value:.2f}' for value in reference['pca_training_shares'])} percent (total {100*sum(reference['pca_training_shares']):.2f} percent). No whitening or per-model refitting of this audit basis is used.",
        r"Let $x_t$ be the common log-IV vector, $U$ the fixed basis, $\Delta x_{t+1}=x_{t+1}-x_t$, $m_t=E_t[\Delta x_{t+1}]$ and $C_t=\operatorname{Cov}_t(\Delta x_{t+1})$. We compare both $\bar C$ and $\bar C+\widehat{\operatorname{Cov}}_t(m_t)$ with the sample covariance of observed changes. The latter retains between-origin variation in conditional means; it must not be mistaken for conditional shock dispersion. Across-origin covariance uses denominator $n-1$; the displayed sum is the fixed-design, sample-covariance convention. Monte Carlo forecast covariance uses the sample convention, whereas exact enumeration of the 252-atom DYSANOS law uses its population denominator 252.",
        r"\[E_F=\frac{\|\Sigma_{\rm model}-\Sigma_{\rm data}\|_F}{\|\Sigma_{\rm data}\|_F}.\]",
        _table(["Model", "Days", "Frob. cond.", "Frob. pooled", "Trace cond.", "Trace pooled"], covariance_rows,
            caption="Surface covariance errors and variance ratios on each model's available dates.", label="tab:surface-covariance"),
        r"In Table~\ref{tab:surface-covariance}, the two Frobenius columns measure relative covariance-matrix error (zero is exact agreement); the two trace columns divide model by observed total log-IV-change variance (one is scale agreement). Figure~\ref{fig:surface-variance} plots those same trace columns. Its conditional bar uses the average within-origin forecast covariance; its pooled bar additionally includes between-origin forecast-mean variation. The color bands in the table caption are retrospective engineering review tolerances, not precommitted statistical pass/fail criteria. In particular, the conditional ratio need not equal one because the observed denominator is unconditional.",
        (f"Table~\\ref{{tab:surface-covariance}} uses each model's available dates. Table~\\ref{{tab:matched-covariance}} restricts every model to the same {len(common_dates)} common origins ({_escape(common_dates[0])}--{_escape(common_dates[-1])}). The PCA basis remains unchanged."
         if common_dates else r"The new Tail Diffusion cases are still being scored, so an all-model common-date comparison is not yet available. Individual-model results use only their completed dates; the PCA basis remains unchanged."),
        _table(["Model", "Matched days", "Frob. pooled", "Trace cond.", "Trace pooled"], [[model, item["surface_days"],
            _rated("cov_frob_rel", item["pooled_covariance"]["cov_frob_rel"]), _rated("trace_ratio", item["conditional_covariance"]["trace_ratio"]), _rated("trace_ratio", item["pooled_covariance"]["trace_ratio"]) ]
            for model, item in matched.items() if item.get("surface_days", 0) >= 2],
            caption="Surface covariance comparison restricted to dates available for every model.", label="tab:matched-covariance"),
        _figure("covariance.pdf", "Conditional and pooled surface-variance ratios. The logarithmic scale makes underdispersion visible rather than hiding it behind changing forecast means."),
        r"\subsection{Why native PCA coverage need not preserve IV variance}",
        r"The retained DYSANOS PCs minimize Euclidean reconstruction error in 1809 native log-DLV/forward-vol coordinates, not in option prices or log implied volatilities. Locally, if $J$ is the pricing/IV decoder Jacobian, the observable covariance is $J\Sigma_uJ'$. A PCA truncation that retains much of $\operatorname{tr}(\Sigma_u)$ need not retain much of $\operatorname{tr}(J\Sigma_uJ')$. Drift-adjusting the retained factors for AR fitting can further reduce innovation variance. Floors and nonlinear decoder guards also affect this mapping. Therefore a native coverage percentage is not an observable-variance guarantee.",
        _table(["Model", "Origin", "Native retained %", "PCA IV %", "AR innovation IV %"], reconciliation_rows,
            caption="Same-shock variance reconciliation. Each origin uses the same 252 trailing centered native changes and current decoder. Percentages divide by the full centered-native-shock comparator: native energy for the first numeric column, decoded log-IV variance for the others. No shocks are inflated; the comparator is an ablation, not a fitted full-rank alternative. Original decoder guards remain active and their counts are retained.",
            label="tab:variance-reconciliation"),
        *_rank_comparison_section(output),
        r"Leading eigenvalue shares use the full covariance trace as denominator. Principal angles compare the leading three-dimensional subspaces of the full observable covariance; the table gives the maximum angle in degrees. The full machine-readable summary also contains the three-PC projected covariance comparison. Using three dimensions for a three-dimensional projected subspace angle would be vacuous, so only the full-space angles are interpreted.",
        _table(["Model", "Forecast RMSE", "Persistence RMSE", "Max angle", "Share L1 error"], forecast_rows,
            caption="Log-IV forecast errors and leading covariance-subspace agreement.", label="tab:forecast-subspace"),
        r"RMSE is in log-IV units over scored dates and all 49 cells. Persistence predicts no surface change. RMSE values concern raw model forecasts, not the artificial demeaning used by some hedge optimizers.",
        r"Spot leverage is $\operatorname{Corr}(r_{t+1},U_j^\top\Delta x_{t+1})$, where $r$ is the one-session forward-normalized log return. Signs are fixed by the training basis. Model values below use pooled joint moments; per-component absolute errors are in the JSON. A native DLV loading is not substituted for this observable correlation.",
        _table(["Model", "Days", "Spot/PC1", "Spot/PC2", "Spot/PC3"], leverage_rows,
            caption="Pooled correlations between spot returns and fixed-basis observable PC changes.", label="tab:spot-leverage"),
        r"The JSON additionally separates correlations from the average conditional joint covariance and correlations of realized forecast residuals (observed changes minus each model's predicted means). This prevents between-origin mean variation from being mistaken for conditional spot leverage.",
        r"\section{Bid--ask fit and next-day prediction}",
        r"For quoted bid $b$, ask $a$ and predicted price $p$, outside distance is $d=\max(b-p,p-a,0)/(a-b)$. We report the in-spread share, mean and 95th percentile of $d$, and cash-price MAE. Crossed/zero-width/nonfinite inputs are excluded with counts. Quotes are licensed observed bid/ask, not fabricated spreads around fitted prices.",
        _table(["Model", "Recon. in", "Pred. N", "Pred. in", "Mean distance", "MAE / spot", "MAE ratio"], quote_rows,
            caption="Same-day quoted-spread reconstruction and next-day conditional-mean price prediction.", label="tab:bidask-fit"),
        r"Reconstruction uses all available quoted sides with origin tenors 1--10D and $|\mathrm{NMS}|\le1.5$ for DYSANOS and GROMM. Conditional IV generators do not have an independently fitted same-day observable decoder in this audit: their reconstruction is n/a, not perfect by copying their conditioning input. GROMM's stored 252-transition fit percentages are not substituted for today's measured quoted reconstruction.",
        r"Next-day prediction uses surviving OTM hedge-book contracts with origin tenors 2--5D and $|\mathrm{NMS}|\le1.5$, matched at their exact cash strike, expiry and settlement type. The expected raw scenario price is compared with tomorrow's observed spread. Financing/carry is fixed from the origin; tomorrow's spot/forward curve is not inserted into the forecast. Expiring 1D contracts are settlements, not next-day quotes. This is a limited retained-book forecast population, not every model-grid contract. Population, matched and excluded counts and moneyness-bucket scores are retained per date. Reconstruction and forecast columns have different populations and should not be compared as like-for-like fit rates.",
        r"A correct stochastic model need not put its conditional-mean forecast inside a narrow realized next-day spread. The forecast in-spread rate is a prediction diagnostic, not an arbitrage criterion. Quote aggregates give equal weight to dates, not to the number of contracts on each date. All daily counts, mean and 95th-percentile outside distances, price MAEs and buckets are retained in the CSV/JSON outputs.",
        r"\texttt{iv\_rmse\_buckets.csv} separately reports both log-IV RMSE and annual-IV RMSE for the explicitly defined point forecast $\hat\sigma_{t+1}=\exp(x_t+m_t)$, anchored to the common observed origin surface. This is a geometric-IV point forecast, not $E_t[\sigma_{t+1}]$ and not IV obtained by inverting a mean option price. Errors are bucketed by expiry and NMS. No realized forward is used to manufacture an IV from a potentially non-arbitrage-bounded mean price.",
        r"\section{Out-of-sample window stability}",
        r"Every available saved origin is scored. Monthly RMS errors are separated by April and May; no six-month evaluation is claimed. Both monthly and 21-origin rolling RMSE take the square root after averaging squared daily RMSE, consistent with the full-window metric. Rolling deterioration subtracts the first 21-origin value. A 63-session deterioration estimate is unavailable. These overlapping summaries are descriptive, not independent replications. Forecast horizon (one session), surface maturity (1--10 sessions) and evaluation-window length (44 origins) remain separate.",
        _table(["Model", "April days", "April RMSE", "May days", "May RMSE", "Skill deterioration"], monthly_rows,
            caption="Monthly next-day log-IV forecast RMSE and evaluated origins.", label="tab:monthly-errors"),
        r"\section{Tail risk calibration}",
        r"Targets are the three observable PC changes and spot log return, not hedge P\&L. Quantile levels are 1, 5, 95 and 99 percent. For $q_\alpha$, pinball loss is $(\alpha-\mathbf1_{\{y<q_\alpha\}})(y-q_\alpha)$. Lower-tail exceptions are $y<q_\alpha$ and upper-tail exceptions $y>q_\alpha$. The outputs contain exception rates, exact binomial coverage intervals and p-values, asymptotic Kupiec p-values and, when identifiable, Christoffersen independence p-values. Clustering tests are n/a when exception histories do not identify both transition rows.",
        r"The following counts are for PC1; all four targets and all probabilities are in \texttt{tail\_scores.csv}. Expected exception counts with 44 dates are 0.44 in each 1\% tail and 2.2 in each 5\% tail. Every tail test is therefore labelled exploratory. Ten thousand Monte Carlo scenarios do not create ten thousand held-out market dates. No green pass is awarded from a nonsignificant p-value. GROMM-Q is compared with historical outcomes but is not thereby tested for physical-law calibration.",
        _table(["Model", "Days", "Below 1%", "Below 5%", "Above 95%", "Above 99%", "5% pinball"], tail_rows,
            caption="Exploratory PC1 tail-forecast exception counts and lower-5-percent pinball loss.", label="tab:tail-pc1"),
        _figure("tails.pdf", "Raw PC1 forecast bands and observed changes. Narrow forecast bands can reveal severe conditional underdispersion even when pooled covariance looks realistic."),
        r"The lower 5\% ES estimate is the Monte Carlo mean below the empirical 5\% quantile. The JSON contains the joint Fissler--Ziegel score, with $G_1=0$, $G_2(e)=\exp(e)$, for signed lower-tail ES:",
        r"\[S_\alpha(y,q,e)=e^e\left[e-q+\frac{\mathbf1_{\{y\le q\}}}{\alpha}(q-y)-1\right].\]",
        r"Lower expected score is better for a fixed target and units. Unlike the negative-ES-only FZ0 specialization, this choice does not discard forecasts whose lower-tail ES is positive. Scores of differently scaled targets are not pooled into a headline ranking.",
        r"\section{PCA--AR(1) stationarity and floating-IV persistence}",
        r"The analytic stationarity question is restricted to the fitted floating-volatility PCA backbone, with coefficients frozen at each origin. Write its transition as",
        r"\begin{equation} h_{n+1}=b+A h_n+\varepsilon_{n+1},\qquad E[\varepsilon_{n+1}]=0,\quad\operatorname{Cov}(\varepsilon_{n+1})=Q.\end{equation}",
        r"Innovations are independent across sessions under the fitted law and have finite covariance. A Gaussian stock-return innovation may contribute to $\varepsilon$; it changes $Q$, including any stock--factor cross-covariance, not the stability criterion. No stationarity condition is imposed on the stock price. A sufficient condition for a unique causal stationary backbone, and the criterion tested here, is",
        r"\begin{equation}\rho(A)<1,\qquad \bar h=(I-A)^{-1}b,\qquad V=A V A^\top+Q.\end{equation}",
        f"The Lyapunov solution $V$ is the stationary covariance. If the complete innovation vector is jointly Gaussian, the stationary law is $N(\\bar h,V)$; Gaussianity is not needed for covariance stationarity. For the orthogonal variant, the innovation covariance includes the stock loading projected onto the {dysanos_rank} PCA coordinates. Starting a forecast at the observed state need not start it in its invariant distribution: the stable backbone converges toward that law. This is a per-fit assessment, not a claim that a daily re-estimated sequence has constant parameters.",
        r"Let $z_{j,t}=U_j^\top x_t$. The PCA basis is fixed before April, while each model may continue using its own daily rolling factors internally. We never concatenate native factor scores across changing bases. We separately retain level ACF, change ACF and expiry-by-expiry ATM log-IV level ACF. Exact session separations are used; lag 21 has 23 observed pairs and lag 63 has none.",
        _figure("acf.pdf", "PCA-level autocorrelation decay. Model curves average within-path Pearson ACFs from the first 44 simulated sessions, matching the observational window length; they are not formed by concatenating different dates' rolling models. The full 128-session checks are stored separately. Short-window detrending can itself produce negative long-lag estimates."),
        _table(["Model", "Lag 1 error", "Lag 5 error", "Lag 21 error"], acf_rows,
            caption="Maximum absolute floating-IV ACF discrepancy across the three fixed observable PCs. Lag 21 has only 23 observed pairs, so cannot receive black.", label="tab:acf-errors"),
        r"For DYSANOS only, we report the saved backbone spectral radius, modal damping half-lives $\log(1/2)/\log|\lambda_j|$ for stable nonzero eigenvalues, and innovation variance traces at horizons 1,5,21,63. Complex/negative eigenvalues describe oscillatory damping, not a monotone scalar half-life. Half-lives are measured in business sessions. Stationary means, covariances and numerical Lyapunov residuals are retained in the dated JSON files. All 88 fitted radii are already below one before the configured spectral-radius cap is applied.",
        _table(["DYSANOS backbone", "Stable fits", "Min radius", "Max radius", "Max half-life"], [[model,
            f"{int(stability.loc[stability.model == model, 'stationary_backbone'].sum())}/44",
            _rated("ar_radius", float(stability.loc[stability.model == model, "spectral_radius"].min())),
            _rated("ar_radius", float(stability.loc[stability.model == model, "spectral_radius"].max())),
            float(stability.loc[stability.model == model, "max_modal_half_life"].max())] for model in MODEL_ORDER[:2]],
            caption="Stationarity of the floating-volatility PCA--AR(1) backbones; half-lives are in business sessions.", label="tab:ar-stationarity"),
        r"The separate recursive tests below concern decoded floating IVs, not stationarity of spot and not an additional acceptance condition for the PCA--AR backbone. They retain the original model laws without substituting Gaussian stock shocks or changing fitted coefficients.",
        _table(["Model", "Recursive origins", "Extreme surface rate", "Invalid path rate"], [[model,
            len(_completed_recursive(persistence[model])),
            _rated("extreme_rate", float(np.mean([row["diagnostics"]["extreme_surface_rate"] for row in _completed_recursive(persistence[model])])) if _completed_recursive(persistence[model]) else None),
            _rated("invalid_rate", float(np.mean([row["diagnostics"]["invalid_path_rate"] for row in _completed_recursive(persistence[model])])) if _completed_recursive(persistence[model]) else None)] for model in MODEL_ORDER],
            caption="Recursive floating-IV diagnostics: evaluated origins and extreme/invalid rates.", label="tab:recursive-stability"),
        r"An extreme surface has at least one common-grid annual IV above 3.0 (300\%). All option-price inversion uses vectorized \texttt{cdxcore.bs.implied} on raw, unclipped inputs, with its $10^{-8}$ price tolerance and an independent vectorized \texttt{bs.price} repricing check. The former additional time-value/interiority veto has been removed. The library's default 1\%--500\% volatility bounds and intrinsic-price conventions are retained and counted explicitly; a library boundary estimate is not called a failed model path solely for having tiny time value. Native-price bound violations and repricing residuals remain recorded. These numerical point estimates do not imply uniquely identified wing volatilities at sub-tolerance prices. A completed test with no usable complete IV paths remains in coverage and invalid-rate statistics but has no ACF. All DLV clipping counts are retained separately. Recursive paths use frozen fits and their original innovation laws; finite-horizon survival is not a non-explosion proof.",
        r"\appendix\section{Model-specific implementation choices and difficulties}",
        r"This appendix is AI-agent implementation feedback based on actual retained code and artifacts, not a claim that these choices are prescribed by the cited papers. Existing model laws and fits are unchanged; the additional May FT fits use the same rolling recipe and are separately identified. The audit does not treat legacy hedging results as newly validated.",
        r"\subsection{Our DYSANOS fits: ordinary and spot-orthogonal}",
        r"Both variants use the retained nine tenors $(1,2,3,4,5,10,15,20,25)$, 200 interior strikes and native NMS $(-4,3)$. Daily source prices came from saved 0.1-SANOS surfaces; the representation itself has zero backbone smoothing. Native coordinates contain nine blocks of 200 log-DLV-squared coordinates plus a log-forward-volatility coordinate (1809 in total). This is not the later requested 13-tenor correction, and not the former 1811-coordinate smoothed SANOS state.",
        f"Each daily fit uses 252 eligible exact-session transitions, rank {dysanos_rank}, ridge $10^{{-6}}$ and spectral-radius cap 0.999. Factor orientation is aligned to the preceding fit. The raw and constrained radii are both saved. The orthogonal variant first removes the spot projection from state changes and adds its loading back during simulation. The analytic stationarity score tests the {dysanos_rank} floating-volatility AR coordinates only; its innovation covariance includes the projected stock contribution and cross-covariances, without testing stock-price stationarity.",
        r"The legacy conversion contains zero-over-zero/unidentified and floored DLV coordinates. The report preserves their counts rather than explaining them away as successful calibration. The model decoder retains local-volatility bounds 0.001--1000 and the numerical log-strike bound 30; it records clipping. For one-day forecasts we enumerate all 252 paired empirical innovations exactly. These repeated historical shocks are not 10,000 independent observations. Spot follows the original residual bootstrap here, not a newly imposed front-density marginal. These defects and choices limit the interpretation of this legacy model.",
        f"Across the 44 origins, the mean native {dysanos_rank}-PC variance coverage is {100*stability.loc[stability.model == MODEL_ORDER[0], 'native_pca_coverage'].mean():.2f} percent for ordinary PCA and {100*stability.loc[stability.model == MODEL_ORDER[1], 'native_pca_coverage'].mean():.2f} percent after spot orthogonalisation. The daily conversion has {int(stability.floored_nodes.min())}--{int(stability.floored_nodes.max())} floored nodes out of 1800 DLV coordinates (median {stability.floored_nodes.median():.0f}). Native-coordinate coverage is not the same denominator as observable log-IV-change coverage.",
        r"\subsection{Additional causal Tail Diffusion cases}",
        f"Both ordinary and spot-orthogonal variants retain the same surface decoder and {dysanos_rank}-factor AR coefficients but replace empirical surface-innovation replay by the paper's scalar log-variance OU scale mixture. Normal spot innovations are separated by the fitted joint covariance; tail scaling applies to the spot-conditioned Gaussian surface residual. Each dated tail law uses only that fit's preceding 252 exact-session innovations, resetting the latent filter across gaps. Its likelihood uses the existing Gaussian approximation to log-chi-square radius measurements, here in {dysanos_rank} dimensions rather than the paper's 200-dimensional application. This approximation is an explicit limitation.",
        r"With fitted conditional factor loading $L$, the surface shock is $\sqrt{\rho\exp(g-v_g/2)}Lz$, while $g^+=\phi g+\sqrt{v_g(1-\phi^2)}\xi$. The existing tail fitter estimates $(\rho,\phi,v_g)$, adding the previous origin's parameters as an optimizer start. The initial latent law is the causal one-step predictive distribution from the trailing radius filter, not an unconditional stationary reset at every simulated session. No April--May future observation tunes these parameters. One-day forecasts use 10,000 draws and recursive tests the same 32 paths over 115 sessions. Tail thickness is not artificially rescaled to match test-window variance.",
        f"At this build there are {len(tail_fits)} tail fits and {len(tail_warnings)} raw optimizer warnings. {sum(row['local_accuracy_verified'] for row in tail_reviews)} warning cases have independent projected-gradient verification at $10^{{-6}}$ mean-log-likelihood accuracy, without changing parameters. Exact original messages remain in the fitted records and review sidecars; a local gradient check is not a global-optimum certificate.",
        r"\subsection{FuNVol}",
        r"This is the single-asset adaptation on the seven-tenor, 19-NMS-node short-end grid. Inverse-softplus IVs are smoothed on a fourth-order Legendre basis, then reduced to three functional PCs. Joint score/log-spot dynamics use separate drift and diffusion GRUs: three layers, width 16, ten-state history, Gaussian increments over 1/252 year. The FPC basis and normalization are inherited through the rolling checkpoint chain, not recomputed daily. A 2022 comparison selected 100 rolling epochs; subsequent daily fits used all their causal window rather than a future holdout.",
        r"The first causal April fit generated spot volatility 0.333\% against trailing historical 0.778\%, failing the original 0.5-ratio dispersion gate. The user explicitly authorized proceeding with those numbers. The warning remains part of the model interpretation; no calibration pass is retroactively claimed. Softplus ensures positive IV, not absence of static/dynamic arbitrage. The earlier narrow static-arbitrage screen covered only 11 strikes within approximately 0.5\% of ATM. The present central-grid audit is different and is not an arbitrage proof.",
        r"The retained first-April FPC basis reports 97.849\% cumulative training variance coverage for its three components. This is variance of Legendre-smoothed inverse-softplus IV levels in its initializer coordinate system, not variance coverage of daily observable log-IV changes in the audit. These percentages must not be compared without their denominators.",
        r"Reference: Choudhary, Jaimungal and Bergeron, FuNVol, Quantitative Finance 24 (2024), DOI \href{https://doi.org/10.1080/14697688.2024.2396977}{10.1080/14697688.2024.2396977}.",
        r"\subsection{VolGAN PCA8}",
        r"The retained model is the paper-inspired eight-PC adaptation, not the original full-surface VolGAN. Separate train-only bases compress log-IV levels in the condition and log-IV changes in the output. Spot return remains an explicit output. Frozen initializer bases are inherited across dates; network weights and latent normalization are updated causally. Conditioning includes two previous annualized returns and trailing 21-session realized volatility. Softplus networks use noise dimension 32 and hidden dimension 16.",
        r"The 2022 initializer trained for 10,000 epochs; selected rolling updates use 300 epochs, and 10,000 sampled scenarios per origin is a different quantity. PCA bases and scaling use only training data. The gradient-matching phase continues from updated weights rather than restarting; its 25 epochs are included in the training total. The initializer selected rolling-update settings on a shorter 200-observation 2022 window because fewer than 252 observations were available. The native-model and short-end-data adaptations must not be described as an exact reproduction of published results.",
        r"The one-day surface audit re-samples the saved, version-checked checkpoints on CPU with seed 20261111, preserving the model's decoding and conditioning. Recursive sampling updates each path's own IV surface, recent returns and realized-volatility condition; it does not feed actual future market data back in. Conditional collapse and recursive extreme IVs are reported, not repaired by rescaling shocks.",
        r"Reference: Vuletic and Cont, VolGAN, DOI \href{https://doi.org/10.1080/1350486X.2025.2471317}{10.1080/1350486X.2025.2471317}.",
        r"\subsection{AD-Seq-Vol and fine-tuned AD-Seq-Vol-FT}",
        r"The short-end input is a 22-session sequence of log-IV and broadcast normalized-forward return channels; 21 sessions condition the next. The retained architecture uses a causal transformer of width 256, eight layers, four heads and a 200-step cosine diffusion. The fine-tuned variant freezes the EMA base and fits a zero-initialized rank-four output adapter with static-arbitrage penalties. Neither variant is a risk-neutral generator.",
        f"Published training settings include learning rate $7\\cdot10^{{-5}}$, batch size 256, weight decay 0.01, 1600 warm-up steps, EMA 0.995 and a 3000-epoch initializer. Dated rolling fits warm-start from prior weights; their original normalization is kept fixed because rebasing would change the diffusion-noise law. The newest 20\\% of training sequences is validation. The April base checkpoint is present but its first next-IV scenario artifact was missing; the audit regenerates 512 CPU paths in eight batches of 64 from the authenticated saved EMA, using batch seeds 20261111--20261118. This new forecast does not pretend to be the missing original 10,000-path sample. The {may_ft_fits} new May FT fits continue the 30 April checkpoint, freeze the original base, and train only the rank-four output adapter for the same one selected epoch per date. All new parent/base hashes and training dates are retained. Forecast and recursive-test completion are shown in Table~\\ref{{tab:audit-coverage}}, not inferred from training completion. Recursive diffusion uses all 200 denoising steps each session and the original weekly adapter's context rule: spatially average the generated return channel and rebroadcast that scalar when appending each path's own next state. Per-session checkpoints retain context, generated IVs and RNG state.",
        r"Reference: Han, Zhang, Torres, Acero and Xu, Diffusion Models for Dynamic Volatility Surface Generation and Data-Driven Hedging, \href{https://arxiv.org/abs/2609.13402}{arXiv:2609.13402}.",
        r"\subsection{GROMM under P and Q}",
        r"P and Q share the same 44 dated affine-200 fits and rank-16 dynamics; they are not separately duplicated trained models. The retained native grid uses 200 interior strikes, NMS $(-3,2)$ and a changing union of reference/quoted expiries, not the proposed new $(-4,3)$ grid. The Black background annual volatility is 5\%. Affine fitting uses 2000 epochs, learning rate 0.01 and DLV-squared bounds $10^{-4}$--9. The original fit target is the released SANOS price panel with stored half-spreads, not observed midpoints. Stored training-window fit rates are therefore kept distinct from the direct licensed-quote measurements in this audit.",
        r"The retained artifacts were migrated by exact input/geometry reuse between public manifests; their source/target fingerprints and hashes remain in each dated manifest. Some migrated fits have no affine optimizer checkpoint or warm-start date. We do not silently claim they were sequentially retrained. P uses the accepted attainable physical risk-price projection; Q omits that Brownian drift shift. Both use eight Brownian DLV substeps and the original numerical scheme, including its numerical safeguards. An AR-related physical drift estimate is not a global linear VAR transition for the decoded risk-neutral surface; spectral radius is therefore n/a.",
        r"The stored native explained-variance ratio of 1.0 on the first fit uses Gaussian-smoothed, centered native log-DLV innovations restricted to the fitted strike core, including the spot projection plus residual factors. It is not 100\% coverage of raw full-grid IV changes or an out-of-sample claim. This report instead measures all models directly in the same observable space.",
        r"The new observable audit generates 512 antithetic paths (256 shock pairs) per date and measure with seed 20261111 on CPU, using the saved fits without training. It prices the native density tower plus background factor and converts to floating common-grid IVs. Terminal tenors are aged correctly: tomorrow's 1D row comes from today's 2D expiry. This differs from the retained 10,000-path price-return books used for the quoted-book forecast audit. Monte Carlo uncertainty of 512 paths and 44 realized dates must not be conflated. Common random seeds across origins are intentional, not additional independent market observations. Q-versus-P empirical comparisons are explicitly labelled descriptive.",
        r"The retained GROMM integrator tames an Euler relative-drift increment $x$ only when $|x|>0.5$, replacing it with $\operatorname{sgn}(x)[0.5+\log(1+|x|-0.5)]$ before the exponential variance update. This is an implementation stabilization, not the unchanged continuous-time drift restriction and not a proof of exact finite-step martingality. Taming frequency was not retained in the new observable extracts, so no zero-intervention claim is made. Float32 saved density inputs are decoded in float64 and normalized only within the source-precision mass-error tolerance; the diagnostic logs preserve the errors and tolerances.",
        r"\subsection{Hedge wrappers and non-generative baselines}",
        r"Greek, parametric and statistical hedge wrappers do not constitute three separately trained surface generators when they reuse the same DYSANOS state and dynamics. Their underlying-law diagnostics are shared. Unhedged, Black delta and Black delta-plus-vega are hedging baselines, not joint stochastic IV-surface models, so covariance, PCA persistence and linear-core spectral radius are not fabricated for them. No hedge objective, risk aversion, transaction-cost setting or optimizer was changed or rerun by this audit.",
        r"\subsection{Inference-only model retention}",
        f"At the user's request, the diffusion models now have an EMA-only inference archive with exact content-addressed backbone sharing. {len(inference['models'])} dated exports have been checked tensor by tensor and through two seeded complete 200-step CPU sampling paths against their original checkpoints, with zero differences. Geometry, versions, normalization, dated adapters and historical source hashes are retained. {len(retired_models)} redundant restart checkpoints ({retired_gib:.3f} GiB gross) have been retired; the inference archive occupies {inference_gib:.3f} GiB. This preserves inference, not exact optimizer/RNG restart. Original fitted laws are unchanged; inference operability is not a calibration-quality pass. Other models, historical contexts, retained scenarios and results have not been deleted by this step.",
        r"\section{Reproducibility and missing evidence}",
        r"Licensed data and fitted parameters are unchanged. Large models, retained scenario arrays and licensed quote rows remain external; only aggregates, diagnostics and checksummed manifests are suitable for Git. Retired checkpoint identities resolve through \path{INFERENCE_MODELS.json}; \path{model_audit/inference.py} loads the verified inference replacements without optimizer or training data.",
        r"The report is generated by \path{quantdata/hedging/model_audit/report.py}; \path{run.py}, \path{persistence.py}, \path{gromm.py} and \path{quotes.py} reproduce its stages. Interpreter: \path{/home/hans/vqdata/bin/python}. Jobs set explicit \texttt{QUANTDATA\_ROOT} and \texttt{PYTHONPATH}. Neural recursive paths use a managed GPU; other audit stages run on CPU. Disjoint origin batches preserve each origin's seed, saved context and RNG state. No diffusion denoising steps are dropped.",
        r"The JSON daily files retain failure reasons, no-data exclusions, quantile forecasts, full conditional covariance and source diagnostics. CSVs provide coverage, daily/monthly/rolling errors, per-expiry/NMS log-IV RMSE, tail tests and linear-core stability. Raw modeled means are retained: this is a model-law audit, not the demeaned hedge-optimization experiment. The 44-day report cannot establish 63-day observed ACF or reliable 1\% tail calibration. Missing diagnostics remain n/a.",
        r"\end{document}"]
    tex = output / "MODEL_DYNAMICS_REPORT.tex"
    tex.write_text("\n\n".join(body) + "\n")
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=output, capture_output=True, text=True)
        (output / "latex-build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"LaTeX compilation failed; see {output / 'latex-build.log'}")
    manifest = {"schema": "hedging-model-dynamics-audit-v1", "client_date": "2026-11-11",
        "models": list(MODEL_ORDER), "requested_origins": 44, "code_files": {}, "report_files": {}}
    for path in Path(__file__).parent.glob("*.py"):
        manifest["code_files"][path.name] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    for pattern in ("*.csv", "*.json", "*.pdf", "*.tex", "*.npz"):
        for path in output.glob(pattern):
            if path.name != "MANIFEST.json":
                manifest["report_files"][path.name] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    _json(output / "MANIFEST.json", manifest)
    stats = ["# April–May 2024 retained-model audit", "", f"Audit of 44 origins; existing fits unchanged. {may_ft_fits} missing May FT fits added by the original causal rolling recipe; no hedge reruns.", "",
             "The common observable PCA basis is frozen before April. Native rolling PCs are not concatenated.", "",
             "| Model | Surface days | Conditional trace ratio | Pooled trace ratio | Relative covariance error |",
             "|---|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        item = summaries[model]
        if item.get("surface_days", 0) > 1:
            stats.append(f"| {model} | {item['surface_days']} | {item['conditional_covariance']['trace_ratio']:.6g} | {item['pooled_covariance']['trace_ratio']:.6g} | {item['pooled_covariance']['cov_frob_rel']:.6g} |")
    stats += ["", "Full definitions, quote scores, tail diagnostics, PCA/ATM persistence and model-choice appendices are in `MODEL_DYNAMICS_REPORT.pdf`.",
        "", f"Inference retention: {len(inference['models'])} exact EMA exports validated; {len(retired_models)} full restart checkpoints retired ({retired_gib:.3f} GiB gross). Compact archive: {inference_gib:.3f} GiB. Historical contexts, scenarios and other model files remain retained.",
        "", f"Coverage limits: base diffusion has 43 retained surface forecasts plus one regenerated saved-checkpoint forecast; fine-tuned diffusion has {summaries['AD-Seq-Vol-FT']['surface_days']} scored forecasts and {22+may_ft_fits} trained dates. Every recursive test uses 32 paths over 115 sessions. See coverage.csv for actual completed dates. All 1%/5% tail inference is exploratory with 44 or fewer realized dates.",
        "", "Traffic lights: pass black, warn orange, fail red. Except for the AR stationarity boundary and conventional exact-binomial p-value cutoffs, thresholds are explicit retrospective engineering tolerances, not confidence bounds or arbitrage proofs. See TRAFFIC_LIGHT_RULES.json and the report's threshold table.",
        "", "DYSANOS uses our own legacy nine-tenor fits to 0.1-SANOS, not the unimplemented corrected 13-tenor specification. Its floored/unidentified coordinates and numerical guards are disclosed, not validated away.",
        "", "VolGAN's pooled variance must not be interpreted as conditional dispersion.",
        "", f"Stationarity scope: the {dysanos_rank} floating-volatility PCA–AR(1) coordinates only. Spot-level stationarity is not required or tested. Gaussian stock-return shocks contribute to the factor innovation covariance without changing the AR stability criterion.",
        "", "| DYSANOS backbone | Stable fits | Minimum AR radius | Maximum AR radius | Maximum modal half-life (business sessions) |",
        "|---|---:|---:|---:|---:|"]
    for model in MODEL_ORDER[:2]:
        subset = stability.loc[stability.model == model]
        stats.append(f"| {model} | {int(subset.stationary_backbone.sum())}/{len(subset)} | {subset.spectral_radius.min():.6f} | {subset.spectral_radius.max():.6f} | {subset.max_modal_half_life.max():.6f} |")
    (output / "STATS.md").write_text("\n".join(stats) + "\n")
    manifest["report_files"]["STATS.md"] = {"sha256": _hash(output / "STATS.md"), "bytes": (output / "STATS.md").stat().st_size}
    _json(output / "MANIFEST.json", manifest)
    print(f"Compiled {output / 'MODEL_DYNAMICS_REPORT.pdf'}", flush=True)


if __name__ == "__main__":
    main()
