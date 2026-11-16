"""Publish completed rolling results immediately, with prior evidence visibly blue."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedging.model_audit import report
from quantdata.hedging.model_audit.grading import tail_grade
from quantdata.hedging.model_audit.run import _hash, _json

COMPETITORS = ("FuNVol", "VolGAN PCA8", "AD-Seq-Vol", "AD-Seq-Vol-FT")
STATIC_CONDITIONS = ("decreasing", "convexity", "vertical_spread", "intrinsic_lower_bound",
                     "upper_bound", "zero_strike_identity", "calendar")


def rows(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def literature_static_summary(records: list[dict], dates: list[str], *, horizons: tuple[int, ...] = (1, 5, 10, 125, 259)) -> pd.DataFrame:
    """Aggregate saved checks without treating unavailable prices as passes."""
    if sorted(record["origin_date"] for record in records) != sorted(dates):
        raise ValueError("Literature static results do not cover exactly the approved origins")
    expected = {(day, tenor) for day in horizons for tenor in (1., 2., 5., 10.)}
    entries = []
    for record in records:
        checks = record["results"]
        if record["status"] != "scored" or {(row["simulation_day"], row["business_dte"]) for row in checks} != expected or len(checks) != len(expected):
            raise ValueError("Incomplete literature static-arbitrage simulation")
        for row in checks:
            entry = {"simulation_day": row["simulation_day"], "business_dte": row["business_dte"],
                "price_nodes": row["paths"] * row["strikes"], "unavailable_prices": row["invalid_price_nodes"]}
            for name in STATIC_CONDITIONS:
                entry[f"{name}_violations"] = row.get(f"{name}_violations", 0) or 0
                entry[f"{name}_tests"] = row.get(f"{name}_tests", 0) or 0
            for name in ("count", "library_min_vol_cases", "library_max_vol_cases", "failed"):
                entry[f"iv_{name}"] = sum(item[name] for item in row.get("inversion", []))
            entries.append(entry)
    grouped = pd.DataFrame(entries).groupby(["simulation_day", "business_dte"], as_index=False).sum()
    for name in STATIC_CONDITIONS:
        denominator = grouped[f"{name}_tests"]
        grouped[f"{name}_pct"] = 100. * grouped[f"{name}_violations"].div(denominator.where(denominator > 0))
    grouped["unavailable_pct"] = 100. * grouped["unavailable_prices"] / grouped["price_nodes"]
    for name in ("library_min_vol_cases", "library_max_vol_cases", "failed"):
        grouped[f"iv_{name}_pct"] = 100. * grouped[f"iv_{name}"].div(grouped["iv_count"].where(grouped["iv_count"] > 0))
    return grouped


def legacy_blue(tex: str) -> str:
    """Preserve historic values and make all their text, math and former ratings blue."""
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = body.replace(r"\maketitle", "").replace(r"\appendix", "")
    for label in ("bidask-fit", "density-quotes"):
        pattern = (r"\\begin\{table\}\[H\](?:(?!\\end\{table\}).)*?\\label\{tab:" + label
            + r"\}(?:(?!\\end\{table\}).)*?\\end\{table\}")
        def same_day_table(match):
            lines = match.group().splitlines()
            start = lines.index(r"\midrule") + 1
            end = next(index for index in range(start, len(lines)) if lines[index].startswith(r"\bottomrule"))
            retained = [" & ".join(line.split(" & ")[:2]) + r" \\" for line in lines[start:end]]
            return (r"\begin{table}[H]\centering\small" + "\n"
                + r"\caption{Earlier same-day quoted-spread reconstruction. Blue entries are historical results.}\label{tab:" + label + "}\n"
                + r"\begin{tabular}{@{}lr@{}}\toprule Model & Reconstruction in spread \\" + "\n"
                + r"\midrule" + "\n" + "\n".join(retained) + "\n"
                + r"\bottomrule\end{tabular}\end{table}")
        body = re.sub(pattern, same_day_table, body, flags=re.DOTALL)
    body = body.replace(r"\section{Bid--ask fit and next-day prediction}", r"\section{Same-day bid--ask fit}")
    body = re.sub(r"Next-day prediction uses surviving OTM hedge-book contracts.*?\n\n", "", body, flags=re.DOTALL)
    body = re.sub(r"A correct stochastic model need not put its conditional-mean forecast.*?\n\n", "", body, flags=re.DOTALL)
    body = re.sub(r"Next-day forecast in-spread rates and mean outside distances.*?Price-prediction grading instead compares MAE with the same contracts' origin observed midpoints carried by the origin-only one-day discount factor\. ", "", body, flags=re.DOTALL)
    body = body.replace(", tail/ACF mismatches remain, and next-day quoted-price MAE is slightly worse than persistence", ", and tail/ACF mismatches remain")
    body = re.sub(r"\\textcolor\{(?:black|orange|red)\}", r"\\textcolor{blue}", body)
    body = re.sub(r"\\includegraphics(\[[^]]*\])?\{([^}]+)\}",
        lambda match: r"\includegraphics" + (match[1] or "") + "{legacy/" + match[2] + "}", body)
    return r"\begingroup\color{blue}" + "\n" + body + "\n" + r"\endgroup"


def legacy_figures(source: Path, output: Path) -> None:
    """Replot retained numbers without changing them; old tick/annotation values are blue."""
    destination = output / "legacy"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("observable_reference.json", "covariance_extension.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    summaries = json.loads((source / "summary.json").read_text())
    reference = json.loads((source / "observable_reference.json").read_text())
    order = tuple(json.loads((source / "MODEL_SELECTION.json").read_text())["models"])
    records = {model: rows(source / "daily" / model.replace(" ", "_")) for model in order}
    persistence = {model: rows(source / "persistence" / model.replace(" ", "_")) for model in order}
    original = report.MODEL_ORDER
    report.MODEL_ORDER = order
    try:
        with plt.rc_context({"text.color": "blue", "axes.labelcolor": "blue", "xtick.color": "blue",
                "ytick.color": "blue", "legend.labelcolor": "blue", "pdf.fonttype": 42}):
            report._plot_covariance(destination, summaries)
            report._plot_tails(destination, records)
            report._plot_acf(destination, reference, persistence)
            weighted = json.loads((source / "density_weighted_diagnostics.json").read_text())
            names = tuple(entry["model"] for entry in weighted["models"])
            inputs = json.loads((source / "DENSITY_WEIGHTED_INPUTS.json").read_text())
            directory = next(Path(entry["path"]).parents[2] for entry in inputs if "/daily/" in entry["path"])
            weighted_output = destination / "weighted"
            weighted_output.mkdir(exist_ok=True)
            report._plot_tails(weighted_output,
                {model: rows(directory / "daily" / model.replace(" ", "_")) for model in names}, model_order=names)
            report._plot_acf(weighted_output, reference,
                {model: rows(directory / "persistence" / model.replace(" ", "_")) for model in names})
            shutil.move(weighted_output / "tails.pdf", destination / "density_tails.pdf")
            shutil.move(weighted_output / "acf.pdf", destination / "density_acf.pdf")
    finally:
        report.MODEL_ORDER = original


def updated_section(root: Path, gromm: Path, source: Path, output: Path) -> tuple[list[str], dict]:
    summary = json.loads((root / "DYSANOS_v2-P-summary.json").read_text())
    reference = json.loads((source / "observable_reference.json").read_text())
    dates = reference["dates"]
    daily = rows(root / "daily/DYSANOS_v2-P")
    diagnostics = rows(root / "diagnostics/v2-full-20261114/daily")
    persistence = rows(root / "diagnostics/v2-persistence-20261114/daily")
    for name, values in (("forecasts", daily), ("quote/static checks", diagnostics), ("persistence", persistence)):
        actual = sorted(row.get("origin_date", row.get("origin")) for row in values)
        if actual != dates:
            raise ValueError(f"V2 {name} do not cover exactly the approved 44 origins")
    for entry in daily:
        baseline = json.loads((source / f"daily/FuNVol/{entry['origin_date']}.json").read_text())
        np.testing.assert_allclose(entry["actual_change"], baseline["actual_change"], rtol=0., atol=1e-12,
            err_msg="Current and historical model rows do not share the same realized observable target")
    if any(row["status"] != "complete" for row in diagnostics + persistence):
        raise ValueError("V2 has a recorded diagnostic failure; review before rendering a completion claim")
    old = json.loads((source / "summary.json").read_text())
    current = {"STRT-16 alternative": summary}
    current.update({model: old[model] for model in COMPETITORS})
    covariance_rows = []
    for model, item in current.items():
        vintage = True
        values = [item["surface_days"], item["conditional_covariance"]["trace_ratio"],
            item["pooled_covariance"]["trace_ratio"], item["pooled_covariance"]["cov_frob_rel"],
            item["forecast_rmse"], item["forecast_rmse"] / item["persistence_rmse"]]
        covariance_rows.append([model, *[report.Rated(value, "old") if vintage else value for value in values]])
    variance = summary["forecast_variance_calibration"]
    radius = [row["backbone"]["spectral_radius"] for row in diagnostics]
    static = pd.DataFrame([entry for row in diagnostics for entry in row["static"]])
    conditions = ("decreasing", "convexity", "vertical_spread", "intrinsic_lower_bound", "upper_bound", "zero_strike_identity", "calendar")
    checks = {name: {"violations": int(static[f"{name}_violations"].sum()), "tests": int(static[f"{name}_tests"].sum())}
        for name in conditions}
    static_rows = []
    for (horizon, tenor), group in static.groupby(["simulation_day", "business_dte"]):
        static_rows.append([int(horizon), int(tenor), *[report.Rated(int(group[f"{name}_violations"].sum()),
            "pass" if group[f"{name}_violations"].sum() == 0 else "fail") for name in conditions],
            int(group.invalid_iv_count.sum())])
    quote_rows = []
    for name in ("reconstruction",):
        counts = np.array([row["quotes"][name]["count"] for row in diagnostics])
        weight = counts / counts.sum()
        quote_rows.append([name.replace("_", " "), int(counts.sum()),
            100 * float(weight @ np.array([row["quotes"][name]["in_spread_rate"] for row in diagnostics])),
            float(weight @ np.array([row["quotes"][name]["out_dist_mean"] for row in diagnostics])),
            float(weight @ np.array([row["quotes"][name]["price_mae"] / row["quotes"]["origin_spot"] for row in diagnostics]))])
    acf_rows = []
    for lag in (1, 5, 21, 63):
        modeled = np.array([next(entry["correlations"] for entry in row["observable_acf"] if entry["lag"] == lag) for row in persistence])
        historical = np.array([next(entry["correlations"] for entry in row["historical_observable_acf"] if entry["lag"] == lag) for row in persistence])
        acf_rows.append([lag, *modeled.mean(0), *historical.mean(0)])
    tail_rows = [[level, item["count"], report.Rated(item["exceptions"], tail_grade(item)), item["expected_exceptions"],
        item["exact_binomial_p"], item["pinball_q"]] for level, item in summary["tails"]["PC1"].items() if level != "ES"]
    v1_root = root.parent / "paper-dysanos1-rolling-20261114"
    v1_count = sum((v1_root / f"fits/dysanos1/{day}/FIT_COMPLETE.json").exists() for day in dates)
    g2_count = sum((gromm / f"gromm2/{day}/ESTIMATION_COMPLETE.json").exists() for day in dates)
    g2_forecasts = sum((gromm / f"gromm2/{day}/FORECAST_COMPLETE.json").exists() for day in dates)
    standalone_path = root.parent / "standalone-v2-paper-metrics-20261114/STANDALONE_V2_METRICS.json"
    forecast_dir = root.parent / "standalone-v2-forecast-nms-20261114"
    standalone = json.loads(standalone_path.read_text()) if standalone_path.exists() else None
    standalone_forecast = json.loads((forecast_dir / "SUMMARY.json").read_text()) if (forecast_dir / "SUMMARY.json").exists() else None
    if standalone is not None and standalone["dates"] != dates:
        raise ValueError("Standalone V2 statistics have different comparison dates")
    forecast_count = standalone_forecast["scored"] if standalone_forecast else 0
    coverage = [["DYSANOS v1-P", v1_count, "stopped at user request"],
        ["Standalone DYSANOS v2-P", len(dates) if standalone else 0, f"{forecast_count}/44 one-day forecasts"],
        ["STRT-16 alternative", 44, "earlier separate experiment"],
        ["GROMM2-P/Q", g2_count, f"fits; {g2_forecasts}/44 paired P/Q forecasts"]]
    metrics = {"schema": "current-rolling-model-report-v2", "origins": dates, "earlier_results": {"STRT-16 alternative": summary},
        "legacy_models": list(COMPETITORS), "static_checks": checks, "static_max_iv": float(static.iv_max.max()),
        "invalid_iv_count": int(static.invalid_iv_count.sum()), "above_300pct_iv_count": int(static.above_300pct_iv_count.sum()),
        "spectral_radius_range": [min(radius), max(radius)], "coverage": coverage,
        "standalone_v2": standalone, "standalone_v2_forecast": standalone_forecast,
        "standalone_v2_forecast_protocol_sha256": _hash(forecast_dir / "PROTOCOL.json") if standalone_forecast else None,
        "objective_audit": "/home/hans/python/quant_data-hedging/quantdata/hedging/OBJECTIVES.md"}
    gromm_section = []
    gromm_rows, gromm_summaries = [], {}
    for measure in ("P", "Q"):
        path = gromm / f"GROMM2-{measure}-summary.json"
        if not path.exists():
            continue
        item = json.loads(path.read_text())
        gromm_summaries[measure] = item
        gromm_rows.append([measure, item["surface_days"], item["conditional_covariance"]["trace_ratio"],
            item["pooled_covariance"]["trace_ratio"], item["pooled_covariance"]["cov_frob_rel"],
            item["forecast_rmse"], item["forecast_variance_calibration"]["conditional_variance_over_forecast_mse"]])
    if gromm_rows:
        metrics["gromm2_partial_forecasts"] = gromm_summaries
        gromm_section = [r"\section{GROMM2: matched forecasts}",
            r"The model is fitted independently at each origin using the repository model and the paper quote objective. Results below use the available origins only; Q against realized data is descriptive, not a claim that Q is the physical law. Long-horizon diagnostics and full-origin static-arbitrage diagnostics remain outstanding.",
            report._table(["Measure", "Dates", "Cond. trace", "Pooled trace", "Frob.", "RMSE", "Variance/MSE"], gromm_rows,
                caption="GROMM2 one-day forecasts, 1,000 paths per origin and measure. Dates show completed origins out of 44. Black: descriptive, ungraded statistics.", label="tab:gromm2-partial")]
        leverage_rows, gromm_tail_rows, monthly_rows, rolling_rows, covariance_detail = [], [], [], [], []
        for measure, summary_item in gromm_summaries.items():
            covariance_detail.append([measure, summary_item["pooled_covariance"]["eig_share_err_k"],
                summary_item["pooled_covariance"]["subspace_angle_max"]])
            for index, (model, observed) in enumerate(zip(summary_item["spot_pc_correlations_model"],
                    summary_item["spot_pc_correlations_data"], strict=True), start=1):
                leverage_rows.append([measure, index, model, observed, model - observed])
            for level, entry in summary_item["tails"]["PC1"].items():
                if level != "ES":
                    gromm_tail_rows.append([measure, level, entry["count"], report.Rated(entry["exceptions"], tail_grade(entry)),
                        entry["expected_exceptions"], entry["exact_binomial_p"], entry["pinball_q"]])
            forecasts = rows(gromm / f"daily/GROMM2-{measure}")
            if len(forecasts) >= 21:
                window_ratios = []
                for start in range(len(forecasts) - 20):
                    window = forecasts[start:start + 21]
                    realized = np.asarray([row["actual_change"] for row in window])
                    expected = np.asarray([row["surface_mean"] for row in window])
                    denominator = float(np.sqrt(np.mean(realized ** 2)))
                    window_ratios.append(float(np.sqrt(np.mean((realized - expected) ** 2))) / denominator if denominator > 0 else None)
                first, last = window_ratios[0], window_ratios[-1]
                rolling_rows.append([measure, 21, len(window_ratios), first, last,
                    last / first if first is not None and last is not None and first > 0 else None])
            rolling_rows.append([measure, 63, max(0, len(forecasts) - 62), None, None, None])
            for month in sorted({row["origin_date"][:7] for row in forecasts}):
                selected = [row for row in forecasts if row["origin_date"].startswith(month)]
                observed = np.asarray([row["actual_change"] for row in selected])
                predicted = np.asarray([row["surface_mean"] for row in selected])
                error = float(np.sqrt(np.mean((observed - predicted) ** 2)))
                baseline = float(np.sqrt(np.mean(observed ** 2)))
                monthly_rows.append([measure, month, len(selected), error, baseline, error / baseline if baseline > 0 else None])
        gromm_section.extend([
            report._table(["Measure", "Eigen-share error", "Max angle (degrees)"], covariance_detail,
                caption="GROMM2 pooled observable covariance: error of the first three eigenvalue shares and maximum angle of the leading three-dimensional subspaces against the data. Black: descriptive, ungraded.", label="tab:gromm2-covariance-detail"),
            report._table(["Measure", "Observable PC", "Model spot corr.", "Data spot corr.", "Difference"], leverage_rows,
                caption="GROMM2 pooled spot--surface correlations in the fixed observable PCA basis. P and Q use the same observed data; Q comparison is descriptive. Black: ungraded.", label="tab:gromm2-leverage"),
            report._table(["Measure", "Quantile", "Dates", "Exceptions", "Expected", "Exact p", "Pinball"], gromm_tail_rows,
                caption="GROMM2 one-day observable PC1 tail diagnostics. Forty-four dates make 1% and 5% tails exploratory. Colors apply only to exception counts; other entries are ungraded.", label="tab:gromm2-tails"),
            report._table(["Measure", "Month", "Dates", "RMSE", "Persistence", "Ratio"], monthly_rows,
                caption="GROMM2 one-day observable log-IV forecast errors by month; ratio compares the model with the zero-change persistence forecast. These are surface-dynamics diagnostics, not next-day option-price alpha tests. Black: descriptive, ungraded.", label="tab:gromm2-monthly"),
            report._table(["Measure", "Window", "Windows", "First ratio", "Last ratio", "Last / first"], rolling_rows,
                caption="GROMM2 rolling persistence-normalized log-IV RMSE: first and last full windows and their ratio. The 44-date sample cannot form a 63-date window. Black: descriptive, ungraded.", label="tab:gromm2-rolling")])
        fit_rows = []
        for day in dates:
            folder = gromm / f"gromm2/{day}"
            if (folder / "ESTIMATION_COMPLETE.json").exists():
                fitted = json.loads((folder / "fit.json").read_text())["current"]
                if fitted["date"] != day or fitted["kind"] != "start":
                    raise ValueError(f"GROMM2 current fit does not match {day}")
                fit_rows.append(fitted)
        if fit_rows:
            total = sum(row["reported"] for row in fit_rows)
            weighted = lambda key: sum(row["reported"] * row[key] for row in fit_rows) / total
            metrics["gromm2_same_day_quote_fit"] = {"origins": len(fit_rows), "reported_quotes": total,
                "in_spread_rate": weighted("coverage"), "mean_outside_half_spreads": weighted("mean_excess"),
                "mid_mae_half_spreads": weighted("mid_mae")}
            gromm_section.append(report._table(["Dates", "Quotes", "Inside spread %", "Mean outside / half-spread", "MAE / half-spread"],
                [[len(fit_rows), total, 100 * weighted("coverage"), weighted("mean_excess"), weighted("mid_mae")]],
                caption="GROMM2 same-day observed-quote fit on the repository's reported calibration population, pooled by quote count. Outside distance and midpoint MAE are in units of half-spread. This is a different quote population from the standalone V2 table. Black: descriptive and ungraded.", label="tab:gromm2-same-day-fit"))
    gromm_static = root.parent / "repository-gromm2-static-full-20261115/static"
    for measure in ("P", "Q"):
        saved = rows(gromm_static / measure)
        if not saved:
            continue
        if any(row["origin"] not in dates or row["model"] != f"repository GROMM2-{measure}" or row["paths"] != 1000 for row in saved):
            raise ValueError("GROMM2 static results do not match the approved 1,000-path comparison")
        grouped = literature_static_summary([{"origin_date": row["origin"], "status": "scored", "results": row["results"]}
            for row in saved], [row["origin"] for row in saved], horizons=(1, 5, 10))
        grouped.to_csv(output / f"current_gromm2_{measure.lower()}_static.csv", index=False)
        counts = [[int(row.simulation_day), int(row.business_dte),
            *[int(getattr(row, f"{name}_violations")) for name in STATIC_CONDITIONS], int(row.unavailable_prices)]
            for row in grouped.itertuples()]
        boundaries = [[int(row.simulation_day), int(row.business_dte), row.iv_library_min_vol_cases_pct,
            row.iv_library_max_vol_cases_pct, row.iv_failed_pct] for row in grouped.itertuples()]
        metrics.setdefault("gromm2_static_partial", {})[measure] = {"origins": len(saved), "paths": 1000,
            "source": str(gromm_static / measure), "unsupported_horizons": [125, 259]}
        gromm_section.extend([rf"GROMM2-{measure} static checks currently cover {len(saved)} of 44 origins. The fixed-expiry tower does not support simulation days 125 or 259 without an unapproved expiry-replenishment extension.",
            report._table(["Day", "Tenor", "Decr.", "Convex", "Vert.", "Lower", "Upper", "K=0", "Cal.", "N/A"], counts,
                caption=f"GROMM2-{measure} static violation counts on completed origins only, with 1,000 paths per origin. N/A counts unavailable price nodes; a zero count does not prove no arbitrage.", label=f"tab:gromm2-{measure.lower()}-static"),
            report._table(["Day", "Tenor", "IV 1%", "IV 500%", "Reprice fail %"], boundaries,
                caption=f"GROMM2-{measure} library-implied volatility boundary and repricing-failure percentages among finite prices. Boundary estimates are retained, and boundary and repricing-failure counts can overlap.", label=f"tab:gromm2-{measure.lower()}-static-iv")])
    recursive_root = root.parent / "repository-gromm2-recursive-full-20261116/recursive"
    for measure in ("P", "Q"):
        saved = rows(recursive_root / measure)
        if not saved:
            continue
        if any(row["origin"] not in dates or row["model"] != f"repository GROMM2-{measure}"
                or row["paths"] != 32 or row["total_static_paths"] != 1000 or row["steps"] != 115 for row in saved):
            raise ValueError("GROMM2 recursive results do not match the approved comparison")
        valid = [row for row in saved if row["status"] == "complete"]
        metrics.setdefault("gromm2_recursive_partial", {})[measure] = {"origins": len(saved),
            "valid_origins": len(valid), "source": str(recursive_root / measure)}
        if valid:
            acf = []
            for lag in (1, 5, 21, 63):
                model = np.asarray([next(item["correlations"] for item in row["observable_acf"] if item["lag"] == lag) for row in valid])
                observed = np.asarray([next(item["correlations"] for item in row["historical_observable_acf"] if item["lag"] == lag) for row in valid])
                acf.append([lag, *model.mean(axis=0), *observed.mean(axis=0)])
            invalid = np.mean([row["invalid_path_rate"] for row in saved])
            extreme = np.mean([row["extreme_surface_rate"] for row in valid])
            gromm_section.extend([rf"GROMM2-{measure} recursive checks cover {len(saved)} of 44 origins, with {len(valid)} origins admitting at least one valid path. Mean invalid-path rate: {100*invalid:.3f}\%; mean extreme-surface rate among valid paths: {100*extreme:.3f}\%.",
                report._table(["Lag", "Model PC1", "PC2", "PC3", "History PC1", "PC2", "PC3"], acf,
                    caption=f"GROMM2-{measure} mean within-path observable PCA autocorrelations. The 32 simulated paths are the leading paths of the 1,000-path static audit; historical columns use causal training windows. Black: descriptive, ungraded.", label=f"tab:gromm2-{measure.lower()}-acf")])
    literature_root = root.parent / "competitor-static-20261112-v1"
    literature = {}
    literature_tables = []
    for model, label in (("FuNVol", "funvol"), ("VolGAN PCA8", "volgan")):
        records = rows(literature_root / "static/daily" / model.replace(" ", "_"))
        if not records:
            continue
        result = literature_static_summary(records, dates)
        result.insert(0, "model", model)
        literature[model] = result
        violations = [[int(row.simulation_day), int(row.business_dte), *[getattr(row, f"{name}_pct") for name in STATIC_CONDITIONS], row.unavailable_pct]
            for row in result.itertuples()]
        boundaries = [[int(row.simulation_day), int(row.business_dte),
            100. * row.iv_count / row.price_nodes, row.iv_library_min_vol_cases_pct,
            row.iv_library_max_vol_cases_pct, row.iv_failed_pct] for row in result.itertuples()]
        literature_tables.extend([report._table(["Day", "Tenor", "Decr.", "Convex", "Vert.", "Lower", "Upper", "K=0", "Cal.", "N/A"], violations,
            caption=f"{model}: percentages of eligible finite price tests violating each condition by simulation day and business-day tenor. N/A is the percentage of price nodes unavailable; unsupported calendar tests show a dash.", label=f"tab:literature-static-{label}"),
            report._table(["Day", "Tenor", "Priced %", "IV 1%", "IV 500%", "Reprice fail %"], boundaries,
            caption=f"{model}: price availability and library IV boundary percentages by simulation day and tenor. Priced uses all requested price nodes; other columns use finite prices submitted to inversion. Boundary entries and repricing failures can overlap.", label=f"tab:literature-boundary-{label}")])
    if literature:
        pd.concat(literature.values(), ignore_index=True).to_csv(output / "current_literature_static.csv", index=False)
        metrics["literature_static"] = {"models": list(literature), "origins": len(dates),
            "protocol_path": str(literature_root / "STATIC_PROTOCOL.json"),
            "protocol_sha256": _hash(literature_root / "STATIC_PROTOCOL.json"),
            "source_daily_csv_sha256": _hash(literature_root / "static_arb_daily.csv")}
    diffusion_tables, diffusion_frames = [], []
    for variant, model in (("base", "AD-Seq-Vol"), ("ft", "AD-Seq-Vol-FT")):
        partial_root = root.parent / f"diffusion-static-jax-{variant}-20261114/static/partial" / model
        grouped = {}
        for partial in sorted(partial_root.glob("*/day-*.json")):
            record = json.loads(partial.read_text())
            if record["model"] != model or record["origin_date"] not in dates:
                raise ValueError("Diffusion partial simulation provenance does not match the comparison")
            day = record["results"][0]["simulation_day"]
            grouped.setdefault(day, []).append({**record, "status": "scored"})
        if not grouped:
            continue
        frames = []
        for day, records in sorted(grouped.items()):
            current = literature_static_summary(records, [record["origin_date"] for record in records], horizons=(day,))
            current.insert(0, "origins", len(records))
            frames.append(current)
        combined = pd.concat(frames, ignore_index=True)
        combined.insert(0, "model", model)
        diffusion_frames.append(combined)
        violations = [[int(row.simulation_day), int(row.business_dte), int(row.origins),
            *[getattr(row, f"{name}_pct") for name in STATIC_CONDITIONS], row.unavailable_pct] for row in combined.itertuples()]
        boundaries = [[int(row.simulation_day), int(row.business_dte), int(row.origins), 100. * row.iv_count / row.price_nodes,
            row.iv_library_min_vol_cases_pct, row.iv_library_max_vol_cases_pct, row.iv_failed_pct] for row in combined.itertuples()]
        diffusion_tables.extend([report._table(["Day", "Tenor", "Dates", "Decr.", "Convex", "Vert.", "Lower", "Upper", "K=0", "Cal.", "N/A"], violations,
            caption=f"{model}: preliminary percentages of eligible finite tests. Dates are completed origins out of 44. N/A is unavailable price nodes; unsupported calendar comparisons show a dash.", label=f"tab:diffusion-static-{variant}"),
            report._table(["Day", "Tenor", "Dates", "Priced %", "IV 1%", "IV 500%", "Reprice fail %"], boundaries,
            caption=f"{model}: preliminary library IV boundaries and repricing errors. Priced uses all requested price nodes; other columns use finite prices submitted to inversion. Dates are completed origins out of 44.", label=f"tab:diffusion-boundary-{variant}")])
    if diffusion_frames:
        pd.concat(diffusion_frames, ignore_index=True).to_csv(output / "current_diffusion_static_partial.csv", index=False)
        metrics["diffusion_static_partial"] = {frame.model.iloc[0]: {str(int(day)): int(group.origins.iloc[0])
            for day, group in frame.groupby("simulation_day")} for frame in diffusion_frames}
    v2_root = root.parent / "standalone-v2-full-diagnostics-20261115-v2"
    v2_records = rows(v2_root / "daily")
    v2_partial = rows(v2_root / "partial")
    v2_section = []
    if v2_partial:
        if any(row["origin"] not in dates for row in v2_partial):
            raise ValueError("Standalone V2 diagnostic origin is outside the comparison")
        checked = [{"origin_date": row["origin"], "status": "scored", "results": row["static"]} for row in v2_partial]
        grid = literature_static_summary(checked, [row["origin"] for row in v2_partial])
        grid.to_csv(output / "current_standalone_v2_static.csv", index=False)
        v2_static_rows = [[int(row.simulation_day), int(row.business_dte),
            *[getattr(row, f"{name}_pct") for name in STATIC_CONDITIONS], row.unavailable_pct] for row in grid.itertuples()]
        v2_count_rows = [[int(row.simulation_day), int(row.business_dte),
            *[int(getattr(row, f"{name}_violations")) for name in STATIC_CONDITIONS], int(row.unavailable_prices)] for row in grid.itertuples()]
        v2_iv_rows = [[int(row.simulation_day), int(row.business_dte), row.iv_library_min_vol_cases_pct,
            row.iv_library_max_vol_cases_pct, row.iv_failed_pct] for row in grid.itertuples()]
        v2_quote_rows = []
        for mode in ("reconstruction",):
            count = sum(row["quotes"][mode]["count"] for row in v2_partial)
            def weighted(key):
                return sum(row["quotes"][mode]["count"] * row["quotes"][mode][key] for row in v2_partial) / count
            mae = sum(row["quotes"][mode]["count"] * row["quotes"][mode]["price_mae"] / row["quotes"]["origin_spot"] for row in v2_partial) / count
            v2_quote_rows.append([mode.replace("_", " "), count, 100 * weighted("in_spread_rate"), weighted("out_dist_mean"), mae])
        complete = [row for row in v2_records if row.get("status") == "complete"]
        metrics["standalone_v2_full_diagnostics"] = {"quotes_and_static_origins": len(v2_partial),
            "long_complete_origins": len(complete), "failures": [row["origin"] for row in v2_records if row.get("status") != "complete"],
            "source": str(v2_root), "paths_per_origin": 1000}
        v2_section = [r"\subsection{Standalone V2: quote and simulation diagnostics}",
            rf"These diagnostics cover {len(v2_partial)} of 44 origins. Same-day reconstruction uses the origin surface and OTM quotes with 1<business-days-to-expiry$\leq5$ and absolute NMS$\leq1.5$. Black numbers are descriptive, not pass/fail grades.",
            report._table(["Mode", "Quotes", "Inside spread %", "Mean outside", "MAE / spot"], v2_quote_rows,
                caption="Standalone V2 quote-count-weighted results on the available origins. Outside distance uses full bid--ask spread width.", label="tab:standalone-v2-quotes"),
            report._table(["Day", "Tenor", "Decr.", "Convex", "Vert.", "Lower", "Upper", "K=0", "Cal.", "N/A"], v2_count_rows,
                caption="Standalone V2 counts of violated individual tests and unavailable price nodes, summed across the completed origins and 1,000 paths per origin. A zero calendar count may also mean no eligible tests; consult the percentage table.", label="tab:standalone-v2-static-counts"),
            report._table(["Day", "Tenor", "Decr.", "Convex", "Vert.", "Lower", "Upper", "K=0", "Cal.", "N/A"], v2_static_rows,
                caption="Standalone V2 static violation percentages of eligible tests, by simulation day and business-day tenor. N/A is unavailable price nodes; a dash denotes no eligible calendar tests. Zero observed violations is not a proof of no arbitrage.", label="tab:standalone-v2-static"),
            report._table(["Day", "Tenor", "IV 1%", "IV 500%", "Reprice fail %"], v2_iv_rows,
                caption="Standalone V2 library-implied volatility boundary and repricing-failure percentages among inverted finite prices. Boundary values are retained; boundary and repricing-failure counts can overlap.", label="tab:standalone-v2-iv")]
        if complete:
            v2_acf = []
            for lag in (1, 5, 21, 63):
                model = np.array([next(item["correlations"] for item in row["long"]["observable_acf"] if item["lag"] == lag) for row in complete])
                history = np.array([next(item["correlations"] for item in row["long"]["historical_observable_acf"] if item["lag"] == lag) for row in complete])
                v2_acf.append([lag, *model.mean(axis=0), *history.mean(axis=0)])
            v2_section += [rf"Long-path diagnostics are complete for {len(complete)} origins. Each uses 32 paths over 115 sessions in a fixed observable PCA basis. History uses each origin's causal training window.",
                report._table(["Lag", "Model PC1", "PC2", "PC3", "History PC1", "PC2", "PC3"], v2_acf,
                    caption="Standalone V2 mean within-path autocorrelations on completed origins. Black: descriptive, ungraded.", label="tab:standalone-v2-acf")]
            invalid = np.mean([row["long"]["invalid_path_rate"] for row in complete])
            extreme = np.mean([row["long"]["extreme_surface_rate"] for row in complete])
            v2_section.append(rf"Mean invalid long-path rate: {100*invalid:.3f}\%; mean extreme-surface rate: {100*extreme:.3f}\%.")
    _json(output / "CURRENT_RESULTS.json", metrics)
    static.to_csv(output / "current_v2_static_daily.csv", index=False)
    pd.DataFrame(quote_rows, columns=["mode", "quotes", "in_spread_pct", "mean_outside_spreads", "mae_spot_units"]).to_csv(output / "current_v2_quotes.csv", index=False)
    pd.DataFrame(acf_rows, columns=["lag", "model_pc1", "model_pc2", "model_pc3", "data_pc1", "data_pc2", "data_pc3"]).to_csv(output / "current_v2_acf.csv", index=False)
    table = report._table
    standalone_section = []
    if standalone is not None:
        shares = dict(zip(standalone["projection"]["factors"], standalone["projection"]["shares"], strict=True))
        quote = standalone["same_day_quotes"]
        standalone_section = [r"\section{Standalone DYSANOS v2: matched-date results}",
            r"This is the 41-expiry density-weighted standalone model, with sixteen factors, a 250-change age-weighted window and Gaussian joint spot/factor innovations. It is distinct from the older nine-expiry STRT-16 alternative. The input SANOS smoothness is 0.1; the common observable reference has smoothness 0.25.",
            table(["Dates", "Quotes", "Inside spread %", "Spot + 8 %", "Spot + 16 %"],
                [[len(dates), quote["count"], 100 * quote["inside_rate"], 100 * shares[8], 100 * shares[16]]],
                caption="Standalone V2 same-day reconstruction and out-of-sample straddle sensitivity-span capture. Each realized next-day change is projected ex post onto model sensitivities; these are not forecast variance or attainable trading returns. Black entries are descriptive and ungraded.", label="tab:standalone-v2-span")]
        if standalone_forecast and forecast_count == len(dates) and not standalone_forecast["failures"]:
            result = standalone_forecast["statistics"]
            calibration = result["forecast_variance_calibration"]
            standalone_records = rows(forecast_dir / "daily")
            if sorted(entry["origin_date"] for entry in standalone_records) != dates:
                raise ValueError("Standalone V2 forecast records do not cover the approved origins")
            standalone_radii = [entry["diagnostics"]["spectral_radius"] for entry in standalone_records]
            leverage_rows = [[index + 1, model, observed] for index, (model, observed) in enumerate(zip(
                result["spot_pc_correlations_model"], result["spot_pc_correlations_data"], strict=True))]
            standalone_tail_rows = [[level, item["count"], report.Rated(item["exceptions"], tail_grade(item)),
                item["expected_exceptions"], item["exact_binomial_p"], item["pinball_q"]]
                for level, item in result["tails"]["PC1"].items() if level != "ES"]
            standalone_section += [table(["Dates", "Cond. trace", "Pooled trace", "Frob.", "RMSE", "Variance/MSE"],
                [[forecast_count, result["conditional_covariance"]["trace_ratio"], result["pooled_covariance"]["trace_ratio"],
                    result["pooled_covariance"]["cov_frob_rel"], result["forecast_rmse"], calibration["conditional_variance_over_forecast_mse"]]],
                caption="Standalone V2 one-day Gaussian forecasts, 1,000 paths per origin, in the common 49-node observable grid. Conditional and pooled traces use observed change covariance as denominator; variance/MSE uses squared forecast errors. Black entries are descriptive and ungraded.", label="tab:standalone-v2-forecast"),
                table(["Observable PC", "Model spot corr.", "Data spot corr."], leverage_rows,
                    caption="Pooled spot--surface correlations in the fixed observable PCA basis. Black entries are descriptive and ungraded. Model and realized spot returns use the same forward numeraire.", label="tab:standalone-v2-leverage"),
                table(["Quantile", "Dates", "PC1 exceptions", "Expected", "Exact p", "Pinball"], standalone_tail_rows,
                    caption="Standalone V2 Gaussian one-day PC1 tail diagnostics. Forty-four dates are too few for conclusive 1% or 5% tail calibration.", label="tab:standalone-v2-tails"),
                rf"The maximum fitted VAR spectral radius is {max(standalone_radii):.4f}; this checks the fitted factor backbone, not the full nonlinear surface process.",
                r"Forecast changes are measured relative to each model's own initial surface. Standalone V2 spot shocks are transformed deterministically from the fitted dividend-adjusted return to the common one-day-forward convention."]
        else:
            standalone_section.append(r"Matched one-day forecast covariance, long-horizon recursive stability and static-arbitrage simulation remain pending for this standalone model.")
    standalone_section.extend(v2_section)
    section = [r"\section*{Executive update: 14 November 2026}",
        r"\textbf{Standalone DYSANOS v2 and the earlier STRT-16 alternative are distinct models.} Blue values throughout this document are earlier, unrefreshed evidence, not current refits. Current black/orange/red ratings retain their stated diagnostic meanings. Blue overrides the earlier traffic-light colors in the historical appendix, including plot numbers. Next-day conditional-mean price prediction is excluded: forecasting price drift is not the aim of this model comparison.",
        table(["Model", "Completed origins", "Current stage"], coverage,
            caption="Model-fit counts out of 44 April--May 2024 origins. Training does not imply forecast or diagnostic completion. Black: ungraded counts; no missing result is a pass.", label="tab:current-coverage"),
        *standalone_section,
        *gromm_section,
        r"\begingroup\color{blue}",
        r"\section{Earlier STRT-16 alternative and competitors}",
        rf"The STRT-16 alternative predicts {variance['conditional_variance_over_forecast_mse']:.3f} times realized squared forecast-error variance. Its common log-IV forecast RMSE is {summary['forecast_rmse']:.4f}, versus {summary['persistence_rmse']:.4f} for persistence. All 44 fitted floating VAR backbones are stable, with spectral radii {min(radius):.4f}--{max(radius):.4f}. No tested static-arbitrage violation or unusable IV occurs in these checks. These are guarded-decoder results, not unguarded diffusion or dynamic-martingale proofs.",
        r"All rows use the same 44 forecast dates, common 49-coordinate floating log-IV grid and pre-April diagnostic PCA. Let $m_t=E_t[\Delta x]$ and $C_t=\operatorname{Cov}_t(\Delta x)$. Conditional trace divides $\operatorname{tr}(\bar C)$ by observed change covariance trace; pooled trace adds the across-origin covariance of $m_t$. These are different from forecast variance/MSE, $\sum_t\operatorname{tr}C_t/\sum_t\|\Delta x_t-m_t\|^2$. Smaller pooled Frobenius error and forecast RMSE are better; RMSE/persistence below one beats the zero-change forecast. None is an arbitrage test.",
        table(["Model", "Dates", "Cond. trace", "Pooled trace", "Frob. pooled", "RMSE", "/persist."], covariance_rows,
            caption="Earlier STRT-16 and competitor results, not the standalone DYSANOS v2. Blue entries are retained earlier evidence, not current refits.", label="tab:current-covariance"),
        r"STRT-16 is an earlier spot-orthogonal tangent projection followed by a separately fitted daily VAR and scalar tail diffusion. Native spot loading outside the retained span is preserved. All fits use independent causal windows, without earlier-origin weights. Decoder guards activate frequently, so static validity does not establish calibrated diffusion tails.",
        r"\subsection{Same-day reconstruction}",
        table(["Mode", "Quotes", "Inside spread %", "Mean outside", "MAE / spot"], quote_rows,
            caption="Earlier STRT-16 same-day reconstruction on the matched diagnostic book: OTM, 1<business-days-to-expiry<=5 and absolute NMS<=1.5. Statistics are quote-count weighted across dates. Outside distance uses full spread width.", label="tab:current-quotes"),
        r"\subsection{Static arbitrage and simulated-IV stability}",
        r"Each of 44 origins uses 1,000 paths, nested simulation days 1, 5, 10, 125 and 259, twenty-one NMS strikes in [-3,2], and remaining tenors 1, 2, 5 and 10 business days. Calendar checks use the next tenor in the native model grid at the same cash strikes. No separate stock path is needed. Prices and vector IV inversion use the existing decoder and cdxcore.bs without rejected-path filtering.",
        table(["Day", "Tenor", "Decr.", "Convex", "Vertical", "Intrinsic", "Upper", "K=0", "Calendar", "Bad IV"], static_rows,
            caption="Earlier STRT-16 violation counts summed across all 44 origins at each horizon and tenor. Zero violations in finite paths do not prove absence of arbitrage.", label="tab:current-static"),
        rf"The largest decoded annual IV is {100*metrics['static_max_iv']:.2f}\%; no IV exceeds 300\%. No inversion is unusable. Numerical DLV/forward bounds remain active and are disclosed in the dated records. Historical-range exceedances are recorded, not interpreted as arbitrage.",
        r"\subsection{Literature-model static-arbitrage and IV boundaries}",
        r"The retained FuNVol and PCA8 VolGAN tests use the same 44 origins, 1,000 paths per origin, five simulation days and four tenor buckets described above. Percentages are pooled test-level counts, not percentages of paths. Decr., Convex, Vert., Lower, Upper, K=0 and Cal. are strike monotonicity, convexity, vertical spread, intrinsic and upper bounds, zero-strike identity and calendar checks. The 10D calendar comparison is unsupported where no later native tenor exists. The AD-Seq-Vol tables use only the counted completed origins.\footnote{The AD-Seq-Vol and AD-Seq-Vol-FT simulations were stopped at the user's request. Results are partial, their date counts vary by simulation horizon, and they are not a complete 44-origin comparison.}",
        r"The generated-surface repricer marks a whole surface unavailable when any log-IV is non-finite or too large for its numerical guard. These entries do not enter finite-test denominators and must not be counted as passes. Boundary percentages are reported separately from repricing errors: vectorized cdxcore returns 1\% and 500\% limits, and the lower limit can also reflect its intrinsic-price convention. A bounded IV need not reproduce the input price within tolerance. VolGAN is our trained PCA8 adaptation, not published model weights.",
        *literature_tables,
        *diffusion_tables,
        r"\subsection{Frozen observable-PC persistence and tails}",
        table(["Lag", "Model PC1", "PC2", "PC3", "History PC1", "PC2", "PC3"], acf_rows,
            caption="Mean within-path Pearson ACF: first 32 of the same 1,000-path stream, 115 sessions, all 44 origins. Historical columns average each origin's causal training window, not a 44-day held-out ACF. A fixed pre-April observable basis is used throughout. Black: ungraded descriptive comparisons; no color inference is assigned.", label="tab:current-acf"),
        table(["Quantile", "Dates", "PC1 exceptions", "Expected", "Exact p", "Pinball"], tail_rows,
            caption="Earlier STRT-16 PC1 pointwise tail diagnostics. With only 44 dates all tails remain exploratory.", label="tab:current-tails"),
        r"\endgroup",
        r"\subsection{Outstanding work and objective differences}",
        r"DYSANOS v1 used the paper's full joint affine-200/five-session VAR; the run was stopped at the user's request and its partial fits are retained. No completed 44-origin result is claimed. GROMM2 uses a separately identified calibration with actual observed bid/ask, the paper's linear excess and 0.05 midpoint term plus restoring cap penalty. The old pseudo-Huber chunks are not loaded. Its requested GROMM2 geometry and latent-X dependence estimator remain explicit extensions. The coverage table distinguishes completed fits and one-day forecasts from outstanding diagnostics.",
        r"\clearpage\appendix\section{Historical evidence: blue values are not updated refits}",
        r"The following earlier report is retained for traceability. Its numbers, former quality colors and plot labels are blue. Its completion claims refer to those older model variants only, not to the currently requested DYSANOS v1/v2 and repository GROMM2 refits. Refer to the current coverage table for outstanding work.",
    ]
    start = section.index(r"\begingroup\color{blue}")
    end = section.index(r"\endgroup", start)
    for index in range(start + 1, end):
        section[index] = re.sub(r"\\textcolor\{(?:black|orange|red)\}", r"\\textcolor{blue}", section[index])
    return section, metrics


def build(root: Path, gromm: Path, source: Path, output: Path, retain: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".report.lock"):
        section, metrics = updated_section(root, gromm, source, output)
        legacy_figures(source, output)
        old = (source / "MODEL_DYNAMICS_REPORT.tex").read_text()
        header = old.split(r"\begin{document}", 1)[0]
        header = header.replace(r"\date{11 November 2026}", r"\date{14 November 2026}")
        content = header + "\n" + r"\begin{document}\maketitle" + "\n" + "\n\n".join(section)
        content += "\n" + legacy_blue(old) + "\n" + r"\end{document}" + "\n"
        tex = output / "MODEL_DYNAMICS_REPORT.tex"
        tex.write_text(content)
        for iteration in range(2):
            result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=output, capture_output=True, text=True)
            (output / f"latex-build-{iteration}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"Current report compilation failed: {output}")
        (output / "STATS.md").write_text("# Current rolling-model results — 14 November 2026\n\n"
            "Standalone DYSANOS v2: 44/44 fitted dates; matched quote and ex-post sensitivity-span results are in CURRENT_RESULTS.json.\n"
            f"Standalone one-day forecasts scored: {metrics['standalone_v2_forecast']['scored'] if metrics['standalone_v2_forecast'] else 0}/44.\n"
            f"Standalone quote/static origins: {metrics.get('standalone_v2_full_diagnostics', {}).get('quotes_and_static_origins', 0)}/44; long-path origins: {metrics.get('standalone_v2_full_diagnostics', {}).get('long_complete_origins', 0)}/44.\n"
            "Earlier STRT-16 is a different model; its old guarded-decoder results are blue.\n"
            "Saved FuNVol and VolGAN PCA8 static checks include per-tenor violation and IV-boundary rates in current_literature_static.csv. AD-Seq-Vol partial results are in current_diffusion_static_partial.csv; simulations were stopped at the user's request.\n"
            "DYSANOS v1 was stopped at the user's request; "
            f"repository GROMM2: {metrics['coverage'][-1][1]}/44 fits; {metrics['coverage'][-1][2]}.\n")
        retain.mkdir(parents=True, exist_ok=True)
        selected = [tex, output / "MODEL_DYNAMICS_REPORT.pdf", output / "CURRENT_RESULTS.json", output / "STATS.md",
            *output.glob("current_*.csv"), *list((output / "legacy").glob("*.pdf"))]
        files = {}
        for path in selected:
            relative = path.relative_to(output)
            destination = retain / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files[str(relative)] = {"sha256": _hash(destination), "bytes": destination.stat().st_size}
        manifest = {"external_run": str(output), "prior_report_source": str(source), "new_results_source": str(root),
            "gromm_pending_run": str(gromm), "files": files, "large_files_external": True, "raw_licensed_quotes_included": False,
            "legacy_tex_sha256": _hash(source / "MODEL_DYNAMICS_REPORT.tex"), "code_sha256": _hash(Path(__file__))}
        _json(retain / "RETAINED_ARTIFACTS.json", manifest)
        _json(output / "RETAINED_ARTIFACTS.json", manifest)
        print(f"Updated report: {retain / 'MODEL_DYNAMICS_REPORT.pdf'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "gromm", "source", "output", "retain"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.gromm, args.source, args.output, args.retain)


if __name__ == "__main__":
    main()
