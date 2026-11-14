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


def rows(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def legacy_blue(tex: str) -> str:
    """Preserve historic values and make all their text, math and former ratings blue."""
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = body.replace(r"\maketitle", "").replace(r"\appendix", "")
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
    for name in ("reconstruction", "next_day"):
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
    standalone_path = root.parent / "standalone-v2-paper-metrics-20261114/STANDALONE_V2_METRICS.json"
    forecast_dir = root.parent / "standalone-v2-forecast-nms-20261114"
    standalone = json.loads(standalone_path.read_text()) if standalone_path.exists() else None
    standalone_forecast = json.loads((forecast_dir / "SUMMARY.json").read_text()) if (forecast_dir / "SUMMARY.json").exists() else None
    if standalone is not None and standalone["dates"] != dates:
        raise ValueError("Standalone V2 statistics have different comparison dates")
    forecast_count = standalone_forecast["scored"] if standalone_forecast else 0
    coverage = [["DYSANOS v1-P", v1_count, "paper fits; scoring pending"],
        ["Standalone DYSANOS v2-P", len(dates) if standalone else 0, f"{forecast_count}/44 one-day forecasts"],
        ["STRT-16 alternative", 44, "earlier separate experiment"],
        ["GROMM2-P/Q", g2_count, "fits; P/Q forecasts pending"]]
    metrics = {"schema": "current-rolling-model-report-v2", "origins": dates, "earlier_results": {"STRT-16 alternative": summary},
        "legacy_models": list(COMPETITORS), "static_checks": checks, "static_max_iv": float(static.iv_max.max()),
        "invalid_iv_count": int(static.invalid_iv_count.sum()), "above_300pct_iv_count": int(static.above_300pct_iv_count.sum()),
        "spectral_radius_range": [min(radius), max(radius)], "coverage": coverage,
        "standalone_v2": standalone, "standalone_v2_forecast": standalone_forecast,
        "standalone_v2_forecast_protocol_sha256": _hash(forecast_dir / "PROTOCOL.json") if standalone_forecast else None,
        "objective_audit": "/home/hans/python/quant_data-hedging/quantdata/hedging/OBJECTIVES.md"}
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
                r"Forecast changes are measured relative to each model's own initial surface. Standalone V2 spot shocks are transformed deterministically from the fitted dividend-adjusted return to the common one-day-forward convention. Long-horizon recursive stability and static-arbitrage simulation remain pending for this standalone model."]
        else:
            standalone_section.append(r"Matched one-day forecast covariance, long-horizon recursive stability and static-arbitrage simulation remain pending for this standalone model.")
    section = [r"\section*{Executive update: 14 November 2026}",
        r"\textbf{Standalone DYSANOS v2 and the earlier STRT-16 alternative are distinct models.} Blue values throughout this document are earlier, unrefreshed evidence, not current refits. Current black/orange/red ratings retain their stated diagnostic meanings. Blue overrides the earlier traffic-light colors in the historical appendix, including plot numbers.",
        table(["Model", "Completed origins", "Current stage"], coverage,
            caption="Model-fit counts out of 44 April--May 2024 origins. Training does not imply forecast or diagnostic completion. Black: ungraded counts; no missing result is a pass.", label="tab:current-coverage"),
        *standalone_section,
        r"\begingroup\color{blue}",
        r"\section{Earlier STRT-16 alternative and competitors}",
        rf"The STRT-16 alternative predicts {variance['conditional_variance_over_forecast_mse']:.3f} times realized squared forecast-error variance. Its common log-IV forecast RMSE is {summary['forecast_rmse']:.4f}, versus {summary['persistence_rmse']:.4f} for persistence. All 44 fitted floating VAR backbones are stable, with spectral radii {min(radius):.4f}--{max(radius):.4f}. No tested static-arbitrage violation or unusable IV occurs in these checks. These are guarded-decoder results, not unguarded diffusion or dynamic-martingale proofs.",
        r"All rows use the same 44 forecast dates, common 49-coordinate floating log-IV grid and pre-April diagnostic PCA. Let $m_t=E_t[\Delta x]$ and $C_t=\operatorname{Cov}_t(\Delta x)$. Conditional trace divides $\operatorname{tr}(\bar C)$ by observed change covariance trace; pooled trace adds the across-origin covariance of $m_t$. These are different from forecast variance/MSE, $\sum_t\operatorname{tr}C_t/\sum_t\|\Delta x_t-m_t\|^2$. Smaller pooled Frobenius error and forecast RMSE are better; RMSE/persistence below one beats the zero-change forecast. None is an arbitrage test.",
        table(["Model", "Dates", "Cond. trace", "Pooled trace", "Frob. pooled", "RMSE", "/persist."], covariance_rows,
            caption="Earlier STRT-16 and competitor results, not the standalone DYSANOS v2. Blue entries are retained earlier evidence, not current refits.", label="tab:current-covariance"),
        r"STRT-16 is an earlier spot-orthogonal tangent projection followed by a separately fitted daily VAR and scalar tail diffusion. Native spot loading outside the retained span is preserved. All fits use independent causal windows, without earlier-origin weights. Decoder guards activate frequently, so static validity does not establish calibrated diffusion tails.",
        r"\subsection{Current reconstruction and next-day price forecast}",
        table(["Mode", "Quotes", "Inside spread %", "Mean outside", "MAE / spot"], quote_rows,
            caption="Earlier STRT-16 on the matched diagnostic book: OTM, 1<business-days-to-expiry<=5 and absolute NMS<=1.5. Statistics are quote-count weighted across dates. Outside distance uses full spread width. Next-day point forecasts versus realized quotes are distinct from same-day reconstruction.", label="tab:current-quotes"),
        r"\subsection{Static arbitrage and simulated-IV stability}",
        r"Each of 44 origins uses 1,000 paths, nested simulation days 1, 5, 10, 125 and 259, twenty-one NMS strikes in [-3,2], and remaining tenors 1, 2, 5 and 10 business days. Calendar checks use the next tenor in the native model grid at the same cash strikes. No separate stock path is needed. Prices and vector IV inversion use the existing decoder and cdxcore.bs without rejected-path filtering.",
        table(["Day", "Tenor", "Decr.", "Convex", "Vertical", "Intrinsic", "Upper", "K=0", "Calendar", "Bad IV"], static_rows,
            caption="Earlier STRT-16 violation counts summed across all 44 origins at each horizon and tenor. Zero violations in finite paths do not prove absence of arbitrage.", label="tab:current-static"),
        rf"The largest decoded annual IV is {100*metrics['static_max_iv']:.2f}\%; no IV exceeds 300\%. No inversion is unusable. Numerical DLV/forward bounds remain active and are disclosed in the dated records. Historical-range exceedances are recorded, not interpreted as arbitrage.",
        r"\subsection{Frozen observable-PC persistence and tails}",
        table(["Lag", "Model PC1", "PC2", "PC3", "History PC1", "PC2", "PC3"], acf_rows,
            caption="Mean within-path Pearson ACF: first 32 of the same 1,000-path stream, 115 sessions, all 44 origins. Historical columns average each origin's causal training window, not a 44-day held-out ACF. A fixed pre-April observable basis is used throughout. Black: ungraded descriptive comparisons; no color inference is assigned.", label="tab:current-acf"),
        table(["Quantile", "Dates", "PC1 exceptions", "Expected", "Exact p", "Pinball"], tail_rows,
            caption="Earlier STRT-16 PC1 pointwise tail diagnostics. With only 44 dates all tails remain exploratory.", label="tab:current-tails"),
        r"\endgroup",
        r"\subsection{Outstanding work and objective differences}",
        r"DYSANOS v1 uses the paper's full joint affine-200/five-session VAR. The replacement run centers its quote loss on observed bid/ask midpoints; the earlier SANOS-centered trial is not reused as a paper-objective fit. An intermediate, unaccepted weekly covariance from that earlier trial failed the daily-embedding positivity check; no eigenvalue clipping is used. No completed 44-origin result is claimed. GROMM2 uses a separately identified calibration with actual observed bid/ask, the paper's linear excess and 0.05 midpoint term plus restoring cap penalty. The old pseudo-Huber chunks are not loaded. Its requested GROMM2 geometry and latent-X dependence estimator remain explicit extensions. GPU training and GPU-dependent forecasts share one managed allocation, released when that work ends; remaining statistics run on CPU.",
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
            "Standalone long-horizon recursive and static diagnostics remain pending.\n"
            "Earlier STRT-16 is a different model; its old guarded-decoder results are blue.\n"
            "DYSANOS v1 and paper-objective repository GROMM2 remain incomplete.\n")
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
