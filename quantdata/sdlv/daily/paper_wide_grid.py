"""Build the retained main21 addendum tables and VIX comparison from final reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

from quantdata.sdlv.daily import FITS
from quantdata.sdlv.daily.vix_comparison import _metrics

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SERIES = (
    ("market_discrete", "Raw-market strip"),
    ("sanos_input", "Filtered SANOS input"),
    ("v1_discrete", "GROMM v1, discrete"),
    ("v1_analytic", "GROMM v1, analytic"),
    ("wide_discrete", "GROMM v1-4,3, discrete"),
    ("wide_analytic", "GROMM v1-4,3, analytic"),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison_frame(baseline: pl.DataFrame, wider: pl.DataFrame, filtered: pl.DataFrame) -> pl.DataFrame:
    """Reject date shifts or benchmark changes before computing paired comparisons."""
    for frame in (baseline, wider, filtered):
        if frame["date"].n_unique() != frame.height:
            raise ValueError("Duplicate dates in VIX input")
        if frame["date"].to_list() != baseline["date"].to_list():
            raise ValueError("VIX date grids differ")
        np.testing.assert_allclose(frame["cboe_vix"], baseline["cboe_vix"], rtol=0, atol=0)
    np.testing.assert_allclose(wider["market_discrete"], baseline["market_discrete"], rtol=0, atol=0)
    columns = {"gromm_discrete": "discrete", "gromm_friday_continuous": "analytic"}
    return baseline.select("date", "cboe_vix", "market_discrete").with_columns(
        *[frame[name].alias(prefix + suffix)
          for frame, prefix in ((baseline, "v1_"), (wider, "wide_")) for name, suffix in columns.items()],
        filtered["sanos_input"],
    )


def _table(headers: list[str], rows: list[list[str]], *, caption: str, label: str, alignment: str) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (r"\begin{table}[H]" + "\n" + r"\centering\small" + "\n"
            + r"\begin{tabular}{" + alignment + "}\n" + r"\toprule" + "\n"
            + " & ".join(headers) + r" \\" + "\n" + r"\midrule" + "\n" + body + "\n"
            + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
            + r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n" + r"\end{table}" + "\n")


def _write_tables(report: dict, output: Path) -> None:
    fit = report["wide_fit"]
    groups = fit["fit_by_weight_group"]
    rows = [("Original v1, original range", report["baseline_fit"]),
            ("Wider fit, original range", groups["paper_nms_region"]),
            ("Wider fit, core", groups["core"]), ("Wider fit, wings", groups["tail"]),
            ("Wider fit, all quotes", fit["overall"]),
            ("Wider fit, 29 Aug. (all)", fit["reference_date"]["2025-08-29"])]
    rendered = [[label, f"{row['num_quotes']:,}", f"{100 - row['percent_inside_bid_ask']:.3f}\\%",
                 f"{row['mean_excess_half_spreads']:.5f}", f"{row['conditional_mean_excess_half_spreads']:.5f}"]
                for label, row in rows]
    (output / "static_fit.tex").write_text(_table(
        ["Fit / quote subset", "Quotes", r"\% excess", "Mean excess", "Cond. mean"], rendered,
        caption=(r"Static-fit errors beyond the bid--ask interval, in half-spread units. All statistics "
                 r"are unweighted quote-level summaries, not the weighted training objective. Conditional "
                 r"means include only quotes with positive excess. The original-range rows cover the same "
                 r"2,192,458 retained quotes; the final row is a separate single-date summary."),
        label="tab:wide-static-fit", alignment="lrrrr"))
    audit = report["q_audit"]
    unconditional = audit["unconditional"]["all_instruments"]
    conditional = {key: value["combined_out_of_fold"] for key, value in audit["conditional"].items()}
    row = [str(unconditional["tested_coordinate_count"]), f"{unconditional['max_absolute_z']:.3f}",
           f"{unconditional['p99_absolute_z']:.3f}", str(unconditional["count_above_2_576"]),
           str(unconditional["count_above_5"]), f"{conditional['squared']['z_score']:.3f}",
           f"{conditional['entropy']['z_score']:.3f}"]
    (output / "q_audit.tex").write_text(_table(
        [r"$N$", r"Max. $|z|$", r"99\% $|z|$", r"$N_{2.576}$", r"$N_5$", r"$T_{\rm sqr}$", r"$T_{\rm ent}$"],
        [row], caption=(r"Fresh GROMM v1-4,3 restricted-universe $\Q$ audit: 100,000 paths, "
                        r"50,000 antithetic pairs, 20 business days, spot plus 44 options; eight factors "
                        r"and eight steps per day. The statistics and colour conventions are those of "
                        r"Appendix~\ref{sec:wide-audit-definition}."),
        label="tab:wide-q-audit", alignment="rrrrrrr"))
    rows = []
    for key, label in SERIES:
        metric = report["vix_metrics"][key]
        rows.append([label, f"{metric['bias_vol_points']:+.3f}", f"{metric['mae_vol_points']:.3f}",
                     f"{metric['rmse_vol_points']:.3f}", f"{metric['correlation']:.5f}",
                     f"{report['reference_date'][key]:.3f}"])
    (output / "vix_metrics.tex").write_text(_table(
        ["Series", "Bias", "MAE", "RMSE", "Corr.", "29 Aug."], rows,
        caption=(r"Historical VIX comparison on the same 696 complete dates. Errors and the "
                 r"reference-date levels are in volatility percentage points; the benchmark is the "
                 r"official Cboe close (15.360 on 29 August 2025). Analytic values use the full-strike "
                 r"log contract at the same two expiries and the same calendar interpolation as the "
                 r"discrete strip, not direct evaluation at a different 30-day business maturity."),
        label="tab:wide-vix", alignment="lrrrrr"))
    macros = {"WideVixMaeReduction": f"{report['discrete_mae_reduction_percent']:.2f}",
              "WideCoreCoverage": f"{groups['core']['percent_inside_bid_ask']:.3f}",
              "WideExplainedVariance": f"{100 * audit['calibration']['smoothed_explained_variance_fraction']:.2f}",
              "WideOverallConditionalMedian": f"{fit['overall']['conditional_median_excess_half_spreads']:.5f}",
              "WideOverallMidpointMae": f"{fit['overall']['midpoint_mae_half_spreads']:.5f}"}
    for name, value in conditional.items():
        prefix = "WideSquared" if name == "squared" else "WideEntropy"
        for field, suffix in (("mean_daily_conditional_drift", "Mean"), ("pair_based_standard_error", "Se")):
            macros[prefix + suffix] = f"{value[field] * 1e9:.3f}"
        macros[prefix + "CiLow"] = f"{value['confidence_interval_95'][0] * 1e9:.3f}"
        macros[prefix + "CiHigh"] = f"{value['confidence_interval_95'][1] * 1e9:.3f}"
    (output / "numbers.tex").write_text("% Generated from completed reports; see metrics.json.\n" + "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) + "\n")


def _plot(frame: pl.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.9), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.8, 1]})
    dates = frame["date"].to_list()
    top = (("cboe_vix", "#171717", "Official Cboe VIX", 1.2),
           ("sanos_input", "#999999", "Filtered-market VIX", 0.85),
           ("v1_discrete", "#D55E00", "GROMM v1, discrete", 1.0),
           ("wide_discrete", "#0072B2", "GROMM v1-4,3, discrete", 1.0))
    for name, color, label, width in top:
        axes[0].plot(dates, frame[name].to_numpy(), label=label, color=color, lw=width)
    for prefix, color, label in (("v1", "#D55E00", "v1"), ("wide", "#0072B2", "v1-4,3")):
        for kind, style in (("discrete", "-"), ("analytic", "--")):
            error = frame[f"{prefix}_{kind}"].to_numpy() - frame["cboe_vix"].to_numpy()
            axes[1].plot(dates, error, color=color, lw=0.9, ls=style, label=f"{label}, {kind}")
    axes[0].set_title("Historical VIX: original and wider-grid GROMM v1\nCompleted fits; identical expiry selection and calendar interpolation", fontsize=12)
    axes[0].set_ylabel("VIX (volatility points)", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Model minus official VIX\n(volatility points)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    axes[1].set_xlabel("Date (statistics use 696 common observations)", fontsize=11)
    for axis in axes:
        axis.grid(alpha=0.2)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"gromm_v1_wide_vix.{extension}", dpi=180)
    plt.close(fig)


def run(*, training_run: Path, audit_directory: Path, baseline_vix: Path, filtered_vix: Path,
        output_directory: Path, publish_directory: Path | None = None) -> dict:
    """Generate tables, figure and provenance; never train or change model artifacts."""
    wider_vix = training_run / "vix-comparison"
    fit_path, audit_path = training_run / "fit/report.json", audit_directory / "q-audit.json"
    fit, audit = (json.loads(path.read_text()) for path in (fit_path, audit_path))
    reports = {key: json.loads((folder / "report.json").read_text())
               for key, folder in (("baseline", baseline_vix), ("wide", wider_vix))}
    if fit["status"] != "complete" or fit["completed_epochs"] != 1000 or audit["status"] != "complete":
        raise ValueError("The paper requires completed training and Q diagnostics")
    expected_audit = dict(paths=100000, factor_rank=8, dlv_substeps=8, horizon_days=20, instrument_count=45)
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ValueError("Q audit does not match the reported restricted-universe geometry")
    if audit["source"]["fit_epoch"] != 1000 or audit["source"]["fit_status"] != "complete":
        raise ValueError("Interim Q results cannot enter the completed-fit paper")
    if audit["source"]["dataset_sha256"] != fit["dataset_sha256"]:
        raise ValueError("Fit and Q diagnostics refer to different training datasets")
    for key in ("checkpoint_sha256", "dataset_sha256"):
        if reports["wide"][key] != audit["source"][key]:
            raise ValueError(f"VIX and Q audit use different {key}")
    if reports["wide"]["checkpoint_sha256"] != sha256(Path(fit["checkpoint"])):
        raise ValueError("Final wider checkpoint does not match its results")
    if reports["baseline"]["checkpoint_sha256"] != sha256(FITS["full"].checkpoint):
        raise ValueError("Baseline VIX is not from the paper's retained v1 checkpoint")
    inputs = [fit_path, audit_path, FITS["full"].report]
    frames = []
    for folder in (baseline_vix, wider_vix, filtered_vix):
        source = folder / "vix_comparison.parquet"
        frames.append(pl.read_parquet(source))
        inputs.extend([source, folder / "report.json"])
    frame = comparison_frame(*frames)
    # Identical market contracts and expiry clocks isolate model repricing.
    for filename, columns in (
        ("selected_contracts.parquet", ["date", "expiry", "is_am", "strike", "delta_strike", "market_price", "side"]),
        ("expiry_audit.parquet", ["date", "expiry", "is_am", "calendar_days", "business_days", "forward", "k0", "discount_factor"]),
    ):
        left, right = (pl.read_parquet(folder / filename).select(columns) for folder in (baseline_vix, wider_vix))
        if not left.equals(right):
            raise ValueError(f"Baseline and wider market selection differ: {filename}")
        inputs.extend([baseline_vix / filename, wider_vix / filename])
    common = frame.filter(pl.all_horizontal([pl.col(key).is_finite().fill_null(False) for key, _ in SERIES]))
    if common.height != 696:
        raise ValueError(f"Expected the audited 696-date comparison, found {common.height}")
    metrics = {key: _metrics(common[key].to_numpy(), common["cboe_vix"].to_numpy()) for key, _ in SERIES}
    for prefix, report_key in (("v1", "baseline"), ("wide", "wide")):
        for kind, old_key in (("discrete", "gromm_discrete"), ("analytic", "gromm_friday_continuous")):
            for metric_key in ("bias_vol_points", "mae_vol_points", "rmse_vol_points", "correlation"):
                np.testing.assert_allclose(metrics[f"{prefix}_{kind}"][metric_key],
                                           reports[report_key]["common_sample_metrics"][old_key][metric_key], atol=1e-12)
    reference = frame.filter(pl.col("date") == date(2025, 8, 29)).with_columns(pl.col("date").cast(pl.String)).to_dicts()[0]
    report = dict(schema="gromm-main21-wide-grid-v1", wide_fit=fit, q_audit=audit,
                  baseline_fit=json.loads(FITS["full"].report.read_text())["final_statistics"],
                  vix_metrics=metrics, common_dates=common.height, reference_date=reference,
                  discrete_mae_reduction_percent=100 * (1 - metrics["wide_discrete"]["mae_vol_points"] / metrics["v1_discrete"]["mae_vol_points"]),
                  unavailable_reasons=reports["wide"]["unavailable_reasons"],
                  input_sha256={str(path.resolve()): sha256(path) for path in inputs},
                  source_code_sha256=sha256(Path(__file__)),
                  conventions=reports["wide"]["conventions"])
    output_directory.mkdir(parents=True, exist_ok=False)
    frame.write_csv(output_directory / "vix_series.csv")
    (output_directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_tables(report, output_directory)
    _plot(frame, output_directory)
    if publish_directory is not None:
        publish_directory.mkdir(parents=True, exist_ok=True)
        for artifact in output_directory.iterdir():
            shutil.copy2(artifact, publish_directory / artifact.name)
    print(json.dumps(dict(output=str(output_directory), common_dates=common.height,
                          vix_metrics=metrics, mae_reduction_percent=report["discrete_mae_reduction_percent"]), indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-run", "audit-directory", "baseline-vix", "filtered-vix", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publish-directory", type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
