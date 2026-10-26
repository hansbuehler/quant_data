"""Plot official, filtered-data, discrete-model and analytic-model VIX together."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

from quantdata.sdlv.daily.vix_comparison import _metrics

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SERIES = (
    ("cboe_vix", "#171717", "VIX: official Cboe close", "-"),
    ("sanos_input", "#0072B2", "Data VIX: filtered SANOS input spreads", "-"),
    ("gromm_discrete", "#D55E00", "GROMM VIX: market-selected discrete strip", "-"),
    ("gromm_friday_continuous", "#CC79A7", "Analytic GROMM VIX: full-strike variance swap", "--"),
)


def run(*, model_directory: Path, filtered_directory: Path, output_directory: Path) -> dict:
    """Use identical expiry interpolation and common dates for fit statistics."""
    model = pl.read_parquet(model_directory / "vix_comparison.parquet")
    filtered = pl.read_parquet(filtered_directory / "vix_comparison.parquet")
    model_report = json.loads((model_directory / "report.json").read_text())
    columns = [name for name, *_ in SERIES]
    frame = model.join(filtered.select("date", "sanos_input"), on="date", how="left")
    common = frame.filter(pl.all_horizontal([pl.col(name).is_finite().fill_null(False) for name in columns]))
    if not common.height:
        raise ValueError("No common complete dates for the four-way comparison")
    # No silent date shifts or mixing of two official-series vintages.
    actuals = model.select("date", "cboe_vix").join(filtered.select("date", "cboe_vix"), on="date", suffix="_filtered")
    np.testing.assert_allclose(actuals["cboe_vix"], actuals["cboe_vix_filtered"], rtol=0, atol=0)
    metrics = {name: _metrics(common[name].to_numpy(), common["cboe_vix"].to_numpy()) for name in columns[1:]}
    model_label = model_report.get("model_label", "Gromm-v1")
    def sha(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(model_label=model_label, common_observations=common.height,
                  common_sample_metrics=metrics,
                  reference_date=frame.filter(pl.col("date") == date(2025, 8, 29)).select("date", *columns)
                    .with_columns(pl.col("date").cast(pl.String)).to_dicts(),
                  inputs={str(folder / "vix_comparison.parquet"): sha(folder / "vix_comparison.parquet")
                          for folder in (model_directory, filtered_directory)},
                  code_sha256=sha(Path(__file__)),
                  conventions=["Bias, MAE and RMSE are volatility-point differences from official VIX on identical complete dates.",
                               "Analytic GROMM uses the full-strike log contract at the same two expiries and the same calendar interpolation as discrete GROMM; not direct evaluation at a different 30-day business-clock point.",
                               "Data VIX uses market input spreads, not SANOS model prices. Filtered data are an incomplete strip.",
                               "GROMM discrete cutoffs are determined by raw observed bids, not by invented model bid/ask quotes.",
                               "Contemporaneous option-surface fits; no VIX fitting, rescaling or out-of-sample forecast claim."])
    output_directory.mkdir(parents=True, exist_ok=False)
    frame.select("date", *columns).write_csv(output_directory / "four_series.csv")
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    x = frame["date"].to_list()
    for name, color, label, linestyle in SERIES:
        y = frame[name].to_numpy().astype(float)
        axes[0].plot(x, y, color=color, lw=1.05, ls=linestyle, label=label)
        if name != "cboe_vix":
            axes[1].plot(x, y - frame["cboe_vix"].to_numpy(), color=color, lw=.9, ls=linestyle)
    axes[0].set_title(f"{model_label} — historical VIX comparison\nSame expiry pair and 30-calendar-day total-variance interpolation")
    axes[0].set_ylabel("VIX (volatility points)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("Difference from official VIX\n(volatility points)")
    axes[1].set_xlabel(f"Fit statistics use {common.height} common dates; gaps are unavailable model maturities, not extrapolated values")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output_directory / "four_series.png", dpi=180)
    fig.savefig(output_directory / "four_series.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--filtered-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
