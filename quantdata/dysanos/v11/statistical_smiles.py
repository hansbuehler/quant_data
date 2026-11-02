"""Statistical option smiles of saved variance-index LEVEL distributions.

For each tenor and population, normalize by its own sample mean: Y=V/mean(V).
Thus Y-1 is centered and dimensionless, but is not standardized to unit
variance. Empirical option values on Y are inverted with unit Black maturity
and zero discounting. The result is total implied standard deviation, NOT an
annualized volatility or a risk-neutral market option quote. Index tenor is
not used as an option maturity. No dynamics are re-estimated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.util import is_filename

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

TENORS = tuple(range(5, 71, 5))
LABELS = {"historical": "Historical SANOS", "gaussian": "Gaussian AR", "lstm": "LSTM",
          "selective_ssm": "Mamba-style SSM"}
COLORS = {"historical": "#348abd", "gaussian": "#777777", "lstm": "#d95f02", "selective_ssm": "#7a5195"}


def implied_total_standard_deviation(
    strikes: np.ndarray, prices: np.ndarray, *, is_call: np.ndarray,
) -> np.ndarray:
    """Invert unit-forward undiscounted OTM prices with sqrt(T)=1.

    Zero empirical wing prices return zero, but the plotter excludes wings
    with insufficient observed payoffs rather than presenting these as a
    well-identified zero implied volatility.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)
    verify_inp(strikes.shape == prices.shape == is_call.shape, "Price, strike and call/put shapes must match")
    verify_inp(np.all(np.isfinite(strikes)) and np.all(strikes > 0), "Strikes must be finite and positive")
    verify_inp(np.all(np.isfinite(prices)) and np.all(prices >= 0), "Prices must be finite and nonnegative")
    verify_inp(np.all(is_call == (strikes >= 1)), "Use puts below the forward and calls at/above it")
    upper = np.where(is_call, 1., strikes)
    verify_inp(np.all(prices < upper), "Empirical OTM prices must be strictly below their upper bounds")
    result = np.zeros_like(prices)
    positive = prices > 1e-12
    if np.any(positive):
        fit = bs.implied(
            strikes[positive], prices[positive], is_call=is_call[positive], sqrtT=1.,
            price_tol=1e-8, vol_min=1e-7, vol_max=5., ret_only_vols=False,
            on_exceed_bounds="error",
        )
        if np.any(fit.failed) or np.max(np.abs(fit.fits - prices[positive])) > 1e-8:
            raise RuntimeError("Black inversion did not reach its stated price tolerance")
        result[positive] = fit.vols
    return result


def empirical_smile(
    levels: np.ndarray, *, strikes: np.ndarray, minimum_payoffs: int = 10,
) -> dict[str, np.ndarray | float | int]:
    """Price options on mean-normalized levels; retain missing-tenor counts.

    NaNs indicate unsupported historical tenors and are excluded per tenor.
    Infinite, zero or negative supplied levels are invalid, not filtered.
    All included observations have equal probability. Sample standard
    deviation uses ddof=1; the pricing expectation uses 1/N.
    """
    levels = np.asarray(levels, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    verify_inp(levels.ndim == 1, "Supply one tenor/population at a time")
    verify_inp(not np.any(np.isinf(levels)), "Infinite levels are invalid")
    verify_inp(strikes.ndim == 1 and len(strikes) > 0, "Need a nonempty strike vector")
    verify_inp(np.all(np.isfinite(strikes)) and np.all(strikes > 0), "Strikes must be finite and positive")
    verify_inp(np.all(np.diff(strikes) > 0), "Strikes must be strictly increasing")
    verify_inp(isinstance(minimum_payoffs, int) and minimum_payoffs >= 1, "Minimum payoff count must be a positive integer")
    observed = levels[~np.isnan(levels)]
    verify_inp(len(observed) >= 2 and np.all(observed > 0), "Need at least two positive observed levels")
    forward = float(observed.mean())
    normalized = observed / forward
    centered = normalized - 1
    calls = np.maximum(normalized[:, None] - strikes, 0).mean(axis=0)
    puts = np.maximum(strikes - normalized[:, None], 0).mean(axis=0)
    is_call = strikes >= 1
    otm = np.where(is_call, calls, puts)
    counts = np.where(is_call, (normalized[:, None] > strikes).sum(axis=0),
                      (normalized[:, None] < strikes).sum(axis=0))
    total_vol = implied_total_standard_deviation(strikes, otm, is_call=is_call)
    parity_error = float(np.max(np.abs(calls - puts - (1 - strikes))))
    if abs(float(centered.mean())) > 1e-12 or parity_error > 1e-12:
        raise RuntimeError("Normalization or empirical put-call parity failed")
    return {
        "sample_count": len(observed), "sample_mean_level": forward,
        "sample_std_level": float(observed.std(ddof=1)),
        "normalized_sample_std": float(centered.std(ddof=1)),
        "normalized_mean": float(normalized.mean()), "parity_max_error": parity_error,
        "call_prices": calls, "put_prices": puts, "otm_prices": otm,
        "is_call": is_call, "payoff_count": counts, "total_implied_std": total_vol,
        "supported": counts >= minimum_payoffs,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, source_run: Path, historic_targets: Path, output: Path, minimum_payoffs: int = 10) -> dict:
    """Reprice existing distributions on CPU, without changing source artifacts."""
    verify_inp(is_filename(output.name), "Output run name must be a valid filename")
    verify_inp(not output.exists(), "Use a fresh unique output run directory")
    history = pl.read_parquet(historic_targets)
    paths = {"gaussian": source_run / "gaussian_audit.npz",
             **{name: source_run / name / "audit_indices.npz" for name in ("lstm", "selective_ssm")}}
    samples = {"historical": np.column_stack([history[f"vix_{tenor}d"].to_numpy() for tenor in TENORS])}
    for name, path in paths.items():
        with np.load(path) as payload:
            samples[name] = payload["indices"].copy()
        verify_inp(samples[name].ndim == 2 and samples[name].shape[1] == len(TENORS), "Sample tenor geometry mismatch")
    source = json.loads((source_run / "comparison.json").read_text())["source"]
    verify_inp(_sha256(historic_targets) == source["target_sha256"], "Historical targets differ from the plotted calibration run")
    output.mkdir(parents=True, exist_ok=False)
    strikes = np.linspace(.60, 1.80, 61)
    results = {}
    records, summaries = [], []
    for name, values in samples.items():
        for j, tenor in enumerate(TENORS):
            smile = empirical_smile(values[:, j], strikes=strikes, minimum_payoffs=minimum_payoffs)
            results[name, tenor] = smile
            for k, strike in enumerate(strikes):
                records.append({"population": LABELS[name], "index_tenor_business_days": tenor,
                                "strike_over_sample_mean": float(strike), "centered_normalized_strike": float(strike - 1),
                                "call_price": float(smile["call_prices"][k]), "put_price": float(smile["put_prices"][k]),
                                "otm_price": float(smile["otm_prices"][k]), "is_call": bool(smile["is_call"][k]),
                                "total_implied_std": float(smile["total_implied_std"][k]),
                                "payoff_count": int(smile["payoff_count"][k]), "supported": bool(smile["supported"][k]),
                                "sample_count": smile["sample_count"]})
            atm = int(np.argmin(np.abs(strikes - 1)))
            summaries.append({"population": LABELS[name], "index_tenor_business_days": tenor,
                              **{k: v for k, v in smile.items() if not isinstance(v, np.ndarray)},
                              "atm_total_implied_std": float(smile["total_implied_std"][atm])})
        print(f"priced {LABELS[name]}: {len(TENORS)} tenors, {len(strikes)} strikes", flush=True)
    pl.DataFrame(records).write_csv(output / "statistical_option_smiles.csv")
    pl.DataFrame(summaries).write_csv(output / "summary.csv")
    # Compare on the same supported strike/tenor cells for every population.
    squared_vol_errors = {name: [] for name in paths}
    squared_price_errors = {name: [] for name in paths}
    for tenor in TENORS:
        common = np.logical_and.reduce([results[name, tenor]["supported"] for name in samples])
        reference = results["historical", tenor]
        for name in paths:
            squared_vol_errors[name].extend((results[name, tenor]["total_implied_std"][common]
                                            - reference["total_implied_std"][common])**2)
            squared_price_errors[name].extend((results[name, tenor]["call_prices"][common]
                                              - reference["call_prices"][common])**2)
    comparison = {
        LABELS[name]: {"total_implied_std_rmse": float(np.sqrt(np.mean(squared_vol_errors[name]))),
                       "normalized_call_price_rmse": float(np.sqrt(np.mean(squared_price_errors[name]))),
                       "common_supported_points": len(squared_vol_errors[name])}
        for name in paths
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    def plot(tenors: tuple[int, ...], *, rows: int, cols: int, filename: str) -> None:
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.1 * rows), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.99, bottom=.18 if rows == 2 else .105,
                            top=.83 if rows == 2 else .91, hspace=.32, wspace=.13)
        for tenor, ax in zip(tenors, axes.flat, strict=False):
            for name in samples:
                smile = results[name, tenor]
                vols = np.where(smile["supported"], 100 * smile["total_implied_std"], np.nan)
                ax.plot(strikes - 1, vols, color=COLORS[name], linewidth=2 if name == "historical" else 1.5,
                        linestyle="--" if name == "gaussian" else "-", label=LABELS[name])
            n = results["historical", tenor]["sample_count"]
            ax.set_title(f"{tenor}D index; historical N={n}", fontsize=11)
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(tenors):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .002), ncol=4, frameon=False, fontsize=10)
        fig.supxlabel(r"Centered, mean-normalized strike: $K/\overline{V}-1$", fontsize=11, y=.075 if rows == 2 else .047)
        fig.supylabel("Black total implied standard deviation (%)", fontsize=11, x=.01)
        fig.suptitle("Statistical smiles of VIX-type level distributions\nEach population normalized by its own mean; no option-time annualization", fontsize=13)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    plot(TENORS, rows=4, cols=4, filename="statistical_smiles_all_tenors.png")
    plot((5, 10, 20, 30, 50, 70), rows=2, cols=3, filename="statistical_smiles_overview.png")
    report = {
        "schema": "dysanos-statistical-level-smiles-v1", "status": "complete",
        "source_run": str(source_run), "source_files_sha256": {str(p): _sha256(p) for p in (*paths.values(), historic_targets)},
        "historical_period": [source["first_date"], source["last_date"]],
        "normalization": "Y=V/mean(V), X=Y-1; own mean for each population and index tenor; NOT unit-variance standardization",
        "option_prices": "Equal-weight empirical E[(Y-k)+], E[(k-Y)+]; no risk-neutral reweighting; discount=1",
        "volatility_convention": "Black total implied standard deviation; sqrt(T)=1, NOT annualized by index tenor",
        "plot_filter": f"At least {minimum_payoffs} strictly positive OTM payoffs; no zero-payoff empirical wing extrapolation",
        "index_tenors_business_days": list(TENORS), "strike_over_mean_range": [.6, 1.8], "number_of_strikes": len(strikes),
        "in_sample_calibration": True, "risk_neutral_option_prices": False,
        "uncertainty": "No confidence bands; historical levels are serially dependent and 70D has only 83 observations",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--historic-targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-payoffs", type=int, default=10)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
