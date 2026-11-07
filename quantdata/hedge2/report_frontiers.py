"""Reproducible Greek-frontier reporting with explicit pending and infeasible outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from quantdata.hedge2.frontier_metrics import matched_frontier_metrics
from quantdata.hedge2.frontier_population import (
    external_output_directory,
    load_frontier_population,
    load_reference_population,
)

DATA_KEYS = ("source_root", "market_directory", "licensed_fit_directory", "raw_quote_directory")
ECONOMIC_SPECIFICATION = {"option_side": "itm", "include_co_terminal_forwards": True,
    "spot_cost": 0., "option_half_spread_multiplier": 1., "terminal_exit_charge": 0.}


def _display(value: float) -> str:
    return f"{value * 1e6:,.0f}" if np.isfinite(value) else "--"


def aggregate_frontier_products(metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight product-specific statistics; never drop unmeasured products.

    A configuration with even one product lacking two matched observations
    has no family aggregate risk estimate. Counts remain visible. Standard
    deviations are across origins for each product, not pooled row risk.
    """
    statistics = ("mean_pnl", "median_pnl", "std_pnl", "scaled_mad", "mean_entry_cost", "es95_loss",
                  "reference_mean_pnl", "reference_std_pnl", "reference_mean_entry_cost")
    rows = []
    for key, group in metrics.groupby(["reference", "variant", "family", "side", "greek_budget"], sort=True):
        measured = bool(group.performance_available.all())
        rows.append(dict(zip(("reference", "variant", "family", "side", "greek_budget"), key)) | {
            "products": len(group), "all_products_measured": measured,
            "minimum_paired_origins": int(group.paired_origins.min()),
            "maximum_paired_origins": int(group.paired_origins.max()),
            "requested_product_origins": int(group.requested_origins.sum()),
            "completed_product_origins": int(group.completed_origins.sum()),
            "unavailable_product_origins": int(group.unavailable_origins.sum()),
            "pending_product_origins": int(group.pending_origins.sum()),
            **{name: float(np.mean(group[name].to_numpy(dtype=float))) if measured else np.nan
               for name in statistics}})
    return pd.DataFrame(rows)


def build_frontier_report(*, configuration_path: Path, output_directory: Path) -> dict[str, object]:
    """Audit exact requested populations and compare only matched target/origin/side rows."""
    started = time.monotonic()
    configuration = json.loads(configuration_path.read_text())
    external_output_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).with_name(name) for name in (
        "report_frontiers.py", "frontier_population.py", "frontier_metrics.py", "metrics.py",
        "audit_greek_frontier.py", "episode_artifacts.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    requested_path = Path(configuration["requested_population"])
    requested = pd.read_parquet(requested_path)
    identities = configuration["expected_identities"]
    if set(identities) != set(configuration["sources"]):
        raise ValueError("Each report variant requires its explicit economic/model identity")
    common_identity = None
    for identity in identities.values():
        if (any(identity.get(name) != value for name, value in ECONOMIC_SPECIFICATION.items())
                or identity.get("optimized_premium_cap") != 10.
                or any(not isinstance(identity.get(name), str) or not identity[name]
                       for name in (*DATA_KEYS, "dynamics_directory"))):
            raise ValueError("The source identity does not support the report's economic specification")
        current = {name: identity[name] for name in (*DATA_KEYS, *ECONOMIC_SPECIFICATION)}
        if common_identity is not None and current != common_identity:
            raise ValueError("Report variants do not share the same source data and economic conventions")
        common_identity = current
    population = load_frontier_population(requested=requested,
        sources={name: tuple(Path(value) for value in paths) for name, paths in configuration["sources"].items()},
        expected_identities=identities, audit_directory=output_directory / "audits",
        progress=lambda message: print(message, flush=True))
    if len(population.pending) and not configuration.get("allow_partial", False):
        raise ValueError("The requested population is unfinished; an interim report requires explicit allow_partial")
    if population.available.empty:
        raise ValueError("No independently audited completed portfolios are available for performance")
    frames, references = [], {}
    comparison = configuration.get("comparison_sample", "common")
    for label, directory in configuration["references"].items():
        reference, evidence = load_reference_population(directory=Path(directory))
        if any(evidence["configuration"].get(name) != value for name, value in common_identity.items()):
            raise ValueError("The analytic reference uses another dataset, hedge book or economic convention")
        metrics = matched_frontier_metrics(requested=population.requested, available=population.available,
            unavailable=population.unavailable, pending=population.pending, reference=reference,
            comparison_sample=comparison, allow_reference_gaps=True)
        frames.append(metrics.assign(reference=label))
        references[label] = evidence
        print(f"Frontier report reference={label} products={len(metrics)} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    if not frames:
        raise ValueError("The frontier report requires at least one explicit analytic reference")
    products = pd.concat(frames, ignore_index=True)
    summary = aggregate_frontier_products(products)
    population.requested.to_parquet(output_directory / "requested.parquet", index=False)
    population.available.to_parquet(output_directory / "completed.parquet", index=False)
    population.unavailable.to_parquet(output_directory / "unavailable.parquet", index=False)
    population.pending.to_parquet(output_directory / "pending.parquet", index=False)
    population.episodes.to_parquet(output_directory / "episode_coverage.parquet", index=False)
    products.to_parquet(output_directory / "per_product.parquet", index=False)
    summary.to_parquet(output_directory / "summary.parquet", index=False)
    coverage = population.episodes.groupby(["variant", "family"], sort=True)[[
        "requested_portfolios", "completed_portfolios", "unavailable_portfolios", "pending_portfolios"]].sum().reset_index()
    coverage.to_parquet(output_directory / "coverage.parquet", index=False)
    pdf_path = output_directory / "GREEK_FRONTIERS.pdf"
    plt.rcParams.update({"font.size": 11, "text.parse_math": False})
    with PdfPages(pdf_path) as pdf:
        figure = plt.figure(figsize=(15, 9))
        figure.suptitle("Hedge2 — multi-day Greek cost frontiers", fontsize=20, y=.96)
        lines = [
            "INTERIM COMPUTATIONAL SNAPSHOT — not a completed rolling benchmark." if len(population.pending)
            else "All requested Greek portfolio computations are reconciled; economic infeasibility remains explicit.",
            f"Requested {len(requested):,}; completed {len(population.available):,}; economically unavailable {len(population.unavailable):,}; pending {len(population.pending):,}.",
            "Saved SANOS prices and direct DLV states; all recorded data, model-directory and economic identities are checked.",
            "Daily spot and frozen ITM European contracts, including co-terminal prepaid forwards. Original cash strikes and episode S0 retained.",
            "$10 current-premium net position box per instrument. Spot cost zero; full observed option half-spreads on actual rebalancing.",
            "No charged expiry/KO exit fee. Actual realized P&L is never demeaned; future realized market data never enter forecast fitting.",
            f"Greek budgets {', '.join(f'{100 * value:g}%' for value in sorted(requested.greek_budget.unique(), reverse=True))} re-solve the daily exposure problem; they do not scale existing positions.",
            "Unavailable means independently proven exact-match box or mandatory-turnover-budget infeasibility, not solver nonconvergence.",
            "Unavailable chains retain executed-prefix evidence but have no fabricated terminal result. Pending computation is a separate count.",
            f"Comparison sample: {comparison}; each reference is paired separately on exactly matching origin, target, claim side and inception price.",
            "Common sampling requires every requested variant/budget plus that reference. Reference gaps and all per-product counts are saved.",
            "Tables average product-specific across-origin sample statistics. A missing product statistic suppresses the family aggregate.",
            "Amounts are USD per $1m digital payout, initial-spot KO notional, or ratio-straddle multiplier, respectively; these are distinct units.",
            "These are daily local finite-shock Greek hedges, not terminal-horizon statistical policies. Atomic digital exposure limitations remain.",
            "No OOS configuration selection, pooled-strike independence, confidence interval or global neural-optimality claim is made here."]
        for index, line in enumerate(lines):
            figure.text(.045, .89 - .052 * index, line, fontsize=10.5, va="top")
        pdf.savefig(figure)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(15, 9))
        axis.axis("off")
        axis.set_title("Full requested population — computation and economic availability", fontsize=18, pad=24)
        table = axis.table(cellText=coverage.values.tolist(), colLabels=["Variant", "Family", "Requested", "Completed", "Unavailable", "Pending"],
                           loc="center", cellLoc="right")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(list(range(6)))
        table.scale(1., 2.)
        figure.text(.05, .10, "Counts are portfolio chains, not independent observations. Pending does not imply economic infeasibility or prove a worker is live.", fontsize=11)
        pdf.savefig(figure)
        plt.close(figure)
        for key, group in summary.groupby(["reference", "family", "side"], sort=True):
            label, family, side = key
            figure, axis = plt.subplots(figsize=(15, 9))
            axis.axis("off")
            axis.set_title(f"{family} — {'short' if side == -1 else 'long'} — matched to {label}", fontsize=18, pad=24)
            rows = [[row.variant, f"{100 * row.greek_budget:.0f}%", f"{row.minimum_paired_origins}–{row.maximum_paired_origins}",
                _display(row.std_pnl), _display(row.reference_std_pnl), _display(row.mean_entry_cost),
                _display(row.mean_pnl), _display(row.es95_loss)] for row in group.itertuples()]
            table = axis.table(cellText=rows, colLabels=["PCA", "Budget", "Paired origins", "Std net", "Reference std", "Entry cost", "Mean net", "ES95 loss"],
                               loc="center", cellLoc="right")
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.auto_set_column_width(list(range(8)))
            table.scale(1., 1.9)
            figure.text(.045, .10, "USD per $1m in the stated family units; costed realized P&L. Origin counts show the minimum/maximum across products.\n"
                "Every target and setting remains in per_product.parquet, including empty or single-observation samples. '--' does not mean zero risk.\n"
                "Comparisons use only completed, independently checked chains; coverage exclusions and pending work remain explicit.", fontsize=10.5)
            pdf.savefig(figure)
            plt.close(figure)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise RuntimeError("Reporting source changed during the audited snapshot")
    provenance = {"pdf": str(pdf_path), "partial_benchmark": True,
        "pending_computation": len(population.pending), "requested_portfolios": len(requested),
        "completed_portfolios": len(population.available), "unavailable_portfolios": len(population.unavailable),
        "comparison_sample": comparison, "configuration_selection": "none", "source_sha256": hashes,
        "configuration_sha256": hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
        "requested_population_sha256": hashlib.sha256(requested_path.read_bytes()).hexdigest(),
        "frontier_evidence": population.evidence, "reference_evidence": references,
        "elapsed_seconds": time.monotonic() - started}
    (output_directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    result = build_frontier_report(**vars(parser.parse_args()))
    print(json.dumps({name: value for name, value in result.items() if name != "frontier_evidence" and name != "reference_evidence"}), flush=True)


if __name__ == "__main__":
    main()
