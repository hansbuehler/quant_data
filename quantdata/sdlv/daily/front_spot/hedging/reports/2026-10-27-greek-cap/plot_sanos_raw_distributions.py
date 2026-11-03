"""Compare raw one-day hedge P&L with fitted laws and drift-aware frontiers."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from numpy.typing import NDArray

DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data")) / "experiments/hedging"
SCENARIOS = DATA_ROOT / "sanos-objective-grid-20261027-v1/scenarios"
REFERENCE = Path(__file__).with_name("PROVENANCE.json")
OBJECTIVES = ("mean-variance", "entropy", "power-exponential")
RISKS = ("1", "10", "100", "1000")
MODELS = ("historical", "parametric")
BASELINES = ("unhedged", "bs-delta-vega", "sanos-greek")
PRODUCTS = (
    "digital-put-1d-nms-1.00", "digital-put-1d-nms-0.75",
    "digital-put-1d-nms-0.50", "digital-put-1d-nms-0.25",
    "digital-put-1d-atm", "digital-call-1d-atm",
    "digital-call-1d-nms+0.25", "digital-call-1d-nms+0.50",
    "digital-call-1d-nms+0.75", "digital-call-1d-nms+1.00",
)
TITLES = {
    "historical": "SANOS historical", "parametric": "SANOS parametric",
    "unhedged": "Unhedged", "bs-delta-vega": "BS delta + vega",
    "sanos-greek": "SANOS Greek",
}
OBJECTIVE_TITLES = {
    "mean-variance": "Mean-variance", "entropy": "Entropy",
    "power-exponential": "Power-exponential",
}
COLORS = {
    "historical": "#0072B2", "parametric": "#D55E00",
    "unhedged": "#555555", "bs-delta-vega": "#8C564B",
    "sanos-greek": "#2E7D32",
}
EDGES = np.linspace(-1200.0, 1200.0, 49)
TAIL_PROBABILITY = 0.05


@dataclass(frozen=True)
class ScenarioBook:
    origin_date: str
    instrument_ids: tuple[str, ...]
    target_labels: tuple[str, ...]
    raw_instruments: NDArray[np.float64]
    raw_targets: NDArray[np.float64]
    instrument_means: NDArray[np.float64]
    target_means: NDArray[np.float64]


@dataclass(frozen=True)
class HedgeLaw:
    raw: NDArray[np.float64]
    corrected: NDArray[np.float64]
    cost: NDArray[np.float64]
    fitted_returns: NDArray[np.float64] | None
    fitted_means: NDArray[np.float64] | None
    fitted_standard_deviations: NDArray[np.float64] | None


def load_books() -> dict[str, ScenarioBook]:
    books = {}
    for path in sorted(SCENARIOS.glob("2024-*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"].item()))
            instruments = np.asarray(saved["instrument_returns"], dtype=np.float64)
            targets = np.asarray(saved["target_returns"], dtype=np.float64)
        if (
            identity["origin_date"] != path.stem
            or identity["training_end"] != path.stem
            or instruments.shape != (252, len(identity["instrument_ids"]))
            or targets.shape != (252, 10)
            or set(identity["target_labels"]) != set(PRODUCTS)
            or not np.isfinite(instruments).all()
            or not np.isfinite(targets).all()
        ):
            raise RuntimeError(f"Invalid preceding-252-day scenario book: {path}")
        books[path.stem] = ScenarioBook(
            path.stem, tuple(identity["instrument_ids"]), tuple(identity["target_labels"]),
            instruments, targets, instruments.mean(axis=0), targets.mean(axis=0),
        )
    if len(books) != 44:
        raise RuntimeError(f"Expected 44 April-May dates, found {len(books)}")
    return books


def record_path(*, model: str, objective: str, risk: str, origin_date: str) -> Path:
    if model == "historical":
        return (DATA_ROOT / "sanos-objective-grid-20261027-v1/historical-sanos"
                / f"{objective}-lambda{risk}.0/records/{origin_date}.json")
    if model == "parametric":
        return (DATA_ROOT / "sanos-parametric-objectives-20261027-v1"
                / objective / f"lambda{risk}/records/{origin_date}.json")
    if model == "sanos-greek":
        return DATA_ROOT / f"sanos-full-aprmay-20261026-v1/results/{origin_date}.json"
    return (DATA_ROOT / "entry-only-replay-20261025-v1"
            / f"bs-greeks-mean-variance-lambda10/records/{origin_date}.json")


def read_rows(*, model: str, objective: str, risk: str,
              book: ScenarioBook) -> list[dict[str, object]]:
    path = record_path(model=model, objective=objective, risk=risk, origin_date=book.origin_date)
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload["results"]
    label = {
        "historical": "historical-sanos-objective-grid",
        "parametric": "sanos-parametric-ordinary",
        "sanos-greek": "sanos-greek-ordinary",
        "unhedged": "bs-delta-vega",
        "bs-delta-vega": "bs-delta-vega",
    }[model]
    matching = {row["claim"]: row for row in rows if row["model"] == label}
    if len(matching) != 10 or set(matching) != set(book.target_labels):
        raise RuntimeError(f"Unmatched ten-target record: {path}")
    ordered = [matching[target] for target in book.target_labels]
    if any(
        row["origin_date"] != book.origin_date
        or not row["converged"] or row["cap_hits"]
        for row in ordered
    ):
        raise RuntimeError(f"Uncertified record: {path}")
    return ordered


def collect_law(books: dict[str, ScenarioBook], *, model: str,
                objective: str = "mean-variance", risk: str = "1") -> HedgeLaw:
    raw = np.empty((44, 10), dtype=np.float64)
    corrected = np.empty_like(raw)
    cost = np.empty_like(raw)
    fitted_returns = []
    fitted_means = []
    fitted_deviations = []
    for date_index, book in enumerate(books.values()):
        rows = read_rows(model=model, objective=objective, risk=risk, book=book)
        quantities = np.asarray([row["claim_quantity"] for row in rows], dtype=np.float64)
        entry_costs = np.asarray([row["transaction_cost"] for row in rows], dtype=np.float64)
        if np.any(quantities <= 0.0):
            raise RuntimeError(f"Invalid target quantity: {book.origin_date}")
        scale = 1000.0 / quantities
        if model == "unhedged":
            actions = np.zeros((len(book.instrument_ids), 10), dtype=np.float64)
            entry_costs = np.zeros(10, dtype=np.float64)
            pnl = np.asarray([row["unhedged_pnl"] for row in rows], dtype=np.float64)
        else:
            if any(set(row["actions"]) - set(book.instrument_ids) for row in rows):
                raise RuntimeError(f"Action outside matched book: {book.origin_date}")
            actions = np.asarray(
                [[row["actions"].get(instrument_id, 0.0) for row in rows]
                 for instrument_id in book.instrument_ids],
                dtype=np.float64,
            )
            pnl = np.asarray([row["hedged_pnl"] for row in rows], dtype=np.float64)
        raw[date_index] = scale * pnl
        corrected[date_index] = raw[date_index] - scale * (
            quantities * book.target_means + book.instrument_means @ actions
        )
        cost[date_index] = scale * entry_costs
        if model in (*BASELINES, "historical"):
            instrument_returns = book.raw_instruments
            target_returns = book.raw_targets
            if model == "historical":
                instrument_returns = instrument_returns - book.instrument_means
                target_returns = target_returns - book.target_means
            fitted_returns.append(
                (target_returns * quantities + instrument_returns @ actions - entry_costs)
                * scale
            )
        else:
            fitted_means.extend((-cost[date_index]).tolist())
            fitted_deviations.extend(
                scale[target_index] * math.sqrt(max(row["diagnostics"]["risk_variance"], 0.0))
                for target_index, row in enumerate(rows)
            )
    fitted = np.stack(fitted_returns) if fitted_returns else None
    means = np.asarray(fitted_means, dtype=np.float64) if fitted_means else None
    deviations = np.asarray(fitted_deviations, dtype=np.float64) if fitted_deviations else None
    if (
        not np.isfinite(raw).all() or not np.isfinite(corrected).all()
        or not np.isfinite(cost).all()
        or (fitted is not None and fitted.shape != (44, 252, 10))
        or (deviations is not None and (deviations.shape != (440,) or np.any(deviations <= 0.0)))
    ):
        raise RuntimeError(f"Invalid fitted law: {model}, {objective}, {risk}")
    return HedgeLaw(raw, corrected, cost, fitted, means, deviations)


def validate_law(law: HedgeLaw, *, model: str, objective: str, risk: str,
                 reference: dict[str, object]) -> None:
    measured = np.asarray([np.mean(np.std(law.raw, axis=0, ddof=1)), np.mean(law.cost)])
    if model in BASELINES:
        expected = reference["baselines"][TITLES[model]]
        tolerance = 1.0e-8
    else:
        expected = reference["points"][objective][TITLES[model]][risk]
        tolerance = 0.0051
    if not np.allclose(measured, expected, rtol=0.0, atol=tolerance):
        raise RuntimeError(f"Disagrees with retained out-of-sample cohort: {model}, {objective}, {risk}")


def density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    counts, _ = np.histogram(values.ravel(), bins=EDGES)
    return counts / (values.size * float(EDGES[1] - EDGES[0]))


def draw_distribution(axis: plt.Axes, law: HedgeLaw, *, target_index: int) -> None:
    centers = (EDGES[:-1] + EDGES[1:]) / 2.0
    raw_density = density(law.raw[:, target_index])
    axis.plot(centers, np.where(raw_density > 0, raw_density, np.nan),
              drawstyle="steps-mid", color="#0072B2", linewidth=1.8)
    if law.fitted_returns is not None:
        fitted_density = density(law.fitted_returns[:, :, target_index])
        axis.plot(centers, np.where(fitted_density > 0, fitted_density, np.nan),
                  drawstyle="steps-mid", color="#D55E00", linewidth=2.0)
    else:
        if law.fitted_means is None or law.fitted_standard_deviations is None:
            raise RuntimeError("Missing fitted Gaussian law")
        locations = np.linspace(EDGES[0], EDGES[-1], 1201)
        means = law.fitted_means.reshape(44, 10)[:, target_index]
        deviations = law.fitted_standard_deviations.reshape(44, 10)[:, target_index]
        scores = (locations[:, None] - means) / deviations
        fitted_density = np.mean(
            np.exp(-0.5 * scores**2) / (math.sqrt(2.0 * math.pi) * deviations), axis=1
        )
        axis.plot(locations, np.where(fitted_density > 0, fitted_density, np.nan),
                  color="#D55E00", linewidth=2.0)
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.set_xlim(EDGES[0], EDGES[-1])
    axis.set_yscale("log")
    axis.set_ylim(1.0e-6, 3.0e-2)
    axis.grid(color="#D9DEE3", linewidth=0.7, alpha=0.75)
    axis.set_xlabel("Costed P&L · USD thousands per $1m digital")
    axis.set_ylabel("Density · log scale")


def add_legend(figure: plt.Figure, *, fitted_label: str) -> None:
    figure.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", linewidth=2, label="Realized, raw"),
            Line2D([0], [0], color="#D55E00", linewidth=2, label=fitted_label),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.068), frameon=False, fontsize=10,
    )


def plot_distributions(pdf: PdfPages, output: Path,
                       laws: dict[str, dict[str, dict[str, HedgeLaw]]],
                       baselines: dict[str, HedgeLaw]) -> None:
    for target_index, product in enumerate(PRODUCTS):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        for axis, model in zip(axes, BASELINES, strict=True):
            draw_distribution(axis, baselines[model], target_index=target_index)
            axis.set_title(TITLES[model], fontsize=13, weight="bold")
        figure.suptitle(f"{product}: fixed-reference hedge distributions", fontsize=16, y=0.97)
        add_legend(figure, fitted_label="Trailing 252-day SANOS replay, raw")
        figure.text(0.5, 0.012,
                    "No mean correction: 44 realized next-day outcomes versus preceding 252 raw returns. "
                    "References have no fitted hedge-return law.\n"
                    "Density peaks above 0.03 and outcomes beyond ±$1.2m are clipped in panels only.",
                    ha="center", fontsize=8.5, color="#555555")
        figure.subplots_adjust(left=0.06, right=0.975, bottom=0.24, top=0.84, wspace=0.30)
        if product == "digital-call-1d-atm":
            figure.savefig(output / "BASELINES_DIGITAL_CALL_ATM.png", dpi=160)
        pdf.savefig(figure)
        plt.close(figure)
        for objective in OBJECTIVES:
            for model in MODELS:
                figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
                for axis, risk in zip(axes.flat, RISKS, strict=True):
                    draw_distribution(axis, laws[model][objective][risk], target_index=target_index)
                    axis.set_title(f"λ={risk}", fontsize=13, weight="bold")
                figure.suptitle(
                    f"{product}: {TITLES[model]} · {OBJECTIVE_TITLES[objective]}",
                    fontsize=16, y=0.97,
                )
                fitted_label = (
                    "Fitted 252-day demeaned SANOS returns" if model == "historical"
                    else "Fitted centered factor Gaussian"
                )
                add_legend(figure, fitted_label=fitted_label)
                figure.text(0.5, 0.015,
                            "Realized P&L is not mean-adjusted; each fitted law precedes its next-day "
                            "outcome across 44 dates.\n"
                            "Density peaks above 0.03 and outcomes beyond ±$1.2m are clipped in panels only.",
                            ha="center", fontsize=8.5, color="#555555")
                figure.subplots_adjust(left=0.075, right=0.97, bottom=0.18, top=0.90,
                                       wspace=0.24, hspace=0.38)
                if product == "digital-call-1d-atm" and objective == "entropy":
                    figure.savefig(output / f"{model.upper()}_ENTROPY_DIGITAL_CALL_ATM.png",
                                   dpi=160)
                pdf.savefig(figure)
                plt.close(figure)


def risk_metrics(law: HedgeLaw) -> dict[str, float]:
    entry_cost = float(np.mean(law.cost))
    corrected_pre_cost_loss = float(-np.mean(law.corrected + law.cost))
    sd = float(np.mean(np.std(law.raw, axis=0, ddof=1)))
    centered = law.raw - np.mean(law.raw, axis=0)
    tail_mass = TAIL_PROBABILITY * len(law.raw)
    full_observations = int(tail_mass)
    sorted_centered = np.sort(centered, axis=0)
    tail_returns = np.sum(sorted_centered[:full_observations], axis=0)
    tail_returns += (tail_mass - full_observations) * sorted_centered[full_observations]
    tail = float(-np.mean(tail_returns / tail_mass))
    if sd <= 0.0 or tail <= 0.0:
        raise RuntimeError("A realized risk denominator is not positive")
    return {
        "mean_entry_cost": entry_cost,
        "mean_training_corrected_pre_cost_loss": corrected_pre_cost_loss,
        "cost_plus_corrected_mean_loss": entry_cost + corrected_pre_cost_loss,
        "raw_realized_sd": sd,
        "raw_realized_centered_es05": tail,
    }


def plot_frontiers(pdf: PdfPages, output: Path,
                   laws: dict[str, dict[str, dict[str, HedgeLaw]]],
                   baselines: dict[str, HedgeLaw]) -> dict[str, object]:
    baseline_metrics = {model: risk_metrics(law) for model, law in baselines.items()}
    objective_metrics = {
        model: {
            objective: {risk: risk_metrics(laws[model][objective][risk]) for risk in RISKS}
            for objective in OBJECTIVES
        }
        for model in MODELS
    }
    frontiers = (
        ("raw_realized_sd", "mean_entry_cost", "Entry cost versus realized SD", "COST_VERSUS_SD"),
        ("raw_realized_sd", "cost_plus_corrected_mean_loss",
         "Entry cost + corrected mean loss versus realized SD", "COST_PLUS_CORRECTED_MEAN_VERSUS_SD"),
        ("raw_realized_centered_es05", "mean_entry_cost",
         "Entry cost versus realized centered 5% ES", "COST_VERSUS_TAIL"),
        ("raw_realized_centered_es05", "cost_plus_corrected_mean_loss",
         "Entry cost + corrected mean loss versus realized centered 5% ES",
         "COST_PLUS_CORRECTED_MEAN_VERSUS_TAIL"),
    )
    for risk_metric, cost_metric, title, filename in frontiers:
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        for axis, objective in zip(axes, OBJECTIVES, strict=True):
            for model in MODELS:
                model_metrics = objective_metrics[model][objective]
                axis.plot(
                    [model_metrics[risk][risk_metric] for risk in RISKS],
                    [model_metrics[risk][cost_metric] for risk in RISKS],
                    color=COLORS[model], linewidth=2.0, marker="o", markersize=5,
                )
                for risk in (RISKS[0], RISKS[-1]):
                    axis.annotate(
                        f"λ={risk}",
                        (model_metrics[risk][risk_metric], model_metrics[risk][cost_metric]),
                        xytext=(4, 6 if model == "historical" else -12),
                        textcoords="offset points", color=COLORS[model], fontsize=7.5,
                    )
            for model in BASELINES:
                axis.scatter(baseline_metrics[model][risk_metric],
                             baseline_metrics[model][cost_metric], color=COLORS[model],
                             marker="D", s=47, zorder=4)
            axis.set_title(OBJECTIVE_TITLES[objective], fontsize=13, weight="bold")
            axis.set_xlabel("Realized P&L SD" if risk_metric == "raw_realized_sd"
                            else "Realized centered 5% ES")
            axis.set_ylabel("Entry cost" if cost_metric == "mean_entry_cost"
                            else "Entry cost + corrected mean loss")
            axis.margins(x=0.10, y=0.20)
            axis.grid(color="#D9DEE3", linewidth=0.7, alpha=0.75)
        figure.suptitle(title, fontsize=16, y=0.97)
        figure.legend(
            handles=[Line2D([0], [0], color=COLORS[model], linewidth=2.0,
                            marker="o" if model in MODELS else "D", label=TITLES[model])
                     for model in (*MODELS, *BASELINES)],
            loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.10), frameon=False, fontsize=9,
        )
        figure.text(0.5, 0.02,
                    "Corrected mean subtracts preceding 252-day target/instrument sample means; "
                    "risk axes use raw next-day P&L.\n"
                    "Centered 5% ES uses each target's worst two dates plus 20% of its third-worst "
                    "after removing its realized mean. Lines join λ=1, 10, 100, 1000.",
                    ha="center", fontsize=8.5, color="#555555")
        figure.subplots_adjust(left=0.065, right=0.975, bottom=0.26, top=0.82, wspace=0.25)
        figure.savefig(output / f"{filename}.png", dpi=160)
        pdf.savefig(figure)
        plt.close(figure)
    return {"baselines": baseline_metrics, "objectives": objective_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Unique run directory outside the Git working tree")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or not output.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"Output must be a new run directory under {DATA_ROOT}: {output}")
    books = load_books()
    reference = json.loads(REFERENCE.read_text())
    baselines = {model: collect_law(books, model=model) for model in BASELINES}
    laws = {
        model: {
            objective: {
                risk: collect_law(books, model=model, objective=objective, risk=risk)
                for risk in RISKS
            }
            for objective in OBJECTIVES
        }
        for model in MODELS
    }
    for model, law in baselines.items():
        validate_law(law, model=model, objective="mean-variance", risk="1", reference=reference)
    for model in MODELS:
        for objective in OBJECTIVES:
            for risk in RISKS:
                validate_law(laws[model][objective][risk], model=model,
                             objective=objective, risk=risk, reference=reference)
    output.mkdir(parents=True)
    with PdfPages(output / "SANOS_RAW_DISTRIBUTIONS_AND_FRONTIERS.pdf") as pdf:
        plot_distributions(pdf, output, laws, baselines)
        frontier_metrics = plot_frontiers(pdf, output, laws, baselines)
    (output / "SUMMARY.json").write_text(json.dumps({
        "schema": "sanos-raw-distributions-v2",
        "cohort": "44 April-May 2024 origins, ten one-day digitals",
        "units": "USD thousands per USD 1 million digital payout",
        "products": PRODUCTS,
        "risk_parameters": RISKS,
        "estimated_distributions": {
            "unhedged": "252 preceding raw historical SANOS returns per origin",
            "bs-delta-vega": "252 preceding raw historical SANOS returns per origin",
            "sanos-greek": "252 preceding raw historical SANOS returns per origin",
            "historical": "252 preceding demeaned historical SANOS returns per origin",
            "parametric": "centered first-order SANOS factor Gaussian mixture",
        },
        "realized_distributions": "raw market P&L, with no mean adjustment",
        "corrected_mean": "frozen training means used only on corrected frontier axes",
        "tail_risk": "target-averaged centered 5% ES: worst two of 44 dates plus 20% of third worst, divided by 2.2",
        "frontier_metrics": frontier_metrics,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
