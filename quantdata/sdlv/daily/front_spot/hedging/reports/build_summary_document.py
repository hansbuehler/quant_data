"""Render the final public-data hedging benchmark from authenticated metrics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import TypeAlias

from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    SUMMARY_SCHEMA as DAILY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    SUMMARY_SCHEMA as WEEKLY_SUMMARY_SCHEMA,
)

JsonObject: TypeAlias = dict[str, object]

DOCUMENT_SCHEMA = "hedging-benchmark-summary-document-v1"
DISPLAY_SCALE = 10_000.0


def build_summary_document(
    *,
    daily: JsonObject,
    weekly: JsonObject,
    status_date: date,
    daily_objective_tables: str,
    weekly_objective_tables: str,
    daily_target_appendix: str,
    weekly_target_appendix: str,
) -> str:
    """Return a standalone LaTeX benchmark document from authenticated inputs."""
    _validate_inputs(daily=daily, weekly=weekly)
    daily_models = _require_object(daily, "models")
    weekly_models = _require_object(weekly, "models")
    daily_best = _best_daily_by_model(daily_models)
    weekly_best = _best_weekly_by_schedule_and_model(weekly_models)
    daily_reference = _daily_reference(daily_models)
    daily_unhedged = _daily_metric(daily_reference, "unhedged_rmse")
    daily_best_reduction = max(
        _daily_metric(model, "pure_rmse_reduction") for model in daily_best
    )
    weekly_schedules = sorted(
        {_require_string(model, "schedule") for model in weekly_best}
    )
    model_names = sorted(
        {
            _require_string(model, "model")
            for model in (*daily_best, *weekly_best)
        }
    )
    manifest = _require_string(daily, "dataset_manifest_sha256")
    return rf"""\documentclass[10pt]{{article}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage[a4paper,margin=17mm,headheight=14pt]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{pdflscape}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage[numbers,sort&compress]{{natbib}}

\definecolor{{navy}}{{HTML}}{{15324A}}
\definecolor{{blue}}{{HTML}}{{2F6F9F}}
\definecolor{{pale}}{{HTML}}{{EAF2F8}}
\definecolor{{grey}}{{HTML}}{{5D6973}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=blue,
  pdftitle={{Public-Data Hedging Benchmark Summary}},pdfauthor={{Quantitative Research}}}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{Public-Data Hedging Benchmark}}
\rhead{{{_tex_escape(status_date.strftime('%d %B %Y'))}}}
\cfoot{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\titleformat{{\section}}{{\large\bfseries\color{{navy}}}}{{\thesection}}{{0.6em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries\color{{blue}}}}{{\thesubsection}}{{0.5em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{1.1em}}{{0.45em}}
\titlespacing*{{\subsection}}{{0pt}}{{0.8em}}{{0.3em}}
\setlist[itemize]{{leftmargin=1.4em,itemsep=0.15em,topsep=0.25em}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.35em}}
\renewcommand{{\arraystretch}}{{1.16}}
\raggedbottom

\begin{{document}}
\begin{{center}}
  {{\LARGE\bfseries\color{{navy}} Public-Data Hedging Benchmark}}\par
  \vspace{{0.2em}}
  {{\large Approach, authenticated results, and model catalogue}}\par
  \vspace{{0.35em}}
  {{\color{{grey}}Status date: {_tex_escape(status_date.strftime('%d %B %Y'))}}}
\end{{center}}

\vspace{{0.4em}}
\noindent\colorbox{{pale}}{{\parbox{{\dimexpr\textwidth-2\fboxsep\relax}}{{%
\textbf{{Executive summary.}}
The distributable-candidate benchmark contains 891 dates and 2,014,740
eligible option coordinates without original WRDS prices or identifiers.
The final daily comparison contains {_integer(sum(_result_count(model) for model in daily_models.values()))}
authenticated claim/origin evaluations across {len(daily_models)} policies;
the weekly comparison contains {_integer(sum(_result_count(model) for model in weekly_models.values()))}
evaluations across {len(weekly_models)} policies and {len(weekly_schedules)} schedules.
The common daily unhedged RMSE is {_basis_points(daily_unhedged)} basis points;
the largest descriptive pure-RMSE reduction among the reported daily policies
is {_percent(daily_best_reduction)}. Headline rows below select the lowest
pure-RMSE policy within each model family only for compact presentation; every
precommitted objective and parameter remains reported in the full tables.}}}}

\section{{Approach}}
\subsection{{Public data boundary and common market}}
The sole runtime input is the normalized, original-data-free
\texttt{{spx-2022-2025-hedging-public-candidate-v1}} panel. It releases fitted
SANOS and DLV states, reconstructable SANOS and DLV prices, Black implied
volatilities, forward/spot ratios, discount factors, normalized half-spreads,
ATM volatilities, and forward-adjusted spot returns. It does not release
original WRDS prices, midpoints, identifiers, volumes, open interest, vendor
Greeks, or vendor implied volatilities. The executable market mark is the
released SANOS price; buys pay one half-spread, sells receive one half-spread,
spot costs one basis point of traded notional, and cash settlement is free.

\subsection{{Out-of-sample protocol}}
\begin{{itemize}}
  \item Architecture validation and target sizing use 2022 only. At date $t$,
  rolling estimators use the current state and all 252 exact one-session
  transitions immediately preceding $t$, with no future observation.
  \item Daily evaluation covers 611 origins and twenty 1D/3D digital targets,
  producing 12,220 claim/origin observations per policy.
  \item Weekly evaluation starts a five-session episode at each of 528
  overlapping origins. Frozen-weekly policies retain the inception model and
  contract book; restarted-weekly policies refit daily.
  \item Hedge instruments are normalized spot plus all eligible OTM options
  strictly inside $(-2,2)$ contemporaneous NMS with expiries through five
  business days, or the first later expiry if five days is absent.
  \item Every method starts at the same market claim price. Pricing loss, pure
  hedge loss, and total economic loss are reported separately. Target sizes are
  fixed from 2022 at one maturity-matched ATM spot-risk unit.
\end{{itemize}}

\subsection{{Objectives and inference}}
The benchmark reports mean--variance, mean--volatility, entropy OCE, and
power-exponential OCE criteria \citep{{BuehlerEtAl2019DeepHedging}}. Daily
uncertainty uses complete daily episodes. Weekly comparisons first form the
equal-weighted thirteen-target episode book, then report paired
candidate-minus-SANOS differences with Hansen--Hodrick and moving-block
bootstrap intervals, preserving five-session overlap.

\section{{Results}}
\subsection{{Daily headline}}
Table~\ref{{tab:daily-headline}} reports the lowest pure-RMSE policy within each
model family. This is a descriptive compression of the completed grid, not a
new tuning rule. Losses, costs, and drawdowns are basis points of inception
normalized spot.

{_daily_headline_table(daily_best)}

\subsection{{Weekly headline}}
Table~\ref{{tab:weekly-headline}} applies the same descriptive compression
separately to frozen and restarted schedules. The final column is the mean
paired economic-book loss difference against the schedule-matched SANOS--LV
reference; negative values favour the candidate.

{_weekly_headline_table(weekly_best, weekly)}

\subsection{{All objective grids}}
The following tables retain every completed policy and evaluate every
precommitted ex-post risk parameter on inception-spot-normalized economic loss.
\input{{{_tex_escape(daily_objective_tables)}}}
\input{{{_tex_escape(weekly_objective_tables)}}}

\section{{Model Catalogue}}
{_model_catalogue(model_names)}

\section{{Provenance}}
The public dataset manifest is \texttt{{{manifest}}}. The daily target-notional
identity is \texttt{{{_tex_escape(_require_string(daily, 'target_notional_identity'))}}};
the weekly identity is \texttt{{{_tex_escape(_require_string(weekly, 'target_notional_identity'))}}}.
Every number is rebuilt from immutable run specifications and authenticated
dated records. Results with incomplete populations, mismatched model/schedule
identities, changed source bytes, different target quantities, cap hits where
forbidden, or nonconverged fits where forbidden are rejected before rendering.

\clearpage
\appendix
\section{{Daily Target-Level Objectives}}
\input{{{_tex_escape(daily_target_appendix)}}}
\clearpage
\section{{Weekly Target-Level Results}}
\input{{{_tex_escape(weekly_target_appendix)}}}

\clearpage
\bibliographystyle{{plainnat}}
\bibliography{{hedging_benchmark_summary_refs}}
\end{{document}}
"""


def write_summary_document(
    *,
    daily_metrics: Path,
    weekly_metrics: Path,
    output: Path,
    bibliography_source: Path,
    daily_objective_tables: Path,
    weekly_objective_tables: Path,
    daily_target_appendix: Path,
    weekly_target_appendix: Path,
    status_date: date | None = None,
    compile_pdf: bool = False,
) -> Path | None:
    """Write the final TeX document and optionally compile its PDF."""
    daily = _read_json_object(daily_metrics)
    weekly = _read_json_object(weekly_metrics)
    required_inputs = (
        bibliography_source,
        daily_objective_tables,
        weekly_objective_tables,
        daily_target_appendix,
        weekly_target_appendix,
    )
    missing = [str(path) for path in required_inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"summary document inputs are missing: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    local_inputs = {}
    for source in required_inputs:
        destination = output.parent / Path(source).name
        if Path(source).resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        local_inputs[Path(source)] = destination.name
    output.write_text(
        build_summary_document(
            daily=daily,
            weekly=weekly,
            status_date=date.today() if status_date is None else status_date,
            daily_objective_tables=local_inputs[daily_objective_tables],
            weekly_objective_tables=local_inputs[weekly_objective_tables],
            daily_target_appendix=local_inputs[daily_target_appendix],
            weekly_target_appendix=local_inputs[weekly_target_appendix],
        ),
        encoding="utf-8",
    )
    if not compile_pdf:
        return None
    subprocess.run(
        (
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            output.name,
        ),
        cwd=output.parent,
        check=True,
    )
    pdf = output.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError("latexmk completed without producing the summary PDF")
    return pdf


def _validate_inputs(*, daily: JsonObject, weekly: JsonObject) -> None:
    if daily.get("schema") != DAILY_SUMMARY_SCHEMA:
        raise ValueError("daily metrics use an incompatible schema")
    if weekly.get("schema") != WEEKLY_SUMMARY_SCHEMA:
        raise ValueError("weekly metrics use an incompatible schema")
    if _require_string(daily, "dataset_manifest_sha256") != _require_string(
        weekly,
        "dataset_manifest_sha256",
    ):
        raise ValueError("daily and weekly metrics use different public datasets")
    if not _require_object(daily, "models") or not _require_object(weekly, "models"):
        raise ValueError("summary document requires daily and weekly policies")


def _best_daily_by_model(models: JsonObject) -> tuple[JsonObject, ...]:
    grouped: dict[str, list[JsonObject]] = {}
    for key, value in models.items():
        model = _as_object(value, key)
        grouped.setdefault(_require_string(model, "model"), []).append(model)
    return tuple(
        min(values, key=lambda model: _daily_metric(model, "pure_rmse"))
        for _, values in sorted(grouped.items())
    )


def _best_weekly_by_schedule_and_model(models: JsonObject) -> tuple[JsonObject, ...]:
    grouped: dict[tuple[str, str], list[JsonObject]] = {}
    for key, value in models.items():
        model = _as_object(value, key)
        group = (_require_string(model, "schedule"), _require_string(model, "model"))
        grouped.setdefault(group, []).append(model)
    return tuple(
        min(values, key=_weekly_pure_rmse)
        for _, values in sorted(grouped.items())
    )


def _daily_reference(models: JsonObject) -> JsonObject:
    references = tuple(
        _as_object(value, key)
        for key, value in models.items()
        if _require_string(_as_object(value, key), "model") == "sanos-lv"
    )
    if len(references) != 1:
        raise ValueError("daily report requires one SANOS-LV reference")
    return references[0]


def _daily_headline_table(models: tuple[JsonObject, ...]) -> str:
    rows = []
    for model in models:
        rows.append(
            " & ".join(
                (
                    _require_string(model, "display_name"),
                    _fit_label(model),
                    _basis_points(_daily_metric(model, "pure_rmse")),
                    _percent(_daily_metric(model, "pure_rmse_reduction")),
                    _basis_points(_daily_metric(model, "expected_shortfall_95")),
                    _basis_points(_daily_metric(model, "mean_transaction_cost")),
                    f"{_daily_metric(model, 'mean_gross_exposure'):.4f}",
                    _basis_points(
                        _daily_metric(model, "economic_book_maximum_drawdown")
                    ),
                )
            )
            + r" \\"
        )
    return "\n".join(
        (
            r"\begin{table}[H]",
            r"\centering\scriptsize",
            r"\caption{Daily headline: lowest pure RMSE within each model family.}",
            r"\label{tab:daily-headline}",
            r"\begin{tabular}{@{}llrrrrrr@{}}",
            r"\toprule",
            r"Model & Fitted policy & Pure RMSE & Reduction & ES$_{95}$ & Cost & Gross & Economic MDD \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        )
    )


def _weekly_headline_table(
    models: tuple[JsonObject, ...],
    weekly: JsonObject,
) -> str:
    comparisons = _require_object(weekly, "comparisons")
    rows = []
    for model in models:
        schedule = _require_string(model, "schedule")
        comparison = _require_object(comparisons, schedule)
        paired = _require_object(comparison, "paired_book_loss_differences")
        case = _require_string(model, "key")
        case_comparison = _require_object(paired, case)
        economic = _require_object(case_comparison, "economic_loss")
        interval = _number_pair(economic, "hac_confidence_interval_95")
        all_metrics = _weekly_all(model)
        rows.append(
            " & ".join(
                (
                    _tex_escape(schedule),
                    _require_string(model, "display_name"),
                    _fit_label(model),
                    _basis_points(_weekly_pure_rmse(model)),
                    _percent(_require_number(all_metrics, "pure_rmse_reduction")),
                    _basis_points(_require_number(all_metrics, "mean_transaction_cost")),
                    _basis_points(_require_number(economic, "mean")),
                    "["
                    + _basis_points(interval[0])
                    + ", "
                    + _basis_points(interval[1])
                    + "]",
                )
            )
            + r" \\"
        )
    return "\n".join(
        (
            r"\begin{table}[H]",
            r"\centering\scriptsize",
            r"\caption{Weekly headline by schedule and model family.}",
            r"\label{tab:weekly-headline}",
            r"\begin{tabular}{@{}lllrrrrr@{}}",
            r"\toprule",
            r"Schedule & Model & Fitted policy & Pure RMSE & Reduction & Cost & $\Delta$ economic & HH 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        )
    )


def _model_catalogue(model_names: list[str]) -> str:
    descriptions = {
        "sanos-lv": (
            "SANOS--LV",
            "SANOS supplies the common market interpolation and digital price; "
            "DLV local volatility supplies delta and eight rolling factor vegas. "
            "A minimum-cost linear program selects sparse option hedges "
            r"\citep{SANOS2026}.",
        ),
        "historical-sanos": (
            "Historical SANOS",
            "This baseline reconstructs same-coordinate public option and target "
            "returns from the trailing SANOS states and solves all four risk "
            r"criteria directly on those empirical scenarios \citep{SANOS2026}.",
        ),
        "robust-covariance": (
            "Robust covariance",
            "This independent baseline uses HAC-screened means, robust regression, "
            "regularized Tyler scatter, and spread-weighted lasso costs "
            r"\citep{Tyler1987,ChenWieselHero2011}.",
        ),
        "gromm": (
            "GROMM",
            "GROMM is the native risk-neutral generator: a 200-strike JAX market "
            "fit, rank-16 AR(1) log-DLV-squared dynamics, and eight Brownian "
            r"substeps per business day \citep{GROMM2026}.",
        ),
        "gromm-q-dlv-vega": (
            "GROMM Q + DLV vegas",
            "This conventional comparator prices by the GROMM risk-neutral "
            "expectation and neutralizes spot delta plus eight contemporaneous "
            r"DLV-factor vegas using common random numbers \citep{GROMM2026}.",
        ),
        "volgan": (
            "VolGAN",
            "VolGAN is a historical-P fixed-grid implied-volatility surface "
            r"generator evaluated with published and rolling-trained settings "
            r"\citep{VuleticCont2024}.",
        ),
        "diffusion-ad-seq-vol": (
            "AD--Seq--Vol",
            "AD--Seq--Vol is a causal sequence diffusion model on 22-session "
            r"surface and normalized-return histories \citep{HanEtAl2026}.",
        ),
        "diffusion-ad-seq-vol-ft": (
            "AD--Seq--Vol--FT",
            "The fine-tuned variant freezes the EMA base and trains a rank-four "
            "adapter with calendar, call-spread, and butterfly penalties "
            r"\citep{HanEtAl2026}.",
        ),
    }
    sections = []
    for model in model_names:
        title, description = descriptions.get(
            model,
            (model.replace("-", " ").title(), "Benchmark model."),
        )
        sections.append(f"\\subsection{{{title}}}\n{description}")
    return "\n\n".join(sections)


def _fit_label(model: JsonObject) -> str:
    objective = _require_string(model, "fitted_objective")
    parameter = model.get("fitted_parameter")
    if parameter is None:
        return _tex_escape(objective)
    if isinstance(parameter, bool) or not isinstance(parameter, int | float):
        raise ValueError("fitted_parameter must be numeric or null")
    return _tex_escape(objective) + f", $\\lambda={float(parameter):g}$"


def _daily_metric(model: JsonObject, name: str) -> float:
    return _require_number(
        _require_object(_require_object(model, "metrics"), "all"),
        name,
    )


def _weekly_all(model: JsonObject) -> JsonObject:
    return _require_object(_require_object(model, "metrics"), "all")


def _weekly_pure_rmse(model: JsonObject) -> float:
    return _require_number(
        _require_object(_weekly_all(model), "pure_hedge_loss"),
        "root_mean_square",
    )


def _result_count(value: object) -> int:
    model = _as_object(value, "model")
    result_count = model.get("result_count")
    if isinstance(result_count, bool) or not isinstance(result_count, int):
        raise ValueError("result_count must be an integer")
    return result_count


def _basis_points(value: float) -> str:
    return f"{DISPLAY_SCALE * value:,.2f}"


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}\\%"


def _integer(value: int) -> str:
    return f"{value:,}"


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _number_pair(payload: JsonObject, name: str) -> tuple[float, float]:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a numeric pair")
    left, right = value
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise ValueError(f"{name} must be a numeric pair")
    return float(left), float(right)


def _read_json_object(path: Path) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary metrics must be an object: {path}")
    return payload


def _as_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_object(payload: JsonObject, name: str) -> JsonObject:
    return _as_object(payload.get(name), name)


def _require_string(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_number(payload: JsonObject, name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def main() -> None:
    """Render and optionally compile a complete authenticated benchmark report."""
    report_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-metrics", type=Path, required=True)
    parser.add_argument("--weekly-metrics", type=Path, required=True)
    parser.add_argument("--daily-objective-tables", type=Path, required=True)
    parser.add_argument("--weekly-objective-tables", type=Path, required=True)
    parser.add_argument("--daily-target-appendix", type=Path, required=True)
    parser.add_argument("--weekly-target-appendix", type=Path, required=True)
    parser.add_argument(
        "--bibliography-source",
        type=Path,
        default=report_directory / "hedging_benchmark_summary_refs.bib",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-date", type=date.fromisoformat)
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()
    pdf = write_summary_document(
        daily_metrics=arguments.daily_metrics,
        weekly_metrics=arguments.weekly_metrics,
        output=arguments.output,
        bibliography_source=arguments.bibliography_source,
        daily_objective_tables=arguments.daily_objective_tables,
        weekly_objective_tables=arguments.weekly_objective_tables,
        daily_target_appendix=arguments.daily_target_appendix,
        weekly_target_appendix=arguments.weekly_target_appendix,
        status_date=arguments.status_date,
        compile_pdf=arguments.compile,
    )
    print(
        json.dumps(
            {
                "schema": DOCUMENT_SCHEMA,
                "tex_output": str(arguments.output),
                "pdf_output": None if pdf is None else str(pdf),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
