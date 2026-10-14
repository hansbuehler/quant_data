"""Render the final public-data hedging benchmark from authenticated metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import TypeAlias

from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    SUMMARY_SCHEMA as DAILY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    build_daily_policy_summary_metrics,
    daily_policy_inputs_from_matrices,
    write_summary_metrics,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    SUMMARY_SCHEMA as WEEKLY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    build_weekly_policy_summary_metrics,
    weekly_policy_inputs_from_matrices,
    write_weekly_summary_metrics,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    DailyProductionMatrixConfig,
    build_daily_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_production_matrix import (
    WeeklyProductionMatrixConfig,
    build_weekly_production_matrix,
)

JsonObject: TypeAlias = dict[str, object]

DOCUMENT_SCHEMA = "hedging-benchmark-summary-document-v4"
DISPLAY_SCALE = 10_000.0
DAILY_MATRIX_SCHEMA = "hedging-daily-production-matrix-v1"
WEEKLY_MATRIX_SCHEMA = "hedging-weekly-production-matrix-v1"
EXPECTED_DAILY_MATRIX_CASES = 57
EXPECTED_WEEKLY_MATRIX_CASES = 94


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
    daily_reference = _daily_reference(daily_models)
    daily_unhedged = _daily_metric(daily_reference, "unhedged_rmse")
    weekly_schedules = sorted(
        {
            _require_string(_as_object(model, "weekly model"), "schedule")
            for model in weekly_models.values()
        }
    )
    model_names = list(
        dict.fromkeys(
            _require_string(_as_object(model, "model"), "model")
            for model in (*daily_models.values(), *weekly_models.values())
        )
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
The common daily unhedged RMSE is {_basis_points(daily_unhedged)} basis points.
Every precommitted objective and parameter is reported; no policy is selected
using the 2023--2025 evaluation outcomes.}}}}

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
\subsection{{Daily objective grid}}
Losses, costs, and drawdowns are basis points of inception normalized spot.
The following tables retain every completed daily policy and evaluate every
precommitted ex-post risk parameter on inception-spot-normalized economic loss.
\input{{{_tex_escape(daily_objective_tables)}}}

\subsection{{Weekly objective grid}}
The following tables retain every completed weekly policy, separately by
schedule and authenticated policy specification. Weekly drawdown is computed
from the cumulative P\&L, equal to minus loss, of the chronologically ordered,
overlapping equal-weighted episode books; it is an episode-origin risk
statistic rather than the drawdown of a non-overlapping self-financing account.
Scores, costs, and drawdowns use basis points of inception normalized spot.
\input{{{_tex_escape(weekly_objective_tables)}}}

\subsection{{Weekly paired reference comparisons}}
For every completed policy and both loss definitions, the tables below report
the equal-weighted thirteen-target episode-book loss minus the schedule-matched
SANOS--LV loss. Negative values favour the candidate. Both the precommitted
Hansen--Hodrick and moving-block-bootstrap 95\% intervals are shown.

{_weekly_comparison_tables(weekly)}

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
The price tables report cross-origin means of inception unit prices. Market is
the common SANOS mark; accounting is the price used in the economic-P\&L
decomposition. A genuine risk-neutral expectation is shown only for a
risk-neutral model, while buyer and seller columns are shown only when the
policy supplies the corresponding indifference prices.
\input{{{_tex_escape(weekly_target_appendix)}}}

\clearpage
\bibliographystyle{{plainnat}}
\bibliography{{hedging_benchmark_summary_refs}}
\end{{document}}
"""


def write_summary_document(
    *,
    dataset: Path,
    daily_matrix: Path,
    daily_matrix_sha256: str,
    weekly_matrix: Path,
    weekly_matrix_sha256: str,
    output: Path,
    bibliography_source: Path,
    status_date: date | None = None,
    compile_pdf: bool = False,
) -> Path | None:
    """Authenticate production matrices and render their final report."""
    dataset = Path(dataset)
    daily_matrix = Path(daily_matrix)
    weekly_matrix = Path(weekly_matrix)
    output = Path(output)
    bibliography_source = Path(bibliography_source)
    _validate_output_boundaries(
        dataset=dataset,
        daily_matrix=daily_matrix,
        weekly_matrix=weekly_matrix,
        bibliography_source=bibliography_source,
        output=output,
    )
    _validate_production_matrix(
        daily_matrix,
        expected_sha256=daily_matrix_sha256,
        expected_schema=DAILY_MATRIX_SCHEMA,
        expected_cases=EXPECTED_DAILY_MATRIX_CASES,
        dataset=dataset,
        weekly=False,
    )
    _validate_production_matrix(
        weekly_matrix,
        expected_sha256=weekly_matrix_sha256,
        expected_schema=WEEKLY_MATRIX_SCHEMA,
        expected_cases=EXPECTED_WEEKLY_MATRIX_CASES,
        dataset=dataset,
        weekly=True,
    )
    daily = build_daily_policy_summary_metrics(
        policies=daily_policy_inputs_from_matrices(
            (daily_matrix,),
            require_ready_complete=True,
        ),
        dataset=dataset,
    )
    weekly = build_weekly_policy_summary_metrics(
        policies=weekly_policy_inputs_from_matrices(
            (weekly_matrix,),
            require_ready_complete=True,
        ),
        dataset=dataset,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_metrics = output.parent / "daily-metrics.json"
    weekly_metrics = output.parent / "weekly-metrics.json"
    daily_macros = output.parent / "daily-macros.tex"
    daily_objective_tables = output.parent / "daily-objective-tables.tex"
    weekly_objective_tables = output.parent / "weekly-objective-tables.tex"
    daily_target_appendix = output.parent / "daily-target-appendix.tex"
    weekly_target_appendix = output.parent / "weekly-target-appendix.tex"
    write_summary_metrics(
        daily,
        json_output=daily_metrics,
        tex_output=daily_macros,
        objective_tables_output=daily_objective_tables,
        objective_appendix_output=daily_target_appendix,
    )
    write_weekly_summary_metrics(
        weekly,
        json_output=weekly_metrics,
        objective_tables_output=weekly_objective_tables,
        target_appendix_output=weekly_target_appendix,
    )
    pdf = _write_rendered_summary_document(
        daily=daily,
        weekly=weekly,
        output=output,
        bibliography_source=bibliography_source,
        daily_objective_tables=daily_objective_tables,
        weekly_objective_tables=weekly_objective_tables,
        daily_target_appendix=daily_target_appendix,
        weekly_target_appendix=weekly_target_appendix,
        status_date=status_date,
        compile_pdf=compile_pdf,
    )
    _require_sha256(daily_matrix, daily_matrix_sha256, "daily production matrix")
    _require_sha256(weekly_matrix, weekly_matrix_sha256, "weekly production matrix")
    _write_document_manifest(
        dataset=dataset,
        daily_matrix=daily_matrix,
        daily_matrix_sha256=daily_matrix_sha256,
        weekly_matrix=weekly_matrix,
        weekly_matrix_sha256=weekly_matrix_sha256,
        daily=daily,
        weekly=weekly,
        output=output,
        artifact_paths=(
            daily_metrics,
            weekly_metrics,
            daily_macros,
            daily_objective_tables,
            weekly_objective_tables,
            daily_target_appendix,
            weekly_target_appendix,
            output.parent / Path(bibliography_source).name,
            output,
            *(tuple() if pdf is None else (pdf,)),
        ),
    )
    return pdf


def _write_rendered_summary_document(
    *,
    daily: JsonObject,
    weekly: JsonObject,
    output: Path,
    bibliography_source: Path,
    daily_objective_tables: Path,
    weekly_objective_tables: Path,
    daily_target_appendix: Path,
    weekly_target_appendix: Path,
    status_date: date | None = None,
    compile_pdf: bool = False,
) -> Path | None:
    """Render already authenticated in-memory summaries and local fragments."""
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


def _validate_production_matrix(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
    expected_cases: int,
    dataset: Path,
    weekly: bool,
) -> None:
    _require_sha256(path, expected_sha256, "production matrix")
    payload = _read_json_object(path)
    if payload.get("schema") != expected_schema:
        raise ValueError(f"production matrix uses an incompatible schema: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError(
            f"production matrix must contain {expected_cases} cases: {path}"
        )
    config = (
        WeeklyProductionMatrixConfig()
        if weekly
        else DailyProductionMatrixConfig()
    )
    if payload.get("code_version") != str(type(config).version):
        raise ValueError(f"production matrix code version is not canonical: {path}")
    canonical_config = json.loads(json.dumps(config.to_dict()))
    if payload.get("config") != canonical_config:
        raise ValueError(f"production matrix configuration is not canonical: {path}")
    if payload.get("case_count") != expected_cases:
        raise ValueError(f"production matrix case count is not canonical: {path}")
    if (
        payload.get("ready_count") != expected_cases
        or payload.get("blocked_count") != 0
        or payload.get("waiting_gate_count") != 0
    ):
        raise ValueError(f"final production matrix is not fully ready: {path}")
    canonical = _canonical_case_identities(dataset=dataset, weekly=weekly)
    identities: set[str] = set()
    for raw_case in cases:
        case = _as_object(raw_case, "production matrix case")
        case_id = _require_string(case, "case_id")
        if case_id in identities:
            raise ValueError(f"production matrix contains duplicate case {case_id}")
        identities.add(case_id)
        identity = _case_identity(case)
        if canonical.get(case_id) != identity:
            raise ValueError(
                f"production matrix case is not canonical: {case_id}"
            )
        readiness = _require_string(case, "readiness")
        if readiness != "ready":
            raise ValueError(f"final production case is not ready: {case_id}")
        command = case.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ValueError(f"final production case has no command: {case_id}")
        if case.get("blocked_reason") is not None:
            raise ValueError(f"ready production case has a blocked reason: {case_id}")
    if identities != set(canonical):
        raise ValueError(f"production matrix case set is not canonical: {path}")


def _canonical_case_identities(
    *,
    dataset: Path,
    weekly: bool,
) -> dict[str, tuple[object, ...]]:
    sentinel = Path("/__hedging_report_canonical_matrix__")
    if weekly:
        cases = build_weekly_production_matrix(
            dataset=dataset,
            output_root=sentinel / "weekly-output",
            checkpoint_root=sentinel / "weekly-checkpoints",
            weekly_input_preflight=None,
        )
    else:
        cases = build_daily_production_matrix(
            dataset=dataset,
            output_root=sentinel / "daily-output",
            checkpoint_root=sentinel / "daily-checkpoints",
        )
    return {
        case.case_id: (
            case.model,
            case.measure,
            case.variant,
            case.objective,
            case.risk_parameter,
        )
        for case in cases
    }


def _case_identity(case: JsonObject) -> tuple[object, ...]:
    parameter = case.get("risk_parameter")
    if parameter is not None:
        parameter = _require_number(case, "risk_parameter")
    objective = case.get("objective")
    if objective is not None:
        objective = _require_string(case, "objective")
    return (
        _require_string(case, "model"),
        _require_string(case, "measure"),
        _require_string(case, "variant"),
        objective,
        parameter,
    )


def _validate_output_boundaries(
    *,
    dataset: Path,
    daily_matrix: Path,
    weekly_matrix: Path,
    bibliography_source: Path,
    output: Path,
) -> None:
    dataset_root = dataset.resolve()
    output_root = output.parent.resolve()
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError("report output must not be inside the public dataset")
    inputs = {
        "dataset manifest": (dataset / "manifest.json").resolve(),
        "daily matrix": daily_matrix.resolve(),
        "weekly matrix": weekly_matrix.resolve(),
        "bibliography": bibliography_source.resolve(),
    }
    if len(set(inputs.values())) != len(inputs):
        raise ValueError("report inputs must be distinct files")
    outputs = tuple(
        (output.parent / name).resolve()
        for name in (
            "daily-metrics.json",
            "weekly-metrics.json",
            "daily-macros.tex",
            "daily-objective-tables.tex",
            "weekly-objective-tables.tex",
            "daily-target-appendix.tex",
            "weekly-target-appendix.tex",
            bibliography_source.name,
            output.name,
            output.with_suffix(".pdf").name,
            "hedging_benchmark_summary_manifest.json",
        )
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("report output artifacts must use distinct paths")
    aliases = set(inputs.values()).intersection(outputs)
    if aliases:
        raise ValueError(f"report output aliases an authenticated input: {aliases}")
    for input_path in inputs.values():
        for output_path in outputs:
            if (
                input_path.exists()
                and output_path.exists()
                and os.path.samefile(input_path, output_path)
            ):
                raise ValueError(
                    "report output hardlinks an authenticated input: "
                    f"{output_path}"
                )
    for index, left in enumerate(outputs):
        if not left.exists():
            continue
        for right in outputs[index + 1 :]:
            if right.exists() and os.path.samefile(left, right):
                raise ValueError(
                    f"report output artifacts are hardlinked: {left}, {right}"
                )


def _write_document_manifest(
    *,
    dataset: Path,
    daily_matrix: Path,
    daily_matrix_sha256: str,
    weekly_matrix: Path,
    weekly_matrix_sha256: str,
    daily: JsonObject,
    weekly: JsonObject,
    output: Path,
    artifact_paths: tuple[Path, ...],
) -> None:
    dataset_manifest = Path(dataset) / "manifest.json"
    manifest_sha256 = _sha256(dataset_manifest)
    if manifest_sha256 != _require_string(daily, "dataset_manifest_sha256"):
        raise ValueError("rendered daily metrics do not match the public dataset")
    if manifest_sha256 != _require_string(weekly, "dataset_manifest_sha256"):
        raise ValueError("rendered weekly metrics do not match the public dataset")
    artifacts = {path.name: _sha256(path) for path in artifact_paths}
    payload = {
        "schema": "hedging-benchmark-summary-bundle-v1",
        "dataset_manifest_sha256": manifest_sha256,
        "daily_matrix": str(daily_matrix),
        "daily_matrix_sha256": daily_matrix_sha256,
        "weekly_matrix": str(weekly_matrix),
        "weekly_matrix_sha256": weekly_matrix_sha256,
        "daily_target_notional_identity": _require_string(
            daily,
            "target_notional_identity",
        ),
        "weekly_target_notional_identity": _require_string(
            weekly,
            "target_notional_identity",
        ),
        "daily_source_identities": _source_identities(daily),
        "weekly_source_identities": _source_identities(weekly),
        "artifacts": artifacts,
    }
    manifest = output.parent / "hedging_benchmark_summary_manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)


def _source_identities(payload: JsonObject) -> JsonObject:
    models = _require_object(payload, "models")
    return {
        key: {
            "run_identity": _require_string(_as_object(model, key), "run_identity"),
            "aggregation_identity": _require_string(
                _as_object(model, key),
                "aggregation_identity",
            ),
            "result_count": _result_count(model),
        }
        for key, model in models.items()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected.lower())
    ):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    actual = _sha256(path)
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )


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


def _daily_reference(models: JsonObject) -> JsonObject:
    references = tuple(
        _as_object(value, key)
        for key, value in models.items()
        if _require_string(_as_object(value, key), "model") == "sanos-lv"
    )
    if len(references) != 1:
        raise ValueError("daily report requires one SANOS-LV reference")
    return references[0]


def _weekly_comparison_tables(weekly: JsonObject) -> str:
    models = _require_object(weekly, "models")
    comparisons = _require_object(weekly, "comparisons")
    schedules = sorted(
        {
            _require_string(_as_object(model, key), "schedule")
            for key, model in models.items()
        }
    )
    lines: list[str] = []
    for schedule in schedules:
        comparison = _require_object(comparisons, schedule)
        paired = _require_object(comparison, "paired_book_loss_differences")
        lines.extend(
            (
                r"\begin{landscape}",
                r"\scriptsize",
                r"\begin{longtable}{@{}llllrrrrrr@{}}",
                r"\caption{Paired "
                + _tex_escape(schedule)
                + r" episode-book loss differences against SANOS--LV.}\\",
                r"\toprule",
                r"Policy & Specification & Fit & Loss & $\Delta$ mean & HH 95\% & MBB 95\% & $n$ & HH lags & MBB block \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Policy & Specification & Fit & Loss & $\Delta$ mean & HH 95\% & MBB 95\% & $n$ & HH lags & MBB block \\",
                r"\midrule",
                r"\endhead",
            )
        )
        schedule_models = sorted(
            (
                _as_object(model, key)
                for key, model in models.items()
                if _require_string(_as_object(model, key), "schedule") == schedule
            ),
            key=lambda model: _require_string(model, "key"),
        )
        for model in schedule_models:
            case = _require_string(model, "key")
            case_comparison = _require_object(paired, case)
            for loss_name, loss_label in (
                ("pure_hedge_loss", "Pure"),
                ("economic_loss", "Economic"),
            ):
                loss = _require_object(case_comparison, loss_name)
                hac = _number_pair(loss, "hac_confidence_interval_95")
                bootstrap = _number_pair(
                    loss,
                    "bootstrap_confidence_interval_95",
                )
                lines.append(
                    " & ".join(
                        (
                            _require_string(model, "display_name"),
                            _tex_escape(_require_string(model, "specification")),
                            _fit_label(model),
                            loss_label,
                            _basis_points(_require_number(loss, "mean")),
                            _interval_basis_points(hac),
                            _interval_basis_points(bootstrap),
                            str(_require_integer(loss, "observations")),
                            str(_require_integer(loss, "hac_lags")),
                            str(_require_integer(loss, "bootstrap_block_length")),
                        )
                    )
                    + r" \\"
                )
        lines.extend(
            (r"\bottomrule", r"\end{longtable}", r"\end{landscape}", "")
        )
    return "\n".join(lines)


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
            "\nUnder its Gaussian law, entropy equals the corresponding mean--variance "
            "policy and power-exponential uses the exact Gaussian OCE "
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
        if model not in descriptions:
            raise ValueError(f"unknown benchmark model: {model}")
        title, description = descriptions[model]
        sections.append(f"\\subsection{{{title}}}\n{description}")
    return "\n\n".join(sections)


def _fit_label(model: JsonObject) -> str:
    objective = _require_string(model, "fitted_objective")
    parameter = model.get("fitted_parameter")
    if parameter is None:
        return _tex_escape(objective)
    if isinstance(parameter, bool) or not isinstance(parameter, int | float):
        raise ValueError("fitted_parameter must be numeric or null")
    symbol = "\\kappa" if objective == "mean-volatility" else "\\lambda"
    return _tex_escape(objective) + f", ${symbol}={float(parameter):g}$"


def _daily_metric(model: JsonObject, name: str) -> float:
    return _require_number(
        _require_object(_require_object(model, "metrics"), "all"),
        name,
    )


def _result_count(value: object) -> int:
    model = _as_object(value, "model")
    result_count = model.get("result_count")
    if isinstance(result_count, bool) or not isinstance(result_count, int):
        raise ValueError("result_count must be an integer")
    return result_count


def _basis_points(value: float) -> str:
    return f"{DISPLAY_SCALE * value:,.2f}"


def _interval_basis_points(value: tuple[float, float]) -> str:
    left, right = value
    return f"[{_basis_points(left)}, {_basis_points(right)}]"


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


def _require_integer(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def main() -> None:
    """Render and optionally compile a complete authenticated benchmark report."""
    report_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--daily-matrix", type=Path, required=True)
    parser.add_argument("--daily-matrix-sha256", required=True)
    parser.add_argument("--weekly-matrix", type=Path, required=True)
    parser.add_argument("--weekly-matrix-sha256", required=True)
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
        dataset=arguments.dataset,
        daily_matrix=arguments.daily_matrix,
        daily_matrix_sha256=arguments.daily_matrix_sha256,
        weekly_matrix=arguments.weekly_matrix,
        weekly_matrix_sha256=arguments.weekly_matrix_sha256,
        output=arguments.output,
        bibliography_source=arguments.bibliography_source,
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
