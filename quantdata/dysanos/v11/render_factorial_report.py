"""Render the evolving four-cell DYSANOS 1.1 audit as a concise PDF."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_page(
    document: PdfPages,
    *,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    notes: list[str] = (),
    font_size: float = 9.0,
) -> None:
    figure, axis = plt.subplots(figsize=(12.0, 8.2))
    axis.axis("off")
    figure.suptitle(title, x=0.05, y=0.96, ha="left", fontsize=16)
    column_count = len(columns)
    header_width = max(11, 100 // column_count)
    body_width = max(12, 125 // column_count)

    def wrap(value: str, width: int) -> str:
        return textwrap.fill(value, width=width, break_long_words=False)

    table = axis.table(
        cellText=[[wrap(value, body_width) for value in row] for row in rows],
        colLabels=[wrap(value, header_width) for value in columns],
        cellLoc="center",
        colLoc="center",
        bbox=(0.03, max(0.16, 0.87 - 0.055 * (len(rows) + 1)), 0.94, min(0.68, 0.055 * (len(rows) + 1))),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c6cdd4")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#e9eff4")
            cell.get_text().set_weight("bold")
    for index, note in enumerate(notes):
        figure.text(0.05, 0.12 - 0.028 * index, note, ha="left", fontsize=9)
    document.savefig(figure)
    plt.close(figure)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _predictive_rows(
    reports: list[tuple[str, dict[str, object]]],
    *,
    period: str,
    dtes: list[int],
) -> list[list[str]]:
    rows = []
    for label, report in reports:
        panel = report[period]
        if panel is None:
            continue
        for index, tenor in enumerate(dtes):
            coverage = panel["central_interval_coverage"]
            rows.append(
                [
                    label,
                    f"{tenor}D",
                    f"{panel['innovation_bias'][index]:+.4f}",
                    f"{panel['predictive_variance_ratio'][index]:.2f}",
                    f"{panel['mean_crps'][index]:.4f}",
                    *(
                        _percent(coverage[key][index])
                        for key in ("0.50", "0.80", "0.95")
                    ),
                ]
            )
    return rows


def run(
    *,
    observation_report: Path,
    training_fit_report: Path,
    radius_cap_report: Path,
    one_step_report: Path,
    output: Path,
    forward_fit_report: Path | None = None,
    daily_forward_fit_report: Path | None = None,
    weekly_tail_fit_report: Path | None = None,
    daily_normal_fit_report: Path | None = None,
    daily_tail_fit_report: Path | None = None,
    daily_normal_one_step_report: Path | None = None,
    weekly_tail_one_step_report: Path | None = None,
    daily_tail_one_step_report: Path | None = None,
    weekly_predictive_report: Path | None = None,
    weekly_tail_predictive_report: Path | None = None,
    daily_predictive_report: Path | None = None,
    daily_tail_predictive_report: Path | None = None,
    weekly_front_pit_report: Path | None = None,
    daily_front_pit_report: Path | None = None,
    daily_radius_cap_report: Path | None = None,
    conditional_sequence_report: Path | None = None,
    multistep_reports: list[Path] | None = None,
) -> Path:
    """Render only completed results; label absent forward metrics explicitly."""
    observed = _load(observation_report)
    training = _load(training_fit_report)
    radius = _load(radius_cap_report)
    weekly_front_pit = (
        _load(weekly_front_pit_report)
        if weekly_front_pit_report is not None else None
    )
    daily_front_pit = (
        _load(daily_front_pit_report)
        if daily_front_pit_report is not None else None
    )
    daily_radius = (
        _load(daily_radius_cap_report)
        if daily_radius_cap_report is not None else None
    )
    sequence_models = (
        _load(conditional_sequence_report)
        if conditional_sequence_report is not None else None
    )
    path_models = {
        report["variant"]: report
        for report in (_load(path) for path in (multistep_reports or []))
    }
    one_step = _load(one_step_report)
    forward = _load(forward_fit_report) if forward_fit_report is not None else None
    daily_forward = (
        _load(daily_forward_fit_report)
        if daily_forward_fit_report is not None else None
    )
    weekly_tail = (
        _load(weekly_tail_fit_report) if weekly_tail_fit_report is not None else None
    )
    daily_normal = (
        _load(daily_normal_fit_report) if daily_normal_fit_report is not None else None
    )
    daily_tail = (
        _load(daily_tail_fit_report) if daily_tail_fit_report is not None else None
    )
    daily_joint = (
        daily_normal is not None
        and daily_normal.get("schema") == "dysanos-v11-pre2025-daily-spot-joint-v1"
    )
    daily_one_step = (
        _load(daily_normal_one_step_report)
        if daily_normal_one_step_report is not None
        else None
    )
    weekly_tail_one_step = (
        _load(weekly_tail_one_step_report)
        if weekly_tail_one_step_report is not None
        else None
    )
    daily_tail_one_step = (
        _load(daily_tail_one_step_report)
        if daily_tail_one_step_report is not None
        else None
    )
    predictive_reports = [
        (label, _load(path))
        for label, path in (
            ("Weekly Normal", weekly_predictive_report),
            ("Weekly Tail", weekly_tail_predictive_report),
            ("Daily Normal" if daily_joint else "Daily Normal pilot", daily_predictive_report),
            ("Daily Tail", daily_tail_predictive_report),
        )
        if path is not None
    ]
    dtes = observed["business_dtes"]
    before = observed["pre_evaluation"]
    after = observed["forward_evaluation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as document:
        train_fit = training["joint_fit"]["statistics"]["all"]
        daily_train_fit = (
            daily_normal["joint_fit"]["statistics"]["all"]
            if daily_joint else train_fit
        )
        cells = [
            [
                "Weekly / Normal",
                "600 (capped)",
                f"{train_fit['percent_zero']:.2f}%",
                (
                    "pending calibration"
                    if forward is None
                    else f"{forward['quote_fit']['statistics']['all']['percent_zero']:.2f}%"
                ),
                (
                    "pre-2025 32-draw forecast pilot"
                    if weekly_predictive_report is not None
                    else "one-step audit available"
                ),
            ],
            [
                "Weekly / Tail Diffusion",
                "shared 600 + tail MLE" if weekly_tail is not None else "pending",
                f"{train_fit['percent_zero']:.2f}%" if weekly_tail is not None else "—",
                (
                    "pending calibration"
                    if forward is None
                    else f"{forward['quote_fit']['statistics']['all']['percent_zero']:.2f}%"
                ) if weekly_tail is not None else "—",
                (
                    "stationary one-step pilot"
                    if weekly_tail_one_step is not None
                    else "tail fit complete; paths pending"
                ) if weekly_tail is not None else "pending",
            ],
            [
                "Daily / Normal",
                ("600 (joint)" if daily_joint else "shared 600 + AR pilot") if daily_normal is not None else "pending",
                f"{daily_train_fit['percent_zero']:.2f}%" if daily_normal is not None else "—",
                (
                    "pending calibration" if daily_forward is None
                    else f"{daily_forward['quote_fit']['statistics']['all']['percent_zero']:.2f}%"
                ) if daily_joint else ("pending calibration" if daily_normal is not None else "—"),
                (
                    ("one-step audit" if daily_joint else "one-step pilot audit")
                    if daily_one_step is not None
                    else "held-out audit pending"
                ) if daily_normal is not None else "pending",
            ],
            [
                "Daily / Tail Diffusion",
                ("daily joint 600 + tail MLE" if daily_joint else "shared 600 + AR + tail pilot") if daily_tail is not None else "pending",
                f"{daily_train_fit['percent_zero']:.2f}%" if daily_tail is not None else "—",
                (
                    "pending calibration" if daily_forward is None
                    else f"{daily_forward['quote_fit']['statistics']['all']['percent_zero']:.2f}%"
                ) if daily_joint and daily_tail is not None else ("pending calibration" if daily_tail is not None else "—"),
                (
                    ("stationary one-step audit" if daily_joint else "stationary one-step pilot")
                    if daily_tail_one_step is not None
                    else "held-out audit pending"
                ) if daily_tail is not None else "pending",
            ],
        ]
        _table_page(
            document,
            title="DYSANOS 1.1 — four-cell model audit",
            columns=["Dynamics / innovation", "Fit epochs", "<2025 inside bid–ask", "2025 state fit", "Forecast status"],
            rows=cells,
            notes=[
                "2022–2024 shared decoder and weekly AR(1) are frozen for 2025; only each date's 200 state coordinates are calibrated.",
                "The 600-epoch weekly-Normal checkpoint did not meet its strict loss-plateau criterion; no extra epochs were added.",
                (
                    "The daily cells use a separate 600-epoch joint decoder/AR fit; each Tail Diffusion is fitted afterward."
                    if daily_joint else
                    "Other cells currently share that decoder; their post-fit dynamics pilots are not separate 600-epoch joint refits."
                ),
                f"Observed transitions: {before['observations']} before 2025 and {after['observations']} in 2025.",
            ],
        )

        if weekly_tail is not None or daily_normal is not None or daily_tail is not None:
            dynamics_rows = []
            if weekly_tail is not None:
                weekly_config = weekly_tail["fit"]["config"]
                dynamics_rows.append(
                    [
                        "Weekly / Tail Diffusion",
                        str(weekly_tail["weekly_residual_count"]),
                        f"{weekly_config['persistence']:.3f}",
                        f"{weekly_config['stationary_log_variance']:.4f}",
                        "one-step pilot" if weekly_tail_one_step is not None else "pending paths",
                    ]
                )
            if daily_normal is not None:
                dynamics_rows.append(
                    [
                        "Daily / Normal" if daily_joint else "Daily / Normal (pilot)",
                        str(daily_normal["daily_dynamics_transition_count"] if daily_joint else daily_normal["daily_transitions"]),
                        "—",
                        "—",
                        f"AR radius {(daily_normal['joint_dynamics']['transition_spectral_radius'] if daily_joint else daily_normal['spectral_radius']):.3f}",
                    ]
                )
            if daily_tail is not None:
                daily_config = daily_tail["fit"]["config"]
                dynamics_rows.append(
                    [
                        "Daily / Tail Diffusion" if daily_joint else "Daily / Tail Diffusion (pilot)",
                        str(daily_tail["daily_residual_count"]),
                        f"{daily_config['persistence']:.3f}",
                        f"{daily_config['stationary_log_variance']:.4f}",
                        f"white radius p99 {daily_tail['whitened_radius_quantiles']['0.99']:.2f}",
                    ]
                )
            _table_page(
                document,
                title=("Innovation-law fits — dynamics comparison" if daily_joint else "Innovation-law fits — provisional dynamics comparison"),
                columns=["Cell", "Training residuals", "OU persistence", "OU log-variance", "Audit note"],
                rows=dynamics_rows,
                notes=[
                    (
                        "Daily rank-201 joint fit uses the final 252 pre-2025 sessions for its dynamics term; the full pre-2025 sample informs quote fit."
                        if daily_joint else
                        "Daily rank-201 pilot uses a diagonal AR transition: the dense transition overfit 251 observations."
                    ),
                    "Tail likelihood improvements are in-sample and are not evidence of better forward paths.",
                    "Spot retains its front-SANOS marginal; joint Tail simulation requires a marginal-preserving copula transform.",
                ],
            )

        before_lev = before["leverage"]
        after_lev = after["leverage"]
        decoded = one_step["decoded_training_leverage"]
        simulated = one_step["simulated_terminal_one_step_leverage"]
        model_panels = [("Weekly Normal", simulated)]
        if weekly_tail_one_step is not None:
            model_panels.append(("Weekly Tail", weekly_tail_one_step["leverage"]))
        if daily_one_step is not None:
            model_panels.append(("Daily Normal", daily_one_step["leverage"]))
        if daily_tail_one_step is not None:
            model_panels.append(("Daily Tail", daily_tail_one_step["leverage"]))
        rows = []
        for index, tenor in enumerate(dtes):
            rows.append(
                [
                    f"{tenor}D",
                    f"{before_lev['correlation'][index]:.3f} [{before_lev['correlation_interval_95'][index][0]:.3f}, {before_lev['correlation_interval_95'][index][1]:.3f}]",
                    f"{after_lev['correlation'][index]:.3f} [{after_lev['correlation_interval_95'][index][0]:.3f}, {after_lev['correlation_interval_95'][index][1]:.3f}]",
                    f"{decoded['correlation'][index]:.3f}",
                    f"{simulated['correlation'][index]:.3f} ± {simulated['antithetic_pair_correlation_standard_error'][index]:.3f}",
                ]
            )
        _table_page(
            document,
            title="Spot / decoded ATM-IV leverage — correlation",
            columns=["Tenor", "Historical <2025 [95% block CI]", "Historical 2025 [95% block CI]", "Decoded <2025", "Model next day ± MC SE"],
            rows=rows,
            notes=[
                "Historical ATM IV is decoded from DYSANOS's independently fitted native SANOS surfaces.",
                "Spot is the forward-adjusted log return; IV is the change in log rolling-tenor forward-ATM Black IV.",
                "The model column samples one day from the 2024-12-31 state; it is not the 2025 forecast score.",
                "Historical uncertainty uses 10-session moving blocks; model MC SE uses independent antithetic pairs.",
            ],
            font_size=8.4,
        )
        if len(model_panels) > 1:
            comparison_rows = []
            for index, tenor in enumerate(dtes):
                comparison_rows.append(
                    [
                        f"{tenor}D",
                        f"{before_lev['correlation'][index]:.3f}",
                        f"{after_lev['correlation'][index]:.3f}",
                        *(
                            f"{panel['correlation'][index]:.3f} ± {panel['antithetic_pair_correlation_standard_error'][index]:.3f}"
                            for _, panel in model_panels
                        ),
                    ]
                )
            _table_page(
                document,
                title="One-step leverage correlation — model comparison",
                columns=["Tenor", "Historical <2025", "Historical 2025", *(label + " ± MC SE" for label, _ in model_panels)],
                rows=comparison_rows,
                notes=[
                    "Every model column simulates 10,000 paths from 2024-12-31; none is a 2025 forecast score.",
                    "Tail one-step pilots draw the OU scale stationarily and preserve the front SANOS spot marginal.",
                    (
                        "Daily models use their own 600-epoch joint decoder/AR fit; Tail scales are fitted afterward."
                        if daily_joint else
                        "Daily pilots share the 600-epoch decoder and fit a diagonal daily AR separately."
                    ),
                ],
                font_size=8.0,
            )
        for statistic, title in (
            ("covariance", "Spot / ATM-IV leverage — covariance"),
            ("slope", "Spot / ATM-IV leverage — regression slope"),
        ):
            rows = []
            for index, tenor in enumerate(dtes):
                key = f"{statistic}_interval_95"
                rows.append(
                    [
                        f"{tenor}D",
                        f"{before_lev[statistic][index]:.4f} [{before_lev[key][index][0]:.4f}, {before_lev[key][index][1]:.4f}]",
                        f"{after_lev[statistic][index]:.4f} [{after_lev[key][index][0]:.4f}, {after_lev[key][index][1]:.4f}]",
                        f"{decoded[statistic][index]:.4f}",
                        f"{simulated[statistic][index]:.4f}",
                    ]
                )
            _table_page(
                document,
                title=title,
                columns=["Tenor", "Historical <2025 [95% block CI]", "Historical 2025 [95% block CI]", "Decoded <2025", "Model next day"],
                rows=rows,
                notes=[
                    "Covariance uses 1/n within each sample; slope = covariance / spot-return variance.",
                    "Model one-step values are from the terminal pre-2025 state; 2025 forecast scores are reported separately.",
                ],
                font_size=8.5,
            )
        if len(model_panels) > 1:
            for statistic in ("covariance", "slope"):
                comparison_rows = []
                for index, tenor in enumerate(dtes):
                    comparison_rows.append(
                        [
                            f"{tenor}D",
                            f"{before_lev[statistic][index]:.5f}",
                            f"{after_lev[statistic][index]:.5f}",
                            *(
                                f"{panel[statistic][index]:.5f}"
                                for _, panel in model_panels
                            ),
                        ]
                    )
                _table_page(
                    document,
                    title=f"One-step leverage {statistic} — model comparison",
                    columns=["Tenor", "Historical <2025", "Historical 2025", *(label for label, _ in model_panels)],
                    rows=comparison_rows,
                    notes=[
                        "Historical 95% block-bootstrap intervals are on the preceding covariance and slope pages.",
                        (
                            "The daily cells use their own joint decoder/AR fit; Tail laws are fitted afterward."
                            if daily_joint else
                            "The daily and Tail cells remain post-fit pilots, not separate joint decoder refits."
                        ),
                    ],
                    font_size=8.0,
                )

        for label, panel in (("Historical <2025", before), ("Historical 2025", after)):
            ac = panel["autocorrelation"]
            rows = []
            for index, tenor in enumerate(dtes):
                rows.append(
                    [
                        f"{tenor}D",
                        *(
                            f"{ac['atm_log_iv_changes'][str(lag)][index]:.2f}"
                            for lag in (1, 5, 20)
                        ),
                        *(
                            f"{ac['atm_log_iv_levels'][str(lag)][index]:.2f}"
                            for lag in (1, 5, 20)
                        ),
                        *(
                            f"{ac['absolute_atm_log_iv_changes'][str(lag)][index]:.2f}"
                            for lag in (1, 5, 20)
                        ),
                    ]
                )
            _table_page(
                document,
                title=f"{label} — 1/5/20-session autocorrelation",
                columns=["Tenor", "ΔIV 1", "ΔIV 5", "ΔIV 20", "IV 1", "IV 5", "IV 20", "|ΔIV| 1", "|ΔIV| 5", "|ΔIV| 20"],
                rows=rows,
                notes=[
                    "Absolute spot-return autocorrelation: "
                    + ", ".join(
                        f"lag {lag}: {ac['absolute_spot_log_returns'][str(lag)][0]:.3f}"
                        for lag in (1, 5, 20)
                    ),
                    (
                        "The corresponding model path-autocorrelation panel follows below."
                        if path_models else
                        "The corresponding model path-autocorrelation panel is pending multi-step simulation."
                    ),
                ],
                font_size=8.1,
            )

        rows = []
        for index, tenor in enumerate(dtes):
            train_tail = before["downside_leverage"]
            forward_tail = after["downside_leverage"]
            rows.append(
                [
                    f"{tenor}D",
                    f"{train_tail['negative_spot_slope'][index]:.2f}",
                    f"{train_tail['nonnegative_spot_slope'][index]:.2f}",
                    _percent(train_tail["joint_down_up_rate"][index]),
                    f"{forward_tail['negative_spot_slope'][index]:.2f}",
                    f"{forward_tail['nonnegative_spot_slope'][index]:.2f}",
                    _percent(forward_tail["joint_down_up_rate"][index]),
                ]
            )
        _table_page(
            document,
            title="Downside leverage and joint spot-down / IV-up tails",
            columns=["Tenor", "<2025 down slope", "<2025 up slope", "<2025 joint", "2025 down slope", "2025 up slope", "2025 joint"],
            rows=rows,
            notes=[
                "Co-exceedance: spot below its pre-2025 5th percentile AND Δlog IV above its pre-2025 95th percentile.",
                "The thresholds are frozen before 2025; slopes condition on the sign of the spot return.",
            ],
            font_size=8.4,
        )

        if path_models:
            labels = {
                "weekly_normal": "Weekly Normal",
                "weekly_tail": "Weekly Tail",
                "daily_normal": "Daily Normal",
                "daily_tail": "Daily Tail",
                "lstm": "LSTM",
                "selective_ssm": "Selective SSM",
            }
            path_rows = []
            for index, tenor in enumerate(dtes):
                path_rows.append(
                    [
                        f"{tenor}D",
                        f"{before_lev['correlation'][index]:.3f}",
                        f"{after_lev['correlation'][index]:.3f}",
                        *(
                            f"{path_models[variant]['leverage']['correlation'][index]:.3f}"
                            for variant in labels if variant in path_models
                        ),
                    ]
                )
            _table_page(
                document,
                title="Recursive-path leverage correlation — model comparison",
                columns=["Tenor", "Historical <2025", "Historical 2025", *(
                    label for variant, label in labels.items() if variant in path_models
                )],
                rows=path_rows,
                notes=[
                    "Each model is rolled forward for 60 sessions from the final pre-2025 state; correlations pool within-path returns.",
                    "These are prospective simulations, not the held-out 2025 forecast scores reported separately.",
                ],
                font_size=8.0,
            )

            rows = []
            for label, panel in (
                ("Historical <2025", before),
                ("Historical 2025", after),
                *((label, path_models[variant]) for variant, label in labels.items() if variant in path_models),
            ):
                ac = panel["autocorrelation"]
                index = dtes.index(20)
                rows.append(
                    [
                        label,
                        *(f"{ac['atm_log_iv_changes'][str(lag)][index]:.2f}" for lag in (1, 5, 20)),
                        *(f"{ac['atm_log_iv_levels'][str(lag)][index]:.2f}" for lag in (1, 5, 20)),
                        *(f"{ac['absolute_atm_log_iv_changes'][str(lag)][index]:.2f}" for lag in (1, 5, 20)),
                    ]
                )
            _table_page(
                document,
                title="Twenty-day ATM-IV autocorrelation — historical and simulated",
                columns=["Law", "ΔIV 1", "ΔIV 5", "ΔIV 20", "IV 1", "IV 5", "IV 20", "|ΔIV| 1", "|ΔIV| 5", "|ΔIV| 20"],
                rows=rows,
                notes=[
                    "Lags are sessions. Simulated correlations pool only within-path lag pairs.",
                    "The 60-session simulated samples start from the same final pre-2025 state.",
                ],
                font_size=7.8,
            )

            rows = []
            for label, panel in (
                ("Historical <2025", before),
                ("Historical 2025", after),
                *((label, path_models[variant]) for variant, label in labels.items() if variant in path_models),
            ):
                index = dtes.index(20)
                ac = panel["autocorrelation"]["absolute_spot_log_returns"]
                tail = panel["downside_leverage"]
                rows.append(
                    [
                        label,
                        *(f"{ac[str(lag)][0]:.2f}" for lag in (1, 5, 20)),
                        f"{tail['negative_spot_slope'][index]:.2f}",
                        f"{tail['nonnegative_spot_slope'][index]:.2f}",
                        _percent(tail["joint_down_up_rate"][index]),
                    ]
                )
            _table_page(
                document,
                title="Spot clustering and downside 20D ATM-IV response",
                columns=["Law", "|spot| 1", "|spot| 5", "|spot| 20", "Down slope", "Up slope", "Down/up joint"],
                rows=rows,
                notes=[
                    "Joint event uses the fixed pre-2025 spot 5th and IV-change 95th percentile thresholds.",
                    "Down/up slopes condition on the sign of the spot return.",
                ],
            )

            radius_rows = []
            for variant, label in labels.items():
                if variant not in path_models:
                    continue
                panel = path_models[variant]
                radius_rows.append(
                    [
                        label,
                        *(f"{panel['innovation_radius'][str(weeks)]['p99']:.2f}" for weeks in (1, 2, 3, 4)),
                        _percent(panel["innovation_radius"]["4"]["fraction_above_gaussian_p99"]),
                        _percent(panel["decoder_cap"]["fraction_at_hard_cap"]),
                    ]
                )
            _table_page(
                document,
                title="Simulated 1–4-week radius tails and decoder saturation",
                columns=["Law", "1W p99", "2W p99", "3W p99", "4W p99", "4W > Gauss p99", "DLV at cap"],
                rows=radius_rows,
                notes=[
                    "Radius uses each law's normal baseline covariance compounded over 5–20 session steps.",
                    "Tail Diffusion scales spot and surface shocks; a conditional PIT preserves the front SANOS spot marginal.",
                ],
            )

        pit = one_step["historical_training_front_pit"]
        tail_rows = [
            ["PIT mean", f"{pit['pit_mean']:.3f}"],
            ["PIT KS distance", f"{pit['pit_ks_statistic']:.3f}"],
            ["PIT < 1% / < 5%", f"{_percent(pit['tail_frequency']['below_0_01'])} / {_percent(pit['tail_frequency']['below_0_05'])}"],
            ["PIT > 95% / > 99%", f"{_percent(pit['tail_frequency']['above_0_95'])} / {_percent(pit['tail_frequency']['above_0_99'])}"],
            ["Training DLV coordinates at cap", _percent(radius["decoder_cap"]["pre_evaluation"]["fraction_at_hard_cap"])],
            [
                "Simulated next-state DLV coordinates at cap",
                f"{one_step['simulated_next_state_decoder_cap']['fraction_at_hard_cap'] * one_step['simulated_next_state_decoder_cap']['dlv_coordinates']:.0f} / {one_step['simulated_next_state_decoder_cap']['dlv_coordinates']:,}",
            ],
            [
                "Simulated next-state coordinates within 0.1 log-cap",
                f"{one_step['simulated_next_state_decoder_cap']['fraction_within_0_1_log_unit_of_cap'] * one_step['simulated_next_state_decoder_cap']['dlv_coordinates']:.0f} / {one_step['simulated_next_state_decoder_cap']['dlv_coordinates']:,}",
            ],
        ]
        for weeks in (1, 2, 3, 4):
            item = radius["weeks"][str(weeks)]
            stats = item["pre_evaluation"]
            tail_rows.append(
                [
                    f"{weeks}W whitened radius p99 / > Gaussian p99",
                    f"{stats['p99']:.2f} / {_percent(stats['fraction_above_gaussian_p99'])}",
                ]
            )
        _table_page(
            document,
            title="Front-law PIT, innovation tails and decoder cap",
            columns=["Statistic", "Weekly-Normal / pre-2025"],
            rows=tail_rows,
            notes=[
                "PIT uses the realized next-day forward-adjusted spot return under the previous day's decoded front SANOS law.",
                "The PIT KS p-value assumes iid data and is not used as a calibrated time-series test.",
                "Radius = squared whitened state innovation / 200; Gaussian reference p99 is horizon-specific.",
            ],
        )

        if weekly_front_pit is not None and daily_front_pit is not None:
            pit_rows = []
            for label, key in (
                ("PIT mean", "pit_mean"),
                ("PIT KS distance", "pit_ks_statistic"),
                ("Gaussian-score SD", "gaussian_score_standard_deviation"),
            ):
                pit_rows.append(
                    [label, *(
                        f"{report[period][key]:.3f}"
                        for report, period in (
                            (weekly_front_pit, "pre_evaluation"),
                            (weekly_front_pit, "forward_evaluation"),
                            (daily_front_pit, "pre_evaluation"),
                            (daily_front_pit, "forward_evaluation"),
                        )
                    )]
                )
            for label, key in (("PIT < 1%", "below_0_01"), ("PIT > 99%", "above_0_99")):
                pit_rows.append(
                    [label, *(
                        _percent(report[period]["tail_frequency"][key])
                        for report, period in (
                            (weekly_front_pit, "pre_evaluation"),
                            (weekly_front_pit, "forward_evaluation"),
                            (daily_front_pit, "pre_evaluation"),
                            (daily_front_pit, "forward_evaluation"),
                        )
                    )]
                )
            _table_page(
                document,
                title="Front-spot law calibration — chronological comparison",
                columns=["Metric", "Weekly <2025", "Weekly 2025", "Daily <2025", "Daily 2025"],
                rows=pit_rows,
                notes=[
                    "Each PIT uses the prior date's fitted state and the realized next-session spot return.",
                    "The KS distance is descriptive for this serially dependent sample; no iid p-value is interpreted.",
                ],
            )

        if (
            daily_radius is not None
            and radius["weeks"]["1"]["forward_evaluation"].get("observations", 0) > 0
        ):
            radius_rows = []
            for weeks in (1, 2, 3, 4):
                weekly_panel = radius["weeks"][str(weeks)]
                daily_panel = daily_radius["weeks"][str(weeks)]
                radius_rows.append(
                    [
                        f"{weeks}W p99",
                        f"{weekly_panel['pre_evaluation']['p99']:.2f}",
                        f"{weekly_panel['forward_evaluation']['p99']:.2f}",
                        f"{daily_panel['pre_evaluation']['p99']:.2f}",
                        f"{daily_panel['forward_evaluation']['p99']:.2f}",
                    ]
                )
                radius_rows.append(
                    [
                        f"{weeks}W > Gaussian p99",
                        *(
                            _percent(panel[period]["fraction_above_gaussian_p99"])
                            for panel, period in (
                                (weekly_panel, "pre_evaluation"),
                                (weekly_panel, "forward_evaluation"),
                                (daily_panel, "pre_evaluation"),
                                (daily_panel, "forward_evaluation"),
                            )
                        ),
                    ]
                )
            _table_page(
                document,
                title="Whitened innovation-radius tails — 1–4 weeks",
                columns=["Metric", "Weekly <2025", "Weekly 2025", "Daily <2025", "Daily 2025"],
                rows=radius_rows,
                notes=[
                    "Radius is squared Mahalanobis length divided by 200; covariance is compounded over the actual horizon.",
                    "The daily law uses exact 5, 10, 15 and 20-session steps and includes spot-loading variance.",
                ],
            )

        if sequence_models is not None:
            sequence_rows = []
            for fit in sequence_models["fits"]:
                baseline = fit["baseline_validation_nll_per_dimension"]
                fitted = fit["best_validation_nll_per_dimension"]
                sequence_rows.append(
                    [
                        fit["architecture"],
                        str(fit["parameter_count"]),
                        str(fit["completed_epochs"]),
                        str(fit["best_epoch"]),
                        f"{baseline:.4f}",
                        f"{fitted:.4f}",
                        f"{baseline - fitted:+.4f}",
                    ]
                )
            _table_page(
                document,
                title="Conditional normal-innovation sequence models",
                columns=["Architecture", "Parameters", "Epochs", "Best epoch", "AR val NLL", "Model val NLL", "Improvement"],
                rows=sequence_rows,
                notes=[
                    "LSTM and selective diagonal SSM (Mamba-style) use pre-2025 sequences and chronological validation.",
                    "NLL is per state coordinate per session, omitting the common normal constant; positive improvement is better.",
                    "These networks do not themselves create heavy conditional tails.",
                ],
            )

        spot_quantiles = [
            ("Weekly Normal", one_step["simulated_spot_log_return_quantiles"])
        ]
        if weekly_tail_one_step is not None:
            spot_quantiles.append(
                ("Weekly Tail", weekly_tail_one_step["spot_return_quantiles"])
            )
        if daily_one_step is not None:
            spot_quantiles.append(
                ("Daily Normal", daily_one_step["spot_return_quantiles"])
            )
        if daily_tail_one_step is not None:
            spot_quantiles.append(
                ("Daily Tail", daily_tail_one_step["spot_return_quantiles"])
            )
        spot_rows = []
        for quantile in ("0.01", "0.05", "0.5", "0.95", "0.99"):
            spot_rows.append(
                [
                    quantile,
                    f"{before['spot_log_return_quantiles'][quantile]:.4f}",
                    f"{after['spot_log_return_quantiles'][quantile]:.4f}",
                    *(
                        f"{values[quantile]:.4f}"
                        for _, values in spot_quantiles
                    ),
                ]
            )
        _table_page(
            document,
            title="Spot-return tail quantiles",
            columns=["Quantile", "Historical <2025", "Historical 2025", *(label for label, _ in spot_quantiles)],
            rows=spot_rows,
            notes=[
                "Returns are next-session forward-adjusted log spot returns.",
                "Model columns are 2024-12-31 conditional draws; historical columns pool all transitions in each period.",
                "The model spot marginal is generated from the decoded one-day front SANOS law.",
            ],
        )

        for period, title in (
            ("pre_evaluation", "Pre-2025 next-session ATM-IV forecast scores"),
            ("forward_evaluation", "Forward-2025 next-session ATM-IV forecast scores"),
        ):
            score_rows = _predictive_rows(
                predictive_reports, period=period, dtes=dtes
            )
            if score_rows:
                _table_page(
                    document,
                    title=title,
                    columns=["Model", "Tenor", "Bias", "Var ratio", "CRPS", "Cov 50%", "Cov 80%", "Cov 95%"],
                    rows=score_rows,
                    notes=[
                        "Target is next-session log forward-ATM IV. Bias = observed minus predictive mean.",
                        "Variance ratio = realized innovation variance / average predictive path variance.",
                        "Coverage uses central predictive intervals; draw counts: "
                        + ", ".join(
                            f"{label}={report['paths_per_origin']}"
                            for label, report in predictive_reports
                        )
                        + ".",
                        (
                            "Forward rows use 2025 origins with frozen decoder/dynamics; only same-day latent states are recalibrated."
                            if period == "forward_evaluation" else
                            "Pre-2025 rows are in-sample dynamics diagnostics, not forward-2025 scores."
                        ),
                        "Tail forecasts use a stationary one-step scale approximation at each origin.",
                    ],
                    font_size=8.0,
                )

        _table_page(
            document,
            title="Forward prediction and remaining cells — status",
            columns=["Requested diagnostic", "Weekly / Normal", "Other three AR(1) cells"],
            rows=[
                ["2025 decoded state calibration", "pending GPU run" if forward is None else "complete", "complete" if daily_forward is not None else "pending"],
                ["ATM-IV innovation bias", "reported" if weekly_predictive_report is not None and forward is not None else "pending", "reported" if daily_predictive_report is not None and daily_forward is not None else "pending"],
                ["Predictive variance ratio", "reported" if weekly_predictive_report is not None and forward is not None else "pending", "reported" if daily_predictive_report is not None and daily_forward is not None else "pending"],
                ["CRPS and interval coverage", "reported" if weekly_predictive_report is not None and forward is not None else "pending", "reported" if daily_predictive_report is not None and daily_forward is not None else "pending"],
                ["2025 front-law PIT", "reported" if weekly_front_pit is not None else "pending", "reported" if daily_front_pit is not None else "pending"],
                ["Model path autocorrelation", "reported" if path_models else "pending multi-step paths", "reported" if path_models else "pending"],
                ["Mamba-style SSM / LSTM", "reported" if sequence_models is not None else "after four AR(1) cells", "reported" if sequence_models is not None else "after four AR(1) cells"],
            ],
            notes=[
                "This PDF is an interim audit. Blank results are not inferred from training fit or a terminal-state simulation.",
                "It will be regenerated as the remaining fits and genuinely forward evaluations finish.",
            ],
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--training-fit-report", type=Path, required=True)
    parser.add_argument("--radius-cap-report", type=Path, required=True)
    parser.add_argument("--one-step-report", type=Path, required=True)
    parser.add_argument("--forward-fit-report", type=Path)
    parser.add_argument("--daily-forward-fit-report", type=Path)
    parser.add_argument("--weekly-tail-fit-report", type=Path)
    parser.add_argument("--daily-normal-fit-report", type=Path)
    parser.add_argument("--daily-tail-fit-report", type=Path)
    parser.add_argument("--daily-normal-one-step-report", type=Path)
    parser.add_argument("--weekly-tail-one-step-report", type=Path)
    parser.add_argument("--daily-tail-one-step-report", type=Path)
    parser.add_argument("--weekly-predictive-report", type=Path)
    parser.add_argument("--weekly-tail-predictive-report", type=Path)
    parser.add_argument("--daily-predictive-report", type=Path)
    parser.add_argument("--daily-tail-predictive-report", type=Path)
    parser.add_argument("--weekly-front-pit-report", type=Path)
    parser.add_argument("--daily-front-pit-report", type=Path)
    parser.add_argument("--daily-radius-cap-report", type=Path)
    parser.add_argument("--conditional-sequence-report", type=Path)
    parser.add_argument("--multistep-report", type=Path, action="append", dest="multistep_reports")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = run(**vars(arguments))
    print(f"DYSANOS factorial PDF: {output}", flush=True)


if __name__ == "__main__":
    main()
