"""Build the fixed-grid analytical-versus-learned-Q comparison PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path(__file__).with_name("FIXED_GRID_RISK_NEUTRAL_COMPARISON.pdf")

NAVY = "#172A46"
BLUE = "#2D6CDF"
TEAL = "#159A9C"
GREEN = "#2E8B57"
ORANGE = "#D97706"
RED = "#C43D3D"
INK = "#1F2937"
MUTED = "#5F6B7A"
PALE = "#F3F6FA"
GRID = "#D8E0EA"
WHITE = "#FFFFFF"


def page(title: str, section: str) -> tuple[plt.Figure, plt.Axes]:
    """Create one A4 page with a consistent title band."""

    figure = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")
    axes.add_patch(
        FancyBboxPatch(
            (0, 0.905),
            1,
            0.095,
            boxstyle="square,pad=0",
            facecolor=NAVY,
            edgecolor=NAVY,
        )
    )
    axes.text(0.065, 0.973, section.upper(), color="#9EC5FF", fontsize=8.5, weight="bold")
    axes.text(0.065, 0.932, title, color=WHITE, fontsize=18, weight="bold")
    return figure, axes


def footer(axes: plt.Axes, number: int) -> None:
    """Add report date and page number."""

    axes.plot((0.065, 0.935), (0.047, 0.047), color=GRID, linewidth=0.8)
    axes.text(
        0.065,
        0.026,
        "Fixed-grid risk-neutral experiments | 14 Sep 2026",
        fontsize=7.5,
        color=MUTED,
    )
    axes.text(0.935, 0.026, str(number), fontsize=7.5, color=MUTED, ha="right")


def paragraph(
    axes: plt.Axes,
    text: str,
    y: float,
    *,
    width: int = 92,
    fontsize: float = 9.2,
    color: str = INK,
    weight: str = "normal",
    line_height: float = 0.0205,
) -> float:
    """Draw a wrapped paragraph and return the next vertical position."""

    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    axes.text(
        0.075,
        y,
        "\n".join(lines),
        va="top",
        fontsize=fontsize,
        color=color,
        weight=weight,
        linespacing=1.33,
    )
    return y - line_height * len(lines)


def bullet(axes: plt.Axes, text: str, y: float, *, width: int = 87) -> float:
    """Draw a wrapped bullet and return the next vertical position."""

    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    axes.text(0.078, y, "•", va="top", fontsize=12, color=BLUE, weight="bold")
    axes.text(
        0.102,
        y,
        "\n".join(lines),
        va="top",
        fontsize=9.1,
        color=INK,
        linespacing=1.33,
    )
    return y - 0.0205 * len(lines) - 0.007


def subhead(axes: plt.Axes, text: str, y: float) -> float:
    """Draw an in-page heading."""

    axes.text(0.075, y, text, va="top", fontsize=11, weight="bold", color=NAVY)
    return y - 0.033


def formula(
    axes: plt.Axes,
    expression: str,
    y: float,
    *,
    color: str = NAVY,
    fontsize: float = 11.5,
) -> float:
    """Draw a highlighted mathematical identity."""

    axes.add_patch(
        FancyBboxPatch(
            (0.085, y - 0.055),
            0.83,
            0.065,
            boxstyle="round,pad=0.009,rounding_size=0.008",
            facecolor=PALE,
            edgecolor=GRID,
            linewidth=0.8,
        )
    )
    axes.text(0.5, y - 0.022, expression, ha="center", va="center", fontsize=fontsize, color=color)
    return y - 0.078


def styled_table(
    axes: plt.Axes,
    columns: list[str],
    rows: list[list[str]],
    bbox: tuple[float, float, float, float],
    *,
    fontsize: float = 8.2,
    col_widths: list[float] | None = None,
) -> None:
    """Draw a table in the report style."""

    table = axes.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        bbox=bbox,
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else PALE)
            if column == 0:
                cell.get_text().set_ha("left")


def summary_cards(axes: plt.Axes) -> None:
    """Draw the two method summary cards."""

    cards = [
        (
            0.075,
            "ANALYTICAL DLVsq DRIFT",
            "A transition simulator",
            "Ten-substep stochastic Heun\nTheoretical infinitesimal drift\nNo joint rejection: p = 0.2746",
            BLUE,
            "#EAF2FF",
        ),
        (
            0.52,
            "MARGINAL-UTILITY Q4",
            "Weights on one fixed path set",
            "Four sequential utility tilts\nRestricted frozen-feature adversary\nNo joint rejection: p = 1.0000",
            TEAL,
            "#EAF7F6",
        ),
    ]
    for x, heading, title, body, color, fill in cards:
        axes.add_patch(
            FancyBboxPatch(
                (x, 0.555),
                0.405,
                0.205,
                boxstyle="round,pad=0.016,rounding_size=0.012",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        axes.text(x + 0.024, 0.725, heading, fontsize=8.2, color=color, weight="bold")
        axes.text(x + 0.024, 0.685, title, fontsize=11.5, color=NAVY, weight="bold")
        axes.text(x + 0.024, 0.645, body, fontsize=9.0, color=INK, va="top", linespacing=1.55)


def comparison_table(axes: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    """Draw the main comparable-results table."""

    styled_table(
        axes,
        ["Primary one-step audit", "Analytical drift", "Empirical Q4"],
        [
            ["Option coordinates", "2,000", "2,000"],
            ["max |t|", "3.334", "2.047"],
            ["RMS t", "1.109", "0.441"],
            ["combined p", "0.2746", "1.0000"],
            ["5% decision", "do not reject", "do not reject"],
            ["weekly-spot combined p", "0.0876", "1.0000"],
            ["effective sample size", "100,000", "69,272"],
            ["object produced", "transition law", "fixed-sample weights"],
        ],
        bbox,
        fontsize=8.5,
        col_widths=[0.43, 0.285, 0.285],
    )


def build() -> None:
    """Render the eight-page comparison report."""

    with PdfPages(
        OUTPUT,
        metadata={
            "Title": "Fixed-grid risk-neutral experiments",
            "Author": "Quant Data research",
            "Subject": "Analytical DLVsq drift versus marginal-utility measure change",
        },
    ) as pdf:
        figure, axes = page("Fixed-grid risk-neutral experiments", "Research comparison")
        axes.text(
            0.075,
            0.858,
            "ANALYTICAL DLVsq-DRIFT MONTE CARLO VERSUS POWER-EXPONENTIAL MEASURE CHANGE",
            fontsize=8.7,
            color=BLUE,
            weight="bold",
        )
        axes.text(
            0.075,
            0.812,
            "Two ways to remove statistical arbitrage on a four-week fixed grid",
            fontsize=14.0,
            color=NAVY,
            weight="bold",
        )
        summary_cards(axes)
        y = subhead(axes, "Bottom line", 0.505)
        y = paragraph(
            axes,
            "Both experiments pass the same dependence-aware zero-drift test for all 2,000 one-step option returns. The analytical construction has the stronger model interpretation: its drift is fixed before the Monte Carlo draw and it produces a simulator. Q4 gives smaller fitted-sample t-statistics, but it is an empirical reweighting of one path set and only defeats the retained frozen-feature strategy class.",
            y,
            width=89,
            fontsize=9.4,
            weight="bold",
        )
        y -= 0.018
        y = subhead(axes, "Most important qualification", y)
        y = paragraph(
            axes,
            "The project often called the second run an OCE measure change. The implemented objective was direct power-exponential utility followed by normalized marginal-utility weights. No OCE cash translation y was optimized. This report uses the more exact name throughout.",
            y,
            width=89,
        )
        axes.text(
            0.075,
            0.083,
            "Scope: X-only, state 44, 200 interior strikes, no re-gridding, 100,000 paths per experiment.",
            fontsize=8.3,
            color=MUTED,
        )
        footer(axes, 1)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Common experiment and comparability", "Design")
        y = 0.862
        y = paragraph(
            axes,
            "Both experiments start from SANOS state 44 (8 March 2022) and use an X-only shrinking tower with four weekly trade dates. Cash strikes remain fixed within every weekly transition; no re-gridding is performed.",
            y,
            weight="bold",
        )
        y = subhead(axes, "Matched dimensions", y - 0.012)
        for text in (
            "200 interior strike nodes and four starting expiries; the live one-step return counts are 800, 600, 400, and 200, for 2,000 option moments in total.",
            "100,000 paths, arranged as 50,000 antithetic pairs. Fifty pairs form one randomized-stratification block, giving 1,000 independent block observations.",
            "Rank-25 fixed-expiry DLV innovations estimated from 703 historical weekly observations through 31 December 2024.",
            "All final reported martingale tests use the raw, original spot and option returns—not the beta-adjusted or whitened factors used to condition the optimizer.",
        ):
            y = bullet(axes, text, y)
        y = subhead(axes, "One-step return being tested", y - 0.006)
        y = formula(axes, r"$R_t^{j,i}=P_{t+1}^{j,i}/P_t^{j,i}-1$", y)
        y = paragraph(
            axes,
            "P_t^{j,i} and P_{t+1}^{j,i} are prices of the same fixed-cash-strike, fixed-expiry claim. At the front expiry, the endpoint value is its payoff. An ITM call or put is selected pathwise to avoid division by a near-zero OTM price.",
            y,
        )
        y = subhead(axes, "What is not matched", y - 0.006)
        y = bullet(axes, "The two reports come from independently generated random samples, not common random numbers.", y)
        y = bullet(axes, "Their dimensions, state, covariance estimator, and inference protocol match; their dynamics and probability measures intentionally differ.", y)
        footer(axes, 2)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Analytical fixed-grid Monte Carlo", "Method 1")
        y = 0.86
        y = paragraph(
            axes,
            "Let v be DLVsq and z = log(v − v_min). The rank-25 loading matrix supplies the diffusion of z. The drift is computed from the theoretical fixed-contract condition that every option price has zero instantaneous drift, including the Itô correction in log excess-DLV coordinates.",
            y,
        )
        y = subhead(axes, "Numerical transition", y - 0.01)
        for text in (
            "Integrate each one-week transition with ten stochastic-Heun substeps in 64-bit precision.",
            "Use the same Gaussian increment in the predictor and corrector because the diffusion is additive in z.",
            "Draw spot from the front density, remove the expired slab, and propagate surviving conditional densities on the unchanged cash grid.",
            "This solves the infinitesimal drift condition numerically; it is not an exact finite-one-week transition formula.",
        ):
            y = bullet(axes, text, y)
        y = subhead(axes, "Audits run", y - 0.004)
        styled_table(
            axes,
            ["Audit family", "Moments", "Purpose"],
            [
                ["one-step ITM returns", "2,000", "primary comparison"],
                ["terminal ITM returns", "2,000", "hold claims to expiry"],
                ["terminal price changes", "2,000", "unscaled claim drift"],
                ["integrated DLV control", "1,200 active", "pre-spot drift check"],
                ["weekly spot returns", "4", "spot martingale"],
            ],
            (0.075, 0.18, 0.85, 0.235),
            fontsize=8.3,
            col_widths=[0.39, 0.20, 0.41],
        )
        axes.text(
            0.075,
            0.128,
            "800 further integrated-control coordinates had variance below numerical resolution and were not t-tested.",
            fontsize=8.2,
            color=MUTED,
        )
        footer(axes, 3)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Analytical Monte Carlo results", "Method 1 results")
        styled_table(
            axes,
            ["Audit", "Active", "max |t|", "RMS t", "combined p", "5% decision"],
            [
                ["one-step option returns", "2,000", "3.334", "1.109", "0.2746", "do not reject"],
                ["terminal option returns", "2,000", "3.334", "1.044", "0.3210", "do not reject"],
                ["terminal price changes", "2,000", "2.934", "1.156", "0.3500", "do not reject"],
                ["integrated DLV control", "1,200", "1.950", "0.878", "1.0000", "do not reject"],
                ["weekly spot returns", "4", "2.431", "1.561", "0.0876", "do not reject"],
            ],
            (0.055, 0.625, 0.89, 0.245),
            fontsize=7.9,
            col_widths=[0.29, 0.12, 0.13, 0.13, 0.15, 0.18],
        )
        y = subhead(axes, "Primary one-step option test", 0.57)
        y = bullet(axes, "Shared-sign bootstrap: max-t p = 0.1373; RMS-t p = 0.2465; combined p = 0.2746.", y)
        y = bullet(axes, "Bonferroni familywise p = 1.000; simultaneous 5% max-t cutoff = 4.235.", y)
        y = bullet(axes, "The worst coordinate |t| = 3.334 is therefore not a familywise rejection across 2,000 correlated returns.", y)
        y = subhead(axes, "Numerical and speed diagnostics", y - 0.005)
        y = bullet(axes, "Maximum density-mass residual 9.33e−14; maximum conditional-forward residual 8.93e−14; minimum density exactly zero.", y)
        y = bullet(axes, "72.07 seconds for 100,000 paths, or 1,387.6 paths per second.", y)
        y = paragraph(
            axes,
            "Interpretation: this is positive Monte Carlo evidence that the analytical drift plus ten-step integration is sufficiently accurate in this experiment. It does not convert the infinitesimal SDE result into an exact finite-step proof.",
            y - 0.008,
            weight="bold",
        )
        footer(axes, 4)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Marginal-utility measure change", "Method 2")
        y = 0.862
        y = paragraph(
            axes,
            "The physical path law uses the same rank-25 DLV innovation covariance but sets the raw log-DLV drift to zero. It contains a strong statistical-arbitrage signal: the nearly uniform initialization had worst option |t| about 80.63, RMS t about 16.77, and the minimum resolvable combined p = 0.00797 in its 250-bootstrap diagnostic.",
            y,
        )
        y = subhead(axes, "Utility and positive empirical density", y - 0.008)
        y = formula(
            axes,
            r"$u(\pi)=1-e^{-\pi}\;(\pi>0),\quad u(\pi)=\pi-\pi^2/2\;(\pi\leq0)$",
            y,
            fontsize=10.7,
        )
        y = formula(
            axes,
            r"$w_n^{\mathrm{new}}=\dfrac{w_n^{\mathrm{base}}u'(\pi_n)}{N^{-1}\sum_m w_m^{\mathrm{base}}u'(\pi_m)}$",
            y,
            color=TEAL,
        )
        y = paragraph(
            axes,
            "Four sequential adversarial root corrections produce Q4. Earlier weights are detached, so a later adversary cannot alter the measure it is challenging.",
            y,
        )
        y = subhead(axes, "Conditioning used only during optimization", y - 0.006)
        for text in (
            "Precompute beta under P and use U_t = R_t − beta_t S_t; do not regress out the realized spot return dynamically.",
            "Diagonalize the unconditional residual covariance, lift eigenvalues below lambda_max/1000 using fixed-seed antithetic Gaussian noise, and whiten; retain standardized spot separately.",
            "Freeze two nonlinear density-feature layers of width four. Train only 10,020 final-layer coefficients out of 18,196 total parameters with full-path L-BFGS.",
            "Use the identical feature seed and policy class for construction and the fresh adversarial certificate. Final martingale tests return to the 2,000 original noise-free returns.",
        ):
            y = bullet(axes, text, y)
        footer(axes, 5)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Empirical Q4 results", "Method 2 results")
        styled_table(
            axes,
            ["Stage", "Measure challenged", "max utility", "resulting ESS"],
            [
                ["1", "uniform P", "0.204156", "69,290"],
                ["2", "Q1", "9.65e−5", "69,283"],
                ["3", "Q2", "1.92e−5", "69,263"],
                ["4", "Q3", "7.55e−6", "69,272 (Q4)"],
                ["5 certificate", "Q4", "3.91e−6", "Q5 not certified"],
            ],
            (0.075, 0.655, 0.85, 0.215),
            fontsize=8.2,
            col_widths=[0.20, 0.29, 0.25, 0.26],
        )
        loss_axes = figure.add_axes((0.105, 0.405, 0.79, 0.18))
        stages = np.arange(1, 6)
        utilities = np.array([0.204156, 9.65046e-5, 1.91819e-5, 7.55159e-6, 3.91060e-6])
        loss_axes.semilogy(stages, utilities, marker="o", color=TEAL, linewidth=2.0, markersize=5)
        loss_axes.set(xticks=stages, xlabel="adversarial stage", ylabel="maximum utility", title="Same-class utility remaining")
        loss_axes.grid(color=GRID, linewidth=0.6, which="both")
        loss_axes.spines[["top", "right"]].set_visible(False)
        loss_axes.tick_params(labelsize=7.5)
        loss_axes.title.set_fontsize(9.5)
        loss_axes.xaxis.label.set_fontsize(8)
        loss_axes.yaxis.label.set_fontsize(8)
        y = subhead(axes, "Raw Q4 martingale audit", 0.35)
        y = bullet(axes, "All 2,000 one-step option returns: max |t| = 2.047, RMS t = 0.441, max-t p = 0.9640, RMS-t p = 0.9998, combined p = 1.0000.", y)
        y = bullet(axes, "Four weekly spot returns: max |t| = 8.02e−7 and combined p = 1.0000.", y)
        y = bullet(axes, "Q4 effective sample size is 69,272, or 69.3% of the original path count.", y)
        y = bullet(axes, "Zero-strategy objective-gradient RMS fell from 5.38e−3 under P to 1.69e−6 under Q4: a ratio of 3.15e−4.", y)
        y = bullet(axes, "The frozen Stage-5 strategy has mean-P&L t = 0.776 and utility t = 0.387 on 1,000 blocks; these are descriptive because strategy selection used the same paths.", y)
        footer(axes, 6)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Comparable results and interpretation", "Head-to-head")
        comparison_table(axes, (0.075, 0.545, 0.85, 0.325))
        y = subhead(axes, "Why Q4's smaller t-statistics do not make it the stronger model", 0.495)
        for text in (
            "Q4 was chosen against the same finite sample that is audited. This is intentional for the fixed-path root problem, but its p-values do not have the same pre-specified-model interpretation as the analytical run.",
            "Q4 is a positive density over these 100,000 paths; it is not a reusable conditional transition kernel and has not been fitted across all 891 historical states.",
            "The analytical drift produces new simulations. Its larger z-scores are assessed on random paths generated after the model law has already been specified.",
            "The analytical run additionally passes terminal-return and terminal-price-change audits; no corresponding terminal learned-Q result is claimed.",
        ):
            y = bullet(axes, text, y)
        y = subhead(axes, "Indicative cost", y - 0.005)
        y = paragraph(
            axes,
            "The analytical run took 72.1 seconds. Physical path generation for Q took about 4.5 seconds, but four root-correction solves took about 222 seconds and the Stage-5 certificate another 56 seconds. This is not a controlled speed benchmark because the staged processes also regenerated and re-audited deterministic paths.",
            y,
        )
        footer(axes, 7)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Statistical test and final assessment", "Inference and limits")
        y = 0.865
        y = paragraph(
            axes,
            "For each coordinate k, X_bk is the mean contribution of the 100 paths in independent block b. With B = 1,000 blocks, the coordinate statistic, max statistic, and RMS statistic are:",
            y,
        )
        y = formula(
            axes,
            r"$t_k=\sqrt{B}\,\bar X_k/s_k,\quad T_{\max}=\max_k|t_k|,\quad T_{\mathrm{RMS}}=\sqrt{d^{-1}\sum_k t_k^2}$",
            y,
            fontsize=10.5,
        )
        y = paragraph(
            axes,
            "A shared-sign Rademacher multiplier bootstrap is applied to centered block vectors, preserving cross-option dependence. With p_max and p_RMS from 10,000 replications, the decision value is p_combined = min(1, 2 min(p_max, p_RMS)). Reject zero drift at 5% only when p_combined < 0.05. A high p means compatibility with Monte Carlo noise—not proof of exact zero population drift.",
            y,
        )
        axes.add_patch(FancyBboxPatch((0.075, 0.465), 0.405, 0.11, boxstyle="round,pad=0.014", facecolor="#EAF7EF", edgecolor="#A9D9BB"))
        axes.text(0.095, 0.54, "ANALYTICAL DLVsq", fontsize=8.4, color=GREEN, weight="bold")
        axes.text(0.095, 0.505, "Credible approximate simulator", fontsize=10.4, color=NAVY, weight="bold")
        axes.text(0.095, 0.48, "No 5% rejection in this experiment", fontsize=8.6, color=MUTED)
        axes.add_patch(FancyBboxPatch((0.52, 0.465), 0.405, 0.11, boxstyle="round,pad=0.014", facecolor="#FFF4E5", edgecolor="#F1C88A"))
        axes.text(0.54, 0.54, "EMPIRICAL Q4", fontsize=8.4, color=ORANGE, weight="bold")
        axes.text(0.54, 0.505, "Restricted fixed-sample root", fontsize=10.4, color=NAVY, weight="bold")
        axes.text(0.54, 0.48, "Not a general no-arbitrage model", fontsize=8.6, color=MUTED)
        y = subhead(axes, "Control that failed", 0.415)
        y = paragraph(
            axes,
            "When both hidden layers were learned at width 16, fresh same-architecture adversaries continued to find utilities 0.5714 and 0.5922 after successive measure changes. Thus the favorable Q4 result depends materially on the frozen width-four feature class; the broader learned conditional problem remains open.",
            y,
            weight="bold",
        )
        y = subhead(axes, "Reproducibility", y - 0.008)
        y = bullet(axes, "Analytical report: dlv-analytic-baselines/shrinking-tower-state0044-100k-v2/report.json.", y)
        y = bullet(axes, "Q4 and certificate: direct-utility/empirical-density-state0044-fixed-features-w4-v1/stage4-certificate.pt and stage5-certificate.json.", y)
        y = bullet(axes, "Focused regression suite: 88 tests passed; compilation, import-order, shell-syntax, and Git-diff checks passed. Research implementation commit: 7b2aad7.", y)
        axes.text(
            0.075,
            0.078,
            "Separate result: the density-first finite-support construction is exact; see DLV_REGRID_MARTINGALE_SUMMARY.pdf.",
            fontsize=8.1,
            color=MUTED,
        )
        footer(axes, 8)
        pdf.savefig(figure)
        plt.close(figure)


if __name__ == "__main__":
    build()
