"""Build the committed DLV martingale research-summary PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path(__file__).with_name("DLV_REGRID_MARTINGALE_SUMMARY.pdf")

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
    axes.text(0.065, 0.026, "DLV martingale simulation | revised 11 Sep 2026", fontsize=7.5, color=MUTED)
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
    fontsize: float = 12,
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
    axes.text(
        0.5,
        y - 0.022,
        expression,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
    )
    return y - 0.078


def draw_process(axes: plt.Axes) -> None:
    """Draw the one-week density-first transition."""

    items = [
        (0.08, "Current surface", "q and DLVsq"),
        (0.27, "Density branches", "q_m^b\nbranch marginal"),
        (0.46, "Draw path", "b ~ π; i ~ q_1\nindependently"),
        (0.65, "Condition", "r_ij^(b,m)\npath density"),
        (0.84, "Re-grid", "x_j / x_i\nrelative support"),
    ]
    for index, (x, title, body) in enumerate(items):
        axes.add_patch(
            FancyBboxPatch(
                (x - 0.072, 0.535),
                0.145,
                0.105,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=WHITE if index % 2 == 0 else PALE,
                edgecolor=BLUE if index < 4 else TEAL,
                linewidth=1.4,
            )
        )
        axes.text(x, 0.603, title, ha="center", fontsize=8.5, color=NAVY, weight="bold")
        axes.text(x, 0.566, body, ha="center", fontsize=6.9, color=MUTED, linespacing=1.25)
        if index < len(items) - 1:
            axes.annotate(
                "",
                xy=(items[index + 1][0] - 0.082, 0.586),
                xytext=(x + 0.082, 0.586),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.2},
            )


def result_table(axes: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    """Draw the primary four-week audit table."""

    labels = ["Week", "worst |z|", "max mean\n(bp spot)", "J", "df", "p"]
    values = [
        ["1", "1.796", "0.497", "10.980", "7", "0.139"],
        ["2", "2.143", "0.863", "7.228", "7", "0.406"],
        ["3", "1.371", "0.696", "5.858", "6", "0.439"],
        ["4", "2.057", "1.245", "5.506", "5", "0.357"],
    ]
    table = axes.table(cellText=values, colLabels=labels, cellLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else PALE)


def matched_table(axes: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    """Draw the matched re-grid diagnostic table."""

    labels = ["Week", "fixed-expiry\np", "direct = re-grid\nfirst-member p", "full pair\np", "max price diff"]
    values = [
        ["1", "0.366", "0.684", "0.139", "0.0"],
        ["2", "0.538", "0.163", "0.128", "2.22e-16"],
        ["3", "0.911", "0.414", "0.0208", "2.22e-16"],
        ["4", "0.945", "0.0808", "0.546", "2.22e-16"],
    ]
    table = axes.table(cellText=values, colLabels=labels, cellLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else PALE)


def build() -> None:
    """Render the eight-page summary."""

    with PdfPages(
        OUTPUT,
        metadata={
            "Title": "X-only finite-grid DLV martingale simulation and weekly re-gridding",
            "Author": "Quant Data research",
            "Subject": "Construction, proof, numerical audits, and limitations",
        },
    ) as pdf:
        figure, axes = page("X-only finite-grid DLV martingale simulation", "Research summary")
        axes.text(0.075, 0.865, "WEEKLY RE-GRIDDING, EXACT TOWERS, AND MONTE CARLO EVIDENCE", fontsize=9, color=BLUE, weight="bold")
        axes.text(0.075, 0.814, "A density-first risk-neutral law for the modeled finite option surface", fontsize=14.5, color=NAVY, weight="bold")
        axes.text(0.075, 0.775, "Y ≡ 1  •  all 891 certified states  •  four 1W steps  •  64-bit floats  •  antithetic paths", fontsize=9.5, color=MUTED)
        axes.add_patch(FancyBboxPatch((0.075, 0.56), 0.85, 0.16, boxstyle="round,pad=0.018,rounding_size=0.012", facecolor="#EAF2FF", edgecolor="#B8CFF5"))
        axes.text(0.10, 0.682, "BOTTOM LINE", fontsize=8.5, color=BLUE, weight="bold")
        axes.text(0.10, 0.640, "The finite-grid density tower is an exact martingale construction.", fontsize=13.5, color=NAVY, weight="bold")
        axes.text(0.10, 0.602, "Re-gridding is an exact coordinate change for surviving fixed-cash contracts.", fontsize=11, color=INK)
        axes.text(0.10, 0.572, "The unresolved modeling issue is absorbing tail support—not learned drift.", fontsize=11, color=INK)
        y = 0.50
        y = subhead(axes, "High-power result", y)
        y = paragraph(axes, "A 100,000-path X-only four-week audit found covariance-aware p-values of 0.139, 0.406, 0.439, and 0.357 for the surviving original fixed-cash book plus spot. No original-contract horizon rejected zero drift at 5%.", y, width=88)
        y -= 0.018
        y = subhead(axes, "Exact numerical identities", y)
        y = bullet(axes, "All-state option-tower residual: 1.67e-16; density-tower residual: 6.49e-15.", y)
        y = bullet(axes, "Direct cash-grid and re-gridded survivor prices agree pathwise to at most 2.22e-16.", y)
        y = bullet(axes, "Four weekly rolls remain stable; boundary incidence reaches only 0.006% at week 4.", y)
        axes.text(0.075, 0.082, "Scope: pure discrete DLV factor X. The independent Black factor is disabled exactly at the pricing and path layer.", fontsize=8.3, color=MUTED)
        footer(axes, 1)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Indices and probability law", "Notation")
        axes.text(
            0.075,
            0.858,
            "The branch marginal and the pathwise conditional density are different objects.",
            fontsize=11.5,
            color=NAVY,
            weight="bold",
        )
        index_items = [
            (0.075, "m", "expiry", "m = 1,…,M; T_m = t + m h"),
            (0.295, "i", "front node", "observed at T_1; S_(t+h) = s x_i"),
            (0.515, "j", "later node", "possible node at surviving T_m"),
            (0.735, "b", "branch", "one of 2^F surface-shock signs"),
        ]
        for x, symbol, title, body in index_items:
            axes.add_patch(
                FancyBboxPatch(
                    (x, 0.715),
                    0.19,
                    0.105,
                    boxstyle="round,pad=0.012,rounding_size=0.010",
                    facecolor=PALE,
                    edgecolor=GRID,
                )
            )
            axes.text(x + 0.022, 0.782, symbol, fontsize=16, color=BLUE, weight="bold")
            axes.text(x + 0.058, 0.785, title, fontsize=9.4, color=NAVY, weight="bold")
            axes.text(x + 0.022, 0.738, body, fontsize=7.5, color=MUTED, wrap=True)
        y = subhead(axes, "State and branch symbols", 0.675)
        y = bullet(axes, "All formulas condition on one fixed current state; its state/batch index is suppressed. P_t and E_t mean probability and expectation given that state's current information.", y)
        y = bullet(axes, "t is current calendar time; h is one week; M = 4. S_t = s is the X-only spot. Y ≡ 1, so there is no independent Black spot factor.", y)
        y = bullet(axes, "There are N_x ordered support multipliers x_j. Node j represents the cash value s x_j. The indices i and j range from 0 to N_x−1 but play the distinct roles shown above.", y)
        y = bullet(axes, "q_m(j) is the time-t risk-neutral probability mass at node j for expiry T_m. It has unit mass and unit mean: Σ_j q_m(j)=1 and Σ_j q_m(j)x_j=1.", y)
        y = bullet(axes, "DLV means discrete local volatility; DLVsq is its square. F = 4 is the number of DLVsq shock factors; ε_b is branch b's vector of ±1 Rademacher signs; Γ_m maps those signs into expiry-m probability-mass shocks.", y)
        y = bullet(axes, "π_b is the probability of branch b—not the circle constant. All 2^F = 16 branches are equally weighted here, so π_b = 2^(−F) = 1/16 and Σ_b π_b = 1.", y)
        y = bullet(axes, "q_m^b(j) is the branch-conditioned marginal before front node i is observed. r_ij^(b,m) is the probability of later node j conditional on both branch b and observed front node i; that row is the actual pathwise survivor density.", y)
        y = formula(axes, r"$P_t(B=b,I=i)=\pi_b q_1(i),\qquad q_1^b=q_1\;(\Gamma_1=0)$", y - 0.002)
        y = paragraph(
            axes,
            "The implementation draws the branch B and front-node index I independently given current information. Conditioning and re-gridding then make the rolled surface depend on the realized spot. This exact law does not model contemporaneous correlation between the raw branch shock and the front-node draw.",
            y - 0.004,
            width=90,
            fontsize=8.8,
        )
        footer(axes, 2)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("One-week construction", "Method")
        draw_process(axes)
        y = 0.855
        y = paragraph(axes, "The method steps in density space, where the martingale restriction is linear, and only then recovers DLVsq coordinates. This replaces an approximate Euler drift calculation with an exact finite-step density mean.", y)
        y = formula(axes, r"$q_m^b(j)=q_m(j)+\sqrt{h}\,[\Gamma_m\varepsilon_b]_j,\qquad \sum_b\pi_b q_m^b(j)=q_m(j)$", y - 0.015)
        y = subhead(axes, "Implementation sequence", 0.495)
        for text in (
            "Derive density diffusion from the fixed-contract DLVsq loadings and the discrete SANOS operator.",
            "Enumerate symmetric Rademacher branches; reject a step if a branch is materially negative.",
            "Recover branch DLVsq algebraically from adjacent-expiry call values, then verify the decoder reproduces the branch density.",
            "Draw branch b with probability π_b and, independently, front node i with probability q_1(i); propagate the point mass at i to obtain r_ij^(b,m).",
            "Re-center support around realized spot. Old 2W–4W expiries become floating 1W–3W; a separate AR boundary supplies the newly issued 4W DLVsq.",
        ):
            y = bullet(axes, text, y)
        y -= 0.004
        y = subhead(axes, "What is fitted?", y)
        y = paragraph(axes, "Only the new-4W boundary law is fitted: 25 principal-component factors and a weekly vector autoregression of order one, trained on 703 pairs and checked on 155 untouched 2025 pairs. The martingale transition itself is analytical.", y)
        footer(axes, 3)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Exact re-grid martingale identity", "Proof")
        y = 0.865
        y = paragraph(axes, "For survivor expiry m ≥ 2, r_ij^(b,m) is the later-node probability after branch b and front node i are both known. It obeys a conditional martingale identity, an i-to-branch tower, and a branch-to-current tower.", y)
        y = formula(axes, r"$\sum_jr_{ij}^{b,m}x_j=x_i,\quad\sum_iq_1(i)r_{ij}^{b,m}=q_m^b(j),\quad\sum_b\pi_bq_m^b(j)=q_m(j)$", y - 0.01)
        y = paragraph(axes, "Here S_t=s, K is a fixed cash strike, and (a)^+=max(a,0). Re-gridding divides every inherited node by the same realized front multiplier.", y - 0.005)
        y = formula(axes, r"$S_{t+h}=sx_i,\qquad x_j'{}^{(i)}=x_j/x_i,\qquad S_{t+h}x_j'{}^{(i)}=sx_j$", y - 0.01, color=TEAL)
        y = paragraph(axes, "The conditional call value is therefore identical on the cash grid and the re-gridded relative grid.", y - 0.005)
        y = formula(axes, r"$C_{t+h}^{b,i,m}(K)=\sum_jr_{ij}^{b,m}(sx_j-K)^+$", y - 0.01)
        y = paragraph(axes, "E_t means risk-neutral expectation conditional on current information. Applying both towers returns the current expiry-m call price.", y - 0.005)
        y = formula(
            axes,
            r"$E_t[C_{t+h}^{m}(K)]=\sum_b\pi_b\sum_iq_1(i)C_{t+h}^{b,i,m}(K)=\sum_jq_m(j)(sx_j-K)^+=C_t^m(K)$",
            y - 0.01,
            color=GREEN,
            fontsize=10.2,
        )
        y = subhead(axes, "Consequences", y - 0.005)
        y = bullet(axes, "The result holds for every strike and payoff represented on the finite support; puts follow by parity.", y)
        y = bullet(axes, "No survivor-density interpolation is used. Re-gridding cannot introduce drift because it changes coordinates, not cash payoffs.", y)
        y = bullet(axes, "This audit is pure X: the independent Black factor is identically one, so spot moves only when the front DLV node changes.", y)
        footer(axes, 4)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("How zero drift is tested", "Inference")
        y = 0.86
        y = paragraph(axes, "Index a=1,…,n labels independent antithetic pairs; k=1,…,d labels spot/option return coordinates. R_ak is pair a's mean return in coordinate k, R-bar is the d-vector of sample means, and sigma-hat_k is the sample standard deviation using denominator n−1.", y)
        y = formula(axes, r"$z_k=\sqrt{n}\,\bar R_k/\hat\sigma_k,\qquad J=n\,\bar R^T\hat\Sigma^+\bar R$", y - 0.01)
        y = paragraph(axes, "Sigma-hat is the d × d sample covariance of pair observations. The superscript + denotes its truncated Moore–Penrose inverse: correlation eigenmodes at or below 0.01 are discarded. If ν modes remain, the reference is J approximately chi-square(ν); p is P(chi-square(ν) ≥ observed J) under zero population means.", y - 0.005, fontsize=8.8)
        axes.add_patch(FancyBboxPatch((0.075, 0.405), 0.40, 0.105, boxstyle="round,pad=0.014", facecolor="#FDECEC", edgecolor="#F3B4B4"))
        axes.text(0.095, 0.478, "SMALL p", fontsize=9, color=RED, weight="bold")
        axes.text(0.095, 0.449, "Evidence against zero drift", fontsize=10.5, color=INK, weight="bold")
        axes.text(0.095, 0.421, "Reject per test if p < 0.05", fontsize=8.7, color=MUTED)
        axes.add_patch(FancyBboxPatch((0.525, 0.405), 0.40, 0.105, boxstyle="round,pad=0.014", facecolor="#EAF7EF", edgecolor="#A9D9BB"))
        axes.text(0.545, 0.478, "LARGE p", fontsize=9, color=GREEN, weight="bold")
        axes.text(0.545, 0.449, "Compatible with MC noise", fontsize=10.5, color=INK, weight="bold")
        axes.text(0.545, 0.421, "Not proof of exact equivalence", fontsize=8.7, color=MUTED)
        y = 0.355
        y = subhead(axes, "Decision thresholds", y)
        y = bullet(axes, "One test: 0.05. Four pre-specified weekly looks: Bonferroni 0.0125.", y)
        y = bullet(axes, "Worst |z| above two is not an automatic rejection across many correlated options.", y)
        y = subhead(axes, "Calibration caveat", y - 0.005)
        y = paragraph(axes, "The option returns are highly collinear, and the eigenmodes are selected from a sample covariance. The nominal chi-square p-value can therefore move with sample size and cutoff. Exact tower identities and matched direct/re-grid residuals carry more weight than an isolated low p-value.", y)
        footer(axes, 5)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Surviving original contracts", "Results")
        result_table(axes, (0.075, 0.60, 0.85, 0.215))
        axes.text(0.075, 0.856, "X-only • 100,000 paths • 50,000 independent antithetic pairs • all 891 states • bp = 10⁻⁴", fontsize=8.6, color=MUTED)
        plot_p = figure.add_axes((0.09, 0.35, 0.38, 0.18))
        plot_z = figure.add_axes((0.55, 0.35, 0.38, 0.18))
        weeks = np.arange(1, 5)
        p_values = np.array([0.139, 0.406, 0.439, 0.357])
        z_values = np.array([1.796, 2.143, 1.371, 2.057])
        plot_p.bar(weeks, p_values, color=[TEAL] * 4, width=0.62)
        plot_p.axhline(0.05, color=RED, linestyle="--", linewidth=1.1, label="5% cutoff")
        plot_p.set(title="Joint p-value", xlabel="week", ylim=(0, 1), xticks=weeks)
        plot_p.grid(axis="y", color=GRID, linewidth=0.6)
        plot_p.legend(frameon=False, fontsize=7, loc="upper left")
        plot_z.bar(weeks, z_values, color=[BLUE] * 4, width=0.62)
        plot_z.axhline(2.0, color=ORANGE, linestyle="--", linewidth=1.1, label="|z| = 2")
        plot_z.set(title="Worst coordinate |z|", xlabel="week", ylim=(0, 3), xticks=weeks)
        plot_z.grid(axis="y", color=GRID, linewidth=0.6)
        plot_z.legend(frameon=False, fontsize=7, loc="upper right")
        for plot in (plot_p, plot_z):
            plot.tick_params(labelsize=7.5)
            plot.title.set_fontsize(9.5)
            plot.xaxis.label.set_fontsize(8)
            plot.spines[["top", "right"]].set_visible(False)
        y = 0.292
        y = subhead(axes, "Interpretation", y)
        y = bullet(axes, "All four original-contract covariance-aware tests fail to reject at 5%; each is also clear of the 0.0125 four-look cutoff.", y)
        y = bullet(axes, "The maximum option mean is at most 1.245 basis points of initial spot.", y)
        y = bullet(axes, "Exact residuals are 1e-13 or smaller, separating numerical consistency from Monte Carlo fluctuation.", y)
        footer(axes, 6)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Fixed-support / floating-coordinate decomposition", "Diagnostics")
        axes.text(0.075, 0.856, "Matched X-only 100,000-path transition decomposition", fontsize=10.5, color=NAVY, weight="bold")
        matched_table(axes, (0.075, 0.615, 0.85, 0.205))
        y = 0.575
        y = paragraph(axes, "Direct inherited-grid and floating re-grid cash prices have identical z/J/p statistics to numerical precision. The pair column uses antithetic means; component columns use one independent member per pair.", y)
        y = subhead(axes, "Objectives and findings", y - 0.01)
        for text in (
            "Fixed-grid objective: hold cash strikes and expiries fixed during each step and enforce the DLV surface-density mean exactly.",
            "Floating-grid objective: condition on the realized X node, divide support by that node, and repeat with no change to survivor cash values.",
            "Fixed-expiry branch p-values are 0.366, 0.538, 0.911, 0.945; its maximum option means are at most 0.0056 bp of spot.",
            "Rolling-book pair p-values are 0.139, 0.128, 0.0208, 0.546. Week 3 is below 0.05 but above the four-look cutoff 0.0125.",
            "The week-3 result cannot be re-grid drift: direct and re-grid returns are identical; p rises to 0.0879/0.295 at cutoffs 0.02/0.05, and split-covariance p is 0.214/0.338.",
        ):
            y = bullet(axes, text, y)
        axes.add_patch(FancyBboxPatch((0.075, 0.105), 0.85, 0.10, boxstyle="round,pad=0.014", facecolor=PALE, edgecolor=GRID))
        axes.text(0.095, 0.169, "RESULT", fontsize=8.5, color=BLUE, weight="bold")
        axes.text(0.095, 0.146, "This is one matched process—not two independently specified models.", fontsize=9.4, color=INK, weight="bold")
        axes.text(0.095, 0.122, "Exact direct/re-grid equality rules out coordinate drift; the low p is covariance-mode sensitive.", fontsize=8.8, color=MUTED)
        footer(axes, 7)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = page("Boundary behavior and assessment", "Limitations")
        y = 0.86
        y = paragraph(axes, "At an extreme finite support node, a bounded martingale must stay there. If x_0 is the lowest node and the front draw selects it, every x_j is at least x_0.", y)
        y = formula(axes, r"$\sum_j r_{0j}^{b,m}x_j=x_0\quad\Longrightarrow\quad r_{00}^{b,m}=1$", y - 0.01, color=RED)
        y = paragraph(axes, "The conditional density becomes a point mass and survivor options become intrinsic. Interior DLVsq coordinates are then price-unidentified: the positive-density formula divides by q and ceases to apply. The implementation's fallback continuation is a numerical convention invisible to the absorbed X law.", y - 0.005)
        y = subhead(axes, "Observed in the X-only four-week run", y - 0.01)
        y = bullet(axes, "Boundary-node incidence: 0.003%, 0.003%, 0.003%, and 0.006% over weeks 1–4.", y)
        y = bullet(axes, "Median effective front nodes: 34.81 at week 1 and 38.75 at week 4.", y)
        y = bullet(axes, "At week 4, the first-percentile effective-node count is 7.44 and the 99th-percentile largest atom is 27.45%.", y)
        y = subhead(axes, "What would remove the artifact?", y - 0.005)
        y = bullet(axes, "A genuine unbounded tail continuation or dynamic support expansion before the boundary is selected.", y)
        y = bullet(axes, "A tail atom with an internal conditional distribution, rather than treating it as one terminal point.", y)
        y = bullet(axes, "Re-gridding alone cannot create missing downside or upside support while preserving the complete old option surface exactly.", y)
        y = subhead(axes, "Assessment", y - 0.005)
        y = paragraph(axes, "The approach establishes an internally consistent X-only risk-neutral law for the modeled finite surface. It preserves survivor prices exactly and remains stable through four weekly rolls. It is not a uniqueness result or a continuous-strike, unbounded-support construction. The next research step is explicit tail/support modeling.", y, weight="bold")
        axes.text(0.075, 0.081, "Reviewed independently with QuantAgent: tower proof, re-grid identity, boundary degeneracy, and interpretation of z/J/p.", fontsize=8.2, color=MUTED)
        footer(axes, 8)
        pdf.savefig(figure)
        plt.close(figure)


if __name__ == "__main__":
    build()
