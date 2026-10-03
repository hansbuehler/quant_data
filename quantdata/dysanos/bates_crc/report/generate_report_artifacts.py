r"""Reproducible generator for this report's data tables and figures.

Run with (from quant_data-main/):
    QUANTDATA_TEST_MODE=1 .venv312/bin/python \
        quantdata/dysanos/bates_crc/report/generate_report_artifacts.py

Uses ONLY the manuscript's own fully-specified synthetic reset sequence
(Section 12) -- no market data, per the task's "must run without market
data" requirement. All randomness (the Monte Carlo independent pricing
check) uses the fixed seeds already committed in the test suite
(seed=7, seed=123), reproduced identically here.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantdata.dysanos.bates_crc.bates_transform import FourierPricingConfig, gauss_hermite_mc_cross_check, fourier_call_price
from quantdata.dysanos.bates_crc.comparison_interface import ChainGeneratorConfig
from quantdata.dysanos.bates_crc.dlv_bates_product import DlvBatesProductModel
from quantdata.dysanos.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.dysanos.bates_crc.reserve_crc import (
    JumpBox,
    ReserveSchedule,
    apply_reset,
    dominating_envelope_params,
    initial_state,
)
from quantdata.dysanos.bates_crc.verification import protected_option_sweep, verify_negative_control

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# The manuscript's own fully-specified synthetic reset sequence.
# ---------------------------------------------------------------------------

HESTON = HestonParams(kappa=2.0, theta=0.040, xi=0.30, rho=-0.70)
BOX = JumpBox(Lambda=0.5, M=0.05, delta_min=0.08, delta_max=0.15, s_star=0.25)
SCHEDULE = ReserveSchedule(
    reset_times=(2 / 252, 7 / 252, 12 / 252), reserve_sizes=(0.006, 0.006, 0.006),
    theta_min=0.025, theta_max=0.055,
)
MODEL_EXPIRIES = (5 / 252, 10 / 252, 15 / 252, 20 / 252)
JUMPS_PATH = (
    GaussianJumpParams(lambda_J=0.20, m_J=-0.03, delta_J=0.10),
    GaussianJumpParams(lambda_J=0.35, m_J=-0.02, delta_J=0.12),
    GaussianJumpParams(lambda_J=0.15, m_J=0.01, delta_J=0.09),
    GaussianJumpParams(lambda_J=0.40, m_J=-0.04, delta_J=0.14),
)
DELTAS = (0.004, -0.007, 0.005)
V_0 = 0.04
CHAIN_CONFIG = ChainGeneratorConfig(anchors=(0.7, 0.85, 1.0, 1.15, 1.3), initial_index=2, kappa=8.0, sigma_y=0.0)


def build_states():
    states = [initial_state(theta_0=0.040, jumps_0=JUMPS_PATH[0], schedule=SCHEDULE)]
    diagnostics = []
    for k in range(3):
        new_state, diag = apply_reset(
            states[-1], proposed_delta_n=DELTAS[k], proposed_jumps_n=JUMPS_PATH[k + 1],
            box=BOX, schedule=SCHEDULE,
        )
        states.append(new_state)
        diagnostics.append(diag)
    return states, diagnostics


def export_parameter_arrays(states, diagnostics) -> None:
    payload = dict(
        heston=asdict(HESTON),
        jump_box=asdict(BOX),
        schedule=dict(reset_times=list(SCHEDULE.reset_times), reserve_sizes=list(SCHEDULE.reserve_sizes),
                      theta_min=SCHEDULE.theta_min, theta_max=SCHEDULE.theta_max),
        model_expiries=list(MODEL_EXPIRIES),
        V_0=V_0,
        chain_anchors=list(CHAIN_CONFIG.anchors), chain_initial_index=CHAIN_CONFIG.initial_index,
        chain_kappa=CHAIN_CONFIG.kappa,
        jump_params_path=[asdict(j) for j in JUMPS_PATH],
        proposed_deltas=list(DELTAS),
        theta_path=list(states[-1].theta_path),
        b_used_final=list(states[-1].b_used),
        reset_diagnostics=[
            dict(reset_index=d["reset_index"], theta_prev=d["theta_prev"], theta_n=d["theta_n"],
                 delta_n=d["delta_n"], proposed_delta_n=d["proposed_delta_n"], projected=d["projected"],
                 admissible_interval=list(d["admissible_interval"]), r_n=d["r_n"], b_used_n=d["b_used_n"])
            for d in diagnostics
        ],
        reproduction_command=(
            "QUANTDATA_TEST_MODE=1 .venv312/bin/python "
            "quantdata/dysanos/bates_crc/report/generate_report_artifacts.py"
        ),
        monte_carlo_seeds=dict(primary=7, secondary=123, n_paths=400_000, n_var_substeps=100),
    )
    with open(DATA / "parameters.json", "w") as f:
        json.dump(payload, f, indent=2)


def make_model(state, n_quad=150, u_max=150.0, n_reserve_quad=80) -> DlvBatesProductModel:
    envelope = dominating_envelope_params(BOX)
    return DlvBatesProductModel(
        chain_config=CHAIN_CONFIG, heston=HESTON, state=state, schedule=SCHEDULE, envelope=envelope,
        V_t=V_0, fourier_config=FourierPricingConfig(n_quad=n_quad, u_max=u_max), n_reserve_quad=n_reserve_quad,
    )


def plot_protected_vs_unprotected(states) -> None:
    """Reset 1's price curve across strikes (top row, absolute levels), and
    the post-minus-pre-reset price DIFFERENCE (bottom row) at a PROTECTED
    maturity (T<t_1) and an UNPROTECTED one (T>t_1). The absolute-level
    curves are visually indistinguishable in both cases (the reserve
    mechanism is a small, carefully bounded correction by design) -- the
    difference row is what actually shows the protected/unprotected
    contrast, on a common y-axis so it is directly comparable."""
    t1 = SCHEDULE.reset_times[0]
    strikes = np.linspace(0.72, 1.28, 15)
    model_before = make_model(states[0])
    model_after = make_model(states[1])

    T_protected, T_unprotected = 0.9 * t1, 15 / 252
    p_before_prot = np.array([model_before.price(0.0, T_protected, float(k)) for k in strikes])
    p_after_prot = np.array([model_after.price(0.0, T_protected, float(k)) for k in strikes])
    p_before_unprot = np.array([model_before.price(0.0, T_unprotected, float(k)) for k in strikes])
    p_after_unprot = np.array([model_after.price(0.0, T_unprotected, float(k)) for k in strikes])

    diff_prot = (p_after_prot - p_before_prot) * 1.0e6
    diff_unprot = (p_after_unprot - p_before_unprot) * 1.0e6
    y_lo = min(diff_prot.min(), diff_unprot.min()) * 1.15
    y_hi = max(diff_prot.max(), diff_unprot.max()) * 1.15
    y_lo, y_hi = min(y_lo, -1.0), max(y_hi, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, (T, title, p_before, p_after, diff) in enumerate((
        (T_protected, f"Protected: $T={T_protected:.4f}<t_1$", p_before_prot, p_after_prot, diff_prot),
        (T_unprotected, f"Unprotected: $T={T_unprotected:.4f}>t_1$", p_before_unprot, p_after_unprot, diff_unprot),
    )):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.plot(strikes, p_before, label="pre-reset 1", color="#2b6cb0", lw=2)
        ax_top.plot(strikes, p_after, label="post-reset 1", color="#c05621", lw=1.5, ls="--")
        ax_top.set_ylabel("pure call price")
        ax_top.set_title(title)
        ax_top.legend(frameon=False)

        ax_bot.axhline(0.0, color="black", lw=0.8)
        ax_bot.plot(strikes, diff, color="#805ad5", lw=2, marker="o", ms=3)
        ax_bot.set_ylim(y_lo, y_hi)
        ax_bot.set_xlabel("pure strike $K$")
        ax_bot.set_ylabel("(post $-$ pre) price\n" r"$\times 10^{-6}$")
    fig.suptitle("Pre- vs post-reset prices (top) and their difference (bottom, note "
                  r"$\times10^{-6}$ scale):" "\nprotected options are exactly unchanged; "
                  "unprotected (longer-dated) options respond to recalibration")
    fig.tight_layout()
    fig.savefig(FIGURES / "protected_vs_unprotected.png", dpi=150)
    plt.close(fig)


def plot_surface_dynamics(states) -> None:
    """Price DIFFERENCE from the initial state, vs strike, one panel per
    model expiry, one line per post-reset state -- the requested
    "subsequent surface dynamics" plot. Absolute price levels are visually
    indistinguishable across states (the reserve correction is a small,
    deliberately bounded effect, see Fig. protected_vs_unprotected), so the
    difference (in units of 1e-6) is what actually shows the dynamics; each
    panel's title also reports which of that expiry's resets are still
    "protected" (T<=t_n) vs already unprotected."""
    strikes = np.linspace(0.72, 1.28, 12)
    models = [make_model(s) for s in states]
    base_prices = [np.array([models[0].price(0.0, T, float(k)) for k in strikes]) for T in MODEL_EXPIRIES]
    labels = ["after reset 1", "after reset 2", "after reset 3"]
    colors = ["#38a169", "#d69e2e", "#c05621"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for ax, T, base in zip(axes.flat, MODEL_EXPIRIES, base_prices):
        protected_n = sum(1 for t_n in SCHEDULE.reset_times if T <= t_n)
        for model, label, color in zip(models[1:], labels, colors):
            prices = np.array([model.price(0.0, T, float(k)) for k in strikes])
            diff = (prices - base) * 1.0e6
            ax.plot(strikes, diff, label=label, color=color, lw=1.8, marker="o", ms=3)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"$T={T:.4f}$ ({3 - protected_n} of 3 resets unprotected)")
        ax.set_xlabel("pure strike $K$")
        ax.set_ylabel(r"(state $-$ initial) price $\times 10^{-6}$")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Price-surface evolution relative to the initial state, across the\n"
                  "synthetic reset sequence, one panel per scheduled model expiry")
    fig.tight_layout()
    fig.savefig(FIGURES / "surface_dynamics.png", dpi=150)
    plt.close(fig)


def plot_reset_invariance_by_maturity(states) -> None:
    """|Phi_before - Phi_after| as a function of maturity, for each reset,
    showing the sharp protected/unprotected boundary at T=t_n."""
    from quantdata.dysanos.bates_crc.bates_transform import continuation_transform
    envelope = dominating_envelope_params(BOX)
    u_probe = 0.7 + 0.1j

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#2b6cb0", "#38a169", "#c05621"]
    for n, (t_n, color) in enumerate(zip(SCHEDULE.reset_times, colors)):
        Ts = np.linspace(0.05 * t_n, 4.0 * t_n, 60)
        diffs = []
        for T in Ts:
            phi_before = continuation_transform(u_probe, 0.0, float(T), V_0, HESTON, states[n], SCHEDULE, envelope, n_quad=300)
            phi_after = continuation_transform(u_probe, 0.0, float(T), V_0, HESTON, states[n + 1], SCHEDULE, envelope, n_quad=300)
            diffs.append(abs(phi_before - phi_after))
        ax.semilogy(Ts / t_n, np.maximum(diffs, 1e-20), label=f"reset {n+1} ($t_{n+1}={t_n:.4f}$)", color=color)
        ax.axvline(1.0, color=color, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel(r"$T / t_n$")
    ax.set_ylabel(r"$|\Phi_{\rm before}(u,T) - \Phi_{\rm after}(u,T)|$ (log scale)")
    ax.set_title("Reset-invariance sharply switches on at $T=t_n$\n(protected options: left of dotted line)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "reset_invariance_by_maturity.png", dpi=150)
    plt.close(fig)


def export_verification_summary(states) -> None:
    envelope = dominating_envelope_params(BOX)
    u_mesh = (0.3, 0.7 + 0.1j, 1.5 - 0.4j, -0.9 + 0.2j)
    protected = protected_option_sweep(states, SCHEDULE, HESTON, V_0, envelope, u_mesh, n_quad=400)
    negative_control = verify_negative_control(states, SCHEDULE, HESTON, V_0)

    mc_config = FourierPricingConfig(n_quad=600, u_max=300.0)
    T1 = SCHEDULE.reset_times[0]
    fourier_price = fourier_call_price(1.0, 0.0, T1, V_0, HESTON, states[0], SCHEDULE, envelope, mc_config, n_reserve_quad=400)
    mc = gauss_hermite_mc_cross_check(1.0, 0.0, T1, V_0, HESTON, envelope, n_paths=400_000, n_var_substeps=100, seed=7)

    summary = dict(
        protected_option_sweep_max_abs_diff=max(r.abs_phi_difference for r in protected),
        protected_option_sweep_n_checks=len(protected),
        negative_control=negative_control,
        independent_mc_check=dict(
            strike=1.0, maturity=T1, fourier_price=fourier_price,
            mc_price=mc["mc_price"], mc_standard_error=mc["mc_standard_error"], seed=7, n_paths=400_000,
            abs_diff=abs(fourier_price - mc["mc_price"]),
            within_6_se=abs(fourier_price - mc["mc_price"]) < 6.0 * mc["mc_standard_error"],
        ),
    )
    with open(DATA / "verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    mc = summary["independent_mc_check"]
    nc = summary["negative_control"]
    rows = [
        ("Protected-option sweep, max $|\\Phi_{\\rm before}-\\Phi_{\\rm after}|$",
         f"${summary['protected_option_sweep_max_abs_diff']:.2e}$"),
        ("Protected-option sweep, combinations tested",
         f"{summary['protected_option_sweep_n_checks']}"),
        ("Negative control, max $|\\Phi_{\\rm before}-\\Phi_{\\rm after}|$",
         f"${nc['max_abs_phi_difference']:.2e}$"),
        ("Negative control fails as expected",
         "Yes" if nc["control_fails_as_expected"] else "No"),
        ("Independent MC: Fourier price ($K=1$, $T=t_1$)", f"${mc['fourier_price']:.6f}$"),
        ("Independent MC: MC price $\\pm$ 1 s.e.", f"${mc['mc_price']:.6f} \\pm {mc['mc_standard_error']:.6f}$"),
        ("Independent MC: $|$diff$|$ within 6 s.e.", "Yes" if mc["within_6_se"] else "No"),
        ("Independent MC: seed, $n_{\\rm paths}$", f"{mc['seed']}, {mc['n_paths']:,}"),
    ]
    with open(DATA / "verification_summary_table.tex", "w") as f:
        f.write("\\begin{tabular}{lr}\n\\toprule\nCheck & Result \\\\\n\\midrule\n")
        for label, value in rows:
            f.write(f"{label} & {value} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return summary


def main() -> None:
    states, diagnostics = build_states()
    export_parameter_arrays(states, diagnostics)
    plot_protected_vs_unprotected(states)
    plot_surface_dynamics(states)
    plot_reset_invariance_by_maturity(states)
    summary = export_verification_summary(states)
    print("Wrote figures to", FIGURES)
    print("Wrote data to", DATA)
    print("Verification summary:", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
