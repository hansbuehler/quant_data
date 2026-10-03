r"""A mechanical (not statistical) CRC-routine check on the REAL calibrated
DLV x Bates parameter path across the 6 available SPX dates.

This is NOT the deferred "Historical Simulator" task (see project memory) --
it does not estimate a physical law P, does not filter a latent state, and
makes no forecasting claim. It asks a narrower, purely constructive
question: chaining the 6 dates' ALREADY-CALIBRATED (cross-sectional)
backgrounds in chronological order, does the REALISED change in theta (and
jump parameters) from one date to the next fall inside the reserve-CRC's
admissible ("permissible drift") region -- and when it does, does actually
running reserve_crc.apply_reset on these real numbers preserve protected-
option prices exactly, as the construction guarantees?

Each transition is evaluated independently and exactly (no fitting, no
statistical estimation) via the already-verified reserve_crc/bates_transform/
chain_pricer machinery, reusing the calibrated (chain, heston, jumps, V_t)
fits already produced by chain_bates_calibration.py and stored in
report/data/full_study_results.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.dysanos.bates_crc.comparison_interface import ChainGeneratorConfig
from quantdata.dysanos.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.dysanos.bates_crc.reserve_crc import (
    JumpBox,
    ReserveSchedule,
    admissible_delta_interval,
    apply_reset,
    dominating_envelope_params,
    initial_state,
    verify_envelope_dominates,
)
from quantdata.dysanos.bates_crc.real_data_study import config as cfg
from quantdata.dysanos.bates_crc.real_data_study.chain_pricer import price_calls_chain_extended_bates

HERE = Path(__file__).resolve().parent
DATA = HERE / "report" / "data"

THETA_BAND_HALF_WIDTH = cfg.THETA_BAND_HALF_WIDTH  # 0.03, same convention as the reserve-sweep study
JUMP_BOX_WIDEN_FACTOR = cfg.JUMP_BOX_WIDEN_FACTOR  # 1.5
RESERVE_SWEEP_FOR_SUFFICIENT_BUDGET = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)


@dataclass(frozen=True)
class DatedBackground:
    d: date
    heston: HestonParams
    jumps: GaussianJumpParams
    chain_config: ChainGeneratorConfig
    V_t: float


def load_real_path(ticker: str = "SPX") -> list:
    with open(DATA / "full_study_results.json") as f:
        results = json.load(f)
    path = []
    for r in results:
        if r.get("ticker") != ticker or "models" not in r:
            continue
        m = r["models"].get("dlv_bates")
        if m is None or "error" in m or not m.get("converged"):
            continue
        y, mo, dd = (int(x) for x in r["date"].split("-"))
        heston = HestonParams(**m["heston"])
        jumps = GaussianJumpParams(**m["jumps"])
        anchors = tuple(m["anchors"])
        i0 = len(anchors) // 2
        chain_config = ChainGeneratorConfig(anchors=anchors, initial_index=i0, kappa=m["chain_kappa"], sigma_y=0.0)
        path.append(DatedBackground(d=date(y, mo, dd), heston=heston, jumps=jumps, chain_config=chain_config,
                                     V_t=m["V_t"]))
    path.sort(key=lambda db: db.d)
    return path


def _years_between(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.25


def _minimum_sufficient_reserve(delta_theta: float) -> float | None:
    """Exact: admissible_delta_interval's hi = min(r, theta_max-theta_prev)
    only binds for an UPWARD move (delta_theta>0); a downward move's
    feasibility depends only on theta_min, never on r. Returns None if no
    reserve can help (downward move: N/A, reserve not the limiting factor;
    or delta_theta itself already exceeds the band, band_ok=False handles
    that case separately)."""
    if delta_theta <= 0.0:
        return 0.0  # any positive r is technically sufficient; reserve is not the binding constraint
    return delta_theta


def evaluate_transition(prev: DatedBackground, nxt: DatedBackground, eval_contracts: list | None = None) -> dict:
    theta_prev = prev.heston.theta
    theta_next = nxt.heston.theta
    delta_theta_realised = theta_next - theta_prev
    gap_years = _years_between(prev.d, nxt.d)

    theta_min = theta_prev - THETA_BAND_HALF_WIDTH
    theta_max = theta_prev + THETA_BAND_HALF_WIDTH
    band_ok = theta_min <= theta_next <= theta_max
    min_band_half_width_needed = abs(delta_theta_realised)

    box = JumpBox(
        Lambda=max(prev.jumps.lambda_J * JUMP_BOX_WIDEN_FACTOR, 0.05),
        M=max(abs(prev.jumps.m_J) * JUMP_BOX_WIDEN_FACTOR, 0.01),
        delta_min=prev.jumps.delta_J / JUMP_BOX_WIDEN_FACTOR,
        delta_max=prev.jumps.delta_J * JUMP_BOX_WIDEN_FACTOR,
        s_star=prev.jumps.delta_J * JUMP_BOX_WIDEN_FACTOR * 1.3,
    )
    jumps_in_box = box.contains(nxt.jumps)

    result = dict(
        date_prev=prev.d.isoformat(), date_next=nxt.d.isoformat(), gap_years=gap_years,
        theta_prev=theta_prev, theta_next=theta_next, delta_theta_realised=delta_theta_realised,
        theta_band=(theta_min, theta_max), band_ok=band_ok,
        min_band_half_width_needed=min_band_half_width_needed,
        jumps_prev=dict(lambda_J=prev.jumps.lambda_J, m_J=prev.jumps.m_J, delta_J=prev.jumps.delta_J),
        jumps_next=dict(lambda_J=nxt.jumps.lambda_J, m_J=nxt.jumps.m_J, delta_J=nxt.jumps.delta_J),
        jump_box=dict(Lambda=box.Lambda, M=box.M, delta_min=box.delta_min, delta_max=box.delta_max, s_star=box.s_star),
        jumps_in_box=jumps_in_box,
    )

    if not band_ok:
        result["admissible"] = False
        result["reason"] = (
            f"realised theta move {delta_theta_realised:+.5f} exceeds the declared "
            f"+/-{THETA_BAND_HALF_WIDTH} operating band regardless of reserve budget "
            f"(needs a band of at least +/-{min_band_half_width_needed:.5f})"
        )
        return result
    if not jumps_in_box:
        result["admissible"] = False
        result["reason"] = f"realised jump parameters at {nxt.d} fall outside the {JUMP_BOX_WIDEN_FACTOR}x-widened box"
        return result

    min_reserve = _minimum_sufficient_reserve(delta_theta_realised)
    sufficient_budget = next((r for r in RESERVE_SWEEP_FOR_SUFFICIENT_BUDGET if r >= max(min_reserve, 1e-6)), None)
    if sufficient_budget is None:
        result["admissible"] = False
        result["reason"] = f"needed reserve {min_reserve:.5f} exceeds every swept budget {RESERVE_SWEEP_FOR_SUFFICIENT_BUDGET}"
        return result

    schedule = ReserveSchedule(reset_times=(gap_years,), reserve_sizes=(sufficient_budget,),
                                theta_min=theta_min, theta_max=theta_max)
    state0 = initial_state(theta_0=theta_prev, jumps_0=prev.jumps, schedule=schedule)
    new_state, diag = apply_reset(state0, proposed_delta_n=delta_theta_realised, proposed_jumps_n=nxt.jumps,
                                   box=box, schedule=schedule)

    result["admissible"] = True
    result["min_reserve_required"] = min_reserve
    result["reserve_budget_used"] = sufficient_budget
    result["apply_reset_diagnostics"] = dict(
        projected=diag["projected"], delta_n=diag["delta_n"], proposed_delta_n=diag["proposed_delta_n"],
        admissible_interval=diag["admissible_interval"], b_used_n=diag["b_used_n"],
    )
    result["projection_free"] = not diag["projected"]  # True means the REAL delta was used exactly, no clamping

    if eval_contracts:
        envelope = dominating_envelope_params(box)
        strikes = np.array([c.strike for c in eval_contracts])
        expiries = np.array([c.expiry_years for c in eval_contracts])
        protected_mask = expiries < gap_years
        if np.any(protected_mask):
            p_strikes = strikes[protected_mask]
            p_expiries = expiries[protected_mask]
            price_before = price_calls_chain_extended_bates(
                prev.chain_config, prev.heston, prev.jumps, envelope, prev.V_t, p_strikes, p_expiries,
                schedule.reset_times, state0.b_used, t=0.0, n_quad=150, u_max=200.0, n_reserve_quad=60,
            )
            price_after = price_calls_chain_extended_bates(
                prev.chain_config, prev.heston, new_state.jump_params_path[-1], envelope, prev.V_t, p_strikes,
                p_expiries, schedule.reset_times, new_state.b_used, t=0.0, n_quad=150, u_max=200.0, n_reserve_quad=60,
            )
            result["protected_option_check"] = dict(
                n_protected_contracts=int(protected_mask.sum()),
                max_abs_price_diff=float(np.max(np.abs(price_after - price_before))),
                mean_abs_price_diff=float(np.mean(np.abs(price_after - price_before))),
            )
        else:
            result["protected_option_check"] = dict(n_protected_contracts=0, note="no contracts with expiry < gap_years available")

    return result


def evaluate_transition_forced_admissible(prev: DatedBackground, nxt: DatedBackground,
                                           eval_contracts: list | None = None) -> dict:
    """Separate, clearly-labeled sensitivity pass: widens the theta band and
    jump box just enough (per transition) to make the REAL realised move
    admissible, purely to verify the apply_reset + protected-option-
    invariance MECHANISM itself is correct on real calibrated numbers --
    NOT a claim that this widened region is a reasonable/small "permissible
    drift region". Reports the widening factor actually required, so the
    two questions (does the mechanism work? / is the realistic region large
    enough?) stay separated."""
    theta_prev = prev.heston.theta
    theta_next = nxt.heston.theta
    delta_theta_realised = theta_next - theta_prev
    gap_years = _years_between(prev.d, nxt.d)

    needed_half_width = abs(delta_theta_realised) * 1.05 + 1e-6
    theta_min = theta_prev - needed_half_width
    theta_max = theta_prev + needed_half_width

    # Widen the jump box multiplicatively until it contains jumps_next, tracking the factor
    # needed. s_star must dominate BOTH delta_max and M with a comfortable margin -- R_* =
    # (s_*/delta_min)*exp(M^2/(2*(s_*^2-delta_max^2))) blows up if s_* only barely exceeds
    # delta_max while M is not small relative to that gap (this is exactly what happens for
    # the 2024-12-26 -> 2025-01-03 transition, whose target m_J=-0.4999 sits at the Bates
    # calibration's own bound -- a real, reportable fragility of the envelope construction
    # under an extreme/bound-hitting calibrated jump mean, not a pricing bug).
    widen = JUMP_BOX_WIDEN_FACTOR
    box = None
    for _ in range(200):
        delta_max = max(prev.jumps.delta_J, nxt.jumps.delta_J) * widen
        M = max(abs(prev.jumps.m_J) * widen, abs(nxt.jumps.m_J) * 1.05, 0.01)
        box = JumpBox(
            Lambda=max(prev.jumps.lambda_J * widen, nxt.jumps.lambda_J * 1.05, 0.05),
            M=M,
            delta_min=min(prev.jumps.delta_J, nxt.jumps.delta_J) / widen,
            delta_max=delta_max,
            s_star=float(np.sqrt(delta_max ** 2 + (3.0 * M) ** 2)) * 1.05,
        )
        if box.contains(nxt.jumps):
            break
        widen *= 1.2
    assert box is not None and box.contains(nxt.jumps)

    min_reserve = _minimum_sufficient_reserve(delta_theta_realised)
    reserve_budget = max(min_reserve * 1.05, 1e-4)

    schedule = ReserveSchedule(reset_times=(gap_years,), reserve_sizes=(reserve_budget,),
                                theta_min=theta_min, theta_max=theta_max)
    state0 = initial_state(theta_0=theta_prev, jumps_0=prev.jumps, schedule=schedule)
    new_state, diag = apply_reset(state0, proposed_delta_n=delta_theta_realised, proposed_jumps_n=nxt.jumps,
                                   box=box, schedule=schedule)

    result = dict(
        date_prev=prev.d.isoformat(), date_next=nxt.d.isoformat(), gap_years=gap_years,
        delta_theta_realised=delta_theta_realised,
        theta_band_half_width_required=needed_half_width,
        theta_band_widening_vs_standard=needed_half_width / THETA_BAND_HALF_WIDTH,
        jump_box_widen_factor_required=widen,
        jump_box_widening_vs_standard=widen / JUMP_BOX_WIDEN_FACTOR,
        reserve_budget_used=reserve_budget,
        projection_free=not diag["projected"],
        b_used_n=diag["b_used_n"],
    )

    if eval_contracts:
        envelope = dominating_envelope_params(box)
        strikes = np.array([c.strike for c in eval_contracts])
        expiries = np.array([c.expiry_years for c in eval_contracts])
        protected_mask = expiries < gap_years
        if np.any(protected_mask):
            p_strikes = strikes[protected_mask]
            p_expiries = expiries[protected_mask]
            try:
                with np.errstate(over="raise", invalid="raise"):
                    price_before = price_calls_chain_extended_bates(
                        prev.chain_config, prev.heston, prev.jumps, envelope, prev.V_t, p_strikes, p_expiries,
                        schedule.reset_times, state0.b_used, t=0.0, n_quad=150, u_max=200.0, n_reserve_quad=60,
                    )
                    price_after = price_calls_chain_extended_bates(
                        prev.chain_config, prev.heston, new_state.jump_params_path[-1], envelope, prev.V_t,
                        p_strikes, p_expiries, schedule.reset_times, new_state.b_used, t=0.0, n_quad=150,
                        u_max=200.0, n_reserve_quad=60,
                    )
                result["protected_option_check"] = dict(
                    n_protected_contracts=int(protected_mask.sum()),
                    max_abs_price_diff=float(np.max(np.abs(price_after - price_before))),
                    mean_abs_price_diff=float(np.mean(np.abs(price_after - price_before))),
                )
            except FloatingPointError as exc:
                # Genuine numerical breakdown, not a tuning issue: when the dominating
                # envelope must be this wide (here driven by a bound-hitting target m_J),
                # the Carr-Madan alpha-shifted contour leaves the characteristic function's
                # strip of decay and the Fourier integrand overflows at every quadrature
                # setting tried (confirmed down to u_max=10). Reported explicitly rather
                # than silently returning NaN.
                result["protected_option_check"] = dict(
                    n_protected_contracts=int(protected_mask.sum()),
                    numerical_failure=(
                        f"Carr-Madan Fourier pricing overflowed ({exc}) -- the dominating "
                        f"envelope forced this wide (delta_J={envelope.delta_J:.3f}, "
                        f"lambda_J={envelope.lambda_J:.3g}) exceeds the standard damped-"
                        f"contour method's stable range; not resolved by reducing n_quad/u_max"
                    ),
                )
        else:
            result["protected_option_check"] = dict(n_protected_contracts=0, note="no protected contracts available")

    return result


def run_full_sequence() -> tuple[list, list]:
    path = load_real_path("SPX")
    from quantdata.dysanos.bates_crc.real_data_study.day_loader import build_evaluation_set, load_day
    standard, forced = [], []
    for i in range(len(path) - 1):
        prev, nxt = path[i], path[i + 1]
        try:
            book = load_day("SPX", prev.d)
            ev = build_evaluation_set(book)
            contracts = ev.contracts
        except Exception:
            contracts = None
        standard.append(evaluate_transition(prev, nxt, eval_contracts=contracts))
        forced.append(evaluate_transition_forced_admissible(prev, nxt, eval_contracts=contracts))
    return standard, forced


if __name__ == "__main__":
    import os
    if os.environ.get("QUANTDATA_TEST_MODE") != "generate_tests":
        raise RuntimeError("run with QUANTDATA_TEST_MODE=generate_tests")
    standard, forced = run_full_sequence()
    with open(DATA / "crc_real_sequence.json", "w") as f:
        json.dump(standard, f, indent=2, default=str)
    with open(DATA / "crc_real_sequence_forced.json", "w") as f:
        json.dump(forced, f, indent=2, default=str)

    print("=== standard region (+/-0.03 theta band, 1.5x jump-box widen) ===")
    for t in standard:
        print(t["date_prev"], "->", t["date_next"], f"(gap={t['gap_years']:.2f}y)",
              f"delta_theta={t['delta_theta_realised']:+.5f}", "admissible=" + str(t["admissible"]),
              t.get("reason", ""))
    print("=== forced-admissible sensitivity pass (mechanism check only) ===")
    for t in forced:
        pc = t.get("protected_option_check", {})
        print(t["date_prev"], "->", t["date_next"],
              f"theta_widen={t['theta_band_widening_vs_standard']:.1f}x "
              f"jumpbox_widen={t['jump_box_widening_vs_standard']:.1f}x "
              f"projection_free={t['projection_free']} "
              f"protected_max_diff={pc.get('max_abs_price_diff')} (n={pc.get('n_protected_contracts')})")
    print("Wrote", DATA / "crc_real_sequence.json", "and", DATA / "crc_real_sequence_forced.json")
