r"""One-command reproduction route for the real-data calibration comparison.

Run (from quant_data-main/):
    QUANTDATA_TEST_MODE=generate_tests .venv312/bin/python \
        quantdata/dysanos/bates_crc/report/../real_data_study/run_full_study.py

Produces `report/data/*.json` consumed by the LaTeX report and `plots.py`.

Compute-bounding choices (deliberate, documented rather than silent):
- Main comparison table (dy2, SANOS snapshot, DLV x Black, DLV x ordinary
  Bates): run on ALL SIX dates for SPX (primary), and on the FOUR held-out
  dates only for AAPL (secondary robustness check, DATA_PROVENANCE.md
  Sec. 4).
- Reserve budget sweep + node-count sweep (Section C, the most expensive
  part -- each reserve-budget refit is a full multi-start nonlinear fit):
  run on THREE representative SPX dates only (one selection date, one
  stressed held-out date, one dense modern held-out date), not all six --
  documented here, not silently narrowed.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.dysanos.bates_crc.real_data_study import config as cfg
from quantdata.dysanos.bates_crc.real_data_study.chain_bates_calibration import (
    ChainBatesCalibrationError,
    calibrate_chain_bates,
)
from quantdata.dysanos.bates_crc.real_data_study.chain_black_calibration import (
    ChainBlackCalibrationError,
    calibrate_chain_black,
)
from quantdata.dysanos.bates_crc.real_data_study.day_loader import build_evaluation_set, load_day
from quantdata.dysanos.bates_crc.real_data_study.dy2_baseline import price_dy2
from quantdata.dysanos.bates_crc.real_data_study.extended_bates_reserve_study import (
    ChainExtendedBatesCalibrationError,
    admissible_theta_interval,
    calibrate_chain_extended_bates,
    reserve_price_shift,
    reset_surface_change,
)
from quantdata.dysanos.bates_crc.real_data_study.martingale_completion import verify_martingale_completion
from quantdata.dysanos.bates_crc.real_data_study.metrics import evaluate_model
from quantdata.dysanos.bates_crc.real_data_study.sanos_snapshot import fit_sanos_snapshot, price_sanos_snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "data"
OUT.mkdir(parents=True, exist_ok=True)

RESERVE_SWEEP_DATES = (date(2005, 11, 21), date(2011, 7, 7), date(2024, 3, 5))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (date,)):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Exception):
        return str(obj)
    return str(obj)


def _metrics_summary(m) -> dict:
    return dict(
        model_name=m.model_name, n_contracts=m.n_contracts, pcnt_inside_bid_ask=m.pcnt_inside_bid_ask,
        median_fit_err=m.median_fit_err, mean_fit_err=m.mean_fit_err,
        mean_iv_error_bps=m.mean_iv_error_bps, median_abs_iv_error_bps=m.median_abs_iv_error_bps,
        n_iv_inversions_failed=m.n_iv_inversions_failed, runtime_seconds=m.runtime_seconds,
        peak_memory_bytes=m.peak_memory_bytes, by_bucket=m.by_bucket,
    )


def run_one_day(ticker: str, d: date) -> dict:
    t_start = time.time()
    result: dict = dict(ticker=ticker, date=d.isoformat())
    try:
        book = load_day(ticker, d)
        ev = build_evaluation_set(book)
        result["n_raw_contracts"] = book.n_raw_contracts
        result["n_evaluation_contracts"] = len(ev.contracts)
        result["n_excluded_target_points"] = len(ev.excluded)
        result["excluded"] = ev.excluded
    except Exception as exc:  # noqa: BLE001
        result["load_error"] = f"{exc}\n{traceback.format_exc()}"
        return result

    models: dict = {}

    models["dy2"] = _metrics_summary(evaluate_model("dy2", lambda s, e: price_dy2(book, s, e), ev.contracts))
    models["dy2"]["fit_result_median_err_half_spreads"] = float(book.dy2_fit_result["median_err"])
    models["dy2"]["fit_result_pcnt_fit"] = float(book.dy2_fit_result["pcnt_fit"])

    try:
        pricer, sanos_time = fit_sanos_snapshot(book)
        m = evaluate_model("sanos_snapshot", lambda s, e: price_sanos_snapshot(pricer, s, e), ev.contracts)
        models["sanos_snapshot"] = _metrics_summary(m)
        models["sanos_snapshot"]["fit_time_seconds"] = sanos_time
    except Exception as exc:  # noqa: BLE001
        models["sanos_snapshot"] = dict(error=str(exc))

    try:
        calib_black = calibrate_chain_black(ev)
        m = evaluate_model("dlv_black", calib_black.price, ev.contracts)
        models["dlv_black"] = _metrics_summary(m)
        models["dlv_black"]["kappa"] = calib_black.chain_config.kappa
        models["dlv_black"]["sigma_Y"] = calib_black.sigma_Y
        models["dlv_black"]["anchors"] = list(calib_black.chain_config.anchors)
        models["dlv_black"]["final_cost"] = calib_black.final_cost
        models["dlv_black"]["converged"] = calib_black.converged
        models["dlv_black"]["chapman_kolmogorov_residual"] = calib_black.chapman_kolmogorov_residual
        models["dlv_black"]["generator_residual"] = calib_black.generator_residual
        models["dlv_black"]["starts"] = calib_black.starts
    except ChainBlackCalibrationError as exc:
        models["dlv_black"] = dict(error=str(exc))

    calib_bates = None
    try:
        calib_bates = calibrate_chain_bates(ev, n_quad=cfg.FOURIER_N_QUAD_OPTIMIZATION,
                                             u_max=cfg.FOURIER_U_MAX_OPTIMIZATION)
        m = evaluate_model("dlv_bates", calib_bates.price, ev.contracts)
        models["dlv_bates"] = _metrics_summary(m)
        models["dlv_bates"]["heston"] = asdict(calib_bates.heston)
        models["dlv_bates"]["jumps"] = asdict(calib_bates.jumps)
        models["dlv_bates"]["V_t"] = calib_bates.V_t
        models["dlv_bates"]["chain_kappa"] = calib_bates.chain_config.kappa
        models["dlv_bates"]["anchors"] = list(calib_bates.chain_config.anchors)
        models["dlv_bates"]["final_cost"] = calib_bates.final_cost
        models["dlv_bates"]["converged"] = calib_bates.converged
        models["dlv_bates"]["chapman_kolmogorov_residual"] = calib_bates.chapman_kolmogorov_residual
        models["dlv_bates"]["generator_residual"] = calib_bates.generator_residual
        models["dlv_bates"]["starts"] = calib_bates.starts

        mc = verify_martingale_completion(calib_bates.chain_config, cfg.TARGET_TENORS_YEARS)
        models["dlv_bates"]["martingale_completion"] = dict(
            chapman_kolmogorov_residual=mc.chapman_kolmogorov_residual, generator_residual=mc.generator_residual,
            mass_residual=mc.mass_residual, mean_residual=mc.mean_residual, variant_label=mc.variant_label,
        )
    except ChainBatesCalibrationError as exc:
        models["dlv_bates"] = dict(error=str(exc))

    result["models"] = models
    result["wall_clock_seconds"] = time.time() - t_start

    if calib_bates is not None and calib_bates.converged and d in RESERVE_SWEEP_DATES and ticker == cfg.PRIMARY_TICKER:
        result["reserve_sweep"] = run_reserve_sweep(ev, calib_bates)

    return result


def run_reserve_sweep(ev, calib_bates) -> dict:
    strikes = np.array([c.strike for c in ev.contracts])
    expiries = np.array([c.expiry_years for c in ev.contracts])
    sweep: dict = dict(budgets={})

    for budget in cfg.RESERVE_BUDGET_SWEEP:
        entry: dict = dict(reserve_size=budget)
        t0 = time.time()
        shift = reserve_price_shift(calib_bates, budget, strikes, expiries)
        entry["price_shift_before_recalibration"] = dict(
            mean_abs_shift=float(np.mean(np.abs(shift["shift"]))),
            max_abs_shift=float(np.max(np.abs(shift["shift"]))),
            mean_relative_shift_pct=float(np.mean(np.abs(shift["shift"]) / np.maximum(shift["baseline_price"], 1e-6)) * 100),
        )
        price_key = "with_reserve_price" if budget > 0.0 else "baseline_price"
        m_before = evaluate_model(f"dlv_extbates_prerecalib_r{budget}",
                                    lambda s, e, _shift=shift, _key=price_key: _shift[_key],
                                    ev.contracts)
        entry["fit_before_recalibration"] = _metrics_summary(m_before)

        if budget > 0.0:
            try:
                refit = calibrate_chain_extended_bates(ev, budget, warm_start=calib_bates, n_starts=2,
                                                         n_quad=cfg.FOURIER_N_QUAD_OPTIMIZATION,
                                                         u_max=cfg.FOURIER_U_MAX_OPTIMIZATION)
                m_after = evaluate_model(f"dlv_extbates_postrecalib_r{budget}", refit.price, ev.contracts)
                entry["fit_after_recalibration"] = _metrics_summary(m_after)
                entry["recalibration_converged"] = refit.converged
                entry["recalibration_final_cost"] = refit.final_cost
                cost_before = m_before.mean_fit_err
                cost_after = m_after.mean_fit_err
                cost_baseline = 0.0  # ordinary-Bates fit's own mean_fit_err, filled in below by caller if needed
                entry["mean_fit_err_before_recalibration"] = cost_before
                entry["mean_fit_err_after_recalibration"] = cost_after

                lo, hi = admissible_theta_interval(calib_bates, budget)
                entry["admissible_theta_interval"] = dict(lo=lo, hi=hi, theta_0=calib_bates.heston.theta)

                mask = expiries > cfg.RESERVE_RESET_HORIZON_YEARS
                if np.any(mask):
                    change_up = reset_surface_change(calib_bates, budget, delta_fraction=1.0, strikes=strikes,
                                                        expiries=expiries)
                    change_down = reset_surface_change(calib_bates, budget, delta_fraction=0.0, strikes=strikes,
                                                          expiries=expiries)
                    entry["surface_change_after_admissible_reset"] = dict(
                        upward=dict(mean_abs_price_change=float(np.mean(np.abs(change_up["price_change"]))),
                                     theta_after=change_up["theta_after"]),
                        downward=dict(mean_abs_price_change=float(np.mean(np.abs(change_down["price_change"]))),
                                       theta_after=change_down["theta_after"]),
                    )
            except ChainExtendedBatesCalibrationError as exc:
                entry["recalibration_error"] = str(exc)
        entry["wall_clock_seconds"] = time.time() - t0
        sweep["budgets"][str(budget)] = entry

    return sweep


def main() -> None:
    results = []
    for d in cfg.ALL_DATES:
        print(f"=== SPX {d} ===", flush=True)
        results.append(run_one_day(cfg.PRIMARY_TICKER, d))
    for d in cfg.HELD_OUT_DATES:
        print(f"=== AAPL {d} (secondary robustness) ===", flush=True)
        results.append(run_one_day(cfg.SECONDARY_TICKER, d))

    with open(OUT / "full_study_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print("Wrote", OUT / "full_study_results.json")


if __name__ == "__main__":
    main()
