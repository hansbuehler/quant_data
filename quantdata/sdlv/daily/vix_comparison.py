"""Historical paper GROMM/SDLV VIX: full-strike and Cboe discrete formula.

Loads the retained checkpoint without fitting. Reads unfiltered OptionMetrics
books and zero curves from the canonical NAS cache, never downloading them.
The Cboe formula is exact, but EOD OptionMetrics quotes/rates are not Cboe's
timestamped constituent feed. No extrapolation beyond the trained 25-day
surface is used for the primary comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from cdxcore.err import verify_inp
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import ndtr

from quantdata import SubDir, env
from quantdata.biz_calendar import AM_EXIRY_SUBTRACT, DAYS_PER_YEAR, get_business_days
from quantdata.sdlv.daily import DATASET_PATH, FITS, load_raw_grids
from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL, decode_dlvsq
from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower
from quantdata.vixdata import load_cboe_vix
from quantdata.vixdata.formula import (
    constant_maturity_vix,
    discrete_total_variance,
    select_vix_strip,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

METHODOLOGY_URL = "https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf"


def log_contract_total_variance(
    support: NDArray[np.float64], density: NDArray[np.float64], *,
    business_time_years: float, background_volatility: float = BACKGROUND_ANNUAL_VOL,
) -> float:
    """Exact full-strike integral: -2 E[log U] + sigma_background²*t_business.

    This includes all model tails, not just the fitted strike interval.
    Calendar annualization is applied separately, after expiry interpolation.
    """
    support, density = np.asarray(support), np.asarray(density)
    verify_inp(support.ndim == 1 and density.shape == support.shape,
               lambda: "Support and density must be same-length vectors")
    verify_inp(np.all(np.isfinite(support)) and np.all(support > 0), lambda: "Invalid support")
    verify_inp(np.all(np.isfinite(density)) and np.all(density >= 0), lambda: "Invalid density")
    verify_inp(business_time_years > 0 and np.isfinite(business_time_years)
               and background_volatility >= 0 and np.isfinite(background_volatility),
               lambda: "Invalid maturity or background volatility")
    if not np.isclose(density.sum(), 1, atol=1e-10, rtol=0) or not np.isclose(density @ support, 1, atol=1e-10, rtol=0):
        raise ValueError("Density must have unit mass and unit forward")
    result = float(-2 * (density @ np.log(support)) + background_volatility**2 * business_time_years)
    if result < -1e-12:
        raise ValueError("Negative total log variance")
    return max(0.0, result)


def interpolate_density(tower: NDArray[np.float64], business_days: float) -> NDArray[np.float64]:
    """Use the fitted surface's linear density interpolation, with no extrapolation."""
    verify_inp(np.isfinite(business_days) and 1 <= business_days <= len(tower),
               lambda: f"Maturity {business_days} is outside [1, {len(tower)}] business days")
    lower, upper = int(np.floor(business_days)) - 1, int(np.ceil(business_days)) - 1
    alpha = business_days - np.floor(business_days)
    return (1 - alpha) * tower[lower] + alpha * tower[upper]


def mixture_prices(
    support: NDArray[np.float64], density: NDArray[np.float64],
    strikes: NDArray[np.float64], *, business_days: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Undiscounted, forward-normalized calls and puts, with stable OTM puts."""
    x, k = support[None, :], np.asarray(strikes)[:, None]
    vol = BACKGROUND_ANNUAL_VOL * np.sqrt(business_days / DAYS_PER_YEAR)
    d1 = np.log(x / k) / vol + vol / 2
    d2 = d1 - vol
    calls = (x * ndtr(d1) - k * ndtr(d2)) @ density
    puts = (k * ndtr(-d2) - x * ndtr(-d1)) @ density
    return np.maximum(calls, 0), np.maximum(puts, 0)


def calendar_business_maturity(trading_date: date, sessions: NDArray[np.datetime64]) -> float:
    """Map date+30 calendar days to bracketing trading-day endpoints.

    For weekends/holidays, interpolate total variance in calendar time between
    the adjacent trading closes. Half sessions count as a day, as in SDLV.
    """
    origin = int(np.searchsorted(sessions, np.datetime64(trading_date), side="right"))
    future = sessions[origin:origin + 25]
    offsets = (future - np.datetime64(trading_date)).astype(int)
    if len(future) < 25 or not offsets[0] <= 30 <= offsets[-1]:
        raise ValueError("Insufficient calendar/model horizon")
    return float(np.interp(30.0, offsets, np.arange(1, 26)))


def eligible_expiries(trading_date: date, raw: pd.DataFrame, sessions: NDArray[np.datetime64]) -> list[tuple[date, bool, float, float]]:
    """Friday options (holiday-adjusted), AM standard third Friday, PM others."""
    entries = []
    available = set(zip(raw["exdate"].astype(str), raw["am_settlement"].astype(float)))
    origin = np.searchsorted(sessions, np.datetime64(trading_date), side="right")
    for days in range(23, 39):
        nominal = trading_date + timedelta(days=days)
        if nominal.weekday() != 4:
            continue
        settlement = sessions[np.searchsorted(sessions, np.datetime64(nominal), side="right") - 1].astype(object)
        is_am = 15 <= nominal.day <= 21
        if (settlement.isoformat(), float(is_am)) not in available:
            continue
        # OptionMetrics provides EOD books, not the exact Cboe closing tick.
        calendar_days = (settlement - trading_date).days - (405 / 1440 if is_am else 15 / 1440)
        if not 23 < calendar_days < 37:
            continue
        bdays = float(np.searchsorted(sessions, np.datetime64(settlement), side="right") - origin)
        bdays -= AM_EXIRY_SUBTRACT if is_am else 0
        entries.append((settlement, is_am, calendar_days, bdays))
    return sorted(entries, key=lambda value: value[2])


def _metrics(model: np.ndarray, actual: np.ndarray) -> dict[str, float | int | None]:
    model, actual = np.asarray(model, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(model) & np.isfinite(actual)
    model, actual = model[valid], actual[valid]
    if not len(model):
        return {"observations": 0}
    error = model - actual
    return {
        "observations": len(error), "mean_model": float(model.mean()), "mean_actual": float(actual.mean()),
        "bias_vol_points": float(error.mean()), "mae_vol_points": float(np.abs(error).mean()),
        "rmse_vol_points": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_vol_points": float(np.median(np.abs(error))),
        "max_absolute_error_vol_points": float(np.max(np.abs(error))),
        "correlation": float(np.corrcoef(model, actual)[0, 1]) if len(model) > 1 else None,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _term(
    *, trading_date: date, expiry: tuple[date, bool, float, float], raw: pd.DataFrame,
    zero: pd.DataFrame, support: np.ndarray, tower: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settlement, is_am, calendar_days, business_days = expiry
    rows = raw[(raw["exdate"].astype(str) == settlement.isoformat()) & (raw["am_settlement"].astype(float) == float(is_am))].copy()
    rows = rows[(rows["exercise_style"] == "E") & (rows["ss_flag"].astype(str) == "0")]
    if rows.duplicated(["strike_price", "cp_flag"]).any():
        raise ValueError("Duplicate strike/side contracts in expiry")
    books = rows.pivot(index="strike_price", columns="cp_flag", values=["best_bid", "best_offer"]).sort_index()
    strikes = books.index.to_numpy(dtype=np.float64)
    def book(field: str, side: str) -> np.ndarray:
        return books[(field, side)].to_numpy(dtype=np.float64, na_value=np.nan)
    curve = zero.sort_values("days")
    rate = float(np.interp(calendar_days, curve["days"].astype(float), curve["rate"].astype(float))) / 100
    discount = float(np.exp(-rate * calendar_days / 365))
    cb, ca, pb, pa = [book(field, side) for field, side in (("best_bid", "C"), ("best_offer", "C"), ("best_bid", "P"), ("best_offer", "P"))]
    strip = select_vix_strip(strikes=strikes, call_bids=cb, call_asks=ca, put_bids=pb, put_asks=pa, discount_factor=discount)
    market_variance = discrete_total_variance(strip, discount_factor=discount)
    within_horizon = 1 <= business_days <= len(tower)
    model_variance, full_variance = None, None
    outside_excess = None
    model_prices = np.full(len(strip.strikes), np.nan)
    normalized_strikes = strip.strikes / strip.forward
    outside_support = (normalized_strikes < support[1]) | (normalized_strikes > support[-2])
    if within_horizon:
        density = interpolate_density(tower, business_days)
        calls, puts = mixture_prices(support, density, strip.strikes / strip.forward, business_days=business_days)
        model_prices = np.where(strip.strikes < strip.k0, puts, calls)
        at_k0 = strip.strikes == strip.k0
        model_prices[at_k0] = (calls[at_k0] + puts[at_k0]) / 2
        model_prices *= strip.forward * discount
        model_variance = discrete_total_variance(strip, discount_factor=discount, prices=model_prices)
        full_variance = log_contract_total_variance(support, density, business_time_years=business_days / DAYS_PER_YEAR)
        # The common K0 correction cancels in this model-minus-market decomposition.
        differences = 2 * strip.delta_strikes * (model_prices - strip.prices) / (discount * strip.strikes**2)
        outside_excess = float(differences[outside_support].sum())
    summary = {
        "date": trading_date.isoformat(), "expiry": settlement.isoformat(), "is_am": is_am,
        "calendar_days": calendar_days, "business_days": business_days,
        "discount_factor": discount, "forward": strip.forward, "k0": strip.k0,
        "market_total_variance": market_variance, "gromm_total_variance": model_variance,
        "gromm_full_strike_total_variance": full_variance, "selected_strikes": len(strip.strikes),
        "minimum_strike": float(strip.strikes[0]), "maximum_strike": float(strip.strikes[-1]),
        "put_zero_bid_stop": strip.put_zero_bid_stop, "call_zero_bid_stop": strip.call_zero_bid_stop,
        "model_within_horizon": within_horizon,
        "outside_interior_support_excess_total_variance": outside_excess,
        "invalid_or_absent_call_books": int(np.sum(~np.isfinite(cb) | ~np.isfinite(ca) | (cb < 0) | (ca < cb))),
        "invalid_or_absent_put_books": int(np.sum(~np.isfinite(pb) | ~np.isfinite(pa) | (pb < 0) | (pa < pb))),
    }
    constituents = [dict(date=trading_date.isoformat(), expiry=settlement.isoformat(), is_am=is_am,
                         strike=float(k), delta_strike=float(dk), market_price=float(q),
                         model_price=float(mq) if np.isfinite(mq) else None,
                         normalized_strike=float(k / strip.forward),
                         outside_interior_support=bool(k / strip.forward < support[1] or k / strip.forward > support[-2]),
                         side="both" if k == strip.k0 else "put" if k < strip.k0 else "call")
                    for k, dk, q, mq in zip(strip.strikes, strip.delta_strikes, strip.prices, model_prices, strict=True)]
    return summary, constituents


def _model_inputs(fit_label: str, training_run_directory: Path | None) -> tuple[Path, Path, str, np.ndarray]:
    """Load a retained paper fit or a completed, dataset-bound wider-grid fit."""
    if training_run_directory is None:
        verify_inp(fit_label in FITS, lambda: f"Unknown fit {fit_label}")
        return DATASET_PATH, FITS[fit_label].checkpoint, f"Gromm-v1 ({fit_label})", load_raw_grids(fit_label)
    dataset = training_run_directory / "market.npz"
    report = json.loads((training_run_directory / "fit/report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Wider-grid training has not completed")
    checkpoint = Path(report["checkpoint"])
    dataset_hash = _sha256(dataset)
    with np.load(checkpoint, allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "daily25-jax-weighted-fit-v1" or str(saved["mode"].item()) != "direct":
            raise ValueError("Unexpected wider-grid checkpoint architecture")
        if str(saved["dataset_sha256"].item()) != dataset_hash or report["dataset_sha256"] != dataset_hash:
            raise ValueError("Checkpoint does not belong to this wider-grid dataset")
        raw = np.asarray(saved["raw"], dtype=np.float64)
        label = str(saved["model_label"].item())
    if not np.all(np.isfinite(raw)):
        raise ValueError("Checkpoint contains non-finite parameters")
    return dataset, checkpoint, label, raw


def run_comparison(*, output_directory: Path, fit_label: str = "full", limit: int | None = None,
                   training_run_directory: Path | None = None) -> dict[str, object]:
    """Write historical levels, constituent audit, metrics, and PDF/PNG figures."""
    verify_inp(limit is None or limit > 0, lambda: "limit must be positive")
    dataset_path, checkpoint_path, model_label, raw_grids = _model_inputs(fit_label, training_run_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jax.config.update("jax_enable_x64", True)
    with np.load(dataset_path, allow_pickle=False) as archive:
        dates = [date.fromisoformat(str(value)) for value in archive["dates"]]
        supports = np.asarray(archive["supports"], dtype=np.float64)
        model_times = np.asarray(archive["model_times_y"], dtype=np.float64)
        if not np.allclose(model_times, np.arange(1, 26) / DAYS_PER_YEAR, rtol=1e-7, atol=0):
            raise ValueError("Unexpected SDLV maturity grid")
        if not np.isclose(float(archive["backbone_annual_vol"]), BACKGROUND_ANNUAL_VOL):
            raise ValueError("Unexpected smoothing volatility")
    if raw_grids.shape != (len(dates), 25, supports.shape[1] - 2):
        raise ValueError("Checkpoint shape does not match its dataset")
    dlvsq = decode_dlvsq(raw_grids)
    if limit is not None:
        dates, supports, dlvsq = dates[:limit], supports[:limit], dlvsq[:limit]
    sessions = np.asarray(get_business_days(dates[0] - timedelta(days=10), dates[-1] + timedelta(days=65)), dtype="datetime64[D]")
    with jax.default_device(jax.devices("cpu")[0]):
        decode = jax.jit(jax.vmap(lambda s, v: decode_density_tower(s, v, dt=1 / DAYS_PER_YEAR)))
        towers = np.asarray(decode(jnp.asarray(supports), jnp.asarray(dlvsq)))
    mass_error = float(np.max(np.abs(towers.sum(axis=-1) - 1)))
    forward_error = float(np.max(np.abs(np.einsum("dti,di->dt", towers, supports) - 1)))
    if towers.min() < -1e-14 or mass_error > 1e-10 or forward_error > 1e-10:
        raise RuntimeError("Decoded densities violate positivity/mass/forward invariants")
    raw_cache = env.locations("widb", default_fmt=SubDir.PANDAS_PARQUET).raw
    official = load_cboe_vix(dates[0], dates[-1]).select("date", pl.col("close").alias("cboe_vix"))
    records, terms, constituents, input_files = [], [], [], []
    progress_path = output_directory / "progress.jsonl"
    for index, (day, support, tower) in enumerate(zip(dates, supports, towers, strict=True)):
        business_days = calendar_business_maturity(day, sessions)
        density = interpolate_density(tower, business_days)
        total = log_contract_total_variance(support, density, business_time_years=business_days / DAYS_PER_YEAR)
        # Flat annual-clock convention is a sensitivity, not the primary date-aware result.
        flat_days = 30 / 365 * DAYS_PER_YEAR
        flat_total = log_contract_total_variance(support, interpolate_density(tower, flat_days), business_time_years=30 / 365)
        record = dict(date=day, gromm_continuous=100 * np.sqrt(total * 365 / 30),
                      gromm_flat_clock=100 * np.sqrt(flat_total * 365 / 30),
                      business_days_30_calendar=business_days, gromm_discrete=None,
                      market_discrete=None, gromm_friday_continuous=None,
                      outside_support_excess_annual_variance=None,
                      discrete_status="unavailable", discrete_reason=None)
        option_name, zero_name = f"options/108105/{day.isoformat()}", f"zero_curve/{day.isoformat()}"
        raw, zero = raw_cache.read(option_name), raw_cache.read(zero_name)
        if raw is None or zero is None:
            record["discrete_reason"] = "missing cached raw options or zero curve; no download attempted"
        else:
            # Keep zero bids: never use the paper's prefiltered/NMS-limited training book here.
            raw_hash = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
            zero_hash = hashlib.sha256(pd.util.hash_pandas_object(zero, index=True).values.tobytes()).hexdigest()
            input_files.append(dict(date=day.isoformat(), options_cache_key=option_name,
                                    zero_cache_key=zero_name, options_content_sha256=raw_hash,
                                    zero_content_sha256=zero_hash))
            try:
                expiries = eligible_expiries(day, raw, sessions)
                near = [entry for entry in expiries if entry[2] <= 30]
                far = [entry for entry in expiries if entry[2] > 30]
                if not near or not far:
                    raise ValueError("Missing eligible bracketing Friday expiries")
                pair = [near[-1], far[0]]
                summaries = []
                for expiry in pair:
                    summary, selected = _term(trading_date=day, expiry=expiry, raw=raw, zero=zero,
                                              support=support, tower=tower)
                    summaries.append(summary)
                    terms.append(summary)
                    constituents.extend(selected)
                n, f = summaries
                record["near_expiry"], record["next_expiry"] = n["expiry"], f["expiry"]
                record["market_discrete"] = constant_maturity_vix(near_days=n["calendar_days"], next_days=f["calendar_days"],
                    near_total_variance=n["market_total_variance"], next_total_variance=f["market_total_variance"])
                if n["model_within_horizon"] and f["model_within_horizon"]:
                    for field, key in (("gromm_discrete", "gromm_total_variance"), ("gromm_friday_continuous", "gromm_full_strike_total_variance")):
                        record[field] = constant_maturity_vix(near_days=n["calendar_days"], next_days=f["calendar_days"],
                            near_total_variance=n[key], next_total_variance=f[key])
                    record["discrete_status"] = "ok"
                    alpha = (30 - n["calendar_days"]) / (f["calendar_days"] - n["calendar_days"])
                    record["outside_support_excess_annual_variance"] = ((1 - alpha) * n["outside_interior_support_excess_total_variance"]
                        + alpha * f["outside_interior_support_excess_total_variance"]) * 365 / 30
                else:
                    record["discrete_reason"] = "bracketing Friday expiry exceeds trained 25-business-day horizon"
            except ValueError as error:
                record["discrete_reason"] = str(error)
        records.append(record)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(dates):
            elapsed = time.monotonic() - started
            progress = dict(completed=index + 1, total=len(dates), elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed * (len(dates) - index - 1) / (index + 1))
            line = json.dumps(progress)
            print(line, flush=True)
            with progress_path.open("a") as stream:
                stream.write(line + "\n")
    frame = pl.from_dicts(records, infer_schema_length=None).join(official, on="date", how="left").sort("date")
    if frame["cboe_vix"].null_count():
        raise RuntimeError("Official VIX is missing fitted-state dates")
    columns = ("gromm_continuous", "gromm_discrete", "market_discrete", "gromm_friday_continuous", "gromm_flat_clock")
    frame = frame.with_columns(*[(pl.col(name) - pl.col("cboe_vix")).alias(name + "_error") for name in columns])
    metrics = {name: _metrics(frame[name].to_numpy(), frame["cboe_vix"].to_numpy()) for name in columns}
    metrics["gromm_discrete_vs_market_discrete"] = _metrics(frame["gromm_discrete"].to_numpy(), frame["market_discrete"].to_numpy())
    common = frame.filter(pl.col("gromm_discrete").is_not_null())
    common_metrics = {name: _metrics(common[name].to_numpy(), common["cboe_vix"].to_numpy()) for name in columns}
    excess_variance = np.sum((common["gromm_discrete"].to_numpy() / 100)**2 - (common["market_discrete"].to_numpy() / 100)**2)
    outside_variance = float(common["outside_support_excess_annual_variance"].sum())
    yearly = {str(year): {name: _metrics(part[name].to_numpy(), part["cboe_vix"].to_numpy()) for name in columns}
              for year in sorted({d.year for d in dates})
              for part in [frame.filter(pl.col("date").dt.year() == year)]}

    # Independent numerical OTM option integration verifies the exact log-contract result.
    checks = []
    for index in sorted({0, len(dates) - 1, int(frame["gromm_continuous"].arg_max()), int(frame["gromm_continuous"].arg_min())}):
        support = supports[index]
        bdays = calendar_business_maturity(dates[index], sessions)
        density = interpolate_density(towers[index], bdays)
        sd = BACKGROUND_ANNUAL_VOL * np.sqrt(bdays / DAYS_PER_YEAR)
        def integrand(log_strike: float) -> float:
            strike = np.exp(log_strike)
            calls, puts = mixture_prices(support, density, np.asarray([strike]), business_days=bdays)
            return float((puts if log_strike < 0 else calls)[0] / strike)
        value = 2 * (quad(integrand, np.log(support.min()) - 12 * sd, 0, epsabs=1e-11)[0]
                     + quad(integrand, 0, np.log(support.max()) + 12 * sd, epsabs=1e-11)[0])
        numeric_vix = 100 * np.sqrt(value * 365 / 30)
        error = abs(numeric_vix - float(frame["gromm_continuous"][index]))
        if error > 1e-6:
            raise RuntimeError(f"OTM integral/log-contract mismatch {error}")
        checks.append(dict(date=dates[index].isoformat(), absolute_vix_difference=error))
    report = dict(
        schema="sdlv-paper-vix-comparison-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        fit_label=fit_label if training_run_directory is None else model_label, model_label=model_label,
        checkpoint=str(checkpoint_path), checkpoint_sha256=_sha256(checkpoint_path),
        dataset=str(dataset_path), dataset_sha256=_sha256(dataset_path), code_sha256=_sha256(Path(__file__)),
        formula_sha256=_sha256(Path(__file__).parents[2] / "vixdata/formula.py"),
        state_date_range=[dates[0].isoformat(), dates[-1].isoformat()], observations=len(dates),
        actual_vix_cache=str(env.locations("vixdata", fmt_raw=SubDir.POLARS_PARQUET).raw.path),
        raw_option_cache=str(raw_cache.path), actual_source="https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        official_series_sha256=hashlib.sha256(official.write_csv().encode()).hexdigest(),
        methodology=METHODOLOGY_URL, metrics=metrics, common_sample_metrics=common_metrics, yearly_metrics=yearly,
        tail_diagnostic=dict(
            observations=common.height,
            fraction_net_model_market_variance_gap_outside_interior_support=float(outside_variance / excess_variance) if excess_variance != 0 else None,
            interpretation="Sum of calendar-weighted model-minus-market variance contributions outside each date's 200 interior model nodes, divided by the total gap on the same complete-date sample; may exceed one if inside contributions offset. This is a broad support-bound diagnostic, not the expiry-specific NMS fit mask.",
        ),
        reference_date=frame.filter(pl.col("date") == date(2025, 8, 29)).with_columns(pl.col("date").cast(pl.String)).to_dicts(),
        unavailable_reasons=frame.filter(pl.col("discrete_status") != "ok").group_by("discrete_reason").len().to_dicts(),
        validation=dict(maximum_mass_error=mass_error, maximum_forward_error=forward_error,
                        minimum_probability=float(towers.min()), independent_otm_quadrature=checks),
        conventions=[
            "Primary continuous series: interpolate daily fitted densities to 30 calendar days using actual NYSE session dates; annualize total variance by 365/30.",
            "Cboe discrete formula: actual unfiltered OptionMetrics SPX (secid 108105) Friday/holiday-adjusted books; AM standard third-Friday, PM other Fridays; two-zero-bid cutoff; exact deltaK and F/K0 correction.",
            "Index valuation timestamp 16:15 New York (Cboe RTH closing tick); AM maturity 09:30, PM 16:00; calendar minutes/525600. Archived books are EOD, not guaranteed synchronous. SDLV AM business maturity follows its existing 0.8-day subtraction.",
            "Discount factors use cached OptionMetrics zero curves with the repository's linear rate interpolation, not the exact Cboe Treasury spline inputs.",
            "Model normalized prices use the observed parity-implied forward; market and model use identical selected contracts, K0 and discount factors.",
            "No maturity extrapolation; unavailable model strips are reported, while the market-formula reconstruction can still be calculated.",
            "Not an exact reproduction of the published VIX tick: OptionMetrics EOD versus Cboe timestamp/venue, rate-source, settlement and intraday index filtering differences remain.",
            "Historical contemporaneous fitted-surface comparison, not an out-of-sample forecast or a new P/Q dynamics audit; no VIX fitting or rescaling.",
        ], elapsed_seconds=time.monotonic() - started,
    )
    frame.write_parquet(output_directory / "vix_comparison.parquet")
    frame.write_csv(output_directory / "vix_comparison.csv")
    pl.from_dicts(terms, infer_schema_length=None).write_parquet(output_directory / "expiry_audit.parquet")
    pl.from_dicts(constituents, infer_schema_length=None).write_parquet(output_directory / "selected_contracts.parquet")
    (output_directory / "input_manifest.json").write_text(json.dumps(input_files, indent=2) + "\n")
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _plot(frame, report, output_directory)
    print(json.dumps(metrics, indent=2), flush=True)
    return report


def _plot(frame: pl.DataFrame, report: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    dates = frame["date"].to_list()
    axes[0].plot(dates, frame["cboe_vix"], color="black", lw=1, label="Official Cboe VIX close")
    axes[0].plot(dates, frame["gromm_continuous"], color="#2166ac", lw=0.8, label="GROMM/SDLV: full-strike 30-day")
    axes[0].plot(dates, frame["gromm_discrete"], color="#d6604d", lw=0.8, label="GROMM/SDLV: Cboe discrete formula")
    axes[0].set_title(f"{report['model_label']} — historical VIX comparison\nMarket-selected Cboe strip; no VIX fitting or rescaling")
    axes[0].set_ylabel("Volatility points")
    axes[0].legend(fontsize=9, frameon=False)
    for name, color, label in (("gromm_continuous_error", "#2166ac", "Full-strike model − official"),
                               ("gromm_discrete_error", "#d6604d", "Discrete-formula model − official")):
        axes[1].plot(dates, frame[name], color=color, lw=0.8, label=label)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_ylabel("Difference (vol points)")
    axes[1].legend(fontsize=9, frameon=False)
    axes[2].plot(dates, frame["market_discrete_error"], color="#756bb1", lw=0.8,
                 label="Cboe formula on archived market mids − official VIX")
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[2].set_ylabel("Data reconstruction gap")
    axes[2].legend(fontsize=9, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "vix_comparison.png", dpi=170)
    figure.savefig(output / "vix_comparison.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fit-label", choices=tuple(FITS), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--training-run-directory", type=Path)
    parser.add_argument("--wait-for-training-unit")
    parser.add_argument("--filtered-data-directory", type=Path)
    args = parser.parse_args()
    if args.wait_for_training_unit:
        if args.training_run_directory is None or not args.wait_for_training_unit.startswith("quant-data-gromm-v1-43-full-"):
            raise ValueError("Waiting requires this project's wider-grid training unit and run directory")
        started = time.monotonic()
        while True:
            state = subprocess.run(["systemctl", "--user", "show", "--property=ActiveState", "--value", args.wait_for_training_unit],
                                   capture_output=True, text=True, timeout=15)
            if state.stdout.strip() not in {"active", "activating", "deactivating"}:
                break
            print(json.dumps(dict(stage="waiting_for_completed_training", elapsed_seconds=time.monotonic() - started)), flush=True)
            time.sleep(60)
        checks = json.loads(subprocess.run(["gpuctl", "resources", "--json"], capture_output=True,
                                          text=True, check=True, timeout=15).stdout)["checks"]
        if not checks or any(check["health"] != "OK" for check in checks):
            raise RuntimeError("Resource health is not all OK; VIX evaluation not launched")
    run_comparison(output_directory=args.output_directory, fit_label=args.fit_label, limit=args.limit,
                   training_run_directory=args.training_run_directory)
    if args.filtered_data_directory is not None:
        from quantdata.sdlv.daily.plot_vix import run as plot_comparison
        plot_comparison(model_directory=args.output_directory, filtered_directory=args.filtered_data_directory,
                        output_directory=args.output_directory / "four-series")


if __name__ == "__main__":
    main()
