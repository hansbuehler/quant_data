"""
Single-option nonlinear Kalman filter for intraday log implied volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quantdata.biz_calendar import DAYS_PER_YEAR

from .constants import MAX_VOL, MIN_VOL, SINGLE_OPTION_BID_ASK_95_Z
from .data import single_option_frame
from .iv_math import finite_difference_jacobian, option_price_from_logvol, realized_scale, sanitize_interval_days, surface_z


@dataclass(frozen=True)
class SingleOptionKalmanConfig:
    rho: float = 0.0
    drop_intrinsic: bool = True
    process_scale: float | None = None
    init_log_vol: float | None = None
    p0: float = 0.10**2
    obs_half_spread_z: float = SINGLE_OPTION_BID_ASK_95_Z
    obs_floor_scale: float = 0.25


@dataclass(frozen=True)
class SingleOptionKalmanResult:
    frame: pl.DataFrame
    summary: dict[str, float | int | str]


def run_single_option_filter(
    df_expiry: pl.DataFrame,
    opt_ticker: str,
    config: SingleOptionKalmanConfig | None = None,
    **overrides,
) -> SingleOptionKalmanResult:
    if config is None:
        config = SingleOptionKalmanConfig(**overrides)
    elif overrides:
        raise TypeError("Pass either config or keyword overrides, not both.")

    data = single_option_frame(df_expiry, opt_ticker)
    df_opt = data["df"]
    n = len(df_opt)
    strike = data["strike"]
    is_call = data["is_call"]
    spot = data["spot"]
    sqrt_t = data["sqrt_t"]
    mid_price = df_opt["mid_price"].to_numpy()
    trade_price = df_opt["trade_price"].to_numpy()
    dt_days = sanitize_interval_days(df_opt["dt_d"].to_numpy())
    spot_dt_days = sanitize_interval_days(df_opt["spot_quote_dt_d"].to_numpy())
    spot_ret = np.nan_to_num(df_opt["spot_mid_ret"].to_numpy(), nan=0.0)
    is_intrinsic = df_opt["is_intrinsic"].to_numpy().astype(bool)
    half_spread = np.maximum(df_opt["half_spread"].to_numpy(), 0.5 * df_opt["opt_tick_size"].to_numpy())

    direct_vol = data["direct_mid_vol"]
    stored_mid_vol = data["stored_mid_vol"]

    valid_obs = np.isfinite(mid_price) & np.isfinite(spot) & np.isfinite(sqrt_t)
    if config.drop_intrinsic:
        valid_obs &= ~is_intrinsic

    baseline_mask = np.isfinite(direct_vol) & (direct_vol > 0.0)
    stored_mask = np.isfinite(stored_mid_vol) & (stored_mid_vol > 0.0)
    anchor_vol = direct_vol[baseline_mask]
    if anchor_vol.size == 0:
        anchor_vol = stored_mid_vol[stored_mask]
    if anchor_vol.size == 0:
        anchor_vol = np.array([0.2], dtype=float)

    init_log_vol = config.init_log_vol
    if init_log_vol is None:
        init_log_vol = float(np.log(np.nanmedian(anchor_vol[: min(anchor_vol.size, 20)])))

    process_scale = config.process_scale
    if process_scale is None:
        if baseline_mask.sum() >= 3:
            process_scale = realized_scale(np.log(direct_vol[baseline_mask]), dt_days[baseline_mask], fallback=0.35)
        elif stored_mask.sum() >= 3:
            process_scale = realized_scale(np.log(stored_mid_vol[stored_mask]), dt_days[stored_mask], fallback=0.35)
        else:
            process_scale = 0.35

    rho = float(np.clip(config.rho, -0.999, 0.999))
    obs_half_spread_z = float(max(config.obs_half_spread_z, 1e-6))
    obs_sigma_scale = 1.0 / obs_half_spread_z
    finite_half_spread = half_spread[np.isfinite(half_spread)]
    median_half_spread = float(np.nanmedian(finite_half_spread)) if finite_half_spread.size else np.nan
    obs_floor = max(float(np.nan_to_num(median_half_spread, nan=0.0)) * config.obs_floor_scale * obs_sigma_scale, 1e-4)

    x = float(init_log_vol)
    p = float(config.p0)

    x_filt = np.full(n, np.nan, dtype=float)
    x_prior_hist = np.full(n, np.nan, dtype=float)
    prev_x_filt = np.full(n, np.nan, dtype=float)
    fitted_price = np.full(n, np.nan, dtype=float)
    prior_price = np.full(n, np.nan, dtype=float)
    process_var = np.full(n, np.nan, dtype=float)
    prior_var = np.full(n, np.nan, dtype=float)
    filtered_var = np.full(n, np.nan, dtype=float)
    kalman_gain = np.full(n, np.nan, dtype=float)
    obs_sensitivity = np.full(n, np.nan, dtype=float)
    obs_sigma = np.full(n, np.nan, dtype=float)
    innovation = np.full(n, np.nan, dtype=float)
    log_likelihood = np.full(n, np.nan, dtype=float)

    for i in range(n):
        dt_y = max(float(dt_days[i]) / DAYS_PER_YEAR, 1e-12)
        vol_prev = float(np.exp(np.clip(x, np.log(MIN_VOL), np.log(MAX_VOL))))
        prev_x_filt[i] = x
        spot_shock = spot_ret[i] / np.sqrt(max(float(spot_dt_days[i]) / DAYS_PER_YEAR, 1e-12)) / max(vol_prev, MIN_VOL)
        x_mean_shift = process_scale * np.sqrt(dt_y) * rho * spot_shock
        q = (process_scale**2) * dt_y * (1.0 - rho * rho)
        x_prior = float(np.clip(x + x_mean_shift, np.log(MIN_VOL), np.log(MAX_VOL)))
        p_prior = float(max(p + q, 1e-10))
        process_var[i] = q
        prior_var[i] = p_prior
        x_prior_hist[i] = x_prior
        prior_price[i] = float(option_price_from_logvol(x_prior, spot[i], strike, sqrt_t[i], is_call))

        if valid_obs[i]:

            def obs_fn(state):
                return np.array([option_price_from_logvol(state[0], spot[i], strike, sqrt_t[i], is_call)], dtype=float)

            h, h_jac = finite_difference_jacobian(obs_fn, np.array([x_prior], dtype=float))
            h0 = float(h_jac[0, 0])
            gamma = float(max(half_spread[i] * obs_sigma_scale, obs_floor))
            s = float(max(h0 * h0 * p_prior + gamma * gamma, 1e-12))
            k = float(p_prior * h0 / s)
            innovation_i = float(mid_price[i] - h[0])
            x = float(np.clip(x_prior + k * innovation_i, np.log(MIN_VOL), np.log(MAX_VOL)))
            p = float(max((1.0 - k * h0) * p_prior, 1e-10))
            obs_sensitivity[i] = h0
            obs_sigma[i] = gamma
            kalman_gain[i] = k
            innovation[i] = innovation_i
            log_likelihood[i] = -0.5 * (np.log(2.0 * np.pi * s) + innovation_i * innovation_i / s)
        else:
            x = x_prior
            p = p_prior

        x_filt[i] = x
        filtered_var[i] = p
        fitted_price[i] = float(option_price_from_logvol(x, spot[i], strike, sqrt_t[i], is_call))

    filtered_vol = np.exp(x_filt)
    kalman_prior_vol = np.exp(x_prior_hist)
    kalman_prev_filtered_vol = np.exp(prev_x_filt)
    previous_stored_mid_vol = np.roll(stored_mid_vol, 1)
    previous_stored_mid_vol[0] = np.nan
    direct_minus_previous_stored_mid_vol = direct_vol - previous_stored_mid_vol
    direct_minus_kalman_prior_vol = direct_vol - kalman_prior_vol
    direct_minus_filtered_vol = direct_vol - filtered_vol

    direct_mask = np.isfinite(direct_vol) & np.isfinite(filtered_vol)
    stored_mask = np.isfinite(stored_mid_vol) & np.isfinite(filtered_vol)
    price_mask = np.isfinite(mid_price) & np.isfinite(fitted_price)

    rmse_direct = float(np.sqrt(np.mean((filtered_vol[direct_mask] - direct_vol[direct_mask]) ** 2))) if direct_mask.any() else np.nan
    rmse_stored = float(np.sqrt(np.mean((filtered_vol[stored_mask] - stored_mid_vol[stored_mask]) ** 2))) if stored_mask.any() else np.nan
    rmse_price = float(np.sqrt(np.mean((fitted_price[price_mask] - mid_price[price_mask]) ** 2))) if price_mask.any() else np.nan

    strike_arr = np.full(n, float(strike), dtype=float)
    mid_vega = df_opt["mid_vega"].to_numpy() if "mid_vega" in df_opt.columns else np.full(n, np.nan, dtype=float)
    moneyness = surface_z(strike_arr, spot)
    frame = pl.DataFrame(
        {
            "opt_ticker": [opt_ticker] * n,
            "t_d": df_opt["t_d"].to_numpy(),
            "mid_price": mid_price,
            "trade_price": trade_price,
            "direct_mid_vol": direct_vol,
            "stored_mid_vol": stored_mid_vol,
            "filtered_vol": filtered_vol,
            "kalman_prior_vol": kalman_prior_vol,
            "kalman_prev_filtered_vol": kalman_prev_filtered_vol,
            "previous_stored_mid_vol": previous_stored_mid_vol,
            "direct_minus_previous_stored_mid_vol": direct_minus_previous_stored_mid_vol,
            "direct_minus_kalman_prior_vol": direct_minus_kalman_prior_vol,
            "direct_minus_filtered_vol": direct_minus_filtered_vol,
            "filtered_price": fitted_price,
            "kalman_prior_price": prior_price,
            "kalman_process_var": process_var,
            "kalman_prior_var": prior_var,
            "kalman_filtered_var": filtered_var,
            "kalman_gain": kalman_gain,
            "kalman_obs_sensitivity": obs_sensitivity,
            "kalman_obs_sigma": obs_sigma,
            "innovation": innovation,
            "is_intrinsic": is_intrinsic,
            "spot_mid_price": spot,
            "strike": strike_arr,
            "mid_vega": mid_vega,
            "moneyness": moneyness,
            "abs_moneyness": np.abs(moneyness),
        }
    )
    summary = {
        "opt_ticker": opt_ticker,
        "rho": rho,
        "process_scale": float(process_scale),
        "obs_half_spread_z": float(obs_half_spread_z),
        "obs_sigma_scale": float(obs_sigma_scale),
        "obs_floor": float(obs_floor),
        "log_likelihood": float(np.nansum(log_likelihood)),
        "rmse_direct": rmse_direct,
        "rmse_stored": rmse_stored,
        "rmse_price": rmse_price,
        "n_obs": int(n),
        "n_updates": int(valid_obs.sum()),
    }
    return SingleOptionKalmanResult(frame=frame, summary=summary)


def sweep_rho(
    df_expiry: pl.DataFrame,
    opt_tickers: list[str],
    rho_grid,
    *,
    drop_intrinsic: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for opt_ticker in opt_tickers:
        for rho in rho_grid:
            result = run_single_option_filter(
                df_expiry,
                opt_ticker,
                rho=float(rho),
                drop_intrinsic=drop_intrinsic,
            )
            rows.append(result.summary)

    detail = pl.DataFrame(rows)
    aggregate = (
        detail.group_by("rho")
        .agg(
            [
                pl.col("log_likelihood").sum().alias("log_likelihood_sum"),
                pl.col("rmse_direct").mean().alias("rmse_direct_mean"),
                pl.col("rmse_stored").mean().alias("rmse_stored_mean"),
                pl.col("rmse_price").mean().alias("rmse_price_mean"),
            ]
        )
        .sort("log_likelihood_sum", descending=True)
    )
    return detail, aggregate
