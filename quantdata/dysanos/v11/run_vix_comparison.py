"""Compare Cboe VIX closes with model-free variance from daily DYSANOS states.

The DYSANOS value is the continuous-strike model-free variance integral of
the decoded normalized call surface.  It is therefore a clean mark-to-market
test of the fitted surface, not a simulation or AR(1) forecast test.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from cdxcore.verbose import Context

from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.dysanos.v11.run_rolling_spot_feature_loading import _load_model
from quantdata.vixdata import load_cboe_vix
from quantdata.widb.options_filter import OptionsFilter

_TARGET_VIX_YEARS = 30.0 / 365.0
_RAW_OPTION_FILTER = OptionsFilter(
    max_num_options=100_000,
    max_business_days_to_expiry=70,
    max_nms=(-100.0, 100.0),
)


def model_free_variance(
    model: object,
    states: torch.Tensor,
    *,
    expiry_y: float,
    num_strikes_per_side: int = 801,
    minimum_strike: float = 0.0025,
    maximum_strike: float = 6.0,
) -> torch.Tensor:
    """Return continuous-strike fair variance from normalized decoded calls.

    The forward is one in the DYSANOS normalized chart.  Hence

    ``2 / T * (integral(P(K)/K^2, 0, 1) + integral(C(K)/K^2, 1, inf))``

    is evaluated numerically.  Finite tails are conservatively truncated only
    once the normalized put/call values are numerically zero.
    """
    if expiry_y <= 0.0 or num_strikes_per_side < 3:
        raise ValueError("expiry_y must be positive and each strike side needs at least 3 nodes")
    if not 0.0 < minimum_strike < 1.0 < maximum_strike:
        raise ValueError("strikes must satisfy 0 < minimum < 1 < maximum")
    if states.ndim != 2:
        raise ValueError("states must have shape (date, state)")
    geometry = model.config.geometry
    raw = model.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    device, dtype = states.device, next(model.parameters()).dtype
    left = torch.exp(
        torch.linspace(
            float(np.log(minimum_strike)),
            0.0,
            num_strikes_per_side,
            dtype=dtype,
            device=device,
        )
    )
    right = torch.exp(
        torch.linspace(
            0.0,
            float(np.log(maximum_strike)),
            num_strikes_per_side,
            dtype=dtype,
            device=device,
        )
    )
    strikes = torch.cat((left, right[1:])).unsqueeze(0).expand(len(states), -1)
    expiries = torch.full_like(strikes, expiry_y)
    calls = model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=expiries,
        strikes=strikes,
    )
    left_count = len(left)
    puts = torch.clamp(calls[:, :left_count] + left - 1.0, min=0.0)
    left_integral = torch.trapezoid(puts / left.square(), left, dim=1)
    right_calls = calls[:, left_count - 1 :]
    right_integral = torch.trapezoid(right_calls / right.square(), right, dim=1)
    return 2.0 * (left_integral + right_integral) / expiry_y


def _select_vix_strikes(
    *,
    strikes: np.ndarray,
    call_bids: np.ndarray,
    put_bids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Select the Cboe-style OTM strike set using the observed bid cut-off.

    The selection rule belongs to the market snapshot, not the fitted model:
    the same selected strikes are repriced under SANOS and from raw midpoint
    quotes.  One zero bid is skipped; a second consecutive zero bid terminates
    the respective wing, as in the published VIX methodology.
    """
    if strikes.ndim != call_bids.ndim != put_bids.ndim != 1:
        raise ValueError("VIX strike inputs must be one-dimensional")
    if not (len(strikes) == len(call_bids) == len(put_bids)):
        raise ValueError("VIX strike inputs have inconsistent lengths")
    if not np.all(np.diff(strikes) > 0.0):
        raise ValueError("VIX strikes must be strictly increasing")
    k0 = int(np.searchsorted(strikes, 1.0, side="right") - 1)
    if k0 < 0 or k0 >= len(strikes):
        raise ValueError("normalized strike grid does not contain the forward")

    selected: list[int] = [k0]
    zero_bids = 0
    for index in range(k0 - 1, -1, -1):
        bid = put_bids[index]
        if not np.isfinite(bid) or bid <= 0.0:
            zero_bids += 1
            if zero_bids >= 2:
                break
            continue
        zero_bids = 0
        selected.append(index)
    zero_bids = 0
    for index in range(k0 + 1, len(strikes)):
        bid = call_bids[index]
        if not np.isfinite(bid) or bid <= 0.0:
            zero_bids += 1
            if zero_bids >= 2:
                break
            continue
        zero_bids = 0
        selected.append(index)
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) < 3:
        raise ValueError("VIX selection retained fewer than three strikes")
    return result, int(np.searchsorted(result, k0))


def _cboe_term_variance(
    *,
    expiry_y: float,
    strikes: np.ndarray,
    calls: np.ndarray,
    puts: np.ndarray,
    selected: np.ndarray,
    selected_k0_index: int,
) -> float:
    """Compute one normalized-forward discrete VIX variance term."""
    if expiry_y <= 0.0:
        raise ValueError("expiry must be positive")
    used_strikes = strikes[selected]
    option_values = np.where(
        np.arange(len(selected)) < selected_k0_index,
        puts[selected],
        calls[selected],
    )
    option_values[selected_k0_index] = 0.5 * (
        calls[selected[selected_k0_index]] + puts[selected[selected_k0_index]]
    )
    delta_strikes = np.empty_like(used_strikes)
    delta_strikes[0] = used_strikes[1] - used_strikes[0]
    delta_strikes[-1] = used_strikes[-1] - used_strikes[-2]
    delta_strikes[1:-1] = 0.5 * (used_strikes[2:] - used_strikes[:-2])
    variance = (
        2.0 / expiry_y * np.sum(delta_strikes * option_values / np.square(used_strikes))
        - np.square(1.0 / strikes[selected[selected_k0_index]] - 1.0) / expiry_y
    )
    return float(max(variance, 0.0))


def _term_variance_for_mask(
    *,
    expiry_y: float,
    strikes: np.ndarray,
    call_bids: np.ndarray,
    call_asks: np.ndarray,
    put_bids: np.ndarray,
    put_asks: np.ndarray,
    model_calls: np.ndarray,
    keep: np.ndarray,
) -> tuple[float, float]:
    """Return SANOS and midpoint variance on one explicitly retained book."""
    finite_quotes = (
        np.isfinite(call_bids)
        & np.isfinite(call_asks)
        & np.isfinite(put_bids)
        & np.isfinite(put_asks)
    )
    valid = keep & finite_quotes
    selected, selected_k0 = _select_vix_strikes(
        strikes=strikes[valid],
        call_bids=call_bids[valid],
        put_bids=put_bids[valid],
    )
    used_strikes = strikes[valid]
    used_model_calls = model_calls[valid]
    used_model_puts = used_model_calls + used_strikes - 1.0
    raw_calls = 0.5 * (call_bids[valid] + call_asks[valid])
    raw_puts = 0.5 * (put_bids[valid] + put_asks[valid])
    return (
        _cboe_term_variance(
            expiry_y=expiry_y,
            strikes=used_strikes,
            calls=used_model_calls,
            puts=used_model_puts,
            selected=selected,
            selected_k0_index=selected_k0,
        ),
        _cboe_term_variance(
            expiry_y=expiry_y,
            strikes=used_strikes,
            calls=raw_calls,
            puts=raw_puts,
            selected=selected,
            selected_k0_index=selected_k0,
        ),
    )


def _bracketing_expiry_indices(
    *,
    expiries_y: np.ndarray,
    is_am: np.ndarray,
) -> tuple[int, int]:
    """Return the nearest PM-listed expiries immediately around 30 days."""
    eligible = np.flatnonzero(~is_am)
    before = eligible[expiries_y[eligible] < _TARGET_VIX_YEARS]
    after = eligible[expiries_y[eligible] >= _TARGET_VIX_YEARS]
    if len(before) == 0 or len(after) == 0:
        raise ValueError("no PM-listed expiries bracket the 30-day VIX horizon")
    return int(before[-1]), int(after[0])


def listed_expiry_vix(
    model: object,
    state: torch.Tensor,
    *,
    trading_date: date,
) -> tuple[float, float, float, float]:
    """Price a Cboe-style 30-day VIX from the two listed PM expiry books.

    Returns `(SANOS full-book, raw full-book, SANOS fitted-book, raw
    fitted-book)`.  The fitted book uses `market.model_masks`, the exact
    bid/ask and static-arbitrage filter retained by the source SANOS fit.
    Every value uses Cboe total-variance interpolation but daily dates rather
    than Cboe's precise settlement-minute convention.
    """
    payload = smooth_interpolated_market(
        "SPX",
        trading_date,
        options_filter=_RAW_OPTION_FILTER,
        verbose=Context.quiet,
    )
    market = payload.get("market")
    if not isinstance(market, dict):
        raise RuntimeError(f"missing raw SPX market for {trading_date}")
    expiries_y = np.asarray(market["expiries"], dtype=np.float64)
    is_am = np.asarray(market["is_am"], dtype=bool)
    lower, upper = _bracketing_expiry_indices(expiries_y=expiries_y, is_am=is_am)

    raw = model.raw_states_from_states(state.unsqueeze(0))
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    sanos_terms: list[float] = []
    raw_terms: list[float] = []
    sanos_fitted_terms: list[float] = []
    raw_fitted_terms: list[float] = []
    term_expiries = expiries_y[[lower, upper]]
    for index in (lower, upper):
        strikes = np.asarray(market["strikes"][index], dtype=np.float64)
        call_bids = np.asarray(market["call_bids"][index], dtype=np.float64)
        call_asks = np.asarray(market["call_asks"][index], dtype=np.float64)
        put_bids = np.asarray(market["put_bids"][index], dtype=np.float64)
        put_asks = np.asarray(market["put_asks"][index], dtype=np.float64)
        expiry = torch.full(
            (1, len(strikes)), float(expiries_y[index]), dtype=state.dtype
        )
        call_tensor = model.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiry,
            strikes=torch.as_tensor(strikes, dtype=state.dtype).unsqueeze(0),
        )
        model_calls = call_tensor.squeeze(0).detach().cpu().numpy()
        full, raw_full = _term_variance_for_mask(
            expiry_y=float(expiries_y[index]),
            strikes=strikes,
            call_bids=call_bids,
            call_asks=call_asks,
            put_bids=put_bids,
            put_asks=put_asks,
            model_calls=model_calls,
            keep=np.ones(len(strikes), dtype=bool),
        )
        fitted, raw_fitted = _term_variance_for_mask(
            expiry_y=float(expiries_y[index]),
            strikes=strikes,
            call_bids=call_bids,
            call_asks=call_asks,
            put_bids=put_bids,
            put_asks=put_asks,
            model_calls=model_calls,
            keep=np.asarray(market["model_masks"][index], dtype=bool),
        )
        sanos_terms.append(full)
        raw_terms.append(raw_full)
        sanos_fitted_terms.append(fitted)
        raw_fitted_terms.append(raw_fitted)

    lower_weight = (term_expiries[1] - _TARGET_VIX_YEARS) / (
        term_expiries[1] - term_expiries[0]
    )
    upper_weight = 1.0 - lower_weight

    def blend(terms: list[float]) -> float:
        total_variance = (
            lower_weight * term_expiries[0] * terms[0]
            + upper_weight * term_expiries[1] * terms[1]
        ) / _TARGET_VIX_YEARS
        return 100.0 * float(np.sqrt(max(total_variance, 0.0)))

    return (
        blend(sanos_terms),
        blend(raw_terms),
        blend(sanos_fitted_terms),
        blend(raw_fitted_terms),
    )


def _load_history(path: Path) -> tuple[list[date], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    states = payload.get("states")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError(f"invalid state history: {path}")
    return [date.fromisoformat(value) for value in dates], states


def _plot(frame: pl.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    dates = frame["date"].to_list()
    axis.plot(dates, frame["cboe_vix"], color="#b2182b", lw=1.2, label="Cboe VIX close")
    axis.plot(
        dates,
        frame["dysanos_vix"],
        color="#2166ac",
        lw=1.2,
        label="DYSANOS fitted-book repricing",
    )
    axis.plot(
        dates,
        frame["wrds_fitted_vix"],
        color="#f46d43",
        lw=0.8,
        alpha=0.8,
        label="WRDS fitted-book midpoint",
    )
    axis.plot(
        dates,
        frame["dysanos_marginal_vix"],
        color="#542788",
        lw=0.8,
        alpha=0.7,
        ls="--",
        label="DYSANOS continuous marginal proxy",
    )
    axis.set_title("30-day model-free implied volatility")
    axis.set_ylabel("annualized volatility (%)")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_comparison(
    *,
    checkpoint: Path,
    pre2025_history: Path,
    forward_history: Path,
    output_directory: Path,
    horizon_calendar_days: int = 30,
    strike_nodes_per_side: int = 801,
    batch_size: int = 32,
) -> dict[str, object]:
    """Write aligned official/model VIX levels and their comparison report."""
    if horizon_calendar_days <= 0 or batch_size <= 0:
        raise ValueError("horizon_calendar_days and batch_size must be positive")
    pre_dates, pre_states = _load_history(pre2025_history)
    forward_dates, forward_states = _load_history(forward_history)
    if pre_dates[-1] >= forward_dates[0]:
        raise ValueError("state histories overlap or are out of order")
    dates = pre_dates + forward_dates
    # The variance integral deliberately reaches far into the left wing.  The
    # otherwise harmless float32 roundoff in the repeated SANOS tridiagonal
    # solves can then be amplified by the 1/K^2 weighting, and even makes the
    # result batch-size dependent.  This is an evaluation diagnostic, so use
    # float64 rather than the float32 training representation.
    states = torch.cat((pre_states, forward_states)).double()
    model = _load_model(checkpoint).double()
    marginal_levels: list[np.ndarray] = []
    expiry_y = horizon_calendar_days / 365.0
    with torch.inference_mode():
        for first in range(0, len(states), batch_size):
            variance = model_free_variance(
                model,
                states[first : first + batch_size].to(
                    dtype=next(model.parameters()).dtype
                ),
                expiry_y=expiry_y,
                num_strikes_per_side=strike_nodes_per_side,
            )
            marginal_levels.append(
                (100.0 * torch.sqrt(torch.clamp(variance, min=0.0))).cpu().numpy()
            )
    listed_levels: list[float] = []
    raw_listed_levels: list[float] = []
    fitted_listed_levels: list[float] = []
    raw_fitted_listed_levels: list[float] = []
    with torch.inference_mode():
        for index, (trading_date, state) in enumerate(zip(dates, states, strict=True)):
            listed, raw_listed, fitted_listed, raw_fitted_listed = listed_expiry_vix(
                model,
                state,
                trading_date=trading_date,
            )
            listed_levels.append(listed)
            raw_listed_levels.append(raw_listed)
            fitted_listed_levels.append(fitted_listed)
            raw_fitted_listed_levels.append(raw_fitted_listed)
            if (index + 1) % 50 == 0 or index + 1 == len(dates):
                print(
                    f"listed-expiry VIX {index + 1}/{len(dates)} "
                    f"({trading_date.isoformat()})",
                    flush=True,
                )
    model_frame = pl.DataFrame(
        {
            "date": dates,
            "dysanos_vix": np.asarray(fitted_listed_levels, dtype=np.float64),
            "dysanos_marginal_vix": np.concatenate(marginal_levels).astype(np.float64),
            "wrds_listed_vix": np.asarray(raw_listed_levels, dtype=np.float64),
            "dysanos_full_book_vix": np.asarray(listed_levels, dtype=np.float64),
            "wrds_fitted_vix": np.asarray(raw_fitted_listed_levels, dtype=np.float64),
        }
    )
    cboe = load_cboe_vix(start=dates[0], end=dates[-1]).select(
        "date", pl.col("close").alias("cboe_vix")
    )
    comparison = model_frame.join(cboe, on="date", how="inner").with_columns(
        (pl.col("dysanos_vix") - pl.col("cboe_vix")).alias("difference_vol_points")
    )
    if comparison.height != len(model_frame):
        raise RuntimeError("official VIX history is missing model-state dates")
    difference = comparison["difference_vol_points"].to_numpy()
    report: dict[str, object] = {
        "schema": "dysanos-v11-vix-comparison-v1",
        "checkpoint": str(checkpoint.resolve()),
        "state_dates": [dates[0].isoformat(), dates[-1].isoformat()],
        "observations": comparison.height,
        "horizon_calendar_days": horizon_calendar_days,
        "strike_nodes_per_side": strike_nodes_per_side,
        "evaluation_dtype": str(next(model.parameters()).dtype),
        "dysanos_vix_convention": (
            "two listed PM expiries bracketing 30 calendar days, Cboe-style "
            "discrete strike sum and total-variance interpolation"
        ),
        "dysanos_marginal_vix_convention": (
            "continuous 30-calendar-day marginal-density variance proxy"
        ),
        "mean_difference_vol_points": float(np.mean(difference)),
        "median_absolute_difference_vol_points": float(np.median(np.abs(difference))),
        "rmse_vol_points": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference_vol_points": float(np.max(np.abs(difference))),
        "fraction_absolute_difference_above_one_vol_point": float(np.mean(np.abs(difference) > 1.0)),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison.write_parquet(output_directory / "vix_comparison.parquet")
    (output_directory / "vix_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(comparison, output_directory / "vix_comparison.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pre2025-history", required=True, type=Path)
    parser.add_argument("--forward-history", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--horizon-calendar-days", type=int, default=30)
    parser.add_argument("--strike-nodes-per-side", type=int, default=801)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    report = run_comparison(**vars(arguments))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
