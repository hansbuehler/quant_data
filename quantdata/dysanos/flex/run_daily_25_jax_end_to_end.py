"""End-to-end JAX calibration of daily 25-expiry DLV states.

This runner deliberately does not read an LP artifact.  It calibrates the
uncompressed and affine log-DLV² states directly to the retained market books,
using the same fixed 5% Black backbone and the same 25-by-200 shared-strike
geometry as the LP benchmark.  The LP is therefore a separate convex upper
bound on attainable quote fit, rather than a training target.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

_MODEL_BUSINESS_DTES = tuple(float(value) for value in range(1, 26))
_MIN_DLV_SQ = 0.01**2
_MAX_DLV_SQ = 3.0**2
_BACKBONE_ANNUAL_VOL = 0.05
_CAP_BARRIER_LOG_WIDTH = 0.10
_CAP_BARRIER_LOG_STD = 0.35
_FIT_WEIGHT = 0.05


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_npz(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _geometry(num_inner_strikes: int) -> Any:
    """Build the exact daily LP geometry lazily for preparation only."""
    from quantdata.dysanos.flex.config import (
        SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    )

    return SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=num_inner_strikes,
        nms_range=(-2.5, 1.5),
        num_model_expiries=25,
        first_model_business_dte=1.0,
        model_expiry_step_business_days=1.0,
        explicit_model_business_dtes=_MODEL_BUSINESS_DTES,
    )


def _load_dates(path: Path) -> tuple[date, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("market_dates")
    if not isinstance(values, list) or not values:
        raise ValueError(f"No 'market_dates' in {path}")
    dates = tuple(date.fromisoformat(str(value)) for value in values)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("Market dates must be unique and sorted")
    return dates


def _prepare_one_day(
    trading_date: date,
    *,
    ticker: str,
    num_inner_strikes: int,
) -> dict[str, np.ndarray | str]:
    """Load one real book and compile only immutable JAX calibration inputs."""
    from cdxcore.verbose import Context

    from quantdata.dysanos.flex.config import FlexMarketConfig
    from quantdata.dysanos.flex.market import load_market_day
    from quantdata.dysanos.flex.sanos import make_supports_from_total_vols

    geometry = _geometry(num_inner_strikes)
    day = load_market_day(
        ticker,
        trading_date,
        geometry=geometry,
        config=FlexMarketConfig(excluded_business_dtes=(0.2,)),
        verbose=Context.quiet,
    )
    if day is None:
        raise RuntimeError(f"No eligible market day for {trading_date}")
    model_times = np.asarray(geometry.model_expiries_y, dtype=np.float64)
    expiry_y = np.asarray(day.expiries_y[day.expiry_index], dtype=np.float64)
    raw_upper = np.searchsorted(model_times, expiry_y, side="left")
    upper = np.minimum(raw_upper, len(model_times) - 1).astype(np.int32)
    lower = np.maximum(raw_upper - 1, 0).astype(np.int32)
    before_first = raw_upper == 0
    previous = np.where(before_first, 0.0, model_times[lower])
    alpha = np.clip(
        (expiry_y - previous) / (model_times[upper] - previous),
        0.0,
        1.0,
    )
    annual_atm = day.model_atm_total_vols / np.sqrt(model_times)
    raw_initial = np.log(
        np.maximum(annual_atm**2 - _MIN_DLV_SQ, 1.0e-8)
    )[:, None] * np.ones((1, num_inner_strikes), dtype=np.float64)
    supports = make_supports_from_total_vols(
        day.model_atm_total_vols,
        geometry=geometry,
    )
    if not np.allclose(supports, supports[:1], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Shared-strike geometry unexpectedly changed support by expiry")
    return {
        "date": trading_date.isoformat(),
        "support": supports[0].astype(np.float32),
        "raw_initial": raw_initial.astype(np.float32),
        "strikes": np.asarray(day.strikes, dtype=np.float32),
        "is_call": np.asarray(day.is_call, dtype=np.bool_),
        "bids": np.asarray(day.bids, dtype=np.float32),
        "asks": np.asarray(day.asks, dtype=np.float32),
        "mids": np.asarray(0.5 * (day.bids + day.asks), dtype=np.float32),
        "lower": lower,
        "upper": upper,
        "alpha": alpha.astype(np.float32),
        "before_first": before_first.astype(np.bool_),
        "business_dtes": np.asarray(day.quote_business_dtes, dtype=np.float32),
    }


def prepare_dataset(
    *,
    market_dates_report: Path,
    output: Path,
    ticker: str,
    num_inner_strikes: int,
    quote_chunk_size: int,
    workers: int,
) -> None:
    """Materialize a padded immutable real-market calibration tensor.

    This has no dependency on any LP fit.  Padding to a whole quote chunk lets
    the fitted JAX function scan a static shape while retaining the exact
    all-quote objective through the boolean mask.
    """
    if quote_chunk_size <= 0:
        raise ValueError("'quote_chunk_size' must be positive")
    dates = _load_dates(market_dates_report)
    loader = partial(
        _prepare_one_day,
        ticker=ticker,
        num_inner_strikes=num_inner_strikes,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        days = list(executor.map(loader, dates))
    quote_counts = np.asarray([len(day["strikes"]) for day in days], dtype=np.int32)
    max_quotes = int(np.max(quote_counts))
    padded_quotes = quote_chunk_size * int(
        np.ceil(max_quotes / quote_chunk_size)
    )
    day_count = len(days)
    supports = np.stack([np.asarray(day["support"]) for day in days])
    raw_initial = np.stack([np.asarray(day["raw_initial"]) for day in days])
    quote_shape = (day_count, padded_quotes)
    strikes = np.ones(quote_shape, dtype=np.float32)
    is_call = np.ones(quote_shape, dtype=np.bool_)
    bids = np.zeros(quote_shape, dtype=np.float32)
    asks = np.ones(quote_shape, dtype=np.float32)
    mids = np.zeros(quote_shape, dtype=np.float32)
    lower = np.zeros(quote_shape, dtype=np.int32)
    upper = np.zeros(quote_shape, dtype=np.int32)
    alpha = np.zeros(quote_shape, dtype=np.float32)
    before_first = np.ones(quote_shape, dtype=np.bool_)
    business_dtes = np.zeros(quote_shape, dtype=np.float32)
    mask = np.zeros(quote_shape, dtype=np.bool_)
    for index, day in enumerate(days):
        count = int(quote_counts[index])
        for name, array in (
            ("strikes", strikes),
            ("is_call", is_call),
            ("bids", bids),
            ("asks", asks),
            ("mids", mids),
            ("lower", lower),
            ("upper", upper),
            ("alpha", alpha),
            ("before_first", before_first),
            ("business_dtes", business_dtes),
        ):
            array[index, :count] = np.asarray(day[name])
        mask[index, :count] = True
    geometry = _geometry(num_inner_strikes)
    _atomic_write_npz(
        {
            "schema": np.asarray("daily25-jax-market-v1"),
            "dates": np.asarray([str(day["date"]) for day in days]),
            "model_times_y": np.asarray(geometry.model_expiries_y, dtype=np.float32),
            "supports": supports,
            "raw_initial": raw_initial,
            "strikes": strikes,
            "is_call": is_call,
            "bids": bids,
            "asks": asks,
            "mids": mids,
            "lower": lower,
            "upper": upper,
            "alpha": alpha,
            "before_first": before_first,
            "business_dtes": business_dtes,
            "mask": mask,
            "quote_counts": quote_counts,
            "quote_chunk_size": np.asarray(quote_chunk_size, dtype=np.int32),
            "fit_weight": np.asarray(_FIT_WEIGHT, dtype=np.float32),
            "backbone_annual_vol": np.asarray(
                _BACKBONE_ANNUAL_VOL,
                dtype=np.float32,
            ),
        },
        output,
    )
    _atomic_write_json(
        {
            "schema": "daily25-jax-market-v1",
            "status": "complete",
            "source": "direct market books; no LP artifacts used",
            "ticker": ticker,
            "market_dates_report": str(market_dates_report.resolve()),
            "num_dates": day_count,
            "num_inner_strikes": num_inner_strikes,
            "num_model_expiries": 25,
            "quote_chunk_size": quote_chunk_size,
            "max_quotes_before_padding": max_quotes,
            "padded_quotes": padded_quotes,
            "num_quotes": int(np.sum(quote_counts)),
            "nms_range": [-2.5, 1.5],
            "objective": "mean_date[(bid_ask_excess + 0.05*abs(mid_error))/half_spread]",
            "backbone_annual_vol": _BACKBONE_ANNUAL_VOL,
        },
        output.with_suffix(".json"),
    )


def _jax_modules() -> tuple[Any, Any, Any]:
    """Import JAX only after '--prepare' has avoided JAX dependencies."""
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    return jax, jnp, jsp


def _initial_parameters(
    raw_initial: np.ndarray,
    *,
    mode: Literal["direct", "affine"],
    rank: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Initialize directly from ATM-scale DLVs, never from an LP solution."""
    if mode == "direct":
        return {"raw": np.asarray(raw_initial, dtype=np.float32)}
    if rank <= 0:
        raise ValueError("An affine calibration requires a positive rank")
    generator = np.random.default_rng(seed)
    mean = np.mean(raw_initial, axis=0, dtype=np.float64).astype(np.float32)
    # The ATM initializer is nearly strike-flat, so its PCA would have rank at
    # most 25.  Small non-zero factors deliberately break that degeneracy while
    # leaving the initial decoded surface indistinguishable from the ATM seed.
    codes = generator.normal(
        loc=0.0,
        scale=1.0e-3,
        size=(len(raw_initial), rank),
    ).astype(np.float32)
    loadings = generator.normal(
        loc=0.0,
        scale=1.0e-3,
        size=(rank, *mean.shape),
    ).astype(np.float32)
    return {"mean": mean, "codes": codes, "loadings": loadings}


def _raw_states(params: dict[str, Any], mode: str, jnp: Any) -> Any:
    if mode == "direct":
        return params["raw"]
    return params["mean"][None, ...] + jnp.einsum(
        "dr,rtk->dtk",
        params["codes"],
        params["loadings"],
    )


def _make_loss_functions(
    data: dict[str, Any],
    *,
    mode: str,
    cap_penalty_weight: float,
    quote_chunk_size: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Compile the exact fixed-grid SANOS quote objective and its gradient."""
    jax, jnp, jsp = _jax_modules()
    supports = data["supports"]
    model_times_y = data["model_times_y"]
    if data["strikes"].shape[-1] % quote_chunk_size != 0:
        raise ValueError("Padded quote count must be divisible by quote chunk size")
    quote_blocks = {
        name: value.reshape(
            value.shape[0], -1, quote_chunk_size
        ).transpose(1, 0, 2)
        for name, value in data.items()
        if name
        in {
            "strikes",
            "is_call",
            "bids",
            "asks",
            "mids",
            "lower",
            "upper",
            "alpha",
            "before_first",
            "mask",
        }
    }
    model_dts_y = jnp.diff(model_times_y, prepend=jnp.zeros((1,), dtype=jnp.float32))
    maximum_log_dlvsq = np.log(_MAX_DLV_SQ - _MIN_DLV_SQ)

    def density0_from_support(support: Any) -> Any:
        calls = jnp.maximum(1.0 - support, 0.0)
        slopes = jnp.diff(
            calls,
            axis=-1,
            prepend=jnp.ones_like(calls[..., :1]),
        ) / jnp.diff(
            support,
            axis=-1,
            prepend=jnp.zeros_like(support[..., :1]),
        )
        return jnp.diff(slopes, axis=-1, append=jnp.zeros_like(slopes[..., :1]))

    def sanos_step(rhs: Any, dlvsq: Any) -> Any:
        box = (
            jnp.square(supports[:, 1:-1])
            / (supports[:, 2:] - supports[:, :-2])
        )
        w_plus = dlvsq * box / (supports[:, 2:] - supports[:, 1:-1])
        w_minus = dlvsq * box / (supports[:, 1:-1] - supports[:, :-2])
        size = rhs.shape[-1]
        # Match FlexSanosDecoder._solve_sanos_step exactly.  The first inner
        # node has no lower neighbour and the first two rows carry no
        # ``w_plus`` coefficient; conversely ``w_minus`` fills the first 200
        # upper-diagonal entries and the two boundary rows at the end are
        # zero.  A one-node shift here breaks the M-matrix structure and can
        # produce enormous signed pseudo-densities.
        lower = jnp.concatenate(
            (jnp.zeros_like(w_plus[:, :2]), -w_plus),
            axis=-1,
        )
        diagonal = jnp.concatenate(
            (
                jnp.ones_like(rhs[:, :1]),
                1.0 + w_minus + w_plus,
                jnp.ones_like(rhs[:, :1]),
            ),
            axis=-1,
        )
        upper = jnp.concatenate(
            (-w_minus, jnp.zeros_like(w_minus[:, :2])),
            axis=-1,
        )

        cp0 = upper[:, 0] / diagonal[:, 0]
        y0 = rhs[:, 0] / diagonal[:, 0]

        def forward(carry: tuple[Any, Any], index: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
            previous_cp, previous_y = carry
            denominator = diagonal[:, index] - lower[:, index] * previous_cp
            cp = upper[:, index] / denominator
            y = (rhs[:, index] - lower[:, index] * previous_y) / denominator
            return (cp, y), (cp, y)

        (_, _), values = jax.lax.scan(
            forward,
            (cp0, y0),
            jnp.arange(1, size),
        )
        cps = jnp.concatenate((cp0[:, None], values[0].T), axis=1)
        ys = jnp.concatenate((y0[:, None], values[1].T), axis=1)

        def backward(next_x: Any, pair: tuple[Any, Any]) -> tuple[Any, Any]:
            y, cp = pair
            value = y - cp * next_x
            return value, value

        _, reversed_values = jax.lax.scan(
            backward,
            ys[:, -1],
            (ys[:, :-1].T[::-1], cps[:, :-1].T[::-1]),
        )
        return jnp.concatenate((reversed_values[::-1].T, ys[:, -1:]), axis=1)

    def decode(params: dict[str, Any]) -> tuple[Any, Any, Any]:
        raw = _raw_states(params, mode, jnp)
        limited_raw = jnp.minimum(raw, maximum_log_dlvsq)
        dlvsqs = _MIN_DLV_SQ + jnp.exp(limited_raw)
        density0 = density0_from_support(supports)

        def one_expiry(previous: Any, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
            dlvsq, dt = inputs
            # The support is shared across all expiries within the day, so no
            # transfer/interpolation is needed between SANOS steps.
            current = sanos_step(previous, dlvsq * dt)
            return current, current

        _, density_time_major = jax.lax.scan(
            one_expiry,
            density0,
            (jnp.moveaxis(dlvsqs, 1, 0), model_dts_y),
        )
        densities = jnp.moveaxis(density_time_major, 0, 1)
        scaled = (raw - maximum_log_dlvsq) / _CAP_BARRIER_LOG_WIDTH
        barrier = (
            _CAP_BARRIER_LOG_WIDTH * jax.nn.softplus(scaled) / _CAP_BARRIER_LOG_STD
        )
        cap_penalty = 0.5 * jnp.mean(jnp.sum(jnp.square(barrier), axis=(1, 2)))
        return densities, density0, cap_penalty

    def black_call(spot: Any, strike: Any, total_vol: Any) -> Any:
        safe_vol = jnp.maximum(total_vol, 1.0e-8)
        d1 = (jnp.log(spot / strike) + 0.5 * jnp.square(safe_vol)) / safe_vol
        d2 = d1 - safe_vol
        value = spot * jsp.special.ndtr(d1) - strike * jsp.special.ndtr(d2)
        return jnp.where(total_vol > 0.0, value, jnp.maximum(spot - strike, 0.0))

    def gather_density(densities: Any, indices: Any) -> Any:
        return jax.vmap(lambda value, index: value[index])(densities, indices)

    def one_quote_block(
        carry: tuple[Any, Any], block: dict[str, Any], *, densities: Any, density0: Any
    ) -> tuple[tuple[Any, Any], None]:
        sums, counts = carry
        strikes = block["strikes"]
        upper_density = gather_density(densities, block["upper"])
        lower_density = gather_density(densities, block["lower"])
        lower_density = jnp.where(
            block["before_first"][..., None],
            density0[:, None, :],
            lower_density,
        )
        total_vol = _BACKBONE_ANNUAL_VOL * jnp.sqrt(
            jnp.maximum(
                (1.0 - block["alpha"])
                * model_times_y[block["lower"]]
                + block["alpha"] * model_times_y[block["upper"]],
                0.0,
            )
        )
        kernel = black_call(
            supports[:, None, :],
            strikes[..., None],
            total_vol[..., None],
        )
        calls = (1.0 - block["alpha"]) * jnp.sum(kernel * lower_density, axis=-1)
        calls += block["alpha"] * jnp.sum(kernel * upper_density, axis=-1)
        prices = jnp.where(block["is_call"], calls, calls - 1.0 + strikes)
        half_spreads = 0.5 * (block["asks"] - block["bids"])
        quote_loss = (
            jax.nn.relu(prices - block["asks"])
            + jax.nn.relu(block["bids"] - prices)
            + _FIT_WEIGHT * jnp.abs(prices - block["mids"])
        ) / half_spreads
        valid = block["mask"].astype(prices.dtype)
        return (sums + jnp.sum(quote_loss * valid, axis=1), counts + jnp.sum(valid, axis=1)), None

    # Reverse mode over every quote block retains all Black kernels and exceeds
    # GPU activation memory.  We instead sum each static quote block's gradient
    # at the same parameters and apply Adam once: exact population gradient,
    # without stochastic minibatching.
    quote_counts = jnp.sum(data["mask"], axis=1).astype(jnp.float32)
    num_quote_blocks = next(iter(quote_blocks.values())).shape[0]

    def block_loss(
        params: dict[str, Any],
        block: dict[str, Any],
    ) -> tuple[Any, tuple[Any, Any]]:
        densities, density0, cap_penalty = decode(params)
        initial = (
            jnp.zeros((supports.shape[0],), dtype=jnp.float32),
            jnp.zeros((supports.shape[0],), dtype=jnp.float32),
        )
        (sums, _), _ = one_quote_block(
            initial,
            block,
            densities=densities,
            density0=density0,
        )
        quote_loss = jnp.mean(sums / quote_counts)
        objective = quote_loss + cap_penalty_weight * cap_penalty / num_quote_blocks
        return objective, (quote_loss, cap_penalty)

    block_value_and_grad = jax.jit(
        jax.value_and_grad(block_loss, has_aux=True)
    )

    @jax.jit
    def adam_update(
        params: dict[str, Any],
        first: dict[str, Any],
        second: dict[str, Any],
        step: Any,
        gradients: dict[str, Any],
        learning_rate: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        next_step = step + 1
        next_first = jax.tree.map(
            lambda value, gradient: 0.9 * value + 0.1 * gradient,
            first,
            gradients,
        )
        next_second = jax.tree.map(
            lambda value, gradient: 0.999 * value + 0.001 * jnp.square(gradient),
            second,
            gradients,
        )
        corrected_first = jax.tree.map(
            lambda value: value / (1.0 - 0.9**next_step),
            next_first,
        )
        corrected_second = jax.tree.map(
            lambda value: value / (1.0 - 0.999**next_step),
            next_second,
        )
        gradient_sq = sum(
            jnp.sum(jnp.square(value)) for value in jax.tree.leaves(gradients)
        )
        next_params = jax.tree.map(
            lambda value, first_value, second_value: value
            - learning_rate * first_value / (jnp.sqrt(second_value) + 1.0e-8),
            params,
            corrected_first,
            corrected_second,
        )
        return next_params, next_first, next_second, next_step, jnp.sqrt(gradient_sq)

    @jax.jit
    def decode_only(params: dict[str, Any]) -> tuple[Any, Any]:
        densities, density0, _ = decode(params)
        return densities, density0

    @jax.jit
    def predict_block(densities: Any, density0: Any, block: dict[str, Any]) -> Any:
        strikes = block["strikes"]
        upper_density = gather_density(densities, block["upper"])
        lower_density = gather_density(densities, block["lower"])
        lower_density = jnp.where(
            block["before_first"][..., None],
            density0[:, None, :],
            lower_density,
        )
        total_vol = _BACKBONE_ANNUAL_VOL * jnp.sqrt(
            jnp.maximum(
                (1.0 - block["alpha"])
                * model_times_y[block["lower"]]
                + block["alpha"] * model_times_y[block["upper"]],
                0.0,
            )
        )
        kernel = black_call(
            supports[:, None, :],
            strikes[..., None],
            total_vol[..., None],
        )
        calls = (1.0 - block["alpha"]) * jnp.sum(kernel * lower_density, axis=-1)
        calls += block["alpha"] * jnp.sum(kernel * upper_density, axis=-1)
        return jnp.where(block["is_call"], calls, calls - 1.0 + strikes)

    return block_value_and_grad, adam_update, decode_only, predict_block, quote_blocks


def _load_data(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]) for name in loaded.files}
    if str(payload.pop("schema").item()) != "daily25-jax-market-v1":
        raise ValueError(f"Unexpected JAX market schema in {path}")
    metadata = {
        "dates": payload.pop("dates"),
        "quote_counts": payload.pop("quote_counts"),
        "quote_chunk_size": int(payload.pop("quote_chunk_size").item()),
        "fit_weight": float(payload.pop("fit_weight").item()),
        "backbone_annual_vol": float(payload.pop("backbone_annual_vol").item()),
    }
    if not (
        np.isclose(metadata["fit_weight"], _FIT_WEIGHT, rtol=0.0, atol=1.0e-7)
        and np.isclose(
            metadata["backbone_annual_vol"],
            _BACKBONE_ANNUAL_VOL,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        raise ValueError("Dataset objective does not match this JAX runner")
    return payload, metadata


def _statistics(
    predictions: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, object]:
    mask = data["mask"]
    fits = predictions[mask]
    bids = data["bids"][mask]
    asks = data["asks"][mask]
    excess = (np.maximum(fits - asks, 0.0) + np.maximum(bids - fits, 0.0)) / (
        0.5 * (asks - bids)
    )
    positive = excess[excess > 0.0]
    return {
        "num_quotes": int(len(excess)),
        "percent_inside_bid_ask": float(100.0 * np.mean(excess == 0.0)),
        "mean_excess_half_spreads": float(np.mean(excess)),
        "conditional_mean_excess_half_spreads": float(np.mean(positive)) if len(positive) else 0.0,
        "conditional_median_excess_half_spreads": float(np.median(positive)) if len(positive) else 0.0,
        "midpoint_mae_half_spreads": float(
            np.mean(np.abs(fits - data["mids"][mask]) / (0.5 * (asks - bids)))
        ),
    }


def _group_statistics(
    predictions: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    labels: tuple[str, ...],
    selections: tuple[np.ndarray, ...],
) -> dict[str, dict[str, float]]:
    """Summarize bid--ask fit for explicitly specified quote groups."""
    mask = np.asarray(data["mask"], dtype=bool)
    bids = np.asarray(data["bids"], dtype=np.float64)
    asks = np.asarray(data["asks"], dtype=np.float64)
    mids = np.asarray(data["mids"], dtype=np.float64)
    half_spreads = 0.5 * (asks - bids)
    excess = (
        np.maximum(predictions - asks, 0.0) + np.maximum(bids - predictions, 0.0)
    ) / half_spreads
    midpoint_error = (predictions - mids) / half_spreads
    result: dict[str, dict[str, float]] = {}
    for label, selection in zip(labels, selections, strict=True):
        active = mask & selection
        if not np.any(active):
            continue
        active_excess = excess[active]
        positive = active_excess[active_excess > 0.0]
        result[label] = {
            "num_quotes": int(np.sum(active)),
            "percent_inside_bid_ask": float(100.0 * np.mean(active_excess == 0.0)),
            "mean_excess_half_spreads": float(np.mean(active_excess)),
            "conditional_mean_excess_half_spreads": (
                float(np.mean(positive)) if len(positive) else 0.0
            ),
            "midpoint_bias_half_spreads": float(np.mean(midpoint_error[active])),
            "percent_below_bid": float(
                100.0 * np.mean(predictions[active] < bids[active])
            ),
            "percent_above_ask": float(
                100.0 * np.mean(predictions[active] > asks[active])
            ),
        }
    return result


def _fit_diagnostics(
    predictions: np.ndarray,
    data: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Locate fit failures without retaining an unnecessary prediction tensor."""
    business_dtes = np.asarray(data["business_dtes"], dtype=np.float64)
    expiry_labels = ("1D", "2D", "3-5D", "6-10D", "11-15D", "16-25D")
    expiry_selections = (
        business_dtes <= 1.5,
        (business_dtes > 1.5) & (business_dtes <= 2.5),
        (business_dtes > 2.5) & (business_dtes <= 5.5),
        (business_dtes > 5.5) & (business_dtes <= 10.5),
        (business_dtes > 10.5) & (business_dtes <= 15.5),
        business_dtes > 15.5,
    )
    raw_initial = np.asarray(data["raw_initial"], dtype=np.float64)
    model_times = np.asarray(data["model_times_y"], dtype=np.float64)
    total_atm = np.sqrt(
        np.exp(raw_initial[..., 0]) + _MIN_DLV_SQ
    ) * np.sqrt(model_times)[None, :]
    lower = np.asarray(data["lower"], dtype=np.intp)
    upper = np.asarray(data["upper"], dtype=np.intp)
    alpha = np.asarray(data["alpha"], dtype=np.float64)
    lower_var = np.take_along_axis(total_atm**2, lower, axis=1)
    upper_var = np.take_along_axis(total_atm**2, upper, axis=1)
    quote_total_vol = np.sqrt((1.0 - alpha) * lower_var + alpha * upper_var)
    standardized_log_moneyness = np.log(
        np.asarray(data["strikes"], dtype=np.float64)
    ) / quote_total_vol
    moneyness_edges = (-np.inf, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, np.inf)
    moneyness_labels = (
        "<=-1.5",
        "-1.5:-0.75",
        "-0.75:-0.25",
        "-0.25:0.25",
        "0.25:0.75",
        "0.75:1.5",
        ">1.5",
    )
    moneyness_selections = tuple(
        (standardized_log_moneyness >= lower_edge)
        & (standardized_log_moneyness < upper_edge)
        for lower_edge, upper_edge in zip(
            moneyness_edges[:-1], moneyness_edges[1:], strict=True
        )
    )
    dates = tuple(str(value) for value in metadata["dates"])
    date_statistics = _group_statistics(
        predictions,
        data,
        labels=dates,
        selections=tuple(
            np.arange(predictions.shape[0])[:, None] == index
            for index in range(predictions.shape[0])
        ),
    )
    worst_dates = sorted(
        ({"date": label, **statistics} for label, statistics in date_statistics.items()),
        key=lambda value: float(value["mean_excess_half_spreads"]),
        reverse=True,
    )[:20]
    is_call = np.asarray(data["is_call"], dtype=bool)
    return {
        "overall": _statistics(predictions, data),
        "by_business_dte": _group_statistics(
            predictions,
            data,
            labels=expiry_labels,
            selections=expiry_selections,
        ),
        "by_standardized_log_moneyness": _group_statistics(
            predictions,
            data,
            labels=moneyness_labels,
            selections=moneyness_selections,
        ),
        "by_option_type": _group_statistics(
            predictions,
            data,
            labels=("call", "put"),
            selections=(is_call, ~is_call),
        ),
        "worst_dates": worst_dates,
        "standardized_log_moneyness_definition": (
            "log(strike) divided by interpolated day-specific ATM total volatility"
        ),
    }


def audit_checkpoint(*, dataset: Path, checkpoint: Path, output: Path) -> None:
    """Evaluate an end-to-end fitted checkpoint by location of market errors."""
    data_cpu, metadata = _load_data(dataset)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        if str(loaded["schema"].item()) != "daily25-jax-end-to-end-v1":
            raise ValueError(f"Unexpected checkpoint schema in {checkpoint}")
        mode = str(loaded["mode"].item())
        rank = int(loaded["rank"].item())
        params_cpu = {
            name: np.asarray(loaded[name])
            for name in loaded.files
            if name not in {"schema", "mode", "rank"}
        }
    jax, jnp, _ = _jax_modules()
    if jax.default_backend() not in {"cuda", "gpu"}:
        raise RuntimeError("This runner is intended for CUDA JAX; GPU backend unavailable")
    data = {
        name: jax.device_put(jnp.asarray(value))
        for name, value in data_cpu.items()
    }
    params = jax.device_put(
        {name: jnp.asarray(value) for name, value in params_cpu.items()}
    )
    _, _, decode_only, predict_block, quote_blocks = _make_loss_functions(
        data,
        mode=mode,
        cap_penalty_weight=0.0,
        quote_chunk_size=int(metadata["quote_chunk_size"]),
    )
    densities, density0 = decode_only(params)
    prediction_blocks = [
        np.asarray(
            predict_block(
                densities,
                density0,
                {name: value[index] for name, value in quote_blocks.items()},
            )
        )
        for index in range(next(iter(quote_blocks.values())).shape[0])
    ]
    predictions = np.concatenate(prediction_blocks, axis=1)
    _atomic_write_json(
        {
            "schema": "daily25-jax-fit-diagnostics-v1",
            "checkpoint": str(checkpoint.resolve()),
            "mode": mode,
            "rank": rank,
            "source": "direct market calibration; no LP artifacts used",
            "diagnostics": _fit_diagnostics(predictions, data_cpu, metadata),
        },
        output,
    )


def fit_model(
    *,
    dataset: Path,
    output_directory: Path,
    mode: Literal["direct", "affine"],
    rank: int,
    epochs: int,
    learning_rate: float,
    minimum_learning_rate_fraction: float,
    cap_penalty_weight: float,
    seed: int,
    checkpoint_every: int,
    initial_checkpoint: Path | None,
    epoch_offset: int,
    gradient_tolerance: float | None,
) -> None:
    """Fit a direct or affine DLV model to all market dates end-to-end."""
    if epochs <= 0 or learning_rate <= 0.0 or checkpoint_every <= 0:
        raise ValueError("epochs, learning rate, and checkpoint interval must be positive")
    if not 0.0 <= minimum_learning_rate_fraction <= 1.0:
        raise ValueError("minimum learning-rate fraction must lie in [0,1]")
    data_cpu, metadata = _load_data(dataset)
    jax, jnp, _ = _jax_modules()
    if jax.default_backend() not in {"cuda", "gpu"}:
        raise RuntimeError(
            "This runner is intended for CUDA JAX; GPU backend unavailable"
        )
    data = {name: jax.device_put(jnp.asarray(value)) for name, value in data_cpu.items()}
    raw_initial = np.asarray(data_cpu["raw_initial"], dtype=np.float32)
    params_cpu = _initial_parameters(raw_initial, mode=mode, rank=rank, seed=seed)
    if initial_checkpoint is not None:
        with np.load(initial_checkpoint, allow_pickle=False) as loaded:
            checkpoint_mode = str(loaded["mode"].item())
            checkpoint_rank = int(loaded["rank"].item())
            if checkpoint_mode != mode or checkpoint_rank != rank:
                raise ValueError(
                    "Initial checkpoint architecture does not match the requested fit"
                )
            params_cpu = {
                name: np.asarray(loaded[name], dtype=np.float32)
                for name in params_cpu
            }
    params = jax.device_put({name: jnp.asarray(value) for name, value in params_cpu.items()})
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    step = jnp.asarray(0, dtype=jnp.int32)
    (
        block_value_and_grad,
        adam_update,
        decode_only,
        predict_block,
        quote_blocks,
    ) = _make_loss_functions(
        data,
        mode=mode,
        cap_penalty_weight=cap_penalty_weight,
        quote_chunk_size=int(metadata["quote_chunk_size"]),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    started = perf_counter()
    checkpoint = output_directory / "checkpoint.npz"
    stopped_at_gradient_tolerance = False
    for epoch in range(epochs):
        fraction = epoch / max(epochs - 1, 1)
        current_learning_rate = learning_rate * (
            minimum_learning_rate_fraction
            + (1.0 - minimum_learning_rate_fraction)
            * 0.5
            * (1.0 + np.cos(np.pi * fraction))
        )
        gradients = jax.tree.map(jnp.zeros_like, params)
        objective = jnp.zeros((), dtype=jnp.float32)
        quote_loss = jnp.zeros((), dtype=jnp.float32)
        cap_penalty = jnp.zeros((), dtype=jnp.float32)
        for block_index in range(next(iter(quote_blocks.values())).shape[0]):
            block = {
                name: value[block_index]
                for name, value in quote_blocks.items()
            }
            (block_objective, auxiliary), block_gradients = block_value_and_grad(
                params,
                block,
            )
            gradients = jax.tree.map(
                lambda left, right: left + right,
                gradients,
                block_gradients,
            )
            # Make each exact contribution complete before dispatching the
            # next. Otherwise JAX queues all 40 reverse passes and reproduces
            # the activation-memory peak that blockwise accumulation avoids.
            gradients = jax.block_until_ready(gradients)
            objective += block_objective
            quote_loss += auxiliary[0]
            cap_penalty += auxiliary[1]
        params, first, second, step, gradient_norm = adam_update(
            params,
            first,
            second,
            step,
            gradients,
            jnp.asarray(current_learning_rate, dtype=jnp.float32),
        )
        objective, quote_loss, cap_penalty, gradient_norm = (
            float(value.block_until_ready())
            for value in (
                objective,
                quote_loss,
                cap_penalty / next(iter(quote_blocks.values())).shape[0],
                gradient_norm,
            )
        )
        row = {
            "epoch": float(epoch_offset + epoch + 1),
            "objective": objective,
            "quote_loss": quote_loss,
            "cap_penalty": cap_penalty,
            "gradient_l2_norm": gradient_norm,
            "learning_rate": current_learning_rate,
        }
        history.append(row)
        reached_tolerance = (
            gradient_tolerance is not None and gradient_norm <= gradient_tolerance
        )
        if (
            epoch == 0
            or (epoch + 1) % checkpoint_every == 0
            or epoch + 1 == epochs
            or reached_tolerance
        ):
            host_params = jax.device_get(params)
            _atomic_write_npz(
                {
                    "schema": np.asarray("daily25-jax-end-to-end-v1"),
                    "mode": np.asarray(mode),
                    "rank": np.asarray(rank, dtype=np.int32),
                    **{name: np.asarray(value) for name, value in host_params.items()},
                },
                checkpoint,
            )
            print(
                f"{mode} rank={rank} epoch={epoch_offset + epoch + 1}/"
                f"{epoch_offset + epochs} "
                f"objective={objective:.7g} quote={quote_loss:.7g} "
                f"cap={cap_penalty:.3g} grad={gradient_norm:.3g}",
                flush=True,
            )
        if reached_tolerance:
            stopped_at_gradient_tolerance = True
            break
    densities, density0 = decode_only(params)
    prediction_blocks: list[np.ndarray] = []
    for index in range(next(iter(quote_blocks.values())).shape[0]):
        block = {name: value[index] for name, value in quote_blocks.items()}
        prediction_blocks.append(np.asarray(predict_block(densities, density0, block)))
    predictions = np.concatenate(prediction_blocks, axis=1)
    report = {
        "schema": "daily25-jax-end-to-end-v1",
        "status": "complete",
        "mode": mode,
        "rank": rank,
        "source": "direct market calibration; no LP artifacts used",
        "dataset": str(dataset.resolve()),
        "num_dates": int(len(metadata["dates"])),
        "num_quotes": int(np.sum(metadata["quote_counts"])),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "epoch_offset": epoch_offset,
        "initial_checkpoint": (
            str(initial_checkpoint.resolve()) if initial_checkpoint is not None else None
        ),
        "gradient_tolerance": gradient_tolerance,
        "stopped_at_gradient_tolerance": stopped_at_gradient_tolerance,
        "elapsed_seconds": perf_counter() - started,
        "history": history,
        "final_statistics": _statistics(predictions, data_cpu),
        "checkpoint": str(checkpoint.resolve()),
        "backend": jax.default_backend(),
        "devices": [str(value) for value in jax.devices()],
        "cap": {
            "min_dlvsq": _MIN_DLV_SQ,
            "max_dlvsq": _MAX_DLV_SQ,
            "barrier_log_width": _CAP_BARRIER_LOG_WIDTH,
            "barrier_log_std": _CAP_BARRIER_LOG_STD,
            "penalty_weight": cap_penalty_weight,
        },
    }
    _atomic_write_json(report, output_directory / "report.json")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--market-dates-report", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--ticker", default="SPX")
    prepare.add_argument("--num-inner-strikes", type=int, default=200)
    prepare.add_argument("--quote-chunk-size", type=int, default=128)
    prepare.add_argument("--workers", type=int, default=4)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--output-directory", type=Path, required=True)
    fit.add_argument("--mode", choices=("direct", "affine"), required=True)
    fit.add_argument("--rank", type=int, default=0)
    fit.add_argument("--epochs", type=int, default=300)
    fit.add_argument("--learning-rate", type=float, default=2.0e-2)
    fit.add_argument("--minimum-learning-rate-fraction", type=float, default=0.05)
    fit.add_argument("--cap-penalty-weight", type=float, default=1.0)
    fit.add_argument("--seed", type=int, default=1729)
    fit.add_argument("--checkpoint-every", type=int, default=10)
    fit.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Warm-start parameters from a matching direct or affine checkpoint.",
    )
    fit.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Completed epochs represented by the warm-start checkpoint.",
    )
    fit.add_argument(
        "--gradient-tolerance",
        type=float,
        help="Stop after a full-gradient update once its L2 norm is at most this value.",
    )
    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.command == "prepare":
        prepare_dataset(
            market_dates_report=arguments.market_dates_report,
            output=arguments.output,
            ticker=arguments.ticker,
            num_inner_strikes=arguments.num_inner_strikes,
            quote_chunk_size=arguments.quote_chunk_size,
            workers=arguments.workers,
        )
    elif arguments.command == "fit":
        fit_model(
            dataset=arguments.dataset,
            output_directory=arguments.output_directory,
            mode=arguments.mode,
            rank=arguments.rank,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            minimum_learning_rate_fraction=arguments.minimum_learning_rate_fraction,
            cap_penalty_weight=arguments.cap_penalty_weight,
            seed=arguments.seed,
            checkpoint_every=arguments.checkpoint_every,
            initial_checkpoint=arguments.initial_checkpoint,
            epoch_offset=arguments.epoch_offset,
            gradient_tolerance=arguments.gradient_tolerance,
        )
    else:
        audit_checkpoint(
            dataset=arguments.dataset,
            checkpoint=arguments.checkpoint,
            output=arguments.output,
        )


if __name__ == "__main__":
    main()
