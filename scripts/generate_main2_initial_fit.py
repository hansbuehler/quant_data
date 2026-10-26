"""Generate the matched 2025-08-29 initial-fit figures for ``main2.tex``.

The comparison deliberately uses the exact GROMM paper quote slice for all
three models.  Tangent Levy is valued from the final daily-grid reconstruction
used by its Q simulation, rather than from its independent per-maturity fits.
CRC prices come from the retained generalized-Bates target-date calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_data/matplotlib")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from numpy.typing import NDArray
from scipy.special import ndtr

from quantdata.sdlv.daily.artifacts import load_raw_grid
from quantdata.sdlv.daily.train import _load_data, _make_loss_functions
from quantdata.sdlv.tangent_levy.pricing import (
    average_density_from_forward,
    density_call_prices,
)
from quantdata.sdlv.tangent_levy.run_v2 import _load_day
from quantdata.sdlv.tangent_levy.v2_config import TangentLevyV2Config

FloatArray = NDArray[np.float64]

TARGET_DATE = "2025-08-29"
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data"))
DEFAULT_GROMM_DATASET = (
    DEFAULT_DATA_ROOT
    / "training/flex-shared-strikes/daily-25-jax/"
    "spx-2022-2025-daily25-shared200-fixed5-market-v1.npz"
)
DEFAULT_TANGENT_CHECKPOINT = (
    DEFAULT_DATA_ROOT
    / "training/dysanos/tangent-levy/v2/"
    "spx-2025-08-29-paper-aligned-v2/initial-day/2025-08-29.npz"
)
DEFAULT_CRC_RUN = (
    DEFAULT_DATA_ROOT
    / "training/dysanos/bates-crc/spx-2025-08-29-gambara-teichmann-v4-accurate"
)
DEFAULT_OUTPUT = REPOSITORY / "quantdata/sdlv/tex/generated"
GROMM_RUN_LABEL = "direct-continuation-1000-v1"
GROMM_ARTIFACT_DIRECTORY = (
    REPOSITORY / "quantdata/sdlv/daily/artifacts" / GROMM_RUN_LABEL
)
GROMM_REPORT = GROMM_ARTIFACT_DIRECTORY / "report.json"
EXPECTED_GROMM_POOLED_STATISTICS = {
    "num_quotes": 2_192_458,
    "percent_inside_bid_ask": 97.32122576578433,
    "mean_excess_half_spreads": 0.03450184687972069,
    "conditional_mean_excess_half_spreads": 1.2879713773727417,
    "conditional_median_excess_half_spreads": 0.45028775930404663,
    "midpoint_mae_half_spreads": 0.23675066232681274,
}


def _black_call(strikes: FloatArray, tau: FloatArray, volatility: FloatArray) -> FloatArray:
    total_volatility = volatility * np.sqrt(tau)
    d1 = -np.log(strikes) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    return ndtr(d1) - strikes * ndtr(d2)


def _implied_volatility(
    prices: FloatArray,
    strikes: FloatArray,
    tau: FloatArray,
) -> FloatArray:
    """Invert normalized Black calls by deterministic vectorized bisection."""

    intrinsic = np.maximum(1.0 - strikes, 0.0)
    valid = (prices > intrinsic + 1.0e-10) & (prices < 1.0 - 1.0e-10)
    lower = np.full(prices.shape, 1.0e-5, dtype=np.float64)
    upper = np.full(prices.shape, 5.0, dtype=np.float64)
    for _ in range(70):
        middle = 0.5 * (lower + upper)
        below = _black_call(strikes, tau, middle) < prices
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return np.where(valid, 0.5 * (lower + upper), np.nan)


def _target_gromm_book(
    dataset_path: Path,
) -> tuple[dict[str, FloatArray], dict[str, object], int, int]:
    data, metadata = _load_data(dataset_path)
    dates = np.asarray(metadata["dates"]).astype("U10")
    matches = np.flatnonzero(dates == TARGET_DATE)
    if matches.size != 1:
        raise RuntimeError(f"expected exactly one {TARGET_DATE} row in {dataset_path}")
    date_index = int(matches.item())
    quote_count = int(np.asarray(metadata["quote_counts"])[date_index])
    return data, metadata, date_index, quote_count


def _verified_gromm_report(dataset_path: Path) -> dict[str, object]:
    report = json.loads(GROMM_REPORT.read_text(encoding="utf-8"))
    if report.get("schema") != "daily25-jax-end-to-end-v1":
        raise RuntimeError(f"unexpected GROMM report schema in {GROMM_REPORT}")
    if report.get("status") != "complete" or report.get("mode") != "direct":
        raise RuntimeError(f"GROMM run is not a completed direct fit: {GROMM_REPORT}")
    completed_epochs = int(report["epoch_offset"]) + int(report["epochs_completed"])
    if completed_epochs != 1_000:
        raise RuntimeError(f"expected 1,000 GROMM epochs, found {completed_epochs}")
    if Path(str(report["dataset"])).resolve() != dataset_path.resolve():
        raise RuntimeError("GROMM report and initial-fit figure use different datasets")
    statistics = report.get("final_statistics")
    if not isinstance(statistics, dict):
        raise RuntimeError(f"missing final statistics in {GROMM_REPORT}")
    for name, expected in EXPECTED_GROMM_POOLED_STATISTICS.items():
        observed = statistics.get(name)
        if observed is None or not np.isclose(observed, expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                f"GROMM pooled statistic {name!r} changed: "
                f"expected {expected}, found {observed}"
            )
    return report


def _gromm_prices(
    data: dict[str, FloatArray],
    metadata: dict[str, object],
    *,
    date_index: int,
    date_count: int,
    quote_count: int,
) -> FloatArray:
    # These functions are the exact decoder and quote interpolation used by
    # the retained 25-by-200 direct fit.  The script is forced to CPU by main().
    import jax.numpy as jnp

    one_day = {
        name: jnp.asarray(
            value[date_index : date_index + 1]
            if value.ndim > 0 and value.shape[0] == date_count
            else value
        )
        for name, value in data.items()
    }
    _, _, decode, predict_block, quote_blocks = _make_loss_functions(
        one_day,
        mode="direct",
        cap_penalty_weight=1.0,
        quote_chunk_size=int(metadata["quote_chunk_size"]),
    )
    raw = load_raw_grid("full", date_index)[None]
    densities, density0 = decode({"raw": jnp.asarray(raw)})
    block_count = next(iter(quote_blocks.values())).shape[0]
    predictions = []
    for block_index in range(block_count):
        block = {
            name: values[block_index]
            for name, values in quote_blocks.items()
        }
        predictions.append(np.asarray(predict_block(densities, density0, block))[0])
    return np.concatenate(predictions)[:quote_count].astype(np.float64)


def _tangent_prices(
    checkpoint_path: Path,
    *,
    strikes: FloatArray,
    is_call: NDArray[np.bool_],
    business_dtes: FloatArray,
) -> FloatArray:
    _, forward_density, _ = _load_day(checkpoint_path)
    config = TangentLevyV2Config(label="spx-2025-08-29-paper-aligned-v2")
    tau_grid = np.asarray(config.dynamics.maturity_grid, dtype=np.float64)
    jump_x = np.asarray(config.dynamics.jump_grid, dtype=np.float64)
    average_density = average_density_from_forward(
        tau=tau_grid,
        forward_density=forward_density[None],
    )[0]
    prices = np.empty(strikes.shape, dtype=np.float64)
    for business_dte in np.unique(business_dtes):
        selected = np.isclose(business_dtes, business_dte, rtol=0.0, atol=1.0e-4)
        tau = float(business_dte / 252.0)
        right = int(np.searchsorted(tau_grid, tau, side="left"))
        right = min(max(right, 1), len(tau_grid) - 1)
        left = right - 1
        weight = (tau - tau_grid[left]) / (tau_grid[right] - tau_grid[left])
        interpolated_average_density = (
            (1.0 - weight) * tau_grid[left] * average_density[left]
            + weight * tau_grid[right] * average_density[right]
        ) / tau
        calls = density_call_prices(
            tau=tau,
            log_strikes=np.log(strikes[selected]),
            jump_x=jump_x,
            average_density=interpolated_average_density,
        )
        prices[selected] = np.where(
            is_call[selected],
            calls,
            calls - 1.0 + strikes[selected],
        )
    return prices


def _crc_prices(
    run_directory: Path,
    *,
    strikes: FloatArray,
    is_call: NDArray[np.bool_],
    bids: FloatArray,
    asks: FloatArray,
    business_dtes: FloatArray,
) -> tuple[FloatArray, dict[str, object]]:
    """Load a provenance-checked generalized-Bates target-date fit."""

    report_path = run_directory / "fit_report.json"
    prices_path = run_directory / "initial_fit_prices.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "gambara-teichmann-generalized-bates-target-fit-v1":
        raise RuntimeError(f"unexpected CRC report schema in {report_path}")
    if report.get("target_date") != TARGET_DATE or report.get("quote_count") != len(strikes):
        raise RuntimeError("CRC fit does not use the common target-date quote book")
    if report.get("finite_dlv_chain_used") is not False:
        raise RuntimeError("CRC comparison must use generalized Bates without a DLV chain")
    with np.load(prices_path, allow_pickle=False) as archive:
        expected = {
            "strikes": strikes,
            "is_call": is_call,
            "bids": bids,
            "asks": asks,
            "business_dtes": business_dtes,
        }
        for name, values in expected.items():
            observed = np.asarray(archive[name])
            if observed.shape != values.shape or not np.array_equal(observed, values):
                raise RuntimeError(f"CRC artifact field {name!r} differs from the common book")
        prices = np.asarray(archive["model_prices"], dtype=np.float64)
    if prices.shape != strikes.shape or np.any(~np.isfinite(prices)):
        raise RuntimeError("CRC artifact contains invalid target-date prices")
    return prices, report


def _normalized_moneyness(
    *,
    strikes: FloatArray,
    tau: FloatArray,
    market_iv: FloatArray,
    business_dtes: FloatArray,
) -> FloatArray:
    atm_iv = np.empty(strikes.shape, dtype=np.float64)
    for business_dte in np.unique(business_dtes):
        selected = np.isclose(business_dtes, business_dte, rtol=0.0, atol=1.0e-4)
        log_strikes = np.log(strikes[selected])
        order = np.argsort(log_strikes)
        atm_iv[selected] = np.interp(
            0.0,
            log_strikes[order],
            market_iv[selected][order],
        )
    return np.log(strikes) / (atm_iv * np.sqrt(tau))


def _metrics(
    prices: FloatArray,
    *,
    bids: FloatArray,
    asks: FloatArray,
) -> dict[str, float | int]:
    half_spreads = 0.5 * (asks - bids)
    excess = (
        np.maximum(prices - asks, 0.0) + np.maximum(bids - prices, 0.0)
    ) / half_spreads
    positive = excess[excess > 0.0]
    return {
        "quotes": int(excess.size),
        "percent_positive_excess": float(100.0 * np.mean(excess > 0.0)),
        "conditional_mean_excess_half_spreads": float(np.mean(positive)),
        "conditional_median_excess_half_spreads": float(np.median(positive)),
    }


def _signed_price_excess(
    prices: FloatArray,
    *,
    bids: FloatArray,
    asks: FloatArray,
) -> FloatArray:
    half_spreads = 0.5 * (asks - bids)
    return np.where(
        prices > asks,
        (prices - asks) / half_spreads,
        np.where(prices < bids, (prices - bids) / half_spreads, 0.0),
    )


def _signed_iv_excess(
    model_iv: FloatArray,
    model_prices: FloatArray,
    *,
    bid_iv: FloatArray,
    ask_iv: FloatArray,
    bids: FloatArray,
    asks: FloatArray,
) -> FloatArray:
    above = model_prices > asks
    below = model_prices < bids
    if not np.all(np.isfinite(ask_iv[above])):
        raise RuntimeError("model exceeds an ask without a finite Black volatility")
    if not np.all(np.isfinite(bid_iv[below])):
        raise RuntimeError("model falls below a bid without a finite Black volatility")
    excess = np.zeros(model_iv.shape, dtype=np.float64)
    excess[above] = model_iv[above] - ask_iv[above]
    excess[below] = model_iv[below] - bid_iv[below]
    return 100.0 * excess


def _surface_figure(
    output: Path,
    *,
    nms: FloatArray,
    business_dtes: FloatArray,
    values: tuple[FloatArray, FloatArray, FloatArray],
    title: str,
    colorbar_label: str,
    norm: TwoSlopeNorm | SymLogNorm,
) -> None:
    figure = plt.figure(figsize=(10.8, 4.0))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.065),
        left=0.065,
        right=0.985,
        bottom=0.13,
        top=0.82,
        hspace=0.42,
        wspace=0.17,
    )
    axes = [figure.add_subplot(grid[0, 0])]
    axes.append(figure.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0]))
    axes.append(figure.add_subplot(grid[0, 2], sharex=axes[0], sharey=axes[0]))
    last = None
    for axis, model, model_values in zip(
        axes,
        ("GROMM", "Tangent Levy", "CRC"),
        values,
        strict=True,
    ):
        outside = model_values != 0.0
        last = axis.scatter(
            nms[outside],
            business_dtes[outside],
            c=model_values[outside],
            s=7.0,
            marker="s",
            linewidths=0.0,
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
        )
        outside_percentage = 100.0 * np.mean(outside)
        axis.set_title(model, loc="left", fontsize=10)
        axis.set_title(
            f"{outside_percentage:.2f}% outside",
            loc="right",
            fontsize=10,
            color="#b2182b" if model in {"Tangent Levy", "CRC"} else "black",
        )
        axis.set_xlabel("normalized log-moneyness")
        axis.grid(alpha=0.14, linewidth=0.4)
    axes[0].set_ylabel("business days to expiry")
    assert last is not None
    colorbar_axis = figure.add_subplot(grid[1, :])
    colorbar = figure.colorbar(last, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(colorbar_label)
    figure.suptitle(title, fontsize=11)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def _write_metrics_table(output: Path, metrics: dict[str, dict[str, float | int]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Quotes with excess & Conditional mean & Conditional median\\",
        r"\midrule",
    ]
    for label in ("GROMM", "Tangent Levy", "CRC"):
        values = metrics[label]
        percentage = f"{values['percent_positive_excess']:.2f}\\%"
        if label in {"Tangent Levy", "CRC"}:
            percentage = rf"\textcolor{{red}}{{{percentage}}}"
        lines.append(
            f"{label} & {percentage} & "
            f"{values['conditional_mean_excess_half_spreads']:.3f} & "
            f"{values['conditional_median_excess_half_spreads']:.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Initial-fit excess on the common 29 August 2025 quote book.",
            r"Quotes with excess is the percentage with $e>0$~\eqref{eq:excess}; the final two",
            r"columns report the mean and median of $e$ conditional on $e>0$.",
            r"The GROMM row uses the same checkpoint as",
            r"Table~\ref{tab:numerics-static-market-fit}, restricted to this date.}",
            r"\label{tab:initial-fit-excess}",
            r"\end{table}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(
    *,
    dataset_path: Path,
    tangent_checkpoint: Path,
    crc_run: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    gromm_report = _verified_gromm_report(dataset_path)
    data, metadata, date_index, quote_count = _target_gromm_book(dataset_path)
    dates = np.asarray(metadata["dates"])
    selected = slice(0, quote_count)
    strikes = np.asarray(data["strikes"][date_index, selected], dtype=np.float64)
    is_call = np.asarray(data["is_call"][date_index, selected], dtype=bool)
    bids = np.asarray(data["bids"][date_index, selected], dtype=np.float64)
    asks = np.asarray(data["asks"][date_index, selected], dtype=np.float64)
    mids = np.asarray(data["mids"][date_index, selected], dtype=np.float64)
    business_dtes = np.asarray(
        data["business_dtes"][date_index, selected], dtype=np.float64
    )
    tau = business_dtes / 252.0

    gromm_prices = _gromm_prices(
        data,
        metadata,
        date_index=date_index,
        date_count=len(dates),
        quote_count=quote_count,
    )
    tangent_prices = _tangent_prices(
        tangent_checkpoint,
        strikes=strikes,
        is_call=is_call,
        business_dtes=business_dtes,
    )
    crc_prices, crc_report = _crc_prices(
        crc_run,
        strikes=strikes,
        is_call=is_call,
        bids=bids,
        asks=asks,
        business_dtes=business_dtes,
    )

    market_calls = np.where(is_call, mids, mids + 1.0 - strikes)
    bid_calls = np.where(is_call, bids, bids + 1.0 - strikes)
    ask_calls = np.where(is_call, asks, asks + 1.0 - strikes)
    gromm_calls = np.where(is_call, gromm_prices, gromm_prices + 1.0 - strikes)
    tangent_calls = np.where(
        is_call,
        tangent_prices,
        tangent_prices + 1.0 - strikes,
    )
    crc_calls = np.where(is_call, crc_prices, crc_prices + 1.0 - strikes)
    market_iv = _implied_volatility(market_calls, strikes, tau)
    bid_iv = _implied_volatility(bid_calls, strikes, tau)
    ask_iv = _implied_volatility(ask_calls, strikes, tau)
    bid_iv = np.where(
        np.isclose(
            bid_calls,
            np.maximum(1.0 - strikes, 0.0),
            rtol=0.0,
            atol=1.0e-10,
        ),
        0.0,
        bid_iv,
    )
    gromm_iv = _implied_volatility(gromm_calls, strikes, tau)
    tangent_iv = _implied_volatility(tangent_calls, strikes, tau)
    crc_iv = _implied_volatility(crc_calls, strikes, tau)
    if not np.all(np.isfinite(np.stack((market_iv, gromm_iv, tangent_iv, crc_iv)))):
        raise RuntimeError("initial-fit comparison contains a non-invertible Black price")
    nms = _normalized_moneyness(
        strikes=strikes,
        tau=tau,
        market_iv=market_iv,
        business_dtes=business_dtes,
    )
    signed_half_spread_excess = (
        _signed_price_excess(gromm_prices, bids=bids, asks=asks),
        _signed_price_excess(tangent_prices, bids=bids, asks=asks),
        _signed_price_excess(crc_prices, bids=bids, asks=asks),
    )
    iv_excess_percentage_points = (
        _signed_iv_excess(
            gromm_iv,
            gromm_prices,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            bids=bids,
            asks=asks,
        ),
        _signed_iv_excess(
            tangent_iv,
            tangent_prices,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            bids=bids,
            asks=asks,
        ),
        _signed_iv_excess(
            crc_iv,
            crc_prices,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            bids=bids,
            asks=asks,
        ),
    )

    _surface_figure(
        output / "initial_fit_iv_errors.pdf",
        nms=nms,
        business_dtes=business_dtes,
        values=iv_excess_percentage_points,
        title=f"Initial implied-volatility excess on {TARGET_DATE}",
        colorbar_label="signed excess beyond IV bid--ask (percentage points)",
        norm=TwoSlopeNorm(vmin=-7.5, vcenter=0.0, vmax=7.5),
    )
    _surface_figure(
        output / "initial_fit_iv_errors.png",
        nms=nms,
        business_dtes=business_dtes,
        values=iv_excess_percentage_points,
        title=f"Initial implied-volatility excess on {TARGET_DATE}",
        colorbar_label="signed excess beyond IV bid--ask (percentage points)",
        norm=TwoSlopeNorm(vmin=-7.5, vcenter=0.0, vmax=7.5),
    )
    _surface_figure(
        output / "initial_fit_half_spread_errors.pdf",
        nms=nms,
        business_dtes=business_dtes,
        values=signed_half_spread_excess,
        title=f"Initial price excess on {TARGET_DATE}",
        colorbar_label="signed excess beyond bid--ask / half-spread",
        norm=SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-350.0, vmax=350.0),
    )
    _surface_figure(
        output / "initial_fit_half_spread_errors.png",
        nms=nms,
        business_dtes=business_dtes,
        values=signed_half_spread_excess,
        title=f"Initial price excess on {TARGET_DATE}",
        colorbar_label="signed excess beyond bid--ask / half-spread",
        norm=SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-350.0, vmax=350.0),
    )

    metrics = {
        "GROMM": _metrics(gromm_prices, bids=bids, asks=asks),
        "Tangent Levy": _metrics(tangent_prices, bids=bids, asks=asks),
        "CRC": _metrics(crc_prices, bids=bids, asks=asks),
    }
    _write_metrics_table(output / "initial_fit_metrics.tex", metrics)
    report = {
        "schema": "main2-initial-fit-comparison-v3",
        "target_date": TARGET_DATE,
        "quote_population": "GROMM daily-25 retained market book",
        "quote_count": quote_count,
        "maximum_business_dte": float(np.max(business_dtes)),
        "dataset": str(dataset_path.resolve()),
        "gromm_training_run": GROMM_RUN_LABEL,
        "gromm_checkpoint": str((GROMM_ARTIFACT_DIRECTORY / "checkpoint.npz").resolve()),
        "gromm_pooled_statistics": gromm_report["final_statistics"],
        "tangent_checkpoint": str(tangent_checkpoint.resolve()),
        "tangent_state": "final daily-grid reconstruction used by the v2 Q audit",
        "crc_run": str(crc_run.resolve()),
        "crc_model": crc_report["model_family"],
        "crc_dynamic_audit": crc_report["dynamic_audit"],
        "metrics": metrics,
    }
    temporary = output / ".initial_fit_metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "initial_fit_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GROMM_DATASET)
    parser.add_argument(
        "--tangent-checkpoint",
        type=Path,
        default=DEFAULT_TANGENT_CHECKPOINT,
    )
    parser.add_argument("--crc-run", type=Path, default=DEFAULT_CRC_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(
        dataset_path=arguments.dataset,
        tangent_checkpoint=arguments.tangent_checkpoint,
        crc_run=arguments.crc_run,
        output=arguments.output,
    )


if __name__ == "__main__":
    main()
