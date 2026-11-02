"""Cboe discrete VIX calculation on the cached spreads supplied to smooth SANOS.

This is a filtered-input diagnostic, not a recreation of the official index:
missing market quotes cannot be restored from a filtered calibration book.
The raw-market expiry audit supplies the same expiries and original zero-bid
stopping strikes, preventing removed zero bids from reopening either wing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.vixdata.formula import (
    constant_maturity_vix,
    discrete_total_variance,
    select_vix_strip,
)
from quantdata.widb.arbfree import mk_arb_free_day
from quantdata.widb.day import ExpiryFilter, RawDayFilter
from quantdata.widb.options_filter import OptionsFilter

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

OPTIONS_FILTER = OptionsFilter(max_num_options=100000, max_business_days_to_expiry=70, max_nms=(-100., 100.))
VARIANTS = ("sanos_input", "sanos_retained", "real_books_on_sanos_input")


def filtered_term(
    *, strikes: np.ndarray, call_bids: np.ndarray, call_asks: np.ndarray,
    put_bids: np.ndarray, put_asks: np.ndarray, retained: np.ndarray,
    raw_put_stop: float | None, raw_call_stop: float | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Apply the formula in forward units (K/F, price/(DF*F)), with DF=1.

Cutoffs must use those same strike units. Recompute deltaK on the surviving
strip; do not reuse quadrature weights from the full market strip.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    retained = np.asarray(retained, dtype=bool)
    verify_inp(retained.shape == strikes.shape, lambda: "Retained mask must match strikes")
    cb, ca, pb, pa = [np.asarray(book, dtype=np.float64)[retained].copy()
                      for book in (call_bids, call_asks, put_bids, put_asks)]
    k = strikes[retained]
    # The training pipeline has discarded zero bids. Preserve the stopping
    # information from the unfiltered chain, instead of treating NaNs as zeros.
    if raw_put_stop is not None:
        pb[k <= raw_put_stop] = np.nan
        pa[k <= raw_put_stop] = np.nan
    if raw_call_stop is not None:
        cb[k >= raw_call_stop] = np.nan
        ca[k >= raw_call_stop] = np.nan
    strip = select_vix_strip(strikes=k, call_bids=cb, call_asks=ca, put_bids=pb, put_asks=pa, discount_factor=1.)
    mid_variance = discrete_total_variance(strip, discount_factor=1.)
    # Conditional quote bands hold the midpoint-derived F, K0 and selected
    # contracts fixed. They are not official VWB/VWA index replications.
    def prices(calls: np.ndarray, puts: np.ndarray) -> np.ndarray:
        q = np.where(strip.strikes < strip.k0, puts[strip.indices], calls[strip.indices])
        at_k0 = strip.strikes == strip.k0
        q[at_k0] = (puts[strip.indices][at_k0] + calls[strip.indices][at_k0]) / 2
        return q
    bid_prices, ask_prices = prices(cb, pb), prices(ca, pa)
    bid_variance = discrete_total_variance(strip, discount_factor=1., prices=bid_prices)
    ask_variance = discrete_total_variance(strip, discount_factor=1., prices=ask_prices)
    result = dict(total_variance=mid_variance, bid_total_variance=bid_variance, ask_total_variance=ask_variance,
                  selected_strikes=len(strip.strikes), minimum_normalized_strike=float(strip.strikes[0]),
                  maximum_normalized_strike=float(strip.strikes[-1]), normalized_forward=strip.forward,
                  normalized_k0=strip.k0, retained_inputs=int(retained.sum()))
    constituents = [dict(normalized_strike=float(x), delta_normalized_strike=float(dk),
                         normalized_price=float(q), normalized_bid=float(bid), normalized_ask=float(ask))
                    for x, dk, q, bid, ask in zip(strip.strikes, strip.delta_strikes, strip.prices,
                                                bid_prices, ask_prices, strict=True)]
    return result, constituents


def _day(day: str, audit: list[dict[str, object]]) -> tuple[dict, list[dict], list[dict], dict]:
    payload = smooth_interpolated_market("SPX", date.fromisoformat(day), options_filter=OPTIONS_FILTER,
                                         override_cache_mode="cacheonly", verbose=None)
    if payload is None:
        raise ValueError(f"Missing original SANOS market cache for {day}; no recomputation attempted")
    # Recover the normalization from its cached upstream table, not a new fit.
    upstream = mk_arb_free_day(ticker="SPX", d=date.fromisoformat(day), override_cache_mode="cacheonly", verbose=Context.quiet)
    if upstream is None:
        raise ValueError(f"Missing cached normalization for {day}")
    market = payload["market"]
    summaries, constituents = [], []
    digest = hashlib.sha256()
    record = dict(date=date.fromisoformat(day))
    for term in sorted(audit, key=lambda x: x["calendar_days"]):
        matches = [j for j, (expiry, is_am) in enumerate(zip(market["expiry_dates"], market["is_am"], strict=True))
                   if str(expiry) == term["expiry"] and bool(is_am) == term["is_am"]]
        if len(matches) != 1:
            for variant in VARIANTS:
                record[variant + "_reason"] = f"Filtered input lacks expiry {term['expiry']}"
            continue
        j = matches[0]
        source = upstream[(upstream["expiry"].astype(str) == term["expiry"]) & (upstream["is_am"] == term["is_am"])]
        forwards = source["fwd"].unique()
        if len(forwards) != 1:
            raise ValueError(f"Ambiguous cached forward for {day}/{term['expiry']}")
        forward = float(forwards[0])
        k = np.asarray(market["strikes"][j], dtype=np.float64)
        # Verify that the normalization table still corresponds to the cached
        # training input, rather than silently mixing cache generations.
        available = np.sort(source["pure_strike"].to_numpy(dtype=np.float64))
        ix = np.searchsorted(available, k).clip(0, len(available) - 1)
        if not np.allclose(available[ix], k, rtol=0, atol=1e-8):
            raise ValueError(f"Cached SANOS strikes and upstream table disagree on {day}")
        c_bid, c_ask = [np.asarray(market[key][j], dtype=np.float64) for key in ("bids", "asks")]
        put_bid, put_ask = c_bid - 1. + k, c_ask - 1. + k
        european = ~np.asarray(market["is_american"][j], dtype=bool)
        raw_books = [np.asarray(market[key][j], dtype=np.float64)
                     for key in ("call_bids", "call_asks", "put_bids", "put_asks")]
        model_mask = np.asarray(market["model_masks"][j], dtype=bool)
        for name, value in zip(("k", "bid", "ask", "cb", "ca", "pb", "pa", "mask"),
                               (k, c_bid, c_ask, *raw_books, model_mask), strict=True):
            digest.update(f"{term['expiry']}/{term['is_am']}/{name}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        for variant in VARIANTS:
            books = raw_books if variant == "real_books_on_sanos_input" else [c_bid, c_ask, put_bid, put_ask]
            mask = european & model_mask if variant == "sanos_retained" else european
            try:
                summary, selected = filtered_term(strikes=k, call_bids=books[0], call_asks=books[1],
                    put_bids=books[2], put_asks=books[3], retained=mask,
                    raw_put_stop=None if term["put_zero_bid_stop"] is None else term["put_zero_bid_stop"] / forward,
                    raw_call_stop=None if term["call_zero_bid_stop"] is None else term["call_zero_bid_stop"] / forward)
            except ValueError as error:
                record[variant + "_reason"] = f"{term['expiry']}: {error}"
                continue
            identity = dict(date=day, expiry=term["expiry"], is_am=term["is_am"], variant=variant)
            summaries.append(dict(**identity, **summary, calendar_days=term["calendar_days"],
                                  pipeline_forward=forward, raw_put_stop=term["put_zero_bid_stop"],
                                  raw_call_stop=term["call_zero_bid_stop"]))
            constituents.extend(dict(**identity, **row) for row in selected)
    for variant in VARIANTS:
        pair = [row for row in summaries if row["variant"] == variant]
        for suffix in ("", "_bid", "_ask"):
            record[variant + suffix] = None
        if len(pair) != 2:
            record.setdefault(variant + "_reason", "Missing eligible filtered expiry pair")
            continue
        near, far = pair
        for suffix, key in (("", "total_variance"), ("_bid", "bid_total_variance"), ("_ask", "ask_total_variance")):
            record[variant + suffix] = constant_maturity_vix(near_days=near["calendar_days"], next_days=far["calendar_days"],
                                                            near_total_variance=near[key], next_total_variance=far[key])
        record[variant + "_reason"] = None
    return record, summaries, constituents, dict(date=day, selected_cached_inputs_sha256=digest.hexdigest())


def _metrics(values: np.ndarray, actual: np.ndarray) -> dict:
    valid = np.isfinite(values) & np.isfinite(actual)
    error = values[valid] - actual[valid]
    return dict(observations=len(error), bias_vol_points=float(error.mean()), mae_vol_points=float(np.abs(error).mean()),
                rmse_vol_points=float(np.sqrt(np.mean(error**2)))) if len(error) else dict(observations=0)


def run(*, baseline_directory: Path, output_directory: Path, workers: int = 4, limit: int | None = None) -> dict:
    """Compare filtered input spreads, their raw books, full market, and VIX."""
    verify_inp(1 <= workers <= 4 and (limit is None or limit > 0), lambda: "Invalid worker count or limit")
    output_directory.mkdir(parents=True, exist_ok=False)
    base = pl.read_parquet(baseline_directory / "vix_comparison.parquet")
    audits = pl.read_parquet(baseline_directory / "expiry_audit.parquet")
    dates = base.filter(pl.col("market_discrete").is_not_null())["date"].cast(pl.String).to_list()
    dates = dates if limit is None else dates[:limit]
    groups = {day: audits.filter(pl.col("date") == day).to_dicts() for day in dates}
    started = time.monotonic()
    records, terms, contracts, manifest = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (record, summaries, selected, provenance) in enumerate(pool.map(lambda day: _day(day, groups[day]), dates)):
            records.append(record)
            terms.extend(summaries)
            contracts.extend(selected)
            manifest.append(provenance)
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(dates):
                elapsed = time.monotonic() - started
                print(json.dumps(dict(completed=index + 1, total=len(dates), elapsed_seconds=elapsed,
                                      estimated_remaining_seconds=elapsed * (len(dates) - index - 1) / (index + 1))), flush=True)
    frame = base.join(pl.from_dicts(records, infer_schema_length=None), on="date", how="left")
    metrics = {name: _metrics(frame[name].to_numpy().astype(float), frame["cboe_vix"].to_numpy())
               for name in (*VARIANTS, "market_discrete", "gromm_discrete")}
    common = frame.filter(pl.all_horizontal([pl.col(name).is_not_null() for name in (*VARIANTS, "market_discrete", "gromm_discrete")]))
    common_metrics = {name: _metrics(common[name].to_numpy().astype(float), common["cboe_vix"].to_numpy())
                      for name in (*VARIANTS, "market_discrete", "gromm_discrete")}
    def sha(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(schema="sanos-filtered-input-vix-v1", baseline_directory=str(baseline_directory),
        methodology="https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf",
        code_sha256=sha(Path(__file__)), formula_sha256=sha(Path(__file__).parents[1] / "vixdata/formula.py"),
        baseline_hashes={name: sha(baseline_directory / name) for name in ("vix_comparison.parquet", "expiry_audit.parquet", "report.json")},
        options_filter=asdict(OPTIONS_FILTER), raw_day_filter=asdict(RawDayFilter()), expiry_filter=asdict(ExpiryFilter()),
        metrics=metrics, common_sample_metrics=common_metrics,
        reference_date=frame.filter(pl.col("date") == date(2025, 8, 29)).with_columns(pl.col("date").cast(pl.String)).to_dicts(),
        unavailable={name: frame.filter(pl.col(name + "_reason").is_not_null()).group_by(name + "_reason").len().to_dicts() for name in VARIANTS},
        conventions=[
            "sanos_input uses midpoints of the synthetic call-equivalent bid/ask spreads actually passed to the cached smooth SANOS fit; puts follow normalized parity. No fitted SANOS prices are used.",
            "sanos_retained additionally applies cached model_masks; real_books_on_sanos_input uses archived actual call/put mids at the input strikes without that extra mask.",
            "Original raw-market zero-bid stopping strikes are preserved from the baseline audit. Zero bids deleted by training filters are not reinterpreted as absent positive liquidity.",
            "Each filtered strip independently infers F and K0 and recomputes deltaK. The same 23–37-calendar-day Friday expiry pair and minute-accurate 30-day total-variance interpolation as the baseline are used.",
            "K/F and C/(DF*F) give the identical dimensionless formula with DF=1. The pipeline normalization uses its cached weighted-parity forward and discount factor.",
            "Bid/ask sensitivity holds midpoint F, K0 and constituents fixed; it is not an official VWB/VWA replication.",
            "Filtered strips are incomplete and are not official VIX. EOD timing, rate source and intraday dissemination-filter limitations of the raw-market baseline also remain.",
        ], elapsed_seconds=time.monotonic() - started)
    frame.write_parquet(output_directory / "vix_comparison.parquet")
    frame.write_csv(output_directory / "vix_comparison.csv")
    pl.from_dicts(terms, infer_schema_length=None).write_parquet(output_directory / "expiry_audit.parquet")
    pl.from_dicts(contracts, infer_schema_length=None).write_parquet(output_directory / "selected_contracts.parquet")
    (output_directory / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    x = frame["date"].to_list()
    axes[0].plot(x, frame["cboe_vix"], color="black", lw=.9, label="Official VIX close")
    for name, color, label in (("sanos_input", "#d6604d", "SANOS input spreads"),
                                ("real_books_on_sanos_input", "#756bb1", "Real books at SANOS input strikes"),
                                ("market_discrete", "#2166ac", "Full raw market")):
        axes[0].plot(x, frame[name], color=color, lw=.7, label=label)
        axes[1].plot(x, frame[name] - frame["cboe_vix"], color=color, lw=.7, label=label)
    axes[0].set_title("Cboe VIX formula — filtered SANOS market inputs (not model prices)")
    axes[0].set_ylabel("Volatility points")
    axes[1].set_ylabel("Difference from official VIX")
    axes[1].axhline(0, color="grey", lw=.7)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_directory / "vix_comparison.pdf")
    fig.savefig(output_directory / "vix_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
