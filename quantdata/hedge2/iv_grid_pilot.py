"""Compare declared IV-grid resolutions and extract dated, frozen-source states.

The pilot's numerical gate requires every ordinary ITM option's price error
against the saved SANOS interpolator to be at most 1% of its observed
half-spread, with no wing extrapolation or failed grid inversions. This is a
declared approximation budget relative to trading costs, not a statistical
confidence limit or acceptance of a forecasting/hedging model. Market mids
remain observed quotes and are saved separately from both model prices.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import CashCurve, MarketStore, select_book, zero_strike_mask
from quantdata.hedge2.historical import sanos_europeans
from quantdata.hedge2.iv_grid import IVGrid, IVGridConfig, IVGridExtraction, extract_iv_grid
from quantdata.hedge2.source import SourceArchive
from quantdata.sanos.arbfree import ArbFreePricer


@version("0.1", dependencies=[IVGridConfig])
@dataclass(frozen=True, kw_only=True)
class IVGridPilotConfig:
    grid_config: IVGridConfig = IVGridConfig()
    node_counts: tuple[int, ...] = (29, 100, 200, 400)
    maximum_half_spread_fraction: float = 0.01

    def __post_init__(self) -> None:
        verify_inp(len(self.node_counts) > 0 and all(isinstance(count, int) and not isinstance(count, bool)
                   and count >= 2 for count in self.node_counts) and len(set(self.node_counts)) == len(self.node_counts),
                   lambda: f"Node counts must be distinct integers of at least two: {self.node_counts}")
        verify_inp(np.isfinite(self.maximum_half_spread_fraction) and 0 < self.maximum_half_spread_fraction < 1,
                   lambda: f"Declare a positive sub-spread error budget: {self.maximum_half_spread_fraction}")


def _sha256(path: Path) -> str:
    with path.open("rb") as saved:
        return hashlib.file_digest(saved, "sha256").hexdigest()


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    paths = [package / name for name in ("iv_grid_pilot.py", "iv_grid.py", "source.py", "cash.py", "historical.py")]
    paths.extend([Path(inspect.getfile(ArbFreePricer)), Path(inspect.getfile(type(bs)))])
    return {str(path): _sha256(path) for path in paths}


def _prepare_run(directory: Path, source_root: Path) -> None:
    resolved = directory.resolve()
    if resolved.is_relative_to(source_root.resolve()) or any(
            (parent / ".git" / "HEAD").is_file() or (parent / ".git").is_file()
            for parent in (resolved, *resolved.parents)):
        raise ValueError("IV diagnostic outputs must be outside source archives and every Git working tree")
    directory.mkdir(parents=True, exist_ok=False)


def _save_extraction(path: Path, extraction: IVGridExtraction) -> str:
    arrays = {name: value for name, value in vars(extraction).items() if isinstance(value, np.ndarray)}
    arrays.update({"date": np.array(extraction.grid.day.isoformat()),
        "business_dtes": np.asarray(extraction.grid.config.business_dtes),
        "nms_nodes": extraction.grid.config.nms_nodes,
        "reference_vols": extraction.grid.reference_vols,
        "log_implied_vols": extraction.grid.log_implied_vols, "state": extraction.grid.state})
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _sha256(path)


def compare_book(*, extraction: IVGridExtraction, source: ArbFreePricer,
                 curve: CashCurve, book: pd.DataFrame,
                 maximum_half_spread_fraction: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep observed mids, source prices and IV approximations as distinct columns."""
    verify_inp(np.isfinite(maximum_half_spread_fraction) and 0 < maximum_half_spread_fraction < 1,
               lambda: f"Invalid half-spread error budget: {maximum_half_spread_fraction}")
    synthetic = zero_strike_mask(book)
    ordinary = ~synthetic
    spreads = book["observed_half_spread"].to_numpy(dtype=float)
    mids = book["observed_mid"].to_numpy(dtype=float)
    if (not ordinary.any() or not np.isfinite(spreads).all() or np.any(spreads[ordinary] <= 0)
            or np.any(spreads[synthetic] != 0) or not np.isfinite(mids).all() or np.any(mids <= 0)):
        raise ValueError("Comparison requires observed positive ordinary premiums/spreads and explicit cost-free forwards")
    days = book["bdte"].to_numpy(dtype=float)
    strikes = book["strike"].to_numpy(dtype=float)
    calls = book["option_type"].to_numpy() == "call"
    priced = extraction.grid.price_europeans(curve=curve, business_dtes=days, cash_strikes=strikes, is_call=calls)
    source_prices = sanos_europeans(source=source, curve=curve, business_dtes=days,
                                    cash_strikes=strikes, is_call=calls)
    errors = priced.prices - source_prices
    error_fractions = np.abs(errors[ordinary]) / spreads[ordinary]
    frame = book.copy()
    frame["source_price"], frame["iv_price"] = source_prices, priced.prices
    frame["iv_source_error_cash"] = errors
    frame["source_mid_error_cash"] = source_prices - mids
    frame["iv_mid_error_cash"] = priced.prices - mids
    frame["iv_source_error_fraction_half_spread"] = np.nan
    frame.loc[ordinary, "iv_source_error_fraction_half_spread"] = error_fractions
    frame["query_nms"] = priced.nms
    frame["nms_extrapolated"] = priced.nms_extrapolated
    frame["pricing_forward"], frame["pricing_discount"] = priced.forwards, priced.discounts
    exceeded = error_fractions > maximum_half_spread_fraction
    summary = {**extraction.summary(), "num_nms_nodes": extraction.grid.config.num_nms_nodes,
        "ordinary_contracts": int(ordinary.sum()), "synthetic_forwards": int(synthetic.sum()),
        "maximum_half_spread_fraction": maximum_half_spread_fraction,
        "max_error_fraction_half_spread": float(error_fractions.max()),
        "p99_error_fraction_half_spread": float(np.quantile(error_fractions, 0.99)),
        "above_half_spread_budget": int(exceeded.sum()),
        "nms_extrapolated_contracts": int(priced.nms_extrapolated.sum()),
        "max_abs_error_bps_spot": float(np.max(np.abs(errors)) / curve.spot * 10000.),
        "error_rmse_bps_spot": float(np.sqrt(np.mean(errors[ordinary]**2)) / curve.spot * 10000.),
        "max_abs_source_mid_error_bps_spot": float(np.max(np.abs(source_prices[ordinary] - mids[ordinary])) / curve.spot * 10000.),
        "max_abs_synthetic_error_cash": float(np.max(np.abs(errors[synthetic]))) if synthetic.any() else 0.,
        "pilot_european_pricing_gate_passed": bool(not exceeded.any() and not priced.nms_extrapolated.any()
            and not extraction.inversion_failed.any() and not extraction.inversion_cap.any())}
    return frame, summary


def select_resolution(*, records: list[dict[str, object]], days: tuple[date, ...],
                      node_counts: tuple[int, ...]) -> int | None:
    """Choose the smallest resolution passing every requested pilot date."""
    for count in sorted(node_counts):
        matching = [record for record in records if record.get("num_nms_nodes") == count]
        if (len(matching) == len(days) and {record["date"] for record in matching} == {day.isoformat() for day in days}
                and all(record.get("pilot_european_pricing_gate_passed", False) for record in matching)):
            return count
    return None


def run_comparison(*, source_root: Path, licensed_fit_directory: Path, market_directory: Path,
                   raw_quote_directory: Path | None, output_directory: Path, days: tuple[date, ...],
                   config: IVGridPilotConfig = IVGridPilotConfig()) -> dict[str, object]:
    """Compare actual eligible ITM books, including their explicit prepaid forwards."""
    verify_inp(len(days) > 0 and len(set(days)) == len(days), lambda: "Pilot dates must be nonempty and distinct")
    _prepare_run(output_directory, source_root)
    manifest = {"schema": "hedge2-iv-grid-comparison-v1", "config": asdict(config),
        "dates": [day.isoformat() for day in days], "source_root": str(source_root),
        "licensed_fit_directory": str(licensed_fit_directory), "market_directory": str(market_directory),
        "raw_quote_directory": None if raw_quote_directory is None else str(raw_quote_directory),
        "book": "existing select_book; ITM side, strict -2<NMS<2, front through first >=5D expiry; explicit prepaid forwards",
        "price_reference": "frozen SANOS market interpolator; observed mids and spreads retained separately",
        "declared_gate": "every ordinary option <= configured fraction of observed half-spread; no wing extrapolation, cap, or inversion error",
        "scope": "pilot European repricing only; no forecasting, hedging, or no-arbitrage acceptance",
        "implementation_sha256": _implementation_hashes()}
    (output_directory / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                        raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    records = []
    for day in days:
        source, metadata = archive.load(day)
        market = store.load(day)
        vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
        book = select_book(market, atm_vols=vols, option_side="itm", include_co_terminal_forwards=True)
        directory = output_directory / day.isoformat()
        directory.mkdir()
        book_path = directory / "book.parquet"
        book.to_parquet(book_path, index=False)
        inputs = [market_directory / "days" / f"{day.isoformat()}.parquet",
                  licensed_fit_directory / "SPX" / f"{day.isoformat()}.pdq"]
        if raw_quote_directory is not None:
            inputs.append(raw_quote_directory / "SPX" / f"{day.isoformat()}.pdq")
        provenance = {**metadata, "book_sha256": _sha256(book_path),
                      "input_sha256": {str(path): _sha256(path) for path in inputs}}
        (directory / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for count in config.node_counts:
            started = time.monotonic()
            extraction = extract_iv_grid(source=source, day=day, config=replace(config.grid_config, num_nms_nodes=count))
            compared, record = compare_book(extraction=extraction, source=source, curve=market.curve,
                                            book=book, maximum_half_spread_fraction=config.maximum_half_spread_fraction)
            state_path = directory / f"nms-{count}.npz"
            prices_path = directory / f"nms-{count}-prices.parquet"
            state_hash = _save_extraction(state_path, extraction)
            compared.to_parquet(prices_path, index=False)
            record.update({"elapsed_seconds": time.monotonic() - started, "source_sha256": metadata["source_sha256"],
                           "artifacts_sha256": {state_path.name: state_hash, prices_path.name: _sha256(prices_path)}})
            records.append(record)
            print(json.dumps(record), flush=True)
    selected = select_resolution(records=records, days=days, node_counts=config.node_counts)
    summary = {"schema": manifest["schema"], "selected_num_nms_nodes": selected,
        "selection": "smallest tested resolution passing the declared pilot European-pricing gate on every date",
        "maximum_half_spread_fraction": config.maximum_half_spread_fraction,
        "scope": manifest["scope"], "records": records}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_dated_extraction(*, source_root: Path, output_directory: Path,
                         start: date, end: date, config: IVGridConfig) -> dict[str, object]:
    """Extract every available source date in the explicit interval, without fitting dynamics.

    Dates that fail extraction remain failed and are never filled or replaced.
    A new output directory is required; each date has a hashed completion record.
    Numerical floor/cap and round-trip diagnostics accompany every state.
    """
    verify_inp(end >= start, lambda: f"Extraction end precedes start: {start}, {end}")
    archive = SourceArchive(root=source_root)
    days = tuple(day for day in archive.dates() if start <= day <= end)
    verify_inp(len(days) > 0, lambda: f"No saved source dates between {start} and {end}")
    _prepare_run(output_directory, source_root)
    directory = output_directory / "days"
    directory.mkdir()
    configuration = {"schema": "hedge2-dated-iv-grid-v1", "grid_config": asdict(config),
        "grid_config_version": IVGridConfig.version.unique_id48, "source_root": str(source_root),
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "available_first": days[0].isoformat(), "available_last": days[-1].isoformat(),
        "available_dates": [day.isoformat() for day in days], "implementation_sha256": _implementation_hashes(),
        "scope": "frozen-source extraction only; no fitted PCA, AR or hedging result"}
    config_path = output_directory / "config.json"
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    config_hash = _sha256(config_path)
    started = time.monotonic()
    records = []
    for index, day in enumerate(days):
        try:
            source, metadata = archive.load(day)
            extraction = extract_iv_grid(source=source, day=day, config=config)
            path = directory / f"{day.isoformat()}.npz"
            record = {**extraction.summary(), **metadata, "status": "extracted",
                "config_sha256": config_hash, "arrays_sha256": _save_extraction(path, extraction)}
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            record = {"date": day.isoformat(), "status": "failed", "issue": str(error)}
        (directory / f"{day.isoformat()}.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        if (index + 1) % 25 == 0 or index + 1 == len(days):
            elapsed = time.monotonic() - started
            print(f"IV extraction completed={index + 1}/{len(days)} elapsed_s={elapsed:.1f} "
                  f"remaining_s={elapsed * (len(days) - index - 1) / (index + 1):.1f}", flush=True)
    failed = sum(record["status"] == "failed" for record in records)
    summary = {"schema": configuration["schema"], "status": "complete" if not failed else "partial",
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "available_first": days[0].isoformat(), "available_last": days[-1].isoformat(),
        "available_dates": len(days), "extracted_dates": len(days) - failed, "failed_dates": failed,
        "trailing_source_gap": days[-1] < end, "elapsed_seconds": time.monotonic() - started,
        "scope": configuration["scope"], "records": records}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_dated_grid(directory: Path, day: date) -> IVGrid:
    """Load only a completed, hashed dated state and its original NMS geometry."""
    config_path = directory / "config.json"
    configuration = json.loads(config_path.read_text())
    if (configuration.get("schema") != "hedge2-dated-iv-grid-v1"
            or configuration.get("grid_config_version") != IVGridConfig.version.unique_id48):
        raise ValueError("Dated IV-grid schema or configuration version is incompatible")
    record = json.loads((directory / "days" / f"{day.isoformat()}.json").read_text())
    path = directory / "days" / f"{day.isoformat()}.npz"
    if (record.get("status") != "extracted" or record.get("date") != day.isoformat()
            or record.get("config_sha256") != _sha256(config_path)
            or record.get("arrays_sha256") != _sha256(path)):
        raise ValueError(f"Dated IV grid is incomplete or its hashes changed: {day}")
    config = IVGridConfig(**configuration["grid_config"])
    with np.load(path, allow_pickle=False) as arrays:
        if (str(arrays["date"].item()) != day.isoformat()
                or not np.array_equal(arrays["business_dtes"], config.business_dtes)
                or not np.array_equal(arrays["nms_nodes"], config.nms_nodes)):
            raise ValueError(f"Saved IV state and configured date/geometry disagree: {day}")
        return IVGrid(day=day, config=config, reference_vols=arrays["reference_vols"],
                      log_implied_vols=arrays["log_implied_vols"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare grid sizes against actual ITM hedge books")
    compare.add_argument("--licensed-fit-directory", type=Path, required=True)
    compare.add_argument("--market-directory", type=Path, required=True)
    compare.add_argument("--raw-quote-directory", type=Path)
    compare.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    compare.add_argument("--node-counts", type=int, nargs="+", default=(29, 100, 200, 400))
    compare.add_argument("--maximum-half-spread-fraction", type=float, default=0.01)
    extract = commands.add_parser("extract", help="Extract dated states only, without fitting dynamics")
    extract.add_argument("--start", type=date.fromisoformat, required=True)
    extract.add_argument("--end", type=date.fromisoformat, required=True)
    extract.add_argument("--num-nms-nodes", type=int, required=True)
    for command in (compare, extract):
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        run_comparison(source_root=args.source_root, licensed_fit_directory=args.licensed_fit_directory,
            market_directory=args.market_directory, raw_quote_directory=args.raw_quote_directory,
            output_directory=args.output_directory, days=tuple(args.dates),
            config=IVGridPilotConfig(node_counts=tuple(args.node_counts),
                                     maximum_half_spread_fraction=args.maximum_half_spread_fraction))
    else:
        run_dated_extraction(source_root=args.source_root, output_directory=args.output_directory,
            start=args.start, end=args.end, config=IVGridConfig(num_nms_nodes=args.num_nms_nodes))


if __name__ == "__main__":
    main()
