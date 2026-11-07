"""Current-book one-day unhedged and analytic Black references with raw cash replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.black_hedge import black_digital_exposures, black_digital_hedge
from quantdata.hedge2.cash import MarketStore, observed_marks, select_book, zero_strike_mask
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.source import SourceArchive, surface_digitals, surface_implied_vols


def score_reference_positions(*, positions: np.ndarray, initial_spot: float,
                              cash_prices: np.ndarray, cash_gains: np.ndarray,
                              half_spreads: np.ndarray, next_half_spreads: np.ndarray,
                              discount: float, signed_target_gain: float) -> tuple[dict[str, float | int | bool], np.ndarray]:
    """Convert spot-normalized coefficients to cash holdings and current premium dollars.

    Analytic references are not clipped to the optimizer's box. Breaches of
    the $10 comparison threshold are recorded, not hidden. Only entry costs
    reduce realized P&L; a hypothetical exit fee is a separate statistic.
    """
    values = [np.asarray(value, dtype=float) for value in
        (positions, cash_prices, cash_gains, half_spreads, next_half_spreads)]
    holdings, prices, gains, spreads, next_spreads = values
    if (holdings.ndim != 1 or not len(holdings) or any(value.shape != holdings.shape for value in values)
            or not all(np.isfinite(value).all() for value in values) or np.any(prices <= 0)
            or np.any(spreads < 0) or np.any(next_spreads < 0) or spreads[0] != 0 or next_spreads[0] != 0
            or not np.isfinite([initial_spot, discount, signed_target_gain]).all()
            or initial_spot <= 0 or discount <= 0 or not np.isclose(prices[0], initial_spot, rtol=1e-12)):
        raise ValueError("Reference holdings, cash prices/gains, free spot and costs must align")
    cash_holdings = holdings / initial_spot
    premium_positions = cash_holdings * prices
    gain = float(cash_holdings @ gains)
    fee = float(np.abs(cash_holdings) @ spreads)
    return {"hedge_gain": gain, "entry_cost": fee, "costed_pnl": signed_target_gain + gain - fee,
        "hypothetical_exit_cost": float(discount * np.abs(cash_holdings) @ next_spreads),
        "gross_positions": float(np.abs(premium_positions).sum()),
        "maximum_absolute_premium_position": float(np.abs(premium_positions).max()),
        "cap_hits": int(np.sum(np.abs(premium_positions) >= 10. * (1. - 1e-6))),
        "comparison_box_exceeded": bool(np.any(np.abs(premium_positions) > 10. * (1. + 1e-12))),
        "reference_is_uncapped": True}, premium_positions


def run_references(*, source_configuration: Path, coverage_path: Path,
                   output_directory: Path) -> dict[str, object]:
    """Use the exact current ITM complete-book cohort; retain every portfolio and source hash."""
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Reference outputs must be outside Git working trees")
    sources = json.loads(source_configuration.read_text())
    coverage_configuration_path = coverage_path.parent / "config.json"
    coverage_configuration = json.loads(coverage_configuration_path.read_text())
    if (coverage_configuration.get("option_side") != "itm" or not coverage_configuration.get("include_co_terminal_forwards")
            or any(Path(coverage_configuration[name]).resolve() != Path(sources[name]).resolve()
                for name in ("source_root", "market_directory", "licensed_fit_directory", "raw_quote_directory"))):
        raise ValueError("Reference coverage must identify the same current ITM/forward source population")
    coverage = pd.read_parquet(coverage_path)
    dates = coverage.loc[(coverage["group"] == "digital-1D") & coverage["eligible"], "origin"].tolist()
    if not dates or dates != sorted(set(dates)) or any(not date(2023, 1, 1) <= date.fromisoformat(value) <= date(2025, 12, 31) for value in dates):
        raise ValueError("Reference dates must be distinct current-book complete cases in 2023--2025")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        implementation = sorted(Path(__file__).parent.glob("*.py"))
        identity = {"method": "references", "dates": dates, "itm_premium_book": True,
            **{name: sources[name] for name in ("source_root", "market_directory", "licensed_fit_directory", "raw_quote_directory")},
            "source_configuration": str(source_configuration.resolve()),
            "source_configuration_sha256": hashlib.sha256(source_configuration.read_bytes()).hexdigest(),
            "coverage_path": str(coverage_path.resolve()), "coverage_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
            "coverage_configuration_sha256": hashlib.sha256(coverage_configuration_path.read_bytes()).hexdigest(),
            "spot_cost": 0., "option_spread_multiplier": 1., "entry_only_costs": True,
            "black_iv_convention": "same saved OTM prices; relative log-price inversion with cdxcore Black pricing",
            "capacity": "analytic references are uncapped; $10 net-instrument breaches are reported, not clipped",
            "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}}
        config_path = output_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != identity:
            raise ValueError("Reference run belongs to different code, inputs or cohort")
        config_path.write_text(json.dumps(identity, indent=2) + "\n")
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as archive:
                for path in implementation:
                    archive.write(path, arcname=path.name)
        store = MarketStore(**{name: Path(sources[name]) for name in
            ("market_directory", "licensed_fit_directory", "raw_quote_directory")})
        archive = SourceArchive(root=Path(sources["source_root"]))
        rows, exclusions, started = [], [], time.monotonic()
        for date_index, value in enumerate(dates):
            completion = output_directory / f"{value}.complete.json"
            table = output_directory / f"{value}.parquet"
            if completion.exists():
                for name, digest in json.loads(completion.read_text()).items():
                    if hashlib.sha256((output_directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Completed reference evidence changed: {name}")
                rows.extend(pd.read_parquet(table).to_dict("records"))
                exclusions.extend(json.loads((output_directory / f"{value}-inputs.json").read_text())["reference_exclusions"])
                continue
            day = date.fromisoformat(value)
            following_day = get_business_days(day, day + timedelta(days=10))[1]
            current, following = store.load(day), store.load(following_day)
            source, source_metadata = archive.load(day)
            atm_vols = np.asarray(source.interpolate_atm_vols(current.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
            book = select_book(current, atm_vols=atm_vols, option_side="itm", include_co_terminal_forwards=True)
            vols = np.asarray(source.interpolate_atm_vols(book["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR)).copy()
            ordinary = ~zero_strike_mask(book)
            if ordinary.any():
                ordinary_vols = surface_implied_vols(source, expiries_y=book.loc[ordinary, "bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR,
                    strikes=book.loc[ordinary, "strike"].to_numpy(dtype=float) / book.loc[ordinary, "fwd"].to_numpy(dtype=float),
                )
                vols[ordinary] = ordinary_vols
            claims = tuple(claim for claim in make_claims(initial_spot=current.curve.spot,
                atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
                if claim.kind == "digital" and claim.terminal_day == 1)
            following_marks, following_spreads = observed_marks(book, following)
            prices = np.r_[current.curve.spot, book["observed_mid"].to_numpy(dtype=float)]
            spreads = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float)]
            forward, reference, discount = current.curve.at(1.)
            gains = np.r_[discount * (following.curve.spot - reference), discount * following_marks - prices[1:]]
            dated_rows, portfolios, dated_exclusions = [], [], []
            for claim in claims:
                mark = float(discount * surface_digitals(source, expiry_y=1. / DAYS_PER_YEAR,
                    strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0])
                target_vols = surface_implied_vols(source, expiries_y=np.array([1. / DAYS_PER_YEAR]),
                    strikes=np.array([claim.strike / forward]))
                target_vol = float(target_vols[0])
                delta, _ = black_digital_exposures(claim=claim, curve=current.curve, target_vol=target_vol)
                for side in (-1, 1):
                    target_gain = side * (float(discount * claim.payoff(following.curve.spot)) - mark)
                    delta_positions = np.zeros(len(prices))
                    delta_positions[0] = -side * delta * current.curve.spot
                    candidates = [("unhedged", np.zeros(len(prices))), ("BS delta", delta_positions)]
                    try:
                        candidates.append(("BS delta+vega", black_digital_hedge(claim=claim, curve=current.curve,
                            book=book, target_vol=target_vol, option_vols=vols, target_side=side)))
                    except ValueError as error:
                        if str(error) not in ("The Black vega hedge needs two surrounding observed option strikes",
                                              "The Black vega hedge has no expiry at or after the target"):
                            raise
                        dated_exclusions.append({"date": value, "claim": claim.label, "side": side,
                            "method": "BS delta+vega", "exact_error": str(error)})
                    for method, positions in candidates:
                        accounting, allocations = score_reference_positions(positions=positions,
                            initial_spot=current.curve.spot, cash_prices=prices, cash_gains=gains,
                            half_spreads=spreads, next_half_spreads=np.r_[0., following_spreads],
                            discount=float(discount), signed_target_gain=target_gain)
                        dated_rows.append({"date": value, "next_date": following_day.isoformat(), "claim": claim.label,
                            "side": side, "method": method, "market_target_price": mark,
                            "target_gain": target_gain, "target_vol": target_vol, **accounting})
                        portfolios.append(allocations)
            inputs = {"source_model": source_metadata, "reference_exclusions": dated_exclusions,
                "target_mark": "common source-SANOS digital; not an observed digital quote",
                "synthetic_option_vols": "unused ATM placeholders; zero-strike calls never form Black brackets",
                "input_sha256": {f"{kind}_{selected}": hashlib.sha256((Path(sources[kind]) / suffix /
                    f"{selected}.{extension}").read_bytes()).hexdigest()
                    for kind, suffix, extension in (("market_directory", "days", "parquet"),
                        ("licensed_fit_directory", "SPX", "pdq"), ("raw_quote_directory", "SPX", "pdq"))
                    for selected in (value, following_day.isoformat())}}
            artifacts = {table.name: pd.DataFrame(dated_rows), f"{value}-book.parquet": book}
            for name, frame in artifacts.items():
                destination = output_directory / name
                temporary = destination.with_suffix(".partial.parquet")
                frame.to_parquet(temporary, index=False)
                temporary.replace(destination)
            portfolio_path = output_directory / f"{value}-positions.npz"
            temporary_positions = portfolio_path.with_suffix(".partial.npz")
            np.savez_compressed(temporary_positions, positions=np.asarray(portfolios), option_vols=vols)
            temporary_positions.replace(portfolio_path)
            inputs_path = output_directory / f"{value}-inputs.json"
            inputs_path.write_text(json.dumps(inputs, indent=2) + "\n")
            names = (*artifacts, portfolio_path.name, inputs_path.name)
            marker = {name: hashlib.sha256((output_directory / name).read_bytes()).hexdigest() for name in names}
            temporary_completion = completion.with_suffix(".partial.json")
            temporary_completion.write_text(json.dumps(marker, indent=2) + "\n")
            temporary_completion.replace(completion)
            rows.extend(dated_rows)
            exclusions.extend(dated_exclusions)
            if (date_index + 1) % 25 == 0 or date_index == 0:
                elapsed = time.monotonic() - started
                print(f"references dates={date_index+1}/{len(dates)} rows={len(rows)} exclusions={len(exclusions)} "
                    f"elapsed_s={elapsed:.1f} remaining_s~{elapsed*(len(dates)-date_index-1)/(date_index+1):.0f}", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(output_directory / "results.parquet", index=False)
        pd.DataFrame(exclusions, columns=("date", "claim", "side", "method", "exact_error")).to_parquet(
            output_directory / "reference_exclusions.parquet", index=False)
        summary = {"status": "complete", "dates": len(dates), "rows": len(frame),
            "rows_by_method": frame.groupby("method").size().to_dict(), "reference_exclusions": len(exclusions),
            "uncapped_analytic_references": True, "entry_only_costs": True, "elapsed_seconds": time.monotonic() - started}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-configuration", "coverage-path", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(run_references(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
