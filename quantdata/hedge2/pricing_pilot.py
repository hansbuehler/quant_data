"""Persist a small dated audit of cash normalization and PCA price exposures."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import MarketStore, select_book
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.greeks import calculate_greeks
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive, surface_digitals
from quantdata.hedge2.state import DailyState


def run_pilot(*, source_root: Path, licensed_fit_directory: Path, market_directory: Path,
              daily_directory: Path, dynamics_directory: Path, output_directory: Path,
              dates: tuple[date, ...]) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    market_store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory)
    rows, claims_rows, pc_rows = [], [], []
    started = time.monotonic()
    for day in dates:
        market = market_store.load(day)
        source, _ = SourceArchive(root=source_root).load(day)
        native = DailyState.load(daily_directory, day)
        model = load_dynamics(dynamics_directory / "days" / f"{day.isoformat()}.npz")
        vols = source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR)
        book = select_book(market, atm_vols=np.asarray(vols))
        tenor_vols = {tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)}
        claims = make_claims(initial_spot=market.curve.spot, atm_vols_by_day=tenor_vols)
        result = calculate_greeks(state=native, dynamics=model, curve=market.curve, book=book,
                                   claims=claims, initial_spot=market.curve.spot)
        normalized_strikes = book["strike"].to_numpy(dtype=float) / book["fwd"].to_numpy(dtype=float)
        source_calls = source.price(expiries=book["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR,
                                     strikes=normalized_strikes, mode="surface")
        source_values = np.where(book["option_type"] == "call", source_calls, source_calls - 1. + normalized_strikes)
        source_cash = source_values * book["df"].to_numpy(dtype=float) * book["fwd"].to_numpy(dtype=float)
        price_error = np.abs(source_cash - book["fit_price"].to_numpy())
        for index, claim in enumerate(claims):
            reference_price = None
            if claim.kind == "digital":
                forward, _, discount = market.curve.at(float(claim.terminal_day))
                reference_price = float(discount * surface_digitals(source,
                    expiry_y=claim.terminal_day / DAYS_PER_YEAR,
                    strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0])
            claims_rows.append({
                "date": day.isoformat(), "claim": claim.label, "kind": claim.kind,
                "dlv_price": float(result.claim_prices[index]), "source_digital_price": reference_price,
                "spot_finite_exposure": float(result.claim_exposures[index, 0]),
                "pca_finite_exposure_norm": float(np.linalg.norm(result.claim_exposures[index, 1:])),
            })
        for factor in range(model.directions.shape[1]):
            pc_rows.append({
                "date": day.isoformat(), "factor": factor + 1,
                "option_max_absolute_price_exposure": float(np.max(np.abs(result.instrument_exposures[1:, factor + 1]))),
                "claim_max_absolute_price_exposure": float(np.max(np.abs(result.claim_exposures[:, factor + 1]))),
                "lower_clips": int(result.pc_lower_clips[factor]), "upper_clips": int(result.pc_upper_clips[factor]),
            })
        np.savez_compressed(output_directory / f"{day.isoformat()}.npz",
                            instrument_prices=result.instrument_prices, claim_prices=result.claim_prices,
                            instrument_exposures=result.instrument_exposures, claim_exposures=result.claim_exposures,
                            claim_labels=np.array([claim.label for claim in claims]),
                            pc_lower_clips=result.pc_lower_clips, pc_upper_clips=result.pc_upper_clips)
        book.to_parquet(output_directory / f"{day.isoformat()}-book.parquet", index=False)
        rows.append({"date": day.isoformat(), "options": len(book), "claims": len(claims),
                     "max_source_cash_repricing_error": float(price_error.max()),
                     "max_source_normalized_repricing_error": float(price_error.max() / market.curve.spot),
                     "max_source_error_in_halfspreads": float(np.max(price_error / book["observed_half_spread"].to_numpy())),
                     "pca_lower_clips": int(result.pc_lower_clips.sum()),
                     "pca_upper_clips": int(result.pc_upper_clips.sum())})
        print(f"pricing pilot {len(rows)}/{len(dates)} date={day} options={len(book)} "
              f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    pd.DataFrame(rows).to_parquet(output_directory / "dates.parquet", index=False)
    pd.DataFrame(claims_rows).to_parquet(output_directory / "claims.parquet", index=False)
    pd.DataFrame(pc_rows).to_parquet(output_directory / "factors.parquet", index=False)
    summary = {"dates": rows, "elapsed_seconds": time.monotonic() - started,
               "daily_directory": str(daily_directory), "dynamics_directory": str(dynamics_directory),
               "digital_convention": "right strike derivative of piecewise-linear DLV calls; finite-shock exposures, not smooth Greeks",
               "source_normalization": "cash strike/fwd and df*fwd, never mixed with fwd_ref",
               "does_not_contain_hedge_results": True}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "market-directory", "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
