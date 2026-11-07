"""One-session realized hedge checks before the full rolling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.black_hedge import black_digital_hedge
from quantdata.hedge2.cash import MarketStore, discounted_daily_gains, observed_marks, select_book
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.greek_hedge import solve_greek_hedge
from quantdata.hedge2.greeks import calculate_greeks, factor_risk_covariance
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive, surface_digitals, surface_implied_vols
from quantdata.hedge2.state import DailyState


def run_pilot(*, source_root: Path, licensed_fit_directory: Path, market_directory: Path,
              daily_directory: Path, dynamics_directory: Path, output_directory: Path,
              raw_quote_directory: Path, dates: tuple[date, ...],
              diagnostic_only: bool = True) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parent
    implementation = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                      for name in ("greek_hedge.py", "greeks.py", "black_hedge.py", "claims.py", "cash.py",
                                   "state.py", "pricing.py", "source.py", "hedge_pilot.py")}
    identity = {"source_root": str(source_root), "licensed_fit_directory": str(licensed_fit_directory),
                "raw_quote_directory": str(raw_quote_directory), "market_directory": str(market_directory),
                "daily_directory": str(daily_directory), "dynamics_directory": str(dynamics_directory),
                "dates": [day.isoformat() for day in dates], "implementation_sha256": implementation,
                "diagnostic_only": diagnostic_only}
    identity_path = output_directory / "config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Hedge output belongs to a different implementation, dataset or date set")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    configuration = json.loads((dynamics_directory / "config.json").read_text())
    started = time.monotonic()
    results = []
    for day in dates:
        path = output_directory / f"{day.isoformat()}.parquet"
        positions_path = output_directory / f"{day.isoformat()}-positions.npz"
        book_path = output_directory / f"{day.isoformat()}-book.parquet"
        completion_path = output_directory / f"{day.isoformat()}.complete.json"
        if completion_path.exists():
            completion = json.loads(completion_path.read_text())
            for artifact in (path, positions_path, book_path):
                if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != completion[artifact.name]:
                    raise ValueError(f"Completed hedge artifact is missing or changed: {artifact}")
            results.extend(pd.read_parquet(path).to_dict("records"))
            continue
        market = store.load(day)
        next_day = get_business_days(day, day + timedelta(days=10))[1]
        following = store.load(next_day)
        source, _ = SourceArchive(root=source_root).load(day)
        native = DailyState.load(daily_directory, day)
        model = load_dynamics(dynamics_directory / "days" / f"{day.isoformat()}.npz")
        vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
        book = select_book(market, atm_vols=vols)
        book_vols = surface_implied_vols(source, expiries_y=book["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR,
            strikes=book["strike"].to_numpy(dtype=float) / book["fwd"].to_numpy(dtype=float))
        claims = tuple(claim for claim in make_claims(initial_spot=market.curve.spot,
                       atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR))
                                       for tenor in (1, 2, 3, 5)})
                       if claim.kind == "digital" and claim.terminal_day == 1)
        greeks = calculate_greeks(state=native, dynamics=model, curve=market.curve,
                                   book=book, claims=claims, initial_spot=market.curve.spot)
        next_marks, next_spreads = observed_marks(book, following)
        gains = discounted_daily_gains(previous_option_marks=book["observed_mid"].to_numpy(),
                                       next_option_marks=next_marks, next_spot=following.curve.spot,
                                       previous_curve=market.curve, episode_initial_spot=market.curve.spot)
        costs = np.r_[0., book["observed_half_spread"].to_numpy() / market.curve.spot]
        risks = np.sqrt(np.maximum(np.diag(factor_risk_covariance(model,
                                    spot_orthogonal=configuration["spot_orthogonal"])), 1e-20))
        _, _, discount = market.curve.at(1.)
        forward, _, _ = market.curve.at(1.)
        rows = []
        portfolios = []
        for index, claim in enumerate(claims):
            market_price = float(discount * surface_digitals(source, expiry_y=1. / DAYS_PER_YEAR,
                                 strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0])
            target_gain = float(discount * claim.payoff(following.curve.spot) - market_price)
            target_vol = surface_implied_vols(source, expiries_y=np.array([1. / DAYS_PER_YEAR]),
                strikes=np.array([claim.strike / forward]))
            for side in (-1, 1):
                candidates = [("unhedged", np.zeros(len(costs)), 0., 0., 0., "not applicable", 0, "none", ())]
                black = black_digital_hedge(claim=claim, curve=market.curve, book=book,
                                             target_vol=float(target_vol[0]), option_vols=np.asarray(book_vols),
                                             target_side=side)
                candidates.append(("BS delta+vega", black, float(costs @ np.abs(black)), 0., 0., "analytic", 0, "analytic", ()))
                for fraction in (1., .8, .6, .4, .2, 0.):
                    hedge = solve_greek_hedge(exposures=greeks.instrument_exposures,
                        target_exposures=greeks.claim_exposures[index], costs=costs,
                        risk_scales=risks, target_side=side, budget_fraction=fraction)
                    candidates.append((f"DLV Greek cost {fraction:.1f}", hedge.positions, hedge.entry_cost,
                                       hedge.full_entry_cost, hedge.risk_residual_norm, hedge.status, hedge.cap_hits,
                                       hedge.solver, hedge.warnings))
                for method, positions, entry, full_entry, residual, status, caps, solver, solver_warnings in candidates:
                    hedge_gain = float(positions @ gains)
                    rows.append({"date": day.isoformat(), "next_date": next_day.isoformat(),
                                 "claim": claim.label, "side": side, "method": method,
                                 "market_target_price": market_price, "dlv_target_price": float(greeks.claim_prices[index]),
                                 "target_gain": side * target_gain, "hedge_gain": hedge_gain,
                                 "entry_cost": entry, "full_greek_entry_cost": full_entry,
                                 "costed_pnl": side * target_gain + hedge_gain - entry,
                                 "hypothetical_exit_cost": float(discount * np.abs(positions[1:]) @ next_spreads / market.curve.spot),
                                 "gross_positions": float(np.abs(positions).sum()), "cap_hits": caps,
                                 "risk_residual_norm": residual, "status": status,
                                 "solver": solver, "warnings": json.dumps(solver_warnings)})
                    portfolios.append(positions)
        temporary = path.with_suffix(".partial.parquet")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(path)
        temporary_positions = positions_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary_positions, positions=np.asarray(portfolios))
        temporary_positions.replace(positions_path)
        temporary_book = book_path.with_suffix(".partial.parquet")
        book.to_parquet(temporary_book, index=False)
        temporary_book.replace(book_path)
        completion = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()
                      for artifact in (path, positions_path, book_path)}
        temporary_completion = completion_path.with_suffix(".partial.json")
        temporary_completion.write_text(json.dumps(completion, indent=2) + "\n")
        temporary_completion.replace(completion_path)
        results.extend(rows)
        print(f"hedge pilot dates={len({row['date'] for row in results})}/{len(dates)} "
              f"date={day} rows={len(rows)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    frame = pd.DataFrame(results)
    frame.to_parquet(output_directory / "results.parquet", index=False)
    summary = {"dates": len(dates), "rows": len(results), "elapsed_seconds": time.monotonic() - started,
               "entry_cost_only": True, "spot_cost": 0., "option_spread_multiplier": 1.,
               "results_are_diagnostic_not_oos_performance_evidence": diagnostic_only,
               "source_root": str(source_root), "dynamics_directory": str(dynamics_directory)}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory", "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
