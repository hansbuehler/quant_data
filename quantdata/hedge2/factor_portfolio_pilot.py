"""Dated eight-factor replication on the saved SANOS ITM/premium-return book."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import MarketStore, select_book, zero_strike_mask
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.factor_portfolios import (
    FactorPortfolioConfig,
    fit_factor_portfolios,
    premium_to_cash_holdings,
)
from quantdata.hedge2.historical import build_historical_returns, historical_pairs
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive


def run_factor_pilot(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
                     market_directory: Path, daily_directory: Path, dynamics_directory: Path,
                     output_directory: Path, dates: tuple[date, ...],
                     config: FactorPortfolioConfig = FactorPortfolioConfig()) -> dict[str, object]:
    """Fit all eight shock portfolios together on each exact causal AR training window."""
    if not dates or tuple(sorted(set(dates))) != dates:
        raise ValueError("Specify distinct chronological factor-portfolio diagnostic dates")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        package = Path(__file__).parent
        names = ("factor_portfolio_pilot.py", "factor_portfolios.py", "historical.py", "cash.py", "source.py",
                 "claims.py", "rolling.py", "dynamics.py")
        configuration = {"dates": [day.isoformat() for day in dates], "portfolio_config": asdict(config),
            "source_root": str(source_root), "licensed_fit_directory": str(licensed_fit_directory),
            "raw_quote_directory": str(raw_quote_directory), "market_directory": str(market_directory),
            "daily_directory": str(daily_directory), "dynamics_directory": str(dynamics_directory),
            "option_side": "itm", "return_basis": "historical_entry_premium",
            "zero_strike_calls": "one_per_expiry_at_DF; zero_spread", "spot_cost": 0.,
            "target_units": "AR factor innovations divided by daily PCA scales; not state levels or unit-variance shocks",
            "objective": "joint weighted squared tracking error; no orthogonality, hidden ridge or cost penalty",
            "diagnostic_only": True, "statistics": "in_sample_not_OOS_hedging_performance",
            "implementation_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}}
        path = output_directory / "config.json"
        if path.exists() and json.loads(path.read_text()) != configuration:
            raise ValueError("The portfolio output belongs to different sources, target units or limits")
        path.write_text(json.dumps(configuration, indent=2) + "\n")
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as saved:
                for name in names:
                    saved.write(package / name, arcname=name)
        store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                            raw_quote_directory=raw_quote_directory)
        archive = SourceArchive(root=source_root)
        state_dates = tuple(sorted(date.fromisoformat(path.stem) for path in (daily_directory / "days").glob("*.npz")))
        rows = []
        started = time.monotonic()
        for day_index, day in enumerate(dates):
            directory = output_directory / day.isoformat()
            directory.mkdir(parents=True, exist_ok=True)
            completion = directory / "completed.json"
            if completion.exists():
                manifest = json.loads(completion.read_text())
                for name, digest in manifest["artifacts_sha256"].items():
                    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Completed factor artifact changed: {directory / name}")
                rows.extend(pd.read_parquet(directory / "results.parquet").to_dict("records"))
                continue
            market = store.load(day)
            source, source_metadata = archive.load(day)
            model_path = dynamics_directory / "days" / f"{day.isoformat()}.npz"
            model = load_dynamics(model_path)
            if model.as_of_date != day or model.residuals.shape[1] != 9 or model.training_start < date(2022, 1, 1):
                raise ValueError("The diagnostic requires today's causal eight-factor AR fit")
            pairs = historical_pairs(dates=state_dates, as_of_date=day, window_transitions=len(model.residuals))
            if model.training_start != pairs[0][0] or model.training_end != pairs[-1][1]:
                raise ValueError("The reconstructed AR transitions do not match the saved fit's window")
            vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
            book = select_book(market, atm_vols=vols, option_side="itm", include_co_terminal_forwards=True)
            claims = tuple(claim for claim in make_claims(initial_spot=market.curve.spot,
                atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
                if claim.kind == "digital" and claim.terminal_day == 1)
            returns = build_historical_returns(archive=archive, licensed_fit_directory=licensed_fit_directory,
                daily_directory=daily_directory,
                as_of_date=day, current_spot=market.curve.spot, book=book, claims=claims,
                window_transitions=len(pairs), training_pairs=pairs, return_basis="premium",
                progress=lambda message: print(message, flush=True))
            if not (np.array_equal(returns.pair_start_dates, [before.isoformat() for before, _ in pairs])
                    and np.array_equal(returns.pair_end_dates, [after.isoformat() for _, after in pairs])):
                raise ValueError("Historical returns and AR innovations have different row identities")
            targets = model.residuals[:, 1:] / model.daily_factor_scales
            cash_prices = np.r_[market.curve.spot, book["observed_mid"].to_numpy(dtype=float)]
            normalized_prices = cash_prices / market.curve.spot
            if not np.isfinite(normalized_prices).all() or np.any(normalized_prices <= 0):
                raise ValueError("ITM/forward entry prices must be positive without a numerical price floor")
            costs = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float) / market.curve.spot] / normalized_prices
            book.to_parquet(directory / "book.parquet", index=False)
            np.savez_compressed(directory / "historical_returns.npz", **asdict(returns))
            np.savez_compressed(directory / "inputs.npz", factor_shocks=targets, costs=costs,
                current_normalized_prices=normalized_prices, current_cash_prices=cash_prices,
                daily_factor_scales=model.daily_factor_scales,
                directions=model.directions, residuals=model.residuals,
                explained_variance_fraction=model.explained_variance_fraction)
            print(f"joint factor fit date={day} factors=8 instruments={len(costs)} "
                  f"pairs={len(pairs)} per_instrument_cap={config.instrument_cap} gross_cap={config.gross_cap}", flush=True)
            fit = fit_factor_portfolios(instrument_returns=returns.instrument_gains, factor_shocks=targets,
                weights=returns.weights, proportional_costs=costs, config=config)
            np.savez_compressed(directory / "portfolios.npz", **asdict(fit),
                spot_normalized_positions=fit.premium_positions / normalized_prices[:, None],
                cash_unit_holdings=premium_to_cash_holdings(fit.premium_positions, cash_prices))
            metadata = {"date": day.isoformat(), "source_provenance": source_metadata,
                "dynamics_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "training_start": pairs[0][0].isoformat(), "training_end": pairs[-1][1].isoformat(),
                "training_pairs": len(pairs), "spot_orthogonal": bool(np.any(model.spot_state_loading)),
                "instruments": len(costs), "ordinary_options": int((~zero_strike_mask(book)).sum()),
                "co_terminal_zero_strike_calls": int(zero_strike_mask(book).sum()),
                "option_cost": "observed_half_spread/current_observed_ITM_price",
                "guarded_tail_warning": "DLV coordinate guards can drive PCA shocks; tracking fit is not OOS evidence",
                "certified": fit.certified, "solver": fit.solver, "status": fit.status,
                "warnings": list(fit.warnings), "instrument_return_rank": fit.instrument_return_rank}
            (directory / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")
            dated_rows = []
            for factor in range(8):
                dated_rows.append({"date": day.isoformat(), "factor": factor + 1,
                    "training_r_squared": float(fit.training_r_squared[factor]),
                    "training_rmse": float(fit.training_rmse[factor]),
                    "target_std": float(fit.target_std[factor]), "portfolio_std": float(fit.portfolio_std[factor]),
                    "target_portfolio_correlation": float(fit.target_portfolio_correlation[factor]),
                    "gross_positions": float(fit.gross_positions[factor]),
                    "maximum_position": float(np.max(np.abs(fit.premium_positions[:, factor]))),
                    "entry_cost": float(fit.entry_cost[factor]),
                    "instrument_cap_hits": int(fit.instrument_cap_hits[factor]),
                    "gross_cap_binding": bool(fit.gross_cap_binding[factor]),
                    "objective_gap": float(fit.objective_gap[factor]), "certified": fit.certified})
            pd.DataFrame(dated_rows).to_parquet(directory / "results.parquet", index=False)
            names = ("book.parquet", "historical_returns.npz", "inputs.npz", "portfolios.npz", "diagnostics.json", "results.parquet")
            marker = {"certified": fit.certified, "artifacts_sha256": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
            temporary = completion.with_suffix(".partial.json")
            temporary.write_text(json.dumps(marker, indent=2) + "\n")
            temporary.replace(completion)
            rows.extend(dated_rows)
            print(f"joint factors completed dates={day_index+1}/{len(dates)} certified={fit.certified} "
                  f"maximum_gap={np.max(fit.objective_gap):.6g} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        pd.DataFrame(rows).to_parquet(output_directory / "results.parquet", index=False)
        summary = {"dates": len(dates), "factor_portfolios": len(rows),
            "all_certified": all(row["certified"] for row in rows), "elapsed_seconds": time.monotonic()-started,
            "diagnostic_only": True, "in_sample_only": True}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory",
                 "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--instrument-cap", type=float, default=10.)
    parser.add_argument("--gross-cap", type=float, default=100.)
    arguments = vars(parser.parse_args())
    config = FactorPortfolioConfig(instrument_cap=arguments.pop("instrument_cap"), gross_cap=arguments.pop("gross_cap"))
    arguments["dates"] = tuple(arguments["dates"])
    print(json.dumps(run_factor_pilot(**arguments, config=config), indent=2), flush=True)


if __name__ == "__main__":
    main()
