"""Separate factor payout scaling from local exposure matching, without changing caps."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedge2.cash import MarketStore
from quantdata.hedge2.factor_portfolios import FactorPortfolioConfig, fit_factor_portfolios
from quantdata.hedge2.greeks import calculate_greeks
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState


@dataclass(frozen=True, kw_only=True)
class UnconstrainedTracking:
    premium_positions: np.ndarray
    training_r_squared: np.ndarray
    training_rmse: np.ndarray
    rank: int
    singular_values: np.ndarray
    relative_cutoff: float


def unconstrained_tracking(*, instrument_returns: np.ndarray, factor_shocks: np.ndarray,
                           weights: np.ndarray, relative_cutoff: float | None = None) -> UnconstrainedTracking:
    """Unregularized minimum-Euclidean-norm dollar solution, NOT minimum gross exposure.

    An underdetermined history has no unique unconstrained portfolio.
    The Moore--Penrose convention chooses the smallest L2 norm at the
    reported singular-value cutoff. Near-null return directions can
    produce enormous positions and a perfect, meaningless training fit.
    """
    returns, targets = np.asarray(instrument_returns, dtype=float), np.asarray(factor_shocks, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if (returns.ndim != 2 or targets.ndim != 2 or len(returns) != len(targets)
            or min(*returns.shape, targets.shape[1]) < 1 or probabilities.shape != (len(returns),)
            or not all(np.isfinite(values).all() for values in (returns, targets, probabilities))
            or np.any(probabilities < 0) or probabilities.sum() <= 0
            or relative_cutoff is not None and (not np.isfinite(relative_cutoff) or not 0 <= relative_cutoff < 1)):
        raise ValueError("Unconstrained tracking needs finite aligned returns, targets, weights and a valid SVD cutoff")
    probabilities = probabilities / probabilities.sum()
    if np.max(np.abs(probabilities @ returns), initial=0.) > 1e-10:
        raise ValueError("Remove instrument alpha before unconstrained factor tracking")
    targets = targets - probabilities @ targets
    matrix = np.sqrt(probabilities)[:, None] * returns
    target = np.sqrt(probabilities)[:, None] * targets
    cutoff = np.finfo(matrix.dtype).eps * max(matrix.shape) if relative_cutoff is None else relative_cutoff
    positions, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=cutoff)
    mse = probabilities @ (returns @ positions - targets)**2
    variance = probabilities @ targets**2
    r_squared = 1. - np.divide(mse, variance, out=np.zeros_like(mse), where=variance > 0.)
    return UnconstrainedTracking(premium_positions=positions, training_r_squared=r_squared,
        training_rmse=np.sqrt(mse), rank=int(rank), singular_values=singular, relative_cutoff=float(cutoff))


def tracking_payout_curve(*, instrument_returns: np.ndarray, factor_shocks: np.ndarray,
                         weights: np.ndarray, proportional_costs: np.ndarray,
                         payouts: tuple[float, ...],
                         config: FactorPortfolioConfig = FactorPortfolioConfig(),
                         progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Change declared factor payout, not budgets; report only in-sample tracking.

    A smaller payout is a different economic target. Its improved R-squared
    does not repair the original $1-payout portfolio or establish OOS value.
    """
    if (not payouts or len(set(payouts)) != len(payouts)
            or any(not np.isfinite(value) or value <= 0. for value in payouts)):
        raise ValueError("Diagnostic payouts must be distinct finite positive amounts")
    rows, arrays = [], {}
    started = time.monotonic()
    for index, payout in enumerate(payouts):
        fit = fit_factor_portfolios(instrument_returns=instrument_returns / payout, factor_shocks=factor_shocks,
            weights=weights, proportional_costs=proportional_costs, config=config)
        arrays[f"premium_positions_{index}"] = fit.premium_positions
        for factor in range(factor_shocks.shape[1]):
            rows.append({"payout": payout, "factor": factor + 1,
                "training_r_squared": float(fit.training_r_squared[factor]),
                "training_rmse_per_payout": float(fit.training_rmse[factor]),
                "target_portfolio_correlation": float(fit.target_portfolio_correlation[factor]),
                "gross_positions": float(fit.gross_positions[factor]),
                "maximum_position": float(np.max(np.abs(fit.premium_positions[:, factor]))),
                "entry_cost": float(fit.entry_cost[factor]),
                "entry_cost_per_payout": float(fit.entry_cost[factor] / payout),
                "gross_cap_binding": bool(fit.gross_cap_binding[factor]),
                "instrument_cap_hits": int(fit.instrument_cap_hits[factor]),
                "objective_gap": float(fit.objective_gap[factor] * payout**2),
                "objective_gap_per_payout_squared": float(fit.objective_gap[factor]), "certified": fit.certified})
        if progress is not None:
            progress(f"payout diagnostic {index+1}/{len(payouts)} payout={payout:g} "
                f"R2={fit.training_r_squared.min():.4f}..{fit.training_r_squared.max():.4f} "
                f"certified={fit.certified} elapsed_s={time.monotonic()-started:.1f}")
    return pd.DataFrame(rows), arrays


def run_diagnostics(*, input_directory: Path, output_directory: Path,
                    payouts: tuple[float, ...] = (1., .25, .1, .01)) -> dict[str, object]:
    """Replay one dated factor experiment and compare its local price-exposure problem."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        if (output_directory / "config.json").exists():
            raise ValueError("Use a new diagnostic run directory; preserve previous fits")
        configuration = json.loads((input_directory.parent / "config.json").read_text())
        manifest = json.loads((input_directory / "completed.json").read_text())
        for name, digest in manifest["artifacts_sha256"].items():
            if hashlib.sha256((input_directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source factor experiment has changed: {name}")
        config = FactorPortfolioConfig(**configuration["portfolio_config"])
        metadata = {"input_directory": str(input_directory), "payouts": payouts,
            "portfolio_config": asdict(config), "input_completed": manifest,
            "statistics": "in_sample_only; smaller payouts do not fix the original economic target",
            "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("factor_diagnostics.py", "factor_portfolios.py", "greeks.py", "cash.py", "state.py")}}
        (output_directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with np.load(input_directory / "historical_returns.npz", allow_pickle=False) as saved:
            returns, weights = saved["instrument_gains"], saved["weights"]
        with np.load(input_directory / "inputs.npz", allow_pickle=False) as saved:
            shocks, costs, normalized_prices = saved["factor_shocks"], saved["costs"], saved["current_normalized_prices"]
        results, arrays = tracking_payout_curve(instrument_returns=returns, factor_shocks=shocks,
            weights=weights, proportional_costs=costs, payouts=payouts, config=config,
            progress=lambda message: print(message, flush=True))
        results.to_parquet(output_directory / "payout_tracking.parquet", index=False)
        np.savez_compressed(output_directory / "payout_positions.npz", **arrays)
        day = date.fromisoformat(input_directory.name)
        store = MarketStore(market_directory=Path(configuration["market_directory"]),
            licensed_fit_directory=Path(configuration["licensed_fit_directory"]),
            raw_quote_directory=Path(configuration["raw_quote_directory"]))
        market = store.load(day)
        state = DailyState.load(Path(configuration["daily_directory"]), day)
        dynamics = load_dynamics(Path(configuration["dynamics_directory"]) / "days" / f"{day.isoformat()}.npz")
        book = pd.read_parquet(input_directory / "book.parquet")
        greeks = calculate_greeks(state=state, dynamics=dynamics, curve=market.curve,
            book=book, claims=(), initial_spot=market.curve.spot)
        axis_scales = np.r_[np.sqrt(dynamics.joint_innovation_covariance[0, 0]), np.ones(len(shocks[0]))]
        exposure_map = axis_scales[:, None] * greeks.instrument_exposures.T / normalized_prices[None, :]
        identity = np.eye(len(axis_scales))[:, 1:]
        local_unconstrained = np.linalg.lstsq(exposure_map, identity, rcond=None)[0]
        local_results, local_arrays = tracking_payout_curve(instrument_returns=np.r_[exposure_map, -exposure_map],
            factor_shocks=np.r_[identity, -identity], weights=np.full(2 * len(axis_scales), .5 / len(axis_scales)),
            proportional_costs=costs, payouts=payouts, config=config,
            progress=lambda message: print("local Greek " + message, flush=True))
        local_results.to_parquet(output_directory / "local_exposure_tracking.parquet", index=False)
        np.savez_compressed(output_directory / "local_exposures.npz", exposure_map=exposure_map,
            target_exposures=identity, axis_scales=axis_scales, unconstrained_positions=local_unconstrained,
            pc_lower_clips=greeks.pc_lower_clips,
            pc_upper_clips=greeks.pc_upper_clips, **local_arrays)
        summary = {"historical_return_rank": int(np.linalg.matrix_rank(returns)),
            "historical_rows": len(returns), "instruments": returns.shape[1],
            "local_exposure_rank": int(np.linalg.matrix_rank(exposure_map)),
            "local_exposure_singular_values": np.linalg.svd(exposure_map, compute_uv=False).tolist(),
            "local_unconstrained_relative_error": np.linalg.norm(exposure_map @ local_unconstrained - identity, axis=0).tolist(),
            "local_unconstrained_gross_positions": np.abs(local_unconstrained).sum(axis=0).tolist(),
            "local_unconstrained_maximum_positions": np.abs(local_unconstrained).max(axis=0).tolist(),
            "all_certified": bool(results.certified.all() and local_results.certified.all()),
            "interpretation": "historical innovation tracking and current finite-Greek matching are different tasks; neither is OOS evidence"}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--payouts", nargs="+", type=float, default=[1., .25, .1, .01])
    arguments = vars(parser.parse_args())
    arguments["payouts"] = tuple(arguments["payouts"])
    print(json.dumps(run_diagnostics(**arguments), indent=2), flush=True)


if __name__ == "__main__":
    main()
