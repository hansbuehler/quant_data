"""Execute historical-state mean-variance portfolios across risk aversions."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR

from .historical_state_period_audit import (
    conditional_period_returns,
    decode_historical_fit,
    execute_frozen_action,
    observed_surface_calls,
    solve_mean_variance_portfolio,
    summarize_fit_errors,
)
from .torch_prep import GenDysanosExperiment, _load_generator_components

_DEFAULT_OUTPUT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/historical-state-period-audit/"
    "dysanos-historical-states-1d-raw-0bps-mean-variance-"
    "gamma-1-10-100.json"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_summary(
    action: np.ndarray,
    scenario_return: np.ndarray,
    *,
    risk_aversion: float,
) -> dict[str, float | int]:
    pnl = np.asarray(scenario_return, dtype=np.float64) @ np.asarray(
        action, dtype=np.float64
    )
    mean = float(np.mean(pnl))
    variance = float(np.var(pnl, ddof=0))
    return {
        "paths": int(pnl.size),
        "mean_pnl": mean,
        "median_pnl": float(np.median(pnl)),
        "pnl_standard_deviation": math.sqrt(variance),
        "mean_pnl_standard_error": math.sqrt(variance / pnl.size),
        "loss_count": int(np.count_nonzero(pnl < 0.0)),
        "loss_frequency": float(np.mean(pnl < 0.0)),
        "mean_variance_objective": mean - 0.5 * risk_aversion * variance,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "standard_error": float(
            np.std(values, ddof=0) / math.sqrt(values.size)
        ),
        "minimum": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _aggregate(
    rows: list[dict[str, Any]], risk_aversions: tuple[float, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for risk_aversion in risk_aversions:
        key = f"{risk_aversion:g}"
        realized = np.asarray(
            [row["risk_aversions"][key]["realized"]["net_pnl"] for row in rows]
        )
        validation = np.asarray(
            [
                row["risk_aversions"][key]["model_validation"]["mean_pnl"]
                for row in rows
            ]
        )
        l1_norm = np.asarray(
            [row["risk_aversions"][key]["portfolio"]["l1_norm"] for row in rows]
        )
        converged = np.asarray(
            [
                row["risk_aversions"][key]["portfolio"]["optimizer"][
                    "converged"
                ]
                for row in rows
            ],
            dtype=bool,
        )
        result[key] = {
            "risk_aversion": risk_aversion,
            "realized_trade_pnl": _distribution(realized),
            "realized_loss_count": int(np.count_nonzero(realized < 0.0)),
            "realized_loss_frequency": float(np.mean(realized < 0.0)),
            "model_validation_mean_pnl": _distribution(validation),
            "model_validation_positive_mean_count": int(
                np.count_nonzero(validation > 0.0)
            ),
            "model_validation_to_realized_correlation": (
                float(np.corrcoef(validation, realized)[0, 1])
                if np.std(validation) > 0.0 and np.std(realized) > 0.0
                else None
            ),
            "portfolio_l1_norm": _distribution(l1_norm),
            "optimizer_converged_count": int(np.count_nonzero(converged)),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-aversions", type=float, nargs="+", default=(1.0, 10.0, 100.0)
    )
    parser.add_argument("--training-paths", type=int, default=1_024)
    parser.add_argument("--validation-paths", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=731_991)
    parser.add_argument("--decoder-block-size", type=int, default=8_192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    risk_aversions = tuple(float(value) for value in args.risk_aversions)
    if not risk_aversions or not all(
        math.isfinite(value) and value > 0.0 for value in risk_aversions
    ):
        raise ValueError("risk aversions must be finite and positive")
    started = time.perf_counter()
    experiment = GenDysanosExperiment(
        num_samples=2,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        trading_frequency="daily",
        return_horizon="incremental",
        delta_hedged=False,
        cost_rate=0.0,
        num_trading_times=2,
        seed=args.seed,
        importance_sampling=None,
    )
    _, market, model, sampler = _load_generator_components(
        experiment, verbose=Context.quiet
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {args.device!r} is unavailable")
    model.env.device = device
    model.to(device)
    model.decoder.xnms = model.decoder.xnms.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("decoding DYSANOS fits for all historical states", flush=True)
    fitted_calls = decode_historical_fit(model, sampler, market)
    fit_rows, fit_aggregate = summarize_fit_errors(
        fitted_calls=fitted_calls,
        market_calls=np.asarray(market["mids"]),
        relative_strikes=np.asarray(market["xstrikes"])[:, :, 1:-1],
        expiries=np.asarray(market["expiries"]),
        market_vols=np.asarray(market["mid_vols"]),
        weights=np.asarray(market["weights"]),
    )
    del fitted_calls

    maximum_index = len(market["dates"]) - 1
    start = max(0, args.start_index)
    stop = maximum_index
    if args.max_dates is not None:
        stop = min(stop, start + args.max_dates)
    if start >= stop:
        raise ValueError(f"empty historical interval [{start}, {stop})")
    rows: list[dict[str, Any]] = []

    def report(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "objective": (
                "maximize a^T mean(R|h) - gamma/2 * "
                "a^T covariance(R|h) a subject to ||a||_1 <= 1"
            ),
            "risk_aversions": list(risk_aversions),
            "market_case": {
                "return_horizon": "one trading day",
                "delta_hedged": False,
                "transaction_cost_rate": 0.0,
                "spot_reset_to_one": True,
                "num_alpha": None,
                "use_matrix_exp": True,
                "surface_first": True,
            },
            "sampling": {
                "training_paths_per_state": args.training_paths,
                "model_validation_paths_per_state": args.validation_paths,
                "historical_states": stop - start,
                "next_observed_market_is_used_only_for_realized_execution": True,
            },
            "historical_interval": {
                "first_date": str(np.asarray(market["dates"])[start]),
                "last_entry_date": str(np.asarray(market["dates"])[stop - 1]),
                "last_exit_date": str(np.asarray(market["dates"])[stop]),
            },
            "surface_fit_error_all_1412_states": fit_aggregate,
            "states": rows,
            "aggregate": _aggregate(rows, risk_aversions) if rows else None,
            "elapsed_seconds": time.perf_counter() - started,
        }

    for ordinal, index in enumerate(range(start, stop), start=1):
        expiries = np.asarray(market["expiries"][index], dtype=np.float64)
        training_return, relative_strikes, initial_model_calls = (
            conditional_period_returns(
                model,
                sampler,
                surface_state=sampler.xstate[index, 1:],
                expiries=expiries,
                paths=args.training_paths,
                seed=args.seed + 2_000_003 * index,
                decoder_block_size=args.decoder_block_size,
            )
        )
        validation_return, validation_strikes, validation_initial_calls = (
            conditional_period_returns(
                model,
                sampler,
                surface_state=sampler.xstate[index, 1:],
                expiries=expiries,
                paths=args.validation_paths,
                seed=args.seed + 2_000_003 * index + 1,
                decoder_block_size=args.decoder_block_size,
            )
        )
        if not np.allclose(validation_strikes, relative_strikes) or not np.allclose(
            validation_initial_calls, initial_model_calls
        ):
            raise RuntimeError("conditional train/validation contract grids differ")
        entry_calls = observed_surface_calls(
            grid_expiries=np.asarray(market["expiries"][index]),
            extended_strikes=np.asarray(market["xstrikes"][index]),
            inner_calls=np.asarray(market["mids"][index]),
            query_expiries=expiries,
            query_strikes=relative_strikes,
        )
        spot_ratio = float(
            market["spot_ref"][index + 1] / market["spot_ref"][index]
        )
        exit_calls = observed_surface_calls(
            grid_expiries=np.asarray(market["expiries"][index + 1]),
            extended_strikes=np.asarray(market["xstrikes"][index + 1]),
            inner_calls=np.asarray(market["mids"][index + 1]),
            query_expiries=np.maximum(expiries - 1.0 / DAYS_PER_YEAR, 0.0),
            query_strikes=relative_strikes / spot_ratio,
        )
        row: dict[str, Any] = {
            "state_index": index,
            "date": str(np.asarray(market["dates"])[index]),
            "next_market_date": str(np.asarray(market["dates"])[index + 1]),
            "surface_fit_error": fit_rows[index],
            "risk_aversions": {},
        }
        for risk_aversion in risk_aversions:
            key = f"{risk_aversion:g}"
            action, optimizer = solve_mean_variance_portfolio(
                training_return,
                risk_aversion=risk_aversion,
            )
            row["risk_aversions"][key] = {
                "portfolio": {
                    "l1_norm": float(np.abs(action).sum()),
                    "nonzero_positions": int(
                        np.count_nonzero(np.abs(action) > 1.0e-10)
                    ),
                    "optimizer": optimizer,
                    "action": action.tolist(),
                },
                "model_training": _sample_summary(
                    action,
                    training_return,
                    risk_aversion=risk_aversion,
                ),
                "model_validation": _sample_summary(
                    action,
                    validation_return,
                    risk_aversion=risk_aversion,
                ),
                "realized": execute_frozen_action(
                    action,
                    spot_ratio=spot_ratio,
                    entry_calls=entry_calls,
                    exit_calls=exit_calls,
                    cost_rate=0.0,
                ),
            }
        rows.append(row)
        if ordinal == 1 or ordinal % args.progress_every == 0 or index + 1 == stop:
            values = ", ".join(
                f"g={key}: {row['risk_aversions'][key]['realized']['net_pnl']:.6g}"
                for key in row["risk_aversions"]
            )
            print(
                f"{ordinal:,}/{stop - start:,} {row['date']}: {values}",
                flush=True,
            )
        if ordinal % args.checkpoint_every == 0:
            _atomic_json(args.output, report("running"))

    result = report("complete")
    _atomic_json(args.output, result)
    for key, aggregate in result["aggregate"].items():
        pnl = aggregate["realized_trade_pnl"]
        print(
            f"gamma={key}: mean={pnl['mean']:.8g}, median={pnl['median']:.8g}",
            flush=True,
        )
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
