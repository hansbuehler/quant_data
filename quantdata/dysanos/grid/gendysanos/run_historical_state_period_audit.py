"""Run one-day model-to-history portfolio execution from every fitted state."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.verbose import Context

from .historical_state_period_audit import (
    HistoricalStateAuditConfig,
    aggregate_historical_audits,
    audit_historical_state,
    decode_historical_fit,
    summarize_fit_errors,
)
from .torch_prep import GenDysanosExperiment, _load_generator_components

_DEFAULT_OUTPUT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/historical-state-period-audit/"
    "dysanos-historical-states-1d-raw-0bps.json"
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


def _parse_num_alpha(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("num-alpha must be positive or 'none'")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-paths", type=int, default=1_024)
    parser.add_argument("--validation-paths", type=int, default=4_096)
    parser.add_argument("--cost-bps", type=int, choices=(0, 1, 10), default=0)
    parser.add_argument("--num-alpha", type=_parse_num_alpha, default=None)
    parser.add_argument("--seed", type=int, default=731_991)
    parser.add_argument("--decoder-block-size", type=int, default=8_192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    started = time.perf_counter()
    cost_rate = args.cost_bps * 1.0e-4
    config = HistoricalStateAuditConfig(
        training_paths=args.training_paths,
        validation_paths=args.validation_paths,
        cost_rate=cost_rate,
        seed=args.seed,
        decoder_block_size=args.decoder_block_size,
    )
    experiment = GenDysanosExperiment(
        num_samples=2,
        num_alpha=args.num_alpha,
        use_matrix_exp=True,
        bandwidth=1.0,
        trading_frequency="daily",
        return_horizon="incremental",
        delta_hedged=False,
        cost_rate=cost_rate,
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
            "experiment": (
                "historical DYSANOS states; spot reset to one; model-fitted "
                "maximum-mean one-day portfolio; frozen execution "
                "on the next observed market"
            ),
            "model": {
                "surface_state_dimensions": int(sampler.xstate.shape[1] - 1),
                "num_alpha": args.num_alpha,
                "use_matrix_exp": True,
                "surface_first": True,
                "bandwidth": 1.0,
            },
            "portfolio_problem": {
                "objective": "maximize conditional sample mean net PnL",
                "constraints": [
                    "global L1 action norm is at most one",
                ],
                "instruments": (
                    "one spot coordinate and 7x20 DYSANOS-grid calls with "
                    "fixed cash strikes"
                ),
                "transaction_cost_rate": cost_rate,
                "delta_hedged": False,
            },
            "out_of_sample_rule": (
                "freeze each fitted action and mark its fixed contracts from "
                "the observed surface on date t to the observed surface at the "
                "next market record; the next record is treated as one trading "
                "day and is not used during optimization"
            ),
            "configuration": asdict(config),
            "historical_interval": {
                "start_index": start,
                "stop_index_exclusive": stop,
                "states_requested": stop - start,
                "first_date": str(np.asarray(market["dates"])[start]),
                "last_entry_date": str(np.asarray(market["dates"])[stop - 1]),
                "last_exit_date": str(np.asarray(market["dates"])[stop]),
            },
            "surface_fit_error_all_1412_states": fit_aggregate,
            "states": rows,
            "aggregate": aggregate_historical_audits(rows) if rows else None,
            "elapsed_seconds": time.perf_counter() - started,
        }

    for ordinal, index in enumerate(range(start, stop), start=1):
        row = audit_historical_state(
            model,
            sampler,
            market,
            index=index,
            fit_error=fit_rows[index],
            config=config,
        )
        rows.append(row)
        if ordinal == 1 or ordinal % args.progress_every == 0 or index + 1 == stop:
            execution = row["next_market_execution"]
            print(
                f"{ordinal:,}/{stop - start:,} {row['date']}: "
                f"L1={row['portfolio']['l1_norm']:.4g}, "
                f"validation-mean={row['model_validation']['mean_pnl']:.7g}, "
                f"next-market-pnl={execution['net_pnl']:.7g}",
                flush=True,
            )
        if ordinal % args.checkpoint_every == 0:
            _atomic_json(args.output, report("running"))

    result = report("complete")
    _atomic_json(args.output, result)
    aggregate = result["aggregate"]
    print(
        f"complete: {aggregate['positive_model_validation_mean_count']:,}/"
        f"{aggregate['historical_states_executed']:,} positive validation "
        f"means; next-market mean="
        f"{aggregate['realized_next_market_pnl']['mean']:.7g}; "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
