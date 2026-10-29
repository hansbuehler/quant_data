"""Matched SANOS normal-statistical PCA hedges at minimum variance."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.minimum_variance_daily import (
    _reference_quantities,
    minimum_normal_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    build_historical_return_panel,
)


def run_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    reference_run: Path,
    position_cap: float = 1_000.0,
) -> list[dict]:
    """Fit eight spot-residual covariance axes on 252 prior SANOS returns."""
    targets = daily_digital_targets(maturity_business_days=1)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != quotes for sample in samples[1:]):
        raise RuntimeError("Targets do not share one hedge book")
    quantities_by_label = _reference_quantities(reference_run, origin_date)
    if set(quantities_by_label) != {target.label for target in targets}:
        raise ValueError("Reference target quantities do not align")
    quantities = np.asarray([quantities_by_label[target.label] for target in targets])
    returns = build_historical_return_panel(
        panel,
        origin_date=origin_date,
        option_quotes=quotes,
        targets=targets,
        count=252,
        price_source="sanos",
    )
    actions, diagnostics = minimum_normal_actions(
        returns.instrument_returns,
        returns.target_returns * quantities[None, :],
        position_cap=position_cap,
        spot_residual_pca_rank=8,
    )
    results = []
    for index, sample in enumerate(samples):
        action = actions[:, index]
        results.append(asdict(evaluate_daily_digital_hedge(
            sample,
            model="sanos-normal-statistical-pca",
            measure="historical-P",
            spot_action=float(action[0]),
            option_actions=action[1:],
            model_price=sample.market_price,
            objective_value=float(diagnostics[index]["training_variance"]),
            converged=True,
            cap_hits=int(diagnostics[index]["cap_hits"]),
            claim_quantity=float(quantities[index]),
            diagnostics={
                **diagnostics[index],
                "objective": "minimum-variance",
                "risk_parameter": "infinity",
                "training_observations": len(returns.training_origin_dates),
                "instrument_count": returns.instrument_returns.shape[1],
                "historical_return_price_source": "sanos",
            },
        )))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dates-file", type=Path, required=True)
    parser.add_argument("--reference-statistical-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    dates = tuple(date.fromisoformat(value) for value in arguments.dates_file.read_text().splitlines())
    if not dates:
        raise ValueError("No evaluation dates")
    panel = PublicHedgingPanel(arguments.dataset)
    records = arguments.output / "records"
    records.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for completed, origin_date in enumerate(dates, start=1):
        path = records / f"{origin_date.isoformat()}.json"
        if not path.exists():
            results = run_day(
                panel,
                origin_date=origin_date,
                reference_run=arguments.reference_statistical_run,
            )
            temporary = path.with_suffix(".partial.json")
            temporary.write_text(json.dumps({"results": results}, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        elapsed = time.monotonic() - started
        remaining = elapsed * (len(dates) - completed) / completed
        print(
            f"normal-PCA progress {completed}/{len(dates)} origin={origin_date} "
            f"elapsed_s={elapsed:.1f} eta_s={remaining:.1f}",
            flush=True,
        )
    (arguments.output / "summary.json").write_text(
        json.dumps({
            "schema": "sanos-normal-statistical-pca-minimum-variance-v1",
            "dates": [day.isoformat() for day in dates],
            "spot_residual_pca_rank": 8,
            "risk_parameter": "infinity",
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
