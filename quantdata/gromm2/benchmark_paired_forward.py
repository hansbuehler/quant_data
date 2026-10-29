"""Validate shared active towers against the unchanged reference on allocated CUDA."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from quantdata.gromm2.forward_hedging import (
    ForwardScenarioConfig,
    independent_normal_block,
    scenario_function,
)
from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model
from quantdata.gromm2.paired_forward import PairedSimulator
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--factor-law", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1024)
    args = parser.parse_args()
    progress = {"stage": "loading", "completed": 0, "total": 3}
    overall_started = time.monotonic()

    def heartbeat():
        while True:
            time.sleep(60)
            print(json.dumps(dict(event="equivalence_progress", elapsed_seconds=time.monotonic() - overall_started,
                **progress)), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    check_resource_headroom()
    arrays, metadata = load_legacy_ar1_model(history=args.history, estimate=args.estimate,
        factor_law=args.factor_law, origin="2025-08-29", backend="gpu")
    config = ForwardScenarioConfig(paths=args.paths, batch_size=args.paths, substeps=8,
        seed=2026102801, backend="gpu", linear_solver="pcr")
    simulator = PairedSimulator(arrays, config)
    reference, book = scenario_function(**arrays, background_volatility=0.05, config=config)
    records = []

    def reference_blocks(function, factors, background):
        blocks = []
        for start in range(0, args.paths, 1024):
            progress["reference_chunk"] = start // 1024
            values = function(factors[:, :, start:start + 1024], background[:, start:start + 1024])
            blocks.append(tuple(np.asarray(value) for value in values[:3]))
        return tuple(np.concatenate([block[index] for block in blocks], axis=0) for index in range(3))

    for split in ("train", "validation"):
        progress["stage"] = f"{split}-reference"
        factors, background = independent_normal_block(config=config, split=split, completed=0,
            count=args.paths, factors=arrays["loadings"].shape[-1])
        started = time.monotonic()
        baseline = reference_blocks(reference, factors, background)
        reference_seconds = time.monotonic() - started
        started = time.monotonic()
        paired = simulator.simulate(factors, background, progress)
        paired_seconds = time.monotonic() - started
        front_count = arrays["supports"].shape[-1]
        factor_count = arrays["loadings"].shape[-1]
        prefix = front_count + 2 * factor_count + len(arrays["supports"])
        time_offset = prefix + 3
        reference_core = np.concatenate((baseline[0][:, :, :prefix], baseline[0][:, :, -2 * factor_count:],
            baseline[0][:, :, time_offset:time_offset + 2]), axis=-1).astype("f4")
        normalization = np.maximum(reference_core.std(axis=(0, 1), dtype=np.float64), 1e-4)
        normalized_error = float(np.max(np.abs(paired["core_features"] - reference_core) / normalization))
        error = float(np.max(np.abs(paired["core_features"] - reference_core)))
        price_errors = []
        for old, (expiry, strike) in enumerate(zip(book["option_expiry_days"], book["option_strikes"], strict=True)):
            matches = np.flatnonzero((simulator.primary["starts"] == 0) & (simulator.primary["expiries"] == expiry)
                & (simulator.primary["levels"] == strike))
            if len(matches):
                actual = paired["option_prices"][0, :, :, matches[0]]
                np.testing.assert_allclose(actual, baseline[1][:, :, old + 1], rtol=1e-10, atol=5e-12)
                price_errors.append(float(np.max(np.abs(actual - baseline[1][:, :, old + 1]))))
        np.testing.assert_allclose(paired["payoff"][0], baseline[2], rtol=1e-11, atol=5e-12)
        np.testing.assert_allclose(paired["spots"][0], baseline[1][:, :, 0], rtol=1e-11, atol=5e-12)
        if normalized_error > 1e-3:
            raise AssertionError(f"normalized latent feature discrepancy {normalized_error} exceeds 1e-3")
        record = dict(split=split, paths=args.paths, reference_seconds=reference_seconds,
            paired_both_backgrounds_seconds=paired_seconds, maximum_price_error=max(price_errors),
            maximum_feature_error=error, maximum_floor_normalized_feature_error=normalized_error,
            maximum_payoff_error=float(np.max(np.abs(paired["payoff"][0] - baseline[2]))),
            maximum_mass_or_mean_error=float(np.max(paired["diagnostics"][:, 1:3])))
        print(json.dumps(record), flush=True)
        records.append(record)
        progress["completed"] += 1
        check_resource_headroom()
    zero_reference, _ = scenario_function(**arrays, background_volatility=0.0, config=replace(config))
    progress["stage"] = "zero-background-reference"
    zero = reference_blocks(zero_reference, factors, background)
    np.testing.assert_allclose(paired["payoff"][1], zero[2], rtol=1e-11, atol=5e-12)
    np.testing.assert_allclose(paired["spots"][1], zero[1][:, :, 0], rtol=1e-11, atol=5e-12)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_json(args.output_directory / "equivalence.json", dict(passed=True, records=records,
        model=metadata, factor_basket_reports=simulator.basket_reports,
        zero_background_payoff_error=float(np.max(np.abs(paired["payoff"][1] - zero[2]))),
        interpretation="Prices/spot/payoff tested to float64 tolerances; float32 latent features additionally checked after normalization with floor 1e-4. Warm timing is the validation record; first record includes compilation."))


if __name__ == "__main__":
    main()
