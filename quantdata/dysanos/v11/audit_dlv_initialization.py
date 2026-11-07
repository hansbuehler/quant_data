"""Evaluate algebraic DLV initial states against licensed market quotes."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quantdata.dysanos.flex.config import FlexObjectiveConfig, NativeStateConfig, PcaDecoderConfig
from quantdata.dysanos.flex.model import LatentStateSanosModel, NativeStateSanosModel
from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.training import evaluate_model, initialize_pca_decoder_, materialize_day
from quantdata.dysanos.v11.licensed_market import licensed_market_day
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry, _primary_fit_metrics
from quantdata.torch import BaseEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    for argument in ("completion", "panel", "archive", "initialization", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    with np.load(args.initialization / "states.npz") as saved:
        dates = tuple(date.fromisoformat(str(value)) for value in saved["dates"])
        raw = saved["raw_states"].copy()
    env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
    days = []
    for i, day in enumerate(dates, 1):
        days.append(licensed_market_day(day, panel=args.panel, archive=args.archive, geometry=geometry))
        if i % 25 == 0:
            print(f"loaded={i}/{len(dates)}", flush=True)
    markets = tuple(materialize_day(day, env=env) for day in days)
    raw_tensor = torch.as_tensor(raw, dtype=env.fdtype, device=env.device)
    native = NativeStateSanosModel(env, num_days=len(dates), config=NativeStateConfig(label="dlv-initial-audit", geometry=geometry))
    with torch.no_grad():
        native.daily_raw_states.copy_(raw_tensor)
    objective = FlexObjectiveConfig(fit_weight=0.05)
    native_fit = evaluate_model(native, markets, env=env, objective_config=objective, day_batch_size=64, equal_date_weighting=True)
    print("native market evaluation complete", flush=True)
    del native
    model = LatentStateSanosModel(env, num_days=len(dates), config=PcaDecoderConfig(label="dlv-initial-affine-audit", state_size=200, geometry=geometry))
    pca = initialize_pca_decoder_(model, raw_tensor)
    affine_fit = evaluate_model(model, markets, env=env, objective_config=objective, day_batch_size=64, equal_date_weighting=True)
    del model
    model400 = LatentStateSanosModel(env, num_days=len(dates), config=PcaDecoderConfig(label="dlv-initial-affine400-audit", state_size=400, geometry=geometry))
    pca400 = initialize_pca_decoder_(model400, raw_tensor)
    affine400_fit = evaluate_model(model400, markets, env=env, objective_config=objective, day_batch_size=64, equal_date_weighting=True)
    records = []
    for day, native_prices, affine_prices, affine400_prices in zip(days, native_fit.fits, affine_fit.fits, affine400_fit.fits, strict=True):
        quotes = pd.read_parquet(args.panel / "days" / f"{day.trading_date}.parquet")
        quotes = quotes.loc[~quotes.is_am]
        keys = [f"{day.trading_date}:{expiry}:{strike:.8g}:{side}" for expiry, strike, side in zip(quotes.expiry, quotes.strike, quotes.option_type, strict=True)]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate licensed quote keys on {day.trading_date}")
        reference = pd.Series(quotes.fit_price.to_numpy(dtype=np.float64) / (quotes.df.to_numpy(dtype=np.float64) * quotes.fwd.to_numpy(dtype=np.float64)), index=keys)
        sanos = reference.loc[list(day.quote_ids)].to_numpy()
        half = (day.asks - day.bids) / 2
        records.append(pd.DataFrame({"dte": day.quote_business_dtes, "bid": day.bids, "ask": day.asks,
                                     "half": half, "sanos": sanos, "native": native_prices, "affine": affine_prices, "affine400": affine400_prices}))
    book = pd.concat(records, ignore_index=True)
    for name in ("sanos", "native", "affine", "affine400"):
        book[f"{name}_inside"] = (book[name] >= book.bid) & (book[name] <= book.ask)
        book[f"{name}_excess"] = (np.maximum(book.bid-book[name], 0) + np.maximum(book[name]-book.ask, 0)) / book.half
    book["affine_native_abs"] = np.abs(book.affine-book.native) / book.half
    book["native_sanos_abs"] = np.abs(book.native-book.sanos) / book.half
    summary = book.groupby("dte").agg(quotes=("bid", "size"), sanos_inside=("sanos_inside", "mean"),
        native_inside=("native_inside", "mean"), affine_inside=("affine_inside", "mean"), affine400_inside=("affine400_inside", "mean"),
        sanos_excess=("sanos_excess", "mean"), native_excess=("native_excess", "mean"),
        affine_excess=("affine_excess", "mean"), affine400_excess=("affine400_excess", "mean"), native_sanos_abs=("native_sanos_abs", "mean"),
        affine_native_abs=("affine_native_abs", "mean"))
    summary.to_csv(args.output.with_name("expiry-breakdown.csv"))
    report = {"dates": [str(dates[0]), str(dates[-1])], "native": _primary_fit_metrics(native_fit),
              "affine_initial": _primary_fit_metrics(affine_fit), "affine400_initial": _primary_fit_metrics(affine400_fit), "pca": pca, "pca400": pca400,
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "training_performed": False}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
