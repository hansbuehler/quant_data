"""Train the Torch-native, tick-aware Phase 3 SPX half-spread network."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch

from quantdata.dysanos.flex.config import FlexMarketConfig, ForwardMuFlexGeometryConfig
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import market_process_count
from quantdata.dysanos.flex.run_spread_benchmark import (
    DEFAULT_DATES_REPORT,
    DEFAULT_OUTPUT_DIRECTORY,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    _dates_from_report,
    _load_observations,
    _proportional_diagnostics,
    _stack,
    regression_metrics,
)
from quantdata.dysanos.flex.spread_cost import (
    SpreadFeedForwardConfig,
    SpreadObservations,
    TorchSpreadCostModel,
    equal_date_sample_weights,
    save_torch_spread_cost_model,
)
from quantdata.pool import PoolConfig
from quantdata.torch import BaseEnvironment


def _training_mask(
    data: dict[str, np.ndarray],
    *,
    config: SpreadFeedForwardConfig,
) -> np.ndarray:
    """Remove books whose dollar premium is dominated by minimum-tick effects."""
    premium_dollars = data["fit_price"] * data["spot_ref"]
    return np.isfinite(premium_dollars) & (
        premium_dollars >= config.minimum_option_price_dollars
    )


def _normalizers(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    weights = equal_date_sample_weights(data["date_index"][mask])
    premium_dollars = data["fit_price"][mask] * data["spot_ref"][mask]
    features = np.column_stack(
        (
            data["vega"][mask],
            data["sqrt_tte"][mask],
            data["nms"][mask],
            data["log_strike"][mask],
            data["delta"][mask],
            np.log(np.maximum(premium_dollars, 1.0e-12)),
        )
    )
    total = np.sum(weights)
    feature_mean = np.sum(weights[:, None] * features, axis=0) / total
    feature_variance = (
        np.sum(weights[:, None] * (features - feature_mean) ** 2, axis=0) / total
    )
    feature_scale = np.sqrt(np.maximum(feature_variance, 1.0e-12))
    log_target = np.log(data["half_spread"][mask])
    target_mean = float(np.sum(weights * log_target) / total)
    target_variance = float(
        np.sum(weights * (log_target - target_mean) ** 2) / total
    )
    return feature_mean, feature_scale, target_mean, np.sqrt(max(target_variance, 1.0e-12))


def _tensor_data(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    env: BaseEnvironment,
) -> dict[str, torch.Tensor]:
    weights = equal_date_sample_weights(data["date_index"][mask])
    values = {
        "option_price": data["fit_price"][mask],
        "spot": data["spot_ref"][mask],
        "vega": data["vega"][mask],
        "sqrt_tte": data["sqrt_tte"][mask],
        "nms": data["nms"][mask],
        "log_strike": data["log_strike"][mask],
        "delta": data["delta"][mask],
        "target": data["half_spread"][mask],
        "weight": weights,
    }
    return {
        name: torch.as_tensor(value, dtype=env.fdtype, device=env.device)
        for name, value in values.items()
    }


def _predict_numpy(
    model: TorchSpreadCostModel,
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["half_spread"]), batch_size):
            stop = min(start + batch_size, len(data["half_spread"]))
            values = [
                torch.as_tensor(
                    data[name][start:stop],
                    dtype=model.fdtype,
                    device=model.device,
                )
                for name in (
                    "fit_price",
                    "spot_ref",
                    "vega",
                    "sqrt_tte",
                    "nms",
                    "log_strike",
                    "delta",
                )
            ]
            predictions.append(model(*values).detach().cpu().numpy())
    return np.concatenate(predictions)


def _metrics(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    price_floor: float,
) -> dict[str, object]:
    selected = {name: value[mask] for name, value in data.items()}
    selected_prediction = prediction[mask]
    return {
        "regression": regression_metrics(
            selected["half_spread"],
            selected_prediction,
            selected["date_index"],
        ),
        "proportional_cost": _proportional_diagnostics(
            selected,
            selected_prediction,
            price_floor=price_floor,
        ),
    }


def _train(
    model: TorchSpreadCostModel,
    train: dict[str, torch.Tensor],
    *,
    config: SpreadFeedForwardConfig,
    compile_model: bool,
) -> tuple[list[dict[str, float]], float]:
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    callable_model = torch.compile(model, dynamic=True) if compile_model else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
    )
    generator = torch.Generator(device=model.device)
    generator.manual_seed(config.random_seed)
    num_quotes = len(train["target"])
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for epoch in range(config.num_epochs):
        order = torch.randperm(num_quotes, generator=generator, device=model.device)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, num_quotes, config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = callable_model(
                train["option_price"][indices],
                train["spot"][indices],
                train["vega"][indices],
                train["sqrt_tte"][indices],
                train["nms"][indices],
                train["log_strike"][indices],
                train["delta"][indices],
            )
            log_error = torch.log(prediction) - torch.log(train["target"][indices])
            loss_values = torch.nn.functional.smooth_l1_loss(
                log_error,
                torch.zeros_like(log_error),
                beta=0.10,
                reduction="none",
            )
            weight = train["weight"][indices]
            loss = torch.sum(weight * loss_values) / torch.sum(weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            numerator += float(torch.sum(weight * loss_values).detach().cpu())
            denominator += float(torch.sum(weight).detach().cpu())
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "date_weighted_log_huber": numerator / denominator,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"spread-ffw epoch={epoch + 1}/{config.num_epochs} "
                f"log_huber={history[-1]['date_weighted_log_huber']:.8g}",
                flush=True,
            )
    return history, perf_counter() - started


def _markdown(report: dict[str, object]) -> str:
    filtering = cast(dict[str, object], report["filtering"])
    validation = cast(dict[str, object], report["validation"])
    eligible = cast(dict[str, object], validation["eligible"])
    full = cast(dict[str, object], validation["all_with_tick_floor"])
    eligible_regression = cast(dict[str, float], eligible["regression"])
    full_regression = cast(dict[str, float], full["regression"])
    return "\n".join(
        [
            "# Phase 3 differentiable half-spread network",
            "",
            "A two-hidden-layer width-64 SiLU network predicts log normalized "
            "half-spread and remains native Torch code inside the simulator. Training "
            "uses only 2022--2024 and gives each date equal total weight; 2025 is "
            "frozen temporal validation. No 2026 data is read.",
            "",
            "## Tick handling",
            "",
            f"- Input books below `${filtering['minimum_option_price_dollars']}` "
            "premium are excluded from training.",
            "- Inference is floored at half of a $0.05 tick below $3 premium and "
            "half of a $0.10 tick otherwise.",
            f"- The filter retains {float(filtering['train_retained_percent']):.3f}% "
            f"of training books and {float(filtering['validation_retained_percent']):.3f}% "
            "of validation books.",
            "- Dollar premium is approximated as normalized AF price times spot. "
            "This intentionally avoids rates/forward inputs but is an approximation.",
            "",
            "## 2025 results",
            "",
            "| universe | date-balanced MAE | date-balanced RMSE | date-balanced R2 |",
            "| --- | ---: | ---: | ---: |",
            f"| eligible | {eligible_regression['date_balanced_mae']:.8g} | "
            f"{eligible_regression['date_balanced_rmse']:.8g} | "
            f"{eligible_regression['date_balanced_r2']:.6f} |",
            f"| all, tick-floor extrapolation | {full_regression['date_balanced_mae']:.8g} | "
            f"{full_regression['date_balanced_rmse']:.8g} | "
            f"{full_regression['date_balanced_r2']:.6f} |",
            "",
            "The low-premium threshold, exchange-tick approximation, architecture, "
            "log-Huber loss, and optimizer schedule are ad-hoc Phase 3 choices. The "
            "all-book row explicitly extrapolates the network beyond its training "
            "filter; the eligible row is the primary test.",
            "",
        ]
    )


def run_spread_ffw(
    *,
    ticker: str,
    dates_report: Path,
    output_directory: Path,
    workers: int,
    config: SpreadFeedForwardConfig,
    compile_model: bool,
) -> dict[str, object]:
    """Load the frozen split, train on GPU0, and write exact evaluation output."""
    dates = _dates_from_report(dates_report)
    geometry = ForwardMuFlexGeometryConfig()
    market_config = FlexMarketConfig()
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    jobs = {
        value: pool.delayed(_load_observations)(
            ticker,
            value,
            geometry=geometry,
            market_config=market_config,
        )
        for value in dates
    }
    observations: list[SpreadObservations] = []
    for index, (trading_date, value) in enumerate(pool.parallel(jobs), start=1):
        if value is None:
            raise RuntimeError(f"Missing previously valid market date {trading_date}")
        observations.append(value)
        if index % 50 == 0 or index == len(dates):
            print(f"spread-ffw markets={index}/{len(dates)} date={trading_date}", flush=True)
    observations.sort(key=lambda value: value.trading_date)
    train_observations = tuple(
        value for value in observations if value.trading_date <= TRAIN_END
    )
    validation_observations = tuple(
        value
        for value in observations
        if VALIDATION_START <= value.trading_date <= VALIDATION_END
    )
    if len(train_observations) != 712 or len(validation_observations) != 162:
        raise RuntimeError("Unexpected frozen spread train/validation date counts")
    train_data = _stack(train_observations)
    validation_data = _stack(validation_observations)
    train_mask = _training_mask(train_data, config=config)
    validation_mask = _training_mask(validation_data, config=config)
    normalizers = _normalizers(train_data, train_mask)
    env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    model = TorchSpreadCostModel(
        env,
        config=config,
        feature_mean=torch.as_tensor(normalizers[0]),
        feature_scale=torch.as_tensor(normalizers[1]),
        log_target_mean=torch.as_tensor(normalizers[2]),
        log_target_scale=torch.as_tensor(normalizers[3]),
    )
    history, elapsed = _train(
        model,
        _tensor_data(train_data, train_mask, env=env),
        config=config,
        compile_model=compile_model,
    )
    train_prediction = _predict_numpy(model, train_data, batch_size=config.batch_size)
    validation_prediction = _predict_numpy(
        model,
        validation_data,
        batch_size=config.batch_size,
    )
    all_train = np.ones(len(train_mask), dtype=bool)
    all_validation = np.ones(len(validation_mask), dtype=bool)
    report: dict[str, object] = {
        "schema": "flex-phase3-spread-ffw-v1",
        "status": "complete",
        "ticker": ticker,
        "source_dates_report": str(dates_report),
        "split": {
            "train_start": train_observations[0].trading_date.isoformat(),
            "train_end": train_observations[-1].trading_date.isoformat(),
            "validation_start": validation_observations[0].trading_date.isoformat(),
            "validation_end": validation_observations[-1].trading_date.isoformat(),
            "no_2026": True,
        },
        "config": asdict(config),
        "compile_model": compile_model,
        "features": [
            "vega",
            "sqrt_tte",
            "nms",
            "log_strike",
            "signed_delta",
            "log_option_price_dollars",
        ],
        "target": "log normalized price half-spread, date-weighted Huber loss",
        "filtering": {
            "minimum_option_price_dollars": config.minimum_option_price_dollars,
            "train_quotes_before": int(len(train_mask)),
            "train_quotes_after": int(np.sum(train_mask)),
            "train_retained_percent": float(np.mean(train_mask) * 100.0),
            "validation_quotes_before": int(len(validation_mask)),
            "validation_quotes_after": int(np.sum(validation_mask)),
            "validation_retained_percent": float(np.mean(validation_mask) * 100.0),
        },
        "normalizers": {
            "feature_mean": normalizers[0].tolist(),
            "feature_scale": normalizers[1].tolist(),
            "log_target_mean": normalizers[2],
            "log_target_scale": normalizers[3],
        },
        "training": {
            "history": history,
            "elapsed_seconds": elapsed,
            "train": {
                "eligible": _metrics(
                    train_data,
                    train_prediction,
                    train_mask,
                    price_floor=config.proportional_price_floor,
                ),
                "all_with_tick_floor": _metrics(
                    train_data,
                    train_prediction,
                    all_train,
                    price_floor=config.proportional_price_floor,
                ),
            },
        },
        "validation": {
            "eligible": _metrics(
                validation_data,
                validation_prediction,
                validation_mask,
                price_floor=config.proportional_price_floor,
            ),
            "all_with_tick_floor": _metrics(
                validation_data,
                validation_prediction,
                all_validation,
                price_floor=config.proportional_price_floor,
            ),
        },
        "ad_hoc_choices": [
            "$0.25 minimum input premium",
            "$0.05/$0.10 SPX tick schedule split at $3 premium",
            "two width-64 SiLU hidden layers",
            "log-half-spread Huber loss with beta 0.10",
            "AdamW learning rate, weight decay, batch size, and cosine horizon",
        ],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = output_directory / "feed_forward.pt"
    save_torch_spread_cost_model(
        model,
        checkpoint,
        metadata={
            "split": report["split"],
            "filtering": report["filtering"],
            "training_history": history,
        },
    )
    report["checkpoint"] = str(checkpoint)
    json_path = output_directory / "phase3-spread-ffw.json"
    markdown_path = output_directory / "phase3-spread-ffw.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {json_path} and {checkpoint}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--dates-report", type=Path, default=DEFAULT_DATES_REPORT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=SpreadFeedForwardConfig().num_epochs)
    arguments = parser.parse_args()
    if arguments.workers <= 0 or arguments.epochs <= 0:
        parser.error("--workers and --epochs must be positive")
    run_spread_ffw(
        ticker=arguments.ticker,
        dates_report=arguments.dates_report,
        output_directory=arguments.output_directory,
        workers=arguments.workers,
        config=SpreadFeedForwardConfig(num_epochs=arguments.epochs),
        compile_model=not arguments.no_compile,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_spread_ffw")
