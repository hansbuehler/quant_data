"""Calibrate 2025 latent states with the pre-2025 decoder and AR law frozen."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.dysanos.flex.config import FlexObjectiveConfig
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import (
    _model_prices,
    equal_date_quote_objective,
    evaluate_model,
    materialize_batch,
    materialize_day,
    model_dlvsq_cap_penalty,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.run_strict_pre2025_direct import _train_count
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class ForwardStateFitConfig:
    """State-only quote calibration schedule; shared parameters never update."""

    maximum_epochs: int = 600
    day_batch_size: int = 16
    learning_rate: float = 1.0e-3
    cap_penalty_weight: float = 1.0
    gradient_clip: float = 10.0
    checkpoint_every: int = 5
    seed: int = 20261018

    def __post_init__(self) -> None:
        verify_inp(
            0 < self.maximum_epochs <= 600
            and self.day_batch_size > 0
            and self.learning_rate > 0.0
            and self.cap_penalty_weight >= 0.0
            and self.gradient_clip > 0.0
            and self.checkpoint_every > 0,
            lambda: f"invalid forward state fitting configuration: {self}",
        )


def _project_native_initial_states(
    base_model: JointDirectDynamicsModel,
    native_rows: torch.Tensor,
) -> torch.Tensor:
    """Least-squares initialization only; all quote fitting follows on the held-out day."""
    decoder = base_model.raw_decoder.I0
    loading = decoder.weight.detach().cpu().double()
    bias = decoder.bias.detach().cpu().double()
    targets = native_rows.detach().cpu().double()
    if targets.ndim != 2 or targets.shape[1] != loading.shape[0]:
        raise ValueError("native rows do not match the frozen affine decoder")
    states = torch.linalg.lstsq(loading, (targets - bias).T).solution.T
    if not bool(torch.all(torch.isfinite(states))):
        raise ValueError("forward latent-state initialization is non-finite")
    return states


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(
    *,
    joint_checkpoint: Path,
    native_completion: Path,
    native_checkpoint: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    ticker: str,
    workers: int,
    device: str,
    fit_config: ForwardStateFitConfig,
) -> dict[str, object]:
    """Fit one free 200-coordinate state per 2025 date, never shared weights."""
    base_model, base_audit = load_capped_joint_model(
        joint_checkpoint, device=device, maximum_epochs=600
    )
    completion = json.loads(native_completion.read_text(encoding="utf-8"))
    geometry = _geometry_from_completion(completion, requested_model_business_dtes=None)
    if geometry != base_model.config.geometry:
        raise ValueError("holdout market and frozen decoder geometry differ")
    dates = _market_dates_from_completion(native_completion)
    train_count = _train_count(dates)
    if train_count != base_model.num_days:
        raise ValueError("frozen checkpoint training dates differ from holdout split")
    markets = _load_markets(
        ticker=ticker,
        dates=dates,
        geometry=geometry,
        market_config=_market_config_from_completion(completion),
        workers=workers,
    )
    native_rows, native_source = _load_native_targets(
        native_checkpoint, markets, geometry=geometry
    )
    holdout_days = markets[train_count:]
    holdout_count = len(holdout_days)
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = JointDirectDynamicsModel(
        env, num_days=holdout_count, config=base_model.config
    )
    shared_weights = {
        key: value
        for key, value in base_model.state_dict().items()
        if key != "daily_state_embedding.weight"
    }
    missing, unexpected = model.load_state_dict(shared_weights, strict=False)
    if missing != ["daily_state_embedding.weight"] or unexpected:
        raise ValueError(f"unexpected frozen-weight transfer: {missing}, {unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.daily_state_embedding.weight.requires_grad_(True)
    initial = _project_native_initial_states(base_model, native_rows[train_count:])
    with torch.no_grad():
        model.daily_states.copy_(initial.to(device=model.device, dtype=model.fdtype))
    del base_model

    torch_days = tuple(materialize_day(day, env=env) for day in holdout_days)
    optimizer = torch.optim.Adam(
        [model.daily_state_embedding.weight], lr=fit_config.learning_rate
    )
    checkpoint = checkpoint_directory / "forward-state-only.pt"
    joint_sha256 = checkpoint_digest(joint_checkpoint)
    market_fingerprints = [day.training_fingerprint for day in holdout_days]
    history: list[dict[str, float]] = []
    start_epoch = 0
    with FileLock(str(checkpoint.with_suffix(".lock"))):
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if (
                payload.get("schema") != "dysanos-v11-forward-states-v1"
                or payload.get("joint_sha256") != joint_sha256
                or payload.get("market_fingerprints") != market_fingerprints
                or payload.get("fit_config") != asdict(fit_config)
            ):
                raise ValueError("forward-state checkpoint identity changed")
            start_epoch = int(payload["completed_epochs"])
            if start_epoch > fit_config.maximum_epochs:
                raise ValueError("forward-state checkpoint exceeds the epoch cap")
            with torch.no_grad():
                model.daily_states.copy_(payload["states"].to(model.device))
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            history = payload["history"]

        generator = np.random.default_rng(fit_config.seed)
        for _ in range(start_epoch):
            generator.permutation(holdout_count)
        objective_config = FlexObjectiveConfig(fit_weight=0.05)
        started = perf_counter()
        for epoch in range(start_epoch, fit_config.maximum_epochs):
            permutation = generator.permutation(holdout_count)
            total_loss = 0.0
            batches = 0
            for first in range(0, holdout_count, fit_config.day_batch_size):
                indices = permutation[first : first + fit_config.day_batch_size]
                batch = materialize_batch(torch_days, indices, env=env)
                optimizer.zero_grad(set_to_none=True)
                prices = _model_prices(model, batch)
                quote = equal_date_quote_objective(
                    prices, batch, objective_config=objective_config
                )
                cap = model_dlvsq_cap_penalty(model, batch.day_indices)
                loss = quote + fit_config.cap_penalty_weight * cap
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite forward-state loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [model.daily_state_embedding.weight], fit_config.gradient_clip
                )
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            elapsed = perf_counter() - started
            item = {"epoch": float(epoch + 1), "objective": total_loss / batches}
            history.append(item)
            remaining = elapsed * (fit_config.maximum_epochs - epoch - 1) / (epoch + 1 - start_epoch)
            print(
                f"forward-state epoch={epoch + 1}/{fit_config.maximum_epochs} "
                f"objective={item['objective']:.7g} elapsed={elapsed:.1f}s "
                f"remaining_estimate={remaining:.1f}s",
                flush=True,
            )
            if (epoch + 1) % fit_config.checkpoint_every == 0 or (
                epoch + 1 == fit_config.maximum_epochs
            ):
                _atomic_torch_save(
                    {
                        "schema": "dysanos-v11-forward-states-v1",
                        "joint_sha256": joint_sha256,
                        "market_fingerprints": market_fingerprints,
                        "fit_config": asdict(fit_config),
                        "completed_epochs": epoch + 1,
                        "states": model.daily_states.detach().cpu(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history,
                    },
                    checkpoint,
                )

    evaluation = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=FlexObjectiveConfig(fit_weight=0.05),
        day_batch_size=fit_config.day_batch_size,
        equal_date_weighting=True,
    )
    state_history = checkpoint_directory / "forward-state-history.pt"
    _atomic_torch_save(
        {
            "schema": "dysanos-v11-forward-state-history-v1",
            "joint_sha256": joint_sha256,
            "dates": [value.isoformat() for value in dates[train_count:]],
            "states": model.daily_states.detach().cpu().double(),
            "spot_refs": torch.as_tensor([day.spot_ref for day in holdout_days]),
        },
        state_history,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-forward-state-only-fit-v1",
        "status": "capped" if len(history) == fit_config.maximum_epochs else "incomplete",
        "shared_weights": "frozen throughout; only 2025 daily_state_embedding learned",
        "joint_model": base_audit,
        "joint_sha256": joint_sha256,
        "native_source": native_source,
        "fit_config": asdict(fit_config),
        "training_range": [str(dates[0]), str(dates[train_count - 1])],
        "state_calibration_range": [str(dates[train_count]), str(dates[-1])],
        "objective_history_tail": history[-30:],
        "quote_fit": evaluation.to_dict(),
        "state_history": str(state_history),
        "checkpoint": str(checkpoint),
    }
    write_json_atomic(report, report_directory / "forward-state-only-fit.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-epochs", type=int, default=600)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()
    report = run(
        joint_checkpoint=args.joint_checkpoint,
        native_completion=args.native_completion,
        native_checkpoint=args.native_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        report_directory=args.report_directory,
        ticker=args.ticker,
        workers=args.workers,
        device=args.device,
        fit_config=ForwardStateFitConfig(
            maximum_epochs=args.maximum_epochs,
            day_batch_size=args.day_batch_size,
            learning_rate=args.learning_rate,
        ),
    )
    print(
        f"forward 2025 state calibration {report['status']}: {report['quote_fit']['statistics']['all']['percent_zero']:.3f}% inside bid-ask",
        flush=True,
    )


if __name__ == "__main__":
    main()
