"""Train LSTM and selective-SSM conditional normal innovations after AR fits."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.v11.conditional_sequence import (
    ConditionalInnovationModel,
    ConditionalSequenceConfig,
    baseline_normal_nll,
    conditional_normal_nll,
)
from quantdata.dysanos.v11.conditional_sequence_data import (
    ConditionalSequenceData,
    prepare_conditional_sequence_data,
)
from quantdata.torch import BaseEnvironment


def _data_tensors(
    data: ConditionalSequenceData, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = torch.as_tensor(data.sequences, dtype=torch.float32, device=device)
    score = torch.as_tensor(data.current_spot_scores, dtype=torch.float32, device=device)
    white = torch.as_tensor(data.observed_white, dtype=torch.float32, device=device)
    train = torch.as_tensor(data.train_mask, dtype=torch.bool, device=device)
    validation = torch.as_tensor(data.validation_mask, dtype=torch.bool, device=device)
    return sequence, score, white, train, validation


def _fit_one(
    *,
    architecture: str,
    data: ConditionalSequenceData,
    env: BaseEnvironment,
    checkpoint: Path,
    max_epochs: int,
    learning_rate: float,
    source_identity: dict[str, str],
) -> dict[str, object]:
    configuration = ConditionalSequenceConfig(
        label=f"dysanos-v11-daily-{architecture}",
        architecture=architecture,
        input_size=data.sequences.shape[-1],
        mean_rank=data.white_mean_basis.shape[1],
        state_size=data.observed_white.shape[1],
        sequence_length=data.sequences.shape[1],
    )
    model = ConditionalInnovationModel(
        env,
        config=configuration,
        white_mean_basis=torch.as_tensor(data.white_mean_basis),
    )
    sequence, score, white, train, validation = _data_tensors(
        data, device=env.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1.0e-6
    )
    best_value = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(checkpoint.with_suffix(".lock"))):
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mean, log_scale = model(sequence[train], score[train])
            loss = conditional_normal_nll(white[train], mean, log_scale)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite {architecture} training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_mean, val_log_scale = model(
                    sequence[validation], score[validation]
                )
                validation_loss = conditional_normal_nll(
                    white[validation], val_mean, val_log_scale,
                    mean_penalty=0.0, scale_penalty=0.0,
                )
            train_value = float(loss.detach().cpu())
            validation_value = float(validation_loss.cpu())
            if not math.isfinite(validation_value):
                raise RuntimeError(
                    f"nonfinite {architecture} validation loss at epoch {epoch}"
                )
            scheduler.step(validation_value)
            if validation_value < best_value - 1.0e-5:
                best_value = validation_value
                best_epoch = epoch
                stale_epochs = 0
                payload = {
                    "schema": "dysanos-v11-conditional-sequence-v1",
                    "model_config": asdict(configuration),
                    "model_config_version": configuration.version.unique_id48,
                    "model_user_version": configuration.user_version,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "epoch": epoch,
                    "validation_nll_per_dimension": best_value,
                    "source_identity": source_identity,
                    "feature_mean": data.feature_mean,
                    "feature_std": data.feature_std,
                    "state_mean": data.state_mean,
                    "state_basis": data.state_basis,
                    "initial_context": data.initial_context,
                    "initial_state": data.initial_state,
                }
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_nll_per_dimension": train_value,
                    "validation_nll_per_dimension": validation_value,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if epoch == 1 or epoch % 10 == 0:
                elapsed = perf_counter() - started
                eta = elapsed * (max_epochs - epoch) / epoch
                print(
                    f"conditional {architecture} epoch={epoch}/{max_epochs} "
                    f"train={train_value:.6f} val={validation_value:.6f} "
                    f"best_epoch={best_epoch} elapsed={elapsed:.1f}s "
                    f"remaining_estimate={eta:.1f}s",
                    flush=True,
                )
            if epoch >= 50 and stale_epochs >= 50:
                break
    with torch.inference_mode():
        baseline_train = baseline_normal_nll(white[train])
        baseline_validation = baseline_normal_nll(white[validation])
    return {
        "architecture": architecture,
        "description": (
            "LSTM" if architecture == "lstm" else
            "input-selective diagonal state-space model (Mamba-style, not Mamba-3)"
        ),
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "epoch_cap": max_epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll_per_dimension": best_value,
        "baseline_train_nll_per_dimension": baseline_train,
        "baseline_validation_nll_per_dimension": baseline_validation,
        "best_checkpoint": str(checkpoint.resolve()),
        "history_tail": history[-50:],
    }


def run(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    dynamics_parameters: Path,
    native_atm_history: Path,
    historical_dataset: Path,
    checkpoint_directory: Path,
    report_output: Path,
    max_epochs: int = 600,
    learning_rate: float = 2.0e-3,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Fit two small conditional-law alternatives with a 2024 validation block."""
    if max_epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("epoch cap and learning rate must be positive")
    torch.manual_seed(20261018)
    data = prepare_conditional_sequence_data(
        joint_checkpoint=joint_checkpoint,
        target_history=target_history,
        dynamics_parameters=dynamics_parameters,
        native_atm_history=native_atm_history,
        historical_dataset=historical_dataset,
        device=device,
    )
    env = BaseEnvironment(precision=32, device=device, debug=False)
    identity = {
        "daily_joint_checkpoint_sha256": checkpoint_digest(joint_checkpoint),
        "daily_dynamics_parameters_sha256": checkpoint_digest(dynamics_parameters),
        "native_atm_history_sha256": checkpoint_digest(native_atm_history),
    }
    print(
        f"conditional dataset train={int(np.sum(data.train_mask))} "
        f"validation={int(np.sum(data.validation_mask))} "
        f"window={data.sequences.shape[1]}", flush=True,
    )
    fits = [
        _fit_one(
            architecture=architecture,
            data=data,
            env=env,
            checkpoint=checkpoint_directory / f"{architecture}.pt",
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            source_identity=identity,
        )
        for architecture in ("lstm", "selective_ssm")
    ]
    report: dict[str, object] = {
        "schema": "dysanos-v11-conditional-sequence-comparison-v1",
        "status": "complete",
        "source_identity": identity,
        "training_samples": int(np.sum(data.train_mask)),
        "validation_samples": int(np.sum(data.validation_mask)),
        "validation_period": [
            data.dates[int(data.transition_indices[data.validation_mask][0])],
            data.dates[int(data.transition_indices[data.validation_mask][-1])],
        ],
        "objective": "regularized one-day conditional normal NLL of 200 white state innovations, given past 20 sessions and realized front spot score",
        "fits": fits,
    }
    write_json_atomic(report, report_output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--dynamics-parameters", type=Path, required=True)
    parser.add_argument("--native-atm-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"conditional sequence comparison {report['status']}: {arguments.report_output}", flush=True)


if __name__ == "__main__":
    main()
