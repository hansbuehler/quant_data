"""Calibrate recurrent DYSANOS dynamics to full-history variance-index laws.

The August-29-2025 [-4,3] affine decoder and AR coefficients remain frozen.
Only small LSTM/selective-SSM innovation models are trained. All available
historical SANOS dates supply marginal targets; this is not forward testing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch
from filelock import FileLock

from quantdata.biz_calendar import count_business_days_between
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import load_presentation_model
from quantdata.dysanos.v11.cpu_pricing import enable_fast_cpu_decoder
from quantdata.dysanos.v11.plot_vix_sequence import plot_run
from quantdata.dysanos.v11.variance_indices import BUSINESS_DTES, decoded_variance_indices
from quantdata.dysanos.v11.vix_sequence import (
    VixSequenceConfig,
    VixSequenceModel,
    quantile_distribution_loss,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class VixSequenceTrainConfig:
    epochs: int = 200
    learning_rate: float = .003
    chains: int = 128
    steps_per_update: int = 4
    audit_paths: int = 2048
    audit_burn_in: int = 256
    seed: int = 20261027

    def __post_init__(self) -> None:
        if min(self.epochs, self.chains, self.steps_per_update, self.audit_paths, self.audit_burn_in) < 1 or self.learning_rate <= 0:
            raise ValueError("Training dimensions, learning rate and audit length must be positive")


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _correlation(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0)
    centered = centered / torch.clamp(centered.square().mean(dim=0).sqrt(), min=1e-8)
    return centered.T @ centered / len(values)


def run(*, run_directory: Path, historic_targets: Path, gaussian_samples: Path, output: Path, train: VixSequenceTrainConfig) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train.seed)
    checkpoint = run_directory / "checkpoints/joint-direct-200.pt"
    source_sha = checkpoint_digest(checkpoint)
    reference = json.loads((gaussian_samples / "summary.json").read_text())
    if reference["checkpoint_sha256"] != source_sha:
        raise ValueError("Gaussian dynamics recovery does not match the frozen model")
    _, decoder = load_presentation_model(checkpoint, device="cpu", precision=64)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    enable_fast_cpu_decoder(decoder)
    data = pl.read_parquet(historic_targets)
    values = np.column_stack([data[f"vix_{tenor}d"].to_numpy() for tenor in BUSINESS_DTES])
    log_values = np.log(values)
    if any(np.isfinite(column).sum() < 50 for column in log_values.T):
        raise ValueError("Each requested tenor needs at least 50 historic observations")
    dtype = torch.float64
    probabilities = torch.linspace(.005, .995, 101, dtype=dtype)
    target_quantiles = torch.tensor(np.nanquantile(log_values, probabilities.numpy(), axis=0), dtype=dtype)
    center = torch.tensor(np.nanmean(log_values, axis=0), dtype=dtype)
    scale = torch.tensor(np.nanstd(log_values, axis=0), dtype=dtype)
    dates = np.array(data["date"].to_list(), dtype="datetime64[D]")
    daily = count_business_days_between(dates[:-1], dates[1:]) == 1
    increments = np.diff(log_values, axis=0)[daily]
    increment_counts = np.isfinite(increments).sum(axis=0)
    increment_mask = torch.tensor(increment_counts >= 10)
    target_increment_variance = torch.tensor([
        np.nanvar(column) if count >= 10 else 1.
        for column, count in zip(increments.T, increment_counts, strict=True)
    ], dtype=dtype)
    target_correlation = np.empty((len(BUSINESS_DTES), len(BUSINESS_DTES)))
    for i in range(len(BUSINESS_DTES)):
        for j in range(len(BUSINESS_DTES)):
            mask = np.isfinite(log_values[:, i]) & np.isfinite(log_values[:, j])
            target_correlation[i, j] = np.corrcoef(log_values[mask, i], log_values[mask, j])[0, 1]
    target_correlation = torch.tensor(target_correlation, dtype=dtype)
    with np.load(gaussian_samples / "sampled_states.npz") as payload:
        stationary_mean = torch.tensor(payload["mean"], dtype=dtype)
        stationary_factor = torch.tensor(np.linalg.cholesky(payload["covariance"]), dtype=dtype)
        beta = torch.tensor(payload["spot_loading"], dtype=dtype)
        score_mean = float(payload["spot_score_mean"])
    with torch.no_grad():
        transition = decoder.weekly_transition().detach()
        mean = decoder.equilibrium_mean.detach()
        factor = decoder.innovation_cholesky().detach()
    point = stationary_mean[None].clone().requires_grad_()
    log_indices = torch.log(decoded_variance_indices(decoder, point))
    jacobian = torch.stack([torch.autograd.grad(log_indices[0, j], point, retain_graph=True)[0][0]
                            for j in range(len(BUSINESS_DTES))])
    _, singular, right = torch.linalg.svd((jacobian / scale[:, None]) @ factor, full_matrices=False)
    white_basis = right.T.detach()
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    identity = {"decoder_sha256": source_sha, "target_sha256": checkpoint_digest(historic_targets),
                "historical_rows": len(data), "first_date": str(data["date"][0]), "last_date": str(data["date"][-1]),
                "definition": "analytic log-contract variance, business-time annualization; not Cboe discrete VIX",
                "not_forward_validation": True, "frozen_decoder_and_ar": True,
                "one_session_increment_counts": {str(tenor): int(count) for tenor, count in zip(BUSINESS_DTES, increment_counts, strict=True)},
                "white_basis_singular_values": singular.detach().numpy().tolist()}
    write_json_atomic({"source": identity, "train_config": asdict(train)}, output / "manifest.json")

    def feature(log_vix: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
        return torch.cat(((log_vix - center) / scale, spot[:, None]), dim=-1)

    def simulate(model: VixSequenceModel | None, *, paths: int, burn_in: int, seed: int) -> tuple[np.ndarray, dict]:
        generator = torch.Generator().manual_seed(seed)
        h = stationary_mean + torch.randn(paths, len(mean), generator=generator, dtype=dtype) @ stationary_factor.T
        with torch.inference_mode():
            log_vix = torch.log(decoded_variance_indices(decoder, h))
            context = feature(log_vix, torch.zeros(paths, dtype=dtype))[:, None].expand(-1, 20, -1).clone()
            started = perf_counter()
            for step in range(burn_in if model is not None else 0):
                z = torch.randn(paths, generator=generator, dtype=dtype)
                epsilon = torch.randn(paths, len(mean), generator=generator, dtype=dtype)
                white, _ = model(context, z, epsilon)
                h = mean + (h - mean) @ transition + (z - score_mean)[:, None] * beta + white @ factor.T
                log_vix = torch.log(decoded_variance_indices(decoder, h))
                context = torch.cat((context[:, 1:], feature(log_vix, z)[:, None]), dim=1)
                if step == 0 or (step + 1) % 25 == 0:
                    elapsed = perf_counter() - started
                    print(f"audit={model.config.architecture} step={step + 1}/{burn_in} elapsed={elapsed:.1f}s "
                          f"remaining={elapsed * (burn_in - step - 1) / (step + 1):.1f}s", flush=True)
            raw = decoder.raw_states_from_states(h)[:, :decoder.config.geometry.num_inner_strikes * decoder.config.geometry.num_model_expiries]
            cap_fraction = float((raw >= decoder.config.geometry.maximum_log_dlvsq).double().mean())
        return torch.exp(log_vix).numpy(), {"dlv_fraction_at_cap": cap_fraction}

    def metrics(samples: np.ndarray) -> dict:
        tensor = torch.tensor(np.log(samples), dtype=dtype)
        return {"quantile_loss": float(quantile_distribution_loss(tensor, probabilities=probabilities,
                 target_quantiles=target_quantiles, target_scale=scale)),
                "tenors": {str(tenor): {"historic_n": int(np.isfinite(values[:, j]).sum()),
                           "historic_mean": float(np.nanmean(values[:, j])), "model_mean": float(np.mean(samples[:, j])),
                           **{f"historic_p{p}": float(np.nanpercentile(values[:, j], p)) for p in (5, 50, 95, 99)},
                           **{f"model_p{p}": float(np.percentile(samples[:, j], p)) for p in (5, 50, 95, 99)}}
                           for j, tenor in enumerate(BUSINESS_DTES)}}

    baseline, baseline_cap = simulate(None, paths=train.audit_paths, burn_in=0, seed=train.seed + 1000)
    baseline_metrics = {**metrics(baseline), **baseline_cap}
    np.savez_compressed(output / "gaussian_audit.npz", indices=baseline)
    write_json_atomic(baseline_metrics, output / "gaussian_metrics.json")
    reports = {}
    for architecture in ("lstm", "selective_ssm"):
        torch.manual_seed(train.seed)
        config = VixSequenceConfig(label=f"vix-{architecture}", architecture=architecture,
                                   input_size=len(BUSINESS_DTES) + 1, factor_rank=white_basis.shape[1])
        model = VixSequenceModel(env, config=config, white_basis=white_basis)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train.epochs,
                                                             eta_min=train.learning_rate * .1)
        generator = torch.Generator().manual_seed(train.seed + 1)
        h = stationary_mean + torch.randn(train.chains, len(mean), generator=generator, dtype=dtype) @ stationary_factor.T
        with torch.no_grad():
            log_vix = torch.log(decoded_variance_indices(decoder, h))
            context = feature(log_vix, torch.zeros(train.chains, dtype=dtype))[:, None].expand(-1, config.sequence_length, -1).clone()
        model_directory = output / architecture
        model_directory.mkdir(exist_ok=True)
        history = []
        start_epoch = 0
        last_path = model_directory / "last.pt"
        with FileLock(str(last_path.with_suffix(".lock"))):
            if last_path.exists():
                saved = torch.load(last_path, weights_only=False, map_location="cpu")
                same_inputs = all(saved["source"][key] == identity[key] for key in ("decoder_sha256", "target_sha256", "definition"))
                if saved["model_version"] != config.version.unique_id48 or saved["model_config"] != asdict(config) or not same_inputs:
                    raise ValueError("VIX sequence checkpoint identity has changed")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                generator.set_state(saved["simulation_rng"])
                torch.set_rng_state(saved["torch_rng"])
                h, log_vix, context = saved["h"], saved["log_vix"], saved["context"]
                history, start_epoch = saved["history"], saved["epoch"]
                if train.epochs < start_epoch:
                    raise ValueError("Requested epoch cap precedes the saved checkpoint")
                previous_train = saved["train_config"]
                if previous_train != asdict(train):
                    print("Changed TrainConfig: retaining network weights and optimizer moments.", flush=True)
                if (previous_train["epochs"], previous_train["learning_rate"]) != (train.epochs, train.learning_rate):
                    scheduler.T_max = train.epochs
                    scheduler.eta_min = train.learning_rate * .1
                    scheduler.base_lrs = [train.learning_rate for _ in optimizer.param_groups]
                    rate = scheduler.eta_min + .5 * (train.learning_rate - scheduler.eta_min) * (1 + math.cos(math.pi * start_epoch / train.epochs))
                    for group in optimizer.param_groups:
                        group["lr"], group["initial_lr"] = rate, train.learning_rate
                    scheduler._last_lr = [rate for _ in optimizer.param_groups]
                if previous_train["seed"] != train.seed:
                    generator.manual_seed(train.seed + 1)
                    torch.manual_seed(train.seed)
                if h.shape[0] != train.chains:
                    print("Changed chain count: resampling simulation chains, not model weights.", flush=True)
                    with torch.no_grad():
                        h = stationary_mean + torch.randn(train.chains, len(mean), generator=generator, dtype=dtype) @ stationary_factor.T
                        log_vix = torch.log(decoded_variance_indices(decoder, h))
                        context = feature(log_vix, torch.zeros(train.chains, dtype=dtype))[:, None].expand(-1, config.sequence_length, -1).clone()
            started = perf_counter()
            for epoch in range(start_epoch + 1, train.epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                h, log_vix, context = h.detach(), log_vix.detach(), context.detach()
                batch, changes, penalties = [], [], []
                for _ in range(train.steps_per_update):
                    z = torch.randn(train.chains, generator=generator, dtype=dtype)
                    epsilon = torch.randn(train.chains, len(mean), generator=generator, dtype=dtype)
                    white, regularizer = model(context, z, epsilon)
                    h = mean + (h - mean) @ transition + (z - score_mean)[:, None] * beta + white @ factor.T
                    next_log_vix = torch.log(decoded_variance_indices(decoder, h))
                    batch.append(next_log_vix)
                    changes.append(next_log_vix - log_vix)
                    penalties.append(regularizer)
                    log_vix = next_log_vix
                    context = torch.cat((context[:, 1:], feature(log_vix, z)[:, None]), dim=1)
                generated, increments_batch = torch.cat(batch), torch.cat(changes)
                marginal_loss = quantile_distribution_loss(generated, probabilities=probabilities,
                                                           target_quantiles=target_quantiles, target_scale=scale)
                increment_loss = torch.mean((torch.sqrt(increments_batch.var(dim=0, unbiased=False)[increment_mask] / target_increment_variance[increment_mask]) - 1).square())
                correlation_loss = torch.mean((_correlation(generated) - target_correlation).square())
                loss = (marginal_loss + config.increment_weight * increment_loss
                        + config.correlation_weight * correlation_loss + config.control_penalty * torch.stack(penalties).mean())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite {architecture} loss at epoch {epoch}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                scheduler.step()
                history.append({"epoch": epoch, "loss": float(loss.detach()), "quantile_loss": float(marginal_loss.detach()),
                                "increment_loss": float(increment_loss.detach()), "correlation_loss": float(correlation_loss.detach()),
                                "gradient_norm_before_clip": float(gradient_norm), "elapsed_seconds": perf_counter() - started})
                if epoch == 1 or epoch % 10 == 0 or epoch == train.epochs:
                    elapsed = perf_counter() - started
                    eta = elapsed * (train.epochs - epoch) / (epoch - start_epoch)
                    print(f"vix {architecture} epoch={epoch}/{train.epochs} loss={history[-1]['loss']:.5f} "
                          f"quantile={history[-1]['quantile_loss']:.5f} elapsed={elapsed:.1f}s remaining={eta:.1f}s", flush=True)
                    _save({"schema": "dysanos-v11-vix-sequence-v1", "model_config": asdict(config),
                           "model_version": config.version.unique_id48, "source": identity, "train_config": asdict(train),
                           "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "simulation_rng": generator.get_state(),
                           "torch_rng": torch.get_rng_state(), "h": h.detach(), "log_vix": log_vix.detach(),
                           "context": context.detach(), "history": history}, last_path)
                    write_json_atomic({"architecture": architecture, "history": history}, model_directory / "progress.json")
            sampled, cap = simulate(model.eval(), paths=train.audit_paths, burn_in=train.audit_burn_in, seed=train.seed + 1000)
            report = {"status": "complete", "description": "LSTM" if architecture == "lstm" else "Mamba-style selective diagonal SSM; not Mamba-3",
                      "parameters": sum(p.numel() for p in model.parameters()), "epochs": train.epochs,
                      "audit_paths": train.audit_paths, "audit_burn_in": train.audit_burn_in,
                      **metrics(sampled), **cap}
            np.savez_compressed(model_directory / "audit_indices.npz", indices=sampled)
            write_json_atomic(report, model_directory / "metrics.json")
            reports[architecture] = report
    report = {"status": "complete", "source": identity, "baseline": baseline_metrics, "models": reports}
    write_json_atomic(report, output / "comparison.json")
    plot_run(output=output, historic_targets=historic_targets)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--historic-targets", type=Path, required=True)
    parser.add_argument("--gaussian-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--steps-per-update", type=int, default=4)
    parser.add_argument("--audit-paths", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    arguments = vars(parser.parse_args())
    torch.set_num_threads(arguments.pop("threads"))
    train = VixSequenceTrainConfig(**{k: arguments.pop(k) for k in ("epochs", "chains", "steps_per_update", "audit_paths")})
    print(json.dumps(run(**arguments, train=train), indent=2), flush=True)


if __name__ == "__main__":
    main()
