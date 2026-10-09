"""Restricted-universe no-arbitrage diagnostics for Tangent Levy v2."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

from .v2_config import TangentLevyV2AuditConfig


@dataclass(frozen=True)
class CoordinateAudit:
    """Pair-based unconditional zero-mean screen."""

    tested_coordinates: int
    maximum_absolute_z: float
    p99_absolute_z: float
    count_above_1_96: int
    count_above_2_576: int
    count_above_5: int


@dataclass(frozen=True)
class ConditionalAudit:
    """Two-fold out-of-fold conditional-strategy statistic."""

    objective: str
    path_count: int
    pair_count: int
    mean_path_score: float
    pair_standard_error: float
    t_statistic: float
    confidence_interval_95: tuple[float, float]


@dataclass(frozen=True)
class RestrictedPopulation:
    """Simulated predictable states and next-step fixed-contract returns."""

    features: np.ndarray
    returns: np.ndarray
    active_mask: np.ndarray
    pair_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        returns = np.asarray(self.returns)
        active = np.asarray(self.active_mask)
        verify_inp(
            features.ndim == 3
            and returns.ndim == 3
            and features.shape[:2] == returns.shape[:2]
            and active.shape in {returns.shape, returns.shape[1:]}
            and features.shape[0] >= 4
            and features.shape[0] % 4 == 0,
            lambda: (
                "population must contain pair-preserving path folds with matching "
                f"features/returns/masks; got {features.shape}, {returns.shape}, "
                f"{active.shape}"
            ),
        )
        verify_inp(
            np.all(np.isfinite(features))
            and np.all(np.isfinite(returns)),
            lambda: "population arrays must be finite",
        )
        if self.pair_ids is not None:
            pair_ids = np.asarray(self.pair_ids, dtype=np.int64)
            expected = np.repeat(np.arange(features.shape[0] // 2), 2)
            verify_inp(
                np.array_equal(pair_ids, expected),
                lambda: "pair IDs must prove adjacent, unique antithetic pairs",
            )


def unconditional_coordinate_audit(
    returns: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
    minimum_observations: int = 20,
) -> CoordinateAudit:
    """Screen every day/instrument mean after averaging antithetic pairs."""

    values = np.asarray(returns, dtype=np.float64)
    verify_inp(
        values.ndim == 3 and values.shape[0] >= 4 and values.shape[0] % 2 == 0,
        lambda: "returns must have an even path axis and shape (P,H,I)",
    )
    mask = (
        np.ones_like(values, dtype=bool)
        if active_mask is None
        else np.broadcast_to(np.asarray(active_mask, dtype=bool), values.shape)
    )
    verify_inp(
        np.all(np.isfinite(values)),
        lambda: "unconditional returns must be finite",
    )
    pair_values = values.reshape(-1, 2, *values.shape[1:]).mean(axis=1)
    pair_mask = mask.reshape(-1, 2, *mask.shape[1:]).all(axis=1)
    counts = np.sum(pair_mask, axis=0)
    sums = np.sum(np.where(pair_mask, pair_values, 0.0), axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(pair_mask, pair_values - means, 0.0)
    variances = np.divide(
        np.sum(np.square(centered), axis=0),
        counts - 1,
        out=np.zeros_like(means),
        where=counts > 1,
    )
    standard_error = np.sqrt(
        np.divide(variances, counts, out=np.zeros_like(means), where=counts > 0)
    )
    tested = (
        (counts >= minimum_observations)
        & (standard_error > 0.0)
        & np.isfinite(standard_error)
    )
    z = np.divide(means, standard_error, out=np.zeros_like(means), where=tested)
    absolute = np.abs(z[tested])
    return CoordinateAudit(
        tested_coordinates=int(absolute.size),
        maximum_absolute_z=float(np.max(absolute, initial=0.0)),
        p99_absolute_z=float(np.quantile(absolute, 0.99)) if absolute.size else 0.0,
        count_above_1_96=int(np.count_nonzero(absolute > 1.96)),
        count_above_2_576=int(np.count_nonzero(absolute > 2.576)),
        count_above_5=int(np.count_nonzero(absolute > 5.0)),
    )


@version("2.0")
@dataclass(frozen=True, kw_only=True)
class ConditionalPolicyConfig:
    """Geometry and version identity of one conditional audit policy."""

    feature_count: int
    instrument_count: int
    network: FeedForwardConfig


class ConditionalPolicy(BaseModel):
    """Uncapped affine-plus-deep predictable action."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ConditionalPolicyConfig,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.affine = torch.nn.Linear(
            config.feature_count,
            config.instrument_count,
            dtype=env.fdtype,
            device=env.device,
        )
        self.deep = FeedForward(
            env,
            num_input=config.feature_count,
            num_output=config.instrument_count,
            config=config.network,
        )
        final = self.deep.I0 if len(self.deep.In) == 0 else self.deep.In[-1]
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias.zero_()
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        action = self.affine(normalized) + self.deep(normalized)
        mask = active_mask if active_mask.ndim == 3 else active_mask[None]
        return torch.where(mask, action, torch.zeros_like(action))


def _loss(
    policy: ConditionalPolicy,
    features: torch.Tensor,
    returns: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    objective: str,
    lambda_second_moment: float,
) -> torch.Tensor:
    action = policy(features, active_mask)
    portfolio_return = torch.sum(action * returns, dim=-1)
    action_energy = torch.sum(action.square(), dim=-1)
    if objective == "squared":
        reward = (
            portfolio_return
            - 0.5 * action_energy
            - 0.5 * lambda_second_moment * portfolio_return.square()
        )
        return -torch.mean(reward)
    if objective == "entropy":
        exponent = -portfolio_return + 0.5 * action_energy
        flat = exponent.double().reshape(-1)
        return (torch.logsumexp(flat, dim=0) - math.log(flat.numel())).to(
            dtype=features.dtype
        )
    raise ValueError(f"unknown conditional objective {objective!r}")


def _entropy_exponent(
    policy: ConditionalPolicy,
    features: torch.Tensor,
    returns: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    action = policy(features, active_mask)
    portfolio_return = torch.sum(action * returns, dim=-1)
    action_energy = torch.sum(action.square(), dim=-1)
    return -portfolio_return + 0.5 * action_energy


def _fit_fold(
    population: RestrictedPopulation,
    indices: np.ndarray,
    *,
    objective: str,
    config: TangentLevyV2AuditConfig,
    checkpoint_path: Path,
    device: str,
) -> ConditionalPolicy:
    feature_values = population.features[indices]
    flat = feature_values.reshape(-1, feature_values.shape[-1])
    feature_mean_np = np.mean(flat, axis=0, dtype=np.float64)
    feature_scale_np = np.std(flat, axis=0, ddof=0, dtype=np.float64)
    feature_scale_np = np.maximum(feature_scale_np, 1.0e-8)
    env = BaseEnvironment(precision=32, device=device, debug=False)
    policy_config = ConditionalPolicyConfig(
        feature_count=population.features.shape[-1],
        instrument_count=population.returns.shape[-1],
        network=config.policy_network,
    )
    policy = ConditionalPolicy(
        env,
        config=policy_config,
        feature_mean=torch.as_tensor(feature_mean_np, dtype=env.fdtype, device=env.device),
        feature_scale=torch.as_tensor(feature_scale_np, dtype=env.fdtype, device=env.device),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    fold_identity = hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()
    start_epoch = 0
    history: list[float] = []
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        if payload.get("schema") != "tangent-levy-v2-conditional-policy":
            raise ValueError(f"incompatible checkpoint {checkpoint_path}")
        expected_identity = {
            "policy_version": ConditionalPolicyConfig.version.unique_id48,
            "audit_config": asdict(config),
            "objective": objective,
            "fold_identity": fold_identity,
            "feature_shape": tuple(population.features.shape),
            "return_shape": tuple(population.returns.shape),
        }
        if any(payload.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"stale or incompatible checkpoint {checkpoint_path}")
        if not np.array_equal(
            np.asarray(payload["feature_mean"]), feature_mean_np
        ) or not np.array_equal(
            np.asarray(payload["feature_scale"]), feature_scale_np
        ):
            raise ValueError(f"checkpoint normalization mismatch {checkpoint_path}")
        policy.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["generator_state"].cpu())
        start_epoch = int(payload["epoch"])
        history = [float(value) for value in payload["history"]]
    started = time.perf_counter()
    for epoch in range(start_epoch, config.policy_epochs):
        order = indices[
            torch.randperm(len(indices), generator=generator).numpy()
        ]
        if objective == "entropy":
            policy.train()
            maximum = -math.inf
            with torch.no_grad():
                for start in range(0, len(order), config.policy_batch_size):
                    selected = order[start : start + config.policy_batch_size]
                    features = torch.as_tensor(
                        population.features[selected],
                        dtype=env.fdtype,
                        device=env.device,
                    )
                    returns = torch.as_tensor(
                        population.returns[selected],
                        dtype=env.fdtype,
                        device=env.device,
                    )
                    mask_values = (
                        population.active_mask[selected]
                        if population.active_mask.ndim == 3
                        else population.active_mask
                    )
                    active = torch.as_tensor(
                        mask_values, dtype=torch.bool, device=env.device
                    )
                    exponent = _entropy_exponent(
                        policy, features, returns, active
                    )
                    maximum = max(maximum, float(torch.max(exponent).cpu()))
            denominator = 0.0
            with torch.no_grad():
                for start in range(0, len(order), config.policy_batch_size):
                    selected = order[start : start + config.policy_batch_size]
                    features = torch.as_tensor(
                        population.features[selected],
                        dtype=env.fdtype,
                        device=env.device,
                    )
                    returns = torch.as_tensor(
                        population.returns[selected],
                        dtype=env.fdtype,
                        device=env.device,
                    )
                    mask_values = (
                        population.active_mask[selected]
                        if population.active_mask.ndim == 3
                        else population.active_mask
                    )
                    active = torch.as_tensor(
                        mask_values, dtype=torch.bool, device=env.device
                    )
                    exponent = _entropy_exponent(
                        policy, features, returns, active
                    )
                    denominator += float(torch.sum(torch.exp(exponent - maximum)).cpu())
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(order), config.policy_batch_size):
                selected = order[start : start + config.policy_batch_size]
                features = torch.as_tensor(
                    population.features[selected], dtype=env.fdtype, device=env.device
                )
                returns = torch.as_tensor(
                    population.returns[selected], dtype=env.fdtype, device=env.device
                )
                mask_values = (
                    population.active_mask[selected]
                    if population.active_mask.ndim == 3
                    else population.active_mask
                )
                active = torch.as_tensor(
                    mask_values, dtype=torch.bool, device=env.device
                )
                exponent = _entropy_exponent(policy, features, returns, active)
                (torch.sum(torch.exp(exponent - maximum)) / denominator).backward()
            optimizer.step()
            observation_count = len(order) * population.features.shape[1]
            history.append(maximum + math.log(denominator / observation_count))
            epoch_loss = history[-1]
            epoch_paths = 1
        else:
            epoch_loss = 0.0
            epoch_paths = 0
            policy.train()
            for start in range(0, len(order), config.policy_batch_size):
                selected = order[start : start + config.policy_batch_size]
                features = torch.as_tensor(
                    population.features[selected], dtype=env.fdtype, device=env.device
                )
                returns = torch.as_tensor(
                    population.returns[selected], dtype=env.fdtype, device=env.device
                )
                mask_values = (
                    population.active_mask[selected]
                    if population.active_mask.ndim == 3
                    else population.active_mask
                )
                active = torch.as_tensor(
                    mask_values, dtype=torch.bool, device=env.device
                )
                optimizer.zero_grad(set_to_none=True)
                value = _loss(
                    policy,
                    features,
                    returns,
                    active,
                    objective=objective,
                    lambda_second_moment=config.lambda_second_moment,
                )
                value.backward()
                optimizer.step()
                epoch_loss += float(value.detach().cpu()) * len(selected)
                epoch_paths += len(selected)
            history.append(epoch_loss / epoch_paths)
        if (epoch + 1) % 10 == 0 or epoch + 1 == config.policy_epochs:
            elapsed = time.perf_counter() - started
            rate = (epoch + 1 - start_epoch) / max(elapsed, 1.0e-12)
            remaining = (config.policy_epochs - epoch - 1) / max(rate, 1.0e-12)
            print(
                f"conditional {objective} epoch={epoch + 1}/{config.policy_epochs} "
                f"loss={history[-1]:.6g} elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "schema": "tangent-levy-v2-conditional-policy",
                    "policy_version": ConditionalPolicyConfig.version.unique_id48,
                    "audit_config": asdict(config),
                    "objective": objective,
                    "fold_identity": fold_identity,
                    "feature_shape": tuple(population.features.shape),
                    "return_shape": tuple(population.returns.shape),
                    "feature_mean": feature_mean_np,
                    "feature_scale": feature_scale_np,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                    "epoch": epoch + 1,
                    "history": history,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
    return policy


def _path_scores(
    policy: ConditionalPolicy,
    population: RestrictedPopulation,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.float64)
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            features = torch.as_tensor(
                population.features[selected],
                dtype=policy.fdtype,
                device=policy.device,
            )
            returns = torch.as_tensor(
                population.returns[selected],
                dtype=policy.fdtype,
                device=policy.device,
            )
            mask_values = (
                population.active_mask[selected]
                if population.active_mask.ndim == 3
                else population.active_mask
            )
            active = torch.as_tensor(
                mask_values, dtype=torch.bool, device=policy.device
            )
            action = policy(features, active)
            scores = torch.mean(torch.sum(action * returns, dim=-1), dim=-1)
            result[start : start + len(selected)] = scores.double().cpu().numpy()
    return result


def crossfit_conditional_audit(
    population: RestrictedPopulation,
    *,
    objective: str,
    config: TangentLevyV2AuditConfig,
    checkpoint_directory: Path,
    device: str = "cuda:0",
) -> ConditionalAudit:
    """Fit pair-preserving two-fold policies and evaluate opposite folds once."""

    verify_inp(
        objective in {"squared", "entropy"},
        lambda: f"unknown objective {objective!r}",
    )
    path_count = population.features.shape[0]
    verify_inp(
        population.pair_ids is not None,
        lambda: "conditional audit requires explicit pair identity metadata",
    )
    verify_inp(
        path_count == config.path_count,
        lambda: f"population has {path_count} paths, expected {config.path_count}",
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    split = path_count // 2
    folds = (np.arange(0, split), np.arange(split, path_count))
    out_of_fold = np.empty(path_count, dtype=np.float64)
    with FileLock(checkpoint_directory / f"{objective}.lock"):
        for train_fold, heldout_fold in ((0, 1), (1, 0)):
            policy = _fit_fold(
                population,
                folds[train_fold],
                objective=objective,
                config=config,
                checkpoint_path=checkpoint_directory
                / f"{objective}-fold-{train_fold}.pt",
                device=device,
            )
            out_of_fold[folds[heldout_fold]] = _path_scores(
                policy,
                population,
                folds[heldout_fold],
                batch_size=config.policy_batch_size,
            )
    pair_scores = out_of_fold.reshape(-1, 2).mean(axis=1)
    estimate = float(np.mean(pair_scores, dtype=np.float64))
    standard_error = float(
        np.std(pair_scores, ddof=1, dtype=np.float64) / math.sqrt(len(pair_scores))
    )
    statistic = estimate / standard_error if standard_error > 0.0 else 0.0
    return ConditionalAudit(
        objective=objective,
        path_count=path_count,
        pair_count=len(pair_scores),
        mean_path_score=estimate,
        pair_standard_error=standard_error,
        t_statistic=statistic,
        confidence_interval_95=(
            estimate - 1.959963984540054 * standard_error,
            estimate + 1.959963984540054 * standard_error,
        ),
    )
