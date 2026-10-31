"""Training and evaluation for the native and compressed FLEX states."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeAlias, cast

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FixedEtaFlexGeometryConfig,
    FlexGeometryConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    ForwardMuFlexGeometryConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    LatentDecoderConfig,
    NativeStateConfig,
    PcaDecoderConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.metrics import (
    ExpiryExcessStatistics,
    SurfaceFitStatistics,
    expiry_excess_statistics,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    JointHierarchicalDynamicsModel,
    LatentStateSanosModel,
    NativeParameters,
    NativeStateSanosModel,
    dlvsq_cap_penalty,
    forward_vols_from_total_vols,
    raw_from_native_parameters,
    split_forward_mu_native_raw,
)
from quantdata.torch import BaseEnvironment

FlexTorchModel: TypeAlias = (
    NativeStateSanosModel
    | LatentStateSanosModel
    | JointDirectDynamicsModel
    | JointHierarchicalDynamicsModel
)
TrainingState: TypeAlias = dict[str, object]
_TRAINING_ALGORITHM_VERSION = 9


class TrainingConvergenceError(RuntimeError):
    """A training phase ended without a usable converged candidate."""


def _seed_model_initialization(*, seed: int, device: str) -> None:
    """Seed Torch model initialization on CPU and the selected CUDA device."""
    torch.random.manual_seed(seed)
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed(seed)


def native_target_identity(
    model: NativeStateSanosModel,
    *,
    market_fingerprints: tuple[str, ...],
) -> dict[str, object]:
    """Bind latent supervision to one exact ordered native-state artifact."""
    if model.num_days != len(market_fingerprints):
        raise ValueError("Native target and market-day counts do not match")
    raw = model.daily_raw_states.detach().cpu().contiguous()
    raw_array = raw.numpy()
    return {
        "model_identity": _model_checkpoint_identity(model),
        "market_fingerprints": list(market_fingerprints),
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
        "raw_state_sha256": hashlib.sha256(raw_array.tobytes()).hexdigest(),
    }


def native_checkpoint_proof(path: Path) -> dict[str, object]:
    """Return immutable provenance for one native checkpoint file."""
    file_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            file_digest.update(chunk)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    raw = model_state["daily_raw_embedding.weight"].detach().contiguous()
    fingerprint_digest = hashlib.sha256()
    fingerprints = cast(list[str], payload["market_fingerprints"])
    for fingerprint in fingerprints:
        encoded = fingerprint.encode("ascii")
        fingerprint_digest.update(len(encoded).to_bytes(8, "little"))
        fingerprint_digest.update(encoded)
    return {
        "checkpoint_sha256": file_digest.hexdigest(),
        "schema_version": int(payload["schema_version"]),
        "phase": payload.get("phase"),
        "model_identity": json.loads(
            json.dumps(payload.get("model_identity"))
        ),
        "num_market_days": len(fingerprints),
        "market_fingerprints_sha256": fingerprint_digest.hexdigest(),
        "native_state_dtype": str(raw.dtype),
        "native_state_shape": list(raw.shape),
        "native_state_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def latent_checkpoint_proof(path: Path) -> dict[str, object]:
    """Return immutable provenance for one completed latent checkpoint."""
    file_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            file_digest.update(chunk)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError(f"Latent FLEX checkpoint is not complete in {path}")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or not all(
        isinstance(audits.get(name), dict)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError(
            f"Latent FLEX checkpoint has no successful convergence proof in {path}"
        )
    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, dict) or model_identity.get(
        "model_class"
    ) != LatentStateSanosModel.__qualname__:
        raise ValueError(f"Invalid latent FLEX model identity in {path}")
    model_state = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    if not all(torch.isfinite(value).all().item() for value in model_state.values()):
        raise ValueError(f"Latent FLEX checkpoint contains non-finite tensors in {path}")
    latent = model_state["daily_state_embedding.weight"].detach().contiguous()
    fingerprints = cast(list[str], payload["market_fingerprints"])
    model_config = cast(dict[str, object], model_identity["model_config"])
    expected_shape = (len(fingerprints), int(model_config["state_size"]))
    if tuple(latent.shape) != expected_shape:
        raise ValueError(
            f"Latent checkpoint state shape {tuple(latent.shape)} does not match "
            f"its identity {expected_shape} in {path}"
        )
    supervision = cast(
        dict[str, object],
        payload["supervision_target_identity"],
    )
    supervision_fingerprints = cast(
        list[str],
        supervision["market_fingerprints"],
    )
    if fingerprints != supervision_fingerprints:
        raise ValueError(
            f"Latent checkpoint and native target fingerprints differ in {path}"
        )
    native_shape = cast(list[int], supervision["shape"])
    if len(native_shape) != 2 or native_shape[0] != len(fingerprints):
        raise ValueError(
            f"Latent checkpoint native target shape {native_shape} does not "
            f"match {len(fingerprints)} market days in {path}"
        )
    fingerprint_digest = hashlib.sha256()
    for fingerprint in fingerprints:
        encoded = fingerprint.encode("ascii")
        fingerprint_digest.update(len(encoded).to_bytes(8, "little"))
        fingerprint_digest.update(encoded)
    return {
        "checkpoint_sha256": file_digest.hexdigest(),
        "schema_version": int(payload["schema_version"]),
        "phase": payload["phase"],
        "model_identity": json.loads(json.dumps(model_identity)),
        "num_market_days": len(fingerprints),
        "market_fingerprints_sha256": fingerprint_digest.hexdigest(),
        "latent_state_dtype": str(latent.dtype),
        "latent_state_shape": list(latent.shape),
        "latent_state_sha256": hashlib.sha256(latent.numpy().tobytes()).hexdigest(),
        "native_state_dtype": supervision["dtype"],
        "native_state_shape": supervision["shape"],
        "native_state_sha256": supervision["raw_state_sha256"],
        "training_audits": json.loads(json.dumps(audits)),
    }


def native_checkpoint_state_audit(
    path: Path,
    *,
    geometry: FlexGeometryConfig,
    boundary_probability: float = 1.0e-4,
) -> dict[str, object]:
    """Summarize bounded native coordinates without repricing any market."""
    if not 0.0 < boundary_probability < 0.5:
        raise ValueError(
            "'boundary_probability' must lie in (0,0.5), found "
            f"{boundary_probability}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    raw = model_state["daily_raw_embedding.weight"].detach().contiguous()
    if raw.ndim != 2 or raw.shape[1] != geometry.native_state_size:
        raise ValueError(
            "Native checkpoint state shape does not match geometry: "
            f"{tuple(raw.shape)} versus (*,{geometry.native_state_size})"
        )
    if not bool(torch.all(torch.isfinite(raw))):
        raise ValueError(f"Native checkpoint contains non-finite states in {path}")
    probability = torch.sigmoid(raw)
    if isinstance(geometry, ForwardMuFlexGeometryConfig):
        dlv_raw, forward_raw, eta_raw = split_forward_mu_native_raw(
            raw,
            geometry=geometry,
        )
        dlv_probability = torch.sigmoid(dlv_raw)
        forward_probability = torch.sigmoid(forward_raw)
        eta_probability = torch.sigmoid(eta_raw)
        eta_names = (
            ("front_eta", "term_eta")
            if isinstance(
                geometry,
                FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
            )
            else ("forward_mu",)
        )
    else:
        dlv_stop = geometry.num_model_expiries * geometry.num_inner_strikes
        dlv_probability = probability[:, :dlv_stop]
        forward_probability = probability[
            :,
            dlv_stop : dlv_stop + geometry.num_model_expiries,
        ]
        eta_probability = (
            torch.full_like(
                probability[:, -1:],
                (geometry.fixed_eta - geometry.min_eta)
                / (geometry.max_eta - geometry.min_eta),
            )
            if isinstance(geometry, FixedEtaFlexGeometryConfig)
            else probability[:, -1:]
        )
        eta_names = ("eta",)
    if isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        maximum_log_dlvsq = torch.as_tensor(
            geometry.maximum_log_dlvsq,
            dtype=raw.dtype,
            device=raw.device,
        )
        dlv_values = geometry.min_dlvsq + torch.exp(
            torch.minimum(dlv_raw, maximum_log_dlvsq)
        )
        dlv_probability = (dlv_values - geometry.min_dlvsq) / (
            geometry.max_dlvsq - geometry.min_dlvsq
        )
    else:
        dlv_values = geometry.min_dlvsq + (
            geometry.max_dlvsq - geometry.min_dlvsq
        ) * dlv_probability
    bounded: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (
            "dlvsq",
            dlv_values,
            dlv_probability,
        ),
        (
            "forward_vol",
            geometry.min_forward_vol
            + (geometry.max_forward_vol - geometry.min_forward_vol)
            * forward_probability,
            forward_probability,
        ),
    ]
    if len(eta_names) == 1:
        bounded.append(
            (
                eta_names[0],
                geometry.min_eta
                + (geometry.max_eta - geometry.min_eta) * eta_probability,
                eta_probability,
            )
        )
    else:
        for eta_index, eta_name in enumerate(eta_names):
            coordinate_probability = eta_probability[:, eta_index]
            bounded.append(
                (
                    eta_name,
                    geometry.min_eta
                    + (geometry.max_eta - geometry.min_eta)
                    * coordinate_probability,
                    coordinate_probability,
                )
            )
    coordinate_statistics: dict[str, dict[str, float]] = {}
    for name, values, probabilities in bounded:
        coordinate_statistics[name] = {
            "minimum": float(torch.min(values)),
            "median": float(torch.quantile(values, 0.5)),
            "maximum": float(torch.max(values)),
            "fraction_near_lower_bound": float(
                torch.mean((probabilities <= boundary_probability).float())
            ),
            "fraction_near_upper_bound": float(
                torch.mean(
                    (probabilities >= 1.0 - boundary_probability).float()
                )
            ),
        }
    return {
        "all_raw_states_finite": True,
        "num_market_days": int(raw.shape[0]),
        "native_state_size": int(raw.shape[1]),
        "boundary_probability": boundary_probability,
        "coordinates": coordinate_statistics,
    }


def model_checkpoint_phase(path: Path) -> str:
    """Read a checkpoint phase for restart orchestration without a model load."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        return "legacy_model"
    if schema_version != 2:
        raise ValueError(f"Unsupported FLEX checkpoint schema in {path}")
    phase = payload.get("phase")
    if phase not in {
        "supervised",
        "supervised_complete",
        "market",
        "complete",
    }:
        raise ValueError(f"Unsupported FLEX checkpoint phase {phase!r} in {path}")
    return str(phase)


@dataclass(frozen=True)
class TorchMarketDay:
    """One market day materialized once on a training device."""

    source: FlexMarketDay
    expiries_y: torch.Tensor
    strikes: torch.Tensor
    price_expiries_y: torch.Tensor
    price_strikes: torch.Tensor
    quote_price_index: torch.Tensor
    is_call: torch.Tensor
    bids: torch.Tensor
    asks: torch.Tensor
    fit_prices: torch.Tensor


@dataclass(frozen=True)
class TorchMarketBatch:
    """A padded batch of days with a mask for the real quote locations."""

    day_indices: torch.Tensor
    strikes: torch.Tensor
    price_expiries_y: torch.Tensor
    price_strikes: torch.Tensor
    quote_price_index: torch.Tensor
    is_call: torch.Tensor
    bids: torch.Tensor
    asks: torch.Tensor
    fit_prices: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class FitEvaluation:
    """Exact quote-level objective and requested containment statistics."""

    objective: float
    statistics: ExpiryExcessStatistics
    surface_fit: SurfaceFitStatistics
    fits: tuple[np.ndarray, ...]

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-safe evaluation fields."""
        return {
            "objective": self.objective,
            "statistics": self.statistics.to_dict(),
            "surface_fit": self.surface_fit.to_dict(),
            "num_days": len(self.fits),
        }


@dataclass(frozen=True)
class NativeStateFitResult:
    """Native-state fit with its reusable supervised compression target."""

    model: NativeStateSanosModel
    evaluation: FitEvaluation
    history: tuple[dict[str, float], ...]
    elapsed_seconds: float
    market_fingerprints: tuple[str, ...]
    state_dtype: str
    convergence: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return report fields without tensors."""
        return {
            "evaluation": self.evaluation.to_dict(),
            "history": list(self.history),
            "elapsed_seconds": self.elapsed_seconds,
            "num_market_fingerprints": len(self.market_fingerprints),
            "state_dtype": self.state_dtype,
            "convergence": self.convergence,
        }


@dataclass(frozen=True)
class LatentStateFitResult:
    """One lower-state decoder fit after supervised and market phases."""

    model: LatentStateSanosModel
    evaluation: FitEvaluation
    initial_evaluation: FitEvaluation | None
    supervised_history: tuple[dict[str, float], ...]
    market_history: tuple[dict[str, float], ...]
    elapsed_seconds: float
    supervised_convergence: dict[str, object] | None
    market_convergence: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return report fields without tensors."""
        return {
            "state_size": self.model.config.state_size,
            "evaluation": self.evaluation.to_dict(),
            "initial_evaluation": (
                None
                if self.initial_evaluation is None
                else self.initial_evaluation.to_dict()
            ),
            "supervised_history": list(self.supervised_history),
            "market_history": list(self.market_history),
            "elapsed_seconds": self.elapsed_seconds,
            "supervised_convergence": self.supervised_convergence,
            "market_convergence": self.market_convergence,
        }


def materialize_day(
    day: FlexMarketDay,
    *,
    env: BaseEnvironment,
) -> TorchMarketDay:
    """Move one immutable market objective to the selected device."""
    tensor = lambda values: torch.as_tensor(  # noqa: E731
        values,
        dtype=env.fdtype,
        device=env.device,
    )
    price_location_by_key: dict[tuple[int, float], int] = {}
    price_expiry_indices: list[int] = []
    price_strikes: list[float] = []
    quote_price_index = np.empty(day.num_quotes, dtype=np.int64)
    for quote_index, (expiry_index, strike) in enumerate(
        zip(day.expiry_index, day.strikes, strict=True)
    ):
        key = (int(expiry_index), float(strike))
        price_index = price_location_by_key.get(key)
        if price_index is None:
            price_index = len(price_strikes)
            price_location_by_key[key] = price_index
            price_expiry_indices.append(int(expiry_index))
            price_strikes.append(float(strike))
        quote_price_index[quote_index] = price_index
    return TorchMarketDay(
        source=day,
        expiries_y=tensor(day.expiries_y[day.expiry_index]),
        strikes=tensor(day.strikes),
        price_expiries_y=tensor(
            day.expiries_y[np.asarray(price_expiry_indices, dtype=np.int64)]
        ),
        price_strikes=tensor(np.asarray(price_strikes, dtype=np.float64)),
        quote_price_index=torch.as_tensor(
            quote_price_index,
            dtype=torch.long,
            device=env.device,
        ),
        is_call=torch.as_tensor(day.is_call, dtype=torch.bool, device=env.device),
        bids=tensor(day.bids),
        asks=tensor(day.asks),
        fit_prices=tensor(day.fit_prices),
    )


def materialize_batch(
    days: tuple[TorchMarketDay, ...],
    indices: np.ndarray,
    *,
    env: BaseEnvironment,
) -> TorchMarketBatch:
    """Pad selected days without changing any real-quote objective weight."""
    selected = [days[int(index)] for index in indices]
    max_quotes = max(len(day.strikes) for day in selected)
    max_price_locations = max(len(day.price_strikes) for day in selected)

    def pad(
        values: torch.Tensor,
        value: float | bool | int,
        length: int = max_quotes,
    ) -> torch.Tensor:
        return torch.nn.functional.pad(
            values,
            (0, length - len(values)),
            value=value,
        )

    mask = torch.zeros(
        (len(selected), max_quotes),
        dtype=torch.bool,
        device=env.device,
    )
    for row, day in enumerate(selected):
        mask[row, : len(day.strikes)] = True
    return TorchMarketBatch(
        day_indices=torch.as_tensor(indices, dtype=torch.long, device=env.device),
        strikes=torch.stack([pad(day.strikes, 1.0) for day in selected]),
        price_expiries_y=torch.stack(
            [
                pad(
                    day.price_expiries_y,
                    0.0,
                    max_price_locations,
                )
                for day in selected
            ]
        ),
        price_strikes=torch.stack(
            [
                pad(day.price_strikes, 1.0, max_price_locations)
                for day in selected
            ]
        ),
        quote_price_index=torch.stack(
            [pad(day.quote_price_index, 0) for day in selected]
        ),
        is_call=torch.stack([pad(day.is_call, True) for day in selected]),
        bids=torch.stack([pad(day.bids, 0.0) for day in selected]),
        asks=torch.stack([pad(day.asks, 1.0) for day in selected]),
        fit_prices=torch.stack([pad(day.fit_prices, 0.0) for day in selected]),
        mask=mask,
    )


def quote_objective(
    prices: torch.Tensor,
    batch: TorchMarketBatch,
    *,
    objective_config: FlexObjectiveConfig,
) -> torch.Tensor:
    """Apply the exact quote-weighted FLEX objective to Torch prices."""
    half_spreads = 0.5 * (batch.asks - batch.bids)
    safe_half_spreads = torch.where(
        batch.mask,
        half_spreads,
        torch.ones_like(half_spreads),
    )
    excess = (torch.relu(prices - batch.asks) + torch.relu(batch.bids - prices)) / safe_half_spreads
    central = torch.abs(prices - batch.fit_prices) / safe_half_spreads
    quote_loss = excess + objective_config.fit_weight * central
    return torch.sum(torch.where(batch.mask, quote_loss, 0.0)) / torch.sum(batch.mask)


def per_date_quote_objective(
    prices: torch.Tensor,
    batch: TorchMarketBatch,
    *,
    objective_config: FlexObjectiveConfig,
) -> torch.Tensor:
    """Return one mean quote loss per date, independent of quote count."""
    half_spreads = 0.5 * (batch.asks - batch.bids)
    safe_half_spreads = torch.where(
        batch.mask,
        half_spreads,
        torch.ones_like(half_spreads),
    )
    quote_loss = (
        torch.relu(prices - batch.asks)
        + torch.relu(batch.bids - prices)
        + objective_config.fit_weight * torch.abs(prices - batch.fit_prices)
    ) / safe_half_spreads
    valid = batch.mask.to(dtype=prices.dtype)
    quote_counts = torch.sum(valid, dim=1)
    if torch.any(quote_counts <= 0.0):
        raise ValueError("Every market date must contain at least one quote")
    return torch.sum(quote_loss * valid, dim=1) / quote_counts


def equal_date_quote_objective(
    prices: torch.Tensor,
    batch: TorchMarketBatch,
    *,
    objective_config: FlexObjectiveConfig,
) -> torch.Tensor:
    """Average within dates and then across dates with equal date weight."""
    return torch.mean(
        per_date_quote_objective(
            prices,
            batch,
            objective_config=objective_config,
        )
    )


def _model_prices(
    model: FlexTorchModel,
    batch: TorchMarketBatch,
) -> torch.Tensor:
    """Price each expiry/strike once, then restore separate call/put books."""
    result = model(
        day_indices=batch.day_indices,
        expiries_y=batch.price_expiries_y,
        strikes=batch.price_strikes,
        is_call=torch.ones_like(batch.price_strikes, dtype=torch.bool),
    )
    call_prices = torch.gather(
        result[0],
        dim=1,
        index=batch.quote_price_index,
    )
    return torch.where(
        batch.is_call,
        call_prices,
        call_prices - 1.0 + batch.strikes,
    )


def _model_raw_states(
    model: FlexTorchModel,
    day_indices: torch.Tensor,
) -> torch.Tensor:
    """Return native raw coordinates for a selected batch of market dates."""
    if isinstance(model, NativeStateSanosModel):
        return model.daily_raw_embedding(day_indices)
    return model.raw_states_for_days(day_indices)


def model_dlvsq_cap_penalty(
    model: FlexTorchModel,
    day_indices: torch.Tensor,
) -> torch.Tensor:
    """Return the mean pricing-cap barrier, or zero for legacy sigmoid DLVs."""
    geometry = model.config.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        return torch.zeros((), dtype=model.fdtype, device=model.device)
    return torch.mean(
        dlvsq_cap_penalty(
            _model_raw_states(model, day_indices),
            geometry=geometry,
        )
    )


def evaluate_model(
    model: FlexTorchModel,
    days: tuple[TorchMarketDay, ...],
    *,
    env: BaseEnvironment,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    equal_date_weighting: bool = False,
) -> FitEvaluation:
    """Reprice every quote and aggregate the requested objective convention."""
    fits: list[np.ndarray] = []
    total_quote_objective = 0.0
    total_quotes = 0
    total_date_objective = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(days), day_batch_size):
            indices = np.arange(start, min(start + day_batch_size, len(days)))
            batch = materialize_batch(days, indices, env=env)
            prices = _model_prices(model, batch)
            per_date = per_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            total_date_objective += float(torch.sum(per_date).cpu())
            half_spreads = 0.5 * (batch.asks - batch.bids)
            safe_half_spreads = torch.where(
                batch.mask,
                half_spreads,
                torch.ones_like(half_spreads),
            )
            quote_values = (
                torch.relu(prices - batch.asks)
                + torch.relu(batch.bids - prices)
                + objective_config.fit_weight
                * torch.abs(prices - batch.fit_prices)
            ) / safe_half_spreads
            total_quote_objective += float(
                torch.sum(torch.where(batch.mask, quote_values, 0.0)).cpu()
            )
            total_quotes += int(torch.sum(batch.mask).cpu())
            for row, day_index in enumerate(indices):
                count = days[int(day_index)].source.num_quotes
                fits.append(prices[row, :count].detach().cpu().numpy().astype(np.float64))

    fit_vector = np.concatenate(fits)
    bid_vector = np.concatenate([day.source.bids for day in days])
    ask_vector = np.concatenate([day.source.asks for day in days])
    market_fit_vector = np.concatenate(
        [day.source.fit_prices for day in days]
    )
    dte_vector = np.concatenate([day.source.quote_business_dtes for day in days])
    return FitEvaluation(
        objective=(
            total_date_objective / len(days)
            if equal_date_weighting
            else total_quote_objective / total_quotes
        ),
        statistics=expiry_excess_statistics(
            fit_vector,
            bid_vector,
            ask_vector,
            dte_vector,
        ),
        surface_fit=surface_fit_statistics(
            fit_vector,
            market_fit_vector,
            bid_vector,
            ask_vector,
        ),
        fits=tuple(fits),
    )


def market_native_raw_initial_states(
    market_days: tuple[FlexMarketDay, ...],
    *,
    env: BaseEnvironment,
    config: NativeStateConfig,
    initial_eta: float = 0.20,
) -> torch.Tensor:
    """Initialize direct states from the market ATM curve without an LP."""
    if len(market_days) == 0:
        raise ValueError("At least one market day is required")
    geometry = config.geometry
    total_vols = torch.as_tensor(
        np.stack([day.model_atm_total_vols for day in market_days]),
        dtype=env.fdtype,
        device=env.device,
    )
    forward_vols = forward_vols_from_total_vols(total_vols, geometry=geometry)
    forward_vols = torch.clamp(
        forward_vols,
        min=geometry.min_forward_vol,
        max=geometry.max_forward_vol,
    )
    model_expiries = torch.as_tensor(
        geometry.model_expiries_y,
        dtype=env.fdtype,
        device=env.device,
    )
    atm_annual_vols = total_vols / torch.sqrt(model_expiries)
    dlvsqs = torch.clamp(
        atm_annual_vols.square().unsqueeze(-1).expand(
            -1,
            -1,
            geometry.num_inner_strikes,
        ),
        min=geometry.min_dlvsq,
        max=geometry.max_dlvsq,
    )
    eta_shape = (
        (len(market_days), geometry.num_model_expiries)
        if isinstance(geometry, ForwardMuFlexGeometryConfig)
        else (len(market_days),)
    )
    eta = torch.full(
        eta_shape,
        initial_eta,
        dtype=env.fdtype,
        device=env.device,
    )
    return raw_from_native_parameters(
        NativeParameters(
            dlvsqs=dlvsqs,
            forward_vols=forward_vols,
            eta=eta,
        ),
        geometry=geometry,
    )


def _learning_rate_multiplier(epoch: int, config: FlexTrainConfig) -> float:
    """Return the LR multiplier used by zero-based training epoch ``epoch``.

    Warm-up reaches the configured peak on its last epoch. Cosine decay then
    starts from that peak and reaches the configured floor on the final epoch.
    Clamping prevents the post-final ``LambdaLR.step()`` from rebounding.
    """
    epoch = min(max(epoch, 0), config.num_epochs - 1)
    floor = config.minimum_learning_rate_fraction
    warmup_epochs = config.learning_rate_warmup_epochs
    if warmup_epochs > 0 and epoch < warmup_epochs:
        progress = (epoch + 1) / warmup_epochs
        return floor + (1.0 - floor) * progress
    if config.learning_rate_schedule == "constant":
        return 1.0
    decay_epochs = config.num_epochs - warmup_epochs
    decay_index = epoch - warmup_epochs
    progress = decay_index / max(decay_epochs - 1, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return floor + (1.0 - floor) * cosine


def _evaluation_score(evaluation: FitEvaluation) -> dict[str, float]:
    """Retain compact eager metrics for pre/post-training comparisons."""
    return {
        "objective": float(evaluation.objective),
        "percent_zero": float(evaluation.statistics.all.percent_zero),
    }


def _evaluation_score_is_valid(score: dict[str, float]) -> bool:
    """Recognize finite checkpoint-safe evaluation metrics."""
    return (
        set(score) == {"objective", "percent_zero"}
        and all(np.isfinite(value) for value in score.values())
        and score["objective"] >= 0.0
        and 0.0 <= score["percent_zero"] <= 100.0
    )


def training_convergence_audit(
    history: tuple[dict[str, float], ...] | list[dict[str, float]],
    *,
    metric: str,
    train_config: FlexTrainConfig,
) -> dict[str, int | float | bool | str | None]:
    """Test whether the trailing objective window has reached a stable plateau."""
    values = [float(item[metric]) for item in history]
    finite = all(np.isfinite(value) and value >= 0.0 for value in values)
    enough_history = len(values) > train_config.convergence_window
    if not finite or not enough_history:
        return {
            "metric": metric,
            "num_epochs": len(values),
            "window": train_config.convergence_window,
            "relative_tolerance": train_config.convergence_relative_tolerance,
            "window_relative_improvement": None,
            "final_relative_gap_to_best": None,
            "best_epoch": None,
            "converged": False,
            "reason": (
                "non_finite_history" if not finite else "insufficient_history"
            ),
        }
    window_start = values[-train_config.convergence_window - 1]
    final = values[-1]
    best = min(values)
    denominator = max(abs(window_start), abs(best), 1.0e-16)
    window_improvement = (window_start - final) / denominator
    final_gap = (final - best) / max(abs(best), 1.0e-16)
    tolerance = train_config.convergence_relative_tolerance
    converged = (
        abs(window_improvement) <= tolerance
        and final_gap <= tolerance
    )
    return {
        "metric": metric,
        "num_epochs": len(values),
        "window": train_config.convergence_window,
        "relative_tolerance": tolerance,
        "window_relative_improvement": window_improvement,
        "final_relative_gap_to_best": final_gap,
        "best_epoch": values.index(best) + 1,
        "converged": converged,
        "reason": "converged" if converged else "trailing_window_not_converged",
    }


def _require_training_convergence(
    history: tuple[dict[str, float], ...] | list[dict[str, float]],
    *,
    metric: str,
    phase: str,
    train_config: FlexTrainConfig,
) -> dict[str, int | float | bool | str | None]:
    """Raise instead of promoting a phase without measured convergence."""
    audit = training_convergence_audit(
        history,
        metric=metric,
        train_config=train_config,
    )
    print(
        f"convergence phase={phase} converged={audit['converged']} "
        f"window_improvement={audit['window_relative_improvement']} "
        f"final_gap={audit['final_relative_gap_to_best']}",
        flush=True,
    )
    if not audit["converged"]:
        raise TrainingConvergenceError(
            f"{phase} failed convergence: {json.dumps(audit, sort_keys=True)}"
        )
    return audit


def _evaluate_fine_tuned_model(
    model: FlexTorchModel,
    days: tuple[TorchMarketDay, ...],
    *,
    env: BaseEnvironment,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    pre_training_score: dict[str, float],
    phase: str,
    equal_date_weighting: bool = False,
) -> FitEvaluation:
    """Evaluate the actual candidate and reject any objective regression."""
    if not _evaluation_score_is_valid(pre_training_score):
        raise ValueError("Invalid pre-training evaluation score")
    try:
        evaluation = evaluate_model(
            model,
            days,
            env=env,
            objective_config=objective_config,
            day_batch_size=day_batch_size,
            equal_date_weighting=equal_date_weighting,
        )
    except ValueError as error:
        raise TrainingConvergenceError(
            f"{phase} produced an invalid final candidate; no fallback was applied"
        ) from error
    final_score = _evaluation_score(evaluation)
    if not _evaluation_score_is_valid(final_score):
        raise TrainingConvergenceError(
            f"{phase} produced non-finite final metrics; no fallback was applied"
        )
    allowed_regression = 1.0e-10 * max(
        1.0,
        abs(pre_training_score["objective"]),
    )
    if final_score["objective"] > (
        pre_training_score["objective"] + allowed_regression
    ):
        raise TrainingConvergenceError(
            f"{phase} objective regressed from "
            f"{pre_training_score['objective']:.10g} to "
            f"{final_score['objective']:.10g}; no fallback was applied"
        )
    return evaluation


def _clip_sparse_gradient_(parameter: torch.Tensor, max_norm: float) -> None:
    """Clip the populated rows of one sparse embedding gradient in place."""
    gradient = parameter.grad
    if gradient is None:
        return
    if not gradient.is_sparse:
        raise ValueError("Expected a sparse embedding gradient")
    gradient = gradient.coalesce()
    norm = torch.linalg.vector_norm(gradient.values())
    scale = torch.clamp(max_norm / (norm + 1.0e-12), max=1.0)
    gradient.values().mul_(scale)
    parameter.grad = gradient


def _market_optimizers(
    model: FlexTorchModel,
    *,
    train_config: FlexTrainConfig,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LambdaLR]]:
    """Use sparse Adam for daily codes and dense Adam for the shared decoder."""
    sparse_parameter = (
        model.daily_raw_states
        if isinstance(model, NativeStateSanosModel)
        else model.daily_states
    )
    sparse_optimizer = torch.optim.SparseAdam(
        [sparse_parameter],
        lr=train_config.learning_rate,
    )
    optimizers: list[torch.optim.Optimizer] = [sparse_optimizer]
    if isinstance(model, LatentStateSanosModel):
        optimizers.append(
            torch.optim.Adam(
                model.raw_decoder.parameters(),
                lr=(
                    train_config.learning_rate
                    if train_config.decoder_learning_rate is None
                    else train_config.decoder_learning_rate
                ),
                weight_decay=train_config.weight_decay,
            )
        )
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, train_config),
        )
        for optimizer in optimizers
    ]
    return optimizers, schedulers


def _torch_rng_state(model: FlexTorchModel) -> TrainingState:
    """Capture CPU and selected-device Torch RNG state for exact restarts."""
    state: TrainingState = {"torch_rng_state": torch.random.get_rng_state()}
    device = torch.device(model.device)
    if device.type == "cuda":
        state["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return state


def _restore_torch_rng_state(
    model: FlexTorchModel,
    state: TrainingState,
) -> None:
    """Restore RNG state after model and optimizer construction."""
    torch.random.set_rng_state(cast(torch.Tensor, state["torch_rng_state"]))
    device = torch.device(model.device)
    if device.type == "cuda":
        torch.cuda.set_rng_state(
            cast(torch.Tensor, state["cuda_rng_state"]),
            device=device,
        )


def _resume_signature(
    train_config: FlexTrainConfig,
    *,
    objective_config: FlexObjectiveConfig | None,
) -> dict[str, object]:
    """Identify settings that must match for an exact optimizer continuation."""
    return {
        "training_algorithm_version": _TRAINING_ALGORITHM_VERSION,
        "num_epochs": train_config.num_epochs,
        "learning_rate": train_config.learning_rate,
        "decoder_learning_rate": train_config.decoder_learning_rate,
        "learning_rate_schedule": train_config.learning_rate_schedule,
        "learning_rate_warmup_epochs": train_config.learning_rate_warmup_epochs,
        "minimum_learning_rate_fraction": (
            train_config.minimum_learning_rate_fraction
        ),
        "weight_decay": train_config.weight_decay,
        "gradient_clip": train_config.gradient_clip,
        "seed": train_config.seed,
        "precision": train_config.precision,
        "device": train_config.device,
        "compile_sanos_solver": train_config.compile_sanos_solver,
        "objective_config": (
            None if objective_config is None else asdict(objective_config)
        ),
    }


def _resume_signatures_match(
    stored_signature: object,
    current_signature: dict[str, object],
) -> bool:
    """Ignore legacy operational fields when validating a continuation."""
    if not isinstance(stored_signature, dict):
        return False
    normalized_stored = dict(stored_signature)
    # Older schema-2 checkpoints treated this throughput control as part of
    # exact optimizer compatibility even though it changes no state topology.
    normalized_stored.pop("day_batch_size", None)
    return normalized_stored == current_signature


def _market_training_phase(
    model: FlexTorchModel,
    days: tuple[TorchMarketDay, ...],
    *,
    env: BaseEnvironment,
    objective_config: FlexObjectiveConfig,
    train_config: FlexTrainConfig,
    pre_training_score: dict[str, float],
    resume_state: TrainingState | None = None,
    checkpoint_callback: Callable[[TrainingState], None] | None = None,
    evaluation_callback: Callable[[int, FitEvaluation], None] | None = None,
) -> tuple[tuple[dict[str, float], ...], TrainingState]:
    """Fine-tune a native or compressed model against every real quote."""
    optimizers, schedulers = _market_optimizers(
        model,
        train_config=train_config,
    )
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    start_epoch = 0
    resume_signature = _resume_signature(
        train_config,
        objective_config=objective_config,
    )
    if resume_state is not None:
        if resume_state.get("phase") != "market":
            raise ValueError(
                "Expected a market-phase FLEX training state, found "
                f"{resume_state.get('phase')!r}"
            )
        if not _resume_signatures_match(
            resume_state.get("resume_signature"),
            resume_signature,
        ):
            raise ValueError(
                "FLEX market checkpoint resume signature mismatch; use an "
                "explicit weights-only warm start for changed settings"
            )
        start_epoch = int(resume_state["completed_epochs"])
        if start_epoch > train_config.num_epochs:
            raise ValueError(
                "Checkpoint has more completed market epochs than requested: "
                f"{start_epoch} > {train_config.num_epochs}"
            )
        history = [
            {str(key): float(value) for key, value in item.items()}
            for item in cast(list[dict[str, float]], resume_state["history"])
        ]
        optimizer_states = cast(
            list[dict[str, object]],
            resume_state["optimizer_state_dicts"],
        )
        scheduler_states = cast(
            list[dict[str, object]],
            resume_state["scheduler_state_dicts"],
        )
        if len(optimizer_states) != len(optimizers) or len(
            scheduler_states
        ) != len(schedulers):
            raise ValueError("FLEX checkpoint optimizer topology mismatch")
        for optimizer, state in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_states, strict=True):
            scheduler.load_state_dict(state)
        rng.bit_generator.state = cast(
            dict[str, object],
            resume_state["numpy_rng_state"],
        )
        _restore_torch_rng_state(model, resume_state)
        pre_training_score = cast(
            dict[str, float],
            resume_state["pre_training_score"],
        )
    if not _evaluation_score_is_valid(pre_training_score):
        raise ValueError("FLEX checkpoint has an invalid pre-training score")

    def training_state(completed_epochs: int) -> TrainingState:
        return {
            "phase": "market",
            "resume_signature": resume_signature,
            "completed_epochs": completed_epochs,
            "history": list(history),
            "optimizer_state_dicts": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "scheduler_state_dicts": [
                scheduler.state_dict() for scheduler in schedulers
            ],
            "pre_training_score": pre_training_score,
            "numpy_rng_state": rng.bit_generator.state,
            **_torch_rng_state(model),
        }

    started = perf_counter()
    model.train()
    for epoch in range(start_epoch, train_config.num_epochs):
        used_learning_rate = float(optimizers[0].param_groups[0]["lr"])
        used_decoder_learning_rate = (
            float(optimizers[1].param_groups[0]["lr"])
            if len(optimizers) > 1
            else used_learning_rate
        )
        order = rng.permutation(len(days))
        objective_terms: list[torch.Tensor] = []
        quote_objective_terms: list[torch.Tensor] = []
        cap_penalty_terms: list[torch.Tensor] = []
        date_count = 0
        for start in range(0, len(days), train_config.day_batch_size):
            indices = order[start : start + train_config.day_batch_size]
            batch = materialize_batch(days, indices, env=env)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            prices = _model_prices(model, batch)
            quote_loss = equal_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            cap_penalty = model_dlvsq_cap_penalty(model, batch.day_indices)
            loss = quote_loss + cap_penalty
            loss.backward()
            sparse_parameter = (
                model.daily_raw_states
                if isinstance(model, NativeStateSanosModel)
                else model.daily_states
            )
            _clip_sparse_gradient_(sparse_parameter, train_config.gradient_clip)
            if isinstance(model, LatentStateSanosModel):
                torch.nn.utils.clip_grad_norm_(
                    model.raw_decoder.parameters(),
                    train_config.gradient_clip,
                )
            for optimizer in optimizers:
                optimizer.step()
            count = len(indices)
            objective_terms.append(loss.detach() * count)
            quote_objective_terms.append(quote_loss.detach() * count)
            cap_penalty_terms.append(cap_penalty.detach() * count)
            date_count += count
        for scheduler in schedulers:
            scheduler.step()
        objective_sum = float(torch.stack(objective_terms).sum().cpu())
        quote_objective_sum = float(
            torch.stack(quote_objective_terms).sum().cpu()
        )
        cap_penalty_sum = float(torch.stack(cap_penalty_terms).sum().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "objective": objective_sum / date_count,
                "quote_objective": quote_objective_sum / date_count,
                "date_weighted_quote_objective": quote_objective_sum / date_count,
                "dlvsq_cap_penalty": cap_penalty_sum / date_count,
                "learning_rate": used_learning_rate,
                "decoder_learning_rate": used_decoder_learning_rate,
                "day_batch_size": float(train_config.day_batch_size),
            }
        )
        if (epoch + 1) % train_config.report_every == 0 or epoch == 0:
            state_size = (
                model.config.geometry.native_state_size
                if isinstance(model, NativeStateSanosModel)
                else model.config.state_size
            )
            elapsed = perf_counter() - started
            remaining = elapsed * (train_config.num_epochs - epoch - 1) / (epoch + 1 - start_epoch)
            print(
                f"batch-descent state={state_size} epoch={epoch + 1}/"
                f"{train_config.num_epochs} objective="
                f"{history[-1]['objective']:.8g} elapsed={elapsed:.1f}s "
                f"remaining={remaining:.1f}s",
                flush=True,
            )
        if checkpoint_callback is not None and (epoch + 1) % train_config.checkpoint_every == 0:
            checkpoint_callback(training_state(epoch + 1))
        if evaluation_callback is not None and (
            epoch == start_epoch
            or (epoch + 1) % train_config.report_every == 0
            or epoch + 1 == train_config.num_epochs
        ):
            evaluation = evaluate_model(
                model, days, env=env, objective_config=objective_config,
                day_batch_size=train_config.day_batch_size, equal_date_weighting=True,
            )
            evaluation_callback(epoch + 1, evaluation)
            model.train()
    final_state = training_state(train_config.num_epochs)
    return tuple(history), final_state


def fit_native_states(
    market_days: tuple[FlexMarketDay, ...],
    *,
    config: NativeStateConfig,
    objective_config: FlexObjectiveConfig = FlexObjectiveConfig(),
    train_config: FlexTrainConfig = FlexTrainConfig(),
    checkpoint_path: Path | None = None,
    weights_only_warm_start_path: Path | None = None,
    evaluation_callback: Callable[[int, FitEvaluation], None] | None = None,
) -> NativeStateFitResult:
    """Fit one independent bounded raw state per day by batch descent."""
    if len(market_days) == 0:
        raise ValueError("At least one market day is required")
    env = BaseEnvironment(
        precision=train_config.precision,
        device=train_config.device,
        debug=False,
    )
    days = tuple(materialize_day(day, env=env) for day in market_days)
    _seed_model_initialization(
        seed=train_config.seed,
        device=train_config.device,
    )
    model = NativeStateSanosModel(
        env,
        num_days=len(days),
        config=config,
    )
    resumed = checkpoint_path is not None and checkpoint_path.exists()
    if resumed and weights_only_warm_start_path is not None:
        raise ValueError(
            "A native target checkpoint and a weights-only warm start cannot "
            "both be supplied"
        )
    checkpoint_payload: TrainingState | None = None
    if resumed:
        assert checkpoint_path is not None
        checkpoint_payload = load_model_checkpoint(
            model,
            checkpoint_path,
            market_days=market_days,
            require_complete=train_config.just_load,
        )
    elif weights_only_warm_start_path is not None:
        install_weights_only_warm_start(model, weights_only_warm_start_path)
    else:
        with torch.no_grad():
            model.daily_raw_states.copy_(
                market_native_raw_initial_states(
                    market_days,
                    env=env,
                    config=config,
                )
            )
    if train_config.just_load and not resumed:
        raise FileNotFoundError(f"No native FLEX checkpoint exists at {checkpoint_path}")
    if (
        checkpoint_payload is not None
        and checkpoint_payload.get("phase") in {"legacy_model", "complete"}
        and not train_config.just_load
        and not train_config.warm_start
    ):
        raise ValueError(
            "Native checkpoint is weights-only or complete; use 'just_load' "
            "or explicit 'warm_start' instead of implicit retraining"
        )
    started = perf_counter()
    initial_evaluation = evaluate_model(
        model,
        days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
    )
    if train_config.just_load:
        history = tuple(
            {
                str(key): float(value)
                for key, value in item.items()
            }
            for item in cast(
                list[dict[str, float]],
                (
                    checkpoint_payload.get("market_history", [])
                    if checkpoint_payload is not None
                    else []
                ),
            )
        )
        return NativeStateFitResult(
            model=model,
            evaluation=initial_evaluation,
            history=history,
            elapsed_seconds=perf_counter() - started,
            market_fingerprints=tuple(
                day.training_fingerprint for day in market_days
            ),
            state_dtype=str(model.daily_raw_states.dtype),
            convergence=(
                training_convergence_audit(
                    history,
                    metric="objective",
                    train_config=train_config,
                )
                if history
                else None
            ),
        )
    if train_config.compile_sanos_solver:
        model.sanos_decoder.enable_solver_compile()
    resume_state = (
        cast(TrainingState, checkpoint_payload["training_state"])
        if checkpoint_payload is not None
        and checkpoint_payload.get("phase") == "market"
        and not train_config.warm_start
        else None
    )
    history, market_training_state = _market_training_phase(
        model,
        days,
        env=env,
        objective_config=objective_config,
        train_config=train_config,
        pre_training_score=_evaluation_score(initial_evaluation),
        resume_state=resume_state,
        evaluation_callback=evaluation_callback,
        checkpoint_callback=(
            None
            if checkpoint_path is None
            else lambda state: save_model_checkpoint(
                model,
                checkpoint_path,
                market_days=market_days,
                phase="market",
                training_state=state,
            )
        ),
    )
    convergence = _require_training_convergence(
        history,
        metric="objective",
        phase="native market training",
        train_config=train_config,
    )
    if train_config.compile_sanos_solver:
        model.sanos_decoder.disable_solver_compile()
    evaluation = _evaluate_fine_tuned_model(
        model,
        days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        pre_training_score=cast(
            dict[str, float],
            market_training_state["pre_training_score"],
        ),
        phase="native market training",
    )
    if checkpoint_path is not None:
        save_model_checkpoint(
            model,
            checkpoint_path,
            market_days=market_days,
            phase="complete",
            market_history=history,
            training_audits={"market": dict(convergence)},
        )
    return NativeStateFitResult(
        model=model,
        evaluation=evaluation,
        history=history,
        elapsed_seconds=perf_counter() - started,
        market_fingerprints=tuple(
            day.training_fingerprint for day in market_days
        ),
        state_dtype=str(model.daily_raw_states.dtype),
        convergence=dict(convergence),
    )


def _supervised_decoder_phase(
    model: LatentStateSanosModel,
    target_raw: torch.Tensor,
    *,
    train_config: FlexTrainConfig,
    resume_state: TrainingState | None = None,
    checkpoint_callback: Callable[[TrainingState], None] | None = None,
) -> tuple[tuple[dict[str, float], ...], TrainingState]:
    """Warm-start market compression by directly learning native raw states."""
    decoder_optimizer = torch.optim.Adam(
        model.raw_decoder.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    state_optimizer = torch.optim.SparseAdam(
        [model.daily_states],
        lr=train_config.learning_rate,
    )
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    start_epoch = 0
    resume_signature = _resume_signature(
        train_config,
        objective_config=None,
    )
    if resume_state is not None:
        if resume_state.get("phase") != "supervised":
            raise ValueError(
                "Expected a supervised FLEX training state, found "
                f"{resume_state.get('phase')!r}"
            )
        if not _resume_signatures_match(
            resume_state.get("resume_signature"),
            resume_signature,
        ):
            raise ValueError(
                "FLEX supervised checkpoint resume signature mismatch; use "
                "an explicit weights-only warm start for changed settings"
            )
        start_epoch = int(resume_state["completed_epochs"])
        if start_epoch > train_config.num_epochs:
            raise ValueError(
                "Checkpoint has more completed supervised epochs than "
                f"requested: {start_epoch} > {train_config.num_epochs}"
            )
        history = [
            {str(key): float(value) for key, value in item.items()}
            for item in cast(list[dict[str, float]], resume_state["history"])
        ]
        decoder_optimizer.load_state_dict(
            cast(
                dict[str, object],
                resume_state["decoder_optimizer_state_dict"],
            )
        )
        state_optimizer.load_state_dict(
            cast(
                dict[str, object],
                resume_state["state_optimizer_state_dict"],
            )
        )
        rng.bit_generator.state = cast(
            dict[str, object],
            resume_state["numpy_rng_state"],
        )
        _restore_torch_rng_state(model, resume_state)

    def training_state(completed_epochs: int) -> TrainingState:
        return {
            "phase": "supervised",
            "resume_signature": resume_signature,
            "completed_epochs": completed_epochs,
            "history": list(history),
            "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
            "state_optimizer_state_dict": state_optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            **_torch_rng_state(model),
        }

    model.train()
    for epoch in range(start_epoch, train_config.num_epochs):
        order = rng.permutation(model.num_days)
        squared_error_terms: list[torch.Tensor] = []
        value_count = 0
        for start in range(0, model.num_days, train_config.day_batch_size):
            indices = order[start : start + train_config.day_batch_size]
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=model.device,
            )
            decoder_optimizer.zero_grad(set_to_none=True)
            state_optimizer.zero_grad(set_to_none=True)
            decoded = model.raw_states_for_days(index_tensor)
            loss = torch.mean((decoded - target_raw[index_tensor]).square())
            loss.backward()
            _clip_sparse_gradient_(
                model.daily_states,
                train_config.gradient_clip,
            )
            torch.nn.utils.clip_grad_norm_(
                model.raw_decoder.parameters(),
                train_config.gradient_clip,
            )
            decoder_optimizer.step()
            state_optimizer.step()
            count = int(decoded.numel())
            squared_error_terms.append(loss.detach() * count)
            value_count += count
        squared_error = float(
            torch.stack(squared_error_terms).sum().cpu()
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "mse": squared_error / value_count,
                "day_batch_size": float(train_config.day_batch_size),
            }
        )
        if (epoch + 1) % train_config.report_every == 0 or epoch == 0:
            print(
                f"compression state={model.config.state_size} "
                f"epoch={epoch + 1}/{train_config.num_epochs} "
                f"native_mse={history[-1]['mse']:.8g}",
                flush=True,
            )
        if checkpoint_callback is not None and (
            epoch + 1
        ) % train_config.checkpoint_every == 0:
            checkpoint_callback(training_state(epoch + 1))
    final_state = training_state(train_config.num_epochs)
    return tuple(history), final_state


def initialize_pca_decoder_(
    model: LatentStateSanosModel,
    target_raw: torch.Tensor,
) -> dict[str, float]:
    """Initialize an affine decoder to the optimal PCA reconstruction."""
    if not isinstance(model.config, PcaDecoderConfig):
        raise ValueError("PCA initialization requires a PcaDecoderConfig")
    if target_raw.shape != (
        model.num_days,
        model.config.geometry.native_state_size,
    ):
        raise ValueError(
            "PCA target shape does not match model geometry: "
            f"{tuple(target_raw.shape)}"
        )
    if len(model.raw_decoder.In) != 0:
        raise ValueError("PCA decoder must contain exactly one affine layer")

    mean = torch.mean(target_raw, dim=0)
    centered = target_raw - mean
    left, singular_values, right_t = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    rank = min(model.config.state_size, int(singular_values.numel()))
    loadings = right_t[:rank].transpose(0, 1).contiguous()
    scores = left[:, :rank] * singular_values[:rank]

    # Anchor each PCA sign by its largest loading so repeated states have a
    # stable orientation for the later time-series model.
    loading_anchor = torch.argmax(torch.abs(loadings), dim=0)
    component = torch.arange(rank, device=model.device)
    signs = torch.where(
        loadings[loading_anchor, component] < 0.0,
        -torch.ones(rank, dtype=model.fdtype, device=model.device),
        torch.ones(rank, dtype=model.fdtype, device=model.device),
    )
    loadings = loadings * signs
    scores = scores * signs

    linear = model.raw_decoder.I0
    if linear.bias is None:
        raise ValueError("PCA decoder requires a final output bias")
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(mean)
        model.daily_states.zero_()
        linear.weight[:, :rank].copy_(loadings)
        model.daily_states[:, :rank].copy_(scores)

    with torch.no_grad():
        indices = torch.arange(
            model.num_days,
            dtype=torch.long,
            device=model.device,
        )
        reconstruction = model.raw_states_for_days(indices)
        mse = torch.mean((reconstruction - target_raw).square())
        total_energy = torch.sum(singular_values.square())
        residual_energy = torch.sum(singular_values[rank:].square())
        explained = torch.where(
            total_energy > 0.0,
            1.0 - residual_energy / total_energy,
            torch.ones_like(total_energy),
        )
    return {
        "epoch": 0.0,
        "mse": float(mse.cpu()),
        "explained_variance_fraction": float(explained.cpu()),
        "pca_rank": float(rank),
    }


def _triangular_pca_components(
    target_raw: torch.Tensor,
    *,
    geometry: ForwardMuFlexGeometryConfig,
    state_sizes_by_expiry: tuple[int, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[dict[str, float], ...],
    tuple[torch.Tensor, ...],
]:
    """Construct a sequential conditional-PCA factorization of raw states."""
    num_days = target_raw.shape[0]
    T = geometry.num_model_expiries
    block_width = geometry.expiry_block_state_size
    total_state_size = sum(state_sizes_by_expiry)
    if target_raw.shape != (num_days, geometry.native_state_size):
        raise ValueError("Triangular PCA target does not match its geometry")
    if len(state_sizes_by_expiry) != T:
        raise ValueError("Triangular PCA requires one state block per expiry")

    # The smaller conditional modes are economically meaningful even when the
    # production model is float32.  A float32 LAPACK rank tolerance can discard
    # requested modes that remain well resolved in the source history, leaving
    # exact zero factor/decoder pairs that cannot subsequently train through
    # their bilinear parameterization.  Perform the small PCA problems in
    # float64, then cast the exact factorization back to the model dtype.
    working_raw = target_raw.to(dtype=torch.float64)

    scores = torch.zeros(
        (num_days, total_state_size),
        dtype=working_raw.dtype,
        device=target_raw.device,
    )
    weight = torch.zeros(
        (geometry.native_state_size, total_state_size),
        dtype=working_raw.dtype,
        device=target_raw.device,
    )
    bias = torch.zeros(
        geometry.native_state_size,
        dtype=working_raw.dtype,
        device=target_raw.device,
    )
    diagnostics: list[dict[str, float]] = []
    spectra: list[torch.Tensor] = []
    state_start = 0
    eps = torch.finfo(working_raw.dtype).eps

    for expiry_index, requested_rank in enumerate(state_sizes_by_expiry):
        output_start = expiry_index * block_width
        output_stop = output_start + block_width
        block = working_raw[:, output_start:output_stop]
        block_mean = torch.mean(block, dim=0)
        centered = block - block_mean
        bias[output_start:output_stop] = block_mean

        if state_start > 0:
            prior_scores = scores[:, :state_start]
            score_energy = torch.sum(prior_scores.square(), dim=0)
            active = torch.nonzero(
                score_energy > eps * max(num_days, 1),
                as_tuple=False,
            ).flatten()
            if active.numel() > 0:
                active_scores = prior_scores[:, active]
                coefficients = (
                    active_scores.transpose(0, 1) @ centered
                ) / score_energy[active].unsqueeze(-1)
                centered = centered - active_scores @ coefficients
                weight[output_start:output_stop, active] = coefficients.transpose(
                    0,
                    1,
                )

        left, singular_values, right_t = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        spectra.append(singular_values.detach().clone())
        tolerance = (
            max(centered.shape)
            * eps
            * (singular_values[0] if singular_values.numel() else 0.0)
        )
        numerical_rank = int(torch.sum(singular_values > tolerance).item())
        rank = min(requested_rank, numerical_rank)
        state_stop = state_start + requested_rank
        if rank > 0:
            loadings = right_t[:rank].transpose(0, 1).contiguous()
            new_scores = left[:, :rank] * singular_values[:rank]
            loading_anchor = torch.argmax(torch.abs(loadings), dim=0)
            component = torch.arange(rank, device=target_raw.device)
            signs = torch.where(
                loadings[loading_anchor, component] < 0.0,
                -torch.ones(rank, dtype=working_raw.dtype, device=target_raw.device),
                torch.ones(rank, dtype=working_raw.dtype, device=target_raw.device),
            )
            loadings = loadings * signs
            new_scores = new_scores * signs
            scores[:, state_start : state_start + rank] = new_scores
            weight[
                output_start:output_stop,
                state_start : state_start + rank,
            ] = loadings

        residual_energy = torch.sum(singular_values.square())
        remaining_energy = torch.sum(singular_values[rank:].square())
        explained = torch.where(
            residual_energy > 0.0,
            1.0 - remaining_energy / residual_energy,
            torch.ones_like(residual_energy),
        )
        diagnostics.append(
            {
                "expiry_index": float(expiry_index),
                "requested_rank": float(requested_rank),
                "numerical_rank": float(numerical_rank),
                "conditional_explained_variance_fraction": float(
                    explained.detach().cpu()
                ),
                "conditional_residual_energy": float(
                    remaining_energy.detach().cpu()
                ),
            }
        )
        state_start = state_stop

    return (
        scores.to(dtype=target_raw.dtype),
        weight.to(dtype=target_raw.dtype),
        bias.to(dtype=target_raw.dtype),
        tuple(diagnostics),
        tuple(value.to(dtype=target_raw.dtype) for value in spectra),
    )


def initialize_triangular_pca_decoder_(
    model: LatentStateSanosModel,
    target_raw: torch.Tensor,
) -> dict[str, float]:
    """Initialize the masked affine decoder by sequential conditional PCA."""
    if not isinstance(model.config, TriangularPcaDecoderConfig):
        raise ValueError(
            "Triangular PCA initialization requires TriangularPcaDecoderConfig"
        )
    scores, weight, bias, diagnostics, _ = _triangular_pca_components(
        target_raw,
        geometry=model.config.geometry,
        state_sizes_by_expiry=model.config.state_sizes_by_expiry,
    )
    linear = model.raw_decoder.I0
    if linear.bias is None or len(model.raw_decoder.In) != 0:
        raise ValueError("Triangular PCA decoder must contain one affine layer")
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
        model.daily_states.copy_(scores)

        indices = torch.arange(
            model.num_days,
            dtype=torch.long,
            device=model.device,
        )
        reconstruction = model.raw_states_for_days(indices)
        squared_error = (reconstruction - target_raw).square()
        mse = torch.mean(squared_error)
        centered = target_raw - torch.mean(target_raw, dim=0)
        total_energy = torch.sum(centered.square())
        residual_energy = torch.sum(squared_error)
        explained = torch.where(
            total_energy > 0.0,
            1.0 - residual_energy / total_energy,
            torch.ones_like(total_energy),
        )
    output = {
        "epoch": 0.0,
        "mse": float(mse.cpu()),
        "explained_variance_fraction": float(explained.cpu()),
        "pca_rank": float(sum(model.config.state_sizes_by_expiry)),
    }
    for diagnostic in diagnostics:
        expiry_index = int(diagnostic["expiry_index"]) + 1
        output[f"expiry_{expiry_index}_state_size"] = diagnostic[
            "requested_rank"
        ]
        output[f"expiry_{expiry_index}_conditional_explained"] = diagnostic[
            "conditional_explained_variance_fraction"
        ]
    return output


def fit_latent_states(
    market_days: tuple[FlexMarketDay, ...],
    native_fit: NativeStateFitResult,
    *,
    config: (
        LatentDecoderConfig
        | PcaDecoderConfig
        | TriangularPcaDecoderConfig
    ),
    objective_config: FlexObjectiveConfig = FlexObjectiveConfig(),
    supervised_train_config: FlexTrainConfig = FlexTrainConfig(),
    market_train_config: FlexTrainConfig = FlexTrainConfig(),
    checkpoint_path: Path | None = None,
    evaluation_callback: Callable[[int, FitEvaluation], None] | None = None,
) -> LatentStateFitResult:
    """Fit and fine-tune one lower-dimensional shared decoder."""
    if len(market_days) == 0 or len(market_days) != native_fit.model.num_days:
        raise ValueError("Market days must match the fitted native daily states")
    market_fingerprints = tuple(
        day.training_fingerprint for day in market_days
    )
    if market_fingerprints != native_fit.market_fingerprints:
        raise ValueError(
            "Compression market days do not match the native fit provenance"
        )
    if (
        native_fit.convergence is None
        or native_fit.convergence.get("converged") is not True
    ):
        raise TrainingConvergenceError(
            "Latent fine-tuning requires a converged native-state target"
        )
    if (
        supervised_train_config.device != market_train_config.device
        or supervised_train_config.precision != market_train_config.precision
    ):
        raise ValueError("Both latent training phases must use one device and dtype")
    env = BaseEnvironment(
        precision=market_train_config.precision,
        device=market_train_config.device,
        debug=False,
    )
    days = tuple(materialize_day(day, env=env) for day in market_days)
    if native_fit.state_dtype != str(env.fdtype):
        raise ValueError(
            "Compression dtype does not match the exact native target: "
            f"{env.fdtype} versus {native_fit.state_dtype}"
        )
    supervision_target_identity = native_target_identity(
        native_fit.model,
        market_fingerprints=native_fit.market_fingerprints,
    )
    target_raw = native_fit.model.daily_raw_states.detach().to(
        dtype=env.fdtype,
        device=env.device,
    ).clone()
    _seed_model_initialization(
        seed=supervised_train_config.seed,
        device=market_train_config.device,
    )
    model = LatentStateSanosModel(
        env,
        num_days=len(days),
        config=config,
    )
    resumed = checkpoint_path is not None and checkpoint_path.exists()
    checkpoint_payload: TrainingState | None = None
    if resumed:
        assert checkpoint_path is not None
        checkpoint_payload = load_model_checkpoint(
            model,
            checkpoint_path,
            market_days=market_days,
            require_complete=market_train_config.just_load,
            expected_supervision_target_identity=supervision_target_identity,
        )
    if market_train_config.just_load and not resumed:
        raise FileNotFoundError(f"No latent FLEX checkpoint exists at {checkpoint_path}")
    if (
        checkpoint_payload is not None
        and checkpoint_payload.get("phase") == "complete"
        and not market_train_config.just_load
        and not market_train_config.warm_start
    ):
        raise ValueError(
            "Latent checkpoint is complete; use 'just_load' or explicit "
            "'warm_start' instead of implicit retraining"
        )
    started = perf_counter()
    saved_supervised_history = (
        tuple(
            {
                str(key): float(value)
                for key, value in item.items()
            }
            for item in cast(
                list[dict[str, float]],
                checkpoint_payload.get("supervised_history", []),
            )
        )
        if checkpoint_payload is not None
        else ()
    )
    checkpoint_phase = (
        str(checkpoint_payload.get("phase"))
        if checkpoint_payload is not None
        else None
    )
    skip_supervised = market_train_config.just_load or checkpoint_phase in {
        "legacy_model",
        "supervised_complete",
        "market",
        "complete",
    }
    if skip_supervised:
        supervised_history = saved_supervised_history
    elif isinstance(config, TriangularPcaDecoderConfig):
        supervised_history = (
            initialize_triangular_pca_decoder_(model, target_raw),
        )
    elif isinstance(config, PcaDecoderConfig):
        supervised_history = (
            initialize_pca_decoder_(model, target_raw),
        )
    else:
        supervised_resume_state = (
            cast(TrainingState, checkpoint_payload["training_state"])
            if checkpoint_payload is not None
            and checkpoint_phase == "supervised"
            and not supervised_train_config.warm_start
            else None
        )
        supervised_history, _ = _supervised_decoder_phase(
            model,
            target_raw,
            train_config=supervised_train_config,
            resume_state=supervised_resume_state,
            checkpoint_callback=(
                None
                if checkpoint_path is None
                else lambda state: save_model_checkpoint(
                    model,
                    checkpoint_path,
                    market_days=market_days,
                    phase="supervised",
                    training_state=state,
                    supervision_target_identity=supervision_target_identity,
                )
            ),
        )
    if isinstance(config, PcaDecoderConfig | TriangularPcaDecoderConfig):
        supervised_convergence: dict[str, object] = {
            "metric": "mse",
            "num_epochs": 0,
            "window": 0,
            "relative_tolerance": 0.0,
            "window_relative_improvement": 0.0,
            "final_relative_gap_to_best": 0.0,
            "best_epoch": 0,
            "converged": True,
            "reason": "analytic_pca_solution",
        }
    else:
        supervised_convergence = dict(
            _require_training_convergence(
                supervised_history,
                metric="mse",
                phase="latent supervised reconstruction",
                train_config=supervised_train_config,
            )
        )
    if checkpoint_path is not None and not skip_supervised:
        save_model_checkpoint(
            model,
            checkpoint_path,
            market_days=market_days,
            phase="supervised_complete",
            supervised_history=supervised_history,
            supervision_target_identity=supervision_target_identity,
            training_audits={"supervised": supervised_convergence},
        )
    if market_train_config.just_load:
        market_history = tuple(
            {
                str(key): float(value)
                for key, value in item.items()
            }
            for item in cast(
                list[dict[str, float]],
                (
                    checkpoint_payload.get("market_history", [])
                    if checkpoint_payload is not None
                    else []
                ),
            )
        )
        initial_evaluation: FitEvaluation | None = None
        if isinstance(config, PcaDecoderConfig | TriangularPcaDecoderConfig):
            initial_model = LatentStateSanosModel(
                env,
                num_days=len(days),
                config=config,
            )
            if isinstance(config, TriangularPcaDecoderConfig):
                initialize_triangular_pca_decoder_(initial_model, target_raw)
            else:
                initialize_pca_decoder_(initial_model, target_raw)
            initial_evaluation = evaluate_model(
                initial_model,
                days,
                env=env,
                objective_config=objective_config,
                day_batch_size=market_train_config.day_batch_size,
                equal_date_weighting=True,
            )
        evaluation = evaluate_model(
            model,
            days,
            env=env,
            objective_config=objective_config,
            day_batch_size=market_train_config.day_batch_size,
            equal_date_weighting=True,
        )
        market_convergence = _require_training_convergence(
            market_history,
            metric="objective",
            phase="latent market fine-tuning",
            train_config=market_train_config,
        )
        return LatentStateFitResult(
            model=model,
            evaluation=evaluation,
            initial_evaluation=initial_evaluation,
            supervised_history=supervised_history,
            market_history=market_history,
            elapsed_seconds=perf_counter() - started,
            supervised_convergence=supervised_convergence,
            market_convergence=dict(market_convergence),
        )
    supervised_evaluation = evaluate_model(
        model,
        days,
        env=env,
        objective_config=objective_config,
        day_batch_size=market_train_config.day_batch_size,
        equal_date_weighting=True,
    )
    if market_train_config.compile_sanos_solver:
        model.sanos_decoder.enable_solver_compile()
    market_resume_state = (
        cast(TrainingState, checkpoint_payload["training_state"])
        if checkpoint_payload is not None
        and checkpoint_phase == "market"
        and not market_train_config.warm_start
        else None
    )
    market_history, market_training_state = _market_training_phase(
        model,
        days,
        env=env,
        objective_config=objective_config,
        train_config=market_train_config,
        pre_training_score=_evaluation_score(supervised_evaluation),
        resume_state=market_resume_state,
        evaluation_callback=evaluation_callback,
        checkpoint_callback=(
            None
            if checkpoint_path is None
            else lambda state: save_model_checkpoint(
                model,
                checkpoint_path,
                market_days=market_days,
                phase="market",
                training_state=state,
                supervised_history=supervised_history,
                supervision_target_identity=supervision_target_identity,
            )
        ),
    )
    market_convergence = _require_training_convergence(
        market_history,
        metric="objective",
        phase="latent market fine-tuning",
        train_config=market_train_config,
    )
    if market_train_config.compile_sanos_solver:
        model.sanos_decoder.disable_solver_compile()
    evaluation = _evaluate_fine_tuned_model(
        model,
        days,
        env=env,
        objective_config=objective_config,
        day_batch_size=market_train_config.day_batch_size,
        pre_training_score=cast(
            dict[str, float],
            market_training_state["pre_training_score"],
        ),
        phase="latent market fine-tuning",
        equal_date_weighting=True,
    )
    if checkpoint_path is not None:
        save_model_checkpoint(
            model,
            checkpoint_path,
            market_days=market_days,
            phase="complete",
            supervised_history=supervised_history,
            market_history=market_history,
            supervision_target_identity=supervision_target_identity,
            training_audits={
                "supervised": supervised_convergence,
                "market": dict(market_convergence),
            },
        )
    return LatentStateFitResult(
        model=model,
        evaluation=evaluation,
        initial_evaluation=supervised_evaluation,
        supervised_history=supervised_history,
        market_history=market_history,
        elapsed_seconds=perf_counter() - started,
        supervised_convergence=supervised_convergence,
        market_convergence=dict(market_convergence),
    )


def save_model_checkpoint(
    model: FlexTorchModel,
    path: Path,
    *,
    market_days: tuple[FlexMarketDay, ...],
    phase: str,
    training_state: TrainingState | None = None,
    supervised_history: tuple[dict[str, float], ...] = (),
    market_history: tuple[dict[str, float], ...] = (),
    supervision_target_identity: dict[str, object] | None = None,
    training_audits: dict[str, dict[str, object]] | None = None,
) -> None:
    """Atomically persist model identity and near-perfect restart state."""
    if phase not in {
        "supervised",
        "supervised_complete",
        "market",
        "complete",
    }:
        raise ValueError(f"Unsupported FLEX checkpoint phase {phase!r}")
    if (phase in {"supervised", "market"}) != (training_state is not None):
        raise ValueError(
            "In-progress FLEX checkpoints require training state and completed "
            "checkpoints must not contain it"
        )
    is_latent = isinstance(model, LatentStateSanosModel)
    if is_latent != (supervision_target_identity is not None):
        raise ValueError(
            "Latent FLEX checkpoints require an exact native supervision "
            "target identity"
        )
    if phase == "complete":
        required_audits = {"market", "supervised"} if is_latent else {"market"}
        if training_audits is None or not required_audits.issubset(
            training_audits
        ) or not all(
            training_audits[name].get("converged") is True
            for name in required_audits
        ):
            raise ValueError(
                "Complete FLEX checkpoints require successful convergence audits"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "model_identity": _model_checkpoint_identity(model),
            "model_state_dict": model.state_dict(),
            "market_fingerprints": [day.training_fingerprint for day in market_days],
            "phase": phase,
            "training_state": training_state,
            "supervised_history": list(supervised_history),
            "market_history": list(market_history),
            "supervision_target_identity": supervision_target_identity,
            "training_audits": training_audits,
        },
        temporary,
    )
    temporary.replace(path)


def _model_checkpoint_identity(model: FlexTorchModel) -> dict[str, object]:
    """Return the logical model identity required by repository policy."""
    return {
        "model_class": type(model).__qualname__,
        "config_version": model.config.version.unique_id48,
        "user_version": model.config.user_version,
        "model_config": asdict(model.config),
    }


def load_model_checkpoint(
    model: FlexTorchModel,
    path: Path,
    *,
    market_days: tuple[FlexMarketDay, ...],
    require_complete: bool = False,
    expected_supervision_target_identity: dict[str, object] | None = None,
) -> TrainingState:
    """Load only a checkpoint fitted to the identical ordered quote universe."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"Unsupported FLEX checkpoint schema in {path}")
    if schema_version == 1 and not isinstance(model, NativeStateSanosModel):
        raise ValueError("Legacy schema-1 FLEX checkpoints are native-only")
    expected = [day.training_fingerprint for day in market_days]
    if payload.get("market_fingerprints") != expected:
        raise ValueError(f"FLEX checkpoint market fingerprint mismatch in {path}")
    if schema_version == 2 and payload.get(
        "model_identity"
    ) != _model_checkpoint_identity(model):
        raise ValueError(f"FLEX checkpoint model identity mismatch in {path}")
    if schema_version == 2 and require_complete and payload.get(
        "phase"
    ) != "complete":
        raise ValueError(f"FLEX checkpoint is not complete in {path}")
    if isinstance(model, LatentStateSanosModel):
        if expected_supervision_target_identity is None:
            raise ValueError("Latent checkpoint load requires a native target identity")
        if payload.get(
            "supervision_target_identity"
        ) != expected_supervision_target_identity:
            raise ValueError(
                f"FLEX checkpoint native supervision target mismatch in {path}"
            )
        if payload.get("phase") == "complete":
            audits = payload.get("training_audits")
            if not isinstance(audits, dict) or not all(
                isinstance(audits.get(name), dict)
                and audits[name].get("converged") is True
                for name in ("supervised", "market")
            ):
                raise ValueError(
                    "Completed latent FLEX checkpoint has no successful "
                    f"convergence proof in {path}; rerun it"
                )
    model.load_state_dict(payload["model_state_dict"])
    if schema_version == 1:
        return {
            "schema_version": 1,
            "phase": "legacy_model",
        }
    return cast(TrainingState, payload)


def install_weights_only_warm_start(
    model: FlexTorchModel,
    path: Path,
) -> dict[str, object]:
    """Copy compatible learned parameters without replacing geometry buffers."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"Unsupported FLEX warm-start checkpoint schema in {path}")
    source_state = payload.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(f"FLEX warm-start checkpoint has no model state in {path}")
    target_parameters = dict(model.named_parameters())
    if not target_parameters:
        raise ValueError("FLEX warm-start target has no learned parameters")
    with torch.no_grad():
        for name, target in target_parameters.items():
            source = source_state.get(name)
            if not isinstance(source, torch.Tensor):
                raise ValueError(
                    f"FLEX warm-start checkpoint has no parameter {name!r} in {path}"
                )
            if source.shape != target.shape:
                raise ValueError(
                    f"FLEX warm-start parameter {name!r} has shape "
                    f"{tuple(source.shape)}, expected {tuple(target.shape)}"
                )
            if not torch.isfinite(source).all().item():
                raise ValueError(
                    f"FLEX warm-start parameter {name!r} is non-finite in {path}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    return {
        "source_phase": payload.get("phase"),
        "source_model_identity": json.loads(
            json.dumps(payload.get("model_identity"))
        ),
        "parameter_names": sorted(target_parameters),
    }
