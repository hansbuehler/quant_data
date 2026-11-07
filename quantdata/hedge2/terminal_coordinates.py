"""Invertible training coordinates with ordinary, hook-free inference snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from quantdata.hedge2.terminal_policy import TerminalPolicy


@dataclass(frozen=True, kw_only=True)
class ActorInputCoordinates:
    mean: torch.Tensor
    matrix: torch.Tensor
    mode: str
    fitting_data_sha256: str
    fitting_summary: dict[str, int | float]
    inverse: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.mode not in ("diagonal", "whitened") or self.mean.ndim != 1
                or self.matrix.shape != (len(self.mean), len(self.mean)) or not len(self.mean)
                or self.mean.dtype != torch.float64 or self.matrix.dtype != self.mean.dtype
                or self.matrix.device != self.mean.device
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.matrix).all()
                or not isinstance(self.fitting_data_sha256, str) or len(self.fitting_data_sha256) != 64
                or any(value not in "0123456789abcdef" for value in self.fitting_data_sha256)):
            raise ValueError("Actor coordinates require finite aligned float64 inputs and training-data identity")
        counts = ("paths", "steps", "instruments")
        if (not isinstance(self.fitting_summary, dict)
                or set(self.fitting_summary) != {*counts, "probability_mass", "whitening_eigenvalue_floor"}
                or any(type(self.fitting_summary[name]) is not int or self.fitting_summary[name] < 1 for name in counts)
                or any(type(self.fitting_summary[name]) not in (int, float)
                    or not np.isfinite(self.fitting_summary[name])
                    for name in ("probability_mass", "whitening_eigenvalue_floor"))
                or abs(self.fitting_summary["probability_mass"] - 1.) > 512. * torch.finfo(torch.float64).eps
                or self.fitting_summary["whitening_eigenvalue_floor"] < 0):
            raise ValueError("Actor coordinates require finite, complete training-moment provenance")
        singular = torch.linalg.svdvals(self.matrix)
        tolerance = torch.finfo(self.matrix.dtype).eps * len(self.mean) * singular.max()
        if singular.min() <= tolerance:
            raise ValueError("Actor coordinates must be invertible without rank truncation")
        inverse = torch.linalg.inv(self.matrix)
        if not torch.isfinite(inverse).all():
            raise ValueError("Actor coordinate inverse is non-finite")
        object.__setattr__(self, "inverse", inverse)

    def record(self) -> dict[str, object]:
        return {"version": "actor-input-coordinates-v1", "mode": self.mode,
            "mean": self.mean.detach().clone(), "matrix": self.matrix.detach().clone(),
            "fitting_data_sha256": self.fitting_data_sha256, "fitting_summary": copy.deepcopy(self.fitting_summary)}

    @classmethod
    def from_record(cls, record: dict[str, object], *, policy: TerminalPolicy,
                    expected_data_sha256: str, expected_mode: str) -> ActorInputCoordinates:
        if (record.get("version") != "actor-input-coordinates-v1" or record.get("mode") != expected_mode
                or record.get("fitting_data_sha256") != expected_data_sha256
                or set(record) != {"version", "mode", "mean", "matrix", "fitting_data_sha256", "fitting_summary"}
                or not isinstance(record["mean"], torch.Tensor) or not isinstance(record["matrix"], torch.Tensor)
                or not isinstance(record["fitting_summary"], dict)):
            raise ValueError("Incompatible or incomplete actor-coordinate checkpoint")
        device = policy.affine.I0.weight.device
        return cls(mean=record["mean"].to(dtype=policy.fdtype, device=device),
            matrix=record["matrix"].to(dtype=policy.fdtype, device=device), mode=expected_mode,
            fitting_data_sha256=expected_data_sha256, fitting_summary=copy.deepcopy(record["fitting_summary"]))


def coordinate_checkpoint_binding(*, coordinates: dict[str, object], layout: tuple,
                                  optimization_state: Mapping[str, torch.Tensor],
                                  physical_state: Mapping[str, torch.Tensor],
                                  best_state: Mapping[str, torch.Tensor]) -> str:
    """Bind frozen coordinates, optimizer geometry and both raw inference snapshots."""
    metadata = {name: value for name, value in coordinates.items() if name not in ("mean", "matrix")}
    digest = hashlib.sha256(json.dumps({"coordinates": metadata, "layout": layout},
        sort_keys=True, allow_nan=False).encode())
    states = {"coordinates": {name: coordinates[name] for name in ("mean", "matrix")},
        "optimization": optimization_state, "physical": physical_state, "best": best_state}
    for group, state in states.items():
        for name, value in sorted(state.items()):
            if not isinstance(value, torch.Tensor):
                raise ValueError("Actor-coordinate checkpoint binding requires tensor states")
            array = value.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([group, name, array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class TerminalCoordinateMap:
    """Optimize normalized first layers while keeping raw inference weights explicit.

    W_opt=W_raw*C and b_opt=b_raw+W_raw*mean; the fixed input hook uses
    (x-mean)*C^{-T}. Parameter/direction maps are linear. Gradient mapping
    is the transpose map, not the parameter inverse. No model input, width,
    fee, payoff, action range or financial risk is changed.
    """

    def __init__(self, *, policy: TerminalPolicy, coordinates: ActorInputCoordinates) -> None:
        self.policy, self.coordinates = policy, coordinates
        self.networks = [(name, getattr(policy, name)) for name in ("affine", "deep") if getattr(policy, name) is not None]
        if (coordinates.mean.device != policy.affine.I0.weight.device or coordinates.mean.dtype != policy.fdtype
                or any(network.I0.weight.shape[1] != len(coordinates.mean) for _, network in self.networks)
                or any(network.I0.weight.device != coordinates.mean.device for _, network in self.networks)
                or len({id(network.I0) for _, network in self.networks}) != len(self.networks)):
            raise ValueError("Actor coordinates must match distinct first layers and policy precision/device")
        self.layout = tuple((name, tuple(parameter.shape), parameter.numel()) for name, parameter in policy.named_parameters())
        self.handles = []

    def state_dict(self, state: Mapping[str, torch.Tensor], *, to_physical: bool) -> dict[str, torch.Tensor]:
        result = copy.deepcopy(dict(state))
        for name, _ in self.networks:
            weight, bias = state[name + ".I0.weight"], state[name + ".I0.bias"]
            if to_physical:
                mapped = weight @ self.coordinates.inverse
                result[name + ".I0.weight"] = mapped
                result[name + ".I0.bias"] = bias - mapped @ self.coordinates.mean
            else:
                result[name + ".I0.weight"] = weight @ self.coordinates.matrix
                result[name + ".I0.bias"] = bias + weight @ self.coordinates.mean
        return result

    def vector(self, vector: torch.Tensor, *, to_physical: bool = True,
               gradient: bool = False) -> torch.Tensor:
        if (vector.ndim != 1 or len(vector) != sum(size for _, _, size in self.layout)
                or vector.dtype != self.coordinates.mean.dtype or vector.device != self.coordinates.mean.device
                or not torch.isfinite(vector).all() or gradient and not to_physical):
            raise ValueError("Actor-coordinate parameter/gradient vector is incompatible")
        entries, offset = {}, 0
        for name, shape, size in self.layout:
            entries[name] = vector[offset:offset + size].reshape(shape)
            offset += size
        mapped = dict(entries)
        for name, _ in self.networks:
            weight, bias = entries[name + ".I0.weight"], entries[name + ".I0.bias"]
            if gradient:
                mapped[name + ".I0.weight"] = weight @ self.coordinates.matrix.T + bias[:, None] * self.coordinates.mean[None, :]
            elif to_physical:
                raw = weight @ self.coordinates.inverse
                mapped[name + ".I0.weight"], mapped[name + ".I0.bias"] = raw, bias - raw @ self.coordinates.mean
            else:
                mapped[name + ".I0.weight"] = weight @ self.coordinates.matrix
                mapped[name + ".I0.bias"] = bias + weight @ self.coordinates.mean
        return torch.cat([mapped[name].reshape(-1) for name, _, _ in self.layout])

    def activate(self, *, map_weights: bool = True) -> None:
        if self.handles:
            raise RuntimeError("Actor input coordinates are already active")
        if map_weights:
            self.policy.load_state_dict(self.state_dict(self.policy.state_dict(), to_physical=False), strict=True)
        mean, inverse = self.coordinates.mean, self.coordinates.inverse
        for _, network in self.networks:
            self.handles.append(network.I0.register_forward_pre_hook(
                lambda module, arguments: ((arguments[0] - mean) @ inverse.T,)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def fit_actor_coordinates(*, policy: TerminalPolicy, evaluate: Callable[[], object],
                          weights: torch.Tensor, steps: int, instruments: int,
                          fitting_data_sha256: str, mode: str,
                          history_ids: np.ndarray | None = None) -> ActorInputCoordinates:
    """Stream training-only weighted moments, without retaining full actor-input arrays.

    Each path/time/asset has its original probability divided by steps and
    instruments. Shared actor rows receive the sum of their path masses.
    Statistics include current inventory feedback from the unchanged policy
    and are frozen for the entire subsequent fit. No target or future-return
    row is added to the actor's information set.
    """
    if (mode not in ("diagonal", "whitened") or policy.fdtype != torch.float64
            or weights.dtype != policy.fdtype or weights.device != policy.affine.I0.weight.device
            or weights.ndim != 1 or not len(weights) or not torch.isfinite(weights).all()
            or torch.any(weights < 0) or weights.sum() <= 0 or steps < 1 or instruments < 1
            or history_ids is not None and np.asarray(history_ids).shape != (len(weights), steps)):
        raise ValueError("Actor moment fitting needs float64 training probabilities and aligned history geometry")
    probabilities = weights / weights.sum()
    history_masses = None
    if history_ids is not None:
        history_masses = []
        cpu_probabilities = probabilities.detach().cpu().numpy()
        for step in range(steps):
            _, inverse = np.unique(np.asarray(history_ids)[:, step], return_inverse=True)
            history_masses.append(probabilities if len(np.unique(inverse)) == len(probabilities) else
                torch.tensor(np.bincount(inverse, weights=cpu_probabilities),
                    dtype=probabilities.dtype, device=probabilities.device))
    mass, mean, scatter, captured = None, None, None, 0

    def capture(module, arguments):
        nonlocal mass, mean, scatter, captured
        if captured >= steps:
            raise RuntimeError("Actor coordinate fitting observed unexpected extra first-layer calls")
        values = arguments[0].detach().reshape(-1, arguments[0].shape[-1])
        parent_mass = probabilities if history_masses is None else history_masses[captured]
        row_weights = parent_mass.repeat_interleave(instruments) / (steps * instruments)
        if len(values) != len(row_weights) or not torch.isfinite(values).all():
            raise RuntimeError("Actor coordinate moments lost the exact history/time/asset weighting")
        batch_mass = row_weights.sum()
        batch_mean = (values * row_weights[:, None]).sum(dim=0) / batch_mass
        centered = values - batch_mean
        batch_scatter = centered.T @ (centered * row_weights[:, None])
        if mass is None:
            mass, mean, scatter = batch_mass, batch_mean, batch_scatter
        else:
            total = mass + batch_mass
            difference = batch_mean - mean
            scatter = scatter + batch_scatter + torch.outer(difference, difference) * (mass * batch_mass / total)
            mean = mean + difference * (batch_mass / total)
            mass = total
        captured += 1

    handle = policy.affine.I0.register_forward_pre_hook(capture)
    try:
        evaluate()
    finally:
        handle.remove()
    if captured != steps or mass is None or abs(float(mass) - 1.) > 512. * torch.finfo(weights.dtype).eps:
        raise RuntimeError("Actor coordinate fitting did not cover the complete training law")
    covariance = scatter / mass
    covariance = .5 * (covariance + covariance.T)
    if mode == "diagonal":
        scales = torch.sqrt(torch.clamp_min(covariance.diag(), 0.))
        matrix = torch.diag(torch.where(scales > 1e-8, scales, 1.))
        floor = 0.
    else:
        values, vectors = torch.linalg.eigh(covariance)
        floor = 1e-6 * max(float(values.max()), 1.)
        matrix = (vectors * torch.sqrt(torch.clamp_min(values, floor))[None, :]) @ vectors.T
    return ActorInputCoordinates(mean=mean, matrix=matrix, mode=mode, fitting_data_sha256=fitting_data_sha256,
        fitting_summary={"paths": len(weights), "steps": steps, "instruments": instruments,
            "probability_mass": float(mass), "whitening_eigenvalue_floor": floor})
