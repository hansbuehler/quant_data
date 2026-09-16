"""Orchestrates the full finite-grid / dense-mesh computation across both
models and all observation states, and writes the two declared export files:

    grid_visible.npz -- per model, per observation time: t, S_t, V_t, the
        grid (K_n), (T_m), C_mn, q_mn, ell_mn, the admissibility mask, and
        all parameters. This is what the downstream SANOS-reconstruction
        task is meant to load.
    dense_hidden.npz -- per model, per observation time: the dense (T,K)
        mesh, C, continuous q, continuous ell, and the L, Gamma and jump
        terms needed to recompute R_cont. Held out from the downstream task,
        used only to evaluate its reconstruction.

Naming convention inside each npz (flat dict of arrays; no native nesting):
    "{model}_t{i}_{field}" for per-(model, observation-time) arrays, plus
    shared "K_n"/"T_m" (finite grid) or "K_mesh"/"T_mesh" (dense grid) and a
    top-level "observation_times", "models" record for bookkeeping.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import oracle as orc

MODELS: dict[str, orc.JumpParams] = {
    "heston": orc.JumpParams(),
    "bates": orc.BATES_JUMPS_DEFAULT,
}


def compute_finite_grid(t: float, S: float, V: float, heston: orc.HestonParams, jumps: orc.JumpParams) -> dict[str, np.ndarray]:
    """All (interior strike, back-slab expiry) finite-grid diagnostics at one (t,S,V)."""
    n_list = orc.FIXED_STRIKE_INTERIOR_INDICES
    m_list = orc.FIXED_BACK_SLAB_INDICES
    shape = (len(n_list), len(m_list))
    out = {
        name: np.zeros(shape)
        for name in ("C", "q", "ell", "L_C", "L_q", "R")
    }
    out["mask"] = np.zeros(shape, dtype=bool)
    if jumps.lambda_ > 0.0:
        out["R_nojump"] = np.zeros(shape)

    for i, n_idx in enumerate(n_list):
        for j, m_idx in enumerate(m_list):
            r = orc.finite_grid_node_residual(n_idx, m_idx, t, S, V, heston, jumps)
            out["C"][i, j] = r.C
            out["q"][i, j] = r.q
            out["ell"][i, j] = r.ell
            out["mask"][i, j] = r.admissible
            out["L_C"][i, j] = r.L_C
            out["L_q"][i, j] = r.L_q
            out["R"][i, j] = r.R
            if jumps.lambda_ > 0.0:
                out["R_nojump"][i, j] = r.R_nojump

    out["K_n"] = np.array([orc.FIXED_STRIKES[n] for n in n_list])
    out["T_m"] = np.array([orc.FIXED_EXPIRIES[m] for m in m_list])
    return out


def compute_dense_grid(t: float, S: float, V: float, heston: orc.HestonParams, jumps: orc.JumpParams) -> dict[str, np.ndarray]:
    """Full dense-mesh continuous DLV diagnostics at one (t,S,V), vectorized."""
    res = orc.continuous_grid_residuals(
        orc.DENSE_STRIKE_MESH, orc.DENSE_EXPIRY_MESH, t, S, V, heston, jumps
    )
    out = {
        "K_mesh": res.K_mesh, "T_mesh": res.T_mesh,
        "C": res.C, "q": res.q, "ell": res.ell,
        "L_q": res.L_q, "L_ell": res.L_ell, "gamma_over_q": res.gamma_over_q,
        "jump_term": res.jump_term, "R": res.R, "R_normalized": res.R_normalized,
        "mask": res.admissible,
    }
    if res.R_nojump is not None:
        out["R_nojump"] = res.R_nojump
    return out


def _heston_params_dict(heston: orc.HestonParams) -> dict[str, float]:
    return {"kappa": heston.kappa, "theta": heston.theta, "xi": heston.xi, "rho": heston.rho}


def _jump_params_dict(jumps: orc.JumpParams) -> dict[str, float]:
    return {"lambda_": jumps.lambda_, "mu_j": jumps.mu_j, "delta_j": jumps.delta_j}


def export_all(
    output_dir: Path,
    heston: orc.HestonParams = orc.HESTON_DEFAULT,
    models: dict[str, orc.JumpParams] = MODELS,
    observation_times: tuple[float, ...] = orc.DEFAULT_OBSERVATION_TIMES,
    S0: float = orc.S0_DEFAULT,
    V0: float = orc.V0_DEFAULT,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Compute everything and write grid_visible.npz + dense_hidden.npz."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visible: dict[str, object] = {
        "observation_times": np.array(observation_times),
        "models": np.array(list(models.keys())),
        "S0": S0, "V0": V0,
        "heston_params_json": json.dumps(_heston_params_dict(heston)),
    }
    hidden: dict[str, object] = {
        "observation_times": np.array(observation_times),
        "models": np.array(list(models.keys())),
        "S0": S0, "V0": V0,
    }

    t0 = time.perf_counter()
    for model_name, jumps in models.items():
        visible[f"{model_name}_jump_params_json"] = json.dumps(_jump_params_dict(jumps))
        for i, t in enumerate(observation_times):
            if verbose:
                print(f"[{time.perf_counter()-t0:.1f}s] {model_name} t={t}: finite grid...")
            finite = compute_finite_grid(t, S0, V0, heston, jumps)
            for key, arr in finite.items():
                visible[f"{model_name}_t{i}_{key}"] = arr
            visible[f"{model_name}_t{i}_t"] = t
            visible[f"{model_name}_t{i}_S"] = S0
            visible[f"{model_name}_t{i}_V"] = V0

            if verbose:
                print(f"[{time.perf_counter()-t0:.1f}s] {model_name} t={t}: dense mesh...")
            dense = compute_dense_grid(t, S0, V0, heston, jumps)
            for key, arr in dense.items():
                hidden[f"{model_name}_t{i}_{key}"] = arr
            hidden[f"{model_name}_t{i}_t"] = t

    visible_path = output_dir / "grid_visible.npz"
    hidden_path = output_dir / "dense_hidden.npz"
    np.savez_compressed(visible_path, **visible)
    np.savez_compressed(hidden_path, **hidden)
    if verbose:
        print(f"[{time.perf_counter()-t0:.1f}s] wrote {visible_path} and {hidden_path}")
    return visible_path, hidden_path


if __name__ == "__main__":
    export_all(Path(__file__).parent)
