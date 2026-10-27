"""Explicit compute backend selection for GROMM2 experiment workers."""

from __future__ import annotations

import os


def configure_backend(backend: str) -> str:
    """Reject unallocated GPU requests before initializing either accelerator."""
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be explicitly cpu or gpu")
    if backend == "gpu":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or visible.lower() in {"-1", "none", "void"}:
            raise RuntimeError("GPU sampling requires gpuctl-provided CUDA_VISIBLE_DEVICES; no GPU is probed without an allocation")
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cuda" if backend == "gpu" else "cpu")
    actual = jax.default_backend()
    if actual != backend:
        raise RuntimeError(f"requested {backend} backend but initialized {actual}; refusing a silent fallback")
    return actual


def gpu_preflight() -> dict[str, object]:
    """Check JAX float64 and Torch differentiation only inside an allocation."""
    configure_backend("gpu")
    import jax
    import jax.numpy as jnp
    import torch

    vector = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    jax_value = float(jnp.dot(vector, vector))
    if vector.dtype != jnp.float64 or jax_value != 5.0:
        raise RuntimeError("allocated JAX float64 preflight failed")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda:0", requires_grad=True)
    tensor.square().sum().backward()
    if tensor.grad is None or not torch.equal(tensor.grad, 2.0 * tensor):
        raise RuntimeError("allocated Torch gradient preflight failed")
    name = torch.cuda.get_device_name(0)
    del tensor, vector
    torch.cuda.empty_cache()
    return {"backend": "gpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "jax_device": str(jax.devices()[0]), "torch_device": name,
            "jax_float64_verified": True, "torch_gradient_verified": True}
