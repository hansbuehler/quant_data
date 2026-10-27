from __future__ import annotations

import jax
import pytest

from quantdata.gromm2.runtime import configure_backend, gpu_preflight


@pytest.mark.parametrize("visible", ["", "-1", "none", "void"])
def test_gpu_requires_allocation_before_any_backend_probe(monkeypatch, visible: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(jax, "default_backend", lambda: pytest.fail("unallocated GPU was probed"))
    with pytest.raises(RuntimeError, match="gpuctl"):
        configure_backend("gpu")
    with pytest.raises(RuntimeError, match="gpuctl"):
        gpu_preflight()


def test_explicit_gpu_never_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-allocation")
    updates = []
    monkeypatch.setattr(jax.config, "update", lambda *args: updates.append(args))
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="refusing a silent fallback"):
        configure_backend("gpu")
    assert ("jax_platforms", "cuda") in updates


def test_invalid_backend_is_rejected_before_probe(monkeypatch) -> None:
    monkeypatch.setattr(jax, "default_backend", lambda: pytest.fail("invalid backend was probed"))
    with pytest.raises(ValueError, match="explicitly"):
        configure_backend("automatic")
