import os

from quantdata.sdlv.daily.front_spot.hedging.runners.jax_environment import (
    force_cpu_for_prepare_cli,
)


def test_prepare_stage_hides_gpu_before_jax_import(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    changed = force_cpu_for_prepare_cli(
        ("runner", "--stage", "prepare", "--artifact-root", "/tmp/artifacts")
    )

    assert changed
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_fit_stage_preserves_managed_gpu_environment(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    changed = force_cpu_for_prepare_cli(("runner", "--stage=fit"))

    assert not changed
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert "JAX_PLATFORMS" not in os.environ
