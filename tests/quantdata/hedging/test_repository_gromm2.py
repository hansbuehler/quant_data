"""Operational GROMM2 checkpointing must not alter the repository's Adam loop."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.hedging.model_audit import repository_gromm2


def test_chunk_restart_preserves_iteration_and_adam_moments(tmp_path, monkeypatch):
    calibration = pytest.importorskip("openqdata.models.gromm.calibrate")
    calls = []

    def step(state, epoch, *args):
        calls.append(int(epoch))
        return tuple(value + increment for value, increment in zip(state, (.1, .2, .3), strict=True)), np.ones(2)

    monkeypatch.setattr(calibration, "_adam_step", step)
    monkeypatch.setattr(calibration, "_select", lambda *_: ())
    monkeypatch.setattr(calibration, "coverage", lambda *_: np.ones(2))
    health_calls = []

    def interrupted():
        health_calls.append(True)
        if len(health_calls) == 1:
            raise InterruptedError("stop after saved epoch two")

    monkeypatch.setattr(repository_gromm2, "health_check", interrupted)
    batch = SimpleNamespace(views=[None, None], geom=SimpleNamespace(
        config=SimpleNamespace(fit_blocks=[(1,)]), strikes=np.ones((1, 2))))
    arguments = dict(epochs=4, x0=np.zeros((2, 1, 2)), log_every=2, log=lambda *_: None)
    with pytest.raises(InterruptedError, match="saved epoch two"):
        repository_gromm2.checkpointed_fit(tmp_path, calibration.fit)(batch, **arguments)
    monkeypatch.setattr(repository_gromm2, "health_check", lambda: None)
    values, history = repository_gromm2.checkpointed_fit(tmp_path, calibration.fit)(batch, **arguments)
    np.testing.assert_allclose(values, .4)
    assert calls == [0, 1, 2, 3]
    assert history["epoch"] == [2, 4]
    with np.load(tmp_path / "chunks/0000.npz", allow_pickle=False) as saved:
        assert int(saved["completed_epochs"]) == 4
        np.testing.assert_allclose(saved["adam_m"], .8)
        np.testing.assert_allclose(saved["adam_v"], 1.2)


def test_observable_uses_model_backbone_and_current_spot_normalization():
    pytest.importorskip("openqdata.models.gromm.config")
    import jax.numpy as jnp

    from quantdata.hedging.model_audit.repository_gromm2_forecast import observable

    class Simulator:
        strikes = np.broadcast_to([.8, 1., 1.2], (12, 3))
        yvar = np.arange(13) * .04 / 252.

        def front(self, boundaries, brownian, tau):
            return jnp.asarray([.1, .8, .1]), jnp.zeros(3)

        def chain(self, logsig, front, loading, elapsed):
            return jnp.broadcast_to(front, (12, 3)), jnp.zeros((12, 3))

    state = (jnp.zeros((2, 12, 3)), jnp.zeros((2, 4)), jnp.zeros(2), jnp.asarray([0., np.log(1.1)]))
    surface, spots = observable(Simulator(), state, elapsed=1)
    assert surface.shape == (2, 49)
    assert np.isfinite(surface).all()
    np.testing.assert_allclose(spots, [1., 1.1])
    np.testing.assert_allclose(surface[0], surface[1], atol=1e-10)


def test_prepared_view_roundtrip_preserves_arrays_and_initializer(tmp_path):
    from dataclasses import fields
    from datetime import date

    calibration = pytest.importorskip("openqdata.models.gromm.calibrate")
    from quantdata.hedging.model_audit.repository_gromm2_inputs import read_view, write_view

    arguments = {field.name: np.arange(6, dtype=np.float64) for field in fields(calibration.View)}
    arguments.update(date=date(2024, 4, 1), kind="end", ratio=1.017,
        kernels0=np.arange(12, dtype=np.float32).reshape(2, 6),
        kernels1=np.arange(12, dtype=np.float32).reshape(2, 6) * .25,
        j0=np.array([0, 1], dtype=np.int32), j1=np.array([1, 2], dtype=np.int32),
        report=np.array([True, False]))
    view = calibration.View(**arguments)
    initial = np.arange(18, dtype=np.float64).reshape(3, 6) / 7.
    path = tmp_path / "view.npz"
    write_view(path, view, initial)
    restored, restored_initial = read_view(path)
    for field in fields(view):
        expected, actual = getattr(view, field.name), getattr(restored, field.name)
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    np.testing.assert_array_equal(restored_initial, initial)


def test_missing_preparation_arrays_are_not_marked_complete(tmp_path):
    import json

    from quantdata.hedging.model_audit.repository_gromm2_inputs import prepared_complete

    manifest = {"files_sha256": {"0000.npz": "digest"}}
    (tmp_path / "COMPLETE.json").write_text(json.dumps(manifest))
    assert prepared_complete(tmp_path) is None
    (tmp_path / "0000.npz").touch()
    assert prepared_complete(tmp_path) == manifest


def test_all_cpu_preparation_finishes_before_gpu_allocation(tmp_path, monkeypatch):
    import sys

    from quantdata.hedging.model_audit import repository_gromm2_inputs

    (tmp_path / "INPUTS_READY.json").write_text("{}")
    events = []
    monkeypatch.setattr(repository_gromm2, "origins", lambda *_: ("2024-04-01", "2024-04-02"))
    monkeypatch.setattr(repository_gromm2, "health_check", lambda: None)
    monkeypatch.setattr(repository_gromm2_inputs, "prepare_origin", lambda run, origin, **kwargs: events.append(origin))
    monkeypatch.setattr(repository_gromm2, "wait_gpu", lambda *args, **kwargs: events.append("wait_gpu"))
    monkeypatch.setattr(repository_gromm2, "run_owned", lambda *args, **kwargs: events.append("run_gpu"))
    monkeypatch.setattr(sys, "argv", ["repository_gromm2", "--run", str(tmp_path),
        "--reference", str(tmp_path), "--wait-and-train", "--prepare-workers", "1"])
    repository_gromm2.main()
    assert events == ["2024-04-01", "2024-04-02", "wait_gpu", "run_gpu"]
