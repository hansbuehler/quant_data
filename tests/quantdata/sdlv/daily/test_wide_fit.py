from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.sdlv.daily import train
from quantdata.sdlv.daily.wide_fit import (
    CalibrationConfig,
    TrainConfig,
    _validate_resume_identity,
    checkpoint_density_validation,
    density_diagnostics,
    independent_density_validation,
)


def _data() -> dict[str, np.ndarray]:
    support = np.array([[0.4, 0.8, 1.0, 1.2, 1.4, 2.1]], dtype=np.float32)
    strikes = np.array([[0.9, 1.1, 0.7, 1.4]], dtype=np.float32)
    return dict(supports=support, raw_initial=np.full((1, 2, 4), np.log(.04), dtype=np.float32),
                model_times_y=np.array([1, 2], dtype=np.float32) / 252,
                strikes=strikes, is_call=np.array([[False, True, False, True]]),
                bids=np.zeros((1, 4), dtype=np.float32), asks=np.full((1, 4), .002, dtype=np.float32),
                mids=np.full((1, 4), .001, dtype=np.float32), lower=np.zeros((1, 4), dtype=np.int32),
                upper=np.ones((1, 4), dtype=np.int32), alpha=np.ones((1, 4), dtype=np.float32),
                before_first=np.zeros((1, 4), dtype=bool), mask=np.array([[True, True, True, False]]))


@pytest.fixture
def float32_training():
    # Audit modules enable float64 at collection time; exercise the training
    # kernel under the paper's float32 mode without changing other tests.
    jax = pytest.importorskip("jax")
    with jax.enable_x64(False):
        yield


@pytest.mark.parametrize("weighted", [False, True])
def test_exact_weighted_loss_and_finite_gradient(weighted: bool, float32_training) -> None:
    import jax.numpy as jnp

    data = _data()
    if weighted:
        data["quote_weights"] = np.array([[1, 1, .1, 0]], dtype=np.float32)
    device = {key: jnp.asarray(value) for key, value in data.items()}
    grad, _, decode, predict, blocks = train._make_loss_functions(
        device, mode="direct", cap_penalty_weight=0, quote_chunk_size=2)
    params = {"raw": device["raw_initial"]}
    density, initial = decode(params)
    values, objective = [], 0.0
    for index in range(2):
        block = {key: value[index] for key, value in blocks.items()}
        (loss, _), gradient = grad(params, block)
        assert np.all(np.isfinite(gradient["raw"]))
        values.append(np.asarray(predict(density, initial, block)))
        objective += float(loss)
    prices = np.concatenate(values, axis=1)
    assert np.all(prices >= -1e-10)
    half = .5 * (data["asks"] - data["bids"])
    errors = (np.maximum(prices - data["asks"], 0) + np.maximum(data["bids"] - prices, 0)
              + .05 * np.abs(prices - data["mids"])) / half
    weights = data.get("quote_weights", np.ones_like(prices)) * data["mask"]
    expected = np.mean(np.sum(errors * weights, axis=1) / np.sum(weights, axis=1))
    assert objective == pytest.approx(expected, rel=2e-6)
    assert density_diagnostics(data["supports"], np.asarray(density))["maximum_mass_error"] < 1e-5


def test_same_paper_learning_rate_schedule() -> None:
    schedule = TrainConfig()
    assert schedule.epochs == 1000
    assert schedule.learning_rate(0) == .02
    assert schedule.learning_rate(299) == pytest.approx(.001)
    assert schedule.learning_rate(300) == .005
    assert schedule.learning_rate(999) == pytest.approx(.00005)


def test_model_identity_and_two_template_grid() -> None:
    from quantdata.dysanos.flex.sanos import (
        make_shared_nms_node_sets,
        make_supports_from_total_vols,
    )

    config = CalibrationConfig()
    assert config.label == "Gromm-v1-4,3"
    geometry = train._geometry(config.num_inner_strikes, config.grid_nms)
    front, back = make_shared_nms_node_sets(num_inner_strikes=200, nms_range=config.grid_nms)
    assert front.shape == back.shape == (100,)
    assert front[0] == back[0] == -4 and front[-1] == back[-1] == 3
    supports = make_supports_from_total_vols(.15 * np.sqrt(geometry.model_expiries_y), geometry=geometry)
    assert supports.shape == (25, 202)
    np.testing.assert_array_equal(supports, np.broadcast_to(supports[0], supports.shape))
    assert np.all(np.diff(supports, axis=-1) > 0)


def test_reject_invalid_weight_and_density() -> None:
    with pytest.raises(ValueError):
        CalibrationConfig(tail_weight=0)
    with pytest.raises(ValueError):
        CalibrationConfig(core_nms=(1, 2))
    with pytest.raises(FloatingPointError):
        density_diagnostics(np.array([[1.0]]), np.array([[[np.nan]]]))


def test_independent_validation_uses_available_jax_x64_context() -> None:
    import jax

    data = _data()
    before = jax.config.x64_enabled
    diagnostics, dlvsq = independent_density_validation(data["supports"], data["raw_initial"])
    assert jax.config.x64_enabled == before
    assert dlvsq.dtype == np.float64
    assert diagnostics["minimum_probability"] >= -1e-13
    assert diagnostics["maximum_mass_error"] < 1e-10
    assert diagnostics["maximum_forward_error"] < 1e-10


def test_checkpoint_roundoff_requires_float64_confirmation_and_bounded_negative_mass() -> None:
    data = _data()
    eps = np.finfo(np.float32).eps
    initial = np.array([[-1.5 * eps, 0, 1 + 1.5 * eps, 0, 0, 0]], dtype=np.float32)
    densities = np.repeat(initial[:, None, :], 2, axis=1)
    approximate, precise = checkpoint_density_validation(data["supports"], data["raw_initial"], densities, initial)
    assert approximate["minimum_probability"] < -1e-7
    assert precise is not None and precise["minimum_probability"] >= -1e-13
    # A transition that amplifies negative mass must not be excused as rounding.
    bad = densities.copy()
    bad[..., 0] = -8 * eps
    with pytest.raises(FloatingPointError, match="exceeds initializer"):
        checkpoint_density_validation(data["supports"], data["raw_initial"], bad, initial)
    with pytest.raises(FloatingPointError, match="invariant failure"):
        checkpoint_density_validation(data["supports"], data["raw_initial"], densities * .9, initial)


def test_resume_repair_requires_exact_old_runner_and_unchanged_training_inputs() -> None:
    from quantdata.sdlv.daily import wide_fit

    runner = wide_fit.__file__
    saved = dict(dataset_sha256="original-data", train_config=dict(epochs=100),
                 code_sha256={runner: "old-runner", train.__file__: "same-kernel"})
    current = {**saved, "code_sha256": {**saved["code_sha256"], runner: "new-runner"}}
    _validate_resume_identity(saved, saved)
    _validate_resume_identity(saved, current, resume_code_sha256="old-runner")
    with pytest.raises(ValueError, match="Resume identity"):
        _validate_resume_identity(saved, current)
    with pytest.raises(ValueError, match="Resume identity"):
        _validate_resume_identity(saved, current, resume_code_sha256="wrong-runner")
    for changed in ({**current, "dataset_sha256": "different-data"},
                    {**current, "train_config": dict(epochs=200)},
                    {**current, "code_sha256": {**current["code_sha256"], train.__file__: "different-kernel"}}):
        with pytest.raises(ValueError, match="Resume identity"):
            _validate_resume_identity(saved, changed, resume_code_sha256="old-runner")


def test_prepare_uses_expiry_specific_nms_and_keeps_tail_quotes(monkeypatch) -> None:
    from quantdata.biz_calendar import NMS
    from quantdata.dysanos.flex import market

    geometry = train._geometry(200, (-4, 3))
    expiries = np.array([1, 25]) / 252
    indices = np.repeat([0, 1], 5)
    z = np.tile([-4, -3, 0, 2, 3], 2)
    strikes = NMS.from_nms(z, t=expiries[indices], vol=.2, eps_vol=.2)
    day = SimpleNamespace(expiries_y=expiries, expiry_index=indices,
                          model_atm_total_vols=.2 * np.sqrt(geometry.model_expiries_y),
                          atm_annual_vols=np.full(2, .2), strikes=strikes,
                          is_call=np.ones(10, dtype=bool), bids=np.zeros(10), asks=np.ones(10),
                          quote_business_dtes=expiries[indices] * 252, spot_ref=5000)

    def load(*args, **kwargs):
        assert kwargs["restrict_nms"] is False
        assert kwargs["override_cache_mode"] == "cacheonly"
        assert kwargs["geometry"].nms_range == (-4, 3)
        return day

    monkeypatch.setattr(market, "load_market_day", load)
    data = train._prepare_one_day(date(2025, 8, 29), ticker="SPX", num_inner_strikes=200,
                                  grid_nms_range=(-4, 3), core_nms_range=(-3.5, 2.5),
                                  tail_weight=.1, cache_only=True)
    assert len(data["strikes"]) == 10
    np.testing.assert_allclose(data["quote_nms"], z, atol=1e-5)
    np.testing.assert_allclose(data["quote_weights"], np.tile([.1, 1, 1, 1, .1], 2))
