from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from cdxcore.bs import bs

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex import market as flex_market
from quantdata.dysanos.flex import run_production_gate, run_shared_strikes
from quantdata.dysanos.flex import training as flex_training
from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexGeometryConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    ForwardMuFlexGeometryConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    LatentDecoderConfig,
    NativeStateConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lp import (
    FlexLpConfig,
    fit_flex_density_lp,
    reprice_dlvsqs_with_torch,
)
from quantdata.dysanos.flex.market import (
    FlexMarketDay,
    MarketEligibilityAudit,
    market_day_from_payload,
)
from quantdata.dysanos.flex.metrics import (
    expiry_excess_statistics,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.model import (
    LatentStateSanosModel,
    NativeParameters,
    NativeStateSanosModel,
    bound_native_raw,
    dlvsq_cap_penalty,
    front_term_eta_coordinates,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.sanos import (
    FlexSanosDecoder,
    make_inner_nms_nodes,
    make_supports_from_total_vols,
    sanos_time_interpolation_numpy,
    total_vols_from_forward_vols_numpy,
)
from quantdata.dysanos.flex.training import (
    fit_latent_states,
    fit_native_states,
    initialize_pca_decoder_,
    install_weights_only_warm_start,
    market_native_raw_initial_states,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_default_geometry_is_the_exact_project_geometry() -> None:
    geometry = FlexGeometryConfig()
    np.testing.assert_allclose(
        geometry.model_business_dtes,
        4.8 + 5.0 * np.arange(14),
    )
    assert geometry.model_business_dtes[-1] == pytest.approx(69.8)
    assert geometry.model_expiries_y[0] == pytest.approx(4.8 / DAYS_PER_YEAR)
    assert geometry.nms_range == (-3.0, 2.0)
    assert geometry.num_inner_strikes == 200
    assert geometry.native_state_size == 2_815
    assert geometry.nms_epsilon_y == pytest.approx(NMS.EPS)

    explicit = FlexGeometryConfig(
        num_model_expiries=8,
        explicit_model_business_dtes=(2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0),
    )
    np.testing.assert_array_equal(
        explicit.model_business_dtes,
        np.array([2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0]),
    )
    assert explicit.native_state_size == 1_609


def test_market_dates_report_preserves_the_exact_completed_universe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "completed.json"
    path.write_text(
        json.dumps({"market_dates": ["2024-01-02", "2024-01-04"]}),
        encoding="utf-8",
    )
    assert run_production_gate._market_dates_from_report(path) == (
        date(2024, 1, 2),
        date(2024, 1, 4),
    )

    path.write_text(
        json.dumps({"market_dates": ["2024-01-04", "2024-01-02"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_production_gate._market_dates_from_report(path)

def test_nms_nodes_and_supports_have_one_atm_and_two_exact_boundaries() -> None:
    geometry = FlexGeometryConfig()
    nodes = make_inner_nms_nodes(
        num_inner_strikes=geometry.num_inner_strikes,
        nms_range=geometry.nms_range,
    )
    assert nodes.shape == (200,)
    assert nodes[0] == -3.0
    assert nodes[-1] == 2.0
    assert np.sum(nodes == 0.0) == 1

    forward_vols = np.linspace(0.15, 0.25, 14)
    total_vols = total_vols_from_forward_vols_numpy(
        forward_vols,
        geometry=geometry,
    )
    expected = np.cumsum(
        forward_vols
        * np.sqrt(np.diff(geometry.model_expiries_y, prepend=0.0))
    )
    np.testing.assert_allclose(total_vols, expected)
    supports = make_supports_from_total_vols(total_vols, geometry=geometry)
    assert supports.shape == (14, 202)
    np.testing.assert_allclose(supports[:, 0], 0.5 * supports[:, 1])
    np.testing.assert_allclose(supports[:, -1], 1.5 * supports[:, -2])
    np.testing.assert_array_equal(np.sum(supports == 1.0, axis=1), 1)


def test_shared_strikes_merge_half_front_half_last_and_float_only_by_day() -> None:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    assert geometry.native_state_size == 2_828

    total_vols_a = total_vols_from_forward_vols_numpy(
        np.linspace(0.15, 0.25, geometry.num_model_expiries),
        geometry=geometry,
    )
    total_vols_b = total_vols_from_forward_vols_numpy(
        np.linspace(0.25, 0.35, geometry.num_model_expiries),
        geometry=geometry,
    )
    supports_a = make_supports_from_total_vols(total_vols_a, geometry=geometry)
    supports_b = make_supports_from_total_vols(total_vols_b, geometry=geometry)

    np.testing.assert_allclose(
        supports_a,
        np.broadcast_to(supports_a[:1], supports_a.shape),
    )
    np.testing.assert_allclose(
        supports_b,
        np.broadcast_to(supports_b[:1], supports_b.shape),
    )
    assert not np.allclose(supports_a[0], supports_b[0])

    inner_log_strikes = np.log(supports_a[0, 1:-1])
    assert inner_log_strikes.shape == (geometry.num_inner_strikes,)
    assert np.sum(inner_log_strikes == 0.0) == 1
    assert np.all(np.diff(inner_log_strikes) > 0.0)


def test_front_term_eta_geometry_has_two_coordinates_and_exact_raw_round_trip() -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=4,
        explicit_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
    )
    assert geometry.native_state_size == 4 * 7 + 2

    dlvsqs = torch.linspace(0.03, 0.10, 24, dtype=torch.float64).reshape(1, 4, 6)
    forward_vols = torch.tensor(
        ((0.18, 0.20, 0.22, 0.24),),
        dtype=torch.float64,
    )
    eta = torch.tensor(
        ((0.31, 0.43, 0.43, 0.43),),
        dtype=torch.float64,
    )
    raw = raw_from_native_parameters(
        NativeParameters(dlvsqs=dlvsqs, forward_vols=forward_vols, eta=eta),
        geometry=geometry,
    )
    parameters = bound_native_raw(raw, geometry=geometry)

    assert raw.shape == (1, geometry.native_state_size)
    torch.testing.assert_close(parameters.dlvsqs, dlvsqs)
    torch.testing.assert_close(parameters.forward_vols, forward_vols)
    torch.testing.assert_close(parameters.eta, eta)


def test_front_term_eta_projection_preserves_final_backbone_total_volatility() -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=4,
        explicit_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
    )
    forward_vols = torch.tensor(
        ((0.18, 0.20, 0.22, 0.24),),
        dtype=torch.float64,
    )
    eta = torch.tensor(
        ((0.31, 0.35, 0.42, 0.47),),
        dtype=torch.float64,
    )
    coordinates = front_term_eta_coordinates(
        eta,
        forward_vols,
        geometry=geometry,
    )
    projected_eta = torch.cat(
        (coordinates[..., :1], coordinates[..., 1:].expand(-1, 3)),
        dim=-1,
    )
    expiries = torch.as_tensor(geometry.model_expiries_y, dtype=torch.float64)
    sqrt_dts = torch.sqrt(torch.diff(expiries, prepend=torch.zeros(1)))

    torch.testing.assert_close(
        torch.sum(projected_eta * forward_vols * sqrt_dts, dim=-1),
        torch.sum(eta * forward_vols * sqrt_dts, dim=-1),
    )


def test_front_term_native_audit_reports_exponential_dlv_values(tmp_path: Path) -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=2,
        explicit_model_business_dtes=(1.0, 5.0),
    )
    dlvsqs = torch.tensor(
        (
            (
                (0.02, 0.03, 0.04, 0.05, 0.06, 0.07),
                (0.08, 0.09, 0.10, 0.11, 0.12, 0.13),
            ),
        ),
        dtype=torch.float64,
    )
    raw = raw_from_native_parameters(
        NativeParameters(
            dlvsqs=dlvsqs,
            forward_vols=torch.tensor(((0.20, 0.24),), dtype=torch.float64),
            eta=torch.tensor(((0.25, 0.35),), dtype=torch.float64),
        ),
        geometry=geometry,
    )
    checkpoint = tmp_path / "native.pt"
    torch.save(
        {"model_state_dict": {"daily_raw_embedding.weight": raw}},
        checkpoint,
    )

    audit = flex_training.native_checkpoint_state_audit(
        checkpoint,
        geometry=geometry,
    )

    assert audit["coordinates"]["dlvsq"]["minimum"] == pytest.approx(0.02)
    assert audit["coordinates"]["dlvsq"]["median"] == pytest.approx(0.075)
    assert audit["coordinates"]["dlvsq"]["maximum"] == pytest.approx(0.13)


def test_fixed_relative_strikes_do_not_move_with_state_or_expiry() -> None:
    inner = (0.8, 0.9, 1.0, 1.1, 1.25)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=len(inner),
        num_model_expiries=4,
        explicit_model_business_dtes=(5.0, 10.0, 15.0, 20.0),
        inner_relative_strikes=inner,
    )
    total_vols_a = total_vols_from_forward_vols_numpy(
        np.linspace(0.15, 0.25, 4),
        geometry=geometry,
    )
    total_vols_b = total_vols_from_forward_vols_numpy(
        np.linspace(0.35, 0.45, 4),
        geometry=geometry,
    )
    expected = np.array((0.4, *inner, 1.875))
    supports_a = make_supports_from_total_vols(total_vols_a, geometry=geometry)
    supports_b = make_supports_from_total_vols(total_vols_b, geometry=geometry)
    np.testing.assert_allclose(supports_a, np.broadcast_to(expected, (4, 7)))
    np.testing.assert_allclose(supports_b, supports_a)

    decoder = FlexSanosDecoder(
        BaseEnvironment(precision=64, device="cpu", debug=True),
        geometry=geometry,
    )
    _, supports = decoder._support_strikes_from_forward_vols(
        torch.tensor(
            ((0.15, 0.20, 0.25, 0.30), (0.35, 0.40, 0.45, 0.50)),
            dtype=torch.float64,
        )
    )
    torch.testing.assert_close(
        supports,
        torch.tensor(expected, dtype=torch.float64).view(1, 1, -1).expand(2, 4, -1),
    )

    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=torch.full(
            (1, geometry.num_model_expiries, geometry.num_inner_strikes),
            0.20**2,
            dtype=env.fdtype,
        ),
        forward_vols=torch.as_tensor(
            np.linspace(0.15, 0.25, geometry.num_model_expiries)[None, :],
            dtype=env.fdtype,
        ),
        eta=torch.full(
            (1, geometry.num_model_expiries),
            0.20,
            dtype=env.fdtype,
        ),
    )
    torch.testing.assert_close(
        surface.support_strikes[0],
        torch.as_tensor(supports_a, dtype=env.fdtype),
    )


def test_sanos_time_rule_interpolates_and_left_extrapolates() -> None:
    geometry = FlexGeometryConfig()
    nodes = geometry.model_expiries_y
    query = np.array([0.0, 0.5 * nodes[0], nodes[0], 0.5 * (nodes[0] + nodes[1])])
    interpolation = sanos_time_interpolation_numpy(query, geometry=geometry)
    np.testing.assert_array_equal(interpolation.before_first, [True, True, True, False])
    np.testing.assert_allclose(interpolation.upper_weight, [0.0, 0.5, 1.0, 0.5])
    np.testing.assert_array_equal(interpolation.upper_index, [0, 0, 0, 1])
    np.testing.assert_array_equal(interpolation.lower_index, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="beyond its final model expiry"):
        sanos_time_interpolation_numpy(
            np.array([nodes[-1] + 1.0e-6]),
            geometry=geometry,
        )


def test_torch_decoder_preserves_martingale_and_has_finite_0dte_gradients() -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=9, num_model_expiries=3)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.full(
        (3, 9),
        0.20**2,
        dtype=env.fdtype,
        requires_grad=True,
    )
    forward_vols = torch.full(
        (3,),
        0.20,
        dtype=env.fdtype,
        requires_grad=True,
    )
    eta = torch.tensor(0.40, dtype=env.fdtype, requires_grad=True)
    surface = decoder(dlvsqs=dlvsqs, forward_vols=forward_vols, eta=eta)
    np.testing.assert_allclose(
        surface.densities.detach().sum(dim=-1).numpy(),
        1.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (surface.densities * surface.support_strikes)
        .detach()
        .sum(dim=-1)
        .numpy(),
        1.0,
        atol=1.0e-12,
    )
    expiries = torch.tensor(
        [0.0, 0.5 * geometry.model_expiries_y[0], geometry.model_expiries_y[1]],
        dtype=env.fdtype,
    )
    strikes = torch.tensor([0.9, 1.0, 1.1], dtype=env.fdtype)
    prices = decoder.price_calls(
        surface,
        expiries_y=expiries,
        strikes=strikes,
    )
    assert prices[0].item() == pytest.approx(0.1, abs=1.0e-12)
    prices.sum().backward()
    assert torch.all(torch.isfinite(dlvsqs.grad))
    assert torch.all(torch.isfinite(forward_vols.grad))
    assert torch.isfinite(eta.grad)


def test_production_geometry_float32_decode_and_gradients_are_finite() -> None:
    geometry = FlexGeometryConfig()
    env = BaseEnvironment(precision=32, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.full(
        (2, geometry.num_model_expiries, geometry.num_inner_strikes),
        0.20**2,
        dtype=env.fdtype,
        requires_grad=True,
    )
    forward_vols = torch.full(
        (2, geometry.num_model_expiries),
        0.20,
        dtype=env.fdtype,
        requires_grad=True,
    )
    eta = torch.full((2,), 0.30, dtype=env.fdtype, requires_grad=True)
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    expiries = torch.tensor(
        [
            [
                0.0,
                geometry.model_expiries_y[0],
                geometry.model_expiries_y[-1],
            ],
            [
                0.5 * geometry.model_expiries_y[0],
                geometry.model_expiries_y[4],
                geometry.model_expiries_y[-1],
            ],
        ],
        dtype=env.fdtype,
    )
    strikes = torch.tensor(
        [[0.8, 1.0, 1.2], [0.9, 1.1, 1.3]],
        dtype=env.fdtype,
    )
    prices = decoder.price_calls_batch(
        surface,
        expiries_y=expiries,
        strikes=strikes,
    )
    prices.sum().backward()

    assert surface.support_strikes.shape == (2, 14, 202)
    assert surface.densities.shape == (2, 14, 202)
    torch.testing.assert_close(
        surface.densities.detach().sum(dim=-1),
        torch.ones((2, 14), dtype=env.fdtype),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        (surface.densities * surface.support_strikes)
        .detach()
        .sum(dim=-1),
        torch.ones((2, 14), dtype=env.fdtype),
        atol=1.0e-5,
        rtol=0.0,
    )
    assert torch.all(torch.isfinite(prices))
    assert torch.all(torch.isfinite(dlvsqs.grad))
    assert torch.all(torch.isfinite(forward_vols.grad))
    assert torch.all(torch.isfinite(eta.grad))


def test_metrics_use_real_half_spread_and_nominal_am_dte_buckets() -> None:
    fit = np.array([1.0, 1.3, 0.6, 0.5])
    bid = np.array([0.9, 1.0, 0.8, 0.4])
    ask = np.array([1.1, 1.2, 1.0, 0.6])
    statistics = expiry_excess_statistics(
        fit,
        bid,
        ask,
        np.array([0.2, 1.2, 2.2, 3.2]),
    )
    assert statistics.all.num_zero == 2
    assert statistics.all.percent_zero == 50.0
    assert statistics.all.positive_mean == pytest.approx(1.5)
    assert statistics.all.positive_median == pytest.approx(1.5)
    assert statistics.all.surface_mean == pytest.approx(0.75)
    assert statistics.all.surface_median == pytest.approx(0.5)
    assert statistics.dte_0.num_zero == 1
    assert statistics.dte_1.positive_mean == pytest.approx(1.0)
    assert statistics.dte_2.positive_mean == pytest.approx(2.0)

    surface_fit = surface_fit_statistics(
        fit,
        np.array([1.0, 1.1, 0.7, 0.5]),
        bid,
        ask,
    )
    assert surface_fit.half_spread_mae == pytest.approx(0.75)
    assert surface_fit.half_spread_median_absolute_error == pytest.approx(0.5)
    assert surface_fit.normalized_price_mae == pytest.approx(0.075)
    assert surface_fit.normalized_price_median_absolute_error == pytest.approx(
        0.05
    )
    assert surface_fit.normalized_price_rmse == pytest.approx(np.sqrt(0.0125))
    assert surface_fit.normalized_price_bias == pytest.approx(0.025)

    contained = expiry_excess_statistics(
        np.array([1.0]),
        np.array([0.9]),
        np.array([1.1]),
        np.array([0.2]),
    )
    assert contained.all.positive_mean == 0.0
    assert contained.all.positive_median == 0.0

def test_failed_final_evaluation_is_a_hard_failure_without_fallback(
    monkeypatch,
) -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    model = NativeStateSanosModel(
        env,
        num_days=1,
        config=NativeStateConfig(label="no-selection-fallback", geometry=geometry),
    )
    with torch.no_grad():
        model.daily_raw_states.add_(1.0)
    candidate_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    statistics = expiry_excess_statistics(
        np.array([1.0]),
        np.array([0.9]),
        np.array([1.1]),
        np.array([0.2]),
    )
    calls = 0

    def fail_evaluation(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("non-finite candidate")

    monkeypatch.setattr(flex_training, "evaluate_model", fail_evaluation)
    with pytest.raises(
        flex_training.TrainingConvergenceError,
        match="no fallback was applied",
    ):
        flex_training._evaluate_fine_tuned_model(
            model,
            (),
            env=env,
            objective_config=FlexObjectiveConfig(),
            day_batch_size=1,
            pre_training_score={"objective": 1.0, "percent_zero": 50.0},
            phase="test fine-tuning",
        )
    assert calls == 1
    for name, value in candidate_state.items():
        torch.testing.assert_close(
            model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_training_convergence_requires_a_trailing_plateau() -> None:
    config = FlexTrainConfig(
        convergence_window=2,
        convergence_relative_tolerance=1.0e-3,
    )
    converged = flex_training.training_convergence_audit(
        [
            {"objective": 1.001},
            {"objective": 1.0004},
            {"objective": 1.0002},
        ],
        metric="objective",
        train_config=config,
    )
    assert converged["converged"] is True

    improving = flex_training.training_convergence_audit(
        [
            {"objective": 2.0},
            {"objective": 1.5},
            {"objective": 1.0},
        ],
        metric="objective",
        train_config=config,
    )
    assert improving["converged"] is False
    assert improving["reason"] == "trailing_window_not_converged"


def test_fine_tuning_objective_regression_is_a_hard_failure(monkeypatch) -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    model = NativeStateSanosModel(
        env,
        num_days=1,
        config=NativeStateConfig(label="objective-regression", geometry=geometry),
    )
    statistics = expiry_excess_statistics(
        np.array([1.0]),
        np.array([0.9]),
        np.array([1.1]),
        np.array([0.2]),
    )
    regressed = flex_training.FitEvaluation(
        objective=2.0,
        statistics=statistics,
        surface_fit=surface_fit_statistics(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.9]),
            np.array([1.1]),
        ),
        fits=(),
    )
    monkeypatch.setattr(
        flex_training,
        "evaluate_model",
        lambda *args, **kwargs: regressed,
    )
    with pytest.raises(
        flex_training.TrainingConvergenceError,
        match="objective regressed from 1 to 2; no fallback was applied",
    ):
        flex_training._evaluate_fine_tuned_model(
            model,
            (),
            env=env,
            objective_config=FlexObjectiveConfig(),
            day_batch_size=1,
            pre_training_score={"objective": 1.0, "percent_zero": 50.0},
            phase="test fine-tuning",
        )


class _FakeArbFreePricer:
    def interpolate_atm_vols(self, expiries: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(expiries).shape, 0.20)


def _market_payload() -> dict[str, object]:
    expiries = np.array([0.2, 4.2, 69.2, 70.2]) / DAYS_PER_YEAR
    strikes = [np.array([0.5, 0.9, 1.0, 1.1, 2.0]) for _ in expiries]
    call_fit = [
        bs.price(k=strike, vol=0.2, sqrtT=np.sqrt(expiry))
        for strike, expiry in zip(strikes, expiries, strict=True)
    ]
    put_fit = [
        call - 1.0 + strike
        for call, strike in zip(call_fit, strikes, strict=True)
    ]
    active = [np.ones(5, dtype=bool) for _ in expiries]
    zeros = [np.zeros(5) for _ in expiries]
    return {
        "d": date(2024, 3, 5),
        "af_pricer": {"serialized": "reference"},
        "market": {
            "uid": "test-market",
            "expiries": expiries,
            "expiry_dates": ["2024-03-05", "2024-03-11", "2024-06-12", "2024-06-13"],
            "expiry_ids": np.arange(4),
            "is_am": np.ones(4, dtype=bool),
            "strikes": strikes,
            "is_american": [np.zeros(5, dtype=bool) for _ in expiries],
            "af_mid_available": active,
            "call_active": active,
            "put_active": active,
            "call_bids": [np.maximum(value - 0.001, 0.0) for value in call_fit],
            "call_asks": [value + 0.001 for value in call_fit],
            "put_bids": [np.maximum(value - 0.001, 0.0) for value in put_fit],
            "put_asks": [value + 0.001 for value in put_fit],
            "af_call_mids": call_fit,
            "af_put_mids": put_fit,
            "call_log_volumes": zeros,
            "call_log_open_interests": zeros,
            "put_log_volumes": zeros,
            "put_log_open_interests": zeros,
        },
    }


def test_market_adapter_uses_fit_targets_and_exact_flex_eligibility(monkeypatch) -> None:
    calls = 0

    def restore(payload):
        nonlocal calls
        calls += 1
        assert payload == {"serialized": "reference"}
        return _FakeArbFreePricer()

    monkeypatch.setattr(flex_market.ArbFreePricer, "from_dict", restore)
    day = market_day_from_payload(_market_payload())
    assert day is not None
    assert calls == 1
    assert np.max(day.quote_business_dtes) <= 69.8
    assert 70.2 not in day.business_dtes
    assert np.any(day.is_call & (day.strikes < 1.0))
    assert np.any((~day.is_call) & (day.strikes > 1.0))
    np.testing.assert_allclose(
        day.call_equivalent_fit_prices[~day.is_call],
        day.fit_prices[~day.is_call] + 1.0 - day.strikes[~day.is_call],
    )
    assert np.all(
        day.strikes
        >= day.strike_lower_bounds[day.expiry_index] - 1.0e-15
    )
    assert np.all(
        day.strikes
        <= day.strike_upper_bounds[day.expiry_index] + 1.0e-15
    )


def test_market_adapter_excludes_configured_business_dte(monkeypatch) -> None:
    monkeypatch.setattr(
        flex_market.ArbFreePricer,
        "from_dict",
        lambda payload: _FakeArbFreePricer(),
    )
    unfiltered = market_day_from_payload(_market_payload())
    day = market_day_from_payload(
        _market_payload(),
        config=flex_market.FlexMarketConfig(excluded_business_dtes=(0.2,)),
    )
    assert unfiltered is not None
    assert day is not None
    assert not np.any(np.isclose(day.business_dtes, 0.2, atol=1.0e-4))
    excluded_quotes = np.sum(
        np.isclose(unfiltered.quote_business_dtes, 0.2, atol=1.0e-4)
    )
    assert day.num_quotes == unfiltered.num_quotes - excluded_quotes


def test_shorter_geometry_reuses_canonical_weekly_market_cache(monkeypatch) -> None:
    geometry = FlexGeometryConfig(
        num_model_expiries=9,
        explicit_model_business_dtes=(
            1.0,
            3.0,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            40.0,
            60.0,
        ),
    )
    captured: dict[str, object] = {}
    expected_payload = {"market": "weekly-cache"}
    expected_day = object()

    def smooth_interpolated_market(*args, **kwargs):
        captured["options_filter"] = kwargs["options_filter"]
        return expected_payload

    def adapt(payload, **kwargs):
        assert payload is expected_payload
        assert kwargs["geometry"] is geometry
        return expected_day

    monkeypatch.setattr(
        flex_market,
        "smooth_interpolated_market",
        smooth_interpolated_market,
    )
    monkeypatch.setattr(flex_market, "market_day_from_payload", adapt)

    result = flex_market.load_market_day(
        "SPX",
        date(2024, 3, 5),
        geometry=geometry,
        verbose=None,
    )

    assert result is expected_day
    options_filter = captured["options_filter"]
    assert options_filter.max_business_days_to_expiry == 70


def test_native_weights_warm_start_preserves_new_expiry_buffers(
    tmp_path: Path,
) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source_geometry = FlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=3,
        explicit_model_business_dtes=(1.0, 3.0, 5.0),
    )
    target_geometry = FlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=3,
        explicit_model_business_dtes=(0.8, 2.8, 4.8),
    )
    source = NativeStateSanosModel(
        env,
        num_days=2,
        config=NativeStateConfig(label="source", geometry=source_geometry),
    )
    target = NativeStateSanosModel(
        env,
        num_days=2,
        config=NativeStateConfig(label="target", geometry=target_geometry),
    )
    with torch.no_grad():
        source.daily_raw_states.copy_(
            torch.arange(
                source.daily_raw_states.numel(),
                dtype=env.fdtype,
            ).reshape_as(source.daily_raw_states)
        )
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "schema_version": 2,
            "phase": "market",
            "model_identity": {"source": True},
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )

    audit = install_weights_only_warm_start(target, checkpoint)

    torch.testing.assert_close(target.daily_raw_states, source.daily_raw_states)
    torch.testing.assert_close(
        target.sanos_decoder.model_expiries_y,
        torch.as_tensor(target_geometry.model_expiries_y, dtype=env.fdtype),
    )
    assert not torch.equal(
        target.sanos_decoder.model_expiries_y,
        source.sanos_decoder.model_expiries_y,
    )
    assert audit["source_phase"] == "market"
    assert audit["parameter_names"] == ["daily_raw_embedding.weight"]


def test_market_loader_records_an_isolated_bad_day(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("reference timeout")

    monkeypatch.setattr(run_production_gate, "load_market_day", fail)
    outcome = run_production_gate._load_market_job(
        "SPX",
        date(2024, 3, 5),
        geometry=FlexGeometryConfig(),
        market_config=flex_market.FlexMarketConfig(),
    )
    assert outcome.day is None
    assert outcome.error == "RuntimeError: reference timeout"


def test_compression_restart_loads_completed_sizes_and_resumes_active_size(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "latent.pt"
    train_config = FlexTrainConfig()
    assert (
        run_production_gate._latent_restart_config(
            train_config,
            checkpoint_path,
        )
        is train_config
    )

    torch.save(
        {"schema_version": 2, "phase": "market"},
        checkpoint_path,
    )
    assert (
        run_production_gate._latent_restart_config(
            train_config,
            checkpoint_path,
        )
        is train_config
    )

    torch.save(
        {"schema_version": 2, "phase": "complete"},
        checkpoint_path,
    )
    completed_config = run_production_gate._latent_restart_config(
        train_config,
        checkpoint_path,
    )
    assert completed_config.just_load
    assert not completed_config.warm_start

    warm_start_config = replace(train_config, warm_start=True)
    assert (
        run_production_gate._latent_restart_config(
            warm_start_config,
            checkpoint_path,
        )
        is warm_start_config
    )


def test_expiry_benchmark_rejects_an_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expiry benchmark variants"):
        run_production_gate.run_production_gates(
            ticker="SPX",
            dates=(date(2024, 3, 5),),
            geometry=ForwardMuFlexGeometryConfig(),
            market_config=flex_market.FlexMarketConfig(),
            objective_config=FlexObjectiveConfig(),
            train_config=FlexTrainConfig(),
            supervised_train_config=FlexTrainConfig(),
            workers=1,
            output_directory=tmp_path / "reports",
            checkpoint_directory=tmp_path / "checkpoints",
            label="invalid-expiry-variant",
            run_compression=True,
            expiry_benchmark=True,
            expiry_benchmark_variants=("triangular-uniform", "unknown"),
        )


def test_rebalanced_expiry_allocations_preserve_the_140_state_budget() -> None:
    expected = {
        "triangular-conservative": (
            14,
            15,
            14,
            13,
            12,
            10,
            10,
            10,
            7,
            7,
            7,
            7,
            7,
            7,
        ),
        "triangular-stronger": (
            16,
            19,
            17,
            15,
            13,
            10,
            10,
            10,
            5,
            5,
            5,
            5,
            5,
            5,
        ),
    }

    for variant, allocation in expected.items():
        assert run_production_gate._EXPIRY_TRIANGULAR_ALLOCATIONS[variant] == allocation
        config = run_production_gate._triangular_pca_decoder_config(
            label=variant,
            state_sizes_by_expiry=allocation,
            geometry=ForwardMuFlexGeometryConfig(),
        )
        assert config.state_size == 140
        assert config.state_sizes_by_expiry == allocation


def test_shared_strikes_strong_allocation_uses_200_states() -> None:
    allocation = run_production_gate._EXPIRY_TRIANGULAR_ALLOCATIONS[
        "triangular-strong-200"
    ]
    assert allocation == (
        23,
        27,
        25,
        22,
        19,
        14,
        14,
        14,
        7,
        7,
        7,
        7,
        7,
        7,
    )
    config = run_production_gate._triangular_pca_decoder_config(
        label="shared-strong-200",
        state_sizes_by_expiry=allocation,
        geometry=SharedStrikeCappedUnboundedDlvFlexGeometryConfig(),
    )
    assert config.state_size == 200
    assert config.state_sizes_by_expiry == allocation


def test_four_week_fixed_grid_allocations_use_200_states() -> None:
    allocations = run_production_gate._EXPIRY_TRIANGULAR_ALLOCATIONS
    uniform = allocations["triangular-four-week-uniform-200"]
    front_loaded = allocations["triangular-four-week-front-loaded-200"]

    assert uniform == (50, 50, 50, 50)
    assert front_loaded == (100, 50, 25, 25)
    assert sum(uniform) == 200
    assert sum(front_loaded) == 200


def test_shared_geometry_uses_the_existing_nms_market_cache_identity() -> None:
    shared = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    cached = run_production_gate._market_cache_geometry(shared)
    assert type(cached) is FlexGeometryConfig
    assert cached.model_business_dtes.tolist() == shared.model_business_dtes.tolist()
    assert cached.nms_range == shared.nms_range


def test_latent_checkpoint_proof_binds_state_and_native_target(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "latent.pt"
    latent_state = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    fingerprints = ["first", "second"]
    native_state = torch.arange(2 * 2_815, dtype=torch.float32).reshape(2, 2_815)
    native_state_sha256 = hashlib.sha256(native_state.numpy().tobytes()).hexdigest()
    torch.save(
        {
            "schema_version": 2,
            "phase": "complete",
            "model_identity": {
                "model_class": "LatentStateSanosModel",
                "config_version": "test-version",
                "model_config": {"state_size": 20},
            },
            "model_state_dict": {
                "daily_state_embedding.weight": latent_state,
            },
            "market_fingerprints": fingerprints,
            "supervision_target_identity": {
                "market_fingerprints": fingerprints,
                "dtype": str(native_state.dtype),
                "shape": list(native_state.shape),
                "raw_state_sha256": native_state_sha256,
            },
            "training_audits": {
                "supervised": {"converged": True},
                "market": {"converged": True},
            },
        },
        checkpoint_path,
    )

    proof = flex_training.latent_checkpoint_proof(checkpoint_path)

    assert proof["schema_version"] == 2
    assert proof["phase"] == "complete"
    assert proof["num_market_days"] == 2
    assert proof["latent_state_shape"] == [2, 20]
    assert proof["latent_state_sha256"] == hashlib.sha256(
        latent_state.numpy().tobytes()
    ).hexdigest()
    assert proof["native_state_shape"] == [2, 2_815]
    assert proof["native_state_sha256"] == native_state_sha256
    assert len(proof["checkpoint_sha256"]) == 64
    assert len(proof["market_fingerprints_sha256"]) == 64
    rendered = "\n".join(
        run_production_gate._fit_section(
            "Compression to state size 20",
            _minimal_native_fit_report() | {"checkpoint_proof": proof},
        )
    )
    assert proof["checkpoint_sha256"] in rendered
    assert proof["latent_state_sha256"] in rendered
    assert "LatentStateSanosModel" in rendered

    payload = torch.load(checkpoint_path, weights_only=False)
    payload["phase"] = "market"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="not complete"):
        flex_training.latent_checkpoint_proof(checkpoint_path)

    payload["phase"] = "complete"
    payload["model_state_dict"]["daily_state_embedding.weight"][0, 0] = float(
        "nan"
    )
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="non-finite tensors"):
        flex_training.latent_checkpoint_proof(checkpoint_path)


def _minimal_native_fit_report() -> dict[str, object]:
    excess = {
        "num_options": 0,
        "num_zero": 0,
        "percent_zero": 0.0,
        "num_positive": 0,
        "positive_mean": 0.0,
        "positive_median": 0.0,
        "surface_mean": 0.0,
        "surface_median": 0.0,
    }
    return {
        "objective": 1.0,
        "statistics": {
            bucket: dict(excess)
            for bucket in ("all", "0DTE", "1DTE", "2DTE")
        },
        "surface_fit": {
            "num_options": 0,
            "half_spread_mae": 0.0,
            "half_spread_median_absolute_error": 0.0,
            "normalized_price_mae": 0.0,
            "normalized_price_median_absolute_error": 0.0,
            "normalized_price_rmse": 0.0,
            "normalized_price_bias": 0.0,
        },
        "history": [],
    }


def test_native_completion_rejects_nonfinite_or_inconsistent_metrics() -> None:
    nonfinite = _minimal_native_fit_report()
    nonfinite["objective"] = float("nan")
    with pytest.raises(ValueError, match="objective must be finite"):
        run_production_gate._validate_native_fit_report(nonfinite)

    inconsistent = _minimal_native_fit_report()
    inconsistent["statistics"]["all"]["num_positive"] = 1
    with pytest.raises(ValueError, match="inconsistent all counts"):
        run_production_gate._validate_native_fit_report(inconsistent)

    inconsistent_percent = _minimal_native_fit_report()
    inconsistent_percent["statistics"]["all"].update(
        num_options=2,
        num_zero=1,
        num_positive=1,
        percent_zero=60.0,
    )
    inconsistent_percent["surface_fit"]["num_options"] = 2
    with pytest.raises(ValueError, match="inconsistent all metrics"):
        run_production_gate._validate_native_fit_report(inconsistent_percent)

    negative_surface = _minimal_native_fit_report()
    negative_surface["surface_fit"]["normalized_price_mae"] = -1.0
    with pytest.raises(ValueError, match="negative surface errors"):
        run_production_gate._validate_native_fit_report(negative_surface)


def test_native_history_audit_supports_sparse_journal_milestones() -> None:
    history = [
        {"epoch": 1.0, "objective": 25.0},
        {"epoch": 25.0, "objective": 4.0},
        {"epoch": 50.0, "objective": 2.0},
    ]
    audit = run_production_gate._native_history_audit(history)
    assert audit["num_epochs"] == 50
    assert audit["num_recorded_epochs"] == 3
    assert audit["first_objective"] == 25.0
    assert audit["last_objective"] == 2.0
    assert audit["best_objective"] == 2.0
    assert audit["best_epoch"] == 50
    assert audit["last_25_epoch_relative_improvement"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="strictly increasing"):
        run_production_gate._native_history_audit(
            [
                {"epoch": 25.0, "objective": 4.0},
                {"epoch": 25.0, "objective": 3.0},
            ]
        )


def test_compression_preserves_only_a_matching_certified_native_payload() -> None:
    certified_native = _minimal_native_fit_report()
    certified_native.update(
        elapsed_seconds=123.0,
        history=[{"epoch": 300.0, "objective": 1.0}],
        target_identity={
            "market_fingerprints_sha256": "certified",
            "nms_range": [-3.0, 2.0],
        },
    )
    repriced_native = dict(certified_native)
    repriced_native.update(
        elapsed_seconds=0.5,
        history=[],
        target_identity={
            "market_fingerprints_sha256": "certified",
            "nms_range": (-3.0, 2.0),
        },
    )

    preserved = run_production_gate._certified_native_payload(
        repriced_native,
        {"native": certified_native},
    )
    assert preserved == certified_native
    assert preserved is not certified_native

    mismatch_values = {
        "statistics": {
            **repriced_native["statistics"],
            "0DTE": {
                **repriced_native["statistics"]["0DTE"],
                "positive_mean": 1.0,
            },
        },
        "surface_fit": {
            **repriced_native["surface_fit"],
            "normalized_price_mae": 1.0,
        },
        "objective": 2.0,
        "target_identity": {"market_fingerprints_sha256": "mismatch"},
    }
    for field, mismatch in mismatch_values.items():
        mismatched = dict(repriced_native)
        mismatched[field] = mismatch
        with pytest.raises(ValueError, match=field):
            run_production_gate._certified_native_payload(
                mismatched,
                {"native": certified_native},
            )


def test_compression_reprices_native_with_certified_batch_size() -> None:
    requested = FlexTrainConfig(
        day_batch_size=128,
        just_load=False,
        warm_start=True,
    )
    effective = run_production_gate._certified_native_reprice_train_config(
        requested,
        completed_report={"train_config": {"day_batch_size": 16}},
    )

    assert requested.day_batch_size == 128
    assert effective.day_batch_size == 16
    assert effective.just_load is True
    assert effective.warm_start is False


def test_compression_uses_exact_completed_native_market_dates() -> None:
    requested_dates = (
        date(2024, 3, 4),
        date(2024, 3, 5),
        date(2024, 3, 6),
    )
    report = {
        "market_dates": ["2024-03-04", "2024-03-06"],
        "num_days": 2,
    }
    assert run_production_gate._completed_native_market_dates(
        report,
        requested_dates=requested_dates,
        expected_num_days=2,
    ) == (requested_dates[0], requested_dates[2])

    report["market_dates"] = ["2024-03-06", "2024-03-04"]
    with pytest.raises(ValueError, match="unique and sorted"):
        run_production_gate._completed_native_market_dates(
            report,
            requested_dates=requested_dates,
            expected_num_days=2,
        )


def test_compression_requires_a_matching_completed_native_report(
    tmp_path: Path,
) -> None:
    geometry = FlexGeometryConfig()
    market_config = flex_market.FlexMarketConfig()
    objective_config = FlexObjectiveConfig()
    dates = (date(2024, 3, 5),)
    label = "native-completion-gate"
    report_path = tmp_path / f"{label}.json"
    checkpoint_path = tmp_path / "native.pt"

    def write_checkpoint(*, phase: str, raw_value: float = 0.0) -> None:
        torch.save(
            {
                "schema_version": 2,
                "model_identity": {"test": True},
                "model_state_dict": {
                    "daily_raw_embedding.weight": torch.full(
                        (1, geometry.native_state_size),
                        raw_value,
                    ),
                },
                "market_fingerprints": ["test-fingerprint"],
                "phase": phase,
            },
            checkpoint_path,
        )

    write_checkpoint(phase="market")
    report = {
        "schema": "flex-batch-descent-capacity-v1",
        "label": label,
        "status": "native_running",
        "ticker": "SPX",
        "requested_dates": [dates[0].isoformat()],
        "market_dates": [dates[0].isoformat()],
        "num_days": 1,
        "geometry": asdict(geometry),
        "market_config": asdict(market_config),
        "objective_config": asdict(objective_config),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    arguments = {
        "ticker": "SPX",
        "dates": dates,
        "geometry": geometry,
        "market_config": market_config,
        "objective_config": objective_config,
        "checkpoint_path": checkpoint_path,
        "output_directory": tmp_path,
        "label": label,
    }
    with pytest.raises(RuntimeError, match="status 'native_complete'"):
        run_production_gate._require_completed_native_report(**arguments)

    report["status"] = "native_complete"
    report["native"] = _minimal_native_fit_report()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="found phase 'market'"):
        run_production_gate._require_completed_native_report(**arguments)

    write_checkpoint(phase="complete")
    loaded = run_production_gate._require_completed_native_report(**arguments)
    assert loaded["native"] == report["native"]
    assert loaded["native_checkpoint_proof"]["schema_version"] == 2
    assert (tmp_path / f"{label}-native-complete.json").exists()
    reloaded = run_production_gate._require_completed_native_report(**arguments)
    assert reloaded["native"] == report["native"]

    with pytest.raises(ValueError, match="ticker"):
        run_production_gate._require_completed_native_report(
            **(arguments | {"ticker": "QQQ"}),
        )

    write_checkpoint(phase="complete", raw_value=1.0)
    with pytest.raises(ValueError, match="current checkpoint"):
        run_production_gate._require_completed_native_report(**arguments)


def test_compression_accepts_completed_schema_one_native_artifact(
    tmp_path: Path,
) -> None:
    """The active long run must hand off without retraining its legacy file."""
    geometry = FlexGeometryConfig()
    market_config = flex_market.FlexMarketConfig()
    objective_config = FlexObjectiveConfig()
    dates = (date(2024, 3, 5),)
    label = "legacy-native-completion-gate"
    checkpoint_path = tmp_path / "native.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": {
                "daily_raw_embedding.weight": torch.zeros(
                    (1, geometry.native_state_size),
                ),
            },
            "market_fingerprints": ["test-fingerprint"],
        },
        checkpoint_path,
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    report = {
        "schema": "flex-batch-descent-capacity-v1",
        "label": label,
        "status": "native_complete",
        "ticker": "SPX",
        "requested_dates": [dates[0].isoformat()],
        "market_dates": [dates[0].isoformat()],
        "num_days": 1,
        "geometry": asdict(geometry),
        "market_config": asdict(market_config),
        "objective_config": asdict(objective_config),
        "native": _minimal_native_fit_report(),
    }
    (tmp_path / f"{label}.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    arguments = {
        "ticker": "SPX",
        "dates": dates,
        "geometry": geometry,
        "market_config": market_config,
        "objective_config": objective_config,
        "checkpoint_path": checkpoint_path,
        "output_directory": tmp_path,
        "label": label,
    }

    completion = run_production_gate.certify_native_completion(
        ticker="SPX",
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        objective_config=objective_config,
        checkpoint_directory=tmp_path,
        output_directory=tmp_path,
        label=label,
    )
    assert completion["status"] == "native_complete"
    completion_path = tmp_path / f"{label}-native-complete.json"
    assert completion == json.loads(completion_path.read_text(encoding="utf-8"))
    review = (tmp_path / f"{label}-native-review.md").read_text(
        encoding="utf-8"
    )
    assert "# FLEX native-state completion review" in review
    assert "Checkpoint schema: `1`" in review
    assert "## Market reconstruction" in review
    assert completion["native_checkpoint_proof"]["schema_version"] == 1
    assert completion["native_checkpoint_proof"]["phase"] is None
    completion_audit = completion["native_completion_audit"]
    assert completion_audit["history"]["num_epochs"] == 0
    assert completion_audit["state"]["all_raw_states_finite"]
    assert completion_audit["state"]["num_market_days"] == 1
    assert completion_audit["state"]["native_state_size"] == 2_815
    assert completion_audit["state"]["coordinates"]["forward_vol"][
        "median"
    ] == pytest.approx(0.505)
    assert completion_audit["state"]["coordinates"]["eta"][
        "fraction_near_lower_bound"
    ] == 0.0
    assert checkpoint_path.read_bytes() == checkpoint_bytes

    reloaded = run_production_gate._require_completed_native_report(**arguments)
    assert reloaded["native_checkpoint_proof"] == completion[
        "native_checkpoint_proof"
    ]

    nonfinite_checkpoint_path = tmp_path / "nonfinite-native.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": {
                "daily_raw_embedding.weight": torch.full(
                    (1, geometry.native_state_size),
                    float("nan"),
                ),
            },
            "market_fingerprints": ["test-fingerprint"],
        },
        nonfinite_checkpoint_path,
    )
    with pytest.raises(ValueError, match="non-finite states"):
        flex_training.native_checkpoint_state_audit(
            nonfinite_checkpoint_path,
            geometry=geometry,
        )


def test_native_runner_cannot_promote_schema_one_or_erase_a_certificate(
    tmp_path: Path,
) -> None:
    geometry = FlexGeometryConfig()
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    checkpoint_path = checkpoint_directory / "native.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": {
                "daily_raw_embedding.weight": torch.zeros(
                    (1, geometry.native_state_size),
                ),
            },
            "market_fingerprints": ["partial"],
        },
        checkpoint_path,
    )
    output_directory = tmp_path / "reports"
    label = "unsafe-legacy-load"
    arguments = {
        "ticker": "SPX",
        "dates": (date(2024, 3, 5),),
        "geometry": geometry,
        "market_config": flex_market.FlexMarketConfig(),
        "objective_config": FlexObjectiveConfig(),
        "train_config": FlexTrainConfig(just_load=True),
        "supervised_train_config": FlexTrainConfig(),
        "workers": 1,
        "output_directory": output_directory,
        "checkpoint_directory": checkpoint_directory,
        "label": label,
    }
    with pytest.raises(RuntimeError, match="schema-1 checkpoint is complete"):
        run_production_gate.run_production_gates(**arguments)
    assert not (output_directory / f"{label}.json").exists()

    output_directory.mkdir()
    certificate_path = output_directory / f"{label}-native-complete.json"
    certificate_path.write_text("preserve me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already certified"):
        run_production_gate.run_production_gates(**arguments)
    assert certificate_path.read_text(encoding="utf-8") == "preserve me"
    assert not (output_directory / f"{label}.json").exists()


def _synthetic_market_and_surface() -> tuple[
    FlexGeometryConfig,
    FlexMarketDay,
    np.ndarray,
]:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    forward_vols = np.full(2, 0.20)
    total_vols = total_vols_from_forward_vols_numpy(
        forward_vols,
        geometry=geometry,
    )
    surface = decoder(
        dlvsqs=torch.full((2, 5), 0.20**2, dtype=env.fdtype),
        forward_vols=torch.as_tensor(forward_vols, dtype=env.fdtype),
        eta=torch.tensor(0.50, dtype=env.fdtype),
    )
    expiries = np.array(
        [
            0.0,
            0.5 * geometry.model_expiries_y[0],
            geometry.model_expiries_y[0],
            geometry.model_expiries_y[1],
        ]
    )
    strikes = np.tile(np.array([0.9, 1.0, 1.1]), len(expiries))
    quote_expiries = np.repeat(expiries, 3)
    call_prices = decoder.price_calls(
        surface,
        expiries_y=torch.as_tensor(quote_expiries, dtype=env.fdtype),
        strikes=torch.as_tensor(strikes, dtype=env.fdtype),
    ).detach().numpy()
    is_call = np.arange(len(strikes)) % 2 == 0
    option_prices = np.where(is_call, call_prices, call_prices - 1.0 + strikes)
    quote_count = len(strikes)
    day = FlexMarketDay(
        ticker="TEST",
        trading_date=date(2025, 1, 2),
        expiries_y=expiries,
        expiry_dates=tuple(str(index) for index in range(len(expiries))),
        expiry_ids=np.arange(len(expiries)),
        expiry_is_am=np.zeros(len(expiries), dtype=bool),
        expiry_index=np.repeat(np.arange(len(expiries)), 3),
        strikes=strikes,
        is_call=is_call,
        bids=np.maximum(option_prices - 1.0e-4, 0.0),
        asks=option_prices + 1.0e-4,
        fit_prices=option_prices,
        log_volumes=np.zeros(quote_count),
        log_open_interests=np.zeros(quote_count),
        quote_ids=tuple(str(index) for index in range(quote_count)),
        atm_annual_vols=np.full(len(expiries), 0.20),
        model_atm_total_vols=total_vols,
        strike_lower_bounds=np.full(len(expiries), 0.7),
        strike_upper_bounds=np.full(len(expiries), 1.3),
        market_uid="synthetic",
        audit=MarketEligibilityAudit(
            quote_count,
            quote_count,
            quote_count,
            len(expiries),
            len(expiries),
        ),
    )
    return geometry, day, option_prices


def test_density_slack_lp_reprices_a_synthetic_sanos_surface() -> None:
    geometry, day, option_prices = _synthetic_market_and_surface()
    result = fit_flex_density_lp(
        day,
        geometry=geometry,
        config=FlexLpConfig(timeout_seconds=30.0),
    )

    assert result.status == "optimal"
    assert result.statistics.all.percent_zero == 100.0
    np.testing.assert_allclose(result.fits, option_prices, atol=1.0e-4)
    np.testing.assert_allclose(result.densities.sum(axis=1), 1.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.sum(result.densities * result.support_strikes, axis=1),
        1.0,
        atol=1.0e-7,
    )
    assert np.all(np.isfinite(result.dlvsqs))
    assert result.dlv_behavior.num_supported_nodes > 0
    torch_reprice = reprice_dlvsqs_with_torch(
        day,
        result.dlvsqs,
        geometry=geometry,
        config=FlexLpConfig(timeout_seconds=30.0),
    )
    np.testing.assert_allclose(torch_reprice, result.fits, atol=2.0e-7)


def test_lp_fixed_backbone_annual_vol_uses_calendar_total_vols() -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    config = FlexLpConfig(
        backbone_mu=None,
        fixed_backbone_annual_vol=0.05,
    )

    np.testing.assert_allclose(
        config.backbone_total_vols(day, geometry=geometry),
        0.05 * np.sqrt(geometry.model_expiries_y),
    )


def test_lp_allows_zero_fixed_backbone_smoothing() -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    config = FlexLpConfig(
        backbone_mu=None,
        fixed_backbone_annual_vol=0.0,
    )

    np.testing.assert_array_equal(
        config.backbone_total_vols(day, geometry=geometry),
        np.zeros(geometry.num_model_expiries),
    )


def test_lp_book_mid_target_ignores_smooth_reference_prices() -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    shifted_day = replace(
        day,
        fit_prices=np.minimum(day.asks, day.fit_prices + 5.0e-5),
    )
    config = FlexLpConfig(
        central_fit_target="book_mid",
        timeout_seconds=30.0,
    )

    base = fit_flex_density_lp(day, geometry=geometry, config=config)
    shifted = fit_flex_density_lp(
        shifted_day,
        geometry=geometry,
        config=config,
    )

    np.testing.assert_allclose(base.fits, shifted.fits, atol=1.0e-10)


def test_direct_highs_lp_matches_cvxpy_lp() -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    common = dict(
        backbone_mu=None,
        fixed_backbone_annual_vol=0.05,
        central_fit_target="book_mid",
        enforce_dlv_bounds=False,
        timeout_seconds=30.0,
    )
    reference = fit_flex_density_lp(
        day,
        geometry=geometry,
        config=FlexLpConfig(**common, solver_algorithm="simplex"),
    )
    direct = fit_flex_density_lp(
        day,
        geometry=geometry,
        config=FlexLpConfig(**common, solver_algorithm="direct-simplex"),
        initial_densities=reference.densities,
    )

    assert direct.status == "optimal"
    assert direct.objective == pytest.approx(reference.objective, abs=1.0e-8)
    np.testing.assert_allclose(direct.fits, reference.fits, atol=2.0e-7)
    np.testing.assert_allclose(direct.densities.sum(axis=1), 1.0, atol=1.0e-8)
    np.testing.assert_allclose(
        np.sum(direct.densities * direct.support_strikes, axis=1),
        1.0,
        atol=1.0e-8,
    )


def test_shared_strikes_report_renders_a_partial_geometry_stage() -> None:
    report = {
        "status": "direct_running",
        "results": {
            "direct": {
                "nms_strikes": {
                    "aggregate": {
                        "statistics": {
                            "all": {
                                "num_options": 1,
                                "percent_zero": 100.0,
                                "positive_mean": 0.0,
                                "positive_median": 0.0,
                            }
                        },
                        "surface_fit": {"half_spread_mae": 0.0},
                    },
                    "dlv_behavior": {
                        "supported_percent": 100.0,
                        "minimum_bound_percent": 0.0,
                        "maximum_bound_percent": 0.0,
                        "spatial_log_jump_median": 0.0,
                        "spatial_log_jump_p90": 0.0,
                        "expiry_log_change_median": 0.0,
                        "expiry_log_change_p90": 0.0,
                    },
                }
            }
        },
    }
    markdown = run_shared_strikes._markdown(report)
    assert "| nms strikes |" in markdown
    assert "| shared strikes |" not in markdown


def test_torch_training_prices_shared_call_put_locations_once() -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    quote_count = day.num_quotes
    strikes = np.concatenate((day.strikes, day.strikes))
    call_prices = day.call_equivalent_fit_prices
    fit_prices = np.concatenate(
        (call_prices, call_prices - 1.0 + day.strikes)
    )
    paired_day = replace(
        day,
        expiry_index=np.concatenate((day.expiry_index, day.expiry_index)),
        strikes=strikes,
        is_call=np.concatenate(
            (
                np.ones(quote_count, dtype=bool),
                np.zeros(quote_count, dtype=bool),
            )
        ),
        bids=fit_prices + 2.0e-4,
        asks=fit_prices + 4.0e-4,
        fit_prices=fit_prices,
        log_volumes=np.zeros(2 * quote_count),
        log_open_interests=np.zeros(2 * quote_count),
        quote_ids=tuple(f"paired-{index}" for index in range(2 * quote_count)),
        market_uid="synthetic-paired-books",
        audit=MarketEligibilityAudit(
            2 * quote_count,
            2 * quote_count,
            2 * quote_count,
            len(day.expiries_y),
            len(day.expiries_y),
        ),
    )
    short_indices = np.arange(5)
    short_fit_prices = day.fit_prices[short_indices]
    short_day = replace(
        day,
        expiry_index=day.expiry_index[short_indices],
        strikes=day.strikes[short_indices],
        is_call=day.is_call[short_indices],
        bids=short_fit_prices + 2.0e-4,
        asks=short_fit_prices + 4.0e-4,
        fit_prices=short_fit_prices,
        log_volumes=day.log_volumes[short_indices],
        log_open_interests=day.log_open_interests[short_indices],
        quote_ids=tuple(day.quote_ids[index] for index in short_indices),
        market_uid="synthetic-short-books",
        audit=MarketEligibilityAudit(
            len(short_indices),
            len(short_indices),
            len(short_indices),
            len(day.expiries_y),
            2,
        ),
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    torch_day = flex_training.materialize_day(paired_day, env=env)
    short_torch_day = flex_training.materialize_day(short_day, env=env)
    assert len(torch_day.price_strikes) == quote_count
    assert len(short_torch_day.price_strikes) == len(short_indices)
    torch.testing.assert_close(
        torch_day.quote_price_index[:quote_count],
        torch_day.quote_price_index[quote_count:],
        rtol=0.0,
        atol=0.0,
    )
    batch = flex_training.materialize_batch(
        (torch_day, short_torch_day),
        np.array([0, 1]),
        env=env,
    )
    model = NativeStateSanosModel(
        env,
        num_days=2,
        config=NativeStateConfig(
            label="paired-location-test",
            geometry=geometry,
        ),
    )
    with torch.no_grad():
        model.daily_raw_states.copy_(
            raw_from_native_parameters(
                NativeParameters(
                    dlvsqs=torch.full(
                        (2, geometry.num_model_expiries, geometry.num_inner_strikes),
                        0.20**2,
                        dtype=env.fdtype,
                    ),
                    forward_vols=torch.full(
                        (2, geometry.num_model_expiries),
                        0.20,
                        dtype=env.fdtype,
                    ),
                    eta=torch.full((2,), 0.50, dtype=env.fdtype),
                ),
                geometry=geometry,
            )
    )

    deduplicated = flex_training._model_prices(model, batch)
    deduplicated_objective = flex_training.quote_objective(
        deduplicated,
        batch,
        objective_config=FlexObjectiveConfig(),
    )
    deduplicated_gradient = torch.autograd.grad(
        deduplicated_objective,
        model.daily_raw_states,
    )[0].to_dense()
    max_quotes = batch.strikes.shape[1]
    direct_expiries = torch.stack(
        [
            torch.nn.functional.pad(
                value.expiries_y,
                (0, max_quotes - len(value.expiries_y)),
                value=0.0,
            )
            for value in (torch_day, short_torch_day)
        ]
    )
    direct = model(
        day_indices=batch.day_indices,
        expiries_y=direct_expiries,
        strikes=batch.strikes,
        is_call=batch.is_call,
    )[0]
    direct_objective = flex_training.quote_objective(
        direct,
        batch,
        objective_config=FlexObjectiveConfig(),
    )
    direct_gradient = torch.autograd.grad(
        direct_objective,
        model.daily_raw_states,
    )[0].to_dense()

    torch.testing.assert_close(
        deduplicated[batch.mask],
        direct[batch.mask],
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        deduplicated_objective,
        direct_objective,
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        deduplicated_gradient,
        direct_gradient,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_sanos_solver_compile_can_be_restored_to_eager(monkeypatch) -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    decoder = FlexSanosDecoder(
        BaseEnvironment(precision=64, device="cpu", debug=True),
        geometry=geometry,
    )
    eager_function = decoder._solve_sanos_step
    compiled = object()
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: compiled)

    decoder.enable_solver_compile()
    assert decoder._sanos_step is compiled
    decoder.disable_solver_compile()
    assert decoder._sanos_step is eager_function


def test_native_state_is_direct_and_sigmoid_bounded() -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = NativeStateSanosModel(
        env,
        num_days=3,
        config=NativeStateConfig(label="unit-native", geometry=geometry),
    )
    assert model.daily_raw_states.shape == (3, geometry.native_state_size)
    assert not hasattr(model, "raw_decoder")
    parameters = bound_native_raw(model.daily_raw_states, geometry=geometry)
    assert torch.all(parameters.dlvsqs > geometry.min_dlvsq)
    assert torch.all(parameters.dlvsqs < geometry.max_dlvsq)
    assert torch.all(parameters.forward_vols > geometry.min_forward_vol)
    assert torch.all(parameters.forward_vols < geometry.max_forward_vol)
    assert torch.all(parameters.eta > geometry.min_eta)
    assert torch.all(parameters.eta < geometry.max_eta)

    recovered = raw_from_native_parameters(parameters, geometry=geometry)
    torch.testing.assert_close(recovered, model.daily_raw_states)


def test_unbounded_log_dlvsq_hits_price_cap_without_sigmoid_saturation() -> None:
    geometry = CappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=2,
    )
    native = NativeParameters(
        dlvsqs=torch.full((1, 2, 5), geometry.max_dlvsq, dtype=torch.float64),
        forward_vols=torch.full((1, 2), 0.20, dtype=torch.float64),
        eta=torch.full((1, 2), 0.30, dtype=torch.float64),
    )
    raw_at_cap = raw_from_native_parameters(native, geometry=geometry)
    dlv_raw_at_cap = raw_at_cap.reshape(1, 2, 7)[..., :5]
    torch.testing.assert_close(
        dlv_raw_at_cap,
        torch.full_like(dlv_raw_at_cap, geometry.maximum_log_dlvsq),
    )
    assert torch.all(torch.isfinite(raw_at_cap))
    torch.testing.assert_close(
        bound_native_raw(raw_at_cap, geometry=geometry).dlvsqs,
        native.dlvsqs,
    )
    expected_barrier_at_cap = (
        geometry.dlv_cap_barrier_log_width
        * torch.nn.functional.softplus(torch.zeros((), dtype=torch.float64))
        / geometry.dlv_cap_barrier_log_std
    )
    torch.testing.assert_close(
        dlvsq_cap_penalty(raw_at_cap, geometry=geometry),
        torch.full(
            (1,),
            0.5
            * geometry.num_model_expiries
            * geometry.num_inner_strikes
            * expected_barrier_at_cap.square(),
            dtype=torch.float64,
        ),
    )

    raw_above_cap = raw_at_cap.detach().clone()
    raw_above_cap.reshape(1, 2, 7)[..., :5].add_(0.25)
    raw_above_cap.requires_grad_(True)
    priced = bound_native_raw(raw_above_cap, geometry=geometry).dlvsqs
    torch.testing.assert_close(priced, native.dlvsqs)

    price_gradient = torch.autograd.grad(
        priced.sum(),
        raw_above_cap,
        retain_graph=True,
    )[0].reshape(1, 2, 7)[..., :5]
    torch.testing.assert_close(price_gradient, torch.zeros_like(price_gradient))

    penalty = dlvsq_cap_penalty(raw_above_cap, geometry=geometry).mean()
    penalty_gradient = torch.autograd.grad(penalty, raw_above_cap)[0]
    penalty_dlv_gradient = penalty_gradient.reshape(1, 2, 7)[..., :5]
    assert torch.all(penalty_dlv_gradient > 0.0)
    assert torch.all(torch.isfinite(penalty_dlv_gradient))


def test_latent_decoders_keep_the_shared_network_architecture_controlled() -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    network = FeedForwardConfig(activation="SiLU", num_depth=1, num_width=10)
    models = [
        LatentStateSanosModel(
            env,
            num_days=2,
            config=LatentDecoderConfig(
                label=f"latent-{state_size}",
                state_size=state_size,
                decoder_input_size=5,
                geometry=geometry,
                decoder_network=network,
            ),
        )
        for state_size in (2, 3, 5)
    ]
    decoder_counts = {
        sum(parameter.numel() for parameter in model.raw_decoder.parameters())
        for model in models
    }
    assert len(decoder_counts) == 1
    assert [model.daily_states.shape for model in models] == [(2, 2), (2, 3), (2, 5)]
    assert models[0].config.version.unique_id48


def test_affine_pca_decoder_is_the_optimal_rank_reconstruction() -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = LatentStateSanosModel(
        env,
        num_days=5,
        config=PcaDecoderConfig(
            label="unit-pca-decoder",
            state_size=2,
            geometry=geometry,
        ),
    )
    generator = torch.Generator().manual_seed(1234)
    target = torch.randn(
        (5, geometry.native_state_size),
        dtype=env.fdtype,
        generator=generator,
    )
    diagnostics = initialize_pca_decoder_(model, target)
    reconstructed = model.raw_states_for_days(torch.arange(5))

    centered = target - torch.mean(target, dim=0)
    singular_values = torch.linalg.svdvals(centered)
    expected_mse = torch.sum(singular_values[2:].square()) / target.numel()
    torch.testing.assert_close(
        torch.mean((reconstructed - target).square()),
        expected_mse,
    )
    torch.testing.assert_close(
        model.raw_decoder.I0.bias,
        torch.mean(target, dim=0),
    )
    assert diagnostics["mse"] == pytest.approx(float(expected_mse))
    assert diagnostics["pca_rank"] == 2.0
    reconstructed.sum().backward()
    assert torch.all(torch.isfinite(model.raw_decoder.I0.weight.grad))


def test_full_capacity_decoder_control_has_no_hidden_dimensional_bottleneck() -> None:
    geometry = FlexGeometryConfig()
    configs = [
        run_production_gate._latent_decoder_config(
            label=f"latent-{state_size}",
            state_size=state_size,
            geometry=geometry,
        )
        for state_size in (20, 100, 200, 1_000, geometry.native_state_size)
    ]
    assert [config.decoder_input_size for config in configs] == [
        1_000,
        1_000,
        1_000,
        1_000,
        2_815,
    ]
    assert [config.decoder_network.num_width for config in configs] == [
        1_000,
        1_000,
        1_000,
        1_000,
        2_815,
    ]
    for state_size, config in zip(
        (20, 100, 200, 1_000), configs[:4], strict=True
    ):
        assert config == LatentDecoderConfig(
            label=f"latent-{state_size}",
            state_size=state_size,
            geometry=geometry,
        )

    pca_config = run_production_gate._pca_decoder_config(
        label="pca-100",
        state_size=100,
        geometry=geometry,
    )
    assert isinstance(pca_config, PcaDecoderConfig)
    assert pca_config.decoder_input_size == 100
    assert pca_config.decoder_network.num_depth == 0


def test_batch_descent_and_compression_do_not_require_an_lp(
    tmp_path: Path,
) -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    native_config = NativeStateConfig(
        label="batch-descent-native",
        geometry=geometry,
    )
    initial_raw = market_native_raw_initial_states(
        (day,),
        env=env,
        config=native_config,
    )
    assert initial_raw.shape == (1, geometry.native_state_size)
    assert torch.all(torch.isfinite(initial_raw))

    train_config = FlexTrainConfig(
        num_epochs=2,
        day_batch_size=1,
        learning_rate=1.0e-3,
        checkpoint_every=1,
        report_every=1,
        convergence_window=1,
        convergence_relative_tolerance=0.999999,
        precision=64,
        device="cpu",
    )
    native_fit = fit_native_states(
        (day,),
        config=native_config,
        train_config=train_config,
        checkpoint_path=tmp_path / "native.pt",
    )
    assert np.isfinite(native_fit.evaluation.objective)
    assert len(native_fit.history) == 2

    latent_fit = fit_latent_states(
        (day,),
        native_fit,
        config=LatentDecoderConfig(
            label="batch-descent-latent",
            state_size=2,
            decoder_input_size=5,
            geometry=geometry,
            decoder_network=FeedForwardConfig(
                activation="SiLU",
                num_depth=1,
                num_width=10,
            ),
        ),
        supervised_train_config=train_config,
        market_train_config=train_config,
        checkpoint_path=tmp_path / "latent.pt",
    )
    assert np.isfinite(latent_fit.evaluation.objective)
    assert len(latent_fit.supervised_history) == 2
    assert len(latent_fit.market_history) == 2

    repeated_latent_fit = fit_latent_states(
        (day,),
        native_fit,
        config=latent_fit.model.config,
        supervised_train_config=train_config,
        market_train_config=train_config,
    )
    for name, value in latent_fit.model.state_dict().items():
        torch.testing.assert_close(
            repeated_latent_fit.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    pca_checkpoint_path = tmp_path / "pca.pt"
    pca_fit = fit_latent_states(
        (day,),
        native_fit,
        config=PcaDecoderConfig(
            label="batch-descent-pca",
            state_size=2,
            geometry=geometry,
        ),
        supervised_train_config=train_config,
        market_train_config=train_config,
        checkpoint_path=pca_checkpoint_path,
    )
    assert np.isfinite(pca_fit.evaluation.objective)
    assert pca_fit.supervised_history == (
        {
            "epoch": 0.0,
            "mse": pytest.approx(0.0, abs=1.0e-20),
            "explained_variance_fraction": 1.0,
            "pca_rank": 1.0,
        },
    )
    assert len(pca_fit.market_history) == 2
    pca_payload = torch.load(
        pca_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert pca_payload["phase"] == "complete"
    assert pca_payload["model_identity"]["model_config"]["group"] == (
        "flex/pca-decoder"
    )

    loaded_pca_fit = fit_latent_states(
        (day,),
        native_fit,
        config=pca_fit.model.config,
        supervised_train_config=train_config,
        market_train_config=replace(train_config, just_load=True),
        checkpoint_path=pca_checkpoint_path,
    )
    assert pca_fit.initial_evaluation is not None
    assert loaded_pca_fit.initial_evaluation is not None
    assert loaded_pca_fit.initial_evaluation.objective == pytest.approx(
        pca_fit.initial_evaluation.objective,
    )
    for name, value in pca_fit.model.state_dict().items():
        torch.testing.assert_close(
            loaded_pca_fit.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_native_checkpoint_resumes_the_exact_training_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    config = NativeStateConfig(
        label="resumable-native",
        geometry=geometry,
    )
    train_config = FlexTrainConfig(
        num_epochs=2,
        day_batch_size=1,
        learning_rate=1.0e-3,
        checkpoint_every=1,
        report_every=1,
        convergence_window=1,
        convergence_relative_tolerance=0.999999,
        precision=64,
        device="cpu",
    )
    checkpoint_path = tmp_path / "resumable-native.pt"
    uninterrupted = fit_native_states(
        (day,),
        config=config,
        train_config=train_config,
    )
    original_save = flex_training.save_model_checkpoint
    interrupted = False

    class _ExpectedInterruption(Exception):
        pass

    def save_then_interrupt(*args, **kwargs) -> None:
        nonlocal interrupted
        original_save(*args, **kwargs)
        if kwargs["phase"] == "market" and not interrupted:
            interrupted = True
            raise _ExpectedInterruption

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        save_then_interrupt,
    )
    with pytest.raises(_ExpectedInterruption):
        fit_native_states(
            (day,),
            config=config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema_version"] == 2
    assert payload["phase"] == "market"
    assert payload["training_state"]["completed_epochs"] == 1
    assert payload["training_state"]["resume_signature"][
        "training_algorithm_version"
    ] == 9
    assert "day_batch_size" not in payload["training_state"]["resume_signature"]
    assert not payload["training_state"]["resume_signature"][
        "compile_sanos_solver"
    ]
    # Legacy schema-2 checkpoints included this operational field. It must not
    # block restoration of optimizer, scheduler, history, and RNG state.
    payload["training_state"]["resume_signature"]["day_batch_size"] = 1
    torch.save(payload, checkpoint_path)
    assert len(payload["training_state"]["optimizer_state_dicts"]) == 1
    assert len(payload["training_state"]["scheduler_state_dicts"]) == 1
    assert set(payload["training_state"]["pre_training_score"]) == {
        "objective",
        "percent_zero",
    }
    assert "selection_baseline_state_dict" not in payload["training_state"]
    with pytest.raises(ValueError, match="checkpoint is not complete"):
        fit_native_states(
            (day,),
            config=config,
            train_config=replace(train_config, just_load=True),
            checkpoint_path=checkpoint_path,
        )
    with pytest.raises(ValueError, match="resume signature mismatch"):
        fit_native_states(
            (day,),
            config=config,
            train_config=replace(train_config, learning_rate=2.0e-3),
            checkpoint_path=checkpoint_path,
        )
    with pytest.raises(ValueError, match="resume signature mismatch"):
        fit_native_states(
            (day,),
            config=config,
            objective_config=FlexObjectiveConfig(fit_weight=0.10),
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
    checkpoint_bytes = checkpoint_path.read_bytes()
    payload["training_state"]["resume_signature"][
        "compile_sanos_solver"
    ] = True
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="resume signature mismatch"):
        fit_native_states(
            (day,),
            config=config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
    payload["training_state"]["resume_signature"][
        "compile_sanos_solver"
    ] = False
    payload["training_state"]["resume_signature"][
        "training_algorithm_version"
    ] = 1
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="resume signature mismatch"):
        fit_native_states(
            (day,),
            config=config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
    checkpoint_path.write_bytes(checkpoint_bytes)

    invalid_score_path = tmp_path / "invalid-native-pre-training-score.pt"
    invalid_score_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    invalid_score_payload["training_state"]["pre_training_score"][
        "objective"
    ] = -1.0
    torch.save(invalid_score_payload, invalid_score_path)
    with pytest.raises(ValueError, match="invalid pre-training score"):
        fit_native_states(
            (day,),
            config=config,
            train_config=train_config,
            checkpoint_path=invalid_score_path,
        )

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        original_save,
    )
    resumed = fit_native_states(
        (day,),
        config=config,
        train_config=replace(train_config, day_batch_size=2),
        checkpoint_path=checkpoint_path,
    )
    assert [item["epoch"] for item in resumed.history] == [1.0, 2.0]
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    completed_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert completed_payload["phase"] == "complete"
    assert completed_payload["training_state"] is None
    assert len(completed_payload["market_history"]) == 2
    completed_bytes = checkpoint_path.read_bytes()
    loaded = fit_native_states(
        (day,),
        config=config,
        train_config=replace(train_config, just_load=True),
        checkpoint_path=checkpoint_path,
    )
    assert len(loaded.history) == 2
    assert checkpoint_path.read_bytes() == completed_bytes
    with pytest.raises(ValueError, match="implicit retraining"):
        fit_native_states(
            (day,),
            config=config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )

    with pytest.raises(ValueError, match="model identity mismatch"):
        fit_native_states(
            (day,),
            config=NativeStateConfig(
                label="different-native-identity",
                geometry=geometry,
            ),
            train_config=replace(train_config, just_load=True),
            checkpoint_path=checkpoint_path,
        )


def test_schema_one_native_checkpoint_load_is_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    config = NativeStateConfig(
        label="legacy-native-upgrade",
        geometry=geometry,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    model = NativeStateSanosModel(env, num_days=1, config=config)
    checkpoint_path = tmp_path / "legacy-native.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "market_fingerprints": [day.training_fingerprint],
        },
        checkpoint_path,
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    evaluation_calls = 0
    original_evaluate = flex_training.evaluate_model

    def count_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(flex_training, "evaluate_model", count_evaluation)
    result = fit_native_states(
        (day,),
        config=config,
        train_config=FlexTrainConfig(
            num_epochs=1,
            day_batch_size=1,
            precision=64,
            device="cpu",
            just_load=True,
        ),
        checkpoint_path=checkpoint_path,
    )
    assert result.history == ()
    assert evaluation_calls == 1
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert loaded["schema_version"] == 1
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_latent_checkpoint_resumes_supervised_then_market_phases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    geometry, day, _ = _synthetic_market_and_surface()
    train_config = FlexTrainConfig(
        num_epochs=2,
        day_batch_size=1,
        learning_rate=1.0e-3,
        checkpoint_every=1,
        report_every=1,
        convergence_window=1,
        convergence_relative_tolerance=0.999999,
        precision=64,
        device="cpu",
    )
    native_fit = fit_native_states(
        (day,),
        config=NativeStateConfig(
            label="latent-resume-native-target",
            geometry=geometry,
        ),
        train_config=train_config,
    )
    native_target = native_fit.model.daily_raw_states.detach().clone()
    latent_config = LatentDecoderConfig(
        label="resumable-latent",
        state_size=2,
        decoder_input_size=5,
        geometry=geometry,
        decoder_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=1,
            num_width=10,
        ),
    )
    checkpoint_path = tmp_path / "resumable-latent.pt"
    uninterrupted = fit_latent_states(
        (day,),
        native_fit,
        config=latent_config,
        supervised_train_config=train_config,
        market_train_config=train_config,
    )
    original_save = flex_training.save_model_checkpoint
    interrupted = False

    class _ExpectedInterruption(Exception):
        pass

    def save_then_interrupt(*args, **kwargs) -> None:
        nonlocal interrupted
        original_save(*args, **kwargs)
        if kwargs["phase"] == "supervised" and not interrupted:
            interrupted = True
            raise _ExpectedInterruption

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        save_then_interrupt,
    )
    with pytest.raises(_ExpectedInterruption):
        fit_latent_states(
            (day,),
            native_fit,
            config=latent_config,
            supervised_train_config=train_config,
            market_train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert payload["phase"] == "supervised"
    assert payload["training_state"]["completed_epochs"] == 1

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        original_save,
    )
    resumed = fit_latent_states(
        (day,),
        native_fit,
        config=latent_config,
        supervised_train_config=train_config,
        market_train_config=train_config,
        checkpoint_path=checkpoint_path,
    )
    assert [item["epoch"] for item in resumed.supervised_history] == [1.0, 2.0]
    assert [item["epoch"] for item in resumed.market_history] == [1.0, 2.0]
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    completed_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert completed_payload["phase"] == "complete"
    assert len(completed_payload["supervised_history"]) == 2
    torch.testing.assert_close(
        native_fit.model.daily_raw_states,
        native_target,
    )

    market_checkpoint_path = tmp_path / "resumable-latent-market.pt"
    market_interrupted = False

    def save_then_interrupt_market(*args, **kwargs) -> None:
        nonlocal market_interrupted
        original_save(*args, **kwargs)
        if kwargs["phase"] == "market" and not market_interrupted:
            market_interrupted = True
            raise _ExpectedInterruption

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        save_then_interrupt_market,
    )
    with pytest.raises(_ExpectedInterruption):
        fit_latent_states(
            (day,),
            native_fit,
            config=latent_config,
            supervised_train_config=train_config,
            market_train_config=train_config,
            checkpoint_path=market_checkpoint_path,
        )
    market_payload = torch.load(
        market_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert market_payload["phase"] == "market"
    assert market_payload["training_state"]["completed_epochs"] == 1
    assert len(market_payload["training_state"]["optimizer_state_dicts"]) == 2
    assert len(market_payload["training_state"]["scheduler_state_dicts"]) == 2
    assert set(market_payload["training_state"]["pre_training_score"]) == {
        "objective",
        "percent_zero",
    }
    assert "selection_baseline_state_dict" not in market_payload["training_state"]

    monkeypatch.setattr(
        flex_training,
        "save_model_checkpoint",
        original_save,
    )
    market_resumed = fit_latent_states(
        (day,),
        native_fit,
        config=latent_config,
        supervised_train_config=train_config,
        market_train_config=train_config,
        checkpoint_path=market_checkpoint_path,
    )
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(
            market_resumed.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    with pytest.raises(ValueError, match="native fit provenance"):
        fit_latent_states(
            (replace(day, market_uid="different-market"),),
            native_fit,
            config=latent_config,
            supervised_train_config=train_config,
            market_train_config=train_config,
        )
    float32_train_config = replace(train_config, precision=32)
    with pytest.raises(ValueError, match="dtype does not match"):
        fit_latent_states(
            (day,),
            native_fit,
            config=latent_config,
            supervised_train_config=float32_train_config,
            market_train_config=float32_train_config,
        )

    evaluation_calls = 0
    original_evaluate = flex_training.evaluate_model

    def count_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(flex_training, "evaluate_model", count_evaluation)
    loaded_latent = fit_latent_states(
        (day,),
        native_fit,
        config=latent_config,
        supervised_train_config=train_config,
        market_train_config=replace(train_config, just_load=True),
        checkpoint_path=checkpoint_path,
    )
    assert evaluation_calls == 1
    assert len(loaded_latent.supervised_history) == 2
    assert len(loaded_latent.market_history) == 2
    monkeypatch.setattr(flex_training, "evaluate_model", original_evaluate)

    with torch.no_grad():
        native_fit.model.daily_raw_states[0, 0].add_(1.0e-4)
    with pytest.raises(ValueError, match="native supervision target mismatch"):
        fit_latent_states(
            (day,),
            native_fit,
            config=latent_config,
            supervised_train_config=train_config,
            market_train_config=replace(train_config, just_load=True),
            checkpoint_path=checkpoint_path,
        )


def test_native_parameter_dataclass_accepts_batched_geometry() -> None:
    geometry = FlexGeometryConfig(num_inner_strikes=5, num_model_expiries=2)
    parameters = NativeParameters(
        dlvsqs=torch.full((4, 2, 5), 0.04),
        forward_vols=torch.full((4, 2), 0.2),
        eta=torch.full((4,), 0.3),
    )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    assert raw.shape == (4, geometry.native_state_size)
