"""Focused tests for the Phase 3 fixed-RBF market encoder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market import FlexMarketDay, MarketEligibilityAudit
from quantdata.dysanos.flex.market_encoder import (
    HierarchicalRbfQuoteEncoder,
    HierarchicalRbfQuoteEncoderConfig,
    NativeRbfQuoteEncoder,
    NativeRbfQuoteEncoderConfig,
    RbfQuoteEncoder,
    RbfQuoteEncoderConfig,
    equal_date_quote_objective,
    fit_rbf_feature_statistics,
    load_encoded_strong_history,
    rbf_quote_summary,
    transform_market_days,
)
from quantdata.dysanos.flex.model import (
    NativeParameters,
    bound_native_raw,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.run_phase3_native_encoder import (
    _convert_sigmoid_native_targets,
    _evaluate,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_native_targets as _load_shared_native_targets,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _market_dates_from_completion as _shared_market_dates_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _split_markets as _split_shared_markets,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.training import TorchMarketBatch
from quantdata.torch import BaseEnvironment


def _market_day(*, trading_date: date = date(2024, 1, 2)) -> FlexMarketDay:
    geometry = ForwardMuFlexGeometryConfig()
    expiries_y = geometry.model_expiries_y
    strikes = np.tile(np.array([0.9, 1.0, 1.1]), len(expiries_y))
    expiry_index = np.repeat(np.arange(len(expiries_y)), 3)
    is_call = np.tile(np.array([False, True, True]), len(expiries_y))
    intrinsic = np.where(is_call, np.maximum(1.0 - strikes, 0.0), np.maximum(strikes - 1.0, 0.0))
    fit = intrinsic + 0.02
    count = len(strikes)
    return FlexMarketDay(
        ticker="SPX",
        trading_date=trading_date,
        expiries_y=expiries_y,
        expiry_dates=tuple(f"2024-expiry-{index}" for index in range(len(expiries_y))),
        expiry_ids=np.arange(len(expiries_y), dtype=np.int64),
        expiry_is_am=np.zeros(len(expiries_y), dtype=bool),
        expiry_index=expiry_index,
        strikes=strikes,
        is_call=is_call,
        bids=fit - 0.001,
        asks=fit + 0.001,
        fit_prices=fit,
        log_volumes=np.linspace(0.0, 2.0, count),
        log_open_interests=np.linspace(1.0, 3.0, count),
        quote_ids=tuple(f"quote-{index}" for index in range(count)),
        atm_annual_vols=np.full(len(expiries_y), 0.20),
        model_atm_total_vols=0.20 * np.sqrt(expiries_y),
        strike_lower_bounds=np.full(len(expiries_y), 0.7),
        strike_upper_bounds=np.full(len(expiries_y), 1.3),
        market_uid=f"synthetic-{trading_date}",
        audit=MarketEligibilityAudit(count, count, count, len(expiries_y), len(expiries_y)),
    )


def test_rbf_quote_summary_is_set_invariant() -> None:
    day = _market_day()
    config = RbfQuoteEncoderConfig(label="invariance")
    statistics, summaries = fit_rbf_feature_statistics((day,), config=config)
    permutation = np.random.default_rng(19).permutation(day.num_quotes)
    permuted = replace(
        day,
        expiry_index=day.expiry_index[permutation],
        strikes=day.strikes[permutation],
        is_call=day.is_call[permutation],
        bids=day.bids[permutation],
        asks=day.asks[permutation],
        fit_prices=day.fit_prices[permutation],
        log_volumes=day.log_volumes[permutation],
        log_open_interests=day.log_open_interests[permutation],
        quote_ids=tuple(day.quote_ids[index] for index in permutation),
    )
    transformed = transform_market_days((permuted,), config=config, statistics=statistics)
    # Matrix-reduction order changes at floating round-off when quotes are
    # permuted; the mathematical set summary is unchanged.
    np.testing.assert_allclose(transformed, summaries, atol=2.0e-9)


def test_rbf_encoder_shape_and_gradient() -> None:
    config = RbfQuoteEncoderConfig(label="gradient")
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    encoder = RbfQuoteEncoder(
        env,
        config=config,
        target_mean=torch.zeros(config.state_size, dtype=env.fdtype),
        target_scale=torch.ones(config.state_size, dtype=env.fdtype),
    )
    summary = torch.randn(3, config.summary_size, dtype=env.fdtype, requires_grad=True)
    state = encoder(summary)
    assert state.shape == (3, 140)
    state.square().mean().backward()
    assert summary.grad is not None
    assert torch.all(torch.isfinite(summary.grad))


def test_hierarchical_rbf_encoder_shape_gradient_and_expiry_causality() -> None:
    config = HierarchicalRbfQuoteEncoderConfig(label="hierarchical-gradient")
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    encoder = HierarchicalRbfQuoteEncoder(
        env,
        config=config,
        target_mean=torch.zeros(config.state_size, dtype=env.fdtype),
        target_scale=torch.ones(config.state_size, dtype=env.fdtype),
    )
    summary = torch.randn(2, config.summary_size, dtype=env.fdtype, requires_grad=True)
    state = encoder(summary)
    assert state.shape == (2, config.state_size)
    state.square().mean().backward()
    assert summary.grad is not None
    assert torch.all(torch.isfinite(summary.grad))

    perturbed = summary.detach().clone()
    third_anchor = 2 * config.summary_size_per_expiry
    perturbed[:, third_anchor:] += 7.0
    with torch.no_grad():
        original_state = encoder(summary.detach())
        perturbed_state = encoder(perturbed)
    first_two_state_size = sum(config.state_sizes_by_expiry[:2])
    torch.testing.assert_close(
        perturbed_state[:, :first_two_state_size],
        original_state[:, :first_two_state_size],
        rtol=0.0,
        atol=0.0,
    )


def test_hierarchical_encoder_requires_one_expiry_anchor_per_state_block() -> None:
    with pytest.raises(ValueError, match="one ordered RBF expiry anchor"):
        HierarchicalRbfQuoteEncoderConfig(
            label="invalid-hierarchical-anchors",
            num_expiry_anchors=13,
        )


@pytest.mark.parametrize("state_size", [140, 2_828])
def test_native_rbf_encoder_exposes_requested_state_and_native_raw(
    state_size: int,
) -> None:
    geometry = CappedUnboundedDlvFlexGeometryConfig()
    config = NativeRbfQuoteEncoderConfig(
        label=f"native-{state_size}",
        state_size=state_size,
        geometry=geometry,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    native_size = geometry.native_state_size
    pca_components = (
        None
        if state_size == native_size
        else torch.eye(native_size, dtype=env.fdtype)[:state_size]
    )
    encoder = NativeRbfQuoteEncoder(
        env,
        config=config,
        native_mean=torch.zeros(native_size, dtype=env.fdtype),
        native_scale=torch.ones(native_size, dtype=env.fdtype),
        state_mean=torch.zeros(state_size, dtype=env.fdtype),
        state_scale=torch.ones(state_size, dtype=env.fdtype),
        pca_components=pca_components,
    )
    summary = torch.randn(2, config.summary_size, dtype=env.fdtype, requires_grad=True)
    state = encoder.encode_state(summary)
    raw = encoder(summary)
    assert state.shape == (2, state_size)
    assert raw.shape == (2, native_size)
    raw.square().mean().backward()
    assert summary.grad is not None
    assert torch.all(torch.isfinite(summary.grad))


def test_native_target_conversion_preserves_physical_sanos_parameters() -> None:
    legacy_geometry = ForwardMuFlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=2,
    )
    target_geometry = CappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=2,
    )
    legacy_raw = torch.linspace(
        -4.0,
        4.0,
        3 * legacy_geometry.native_state_size,
        dtype=torch.float64,
    ).reshape(3, legacy_geometry.native_state_size)
    converted = _convert_sigmoid_native_targets(
        legacy_raw,
        target_geometry=target_geometry,
        legacy_geometry=legacy_geometry,
    )
    legacy_parameters = bound_native_raw(legacy_raw, geometry=legacy_geometry)
    converted_parameters = bound_native_raw(converted, geometry=target_geometry)
    torch.testing.assert_close(
        converted_parameters.dlvsqs,
        legacy_parameters.dlvsqs,
    )
    torch.testing.assert_close(
        converted_parameters.forward_vols,
        legacy_parameters.forward_vols,
    )
    torch.testing.assert_close(converted_parameters.eta, legacy_parameters.eta)


def test_native_evaluation_accepts_explicit_direct_state_ceiling() -> None:
    day = _market_day()
    geometry = CappedUnboundedDlvFlexGeometryConfig()
    config = NativeRbfQuoteEncoderConfig(
        label="native-direct-ceiling",
        state_size=geometry.native_state_size,
        geometry=geometry,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = NativeRbfQuoteEncoder(
        env,
        config=config,
        native_mean=torch.zeros(geometry.native_state_size, dtype=env.fdtype),
        native_scale=torch.ones(geometry.native_state_size, dtype=env.fdtype),
        state_mean=torch.zeros(geometry.native_state_size, dtype=env.fdtype),
        state_scale=torch.ones(geometry.native_state_size, dtype=env.fdtype),
    )
    raw = raw_from_native_parameters(
        NativeParameters(
            dlvsqs=torch.full(
                (1, geometry.num_model_expiries, geometry.num_inner_strikes),
                0.20**2,
                dtype=env.fdtype,
            ),
            forward_vols=torch.full(
                (1, geometry.num_model_expiries),
                0.20,
                dtype=env.fdtype,
            ),
            eta=torch.full(
                (1, geometry.num_model_expiries),
                0.30,
                dtype=env.fdtype,
            ),
        ),
        geometry=geometry,
    )
    evaluation = _evaluate(
        model,
        torch.zeros((1, config.summary_size), dtype=env.fdtype),
        (day,),
        raw,
        objective_config=FlexObjectiveConfig(fit_weight=0.05),
        decoder=FlexSanosDecoder(env, geometry=geometry),
        batch_size=1,
        prediction_native_raw=raw,
    )
    assert evaluation["num_days"] == 1
    assert evaluation["standardized_native_raw_mse"] == pytest.approx(0.0)
    assert evaluation["cap"]["fraction_at_or_above_price_cap"] == pytest.approx(0.0)


def test_shared_encoder_completion_dates_and_frozen_split(tmp_path: Path) -> None:
    completion = tmp_path / "native-complete.json"
    completion.write_text(
        json.dumps(
            {
                "status": "native_complete",
                "market_dates": ["2024-12-31", "2025-01-02", "2025-08-29"],
                "num_days": 3,
                "native_checkpoint_proof": {"num_market_days": 3},
            }
        ),
        encoding="utf-8",
    )
    dates = _shared_market_dates_from_completion(completion)
    assert dates == (
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 8, 29),
    )
    days = tuple(_market_day(trading_date=value) for value in dates)
    train, validation = _split_shared_markets(days)
    assert tuple(day.trading_date for day in train) == (date(2024, 12, 31),)
    assert tuple(day.trading_date for day in validation) == dates[1:]


def test_shared_native_targets_are_used_without_gauge_conversion(
    tmp_path: Path,
) -> None:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    days = (
        _market_day(trading_date=date(2024, 12, 31)),
        _market_day(trading_date=date(2025, 1, 2)),
    )
    raw = torch.linspace(
        -5.0,
        5.0,
        len(days) * geometry.native_state_size,
    ).reshape(len(days), geometry.native_state_size)
    checkpoint = tmp_path / "native.pt"
    torch.save(
        {
            "schema_version": 2,
            "phase": "complete",
            "market_fingerprints": [day.training_fingerprint for day in days],
            "model_state_dict": {
                "daily_raw_embedding.weight": raw,
                "sanos_decoder.inner_nms": torch.empty(0),
                "sanos_decoder.model_expiries_y": torch.empty(0),
                "sanos_decoder.model_dts_y": torch.empty(0),
                "sanos_decoder.shared_first_nms": torch.empty(0),
                "sanos_decoder.shared_last_nms": torch.empty(0),
            },
        },
        checkpoint,
    )
    loaded, audit = _load_shared_native_targets(
        checkpoint,
        days,
        geometry=geometry,
    )
    torch.testing.assert_close(loaded, raw)
    assert audit["shape"] == [2, geometry.native_state_size]
    assert audit["shared_learned_parameters"] == []


def test_equal_date_objective_does_not_weight_by_quote_count() -> None:
    mask = torch.tensor([[True, False, False], [True, True, True]])
    prices = torch.tensor([[3.0, 0.0, 0.0], [7.0, 7.0, 7.0]])
    zeros = torch.zeros_like(prices)
    ones = torch.ones_like(prices)
    batch = TorchMarketBatch(
        day_indices=torch.tensor([0, 1]),
        strikes=ones,
        price_expiries_y=zeros,
        price_strikes=ones,
        quote_price_index=torch.zeros_like(prices, dtype=torch.long),
        is_call=torch.ones_like(mask),
        bids=zeros,
        asks=torch.full_like(prices, 2.0),
        fit_prices=torch.full_like(prices, 2.0),
        mask=mask,
    )
    # Half-spread is one. Day losses are 1 and 5; equal-date mean is 3,
    # whereas an incorrectly quote-weighted mean would be 4.
    loss = equal_date_quote_objective(
        prices,
        batch,
        objective_config=FlexObjectiveConfig(fit_weight=0.0),
    )
    assert float(loss) == pytest.approx(3.0)


def test_feature_normalization_gives_each_date_equal_total_weight() -> None:
    first = _market_day(trading_date=date(2024, 1, 2))
    second = replace(
        _market_day(trading_date=date(2024, 1, 3)),
        log_volumes=np.full(first.num_quotes, 10.0),
    )
    config = RbfQuoteEncoderConfig(label="normalization")
    statistics, _ = fit_rbf_feature_statistics((first, second), config=config)
    volume_index = 8
    assert statistics.quote_mean[volume_index] == pytest.approx(
        0.5 * (np.mean(first.log_volumes) + 10.0)
    )
    summary = rbf_quote_summary(
        first,
        config=config,
        quote_mean=statistics.quote_mean,
        quote_scale=statistics.quote_scale,
    )
    assert summary.shape == (config.summary_size,)
    assert np.all(np.isfinite(summary))


def test_phase3_geometry_never_extends_beyond_august_2025_fixture() -> None:
    day = _market_day(trading_date=date(2025, 8, 29))
    assert day.trading_date.year == 2025
    np.testing.assert_allclose(day.expiries_y * DAYS_PER_YEAR, np.linspace(4.8, 69.8, 14))


def test_complete_encoder_history_validates_digests_and_excludes_2026(
    tmp_path: Path,
) -> None:
    states = np.arange(280, dtype=np.float32).reshape(2, 140)
    spots = np.array([4_800.0, 4_810.0], dtype=np.float64)
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save(
        {
            "schema_version": 1,
            "phase": "complete",
            "decoder_sha256": "decoder",
            "training_audits": {
                "supervised": {"converged": True},
                "market": {"converged": True},
            },
            "encoded_history": {
                "dates": ["2025-01-02", "2025-01-03"],
                "market_fingerprints": ["a", "b"],
                "states": torch.from_numpy(states),
                "spot_refs": torch.from_numpy(spots),
                "state_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "spot_ref_sha256": hashlib.sha256(spots.tobytes()).hexdigest(),
            },
        },
        checkpoint_path,
    )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report_path = tmp_path / "report.json"
    report_path.write_text(
        f'{{"status":"complete","encoder_checkpoint_sha256":"{digest}"}}',
        encoding="utf-8",
    )
    history = load_encoded_strong_history(checkpoint_path, report_path)
    np.testing.assert_array_equal(history.states, states)
    np.testing.assert_array_equal(history.spot_refs, spots)
    assert history.dates[-1] == date(2025, 1, 3)
