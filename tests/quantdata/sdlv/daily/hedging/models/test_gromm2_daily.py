from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dynamics import EstimatedDynamics
from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    expanding_supports,
    project_total_variance,
    transport_tower,
)
from quantdata.gromm2.market import (
    load_public_sanos_days,
    public_sanos_atm_total_variances,
    reconstruct_public_day,
)
from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm import (
    gromm2_daily as gromm2_daily_module,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    Gromm2MarketState,
    _digital_probability,
    _normalize_decoded_density_tower,
    generate_gromm2_daily_scenarios,
    load_gromm2_market_state,
    predict_physical_market_price_of_risk,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Artifacts,
    RollingGromm2Paths,
)

PUBLIC_CANDIDATE = Path(
    "/home/hans/quantdata_data/datasets/hedging/"
    "spx-2022-2025-hedging-public-candidate-v1"
)


def _dynamics() -> EstimatedDynamics:
    return EstimatedDynamics(
        log_dlvsq_loadings=np.zeros((2, 3, 2)),
        physical_ar_intercept=np.zeros((2, 3)),
        physical_ar_slope=np.ones((2, 3)),
        level_mean=np.zeros((2, 3)),
        level_basis=np.ones((6, 1)) / np.sqrt(6.0),
        level_scale=np.asarray([2.0]),
        physical_factor_coefficients=np.asarray(
            [
                [1.0, -1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
            ]
        ),
        ssm_input_mean=np.zeros(3),
        ssm_input_scale=np.ones(3),
        ssm_terminal_hidden=np.zeros(2),
        ssm_parameters={},
    )


def test_physical_risk_price_uses_current_level_and_previous_factor_score() -> None:
    dynamics = _dynamics()
    dlvsq = np.vstack((np.full((1, 3), 0.04), np.exp(np.ones((2, 3)))))

    actual = predict_physical_market_price_of_risk(
        dlvsq=dlvsq,
        dynamics=dynamics,
        previous_factor_score=np.asarray([0.5, -0.25]),
    )

    level_score = np.sqrt(6.0) / 2.0
    expected = (
        np.asarray([1.0, level_score, 0.5, -0.25])
        @ dynamics.physical_factor_coefficients
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_digital_probability_respects_discrete_settlement_convention() -> None:
    support = np.asarray([0.8, 1.0, 1.2])
    density = np.asarray([[0.2, 0.3, 0.5]])

    call = _digital_probability(
        support=support,
        density=density,
        background_level=np.ones(1),
        strike=1.0,
        remaining_days=0,
        right=OptionRight.CALL,
    )
    put = _digital_probability(
        support=support,
        density=density,
        background_level=np.ones(1),
        strike=1.0,
        remaining_days=0,
        right=OptionRight.PUT,
    )

    assert call[0] == pytest.approx(0.5)
    assert put[0] == pytest.approx(0.5)
    assert call[0] + put[0] == pytest.approx(1.0)


def test_density_normalization_only_accepts_bounded_decode_roundoff() -> None:
    raw = np.asarray(
        [
            [0.2, 0.3, 0.50000006],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )

    normalized, maximum_error, tolerance = _normalize_decoded_density_tower(raw)

    assert normalized.dtype == np.float64
    assert maximum_error > 2.0e-10
    assert maximum_error < tolerance
    np.testing.assert_allclose(
        np.sum(normalized, axis=-1),
        1.0,
        rtol=0.0,
        atol=2.0e-10,
    )
    with pytest.raises(ValueError, match="precision bound"):
        _normalize_decoded_density_tower(
            np.asarray([[0.2, 0.3, 0.501]], dtype=np.float32)
        )


def test_loader_normalizes_float32_artifact_before_q_replay(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    origin = date(2023, 1, 17)
    affine_history = tmp_path / "affine-history.h5"
    dynamics_path = tmp_path / "dynamics.npz"
    paths = RollingGromm2Paths(
        directory=tmp_path,
        source_history=tmp_path / "source.h5",
        prepared_books=tmp_path / "books.h5",
        affine_history=affine_history,
        affine_checkpoint=tmp_path / "checkpoint.npz",
        dynamics=dynamics_path,
        manifest=tmp_path / "manifest.json",
    )
    with h5py.File(affine_history, "w") as stream:
        stream.create_dataset(
            "supports",
            data=np.asarray(
                [
                    [0.8, 0.9, 1.0, 1.1, 1.2],
                    [0.7, 0.85, 1.0, 1.15, 1.3],
                    [0.6, 0.8, 1.0, 1.2, 1.4],
                ],
                dtype=np.float32,
            ),
        )
        stream.create_dataset(
            "transports",
            data=np.broadcast_to(np.eye(5, dtype=np.float32), (2, 5, 5)),
        )
        stream.create_dataset(
            "model_expiries_y",
            data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32) / 252.0,
        )
        stream.create_dataset(
            "model_business_dtes",
            data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )
        stream.create_dataset(
            "dlvsq",
            data=np.full((1, 3, 3), 0.04, dtype=np.float32),
        )
    np.savez_compressed(
        dynamics_path,
        factor_scores=np.zeros((1, 2)),
        physical_market_price_of_risk=np.zeros(2),
    )
    decoded = np.broadcast_to(
        np.asarray([0.1, 0.2, 0.3, 0.1, 0.30000002]),
        (3, 5),
    ).copy()
    monkeypatch.setattr(
        gromm2_daily_module,
        "decode_density_tower",
        lambda *args: decoded,
    )
    monkeypatch.setattr(
        gromm2_daily_module,
        "load_estimated_dynamics",
        lambda path: _dynamics(),
    )
    artifacts = RollingGromm2Artifacts(
        paths=paths,
        as_of_date=origin,
        training_dates=(origin,),
        percent_inside_spread=84.0,
        explained_variance_ratio=0.95,
        physical_projection_fraction=0.0,
        maximum_daily_risk_price_norm=0.0,
        physical_accepted=False,
        dataset_manifest_sha256="a" * 64,
    )

    state = load_gromm2_market_state(artifacts)

    assert state.density_tower.dtype == np.float64
    np.testing.assert_allclose(
        np.sum(state.density_tower, axis=-1),
        1.0,
        rtol=0.0,
        atol=2.0e-10,
    )
    diagnostic = capsys.readouterr().out
    assert '"serialized_dtypes": ["float32"]' in diagnostic
    assert '"maximum_mass_error_before_normalization"' in diagnostic


def test_gromm2_daily_scenario_smoke_uses_public_book_geometry() -> None:
    jax.config.update("jax_enable_x64", True)
    origin_date = date(2023, 1, 17)
    panel = PublicHedgingPanel(PUBLIC_CANDIDATE)
    public_day = next(
        day
        for day in load_public_sanos_days(PUBLIC_CANDIDATE)
        if day.trading_date == origin_date
    )
    geometry = Gromm2GeometryConfig(num_inner_strikes=16)
    expiries = geometry.model_business_dtes / 252.0
    raw_variances = public_sanos_atm_total_variances(public_day, expiries)
    projected = project_total_variance(
        expiries,
        raw_variances,
        forward_volatility_floor=geometry.forward_volatility_floor,
    )
    supports = expanding_supports(projected, config=geometry)
    reconstructed = reconstruct_public_day(
        public_day,
        supports=supports,
        model_expiries_y=expiries,
        geometry=geometry,
    )
    transports = transport_tower(supports)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(reconstructed.dlvsq),
            jnp.asarray(np.diff(expiries, prepend=0.0)),
        )
    )
    factor_count = 2
    coordinate_count = (len(expiries) - 1) * geometry.num_inner_strikes
    dynamics = EstimatedDynamics(
        log_dlvsq_loadings=np.zeros(
            (len(expiries) - 1, geometry.num_inner_strikes, factor_count)
        ),
        physical_ar_intercept=np.zeros(
            (len(expiries) - 1, geometry.num_inner_strikes)
        ),
        physical_ar_slope=np.ones(
            (len(expiries) - 1, geometry.num_inner_strikes)
        ),
        level_mean=np.zeros((len(expiries) - 1, geometry.num_inner_strikes)),
        level_basis=np.zeros((coordinate_count, 1)),
        level_scale=np.ones(1),
        physical_factor_coefficients=np.zeros((4, factor_count)),
        ssm_input_mean=np.zeros(3),
        ssm_input_scale=np.ones(3),
        ssm_terminal_hidden=np.zeros(2),
        ssm_parameters={},
    )
    state = Gromm2MarketState(
        as_of_date=origin_date,
        supports=supports,
        transports=transports,
        expiries_y=expiries,
        business_dtes=geometry.model_business_dtes,
        dlvsq=reconstructed.dlvsq,
        density_tower=tower,
        dynamics=dynamics,
        previous_factor_score=np.zeros(factor_count),
        physical_market_price_of_risk=np.zeros(factor_count),
        percent_inside_spread=83.0,
        explained_variance_ratio=0.9,
        physical_projection_fraction=0.6,
        physical_accepted=True,
    )

    scenarios = generate_gromm2_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.CALL,
            nms=0.0,
        ),
        market_state=state,
        config=GrommConfig(path_count=8, batch_size=8),
    )

    assert scenarios.instrument_returns.shape == (
        8,
        1 + len(scenarios.sample.hedge_quotes),
    )
    assert np.all(np.isfinite(scenarios.instrument_returns))
    assert np.all(np.isfinite(scenarios.claim_returns))
    assert scenarios.claim_model_price > 0.0
    np.testing.assert_allclose(
        scenarios.instrument_entry_prices[1:],
        [quote.mark for quote in scenarios.sample.hedge_quotes],
        rtol=0.0,
        atol=0.0,
    )

    physical = generate_gromm2_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=scenarios.target,
        market_state=state,
        config=GrommConfig(
            measure=GrommMeasure.P,
            path_count=8,
            batch_size=8,
        ),
    )
    np.testing.assert_array_equal(
        physical.instrument_returns,
        scenarios.instrument_returns,
    )

    shifted_dynamics = replace(
        dynamics,
        log_dlvsq_loadings=np.full_like(dynamics.log_dlvsq_loadings, 0.02),
    )
    shifted_state = replace(
        state,
        dynamics=shifted_dynamics,
        physical_market_price_of_risk=np.asarray([0.75, -0.4]),
    )
    shifted_q = generate_gromm2_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=scenarios.target,
        market_state=shifted_state,
        config=GrommConfig(path_count=8, batch_size=8),
    )
    shifted_p = generate_gromm2_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=scenarios.target,
        market_state=shifted_state,
        config=GrommConfig(
            measure=GrommMeasure.P,
            path_count=8,
            batch_size=8,
        ),
    )
    assert np.all(np.isfinite(shifted_p.instrument_returns))
    assert not np.array_equal(
        shifted_p.instrument_returns,
        shifted_q.instrument_returns,
    )
    np.testing.assert_array_equal(
        shifted_p.pure_instrument_returns,
        shifted_q.pure_instrument_returns,
    )
