from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dynamics import EstimatedDynamics
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyFitInstruction,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    Gromm2MarketState,
    predict_physical_market_price_of_risk,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_weekly import (
    generate_gromm2_weekly_scenarios,
    gromm2_weekly_training_data,
    live_gromm2_weekly_feature,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Artifacts,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.weekly_adapter import (
    GrommWeeklyAdapter,
    GrommWeeklyMarketFit,
)
from quantdata.torch import BaseEnvironment


def _market_state() -> Gromm2MarketState:
    jax.config.update("jax_enable_x64", True)
    day = synthetic_weekly_market_day()
    support = day.dlv_support_strikes
    supports = np.broadcast_to(support, (5, len(support))).copy()
    transports = np.broadcast_to(np.eye(len(support)), (4, len(support), len(support))).copy()
    dlvsq = day.dlvsq
    expiries = day.dlv_model_expiries_y
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(np.diff(expiries, prepend=0.0)),
        )
    )
    factors = 2
    coordinate_count = 4 * 3
    dynamics = EstimatedDynamics(
        log_dlvsq_loadings=np.zeros((4, 3, factors)),
        physical_ar_intercept=np.zeros((4, 3)),
        physical_ar_slope=np.ones((4, 3)),
        level_mean=np.log(dlvsq[1:]),
        level_basis=np.zeros((coordinate_count, 1)),
        level_scale=np.ones(1),
        physical_factor_coefficients=np.zeros((4, factors)),
        ssm_input_mean=np.zeros(3),
        ssm_input_scale=np.ones(3),
        ssm_terminal_hidden=np.zeros(factors),
        ssm_parameters={},
    )
    return Gromm2MarketState(
        as_of_date=day.trading_date,
        supports=supports,
        transports=transports,
        expiries_y=expiries,
        business_dtes=day.business_dtes,
        dlvsq=dlvsq,
        density_tower=tower,
        dynamics=dynamics,
        previous_factor_score=np.zeros(factors),
        physical_market_price_of_risk=np.zeros(factors),
        percent_inside_spread=85.0,
        explained_variance_ratio=0.9,
        physical_projection_fraction=0.6,
        physical_accepted=True,
    )


def _decision_state() -> WeeklyDecisionState:
    day = synthetic_weekly_market_day()
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        episode_origin_date=day.trading_date,
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        model="gromm",
        measure="Q",
        target_label="digital-call-5d-atm",
        claim_quantity=1.0,
        model_config_identity="market",
        policy_config_identity="policy",
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    return WeeklyDecisionState(
        instruction=instruction,
        decision_index=0,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=1.0,
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=np.ones(1),
        rolling_forward_path=np.ones(1),
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )


def test_physical_risk_price_supports_pathwise_states() -> None:
    market_state = _market_state()
    levels = np.broadcast_to(market_state.dlvsq[1:], (3, 4, 3))
    scores = np.zeros((3, 2))

    actual = predict_physical_market_price_of_risk(
        dlvsq=levels,
        dynamics=market_state.dynamics,
        previous_factor_score=scores,
    )

    assert actual.shape == (3, 2)
    np.testing.assert_array_equal(actual, np.zeros((3, 2)))


def test_gromm2_weekly_scenario_and_training_smoke() -> None:
    jax.config.update("jax_enable_x64", True)
    state = _decision_state()
    base_market_state = _market_state()
    market_state = replace(
        base_market_state,
        dlvsq=1.01 * base_market_state.dlvsq,
    )
    claim = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    generated = generate_gromm2_weekly_scenarios(
        market_state=market_state,
        state=state,
        claim=claim,
        config=GrommConfig(path_count=8, batch_size=8),
        model_fit_identity="weekly-smoke",
    )
    data = gromm2_weekly_training_data(generated, claim_quantity=1.0)

    assert generated.scenarios.spot_step_gains.shape == (8, 5)
    assert generated.scenarios.option_step_gains.shape[0:2] == (8, 5)
    assert data.features.shape[:2] == (8, 5)
    assert data.instrument_returns.shape == (
        8,
        5,
        1 + len(state.contracts),
    )
    assert data.linear_transaction_costs.shape == data.instrument_returns.shape
    assert np.all(np.isfinite(data.features))
    assert np.all(np.isfinite(data.instrument_returns))
    live_feature = live_gromm2_weekly_feature(
        market_state=market_state,
        state=state,
        claim=claim,
        measure=GrommMeasure.Q,
        initial_positions=generated.initial_positions,
    )
    np.testing.assert_allclose(
        generated.features[:, 0],
        np.broadcast_to(live_feature, generated.features[:, 0].shape),
        rtol=0.0,
        atol=1.0e-14,
    )

    shifted_state = replace(
        market_state,
        dynamics=replace(
            market_state.dynamics,
            log_dlvsq_loadings=np.full_like(
                market_state.dynamics.log_dlvsq_loadings,
                0.02,
            ),
        ),
        physical_market_price_of_risk=np.asarray([0.75, -0.4]),
    )
    shifted_q = generate_gromm2_weekly_scenarios(
        market_state=shifted_state,
        state=state,
        claim=claim,
        config=GrommConfig(path_count=8, batch_size=8),
        model_fit_identity="weekly-smoke-shifted-q",
    )
    physical = generate_gromm2_weekly_scenarios(
        market_state=shifted_state,
        state=state,
        claim=claim,
        config=GrommConfig(
            measure=GrommMeasure.P,
            path_count=8,
            batch_size=8,
        ),
        model_fit_identity="weekly-smoke-p",
    )
    assert not np.array_equal(
        physical.scenarios.option_step_gains,
        shifted_q.scenarios.option_step_gains,
    )
    np.testing.assert_array_equal(
        physical.pure_instrument_returns[:, 0],
        shifted_q.pure_instrument_returns[:, 0],
    )


def test_gromm2_weekly_adapter_fits_signed_policies_and_returns_action(
    tmp_path: Path,
) -> None:
    jax.config.update("jax_enable_x64", True)
    adapter = GrommWeeklyAdapter(
        artifact_root=tmp_path / "market",
        checkpoint_directory=tmp_path / "policy",
        config=GrommConfig(path_count=8, batch_size=8),
        train_config=GrommTrainConfig(
            maximum_epochs=1,
            minimum_epochs=1,
            patience_epochs=1,
            batch_pairs=3,
            checkpoint_epochs=1,
            progress_seconds=300.0,
        ),
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )
    base_state = _decision_state()
    base = base_state.instruction
    instruction = WeeklyFitInstruction(
        schedule=base.schedule,
        model=adapter.model,
        measure=adapter.measure,
        target_label=base.target_label,
        claim_quantity=base.claim_quantity,
        model_config_identity=adapter.model_config_identity,
        policy_config_identity=adapter.policy_config_identity,
        episode_origin_date=base.episode_origin_date,
        decision_date=base.decision_date,
        fit_as_of_date=base.fit_as_of_date,
        training_dates=base.training_dates,
        frozen_contract_ids=base.frozen_contract_ids,
    )
    state = replace(base_state, instruction=instruction)
    claim = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    market_fit = GrommWeeklyMarketFit(
        market_fit_identity=instruction.market_fit_identity,
        artifacts=cast(RollingGromm2Artifacts, None),
        market_state=_market_state(),
        affine_history_sha256="0" * 64,
        dynamics_sha256="1" * 64,
    )

    fitted = adapter.fit_policy(
        market_fit,
        instruction=instruction,
        claim=claim,
        contracts=state.contracts,
        state=state,
    )
    action = adapter.action(fitted, state=state)

    assert np.isfinite(fitted.prices.q_expectation_per_fit_spot)
    assert np.isfinite(fitted.prices.buyer_indifference_per_fit_spot)
    assert np.isfinite(fitted.prices.seller_indifference_per_fit_spot)
    assert np.isfinite(action.spot_position)
    assert action.option_positions.shape == (len(state.contracts),)
    assert fitted.hedge.short.checkpoint_path.is_file()
