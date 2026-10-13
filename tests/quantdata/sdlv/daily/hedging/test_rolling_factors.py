from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    FixedContractDlvHistory,
    RollingDlvFactorConfig,
    RollingDlvFactorFit,
    RollingFactorAlignmentConfig,
    align_fixed_contract_pair,
    align_rolling_dlv_factors,
    estimate_front_spot_market_price_of_risk,
    filtered_front_spot_innovations,
    fit_dlv_factors,
    raw_dlv_state_with_factor_bump,
)


def _market_day(
    *,
    trading_date: date,
    dlvsq: np.ndarray,
    support: np.ndarray,
    forward_over_spot: tuple[float, ...],
) -> PublicMarketDay:
    model_dtes = np.arange(1, dlvsq.shape[0] + 1, dtype=np.float64)
    minimum = 0.01**2
    return PublicMarketDay(
        trading_date=trading_date,
        expiry_dates=tuple(
            date(2023, 1, 10 + index) for index in range(dlvsq.shape[0])
        ),
        expiries_y=model_dtes / 252.0,
        business_dtes=model_dtes,
        is_am_settlement=np.zeros(dlvsq.shape[0], dtype=bool),
        forward_over_spot=np.asarray(forward_over_spot),
        discount_factors=np.ones(dlvsq.shape[0]),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.full(dlvsq.shape[0], 0.2),
        expiry_indices=np.asarray([0]),
        strikes_over_forward=np.asarray([1.0]),
        standardized_log_moneyness=np.asarray([0.0]),
        in_recommended_hedge_nms_population=np.asarray([True]),
        is_call=np.asarray([True]),
        half_spreads_over_spot=np.asarray([0.001]),
        sanos_prices_over_discounted_forward=np.asarray([0.05]),
        dlv_prices_over_discounted_forward=np.asarray([0.05]),
        sanos_model_expiries_y=np.asarray([1.0 / 252.0]),
        sanos_support_strikes=(np.asarray([0.8, 1.0, 1.2]),),
        sanos_kernel_total_variances=np.asarray([0.2**2 / 252.0]),
        sanos_densities=(np.asarray([0.25, 0.5, 0.25]),),
        sanos_smoothness=0.0,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=model_dtes / 252.0,
        dlv_support_strikes=support,
        dlv_raw_log_excess_dlvsq=np.log(dlvsq - minimum),
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )


def test_fixed_contract_pair_uses_spot_and_both_forward_curves() -> None:
    support = np.asarray([0.5, 0.8, 1.0, 1.2, 1.5])
    origin_dlvsq = np.asarray(
        [
            [0.04, 0.04, 0.04],
            [0.05, 0.06, 0.07],
            [0.08, 0.09, 0.10],
        ]
    )
    endpoint_dlvsq = np.asarray(
        [
            [0.04, 0.08, 0.16],
            [0.09, 0.18, 0.36],
            [0.16, 0.32, 0.64],
        ]
    )
    origin = _market_day(
        trading_date=date(2023, 1, 3),
        dlvsq=origin_dlvsq,
        support=support,
        forward_over_spot=(1.01, 1.02, 1.03),
    )
    endpoint = _market_day(
        trading_date=date(2023, 1, 4),
        dlvsq=endpoint_dlvsq,
        support=support,
        forward_over_spot=(1.015, 1.025, 1.035),
    )
    transition = OneDayTransition(
        origin=origin,
        next_date=endpoint.trading_date,
        forward_adjusted_spot_return=0.0,
        discount_factor=1.0,
        forward_over_spot=1.01,
    )
    pair = align_fixed_contract_pair(transition, endpoint)

    first_query = support[1:-1] * 1.02 / (1.01 * 1.015)
    expected_first = np.interp(
        first_query,
        support[1:-1],
        np.log(endpoint_dlvsq[0]),
    )
    assert pair.origin_log_dlvsq == pytest.approx(np.log(origin_dlvsq[1:]))
    assert pair.endpoint_log_dlvsq[0] == pytest.approx(expected_first)
    background_variance = origin.backbone_annual_volatility**2 / 252.0
    expected_cdf = float(
        pair.front_density
        @ ndtr(
            (
                np.log(transition.terminal_normalized_spot / pair.front_support)
                + 0.5 * background_variance
            )
            / np.sqrt(background_variance)
        )
    )
    assert pair.front_cdf == pytest.approx(expected_cdf)
    assert pair.spot_gaussian_score == pytest.approx(ndtri(expected_cdf))
    assert pair.front_support == pytest.approx(support)
    assert pair.front_density.sum() == pytest.approx(1.0)
    assert pair.interpolation_clip_fraction == pytest.approx(1.0 / 3.0)
    assert pair.forward_tail_extrapolation_fraction == 0.0


def test_factor_fit_is_spot_oriented_and_obeys_as_of_boundary() -> None:
    sample_count = 12
    scores = np.linspace(-1.5, 1.5, sample_count)
    second = np.sin(np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False))
    second -= np.mean(second)
    second -= scores * float(second @ scores) / float(scores @ scores)
    origin = np.zeros((sample_count, 2, 3))
    spot_loading = np.asarray([[0.02, 0.03, 0.04], [0.05, 0.06, 0.07]])
    second_loading = np.asarray([[0.03, -0.02, 0.01], [-0.01, 0.02, -0.03]])
    endpoint = (
        0.2
        + 0.7 * origin
        + scores[:, None, None] * spot_loading
        + second[:, None, None] * second_loading
    )
    origins = tuple(date(2022, 1, 1 + index) for index in range(sample_count))
    endpoints = tuple(date(2022, 1, 2 + index) for index in range(sample_count))
    cdf = np.linspace(0.1, 0.9, sample_count)
    history = FixedContractDlvHistory(
        origin_dates=origins,
        endpoint_dates=endpoints,
        origin_log_dlvsq=origin,
        endpoint_log_dlvsq=endpoint,
        front_cdf_values=cdf,
        spot_gaussian_scores=scores,
        interpolation_clip_fractions=np.zeros(sample_count),
        forward_tail_extrapolation_fractions=np.zeros(sample_count),
    )
    config = RollingDlvFactorConfig(
        window_sessions=sample_count,
        factor_rank=2,
        smoothing_expiry_sigma=0.0,
        smoothing_strike_sigma=0.0,
    )
    fit = fit_dlv_factors(
        history,
        as_of_date=endpoints[-1],
        config=config,
    )
    standardized_scores = (scores - np.mean(scores)) / np.std(scores)
    expected_spot_loading = spot_loading * np.std(scores)
    assert fit.factor_scores[:, 0] == pytest.approx(standardized_scores)
    assert fit.daily_loadings[..., 0] == pytest.approx(expected_spot_loading)
    assert fit.factor_rank == 2
    assert fit.explained_variance_fraction == pytest.approx(1.0)

    masked = fit_dlv_factors(
        history,
        as_of_date=endpoints[-1],
        config=config,
        active_coordinate_mask=np.asarray(
            [[True, True, False], [True, True, False]]
        ),
    )
    assert masked.daily_loadings[:, -1] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="after the factor as-of date"):
        fit_dlv_factors(
            history,
            as_of_date=endpoints[-2],
            config=config,
        )


def test_factor_bump_leaves_front_slab_unchanged() -> None:
    sample_count = 4
    origins = tuple(date(2022, 1, 1 + index) for index in range(sample_count))
    endpoints = tuple(date(2022, 1, 2 + index) for index in range(sample_count))
    origin = np.arange(sample_count * 2 * 3, dtype=np.float64).reshape(
        sample_count, 2, 3
    )
    endpoint = origin + np.asarray([0.1, -0.1, 0.2, -0.2])[:, None, None]
    history = FixedContractDlvHistory(
        origin_dates=origins,
        endpoint_dates=endpoints,
        origin_log_dlvsq=origin,
        endpoint_log_dlvsq=endpoint,
        front_cdf_values=np.linspace(0.2, 0.8, sample_count),
        spot_gaussian_scores=np.asarray([-1.0, -0.2, 0.4, 1.1]),
        interpolation_clip_fractions=np.zeros(sample_count),
        forward_tail_extrapolation_fractions=np.zeros(sample_count),
    )
    fit = fit_dlv_factors(
        history,
        as_of_date=endpoints[-1],
        config=RollingDlvFactorConfig(
            window_sessions=sample_count,
            factor_rank=2,
            smoothing_expiry_sigma=0.0,
            smoothing_strike_sigma=0.0,
        ),
    )
    day = _market_day(
        trading_date=date(2023, 1, 3),
        dlvsq=np.full((3, 3), 0.04),
        support=np.asarray([0.5, 0.8, 1.0, 1.2, 1.5]),
        forward_over_spot=(1.0, 1.0, 1.0),
    )
    bumped = raw_dlv_state_with_factor_bump(
        day,
        fit,
        factor_index=0,
        standard_deviations=1.0,
    )
    decoded = day.minimum_dlvsq + np.exp(bumped)
    assert decoded[0] == pytest.approx(day.dlvsq[0])
    assert np.log(decoded[1:]) == pytest.approx(
        np.log(day.dlvsq[1:]) + fit.daily_loadings[..., 0]
    )


def _factor_fit(
    *,
    loadings: np.ndarray,
    scores: np.ndarray,
    as_of_date: date,
) -> RollingDlvFactorFit:
    sample_count = scores.shape[0]
    origins = tuple(date(2022, 1, 1 + index) for index in range(sample_count))
    endpoints = tuple(date(2022, 1, 2 + index) for index in range(sample_count))
    grid_shape = loadings.shape[:-1]
    return RollingDlvFactorFit(
        as_of_date=as_of_date,
        training_origin_dates=origins,
        training_endpoint_dates=endpoints,
        conditional_intercept=np.zeros(grid_shape),
        conditional_slope=np.ones(grid_shape),
        centered_residuals=np.zeros((sample_count, *grid_shape)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        daily_loadings=loadings,
        daily_factor_variances=np.sum(loadings.reshape(-1, loadings.shape[-1]) ** 2, axis=0),
        factor_scores=scores,
        explained_variance_fraction=1.0,
        interpolation_clip_fraction=0.0,
        forward_tail_extrapolation_fraction=0.0,
    )


def test_factor_alignment_recovers_surface_permutation_and_signs() -> None:
    flat_reference = np.asarray(
        [
            [0.3, 3.0, 0.0, 0.0],
            [0.2, 0.0, 2.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
            [0.4, 0.0, 0.0, 0.5],
        ]
    )
    reference_loadings = flat_reference.reshape(2, 2, 4)
    reference_scores = np.arange(32, dtype=np.float64).reshape(8, 4)
    surface_order = np.asarray([2, 0, 1])
    surface_signs = np.asarray([-1.0, 1.0, -1.0])
    current_loadings = np.column_stack(
        (
            flat_reference[:, 0],
            flat_reference[:, 1:][:, surface_order] * surface_signs,
        )
    ).reshape(2, 2, 4)
    current_scores = np.column_stack(
        (
            reference_scores[:, 0],
            reference_scores[:, 1:][:, surface_order] * surface_signs,
        )
    )
    reference = _factor_fit(
        loadings=reference_loadings,
        scores=reference_scores,
        as_of_date=date(2022, 1, 9),
    )
    current = _factor_fit(
        loadings=current_loadings,
        scores=current_scores,
        as_of_date=date(2022, 1, 10),
    )

    alignment = align_rolling_dlv_factors(
        reference,
        current,
        config=RollingFactorAlignmentConfig(degenerate_standard_errors=0.0),
    )

    assert alignment.fit.daily_loadings == pytest.approx(reference_loadings)
    assert alignment.fit.factor_scores == pytest.approx(reference_scores)
    assert alignment.surface_factor_permutation == (2, 3, 1)
    assert alignment.procrustes_blocks == ()
    assert alignment.post_procrustes_cosines == pytest.approx(np.ones(3))


def test_factor_alignment_rejects_future_reference() -> None:
    loadings = np.ones((2, 2, 2))
    scores = np.ones((8, 2))
    current = _factor_fit(
        loadings=loadings,
        scores=scores,
        as_of_date=date(2022, 1, 10),
    )
    future = _factor_fit(
        loadings=loadings,
        scores=scores,
        as_of_date=date(2022, 1, 11),
    )

    with pytest.raises(ValueError, match="strictly precede"):
        align_rolling_dlv_factors(future, current)


@pytest.mark.parametrize("true_risk_price", [2.0, 5.0])
def test_front_mixture_likelihood_recovers_risk_price(
    true_risk_price: float,
) -> None:
    sample_count = 30_000
    support = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
    density = np.asarray([0.06, 0.19, 0.5, 0.19, 0.06])
    cumulative = np.cumsum(density)
    generator = np.random.default_rng(173 + int(true_risk_price))
    physical_normal = generator.standard_normal(sample_count)
    shifted_normal = physical_normal + true_risk_price / np.sqrt(252.0)
    sampled_index = np.sum(
        ndtr(shifted_normal)[:, None] > cumulative[None, :],
        axis=1,
    )
    background_volatility = 0.05
    background_normal = generator.standard_normal(sample_count)
    terminal_spot = support[sampled_index] * np.exp(
        -0.5 * background_volatility**2 / 252.0
        + background_volatility / np.sqrt(252.0) * background_normal
    )
    first_origin = date(1900, 1, 1)
    origins = tuple(
        first_origin + timedelta(days=index) for index in range(sample_count)
    )
    endpoints = tuple(value + timedelta(days=1) for value in origins)
    history = FixedContractDlvHistory(
        origin_dates=origins,
        endpoint_dates=endpoints,
        origin_log_dlvsq=np.zeros((sample_count, 1, 1)),
        endpoint_log_dlvsq=np.zeros((sample_count, 1, 1)),
        front_cdf_values=np.full(sample_count, 0.5),
        spot_gaussian_scores=np.zeros(sample_count),
        interpolation_clip_fractions=np.zeros(sample_count),
        forward_tail_extrapolation_fractions=np.zeros(sample_count),
        front_supports=np.broadcast_to(support, (sample_count, len(support))).copy(),
        front_densities=np.broadcast_to(density, (sample_count, len(density))).copy(),
        terminal_normalized_spots=terminal_spot,
        background_annual_volatilities=np.full(
            sample_count, background_volatility
        ),
    )

    estimate, standard_error = estimate_front_spot_market_price_of_risk(history)
    filtered = filtered_front_spot_innovations(
        history,
        market_price_of_risk=estimate,
    )

    assert estimate == pytest.approx(true_risk_price, abs=0.2)
    assert standard_error > 0.0
    assert float(np.mean(filtered)) == pytest.approx(0.0, abs=0.02)


def test_partial_front_observation_does_not_duplicate_factor_covariance() -> None:
    sample_count = 800
    support = np.asarray([0.98, 1.0, 1.02])
    density = np.asarray([0.2, 0.6, 0.2])
    cumulative = np.cumsum(density)
    generator = np.random.default_rng(914)
    front_shock = generator.standard_normal(sample_count)
    surface_shock = generator.standard_normal(sample_count)
    sampled_index = np.sum(
        ndtr(front_shock)[:, None] > cumulative[None, :],
        axis=1,
    )
    background_volatility = 0.25
    background_shock = generator.standard_normal(sample_count)
    terminal_spot = support[sampled_index] * np.exp(
        -0.5 * background_volatility**2 / 252.0
        + background_volatility / np.sqrt(252.0) * background_shock
    )
    front_loading = np.asarray([0.012, 0.006, -0.004])
    surface_loading = np.asarray([0.003, -0.008, 0.010])
    increments = (
        front_shock[:, None] * front_loading[None]
        + surface_shock[:, None] * surface_loading[None]
    )
    first_origin = date(1900, 1, 1)
    origins = tuple(
        first_origin + timedelta(days=index) for index in range(sample_count)
    )
    endpoints = tuple(value + timedelta(days=1) for value in origins)
    history = FixedContractDlvHistory(
        origin_dates=origins,
        endpoint_dates=endpoints,
        origin_log_dlvsq=np.zeros((sample_count, 1, 3)),
        endpoint_log_dlvsq=increments[:, None, :],
        front_cdf_values=np.full(sample_count, 0.5),
        spot_gaussian_scores=np.zeros(sample_count),
        interpolation_clip_fractions=np.zeros(sample_count),
        forward_tail_extrapolation_fractions=np.zeros(sample_count),
        front_supports=np.broadcast_to(
            support,
            (sample_count, len(support)),
        ).copy(),
        front_densities=np.broadcast_to(
            density,
            (sample_count, len(density)),
        ).copy(),
        terminal_normalized_spots=terminal_spot,
        background_annual_volatilities=np.full(
            sample_count,
            background_volatility,
        ),
    )

    fit = fit_dlv_factors(
        history,
        as_of_date=endpoints[-1],
        config=RollingDlvFactorConfig(
            window_sessions=sample_count,
            factor_rank=2,
            smoothing_expiry_sigma=0.0,
            smoothing_strike_sigma=0.0,
        ),
    )

    empirical = increments - np.mean(increments, axis=0, keepdims=True)
    empirical_covariance = empirical.T @ empirical / sample_count
    fitted_loadings = fit.daily_loadings.reshape(3, 2)
    fitted_covariance = fitted_loadings @ fitted_loadings.T
    relative_error = np.linalg.norm(
        fitted_covariance - empirical_covariance,
        ord="fro",
    ) / np.linalg.norm(empirical_covariance, ord="fro")
    filtered = filtered_front_spot_innovations(
        history,
        market_price_of_risk=fit.front_spot_market_price_of_risk,
    )
    filtered -= np.mean(filtered)
    fitted_front = fitted_loadings[:, 0]
    old_residual = empirical - filtered[:, None] * fitted_front[None]
    old_values, old_vectors = np.linalg.eigh(
        old_residual.T @ old_residual / sample_count
    )
    old_surface = (
        old_vectors[:, -1:] * np.sqrt(max(float(old_values[-1]), 0.0))
    )
    old_covariance = (
        np.outer(fitted_front, fitted_front) + old_surface @ old_surface.T
    )
    old_relative_error = np.linalg.norm(
        old_covariance - empirical_covariance,
        ord="fro",
    ) / np.linalg.norm(empirical_covariance, ord="fro")
    assert relative_error < 0.12
    assert old_relative_error > 0.2


def test_degenerate_factor_alignment_preserves_covariance_and_reconstruction() -> None:
    inverse_square_root_two = 1.0 / np.sqrt(2.0)
    flat_reference = np.asarray(
        [
            [0.2, 1.0, 0.0],
            [0.1, 0.0, 1.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    )
    rotation = np.asarray(
        [
            [inverse_square_root_two, -inverse_square_root_two],
            [inverse_square_root_two, inverse_square_root_two],
        ]
    )
    reference_scores = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0],
            [0.2, -1.0, 0.0],
            [0.3, 0.0, -1.0],
        ]
    )
    current_flat = np.column_stack(
        (flat_reference[:, 0], flat_reference[:, 1:] @ rotation)
    )
    current_scores = np.column_stack(
        (reference_scores[:, 0], reference_scores[:, 1:] @ rotation)
    )
    reference = _factor_fit(
        loadings=flat_reference.reshape(2, 2, 3),
        scores=reference_scores,
        as_of_date=date(2022, 1, 5),
    )
    current = _factor_fit(
        loadings=current_flat.reshape(2, 2, 3),
        scores=current_scores,
        as_of_date=date(2022, 1, 6),
    )
    covariance_before = current_flat @ current_flat.T
    reconstruction_before = current_scores @ current_flat.T

    alignment = align_rolling_dlv_factors(reference, current)
    aligned_flat = alignment.fit.daily_loadings.reshape(-1, 3)

    assert alignment.procrustes_blocks == ((1, 2),)
    assert aligned_flat == pytest.approx(flat_reference)
    assert alignment.fit.factor_scores == pytest.approx(reference_scores)
    assert aligned_flat @ aligned_flat.T == pytest.approx(covariance_before)
    assert alignment.fit.factor_scores @ aligned_flat.T == pytest.approx(
        reconstruction_before
    )
