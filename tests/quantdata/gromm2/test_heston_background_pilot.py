from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.dysanos.bates_crc.heston_riccati import (
    GaussianJumpParams,
    HestonParams,
)
from quantdata.dysanos.bates_crc.real_data_study.fast_bates_pricer import (
    price_calls_ordinary_bates,
)
from quantdata.gromm2.calibration import (
    DlvCalibrationConfig,
    _black_call_kernels,
    _checkpoint_identity,
    _load_call_kernel_specification,
    _load_warm_start_parameters,
    _price_call_kernels,
)


def test_zero_black_backbone_kernel_is_intrinsic_value() -> None:
    supports = np.asarray([[0.8, 1.0, 1.2]])
    strikes = np.asarray([[0.9, 1.1]])
    kernels = _black_call_kernels(
        supports,
        strikes,
        np.asarray([5.0 / 252.0]),
        annual_volatility=0.0,
    )
    expected = np.maximum(supports[:, None, :] - strikes[:, :, None], 0.0)
    assert np.array_equal(kernels, expected)


def test_zero_backbone_is_a_valid_calibration_configuration() -> None:
    assert DlvCalibrationConfig(backbone_annual_volatility=0.0).backbone_annual_volatility == 0.0
from quantdata.gromm2.heston_background_pilot import (
    HestonBackground,
    _heston_calls_one_expiry,
    _heston_calls_one_expiry_fft,
    heston_return_cdf_grid_fft,
    scale_heston_background,
    write_call_kernel_bundle,
    write_call_kernel_grid_bundle,
)


def test_heston_pilot_pricer_matches_verified_fourier_pricer() -> None:
    background = HestonBackground(
        mean_reversion=8.0,
        long_run_variance=0.05**2,
        volatility_of_variance=0.25,
        correlation=-0.6,
        initial_variance=0.05**2,
    )
    strikes = np.asarray([0.94, 0.98, 1.0, 1.03, 1.08])
    expiry = 20.0 / 252.0
    actual = _heston_calls_one_expiry(
        background,
        strikes=strikes,
        expiry_y=expiry,
        quadrature_order=400,
        frequency_limit=1_000.0,
    )
    expected = price_calls_ordinary_bates(
        HestonParams(8.0, 0.05**2, 0.25, -0.6),
        GaussianJumpParams(0.0, 0.0, 0.1),
        0.05**2,
        strikes,
        np.full_like(strikes, expiry),
        n_quad=400,
        u_max=1_000.0,
    )
    assert np.allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_heston_pilot_pricer_respects_call_bounds() -> None:
    background = HestonBackground(
        mean_reversion=12.0,
        long_run_variance=0.05**2,
        volatility_of_variance=0.30,
        correlation=-0.65,
        initial_variance=0.05**2,
    )
    strikes = np.linspace(0.8, 1.2, 41)
    prices = _heston_calls_one_expiry(
        background,
        strikes=strikes,
        expiry_y=1.0 / 252.0,
        quadrature_order=600,
        frequency_limit=1_500.0,
    )
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    assert np.all(prices >= intrinsic)
    assert np.all(prices <= 1.0)
    assert np.all(np.diff(prices) <= 0.0)


def test_heston_fft_matches_verified_fourier_pricer() -> None:
    background = HestonBackground(
        mean_reversion=8.0,
        long_run_variance=0.05**2,
        volatility_of_variance=0.25,
        correlation=-0.6,
        initial_variance=0.05**2,
    )
    strikes = np.linspace(0.85, 1.15, 31)
    for expiry in (1.0 / 252.0, 5.0 / 252.0, 25.0 / 252.0, 125.0 / 252.0):
        actual = _heston_calls_one_expiry_fft(
            background,
            strikes=strikes,
            expiry_y=expiry,
        )
        expected = price_calls_ordinary_bates(
            background.heston(),
            GaussianJumpParams(0.0, 0.0, 0.1),
            background.initial_variance,
            strikes,
            np.full_like(strikes, expiry),
            n_quad=600,
            u_max=1_500.0,
        )
        assert np.allclose(actual, expected, atol=3.0e-7, rtol=0.0)


def test_heston_fft_cdf_is_monotone_and_matches_call_spread() -> None:
    background = HestonBackground(
        mean_reversion=8.0,
        long_run_variance=0.05**2,
        volatility_of_variance=0.25,
        correlation=-0.6,
        initial_variance=0.05**2,
    )
    expiry = 5.0 / 252.0
    log_strikes, cumulative = heston_return_cdf_grid_fft(
        background,
        expiry_y=expiry,
    )
    assert np.all(np.diff(cumulative) >= 0.0)
    assert cumulative[0] >= 0.0
    assert cumulative[-1] <= 1.0
    width = 1.0e-4
    calls = _heston_calls_one_expiry_fft(
        background,
        strikes=np.asarray([1.0 - width, 1.0 + width]),
        expiry_y=expiry,
    )
    expected = 1.0 + (calls[1] - calls[0]) / (2.0 * width)
    actual = np.interp(0.0, log_strikes, cumulative)
    assert abs(actual - expected) < 1.0e-4


def test_scale_heston_background_scales_variance_state_consistently() -> None:
    background = HestonBackground(
        mean_reversion=4.0,
        long_run_variance=0.12**2,
        volatility_of_variance=0.60,
        correlation=-0.7,
        initial_variance=0.10**2,
    )
    scaled = scale_heston_background(background, volatility_scale=0.5)
    assert scaled.mean_reversion == background.mean_reversion
    assert scaled.correlation == background.correlation
    assert scaled.long_run_variance == 0.25 * background.long_run_variance
    assert scaled.initial_variance == 0.25 * background.initial_variance
    assert scaled.volatility_of_variance == 0.5 * background.volatility_of_variance


def test_heston_call_kernel_bundle_round_trip(tmp_path: Path) -> None:
    background = HestonBackground(
        mean_reversion=4.0,
        long_run_variance=0.03**2,
        volatility_of_variance=0.20,
        correlation=-0.8,
        initial_variance=0.03**2,
    )
    supports = np.asarray([[0.5, 1.0, 1.5]])
    strikes = np.asarray([[0.9, 1.1]])
    expiries = np.asarray([1.0 / 252.0])
    kernels = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    path = tmp_path / "kernels.npz"
    identity = write_call_kernel_bundle(
        path,
        kernels=kernels,
        supports=supports,
        strikes=strikes,
        expiries_y=expiries,
        background=background,
        reference_date="2025-08-29",
    )
    with np.load(path, allow_pickle=False) as values:
        assert str(values["schema"].item()) == "gromm2-call-kernels-v1"
        assert str(values["identity"].item()) == identity
        assert np.array_equal(values["call_kernels"], kernels.astype(np.float32))
        metadata = json.loads(str(values["metadata"].item()))
    assert metadata["background"]["correlation"] == background.correlation


def test_custom_kernel_changes_affine_checkpoint_identity() -> None:
    config = DlvCalibrationConfig()
    legacy = _checkpoint_identity(config)
    assert legacy == _checkpoint_identity(config, data_identity="black-default")
    assert legacy != _checkpoint_identity(config, data_identity="heston-prefix")


def test_warm_start_checkpoint_loads_parameters_and_validates_shapes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "warm-start.npz"
    np.savez(
        path,
        schema=np.asarray("gromm2-affine-jax-checkpoint-v1"),
        identity=np.asarray("source-identity"),
        completed_epochs=np.asarray(17, dtype=np.int32),
        parameter_mean=np.ones((2, 3), dtype=np.float32),
        parameter_codes=np.ones((4, 5), dtype=np.float32),
        parameter_loadings=np.ones((5, 2, 3), dtype=np.float32),
    )
    parameters, identity = _load_warm_start_parameters(
        path,
        expected_shapes={
            "mean": (2, 3),
            "codes": (4, 5),
            "loadings": (5, 2, 3),
        },
    )
    assert parameters["codes"].shape == (4, 5)
    assert identity == "source-identity:completed-epochs=17"
    with pytest.raises(ValueError, match="does not match"):
        _load_warm_start_parameters(
            path,
            expected_shapes={
                "mean": (2, 3),
                "codes": (3, 5),
                "loadings": (5, 2, 3),
            },
        )


def test_rolling_warm_start_shifts_date_codes_causally(tmp_path: Path) -> None:
    checkpoint = tmp_path / "warm-start.npz"
    np.savez(
        checkpoint,
        schema=np.asarray("gromm2-affine-jax-checkpoint-v1"),
        identity=np.asarray("prior-fit"),
        completed_epochs=np.asarray(20, dtype=np.int32),
        parameter_mean=np.ones((2, 3), dtype=np.float32),
        parameter_codes=np.arange(12, dtype=np.float32).reshape(4, 3),
        parameter_loadings=np.ones((3, 2, 3), dtype=np.float32),
    )
    expected_shapes = {"mean": (2, 3), "codes": (4, 3), "loadings": (3, 2, 3)}
    source_dates = np.asarray(
        ["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28"]
    )
    target_dates = np.asarray(
        ["2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01"]
    )
    parameters, identity = _load_warm_start_parameters(
        checkpoint,
        expected_shapes=expected_shapes,
        source_dates=source_dates,
        target_dates=target_dates,
    )
    np.testing.assert_array_equal(
        parameters["codes"],
        np.asarray([[3, 4, 5], [6, 7, 8], [9, 10, 11], [9, 10, 11]]),
    )
    assert identity == "prior-fit:completed-epochs=20:rolled=2024-03-28->2024-04-01"
    with pytest.raises(ValueError, match="date windows do not align"):
        _load_warm_start_parameters(
            checkpoint,
            expected_shapes=expected_shapes,
            source_dates=source_dates,
            target_dates=np.asarray(
                ["2024-03-25", "2024-03-27", "2024-03-28", "2024-04-01"]
            ),
        )


def test_date_varying_kernel_bundle_interpolates_prices(tmp_path: Path) -> None:
    background = HestonBackground(
        mean_reversion=4.0,
        long_run_variance=0.03**2,
        volatility_of_variance=0.20,
        correlation=-0.8,
        initial_variance=0.03**2,
    )
    supports = np.asarray([[0.5, 1.0, 1.5]])
    strikes = np.asarray([[0.9, 1.1]])
    expiries = np.asarray([1.0 / 252.0])
    variance_grid = np.asarray([0.01, 0.04, 0.09])
    kernel_grid = np.stack(
        [
            np.full((1, 2, 3), variance, dtype=np.float64)
            for variance in variance_grid
        ]
    )
    dates = np.asarray(["2025-08-28", "2025-08-29"])
    date_variances = np.asarray([0.025, 0.065])
    path = tmp_path / "kernel-grid.npz"
    write_call_kernel_grid_bundle(
        path,
        kernel_grid=kernel_grid,
        variance_grid=variance_grid,
        date_variances=date_variances,
        dates=dates,
        supports=supports,
        strikes=strikes,
        expiries_y=expiries,
        background=background,
        reference_date="2025-08-29",
    )
    specification = _load_call_kernel_specification(
        path,
        dates=dates,
        expected_shape=(1, 2, 3),
    )
    densities = np.asarray([[[0.2, 0.3, 0.5]], [[0.1, 0.8, 0.1]]])
    prices = np.asarray(
        _price_call_kernels(
            specification.kernels,
            densities,
            lower_indices=specification.lower_indices,
            interpolation_weights=specification.interpolation_weights,
        )
    )
    assert prices[:, 0, 0] == pytest.approx(date_variances)
    assert prices[:, 0, 1] == pytest.approx(date_variances)


def test_date_varying_kernel_bundle_allows_chronological_prefix(
    tmp_path: Path,
) -> None:
    background = HestonBackground(
        mean_reversion=4.0,
        long_run_variance=0.03**2,
        volatility_of_variance=0.20,
        correlation=-0.8,
        initial_variance=0.03**2,
    )
    dates = np.asarray(["2025-08-27", "2025-08-28", "2025-08-29"])
    path = tmp_path / "kernel-grid.npz"
    write_call_kernel_grid_bundle(
        path,
        kernel_grid=np.zeros((2, 1, 1, 1)),
        variance_grid=np.asarray([0.01, 0.04]),
        date_variances=np.asarray([0.01, 0.02, 0.04]),
        dates=dates,
        supports=np.ones((1, 1)),
        strikes=np.ones((1, 1)),
        expiries_y=np.asarray([1.0 / 252.0]),
        background=background,
        reference_date="2025-08-29",
    )
    specification = _load_call_kernel_specification(
        path,
        dates=dates[:2],
        expected_shape=(1, 1, 1),
    )
    assert specification.lower_indices is not None
    assert len(specification.lower_indices) == 2


def test_date_varying_kernel_gradient_matches_explicit_interpolation() -> None:
    kernels = jnp.arange(3 * 2 * 2 * 3, dtype=jnp.float32).reshape(3, 2, 2, 3)
    densities = jnp.asarray(
        [
            [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
            [[0.4, 0.4, 0.2], [0.6, 0.1, 0.3]],
        ]
    )
    lower_indices = jnp.asarray([0, 1], dtype=jnp.int32)
    weights = jnp.asarray([0.25, 0.75])

    def streamed(values: jax.Array) -> jax.Array:
        return jnp.sum(
            _price_call_kernels(
                kernels,
                values,
                lower_indices=lower_indices,
                interpolation_weights=weights,
            )
            ** 2
        )

    interpolated = jnp.stack(
        [
            kernels[0] + weights[0] * (kernels[1] - kernels[0]),
            kernels[1] + weights[1] * (kernels[2] - kernels[1]),
        ]
    )

    def explicit(values: jax.Array) -> jax.Array:
        return jnp.sum(jnp.einsum("djkn,djn->djk", interpolated, values) ** 2)

    assert np.asarray(streamed(densities)) == pytest.approx(
        np.asarray(explicit(densities))
    )
    assert np.asarray(jax.grad(streamed)(densities)) == pytest.approx(
        np.asarray(jax.grad(explicit)(densities))
    )
