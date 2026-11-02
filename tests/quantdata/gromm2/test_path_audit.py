from __future__ import annotations

from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dlv_variation_audit import audit_dlv_variation
from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower
from quantdata.gromm2.path_audit import (
    PathAuditConfig,
    audit_contracts,
    generate_audit_paths,
    normal_block,
    path_function,
)

jax.config.update("jax_enable_x64", True)


def small_model() -> tuple[dict, np.ndarray]:
    expiries = np.asarray([1., 2., 3., 5., 10.]) / 252.
    variances = .2**2 * expiries
    supports = expanding_supports(variances, config=Gromm2GeometryConfig(num_inner_strikes=16))
    transports = transport_tower(supports)
    dlvsq = np.full((5, 16), .18**2)
    densities = np.asarray(decode_density_tower(jnp.asarray(supports), jnp.asarray(transports), jnp.asarray(dlvsq), jnp.asarray(np.diff(expiries, prepend=0.))))
    return {'supports': supports, 'transports': transports, 'expiries_y': expiries, 'dlvsq': dlvsq,
            'densities': densities, 'loadings': np.full((4, 16, 2), .1)}, variances


def test_contracts_fix_strikes_and_normals_preserve_antithetic_pairs() -> None:
    arrays, variances = small_model()
    config = PathAuditConfig(paths=8, batch_size=4, days=10, nms=(-1., 0., 1.))
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=variances, config=config)
    np.testing.assert_array_equal(contracts['expiry_days'], [5, 10])
    np.testing.assert_allclose(contracts['strikes'], np.exp(np.sqrt(variances[[3, 4]])[:, None] * np.asarray(config.nms)))
    factor, background = normal_block(config=config, start=0, count=4, factors=2)
    np.testing.assert_array_equal(factor[:, :, ::2] + factor[:, :, 1::2], 0.)
    np.testing.assert_array_equal(background[:, ::2] + background[:, 1::2], 0.)
    assert not np.array_equal(factor, normal_block(config=config, start=4, count=4, factors=2)[0])


def test_irregular_front_roll_settlement_and_predictable_features() -> None:
    arrays, variances = small_model()
    config = PathAuditConfig(paths=4, batch_size=4, days=10, substeps=1, nms=(-1., 0., 1.))
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=variances, config=config)
    simulate = path_function(arrays=arrays, background_volatility=.05, contracts=contracts, config=config)
    normals, background = normal_block(config=config, start=0, count=4, factors=2)
    features, gains, spots, terminal, diagnostics = map(np.asarray, simulate(jnp.asarray(normals), jnp.asarray(background)))
    assert features.shape[:2] == (4, 10)
    assert gains.shape == (4, 10, 7)
    assert spots.shape == (4, 11)
    assert np.all(np.isfinite(features)) and np.all(np.isfinite(gains))
    np.testing.assert_allclose(gains[:, :, 0], np.diff(spots, axis=-1) / spots[:, :-1], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(gains[:, 5:, 1:4], 0.)
    np.testing.assert_allclose(terminal[:, 0], spots[:, -1] - 1., atol=1e-14)
    np.testing.assert_allclose(np.sum(gains[:, :, 1:] * spots[:, :-1, None], axis=1), terminal[:, 1:], atol=1e-13)
    assert np.max(diagnostics[:, 1:3]) < 1e-12
    changed = normals.copy()
    changed[5:] += 1.
    later_features, later_gains, *_ = map(np.asarray, simulate(jnp.asarray(changed), jnp.asarray(background)))
    np.testing.assert_array_equal(features[:, :6], later_features[:, :6])
    np.testing.assert_array_equal(gains[:, :5], later_gains[:, :5])


def test_committed_paths_resume_and_keep_float64_moments(tmp_path, monkeypatch) -> None:
    arrays, variances = small_model()
    config = PathAuditConfig(paths=4, batch_size=4, days=5, substeps=1, nms=(-1., 0., 1.))
    monkeypatch.setattr('quantdata.gromm2.path_audit.check_resource_headroom', lambda: None)
    metadata = {'model_label': 'test', 'background_volatility': .05}
    first = generate_audit_paths(arrays=arrays, metadata=metadata, reference_variances=variances,
                                output_directory=tmp_path, config=config)
    second = generate_audit_paths(arrays=arrays, metadata=metadata, reference_variances=variances,
                                 output_directory=tmp_path, config=config)
    assert first == second
    assert first['independent_pairs'] == 2
    assert first['daily']['tested_coordinates'] == 20
    with h5py.File(tmp_path / 'paths.h5', 'r') as stream:
        assert stream.attrs['completed_paths'] == 4
        assert stream['returns'].dtype == np.float32
        assert stream['daily_pair_moments'].dtype == np.float64
        returns = np.asarray(stream['returns'], dtype=np.float64)
        pairs = .5 * (returns[::2] + returns[1::2])
        np.testing.assert_allclose(stream['daily_pair_moments'][0, 0], pairs.mean(axis=0), atol=1e-9)
    assert replace(config, substeps=8).substeps == 8


@pytest.mark.parametrize('failure', ['nonfinite', 'storage_overflow'])
def test_failure_census_retains_paths_without_reporting_valid_population_metrics(tmp_path, monkeypatch, failure: str) -> None:
    arrays, variances = small_model()
    config = PathAuditConfig(paths=4, batch_size=4, days=5, retain_nonfinite=True, nms=(0.,))

    def simulated(*args):
        gains = np.zeros((4, 5, 2))
        gains[0, 2, 1] = np.nan if failure == 'nonfinite' else 1.e100
        diagnostics = np.zeros((5, 6))
        diagnostics[:, 4] = 1.
        health = np.zeros((4, 5), dtype=np.uint8)
        health[0, 2] = 1 if failure == 'nonfinite' else 32
        return np.zeros((4, 5, 3)), gains, np.ones((4, 6)), np.zeros((4, 2)), diagnostics, health

    monkeypatch.setattr('quantdata.gromm2.path_audit.path_function', lambda **kwargs: simulated)
    monkeypatch.setattr('quantdata.gromm2.path_audit.check_resource_headroom', lambda: None)
    with np.errstate(over='ignore', invalid='ignore'):
        report = generate_audit_paths(arrays=arrays, metadata={'background_volatility': .05},
            reference_variances=variances, output_directory=tmp_path, config=config)
    assert report['stability']['invalid_retained_paths'] == 1
    assert report['stability']['paths_dropped'] == 0
    assert report['status'].startswith('numerically_invalid')
    assert 'daily' not in report
    with h5py.File(tmp_path / 'paths.h5', 'r') as stream:
        assert stream.attrs['completed_paths'] == 4
        assert not stream['path_validity'][0]
        assert stream['first_invalid_day'][0] == 2
        assert not np.isfinite(stream['returns'][0, 2, 1])


def test_passive_variation_is_stored_and_compares_actual_input_without_changing_paths(tmp_path, monkeypatch):
    arrays, variances = small_model()
    base = PathAuditConfig(paths=4, batch_size=4, days=5, substeps=1,
                          retain_nonfinite=True, drift_taming=False, nms=(-1., 0., 1.))
    measured_config = replace(base, collect_dlv_variation=True)
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=variances, config=base)
    normals, background = normal_block(config=base, start=0, count=4, factors=2)
    plain = path_function(arrays=arrays, background_volatility=.05, contracts=contracts, config=base)
    measured = path_function(arrays=arrays, background_volatility=.05, contracts=contracts, config=measured_config)
    plain_values = jax.device_get(plain(jnp.asarray(normals), jnp.asarray(background)))
    measured_values = jax.device_get(measured(jnp.asarray(normals), jnp.asarray(background)))
    for left, right in zip(plain_values, measured_values[:-1]):
        np.testing.assert_allclose(left, right, rtol=1.e-13, atol=1.e-14)
    assert measured_values[-1].shape == (6, 5, 16)
    monkeypatch.setattr('quantdata.gromm2.path_audit.check_resource_headroom', lambda: None)
    generate_audit_paths(arrays=arrays, metadata={'background_volatility': .05}, reference_variances=variances,
                         output_directory=tmp_path / 'sample', config=measured_config)
    report = audit_dlv_variation(path=tmp_path / 'sample/paths.h5', output=tmp_path / 'audit')
    assert report['status'] == 'complete'
    assert report['independent_antithetic_pairs'] == 2
    assert report['maximum_log_update_identity_error'] < 1.e-12
    with np.load(tmp_path / 'audit/dlv-variation-statistics.npz') as saved:
        np.testing.assert_array_equal(saved['active_business_days'], [0, 1, 2, 3, 5])


def test_log_and_level_paths_match_in_moderate_states():
    arrays, variances = small_model()
    config = PathAuditConfig(paths=4, batch_size=4, days=5, substeps=8,
                            drift_taming=False, retain_nonfinite=True, nms=(-1., 0., 1.))
    contracts = audit_contracts(expiries_y=arrays['expiries_y'], reference_variances=variances, config=config)
    normals, background = normal_block(config=config, start=0, count=4, factors=2)
    log_simulator = path_function(arrays=arrays, background_volatility=.05, contracts=contracts, config=config)
    level_simulator = path_function(arrays=arrays, background_volatility=.05, contracts=contracts,
                                    config=replace(config, preserve_log_dlvsq=False))
    logarithmic = jax.device_get(log_simulator(jnp.asarray(normals), jnp.asarray(background)))
    ordinary = jax.device_get(level_simulator(jnp.asarray(normals), jnp.asarray(background)))
    for index in (0, 1, 2, 3, 5):
        np.testing.assert_allclose(logarithmic[index], ordinary[index], rtol=1.e-10, atol=2.e-13)
    np.testing.assert_allclose(logarithmic[4][:, :6], ordinary[4], rtol=1.e-10, atol=2.e-13)


def test_log_state_is_preserved_across_rolls_and_passive_variation(tmp_path, monkeypatch):
    arrays, variances = small_model()
    config = PathAuditConfig(paths=4, batch_size=4, days=5, substeps=1, drift_taming=False,
                            retain_nonfinite=True, collect_dlv_variation=True, nms=(-1., 0., 1.))

    def increments(*args, **kwargs):
        shape = args[3].shape
        return jnp.full(shape, -252000.), jnp.zeros(shape), jnp.zeros(shape)

    monkeypatch.setattr('quantdata.gromm2.simulation.fused_risk_neutral_increments', increments)
    monkeypatch.setattr('quantdata.gromm2.path_audit.check_resource_headroom', lambda: None)
    report = generate_audit_paths(arrays=arrays, metadata={'background_volatility': .05},
                                 reference_variances=variances, output_directory=tmp_path, config=config)
    assert report['stability']['invalid_retained_paths'] == 0
    with h5py.File(tmp_path / 'paths.h5') as saved:
        assert np.isfinite(saved['features'][:]).all()
        assert np.any(saved['batch_diagnostics'][0, :, 6] > 0)
        assert np.max(saved['dlv_variation_moments'][0, 5]) < 1.e-10
