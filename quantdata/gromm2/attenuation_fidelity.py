"""Prefix-only historical covariance checks for density attenuation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.biz_calendar import count_business_days_between
from quantdata.gromm2.estimation import _fixed_contract_endpoint_surface, _spot_gaussian_score
from quantdata.gromm2.rolling_calibration import (
    ar1_estimate,
    joint_innovation_loadings,
    write_json,
)


def covariance_score(observed: np.ndarray, forecast: np.ndarray) -> float:
    """Proper second-moment score, omitting the forecast-independent target norm."""
    return float((np.sum(forecast**2) - 2. * observed @ forecast @ observed) / len(observed)**2)


def block_mean_interval(values: np.ndarray, *, seed: int = 2026110102,
                        block_days: int = 5, replications: int = 10000) -> dict:
    """Circular moving-block interval for a paired historical score difference."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_days or not np.isfinite(values).all():
        raise ValueError('finite paired historical scores and enough days are required')
    generator = np.random.default_rng(seed)
    starts = generator.integers(len(values), size=(replications, (len(values) + block_days - 1) // block_days))
    indices = (starts[..., None] + np.arange(block_days)) % len(values)
    means = values[indices.reshape(replications, -1)[:, :len(values)]].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'ci95_lower': float(lower), 'ci95_upper': float(upper),
        'block_days': block_days, 'bootstrap_replications': replications, 'seed': seed}


def run_fidelity(*, calibration_root: Path, origin: str, output: Path,
                 holdout_pairs: int = 63, floor: float = 1.e-4) -> dict:
    """Refit only the factor law on a prefix; never refit source SANOS surfaces.

    Geometry is the already specified date-origin chart in both cases. This is
    a conditional-on-chart historical diagnostic, not a deployable rolling-date
    backtest. Production loadings and production calibration remain unchanged.
    """
    output.mkdir(parents=True, exist_ok=True)
    rolling = calibration_root / 'rolling' / origin
    with np.load(rolling / 'ar1.npz', allow_pickle=False) as saved:
        dates = list(map(str, saved['history_dates']))
        supports = np.asarray(saved['supports'])
        expiries = np.asarray(saved['model_expiries_y'])
    states, densities, spots, forwards = [], [], [], []
    for day in dates:
        with np.load(rolling / 'history' / day / 'fit.npz', allow_pickle=False) as saved:
            states.append(np.asarray(saved['dlvsq'], dtype=np.float64))
            densities.append(np.asarray(saved['densities'], dtype=np.float64))
            spots.append(float(saved['spot_ref']))
            forwards.append(np.asarray(saved['forward_over_spot'], dtype=np.float64))
    states, densities, spots, forwards = map(np.asarray, (states, densities, spots, forwards))
    if np.any(states <= 0.) or not np.isfinite(states).all():
        raise ValueError('historical fitted local variances must be positive and finite')
    logs = np.log(states)
    origins, endpoints, scores, start_indices, pair_dates, clipped_counts = [], [], [], [], [], []
    for endpoint in range(1, len(dates)):
        if count_business_days_between(date.fromisoformat(dates[endpoint - 1]), date.fromisoformat(dates[endpoint])) != 1:
            continue
        gross_spot = float(spots[endpoint] / spots[endpoint - 1])
        aligned, clipped = _fixed_contract_endpoint_surface(endpoint_log_dlvsq=logs[endpoint],
            supports=supports, reference_business_dtes=252. * expiries,
            origin_forward_over_spot=forwards[endpoint - 1, 1:],
            endpoint_forward_over_spot=forwards[endpoint], gross_spot_return=gross_spot)
        origins.append(logs[endpoint - 1, 1:])
        endpoints.append(aligned)
        scores.append(_spot_gaussian_score(normalized_spot=gross_spot / forwards[endpoint - 1, 0],
            support=supports[0], density=densities[endpoint - 1, 0], expiry_y=float(expiries[0]),
            backbone_annual_volatility=.05))
        start_indices.append(endpoint - 1)
        pair_dates.append(dates[endpoint])
        clipped_counts.append(clipped)
    origins, endpoints, scores = map(np.asarray, (origins, endpoints, scores))
    split = len(origins) - holdout_pairs
    if split <= 16 or holdout_pairs < 10:
        raise ValueError('insufficient independent training prefix or historical holdout')
    estimate = ar1_estimate(origins[:split], endpoints[:split])
    factor = joint_innovation_loadings(estimate['innovations'], scores[:split], factor_count=16)
    loadings = np.asarray(factor['q_log_dlvsq_loadings']).reshape(-1, 16)
    basis, _ = np.linalg.qr(loadings)
    training_projected = estimate['innovations'].reshape(split, -1) @ basis * np.sqrt(252.)
    scales = np.maximum(training_projected.std(axis=0, ddof=1), 1.e-12)
    score_mean = float(factor['historical_spot_score_mean'])
    score_sd = float(factor['historical_spot_score_sd'])
    residuals = endpoints[split:] - estimate['intercept'] - estimate['slope'] * origins[split:]
    projected = residuals.reshape(holdout_pairs, -1) @ basis * np.sqrt(252.) / scales
    spot_score = (scores[split:] - score_mean) / score_sd
    observed = np.column_stack((spot_score, projected))
    forecasts, losses = {name: [] for name in ('off', 'on')}, {name: [] for name in ('off', 'on')}
    affected_weights = []
    for index in range(holdout_pairs):
        mass = densities[start_indices[split + index], 1:, 1:-1]
        if np.any(mass < 0.) or not np.isfinite(mass).all():
            raise ValueError('historical attenuation requires finite nonnegative origin densities')
        denominator = mass + floor * np.max(mass, axis=-1, keepdims=True)
        weights = np.divide(mass, denominator, out=np.zeros_like(mass), where=denominator > 0.)
        affected_weights.append(weights)
        for name, weight in (('off', np.ones_like(weights)), ('on', weights)):
            projected_loading = (basis.T @ (loadings * weight.reshape(-1, 1))) / scales[:, None]
            joint_loading = np.vstack((np.eye(1, 16)[0], projected_loading))
            covariance = joint_loading @ joint_loading.T
            forecasts[name].append(covariance)
            losses[name].append(covariance_score(observed[index], covariance))
    realized_second = observed.T @ observed / holdout_pairs
    realized_correlation = np.corrcoef(observed, rowvar=False)[0, 1:]
    result = {'scope': 'auxiliary prefix-only historical factor-law evaluation, conditional on the frozen origin grid',
        'origin': origin, 'production_calibration_changed': False, 'production_loading_field_changed': False,
        'training_pairs': split, 'holdout_pairs': holdout_pairs,
        'training_last_endpoint': pair_dates[split - 1], 'holdout_first_endpoint': pair_dates[split],
        'holdout_last_endpoint': pair_dates[-1], 'density_is_start_of_pair': True,
        'holdout_endpoint_clipped_query_fraction': float(np.sum(clipped_counts[split:]) / residuals.size),
        'spot_definition': 'front-distribution Gaussianized daily spot score, standardized using training prefix only',
        'projection': '16 orthonormal directions spanning prefix-estimated joint loading field',
        'metric_definition': '(||C||_F^2 - 2 y^T C y) / dimension^2; lower is better',
        'heldout_residual_rms': float(np.sqrt(np.mean(residuals**2))),
        'heldout_residual_mean_rms': float(np.sqrt(np.mean(np.mean(residuals, axis=0)**2))),
        'holdout_spot_score_mean': float(spot_score.mean()), 'holdout_spot_score_second_moment': float(np.mean(spot_score**2)),
        'individual_node_instantaneous_correlation': 'unchanged by positive scalar attenuation; not claimed as an improvement',
        'variants': {}}
    for name in ('off', 'on'):
        average = np.mean(forecasts[name], axis=0)
        correlation = average[0, 1:] / np.sqrt(average[0, 0] * np.diag(average)[1:])
        result['variants'][name] = {'mean_covariance_score': float(np.mean(losses[name])),
            'projected_second_moment_rmse': float(np.sqrt(np.mean((average - realized_second)**2))),
            'spot_projected_cross_moment_rmse': float(np.sqrt(np.mean((average[0, 1:] - realized_second[0, 1:])**2))),
            'spot_projected_correlation_rmse': float(np.sqrt(np.mean((correlation - realized_correlation)**2)))}
    differences = np.asarray(losses['off']) - np.asarray(losses['on'])
    result['attenuation_covariance_score_improvement'] = block_mean_interval(differences)
    np.savez(output / 'historical-fidelity-arrays.npz', observed=observed,
        covariance_off=np.asarray(forecasts['off']), covariance_on=np.asarray(forecasts['on']),
        loss_off=losses['off'], loss_on=losses['on'], paired_score_difference=differences,
        basis=basis, prefix_scales=scales, prefix_loadings=loadings,
        attenuation=np.asarray(affected_weights), test_endpoint_dates=pair_dates[split:])
    write_json(output / 'historical-fidelity.json', result)
    print(json.dumps({'event': 'historical_attenuation_fidelity_complete', **result}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--origin', default='2025-08-29')
    parser.add_argument('--output-directory', type=Path, required=True)
    arguments = parser.parse_args()
    run_fidelity(calibration_root=arguments.calibration_root, origin=arguments.origin, output=arguments.output_directory)


if __name__ == '__main__':
    main()
