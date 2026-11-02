"""Compare realized relative DLV-squared volatility with predictable simulation input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from quantdata.gromm2.path_audit import summarize_statistics
from quantdata.gromm2.rolling_audit import moment_statistics
from quantdata.gromm2.rolling_calibration import write_json


def audit_dlv_variation(*, path: Path, output: Path) -> dict:
    """Use complete antithetic pairs as independent units across the whole path.

    At each live node I_d is the daily sum of xi sqrt(h) Z, and Q_d is the
    sum of h ||xi||^2 using the actual predictable attenuated loading. Compare
    E[sum_d I_d^2] with E[sum_d Q_d]. This concerns relative DLV-squared
    diffusion (equivalently log-DLV-squared diffusion), not DLV levels.
    Raw daily log-return volatility is also reported, separately from the
    drift-adjusted diffusion volatility. No path or node is censored for
    failing a numerical invariant.
    """
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'r') as saved:
        identity = json.loads(saved.attrs['identity'])
        config = identity['config']
        if saved.attrs['completed_paths'] != config['paths']:
            raise ValueError('realized-volatility inference needs the complete requested population')
        if 'dlv_variation_moments' not in saved:
            raise ValueError('population lacks passive diffusion-variation accumulators')
        if np.any(~saved['path_validity'][:]) or np.any(~saved['batch_diagnostic_validity'][:]):
            report = {'status': 'unavailable: numerically invalid complete population',
                      'paths_dropped': 0, 'source': str(path)}
            write_json(output / 'dlv-variation-report.json', report)
            return report
        moments = saved['dlv_variation_moments'][:]
        days = saved['dlv_model_expiry_days'][:]
    sums = moments[:, :5].sum(axis=0)
    identity_error = moments[:, 5].max(axis=0)
    pairs = config['paths'] // 2
    active_days = np.minimum(np.maximum(np.r_[0, days[:-1]], 0), config['days'])
    exposure = active_days[:, None] / 252.
    live = np.broadcast_to(exposure > 0., sums[0].shape)
    safe_exposure = np.where(live, exposure, 1.)
    realized_variance = sums[0] / pairs / safe_exposure
    input_variance = sums[1] / pairs / safe_exposure
    mean_difference = realized_variance - input_variance
    difference_squares = sums[4] / safe_exposure**2 - pairs * mean_difference**2
    if not np.isfinite(moments).all():
        raise FloatingPointError('non-finite passive volatility diagnostics; no valid-only analysis')
    statistics = moment_statistics(pairs, mean_difference, difference_squares, epsilon=1.e-8)
    informative = live & (statistics['standard_error'] > 0.)
    deterministic = live & ~informative
    raw_mean = sums[2] / pairs / np.maximum(active_days[:, None], 1)
    raw_variance = (sums[3] / pairs / np.maximum(active_days[:, None], 1) - raw_mean**2) * 252.
    ratio = np.divide(realized_variance, input_variance, out=np.full_like(input_variance, np.nan), where=input_variance > 0.)
    positive_input = live & (input_variance > 0.)
    screen = summarize_statistics(statistics, informative) if np.any(informative) else {
        'status': 'degenerate: no nonzero sampling variance', 'tested_coordinates': 0}
    if 'maximum_absolute_mean_gain_over_spot' in screen:
        screen['maximum_absolute_annualized_variance_difference'] = screen.pop('maximum_absolute_mean_gain_over_spot')
    report = {'status': 'complete', 'source': str(path), 'paths': config['paths'],
        'independent_antithetic_pairs': pairs, 'substeps': config['substeps'],
        'volatility_definition': 'annualized relative DLV-squared/log-DLV-squared diffusion',
        'estimator': 'sqrt(mean path sum of daily drift-adjusted log innovations squared / live exposure)',
        'input': 'sqrt(mean path integral of actual state-dependent attenuated loading norm squared / live exposure)',
        'uncertainty': 'one independent observation per complete antithetic pair, after summing its entire live path',
        'realized_over_input_variance_quantiles': np.quantile(ratio[positive_input], [.01, .1, .5, .9, .99]).tolist() if np.any(positive_input) else None,
        'zero_input_variance_coordinates': int(np.sum(live & ~positive_input)),
        'zero_sampling_error_coordinates': int(np.sum(deterministic)),
        'nonzero_variance_difference_with_zero_sampling_error': int(np.sum(deterministic & (mean_difference != 0.))),
        'maximum_log_update_identity_error': float(np.max(identity_error[live])),
        'variance_difference_screen': screen,
        'paths_dropped': 0, 'raw_log_return_volatility_includes_drift': True}
    np.savez(output / 'dlv-variation-statistics.npz', expiry_days=days, active_business_days=active_days,
        realized_diffusion_vol=np.sqrt(np.maximum(realized_variance, 0.)),
        input_diffusion_vol=np.sqrt(np.maximum(input_variance, 0.)),
        raw_daily_log_return_vol=np.sqrt(np.maximum(raw_variance, 0.)),
        realized_over_input_variance_ratio=ratio, live=live,
        **{('annualized_variance_difference' if name == 'mean_gain_over_spot' else name): value
           for name, value in statistics.items()})
    write_json(output / 'dlv-variation-report.json', report)
    print(json.dumps({'event': 'dlv_variation_audit_complete', **report}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    arguments = parser.parse_args()
    audit_dlv_variation(path=arguments.paths, output=arguments.output_directory)


if __name__ == '__main__':
    main()
