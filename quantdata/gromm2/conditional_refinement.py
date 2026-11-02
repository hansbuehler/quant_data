"""Select local-action batch/LR trials without evaluating outer holdout paths."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import torch

from quantdata.gromm2.audit_host_cache import cache_audit_arrays
from quantdata.gromm2.conditional_audit import DIAGNOSTICS, fold_path_ranges
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _path_identity,
    train_fold,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json


def trial_settings(*, base: LocalAuditTrainConfig, batch_sizes: tuple[int, ...],
                   learning_rates: tuple[float, ...]) -> list[LocalAuditTrainConfig]:
    """Declare the entire candidate set before inspecting training outcomes."""
    if not batch_sizes or not learning_rates or len(set(batch_sizes)) != len(batch_sizes) or len(set(learning_rates)) != len(learning_rates):
        raise ValueError('candidate controls must be nonempty and unique')
    return [replace(base, batch_size=batch, learning_rate=rate)
        for batch, rate in itertools.product(batch_sizes, learning_rates)]


def select_trials(reports: list[dict]) -> dict[str, dict]:
    """Choose each objective from its inner-validation loss, never an outer score."""
    if not reports or any(report['stop_reason'] != 'internal_validation_plateau' for report in reports):
        raise RuntimeError('all declared trials must finish their training-only stopping rule')
    selected = {}
    for diagnostic in DIAGNOSTICS:
        report = min(reports, key=lambda candidate: candidate['best_internal_losses'][diagnostic])
        checkpoint = Path(report['checkpoint'])
        selected[diagnostic] = {'checkpoint': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'selected_epoch': report['best_epochs'][diagnostic],
            'inner_validation_loss': report['best_internal_losses'][diagnostic],
            'zero_baseline_selected': report['best_epochs'][diagnostic] == 0,
            'training_report': str(checkpoint.parent / 'training_report.json'),
            'normalization': str(checkpoint.parent.parent / 'normalization.npz')}
    return selected


def fit_trials(*, path: Path, output: Path, candidates: list[LocalAuditTrainConfig],
               protocol: LocalAuditProtocol, fresh_evaluation_seed: int | None = None) -> dict:
    """Fit on each complete-path fold and freeze selection before fresh evaluation.

    This function does not simulate paths or compute any outer-holdout statistic.
    A cache covers only the current outer training fold, not the other fold.
    Source-population methodology is retained verbatim in the frozen manifest;
    calling this function cannot turn a modified scheme into the paper model.
    """
    identity, geometry = _path_identity(path)
    parsed = json.loads(identity)
    if fresh_evaluation_seed is not None and parsed.get('config', {}).get('seed') == fresh_evaluation_seed:
        raise ValueError('fresh evaluation requires an independent simulation seed')
    check_resource_headroom()
    output.mkdir(parents=True, exist_ok=True)
    plan = {'source_population': str(path.resolve()), 'source_identity': parsed,
        'source_identity_sha256': hashlib.sha256(identity.encode()).hexdigest(),
        'protocol': asdict(protocol), 'protocol_version': LocalAuditProtocol.version.unique_id48,
        'candidate_controls': [asdict(candidate) for candidate in candidates],
        'fresh_evaluation_seed': fresh_evaluation_seed,
        'evaluation_design': 'same-population two-fold crossfit' if fresh_evaluation_seed is None else 'separate fresh evaluation',
        'selection_uses_only_inner_validation': True, 'outer_holdout_evaluated': False}
    plan_path = output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('this run directory belongs to a different trial plan')
    write_json(plan_path, plan)
    frozen = {**plan, 'folds': []}
    for fold, rows in enumerate(fold_path_ranges(geometry['paths'])):
        cache = None
        try:
            if any(candidate.host_cache for candidate in candidates):
                with h5py.File(path, 'r') as stream:
                    cache = cache_audit_arrays(stream, rows=rows)
            reports = []
            for index, candidate in enumerate(candidates):
                directory = output / f'fold-{fold}' / f'trial-{index}'
                reports.append(train_fold(path=path, output=directory, fold=fold,
                    protocol=protocol, train_config=candidate, cache=cache))
                if candidate.device == 'cuda':
                    torch.cuda.empty_cache()
            frozen['folds'].append({'fold': fold, 'selected': select_trials(reports),
                'reports': [report['checkpoint'] for report in reports]})
            write_json(output / 'training-progress.json', frozen)
        finally:
            if cache is not None:
                cache.release()
    frozen['status'] = 'policies_frozen; inference not yet performed'
    write_json(output / 'frozen-policies.json', frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1024, 4096])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[1.e-6, 1.e-5])
    parser.add_argument('--minimum-learning-rate', type=float, default=1.e-8)
    parser.add_argument('--fresh-evaluation-seed', type=int)
    parser.add_argument('--host-cache', action='store_true')
    arguments = parser.parse_args()
    base = LocalAuditTrainConfig(device=arguments.device,
        learning_rate=max(arguments.learning_rates), minimum_learning_rate=arguments.minimum_learning_rate,
        host_cache=arguments.host_cache)
    candidates = trial_settings(base=base, batch_sizes=tuple(arguments.batch_sizes),
        learning_rates=tuple(arguments.learning_rates))
    fit_trials(path=arguments.paths, output=arguments.output_directory, candidates=candidates,
        protocol=LocalAuditProtocol(), fresh_evaluation_seed=arguments.fresh_evaluation_seed)


if __name__ == '__main__':
    main()
