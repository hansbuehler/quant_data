"""Restore original training folds for independent strict-gradient receipts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    fold_path_ranges,
)
from quantdata.gromm2.local_audit_training import (
    LocalAuditTrainConfig,
    _path_identity,
    _resident_fold,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.strict_action_bundle import (
    _declaration,
    _native_source,
    _original_normalization,
    verify_declared_strict_action,
)
from quantdata.torch import BaseEnvironment


def verify_strict_population_action(*, declaration: Path, declaration_sha256: str,
                                    fold: int, diagnostic: str, device: str,
                                    upload_paths: int = 512) -> dict:
    """Restore one original fold and compute exactly one canonical gradient.

    Only the declared training fold's feature/return arrays are loaded. The
    saved inner-training rows, F32 normalization and fixed optimizer metric
    are restored; no normalization, optimization or validation selection is
    performed. Existing receipts and every native artifact remain untouched.
    Operational upload blocks do not change the canonical working chunk.
    """
    verify_inp(type(fold) is int and fold in (0, 1) and diagnostic in DIAGNOSTICS,
        lambda: 'one original fold/objective endpoint is required')
    verify_inp(device in ('cpu', 'cuda') and type(upload_paths) is int and upload_paths > 0,
        lambda: 'an explicit compute device and positive upload block are required')
    if device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('strict population verification requires a managed GPU allocation')
    declared, _ = _declaration(declaration, declaration_sha256)
    entry = next(item for item in declared['sources'] if (item['fold'], item['diagnostic']) == (fold, diagnostic))
    plan, native, active = _native_source(entry)
    normalization = _original_normalization(plan=plan, native=native, active=active,
        source=declared['source_population_identity'])
    if Path(entry['receipt']).exists():
        raise ValueError('preserve the existing immutable strict verification receipt')
    check_resource_headroom()
    started = checked = time.monotonic()
    stop, done = threading.Event(), threading.Event()
    progress = {'stage': 'checking_original_training_population', 'fold': fold, 'diagnostic': diagnostic}

    def emit(event: str) -> None:
        print(json.dumps({'event': event, **progress, 'elapsed_seconds': time.monotonic()-started,
            'remaining_seconds': None}), flush=True)

    def observe(values: dict) -> None:
        nonlocal checked
        if stop.is_set():
            raise InterruptedError('strict verification stopped; no optimizer update or partial receipt is published')
        progress.clear()
        progress.update(fold=fold, diagnostic=diagnostic, **values)
        if time.monotonic()-checked >= 60.:
            emit('strict_population_verification_progress')
            check_resource_headroom()
            checked = time.monotonic()

    def heartbeat() -> None:
        while not done.wait(60.):
            emit('strict_population_verification_heartbeat')
            try:
                check_resource_headroom()
            except RuntimeError:
                stop.set()

    handlers = {number: signal.signal(number, lambda *_: stop.set())
        for number in (signal.SIGINT, signal.SIGTERM)} if threading.current_thread() is threading.main_thread() else {}
    threading.Thread(target=heartbeat, daemon=True).start()
    devices = [] if device == 'cpu' else [torch.device('cuda')]
    try:
        with torch.random.fork_rng(devices=devices):
            source_identity, geometry = _path_identity(Path(plan['population']))
            if (json.loads(source_identity) != declared['source_population_identity']
                or hashlib.sha256(source_identity.encode()).hexdigest()
                    != native['identity']['source_population_identity_sha256']):
                raise ValueError('the declared original training population identity changed')
            fields = dict(native['state']['identity']['model_config'])
            fields['network'] = ActionNetworkConfig(**fields['network'])
            config = LocalActionConfig(**fields)
            if (config.feature_count != geometry['features'] or config.instrument_count != geometry['instruments']
                or geometry['paths'] != declared['source_population_identity']['config']['paths']
                or geometry['days'] != declared['source_population_identity']['config']['days']):
                raise ValueError('the original population geometry differs from the native policy')
            with np.load(io.BytesIO(normalization), allow_pickle=False) as saved:
                values = {name: np.asarray(saved[name]) for name in saved.files if name != 'identity'}
            outer = fold_path_ranges(geometry['paths'])[fold]
            states, returns, eligible = _resident_fold(Path(plan['population']), outer=outer, normalization=values,
                env=BaseEnvironment(device=device, precision=32),
                train_config=LocalAuditTrainConfig(device=device, upload_paths=upload_paths, host_cache=False),
                progress=progress, stop_requested=stop.is_set)
            rows = torch.arange(len(states), device=states.device).reshape(-1, geometry['days'])
            problem = AffineAuditProblem(states=states, returns=returns, eligible=eligible,
                rows=rows[torch.as_tensor(values['inner_training_mask'], dtype=torch.bool, device=states.device)].flatten(),
                action_scale=config.action_scale, squared_lambda=config.squared_lambda,
                config=AffineSolveConfig(batch_size=active['identity']['working_memory_batch_size']), progress=observe)
            if (len(problem.rows) != active['identity']['training_rows']
                or hashlib.sha256(problem.rows.cpu().numpy().tobytes()).hexdigest()
                    != active['identity']['training_rows_sha256']
                or hashlib.sha256(eligible.cpu().numpy().tobytes()).hexdigest()
                    != active['identity']['eligibility_sha256']):
                raise ValueError('the original complete-path training rows or eligibility changed')
            policy = AffineDeepAction(BaseEnvironment(device=device, precision=64), config=config)
            policy.load_state_dict(native['model'], strict=True)
            policy.affine_only = False
            problem.preconditioner_factors = native['optimizer']['metric_factors'].to(states.device)
            metric = AffineBlockMetric(policy, problem, deep_scale=native['optimizer']['metric_identity']['deep_scale'])
            if metric.identity != native['optimizer']['metric_identity']:
                raise ValueError('the original fixed optimizer metric changed')
            optimizer = MetricLBFGS(metric, history_size=native['optimizer']['history_size'],
                curvature_floor=native['optimizer']['curvature_floor'])
            optimizer.load_state_dict({**native['optimizer'], 'metric_factors': problem.preconditioner_factors,
                'history': [tuple(value.to(states.device) for value in retained) for retained in native['optimizer']['history']]})
            observe({'stage': 'recomputing_one_original_canonical_gradient', 'completed_observations': 0,
                'total_observations': len(problem.rows), 'outer_training_paths': [outer.start, outer.stop]})
            emit('strict_population_verification_started')
            receipt = verify_declared_strict_action(declaration=declaration, declaration_sha256=declaration_sha256,
                fold=fold, diagnostic=diagnostic, policy=policy, problem=problem, optimizer=optimizer)
            progress.clear()
            progress.update(stage='strict_verification_receipt_complete', fold=fold, diagnostic=diagnostic,
                completed_observations=len(problem.rows), total_observations=len(problem.rows),
                completed_full_objective_passes=1, completed_curvature_passes=0, optimizer_updates=0)
            emit('strict_population_verification_complete')
            return receipt
    finally:
        done.set()
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1), required=True)
    parser.add_argument('--diagnostic', choices=DIAGNOSTICS, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--upload-paths', type=int, default=512)
    verify_strict_population_action(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
