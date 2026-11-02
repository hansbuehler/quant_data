"""Cross-fitted local action tests on retained, unchanged-Q GROMM2 paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy.stats import norm

from quantdata.gromm2.audit_host_cache import AuditHostCache, cache_audit_arrays, read_audit_block
from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    ActionNetworkConfig,
    IndependentLocalActions,
    LocalActionConfig,
    LocalActionStep,
    fold_path_ranges,
    frozen_score_statistics,
    local_objective,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.torch import BaseEnvironment

TRAINING_CONTROL_FIELDS = ('batch_size', 'affine_epochs', 'minimum_full_epochs', 'maximum_full_epochs',
    'learning_rate', 'minimum_learning_rate', 'learning_rate_patience', 'convergence_patience',
    'absolute_improvement', 'relative_improvement', 'gradient_clip')


@version('0.2', dependencies=[ActionNetworkConfig, LocalActionConfig])
@dataclass(frozen=True, kw_only=True)
class LocalAuditProtocol:
    """Training-only selection, with the opposite outer fold untouched."""

    internal_validation_fraction: float = .1
    split_seed: int = 20261031
    normalization_floor: float = 1.e-4
    minimum_action_scale: float = 1.e-6
    squared_lambda: float = 1.
    network: ActionNetworkConfig = ActionNetworkConfig()

    def __post_init__(self) -> None:
        verify_inp(0. < self.internal_validation_fraction < .5, lambda: 'invalid internal validation fraction')
        verify_inp(self.normalization_floor > 0. and self.minimum_action_scale > 0., lambda: 'normalization floors must be positive')
        verify_inp(math.isfinite(self.squared_lambda) and self.squared_lambda >= 0., lambda: 'invalid squared lambda')


@dataclass(frozen=True, kw_only=True)
class LocalAuditTrainConfig:
    """Operational controls; a changed schedule preserves the same checkpoint."""

    device: str = 'cuda'
    precision: int = 32
    seed: int = 20261031
    batch_size: int = 16384
    evaluation_batch_size: int = 32768
    upload_paths: int = 512
    affine_epochs: int = 100
    minimum_full_epochs: int = 200
    maximum_full_epochs: int = 2000
    learning_rate: float = 2.5e-4
    minimum_learning_rate: float = 1.e-6
    learning_rate_patience: int = 50
    convergence_patience: int = 100
    absolute_improvement: float = 1.e-9
    relative_improvement: float = 1.e-4
    gradient_clip: float = 5.
    cpu_threads: int = 2
    compile: bool = True
    use_cuda_graphs: bool = True
    heartbeat_seconds: float = 60.
    host_cache: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.device in {'cpu', 'cuda'} and self.precision in {32, 64}, lambda: 'invalid training device or precision')
        verify_inp(min(self.batch_size, self.evaluation_batch_size, self.upload_paths, self.cpu_threads) > 0, lambda: 'invalid batch or worker size')
        verify_inp(self.affine_epochs >= 0 and 1 <= self.minimum_full_epochs <= self.maximum_full_epochs, lambda: 'invalid phase lengths')
        verify_inp(0. < self.minimum_learning_rate <= self.learning_rate, lambda: 'invalid learning rates')
        verify_inp(min(self.learning_rate_patience, self.convergence_patience, self.gradient_clip) > 0, lambda: 'invalid optimizer controls')
        verify_inp(0. < self.heartbeat_seconds <= 60., lambda: 'progress must be emitted at least every minute')
        verify_inp(isinstance(self.host_cache, bool), lambda: 'host_cache must be boolean')


def internal_training_mask(path_count: int, *, protocol: LocalAuditProtocol, fold: int) -> np.ndarray:
    """Reserve training-fold pairs for checkpoint selection, never the outer holdout."""
    verify_inp(path_count >= 8 and path_count % 2 == 0, lambda: 'each training fold requires at least four antithetic pairs')
    pairs = path_count // 2
    validation_pairs = min(pairs - 2, max(2, round(protocol.internal_validation_fraction * pairs)))
    generator = np.random.default_rng(np.random.SeedSequence([protocol.split_seed, fold]))
    selected = generator.permutation(pairs)[:validation_pairs]
    mask = np.ones(pairs, dtype=bool)
    mask[selected] = False
    return np.repeat(mask, 2)


def effective_batch_size(observations: int, preferred: int) -> int:
    """Avoid repeated partial-batch graph captures when a nearby divisor exists."""
    preferred = min(observations, preferred)
    for candidate in range(preferred, max(1, preferred // 2) - 1, -1):
        if observations % candidate == 0:
            return candidate
    return preferred


def _path_identity(path: Path) -> tuple[str, dict[str, int]]:
    with h5py.File(path, 'r') as stream:
        identity = str(stream.attrs['identity'])
        geometry = {'paths': int(stream['features'].shape[0]), 'days': int(stream['features'].shape[1]),
                    'features': int(stream['features'].shape[2]), 'instruments': int(stream['returns'].shape[2])}
        if stream.attrs.get('completed_paths') != geometry['paths'] or geometry['paths'] < 16:
            raise ValueError('the conditional test requires a completed path population with at least eight pairs')
        if stream['returns'].shape[:2] != stream['features'].shape[:2]:
            raise ValueError('state and next-day return geometries disagree')
        for name in ('path_validity', 'batch_diagnostic_validity'):
            if name in stream and not np.all(stream[name][:]):
                raise ValueError('conditional inference is not defined on a numerically invalid complete population; no paths may be discarded')
    return identity, geometry


def _normalization(path: Path, *, outer: slice, mask: np.ndarray, protocol: LocalAuditProtocol,
                   output: Path, identity: str, upload_paths: int,
                   cache: AuditHostCache | None = None) -> dict[str, np.ndarray]:
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            if str(saved['identity'].item()) != identity:
                raise ValueError('saved training-only normalization has a different identity')
            return {name: np.asarray(saved[name]) for name in saved.files if name != 'identity'}
    count = 0
    mean = squares = None
    return_squares = 0.
    return_count = 0
    started = checked = time.monotonic()
    with h5py.File(path, 'r') as stream:
        for offset in range(0, outer.stop - outer.start, upload_paths):
            stop = min(offset + upload_paths, len(mask))
            selected = mask[offset:stop]
            if not np.any(selected):
                continue
            rows = slice(outer.start + offset, outer.start + stop)
            features = np.asarray(read_audit_block(stream, 'features', rows, cache=cache)[selected], dtype=np.float64)
            returns = np.asarray(read_audit_block(stream, 'returns', rows, cache=cache)[selected], dtype=np.float64)
            if np.any(~np.isfinite(features)) or np.any(~np.isfinite(returns)):
                raise FloatingPointError('non-finite training data; no observations are dropped')
            flat = features.reshape(-1, features.shape[-1])
            batch_mean = flat.mean(axis=0)
            batch_squares = np.sum((flat - batch_mean)**2, axis=0)
            if mean is None:
                mean, squares = np.zeros_like(batch_mean), np.zeros_like(batch_squares)
            difference = batch_mean - mean
            total = count + len(flat)
            squares += batch_squares + difference**2 * count * len(flat) / total
            mean += difference * len(flat) / total
            count = total
            return_squares += float(np.sum(returns**2))
            return_count += returns.size
            if time.monotonic() - checked >= 60.:
                if cache is not None:
                    cache.check_headroom()
                check_resource_headroom()
                print(json.dumps({'event': 'local_action_normalization_progress',
                    'completed_paths': stop, 'total_paths': len(mask),
                    'elapsed_seconds': time.monotonic() - started}), flush=True)
                checked = time.monotonic()
    values = {'mean': mean, 'scale': np.maximum(np.sqrt(squares / count), protocol.normalization_floor),
              'action_scale': np.asarray(max(np.sqrt(return_squares / return_count), protocol.minimum_action_scale)),
              'inner_training_mask': mask, 'training_observations': np.asarray(count)}
    _atomic_npz(output, {**values, 'identity': np.asarray(identity)})
    return values


def _cpu_tensor_block(values: np.ndarray, *, pin_memory: bool) -> torch.Tensor:
    """Stage read-only raw arrays without unsupported writable NumPy aliases."""
    if values.flags.writeable:
        result = torch.from_numpy(values)
        return result.pin_memory() if pin_memory else result
    dtype = torch.from_numpy(np.empty(0, dtype=values.dtype)).dtype
    result = torch.empty(values.shape, dtype=dtype, device='cpu', pin_memory=pin_memory)
    np.copyto(result.numpy(), values, casting='no')
    return result


def _resident_fold(path: Path, *, outer: slice, normalization: dict[str, np.ndarray],
                   env: BaseEnvironment, train_config: LocalAuditTrainConfig, progress: dict,
                   cache: AuditHostCache | None = None,
                   stop_requested: Callable[[], bool] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if stop_requested is not None and stop_requested():
        raise InterruptedError('stopped before fold allocation; no optimizer update committed')
    with h5py.File(path, 'r') as stream:
        paths, days, features = outer.stop - outer.start, stream['features'].shape[1], stream['features'].shape[2]
        instruments = stream['returns'].shape[-1]
        required_bytes = paths * days * (features + instruments) * torch.empty((), dtype=env.fdtype).element_size()
        if env.device.type == 'cuda':
            available, _ = torch.cuda.mem_get_info(env.device)
            if required_bytes > .85 * available:
                raise RuntimeError(f'training fold requires {required_bytes} VRAM bytes plus working space; only {available} are free')
        states = torch.empty((paths, days, features), device=env.device, dtype=env.fdtype)
        returns = torch.empty((paths, days, instruments), device=env.device, dtype=env.fdtype)
        mean = torch.as_tensor(normalization['mean'], dtype=env.fdtype, device=env.device)
        scale = torch.as_tensor(normalization['scale'], dtype=env.fdtype, device=env.device)
        for offset in range(0, paths, train_config.upload_paths):
            if stop_requested is not None and stop_requested():
                raise InterruptedError('stopped during fold upload; no optimizer update committed')
            stop = min(offset + train_config.upload_paths, paths)
            rows = slice(outer.start + offset, outer.start + stop)
            state_values = read_audit_block(stream, 'features', rows, cache=cache)
            return_values = read_audit_block(stream, 'returns', rows, cache=cache)
            raw_states = _cpu_tensor_block(state_values, pin_memory=env.device.type == 'cuda')
            raw_returns = _cpu_tensor_block(return_values, pin_memory=env.device.type == 'cuda')
            if not torch.isfinite(raw_states).all() or not torch.isfinite(raw_returns).all():
                raise FloatingPointError('non-finite retained fold data')
            states[offset:stop].copy_(raw_states, non_blocking=False)
            states[offset:stop].sub_(mean).div_(scale)
            returns[offset:stop].copy_(raw_returns, non_blocking=False)
            returns[offset:stop].div_(float(normalization['action_scale']))
            progress.update(stage='uploading_training_fold', completed_paths=stop, total_paths=paths)
        option_expiries = np.repeat(np.asarray(stream['expiry_days']), stream['strikes'].shape[-1])
        eligible = np.column_stack((np.ones(days, dtype=bool), option_expiries[None] > np.arange(days)[:, None]))
    return states.reshape(-1, features), returns.reshape(-1, instruments), torch.as_tensor(eligible, device=env.device, dtype=torch.bool)


def _population_loss(model: IndependentLocalActions, states: torch.Tensor, returns: torch.Tensor,
                     eligible: torch.Tensor, rows: torch.Tensor, *, batch_size: int, forwards=None) -> dict[str, float]:
    model.eval()
    sums = torch.zeros(len(DIAGNOSTICS), device=states.device, dtype=torch.float64)
    forwards = model.policies if forwards is None else forwards
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            indices = rows[start:start + batch_size]
            observed, gains = states[indices], returns[indices]
            mask = eligible[indices % len(eligible)]
            sums += torch.stack([local_objective(forwards[name](observed, mask), gains,
                diagnostic=name, action_scale=model.config.action_scale, squared_lambda=model.config.squared_lambda)
                for name in DIAGNOSTICS]).to(torch.float64) * len(indices)
    result = dict(zip(DIAGNOSTICS, (sums / len(rows)).cpu().tolist(), strict=True))
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError('non-finite full training-fold loss')
    return result


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_checkpoint(path: Path, **values) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(values, temporary)
    temporary.replace(path)


def _restore_controls(*, optimizers, schedulers, state: dict, previous: dict,
                      requested: LocalAuditTrainConfig) -> bool:
    """Apply explicitly changed controls without resetting Adam moments or RNG."""
    current = asdict(requested)
    changed = {name for name in TRAINING_CONTROL_FIELDS if previous.get(name) != current[name]}
    if not changed:
        return False
    trained_deep = state['epoch'] > previous['affine_epochs'] or (
        state['epoch'] == previous['affine_epochs'] and state.get('next_batch', 0) > 0)
    if trained_deep and requested.affine_epochs > state['epoch']:
        raise ValueError('cannot resume a trained deep phase as affine-only; preserve its phase boundary')
    for name in DIAGNOSTICS:
        optimizer, scheduler = optimizers[name], schedulers[name]
        if 'learning_rate' in changed:
            for group in optimizer.param_groups:
                group['lr'] = requested.learning_rate
        if 'minimum_learning_rate' in changed:
            for group in optimizer.param_groups:
                group['lr'] = max(float(group['lr']), requested.minimum_learning_rate)
        scheduler.patience = requested.learning_rate_patience
        scheduler.min_lrs = [requested.minimum_learning_rate] * len(optimizer.param_groups)
        scheduler.threshold_mode = 'abs'
        scheduler.threshold = requested.absolute_improvement
        scheduler.best = state['best_loss'][name]
        scheduler.num_bad_epochs = 0
        scheduler._last_lr = [float(group['lr']) for group in optimizer.param_groups]
        state['stopped'][name] = False
        state['last_improvement'][name] = state['epoch']
        state['floor_since'][name] = None
    return True


def train_fold(*, path: Path, output: Path, fold: int, protocol: LocalAuditProtocol,
               train_config: LocalAuditTrainConfig, load_only: bool = False,
               cache: AuditHostCache | None = None) -> dict[str, object]:
    """Fit on one outer fold, using only its internal pairs for model selection."""
    identity, geometry = _path_identity(path)
    outer = fold_path_ranges(geometry['paths'])[fold]
    protocol_identity = json.dumps({'path_identity': identity, 'protocol': asdict(protocol), 'fold': fold,
        'protocol_version': LocalAuditProtocol.version.unique_id48}, sort_keys=True)
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('local_audit_training.py', 'conditional_audit.py', 'market_training_runtime.py', 'audit_host_cache.py')}
    output.mkdir(parents=True, exist_ok=True)
    mask = internal_training_mask(outer.stop - outer.start, protocol=protocol, fold=fold)
    if load_only and not (output / 'normalization.npz').is_file():
        raise FileNotFoundError(output / 'normalization.npz')
    with nullcontext() if load_only else FileLock(str(output / 'train.lock'), timeout=0):
        check_resource_headroom()
        normalization_started = time.monotonic()
        if not load_only:
            print(json.dumps({'event': 'local_action_stage', 'fold': fold, 'stage': 'training-only-normalization',
                'completed_paths': 0, 'total_paths': len(mask)}), flush=True)
        normalization = _normalization(path, outer=outer, mask=mask, protocol=protocol,
            output=output / 'normalization.npz', identity=protocol_identity, upload_paths=train_config.upload_paths, cache=cache)
        if not load_only:
            print(json.dumps({'event': 'local_action_stage', 'fold': fold, 'stage': 'training-only-normalization-complete',
                'completed_paths': len(mask), 'total_paths': len(mask),
                'elapsed_seconds': time.monotonic() - normalization_started}), flush=True)
        config = LocalActionConfig(label=f'fold-{fold}', feature_count=geometry['features'], instrument_count=geometry['instruments'],
            action_scale=float(normalization['action_scale']), network=protocol.network, squared_lambda=protocol.squared_lambda)
        directory = output / config.filename
        directory.mkdir(exist_ok=True)
        checkpoint_path = directory / 'checkpoint.pt'
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
        if saved is not None and (saved['code_version'] != LocalActionConfig.version.unique_id48 or saved['user_version'] != config.user_version):
            raise RuntimeError('Action checkpoint version changed; preserve this run and use a new directory for a cold start')
        if saved is not None and saved['identity'] != protocol_identity:
            raise ValueError('saved action model belongs to different paths or folds')
        if load_only:
            if saved is None:
                raise FileNotFoundError(checkpoint_path)
            return saved['report']
        if saved is not None and saved['report']['stop_reason'] == 'internal_validation_plateau':
            if (all(saved['report']['train_config'].get(name) == asdict(train_config)[name] for name in TRAINING_CONTROL_FIELDS)
                and saved['state']['epoch'] >= train_config.affine_epochs + train_config.minimum_full_epochs):
                return saved['report']
            saved['state']['stopped'] = {name: False for name in DIAGNOSTICS}
        if train_config.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
            raise RuntimeError('GPU conditional fitting requires an explicit gpuctl allocation')
        torch.set_num_threads(train_config.cpu_threads)
        torch.random.default_generator.manual_seed(train_config.seed + fold)
        env = BaseEnvironment(device=train_config.device, precision=train_config.precision)
        model = IndependentLocalActions(env, config=config)
        optimizers = {name: torch.optim.Adam(policy.parameters(), lr=train_config.learning_rate) for name, policy in model.policies.items()}
        schedulers = {name: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[name], factor=.5,
            patience=train_config.learning_rate_patience, min_lr=train_config.minimum_learning_rate,
            threshold_mode='abs', threshold=train_config.absolute_improvement, eps=0.) for name in DIAGNOSTICS}
        for scheduler in schedulers.values():
            scheduler.best = 0.
        generator = torch.Generator(device=env.device).manual_seed(train_config.seed + fold)
        state = {'epoch': 0, 'next_batch': 0, 'order': None, 'history': [], 'stopped': {name: False for name in DIAGNOSTICS},
                 'best_loss': {name: 0. for name in DIAGNOSTICS}, 'best_epoch': {name: 0 for name in DIAGNOSTICS},
                 'best_state': {name: _cpu_state(policy) for name, policy in model.policies.items()},
                 'last_improvement': {name: 0 for name in DIAGNOSTICS},
                 'floor_since': {name: None for name in DIAGNOSTICS}}
        if saved is not None:
            model.load_state_dict(saved['model'])
            for name in DIAGNOSTICS:
                optimizers[name].load_state_dict(saved['optimizers'][name])
                schedulers[name].load_state_dict(saved['schedulers'][name])
            state = saved['state']
            _restore_controls(optimizers=optimizers, schedulers=schedulers, state=state,
                previous=saved['report']['train_config'], requested=train_config)
            generator.set_state(saved['generator_rng'].cpu())
            torch.set_rng_state(saved['torch_rng'].cpu())
            if env.device.type == 'cuda' and saved['cuda_rng'] is not None:
                torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), env.device)
        progress = {'fold': fold, 'epoch': state['epoch'], 'stage': 'initializing'}
        stop, done = threading.Event(), threading.Event()
        started = time.monotonic()

        def heartbeat():
            while not done.wait(train_config.heartbeat_seconds):
                print(json.dumps({'event': 'local_action_progress', **progress, 'elapsed_seconds': time.monotonic() - started}), flush=True)
                try:
                    if cache is not None:
                        cache.check_headroom()
                    check_resource_headroom()
                except RuntimeError:
                    if cache is not None:
                        cache.release()
                    stop.set()

        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        threading.Thread(target=heartbeat, daemon=True).start()

        def save(reason):
            report = {'fold': fold, 'epoch': state['epoch'], 'stop_reason': reason, 'best_epochs': state['best_epoch'],
                'best_internal_losses': state['best_loss'], 'model_config': asdict(config), 'train_config': asdict(train_config),
                'implementation_sha256': implementation,
                'path_identity_sha256': hashlib.sha256(identity.encode()).hexdigest(),
                'outer_training_path_range': [outer.start, outer.stop], 'outer_holdout_used_for_selection': False,
                'checkpoint': str(checkpoint_path), 'history': state['history']}
            report['effective_optimizer_controls'] = {name: {
                'learning_rates': [float(group['lr']) for group in optimizers[name].param_groups],
                'minimum_learning_rates': schedulers[name].min_lrs,
                'scheduler_patience': schedulers[name].patience,
                'scheduler_best': schedulers[name].best} for name in DIAGNOSTICS}
            report['cuda_timing_note'] = 'CUDA stream intervals include host launch gaps; they are not kernel busy time or utilization.'
            for name in DIAGNOSTICS:
                counters = [values['step'].reshape(()) for values in optimizers[name].state.values() if 'step' in values]
                bounds = torch.stack(counters) if counters else torch.zeros(1, device=env.device)
                report['effective_optimizer_controls'][name]['optimizer_step_min_max'] = torch.stack((bounds.min(), bounds.max())).cpu().tolist()
            _save_checkpoint(checkpoint_path, code_version=LocalActionConfig.version.unique_id48, user_version=config.user_version,
                identity=protocol_identity, model=model.state_dict(), optimizers={name: value.state_dict() for name, value in optimizers.items()},
                schedulers={name: value.state_dict() for name, value in schedulers.items()}, state=state, report=report,
                generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(env.device) if env.device.type == 'cuda' else None)
            write_json(directory / 'training_report.json', report)
            return report

        try:
            states, returns, eligible = _resident_fold(path, outer=outer, normalization=normalization, env=env,
                train_config=train_config, progress=progress, cache=cache, stop_requested=stop.is_set)
            days = geometry['days']
            inner_mask = torch.as_tensor(normalization['inner_training_mask'], device=env.device, dtype=torch.bool)
            inner_train = torch.arange(len(states), device=env.device, dtype=env.idtype).reshape(-1, days)[inner_mask].reshape(-1)
            inner_validation = torch.arange(len(states), device=env.device, dtype=env.idtype).reshape(-1, days)[~inner_mask].reshape(-1)
            if state['order'] is not None:
                state['order'] = state['order'].to(env.device)
            training_batch_size = effective_batch_size(len(inner_train), train_config.batch_size)
            evaluation_batch_size = effective_batch_size(len(inner_train), train_config.evaluation_batch_size)
            full_phase = None
            step = None
            last_saved = time.monotonic()
            maximum = train_config.affine_epochs + train_config.maximum_full_epochs
            while state['epoch'] < maximum and not all(state['stopped'].values()):
                epoch_started = time.monotonic()
                progress.update(stage='preparing-local-training-epoch', epoch=state['epoch'] + 1,
                    completed_observations=state['next_batch'], total_observations=len(inner_train))
                full = state['epoch'] >= train_config.affine_epochs
                if full != full_phase:
                    for policy in model.policies.values():
                        policy.deep.requires_grad_(full)
                        policy.affine_only = not full
                    step = LocalActionStep(model, optimizers, gradient_clip=train_config.gradient_clip,
                        compile=train_config.compile, use_cuda_graphs=train_config.use_cuda_graphs)
                    full_phase = full
                if state['order'] is None:
                    state['order'] = inner_train[torch.randperm(len(inner_train), generator=generator, device=env.device)]
                    state['next_batch'] = 0
                active_names = tuple(name for name in DIAGNOSTICS if not state['stopped'][name])
                captures_before = step.capture_count
                optimizer_captures_before = step.optimizer_capture_count
                capture_seconds_before = step.capture_seconds
                optimizer_seconds_before = step.optimizer_recapture_seconds
                training_started = time.monotonic()
                rates_used = {name: [float(group['lr']) for group in optimizers[name].param_groups] for name in active_names}
                stream_start = torch.cuda.Event(enable_timing=True) if env.device.type == 'cuda' else None
                stream_end = torch.cuda.Event(enable_timing=True) if env.device.type == 'cuda' else None
                if stream_start is not None:
                    stream_start.record()
                for offset in range(state['next_batch'], len(inner_train), training_batch_size):
                    indices = state['order'][offset:offset + training_batch_size]
                    progress.update(stage='compiling-or-training-local-transitions', epoch=state['epoch'] + 1,
                        completed_observations=offset, total_observations=len(inner_train), full_network=full)
                    step(states[indices], returns[indices], eligible[indices % days], active_names)
                    state['next_batch'] = offset + len(indices)
                    progress.update(stage='training-local-transitions', epoch=state['epoch'] + 1,
                        completed_observations=state['next_batch'], total_observations=len(inner_train), full_network=full)
                    if time.monotonic() - last_saved >= 60. or stop.is_set():
                        save('checkpointed_stop' if stop.is_set() else 'running')
                        last_saved = time.monotonic()
                    if stop.is_set():
                        raise InterruptedError('local action training stopped at a committed optimizer batch')
                training_seconds = time.monotonic() - training_started
                if stream_end is not None:
                    stream_end.record()
                progress['stage'] = 'internal-training-fold-evaluation'
                evaluation_started = time.monotonic()
                training_loss = _population_loss(model, states, returns, eligible, inner_train,
                    batch_size=evaluation_batch_size, forwards=step.forwards)
                training_evaluation_seconds = time.monotonic() - evaluation_started
                progress['stage'] = 'internal-validation-fold-evaluation'
                evaluation_started = time.monotonic()
                validation_loss = _population_loss(model, states, returns, eligible, inner_validation,
                    batch_size=effective_batch_size(len(inner_validation), train_config.evaluation_batch_size), forwards=step.forwards)
                validation_evaluation_seconds = time.monotonic() - evaluation_started
                state.update(epoch=state['epoch'] + 1, order=None, next_batch=0)
                state['history'].append({'epoch': state['epoch'], 'full_network': full,
                    'training': training_loss, 'internal_validation': validation_loss,
                    'learning_rates_used': rates_used, 'effective_batch_size': training_batch_size,
                    'cuda_stream_interval_seconds': stream_start.elapsed_time(stream_end) / 1000. if stream_start is not None else None})
                for name in active_names:
                    value = validation_loss[name]
                    tolerance = max(train_config.absolute_improvement, train_config.relative_improvement * abs(state['best_loss'][name]))
                    if value < state['best_loss'][name] - tolerance:
                        state['best_loss'][name], state['best_epoch'][name] = value, state['epoch']
                        state['best_state'][name] = _cpu_state(model.policies[name])
                        state['last_improvement'][name] = state['epoch']
                    schedulers[name].threshold = tolerance
                    schedulers[name].step(value)
                    at_floor = all(rate <= train_config.minimum_learning_rate * (1. + 1.e-12) for rate in rates_used[name])
                    if at_floor and state['floor_since'][name] is None:
                        state['floor_since'][name] = state['epoch']
                    floor_since = state['floor_since'][name]
                    since_improvement = state['epoch'] - max(state['last_improvement'][name], train_config.affine_epochs,
                        state['epoch'] if floor_since is None else floor_since)
                    if full and state['epoch'] >= train_config.affine_epochs + train_config.minimum_full_epochs and since_improvement >= train_config.convergence_patience:
                        state['stopped'][name] = True
                progress['stage'] = 'saving-local-action-checkpoint'
                checkpoint_started = time.monotonic()
                save('running')
                checkpoint_seconds = time.monotonic() - checkpoint_started
                print(json.dumps({'event': 'local_action_epoch', 'fold': fold, 'epoch': state['epoch'],
                    'full_network': full, 'training': training_loss, 'internal_validation': validation_loss,
                    'best_epochs': state['best_epoch'], 'elapsed_seconds': time.monotonic() - started,
                    'learning_rates_used': rates_used, 'effective_batch_size': training_batch_size,
                    'cuda_stream_interval_seconds': state['history'][-1]['cuda_stream_interval_seconds'],
                    'timings': {'epoch_wall_seconds': time.monotonic() - epoch_started,
                        'training_wall_seconds': training_seconds,
                        'full_graph_capture_seconds': step.capture_seconds - capture_seconds_before,
                        'optimizer_recapture_seconds': step.optimizer_recapture_seconds - optimizer_seconds_before,
                        'full_graph_captures': step.capture_count - captures_before,
                        'optimizer_graph_captures': step.optimizer_capture_count - optimizer_captures_before,
                        'training_population_evaluation_seconds': training_evaluation_seconds,
                        'internal_validation_evaluation_seconds': validation_evaluation_seconds,
                        'checkpoint_seconds': checkpoint_seconds}}), flush=True)
            converged = all(state['stopped'].values())
            return save('internal_validation_plateau' if converged else 'maximum_epochs_not_converged')
        finally:
            done.set()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def evaluate_fold(*, path: Path, output: Path, report: dict[str, object], train_config: LocalAuditTrainConfig,
                  cache: AuditHostCache | None = None) -> dict[str, object]:
    """Freeze training-selected actions before reading the outer holdout data."""
    checkpoint = torch.load(report['checkpoint'], map_location='cpu', weights_only=False)
    fields = dict(report['model_config'])
    fields['network'] = ActionNetworkConfig(**fields['network'])
    config = LocalActionConfig(**fields)
    if checkpoint['code_version'] != LocalActionConfig.version.unique_id48 or checkpoint['user_version'] != config.user_version:
        raise RuntimeError('frozen action checkpoint version mismatch')
    if train_config.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise RuntimeError('GPU evaluation requires a managed allocation')
    env = BaseEnvironment(device=train_config.device, precision=train_config.precision)
    model = IndependentLocalActions(env, config=config)
    for name in DIAGNOSTICS:
        model.policies[name].load_state_dict(checkpoint['state']['best_state'][name])
    model.eval()
    normalization_path = Path(report['checkpoint']).parent.parent / 'normalization.npz'
    with np.load(normalization_path, allow_pickle=False) as saved:
        mean = torch.as_tensor(saved['mean'], dtype=env.fdtype, device=env.device)
        scale = torch.as_tensor(saved['scale'], dtype=env.fdtype, device=env.device)
    identity, geometry = _path_identity(path)
    if hashlib.sha256(identity.encode()).hexdigest() != report['path_identity_sha256']:
        raise ValueError('frozen-policy evaluation paths differ from the fit')
    fold = int(report['fold'])
    training, holdout = fold_path_ranges(geometry['paths'])[fold], fold_path_ranges(geometry['paths'])[1 - fold]
    result = {'fold': fold, 'selection': report, 'measure_preconditioning': False, 'diagnostics': {}}
    arrays = {}
    checked = time.monotonic()
    with h5py.File(path, 'r') as stream, torch.no_grad():
        expiries = np.repeat(np.asarray(stream['expiry_days']), stream['strikes'].shape[-1])
        mask = torch.as_tensor(np.column_stack((np.ones(geometry['days'], dtype=bool),
            expiries[None] > np.arange(geometry['days'])[:, None])), device=env.device, dtype=torch.bool)
        for split, selected in [('training', training), ('holdout', holdout)]:
            records = {name: np.empty((selected.stop - selected.start, 4), dtype=np.float64) for name in DIAGNOSTICS}
            for start in range(selected.start, selected.stop, train_config.upload_paths):
                if time.monotonic() - checked >= 60.:
                    if cache is not None:
                        cache.check_headroom()
                    check_resource_headroom()
                    checked = time.monotonic()
                stop = min(start + train_config.upload_paths, selected.stop)
                state_values = read_audit_block(stream, 'features', slice(start, stop), cache=cache)
                return_values = read_audit_block(stream, 'returns', slice(start, stop), cache=cache)
                states = (torch.as_tensor if state_values.flags.writeable else torch.tensor)(state_values, dtype=env.fdtype, device=env.device)
                returns = (torch.as_tensor if return_values.flags.writeable else torch.tensor)(return_values, dtype=env.fdtype, device=env.device)
                states = (states - mean) / scale
                for name in DIAGNOSTICS:
                    action = config.action_scale * model.policies[name](states, mask)
                    gain = (action.to(torch.float64) * returns.to(torch.float64)).sum(axis=-1)
                    action_squared = action.to(torch.float64).square().sum(axis=-1)
                    exponent = -gain + .5 * action_squared
                    squared_value = gain - .5 * action_squared - .5 * config.squared_lambda * gain.square()
                    packed = torch.stack((gain.mean(axis=-1), action_squared.mean(axis=-1),
                        torch.expm1(exponent).mean(axis=-1), squared_value.mean(axis=-1)), dim=-1)
                    values = packed.cpu().numpy()
                    if np.any(~np.isfinite(values)):
                        raise FloatingPointError('non-finite held-out objective; no observations discarded')
                    records[name][start - selected.start:stop - selected.start] = values
                print(json.dumps({'event': 'frozen_local_action_evaluation', 'fold': fold, 'split': split,
                    'completed_paths': stop - selected.start, 'total_paths': selected.stop - selected.start}), flush=True)
            for name in DIAGNOSTICS:
                values = records[name]
                entropy_loss = float(np.log1p(values[:, 2].mean()))
                result['diagnostics'].setdefault(name, {})[split] = {
                    'frozen_action_return': frozen_score_statistics(values[:, 0]),
                    'mean_squared_action': float(values[:, 1].mean()), 'log_exponential_loss_relative_to_zero': entropy_loss,
                    'mean_squared_objective_relative_to_zero': float(values[:, 3].mean())}
                arrays[f'{name}_{split}'] = values
    _atomic_npz(output / 'frozen_path_scores.npz', arrays)
    write_json(output / 'frozen_evaluation.json', result)
    return result


def run_conditional_audit(*, path: Path, output: Path, protocol: LocalAuditProtocol,
                          train_config: LocalAuditTrainConfig, load_only: bool = False) -> dict[str, object]:
    """Own an optional raw cache across both unchanged training/holdout folds."""
    output.mkdir(parents=True, exist_ok=True)
    cache = None
    try:
        if train_config.host_cache and not load_only:
            check_resource_headroom()
            with h5py.File(path, 'r') as stream:
                cache = cache_audit_arrays(stream)
        return _run_conditional_audit(path=path, output=output, protocol=protocol,
            train_config=train_config, load_only=load_only, cache=cache)
    finally:
        if cache is not None:
            cache.release()


def _run_conditional_audit(*, path: Path, output: Path, protocol: LocalAuditProtocol,
                           train_config: LocalAuditTrainConfig, load_only: bool,
                           cache: AuditHostCache | None) -> dict[str, object]:
    """Run both full-path folds and report their frozen action statistics separately."""
    output.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in range(2):
        directory = output / f'fold-{fold}'
        report = train_fold(path=path, output=directory, fold=fold, protocol=protocol,
                            train_config=train_config, load_only=load_only, cache=cache)
        if report['stop_reason'] != 'internal_validation_plateau':
            raise RuntimeError('conditional policy has not met the predeclared training-only convergence rule')
        evaluation_path = directory / 'frozen_evaluation.json'
        retained = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else None
        if retained is not None and retained['selection'] == report:
            folds.append(retained)
        else:
            folds.append(evaluate_fold(path=path, output=directory, report=report, train_config=train_config, cache=cache))
        if train_config.device == 'cuda':
            torch.cuda.empty_cache()
    result = {'measure': 'original Q; no fitted weights or return centering', 'folds': folds,
              'diagnostics': {}, 'interpretation': 'conditional test relative to the predictable state and uncapped affine-plus-deep strategy class'}
    for name in DIAGNOSTICS:
        statistics = [fold['diagnostics'][name]['holdout']['frozen_action_return'] for fold in folds]
        weights = np.asarray([entry['paths'] for entry in statistics], dtype=np.float64)
        weights /= weights.sum()
        mean = float(sum(weight * entry['mean'] for weight, entry in zip(weights, statistics, strict=True)))
        conservative_error = float(sum(weight * entry['standard_error'] for weight, entry in zip(weights, statistics, strict=True)))
        statistic = mean / conservative_error if conservative_error > 0. else None
        invalid_zero_variance = any(entry['standard_error'] == 0. and entry['mean'] != 0. for entry in statistics)
        if invalid_zero_variance:
            statistic = None
        pvalues = [2. * norm.sf(abs(float(entry['action_conditional_statistic']))) for entry in statistics
            if entry['action_conditional_statistic'] is not None]
        result['diagnostics'][name] = {'cross_fitted_mean': mean, 'cross_fitted_standard_error_upper_bound_estimate': conservative_error,
            'cross_fitted_action_statistic': statistic,
            'test_status': 'degenerate_nonzero_fold' if invalid_zero_variance else ('ok' if conservative_error > 0. else 'degenerate_zero_score'),
            'zero_score_folds': [fold['fold'] for fold, entry in zip(folds, statistics, strict=True)
                if entry['standard_error'] == 0. and entry['mean'] == 0.],
            'foldwise_bonferroni_two_sided_pvalue': float(min(1., 2. * min(pvalues))) if pvalues else None,
            'uncertainty_note': 'Opposite-fold fitted scores need not be independent. The combined SE sums weighted fold SEs instead of assuming zero cross-fold covariance; individual fold scores use independent antithetic pairs.'}
    write_json(output / 'conditional_report.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16384)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--minimum-learning-rate', type=float, default=1.e-6)
    parser.add_argument('--learning-rate-patience', type=int, default=50)
    parser.add_argument('--convergence-patience', type=int, default=100)
    parser.add_argument('--affine-epochs', type=int, default=100)
    parser.add_argument('--minimum-full-epochs', type=int, default=200)
    parser.add_argument('--maximum-full-epochs', type=int, default=2000)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--host-cache', action='store_true')
    arguments = parser.parse_args()
    run_conditional_audit(path=arguments.paths, output=arguments.output_directory, protocol=LocalAuditProtocol(),
        train_config=LocalAuditTrainConfig(device=arguments.device, batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate, minimum_learning_rate=arguments.minimum_learning_rate,
            learning_rate_patience=arguments.learning_rate_patience, convergence_patience=arguments.convergence_patience,
            affine_epochs=arguments.affine_epochs, minimum_full_epochs=arguments.minimum_full_epochs,
            maximum_full_epochs=arguments.maximum_full_epochs,
            compile=arguments.compile, host_cache=arguments.host_cache),
        load_only=arguments.load_only)


if __name__ == '__main__':
    main()
