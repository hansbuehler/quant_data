"""Explicit optimizer-only fixed-metric transitions with retained raw secants."""

from __future__ import annotations

import hashlib
import io
import math
import statistics
from copy import deepcopy
from pathlib import Path

import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.full_gradient_audit import _atomic_torch_save
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.torch import BaseEnvironment


def prepare_deep_metric_continuation(*, source: Path, output: Path, source_sha256: str,
                                     history_window: int = 30, curvature_floor: float = 1.e-12) -> dict:
    """Freeze a reliable positive deep secant scale without changing the policy.

    Raw-coordinate inverse-BFGS updates preserve positive definiteness for
    any positive-definite initial metric. Existing accepted raw curvature
    pairs therefore remain usable after this explicitly recorded transition.
    No objective, model, normalization, stopping threshold or path weight is
    changed. The derived checkpoint must enter a distinct declared run.
    """
    verify_inp(type(history_window) is int and history_window > 0 and 0. < curvature_floor < 1.,
        lambda: 'a positive secant window and reliable-curvature threshold are required')
    verify_inp(source.resolve() != output.resolve(), lambda: 'metric continuation must not overwrite its source')
    verify_inp(output.resolve() != output.with_suffix('.json').resolve(),
        lambda: 'the checkpoint and its manifest must have distinct paths')
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != source_sha256:
        raise ValueError('the accepted metric-continuation source changed')
    saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    state, optimizer = saved['state'], saved['optimizer']
    fields = dict(state['identity']['model_config'])
    fields['network'] = ActionNetworkConfig(**fields['network'])
    if saved['code_version'] != LocalActionConfig.version.unique_id48 or saved['user_version'] != fields['user_version']:
        raise ValueError('metric continuation model version changed')
    with torch.random.fork_rng(devices=[]):
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=LocalActionConfig(**fields))
    policy.load_state_dict(saved['model'], strict=True)
    shapes = {name: list(parameter.shape) for name, parameter in policy.named_parameters()}
    metric = optimizer['metric_identity']
    if (metric != state['identity']['metric_identity'] or metric['parameter_shapes'] != shapes
        or metric['factor_sha256'] != hashlib.sha256(optimizer['metric_factors'].contiguous().numpy().tobytes()).hexdigest()
        or not math.isfinite(metric['deep_scale']) or metric['deep_scale'] <= 0.
        or any(value.dtype != torch.float64 or not torch.isfinite(value).all() for value in saved['model'].values())):
        raise ValueError('metric continuation geometry, factors or accepted weights changed')
    positions, offset = [], 0
    for name, parameter in policy.named_parameters():
        if name.startswith('deep.'):
            positions.extend(range(offset, offset+parameter.numel()))
        offset += parameter.numel()
    indices = torch.tensor(positions, dtype=torch.int64)
    history = optimizer['history']
    if not history or len(history) > optimizer['history_size'] or not positions:
        raise ValueError('metric continuation needs retained nonlinear curvature history')
    estimates = []
    for number, (step, change, reciprocal) in enumerate(history):
        if (step.shape != (offset,) or change.shape != step.shape or reciprocal.numel() != 1
            or any(value.dtype != torch.float64 or not torch.isfinite(value).all() for value in (step, change, reciprocal))):
            raise ValueError('retained raw secant geometry or precision changed')
        raw_curvature = float(torch.dot(step, change))
        if not raw_curvature > 0. or not math.isclose(float(reciprocal)*raw_curvature, 1., rel_tol=1.e-10, abs_tol=1.e-12):
            raise ValueError('retained raw curvature or reciprocal is invalid')
        if number < len(history)-history_window:
            continue
        deep_step, deep_change = step[indices], change[indices]
        curvature = float(torch.dot(deep_step, deep_change))
        threshold = curvature_floor*float(torch.linalg.vector_norm(deep_step)*torch.linalg.vector_norm(deep_change))
        squared_change = float(torch.dot(deep_change, deep_change))
        if math.isfinite(curvature) and curvature > threshold and math.isfinite(squared_change) and squared_change > 0.:
            inverse_curvature = curvature/squared_change
            if math.isfinite(inverse_curvature) and inverse_curvature > 0.:
                estimates.append({'history_index': number, 'curvature': curvature,
                    'curvature_threshold': threshold, 'squared_gradient_change': squared_change,
                    'inverse_curvature': inverse_curvature})
    if not estimates:
        raise ValueError('no reliable positive deep secant supports a scale transition')
    scale = statistics.median(item['inverse_curvature'] for item in estimates)
    if not math.isfinite(scale) or scale <= 0.:
        raise ValueError('the derived median metric scale must be finite and positive')
    new_metric = deepcopy(metric)
    new_metric['deep_scale'] = scale
    transition = {'source_checkpoint': str(source.resolve()), 'source_checkpoint_sha256': digest,
        'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'history_window': history_window, 'curvature_floor': curvature_floor, 'reliable_secants': estimates,
        'source_metric_identity': deepcopy(metric), 'derived_metric_identity': deepcopy(new_metric),
        'native_iteration': state['history'][-1]['iteration'],
        'committed_full_objective_evaluations': state['full_objective_evaluations'],
        'original_initial_gradient_norm': state['initial_gradient_norm'],
        'original_gradient_threshold': state['history'][-1]['gradient_threshold'],
        'raw_history_retained': True, 'model_weights_preserved': True, 'rng_preserved': True,
        'data_objective_and_normalization_unchanged': True,
        'shared_optimizer_controls_inference': 'independent fresh confirmation only, not original-population crossfit'}
    saved['optimizer']['metric_identity'] = deepcopy(new_metric)
    saved['state']['identity']['metric_identity'] = deepcopy(new_metric)
    saved['initializer_provenance'] = {'previous_initializer_provenance': saved['initializer_provenance'],
        'fixed_metric_continuation': transition}
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output)+'.lock', timeout=0):
        if output.exists() or output.with_suffix('.json').exists():
            raise ValueError('derived optimizer checkpoint already exists; preserve it')
        _atomic_torch_save(output, saved)
        manifest = {'transition': transition, 'derived_checkpoint': str(output.resolve()),
            'derived_checkpoint_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
        write_json(output.with_suffix('.json'), manifest)
    return manifest
