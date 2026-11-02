"""Read-only training-row GGN factor collection for a numerical solve base."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import _forward_geometry, action_point_sha256
from quantdata.gromm2.kronecker_audit_preconditioner import (
    exact_head_ggn_factor,
    output_curvature_sqrt_probe,
)


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class KroneckerCollectionConfig:
    """Private probes; seed, row order and chunking jointly define replay."""

    probe_seed: int = 2026110201
    probe_count: int = 2

    def __post_init__(self) -> None:
        verify_inp(type(self.probe_seed) is int and 0 <= self.probe_seed < 2**63-1024,
            lambda: 'a nonnegative reproducible private probe seed is required')
        verify_inp(type(self.probe_count) is int and 1 <= self.probe_count <= 1024,
            lambda: 'a positive bounded number of independent curvature probes is required')


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def layerwise_ggn_block(*, policy: AffineDeepAction, features: torch.Tensor, returns: torch.Tensor,
                        eligible: torch.Tensor, probes: torch.Tensor, diagnostic: str) -> dict:
    """Return unnormalized small factors from the actual canonical forward.

    Hooks observe linear inputs/preactivations and are always removed. Hidden
    covectors use the actual SiLU derivative and detached masked output seeds.
    There is no per-observation loss mean in these adjoints. Only the caller
    divides accumulated factors by the complete selected training-row count.
    """
    if (policy.affine_only or type(policy.deep.activation) is not torch.nn.SiLU
        or policy.deep.activation.inplace or policy.deep.use_final_activation
        or any(module._forward_hooks or module._forward_pre_hooks for module in policy.modules())
        or features.ndim != 2 or features.dtype != torch.float64 or not len(features)
        or features.shape[1] != policy.config.feature_count
        or returns.shape != (len(features), policy.config.instrument_count)
        or eligible.shape != returns.shape or eligible.dtype != torch.bool
        or features.device != returns.device or eligible.device != features.device
        or probes.ndim != 3 or probes.shape[1:] != returns.shape or not len(probes)):
        raise ValueError('factor collection requires the unmodified full SiLU policy, canonical rows and explicit output probes')
    layers = [(name, module) for name, module in policy.named_modules()
        if name.startswith('deep.') and type(module) is torch.nn.Linear]
    expected = [('deep.I0', policy.deep.I0),
        *[(f'deep.In.{index}', module) for index, module in enumerate(policy.deep.In)]]
    if layers != expected or any(value.dtype != torch.float64 or value.device != features.device for value in policy.parameters()):
        raise ValueError('deep linear-layer geometry or canonical parameter precision changed')
    inputs, preactivations, hooks = {}, {}, []
    try:
        for name, module in layers:
            hooks.append(module.register_forward_pre_hook(
                lambda module, arguments, key=name: inputs.__setitem__(key, arguments[0])))
            hooks.append(module.register_forward_hook(
                lambda module, arguments, output, key=name: preactivations.__setitem__(key, output)))
        action = policy(features, eligible)
    finally:
        for handle in hooks:
            handle.remove()
    activation_sums, probe_sums, derivatives = {}, {}, {}
    for index, (name, module) in enumerate(layers):
        activation = inputs[name]
        if module.bias is not None:
            activation = torch.cat((activation, torch.ones_like(activation[:, :1])), dim=1)
        activation_sums[name] = activation.T@activation
        probe_sums[name] = features.new_zeros((len(probes), module.out_features, module.out_features))
        if index+1 < len(layers):
            preactivation = preactivations[name]
            derivatives[name] = torch.func.jvp(policy.deep.activation, (preactivation,),
                (torch.ones_like(preactivation),))[1]
    for probe_index, probe in enumerate(probes):
        delta = output_curvature_sqrt_probe(action, returns, eligible, probe,
            diagnostic=diagnostic, action_scale=policy.config.action_scale, squared_lambda=policy.config.squared_lambda)
        for index in range(len(layers)-1, -1, -1):
            name, module = layers[index]
            probe_sums[name][probe_index] = delta.T@delta
            if index:
                delta = (delta@module.weight)*derivatives[layers[index-1][0]]
    head = layers[-1][0]
    exact_head = len(features)*exact_head_ggn_factor(action, returns, eligible,
        diagnostic=diagnostic, action_scale=policy.config.action_scale, squared_lambda=policy.config.squared_lambda)
    values = (*activation_sums.values(), *probe_sums.values(), exact_head)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError('nonfinite layer moments; no row, covector or factor may be censored')
    return {'activation_sums': activation_sums, 'probe_output_sums': probe_sums,
        'exact_head_sum': exact_head, 'head_name': head, 'observations': len(features)}


@torch.no_grad()
def collect_kronecker_moments(*, policy: AffineDeepAction, problem: AffineAuditProblem, diagnostic: str,
                             data_identity: str, source_point_sha256: str,
                             config: KroneckerCollectionConfig = KroneckerCollectionConfig(),
                             progress: Progress | None = None) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Collect all and only original inner rows; leave native state untouched.

    Activation factors and head output curvature are exact empirical means.
    Hidden output factors use explicitly seeded probes, not empirical losses
    or holdout data. Kronecker separation remains an optimizer approximation.
    The generated factor identity binds source, rows, eligibility, precision,
    chunking, probes and actual moment bytes; it is not a convergence receipt.
    """
    if (len(data_identity) != 64 or any(value not in '0123456789abcdef' for value in data_identity)
        or action_point_sha256(policy) != source_point_sha256
        or diagnostic not in ('squared', 'entropy')
        or policy.config.action_scale != problem.action_scale or policy.config.squared_lambda != problem.squared_lambda):
        raise ValueError('the unchanged original point, objective, scales and training-data identity are required')
    started = time.monotonic()
    entries = (*policy.named_parameters(), *policy.named_buffers())
    versions = tuple(value._version for name, value in entries)
    gradients = tuple((parameter.grad, None if parameter.grad is None else parameter.grad._version) for parameter in policy.parameters())
    geometry, source_config = _forward_geometry(policy), policy.config
    data = (problem.states, problem.returns, problem.rows, problem.eligible)
    data_versions = tuple(value._version for value in data)
    controls = (problem.action_scale, problem.squared_lambda, problem.config)

    def check_fixed() -> None:
        current = (*policy.named_parameters(), *policy.named_buffers())
        actual_gradients = tuple((parameter.grad, None if parameter.grad is None else parameter.grad._version) for parameter in policy.parameters())
        actual_data = (problem.states, problem.returns, problem.rows, problem.eligible)
        if (len(current) != len(entries)
            or any(name != saved_name or value is not saved for (name, value), (saved_name, saved)
                in zip(current, entries, strict=True))
            or tuple(value._version for name, value in current) != versions
            or len(actual_gradients) != len(gradients)
            or any(value is not saved or revision != saved_revision for (value, revision), (saved, saved_revision)
                in zip(actual_gradients, gradients, strict=True))
            or any(value is not saved for value, saved in zip(actual_data, data, strict=True))
            or tuple(value._version for value in actual_data) != data_versions
            or (problem.action_scale, problem.squared_lambda, problem.config) != controls
            or policy.affine_only or policy.config != source_config or _forward_geometry(policy) != geometry):
            raise ValueError('canonical source, training rows, original objective or gradient buffers changed during collection')

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(problem.states.device).clone() if problem.states.is_cuda else None
    generators = [torch.Generator(device=problem.states.device).manual_seed(config.probe_seed+index)
        for index in range(config.probe_count)]
    initial_probe_rng = [_digest(generator.get_state()) for generator in generators]
    activations, probe_outputs, exact_head = {}, {}, None
    completed = 0
    check_fixed()
    for augmented, returns, mask in problem.blocks():
        check_fixed()
        probes = torch.stack([torch.randint(0, 2, returns.shape, generator=generator,
            device=returns.device, dtype=torch.int8).to(torch.float64)*2.-1. for generator in generators])
        block = layerwise_ggn_block(policy=policy, features=augmented[:, :-1], returns=returns,
            eligible=mask, probes=probes, diagnostic=diagnostic)
        if not activations:
            activations = {name: torch.zeros_like(value) for name, value in block['activation_sums'].items()}
            probe_outputs = {name: torch.zeros_like(value) for name, value in block['probe_output_sums'].items()}
            exact_head = torch.zeros_like(block['exact_head_sum'])
        for name in activations:
            activations[name].add_(block['activation_sums'][name])
            probe_outputs[name].add_(block['probe_output_sums'][name])
        exact_head.add_(block['exact_head_sum'])
        completed += block['observations']
        check_fixed()
        if progress is not None:
            progress({'stage': 'kronecker_training_factor_collection', 'completed_observations': completed,
                'total_observations': len(problem.rows), 'probe_count': config.probe_count,
                'elapsed_seconds': time.monotonic()-started,
                'remaining_seconds': (time.monotonic()-started)*(len(problem.rows)-completed)/completed})
    check_fixed()
    if (completed != len(problem.rows) or not torch.equal(cpu_rng, torch.get_rng_state())
        or cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(problem.states.device))
        or action_point_sha256(policy) != source_point_sha256):
        raise RuntimeError('factor collection changed canonical state/RNG or omitted selected training observations')
    moments, variability = {}, {}
    for name in activations:
        activation = activations[name]/completed
        separate_probes = probe_outputs[name]/completed
        average = separate_probes.mean(dim=0)
        output = exact_head/completed if name == block['head_name'] else average
        if not bool(torch.isfinite(activation).all()) or not bool(torch.isfinite(output).all()):
            raise FloatingPointError('nonfinite full-row moments; no factor may be censored')
        moments[name] = (activation, output)
        denominator = float(torch.linalg.matrix_norm(average))
        variability[name] = {'probe_relative_disagreement': None if not denominator else
            float(torch.linalg.vector_norm(separate_probes-average, dim=(-2, -1)).mean())/denominator,
            'head_uses_exact_output_factor': name == block['head_name'],
            'probe_mean_relative_error_vs_exact_head': None if name != block['head_name']
                or float(torch.linalg.matrix_norm(output)) == 0. else float(torch.linalg.matrix_norm(average-output))
                    /float(torch.linalg.matrix_norm(output))}
    identity = {'schema': 'gromm2-training-only-layerwise-GGN-factor-collection-v1',
        'data_identity': data_identity, 'point_sha256': source_point_sha256, 'diagnostic': diagnostic,
        'model_config': asdict(policy.config), 'model_code_version': policy.config.version.unique_id48,
        'collection_config': asdict(config), 'collection_code_version': config.version.unique_id48,
        'action_scale': problem.action_scale, 'squared_lambda': problem.squared_lambda,
        'rows_sha256': _digest(problem.rows), 'eligibility_sha256': _digest(problem.eligible),
        'resident_shapes': [list(value.shape) for value in data],
        'resident_dtypes': [str(value.dtype) for value in data], 'arithmetic_dtype': 'torch.float64',
        'working_memory_batch_size': problem.config.batch_size, 'training_observations': completed,
        'probe_rng_initial_sha256': initial_probe_rng,
        'probe_rng_final_sha256': [_digest(generator.get_state()) for generator in generators],
        'moment_sha256': {name: [_digest(value) for value in pair] for name, pair in moments.items()},
        'normalization': 'sum per-row moments then divide exactly once by all selected rows',
        'kronecker_separation_is_exact': False, 'original_objective_changed': False,
        'original_stopping_metric_changed': False, 'native_optimizer_updates': 0}
    return moments, {'identity': identity,
        'factor_data_identity': hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'probe_variability': variability, 'elapsed_seconds': time.monotonic()-started,
        'complete_factor_collection_forward_passes': 1,
        'complete_probe_adjoint_passes': config.probe_count,
        'canonical_objective_gradient_passes': 0, 'exact_hessian_vector_products': 0,
        'global_rng_preserved': True, 'outer_gradient_convergence_certified': False}
