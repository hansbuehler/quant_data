"""Training-only SPD layer geometry, separate from original audit gates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import _forward_geometry, action_point_sha256
from quantdata.gromm2.lbfgs_audit_preconditioner import FrozenLBFGSPreconditioner, metric_energy
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.metric_gmres_audit_proposal import TransposeAffineMetric


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class KroneckerBaseConfig:
    """Positive factor floors regularize only the numerical inverse base."""

    relative_factor_floor: float = 1.e-6
    absolute_factor_floor: float = 1.e-12

    def __post_init__(self) -> None:
        verify_inp(math.isfinite(self.relative_factor_floor) and 0. < self.relative_factor_floor < 1.,
            lambda: 'the numerical relative factor floor must lie strictly between zero and one')
        verify_inp(math.isfinite(self.absolute_factor_floor) and self.absolute_factor_floor > 0.,
            lambda: 'the numerical absolute factor floor must be finite and positive')


def _curvature_parts(action: torch.Tensor, returns: torch.Tensor, eligible: torch.Tensor, *,
                     diagnostic: str, action_scale: float, squared_lambda: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    if (action.ndim != 2 or action.shape != returns.shape or eligible.shape != action.shape
        or action.dtype != torch.float64 or returns.dtype != action.dtype or eligible.dtype != torch.bool
        or action.device != returns.device or action.device != eligible.device or not len(action)
        or not math.isfinite(action_scale) or action_scale <= 0.
        or not math.isfinite(squared_lambda) or squared_lambda < 0.
        or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(returns).all())):
        raise ValueError('output curvature requires finite float64 actions/returns, boolean eligibility and original scales')
    if bool(action[~eligible].ne(0.).any()):
        raise ValueError('output curvature requires the already eligibility-masked canonical action')
    if diagnostic == 'squared':
        vector, weight, coefficient = returns, torch.ones_like(action[:, 0]), squared_lambda*action_scale**2
    elif diagnostic == 'entropy':
        vector = action-returns
        exponent = action_scale**2*(-(action*returns).sum(dim=1)+.5*action.square().sum(dim=1))
        weight, coefficient = torch.exp(exponent), action_scale**2
    else:
        raise ValueError('unknown original conditional-audit objective')
    if not math.isfinite(coefficient) or not bool(torch.isfinite(weight).all()) or not bool((weight > 0.).all()):
        raise FloatingPointError('output curvature weight is unrepresentable; no row or exponent may be clipped')
    return vector, weight, coefficient


@torch.no_grad()
def output_curvature_sqrt_probe(action: torch.Tensor, returns: torch.Tensor, eligible: torch.Tensor,
                                probe: torch.Tensor, *, diagnostic: str, action_scale: float,
                                squared_lambda: float = 1.) -> torch.Tensor:
    """Return a DETACHED per-row action covector, with no sample-mean factor.

    Before eligibility, Q=c(I+alpha vv'). Independent unit-covariance probes
    have E[(Q^.5 probe)(Q^.5 probe)']=Q. Eligibility belongs to the actual
    action Jacobian and is applied after this symmetric square root. A future
    VJP must treat this covector as fixed, not differentiate Q through it.
    """
    vector, weight, coefficient = _curvature_parts(action, returns, eligible,
        diagnostic=diagnostic, action_scale=action_scale, squared_lambda=squared_lambda)
    if (probe.shape != action.shape or probe.dtype != action.dtype or probe.device != action.device
        or not bool(torch.isfinite(probe).all())):
        raise ValueError('output curvature probe geometry or precision changed')
    denominator = torch.sqrt(1.+coefficient*vector.square().sum(dim=1))+1.
    result = torch.sqrt(weight)[:, None]*(probe+coefficient/denominator[:, None]
        *vector*(vector*probe).sum(dim=1)[:, None])
    result = result*eligible
    if not bool(torch.isfinite(denominator).all()) or not bool(torch.isfinite(result).all()):
        raise FloatingPointError('nonfinite output curvature probe; no observations may be discarded')
    return result.detach()


@torch.no_grad()
def exact_head_ggn_factor(action: torch.Tensor, returns: torch.Tensor, eligible: torch.Tensor, *,
                          diagnostic: str, action_scale: float, squared_lambda: float = 1.) -> torch.Tensor:
    """Mean masked head curvature, not the joint layer Kronecker block.

    This small factor is exact on supplied rows. E[aa' tensor Q] generally
    differs from E[aa'] tensor E[Q]; only the numerical base uses that latter
    approximation. Collection over blocks must weight this mean by row count.
    """
    vector, weight, coefficient = _curvature_parts(action, returns, eligible,
        diagnostic=diagnostic, action_scale=action_scale, squared_lambda=squared_lambda)
    masked = vector*eligible
    result = (torch.diag((weight[:, None]*eligible).sum(dim=0))
        +coefficient*masked.T@(weight[:, None]*masked))/len(action)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('nonfinite exact head curvature factor')
    return result


@torch.no_grad()
def positive_inverse_factor(moment: torch.Tensor, *, config: KroneckerBaseConfig) -> tuple[torch.Tensor, dict]:
    """Factor a full-rank SPD inverse without truncating any direction."""
    if (moment.ndim != 2 or moment.shape[0] != moment.shape[1] or not len(moment)
        or moment.dtype != torch.float64 or not bool(torch.isfinite(moment).all())):
        raise ValueError('a finite square float64 training moment is required')
    scale = float(moment.abs().max())
    tolerance = 128*torch.finfo(moment.dtype).eps*len(moment)*max(scale, torch.finfo(moment.dtype).tiny)
    if float((moment-moment.T).abs().max()) > tolerance:
        raise ValueError('training moment is not symmetric within arithmetic precision')
    eigenvalues, vectors = torch.linalg.eigh(.5*(moment+moment.T))
    if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[0]) < -tolerance:
        raise FloatingPointError('the supplied training moment has nonfinite or materially negative curvature')
    cutoff = max(config.absolute_factor_floor, config.relative_factor_floor*float(eigenvalues[-1]))
    stabilized = torch.maximum(eigenvalues, eigenvalues.new_tensor(cutoff))
    factor = vectors/torch.sqrt(stabilized)[None, :]
    if not bool(torch.isfinite(factor).all()):
        raise FloatingPointError('unrepresentable positive inverse factor')
    return factor, {'width': len(moment), 'minimum_unfloored_eigenvalue': float(eigenvalues[0]),
        'maximum_unfloored_eigenvalue': float(eigenvalues[-1]), 'positive_eigenvalue_floor': cutoff,
        'eigenvalues_floored_for_preconditioning': int((eigenvalues < cutoff).sum()),
        'stabilized_condition': float(stabilized[-1]/stabilized[0]),
        'objective_ridge_added': False, 'coefficient_directions_discarded': False}


class FrozenKroneckerLBFGS:
    """Replay every native positive secant over a separately identified SPD base.

    Original affine base and stopping metric M are retained. Every deep linear
    block, including its bias, receives B^-1 D A^-1. This is a solve operator
    only: native optimizer state, model, objectives and original gates remain
    untouched. It neither collects moments nor certifies convergence.
    """

    def __init__(self, optimizer: MetricLBFGS, policy: AffineDeepAction, *,
                 moments: dict[str, tuple[torch.Tensor, torch.Tensor]], factor_data_identity: str,
                 config: KroneckerBaseConfig = KroneckerBaseConfig()) -> None:
        if (len(factor_data_identity) != 64 or any(value not in '0123456789abcdef' for value in factor_data_identity)
            or policy.affine_only):
            raise ValueError('the original full policy and source-bound training-factor identity are required')
        self.frozen = FrozenLBFGSPreconditioner(optimizer, policy)
        self.original_metric_guard = TransposeAffineMetric(optimizer.metric)
        self.source_sha256 = self.frozen.source_sha256
        self.point_sha256 = action_point_sha256(policy)
        self.policy, self.geometry = policy, _forward_geometry(policy)
        self.model_entries = (*policy.named_parameters(), *policy.named_buffers())
        self.model_objects = tuple(value for name, value in self.model_entries)
        self.model_versions = tuple(value._version for value in self.model_objects)
        self.model_config = policy.config
        names = tuple(name[:-7] for name in self.frozen.metric.slices if name.startswith('deep.') and name.endswith('.weight'))
        if set(names) != set(moments):
            raise ValueError('training factors must cover every original deep linear block, not a reduced policy')
        self.layouts, self.factors, reports = {}, {}, {}
        covered = set()
        for name in names:
            weight_name, bias_name = f'{name}.weight', f'{name}.bias'
            weight = self.frozen.metric.slices[weight_name]
            bias = self.frozen.metric.slices.get(bias_name)
            rows, columns = weight[1]
            activation, output = moments[name]
            expected = ((columns+int(bias is not None),)*2, (rows,)*2)
            if ((tuple(activation.shape), tuple(output.shape)) != expected
                or activation.device != self.frozen.metric.factors.device or output.device != activation.device):
                raise ValueError('layer factor shapes or managed device differ from the original bias-augmented policy')
            input_factor, input_report = positive_inverse_factor(activation, config=config)
            output_factor, output_report = positive_inverse_factor(output, config=config)
            self.factors[name] = (input_factor, output_factor)
            self.layouts[name] = (weight, bias)
            reports[name] = {'activation': input_report, 'output': output_report}
            covered.add(weight_name)
            if bias is not None:
                covered.add(bias_name)
        if covered != {name for name in self.frozen.metric.slices if name.startswith('deep.')}:
            raise ValueError('the original deep parameter geometry contains an unsupported non-linear-layer parameter')
        self.objects = tuple(tensor for factors in self.factors.values() for tensor in factors)
        self.versions = tuple(value._version for value in self.objects)
        self.layout_identity = tuple(self.layouts.items())
        self.identity = {'point_sha256': self.point_sha256, 'factor_data_identity': factor_data_identity,
            'base_config': asdict(config), 'source_optimizer_sha256': self.source_sha256,
            'factor_sha256': {name: [hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
                for value in factors] for name, factors in self.factors.items()},
            'layer_reports': reports, 'all_raw_secants_retained': True,
            'original_stopping_metric_unchanged': True, 'native_optimizer_changed': False,
            'objective_ridge_added': False, 'coefficient_directions_discarded': False}

    def check_fixed(self) -> None:
        self.frozen.check_fixed()
        self.original_metric_guard.check_fixed()
        actual = tuple(tensor for factors in self.factors.values() for tensor in factors)
        model = (*self.policy.named_parameters(), *self.policy.named_buffers())
        if (len(actual) != len(self.objects)
            or any(value is not saved for value, saved in zip(actual, self.objects, strict=True))
            or tuple(value._version for value in actual) != self.versions
            or tuple(self.layouts.items()) != self.layout_identity
            or len(model) != len(self.model_entries)
            or any(name != saved_name or value is not saved for (name, value), (saved_name, saved)
                in zip(model, self.model_entries, strict=True))
            or tuple(value._version for value in self.model_objects) != self.model_versions
            or self.policy.affine_only or self.policy.config != self.model_config
            or _forward_geometry(self.policy) != self.geometry):
            raise ValueError('the frozen Kronecker base or its canonical source geometry changed')

    def _check_vector(self, value: torch.Tensor) -> None:
        metric = self.frozen.metric
        if (value.shape != (metric.size,) or value.dtype != torch.float64
            or value.device != metric.factors.device or not bool(torch.isfinite(value).all())):
            raise ValueError('the Kronecker solve vector geometry, precision or finiteness changed')

    @torch.no_grad()
    def base_and_energy(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.check_fixed()
        metric = self.frozen.metric
        self._check_vector(value)
        affine = torch.zeros_like(value)
        affine[metric.weight[0]], affine[metric.bias[0]] = value[metric.weight[0]], value[metric.bias[0]]
        result, energy = metric(affine), metric_energy(metric, affine)
        for name, (weight, bias) in self.layouts.items():
            gradient = value[weight[0]].reshape(weight[1])
            if bias is not None:
                gradient = torch.cat((gradient, value[bias[0]][:, None]), dim=1)
            input_factor, output_factor = self.factors[name]
            coordinates = output_factor.T@gradient@input_factor
            transformed = output_factor@coordinates@input_factor.T
            energy.add_(coordinates.square().sum())
            result[weight[0]] = transformed[:, :weight[1][1]].reshape(-1)
            if bias is not None:
                result[bias[0]] = transformed[:, -1]
        if not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(energy)):
            raise FloatingPointError('unrepresentable Kronecker base image or energy')
        return result, energy

    @torch.no_grad()
    def apply_and_energy(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.check_fixed()
        self._check_vector(value)
        residual, coefficients = value.clone(), []
        energy = value.new_zeros(())
        for step, change, reciprocal in reversed(self.frozen.history):
            projection = torch.dot(step, residual)
            coefficient = reciprocal*projection
            coefficients.append(coefficient)
            energy.add_((torch.sqrt(reciprocal)*projection).square())
            residual.sub_(coefficient*change)
        image, base_energy = self.base_and_energy(residual)
        energy.add_(base_energy)
        for (step, change, reciprocal), coefficient in zip(self.frozen.history, reversed(coefficients), strict=True):
            image.add_(step*(coefficient-reciprocal*torch.dot(change, image)))
        if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(energy)):
            raise FloatingPointError('unrepresentable Kronecker/raw-history solve operator')
        return image, energy
