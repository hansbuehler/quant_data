"""Focused tests for the streamed affine-200 martingale pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import t as student_t

from quantdata.dysanos.flex.run_affine200_one_day_martingale import (
    EXPIRY_DAYS,
    NMS_GRID,
    _aggregate_audit,
    _antithetic_pair_means,
    _arguments,
    _audit_from_moments,
    _batched_block_jacobian,
    _bounded_spot_basis,
    _compensating_spot_shift,
    _conditional_metric_inverse_sqrt,
    _fit_state_batch_pathwise_adam,
    _identity,
    _load_warm_start,
    _newton_update,
    _nms_book,
    _OneDayDraw,
    _pathwise_moment_surrogate,
    _project_controls_,
    _truncated_minimum_norm_steps,
    _truncated_minimum_norm_targets,
    _validate_warm_start_states,
    _whitened_moment_loss,
)


class _SilentProgress:
    def write(self, stage: str, **values: object) -> None:
        del stage, values


class _ToyPathwiseProblem:
    """Small differentiable problem used to exercise Adam restart semantics."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, dtype=torch.float64)
        self.model.requires_grad_(False)
        self.current_states = torch.zeros((2, 2), dtype=torch.float64)
        self.spot_basis_size = 1
        self.state_size = 2
        self.num_states = 2
        self.num_option_moments = 2
        self.offset = torch.tensor(
            ((0.4, -0.2), (-0.3, 0.5)), dtype=torch.float64
        )

    def draw(
        self,
        num_paths: int,
        *,
        seeds: np.ndarray,
        method: str,
        pair_offset: int = 0,
    ) -> _OneDayDraw:
        assert seeds.shape == (self.num_states,)
        assert method == "scrambled-sobol"
        half = num_paths // 2
        base = torch.arange(
            pair_offset + 1,
            pair_offset + half + 1,
            dtype=torch.float64,
        )
        first = torch.stack((base, 0.5 * base), dim=1)
        antithetic = torch.cat((first, -first), dim=0)
        states = antithetic.unsqueeze(0).expand(self.num_states, -1, -1).clone()
        zeros = torch.zeros((self.num_states, num_paths), dtype=torch.float64)
        state_normals = torch.zeros_like(states)
        return _OneDayDraw(
            base_next_states=states,
            gross_spot=zeros,
            gaussian_score=zeros,
            state_normals=state_normals,
            independent_spot_normal=zeros,
        )

    def returns(
        self,
        controls: torch.Tensor,
        base_next_states: torch.Tensor,
        gross_spot: torch.Tensor,
        gaussian_score: torch.Tensor,
    ) -> torch.Tensor:
        del gross_spot, gaussian_score
        shift = torch.stack((controls[:, 0], 2.0 * controls[:, 1]), dim=1)
        scale = 1.0 + 0.25 * controls
        options = (
            self.offset[:, None]
            + scale[:, None] * base_next_states
            + shift[:, None]
        )
        spot = torch.zeros(
            (*options.shape[:2], 1), dtype=options.dtype, device=options.device
        )
        return torch.cat((spot, options), dim=2)


def test_antithetic_pair_means_use_first_and_second_halves() -> None:
    values = torch.tensor([[[1.0], [2.0], [-1.0], [4.0]]])

    paired = _antithetic_pair_means(values)

    torch.testing.assert_close(paired, torch.tensor([[[0.0], [3.0]]]))


def test_block_jacobian_collects_only_each_states_derivatives() -> None:
    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    outputs = torch.stack(
        (
            inputs[:, 0].square() + 3.0 * inputs[:, 1],
            inputs[:, 0] * inputs[:, 1],
        ),
        dim=1,
    )

    jacobian = _batched_block_jacobian(outputs, inputs)

    expected = torch.tensor(
        [[[2.0, 3.0], [2.0, 1.0]], [[6.0, 3.0], [4.0, 3.0]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize(
    ("control_dtype", "tolerance"),
    ((torch.float64, 2.0e-14), (torch.float32, 2.0e-6)),
)
def test_streamed_pathwise_surrogates_equal_full_moment_loss_gradient(
    control_dtype: torch.dtype,
    tolerance: float,
) -> None:
    generator = torch.Generator().manual_seed(712)
    features = torch.randn(
        (2, 8, 3, 4), generator=generator, dtype=control_dtype
    )
    offsets = torch.randn((2, 8, 3), generator=generator, dtype=control_dtype)
    scale = torch.tensor(
        ((0.7, 1.2, 2.0), (1.1, 0.9, 1.5)), dtype=torch.float64
    )
    initial = torch.randn((2, 4), generator=generator, dtype=control_dtype)

    direct_controls = initial.clone().requires_grad_(True)
    direct_linear = torch.einsum(
        "bpmc,bc->bpm", features, direct_controls
    )
    direct_returns = (
        offsets + direct_linear + 0.1 * direct_linear.square()
    ).double()
    direct_mean = torch.mean(direct_returns, dim=1)
    direct_loss = torch.sum(_whitened_moment_loss(direct_mean, scale))
    direct_gradient = torch.autograd.grad(direct_loss, direct_controls)[0]

    streamed_controls = initial.clone().requires_grad_(True)
    streamed_linear = torch.einsum(
        "bpmc,bc->bpm", features, streamed_controls
    )
    streamed_returns = (
        offsets + streamed_linear + 0.1 * streamed_linear.square()
    ).double()
    detached_mean = torch.mean(streamed_returns, dim=1).detach()
    for path_slice in (slice(0, 3), slice(3, None)):
        batch_linear = torch.einsum(
            "bpmc,bc->bpm", features[:, path_slice], streamed_controls
        )
        batch = (
            offsets[:, path_slice]
            + batch_linear
            + 0.1 * batch_linear.square()
        ).double()
        surrogate = _pathwise_moment_surrogate(
            torch.sum(batch, dim=1),
            total_paths=streamed_returns.shape[1],
            detached_full_mean=detached_mean,
            scale=scale,
        )
        surrogate.backward()

    assert streamed_controls.grad is not None
    torch.testing.assert_close(
        streamed_controls.grad,
        direct_gradient,
        atol=tolerance,
        rtol=tolerance,
    )


def test_control_projection_is_per_state_and_preserves_direction() -> None:
    controls = torch.tensor(
        ((0.3, 0.4), (3.0, 4.0), (0.0, 0.0)), dtype=torch.float64
    )

    norms = _project_controls_(controls, maximum_norm=2.0)

    torch.testing.assert_close(
        controls,
        torch.tensor(((0.3, 0.4), (1.2, 1.6), (0.0, 0.0)), dtype=torch.float64),
    )
    torch.testing.assert_close(
        norms, torch.tensor((0.5, 2.0, 0.0), dtype=torch.float64)
    )


def test_pathwise_adam_resume_reproduces_uninterrupted_controls() -> None:
    common = {
        "problem": _ToyPathwiseProblem(),
        "state_indices": np.asarray((7, 11), dtype=np.int64),
        "fit_paths": 8,
        "fit_path_batch_size": 4,
        "fit_seed": 91,
        "adam_epochs": 3,
        "adam_learning_rate": 0.05,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "adam_control_l2": 0.01,
        "adam_absolute_sd_floor": 1.0e-4,
        "adam_relative_sd_floor": 1.0e-2,
        "maximum_control_norm": 10.0,
        "progress": _SilentProgress(),
        "initial_controls": None,
    }
    uninterrupted = _fit_state_batch_pathwise_adam(
        **common,
        resume_state=None,
        checkpoint_callback=lambda payload: None,
    )
    saved: dict[str, object] = {}

    class _Interrupted(RuntimeError):
        pass

    def interrupt_after_first_epoch(payload: dict[str, object]) -> None:
        saved.clear()
        saved.update(payload)
        if payload["completed_epochs"] == 1:
            raise _Interrupted

    with pytest.raises(_Interrupted):
        _fit_state_batch_pathwise_adam(
            **{**common, "problem": _ToyPathwiseProblem()},
            resume_state=None,
            checkpoint_callback=interrupt_after_first_epoch,
        )
    resumed = _fit_state_batch_pathwise_adam(
        **{**common, "problem": _ToyPathwiseProblem()},
        resume_state=saved,
        checkpoint_callback=lambda payload: None,
    )

    np.testing.assert_allclose(resumed["shifts"], uninterrupted["shifts"], atol=0.0)
    np.testing.assert_allclose(
        resumed["fit_objective"], uninterrupted["fit_objective"], atol=0.0
    )


def test_pathwise_warm_start_rebuilds_p_whitening_and_resets_adam() -> None:
    common = {
        "state_indices": np.asarray((7, 11), dtype=np.int64),
        "fit_paths": 8,
        "fit_path_batch_size": 4,
        "fit_seed": 91,
        "adam_epochs": 1,
        "adam_learning_rate": 0.05,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "adam_control_l2": 0.0,
        "adam_absolute_sd_floor": 1.0e-4,
        "adam_relative_sd_floor": 1.0e-2,
        "maximum_control_norm": 10.0,
        "progress": _SilentProgress(),
        "resume_state": None,
    }
    cold = _fit_state_batch_pathwise_adam(
        **{**common, "problem": _ToyPathwiseProblem()},
        initial_controls=None,
        checkpoint_callback=lambda payload: None,
    )
    warm_controls = torch.tensor(((0.8, -0.4), (-0.6, 0.5)), dtype=torch.float64)
    payloads: list[dict[str, object]] = []
    warm = _fit_state_batch_pathwise_adam(
        **{**common, "problem": _ToyPathwiseProblem()},
        initial_controls=warm_controls,
        checkpoint_callback=payloads.append,
    )

    np.testing.assert_allclose(
        warm["base_mean"], _ToyPathwiseProblem().offset.numpy(), atol=0.0
    )
    np.testing.assert_allclose(
        warm["whitening_population_sd"],
        cold["whitening_population_sd"],
        atol=0.0,
    )
    assert not np.allclose(
        warm["iteration_rms"][:, 0],
        np.sqrt(np.mean(np.square(warm["base_mean"]), axis=1)),
    )
    optimizer_state = payloads[0]["optimizer_state_dict"]
    assert isinstance(optimizer_state, dict)
    state_values = list(optimizer_state["state"].values())
    assert len(state_values) == 1
    assert float(state_values[0]["step"]) == pytest.approx(1.0)


def test_truncated_minimum_norm_step_solves_independent_blocks() -> None:
    residual = np.asarray(((1.0, -2.0), (-3.0, 4.0)))
    jacobian = np.asarray(
        (
            ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
            ((2.0, 0.0, 0.0), (0.0, 4.0, 0.0)),
        )
    )

    result = _truncated_minimum_norm_steps(
        residual,
        jacobian,
        relative_cutoff=1.0e-12,
        relative_damping=0.0,
        maximum_norm=100.0,
    )

    np.testing.assert_allclose(
        np.einsum("bmc,bc->bm", jacobian, result.step) + residual,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(result.effective_rank, (2, 2))


def test_sqp_target_removes_accumulated_null_space_control() -> None:
    residual = np.asarray(((-1.0,),))
    jacobian = np.asarray((((1.0, 0.0, 0.0),),))
    current = np.asarray(((1.0, 5.0, -4.0),))

    result = _truncated_minimum_norm_targets(
        residual,
        jacobian,
        current,
        relative_cutoff=1.0e-12,
        relative_damping=0.0,
        maximum_step_norm=100.0,
        maximum_control_norm=100.0,
    )
    target = current + result.step

    np.testing.assert_allclose(target, ((2.0, 0.0, 0.0),), atol=1.0e-12)
    np.testing.assert_allclose(
        residual + np.einsum("bmc,bc->bm", jacobian, result.step),
        0.0,
        atol=1.0e-12,
    )


def test_sqp_target_respects_explicit_conditional_kl_control_cap() -> None:
    result = _truncated_minimum_norm_targets(
        np.asarray(((-100.0,),)),
        np.asarray((((1.0, 0.0),),)),
        np.zeros((1, 2)),
        relative_cutoff=1.0e-12,
        relative_damping=0.0,
        maximum_step_norm=100.0,
        maximum_control_norm=3.0,
    )

    assert np.linalg.norm(result.step[0]) == pytest.approx(3.0)


def test_newton_mode_dispatch_distinguishes_final_kl_and_incremental_steps() -> None:
    residual = np.asarray(((-1.0,),))
    jacobian = np.asarray((((1.0, 0.0, 0.0),),))
    current = np.asarray(((1.0, 5.0, -4.0),))
    common = {
        "relative_cutoff": 1.0e-12,
        "relative_damping": 0.0,
        "maximum_step_norm": 100.0,
        "maximum_control_norm": 100.0,
    }

    minimum_final = _newton_update(
        residual, jacobian, current, mode="minimum-final-kl", **common
    )
    incremental = _newton_update(
        residual, jacobian, current, mode="incremental", **common
    )

    np.testing.assert_allclose(
        current + minimum_final.step, ((2.0, 0.0, 0.0),), atol=1.0e-12
    )
    np.testing.assert_allclose(
        current + incremental.step, ((2.0, 5.0, -4.0),), atol=1.0e-12
    )


def test_incremental_newton_projects_final_control_into_same_cap() -> None:
    result = _newton_update(
        np.asarray(((-100.0,),)),
        np.asarray((((1.0, 0.0),),)),
        np.zeros((1, 2)),
        mode="incremental",
        relative_cutoff=1.0e-12,
        relative_damping=0.0,
        maximum_step_norm=100.0,
        maximum_control_norm=3.0,
    )

    assert np.linalg.norm(result.step[0]) == pytest.approx(3.0)


def test_cli_defaults_to_minimum_final_kl_and_identity_separates_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "martingale-pilot",
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--target-history",
            str(tmp_path / "history.pt"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    arguments = _arguments()

    assert arguments.newton_mode == "minimum-final-kl"
    assert arguments.fit_backend == "score-newton"
    assert arguments.warm_start_checkpoint is None
    minimum_identity = _identity(arguments)
    arguments.newton_mode = "incremental"
    incremental_identity = _identity(arguments)
    assert minimum_identity["newton_mode"] == "minimum-final-kl"
    assert incremental_identity["newton_mode"] == "incremental"
    assert minimum_identity != incremental_identity

    arguments.state_batch_size += 1
    changed_batch_identity = _identity(arguments)
    assert changed_batch_identity != incremental_identity

    arguments.fit_backend = "pathwise-adam"
    pathwise_identity = _identity(arguments)
    arguments.adam_learning_rate *= 2.0
    changed_pathwise_identity = _identity(arguments)
    assert pathwise_identity["fit_backend"] == "pathwise-adam"
    assert pathwise_identity != incremental_identity
    assert changed_pathwise_identity != pathwise_identity


def test_completed_pathwise_checkpoint_warm_start_validation_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "martingale-pilot",
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--target-history",
            str(tmp_path / "history.pt"),
            "--output",
            str(tmp_path / "new-output"),
            "--fit-backend",
            "pathwise-adam",
            "--num-states",
            "1",
            "--state-source",
            "history",
        ],
    )
    arguments = _arguments()
    source_identity = _identity(arguments)
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    source_path = source_directory / "checkpoint.pt"
    initial_states = torch.zeros((1, 200), dtype=torch.float64)
    source_indices = torch.tensor((17,), dtype=torch.int64)
    controls = torch.linspace(-0.2, 0.2, 600).reshape(1, 600)
    source_payload = {
        "schema": "dysanos-affine200-one-day-martingale-v1",
        "identity": source_identity,
        "fit_complete": torch.ones(1, dtype=torch.bool),
        "validation_complete": torch.ones(1, dtype=torch.bool),
        "active_pathwise_adam": None,
        "kl_control_coefficients": controls,
        "initial_states": initial_states,
        "source_indices": source_indices,
        "fit_iteration_rms": torch.zeros((1, 101), dtype=torch.float64),
    }
    torch.save(source_payload, source_path)
    labels = list(_nms_book(EXPIRY_DAYS, normalized_moneyness=NMS_GRID).labels)
    (source_directory / "report.json").write_text(
        json.dumps({"identity": source_identity, "book_labels": labels})
    )
    arguments.warm_start_checkpoint = source_path

    warm_start = _load_warm_start(arguments)

    assert warm_start is not None
    torch.testing.assert_close(warm_start.controls, controls)
    _validate_warm_start_states(
        warm_start,
        initial_states=initial_states,
        source_indices=source_indices.numpy(),
    )
    with pytest.raises(ValueError, match="different initial states"):
        _validate_warm_start_states(
            warm_start,
            initial_states=initial_states + 1.0,
            source_indices=source_indices.numpy(),
        )
    warm_identity = _identity(
        arguments, warm_start_provenance=warm_start.provenance
    )
    assert "warm_start" not in source_identity
    assert warm_identity["warm_start"] == {
        name: warm_start.provenance[name]
        for name in (
            "checkpoint",
            "source_identity_sha256",
            "source_controls_sha256",
            "source_initial_states_sha256",
        )
    }
    assert warm_start.provenance["optimizer_state_reused"] is False
    assert warm_start.provenance["whitening_reused"] is False

    arguments.maximum_control_norm = 0.1
    with pytest.raises(ValueError, match="exceed the configured control norm"):
        _load_warm_start(arguments)
    arguments.maximum_control_norm = 10.0

    invalid_backend = {
        **source_payload,
        "identity": {**source_identity, "fit_backend": "score-newton"},
    }
    torch.save(invalid_backend, source_path)
    with pytest.raises(ValueError, match="not fitted with pathwise Adam"):
        _load_warm_start(arguments)

    torch.save({**source_payload, "schema": "obsolete"}, source_path)
    with pytest.raises(ValueError, match="schema is incompatible"):
        _load_warm_start(arguments)

    torch.save(source_payload, source_path)
    (source_directory / "report.json").write_text(
        json.dumps({"identity": source_identity, "book_labels": labels[:-1]})
    )
    with pytest.raises(ValueError, match="incompatible option book"):
        _load_warm_start(arguments)


def test_compensating_spot_shift_keeps_gaussian_score_centered() -> None:
    shifts = np.asarray(((1.0, -2.0), (0.5, 0.25)))
    loading = np.asarray((0.3, -0.4))
    orthogonal_std = 0.8

    compensation = _compensating_spot_shift(
        shifts,
        copula_loading=loading,
        orthogonal_std=orthogonal_std,
    )
    np.testing.assert_allclose(
        shifts @ loading + orthogonal_std * compensation,
        0.0,
        atol=1.0e-15,
    )


def test_conditional_metric_coordinates_have_euclidean_gaussian_kl() -> None:
    controls = np.asarray(((1.0, -2.0), (0.5, 0.25)))
    loading = np.asarray((0.3, -0.4))
    orthogonal_std = 0.8
    shifts = controls @ _conditional_metric_inverse_sqrt(
        loading, orthogonal_std=orthogonal_std
    )
    compensation = _compensating_spot_shift(
        shifts,
        copula_loading=loading,
        orthogonal_std=orthogonal_std,
    )

    np.testing.assert_allclose(
        np.sum(shifts**2, axis=1) + compensation**2,
        np.sum(controls**2, axis=1),
        atol=1.0e-14,
    )


def test_bounded_spot_basis_is_orthonormal_under_gaussian_quadrature() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    scores = torch.as_tensor(np.sqrt(2.0) * nodes, dtype=torch.float64)
    basis = _bounded_spot_basis(scores, size=3).numpy()
    gaussian_weights = weights / np.sqrt(np.pi)

    gram = np.einsum("n,ni,nj->ij", gaussian_weights, basis, basis)

    np.testing.assert_allclose(gram, np.eye(3), atol=2.0e-9)


def test_conditional_score_jacobian_matches_pathwise_derivative() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(50)
    z_nodes = np.sqrt(2.0) * nodes
    gaussian_weights = weights / np.sqrt(np.pi)
    z, independent = np.meshgrid(z_nodes, z_nodes, indexing="ij")
    joint_weights = np.outer(gaussian_weights, gaussian_weights)
    z_tensor = torch.as_tensor(z.reshape(-1), dtype=torch.float64)
    independent_tensor = torch.as_tensor(
        independent.reshape(-1), dtype=torch.float64
    )
    weights_tensor = torch.as_tensor(joint_weights.reshape(-1), dtype=torch.float64)
    loading = 0.4
    orthogonal_std = 0.8
    total_std = np.sqrt(loading**2 + orthogonal_std**2)
    score = (
        loading * z_tensor + orthogonal_std * independent_tensor
    ) / total_std
    basis = _bounded_spot_basis(score, size=3)
    coefficients = torch.tensor(
        (0.2, -0.1, 0.05), dtype=torch.float64, requires_grad=True
    )
    shifted = z_tensor + basis @ coefficients
    payoff = torch.exp(0.2 * shifted) + 0.1 * shifted.square()
    expectation = torch.sum(weights_tensor * payoff)

    pathwise = torch.autograd.grad(expectation, coefficients)[0]
    conditional_score = z_tensor - loading * independent_tensor / orthogonal_std
    score_derivative = torch.sum(
        weights_tensor[:, None]
        * (payoff - expectation).detach()[:, None]
        * basis
        * conditional_score[:, None],
        dim=0,
    )

    torch.testing.assert_close(pathwise, score_derivative, atol=2.0e-11, rtol=2.0e-11)


def test_streamed_audit_uses_antithetic_pairs_as_observations() -> None:
    observations = np.asarray(
        [
            [
                [1.0, 2.0],
                [-1.0, -2.0],
                [2.0, -1.0],
                [-2.0, 1.0],
            ]
        ]
    )
    sums = observations.sum(axis=1)
    cross = np.einsum("bpi,bpj->bij", observations, observations)

    audit = _audit_from_moments(
        count=4,
        sums=sums,
        cross_products=cross,
        correlation_eigenvalue_cutoff=0.01,
    )

    np.testing.assert_allclose(audit.mean, 0.0, atol=0.0)
    np.testing.assert_allclose(audit.z, 0.0, atol=0.0)
    assert audit.wald_statistic[0] == pytest.approx(0.0)
    assert audit.wald_p_value[0] == pytest.approx(1.0)


def test_zero_standard_error_z_convention_is_exact() -> None:
    audit = _audit_from_moments(
        count=4,
        sums=np.asarray(((0.0, 4.0),)),
        cross_products=np.asarray((((0.0, 0.0), (0.0, 4.0)),)),
        correlation_eigenvalue_cutoff=0.01,
    )

    assert audit.z[0, 0] == 0.0
    assert audit.z[0, 1] == np.inf


def test_global_coordinate_screen_uses_student_t_bonferroni_threshold() -> None:
    checkpoint: dict[str, object] = {
        "identity": {"validation_paths_per_state": 10_000},
        "base_mean": torch.zeros((2, 3), dtype=torch.float64),
        "base_standard_error": torch.ones((2, 3), dtype=torch.float64),
        "base_z": torch.zeros((2, 3), dtype=torch.float64),
        "base_wald_p_value": torch.ones(2, dtype=torch.float64),
    }

    result = _aggregate_audit(checkpoint, "base")

    screen = result["primary_global_coordinate_screen"]
    assert isinstance(screen, dict)
    assert screen["threshold"] == pytest.approx(
        student_t.ppf(1.0 - 0.05 / (2.0 * 6), 4_999)
    )
    assert screen["pass"] is True
    secondary = result["secondary_covariance_aware_screen"]
    assert isinstance(secondary, dict)
    assert str(secondary["role"]).startswith("secondary")
