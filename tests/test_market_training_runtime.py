from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quantdata.gromm2.joint_hedging import weighted_oce
from quantdata.gromm2.market_training_runtime import (
    IndependentPortfolioStep,
    _graph_owned_cublas_workspaces,
)
from quantdata.gromm2.portfolio_hedging import LIABILITIES
from quantdata.torch import BaseEnvironment, BaseModel


class TinyPolicy(BaseModel):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(env=env)
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1], device=self.device))
        self.translation = torch.nn.Parameter(torch.tensor(0.03, device=self.device))
        self.unused = torch.nn.Parameter(torch.tensor(0.7, device=self.device))

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> SimpleNamespace:
        gain = features @ self.weight + prices[:, 0]
        return SimpleNamespace(trading_gain=gain.to(torch.float64), penalty=gain.square())


@pytest.mark.parametrize("fail", [False, True])
def test_workspace_cache_is_cleared_outside_capture_on_entry_and_exit(monkeypatch, fail):
    events = []
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: events.append("clear"), raising=False)

    class Recording:
        def __enter__(self):
            events.append("begin")

        def __exit__(self, *args):
            events.append("end")

    def record():
        with _graph_owned_cublas_workspaces(), Recording():
            events.append("compute")
            if fail:
                raise RuntimeError("capture failure")

    if fail:
        with pytest.raises(RuntimeError, match="capture failure"):
            record()
    else:
        record()
    assert events == ["clear", "begin", "compute", "end", "clear"]


class TinyHedges(BaseModel):
    def __init__(self, device: str = "cpu") -> None:
        env = BaseEnvironment(device=device, precision=32, debug=False)
        super().__init__(env=env)
        self.config = SimpleNamespace(risk_aversion=2.0, penalty_weight=0.01)
        self.policies = torch.nn.ModuleDict({name: TinyPolicy(env) for name in LIABILITIES})


def batch(device: str = "cpu", count: int = 5) -> tuple[torch.Tensor, ...]:
    features = torch.arange(count * 2, dtype=torch.float32, device=device).reshape(count, 2) / 10
    prices = torch.full((count, 1), 0.02, dtype=torch.float64, device=device)
    payoff = torch.linspace(0.01, 0.2, count, dtype=torch.float64, device=device)
    return features, prices, payoff


def optimizers(model: TinyHedges) -> dict[str, torch.optim.Adam]:
    return {name: torch.optim.Adam(policy.parameters(), lr=0.003) for name, policy in model.policies.items()}


def legacy_step(model: TinyHedges, optimizer_map: dict, inputs: tuple, names: tuple = tuple(LIABILITIES)) -> None:
    features, prices, payoff = inputs
    for name in names:
        policy = model.policies[name]
        optimizer_map[name].zero_grad(set_to_none=True)
        output = policy(features, prices)
        loss = -weighted_oce(output.trading_gain - LIABILITIES[name] * payoff,
                             policy.translation, model.config.risk_aversion, torch.ones_like(payoff))
        loss += model.config.penalty_weight * output.penalty.mean()
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.2, error_if_nonfinite=True)
        optimizer_map[name].step()


def assert_models_equal(first: TinyHedges, second: TinyHedges, *, exact: bool = True) -> None:
    for name, expected in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], expected,
                                   rtol=0 if exact else 2e-5, atol=0 if exact else 2e-7)


def assert_optimizer_equal(first: dict, second: dict, *, exact: bool = True) -> None:
    for name in first:
        expected_states = first[name].state_dict()["state"]
        actual_states = second[name].state_dict()["state"]
        assert set(expected_states) == set(actual_states)
        for parameter, state in expected_states.items():
            for key, expected in state.items():
                actual = actual_states[parameter][key]
                torch.testing.assert_close(actual.cpu(), expected.cpu(),
                                           rtol=0 if exact else 2e-5, atol=0 if exact else 2e-7)


def test_atomic_cpu_matches_multiple_legacy_updates_and_resume() -> None:
    reference = TinyHedges()
    candidate = TinyHedges()
    reference_optimizers, candidate_optimizers = optimizers(reference), optimizers(candidate)
    runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    for index in range(24):
        names = tuple(LIABILITIES) if index < 6 else ("zero", "long")
        if index == 3:
            for optimizer_map in (reference_optimizers, candidate_optimizers):
                for optimizer in optimizer_map.values():
                    optimizer.param_groups[0]["lr"] *= 0.5
        legacy_step(reference, reference_optimizers, batch(count=5 if index % 2 else 3), names)
        runtime(*batch(count=5 if index % 2 else 3), names)
        assert_models_equal(reference, candidate)
        assert_optimizer_equal(reference_optimizers, candidate_optimizers)
        if index == 4:
            saved_model = copy.deepcopy(candidate.state_dict())
            saved_optimizers = {name: copy.deepcopy(value.state_dict()) for name, value in candidate_optimizers.items()}
            candidate = TinyHedges()
            candidate.load_state_dict(saved_model)
            candidate_optimizers = optimizers(candidate)
            for name in candidate_optimizers:
                candidate_optimizers[name].load_state_dict(saved_optimizers[name])
            runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    assert all(policy.unused.grad is None for policy in candidate.policies.values())


@pytest.mark.parametrize("name", tuple(LIABILITIES))
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
@pytest.mark.parametrize("location", ["loss", "gradient"])
def test_nonfinite_is_atomic_for_every_policy(name: str, nonfinite: float, location: str) -> None:
    model = TinyHedges()
    optimizer_map = optimizers(model)
    runtime = IndependentPortfolioStep(model, optimizer_map, gradient_clip=0.2, compile=False)
    runtime(*batch(), tuple(LIABILITIES))
    if location == "loss":
        with torch.no_grad():
            model.policies[name].translation.fill_(nonfinite)
    else:
        model.policies[name].weight.register_hook(lambda gradient: torch.full_like(gradient, nonfinite))
    before = copy.deepcopy(model.state_dict())
    before_optimizers = {key: copy.deepcopy(value.state_dict()) for key, value in optimizer_map.items()}
    with pytest.raises(FloatingPointError, match="no portfolio optimizer committed"):
        runtime(*batch(), tuple(LIABILITIES))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0, equal_nan=True)
    for key in before_optimizers:
        assert optimizer_map[key].state_dict()["param_groups"] == before_optimizers[key]["param_groups"]
        for parameter, state in before_optimizers[key]["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(optimizer_map[key].state_dict()["state"][parameter][field], value, rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_cuda_capture_updates_resume_lr_and_tail() -> None:
    reference, candidate = TinyHedges("cuda"), TinyHedges("cuda")
    reference_optimizers, candidate_optimizers = optimizers(reference), optimizers(candidate)
    runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    for index in range(24):
        names = tuple(LIABILITIES) if index < 6 else ("zero", "long")
        if index == 3:
            for optimizer_map in (reference_optimizers, candidate_optimizers):
                for optimizer in optimizer_map.values():
                    optimizer.param_groups[0]["lr"] *= 0.5
        inputs = batch("cuda", count=5 if index < 5 else 3)
        legacy_step(reference, reference_optimizers, inputs, names)
        runtime(*inputs, names)
        assert_models_equal(reference, candidate, exact=False)
        assert_optimizer_equal(reference_optimizers, candidate_optimizers, exact=False)
        if index == 4:
            model_state = copy.deepcopy(candidate.state_dict())
            optimizer_states = {name: copy.deepcopy(value.state_dict()) for name, value in candidate_optimizers.items()}
            candidate = TinyHedges("cuda")
            candidate.load_state_dict(model_state)
            candidate_optimizers = optimizers(candidate)
            for name, state in optimizer_states.items():
                candidate_optimizers[name].load_state_dict(state)
            runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    assert all(policy.unused.grad is None for policy in candidate.policies.values())
    before = copy.deepcopy(candidate.state_dict())
    inputs = batch("cuda", count=3)
    inputs[2].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        runtime(*inputs, ("zero", "long"))
    for name, value in before.items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_captured_checkpoint_resume_is_bitwise_identical(tmp_path: Path) -> None:
    continuous = TinyHedges("cuda")
    continuous_optimizers = optimizers(continuous)
    continuous_runtime = IndependentPortfolioStep(continuous, continuous_optimizers, gradient_clip=0.2, compile=False)
    names = tuple(LIABILITIES)
    for _ in range(5):
        continuous_runtime(*batch("cuda"), names)
    checkpoint = tmp_path / "captured-checkpoint.pt"
    torch.save({"model": continuous.state_dict(),
                "optimizers": {name: optimizer.state_dict() for name, optimizer in continuous_optimizers.items()},
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = TinyHedges("cuda")
    resumed.load_state_dict(saved["model"])
    resumed_optimizers = optimizers(resumed)
    for name, state in saved["optimizers"].items():
        resumed_optimizers[name].load_state_dict(state)
    resumed_runtime = IndependentPortfolioStep(resumed, resumed_optimizers, gradient_clip=0.2, compile=False)
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    for index in range(10):
        names = tuple(LIABILITIES) if index < 7 else ("zero", "long")
        if index == 3:
            for optimizer_map in (continuous_optimizers, resumed_optimizers):
                for optimizer in optimizer_map.values():
                    optimizer.param_groups[0]["lr"] *= 0.5
        inputs = batch("cuda", count=5 if index < 6 else 3)
        continuous_result = continuous_runtime(*inputs, names)
        resumed_result = resumed_runtime(*inputs, names)
        assert continuous_result == resumed_result
        assert_models_equal(continuous, resumed)
        assert_optimizer_equal(continuous_optimizers, resumed_optimizers)
        assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
        assert torch.equal(torch.cuda.get_rng_state().cpu(), saved["cuda_rng"])


def test_tail_geometry_reuses_only_matching_smaller_batches() -> None:
    model = TinyHedges()
    runtime = IndependentPortfolioStep(model, optimizers(model), gradient_clip=0.2, compile=False)
    names = tuple(LIABILITIES)

    def signature(count: int, *, active=names, rate=0.003):
        geometry = tuple((tuple(value.shape), value.dtype, tuple(value.stride())) for value in batch(count=count))
        return active, geometry, (rate,)

    assert not runtime._is_uncaptured_tail(signature(3))
    runtime._signature = signature(5)
    assert runtime._is_uncaptured_tail(signature(3))
    assert not runtime._is_uncaptured_tail(signature(5))
    assert not runtime._is_uncaptured_tail(signature(7))
    assert not runtime._is_uncaptured_tail(signature(3, active=("zero",)))
    assert not runtime._is_uncaptured_tail(signature(3, rate=0.001))
    runtime._signature = None
    runtime.captured_batch_size = 5
    assert runtime._is_uncaptured_tail(signature(3))
    assert not runtime._is_uncaptured_tail(signature(5))


def test_learning_rate_only_detection_rejects_geometry_and_other_optimizer_changes() -> None:
    model = TinyHedges()
    runtime = IndependentPortfolioStep(model, optimizers(model), gradient_clip=0.2, compile=False)
    names = tuple(LIABILITIES)
    geometry = tuple((tuple(value.shape), value.dtype, tuple(value.stride())) for value in batch())

    def signature(*, rate=.003, betas=(.9, .999), active=names, shape=geometry):
        return active, shape, tuple((name, ((rate, betas, 1.e-8, 0., False, False),)) for name in active)

    assert not runtime._only_learning_rates_changed(signature(rate=.001))
    runtime._signature = signature()
    assert not runtime._only_learning_rates_changed(signature())
    assert runtime._only_learning_rates_changed(signature(rate=.001))
    assert not runtime._only_learning_rates_changed(signature(rate=.001, betas=(.8, .999)))
    assert not runtime._only_learning_rates_changed(signature(rate=.001, active=("zero",)))
    assert not runtime._only_learning_rates_changed(signature(rate=.001, shape=geometry[:1]))


@pytest.mark.parametrize("fail", [False, True])
def test_optimizer_capture_restores_parameters_moments_rng_and_gradient_storage_on_cpu(monkeypatch, fail):
    model = TinyHedges()
    optimizer_map = optimizers(model)
    runtime = IndependentPortfolioStep(model, optimizer_map, gradient_clip=.2, compile=False)
    runtime(*batch(), tuple(LIABILITIES))
    before_model = copy.deepcopy(model.state_dict())
    before_optimizers = {name: copy.deepcopy(value.state_dict()) for name, value in optimizer_map.items()}
    gradients = tuple(value.grad for value in model.parameters())
    rng = torch.get_rng_state()
    fake_cuda_rng = torch.tensor([19], dtype=torch.uint8)

    class Stream:
        def wait_stream(self, other):
            pass

        def synchronize(self):
            pass

    @contextmanager
    def record(*args, **kwargs):
        yield

    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args: fake_cuda_rng.clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, *args: torch.testing.assert_close(value, fake_cuda_rng))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", record)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    original_commit = runtime._commit

    def commit(names):
        original_commit(names)
        torch.rand(4)
        if fail:
            raise RuntimeError("recording failed")

    monkeypatch.setattr(runtime, "_commit", commit)
    previous = runtime._optimizer_graph = object()
    if fail:
        with pytest.raises(RuntimeError, match="recording failed"):
            runtime._recapture_optimizer(tuple(LIABILITIES))
        assert runtime._optimizer_graph is previous
        assert runtime.optimizer_capture_count == 0
    else:
        runtime._recapture_optimizer(tuple(LIABILITIES))
        assert runtime._optimizer_graph is not previous
        assert runtime.optimizer_capture_count == 1
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_model[name], rtol=0, atol=0)
    for name, optimizer in optimizer_map.items():
        assert optimizer.state_dict()["param_groups"] == before_optimizers[name]["param_groups"]
        for parameter, state in optimizer.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, before_optimizers[name]["state"][parameter][key], rtol=0, atol=0)
    assert all(value.grad is gradient for value, gradient in zip(model.parameters(), gradients, strict=True))


def test_eager_tail_keeps_gradient_buffers_and_matches_the_same_cpu_objective() -> None:
    reference, candidate = TinyHedges(), TinyHedges()
    reference_optimizers, candidate_optimizers = optimizers(reference), optimizers(candidate)
    runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    runtime(*batch(count=5), tuple(LIABILITIES))
    legacy_step(reference, reference_optimizers, batch(count=5))
    pointers = tuple(parameter.grad for parameter in candidate.parameters())
    runtime.use_cuda_graphs = True
    status = runtime._compute_tail(batch(count=3), tuple(LIABILITIES))
    assert torch.isfinite(status).all()
    runtime._commit(tuple(LIABILITIES))
    legacy_step(reference, reference_optimizers, batch(count=3))
    assert_models_equal(reference, candidate)
    assert_optimizer_equal(reference_optimizers, candidate_optimizers)
    assert all(parameter.grad is previous for parameter, previous in zip(candidate.parameters(), pointers, strict=True))


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_cuda_tail_retains_full_graph_and_checks_every_observation() -> None:
    reference, candidate = TinyHedges("cuda"), TinyHedges("cuda")
    reference_optimizers, candidate_optimizers = optimizers(reference), optimizers(candidate)
    runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=0.2, compile=False)
    for index, count in enumerate((5, 3, 5, 3, 5, 3, 5)):
        if index == 4:
            for optimizer_map in (reference_optimizers, candidate_optimizers):
                for optimizer in optimizer_map.values():
                    optimizer.param_groups[0]["lr"] *= 0.5
        inputs = batch("cuda", count=count)
        legacy_step(reference, reference_optimizers, inputs)
        runtime(*inputs, tuple(LIABILITIES))
        assert runtime.capture_count == 1
        assert runtime.optimizer_capture_count == (1 if index < 4 else 2)
        assert_models_equal(reference, candidate, exact=False)
        assert_optimizer_equal(reference_optimizers, candidate_optimizers, exact=False)
    assert runtime.tail_step_count == 3
    before = copy.deepcopy(candidate.state_dict())
    before_optimizers = {name: copy.deepcopy(value.state_dict()) for name, value in candidate_optimizers.items()}
    bad = batch("cuda", count=3)
    bad[2][-1] = float("nan")
    with pytest.raises(FloatingPointError, match="no portfolio optimizer committed"):
        runtime(*bad, tuple(LIABILITIES))
    for name, value in before.items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    for name, saved in before_optimizers.items():
        for parameter, state in saved["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(candidate_optimizers[name].state_dict()["state"][parameter][field], value, rtol=0, atol=0)
    valid = batch("cuda", count=5)
    legacy_step(reference, reference_optimizers, valid)
    runtime(*valid, tuple(LIABILITIES))
    assert runtime.capture_count == 1
    assert runtime.optimizer_capture_count == 2
    assert_models_equal(reference, candidate, exact=False)
    assert_optimizer_equal(reference_optimizers, candidate_optimizers, exact=False)


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
@pytest.mark.parametrize("compile", [False, True])
def test_cuda_lr_changes_retain_compute_storage_across_population_evaluation(compile):
    reference, candidate = TinyHedges("cuda"), TinyHedges("cuda")
    reference_optimizers, candidate_optimizers = optimizers(reference), optimizers(candidate)
    runtime = IndependentPortfolioStep(candidate, candidate_optimizers, gradient_clip=.2, compile=compile)
    names = tuple(LIABILITIES)
    original_compute = original_status = original_inputs = original_gradients = None
    for index, count in enumerate((5, 5, 5, 3, 5, 5, 3, 5)):
        if index in (2, 5):
            for mapping in (reference_optimizers, candidate_optimizers):
                for optimizer in mapping.values():
                    optimizer.param_groups[0]["lr"] *= .5
        inputs = batch("cuda", count=count)
        legacy_step(reference, reference_optimizers, inputs)
        before_cpu_rng, before_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        runtime(*inputs, names)
        assert torch.equal(before_cpu_rng, torch.get_rng_state())
        assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state())
        assert runtime.capture_count == 1
        assert runtime.optimizer_capture_count == (1 if index < 2 else 2 if index < 5 else 3)
        if index == 0:
            original_compute, original_status, original_inputs = runtime._compute_graph, runtime._status, runtime._inputs
            original_gradients = tuple(value.grad for value in candidate.parameters())
        assert runtime._compute_graph is original_compute
        assert runtime._status is original_status
        assert runtime._inputs is original_inputs
        assert all(value.grad is gradient for value, gradient in zip(candidate.parameters(), original_gradients, strict=True))
        assert_models_equal(reference, candidate, exact=False)
        assert_optimizer_equal(reference_optimizers, candidate_optimizers, exact=False)
        candidate.eval()
        with torch.no_grad():
            for name in names:
                runtime.forwards[name](inputs[0], inputs[1])
    before = copy.deepcopy(candidate.state_dict())
    bad = batch("cuda")
    bad[2].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="no portfolio optimizer committed"):
        runtime(*bad, names)
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_cuda_resume_directly_at_tail_matches_uninterrupted_updates_exactly() -> None:
    continuous = TinyHedges("cuda")
    continuous_optimizers = optimizers(continuous)
    continuous_runtime = IndependentPortfolioStep(continuous, continuous_optimizers,
        gradient_clip=0.2, compile=False, captured_batch_size=5)
    continuous_runtime(*batch("cuda", count=5), tuple(LIABILITIES))
    saved_model = copy.deepcopy(continuous.state_dict())
    saved_optimizers = {name: copy.deepcopy(value.state_dict()) for name, value in continuous_optimizers.items()}
    resumed = TinyHedges("cuda")
    resumed.load_state_dict(saved_model)
    resumed_optimizers = optimizers(resumed)
    for name, state in saved_optimizers.items():
        resumed_optimizers[name].load_state_dict(state)
    resumed_runtime = IndependentPortfolioStep(resumed, resumed_optimizers,
        gradient_clip=0.2, compile=False, captured_batch_size=5)
    for count in (3, 5, 3, 5):
        continuous_runtime(*batch("cuda", count=count), tuple(LIABILITIES))
        resumed_runtime(*batch("cuda", count=count), tuple(LIABILITIES))
        assert_models_equal(continuous, resumed)
        assert_optimizer_equal(continuous_optimizers, resumed_optimizers)
    assert continuous_runtime.capture_count == resumed_runtime.capture_count == 1
    assert continuous_runtime.tail_step_count == resumed_runtime.tail_step_count == 2
