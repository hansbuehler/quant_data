"""Atomic independent-policy updates with one safety decision per minibatch."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch
from cdxcore.err import verify_inp

from quantdata.gromm2.joint_hedging import weighted_oce
from quantdata.gromm2.market_batching import GPU_CAPTURE_LOCK


@contextmanager
def _graph_owned_cublas_workspaces() -> Iterator[None]:
    """Allocate GEMM workspaces inside the recorded graph's private pool.

    Warmup workspaces belong to the ordinary allocator and can be freed by
    compiler-managed evaluation captures. Clear their process-wide cache
    outside capture, including on exit, while holding ``GPU_CAPTURE_LOCK``.
    Live manual graphs must all use this ownership convention.
    """
    torch._C._cuda_clearCublasWorkspaces()
    try:
        yield
    finally:
        torch._C._cuda_clearCublasWorkspaces()


class IndependentPortfolioStep:
    """Compute all active gradients before committing any independent Adam update.

    CUDA uses a captured computation graph, one packed host safety check, and a
    separate captured optimizer graph. Capture warmup is rolled back in-place.
    Optimizer step tensors become device-resident in the CUDA graph path; their
    serialized values and the model's parameter names remain unchanged.
    """

    def __init__(self, model: torch.nn.Module, optimizers: Mapping[str, torch.optim.Optimizer], *,
                 gradient_clip: float, compile: bool = True, use_cuda_graphs: bool = True,
                 captured_batch_size: int | None = None) -> None:
        from quantdata.gromm2.portfolio_hedging import LIABILITIES

        verify_inp(math.isfinite(gradient_clip) and gradient_clip > 0,
                   lambda: f"gradient_clip must be finite and positive, received {gradient_clip}")
        verify_inp(captured_batch_size is None or (isinstance(captured_batch_size, int) and captured_batch_size > 0),
                   lambda: "captured_batch_size must be a positive integer or None")
        self.model = model
        self.liabilities = LIABILITIES
        self.optimizers = dict(optimizers)
        self.gradient_clip = gradient_clip
        self.captured_batch_size = captured_batch_size
        self.device = next(model.parameters()).device
        self.use_cuda_graphs = use_cuda_graphs and self.device.type == "cuda"
        if self.use_cuda_graphs and not all(isinstance(value, torch.optim.Adam)
                                           for value in self.optimizers.values()):
            raise TypeError("captured portfolio updates require independent Adam optimizers")
        self.forwards = {name: torch.compile(policy, options={"triton.cudagraphs": False}) if compile else policy
                         for name, policy in model.policies.items()}
        self._signature = None
        self._compute_graph = None
        self._optimizer_graph = None
        self._inputs = None
        self._status = None
        self.capture_count = 0
        self.optimizer_capture_count = 0
        self.tail_step_count = 0
        self.capture_seconds = 0.0
        self.optimizer_recapture_seconds = 0.0
        if self.device.type == "cpu":
            for optimizer in self.optimizers.values():
                for group in optimizer.param_groups:
                    group["capturable"] = False
                for state in optimizer.state.values():
                    if "step" in state:
                        state["step"] = state["step"].cpu()

    def _compute(self, features: torch.Tensor, prices: torch.Tensor, payoff: torch.Tensor,
                 active_names: tuple[str, ...]) -> torch.Tensor:
        records = []
        inputs_finite = torch.isfinite(features).all() & torch.isfinite(prices).all() & torch.isfinite(payoff).all()
        for name in active_names:
            policy = self.model.policies[name]
            self.optimizers[name].zero_grad(set_to_none=not self.use_cuda_graphs)
            output = self.forwards[name](features, prices)
            loss = -weighted_oce(output.trading_gain - self.liabilities[name] * payoff,
                                 policy.translation, self.model.config.risk_aversion, torch.ones_like(payoff))
            loss = loss + self.model.config.penalty_weight * output.penalty.mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip,
                                                 error_if_nonfinite=False)
            records.extend(((torch.isfinite(loss) & inputs_finite).to(torch.float64), torch.isfinite(norm).to(torch.float64),
                            loss.detach().to(torch.float64), norm.detach().to(torch.float64)))
        return torch.stack(records)

    def _commit(self, active_names: tuple[str, ...]) -> None:
        for name in active_names:
            self.optimizers[name].step()

    def _capture(self, inputs: tuple[torch.Tensor, ...], active_names: tuple[str, ...]) -> None:
        started = time.monotonic()
        with GPU_CAPTURE_LOCK:
            self._capture_unlocked(inputs, active_names)
        self.capture_seconds += time.monotonic() - started

    def _is_uncaptured_tail(self, signature: tuple) -> bool:
        if self._signature is None:
            counts = tuple(item[0][0] for item in signature[1])
            return (self.captured_batch_size is not None and len(set(counts)) == 1
                and 0 < counts[0] < self.captured_batch_size)
        previous_names, previous_geometry, previous_hyperparameters = self._signature
        names, geometry, hyperparameters = signature
        return (names == previous_names and hyperparameters == previous_hyperparameters
            and all(0 < shape[0] < previous_shape[0] and shape[1:] == previous_shape[1:]
                and dtype == previous_dtype and stride[1:] == previous_stride[1:]
                for (shape, dtype, stride), (previous_shape, previous_dtype, previous_stride)
                    in zip(geometry, previous_geometry, strict=True)))

    def _only_learning_rates_changed(self, signature: tuple) -> bool:
        if self._signature is None or signature[:2] != self._signature[:2]:
            return False
        previous_groups, current_groups = self._signature[2], signature[2]
        if len(previous_groups) != len(current_groups):
            return False
        changed = False
        for (previous_name, previous), (current_name, current) in zip(previous_groups, current_groups, strict=True):
            if previous_name != current_name or len(previous) != len(current):
                return False
            for previous_group, current_group in zip(previous, current, strict=True):
                if previous_group[1:] != current_group[1:]:
                    return False
                changed = changed or previous_group[0] != current_group[0]
        return changed

    def _recapture_optimizer(self, active_names: tuple[str, ...]) -> None:
        """Refresh float learning rates without rebuilding forward/backward graphs.

        Recording preserves parameters, gradients, Adam moments and step counters
        in place. The original graphs stay alive until replacement succeeds.
        """
        started = time.monotonic()
        with GPU_CAPTURE_LOCK:
            parameters = tuple(self.model.parameters())
            parameter_copies = tuple(value.detach().clone() for value in parameters)
            state_copies = tuple((value, value.clone()) for name in active_names
                for state in self.optimizers[name].state.values() for value in state.values())
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.device)
            current = torch.cuda.current_stream(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(current)
            replacement = torch.cuda.CUDAGraph()
            try:
                with _graph_owned_cublas_workspaces(), torch.cuda.graph(replacement, stream=stream):
                    self._commit(active_names)
                stream.synchronize()
            finally:
                stream.synchronize()
                with torch.no_grad():
                    for parameter, saved in zip(parameters, parameter_copies, strict=True):
                        parameter.copy_(saved)
                    for value, saved in state_copies:
                        value.copy_(saved)
                current.synchronize()
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, self.device)
            self._optimizer_graph = replacement
            self.optimizer_capture_count += 1
        self.optimizer_recapture_seconds += time.monotonic() - started

    def _compute_tail(self, inputs: tuple[torch.Tensor, ...], names: tuple[str, ...]) -> torch.Tensor:
        """Retain the full-batch graph and its gradient buffers for a rare tail.

        The same objective and Adam update process every actual observation.
        Uncompiled policy forwards avoid shape-specific compilation or capture
        for one small batch per epoch. Captured optimizer graphs require their
        original gradient buffers to remain attached and alive.
        """
        parameters = tuple(self.model.parameters())
        gradients = tuple(parameter.grad for parameter in parameters)
        forwards = self.forwards
        try:
            self.forwards = dict(self.model.policies.items())
            status = self._compute(*inputs, names)
        finally:
            self.forwards = forwards
        if self._signature is not None and any(parameter.grad is not gradient
                for parameter, gradient in zip(parameters, gradients, strict=True)):
            raise RuntimeError("uncaptured tail changed a captured gradient buffer; no portfolio optimizer committed")
        self.tail_step_count += 1
        return status

    def _capture_unlocked(self, inputs: tuple[torch.Tensor, ...], active_names: tuple[str, ...]) -> None:
        self._signature = None
        self._compute_graph = None
        self._optimizer_graph = None
        self._inputs = tuple(value.clone() for value in inputs)
        parameters = tuple(self.model.parameters())
        parameter_copies = tuple(value.detach().clone() for value in parameters)
        state_copies = {name: copy.deepcopy(self.optimizers[name].state_dict()) for name in active_names}
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for name in active_names:
                    optimizer = self.optimizers[name]
                    for group in optimizer.param_groups:
                        if group.get("differentiable", False):
                            raise ValueError("captured optimizer must not be differentiable")
                        group["capturable"] = True
                    for state in optimizer.state.values():
                        if "step" in state:
                            state["step"] = state["step"].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                for _ in range(3):
                    self._compute(*self._inputs, active_names)
                self._commit(active_names)
            stream.synchronize()
            with torch.no_grad():
                for parameter, saved in zip(parameters, parameter_copies):
                    parameter.copy_(saved)
                for name in active_names:
                    optimizer = self.optimizers[name]
                    old_states = state_copies[name]["state"]
                    parameter_ids = [value for group in state_copies[name]["param_groups"] for value in group["params"]]
                    optimizer_parameters = [value for group in optimizer.param_groups for value in group["params"]]
                    for parameter_id, parameter in zip(parameter_ids, optimizer_parameters):
                        for key, value in optimizer.state.get(parameter, {}).items():
                            if parameter_id in old_states:
                                value.copy_(old_states[parameter_id][key])
                            else:
                                value.zero_()
            stream.wait_stream(torch.cuda.current_stream(self.device))
            self._compute_graph = torch.cuda.CUDAGraph()
            with _graph_owned_cublas_workspaces(), torch.cuda.graph(self._compute_graph, stream=stream):
                self._status = self._compute(*self._inputs, active_names)
            optimizer_snapshots = {name: {parameter: {key: value.clone() for key, value in state.items()}
                                          for parameter, state in self.optimizers[name].state.items()}
                                   for name in active_names}
            self._optimizer_graph = torch.cuda.CUDAGraph()
            with _graph_owned_cublas_workspaces(), torch.cuda.graph(self._optimizer_graph, stream=stream):
                self._commit(active_names)
            stream.synchronize()
            with torch.no_grad():
                for parameter, saved in zip(parameters, parameter_copies):
                    parameter.copy_(saved)
                for name, states in optimizer_snapshots.items():
                    for parameter, state in states.items():
                        for key, saved in state.items():
                            self.optimizers[name].state[parameter][key].copy_(saved)
            self.capture_count += 1
            self.optimizer_capture_count += 1
        except BaseException:
            stream.synchronize()
            with torch.no_grad():
                for parameter, saved in zip(parameters, parameter_copies):
                    parameter.copy_(saved)
            for name, state in state_copies.items():
                self.optimizers[name].load_state_dict(state)
            self._compute_graph = None
            self._optimizer_graph = None
            raise
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)

    def __call__(self, features: torch.Tensor, prices: torch.Tensor, payoff: torch.Tensor,
                 active_names: Sequence[str]) -> dict[str, dict[str, float]]:
        """Reject non-finite computation before committing any policy update.

        Unexpected failures during optimizer replay require the caller to retain
        the last complete checkpoint rather than save partially committed state.
        """
        names = tuple(active_names)
        if not names:
            return {}
        if len(set(names)) != len(names) or any(name not in self.optimizers for name in names):
            raise ValueError("active_names must identify distinct existing policy optimizers")
        self.model.train()
        inputs = (features, prices, payoff)
        if any(value.device != self.device for value in inputs):
            raise ValueError("training inputs must already be on the model device")
        replay_graph = self.use_cuda_graphs
        if self.use_cuda_graphs:
            geometry = tuple((tuple(value.shape), value.dtype, tuple(value.stride())) for value in inputs)
            hyperparameters = tuple((name, tuple((float(group["lr"]), group["betas"], group["eps"],
                                                  group["weight_decay"], group["amsgrad"], group["maximize"])
                                                 for group in self.optimizers[name].param_groups)) for name in names)
            signature = (names, geometry, hyperparameters)
            if self._is_uncaptured_tail(signature):
                replay_graph = False
                status = self._compute_tail(inputs, names)
            elif self._only_learning_rates_changed(signature):
                self._recapture_optimizer(names)
                self._signature = signature
            elif signature != self._signature:
                self._capture(inputs, names)
                self._signature = signature
            if replay_graph:
                for target, source in zip(self._inputs, inputs):
                    target.copy_(source)
                self._compute_graph.replay()
                status = self._status
        else:
            status = self._compute(*inputs, names)
        host_status = status.detach().cpu().tolist()
        diagnostics = {}
        for index, name in enumerate(names):
            finite_loss, finite_norm, loss, norm = host_status[4 * index:4 * index + 4]
            if finite_loss != 1.0 or finite_norm != 1.0:
                raise FloatingPointError(f"non-finite input, {name} loss or gradient norm: loss={loss}, "
                    f"gradient_norm={norm}, finite_input_and_loss={finite_loss}, finite_gradient_norm={finite_norm}; "
                    "no portfolio optimizer committed")
            diagnostics[name] = {"loss": loss, "gradient_norm": norm}
        if replay_graph:
            self._optimizer_graph.replay()
        else:
            self._commit(names)
        return diagnostics

    step = __call__
