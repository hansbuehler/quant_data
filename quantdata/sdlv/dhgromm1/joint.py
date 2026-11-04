"""Joint bid/ask loss, with independent recurrent networks and cash translations."""

from __future__ import annotations

import copy
import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.wide_audit import AuditMonitor
from quantdata.sdlv.dhgromm1.feedback import FeedbackConfig, FeedbackHedge, migrate_policy
from quantdata.sdlv.dhgromm1.hedge import HedgeConfig, HedgeOutput
from quantdata.sdlv.dhgromm1.training import TrainConfig, load_frozen_policy
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.utility import torch_utility

if TYPE_CHECKING:
    from quantdata.sdlv.dhgromm1.feedback_run import ResidentSample


@version("0.1", dependencies=[FeedbackConfig])
@dataclass(frozen=True, kw_only=True)
class JointConfig:
    """Versioned sum objective; neither network nor OCE translation is shared."""

    label: str
    group: str = "sdlv/dhgromm1"
    user_version: str = "0.1"
    long: FeedbackConfig = FeedbackConfig(label="long")
    short: FeedbackConfig = FeedbackConfig(label="short")

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and self.long.label == "long" and self.short.label == "short",
                   lambda: "joint model requires distinct long and short policy labels")


class JointHedges(BaseModel):
    """Return [long-claim loss, short-claim loss]; minimize their SUM."""

    policy_type = FeedbackHedge

    def __init__(self, env: BaseEnvironment, *, config: JointConfig,
                 means: dict[str, torch.Tensor], scales: dict[str, torch.Tensor]) -> None:
        super().__init__(env)
        self.config = config
        self.long = self.policy_type(env, config=config.long, price_mean=means["long"], price_scale=scales["long"])
        self.short = self.policy_type(env, config=config.short, price_mean=means["short"], price_scale=scales["short"])
        # Validate/build the shared utility outside compilation. The convenience
        # OCE wrapper's NumPy input check is not traceable in a full loss graph.
        self._long_utility, _, _ = torch_utility("powexp", config.long.risk_aversion)
        self._short_utility, _, _ = torch_utility("powexp", config.short.risk_aversion)
        if {id(p) for p in self.long.parameters()} & {id(p) for p in self.short.parameters()}:
            raise RuntimeError("joint bid/ask policies must not share parameters")

    def forward(self, prices: torch.Tensor, state: torch.Tensor, returns: torch.Tensor,
                payoff: torch.Tensor, *, eager: bool = False) -> torch.Tensor:
        long = self.long(prices, state, returns=returns, eager=eager)
        short = self.short(prices, state, returns=returns, eager=eager)
        return self.losses(long, short, payoff)

    def losses(self, long: HedgeOutput, short: HedgeOutput, payoff: torch.Tensor) -> torch.Tensor:
        """Sum-compatible OCE components, evaluated with ordinary device tensors."""
        long_value = self._long_utility(long.gain + payoff - self.long.translation, torch.zeros_like(long.gain))
        short_value = self._short_utility(short.gain - payoff - self.short.translation, torch.zeros_like(short.gain))
        long_loss = -(self.long.translation + long_value.mean()) + long.penalty.mean()
        short_loss = -(self.short.translation + short_value.mean()) + short.penalty.mean()
        return torch.stack((long_loss, short_loss))

    def compiled_losses(self) -> Callable[..., torch.Tensor]:
        """Compile each recurrent scan independently, then assemble the sum.

        The installed compiler's combined two-scan/loss graph failed the GPU
        gradient regression. Separate compiled scans preserve the exact joint
        mathematical objective; loss arithmetic still stays entirely on GPU.
        """
        long = torch.compile(self.long, fullgraph=True)
        short = torch.compile(self.short, fullgraph=True)

        def execute(prices: torch.Tensor, state: torch.Tensor, returns: torch.Tensor,
                    payoff: torch.Tensor) -> torch.Tensor:
            return self.losses(long(prices, state, returns=returns),
                               short(prices, state, returns=returns), payoff)

        return execute


def load_joint_checkpoint(path: Path, *, config: JointConfig) -> dict:
    """Validate versions and both geometries before constructing any models."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("model_code_version") != type(config).version.unique_id48
            or saved.get("user_version") != config.user_version
            or saved.get("model_config") != asdict(config)):
        raise ValueError("joint checkpoint objective, version or geometry differs")
    return saved


def load_joint_model(path: Path, *, config: JointConfig, env: BaseEnvironment,
                     model_type: type[JointHedges] = JointHedges) -> JointHedges:
    """Load the selected pair for valuation without starting any fitting."""
    saved = load_joint_checkpoint(path, config=config)
    state = saved["best_state"]
    model = model_type(env, config=config,
        means={label: state[f"{label}.price_mean"] for label in ("long", "short")},
        scales={label: state[f"{label}.price_scale"] for label in ("long", "short")})
    model.load_state_dict(state)
    model.requires_grad_(False)
    return model


def _snapshot(model: JointHedges) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _write_history(output: Path, history: list[dict]) -> None:
    temporary = output / "joint-losses.tmp.csv"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "joint-losses.csv")


def train_joint(*, source: Path, output: Path, train: TrainConfig, sample: ResidentSample,
                monitor: AuditMonitor, state_normalizers: dict, config: JointConfig | None = None,
                initializer: Callable[[BaseEnvironment, JointConfig], tuple[JointHedges, dict[str, int]]] | None = None,
                ) -> dict[str, FeedbackHedge]:
    """One summed loss/backward/Adam step per batch, updating both networks.

    Each joint epoch visits the entire training sample for EACH policy. Separate
    gradient clipping prevents one network's gradient norm rescaling the other.
    Checkpoint selection minimizes the training SUM and selects one common epoch;
    independent evaluation never selects weights, scalers or translations.
    """
    config = config or JointConfig(label="bid-ask")
    checkpoint = output / "joint.checkpoint.pt"
    saved = load_joint_checkpoint(checkpoint, config=config) if checkpoint.exists() else None
    if saved:
        if saved["training_paths"] != sample.size:
            raise ValueError("joint checkpoint belongs to a different path population")
        if any(not np.array_equal(saved["state_normalization"][key], state_normalizers[key])
               for key in ("mean", "scale")):
            raise ValueError("replayed state normalization differs from the joint checkpoint")
    env = BaseEnvironment(device="cuda", precision=32, debug=False)
    torch.manual_seed(train.seed)
    if initializer is not None:
        model, source_epochs = initializer(env, config)
    else:
        old = {label: load_frozen_policy(source / f"{label}.checkpoint.pt", config=HedgeConfig(label=label),
                                         device="cuda") for label in ("long", "short")}
        torch.manual_seed(train.seed)
        model = JointHedges(env, config=config, means={k: p.price_mean for k, p in old.items()},
                            scales={k: p.price_scale for k, p in old.items()})
        source_epochs = {k: p.fit_summary["selected_epoch"] for k, p in old.items()}
        for label, policy in old.items():
            migrate_policy(policy, getattr(model, label))
        del old
    optimizer = torch.optim.Adam([dict(params=list(model.long.parameters()), name="long"),
                                  dict(params=list(model.short.parameters()), name="short")], lr=train.learning_rate)
    rng = np.random.default_rng(train.seed)
    pairs = np.arange(sample.size // 2)
    start_epoch = 0
    if saved:
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = train.learning_rate
        start_epoch = saved["epoch"] + 1
        rng.bit_generator.state = saved["numpy_rng"]
        pairs = saved["pair_order"]
        if not np.array_equal(np.sort(pairs), np.arange(sample.size // 2)):
            raise ValueError("joint checkpoint has a different antithetic pair population")
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    execute = model.compiled_losses() if train.compile_model else model

    def score() -> np.ndarray:
        total = torch.zeros(2, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            for begin in range(0, sample.size, train.batch_size):
                price, ret, payoff, state = sample.batch(slice(begin, begin + train.batch_size))
                total += execute(price, state, ret, payoff).double() * len(price)
        return (total / sample.size).cpu().numpy()

    monitor.stage("initial-joint-score")
    initial = score()
    if not np.isfinite(initial).all():
        raise FloatingPointError("nonfinite joint warm-start loss")
    best_loss = saved["best_loss"] if saved else float(initial.sum())
    best_parts = saved["best_parts"] if saved else initial.copy()
    best_state = saved["best_state"] if saved else _snapshot(model)
    selected_epoch = saved["selected_epoch"] if saved else 0
    history = saved["history"] if saved else [dict(epoch=0, long_loss=float(initial[0]), short_loss=float(initial[1]),
        sum_loss=float(initial.sum()), online_sum_loss=float(initial.sum()))]
    print(json.dumps(dict(stage="joint-warm-start", source=str(source), source_epochs=source_epochs,
        epochs=train.epochs, long_loss=float(initial[0]), short_loss=float(initial[1]), sum_loss=float(initial.sum()),
        networks="independent, no shared parameters", objective="long_loss + short_loss, not averaged")), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, train.epochs):
        monitor.stage("train-joint", epoch, train.epochs)
        rng.shuffle(pairs)
        order = torch.as_tensor(pairs, device="cuda")
        indices = (2 * order[:, None] + torch.arange(2, device="cuda")).flatten()
        online_total = torch.zeros(2, dtype=torch.float64, device="cuda")
        finite = torch.ones((), dtype=torch.bool, device="cuda")
        for begin in range(0, sample.size, train.batch_size):
            selection = indices[begin:begin + train.batch_size]
            price, ret, payoff, state = sample.batch(selection)
            losses = execute(price, state, ret, payoff)
            loss_sum = losses.sum()
            optimizer.zero_grad(set_to_none=True)
            loss_sum.backward()
            finite.logical_and_(torch.isfinite(loss_sum.detach()))
            for policy in (model.long, model.short):
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), train.gradient_clip, foreach=True)
                finite.logical_and_(torch.isfinite(norm))
            optimizer.step()
            online_total += losses.detach().double() * len(selection)
        values = score()
        online = (online_total / sample.size).cpu().numpy()
        if not bool(finite) or not np.isfinite(values).all() or not np.isfinite(online).all():
            raise FloatingPointError(f"nonfinite joint epoch {epoch + 1}; prior atomic checkpoint retained")
        value = float(values.sum())
        if value < best_loss:
            best_loss, best_parts, best_state, selected_epoch = value, values.copy(), _snapshot(model), epoch + 1
        history.append(dict(epoch=epoch + 1, long_loss=float(values[0]), short_loss=float(values[1]),
                            sum_loss=value, online_sum_loss=float(online.sum())))
        elapsed = time.monotonic() - started
        print(json.dumps(dict(stage="train-joint", epoch=epoch + 1, epochs=train.epochs,
            long_loss=float(values[0]), short_loss=float(values[1]), sum_loss=value,
            best_sum_loss=best_loss, selected_epoch=selected_epoch, elapsed_seconds=elapsed,
            completed=epoch + 1 - start_epoch, total=train.epochs - start_epoch,
            estimated_remaining_seconds=elapsed * (train.epochs - epoch - 1) / (epoch + 1 - start_epoch))), flush=True)
        payload = dict(model_config=asdict(config), model_code_version=type(config).version.unique_id48,
            user_version=config.user_version, train_config=asdict(train), state_dict=model.state_dict(),
            best_state=best_state, best_loss=best_loss, best_parts=best_parts, optimizer=optimizer.state_dict(),
            epoch=epoch, selected_epoch=selected_epoch, history=history, numpy_rng=rng.bit_generator.state,
            pair_order=pairs.copy(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            state_normalization=state_normalizers, source=str(source), source_epochs=source_epochs,
            training_paths=sample.size, objective="sum", separate_networks=True, separate_translations=True)
        temporary = checkpoint.with_suffix(".tmp.pt")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        _write_history(output, history)
    model.load_state_dict(best_state)
    policies = {label: getattr(model, label) for label in ("long", "short")}
    for index, (label, policy) in enumerate(policies.items()):
        policy.fit_summary = dict(initial_training_loss=history[0][f"{label}_loss"],
            selected_training_loss=float(best_parts[index]), selected_epoch=selected_epoch,
            selection="minimum joint training SUM", best_joint_loss=best_loss, epochs_run=train.epochs,
            source_selected_epoch=source_epochs[label], joint_checkpoint=str(checkpoint))
        # These exports are inference snapshots. Exact continuation uses the
        # joint checkpoint, which includes both networks, Adam and RNG states.
        torch.save(dict(model_config=asdict(policy.config), model_code_version=type(policy.config).version.unique_id48,
            user_version=policy.config.user_version, best_state=policy.state_dict(), inference_only=True,
            selected_epoch=selected_epoch, source_joint_checkpoint=str(checkpoint)), output / f"{label}.selected.pt")
    return policies


def verify_joint_gpu(*, batch_size: int = 256, config: JointConfig | None = None,
                     model_type: type[JointHedges] = JointHedges,
                     prepare: Callable[[JointHedges], None] | None = None) -> dict:
    """Compare compiled summed CUDA losses and every gradient to the reference."""
    torch.manual_seed(819)
    env = BaseEnvironment(device="cuda", precision=32, debug=False)
    config = config or JointConfig(label="gpu-verification")
    model = model_type(env, config=config, means={k: torch.zeros(21) for k in ("long", "short")},
                        scales={k: torch.ones(21) for k in ("long", "short")})
    with torch.no_grad():
        for policy in (model.long, model.short):
            policy.trade_head.weight.add_(torch.randn_like(policy.trade_head.weight) * 1e-5)
            policy.trade_head.bias.add_(.001)
            policy.recurrence.weight_ih[:, 22:].mul_(.01)
    if prepare is not None:
        prepare(model)
    candidate = copy.deepcopy(model)
    execute = candidate.compiled_losses()
    price = torch.rand(batch_size, 21, 21, device="cuda") + .8
    ret = torch.randn(batch_size, 20, 21, device="cuda") * .01
    state = torch.randn(batch_size, 20, 5003, device="cuda") * .1
    payoff = torch.linspace(.005, .04, batch_size, device="cuda")
    started = time.monotonic()
    expected = model(price, state, ret, payoff, eager=True)
    actual = execute(price, state, ret, payoff)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-3)
    expected.sum().backward()
    actual.sum().backward()
    errors = {}
    for (name, p), (_, q) in zip(model.named_parameters(), candidate.named_parameters(), strict=True):
        if p.grad is not None:
            delta = (p.grad - q.grad).abs()
            ratio = delta / (3e-5 + 2e-3 * p.grad.abs())
            if float(ratio.max()) > 1:
                index = int(ratio.flatten().argmax())
                print(json.dumps(dict(stage="joint-gradient-mismatch", parameter=name,
                    max_absolute_error=float(delta.max()), reference_norm=float(p.grad.norm()),
                    compiled_norm=float(q.grad.norm()), worst_reference=float(p.grad.flatten()[index]),
                    worst_compiled=float(q.grad.flatten()[index]), max_tolerance_ratio=float(ratio.max()))), flush=True)
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=2e-3)
            errors[name] = float((p.grad - q.grad).abs().max())
    if not all(getattr(model, label).translation.grad.abs().item() > 0 for label in ("long", "short")):
        raise RuntimeError("joint verification did not exercise both independent OCE translations")
    return dict(status="pass", objective="sum of independent long/short losses", separate_networks=True,
        batch_size=batch_size, elapsed_seconds=time.monotonic() - started,
        max_loss_error=float((actual - expected).detach().abs().max()),
        gradient_errors=errors, recurrence="two compiled differentiable scans")
