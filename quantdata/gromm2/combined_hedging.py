"""Joint ordinary-option, PCA-basket, spot and one-day-call hedge book."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.hedging_policy import ScenarioData
from quantdata.gromm2.joint_hedging import JointTrainConfig, OptionsOutput
from quantdata.gromm2.market_hedging import MarketView, load_market_scenarios
from quantdata.gromm2.portfolio_hedging import (
    BornOptionConfig,
    BornOptionPolicy,
    IndependentHedges,
    train_independent_portfolios,
)
from quantdata.torch import BaseEnvironment


@version("0.1", dependencies=[BornOptionConfig])
@dataclass(frozen=True, kw_only=True)
class CombinedBookConfig(BornOptionConfig):
    """Ordinary contracts precede PCA controls, one spot and the daily ATM call."""

    kind: str = "combined"
    basket_weights: tuple[tuple[float, ...], ...]
    call_sides: tuple[bool, ...]
    locked: tuple[bool, ...]
    with_fixing_call: bool = True
    spot_cost_bps: float = 0.1
    turnover_cost_basis: str = "unit_l1"

    def __post_init__(self) -> None:
        super().__post_init__()
        weights = np.asarray(self.basket_weights)
        verify_inp(self.kind == "combined" and self.turnover_cost_basis == "unit_l1",
            lambda: "the combined book requires unit-L1 costs")
        verify_inp(weights.ndim == 2 and min(weights.shape) > 0 and np.all(np.isfinite(weights)),
            lambda: "PCA basket weights must be a finite nonempty matrix")
        verify_inp(len(self.locked) == len(self.call_sides) == 84,
            lambda: "the combined ordinary book must contain the saved 84 contracts")
        verify_inp(self.instrument_count == self.option_count + self.basket_count + 2,
            lambda: "combined controls must include ordinary options, baskets, one spot and one daily call")
        verify_inp(all(start == 0 and expiry == 20 for start, expiry in
            zip(self.option_start_days[self.option_count:], self.option_expiry_days[self.option_count:], strict=True)),
            lambda: "PCA, spot and daily-call controls trade throughout the 20-day hedge horizon")
        verify_inp(np.isfinite(self.spot_cost_bps) and self.spot_cost_bps >= 0,
            lambda: "spot costs must be finite and nonnegative")

    @property
    def option_count(self) -> int:
        return len(self.locked)

    @property
    def basket_count(self) -> int:
        return len(self.basket_weights[0])


class CombinedBookPolicy(BornOptionPolicy):
    """One recurrent policy jointly trades both books without duplicating spot."""

    def __init__(self, env: BaseEnvironment, *, config: CombinedBookConfig,
                 normalization: tuple[np.ndarray, np.ndarray]) -> None:
        super().__init__(env, config=config, normalization=normalization)
        self.register_buffer("locked", torch.as_tensor(config.locked, dtype=torch.bool, device=env.device))
        self.register_buffer("calls", torch.as_tensor(config.call_sides, dtype=torch.bool, device=env.device))
        self.register_buffer("basket_weights", torch.as_tensor(config.basket_weights, dtype=env.fdtype, device=env.device))

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> OptionsOutput:
        normalized = (features.to(dtype=self.fdtype) - self.normalization_mean) / self.normalization_scale
        paths, periods = normalized.shape[:2]
        controls, options = self.config.instrument_count, self.config.option_count
        basket_end = options + self.config.basket_count
        previous = normalized.new_zeros((paths, controls))
        hidden = normalized.new_zeros((paths, self.config.hidden_size))
        cell = torch.zeros_like(hidden)
        positions, penalties, caps = [], [], []
        for day in range(periods):
            alive = (day >= self.option_starts) & (day < self.option_expiries)
            previous = previous * alive
            previous = torch.cat((previous[:, :-1], torch.zeros_like(previous[:, -1:])), dim=-1)
            ordinary_live = alive[:options]
            ordinary_trade = ordinary_live & (~self.locked | (day == self.option_starts[:options]))
            if self.config.with_fixing_call:
                ordinary_trade = ordinary_trade | (ordinary_live & (self.option_starts[:options] == 10)
                    & self.calls & (torch.arange(options, device=self.device) == 80))
            strikes = prices[:, day, 1 + controls:1 + controls + options]
            identical = (strikes[:, :, None] - strikes[:, None, :14]).abs() <= 1e-12
            identical = identical & (strikes[:, :, None] > 0)
            identical = identical & (self.option_expiries[:options, None] == self.option_expiries[None, :14])
            identical = identical & (self.calls[:, None] == self.calls[None, :14])
            duplicates = identical.any(dim=-1) & (torch.arange(options, device=self.device) >= 14)
            tradable = torch.cat((ordinary_trade[None] & ~duplicates,
                alive[None, options:].expand(paths, -1)), dim=-1)
            combined = torch.cat((normalized[:, day], hidden, previous / self.config.position_cap), dim=-1)
            raw = self.affine(combined) + self.deep(combined)
            candidate = self.config.position_cap * torch.tanh(raw / self.config.position_cap)
            underlying = candidate[:, options:basket_end] @ self.basket_weights.T
            scale = torch.clamp(underlying.abs().amax(dim=-1, keepdim=True) / self.config.position_cap, min=1)
            candidate = torch.cat((candidate[:, :options], candidate[:, options:basket_end] / scale,
                candidate[:, basket_end:]), dim=-1)
            position = torch.where(tradable, candidate, previous) * alive
            excess = torch.relu(raw.abs() - self.config.soft_cap_fraction * self.config.position_cap)
            penalty = (excess.square() * tradable).sum(dim=-1) / tradable.sum(dim=-1).clamp_min(1)
            penalty = penalty + torch.relu(underlying.abs() - self.config.soft_cap_fraction * self.config.position_cap).square().mean(dim=-1)
            actual_units = torch.cat((position, position[:, options:basket_end] @ self.basket_weights.T), dim=-1)
            caps.append((actual_units.abs() >= 0.95 * self.config.position_cap).to(self.fdtype).mean(dim=-1))
            penalties.append(penalty)
            hidden, cell = self.memory(torch.cat((normalized[:, day], previous / self.config.position_cap,
                position / self.config.position_cap), dim=-1), (hidden, cell))
            positions.append(position)
            previous = position
        model_holdings = torch.stack(positions, dim=1)
        holdings = model_holdings.to(prices.dtype)
        spot_changes = torch.diff(prices[..., 0], dim=1)
        option_gain = (holdings[:, :, :options] * torch.diff(prices[:, :, 1:1 + options], dim=1)).sum(dim=(1, 2))
        basket_gain = (holdings[:, :, options:basket_end]
            * torch.diff(prices[:, :, 1 + options:1 + basket_end], dim=1)).sum(dim=(1, 2))
        spot_gain = (holdings[:, :, -2] * spot_changes).sum(dim=1)
        call_gain = (holdings[:, :, -1] * (torch.relu(spot_changes) - prices[:, :-1, 1 + controls - 1])).sum(dim=1)
        gross = option_gain + basket_gain + spot_gain + call_gain
        prior = torch.cat((torch.zeros_like(holdings[:, :1]), holdings[:, :-1]), dim=1)
        option_alive = torch.arange(periods, device=self.device)[:, None] < self.option_expiries[:options]
        option_turnover = ((holdings[:, :, :options] - prior[:, :, :options]).abs() * option_alive).sum(dim=(1, 2))
        basket_turnover = (holdings[:, :, options:basket_end] - prior[:, :, options:basket_end]).abs().sum(dim=(1, 2))
        basket_turnover += holdings[:, -1, options:basket_end].abs().sum(dim=-1)
        spot_turnover = (holdings[:, :, -2] - prior[:, :, -2]).abs().sum(dim=1) + holdings[:, -1, -2].abs()
        call_turnover = holdings[:, :, -1].abs().sum(dim=1)
        costs = 1e-4 * (self.config.transaction_cost_bps * (option_turnover + basket_turnover + call_turnover)
            + self.config.spot_cost_bps * spot_turnover)
        return OptionsOutput(gross - costs, torch.stack(penalties, dim=1).mean(dim=1),
            model_holdings.square().sum(dim=-1).mean(dim=1), torch.stack(caps, dim=1).mean(dim=1), model_holdings, costs, gross)


class CombinedMarketView(MarketView):
    """Observe the unchanged information set and expose both stored hedge books."""

    def __init__(self, arrays: dict, book: dict, *, features: bool) -> None:
        super().__init__(arrays, book, kind="cohorts", features=features)
        self.kind = "combined"
        if not features:
            self.shape = (self.shape[0], 21, 2 * len(book["starts"]) + arrays["factor_prices"].shape[-1] + 3)

    def __getitem__(self, rows: slice | np.ndarray) -> np.ndarray:
        if self.features:
            return super().__getitem__(rows)
        values = {name: np.asarray(self.arrays[name][rows]) for name in ("spots", "option_prices", "factor_prices", "daily_calls")}
        strikes = values["spots"][:, self.book["starts"]] * self.book["levels"][None]
        known = np.where(np.arange(21)[None, :, None] >= self.book["starts"][None, None], strikes[:, None], 0.0)
        calls = np.pad(values["daily_calls"], ((0, 0), (0, 1)))
        return np.concatenate((values["spots"][..., None], values["option_prices"], values["factor_prices"],
            values["spots"][..., None], calls[..., None], known), axis=-1)


def load_combined_scenarios(path: Path, *, split: str, case: str) -> tuple[ScenarioData, dict]:
    """Reuse the same complete paths, payoff, source identities and causal features."""
    original, book = load_market_scenarios(path, split=split, case=case, kind="cohorts")
    arrays = original.features.arrays
    expiries = original.option_expiry_days + (20,) * (arrays["factor_prices"].shape[-1] + 2)
    return ScenarioData(CombinedMarketView(arrays, book, features=True), CombinedMarketView(arrays, book, features=False),
        original.payoff, expiries, original.identity, original.sha256), book


def embed_pca_policy(source: IndependentHedges, *, config: CombinedBookConfig) -> dict[str, torch.Tensor]:
    """Embed the fitted PCA policies exactly, initially setting ordinary actions to zero."""
    if source.config.kind != "factors" or source.config.instrument_count != config.basket_count + 2:
        raise ValueError("the initialization must be the matching PCA book")
    if (source.config.feature_count != config.feature_count or source.config.hidden_size != config.hidden_size
            or asdict(source.config.deep) != asdict(config.deep) or source.config.position_cap != config.position_cap
            or tuple(source.config.basket_weights) != tuple(config.basket_weights)):
        raise ValueError("PCA architecture, basket identities or action scaling changed")
    normalization = tuple(source.policies["zero"].state_dict()[f"normalization_{key}"].numpy() for key in ("mean", "scale"))
    target = IndependentHedges(source.env, config=config, normalization=normalization, policy_class=CombinedBookPolicy)
    old_controls = source.config.instrument_count
    offset = config.option_count
    input_prefix = config.feature_count + config.hidden_size
    input_indices = torch.cat((torch.arange(input_prefix), torch.arange(old_controls) + input_prefix + offset))
    memory_indices = torch.cat((torch.arange(config.feature_count),
        torch.arange(old_controls) + config.feature_count + offset,
        torch.arange(old_controls) + config.feature_count + config.instrument_count + offset))
    output_indices = torch.arange(old_controls) + offset
    for name, policy in target.policies.items():
        source_state = source.policies[name].state_dict()
        state = policy.state_dict()
        parameter_names = set(dict(policy.named_parameters()))
        for key in parameter_names:
            old, new = source_state[key], state[key]
            if old.shape == new.shape:
                new.copy_(old)
                continue
            new.zero_()
            if key == "memory.weight_ih":
                new[:, memory_indices] = old
            elif key == "affine.I0.weight":
                new[output_indices[:, None], input_indices[None]] = old
            elif key == "affine.I0.bias" or (key.startswith("deep.") and key.endswith("bias")):
                new[output_indices] = old
            elif key == "deep.I0.weight" and config.deep.num_depth > 0:
                new[:, input_indices] = old
            elif key == "deep.I0.weight":
                new[output_indices[:, None], input_indices[None]] = old
            elif key.startswith("deep.") and key.endswith("weight"):
                new[output_indices] = old
            else:
                raise ValueError(f"unsupported PCA parameter mapping: {key}")
        for key in ("normalization_mean", "normalization_scale"):
            state[key].copy_(source_state[key])
        policy.load_state_dict(state)
    return target.state_dict()


def run_combined_hedges(run_directory: Path, *, output_directory: Path, pca_checkpoint: Path | None,
                        train_config: JointTrainConfig = JointTrainConfig(device="cuda", compile=True),
                        just_load: bool = False) -> dict[str, object]:
    """Train the same three independent objectives on the union of the two books."""
    train, book = load_combined_scenarios(run_directory / "train.h5", split="train", case="black0")
    validation, _ = load_combined_scenarios(run_directory / "validation.h5", split="validation", case="black0")
    if train.sha256 == validation.sha256 or train.identity["model"] != validation.identity["model"]:
        raise ValueError("combined training and validation must be independent samples of the same model")
    weights = json.loads((run_directory / "factor_baskets.json").read_text())["weights"][1]
    config = CombinedBookConfig(label="black0-combined-lambda100-unit-l1", feature_count=train.features.shape[-1],
        option_expiry_days=train.option_expiry_days, option_start_days=tuple(map(int, book["starts"])) + (0,) * (len(weights[0]) + 2),
        call_sides=tuple(map(bool, book["calls"])), locked=tuple(map(bool, book["locked"])),
        basket_weights=tuple(tuple(map(float, row)) for row in weights), risk_aversion=100.0)
    initial_state, initialization = None, None
    if pca_checkpoint is not None:
        from quantdata.gromm2.hedging_cost_comparison import frozen_market_model

        serialized = pca_checkpoint.read_bytes()
        checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
        if (checkpoint["identity"]["training_scenarios_sha256"] != train.sha256
                or checkpoint["identity"]["validation_scenarios_sha256"] != validation.sha256):
            raise ValueError("PCA initialization uses different path populations")
        initial_state = embed_pca_policy(frozen_market_model(checkpoint), config=config)
        initialization = dict(kind="embedded-validation-selected-PCA", source=str(pca_checkpoint),
            checkpoint_sha256=hashlib.sha256(serialized).hexdigest(), training_scenarios_sha256=train.sha256,
            validation_scenarios_sha256=validation.sha256,
            best_epochs={name: record["best_epoch"] for name, record in checkpoint["state"]["policies"].items()},
            description="Same PCA holdings and recurrent policy; ordinary actions initially zero; new independent optimizers.")
    return train_independent_portfolios(train, validation, directory=output_directory / config.filename,
        config=config, train_config=train_config, just_load=just_load, policy_class=CombinedBookPolicy,
        initial_model_state=initial_state, initialization=initialization)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--pca-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--minimum-epochs", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--host-cache-training", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    arguments = parser.parse_args()
    result = run_combined_hedges(arguments.run_directory, output_directory=arguments.output_directory,
        pca_checkpoint=arguments.pca_checkpoint, just_load=arguments.just_load,
        train_config=JointTrainConfig(device=arguments.device, compile=arguments.compile,
            maximum_epochs=arguments.maximum_epochs, minimum_epochs=arguments.minimum_epochs,
            cuda_memory_fraction=1.0, host_cache_training=arguments.host_cache_training))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
