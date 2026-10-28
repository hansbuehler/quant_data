"""Independent recurrent hedges with locked cohorts or financed real-option baskets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.hedging_data import map_dataset, scenario_digest
from quantdata.gromm2.hedging_policy import ScenarioData
from quantdata.gromm2.joint_hedging import JointTrainConfig, OptionsOutput
from quantdata.gromm2.portfolio_hedging import (
    BornOptionConfig,
    BornOptionPolicy,
    train_independent_portfolios,
)
from quantdata.torch import BaseEnvironment


@version("0.1", dependencies=[BornOptionConfig])
@dataclass(frozen=True, kw_only=True)
class MarketBookConfig(BornOptionConfig):
    """Economic controls, not artificial prices of latent model coordinates."""

    kind: str
    basket_weights: tuple[tuple[float, ...], ...] = ()
    call_sides: tuple[bool, ...] = ()
    locked: tuple[bool, ...] = ()
    with_fixing_call: bool = True
    spot_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.kind in {"cohorts", "factors"}, lambda: "unknown market hedge book")
        verify_inp(self.spot_cost_bps >= 0, lambda: "spot costs must be nonnegative")
        if self.kind == "cohorts":
            verify_inp(len(self.locked) == self.instrument_count == len(self.call_sides),
                lambda: "cohort identities must match the action book")
        else:
            weights = np.asarray(self.basket_weights)
            verify_inp(weights.ndim == 2 and weights.shape[1] + 2 == self.instrument_count
                and np.all(np.isfinite(weights)), lambda: "factor weights and controls disagree")


class MarketBookPolicy(BornOptionPolicy):
    """Choose only current actions; settlement values enter gains, never the state."""

    def __init__(self, env: BaseEnvironment, *, config: MarketBookConfig,
                 normalization: tuple[np.ndarray, np.ndarray]) -> None:
        super().__init__(env, config=config, normalization=normalization)
        if config.kind == "cohorts":
            self.register_buffer("locked", torch.as_tensor(config.locked, dtype=torch.bool, device=env.device))
            self.register_buffer("calls", torch.as_tensor(config.call_sides, dtype=torch.bool, device=env.device))
        else:
            self.register_buffer("basket_weights", torch.as_tensor(config.basket_weights, dtype=env.fdtype, device=env.device))

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> OptionsOutput:
        normalized = (features.to(dtype=self.fdtype) - self.normalization_mean) / self.normalization_scale
        paths, periods = normalized.shape[:2]
        controls = self.config.instrument_count
        previous = normalized.new_zeros((paths, controls))
        hidden = normalized.new_zeros((paths, self.config.hidden_size))
        cell = torch.zeros_like(hidden)
        positions, penalties, caps = [], [], []
        for day in range(periods):
            alive = (day >= self.option_starts) & (day < self.option_expiries)
            previous = previous * alive
            if self.config.kind == "factors":
                previous = torch.cat((previous[:, :-1], torch.zeros_like(previous[:, -1:])), dim=-1)
                tradable = alive[None].expand(paths, -1)
            else:
                tradable = alive & (~self.locked | (day == self.option_starts))
                if self.config.with_fixing_call:
                    tradable = tradable | (alive & (self.option_starts == 10) & self.calls
                        & (torch.arange(controls, device=self.device) == 80))
                cash_strikes = prices[:, day, 1 + controls:1 + 2 * controls]
                identical = ((cash_strikes[:, :, None] - cash_strikes[:, None, :14]).abs() <= 1e-12)
                identical = identical & (cash_strikes[:, :, None] > 0)
                identical = identical & (self.option_expiries[:, None] == self.option_expiries[None, :14])
                identical = identical & (self.calls[:, None] == self.calls[None, :14])
                duplicates = identical.any(dim=-1) & (torch.arange(controls, device=self.device) >= 14)
                tradable = tradable[None] & ~duplicates
            combined = torch.cat((normalized[:, day], hidden, previous / self.config.position_cap), dim=-1)
            raw = self.affine(combined) + self.deep(combined)
            candidate = self.config.position_cap * torch.tanh(raw / self.config.position_cap)
            if self.config.kind == "factors":
                underlying = candidate[:, :-2] @ self.basket_weights.T
                scale = torch.clamp(underlying.abs().amax(dim=-1, keepdim=True) / self.config.position_cap, min=1)
                candidate = torch.cat((candidate[:, :-2] / scale, candidate[:, -2:]), dim=-1)
            position = torch.where(tradable, candidate, previous)
            position = position * alive
            excess = torch.relu(raw.abs() - self.config.soft_cap_fraction * self.config.position_cap)
            penalty = (excess.square() * tradable).sum(dim=-1) / tradable.sum(dim=-1).clamp_min(1)
            if self.config.kind == "factors":
                penalty = penalty + torch.relu(underlying.abs() - self.config.soft_cap_fraction * self.config.position_cap).square().mean(dim=-1)
                actual_units = torch.cat((position[:, :-2] @ self.basket_weights.T, position[:, -2:]), dim=-1)
                cap_fraction = (actual_units.abs() >= 0.95 * self.config.position_cap).to(self.fdtype).mean(dim=-1)
            else:
                cap_fraction = ((position.abs() >= 0.95 * self.config.position_cap) & alive).to(self.fdtype).mean(dim=-1)
            penalties.append(penalty)
            caps.append(cap_fraction)
            hidden, cell = self.memory(torch.cat((normalized[:, day], previous / self.config.position_cap,
                position / self.config.position_cap), dim=-1), (hidden, cell))
            positions.append(position)
            previous = position
        holdings = torch.stack(positions, dim=1)
        spot = prices[..., 0]
        if self.config.kind == "cohorts":
            increments = prices[:, 1:, 1:1 + controls] - prices[:, :-1, 1:1 + controls]
            gross = (holdings.to(prices.dtype) * increments).sum(dim=(1, 2))
            prior = torch.cat((torch.zeros_like(holdings[:, :1]), holdings[:, :-1]), dim=1)
            alive = torch.arange(periods, device=self.device)[:, None] < self.option_expiries
            turnover = ((holdings - prior).abs() * alive).sum(dim=-1)
            costs = self.config.transaction_cost_bps * 1e-4 * (spot[:, :-1] * turnover).sum(dim=1)
        else:
            baskets = controls - 2
            factor_gain = (holdings[:, :, :baskets].to(prices.dtype)
                * torch.diff(prices[:, :, 1:1 + baskets], dim=1)).sum(dim=(1, 2))
            spot_gain = (holdings[:, :, -2].to(prices.dtype) * torch.diff(spot, dim=1)).sum(dim=1)
            call_gain = (holdings[:, :, -1].to(prices.dtype)
                * (torch.relu(torch.diff(spot, dim=1)) - prices[:, :-1, -1])).sum(dim=1)
            gross = factor_gain + spot_gain + call_gain
            underlying = holdings[:, :, :baskets] @ self.basket_weights.T
            previous_units = torch.cat((torch.zeros_like(underlying[:, :1]), underlying[:, :-1]), dim=1)
            turnover = (underlying - previous_units).abs().sum(dim=-1) + holdings[:, :, -1].abs()
            costs = self.config.transaction_cost_bps * 1e-4 * ((spot[:, :-1] * turnover).sum(dim=1)
                + spot[:, -1] * underlying[:, -1].abs().sum(dim=-1))
            spot_units = holdings[:, :, -2]
            previous_spot = torch.cat((torch.zeros_like(spot_units[:, :1]), spot_units[:, :-1]), dim=1)
            costs += self.config.spot_cost_bps * 1e-4 * ((spot[:, :-1] * (spot_units - previous_spot).abs()).sum(dim=1)
                + spot[:, -1] * spot_units[:, -1].abs())
        return OptionsOutput(gross - costs, torch.stack(penalties, dim=1).mean(dim=1),
            holdings.square().sum(dim=-1).mean(dim=-1), torch.stack(caps, dim=1).mean(dim=1), holdings, costs, gross)


class MarketView:
    """Map shared latent paths and materialize only the requested neural batch."""

    def __init__(self, arrays: dict, book: dict, *, kind: str, features: bool) -> None:
        self.arrays, self.book, self.kind, self.features = arrays, book, kind, features
        paths = arrays["spots"].shape[0]
        if features:
            columns = arrays["core_features"].shape[-1] + 3 + 3 * len(book["starts"]) + arrays["factor_prices"].shape[-1] + 1
        else:
            columns = 1 + 2 * len(book["starts"]) if kind == "cohorts" else arrays["factor_prices"].shape[-1] + 2
        self.shape = (paths, 20 if features else 21, columns)
        self.ndim, self.dtype = 3, np.dtype("f4" if features else "f8")

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows: slice | np.ndarray) -> np.ndarray:
        needed = tuple(self.arrays) if self.features else (
            ("spots", "option_prices") if self.kind == "cohorts" else ("spots", "factor_prices", "daily_calls"))
        values = {name: np.asarray(self.arrays[name][rows]) for name in needed}
        starts = self.book["starts"]
        days = np.arange(21)
        if self.features or self.kind == "cohorts":
            strikes = values["spots"][:, starts] * self.book["levels"][None]
            known = np.where(days[None, :, None] >= starts[None, None], strikes[:, None], 0.0)
        if self.features:
            spot, background = values["spots"][:, :20], values["backgrounds"][:, :20]
            fixing = np.where(np.arange(20)[None] >= 10, np.log(values["spots"][:, 10:11]), 0.0)
            alive = (days[:20, None] >= starts) & (days[:20, None] < self.book["expiries"])
            return np.concatenate((values["core_features"], np.log(spot)[..., None], np.log(background)[..., None],
                fixing[..., None], values["option_prices"][:, :20], known[:, :20],
                np.broadcast_to(alive, values["option_prices"][:, :20].shape),
                values["factor_prices"][:, :20], values["daily_calls"][..., None]), axis=-1).astype("f4")
        if self.kind == "cohorts":
            return np.concatenate((values["spots"][..., None], values["option_prices"], known), axis=-1)
        calls = np.pad(values["daily_calls"], ((0, 0), (0, 1)))
        return np.concatenate((values["spots"][..., None], values["factor_prices"], calls[..., None]), axis=-1)


def load_market_scenarios(path: Path, *, split: str, case: str, kind: str) -> tuple[ScenarioData, dict]:
    """Require complete independent samples; both books observe identical information."""
    with h5py.File(path, "r") as stream:
        identity = json.loads(str(stream.attrs["identity"]))
        if identity["split"] != split or stream.attrs.get("completed_paths", 0) != identity["config"]["paths"]:
            raise ValueError("market scenarios must be complete and correctly labelled")
        names = ("spots", "backgrounds", "option_prices", "factor_prices", "daily_calls")
        arrays = {name: map_dataset(path, stream[f"{case}/{name}"]) for name in names}
        arrays["core_features"] = map_dataset(path, stream["core_features"])
        payoff = map_dataset(path, stream[f"{case}/payoff"])
        book = {name: stream[f"primary/{name}"][:] for name in ("starts", "expiries", "levels", "locked", "calls")}
    identity["model"] = identity["model"] | dict(simulation_case=case,
        simulated_background_volatility=0.05 if case == "black5" else 0.0)
    features = MarketView(arrays, book, kind=kind, features=True)
    prices = MarketView(arrays, book, kind=kind, features=False)
    expiries = tuple(map(int, book["expiries"])) if kind == "cohorts" else (20,) * (arrays["factor_prices"].shape[-1] + 2)
    digest = scenario_digest(path)
    return ScenarioData(features, prices, payoff, expiries, identity, digest), book


def run_market_hedges(directory: Path, *, case: str, kind: str, device: str,
                      compile_policies: bool = True, maximum_epochs: int = 2000,
                      just_load: bool = False) -> dict[str, object]:
    train, book = load_market_scenarios(directory / "train.h5", split="train", case=case, kind=kind)
    validation, _ = load_market_scenarios(directory / "validation.h5", split="validation", case=case, kind=kind)
    if train.sha256 == validation.sha256 or train.identity["model"] != validation.identity["model"]:
        raise ValueError("training and validation must be independent samples of the same model")
    report = json.loads((directory / "factor_baskets.json").read_text())
    weights = report["weights"][0 if case == "black5" else 1]
    config = MarketBookConfig(label=f"{case}-{kind}-lambda100", kind=kind,
        feature_count=train.features.shape[-1], option_expiry_days=train.option_expiry_days,
        option_start_days=tuple(map(int, book["starts"])) if kind == "cohorts" else (0,) * len(train.option_expiry_days),
        call_sides=tuple(map(bool, book["calls"])) if kind == "cohorts" else (),
        locked=tuple(map(bool, book["locked"])) if kind == "cohorts" else (),
        basket_weights=tuple(tuple(map(float, row)) for row in weights) if kind == "factors" else (),
        risk_aversion=100.0)
    return train_independent_portfolios(train, validation, directory=directory / "hedges" / case / kind / config.filename,
        config=config, train_config=JointTrainConfig(device=device, compile=compile_policies, maximum_epochs=maximum_epochs),
        policy_class=MarketBookPolicy, just_load=just_load)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--case", choices=("black5", "black0"), required=True)
    parser.add_argument("--kind", choices=("cohorts", "factors"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    args = parser.parse_args()
    run_market_hedges(args.run_directory, case=args.case, kind=args.kind, device=args.device,
        compile_policies=args.compile, maximum_epochs=args.maximum_epochs, just_load=args.just_load)


if __name__ == "__main__":
    main()
