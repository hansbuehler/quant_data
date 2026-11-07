"""Shared asset-conditioned terminal hedges with explicit premium-dollar limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.hedge2.cash import zero_strike_mask
from quantdata.hedge2.terminal import TerminalPaths, shared_history_values
from quantdata.hedge2.terminal_features import TerminalFeatureMapping
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class TerminalNetworkConfig(FeedForwardConfig):
    activation: str = "Tanh"
    num_depth: int = 2
    num_width: int = 32


@version("0.3", dependencies=[TerminalNetworkConfig])
@dataclass(frozen=True, kw_only=True)
class TerminalPolicyConfig:
    label: str
    feature_count: int
    group: str = "hedge2/terminal"
    user_version: str = "0.1"
    instrument_cap: float = 10.
    deep: TerminalNetworkConfig = TerminalNetworkConfig()
    use_deep: bool = True
    fee_preconditioning: bool = False

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
                   lambda: f"Invalid terminal-policy label: {self.label!r}")
        verify_inp(bool(self.group) and all(is_filename(part) and part not in ("", ".", "..")
                   for part in self.group.split("/")), lambda: f"Invalid terminal-policy group: {self.group!r}")
        verify_inp(isinstance(self.feature_count, int) and self.feature_count > 0
                   and np.isfinite(self.instrument_cap) and self.instrument_cap > 0,
                   lambda: "Terminal policy requires positive feature count and dollar limit")
        verify_inp(isinstance(self.fee_preconditioning, bool),
                   lambda: "Fee preconditioning must be an explicit boolean model choice")

    @property
    def filename(self) -> str:
        return (label_unique_filename48_8(self.label[1:]) if self.label.startswith("*")
                else named_unique_filename48_8(self.label, self))


@dataclass(frozen=True, kw_only=True)
class PolicyBook:
    """Predictable cash-price inputs; prices are in the episode's original spot units."""

    prices: np.ndarray
    asset_descriptors: np.ndarray
    initial_business_dtes: np.ndarray

    def __post_init__(self) -> None:
        if (self.prices.ndim != 3 or min(self.prices.shape) < 1
                or self.asset_descriptors.shape != (self.prices.shape[2], 4)
                or self.initial_business_dtes.shape != (self.prices.shape[2],)
                or not all(np.isfinite(value).all() for value in
                    (self.prices, self.asset_descriptors, self.initial_business_dtes))
                or np.any(self.prices < 0) or np.any(self.prices[:, :, 0] <= 0)
                or not np.array_equal(self.asset_descriptors[:, 0], np.r_[1., np.zeros(self.prices.shape[2] - 1)])
                or np.any(self.initial_business_dtes < 0)):
            raise ValueError("Policy book needs finite nonnegative prices, descriptors, maturities and first-asset spot")

    def validate_histories(self, paths: TerminalPaths) -> None:
        if self.prices.shape != paths.gains.shape:
            raise ValueError("Policy prices and frozen terminal instruments must align")
        for step in range(self.prices.shape[1]):
            if not shared_history_values(self.prices[:, step], paths.history_ids[:, step], tolerance=1e-12):
                raise ValueError("Policy prices reveal information not available at the trade-time history")

    def tensor(self, env: BaseEnvironment) -> TensorPolicyBook:
        return TensorPolicyBook(prices=torch.tensor(self.prices, dtype=env.fdtype, device=env.device),
            asset_descriptors=torch.tensor(self.asset_descriptors, dtype=env.fdtype, device=env.device),
            initial_business_dtes=torch.tensor(self.initial_business_dtes, dtype=env.fdtype, device=env.device))


@dataclass(frozen=True, kw_only=True)
class TensorPolicyBook:
    prices: torch.Tensor
    asset_descriptors: torch.Tensor
    initial_business_dtes: torch.Tensor


def policy_book(*, book: pd.DataFrame, root_spot: float, prices: np.ndarray) -> PolicyBook:
    """root_spot is TODAY'S forecast-root spot, not the older episode-inception spot."""
    if not np.isfinite(root_spot) or root_spot <= 0:
        raise ValueError("The policy book's spot reference must be positive")
    zero = zero_strike_mask(book)
    strikes = book["strike"].to_numpy(dtype=float)
    descriptors = np.zeros((1 + len(book), 4))
    descriptors[0, 0] = 1.
    descriptors[1:, 1] = zero
    descriptors[1:, 2] = (book["option_type"] == "call").to_numpy()
    descriptors[1:, 3] = np.log(np.where(zero, root_spot, strikes) / root_spot)
    return PolicyBook(prices=prices, asset_descriptors=descriptors,
        initial_business_dtes=np.r_[0., book["bdte"].to_numpy(dtype=float)])


def fee_preconditioned_positions(raw: torch.Tensor, prices: torch.Tensor, costs: torch.Tensor,
                                *, instrument_cap: float) -> torch.Tensor:
    """Optional causal output coordinates, not a price floor or a changed risk objective.

    Premium allocation is L*tanh(raw*p/(L*(p+c))). Every positive-price
    asset retains its full (-L,L) allocation range, but this changes the
    finite shared-network strategy class. At zero raw output, current fee
    sensitivity is bounded by c/(p+c). Future fees and gains are not bounded
    by this observation. Zero prices still force liquidation. Inputs must
    be consistently normalized, finite current prices and nonnegative fees.
    """
    if (raw.shape != prices.shape or costs.shape != prices.shape
            or not raw.is_floating_point() or not prices.is_floating_point() or not costs.is_floating_point()
            or raw.device != prices.device or raw.device != costs.device
            or not np.isfinite(instrument_cap) or instrument_cap <= 0):
        raise ValueError("Fee-preconditioned outputs need aligned floating tensors and a positive premium cap")
    positive = prices > 0.
    safe_price = torch.where(positive, prices.to(torch.float64), 1.)
    coefficient = raw.to(torch.float64) / (safe_price + costs.to(torch.float64))
    argument = coefficient * safe_price / instrument_cap
    small = torch.abs(argument) < 1e-4
    small_argument = torch.where(small, argument, 0.)
    series = 1. - small_argument.square() / 3. + 2. * small_argument.pow(4) / 15.
    small_positions = torch.where(small, coefficient, 0.) * series
    direct_argument = torch.where(small, 0., argument)
    direct_price = torch.where(small, 1., safe_price)
    direct_positions = instrument_cap * torch.tanh(direct_argument) / direct_price
    positions = torch.where(small, small_positions, direct_positions)
    return torch.where(positive, positions, 0.).to(raw.dtype)


class TerminalPolicy(BaseModel):
    """One affine-plus-deep function shared across assets, dates and realized histories.

    It observes current state, contract geometry, current premium and fees,
    current own inventory and aggregate inventory. There are no path IDs,
    future returns, future fitted models or cross-path batch statistics.
    The same weights support changing books without an output-head mapping.
    Actions are capped current-premium dollars per asset, NOT raw contract
    counts. No undeclared gross-portfolio cap or economic ridge is added.
    A zero live model price forces a zero holding, including charged
    liquidation of an existing position. Small positive prices are not
    floored and can permit very large contract units; audit those tails.
    """

    def __init__(self, *, env: BaseEnvironment, config: TerminalPolicyConfig) -> None:
        super().__init__(env=env)
        self.config = config
        self.affine = FeedForward(env=env, num_input=config.feature_count + 13, num_output=0,
            config=FeedForwardConfig.linear_config())
        self.deep = (FeedForward(env=env, num_input=config.feature_count + 13, num_output=0, config=config.deep)
                     if config.use_deep else None)
        self.register_buffer("feature_mean", torch.zeros(config.feature_count, dtype=env.fdtype, device=env.device))
        self.register_buffer("feature_scale", torch.ones(config.feature_count, dtype=env.fdtype, device=env.device))
        with torch.no_grad():
            self.affine.I0.weight.zero_()
            self.affine.I0.bias.zero_()
            if self.deep is not None:
                output = self.deep.In[-1] if len(self.deep.In) else self.deep.I0
                output.weight.zero_()
                output.bias.zero_()

    def rebase_normalization(self, features: torch.Tensor, weights: torch.Tensor,
                             mapping: TerminalFeatureMapping | None = None) -> None:
        """Refit training scaling and apply an explicitly supplied raw-coordinate mapping.

        Apply this before creating a new date's optimizer. A same-run resume
        restores the saved scaling and optimizer instead of rebasing again.
        """
        if (features.ndim != 3 or features.shape[2] != self.config.feature_count
                or weights.shape != (len(features),) or torch.any(weights < 0) or weights.sum() <= 0
                or not torch.isfinite(features).all() or not torch.isfinite(weights).all()):
            raise ValueError("Policy normalization requires finite training features and positive weight mass")
        with torch.no_grad():
            probabilities = weights / weights.sum()
            mean = torch.sum(probabilities[:, None] * features.mean(dim=1), dim=0)
            variance = torch.sum(probabilities[:, None] * (features - mean).square().mean(dim=1), dim=0)
            scale = torch.sqrt(torch.clamp_min(variance, 0.))
            scale = torch.where(scale > 1e-6, scale, 1.)
            feature_map = torch.eye(self.config.feature_count, dtype=self.fdtype, device=self.device)
            offset, step_offset = torch.zeros_like(mean), 0.
            if mapping is not None:
                if mapping.matrix.shape != (self.config.feature_count, self.config.feature_count):
                    raise ValueError("Rolling policy coordinate mapping has incompatible feature geometry")
                feature_map = torch.tensor(mapping.matrix, dtype=self.fdtype, device=self.device)
                offset = torch.tensor(mapping.offset, dtype=self.fdtype, device=self.device)
                step_offset = mapping.elapsed_step_offset / 5.
            transformed = feature_map * scale[None, :] / self.feature_scale[:, None]
            shift = (feature_map @ mean + offset - self.feature_mean) / self.feature_scale
            for network in (self.affine, self.deep):
                if network is not None:
                    original = network.I0.weight[:, :self.config.feature_count].clone()
                    network.I0.bias.add_(original @ shift + network.I0.weight[:, self.config.feature_count + 1] * step_offset)
                    network.I0.weight[:, :self.config.feature_count].copy_(original @ transformed)
            self.feature_mean.copy_(mean)
            self.feature_scale.copy_(scale)

    def forward(self, features: torch.Tensor, inventory: torch.Tensor, step: int, *, book: TensorPolicyBook,
                costs: torch.Tensor) -> torch.Tensor:
        count, horizon, instruments = book.prices.shape
        if (features.shape != (count, self.config.feature_count) or inventory.shape != (count, instruments)
                or costs.shape != inventory.shape or not 0 <= step < horizon):
            raise ValueError("Terminal actor inputs must align the current state and frozen book")
        price = book.prices[:, step]
        positive = price > 0.
        safe_price = torch.where(positive, price, 1.)
        dollars = inventory * price
        cap = self.config.instrument_cap
        spot_inventory = dollars[:, :1] / cap
        net_inventory = dollars.sum(dim=1, keepdim=True) / (cap * instruments)
        gross_inventory = dollars.abs().sum(dim=1, keepdim=True) / (cap * instruments)
        normalized = (features - self.feature_mean) / self.feature_scale
        time = torch.tensor([(horizon - step) / 5., step / 5.], dtype=self.fdtype, device=self.device)
        global_features = torch.cat([normalized, time.expand(count, 2)], dim=1)
        descriptors = book.asset_descriptors.expand(count, instruments, 4).clone()
        ordinary = (descriptors[:, :, 0] == 0.) & (descriptors[:, :, 1] == 0.)
        log_spot_move = torch.log(price[:, :1] / book.prices[:, :1, 0])
        descriptors[:, :, 3] -= torch.where(ordinary, log_spot_move, 0.)
        remaining = torch.clamp_min(book.initial_business_dtes - step, 0.) / 5.
        asset_features = torch.stack([torch.log(safe_price), torch.log1p(costs / safe_price),
            dollars / cap, spot_inventory.expand_as(price), net_inventory.expand_as(price), gross_inventory.expand_as(price)], dim=2)
        actor_inputs = torch.cat([global_features[:, None, :].expand(count, instruments, -1), descriptors,
            remaining[None, :, None].expand(count, instruments, 1), asset_features], dim=2)
        raw = self.affine(actor_inputs)
        if self.deep is not None:
            raw = raw + self.deep(actor_inputs)
        if self.config.fee_preconditioning:
            positions = fee_preconditioned_positions(raw, price, costs, instrument_cap=cap)
        else:
            allocation = cap * torch.tanh(raw / cap)
            positions = torch.where(positive, allocation / safe_price, 0.)
        self.verify(lambda: torch.isfinite(positions), lambda: "Terminal policy produced non-finite holdings")
        return positions

    def bind(self, *, book: TensorPolicyBook, costs: torch.Tensor):
        """Bind exogenous current-price rows, not a future-return table, to the evaluator."""
        return lambda features, inventory, step: self(features, inventory, step, book=book, costs=costs[:, step])
