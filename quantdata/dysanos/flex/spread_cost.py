"""Date-balanced half-spread attribution for the Phase 3 option universe."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch import Tensor

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

LINEAR_FEATURE_NAMES = ("vega_over_sqrt_tte", "delta", "nms")
ML_FEATURE_NAMES = ("vega", "sqrt_tte", "nms", "log_strike", "delta")
TORCH_ML_FEATURE_NAMES = (*ML_FEATURE_NAMES, "log_option_price_dollars")


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class SpreadAttributionConfig:
    """Ad-hoc, deliberately small attribution-model hyperparameters."""

    label: str = "phase3-spread-v1"
    group: str = "dysanos/flex/phase3-spread"
    user_version: str = "0.1"
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 100
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 100
    hgb_l2_regularization: float = 1.0e-6
    hgb_loss: str = "squared_error"
    random_seed: int = 20260819
    proportional_price_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        verify_inp(
            self.hgb_learning_rate > 0.0,
            lambda: (
                "'hgb_learning_rate' must be positive, found "
                f"{self.hgb_learning_rate}"
            ),
        )
        verify_inp(
            self.hgb_max_iter > 0,
            lambda: f"'hgb_max_iter' must be positive, found {self.hgb_max_iter}",
        )
        verify_inp(
            self.hgb_max_leaf_nodes >= 2,
            lambda: (
                "'hgb_max_leaf_nodes' must be at least two, found "
                f"{self.hgb_max_leaf_nodes}"
            ),
        )
        verify_inp(
            self.hgb_min_samples_leaf > 0,
            lambda: (
                "'hgb_min_samples_leaf' must be positive, found "
                f"{self.hgb_min_samples_leaf}"
            ),
        )
        verify_inp(
            self.hgb_l2_regularization >= 0.0,
            lambda: (
                "'hgb_l2_regularization' must be non-negative, found "
                f"{self.hgb_l2_regularization}"
            ),
        )
        verify_inp(
            self.hgb_loss in {"squared_error", "absolute_error"},
            lambda: (
                "'hgb_loss' must be 'squared_error' or 'absolute_error', found "
                f"{self.hgb_loss!r}"
            ),
        )
        verify_inp(
            self.proportional_price_floor > 0.0,
            lambda: (
                "'proportional_price_floor' must be positive, found "
                f"{self.proportional_price_floor}"
            ),
        )


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class SpreadFeedForwardConfig:
    """Small differentiable half-spread model and explicit SPX tick policy.

    The tick schedule and low-premium input filter are deliberately exposed:
    they are practical Phase 3 assumptions, not learned market microstructure.
    """

    label: str = "phase3-spread-ffw-v1"
    group: str = "dysanos/flex/phase3-spread"
    user_version: str = "0.1"
    network: FeedForwardConfig = field(
        default_factory=lambda: FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=64,
            use_bias=True,
        )
    )
    minimum_option_price_dollars: float = 0.25
    low_premium_threshold_dollars: float = 3.0
    low_premium_tick_dollars: float = 0.05
    high_premium_tick_dollars: float = 0.10
    proportional_price_floor: float = 1.0e-6
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 65_536
    num_epochs: int = 100
    gradient_clip: float = 10.0
    random_seed: int = 20260819

    def __post_init__(self) -> None:
        positive = {
            "minimum_option_price_dollars": self.minimum_option_price_dollars,
            "low_premium_threshold_dollars": self.low_premium_threshold_dollars,
            "low_premium_tick_dollars": self.low_premium_tick_dollars,
            "high_premium_tick_dollars": self.high_premium_tick_dollars,
            "proportional_price_floor": self.proportional_price_floor,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "gradient_clip": self.gradient_clip,
        }
        verify_inp(
            all(value > 0 for value in positive.values()),
            lambda: f"Spread feed-forward parameters must be positive: {positive}",
        )
        verify_inp(
            self.high_premium_tick_dollars >= self.low_premium_tick_dollars,
            lambda: "The high-premium tick cannot be below the low-premium tick",
        )
        verify_inp(
            self.weight_decay >= 0.0,
            lambda: f"'weight_decay' must be non-negative, found {self.weight_decay}",
        )


@dataclass(frozen=True)
class SpreadObservations:
    """One date's eligible normalized books and smooth-reference Greeks."""

    trading_date: date
    half_spread: np.ndarray
    fit_price: np.ndarray
    business_dte: np.ndarray
    sqrt_tte: np.ndarray
    strike: np.ndarray
    log_strike: np.ndarray
    nms: np.ndarray
    implied_vol: np.ndarray
    vega: np.ndarray
    delta: np.ndarray
    is_call: np.ndarray
    spot_ref: float

    def __post_init__(self) -> None:
        Q = len(self.half_spread)
        arrays = (
            self.fit_price,
            self.business_dte,
            self.sqrt_tte,
            self.strike,
            self.log_strike,
            self.nms,
            self.implied_vol,
            self.vega,
            self.delta,
            self.is_call,
        )
        if Q == 0 or any(value.shape != (Q,) for value in arrays):
            raise ValueError("Spread observations must be non-empty aligned vectors")
        finite = (self.half_spread, *arrays[:-1])
        if not np.all(np.concatenate([np.isfinite(value) for value in finite])):
            raise ValueError("Spread observations must be finite")
        if not np.all(self.half_spread > 0.0):
            raise ValueError("Observed half-spreads must be positive")
        if not np.all(self.fit_price >= 0.0):
            raise ValueError("Smooth fit prices must be non-negative")
        if not np.all(self.business_dte > 0.0):
            raise ValueError("Zero-time books are outside the spread benchmark")
        if not np.all(self.strike > 0.0):
            raise ValueError("The spread attribution requires positive strikes")
        if not np.isfinite(self.spot_ref) or self.spot_ref <= 0.0:
            raise ValueError("Spread observations require a positive spot reference")

    @property
    def num_quotes(self) -> int:
        """Number of call and put books retained on this date."""
        return len(self.half_spread)

    @property
    def vega_over_sqrt_tte(self) -> np.ndarray:
        """Black vega divided by square-root time, equal to normal density at d1."""
        return self.vega / self.sqrt_tte

    @property
    def linear_features(self) -> np.ndarray:
        """Linear attribution design, excluding the estimator's intercept."""
        return np.column_stack((self.vega_over_sqrt_tte, self.delta, self.nms))

    @property
    def ml_features(self) -> np.ndarray:
        """Small nonlinear attribution design."""
        return np.column_stack(
            (self.vega, self.sqrt_tte, self.nms, self.log_strike, self.delta)
        )


def observations_from_market_day(
    day: FlexMarketDay,
    *,
    first_model_business_dte: float = 4.8,
) -> SpreadObservations:
    """Derive zero-carry Black Greeks from the smooth AF fit on one date.

    The AF fit is converted to a call through normalized put-call parity before
    implied-volatility inversion. Delta is side-specific forward delta:
    ``Phi(d1)`` for calls and ``Phi(d1)-1`` for puts. Vega is the derivative of
    normalized option price with respect to one unit of annualized volatility.
    Books with exactly zero time are excluded; Phase 3 does not trade 0DTE.
    """
    verify_inp(
        first_model_business_dte > 0.0,
        lambda: (
            "'first_model_business_dte' must be positive, found "
            f"{first_model_business_dte}"
        ),
    )
    quote_tte = day.expiries_y[day.expiry_index]
    active = quote_tte > 0.0
    if not np.any(active):
        raise ValueError(f"No positive-time books for {day.trading_date}")
    if day.spot_ref is None:
        raise ValueError(f"No spot reference for {day.trading_date}")

    tte = np.asarray(quote_tte[active], dtype=np.float64)
    sqrt_tte = np.sqrt(tte)
    strike = np.asarray(day.strikes[active], dtype=np.float64)
    log_strike = np.log(strike)
    call_fit = np.asarray(day.call_equivalent_fit_prices[active], dtype=np.float64)
    intrinsic = np.maximum(1.0 - strike, 0.0)
    call_fit = np.clip(call_fit, intrinsic, 1.0)
    implied_vol = np.asarray(
        bs.implied(
            k=strike,
            prices=call_fit,
            sqrtT=sqrt_tte,
            on_exceed_bounds="quiet",
        ),
        dtype=np.float64,
    )
    total_vol = np.maximum(implied_vol * sqrt_tte, 1.0e-12)
    d1 = -log_strike / total_vol + 0.5 * total_vol
    call_delta = ndtr(d1)
    is_call = np.asarray(day.is_call[active], dtype=bool)
    delta = call_delta - (~is_call).astype(np.float64)
    vega = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi) * sqrt_tte

    expiry_index = day.expiry_index[active]
    quote_atm_vol = day.atm_annual_vols[expiry_index]
    # Recover epsilon volatility from the first fixed FLEX tenor. The explicit
    # form below avoids importing dynamics code into this independent model.
    first_model_tte = first_model_business_dte / DAYS_PER_YEAR
    first_model_atm_vol = day.model_atm_total_vols[0] / np.sqrt(first_model_tte)
    nms = np.asarray(
        NMS.to_nms(
            strike,
            tte,
            vol=quote_atm_vol,
            eps_vol=float(first_model_atm_vol),
        ),
        dtype=np.float64,
    )
    return SpreadObservations(
        trading_date=day.trading_date,
        half_spread=np.asarray(day.half_spreads[active], dtype=np.float64),
        fit_price=np.asarray(day.fit_prices[active], dtype=np.float64),
        business_dte=tte * DAYS_PER_YEAR,
        sqrt_tte=sqrt_tte,
        strike=strike,
        log_strike=log_strike,
        nms=nms,
        implied_vol=implied_vol,
        vega=vega,
        delta=delta,
        is_call=is_call,
        spot_ref=day.spot_ref,
    )


def equal_date_sample_weights(date_index: np.ndarray) -> np.ndarray:
    """Return mean-one quote weights assigning equal total mass to each date."""
    values = np.asarray(date_index)
    verify_inp(
        values.ndim == 1 and values.size > 0,
        lambda: f"'date_index' must be a non-empty vector, found {values.shape}",
    )
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    num_quotes = len(values)
    num_dates = len(counts)
    weights = num_quotes / (num_dates * counts[inverse].astype(np.float64))
    assert np.isclose(np.sum(weights), num_quotes)
    return weights


@dataclass(frozen=True)
class FittedSpreadCostModel:
    """Simulation-facing wrapper around a fitted normalized half-spread model."""

    name: str
    estimator: Any
    feature_names: tuple[str, ...]
    proportional_price_floor: float = 1.0e-6

    def _design(
        self,
        *,
        vega: np.ndarray,
        sqrt_tte: np.ndarray,
        nms: np.ndarray,
        log_strike: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        values = {
            "vega": np.asarray(vega, dtype=np.float64),
            "sqrt_tte": np.asarray(sqrt_tte, dtype=np.float64),
            "nms": np.asarray(nms, dtype=np.float64),
            "log_strike": np.asarray(log_strike, dtype=np.float64),
            "delta": np.asarray(delta, dtype=np.float64),
        }
        values["vega_over_sqrt_tte"] = values["vega"] / np.maximum(
            values["sqrt_tte"], 1.0e-12
        )
        arrays = np.broadcast_arrays(*(values[name] for name in self.feature_names))
        design = np.column_stack([value.reshape(-1) for value in arrays])
        if not np.all(np.isfinite(design)):
            raise ValueError("Spread-cost features must be finite")
        return design

    def predict_half_spread(
        self,
        *,
        vega: np.ndarray,
        sqrt_tte: np.ndarray,
        nms: np.ndarray,
        log_strike: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Predict non-negative half-spread in normalized price units."""
        design = self._design(
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        prediction = np.asarray(self.estimator.predict(design), dtype=np.float64)
        return np.maximum(prediction, 0.0)

    def predict_proportional_cost(
        self,
        option_price: np.ndarray,
        *,
        vega: np.ndarray,
        sqrt_tte: np.ndarray,
        nms: np.ndarray,
        log_strike: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Predict ``half_spread / option_price`` for strictly positive strikes.

        Zero-strike calls represent spot and deliberately bypass this model;
        their proportional cost is a separate simulator configuration.
        """
        price = np.asarray(option_price, dtype=np.float64)
        if not np.all(np.isfinite(price) & (price >= 0.0)):
            raise ValueError("Option prices must be finite and non-negative")
        half_spread = self.predict_half_spread(
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        return half_spread / np.maximum(price.reshape(-1), self.proportional_price_floor)

    def predict_half_spread_torch(
        self,
        *,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Evaluate the fitted linear attribution inside a Torch simulation.

        The sklearn tree benchmark is intentionally not exposed here: its hard
        splits would not provide useful state gradients. The linear expression
        preserves gradients through every feature and requires no CPU transfer.
        """
        if self.feature_names != LINEAR_FEATURE_NAMES or not isinstance(
            self.estimator, LinearRegression
        ):
            raise ValueError("Torch inference is supported only by the linear model")
        values = {
            "vega": vega,
            "sqrt_tte": sqrt_tte,
            "nms": nms,
            "log_strike": log_strike,
            "delta": delta,
        }
        values["vega_over_sqrt_tte"] = values["vega"] / torch.clamp_min(
            values["sqrt_tte"], 1.0e-12
        )
        design = torch.stack(
            torch.broadcast_tensors(*(values[name] for name in self.feature_names)),
            dim=-1,
        )
        coefficients = torch.as_tensor(
            self.estimator.coef_, dtype=design.dtype, device=design.device
        )
        intercept = torch.as_tensor(
            self.estimator.intercept_, dtype=design.dtype, device=design.device
        )
        return torch.clamp_min(design @ coefficients + intercept, 0.0)

    def predict_proportional_cost_torch(
        self,
        option_price: Tensor,
        *,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Torch-native linear half-spread divided by current option price."""
        half_spread = self.predict_half_spread_torch(
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        return half_spread / torch.clamp_min(
            option_price, self.proportional_price_floor
        )


class TorchSpreadCostModel(BaseModel):
    """Differentiable feed-forward prediction of normalized price half-spread."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: SpreadFeedForwardConfig,
        feature_mean: Tensor,
        feature_scale: Tensor,
        log_target_mean: Tensor,
        log_target_scale: Tensor,
    ):
        super().__init__(env)
        num_features = len(TORCH_ML_FEATURE_NAMES)
        if feature_mean.shape != (num_features,) or feature_scale.shape != (
            num_features,
        ):
            raise ValueError("Spread feature normalizers have the wrong shape")
        if log_target_mean.numel() != 1 or log_target_scale.numel() != 1:
            raise ValueError("Spread target normalizers must be scalar")
        if torch.any(feature_scale <= 0.0) or torch.any(log_target_scale <= 0.0):
            raise ValueError("Spread normalizer scales must be positive")
        self.config = config
        self.register_buffer(
            "feature_mean",
            feature_mean.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "feature_scale",
            feature_scale.to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "log_target_mean",
            log_target_mean.reshape(()).to(device=env.device, dtype=env.fdtype),
        )
        self.register_buffer(
            "log_target_scale",
            log_target_scale.reshape(()).to(device=env.device, dtype=env.fdtype),
        )
        self.network = FeedForward(
            env,
            num_input=num_features,
            num_output=0,
            config=config.network,
        )

    def forward(
        self,
        option_price: Tensor,
        spot: Tensor,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Delegate module calls to tick-floored half-spread inference."""
        return self.predict_half_spread_torch(
            option_price,
            spot,
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )

    def _features(
        self,
        option_price: Tensor,
        spot: Tensor,
        *,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        values = torch.broadcast_tensors(
            vega,
            sqrt_tte,
            nms,
            log_strike,
            delta,
            option_price,
            spot,
        )
        premium_dollars = torch.clamp_min(values[5] * values[6], 1.0e-12)
        features = torch.stack((*values[:5], torch.log(premium_dollars)), dim=-1)
        return (features - self.feature_mean) / self.feature_scale

    def minimum_half_tick(
        self,
        option_price: Tensor,
        spot: Tensor,
    ) -> Tensor:
        """Return the configured minimum half-tick in normalized price units."""
        price, spot_value = torch.broadcast_tensors(option_price, spot)
        premium_dollars = price * spot_value
        tick_dollars = torch.where(
            premium_dollars < self.config.low_premium_threshold_dollars,
            torch.full_like(premium_dollars, self.config.low_premium_tick_dollars),
            torch.full_like(premium_dollars, self.config.high_premium_tick_dollars),
        )
        return 0.5 * tick_dollars / spot_value

    def predict_log_half_spread_torch(
        self,
        option_price: Tensor,
        spot: Tensor,
        *,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Predict unconstrained log half-spread before applying the tick floor."""
        features = self._features(
            option_price,
            spot,
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        standardized = self.network(features)
        return self.log_target_mean + self.log_target_scale * standardized

    def predict_half_spread_torch(
        self,
        option_price: Tensor,
        spot: Tensor,
        *,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Predict normalized half-spread with the configured minimum tick."""
        log_prediction = self.predict_log_half_spread_torch(
            option_price,
            spot,
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        prediction = torch.exp(torch.clamp(log_prediction, min=-30.0, max=5.0))
        return torch.maximum(prediction, self.minimum_half_tick(option_price, spot))

    def predict_proportional_cost_torch(
        self,
        option_price: Tensor,
        *,
        spot: Tensor,
        vega: Tensor,
        sqrt_tte: Tensor,
        nms: Tensor,
        log_strike: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Return differentiable half-spread divided by normalized option price."""
        half_spread = self.predict_half_spread_torch(
            option_price,
            spot,
            vega=vega,
            sqrt_tte=sqrt_tte,
            nms=nms,
            log_strike=log_strike,
            delta=delta,
        )
        return half_spread / torch.clamp_min(
            option_price, self.config.proportional_price_floor
        )


def fit_spread_cost_models(
    observations: tuple[SpreadObservations, ...],
    *,
    config: SpreadAttributionConfig = SpreadAttributionConfig(),
) -> dict[str, FittedSpreadCostModel]:
    """Fit weighted linear and light nonlinear half-spread models."""
    if len(observations) == 0:
        raise ValueError("At least one spread-observation date is required")
    target = np.concatenate([value.half_spread for value in observations])
    date_index = np.concatenate(
        [np.full(value.num_quotes, index) for index, value in enumerate(observations)]
    )
    sample_weight = equal_date_sample_weights(date_index)
    linear_features = np.concatenate(
        [value.linear_features for value in observations], axis=0
    )
    ml_features = np.concatenate([value.ml_features for value in observations], axis=0)

    linear = LinearRegression(fit_intercept=True)
    linear.fit(linear_features, target, sample_weight=sample_weight)
    hgb = HistGradientBoostingRegressor(
        loss=config.hgb_loss,
        learning_rate=config.hgb_learning_rate,
        max_iter=config.hgb_max_iter,
        max_leaf_nodes=config.hgb_max_leaf_nodes,
        min_samples_leaf=config.hgb_min_samples_leaf,
        l2_regularization=config.hgb_l2_regularization,
        random_state=config.random_seed,
    )
    hgb.fit(ml_features, target, sample_weight=sample_weight)
    return {
        "linear": FittedSpreadCostModel(
            name="linear",
            estimator=linear,
            feature_names=LINEAR_FEATURE_NAMES,
            proportional_price_floor=config.proportional_price_floor,
        ),
        "hist_gradient_boosting": FittedSpreadCostModel(
            name="hist_gradient_boosting",
            estimator=hgb,
            feature_names=ML_FEATURE_NAMES,
            proportional_price_floor=config.proportional_price_floor,
        ),
    }


def save_spread_cost_model(model: FittedSpreadCostModel, path: Path) -> None:
    """Serialize a fitted sklearn wrapper for Phase 3 simulation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_spread_cost_model(path: Path) -> FittedSpreadCostModel:
    """Load a fitted spread-cost wrapper."""
    value = joblib.load(path)
    if not isinstance(value, FittedSpreadCostModel):
        raise ValueError(f"Not a FittedSpreadCostModel checkpoint: {path}")
    return value


def save_torch_spread_cost_model(
    model: TorchSpreadCostModel,
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save a portable Torch spread-cost checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "config": model.config,
            "model_state_dict": model.state_dict(),
            "metadata": {} if metadata is None else metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_torch_spread_cost_model(
    path: Path,
    *,
    env: BaseEnvironment,
) -> tuple[TorchSpreadCostModel, dict[str, object]]:
    """Restore a Torch spread model on the requested simulation environment."""
    # Some cached/report filesystems expose a stale handle to PyTorch's C++
    # zip reader after an atomic replace even though ordinary reads succeed.
    # This checkpoint is tiny, so a bytes-backed reader is both robust and
    # avoids any filesystem-specific mmap/open behavior.
    payload = torch.load(
        BytesIO(path.read_bytes()),
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("config"), SpreadFeedForwardConfig
    ):
        raise ValueError(f"Not a supported Torch spread-cost checkpoint: {path}")
    state_dict = payload["model_state_dict"]
    model = TorchSpreadCostModel(
        env,
        config=payload["config"],
        feature_mean=state_dict["feature_mean"],
        feature_scale=state_dict["feature_scale"],
        log_target_mean=state_dict["log_target_mean"],
        log_target_scale=state_dict["log_target_scale"],
    )
    model.load_state_dict(state_dict)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Torch spread-cost checkpoint metadata must be a dictionary")
    return model, metadata
