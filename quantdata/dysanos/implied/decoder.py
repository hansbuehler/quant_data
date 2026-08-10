"""Continuous total-variance interpolation for direct PCA IV surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import NMS, NormalizedStrikes

from .pca import LogIVPCA


@dataclass(frozen=True, kw_only=True)
class WingHitDiagnostics:
    """Counts of flat normalized-moneyness extrapolation."""

    queries: int
    left_hits: int
    right_hits: int

    @property
    def total_hits(self) -> int:
        return self.left_hits + self.right_hits

    @property
    def hit_fraction(self) -> float:
        return self.total_hits / self.queries if self.queries else 0.0

    def __add__(self, other: WingHitDiagnostics) -> WingHitDiagnostics:
        return WingHitDiagnostics(
            queries=self.queries + other.queries,
            left_hits=self.left_hits + other.left_hits,
            right_hits=self.right_hits + other.right_hits,
        )


def _linear_flat(
    x_grid: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Vectorized linear interpolation with flat endpoint extrapolation."""

    verify_inp(
        x_grid.ndim == 1
        and values.ndim == 2
        and values.shape[1] == x_grid.size
        and query.ndim == 2
        and query.shape[0] == values.shape[0],
        lambda: (
            "invalid interpolation geometry "
            f"{x_grid.shape}/{values.shape}/{query.shape}"
        ),
    )
    right = np.searchsorted(x_grid, query, side="right")
    right = np.clip(right, 1, x_grid.size - 1)
    left = right - 1
    row = np.arange(values.shape[0])[:, None]
    x_left = x_grid[left]
    x_right = x_grid[right]
    weight = (query - x_left) / (x_right - x_left)
    output = values[row, left] + weight * (
        values[row, right] - values[row, left]
    )
    output = np.where(query <= x_grid[0], values[:, :1], output)
    output = np.where(query >= x_grid[-1], values[:, -1:], output)
    return output


@version("0.1", dependencies=[LogIVPCA])
class PCATotalVarianceDecoder:
    """Decode and interpolate a PCA state as a fixed-surface price map.

    Total variance is linearly interpolated first in normalized moneyness and
    then in maturity after adding ``w(0,z)=0``. The normalized-moneyness wings
    are flat. Maturities outside ``[0, max(training_expiry)]`` are rejected.

    Relative strikes use :class:`NormalizedStrikes` with the interpolated ATM
    volatility at the query maturity. The epsilon-volatility scale is the ATM
    volatility at the shortest decoded expiry, matching how the historical
    regular grid is constructed.
    """

    def __init__(
        self,
        pca: LogIVPCA,
        *,
        normalized_strikes: NormalizedStrikes | None = None,
    ) -> None:
        self.pca = pca
        self.normalized_strikes = (
            NormalizedStrikes(EPS=NMS.EPS, SCALE=NMS.SCALE)
            if normalized_strikes is None
            else normalized_strikes
        )
        verify_inp(
            isinstance(self.normalized_strikes, NormalizedStrikes),
            lambda: (
                "'normalized_strikes' must be a NormalizedStrikes instance, got "
                f"{type(self.normalized_strikes)}"
            ),
        )

    def _scores_2d(self, scores: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        scores = np.asarray(scores)
        verify_inp(
            scores.ndim >= 1
            and scores.shape[-1] == self.pca.num_components
            and np.issubdtype(scores.dtype, np.floating)
            and np.all(np.isfinite(scores)),
            lambda: (
                f"scores must be finite floating arrays ending in "
                f"{self.pca.num_components}, got {scores.shape}/{scores.dtype}"
            ),
        )
        batch_shape = scores.shape[:-1]
        return scores.reshape(-1, self.pca.num_components), batch_shape

    @staticmethod
    def _queries_2d(
        values: tuple[np.ndarray | float, ...],
        *,
        batch_shape: tuple[int, ...],
    ) -> tuple[list[np.ndarray], tuple[int, ...]]:
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
        verify_inp(
            all(np.all(np.isfinite(value)) for value in arrays),
            "decoder queries must be finite",
        )
        arrays = list(np.broadcast_arrays(*arrays))
        common_shape = arrays[0].shape
        batch_size = math.prod(batch_shape) if batch_shape else 1

        # A query is batch-specific only when its broadcast shape contains the
        # complete score-batch prefix and at least one additional query axis.
        batch_specific = (
            bool(batch_shape)
            and len(common_shape) > len(batch_shape)
            and common_shape[: len(batch_shape)] == batch_shape
        )
        if batch_specific:
            query_shape = common_shape[len(batch_shape) :]
            matrices = [value.reshape(batch_size, -1) for value in arrays]
        else:
            query_shape = common_shape
            matrices = [
                np.broadcast_to(value, (*batch_shape, *query_shape)).reshape(
                    batch_size, -1
                )
                for value in arrays
            ]
        return matrices, query_shape

    def grid_implied_vols(self, scores: np.ndarray) -> np.ndarray:
        return self.pca.decode(scores)

    def _grid_total_variance(self, scores_2d: np.ndarray) -> np.ndarray:
        vols = self.pca.decode(scores_2d)
        return np.square(vols) * self.pca.expiries[None, :, None]

    def _atm_terms(
        self,
        grid_total_variance: np.ndarray,
        maturities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return query ATM vol and shortest-expiry epsilon ATM vol."""

        batch_size, expiry_count, _ = grid_total_variance.shape
        atm_grid = np.empty((batch_size, expiry_count), dtype=np.float64)
        zero = np.zeros((batch_size, 1), dtype=np.float64)
        for expiry_index in range(expiry_count):
            atm_grid[:, expiry_index] = _linear_flat(
                self.pca.nms[expiry_index],
                grid_total_variance[:, expiry_index],
                zero,
            )[:, 0]
        atm_w = self._interpolate_time(
            np.broadcast_to(
                atm_grid[:, :, None],
                (batch_size, expiry_count, maturities.shape[1]),
            ),
            maturities,
        )
        positive_t = maturities > 0.0
        atm_vol = np.empty_like(atm_w)
        atm_vol[positive_t] = np.sqrt(atm_w[positive_t] / maturities[positive_t])
        shortest_atm = np.sqrt(atm_grid[:, 0] / self.pca.expiries[0])
        atm_vol[~positive_t] = np.broadcast_to(shortest_atm[:, None], atm_vol.shape)[
            ~positive_t
        ]
        verify_inp(
            np.all(np.isfinite(atm_vol))
            and np.all(atm_vol > 0.0)
            and np.all(shortest_atm > 0.0),
            "interpolated ATM volatility is non-finite or non-positive",
        )
        return atm_vol, shortest_atm[:, None]

    def _interpolate_time(
        self,
        values: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """Interpolate ``(B,T,Q)`` values from the zero anchor in maturity."""

        batch_size, expiry_count, query_count = values.shape
        verify_inp(
            maturities.shape == (batch_size, query_count),
            lambda: (
                f"maturities must have shape {(batch_size, query_count)}, got "
                f"{maturities.shape}"
            ),
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(self.pca.expiries[-1])
        )
        verify_inp(
            np.all(maturities >= -tolerance)
            and np.all(maturities <= self.pca.expiries[-1] + tolerance),
            lambda: (
                "maturities must lie inside the decoder domain "
                f"[0,{self.pca.expiries[-1]:g}], got range "
                f"[{np.min(maturities):g},{np.max(maturities):g}]"
            ),
        )
        maturities = np.clip(maturities, 0.0, self.pca.expiries[-1])
        high = np.searchsorted(self.pca.expiries, maturities, side="left")
        high = np.clip(high, 0, expiry_count - 1)
        output = np.empty_like(maturities)
        row, column = np.indices(maturities.shape)
        high_t = self.pca.expiries[high]
        high_value = values[row, high, column]
        first = high == 0
        low_t = np.where(first, 0.0, self.pca.expiries[np.maximum(high - 1, 0)])
        low_value = np.where(first, 0.0, values[row, np.maximum(high - 1, 0), column])
        weight = np.divide(
            maturities - low_t,
            high_t - low_t,
            out=np.zeros_like(maturities),
            where=high_t > low_t,
        )
        output[:] = low_value + weight * (high_value - low_value)
        return output

    def total_variance(
        self,
        scores: np.ndarray,
        maturities: np.ndarray | float,
        normalized_moneyness: np.ndarray | float,
    ) -> tuple[np.ndarray, WingHitDiagnostics]:
        """Interpolate decoded total variance at arbitrary ``(T,z)`` queries."""

        scores_2d, batch_shape = self._scores_2d(scores)
        (maturity_2d, nms_2d), query_shape = self._queries_2d(
            (maturities, normalized_moneyness),
            batch_shape=batch_shape,
        )
        grid_w = self._grid_total_variance(scores_2d)
        batch_size, expiry_count, _ = grid_w.shape
        query_count = maturity_2d.shape[1]
        by_expiry = np.empty(
            (batch_size, expiry_count, query_count), dtype=np.float64
        )
        for expiry_index in range(expiry_count):
            by_expiry[:, expiry_index] = _linear_flat(
                self.pca.nms[expiry_index],
                grid_w[:, expiry_index],
                nms_2d,
            )
        output = self._interpolate_time(by_expiry, maturity_2d)
        if not np.all(np.isfinite(output)) or np.any(output < 0.0):
            raise FloatingPointError("interpolated PCA total variance is invalid")
        minimum_nms = float(np.min(self.pca.nms))
        maximum_nms = float(np.max(self.pca.nms))
        wing_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, abs(minimum_nms), abs(maximum_nms)
        )
        diagnostics = WingHitDiagnostics(
            queries=output.size,
            left_hits=int(
                np.count_nonzero(nms_2d < minimum_nms - wing_tolerance)
            ),
            right_hits=int(
                np.count_nonzero(nms_2d > maximum_nms + wing_tolerance)
            ),
        )
        return output.reshape(*batch_shape, *query_shape), diagnostics

    def relative_strikes_from_nms(
        self,
        scores: np.ndarray,
        maturities: np.ndarray | float,
        normalized_moneyness: np.ndarray | float,
    ) -> np.ndarray:
        """Convert NMS coordinates to relative strikes using decoded ATM IV."""

        scores_2d, batch_shape = self._scores_2d(scores)
        (maturity_2d, nms_2d), query_shape = self._queries_2d(
            (maturities, normalized_moneyness), batch_shape=batch_shape
        )
        grid_w = self._grid_total_variance(scores_2d)
        atm_vol, eps_vol = self._atm_terms(grid_w, maturity_2d)
        output = self.normalized_strikes.from_nms(
            z=nms_2d,
            t=maturity_2d,
            vol=atm_vol,
            eps_vol=np.broadcast_to(eps_vol, atm_vol.shape),
        )
        if not np.all(np.isfinite(output)) or not np.all(output > 0.0):
            raise FloatingPointError("NMS conversion produced invalid relative strikes")
        return output.reshape(*batch_shape, *query_shape)

    def nms_from_relative_strikes(
        self,
        scores: np.ndarray,
        maturities: np.ndarray | float,
        relative_strikes: np.ndarray | float,
    ) -> np.ndarray:
        """Convert relative strikes to NMS using the decoded ATM IV."""

        scores_2d, batch_shape = self._scores_2d(scores)
        (maturity_2d, strike_2d), query_shape = self._queries_2d(
            (maturities, relative_strikes), batch_shape=batch_shape
        )
        verify_inp(
            np.all(strike_2d > 0.0),
            "relative strikes must be strictly positive",
        )
        grid_w = self._grid_total_variance(scores_2d)
        atm_vol, eps_vol = self._atm_terms(grid_w, maturity_2d)
        output = self.normalized_strikes.to_nms(
            k=strike_2d,
            t=maturity_2d,
            vol=atm_vol,
            eps_vol=np.broadcast_to(eps_vol, atm_vol.shape),
        )
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("relative-strike conversion produced invalid NMS")
        return output.reshape(*batch_shape, *query_shape)

    def implied_volatility(
        self,
        scores: np.ndarray,
        maturities: np.ndarray | float,
        relative_strikes: np.ndarray | float,
    ) -> tuple[np.ndarray, WingHitDiagnostics]:
        """Evaluate positive implied volatility at arbitrary fixed strikes."""

        maturities_array = np.asarray(maturities, dtype=np.float64)
        verify_inp(
            np.all(maturities_array > 0.0),
            "implied volatility is undefined at zero maturity",
        )
        nms = self.nms_from_relative_strikes(scores, maturities, relative_strikes)
        total_variance, diagnostics = self.total_variance(scores, maturities, nms)
        maturity = np.broadcast_to(maturities_array, total_variance.shape)
        output = np.sqrt(total_variance / maturity)
        if not np.all(np.isfinite(output)) or not np.all(output > 0.0):
            raise FloatingPointError("interpolated PCA implied volatility is invalid")
        return output, diagnostics

    def normalized_call_prices(
        self,
        scores: np.ndarray,
        maturities: np.ndarray | float,
        relative_strikes: np.ndarray | float,
    ) -> tuple[np.ndarray, WingHitDiagnostics]:
        """Price zero-rate normalized calls from the complete PCA surface."""

        scores_2d, batch_shape = self._scores_2d(scores)
        (maturity_2d, strike_2d), query_shape = self._queries_2d(
            (maturities, relative_strikes), batch_shape=batch_shape
        )
        verify_inp(
            np.all(maturity_2d >= 0.0) and np.all(strike_2d > 0.0),
            "call maturities must be nonnegative and relative strikes positive",
        )
        output = np.maximum(1.0 - strike_2d, 0.0)
        diagnostics = WingHitDiagnostics(queries=0, left_hits=0, right_hits=0)
        positive = maturity_2d > 0.0
        if np.any(positive):
            nms = self.nms_from_relative_strikes(
                scores_2d,
                maturity_2d,
                strike_2d,
            ).reshape(maturity_2d.shape)
            total_variance, diagnostics = self.total_variance(
                scores_2d,
                maturity_2d,
                nms,
            )
            total_variance = total_variance.reshape(maturity_2d.shape)
            vol = np.zeros_like(total_variance)
            vol[positive] = np.sqrt(total_variance[positive] / maturity_2d[positive])
            output[positive] = bs(
                k=strike_2d[positive],
                vol=vol[positive],
                sqrtT=np.sqrt(maturity_2d[positive]),
                what=bs.PRICE,
            )
        intrinsic = np.maximum(1.0 - strike_2d, 0.0)
        tolerance = 32.0 * np.finfo(output.dtype).eps * np.maximum(
            1.0, np.abs(output) + np.abs(intrinsic)
        )
        if (
            not np.all(np.isfinite(output))
            or np.any(output < intrinsic - tolerance)
            or np.any(output > 1.0 + tolerance)
        ):
            raise FloatingPointError("Black--Scholes call price violates finite bounds")
        return output.reshape(*batch_shape, *query_shape), diagnostics

    def cash_call_prices(
        self,
        scores: np.ndarray,
        spots: np.ndarray | float,
        maturities: np.ndarray | float,
        cash_strikes: np.ndarray | float,
    ) -> tuple[np.ndarray, WingHitDiagnostics]:
        """Price fixed cash-strike contracts as ``S*c(T,k/S)``."""

        spots = np.asarray(spots, dtype=np.float64)
        cash_strikes = np.asarray(cash_strikes, dtype=np.float64)
        verify_inp(
            np.all(np.isfinite(spots))
            and np.all(np.isfinite(cash_strikes))
            and np.all(spots > 0.0)
            and np.all(cash_strikes > 0.0),
            "cash spots and strikes must be finite and strictly positive",
        )
        prices, diagnostics = self.normalized_call_prices(
            scores,
            maturities,
            cash_strikes / spots,
        )
        return prices * np.broadcast_to(spots, prices.shape), diagnostics

    def cash_deltas(
        self,
        scores: np.ndarray,
        spots: np.ndarray | float,
        maturities: np.ndarray | float,
        cash_strikes: np.ndarray | float,
        *,
        relative_step: float = 1.0e-5,
    ) -> tuple[np.ndarray, WingHitDiagnostics]:
        """Differentiate the complete fixed-surface cash price map centrally."""

        verify_inp(
            math.isfinite(relative_step) and 0.0 < relative_step < 1.0,
            lambda: f"'relative_step' must lie in (0,1), got {relative_step}",
        )
        spots = np.asarray(spots, dtype=np.float64)
        upper_spot = spots * (1.0 + relative_step)
        lower_spot = spots * (1.0 - relative_step)
        upper, upper_hits = self.cash_call_prices(
            scores, upper_spot, maturities, cash_strikes
        )
        lower, lower_hits = self.cash_call_prices(
            scores, lower_spot, maturities, cash_strikes
        )
        denominator = 2.0 * relative_step * np.broadcast_to(spots, upper.shape)
        delta = (upper - lower) / denominator
        if not np.all(np.isfinite(delta)):
            raise FloatingPointError("fixed-surface cash delta is non-finite")
        return delta, upper_hits + lower_hits


__all__ = ["PCATotalVarianceDecoder", "WingHitDiagnostics"]
