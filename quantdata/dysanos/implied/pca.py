"""Validated centered PCA of log implied-volatility surface levels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version


@dataclass(frozen=True, kw_only=True)
class ValidatedPCAMarket:
    """Arrays required to fit and simulate the direct PCA benchmark."""

    dates: np.ndarray
    time_y: np.ndarray
    mid_vols: np.ndarray
    nms: np.ndarray
    expiries: np.ndarray
    xstrikes: np.ndarray
    spot_ref: np.ndarray
    spot_ret: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.mid_vols.shape


def _floating_array(value: Any, *, name: str) -> np.ndarray:
    output = np.asarray(value)
    verify_inp(
        np.issubdtype(output.dtype, np.floating),
        lambda: f"market field {name!r} must have floating dtype, got {output.dtype}",
    )
    verify_inp(
        np.all(np.isfinite(output)),
        lambda: f"market field {name!r} contains non-finite values",
    )
    return output.astype(np.float64, copy=False)


@version("0.1")
def validate_pca_market(market: Mapping[str, Any]) -> ValidatedPCAMarket:
    """Validate the exact historical-grid inputs used by the PCA benchmark."""

    required = {
        "dates",
        "time_y",
        "mid_vols",
        "nms",
        "expiries",
        "xstrikes",
        "spot_ref",
        "spot_ret",
    }
    missing = required.difference(market)
    verify_inp(not missing, lambda: f"PCA market is missing fields {sorted(missing)}")

    dates = np.asarray(market["dates"])
    time_y = _floating_array(market["time_y"], name="time_y")
    mid_vols = _floating_array(market["mid_vols"], name="mid_vols")
    nms = _floating_array(market["nms"], name="nms")
    expiries = _floating_array(market["expiries"], name="expiries")
    xstrikes = _floating_array(market["xstrikes"], name="xstrikes")
    spot_ref = _floating_array(market["spot_ref"], name="spot_ref")
    spot_ret = _floating_array(market["spot_ret"], name="spot_ret")

    verify_inp(
        mid_vols.ndim == 3 and min(mid_vols.shape) > 0,
        lambda: f"'mid_vols' must have shape (D,T,N), got {mid_vols.shape}",
    )
    D, T, N = mid_vols.shape
    expected = {
        "dates": (D,),
        "time_y": (D,),
        "nms": (D, T, N),
        "expiries": (D, T),
        "xstrikes": (D, T, N + 2),
        "spot_ref": (D,),
        "spot_ret": (D,),
    }
    actual = {
        "dates": dates.shape,
        "time_y": time_y.shape,
        "nms": nms.shape,
        "expiries": expiries.shape,
        "xstrikes": xstrikes.shape,
        "spot_ref": spot_ref.shape,
        "spot_ret": spot_ret.shape,
    }
    for name, shape in expected.items():
        verify_inp(
            actual[name] == shape,
            lambda name=name, shape=shape: (
                f"market field {name!r} must have shape {shape}, got {actual[name]}"
            ),
        )
    verify_inp(
        np.all(mid_vols > 0.0),
        lambda: (
            "PCA input 'mid_vols' must be strictly positive; minimum is "
            f"{float(np.min(mid_vols)):g}"
        ),
    )
    verify_inp(
        np.all(spot_ref > 0.0),
        lambda: f"cash spots must be strictly positive, minimum {np.min(spot_ref):g}",
    )
    verify_inp(
        np.all(np.diff(time_y) > 0.0),
        lambda: f"'time_y' must increase strictly, got {time_y}",
    )
    verify_inp(
        np.all(expiries > 0.0) and np.all(np.diff(expiries, axis=1) > 0.0),
        "expiries must be strictly positive and increasing within every date",
    )
    verify_inp(
        np.all(np.diff(nms, axis=2) > 0.0),
        "normalized-moneyness coordinates must increase within every expiry",
    )
    verify_inp(
        np.all(np.diff(xstrikes, axis=2) > 0.0) and np.all(xstrikes > 0.0),
        "extended relative-strike grids must be positive and increasing",
    )

    # The decoder has one fixed (T,z) grid. The data route should produce this
    # exactly, but tolerate insignificant serialization roundoff.
    grid_tolerance = 64.0 * np.finfo(np.float64).eps
    verify_inp(
        np.allclose(nms, nms[:1], rtol=grid_tolerance, atol=grid_tolerance),
        "the PCA benchmark requires one common NMS grid across dates",
    )
    verify_inp(
        np.allclose(
            expiries,
            expiries[:1],
            rtol=grid_tolerance,
            atol=grid_tolerance,
        ),
        "the PCA benchmark requires one common expiry grid across dates",
    )
    return ValidatedPCAMarket(
        dates=dates,
        time_y=time_y,
        mid_vols=mid_vols,
        nms=nms,
        expiries=expiries,
        xstrikes=xstrikes,
        spot_ref=spot_ref,
        spot_ret=spot_ret,
    )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class LogIVPCA:
    """Centered, unweighted PCA representation of log-IV surface levels."""

    mean_log_iv: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    expiries: np.ndarray
    nms: np.ndarray

    def __post_init__(self) -> None:
        verify_inp(
            self.mean_log_iv.ndim == 1,
            lambda: f"mean log-IV must be one-dimensional, got {self.mean_log_iv.shape}",
        )
        H, P = self.components.shape
        verify_inp(
            self.mean_log_iv.shape == (P,),
            lambda: (
                f"component width {P} differs from mean width "
                f"{self.mean_log_iv.shape}"
            ),
        )
        verify_inp(
            self.expiries.ndim == 1
            and self.nms.ndim == 2
            and self.nms.shape[0] == self.expiries.size
            and self.nms.size == P,
            lambda: (
                "expiry/NMS grid does not match PCA width: "
                f"{self.expiries.shape}/{self.nms.shape}/{P}"
            ),
        )
        verify_inp(
            self.explained_variance.shape
            == self.explained_variance_ratio.shape
            == self.cumulative_explained_variance_ratio.shape
            == (H,),
            "PCA explained-variance arrays must contain one value per component",
        )
        for name, value in (
            ("mean_log_iv", self.mean_log_iv),
            ("components", self.components),
            ("explained_variance", self.explained_variance),
            ("explained_variance_ratio", self.explained_variance_ratio),
            (
                "cumulative_explained_variance_ratio",
                self.cumulative_explained_variance_ratio,
            ),
            ("expiries", self.expiries),
            ("nms", self.nms),
        ):
            verify_inp(
                np.issubdtype(value.dtype, np.floating)
                and np.all(np.isfinite(value)),
                lambda name=name: f"PCA field {name!r} must be finite floating data",
            )
        verify_inp(
            np.all(self.expiries > 0.0) and np.all(np.diff(self.expiries) > 0.0),
            "PCA expiries must be positive and increasing",
        )
        verify_inp(
            np.all(np.diff(self.nms, axis=1) > 0.0),
            "PCA NMS coordinates must increase within every expiry",
        )
        identity = self.components @ self.components.T
        verify_inp(
            np.allclose(identity, np.eye(H), atol=1.0e-10, rtol=1.0e-10),
            lambda: (
                "PCA component rows are not orthonormal; maximum error "
                f"{float(np.max(np.abs(identity - np.eye(H)))):g}"
            ),
        )

    @property
    def num_components(self) -> int:
        return self.components.shape[0]

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.nms.shape

    def encode(self, implied_vols: np.ndarray) -> np.ndarray:
        """Encode positive IV surfaces whose trailing axes are ``(T,N)``."""

        implied_vols = np.asarray(implied_vols)
        verify_inp(
            implied_vols.shape[-2:] == self.surface_shape,
            lambda: (
                f"IV surfaces must end in {self.surface_shape}, got "
                f"{implied_vols.shape}"
            ),
        )
        verify_inp(
            np.issubdtype(implied_vols.dtype, np.floating)
            and np.all(np.isfinite(implied_vols))
            and np.all(implied_vols > 0.0),
            "encoded implied volatilities must be finite, floating, and positive",
        )
        flat = np.log(implied_vols).reshape(*implied_vols.shape[:-2], -1)
        return (flat - self.mean_log_iv) @ self.components.T

    def decode_log_iv(self, scores: np.ndarray) -> np.ndarray:
        """Decode PCA scores to log-IV surfaces on the training grid."""

        scores = np.asarray(scores)
        verify_inp(
            scores.ndim >= 1 and scores.shape[-1] == self.num_components,
            lambda: (
                f"scores must end in {self.num_components}, got {scores.shape}"
            ),
        )
        verify_inp(
            np.issubdtype(scores.dtype, np.floating) and np.all(np.isfinite(scores)),
            "PCA scores must be finite floating data",
        )
        output = self.mean_log_iv + scores @ self.components
        return output.reshape(*scores.shape[:-1], *self.surface_shape)

    def decode(self, scores: np.ndarray) -> np.ndarray:
        """Decode PCA scores to strictly positive IV grid surfaces."""

        output = np.exp(self.decode_log_iv(scores))
        if not np.all(np.isfinite(output)) or not np.all(output > 0.0):
            raise FloatingPointError("decoded PCA implied volatility is non-finite")
        return output

    def to_state_dict(self) -> dict[str, np.ndarray]:
        """Return a cache-safe array tree."""

        return {
            "mean_log_iv": self.mean_log_iv,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_explained_variance_ratio": (
                self.cumulative_explained_variance_ratio
            ),
            "expiries": self.expiries,
            "nms": self.nms,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> LogIVPCA:
        return cls(**{name: np.asarray(value) for name, value in state.items()})


@dataclass(frozen=True, kw_only=True)
class PCAFitDiagnostics:
    """Reconstruction and eigensurface diagnostics for one PCA fit."""

    scores: np.ndarray
    reconstruction: dict[str, Any]
    errors_by_expiry: list[dict[str, float]]
    errors_by_nms_region: dict[str, dict[str, float | int]]
    first_five_iv_eigensurfaces: np.ndarray

    def report(self, model: LogIVPCA) -> dict[str, Any]:
        return {
            "explained_variance": model.explained_variance.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio.tolist(),
            "cumulative_explained_variance_ratio": (
                model.cumulative_explained_variance_ratio.tolist()
            ),
            "reconstruction": self.reconstruction,
            "errors_by_expiry": self.errors_by_expiry,
            "errors_by_nms_region": self.errors_by_nms_region,
            "first_five_iv_eigensurfaces": (
                self.first_five_iv_eigensurfaces.tolist()
            ),
            "eigensurface_definition": (
                "one-score-standard-deviation change from the mean log-IV surface"
            ),
        }


def _deterministic_component_signs(
    components: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make the largest-magnitude loading of every component positive."""

    components = components.copy()
    scores = scores.copy()
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivots])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    scores *= signs[None, :]
    return components, scores


def _error_summary(error: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


@version("0.1", dependencies=[LogIVPCA])
def fit_log_iv_pca(
    implied_vols: np.ndarray,
    *,
    expiries: np.ndarray,
    nms: np.ndarray,
    num_components: int = 20,
) -> tuple[LogIVPCA, PCAFitDiagnostics]:
    """Fit ordinary centered, unweighted PCA to log-IV surface levels."""

    implied_vols = np.asarray(implied_vols)
    expiries = np.asarray(expiries)
    nms = np.asarray(nms)
    verify_inp(
        implied_vols.ndim == 3 and min(implied_vols.shape) > 0,
        lambda: f"implied volatilities must have shape (D,T,N), got {implied_vols.shape}",
    )
    D, T, N = implied_vols.shape
    verify_inp(
        expiries.shape == (T,) and nms.shape == (T, N),
        lambda: (
            f"expiry/NMS geometry must be {(T,)}/{(T, N)}, got "
            f"{expiries.shape}/{nms.shape}"
        ),
    )
    verify_inp(
        np.issubdtype(implied_vols.dtype, np.floating)
        and np.all(np.isfinite(implied_vols))
        and np.all(implied_vols > 0.0),
        "PCA input implied volatilities must be finite, floating, and positive",
    )
    maximum_components = min(D, T * N)
    verify_inp(
        isinstance(num_components, int) and 0 < num_components <= maximum_components,
        lambda: (
            f"'num_components' must be in [1,{maximum_components}], got "
            f"{num_components!r}"
        ),
    )

    log_iv = np.log(implied_vols.astype(np.float64, copy=False)).reshape(D, T * N)
    mean_log_iv = np.mean(log_iv, axis=0)
    centered = log_iv - mean_log_iv
    left, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    scores = left[:, :num_components] * singular_values[:num_components]
    components = components[:num_components]
    components, scores = _deterministic_component_signs(components, scores)

    denominator = max(D - 1, 1)
    all_explained_variance = np.square(singular_values) / denominator
    total_variance = float(np.sum(all_explained_variance))
    verify_inp(total_variance > 0.0, "log-IV surface levels have zero total variance")
    explained_variance = all_explained_variance[:num_components]
    explained_ratio = explained_variance / total_variance
    model = LogIVPCA(
        mean_log_iv=mean_log_iv,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_ratio,
        cumulative_explained_variance_ratio=np.cumsum(explained_ratio),
        expiries=np.asarray(expiries, dtype=np.float64),
        nms=np.asarray(nms, dtype=np.float64),
    )

    reconstructed_log = model.decode_log_iv(scores)
    reconstructed_iv = np.exp(reconstructed_log)
    log_error = reconstructed_log - log_iv.reshape(D, T, N)
    iv_error = reconstructed_iv - implied_vols
    errors_by_expiry = [
        {
            "expiry": float(expiries[index]),
            **{f"iv_{name}": value for name, value in _error_summary(iv_error[:, index]).items()},
            **{
                f"log_iv_{name}": value
                for name, value in _error_summary(log_error[:, index]).items()
            },
        }
        for index in range(T)
    ]
    region_masks = {
        "left_wing_z_lt_minus_0.5": nms < -0.5,
        "center_abs_z_le_0.5": np.abs(nms) <= 0.5,
        "right_wing_z_gt_0.5": nms > 0.5,
    }
    region_errors: dict[str, dict[str, float | int]] = {}
    for name, mask in region_masks.items():
        if not np.any(mask):
            region_errors[name] = {"grid_points": 0, "iv_rmse": float("nan"), "log_iv_rmse": float("nan")}
            continue
        region_errors[name] = {
            "grid_points": int(np.count_nonzero(mask)),
            "iv_rmse": float(np.sqrt(np.mean(np.square(iv_error[:, mask])))),
            "iv_maximum_absolute_error": float(np.max(np.abs(iv_error[:, mask]))),
            "log_iv_rmse": float(np.sqrt(np.mean(np.square(log_error[:, mask])))),
            "log_iv_maximum_absolute_error": float(np.max(np.abs(log_error[:, mask]))),
        }

    component_count = min(5, num_components)
    mean_iv = np.exp(mean_log_iv.reshape(T, N))
    shifted_log = (
        mean_log_iv[None, :]
        + np.sqrt(explained_variance[:component_count])[:, None]
        * components[:component_count]
    )
    eigensurfaces = np.exp(shifted_log).reshape(component_count, T, N) - mean_iv
    diagnostics = PCAFitDiagnostics(
        scores=scores,
        reconstruction={
            **{f"iv_{name}": value for name, value in _error_summary(iv_error).items()},
            **{
                f"log_iv_{name}": value
                for name, value in _error_summary(log_error).items()
            },
        },
        errors_by_expiry=errors_by_expiry,
        errors_by_nms_region=region_errors,
        first_five_iv_eigensurfaces=eigensurfaces,
    )
    return model, diagnostics


__all__ = [
    "LogIVPCA",
    "PCAFitDiagnostics",
    "ValidatedPCAMarket",
    "fit_log_iv_pca",
    "validate_pca_market",
]
