"""Surface-first exact Gaussian simulator for the direct PCA state."""

from __future__ import annotations

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.dysanos.grid.torch_simulator import (
    AR1PCA,
    ARSampler,
    ARSamplerConfig,
    fit_surface_first_xar1_pca,
)

from .pca import LogIVPCA


@version(
    "0.1",
    dependencies=[LogIVPCA, ARSamplerConfig, fit_surface_first_xar1_pca],
)
class ImpliedPCASampler(ARSampler):
    """Adapt direct PCA scores to the shared DYSANOS AR/importance sampler.

    The surface block is fitted first and is autonomous. Log-spot is appended
    only after that fit, with one orthogonal spot residual retained regardless
    of the number of surface innovation factors.
    """

    def __init__(
        self,
        *,
        pca: LogIVPCA,
        scores: np.ndarray,
        time_y: np.ndarray,
        spot_ref: np.ndarray,
        num_alpha: int | None,
        seed: int,
        rescale_variance: bool = False,
        verbose: Context = Context.all,
    ) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        time_y = np.asarray(time_y, dtype=np.float64)
        spot_ref = np.asarray(spot_ref, dtype=np.float64)
        verify_inp(
            scores.ndim == 2
            and scores.shape[1] == pca.num_components
            and scores.shape[0] >= 3,
            lambda: (
                f"scores must have shape (D,{pca.num_components}) with D>=3, "
                f"got {scores.shape}"
            ),
        )
        D = scores.shape[0]
        verify_inp(
            time_y.shape == spot_ref.shape == (D,),
            lambda: (
                f"time/spot arrays must both have shape {(D,)}, got "
                f"{time_y.shape}/{spot_ref.shape}"
            ),
        )
        verify_inp(
            np.all(np.isfinite(scores))
            and np.all(np.isfinite(time_y))
            and np.all(np.isfinite(spot_ref))
            and np.all(np.diff(time_y) > 0.0)
            and np.all(spot_ref > 0.0),
            "sampler inputs must be finite with increasing time and positive spot",
        )
        verify_inp(
            num_alpha is None
            or (
                isinstance(num_alpha, int)
                and 0 < num_alpha <= pca.num_components
            ),
            lambda: f"invalid retained surface-factor count {num_alpha!r}",
        )
        verify_inp(
            isinstance(seed, int), lambda: f"'seed' must be an integer, got {seed!r}"
        )

        self.pca = pca
        self.config = ARSamplerConfig(
            num_alpha=num_alpha,
            rescale_var=rescale_variance,
            seed=seed,
            use_matrix_exp=True,
            surface_first=True,
        )
        log_normalized_spot = np.log(spot_ref / spot_ref[0])
        self.log_nspot = log_normalized_spot
        self.xstate = np.concatenate((log_normalized_spot[:, None], scores), axis=1)
        self.time_y = time_y
        self.spot_ref = log_normalized_spot
        self.rng = np.random.default_rng(seed=seed)
        self.pcaxar1: AR1PCA = fit_surface_first_xar1_pca(
            self.xstate,
            time_y,
            num_alpha=num_alpha,
            rescale_var=rescale_variance,
            use_matrix_exp=True,
        )

        surface_eigenvalues = np.linalg.eigvals(self.pcaxar1.k[1:, 1:])
        verify_inp(
            np.all(np.isfinite(surface_eigenvalues))
            and np.all(surface_eigenvalues.real > 0.0),
            lambda: (
                "PCA surface generator is not stationary; eigenvalues are "
                f"{surface_eigenvalues}"
            ),
        )
        verify_inp(
            self.pcaxar1.spot_residual_std is not None
            and np.isfinite(self.pcaxar1.spot_residual_std)
            and self.pcaxar1.spot_residual_std > 0.0,
            lambda: (
                "orthogonal spot residual must have positive volatility, got "
                f"{self.pcaxar1.spot_residual_std}"
            ),
        )
        alpha_hist = self.pcaxar1.alpha
        self.alpha_mean = np.mean(alpha_hist, axis=0)
        alpha_covariance = np.atleast_2d(
            np.cov(alpha_hist, rowvar=False, bias=True)
        )
        self.cholesky = np.linalg.cholesky(
            alpha_covariance + 1.0e-12 * np.eye(self.num_alpha)
        )
        verbose.write(
            lambda: (
                f"Direct PCA-IV sampler: {pca.num_components} surface states, "
                f"{self.pcaxar1.num_surface_alpha} surface innovations plus one "
                "orthogonal spot innovation; "
                f"surface variance retained {self.pcaxar1.surface_var_explained:.1%}."
            )
        )

    @property
    def num_xstate(self) -> int:
        return self.xstate.shape[1]


__all__ = ["ImpliedPCASampler"]
