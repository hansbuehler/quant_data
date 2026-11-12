"""Observable covariance, probabilistic forecast and persistence diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov, subspace_angles
from scipy.special import xlogy
from scipy.stats import binomtest, chi2


def covariance(values: ArrayLike, *, ddof: int = 1) -> np.ndarray:
    """Covariance with explicit sample/population normalization and first-axis rows."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2 or ddof not in (0, 1) or not np.isfinite(matrix).all():
        raise ValueError("Covariance needs at least two finite vector observations")
    return np.atleast_2d(np.cov(matrix, rowvar=False, ddof=ddof))


def frozen_pca(changes: ArrayLike, *, rank: int = 3) -> dict[str, np.ndarray]:
    """Fit an orthonormal, unwhitened basis to training increments only."""
    matrix = np.asarray(changes, dtype=float)
    spectrum, vectors = np.linalg.eigh(covariance(matrix))
    order = np.argsort(spectrum)[::-1]
    spectrum, vectors = spectrum[order], vectors[:, order]
    if rank < 1 or rank > matrix.shape[1] or spectrum[rank - 1] <= 0:
        raise ValueError("Requested PCA rank is unidentified")
    basis = vectors[:, :rank]
    pivots = np.argmax(np.abs(basis), axis=0)
    basis *= np.where(basis[pivots, np.arange(rank)] < 0, -1., 1.)
    return {"basis": basis, "mean_change": matrix.mean(axis=0),
            "eigenvalues": spectrum, "shares": spectrum[:rank] / spectrum.sum()}


def covariance_comparison(model: ArrayLike, data: ArrayLike, *, rank: int = 3) -> dict[str, object]:
    """Compare full covariance and leading subspaces, in the same coordinates."""
    model_cov, data_cov = np.asarray(model, dtype=float), np.asarray(data, dtype=float)
    if model_cov.shape != data_cov.shape or model_cov.ndim != 2:
        raise ValueError("Model and data covariance dimensions differ")
    denominator = float(np.linalg.norm(data_cov))
    if denominator <= 0 or np.trace(data_cov) <= 0 or np.trace(model_cov) < 0:
        raise ValueError("Covariance comparisons require positive data variance and nonnegative model variance")
    if not np.any(model_cov):
        data_values = np.linalg.eigvalsh(data_cov)
        return {"cov_frob_rel": 1., "trace_ratio": 0., "model_eigenvalue_shares": [None] * rank,
                "data_eigenvalue_shares": (data_values[::-1][:rank] / data_values.sum()).tolist(),
                "eig_share_err_k": None, "subspace_angle_max": None}
    model_values, model_vectors = np.linalg.eigh(model_cov)
    data_values, data_vectors = np.linalg.eigh(data_cov)
    model_shares = model_values[::-1][:rank] / model_values.sum()
    data_shares = data_values[::-1][:rank] / data_values.sum()
    angles = subspace_angles(model_vectors[:, -rank:], data_vectors[:, -rank:])
    return {"cov_frob_rel": float(np.linalg.norm(model_cov - data_cov) / denominator),
            "trace_ratio": float(np.trace(model_cov) / np.trace(data_cov)),
            "model_eigenvalue_shares": model_shares.tolist(),
            "data_eigenvalue_shares": data_shares.tolist(),
            "eig_share_err_k": float(np.abs(model_shares - data_shares).sum()),
            "subspace_angle_max": float(np.rad2deg(angles.max()))}


def mixture_covariance(means: ArrayLike, covariances: ArrayLike) -> np.ndarray:
    """Average conditional covariance plus between-origin sample covariance."""
    mean_array, covariance_array = np.asarray(means), np.asarray(covariances)
    if len(mean_array) < 2 or len(mean_array) != len(covariance_array):
        raise ValueError("At least two aligned forecast moments are needed")
    return covariance_array.mean(axis=0) + covariance(mean_array)


def forecast_variance_calibration(means: ArrayLike, covariances: ArrayLike,
                                  observed: ArrayLike) -> dict[str, float]:
    """Compare average predictive variance with realized squared forecast errors."""
    predictions, conditional, actual = np.asarray(means), np.asarray(covariances), np.asarray(observed)
    if (predictions.ndim != 2 or predictions.shape != actual.shape or len(predictions) < 2
            or conditional.shape != (len(predictions), predictions.shape[1], predictions.shape[1])
            or not all(np.isfinite(values).all() for values in (predictions, conditional, actual))):
        raise ValueError("Predictive variance calibration needs aligned finite forecast moments and outcomes")
    errors = actual - predictions
    variance = float(np.trace(conditional.mean(axis=0)))
    mse = float(np.mean(np.sum(errors**2, axis=1)))
    if mse <= 0. or variance < 0.:
        raise ValueError("Forecast-error variance comparison requires positive MSE and nonnegative modeled variance")
    return {"predicted_conditional_variance_trace": variance, "forecast_error_squared_norm_mean": mse,
        "conditional_variance_over_forecast_mse": variance / mse,
        "forecast_error_mean_squared_norm": float(np.sum(errors.mean(axis=0)**2)),
        "centered_forecast_error_covariance_trace": float(np.trace(covariance(errors)))}


def correlation_from_covariance(matrix: ArrayLike) -> np.ndarray:
    """Return correlations; unidentified zero-variance entries are NaN."""
    values = np.asarray(matrix, dtype=float)
    scale = np.sqrt(np.maximum(np.diag(values), 0.))
    denominator = np.outer(scale, scale)
    return np.divide(values, denominator, out=np.full_like(values, np.nan), where=denominator > 0)


def quote_scores(prices: ArrayLike, bid: ArrayLike, ask: ArrayLike) -> dict[str, float | int]:
    """Price errors in cash units; outside distance uses full spread width."""
    prices, bid, ask = np.broadcast_arrays(np.asarray(prices), np.asarray(bid), np.asarray(ask))
    valid = np.isfinite(prices) & np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask > bid)
    if not valid.any():
        return {"count": 0, "excluded": int(valid.size)}
    prices, bid, ask = prices[valid], bid[valid], ask[valid]
    outside = np.maximum(np.maximum(bid - prices, prices - ask), 0.) / (ask - bid)
    return {"count": len(prices), "excluded": int(valid.size - valid.sum()),
            "in_spread_rate": float(np.mean(outside <= 1e-12)),
            "out_dist_mean": float(outside.mean()), "out_dist_q95": float(np.quantile(outside, .95)),
            "price_mae": float(np.mean(np.abs(prices - .5 * (bid + ask))))}


def tail_scores(observed: ArrayLike, quantiles: ArrayLike, *, probability: float) -> dict[str, object]:
    """Pinball, exact/asymptotic coverage and identifiable clustering tests."""
    observations, forecasts = np.asarray(observed, dtype=float), np.asarray(quantiles, dtype=float)
    if observations.ndim != 1 or observations.shape != forecasts.shape or len(observations) < 2:
        raise ValueError("Tail observations and forecasts must be aligned vectors")
    if not 0 < probability < 1 or not np.isfinite(observations).all() or not np.isfinite(forecasts).all():
        raise ValueError("Invalid quantile probability or nonfinite forecast")
    lower = probability < .5
    exceptions = observations < forecasts if lower else observations > forecasts
    tail_probability = probability if lower else 1. - probability
    total, count = len(exceptions), int(exceptions.sum())
    likelihood_null = xlogy(count, tail_probability) + xlogy(total - count, 1. - tail_probability)
    empirical = count / total
    likelihood_fitted = xlogy(count, empirical) + xlogy(total - count, 1. - empirical)
    exact = binomtest(count, total, tail_probability)
    interval = exact.proportion_ci()
    independence_p = None
    transition_counts = np.zeros((2, 2), dtype=int)
    for previous, following in zip(exceptions[:-1], exceptions[1:], strict=True):
        transition_counts[int(previous), int(following)] += 1
    if transition_counts.sum(axis=1).min() > 0 and 0 < transition_counts[:, 1].sum() < total - 1:
        fitted = transition_counts[:, 1] / transition_counts.sum(axis=1)
        common = transition_counts[:, 1].sum() / (total - 1)
        alternative = np.sum(xlogy(transition_counts[:, 1], fitted)
                             + xlogy(transition_counts[:, 0], 1. - fitted))
        null = xlogy(transition_counts[:, 1].sum(), common) + xlogy(transition_counts[:, 0].sum(), 1. - common)
        independence_p = float(chi2.sf(max(0., 2. * (alternative - null)), 1))
    residual = observations - forecasts
    return {"count": total, "exceptions": count, "expected_exceptions": total * tail_probability,
            "exception_rate_q": empirical,
            "pinball_q": float(np.mean(np.maximum(probability * residual, (probability - 1.) * residual))),
            "kupiec_p": float(chi2.sf(max(0., 2. * (likelihood_fitted - likelihood_null)), 1)),
            "exact_binomial_p": float(exact.pvalue),
            "coverage_ci_low": float(interval.low), "coverage_ci_high": float(interval.high),
            "christoffersen_p": independence_p, "transition_counts": transition_counts.tolist(),
            "exploratory": bool(total * tail_probability < 10)}


def var_es_score(observed: ArrayLike, var_lower: ArrayLike, es_lower: ArrayLike, *, alpha: float = .05) -> dict[str, float | int]:
    """Joint Fissler--Ziegel score with G2(e)=exp(e), valid for signed ES."""
    observed, quantile, shortfall = np.broadcast_arrays(observed, var_lower, es_lower)
    valid = np.isfinite(observed) & np.isfinite(quantile) & np.isfinite(shortfall) & (shortfall <= quantile)
    if not valid.any():
        return {"count": 0, "excluded": int(valid.size)}
    observed, quantile, shortfall = observed[valid], quantile[valid], shortfall[valid]
    value = np.exp(shortfall) * (shortfall - quantile + (quantile - observed) * (observed <= quantile) / alpha - 1.)
    if not np.isfinite(value).all():
        raise FloatingPointError("Joint VaR-ES score overflow; do not clip or drop observations")
    return {"count": len(value), "excluded": int(valid.size - valid.sum()),
            "es_5": float(shortfall.mean()), "varES_score_5": float(value.mean())}


def autocorrelation(values: ArrayLike, sessions: ArrayLike, *, lags: tuple[int, ...]) -> list[dict[str, object]]:
    """Pearson ACF using exact session separations, without bridging missing days."""
    matrix, index = np.asarray(values, dtype=float), np.asarray(sessions, dtype=int)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(index) or np.any(np.diff(index) <= 0):
        raise ValueError("ACF needs aligned increasing session indices")
    lookup = {int(session): position for position, session in enumerate(index)}
    output = []
    for lag in lags:
        pairs = [(position, lookup[int(session + lag)]) for position, session in enumerate(index) if int(session + lag) in lookup]
        correlations = [None] * matrix.shape[1]
        if len(pairs) >= 3:
            before, after = np.asarray(pairs).T
            for component in range(matrix.shape[1]):
                left, right = matrix[before, component], matrix[after, component]
                if min(np.std(left), np.std(right)) > 0:
                    correlations[component] = float(np.corrcoef(left, right)[0, 1])
        output.append({"lag": lag, "pair_count": len(pairs), "correlations": correlations})
    return output


def factor_innovation_covariance(joint_covariance: ArrayLike, spot_loading: ArrayLike) -> np.ndarray:
    """Covariance of factor noise plus its correlated stock-return contribution."""
    joint, loading = np.asarray(joint_covariance, dtype=float), np.asarray(spot_loading, dtype=float)
    if (loading.ndim != 1 or joint.shape != (len(loading) + 1, len(loading) + 1)
            or not np.isfinite(joint).all() or not np.isfinite(loading).all()):
        raise ValueError("Joint spot/factor covariance and factor loading must be finite and aligned")
    innovation_map = np.column_stack((loading, np.eye(len(loading))))
    return innovation_map @ joint @ innovation_map.T


def linear_stability(matrix: ArrayLike, innovation_covariance: ArrayLike) -> dict[str, object]:
    """Frozen floating-volatility AR stability, stationary and forecast covariances."""
    transition, innovations = np.asarray(matrix, dtype=float), np.asarray(innovation_covariance, dtype=float)
    if (transition.ndim != 2 or transition.shape[0] != transition.shape[1] or not transition.size
            or innovations.shape != transition.shape or not np.isfinite(transition).all()
            or not np.isfinite(innovations).all()):
        raise ValueError("AR transition and innovation covariance must be finite aligned square matrices")
    eigenvalues = np.linalg.eigvals(transition)
    magnitudes = np.abs(eigenvalues)
    stable = bool(magnitudes.max() < 1.)
    stationary = solve_discrete_lyapunov(transition, innovations) if stable else None
    residual = (stationary - transition @ stationary @ transition.T - innovations) if stable else None
    variance = np.zeros_like(transition)
    growth = {}
    for horizon in range(1, 64):
        variance = transition @ variance @ transition.T + innovations
        if horizon in (1, 5, 21, 63):
            growth[str(horizon)] = float(np.trace(variance))
    return {"spectral_radius": float(magnitudes.max()), "stable_linear_core": stable,
            "modal_half_lives": [float(np.log(.5) / np.log(value)) if 0 < value < 1 else None for value in magnitudes],
            "conditional_variance_trace": growth,
            "stationary_covariance": stationary.tolist() if stable else None,
            "stationary_variance_trace": float(np.trace(stationary)) if stable else None,
            "lyapunov_residual_relative": float(np.linalg.norm(residual)
                / max(np.linalg.norm(stationary), np.linalg.norm(innovations), np.finfo(float).tiny)) if stable else None,
            "eigenvalues_real": eigenvalues.real.tolist(), "eigenvalues_imag": eigenvalues.imag.tolist()}
