"""Joint front laws preserve calibrated spot risk without reusing historical spot volatility."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.front_joint import EmpiricalFrontCopula, GaussianFrontCopula
from quantdata.hedge2.front_spot import DiscreteFrontLaw
from quantdata.hedge2.greeks import parametric_factor_covariance

DAY = date(2024, 4, 3)


def _front() -> DiscreteFrontLaw:
    return DiscreteFrontLaw(cash_support=np.array([90., 100., 115.]), probabilities=np.array([.2, .5, .3]))


def _samples() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(62)
    normal = rng.normal(size=252)
    residuals = np.c_[.2 * normal + rng.normal(scale=.3, size=252),
        -.1 * normal + rng.normal(scale=.2, size=252)]
    residuals -= residuals.mean(axis=0)
    return normal, residuals


def test_empirical_branches_preserve_both_marginals_and_physical_beta_units() -> None:
    front = _front()
    normal = np.array([-2., -1., 0., 1., 2.])
    residuals = np.c_[normal, normal**2]
    model = EmpiricalFrontCopula(historical_scores=normal, factor_residuals=residuals, as_of_date=DAY)
    branches = model.branches(front=front, reference_forward=101.)
    assert len(branches.probabilities) <= len(normal) + len(front.probabilities) - 1
    np.testing.assert_allclose(np.bincount(branches.front_indices, weights=branches.probabilities),
        front.probabilities, atol=1e-15)
    np.testing.assert_allclose(np.bincount(branches.historical_indices, weights=branches.probabilities),
        np.full(len(normal), 1. / len(normal)), atol=1e-15)
    np.testing.assert_allclose(branches.probabilities @ branches.factor_residuals, residuals.mean(axis=0), atol=1e-15)
    np.testing.assert_array_equal(branches.physical_returns, (branches.next_spots - front.mean) / 101.)
    assert branches.probabilities @ branches.physical_returns == pytest.approx(0., abs=1e-16)
    covariance = model.innovation_covariance(front=front, reference_forward=101.)
    assert covariance[0, 0] == pytest.approx(front.variance / 101.**2)
    np.testing.assert_allclose(covariance[1:, 1:], np.cov(residuals, rowvar=False, ddof=0), atol=1e-14)
    assert covariance[0, 1] > 0.
    assert np.linalg.eigvalsh(covariance).min() > -1e-14
    strikes = np.array([85., 99., 102., 116.])
    payoffs = np.maximum(branches.next_spots[:, None] - strikes, 0.)
    expected = front.probabilities @ np.maximum(front.cash_support[:, None] - strikes, 0.)
    np.testing.assert_allclose(branches.probabilities @ payoffs, expected, atol=2e-15)


def test_empirical_ties_preserve_distinct_rows_and_are_permutation_invariant() -> None:
    normal = np.array([0., 0., 1., 1.])
    residuals = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    permutation = np.array([3, 0, 2, 1])
    original = EmpiricalFrontCopula(historical_scores=normal, factor_residuals=residuals, as_of_date=DAY)
    permuted = EmpiricalFrontCopula(historical_scores=normal[permutation],
        factor_residuals=residuals[permutation], as_of_date=DAY)

    def measure(model):
        branches = model.branches(front=_front(), reference_forward=100.)
        return {(float(spot), *factor): float(weight) for spot, factor, weight in
            zip(branches.next_spots, branches.factor_residuals, branches.probabilities, strict=True)}

    assert measure(original) == measure(permuted)
    np.testing.assert_allclose(original.innovation_covariance(front=_front(), reference_forward=100.),
        permuted.innovation_covariance(front=_front(), reference_forward=100.), atol=1e-15)


def test_constant_scores_make_empirical_surface_independent_of_spot_without_averaging() -> None:
    model = EmpiricalFrontCopula(historical_scores=np.zeros(4),
        factor_residuals=np.arange(8.).reshape(4, 2), as_of_date=DAY)
    branches = model.branches(front=_front(), reference_forward=100.)
    assert len(branches.probabilities) == 12
    expected = np.broadcast_to(_front().probabilities[:, None] / 4., (3, 4))
    np.testing.assert_allclose(branches.probabilities.reshape(3, 4), expected)
    covariance = model.innovation_covariance(front=_front(), reference_forward=100.)
    np.testing.assert_allclose(covariance[0, 1:], 0., atol=1e-16)
    sample = model.sample(front=_front(), reference_forward=100., count=30000, rng=np.random.default_rng(20))
    assert len(np.unique(sample.factor_residuals, axis=0)) == 4
    assert abs(np.corrcoef(sample.normal_spot_driver, sample.factor_residuals[:, 0])[0, 1]) < .02


def test_empirical_transport_keeps_tiny_upper_and_lower_atoms_with_ties() -> None:
    front = DiscreteFrontLaw(cash_support=np.array([80., 90., 110., 120.]),
        probabilities=np.array([1e-30, .5, .5, 1e-30]))
    model = EmpiricalFrontCopula(historical_scores=np.array([-1., -1., 1., 1.]),
        factor_residuals=np.arange(4.)[:, None], as_of_date=DAY)
    branches = model.branches(front=front, reference_forward=100.)
    mass = np.bincount(branches.front_indices, weights=branches.probabilities, minlength=4)
    np.testing.assert_allclose(mass, front.probabilities, rtol=2e-15, atol=0.)
    assert np.all(branches.probabilities > 0.)
    assert np.sum(branches.front_indices == 0) == 2
    assert np.sum(branches.front_indices == 3) == 2


def test_empirical_probability_underflow_is_reported_not_silently_truncated() -> None:
    front = DiscreteFrontLaw(cash_support=np.array([90., 110.]),
        probabilities=np.array([np.nextafter(0., 1.), 1.]))
    model = EmpiricalFrontCopula(historical_scores=np.zeros(3), factor_residuals=np.zeros((3, 1)), as_of_date=DAY)
    with pytest.raises(ValueError, match="underflows"):
        model.branches(front=front, reference_forward=100.)


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_samples_are_reproducible_use_standard_normal_and_current_front(kind) -> None:
    normal, residuals = _samples()
    model = kind(historical_scores=3. + 2. * normal, factor_residuals=residuals, as_of_date=DAY)
    front = _front()
    first = model.sample(front=front, reference_forward=101., count=30000, rng=np.random.default_rng(35))
    second = model.sample(front=front, reference_forward=101., count=30000, rng=np.random.default_rng(35))
    np.testing.assert_array_equal(first.next_spots, second.next_spots)
    np.testing.assert_array_equal(first.factor_residuals, second.factor_residuals)
    np.testing.assert_array_equal(first.next_spots, front.from_normal(first.normal_spot_driver))
    np.testing.assert_array_equal(first.physical_returns, (first.next_spots - front.mean) / 101.)
    assert abs(first.normal_spot_driver.mean()) < .02
    assert abs(first.normal_spot_driver.std() - 1.) < .02
    frequencies = np.array([(first.next_spots == spot).mean() for spot in front.cash_support])
    np.testing.assert_allclose(frequencies, front.probabilities, atol=.01)
    if kind is EmpiricalFrontCopula:
        np.testing.assert_array_equal(first.factor_residuals, residuals[first.historical_indices])
    else:
        assert first.historical_indices is None


def test_gaussian_fit_preserves_factor_covariance_and_score_rescaling_only_fits_dependence() -> None:
    normal, residuals = _samples()
    first = GaussianFrontCopula(historical_scores=normal, factor_residuals=residuals, as_of_date=DAY)
    second = GaussianFrontCopula(historical_scores=3. + 2. * normal, factor_residuals=residuals, as_of_date=DAY)
    mean, loading, root = first.conditional_parameters
    np.testing.assert_allclose(mean, residuals.mean(axis=0), atol=1e-16)
    np.testing.assert_allclose(np.outer(loading, loading) + root @ root.T,
        np.cov(residuals, rowvar=False, ddof=0), atol=2e-16)
    np.testing.assert_allclose(first.innovation_covariance(front=_front(), reference_forward=101.),
        second.innovation_covariance(front=_front(), reference_forward=101.), atol=1e-16)


def test_gaussian_physical_covariance_matches_independent_simulation() -> None:
    normal, residuals = _samples()
    model = GaussianFrontCopula(historical_scores=normal, factor_residuals=residuals, as_of_date=DAY)
    front = _front()
    sample = model.sample(front=front, reference_forward=101., count=200000, rng=np.random.default_rng(101))
    expected = model.innovation_covariance(front=front, reference_forward=101.)
    empirical = np.cov(np.c_[sample.physical_returns, sample.factor_residuals], rowvar=False, ddof=0)
    np.testing.assert_allclose(empirical, expected, rtol=.025, atol=2e-5)
    assert expected[0, 0] == pytest.approx(front.variance / 101.**2)
    assert np.linalg.eigvalsh(expected).min() > -1e-14


def test_gaussian_perfect_score_loading_and_degenerate_spot_are_valid() -> None:
    normal = np.array([-1., 0., 1.])
    model = GaussianFrontCopula(historical_scores=normal, factor_residuals=2. * normal[:, None], as_of_date=DAY)
    mean, loading, root = model.conditional_parameters
    np.testing.assert_allclose(root, 0., atol=1e-15)
    assert loading[0] == pytest.approx(2. * normal.std())
    front = DiscreteFrontLaw(cash_support=np.array([100.]), probabilities=np.ones(1))
    covariance = model.innovation_covariance(front=front, reference_forward=101.)
    np.testing.assert_array_equal(covariance[0], [0., 0.])
    sample = model.sample(front=front, reference_forward=101., count=5, rng=np.random.default_rng(2))
    np.testing.assert_allclose(sample.factor_residuals[:, 0], mean[0] + loading[0] * sample.normal_spot_driver, atol=1e-15)


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_factory_enforces_dated_causal_rows_and_copies_inputs(kind) -> None:
    frame = pd.DataFrame({"date": ["2024-04-02", "2024-04-03", "2024-04-04"],
        "pair_start": ["2024-04-01", "2024-04-02", "2024-04-03"], "sanos_gaussian_score": [.1, .8, np.nan]})
    model = SimpleNamespace(training_start=date(2024, 4, 1), training_end=DAY, as_of_date=DAY,
        residuals=np.array([[-.01, .2], [.01, -.2]]))
    fitted = kind.from_dynamics(model=model, scores=frame)
    np.testing.assert_array_equal(fitted.historical_scores, [.1, .8])
    fitted.validate_model(model)
    identity = fitted.identity
    model.residuals[0, 1] = 4.
    assert fitted.identity == identity
    assert fitted.factor_residuals[0, 0] == .2
    assert not fitted.factor_residuals.flags.writeable
    with pytest.raises(ValueError, match="same dated surface"):
        fitted.validate_model(model)


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
@pytest.mark.parametrize("scores,residuals", [([0.], [[1.]]), ([0., np.nan], [[1.], [2.]]),
    ([0., 1.], [[1.]]), ([0., 1.], [[1., np.inf], [2., 3.]]), ([0., 1.], [[], []])])
def test_invalid_fitted_inputs_are_rejected(kind, scores, residuals) -> None:
    with pytest.raises(ValueError, match="aligned finite"):
        kind(historical_scores=np.asarray(scores), factor_residuals=np.asarray(residuals), as_of_date=DAY)


def test_constant_scores_cannot_estimate_gaussian_dependence() -> None:
    with pytest.raises(ValueError, match="nonzero historical score variance"):
        GaussianFrontCopula(historical_scores=np.zeros(3), factor_residuals=np.ones((3, 2)), as_of_date=DAY)


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_parametric_risk_uses_current_front_variance_in_the_same_greek_units(kind) -> None:
    normal, residuals = _samples()
    raw = np.c_[.001 * normal, residuals]
    model = SimpleNamespace(as_of_date=DAY, residuals=raw, daily_factor_scales=np.array([.2, .1]),
        joint_innovation_covariance=np.cov(raw, rowvar=False, ddof=0))
    fitted = kind(historical_scores=normal, factor_residuals=residuals, as_of_date=DAY)
    covariance = parametric_factor_covariance(model, front_coupling=fitted, front=_front(), reference_forward=101.)
    scales = np.r_[1., model.daily_factor_scales]
    np.testing.assert_allclose(covariance * np.outer(scales, scales),
        fitted.innovation_covariance(front=_front(), reference_forward=101.), atol=1e-16)
    assert covariance[0, 0] == pytest.approx(_front().variance / 101.**2)
    assert covariance[0, 0] > 100. * parametric_factor_covariance(model)[0, 0]
    with pytest.raises(ValueError, match="together"):
        parametric_factor_covariance(model, front_coupling=fitted)
