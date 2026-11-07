"""Front quantile sampling calibrates spot marginals without historical-vol scaling."""

import numpy as np
import pytest
from scipy.special import ndtri

from quantdata.hedge2.front_spot import DiscreteFrontLaw, gaussian_scores, historical_gaussian_score
from tests.quantdata.hedge2.test_pricing import _pricer


def test_normal_driver_reproduces_every_front_atom_and_option_price() -> None:
    pricer = _pricer()
    law = DiscreteFrontLaw.from_pricer(pricer=pricer, parity_forward=103.)
    lower = np.cumsum(law.probabilities) - .5 * law.probabilities
    upper = np.cumsum(law.probabilities[::-1])[::-1] - .5 * law.probabilities
    quantiles = ndtri(np.minimum(lower, upper))
    normal = np.where(lower <= upper, quantiles, -quantiles)
    np.testing.assert_array_equal(law.from_normal(normal), law.cash_support)
    assert law.mean == pytest.approx(103., abs=1e-10)
    strikes = np.array([99., 103., 106.])
    cash_calls = law.probabilities @ np.maximum(law.cash_support[:, None] - strikes, 0.)
    expected = 103. * pricer.european_at(expiry_y=1. / 252., strikes=strikes / 103.)
    np.testing.assert_allclose(cash_calls, expected, atol=1e-12)
    strict_calls = law.probabilities @ (law.cash_support[:, None] > strikes)
    np.testing.assert_allclose(strict_calls, 1. - law.cdf(strikes), atol=2e-15)


def test_zero_mass_atoms_are_never_sampled_at_either_probability_endpoint() -> None:
    law = DiscreteFrontLaw(cash_support=np.array([90., 95., 100., 105., 110.]),
        probabilities=np.array([0., .25, .5, .25, 0.]))
    np.testing.assert_array_equal(law.quantile(np.array([0., .25, .5, .75, 1.])), [95., 95., 100., 100., 105.])
    np.testing.assert_array_equal(law.from_normal(np.array([-40., 40.])), [95., 105.])
    np.testing.assert_allclose(law.cdf(np.array([-np.inf, 95., 100., np.inf])), [0., .25, .75, 1.])
    assert law.variance == pytest.approx(12.5)


def test_changed_front_density_changes_spot_risk_without_a_historical_scale() -> None:
    normal = ndtri(np.array([.1, .5, .9]))
    quiet = DiscreteFrontLaw(cash_support=np.array([99., 100., 101.]), probabilities=np.array([.25, .5, .25]))
    volatile = DiscreteFrontLaw(cash_support=np.array([90., 100., 110.]), probabilities=quiet.probabilities)
    np.testing.assert_array_equal(quiet.from_normal(normal), [99., 100., 101.])
    np.testing.assert_array_equal(volatile.from_normal(normal), [90., 100., 110.])
    assert volatile.variance == pytest.approx(100. * quiet.variance)


def test_normal_sampling_preserves_extreme_atoms_beyond_cdf_rounding() -> None:
    law = DiscreteFrontLaw(cash_support=np.array([90., 99., 101., 110.]),
        probabilities=np.array([1e-30, .5, .5, 1e-30]))
    np.testing.assert_array_equal(law.from_normal(np.array([-12., -10., 0., 10., 12.])),
        [90., 99., 99., 101., 110.])


def test_degenerate_front_law_accepts_every_finite_normal_driver() -> None:
    law = DiscreteFrontLaw(cash_support=np.array([100.]), probabilities=np.ones(1))
    np.testing.assert_array_equal(law.from_normal(np.array([-100., 0., 100.])), [100., 100., 100.])
    assert law.from_normal(0.) == 100.
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            law.from_normal(invalid)


def test_parity_forward_and_financing_forward_remain_distinct_before_alpha_removal() -> None:
    law = DiscreteFrontLaw(cash_support=np.array([100., 102., 104.]), probabilities=np.array([.25, .5, .25]))
    gains = law.financed_spot_gains(reference_forward=101., discount=.99, initial_spot=100.)
    assert law.probabilities @ gains == pytest.approx(.0099)
    innovations = law.centered_physical_returns(reference_forward=101.)
    np.testing.assert_allclose(innovations, (law.cash_support - 102.) / 101., atol=1e-15)
    assert law.probabilities @ innovations == pytest.approx(0., abs=1e-15)
    assert law.probabilities @ innovations**2 == pytest.approx(law.variance / 101.**2)


def test_gaussian_scores_preserve_raw_probabilities_and_explicit_tail_flags() -> None:
    values = np.array([0., .1, .5, .9, 1.])
    result = gaussian_scores(values, probability_floor=1e-7)
    np.testing.assert_array_equal(result.cdf_probabilities, values)
    np.testing.assert_array_equal(result.lower_clipped, [True, False, False, False, False])
    np.testing.assert_array_equal(result.upper_clipped, [False, False, False, False, True])
    assert np.isfinite(result.scores).all()
    assert result.scores[2] == 0.
    assert result.scores[1] == pytest.approx(ndtri(.1))


def test_historical_score_uses_only_previous_source_and_parity_normalization(monkeypatch) -> None:
    source = object()
    captured = {}

    def cumulative(pricer, **arguments):
        assert pricer is source
        captured.update(arguments)
        return np.array([.7])

    monkeypatch.setattr("quantdata.hedge2.front_spot.surface_digitals", cumulative)
    result = historical_gaussian_score(previous_source=source, realized_spot=104., previous_parity_forward=102.)
    assert captured["expiry_y"] == 1. / 252.
    assert captured["is_call"] is False
    np.testing.assert_array_equal(captured["strikes"], [104. / 102.])
    assert result.scores[0] == pytest.approx(ndtri(.7))


@pytest.mark.parametrize("support,probabilities", [([100., 99.], [.5, .5]),
    ([99., 100.], [-.1, 1.1]), ([99., 100.], [.2, .2]), ([0., 100.], [.5, .5]),
    ([99., 100.], [.5, np.nan]), ([], [])])
def test_front_law_rejects_invalid_support_or_probability_mass(support, probabilities) -> None:
    with pytest.raises(ValueError):
        DiscreteFrontLaw(cash_support=np.array(support), probabilities=np.array(probabilities))


@pytest.mark.parametrize("probabilities", [np.array([np.nan]), np.array([-1e-6]), np.array([1.1])])
def test_invalid_gaussian_score_probabilities_are_not_silently_clipped(probabilities) -> None:
    with pytest.raises(ValueError):
        gaussian_scores(probabilities)
