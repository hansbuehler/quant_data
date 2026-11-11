"""Deterministic mathematical checks for the model-law audit."""

import numpy as np
import pytest

from quantdata.hedging.model_audit.grading import grade, tail_grade
from quantdata.hedging.model_audit.metrics import (
    autocorrelation,
    covariance,
    covariance_comparison,
    factor_innovation_covariance,
    frozen_pca,
    linear_stability,
    mixture_covariance,
    quote_scores,
    tail_scores,
    var_es_score,
)


def test_covariance_comparison_identical() -> None:
    matrix = np.diag([4., 2., 1., .5])
    result = covariance_comparison(matrix, matrix)
    assert result["cov_frob_rel"] == 0
    assert result["eig_share_err_k"] == 0
    assert result["subspace_angle_max"] == pytest.approx(0)


def test_recursive_worker_partitions_are_disjoint_and_complete() -> None:
    from quantdata.hedging.model_audit.parallel_recursive import partition_origins

    origins = [f"origin-{index:02d}" for index in range(44)]
    partitions = partition_origins(origins, 4)
    assert len(partitions) == 4
    assert all(len(partition) == 11 for partition in partitions)
    assert sorted(origin for partition in partitions for origin in partition) == origins
    assert all(list(partition) == sorted(partition) for partition in partitions)
    with pytest.raises(ValueError):
        partition_origins(origins, 0)
    with pytest.raises(ValueError):
        partition_origins(["same", "same"], 1)


def test_tail_fitter_adds_prior_origin_without_removing_default_starts(monkeypatch) -> None:
    from quantdata.dysanos.flex import stochastic_volatility_diffusion as tail

    previous = tail.LogVarianceDiffusionConfig(persistence=.4, stationary_log_variance=.2, unconditional_scale=1.1)
    radii = np.random.default_rng(64).chisquare(16, size=64) / 16.
    starts = np.r_[True, np.zeros(63, dtype=bool)]
    attempted = []
    original = tail.minimize

    def record_start(objective, initial, **arguments):
        attempted.append(np.asarray(initial).copy())
        return original(objective, initial, **arguments)

    monkeypatch.setattr(tail, "minimize", record_start)
    result = tail.fit_log_variance_diffusion(radii, starts, dimension=16, initial_config=previous)
    assert len(attempted) == 5
    np.testing.assert_array_equal(attempted[0], [.4, .2, 1.1])
    np.testing.assert_array_equal([initial[0] for initial in attempted[1:]], [.2, .6, .9, .98])
    assert np.isfinite(result.negative_log_likelihood)


def test_exact_empirical_law_uses_population_covariance() -> None:
    assert covariance([[0.], [2.]], ddof=0)[0, 0] == 1.
    assert covariance([[0.], [2.]], ddof=1)[0, 0] == 2.


def test_collapsed_model_is_scored_not_mistaken_for_input_failure() -> None:
    result = covariance_comparison(np.zeros((4, 4)), np.eye(4))
    assert result["cov_frob_rel"] == 1.
    assert result["trace_ratio"] == 0.
    assert result["subspace_angle_max"] is None


def test_stock_innovation_changes_factor_covariance_not_ar_stability() -> None:
    joint = np.array([[4., .5], [.5, 2.]])
    innovations = factor_innovation_covariance(joint, [3.])
    np.testing.assert_allclose(innovations, [[41.]])
    result = linear_stability([[.5]], innovations)
    assert result["stable_linear_core"]
    assert result["stationary_variance_trace"] == pytest.approx(41. / .75)


def test_unit_root_backbone_has_no_stationary_covariance() -> None:
    result = linear_stability([[1.]], [[1.]])
    assert not result["stable_linear_core"]
    assert result["stationary_covariance"] is None


def test_mixture_preserves_between_state_mean_variation() -> None:
    means = np.array([[0.], [2.]])
    result = mixture_covariance(means, np.ones((2, 1, 1)))
    assert result[0, 0] == 3.


def test_pca_is_orthonormal_and_shares_use_full_trace() -> None:
    changes = np.random.default_rng(41).normal(size=(100, 6))
    result = frozen_pca(changes)
    np.testing.assert_allclose(result["basis"].T @ result["basis"], np.eye(3), atol=1e-14)
    assert 0 < result["shares"].sum() < 1


def test_quote_scores_use_full_spread() -> None:
    result = quote_scores([.5, 2., 5., 4.], [1., 1., 1., 4.], [3., 3., 3., 4.])
    assert result["count"] == 3
    assert result["excluded"] == 1
    assert result["in_spread_rate"] == pytest.approx(1 / 3)
    assert result["out_dist_mean"] == pytest.approx((.25 + 0. + 1.) / 3)


def test_tail_zero_exceptions_is_not_clustering_evidence() -> None:
    result = tail_scores(np.ones(44), np.zeros(44), probability=.05)
    assert result["exceptions"] == 0
    assert result["expected_exceptions"] == pytest.approx(2.2)
    assert result["christoffersen_p"] is None
    assert result["exploratory"]
    assert np.isfinite(result["kupiec_p"])


def test_acf_respects_session_gaps() -> None:
    result = autocorrelation([0., 1., 3., 4., 5.], [0, 1, 3, 4, 5], lags=(1, 63))
    assert result[0]["pair_count"] == 3
    assert result[0]["correlations"][0] == pytest.approx(1.)
    assert result[1]["correlations"] == [None]


def test_stability_half_life_and_variance() -> None:
    result = linear_stability([[.5]], [[1.]])
    assert result["spectral_radius"] == .5
    assert result["modal_half_lives"] == [1.]
    assert result["conditional_variance_trace"]["63"] == pytest.approx(4 / 3)
    np.testing.assert_allclose(result["stationary_covariance"], [[4. / 3.]])
    assert result["lyapunov_residual_relative"] < 1e-14


def test_var_es_includes_positive_shortfall() -> None:
    result = var_es_score([-.3, .2], [-.2, .1], [-.4, .05])
    assert result["count"] == 2
    assert result["excluded"] == 0
    assert np.isfinite(result["varES_score_5"])


@pytest.mark.parametrize(("metric", "value", "expected"), [
    ("cov_frob_rel", .25, "pass"), ("cov_frob_rel", .5, "warn"), ("cov_frob_rel", .501, "fail"),
    ("trace_ratio", .8, "pass"), ("trace_ratio", .5, "warn"), ("trace_ratio", .49, "fail"),
    ("trace_ratio", 1.25, "pass"), ("trace_ratio", 2., "warn"), ("trace_ratio", 0., "fail"),
    ("fit_in_spread", .95, "pass"), ("fit_in_spread", .8, "warn"), ("fit_in_spread", .79, "fail"),
    ("ar_radius", .989, "pass"), ("ar_radius", .99, "warn"), ("ar_radius", 1., "fail"),
    ("ar_radius", None, "warn"), ("forecast_ratio", 1., "pass"), ("forecast_ratio", 1.11, "fail"),
])
def test_published_traffic_light_boundaries(metric: str, value: float | None, expected: str) -> None:
    assert grade(metric, value) == expected


def test_sparse_tail_nonrejection_is_warning_not_pass() -> None:
    assert tail_grade({"exact_binomial_p": .8, "expected_exceptions": 2.2}) == "warn"
    assert tail_grade({"exact_binomial_p": .005, "expected_exceptions": 2.2}) == "fail"
    assert tail_grade({"exact_binomial_p": .8, "expected_exceptions": 20}) == "pass"


def test_report_table_has_caption_label_and_fixed_anchor() -> None:
    from quantdata.hedging.model_audit.report import _table

    table = _table(["Model", "Days"], [["Example", 44]], caption="Coverage", label="tab:coverage")
    assert r"\begin{table}[H]" in table
    assert r"\caption{Coverage Color rules:" in table
    assert r"\label{tab:coverage}" in table


def test_all_audit_tables_have_explicit_caption_color_rules() -> None:
    from quantdata.hedging.model_audit.report import TABLE_COLOR_RULES, _table

    assert len(TABLE_COLOR_RULES) == 14
    for label, rules in TABLE_COLOR_RULES.items():
        assert "black" in rules.lower()
        assert "orange" in rules.lower()
        assert "red" in rules.lower()
        rendered = _table(["Model", "Value"], [], caption="Diagnostic.", label=label)
        assert "Color rules:" in rendered
        assert "Missing graded entries are orange" in rendered


@pytest.mark.parametrize("recursive_count", [0, 44])
def test_executive_summary_reports_live_coverage_without_claiming_missing_tests_pass(recursive_count: int) -> None:
    import pandas as pd

    from quantdata.hedging.model_audit.report import _executive_summary

    summaries = {name: {"pooled_covariance": {"trace_ratio": value}}
        for name, value in (("DYSANOS ordinary", .06), ("DYSANOS spot-orthogonal", .15), ("VolGAN PCA8", .97))}
    summaries["VolGAN PCA8"]["conditional_covariance"] = {"trace_ratio": .00025}
    coverage = pd.DataFrame([{"requested_origins": 44, "surface_origins": 44,
                             "quote_forecast_origins": 44, "recursive_origins": recursive_count}])
    stability = pd.DataFrame({"stationary_backbone": [True, True], "spectral_radius": [.6, .7]})
    text = _executive_summary(summaries, coverage, stability)
    assert r"\section*{Executive summary}" in text
    assert f"recursive path tests {recursive_count}/44" in text
    assert "6.0\\%" in text
    assert "0.025\\%" in text
    assert ("Coverage is still being completed" in text) == (recursive_count < 44)
    assert "Stationarity is not the same as calibration" in text


def test_recursive_model_failure_is_completed_evidence_not_missing_coverage() -> None:
    from quantdata.hedging.model_audit.report import _completed_recursive

    rows = [{"status": "scored"}, {"status": "model_invalid"}, {"status": "error"}]
    assert _completed_recursive(rows) == rows[:2]


def test_direct_cdx_inversion_keeps_its_tiny_price_estimate_without_custom_veto() -> None:
    from cdxcore.bs import bs

    from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity

    strikes = np.array([1., 1.5])
    prices = bs.price(k=strikes, vol=.2, sqrtT=np.sqrt(1. / 252.))
    direct = bs.implied(k=strikes, prices=prices, sqrtT=np.sqrt(1. / 252.),
                        price_tol=1e-8, eps=1e-10, ret_only_vols=False)
    diagnostics = []
    vols, valid = implied_volatilities_with_validity(expiries_y=np.full(2, 1. / 252.),
        normalized_strikes=strikes, normalized_prices=prices, is_call=np.ones(2, dtype=bool), diagnostics=diagnostics)
    np.testing.assert_array_equal(vols, direct.vols)
    assert valid.all()
    assert diagnostics[0]["library_intrinsic_cases"] == 1
    assert diagnostics[0]["unusable"] == 0


def test_direct_cdx_inversion_preserves_bound_warning_and_does_not_accept_violations() -> None:
    from cdxcore.bs import bs

    from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity

    diagnostics = []
    with pytest.warns(UserWarning, match="below intrinsic"):
        _, valid = implied_volatilities_with_validity(expiries_y=np.array([1.]),
            normalized_strikes=np.array([1.]), normalized_prices=np.array([-.001]),
            is_call=np.array([True]), diagnostics=diagnostics)
    assert not valid[0]
    assert diagnostics[0]["outside_price_bounds"] == 1
    expected = float(bs.price(k=np.array([1.]), vol=.01, sqrtT=1.)[0]) + .001
    assert diagnostics[0]["max_repricing_error"] == pytest.approx(expected)


def test_tail_conditional_spot_split_preserves_original_gaussian_covariance() -> None:
    from types import SimpleNamespace

    from quantdata.hedging.model_audit.tail import conditional_factor_innovations

    joint = np.array([[4., 1., -.5], [1., 3., .2], [-.5, .2, 2.]])
    values = np.random.default_rng(17).multivariate_normal(np.zeros(3), joint, size=252)
    dynamics = SimpleNamespace(joint_innovation_covariance=joint, residuals=values)
    loading, factor, residuals = conditional_factor_innovations(dynamics)
    np.testing.assert_allclose(4 * np.outer(loading, loading) + factor @ factor.T, joint[1:, 1:])
    np.testing.assert_allclose(residuals, values[:, 1:] - values[:, :1] * loading)


def test_float32_contract_identity_is_unique_and_bounded() -> None:
    import pandas as pd

    from quantdata.hedging.model_audit.quotes import contract_strike_keys

    common = {"expiry": pd.Timestamp("2024-05-22"), "option_type": "call", "is_am": False}
    origin = pd.DataFrame([{**common, "strike": value} for value in (5290., 5295., 5300.)])
    identifiers = pd.DataFrame([{**common, "strike": value} for value in (5294.999720978239, 5293.)])
    keys, errors = contract_strike_keys(identifiers, origin)
    assert keys[0] == 5295.
    assert errors[0] < .001
    assert np.isnan(keys[1])
    ambiguous = pd.concat((origin, pd.DataFrame([{**common, "strike": 5295.0001}])), ignore_index=True)
    assert np.isnan(contract_strike_keys(identifiers, ambiguous)[0][0])


def test_diffusion_recursive_context_uses_each_own_surface_and_broadcast_return() -> None:
    from quantdata.hedging.model_audit.recursive import append_diffusion_context

    context = np.arange(2 * 3 * 2 * 2 * 3).reshape(2, 3, 2, 2, 3).astype(float)
    generated = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(float)
    following = append_diffusion_context(context, generated)
    np.testing.assert_array_equal(following[:, :-1], context[:, 1:])
    np.testing.assert_array_equal(following[:, -1, 0], generated[:, 0])
    np.testing.assert_array_equal(following[:, -1, 1],
        np.broadcast_to(generated[:, 1].mean(axis=(1, 2))[:, None, None], (2, 2, 3)))


def test_gromm_path_horizon_does_not_create_unobserved_expiries() -> None:
    from quantdata.hedging.model_audit.recursive import available_path_steps

    assert available_path_steps(np.array([1., 5., 125.]), 128) == 115
    assert available_path_steps(np.array([1., 5., 125.]), 44) == 44
    with pytest.raises(ValueError, match="cannot support"):
        available_path_steps(np.array([1., 5., 10.]), 128)


def test_gromm_aging_observable_preserves_flat_black_iv_without_long_extrapolation() -> None:
    from quantdata.hedging.model_audit.recursive import aged_observable_surface

    supports = np.array([[.5, 1., 2.], [.5, 1., 2.]])
    densities = np.array([[[0., 1., 0.], [0., 1., 0.]]])
    decoded = aged_observable_surface(supports, densities, np.ones(1), np.ones(1), np.array([3., 10.]))
    np.testing.assert_allclose(decoded, np.log(.05), atol=5e-6, rtol=0)
    with pytest.raises(ValueError, match="exceeds"):
        aged_observable_surface(supports, densities, np.ones(1), np.ones(1), np.array([3., 9.]))
