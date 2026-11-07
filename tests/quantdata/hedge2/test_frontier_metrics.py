"""Infeasible portfolios cannot disappear, become zero-P&L hedges or change a baseline sample."""

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.frontier_metrics import CORE_KEYS, PORTFOLIO_KEYS, matched_frontier_metrics


def _tables():
    dates = ("2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04")
    reference = pd.DataFrame([{"origin": day, "family": "digital-3D", "claim": "target", "side": -1,
        "costed_pnl": pnl, "entry_cost": 0., "initial_price": .5}
        for day, pnl in zip(dates, (.1, -.2, .3, -.4), strict=True)])
    requested = pd.DataFrame([{"variant": variant, **row, "greek_budget": budget}
        for variant in ("ordinary", "orthogonal") for row in reference[list(CORE_KEYS)].to_dict("records")
        for budget in (.2, 1.)])
    available = requested.merge(reference, on=list(CORE_KEYS), validate="many_to_one")
    available["costed_pnl"] = .5 * available.costed_pnl + .01
    available["entry_cost"] = .001
    removed = available.variant.eq("ordinary") & available.greek_budget.eq(.2) & available.origin.eq(dates[1])
    unavailable = available.loc[removed].copy()
    unavailable["costed_pnl"] = np.nan
    return dict(requested=requested, available=available.loc[~removed].copy(), unavailable=unavailable,
        reference=reference)


def test_per_configuration_baseline_uses_exactly_the_same_available_dates() -> None:
    tables = _tables()
    metrics = matched_frontier_metrics(**tables, comparison_sample="per_configuration")
    assert len(metrics) == 4
    ordinary = metrics.loc[metrics.variant.eq("ordinary") & metrics.greek_budget.eq(.2)].iloc[0]
    baseline = tables["reference"].loc[~tables["reference"].origin.eq("2024-04-02"), "costed_pnl"]
    assert ordinary.completed_origins == ordinary.paired_origins == 3
    assert ordinary.unavailable_origins == 1
    assert ordinary.availability_fraction == .75
    assert ordinary.reference_std_pnl == pytest.approx(baseline.std(ddof=1))
    assert ordinary.std_pnl == pytest.approx(.5 * baseline.std(ddof=1))
    assert ordinary.mean_entry_cost == pytest.approx(.001)


def test_common_sampling_is_identical_across_all_requested_settings() -> None:
    metrics = matched_frontier_metrics(**_tables(), comparison_sample="common")
    assert set(metrics.paired_origins) == {3}
    assert metrics.reference_std_pnl.nunique() == 1
    assert metrics.std_pnl.nunique() == 1
    assert set(metrics.completed_origins) == {3, 4}
    assert set(metrics.unavailable_origins) == {0, 1}


def test_zero_availability_is_visible_and_never_becomes_zero_risk() -> None:
    tables = _tables()
    missing = tables["available"].variant.eq("orthogonal") & tables["available"].greek_budget.eq(1.)
    extra = tables["available"].loc[missing].copy()
    extra["costed_pnl"] = np.nan
    tables["unavailable"] = pd.concat([tables["unavailable"], extra], ignore_index=True)
    tables["available"] = tables["available"].loc[~missing]
    separate = matched_frontier_metrics(**tables, comparison_sample="per_configuration")
    absent = separate.loc[separate.variant.eq("orthogonal") & separate.greek_budget.eq(1.)].iloc[0]
    assert absent.availability_fraction == 0.
    assert absent.paired_origins == 0 and absent.unavailable_origins == 4
    assert not absent.performance_available and np.isnan(absent.std_pnl)
    common = matched_frontier_metrics(**tables, comparison_sample="common")
    assert (common.paired_origins == 0).all()
    assert common.std_pnl.isna().all()


def test_reference_gaps_need_explicit_permission_and_are_counted() -> None:
    tables = _tables()
    tables["reference"] = tables["reference"].iloc[:-1]
    with pytest.raises(ValueError, match="disclose reference gaps"):
        matched_frontier_metrics(**tables)
    metrics = matched_frontier_metrics(**tables, comparison_sample="per_configuration", allow_reference_gaps=True)
    assert (metrics.missing_reference_origins == 1).all()
    assert set(metrics.paired_origins) == {2, 3}


@pytest.mark.parametrize("problem", ["omitted", "overlap", "fabricated", "price", "negative_cost"])
def test_missing_or_inconsistent_outcomes_cannot_enter_a_report(problem) -> None:
    tables = _tables()
    if problem == "omitted":
        tables["available"] = tables["available"].iloc[1:]
    elif problem == "overlap":
        tables["unavailable"] = pd.concat([tables["unavailable"], tables["available"].iloc[:1]], ignore_index=True)
    elif problem == "fabricated":
        tables["unavailable"]["costed_pnl"] = 0.
    elif problem == "price":
        tables["available"].loc[tables["available"].index[0], "initial_price"] = .6
    else:
        tables["available"].loc[tables["available"].index[0], "entry_cost"] = -.1
    with pytest.raises(ValueError):
        matched_frontier_metrics(**tables)


def test_configurations_with_one_observation_do_not_claim_a_sample_standard_deviation() -> None:
    tables = _tables()
    day = "2024-04-01"
    tables = {name: frame.loc[frame.origin.eq(day)] for name, frame in tables.items()}
    result = matched_frontier_metrics(**tables)
    assert (result.paired_origins == 1).all()
    assert result.std_pnl.isna().all() and not result.performance_available.any()
    assert len(result) == 4
    assert set(PORTFOLIO_KEYS).issubset(tables["requested"].columns)


def test_pending_computation_is_not_economic_infeasibility_or_a_completed_zero_return() -> None:
    tables = _tables()
    selected = tables["available"].variant.eq("orthogonal") & tables["available"].origin.eq("2024-04-04")
    tables["pending"] = tables["available"].loc[selected].copy()
    tables["pending"]["costed_pnl"] = np.nan
    tables["available"] = tables["available"].loc[~selected]
    metrics = matched_frontier_metrics(**tables, comparison_sample="common")
    assert set(metrics.paired_origins) == {2}
    assert (metrics.loc[metrics.variant.eq("orthogonal"), "pending_origins"] == 1).all()
    assert metrics.unavailable_origins.sum() == 1
    assert (metrics.requested_origins == metrics.completed_origins + metrics.unavailable_origins + metrics.pending_origins).all()
    tables["pending"]["costed_pnl"] = 0.
    with pytest.raises(ValueError, match="Pending portfolios"):
        matched_frontier_metrics(**tables)


def test_an_entirely_pending_variant_still_limits_the_common_sample() -> None:
    tables = _tables()
    selected = tables["available"].variant.eq("orthogonal")
    tables["pending"] = tables["available"].loc[selected].copy()
    tables["pending"]["costed_pnl"] = np.nan
    tables["available"] = tables["available"].loc[~selected]
    metrics = matched_frontier_metrics(**tables, comparison_sample="common")
    assert (metrics.paired_origins == 0).all()
    assert metrics.std_pnl.isna().all()
    assert set(metrics.variant) == {"ordinary", "orthogonal"}


def test_labels_and_equal_inception_price_do_not_override_a_claim_definition_mismatch() -> None:
    tables = _tables()
    tables["available"]["contract_definition_sha256"] = "1" * 64
    tables["reference"]["contract_definition_sha256"] = "1" * 64
    metrics = matched_frontier_metrics(**tables)
    assert (metrics.paired_origins == 3).all()
    tables["reference"].loc[0, "contract_definition_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="claim definition and numeraire"):
        matched_frontier_metrics(**tables)


def test_common_cohort_intersects_different_requested_date_ranges() -> None:
    tables = _tables()
    for name in ("requested", "available"):
        frame = tables[name]
        remove = frame.variant.eq("orthogonal") & frame.greek_budget.eq(1.) & frame.origin.eq("2024-04-01")
        tables[name] = frame.loc[~remove]
    result = matched_frontier_metrics(**tables, comparison_sample="common")
    assert set(result.paired_origins) == {2}
    assert result.reference_std_pnl.nunique() == 1
