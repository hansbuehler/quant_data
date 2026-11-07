"""Incomplete products cannot make an apparent family risk improvement."""

import json

import numpy as np
import pandas as pd

from quantdata.hedge2 import report_frontiers
from quantdata.hedge2.frontier_metrics import CORE_KEYS, PORTFOLIO_KEYS
from quantdata.hedge2.frontier_population import FrontierPopulation
from quantdata.hedge2.report_frontiers import aggregate_frontier_products, build_frontier_report


def _metrics():
    return pd.DataFrame([{"reference": "Unhedged", "variant": "ordinary", "family": "digital-3D", "side": -1,
        "greek_budget": .2, "claim": claim, "performance_available": True, "paired_origins": count,
        "requested_origins": 10, "completed_origins": 7, "unavailable_origins": 1, "pending_origins": 2,
        **{name: value for name in ("mean_pnl", "median_pnl", "std_pnl", "scaled_mad", "mean_entry_cost", "es95_loss",
            "reference_mean_pnl", "reference_std_pnl", "reference_mean_entry_cost")}}
        for claim, count, value in (("first", 4, .1), ("second", 3, .3))])


def test_report_averages_product_statistics_not_pooled_rows() -> None:
    result = aggregate_frontier_products(_metrics()).iloc[0]
    assert result.products == 2 and result.all_products_measured
    assert result.std_pnl == .2
    assert result.minimum_paired_origins == 3 and result.maximum_paired_origins == 4
    assert result.requested_product_origins == 20
    assert result.completed_product_origins == 14 and result.pending_product_origins == 4


def test_a_missing_product_suppresses_family_performance_instead_of_dropping_it() -> None:
    metrics = _metrics()
    metrics.loc[1, "performance_available"] = False
    metrics.loc[1, "paired_origins"] = 1
    metrics.loc[1, "std_pnl"] = np.nan
    result = aggregate_frontier_products(metrics).iloc[0]
    assert not result.all_products_measured and result.products == 2
    assert np.isnan(result.std_pnl) and np.isnan(result.mean_pnl)
    assert result.minimum_paired_origins == 1


def test_interim_pdf_preserves_pending_counts_and_writes_reproducible_tables(tmp_path, monkeypatch) -> None:
    reference = pd.DataFrame([{"origin": day, "family": "digital-3D", "claim": "target", "side": -1,
        "costed_pnl": pnl, "entry_cost": 0., "initial_price": .5}
        for day, pnl in (("2024-04-01", .1), ("2024-04-02", -.2), ("2024-04-03", .3), ("2024-04-04", -.4))])
    requested = pd.DataFrame([{"variant": "ordinary", **row, "greek_budget": budget}
        for row in reference[list(CORE_KEYS)].to_dict("records") for budget in (.2, 1.)])
    outcomes = requested.merge(reference, on=list(CORE_KEYS), validate="many_to_one")
    pending = outcomes.loc[outcomes.origin.eq("2024-04-04")].copy()
    pending["costed_pnl"] = np.nan
    completed = outcomes.loc[~outcomes.origin.eq("2024-04-04")].copy()
    completed["costed_pnl"] *= .5
    episodes = requested.groupby(["variant", "origin", "family"]).size().rename("requested_portfolios").reset_index()
    episodes["pending_portfolios"] = np.where(episodes.origin.eq("2024-04-04"), 2, 0)
    episodes["completed_portfolios"] = 2 - episodes.pending_portfolios
    episodes["unavailable_portfolios"] = 0
    population = FrontierPopulation(requested=requested, available=completed,
        unavailable=pd.DataFrame(columns=[*PORTFOLIO_KEYS, "costed_pnl"]), pending=pending,
        episodes=episodes, evidence={"test_only": True})
    monkeypatch.setattr(report_frontiers, "load_frontier_population", lambda **kwargs: population)
    identity = {**report_frontiers.ECONOMIC_SPECIFICATION, "optimized_premium_cap": 10.,
        "dynamics_directory": "/test/dynamics", **{name: "/test/data" for name in report_frontiers.DATA_KEYS}}
    monkeypatch.setattr(report_frontiers, "load_reference_population", lambda **kwargs: (
        reference, {"test_only": True, "configuration": identity}))
    path = tmp_path / "requested.parquet"
    requested.to_parquet(path, index=False)
    configuration = tmp_path / "configuration.json"
    configuration.write_text(json.dumps({"requested_population": str(path), "sources": {"ordinary": ["test-only"]},
        "expected_identities": {"ordinary": identity}, "references": {"Unhedged": "test-only"}, "allow_partial": True}))
    output = tmp_path / "report"
    result = build_frontier_report(configuration_path=configuration, output_directory=output)
    assert result["requested_portfolios"] == 8 and result["completed_portfolios"] == 6
    assert result["pending_computation"] == 2 and result["unavailable_portfolios"] == 0
    assert (output / "GREEK_FRONTIERS.pdf").read_bytes().startswith(b"%PDF")
    assert len(pd.read_parquet(output / "pending.parquet")) == 2
    summary = pd.read_parquet(output / "summary.parquet")
    assert set(summary.minimum_paired_origins) == {3}
    assert json.loads((output / "provenance.json").read_text())["configuration_selection"] == "none"
