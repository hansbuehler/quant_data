"""Conventional GROMM Q pricing with rolling eight-factor DLV-vega hedges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    load_gromm2_market_state,
    price_gromm2_daily_target,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
    load_rolling_gromm2,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.adapter import (
    run_sanos_lv_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig


def run_gromm2_conventional_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    artifact_root: Path,
    config: SanosLvConfig,
    rolling_config: RollingGromm2Config = RollingGromm2Config(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Price under dated GROMM Q and hedge by eight rolling DLV factors."""
    require_public_hedging_panel(panel)
    artifacts = load_rolling_gromm2(
        artifact_root,
        as_of_date=origin_date,
        config=rolling_config,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
    )
    market_state = load_gromm2_market_state(artifacts)
    reference_results = run_sanos_lv_daily_book(
        panel,
        origin_date=origin_date,
        config=config,
        claim_quantities=claim_quantities,
    )
    targets = daily_digital_targets()
    if len(reference_results) != len(targets):
        raise RuntimeError("conventional GROMM target book does not align")
    artifact_fingerprints = {
        "affine_history_sha256": file_sha256(artifacts.paths.affine_history),
        "dynamics_sha256": file_sha256(artifacts.paths.dynamics),
    }
    results = []
    for target, result in zip(targets, reference_results, strict=True):
        sample = build_daily_digital_sample(
            panel,
            origin_date=origin_date,
            target=target,
        )
        model_price = price_gromm2_daily_target(market_state, sample=sample)
        pricing_difference = result.claim_quantity * (
            model_price - result.market_price
        )
        diagnostics = {
            **result.diagnostics,
            **artifact_fingerprints,
            "price_semantics": (
                "exact GROMM2 Q expectation from the current dated density state"
            ),
            "hedge_semantics": (
                "minimum predictable-cost exact first-eight rolling DLV-factor "
                "hedge, then spot delta"
            ),
            "gromm_factor_rank": market_state.dynamics.factor_count,
            "dlv_hedge_factor_rank": config.rolling_factors.factor_rank,
        }
        results.append(
            replace(
                result,
                model="gromm-q-dlv-vega",
                measure="Q-price/historical-transition",
                model_price=model_price,
                pricing_difference=pricing_difference,
                pure_hedge_error=result.hedged_pnl - pricing_difference,
                diagnostics=diagnostics,
            )
        )
    return tuple(results)
