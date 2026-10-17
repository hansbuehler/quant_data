"""Daily Black delta/vega hedge on the common public market panel."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.config import (
    BsGreeksConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.greeks import (
    black_digital_greeks,
    black_option_greeks,
    interpolate_vega_hedge,
    strike_interpolation,
)


def run_bs_greeks_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target_maturity: int | None = None,
    config: BsGreeksConfig = BsGreeksConfig(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Hedge each digital with two bracketing vanillas and the one-day spot gain."""
    require_public_hedging_panel(panel)
    targets = daily_digital_targets(maturity_business_days=target_maturity)
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    day = samples[0].transition.origin
    quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != quotes for sample in samples[1:]):
        raise RuntimeError("Black-Greek targets do not share a hedge universe")
    expiry_index = {expiry: index for index, expiry in enumerate(day.expiry_dates)}
    quote_greeks = tuple(
        black_option_greeks(
            quote,
            forward_over_spot=float(day.forward_over_spot[expiry_index[quote.contract.expiry_date]]),
            discount_factor=float(day.discount_factors[expiry_index[quote.contract.expiry_date]]),
        )
        for quote in quotes
    )
    results: list[HedgeSmokeResult] = []
    for sample, quantity in zip(samples, quantities, strict=True):
        target = sample.target
        target_greeks = black_digital_greeks(
            strike_over_forward=sample.target_strike_over_forward,
            annual_vol=day.atm_vol_at_business_dte(target.maturity_business_days),
            business_dte=float(target.maturity_business_days),
            discount_factor=sample.target_discount_factor,
            right=target.right,
        )
        bracket = strike_interpolation(
            quotes,
            fixed_strike=sample.claim.strike,
            target_business_dte=float(target.maturity_business_days),
        )
        spot_action, option_actions, delta_residual, vega_residual, cap_hits = (
            interpolate_vega_hedge(
                target=target_greeks,
                quote_greeks=quote_greeks,
                bracket=bracket,
                claim_quantity=float(quantity),
                position_cap=config.position_cap,
                minimum_vega=config.minimum_vega,
            )
        )
        tolerance = 1.0e-8 * max(1.0, float(quantity))
        results.append(
            evaluate_daily_digital_hedge(
                sample,
                model="bs-delta-vega",
                measure="Q-price/historical-transition",
                spot_action=spot_action,
                option_actions=option_actions,
                model_price=target_greeks.price,
                objective_value=abs(vega_residual),
                converged=(
                    cap_hits == 0
                    and abs(delta_residual) <= tolerance
                    and abs(vega_residual) <= tolerance
                ),
                cap_hits=cap_hits,
                claim_quantity=float(quantity),
                diagnostics={
                    "claim_quantity": float(quantity),
                    "hedge_instrument_count": len(quotes),
                    "bracket_contract_ids": [
                        quotes[index].contract.contract_id for index, _ in bracket
                    ],
                    "bracket_distance_weights": [weight for _, weight in bracket],
                    "bracket_fixed_strikes": [
                        quotes[index].contract.fixed_strike for index, _ in bracket
                    ],
                    "delta_residual": delta_residual,
                    "vega_residual": vega_residual,
                    "vega_semantics": (
                        "parallel annualized Black ATM-volatility shift; vanillas "
                        "use their own expiry ATM volatility"
                    ),
                    "action_semantics": (
                        "trade the two surrounding options in strike-distance "
                        "weights to neutralize target vega, then spot to neutralize "
                        "the delta of the digital and both options"
                    ),
                    "payoff_units": "discounted origin-PV per unit origin spot",
                },
            )
        )
    return tuple(results)
