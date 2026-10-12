"""Model-neutral contracts, claims, accounting, and benchmark protocols."""

from quantdata.hedging.accounting import (
    ClaimPnlBreakdown,
    TradeExecution,
    claim_pnl_breakdown,
    execute_option_trade,
    execute_spot_trade,
    settlement_cashflow,
)
from quantdata.hedging.claims import (
    BarrierOptionClaim,
    DigitalClaim,
    RealizedVarianceClaim,
    normalized_strike_from_nms,
)
from quantdata.hedging.contracts import (
    MarketSnapshot,
    OptionContract,
    OptionQuote,
    OptionRight,
    eligible_option_quotes,
)
from quantdata.hedging.protocols import BenchmarkConfig, EpisodeSpec, HedgeSchedule

__all__ = [
    "BarrierOptionClaim",
    "BenchmarkConfig",
    "ClaimPnlBreakdown",
    "DigitalClaim",
    "EpisodeSpec",
    "HedgeSchedule",
    "MarketSnapshot",
    "OptionContract",
    "OptionQuote",
    "OptionRight",
    "RealizedVarianceClaim",
    "TradeExecution",
    "claim_pnl_breakdown",
    "eligible_option_quotes",
    "execute_option_trade",
    "execute_spot_trade",
    "normalized_strike_from_nms",
    "settlement_cashflow",
]
