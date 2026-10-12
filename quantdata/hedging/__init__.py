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
from quantdata.hedging.inference import (
    EpisodeInferenceConfig,
    EpisodeLossSummary,
    HacKernel,
    MeanConfidenceInterval,
    hac_long_run_variance,
    hac_standard_error_of_mean,
    moving_block_bootstrap_mean_interval,
    paired_episode_differences,
    summarize_episode_losses,
)
from quantdata.hedging.metrics import HedgeSmokeResult, assert_smoke_fixture
from quantdata.hedging.objectives import (
    CostLassoConfig,
    CovarianceEstimate,
    LinearHedge,
    TylerConfig,
    cost_lasso_minimum_variance_hedge,
    regularized_tyler_covariance,
)
from quantdata.hedging.protocols import BenchmarkConfig, EpisodeSpec, HedgeSchedule

__all__ = [
    "BarrierOptionClaim",
    "BenchmarkConfig",
    "ClaimPnlBreakdown",
    "CostLassoConfig",
    "CovarianceEstimate",
    "DigitalClaim",
    "EpisodeSpec",
    "EpisodeInferenceConfig",
    "EpisodeLossSummary",
    "HacKernel",
    "HedgeSchedule",
    "HedgeSmokeResult",
    "MarketSnapshot",
    "MeanConfidenceInterval",
    "LinearHedge",
    "OptionContract",
    "OptionQuote",
    "OptionRight",
    "RealizedVarianceClaim",
    "TradeExecution",
    "TylerConfig",
    "claim_pnl_breakdown",
    "assert_smoke_fixture",
    "cost_lasso_minimum_variance_hedge",
    "eligible_option_quotes",
    "execute_option_trade",
    "execute_spot_trade",
    "hac_standard_error_of_mean",
    "hac_long_run_variance",
    "moving_block_bootstrap_mean_interval",
    "normalized_strike_from_nms",
    "paired_episode_differences",
    "regularized_tyler_covariance",
    "settlement_cashflow",
    "summarize_episode_losses",
]
