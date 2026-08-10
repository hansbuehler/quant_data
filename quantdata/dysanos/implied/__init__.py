"""Direct log-implied-volatility PCA benchmark for DYSANOS."""

from .comparison import write_aggregate_comparison
from .decoder import PCATotalVarianceDecoder, WingHitDiagnostics
from .experiment import ImpliedPCAExperiment, PCAFitConfig
from .pca import (
    LogIVPCA,
    PCAFitDiagnostics,
    ValidatedPCAMarket,
    fit_log_iv_pca,
    validate_pca_market,
)
from .simulator import ImpliedPCASampler
from .static_arbitrage import (
    StaticArbitrageAccumulator,
    check_sanos_static_control,
    check_static_arbitrage,
)

__all__ = [
    "ImpliedPCAExperiment",
    "ImpliedPCASampler",
    "LogIVPCA",
    "PCAFitConfig",
    "PCAFitDiagnostics",
    "PCATotalVarianceDecoder",
    "StaticArbitrageAccumulator",
    "ValidatedPCAMarket",
    "WingHitDiagnostics",
    "check_static_arbitrage",
    "check_sanos_static_control",
    "fit_log_iv_pca",
    "validate_pca_market",
    "write_aggregate_comparison",
]
