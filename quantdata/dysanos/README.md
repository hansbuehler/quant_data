# dysanos

Dynamic Smooth Arbitrage-Free Non-Parametric Option Surfaces.

The autonomous-model notation reserves
$h=(h^1,\ldots,h^{n_h})$ for the implied-volatility parameters and writes
$s=\log S$ for log-spot. The implementation continues to store the extended
state in the existing order $xstate=(s,h)$.
