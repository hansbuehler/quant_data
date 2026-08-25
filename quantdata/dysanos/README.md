# dysanos

Dynamic Smooth Arbitrage-Free Non-Parametric Option Surfaces.

The new Torch-native dynamic SANOS project is documented in
[`flex/FLEX.md`](flex/FLEX.md). Its source, experiments, and reports live under
`quantdata/dysanos/flex`.

The autonomous-model notation reserves
$h=(h^1,\ldots,h^{n_h})$ for the implied-volatility parameters and writes
$s=\log S$ for log-spot. The implementation continues to store the extended
state in the existing order $xstate=(s,h)$.
