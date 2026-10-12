# Daily front-spot stochastic DLV

This package estimates and simulates the 25-expiry by 200-strike model described
in `quantdata/sdlv/tex/main2.tex`.

## Estimation convention

- Each historical observation is one exact market-session pair.
- Inside a pair, expiry and cash strike are fixed.  If the origin relative
  strike is `x` and the realized gross spot return is `R`, the endpoint surface
  is interpolated at `x / R`.
- Across pairs the fitted support floats with that day's market state.
- The full/direct DLV fit is the production default.  Its rank-8 2025
  out-of-sample residual RMS was 0.256, versus 0.423 for affine-200.
- The front spot shock is `Phi^-1(F_t^1(S_{t+1}/S_t))`, where `F_t^1` is the
  fitted one-day DLV mixture including the independent 5% Black background.
- Functional PCA is applied to locally smoothed fixed-contract residual
  surfaces.  The default Gaussian smoothing widths are one expiry and two
  strike nodes; raw and lag-one-corrected estimators remain available.

The first Brownian factor is the front-spot factor.  The remaining factors are
orthogonal surface shocks.  The risk-neutral DLV drift is recomputed node by
node from the density recursion.  Under `dW_Q = dW_P + lambda dt`, the physical
drift is constrained to `m_P = m_Q + xi lambda`.  Both a statewise projection
and a stable global constant `lambda` are estimated; simulation uses the latter.
The independent 5% Black process has zero risk premium and is identical under P
and Q.

## Numerical scheme

The front density uses the exact Gaussian information-bridge transition.  The
DLV-squared SDE uses the paper's frozen log-Euler update whenever the absolute
relative-drift increment is at most `0.5`.  Only larger increments are tamed by
continuing linearly in log scale as `sign(x) * (0.5 + log(1 + |x| - 0.5))`.
This leaves ordinary paths unchanged while preventing an exceptional
near-zero-density tail drift from overflowing as though it stayed frozen over
the whole substep.  The default daily step uses eight uniform intraday DLV
substeps, recomputes the Q drift along the bridge, reconstructs all future
densities algebraically, and drops the expired slab.  Mass and forward errors
in CPU pilots were below `2e-13`.

Raw nodewise DLV squared can become very large in low-probability regions under
Q.  This is reported rather than clipped.  In the current CPU pilots the
density-weighted variance remained much smaller, but tail behavior must remain
part of the 100,000-path validation.

## Commands

```bash
JAX_ENABLE_X64=1 python -m quantdata.sdlv.daily.front_spot.run calibrate
JAX_ENABLE_X64=1 python -m quantdata.sdlv.daily.front_spot.run pilot --paths 32
JAX_ENABLE_X64=1 python -m quantdata.sdlv.daily.front_spot.run audit --paths 1000
```

The audit checks:

1. every live daily latent-density coordinate;
2. every live daily latent call price, computed in `O(N)` by reverse sums;
3. every option payoff through expiry at all 200 strikes, including the 5%
   Black background; and
4. every daily spot return.

All live coordinates remain in the report.  A z-test is only assigned where
the simulation contains at least 20 expected effective observations.  Front
density and front-option tests use predictable conditional variance, and
terminal option tests use the analytic Black-mixture payoff variance.  Because
a rare high-payoff draw can still violate the Gaussian approximation even when
the expected exercise count exceeds 20, inference uses the larger of that
known standard error and the realized empirical standard error.  The report
also retains the maximum absolute mean error across every coordinate,
including those below the statistical-identifiability threshold.

The 100,000-path run is batched and can be executed on CPU while the GPU is
occupied:

```bash
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
python -m quantdata.sdlv.daily.front_spot.run audit \
  --paths 100000 --batch-size 4096 --substeps 8
```
