# DYSANOS paper style guide

## Scope and voice

- Keep the paper concise, mathematical, and direct. Preserve the existing authorial voice.
- Present DYSANOS as the first generative market model built from smooth, statically arbitrage-free SANOS surfaces. The paper establishes a simple, useful baseline; it does not claim an optimal SANOS geometry or an optimal state driver.
- State empirical limitations next to empirical successes. Do not add tangential implementation history, failed experiments, or speculative claims.
- Do not discuss constructing or estimating a risk-neutral measure for the dynamic DYSANOS process.
- Distinguish exact mathematical guarantees from empirical diagnostics. SANOS guarantees static arbitrage-freedom of each decoded surface. It does not by itself guarantee dynamic no-arbitrage of the generated price process.

## Strike and expiry notation

Follow the notation already used in the paper and the current SANOS paper.

- Observation date: (t).
- Time-to-expiry: (T). Relative or spot-normalized strike: (K).
- Observed market coordinates: (T_t^\ell) and (K_t^\ell). Market prices and spot are bold, for example (\bm B_t^\ell), (\bm A_t^\ell), (\bm C_t^\ell), and (\bm S_t).
- A fixed tradable contract has calendar expiry (\tau) and cash strike (k). At date (t), its coordinates are (T=\tau-t) and (K=k/S_t). Use (dC_t(\tau,k)) for the return of the same contract through time.
- Internal SANOS model expiries and strikes are hatted: (0=\hat T_0<\cdots<\hat T_M) and (\hat K_j^i).
- Preprocessing or reference-grid quantities are tilded when they must be distinguished from the internal SANOS grid, for example (\tilde T_j) and (\tilde K_j^i).
- Normalized-moneyness coordinates used to position strikes are (z^i).
- Do not exchange the roles of ((T,K)) and ((\tau,k)). Do not remove hats from internal SANOS grid quantities.

## SANOS terminology

- Retain the term *martingale density* for the SANOS cross-sectional density (q=(q_1,\ldots,q_M)), including positivity, unit mass, unit mean, and convex-order conditions.
- Retain the term *martingale transition operator* for the DLV transition between model-expiry densities.
- Keep the paper's (W_j) notation for total volatility and (\mu\in[0,1)) for the SANOS smoothing parameter unless the whole paper is deliberately migrated to the current SANOS notation.
- Static SANOS martingale conditions are not a dynamic equivalent-martingale-measure construction.

## Dynamic-state notation

- ML-SANOS surface state: (h_t=(h_t^1,\ldots,h_t^{n_h})\in\mathbb R^{n_h}).
- Log-spot: (s_t=\log S_t). The symbol (h_t) is reserved for the implied-volatility parameters and never includes spot. Observed states may be denoted by ((\tilde s_t,\tilde h_t)) and simulated states by ((s_t,h_t)).
- The implementation's extended array is ordered as (xstate=(s,h)); this storage convention does not change the mathematical notation.
- Fit the autonomous surface process before spot. Write it as (dh_t=\kappa(m-h_t)dt+\Sigma\,dW_t^h), where (m) is the invariant mean, (\kappa) is the mean-reversion matrix, and (\Sigma) is the diffusion loading.
- Add spot conditionally using drift (\mu), contemporaneous surface-factor loading (\beta), and orthogonal volatility (\varsigma).
- Euler is the introductory discretization. For Gaussian simulation use the exact affine matrix-exponential transition and state its integrated covariance explicitly.
- State that the surface block is stationary only when every eigenvalue of (\kappa) has strictly positive real part. Its invariant covariance (P) solves (\kappa P+P\kappa'=\Sigma\Sigma'). Log-spot is not part of its invariant distribution.

## Arbitrage claims

- Define dynamic arbitrage using predictable self-financing trading in fixed contracts.
- Do not claim that static arbitrage-freedom implies dynamic no-arbitrage. State the additional martingale-price-process condition and give the short expectation proof.
- Describe sign, nearest-neighbour, and importance-sampling checks as empirical diagnostics. They can reject observed sampling arbitrage but do not constitute a theorem of dynamic no-arbitrage.
- Numerical tolerances absorb floating-point error only; state their formula and precision.

## Empirical claims

- State the date range, historical sample count, simulated sample count, time step, factor count, and whether a statistic uses surface levels, changes, or fixed-contract returns.
- Distinguish mean fit, covariance fit, marginal-distribution fit, tail fit, and support coverage.
- For PCA comparisons, identify whether the data are moving normalized-moneyness surfaces or fixed-cash options repriced from one day to the next.
- Eigensurfaces are PCA loadings of implied-volatility surface levels on the (7\times100) normalized-moneyness grid.
- Quantitative surface-driver results are stored in `generated/surface_first_diagnostics.json`; figures derived from them stay in `generated/`.
- The chronological comparison of autonomous and raw-log-spot surface drifts is stored in `generated/surface_spot_drift_comparison.json`.
- Distinguish PCA loadings of IV surface levels from PCA loadings of one-day IV changes on the moving normalized-moneyness grid; the latter are independent of spot.
- Current importance-sampled arbitrage counts are sourced from `../grid/gendysanos/reports/terminal-importance-arbitrage-tests-autonomous-alpha5-1000000/` and `../grid/gendysanos/reports/terminal-importance-arbitrage-tests-autonomous-alphaNone-1000000/`; retain the report name, sample size, weighting law, and tolerance definition when updating them.
