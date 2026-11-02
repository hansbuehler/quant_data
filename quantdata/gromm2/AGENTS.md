# GROMM2 methodological fidelity

The paper methodology and the user's explicit specifications govern model
dynamics. No undocumented deviations are permitted.

- Do not silently add drift taming, coefficient clipping, density-dependent
  loading changes, probability projections, return centering, fitted path
  weights, or other safeguards that alter the specified stochastic law.
- Before a paper-faithful production run, reconcile its dynamics and numerical
  scheme against the relevant paper equations. Record the source equations,
  implementation identity and any differences in the external run directory.
- Flag methodological differences to the user for discussion before changing
  or using them in a new production run; documentation elsewhere is not a
  substitute for agreement on the governing paper methodology.
- If the specified scheme is unstable, record the failure and investigate it.
  Do not silently substitute a stable but different model. Obtain explicit user
  approval for proposed methodological deviations and document them separately.
- Historical results from modified schemes must remain provenance-labelled;
  do not relabel them as validation of the unchanged paper model.
- Zero-action or zero-variance conditional diagnostics are uninformative, not
  successful martingale tests. Report their status explicitly.
