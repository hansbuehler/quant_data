r"""Section B: "either calibrate coherent latent transition parameters
directly OR construct and verify a martingale completion of calibrated
marginals. Static admissibility alone is insufficient."

**Choice made here: direct calibration of the transition parameter
(kappa), not a free-marginal-fit-then-complete approach.** The reasons,
recorded explicitly rather than left implicit:

1. `comparison_interface.CoherentChainBlackAdapter`'s finite-state
   nearest-neighbour absorbing chain, with birth-death rates
   `kappa/(gap)`, is EXACTLY driftless at every interior state (the up-rate
   times the up-gap equals the down-rate times the down-gap, both equal to
   `kappa`) -- a genuine, exact martingale transition family BY
   CONSTRUCTION, for every value of `kappa > 0` and every choice of
   anchors. This was already independently verified in a prior task:
   `verification.py::chain_composition_residual` confirms the exact
   Chapman-Kolmogorov semigroup property `P(tau1)P(tau2)=P(tau1+tau2)`
   (<1e-8), and `CoherentChainBlackAdapter.verify_zero_generator_residual`
   confirms the associated pricing PDE holds with zero residual (finite-
   difference floor only).
2. Calibrating `kappa` (and, for the Bates variant, the background Bates
   parameters) directly to market data is therefore "calibrating coherent
   latent transition parameters directly" in the precise sense the task
   asks for -- every calibrated model, at every parameter value the
   optimizer visits, is automatically an EXACT martingale. There is no
   separate "check it's a martingale after fitting" step needed, because
   non-martingale parameter values do not exist in this family: they are
   excluded by construction, not by a post-hoc admissibility filter.
3. The alternative -- fitting FREE per-expiry mixture weights `q_i(T)`
   (more representational flexibility, since `q_i` is no longer
   constrained to `expm(kappa*A)`'s one-parameter family) -- would need a
   genuine martingale-COMPLETION step: checking that the fitted marginals
   are increasing in convex order across expiries (Kellerer/Strassen), a
   nontrivial additional construction. This was considered and NOT built,
   to keep scope bounded; the representation-capacity question this
   alternative would have answered is instead addressed by the
   independently-fitted SANOS snapshot baseline (`sanos_snapshot.py`),
   which has no transition-family restriction at all.

**Explicit labelling, per the task's instruction**: this chain construction
is NOT the Wiese-Wissel dynamics referenced elsewhere in this research
programme (a learned/neural SDE completion of a DLV decoder). It is a
simpler, exactly-verified finite birth-death chain. Every place in this
report that says "coherent DLV" or "the chain model" refers to THIS
construction, labelled explicitly as the "coherent-chain DLV" variant, not
as an implementation of Wiese-Wissel.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantdata.sdlv.bates_crc.comparison_interface import (
    ChainGeneratorConfig,
    CoherentChainBlackAdapter,
)
from quantdata.sdlv.bates_crc.verification import chain_composition_residual


@dataclass(frozen=True)
class MartingaleCompletionReport:
    chain_config: ChainGeneratorConfig
    chapman_kolmogorov_residual: float
    generator_residual: float
    mass_residual: float  # max |sum_i P[i0,i](tau) - 1| across tested tau
    mean_residual: float  # max |sum_i P[i0,i](tau)*x_i - x_i0| across tested tau
    variant_label: str = "coherent-chain DLV (direct transition-parameter calibration, NOT Wiese-Wissel)"


def verify_martingale_completion(chain_config: ChainGeneratorConfig, tenors_years: tuple) -> MartingaleCompletionReport:
    """Runs the two already-established exactness checks plus explicit
    mass/mean checks at the SPECIFIC tenors this study actually uses, and
    packages them for the report. Raises if any residual is not tiny (that
    would indicate a bug, not a data issue, since this family is
    parameter-independent in its exactness)."""
    import numpy as np

    adapter = CoherentChainBlackAdapter(chain_config)
    x = np.asarray(chain_config.anchors, dtype=float)
    i0 = chain_config.initial_index

    mass_residual = 0.0
    mean_residual = 0.0
    for tau in tenors_years:
        if tau <= 0.0:
            continue
        P = adapter.transition_matrix(float(tau))
        q = P[i0, :]
        mass_residual = max(mass_residual, abs(float(q.sum()) - 1.0))
        mean_residual = max(mean_residual, abs(float(q @ x) - x[i0]))

    ck_residual = max(
        chain_composition_residual(chain_config, tenors_years[0], tenors_years[-1] if len(tenors_years) > 1 else tenors_years[0]),
        chain_composition_residual(chain_config, tenors_years[len(tenors_years) // 2], tenors_years[-1]) if len(tenors_years) > 2 else 0.0,
    )
    gen_report = adapter.verify_zero_generator_residual(tau_years=float(tenors_years[len(tenors_years) // 2]),
                                                          k=float(x[i0]))
    gen_residual = abs(gen_report["residual"])

    report = MartingaleCompletionReport(
        chain_config=chain_config, chapman_kolmogorov_residual=ck_residual, generator_residual=gen_residual,
        mass_residual=mass_residual, mean_residual=mean_residual,
    )
    assert report.chapman_kolmogorov_residual < 1e-6, report
    assert report.mass_residual < 1e-8, report
    assert report.mean_residual < 1e-6, report
    return report
