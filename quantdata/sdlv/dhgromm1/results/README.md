# Deliberately retained corrected floating-contract results

`20261105-floating-zero-cost-a10/` is the completed zero-cost Q baseline,
with generic causal LSTM actions plus a0/a10 vectors. `retention.json` binds
each compact report, loss CSV and PNG to its external source by SHA256.
The source report is preserved verbatim, including its original code hashes.
It uses corrected `(close-open)/S_open` gains, not premium returns.

The zero-cost continuation finished the original 500-total epoch target.
Its phase history restarts after the objective change. Independent best
phase epochs: empty 362, long 357, short 355. Selected centered OCEs:
+0.000303270, −0.000287635, −0.000090033. These are training-only results,
not a no-arbitrage pass. MC errors condition on fitted means/policies.

The no-a10 and P comparisons are not retained here; their dated snapshots and
external paths are documented in the parent RESULTS.md. Earlier queue plans
are historical, not a list of currently active jobs. Old premium-control
experiment snapshots have been removed.

`20261106-independent-validation/` retains the completed 100,000-path frozen
policy validation report, including both fixed training cash and globally
profiled OCE. Original training centering is unchanged. The discontinued
bottom-90% model appears only as historical evidence in this verbatim report.

`20261106-conditional-boundary-pilot/` retains the completed CPU pilot for
ordinary DH and the historical L1/L2 control: 1,024 outer paths, weighted
absorbing endpoint continuations, outer-antithetic-pair errors, and a matched
plain-MC comparator. It is an evaluation, not conditional-weighted training.

Each retention manifest binds compact files to their external sources by
SHA256. Neither folder is a result for the new 1 bp premium-cost model,
which was stopped after 128 complete epochs; its checkpoint remains outside Git.

`python -m quantdata.sdlv.dhgromm1.retain_result --source <completed-run>
--output <new-results-directory>` intentionally retains small final outputs.
Raw paths, weights, caches, logs, scratch and LaTeX-generated PDFs stay outside Git.
