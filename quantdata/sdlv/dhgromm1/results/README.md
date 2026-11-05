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

The no-a10 and P comparisons are running/queued, not included as completed
results. Paths and sequence are documented in the parent README. Old
premium-control experiment snapshots have been removed.

`python -m quantdata.sdlv.dhgromm1.retain_result --source <completed-run>
--output <new-results-directory>` intentionally retains small final outputs.
Raw paths, weights, caches, logs, scratch and LaTeX-generated PDFs stay outside Git.
