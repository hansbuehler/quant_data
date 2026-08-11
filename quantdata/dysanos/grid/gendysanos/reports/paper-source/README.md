# DYSANOS paper source reports

This directory retains the machine-readable reports used by the arbitrage
tables in `tex/arbitrage.tex`. Large JSON files are stored as gzip-compressed
archives to keep the working tree small.

- `to-expiry/` contains the 24 statewise multi-expiry reports underlying
  Tables 2 and 3: DYSANOS and PCA-IV, daily and weekly trading, 100,000 and
  one million paths, and costs of 0, 1, and 10 basis points.
- `period/` contains the corresponding 24 raw-coordinate period reports
  underlying Tables 4 and 5.

Each archive contains one JSON document. Read one without restoring it using
`gzip -cd report.json.gz`, or restore it using `gzip -dk report.json.gz`.
The experiment programs recreate their original uncompressed report
directories when rerun.
