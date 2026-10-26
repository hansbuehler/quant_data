# SDLV / GROMM and reference models

This project contains the GROMM implementation and the two reference models
used in its no-arbitrage comparisons:

- `daily/`: the daily-grid GROMM model, calibration, simulation and audits.
- `results/`: deliberately retained audit reports with source hashes and verification.
- `tangent_levy/`: the Tangent Lévy reference implementation and its reports.
- `bates_crc/`: the Bates consistent-recalibration reference implementation
  and its reports.
- `tex/`: the GROMM paper and addendum, with retained publication artifacts.

The comparison packages were moved from `quantdata.dysanos` to this project.
Use `quantdata.sdlv.tangent_levy` and `quantdata.sdlv.bates_crc` in imports and
`python -m` commands. Their tests are under `tests/quantdata/sdlv/`.
For example, from the repository root with the project environment configured:

```bash
/home/hans/vqdata/bin/python -m quantdata.sdlv.tangent_levy.run_v2_same_date --help
/home/hans/vqdata/bin/python -m quantdata.sdlv.bates_crc.gambara_teichmann_study --help
```

## Existing datasets and checkpoints

The source move does not relocate or invalidate the external data, change
model mathematics, or rerun any calibration. Existing storage groups remain
unchanged for checkpoint and report-provenance compatibility. In this host's
configured `QUANTDATA_ROOT` (`/home/hans/quantdata_data`), the retained paper
artifacts remain beneath:

- `training/dysanos/tangent-levy/` (including its retained audit runs).
- `training/dysanos/bates-crc/` and `experiments/dysanos/bates-crc/`.

Historical generated reports may record the original source paths. These
provenance records are preserved; current source commands use the new package
names above. New generated data must remain outside the source worktree in
uniquely identified run directories, as required by the repository instructions.

## Retained v1-(4,3) result

The [29 August 2025 rolling-252 full Q audit](results/gromm_v1_4_3_2025_08_29_rolling252_q_audit.json)
is retained under the user label `v1-(4,3)` (checkpoint label `Gromm-v1-4,3`).
It contains the full report, independent verification, and external artifact paths
and hashes. The run uses 100,000 paths / 50,000 antithetic pairs, 20 business days,
spot plus 44 options, eight factors and eight substeps per day.

The unconditional maximum absolute z score is 3.517385, with 21 of 570
coordinates above 2.576, all on simulated day 6. The conditional squared and
entropy z scores are -0.879815 and -0.879839: neither detects significant drift.
This is not an unqualified pass. Earlier fixed-window and unconditional-only
results remain separate; the paper tables have not been replaced by this record.
