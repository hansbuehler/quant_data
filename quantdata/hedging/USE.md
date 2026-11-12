# Loading the April–May 2024 model audit

This audits retained model laws, not a restart of the paused hedging experiment
or a validation of the corrected hedge2 design. The selected DYSANOS law uses
sixteen PCA/AR factors; its daily native surface fits remain unchanged. Other
generators retain their original fitted parameters.
At the user's request, the 22 missing May AD-Seq-Vol-FT fits were added using
the existing causal one-epoch rolling recipe and previous-date warm starts.

## Current artifact location

The complete report and small diagnostic tables are at:

`/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261111-final16-v3/`

The intentionally retained Git artifacts are under
`quantdata/hedging/reports/model_dynamics_2024_aprmay/`. This includes the PDF,
LaTeX, plots, aggregate diagnostics and checksummed manifests, not licensed quote
rows, model tensors or recursive scenario arrays. `RETAINED_ARTIFACTS.json`
identifies the exact external run and verifies every retained file.

The earlier `model-dynamics-audit-20261111-v1` run is retained as provenance.
Its unequal 128/115-session path comparison and extra IV-interiority rejection
are superseded. The intermediate `model-dynamics-audit-20261111-common115-v2`
completed every other model before the sixteen-factor promotion. The final
selection reuses those identical 32-path, 115-session diagnostics and unchanged
one-day forecasts; it does not retrain the other generators.

- `MODEL_DYNAMICS_REPORT.pdf`: definitions, results and model-choice appendix.
- `MODEL_DYNAMICS_REPORT.tex`: the compiled report's source.
- `STATS.md`: compact numerical summary and interpretation limits.
- `summary.json`: full covariance, leverage and tail results by model.
- `summary_matched_dates.json`: companion comparison on common available dates.
- `coverage.csv`: requested versus actually scored origins by diagnostic.
- `daily_scores.csv`, `monthly_scores.csv`, `rolling_scores.csv`: one-day
  log-IV forecast errors and descriptive window deterioration.
- `iv_rmse_buckets.csv`: log-IV and explicitly defined geometric-point
  annual-IV forecast errors by tenor and NMS.
- `reconstruction_scores.csv`, `quote_forecast_scores.csv`: observed-spread
  scores; original quotes are not copied into these files.
- `tail_scores.csv`: quantile losses, exception counts, intervals and tests.
- `linear_stability.csv`: DYSANOS linear-core checks and legacy floor counts.
- `TRAFFIC_LIGHT_RULES.json`: pass/warn/fail tolerances and their interpretation.
- `observable_reference.json`, `frozen_observable_pca.npz`: fixed observable
  coordinates and basis-free ATM-IV persistence checks.
- `daily/`, `quote_forecasts/`, `persistence/`: detailed per-origin outputs,
  exclusions, numerical guards and failure reasons.
- `SOURCE_INPUTS.json`: retained external model/scenario/input paths, sizes and SHA-256.
- `INFERENCE_MODELS.json`: exact EMA-only diffusion replacements, original source
  identities, validation evidence and retirement status.
- `MANIFEST.json`: report/table/code checksums.
- `MODEL_SELECTION.json`, `PROTOCOL.json`, `SOURCE_REUSE.json`: selected model
  ranks, common simulation settings and exact reused diagnostic identities.
- `rank_comparison.json`: the retained eight-versus-sixteen-factor experiment.

Large weights, scenario arrays, training histories and licensed quote panels
remain external. P/Q GROMM share the same fits; do not count or copy them twice.
Keeping only the 44 inference dates does not eliminate their historical input
dependencies. DYSANOS uses our own saved ordinary/spot-orthogonal fits, not
`openqdata_models_sanos`. Fine-tuned diffusion now has 44 trained origins.
Forecast sampling is separate: consult `coverage.csv` for evaluated dates, not
the training count. New May checkpoints and authenticated lineage records are
under `completion/ft-may/`; the managed GPU forecast queue logs to
`completion/ft-may/gpu-queue.log` and refreshes the report after sampling.
See the appendix rather than inferring missing results.

The former diffusion checkpoint totals were rolling restart copies, not a single
model's inference footprint. Each base checkpoint held about 68.7 MiB: 17.1 MiB
model weights, 17.1 MiB EMA weights and 34.3 MiB Adam state. Each FT checkpoint
held about 34.4 MiB, duplicating the frozen backbone and its EMA, although the
rank-four adapter itself has only 2,088 float32 parameters (about 8 KiB).

The inference archive retains the exact EMA tensors actually used for sampling.
Byte-identical backbones are shared by content hash; dated adapters and all
normalization buffers are retained without quantization. Each dated model has
a small JSON manifest with geometry, versions, training dates and checksums.
Before an original checkpoint is retired, every tensor and two complete
seeded 200-step diffusion samples must agree exactly with its exported model.
Open checkpoints/locks are excluded from deletion. `INFERENCE_MODELS.json`
records which originals have actually been retired and their historical hashes.
Optimizer/RNG/non-EMA state from retired files is no longer available for an
exact training restart. Existing result records keep their original hashes as
provenance, rather than being rewritten to pretend exports were the source fits.

## Density-weighted DYSANOS follow-up

The PDF now includes the density-weighted ordinary and spot-orthogonal
sixteen-factor PCA/VAR models. Both have 44/44 surface forecasts, quote forecasts
and recursive tests, using the same 32 paths, 115 sessions and seed as the
baseline comparison. Baseline fitted laws are not replaced. The follow-up lives at:

`/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261112-density-weighted-v1/`

`density_weighted_diagnostics.json` contains the complete aggregate diagnostics;
`density_*.csv` and `density_acf.pdf` / `density_tails.pdf` accompany the retained
report. `DENSITY_WEIGHTED_INPUTS.json` hashes the external dated checkpoints,
metrics and detailed diagnostics. The experimental directory additionally holds
`SPOT_IMPLEMENTATION_CHECK.json`, `PROTOCOL.json` and `COMPLETE.json`.

Each daily metric uses the mean **training-origin node probabilities** for
squared log-DLV errors and unit weights for log-forward-vol coordinates. Fit and
simulation use `sqrt(weight) * u`; native changes are unscaled before pricing.
Load **both** the metric and fitted VAR, never the weighted `dynamics.npz` alone:

```python
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.density_weighted import load_weighted_model

experiment = Path("/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261112-density-weighted-v1")
model = load_weighted_model(experiment,
    "DYSANOS spot-orthogonal (16 factors) density-weighted", date(2024, 4, 1))
with np.load(experiment.parent / "dlv-full-v4-20261106T1130/days/2024-04-01.npz") as saved:
    current_u = saved["raw_u"]
following_u = model.next_native(current_u)
```

The returned rows enumerate the same 252 paired empirical innovations. The
spot-orthogonal fit removes contemporaneous spot projection before PCA and adds
it back during simulation, preserving residual spot/factor covariance. Recursive
paths keep the dated metric fixed. `weighted_report.py` completes the diagnostics
and exports the follow-up tables; `quotes.py` and `persistence.py` accept the
explicit density-weighted labels.

Predictive variance is also compared with **squared forecast errors**, not only
unconditional observed changes. The denominator is `mean(||actual - mean||²)`;
one is the population scale reference only with a correct conditional mean and
covariance. This descriptive ratio is not a separate no-arbitrage test.

## Load a compact diffusion model for inference

```python
from pathlib import Path
from types import SimpleNamespace

from quantdata.hedging.model_audit.inference import model_from_export
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import sample_diffusion_transitions
from quantdata.torch import BaseEnvironment

catalog = json.loads((run / "INFERENCE_MODELS.json").read_text())
entry = next(item for item in catalog["models"].values()
             if item["variant"] == "ad-seq-vol-ft" and item["origin_date"] == "2024-05-31")
model = model_from_export(Path(entry["manifest"]),
                         expected_sha256=entry["manifest_sha256"],
                         environment=BaseEnvironment(precision=32, device="cpu"))
generated = sample_diffusion_transitions(SimpleNamespace(ema_model=model),
                                        observed_sequence=live_context.sequence,
                                        path_count=100, seed=20261111)
```

`live_context.sequence` contains the documented 21 observed surfaces and returns;
it is not a training dataset. Model loading needs neither the original checkpoint
nor the original frozen-base checkpoint. Normalization buffers are placed on the
requested device together with network parameters. Use a managed `gpuctl`
allocation before selecting CUDA. The archive is inference-only: do not pass it
to the old exact-restart training loader.

## Load diagnostics without loading any model

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

run = Path("/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261111-final16-v3")
summary = json.loads((run / "summary.json").read_text())
coverage = pd.read_csv(run / "coverage.csv")
tails = pd.read_csv(run / "tail_scores.csv")
with np.load(run / "frozen_observable_pca.npz", allow_pickle=False) as artifact:
    basis = artifact["basis"]
    training_endpoints = artifact["training_endpoints"]
```

`basis` is fitted before April and stays fixed throughout the audit. It is not
any model's native rolling PCA basis. Reports never concatenate native scores
from different rolling fits.

## Complete the coverage diagnostics

`coverage.csv` counts completed origins, not successful model-quality tests.
It also records minimum and maximum simulated horizons. Each frozen fit is
tested with 32 independent recursive paths over the same 115 sessions. GROMM's
last retained fixed expiry is 125D and the common observable grid extends to
10D; this determines the shared horizon. No new expiries are synthesized. Its aged
surface is decoded by total-variance interpolation at fixed normalized strike,
with a zero-maturity variance anchor and no long-end extrapolation.

```bash
python -m quantdata.hedging.model_audit.recursive --output "$RUN" \
  --models GROMM-Q GROMM-P --refresh-report
gpuctl run --gpus 1 -- python -m quantdata.hedging.model_audit.recursive \
  --output "$RUN" --models AD-Seq-Vol AD-Seq-Vol-FT --device cuda --refresh-report
python -m quantdata.hedging.model_audit.refresh --output "$RUN"
```

Set the explicit project interpreter, `QUANTDATA_ROOT`, `PYTHONPATH` and owned
temporary/cache directories as described above. `queue_ft_forecasts.py` waits
for managed GPU availability, completes the 22 May FT forecast distributions,
then runs both diffusion recursive tests. All original 200 denoising steps are
used for each simulated session. Session-boundary checkpoints preserve the
path-specific contexts, generated surfaces and RNG state. A signal requests a
stop after the current session; restart the same command to continue. Reports
refresh under a lock after each completed origin so independent jobs cannot
overwrite each other's figures during compilation.

The quote identity join recovers an original licensed cash strike only if it
is the unique candidate within the documented float32 serialization bound.
It does not substitute a nearby different option. The maximum round-trip
identity error and unresolved counts are retained per date. This fixes the
previously absent 20 May join without changing model forecasts or source quotes.

The primary DYSANOS stationarity assessment concerns its sixteen floating-volatility
PCA–AR(1) coordinates only, not spot levels. It reports the AR spectral radius,
modal half-lives and stationary covariance. Gaussian stock-return innovations
can contribute to the factor innovation covariance without changing the
stability criterion. The dated JSON files retain stationary means, covariances
and Lyapunov residuals. Separate recursive floating-IV checks preserve the
original saved scenario law; they do not change it to a Gaussian law.

## Controlled sixteen-factor comparison

The separate run is:

`/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261111-rank16-v1/`

It retains the original daily nine-tenor DLV representations, decoder guards,
252 exact-session estimation windows and paired empirical spot innovations.
Only PCA/AR rank increases from eight to sixteen. March 28 initializes causal
factor alignment; all 44 April–May origins are scored. Every dated fit records
its previous initialization date. No tail multiplier or shock inflation is
introduced. This is not a retraining of a joint market-surface model.

`fits/{ordinary,orthogonal}/days/` contains the new inference-ready dynamics.
`RANK_PROTOCOL.json` records the controls; `COMPARATOR_INPUTS.json` records
the snapshotted eight-factor inputs. `rank_comparison.json` and
`RANK_COMPARISON.md` summarize the matched experiment. The JSON retains
same-shock projection/decoder ablations, tail counts, guard counts and the
32-path, 115-session recursive tests. The report includes the comparison only
after all 44 origins have been evaluated.

An unusable ATM inversion now invalidates its own scenario, not every scenario
in its decoding batch. It remains NaN and contributes to the invalid-path
diagnostic; no replacement IV is invented. Original batch-error records are
retained under `superseded_path_errors/` when the same scenarios are re-evaluated
with this finer-grained accounting.

To resume or reproduce with the explicit environment above:

```bash
/home/hans/vqdata/bin/python -u -m quantdata.hedging.model_audit.rank_comparison \
  --source "$RUN" \
  --output /home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261111-rank16-v1 \
  --publish "$RUN"
```

The source and common observable reference have different retained smoothness
settings (0.1 and 0.25). This controlled rank experiment leaves that discrepancy
unchanged. Higher native PCA coverage is not automatically higher log-IV
variance coverage or evidence of tail calibration.

## Locate and verify saved models

```python
import hashlib

registry = json.loads((run / "SOURCE_INPUTS.json").read_text())
for item in registry["files"]:
    if "rolling neural checkpoint" in item["roles"]:
        path = Path(item["path"])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
```

Use the source adapters for inference; do not call a fit function just to load.
`model_audit/run.py` demonstrates version-checked CPU VolGAN loading;
`persistence.py` loads the existing FuNVol and DYSANOS models;
`gromm.py` authenticates shared GROMM fits before calling the existing sampler;
`recover_forecast.py` loads the saved diffusion EMA without training.
HDF5/NPZ readers use read-only access and `allow_pickle=False`. Torch files
that require `weights_only=False` must be trusted local checkpoints authenticated
against the registry; do not load an untrusted downloaded pickle.

## Reproduce into a fresh external run directory

Run from this branch's source checkout with the project interpreter. The
following rebuilds the final selected-model deck from the authenticated,
completed other-model diagnostics and refits the causal sixteen-factor tail
laws. It does not imply that those input artifacts are embedded in Git.

```bash
export QUANTDATA_ROOT=/home/hans/quantdata_data
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/tmp/quant_data/model-dynamics-audit-20261111-v1
export SOURCE="$QUANTDATA_ROOT/quant_data/hedge2/runs/model-dynamics-audit-20261111-common115-v2"
export RANK_RUN="$QUANTDATA_ROOT/quant_data/hedge2/runs/model-dynamics-audit-20261111-rank16-v1"
export OUTPUT="$(mktemp -d "$QUANTDATA_ROOT/quant_data/hedge2/runs/model-audit-final16-XXXXXXXX")"
export MPLCONFIGDIR="$OUTPUT/mplconfig"
gpuctl resources --json
/home/hans/vqdata/bin/python -u -m quantdata.hedging.model_audit.finalize \
  --source "$SOURCE" --rank-run "$RANK_RUN" --output "$OUTPUT" \
  --retain "$OUTPUT/retained-deck"
```

Check resources before material stages and during long inspections as required
by host policy. Reusing completed other-model diagnostics needs no GPU; generating
the full neural recursive tests uses `gpuctl run` and all original 200 denoising
steps. `parallel_recursive.py` partitions origins without overlapping writers;
it never splits a path across workers or changes an origin's RNG. The PDF build
requires `pdflatex`. Completion gates reject missing dates and mismatched path
horizons; genuine model-law failures remain explicit rather than being counted
as quality passes.
