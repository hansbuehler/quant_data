# Generated-model rolling initializer/update quant review

## Verdict

**PASS.** The fixed-source rolling VolGAN and diffusion implementation is
quantitatively coherent and ready for the planned GPU initializer, 2022 budget
selection, and 2023--2025 production runs. No correctness blocker remains in
the reviewed scope.

This is a code-readiness verdict, not a claim that the full GPU workloads have
already completed. The final wall-clock and VRAM envelope must still be
measured under `gpuctl` during the first material diffusion run.

## Scope

- `models/volgan`
- `models/diffusion`
- `runners/train_generated.py`
- `runners/select_generated_budget.py`
- `runners/run_daily.py`
- `runners/config.py`
- focused tests and model/runner documentation

No GPU was used for this review.

## Findings

### 1. Timing and no-lookahead gates pass

- Initializer training is rejected after 2022-12-30
  (`runners/train_generated.py:66`).
- Update-budget selection rejects post-2022 validation dates and requires the
  immutable source checkpoint to predate every selected validation origin
  (`runners/select_generated_budget.py:119`,
  `runners/select_generated_budget.py:144`).
- Every VolGAN and diffusion rolling source must be complete, compatible, and
  strictly earlier than the current training-window end
  (`models/volgan/training.py:724`, `models/volgan/training.py:744`,
  `models/diffusion/training.py:725`, `models/diffusion/training.py:742`).
- A diffusion fine-tuning base may end on the current date, because all of that
  date's released state is available at the decision time, but a future base is
  rejected (`models/diffusion/training.py:498`,
  `models/diffusion/training.py:518`).
- Rolling training data and normalization use only the chronological training
  block; the newest block remains validation data
  (`models/volgan/training.py:126`, `models/diffusion/training.py:153`).

### 2. Normalization rebasing is algebraically exact

For old and new condition normalizations,

\[
z_o=\frac{c-m_o}{s_o}
   =\frac{s_n}{s_o}z_n+\frac{m_n-m_o}{s_o}.
\]

The VolGAN generator therefore transforms its condition columns and first-layer
bias by exactly this affine map. Its output layer applies

\[
z_{y,n}=\frac{m_{y,o}+s_{y,o}z_{y,o}-m_{y,n}}{s_{y,n}},
\]

so decoded economic targets are unchanged. The discriminator applies the same
old-coordinate reconstruction to both standardized conditions and targets
(`models/volgan/training.py:771`, `models/volgan/training.py:814`). The
regression checks both decoded generator outputs and discriminator logits
(`tests/quantdata/sdlv/daily/hedging/models/test_volgan.py:136`).

The diffusion rebase performs the corresponding channelwise affine change on
the flattened input projection, output projection, output bias, and the
fine-tuning adapter's up-projection. Thus, for the same economic noisy tensor,
the rebased denoiser returns the same economic clean prediction
(`models/diffusion/training.py:775`,
`tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py:327`). The base EMA
is first rebased to the current window before adapter training, and the
fine-tuning regression verifies current-window normalization and future-base
rejection (`models/diffusion/training.py:520`,
`tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py:215`).

This exact statement concerns the denoiser's economic input/output map. A
standard-normal diffusion noise law is normalization-coordinate dependent, so
it should not be described as invariance of an already trained model's entire
reverse-diffusion path law. Subsequent current-window updates are what adapt
that law.

### 3. Checkpoint identity and resume semantics pass

- Exact training arrays and dates, model identity, precision, base SHA-256, and
  warm-source SHA-256 enter destination checkpoint identity and compatibility
  checks (`models/volgan/training.py:75`, `models/volgan/training.py:92`,
  `models/diffusion/training.py:85`, `models/diffusion/training.py:111`).
- The daily runner pins source hashes once and supplies the expected hashes to
  every origin; the budget selector does the same for every candidate/origin
  (`runners/run_daily.py:367`, `runners/run_daily.py:401`,
  `runners/run_daily.py:459`, `runners/run_daily.py:500`,
  `runners/select_generated_budget.py:137`,
  `runners/select_generated_budget.py:183`).
- Loaders hash before and after deserialization, preventing a concurrent atomic
  replacement from mixing weights and recorded identities
  (`models/volgan/training.py:713`, `models/volgan/training.py:720`,
  `models/diffusion/training.py:487`, `models/diffusion/training.py:490`,
  `models/diffusion/training.py:708`, `models/diffusion/training.py:715`).
- Diffusion base checkpoints also carry a shared base/fine-tuning semantic
  geometry identity. It covers the grid, sequence/channel geometry,
  transformer geometry, dropout, and diffusion-step count, so same-shaped but
  economically different bases are rejected
  (`models/diffusion/training.py:510`, `models/diffusion/training.py:759`).
- Forced interruptions reproduce uninterrupted VolGAN generator/discriminator
  and diffusion model/EMA states exactly
  (`tests/quantdata/sdlv/daily/hedging/models/test_volgan.py:278`,
  `tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py:429`).

### 4. One-standard-error budget selection passes

For each precommitted epoch budget, the selector computes the mean validation
loss and sample standard error across the same 2022 origins. It then chooses the
cheapest budget satisfying

\[
\bar L_e \leq \bar L_{e^\star}+\operatorname{SE}(\bar L_{e^\star}),
\]

where \(e^\star\) has the smallest mean validation loss
(`runners/select_generated_budget.py:53`,
`runners/select_generated_budget.py:81`). The deterministic regression verifies
the cheapest eligible choice
(`tests/quantdata/sdlv/daily/hedging/test_generated_budget_selection.py:18`).

The reported standard error is the ordinary sample standard error, not HAC.
Because rolling validation blocks can overlap, it is best interpreted as the
precommitted model-selection heuristic rather than as an independent-sample
confidence interval.

### 5. Production path is feasible by construction

- Production origins start independently from one fixed immutable initializer,
  avoiding both full cold retraining and error propagation between evaluation
  dates.
- Diffusion rolling updates use zero scheduler warm-up, matching the algorithm
  used in 2022 budget selection; the full 1,600-step warm-up remains confined to
  initializer training (`runners/run_daily.py:426`,
  `runners/select_generated_budget.py:314`). VolGAN similarly skips initializer
  gradient matching on a rolling warm start (`runners/run_daily.py:328`).
- Daily fit checkpoints and budget-selection fit checkpoints are reproducible
  and removed by default after their fingerprints and results are retained;
  explicit flags retain them for diagnosis
  (`runners/run_daily.py:119`, `runners/select_generated_budget.py:114`,
  `runners/select_generated_budget.py:230`).
- Training progress defaults to 240 seconds, inside the host's five-minute GPU
  reporting requirement.

Operationally, production must supply the end-2022 initializer and the selected
epoch budget. The CLI still permits an explicit cold-training invocation, which
is useful for one-origin diagnostics but is not the documented production
procedure. Diffusion's 200 reverse steps times 4,096 scenarios will likely
dominate production runtime; the first managed-GPU origin should therefore be
used to record observed VRAM, training time, sampling time, and ETA before the
full date range is launched.

## Independent verification

- Focused CPU suite: **29 passed** in 23.09 seconds.
- Python compilation: passed for all reviewed model and runner modules.
- Ruff import-order check: passed for all reviewed modules and tests.
- `git diff --check`: passed for the reviewed scope.
- Resource check: RAM and disk were `OK`; `gpuctl` reported that its daemon was
  unreachable and used the local measurement fallback. No GPU work was started.
