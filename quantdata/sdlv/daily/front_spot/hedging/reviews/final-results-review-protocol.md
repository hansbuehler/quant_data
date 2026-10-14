# Final quantitative results review protocol

## Purpose

This protocol is the independent acceptance procedure for the final public-data
hedging benchmark. It is applied only after the canonical daily and weekly
production matrices have terminal outcomes and the final report bundle has been
generated. A visually plausible PDF, a successful renderer, or a partial matrix
is not acceptance evidence.

## Immutable inputs

The reviewer records and authenticates all of the following before reading any
headline result:

1. the full reporting Git commit and a clean detached worktree at that commit;
2. this protocol's separately precommitted SHA-256, its repository commit, and
   the reviewed launcher hash which requires that exact protocol;
3. the public dataset path and manifest SHA-256;
4. the version-2 matrix-commitment path and its separately persisted SHA-256;
5. the exact daily and weekly matrix paths and SHA-256 values bound by that
   commitment;
6. the exact daily and weekly failure-ledger paths and SHA-256 values bound by
   that commitment; and
7. the final report bundle manifest, TeX, bibliography, PDF, JSON summaries,
   and generated appendix-table hashes.

The repository `_validate_matrix_commitment` path must pass using these exact
inputs. The reviewer rejects a recomputed, replaced, or post-outcome commitment,
an uncommitted worktree, or any path/hash mismatch. The reviewed launcher hash,
service journal, persistent pipeline log, and queue state must prove that the
fixed commitment SHA was written before every final generated/GROMM queue-start
event and before every result in those fresh output roots. The self-declared
commitment timestamp is not chronology evidence by itself.

## Population reconciliation

The reviewer independently reconstructs both production populations from the
committed matrices and dated records rather than trusting persisted aggregate
counts.

- The daily grid contains exactly 57 specified cases. Every case is either a
  complete authenticated result or one disjoint authenticated terminal failure.
  Every completed case has exactly 611 eligible origins, 20 targets per origin,
  and 12,220 dated target records.
- The weekly grid contains exactly 94 specified cases. Every case is either a
  complete authenticated result or one disjoint authenticated terminal failure.
  Every completed case has exactly 528 eligible origins, 13 targets per origin,
  and 6,864 dated target records. Frozen and restarted schedules remain
  distinct.
- Repository `_validate_production_matrix` must pass for both grids. The
  committed matrices must equal fresh canonical-builder output in every case
  identity and specification: case ID, model, measure, schedule/variant,
  objective, risk parameter, resource class, readiness/gate inputs, command,
  target-notional identity, and target quantities. Equality is not inferred
  from the 57/94 counts.
- Daily records contain exactly the 20 labels returned by the reporting
  commit's canonical daily target builder: 1D and 3D digital puts and calls at
  ATM and the declared four OTM NMS levels on each side. Weekly records contain
  exactly the 13 labels returned by the canonical weekly target builder: ten 5D
  digitals, realized variance, the down-and-out put, and the up-and-out call.
  Their fixed target quantities must match the authenticated 2022-only target
  notional artifacts exactly.
- No case is silently absent, duplicated, relaxed, relabelled as blocked after
  commitment, or represented simultaneously as complete and failed.
- Every numerical-gate failure is reproduced from its dated evidence and is a
  declared cap, convergence, or validation failure rather than an infrastructure
  or implementation crash.

## Financial and statistical reconciliation

For every completed policy, the reviewer recomputes the following from dated
records and requires exact agreement with the authenticated JSON within the
declared floating-point tolerances:

- unhedged, economic, pure-hedging, and pricing loss distributions, wherever
  each component is defined;
- observations, mean, median, sample standard deviation, RMSE, MAE, 95% and 99%
  quantiles, separate 95% and 99% expected shortfall, and every stored drawdown
  statistic;
- daily HAC 95% and 99% mean confidence intervals, and weekly HAC plus
  moving-block-bootstrap 95% and 99% mean confidence intervals;
- market, accounting, genuine risk-neutral, buyer-indifference, and
  seller-indifference prices with unavailable quantities left unavailable;
- total, spot, and option transaction costs; turnover; gross and net exposure;
  action sparsity; position-cap hits; fit-cap hits; and nonconverged fits;
- all mean--variance, mean--volatility, entropy-OCE, and power-exponential-OCE
  values at every precommitted ex-post risk parameter; and
- every loss component above in the overall, year, 2022-fixed
  ATM-volatility-regime, and year-by-regime slices.

Objective values are first evaluated in unscaled inception-spot-normalized loss
units. Only the final scalar is converted to basis points for display. The
reviewer explicitly checks pricing-plus-pure-hedging equals economic loss using
the common market starting price. Market and genuine-Q prices must be finite and
nonnegative; indifference values may be signed. Buyer and seller signs and
payoff orientation must match their declared economic semantics.

The reviewer independently reconstructs each path's accounting from actions and
released quotes: buys cross to mark plus half-spread, sells to mark minus
half-spread, spot costs one basis point of traded notional, and cash settlement
has zero transaction cost. The same reconstruction must reproduce total, spot,
and option costs; entry and total turnover; gross, net, and absolute-net
exposure; sparsity; and executed/fit cap counts. Agreement with stored aggregate
fields alone is insufficient.

## Paired comparisons and inference

The reviewer reconstructs each paired comparison from common eligible origins
and target books:

- daily comparisons use the equal-weighted 20-target origin book, a five-lag
  Bartlett HAC standard error, and a 2,000-sample moving-block bootstrap with
  block length 50 and seed 20,261,014; and
- weekly comparisons use the equal-weighted 13-target episode book,
  Hansen--Hodrick inference with four autocovariance lags for five-session
  overlap, and a 2,000-sample moving-block bootstrap with block length 50 and
  seed 20,261,012.

The reference is schedule-matched SANOS--LV and the sign is candidate loss minus
reference loss, so negative values favour the candidate. Train, validation, or
2022 selection observations must not enter 2023--2025 evaluation inference.

## Rendered artifact reconciliation

The reviewer confirms that every completed policy, objective, parameter, target,
price field, cost field, numerical diagnostic, paired comparison, and retained
stratum in the authenticated JSON is present in the generated tables or appendix
as specified by the report design. The PDF must compile from the bundled TeX and
bibliography without replacing inputs, missing references, or material layout
failures. Display names and model order must match the authenticated tables.

## Public-data boundary

The final report is rebuilt in an otherwise empty data root containing only the
released public-candidate dataset and the explicitly committed production and
report artifacts. The review rejects any runtime dependency on original WRDS
prices, midpoints, identifiers, vendor fields, private caches, or the original
market loader.

## Required review record

The signed review artifact records:

- reviewer verdict `PASS` or `BLOCKED`;
- every immutable path and SHA-256 above;
- exact commands, interpreter, reporting commit, and test counts;
- the independently reconstructed daily and weekly case/result/failure counts;
- a reconciliation table for headline losses, objectives, prices, costs, caps,
  convergence, and paired inference; and
- every discrepancy, however small, with its resolution or blocking status.

`PASS` is permitted only when all checks succeed. The final PDF is copied to the
repository only after this independent review and the complete hedging test
suite pass on the final reporting source. The reviewer then verifies byte-level
SHA-256 equality between the reviewed PDF/bundle and the repository copies,
records the publication Git commit and remote push identity, and confirms that
the pushed tree contains those exact bytes. Publication is incomplete until
this post-copy check passes.
