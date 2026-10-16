# Historical-SANOS weekly ridge v6 output review

Date: 2026-10-16  
Calibration root: `/home/hans/quantdata_data/training/hedging/historical-sanos-weekly-ridge-v6-full-9367dc1`  
Calibration source: `9367dc1708159abe65d78750dcd66c8b7f717be8`  
Verdict: **NO-GO for production consumption; calibration contents PASS**

## Decision

All 13 pre-outcome selections are complete, internally authentic, numerically
consistent, and reproduce the declared one-standard-error rule under the exact
clean source commit. The current v1.28 production source cannot consume them,
however: every artifact declares Historical-SANOS code version `1.27` and source
commit `9367dc1...`, while the loader requires both the current
`HistoricalSanosConfig.version` and current repository commit. Production must
not rewrite the artifacts or bypass those checks. It may proceed only after an
exact final-commit rerun or an independently reviewed, hash-bound transition
that explicitly permits these exact v1.27 artifacts.

## Completion And Provenance

- Exactly 13 `selection.json` artifacts exist, with no missing or extra scope:
  mean-variance at 0.1/1/10; mean-volatility at 0.1/1/2/3; entropy at
  0.1/1/10; and power-exponential at 0.1/1/10.
- The transient systemd unit logged `all_scopes_completed total=13` at
  `2026-10-16T15:31:53+01:00`. Independent polling then observed
  `selections=13`, `ActiveState=inactive`, `SubState=dead`, and
  `Result=success`. Every scope log contains `scope_completed_at`, its final
  JSON-reported file and selection digests match the artifact, and no log
  contains a traceback or `scope_failed` marker.
- The immutable-source worktree
  `/tmp/quant_data/hedging/worktrees/production-9367dc1` was clean and resolved
  to `9367dc1708159abe65d78750dcd66c8b7f717be8` before and after validation.
- Dataset manifest SHA-256:
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`.
  The pinned-source public loader authenticated 891 dates.
- Weekly target-notional artifact SHA-256:
  `9494bbe23699126b5137b5ab3b6d53070002e2ba8ee60d92b0327ff747fcf9e1`.
  A byte-identical scratch copy passed the pinned authenticated loader; all 13
  target labels and quantities equal the loaded calibration exactly.
- All selections use schema `hedging-historical-sanos-weekly-ridge-v4`, origin
  `2023-01-17`, 252 training origins (`2022-01-03` through `2023-01-13`),
  232 usable scenario origins, a chronological 186/46 fit/validation split,
  and five validation blocks of sizes 10/9/9/9/9. The schedule family is
  exactly `frozen-weekly` plus `restarted-weekly`; position cap is 1,000 and
  maximum latent coordinate is 100,000.

## Selection Results

The candidate grid is exactly
`[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, null]`. For every candidate I
recomputed the three-stratum block regret, mean, sample-standard-error over the
five chronological blocks, feasibility predicate, and one-SE choice. The exact
null baseline is eligible only after at least one finite candidate is feasible.

| Scope | Selected policy | Finite feasible | One-SE evidence |
|---|---:|---:|---|
| entropy 0.1 / 1 / 10 | null | 8/8 each | null is the unique eligible best at threshold 0 |
| mean-variance 0.1 / 1 / 10 | null | 8/8 each | null is the unique eligible best at threshold 0 |
| mean-volatility 0.1 | null | 5/8 | null is the unique eligible best at threshold 0 |
| mean-volatility 1 | 1000 | 8/8 | best 10; threshold -0.0007550871288; eligible 10/100/1000 |
| mean-volatility 2 | 100 | 8/8 | best 10; threshold -0.005674184248; eligible 1/10/100 |
| mean-volatility 3 | 100 | 8/8 | best 10; threshold -0.009625221866; eligible 1/10/100 |
| power-exponential 0.1 | null | 4/8 | null is the unique eligible best at threshold 0 |
| power-exponential 1 | null | 8/8 | null is the unique eligible best at threshold 0 |
| power-exponential 10 | null | 5/8 | null is the unique eligible best at threshold 0 |

Every selected candidate has 80/80 declared components converged, zero
selection/internal/executable cap hits, zero fatal or unregularized recession
directions, maximum objective error at most `2.77556e-17`, maximum constraint
violation zero, and maximum latent coordinate at most `2.05034`. Null selections
are the exact zero-action baseline. The three non-null selections therefore
pass every numerical and economic gate without tolerance relaxation.

The only infeasible finite candidates are transparent and correctly excluded:

- Mean-volatility 0.1: ridge `1e-4` has 81 selection cap hits, 108 internal
  cap hits, 3,276 executable cap hits, and maximum latent coordinate `1e6`;
  ridge `1e-3` has 286 executable cap hits and maximum latent coordinate
  `144433`; ridge `1e-2` has 39 executable cap hits.
- Power-exponential 0.1: ridge `1e-4` has 72/80 converged components and
  maximum objective error `9.19548e-6`; ridges `1e-3` and `1e-2` have maximum
  objective errors `4.40165e-7` and `1.78046e-7`; ridge `100` has 78/80
  converged components and maximum objective error `1.77062e-7`.
- Power-exponential 10: ridges `1e-4`, `1e-2`, and `1` each have 79/80
  converged components, with maximum objective errors `3.06645e-7`,
  `1.09855e-7`, and `1.66607e-7`.

Positive-ridge mean-volatility candidates record 8 unregularized recession
directions in the 0.1 grid and 5 in the 1 grid. They are nonfatal diagnostics
under the declared coercive objective. No selected candidate contains one, and
no candidate reports a fatal ridge-regularized recession direction.

## Artifact Digests

Each file was hashed twice around parsing. Its embedded `selection_sha256` was
independently reconstructed from canonical JSON with the exact pinned loader.

| Scope | File SHA-256 | Selection SHA-256 |
|---|---|---|
| entropy-0p1 | `217b654fc176c72188fd6a4a19f051bee5b1c4a8f0cbd47b6ae24de0c48c7a3b` | `93072471c411a737e9b1d330b4a7c2c755dc02d364aaaa60c2aace3d05cfc1b0` |
| entropy-1 | `b44dbbdaa1443721d89ce352857f372d62e6700f05a88abd395bf71f1a4a1c43` | `396c0fb0f8f012a6ebf00b613f13ba3c4e2dc85480392766a752ce09e629e41a` |
| entropy-10 | `2927e6d136e81c86be4cba92f88870e65ca1005ccd68622bc79050026b250e60` | `d9b12ab4b402b76351664cd1fab26c1de23f622a2d210f2355f52823e5357580` |
| mean_variance-0p1 | `918c2fc7bd95aa4f47be08808c2351686c44d305b6566145ad60fb2740ade906` | `2c00ebb3842fb05165de71d1eb634416519bec533a3f4be57168f3476e157805` |
| mean_variance-1 | `3f932ca02cb6175637618d0078ec0bddcd95f783b5eb065610039e2147b92f1e` | `76464b55f857376c06bd15c2d328ac19e7a6c9ae4cfae3f44ee51aca6bb90552` |
| mean_variance-10 | `359f27413c9ecc33c30fcacf7fdb8c2bf491b4d3c76e91b5fb7e95accc2c820b` | `204ffe85a03953bc64136bb04f4aab8844e86426fb53248481de9bb6209a423f` |
| mean_volatility-0p1 | `0ee188ee822222614b4823a4c9f8802fb99cdab8175e606c380b10e34de36797` | `2d1c98c8a5fea7cf9516296ca3189c1b962f76b05c4780702f726fa85f240881` |
| mean_volatility-1 | `3d9d426c155e00c2b5193826e5282ceec3a8427a0317f1d28135bf06d8b36f50` | `8d6ebc2465c4a2ca50f56b4c473c7f0d20248d805ec7854e73f7a837ef53f124` |
| mean_volatility-2 | `088b7195a25928565b68b87b96e1b4eb113f3668a86e62d254a11c0dbc599a0e` | `284cc0dadda578a0acc75308c40f4f3a3c571fb0c7896fd06b80f25792512f92` |
| mean_volatility-3 | `87e9a34f4cecf27ec3492279a4ba23d5d875fe3cb3631d1210302d2012566ad7` | `fb4d8bee8fc5c927a73c3bbe67a6b8c98376b1c1d06d47c8c9d36464c307c16e` |
| power_exponential-0p1 | `5a7be932e3ce237f5684f73f36beefec3d8c0e71c4540e2f521d302c7b15a9da` | `5fa6f418df3f346d800fe62797510d1fe23271ef686a55c10f49297408644c66` |
| power_exponential-1 | `782d14fa68aef04ba36a4652e7a36f375990dd20d694ccf5d2861ba753b6d4f1` | `76cd0db162d59906c4f3940a5afc389f6b9575dd5d4f2454718fdcf33e808f18` |
| power_exponential-10 | `da07230778c19497e1b993c029bbb0e2194615261f2960f2a1a4c795561fd3b4` | `4b912e823347b4b6b66afe1d0af34de17ece7e3e53073b4a0967b145d5ce3b54` |

The calibration files are mode `0664`, so downstream composition must pin the
exact file digests above and double-hash while loading. Their current writability
is not permission to mutate, relabel, or rewrite provenance.

## Production Blocker

`load_weekly_ridge_selection` at
`/home/hans/python/quant_data/quantdata/sdlv/daily/front_spot/hedging/models/historical_sanos/weekly_ridge.py`
requires the artifact code version to equal the running
`HistoricalSanosConfig.version` and its source commit to equal the production
source commit. The current v1.28 changes make the reviewed v1.27 artifacts
incompatible, and the next committed source will also differ from `9367dc1...`.

The least costly valid alternative to an exact rerun is a separately reviewed
transition manifest that proves the weekly calibration algorithm, weekly
hedger, data, objective, grid, and gates are unchanged; pins this review and all
13 file digests; and makes the loader explicitly authenticate the legacy
calibration source rather than the current execution source. Until either that
transition or a final-commit rerun passes, the bundle and production matrix are
**NO-GO**.

## Daily Scope Boundary

The old v1.25 daily mean-volatility 3 partial ending at 2025-04-23 is not part
of this calibration and is not complete. The reviewed v1.27 certificate-first
logic directly covers the class where `OPTIMAL_INACCURATE` is harmless after
exact nonzero-residual KKT/objective/feasibility certification; v1.28 also
preserves a usable primary candidate if SCS destroys its state. Neither review
authenticates that historical cell without its planned current-version rerun.
