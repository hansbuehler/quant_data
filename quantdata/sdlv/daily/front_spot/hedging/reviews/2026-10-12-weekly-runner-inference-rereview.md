# Weekly runner and inference re-review

## Scope

- `quantdata/sdlv/daily/front_spot/hedging/benchmark/weekly.py`
- `quantdata/sdlv/daily/front_spot/hedging/benchmark/weekly_runner.py`
- `quantdata/hedging/inference.py`
- focused weekly, inference, and protocol tests

The review covered fit timing, frozen versus restarted identities, predictable
information, rolling-window enforcement, model-price units, and inference for
overlapping five-session episodes.

## Disposition

**PASS -- no blockers.**

- Hansen--Hodrick equal overlap weights are the production HAC default. An
  independent 200,000-observation moving-sum check estimated long-run variance
  `24.816`, versus the theoretical value `25`. The rejected Bartlett lag-four
  estimate was `16.873`, versus its theoretical value `17`.
- The circular moving-block bootstrap default now uses 50 episodes rather than
  one five-session overlap block.
- The as-of facade rejects future states and transitions and dates before the
  precommitted rolling fit window. Frozen fits retain one identity; restarted
  fits retain one identity per decision date.
- Fit prices explicitly distinguish units per fit-date spot from discounted
  episode-origin PV, and the conversion passed review.
- The focused suite passed 26 tests.

## Nonblocking follow-ups

1. Report the raw Hansen--Hodrick long-run variance before its non-negative
   numerical floor.
2. Enforce or record chronological episode ordering in production artifacts.
3. Include HAC-lag and bootstrap-block sensitivity in final results.
4. Document how restarted path-dependent model prices condition on the realized
   path prefix.
