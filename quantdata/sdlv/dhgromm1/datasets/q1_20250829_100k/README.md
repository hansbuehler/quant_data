# GROMM v1 (−4,3): one-step Q deep-hedging training dataset

This deliberately retained sharing artifact contains the exact simulated
training inputs used by the one-step GROMM deep-hedging experiments, including
the a0/a10 and global bottom-90% OCE experiments. It is not a newly simulated
sample. All paths start from the same calibration for **29 August 2025**, with
initial spot 1, one simulation step per business day, 20 daily trading
decisions and observations at days 0 through 20.

There are **100,000 paths, arranged as 50,000 independent antithetic pairs**.
Partners are adjacent: `(0,1), (2,3), …`. The 25 chunks each contain 4,000
paths in their original order. About 1.3 GB of losslessly compressed NumPy
data is stored through Git LFS; the manifest gives the exact byte count.

These are synthetic prices/marks/payoffs, not raw market quotes. No underlying
market-calibration dataset, simulator densities, latent states, neural weights,
training logs or evaluation sample is included. A recipient can train a hedge
directly on these arrays without replaying GROMM, accessing the original data
storage, or recalibrating the model. The dataset is independent of whether a
policy uses an explicit a10 action.

## Download and verify

Clone this repository with Git LFS enabled. If it was cloned without downloading
LFS objects, run from the repository root:

```bash
git lfs install --local
git lfs pull --include="quantdata/sdlv/dhgromm1/datasets/q1_20250829_100k/chunks/*.npz"
cd quantdata/sdlv/dhgromm1/datasets/q1_20250829_100k
python dataset.py --verify
```

The standalone `dataset.py` needs only Python and NumPy; importing `quantdata`,
Torch, JAX, host configuration or a GPU is unnecessary. Verification streams
one chunk at a time, checks SHA-256 and shapes/dtypes, confirms the payoff and
day-10 replication identities, and reports the payoff mean with **±3 MC standard
errors** computed from antithetic-pair averages. `VERIFICATION.json` records
the export-time verification. GitHub source ZIP downloads may contain only LFS
pointers, not the bulk arrays; the verification detects that condition.

## Arrays

| Array | Full shape | dtype | Meaning |
|---|---|---|---|
| `prices` | `(100000,21,21)` | float64 | Current cash prices at days 0…20; instrument 0 is spot. Other slots identify contracts newly offered on that observation date. |
| `gains` | `(100000,20,21)` | float64 | **Uncentered**, unit-floating-quantity gain for each contract opened on day 0…19, held to its expiry or marked at day 20. |
| `payoff` | `(100000,)` | float64 | `abs(S_20/S_10 - 1)`, with no change of notional. |
| `changes_packed` | `(100000,3990)` | float32 | Causal current-minus-opening cash marks of previously opened actual contracts, frozen after settlement. |

`gains` is **not an option-premium percentage return**. If `a[t,i]` is an
executed floating-contract quantity, the actual contract position opened is
`a[t,i]/S_t`, and its gain is

```text
gains[t,i] = (closing_cash_mark - opening_cash_mark) / S_t
```

There is no division by the option premium. The original stored sample had a
legacy premium-return representation; this export converts that recognized
schema exactly once using the production `canonical_gains` adapter. The
published `gains` arrays are already corrected: **do not convert them again**.

Spot trades are held to day 20; daily negative spot trades can offset earlier
ones. Each option trade is held to its own expiry. Contracts expiring after
day 20 are marked to market at day 20, not held beyond the payoff horizon.
Options have the original maturity-boundary convention: the longest new
expiry uses `min(10,25-t)` business days at opening date t.

## Instrument slots and strikes

The manifest records the complete order. Zero-based indices are:

| Slots | New contract at day t |
|---|---|
| 0 | Spot |
| 1 | ATM call with strike `S_t`, expiring at fixed day 20 |
| 2…6 | Five ITM calls, 1 business day to expiry |
| 7…11 | Five ITM calls, 5 business days to expiry |
| 12…16 | Five ITM calls, `min(10,25-t)` business days to expiry |
| 17 | Zero-strike call/forward expiring at fixed day 20 |
| 18,19,20 | Zero-strike calls/forwards for the three floating tenors above |

For each five-call block, the opening strikes are
`K_t = S_t * exp(z * 0.20 * sqrt(DTE/252))` in the order
`z = [-2,-1.5,-1,-0.5,-0.25]`. The 0.20 in this strike rule is fixed; it is
not the current ATM implied volatility. The fixed-day-20 ATM call resets its
strike at every new opening date. Its old cohorts retain their original
strikes: subsequent newly offered calls are different contracts.

Despite the `*_forward` slot names, these are funded zero-strike-call cash
marks, not zero-premium forward-contract prices. Their opening cash mark is
included in the gain formula above.

At day 10, the payoff identity is
`H = 2*(S_20/S_10-1)^+ - S_20/S_10 + 1`. Thus the available ATM call and
fixed-terminal forward span the post-fixing straddle payoff, with a stochastic
day-10 purchase price. This identity is checked in the exported gain arrays.

## Causal marks, state and no look-ahead

For a path, day-t packed marks occupy the interval
`[21*t*(t-1)/2 : 21*t*(t+1)/2]`. Within that interval they are ordered by
opening day `r=0…t-1`, then by instrument `i=0…20`. `expand_changes` supplies
the production `(paths,20,420)` layout, or a single day's `(paths,420)` view:

```python
from dataset import Dataset, expand_changes

data = Dataset()  # directory containing this README
normal = data.normalization()
for record, chunk in data.chunks():
    t = 10
    current_prices = chunk["prices"][:, t].astype("float32")
    old_contract_marks = expand_changes(chunk["changes_packed"], day=t)
    # old_contract_marks[:,21*r+i] = current cash mark - opening cash mark
    # For earlier trade a[r,i], actual units are a[r,i]/S_r.
    # Their product gives marked hedge P&L available at decision time t.
```

The padded entries for cohorts not yet opened are zero. Marks stop changing
after settlement. A current policy may use only observations through its
decision time, its own earlier trades/positions and recurrent memory. The
`gains` and `payoff` arrays are **future labels, not decision features**. Storing
them alongside features does not authorize a policy to read ahead. There is
no simulator latent state in this bundle.

## Frozen normalization and objectives

`normalization.npz` retains the original full-precision training statistics:
`return_mean (20,21)`, scalar `payoff_mean`, and `price_mean/price_scale (21,)`.
The corresponding `model_*` arrays contain their exact float32 policy-buffer
values. To match the training calculation, cast inputs to float32 before
subtracting the model buffers:

```python
R = chunk["gains"].astype("float32") - normal["model_return_mean"]
H = chunk["payoff"].astype("float32") - normal["model_payoff_mean"]
# Given a causal policy's executed floating quantities a of shape (B,20,21):
G = (a * R).sum(axis=(1, 2))
# X_s = G + s*H, for s=0 (Empty), +1 (long), -1 (short).
```

Current price features are spot itself followed by the other prices divided
by current spot, then standardized using `model_price_mean/scale`. Time is
`t/20`. Normalization is frozen from **all original training paths**. No P
means or evaluation data are used. If making a new train/validation split,
keep antithetic partners together and refit normalization on the new training
subset; otherwise this original normalization would include the held-out data.

The ordinary objective uses
`U(X) = sup_y (mean(u_100(y+X)) - y)`, where
`u_100(x)=(1-exp(-100*x))/100` for nonnegative x, and `x-50*x*x` otherwise.
The global bottom-90% experiment discards the best 10% of utility outcomes
over all 100,000 individual training paths, separately for each policy. Its
shared-threshold representation is
`sup_(y,eta) [eta-y-mean((eta-u_100(y+X))_+)/0.9]`, not an average of minibatch
tail statistics. Neither objective nor regularization changes the stored
dataset. The current robust experiment has zero costs/L1/L2.

## Provenance and limitations

`MANIFEST.json` identifies the simulation seed, calibration SHA-256, original
scenario and mark-cache identities, every source/output chunk checksum and
the export code hash. It uses logical run names, not machine-specific paths.
`share_dataset.py` in the parent project exports these inputs without rerunning
simulation or fitting new means. The original external files and the live
training run are unchanged.

This is training data only; the original separate 100,000-path evaluation
sample is not bundled. In-sample fitting and checkpoint-selection uncertainty
are not covered by ordinary MC error bars. The accepted one-step Q drift
warnings remain relevant: subtracting unconditional sample means does not
enforce conditional martingale drift. Neither this data export nor robust
hedging establishes that the discretized Q simulator is arbitrage-free.
