# Minimal deep hedging on the shared GROMM sample

[`example_deep_hedging.py`](example_deep_hedging.py) is a standalone teaching
example: a small feed-forward hedge, ordinary power-exponential OCE, and zero
costs or penalties. It trains **three separate policies**, Empty, Long and
Short, by minimizing the sum of their negative objectives. It uses the existing
Q paths; it does not calibrate, simulate, reconstruct states or reprice options.

This is deliberately simpler than the research LSTM experiment. In particular,
there is **no recurrence, past-position input, explicit day-0/day-10 action
vector, robust tail selection, or contract-mark input**. The optional cached
marks stay in the published dataset but this example does not decompress them.
The policy class is therefore smaller than that of the full experiment; do
not interpret its performance as a reproduction of our trained hedge.

## Install and run

First fetch the dataset as described in [README.md](README.md). The example
needs only Python, NumPy and JAX; it does not import `quantdata`, Torch, a market
database, or any machine-local configuration. Install dependencies into your
own environment, for example `python -m pip install numpy jax`.

From this directory, a small CPU run is:

```bash
python example_deep_hedging.py \
  --paths 4000 --epochs 50 --batch-size 256 \
  --output-root /tmp/quant_data/dhgromm1-examples
```

For the complete shared sample, use `--paths 100000`. The default is **only the
first 4,000 paths**, for an accessible example, not a new benchmark result.
Both modes preserve complete adjacent antithetic pairs. The model always uses
CPU, so it does not interfere with GPU training. All days and instruments are
vectorized; JAX compiles the objective, gradients and Adam update.

Each invocation creates a new `q1-simple-*` run beneath `--output-root`:

- `history.jsonl`: flushed epoch progress and the online minibatch loss.
- `RESULTS.json`: parameters, data identity, final OCEs, prices and MC errors.
- `checkpoint.npz`: weights, frozen initial weights, Adam moments/update count,
  normalization and metadata including the NumPy RNG state.

Outputs must be outside Git. If `--output-root` is omitted, `QUANTDATA_ROOT`
must be set; the script uses its `experiments/quant_data/sdlv/dhgromm1-examples`
subdirectory. Existing runs are never overwritten. This small CLI saves the
optimizer state but does not provide automatic restart or best-epoch selection.

## Payoff, trades and information

The payoff, with unchanged notional, is

$$
H=\left|\frac{S_{20}}{S_{10}}-1\right|.
$$

There are 20 daily decisions, $t=0,\ldots,19$, on one-simulation-step-per-day
Q paths. All 21 instruments in [the dataset instrument table](README.md#instrument-slots-and-strikes)
are available: spot, the daily-reset ATM call expiring at fixed day 20, the
three five-call ITM strips, and their zero-strike calls. The latter are funded
zero-strike-call marks, **not zero-premium forwards**.

Action $a_{t,i}$ opens a **floating-contract quantity**, corresponding to
$a_{t,i}/S_t$ actual units. Its supplied gain is

$$
R_{t,i}=\frac{M_{\min(T_{t,i},20)}(t,i)-M_t(t,i)}{S_t}.
$$

An option is held to its own expiry, or marked at day 20 if it expires later.
Spot trades are held to day 20 and can offset earlier spot trades. A slot
offered tomorrow is a new contract, not necessarily yesterday's contract.
There is **no division by option premium** and no cumulative-holdings
multiplication of these expiry gains.

Decision features are $t/20$, current spot, and the other 20 current prices
divided by current spot. Price features are standardized with constants fitted
on the selected training sample. The policy never receives day-20 prices,
future realized gains, the terminal payoff, or simulator states. Using the
whole price tensor in one batched feed-forward calculation does not connect
different dates: the same date-local map is evaluated separately at every t.

## Network and initialization

For each of the three independently parameterized policies,

$$
a_\theta(z)=W_a z+b_a+W_2\tanh(W_1z+b_1)+b_2
             -f_{\widetilde\theta}(z).
$$

The last term is an entire frozen copy of the initial affine-plus-deep map.
Thus all initial actions are exactly zero, while hidden-layer gradients remain
available. Hidden width is 16 by default (`--width`); actions are not capped.
There is no special day-10 vector: the network must learn time dependence
through its clock input. This is an example architecture, not a claim that
width 16 suffices to find the best time-local hedge.

## Centering and objective

The script fits $\bar R_{t,i}$ and $m=\bar H$ **once over the selected
training sample**, never per minibatch, and uses

$$
\widetilde R_{t,i}=R_{t,i}-\bar R_{t,i},\qquad
\widetilde H=H-m,\qquad
G_\theta=\sum_{t,i}a_\theta(z_t)_i\widetilde R_{t,i}.
$$

It refits constants when using a prefix, rather than silently using the
100,000-path frozen constants for a smaller experiment. Raw published arrays
are unchanged. For each sign $s\in\{0,+1,-1\}$, it solves

$$
V_s=\sup_{\theta_s,y_s}
\left\{\frac1N\sum_{n=1}^N
u_{100}\big(y_s+G_{\theta_s}^{(n)}+s\widetilde H^{(n)}\big)-y_s\right\},
$$

where

$$
u_\gamma(x)=
\begin{cases}
(1-e^{-\gamma x})/\gamma,&x\ge0,\\
x-\gamma x^2/2,&x<0,
\end{cases}\qquad \gamma=100.
$$

Each policy has its own cash parameter $y_s$. Adam jointly minimizes the
**sum** of the three negative objectives; no weights are shared between the
policies. Learning rate defaults to 0.0001 (`--learning-rate`); there are no
transaction costs, L1/L2 penalties, weight decay or action caps. At the end,
the script reoptimizes each scalar cash parameter for the fixed learned hedge
using $E[u'(y_s+X_s)]=1$. The final summary uses those profiled OCE values.

The conservative default avoids the large early overshoots seen with 0.001
in this example's 4,000-path smoke run; it is not a tuned optimal rate.

## Reading the output

The JSON reports the raw **Empty**, Long and Short OCEs, then

$$
\Delta_L=V_{+1}-V_0,\qquad \Delta_S=V_{-1}-V_0,
$$

and cash prices

$$
\mathrm{Bid}=m+\Delta_L,\qquad
\mathrm{Ask}=m-\Delta_S,\qquad \widehat E_Q[H]=m.
$$

Higher OCE is better. Empty denotes a **learned hedge with no claim**, not a
constraint to make no trades. The feasible no-trade Empty policy has OCE zero;
a positive learned Empty OCE can reflect in-sample exploitation or simulator
drift. It must be subtracted when comparing claim utilities.
A negative learned Empty OCE means optimization has not even beaten the
feasible no-trade baseline; it is not evidence of a negative optimal value.

Every final estimate includes **±3 MC standard errors**, using independent
antithetic-pair averages, with covariance retained for policy differences.
The delta-method calculation includes estimated return/payoff-centering means;
policy fitting, feature-normalization estimation and selection uncertainty
are **not** included. These are training diagnostics, not validation results.

The per-epoch `minibatch_loss` averages objectives at successive optimizer
states. It is not a fixed-checkpoint full-sample loss, the final profiled OCE,
or a guarantee of monotonic improvement. The example makes no convergence claim.

Ideally the prices bracket $E_Q[H]$; this is not guaranteed for a finite,
incompletely trained network on this discretized sample. Unconditional sample
centering is not a conditional martingale correction. The original accepted
one-step Q drift warnings still apply. See [dataset limitations](README.md#provenance-and-limitations).
