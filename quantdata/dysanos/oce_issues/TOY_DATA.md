# Dynamic OCE toy data

This directory contains a small conditional OCE problem with the state geometry
of the original DYSANOS experiment. Every simulated path begins at the same
initial state. After the first transition, the state is continuous and, in
practice, no later state occurs twice. A single state-dependent policy must
therefore learn across the visited state distribution; it cannot fit a separate
action at each state.

The simulator law plays the role of the preconditioned measure. Its conditional
mean and covariance are known, so a learned policy can be compared with the
population optimum even at states that occur only once.

## Stored samples

`samples/` contains 16 independent Monte Carlo samples. Each member file
stores:

- `features`: shape `(10_000, 3, 3)`, with two state coordinates followed by
  normalized time;
- `returns`: shape `(10_000, 3, 4)`;
- `conditional_mean`: shape `(10_000, 3, 4)`, containing the generator's exact
  conditional mean;
- the common initial state, member index, and random seed.

All tensors are CPU `float32` tensors. Load a member without enabling general
pickle deserialization:

```python
from pathlib import Path

import torch

sample = torch.load(
    Path("quantdata/dysanos/oce_issues/samples/toy_sample_00.pt"),
    map_location="cpu",
    weights_only=True,
)
features = sample["features"]
returns = sample["returns"]
conditional_mean = sample["conditional_mean"]
```

`samples/manifest.json` records each member's seed, geometry, byte size, and
SHA-256 digest. The members were copied without numerical modification from
`toy-samples.pt` in experiment
`toy-dynamic-path-oce-16x10k-epsilon1-v1`. The source archive SHA-256 is
`aef6c57cafbe1d158fbcb7752d03778f24fb34907f4b4a4e79be354b3f8386d9`.

## State process

All 10,000 paths in a member share

\[
x_{p,0}=(0.75,-0.50).
\]

For later steps, with independent standard-normal
\(u_{p,t}\in\mathbb R^2\),

\[
x_{p,t}=0.72x_{p,t-1}+0.65u_{p,t},\qquad t=1,2.
\]

The policy feature is

\[
X_{p,t}=(x_{p,t,1},x_{p,t,2},\tau_t),
\qquad (\tau_0,\tau_1,\tau_2)=(0,0.5,1).
\]

Within every stored member, the initial state occurs 10,000 times and each of
the 10,000 states at either later step is unique.

## Conditional returns

There are four instruments. With rows

\[
L=
\begin{pmatrix}
 0.19 & -0.04\\
-0.15 &  0.10\\
 0.11 &  0.14\\
-0.08 & -0.16
\end{pmatrix},
\]

the exact conditional mean is

\[
\mu_i(x)=0.075\tanh(L_i x),\qquad i=1,\ldots,4.
\]

Let \(f_{p,t}\sim N(0,I_3)\), independent of the state innovations. Returns
are

\[
R_{p,t}=\mu(x_{p,t})+Bf_{p,t},
\]

where

\[
B=
\begin{pmatrix}
0.10 & 0.055 &  0.000\\
0.10 & 0.000 &  0.055\\
0.10 & 0.033 & -0.044\\
0.10 & 0.044 &  0.033
\end{pmatrix}.
\]

Thus \(R\mid x\) is Gaussian with constant conditional covariance

\[
\Sigma=BB'=
\begin{pmatrix}
0.013025 & 0.010000 & 0.011815 & 0.012420\\
0.010000 & 0.013025 & 0.007580 & 0.011815\\
0.011815 & 0.007580 & 0.013025 & 0.010000\\
0.012420 & 0.011815 & 0.010000 & 0.013025
\end{pmatrix}.
\]

The four returns have three noise factors, so \(\Sigma\) has rank at most
three. The \(\varepsilon I\) regularizer makes the analytic action system
nonsingular.

## Antithetic construction

Each member begins with 5,000 independently generated paths. Negating their
state and return innovations supplies another 5,000 paths. Because the initial
state is nonzero, paired later states are symmetric around their conditional
transition centers; they are not generally negatives of one another.

The 16 members use independent seeds

\[
s_m=20{,}261{,}006+10{,}003m,\qquad m=0,\ldots,15.
\]

## Path-level OCE problem

For one state-dependent policy \(a_\theta(X_{p,t})\), the empirical loss is one
exponential objective over complete path P&Ls:

\[
\widehat L(\theta)=
\log\!\left[
\frac1N\sum_{p=1}^N
\exp\!\left(
-\sum_{t=0}^2
\left\{
a_\theta(X_{p,t})'R_{p,t}
-\frac{\varepsilon}{2}\|a_\theta(X_{p,t})\|_2^2
\right\}
\right)
\right].
\]

It is not an average of separate state-time OCE losses. Training and audit do
not use a fixed state grid. Audit paths are fresh draws from the model's state
visitation distribution.

There are no transaction costs or action caps in this diagnostic. Since the
return shock is independent of the next-state innovation, the population
action at every visited state is

\[
a^*(x)=(\varepsilon I+\Sigma)^{-1}\mu(x).
\]

For \(\varepsilon=1\), 16 independently fitted policies had mean individual
action RMSE \(0.00260\) on 50,000 fresh paths. Averaging their actions reduced
RMSE to \(0.000658\). At the common initial state, the ensemble action RMSE was
\(0.0000499\). The mean held-out path log loss was \(-0.000615\), versus
\(-0.000658\) for the analytic action on the same paths.

The ensemble's held-out cross-product estimate was
\(E[a(X)'R]=0.00043438\) per decision, with antithetic-pair standard error
\(0.00000391\). The generator's exact conditional value
\(E[a(X)'\mu(X)]\) was \(0.00042908\). This estimates the covariance-adjusted
drift captured by the learned policy without estimating a conditional
covariance matrix.
