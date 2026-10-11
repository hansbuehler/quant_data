# DLV drift that makes fixed-strike calls martingales

This note records the functional drift used by the repeated-daily DYSANOS
construction. It applies to the SANOS density recursion on a fixed normalized
strike support. The production implementation also carries an explicit
residual-expiry clock.

## SANOS recursion

For expiry slab \(m\), write \(q_m\) for the complete discrete density,
\(\ell_m\) for the interior DLV-squared row, and
\(\phi_m=q_{m,I}\ell_m\) for its variance flux. The decoder equation is

\[
q_m+G_m(\phi_m)=q_{m-1},
\]

where \(G_m\) is the strike-grid operator including that slab's expiry length.
The equation is linear in \((q,\phi)\), although it is nonlinear in
\((q,\ell)\).

## Continuous functional drift

Let calendar factor \(f\) give the physical DLV diffusion
\(\beta_m^f\), and let \(\Psi_m^f\) denote the induced density diffusion.
Differentiating the SANOS recursion gives the forward solve

\[
\Psi_m^f+
G_m\!\left(
  \ell_m\Psi_{m,I}^f+q_{m,I}\beta_m^f
\right)
=\Psi_{m-1}^f,
\qquad \Psi_0^f=0.
\]

The drift that removes the quadratic-covariation drift of the variance flux is

\[
\boxed{
\alpha_{m,i}
=-
\sum_f
\frac{\Psi_{m,i}^f}{q_{m,i}}\,
\beta_{m,i}^f
}
\]

at every identified coordinate \(q_{m,i}>0\). Thus, if

\[
d\ell_{m,i}=\alpha_{m,i}\,dt+
\sum_f\beta_{m,i}^f\,dW_t^f,
\]

then \(d(q_{m,i}\ell_{m,i})\) has zero drift and the entire density tower has
zero drift. More explicitly, \(q_0\) is driftless; the displayed choice of
\(\alpha\) makes each \(\phi_m=q_{m,I}\ell_m\) driftless; and the linear SANOS
recursion then makes each \(q_m\) driftless inductively in \(m\).

For the native coordinate

\[
\ell=\ell_{\min}+e^z,
\qquad
\beta^f=(\ell-\ell_{\min})C^f,
\]

the required raw-coordinate drift is

\[
\boxed{
\mu_z
=-
\sum_f\frac{\Psi^f}{q}C^f
-\frac12\sum_f(C^f)^2
}.
\]

The first term is the martingale correction and the second is the Itô
correction for the exponential coordinate.

## Exact finite daily transition

Production avoids a time-discretization error in this drift. For a symmetric
zero-mean cubature branch \(\xi_r\), define

\[
q_m^r=q_m+\sqrt h\sum_f\Psi_m^f\xi_{r,f},
\]

\[
\phi_m^r=\phi_m+
\sqrt h\sum_f
\left(
  \ell_m\Psi_{m,I}^f+q_{m,I}\beta_m^f
\right)\xi_{r,f},
\qquad
\ell_m^r=\frac{\phi_m^r}{q_{m,I}^r}.
\]

Admissibility is checked before a branch is used. Because the cubature mean
of \(\xi_r\) is zero,

\[
\sum_r\pi_r q_m^r=q_m,
\qquad
\sum_r\pi_r\phi_m^r=\phi_m,
\]

and every branch satisfies the SANOS recursion in exact arithmetic. Production
enforces both identities to a strict decoder tolerance and records their
floating-point residuals. This is stronger than merely applying the continuous
drift with Euler time stepping.

At exact \(q=\phi=0\), \(\ell\) is unidentified. A deterministic completion is
law-neutral only for a fully boundary-absorbed row; the implementation checks
that condition explicitly. Every positive mass remains identified.

## One-day front inside a wider expiry gap

If the next fitted expiry is \(g>1\) days away, let \(p\) be the current
initial density and let \((q_g,\phi_g)\) be the old front row. Insert

\[
q_1=\left(1-\frac1g\right)p+\frac1gq_g,
\qquad
\ell_1=\frac{\phi_g}{q_{1,I}}
\quad\text{where }q_{1,I}>0.
\]

At an exact \(q_{1,I}=\phi_g=0\), the DLV is unidentified. The implementation
retains the old admissible DLV as a deterministic completion, and permits this
only on a fully boundary-absorbed row. Linearity of the grid operator in slab
length gives

\[
q_1+G_1(\phi_g)=p,
\qquad
q_g+G_{g-1}(\phi_g)=q_1.
\]

Here \(G_h\) denotes the same-support grid operator with an \(h\)-day slab;
the subscript indexes duration rather than the expiry-row index used above.

The temporary \(\ell_1\) is therefore price-authoritative. It may lie outside
the fitted coordinate bounds, but it must be finite and strictly positive. It
has zero calendar loading and is discarded during the same daily roll; all
survivor DLV rows remain inside the fitted bounds. Clipping this temporary row
would break the two identities and reintroduce option drift.

## Why all fixed-strike calls are martingales

On a fixed support, every call price is linear in the density. With Black
backbone total volatility \(w_m\),

\[
C_m(K)=\sum_j q_{m,j}\,B(Sx_j,K,w_m),
\]

where \(B\) is the Black call value; this reduces to \((Sx_j-K)^+\) when
\(w_m=0\). The branch mean of \(q_m^r\) is \(q_m\), so the conditional branch
mean of every such component value is its current value. Revealing the one-day
node and then propagating from its one-hot density is a disintegration of the
same SANOS transition. The independent Black multiplier

\[
\exp(-w^2/2+wZ)
\]

has conditional mean one. Moreover, residual Black variance is obtained by
exact variance subtraction, \(w_m^2=w_1^2+w_{m,\mathrm{res}}^2\), so the product
of the front and residual multipliers has exactly the original lognormal law.
Consequently spot and every fixed-cash call in the retained tower are
martingales, up to the explicitly measured floating-point residuals.

The Monte Carlo zero-drift tests are useful specification checks, but the
branchwise density/variance-flux identities are the algebraic martingale
certificate. Large hypothesis-test p-values alone would not prove the result.
