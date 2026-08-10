# Refit LP and frozen out-of-sample validation

Three separate LPs were fitted using only the 40,000-path dataset. For each cost convention the LP maximized mean net payoff subject to net payoff being non-negative on every training path, with `-1 <= beta[t,l] <= 1`. Positions were then frozen and evaluated under the same cost convention on all 400,000 independent paths.

The payoff used was

`beta @ dX[path] - abs(beta) @ (delta * mc_error_train + prop_cost * abs(X[path]))`,

where `mc_error_train = std(dX_train, ddof=1) / sqrt(40000)`. No 400k observations entered any fit.

## Results

| fitted convention | 40k LP min | 40k LP mean | 40k interpretation | frozen 400k min | frozen 400k mean | 400k negative paths | conclusion |
|---|---:|---:|---|---:|---:|---:|---|
| no cost | -1.00002e-10 | 0.07544425 | material empirical certificate | -0.03755529 | 0.07556992 | 874 (0.2185%) | positive expected payoff, not pathwise arbitrage |
| 1 training MC error | -1.96316e-10 | 9.21593e-10 | numerical zero; no material certificate | -4.23407e-9 | 9.24870e-10 | 2,056 (0.5140%) | no material arbitrage |
| 1 bp of `X` | -2.99546e-10 | 0.06029093 | material empirical certificate | -0.02874490 | 0.06038231 | 996 (0.2490%) | positive expected payoff, not pathwise arbitrage |

Boundary values of a few times `1e-10` are at the LP solver's feasibility floor. The delta=1 solution is especially revealing: its beta has L1 norm only `3.82e-5`, maximum absolute coordinate `1.59e-6`, and mean payoff below `1e-9`. Because the model is homogeneous, numerical feasibility tolerance can admit a tiny scaled portfolio even when no nonzero material separating portfolio exists. This result is numerical zero, not evidence of arbitrage.

The two material fitted portfolios do not generalize pathwise. Their means remain strongly positive on independent data, but both incur losses on hundreds of validation paths.

## Solver completion

| convention | cutting-plane iterations | active constraints | HiGHS status | worst residual beyond `-1e-10` | elapsed LP time |
|---|---:|---:|---|---:|---:|
| no cost | 5 | 2,255 | optimal | 1.88e-15 | 15.4 s |
| 1 training MC error | 7 | 3,034 | optimal | 9.63e-11 | 58.3 s |
| 1 bp of `X` | 5 | 2,426 | optimal | 2.00e-10 | 19.5 s |

All remaining deviations are among active boundary constraints and within the explicitly recorded HiGHS feasibility verification slack. Every path was checked after each solve; no materially violated training path remained.

## Conclusion

The 40k sample contains material empirical separating portfolios without costs and with 1 bp costs. Neither frozen beta remains pathwise non-negative on 400k independent paths. With one training MC-error cost, the optimized result collapses to numerical scale on both datasets. The evidence is therefore finite-sample separation plus a positive expected-payoff signal with risk, not structural arbitrage.
