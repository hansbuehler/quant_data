"""Full-sample ordinary/robust OCE and pair-clustered centering-aware MC errors."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from quantdata.sdlv.dhgromm1.floating import LABELS, SIGNS
from quantdata.sdlv.dhgromm1.regularized_evaluation import oce_statistics
from quantdata.sdlv.dhgromm1.robust import profile_tail
from quantdata.sdlv.dhgromm1.robust_training import collect_gains


def evaluate_robust(*, model, sample, arrays: tuple[np.ndarray, ...], batch_size: int, monitor=None) -> tuple[dict,dict]:
    """Reprofile only scalar cash/cutoffs, never trading weights, on training data.

    First collect gains and fit global cutoff/cash. A second streaming pass
    aggregates derivatives times actions for estimated Q-centering uncertainty;
    no density states or N x 20 x 21 action cache are retained. The tail is
    individual-path based, while MC errors cluster both antithetic partners.
    """
    functions = model.policy_functions(compile_model=True)
    gains = collect_gains(model,sample,functions=functions,batch_size=batch_size,monitor=monitor).cpu().numpy().astype(np.float64)
    _,returns,payoff = arrays
    profiles,values,exposures,derivatives = {},{},{},{}
    for kind,fraction in (("ordinary",1.),("bottom90",model.config.tail_fraction)):
        profiles[kind] = [profile_tail(gains[i]+sign*(payoff-float(model.payoff_mean)),fraction=fraction)
                          for i,sign in enumerate(SIGNS)]
        values[kind] = np.column_stack([p["variates"] for p in profiles[kind]])
        exposures[kind],derivatives[kind] = np.zeros((3,20,21)),np.zeros(3)
    sizes,maximum = np.zeros((3,2)),np.zeros(3)
    with torch.no_grad():
        for begin in range(0,sample.size,batch_size):
            price,ret,_,changes = sample.batch(slice(begin,begin+batch_size))
            end = begin+len(price)
            for i,function in enumerate(functions):
                out = function(price,changes,returns=ret)
                trades = out.trades.cpu().numpy().astype(np.float64)
                for kind in profiles:
                    derivative = profiles[kind][i]["derivative_weight"][begin:end]
                    exposures[kind][i] += np.einsum("n,nti->ti",derivative,trades)
                    derivatives[kind][i] += derivative.sum()
                sizes[i] += np.array((np.abs(trades).sum(),np.square(trades).sum()))
                maximum[i] = max(maximum[i],float(np.abs(trades).max()))
            if monitor is not None:
                monitor.record.update(completed=sample.size+end,total=2*sample.size)
    result,retained = {},{}
    for kind in profiles:
        influence = values[kind]-values[kind].mean(0)
        influence -= (returns-returns.mean(0)).reshape(sample.size,-1) @ (exposures[kind]/sample.size).reshape(3,-1).T
        influence -= (payoff-payoff.mean())[:,None]*(np.asarray(SIGNS)*derivatives[kind]/sample.size)[None]
        result[kind] = dict(statistics={k:dict(**oce_statistics(values[kind][:,i],influence[:,i]),
                profiled_cash=profiles[kind][i]["cash"],profiled_utility_cutoff=profiles[kind][i]["threshold"],
                retained_paths=profiles[kind][i]["retained_paths"]) for i,k in enumerate(LABELS)},
            long_minus_empty=oce_statistics(values[kind][:,1]-values[kind][:,0],influence[:,1]-influence[:,0]),
            short_minus_empty=oce_statistics(values[kind][:,2]-values[kind][:,0],influence[:,2]-influence[:,0]))
        retained[f"{kind}_values"],retained[f"{kind}_influence"] = values[kind],influence
    report = dict(**result,tail_fraction=model.config.tail_fraction,transaction_costs=0,l1=0,l2=0,training_only=True,
        action_sizes={k:dict(mean_sum_abs_trades=float(sizes[i,0]/sample.size),
            rms_trade=float(np.sqrt(sizes[i,1]/(sample.size*420))),max_abs_trade=float(maximum[i])) for i,k in enumerate(LABELS)},
        interpretation="OCE profiles scalar cash for each fixed policy. Bottom90 separately discards each policy's best 10% "
            "INDIVIDUAL path utilities over all 100000 training paths; it is not a Q-arbitrage proof.",
        uncertainty="±3 antithetic-pair MC SE; tail cutoff accounted for by winsorized influence; includes Q gain/payoff "
            "centering, holds quote scaling/policies fixed, excludes fitting/selection uncertainty; regular-quantile approximation")
    return report,retained


def plot_robust_history(output: Path) -> Path:
    """True global-tail training utility, Empty adjusted, rather than negative loss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output/"triple-losses.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epoch = np.array([int(r["epoch"]) for r in rows])
    robust = np.array([[float(r[f"{k}_robust_oce"]) for k in LABELS] for r in rows])
    curves = (robust[:,1]-robust[:,0],robust[:,2]-robust[:,0],robust[:,0])
    fig,axes = plt.subplots(1,3,figsize=(14,4.2))
    for axis,curve,title,color in zip(axes,curves,("Long − Empty","Short − Empty","Empty"),
                                     ("#864a9e","#d88424","#c33b43"),strict=True):
        axis.plot(epoch,curve,color=color,linewidth=1)
        axis.axhline(0,color="#777777",linewidth=.6)
        axis.set(title=title,xlabel="Robust phase epoch",ylabel="Full-population bottom-90% OCE (trained cash)")
        axis.grid(alpha=.2)
    fig.suptitle("GROMM Q one-step/day + a₁₀: global bottom-90% objective, no costs/L1/L2",fontsize=12)
    fig.text(.5,.015,"Training means, not historical MC bands. Profiled OCE and ±3 pair-MC SE are in evaluation reports.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    target = output/"robust-history.png"
    fig.savefig(target,dpi=150)
    plt.close(fig)
    return target
