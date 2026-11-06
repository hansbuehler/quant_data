"""Frozen-policy validation: never recenter held-out gains or refit hedge weights."""

from __future__ import annotations

import numpy as np
import torch

from quantdata.sdlv.dhgromm1.floating import LABELS, SIGNS
from quantdata.sdlv.dhgromm1.robust import exact_tail_size, profile_tail
from quantdata.sdlv.dhgromm1.training import _economic_utility


def fixed_cash_tail(net: np.ndarray, *, cash: float, fraction: float) -> dict:
    """Global lower-tail score and quantile-aware influence at a FROZEN cash.

    Tail membership is determined by wealth, whose ordering is preserved by
    increasing utility. The winsorized representation accounts for estimating
    the validation quantile; variance of the selected paths alone would not.
    """
    net = np.asarray(net,dtype=np.float64)
    if net.ndim!=1 or not np.isfinite(net).all() or not np.isfinite(cash):
        raise ValueError("expected finite path wealth and fixed cash")
    count = exact_tail_size(len(net),fraction)
    ordered = np.sort(net)
    value = _economic_utility(net,cash,100.)
    if fraction==1:
        weights,variates,threshold = np.ones_like(net),value,float("nan")
    else:
        cutoff = float(ordered[count-1]+(ordered[count]-ordered[count-1])/2)
        below,tied = net<cutoff,net==cutoff
        weights = below.astype(np.float64)
        if tied.any():
            weights[tied] = (count-int(below.sum()))/int(tied.sum())
        threshold = float(_economic_utility(np.array([cutoff]),cash,100.)[0])
        variates = threshold-np.maximum(threshold-value,0)/fraction
    z = net-cash
    derivative = np.exp(-100*np.maximum(z,0))+100*np.maximum(-z,0)
    expected = float(_economic_utility(ordered[:count],cash,100.).mean())
    np.testing.assert_allclose(variates.mean(),expected,atol=1e-12,rtol=1e-10)
    return dict(variates=variates,derivative_weight=weights*derivative/fraction,cash=cash,
                threshold=threshold if fraction<1 else None,retained_paths=count)


def independent_mc(values: np.ndarray, validation_influence: np.ndarray, training_influence: np.ndarray) -> dict:
    """Independent validation MC plus original-training centering uncertainty.

    Both populations contain adjacent antithetic pairs. No validation mean is
    subtracted from returns. The two MC variance terms ADD: they are not the
    same paths and must not be paired with one another.
    """
    values,vi,ti = (np.asarray(x,dtype=np.float64) for x in (values,validation_influence,training_influence))
    if (values.ndim!=1 or vi.shape!=values.shape or ti.ndim!=1
            or any(len(x)<4 or len(x)%2 or not np.isfinite(x).all() for x in (values,vi,ti))):
        raise ValueError("expected finite complete validation and training antithetic pairs")
    vp,tp = vi.reshape(-1,2).mean(1),ti.reshape(-1,2).mean(1)
    validation_variance,training_variance = vp.var(ddof=1)/len(vp),tp.var(ddof=1)/len(tp)
    return dict(oce=float(values.mean()),three_standard_errors=float(3*np.sqrt(validation_variance+training_variance)),
        three_validation_only_errors=float(3*np.sqrt(validation_variance)),paths=len(values),independent_pairs=len(vp),
        centering_training_paths=len(ti),centering_independent_pairs=len(tp))


def evaluate_heldout(*, model, sample, arrays: tuple[np.ndarray,...], training_arrays: tuple[np.ndarray,...],
                     batch_size: int, monitor=None) -> tuple[dict,dict]:
    """Score fixed training-selected policies on the untouched evaluation split.

    Report both completely fixed training cash and scalar-cash-profiled OCE.
    Profiling estimates the risk measure of a fixed hedge; it does not retrain
    actions, thresholds, feature scaling, or Q gain/payoff centering buffers.
    """
    _,returns,payoff = arrays
    _,q_returns,q_payoff = training_arrays
    q_means = [getattr(model,k).return_mean.detach().cpu().numpy().copy() for k in LABELS]
    payoff_mean = float(model.payoff_mean)
    gains = np.empty((sample.size,3))
    l1,l2,maximum = np.empty_like(gains),np.empty_like(gains),np.zeros(3)
    with torch.no_grad():
        for begin in range(0,sample.size,batch_size):
            price,ret,_,changes = sample.batch(slice(begin,begin+batch_size))
            end = begin+len(price)
            for i,label in enumerate(LABELS):
                out = getattr(model,label)(price,changes,returns=ret,eager=True)
                if bool(torch.count_nonzero(out.penalty)):
                    raise ValueError("unexpected transaction costs in frozen-policy validation")
                trade = out.trades.cpu().numpy().astype(np.float64)
                gains[begin:end,i] = out.gain.cpu().numpy()
                l1[begin:end,i],l2[begin:end,i] = np.abs(trade).sum((1,2)),np.square(trade).sum((1,2))
                maximum[i] = max(maximum[i],float(np.abs(trade).max()))
            if monitor is not None:
                monitor.record.update(completed=end,total=2*sample.size)
    profiles,exposures,derivatives = {},{},{}
    for treatment in ("fixed_training_cash","profiled_cash"):
        for metric,fraction in (("ordinary",1.),("bottom90",.9)):
            key = (treatment,metric)
            profiles[key] = []
            for i,(label,sign) in enumerate(zip(LABELS,SIGNS,strict=True)):
                net = gains[:,i]+sign*(payoff-payoff_mean)
                p = (fixed_cash_tail(net,cash=float(getattr(model,label).translation.detach()),fraction=fraction)
                     if treatment=="fixed_training_cash" else profile_tail(net,fraction=fraction))
                profiles[key].append(p)
            exposures[key],derivatives[key] = np.zeros((3,20,21)),np.zeros(3)
    with torch.no_grad():
        for begin in range(0,sample.size,batch_size):
            price,ret,_,changes = sample.batch(slice(begin,begin+batch_size))
            end = begin+len(price)
            for i,label in enumerate(LABELS):
                trade = getattr(model,label)(price,changes,returns=ret,eager=True).trades.cpu().numpy().astype(np.float64)
                for key,entries in profiles.items():
                    derivative = entries[i]["derivative_weight"][begin:end]
                    exposures[key][i] += np.einsum("n,nti->ti",derivative,trade)
                    derivatives[key][i] += derivative.sum()
            if monitor is not None:
                monitor.record.update(completed=sample.size+end,total=2*sample.size)
    report,retained = {},{}
    q_ret_centered = (q_returns-q_returns.mean(0)).reshape(len(q_returns),-1)
    q_h_centered = q_payoff-q_payoff.mean()
    for key,entries in profiles.items():
        values = np.column_stack([p["variates"] for p in entries])
        vi = values-values.mean(0)
        ti = -q_ret_centered @ (exposures[key]/sample.size).reshape(3,-1).T
        ti -= q_h_centered[:,None]*(np.asarray(SIGNS)*derivatives[key]/sample.size)[None]
        result = dict(statistics={k:dict(**independent_mc(values[:,i],vi[:,i],ti[:,i]),
            cash=entries[i]["cash"],retained_paths=entries[i]["retained_paths"]) for i,k in enumerate(LABELS)},
            long_minus_empty=independent_mc(values[:,1]-values[:,0],vi[:,1]-vi[:,0],ti[:,1]-ti[:,0]),
            short_minus_empty=independent_mc(values[:,2]-values[:,0],vi[:,2]-vi[:,0],ti[:,2]-ti[:,0]),
            bid=independent_mc(payoff_mean+values[:,1]-values[:,0],vi[:,1]-vi[:,0],q_h_centered+ti[:,1]-ti[:,0]),
            ask=independent_mc(payoff_mean+values[:,0]-values[:,2],vi[:,0]-vi[:,2],q_h_centered+ti[:,0]-ti[:,2]))
        report.setdefault(key[0],{})[key[1]] = result
        prefix = "_".join(key)
        retained.update({prefix+"_values":values,prefix+"_validation_influence":vi,prefix+"_training_influence":ti})
    zeros = np.zeros(len(q_payoff))
    def descriptive(values):
        result = independent_mc(values,values,zeros)
        result["mean"] = result.pop("oce")
        return result
    report.update(evaluation_paths=sample.size,training_paths=len(q_payoff),training_payoff_mean=payoff_mean,
        validation_payoff_mean=descriptive(payoff),
        action_sizes={k:dict(mean_sum_abs_trades=descriptive(l1[:,i]),
                            mean_sum_squared_trades=descriptive(l2[:,i]),max_abs_trade=float(maximum[i]))
                      for i,k in enumerate(LABELS)},
        q_centering="Frozen original TRAINING return/payoff means; NEVER subtract validation return means",
        uncertainty="±3 antithetic-pair MC SE: independent validation sampling plus original-training Q-centering variance; "
                    "matched policies retain covariance. Feature/hedge fitting and checkpoint selection excluded.",
        cash_treatment="fixed_training_cash changes no learned parameter; profiled_cash only estimates the OCE of each fixed policy",
        training_or_checkpoint_selection_on_validation=False,transaction_costs=0)
    if payoff_mean!=float(model.payoff_mean) or any(not np.array_equal(mean,getattr(model,k).return_mean.cpu().numpy())
                                                  for mean,k in zip(q_means,LABELS,strict=True)):
        raise RuntimeError("validation altered a frozen centering buffer")
    return report,retained
