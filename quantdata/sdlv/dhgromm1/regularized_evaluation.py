"""Raw OCE, regularized objectives and action-size diagnostics for the new phase."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from quantdata.sdlv.dhgromm1.floating import LABELS, SIGNS, read_floating_sample
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.training import _economic_utility

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def oce_statistics(values: np.ndarray, influence: np.ndarray) -> dict:
    """Three MC SEs from independent antithetic pair averages, ddof=1."""
    if (values.ndim != 1 or influence.shape != values.shape or len(values) < 4 or len(values) % 2
            or not np.isfinite(values).all() or not np.isfinite(influence).all()):
        raise ValueError("expected finite complete antithetic pairs")
    pairs = influence.reshape(-1, 2).mean(1)
    return dict(oce=float(values.mean()), three_standard_errors=float(3*pairs.std(ddof=1)/np.sqrt(len(pairs))),
                paths=len(values), independent_pairs=len(pairs))


def evaluate_regularized(*, model, sample, arrays: tuple[np.ndarray, ...], source_identity: dict,
                         regularizer: ActionRegularizerConfig, batch_size: int,
                         monitor=None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate fixed policies/scaling; include estimated Q-control centering error.

    Initial and final raw utilities/influence are returned for a paired
    before/after comparison. All inference is on the original training paths;
    there is no refit, new simulation, validation selection, or cost deduction.
    """
    _, returns, payoff = arrays
    scenario = json.loads((Path(source_identity["scenarios"])/"identity.json").read_text())
    measure = scenario.get("measure", "Q")
    q_returns, q_payoff = returns, payoff
    if measure == "P":
        reference = Path(source_identity["centering_reference"])
        q_identity = json.loads((reference.parent/"identity.json").read_text())
        q_scenarios = Path(q_identity["scenarios"])
        q_scenario = json.loads((q_scenarios/"identity.json").read_text())
        for key in ("seed", "chunk_size", "batch_size", "paths_per_sample", "substeps", "calibration_sha256"):
            if scenario[key] != q_scenario[key]:
                raise ValueError(f"cannot preserve Q/P paired-centering covariance: {key} differs")
        q_prices, q_returns, q_payoff = read_floating_sample(q_scenarios/"train", paths=sample.size)
        del q_prices
    values, penalties = np.empty((sample.size, 3)), np.empty((sample.size, 3))
    exposure, derivative_sum = np.zeros((3, 20, 21)), np.zeros(3)
    sizes, maximum = np.zeros((3, 2)), np.zeros(3)
    functions = model.policy_functions(compile_model=True)
    with torch.no_grad():
        for begin in range(0, sample.size, batch_size):
            p, r, h, m = sample.batch(slice(begin, begin+batch_size))
            end = begin+len(p)
            for i, (label, sign, function) in enumerate(zip(LABELS, SIGNS, functions, strict=True)):
                out = function(p, m, returns=r)
                if torch.count_nonzero(out.penalty).item():
                    raise ValueError("a transaction cost entered the zero-cost experiment")
                gain = out.gain.detach().cpu().numpy().astype(np.float64)
                trades = out.trades.detach().cpu().numpy().astype(np.float64)
                cash = float(getattr(model, label).translation.detach())
                net = gain+sign*(payoff[begin:end]-float(model.payoff_mean))
                values[begin:end, i] = _economic_utility(net, cash, 100.)
                z = net-cash
                derivative = np.exp(-100*np.maximum(z, 0))+100*np.maximum(-z, 0)
                exposure[i] += np.einsum("n,nti->ti", derivative, trades)
                derivative_sum[i] += derivative.sum()
                l1 = np.abs(trades).sum((1, 2))
                l2 = np.square(trades).sum((1, 2))
                penalties[begin:end, i] = regularizer.l1*l1+regularizer.l2*l2
                sizes[i] += np.array((l1.sum(), l2.sum()))
                maximum[i] = max(maximum[i], float(np.abs(trades).max()))
            if monitor is not None:
                monitor.record.update(completed=end, total=sample.size)
    influence = values-values.mean(0)
    influence -= (q_returns-q_returns.mean(0)).reshape(sample.size, -1) @ (exposure/sample.size).reshape(3, -1).T
    influence -= (q_payoff-q_payoff.mean())[:, None]*(np.asarray(SIGNS)*derivative_sum/sample.size)[None]
    objective = values-penalties
    objective_influence = influence-(penalties-penalties.mean(0))
    report = dict(measure=measure, transaction_costs=0, regularizer_treatment="outside_utility",
        statistics={label:dict(raw_oce=oce_statistics(values[:,i], influence[:,i]),
            regularized_objective=oce_statistics(objective[:,i], objective_influence[:,i]),
            mean_regularizer=float(penalties[:,i].mean()), mean_sum_abs_trades=float(sizes[i,0]/sample.size),
            rms_trade=float(np.sqrt(sizes[i,1]/(sample.size*420))), max_abs_trade=float(maximum[i]))
            for i,label in enumerate(LABELS)},
        raw_long_minus_empty=oce_statistics(values[:,1]-values[:,0], influence[:,1]-influence[:,0]),
        raw_short_minus_empty=oce_statistics(values[:,2]-values[:,0], influence[:,2]-influence[:,0]),
        uncertainty="±3 antithetic-pair MC SE; includes Q gain/payoff-centering influence and paired Q/P covariance; "
                    "fixed selected policies/quote scaling; excludes fitting/selection uncertainty",
        interpretation="Training only. Raw OCE is not the penalized objective. No Q no-arbitrage claim.")
    return report, values, influence


def plot_regularized_history(output: Path) -> Path:
    """Plot raw OCE, optimized OCE-minus-penalty, and action growth separately."""
    with (output/"triple-losses.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epoch = np.array([int(r["epoch"]) for r in rows])
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    colors = ("#c33b43", "#864a9e", "#d88424")
    for i, (label, color) in enumerate(zip(LABELS, colors, strict=True)):
        raw = np.array([float(r[f"{label}_raw_oce"]) for r in rows])
        objective = -np.array([float(r[f"{label}_loss"]) for r in rows])
        l1 = np.array([float(r[f"{label}_mean_sum_abs_trades"]) for r in rows])
        rms = np.sqrt(np.array([float(r[f"{label}_mean_sum_squared_trades"]) for r in rows])/420)
        for j, (value, name) in enumerate(((raw, "Raw OCE"), (objective, "OCE minus regularizer"))):
            axes[i,j].plot(epoch, value, color=color)
            axes[i,j].set(title=f"{label}: {name}", ylabel="Higher is better")
        axes[i,2].plot(epoch, l1, color=color, label="Mean sum |trade|")
        axes[i,2].plot(epoch, rms, color=color, linestyle="--", label="RMS trade")
        axes[i,2].set(title=f"{label}: executed action sizes", yscale="symlog")
        axes[i,2].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Regularized phase epoch (0 = learned warm start)")
        axis.grid(alpha=.2)
    fig.suptitle(f"{output.name}: zero costs, L1/L2 OUTSIDE utility", fontsize=11)
    fig.text(.5, .01, "Historical curves are training means, not MC intervals; evaluated checkpoints carry ±3 MC SE in report.json.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .97))
    target = output/"regularized-history.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target
