"""Plot raw per-epoch training objectives for the three independent policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_triple_history(run: Path, output: Path) -> dict:
    with (run / "triple-losses.csv").open(newline="") as stream:
        rows = [{k:int(v) if k == "epoch" else float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if not rows or [row["epoch"] for row in rows] != list(range(len(rows))):
        raise ValueError("triple history must have contiguous epochs from zero")
    for row in rows:
        if (not all(np.isfinite(v) for v in row.values()) or not np.isclose(row["sum_loss"],
                sum(row[f"{k}_loss"] for k in ("empty","long","short")),rtol=1e-10,atol=1e-12)):
            raise ValueError("invalid triple loss history")
    identity = json.loads((run / "identity.json").read_text())
    selected = min(rows,key=lambda row:row["sum_loss"])
    output.mkdir(parents=True,exist_ok=True)
    if (output / "losses.png").exists():
        raise FileExistsError("choose a new figure output directory")
    objective = identity["objective"]
    policy_kind = identity.get("policy_kind", "observable-only")
    costs_inside = identity.get("cost_treatment") == "inside_utility"
    independent = identity.get("checkpoint_selection") == "independent_policy_best"
    fig,axes = plt.subplots(1,4,figsize=(17,4.8))
    for axis,key,title,color in zip(axes,("empty_loss","long_loss","short_loss","sum_loss"),
            ("Empty portfolio","Long claim","Short claim","Summed objective"),
            ("#c33b43","#864a9e","#d88424","#69564b"),strict=True):
        axis.plot([r["epoch"] for r in rows],[r[key] for r in rows],color=color,linewidth=1,marker=".",markersize=2,
                  label="Full-training epoch-end loss")
        axis.axhline(rows[0][key],color="#303030",linestyle="--",linewidth=.8,label="Initial policy")
        chosen = min(rows,key=lambda row:row[key]) if independent else selected
        axis.scatter([chosen["epoch"]],[chosen[key]],marker="*",s=100,color=color,edgecolors="black",linewidths=.5,
                     zorder=5,label="Minimum training loss" if independent else "Selected common checkpoint")
        # Empty OCE may be negative; empty MSE can be exactly zero. Symlog
        # retains both rather than dropping points or taking log(abs(loss)).
        axis.set_yscale("symlog",linthresh=1e-4)
        ylabel = "Negative net OCE (symlog); lower is better" if costs_inside else "Penalized training loss (symlog); lower is better"
        axis.set(title=title,xlabel="Epoch (0 = initialization)",ylabel=ylabel)
        axis.grid(alpha=.18)
    selection = "independent policy-best selection" if independent else f"selected {selected['epoch']}"
    fig.suptitle(f"GROMM v1 (−4,3), {policy_kind}, one-step/day: {objective.upper()}\n"
        f"Training only; epoch {rows[-1]['epoch']}/{identity['epochs']}; {selection}",fontsize=12)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.78,bottom=.23,left=.05,right=.985,wspace=.35)
    fig.savefig(output / "losses.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    summary = dict(objective=objective,initial=rows[0],latest=rows[-1],selected=selected,validation=False,
        statistic="negative OCE of net terminal P&L; costs inside utility" if costs_inside else
                  "full training population; OCE or MSE plus unchanged trade penalty",
        policy_kind=policy_kind, y_scale="symlog, linear inside +/-1e-4")
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("figure output must be below the external project data root")
    print(json.dumps(plot_triple_history(args.run,args.output),indent=2))


if __name__ == "__main__":
    main()
