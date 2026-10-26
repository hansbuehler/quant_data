r"""Reads report/data/full_study_results.json and writes the LaTeX table
fragments \input by the report, plus a findings summary markdown file.
Run after run_full_study.py and before compiling the report.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "report" / "data"

SELECTION_DATES = {"2005-11-21", "2010-07-07"}


def _fmt_pct(x) -> str:
    return f"{x*100:.0f}\\%" if x is not None else "--"


def _fmt_bps(x) -> str:
    if x is None:
        return "--"
    return f"{x:.1f}"


def main() -> None:
    with open(DATA / "full_study_results.json") as f:
        results = json.load(f)

    # --- main comparison table (SPX) ---
    spx = [r for r in results if r.get("ticker") == "SPX" and "models" in r]
    rows = []
    for r in spx:
        d = r["date"]
        m = r["models"]
        is_selection = d in SELECTION_DATES
        row = [d + (" (sel.)" if is_selection else "")]
        for name in ("dy2", "sanos_snapshot", "dlv_black", "dlv_bates"):
            entry = m.get(name, {})
            if "error" in entry:
                row.append("solver failed")
            else:
                row.append(f"{_fmt_pct(entry.get('pcnt_inside_bid_ask'))} / {_fmt_bps(entry.get('median_abs_iv_error_bps'))}bp")
        rows.append(row)

    with open(DATA / "main_comparison_table.tex", "w") as f:
        f.write("\\begin{tabular}{lllll}\n\\toprule\n")
        f.write("Date & dy2 & SANOS snap. & DLV$\\times$Black & DLV$\\times$Bates \\\\\n")
        f.write("& \\multicolumn{4}{c}{\\% inside bid-ask / median abs IV err (bps)} \\\\\n\\midrule\n")
        for i, row in enumerate(rows):
            if i == len(SELECTION_DATES):
                f.write("\\midrule\n")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # --- AAPL table (NOT RUN, documented) ---
    aapl = [r for r in results if r.get("ticker") == "AAPL"]
    with open(DATA / "aapl_comparison_table.tex", "w") as f:
        f.write("\\begin{tabular}{ll}\n\\toprule\nDate & Status \\\\\n\\midrule\n")
        for r in aapl:
            status = "NOT RUN -- see note below" if "load_error" in r else "OK"
            f.write(f"{r['date']} & {status} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\n\\textbf{NOT RUN, exact reason}: every AAPL date failed with the identical error "
                 "\\texttt{`For live access to WIDB, set the environment variable WRDS\\_USERNAME: "
                 "currently not set.'}, raised inside \\texttt{enrich\\_day} (called from "
                 "\\texttt{mk\\_arb\\_free\\_day} $\\to$ \\texttt{filter\\_options\\_day} $\\to$ "
                 "\\texttt{smooth\\_interpolated\\_market}) -- a single-name dividend/borrow "
                 "input is not present in the local fixture cache for this filter configuration "
                 "(unlike SPX, an index with no dividend-forecast dependency), and its computation "
                 "requires live WRDS access. \\textbf{Exact fix}: a WRDS account with the Option "
                 "Metrics entitlement, \\texttt{WRDS\\_USERNAME} set, and a \\texttt{\\textasciitilde/.pgpass} "
                 "file (per \\texttt{README.md}'s WRDS setup instructions) -- identical requirement "
                 "already recorded in \\texttt{DATA\\_STATUS.md} from the prior task.")

    # --- reserve sweep table ---
    reserve_rows = []
    for r in spx:
        if "reserve_sweep" not in r:
            continue
        d = r["date"]
        baseline_err = r["models"]["dlv_bates"]["mean_fit_err"]
        for budget, entry in sorted(r["reserve_sweep"]["budgets"].items(), key=lambda kv: float(kv[0])):
            if float(budget) == 0.0:
                continue
            shift = entry["price_shift_before_recalibration"]
            before = entry["fit_before_recalibration"]
            after = entry.get("fit_after_recalibration")
            recal_err = entry.get("recalibration_error")
            theta_int = entry.get("admissible_theta_interval")
            surf = entry.get("surface_change_after_admissible_reset")
            reserve_rows.append([
                d, budget,
                f"{shift['mean_relative_shift_pct']:.0f}\\%",
                f"{_fmt_pct(before['pcnt_inside_bid_ask'])} / {before['mean_fit_err']:.2f}",
                ("FAILED" if recal_err else f"{_fmt_pct(after['pcnt_inside_bid_ask'])} / {after['mean_fit_err']:.2f}"),
                f"[{theta_int['lo']:.3f},{theta_int['hi']:.3f}]" if theta_int else "--",
                (f"{surf['upward']['mean_abs_price_change']:.4f} / {surf['downward']['mean_abs_price_change']:.4f}"
                 if surf else "--"),
            ])

    with open(DATA / "reserve_sweep_table.tex", "w") as f:
        f.write("\\begin{tabular}{llllllll}\n\\toprule\n")
        f.write("Date & $r_1$ & baseline (r=0) & shift & before recalib. & after recalib. & "
                 "adm. $\\Delta\\theta$ & post-reset $|\\Delta$price$|$ (up/down) \\\\\n")
        f.write("& & \\%inside/err & (mean rel.) & \\%inside/err & \\%inside/err & interval & mean abs \\\\\n\\midrule\n")
        prev_date = None
        for d, budget, shift, before, after, theta, surf in reserve_rows:
            baseline_str = ""
            if d != prev_date:
                baseline_entry = next(r for r in spx if r["date"] == d)["models"]["dlv_bates"]
                baseline_str = f"{_fmt_pct(baseline_entry['pcnt_inside_bid_ask'])}/{baseline_entry['mean_fit_err']:.2f}"
                prev_date = d
            f.write(f"{d} & {budget} & {baseline_str} & {shift} & {before} & {after} & {theta} & {surf} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("Wrote table fragments to", DATA)


if __name__ == "__main__":
    main()
